# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Authors:
#  - Burkhard Ringlein <ngl@zurich.ibm.com>
#  - Jan van Lunteren <jvl@zurich.ibm.com>
#  - Chih-Chieh Yang <chih.chieh.yang@ibm.com>
#  - Thomas Parnell <tpa@zurich.ibm.com>

from contextlib import nullcontext
from typing import List, Optional, Tuple

import torch
import triton
import triton.language as tl

from utils.req_meta_flag_codec import REQ_META_SINK_SHIFT, validate_sink_tokens

REQ_META_SINK_SHIFT_CONST = tl.constexpr(REQ_META_SINK_SHIFT)










_NULL_CTX = nullcontext()

def _range(name: str):
    return _NULL_CTX

# -----------------------------------------------------------------------------
# Fast-path meta packing (decode, no logits): fuse many tiny torch ops into one
# triton launch to reduce per-layer fixed overhead.
# -----------------------------------------------------------------------------

@triton.jit
def _kernel_pack_req_meta_decode_fast(
    seqused_k_ptr,
    is_compact_ptr,
    compact_kv_len_ptr,
    compact_offset_tokens_ptr,
    req_meta_i32_ptr,
    req_meta_i64_ptr,
    meta_i32_stride0: tl.constexpr,
    meta_i64_stride0: tl.constexpr,
    block_size: tl.constexpr,
    recent_cap: tl.constexpr,
    sink_tokens,
) -> None:
    pid = tl.program_id(0)

    # Load inputs (cast to desired types to avoid relying on upstream dtypes).
    kv_len_visible = tl.load(seqused_k_ptr + pid).to(tl.int32)
    kv_len_visible = tl.maximum(kv_len_visible, 0)

    is_compact = tl.load(is_compact_ptr + pid).to(tl.int32)
    # Keep is_compact strictly {0,1} to match upstream flags semantics.
    is_compact = tl.where(is_compact != 0, 1, 0).to(tl.int32)

    compact_kv_len = tl.load(compact_kv_len_ptr + pid).to(tl.int32)
    compact_kv_len = tl.maximum(compact_kv_len, 0)

    off_tokens = tl.load(compact_offset_tokens_ptr + pid).to(tl.int64)
    off_tokens = tl.maximum(off_tokens, 0)

    # i32 meta fields:
    # [kv_len_visible, compact_block_cnt, logits_last_n, logits_row_offset,
    #  logits_capacity, flags, recent_len]
    compact_block_cnt = (compact_kv_len + (block_size - 1)) // block_size

    # Decode fast-path: store recent_cap and let kernel derive recent_len at runtime.
    RECENT_CAP_FLAG = 4
    # FULL_CONTEXT_FLAG (bit4): is_compact=0 行（short_dense）使用 context_seq_len 作为 recent_len，
    # 让 compact_only kernel 对该行遍历所有 paged blocks。seqused_k 每步实时读取，不会过期。
    # 注意：bit3=8 已被 LOGF_FLAG 占用。
    FULL_CONTEXT_FLAG = 16
    sink_tokens_i32 = tl.maximum(sink_tokens.to(tl.int32), 0)
    sink_bits = sink_tokens_i32 << REQ_META_SINK_SHIFT_CONST
    meta_i32_base = req_meta_i32_ptr + pid * meta_i32_stride0
    tl.store(meta_i32_base + 0, kv_len_visible)
    tl.store(meta_i32_base + 1, compact_block_cnt)
    # logits_last_n: reuse column 2 to carry compact_kv_len for tail masking.
    tl.store(meta_i32_base + 2, compact_kv_len)
    tl.store(meta_i32_base + 3, 0)
    # logits_capacity: unused in decode fast-path; keep aligned with kv_len_visible.
    tl.store(meta_i32_base + 4, kv_len_visible)
    flags = tl.where(is_compact != 0, is_compact | RECENT_CAP_FLAG, FULL_CONTEXT_FLAG)
    flags = flags | sink_bits
    tl.store(meta_i32_base + 5, flags)
    tl.store(meta_i32_base + 6, recent_cap)

    # i64 meta fields:
    # [block_row_base_paged, compact_base_block, logits_base_ptr, token_row_base]
    meta_i64_base = req_meta_i64_ptr + pid * meta_i64_stride0
    tl.store(meta_i64_base + 0, pid)
    tl.store(meta_i64_base + 1, off_tokens // block_size)
    tl.store(meta_i64_base + 2, 0)
    tl.store(meta_i64_base + 3, off_tokens)

def pack_req_meta_decode_fast(
    *,
    seqused_k: torch.Tensor,
    is_compact_i32: torch.Tensor,
    compact_kv_len_i32: torch.Tensor,
    compact_offset_tokens_i64: torch.Tensor,
    req_meta_i32: torch.Tensor,
    req_meta_i64: torch.Tensor,
    block_size: int,
    recent_cap: int,
    sink_tokens: int,
    num_seqs: Optional[int] = None,
) -> None:
    """Pack req_meta for decode fast-path (no logits) in a single triton launch.

    This replaces per-layer sequences of tiny pointwise torch ops which have
    disproportionate overhead in small-batch decode.

    Args:
        num_seqs: Optional. Number of sequences to process. If None, uses
            req_meta_i32.shape[0]. Use this to avoid slicing pre-allocated
            buffers, reducing Python overhead in hot paths.
    """
    if req_meta_i32.numel() == 0 or req_meta_i64.numel() == 0:
        return
    if req_meta_i32.dtype != torch.int32 or req_meta_i64.dtype != torch.int64:
        raise ValueError("req_meta_i32/i64 must use int32/int64 dtypes")
    if req_meta_i32.dim() != 2 or req_meta_i64.dim() != 2:
        raise ValueError("req_meta_i32/i64 must be 2D tensors")
    max_batch = int(req_meta_i32.shape[0])
    # Use num_seqs if provided, otherwise fall back to tensor shape
    batch = int(num_seqs) if num_seqs is not None else max_batch
    if batch <= 0:
        return
    if batch > max_batch:
        raise ValueError(f"num_seqs ({batch}) exceeds buffer size ({max_batch})")
    if batch > int(req_meta_i64.shape[0]):
        raise ValueError("num_seqs exceeds req_meta_i64 buffer size")
    if req_meta_i32.shape[1] < 7 or req_meta_i64.shape[1] < 4:
        raise ValueError("req_meta tensors missing required columns")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    # [F3-PACK-INPUT-LEN 2026-07-11 EXT审计] kernel 按 +pid 读四个输入到
    # batch-1 行，此前单层版对它们零长度校验（layers 版已有 seqused_k 等价
    # 检查）——短 buffer = GPU 越界读。补齐同款合同。
    if int(seqused_k.numel()) < batch:
        raise ValueError("seqused_k length is smaller than num_seqs")
    if int(is_compact_i32.numel()) < batch:
        raise ValueError("is_compact_i32 length is smaller than num_seqs")
    if int(compact_kv_len_i32.numel()) < batch:
        raise ValueError("compact_kv_len_i32 length is smaller than num_seqs")
    if int(compact_offset_tokens_i64.numel()) < batch:
        raise ValueError("compact_offset_tokens_i64 length is smaller than num_seqs")
    validated_sink_tokens = validate_sink_tokens(int(sink_tokens))

    # Strides are in elements (not bytes) for triton pointer arithmetic.
    meta_i32_stride0 = int(req_meta_i32.stride(0))
    meta_i64_stride0 = int(req_meta_i64.stride(0))
    if meta_i32_stride0 < 7 or meta_i64_stride0 < 4:
        raise ValueError("req_meta tensors must have contiguous (or wider) row strides")

    with _range("sparse.pack_req_meta_decode_fast"):
        _kernel_pack_req_meta_decode_fast[(batch,)](
            seqused_k,
            is_compact_i32,
            compact_kv_len_i32,
            compact_offset_tokens_i64,
            req_meta_i32,
            req_meta_i64,
            meta_i32_stride0=meta_i32_stride0,
            meta_i64_stride0=meta_i64_stride0,
            block_size=int(block_size),
            recent_cap=max(0, int(recent_cap)),
            sink_tokens=int(validated_sink_tokens),
            num_warps=1,
            num_stages=1,
        )

@triton.jit
def _kernel_pack_req_meta_decode_fast_layers(
    seqused_k_ptr,
    is_compact_ptr,
    compact_kv_len_ptr,
    compact_offset_tokens_ptr,
    log_f_mask_ptr,
    log_f_q_lens_ptr,
    log_f_stride_head_i32,
    # Optional: patch out_ptr(meta64[2]) for log_f_mask rows using ring capture layout.
    capture_row_by_batch_row_buf0_ptr,
    capture_row_by_batch_row_buf1_ptr,
    scores_base_ptr_buf0_i64,
    scores_base_ptr_buf1_i64,
    scores_stride_chunk_bytes_buf0_i64,
    scores_stride_chunk_bytes_buf1_i64,
    scores_stride_slot_bytes_buf0_i64,
    scores_stride_slot_bytes_buf1_i64,
    buf_id_by_layer_ptr,
    slot_in_chunk_by_layer_ptr,
    layer_logf_enable_ptr,
    req_meta_i32_ptr,
    req_meta_i64_ptr,
    is_compact_stride0: tl.constexpr,
    is_compact_stride1: tl.constexpr,
    compact_kv_len_stride0: tl.constexpr,
    compact_kv_len_stride1: tl.constexpr,
    compact_offset_stride0: tl.constexpr,
    compact_offset_stride1: tl.constexpr,
    meta_i32_stride0: tl.constexpr,
    meta_i32_stride1: tl.constexpr,
    meta_i64_stride0: tl.constexpr,
    meta_i64_stride1: tl.constexpr,
    block_size: tl.constexpr,
    recent_cap: tl.constexpr,
    sink_tokens,
) -> None:
    layer = tl.program_id(0)
    pid = tl.program_id(1)

    kv_len_visible = tl.load(seqused_k_ptr + pid).to(tl.int32)
    kv_len_visible = tl.maximum(kv_len_visible, 0)

    is_compact = tl.load(is_compact_ptr + layer * is_compact_stride0 + pid * is_compact_stride1).to(tl.int32)
    is_compact = tl.where(is_compact != 0, 1, 0).to(tl.int32)

    compact_kv_len = tl.load(compact_kv_len_ptr + layer * compact_kv_len_stride0 + pid * compact_kv_len_stride1).to(tl.int32)
    compact_kv_len = tl.maximum(compact_kv_len, 0)

    # ── compact override: gated-out 层的 refresh 行走 compact ──
    # 不变量: log_f_mask=1 → bootstrap_done → compact_kv_len > 0（跨所有层）
    # compact_kv_len > 0 guard 是防御性编程，使用已有的无条件加载。
    # Triton 不支持链式 and，用嵌套 if 替代。
    _log_f_mask_early = tl.load(log_f_mask_ptr + pid).to(tl.int32)
    _layer_logf_en_early = tl.load(layer_logf_enable_ptr + layer).to(tl.int32)
    if _log_f_mask_early != 0:
        if _layer_logf_en_early == 0:
            if compact_kv_len > 0:
                is_compact = 1

    off_tokens = tl.load(compact_offset_tokens_ptr + layer * compact_offset_stride0 + pid * compact_offset_stride1).to(tl.int64)
    off_tokens = tl.maximum(off_tokens, 0)

    compact_block_cnt = (compact_kv_len + (block_size - 1)) // block_size

    # Decode fast-path: store recent_cap and let kernel derive recent_len at runtime.
    RECENT_CAP_FLAG = 4
    FULL_CONTEXT_FLAG = 16  # bit4: is_compact=0 行使用 context_seq_len 作为 recent_len
    sink_tokens_i32 = tl.maximum(sink_tokens.to(tl.int32), 0)
    sink_bits = sink_tokens_i32 << REQ_META_SINK_SHIFT_CONST
    meta_i32_base = req_meta_i32_ptr + layer * meta_i32_stride0 + pid * meta_i32_stride1
    tl.store(meta_i32_base + 0, kv_len_visible)
    tl.store(meta_i32_base + 1, compact_block_cnt)
    tl.store(meta_i32_base + 2, compact_kv_len)
    tl.store(meta_i32_base + 3, 0)
    tl.store(meta_i32_base + 4, kv_len_visible)
    flags = tl.where(is_compact != 0, is_compact | RECENT_CAP_FLAG, FULL_CONTEXT_FLAG)
    flags = flags | sink_bits
    tl.store(meta_i32_base + 5, flags)
    tl.store(meta_i32_base + 6, recent_cap)

    meta_i64_base = req_meta_i64_ptr + layer * meta_i64_stride0 + pid * meta_i64_stride1
    tl.store(meta_i64_base + 0, pid)
    tl.store(meta_i64_base + 1, off_tokens // block_size)
    tl.store(meta_i64_base + 2, 0)
    tl.store(meta_i64_base + 3, off_tokens)

    # Optional: dense log_f meta override (decode refresh/bootstrap)
    # Contract: when log_f_mask[pid]!=0 AND layer_logf_enable[layer]!=0,
    # force dense and request log_f (last_n==1, logits-only).
    # layer_logf_enable gates per-layer: gated-out layers (layer-group inactive)
    # must NOT get LOGF_FLAG, otherwise compact-only dispatch misinterprets columns.
    LOGF_FLAG = 8
    log_f_mask = tl.load(log_f_mask_ptr + pid).to(tl.int32)
    layer_logf_en = tl.load(layer_logf_enable_ptr + layer).to(tl.int32)
    if log_f_mask * layer_logf_en != 0:
        buf_id = tl.load(buf_id_by_layer_ptr + layer).to(tl.int32)
        slot_in_chunk = tl.load(slot_in_chunk_by_layer_ptr + layer).to(tl.int64)
        q_len = tl.load(log_f_q_lens_ptr + pid).to(tl.int32)
        q_len = tl.maximum(q_len, 1)
        row_offset = tl.maximum(q_len - 1, 0)
        tl.store(meta_i32_base + 1, log_f_stride_head_i32)
        tl.store(meta_i32_base + 2, 1)  # logits_last_n
        tl.store(meta_i32_base + 3, row_offset)
        tl.store(meta_i32_base + 4, kv_len_visible)
        tl.store(meta_i32_base + 5, (RECENT_CAP_FLAG | LOGF_FLAG) | sink_bits)
        tl.store(meta_i32_base + 6, recent_cap)
        # meta64[1]=scratch_ptr (unused for last_n==1), meta64[2]=out_ptr (patched later), meta64[3]=denom_ptr/token_base (unused)
        tl.store(meta_i64_base + 1, 0)
        # Patch out_ptr(meta64[2]) to point to capture_scores[slot_in_chunk, capture_row] base.
        # layout: capture_row_by_batch_row maps batch row -> capture_row (index in slot_list).
        capture_row0 = tl.load(capture_row_by_batch_row_buf0_ptr + pid).to(tl.int64)
        capture_row1 = tl.load(capture_row_by_batch_row_buf1_ptr + pid).to(tl.int64)
        capture_row = tl.where(buf_id == 0, capture_row0, capture_row1).to(tl.int64)
        base0_buf0 = scores_base_ptr_buf0_i64 + slot_in_chunk * scores_stride_chunk_bytes_buf0_i64
        base0_buf1 = scores_base_ptr_buf1_i64 + slot_in_chunk * scores_stride_chunk_bytes_buf1_i64
        base0 = tl.where(buf_id == 0, base0_buf0, base0_buf1).to(tl.int64)
        stride_slot_bytes = tl.where(buf_id == 0, scores_stride_slot_bytes_buf0_i64, scores_stride_slot_bytes_buf1_i64).to(tl.int64)
        out_ptr = base0 + capture_row * stride_slot_bytes
        tl.store(meta_i64_base + 2, out_ptr)
        tl.store(meta_i64_base + 3, 0)

def pack_req_meta_decode_fast_layers(
    *,
    seqused_k: torch.Tensor,
    is_compact_i32: torch.Tensor,
    compact_kv_len_i32: torch.Tensor,
    compact_offset_tokens_i64: torch.Tensor,
    log_f_mask_i32: Optional[torch.Tensor] = None,
    log_f_q_lens_i32: Optional[torch.Tensor] = None,
    log_f_stride_head: int = 0,
    capture_row_by_batch_row_buf0_i32: Optional[torch.Tensor] = None,
    capture_row_by_batch_row_buf1_i32: Optional[torch.Tensor] = None,
    scores_base_ptr_buf0: int = 0,
    scores_base_ptr_buf1: int = 0,
    scores_stride_chunk_bytes_buf0: int = 0,
    scores_stride_chunk_bytes_buf1: int = 0,
    scores_stride_slot_bytes_buf0: int = 0,
    scores_stride_slot_bytes_buf1: int = 0,
    buf_id_by_layer_i32: Optional[torch.Tensor] = None,
    slot_in_chunk_by_layer_i32: Optional[torch.Tensor] = None,
    layer_logf_enable_i32: Optional[torch.Tensor] = None,
    req_meta_i32: torch.Tensor,
    req_meta_i64: torch.Tensor,
    block_size: int,
    recent_cap: int,
    sink_tokens: int,
    num_layers: Optional[int] = None,
    num_seqs: Optional[int] = None,
) -> None:
    """Pack req_meta for all layers in one launch.

    Shapes:
        is_compact_i32: [num_layers, max_batch]
        compact_kv_len_i32: [num_layers, max_batch]
        compact_offset_tokens_i64: [num_layers, max_batch]
        req_meta_i32: [num_layers, max_batch, 7]
        req_meta_i64: [num_layers, max_batch, 4]
    """
    if req_meta_i32.numel() == 0 or req_meta_i64.numel() == 0:
        return
    if req_meta_i32.dtype != torch.int32 or req_meta_i64.dtype != torch.int64:
        raise ValueError("req_meta_i32/i64 must use int32/int64 dtypes")
    if req_meta_i32.dim() != 3 or req_meta_i64.dim() != 3:
        raise ValueError("req_meta_i32/i64 must be 3D tensors")
    if is_compact_i32.dim() != 2 or compact_kv_len_i32.dim() != 2 or compact_offset_tokens_i64.dim() != 2:
        raise ValueError("compact meta tensors must be 2D [layers, batch]")

    max_layers = int(req_meta_i32.shape[0])
    max_batch = int(req_meta_i32.shape[1])
    layers = int(num_layers) if num_layers is not None else max_layers
    batch = int(num_seqs) if num_seqs is not None else max_batch
    if layers <= 0 or batch <= 0:
        return
    if layers > max_layers or batch > max_batch:
        raise ValueError("num_layers/num_seqs exceeds buffer size")
    if req_meta_i32.shape[2] < 7 or req_meta_i64.shape[2] < 4:
        raise ValueError("req_meta tensors missing required columns")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if seqused_k.numel() < batch:
        raise ValueError("seqused_k length is smaller than num_seqs")
    # [F3-PACK-COMPACT-SHAPE 2026-07-11 EXT审计] kernel 以 layer*stride0 +
    # pid*stride1 索引到 (layers-1, batch-1)；此前对 compact 三输入只查 dim
    # ——过小的 [layers, batch] 面 = GPU 越界读。补 shape 合同。
    for _name, _t in (
        ("is_compact_i32", is_compact_i32),
        ("compact_kv_len_i32", compact_kv_len_i32),
        ("compact_offset_tokens_i64", compact_offset_tokens_i64),
    ):
        if int(_t.shape[0]) < layers or int(_t.shape[1]) < batch:
            raise ValueError(
                f"{_name} shape {tuple(_t.shape)} smaller than required "
                f"[num_layers={layers}, num_seqs={batch}]"
            )
    validated_sink_tokens = validate_sink_tokens(int(sink_tokens))

    if log_f_mask_i32 is None:
        log_f_mask_i32 = torch.zeros((batch,), device=seqused_k.device, dtype=torch.int32)
    if log_f_q_lens_i32 is None:
        log_f_q_lens_i32 = torch.ones((batch,), device=seqused_k.device, dtype=torch.int32)
    if log_f_mask_i32.dtype != torch.int32 or log_f_q_lens_i32.dtype != torch.int32:
        raise ValueError("log_f_mask_i32/log_f_q_lens_i32 must be int32 tensors")
    if log_f_mask_i32.numel() < batch or log_f_q_lens_i32.numel() < batch:
        raise ValueError("log_f_mask_i32/log_f_q_lens_i32 length is smaller than num_seqs")

    if capture_row_by_batch_row_buf0_i32 is None:
        capture_row_by_batch_row_buf0_i32 = torch.zeros((batch,), device=seqused_k.device, dtype=torch.int32)
    if capture_row_by_batch_row_buf1_i32 is None:
        capture_row_by_batch_row_buf1_i32 = torch.zeros((batch,), device=seqused_k.device, dtype=torch.int32)
    if capture_row_by_batch_row_buf0_i32.dtype != torch.int32 or capture_row_by_batch_row_buf1_i32.dtype != torch.int32:
        raise ValueError("capture_row_by_batch_row_buf*_i32 must be int32 tensors")
    if capture_row_by_batch_row_buf0_i32.numel() < batch or capture_row_by_batch_row_buf1_i32.numel() < batch:
        raise ValueError("capture_row_by_batch_row_buf*_i32 length is smaller than num_seqs")

    if buf_id_by_layer_i32 is None:
        buf_id_by_layer_i32 = torch.zeros((layers,), device=seqused_k.device, dtype=torch.int32)
    if slot_in_chunk_by_layer_i32 is None:
        slot_in_chunk_by_layer_i32 = torch.zeros((layers,), device=seqused_k.device, dtype=torch.int32)
    if buf_id_by_layer_i32.dtype != torch.int32 or slot_in_chunk_by_layer_i32.dtype != torch.int32:
        raise ValueError("buf_id_by_layer_i32/slot_in_chunk_by_layer_i32 must be int32 tensors")
    if buf_id_by_layer_i32.numel() < layers or slot_in_chunk_by_layer_i32.numel() < layers:
        raise ValueError("buf_id_by_layer_i32/slot_in_chunk_by_layer_i32 length is smaller than num_layers")

    if layer_logf_enable_i32 is None:
        layer_logf_enable_i32 = torch.ones((layers,), device=seqused_k.device, dtype=torch.int32)
    if layer_logf_enable_i32.dtype != torch.int32:
        raise ValueError("layer_logf_enable_i32 must be int32 tensor")
    if layer_logf_enable_i32.numel() < layers:
        raise ValueError("layer_logf_enable_i32 length is smaller than num_layers")

    meta_i32_stride0 = int(req_meta_i32.stride(0))
    meta_i32_stride1 = int(req_meta_i32.stride(1))
    meta_i64_stride0 = int(req_meta_i64.stride(0))
    meta_i64_stride1 = int(req_meta_i64.stride(1))
    if meta_i32_stride1 < 7 or meta_i64_stride1 < 4:
        raise ValueError("req_meta tensors must have contiguous (or wider) row strides")

    is_compact_stride0 = int(is_compact_i32.stride(0))
    is_compact_stride1 = int(is_compact_i32.stride(1))
    compact_kv_len_stride0 = int(compact_kv_len_i32.stride(0))
    compact_kv_len_stride1 = int(compact_kv_len_i32.stride(1))
    compact_offset_stride0 = int(compact_offset_tokens_i64.stride(0))
    compact_offset_stride1 = int(compact_offset_tokens_i64.stride(1))

    with _range("sparse.pack_req_meta_decode_fast_layers"):
        _kernel_pack_req_meta_decode_fast_layers[(layers, batch)](
            seqused_k,
            is_compact_i32,
            compact_kv_len_i32,
            compact_offset_tokens_i64,
            log_f_mask_i32,
            log_f_q_lens_i32,
            int(log_f_stride_head),
            capture_row_by_batch_row_buf0_i32,
            capture_row_by_batch_row_buf1_i32,
            int(scores_base_ptr_buf0),
            int(scores_base_ptr_buf1),
            int(scores_stride_chunk_bytes_buf0),
            int(scores_stride_chunk_bytes_buf1),
            int(scores_stride_slot_bytes_buf0),
            int(scores_stride_slot_bytes_buf1),
            buf_id_by_layer_i32,
            slot_in_chunk_by_layer_i32,
            layer_logf_enable_i32,
            req_meta_i32,
            req_meta_i64,
            is_compact_stride0=is_compact_stride0,
            is_compact_stride1=is_compact_stride1,
            compact_kv_len_stride0=compact_kv_len_stride0,
            compact_kv_len_stride1=compact_kv_len_stride1,
            compact_offset_stride0=compact_offset_stride0,
            compact_offset_stride1=compact_offset_stride1,
            meta_i32_stride0=meta_i32_stride0,
            meta_i32_stride1=meta_i32_stride1,
            meta_i64_stride0=meta_i64_stride0,
            meta_i64_stride1=meta_i64_stride1,
            block_size=int(block_size),
            recent_cap=max(0, int(recent_cap)),
            sink_tokens=int(validated_sink_tokens),
            num_warps=1,
            num_stages=1,
        )

# -----------------------------------------------------------------------------
# Prefill fast meta pack (all layers): dense rows + optional log_f capture.
# Goal: remove per-layer Python meta patching for prefill.
# -----------------------------------------------------------------------------

@triton.jit
def _kernel_pack_req_meta_prefill_fast_layers(
    seqused_k_ptr,
    cu_seqlens_q_ptr,
    log_f_last_n_ptr,
    log_f_capacity_ptr,
    log_f_stride_head_i32,
    capture_row_by_batch_row_buf0_ptr,
    capture_row_by_batch_row_buf1_ptr,
    scores_base_ptr_buf0_i64,
    scores_base_ptr_buf1_i64,
    scores_stride_chunk_bytes_buf0_i64,
    scores_stride_chunk_bytes_buf1_i64,
    scores_stride_slot_bytes_buf0_i64,
    scores_stride_slot_bytes_buf1_i64,
    denoms_base_ptr_buf0_i64,
    denoms_base_ptr_buf1_i64,
    denoms_stride_chunk_bytes_buf0_i64,
    denoms_stride_chunk_bytes_buf1_i64,
    denoms_stride_slot_bytes_buf0_i64,
    denoms_stride_slot_bytes_buf1_i64,
    buf_id_by_layer_ptr,
    slot_in_chunk_by_layer_ptr,
    req_meta_i32_ptr,
    req_meta_i64_ptr,
    meta_i32_stride0: tl.constexpr,
    meta_i32_stride1: tl.constexpr,
    meta_i64_stride0: tl.constexpr,
    meta_i64_stride1: tl.constexpr,
    recent_cap: tl.constexpr,
    sink_tokens,
) -> None:
    layer = tl.program_id(0)
    pid = tl.program_id(1)

    kv_len_visible = tl.load(seqused_k_ptr + pid).to(tl.int32)
    kv_len_visible = tl.maximum(kv_len_visible, 0)

    # q_len = cu[pid+1] - cu[pid]
    q0 = tl.load(cu_seqlens_q_ptr + pid).to(tl.int32)
    q1 = tl.load(cu_seqlens_q_ptr + pid + 1).to(tl.int32)
    q_len = tl.maximum(q1 - q0, 1)

    last_n = tl.load(log_f_last_n_ptr + pid).to(tl.int32)
    last_n = tl.maximum(last_n, 0)
    cap = tl.load(log_f_capacity_ptr + pid).to(tl.int32)
    cap = tl.maximum(cap, 0)

    # Default: dense no-log_f
    meta_i32_base = req_meta_i32_ptr + layer * meta_i32_stride0 + pid * meta_i32_stride1
    tl.store(meta_i32_base + 0, kv_len_visible)
    tl.store(meta_i32_base + 1, 0)
    tl.store(meta_i32_base + 2, 0)
    tl.store(meta_i32_base + 3, 0)
    tl.store(meta_i32_base + 4, 0)
    tl.store(meta_i32_base + 5, 0)
    tl.store(meta_i32_base + 6, 0)

    meta_i64_base = req_meta_i64_ptr + layer * meta_i64_stride0 + pid * meta_i64_stride1
    tl.store(meta_i64_base + 0, pid)
    tl.store(meta_i64_base + 1, 0)  # scratch_ptr patched per-layer for last_n>1 (prefill)
    tl.store(meta_i64_base + 2, 0)
    tl.store(meta_i64_base + 3, 0)

    has_log_f = last_n > 0
    if has_log_f:
        # row_offset = max(q_len - last_n, 0)
        row_offset = tl.maximum(q_len - last_n, 0)
        cap_eff = tl.minimum(kv_len_visible, cap)
        cap_eff = tl.maximum(cap_eff, 0)

        # Flags: request log_f and enable recent_cap semantics (dense)
        RECENT_CAP_FLAG = 4
        LOGF_FLAG = 8
        sink_tokens_i32 = tl.maximum(sink_tokens.to(tl.int32), 0)
        sink_bits = sink_tokens_i32 << REQ_META_SINK_SHIFT_CONST
        tl.store(meta_i32_base + 1, log_f_stride_head_i32)
        tl.store(meta_i32_base + 2, last_n)
        tl.store(meta_i32_base + 3, row_offset)
        tl.store(meta_i32_base + 4, cap_eff)
        tl.store(meta_i32_base + 5, (RECENT_CAP_FLAG | LOGF_FLAG) | sink_bits)
        tl.store(meta_i32_base + 6, recent_cap)

        buf_id = tl.load(buf_id_by_layer_ptr + layer).to(tl.int32)
        slot_in_chunk = tl.load(slot_in_chunk_by_layer_ptr + layer).to(tl.int64)
        capture_row0 = tl.load(capture_row_by_batch_row_buf0_ptr + pid).to(tl.int64)
        capture_row1 = tl.load(capture_row_by_batch_row_buf1_ptr + pid).to(tl.int64)
        capture_row = tl.where(buf_id == 0, capture_row0, capture_row1).to(tl.int64)

        # out_ptr for scores (fp16/fp32): capture_scores[slot_in_chunk, capture_row]
        base0_scores_buf0 = scores_base_ptr_buf0_i64 + slot_in_chunk * scores_stride_chunk_bytes_buf0_i64
        base0_scores_buf1 = scores_base_ptr_buf1_i64 + slot_in_chunk * scores_stride_chunk_bytes_buf1_i64
        base0_scores = tl.where(buf_id == 0, base0_scores_buf0, base0_scores_buf1).to(tl.int64)
        stride_slot_scores = tl.where(buf_id == 0, scores_stride_slot_bytes_buf0_i64, scores_stride_slot_bytes_buf1_i64).to(tl.int64)
        out_ptr = base0_scores + capture_row * stride_slot_scores
        tl.store(meta_i64_base + 2, out_ptr)

        # denom_ptr only meaningful for last_n>1
        if last_n > 1:
            base0_den_buf0 = denoms_base_ptr_buf0_i64 + slot_in_chunk * denoms_stride_chunk_bytes_buf0_i64
            base0_den_buf1 = denoms_base_ptr_buf1_i64 + slot_in_chunk * denoms_stride_chunk_bytes_buf1_i64
            base0_den = tl.where(buf_id == 0, base0_den_buf0, base0_den_buf1).to(tl.int64)
            stride_slot_den = tl.where(buf_id == 0, denoms_stride_slot_bytes_buf0_i64, denoms_stride_slot_bytes_buf1_i64).to(tl.int64)
            denom_ptr = base0_den + capture_row * stride_slot_den
            tl.store(meta_i64_base + 3, denom_ptr)

def pack_req_meta_prefill_fast_layers(
    *,
    seqused_k: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    log_f_last_n_i32: torch.Tensor,
    log_f_capacity_i32: torch.Tensor,
    log_f_stride_head: int,
    capture_row_by_batch_row_buf0_i32: Optional[torch.Tensor] = None,
    capture_row_by_batch_row_buf1_i32: Optional[torch.Tensor] = None,
    scores_base_ptr_buf0: int = 0,
    scores_base_ptr_buf1: int = 0,
    scores_stride_chunk_bytes_buf0: int = 0,
    scores_stride_chunk_bytes_buf1: int = 0,
    scores_stride_slot_bytes_buf0: int = 0,
    scores_stride_slot_bytes_buf1: int = 0,
    denoms_base_ptr_buf0: int = 0,
    denoms_base_ptr_buf1: int = 0,
    denoms_stride_chunk_bytes_buf0: int = 0,
    denoms_stride_chunk_bytes_buf1: int = 0,
    denoms_stride_slot_bytes_buf0: int = 0,
    denoms_stride_slot_bytes_buf1: int = 0,
    buf_id_by_layer_i32: torch.Tensor = None,  # type: ignore[assignment]
    slot_in_chunk_by_layer_i32: torch.Tensor = None,  # type: ignore[assignment]
    req_meta_i32: torch.Tensor,
    req_meta_i64: torch.Tensor,
    recent_cap: int,
    sink_tokens: int,
    num_layers: Optional[int] = None,
    num_seqs: Optional[int] = None,
) -> None:
    """Pack prefill req_meta for all layers in one launch (dense rows, optional log_f capture).

    Notes:
        - meta64[1] (scratch_ptr) remains 0 here; prefill last_n>1 scratch is allocated per-layer
          and should be patched by the caller if needed.
        - out_ptr(meta64[2]) and denom_ptr(meta64[3]) are derived from ring capture layout.

    Shapes:
        req_meta_i32: [num_layers, max_batch, 7]
        req_meta_i64: [num_layers, max_batch, 4]
    """
    if req_meta_i32.numel() == 0 or req_meta_i64.numel() == 0:
        return
    if req_meta_i32.dtype != torch.int32 or req_meta_i64.dtype != torch.int64:
        raise ValueError("req_meta_i32/i64 must use int32/int64 dtypes")
    if req_meta_i32.dim() != 3 or req_meta_i64.dim() != 3:
        raise ValueError("req_meta_i32/i64 must be 3D tensors")

    max_layers = int(req_meta_i32.shape[0])
    max_batch = int(req_meta_i32.shape[1])
    layers = int(num_layers) if num_layers is not None else max_layers
    batch = int(num_seqs) if num_seqs is not None else max_batch
    if layers <= 0 or batch <= 0:
        return
    if layers > max_layers or batch > max_batch:
        raise ValueError("num_layers/num_seqs exceeds buffer size")
    if req_meta_i32.shape[2] < 7 or req_meta_i64.shape[2] < 4:
        raise ValueError("req_meta tensors missing required columns")
    if seqused_k.numel() < batch:
        raise ValueError("seqused_k length is smaller than num_seqs")
    if cu_seqlens_q.numel() < (batch + 1):
        raise ValueError("cu_seqlens_q length is smaller than num_seqs+1")

    if log_f_last_n_i32.dtype != torch.int32 or log_f_capacity_i32.dtype != torch.int32:
        raise ValueError("log_f_last_n_i32/log_f_capacity_i32 must be int32 tensors")
    if log_f_last_n_i32.numel() < batch or log_f_capacity_i32.numel() < batch:
        raise ValueError("log_f_last_n_i32/log_f_capacity_i32 length is smaller than num_seqs")

    if capture_row_by_batch_row_buf0_i32 is None:
        capture_row_by_batch_row_buf0_i32 = torch.zeros((batch,), device=seqused_k.device, dtype=torch.int32)
    if capture_row_by_batch_row_buf1_i32 is None:
        capture_row_by_batch_row_buf1_i32 = torch.zeros((batch,), device=seqused_k.device, dtype=torch.int32)
    if capture_row_by_batch_row_buf0_i32.dtype != torch.int32 or capture_row_by_batch_row_buf1_i32.dtype != torch.int32:
        raise ValueError("capture_row_by_batch_row_buf*_i32 must be int32 tensors")
    if capture_row_by_batch_row_buf0_i32.numel() < batch or capture_row_by_batch_row_buf1_i32.numel() < batch:
        raise ValueError("capture_row_by_batch_row_buf*_i32 length is smaller than num_seqs")

    if buf_id_by_layer_i32 is None or slot_in_chunk_by_layer_i32 is None:
        raise ValueError("buf_id_by_layer_i32/slot_in_chunk_by_layer_i32 must be provided")
    if buf_id_by_layer_i32.dtype != torch.int32 or slot_in_chunk_by_layer_i32.dtype != torch.int32:
        raise ValueError("buf_id_by_layer_i32/slot_in_chunk_by_layer_i32 must be int32 tensors")
    if buf_id_by_layer_i32.numel() < layers or slot_in_chunk_by_layer_i32.numel() < layers:
        raise ValueError("buf_id_by_layer_i32/slot_in_chunk_by_layer_i32 length is smaller than num_layers")
    validated_sink_tokens = validate_sink_tokens(int(sink_tokens))

    meta_i32_stride0 = int(req_meta_i32.stride(0))
    meta_i32_stride1 = int(req_meta_i32.stride(1))
    meta_i64_stride0 = int(req_meta_i64.stride(0))
    meta_i64_stride1 = int(req_meta_i64.stride(1))
    if meta_i32_stride1 < 7 or meta_i64_stride1 < 4:
        raise ValueError("req_meta tensors must have contiguous (or wider) row strides")

    with _range("sparse.pack_req_meta_prefill_fast_layers"):
        _kernel_pack_req_meta_prefill_fast_layers[(layers, batch)](
            seqused_k,
            cu_seqlens_q,
            log_f_last_n_i32,
            log_f_capacity_i32,
            int(log_f_stride_head),
            capture_row_by_batch_row_buf0_i32,
            capture_row_by_batch_row_buf1_i32,
            int(scores_base_ptr_buf0),
            int(scores_base_ptr_buf1),
            int(scores_stride_chunk_bytes_buf0),
            int(scores_stride_chunk_bytes_buf1),
            int(scores_stride_slot_bytes_buf0),
            int(scores_stride_slot_bytes_buf1),
            int(denoms_base_ptr_buf0),
            int(denoms_base_ptr_buf1),
            int(denoms_stride_chunk_bytes_buf0),
            int(denoms_stride_chunk_bytes_buf1),
            int(denoms_stride_slot_bytes_buf0),
            int(denoms_stride_slot_bytes_buf1),
            buf_id_by_layer_i32,
            slot_in_chunk_by_layer_i32,
            req_meta_i32,
            req_meta_i64,
            meta_i32_stride0=meta_i32_stride0,
            meta_i32_stride1=meta_i32_stride1,
            meta_i64_stride0=meta_i64_stride0,
            meta_i64_stride1=meta_i64_stride1,
            recent_cap=max(0, int(recent_cap)),
            sink_tokens=int(validated_sink_tokens),
            num_warps=1,
            num_stages=1,
        )

# -----------------------------------------------------------------------------
# Fused compact gather (refresh/compact rebuild): copy K/V + token positions
# into compact arena in one Triton launch.
# -----------------------------------------------------------------------------










@triton.jit
def kernel_log_f_pre_from_logits_scratch_lastn_gt1(
    req_meta_i32_ptr,  # [num_seqs, 7]
    req_meta_i64_ptr,  # [num_seqs, 4]
    num_seqs: tl.constexpr,
    num_query_heads: tl.constexpr,
    req_meta_i32_stride_row: tl.int64,
    req_meta_i32_stride_col: tl.int64,
    req_meta_i64_stride_row: tl.int64,
    req_meta_i64_stride_col: tl.int64,
    BLOCK_T: tl.constexpr,
    MAX_R: tl.constexpr,
    LOGF_OUT_FP32: tl.constexpr,
    ALPHA: tl.constexpr,
):
    """last_n>1：从 per-layer logits scratch 计算 log_f_pre 与 denom_f。

    约定（仅 dense log_f 场景，use_compact==0）：
    - meta64[1]：scratch_ptr（fp32，布局 [Hq, last_n, K]）
    - meta64[2]：out_ptr（fp16，布局 [Hq, K]），写 log_f_pre
    - meta64[3]：denom_ptr（fp32，布局 [Hq]），写 denom_f=logsumexp(log_f_pre)

    scratch 无效 token 位置需为 -inf（min_val）。
    """
    pid_seq = tl.program_id(0)
    pid_h = tl.program_id(1)
    if pid_seq >= num_seqs or pid_h >= num_query_heads:
        return

    meta32_row_ptr = req_meta_i32_ptr + pid_seq * req_meta_i32_stride_row
    meta64_row_ptr = req_meta_i64_ptr + pid_seq * req_meta_i64_stride_row

    log_f_stride_head_i32 = tl.load(meta32_row_ptr + 1 * req_meta_i32_stride_col).to(tl.int32)
    logits_last_n = tl.load(meta32_row_ptr + 2 * req_meta_i32_stride_col).to(tl.int32)
    logits_capacity = tl.load(meta32_row_ptr + 4 * req_meta_i32_stride_col).to(tl.int32)
    flags = tl.load(meta32_row_ptr + 5 * req_meta_i32_stride_col)

    use_compact = (flags & 1) != 0
    use_log_f = ((flags & 8) != 0) & (use_compact == 0)
    if ((not use_log_f) or (logits_last_n <= 1)) or ((logits_capacity <= 0) or (log_f_stride_head_i32 <= 0)):
        return

    scratch_base_ptr = tl.load(meta64_row_ptr + 1 * req_meta_i64_stride_col).to(tl.int64)
    out_base_ptr = tl.load(meta64_row_ptr + 2 * req_meta_i64_stride_col).to(tl.int64)
    denom_base_ptr = tl.load(meta64_row_ptr + 3 * req_meta_i64_stride_col).to(tl.int64)
    if (scratch_base_ptr == 0) or ((out_base_ptr == 0) or (denom_base_ptr == 0)):
        return

    min_val = -3.402823466e38
    alpha = tl.full([], ALPHA, tl.float32)
    use_mean = tl.abs(alpha) < 1.0e-6

    scratch_ptr = tl.cast(scratch_base_ptr, tl.pointer_type(tl.float32))
    out_ptr = tl.cast(out_base_ptr, tl.pointer_type(tl.float32 if LOGF_OUT_FP32 else tl.float16))
    denom_ptr = tl.cast(denom_base_ptr, tl.pointer_type(tl.float32))

    cap_i64 = logits_capacity.to(tl.int64)
    stride_pad_i64 = log_f_stride_head_i32.to(tl.int64)
    # [F2-STRIDE-PAD-FAILFAST 2026-07-11 EXT审计] 旧 tl.maximum(stride, cap)
    # 把 0<stride<capacity 的布局违约静默"修复"成按 cap 跨行 = 静默错读
    # (违无 fallback 纪律)。改为仅 <=0 哨兵落 capacity(与 C++ 版
    # out_stride_head_i32>0 的哨兵语义同款);违约域由 wrapper host 校验
    # raise。合法域(0 哨兵 / stride>=cap)数值与旧式逐位一致。
    stride_pad_i64 = tl.where(stride_pad_i64 > 0, stride_pad_i64, cap_i64)
    stride_row = stride_pad_i64
    max_r = tl.minimum(logits_last_n, MAX_R)
    stride_head_scratch = logits_last_n.to(tl.int64) * stride_pad_i64
    stride_head_out = stride_pad_i64

    # --- pass1: row_lse ---
    r_ids = tl.arange(0, MAX_R)
    row_max = tl.full([MAX_R], min_val, tl.float32)
    row_sum = tl.zeros([MAX_R], tl.float32)
    row_has = tl.zeros([MAX_R], tl.int1)

    num_blocks = tl.cdiv(logits_capacity, BLOCK_T)
    for j in range(0, num_blocks):
        offs = j * BLOCK_T + tl.arange(0, BLOCK_T)
        mask_k = offs < logits_capacity
        base = pid_h.to(tl.int64) * stride_head_scratch + offs.to(tl.int64)
        for r in tl.static_range(0, MAX_R):
            r_ok = r < max_r
            r_mask = r_ids == r
            vals = tl.load(scratch_ptr + base + r * stride_row, mask=mask_k & r_ok, other=min_val)
            finite = vals > min_val
            blk_max = tl.max(tl.where(finite, vals, min_val), axis=0)
            row_has = tl.where(r_mask & r_ok, row_has | (blk_max > min_val), row_has)
            row_max = tl.where(r_mask & r_ok, tl.maximum(row_max, blk_max), row_max)

    row_max = tl.where(row_has, row_max, 0.0)
    for j in range(0, num_blocks):
        offs = j * BLOCK_T + tl.arange(0, BLOCK_T)
        mask_k = offs < logits_capacity
        base = pid_h.to(tl.int64) * stride_head_scratch + offs.to(tl.int64)
        for r in tl.static_range(0, MAX_R):
            r_ok = r < max_r
            r_mask = r_ids == r
            vals = tl.load(scratch_ptr + base + r * stride_row, mask=mask_k & r_ok, other=min_val)
            row_has_r = tl.max(tl.where(r_mask, row_has.to(tl.int32), 0), axis=0).to(tl.int1)
            row_max_r = tl.max(tl.where(r_mask, row_max, min_val), axis=0)
            finite = (vals > min_val) & row_has_r & r_ok
            v = tl.where(finite, vals, row_max_r)
            exp = tl.exp(v - row_max_r)
            row_sum = tl.where(
                r_mask & r_ok,
                row_sum + tl.sum(tl.where(finite, exp, 0.0), axis=0),
                row_sum,
            )

    row_lse = row_max + tl.log(row_sum + 1.0e-20)
    row_lse = tl.where(row_has, row_lse, min_val)
    row_count = tl.sum(tl.where((tl.arange(0, MAX_R) < max_r) & row_has, 1, 0), axis=0)
    row_count_f = tl.maximum(tl.where(row_count > 0, row_count.to(tl.float32), 1.0), 1.0)

    # --- pass2: log_f_pre + denom ---
    token_max = tl.full([], min_val, tl.float32)
    token_sum = tl.zeros([], tl.float32)
    for j in range(0, num_blocks):
        offs = j * BLOCK_T + tl.arange(0, BLOCK_T)
        mask_k = offs < logits_capacity
        base = pid_h.to(tl.int64) * stride_head_scratch + offs.to(tl.int64)

        if use_mean:
            acc = tl.zeros([BLOCK_T], tl.float32)
            has_tok = tl.zeros([BLOCK_T], tl.int1)
            for r in tl.static_range(0, MAX_R):
                r_ok = r < max_r
                r_mask = r_ids == r
                row_has_r = tl.max(tl.where(r_mask, row_has.to(tl.int32), 0), axis=0).to(tl.int1)
                row_lse_r = tl.max(tl.where(r_mask, row_lse, min_val), axis=0)
                vals = tl.load(scratch_ptr + base + r * stride_row, mask=mask_k & r_ok, other=min_val)
                valid = (vals > min_val) & row_has_r & r_ok
                acc += tl.where(valid, vals - row_lse_r, 0.0)
                has_tok = has_tok | valid
            log_f_pre = tl.where(has_tok, acc / row_count_f, min_val)
        else:
            amax = tl.full([BLOCK_T], min_val, tl.float32)
            for r in tl.static_range(0, MAX_R):
                r_ok = r < max_r
                r_mask = r_ids == r
                row_has_r = tl.max(tl.where(r_mask, row_has.to(tl.int32), 0), axis=0).to(tl.int1)
                row_lse_r = tl.max(tl.where(r_mask, row_lse, min_val), axis=0)
                vals = tl.load(scratch_ptr + base + r * stride_row, mask=mask_k & r_ok, other=min_val)
                valid = (vals > min_val) & row_has_r & r_ok
                a = tl.where(valid, alpha * (vals - row_lse_r), min_val)
                amax = tl.maximum(amax, a)
            amax = tl.where(amax > min_val, amax, 0.0)

            asum = tl.zeros([BLOCK_T], tl.float32)
            has_tok = tl.zeros([BLOCK_T], tl.int1)
            for r in tl.static_range(0, MAX_R):
                r_ok = r < max_r
                r_mask = r_ids == r
                row_has_r = tl.max(tl.where(r_mask, row_has.to(tl.int32), 0), axis=0).to(tl.int1)
                row_lse_r = tl.max(tl.where(r_mask, row_lse, min_val), axis=0)
                vals = tl.load(scratch_ptr + base + r * stride_row, mask=mask_k & r_ok, other=min_val)
                valid = (vals > min_val) & row_has_r & r_ok
                a = tl.where(valid, alpha * (vals - row_lse_r), min_val)
                asum += tl.where(valid, tl.exp(a - amax), 0.0)
                has_tok = has_tok | valid
            lse = amax + tl.log(asum + 1.0e-20)
            log_f_pre = tl.where(has_tok, (lse - tl.log(row_count_f)) / alpha, min_val)

        out_offsets = pid_h.to(tl.int64) * stride_head_out + offs.to(tl.int64)
        if LOGF_OUT_FP32:
            tl.store(out_ptr + out_offsets, log_f_pre.to(tl.float32), mask=mask_k)
        else:
            tl.store(out_ptr + out_offsets, log_f_pre.to(tl.float16), mask=mask_k)

        blk_max = tl.max(log_f_pre, axis=0)
        m_new = tl.maximum(token_max, blk_max)
        m_new_safe = tl.where(m_new > min_val, m_new, 0.0)
        token_max_safe = tl.where(token_max > min_val, token_max, 0.0)
        token_sum = token_sum * tl.exp(token_max_safe - m_new_safe) + tl.sum(
            tl.where(log_f_pre > min_val, tl.exp(log_f_pre - m_new_safe), 0.0),
            axis=0,
        )
        token_max = m_new

    denom = tl.where(token_max > min_val, token_max + tl.log(token_sum + 1.0e-20), 0.0)
    tl.store(denom_ptr + pid_h.to(tl.int64), denom)

def _launch_log_f_pre_from_logits_scratch_lastn_gt1(
    *,
    req_meta_i32: torch.Tensor,
    req_meta_i64: torch.Tensor,
    num_seqs: int,
    num_query_heads: int,
    log_f_out_fp32: bool,
    alpha: float,
) -> None:
    # [F2-STRIDE-PAD-FAILFAST 2026-07-11 EXT审计] kernel 侧已不再静默"修复"
    # 布局违约(见 kernel 内注)。违约域在此 fail-fast:col1(stride_head)
    # 必须 ==0(哨兵→kernel 落 capacity)或 >= col4(capacity)。本 wrapper
    # 仅测试/oracle 调用面(生产走 C++ reduce_log_f_pre_scratch_cuda),
    # 一次 D2H 小拷贝的校验成本可接受。
    _n = int(num_seqs)
    if _n > 0 and req_meta_i32.numel() > 0:
        _meta_host = req_meta_i32[:_n].detach().cpu()
        _stride_head = _meta_host[:, 1]
        _cap = _meta_host[:, 4]
        _bad = (_stride_head > 0) & (_stride_head < _cap)
        if bool(_bad.any()):
            _row = int(_bad.nonzero()[0].item())
            raise ValueError(
                "log_f_pre lastn_gt1: meta row "
                f"{_row} has 0 < stride_head({int(_stride_head[_row])}) < "
                f"capacity({int(_cap[_row])}) — layout contract violation "
                "(the kernel would mis-stride across scratch rows)"
            )
    grid = (int(num_seqs), int(num_query_heads))
    kernel_log_f_pre_from_logits_scratch_lastn_gt1[grid](
        req_meta_i32,
        req_meta_i64,
        num_seqs=int(num_seqs),
        num_query_heads=int(num_query_heads),
        req_meta_i32_stride_row=req_meta_i32.stride(0),
        req_meta_i32_stride_col=req_meta_i32.stride(1),
        req_meta_i64_stride_row=req_meta_i64.stride(0),
        req_meta_i64_stride_col=req_meta_i64.stride(1),
        BLOCK_T=256,
        MAX_R=16,
        LOGF_OUT_FP32=bool(log_f_out_fp32),
        ALPHA=float(alpha),
        num_warps=4,
        num_stages=2,
    )

