# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Authors:
#  - Burkhard Ringlein <ngl@zurich.ibm.com>
#  - Jan van Lunteren <jvl@zurich.ibm.com>
#  - Chih-Chieh Yang <chih.chieh.yang@ibm.com>
#  - Thomas Parnell <tpa@zurich.ibm.com>

import math
import os
from contextlib import nullcontext
from typing import List, Optional, Tuple

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from triton_kernel.req_meta_flag_codec import REQ_META_SINK_SHIFT, validate_sink_tokens

REQ_META_SINK_SHIFT_CONST = tl.constexpr(REQ_META_SINK_SHIFT)









_LAUNCH_CACHE = {}
_HEAD_SIZE_PADDED_CACHE: dict[int, int] = {}
# 性能路径默认关闭严格校验；如需定位输入问题可开启。
# Debug 断言标志：仅在显式开启时编译包含 device_assert 逻辑。
# 与 vLLM 原生相同的 2D/3D 分支阈值：默认 >128。
# 仅用于测试/benchmark 验证 dispatch 路径，不参与实际计算逻辑。
# 仅用于测试/benchmark 验证 dispatch 路径，不参与实际计算逻辑。
# 环境变量读取策略：perf 热路径默认只在 import 时读取一次；
# 仅在显式开启 VLLM_SPARSE_DYNAMIC_ENV=1 时，才在每次调用时读取（避免引入额外 Python 开销）。
_DYNAMIC_ENV = os.environ.get("VLLM_SPARSE_DYNAMIC_ENV", "0") == "1"
_PREFETCH_COMPACT_CACHED = os.environ.get("VLLM_SPARSE_PREFETCH_COMPACT", "0") == "1"

_NULL_CTX = nullcontext()

def _range(name: str):
    return _NULL_CTX

_FP8_DTYPES = tuple(
    dt for dt in (
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e5m2", None),
        getattr(torch, "float8_e4m3fnuz", None),
        getattr(torch, "float8_e5m2fnuz", None),
    ) if dt is not None
)

def _cached_stride_args(
    device: torch.device,
    *,
    req_meta_i32: torch.Tensor,
    req_meta_i64: torch.Tensor,
    key_paged: torch.Tensor,
    value_paged: torch.Tensor,
    block_tables_paged: torch.Tensor,
    key_compact: torch.Tensor,
    value_compact: torch.Tensor,
    token_positions: torch.Tensor,
) -> tuple[dict, dict]:
    """缓存 triton kernel 的 stride 参数 dict，减少每 step 的 Python 开销。

    返回：
    - stride_args：通用/2D 路径使用（包含 token_pos_stride_row）
    - stride_args_3d_compact：3D compact-only 使用（不包含 token_pos_stride_row）
    """

    # 对齐旧行为：空 tensor 时传 0，避免 stride 访问异常。
    has_compact = int(key_compact.numel() > 0)
    has_pos = int(token_positions.numel() > 0)

    # 注意：不能用 Python 对象 id 作为 cache key（pytest 内部/短生命周期张量可能触发 id 复用），
    # 否则会命中错误的 stride pack 导致数值不一致甚至越界。
    #
    # stride_args 只依赖 strides（与 data_ptr 无关），因此用 strides 作为 key 即可，且在真实 runtime 中同样高命中。
    cache_key = (
        device,
        "stride_args",
        has_compact,
        has_pos,
        int(req_meta_i32.stride(0)),
        int(req_meta_i32.stride(1)),
        int(req_meta_i64.stride(0)),
        int(req_meta_i64.stride(1)),
        int(key_paged.stride(0)),
        int(key_paged.stride(1)),
        int(key_paged.stride(2)),
        int(key_paged.stride(3)),
        int(value_paged.stride(0)),
        int(value_paged.stride(1)),
        int(value_paged.stride(2)),
        int(value_paged.stride(3)),
        int(block_tables_paged.stride(0)),
        int(block_tables_paged.stride(1)),
        int(key_compact.stride(0)) if has_compact else 0,
        int(key_compact.stride(1)) if has_compact else 0,
        int(key_compact.stride(2)) if has_compact else 0,
        int(key_compact.stride(3)) if has_compact else 0,
        int(value_compact.stride(0)) if has_compact else 0,
        int(value_compact.stride(1)) if has_compact else 0,
        int(value_compact.stride(2)) if has_compact else 0,
        int(value_compact.stride(3)) if has_compact else 0,
        int(token_positions.stride(0)) if has_pos else 0,
        int(token_positions.stride(1)) if has_pos else 0,
        int(token_positions.stride(2)) if has_pos else 0,
        int(token_positions.stride(3)) if has_pos else 0,
    )
    cached = _LAUNCH_CACHE.get(cache_key)
    if cached is not None:
        return cached  # type: ignore[return-value]

    stride_args = dict(
        req_meta_i32_stride_row=req_meta_i32.stride(0),
        req_meta_i32_stride_col=req_meta_i32.stride(1),
        req_meta_i64_stride_row=req_meta_i64.stride(0),
        req_meta_i64_stride_col=req_meta_i64.stride(1),
        stride_k_cache_0=key_paged.stride(0),
        stride_k_cache_1=key_paged.stride(1),
        stride_k_cache_2=key_paged.stride(2),
        stride_k_cache_3=key_paged.stride(3),
        stride_v_cache_0=value_paged.stride(0),
        stride_v_cache_1=value_paged.stride(1),
        stride_v_cache_2=value_paged.stride(2),
        stride_v_cache_3=value_paged.stride(3),
        stride_k_compact_0=key_compact.stride(0) if has_compact else 0,
        stride_k_compact_1=key_compact.stride(1) if has_compact else 0,
        stride_k_compact_2=key_compact.stride(2) if has_compact else 0,
        stride_k_compact_3=key_compact.stride(3) if has_compact else 0,
        stride_v_compact_0=value_compact.stride(0) if has_compact else 0,
        stride_v_compact_1=value_compact.stride(1) if has_compact else 0,
        stride_v_compact_2=value_compact.stride(2) if has_compact else 0,
        stride_v_compact_3=value_compact.stride(3) if has_compact else 0,
        token_pos_stride_row=token_positions.stride(0) if has_pos else 0,
        token_pos_stride_head=token_positions.stride(1) if has_pos else 0,
        token_pos_stride_block=token_positions.stride(2) if has_pos else 0,
        token_pos_stride_token=token_positions.stride(3) if has_pos else 0,
        block_tables_paged_stride_row=block_tables_paged.stride(0),
    )
    stride_args_3d_compact = dict(stride_args)
    stride_args_3d_compact.pop("token_pos_stride_row", None)

    _LAUNCH_CACHE[cache_key] = (stride_args, stride_args_3d_compact)
    return stride_args, stride_args_3d_compact

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










def _cached_compact_only_stride_pack(
    device: torch.device,
    *,
    req_meta_i32: torch.Tensor,
    req_meta_i64: torch.Tensor,
    key_paged: torch.Tensor,
    value_paged: torch.Tensor,
    block_tables_paged: torch.Tensor,
    key_compact: torch.Tensor,
    value_compact: torch.Tensor,
    token_positions: torch.Tensor,
    query: torch.Tensor,
    output: torch.Tensor,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, int, int, int]]:
    """为 compact-only 2D/3D 路径缓存“positional args 的 stride pack”。

    目标：在小 grid 场景减少 kwargs 组装/合并成本；runtime stride 用 tuple（按固定顺序），
    tl.constexpr 的 stride_*_3 用 tuple 返回（避免每次 launch 的 **dict 展开开销）。
    """

    has_compact = int(key_compact.numel() > 0)
    has_pos = int(token_positions.numel() > 0)
    query_stride_0 = query.stride(0)
    query_stride_1 = query.stride(1)
    output_stride_0 = output.stride(0)
    output_stride_1 = output.stride(1)
    cache_key = (
        device,
        "compact_only_stride_pack",
        has_compact,
        has_pos,
        int(req_meta_i32.stride(0)),
        int(req_meta_i32.stride(1)),
        int(req_meta_i64.stride(0)),
        int(req_meta_i64.stride(1)),
        int(token_positions.stride(1)) if has_pos else 0,
        int(token_positions.stride(2)) if has_pos else 0,
        int(token_positions.stride(3)) if has_pos else 0,
        int(query_stride_0),
        int(query_stride_1),
        int(output_stride_0),
        int(output_stride_1),
        int(key_paged.stride(0)),
        int(key_paged.stride(1)),
        int(key_paged.stride(2)),
        int(key_paged.stride(3)),
        int(value_paged.stride(0)),
        int(value_paged.stride(1)),
        int(value_paged.stride(2)),
        int(value_paged.stride(3)),
        int(key_compact.stride(0)) if has_compact else 0,
        int(key_compact.stride(1)) if has_compact else 0,
        int(key_compact.stride(2)) if has_compact else 0,
        int(key_compact.stride(3)) if has_compact else 0,
        int(value_compact.stride(0)) if has_compact else 0,
        int(value_compact.stride(1)) if has_compact else 0,
        int(value_compact.stride(2)) if has_compact else 0,
        int(value_compact.stride(3)) if has_compact else 0,
        int(block_tables_paged.stride(0)),
        int(block_tables_paged.stride(1)),
    )
    cached = _LAUNCH_CACHE.get(cache_key)
    if cached is not None:
        return cached  # type: ignore[return-value]

    stride_k_cache_3 = int(key_paged.stride(3))
    stride_v_cache_3 = int(value_paged.stride(3))
    stride_k_compact_3 = int(key_compact.stride(3)) if has_compact else 0
    stride_v_compact_3 = int(value_compact.stride(3)) if has_compact else 0

    meta_stride3 = (stride_k_cache_3, stride_v_cache_3, stride_k_compact_3, stride_v_compact_3)

    # 2D compact-only: runtime stride args（与 kernel_unified_attention_2d_compact_only 的参数顺序一致）
    runtime_2d = (
        int(req_meta_i32.stride(0)),
        int(req_meta_i32.stride(1)),
        int(req_meta_i64.stride(0)),
        int(req_meta_i64.stride(1)),
        int(token_positions.stride(1)) if has_pos else 0,
        int(token_positions.stride(2)) if has_pos else 0,
        int(token_positions.stride(3)) if has_pos else 0,
        int(query_stride_0),
        int(query_stride_1),
        int(output_stride_0),
        int(output_stride_1),
        int(key_paged.stride(0)),
        int(key_paged.stride(1)),
        int(key_paged.stride(2)),
        int(value_paged.stride(0)),
        int(value_paged.stride(1)),
        int(value_paged.stride(2)),
        int(key_compact.stride(0)) if has_compact else 0,
        int(key_compact.stride(1)) if has_compact else 0,
        int(key_compact.stride(2)) if has_compact else 0,
        int(value_compact.stride(0)) if has_compact else 0,
        int(value_compact.stride(1)) if has_compact else 0,
        int(value_compact.stride(2)) if has_compact else 0,
        int(block_tables_paged.stride(0)),
    )

    # 3D compact-only: runtime stride args（与 kernel_unified_attention_3d_compact_only 的参数顺序一致）
    runtime_3d = (
        int(req_meta_i32.stride(0)),
        int(req_meta_i32.stride(1)),
        int(req_meta_i64.stride(0)),
        int(req_meta_i64.stride(1)),
        int(token_positions.stride(1)) if has_pos else 0,
        int(token_positions.stride(2)) if has_pos else 0,
        int(token_positions.stride(3)) if has_pos else 0,
        int(query_stride_0),
        int(query_stride_1),
        int(key_paged.stride(0)),
        int(key_paged.stride(1)),
        int(key_paged.stride(2)),
        int(value_paged.stride(0)),
        int(value_paged.stride(1)),
        int(value_paged.stride(2)),
        int(key_compact.stride(0)) if has_compact else 0,
        int(key_compact.stride(1)) if has_compact else 0,
        int(key_compact.stride(2)) if has_compact else 0,
        int(value_compact.stride(0)) if has_compact else 0,
        int(value_compact.stride(1)) if has_compact else 0,
        int(value_compact.stride(2)) if has_compact else 0,
        int(block_tables_paged.stride(0)),
    )

    _LAUNCH_CACHE[cache_key] = (runtime_2d, runtime_3d, meta_stride3)
    return runtime_2d, runtime_3d, meta_stride3



def _is_fp8_dtype(dtype: torch.dtype) -> bool:
    return dtype in _FP8_DTYPES

def _cached_ones_tensor(device: torch.device, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    key = (device, "ones", shape, dtype)
    cached = _LAUNCH_CACHE.get(key)
    if cached is None:
        cached = torch.ones(shape, device=device, dtype=dtype)
        _LAUNCH_CACHE[key] = cached
    return cached

def _cached_3d_workspace_tensors(
    device: torch.device,
    *,
    num_tokens: int,
    num_query_heads: int,
    num_segments: int,
    head_size_padded: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """复用 3D 分段 softmax 的临时 workspace，减少每步分配并利于 CUDA Graph 捕获。"""

    key_full = (device, "workspace_3d_full", num_query_heads, num_segments, head_size_padded)
    entry = _LAUNCH_CACHE.get(key_full)
    if entry is None:
        cap = int(num_tokens)
        segm_output_full = torch.empty(
            (cap, num_query_heads, num_segments, head_size_padded),
            device=device,
            dtype=torch.float32,
        )
        segm_max_full = torch.empty((cap, num_query_heads, num_segments), device=device, dtype=torch.float32)
        segm_expsum_full = torch.empty((cap, num_query_heads, num_segments), device=device, dtype=torch.float32)
        entry = (segm_output_full, segm_max_full, segm_expsum_full)
        _LAUNCH_CACHE[key_full] = entry

    segm_output_full, segm_max_full, segm_expsum_full = entry
    if segm_output_full.shape[0] < num_tokens:
        cap = int(num_tokens)
        segm_output_full = torch.empty(
            (cap, num_query_heads, num_segments, head_size_padded),
            device=device,
            dtype=torch.float32,
        )
        segm_max_full = torch.empty((cap, num_query_heads, num_segments), device=device, dtype=torch.float32)
        segm_expsum_full = torch.empty((cap, num_query_heads, num_segments), device=device, dtype=torch.float32)
        entry = (segm_output_full, segm_max_full, segm_expsum_full)
        _LAUNCH_CACHE[key_full] = entry

    # 复用切片 view，避免每 step 产生新的 view 对象。
    view_key = (device, "workspace_3d_view", id(segm_output_full), num_tokens)
    view = _LAUNCH_CACHE.get(view_key)
    if view is not None:
        return view  # type: ignore[return-value]

    view = (
        segm_output_full[:num_tokens],
        segm_max_full[:num_tokens],
        segm_expsum_full[:num_tokens],
    )
    _LAUNCH_CACHE[view_key] = view
    return view

from vllm.logger import init_logger

logger = init_logger(__name__)

@triton.jit
def cdiv_fn(x, y):
    return (x + y - 1) // y

@triton.jit
def apply_softcap(S, x):
    return x * libdevice.tanh(S / x)

@triton.jit
def _rt_i32(x):
    """Force constexpr → runtime int32 (Triton 3.1.0 compat)."""
    return (x + tl.zeros([], dtype=tl.int64)).to(tl.int32)

@triton.jit
def find_seq_idx(query_start_len_ptr, target_idx, num_seqs,
                 BLOCK_Q: tl.constexpr, use_q_block_mode: tl.constexpr):
    left = 0  # Triton 3.1 compat: no AnnAssign in kernel body
    right = num_seqs
    while left < right:
        mid = (left + right) // 2
        val = tl.load(query_start_len_ptr + mid)
        mid_val = val // BLOCK_Q + mid if use_q_block_mode else val

        if mid_val <= target_idx:
            left = mid + 1
        else:
            right = mid

    return left - 1

@triton.jit
def maybe_store_logits(
        logits_ptr,
        logits_has_ptr,
        logits_row_offset,
        logits_last_n,
        logits_capacity,
        logits_capacity_i64,
        logits_stride_head,
        logits_stride_token,
        query_pos,
        query_mask_0,
        query_mask_1,
        seq_offset,
        seq_mask,
        query_offset_1,
        scores_tile):
    logits_last_n_i64 = tl.where(logits_last_n > 0,
                                 logits_last_n.to(tl.int64), 1)
    logits_stride_row = logits_stride_head // logits_last_n_i64
    row_idx = query_pos - logits_row_offset
    row_mask = (row_idx >= 0) & (row_idx < logits_last_n)
    valid_rows = row_mask & query_mask_0 & query_mask_1
    token_mask = seq_offset < logits_capacity
    head_offsets = query_offset_1.to(tl.int64) * logits_stride_head
    row_idx_i64 = row_idx.to(tl.int64)
    seq_offset_i64 = seq_offset.to(tl.int64)
    token_linear = row_idx_i64[:, None] * logits_stride_row + \
        seq_offset_i64[None, :]
    logits_offsets = head_offsets[:, None] + \
        token_linear * logits_stride_token
    store_mask = valid_rows[:, None] & seq_mask & token_mask[None, :]
    logits_mask = tl.broadcast_to(logits_has_ptr, store_mask.shape)
    store_mask = store_mask & logits_mask
    tl.store(
        logits_ptr + logits_offsets,
        scores_tile,
        mask=store_mask,
    )

@triton.jit
def store_logits_contiguous_rows(
        logits_ptr,
        logits_has_ptr,
        logits_row_offset,
        logits_last_n,
        logits_row_stride_i64,
        logits_stride_head,
        query_pos,
        query_mask_0,
        query_mask_1,
        seq_offset,
        seq_mask,
        query_offset_1,
        scores_tile):
    """Faster store for contiguous layouts with logits_stride_token==1.

    Expected logits layout:
      offset(h, r, t) = h * logits_stride_head + r * logits_row_stride + t

    Caller must ensure seq_mask already includes token_mask (seq_offset < logits_capacity).
    """
    row_idx = query_pos - logits_row_offset
    row_mask = (row_idx >= 0) & (row_idx < logits_last_n)
    valid_rows = row_mask & query_mask_0 & query_mask_1

    head_offsets = query_offset_1.to(tl.int64) * logits_stride_head
    row_idx_i64 = row_idx.to(tl.int64)
    seq_offset_i64 = seq_offset.to(tl.int64)
    token_linear = row_idx_i64[:, None] * logits_row_stride_i64 + seq_offset_i64[None, :]
    logits_offsets = head_offsets[:, None] + token_linear

    store_mask = valid_rows[:, None] & seq_mask
    store_mask = store_mask & tl.broadcast_to(logits_has_ptr, store_mask.shape)
    tl.store(logits_ptr + logits_offsets, scores_tile, mask=store_mask)

@triton.jit
def store_log_f_logits_lastn1(
        log_f_ptr,
        log_f_has_ptr,
        LOGF_OUT_FP32: tl.constexpr,
        flags,
        logits_row_offset,
        logits_capacity,
        recent_len,
        logits_stride_head,
        logits_stride_token,
        query_pos,
        query_mask_0,
        query_mask_1,
        seq_offset,
        seq_mask,
        query_offset_1,
        scores_tile):
    """last_n==1：把 masked logits(fp16) 写入 log_f buffer，后续独立 kernel 就地 log_softmax 得 log_probs."""
    min_val = -3.402823466e38
    sink_len = tl.maximum((flags.to(tl.int32) >> REQ_META_SINK_SHIFT_CONST), 0)
    recent_len_i32 = tl.maximum(recent_len.to(tl.int32), 0)
    logits_capacity_i32 = logits_capacity.to(tl.int32)
    recent_start = tl.maximum(logits_capacity_i32 - recent_len_i32, 0)

    token_mask = seq_offset < logits_capacity_i32
    sink_mask = seq_offset >= sink_len
    recent_mask = seq_offset < recent_start
    valid_tok = token_mask & sink_mask & recent_mask

    row_idx = query_pos - logits_row_offset
    base_valid_rows = (row_idx >= 0) & (row_idx < 1) & query_mask_0 & query_mask_1
    head_offsets = query_offset_1.to(tl.int64) * logits_stride_head
    seq_offset_i64 = seq_offset.to(tl.int64)

    scores = tl.where(base_valid_rows[:, None] & valid_tok[None, :] & seq_mask, scores_tile, min_val)
    out_ptr = log_f_ptr + head_offsets[:, None] + seq_offset_i64[None, :] * logits_stride_token
    store_mask = log_f_has_ptr & base_valid_rows[:, None] & token_mask[None, :]
    if LOGF_OUT_FP32:
        tl.store(out_ptr, scores.to(tl.float32), mask=store_mask)
    else:
        tl.store(out_ptr, scores.to(tl.float16), mask=store_mask)

@triton.jit
def store_log_f_logits_scratch_lastn_gt1(
        logits_scratch_ptr,
        logits_scratch_has_ptr,
        flags,
        logits_row_offset,
        logits_last_n,
        logits_scratch_stride_row_i64,
        logits_capacity_i32,
        recent_len,
        logits_scratch_stride_head,
        query_pos,
        query_mask_0,
        query_mask_1,
        seq_offset,
        seq_mask,
        query_offset_1,
        scores_tile):
    """last_n>1：把 masked logits 写入 per-layer scratch(fp32 [H,last_n,K])，同时保证无效 token 为 -inf."""
    min_val = -3.402823466e38
    sink_len = tl.maximum((flags.to(tl.int32) >> REQ_META_SINK_SHIFT_CONST), 0)
    recent_len_i32 = tl.maximum(recent_len.to(tl.int32), 0)
    recent_start = tl.maximum(logits_capacity_i32 - recent_len_i32, 0)

    token_mask = seq_offset < logits_capacity_i32
    sink_mask = seq_offset >= sink_len
    recent_mask = seq_offset < recent_start
    valid_tok = token_mask & sink_mask & recent_mask

    row_idx = query_pos - logits_row_offset
    base_valid_rows = (row_idx >= 0) & (row_idx < logits_last_n) & query_mask_0 & query_mask_1
    scores = tl.where(base_valid_rows[:, None] & valid_tok[None, :] & seq_mask, scores_tile, min_val)
    store_logits_contiguous_rows(
        logits_scratch_ptr,
        logits_scratch_has_ptr,
        logits_row_offset,
        logits_last_n,
        logits_scratch_stride_row_i64,
        logits_scratch_stride_head,
        query_pos,
        query_mask_0,
        query_mask_1,
        seq_offset,
        token_mask[None, :],
        query_offset_1,
        scores,
    )

@triton.jit
def kernel_unified_attention_2d(
        output_ptr,  # [num_tokens, num_query_heads, head_size]
        query_ptr,  # [num_tokens, num_query_heads, head_size]
        key_cache_ptr,  # [num_blks, blk_size, num_kv_heads, head_size]
        value_cache_ptr,  # [num_blks, blk_size, num_kv_heads, head_size]
        block_tables_paged_ptr,  # [num_seqs, max_blocks_paged]
        key_compact_ptr,  # [num_sparse_blks, blk_size, num_kv_heads, head_size]
        value_compact_ptr,  # [num_sparse_blks, blk_size, num_kv_heads, head_size]
        token_positions_ptr,  # [num_seqs, num_kv_heads, max_blocks, block_size]
        req_meta_i32_ptr,  # [num_seqs, 8]
        req_meta_i64_ptr,  # [num_seqs, 5]
        seqused_k_ptr,  # [num_seqs]
        logits_dummy_ptr,  # float32*
        query_norms_stride_head: tl.int64,
        query_norms_stride_token: tl.int64,
        query_norms_window: tl.int32,
        alibi_slopes_ptr,  # [num_query_heads]
        scale,  # float32
        k_scale,  # float32
        v_scale,  # float32
        softcap,  # float32
        num_query_heads: tl.constexpr,  # int
        num_queries_per_kv: tl.constexpr,  # int
        req_meta_i32_stride_row: tl.int64,
        req_meta_i32_stride_col: tl.int64,
        req_meta_i64_stride_row: tl.int64,
        req_meta_i64_stride_col: tl.int64,
        token_pos_stride_row: tl.int64,
        token_pos_stride_head: tl.int64,
        token_pos_stride_block: tl.int64,
        token_pos_stride_token: tl.int64,
        query_stride_0: tl.int64,  # int
        query_stride_1: tl.int64,  # int, should be equal to head_size
        output_stride_0: tl.int64,  # int
        output_stride_1: tl.int64,  # int, should be equal to head_size
        block_tables_paged_stride_row: tl.int64,
        BLOCK_SIZE: tl.constexpr,  # int
        HEAD_SIZE: tl.constexpr,  # int
        HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
        USE_ALIBI_SLOPES: tl.constexpr,  # bool
        USE_SOFTCAP: tl.constexpr,  # bool
        SLIDING_WINDOW: tl.constexpr,  # int
        stride_k_cache_0: tl.int64,  # int
        stride_k_cache_1: tl.int64,  # int
        stride_k_cache_2: tl.int64,  # int
        stride_k_cache_3: tl.constexpr,  # int
        stride_v_cache_0: tl.int64,  # int
        stride_v_cache_1: tl.int64,  # int
        stride_v_cache_2: tl.int64,  # int
        stride_v_cache_3: tl.constexpr,  # int
        stride_k_compact_0: tl.int64,  # int
        stride_k_compact_1: tl.int64,  # int
        stride_k_compact_2: tl.int64,  # int
        stride_k_compact_3: tl.constexpr,  # int
        stride_v_compact_0: tl.int64,  # int
        stride_v_compact_1: tl.int64,  # int
        stride_v_compact_2: tl.int64,  # int
        stride_v_compact_3: tl.constexpr,  # int
        query_start_len_ptr,  # [num_seqs+1]
        BLOCK_Q: tl.constexpr,  # int
        num_seqs: tl.int32,
	        BLOCK_M: tl.constexpr,  # int
	        COMPACT_ONLY: tl.constexpr,  # bool
	        FORCE_DENSE: tl.constexpr,  # bool
	        ENABLE_LOGITS_CAPTURE: tl.constexpr,  # bool
	        ENABLE_LOGF: tl.constexpr,  # bool
	        WRITE_QUERY_NORMS: tl.constexpr,  # bool
	        PREFETCH_COMPACT: tl.constexpr,  # bool
	        STORE_LOGITS_FOR_LOGF_LASTN1: tl.constexpr,  # bool
	        STORE_LOGITS_FOR_LOGF_GT1_SCRATCH: tl.constexpr,  # bool
	        LOGF_OUT_FP32: tl.constexpr,  # bool
	        SKIP_OUTPUT: tl.constexpr,  # bool (logits-only path: skip softmax/value/out)
):
    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)

    seq_idx = find_seq_idx(query_start_len_ptr, q_block_global_idx, num_seqs,
                           BLOCK_Q, True)

    q_block_start_idx = tl.load(query_start_len_ptr +
                                seq_idx) // BLOCK_Q + seq_idx

    q_block_local_idx = q_block_global_idx - q_block_start_idx

    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)

    cur_batch_query_len = cur_batch_in_all_stop_index \
        - cur_batch_in_all_start_index

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + \
        offs_m % num_queries_per_kv
    query_offset = (query_offset_0[:, None] * query_stride_0 +
                    query_offset_1[:, None] * query_stride_1 + offs_d[None, :])

    dim_mask = tl.where(offs_d < HEAD_SIZE, 1, 0).to(tl.int1)
    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)

    # Q : (BLOCK_M, HEAD_SIZE_PADDED)
    Q = tl.load(
        query_ptr + query_offset,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
    )

    meta32_row_ptr = req_meta_i32_ptr + seq_idx * req_meta_i32_stride_row
    meta64_row_ptr = req_meta_i64_ptr + seq_idx * req_meta_i64_stride_row

    # per-request metadata - new compact layout:
    # req_meta_i32[7]: [kv_len_visible, compact_block_cnt, logits_last_n,
    #                  logits_row_offset, logits_capacity, flags, recent_len]
    # req_meta_i64[4]: [block_row_base_paged, compact_base_block, logits_base_ptr, token_row_base/query_norms_ptr]
    kv_len_visible = tl.load(meta32_row_ptr + 0 * req_meta_i32_stride_col).to(tl.int32)
    compact_block_cnt = tl.load(meta32_row_ptr + 1 * req_meta_i32_stride_col).to(tl.int32)
    logits_last_n = tl.load(meta32_row_ptr + 2 * req_meta_i32_stride_col).to(tl.int32)
    logits_row_offset = tl.load(meta32_row_ptr + 3 * req_meta_i32_stride_col).to(tl.int32)
    logits_capacity = tl.load(meta32_row_ptr + 4 * req_meta_i32_stride_col).to(tl.int32)
    flags = tl.load(meta32_row_ptr + 5 * req_meta_i32_stride_col)
    recent_len = tl.load(meta32_row_ptr + 6 * req_meta_i32_stride_col).to(tl.int32)

    block_row_base_paged = tl.load(meta64_row_ptr + 0 * req_meta_i64_stride_col).to(tl.int64)
    compact_base_block = tl.load(meta64_row_ptr + 1 * req_meta_i64_stride_col).to(tl.int64)
    logits_base_ptr = tl.load(meta64_row_ptr + 2 * req_meta_i64_stride_col).to(tl.int64)
    token_row_base = tl.load(meta64_row_ptr + 3 * req_meta_i64_stride_col).to(tl.int64)

    context_seq_len = tl.load(seqused_k_ptr + seq_idx).to(tl.int32)
    # decode fast-path: optionally derive kv_len/recent_len from seqused_k + recent_cap
    RECENT_CAP_FLAG = 4
    use_cap = (flags & RECENT_CAP_FLAG) != 0
    kv_len_visible = tl.where(use_cap, context_seq_len, kv_len_visible)

    if FORCE_DENSE:
        # 强制 dense：用 int1 避免后续 bitwise (~use_compact) 的类型歧义
        use_compact = tl.full([], 0, tl.int1)
        use_logits = tl.full([], 0, tl.int1)
        use_log_f = tl.full([], 0, tl.int1)
    elif COMPACT_ONLY:
        use_compact = 1
        # compact-only 变体不写 logits
        use_logits = 0
        use_log_f = 0
    else:
        use_compact = (flags & 1) != 0
        # compact 行不写 logits，防止额外寄存器/访存开销
        use_logits = ((flags & 2) != 0) & (use_compact == 0)
        use_log_f = ((flags & 8) != 0) & (use_compact == 0)
        # 变体裁剪：编译期禁用时直接清零，避免后续指针/stride 计算占用寄存器。
        if tl.constexpr(not ENABLE_LOGITS_CAPTURE):
            use_logits = tl.full([], 0, tl.int1)
        if tl.constexpr(not ENABLE_LOGF):
            use_log_f = tl.full([], 0, tl.int1)
    seq_len = kv_len_visible
    context_delta = context_seq_len - cur_batch_query_len
    context_len = tl.where(context_delta > 0, context_delta, 0)
    query_mask = query_mask_0 & query_mask_1
    block_tables_ptr = block_tables_paged_ptr + block_row_base_paged * block_tables_paged_stride_row

    context_upper = context_len + query_pos

    if USE_ALIBI_SLOPES:
        context_len_f32 = context_len.to(tl.float32)

    if tl.constexpr(ENABLE_LOGITS_CAPTURE or ENABLE_LOGF):
        logits_capacity_i64 = logits_capacity.to(tl.int64)
        logits_ptr = tl.cast(logits_dummy_ptr, tl.pointer_type(tl.float32))
        log_f_ptr = tl.cast(logits_dummy_ptr, tl.pointer_type(tl.float32 if LOGF_OUT_FP32 else tl.float16))
        logits_has_ptr = tl.full([], 0, tl.int1)
        log_f_has_ptr = tl.full([], 0, tl.int1)
        logits_stride_token = tl.full([], 0, tl.int64)
        logits_stride_head = tl.full([], 0, tl.int64)

        dummy_logits_addr = tl.cast(logits_dummy_ptr, tl.int64)
        logits_stride_token = tl.full([], 1, tl.int64)

        stride_logits = tl.full([], 0, tl.int64)
        stride_log_f = tl.full([], 0, tl.int64)
        if tl.constexpr(ENABLE_LOGITS_CAPTURE):
            logits_has_ptr = use_logits & (logits_base_ptr != 0) & (logits_last_n > 0) & (logits_capacity > 0)
            stride_logits = (logits_last_n.to(tl.int64) * logits_capacity_i64)
        if tl.constexpr(ENABLE_LOGF):
            log_f_has_ptr = use_log_f & (logits_base_ptr != 0) & (logits_capacity > 0)
            # log_f 的输出 buffer 允许使用"pad stride"（便于上游 step-wise 复用）。
            # 约定：dense log_f 场景下 req_meta_i32[1] 传 log_f_stride_head（>= logits_capacity）。
            stride_log_f = compact_block_cnt.to(tl.int64)
            stride_log_f = tl.where(stride_log_f > 0, stride_log_f, logits_capacity_i64)

        if tl.constexpr(ENABLE_LOGF and ENABLE_LOGITS_CAPTURE):
            logits_stride_head = tl.where(log_f_has_ptr, stride_log_f, stride_logits)
        elif tl.constexpr(ENABLE_LOGF):
            logits_stride_head = stride_log_f
        elif tl.constexpr(ENABLE_LOGITS_CAPTURE):
            logits_stride_head = stride_logits

        selected_logits_addr = tl.where((logits_has_ptr | log_f_has_ptr), logits_base_ptr, dummy_logits_addr)
        logits_ptr = tl.cast(selected_logits_addr, tl.pointer_type(tl.float32))
        log_f_ptr = tl.cast(selected_logits_addr, tl.pointer_type(tl.float32 if LOGF_OUT_FP32 else tl.float16))

        if tl.constexpr(ENABLE_LOGF):
            # last_n>1 log_f fastest-path: store per-row logits into a scratch buffer (per-layer, not accumulated across layers).
            # Contract (dense rows only): meta64[1] is treated as scratch_ptr when STORE_LOGITS_FOR_LOGF_GT1_SCRATCH is enabled.
            logits_scratch_ptr = tl.cast(logits_dummy_ptr, tl.pointer_type(tl.float32))
            logits_scratch_has_ptr = tl.full([], 0, tl.int1)
            logits_scratch_stride_head = tl.full([], 0, tl.int64)
            logits_scratch_stride_token = tl.full([], 1, tl.int64)
            logits_scratch_has_ptr = use_log_f & STORE_LOGITS_FOR_LOGF_GT1_SCRATCH & (logits_last_n > 1) \
                & (use_compact == 0) & (compact_base_block != 0) & (logits_capacity > 0) & (compact_block_cnt > 0)
            logits_scratch_ptr = tl.cast(compact_base_block, tl.pointer_type(tl.float32))
            # scratch layout: [Hq, last_n, stride_log_f] fp32
            logits_scratch_stride_head = logits_last_n.to(tl.int64) * stride_log_f

    query_norms_ptr = tl.cast(logits_dummy_ptr, tl.pointer_type(tl.float32))
    query_norms_has_ptr = tl.full([], 0, tl.int1)
    if tl.constexpr(WRITE_QUERY_NORMS):
        dummy_logits_addr = tl.cast(logits_dummy_ptr, tl.int64)
        query_norms_has_ptr = use_logits & (token_row_base != 0) & (logits_last_n > 0)
        selected_query_norms_addr = tl.where(query_norms_has_ptr, token_row_base, dummy_logits_addr)
        query_norms_ptr = tl.cast(selected_query_norms_addr, tl.pointer_type(tl.float32))

    if tl.constexpr(WRITE_QUERY_NORMS):
        q_f = Q.to(tl.float32)
        norms = tl.sqrt(tl.sum(q_f * q_f, axis=1))
        norm_row_idx = query_pos - logits_row_offset
        start = query_norms_window - logits_last_n
        write_idx = start + norm_row_idx
        store_mask = query_norms_has_ptr & query_mask & (norm_row_idx >= 0) & (norm_row_idx < logits_last_n)
        head_offsets = query_offset_1.to(tl.int64) * query_norms_stride_head
        write_offsets = head_offsets + write_idx.to(tl.int64) * query_norms_stride_token
        tl.store(query_norms_ptr + write_offsets, norms, mask=store_mask)

    M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)

    # alibi slope for this head
    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(alibi_slopes_ptr + query_offset_1,
                              mask=query_mask_1,
                              other=0.0,
                             )

    recent_cap = tl.maximum(recent_len, 0)
    cap = tl.minimum(context_seq_len, recent_cap)
    cap = tl.maximum(cap, 0)
    recent_start_calc = ((context_seq_len - cap) // BLOCK_SIZE) * BLOCK_SIZE
    recent_start_calc = tl.maximum(recent_start_calc, 0)
    recent_len_calc = tl.maximum(context_seq_len - recent_start_calc, 0)
    recent_len_from_cap = tl.where(recent_cap > 0, recent_len_calc, 0)
    recent_len_from_len = tl.maximum(recent_len, 0)
    recent_len = tl.where(use_cap, recent_len_from_cap, recent_len_from_len)
    recent_block_cnt = cdiv_fn(recent_len, BLOCK_SIZE)
    recent_start = context_seq_len - recent_len
    context_upper = context_upper[:, None]

    # --- compact 段 ---
    if use_compact:
        # int32 转换：仅 block_ptr 需要的 stride 转换为 int32
        stride_v_compact_0_i32 = _rt_i32(stride_v_compact_0)
        stride_k_compact_0_i32 = _rt_i32(stride_k_compact_0)
        stride_v_compact_1_i32 = _rt_i32(stride_v_compact_1)
        stride_k_compact_1_i32 = _rt_i32(stride_k_compact_1)
        stride_v_compact_2_i32 = _rt_i32(stride_v_compact_2)
        stride_k_compact_2_i32 = _rt_i32(stride_k_compact_2)
        token_pos_stride_block_i32 = _rt_i32(token_pos_stride_block)
        token_pos_stride_token_i32 = _rt_i32(token_pos_stride_token)

        tl.multiple_of(stride_v_compact_1_i32, 16)
        tl.multiple_of(stride_k_compact_1_i32, 16)

        token_indices_ptr = token_positions_ptr + token_row_base
        token_head_offset = kv_head_idx * token_pos_stride_head
        token_indices_ptr_head = token_indices_ptr + token_head_offset
        offs_n = tl.arange(0, BLOCK_SIZE)

        if PREFETCH_COMPACT:
            # 预取/双缓冲：使用 block_ptr 优化 (模式 B: offsets=0,0)
            head_v_offset = kv_head_idx * stride_v_compact_2
            head_k_offset = kv_head_idx * stride_k_compact_2

            K_next = tl.zeros([HEAD_SIZE_PADDED, BLOCK_SIZE], dtype=Q.dtype)
            V_next = tl.zeros([BLOCK_SIZE, HEAD_SIZE_PADDED], dtype=Q.dtype)
            if compact_block_cnt > 0:
                phys0 = compact_base_block
                v_block_ptr0 = tl.make_block_ptr(
                    base=value_compact_ptr + (phys0 * stride_v_compact_0 + head_v_offset),
                    shape=(BLOCK_SIZE, HEAD_SIZE_PADDED),
                    strides=(stride_v_compact_1, stride_v_compact_3),
                    offsets=(0, 0),
                    block_shape=(BLOCK_SIZE, HEAD_SIZE_PADDED),
                    order=(0, 1),
                )
                k_block_ptr0 = tl.make_block_ptr(
                    base=key_compact_ptr + (phys0 * stride_k_compact_0 + head_k_offset),
                    shape=(HEAD_SIZE_PADDED, BLOCK_SIZE),
                    strides=(stride_k_compact_3, stride_k_compact_1),
                    offsets=(0, 0),
                    block_shape=(HEAD_SIZE_PADDED, BLOCK_SIZE),
                    order=(0, 1),
                )
                V_next = tl.load(v_block_ptr0, boundary_check=(0, 1), padding_option="zero")
                K_next = tl.load(k_block_ptr0, boundary_check=(0, 1), padding_option="zero")

            for j in range(0, compact_block_cnt):
                K_load = K_next
                V_load = V_next

                if j + 1 < compact_block_cnt:
                    nxt = j + 1
                    phys_nxt = compact_base_block + nxt
                    v_block_ptr_nxt = tl.make_block_ptr(
                        base=value_compact_ptr + (phys_nxt * stride_v_compact_0 + head_v_offset),
                        shape=(BLOCK_SIZE, HEAD_SIZE_PADDED),
                        strides=(stride_v_compact_1, stride_v_compact_3),
                        offsets=(0, 0),
                        block_shape=(BLOCK_SIZE, HEAD_SIZE_PADDED),
                        order=(0, 1),
                    )
                    k_block_ptr_nxt = tl.make_block_ptr(
                        base=key_compact_ptr + (phys_nxt * stride_k_compact_0 + head_k_offset),
                        shape=(HEAD_SIZE_PADDED, BLOCK_SIZE),
                        strides=(stride_k_compact_3, stride_k_compact_1),
                        offsets=(0, 0),
                        block_shape=(HEAD_SIZE_PADDED, BLOCK_SIZE),
                        order=(0, 1),
                    )
                    V_next = tl.load(v_block_ptr_nxt, boundary_check=(0, 1), padding_option="zero")
                    K_next = tl.load(k_block_ptr_nxt, boundary_check=(0, 1), padding_option="zero")

                # Token positions 加载使用 block_ptr
                token_blk_ptr = tl.make_block_ptr(
                    base=token_indices_ptr_head + j * token_pos_stride_block,
                    shape=(BLOCK_SIZE,),
                    strides=(token_pos_stride_token,),
                    offsets=(0,),
                    block_shape=(BLOCK_SIZE,),
                    order=(0,),
                )
                seq_offset = tl.load(token_blk_ptr, boundary_check=(0,), padding_option="zero").to(tl.int32)
                valid_token_mask = seq_offset >= 0

                if K_load.dtype.is_fp8():
                    if Q.dtype.is_fp8():
                        K = K_load
                    else:
                        K = (K_load.to(tl.float32) * tl.load(k_scale)).to(Q.dtype)
                else:
                    K = K_load

                if tl.constexpr(not SKIP_OUTPUT):
                    if V_load.dtype.is_fp8():
                        if Q.dtype.is_fp8():
                            V = V_load
                        else:
                            V = (V_load.to(tl.float32) * tl.load(v_scale)).to(Q.dtype)
                    else:
                        V = V_load
                else:
                    V = tl.zeros([BLOCK_SIZE, HEAD_SIZE_PADDED], dtype=Q.dtype)

                seq_mask = valid_token_mask[None, :] & (seq_offset[None, :] < seq_len) & (seq_offset[None, :] <= context_upper)
                if SLIDING_WINDOW > 0:
                    seq_mask = seq_mask & ((context_len + query_pos[:, None] - seq_offset) < SLIDING_WINDOW)

                if USE_ALIBI_SLOPES:
                    seq_offset_f32 = seq_offset.to(tl.float32)

                S = scale * tl.dot(Q, K)

                if USE_SOFTCAP:
                    S = apply_softcap(S, softcap)

                S = tl.where(query_mask[:, None] & seq_mask,
                             S, float("-inf"))

                if USE_ALIBI_SLOPES:
                    S += alibi_slope[:, None] * (seq_offset_f32 - context_len_f32)

                if tl.constexpr(ENABLE_LOGITS_CAPTURE):
                    # compact 行按设计不写 logits/log_f，避免额外寄存器与全局内存开销
                    if use_logits:
                        maybe_store_logits(
                            logits_ptr,
                            logits_has_ptr,
                            logits_row_offset,
                            logits_last_n,
                            logits_capacity,
                            logits_capacity_i64,
                            logits_stride_head,
                            logits_stride_token,
                            query_pos,
                            query_mask_0,
                            query_mask_1,
                            seq_offset,
                            seq_mask,
                            query_offset_1,
                            S,
                        )

                m_j = tl.maximum(M, tl.max(S, axis=1))
                m_j = tl.where(m_j > float("-inf"), m_j, 0.0)

                P = tl.exp(S - m_j[:, None])
                l_j = tl.sum(P, axis=1)
                alpha = tl.exp(M - m_j)
                acc = acc * alpha[:, None]
                L = L * alpha + l_j
                M = m_j
                acc += tl.dot(P.to(V.dtype), V)
        else:
            # 无预取路径：非尾块/尾块分离优化（使用模式 B）
            # 计算非尾块结束位置（尾块可能有无效 token）
            non_tail_end = tl.maximum(0, compact_block_cnt - 1)

            # === 非尾块循环：省去 valid_token_mask 检查 ===
            for j in range(0, non_tail_end):
                phys = compact_base_block + j
                v_block_ptr = tl.make_block_ptr(
                    base=value_compact_ptr + (phys * stride_v_compact_0 + kv_head_idx * stride_v_compact_2),
                    shape=(BLOCK_SIZE, HEAD_SIZE_PADDED),
                    strides=(stride_v_compact_1, stride_v_compact_3),
                    offsets=(0, 0),
                    block_shape=(BLOCK_SIZE, HEAD_SIZE_PADDED),
                    order=(0, 1),
                )
                k_block_ptr = tl.make_block_ptr(
                    base=key_compact_ptr + (phys * stride_k_compact_0 + kv_head_idx * stride_k_compact_2),
                    shape=(HEAD_SIZE_PADDED, BLOCK_SIZE),
                    strides=(stride_k_compact_3, stride_k_compact_1),
                    offsets=(0, 0),
                    block_shape=(HEAD_SIZE_PADDED, BLOCK_SIZE),
                    order=(0, 1),
                )
                V_load = tl.load(v_block_ptr, boundary_check=(0, 1), padding_option="zero")
                K_load = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option="zero")

                # Token positions 加载
                token_blk_ptr = tl.make_block_ptr(
                    base=token_indices_ptr_head + j * token_pos_stride_block,
                    shape=(BLOCK_SIZE,),
                    strides=(token_pos_stride_token,),
                    offsets=(0,),
                    block_shape=(BLOCK_SIZE,),
                    order=(0,),
                )
                seq_offset = tl.load(token_blk_ptr, boundary_check=(0,), padding_option="zero").to(tl.int32)

                if K_load.dtype.is_fp8():
                    if Q.dtype.is_fp8():
                        K = K_load
                    else:
                        K = (K_load.to(tl.float32) * tl.load(k_scale)).to(Q.dtype)
                else:
                    K = K_load

                if V_load.dtype.is_fp8():
                    if Q.dtype.is_fp8():
                        V = V_load
                    else:
                        V = (V_load.to(tl.float32) * tl.load(v_scale)).to(Q.dtype)
                else:
                    V = V_load

                # 非尾块：所有 token 有效（seq_offset >= 0），省去该检查
                seq_mask = (seq_offset[None, :] < seq_len) & (seq_offset[None, :] <= context_upper)
                if SLIDING_WINDOW > 0:
                    seq_mask = seq_mask & ((context_len + query_pos[:, None] - seq_offset) < SLIDING_WINDOW)

                if USE_ALIBI_SLOPES:
                    seq_offset_f32 = seq_offset.to(tl.float32)

                S = scale * tl.dot(Q, K)

                if USE_SOFTCAP:
                    S = apply_softcap(S, softcap)

                S = tl.where(query_mask[:, None] & seq_mask, S, float("-inf"))

                if USE_ALIBI_SLOPES:
                    S += alibi_slope[:, None] * (seq_offset_f32 - context_len_f32)

                if tl.constexpr(ENABLE_LOGITS_CAPTURE):
                    if use_logits:
                        maybe_store_logits(
                            logits_ptr,
                            logits_has_ptr,
                            logits_row_offset,
                            logits_last_n,
                            logits_capacity,
                            logits_capacity_i64,
                            logits_stride_head,
                            logits_stride_token,
                            query_pos,
                            query_mask_0,
                            query_mask_1,
                            seq_offset,
                            seq_mask,
                            query_offset_1,
                            S,
                        )

                if tl.constexpr(not SKIP_OUTPUT):
                    m_j = tl.maximum(M, tl.max(S, axis=1))
                    m_j = tl.where(m_j > float("-inf"), m_j, 0.0)
                    P = tl.exp(S - m_j[:, None])
                    l_j = tl.sum(P, axis=1)
                    alpha = tl.exp(M - m_j)
                    acc = acc * alpha[:, None]
                    L = L * alpha + l_j
                    M = m_j
                    acc += tl.dot(P.to(V.dtype), V)

            # === 尾块处理：可能有无效 token（seq_offset < 0），需要完整检查 ===
            if compact_block_cnt > 0:
                j_tail = compact_block_cnt - 1
                phys_tail = compact_base_block + j_tail
                # 使用模式 B: offsets=0,0
                v_block_ptr = tl.make_block_ptr(
                    base=value_compact_ptr + (phys_tail * stride_v_compact_0 + kv_head_idx * stride_v_compact_2),
                    shape=(BLOCK_SIZE, HEAD_SIZE_PADDED),
                    strides=(stride_v_compact_1, stride_v_compact_3),
                    offsets=(0, 0),
                    block_shape=(BLOCK_SIZE, HEAD_SIZE_PADDED),
                    order=(0, 1),
                )
                k_block_ptr = tl.make_block_ptr(
                    base=key_compact_ptr + (phys_tail * stride_k_compact_0 + kv_head_idx * stride_k_compact_2),
                    shape=(HEAD_SIZE_PADDED, BLOCK_SIZE),
                    strides=(stride_k_compact_3, stride_k_compact_1),
                    offsets=(0, 0),
                    block_shape=(HEAD_SIZE_PADDED, BLOCK_SIZE),
                    order=(0, 1),
                )
                V_load = tl.load(v_block_ptr, boundary_check=(0, 1), padding_option="zero")
                K_load = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option="zero")

                token_blk_ptr = tl.make_block_ptr(
                    base=token_indices_ptr_head + j_tail * token_pos_stride_block,
                    shape=(BLOCK_SIZE,),
                    strides=(token_pos_stride_token,),
                    offsets=(0,),
                    block_shape=(BLOCK_SIZE,),
                    order=(0,),
                )
                seq_offset = tl.load(token_blk_ptr, boundary_check=(0,), padding_option="zero").to(tl.int32)
                valid_token_mask = seq_offset >= 0

                if K_load.dtype.is_fp8():
                    if Q.dtype.is_fp8():
                        K = K_load
                    else:
                        K = (K_load.to(tl.float32) * tl.load(k_scale)).to(Q.dtype)
                else:
                    K = K_load

                if V_load.dtype.is_fp8():
                    if Q.dtype.is_fp8():
                        V = V_load
                    else:
                        V = (V_load.to(tl.float32) * tl.load(v_scale)).to(Q.dtype)
                else:
                    V = V_load

                # 尾块：需要完整的 valid_token_mask 检查
                seq_mask = valid_token_mask[None, :] & (seq_offset[None, :] < seq_len) & (seq_offset[None, :] <= context_upper)
                if SLIDING_WINDOW > 0:
                    seq_mask = seq_mask & ((context_len + query_pos[:, None] - seq_offset) < SLIDING_WINDOW)

                if USE_ALIBI_SLOPES:
                    seq_offset_f32 = seq_offset.to(tl.float32)

                S = scale * tl.dot(Q, K)

                if USE_SOFTCAP:
                    S = apply_softcap(S, softcap)

                S = tl.where(query_mask[:, None] & seq_mask, S, float("-inf"))

                if USE_ALIBI_SLOPES:
                    S += alibi_slope[:, None] * (seq_offset_f32 - context_len_f32)

                if tl.constexpr(ENABLE_LOGITS_CAPTURE):
                    if use_logits:
                        maybe_store_logits(
                            logits_ptr,
                            logits_has_ptr,
                            logits_row_offset,
                            logits_last_n,
                            logits_capacity,
                            logits_capacity_i64,
                            logits_stride_head,
                            logits_stride_token,
                            query_pos,
                            query_mask_0,
                            query_mask_1,
                            seq_offset,
                            seq_mask,
                            query_offset_1,
                            S,
                        )

                if tl.constexpr(not SKIP_OUTPUT):
                    m_j = tl.maximum(M, tl.max(S, axis=1))
                    m_j = tl.where(m_j > float("-inf"), m_j, 0.0)
                    P = tl.exp(S - m_j[:, None])
                    l_j = tl.sum(P, axis=1)
                    alpha = tl.exp(M - m_j)
                    acc = acc * alpha[:, None]
                    L = L * alpha + l_j
                    M = m_j
                    acc += tl.dot(P.to(V.dtype), V)

    # --- recent 段（paged 尾部）---
    if (use_compact != 0) & (recent_block_cnt > 0):
        recent_start = context_seq_len - recent_len
        tail_block_start = (context_seq_len - recent_len) // BLOCK_SIZE
        offs_n = tl.arange(0, BLOCK_SIZE)

        if PREFETCH_COMPACT:
            # 预取首块 recent，使用 block_ptr 优化 (模式 B: offsets=0,0)
            head_v_cache_offset = kv_head_idx * stride_v_cache_2
            head_k_cache_offset = kv_head_idx * stride_k_cache_2

            seq_block_idx_next = tail_block_start
            block_id_raw = tl.load(block_tables_ptr + seq_block_idx_next).to(tl.int64)
            block_valid_next = block_id_raw >= 0
            block_id_next = tl.where(block_valid_next, block_id_raw, 0)
            v_block_ptr_next = tl.make_block_ptr(
                base=value_cache_ptr + (block_id_next * stride_v_cache_0 + head_v_cache_offset),
                shape=(BLOCK_SIZE, HEAD_SIZE_PADDED),
                strides=(stride_v_cache_1, stride_v_cache_3),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE, HEAD_SIZE_PADDED),
                order=(0, 1),
            )
            k_block_ptr_next = tl.make_block_ptr(
                base=key_cache_ptr + (block_id_next * stride_k_cache_0 + head_k_cache_offset),
                shape=(HEAD_SIZE_PADDED, BLOCK_SIZE),
                strides=(stride_k_cache_3, stride_k_cache_1),
                offsets=(0, 0),
                block_shape=(HEAD_SIZE_PADDED, BLOCK_SIZE),
                order=(0, 1),
            )
            V_next = tl.load(v_block_ptr_next, boundary_check=(0, 1), padding_option="zero")
            K_next = tl.load(k_block_ptr_next, boundary_check=(0, 1), padding_option="zero")

            for r in range(0, recent_block_cnt):
                seq_block_idx = tail_block_start + r
                K_load = K_next
                V_load = V_next
                block_is_valid = block_valid_next

                if r + 1 < recent_block_cnt:
                    seq_block_idx_next = tail_block_start + r + 1
                    block_id_raw = tl.load(block_tables_ptr + seq_block_idx_next).to(tl.int64)
                    block_valid_next = block_id_raw >= 0
                    block_id_next = tl.where(block_valid_next, block_id_raw, 0)
                    v_block_ptr_nxt = tl.make_block_ptr(
                        base=value_cache_ptr + (block_id_next * stride_v_cache_0 + head_v_cache_offset),
                        shape=(BLOCK_SIZE, HEAD_SIZE_PADDED),
                        strides=(stride_v_cache_1, stride_v_cache_3),
                        offsets=(0, 0),
                        block_shape=(BLOCK_SIZE, HEAD_SIZE_PADDED),
                        order=(0, 1),
                    )
                    k_block_ptr_nxt = tl.make_block_ptr(
                        base=key_cache_ptr + (block_id_next * stride_k_cache_0 + head_k_cache_offset),
                        shape=(HEAD_SIZE_PADDED, BLOCK_SIZE),
                        strides=(stride_k_cache_3, stride_k_cache_1),
                        offsets=(0, 0),
                        block_shape=(HEAD_SIZE_PADDED, BLOCK_SIZE),
                        order=(0, 1),
                    )
                    V_next = tl.load(v_block_ptr_nxt, boundary_check=(0, 1), padding_option="zero")
                    K_next = tl.load(k_block_ptr_nxt, boundary_check=(0, 1), padding_option="zero")

                if K_load.dtype.is_fp8():
                    if Q.dtype.is_fp8():
                        K = K_load
                    else:
                        K = (K_load.to(tl.float32) * tl.load(k_scale)).to(Q.dtype)
                else:
                    K = K_load

                if V_load.dtype.is_fp8():
                    if Q.dtype.is_fp8():
                        V = V_load
                    else:
                        V = (V_load.to(tl.float32) * tl.load(v_scale)).to(Q.dtype)
                else:
                    V = V_load

                seq_offset = (seq_block_idx * BLOCK_SIZE + offs_n).to(tl.int32)
                valid_token_mask = block_is_valid & (seq_offset >= recent_start)
                seq_mask = valid_token_mask[None, :] & (seq_offset[None, :] < seq_len) & (seq_offset[None, :] <= context_upper)
                if SLIDING_WINDOW > 0:
                    seq_mask = seq_mask & ((context_len + query_pos[:, None] - seq_offset) < SLIDING_WINDOW)

                if USE_ALIBI_SLOPES:
                    seq_offset_f32 = seq_offset.to(tl.float32)

                S = scale * tl.dot(Q, K)

                if USE_SOFTCAP:
                    S = apply_softcap(S, softcap)

                S = tl.where(query_mask[:, None] & seq_mask,
                             S, float("-inf"))

                if USE_ALIBI_SLOPES:
                    S += alibi_slope[:, None] * (seq_offset_f32 - context_len_f32)

                if tl.constexpr(ENABLE_LOGITS_CAPTURE):
                    if use_logits:
                        maybe_store_logits(
                            logits_ptr,
                            logits_has_ptr,
                            logits_row_offset,
                            logits_last_n,
                            logits_capacity,
                            logits_capacity_i64,
                            logits_stride_head,
                            logits_stride_token,
                            query_pos,
                            query_mask_0,
                            query_mask_1,
                            seq_offset,
                            seq_mask,
                            query_offset_1,
                            S,
                        )

                m_j = tl.maximum(M, tl.max(S, axis=1))
                m_j = tl.where(m_j > float("-inf"), m_j, 0.0)
                P = tl.exp(S - m_j[:, None])
                l_j = tl.sum(P, axis=1)
                alpha = tl.exp(M - m_j)
                acc = acc * alpha[:, None]
                L = L * alpha + l_j
                M = m_j
                acc += tl.dot(P.to(V.dtype), V)
        else:
            # 非预取路径：使用 block_ptr 优化 (模式 B: offsets=0,0)
            head_v_cache_offset = kv_head_idx * stride_v_cache_2
            head_k_cache_offset = kv_head_idx * stride_k_cache_2

            for r in range(0, recent_block_cnt):
                seq_block_idx = tail_block_start + r
                block_id_raw = tl.load(block_tables_ptr + seq_block_idx).to(tl.int64)
                block_is_valid = block_id_raw >= 0
                block_id = tl.where(block_is_valid, block_id_raw, 0)
                # KV cache 加载使用 block_ptr (模式 B: offsets=0,0)
                v_block_ptr = tl.make_block_ptr(
                    base=value_cache_ptr + (block_id * stride_v_cache_0 + head_v_cache_offset),
                    shape=(BLOCK_SIZE, HEAD_SIZE_PADDED),
                    strides=(stride_v_cache_1, stride_v_cache_3),
                    offsets=(0, 0),
                    block_shape=(BLOCK_SIZE, HEAD_SIZE_PADDED),
                    order=(0, 1),
                )
                k_block_ptr = tl.make_block_ptr(
                    base=key_cache_ptr + (block_id * stride_k_cache_0 + head_k_cache_offset),
                    shape=(HEAD_SIZE_PADDED, BLOCK_SIZE),
                    strides=(stride_k_cache_3, stride_k_cache_1),
                    offsets=(0, 0),
                    block_shape=(HEAD_SIZE_PADDED, BLOCK_SIZE),
                    order=(0, 1),
                )
                V_load = tl.load(v_block_ptr, boundary_check=(0, 1), padding_option="zero")
                K_load = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option="zero")

                if K_load.dtype.is_fp8():
                    if Q.dtype.is_fp8():
                        K = K_load
                    else:
                        K = (K_load.to(tl.float32) * tl.load(k_scale)).to(Q.dtype)
                else:
                    K = K_load

                if V_load.dtype.is_fp8():
                    if Q.dtype.is_fp8():
                        V = V_load
                    else:
                        V = (V_load.to(tl.float32) * tl.load(v_scale)).to(Q.dtype)
                else:
                    V = V_load

                seq_offset = (seq_block_idx * BLOCK_SIZE + offs_n).to(tl.int32)
                valid_token_mask = block_is_valid & (seq_offset >= recent_start)
                seq_mask = valid_token_mask[None, :] & (seq_offset[None, :] < seq_len) & (seq_offset[None, :] <= context_upper)
                if SLIDING_WINDOW > 0:
                    seq_mask = seq_mask & ((context_len + query_pos[:, None] - seq_offset) < SLIDING_WINDOW)

                if USE_ALIBI_SLOPES:
                    seq_offset_f32 = seq_offset.to(tl.float32)

                S = scale * tl.dot(Q, K)

                if USE_SOFTCAP:
                    S = apply_softcap(S, softcap)

                S = tl.where(query_mask[:, None] & seq_mask,
                             S, float("-inf"))

                if USE_ALIBI_SLOPES:
                    S += alibi_slope[:, None] * (seq_offset_f32 - context_len_f32)

                if tl.constexpr(ENABLE_LOGITS_CAPTURE):
                    if use_logits:
                        maybe_store_logits(
                            logits_ptr,
                            logits_has_ptr,
                            logits_row_offset,
                            logits_last_n,
                            logits_capacity,
                            logits_capacity_i64,
                            logits_stride_head,
                            logits_stride_token,
                            query_pos,
                            query_mask_0,
                            query_mask_1,
                            seq_offset,
                            seq_mask,
                            query_offset_1,
                            S,
                        )

                m_j = tl.maximum(M, tl.max(S, axis=1))
                m_j = tl.where(m_j > float("-inf"), m_j, 0.0)
                P = tl.exp(S - m_j[:, None])
                l_j = tl.sum(P, axis=1)
                alpha = tl.exp(M - m_j)
                acc = acc * alpha[:, None]
                L = L * alpha + l_j
                M = m_j
                acc += tl.dot(P.to(V.dtype), V)

    # --- dense 路径 ---
    # 优化：使用 pointer arithmetic（与 vLLM 一致），简化 seq_mask
    if not COMPACT_ONLY:
        if use_compact == 0:
            num_blocks = cdiv_fn(seq_len, BLOCK_SIZE)
            offs_n = tl.arange(0, BLOCK_SIZE)

            for j in range(0, num_blocks):
                block_id = tl.load(block_tables_ptr + j).to(tl.int64)
                # logits-only(refresh) 路径：允许 block_table 在边界处包含 -1（避免非法访存导致整卡崩溃）
                if tl.constexpr(SKIP_OUTPUT):
                    block_is_valid = block_id >= 0
                    block_id = tl.where(block_is_valid, block_id, 0)
                else:
                    block_is_valid = tl.full([], 1, tl.int1)

                # KV 加载使用 pointer arithmetic（与 vLLM 一致）
                v_offset = (block_id * stride_v_cache_0 +
                            kv_head_idx * stride_v_cache_2 +
                            offs_d[None, :] * stride_v_cache_3 +
                            offs_n[:, None] * stride_v_cache_1)
                k_offset = (block_id * stride_k_cache_0 +
                            kv_head_idx * stride_k_cache_2 +
                            offs_d[:, None] * stride_k_cache_3 +
                            offs_n[None, :] * stride_k_cache_1)

                K_load = tl.load(
                    key_cache_ptr + k_offset,
                    mask=dim_mask[:, None] & block_is_valid,
                    other=0.0,
                )
                if tl.constexpr(not SKIP_OUTPUT):
                    V_load = tl.load(
                        value_cache_ptr + v_offset,
                        mask=dim_mask[None, :] & block_is_valid,
                        other=0.0,
                    )
                else:
                    V_load = tl.zeros([BLOCK_SIZE, HEAD_SIZE_PADDED], dtype=K_load.dtype)

                if K_load.dtype.is_fp8():
                    if Q.dtype.is_fp8():
                        K = K_load
                    else:
                        K = (K_load.to(tl.float32) * tl.load(k_scale)).to(Q.dtype)
                else:
                    K = K_load

                if V_load.dtype.is_fp8():
                    if Q.dtype.is_fp8():
                        V = V_load
                    else:
                        V = (V_load.to(tl.float32) * tl.load(v_scale)).to(Q.dtype)
                else:
                    V = V_load

                seq_offset = j * BLOCK_SIZE + offs_n
                # 简化 seq_mask：只使用 causal mask（与 vLLM 一致）
                seq_mask = seq_offset[None, :] < context_len + query_pos[:, None] + 1
                if SLIDING_WINDOW > 0:
                    seq_mask = seq_mask & ((context_len + query_pos[:, None] - seq_offset) < SLIDING_WINDOW)

                if USE_ALIBI_SLOPES:
                    seq_offset_f32 = seq_offset.to(tl.float32)

                S = scale * tl.dot(Q, K)

                if USE_SOFTCAP:
                    S = apply_softcap(S, softcap)

                S = tl.where(query_mask[:, None] & seq_mask,
                             S, float("-inf"))

                if USE_ALIBI_SLOPES:
                    S += alibi_slope[:, None] * (seq_offset_f32 - context_len_f32)

                if tl.constexpr(ENABLE_LOGITS_CAPTURE):
                    if use_logits:
                        maybe_store_logits(
                            logits_ptr,
                            logits_has_ptr,
                            logits_row_offset,
                            logits_last_n,
                            logits_capacity,
                            logits_capacity_i64,
                            logits_stride_head,
                            logits_stride_token,
                            query_pos,
                            query_mask_0,
                            query_mask_1,
                            seq_offset,
                            seq_mask,
                            query_offset_1,
                            S,
                        )
                if tl.constexpr(ENABLE_LOGF):
                    if use_log_f & STORE_LOGITS_FOR_LOGF_LASTN1 & (logits_last_n == 1):
                        store_log_f_logits_lastn1(
                            log_f_ptr,
                            log_f_has_ptr,
                            LOGF_OUT_FP32,
                            flags,
                            logits_row_offset,
                            logits_capacity,
                            recent_len,
                            logits_stride_head,
                            logits_stride_token,
                            query_pos,
                            query_mask_0,
                            query_mask_1,
                            seq_offset,
                            seq_mask,
                            query_offset_1,
                            S,
                        )
                    elif use_log_f & STORE_LOGITS_FOR_LOGF_GT1_SCRATCH & (logits_last_n > 1):
                        store_log_f_logits_scratch_lastn_gt1(
                            logits_scratch_ptr,
                            logits_scratch_has_ptr,
                            flags,
                            logits_row_offset,
                            logits_last_n,
                            logits_stride_head,
                            logits_capacity,
                            recent_len,
                            logits_scratch_stride_head,
                            query_pos,
                            query_mask_0,
                            query_mask_1,
                            seq_offset,
                            seq_mask,
                            query_offset_1,
                            S,
                        )

                if tl.constexpr(not SKIP_OUTPUT):
                    m_j = tl.maximum(M, tl.max(S, axis=1))
                    m_j = tl.where(m_j > float("-inf"), m_j, 0.0)
                    P = tl.exp(S - m_j[:, None])
                    l_j = tl.sum(P, axis=1)
                    alpha = tl.exp(M - m_j)
                    acc = acc * alpha[:, None]
                    L = L * alpha + l_j
                    M = m_j
                    acc += tl.dot(P.to(V.dtype), V)

    if tl.constexpr(not SKIP_OUTPUT):
        # epilogue: 安全除法，避免 L==0 时除零（当所有 token 被 mask 时）
        acc = tl.where(L[:, None] == 0.0, 0.0, acc / L[:, None])

        output_offset = (query_offset_0[:, None] * output_stride_0 +
                         query_offset_1[:, None] * output_stride_1 +
                         offs_d[None, :])

        tl.store(
            output_ptr + output_offset,
            acc,
            mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        )

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1, num_stages=1),
        triton.Config({}, num_warps=1, num_stages=2),
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=4, num_stages=4),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=3),
        triton.Config({}, num_warps=8, num_stages=4),
    ],
    key=["HEAD_SIZE_PADDED", "BLOCK_SIZE", "NUM_SEGMENTS_PER_SEQ", "FORCE_DENSE"],
)
@triton.jit
def kernel_unified_attention_3d(
        segm_output_ptr,
        # [num_tokens, num_query_heads, num_segments, head_size]
        segm_max_ptr,  # [num_tokens, num_query_heads, num_segments]
        segm_expsum_ptr,  # [num_tokens, num_query_heads, num_segments]
        query_ptr,  # [num_tokens, num_query_heads, head_size]
        key_cache_ptr,  # [num_blks, blk_size, num_kv_heads, head_size]
        value_cache_ptr,  # [num_blks, blk_size, num_kv_heads, head_size]
        block_tables_paged_ptr,
        key_compact_ptr,
        value_compact_ptr,
        token_positions_ptr,
        req_meta_i32_ptr,
        req_meta_i64_ptr,
        seqused_k_ptr,
        logits_dummy_ptr,
        query_norms_stride_head: tl.int64,
        query_norms_stride_token: tl.int64,
        query_norms_window: tl.int32,
        alibi_slopes_ptr,  # [num_query_heads]
        scale,  # float32
        k_scale,  # float32
        v_scale,  # float32
        softcap,  # float32
        num_query_heads: tl.constexpr,  # int
        num_queries_per_kv: tl.constexpr,  # int
        req_meta_i32_stride_row: tl.int64,
        req_meta_i32_stride_col: tl.int64,
        req_meta_i64_stride_row: tl.int64,
        req_meta_i64_stride_col: tl.int64,
        token_pos_stride_row: tl.int64,
        token_pos_stride_head: tl.int64,
        token_pos_stride_block: tl.int64,
        token_pos_stride_token: tl.int64,
        query_stride_0: tl.int64,  # int
        query_stride_1: tl.int64,  # int, should be equal to head_size
        block_tables_paged_stride_row: tl.int64,
        BLOCK_SIZE: tl.constexpr,  # int
        HEAD_SIZE: tl.constexpr,  # int
        HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
        USE_ALIBI_SLOPES: tl.constexpr,  # bool
        USE_SOFTCAP: tl.constexpr,  # bool
        SLIDING_WINDOW: tl.constexpr,  # int
        stride_k_cache_0: tl.int64,  # int
        stride_k_cache_1: tl.int64,  # int
        stride_k_cache_2: tl.int64,  # int
        stride_k_cache_3: tl.constexpr,  # int
        stride_v_cache_0: tl.int64,  # int
        stride_v_cache_1: tl.int64,  # int
        stride_v_cache_2: tl.int64,  # int
        stride_v_cache_3: tl.constexpr,  # int
        stride_k_compact_0: tl.int64,
        stride_k_compact_1: tl.int64,
        stride_k_compact_2: tl.int64,
        stride_k_compact_3: tl.constexpr,
        stride_v_compact_0: tl.int64,
        stride_v_compact_1: tl.int64,
        stride_v_compact_2: tl.int64,
        stride_v_compact_3: tl.constexpr,
        query_start_len_ptr,  # [num_seqs+1]
        BLOCK_Q: tl.constexpr,  # int
        num_seqs: tl.int32,
        BLOCK_M: tl.constexpr,  # int
	        NUM_SEGMENTS_PER_SEQ: tl.constexpr,  # int
	        COMPACT_ONLY: tl.constexpr,  # bool
	        FORCE_DENSE: tl.constexpr,  # bool
	        ENABLE_LOGITS_CAPTURE: tl.constexpr,  # bool
	        ENABLE_LOGF: tl.constexpr,  # bool
	        WRITE_QUERY_NORMS: tl.constexpr,  # bool
	        PREFETCH_COMPACT: tl.constexpr,  # bool
        STORE_LOGITS_FOR_LOGF_LASTN1: tl.constexpr,  # bool
        STORE_LOGITS_FOR_LOGF_GT1_SCRATCH: tl.constexpr,  # bool
        LOGF_OUT_FP32: tl.constexpr,  # bool
):
    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    segm_idx = tl.program_id(2)

    seq_idx = find_seq_idx(query_start_len_ptr, q_block_global_idx, num_seqs,
                           BLOCK_Q, True)

    q_block_start_idx = tl.load(query_start_len_ptr +
                                seq_idx) // BLOCK_Q + seq_idx

    q_block_local_idx = q_block_global_idx - q_block_start_idx

    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)

    cur_batch_query_len = cur_batch_in_all_stop_index \
        - cur_batch_in_all_start_index

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)

    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + \
        offs_m % num_queries_per_kv

    query_offset = (query_offset_0[:, None] * query_stride_0 +
                    query_offset_1[:, None] * query_stride_1 + offs_d[None, :])

    dim_mask = tl.where(offs_d < HEAD_SIZE, 1, 0).to(tl.int1)
    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)

    Q = tl.load(
        query_ptr + query_offset,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
    )

    meta32_row_ptr = req_meta_i32_ptr + seq_idx * req_meta_i32_stride_row
    meta64_row_ptr = req_meta_i64_ptr + seq_idx * req_meta_i64_stride_row

    # per-request metadata - new compact layout:
    # req_meta_i32[7]: [kv_len_visible, compact_block_cnt, logits_last_n,
    #                  logits_row_offset, logits_capacity, flags, recent_len]
    # req_meta_i64[4]: [block_row_base_paged, compact_base_block, logits_base_ptr, token_row_base/query_norms_ptr]
    kv_len_visible = tl.load(meta32_row_ptr + 0 * req_meta_i32_stride_col).to(tl.int32)
    compact_block_cnt = tl.load(meta32_row_ptr + 1 * req_meta_i32_stride_col).to(tl.int32)
    logits_last_n = tl.load(meta32_row_ptr + 2 * req_meta_i32_stride_col).to(tl.int32)
    logits_row_offset = tl.load(meta32_row_ptr + 3 * req_meta_i32_stride_col).to(tl.int32)
    logits_capacity = tl.load(meta32_row_ptr + 4 * req_meta_i32_stride_col).to(tl.int32)
    flags = tl.load(meta32_row_ptr + 5 * req_meta_i32_stride_col)
    recent_len = tl.load(meta32_row_ptr + 6 * req_meta_i32_stride_col).to(tl.int32)

    block_row_base_paged = tl.load(meta64_row_ptr + 0 * req_meta_i64_stride_col).to(tl.int64)
    compact_base_block = tl.load(meta64_row_ptr + 1 * req_meta_i64_stride_col).to(tl.int64)
    logits_base_ptr = tl.load(meta64_row_ptr + 2 * req_meta_i64_stride_col).to(tl.int64)
    token_row_base = tl.load(meta64_row_ptr + 3 * req_meta_i64_stride_col).to(tl.int64)

    context_seq_len = tl.load(seqused_k_ptr + seq_idx).to(tl.int32)
    # decode fast-path: optionally derive kv_len/recent_len from seqused_k + recent_cap
    RECENT_CAP_FLAG = 4
    use_cap = (flags & RECENT_CAP_FLAG) != 0
    kv_len_visible = tl.where(use_cap, context_seq_len, kv_len_visible)
    logits_capacity_i64 = logits_capacity.to(tl.int64)

    if FORCE_DENSE:
        use_compact = tl.full([], 0, tl.int1)
        use_logits = tl.full([], 0, tl.int1)
        use_log_f = tl.full([], 0, tl.int1)
    elif COMPACT_ONLY:
        use_compact = 1
        use_logits = 0
        use_log_f = 0
    else:
        use_compact = (flags & 1) != 0
        use_logits = ((flags & 2) != 0) & (use_compact == 0)
        use_log_f = ((flags & 8) != 0) & (use_compact == 0)
        # 变体裁剪：编译期禁用时直接清零，避免后续指针/stride 计算占用寄存器。
        if tl.constexpr(not ENABLE_LOGITS_CAPTURE):
            use_logits = tl.full([], 0, tl.int1)
        if tl.constexpr(not ENABLE_LOGF):
            use_log_f = tl.full([], 0, tl.int1)
    seq_len = kv_len_visible
    context_delta = context_seq_len - cur_batch_query_len
    context_len = tl.where(context_delta > 0, context_delta, 0)
    recent_cap = tl.maximum(recent_len, 0)
    cap = tl.minimum(context_seq_len, recent_cap)
    cap = tl.maximum(cap, 0)
    recent_start_calc = ((context_seq_len - cap) // BLOCK_SIZE) * BLOCK_SIZE
    recent_start_calc = tl.maximum(recent_start_calc, 0)
    recent_len_calc = tl.maximum(context_seq_len - recent_start_calc, 0)
    recent_len_from_cap = tl.where(recent_cap > 0, recent_len_calc, 0)
    recent_len_from_len = tl.maximum(recent_len, 0)
    recent_len = tl.where(use_cap, recent_len_from_cap, recent_len_from_len)
    recent_block_cnt = cdiv_fn(recent_len, BLOCK_SIZE)
    query_mask = query_mask_0 & query_mask_1
    if USE_ALIBI_SLOPES:
        context_len_f32 = context_len.to(tl.float32)

    if tl.constexpr(ENABLE_LOGITS_CAPTURE or ENABLE_LOGF):
        logits_capacity_i64 = logits_capacity.to(tl.int64)
        logits_ptr = tl.cast(logits_dummy_ptr, tl.pointer_type(tl.float32))
        log_f_ptr = tl.cast(logits_dummy_ptr, tl.pointer_type(tl.float32 if LOGF_OUT_FP32 else tl.float16))
        logits_has_ptr = tl.full([], 0, tl.int1)
        log_f_has_ptr = tl.full([], 0, tl.int1)
        logits_stride_token = tl.full([], 0, tl.int64)
        logits_stride_head = tl.full([], 0, tl.int64)

        dummy_logits_addr = tl.cast(logits_dummy_ptr, tl.int64)
        logits_stride_token = tl.full([], 1, tl.int64)

        stride_logits = tl.full([], 0, tl.int64)
        stride_log_f = tl.full([], 0, tl.int64)
        if tl.constexpr(ENABLE_LOGITS_CAPTURE):
            logits_has_ptr = use_logits & (logits_base_ptr != 0) & (logits_last_n > 0) & (logits_capacity > 0)
            stride_logits = logits_last_n.to(tl.int64) * logits_capacity_i64
        if tl.constexpr(ENABLE_LOGF):
            log_f_has_ptr = use_log_f & (logits_base_ptr != 0) & (logits_capacity > 0)
            # log_f 的输出 buffer 允许使用“pad stride”（便于上游 step-wise 复用）。
            # 约定：dense log_f 场景下 req_meta_i32[1] 传 log_f_stride_head（>= logits_capacity）。
            stride_log_f = compact_block_cnt.to(tl.int64)
            stride_log_f = tl.where(stride_log_f > 0, stride_log_f, logits_capacity_i64)

        if tl.constexpr(ENABLE_LOGF and ENABLE_LOGITS_CAPTURE):
            logits_stride_head = tl.where(log_f_has_ptr, stride_log_f, stride_logits)
        elif tl.constexpr(ENABLE_LOGF):
            logits_stride_head = stride_log_f
        elif tl.constexpr(ENABLE_LOGITS_CAPTURE):
            logits_stride_head = stride_logits

        selected_logits_addr = tl.where((logits_has_ptr | log_f_has_ptr), logits_base_ptr, dummy_logits_addr)
        logits_ptr = tl.cast(selected_logits_addr, tl.pointer_type(tl.float32))
        log_f_ptr = tl.cast(selected_logits_addr, tl.pointer_type(tl.float32 if LOGF_OUT_FP32 else tl.float16))

        if tl.constexpr(ENABLE_LOGF):
            # last_n>1 log_f fastest-path: store per-row logits into a scratch buffer (per-layer, not accumulated across layers).
            logits_scratch_ptr = tl.cast(logits_dummy_ptr, tl.pointer_type(tl.float32))
            logits_scratch_has_ptr = tl.full([], 0, tl.int1)
            logits_scratch_stride_head = tl.full([], 0, tl.int64)
            logits_scratch_stride_token = tl.full([], 1, tl.int64)
            logits_scratch_has_ptr = use_log_f & STORE_LOGITS_FOR_LOGF_GT1_SCRATCH & (logits_last_n > 1) \
                & (use_compact == 0) & (compact_base_block != 0) & (logits_capacity > 0) & (compact_block_cnt > 0)
            logits_scratch_ptr = tl.cast(compact_base_block, tl.pointer_type(tl.float32))
            # scratch layout: [Hq, last_n, stride_log_f] fp32
            logits_scratch_stride_head = logits_last_n.to(tl.int64) * stride_log_f

    query_norms_ptr = tl.cast(logits_dummy_ptr, tl.pointer_type(tl.float32))
    query_norms_has_ptr = tl.full([], 0, tl.int1)
    if tl.constexpr(WRITE_QUERY_NORMS):
        dummy_logits_addr = tl.cast(logits_dummy_ptr, tl.int64)
        query_norms_has_ptr = use_logits & (token_row_base != 0) & (logits_last_n > 0)
        selected_query_norms_addr = tl.where(query_norms_has_ptr, token_row_base, dummy_logits_addr)
        query_norms_ptr = tl.cast(selected_query_norms_addr, tl.pointer_type(tl.float32))

    if tl.constexpr(WRITE_QUERY_NORMS):
        q_f = Q.to(tl.float32)
        norms = tl.sqrt(tl.sum(q_f * q_f, axis=1))
        norm_row_idx = query_pos - logits_row_offset
        start = query_norms_window - logits_last_n
        write_idx = start + norm_row_idx
        segm_mask = segm_idx == 0
        store_mask = query_norms_has_ptr & query_mask & (norm_row_idx >= 0) & (norm_row_idx < logits_last_n) & segm_mask
        head_offsets = query_offset_1.to(tl.int64) * query_norms_stride_head
        write_offsets = head_offsets + write_idx.to(tl.int64) * query_norms_stride_token
        tl.store(query_norms_ptr + write_offsets, norms, mask=store_mask)

    num_blocks = compact_block_cnt + recent_block_cnt if use_compact else cdiv_fn(seq_len, BLOCK_SIZE)
    blocks_per_segment = cdiv_fn(num_blocks, NUM_SEGMENTS_PER_SEQ)
    if segm_idx * blocks_per_segment >= num_blocks:
        # 当前 segment 不含任何 block：写入空输出，避免 reduce_segments 读取未初始化的 segm 缓冲。
        empty_acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)
        segm_output_offset = (
            query_offset_0[:, None].to(tl.int64) *
            (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED) +
            query_offset_1[:, None] * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED) +
            segm_idx * HEAD_SIZE_PADDED + tl.arange(0, HEAD_SIZE_PADDED)[None, :])
        tl.store(
            segm_output_ptr + segm_output_offset,
            empty_acc,
            mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        )
        segm_offset = (
            query_offset_0.to(tl.int64) *
            (num_query_heads * NUM_SEGMENTS_PER_SEQ) +
            query_offset_1 * NUM_SEGMENTS_PER_SEQ + segm_idx)
        tl.store(segm_max_ptr + segm_offset,
                 tl.full([BLOCK_M], float("-inf"), dtype=tl.float32),
                 mask=query_mask_0 & query_mask_1)
        tl.store(segm_expsum_ptr + segm_offset,
                 tl.zeros([BLOCK_M], dtype=tl.float32),
                 mask=query_mask_0 & query_mask_1)
        return

    start_blk = segm_idx * blocks_per_segment
    end_blk = min((segm_idx + 1) * blocks_per_segment, num_blocks)

    M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)

    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(alibi_slopes_ptr + query_offset_1,
                              mask=query_mask_1,
                              other=0.0,
                             )

    block_tables_ptr = block_tables_paged_ptr + block_row_base_paged * block_tables_paged_stride_row
    token_indices_ptr = token_positions_ptr + token_row_base
    token_head_offset = kv_head_idx * token_pos_stride_head
    token_indices_ptr_head = token_indices_ptr + token_head_offset
    recent_start = context_seq_len - recent_len
    tail_block_start = (context_seq_len - recent_len) // BLOCK_SIZE

    if COMPACT_ONLY:
        # === 循环分离 + tl.advance 优化 ===
        # 计算 compact 和 recent 的块范围
        compact_start = tl.minimum(compact_block_cnt, start_blk)
        compact_end = tl.minimum(compact_block_cnt, end_blk)
        recent_start_blk = tl.maximum(compact_block_cnt, start_blk)
        recent_end_blk = end_blk

        # int32 转换：block_ptr Mode A 需要 int32
        stride_v_compact_0_i32 = _rt_i32(stride_v_compact_0)
        stride_k_compact_0_i32 = _rt_i32(stride_k_compact_0)
        stride_v_compact_1_i32 = _rt_i32(stride_v_compact_1)
        stride_k_compact_1_i32 = _rt_i32(stride_k_compact_1)
        stride_v_compact_2_i32 = _rt_i32(stride_v_compact_2)
        stride_k_compact_2_i32 = _rt_i32(stride_k_compact_2)
        token_pos_stride_block_i32 = _rt_i32(token_pos_stride_block)

        tl.multiple_of(stride_v_compact_1_i32, 16)
        tl.multiple_of(stride_k_compact_1_i32, 16)

        head_v_compact_offset = kv_head_idx * stride_v_compact_2_i32
        head_k_compact_offset = kv_head_idx * stride_k_compact_2_i32
        head_v_cache_offset = kv_head_idx * stride_v_cache_2
        head_k_cache_offset = kv_head_idx * stride_k_cache_2
        context_upper = context_len + query_pos

        # === Compact 段：非尾块 ===
        non_tail_end = tl.maximum(compact_start, compact_end - 1)

        if tl.constexpr(PREFETCH_COMPACT):
            # === 预取路径：双缓冲优化 ===
            K_next = tl.zeros([HEAD_SIZE_PADDED, BLOCK_SIZE], dtype=Q.dtype)
            V_next = tl.zeros([BLOCK_SIZE, HEAD_SIZE_PADDED], dtype=Q.dtype)
            seq_next = tl.zeros([BLOCK_SIZE], dtype=tl.int32)

            # 预取第一个块
            if non_tail_end > compact_start:
                init_phys = compact_base_block + compact_start
                v_block_ptr0 = tl.make_block_ptr(
                    base=value_compact_ptr + (init_phys * stride_v_compact_0_i32 + head_v_compact_offset),
                    shape=(BLOCK_SIZE, HEAD_SIZE_PADDED),
                    strides=(stride_v_compact_1_i32, stride_v_compact_3),
                    offsets=(0, 0),
                    block_shape=(BLOCK_SIZE, HEAD_SIZE_PADDED),
                    order=(0, 1),
                )
                k_block_ptr0 = tl.make_block_ptr(
                    base=key_compact_ptr + (init_phys * stride_k_compact_0_i32 + head_k_compact_offset),
                    shape=(HEAD_SIZE_PADDED, BLOCK_SIZE),
                    strides=(stride_k_compact_3, stride_k_compact_1_i32),
                    offsets=(0, 0),
                    block_shape=(HEAD_SIZE_PADDED, BLOCK_SIZE),
                    order=(0, 1),
                )
                V_next = tl.load(v_block_ptr0, boundary_check=(0, 1), padding_option="zero")
                K_next = tl.load(k_block_ptr0, boundary_check=(0, 1), padding_option="zero")
                if USE_ALIBI_SLOPES:
                    token_pos_stride_token_i32 = _rt_i32(token_pos_stride_token)
                    token_block_ptr0 = tl.make_block_ptr(
                        base=token_indices_ptr_head + compact_start * token_pos_stride_block_i32,
                        shape=(BLOCK_SIZE,),
                        strides=(token_pos_stride_token_i32,),
                        offsets=(0,),
                        block_shape=(BLOCK_SIZE,),
                        order=(0,),
                    )
                    seq_next = tl.load(token_block_ptr0, boundary_check=(0,), padding_option="zero").to(tl.int32)

            # 非尾块循环
            for j in range(compact_start, non_tail_end):
                K_load = K_next
                V_load = V_next
                seq_offset = seq_next

                # 预取下一个块：用 tl.where 替代 runtime if 以规避 Triton 3.1.0 scf.if 编译器 bug
                # 最后一次迭代 nxt=j，重加载当前块（数据不使用）
                _prefetch = j + 1 < non_tail_end
                nxt = tl.where(_prefetch, j + 1, j)
                phys_nxt = compact_base_block + nxt
                v_block_ptr_nxt = tl.make_block_ptr(
                    base=value_compact_ptr + (phys_nxt * stride_v_compact_0_i32 + head_v_compact_offset),
                    shape=(BLOCK_SIZE, HEAD_SIZE_PADDED),
                    strides=(stride_v_compact_1_i32, stride_v_compact_3),
                    offsets=(0, 0),
                    block_shape=(BLOCK_SIZE, HEAD_SIZE_PADDED),
                    order=(0, 1),
                )
                k_block_ptr_nxt = tl.make_block_ptr(
                    base=key_compact_ptr + (phys_nxt * stride_k_compact_0_i32 + head_k_compact_offset),
                    shape=(HEAD_SIZE_PADDED, BLOCK_SIZE),
                    strides=(stride_k_compact_3, stride_k_compact_1_i32),
                    offsets=(0, 0),
                    block_shape=(HEAD_SIZE_PADDED, BLOCK_SIZE),
                    order=(0, 1),
                )
                V_next = tl.load(v_block_ptr_nxt, boundary_check=(0, 1), padding_option="zero")
                K_next = tl.load(k_block_ptr_nxt, boundary_check=(0, 1), padding_option="zero")
                if USE_ALIBI_SLOPES:
                    token_pos_stride_token_i32 = _rt_i32(token_pos_stride_token)
                    token_block_ptr_nxt = tl.make_block_ptr(
                        base=token_indices_ptr_head + nxt * token_pos_stride_block_i32,
                        shape=(BLOCK_SIZE,),
                        strides=(token_pos_stride_token_i32,),
                        offsets=(0,),
                        block_shape=(BLOCK_SIZE,),
                        order=(0,),
                    )
                    seq_next = tl.load(token_block_ptr_nxt, boundary_check=(0,), padding_option="zero").to(tl.int32)

                S = scale * tl.dot(Q, K_load)

                if USE_SOFTCAP:
                    S = apply_softcap(S, softcap)

                if USE_ALIBI_SLOPES:
                    S += alibi_slope[:, None] * (seq_offset.to(tl.float32) - context_len_f32)

                # 非尾块所有 token 都有效，只需 query_mask
                S = tl.where(query_mask[:, None], S, float("-inf"))

                # Online softmax 更新
                m_j = tl.maximum(M, tl.max(S, axis=1))
                m_j = tl.where(m_j > float("-inf"), m_j, 0.0)
                P = tl.exp(S - m_j[:, None])
                l_j = tl.sum(P, axis=1)
                alpha = tl.exp(M - m_j)
                acc = acc * alpha[:, None]
                L = L * alpha + l_j
                M = m_j
                acc += tl.dot(P.to(V_load.dtype), V_load)
        else:
            # === 非预取路径：tl.advance 优化 ===
            if non_tail_end > compact_start:
                # 初始化 block_ptr（Mode A: offsets 含动态值）
                init_phys = compact_base_block + compact_start
                v_blk_ptr = tl.make_block_ptr(
                    base=value_compact_ptr + (init_phys * stride_v_compact_0_i32 + head_v_compact_offset),
                    shape=(BLOCK_SIZE, HEAD_SIZE_PADDED),
                    strides=(stride_v_compact_1_i32, stride_v_compact_3),
                    offsets=(0, 0),
                    block_shape=(BLOCK_SIZE, HEAD_SIZE_PADDED),
                    order=(0, 1),
                )
                k_blk_ptr = tl.make_block_ptr(
                    base=key_compact_ptr + (init_phys * stride_k_compact_0_i32 + head_k_compact_offset),
                    shape=(HEAD_SIZE_PADDED, BLOCK_SIZE),
                    strides=(stride_k_compact_3, stride_k_compact_1_i32),
                    offsets=(0, 0),
                    block_shape=(HEAD_SIZE_PADDED, BLOCK_SIZE),
                    order=(0, 1),
                )

                for j in range(compact_start, non_tail_end):
                    # 非尾块：token 维不会越界，仅对 head 维做 boundary_check，
                    # 允许 tl.advance 跨 block 前进且不触发 token 维越界置零。
                    K_load = tl.load(k_blk_ptr, boundary_check=(0,), padding_option="zero")
                    V_load = tl.load(v_blk_ptr, boundary_check=(1,), padding_option="zero")

                    # Token positions 加载（只在 USE_ALIBI_SLOPES 时使用）
                    if USE_ALIBI_SLOPES:
                        token_pos_stride_token_i32 = _rt_i32(token_pos_stride_token)
                        token_block_ptr = tl.make_block_ptr(
                            base=token_indices_ptr_head + j * token_pos_stride_block_i32,
                            shape=(BLOCK_SIZE,),
                            strides=(token_pos_stride_token_i32,),
                            offsets=(0,),
                            block_shape=(BLOCK_SIZE,),
                            order=(0,),
                        )
                        seq_offset = tl.load(token_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.int32)

                    S = scale * tl.dot(Q, K_load)

                    if USE_SOFTCAP:
                        S = apply_softcap(S, softcap)

                    if USE_ALIBI_SLOPES:
                        S += alibi_slope[:, None] * (seq_offset.to(tl.float32) - context_len_f32)

                    # 非尾块所有 token 都有效，只需 query_mask
                    S = tl.where(query_mask[:, None], S, float("-inf"))

                    # Online softmax 更新
                    m_j = tl.maximum(M, tl.max(S, axis=1))
                    m_j = tl.where(m_j > float("-inf"), m_j, 0.0)
                    P = tl.exp(S - m_j[:, None])
                    l_j = tl.sum(P, axis=1)
                    alpha = tl.exp(M - m_j)
                    acc = acc * alpha[:, None]
                    L = L * alpha + l_j
                    M = m_j
                    acc += tl.dot(P.to(V_load.dtype), V_load)

                    # Advance block pointers
                    k_blk_ptr = tl.advance(k_blk_ptr, (0, BLOCK_SIZE))
                    v_blk_ptr = tl.advance(v_blk_ptr, (BLOCK_SIZE, 0))

        # === Compact 段：尾块（需要 valid_token_mask 检查）===
        if compact_end > compact_start:
            j_tail = compact_end - 1
            phys_tail = compact_base_block + j_tail
            v_block_ptr = tl.make_block_ptr(
                base=value_compact_ptr + (phys_tail * stride_v_compact_0_i32 + head_v_compact_offset),
                shape=(BLOCK_SIZE, HEAD_SIZE_PADDED),
                strides=(stride_v_compact_1_i32, stride_v_compact_3),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE, HEAD_SIZE_PADDED),
                order=(0, 1),
            )
            k_block_ptr = tl.make_block_ptr(
                base=key_compact_ptr + (phys_tail * stride_k_compact_0_i32 + head_k_compact_offset),
                shape=(HEAD_SIZE_PADDED, BLOCK_SIZE),
                strides=(stride_k_compact_3, stride_k_compact_1_i32),
                offsets=(0, 0),
                block_shape=(HEAD_SIZE_PADDED, BLOCK_SIZE),
                order=(0, 1),
            )
            K_load = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option="zero")
            V_load = tl.load(v_block_ptr, boundary_check=(0, 1), padding_option="zero")
            token_block_ptr = tl.make_block_ptr(
                base=token_indices_ptr_head + j_tail * token_pos_stride_block_i32,
                shape=(BLOCK_SIZE,),
                strides=(_rt_i32(token_pos_stride_token),),
                offsets=(0,),
                block_shape=(BLOCK_SIZE,),
                order=(0,),
            )
            seq_offset = tl.load(token_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.int32)
            valid_token_mask = seq_offset >= 0

            # 尾块需要完整的 mask 检查
            if SLIDING_WINDOW == 0:
                seq_mask = valid_token_mask[None, :] & (seq_offset[None, :] < seq_len) & (seq_offset[None, :] <= context_upper[:, None])
            else:
                seq_mask = valid_token_mask[None, :] & (seq_offset[None, :] < seq_len) & (seq_offset[None, :] <= context_upper[:, None])
                seq_mask = seq_mask & ((context_upper[:, None] - seq_offset) < SLIDING_WINDOW)

            S = scale * tl.dot(Q, K_load)

            if USE_SOFTCAP:
                S = apply_softcap(S, softcap)

            if USE_ALIBI_SLOPES:
                S += alibi_slope[:, None] * (seq_offset.to(tl.float32) - context_len_f32)

            S = tl.where(query_mask[:, None] & seq_mask, S, float("-inf"))

            # Online softmax 更新
            m_j = tl.maximum(M, tl.max(S, axis=1))
            m_j = tl.where(m_j > float("-inf"), m_j, 0.0)
            P = tl.exp(S - m_j[:, None])
            l_j = tl.sum(P, axis=1)
            alpha = tl.exp(M - m_j)
            acc = acc * alpha[:, None]
            L = L * alpha + l_j
            M = m_j
            acc += tl.dot(P.to(V_load.dtype), V_load)

        # === Recent 段：独立循环，使用 block_ptr Mode B ===
        offs_n = tl.arange(0, BLOCK_SIZE)
        for r_blk in range(recent_start_blk, recent_end_blk):
            seq_block_idx = tail_block_start + (r_blk - compact_block_cnt)
            block_id_raw = tl.load(block_tables_ptr + seq_block_idx).to(tl.int64)
            block_is_valid = block_id_raw >= 0
            block_id = tl.where(block_is_valid, block_id_raw, 0)
            v_block_ptr = tl.make_block_ptr(
                base=value_cache_ptr + (block_id * stride_v_cache_0 + head_v_cache_offset),
                shape=(BLOCK_SIZE, HEAD_SIZE_PADDED),
                strides=(stride_v_cache_1, stride_v_cache_3),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE, HEAD_SIZE_PADDED),
                order=(0, 1),
            )
            k_block_ptr = tl.make_block_ptr(
                base=key_cache_ptr + (block_id * stride_k_cache_0 + head_k_cache_offset),
                shape=(HEAD_SIZE_PADDED, BLOCK_SIZE),
                strides=(stride_k_cache_3, stride_k_cache_1),
                offsets=(0, 0),
                block_shape=(HEAD_SIZE_PADDED, BLOCK_SIZE),
                order=(0, 1),
            )
            V_load = tl.load(v_block_ptr, boundary_check=(0, 1), padding_option="zero")
            K_load = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option="zero")

            # 计算位置和 mask
            seq_offset = (seq_block_idx * BLOCK_SIZE + offs_n).to(tl.int32)
            if SLIDING_WINDOW == 0:
                seq_mask = block_is_valid & (seq_offset[None, :] >= recent_start) & (seq_offset[None, :] < seq_len) & (seq_offset[None, :] <= context_upper[:, None])
            else:
                seq_mask = block_is_valid & (seq_offset[None, :] >= recent_start) & (seq_offset[None, :] < seq_len) & (seq_offset[None, :] <= context_upper[:, None])
                seq_mask = seq_mask & ((context_upper[:, None] - seq_offset) < SLIDING_WINDOW)

            S = scale * tl.dot(Q, K_load)

            if USE_SOFTCAP:
                S = apply_softcap(S, softcap)

            if USE_ALIBI_SLOPES:
                S += alibi_slope[:, None] * (seq_offset.to(tl.float32) - context_len_f32)

            S = tl.where(query_mask[:, None] & seq_mask, S, float("-inf"))

            # Online softmax 更新
            m_j = tl.maximum(M, tl.max(S, axis=1))
            m_j = tl.where(m_j > float("-inf"), m_j, 0.0)
            P = tl.exp(S - m_j[:, None])
            l_j = tl.sum(P, axis=1)
            alpha = tl.exp(M - m_j)
            acc = acc * alpha[:, None]
            L = L * alpha + l_j
            M = m_j
            acc += tl.dot(P.to(V_load.dtype), V_load)
    else:
        for j in range(start_blk, end_blk):
            offs_n = tl.arange(0, BLOCK_SIZE)

            if use_compact:
                is_compact_blk = j < compact_block_cnt
                if is_compact_blk:
                    # compact 段：使用 block_ptr + boundary_check (模式 B)
                    physical_block_idx = compact_base_block + j
                    v_block_ptr = tl.make_block_ptr(
                        base=value_compact_ptr + (physical_block_idx * stride_v_compact_0 + kv_head_idx * stride_v_compact_2),
                        shape=(BLOCK_SIZE, HEAD_SIZE_PADDED),
                        strides=(stride_v_compact_1, stride_v_compact_3),
                        offsets=(0, 0),
                        block_shape=(BLOCK_SIZE, HEAD_SIZE_PADDED),
                        order=(0, 1),
                    )
                    k_block_ptr = tl.make_block_ptr(
                        base=key_compact_ptr + (physical_block_idx * stride_k_compact_0 + kv_head_idx * stride_k_compact_2),
                        shape=(HEAD_SIZE_PADDED, BLOCK_SIZE),
                        strides=(stride_k_compact_3, stride_k_compact_1),
                        offsets=(0, 0),
                        block_shape=(HEAD_SIZE_PADDED, BLOCK_SIZE),
                        order=(0, 1),
                    )
                    V_load = tl.load(v_block_ptr, boundary_check=(0, 1), padding_option="zero")
                    K_load = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option="zero")
                    # token_positions 加载使用 block_ptr (模式 B)
                    token_blk_ptr = tl.make_block_ptr(
                        base=token_indices_ptr_head + j * token_pos_stride_block,
                        shape=(BLOCK_SIZE,),
                        strides=(token_pos_stride_token,),
                        offsets=(0,),
                        block_shape=(BLOCK_SIZE,),
                        order=(0,),
                    )
                    seq_offset = tl.load(token_blk_ptr, boundary_check=(0,), padding_option="zero").to(tl.int32)
                    valid_token_mask = seq_offset >= 0
                else:
                    # recent 段：使用 block_ptr + boundary_check (模式 B)
                    r = j - compact_block_cnt
                    seq_block_idx = tail_block_start + r
                    block_id_raw = tl.load(block_tables_ptr + seq_block_idx).to(tl.int64)
                    block_is_valid = block_id_raw >= 0
                    block_id = tl.where(block_is_valid, block_id_raw, 0)
                    v_block_ptr = tl.make_block_ptr(
                        base=value_cache_ptr + (block_id * stride_v_cache_0 + kv_head_idx * stride_v_cache_2),
                        shape=(BLOCK_SIZE, HEAD_SIZE_PADDED),
                        strides=(stride_v_cache_1, stride_v_cache_3),
                        offsets=(0, 0),
                        block_shape=(BLOCK_SIZE, HEAD_SIZE_PADDED),
                        order=(0, 1),
                    )
                    k_block_ptr = tl.make_block_ptr(
                        base=key_cache_ptr + (block_id * stride_k_cache_0 + kv_head_idx * stride_k_cache_2),
                        shape=(HEAD_SIZE_PADDED, BLOCK_SIZE),
                        strides=(stride_k_cache_3, stride_k_cache_1),
                        offsets=(0, 0),
                        block_shape=(HEAD_SIZE_PADDED, BLOCK_SIZE),
                        order=(0, 1),
                    )
                    V_load = tl.load(v_block_ptr, boundary_check=(0, 1), padding_option="zero")
                    K_load = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option="zero")
                    # 对无效 block 置零
                    V_load = tl.where(block_is_valid, V_load, tl.zeros_like(V_load))
                    K_load = tl.where(block_is_valid, K_load, tl.zeros_like(K_load))
                    seq_offset = (seq_block_idx * BLOCK_SIZE + offs_n).to(tl.int32)
                    valid_token_mask = block_is_valid & (seq_offset >= recent_start)
            else:
                # dense 路径：优化 - 移除不必要的 block_is_valid 检查
                block_id = tl.load(block_tables_ptr + j).to(tl.int64)
                v_block_ptr = tl.make_block_ptr(
                    base=value_cache_ptr + (block_id * stride_v_cache_0 + kv_head_idx * stride_v_cache_2),
                    shape=(BLOCK_SIZE, HEAD_SIZE_PADDED),
                    strides=(stride_v_cache_1, stride_v_cache_3),
                    offsets=(0, 0),
                    block_shape=(BLOCK_SIZE, HEAD_SIZE_PADDED),
                    order=(0, 1),
                )
                k_block_ptr = tl.make_block_ptr(
                    base=key_cache_ptr + (block_id * stride_k_cache_0 + kv_head_idx * stride_k_cache_2),
                    shape=(HEAD_SIZE_PADDED, BLOCK_SIZE),
                    strides=(stride_k_cache_3, stride_k_cache_1),
                    offsets=(0, 0),
                    block_shape=(HEAD_SIZE_PADDED, BLOCK_SIZE),
                    order=(0, 1),
                )
                K_load = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option="zero")
                V_load = tl.load(v_block_ptr, boundary_check=(0, 1), padding_option="zero")
                seq_offset = (j * BLOCK_SIZE + offs_n).to(tl.int32)
                valid_token_mask = tl.full([BLOCK_SIZE], 1, dtype=tl.int1)

            if K_load.dtype.is_fp8():
                if Q.dtype.is_fp8():
                    K = K_load
                else:
                    K = (K_load.to(tl.float32) * tl.load(k_scale)).to(Q.dtype)
            else:
                K = K_load

            if V_load.dtype.is_fp8():
                if Q.dtype.is_fp8():
                    V = V_load
                else:
                    V = (V_load.to(tl.float32) * tl.load(v_scale)).to(Q.dtype)
            else:
                V = V_load

            seq_mask = valid_token_mask[None, :] & (seq_offset[None, :] < context_len + query_pos[:, None] + 1) & (seq_offset[None, :] < seq_len)
            seq_offset_f32 = seq_offset.to(tl.float32)

            S = tl.zeros(shape=(BLOCK_M, BLOCK_SIZE), dtype=tl.float32)
            S += scale * tl.dot(Q, K)

            if USE_SOFTCAP:
                S = apply_softcap(S, softcap)

            S = tl.where(query_mask[:, None] & seq_mask,
                         S, float("-inf"))

            if SLIDING_WINDOW > 0:
                S = tl.where((context_len + query_pos[:, None] - seq_offset)
                             < SLIDING_WINDOW, S, float("-inf"))

            if USE_ALIBI_SLOPES:
                S += alibi_slope[:, None] * (seq_offset_f32 - context_len_f32)

            if tl.constexpr(ENABLE_LOGITS_CAPTURE):
                if use_logits:
                    maybe_store_logits(
                        logits_ptr,
                        logits_has_ptr,
                        logits_row_offset,
                        logits_last_n,
                        logits_capacity,
                        logits_capacity_i64,
                        logits_stride_head,
                        logits_stride_token,
                        query_pos,
                        query_mask_0,
                        query_mask_1,
                        seq_offset,
                        seq_mask,
                        query_offset_1,
                        S,
                    )
            if tl.constexpr(ENABLE_LOGF):
                if use_log_f & STORE_LOGITS_FOR_LOGF_LASTN1 & (logits_last_n.to(tl.int32) == 1):
                    store_log_f_logits_lastn1(
                        log_f_ptr,
                        log_f_has_ptr,
                        LOGF_OUT_FP32,
                        flags,
                        logits_row_offset,
                        logits_capacity,
                        recent_len,
                        logits_stride_head,
                        logits_stride_token,
                        query_pos,
                        query_mask_0,
                        query_mask_1,
                        seq_offset,
                        seq_mask,
                        query_offset_1,
                        S,
                    )
                elif use_log_f & STORE_LOGITS_FOR_LOGF_GT1_SCRATCH & (logits_last_n.to(tl.int32) > 1):
                    store_log_f_logits_scratch_lastn_gt1(
                        logits_scratch_ptr,
                        logits_scratch_has_ptr,
                        flags,
                        logits_row_offset,
                        logits_last_n,
                        logits_stride_head,
                        logits_capacity,
                        recent_len,
                        logits_scratch_stride_head,
                        query_pos,
                        query_mask_0,
                        query_mask_1,
                        seq_offset,
                        seq_mask,
                        query_offset_1,
                        S,
                    )

            m_j = tl.maximum(M, tl.max(S, axis=1))
            m_j = tl.where(m_j > float("-inf"), m_j, 0.0)

            P = tl.exp(S - m_j[:, None])
            l_j = tl.sum(P, axis=1)
            alpha = tl.exp(M - m_j)
            acc = acc * alpha[:, None]
            L = L * alpha + l_j
            M = m_j
            acc += tl.dot(P.to(V.dtype), V)

    segm_output_offset = (
        query_offset_0[:, None].to(tl.int64) *
        (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED) +
        query_offset_1[:, None] * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED) +
        segm_idx * HEAD_SIZE_PADDED + tl.arange(0, HEAD_SIZE_PADDED)[None, :])
    tl.store(
        segm_output_ptr + segm_output_offset,
        acc,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
    )
    segm_offset = (query_offset_0.to(tl.int64) *
                   (num_query_heads * NUM_SEGMENTS_PER_SEQ) +
                   query_offset_1 * NUM_SEGMENTS_PER_SEQ + segm_idx)
    # 若该 segment 全部被 mask（L==0），将 M 置回 -inf，避免 reduce_segments 选到伪 max=0
    M_store = tl.where(L == 0.0, float("-inf"), M)
    tl.store(segm_max_ptr + segm_offset, M_store, mask=query_mask_0 & query_mask_1)
    tl.store(segm_expsum_ptr + segm_offset,
             L,
             mask=query_mask_0 & query_mask_1)

@triton.jit
def reduce_segments(
        output_ptr,  # [num_tokens, num_query_heads, head_size]
        segm_output_ptr,
        #[num_tokens, num_query_heads, max_num_segments, head_size]
        segm_max_ptr,  # [num_tokens, num_query_heads, max_num_segments]
        segm_expsum_ptr,  # [num_tokens, num_query_heads, max_num_segments]
        req_meta_i32_ptr,
        seqused_k_ptr,  # [num_seqs]
        num_seqs,  # int
        output_stride_0: tl.int64,  # int
        output_stride_1: tl.int64,  # int, should be equal to head_size
        req_meta_i32_stride_row: tl.int64,
        req_meta_i32_stride_col: tl.int64,
        query_start_len_ptr,  # [num_seqs+1]
        # ---- tl.constexpr meta parameters (keep at end for fast positional launch) ----
        num_query_heads: tl.constexpr,  # int
        BLOCK_SIZE: tl.constexpr,  # int
        HEAD_SIZE: tl.constexpr,  # int, must be power of 2
        HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
        BLOCK_Q: tl.constexpr,  # int
        NUM_SEGMENTS_PER_SEQ: tl.constexpr,  # int
):
    query_token_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)

    seq_idx = find_seq_idx(query_start_len_ptr, query_token_idx, num_seqs,
                           BLOCK_Q, False)

    # Use seqused_k as the source of truth for KV length. This prevents subtle
    # correctness issues when host-side meta updates are skipped in compact-only
    # decode fast paths.
    seq_len = tl.load(seqused_k_ptr + seq_idx).to(tl.int32)

    # fastpath: 单段直接归约，无需掩码/exp 组合
    if NUM_SEGMENTS_PER_SEQ == 1:
        dim_mask = tl.where(tl.arange(0, HEAD_SIZE_PADDED) < HEAD_SIZE, 1,
                            0).to(tl.int1)
        segm_offset = (query_token_idx.to(tl.int64) *
                       (num_query_heads * NUM_SEGMENTS_PER_SEQ) +
                       query_head_idx * NUM_SEGMENTS_PER_SEQ)
        segm_max = tl.load(segm_max_ptr + segm_offset)
        segm_expsum = tl.load(segm_expsum_ptr + segm_offset)
        segm_output_offset = (
            query_token_idx.to(tl.int64) *
            (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED) +
            query_head_idx * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED) +
            tl.arange(0, HEAD_SIZE_PADDED))
        segm_output = tl.load(
            segm_output_ptr + segm_output_offset,
            mask=dim_mask,
            other=0.0,
        )
        scale = tl.where(segm_expsum == 0.0, 0.0, 1.0 / segm_expsum)
        acc = segm_output * scale
        output_offset = (query_token_idx * output_stride_0 +
                         query_head_idx * output_stride_1 +
                         tl.arange(0, HEAD_SIZE_PADDED))
        tl.store(output_ptr + output_offset, acc, mask=dim_mask)
        return

    # number of segments for this particular sequence
    num_segments = NUM_SEGMENTS_PER_SEQ
    blocks_per_segment = cdiv_fn(seq_len, num_segments * BLOCK_SIZE)

    # create masks for subsequent loads
    act_num_segments = cdiv_fn(seq_len, blocks_per_segment * BLOCK_SIZE)
    segm_mask = tl.arange(0, NUM_SEGMENTS_PER_SEQ) < tl.full(
        [NUM_SEGMENTS_PER_SEQ], act_num_segments, dtype=tl.int32)
    dim_mask = tl.where(tl.arange(0, HEAD_SIZE_PADDED) < HEAD_SIZE, 1,
                        0).to(tl.int1)

    # load segment maxima
    segm_offset = (query_token_idx.to(tl.int64) *
                   (num_query_heads * NUM_SEGMENTS_PER_SEQ) +
                   query_head_idx * NUM_SEGMENTS_PER_SEQ +
                   tl.arange(0, NUM_SEGMENTS_PER_SEQ))
    segm_max = tl.load(segm_max_ptr + segm_offset,
                       mask=segm_mask,
                       other=float("-inf"))
    overall_max = tl.max(segm_max)

    # load and rescale segment exp sums
    segm_expsum = tl.load(segm_expsum_ptr + segm_offset,
                          mask=segm_mask,
                          other=0.0,
                             )
    segm_expsum = segm_expsum * tl.exp(segm_max - overall_max)
    overall_expsum = tl.sum(segm_expsum)

    # load, rescale, and add segment attention outputs
    segm_output_offset = (
        query_token_idx.to(tl.int64) *
        (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED) +
        query_head_idx * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED) +
        tl.arange(0, NUM_SEGMENTS_PER_SEQ)[:, None] * HEAD_SIZE_PADDED +
        tl.arange(0, HEAD_SIZE_PADDED)[None, :])
    segm_output = tl.load(
        segm_output_ptr + segm_output_offset,
        mask=segm_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )
    segm_output *= tl.exp(segm_max - overall_max)[:, None]
    acc_sum = tl.sum(segm_output, axis=0)
    # safely divide by overall_expsum, returning 0.0 if overall_expsum is 0
    acc = tl.where(overall_expsum == 0.0, 0.0, acc_sum / overall_expsum)

    # write result
    output_offset = (query_token_idx * output_stride_0 +
                     query_head_idx * output_stride_1 +
                     tl.arange(0, HEAD_SIZE_PADDED))
    tl.store(output_ptr + output_offset, acc, mask=dim_mask)

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1, num_stages=1),
        triton.Config({}, num_warps=1, num_stages=2),
        triton.Config({}, num_warps=1, num_stages=3),
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=2, num_stages=3),
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=2, num_stages=4),
        triton.Config({}, num_warps=4, num_stages=4),
        # 对长 KV/高并行度场景补充更大 warp 选项，由 autotune 选择最优
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=3),
        triton.Config({}, num_warps=8, num_stages=4),
    ],
    # 分桶 key：避免不同网格规模互相污染 autotune 结果
    key=["HEAD_SIZE_PADDED", "BLOCK_SIZE", "PREFETCH_COMPACT", "GRID_BIN", "NUM_SEQS_BIN"],
)
@triton.jit
def kernel_unified_attention_2d_compact_only(
        output_ptr,  # [num_tokens, num_query_heads, head_size]
        query_ptr,  # [num_tokens, num_query_heads, head_size]
        key_cache_ptr,  # [num_blks, blk_size, num_kv_heads, head_size]
        value_cache_ptr,  # [num_blks, blk_size, num_kv_heads, head_size]
        block_tables_paged_ptr,  # [num_seqs, max_blocks_paged]
        key_compact_ptr,  # [num_sparse_blks, blk_size, num_kv_heads, head_size]
        value_compact_ptr,  # [num_sparse_blks, blk_size, num_kv_heads, head_size]
        token_positions_ptr,  # [num_seqs, num_kv_heads, max_blocks, block_size]
        req_meta_i32_ptr,  # [num_seqs, 8]
        req_meta_i64_ptr,  # [num_seqs, 5]
        seqused_k_ptr,  # [num_seqs]
        alibi_slopes_ptr,  # [num_query_heads]
        scale,  # float32
        k_scale,  # float32
        v_scale,  # float32
        softcap,  # float32
        req_meta_i32_stride_row: tl.int64,
        req_meta_i32_stride_col: tl.int64,
        req_meta_i64_stride_row: tl.int64,
        req_meta_i64_stride_col: tl.int64,
        token_pos_stride_head: tl.int64,
        token_pos_stride_block: tl.int64,
        token_pos_stride_token: tl.int64,
        query_stride_0: tl.int64,  # int
        query_stride_1: tl.int64,  # int, should be equal to head_size
        output_stride_0: tl.int64,  # int
        output_stride_1: tl.int64,  # int, should be equal to head_size
        stride_k_cache_0: tl.int64,  # int
        stride_k_cache_1: tl.int64,  # int
        stride_k_cache_2: tl.int64,  # int
        stride_v_cache_0: tl.int64,  # int
        stride_v_cache_1: tl.int64,  # int
        stride_v_cache_2: tl.int64,  # int
        stride_k_compact_0: tl.int64,  # int
        stride_k_compact_1: tl.int64,  # int
        stride_k_compact_2: tl.int64,  # int
        stride_v_compact_0: tl.int64,  # int
        stride_v_compact_1: tl.int64,  # int
        stride_v_compact_2: tl.int64,  # int
        block_tables_paged_stride_row: tl.int64,
        query_start_len_ptr,  # [num_seqs+1]
        num_seqs: tl.int32,
        # ---- tl.constexpr meta parameters (keep at end for fast positional launch) ----
        num_query_heads: tl.constexpr,  # int
        num_queries_per_kv: tl.constexpr,  # int
        BLOCK_SIZE: tl.constexpr,  # int
        HEAD_SIZE: tl.constexpr,  # int
        HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
        USE_ALIBI_SLOPES: tl.constexpr,  # bool
        USE_SOFTCAP: tl.constexpr,  # bool
        SLIDING_WINDOW: tl.constexpr,  # int
        stride_k_cache_3: tl.constexpr,  # int
        stride_v_cache_3: tl.constexpr,  # int
        stride_k_compact_3: tl.constexpr,  # int
        stride_v_compact_3: tl.constexpr,  # int
        BLOCK_Q: tl.constexpr,  # int
        BLOCK_M: tl.constexpr,  # int
        PREFETCH_COMPACT: tl.constexpr,  # bool
        GRID_BIN: tl.constexpr,  # int, for autotune key separation of grid size
        NUM_SEQS_BIN: tl.constexpr,  # int, separate batch sizes
):
    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)

    seq_idx = find_seq_idx(query_start_len_ptr, q_block_global_idx, num_seqs,
                           BLOCK_Q, True)

    q_block_start_idx = tl.load(query_start_len_ptr +
                                seq_idx) // BLOCK_Q + seq_idx

    q_block_local_idx = q_block_global_idx - q_block_start_idx

    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)

    cur_batch_query_len = cur_batch_in_all_stop_index \
        - cur_batch_in_all_start_index

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_n = tl.arange(0, BLOCK_SIZE)

    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + \
        offs_m % num_queries_per_kv
    query_offset = (query_offset_0[:, None] * query_stride_0 +
                    query_offset_1[:, None] * query_stride_1 + offs_d[None, :])

    dim_mask = tl.where(offs_d < HEAD_SIZE, 1, 0).to(tl.int1)
    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)
    query_mask = query_mask_0 & query_mask_1

    Q = tl.load(
        query_ptr + query_offset,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
    )

    meta32_row_ptr = req_meta_i32_ptr + seq_idx * req_meta_i32_stride_row
    meta64_row_ptr = req_meta_i64_ptr + seq_idx * req_meta_i64_stride_row

    # per-request metadata - new compact layout:
    # req_meta_i32[7]: [kv_len_visible, compact_block_cnt, logits_last_n,
    #                  logits_row_offset, logits_capacity, flags, recent_len]
    # req_meta_i64[4]: [block_row_base_paged, compact_base_block, logits_base_ptr, token_row_base]
    # compact-only kernel 主要使用: kv_len_visible(0), compact_block_cnt(1), recent_len(6)
    # 以及：在 decode 且无需 token_positions（mask 仅用于尾部 padding）时，复用 req_meta_i32[2]
    # 传 compact_kv_len（不含 recent），以避免尾块读取 token_positions。
    # and: block_row_base_paged(0), compact_base_block(1), token_row_base(3)
    compact_block_cnt = tl.load(meta32_row_ptr + 1 * req_meta_i32_stride_col).to(tl.int32)
    compact_kv_len = tl.load(meta32_row_ptr + 2 * req_meta_i32_stride_col).to(tl.int32)
    flags = tl.load(meta32_row_ptr + 5 * req_meta_i32_stride_col).to(tl.int32)
    recent_meta = tl.load(meta32_row_ptr + 6 * req_meta_i32_stride_col).to(tl.int32)

    block_row_base_paged = tl.load(meta64_row_ptr + 0 * req_meta_i64_stride_col).to(tl.int64)
    compact_base_block = tl.load(meta64_row_ptr + 1 * req_meta_i64_stride_col).to(tl.int64)
    token_row_base = tl.load(meta64_row_ptr + 3 * req_meta_i64_stride_col).to(tl.int64)

    # 缩位宽：Triton 的 block_ptr 对 int64 偏移优化有限，尽量转为 int32 减少寄存器
    block_row_base_paged = block_row_base_paged.to(tl.int32)
    compact_base_block = compact_base_block.to(tl.int32)
    token_row_base = token_row_base.to(tl.int32)

    context_seq_len = tl.load(seqused_k_ptr + seq_idx).to(tl.int32)

    # compact-only decode：kv_len_visible 与 seqused_k 对齐，避免每步更新 meta[0]
    seq_len = context_seq_len
    context_delta = context_seq_len - cur_batch_query_len
    context_len = tl.where(context_delta > 0, context_delta, 0)
    context_upper = context_len + query_pos

    # int32 转换：仅 block_ptr 需要的 stride 转换为 int32
    # compact KV block_ptr 需要 int32（offsets 非常量）
    stride_v_compact_0_i32 = _rt_i32(stride_v_compact_0)
    stride_k_compact_0_i32 = _rt_i32(stride_k_compact_0)
    stride_v_compact_1_i32 = _rt_i32(stride_v_compact_1)
    stride_k_compact_1_i32 = _rt_i32(stride_k_compact_1)
    stride_v_compact_2_i32 = _rt_i32(stride_v_compact_2)
    stride_k_compact_2_i32 = _rt_i32(stride_k_compact_2)
    # cache strides 保持 int64（仅用于 pointer arithmetic，非 block_ptr）

    tl.multiple_of(stride_v_compact_1_i32, 16)
    tl.multiple_of(stride_k_compact_1_i32, 16)

    # compact head offset（int32）：prefetch / non-prefetch 共用
    head_v_offset = kv_head_idx * stride_v_compact_2_i32
    head_k_offset = kv_head_idx * stride_k_compact_2_i32

    # recent_len 两种表达：
    # - 默认：meta[6] 为 recent_len（旧路径/兼容路径）
    # - fast-path：meta[6] 为 recent_cap，且 flags bit2=1，kernel 内计算 recent_len（避免每步更新 meta）
    # - FULL_CONTEXT_FLAG (bit4=16)：is_compact=0 行使用 context_seq_len 作为 recent_len，
    #   让 compact_only kernel 对该行遍历所有 paged blocks。seqused_k 每步实时读取，不会过期。
    RECENT_CAP_FLAG = 4
    FULL_CONTEXT_FLAG = 16
    use_cap = (flags & RECENT_CAP_FLAG) != 0
    use_full = (flags & FULL_CONTEXT_FLAG) != 0
    recent_cap = tl.maximum(recent_meta, 0)
    cap = tl.minimum(context_seq_len, recent_cap)
    cap = tl.maximum(cap, 0)
    # BLOCK_SIZE 为 2^n（vLLM 默认 16），用按位对齐避免 int 除法的类型/代价问题。
    align_mask = tl.full([], -BLOCK_SIZE, tl.int32)
    recent_start_calc = (context_seq_len - cap) & align_mask
    recent_start_calc = tl.maximum(recent_start_calc, 0)
    recent_len_calc = tl.maximum(context_seq_len - recent_start_calc, 0)
    recent_len_from_cap = tl.where(recent_cap > 0, recent_len_calc, 0)
    recent_len_from_len = tl.maximum(recent_meta, 0)
    recent_len = tl.where(use_full, context_seq_len,
                 tl.where(use_cap, recent_len_from_cap, recent_len_from_len))
    recent_block_cnt = cdiv_fn(recent_len, BLOCK_SIZE)
    recent_start = context_seq_len - recent_len

    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(alibi_slopes_ptr + query_offset_1,
                              mask=query_mask_1,
                              other=0.0,
                             )
        context_len_f32 = context_len.to(tl.float32)

    M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)
    block_tables_ptr = block_tables_paged_ptr + block_row_base_paged * block_tables_paged_stride_row
    tail_block_start = (context_seq_len - recent_len) // BLOCK_SIZE

    if tl.constexpr(PREFETCH_COMPACT):
        # 预取路径：尾块分离优化
        # 计算非尾块结束位置（尾块可能有无效 token）
        non_tail_end = tl.maximum(0, compact_block_cnt - 1)
        if ((not USE_ALIBI_SLOPES) and (SLIDING_WINDOW == 0)) and (compact_kv_len > 0):
            # 无 padding 时把最后一块并入非尾块循环，避免额外尾块开销
            if compact_kv_len == (compact_block_cnt * BLOCK_SIZE):
                non_tail_end = compact_block_cnt

        # 预取第一个 compact 块
        K_next = tl.zeros([HEAD_SIZE_PADDED, BLOCK_SIZE], dtype=Q.dtype)
        V_next = tl.zeros([BLOCK_SIZE, HEAD_SIZE_PADDED], dtype=Q.dtype)
        seq_next = tl.zeros([BLOCK_SIZE], dtype=tl.int32)  # 仅 ALIBI 时使用

        # 预取：循环外构建 block_ptr，循环内 tl.advance 获取下一块。
        # Triton 需要 ptr 在所有控制流下定义，因此这里无条件创建。
        init_phys = compact_base_block
        v_blk_ptr = tl.make_block_ptr(
            base=value_compact_ptr + (init_phys * stride_v_compact_0_i32 + head_v_offset),
            shape=(BLOCK_SIZE, HEAD_SIZE_PADDED),
            strides=(stride_v_compact_1_i32, stride_v_compact_3),
            offsets=(0, 0),
            block_shape=(BLOCK_SIZE, HEAD_SIZE_PADDED),
            order=(0, 1),
        )
        k_blk_ptr = tl.make_block_ptr(
            base=key_compact_ptr + (init_phys * stride_k_compact_0_i32 + head_k_offset),
            shape=(HEAD_SIZE_PADDED, BLOCK_SIZE),
            strides=(stride_k_compact_3, stride_k_compact_1_i32),
            offsets=(0, 0),
            block_shape=(HEAD_SIZE_PADDED, BLOCK_SIZE),
            order=(0, 1),
        )

        if USE_ALIBI_SLOPES or SLIDING_WINDOW > 0:
            # token_positions 用于 ALIBI/滑窗：仅在需要时计算相关指针/stride，减少寄存器占用
            token_pos_stride_block_i32 = _rt_i32(token_pos_stride_block)
            token_pos_stride_token_i32 = _rt_i32(token_pos_stride_token)
            token_indices_ptr = token_positions_ptr + token_row_base
            token_head_offset = kv_head_idx * token_pos_stride_head  # int64
            token_indices_ptr_head = token_indices_ptr + token_head_offset

        if non_tail_end > 0:
            # 非尾块 token 维度必然在界内，只对 head 维做 boundary_check，减少 mask 开销
            if tl.constexpr(HEAD_SIZE_PADDED == HEAD_SIZE):
                V_next = tl.load(v_blk_ptr)
                K_next = tl.load(k_blk_ptr)
            else:
                V_next = tl.load(v_blk_ptr, boundary_check=(1,), padding_option="zero")
                K_next = tl.load(k_blk_ptr, boundary_check=(0,), padding_option="zero")
            # 条件加载 token_pos：ALIBI 或滑窗时需要
            if USE_ALIBI_SLOPES or SLIDING_WINDOW > 0:
                token_block_ptr0 = tl.make_block_ptr(
                    base=token_indices_ptr_head,
                    shape=(BLOCK_SIZE,),
                    strides=(token_pos_stride_token_i32,),
                    offsets=(0,),
                    block_shape=(BLOCK_SIZE,),
                    order=(0,),
                )
                seq_next = tl.load(token_block_ptr0, boundary_check=(0,), padding_option="zero").to(tl.int32)

        # === 非尾块循环：省去 valid_token_mask 检查 ===
        for j in range(0, non_tail_end):
            K_load = K_next
            V_load = V_next
            seq_offset = seq_next

            # 预取下一个块：用 tl.where 替代 runtime if 以规避 Triton 3.1.0 scf.if 编译器 bug
            # (MLIR pass 错误销毁 scf.if op 导致 SIGABRT)
            # 最后一次迭代 step=0，advance 为 no-op，重加载当前块（数据不使用）
            _prefetch = j + 1 < non_tail_end
            step = tl.where(_prefetch, BLOCK_SIZE, 0)
            v_blk_ptr = tl.advance(v_blk_ptr, (step, 0))
            k_blk_ptr = tl.advance(k_blk_ptr, (0, step))
            if tl.constexpr(HEAD_SIZE_PADDED == HEAD_SIZE):
                V_next = tl.load(v_blk_ptr)
                K_next = tl.load(k_blk_ptr)
            else:
                V_next = tl.load(v_blk_ptr, boundary_check=(1,), padding_option="zero")
                K_next = tl.load(k_blk_ptr, boundary_check=(0,), padding_option="zero")
            # 条件加载 token_pos：ALIBI 或滑窗时需要
            if USE_ALIBI_SLOPES or SLIDING_WINDOW > 0:
                next_j = tl.where(_prefetch, j + 1, j)
                token_block_ptr = tl.make_block_ptr(
                    base=token_indices_ptr_head + next_j * token_pos_stride_block_i32,
                    shape=(BLOCK_SIZE,),
                    strides=(token_pos_stride_token_i32,),
                    offsets=(0,),
                    block_shape=(BLOCK_SIZE,),
                    order=(0,),
                )
                seq_next = tl.load(token_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.int32)

            if K_load.dtype.is_fp8():
                if Q.dtype.is_fp8():
                    K = K_load
                else:
                    K = (K_load.to(tl.float32) * tl.load(k_scale)).to(Q.dtype)
            else:
                K = K_load

            if V_load.dtype.is_fp8():
                if Q.dtype.is_fp8():
                    V = V_load
                else:
                    V = (V_load.to(tl.float32) * tl.load(v_scale)).to(Q.dtype)
            else:
                V = V_load

            # 直接计算 attention score
            S = scale * tl.dot(Q, K)

            if USE_SOFTCAP:
                S = apply_softcap(S, softcap)

            if USE_ALIBI_SLOPES:
                S += alibi_slope[:, None] * (seq_offset.to(tl.float32) - context_len_f32)

            # 非尾块：滑窗开启时需要基于 seq_offset 额外遮罩
            if tl.constexpr(SLIDING_WINDOW > 0):
                window_mask = (context_upper[:, None] - seq_offset[None, :]) < SLIDING_WINDOW
                S = tl.where(query_mask[:, None] & window_mask, S, float("-inf"))
            else:
                S = tl.where(query_mask[:, None], S, float("-inf"))

            m_j = tl.maximum(M, tl.max(S, axis=1))
            m_j = tl.where(m_j > float("-inf"), m_j, 0.0)

            P = tl.exp(S - m_j[:, None])
            l_j = tl.sum(P, axis=1)
            alpha = tl.exp(M - m_j)
            acc = acc * alpha[:, None]
            L = L * alpha + l_j
            M = m_j
            acc += tl.dot(P.to(V.dtype), V)

        # === 尾块处理：仅在存在 padding / 需要 token_positions 时才处理 ===
        if compact_block_cnt > 0:
            skip_tail = False
            if ((not USE_ALIBI_SLOPES) and (SLIDING_WINDOW == 0)) and (compact_kv_len > 0):
                skip_tail = compact_kv_len == (compact_block_cnt * BLOCK_SIZE)

            if skip_tail:
                pass
            else:
                j_tail = compact_block_cnt - 1
                phys_tail = compact_base_block + j_tail
                v_block_ptr_tail = tl.make_block_ptr(
                    base=value_compact_ptr + (phys_tail * stride_v_compact_0_i32 + head_v_offset),
                    shape=(BLOCK_SIZE, HEAD_SIZE_PADDED),
                    strides=(stride_v_compact_1_i32, stride_v_compact_3),
                    offsets=(0, 0),
                    block_shape=(BLOCK_SIZE, HEAD_SIZE_PADDED),
                    order=(0, 1),
                )
                k_block_ptr_tail = tl.make_block_ptr(
                    base=key_compact_ptr + (phys_tail * stride_k_compact_0_i32 + head_k_offset),
                    shape=(HEAD_SIZE_PADDED, BLOCK_SIZE),
                    strides=(stride_k_compact_3, stride_k_compact_1_i32),
                    offsets=(0, 0),
                    block_shape=(HEAD_SIZE_PADDED, BLOCK_SIZE),
                    order=(0, 1),
                )
                if tl.constexpr(HEAD_SIZE_PADDED == HEAD_SIZE):
                    V_load = tl.load(v_block_ptr_tail)
                    K_load = tl.load(k_block_ptr_tail)
                else:
                    V_load = tl.load(v_block_ptr_tail, boundary_check=(0, 1), padding_option="zero")
                    K_load = tl.load(k_block_ptr_tail, boundary_check=(0, 1), padding_option="zero")

                if ((not USE_ALIBI_SLOPES) and (SLIDING_WINDOW == 0)) and (compact_kv_len > 0):
                    # 无 alibi/滑窗：仅需屏蔽 padding，无需读取 token_positions
                    tail_valid = compact_kv_len - j_tail * BLOCK_SIZE
                    valid_token_mask = offs_n < tail_valid
                    # 保持与 else 分支一致的 mask 形状：[BLOCK_M, BLOCK_SIZE]
                    seq_mask = valid_token_mask[None, :] & tl.full([BLOCK_M, 1], 1, dtype=tl.int1)
                else:
                    # 需要 token_positions（ALIBI/滑窗或 meta 未提供 compact_kv_len）
                    token_pos_stride_block_i32 = _rt_i32(token_pos_stride_block)
                    token_pos_stride_token_i32 = _rt_i32(token_pos_stride_token)
                    token_indices_ptr = token_positions_ptr + token_row_base
                    token_head_offset = kv_head_idx * token_pos_stride_head  # int64
                    token_indices_ptr_head = token_indices_ptr + token_head_offset
                    token_block_ptr = tl.make_block_ptr(
                        base=token_indices_ptr_head + j_tail * token_pos_stride_block_i32,
                        shape=(BLOCK_SIZE,),
                        strides=(token_pos_stride_token_i32,),
                        offsets=(0,),
                        block_shape=(BLOCK_SIZE,),
                        order=(0,),
                    )
                    seq_offset = tl.load(token_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.int32)
                    valid_token_mask = seq_offset >= 0

                    # 尾块：需要完整的 valid_token_mask + 位置约束检查
                    seq_mask = valid_token_mask[None, :] & (seq_offset[None, :] < seq_len) & (seq_offset[None, :] <= context_upper[:, None])
                    if tl.constexpr(SLIDING_WINDOW > 0):
                        seq_mask = seq_mask & ((context_len + query_pos[:, None] - seq_offset) < SLIDING_WINDOW)
                    if USE_ALIBI_SLOPES:
                        seq_offset_f32 = seq_offset.to(tl.float32)

                if K_load.dtype.is_fp8():
                    if Q.dtype.is_fp8():
                        K = K_load
                    else:
                        K = (K_load.to(tl.float32) * tl.load(k_scale)).to(Q.dtype)
                else:
                    K = K_load

                if V_load.dtype.is_fp8():
                    if Q.dtype.is_fp8():
                        V = V_load
                    else:
                        V = (V_load.to(tl.float32) * tl.load(v_scale)).to(Q.dtype)
                else:
                    V = V_load

                S = scale * tl.dot(Q, K)

                if USE_SOFTCAP:
                    S = apply_softcap(S, softcap)

                S = tl.where(query_mask[:, None] & seq_mask,
                             S, float("-inf"))

                if USE_ALIBI_SLOPES:
                    S += alibi_slope[:, None] * (seq_offset_f32 - context_len_f32)

                m_j = tl.maximum(M, tl.max(S, axis=1))
                m_j = tl.where(m_j > float("-inf"), m_j, 0.0)

                P = tl.exp(S - m_j[:, None])
                l_j = tl.sum(P, axis=1)
                alpha = tl.exp(M - m_j)
                acc = acc * alpha[:, None]
                L = L * alpha + l_j
                M = m_j
                acc += tl.dot(P.to(V.dtype), V)
    else:
        # 非预取路径：尾块分离 + tl.advance 复用 block_ptr + 条件加载 token_pos
        # 计算非尾块结束位置（尾块可能有无效 token）
        non_tail_end = tl.maximum(0, compact_block_cnt - 1)
        if ((not USE_ALIBI_SLOPES) and (SLIDING_WINDOW == 0)) and (compact_kv_len > 0):
            # 无 padding 时把最后一块并入非尾块循环，避免额外尾块开销
            if compact_kv_len == (compact_block_cnt * BLOCK_SIZE):
                non_tail_end = compact_block_cnt

        # === 非尾块循环：tl.advance 复用 block_ptr ===
        # NOTE: block_ptr 的 offsets 是按维度 stride 解释的，不能混入 head/phys 的 raw element 偏移。
        # 因此把 head/phys 偏移放在 base 上，advance 用 BLOCK_SIZE（= stride0/stride1）。
        if non_tail_end > 0:
            init_phys = compact_base_block
            v_blk_ptr = tl.make_block_ptr(
                base=value_compact_ptr + (init_phys * stride_v_compact_0_i32 + head_v_offset),
                shape=(BLOCK_SIZE, HEAD_SIZE_PADDED),
                strides=(stride_v_compact_1_i32, stride_v_compact_3),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE, HEAD_SIZE_PADDED),
                order=(0, 1),
            )
            k_blk_ptr = tl.make_block_ptr(
                base=key_compact_ptr + (init_phys * stride_k_compact_0_i32 + head_k_offset),
                shape=(HEAD_SIZE_PADDED, BLOCK_SIZE),
                strides=(stride_k_compact_3, stride_k_compact_1_i32),
                offsets=(0, 0),
                block_shape=(HEAD_SIZE_PADDED, BLOCK_SIZE),
                order=(0, 1),
            )

            if USE_ALIBI_SLOPES or SLIDING_WINDOW > 0:
                # token_positions 用于 ALIBI/滑窗：仅在需要时计算相关指针/stride，减少寄存器占用
                token_pos_stride_block_i32 = _rt_i32(token_pos_stride_block)
                token_pos_stride_token_i32 = _rt_i32(token_pos_stride_token)
                token_indices_ptr = token_positions_ptr + token_row_base
                token_head_offset = kv_head_idx * token_pos_stride_head  # int64
                token_indices_ptr_head = token_indices_ptr + token_head_offset

            # 单块循环：避免手工 unroll 带来的寄存器压力/占用下降
            for j in range(0, non_tail_end):
                # --- block j ---
                if tl.constexpr(HEAD_SIZE_PADDED == HEAD_SIZE):
                    K_load = tl.load(k_blk_ptr)
                    V_load = tl.load(v_blk_ptr)
                else:
                    K_load = tl.load(k_blk_ptr, boundary_check=(0,), padding_option="zero")
                    V_load = tl.load(v_blk_ptr, boundary_check=(1,), padding_option="zero")

                if USE_ALIBI_SLOPES or SLIDING_WINDOW > 0:
                    token_block_ptr = tl.make_block_ptr(
                        base=token_indices_ptr_head + j * token_pos_stride_block_i32,
                        shape=(BLOCK_SIZE,),
                        strides=(token_pos_stride_token_i32,),
                        offsets=(0,),
                        block_shape=(BLOCK_SIZE,),
                        order=(0,),
                    )
                    seq_offset = tl.load(token_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.int32)

                if K_load.dtype.is_fp8():
                    if Q.dtype.is_fp8():
                        K = K_load
                    else:
                        K = (K_load.to(tl.float32) * tl.load(k_scale)).to(Q.dtype)
                else:
                    K = K_load

                if V_load.dtype.is_fp8():
                    if Q.dtype.is_fp8():
                        V = V_load
                    else:
                        V = (V_load.to(tl.float32) * tl.load(v_scale)).to(Q.dtype)
                else:
                    V = V_load

                S = scale * tl.dot(Q, K)
                if USE_SOFTCAP:
                    S = apply_softcap(S, softcap)
                if USE_ALIBI_SLOPES:
                    S += alibi_slope[:, None] * (seq_offset.to(tl.float32) - context_len_f32)

                if tl.constexpr(SLIDING_WINDOW > 0):
                    window_mask = (context_upper[:, None] - seq_offset[None, :]) < SLIDING_WINDOW
                    S = tl.where(query_mask[:, None] & window_mask, S, float("-inf"))
                else:
                    S = tl.where(query_mask[:, None], S, float("-inf"))

                m_new = tl.maximum(M, tl.max(S, axis=1))
                m_new = tl.where(m_new > float("-inf"), m_new, 0.0)
                alpha = tl.exp(M - m_new)
                P = tl.exp(S - m_new[:, None])
                L = L * alpha + tl.sum(P, axis=1)
                acc = acc * alpha[:, None] + tl.dot(P.to(V.dtype), V)
                M = m_new

                v_blk_ptr = tl.advance(v_blk_ptr, (BLOCK_SIZE, 0))
                k_blk_ptr = tl.advance(k_blk_ptr, (0, BLOCK_SIZE))

        # === 尾块处理：仅在存在 padding / 需要 token_positions 时才处理 ===
        if compact_block_cnt > 0:
            skip_tail = False
            if ((not USE_ALIBI_SLOPES) and (SLIDING_WINDOW == 0)) and (compact_kv_len > 0):
                skip_tail = compact_kv_len == (compact_block_cnt * BLOCK_SIZE)

            if skip_tail:
                pass
            else:
                j_tail = compact_block_cnt - 1
                phys_tail = compact_base_block + j_tail
                v_block_ptr_tail = tl.make_block_ptr(
                    base=value_compact_ptr + (phys_tail * stride_v_compact_0_i32 + head_v_offset),
                    shape=(BLOCK_SIZE, HEAD_SIZE_PADDED),
                    strides=(stride_v_compact_1_i32, stride_v_compact_3),
                    offsets=(0, 0),
                    block_shape=(BLOCK_SIZE, HEAD_SIZE_PADDED),
                    order=(0, 1),
                )
                k_block_ptr_tail = tl.make_block_ptr(
                    base=key_compact_ptr + (phys_tail * stride_k_compact_0_i32 + head_k_offset),
                    shape=(HEAD_SIZE_PADDED, BLOCK_SIZE),
                    strides=(stride_k_compact_3, stride_k_compact_1_i32),
                    offsets=(0, 0),
                    block_shape=(HEAD_SIZE_PADDED, BLOCK_SIZE),
                    order=(0, 1),
                )
                V_load = tl.load(v_block_ptr_tail, boundary_check=(0, 1), padding_option="zero")
                K_load = tl.load(k_block_ptr_tail, boundary_check=(0, 1), padding_option="zero")

                if ((not USE_ALIBI_SLOPES) and (SLIDING_WINDOW == 0)) and (compact_kv_len > 0):
                    tail_valid = compact_kv_len - j_tail * BLOCK_SIZE
                    valid_token_mask = offs_n < tail_valid
                    # 保持与 else 分支一致的 mask 形状：[BLOCK_M, BLOCK_SIZE]
                    seq_mask = valid_token_mask[None, :] & tl.full([BLOCK_M, 1], 1, dtype=tl.int1)
                else:
                    # 尾块必需 token_positions 以过滤无效 token
                    token_pos_stride_block_i32 = _rt_i32(token_pos_stride_block)
                    token_pos_stride_token_i32 = _rt_i32(token_pos_stride_token)
                    token_indices_ptr = token_positions_ptr + token_row_base
                    token_head_offset = kv_head_idx * token_pos_stride_head  # int64
                    token_indices_ptr_head = token_indices_ptr + token_head_offset
                    token_block_ptr = tl.make_block_ptr(
                        base=token_indices_ptr_head + j_tail * token_pos_stride_block_i32,
                        shape=(BLOCK_SIZE,),
                        strides=(token_pos_stride_token_i32,),
                        offsets=(0,),
                        block_shape=(BLOCK_SIZE,),
                        order=(0,),
                    )
                    seq_offset = tl.load(token_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.int32)
                    valid_token_mask = seq_offset >= 0

                    # 尾块：需要完整的 valid_token_mask 检查
                    seq_mask = valid_token_mask[None, :] & (seq_offset[None, :] < seq_len) & (seq_offset[None, :] <= context_upper[:, None])
                    if tl.constexpr(SLIDING_WINDOW > 0):
                        seq_mask = seq_mask & ((context_len + query_pos[:, None] - seq_offset) < SLIDING_WINDOW)
                    if USE_ALIBI_SLOPES:
                        seq_offset_f32 = seq_offset.to(tl.float32)

                if K_load.dtype.is_fp8():
                    if Q.dtype.is_fp8():
                        K = K_load
                    else:
                        K = (K_load.to(tl.float32) * tl.load(k_scale)).to(Q.dtype)
                else:
                    K = K_load

                if V_load.dtype.is_fp8():
                    if Q.dtype.is_fp8():
                        V = V_load
                    else:
                        V = (V_load.to(tl.float32) * tl.load(v_scale)).to(Q.dtype)
                else:
                    V = V_load

                S = scale * tl.dot(Q, K)

                if USE_SOFTCAP:
                    S = apply_softcap(S, softcap)

                S = tl.where(query_mask[:, None] & seq_mask,
                             S, float("-inf"))

                if USE_ALIBI_SLOPES:
                    S += alibi_slope[:, None] * (seq_offset_f32 - context_len_f32)

                m_j = tl.maximum(M, tl.max(S, axis=1))
                m_j = tl.where(m_j > float("-inf"), m_j, 0.0)

                P = tl.exp(S - m_j[:, None])
                l_j = tl.sum(P, axis=1)
                alpha = tl.exp(M - m_j)
                acc = acc * alpha[:, None]
                L = L * alpha + l_j
                M = m_j
                acc += tl.dot(P.to(V.dtype), V)

    # recent 段（paged 尾部）：使用 block_ptr 优化 (模式 B: offsets=0,0)
    if recent_block_cnt > 0:
        head_v_cache_offset = kv_head_idx * stride_v_cache_2
        head_k_cache_offset = kv_head_idx * stride_k_cache_2

        for r in range(0, recent_block_cnt):
            seq_block_idx = tail_block_start + r
            block_id_raw = tl.load(block_tables_ptr + seq_block_idx).to(tl.int64)
            block_is_valid = block_id_raw >= 0
            block_id = tl.where(block_is_valid, block_id_raw, 0)
            v_block_ptr = tl.make_block_ptr(
                base=value_cache_ptr + (block_id * stride_v_cache_0 + head_v_cache_offset),
                shape=(BLOCK_SIZE, HEAD_SIZE_PADDED),
                strides=(stride_v_cache_1, stride_v_cache_3),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE, HEAD_SIZE_PADDED),
                order=(0, 1),
            )
            k_block_ptr = tl.make_block_ptr(
                base=key_cache_ptr + (block_id * stride_k_cache_0 + head_k_cache_offset),
                shape=(HEAD_SIZE_PADDED, BLOCK_SIZE),
                strides=(stride_k_cache_3, stride_k_cache_1),
                offsets=(0, 0),
                block_shape=(HEAD_SIZE_PADDED, BLOCK_SIZE),
                order=(0, 1),
            )
            if tl.constexpr(HEAD_SIZE_PADDED == HEAD_SIZE):
                V_load = tl.load(v_block_ptr)
                K_load = tl.load(k_block_ptr)
            else:
                V_load = tl.load(v_block_ptr, boundary_check=(0, 1), padding_option="zero")
                K_load = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option="zero")
            # 对无效 block 置零
            V_load = tl.where(block_is_valid, V_load, tl.zeros_like(V_load))
            K_load = tl.where(block_is_valid, K_load, tl.zeros_like(K_load))

            if K_load.dtype.is_fp8():
                if Q.dtype.is_fp8():
                    K = K_load
                else:
                    K = (K_load.to(tl.float32) * tl.load(k_scale)).to(Q.dtype)
            else:
                K = K_load

            if V_load.dtype.is_fp8():
                if Q.dtype.is_fp8():
                    V = V_load
                else:
                    V = (V_load.to(tl.float32) * tl.load(v_scale)).to(Q.dtype)
            else:
                V = V_load

            seq_offset = (seq_block_idx * BLOCK_SIZE + offs_n).to(tl.int32)
            valid_token_mask = block_is_valid & (seq_offset >= recent_start)
            seq_mask = valid_token_mask[None, :] & (seq_offset[None, :] < seq_len) & (seq_offset[None, :] <= context_upper)
            if tl.constexpr(SLIDING_WINDOW > 0):
                seq_mask = seq_mask & ((context_len + query_pos[:, None] - seq_offset) < SLIDING_WINDOW)

            if USE_ALIBI_SLOPES:
                seq_offset_f32 = seq_offset.to(tl.float32)

            S = scale * tl.dot(Q, K)

            if USE_SOFTCAP:
                S = apply_softcap(S, softcap)

            S = tl.where(query_mask[:, None] & seq_mask,
                         S, float("-inf"))

            if USE_ALIBI_SLOPES:
                S += alibi_slope[:, None] * (seq_offset_f32 - context_len_f32)

            m_j = tl.maximum(M, tl.max(S, axis=1))
            m_j = tl.where(m_j > float("-inf"), m_j, 0.0)
            P = tl.exp(S - m_j[:, None])
            l_j = tl.sum(P, axis=1)
            alpha = tl.exp(M - m_j)
            acc = acc * alpha[:, None]
            L = L * alpha + l_j
            M = m_j
            acc += tl.dot(P.to(V.dtype), V)

    # epilogue: 安全除法，避免 L==0 时除零（当所有 token 被 mask 时）
    acc = tl.where(L[:, None] == 0.0, 0.0, acc / L[:, None])

    output_offset = (query_offset_0[:, None] * output_stride_0 +
                     query_offset_1[:, None] * output_stride_1 +
                     offs_d[None, :])

    tl.store(
        output_ptr + output_offset,
        acc,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
    )

@triton.autotune(
    configs=[
        # 覆盖与 general 3D 类似的候选集，让 autotune 在不同设备/形状上选择最优配置。
        triton.Config({}, num_warps=1, num_stages=1),
        triton.Config({}, num_warps=1, num_stages=2),
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=2, num_stages=3),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=4, num_stages=4),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=3),
        triton.Config({}, num_warps=8, num_stages=4),
    ],
    key=["HEAD_SIZE_PADDED", "BLOCK_SIZE", "NUM_SEGMENTS_PER_SEQ", "GRID_BIN", "NUM_SEQS_BIN"],
)
@triton.jit
def kernel_unified_attention_3d_compact_only(
        segm_output_ptr,
        # [num_tokens, num_query_heads, num_segments, head_size]
        segm_max_ptr,  # [num_tokens, num_query_heads, num_segments]
        segm_expsum_ptr,  # [num_tokens, num_query_heads, num_segments]
        query_ptr,  # [num_tokens, num_query_heads, head_size]
        key_cache_ptr,  # [num_blks, blk_size, num_kv_heads, head_size]
        value_cache_ptr,  # [num_blks, blk_size, num_kv_heads, head_size]
        block_tables_paged_ptr,  # [num_seqs, max_blocks_paged]
        key_compact_ptr,
        value_compact_ptr,
        token_positions_ptr,
        req_meta_i32_ptr,
        req_meta_i64_ptr,
        seqused_k_ptr,
        alibi_slopes_ptr,  # [num_query_heads]
        scale,  # float32
        k_scale,  # float32
        v_scale,  # float32
        softcap,  # float32
        req_meta_i32_stride_row: tl.int64,
        req_meta_i32_stride_col: tl.int64,
        req_meta_i64_stride_row: tl.int64,
        req_meta_i64_stride_col: tl.int64,
        token_pos_stride_head: tl.int64,
        token_pos_stride_block: tl.int64,
        token_pos_stride_token: tl.int64,
        query_stride_0: tl.int64,  # int
        query_stride_1: tl.int64,  # int, should be equal to head_size
        stride_k_cache_0: tl.int64,  # int
        stride_k_cache_1: tl.int64,  # int
        stride_k_cache_2: tl.int64,  # int
        stride_v_cache_0: tl.int64,  # int
        stride_v_cache_1: tl.int64,  # int
        stride_v_cache_2: tl.int64,  # int
        stride_k_compact_0: tl.int64,
        stride_k_compact_1: tl.int64,
        stride_k_compact_2: tl.int64,
        stride_v_compact_0: tl.int64,
        stride_v_compact_1: tl.int64,
        stride_v_compact_2: tl.int64,
        block_tables_paged_stride_row: tl.int64,
        query_start_len_ptr,  # [num_seqs+1]
        num_seqs: tl.int32,
        # ---- tl.constexpr meta parameters (keep at end for fast positional launch) ----
        num_query_heads: tl.constexpr,  # int
        num_queries_per_kv: tl.constexpr,  # int
        BLOCK_SIZE: tl.constexpr,  # int
        HEAD_SIZE: tl.constexpr,  # int
        HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
        USE_ALIBI_SLOPES: tl.constexpr,  # bool
        USE_SOFTCAP: tl.constexpr,  # bool
        SLIDING_WINDOW: tl.constexpr,  # int
        stride_k_cache_3: tl.constexpr,  # int
        stride_v_cache_3: tl.constexpr,  # int
        stride_k_compact_3: tl.constexpr,
        stride_v_compact_3: tl.constexpr,
        BLOCK_Q: tl.constexpr,  # int
        BLOCK_M: tl.constexpr,  # int
        NUM_SEGMENTS_PER_SEQ: tl.constexpr,  # int
        GRID_BIN: tl.constexpr,  # int, for autotune key separation of grid size
        NUM_SEQS_BIN: tl.constexpr,  # int, separate batch sizes
        PREFETCH_COMPACT: tl.constexpr,  # bool
):
    tl.static_assert(HEAD_SIZE_PADDED % 8 == 0)
    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    segm_idx = tl.program_id(2)

    seq_idx = find_seq_idx(query_start_len_ptr, q_block_global_idx, num_seqs,
                           BLOCK_Q, True)

    q_block_start_idx = tl.load(query_start_len_ptr +
                                seq_idx) // BLOCK_Q + seq_idx

    q_block_local_idx = q_block_global_idx - q_block_start_idx

    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)

    cur_batch_query_len = cur_batch_in_all_stop_index \
        - cur_batch_in_all_start_index

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_n = tl.arange(0, BLOCK_SIZE)

    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + \
        offs_m % num_queries_per_kv

    query_offset = (query_offset_0[:, None] * query_stride_0 +
                    query_offset_1[:, None] * query_stride_1 + offs_d[None, :])

    dim_mask = tl.where(offs_d < HEAD_SIZE, 1, 0).to(tl.int1)
    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)
    query_mask = query_mask_0 & query_mask_1

    Q = tl.load(
        query_ptr + query_offset,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
    )

    meta32_row_ptr = req_meta_i32_ptr + seq_idx * req_meta_i32_stride_row
    meta64_row_ptr = req_meta_i64_ptr + seq_idx * req_meta_i64_stride_row

    # per-request metadata - new compact layout:
    # req_meta_i32[7]: [kv_len_visible, compact_block_cnt, logits_last_n,
    #                  logits_row_offset, logits_capacity, flags, recent_len]
    # req_meta_i64[4]: [block_row_base_paged, compact_base_block, logits_base_ptr, token_row_base]
    # compact-only kernel 主要使用: kv_len_visible(0), compact_block_cnt(1), recent_len(6)
    # 以及：在 decode 且无需 token_positions（mask 仅用于尾部 padding）时，复用 req_meta_i32[2]
    # 传 compact_kv_len（不含 recent），以避免每段尾块读取 token_positions。
    # and: block_row_base_paged(0), compact_base_block(1), token_row_base(3)
    compact_block_cnt = tl.load(meta32_row_ptr + 1 * req_meta_i32_stride_col).to(tl.int32)
    compact_kv_len = tl.load(meta32_row_ptr + 2 * req_meta_i32_stride_col).to(tl.int32)
    flags = tl.load(meta32_row_ptr + 5 * req_meta_i32_stride_col).to(tl.int32)
    recent_meta = tl.load(meta32_row_ptr + 6 * req_meta_i32_stride_col).to(tl.int32)

    block_row_base_paged = tl.load(meta64_row_ptr + 0 * req_meta_i64_stride_col).to(tl.int64)
    compact_base_block = tl.load(meta64_row_ptr + 1 * req_meta_i64_stride_col).to(tl.int64)
    token_row_base = tl.load(meta64_row_ptr + 3 * req_meta_i64_stride_col).to(tl.int64)

    context_seq_len = tl.load(seqused_k_ptr + seq_idx).to(tl.int32)
    # compact-only decode：kv_len_visible 与 seqused_k 对齐，避免每步更新 meta[0]
    seq_len = context_seq_len
    context_delta = context_seq_len - cur_batch_query_len
    context_len = tl.where(context_delta > 0, context_delta, 0)
    # recent_len 三种表达：
    # - 默认：meta[6] 为 recent_len（旧路径/兼容路径）
    # - fast-path：meta[6] 为 recent_cap，且 flags bit2=1，kernel 内计算 recent_len（避免每步更新 meta）
    # - full-context：flags bit4=1，recent_len = context_seq_len（is_compact=0 行遍历所有 paged blocks）
    RECENT_CAP_FLAG = 4
    FULL_CONTEXT_FLAG = 16
    use_cap = (flags & RECENT_CAP_FLAG) != 0
    use_full = (flags & FULL_CONTEXT_FLAG) != 0
    recent_cap = tl.maximum(recent_meta, 0)
    cap = tl.minimum(context_seq_len, recent_cap)
    cap = tl.maximum(cap, 0)
    # BLOCK_SIZE 为 2^n（vLLM 默认 16），用按位对齐避免 int 除法的类型/代价问题。
    align_mask = tl.full([], -BLOCK_SIZE, tl.int32)
    recent_start_calc = (context_seq_len - cap) & align_mask
    recent_start_calc = tl.maximum(recent_start_calc, 0)
    recent_len_calc = tl.maximum(context_seq_len - recent_start_calc, 0)
    recent_len_from_cap = tl.where(recent_cap > 0, recent_len_calc, 0)
    recent_len_from_len = tl.maximum(recent_meta, 0)
    recent_len = tl.where(use_full, context_seq_len,
                 tl.where(use_cap, recent_len_from_cap, recent_len_from_len))
    recent_block_cnt = cdiv_fn(recent_len, BLOCK_SIZE)

    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(alibi_slopes_ptr + query_offset_1,
                              mask=query_mask_1,
                              other=0.0,
                             )
        context_len_f32 = context_len.to(tl.float32)

    # int32 转换：仅 block_ptr 需要的 stride 转换为 int32
    # compact KV block_ptr 需要 int32（offsets 非常量）
    stride_v_compact_0_i32 = _rt_i32(stride_v_compact_0)
    stride_k_compact_0_i32 = _rt_i32(stride_k_compact_0)
    stride_v_compact_1_i32 = _rt_i32(stride_v_compact_1)
    stride_k_compact_1_i32 = _rt_i32(stride_k_compact_1)
    stride_v_compact_2_i32 = _rt_i32(stride_v_compact_2)
    stride_k_compact_2_i32 = _rt_i32(stride_k_compact_2)
    # cache strides 保持 int64（block_ptr offsets=(0,0) 时 strides 可以是 int64）

    tl.multiple_of(stride_v_compact_1_i32, 16)
    tl.multiple_of(stride_k_compact_1_i32, 16)

    head_v_compact_offset = kv_head_idx * stride_v_compact_2_i32
    head_k_compact_offset = kv_head_idx * stride_k_compact_2_i32
    head_v_cache_offset = kv_head_idx * stride_v_cache_2  # 使用 int64
    head_k_cache_offset = kv_head_idx * stride_k_cache_2  # 使用 int64

    # token_positions 仅在 ALIBI/滑窗开启时需要：避免在常见 decode 场景引入额外寄存器/指针开销
    if USE_ALIBI_SLOPES or SLIDING_WINDOW > 0:
        token_row_base = tl.load(meta64_row_ptr + 3 * req_meta_i64_stride_col).to(tl.int64)
        token_pos_stride_block_i32 = _rt_i32(token_pos_stride_block)
        token_pos_stride_token_i32 = _rt_i32(token_pos_stride_token)
        tl.multiple_of(token_pos_stride_token_i32, 4)

        token_indices_ptr = token_positions_ptr + token_row_base
        token_head_offset = kv_head_idx * token_pos_stride_head  # 使用 int64
        token_indices_ptr_head = token_indices_ptr + token_head_offset

    context_upper = context_len + query_pos
    block_tables_ptr = block_tables_paged_ptr + block_row_base_paged * block_tables_paged_stride_row  # 使用 int64
    tail_block_start = (context_seq_len - recent_len) // BLOCK_SIZE

    num_blocks = compact_block_cnt + recent_block_cnt
    blocks_per_segment = cdiv_fn(num_blocks, NUM_SEGMENTS_PER_SEQ)
    if segm_idx * blocks_per_segment >= num_blocks:
        # 当前 segment 不含任何 block：写入空输出，避免 reduce_segments 读取未初始化的 segm 缓冲。
        empty_acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)
        segm_output_offset = (
            query_offset_0[:, None].to(tl.int64) *
            (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED) +
            query_offset_1[:, None] * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED) +
            segm_idx * HEAD_SIZE_PADDED + tl.arange(0, HEAD_SIZE_PADDED)[None, :])
        tl.store(
            segm_output_ptr + segm_output_offset,
            empty_acc,
            mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        )
        segm_offset = (
            query_offset_0.to(tl.int64) *
            (num_query_heads * NUM_SEGMENTS_PER_SEQ) +
            query_offset_1 * NUM_SEGMENTS_PER_SEQ + segm_idx)
        tl.store(segm_max_ptr + segm_offset,
                 tl.full([BLOCK_M], float("-inf"), dtype=tl.float32),
                 mask=query_mask_0 & query_mask_1)
        tl.store(segm_expsum_ptr + segm_offset,
                 tl.zeros([BLOCK_M], dtype=tl.float32),
                 mask=query_mask_0 & query_mask_1)
        return

    start_blk = segm_idx * blocks_per_segment
    end_blk = min((segm_idx + 1) * blocks_per_segment, num_blocks)
    compact_start = tl.minimum(compact_block_cnt, start_blk)
    compact_end = tl.minimum(compact_block_cnt, end_blk)
    recent_start_blk = tl.maximum(compact_block_cnt, start_blk)
    recent_end_blk = end_blk

    M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)

    # Compact 段：非尾块
    # decode 且无 alibi/滑窗时，若 compact 段末尾无 -1 padding，则每个 segment 都可走满块快路径，
    # 避免“每段一个尾块”导致的 token_positions 读与额外掩码开销。
    need_token_pos = USE_ALIBI_SLOPES or (SLIDING_WINDOW > 0)
    # compact_kv_len（不含 recent）由调度侧提供；若缺失(<=0)，按“满块”处理（保守且避免 token_positions 依赖）
    compact_kv_len_eff = tl.where(compact_kv_len > 0, compact_kv_len, compact_block_cnt * BLOCK_SIZE)
    has_padding = False
    if (not need_token_pos) and (compact_block_cnt > 0):
        has_padding = compact_kv_len_eff != (compact_block_cnt * BLOCK_SIZE)
    per_segment_tail = need_token_pos or has_padding
    non_tail_end = tl.where(per_segment_tail, tl.maximum(compact_start, compact_end - 1), compact_end)

    if tl.constexpr(PREFETCH_COMPACT):
        # === 预取路径：双缓冲优化 ===
        K_next = tl.zeros([HEAD_SIZE_PADDED, BLOCK_SIZE], dtype=Q.dtype)
        V_next = tl.zeros([BLOCK_SIZE, HEAD_SIZE_PADDED], dtype=Q.dtype)
        seq_next = tl.zeros([BLOCK_SIZE], dtype=tl.int32)

        # 预取第一个块
        if non_tail_end > compact_start:
            init_phys = compact_base_block + compact_start
            v_block_ptr0 = tl.make_block_ptr(
                base=value_compact_ptr + (init_phys * stride_v_compact_0_i32 + head_v_compact_offset),
                shape=(BLOCK_SIZE, HEAD_SIZE_PADDED),
                strides=(stride_v_compact_1_i32, stride_v_compact_3),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE, HEAD_SIZE_PADDED),
                order=(0, 1),
            )
            k_block_ptr0 = tl.make_block_ptr(
                base=key_compact_ptr + (init_phys * stride_k_compact_0_i32 + head_k_compact_offset),
                shape=(HEAD_SIZE_PADDED, BLOCK_SIZE),
                strides=(stride_k_compact_3, stride_k_compact_1_i32),
                offsets=(0, 0),
                block_shape=(HEAD_SIZE_PADDED, BLOCK_SIZE),
                order=(0, 1),
            )
            if tl.constexpr(HEAD_SIZE_PADDED == HEAD_SIZE):
                V_next = tl.load(v_block_ptr0)
                K_next = tl.load(k_block_ptr0)
            else:
                V_next = tl.load(v_block_ptr0, boundary_check=(1,), padding_option="zero")
                K_next = tl.load(k_block_ptr0, boundary_check=(0,), padding_option="zero")
            if USE_ALIBI_SLOPES or SLIDING_WINDOW > 0:
                token_block_ptr0 = tl.make_block_ptr(
                    base=token_indices_ptr_head + compact_start * token_pos_stride_block_i32,
                    shape=(BLOCK_SIZE,),
                    strides=(token_pos_stride_token_i32,),
                    offsets=(0,),
                    block_shape=(BLOCK_SIZE,),
                    order=(0,),
                )
                seq_next = tl.load(token_block_ptr0, boundary_check=(0,), padding_option="zero").to(tl.int32)

        # 非尾块循环
        for j in range(compact_start, non_tail_end):
            K_load = K_next
            V_load = V_next
            seq_offset = seq_next

            # 预取下一个块：用 tl.where 替代 runtime if 以规避 Triton 3.1.0 scf.if 编译器 bug
            # 最后一次迭代 nxt=j，重加载当前块（数据不使用）
            _prefetch = j + 1 < non_tail_end
            nxt = tl.where(_prefetch, j + 1, j)
            phys_nxt = compact_base_block + nxt
            v_block_ptr_nxt = tl.make_block_ptr(
                base=value_compact_ptr + (phys_nxt * stride_v_compact_0_i32 + head_v_compact_offset),
                shape=(BLOCK_SIZE, HEAD_SIZE_PADDED),
                strides=(stride_v_compact_1_i32, stride_v_compact_3),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE, HEAD_SIZE_PADDED),
                order=(0, 1),
            )
            k_block_ptr_nxt = tl.make_block_ptr(
                base=key_compact_ptr + (phys_nxt * stride_k_compact_0_i32 + head_k_compact_offset),
                shape=(HEAD_SIZE_PADDED, BLOCK_SIZE),
                strides=(stride_k_compact_3, stride_k_compact_1_i32),
                offsets=(0, 0),
                block_shape=(HEAD_SIZE_PADDED, BLOCK_SIZE),
                order=(0, 1),
            )
            if tl.constexpr(HEAD_SIZE_PADDED == HEAD_SIZE):
                V_next = tl.load(v_block_ptr_nxt)
                K_next = tl.load(k_block_ptr_nxt)
            else:
                V_next = tl.load(v_block_ptr_nxt, boundary_check=(1,), padding_option="zero")
                K_next = tl.load(k_block_ptr_nxt, boundary_check=(0,), padding_option="zero")
            if USE_ALIBI_SLOPES or SLIDING_WINDOW > 0:
                token_block_ptr_nxt = tl.make_block_ptr(
                    base=token_indices_ptr_head + nxt * token_pos_stride_block_i32,
                    shape=(BLOCK_SIZE,),
                    strides=(token_pos_stride_token_i32,),
                    offsets=(0,),
                    block_shape=(BLOCK_SIZE,),
                    order=(0,),
                )
                seq_next = tl.load(token_block_ptr_nxt, boundary_check=(0,), padding_option="zero").to(tl.int32)

            S = scale * tl.dot(Q, K_load)

            if USE_SOFTCAP:
                S = apply_softcap(S, softcap)

            if USE_ALIBI_SLOPES:
                S += alibi_slope[:, None] * (seq_offset.to(tl.float32) - context_len_f32)

            # 非尾块：滑窗开启时需要基于 seq_offset 额外遮罩
            if tl.constexpr(SLIDING_WINDOW > 0):
                window_mask = (context_upper[:, None] - seq_offset[None, :]) < SLIDING_WINDOW
                S = tl.where(query_mask[:, None] & window_mask, S, float("-inf"))
            else:
                S = tl.where(query_mask[:, None], S, float("-inf"))

            # Online softmax 更新（合并计算）
            m_new = tl.maximum(M, tl.max(S, axis=1))
            m_new = tl.where(m_new > float("-inf"), m_new, 0.0)
            alpha = tl.exp(M - m_new)
            P = tl.exp(S - m_new[:, None])
            L = L * alpha + tl.sum(P, axis=1)
            acc = acc * alpha[:, None] + tl.dot(P.to(V_load.dtype), V_load)
            M = m_new
    else:
        # === 非预取路径：复用 block_ptr + tl.advance（保持流水线）===
        if non_tail_end > compact_start:
            head_v_compact_offset64 = kv_head_idx * stride_v_compact_2
            head_k_compact_offset64 = kv_head_idx * stride_k_compact_2

            init_phys = compact_base_block.to(tl.int64) + compact_start.to(tl.int64)
            v_blk_ptr = tl.make_block_ptr(
                base=value_compact_ptr + (init_phys * stride_v_compact_0 + head_v_compact_offset64),
                shape=(BLOCK_SIZE, HEAD_SIZE_PADDED),
                strides=(stride_v_compact_1, stride_v_compact_3),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE, HEAD_SIZE_PADDED),
                order=(0, 1),
            )
            k_blk_ptr = tl.make_block_ptr(
                base=key_compact_ptr + (init_phys * stride_k_compact_0 + head_k_compact_offset64),
                shape=(HEAD_SIZE_PADDED, BLOCK_SIZE),
                strides=(stride_k_compact_3, stride_k_compact_1),
                offsets=(0, 0),
                block_shape=(HEAD_SIZE_PADDED, BLOCK_SIZE),
                order=(0, 1),
            )

            for j in range(compact_start, non_tail_end):
                if tl.constexpr(HEAD_SIZE_PADDED == HEAD_SIZE):
                    V_load = tl.load(v_blk_ptr)
                    K_load = tl.load(k_blk_ptr)
                else:
                    V_load = tl.load(v_blk_ptr, boundary_check=(1,), padding_option="zero")
                    K_load = tl.load(k_blk_ptr, boundary_check=(0,), padding_option="zero")

                if USE_ALIBI_SLOPES or SLIDING_WINDOW > 0:
                    token_block_ptr = tl.make_block_ptr(
                        base=token_indices_ptr_head + j * token_pos_stride_block_i32,
                        shape=(BLOCK_SIZE,),
                        strides=(token_pos_stride_token_i32,),
                        offsets=(0,),
                        block_shape=(BLOCK_SIZE,),
                        order=(0,),
                    )
                    seq_offset = tl.load(token_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.int32)

                S = scale * tl.dot(Q, K_load)

                if USE_SOFTCAP:
                    S = apply_softcap(S, softcap)

                if USE_ALIBI_SLOPES:
                    S += alibi_slope[:, None] * (seq_offset.to(tl.float32) - context_len_f32)

                if tl.constexpr(SLIDING_WINDOW > 0):
                    window_mask = (context_upper[:, None] - seq_offset[None, :]) < SLIDING_WINDOW
                    S = tl.where(query_mask[:, None] & window_mask, S, float("-inf"))
                else:
                    S = tl.where(query_mask[:, None], S, float("-inf"))

                m_new = tl.maximum(M, tl.max(S, axis=1))
                m_new = tl.where(m_new > float("-inf"), m_new, 0.0)
                alpha = tl.exp(M - m_new)
                P = tl.exp(S - m_new[:, None])
                L = L * alpha + tl.sum(P, axis=1)
                acc = acc * alpha[:, None] + tl.dot(P.to(V_load.dtype), V_load)
                M = m_new

                # 复用 block_ptr：跨 block 前进以保持 triton pipeline
                v_blk_ptr = tl.advance(v_blk_ptr, (BLOCK_SIZE, 0))
                k_blk_ptr = tl.advance(k_blk_ptr, (0, BLOCK_SIZE))

    # 尾块：可能含 -1，需要额外掩码（若无 padding 且无需 token_pos，可跳过）
    context_upper = context_len + query_pos
    if per_segment_tail and (compact_end > compact_start):
        j = compact_end - 1
        phys = compact_base_block + j
        v_block_ptr = tl.make_block_ptr(
            base=value_compact_ptr + (phys * stride_v_compact_0_i32 + head_v_compact_offset),
            shape=(BLOCK_SIZE, HEAD_SIZE_PADDED),
            strides=(stride_v_compact_1_i32, stride_v_compact_3),
            offsets=(0, 0),
            block_shape=(BLOCK_SIZE, HEAD_SIZE_PADDED),
            order=(0, 1),
        )
        k_block_ptr = tl.make_block_ptr(
            base=key_compact_ptr + (phys * stride_k_compact_0_i32 + head_k_compact_offset),
            shape=(HEAD_SIZE_PADDED, BLOCK_SIZE),
            strides=(stride_k_compact_3, stride_k_compact_1_i32),
            offsets=(0, 0),
            block_shape=(HEAD_SIZE_PADDED, BLOCK_SIZE),
            order=(0, 1),
        )
        if tl.constexpr(HEAD_SIZE_PADDED == HEAD_SIZE):
            K_load = tl.load(k_block_ptr)
            V_load = tl.load(v_block_ptr)
        else:
            K_load = tl.load(k_block_ptr, boundary_check=(0,), padding_option="zero")
            V_load = tl.load(v_block_ptr, boundary_check=(1,), padding_option="zero")
        if (not USE_ALIBI_SLOPES) and (SLIDING_WINDOW == 0):
            # 无 alibi/滑窗：仅需屏蔽尾部 padding，无需读取 token_positions
            tail_valid = compact_kv_len_eff - j * BLOCK_SIZE
            # 保持与 else 分支一致的 mask 形状：[BLOCK_M, BLOCK_SIZE]
            mask_tail = (offs_n[None, :] < tail_valid) & tl.full([BLOCK_M, 1], 1, dtype=tl.int1)
        else:
            token_block_ptr = tl.make_block_ptr(
                base=token_indices_ptr_head + j * token_pos_stride_block_i32,
                shape=(BLOCK_SIZE,),
                strides=(token_pos_stride_token_i32,),
                offsets=(0,),
                block_shape=(BLOCK_SIZE,),
                order=(0,),
            )
            seq_offset = tl.load(
                token_block_ptr,
                boundary_check=(0,),
                padding_option="zero",
                eviction_policy="evict_last",
            ).to(tl.int32)

            # 合并 mask 计算；滑窗禁用时跳过额外条件
            if SLIDING_WINDOW == 0:
                mask_tail = (seq_offset[None, :] >= 0) & (seq_offset[None, :] < seq_len) & (seq_offset[None, :] <= context_upper[:, None])
            else:
                mask_tail = (seq_offset[None, :] >= 0) & (seq_offset[None, :] < seq_len) & (seq_offset[None, :] <= context_upper[:, None])
                mask_tail = mask_tail & ((context_upper[:, None] - seq_offset) < SLIDING_WINDOW)

        S = scale * tl.dot(Q, K_load)

        if USE_SOFTCAP:
            S = apply_softcap(S, softcap)

        if USE_ALIBI_SLOPES:
            S += alibi_slope[:, None] * (seq_offset.to(tl.float32) - context_len_f32)

        S = tl.where(query_mask[:, None] & mask_tail, S, float("-inf"))

        # Online softmax 更新（合并计算）
        m_new = tl.maximum(M, tl.max(S, axis=1))
        m_new = tl.where(m_new > float("-inf"), m_new, 0.0)
        alpha = tl.exp(M - m_new)
        P = tl.exp(S - m_new[:, None])
        L = L * alpha + tl.sum(P, axis=1)
        acc = acc * alpha[:, None] + tl.dot(P.to(V_load.dtype), V_load)
        M = m_new

    recent_start = context_seq_len - recent_len
    # recent 段（paged 尾部）- 优化：减少中间变量，合并计算
    for r_blk in range(recent_start_blk, recent_end_blk):
        seq_block_idx = tail_block_start + (r_blk - compact_block_cnt)
        block_id_raw = tl.load(block_tables_ptr + seq_block_idx).to(tl.int64)
        block_is_valid = block_id_raw >= 0
        block_id = tl.where(block_is_valid, block_id_raw, 0)
        v_offset = (block_id * stride_v_cache_0 +
                    head_v_cache_offset +
                    offs_d[None, :] * stride_v_cache_3 +
                    offs_n[:, None] * stride_v_cache_1)
        k_offset = (block_id * stride_k_cache_0 +
                    head_k_cache_offset +
                    offs_d[:, None] * stride_k_cache_3 +
                    offs_n[None, :] * stride_k_cache_1)

        V_load = tl.load(
            value_cache_ptr + v_offset,
            mask=dim_mask[None, :] & block_is_valid,
            other=0.0,
        )
        K_load = tl.load(
            key_cache_ptr + k_offset,
            mask=dim_mask[:, None] & block_is_valid,
            other=0.0,
        )

        # 计算位置和 mask（合并）；滑窗禁用时不加额外条件
        seq_offset = (seq_block_idx * BLOCK_SIZE + offs_n).to(tl.int32)
        if SLIDING_WINDOW == 0:
            mask_recent = block_is_valid & (seq_offset[None, :] >= recent_start) & (seq_offset[None, :] < seq_len) & (seq_offset[None, :] <= context_upper[:, None])
        else:
            mask_recent = block_is_valid & (seq_offset[None, :] >= recent_start) & (seq_offset[None, :] < seq_len) & (seq_offset[None, :] <= context_upper[:, None])
            mask_recent = mask_recent & ((context_upper[:, None] - seq_offset) < SLIDING_WINDOW)

        S = scale * tl.dot(Q, K_load)

        if USE_SOFTCAP:
            S = apply_softcap(S, softcap)

        if USE_ALIBI_SLOPES:
            S += alibi_slope[:, None] * (seq_offset.to(tl.float32) - context_len_f32)

        S = tl.where(query_mask[:, None] & mask_recent, S, float("-inf"))

        # Online softmax 更新（合并计算）
        m_new = tl.maximum(M, tl.max(S, axis=1))
        m_new = tl.where(m_new > float("-inf"), m_new, 0.0)
        alpha = tl.exp(M - m_new)
        P = tl.exp(S - m_new[:, None])
        L = L * alpha + tl.sum(P, axis=1)
        acc = acc * alpha[:, None] + tl.dot(P.to(V_load.dtype), V_load)
        M = m_new

    segm_output_offset = (
        query_offset_0[:, None].to(tl.int64) *
        (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED) +
        query_offset_1[:, None] * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED) +
        segm_idx * HEAD_SIZE_PADDED + tl.arange(0, HEAD_SIZE_PADDED)[None, :])
    tl.store(
        segm_output_ptr + segm_output_offset,
        acc,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
    )
    segm_offset = (query_offset_0.to(tl.int64) *
                   (num_query_heads * NUM_SEGMENTS_PER_SEQ) +
                   query_offset_1 * NUM_SEGMENTS_PER_SEQ + segm_idx)
    # 若该 segment 全部被 mask（L==0），将 M 置回 -inf，避免 reduce_segments 选到伪 max=0
    M_store = tl.where(L == 0.0, float("-inf"), M)
    tl.store(segm_max_ptr + segm_offset, M_store, mask=query_mask_0 & query_mask_1)
    tl.store(segm_expsum_ptr + segm_offset,
             L,
             mask=query_mask_0 & query_mask_1)

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
    stride_pad_i64 = tl.maximum(stride_pad_i64, cap_i64)
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

def flash_attn_score_dump_fwd_unified(
    q: torch.Tensor,
    out: torch.Tensor,
    key_paged: torch.Tensor,
    value_paged: torch.Tensor,
    block_tables_paged: torch.Tensor,
    key_compact: torch.Tensor,
    value_compact: torch.Tensor,
    token_positions: torch.Tensor,
    req_meta_i32: torch.Tensor,
    req_meta_i64: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    seqused_k: torch.Tensor,
    max_seqlen_k: int,
    softmax_scale: float,
    softcap: float,
    alibi_slopes: Optional[torch.Tensor] = None,
    window_size: Optional[Tuple[int, int]] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    # 性能优化：允许调用方直接传入 flags，避免 GPU-CPU 同步
    # 如果为 None，则从 req_meta_i32 中计算（会触发 GPU 同步）
    _hint_has_logits: Optional[bool] = None,
    # 可选：标记本次是否需要 log_f（避免为判断 log_f 做 GPU-CPU 同步）
    _hint_has_log_f: Optional[bool] = None,
    # mixed batch 支持：调用方可显式标记 batch 内是否存在 last_n==1 / last_n>1 的 log_f slot，
    # 以避免为判定分支触发 GPU->CPU 同步。
    _hint_log_f_has_last_n_eq1: Optional[bool] = None,
    _hint_log_f_has_last_n_gt1: Optional[bool] = None,
    # 可选：log_f 输出是否使用 fp32（仅影响 log_f buffer 的 dtype）。
    _hint_log_f_out_fp32: Optional[bool] = None,
    # 可选：当 has_log_f=True 时必须显式传入，禁止静默 fallback 到常量。
    _hint_alpha_log_f: Optional[float] = None,
    # 可选：当 has_log_f=True 时必须显式传入（用于编码到 flags 高位，kernel 侧解码）。
    _hint_sink_tokens: Optional[int] = None,
    _hint_compact_only: Optional[bool] = None,
    _hint_strict_no_sync_fallback: Optional[bool] = None,
    # 性能/边界优化：当 compact 覆盖几乎全量 KV（例如 full-compact microbench），
    # 允许调用方显式要求走 dense-only general kernel，避免 compact-only 的额外代价。
    _hint_force_dense: Optional[bool] = None,
    # 性能优化：compact-only decode 允许调用方传入 max compact_kv_len（不含 recent）
    # 以在不触发 GPU-CPU 同步的前提下，选择更低发射开销的 2D 分支。
    _hint_compact_kv_len_max: Optional[int] = None,
    # 性能优化：允许调用方传入 num_seqs，避免 slice 操作。
    # 传入 None 时从 cu_seqlens_q.shape[0] - 1 推断（向后兼容）。
    _num_seqs: Optional[int] = None,
    # 可选：由 kernel 直接写 query_norms（用于 logits capture）
    query_norms: Optional[torch.Tensor] = None,
) -> None:
    """Request-wise unified attention kernel launcher (no legacy fallbacks).

    Args:
        _num_seqs: Optional. Number of sequences to process. If None, inferred
            from cu_seqlens_q.shape[0] - 1. Use this to avoid slicing
            pre-allocated buffers (req_meta_i32, req_meta_i64, etc.), reducing
            Python overhead in hot paths.
    """

    num_query_heads = q.shape[1]
    head_size = q.shape[2]
    num_kv_heads = key_paged.shape[2]
    # Use _num_seqs if provided, otherwise infer from cu_seqlens_q
    num_seqs = int(_num_seqs) if _num_seqs is not None else (cu_seqlens_q.shape[0] - 1)

    device = q.device

    # CPU wall profile for kernel launch/reduce is disabled; use Python-side
    # aggregate timing in patch layer to avoid kernel-side overhead.
    do_sample = False
    t_launch_ns = 0
    t_reduce_ns = 0
    t_total_start_ns = None
    cpu_kernel_name: Optional[str] = None

    # Sliding window：与 vLLM 原生对齐
    # vLLM 调度侧传入的 window_size 表示“窗口长度(不含当前 token)”，kernel 侧使用 (window_size + 1)
    if window_size is None:
        sliding_window = 0
    elif isinstance(window_size, int):
        sliding_window = (window_size + 1) if window_size > 0 else 0
    else:
        sliding_window = (window_size[0] + 1) if window_size[0] > 0 else 0

    if k_descale is None and _is_fp8_dtype(key_paged.dtype):
        k_descale = _cached_ones_tensor(device, (int(num_seqs), int(num_kv_heads)), torch.float32)
    if v_descale is None and _is_fp8_dtype(value_paged.dtype):
        v_descale = _cached_ones_tensor(device, (int(num_seqs), int(num_kv_heads)), torch.float32)

    block_size = value_paged.shape[1]
    num_queries_per_kv = num_query_heads // num_kv_heads
    BLOCK_M = 16
    BLOCK_Q = BLOCK_M // num_queries_per_kv
    total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs

    use_alibi = alibi_slopes is not None
    softcap = float(softcap)

    # 使用 hint 参数避免 GPU-CPU 同步，否则回退到 meta 扫描（会同步，bench/tests 可用）
    strict_no_sync = bool(_hint_strict_no_sync_fallback)
    if strict_no_sync:
        if _hint_has_logits is None or _hint_compact_only is None or _hint_has_log_f is None:
            raise ValueError(
                "flash_attn_score_dump_fwd_unified requires explicit hints in strict mode"
            )
    if _hint_has_logits is not None and _hint_compact_only is not None:
        has_logits_capture = bool(_hint_has_logits)
        if _hint_has_log_f is not None:
            has_log_f = bool(_hint_has_log_f)
        else:
            flags_col = req_meta_i32[:num_seqs, 5]
            has_log_f = bool(torch.any(flags_col.bitwise_and(8)).item())
        has_logits = bool(has_logits_capture or has_log_f)
        compact_only = bool(_hint_compact_only) and (not has_logits)
    else:
        flags_col = req_meta_i32[:num_seqs, 5]  # flags at index 5 in new compact layout
        has_logits_capture = bool(torch.any(flags_col.bitwise_and(2)).item())
        has_log_f = bool(torch.any(flags_col.bitwise_and(8)).item())
        has_logits = bool(has_logits_capture or has_log_f)
        compact_only = bool(torch.all(flags_col.bitwise_and(1)).item() and not has_logits)

    alpha_log_f: Optional[float] = None
    if has_log_f:
        if _hint_alpha_log_f is None:
            raise ValueError("has_log_f=True requires _hint_alpha_log_f; refuse implicit alpha fallback")
        if _hint_sink_tokens is None:
            raise ValueError("has_log_f=True requires _hint_sink_tokens; refuse implicit sink fallback")
        alpha_log_f = float(_hint_alpha_log_f)
        if not math.isfinite(alpha_log_f):
            raise ValueError(f"_hint_alpha_log_f must be finite, got {alpha_log_f}")
        validate_sink_tokens(int(_hint_sink_tokens))

    log_f_has_last_n_eq1 = False
    log_f_has_last_n_gt1 = False
    if has_log_f:
        if _hint_log_f_has_last_n_eq1 is None or _hint_log_f_has_last_n_gt1 is None:
            raise ValueError(
                "has_log_f=True requires _hint_log_f_has_last_n_eq1/_hint_log_f_has_last_n_gt1; "
                "refuse sync fallback for branch split"
            )
        log_f_has_last_n_eq1 = bool(_hint_log_f_has_last_n_eq1)
        log_f_has_last_n_gt1 = bool(_hint_log_f_has_last_n_gt1)

    log_f_out_fp32 = bool(_hint_log_f_out_fp32) if (_hint_log_f_out_fp32 is not None and has_log_f) else False
    # B1：last_n==1 统一采用 logits-only（selector 内做 log_softmax），不再在 kernel 侧做额外 postprocess。
    # 这样可移除一次额外 kernel launch，并避免混合 batch（prefill last_n>1 + decode last_n==1）时的语义分歧。

    query_norms_stride_head = 0
    query_norms_stride_token = 0
    query_norms_window = 0
    write_query_norms = False
    if query_norms is not None and query_norms.numel() > 0:
        query_norms_stride_head = int(query_norms.stride(1))
        query_norms_stride_token = int(query_norms.stride(2))
        query_norms_window = int(query_norms.shape[2])
        write_query_norms = True
    write_query_norms = bool(write_query_norms and has_logits_capture)

    # compact-only kernels 支持 GQA（num_queries_per_kv > 1），不做静默 fallback。

    force_dense_kernel = bool(_hint_force_dense) if _hint_force_dense is not None else False
    if force_dense_kernel and (has_logits or int(max_seqlen_q) != 1):
        # logits/非 decode 场景下强制 dense 可能改变语义；仅允许用于 decode 且无 logits 的边界性能优化。
        force_dense_kernel = False

    # tests/benchmark 通过 monkeypatch.setenv() 动态切换该开关；perf 热路径默认使用 import 时缓存值。
    prefetch_compact_env = (os.environ.get("VLLM_SPARSE_PREFETCH_COMPACT", "0") == "1") if _DYNAMIC_ENV else _PREFETCH_COMPACT_CACHED
    head_size_padded = _HEAD_SIZE_PADDED_CACHE.get(int(head_size))
    if head_size_padded is None:
        head_size_padded = int(triton.next_power_of_2(head_size))
        _HEAD_SIZE_PADDED_CACHE[int(head_size)] = head_size_padded
    # 与 vLLM 原生相同的分支选择：大 q_len 或大网格走 2D，否则走 3D。
    launch_large = (max_seqlen_q > 1 or total_num_q_blocks * num_kv_heads > 128)

    # Debug dispatch info（不要在 kernel 内读取该 dict）
    # compact-only 专用核仅用于 decode（max_seqlen_q==1）。
    # prefill(q_len>1) 若省略 causal mask 会导致数值错误（token attend to future tokens），
    # 因此 prefill 一律走与 vLLM 同构的 general kernel。
    compact_only_special = compact_only and (not has_logits) and (int(max_seqlen_q) == 1) and (not force_dense_kernel)
    if compact_only_special:
        compact_kv_len_max = int(_hint_compact_kv_len_max) if _hint_compact_kv_len_max is not None else None
        use_2d_compact_only = bool(launch_large)

        if use_2d_compact_only:
            cpu_kernel_name = "2d_compact_only"
            # 网格规模分桶，避免 autotune 结果被大/小场景互相污染
            if total_num_q_blocks <= 16:
                grid_bin = 0
            elif total_num_q_blocks <= 64:
                grid_bin = 1
            elif total_num_q_blocks <= 256:
                grid_bin = 2
            else:
                grid_bin = 3
            if num_seqs <= 4:
                num_seqs_bin = 0
            elif num_seqs <= 8:
                num_seqs_bin = 1
            elif num_seqs <= 16:
                num_seqs_bin = 2
            else:
                num_seqs_bin = 3
            runtime_2d, _, meta_stride3 = _cached_compact_only_stride_pack(
                device,
                req_meta_i32=req_meta_i32,
                req_meta_i64=req_meta_i64,
                key_paged=key_paged,
                value_paged=value_paged,
                block_tables_paged=block_tables_paged,
                key_compact=key_compact,
                value_compact=value_compact,
                token_positions=token_positions,
                query=q,
                output=out,
            )
            grid_2d = (total_num_q_blocks, num_kv_heads)
            stride_k_cache_3, stride_v_cache_3, stride_k_compact_3, stride_v_compact_3 = meta_stride3
            with _range("sparse.kernel.launch"):
                kernel_unified_attention_2d_compact_only[grid_2d](
                    out,
                    q,
                    key_paged,
                    value_paged,
                    block_tables_paged,
                    key_compact,
                    value_compact,
                    token_positions,
                    req_meta_i32,
                    req_meta_i64,
                    seqused_k,
                    alibi_slopes,
                    softmax_scale,
                    k_descale,
                    v_descale,
                    softcap,
                    *runtime_2d,
                    cu_seqlens_q,
                    num_seqs,
                    num_query_heads=num_query_heads,
                    num_queries_per_kv=num_queries_per_kv,
                    BLOCK_SIZE=block_size,
                    HEAD_SIZE=head_size,
                    HEAD_SIZE_PADDED=head_size_padded,
                    USE_ALIBI_SLOPES=use_alibi,
                    USE_SOFTCAP=(softcap > 0),
                    SLIDING_WINDOW=sliding_window,
                    BLOCK_Q=BLOCK_Q,
                    BLOCK_M=BLOCK_M,
                    PREFETCH_COMPACT=prefetch_compact_env,
                    GRID_BIN=grid_bin,
                    NUM_SEQS_BIN=num_seqs_bin,
                    stride_k_cache_3=stride_k_cache_3,
                    stride_v_cache_3=stride_v_cache_3,
                    stride_k_compact_3=stride_k_compact_3,
                    stride_v_compact_3=stride_v_compact_3,
                )
        else:
            cpu_kernel_name = "3d_compact_only"
            # 3D compact-only: small-batch decode 的“固定开销”占比很高。
            # 在不触发 GPU-CPU 同步的前提下，使用调用方提供的 compact_kv_len_max 做轻量分段：
            # - 小 compact（<=2048）用更少段数，减少 3D grid/segment reduce 开销；
            # - 其他情况保持 16 段（与 vLLM 3D 默认一致）。
            NUM_SEGMENTS = 16
            if compact_kv_len_max is not None and compact_kv_len_max <= 2048:
                NUM_SEGMENTS = 8

            segm_output, segm_max, segm_expsum = _cached_3d_workspace_tensors(
                device,
                num_tokens=int(q.shape[0]),
                num_query_heads=int(num_query_heads),
                num_segments=int(NUM_SEGMENTS),
                head_size_padded=int(head_size_padded),
            )

            _, runtime_3d, meta_stride3 = _cached_compact_only_stride_pack(
                device,
                req_meta_i32=req_meta_i32,
                req_meta_i64=req_meta_i64,
                key_paged=key_paged,
                value_paged=value_paged,
                block_tables_paged=block_tables_paged,
                key_compact=key_compact,
                value_compact=value_compact,
                token_positions=token_positions,
                query=q,
                output=out,
            )
            stride_k_cache_3, stride_v_cache_3, stride_k_compact_3, stride_v_compact_3 = meta_stride3

            prefetch_compact_flag = prefetch_compact_env

            grid = (total_num_q_blocks, num_kv_heads, NUM_SEGMENTS)
            # 3D small-grid 场景下 Python dispatch 开销占比很高：
            # - 直接调用 .fn 固定 num_warps/num_stages，避免 autotuner 的 key/查表开销；
            # - 同时可避免 autotune 在极短 kernel 上的噪声选型波动。
            kernel = kernel_unified_attention_3d_compact_only.fn
            with _range("sparse.kernel.launch"):
                kernel[grid](
                    segm_output,
                    segm_max,
                    segm_expsum,
                    q,
                    key_paged,
                    value_paged,
                    block_tables_paged,
                    key_compact,
                    value_compact,
                    token_positions,
                    req_meta_i32,
                    req_meta_i64,
                    seqused_k,
                    alibi_slopes,
                    softmax_scale,
                    k_descale,
                    v_descale,
                    softcap,
                    *runtime_3d,
                    cu_seqlens_q,
                    num_seqs,
                    num_query_heads=num_query_heads,
                    num_queries_per_kv=num_queries_per_kv,
                    BLOCK_SIZE=block_size,
                    HEAD_SIZE=head_size,
                    HEAD_SIZE_PADDED=head_size_padded,
                    USE_ALIBI_SLOPES=use_alibi,
                    USE_SOFTCAP=(softcap > 0),
                    SLIDING_WINDOW=sliding_window,
                    BLOCK_Q=BLOCK_Q,
                    BLOCK_M=BLOCK_M,
                    NUM_SEGMENTS_PER_SEQ=NUM_SEGMENTS,
                    GRID_BIN=0,
                    NUM_SEQS_BIN=0,
                    PREFETCH_COMPACT=prefetch_compact_flag,
                    stride_k_cache_3=stride_k_cache_3,
                    stride_v_cache_3=stride_v_cache_3,
                    stride_k_compact_3=stride_k_compact_3,
                    stride_v_compact_3=stride_v_compact_3,
                    num_warps=4,
                    num_stages=2,
                )

            reduce = reduce_segments
            with _range("sparse.kernel.reduce"):
                reduce[(q.shape[0], num_query_heads)](
                    out,
                    segm_output,
                    segm_max,
                    segm_expsum,
                    req_meta_i32,
                    seqused_k,
                    num_seqs,
                    out.stride(0),
                    out.stride(1),
                    req_meta_i32.stride(0),
                    req_meta_i32.stride(1),
                    cu_seqlens_q,
                    num_query_heads=num_query_heads,
                    BLOCK_SIZE=block_size,
                    HEAD_SIZE=head_size,
                    HEAD_SIZE_PADDED=head_size_padded,
                    BLOCK_Q=BLOCK_Q,
                    NUM_SEGMENTS_PER_SEQ=NUM_SEGMENTS,
                )
    else:
        stride_args, stride_args_3d_compact = _cached_stride_args(
            device,
            req_meta_i32=req_meta_i32,
            req_meta_i64=req_meta_i64,
            key_paged=key_paged,
            value_paged=value_paged,
            block_tables_paged=block_tables_paged,
            key_compact=key_compact,
            value_compact=value_compact,
            token_positions=token_positions,
        )
        dummy_logits_buf = _LAUNCH_CACHE.setdefault(
            (device, "dummy_logits_buf"),
            torch.empty(1, device=device, dtype=torch.float32),
        )
        launch_kwargs: dict = {}
        if launch_large:
            cpu_kernel_name = "2d_general"
            # mixed batch 支持：允许同一次 launch 内同时启用 last_n==1 与 last_n>1 的 log_f 路径。
            # - last_n==1：写 fp16 logits → post log_softmax 得到最终 log_f
            # - last_n>1：写 fp32 scratch → post 输出 (log_f_pre, denom_f)
            store_logits_for_logf_lastn1 = bool(has_log_f) and log_f_has_last_n_eq1
            store_logits_for_logf_gt1_scratch = (
                bool(has_log_f) and int(max_seqlen_q) > 1 and log_f_has_last_n_gt1
            )
            with _range("sparse.kernel.launch"):
                kernel_unified_attention_2d[(total_num_q_blocks, num_kv_heads)](
                    output_ptr=out,
                    query_ptr=q,
                    key_cache_ptr=key_paged,
                    value_cache_ptr=value_paged,
                    block_tables_paged_ptr=block_tables_paged,
                    key_compact_ptr=key_compact,
                    value_compact_ptr=value_compact,
                    token_positions_ptr=token_positions,
                    req_meta_i32_ptr=req_meta_i32,
                    req_meta_i64_ptr=req_meta_i64,
                    seqused_k_ptr=seqused_k,
                    logits_dummy_ptr=dummy_logits_buf,
                    query_norms_stride_head=query_norms_stride_head,
                    query_norms_stride_token=query_norms_stride_token,
                    query_norms_window=query_norms_window,
                    alibi_slopes_ptr=alibi_slopes,
                    scale=softmax_scale,
                    k_scale=k_descale,
                    v_scale=v_descale,
                    softcap=softcap,
                    num_query_heads=num_query_heads,
                    num_queries_per_kv=num_queries_per_kv,
                    query_stride_0=q.stride(0),
                    query_stride_1=q.stride(1),
                    output_stride_0=out.stride(0),
                    output_stride_1=out.stride(1),
                    BLOCK_SIZE=block_size,
                    HEAD_SIZE=head_size,
                    HEAD_SIZE_PADDED=head_size_padded,
                    USE_ALIBI_SLOPES=use_alibi,
                    USE_SOFTCAP=(softcap > 0),
                    SLIDING_WINDOW=sliding_window,
                    query_start_len_ptr=cu_seqlens_q,
                    BLOCK_Q=BLOCK_Q,
                    num_seqs=num_seqs,
	                    BLOCK_M=BLOCK_M,
	                    COMPACT_ONLY=(compact_only and (not force_dense_kernel)),
	                    FORCE_DENSE=force_dense_kernel,
	                    ENABLE_LOGITS_CAPTURE=has_logits_capture,
	                    ENABLE_LOGF=has_log_f,
	                    WRITE_QUERY_NORMS=write_query_norms,
	                    PREFETCH_COMPACT=prefetch_compact_env,
	                    STORE_LOGITS_FOR_LOGF_LASTN1=store_logits_for_logf_lastn1,
	                    STORE_LOGITS_FOR_LOGF_GT1_SCRATCH=store_logits_for_logf_gt1_scratch,
	                    LOGF_OUT_FP32=log_f_out_fp32,
	                    SKIP_OUTPUT=False,
	                    **stride_args,
	                    **launch_kwargs,
	                )
            if store_logits_for_logf_gt1_scratch:
                with _range("sparse.kernel.logf_postprocess_gt1"):
                    _launch_log_f_pre_from_logits_scratch_lastn_gt1(
                        req_meta_i32=req_meta_i32,
                        req_meta_i64=req_meta_i64,
                        num_seqs=num_seqs,
                        num_query_heads=num_query_heads,
                        log_f_out_fp32=log_f_out_fp32,
                        alpha=float(alpha_log_f),
                    )
        else:
            cpu_kernel_name = "3d_general"
            # 对齐 vLLM 原生 3D 设置，固定 16 段以获得确定性数值
            NUM_SEGMENTS = 16
            prefetch_compact_flag = bool(prefetch_compact_env)

            store_logits_for_logf_lastn1 = bool(has_log_f) and log_f_has_last_n_eq1
            store_logits_for_logf_gt1_scratch = (
                bool(has_log_f) and int(max_seqlen_q) > 1 and log_f_has_last_n_gt1
            )
            segm_output, segm_max, segm_expsum = _cached_3d_workspace_tensors(
                device,
                num_tokens=int(q.shape[0]),
                num_query_heads=int(num_query_heads),
                num_segments=int(NUM_SEGMENTS),
                head_size_padded=int(head_size_padded),
            )

            with _range("sparse.kernel.launch"):
                kernel_unified_attention_3d[(total_num_q_blocks, num_kv_heads, NUM_SEGMENTS)](
                    segm_output_ptr=segm_output,
                    segm_max_ptr=segm_max,
                    segm_expsum_ptr=segm_expsum,
                    query_ptr=q,
                    key_cache_ptr=key_paged,
                    value_cache_ptr=value_paged,
                    block_tables_paged_ptr=block_tables_paged,
                    key_compact_ptr=key_compact,
                    value_compact_ptr=value_compact,
                    token_positions_ptr=token_positions,
                    req_meta_i32_ptr=req_meta_i32,
                    req_meta_i64_ptr=req_meta_i64,
                    seqused_k_ptr=seqused_k,
                    logits_dummy_ptr=dummy_logits_buf,
                    query_norms_stride_head=query_norms_stride_head,
                    query_norms_stride_token=query_norms_stride_token,
                    query_norms_window=query_norms_window,
                    alibi_slopes_ptr=alibi_slopes,
                    scale=softmax_scale,
                    k_scale=k_descale,
                    v_scale=v_descale,
                    softcap=softcap,
                    num_query_heads=num_query_heads,
                    num_queries_per_kv=num_queries_per_kv,
                    query_stride_0=q.stride(0),
                    query_stride_1=q.stride(1),
                    BLOCK_SIZE=block_size,
                    HEAD_SIZE=head_size,
                    HEAD_SIZE_PADDED=head_size_padded,
                    USE_ALIBI_SLOPES=use_alibi,
                    USE_SOFTCAP=(softcap > 0),
                    SLIDING_WINDOW=sliding_window,
                    query_start_len_ptr=cu_seqlens_q,
                    BLOCK_Q=BLOCK_Q,
                    num_seqs=num_seqs,
                    BLOCK_M=BLOCK_M,
	                    NUM_SEGMENTS_PER_SEQ=NUM_SEGMENTS,
	                    COMPACT_ONLY=(compact_only and (not force_dense_kernel)),
	                    FORCE_DENSE=force_dense_kernel,
	                    ENABLE_LOGITS_CAPTURE=has_logits_capture,
	                    ENABLE_LOGF=has_log_f,
	                    WRITE_QUERY_NORMS=write_query_norms,
	                    PREFETCH_COMPACT=prefetch_compact_flag,
                    STORE_LOGITS_FOR_LOGF_LASTN1=store_logits_for_logf_lastn1,
                    STORE_LOGITS_FOR_LOGF_GT1_SCRATCH=store_logits_for_logf_gt1_scratch,
                    LOGF_OUT_FP32=log_f_out_fp32,
                    **stride_args,
                    **launch_kwargs,
                )
            with _range("sparse.kernel.reduce"):
                reduce_segments[(q.shape[0], num_query_heads)](
                    output_ptr=out,
                    segm_output_ptr=segm_output,
                    segm_max_ptr=segm_max,
                    segm_expsum_ptr=segm_expsum,
                    req_meta_i32_ptr=req_meta_i32,
                    seqused_k_ptr=seqused_k,
                    num_seqs=num_seqs,
                    num_query_heads=num_query_heads,
                    output_stride_0=out.stride(0),
                    output_stride_1=out.stride(1),
                    req_meta_i32_stride_row=req_meta_i32.stride(0),
                    req_meta_i32_stride_col=req_meta_i32.stride(1),
                    BLOCK_SIZE=block_size,
                    HEAD_SIZE=head_size,
                    HEAD_SIZE_PADDED=head_size_padded,
                    query_start_len_ptr=cu_seqlens_q,
                    BLOCK_Q=BLOCK_Q,
                    NUM_SEGMENTS_PER_SEQ=NUM_SEGMENTS,
                )
            if store_logits_for_logf_gt1_scratch:
                with _range("sparse.kernel.logf_postprocess_gt1"):
                    _launch_log_f_pre_from_logits_scratch_lastn_gt1(
                        req_meta_i32=req_meta_i32,
                        req_meta_i64=req_meta_i64,
                        num_seqs=num_seqs,
                        num_query_heads=num_query_heads,
                        log_f_out_fp32=log_f_out_fp32,
                        alpha=float(alpha_log_f),
                    )


def flash_attn_score_dump_fwd(*_, **__):
    # 兼容旧测试用入口：仅支持 dense 路径，内部构造最小 meta 后调用统一 kernel。
    q = __.get("q")
    k = __.get("k")
    v = __.get("v")
    block_table = __.get("block_table")
    out = __.get("out")
    scores_out = __.get("scores_out")
    kv_lengths = __.get("kv_lengths")
    softmax_scale_in = __.get("softmax_scale")
    cu_seqlens_q = __.get("cu_seqlens_q")
    max_seqlen_q = int(__.get("max_seqlen_q", 0))
    prefill_last_n = int(__.get("prefill_last_n", 0))
    dump_scores = bool(__.get("dump_scores", False))

    if q is None or k is None or v is None or block_table is None or out is None or cu_seqlens_q is None:
        raise RuntimeError("legacy flash_attn_score_dump_fwd missing required arguments")

    if dump_scores and scores_out is None:
        raise ValueError("scores_out must be provided when dump_scores=True")

    device = q.device
    batch, num_heads, max_q, head_size = q.shape
    block_size = k.shape[1]
    num_kv_heads = k.shape[2]

    if softmax_scale_in is None:
        softmax_scale = 1.0 / math.sqrt(float(head_size))
    else:
        softmax_scale = float(softmax_scale_in)

    if isinstance(kv_lengths, torch.Tensor):
        kv_lengths_tensor = kv_lengths.to(device=device, dtype=torch.int32)
    else:
        kv_lengths_tensor = torch.tensor(kv_lengths, dtype=torch.int32, device=device)

    if kv_lengths_tensor.dim() != 1:
        raise ValueError("Per-head kv_lengths are not supported")

    token_positions = torch.full(
        (batch, num_kv_heads, 1, block_size),
        -1,
        dtype=torch.int32,
        device=device,
    )
    # New compact layout: i32[7], i64[4]
    req_meta_i32 = torch.zeros((batch, 7), dtype=torch.int32, device=device)
    req_meta_i64 = torch.zeros((batch, 4), dtype=torch.int64, device=device)

    last_n_default = prefill_last_n if prefill_last_n > 0 else (1 if dump_scores else 0)

    q_blocks = q.permute(0, 2, 1, 3).contiguous()
    q_flat_parts: List[torch.Tensor] = []
    q_lens: List[int] = []

    for i in range(batch):
        q_offset = int(cu_seqlens_q[i].item())
        q_len = int(cu_seqlens_q[i + 1].item() - cu_seqlens_q[i].item())
        kv_len = int(kv_lengths_tensor[i].item())
        last_n = max(0, last_n_default)
        row_offset = max(0, q_len - last_n) if last_n > 0 else 0
        logits_capacity = scores_out.shape[-1] if scores_out is not None else kv_len

        # New compact layout:
        # req_meta_i32[7]: [kv_len_visible, compact_block_cnt, logits_last_n,
        #                  logits_row_offset, logits_capacity, flags, recent_len]
        req_meta_i32[i, 0] = kv_len  # kv_len_visible
        req_meta_i32[i, 1] = 0  # compact_block_cnt
        req_meta_i32[i, 2] = last_n  # logits_last_n
        req_meta_i32[i, 3] = row_offset  # logits_row_offset
        req_meta_i32[i, 4] = logits_capacity  # logits_capacity
        req_meta_i32[i, 5] = 2 if last_n > 0 else 0  # flags: bit1=logits, dense only
        req_meta_i32[i, 6] = 0  # recent_len (dense)

        # req_meta_i64[4]: [block_row_base_paged, compact_base_block, logits_base_ptr, token_row_base]
        req_meta_i64[i, 0] = i  # block_row_base_paged
        req_meta_i64[i, 1] = 0  # compact_base_block
        req_meta_i64[i, 2] = scores_out[i].data_ptr() if scores_out is not None and last_n > 0 else 0  # logits_base_ptr
        req_meta_i64[i, 3] = 0  # token_row_base (dense unused)

        q_lens.append(q_len)
        if q_len > 0:
            q_flat_parts.append(q_blocks[i, :q_len])

    q_flat = torch.cat(q_flat_parts, dim=0) if q_flat_parts else torch.empty((0, num_heads, head_size), device=device, dtype=q.dtype)
    out_flat = torch.empty_like(q_flat)

    flash_attn_score_dump_fwd_unified(
        q=q_flat,
        out=out_flat,
        key_paged=k,
        value_paged=v,
        block_tables_paged=block_table,
        key_compact=torch.empty((0, block_size, num_kv_heads, head_size), device=device, dtype=k.dtype),
        value_compact=torch.empty((0, block_size, num_kv_heads, head_size), device=device, dtype=v.dtype),
        token_positions=token_positions,
        req_meta_i32=req_meta_i32,
        req_meta_i64=req_meta_i64,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=max_seqlen_q,
        seqused_k=kv_lengths_tensor,
        max_seqlen_k=int(block_table.shape[1] * block_size),
        softmax_scale=softmax_scale,
        softcap=0.0,
    )

    # 将平铺输出写回原始 padded 布局
    cursor = 0
    for i, q_len in enumerate(q_lens):
        if q_len == 0:
            continue
        slice_flat = out_flat[cursor: cursor + q_len]
        out[i, :, :q_len, :].copy_(slice_flat.permute(1, 0, 2))
        cursor += q_len

    return scores_out
