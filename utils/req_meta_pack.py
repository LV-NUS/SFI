"""Pure-CUDA req_meta pack owner used by sparse decode and prefill.

The CUDA kernels live in :mod:`utils.req_meta_pack_ext`.  This module owns only
shape/stride validation and dispatch; it deliberately has no Triton dependency.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Optional

import torch

from utils.req_meta_flag_codec import REQ_META_SINK_SHIFT, validate_sink_tokens
from utils.req_meta_pack_ext import require_ext as _require_req_meta_pack_ext

_NULL_CTX = nullcontext()

def _range(_name: str):
    return _NULL_CTX

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
        _require_req_meta_pack_ext().pack_req_meta_decode_fast(
            seqused_k,
            is_compact_i32,
            compact_kv_len_i32,
            compact_offset_tokens_i64,
            req_meta_i32,
            req_meta_i64,
            int(meta_i32_stride0),
            int(meta_i64_stride0),
            int(batch),
            int(block_size),
            max(0, int(recent_cap)),
            int(validated_sink_tokens),
            int(REQ_META_SINK_SHIFT),
        )
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
        _require_req_meta_pack_ext().pack_req_meta_decode_fast_layers(
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
            int(is_compact_stride0),
            int(is_compact_stride1),
            int(compact_kv_len_stride0),
            int(compact_kv_len_stride1),
            int(compact_offset_stride0),
            int(compact_offset_stride1),
            int(meta_i32_stride0),
            int(meta_i32_stride1),
            int(meta_i64_stride0),
            int(meta_i64_stride1),
            int(layers),
            int(batch),
            int(block_size),
            max(0, int(recent_cap)),
            int(validated_sink_tokens),
            int(REQ_META_SINK_SHIFT),
        )

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
        _require_req_meta_pack_ext().pack_req_meta_prefill_fast_layers(
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
            int(meta_i32_stride0),
            int(meta_i32_stride1),
            int(meta_i64_stride0),
            int(meta_i64_stride1),
            int(layers),
            int(batch),
            max(0, int(recent_cap)),
            int(validated_sink_tokens),
            int(REQ_META_SINK_SHIFT),
        )
