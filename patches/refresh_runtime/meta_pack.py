from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import torch

_log = logging.getLogger(__name__)

from patches.runtime_deps import require_runtime_dep
from patches.layer_state import LayerState
from patches.refresh_runtime.row_semantic import (
    build_is_compact_i32_from_use_compact,
)
from patches.sparse_constants import (
    _LOGF_PRODUCER_ATTN,
    _LOGF_PRODUCER_NONE,
)
from patches.sparse_types import (
    LogitSpec,
    StepCaptureLayout,
    StepContext,
)
from patches.sparse_utils import _compute_recent_window, _get_row_index_tensor_from_cache

if TYPE_CHECKING:
    from patches.sparse_types import StepEnvelopeV2
    from patches.vllm_sparse_patch import VLLMSparseController

_get_global_decode_req_meta = require_runtime_dep("_get_global_decode_req_meta")
_get_global_prefill_req_meta = require_runtime_dep("_get_global_prefill_req_meta")
_get_cached_logits_patch_i32_stepwise_from_cache = require_runtime_dep(
    "_get_cached_logits_patch_i32_stepwise_from_cache"
)
from triton_kernel.flash_attn_score_dump_fwd import pack_req_meta_decode_fast
from triton_kernel.req_meta_flag_codec import encode_sink_into_scalar_flags
from triton_kernel.req_meta_flag_codec import validate_sink_tokens

from patches.sparse_cache import (
    _ROW_INDEX_TENSOR_CACHE_GLOBAL,
    _LOGITS_PATCH_STEPWISE_CACHE_GLOBAL,
)


def _collect_refresh_dynamic_dirty_rows(
    *,
    batch_size: int,
    slot_by_row: Sequence[int],
    is_prefill_by_row: Sequence[bool],
    refresh_dynamic_slot_set: frozenset,
) -> Tuple[int, ...]:
    """Collect decode rows touched by refresh/bootstrap dynamic slots."""
    if batch_size <= 0 or not refresh_dynamic_slot_set:
        return tuple()
    dirty_rows: List[int] = []
    for row in range(batch_size):
        if row < len(is_prefill_by_row) and bool(is_prefill_by_row[row]):
            continue
        slot = int(slot_by_row[row]) if row < len(slot_by_row) else -1
        if slot >= 0 and slot in refresh_dynamic_slot_set:
            dirty_rows.append(int(row))
    return tuple(dirty_rows)

def _build_logits_capture_specs(
    *,
    controller: "VLLMSparseController",
    state: LayerState,
    step_context: StepContext,
    prefill_capture_plan: Optional[Dict[int, int]],
    needs_logits_by_row: List[bool],
    is_prefill_by_row: List[bool],
    slot_by_row: List[int],
    logits_last_n_by_row: List[int],
    q_len_by_row: Sequence[int],
    context_kv_len_by_row: Sequence[int],
    logits_spec_by_row: List[Optional[LogitSpec]],
    log_f_denom_ptr_by_row: List[int],
    refresh_slot_list_for_payload_plan: Tuple[int, ...],
    key_cache: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqused_k: torch.Tensor,
    num_heads: int,
    device: torch.device,
    chunk_query_lengths: Optional[torch.Tensor],
    batch_size: int,
) -> Tuple[Optional["StepCaptureLayout"], int]:
    layer_index = controller.layer_index_by_cache_key.get(key_cache.data_ptr(), -1)
    if layer_index < 0:
        raise RuntimeError("logits capture requested but layer index missing")
    _, _, slot_in_chunk = controller._map_global_layer_to_capture_slot(layer_index)
    slot_in_chunk_for_log_f_capture = slot_in_chunk

    prefill_layout: Optional[StepCaptureLayout] = None
    refresh_layout: Optional[StepCaptureLayout] = None
    prefill_slot_index_map: Dict[int, int] = {}
    refresh_slot_index_map: Dict[int, int] = {}

    prefill_needs_logits = any(
        bool(needs_logits_by_row[row]) and bool(is_prefill_by_row[row])
        for row in range(batch_size)
    )
    refresh_needs_logits = any(
        bool(needs_logits_by_row[row]) and (not bool(is_prefill_by_row[row]))
        for row in range(batch_size)
    )

    if prefill_needs_logits:
        prefill_slots = sorted(int(slot) for slot in (prefill_capture_plan or {}).keys())
        if not prefill_slots:
            raise RuntimeError("logits capture requested but prefill slot_list is empty")
        prefill_layout = controller._get_step_capture_layout(
            phase="prefill",
            state=state,
            step_context=step_context,
            global_layer_index=layer_index,
            slot_list=prefill_slots,
            cu_seqlens_q=cu_seqlens_q,
            seqused_k=seqused_k,
            num_heads=num_heads,
            device=device,
            chunk_query_lengths=chunk_query_lengths,
            prepared_only=bool(
                getattr(controller, "_prefill_capture_meta_arena_enabled", False)
            ),
        )
        if prefill_layout is None:
            raise RuntimeError("prefill logits capture requested but layout missing")
        prefill_slot_index_map = prefill_layout.slot_to_capture_row
        if not prefill_slot_index_map:
            prefill_slot_index_map = {slot: idx for idx, slot in enumerate(prefill_layout.slot_list)}
            prefill_layout.slot_to_capture_row = prefill_slot_index_map

    refresh_layout_for_log_f_capture: Optional[StepCaptureLayout] = None
    if refresh_needs_logits:
        refresh_slot_list = list(refresh_slot_list_for_payload_plan)
        if not refresh_slot_list:
            raise RuntimeError(
                "logits capture requested but StepAuthority refresh slot_list is empty"
            )
        refresh_layout = controller._get_step_capture_layout(
            phase="refresh",
            state=state,
            step_context=step_context,
            global_layer_index=layer_index,
            slot_list=refresh_slot_list,
            cu_seqlens_q=cu_seqlens_q,
            seqused_k=seqused_k,
            num_heads=num_heads,
            device=device,
            chunk_query_lengths=None,
        )
        if refresh_layout is None:
            raise RuntimeError("refresh logits capture requested but layout missing")
        refresh_layout_for_log_f_capture = refresh_layout
        refresh_slot_index_map = refresh_layout.slot_to_capture_row
        if not refresh_slot_index_map:
            refresh_slot_index_map = {slot: idx for idx, slot in enumerate(refresh_layout.slot_list)}
            refresh_layout.slot_to_capture_row = refresh_slot_index_map

    prefill_scores_ptr = 0
    prefill_scores_stride_chunk = 0
    prefill_scores_stride_slot = 0
    prefill_scores_stride_head = 0
    prefill_scores_stride_token = 0
    prefill_scores_elem_size = 0
    prefill_denoms_ptr = 0
    prefill_denoms_stride_chunk = 0
    prefill_denoms_stride_slot = 0
    prefill_denoms_elem_size = 0
    if prefill_layout is not None:
        scores = prefill_layout.capture_scores
        prefill_scores_ptr = int(scores.data_ptr())
        prefill_scores_stride_chunk = int(scores.stride(0))
        prefill_scores_stride_slot = int(scores.stride(1))
        prefill_scores_stride_head = int(scores.stride(2))
        prefill_scores_stride_token = int(scores.stride(4))
        prefill_scores_elem_size = int(scores.element_size())
        denoms = prefill_layout.log_f_denoms
        prefill_denoms_ptr = int(denoms.data_ptr())
        prefill_denoms_stride_chunk = int(denoms.stride(0))
        prefill_denoms_stride_slot = int(denoms.stride(1))
        prefill_denoms_elem_size = int(denoms.element_size())

    refresh_scores_ptr = 0
    refresh_scores_stride_chunk = 0
    refresh_scores_stride_slot = 0
    refresh_scores_stride_head = 0
    refresh_scores_stride_token = 0
    refresh_scores_elem_size = 0
    refresh_denoms_ptr = 0
    refresh_denoms_stride_chunk = 0
    refresh_denoms_stride_slot = 0
    refresh_denoms_elem_size = 0
    if refresh_layout is not None:
        scores = refresh_layout.capture_scores
        refresh_scores_ptr = int(scores.data_ptr())
        refresh_scores_stride_chunk = int(scores.stride(0))
        refresh_scores_stride_slot = int(scores.stride(1))
        refresh_scores_stride_head = int(scores.stride(2))
        refresh_scores_stride_token = int(scores.stride(4))
        refresh_scores_elem_size = int(scores.element_size())
        denoms = refresh_layout.log_f_denoms
        refresh_denoms_ptr = int(denoms.data_ptr())
        refresh_denoms_stride_chunk = int(denoms.stride(0))
        refresh_denoms_stride_slot = int(denoms.stride(1))
        refresh_denoms_elem_size = int(denoms.element_size())

    for row in range(batch_size):
        if not needs_logits_by_row[row]:
            continue
        if bool(is_prefill_by_row[row]):
            layout = prefill_layout
            slot_index_map = prefill_slot_index_map
            scores_ptr = prefill_scores_ptr
            scores_stride_chunk = prefill_scores_stride_chunk
            scores_stride_slot = prefill_scores_stride_slot
            scores_stride_head = prefill_scores_stride_head
            scores_stride_token = prefill_scores_stride_token
            scores_elem_size = prefill_scores_elem_size
            denoms_ptr = prefill_denoms_ptr
            denoms_stride_chunk = prefill_denoms_stride_chunk
            denoms_stride_slot = prefill_denoms_stride_slot
            denoms_elem_size = prefill_denoms_elem_size
        else:
            layout = refresh_layout
            slot_index_map = refresh_slot_index_map
            scores_ptr = refresh_scores_ptr
            scores_stride_chunk = refresh_scores_stride_chunk
            scores_stride_slot = refresh_scores_stride_slot
            scores_stride_head = refresh_scores_stride_head
            scores_stride_token = refresh_scores_stride_token
            scores_elem_size = refresh_scores_elem_size
            denoms_ptr = refresh_denoms_ptr
            denoms_stride_chunk = refresh_denoms_stride_chunk
            denoms_stride_slot = refresh_denoms_stride_slot
            denoms_elem_size = refresh_denoms_elem_size
        if layout is None:
            raise RuntimeError("logits capture layout missing for mixed prefill/refresh")
        slot = slot_by_row[row]
        capture_row = slot_index_map.get(slot)
        if capture_row is None:
            raise RuntimeError(f"logits capture slot={slot} missing from layout")
        # 非 debug_ref：只需要 base_ptr/stride，不需要构造 per-row view 对象。
        # 通过 data_ptr + stride 做指针算术，避免在 prefill/refresh 上被 layers 放大。
        base_off_elems = int(slot_in_chunk) * int(scores_stride_chunk) + int(capture_row) * int(scores_stride_slot)
        base_ptr = int(scores_ptr + base_off_elems * int(scores_elem_size))
        denom_off_elems = int(slot_in_chunk) * int(denoms_stride_chunk) + int(capture_row) * int(denoms_stride_slot)
        denom_ptr_row = int(denoms_ptr + denom_off_elems * int(denoms_elem_size))
        log_f_denom_ptr_by_row[row] = int(denom_ptr_row)
        last_n = int(logits_last_n_by_row[row])
        row_offset = max(0, int(q_len_by_row[row]) - last_n)
        capacity = min(int(context_kv_len_by_row[row]), int(layout.kv_max))
        logits_spec_by_row[row] = LogitSpec(
            base_ptr=int(base_ptr),
            stride_head=int(scores_stride_head),
            stride_token=int(scores_stride_token),
            row_offset=row_offset,
            capacity=capacity,
        )
        # log_f/logits 写回路径会显式写 min_val；无需在 Python 侧清空大 buffer（避免额外带宽与同步）。

    return refresh_layout_for_log_f_capture, int(slot_in_chunk_for_log_f_capture)

def _pack_meta_from_global_decode(
    *,
    global_req_meta: Tuple[torch.Tensor, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    # decode：global_req_meta 路径下，meta64[2] out_ptr 已在 metadata builder 阶段通过
    # pack_req_meta_decode_fast_layers 一次性写入；dispatcher 不再做 per-layer patch。
    return global_req_meta

def _pack_meta_from_global_prefill(
    *,
    controller: Optional["VLLMSparseController"],
    global_prefill_meta: Tuple[torch.Tensor, torch.Tensor],
    step_context: StepContext,
    batch_size: int,
    is_prefill_by_row: List[bool],
    has_compact_row: bool,
    refresh_slots: Sequence[int],
    dense_log_f_has_last_n_gt1: bool,
    needs_logits_by_row: List[bool],
    use_compact_by_row: List[bool],
    logits_last_n_by_row: List[int],
    num_heads: int,
    device: torch.device,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    # prefill：仅在“纯 prefill step”启用全局 meta（mix chunk/multi-stage 直接禁用）。
    # 额外 runtime guard：避免因为上游时序/状态异常导致 silent wrong。
    prefill_only = bool(batch_size > 0 and all(bool(x) for x in is_prefill_by_row))
    if (not prefill_only) or has_compact_row or bool(refresh_slots) or controller is None:
        return None
    req_meta_i32, req_meta_i64 = global_prefill_meta

    # prefill last_n>1：scratch_ptr(meta64[1]) 仍是 per-layer workspace，需要在 dispatcher 补齐。
    if dense_log_f_has_last_n_gt1:
        stride_epoch = int(getattr(controller, "_prefill_log_f_stride_epoch", -1))
        stride_handle_id = int(getattr(controller, "_prefill_log_f_stride_handle_id", -1))
        stride_handle_generation = int(
            getattr(controller, "_prefill_log_f_stride_handle_generation", -1)
        )
        if (
            stride_epoch == int(step_context.epoch)
            and stride_handle_id == int(step_context.step_handle_id)
            and stride_handle_generation == int(step_context.step_handle_generation)
        ):
            stride_head = int(getattr(controller, "_prefill_log_f_stride_head", 0) or 0)
        else:
            stride_head = 0
        if stride_head > 0:
            logits_rows_gt1 = [
                int(r)
                for r in range(batch_size)
                if needs_logits_by_row[r]
                and (not use_compact_by_row[r])
                and int(logits_last_n_by_row[r]) > 1
            ]
            if logits_rows_gt1:
                max_last_n = max(int(logits_last_n_by_row[r]) for r in logits_rows_gt1)
                scratch = controller._ensure_log_f_workspace(
                    batch=len(logits_rows_gt1),
                    num_heads=num_heads,
                    last_n=int(max_last_n),
                    kv_max=int(stride_head),
                    device=device,
                )
                scratch_base = int(scratch.data_ptr())
                scratch_row_stride_bytes = int(scratch.stride(0) * scratch.element_size())
                row_index_gt1 = controller._get_row_index_tensor(rows=logits_rows_gt1, device=device)
                scratch_slots = controller._get_positions_i64(kv_len=int(row_index_gt1.numel()), device=device)
                scratch_ptrs = scratch_slots * scratch_row_stride_bytes + scratch_base
                req_meta_i64[row_index_gt1, 1] = scratch_ptrs
    # 注意：prefill 的 out_ptr(meta64[2]) 与 denom_ptr(meta64[3]) 已由 pack kernel 写入；
    # dispatcher 不再进行 logits_base_ptrs/logits_denom_ptrs 的 torch.tensor(list) patch。
    return req_meta_i32, req_meta_i64

def _patch_logits_meta_for_rows(
    *,
    controller: Optional["VLLMSparseController"],
    state: LayerState,
    step_context: StepContext,
    step_envelope: Optional["StepEnvelopeV2"],
    batch_size: int,
    device: torch.device,
    needs_logits_by_row: List[bool],
    use_compact_by_row: List[bool],
    logits_last_n_by_row: List[int],
    logits_spec_by_row: List[Optional[LogitSpec]],
    log_f_denom_ptr_by_row: List[int],
    q_len_by_row: Sequence[int],
    context_kv_len_by_row: Sequence[int],
    row_mode_signature: Tuple[int, ...],
    logits_rows_cached: Optional[Tuple[int, ...]],
    logits_rows_gt1_cached: Optional[Tuple[int, ...]],
    refresh_layout_for_log_f_capture: Optional["StepCaptureLayout"],
    slot_in_chunk_for_log_f_capture: int,
    num_heads: int,
    cu_seqlens_q: torch.Tensor,
    seqused_k: torch.Tensor,
    mix_fast_pack: bool,
    req_meta_i32: torch.Tensor,
    req_meta_i64: torch.Tensor,
) -> None:
    if logits_rows_cached is not None:
        logits_rows_seq: Tuple[int, ...] = tuple(
            int(row)
            for row in logits_rows_cached
            if int(row) < int(batch_size)
            and bool(needs_logits_by_row[int(row)])
            and (not bool(use_compact_by_row[int(row)]))
        )
    else:
        logits_rows_seq = tuple(
            row
            for row in range(batch_size)
            if bool(needs_logits_by_row[row]) and (not bool(use_compact_by_row[row]))
        )
    if not logits_rows_seq:
        return

    log_f_stride_head = int(logits_spec_by_row[int(logits_rows_seq[0])].stride_head)  # type: ignore[union-attr]
    row_index = (
        controller._get_row_index_tensor(rows=logits_rows_seq, device=device)
        if controller is not None
        else _get_row_index_tensor_from_cache(
            _ROW_INDEX_TENSOR_CACHE_GLOBAL,
            rows=logits_rows_seq,
            device=device,
        )
    )
    logits_last_n_vals: torch.Tensor
    logits_row_offsets: torch.Tensor
    logits_caps: torch.Tensor
    # 优先使用 step_context 的 GPU tensors 计算/缓存三坨 patch（避免 CPU list → torch.tensor(list)）
    envelope_cache_signature = (
        getattr(step_envelope, "cache_signature", None)
        if step_envelope is not None
        else None
    )
    try:
        if envelope_cache_signature is not None:
            logits_last_n_vals, logits_row_offsets, logits_caps = _get_cached_logits_patch_i32_stepwise_from_cache(
                _LOGITS_PATCH_STEPWISE_CACHE_GLOBAL,
                epoch=int(step_context.epoch),
                logits_rows=logits_rows_seq,
                capsule_signature=envelope_cache_signature,
                row_index=row_index,
                controller=controller,
                step_context=step_context,
                cu_seqlens_q=cu_seqlens_q,
                seqused_k=seqused_k,
                kv_max=int(log_f_stride_head),
                device=device,
            )
        else:
            logits_last_n_vals, logits_row_offsets, logits_caps = _get_cached_logits_patch_i32_stepwise_from_cache(
                _LOGITS_PATCH_STEPWISE_CACHE_GLOBAL,
                epoch=int(step_context.epoch),
                logits_rows=logits_rows_seq,
                capsule_signature=row_mode_signature,
                row_index=row_index,
                controller=controller,
                step_context=step_context,
                cu_seqlens_q=cu_seqlens_q,
                seqused_k=seqused_k,
                kv_max=int(log_f_stride_head),
                device=device,
            )
    except Exception as exc:
        raise RuntimeError(
            "stepwise logits patch build failed: "
            f"epoch={int(step_context.epoch)} layer={int(state.layer_index)} "
            f"rows={len(logits_rows_seq)} kv_max={int(log_f_stride_head)}"
        ) from exc
    if logits_rows_gt1_cached is not None:
        logits_rows_gt1 = tuple(
            int(row)
            for row in logits_rows_gt1_cached
            if int(row) < int(batch_size)
            and bool(needs_logits_by_row[int(row)])
            and (not bool(use_compact_by_row[int(row)]))
            and int(logits_last_n_by_row[int(row)]) > 1
        )
    else:
        logits_rows_gt1 = tuple(
            int(row)
            for row in logits_rows_seq
            if int(logits_last_n_by_row[int(row)]) > 1
        )
    if mix_fast_pack or refresh_layout_for_log_f_capture is None:
        if controller is None:
            raise RuntimeError("sparse controller missing while building logits_base_ptrs")
        logits_base_ptrs_cpu, logits_base_ptrs = controller._get_rebuild_ptr_buffers(
            name="logits_base_ptrs",
            device=device,
            size=len(logits_rows_seq),
        )
        logits_base_ptr_sig = []
        for ptr_idx, row in enumerate(logits_rows_seq):
            base_ptr = int(logits_spec_by_row[int(row)].base_ptr)
            logits_base_ptrs_cpu[ptr_idx] = base_ptr
            logits_base_ptr_sig.append((int(row), base_ptr))
        controller._publish_rebuild_ptr_buffer_if_needed(
            name="logits_base_ptrs",
            cpu=logits_base_ptrs_cpu,
            gpu=logits_base_ptrs,
            signature=tuple(logits_base_ptr_sig),
        )
    else:
        logits_base_ptrs = controller._capture_scores_ptrs_for_rows(
            layout=refresh_layout_for_log_f_capture,
            slot_in_chunk=int(slot_in_chunk_for_log_f_capture),
            row_index=row_index,
            rows_tuple=logits_rows_seq,
            device=device,
        )
    req_meta_i64[row_index, 1] = 0
    req_meta_i64[row_index, 3] = 0
    if logits_rows_gt1 and controller is not None:
        max_last_n = max(int(logits_last_n_by_row[int(row)]) for row in logits_rows_gt1)
        scratch = controller._ensure_log_f_workspace(
            batch=len(logits_rows_gt1),
            num_heads=num_heads,
            last_n=int(max_last_n),
            kv_max=int(log_f_stride_head),
            device=device,
        )
        scratch_base = int(scratch.data_ptr())
        scratch_row_stride_bytes = int(scratch.stride(0) * scratch.element_size())
        row_index_gt1 = controller._get_row_index_tensor(rows=logits_rows_gt1, device=device)
        scratch_slots = controller._get_positions_i64(kv_len=int(row_index_gt1.numel()), device=device)
        scratch_ptrs = scratch_slots * scratch_row_stride_bytes + scratch_base
        req_meta_i64[row_index_gt1, 1] = scratch_ptrs
        if mix_fast_pack:
            denom_ptrs_gt1_cpu, denom_ptrs_gt1 = controller._get_rebuild_ptr_buffers(
                name="log_f_denom_ptrs_gt1",
                device=device,
                size=len(logits_rows_gt1),
            )
            denom_ptrs_gt1_sig = []
            for ptr_idx, row in enumerate(logits_rows_gt1):
                denom_ptr = int(log_f_denom_ptr_by_row[int(row)])
                denom_ptrs_gt1_cpu[ptr_idx] = denom_ptr
                denom_ptrs_gt1_sig.append((int(row), denom_ptr))
            controller._publish_rebuild_ptr_buffer_if_needed(
                name="log_f_denom_ptrs_gt1",
                cpu=denom_ptrs_gt1_cpu,
                gpu=denom_ptrs_gt1,
                signature=tuple(denom_ptrs_gt1_sig),
            )
        else:
            if refresh_layout_for_log_f_capture is None:
                raise RuntimeError("refresh_layout_for_log_f_capture missing while logits_rows_gt1 is non-empty")
            denom_ptrs_gt1 = controller._log_f_denoms_ptrs_for_rows(
                layout=refresh_layout_for_log_f_capture,
                slot_in_chunk=int(slot_in_chunk_for_log_f_capture),
                row_index=row_index_gt1,
                rows_tuple=logits_rows_gt1,
                device=device,
            )
        req_meta_i64[row_index_gt1, 3] = denom_ptrs_gt1
    req_meta_i32[row_index, 2] = logits_last_n_vals
    req_meta_i32[row_index, 3] = logits_row_offsets
    req_meta_i32[row_index, 4] = logits_caps
    req_meta_i32[row_index, 1] = int(log_f_stride_head)
    # ⚠️ 注意：req_meta_i32[row_index, 5] 是 advanced indexing，返回的是拷贝；
    # 直接 bitwise_or_ 不会写回原 tensor。
    # 必须通过赋值触发 scatter 写回，确保 kernel 看到 use_log_f 标记。
    # log_f 行必须是 dense 语义：强制清除 compact 位，再写入 log_f 位，
    # 防止上游复用缓存时出现 compact+log_f 混合 flags。
    req_meta_i32[row_index, 5] = (req_meta_i32[row_index, 5] & (~1)) | 8
    req_meta_i64[row_index, 2] = logits_base_ptrs

def _pack_meta_from_local_plan(
    *,
    controller: Optional["VLLMSparseController"],
    state: LayerState,
    step_context: StepContext,
    step_envelope: Optional["StepEnvelopeV2"],
    use_fast_pack: bool,
    use_step_cache_pack_cached: bool,
    has_decode_row: bool,
    has_prefill_row: bool,
    is_prefill_by_row: Sequence[bool],
    batch_size: int,
    block_size: int,
    num_heads: int,
    device: torch.device,
    seqused_k: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    needs_logits_by_row: List[bool],
    use_compact_by_row: List[bool],
    logf_producer_by_row: Optional[List[int]],
    logits_last_n_by_row: List[int],
    logits_spec_by_row: List[Optional[LogitSpec]],
    log_f_denom_ptr_by_row: List[int],
    compact_kv_len_by_row: List[int],
    compact_offsets: Optional[List[int]],
    slot_by_row: List[int],
    row_mode_signature: Tuple[int, ...],
    logits_rows_cached: Optional[Tuple[int, ...]],
    logits_rows_gt1_cached: Optional[Tuple[int, ...]],
    refresh_layout_for_log_f_capture: Optional["StepCaptureLayout"],
    slot_in_chunk_for_log_f_capture: int,
    q_len_by_row: Sequence[int],
    context_kv_len_by_row: Sequence[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if logf_producer_by_row is None:
        logf_producer_by_row = [
            int(_LOGF_PRODUCER_ATTN) if bool(needs_logits_by_row[row]) else int(_LOGF_PRODUCER_NONE)
            for row in range(batch_size)
        ]

    recent_cap = 0
    sink_tokens = 0
    if controller is not None and getattr(controller, "config", None) is not None:
        semantic_snapshot = controller._get_step_semantic_snapshot()
        recent_cap = max(0, int(semantic_snapshot.recent_tokens))
        sink_tokens = validate_sink_tokens(int(semantic_snapshot.sink_tokens))
    rows_needed = max(1, batch_size)
    req_meta_i32 = state.launch_meta_i32
    req_meta_i64 = state.launch_meta_i64
    if req_meta_i32 is None or req_meta_i32.shape[0] < rows_needed or req_meta_i32.shape[1] < 7:
        req_meta_i32 = torch.empty((rows_needed, 7), dtype=torch.int32, device=device)
    if req_meta_i64 is None or req_meta_i64.shape[0] < rows_needed or req_meta_i64.shape[1] < 4:
        req_meta_i64 = torch.empty((rows_needed, 4), dtype=torch.int64, device=device)
    # #13(b)①: 写回复用缓冲（layer_state.py:141-142 的 launch_meta_*）。原代码 realloc
    # 后从不回填 state，导致每步自废缓存、重复 alloc 两个小张量。回填同形同 dtype 的
    # 张量供下一步复用；不改写入值（fast: pack_req_meta_decode_fast 全量写；slow:
    # zero_()+逐行写），故 byte 恒等。未 realloc 时为自赋值，幂等。
    state.launch_meta_i32 = req_meta_i32
    state.launch_meta_i64 = req_meta_i64

    mix_fast_pack = (
        (not use_fast_pack)
        and batch_size > 0
        and has_decode_row
        and has_prefill_row
        and controller is not None
    )
    for row in range(batch_size):
        producer = (
            int(logf_producer_by_row[row])
            if row < len(logf_producer_by_row)
            else int(_LOGF_PRODUCER_NONE)
        )
        if bool(use_compact_by_row[row]) and producer != int(_LOGF_PRODUCER_NONE):
            raise RuntimeError(
                f"invalid meta( compact row with log_f producer ): row={row} producer={producer}"
            )
    if (use_fast_pack or mix_fast_pack) and batch_size > 0:
        # 设计约束：is_compact 必须来自当步 row_mode/use_compact_by_row（单一真源），
        # 不能复用 step_cache 中的静态 compact 标记，避免动态门禁阶段语义漂移。
        is_compact_t = build_is_compact_i32_from_use_compact(
            use_compact_by_row=use_compact_by_row,
            batch_size=int(batch_size),
            device=device,
        )

        # M5 Part B2 (2026-04-24): compact layout read redirects to the
        # step-level CompactRecentLaunchPlan — the single source of truth for
        # compact descriptors. We still accept `use_step_cache_pack_cached`
        # as an upstream hint, but the backing check is `plan.valid` so the
        # per-layer state.step_cache_compact_*_gpu fields are no longer
        # consulted here (Part C deletes them).
        _local_plan = (
            getattr(controller, "step_bound_meta", None).compact_recent_launch_plan
            if (
                controller is not None
                and getattr(controller, "step_bound_meta", None) is not None
            )
            else None
        )
        use_step_cache_compact_layout = bool(use_step_cache_pack_cached)
        if not use_step_cache_compact_layout:
            use_step_cache_compact_layout = (
                _local_plan is not None
                and bool(getattr(_local_plan, "valid", False))
            )
        if use_step_cache_compact_layout and _local_plan is not None and bool(
            getattr(_local_plan, "valid", False)
        ):
            compact_kv_len_t = _local_plan.compact_valid_tokens_i32[:batch_size]
            compact_offset_tokens_t = _local_plan.compact_offset_tokens_i64[:batch_size]
        else:
            state.ensure_compact_layout_buffers(batch_size)
            if (
                state.compact_layout_kv_len_i32 is None
                or state.compact_layout_offset_tokens_i64 is None
            ):
                raise RuntimeError("compact layout buffers missing after ensure_compact_layout_buffers")
            compact_kv_len_t = state.compact_layout_kv_len_i32[:batch_size]
            compact_offset_tokens_t = state.compact_layout_offset_tokens_i64[:batch_size]

            compact_kv_len_cpu = [int(compact_kv_len_by_row[row]) for row in range(batch_size)]
            if compact_offsets is None:
                compact_offsets = [0 for _ in range(batch_size)]
                for row in range(batch_size):
                    if not use_compact_by_row[row]:
                        continue
                    slot = slot_by_row[row]
                    if slot < 0 or slot >= len(state.compact_offset_tokens):
                        raise RuntimeError(f"compact slot missing for row={row} slot={slot}")
                    compact_offsets[row] = slot * int(state.compact_stride_blocks)
            compact_offset_tokens_cpu = [
                int(compact_offsets[row]) * int(block_size) if use_compact_by_row[row] else 0
                for row in range(batch_size)
            ]
            compact_kv_len_t.copy_(torch.as_tensor(compact_kv_len_cpu, dtype=torch.int32))
            compact_offset_tokens_t.copy_(torch.as_tensor(compact_offset_tokens_cpu, dtype=torch.int64))

        pack_req_meta_decode_fast(
            seqused_k=seqused_k,
            is_compact_i32=is_compact_t,
            compact_kv_len_i32=compact_kv_len_t,
            compact_offset_tokens_i64=compact_offset_tokens_t,
            req_meta_i32=req_meta_i32,
            req_meta_i64=req_meta_i64,
            block_size=int(block_size),
            recent_cap=int(recent_cap),
            sink_tokens=int(sink_tokens),
            num_seqs=int(batch_size),
        )
        _patch_logits_meta_for_rows(
            controller=controller,
            state=state,
            step_context=step_context,
            step_envelope=step_envelope,
            batch_size=int(batch_size),
            device=device,
            needs_logits_by_row=needs_logits_by_row,
            use_compact_by_row=use_compact_by_row,
            logits_last_n_by_row=logits_last_n_by_row,
            logits_spec_by_row=logits_spec_by_row,
            log_f_denom_ptr_by_row=log_f_denom_ptr_by_row,
            q_len_by_row=q_len_by_row,
            context_kv_len_by_row=context_kv_len_by_row,
            row_mode_signature=row_mode_signature,
            logits_rows_cached=logits_rows_cached,
            logits_rows_gt1_cached=logits_rows_gt1_cached,
            refresh_layout_for_log_f_capture=refresh_layout_for_log_f_capture,
            slot_in_chunk_for_log_f_capture=int(slot_in_chunk_for_log_f_capture),
            num_heads=int(num_heads),
            cu_seqlens_q=cu_seqlens_q,
            seqused_k=seqused_k,
            mix_fast_pack=bool(mix_fast_pack),
            req_meta_i32=req_meta_i32,
            req_meta_i64=req_meta_i64,
        )
        if mix_fast_pack:
            prefill_rows = [
                row
                for row in range(batch_size)
                if row < len(is_prefill_by_row) and bool(is_prefill_by_row[row])
            ]
            prefill_rows = [
                row
                for row in prefill_rows
                if (row < batch_size) and (not needs_logits_by_row[row])
            ]
            if prefill_rows:
                row_index_prefill = controller._get_row_index_tensor(rows=prefill_rows, device=device)  # type: ignore[union-attr]
                req_meta_i32[row_index_prefill, 4] = 0
                req_meta_i32[row_index_prefill, 5] = 0
                req_meta_i32[row_index_prefill, 6] = 0
                req_meta_i64[row_index_prefill, 1] = 0
                req_meta_i64[row_index_prefill, 2] = 0
                req_meta_i64[row_index_prefill, 3] = 0
    else:
        req_meta_i32[:rows_needed].zero_()
        req_meta_i64[:rows_needed].zero_()
        log_f_scratch_base = 0
        log_f_scratch_stride_bytes = 0
        log_f_scratch_ptr_by_row: Dict[int, int] = {}
        sink_tokens_for_flags = 0
        if controller is not None:
            sink_tokens_for_flags = validate_sink_tokens(
                int(controller._get_step_semantic_snapshot().sink_tokens)
            )
            dense_logf_rows = [
                r for r in range(batch_size)
                if needs_logits_by_row[r] and (not use_compact_by_row[r])
            ]
            if dense_logf_rows:
                dense_logf_rows_gt1 = [r for r in dense_logf_rows if int(logits_last_n_by_row[r]) > 1]
                max_last_n = max(int(logits_last_n_by_row[r]) for r in dense_logf_rows_gt1) if dense_logf_rows_gt1 else 0
                if max_last_n > 1 and dense_logf_rows_gt1:
                    # stride_head：来自 capture buffer 的 head stride（pad 到 kv_max）
                    stride_head = int(logits_spec_by_row[dense_logf_rows[0]].stride_head)  # type: ignore[union-attr]
                    scratch = controller._ensure_log_f_workspace(
                        batch=len(dense_logf_rows_gt1),
                        num_heads=num_heads,
                        last_n=int(max_last_n),
                        kv_max=int(stride_head),
                        device=device,
                    )
                    log_f_scratch_base = int(scratch.data_ptr())
                    log_f_scratch_stride_bytes = int(scratch.stride(0) * scratch.element_size())
                    for idx_s, row_s in enumerate(dense_logf_rows_gt1):
                        log_f_scratch_ptr_by_row[int(row_s)] = int(log_f_scratch_base + idx_s * log_f_scratch_stride_bytes)
        for row in range(batch_size):
            kv_len_visible = int(context_kv_len_by_row[row])
            compact_kv_len = int(compact_kv_len_by_row[row])
            use_compact = use_compact_by_row[row]
            producer = (
                int(logf_producer_by_row[row])
                if row < len(logf_producer_by_row)
                else (int(_LOGF_PRODUCER_ATTN) if bool(needs_logits_by_row[row]) else int(_LOGF_PRODUCER_NONE))
            )
            if bool(use_compact) and producer != int(_LOGF_PRODUCER_NONE):
                raise RuntimeError(
                    f"invalid meta( compact row with log_f producer ): row={row} producer={producer}"
                )
            q_len = int(q_len_by_row[row])
            producer_is_attn = producer == int(_LOGF_PRODUCER_ATTN)
            logits_spec = logits_spec_by_row[row] if producer_is_attn else None
            logits_last_n = int(logits_last_n_by_row[row]) if producer_is_attn else 0
            logits_row_offset = logits_spec.row_offset if logits_spec is not None else 0
            logits_capacity = logits_spec.capacity if logits_spec is not None else 0

            compact_block_cnt = (compact_kv_len + block_size - 1) // block_size if use_compact else 0
            recent_len = 0
            if use_compact:
                _, recent_len = _compute_recent_window(kv_len_visible, recent_cap, block_size)

            meta2 = logits_last_n
            if use_compact and (q_len == 1) and (logits_last_n == 0):
                meta2 = compact_kv_len

            flags = 0
            if use_compact:
                flags |= 1
            elif producer == int(_LOGF_PRODUCER_ATTN):
                flags |= 8
                # dense log_f：使用 recent_cap 模式（与 decode fast pack 对齐），避免 Python 侧计算 recent_len。
                flags |= 4
            flags = encode_sink_into_scalar_flags(
                base_flags=int(flags),
                sink_tokens=int(sink_tokens_for_flags),
            )

            req_meta_i32[row, 0] = kv_len_visible
            # dense log_f：meta32[1] 复用为 log_f_stride_head（pad stride）；compact 行仍为 compact_block_cnt
            req_meta_i32[row, 1] = int(logits_spec.stride_head) if (logits_spec is not None and (not use_compact) and logits_last_n > 0) else compact_block_cnt
            req_meta_i32[row, 2] = meta2
            req_meta_i32[row, 3] = logits_row_offset
            req_meta_i32[row, 4] = logits_capacity
            req_meta_i32[row, 5] = flags
            req_meta_i32[row, 6] = int(recent_cap) if (not use_compact and logits_last_n > 0) else int(recent_len)

            req_meta_i64[row, 0] = row
            if use_compact:
                req_meta_i64[row, 1] = compact_offsets[row]
            else:
                # dense log_f(last_n>1)：meta64[1] 写 scratch ptr；否则为 0
                if logits_last_n > 1 and log_f_scratch_ptr_by_row:
                    req_meta_i64[row, 1] = int(log_f_scratch_ptr_by_row.get(int(row), 0))
                else:
                    req_meta_i64[row, 1] = 0
            req_meta_i64[row, 2] = logits_spec.base_ptr if logits_spec is not None else 0
            if use_compact:
                token_row_base = compact_offsets[row] * block_size
                req_meta_i64[row, 3] = token_row_base if token_row_base >= 0 else 0
            else:
                # dense log_f(last_n>1)：meta64[3] 写 denom ptr（来自 step capture arena）
                req_meta_i64[row, 3] = int(log_f_denom_ptr_by_row[row]) if logits_last_n > 1 else 0
    return req_meta_i32, req_meta_i64

def _pack_dispatch_meta(
    *,
    controller: Optional["VLLMSparseController"],
    state: LayerState,
    step_context: StepContext,
    step_envelope: Optional["StepEnvelopeV2"],
    is_decode: bool,
    use_fast_pack: bool,
    use_step_cache_pack_cached: bool,
    has_decode_row: bool,
    has_prefill_row: bool,
    has_compact_row: bool,
    refresh_slots: Sequence[int],
    is_prefill_by_row: List[bool],
    needs_logits_by_row: List[bool],
    use_compact_by_row: List[bool],
    logf_producer_by_row: List[int],
    logits_last_n_by_row: List[int],
    logits_spec_by_row: List[Optional[LogitSpec]],
    log_f_denom_ptr_by_row: List[int],
    compact_kv_len_by_row: List[int],
    compact_offsets: Optional[List[int]],
    slot_by_row: List[int],
    row_mode_signature: Tuple[int, ...],
    logits_rows_cached: Optional[Tuple[int, ...]],
    logits_rows_gt1_cached: Optional[Tuple[int, ...]],
    refresh_layout_for_log_f_capture: Optional["StepCaptureLayout"],
    slot_in_chunk_for_log_f_capture: int,
    q_len_by_row: Sequence[int],
    context_kv_len_by_row: Sequence[int],
    batch_size: int,
    block_size: int,
    num_heads: int,
    device: torch.device,
    key_cache: torch.Tensor,
    seqused_k: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    dense_log_f_has_last_n_gt1: bool,
) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    cache_key = int(key_cache.data_ptr())
    global_req_meta = _get_global_decode_req_meta(controller, cache_key) if is_decode else None
    global_prefill_meta = _get_global_prefill_req_meta(controller, cache_key) if (not is_decode) else None
    if is_decode and global_req_meta is None:
        raise RuntimeError(
            "decode dispatch requires step-bound global req_meta; "
            f"cache_key={cache_key} batch_size={batch_size}"
        )
    if global_req_meta is not None:
        req_meta_i32, req_meta_i64 = _pack_meta_from_global_decode(
            global_req_meta=global_req_meta,
        )
        return req_meta_i32, req_meta_i64, False
    if global_prefill_meta is not None:
        prefill_meta = _pack_meta_from_global_prefill(
            controller=controller,
            global_prefill_meta=global_prefill_meta,
            step_context=step_context,
            batch_size=int(batch_size),
            is_prefill_by_row=is_prefill_by_row,
            has_compact_row=bool(has_compact_row),
            refresh_slots=refresh_slots,
            dense_log_f_has_last_n_gt1=bool(dense_log_f_has_last_n_gt1),
            needs_logits_by_row=needs_logits_by_row,
            use_compact_by_row=use_compact_by_row,
            logits_last_n_by_row=logits_last_n_by_row,
            num_heads=int(num_heads),
            device=device,
        )
        if prefill_meta is not None:
            req_meta_i32, req_meta_i64 = prefill_meta
            return req_meta_i32, req_meta_i64, False
    req_meta_i32, req_meta_i64 = _pack_meta_from_local_plan(
        controller=controller,
        state=state,
        step_context=step_context,
        step_envelope=step_envelope,
        use_fast_pack=bool(use_fast_pack),
        use_step_cache_pack_cached=bool(use_step_cache_pack_cached),
        has_decode_row=bool(has_decode_row),
        has_prefill_row=bool(has_prefill_row),
        is_prefill_by_row=is_prefill_by_row,
        batch_size=int(batch_size),
        block_size=int(block_size),
        num_heads=int(num_heads),
        device=device,
        seqused_k=seqused_k,
        cu_seqlens_q=cu_seqlens_q,
        needs_logits_by_row=needs_logits_by_row,
        use_compact_by_row=use_compact_by_row,
        logf_producer_by_row=logf_producer_by_row,
        logits_last_n_by_row=logits_last_n_by_row,
        logits_spec_by_row=logits_spec_by_row,
        log_f_denom_ptr_by_row=log_f_denom_ptr_by_row,
        compact_kv_len_by_row=compact_kv_len_by_row,
        compact_offsets=compact_offsets,
        slot_by_row=slot_by_row,
        row_mode_signature=row_mode_signature,
        logits_rows_cached=logits_rows_cached,
        logits_rows_gt1_cached=logits_rows_gt1_cached,
        refresh_layout_for_log_f_capture=refresh_layout_for_log_f_capture,
        slot_in_chunk_for_log_f_capture=int(slot_in_chunk_for_log_f_capture),
        q_len_by_row=q_len_by_row,
        context_kv_len_by_row=context_kv_len_by_row,
    )
    return req_meta_i32, req_meta_i64, True
