"""
patches/controller_mixins/selector_compute_mixin.py — Selector computation, buffer management, and tracking.

OWNS:
  - _init_selector_compute_state(): selector compute state initialization
  - _get_positions_i32 / _get_positions_i64 / _get_window_idx_i32 / _get_tail_offsets(): cached index tensors
  - _get_row_index_tensor / _get_cached_logits_patch_i32(): cached helper tensors
  - _reorder_selected_indices_physical_block_major(): block-major reorder for compact copy
  - _ensure_selector_key_norms_buffer / _ensure_selector_capture_scores_buffer: per-layer buffer pools
  - _ensure_selector_log_f_denoms_buffer / _ensure_selector_kv_lengths_buffer: per-layer buffer pools
  - _ensure_log_f_workspace(): scratch workspace for log-f computation

DEPENDS_ON:
  - CaptureRingMixin: _map_global_layer_to_capture_slot, _ensure_capture_layout_cpu_tensors,
    _capture_scores_ptrs_for_rows, _log_f_denoms_ptrs_for_rows
  - ProfileMixin: _refresh_profile_should_sample, _step_profile_record_refresh_payload
  - CompactKVMixin: _ensure_compact_capacity, _reset_compact_slot
  - Main controller: _get_step_capture_layout, _rebuild_compact_slots_batched_layers_from_selection,
    _compute_alpha_selection_pipeline_unified

ENTRY_POINTS:
  - _init_selector_compute_state(): called from VLLMSparseController.__init__
"""
from __future__ import annotations

import os
import time
from typing import (
    Any,
    TYPE_CHECKING,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import torch

from hybrid_selectors.alpha_fair_selector import (
    AlphaFairSelectorConfig,
    apply_cross_head_mutex_bounds,
    compute_alpha_scores_batched_layers_triton_logf_bounds,
    compute_alpha_scores_batched_layers_triton_logf_pre_denom_bounds,
)
from patches.selector_runtime.batched_selection import (
    compute_alpha_selection_batched_impl,
)
from patches.selector_runtime.selected_out_ring import (
    SelectedOutRing,
    SlotStableOverrides,
)
from patches.request_intent_ticket import (
    PendingPolicy,
    PendingReasonCode,
    pending_reason_code_to_text,
)
from patches.fa_sparse_runtime.materialize import (
    project_selected_token_indices_to_logical_pages,
)
from patches.fa_sparse_runtime.compact_recent_alignment import (
    compact_recent_effective_k_head,
)
from patches.sparse_types import (
    SelectorBatchPayload,
    SelectorResult,
)
from patches.sparse_constants import (
    _DECODE_BOUNDS_KERNEL_CACHED,
    _REBUILD_PHYSICAL_BLOCK_SORT_CACHED,
    _SELECTOR_CPP_PREPROC_CACHED,
    _SELECTOR_CPP_STACK_CACHED,
    _SELECTOR_FAST_SIG_CACHED,
    _SELECTOR_KBUCKET_CACHED,
    _SELECTOR_KEY_NORMS_CACHE_CAP_CACHED,
    _SELECTOR_PIPELINE_UNIFIED_CACHED,
    _SELECTOR_TRUSTED_SHAPES_CACHED,
    _selector_graph_lru_enabled,
    _is_free_slot_id,
    should_skip_page_sparse_state,
)
from patches.sparse_utils import (
    _align_up_int,
    _get_row_index_tensor_from_cache,
    _get_selector_batch_ext,
    _is_stream_capturing_or_raise,
)
try:
    from vllm.logger import init_logger
except Exception:
    init_logger = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from patches.layer_state import LayerState
    from patches.sparse_types import (
        LogitSpec,
        RequestTracking,
        StepContext,
    )

if init_logger is not None:
    _log = init_logger(__name__)
else:
    import logging
    _log = logging.getLogger(__name__)




def _should_use_cpp_preproc_for_selection_mode(
    *,
    selection_mode: str,
    cpp_preproc_enabled: bool,
) -> bool:
    if str(selection_mode) not in {
        "token_topk",
    }:
        raise ValueError(f"unsupported alpha selection_mode: {selection_mode}")
    return bool(cpp_preproc_enabled) and str(selection_mode) == "token_topk"


def _compute_use_cpp_preproc(
    *,
    selection_mode: str,
) -> bool:
    return _should_use_cpp_preproc_for_selection_mode(
        selection_mode=selection_mode,
        cpp_preproc_enabled=_SELECTOR_CPP_PREPROC_CACHED,
    )


def _should_use_unified_pipeline_for_selection_mode(
    *,
    selection_mode: str,
    unified_pipeline_enabled: bool,
) -> bool:
    if str(selection_mode) not in {
        "token_topk",
    }:
        raise ValueError(f"unsupported alpha selection_mode: {selection_mode}")
    return bool(unified_pipeline_enabled) and str(selection_mode) == "token_topk"


def _normalize_selection_layers_result(
    result,
):
    if not isinstance(result, tuple):
        raise TypeError("alpha selection layers result must be a tuple")
    if len(result) == 9:
        return result
    if len(result) == 8:
        (
            selected_indices_batch,
            head_sink,
            recent_start,
            kv_len_head,
            allowed_lengths,
            selected_middle_pages,
            selected_middle_counts,
            profile_events,
        ) = result
        return (
            selected_indices_batch,
            head_sink,
            recent_start,
            kv_len_head,
            allowed_lengths,
            selected_middle_pages,
            selected_middle_counts,
            None,
            profile_events,
        )
    if len(result) == 6:
        (
            selected_indices_batch,
            head_sink,
            recent_start,
            kv_len_head,
            allowed_lengths,
            profile_events,
        ) = result
        return (
            selected_indices_batch,
            head_sink,
            recent_start,
            kv_len_head,
            allowed_lengths,
            None,
            None,
            None,
            profile_events,
        )
    raise ValueError(f"unexpected alpha selection layers result length: {len(result)}")


class SelectorComputeMixin:
    """Mixin: selector computation, buffer management, and tracking."""

    _MIXIN_REQUIRES: tuple = ("CaptureRingMixin", "ProfileMixin", "CompactKVMixin")

    def _init_selector_compute_state(self) -> None:
        """Initialize selector compute state variables."""
        # #13 STAGE-0 selector-topk captured graph (default OFF). Disjoint
        # namespace from the writer graph: own state dict / mempool / stream /
        # thrash window. None resting state -> the dispatcher allocates lazily.
        self._selector_topk_graph_state: Optional[Dict[str, object]] = None
        self._selector_topk_graph_stream = None
        self._selector_topk_graph_recapture_window: int = 0
        self._selector_topk_graph_recapture_count: int = 0
        # Real (graph-agnostic) replay counter, bumped by the dispatcher in its
        # replay branch -> observable proof the captured graph actually replayed,
        # independent of whatever graph object is cached (mirrors the writer
        # track's _record_deadline_async_producer_count("graph_replay")).
        self._selector_topk_graph_replay_count: int = 0
        self._selector_key_norms_all: Optional[torch.Tensor] = None
        self._selector_key_norms_shape: Optional[Tuple[int, int, int, int]] = None
        self._selector_key_norms_all_reallocated: bool = False
        self._selector_key_norms_all_cache: Dict[object, torch.Tensor] = {}
        self._selector_key_norms_all_cache_override: Optional[
            Dict[object, torch.Tensor]
        ] = None
        self._selector_key_norms_all_active_cache_key: object = None
        self._selector_key_norms_all_valid_cache_keys: Set[object] = set()
        self._selector_key_norms_all_valid_cache_keys_override: Optional[
            Set[object]
        ] = None
        self._selector_capture_scores_all: Optional[torch.Tensor] = None
        self._selector_log_f_denoms_all: Optional[torch.Tensor] = None
        self._selector_kv_lengths_all: Optional[torch.Tensor] = None
        self._selector_key_norms_delta_start_cpu: Optional[torch.Tensor] = None
        self._selector_key_norms_delta_end_cpu: Optional[torch.Tensor] = None
        self._selector_key_norms_delta_start_gpu: Optional[torch.Tensor] = None
        self._selector_key_norms_delta_end_gpu: Optional[torch.Tensor] = None
        self._selector_key_norms_delta_buffer_override: Optional[
            object
        ] = None
        self._selector_key_norms_target_cpu: Optional[torch.Tensor] = None
        self._selector_decode_bounds_key: Optional[Tuple[object, ...]] = None
        self._selector_decode_bounds_buffers: Optional[Tuple[torch.Tensor, ...]] = None
        self._selector_decode_bounds_buffers_override: Optional[
            Dict[Tuple[object, ...], Tuple[torch.Tensor, ...]]
        ] = None
        self._selector_selected_indices_out_key: Optional[Tuple[object, ...]] = None
        self._selector_selected_indices_out: Optional[torch.Tensor] = None
        self._selector_selected_indices_out_override: Optional[
            Dict[Tuple[object, ...], torch.Tensor]
        ] = None
        # [SELECTED-OUT-RING 2026-07-09] pending 路径稳定环(懒构造,见
        # refresh_rebuild_mixin._selected_out_ring_for_pending;env 关=None)。
        self._selected_out_ring: Optional[SelectedOutRing] = None
        # ASYNC_PRODUCER_WRITER_GRAPH: persistent (layers, max_batch) int32 row-tensor buffer so the
        # captured writer reads a STABLE data_ptr on replay (defect #6).
        self._selector_writer_row_tensor_all_key: Optional[Tuple[object, ...]] = None
        self._selector_writer_row_tensor_all: Optional[torch.Tensor] = None
        # ASYNC_PRODUCER_WRITER_GRAPH (#9-KEY): persistent flat int32 [batch] seq_lens buffer so the
        # captured writer reads a STABLE seq_lens data_ptr on replay (was reconstructed
        # fresh every refresh -> key churn -> replay=0).
        self._selector_writer_seq_lens_all_key: Optional[Tuple[object, ...]] = None
        self._selector_writer_seq_lens_all: Optional[torch.Tensor] = None
        # ASYNC_PRODUCER_WRITER_GRAPH (#9-KEY v2): persistent flat int32 [batch] slot_tensor
        # buffer so the captured writer reads a STABLE slot_tensor data_ptr on replay
        # (slot_tensor_i32 is a fresh per-refresh payload carrier -> key field [4] churned 31/31).
        self._selector_writer_slot_tensor_all_key: Optional[Tuple[object, ...]] = None
        self._selector_writer_slot_tensor_all: Optional[torch.Tensor] = None
        # ASYNC_PRODUCER_WRITER_GRAPH (#9-KEY v2): persistent flat int32 buffer for the post-topk
        # selected_indices [L,B,H,k] so the captured writer reads a STABLE selected data_ptr on
        # replay (the shape-keyed _selector_selected_indices_out buffer reallocs on k_head/batch
        # jitter -> key field [1] churned 19/31).
        self._selector_writer_selected_all_key: Optional[Tuple[object, ...]] = None
        self._selector_writer_selected_all: Optional[torch.Tensor] = None
        # ASYNC_PRODUCER_WRITER_GRAPH (#9-KEY v3): defensive ready latch — event recorded after the
        # writer-input copies on their producing stream, drained before replay (same-stream no-op;
        # guards a future side-stream refactor). See _record_writer_input_ready / _wait_writer_input_ready.
        self._selector_writer_input_ready_event = None
        self._log_f_workspace_key: Optional[Tuple[object, ...]] = None
        self._log_f_scratch_workspace: Optional[torch.Tensor] = None
        self._log_f_scratch_workspace_override: Optional[
            Dict[Tuple[object, ...], torch.Tensor]
        ] = None
        self._selector_pipeline_workspace_key: Optional[Tuple[object, ...]] = None
        self._selector_pipeline_workspace_a: Optional[torch.Tensor] = None
        self._selector_pipeline_workspace_b: Optional[torch.Tensor] = None
        self._selector_pipeline_workspace_override: Optional[
            Dict[Tuple[object, ...], Tuple[torch.Tensor, torch.Tensor]]
        ] = None
        self._selector_layer_index_cache_key: Optional[Tuple[int, ...]] = None
        self._selector_layer_index_cache_device: Optional[torch.device] = None
        self._selector_layer_index_cache_tensor: Optional[torch.Tensor] = None
        self._positions_i32_cache_device: Optional[torch.device] = None
        self._positions_i32_cache_cap: int = 0
        self._positions_i32_cache: Optional[torch.Tensor] = None
        self._positions_i64_cache_device: Optional[torch.device] = None
        self._positions_i64_cache_cap: int = 0
        self._positions_i64_cache: Optional[torch.Tensor] = None
        self._window_idx_cache_device: Optional[torch.device] = None
        self._window_idx_cache_cap: int = 0
        self._window_idx_cache: Optional[torch.Tensor] = None
        self._tail_offsets_cache_device: Optional[torch.device] = None
        self._tail_offsets_cache_cap: int = 0
        self._tail_offsets_cache: Optional[torch.Tensor] = None
        self._row_index_cache: Dict[Tuple[str, int, Tuple[int, ...]], torch.Tensor] = {}
        self._logits_patch_cache_key: Optional[Tuple[object, ...]] = None
        self._logits_patch_last_n_i32: Optional[torch.Tensor] = None
        self._logits_patch_row_offsets_i32: Optional[torch.Tensor] = None
        self._logits_patch_caps_i32: Optional[torch.Tensor] = None
        self._logits_patch_rows_gt1: Optional[Tuple[int, ...]] = None
        self._decode_log_f_mask_i32: Optional[torch.Tensor] = None  # [max_batch], int32
        self._decode_q_lens_i32: Optional[torch.Tensor] = None      # [max_batch], int32
        self._decode_logits_cap_i64: Optional[torch.Tensor] = None   # [max_batch], int64
        self._decode_logits_last_n_i64: Optional[torch.Tensor] = None  # [max_batch], int64
        self._decode_logits_last_n_stage_cpu_i64: Optional[torch.Tensor] = None
        self._decode_logits_cap_stage_cpu_i64: Optional[torch.Tensor] = None
        self._step_logits_ready_token: int = -1
        self._step_logits_ready_input_signature: Optional[Tuple[object, ...]] = None
        self._step_logits_ready_bound_signature: Optional[Tuple[object, ...]] = None
        self._decode_row_is_compact_i32: Optional[torch.Tensor] = None  # [max_batch], int32
        self._decode_compact_staging_cpu: Optional[torch.Tensor] = None  # [max_batch], int32, CPU
        self._decode_seqused_k_i32: Optional[torch.Tensor] = None  # [max_batch], int32
        self._decode_cu_seqlens_q_i32: Optional[torch.Tensor] = None  # [max_batch+1], int32
        self.step_prefill_plan_epoch: int = -1
        self.step_prefill_plan_handle_id: int = -1
        self.step_prefill_plan_handle_generation: int = -1
        self.step_prefill_capture_plan_by_req: Dict[str, int] = {}
        self.step_prefill_finalize_req_ids: Tuple[str, ...] = tuple()
        self.step_prefill_capture_last_n_by_row: Optional[Tuple[int, ...]] = None
        self.step_prefill_capture_last_n_epoch: int = -1
        self.step_prefill_capture_last_n_handle_id: int = -1
        self.step_prefill_capture_last_n_handle_generation: int = -1
        self._step_context_slot_row_map_token: int = -1
        self._step_context_slot_row_map_key: Tuple[int, ...] = tuple()
        self._step_context_slot_row_map: Optional[Dict[int, int]] = None

    @staticmethod
    def _tensor_bytes(tensor: Optional[torch.Tensor]) -> int:
        if tensor is None or not isinstance(tensor, torch.Tensor):
            return 0
        try:
            return int(tensor.numel()) * int(tensor.element_size())
        except Exception:
            _log.warning("_tensor_bytes failed for tensor %s", type(tensor), exc_info=True)
            raise

    def _get_positions_i32(self, *, kv_len: int, device: torch.device) -> torch.Tensor:
        """Return positions [kv_len] int32, cached with over-allocation."""
        if kv_len <= 0:
            return torch.empty((0,), device=device, dtype=torch.int32)
        cap = int(kv_len)
        cached = self._positions_i32_cache
        if cached is None or self._positions_i32_cache_device != device or self._positions_i32_cache_cap < cap:
            if cached is not None and cached.is_cuda:
                # [POSITIONS-CACHE-REALLOC-UAF-FIX] P2-d:容量出窗换代弃旧,
                # 消费者(capture_row scatter/selector pipeline)双流上下文。
                # 容量单调增长=冷事件。
                self._uaf_guard_record_streams_before_discard(cached)
            # over-allocate to reduce realloc under growing kv_len
            new_cap = _align_up_int(cap, 256)
            cached = torch.arange(new_cap, device=device, dtype=torch.int32)
            self._positions_i32_cache = cached
            self._positions_i32_cache_device = device
            self._positions_i32_cache_cap = int(new_cap)
        return cached[:cap]

    def _get_positions_i64(self, *, kv_len: int, device: torch.device) -> torch.Tensor:
        """Return positions [kv_len] int64, cached with over-allocation."""
        if kv_len <= 0:
            return torch.empty((0,), device=device, dtype=torch.int64)
        cap = int(kv_len)
        cached = self._positions_i64_cache
        if cached is None or self._positions_i64_cache_device != device or self._positions_i64_cache_cap < cap:
            if cached is not None and cached.is_cuda:
                # [POSITIONS-CACHE-REALLOC-UAF-FIX] P2-d:同族换代守卫。
                self._uaf_guard_record_streams_before_discard(cached)
            new_cap = _align_up_int(cap, 256)
            cached = torch.arange(new_cap, device=device, dtype=torch.int64)
            self._positions_i64_cache = cached
            self._positions_i64_cache_device = device
            self._positions_i64_cache_cap = int(new_cap)
        return cached[:cap]


    def _get_tail_offsets(self, *, window: int, device: torch.device) -> torch.Tensor:
        """Return tail_offsets [1,1,1,1,window] int32, cached with over-allocation.

        tail_offsets[i] = window - 1 - i，即 [window-1, window-2, ..., 1, 0]
        """
        if window <= 0:
            return torch.empty((1, 1, 1, 1, 0), device=device, dtype=torch.int32)
        cap = int(window)
        cached = self._tail_offsets_cache
        if cached is None or self._tail_offsets_cache_device != device or self._tail_offsets_cache_cap < cap:
            if cached is not None and cached.is_cuda:
                # [POSITIONS-CACHE-REALLOC-UAF-FIX] P2-d:同族换代守卫(view
                # 的 record_stream 作用于底层 storage)。
                self._uaf_guard_record_streams_before_discard(cached)
            new_cap = _align_up_int(cap, 64)
            window_idx = torch.arange(new_cap, device=device, dtype=torch.int32)
            # 存储 [new_cap-1, new_cap-2, ..., 1, 0]
            cached = (int(new_cap) - 1 - window_idx).view(1, 1, 1, 1, new_cap)
            self._tail_offsets_cache = cached
            self._tail_offsets_cache_device = device
            self._tail_offsets_cache_cap = int(new_cap)
        # 从末尾取 window 个元素: [..., cap-1, cap-2, ..., 1, 0] 中取最后 cap 个
        # 但 cached 存的是 [new_cap-1, ..., 0]，所以末尾 cap 个是 [cap-1, ..., 0]
        # 需要从 index (new_cap - cap) 开始取
        start_idx = self._tail_offsets_cache_cap - cap
        return cached[..., start_idx:]

    def _get_row_index_tensor(self, *, rows: Sequence[int], device: torch.device) -> torch.Tensor:
        return _get_row_index_tensor_from_cache(
            self._row_index_cache, rows=rows, device=device, cache_owner=self
        )


    @staticmethod
    def _compute_selected_indices_physical_block_major_order(
        *,
        selected_indices: torch.Tensor,
        block_table: torch.Tensor,
        row_tensor_i32: torch.Tensor,
        block_size: int,
    ) -> Optional[torch.Tensor]:
        if selected_indices.numel() == 0:
            return None
        if selected_indices.dim() != 4:
            return None
        if block_size <= 0:
            return None
        if block_table.numel() == 0 or block_table.dim() != 2:
            return None
        if block_table.dtype != torch.int32:
            return None
        if row_tensor_i32.numel() == 0:
            return None

        device = selected_indices.device
        if block_table.device != device:
            return None

        layers, batch, _, k = selected_indices.shape
        max_rows = int(block_table.shape[0])
        if max_rows <= 0:
            return None
        cols = int(block_table.shape[1])
        if cols <= 0:
            return None
        if row_tensor_i32.device != device:
            row_tensor_i32 = row_tensor_i32.to(device=device)
        if row_tensor_i32.dtype != torch.int32:
            row_tensor_i32 = row_tensor_i32.to(dtype=torch.int32)
        if int(row_tensor_i32.numel()) < int(batch):
            return None
        row_tensor_i32 = row_tensor_i32[:batch]

        row_valid = (row_tensor_i32 >= 0) & (row_tensor_i32 < max_rows)
        rows_safe = row_tensor_i32.clamp(min=0, max=max(0, max_rows - 1)).to(dtype=torch.long)
        bt_rows = block_table.index_select(0, rows_safe)
        bt_rows_exp = bt_rows.unsqueeze(0).expand(int(layers), -1, -1)

        sel = selected_indices
        valid = sel >= 0
        pos_safe = sel.clamp(min=0).to(dtype=torch.int64)
        block_idx = torch.div(pos_safe, int(block_size), rounding_mode="floor")
        block_idx = block_idx.clamp(min=0, max=max(0, cols - 1)).to(dtype=torch.int64)

        block_idx_flat = block_idx.reshape(int(layers), int(batch), -1)
        phys_flat = bt_rows_exp.gather(2, block_idx_flat)
        phys = phys_flat.reshape(int(layers), int(batch), -1, int(k)).to(dtype=torch.int64)

        offset = torch.remainder(pos_safe, int(block_size))
        key = phys * int(block_size) + offset

        invalid = (~valid) | (phys < 0)
        if not bool(row_valid.all()):
            invalid = invalid | (~row_valid.view(1, int(batch), 1, 1))
        key_sentinel = (1 << 62)
        key = key.masked_fill(invalid, int(key_sentinel))
        return torch.argsort(key, dim=-1)

    def _get_cached_logits_patch_i32(
        self,
        *,
        epoch: int,
        logits_rows: Sequence[int],
        logits_last_n_by_row: Sequence[int],
        q_len_by_row: Sequence[int],
        context_kv_len_by_row: Sequence[int],
        kv_max: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Tuple[int, ...]]:
        """为 refresh/log_f 的 meta patch 复用 (last_n,row_offset,capacity) 小张量。

        这些值在同一 step 内跨层不变（仅依赖 row/q_len/seq_len/kv_max），适合按 step-epoch 缓存。
        """
        rows_tuple = tuple(int(r) for r in logits_rows)
        last_n_tuple = tuple(int(logits_last_n_by_row[r]) for r in rows_tuple)
        q_len_tuple = tuple(int(q_len_by_row[r]) for r in rows_tuple)
        ctx_len_tuple = tuple(int(context_kv_len_by_row[r]) for r in rows_tuple)
        dev_index = int(device.index) if device.index is not None else -1
        key = (
            int(epoch),
            str(device.type),
            dev_index,
            int(kv_max),
            rows_tuple,
            last_n_tuple,
            q_len_tuple,
            ctx_len_tuple,
        )
        if self._logits_patch_cache_key == key:
            if (
                self._logits_patch_last_n_i32 is not None
                and self._logits_patch_row_offsets_i32 is not None
                and self._logits_patch_caps_i32 is not None
                and self._logits_patch_rows_gt1 is not None
            ):
                row_index = self._get_row_index_tensor(rows=rows_tuple, device=device)
                return (
                    row_index,
                    self._logits_patch_last_n_i32,
                    self._logits_patch_row_offsets_i32,
                    self._logits_patch_caps_i32,
                    self._logits_patch_rows_gt1,
                )

        row_offsets = [max(0, int(q_len_tuple[i]) - int(last_n_tuple[i])) for i in range(len(rows_tuple))]
        caps = [min(int(ctx_len_tuple[i]), int(kv_max)) for i in range(len(rows_tuple))]
        last_n_i32 = torch.tensor(last_n_tuple, dtype=torch.int32, device=device)
        row_offsets_i32 = torch.tensor(row_offsets, dtype=torch.int32, device=device)
        caps_i32 = torch.tensor(caps, dtype=torch.int32, device=device)
        rows_gt1 = tuple(int(r) for r in rows_tuple if int(logits_last_n_by_row[int(r)]) > 1)

        self._logits_patch_cache_key = key
        self._logits_patch_last_n_i32 = last_n_i32
        self._logits_patch_row_offsets_i32 = row_offsets_i32
        self._logits_patch_caps_i32 = caps_i32
        self._logits_patch_rows_gt1 = rows_gt1

        row_index = self._get_row_index_tensor(rows=rows_tuple, device=device)
        return row_index, last_n_i32, row_offsets_i32, caps_i32, rows_gt1

    def _selector_trusted_shapes_enabled(self, *, phase: str) -> bool:
        if phase != "decode":
            return False
        return bool(_SELECTOR_TRUSTED_SHAPES_CACHED)

    def _selector_fast_sig_enabled(self, *, phase: str) -> bool:
        if phase != "decode":
            return False
        return bool(_SELECTOR_FAST_SIG_CACHED)

    def _ensure_selector_key_norms_buffer(
        self,
        *,
        layers: int,
        batch: int,
        num_kv_heads: int,
        kv_len: int,
        device: torch.device,
        dtype: torch.dtype,
        cache_key: object = None,
    ) -> torch.Tensor:
        if layers <= 0 or batch <= 0 or num_kv_heads <= 0 or kv_len <= 0:
            return torch.empty((0, 0, 0, 0), device=device, dtype=dtype)
        # Reserve one K bucket ahead. Sentence refresh commonly alternates
        # between adjacent capture K buckets; exact-size allocation forces a
        # grow and then a sidecar pack when the request crosses the next bucket.
        kv_capacity = _align_up_int(int(kv_len) + 1, 4096)
        shape = (layers, batch, num_kv_heads, kv_len)
        alloc_shape = (layers, batch, num_kv_heads, kv_capacity)
        self._selector_key_norms_all_active_cache_key = cache_key
        override = getattr(self, "_selector_key_norms_all_cache_override", None)
        if isinstance(override, dict):
            # [SELECTED-OUT-RING v2] 槽稳定容器按**容量**键控:语义 cache_key
            # (含层组标识/slot_list)每世代漂移,per-run 私有 dict 下无所谓
            # (dict 一次性),槽持久 dict 下=每代插新 buffer=graph key 的
            # key_norms 指针永新(rv8 keys 取证:其余 9 指针全 3 槽周期,唯
            # key_norms 24 唯一≈每世代一个)。环模式 valid 集每 run 清空=
            # 必重填,语义键的跨 run 复用本已不存在,按容量复用值语义相同。
            if isinstance(override, SlotStableOverrides) or cache_key is None:
                override_key = (
                    "__slot_capacity__" if isinstance(override, SlotStableOverrides) else "__default__",
                    str(device),
                    str(dtype),
                    tuple(int(v) for v in alloc_shape),
                )
            else:
                override_key = cache_key
            buf = override.get(override_key)
            self._selector_key_norms_all_reallocated = False
            if (
                buf is None
                or buf.device != device
                or buf.dtype != dtype
                or buf.shape[0] < shape[0]
                or buf.shape[1] < shape[1]
                or buf.shape[2] < shape[2]
                or buf.shape[3] < shape[3]
            ):
                if buf is not None and buf.is_cuda:
                    # [SELECTED-OUT-RING v2] 槽稳定容器换代弃旧走守卫。
                    self._uaf_guard_record_streams_before_discard(buf)
                buf = torch.empty(alloc_shape, device=device, dtype=dtype)
                override[override_key] = buf
                valid_keys = getattr(
                    self,
                    "_selector_key_norms_all_valid_cache_keys_override",
                    None,
                )
                if isinstance(valid_keys, set):
                    valid_keys.discard(cache_key)
                self._selector_key_norms_all_reallocated = True
            return buf[:layers, :batch, :num_kv_heads, :kv_len]
        if cache_key is None:
            buf = self._selector_key_norms_all
        else:
            cache = getattr(self, "_selector_key_norms_all_cache", None)
            if not isinstance(cache, dict):
                cache = {}
                self._selector_key_norms_all_cache = cache
            buf = cache.get(cache_key)
        self._selector_key_norms_all_reallocated = False
        if (
            buf is None
            or buf.device != device
            or buf.dtype != dtype
            or buf.shape[0] < shape[0]
            or buf.shape[1] < shape[1]
            or buf.shape[2] < shape[2]
            or buf.shape[3] < shape[3]
        ):
            if buf is not None and buf.is_cuda:
                # [SELECTOR-SHARED-SCRATCH-REALLOC-UAF-FIX] R5:共享 key_norms
                # 载体被主流 drain-resolve 与 refresh_stream flush/deferred 双
                # 上下文交替读写,同键换代弃旧时另一流可有在飞读者;prune 臂
                # (_prune_selector_key_norms_cache)已守,替换臂补齐。冷事件。
                self._uaf_guard_record_streams_before_discard(buf)
            buf = torch.empty(alloc_shape, device=device, dtype=dtype)
            if cache_key is not None:
                self._selector_key_norms_all_cache[cache_key] = buf
            valid_keys = getattr(
                self,
                "_selector_key_norms_all_valid_cache_keys",
                None,
            )
            if isinstance(valid_keys, set):
                valid_keys.discard(cache_key)
            self._selector_key_norms_all_reallocated = True
        self._selector_key_norms_all = buf
        self._selector_key_norms_shape = buf.shape
        if cache_key is not None:
            self._prune_selector_key_norms_cache(active_key=cache_key, device=device)
        return buf[:layers, :batch, :num_kv_heads, :kv_len]

    def _prune_selector_key_norms_cache(self, *, active_key, device) -> None:
        """Bound the per-shape key_norms_all GPU buffer cache (LRU, async-safe free).

        The cache holds one [layers, batch, kv_heads, kv_capacity] fp16 buffer per
        distinct (phase, layer_indices, slots, batch, kv_bucket) key. Under a general
        server load (variable context lengths -> many kv_buckets, churning slot sets)
        it would otherwise grow unbounded -- each buffer is multi-GiB at long contexts,
        OOMing outside vLLM's util budget. Keep only the active key plus a small LRU
        window; release evicted buffers via record_stream on the current + refresh
        streams (deferred free, no cross-stream UAF), never during graph capture. The
        active set is always retained, so incremental reuse / output bytes are unchanged.
        """
        cache = getattr(self, "_selector_key_norms_all_cache", None)
        if not isinstance(cache, dict) or not cache:
            return
        # Mark active_key most-recently-used (dict preserves insertion order).
        if active_key in cache:
            cache[active_key] = cache.pop(active_key)
        cap = int(_SELECTOR_KEY_NORMS_CACHE_CAP_CACHED)
        if cap < 1:
            cap = 1
        if len(cache) <= cap:
            return
        # [GUARD-NO-SWALLOW] capture 态查询失败若吞掉则会在 capture 内 free
        # （本函数要防的事故本身）；流解析失败若吞成 cur=None 则跳过
        # record_stream=异步 UAF。两者都必须炸。
        if _is_stream_capturing_or_raise(stage="selector_key_norms_cache_prune"):
            return  # never free during capture; prune on a later eager call
        dev = torch.device(device)
        cur = torch.cuda.current_stream(device=dev) if dev.type == "cuda" else None
        refresh_stream = getattr(self, "refresh_stream", None)
        valid_keys = getattr(self, "_selector_key_norms_all_valid_cache_keys", None)
        for key in list(cache.keys()):
            if len(cache) <= cap:
                break
            if key == active_key:
                continue
            old = cache.pop(key)
            if isinstance(valid_keys, set):
                valid_keys.discard(key)
            if (
                isinstance(old, torch.Tensor)
                and old.numel() > 0
                and old.device.type == "cuda"
            ):
                if cur is not None:
                    old.record_stream(cur)
                if refresh_stream is not None:
                    # [GUARD-NO-SWALLOW] refresh_stream 是 selector/writer 异步
                    # 消费流；record 失败若吞掉=弃旧无延迟释放护栏=跨流 UAF。
                    old.record_stream(refresh_stream)
            del old

    def _selector_key_norms_active_buffer_valid(self) -> bool:
        override_valid_keys = getattr(
            self,
            "_selector_key_norms_all_valid_cache_keys_override",
            None,
        )
        if isinstance(override_valid_keys, set):
            return (
                getattr(self, "_selector_key_norms_all_active_cache_key", None)
                in override_valid_keys
            )
        valid_keys = getattr(self, "_selector_key_norms_all_valid_cache_keys", None)
        if not isinstance(valid_keys, set):
            return False
        return getattr(self, "_selector_key_norms_all_active_cache_key", None) in valid_keys

    def _mark_selector_key_norms_active_buffer_valid(self) -> None:
        override_valid_keys = getattr(
            self,
            "_selector_key_norms_all_valid_cache_keys_override",
            None,
        )
        if isinstance(override_valid_keys, set):
            override_valid_keys.add(
                getattr(self, "_selector_key_norms_all_active_cache_key", None)
            )
            return
        valid_keys = getattr(self, "_selector_key_norms_all_valid_cache_keys", None)
        if not isinstance(valid_keys, set):
            valid_keys = set()
            self._selector_key_norms_all_valid_cache_keys = valid_keys
        valid_keys.add(getattr(self, "_selector_key_norms_all_active_cache_key", None))

    def _ensure_selector_key_norms_delta_buffers(
        self,
        *,
        layers: int,
        batch: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """为 key_norms delta 复用 [layers, batch] 的 int32 staging 缓冲。"""
        if layers <= 0 or batch <= 0:
            empty_cpu = torch.empty((0, 0), device="cpu", dtype=torch.int32)
            empty_gpu = torch.empty((0, 0), device=device, dtype=torch.int32)
            return empty_cpu, empty_cpu, empty_gpu, empty_gpu
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        shape = (int(layers), int(batch))
        override = getattr(self, "_selector_key_norms_delta_buffer_override", None)
        if isinstance(override, dict):
            key = (
                str(device.type),
                int(device.index) if device.index is not None else -1,
                shape,
            )
            buffers = override.get(key)
            if (
                not isinstance(buffers, tuple)
                or len(buffers) != 4
                or any(not isinstance(t, torch.Tensor) for t in buffers)
            ):
                buffers = None
            if buffers is not None:
                cpu_start_o, cpu_end_o, gpu_start_o, gpu_end_o = buffers
            else:
                cpu_start_o = cpu_end_o = gpu_start_o = gpu_end_o = None
            if not (
                isinstance(cpu_start_o, torch.Tensor)
                and isinstance(cpu_end_o, torch.Tensor)
                and isinstance(gpu_start_o, torch.Tensor)
                and isinstance(gpu_end_o, torch.Tensor)
                and cpu_start_o.device.type == "cpu"
                and cpu_end_o.device.type == "cpu"
                and cpu_start_o.dtype == torch.int32
                and cpu_end_o.dtype == torch.int32
                and cpu_start_o.is_pinned()
                and cpu_end_o.is_pinned()
                and gpu_start_o.device == device
                and gpu_end_o.device == device
                and gpu_start_o.dtype == torch.int32
                and gpu_end_o.dtype == torch.int32
                and cpu_start_o.dim() == 2
                and cpu_end_o.dim() == 2
                and gpu_start_o.dim() == 2
                and gpu_end_o.dim() == 2
                and cpu_start_o.shape[0] >= shape[0]
                and cpu_start_o.shape[1] >= shape[1]
                and cpu_end_o.shape[0] >= shape[0]
                and cpu_end_o.shape[1] >= shape[1]
                and gpu_start_o.shape[0] >= shape[0]
                and gpu_start_o.shape[1] >= shape[1]
                and gpu_end_o.shape[0] >= shape[0]
                and gpu_end_o.shape[1] >= shape[1]
            ):
                # [SELECTED-OUT-RING v2] 槽稳定容器换代弃旧走守卫(GPU 对);
                # pinned CPU 对由 CachingHostAllocator 自动挂事件免守卫。
                for _old in (gpu_start_o, gpu_end_o):
                    if isinstance(_old, torch.Tensor) and _old.is_cuda:
                        self._uaf_guard_record_streams_before_discard(_old)
                cpu_start_o = torch.empty(
                    shape,
                    device="cpu",
                    dtype=torch.int32,
                    pin_memory=True,
                )
                cpu_end_o = torch.empty(
                    shape,
                    device="cpu",
                    dtype=torch.int32,
                    pin_memory=True,
                )
                gpu_start_o = torch.empty(shape, device=device, dtype=torch.int32)
                gpu_end_o = torch.empty(shape, device=device, dtype=torch.int32)
                override[key] = (cpu_start_o, cpu_end_o, gpu_start_o, gpu_end_o)
            return (
                cpu_start_o[: shape[0], : shape[1]],
                cpu_end_o[: shape[0], : shape[1]],
                gpu_start_o[: shape[0], : shape[1]],
                gpu_end_o[: shape[0], : shape[1]],
            )
        if isinstance(override, tuple) and len(override) == 4:
            cpu_start_o, cpu_end_o, gpu_start_o, gpu_end_o = override
            if (
                isinstance(cpu_start_o, torch.Tensor)
                and isinstance(cpu_end_o, torch.Tensor)
                and isinstance(gpu_start_o, torch.Tensor)
                and isinstance(gpu_end_o, torch.Tensor)
                and cpu_start_o.device.type == "cpu"
                and cpu_end_o.device.type == "cpu"
                and cpu_start_o.dtype == torch.int32
                and cpu_end_o.dtype == torch.int32
                and cpu_start_o.is_pinned()
                and cpu_end_o.is_pinned()
                and gpu_start_o.device == device
                and gpu_end_o.device == device
                and gpu_start_o.dtype == torch.int32
                and gpu_end_o.dtype == torch.int32
                and cpu_start_o.dim() == 2
                and cpu_end_o.dim() == 2
                and gpu_start_o.dim() == 2
                and gpu_end_o.dim() == 2
                and cpu_start_o.shape[0] >= shape[0]
                and cpu_start_o.shape[1] >= shape[1]
                and cpu_end_o.shape[0] >= shape[0]
                and cpu_end_o.shape[1] >= shape[1]
                and gpu_start_o.shape[0] >= shape[0]
                and gpu_start_o.shape[1] >= shape[1]
                and gpu_end_o.shape[0] >= shape[0]
                and gpu_end_o.shape[1] >= shape[1]
            ):
                return (
                    cpu_start_o[: shape[0], : shape[1]],
                    cpu_end_o[: shape[0], : shape[1]],
                    gpu_start_o[: shape[0], : shape[1]],
                    gpu_end_o[: shape[0], : shape[1]],
                )
        # [KEY-NORMS-DELTA-PINNED-H2D-WAR-FIX] R7:共享持久 pinned/GPU delta 对
        # 即将被本世代覆写(host 写 pinned + copy_ 写 GPU dst),上一世代的
        # non_blocking H2D 与 delta kernel 可能未决(pinned staging 同型撕裂)。
        # 覆写者是 CPU 侧写,设备 wait 无法排它 → host 侧 query-first:稳态上一
        # 世代(隔多步)早已完成=query 即过零成本;仅密集 flush 重叠窗真等。
        _evt = getattr(self, "_selector_key_norms_delta_inflight_evt", None)
        if _evt is not None and not _evt.query():
            _evt.synchronize()
        cpu_start = self._selector_key_norms_delta_start_cpu
        if (
            cpu_start is None
            or cpu_start.dim() != 2
            or cpu_start.shape[0] < shape[0]
            or cpu_start.shape[1] < shape[1]
            or not cpu_start.is_pinned()
        ):
            cpu_start = torch.empty(shape, device="cpu", dtype=torch.int32, pin_memory=True)
            self._selector_key_norms_delta_start_cpu = cpu_start
        cpu_end = self._selector_key_norms_delta_end_cpu
        if (
            cpu_end is None
            or cpu_end.dim() != 2
            or cpu_end.shape[0] < shape[0]
            or cpu_end.shape[1] < shape[1]
            or not cpu_end.is_pinned()
        ):
            cpu_end = torch.empty(shape, device="cpu", dtype=torch.int32, pin_memory=True)
            self._selector_key_norms_delta_end_cpu = cpu_end
        gpu_start = self._selector_key_norms_delta_start_gpu
        if (
            gpu_start is None
            or gpu_start.dim() != 2
            or gpu_start.shape[0] < shape[0]
            or gpu_start.shape[1] < shape[1]
            or gpu_start.device != device
        ):
            if gpu_start is not None and gpu_start.is_cuda:
                # [SELECTOR-SHARED-SCRATCH-REALLOC-UAF-FIX] R5:delta GPU 载体
                # 换代弃旧,消费者(delta kernel/H2D)可在另一流在飞。冷事件。
                self._uaf_guard_record_streams_before_discard(gpu_start)
            gpu_start = torch.empty(shape, device=device, dtype=torch.int32)
            self._selector_key_norms_delta_start_gpu = gpu_start
        gpu_end = self._selector_key_norms_delta_end_gpu
        if (
            gpu_end is None
            or gpu_end.dim() != 2
            or gpu_end.shape[0] < shape[0]
            or gpu_end.shape[1] < shape[1]
            or gpu_end.device != device
        ):
            if gpu_end is not None and gpu_end.is_cuda:
                self._uaf_guard_record_streams_before_discard(gpu_end)
            gpu_end = torch.empty(shape, device=device, dtype=torch.int32)
            self._selector_key_norms_delta_end_gpu = gpu_end
        return (
            cpu_start[: shape[0], : shape[1]],
            cpu_end[: shape[0], : shape[1]],
            gpu_start[: shape[0], : shape[1]],
            gpu_end[: shape[0], : shape[1]],
        )

    def _ensure_selector_key_norms_target_cpu_buffer(
        self,
        *,
        batch: int,
    ) -> torch.Tensor:
        """Reuse a pinned int32 target-lens buffer for key_norms delta staging."""
        if batch <= 0:
            return torch.empty((0,), device="cpu", dtype=torch.int32)
        batch_i = int(batch)
        buf = self._selector_key_norms_target_cpu
        if (
            buf is None
            or buf.dim() != 1
            or buf.shape[0] < batch_i
            or buf.dtype != torch.int32
            or not buf.is_pinned()
        ):
            buf = torch.empty((batch_i,), device="cpu", dtype=torch.int32, pin_memory=True)
            self._selector_key_norms_target_cpu = buf
        return buf[:batch_i]

    def _ensure_selector_decode_bounds_buffers(
        self,
        *,
        layers: int,
        batch: int,
        num_kv_heads: int,
        num_queries_per_kv: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reuse decode bounds CUDA output tensors for the current selector shape."""
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if layers <= 0 or batch <= 0 or num_kv_heads <= 0 or num_queries_per_kv <= 0:
            empty_3d = torch.empty((0, 0, 0), device=device, dtype=torch.int32)
            empty_2d = torch.empty((0, 0), device=device, dtype=torch.int32)
            return empty_3d, empty_3d, empty_3d, empty_3d, empty_2d, empty_2d

        layers_i = int(layers)
        batch_i = int(batch)
        heads_i = int(num_kv_heads)
        group_i = int(num_queries_per_kv)
        shape_3d = (layers_i, batch_i, heads_i)
        shape_2d = (layers_i * batch_i * heads_i, group_i)
        key = (
            str(device.type),
            int(device.index) if device.index is not None else -1,
            shape_3d,
            shape_2d,
        )
        override = getattr(self, "_selector_decode_bounds_buffers_override", None)
        if isinstance(override, dict):
            buffers = override.get(key)
            if (
                not isinstance(buffers, tuple)
                or len(buffers) != 6
                or any(
                    not isinstance(t, torch.Tensor)
                    or t.device != device
                    or t.dtype != torch.int32
                    or not t.is_contiguous()
                    for t in buffers
                )
                or any(tuple(t.shape) != shape_3d for t in buffers[:4])
                or any(tuple(t.shape) != shape_2d for t in buffers[4:])
            ):
                if isinstance(buffers, tuple):
                    # [SELECTED-OUT-RING v2] 槽稳定容器跨 run 持久,换代弃旧
                    # 走守卫(bounds 有 deferred writer 晚读=R5 消费链最长站)。
                    for _old in buffers:
                        if isinstance(_old, torch.Tensor) and _old.is_cuda:
                            self._uaf_guard_record_streams_before_discard(_old)
                buffers = (
                    torch.empty(shape_3d, device=device, dtype=torch.int32),
                    torch.empty(shape_3d, device=device, dtype=torch.int32),
                    torch.empty(shape_3d, device=device, dtype=torch.int32),
                    torch.empty(shape_3d, device=device, dtype=torch.int32),
                    torch.empty(shape_2d, device=device, dtype=torch.int32),
                    torch.empty(shape_2d, device=device, dtype=torch.int32),
                )
                override[key] = buffers
            return buffers
        buffers = self._selector_decode_bounds_buffers
        if (
            self._selector_decode_bounds_key != key
            or not isinstance(buffers, tuple)
            or len(buffers) != 6
            or any(
                not isinstance(t, torch.Tensor)
                or t.device != device
                or t.dtype != torch.int32
                or not t.is_contiguous()
                for t in buffers
            )
            or any(tuple(t.shape) != shape_3d for t in buffers[:4])
            or any(tuple(t.shape) != shape_2d for t in buffers[4:])
        ):
            if isinstance(buffers, tuple):
                # [SELECTOR-SHARED-SCRATCH-REALLOC-UAF-FIX] R5:bounds 六元组
                # 换代弃旧;result.recent_start 等可为旧缓冲视图,deferred
                # writer 在另一流晚读(消费链最长站点)。冷事件。
                for _old in buffers:
                    if isinstance(_old, torch.Tensor) and _old.is_cuda:
                        self._uaf_guard_record_streams_before_discard(_old)
            buffers = (
                torch.empty(shape_3d, device=device, dtype=torch.int32),
                torch.empty(shape_3d, device=device, dtype=torch.int32),
                torch.empty(shape_3d, device=device, dtype=torch.int32),
                torch.empty(shape_3d, device=device, dtype=torch.int32),
                torch.empty(shape_2d, device=device, dtype=torch.int32),
                torch.empty(shape_2d, device=device, dtype=torch.int32),
            )
            self._selector_decode_bounds_key = key
            self._selector_decode_bounds_buffers = buffers
        return buffers

    def _ensure_selector_selected_indices_out(
        self,
        *,
        layers: int,
        batch: int,
        num_kv_heads: int,
        k_head: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Reuse selector post-topk int32 output storage."""
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if layers <= 0 or batch <= 0 or num_kv_heads <= 0 or k_head <= 0:
            return torch.empty((0, 0, 0, 0), device=device, dtype=torch.int32)

        shape = (
            int(layers),
            int(batch),
            int(num_kv_heads),
            int(k_head),
        )
        key = (
            str(device.type),
            int(device.index) if device.index is not None else -1,
            shape,
        )
        override = getattr(self, "_selector_selected_indices_out_override", None)
        if isinstance(override, dict):
            out = override.get(key)
            if (
                out is None
                or out.device != device
                or out.dtype != torch.int32
                or not out.is_contiguous()
                or tuple(out.shape) != shape
            ):
                if out is not None and out.is_cuda:
                    # [SELECTED-OUT-RING v2] 槽稳定容器(SlotStableOverrides)
                    # 跨 run 持久,换代弃旧走守卫(评审清单·换代臂);per-run
                    # 私有 dict 内几乎不换代,守卫为零成本防御。
                    self._uaf_guard_record_streams_before_discard(out)
                out = torch.empty(shape, device=device, dtype=torch.int32)
                override[key] = out
            return out

        out = self._selector_selected_indices_out
        if (
            self._selector_selected_indices_out_key != key
            or out is None
            or out.device != device
            or out.dtype != torch.int32
            or not out.is_contiguous()
            or tuple(out.shape) != shape
        ):
            if out is not None and out.is_cuda:
                # [SELECTOR-SHARED-SCRATCH-REALLOC-UAF-FIX] R5:共享 selected
                # 槽在双流间轮转(flush/deferred 无 override),换代弃旧守卫;
                # 与 SELECTED-PRIVATE-OUT(pending 臂私有化)互补不重复。
                self._uaf_guard_record_streams_before_discard(out)
            out = torch.empty(shape, device=device, dtype=torch.int32)
            self._selector_selected_indices_out_key = key
            self._selector_selected_indices_out = out
        return out

    def _ensure_selector_writer_row_tensor_all(
        self,
        *,
        layers: int,
        batch: int,
        device: torch.device,
        row_tensor_first: torch.Tensor,
        per_layer_rows: Optional[Sequence[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """ASYNC_PRODUCER_WRITER_GRAPH (defect #6): stable-storage CONTIGUOUS [layers, batch] int32 buffer.

        The fused writer kernel takes NO row stride and assumes a contiguous
        row-major ``[layers, batch]`` packing (the OFF path always passes a
        ``.contiguous()`` tensor). A 2-D capacity buffer sliced ``buf[:L,:B]``
        is NON-contiguous when ``B < cap_batch`` (row stride == cap_batch), which
        would corrupt the row indices the kernel reads. So we keep a persistent
        FLAT capacity buffer and return ``flat[:L*B].view(L, B)`` — contiguous
        (stride ``[B, 1]``, byte-identical to the OFF path) AND with a stable
        storage data_ptr within capacity (gates the captured writer's key).
        Realloc only on capacity growth. ``per_layer_rows`` (when given and not
        broadcast) is copied row-by-row; otherwise ``row_tensor_first`` is
        broadcast.
        """
        return self._ensure_writer_stable_row_impl(
            layers=layers,
            batch=batch,
            device=device,
            row_tensor_first=row_tensor_first,
            per_layer_rows=per_layer_rows,
        )

    def _uaf_guard_record_streams_before_discard(self, tensor) -> None:
        """[R4 公共守卫] 弃引用/换代前对三个潜在消费流 record_stream:
        rebuild/writer 链在主流 drain 与 refresh_stream off-loop 双上下文交替,
        弃旧 storage 时另一流可能有在飞读者(graph 烤旧 ptr/launch args 持引用/
        non_blocking copy 未决);record_stream 让 allocator 等全部流进度再回收。
        仅换代/清缓存冷事件调用,零热路径开销。"""
        streams = []
        rs = getattr(self, "refresh_stream", None)
        if rs is not None:
            streams.append(rs)
        for cand in (torch.cuda.current_stream(), torch.cuda.default_stream()):
            if all(cand != s for s in streams):
                streams.append(cand)
        for s in streams:
            tensor.record_stream(s)

    def _wait_writer_dispatch_done_before_stable_overwrite(self) -> None:
        """[WRITER-STABLE-OVERWRITE-WAR-FIX 2026-07-07] R3 守卫,sanitizer 终审
        定谳的 4B illegal 主凶修:100/100 Invalid read 全部落在 writer graph
        REPLAY 内 gather kernel 的 src KV 读(cuda.cu:808),地址=大分配前方
        24KB=block_id≈-1 的负寻址——stable 四单例(row/seq_lens/slot/selected)
        被跨上下文覆写撕裂(drain 主流与 off-loop refresh_stream 交替 rebuild,
        上次 writer replay/eager 在另一流在飞时,本次 copy_ 原位覆写同一
        buffer),撕裂 seq_lens 使 autolen 超长→btable 读未分配列(-1)→负寻址;
        pos trap 与 autolen 同源撕裂值=天然盲区。#9-KEY v3 latch 只护"本次
        copy→本次 replay"的正向序,本守卫补反向序:覆写前设备侧等待最近一次
        writer dispatch 完成。稳态 replay 早已完成=query 即过(零成本);重叠
        窗=当前流 wait_event(设备侧,无 host 阻塞),无 fallback。"""
        evt = getattr(self, "_writer_dispatch_done_evt", None)
        if evt is None:
            return
        # [R3-CAPTURE-BOUNDARY-ASSERT] stable 载体原位覆写在 graph capture 内
        # 构造不可达(覆写只发生在 rebuild eager 段;capture 内 query 本身即
        # 非法)——可执行断言替代论证,给出可定位错误而非 cudaErrorStream
        # CaptureUnsupported 的隐晦形态。evt 存在即 CUDA 已活跃,查询安全。
        if _is_stream_capturing_or_raise(stage="writer_stable_overwrite_gate"):
            raise RuntimeError(
                "writer stable-carrier overwrite entered during CUDA graph "
                "capture; dispatch-done gate cannot serialize here "
                "(unreachable by construction — investigate the capture path)"
            )
        if not evt.query():
            torch.cuda.current_stream().wait_event(evt)

    def _ensure_writer_stable_row_impl(
        self,
        *,
        layers: int,
        batch: int,
        device: torch.device,
        row_tensor_first: torch.Tensor,
        per_layer_rows,
    ) -> torch.Tensor:
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if layers <= 0 or batch <= 0:
            return torch.empty((max(0, int(layers)), max(0, int(batch))), device=device, dtype=torch.int32)
        key = (
            str(device.type),
            int(device.index) if device.index is not None else -1,
        )
        need = int(layers) * int(batch)
        buf = self._selector_writer_row_tensor_all
        need_realloc = (
            buf is None
            or self._selector_writer_row_tensor_all_key != key
            or buf.device != device
            or buf.dtype != torch.int32
            or int(buf.numel()) < need
        )
        if need_realloc:
            cap = need if buf is None else max(need, int(buf.numel()))
            if buf is not None:
                # [WRITER-STABLE-REALLOC-UAF-FIX 2026-07-07] 与 selected 的
                # SELECTED-STABLE-REALLOC-UAF-FIX 同族同窗(R2):弃旧前三流守卫。
                self._uaf_guard_record_streams_before_discard(buf)
            buf = torch.empty((cap,), device=device, dtype=torch.int32)
            self._selector_writer_row_tensor_all = buf
            self._selector_writer_row_tensor_all_key = key
            # #9-KEY: capacity growth moves the row data_ptr -> the captured
            # writer key is stale. Invalidate the graph for a one-shot recapture.
            if getattr(self, "_writer_graph_state", None) is not None:
                self._writer_graph_state = None
        # Contiguous [layers, batch] prefix view over the flat capacity buffer:
        # stable data_ptr within capacity, row stride == batch (matches OFF).
        view = buf[:need].view(int(layers), int(batch))
        self._wait_writer_dispatch_done_before_stable_overwrite()
        if per_layer_rows is not None and len(per_layer_rows) == int(layers):
            for layer_idx in range(int(layers)):
                src = per_layer_rows[layer_idx]
                view[layer_idx].copy_(src[: int(batch)].to(dtype=torch.int32))
        else:
            src = row_tensor_first[: int(batch)].to(dtype=torch.int32)
            view.copy_(src.unsqueeze(0).expand(int(layers), -1))
        return view

    def _ensure_selector_writer_seq_lens_all(
        self,
        *,
        batch: int,
        device: torch.device,
        src: torch.Tensor,
    ) -> torch.Tensor:
        """#9-KEY: stable-storage CONTIGUOUS [batch] int32 seq_lens buffer.

        seq_lens is rebuilt fresh every refresh (payload compaction empties
        seq_lens_batch), so the eager ``.contiguous()`` allocation has an
        unstable data_ptr that churns the captured writer key. We keep a
        persistent FLAT capacity buffer and return ``flat[:batch]`` — a
        contiguous int32 view with a STABLE data_ptr within capacity
        (byte-identical layout to the OFF path's ``.contiguous()``). Realloc
        only on capacity growth; on growth we invalidate the captured writer
        graph so a one-shot recapture happens. ``src`` is copied in
        (non_blocking ok — same-stream program order + the ready latch).
        """
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if batch <= 0:
            return torch.empty((max(0, int(batch)),), device=device, dtype=torch.int32)
        key = (
            str(device.type),
            int(device.index) if device.index is not None else -1,
        )
        need = int(batch)
        buf = self._selector_writer_seq_lens_all
        need_realloc = (
            buf is None
            or self._selector_writer_seq_lens_all_key != key
            or buf.device != device
            or buf.dtype != torch.int32
            or int(buf.numel()) < need
        )
        if need_realloc:
            cap = need if buf is None else max(need, int(buf.numel()))
            if buf is not None:
                # [WRITER-STABLE-REALLOC-UAF-FIX 2026-07-07] R2 三流守卫。
                self._uaf_guard_record_streams_before_discard(buf)
            buf = torch.empty((cap,), device=device, dtype=torch.int32)
            self._selector_writer_seq_lens_all = buf
            self._selector_writer_seq_lens_all_key = key
            # #9-KEY: capacity growth moves the seq_lens data_ptr -> the
            # captured writer key is stale. Invalidate for a one-shot recapture.
            if getattr(self, "_writer_graph_state", None) is not None:
                self._writer_graph_state = None
        view = buf[:need]
        src_i32 = src[: int(batch)]
        if src_i32.dtype != torch.int32:
            src_i32 = src_i32.to(dtype=torch.int32)
        self._wait_writer_dispatch_done_before_stable_overwrite()
        view.copy_(src_i32, non_blocking=True)
        return view

    def _ensure_selector_writer_slot_tensor_all(
        self,
        *,
        batch: int,
        device: torch.device,
        src: torch.Tensor,
    ) -> torch.Tensor:
        """#9-KEY v2: stable-storage CONTIGUOUS [batch] int32 slot_tensor buffer.

        ``slot_tensor_i32`` comes from the refresh payload (a fresh per-refresh
        carrier), so the eager ``.contiguous()`` allocation has an unstable
        data_ptr that churns the captured writer key field [4] (31/31). We keep
        a persistent FLAT capacity buffer and return ``flat[:batch]`` — a
        contiguous int32 view with a STABLE data_ptr within capacity
        (byte-identical layout to the OFF path's ``.contiguous()``). Realloc
        only on capacity growth; on growth we invalidate the captured writer
        graph so a one-shot recapture happens. It is a small [rows] i32 — the
        D2D copy cost is negligible.
        """
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if batch <= 0:
            return torch.empty((max(0, int(batch)),), device=device, dtype=torch.int32)
        key = (
            str(device.type),
            int(device.index) if device.index is not None else -1,
        )
        need = int(batch)
        buf = self._selector_writer_slot_tensor_all
        need_realloc = (
            buf is None
            or self._selector_writer_slot_tensor_all_key != key
            or buf.device != device
            or buf.dtype != torch.int32
            or int(buf.numel()) < need
        )
        if need_realloc:
            cap = need if buf is None else max(need, int(buf.numel()))
            if buf is not None:
                # [WRITER-STABLE-REALLOC-UAF-FIX 2026-07-07] R2 三流守卫。
                self._uaf_guard_record_streams_before_discard(buf)
            buf = torch.empty((cap,), device=device, dtype=torch.int32)
            self._selector_writer_slot_tensor_all = buf
            self._selector_writer_slot_tensor_all_key = key
            # #9-KEY v2: capacity growth moves the slot data_ptr -> the captured
            # writer key is stale. Invalidate for a one-shot recapture.
            if getattr(self, "_writer_graph_state", None) is not None:
                self._writer_graph_state = None
        out = buf[:need]
        src_i32 = src[: int(batch)]
        if src_i32.dtype != torch.int32:
            src_i32 = src_i32.to(dtype=torch.int32)
        self._wait_writer_dispatch_done_before_stable_overwrite()
        out.copy_(src_i32, non_blocking=True)
        return out

    def _ensure_selector_writer_selected_all(
        self,
        *,
        layers: int,
        batch: int,
        num_kv_heads: int,
        k_head: int,
        device: torch.device,
        src: torch.Tensor,
    ) -> torch.Tensor:
        """#9-KEY v2: stable-storage CONTIGUOUS [L,B,H,k] int32 selected buffer.

        The post-topk ``_selector_selected_indices_out`` buffer is keyed by the
        EXACT shape ``(layers, batch, num_kv_heads, k_head)`` and reallocates
        whenever the per-refresh persist width ``k_head`` or ``batch`` wobbles,
        moving the data_ptr ~60% of refreshes (key field [1] churned 19/31). We
        copy into a persistent FLAT capacity buffer and return
        ``flat[:L*B*H*k].view(L, B, H, k)`` — contiguous (row-major, stride
        ``[B*H*k, H*k, k, 1]``, byte-identical to a fresh ``.contiguous()``) and
        with a STABLE data_ptr that survives shape jitter within capacity.
        Realloc only on capacity GROWTH; on growth we invalidate the captured
        writer graph for a one-shot recapture. The index copy is 16 KB-512 KB
        i32 (sub-microsecond) — same byte volume the eager ``.contiguous()``
        fallback already pays — and does NOT eat the replay win (the win is
        skipping the multi-MB fused KV gather, not a 256 KB index copy).
        """
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if layers <= 0 or batch <= 0 or num_kv_heads <= 0 or k_head <= 0:
            return torch.empty(
                (max(0, int(layers)), max(0, int(batch)), max(0, int(num_kv_heads)), max(0, int(k_head))),
                device=device,
                dtype=torch.int32,
            )
        key = (
            str(device.type),
            int(device.index) if device.index is not None else -1,
        )
        need = int(layers) * int(batch) * int(num_kv_heads) * int(k_head)
        buf = self._selector_writer_selected_all
        need_realloc = (
            buf is None
            or self._selector_writer_selected_all_key != key
            or buf.device != device
            or buf.dtype != torch.int32
            or int(buf.numel()) < need
        )
        if need_realloc:
            cap = need if buf is None else max(need, int(buf.numel()))
            if buf is not None:
                # [SELECTED-STABLE-REALLOC-UAF-FIX 2026-07-07] 容量增长弃旧
                # storage 前对全部潜在消费流 record_stream:在飞 deferred writer
                # (graph replay 烤旧 data_ptr / eager launch args 持旧引用)与
                # 跨流 stable copy_ 可能未决;直接 GC 让 allocator 只按创建流序
                # 复用/解映射(expandable segment)= illegal address。4B bs8
                # 错峰 batch 1→8 使 need 波动→本分支反复触发;0.6b bs2 恒定批
                # 从不触发(批组成选择性来源)。realloc 是冷事件,零热路径开销。
                _uaf_guard_streams = []
                _rs = getattr(self, "refresh_stream", None)
                if _rs is not None:
                    _uaf_guard_streams.append(_rs)
                for _cand in (
                    torch.cuda.current_stream(),
                    torch.cuda.default_stream(),
                ):
                    if all(_cand != s for s in _uaf_guard_streams):
                        _uaf_guard_streams.append(_cand)
                for _s in _uaf_guard_streams:
                    buf.record_stream(_s)
            buf = torch.empty((cap,), device=device, dtype=torch.int32)
            self._selector_writer_selected_all = buf
            self._selector_writer_selected_all_key = key
            # #9-KEY v2: capacity growth moves the selected data_ptr -> the
            # captured writer key is stale. Invalidate for a one-shot recapture.
            if getattr(self, "_writer_graph_state", None) is not None:
                self._writer_graph_state = None
        # Contiguous [L,B,H,k] prefix view over the flat capacity buffer:
        # stable data_ptr within capacity, row-major (matches OFF .contiguous()).
        out = buf[:need].view(int(layers), int(batch), int(num_kv_heads), int(k_head))
        src_i32 = src
        if src_i32.dtype != torch.int32:
            src_i32 = src_i32.to(dtype=torch.int32)
        self._wait_writer_dispatch_done_before_stable_overwrite()
        out.copy_(src_i32)
        return out

    def _record_writer_input_ready(self, *, device: torch.device) -> None:
        """#9-KEY v3: latch the writer-input copies' completion on their stream.

        The seq_lens / slot / selected copies above are issued on the producing
        (refresh) stream immediately before the writer dispatch, so an
        intra-stream race with replay is impossible (CUDA preserves stream
        order; non_blocking does not reorder within a stream). This records an
        event AFTER those copies as a defensive guard mirroring the
        pointer-publish ready-event ordering — drained by
        ``_wait_writer_input_ready`` before replay. Fail-open: any error (no
        CUDA, capturing) leaves the latch unset and is ignored downstream.
        """
        try:
            dev = torch.device(device)
            if dev.type != "cuda":
                self._selector_writer_input_ready_event = None
                return
            if bool(torch.cuda.is_current_stream_capturing()):
                # Never record an event while capturing the writer/decode graph.
                self._selector_writer_input_ready_event = None
                return
            ev = self._selector_writer_input_ready_event
            if ev is None:
                ev = torch.cuda.Event(enable_timing=False)
                self._selector_writer_input_ready_event = ev
            ev.record(torch.cuda.current_stream(device=dev))
        except Exception:
            self._selector_writer_input_ready_event = None

    def _wait_writer_input_ready(self, *, device: torch.device) -> None:
        """#9-KEY v3: drain the writer-input ready latch before graph replay.

        Same-stream this is a strict no-op (the event marks a position already
        passed on this stream). It only adds a real dependency if a future
        refactor issues the copies on a different stream. Fail-open: never
        raises, never waits while capturing.
        """
        try:
            ev = self._selector_writer_input_ready_event
            if ev is None:
                return
            dev = torch.device(device)
            if dev.type != "cuda":
                return
            if bool(torch.cuda.is_current_stream_capturing()):
                return
            torch.cuda.current_stream(device=dev).wait_event(ev)
        except Exception:
            return

    def _ensure_selector_capture_scores_buffer(
        self,
        *,
        layers: int,
        batch: int,
        num_heads: int,
        window: int,
        kv_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if layers <= 0 or batch <= 0 or num_heads <= 0 or window <= 0 or kv_len <= 0:
            return torch.empty((0, 0, 0, 0, 0), device=device, dtype=dtype)
        shape = (layers, batch, num_heads, window, kv_len)
        buf = self._selector_capture_scores_all
        if (
            buf is None
            or buf.device != device
            or buf.dtype != dtype
            or buf.shape[0] < shape[0]
            or buf.shape[1] < shape[1]
            or buf.shape[2] < shape[2]
            or buf.shape[3] < shape[3]
            or buf.shape[4] < shape[4]
        ):
            buf = torch.empty(shape, device=device, dtype=dtype)
            self._selector_capture_scores_all = buf
        return buf[:layers, :batch, :num_heads, :window, :kv_len]

    def _ensure_selector_log_f_denoms_buffer(
        self,
        *,
        layers: int,
        batch: int,
        num_heads: int,
        device: torch.device,
    ) -> torch.Tensor:
        if layers <= 0 or batch <= 0 or num_heads <= 0:
            return torch.empty((0, 0, 0), device=device, dtype=torch.float32)
        shape = (layers, batch, num_heads)
        buf = self._selector_log_f_denoms_all
        if (
            buf is None
            or buf.device != device
            or buf.dtype != torch.float32
            or buf.shape[0] < shape[0]
            or buf.shape[1] < shape[1]
            or buf.shape[2] < shape[2]
        ):
            buf = torch.empty(shape, device=device, dtype=torch.float32)
            self._selector_log_f_denoms_all = buf
        return buf[:layers, :batch, :num_heads]

    def _ensure_selector_kv_lengths_buffer(
        self,
        *,
        layers: int,
        batch: int,
        num_heads: int,
        device: torch.device,
    ) -> torch.Tensor:
        if layers <= 0 or batch <= 0 or num_heads <= 0:
            return torch.empty((0, 0, 0), device=device, dtype=torch.long)
        shape = (layers, batch, num_heads)
        buf = self._selector_kv_lengths_all
        if (
            buf is None
            or buf.device != device
            or buf.dtype != torch.long
            or buf.shape[0] < shape[0]
            or buf.shape[1] < shape[1]
            or buf.shape[2] < shape[2]
        ):
            buf = torch.empty(shape, device=device, dtype=torch.long)
            self._selector_kv_lengths_all = buf
        return buf[:layers, :batch, :num_heads]

    def _ensure_log_f_workspace(
        self,
        *,
        batch: int,
        num_heads: int,
        last_n: int,
        kv_max: int,
        device: torch.device,
    ) -> torch.Tensor:
        """为 log_f(last_n>1) 提供 step-wise 可复用 workspace（scratch）。

        - scratch: fp32 [batch, Hq, last_n, kv_max]（dense log_f 行使用）

        该 workspace 需要在 Python 侧持有引用，确保异步 kernel 期间不会被释放。
        denom_f 由 kernel 直接写入 step capture arena（避免额外 copy/归一化扫描）。
        """
        if batch <= 0 or num_heads <= 0 or last_n <= 1 or kv_max <= 0:
            return torch.empty((0, 0, 0, 0), device=device, dtype=torch.float32)

        # kv_len 随 step 增长，kv_max 会频繁变化；若按精确 kv_max 分配会导致频繁重分配。
        # 这里按 256 对齐并允许复用“更大”的 workspace，避免 refresh/prefill 场景 allocator 抖动。
        kv_needed = int(kv_max)
        kv_alloc = _align_up_int(kv_needed, 256)

        key = (int(batch), int(num_heads), int(last_n), str(device))
        override = getattr(self, "_log_f_scratch_workspace_override", None)
        if isinstance(override, dict):
            scratch = override.get(key)
            need_alloc = (
                scratch is None
                or scratch.device != device
                or scratch.dtype != torch.float32
                or scratch.dim() != 4
                or int(scratch.shape[0]) != int(batch)
                or int(scratch.shape[1]) != int(num_heads)
                or int(scratch.shape[2]) != int(last_n)
                or int(scratch.shape[3]) < int(kv_needed)
            )
            if need_alloc:
                if scratch is not None and scratch.is_cuda:
                    # [SELECTED-OUT-RING v2] 槽稳定容器换代弃旧走守卫。
                    self._uaf_guard_record_streams_before_discard(scratch)
                prev_kv = (
                    int(scratch.shape[3])
                    if scratch is not None and scratch.dim() == 4
                    else 0
                )
                kv_alloc = max(int(kv_alloc), int(prev_kv))
                scratch = torch.empty(
                    (batch, num_heads, last_n, kv_alloc),
                    device=device,
                    dtype=torch.float32,
                )
                override[key] = scratch
            return scratch[:, :, :, :kv_needed]

        if getattr(self, "_log_f_workspace_key", None) != key:
            _stale = getattr(self, "_log_f_scratch_workspace", None)
            if _stale is not None and _stale.is_cuda:
                # [SELECTOR-SHARED-SCRATCH-REALLOC-UAF-FIX] R5:key 漂移置 None
                # 直弃共享 scratch,另一流消费者可在飞。冷事件。
                self._uaf_guard_record_streams_before_discard(_stale)
            self._log_f_scratch_workspace = None
            self._log_f_workspace_key = key

        scratch = getattr(self, "_log_f_scratch_workspace", None)
        need_alloc = (
            scratch is None
            or scratch.device != device
            or scratch.dtype != torch.float32
            or scratch.dim() != 4
            or int(scratch.shape[0]) != int(batch)
            or int(scratch.shape[1]) != int(num_heads)
            or int(scratch.shape[2]) != int(last_n)
            or int(scratch.shape[3]) < int(kv_needed)
        )
        if need_alloc:
            if scratch is not None and scratch.is_cuda:
                self._uaf_guard_record_streams_before_discard(scratch)
            prev_kv = int(scratch.shape[3]) if scratch is not None and scratch.dim() == 4 else 0
            kv_alloc = max(int(kv_alloc), int(prev_kv))
            scratch = torch.empty((batch, num_heads, last_n, kv_alloc), device=device, dtype=torch.float32)
            self._log_f_scratch_workspace = scratch

        # 返回 view 以限制访问范围（避免后续误用超过 kv_needed 的尾部数据）。
        return scratch[:, :, :, :kv_needed]

    def _ensure_selector_pipeline_workspaces(
        self,
        *,
        layers: int,
        batch: int,
        num_kv_heads: int,
        kv_max: int,
        device: torch.device,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Per-controller selector pipeline scratch (workspace_a, workspace_b).

        selector_pipeline_ext run_log_s / run_soft_nms / run_cross_head each
        torch::empty a fresh (M, K) f32 output when no workspace is supplied
        (M = layers*batch*num_kv_heads, K = kv tokens). workspace_a feeds
        run_log_s + run_cross_head and workspace_b feeds run_soft_nms; the C++
        workspace_2d_or_empty TORCH_CHECKs an EXACT (M, K) contiguous f32
        tensor, so we cache exact-shaped contiguous buffers keyed by
        (rows, kv_needed, device) and reuse them verbatim in the steady window
        (reallocate only when K leaves the window). The kernel write path is
        bit-identical to the empty path (tests/test_selector_pipeline_ext.py).
        Buffers live on the controller so the async refresh stream can retain
        them across the double-buffer boundary.
        """
        if layers <= 0 or batch <= 0 or num_kv_heads <= 0 or kv_max <= 0:
            return None
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        rows = int(layers) * int(batch) * int(num_kv_heads)
        kv_needed = int(kv_max)
        # The C++ workspace_2d_or_empty TORCH_CHECKs an EXACT (rows, kv_needed)
        # CONTIGUOUS f32 tensor (size(1) == cols, is_contiguous()). A last-dim
        # slice of a wider buffer is non-contiguous, so we cache exact-shaped
        # buffers keyed by (rows, kv_needed, device) and return them verbatim:
        # verbatim reuse inside a stable-K steady window, realloc only when K
        # leaves the window (monotone growth). The retained buffer IS the one
        # the kernel writes, so async double-buffer retain is exact.
        key = (int(rows), int(kv_needed), str(device))

        def _alloc() -> Tuple[torch.Tensor, torch.Tensor]:
            ws_a = torch.empty((rows, kv_needed), device=device, dtype=torch.float32)
            ws_b = torch.empty((rows, kv_needed), device=device, dtype=torch.float32)
            return (ws_a, ws_b)

        def _is_valid(buf: Optional[torch.Tensor]) -> bool:
            return (
                buf is not None
                and buf.device == device
                and buf.dtype == torch.float32
                and buf.dim() == 2
                and int(buf.shape[0]) == rows
                and int(buf.shape[1]) == kv_needed
                and buf.is_contiguous()
            )

        override = getattr(self, "_selector_pipeline_workspace_override", None)
        if isinstance(override, dict):
            pair = override.get(key)
            if pair is None or not _is_valid(pair[0]) or not _is_valid(pair[1]):
                if pair is not None:
                    # [SELECTED-OUT-RING v2] 槽稳定容器换代弃旧走守卫。
                    for _old in pair:
                        if isinstance(_old, torch.Tensor) and _old.is_cuda:
                            self._uaf_guard_record_streams_before_discard(_old)
                pair = _alloc()
                override[key] = pair
            return pair

        if getattr(self, "_selector_pipeline_workspace_key", None) != key:
            for _stale in (
                getattr(self, "_selector_pipeline_workspace_a", None),
                getattr(self, "_selector_pipeline_workspace_b", None),
            ):
                if _stale is not None and _stale.is_cuda:
                    # [SELECTOR-SHARED-SCRATCH-REALLOC-UAF-FIX] R5:K 出窗置
                    # None 直弃共享 workspace 对,另一流消费者可在飞。冷事件。
                    self._uaf_guard_record_streams_before_discard(_stale)
            self._selector_pipeline_workspace_a = None
            self._selector_pipeline_workspace_b = None
            self._selector_pipeline_workspace_key = key
        ws_a = getattr(self, "_selector_pipeline_workspace_a", None)
        ws_b = getattr(self, "_selector_pipeline_workspace_b", None)
        if not _is_valid(ws_a) or not _is_valid(ws_b):
            for _stale in (ws_a, ws_b):
                if _stale is not None and _stale.is_cuda:
                    self._uaf_guard_record_streams_before_discard(_stale)
            ws_a, ws_b = _alloc()
            self._selector_pipeline_workspace_a = ws_a
            self._selector_pipeline_workspace_b = ws_b
        return (ws_a, ws_b)

    # =====================================================================
    # #13 STAGE-0: selector-topk captured graph (default OFF).
    # Disjoint-namespace clone of the ASYNC_PRODUCER_WRITER_GRAPH dispatcher.
    # Engaged only when _SELECTOR_TOPK_GRAPH_CACHED is true at the call site.
    # =====================================================================
    _SELECTOR_TOPK_GRAPH_THRASH_RECAPTURES = 4
    _SELECTOR_TOPK_GRAPH_THRASH_WINDOW = 64

    def _selector_topk_graph_stable_active(self) -> bool:
        """True iff the captured selector graph is SAFE to engage this call.

        The four _ensure_selector_* producers route to per-pending fresh realloc
        dicts whenever ANY selector override attr is an installed dict (the
        deferred / split-writer refresh path). In that mode every buffer's
        data_ptr churns each refresh, so a captured graph could only replay
        stale pointers -> we must bypass and run eager (mirrors
        _writer_graph_split_active, defect #4). On the synchronous flush path the
        override attrs are None and the buffers are shape-keyed stable, so the
        graph is safe.

        [SELECTED-OUT-RING v2 2026-07-09] the production pending path installs
        the ring slot's persistent SlotStableOverrides containers: buffers are
        data_ptr-stable per slot, per-slot release events restore the
        cross-stream ordering, and the slot is never reused before its pending
        is terminal — so the graph is safe there (out/bounds/ws ptrs are part
        of the graph key -> per-slot graphs). Bare per-run dicts (ring spill /
        escape env / replay-refresh private helper) still bypass. Ring spill
        runs are additionally excluded at the dispatch site via
        ptr_rebuild_miss (transient ptrs must never be captured).
        """
        for attr in (
            "_selector_selected_indices_out_override",
            "_selector_decode_bounds_buffers_override",
            "_selector_pipeline_workspace_override",
        ):
            _ov = getattr(self, attr, None)
            if isinstance(_ov, dict) and not isinstance(_ov, SlotStableOverrides):
                return False
        return True

    def _selector_topk_graph_record_recapture(self) -> bool:
        """Thrash detector: recapture 超阈 **且窗口内发生过防御性清库** -> bypass。

        [SELECTED-OUT-RING v2] 旧判据只数 capture 次数——环门控后合法稳态
        key 族(3 槽×两臂+ramp 形状)首窗即可超过 4,误闩死永久 bypass。
        真 churn 的签名是 8-graph 上限清库风暴(population 溢出),故加
        clear 计数联判:无清库=population 有界=不闩(rv7 取证:sync 路径
        churn 正是靠清库风暴打满 64 窗)。
        """
        self._selector_topk_graph_recapture_window += 1
        self._selector_topk_graph_recapture_count += 1
        if self._selector_topk_graph_recapture_window >= int(
            self._SELECTOR_TOPK_GRAPH_THRASH_WINDOW
        ):
            recaptures = int(self._selector_topk_graph_recapture_count)
            clears = int(getattr(self, "_selector_topk_graph_window_clear_count", 0))
            self._selector_topk_graph_recapture_window = 0
            self._selector_topk_graph_recapture_count = 0
            self._selector_topk_graph_window_clear_count = 0
            return (
                recaptures > int(self._SELECTOR_TOPK_GRAPH_THRASH_RECAPTURES)
                and clears > 0
            )
        return False

    def _selector_topk_graph_dispatch(
        self,
        *,
        eager_fn,
        key_fields,
        ptr_rebuild_miss: bool = False,
    ):
        """Replay the captured selector graph on a key hit; else eager (+recapture).

        ``eager_fn`` runs the verbatim _pipeline_with_bounds call and returns the
        selected_indices tensor (a view of the stable selected_indices_out buffer
        when that cache is on). On a replay hit we re-run the captured graph
        (which writes into the SAME stable buffer) and return the return view we
        cached at capture time. Fail-OPEN: any error discards the graph, runs
        eager, latches a permanent bypass.

        ``key_fields`` MUST already encode every consumed data_ptr + shape so any
        regime change (override realloc, kbucket clamp-fallback, slice pad,
        env-flip) naturally misses and refuses to capture.
        """
        import torch as _torch

        # Deferred / split-writer mode -> per-pending buffers -> never capture.
        if not self._selector_topk_graph_stable_active():
            return eager_fn()
        state = self._selector_topk_graph_state
        if isinstance(state, dict) and bool(state.get("bypass")):
            return eager_fn()
        key = tuple(int(v) for v in key_fields)
        graphs = state.get("graphs") if isinstance(state, dict) else None
        entry = graphs.get(key) if isinstance(graphs, dict) else None
        # Hot path: key hit and a captured graph exists -> replay only.
        if entry is not None and not ptr_rebuild_miss:
            if _selector_graph_lru_enabled() and isinstance(graphs, dict):
                # Mark this key most-recently-used (move to the end of the
                # insertion ordering via pop+reinsert, which works on a plain
                # dict) so the LRU eviction at capture time does not drop a
                # still-hot shape.
                try:
                    graphs[key] = graphs.pop(key)
                except KeyError:
                    pass
            try:
                entry["graph"].replay()
                # Real replay counter (graph-agnostic): proof the captured graph
                # replayed, independent of the cached graph object's own .replay
                # (mirrors the writer track's "graph_replay" count).
                self._selector_topk_graph_replay_count = (
                    int(getattr(self, "_selector_topk_graph_replay_count", 0)) + 1
                )
                return entry["result"]
            except Exception:
                _log.warning(
                    "selector topk graph replay failed; bypassing capture",
                    exc_info=True,
                )
                self._selector_topk_graph_state = {"bypass": True}
                return eager_fn()
        # Cold / new-key / ptr-rebuild step: run eager NOW (also the recapture
        # step). Only (re)capture when no pointer rebuild happened this step.
        result = eager_fn()
        if ptr_rebuild_miss:
            if isinstance(graphs, dict):
                graphs.pop(key, None)
            return result
        try:
            self._capture_selector_topk_graph(
                eager_fn=eager_fn, key=key, prewarm_result=result
            )
        except Exception:
            _log.warning(
                "selector topk graph capture failed; bypassing capture",
                exc_info=True,
            )
            self._selector_topk_graph_state = {"bypass": True}
        return result

    def _capture_selector_topk_graph(
        self,
        *,
        eager_fn,
        key,
        prewarm_result,
    ) -> None:
        """Capture the eager selector closure into a per-key CUDA graph.

        The closure is a pure function of its (now data_ptr-stable) launch
        inputs, so the captured graph replays byte-identical output into the
        stable selected_indices_out buffer. The captured ``result`` view is what
        replay returns. Never nests inside the outer decode capture.
        """
        import torch as _torch

        try:
            if bool(_torch.cuda.is_current_stream_capturing()):
                # Already capturing the outer decode graph -> do NOT nest.
                return
        except Exception:
            _log.warning(
                "selector topk graph: failed to query outer capture state; skip",
                exc_info=True,
            )
            return
        state = self._selector_topk_graph_state
        mempool = None
        if isinstance(state, dict):
            mempool = state.get("mempool")
        if mempool is None:
            mempool = _torch.cuda.graph_pool_handle()
        stream = self._selector_topk_graph_stream
        if stream is None:
            stream = _torch.cuda.Stream()
            self._selector_topk_graph_stream = stream
        graph = _torch.cuda.CUDAGraph()
        captured_result = None

        def _body():
            nonlocal captured_result
            captured_result = eager_fn()

        with _torch.cuda.graph(graph, pool=mempool, stream=stream):
            _body()
        key_t = tuple(int(v) for v in key)
        if not isinstance(state, dict) or not isinstance(state.get("graphs"), dict):
            state = {"graphs": {}, "mempool": mempool, "bypass": False}
            self._selector_topk_graph_state = state
        state["mempool"] = mempool
        graphs_map = state["graphs"]
        # Defensive bound: beyond a plausible shape population the keys are
        # churning; drop the stale set (keep the just-captured graph) and let the
        # thrash detector adjudicate a bypass.
        if len(graphs_map) >= 8 and key_t not in graphs_map:
            # [SELECTED-OUT-RING v2] 清库=population 溢出的 churn 实证,计数
            # 供 thrash 联判(合法稳态 key 族有界,永不触发此臂)。
            self._selector_topk_graph_window_clear_count = (
                int(getattr(self, "_selector_topk_graph_window_clear_count", 0)) + 1
            )
            if _selector_graph_lru_enabled():
                # Single-entry LRU: evict only the least-recently-used key
                # (the oldest in the map's insertion ordering, kept fresh by
                # the replay-hit touch below) and keep the other 7 warm graphs.
                try:
                    _lru_victim = next(iter(graphs_map))
                except StopIteration:
                    _lru_victim = None
                if _lru_victim is not None:
                    graphs_map.pop(_lru_victim, None)
            else:
                graphs_map.clear()
        graphs_map[key_t] = {
            "graph": graph,
            "result": captured_result,
        }
        # [SELECTED-OUT-RING v2] 判据仪器:capture 累计(flush 遥测导出)。
        self._selector_topk_graph_capture_count = (
            int(getattr(self, "_selector_topk_graph_capture_count", 0)) + 1
        )
        _tkg_dbg = os.environ.get("VLLM_SPARSE_SELECTOR_TOPK_GRAPH_DEBUG_LOG", "")
        if _tkg_dbg:
            # churn 取证:逐字段落盘 key(诊断档,默认关;bench 吞 stdout 故写文件)。
            try:
                with open(_tkg_dbg, "a") as _fh:
                    _fh.write(
                        f"{os.getpid()}\tcapture#{int(self._selector_topk_graph_capture_count)}\t{key_t}\n"
                    )
            except OSError:
                pass
        if self._selector_topk_graph_record_recapture():
            self._selector_topk_graph_state = {"bypass": True}

    def _get_selector_layer_index_tensor(
        self,
        layer_indices: Sequence[int],
        device: torch.device,
    ) -> torch.Tensor:
        key = tuple(int(x) for x in layer_indices)
        cached = self._selector_layer_index_cache_tensor
        if (
            cached is not None
            and self._selector_layer_index_cache_key == key
            and self._selector_layer_index_cache_device == device
        ):
            return cached
        layer_index_tensor = torch.tensor(key, device=device, dtype=torch.long)
        self._selector_layer_index_cache_key = key
        self._selector_layer_index_cache_device = device
        self._selector_layer_index_cache_tensor = layer_index_tensor
        return layer_index_tensor

    @staticmethod
    def _sorted_list_is_subset(sorted_sup: Sequence[int], sorted_sub: Sequence[int]) -> bool:
        if not sorted_sub:
            return True
        if len(sorted_sub) > len(sorted_sup):
            return False
        i = 0
        j = 0
        sup_len = len(sorted_sup)
        sub_len = len(sorted_sub)
        while i < sup_len and j < sub_len:
            a = int(sorted_sup[i])
            b = int(sorted_sub[j])
            if a == b:
                i += 1
                j += 1
            elif a < b:
                i += 1
            else:
                return False
        return j == sub_len

    @staticmethod
    def _slots_to_rows_cpu(state: LayerState, slot_list: Sequence[int]) -> List[int]:
        if not slot_list:
            return []
        rows = state.slot_batch_rows_cpu
        if rows is None:
            raise RuntimeError("slot_batch_rows_cpu missing for slots")
        max_slot = max(slot_list)
        if max_slot >= len(rows):
            raise RuntimeError("slot_batch_rows_cpu missing rows for slots")
        return [rows[slot] if 0 <= slot < len(rows) else -1 for slot in slot_list]

    @staticmethod
    def _step_context_identity_token(step_context: object) -> int:
        token = int(getattr(step_context, "step_identity_token", 0))
        if token > 0:
            return token
        epoch = int(getattr(step_context, "epoch", -1))
        if epoch < 0:
            return -1
        return (
            epoch * 1_000_000_000
            + int(getattr(step_context, "step_handle_id", -1)) * 1_000_000
            + int(getattr(step_context, "step_handle_generation", -1))
        )

    def _slots_to_rows_for_step_context(
        self,
        *,
        step_context: object,
        state: LayerState,
        slot_list: Sequence[int],
    ) -> List[int]:
        if not slot_list:
            return []

        step_token = SelectorComputeMixin._step_context_identity_token(step_context)
        slot_sources = []
        step_envelope = getattr(step_context, "step_envelope_v2", None)
        if step_envelope is not None:
            slot_sources.append(getattr(step_envelope, "slot_by_row", None))
        step_authority = getattr(step_context, "step_authority", None)
        if step_authority is not None:
            slot_sources.append(getattr(step_authority, "slot_by_row", None))

        for slot_by_row in slot_sources:
            if slot_by_row is None:
                continue
            slot_tuple = tuple(int(v) for v in slot_by_row)
            if not slot_tuple:
                continue
            cached_key = tuple(getattr(self, "_step_context_slot_row_map_key", tuple()))
            slot_row_map = getattr(self, "_step_context_slot_row_map", None)
            if (
                int(getattr(self, "_step_context_slot_row_map_token", -1)) != step_token
                or cached_key != slot_tuple
                or not isinstance(slot_row_map, dict)
            ):
                slot_row_map = {
                    int(slot): int(row)
                    for row, slot in enumerate(slot_tuple)
                    if int(slot) >= 0
                }
                self._step_context_slot_row_map_token = step_token
                self._step_context_slot_row_map_key = slot_tuple
                self._step_context_slot_row_map = slot_row_map
            row_list = [slot_row_map.get(int(slot), -1) for slot in slot_list]
            if all(int(row) >= 0 for row in row_list):
                return [int(row) for row in row_list]

        return SelectorComputeMixin._slots_to_rows_cpu(state, slot_list)

    def _sync_step_bound_meta_logits(
        self,
        *,
        step_context: "StepContext",
    ) -> None:
        """把 step_authority 的 logits 行语义同步回 step_bound_meta（单源保持一致）。"""
        authority = step_context.step_authority
        step_bound_meta = self.step_bound_meta
        if authority is None or step_bound_meta is None:
            return
        if int(step_bound_meta.epoch) != int(step_context.epoch):
            return
        if (
            int(step_bound_meta.step_handle_id) != int(authority.step_handle_id)
            or int(step_bound_meta.step_handle_generation) != int(authority.step_handle_generation)
        ):
            return
        batch_size = int(authority.batch_size)
        logits_last_n = tuple(int(v) for v in authority.logits_last_n_by_row[:batch_size])
        logits_capacity = tuple(int(v) for v in authority.logits_capacity_by_row[:batch_size])
        if (
            step_bound_meta.logits_last_n_by_row == logits_last_n
            and step_bound_meta.logits_capacity_by_row == logits_capacity
        ):
            return
        step_bound_meta.logits_last_n_by_row = logits_last_n
        step_bound_meta.logits_capacity_by_row = logits_capacity
        step_bound_meta.plan_signature = tuple(authority.plan_signature)
        signature_prev = tuple(step_bound_meta.bound_meta_signature)
        if len(signature_prev) >= 9:
            signature_tail = tuple(signature_prev[9:])
            step_bound_meta.bound_meta_signature = (
                int(step_context.epoch),
                int(step_bound_meta.step_handle_id),
                int(step_bound_meta.step_handle_generation),
                signature_prev[3],
                tuple(step_bound_meta.logits_last_n_by_row),
                tuple(step_bound_meta.logits_capacity_by_row),
                tuple(int(v) for v in step_bound_meta.logf_mask_by_row),
                tuple(int(v) for v in step_bound_meta.q_start_loc),
                tuple(int(v) for v in step_bound_meta.prefill_rows),
            ) + signature_tail

    def prepare_step_logits_buffers(
        self,
        *,
        state: LayerState,
        step_context: StepContext,
        capture_plan_by_req: Optional[Dict[str, int]],
        seqused_k: torch.Tensor,
        max_seqlen_k: int,
        block_size: int,
        num_heads: int,
        device: torch.device,
    ) -> None:
        """在 step 级别为需要 logits 的 request 分配缓冲与 LogitSpec。"""
        _sit = int(step_context.step_identity_token)
        num_reqs = step_context.num_reqs
        if num_reqs <= 0:
            return

        if not step_context.seq_lens or len(step_context.seq_lens) < num_reqs:
            raise RuntimeError(
                "selector logits buffer missing seq_lens in default path; "
                f"epoch={int(step_context.epoch)} num_reqs={int(num_reqs)}"
            )
        q_lens_src = step_context.q_lens if step_context.q_lens else tuple()
        max_cap = int(max_seqlen_k)
        blk = max(0, int(block_size))
        step_authority = step_context.step_authority
        if step_authority is None or int(step_authority.epoch) != int(step_context.epoch):
            raise RuntimeError(
                "prepare_step_logits_buffers requires step_authority.layer_effective_refresh_by_row; "
                f"step_epoch={int(step_context.epoch)} auth_epoch="
                f"{-1 if step_authority is None else int(step_authority.epoch)}"
            )
        layer_effective_refresh_by_row = step_authority.layer_effective_refresh_by_row
        if len(layer_effective_refresh_by_row) < int(num_reqs):
            raise RuntimeError(
                "prepare_step_logits_buffers requires full layer_effective_refresh_by_row coverage; "
                f"rows={len(layer_effective_refresh_by_row)} num_reqs={int(num_reqs)}"
            )
        device_index = int(device.index) if device.index is not None else -1
        capture_plan_size = 0 if capture_plan_by_req is None else len(capture_plan_by_req)
        ready_input_signature: Tuple[object, ...] = (
            int(_sit),
            step_context.req_ids,
            step_context.seq_lens,
            q_lens_src,
            capture_plan_by_req,
            int(capture_plan_size),
            layer_effective_refresh_by_row,
            int(max_seqlen_k),
            int(block_size),
            int(num_heads),
            str(device.type),
            int(device_index),
            id(getattr(self, "config", None)),
        )
        if (
            int(self._step_logits_ready_token) == _sit
            and getattr(self, "_step_logits_ready_input_signature", None) == ready_input_signature
        ):
            return

        semantic_snapshot = self._get_step_semantic_snapshot()
        recent_cfg = int(semantic_snapshot.recent_tokens)
        seq_lens = [int(x) for x in step_context.seq_lens[:num_reqs]]
        q_lens = list(q_lens_src) if q_lens_src else [0] * num_reqs
        capture_last_n_by_req: Dict[str, int] = {}
        if capture_plan_by_req:
            # 直接使用 req_id 作为键，避免依赖 per-layer slot 映射。
            for rid, last_n in capture_plan_by_req.items():
                if _is_free_slot_id(rid):
                    continue
                ln = max(0, int(last_n))
                if ln > 0:
                    capture_last_n_by_req[str(rid)] = ln

        logits_last_n: List[int] = [0] * num_reqs
        logits_capacity: List[int] = [0] * num_reqs
        # [CHUNKED-CAPTURE-ACCUMULATE 2026-07-06] per-row 跨片累计元数据
        # （-1=非跨片行走原纯覆写路径；>=0=本片 reduce 的前片累计行数）。
        # 仅 capture 步、planned 行触碰 tracking——稳态 decode 步零新增成本。
        accum_prev_rows: List[int] = [-1] * num_reqs
        accum_prev_capacity: List[int] = [0] * num_reqs
        max_last_n = 0
        max_capacity = 0

        for idx, req_id in enumerate(step_context.req_ids):
            q_len = q_lens[idx] if idx < len(q_lens) else 0
            seq_len = seq_lens[idx] if idx < len(seq_lens) else 0
            if seq_len <= 0:
                continue
            last_n = 0
            is_decode_refresh = False
            # 1) prefill capture（bootstrap）：由 capture_plan 决定，不能依赖 q_len>1。
            #    关键场景：vLLM 可能调度 “prefill chunk 的最后一步 q_len==1”，此时 effective last_n 会被 clamp 成 1；
            #    但仍必须写 logits/log_f，否则会出现“最后一个 chunk 不 capture → 永远不 bootstrap”的静默错误。
            planned = capture_last_n_by_req.get(req_id)
            if planned is not None and int(planned) > 0:
                last_n = int(planned)
                if q_len > 0:
                    last_n = min(int(last_n), int(q_len))
                last_n = max(0, int(last_n))
            # 2) decode refresh：固定 last_n==1（logits-only），由 layer-effective refresh 行语义决定。
            elif bool(layer_effective_refresh_by_row[idx]):
                last_n = 1
                is_decode_refresh = True
            if last_n <= 0:
                continue
            is_prefill_capture = (planned is not None) and int(last_n) > 0
            # [CHUNKED-CAPTURE-ACCUMULATE 2026-07-06] 捕获窗跨 chunk 边界判定：
            # 本片非尾片（total>seq_len）或已有前片累计。跨片行的 reduce 域
            # （capacity/recent 裁剪）必须按最终序列长 total 计算，保证各片
            # 域一致，分片合并才与 no-chunk 单次 reduce 数学等价。
            accum_total = 0
            accum_prev = -1
            if is_prefill_capture:
                _trk = self._ensure_request(req_id)
                accum_total = int(getattr(_trk, "total_prompt_tokens", 0) or 0)
                _prev_committed = int(
                    getattr(_trk, "prefill_capture_rows_accum", 0) or 0
                )
                _accum_sit = int(getattr(_trk, "prefill_capture_accum_sit", -1))
                if _accum_sit == _sit:
                    # 同 step 重算：转移是累加/覆写，取本步开始前的视图幂等恢复。
                    _prev_committed = int(
                        getattr(_trk, "prefill_capture_accum_prev_rows", 0) or 0
                    )
                if accum_total > int(seq_len) or _prev_committed > 0:
                    accum_prev = _prev_committed
            capacity = min(int(seq_len), max_cap)
            cap_basis = int(seq_len)
            if accum_prev >= 0 and accum_total > int(seq_len):
                # 非尾片：域按 total 展宽（仍 clamp 回 seq_len——scratch 只有
                # 因果可见列）。recent>last_n 时展宽后的 recent_start(total)
                # 仍 < 本片 seq_len，各片域自然一致。
                cap_basis = int(accum_total)
                capacity = min(int(accum_total), max_cap)
            # P0.2：logits/log_f capture 裁剪 K 上限到 recent_start（block 对齐）
            # - decode-refresh（last_n==1）+ prefill capture（last_n>=1）均适用
            # - 仅影响 capture/selector/rebuild 的 K 维张量规模，不改语义（selector 只在 [sink, recent_start) 里选）
            # - 当 seq_len<=recent 时 recent_start==0，此时不裁剪，避免把 K 误裁成 0 导致 capture 被错误禁用
            if (is_decode_refresh or is_prefill_capture) and blk > 0 and recent_cfg > 0 and int(cap_basis) > recent_cfg:
                cap_recent = min(int(cap_basis), int(recent_cfg))
                recent_start = ((int(cap_basis) - int(cap_recent)) // int(blk)) * int(blk)
                if int(recent_start) > 0:
                    capacity = min(int(capacity), int(recent_start))
            capacity = min(int(capacity), int(seq_len))
            if capacity <= 0:
                continue
            if accum_prev >= 0:
                _trk = self._ensure_request(req_id)
                if int(getattr(_trk, "prefill_capture_accum_sit", -1)) != _sit:
                    _trk.prefill_capture_accum_prev_rows = int(accum_prev)
                    _trk.prefill_capture_accum_prev_capacity = int(
                        getattr(_trk, "prefill_capture_prev_capacity", 0) or 0
                    )
                    _trk.prefill_capture_accum_sit = int(_sit)
                # 幂等重写（同 sit 任意次重算恒 prev+本次 eff / 本片 capacity）。
                _trk.prefill_capture_rows_accum = int(accum_prev) + int(last_n)
                _trk.prefill_capture_prev_capacity = int(capacity)
                accum_prev_rows[idx] = int(accum_prev)
                accum_prev_capacity[idx] = int(
                    getattr(_trk, "prefill_capture_accum_prev_capacity", 0) or 0
                )
            logits_last_n[idx] = int(last_n)
            logits_capacity[idx] = int(capacity)
            max_last_n = max(max_last_n, int(last_n))
            max_capacity = max(max_capacity, int(capacity))

        if max_last_n <= 0 or max_capacity <= 0:
            if capture_plan_by_req:
                raise RuntimeError("prefill capture plan set but no logits buffers allocated")
            if (
                self._decode_logits_last_n_i64 is not None
                and self._decode_logits_last_n_i64.device == device
                and self._decode_logits_last_n_i64.numel() >= num_reqs
            ):
                self._decode_logits_last_n_i64[:num_reqs].zero_()
            if (
                self._decode_logits_cap_i64 is not None
                and self._decode_logits_cap_i64.device == device
                and self._decode_logits_cap_i64.numel() >= num_reqs
            ):
                self._decode_logits_cap_i64[:num_reqs].zero_()
            if step_context.step_authority is not None and step_context.step_authority.epoch == step_context.epoch:
                step_context.step_authority = step_context.step_authority.with_logits(
                    logits_last_n_by_row=tuple(logits_last_n),
                    logits_capacity_by_row=tuple(logits_capacity),
                )
                self.step_authority = step_context.step_authority
            # 必须在 step_authority.with_logits 之后调用，因为它读 step_authority 的 logits 字段
            self._sync_step_bound_meta_logits(
                step_context=step_context
            )
            self._step_capture_accum_prev_rows_by_row = tuple(accum_prev_rows)
            self._step_capture_accum_prev_capacity_by_row = tuple(accum_prev_capacity)
            self._step_capture_accum_token = int(step_context.epoch)
            self._step_logits_ready_token = _sit
            self._step_logits_ready_input_signature = ready_input_signature
            self._step_logits_ready_bound_signature = (
                tuple(self.step_bound_meta.bound_meta_signature)
                if self.step_bound_meta is not None
                else tuple()
            )
            return

        tensor_cap = max(int(getattr(self, "max_batch_size", 0) or 0), int(num_reqs))
        if (
            self._decode_logits_last_n_i64 is None
            or self._decode_logits_last_n_i64.device != device
            or self._decode_logits_last_n_i64.numel() < tensor_cap
        ):
            self._decode_logits_last_n_i64 = torch.empty(
                (tensor_cap,),
                dtype=torch.long,
                device=device,
            )
        if (
            self._decode_logits_cap_i64 is None
            or self._decode_logits_cap_i64.device != device
            or self._decode_logits_cap_i64.numel() < tensor_cap
        ):
            self._decode_logits_cap_i64 = torch.empty(
                (tensor_cap,),
                dtype=torch.long,
                device=device,
            )

        def _ensure_stage(name: str) -> torch.Tensor:
            stage = getattr(self, name, None)
            if (
                not isinstance(stage, torch.Tensor)
                or stage.device.type != "cpu"
                or stage.dtype != torch.long
                or stage.numel() < tensor_cap
            ):
                try:
                    stage = torch.empty((tensor_cap,), dtype=torch.long, device="cpu", pin_memory=True)
                except RuntimeError:
                    stage = torch.empty((tensor_cap,), dtype=torch.long, device="cpu")
                setattr(self, name, stage)
            return stage

        last_n_stage = _ensure_stage("_decode_logits_last_n_stage_cpu_i64")
        cap_stage = _ensure_stage("_decode_logits_cap_stage_cpu_i64")
        for idx in range(num_reqs):
            last_n_stage[idx] = int(logits_last_n[idx])
            cap_stage[idx] = int(logits_capacity[idx])
        self._decode_logits_last_n_i64[:num_reqs].copy_(
            last_n_stage[:num_reqs],
            non_blocking=True,
        )
        self._decode_logits_cap_i64[:num_reqs].copy_(
            cap_stage[:num_reqs],
            non_blocking=True,
        )
        if step_context.step_authority is not None and step_context.step_authority.epoch == step_context.epoch:
            step_context.step_authority = step_context.step_authority.with_logits(
                logits_last_n_by_row=tuple(logits_last_n),
                logits_capacity_by_row=tuple(logits_capacity),
            )
            self.step_authority = step_context.step_authority
        # 必须在 step_authority.with_logits 之后调用，因为它读 step_authority 的 logits 字段
        self._sync_step_bound_meta_logits(
            step_context=step_context
        )
        # [CHUNKED-CAPTURE-ACCUMULATE 2026-07-06] step 级物化（epoch 对账）：
        # postprocess 提交点（inline/deferred job 构建均在本 step 内）由此读取，
        # deferred 执行时步态已换代故必须在提交步冻结进 job。
        self._step_capture_accum_prev_rows_by_row = tuple(accum_prev_rows)
        self._step_capture_accum_prev_capacity_by_row = tuple(accum_prev_capacity)
        self._step_capture_accum_token = int(step_context.epoch)
        self._step_logits_ready_token = _sit
        self._step_logits_ready_input_signature = ready_input_signature
        self._step_logits_ready_bound_signature = (
            tuple(self.step_bound_meta.bound_meta_signature)
            if self.step_bound_meta is not None
            else tuple()
        )

    def _update_prefill_capture_state_by_req(
        self,
        *,
        step_context: StepContext,
    ) -> Tuple[Dict[str, int], Tuple[str, ...]]:
        """构建 step 级 prefill capture 计划（按 req_id，CPU-only，禁止 DtoH 同步）。

        设计原则：
        - 计划与阶段判定使用 StepContext（CPU tuple）与 RequestTracking（CPU）为真源；
        - 不读取任何 GPU tensor 的 .item()/.tolist()；
        - 不依赖 per-layer slot 索引，避免多 request 下跨层 slot 映射不一致导致的静默错误。
        """
        if step_context.num_reqs <= 0 or not step_context.req_ids:
            return {}, tuple()

        configured = int(self.config.prefill_last_n_query or 0) if self.config is not None else 0
        capture_plan_by_req: Dict[str, int] = {}
        finalize_req_ids: List[str] = []

        req_ids = step_context.req_ids
        q_lens = step_context.q_lens
        seq_lens = step_context.seq_lens
        num_reqs = int(step_context.num_reqs)
        step_authority = getattr(step_context, "step_authority", None)
        is_prefill_by_row = tuple(
            bool(v) for v in getattr(step_authority, "is_prefill_by_row", tuple())
        )

        compact_threshold = self._compact_threshold_tokens()

        for idx in range(num_reqs):
            req_id = req_ids[idx] if idx < len(req_ids) else f"prefill:{idx}"
            tracking = self._ensure_request(req_id)
            if idx < len(is_prefill_by_row) and (not bool(is_prefill_by_row[idx])):
                continue
            # 仅以 bootstrap_done 作为“全层可用”的完成信号：
            # - prefill_done 可能在 chunk 级异步流水线中提前置位（仅代表某个 chunk 已 enqueue），
            #   若在这里跳过会导致后续 chunk 的层不再 capture，从而出现“只 bootstrap 前几层”的静默错误。
            if bool(getattr(tracking, "bootstrap_done", False)):
                continue

            used_tokens = int(seq_lens[idx]) if idx < len(seq_lens) else 0
            if used_tokens <= 0:
                continue

            total_req = int(tracking.total_prompt_tokens or 0)
            if total_req <= 0:
                total_req = used_tokens
            if total_req < used_tokens:
                total_req = used_tokens

            remaining_tokens = max(0, total_req - used_tokens)

            chunk_size = int(tracking.prompt_chunk_size or 0)
            if chunk_size <= 0:
                q_len = int(q_lens[idx]) if idx < len(q_lens) else 1
                chunk_size = max(1, q_len)
                tracking.prompt_chunk_size = int(chunk_size)

            # prefill 阶段无需依赖 per-layer compact_ready 判定 short_dense：
            # - compact 的建立由 prefill capture + refresh_stream 异步完成；
            # - 若 slot 复用/清理存在 bug 导致某层 compact_ready 意外为 True，这属于更高优先级的错误，
            #   不应让 plan 与层相关而产生静默分叉。
            # 与 decode 阶段统一：<= threshold 仍属于 short。
            short_dense = bool(compact_threshold > 0 and int(used_tokens) <= compact_threshold)

            if configured > 0:
                if short_dense:
                    tracking.prefill_capture_ready = False
                    tracking.prefill_capture_last_n = 0
                else:
                    capture_last_n = int(tracking.prefill_capture_last_n or 0)
                    if capture_last_n <= 0:
                        capture_last_n = int(configured)
                    capture_last_n = max(0, int(capture_last_n))
                    if capture_last_n > 0:
                        tracking.prefill_capture_last_n = int(capture_last_n)
                        should_capture = (remaining_tokens == 0) or (remaining_tokens < capture_last_n)
                        if should_capture:
                            tracking.prefill_capture_ready = True
                            # [CHUNKED-CAPTURE-ACCUMULATE 2026-07-06] 捕获窗 =
                            # 全 prompt 的最后 last_n 个 query。窗跨 chunk 边界
                            # （remaining>0 即本片非尾片）时本片只捕获窗口与本
                            # chunk 的交集 = 尾部 last_n-remaining 行（store 的
                            # row_offset=q_len-eff 恰好对齐交集起点）；各片经
                            # reduce accumulate 模式（sum-form 计数加权 LSE 合并）
                            # 拼成与 no-chunk 单次 reduce 数学等价的全窗结果。
                            # 旧行为（每片恒 min(last_n,q_len)）使首片捕获窗错位
                            # 且被尾片纯覆写——selection 只看尾片 query。
                            eff = int(capture_last_n)
                            if remaining_tokens > 0:
                                eff = int(capture_last_n) - int(remaining_tokens)
                            q_len_eff = int(q_lens[idx]) if idx < len(q_lens) else 0
                            if q_len_eff > 0:
                                # prefill capture 语义始终是“当前 chunk 的尾部 eff 行”。
                                # 即便命中 final/tail chunk，也不能把 width 放大成整个 q_len，
                                # 否则会把 tail-window 语义偷换成 full-chunk capture。
                                eff = min(int(eff), int(q_len_eff))
                            eff = max(0, int(eff))
                            tracking.prefill_chunks_seen = (
                                int(getattr(tracking, "prefill_chunks_seen", 0) or 0) + 1
                            )
                            if (
                                remaining_tokens > 0
                                and int(tracking.prefill_chunks_seen) == 1
                            ):
                                _log.info(
                                    "prefill capture window splits across chunks for "
                                    "request %s (remaining=%d < last_n=%d): pieces are "
                                    "merged via the reduce accumulate mode",
                                    req_id,
                                    int(remaining_tokens),
                                    int(capture_last_n),
                                )
                            if eff > 0:
                                capture_plan_by_req[req_id] = int(eff)

            if remaining_tokens == 0 and bool(getattr(tracking, "prefill_capture_ready", False)):
                # [CHUNKED-CAPTURE-ACCUMULATE 2026-07-06] 跨片对账 fail-fast：
                # 前片累计（prepare_step_logits_buffers 幂等推进；本步已推进则取
                # prev 视图）+ 本步尾片 eff 必须凑齐整窗 min(last_n, total)。
                # 不齐=有片被调度丢弃/错斜，selection 会静默用残窗——宁断不糊。
                _accum_sit = int(getattr(tracking, "prefill_capture_accum_sit", -1))
                _step_sit = int(getattr(step_context, "step_identity_token", -2))
                accum_prev = int(
                    getattr(tracking, "prefill_capture_accum_prev_rows", 0)
                    if _accum_sit == _step_sit
                    else getattr(tracking, "prefill_capture_rows_accum", 0) or 0
                )
                if accum_prev > 0:
                    cap_ln = int(getattr(tracking, "prefill_capture_last_n", 0) or 0)
                    expected_rows = min(cap_ln, int(total_req)) if cap_ln > 0 else 0
                    tail_eff = int(capture_plan_by_req.get(req_id, 0) or 0)
                    if expected_rows > 0 and accum_prev + tail_eff != expected_rows:
                        raise RuntimeError(
                            "chunked prefill capture accumulate mismatch for request "
                            f"{req_id}: prev_rows={accum_prev} + tail_eff={tail_eff} "
                            f"!= expected={expected_rows} (last_n={cap_ln}, "
                            f"total_prompt_tokens={int(total_req)})"
                        )
                finalize_req_ids.append(str(req_id))

        return capture_plan_by_req, tuple(finalize_req_ids)

    def get_step_prefill_plan_by_req(
        self,
        *,
        step_context: StepContext,
    ) -> Tuple[Dict[str, int], Tuple[str, ...]]:
        """step 级共享 prefill capture 计划（按 req_id），保证各层一致。"""
        step_handle_id = int(getattr(step_context, "step_handle_id", -1))
        step_handle_generation = int(getattr(step_context, "step_handle_generation", -1))
        if (
            self.step_prefill_plan_epoch == self.step_context_epoch
            and int(getattr(self, "step_prefill_plan_handle_id", -1)) == step_handle_id
            and int(getattr(self, "step_prefill_plan_handle_generation", -1))
            == step_handle_generation
        ):
            return self.step_prefill_capture_plan_by_req, self.step_prefill_finalize_req_ids

        capture_plan_by_req, finalize_req_ids = self._update_prefill_capture_state_by_req(step_context=step_context)
        self.step_prefill_plan_epoch = self.step_context_epoch
        self.step_prefill_plan_handle_id = step_handle_id
        self.step_prefill_plan_handle_generation = step_handle_generation
        self.step_prefill_capture_plan_by_req = capture_plan_by_req
        self.step_prefill_finalize_req_ids = finalize_req_ids
        return capture_plan_by_req, finalize_req_ids

    def _ensure_step_prefill_capture_last_n_by_row(
        self,
        *,
        step_context: StepContext,
        capture_plan_active_by_req: Optional[Dict[str, int]],
    ) -> None:
        """构建并缓存 step 级 per-row last_n（仅针对 active capture 计划）。"""
        epoch = self.step_context_epoch
        handle_id = int(getattr(step_context, "step_handle_id", -1))
        handle_generation = int(getattr(step_context, "step_handle_generation", -1))
        if (
            self.step_prefill_capture_last_n_epoch == epoch
            and int(getattr(self, "step_prefill_capture_last_n_handle_id", -1))
            == handle_id
            and int(getattr(self, "step_prefill_capture_last_n_handle_generation", -1))
            == handle_generation
        ):
            return
        num_reqs = int(getattr(step_context, "num_reqs", 0) or 0)
        if num_reqs <= 0:
            self.step_prefill_capture_last_n_by_row = None
            self.step_prefill_capture_last_n_epoch = epoch
            self.step_prefill_capture_last_n_handle_id = handle_id
            self.step_prefill_capture_last_n_handle_generation = handle_generation
            return
        if not capture_plan_active_by_req:
            self.step_prefill_capture_last_n_by_row = tuple(0 for _ in range(num_reqs))
            self.step_prefill_capture_last_n_epoch = epoch
            self.step_prefill_capture_last_n_handle_id = handle_id
            self.step_prefill_capture_last_n_handle_generation = handle_generation
            return
        req_ids = step_context.req_ids
        last_n_by_row: List[int] = [0 for _ in range(num_reqs)]
        for row in range(num_reqs):
            if row >= len(req_ids):
                break
            rid = req_ids[row]
            ln = int(capture_plan_active_by_req.get(str(rid), 0) or 0)
            if ln > 0:
                last_n_by_row[row] = int(ln)
        self.step_prefill_capture_last_n_by_row = tuple(last_n_by_row)
        self.step_prefill_capture_last_n_epoch = epoch
        self.step_prefill_capture_last_n_handle_id = handle_id
        self.step_prefill_capture_last_n_handle_generation = handle_generation

    def _create_profile_events_batch(
        self,
        n: int,
        device: torch.device,
        profile_enabled: bool,
    ) -> List[Optional[torch.cuda.Event]]:
        """批量创建 n 个 CUDA event，减少重复的 try/except 和条件检查开销。"""
        if not profile_enabled or device.type != "cuda":
            return [None] * n
        try:
            if torch.cuda.is_current_stream_capturing():
                return [None] * n
            return [torch.cuda.Event(enable_timing=True) for _ in range(n)]
        except Exception:
            _log.warning("_create_events_batch failed for n=%d", n, exc_info=True)
            raise

    def _record_event_safe(
        self,
        evt: Optional[torch.cuda.Event],
        device: torch.device,
    ) -> None:
        """安全记录单个 event，无需外层 try/except。"""
        if evt is not None:
            try:
                evt.record(torch.cuda.current_stream(device=device))
            except Exception:
                _log.warning("_record_event_safe failed for device=%s", device, exc_info=True)
                raise

    def _build_pipeline_cfg_signature(self, cfg: AlphaFairSelectorConfig) -> Tuple[object, ...]:
        """Build value-based signature for selector pipeline cache invalidation."""
        if cfg is None:
            raise ValueError("selector pipeline cfg is required")
        return (
            float(cfg.alpha),
            float(cfg.eps),
            float(cfg.gamma),
            float(cfg.prior_weight_l2),
            float(cfg.prior_weight_pos),
            float(cfg.prior_pos_power),
            float(cfg.prior_pos_eta),
            float(cfg.beta()),
            float(cfg.lambda_clip_single),
            float(cfg.lambda_clip_multi),
            float(cfg.lambda_tail_kappa),
            float(cfg.lambda_tail_pivot),
            bool(cfg.lambda_soft),
            int(cfg.nms_window),
            float(cfg.soft_alpha),
            float(cfg.cross_head_alpha),
            float(cfg.cross_head_temperature),
        )

    def _compute_alpha_selection_batched_layers(
        self,
        *,
        capture_scores: torch.Tensor,
        log_f_denoms: Optional[torch.Tensor],
        kv_lengths: torch.Tensor,
        key_norms_full: torch.Tensor,
        num_kv_heads: int,
        num_queries_per_kv: int,
        block_size: int,
        phase: str,
        topk_slice_start: Optional[int] = None,
        topk_slice_end: Optional[int] = None,
        seq_lens_full: Optional[torch.Tensor] = None,
        seq_lens_cpu: Optional[Sequence[int]] = None,
        seq_lens_tensor_cpu: Optional[torch.Tensor] = None,  # 方案 A 优化：复用预创建的 tensor
        profile_detail: bool = False,
        return_selected_token_scores: bool = False,
    ) -> tuple:
        if self.config is None or self.config.alpha_fair is None:
            raise RuntimeError("alpha selector requires config.alpha_fair")

        layers, batch_size, num_heads, window, kv_len_total = capture_scores.shape
        selection_mode = str(getattr(self.config.alpha_fair, "selection_mode", "token_topk"))
        if num_heads != num_kv_heads * num_queries_per_kv:
            raise ValueError("num_heads mismatch with kv heads")
        if log_f_denoms is not None and log_f_denoms.shape != (layers, batch_size, num_heads):
            raise ValueError("log_f_denoms must have shape [layers, batch, num_heads]")
        if kv_lengths.shape != (layers, batch_size, num_heads):
            raise ValueError("kv_lengths must have shape [layers, batch, num_heads]")

        device = capture_scores.device

        # 统一 pipeline 路径：一次 Python→GPU 调用完成全部 selector 计算
        # 仅在 cross_head_power==0 时启用（pipeline 不支持 cross_head_power）
        use_pipeline_unified = _should_use_unified_pipeline_for_selection_mode(
            selection_mode=selection_mode,
            unified_pipeline_enabled=_SELECTOR_PIPELINE_UNIFIED_CACHED,
        )
        cross_head_power = float(getattr(self.config.alpha_fair, "cross_head_power", 0.0) or 0.0)
        decode_bounds_enabled = _DECODE_BOUNDS_KERNEL_CACHED
        if selection_mode == "token_topk" and (
            not use_pipeline_unified
            or cross_head_power != 0.0
            or (log_f_denoms is not None and window != 1)
        ):
            raise RuntimeError(
                "token_topk requires the unified CUDA selector pipeline; "
                "legacy selector fallback is retired"
            )
        if (
            use_pipeline_unified
            and cross_head_power == 0.0
            and window == 1
            and not decode_bounds_enabled
        ):
            raise RuntimeError(
                "unified selector W=1 requires VLLM_SPARSE_DECODE_BOUNDS_KERNEL=1; "
                "unified selector does not fall back"
            )
        # unified pipeline 支持 logits 全路径；pre_denom 仅在 W==1 decode bounds-first 路径支持
        if use_pipeline_unified and cross_head_power == 0.0 and (log_f_denoms is None or window == 1):
            try:
                normalized_result = _normalize_selection_layers_result(
                    self._compute_alpha_selection_pipeline_unified(
                        capture_scores=capture_scores,
                        log_f_denoms=log_f_denoms,
                        kv_lengths=kv_lengths,
                        key_norms_full=key_norms_full,
                        num_kv_heads=num_kv_heads,
                        num_queries_per_kv=num_queries_per_kv,
                        block_size=block_size,
                        topk_slice_start=topk_slice_start,
                        topk_slice_end=topk_slice_end,
                        seq_lens_full=seq_lens_full,
                        seq_lens_cpu=seq_lens_cpu,
                        seq_lens_tensor_cpu=seq_lens_tensor_cpu,
                        profile_detail=profile_detail,
                    )
                )
                if bool(return_selected_token_scores):
                    return normalized_result
                return (
                    normalized_result[0],
                    normalized_result[1],
                    normalized_result[2],
                    normalized_result[3],
                    normalized_result[4],
                    normalized_result[5],
                    normalized_result[6],
                    normalized_result[8],
                )
            except Exception as exc:
                raise RuntimeError(
                    "Unified selector pipeline failed; legacy selector fallback is retired; "
                    f"root_cause={exc!r}"
                ) from exc

        # 方案 I 优化：批量创建 profile events，减少重复条件检查
        # 扩展到10个event: 原有6个 + 4个细粒度计时(seq_full_0/1, pure_preproc_0/1)
        _profile_evts = self._create_profile_events_batch(10, device, profile_detail)
        preproc_evt0, preproc_evt1, log_s_evt0, log_s_evt1, topk_evt0, topk_evt1 = _profile_evts[:6]
        seq_full_evt0, seq_full_evt1, pure_preproc_evt0, pure_preproc_evt1 = _profile_evts[6:10]

        self._record_event_safe(preproc_evt0, device)

        # 方案 B 优化：提前转为 int32，避免后续多次 dtype 转换
        # 方案 D 优化：预处理在 C++ 薄封装中完成（保持语义不变），减少 Python 发射开销
        semantic_snapshot = self._get_step_semantic_snapshot()
        sink_cfg = int(semantic_snapshot.sink_tokens)
        recent_cfg = int(semantic_snapshot.recent_tokens)

        # 细粒度计时：seq_full准备开始
        self._record_event_safe(seq_full_evt0, device)
        seq_full: Optional[torch.Tensor] = None
        if block_size > 0 and recent_cfg > 0:
            if (
                seq_lens_full is not None
                and isinstance(seq_lens_full, torch.Tensor)
                and seq_lens_full.numel() >= batch_size
            ):
                seq_full = seq_lens_full[:batch_size]
                if seq_full.device != device:
                    seq_full = seq_full.to(device=device)
                if seq_full.dtype != torch.int32:
                    seq_full = seq_full.to(dtype=torch.int32)
                seq_full = seq_full.view(1, batch_size, 1).expand(layers, batch_size, num_kv_heads)
            elif seq_lens_tensor_cpu is not None and seq_lens_tensor_cpu.numel() >= batch_size:
                # 方案 A 优化：复用预创建的 seq_lens_tensor，只需 to(device)
                # 方案 B：改用 int32
                seq_full = (
                    seq_lens_tensor_cpu[:batch_size]
                    .to(device=device, dtype=torch.int32)
                    .view(1, batch_size, 1)
                    .expand(layers, batch_size, num_kv_heads)
                )
            elif seq_lens_cpu is not None and len(seq_lens_cpu) >= batch_size:
                seq_full = torch.tensor(
                    [max(0, int(seq_lens_cpu[i])) for i in range(batch_size)],
                    device=device,
                    dtype=torch.int32,
                ).view(1, batch_size, 1).expand(layers, batch_size, num_kv_heads)

        # 细粒度计时：seq_full准备结束
        self._record_event_safe(seq_full_evt1, device)

        kv_lengths_by_kv_i32: torch.Tensor
        kv_len_head: torch.Tensor
        head_sink: torch.Tensor
        recent_start: torch.Tensor
        allowed_lengths: torch.Tensor
        row_lo: torch.Tensor
        row_hi: torch.Tensor
        use_cpp_preproc = _compute_use_cpp_preproc(selection_mode=selection_mode)

        # 细粒度计时：纯preproc_bounds计算开始
        self._record_event_safe(pure_preproc_evt0, device)

        preproc_out = None
        if use_cpp_preproc:
            _ext = _get_selector_batch_ext()
            if _ext is not None:
                try:
                    preproc_out = _ext.preproc_bounds(
                        kv_lengths=kv_lengths,
                        num_kv_heads=num_kv_heads,
                        num_queries_per_kv=num_queries_per_kv,
                        kv_len_total=kv_len_total,
                        sink_cfg=sink_cfg,
                        recent_cfg=recent_cfg,
                        block_size=block_size,
                        window=window,
                        seq_full=seq_full,
                    )
                except Exception:
                    _log.warning("C++ preproc_bounds failed", exc_info=True)
                    raise
        if preproc_out is not None and isinstance(preproc_out, tuple) and len(preproc_out) == 7:
            (
                kv_lengths_by_kv_i32,
                kv_len_head,
                head_sink,
                recent_start,
                allowed_lengths,
                row_lo,
                row_hi,
            ) = preproc_out
        elif use_cpp_preproc:
            raise RuntimeError(
                "selector cpp preproc is enabled but unavailable"
            )
        else:
            kv_lengths_by_kv = kv_lengths.reshape(layers, batch_size, num_kv_heads, num_queries_per_kv)
            kv_lengths_by_kv_i32 = kv_lengths_by_kv.to(dtype=torch.int32)
            kv_len_head = kv_lengths_by_kv_i32.max(dim=3).values.clamp(max=kv_len_total)

            head_sink = torch.clamp(kv_len_head, max=max(0, sink_cfg))

            # recent_start：必须基于“真实 seq_len”（而不是 capture K 上限）计算，否则当 capture 已裁掉 recent
            # 时会发生“重复扣 recent”的静默错误（把 [recent_start-256, recent_start) 也当作 recent 丢弃）。
            #
            # - seq_lens_cpu 来自 StepContext（CPU），不触发 DtoH，同一 batch 内跨层一致；
            # - 最终 recent_start 仍需 clamp 到 kv_len_total（capture 的 K 维上限）。
            # 方案 B：recent_start 也保持 int32
            if block_size > 0 and recent_cfg > 0:
                if seq_full is not None:
                    cap_full = torch.clamp(seq_full, max=int(recent_cfg))
                    rs_full = ((seq_full - cap_full) // int(block_size)) * int(block_size)
                    rs_full = torch.clamp(rs_full, min=0, max=int(kv_len_total))
                    recent_start = rs_full
                else:
                    cap = torch.clamp(kv_len_head, max=int(recent_cfg))
                    recent_start = ((kv_len_head - cap) // int(block_size)) * int(block_size)
                    recent_start = torch.clamp(recent_start, min=0, max=int(kv_len_total))
            else:
                recent_start = kv_len_head.new_zeros(kv_len_head.shape)

            allowed_lengths = torch.clamp(recent_start - head_sink, min=0)

            # bounds：不物化 [L,B,Hkv,Q,W,K] 的 allowed_mask（巨量显存 + 带宽），而是为每个 row
            # 提供允许的区间 [row_lo, row_hi)。这里 row 表示 (query, window_pos) 的笛卡尔积。
            #
            # 旧语义：allowed_mask = length & stair & ~(sink|over|recent)
            # 由于 sink/recent 是“永远保留”，selector 只在 middle 段 [sink, recent_start) 里挑 token；
            # 该有效域天然是连续区间，可用 bounds 精确表达。
            #
            # 方案 B：head_sink/kv_len_head/recent_start 已是 int32，无需转换
            # row_lo/hi: [L,B,Hkv,Q,W]（int32）
            row_lo = head_sink.unsqueeze(-1).unsqueeze(-1).expand(
                layers, batch_size, num_kv_heads, num_queries_per_kv, window
            )

            # row_hi 由以下约束取 min：
            # - per query kv_len（kv_lengths_by_kv_i32）
            # - stair 上界（kv_len_head - tail_offset）
            # - recent_start（排除 recent 段）
            # 最后 clamp 到 [0, kv_len_total]
            kv_len_q = kv_lengths_by_kv_i32.unsqueeze(-1).expand(
                layers, batch_size, num_kv_heads, num_queries_per_kv, window
            )
            row_hi = kv_len_q
            if window > 0:
                # 使用缓存的常量 tensor，避免每次 refresh 重新分配
                tail_offsets = self._get_tail_offsets(window=window, device=device)
                hi_stair = kv_len_head.unsqueeze(-1).unsqueeze(-1) - tail_offsets
                hi_stair = hi_stair.expand(layers, batch_size, num_kv_heads, num_queries_per_kv, window)
                row_hi = torch.minimum(row_hi, hi_stair)
            row_hi = torch.minimum(
                row_hi,
                recent_start.unsqueeze(-1).unsqueeze(-1).expand(
                    layers, batch_size, num_kv_heads, num_queries_per_kv, window
                ),
            )
            row_hi = torch.clamp(row_hi, min=0, max=int(kv_len_total))

        # 细粒度计时：纯preproc_bounds计算结束（覆盖C++和Python两个路径）
        self._record_event_safe(pure_preproc_evt1, device)

        positions_token = self._get_positions_i32(kv_len=kv_len_total, device=device)

        scores_kv = capture_scores.reshape(
            layers, batch_size, num_kv_heads, num_queries_per_kv, window, kv_len_total
        )

        # 方案 I 优化：使用辅助方法记录 event
        # preproc 计时覆盖：kv_len_head/head_sink/recent_start/row_lo/row_hi 等张量算子
        self._record_event_safe(preproc_evt1, device)
        self._record_event_safe(log_s_evt0, device)

        log_s_detail_events: Optional[
            Dict[str, Tuple[Optional[torch.cuda.Event], Optional[torch.cuda.Event]]]
        ] = None
        log_s_adjusted, log_s_detail_events = self._compute_alpha_log_s_batched_layers(
            scores_kv=scores_kv,
            log_f_denoms=log_f_denoms,
            row_lo=row_lo,
            row_hi=row_hi,
            token_lo=head_sink,
            token_hi=recent_start,
            key_norms_full=key_norms_full,
            positions_token=positions_token,
            apply_cross_head=True,
            profile_detail=bool(profile_detail),
        )
        self._record_event_safe(log_s_evt1, device)
        self._record_event_safe(topk_evt0, device)

        selected_middle_pages = None
        selected_middle_counts = None
        selected_token_scores = None
        if selection_mode == "token_topk":
            k_head_cfg = compact_recent_effective_k_head(
                k_head=int(self.config.alpha_fair.k_head or 0),
                sink_tokens=int(sink_cfg),
                attn_mode=str(getattr(self.config, "attn_mode", "compact_recent")),
            )
            selected_indices_batch, selected_token_scores = self._select_alpha_topk_batched_with_scores(
                log_s_adjusted=log_s_adjusted,
                head_sink=head_sink,
                recent_start=recent_start,
                kv_len_head=kv_len_head,
                positions_token=positions_token,
                k_head_cfg=k_head_cfg,
                slice_start=topk_slice_start,
                slice_end=topk_slice_end,
            )
        else:
            raise ValueError(f"unsupported alpha selection_mode: {selection_mode}")
        self._record_event_safe(topk_evt1, device)

        profile_events: Optional[Dict[str, Tuple[Optional[torch.cuda.Event], Optional[torch.cuda.Event]]]] = None
        if profile_detail:
            profile_events = {
                "preproc": (preproc_evt0, preproc_evt1),
                "seq_full": (seq_full_evt0, seq_full_evt1),
                "pure_preproc": (pure_preproc_evt0, pure_preproc_evt1),
                "log_s": (log_s_evt0, log_s_evt1),
                "log_s_triton": (None, None),
                "log_s_mask": (None, None),
                "log_s_cross": (None, None),
                "topk": (topk_evt0, topk_evt1),
            }
            if log_s_detail_events is not None:
                for key in ("log_s_triton", "log_s_mask", "log_s_cross"):
                    pair = log_s_detail_events.get(key)
                    if isinstance(pair, tuple) and len(pair) == 2:
                        profile_events[key] = pair
        result = (
            selected_indices_batch,
            head_sink,
            recent_start,
            kv_len_head,
            allowed_lengths,
            selected_middle_pages,
            selected_middle_counts,
            selected_token_scores,
            profile_events,
        )
        if bool(return_selected_token_scores):
            return result
        return (
            result[0],
            result[1],
            result[2],
            result[3],
            result[4],
            result[5],
            result[6],
            result[8],
        )

    def _compute_alpha_log_s_batched_layers(
        self,
        *,
        scores_kv: torch.Tensor,
        log_f_denoms: Optional[torch.Tensor],
        row_lo: torch.Tensor,
        row_hi: torch.Tensor,
        token_lo: torch.Tensor,
        token_hi: torch.Tensor,
        key_norms_full: torch.Tensor,
        positions_token: torch.Tensor,
        apply_cross_head: bool = True,
        profile_detail: bool = False,
    ) -> Tuple[
        torch.Tensor,
        Optional[Dict[str, Tuple[Optional[torch.cuda.Event], Optional[torch.cuda.Event]]]],
    ]:
        """Selector 阶段 A：计算 log_s（含 cross-head mutex），保持语义与旧路径一致。"""
        if self.config is None or self.config.alpha_fair is None:
            raise RuntimeError("alpha selector requires config.alpha_fair")

        log_s_batch: Optional[torch.Tensor] = None
        device = scores_kv.device
        # 方案 I 优化：批量创建 profile events
        _log_s_evts = self._create_profile_events_batch(6, device, profile_detail)
        log_s_triton_evt0, log_s_triton_evt1, log_s_mask_evt0, log_s_mask_evt1, log_s_cross_evt0, log_s_cross_evt1 = _log_s_evts

        row_dim = int(scores_kv.shape[3]) * int(scores_kv.shape[4])
        if log_s_batch is None:
            if not scores_kv.is_cuda:
                raise RuntimeError("alpha selector requires CUDA tensors (torch fallback removed)")
            if row_dim > 128:
                raise RuntimeError(
                    "legacy selector log_s path only supports rows<=128; "
                    "no fallback is available"
                )

            # 方案 I 优化：使用辅助方法记录 event
            self._record_event_safe(log_s_triton_evt0, device)
            if log_f_denoms is None:
                log_s_batch, _ = compute_alpha_scores_batched_layers_triton_logf_bounds(
                    scores_kv,
                    row_lo,
                    row_hi,
                    key_norms_full,
                    positions_token,
                    self.config.alpha_fair,
                    return_mean_probs=False,
                )
            else:
                # last_n>1：kernel 输出 log_f_pre + denom_f（按 query head）。
                # 这里使用 denom 在 Triton 内恢复每行 log_probs，并完成 rows→log_f 聚合，避免：
                # - torch.log_softmax 回退
                # - 物化大张量 (log_probs = log_f_pre - denom)
                if log_f_denoms.dim() != 3:
                    raise ValueError("log_f_denoms must be [L, B, Hq]")
                layers, batch, num_kv_heads, num_queries_per_kv, window, _ = scores_kv.shape
                if window != 1:
                    raise RuntimeError("log_f_pre+denom path requires window==1")
                if log_f_denoms.shape != (layers, batch, num_kv_heads * num_queries_per_kv):
                    raise ValueError("log_f_denoms shape mismatch with scores_kv heads")
                denoms_rows = log_f_denoms.reshape(layers, batch, num_kv_heads, num_queries_per_kv, 1)
                log_s_batch, _ = compute_alpha_scores_batched_layers_triton_logf_pre_denom_bounds(
                    scores_kv,
                    denoms_rows,
                    row_lo,
                    row_hi,
                    key_norms_full,
                    positions_token,
                    self.config.alpha_fair,
                    return_mean_probs=False,
                )
            self._record_event_safe(log_s_triton_evt1, device)

        layers, batch_size, num_kv_heads, kv_len_total = log_s_batch.shape
        # token_lo/token_hi：用于 cross-head mutex 的有效域裁剪。
        # 方案 I 优化：使用辅助方法记录 event
        log_s_adjusted = log_s_batch
        if apply_cross_head:
            self._record_event_safe(log_s_mask_evt0, device)
            flat_log_s = log_s_batch.reshape(layers * batch_size, num_kv_heads, kv_len_total)
            flat_lo = token_lo.reshape(layers * batch_size, num_kv_heads)
            if flat_lo.device != device:
                flat_lo = flat_lo.to(device=device)
            if flat_lo.dtype != torch.int32:
                flat_lo = flat_lo.to(dtype=torch.int32)
            flat_hi = token_hi.reshape(layers * batch_size, num_kv_heads)
            if flat_hi.device != device:
                flat_hi = flat_hi.to(device=device)
            if flat_hi.dtype != torch.int32:
                flat_hi = flat_hi.to(dtype=torch.int32)
            self._record_event_safe(log_s_mask_evt1, device)
            self._record_event_safe(log_s_cross_evt0, device)
            log_s_adjusted = apply_cross_head_mutex_bounds(
                flat_log_s,
                flat_lo,
                flat_hi,
                self.config.alpha_fair.cross_head_alpha,
                self.config.alpha_fair.cross_head_temperature,
                self.config.alpha_fair.nms_window,
                self.config.alpha_fair.cross_head_power,
            ).reshape(layers, batch_size, num_kv_heads, kv_len_total)
            self._record_event_safe(log_s_cross_evt1, device)

        profile_events: Optional[Dict[str, Tuple[Optional[torch.cuda.Event], Optional[torch.cuda.Event]]]] = None
        if profile_detail:
            profile_events = {
                "log_s_triton": (log_s_triton_evt0, log_s_triton_evt1),
                "log_s_mask": (log_s_mask_evt0, log_s_mask_evt1),
                "log_s_cross": (log_s_cross_evt0, log_s_cross_evt1),
            }
        return log_s_adjusted, profile_events

    def _select_alpha_topk_batched(
        self,
        *,
        log_s_adjusted: torch.Tensor,
        head_sink: torch.Tensor,
        recent_start: torch.Tensor,
        kv_len_head: torch.Tensor,
        positions_token: torch.Tensor,
        k_head_cfg: int,
        slice_start: Optional[int] = None,
        slice_end: Optional[int] = None,
    ) -> torch.Tensor:
        """Selector 阶段 B：从 log_s 中选择 top-k indices（保持旧逻辑）。"""
        topk_idx, _ = self._select_alpha_topk_batched_with_scores(
            log_s_adjusted=log_s_adjusted,
            head_sink=head_sink,
            recent_start=recent_start,
            kv_len_head=kv_len_head,
            positions_token=positions_token,
            k_head_cfg=k_head_cfg,
            slice_start=slice_start,
            slice_end=slice_end,
        )
        return topk_idx

    def _select_alpha_topk_batched_with_scores(
        self,
        *,
        log_s_adjusted: torch.Tensor,
        head_sink: torch.Tensor,
        recent_start: torch.Tensor,
        kv_len_head: torch.Tensor,
        positions_token: torch.Tensor,
        k_head_cfg: int,
        slice_start: Optional[int] = None,
        slice_end: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Selector 阶段 B：选择 top-k indices，并保留对应 score。"""
        layers, batch_size, num_kv_heads, kv_len_total = log_s_adjusted.shape
        device = log_s_adjusted.device
        if k_head_cfg <= 0:
            empty_idx = torch.full((layers, batch_size, num_kv_heads, 1), -1, dtype=torch.int32, device=device)
            empty_scores = torch.full(
                (layers, batch_size, num_kv_heads, 1),
                float("-inf"),
                dtype=log_s_adjusted.dtype,
                device=device,
            )
            return empty_idx, empty_scores

        use_slice = False
        start = 0
        end = kv_len_total
        if slice_start is not None and slice_end is not None:
            start = max(0, int(slice_start))
            end = min(int(slice_end), int(kv_len_total))
            if end > start:
                use_slice = True

        # 方案 C 优化：使用 sorted=True 替代后处理 sort
        # sorted=True 会自动将有效值降序排列，-inf 值排到尾部，节省约 400us
        if use_slice:
            scores = log_s_adjusted[..., start:end]
            topk_shift = int(start)
        else:
            scores = log_s_adjusted
            topk_shift = 0
        scores_len = int(scores.shape[-1]) if scores.ndim > 0 else 0
        if scores_len <= 0:
            topk_idx = torch.full(
                (layers, batch_size, num_kv_heads, int(k_head_cfg)),
                -1,
                dtype=torch.int32,
                device=device,
            )
            topk_vals = torch.full(
                (layers, batch_size, num_kv_heads, int(k_head_cfg)),
                float("-inf"),
                dtype=log_s_adjusted.dtype,
                device=device,
            )
            return topk_idx, topk_vals
        k_eff = min(int(k_head_cfg), int(scores_len))
        topk_vals, topk_idx = torch.topk(
            scores,
            k_eff,
            dim=-1,
            largest=True,
            sorted=True,  # 方案 C：直接排序
        )
        if topk_shift:
            topk_idx = topk_idx + int(topk_shift)
        if k_eff < int(k_head_cfg):
            pad = torch.full(
                (layers, batch_size, num_kv_heads, int(k_head_cfg) - int(k_eff)),
                -1,
                dtype=topk_idx.dtype,
                device=device,
            )
            topk_idx = torch.cat([topk_idx, pad], dim=-1)
            padv = torch.full_like(pad, float("-inf"), dtype=topk_vals.dtype)
            topk_vals = torch.cat([topk_vals, padv], dim=-1)
        # 方案 E 优化：先转 int32 再 masked_fill（减少 ~50us）
        # 重要：后续 fused rebuild / CUDA writer 仅需 int32 token index
        if topk_idx.dtype != torch.int32:
            topk_idx = topk_idx.to(dtype=torch.int32)
        # 将 -inf 对应的 index 标记为 -1（sorted=True 已保证 -inf 在尾部）
        invalid = ~torch.isfinite(topk_vals)
        topk_idx = topk_idx.masked_fill(invalid, -1)
        # 方案 C：由于 sorted=True 已经把有效值排在前面、-inf 排在尾部，
        # 无需再用 sentinel sort 重排
        valid_counts = (topk_idx >= 0).sum(dim=-1)
        min_valid = valid_counts.min(dim=2).values
        k_index_template = torch.arange(int(k_head_cfg), device=device, dtype=min_valid.dtype)
        keep_mask = (
            k_index_template.view(1, 1, 1, -1)
            < min_valid.unsqueeze(-1).unsqueeze(-1)
        )
        topk_idx = topk_idx.masked_fill(~keep_mask, -1)
        topk_vals = topk_vals.masked_fill(~keep_mask, float("-inf"))
        return topk_idx, topk_vals

    def _compute_alpha_selection_batched(
        self,
        payloads: Sequence[SelectorBatchPayload],
        *,
        require_base: bool = False,
        phase: str = "decode",
    ) -> Optional[SelectorResult]:
        return compute_alpha_selection_batched_impl(
            self,
            payloads,
            require_base=bool(require_base),
            phase=str(phase),
            align_up_int_fn=_align_up_int,
            selector_cpp_stack_cached=bool(_SELECTOR_CPP_STACK_CACHED),
            get_selector_batch_ext_fn=_get_selector_batch_ext,
            rebuild_physical_block_sort_cached=bool(_REBUILD_PHYSICAL_BLOCK_SORT_CACHED),
            selector_kbucket_cached=bool(_SELECTOR_KBUCKET_CACHED),
        )

    def _update_selection_tracking(
        self,
        payloads: Sequence[SelectorBatchPayload],
        result: SelectorResult,
        *,
        phase: str,
        profile_cpu_detail: bool,
        t_post0_ns: Optional[int],
        pending_refresh_rebuild: Optional[object] = None,
    ) -> None:
        need_coverage_metrics = False
        cfg = self.config
        selection_mode = str(getattr(cfg.alpha_fair, "selection_mode", "token_topk")) if cfg is not None else "token_topk"
        if cfg is not None:
            if int(cfg.log_interval) > 0:
                need_coverage_metrics = True

        # ========== C2 优化：Request Tracking 批量更新 - 移出 layer 循环 ==========
        # Request tracking 是 per-request 的，不是 per-layer 的，只需更新一次。
        # 只有 per-layer state 更新需要在 layer 循环内。
        batch_size = len(result.slot_list)
        slot_step = self.step_context_epoch
        refreshed_slots = result.slot_list if result.slot_list else list(range(batch_size))
        semantic_snapshot = self._get_step_semantic_snapshot() if self.config is not None else None
        sink_tokens = int(semantic_snapshot.sink_tokens) if semantic_snapshot is not None else 0
        attn_mode = str(getattr(cfg, "attn_mode", "compact_recent"))
        skip_page_sparse_state = should_skip_page_sparse_state(attn_mode)
        produce_page_sparse_state = not skip_page_sparse_state
        max_refreshed_slot = max(
            (int(slot) for slot in refreshed_slots if int(slot) >= 0),
            default=-1,
        )

        # 预先收集所有 slot 的 req_req_step（用于后续 per-layer state 更新）
        slot_req_steps: Dict[int, int] = {}  # slot -> req_req_step
        first_state = payloads[0].state if payloads else None
        pending_cleared = False

        def _pending_refresh_rebuild_publish_is_final(req_id: str) -> bool:
            if pending_refresh_rebuild is None:
                return True
            target_end = int(getattr(pending_refresh_rebuild, "target_layer_end", -1))
            if target_end < 0:
                return True
            pending_by_req = getattr(self, "_pending_refresh_rebuild_by_req", None)
            if not pending_by_req:
                return True
            max_registered_end = target_end
            for key in pending_by_req:
                if not isinstance(key, tuple) or len(key) < 3:
                    continue
                if str(key[0]) != str(req_id):
                    continue
                try:
                    layer_end = int(key[2])
                except (TypeError, ValueError):
                    continue
                max_registered_end = max(max_registered_end, layer_end)
            return target_end >= max_registered_end

        if phase == "decode" and refreshed_slots and first_state is not None:
            # 一次性更新所有 request tracking（不在 layer 循环里重复）
            for slot in refreshed_slots:
                if slot < 0 or slot >= first_state.batch_size:
                    continue
                if 0 <= slot < len(first_state.batch_request_ids):
                    req_id = first_state.batch_request_ids[slot]
                    tracking = self._ensure_request(req_id)
                    # [TP-DET-TRIGGER 2026-07-07] 决策终局已全部提交点化
                    # (enqueue commit single-writer,决定论):last 推进/trigger
                    # 计时归零/清票/_was_short_dense 翻转不再发生于 publish——
                    # publish 时机依赖 GPU writer 完成(per-rank 异步),曾使
                    # partial/final 分叉把非确定时序注入触发决策 → TP>1 各 rank
                    # 决策发散 → NCCL 集合发散挂死。此处仅保留:
                    #   1) slot_req_steps(per-layer state 的 refresh 步标记);
                    #   2) 读侧终局:全层 ready 且 publish final 时清 scheduled_*
                    #      = off-rail/dense-consume 路由的解除点(读路由允许
                    #      per-rank 时序,近似语义;决策路径不再消费该状态)。
                    planned = int(getattr(tracking, "scheduled_decode_refresh_step", -1))
                    if planned >= 0:
                        req_req_step = planned
                    else:
                        req_req_step = int(tracking.decode_step) if tracking.decode_step is not None else -1
                    slot_req_steps[slot] = req_req_step
                    all_layers_compact_ready = self._request_compact_ready_all_layers(req_id)
                    publish_final_for_req = _pending_refresh_rebuild_publish_is_final(req_id)
                    if all_layers_compact_ready and publish_final_for_req:
                        tracking.scheduled_decode_refresh_step = -1
                        tracking.scheduled_refresh_ctrl_step = -1
                        tracking.inflight_reason_code = -1
                        tracking.inflight_policy = -1
        elif phase != "decode" and first_state is not None:
            # prefill 的 request-facing 提交边界不在 selector tracking。
            # bootstrap_pending / bootstrap_done 统一由 flush 成功后的 finalize boundary
            # 与 wait_decider 负责，避免在 prefill 中途提前暴露 selected-ready。
            pass

        # 轻量更新 per-layer state（coverage/capped + refresh steps）
        for layer_idx, payload in enumerate(payloads):
            state = payload.state
            if max_refreshed_slot >= 0:
                state.ensure_batch(int(max_refreshed_slot) + 1)
            need_block_size = produce_page_sparse_state or need_coverage_metrics
            block_size = 0
            if need_block_size:
                block_size = (
                    int(payload.key_cache.shape[1])
                    if payload.key_cache is not None and payload.key_cache.dim() >= 2
                    else 0
                )
            sink_page_slots = (
                max(0, (int(sink_tokens) + int(block_size) - 1) // int(block_size))
                if produce_page_sparse_state and block_size > 0
                else 0
            )
            if produce_page_sparse_state and block_size > 0:
                selected_layer = result.selected_indices[layer_idx]
                recent_start_layer = result.recent_start[layer_idx]
                for batch_idx, slot in enumerate(refreshed_slots):
                    if slot < 0:
                        continue
                    recent_start_values = recent_start_layer[batch_idx]
                    selected_middle_pages, selected_middle_counts = (
                        project_selected_token_indices_to_logical_pages(
                            selected_token_indices=selected_layer[batch_idx],
                            block_size=int(block_size),
                            sink_page_slots=int(sink_page_slots),
                            recent_start_token=recent_start_values,
                        )
                    )
                    next_refresh_generation = (
                        int(state.sparse_request_refresh_generation[slot]) + 1
                    )
                    state.record_sparse_selected_middle(
                        slot=int(slot),
                        selected_middle_pages=selected_middle_pages,
                        selected_middle_counts=selected_middle_counts,
                        refresh_generation=int(next_refresh_generation),
                    )
            # 更新 coverage/capped 统计（可观测性/自适应使用；默认关闭）
            if need_coverage_metrics:
                try:
                    selected = result.selected_indices[layer_idx]
                    sel_counts = (selected >= 0).sum(dim=-1).to(dtype=torch.float32)
                    head_sink = result.head_sink[layer_idx].to(dtype=torch.float32)
                    recent_start = result.recent_start[layer_idx].to(dtype=torch.float32)
                    kv_len_head = result.kv_len_head[layer_idx].to(dtype=torch.float32).clamp(min=1.0)
                    union_len = head_sink + (kv_len_head - recent_start) + sel_counts
                    coverage = (union_len / kv_len_head).reshape(-1)
                    state.last_coverage = coverage
                    state.last_capped = torch.zeros_like(coverage, dtype=torch.bool)
                except Exception:
                    _log.warning("coverage metrics computation failed for layer_idx=%s", layer_idx, exc_info=True)
                    raise
            if phase == "decode" and refreshed_slots:
                # per-layer state 更新
                state._resize_refresh_steps()
                steps_cpu = getattr(state, "last_refresh_step_per_slot_cpu", None)
                steps_decode_cpu = getattr(state, "last_refresh_decode_per_slot_cpu", None)
                for slot in refreshed_slots:
                    if slot < 0 or slot >= state.batch_size:
                        continue
                    if steps_cpu is not None and slot < len(steps_cpu):
                        steps_cpu[slot] = slot_step
                    req_req_step = slot_req_steps.get(slot)
                    if (
                        steps_decode_cpu is not None
                        and req_req_step is not None
                        and int(req_req_step) >= 0
                        and slot < len(steps_decode_cpu)
                    ):
                        steps_decode_cpu[slot] = int(req_req_step)
                state.last_refresh_step = slot_step
            state.bootstrap_done = self._all_slots_bootstrapped(state)
        if pending_cleared:
            self._bump_refresh_nonce()
        if profile_cpu_detail and t_post0_ns is not None:
            result.profile_cpu_post_us = (time.perf_counter_ns() - t_post0_ns) / 1000.0

    def _apply_alpha_selector_batched_fused(
        self,
        payloads: Sequence[SelectorBatchPayload],
        *,
        phase: str,
        update_tracking: bool = True,
    ) -> Optional[SelectorResult]:
        profile_cpu_detail = bool(getattr(self, "_refresh_profile_active", False)) and bool(
            self._refresh_profile_detail_enabled()
        )
        t_compute0_ns: Optional[int] = None
        if profile_cpu_detail:
            t_compute0_ns = time.perf_counter_ns()
        result = self._compute_alpha_selection_batched(
            payloads,
            require_base=(phase in ("decode", "prefill")),
            phase=phase,
        )
        if profile_cpu_detail and t_compute0_ns is not None and result is not None:
            result.profile_cpu_compute_us = (time.perf_counter_ns() - t_compute0_ns) / 1000.0
        if result is None:
            return None

        # ========== C3 优化：使用缓存的环境变量，避免热路径动态检查 ==========
        t_post0_ns: Optional[int] = time.perf_counter_ns() if profile_cpu_detail else None
        if update_tracking:
            self._update_selection_tracking(
                payloads,
                result,
                phase=phase,
                profile_cpu_detail=profile_cpu_detail,
                t_post0_ns=t_post0_ns,
            )
        elif profile_cpu_detail and t_post0_ns is not None:
            result.profile_cpu_post_us = (time.perf_counter_ns() - t_post0_ns) / 1000.0
        return result
