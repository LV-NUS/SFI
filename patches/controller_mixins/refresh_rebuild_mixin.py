"""
patches/controller_mixins/refresh_rebuild_mixin.py — Pending refresh management and refresh stream lifecycle.

OWNS:
  - _init_refresh_rebuild_state(): refresh/rebuild state initialization
  - _async_refresh_enabled: refresh feature flag
  - _pending_refresh_rebuild_register / _pending_refresh_rebuild_clear(): rebuild tracking
  - _submit_pending_refresh_rebuild / _enqueue_pending_refresh_rebuild(): rebuild submission pipeline
  - _compact_pending_refresh_payloads(): payload compaction for chunked refresh

DEPENDS_ON:
  - patches.sparse_utils._is_stream_capturing_or_raise
  - patches.sparse_constants._is_free_slot_id
  - WaitDeciderMixin._pending_work_mark_submitted / _pending_work_reset
  - CaptureRingMixin._map_global_layer_to_capture_slot
  - Main controller: _update_selection_tracking, _rebuild_compact_slots_batched_layers_from_selection,
    _set_request_pending_refresh, _all_slots_bootstrapped

ENTRY_POINTS:
  - _init_refresh_rebuild_state(): called from VLLMSparseController.__init__
"""
from __future__ import annotations

import logging
import os
import time
from collections import deque
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Deque,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import torch

from patches.request_intent_ticket import PendingPolicy, PendingReasonCode
from patches.refresh_runtime.producer_workspace import (
    build_refresh_producer_work_item,
)
from patches.selector_runtime.selected_out_ring import (
    SLOT_VALID_SET_ATTR,
    SelectedOutRing,
    SlotStableOverrides,
)
from patches.sparse_constants import (
    _ASYNC_REFRESH_CACHED,
    _ASYNC_PRODUCER_GPU_PROFILE_CACHED,
    _CAPTURE_CHUNK,
    _CAPTURE_IN_FLIGHT,
    _DEFERRED_SELECTOR_PROFILE_DETAIL_CACHED,
    _selected_out_ring_enabled,
    _selected_out_ring_slots,
    _PENDING_REBUILD_MAX_QUEUE_CACHED,
    _REFRESH_GROUPED_ASYNC_ENVELOPE_CACHED,
    _REFRESH_PRODUCER_SPLIT_SELECTOR_WRITER_RELEASE_AUTO_CACHED,
    _REFRESH_PRODUCER_SPLIT_SELECTOR_WRITER_RELEASE_CACHED,
    _REFRESH_PRODUCER_SPLIT_SELECTOR_WRITER_RELEASE_MAX_PER_HANDLE_CACHED,
    _REFRESH_PRODUCER_SPLIT_SELECTOR_WRITER_RELEASE_MIN_LAYER_START_CACHED,
    _REFRESH_REBUILD_CHECK_CACHED,
    _REFRESH_REBUILD_MAX_DELAY_STEPS_CACHED,
    _REFRESH_STREAM_PRIORITY_CACHED,
    _is_free_slot_id,
)
from patches.sparse_utils import _is_stream_capturing_or_raise

if TYPE_CHECKING:
    from patches.sparse_types import (
        LayerState,
        PendingRefreshRebuild,
        SelectorBatchPayload,
        SelectorResult,
    )

_log = logging.getLogger(__name__)

# Upper bound on the diagnostic _deadline_rebuild_drain_submit_decode_steps set.
# It is populated with a distinct monotonic decode-step int per drain submit and is
# consumed only by the flush_worker refresh-profile snapshot; without a bound it grows
# one int per step forever (slow host-RAM creep in the deadline-rebuild drain regime).
# Keep only the most recent CAP steps. Host-only (cannot cause GPU OOM); generous so
# unit contracts (which add a handful of steps) are unaffected.
_DEADLINE_DRAIN_SUBMIT_DECODE_STEPS_CAP = 4096


def _selected_scope_target_matches_key(
    target_key: object,
    candidate_key: object,
) -> bool:
    if target_key is None or candidate_key is None:
        return False
    if target_key == candidate_key:
        return True
    target_consumer_step_id = getattr(target_key, "consumer_step_id", None)
    target_layer_group_id = getattr(target_key, "layer_group_id", None)
    candidate_consumer_step_id = getattr(candidate_key, "consumer_step_id", None)
    candidate_layer_group_id = getattr(candidate_key, "layer_group_id", None)
    if (
        target_consumer_step_id is None
        or target_layer_group_id is None
        or candidate_consumer_step_id is None
        or candidate_layer_group_id is None
    ):
        return False
    candidate_chunk_id = int(getattr(candidate_key, "chunk_id", 0))
    return (
        int(target_consumer_step_id) == int(candidate_consumer_step_id)
        and int(target_layer_group_id) == int(candidate_layer_group_id)
        and candidate_chunk_id == 0
    )


def _pending_refresh_rebuild_matches_current_target_scope(
    controller: object,
    pending: "PendingRefreshRebuild",
) -> bool:
    pending_target_selected_scope_key = getattr(
        pending,
        "target_selected_scope_key",
        None,
    )
    if pending_target_selected_scope_key is None:
        return True
    step_authority = getattr(controller, "step_authority", None)
    if step_authority is None:
        return True
    current_target_selected_scope_key = getattr(
        step_authority, "target_selected_scope_key", None
    )
    current_consume_selected_scope_key = getattr(
        step_authority, "consume_selected_scope_key", None
    )
    if (
        current_target_selected_scope_key is None
        and current_consume_selected_scope_key is None
    ):
        return True
    return _selected_scope_target_matches_key(
        pending_target_selected_scope_key,
        current_target_selected_scope_key,
    ) or _selected_scope_target_matches_key(
        pending_target_selected_scope_key,
        current_consume_selected_scope_key,
    )


def _mark_pending_selected_scope_terminal(
    pending: "PendingRefreshRebuild",
    *,
    status: str,
) -> None:
    from patches.fa3_native.scope_async import (
        mark_layer_commit_terminal,
    )

    for payload in getattr(pending, "payloads", ()):
        handle = getattr(payload, "selected_scope_wait_handle", None)
        if handle is None:
            continue
        expected_layers = tuple(getattr(handle, "expected_layers", ()))
        if not expected_layers:
            raise RuntimeError(
                "selected scope wait handle requires expected_layers before pending-rebuild terminal mark"
            )
        layer_index = int(getattr(getattr(payload, "state", None), "layer_index", -1))
        if layer_index < 0:
            layer_index = int(getattr(payload, "layer_index", -1))
        if layer_index < 0:
            raise RuntimeError(
                "selected scope wait handle requires non-negative layer_index before pending-rebuild terminal mark"
            )
        mark_layer_commit_terminal(handle, layer_id=layer_index, status=str(status))


class RefreshRebuildMixin:
    """Pending refresh management and refresh stream lifecycle."""

    _MIXIN_REQUIRES: tuple = ("WaitDeciderMixin", "CaptureRingMixin")

    # ------------------------------------------------------------------
    # State init
    # ------------------------------------------------------------------

    def _init_refresh_rebuild_state(self) -> None:
        """Initialise all state owned by RefreshRebuildMixin."""
        # sentence trigger / nonce
        self._refresh_nonce: int = 0
        # step-level refresh commit tracking（用于“计划 vs 提交”强一致性 fail-fast）
        self._step_refresh_commit_id: int = 0
        self._step_refresh_commit_handle_id: int = -1
        self._step_refresh_commit_handle_generation: int = -1
        self._step_refresh_commit_planned_reqs: int = 0
        self._step_refresh_commit_planned_rows: int = 0
        self._step_refresh_commit_num_actual_tokens: int = 0
        self._step_refresh_commit_payload_enqueues: int = 0
        self._step_refresh_commit_post_kernel_calls: int = 0
        # commit-path single writer（exactly-once per request within one step-handle）
        self._step_refresh_commit_written_handle_id: int = -1
        self._step_refresh_commit_written_handle_generation: int = -1
        self._step_refresh_commit_written_req_ids: Set[str] = set()
        # handle-ledger ring（slot = handle_id % ring_size）：
        # [handle_id, generation, planned_reqs, planned_rows, num_actual_tokens, post_kernel_calls, payload_enqueues]
        self._step_refresh_handle_ledger: List[List[int]] = []
        self._step_refresh_handle_ledger_size: int = 0
        # pending refresh rebuilds
        self._pending_refresh_rebuilds: Deque[PendingRefreshRebuild] = deque()
        self._pending_refresh_rebuild_id: int = 0
        self._pending_refresh_rebuild_by_req: Dict[Tuple[str, int, int], int] = {}
        self._refresh_rebuild_delay_max: int = 0
        self._refresh_rebuild_delay_max_epoch: int = -1
        self._deadline_rebuild_drop_finished_count: int = 0
        self._deadline_rebuild_drain_finish_count: int = 0
        self._deadline_rebuild_partial_finish_count: int = 0
        self._deadline_rebuild_drain_submit_count: int = 0
        self._deadline_rebuild_drain_submit_decode_step_min: int = -1
        self._deadline_rebuild_drain_submit_decode_step_max: int = -1
        self._deadline_rebuild_drain_submit_decode_steps: Set[int] = set()
        self._deadline_producer_work_target_layer_start: int = -1
        self._deadline_producer_work_target_layer_end: int = -1
        self._deadline_producer_work_decode_step_min: int = -1
        self._deadline_producer_work_decode_step_max: int = -1
        self._deadline_producer_work_ready_epoch: int = -1
        self._deadline_producer_work_deadline_epoch: int = -1
        self._deadline_producer_work_deadline_handle_id: int = -1
        self._deadline_producer_work_deadline_slack_steps: int = -1
        self._deadline_producer_work_can_drop: int = 0
        self._deadline_producer_work_can_coalesce: int = 0
        self._deadline_producer_work_admission_reason: str = ""
        self._deadline_rebuild_pre_consume_drain_count: int = 0
        self._deadline_rebuild_pre_consume_drop_stale_count: int = 0
        self._deadline_deferred_selector_compute_count: int = 0
        self._deadline_deferred_selector_compute_cpu_us_total: float = 0.0
        self._deadline_deferred_selector_compute_cpu_us_max: float = 0.0
        self._deadline_deferred_selector_inner_compute_cpu_us_total: float = 0.0
        self._deadline_deferred_selector_inner_compute_cpu_us_max: float = 0.0
        self._deadline_deferred_selector_stack_cpu_us_total: float = 0.0
        self._deadline_deferred_selector_stack_cpu_us_max: float = 0.0
        self._deadline_deferred_selector_validate_cpu_us_total: float = 0.0
        self._deadline_deferred_selector_validate_cpu_us_max: float = 0.0
        self._deadline_deferred_selector_key_norms_cpu_us_total: float = 0.0
        self._deadline_deferred_selector_key_norms_cpu_us_max: float = 0.0
        self._deadline_deferred_selector_key_norms_arena_cpu_us_total: float = 0.0
        self._deadline_deferred_selector_key_norms_arena_cpu_us_max: float = 0.0
        self._deadline_deferred_selector_key_norms_direct_cpu_us_total: float = 0.0
        self._deadline_deferred_selector_key_norms_direct_cpu_us_max: float = 0.0
        self._deadline_deferred_selector_key_norms_direct_prepare_cpu_us_total: float = 0.0
        self._deadline_deferred_selector_key_norms_direct_prepare_cpu_us_max: float = 0.0
        self._deadline_deferred_selector_key_norms_direct_launch_cpu_us_total: float = 0.0
        self._deadline_deferred_selector_key_norms_direct_launch_cpu_us_max: float = 0.0
        self._deadline_deferred_selector_key_norms_pack_cpu_us_total: float = 0.0
        self._deadline_deferred_selector_key_norms_pack_cpu_us_max: float = 0.0
        self._deadline_deferred_selector_select_cpu_us_total: float = 0.0
        self._deadline_deferred_selector_select_cpu_us_max: float = 0.0
        self._deadline_deferred_selector_post_cpu_us_total: float = 0.0
        self._deadline_deferred_selector_post_cpu_us_max: float = 0.0
        self._deadline_deferred_selector_wrapper_gap_cpu_us_total: float = 0.0
        self._deadline_deferred_selector_wrapper_gap_cpu_us_max: float = 0.0
        self._deadline_async_producer_body_count: int = 0
        self._deadline_async_producer_body_cpu_us_total: float = 0.0
        self._deadline_async_producer_body_cpu_us_max: float = 0.0
        self._deadline_async_producer_selector_count: int = 0
        self._deadline_async_producer_selector_cpu_us_total: float = 0.0
        self._deadline_async_producer_selector_cpu_us_max: float = 0.0
        self._deadline_async_producer_key_norms_delta_count: int = 0
        self._deadline_async_producer_key_norms_delta_total_tokens_total: int = 0
        self._deadline_async_producer_key_norms_delta_max_tokens_max: int = -1
        self._deadline_async_producer_key_norms_delta_layers_total: int = 0
        self._deadline_async_producer_writer_count: int = 0
        self._deadline_async_producer_writer_cpu_us_total: float = 0.0
        self._deadline_async_producer_writer_cpu_us_max: float = 0.0
        self._deadline_async_producer_graph_replay_count: int = 0
        self._deadline_async_producer_graph_replay_cpu_us_total: float = 0.0
        self._deadline_async_producer_graph_replay_cpu_us_max: float = 0.0
        self._deadline_async_producer_graph_replay_stage_selector_inputs_count: int = 0
        self._deadline_async_producer_graph_replay_stage_selector_inputs_cpu_us_total: float = 0.0
        self._deadline_async_producer_graph_replay_stage_selector_inputs_cpu_us_max: float = 0.0
        self._deadline_async_producer_graph_replay_prepare_writer_count: int = 0
        self._deadline_async_producer_graph_replay_prepare_writer_cpu_us_total: float = 0.0
        self._deadline_async_producer_graph_replay_prepare_writer_cpu_us_max: float = 0.0
        self._deadline_async_producer_graph_replay_stage_lens_count: int = 0
        self._deadline_async_producer_graph_replay_stage_lens_cpu_us_total: float = 0.0
        self._deadline_async_producer_graph_replay_stage_lens_cpu_us_max: float = 0.0
        self._deadline_async_producer_graph_replay_prepare_events_count: int = 0
        self._deadline_async_producer_graph_replay_prepare_events_cpu_us_total: float = 0.0
        self._deadline_async_producer_graph_replay_prepare_events_cpu_us_max: float = 0.0
        self._deadline_async_producer_graph_replay_graph_count: int = 0
        self._deadline_async_producer_graph_replay_graph_cpu_us_total: float = 0.0
        self._deadline_async_producer_graph_replay_graph_cpu_us_max: float = 0.0
        self._deadline_async_producer_graph_capture_count: int = 0
        self._deadline_async_producer_graph_capture_cpu_us_total: float = 0.0
        self._deadline_async_producer_graph_capture_cpu_us_max: float = 0.0
        # ASYNC_PRODUCER_WRITER_GRAPH (task #9): captured writer-graph holder.
        # ``graphs`` maps key -> CUDAGraph (each graph owns a PRIVATE mempool,
        # [POOL-PRIVATE] — never the decode-graph pool, never shared);
        # ``bypass`` is the fail-open / thrash latch.
        self._writer_graph_state: Optional[Dict[str, object]] = None
        self._writer_graph_recapture_window: int = 0
        self._writer_graph_recapture_count: int = 0
        self._writer_graph_evict_clear_count: int = 0
        self._deadline_async_producer_result_precomputed_count: int = 0
        self._deadline_async_producer_split_release_forced_count: int = 0
        self._deadline_async_producer_split_release_adaptive_count: int = 0
        self._deadline_async_producer_split_release_adaptive_gated_count: int = 0
        self._deadline_async_producer_split_release_adaptive_layer_gated_count: int = 0
        self._pending_refresh_grouped_async_records: Optional[
            List[Dict[str, Any]]
        ] = None
        self._pending_refresh_empty_tensor_cache: Dict[Tuple[str, str], torch.Tensor] = {}
        self._refresh_producer_executor: Any = None
        self._refresh_producer_release_condition: Any = None
        self._refresh_producer_released_handle_id: int = -1
        self._refresh_producer_stream_release_events: List[Tuple[int, Any, Any]] = []
        self._refresh_producer_stream_release_pending: bool = False
        self._refresh_producer_stream_release_generation: int = 0
        self._refresh_producer_stream_release_checked_generation: int = -1
        self._refresh_producer_stream_release_checked_handle_id: int = -1
        self._refresh_producer_split_release_counts_by_handle: Dict[int, int] = {}

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    def _async_refresh_enabled(self) -> bool:
        return bool(_ASYNC_REFRESH_CACHED)

    def _refresh_rebuild_check_enabled(self) -> bool:
        return bool(_REFRESH_REBUILD_CHECK_CACHED)

    def _refresh_rebuild_max_delay_steps(self, num_chunks: int) -> int:
        value = int(_REFRESH_REBUILD_MAX_DELAY_STEPS_CACHED)
        if value <= 0:
            chunks = max(1, int(num_chunks))
            if chunks == 1:
                value = 1
            else:
                # Give async producer work enough runway to finish before the
                # consumer step, but cap it so refresh freshness does not drift
                # with model depth. The stagger scheduler uses the same
                # four-step envelope for the 28-layer / two-chunk GT1 path.
                value = max(2, min(4, chunks * 2))
        return max(1, value)

    # ------------------------------------------------------------------
    # Pending rebuild management
    # ------------------------------------------------------------------

    @staticmethod
    def _pending_refresh_rebuild_req_key(
        req_id: str,
        *,
        target_layer_start: int,
        target_layer_end: int,
    ) -> Tuple[str, int, int]:
        return (
            str(req_id),
            int(target_layer_start),
            int(target_layer_end),
        )

    def _pending_refresh_rebuild_ids_for_req(self, req_id: str) -> Tuple[int, ...]:
        pending_by_req = getattr(self, "_pending_refresh_rebuild_by_req", None)
        if not pending_by_req:
            return ()
        rid = str(req_id)
        ids: Set[int] = set()
        for key, value in pending_by_req.items():
            if not isinstance(key, tuple) or not key:
                continue
            if str(key[0]) != rid:
                continue
            try:
                pending_id = int(value)
            except (TypeError, ValueError):
                pending_id = -1
            if pending_id >= 0:
                ids.add(pending_id)
        return tuple(sorted(ids))

    def _pending_refresh_rebuild_has_req(self, req_id: str) -> bool:
        return bool(
            RefreshRebuildMixin._pending_refresh_rebuild_ids_for_req(self, req_id)
        )

    def _pending_refresh_rebuild_register(
        self,
        req_ids: Optional[Tuple[str, ...]],
        *,
        target_layer_start: int = -1,
        target_layer_end: int = -1,
    ) -> Tuple[int, Optional[Tuple[str, ...]]]:
        if not req_ids:
            return -1, req_ids
        self._pending_refresh_rebuild_id += 1
        pending_id = self._pending_refresh_rebuild_id
        for rid in req_ids:
            if not rid or _is_free_slot_id(rid):
                continue
            key = RefreshRebuildMixin._pending_refresh_rebuild_req_key(
                rid,
                target_layer_start=int(target_layer_start),
                target_layer_end=int(target_layer_end),
            )
            self._pending_refresh_rebuild_by_req[key] = pending_id
        return pending_id, req_ids

    def _pending_refresh_rebuild_req_mapping_snapshot(
        self,
        req_ids: Optional[Tuple[str, ...]],
        *,
        target_layer_start: int,
        target_layer_end: int,
    ) -> Dict[Tuple[str, int, int], Optional[int]]:
        snapshot: Dict[Tuple[str, int, int], Optional[int]] = {}
        if not req_ids:
            return snapshot
        for rid in req_ids:
            if not rid or _is_free_slot_id(rid):
                continue
            key = RefreshRebuildMixin._pending_refresh_rebuild_req_key(
                rid,
                target_layer_start=int(target_layer_start),
                target_layer_end=int(target_layer_end),
            )
            snapshot[key] = self._pending_refresh_rebuild_by_req.get(key)
        return snapshot

    def _pending_refresh_rebuild_restore_req_mapping(
        self,
        snapshot: Dict[Tuple[str, int, int], Optional[int]],
        *,
        current_pending_id: Optional[int] = None,
    ) -> None:
        for key, pending_id in snapshot.items():
            if current_pending_id is not None:
                current_id = self._pending_refresh_rebuild_by_req.get(key)
                if current_id is not None and int(current_id) != int(current_pending_id):
                    continue
            if pending_id is None:
                self._pending_refresh_rebuild_by_req.pop(key, None)
            else:
                self._pending_refresh_rebuild_by_req[key] = int(pending_id)

    def _pending_refresh_rebuild_is_latest(self, pending: PendingRefreshRebuild) -> bool:
        req_ids = pending.req_ids
        if not req_ids or pending.pending_id < 0:
            return True
        pid = pending.pending_id
        finished_req_ids = self._finished_req_ids_step
        request_states = self.request_states
        # [ROW-REMAP-PENDING-RETIRE 2026-07-07] 行迁移对账载体:批内任何 req
        # 完成都会让 vLLM condense 迁移"存活" req 的 block_table 行,pending
        # 世代的 row_list 快照随之过期(实证:PRECHECK b=1 快照行 7 整行 -1,
        # 该 req 已被迁走;finished 检查不命中因为毒票 req 全部存活)。对账
        # =快照行 vs 每步冻结的权威映射 _worker_batch_id_to_idx;不一致即
        # 作废整票(现成 drop_non_latest 终局路径清理),下次触发按新行重建。
        _row_map = getattr(self, "_worker_batch_id_to_idx", None)
        _payloads = getattr(pending, "payloads", None) or ()
        _row_snapshot = (
            tuple(getattr(_payloads[0], "row_list", ()) or ()) if _payloads else ()
        )
        for _ri, rid in enumerate(req_ids):
            if not rid or _is_free_slot_id(rid):
                continue
            if rid in finished_req_ids or rid not in request_states:
                # [FINISHED-REQ-PENDING-RETIRE 2026-07-07] 4B illegal 真根因
                # 终局修:req 完成/退场 = 其 block_table 行即刻易主(vLLM 行
                # 迁移/复用),pending 世代的 row_list/seq_lens 快照随之过期。
                # 此前这两类 req 被"豁免检查"(continue),含毒票照常
                # due-submit,writer gather 按旧行号读到易主/未填充行(实证:
                # PRECHECK b=1 row=7 整行 -1 → block_id 负寻址 illegal;决定论
                # step≈EOS 步)。终局语义=完成即作废整票(与"提交即终局"合同
                # 一致),调用方现成 drop_non_latest 路径负责清理;存活 req 的
                # freshness 由下次触发重建。零新路径零兜底。
                return False
            if isinstance(_row_map, dict):
                _cur_row = _row_map.get(str(rid))
                if _cur_row is None:
                    # 不在当前批映射 = 已退场/迁出,票作废。
                    return False
                if _ri < len(_row_snapshot) and int(_row_snapshot[_ri]) != int(
                    _cur_row
                ):
                    # 行迁移:快照行 != 当前权威行,票作废(ROW-REMAP)。
                    return False
            key = RefreshRebuildMixin._pending_refresh_rebuild_req_key(
                rid,
                target_layer_start=int(getattr(pending, "target_layer_start", -1)),
                target_layer_end=int(getattr(pending, "target_layer_end", -1)),
            )
            current_pid = self._pending_refresh_rebuild_by_req.get(key)
            if current_pid != pid:
                return False
        return True

    def _pending_refresh_grouped_async_forget_pending(
        self,
        pending: PendingRefreshRebuild,
    ) -> None:
        records = self._pending_refresh_grouped_async_records
        if records is None or not records:
            return
        original_records = tuple(records)
        removed_records = tuple(
            record
            for record in original_records
            if record["pending"] is pending
        )
        if not removed_records:
            return
        records[:] = [
            record
            for record in original_records
            if record["pending"] is not pending
        ]
        for record in reversed(removed_records):
            cleanup = record.get("cleanup")
            if callable(cleanup):
                cleanup()
        RefreshRebuildMixin._restore_pending_refresh_grouped_async_buf_state(
            self,
            original_records=original_records,
            removed_records=removed_records,
        )

    def _restore_pending_refresh_grouped_async_buf_state(
        self,
        *,
        original_records: Sequence[Dict[str, Any]],
        removed_records: Sequence[Dict[str, Any]],
    ) -> None:
        if not removed_records:
            return
        flags = self._buf_pending_work_flags
        epochs = self._buf_pending_work_epoch
        removed_ids = {id(record) for record in removed_records}
        affected_bufs = {
            int(buf) % len(flags)
            for record in removed_records
            for buf in tuple(record.get("buf_ids", tuple()) or tuple())
        }
        for buf in affected_bufs:
            baseline = (0, -1)
            for record in original_records:
                prior_state = record.get("prior_buf_state", {})
                if int(buf) in prior_state:
                    baseline = prior_state[int(buf)]
                    break
            final_flags, final_epoch = int(baseline[0]), int(baseline[1])
            for record in original_records:
                if id(record) in removed_ids:
                    continue
                record_bufs = {
                    int(raw_buf) % len(flags)
                    for raw_buf in tuple(record.get("buf_ids", tuple()) or tuple())
                }
                if int(buf) not in record_bufs:
                    continue
                final_flags |= 2
                final_epoch = int(record.get("submitted_epoch", final_epoch))
            flags[int(buf)] = int(final_flags)
            epochs[int(buf)] = int(final_epoch)

    def _selected_out_ring_for_pending(self) -> Optional[SelectedOutRing]:
        """[SELECTED-OUT-RING] 懒构造的 pending 路径稳定环(env 关=None)。"""
        if not _selected_out_ring_enabled():
            return None
        ring = getattr(self, "_selected_out_ring", None)
        if ring is None:
            ring = SelectedOutRing(slots=_selected_out_ring_slots())
            self._selected_out_ring = ring
        return ring

    def _selected_out_ring_release_pending_slot(
        self, pending: PendingRefreshRebuild
    ) -> None:
        """[SELECTED-OUT-RING] 释放 pending 占用的环槽(幂等:属性置 None 防
        重入双释放)。writer_done_event 作为消费序存入槽,下任 acquire 在其
        run 流上补 wait。调用面=终局漏斗 _pending_refresh_rebuild_clear+两个
        绕过 clear 的整批丢弃点(release_idle_buffers / reset_for_new_engine)。"""
        _ring_slot = getattr(pending, "selected_out_ring_slot", None)
        if _ring_slot is None:
            return
        pending.selected_out_ring_slot = None
        _ring = getattr(self, "_selected_out_ring", None)
        if _ring is not None:
            _ring.release(
                _ring_slot,
                writer_done_event=getattr(pending, "writer_done_event", None),
            )

    def _pending_refresh_rebuild_clear(self, pending: PendingRefreshRebuild) -> None:
        # [PENDING-FUNNEL-DEBUG 2026-07-09 临时取证探针,破案后拆] serve 写而不读案:
        # 谁在一步内清掉 pending(wrapper drain 计数恒 0 而队列消失)。默认关零成本。
        _funnel_dbg = os.environ.get("VLLM_SPARSE_PENDING_FUNNEL_DEBUG_LOG", "")
        if _funnel_dbg:
            import sys as _fd_sys
            try:
                _fd_caller = _fd_sys._getframe(1).f_code.co_name
                _fd_caller2 = _fd_sys._getframe(2).f_code.co_name
            except Exception:
                _fd_caller = "?"
                _fd_caller2 = "?"
            try:
                with open(_funnel_dbg, "a") as _fd_fh:
                    _fd_fh.write(
                        f"clear\tpid={int(getattr(pending, 'pending_id', -1))}\t"
                        f"reqs={list(getattr(pending, 'req_ids', ()) or ())}\t"
                        f"caller={_fd_caller}<-{_fd_caller2}\t"
                        f"writer_evt={getattr(pending, 'writer_done_event', None) is not None}\t"
                        f"epoch={int(getattr(self, 'step_context_epoch', -1))}\n"
                    )
            except OSError:
                pass
        # [SELECTED-OUT-RING] 终局唯一漏斗释放;必须先于下方 req_ids 早退分支。
        self._selected_out_ring_release_pending_slot(pending)
        RefreshRebuildMixin._pending_refresh_grouped_async_forget_pending(
            self, pending
        )
        self._pending_refresh_rebuild_forget_writer_release(pending)
        pending.selector_scratch_refs = tuple()
        pending.compact_meta_defer_publish = False
        pending.compact_meta_commit_log = None
        req_ids = pending.req_ids
        if not req_ids or pending.pending_id < 0:
            return
        pid = pending.pending_id
        for rid in req_ids:
            if not rid or _is_free_slot_id(rid):
                continue
            key = RefreshRebuildMixin._pending_refresh_rebuild_req_key(
                rid,
                target_layer_start=int(getattr(pending, "target_layer_start", -1)),
                target_layer_end=int(getattr(pending, "target_layer_end", -1)),
            )
            if self._pending_refresh_rebuild_by_req.get(key) == pid:
                del self._pending_refresh_rebuild_by_req[key]

    def _pending_refresh_rebuild_forget_writer_release(
        self,
        pending: PendingRefreshRebuild,
    ) -> None:
        release_after_handle_id = int(
            getattr(pending, "writer_release_after_handle_id", -1) or -1
        )
        if release_after_handle_id <= 0:
            return
        self._forget_refresh_producer_writer_release(
            release_after_handle_id=release_after_handle_id,
        )
        pending.writer_release_after_handle_id = -1

    def _pending_refresh_rebuild_drop_req_ids(
        self,
        pending: PendingRefreshRebuild,
        req_ids: Optional[Tuple[str, ...]] = None,
    ) -> Tuple[str, ...]:
        normalized = self._normalize_refresh_req_ids(
            pending.req_ids if req_ids is None else req_ids
        )
        if not normalized:
            return tuple()
        raw_pending_id = getattr(pending, "pending_id", -1)
        raw_layer_start = getattr(pending, "target_layer_start", -1)
        raw_layer_end = getattr(pending, "target_layer_end", -1)
        pending_id = int(-1 if raw_pending_id is None else raw_pending_id)
        target_layer_start = int(-1 if raw_layer_start is None else raw_layer_start)
        target_layer_end = int(-1 if raw_layer_end is None else raw_layer_end)
        resolved: List[str] = []
        for rid in normalized:
            if not rid or _is_free_slot_id(rid):
                continue
            key = RefreshRebuildMixin._pending_refresh_rebuild_req_key(
                rid,
                target_layer_start=target_layer_start,
                target_layer_end=target_layer_end,
            )
            current_id = self._pending_refresh_rebuild_by_req.get(key)
            if current_id is not None and int(current_id) != pending_id:
                continue
            resolved.append(rid)
        return tuple(resolved)

    def _pending_refresh_rebuild_buf_ref_counts(
        self,
        pending_items: Sequence[PendingRefreshRebuild],
    ) -> Dict[int, int]:
        flags = getattr(self, "_buf_pending_work_flags", ())
        if not flags:
            return {}
        counts: Dict[int, int] = {}
        for item in pending_items:
            for raw_buf in self._pending_refresh_rebuild_buf_ids(item):
                buf = int(raw_buf) % len(flags)
                counts[buf] = int(counts.get(buf, 0)) + 1
        return counts

    def _pending_refresh_rebuild_has_recorded_async_work(
        self,
        pending: PendingRefreshRebuild,
    ) -> bool:
        return getattr(pending, "writer_done_event", None) is not None or bool(
            getattr(pending, "selector_done_event_recorded", False)
        )

    def _pending_refresh_rebuild_clear_completed_buf_work(
        self,
        pending: PendingRefreshRebuild,
        *,
        remaining_pending: Optional[Sequence[PendingRefreshRebuild]] = None,
        pending_buf_ref_counts: Optional[Dict[int, int]] = None,
    ) -> None:
        flags = getattr(self, "_buf_pending_work_flags", ())
        if not flags:
            return
        epochs = getattr(self, "_buf_pending_work_epoch", ())
        pending_buf_ids = tuple(self._pending_refresh_rebuild_buf_ids(pending))
        if not pending_buf_ids:
            return
        if pending_buf_ref_counts is not None:
            for raw_buf in pending_buf_ids:
                buf = int(raw_buf) % len(flags)
                next_count = int(pending_buf_ref_counts.get(buf, 0)) - 1
                if next_count > 0:
                    pending_buf_ref_counts[buf] = next_count
                    continue
                pending_buf_ref_counts.pop(buf, None)
                current_flags = int(flags[buf])
                if (current_flags & 2) == 0:
                    continue
                next_flags = current_flags & ~2
                flags[buf] = int(next_flags)
                if next_flags == 0 and 0 <= buf < len(epochs):
                    epochs[buf] = -1
            return
        other_pending = (
            tuple(self._pending_refresh_rebuilds)
            if remaining_pending is None
            else tuple(remaining_pending)
        )
        for raw_buf in pending_buf_ids:
            buf = int(raw_buf) % len(flags)
            current_flags = int(flags[buf])
            if (current_flags & 2) == 0:
                continue
            covered = False
            for other in other_pending:
                if other is pending:
                    continue
                for other_raw_buf in self._pending_refresh_rebuild_buf_ids(other):
                    if int(other_raw_buf) % len(flags) == buf:
                        covered = True
                        break
                if covered:
                    break
            if not covered:
                next_flags = current_flags & ~2
                flags[buf] = int(next_flags)
                if next_flags == 0 and 0 <= buf < len(epochs):
                    epochs[buf] = -1

    def _wait_pending_refresh_rebuild_writer_done(
        self,
        pending: PendingRefreshRebuild,
    ) -> bool:
        writer_event = getattr(pending, "writer_done_event", None)
        if writer_event is None:
            return False
        payloads = tuple(getattr(pending, "payloads", tuple()) or tuple())
        device = None
        if payloads:
            device = getattr(getattr(payloads[0], "key_cache", None), "device", None)
        if device is None:
            device = getattr(self, "device", None)
        if device is None:
            device = torch.device("cuda")
        cur_stream = torch.cuda.current_stream(device=device)
        cur_stream.wait_event(writer_event)
        return True

    def _wait_pending_refresh_rebuild_selector_done(
        self,
        pending: PendingRefreshRebuild,
    ) -> bool:
        if not bool(getattr(pending, "selector_done_event_recorded", False)):
            return False
        selector_event = getattr(pending, "selector_done_event", None)
        if selector_event is None:
            return False
        payloads = tuple(getattr(pending, "payloads", tuple()) or tuple())
        device = None
        if payloads:
            device = getattr(getattr(payloads[0], "key_cache", None), "device", None)
        if device is None:
            device = getattr(self, "device", None)
        if device is None:
            device = torch.device("cuda")
        cur_stream = torch.cuda.current_stream(device=device)
        cur_stream.wait_event(selector_event)
        return True

    def _drop_pending_refresh_rebuild(
        self,
        pending: PendingRefreshRebuild,
        *,
        status: str,
        lease_reason: str,
        req_ids: Optional[Tuple[str, ...]] = None,
        remaining_pending: Optional[Sequence[PendingRefreshRebuild]] = None,
        pending_buf_ref_counts: Optional[Dict[int, int]] = None,
        wait_recorded_work: bool = True,
    ) -> Tuple[str, ...]:
        # [PENDING-FUNNEL-DEBUG 临时取证探针,破案后拆]
        _funnel_dbg = os.environ.get("VLLM_SPARSE_PENDING_FUNNEL_DEBUG_LOG", "")
        if _funnel_dbg:
            try:
                with open(_funnel_dbg, "a") as _fd_fh:
                    _fd_fh.write(
                        f"drop\tpid={int(getattr(pending, 'pending_id', -1))}\t"
                        f"reqs={list(getattr(pending, 'req_ids', ()) or ())}\t"
                        f"status={status}\tlease={lease_reason}\t"
                        f"epoch={int(getattr(self, 'step_context_epoch', -1))}\n"
                    )
            except OSError:
                pass
        _mark_pending_selected_scope_terminal(
            pending,
            status=str(status),
        )
        if wait_recorded_work:
            self._wait_pending_refresh_rebuild_writer_done(
                pending
            ) or self._wait_pending_refresh_rebuild_selector_done(pending)
        self._pending_refresh_rebuild_clear_completed_buf_work(
            pending,
            remaining_pending=remaining_pending,
            pending_buf_ref_counts=pending_buf_ref_counts,
        )
        drop_req_ids = self._pending_refresh_rebuild_drop_req_ids(
            pending,
            req_ids=req_ids,
        )
        if drop_req_ids:
            self._resolve_refresh_lease(
                req_ids=drop_req_ids,
                reason=str(lease_reason),
            )
        self._pending_refresh_rebuild_clear(pending)
        return drop_req_ids

    def _pending_refresh_rebuild_remove_from_queue(
        self,
        pending: PendingRefreshRebuild,
    ) -> bool:
        if not self._pending_refresh_rebuilds:
            return False
        kept: Deque[PendingRefreshRebuild] = deque()
        removed = False
        while self._pending_refresh_rebuilds:
            item = self._pending_refresh_rebuilds.popleft()
            if item is pending:
                removed = True
                continue
            kept.append(item)
        self._pending_refresh_rebuilds = kept
        return removed

    def _record_pending_refresh_rebuild_stream_lifetime(
        self,
        pending: "PendingRefreshRebuild",
        stream: object,
        *,
        payload_tensor_scope: str = "all",
        include_result: bool = True,
    ) -> int:
        """Bind async producer inputs/results to ``stream`` before refs are trimmed."""
        if stream is None:
            return 0
        seen_ptrs: Set[int] = set()
        recorded = 0

        def _record(tensor: object) -> None:
            nonlocal recorded
            if not isinstance(tensor, torch.Tensor) or not tensor.is_cuda:
                return
            try:
                ptr = int(tensor.untyped_storage().data_ptr())
            except Exception:
                ptr = int(tensor.data_ptr())
            if ptr == 0 or ptr in seen_ptrs:
                return
            tensor.record_stream(stream)
            seen_ptrs.add(ptr)
            recorded += 1

        def _payload_tensors(payload: object) -> Tuple[object, ...]:
            trimmed_tensors = (
                getattr(getattr(payload, "capture_scores", None), "_base", None),
                getattr(payload, "capture_scores", None),
                getattr(getattr(payload, "lastn1_capture_scores", None), "_base", None),
                getattr(payload, "lastn1_capture_scores", None),
                getattr(getattr(payload, "log_f_denoms", None), "_base", None),
                getattr(payload, "log_f_denoms", None),
                getattr(payload, "seq_lens_batch", None),
                getattr(payload, "seq_lens_batch_i32", None),
                getattr(payload, "kv_lengths", None),
                getattr(payload, "kv_len_per_row_i32", None),
                getattr(payload, "q", None),
                getattr(payload, "cu_seqlens_q", None),
                getattr(payload, "refresh_rows_long", None),
                getattr(payload, "refresh_block_table_sub", None),
                getattr(payload, "refresh_seq_lens_i32", None),
            )
            scope = str(payload_tensor_scope or "all")
            if scope == "none":
                return tuple()
            if scope == "trimmed":
                return trimmed_tensors
            return (
                *trimmed_tensors,
                getattr(payload, "row_tensor_i32", None),
                getattr(payload, "row_tensor", None),
                getattr(payload, "slot_tensor", None),
                getattr(payload, "slot_tensor_i32", None),
                getattr(payload, "alibi_slopes", None),
                getattr(payload, "k_descale", None),
                getattr(payload, "block_table", None),
            )

        for payload in getattr(pending, "payloads", ()) or ():
            for tensor in _payload_tensors(payload):
                _record(tensor)

        result = getattr(pending, "result", None)
        if include_result and result is not None:
            for tensor in (
                getattr(result, "selected_indices", None),
                getattr(result, "head_sink", None),
                getattr(result, "recent_start", None),
                getattr(result, "kv_len_head", None),
                getattr(result, "allowed_lengths", None),
                getattr(result, "selected_middle_pages", None),
                getattr(result, "selected_middle_counts", None),
                getattr(result, "selected_token_scores", None),
            ):
                _record(tensor)
        return recorded

    def _retain_pending_refresh_rebuild_selector_scratch(
        self,
        pending: "PendingRefreshRebuild",
        stream: object,
        *containers: object,
    ) -> int:
        """Retain private async selector scratch maps until pending drain."""
        if stream is None:
            pending.selector_scratch_refs = tuple()
            return 0
        refs: List[torch.Tensor] = []
        seen_ptrs: Set[int] = set()

        def _record_and_retain(tensor: torch.Tensor) -> None:
            try:
                ptr = int(tensor.untyped_storage().data_ptr())
            except Exception:
                ptr = int(tensor.data_ptr())
            key = ptr if ptr != 0 else int(id(tensor))
            if key in seen_ptrs:
                return
            seen_ptrs.add(key)
            refs.append(tensor)
            if not tensor.is_cuda:
                return
            tensor.record_stream(stream)

        def _visit(value: object) -> None:
            if isinstance(value, torch.Tensor):
                _record_and_retain(value)
                return
            if isinstance(value, dict):
                for item in value.values():
                    _visit(item)
                return
            if isinstance(value, (list, tuple, set)):
                for item in value:
                    _visit(item)

        for container in containers:
            _visit(container)
        pending.selector_scratch_refs = tuple(refs)
        return len(refs)

    def _pending_refresh_rebuild_decode_steps(
        self,
        pending: "PendingRefreshRebuild",
    ) -> Tuple[int, ...]:
        req_ids = self._normalize_refresh_req_ids(getattr(pending, "req_ids", None))
        if not req_ids:
            return ()
        cached_decode_steps = getattr(self, "_current_decode_step_by_req", None)
        raw_cached_epoch = getattr(self, "_current_decode_step_by_req_epoch", -1)
        raw_current_epoch = getattr(self, "step_context_epoch", -1)
        cached_epoch = int(-1 if raw_cached_epoch is None else raw_cached_epoch)
        current_epoch = int(-1 if raw_current_epoch is None else raw_current_epoch)
        if isinstance(cached_decode_steps, dict) and cached_epoch == current_epoch:
            steps: Set[int] = set()
            for rid in req_ids:
                if not rid or _is_free_slot_id(rid):
                    continue
                raw_decode_step = cached_decode_steps.get(rid, -1)
                decode_step = int(
                    -1 if raw_decode_step is None else raw_decode_step
                )
                if decode_step >= 0:
                    steps.add(decode_step)
            if steps:
                return tuple(sorted(steps))
        request_states = getattr(self, "request_states", None)
        if not isinstance(request_states, dict):
            return ()
        steps: Set[int] = set()
        for rid in req_ids:
            if not rid or _is_free_slot_id(rid):
                continue
            tracking = request_states.get(rid)
            decode_step = int(getattr(tracking, "decode_step", -1))
            if decode_step >= 0:
                steps.add(decode_step)
        return tuple(sorted(steps))

    def _record_deadline_producer_work_item(self, producer_work_item: object) -> None:
        def _int_attr(name: str, default: int = -1) -> int:
            value = getattr(producer_work_item, name, default)
            return int(default if value is None else value)

        target_start = _int_attr("target_layer_start")
        target_end = _int_attr("target_layer_end")
        if target_start >= 0:
            cur = int(getattr(self, "_deadline_producer_work_target_layer_start", -1))
            self._deadline_producer_work_target_layer_start = (
                target_start if cur < 0 else min(cur, target_start)
            )
        if target_end >= 0:
            cur = int(getattr(self, "_deadline_producer_work_target_layer_end", -1))
            self._deadline_producer_work_target_layer_end = max(cur, target_end)

        step_min = _int_attr("decode_step_min")
        step_max = _int_attr("decode_step_max")
        if step_min >= 0:
            cur = int(getattr(self, "_deadline_producer_work_decode_step_min", -1))
            self._deadline_producer_work_decode_step_min = (
                step_min if cur < 0 else min(cur, step_min)
            )
        if step_max >= 0:
            cur = int(getattr(self, "_deadline_producer_work_decode_step_max", -1))
            self._deadline_producer_work_decode_step_max = max(cur, step_max)

        for name in (
            "ready_epoch",
            "deadline_epoch",
            "deadline_handle_id",
            "deadline_slack_steps",
        ):
            value = _int_attr(name)
            if value < 0:
                continue
            field = f"_deadline_producer_work_{name}"
            cur = int(getattr(self, field, -1))
            setattr(self, field, value if cur < 0 else min(cur, value))

        self._deadline_producer_work_can_drop = int(
            getattr(self, "_deadline_producer_work_can_drop", 0)
            or bool(getattr(producer_work_item, "can_drop", False))
        )
        self._deadline_producer_work_can_coalesce = int(
            getattr(self, "_deadline_producer_work_can_coalesce", 0)
            or bool(getattr(producer_work_item, "can_coalesce", False))
        )
        reason = str(getattr(producer_work_item, "admission_reason", "") or "")
        if reason:
            self._deadline_producer_work_admission_reason = reason

    def _record_pending_refresh_rebuild_drain_submit(
        self,
        pending: "PendingRefreshRebuild",
    ) -> None:
        self._deadline_rebuild_drain_submit_count += 1
        producer_work_item = getattr(pending, "producer_work_item", None)
        if producer_work_item is not None:
            self._record_deadline_producer_work_item(producer_work_item)
        steps = self._pending_refresh_rebuild_decode_steps(pending)
        if not steps and producer_work_item is not None:
            raw_step_min = getattr(producer_work_item, "decode_step_min", -1)
            raw_step_max = getattr(producer_work_item, "decode_step_max", -1)
            step_min = int(-1 if raw_step_min is None else raw_step_min)
            step_max = int(-1 if raw_step_max is None else raw_step_max)
            if step_min >= 0 and step_max >= 0:
                steps = tuple(sorted({step_min, step_max}))
        if not steps:
            return
        cur_min = int(self._deadline_rebuild_drain_submit_decode_step_min)
        cur_max = int(self._deadline_rebuild_drain_submit_decode_step_max)
        step_min = int(min(steps))
        step_max = int(max(steps))
        self._deadline_rebuild_drain_submit_decode_step_min = (
            step_min if cur_min < 0 else min(cur_min, step_min)
        )
        self._deadline_rebuild_drain_submit_decode_step_max = max(cur_max, step_max)
        self._deadline_rebuild_drain_submit_decode_steps.update(steps)
        # Bound this diagnostic set (monotonic decode-step ints): drop the oldest
        # beyond the cap so it cannot grow unbounded in host RAM. No-op until it
        # exceeds the cap (i.e. never for unit-scale step counts).
        _ds = self._deadline_rebuild_drain_submit_decode_steps
        if len(_ds) > _DEADLINE_DRAIN_SUBMIT_DECODE_STEPS_CAP:
            _excess = len(_ds) - _DEADLINE_DRAIN_SUBMIT_DECODE_STEPS_CAP
            for _old_step in sorted(_ds)[:_excess]:
                _ds.discard(_old_step)

    def _record_deferred_selector_compute_profile(
        self,
        *,
        elapsed_cpu_us: float,
        result: Optional["SelectorResult"],
    ) -> None:
        if not self._refresh_profile_enabled():
            return
        elapsed = max(0.0, float(elapsed_cpu_us))
        self._deadline_deferred_selector_compute_count += 1
        self._deadline_deferred_selector_compute_cpu_us_total += elapsed
        self._deadline_deferred_selector_compute_cpu_us_max = max(
            float(self._deadline_deferred_selector_compute_cpu_us_max),
            elapsed,
        )
        if result is None:
            return

        def _profile_us(name: str) -> float:
            raw = getattr(result, name, None)
            if raw is None:
                return 0.0
            try:
                return max(0.0, float(raw))
            except (TypeError, ValueError):
                return 0.0

        inner_compute = _profile_us("profile_cpu_compute_us")
        stack = _profile_us("profile_cpu_stack_us")
        validate = _profile_us("profile_cpu_validate_us")
        key_norms = _profile_us("profile_cpu_key_norms_us")
        key_norms_arena = _profile_us("profile_cpu_key_norms_arena_us")
        key_norms_direct = _profile_us("profile_cpu_key_norms_direct_us")
        key_norms_direct_prepare = _profile_us(
            "profile_cpu_key_norms_direct_prepare_us"
        )
        key_norms_direct_launch = _profile_us(
            "profile_cpu_key_norms_direct_launch_us"
        )
        key_norms_pack = _profile_us("profile_cpu_key_norms_pack_us")
        select = _profile_us("profile_cpu_select_us")
        post = _profile_us("profile_cpu_post_us")
        wrapper_gap = max(0.0, elapsed - inner_compute - post)

        def _accumulate(base: str, value: float) -> None:
            total_name = f"_deadline_deferred_selector_{base}_cpu_us_total"
            max_name = f"_deadline_deferred_selector_{base}_cpu_us_max"
            setattr(self, total_name, float(getattr(self, total_name, 0.0)) + value)
            setattr(self, max_name, max(float(getattr(self, max_name, 0.0)), value))

        _accumulate("inner_compute", inner_compute)
        _accumulate("stack", stack)
        _accumulate("validate", validate)
        _accumulate("key_norms", key_norms)
        _accumulate("key_norms_arena", key_norms_arena)
        _accumulate("key_norms_direct", key_norms_direct)
        _accumulate("key_norms_direct_prepare", key_norms_direct_prepare)
        _accumulate("key_norms_direct_launch", key_norms_direct_launch)
        _accumulate("key_norms_pack", key_norms_pack)
        _accumulate("select", select)
        _accumulate("post", post)
        _accumulate("wrapper_gap", wrapper_gap)

    def _record_deadline_async_producer_stage(
        self,
        stage: str,
        *,
        elapsed_cpu_us: float,
    ) -> None:
        if not self._refresh_profile_enabled():
            return
        elapsed = max(0.0, float(elapsed_cpu_us))
        count_name = f"_deadline_async_producer_{stage}_count"
        total_name = f"_deadline_async_producer_{stage}_cpu_us_total"
        max_name = f"_deadline_async_producer_{stage}_cpu_us_max"
        setattr(self, count_name, int(getattr(self, count_name, 0)) + 1)
        setattr(self, total_name, float(getattr(self, total_name, 0.0)) + elapsed)
        setattr(self, max_name, max(float(getattr(self, max_name, 0.0)), elapsed))

    def _record_deadline_async_producer_gpu_event_pair(
        self,
        stage: str,
        pair: object,
    ) -> None:
        if not self._refresh_profile_enabled():
            return
        try:
            evt0, evt1 = pair  # type: ignore[misc]
        except Exception:
            return
        if evt0 is None or evt1 is None:
            return
        pairs = getattr(self, "_deadline_async_producer_gpu_event_pairs", None)
        if not isinstance(pairs, list):
            pairs = []
            self._deadline_async_producer_gpu_event_pairs = pairs
        pairs.append((str(stage), evt0, evt1))

    def _drain_deadline_async_producer_gpu_profile_events(self) -> None:
        pairs = getattr(self, "_deadline_async_producer_gpu_event_pairs", None)
        if not isinstance(pairs, list) or not pairs:
            return
        kept: List[Tuple[str, object, object]] = []
        for stage, evt0, evt1 in pairs:
            try:
                ready0 = getattr(evt0, "query", None)
                ready1 = getattr(evt1, "query", None)
                if callable(ready0) and not bool(ready0()):
                    kept.append((str(stage), evt0, evt1))
                    continue
                if callable(ready1) and not bool(ready1()):
                    kept.append((str(stage), evt0, evt1))
                    continue
                elapsed_ms = max(0.0, float(evt0.elapsed_time(evt1)))
            except Exception:
                _log.warning(
                    "async producer GPU profile event drain failed",
                    exc_info=True,
                )
                kept.append((str(stage), evt0, evt1))
                continue
            count_name = f"_deadline_async_producer_{stage}_gpu_count"
            total_name = f"_deadline_async_producer_{stage}_gpu_ms_total"
            max_name = f"_deadline_async_producer_{stage}_gpu_ms_max"
            setattr(self, count_name, int(getattr(self, count_name, 0)) + 1)
            setattr(
                self,
                total_name,
                float(getattr(self, total_name, 0.0)) + elapsed_ms,
            )
            setattr(
                self,
                max_name,
                max(float(getattr(self, max_name, 0.0)), elapsed_ms),
            )
        self._deadline_async_producer_gpu_event_pairs = kept

    def _record_deadline_async_producer_count(self, name: str) -> None:
        if not self._refresh_profile_enabled():
            return
        attr = f"_deadline_async_producer_{name}_count"
        setattr(self, attr, int(getattr(self, attr, 0)) + 1)

    def _record_deadline_async_producer_key_norms_delta(self, result: object) -> None:
        if not self._refresh_profile_enabled():
            return

        def _int_attr(name: str, default: int = 0) -> int:
            value = getattr(result, name, default)
            try:
                return int(default if value is None else value)
            except (TypeError, ValueError):
                return int(default)

        total_tokens = max(
            0,
            _int_attr("profile_key_norms_delta_total_tokens", 0),
        )
        max_tokens = _int_attr("profile_key_norms_delta_max_tokens", -1)
        layers = max(0, _int_attr("profile_key_norms_delta_layers", 0))
        if total_tokens <= 0 and max_tokens < 0 and layers <= 0:
            return
        self._deadline_async_producer_key_norms_delta_count += 1
        self._deadline_async_producer_key_norms_delta_total_tokens_total += total_tokens
        self._deadline_async_producer_key_norms_delta_max_tokens_max = max(
            int(self._deadline_async_producer_key_norms_delta_max_tokens_max),
            int(max_tokens),
        )
        self._deadline_async_producer_key_norms_delta_layers_total += layers

    def _pending_refresh_rebuild_compact_stale(self) -> int:
        """压力路径压缩 pending 队列，仅移除非 latest 的陈旧项。"""
        if not self._pending_refresh_rebuilds:
            return 0
        kept: Deque[PendingRefreshRebuild] = deque()
        dropped = 0
        while self._pending_refresh_rebuilds:
            pending = self._pending_refresh_rebuilds.popleft()
            if not _pending_refresh_rebuild_matches_current_target_scope(self, pending):
                self._drop_pending_refresh_rebuild(
                    pending,
                    status="drop_stale",
                    lease_reason="pending_rebuild_target_scope_stale",
                    remaining_pending=tuple(kept) + tuple(self._pending_refresh_rebuilds),
                )
                dropped += 1
                continue
            if self._pending_refresh_rebuild_is_latest(pending):
                kept.append(pending)
                continue
            self._drop_pending_refresh_rebuild(
                pending,
                status="drop_non_latest",
                lease_reason="pending_rebuild_drop_non_latest_queue_compact",
                remaining_pending=tuple(kept) + tuple(self._pending_refresh_rebuilds),
            )
            dropped += 1
        self._pending_refresh_rebuilds = kept
        return dropped

    def _pending_refresh_rebuild_deadline_key(
        self,
        pending: "PendingRefreshRebuild",
    ) -> Tuple[int, int, int, int]:
        missing = 1 << 60
        deadline_handle_id = int(getattr(pending, "deadline_handle_id", -1) or -1)
        deadline_epoch = int(getattr(pending, "deadline_epoch", -1) or -1)
        target_layer_start = int(
            getattr(pending, "target_layer_start", -1) or -1
        )
        ready_epoch = int(getattr(pending, "ready_epoch", -1) or -1)
        pending_id = int(getattr(pending, "pending_id", -1) or -1)
        deadline = deadline_handle_id if deadline_handle_id > 0 else deadline_epoch
        if deadline < 0:
            deadline = missing
        if target_layer_start < 0:
            target_layer_start = missing
        if ready_epoch < 0:
            ready_epoch = missing
        if pending_id < 0:
            pending_id = missing
        return (deadline, target_layer_start, ready_epoch, pending_id)

    def _pending_refresh_rebuild_insert_deadline_ordered(
        self,
        pending: "PendingRefreshRebuild",
    ) -> None:
        """Insert pending rebuild work by earliest consumer deadline."""
        queue = self._pending_refresh_rebuilds
        pending_key = RefreshRebuildMixin._pending_refresh_rebuild_deadline_key(
            self,
            pending,
        )
        if not queue:
            queue.append(pending)
            return
        tail_key = RefreshRebuildMixin._pending_refresh_rebuild_deadline_key(
            self,
            queue[-1],
        )
        if pending_key >= tail_key:
            queue.append(pending)
            return

        reordered: Deque[PendingRefreshRebuild] = deque()
        inserted = False
        while queue:
            current = queue.popleft()
            if not inserted:
                current_key = RefreshRebuildMixin._pending_refresh_rebuild_deadline_key(
                    self,
                    current,
                )
                if pending_key < current_key:
                    reordered.append(pending)
                    inserted = True
            reordered.append(current)
        if not inserted:
            reordered.append(pending)
        self._pending_refresh_rebuilds = reordered

    def _pending_refresh_rebuild_can_coalesce(
        self,
        old: "PendingRefreshRebuild",
        new: "PendingRefreshRebuild",
    ) -> bool:
        if not bool(getattr(old, "can_coalesce", True)):
            return False
        if not bool(getattr(new, "can_coalesce", True)):
            return False
        if getattr(old, "producer_kind", "refresh_rebuild") != getattr(
            new,
            "producer_kind",
            "refresh_rebuild",
        ):
            return False
        old_pending_id = int(getattr(old, "pending_id", -1) or -1)
        new_pending_id = int(getattr(new, "pending_id", -1) or -1)
        if (
            old_pending_id >= 0
            and new_pending_id >= 0
            and old_pending_id >= new_pending_id
        ):
            return False
        if getattr(old, "target_selected_scope_key", None) != getattr(
            new,
            "target_selected_scope_key",
            None,
        ):
            return False
        old_req_ids = self._normalize_refresh_req_ids(getattr(old, "req_ids", None))
        new_req_ids = self._normalize_refresh_req_ids(getattr(new, "req_ids", None))
        if not old_req_ids or old_req_ids != new_req_ids:
            return False
        if int(getattr(old, "target_layer_start", -1) or -1) != int(
            getattr(new, "target_layer_start", -1) or -1
        ):
            return False
        if int(getattr(old, "target_layer_end", -1) or -1) != int(
            getattr(new, "target_layer_end", -1) or -1
        ):
            return False
        old_handle_id = int(getattr(old, "capture_handle_id", -1) or -1)
        new_handle_id = int(getattr(new, "capture_handle_id", -1) or -1)
        if old_handle_id > 0 and new_handle_id > 0 and new_handle_id < old_handle_id:
            return False
        old_ready_epoch = int(getattr(old, "ready_epoch", -1) or -1)
        new_ready_epoch = int(getattr(new, "ready_epoch", -1) or -1)
        if (
            old_ready_epoch >= 0
            and new_ready_epoch >= 0
            and new_ready_epoch < old_ready_epoch
        ):
            return False
        return True

    def _pending_refresh_rebuild_coalesce_superseded(
        self,
        pending: "PendingRefreshRebuild",
    ) -> int:
        """Drop same-scope pending work already superseded by this admission."""
        if not self._pending_refresh_rebuilds:
            return 0
        kept: Deque[PendingRefreshRebuild] = deque()
        coalesced = 0
        while self._pending_refresh_rebuilds:
            old = self._pending_refresh_rebuilds.popleft()
            if old is pending:
                kept.append(old)
                continue
            if RefreshRebuildMixin._pending_refresh_rebuild_can_coalesce(
                self,
                old,
                pending,
            ):
                remaining_pending: Tuple[PendingRefreshRebuild, ...] = (
                    tuple(kept) + tuple(self._pending_refresh_rebuilds)
                )
                new_recorded = self._pending_refresh_rebuild_has_recorded_async_work(
                    pending
                )
                new_covers_old_bufs = set(
                    self._pending_refresh_rebuild_buf_ids(old)
                ).issubset(set(self._pending_refresh_rebuild_buf_ids(pending)))
                if new_recorded:
                    remaining_pending = remaining_pending + (pending,)
                self._drop_pending_refresh_rebuild(
                    old,
                    status="coalesced",
                    lease_reason="pending_rebuild_coalesced_by_deadline_admission",
                    remaining_pending=remaining_pending,
                    wait_recorded_work=not (new_recorded and new_covers_old_bufs),
                )
                coalesced += 1
                continue
            kept.append(old)
        self._pending_refresh_rebuilds = kept
        return coalesced

    def _pending_refresh_rebuild_count_superseded(
        self,
        pending: "PendingRefreshRebuild",
    ) -> int:
        """Count same-scope pending work without mutating queue or leases."""
        if not self._pending_refresh_rebuilds:
            return 0
        count = 0
        for old in self._pending_refresh_rebuilds:
            if old is pending:
                continue
            if RefreshRebuildMixin._pending_refresh_rebuild_can_coalesce(
                self,
                old,
                pending,
            ):
                count += 1
        return int(count)

    def _pending_refresh_rebuild_finish_boundary_drain(
        self,
        *,
        finished_ids: Set[str],
    ) -> Tuple[int, int, int]:
        """Resolve pending rebuild work before finished slots are released.

        Deadline rule:
          - if all rows covered by a pending rebuild are finished, the work is
            stale before any future decode can consume it, so drop it;
          - if the item also contains active rows, drain it now so active rows
            can still receive the refresh while finished slots are released
            only after the rebuild has been submitted.
        """
        if not finished_ids or not self._pending_refresh_rebuilds:
            return (0, 0, 0)

        kept: Deque[PendingRefreshRebuild] = deque()
        dropped = 0
        drained = 0
        partial = 0
        pending_buf_ref_counts = self._pending_refresh_rebuild_buf_ref_counts(
            tuple(self._pending_refresh_rebuilds)
        )

        while self._pending_refresh_rebuilds:
            pending = self._pending_refresh_rebuilds.popleft()
            req_ids = self._normalize_refresh_req_ids(pending.req_ids)
            if not req_ids:
                kept.append(pending)
                continue
            req_set = set(req_ids)
            if req_set.isdisjoint(finished_ids):
                kept.append(pending)
                continue
            if req_set.issubset(finished_ids) and bool(
                getattr(pending, "can_drop", True)
            ):
                if self._wait_pending_refresh_rebuild_writer_done(
                    pending
                ) or self._wait_pending_refresh_rebuild_selector_done(pending):
                    self._pending_refresh_rebuild_clear_completed_buf_work(
                        pending,
                        pending_buf_ref_counts=pending_buf_ref_counts,
                    )
                _mark_pending_selected_scope_terminal(
                    pending,
                    status="drop_finished",
                )
                drop_req_ids = self._pending_refresh_rebuild_drop_req_ids(
                    pending,
                    req_ids=req_ids,
                )
                if drop_req_ids:
                    self._resolve_refresh_lease(
                        req_ids=drop_req_ids,
                        reason="pending_rebuild_drop_finished",
                    )
                self._pending_refresh_rebuild_clear(pending)
                dropped += 1
                continue

            if self._wait_pending_refresh_rebuild_writer_done(pending):
                if not self._pending_refresh_rebuild_is_latest(pending):
                    _mark_pending_selected_scope_terminal(
                        pending,
                        status="drop_non_latest",
                    )
                    drop_req_ids = self._pending_refresh_rebuild_drop_req_ids(
                        pending,
                        req_ids=req_ids,
                    )
                    if drop_req_ids:
                        self._resolve_refresh_lease(
                            req_ids=drop_req_ids,
                            reason="pending_rebuild_drop_non_latest_finish_boundary",
                        )
                    self._pending_refresh_rebuild_clear_completed_buf_work(
                        pending,
                        pending_buf_ref_counts=pending_buf_ref_counts,
                    )
                    self._pending_refresh_rebuild_clear(pending)
                    dropped += 1
                    continue
                result = getattr(pending, "result", None)
                if result is None:
                    raise RuntimeError(
                        "pending refresh writer_done_event has no selection result"
                    )
                self._commit_pending_refresh_rebuild_compact_meta(pending)
                if not bool(getattr(pending, "tracking_published", False)):
                    self._publish_pending_refresh_rebuild_selection_tracking(
                        pending,
                        result,
                    )
                    pending.tracking_published = True
                self._mark_pending_refresh_rebuild_accepted(pending)
                self._pending_refresh_rebuild_clear_completed_buf_work(
                    pending,
                    pending_buf_ref_counts=pending_buf_ref_counts,
                )
                self._pending_refresh_rebuild_clear(pending)
                drained += 1
                partial += 1
                continue

            self._record_pending_refresh_rebuild_drain_submit(pending)
            self._submit_pending_refresh_rebuild(pending)
            self._pending_refresh_rebuild_clear(pending)
            drained += 1
            partial += 1

        self._pending_refresh_rebuilds = kept
        self._deadline_rebuild_drop_finished_count += dropped
        self._deadline_rebuild_drain_finish_count += drained
        self._deadline_rebuild_partial_finish_count += partial
        return dropped, drained, partial

    def _pending_refresh_rebuild_pre_consume_drain(self) -> Tuple[int, int, int]:
        """Submit pending producer work before its consumer/deadline boundary.

        This is a deadline drain, not a time-sliced background scheduler. Scoped
        work moves only when its published selected scope is the current consume
        scope; unscoped producer work moves when its explicit handle/epoch
        deadline arrives. Future current-target work stays queued.
        """
        if not self._pending_refresh_rebuilds:
            return (0, 0, 0)
        if _is_stream_capturing_or_raise(stage="pending_refresh_rebuild_pre_consume_drain"):
            raise RuntimeError(
                "pending refresh rebuild pre-consume drain cannot run during CUDA graph capture"
            )

        step_authority = getattr(self, "step_authority", None)
        if step_authority is None:
            step_context = getattr(self, "step_context", None)
            step_authority = getattr(step_context, "step_authority", None)
        consume_scope_key = getattr(step_authority, "consume_selected_scope_key", None)
        current_handle_id = int(getattr(step_authority, "step_handle_id", -1) or -1)
        current_epoch = int(getattr(step_authority, "epoch", -1) or -1)
        if current_handle_id <= 0:
            step_context = getattr(self, "step_context", None)
            current_handle_id = int(getattr(step_context, "step_handle_id", -1) or -1)
        if current_epoch < 0:
            current_epoch = int(getattr(self, "step_context_epoch", -1) or -1)

        def _unscoped_deadline_reached(pending: "PendingRefreshRebuild") -> bool:
            deadline_handle_id = int(getattr(pending, "deadline_handle_id", -1) or -1)
            if deadline_handle_id > 0 and current_handle_id > 0:
                return deadline_handle_id <= current_handle_id
            deadline_epoch = int(getattr(pending, "deadline_epoch", -1) or -1)
            return deadline_epoch >= 0 and current_epoch >= 0 and deadline_epoch <= current_epoch

        def _async_writer_ready(pending: "PendingRefreshRebuild") -> bool:
            writer_event = getattr(pending, "writer_done_event", None)
            query = getattr(writer_event, "query", None)
            if not callable(query):
                return False
            try:
                return bool(query())
            except Exception:
                _log.warning(
                    "pending refresh rebuild writer event query failed",
                    exc_info=True,
                )
                raise

        kept: Deque[PendingRefreshRebuild] = deque()
        due_to_submit: List[PendingRefreshRebuild] = []
        drained = 0
        dropped_stale = 0
        dropped_non_latest = 0
        while self._pending_refresh_rebuilds:
            pending = self._pending_refresh_rebuilds.popleft()
            if not self._pending_refresh_rebuild_is_latest(pending):
                self._drop_pending_refresh_rebuild(
                    pending,
                    status="drop_non_latest",
                    lease_reason="pending_rebuild_drop_non_latest_pre_consume",
                    remaining_pending=(
                        tuple(kept)
                        + tuple(due_to_submit)
                        + tuple(self._pending_refresh_rebuilds)
                    ),
                )
                dropped_non_latest += 1
                continue

            target_scope_key = getattr(pending, "target_selected_scope_key", None)
            due_by_consume_scope = (
                target_scope_key is not None
                and _selected_scope_target_matches_key(target_scope_key, consume_scope_key)
            )
            writer_event = getattr(pending, "writer_done_event", None)
            async_writer_ready = _async_writer_ready(pending)
            deadline_reached = (
                target_scope_key is None and _unscoped_deadline_reached(pending)
            )
            # Unscoped async producer work is covered by full-KV handoff while
            # it is not ready.  Do not turn a freshness deadline into a decode
            # stream wait; publish it as soon as the writer event is ready.
            due_by_unscoped_deadline = deadline_reached and (
                writer_event is None
                or async_writer_ready
            )
            due_by_async_writer_ready = (
                target_scope_key is None
                and not deadline_reached
                and async_writer_ready
            )
            due_by_consume_scope_ready = due_by_consume_scope and (
                writer_event is None or async_writer_ready
            )
            if due_by_consume_scope_ready or due_by_unscoped_deadline or due_by_async_writer_ready:
                self._record_pending_refresh_rebuild_drain_submit(pending)
                due_to_submit.append(pending)
                drained += 1
                continue

            if (
                target_scope_key is not None
                and not _pending_refresh_rebuild_matches_current_target_scope(
                    self,
                    pending,
                )
            ):
                self._drop_pending_refresh_rebuild(
                    pending,
                    status="drop_stale",
                    lease_reason="pending_rebuild_target_scope_stale_pre_consume",
                    remaining_pending=(
                        tuple(kept)
                        + tuple(due_to_submit)
                        + tuple(self._pending_refresh_rebuilds)
                    ),
                )
                dropped_stale += 1
                continue

            kept.append(pending)

        self._pending_refresh_rebuilds = kept
        if due_to_submit:
            pending_buf_ref_counts = self._pending_refresh_rebuild_buf_ref_counts(
                tuple(kept) + tuple(due_to_submit)
            )
            # ===== Off-loop pre-publish drain (spec 2026-05-10) =====
            # async_prepared_due 已经在 enqueue 阶段跑完 selector + writer 并
            # 在 refresh stream 上录了 writer_done_event；这里只 wait + publish
            # tracking + accept，**不再触发 selector / writer 启动**。
            # 其余 sync_due 走旧路径（_async_refresh_enabled() 关闭、enqueue
            # 在 cudagraph capture 中、queue 已满 bypass 等场景）。
            async_prepared_due: List[PendingRefreshRebuild] = []
            sync_due: List[PendingRefreshRebuild] = []
            for pending in due_to_submit:
                if pending.writer_done_event is not None:
                    async_prepared_due.append(pending)
                else:
                    sync_due.append(pending)
            if async_prepared_due:
                device = async_prepared_due[0].payloads[0].key_cache.device
                cur_stream = torch.cuda.current_stream(device=device)
                waited_writer_event_ids: Set[int] = set()
                for pending in async_prepared_due:
                    writer_event = pending.writer_done_event
                    event_id = id(writer_event)
                    if event_id not in waited_writer_event_ids:
                        cur_stream.wait_event(writer_event)
                        waited_writer_event_ids.add(event_id)
                    self._commit_pending_refresh_rebuild_compact_meta(
                        pending
                    )
                    if not pending.tracking_published and pending.result is not None:
                        self._publish_pending_refresh_rebuild_selection_tracking(
                            pending, pending.result
                        )
                        pending.tracking_published = True
                    self._mark_pending_refresh_rebuild_accepted(pending)
                    self._pending_refresh_rebuild_clear_completed_buf_work(
                        pending,
                        pending_buf_ref_counts=pending_buf_ref_counts,
                    )
            if sync_due:
                self._submit_pending_refresh_rebuild_batch(tuple(sync_due))
            for pending in due_to_submit:
                self._pending_refresh_rebuild_clear(pending)
        self._deadline_rebuild_pre_consume_drain_count += drained
        self._deadline_rebuild_pre_consume_drop_stale_count += (
            dropped_stale + dropped_non_latest
        )
        return drained, dropped_stale + dropped_non_latest, len(kept)

    def _validate_pending_refresh_rebuild_result(
        self,
        pending: PendingRefreshRebuild,
        result: "SelectorResult",
    ) -> None:
        payloads = pending.payloads
        if not payloads:
            return
        first = payloads[0]
        sel = result.selected_indices
        if sel.dim() != 4 or sel.shape[0] != len(payloads):
            raise RuntimeError(
                "pending refresh rebuild: selected_indices shape mismatch "
                f"layers={len(payloads)} shape={tuple(sel.shape)}"
            )
        batch = len(first.slot_list or [])
        if batch > 0 and sel.shape[1] != batch:
            raise RuntimeError(
                "pending refresh rebuild: batch mismatch "
                f"slots={batch} sel_b={int(sel.shape[1])}"
            )
        for idx, payload in enumerate(payloads):
            num_kv = int(payload.state.num_kv_heads or 0)
            if num_kv > 0 and int(sel.shape[2]) != num_kv:
                raise RuntimeError(
                    "pending refresh rebuild: kv_head mismatch "
                    f"layer_idx={idx} sel_kv={int(sel.shape[2])} kv_heads={num_kv}"
                )
            if payload.key_cache is None or payload.value_cache is None:
                raise RuntimeError("pending refresh rebuild: key/value cache missing")
            if payload.key_cache.shape != payload.value_cache.shape:
                raise RuntimeError("pending refresh rebuild: key/value shape mismatch")

    def _resolve_pending_refresh_rebuild_result(
        self,
        pending: PendingRefreshRebuild,
        *,
        profile_deferred: bool = True,
    ) -> Optional["SelectorResult"]:
        payloads = pending.payloads
        if not payloads:
            return None
        result = pending.result
        if pending.result is None:
            profile_enabled = bool(self._refresh_profile_enabled())
            profile_detail = profile_enabled and bool(
                _DEFERRED_SELECTOR_PROFILE_DETAIL_CACHED
            )
            profile_t0_ns = time.perf_counter_ns() if profile_enabled else None
            prev_profile_active = bool(getattr(self, "_refresh_profile_active", False))
            if profile_detail and not prev_profile_active:
                self._refresh_profile_active = True
            # [SELECTED-PRIVATE-OUT 2026-07-07] pending 路径的 selected_indices
            # 私有化:共享单槽 _selector_selected_indices_out 会被下一次同 shape
            # selector run 原地覆写,而本 result 的 selected 要活到 deferred
            # writer 消费(OFF 臂直接持引用;ON 臂 stable copy_ 可能跨流未决)
            # ——replay-refresh 路径已有同款防护(_apply_with_private_selected_
            # indices_out),pending 路径此前漏包=4B bs8 illegal address 毒源之一。
            # 只私有化 selected 一槽:其余 scratch(bounds/workspace/key_norms)
            # 均为 run 内消费无 deferred 读者,保持共享复用零额外分配;selected
            # 私有分配为每世代一次 allocator 缓存命中(µs 级),零热路径开销。
            _sel_out_override_prev = getattr(
                self, "_selector_selected_indices_out_override", None
            )
            # [SELECTED-OUT-RING v2 2026-07-09] 槽接管=七 override 载体整套换
            # 装为该槽的持久 SlotStableOverrides 容器(指针稳定→graph key 可
            # 命中;valid 集已在 begin_run 清空=key_norms 每 run 强制重填);
            # spill(容量满)/env 关=旧行为原路(仅 indices per-run 私有,
            # off-loop 外层全量私有化+retention 原样生效)。槽随 end_run 带
            # produce 事件返还并绑定 pending 至终局漏斗;run 失败=即时回收。
            # 配对键=chunk_id(非 buf_id):世代 chunk 的层组(14/8 切分)决定
            # 六容器的形状族,按 chunk 配对令每槽形状恒定(buf_id 配对时
            # slot0 交替 chunk0/chunk2 两形状族→容器逐 run realloc→graph key
            # 永不复现→64 窗 thrash 闩死,rv3g/rv4dbg 取证形态)。
            _sel_out_ring = self._selected_out_ring_for_pending()
            _ring_slot = (
                _sel_out_ring.begin_run(
                    preferred=int(getattr(pending, "chunk_id", -1) or -1)
                )
                if _sel_out_ring is not None
                else None
            )
            _slot_prevs: Optional[Dict[str, object]] = None
            if _ring_slot is not None:
                _slot_prevs = {}
                for _attr, _container in _ring_slot.containers.items():
                    _slot_prevs[_attr] = getattr(self, _attr, None)
                    setattr(self, _attr, _container)
                _slot_prevs[SLOT_VALID_SET_ATTR] = getattr(
                    self, SLOT_VALID_SET_ATTR, None
                )
                setattr(self, SLOT_VALID_SET_ATTR, _ring_slot.valid_keys)
            else:
                self._selector_selected_indices_out_override = {}
            result = None
            try:
                result = self._apply_alpha_selector_batched_fused(
                    payloads,
                    phase=pending.selection_phase,
                    update_tracking=False,
                )
            finally:
                if _slot_prevs is not None:
                    for _attr, _prev in _slot_prevs.items():
                        setattr(self, _attr, _prev)
                else:
                    self._selector_selected_indices_out_override = (
                        _sel_out_override_prev
                    )
                if _sel_out_ring is not None:
                    _end_slot = _sel_out_ring.end_run()
                    if result is None:
                        _sel_out_ring.release(_end_slot, writer_done_event=None)
                    elif _end_slot is not None:
                        pending.selected_out_ring_slot = _end_slot
                if profile_detail and not prev_profile_active:
                    self._refresh_profile_active = prev_profile_active
            if profile_t0_ns is not None and profile_deferred:
                self._record_deferred_selector_compute_profile(
                    elapsed_cpu_us=(time.perf_counter_ns() - profile_t0_ns) / 1000.0,
                    result=result,
                )
            if result is None:
                return None
            pending.result = result
        if self._refresh_rebuild_check_enabled():
            self._validate_pending_refresh_rebuild_result(pending, result)
        return result

    def _publish_pending_refresh_rebuild_selection_tracking(
        self,
        pending: PendingRefreshRebuild,
        result: "SelectorResult",
    ) -> None:
        self._update_selection_tracking(
            pending.payloads,
            result,
            phase=pending.selection_phase,
            profile_cpu_detail=False,
            t_post0_ns=None,
            pending_refresh_rebuild=pending,
        )

    def _commit_compact_meta_log_entries(
        self,
        commit_log: Sequence[Dict[str, object]],
        *,
        source: str = "?",
    ) -> None:
        def _reset_compact_slot_metadata(state: "LayerState", slot: int) -> None:
            if slot < 0 or slot >= state.batch_size:
                return
            if slot >= len(state.compact_k):
                return
            empty_k = torch.empty(0, device=state.device)
            empty_v = torch.empty(0, device=state.device)
            empty_p = torch.empty(0, device=state.device, dtype=torch.int32)
            state.compact_k[slot] = empty_k
            state.compact_v[slot] = empty_v
            state.compact_pos[slot] = empty_p
            state.compact_capacity[slot] = 0
            state.compact_offset_tokens[slot] = 0
            state.compact_sink_len[slot] = 0
            state.compact_persist_len[slot] = 0
            state.compact_kv_len[slot] = 0
            if slot < len(state.compact_pad_zeroed_len):
                state.compact_pad_zeroed_len[slot] = -1
            state.compact_views_bound_slots = min(
                int(getattr(state, "compact_views_bound_slots", 0)),
                int(slot),
            )

        # [DUAL-GEN-LAYER-PARITY] 本 commit 轮的 slot→落代账本(跨层一致性)。
        self._dual_gen_flip_parity_by_slot = {}
        # [DUAL-GEN-PARITY-FORENSICS 2026-07-08] commit 轮审计环:每轮记
        # (来源, 各 entry 的 layer_index+flip_slots)。违例时把近 16 轮历史
        # +全层 read_gen[slot] 快照带进异常消息,一次定名分叉形态(块状丢轮
        # =flush 暂存覆写型 / 单层双翻=double-append 型)。纯诊断,健康路径
        # 仅每 flip 轮一次 O(entries) 追加。
        _audit_ring = getattr(self, "_dual_gen_commit_audit", None)
        if _audit_ring is None:
            from collections import deque

            _audit_ring = deque(maxlen=16)
            self._dual_gen_commit_audit = _audit_ring
        _audit_round: Dict[str, object] = {
            "src": str(source),
            "n": len(tuple(commit_log)),
            "entries": [],
        }
        _audit_pushed = False

        def _dual_gen_read_gen_table(slot: int) -> str:
            rows = []
            for _ck in getattr(self, "layer_cache_keys", ()):  # 层序稳定
                _st = self.layer_states.get(_ck)
                if _st is None:
                    continue
                _rg = getattr(_st, "compact_read_gen", None)
                _li = int(getattr(_st, "layer_index", -1))
                if _rg is not None and 0 <= slot < len(_rg):
                    rows.append(f"L{_li}:{int(_rg[slot])}")
            return ",".join(rows)

        for entry in tuple(commit_log):
            state = entry["state"]
            if str(entry.get("phase", "")) == "refresh":
                state.last_reason = "refresh"
            for raw_slot in tuple(entry.get("reset_slot_commits", tuple())):
                _reset_compact_slot_metadata(state, int(raw_slot))
            # [DUAL-GEN-L2a-C] writer_done 已在 main stream wait 之后:对本批
            # writer 写过的 slot 原子切读代(read_gen 翻转+offset/三视图指向
            # 新半区)。后续 slot_meta/pad_marker commits 描述的即新半区内容,
            # 顺序自洽。关态恒空 tuple=零行为。
            _flip_slots = tuple(entry.get("dual_gen_flip_slots", tuple()))
            if _flip_slots:
                if not _audit_pushed:
                    _audit_ring.append(_audit_round)
                    _audit_pushed = True
                _audit_round["entries"].append(
                    (
                        int(getattr(entry["state"], "layer_index", -1)),
                        str(entry.get("phase", "")),
                        tuple(int(s) for s in _flip_slots),
                    )
                )
                from patches.fa_sparse_runtime.compact_recent_alignment import (
                    compact_slot_offset_tokens,
                )

                _residency = getattr(state, "compact_page_residency", None)
                if _residency is None:
                    raise RuntimeError(
                        "compact dual-gen flip requires page residency"
                    )
                _stride = int(state.compact_stride_tokens)
                _gen_stride = (
                    int(_residency.lease.max_live_sparse_slots) * _stride
                )
                _arena_k = state.compact_arena_k
                _arena_v = state.compact_arena_v
                _arena_pos = state.compact_arena_pos
                _seen_flip_slots: set = set()
                for _raw_slot in _flip_slots:
                    _slot = int(_raw_slot)
                    if _slot < 0 or _slot >= len(state.compact_read_gen):
                        raise RuntimeError(
                            f"dual-gen flip slot {_slot} out of read_gen range"
                        )
                    # [DUAL-GEN-FLIP-UNIQUE] 同一 commit entry 内 slot 重复
                    # 翻两次=读回旧半区(静默 stale),必须炸。
                    if _slot in _seen_flip_slots:
                        raise RuntimeError(
                            f"dual-gen flip slot {_slot} duplicated in one "
                            "commit entry (double flip reads stale half)"
                        )
                    _seen_flip_slots.add(_slot)
                    _new_gen = 1 - int(state.compact_read_gen[_slot])
                    # [DUAL-GEN-LAYER-PARITY] 同一 commit 轮内各层对同 slot 的
                    # 落代必须一致;层间错代=部分层读旧半区(静默 stale)。
                    _parity = self._dual_gen_flip_parity_by_slot
                    _prev_gen = _parity.setdefault(_slot, _new_gen)
                    if _prev_gen != _new_gen:
                        raise RuntimeError(
                            f"dual-gen flip parity violation at slot {_slot}: "
                            f"layer entries land gen {_prev_gen} vs {_new_gen}; "
                            f"offending layer_index="
                            f"{int(getattr(state, 'layer_index', -1))}; "
                            f"read_gen[slot] by layer: "
                            f"{_dual_gen_read_gen_table(_slot)}; "
                            f"recent flip commit rounds (oldest->newest): "
                            f"{list(_audit_ring)}"
                        )
                    state.compact_read_gen[_slot] = _new_gen
                    _off = compact_slot_offset_tokens(
                        slot=_slot,
                        stride_tokens=_stride,
                        read_gen=_new_gen,
                        gen_stride_tokens=_gen_stride,
                    )
                    state.compact_offset_tokens[_slot] = _off
                    state.compact_k[_slot] = _arena_k.narrow(0, _off, _stride)
                    state.compact_v[_slot] = _arena_v.narrow(0, _off, _stride)
                    state.compact_pos[_slot] = _arena_pos.narrow(1, _off, _stride)
                state.bump_compact_meta_epoch()
            for raw_slot, raw_sink_len, raw_persist_len, raw_kv_len in tuple(
                entry.get("slot_meta_commits", tuple())
            ):
                slot = int(raw_slot)
                if slot < 0 or slot >= len(state.compact_kv_len):
                    continue
                state.compact_sink_len[slot] = int(raw_sink_len)
                state.compact_persist_len[slot] = int(raw_persist_len)
                state.compact_kv_len[slot] = int(raw_kv_len)
            for raw_slot, raw_marker in tuple(
                entry.get("pad_marker_commits", tuple())
            ):
                slot = int(raw_slot)
                if 0 <= slot < len(state.compact_pad_zeroed_len):
                    state.compact_pad_zeroed_len[slot] = int(raw_marker)
            for raw_slot in tuple(entry.get("bootstrap_slots", tuple())):
                slot = int(raw_slot)
                if slot >= len(state.compact_kv_len) or int(state.compact_kv_len[slot]) <= 0:
                    raise RuntimeError(
                        f"bootstrap slot {slot} has empty compact buffer after rebuild"
                    )
            state.bump_compact_meta_epoch()

    def _commit_pending_refresh_rebuild_compact_meta(
        self,
        pending: PendingRefreshRebuild,
    ) -> None:
        commit_log = getattr(pending, "compact_meta_commit_log", None)
        # [PENDING-FUNNEL-DEBUG 临时取证探针,破案后拆]
        _funnel_dbg = os.environ.get("VLLM_SPARSE_PENDING_FUNNEL_DEBUG_LOG", "")
        if _funnel_dbg:
            try:
                _fd_detail = [
                    (
                        int(getattr(e.get("state"), "layer_index", -1)),
                        len(tuple(e.get("slot_meta_commits", ()) or ())),
                        len(tuple(e.get("dual_gen_flip_slots", ()) or ())),
                        len(tuple(e.get("reset_slot_commits", ()) or ())),
                    )
                    for e in tuple(commit_log or ())
                ]
                with open(_funnel_dbg, "a") as _fd_fh:
                    _fd_fh.write(
                        f"commit\tpid={int(getattr(pending, 'pending_id', -1))}\t"
                        f"reqs={list(getattr(pending, 'req_ids', ()) or ())}\t"
                        f"log_entries={len(commit_log) if commit_log else 0}\t"
                        f"entries(layer,meta,flip,reset)={_fd_detail[:4]}...\t"
                        f"epoch={int(getattr(self, 'step_context_epoch', -1))}\n"
                    )
            except OSError:
                pass
        if not commit_log:
            pending.compact_meta_commit_log = None
            return
        self._commit_compact_meta_log_entries(
            tuple(commit_log), source="pending_rebuild"
        )
        pending.compact_meta_commit_log = None

    def _commit_flush_compact_meta_for_buf(self, buf_id: int) -> None:
        commit_logs = getattr(self, "_flush_compact_meta_commit_log_by_buf", None)
        if not isinstance(commit_logs, list) or not commit_logs:
            return
        buf = int(buf_id) % len(commit_logs)
        # [FLUSH-META-LOG-QUEUE 2026-07-08] 槽=轮队列(见 flush_worker stage
        # 侧注释:覆写形态会静默丢整轮 slot_meta/翻代提交)。按 stage 顺序
        # 逐轮 commit,轮边界保持=parity 守卫语义不变。
        rounds = commit_logs[buf]
        if not rounds:
            commit_logs[buf] = []
            return
        commit_logs[buf] = []
        for round_idx, commit_log in enumerate(rounds):
            if commit_log:
                self._commit_compact_meta_log_entries(
                    tuple(commit_log), source=f"flush_buf{buf}#{round_idx}"
                )

    # ------------------------------------------------------------------
    # ASYNC_PRODUCER_WRITER_GRAPH (task #9): captured writer-graph dispatcher.
    # ------------------------------------------------------------------
    _WRITER_GRAPH_THRASH_RECAPTURES = 4
    _WRITER_GRAPH_THRASH_WINDOW = 64

    def _writer_graph_split_active(self) -> bool:
        """Defect #4: per-pending fresh selected_indices_out override -> eager.

        In split-writer (deferred) mode ``_selector_selected_indices_out_override``
        is a fresh dict per pending and the selected_indices buffer is freshly
        allocated each pending, so its data_ptr changes every refresh and the
        graph cannot be reused. Bypass capture/replay entirely in that mode.

        [SELECTED-OUT-RING v2 engagement 2026-07-09] the production pending
        path installs the ring slot's persistent ``SlotStableOverrides``
        containers (a dict subclass): slot buffers are data_ptr-stable and the
        writer's four key inputs are ADDITIONALLY copied into writer-side
        persistent stable buffers (``_ensure_selector_writer_*_all``), so
        capture/replay is safe there. The pre-ring bare ``isinstance(dict)``
        test latched that path to eager forever — the "N captures / 0 replays"
        writer-track case (captures only ever happened on the override-less
        sync/bootstrap windows). Mirror of the selector track's
        ``_selector_topk_graph_stable_active``. Bare per-run dicts (ring
        spill / ring escape env / replay-refresh private helper) still bypass:
        their buffers are per-run transient and must never be baked.
        """
        override = getattr(self, "_selector_selected_indices_out_override", None)
        return isinstance(override, dict) and not isinstance(
            override, SlotStableOverrides
        )

    def _writer_graph_record_recapture(self) -> bool:
        """Thrash adjudicator: latch bypass only on UNBOUNDED key churn.

        [SYNC-STORM family 2026-07-09, mirrors the selector track] recapture
        volume alone is NOT churn: a bounded key family (layer-groups x
        writer-stable buffer generations; bootstrap and batch-composition
        windows recapture legitimately) can exceed any count threshold and
        must never latch bypass — that latch was exactly the selector track's
        "capture==THRASH_WINDOW then dead" shape. Unbounded churn ALWAYS
        overflows the 8-entry per-key graph cache (clear-before-insert), so
        require BOTH signals inside one window: recaptures over threshold AND
        at least one cache clear.
        """
        self._writer_graph_recapture_window += 1
        self._writer_graph_recapture_count += 1
        if self._writer_graph_recapture_window >= int(self._WRITER_GRAPH_THRASH_WINDOW):
            recaptures = int(self._writer_graph_recapture_count)
            clears = int(self._writer_graph_evict_clear_count)
            self._writer_graph_recapture_window = 0
            self._writer_graph_recapture_count = 0
            self._writer_graph_evict_clear_count = 0
            return (
                recaptures > int(self._WRITER_GRAPH_THRASH_RECAPTURES)
                and clears > 0
            )
        return False

    def _writer_graph_dispatch_launch(
        self,
        *,
        eager_fn: Callable[[], None],
        ext: Any,
        launch_args: Tuple[Any, ...],
        device: torch.device,
        pointer_rebuild_miss: bool,
        key_fields: Tuple[int, ...],
        key_unresolved: bool = False,
    ) -> None:
        """Replay the captured writer graph on a key hit; else eager (+recapture).

        Fail-OPEN: any capture/replay error discards the graph, runs eager and
        latches bypass. Byte output is identical across eager/cold/replay.

        [PTR-REPUBLISH-REPLAY-SAFE 2026-07-09] ``pointer_rebuild_miss`` (a
        pointer-ARRAY republish happened while building launch_args) no longer
        pops the key's graph nor blocks replay/capture. The graph bakes the
        persistent GPU pointer-array BUFFER addresses, not their contents; a
        republish rewrites contents in place (same buffer) with its H2D +
        ready-event ordered on this stream BEFORE this dispatch, so a replayed
        kernel reads the fresh pointers. This was the writer-track engagement
        killer: the per-flush [DETERMINISTIC-BLOCKTABLE-SNAPSHOT] clone gives
        block_table a fresh data_ptr EVERY generation -> block_table_ptrs
        republishes every generation (wg2 forensics: 462/462 republishes are
        block_table L0:ptr) -> the old pop-on-miss dropped the graph each
        generation (pop/capture alternation, replay forever 0). The case the
        pop actually protected — the pointer-array GPU buffer itself
        REALLOCATING (baked address dies) — is now handled at the realloc
        site: _get_rebuild_ptr_buffers clears the whole writer-graph state
        (cold event, size=layers is steady-state constant).

        ``key_unresolved`` (layer-group identity unresolved, -1 fields): the
        degenerate key would collide both layer groups -> never capture or
        replay under it (the pre-split ROW1-corruption regime); eager only.
        """
        _wg_dbg = os.environ.get("VLLM_SPARSE_SELECTOR_TOPK_GRAPH_DEBUG_LOG", "")

        def _wg_dbg_note(action: str) -> None:
            # engagement 取证(诊断档默认关;与 capture 侧 keys 落盘同文件):
            # 全分支 action 分布是"修后分布变化"检验的判据载体(selector 轨
            # 三层破案同仪器)。
            if not _wg_dbg:
                return
            try:
                with open(_wg_dbg, "a") as _fh:
                    _fh.write(
                        f"{os.getpid()}\twriter_dispatch\t{action}\t"
                        f"{int(bool(pointer_rebuild_miss))}\t"
                        f"{tuple(int(v) for v in key_fields)}\n"
                    )
            except OSError:
                pass

        state = self._writer_graph_state
        if isinstance(state, dict) and bool(state.get("bypass")):
            _wg_dbg_note("bypass_latched")
            eager_fn()
            return
        # Defect #4: never capture/replay under the split/deferred writer path.
        if self._writer_graph_split_active():
            _wg_dbg_note("split_bypass")
            eager_fn()
            return
        if key_unresolved:
            # Degenerate (-1) layer-group identity: both groups collapse onto
            # one key — capturing or replaying here IS the ROW1 corruption.
            _wg_dbg_note("eager_key_unresolved")
            eager_fn()
            return
        key = tuple(int(v) for v in key_fields)
        # #9-KEY v5: per-key graph CACHE. Two writer pendings per refresh
        # (contiguous layer chunks) alternate forever; a single global slot
        # either COLLIDES (pre-v5 ROW1 corruption, when keys lacked group
        # identity) or evicts itself every call (with group-distinct keys),
        # so each key owns its graph. Each entry owns a PRIVATE mempool (the
        # captured region allocates nothing, so the pool stays empty): a
        # shared pool dies with its last graph (allocator use_count hits 0)
        # and the next capture_begin on the dead pool id trips the
        # CUDACachingAllocator "use_count > 0" INTERNAL ASSERT (wg2 forensics
        # traceback) which then latched bypass forever.
        graphs = state.get("graphs") if isinstance(state, dict) else None
        entry = graphs.get(key) if isinstance(graphs, dict) else None
        # Hot path: key hit and a captured graph exists -> replay only.
        if entry is not None:
            try:
                # #9-KEY v3: drain the writer-input ready latch before replay so
                # the captured kernel sees the refreshed seq_lens/slot/selected
                # buffers. Same-stream this is a no-op; it guards a future
                # side-stream refactor. Fail-open inside the helper.
                _wait_input_ready = getattr(self, "_wait_writer_input_ready", None)
                if callable(_wait_input_ready):
                    _wait_input_ready(device=device)
                entry.replay()
                self._record_deadline_async_producer_count("graph_replay")
                _wg_dbg_note("replay")
                return
            except Exception:
                _log.warning(
                    "writer graph replay failed; bypassing capture", exc_info=True
                )
                self._writer_graph_state = {"bypass": True}
                _wg_dbg_note("replay_fail")
                eager_fn()
                return
        # Cold / new-key step: run eager NOW, then capture. A pointer-array
        # republish this step is NOT an obstacle: the H2D lands on this stream
        # before the eager launch and before the capture's pre-drain
        # (_prepare_rebuild_ptr_ready_events_for_capture waits every ready
        # event on the capture stream), and the capture bakes only the kernel
        # launch — never the H2D.
        eager_fn()
        try:
            self._capture_writer_graph(
                ext=ext, launch_args=launch_args, device=device, key=key
            )
            _wg_dbg_note("eager_capture")
        except Exception:
            _log.warning(
                "writer graph capture failed; bypassing capture", exc_info=True
            )
            self._writer_graph_state = {"bypass": True}
            if _wg_dbg:
                # capture 异常全文落盘:vLLM logger 默认走 stdout 被 bench IPC
                # 吞(§10.1 坑),此处是拿到 traceback 的唯一稳定通道。
                try:
                    import traceback as _tb

                    with open(_wg_dbg, "a") as _fh:
                        _fh.write(
                            f"{os.getpid()}\twriter_capture_exc\t"
                            f"{_tb.format_exc()!r}\n"
                        )
                except OSError:
                    pass
            _wg_dbg_note("capture_fail")

    def _capture_writer_graph(
        self,
        *,
        ext: Any,
        launch_args: Tuple[Any, ...],
        device: torch.device,
        key: Tuple[int, ...],
    ) -> None:
        """Capture ONLY the literal ext launch closure (defects #1,#2).

        The full producer impl is NOT re-invoked inside the graph (that would
        re-run the pointer-lookup ``wait_event`` and abort capture). We capture a
        thin closure that re-issues ``ext.gather_...(*launch_args)`` — as of
        #9-KEY v4(b) the STATELESS full-copy ``..._tiled_autolen`` variant
        (identical 37-arg signature; identical final arena bytes; replay-safe
        under any compact_pos arena state) — after draining the
        pointer-buffer ready events onto the capture
        stream via ``_prepare_rebuild_ptr_ready_events_for_capture`` so the
        in-capture ptr lookups take the skip-wait branch.
        """
        # Must run under the existing refresh_stream context; never nested inside
        # the outer decode cudagraph capture.
        try:
            if bool(torch.cuda.is_current_stream_capturing()):
                # Already capturing the outer decode graph -> do NOT nest.
                return
        except Exception:
            _log.warning(
                "writer graph: failed to query outer capture state; skip capture",
                exc_info=True,
            )
            return
        state = self._writer_graph_state
        # [POOL-PRIVATE 2026-07-09] each graph owns a PRIVATE mempool (default
        # capture_begin pool). The former shared handle died with its last
        # graph (allocator pool use_count -> 0 on pop/clear/evict) and the
        # next capture_begin on the dead id tripped the CUDACachingAllocator
        # "use_count > 0" INTERNAL ASSERT -> permanent bypass latch (wg2
        # forensics traceback; the §10.7 "known family noise" true face).
        # The captured region allocates nothing, so a private pool holds no
        # memory — sharing bought nothing and cost the lifetime coupling.
        graph = torch.cuda.CUDAGraph()

        # #9-KEY v4(b): the captured/replayed kernel must be STATELESS. The
        # skip_unchanged variant reads the compact_pos arena (dst_pos) as its
        # skip-copy baseline; the deferred slot reset (defer_compact_meta_publish
        # -> _reset_compact_slot) mutates that arena between capture and replay,
        # so a replayed skip-kernel skips copies it must not skip (the v3
        # deterministic ROW1 divergence). The plain ``_tiled_autolen`` variant
        # has the IDENTICAL 37-arg public signature (impl skip flag false =
        # unconditional copy): its output is a pure function of the launch
        # inputs -> replay-safe under ANY arena state. Eager launches keep the
        # skip_unchanged variant (the eager-order bandwidth saver).
        if not hasattr(ext, "gather_compact_kv_into_arena_ptrs_tiled_autolen"):
            # Prebuilt ext without the stateless symbol: never capture (fail-open).
            self._writer_graph_state = {"bypass": True}
            return

        def _replay_body() -> None:
            ext.gather_compact_kv_into_arena_ptrs_tiled_autolen(
                *launch_args
            )

        # Defects #1,#2: drain ptr ready-events on the capture stream and latch
        # the skip-wait set so the in-capture ptr lookup does NOT wait_event.
        self._prepare_rebuild_ptr_ready_events_for_capture(device=device)
        try:
            # [ASYNC-CAPTURE 2026-07-07 B-1] 手动 capture_begin/end 替代
            # ``with torch.cuda.graph(...)``:其 __enter__ 做全设备
            # torch.cuda.synchronize()+empty_cache()(~26ms 级全机停顿,decode
            # 主流被迫排空;empty_cache 还把 allocator 缓存段清光,后续两侧分配
            # 重新 cudaMalloc)。捕获正确性只需捕获流自身空闲——收窄为
            # refresh_stream.synchronize();无需 empty_cache。捕获内容与 with
            # 版逐字节相同。pool 不传=graph 私有([POOL-PRIVATE] 见上)。
            self.refresh_stream.synchronize()
            with torch.cuda.stream(self.refresh_stream):
                graph.capture_begin()
                try:
                    _replay_body()
                finally:
                    graph.capture_end()
        finally:
            self._clear_rebuild_ptr_capture_wait_satisfied()
        # #9-KEY v5: per-key cache insert (state may be None / legacy shape).
        key_t = tuple(int(v) for v in key)
        if not isinstance(state, dict) or not isinstance(state.get("graphs"), dict):
            state = {"graphs": {}, "bypass": False}
            self._writer_graph_state = state
        graphs_map = state["graphs"]
        # Defensive bound: beyond any plausible (layer-group x batch-shape)
        # population the keys are churning; drop the stale set BEFORE the
        # insert (keep the just-captured graph) and let the thrash detector
        # adjudicate a bypass.
        if len(graphs_map) >= 8 and key_t not in graphs_map:
            graphs_map.clear()
            # Thrash joint-verdict signal: unbounded churn is the only way to
            # overflow this cache (bounded families never reach 8 live keys).
            self._writer_graph_evict_clear_count += 1
        graphs_map[key_t] = graph
        self._record_deadline_async_producer_count("graph_capture")
        _wg_dbg = os.environ.get("VLLM_SPARSE_SELECTOR_TOPK_GRAPH_DEBUG_LOG", "")
        if _wg_dbg:
            # churn 取证:writer 轨与 selector 轨共用一个 keys 落盘(诊断档,
            # 默认关;98 捕/0 replay 与 selector 轨同病,同仪器定名)。
            try:
                with open(_wg_dbg, "a") as _fh:
                    _fh.write(f"{os.getpid()}\twriter\t{key_t}\n")
            except OSError:
                pass
        if self._writer_graph_record_recapture():
            self._writer_graph_state = {"bypass": True}

    def _run_pending_refresh_rebuild_compact_writer(
        self,
        pending: PendingRefreshRebuild,
        result: "SelectorResult",
    ) -> None:
        defer_compact_meta_publish = bool(
            getattr(pending, "compact_meta_defer_publish", False)
        )
        pending.compact_meta_defer_publish = False
        compact_meta_commit_log: Optional[List[Dict[str, object]]] = (
            [] if defer_compact_meta_publish else None
        )
        fused_ok = self._rebuild_compact_slots_batched_layers_from_selection(
            pending.payloads,
            result.selected_indices,
            phase=pending.rebuild_phase,
            bootstrap_slots_by_layer=pending.bootstrap_slots_by_layer,
            defer_compact_meta_publish=bool(defer_compact_meta_publish),
            compact_meta_commit_log=compact_meta_commit_log,
        )
        if not fused_ok:
            pending.compact_meta_commit_log = None
            raise RuntimeError("pending refresh compact rebuild: fused gather failed")
        if defer_compact_meta_publish:
            pending.compact_meta_commit_log = compact_meta_commit_log


    def _mark_pending_refresh_rebuild_accepted(
        self,
        pending: PendingRefreshRebuild,
    ) -> None:
        _mark_pending_selected_scope_terminal(
            pending,
            status="accept",
        )

    # ------------------------------------------------------------------
    def _record_async_refresh_work(
        self,
        *,
        device: torch.device,
        buf_ids: Sequence[int],
        body: Callable[[], bool],
        release_after_handle_id: int = -1,
    ) -> Optional[Any]:
        """Run ``body()`` on the refresh stream with ready/done event bookends.

        Returns ``self.chunk_done_evt[buf_ids[-1]]`` when ``body`` returned
        ``True`` (writer actually launched), else ``None``.
        """
        main_stream = torch.cuda.current_stream(device=device)
        for buf in buf_ids:
            self.chunk_ready_evt[buf].record(main_stream)
        release_event = self._refresh_producer_stream_release_event(
            device=device,
            release_after_handle_id=int(release_after_handle_id),
        )
        return self._record_async_refresh_work_after_ready_event(
            device=device,
            buf_ids=buf_ids,
            ready_event=None,
            release_event=release_event,
            body=body,
        )

    def _record_async_refresh_work_after_ready_event(
        self,
        *,
        device: torch.device,
        buf_ids: Sequence[int],
        ready_event: Optional[Any],
        body: Callable[[], bool],
        release_event: Optional[Any] = None,
    ) -> Optional[Any]:
        """Run ``body()`` on refresh stream after a private ready event.

        ``ready_event`` is used by the host-worker path so the decode thread
        can record readiness once and then leave producer launch to a worker
        without racing the reusable chunk_ready_evt ring.
        """
        if self.refresh_stream is None:
            raise RuntimeError("async refresh producer requires refresh_stream")
        with torch.cuda.device(device):
            with torch.cuda.stream(self.refresh_stream):
                cur_stream = torch.cuda.current_stream(device=device)
                if ready_event is not None:
                    cur_stream.wait_event(ready_event)
                else:
                    for buf in buf_ids:
                        cur_stream.wait_event(self.chunk_ready_evt[buf])
                if release_event is not None:
                    cur_stream.wait_event(release_event)
                with torch.inference_mode():
                    if not body():
                        return None
                for buf in buf_ids:
                    self.refresh_done_evt[buf].record(cur_stream)
                    self.chunk_done_evt[buf].record(cur_stream)
        return self.chunk_done_evt[buf_ids[-1]] if buf_ids else None

    def _record_async_refresh_selector_work(
        self,
        *,
        device: torch.device,
        buf_ids: Sequence[int],
        body: Callable[[], bool],
    ) -> bool:
        """Run selector-only producer work on refresh_stream.

        Split writer release cannot be expressed by waiting on a future CUDA
        event.  Selector-only work intentionally leaves chunk_done unrecorded;
        the release boundary submits the writer and records writer_done_event.
        """
        if self.refresh_stream is None:
            raise RuntimeError("async refresh selector requires refresh_stream")
        main_stream = torch.cuda.current_stream(device=device)
        for buf in buf_ids:
            self.chunk_ready_evt[buf].record(main_stream)
        with torch.cuda.device(device):
            with torch.cuda.stream(self.refresh_stream):
                cur_stream = torch.cuda.current_stream(device=device)
                for buf in buf_ids:
                    cur_stream.wait_event(self.chunk_ready_evt[buf])
                with torch.inference_mode():
                    return bool(body())

    def _begin_pending_refresh_grouped_async_envelope(self) -> bool:
        if not _REFRESH_GROUPED_ASYNC_ENVELOPE_CACHED:
            return False
        if self._pending_refresh_grouped_async_records is not None:
            raise RuntimeError("nested grouped async refresh envelope is not supported")
        self._pending_refresh_grouped_async_records = []
        return True

    def _finish_pending_refresh_grouped_async_envelope(self) -> int:
        records = self._pending_refresh_grouped_async_records
        self._pending_refresh_grouped_async_records = None
        if records is None or not records:
            return 0
        submitted = self._record_grouped_async_refresh_work(records)
        for record in tuple(records):
            pending = record.get("pending")
            if getattr(pending, "writer_done_event", None) is None and not bool(
                getattr(pending, "selector_done_event_recorded", False)
            ):
                continue
            post_submit = record.get("post_submit")
            if callable(post_submit):
                post_submit()
        return int(submitted)

    def _abort_pending_refresh_grouped_async_envelope(self) -> None:
        records = self._pending_refresh_grouped_async_records
        self._pending_refresh_grouped_async_records = None
        if records is None:
            return
        for record in reversed(tuple(records)):
            abort = record["abort"]
            abort("pending_rebuild_grouped_async_envelope_aborted")

    def _record_grouped_async_refresh_work(
        self,
        records: Sequence[Dict[str, Any]],
    ) -> int:
        """Record several replay-refresh bodies inside one stream envelope.

        Each pending keeps its own writer_done_event.  The grouping only removes
        repeated host-side stream wrappers; selector/writer bodies and pending
        publication boundaries remain independent.
        """
        if self.refresh_stream is None:
            raise RuntimeError("async refresh producer requires refresh_stream")
        record_list = tuple(records)

        def _abort_record(record: Dict[str, Any], *, reason: str) -> None:
            release_event = record.get("release_event")
            if release_event is not None and not bool(
                record.get("release_event_waited", False)
            ):
                self._forget_refresh_producer_stream_release_event(release_event)
            abort = record["abort"]
            abort(str(reason))

        by_device: Dict[str, List[Dict[str, Any]]] = {}
        for record in record_list:
            device = record["device"]
            by_device.setdefault(str(device), []).append(record)

        submitted = 0
        for device_records in by_device.values():
            submitted_ids: Set[int] = set()
            aborted_ids: Set[int] = set()
            try:
                device = device_records[0]["device"]
                main_stream = torch.cuda.current_stream(device=device)
                done_buf_counts: Dict[int, int] = {}
                for record in device_records:
                    record_buf_ids = tuple(int(v) for v in record["buf_ids"])
                    done_buf = int(record_buf_ids[-1])
                    done_buf_counts[done_buf] = (
                        int(done_buf_counts.get(done_buf, 0)) + 1
                    )
                for record in device_records:
                    for buf in tuple(record["buf_ids"]):
                        self.chunk_ready_evt[int(buf)].record(main_stream)
                    release_after_handle_id = int(record["release_after_handle_id"])
                    record["release_event_waited"] = False
                    record["release_event"] = self._refresh_producer_stream_release_event(
                        device=device,
                        release_after_handle_id=release_after_handle_id,
                    )

                with torch.cuda.device(device):
                    with torch.cuda.stream(self.refresh_stream), torch.inference_mode():
                        cur_stream = torch.cuda.current_stream(device=device)
                        for record_index, record in enumerate(device_records):
                            buf_ids = tuple(int(v) for v in record["buf_ids"])
                            for buf in buf_ids:
                                cur_stream.wait_event(self.chunk_ready_evt[int(buf)])
                            release_event = record.get("release_event")
                            if release_event is not None:
                                cur_stream.wait_event(release_event)
                                record["release_event_waited"] = True
                            body = record["body"]
                            launched = bool(body())
                            if not launched:
                                _abort_record(
                                    record,
                                    reason="pending_rebuild_async_enqueue_not_launched",
                                )
                                aborted_ids.add(id(record))
                                continue
                            pending = record["pending"]
                            if bool(record.get("selector_only", False)):
                                submitted_ids.add(id(record))
                                submitted += 1
                                continue
                            for buf in buf_ids:
                                self.refresh_done_evt[int(buf)].record(cur_stream)
                                self.chunk_done_evt[int(buf)].record(cur_stream)
                            done_buf = int(buf_ids[-1])
                            if int(done_buf_counts[done_buf]) <= 1:
                                done_event = self.chunk_done_evt[done_buf]
                            else:
                                done_event = torch.cuda.Event(enable_timing=False)
                                done_event.record(cur_stream)
                            pending.writer_done_event = done_event
                            submitted_ids.add(id(record))
                            submitted += 1
            except Exception:
                for record in reversed(device_records):
                    record_id = id(record)
                    if record_id in submitted_ids or record_id in aborted_ids:
                        continue
                    _abort_record(
                        record,
                        reason="pending_rebuild_async_enqueue_failed",
                    )
                raise
        return int(submitted)

    def _refresh_producer_stream_release_event(
        self,
        *,
        device: torch.device,
        release_after_handle_id: int,
    ) -> Optional[Any]:
        release_after_handle_id = int(release_after_handle_id)
        if release_after_handle_id <= 0:
            return None
        event = torch.cuda.Event(enable_timing=False)
        events = self._refresh_producer_stream_release_events
        events.append((release_after_handle_id, event, device))
        self._refresh_producer_stream_release_pending = True
        self._refresh_producer_stream_release_generation = (
            int(self._refresh_producer_stream_release_generation) + 1
        )
        return event

    def _forget_refresh_producer_stream_release_event(self, event: object) -> None:
        events = self._refresh_producer_stream_release_events
        kept = [entry for entry in events if entry[1] is not event]
        if len(kept) == len(events):
            raise RuntimeError("refresh producer release event was not registered")
        self._refresh_producer_stream_release_events = kept
        split_counts = self._refresh_producer_split_release_counts_by_handle
        self._refresh_producer_stream_release_pending = bool(kept) or bool(
            split_counts
        )
        self._refresh_producer_stream_release_generation = (
            int(self._refresh_producer_stream_release_generation) + 1
        )

    def _restore_pending_refresh_rebuild_buf_state(
        self,
        prior_buf_state: Dict[int, Tuple[int, int]],
    ) -> None:
        flags = self._buf_pending_work_flags
        epochs = self._buf_pending_work_epoch
        for buf, (prior_flags, prior_epoch) in prior_buf_state.items():
            b = int(buf) % len(flags)
            flags[b] = int(prior_flags)
            epochs[b] = int(prior_epoch)

    def _abort_pending_refresh_rebuild_async_enqueue(
        self,
        pending: PendingRefreshRebuild,
        *,
        prior_buf_state: Dict[int, Tuple[int, int]],
        reason: str,
        req_mapping_snapshot: Optional[Dict[Tuple[str, int, int], Optional[int]]] = None,
    ) -> None:
        self._pending_refresh_rebuild_remove_from_queue(pending)
        self._pending_refresh_rebuild_clear(pending)
        if req_mapping_snapshot is not None:
            self._pending_refresh_rebuild_restore_req_mapping(
                req_mapping_snapshot,
                current_pending_id=int(getattr(pending, "pending_id", -1)),
            )
        drop_req_ids = self._pending_refresh_rebuild_drop_req_ids(pending)
        if drop_req_ids:
            self._resolve_refresh_lease(
                req_ids=drop_req_ids,
                reason=str(reason),
            )
        self._restore_pending_refresh_rebuild_buf_state(prior_buf_state)

    def _refresh_producer_should_split_selector_writer_release(
        self,
        *,
        release_after_handle_id: int,
        target_layer_start: int,
    ) -> bool:
        release_after_handle_id = int(release_after_handle_id)
        if (
            release_after_handle_id <= 0
            or not _REFRESH_PRODUCER_SPLIT_SELECTOR_WRITER_RELEASE_CACHED
        ):
            return False
        if not _REFRESH_PRODUCER_SPLIT_SELECTOR_WRITER_RELEASE_AUTO_CACHED:
            if self._refresh_profile_enabled():
                self._record_deadline_async_producer_count("split_release_forced")
            return True
        target_layer_start = int(target_layer_start)
        min_layer_start = int(
            _REFRESH_PRODUCER_SPLIT_SELECTOR_WRITER_RELEASE_MIN_LAYER_START_CACHED
        )
        if min_layer_start > 0 and (
            target_layer_start < 0 or target_layer_start < min_layer_start
        ):
            if self._refresh_profile_enabled():
                self._record_deadline_async_producer_count(
                    "split_release_adaptive_layer_gated"
                )
            return False
        counts = self._refresh_producer_split_release_counts_by_handle
        count = int(counts.get(release_after_handle_id, 0))
        max_per_handle = int(
            _REFRESH_PRODUCER_SPLIT_SELECTOR_WRITER_RELEASE_MAX_PER_HANDLE_CACHED
        )
        if count >= max_per_handle:
            if self._refresh_profile_enabled():
                self._record_deadline_async_producer_count(
                    "split_release_adaptive_gated"
                )
            return False
        counts[release_after_handle_id] = count + 1
        self._refresh_producer_stream_release_pending = True
        self._refresh_producer_stream_release_generation = (
            int(self._refresh_producer_stream_release_generation) + 1
        )
        if self._refresh_profile_enabled():
            self._record_deadline_async_producer_count("split_release_adaptive")
        return True

    def _refresh_producer_register_writer_release(
        self,
        *,
        release_after_handle_id: int,
    ) -> None:
        release_after_handle_id = int(release_after_handle_id)
        if release_after_handle_id <= 0:
            return
        counts = self._refresh_producer_split_release_counts_by_handle
        counts[release_after_handle_id] = int(counts.get(release_after_handle_id, 0)) + 1
        self._refresh_producer_stream_release_pending = True
        self._refresh_producer_stream_release_generation = (
            int(self._refresh_producer_stream_release_generation) + 1
        )

    def _forget_refresh_producer_writer_release(
        self,
        *,
        release_after_handle_id: int,
    ) -> None:
        release_after_handle_id = int(release_after_handle_id)
        if release_after_handle_id <= 0:
            return
        counts = self._refresh_producer_split_release_counts_by_handle
        count = int(counts.get(release_after_handle_id, 0) or 0)
        if count <= 1:
            counts.pop(release_after_handle_id, None)
        else:
            counts[release_after_handle_id] = count - 1
        self._refresh_producer_stream_release_pending = bool(
            self._refresh_producer_stream_release_events
        ) or bool(counts)
        self._refresh_producer_stream_release_generation = (
            int(self._refresh_producer_stream_release_generation) + 1
        )

    def _submit_due_selector_prepared_refresh_writers(
        self,
        *,
        handle_id: int,
        force_req_ids: tuple[str, ...] = (),
    ) -> int:
        handle_id = int(handle_id)
        if handle_id <= 0 or not self._pending_refresh_rebuilds:
            return 0
        force_req_id_set = {
            str(req_id) for req_id in tuple(force_req_ids or tuple())
        }
        submit_debug = getattr(
            self,
            "_mixed_page_full_cudagraph_last_selector_writer_submit_debug",
            None,
        )

        def _base_submit_debug_record(pending: object) -> dict[str, object]:
            def _int_debug_attr(name: str, default: int = -1) -> int:
                value = getattr(pending, name, default)
                if value is None:
                    return int(default)
                try:
                    return int(value)
                except (TypeError, ValueError):
                    return int(default)

            pending_req_ids = [
                str(req_id)
                for req_id in tuple(getattr(pending, "req_ids", tuple()) or tuple())
            ]
            release_after = int(
                _int_debug_attr("writer_release_after_handle_id")
            )
            payloads = tuple(getattr(pending, "payloads", tuple()) or tuple())
            return {
                "pending_id": int(_int_debug_attr("pending_id")),
                "producer_kind": str(getattr(pending, "producer_kind", "")),
                "req_ids": pending_req_ids,
                "handle_id": int(handle_id),
                "release_after_handle_id": int(release_after),
                "force_release": bool(
                    force_req_id_set
                    and any(req_id in force_req_id_set for req_id in pending_req_ids)
                ),
                "selector_done_event_recorded": bool(
                    getattr(pending, "selector_done_event_recorded", False)
                ),
                "has_selector_done_event": getattr(
                    pending, "selector_done_event", None
                )
                is not None,
                "has_writer_done_event": getattr(
                    pending, "writer_done_event", None
                )
                is not None,
                "has_result": getattr(pending, "result", None) is not None,
                "payload_count": int(len(payloads)),
                "target_layer_start": int(_int_debug_attr("target_layer_start")),
                "target_layer_end": int(_int_debug_attr("target_layer_end")),
                "deadline_handle_id": int(_int_debug_attr("deadline_handle_id")),
                "deadline_epoch": int(_int_debug_attr("deadline_epoch")),
            }

        def _append_submit_debug(
            record: dict[str, object] | None,
            decision: str,
            **extra: object,
        ) -> None:
            if not isinstance(submit_debug, list) or record is None:
                return
            payload = {"decision": str(decision), **record}
            payload.update(extra)
            submit_debug.append(payload)

        submitted = 0
        for pending in tuple(self._pending_refresh_rebuilds):
            release_after = int(
                getattr(pending, "writer_release_after_handle_id", -1) or -1
            )
            # [DEBUG-DICT-GATE] 15 字段 debug dict 只在 profile/trace 收集时构造
            # （submit_debug 为 list）；生产每 pending 每步不再白付。
            debug_record = (
                _base_submit_debug_record(pending)
                if isinstance(submit_debug, list)
                else None
            )
            force_release = False
            if force_req_id_set:
                pending_req_ids = tuple(
                    str(req_id)
                    for req_id in tuple(getattr(pending, "req_ids", tuple()) or tuple())
                )
                force_release = any(
                    req_id in force_req_id_set for req_id in pending_req_ids
                )
            if release_after <= 0 or (release_after > handle_id and not force_release):
                _append_submit_debug(
                    debug_record,
                    (
                        "skip_no_release_handle"
                        if release_after <= 0
                        else "skip_release_not_due"
                    ),
                )
                continue
            if getattr(pending, "writer_done_event", None) is not None:
                _append_submit_debug(debug_record, "forget_writer_done_event")
                self._pending_refresh_rebuild_forget_writer_release(pending)
                continue
            if not getattr(pending, "selector_done_event_recorded", False):
                _append_submit_debug(debug_record, "skip_selector_not_recorded")
                continue
            if not self._pending_refresh_rebuild_is_latest(pending):
                _append_submit_debug(debug_record, "drop_non_latest", is_latest=False)
                self._drop_pending_refresh_rebuild(
                    pending,
                    status="drop_non_latest",
                    lease_reason="pending_rebuild_drop_non_latest_selector_writer_release",
                    remaining_pending=tuple(
                        item
                        for item in self._pending_refresh_rebuilds
                        if item is not pending
                    ),
                    wait_recorded_work=False,
                )
                self._pending_refresh_rebuild_remove_from_queue(pending)
                continue
            if pending.result is None:
                _append_submit_debug(debug_record, "error_missing_result", is_latest=True)
                raise RuntimeError(
                    "selector-prepared refresh writer release requires selection result"
                )
            if not pending.payloads:
                _append_submit_debug(debug_record, "error_missing_payloads", is_latest=True)
                raise RuntimeError(
                    "selector-prepared refresh writer release requires payloads"
                )
            device = pending.payloads[0].key_cache.device
            writer_event = self._launch_selector_prepared_pending_refresh_writer(
                pending,
                device=device,
            )
            if writer_event is None:
                if self._wait_pending_refresh_rebuild_selector_done(pending):
                    self._pending_refresh_rebuild_clear_completed_buf_work(pending)
                self._pending_refresh_rebuild_remove_from_queue(pending)
                self._pending_refresh_rebuild_clear(pending)
                _append_submit_debug(
                    debug_record,
                    "writer_completed_inline",
                    is_latest=True,
                )
                continue
            pending.writer_done_event = writer_event
            self._pending_refresh_rebuild_forget_writer_release(pending)
            _append_submit_debug(debug_record, "launch_writer", is_latest=True)
            submitted += 1
        return int(submitted)

    def _has_pending_selector_prepared_writer_release(self) -> bool:
        return any(
            int(getattr(pending, "writer_release_after_handle_id", -1) or -1) > 0
            and getattr(pending, "writer_done_event", None) is None
            for pending in tuple(self._pending_refresh_rebuilds)
        )

    def _record_due_refresh_producer_stream_release_events(
        self,
        *,
        handle_id: int,
    ) -> None:
        handle_id = int(handle_id)
        if handle_id <= 0:
            return
        split_counts = self._refresh_producer_split_release_counts_by_handle
        if split_counts:
            for release_after_handle_id in tuple(split_counts):
                if int(release_after_handle_id) <= handle_id:
                    split_counts.pop(release_after_handle_id, None)
        self._submit_due_selector_prepared_refresh_writers(handle_id=handle_id)
        writer_release_pending = self._has_pending_selector_prepared_writer_release()
        events = self._refresh_producer_stream_release_events
        if not events:
            self._refresh_producer_stream_release_pending = bool(split_counts) or bool(
                writer_release_pending
            )
            return
        kept: List[Tuple[int, Any, Any]] = []
        for release_after_handle_id, event, device in events:
            if int(release_after_handle_id) <= handle_id:
                cur_stream = torch.cuda.current_stream(device=device)
                event.record(cur_stream)
            else:
                kept.append((int(release_after_handle_id), event, device))
        self._refresh_producer_stream_release_events = kept
        self._refresh_producer_stream_release_pending = (
            bool(kept) or bool(split_counts) or bool(writer_release_pending)
        )

    def _refresh_producer_current_handle_id(self) -> int:
        step_authority = getattr(self, "step_authority", None)
        if step_authority is None:
            step_context = getattr(self, "step_context", None)
            step_authority = getattr(step_context, "step_authority", None)
        else:
            step_context = getattr(self, "step_context", None)
        handle_id = int(
            getattr(
                step_authority,
                "step_handle_id",
                getattr(step_context, "step_handle_id", -1),
            )
            or -1
        )
        return handle_id

    def _release_refresh_producer_after_decode(self) -> None:
        handle_id = self._refresh_producer_current_handle_id()
        generation = int(self._refresh_producer_stream_release_generation)
        if (
            int(self._refresh_producer_stream_release_checked_handle_id)
            == int(handle_id)
            and int(self._refresh_producer_stream_release_checked_generation) == generation
        ):
            return
        self._record_due_refresh_producer_stream_release_events(handle_id=handle_id)
        self._refresh_producer_stream_release_checked_handle_id = int(handle_id)
        self._refresh_producer_stream_release_checked_generation = int(generation)

    def _pending_refresh_rebuild_buf_ids(
        self,
        pending: PendingRefreshRebuild,
    ) -> Tuple[int, ...]:
        raw_buf_ids = getattr(pending, "buf_ids", tuple()) or tuple()
        resolved: List[int] = []
        for raw in raw_buf_ids:
            try:
                buf = int(raw) % _CAPTURE_IN_FLIGHT
            except (TypeError, ValueError):
                continue
            if buf not in resolved:
                resolved.append(buf)
        if not resolved:
            resolved.append(int(pending.buf_id) % _CAPTURE_IN_FLIGHT)
        return tuple(resolved)

    def _launch_selector_prepared_pending_refresh_writer(
        self,
        pending: PendingRefreshRebuild,
        *,
        device: torch.device,
    ) -> Optional[Any]:
        """Launch writer for selector-future work without republishing selector state."""
        if not pending.payloads or pending.result is None:
            return None
        if not _pending_refresh_rebuild_matches_current_target_scope(self, pending):
            _mark_pending_selected_scope_terminal(
                pending,
                status="drop_stale",
            )
            drop_req_ids = self._pending_refresh_rebuild_drop_req_ids(pending)
            if drop_req_ids:
                self._resolve_refresh_lease(
                    req_ids=drop_req_ids,
                    reason="pending_rebuild_target_scope_stale",
                )
            return None

        self._ensure_refresh_stream(device)
        do_async = (
            self.refresh_stream is not None
            and self.chunk_ready_evt
            and self.chunk_done_evt
            and self._async_refresh_enabled()
        )
        pending_buf_ids = self._pending_refresh_rebuild_buf_ids(pending)

        def _run_writer_with_profile() -> bool:
            if pending.result is None:
                return False
            profile_enabled = bool(self._refresh_profile_enabled())
            writer_t0_ns = time.perf_counter_ns() if profile_enabled else 0
            try:
                pending.compact_meta_defer_publish = bool(do_async)
                self._run_pending_refresh_rebuild_compact_writer(
                    pending,
                    pending.result,
                )
            finally:
                if profile_enabled:
                    self._record_deadline_async_producer_stage(
                        "writer",
                        elapsed_cpu_us=(
                            time.perf_counter_ns() - writer_t0_ns
                        )
                        / 1000.0,
                    )
            return True

        if not do_async:
            for buf in pending_buf_ids:
                self._pending_work_mark_submitted(
                    buf_id=buf,
                    kind="refresh",
                    async_mode=False,
                    epoch=self.step_context_epoch,
                )
            _run_writer_with_profile()
            done_event = torch.cuda.Event()
            done_event.record(torch.cuda.current_stream(device=device))
            return done_event

        for buf in pending_buf_ids:
            self._pending_work_mark_submitted(
                buf_id=buf,
                kind="refresh",
                async_mode=True,
                epoch=self.step_context_epoch,
            )

        def _writer_body() -> bool:
            return _run_writer_with_profile()

        return self._record_async_refresh_work(
            device=device,
            buf_ids=pending_buf_ids,
            body=_writer_body,
        )

    def _run_pending_refresh_rebuild_body(
        self,
        pending: PendingRefreshRebuild,
    ) -> None:
        payloads = pending.payloads
        if not payloads:
            return
        if not _pending_refresh_rebuild_matches_current_target_scope(self, pending):
            _mark_pending_selected_scope_terminal(
                pending,
                status="drop_stale",
            )
            drop_req_ids = self._pending_refresh_rebuild_drop_req_ids(pending)
            if drop_req_ids:
                self._resolve_refresh_lease(
                    req_ids=drop_req_ids,
                    reason="pending_rebuild_target_scope_stale",
                )
            return

        result = self._resolve_pending_refresh_rebuild_result(pending)
        if result is None:
            return
        self._publish_pending_refresh_rebuild_selection_tracking(pending, result)
        pending.tracking_published = True
        self._run_pending_refresh_rebuild_compact_writer(pending, result)
        self._mark_pending_refresh_rebuild_accepted(pending)

    def _submit_pending_refresh_rebuild_batch(
        self,
        pending_items: Sequence[PendingRefreshRebuild],
    ) -> None:
        if (
            type(self)._submit_pending_refresh_rebuild
            is not RefreshRebuildMixin._submit_pending_refresh_rebuild
        ):
            for pending in pending_items:
                self._submit_pending_refresh_rebuild(pending)
            return
        # Off-loop pre-publish defense (spec 2026-05-10): skip any pending
        # whose writer_done_event was already recorded at enqueue time. Such
        # pendings must be drained via the wait-only branch in
        # _pending_refresh_rebuild_pre_consume_drain, never relaunched here.
        pendings = [
            pending
            for pending in pending_items
            if pending.payloads and pending.writer_done_event is None
        ]
        if not pendings:
            return
        first = pendings[0].payloads[0]
        device = first.key_cache.device
        for pending in pendings[1:]:
            cur_payload = pending.payloads[0]
            if cur_payload.key_cache.device != device:
                for item in pendings:
                    self._submit_pending_refresh_rebuild(item)
                return
        self._ensure_refresh_stream(device)
        do_async = (
            self.refresh_stream is not None
            and self.chunk_ready_evt
            and self.chunk_done_evt
            and self._async_refresh_enabled()
        )
        if do_async and _is_stream_capturing_or_raise(
            stage="pending_refresh_rebuild_submit_batch"
        ):
            do_async = False
        if do_async:
            # This fallback submit path clears pending state at the call site and
            # has no per-pending writer_done_event publication boundary. Keep it
            # synchronous; the normal off-loop enqueue path remains async.
            do_async = False

        buf_ids: List[int] = []
        for pending in pendings:
            for buf in self._pending_refresh_rebuild_buf_ids(pending):
                buf_ids.append(int(buf))
                self._pending_work_mark_submitted(
                    buf_id=buf,
                    kind="refresh",
                    async_mode=do_async,
                    epoch=self.step_context_epoch,
                )
        unique_buf_ids = tuple(sorted(set(buf_ids)))
        if do_async:
            def _run_all_bodies() -> bool:
                for pending in pendings:
                    self._run_pending_refresh_rebuild_body(pending)
                return True
            self._record_async_refresh_work(
                device=device,
                buf_ids=unique_buf_ids,
                body=_run_all_bodies,
            )
        else:
            for pending in pendings:
                self._run_pending_refresh_rebuild_body(pending)

    def _submit_pending_refresh_rebuild(self, pending: PendingRefreshRebuild) -> None:
        payloads = pending.payloads
        if not payloads:
            return
        first = payloads[0]
        device = first.key_cache.device
        self._ensure_refresh_stream(device)
        do_async = (
            self.refresh_stream is not None
            and self.chunk_ready_evt
            and self.chunk_done_evt
            and self._async_refresh_enabled()
        )
        if do_async and _is_stream_capturing_or_raise(stage="pending_refresh_rebuild_submit"):
            do_async = False
        if do_async:
            # This fallback submit path does not retain a writer_done_event on
            # the pending item, so async execution would expose metadata before
            # the lifecycle commit boundary.
            do_async = False

        pending_buf_ids = self._pending_refresh_rebuild_buf_ids(pending)
        for buf in pending_buf_ids:
            self._pending_work_mark_submitted(
                buf_id=buf,
                kind="refresh",
                async_mode=do_async,
                epoch=self.step_context_epoch,
            )
        if do_async:
            def _run_one_body() -> bool:
                self._run_pending_refresh_rebuild_body(pending)
                return True
            self._record_async_refresh_work(
                device=device,
                buf_ids=pending_buf_ids,
                body=_run_one_body,
            )
        else:
            self._run_pending_refresh_rebuild_body(pending)

    # ------------------------------------------------------------------
    # Enqueue pending refresh rebuild
    # ------------------------------------------------------------------

    def _enqueue_pending_refresh_rebuild(
        self,
        *,
        payloads: List[SelectorBatchPayload],
        result: Optional[SelectorResult],
        selection_phase: str,
        rebuild_phase: str,
        bootstrap_slots_by_layer: Optional[List[Set[int]]],
        chunk_id: int,
        buf_id: int,
        profile_accum: Optional[Any] = None,
        producer_kind: str = "refresh_rebuild",
        producer_carrier: Optional[Any] = None,
        stage_profile_detail: Optional[dict[str, object]] = None,
    ) -> None:
        from patches.sparse_types import PendingRefreshRebuild

        detail_profile = (
            stage_profile_detail if isinstance(stage_profile_detail, dict) else None
        )

        def _detail_start_ns() -> int:
            return time.perf_counter_ns() if detail_profile is not None else 0

        def _detail_add_elapsed(key: str, start_ns: int) -> None:
            if detail_profile is None:
                return
            detail_profile[key] = float(detail_profile.get(key, 0.0) or 0.0) + (
                time.perf_counter_ns() - int(start_ns)
            ) / 1000.0

        prepare_start_ns = _detail_start_ns()
        capture_handle_id = -1
        capture_handle_generation = -1
        capture_epoch = -1
        target_selected_scope_key = None
        pending_buf_ids: Tuple[int, ...] = tuple()
        req_ids: Optional[Tuple[str, ...]] = None
        if producer_carrier is not None:
            if getattr(producer_carrier, "payloads", None) is not payloads:
                raise RuntimeError(
                    "pending refresh rebuild producer carrier payload mismatch"
                )
            capture_handle_id = int(
                getattr(producer_carrier, "capture_handle_id", -1)
            )
            capture_handle_generation = int(
                getattr(producer_carrier, "capture_handle_generation", -1)
            )
            capture_epoch = int(getattr(producer_carrier, "capture_epoch", -1))
            target_selected_scope_key = getattr(
                producer_carrier,
                "target_selected_scope_key",
                None,
            )
            layer_indices = [
                int(v)
                for v in (getattr(producer_carrier, "layer_indices", ()) or ())
            ]
            target_layer_start = int(
                getattr(producer_carrier, "target_layer_start", -1)
            )
            target_layer_end = int(getattr(producer_carrier, "target_layer_end", -1))
            pending_buf_ids = tuple(
                int(v) % _CAPTURE_IN_FLIGHT
                for v in (getattr(producer_carrier, "pending_buf_ids", ()) or ())
            )
            req_ids = tuple(getattr(producer_carrier, "req_ids", tuple()) or tuple())
            if payloads and (
                capture_handle_id <= 0 or capture_handle_generation <= 0
            ):
                raise RuntimeError(
                    "pending refresh rebuild missing capture handle identity: "
                    f"handle_id={capture_handle_id} "
                    f"generation={capture_handle_generation}"
                )
        elif payloads:
            try:
                capture_handle_id = payloads[0].capture_handle_id
                capture_handle_generation = payloads[0].capture_handle_generation
                capture_epoch = payloads[0].capture_epoch
                target_selected_scope_key = getattr(
                    payloads[0], "target_selected_scope_key", None
                )
            except Exception:
                _log.warning("failed to read capture handle identity from payload")
                raise
            if capture_handle_id <= 0 or capture_handle_generation <= 0:
                raise RuntimeError(
                    "pending refresh rebuild missing capture handle identity: "
                    f"handle_id={capture_handle_id} generation={capture_handle_generation}"
                )
            for payload in payloads[1:]:
                if (
                    payload.capture_handle_id != capture_handle_id
                    or payload.capture_handle_generation != capture_handle_generation
                ):
                    raise RuntimeError(
                        "pending refresh rebuild payload handle mismatch across layers"
                    )
                if getattr(payload, "target_selected_scope_key", None) != target_selected_scope_key:
                    raise RuntimeError(
                        "pending refresh rebuild target scope mismatch across payloads"
                    )
        else:
            layer_indices = []
            target_layer_start = -1
            target_layer_end = -1
        if capture_epoch < 0:
            capture_epoch = self.step_context_epoch
        if producer_carrier is None:
            layer_indices = []
            for payload_index, payload in enumerate(payloads):
                layer_index = int(
                    getattr(getattr(payload, "state", None), "layer_index", -1)
                )
                if layer_index < 0:
                    layer_index = int(getattr(payload, "layer_index", -1))
                if layer_index < 0:
                    layer_index = int(payload_index)
                layer_indices.append(layer_index)
            target_layer_start = min(layer_indices) if layer_indices else -1
            target_layer_end = max(layer_indices) if layer_indices else -1
            pending_buf_ids_list: List[int] = []
            map_layer = getattr(self, "_map_global_layer_to_capture_slot", None)
            if callable(map_layer):
                for layer_index in layer_indices:
                    try:
                        _, layer_buf_id, _ = map_layer(int(layer_index))
                        buf = int(layer_buf_id) % _CAPTURE_IN_FLIGHT
                    except Exception:
                        _log.warning(
                            "pending refresh rebuild failed to map layer to capture slot",
                            exc_info=True,
                        )
                        raise
                    if buf not in pending_buf_ids_list:
                        pending_buf_ids_list.append(buf)
            if not pending_buf_ids_list:
                pending_buf_ids_list.append(int(buf_id) % _CAPTURE_IN_FLIGHT)
            pending_buf_ids = tuple(pending_buf_ids_list)
        elif not pending_buf_ids:
            pending_buf_ids = (int(buf_id) % _CAPTURE_IN_FLIGHT,)
        layer_cache_keys = getattr(self, "layer_cache_keys", ())
        if layer_cache_keys:
            num_chunks = (len(layer_cache_keys) + _CAPTURE_CHUNK - 1) // _CAPTURE_CHUNK
        else:
            num_chunks = 1
        max_delay = self._refresh_rebuild_max_delay_steps(num_chunks)
        if producer_carrier is not None:
            carrier_deadline_slack = int(
                getattr(producer_carrier, "deadline_slack_steps", -1) or -1
            )
            if carrier_deadline_slack > 0:
                max_delay = int(carrier_deadline_slack)
        step_ctx_for_handle = getattr(self, "step_context", None)
        step_authority_for_handle = getattr(self, "step_authority", None)
        if step_authority_for_handle is None and step_ctx_for_handle is not None:
            step_authority_for_handle = getattr(
                step_ctx_for_handle, "step_authority", None
            )
        current_handle_id = int(
            getattr(
                step_authority_for_handle,
                "step_handle_id",
                getattr(step_ctx_for_handle, "step_handle_id", -1),
            )
            or -1
        )
        if producer_carrier is None and payloads:
            for payload in payloads:
                # [DETERMINISTIC-REQIDS-SNAPSHOT 2026-07-03] slot_req_ids 已在
                # _enqueue_refresh_capture 提交步物化;此处为 deferred 晚读点,禁
                # 止 fallback 读 live 的 state.batch_request_ids(batch 成员此刻
                # 可能已变,与 payload.slot_list 不再自洽)。缺失即 fail-fast。
                cur_req_ids: Optional[Tuple[str, ...] | Sequence[object]] = (
                    payload.slot_req_ids
                )
                if cur_req_ids is None:
                    raise RuntimeError(
                        "pending refresh rebuild payload missing submission-step "
                        "slot_req_ids (live batch_request_ids fallback retired)"
                    )
                cur_req_ids = self._normalize_refresh_req_ids(cur_req_ids)
                if not cur_req_ids:
                    raise RuntimeError("pending refresh rebuild missing req_ids")
                if req_ids is None:
                    req_ids = cur_req_ids
                    continue
                if cur_req_ids != req_ids:
                    raise RuntimeError(
                        "pending refresh rebuild req_ids mismatch across payloads"
                    )
        pending_id = -1
        req_ids = self._normalize_refresh_req_ids(req_ids)
        if payloads and not req_ids:
            raise RuntimeError("pending refresh rebuild missing req_ids")
        request_states = getattr(self, "request_states", None)
        _detail_add_elapsed("pending_group_enqueue_prepare_us", prepare_start_ns)
        work_item_start_ns = _detail_start_ns()
        producer_work_item = build_refresh_producer_work_item(
            payloads=payloads,
            request_states=request_states if isinstance(request_states, dict) else None,
            req_ids=req_ids,
            layer_indices=layer_indices,
            max_delay_steps=int(max_delay),
            current_epoch=int(self.step_context_epoch),
            current_handle_id=int(current_handle_id),
            admission_reason="pending_refresh_rebuild",
            can_drop=True,
            can_coalesce=True,
        )
        _detail_add_elapsed("pending_group_enqueue_work_item_us", work_item_start_ns)
        self._record_deadline_producer_work_item(producer_work_item)
        if profile_accum is not None:
            producer_work_item.apply_to_profile(profile_accum)
        req_mapping_snapshot: Optional[Dict[Tuple[str, int, int], Optional[int]]] = None
        if req_ids:
            req_mapping_snapshot = self._pending_refresh_rebuild_req_mapping_snapshot(
                req_ids,
                target_layer_start=int(producer_work_item.target_layer_start),
                target_layer_end=int(producer_work_item.target_layer_end),
            )
            register_start_ns = _detail_start_ns()
            pending_id, req_ids = self._pending_refresh_rebuild_register(
                req_ids,
                target_layer_start=int(producer_work_item.target_layer_start),
                target_layer_end=int(producer_work_item.target_layer_end),
            )
            _detail_add_elapsed("pending_group_enqueue_register_us", register_start_ns)
        construct_start_ns = _detail_start_ns()
        pending = PendingRefreshRebuild(
            payloads=payloads,
            result=result,
            selection_phase=selection_phase,
            rebuild_phase=rebuild_phase,
            bootstrap_slots_by_layer=bootstrap_slots_by_layer,
            chunk_id=chunk_id,
            buf_id=buf_id,
            capture_handle_id=capture_handle_id,
            capture_handle_generation=capture_handle_generation,
            buf_ids=pending_buf_ids,
            target_selected_scope_key=target_selected_scope_key,
            capture_epoch=capture_epoch,
            pending_id=pending_id,
            req_ids=req_ids,
            producer_kind=producer_kind,
            producer_work_item=producer_work_item,
            target_layer_start=producer_work_item.target_layer_start,
            target_layer_end=producer_work_item.target_layer_end,
            ready_epoch=producer_work_item.ready_epoch,
            deadline_epoch=producer_work_item.deadline_epoch,
            deadline_handle_id=producer_work_item.deadline_handle_id,
            can_drop=producer_work_item.can_drop,
            can_coalesce=producer_work_item.can_coalesce,
            admission_reason=producer_work_item.admission_reason,
        )
        _detail_add_elapsed("pending_group_enqueue_construct_us", construct_start_ns)
        coalesced_count = 0
        coalesced_count_applied = False

        def _coalesce_superseded_now() -> None:
            nonlocal coalesced_count, coalesced_count_applied
            if coalesced_count_applied:
                return
            coalesced_count_applied = True
            coalesce_start_ns = _detail_start_ns()
            coalesced_count = self._pending_refresh_rebuild_coalesce_superseded(pending)
            _detail_add_elapsed(
                "pending_group_enqueue_coalesce_us",
                coalesce_start_ns,
            )
            if profile_accum is not None:
                profile_accum.refresh_rebuild_coalesced_count += int(coalesced_count)

        bypass_submit_now = False
        coalesce_after_group_submit = False
        if req_ids:
            max_queue = int(_PENDING_REBUILD_MAX_QUEUE_CACHED)
            queue_size = len(self._pending_refresh_rebuilds)
            if max_queue > 0 and queue_size >= max_queue:
                coalesced_count_estimate = self._pending_refresh_rebuild_count_superseded(
                    pending
                )
                queue_size = max(0, queue_size - int(coalesced_count_estimate))
            if max_queue > 0 and queue_size >= max_queue:
                # 压力路径：先压缩陈旧项，避免队列增长导致 EngineDead。
                self._pending_refresh_rebuild_compact_stale()
                coalesced_count_estimate = self._pending_refresh_rebuild_count_superseded(
                    pending
                )
                queue_size = max(
                    0,
                    len(self._pending_refresh_rebuilds)
                    - int(coalesced_count_estimate),
                )
                if queue_size >= max_queue:
                    # 若仍满，旁路当前项：直接提交 rebuild，不再入队。
                    bypass_submit_now = True
        # ===== Off-loop pre-publish (spec 2026-05-10) =====
        # 异步 refresh 启用且不处于 cudagraph capture 时，立即在 refresh
        # stream 上跑 selector + writer，把 writer_done_event 录到 pending；
        # drain 只 wait + publish + accept，不再触发 selector/writer 启动。
        # tracking publish 仍延迟到 drain：它会 mutate request ticket /
        # scheduled_decode_refresh_step，必须在 GPU writer 完成后才能让下
        # 一次 trigger 看到正确状态。
        if (
            payloads
            and not bypass_submit_now
            and self._async_refresh_enabled()
            and not _is_stream_capturing_or_raise(stage="pending_refresh_rebuild_enqueue")
        ):
            device = payloads[0].key_cache.device
            self._ensure_refresh_stream(device)
            if self.refresh_stream is not None:
                record_stream_start_ns = _detail_start_ns()
                self._record_pending_refresh_rebuild_stream_lifetime(
                    pending,
                    self.refresh_stream,
                    payload_tensor_scope=(
                        "trimmed"
                        if str(producer_kind) == "full_cudagraph_replay_refresh"
                        else "all"
                    ),
                    include_result=True,
                )
                _detail_add_elapsed(
                    "pending_group_enqueue_record_stream_us",
                    record_stream_start_ns,
                )
                pending_buf_ids = self._pending_refresh_rebuild_buf_ids(pending)
                pending_buf_prior_state: Dict[int, Tuple[int, int]] = {}
                pending_flags = self._buf_pending_work_flags
                pending_epochs = self._buf_pending_work_epoch
                for raw_buf in pending_buf_ids:
                    buf = int(raw_buf) % len(pending_flags)
                    pending_buf_prior_state[buf] = (
                        int(pending_flags[buf]),
                        int(pending_epochs[buf]),
                    )
                mark_start_ns = _detail_start_ns()
                for buf in pending_buf_ids:
                    self._pending_work_mark_submitted(
                        buf_id=buf,
                        kind="refresh",
                        async_mode=True,
                        epoch=self.step_context_epoch,
                    )
                _detail_add_elapsed("pending_group_enqueue_mark_us", mark_start_ns)

                selector_done_event: Optional[Any] = None
                if (
                    str(producer_kind) == "full_cudagraph_replay_refresh"
                    and int(max_delay) > 1
                ):
                    selector_done_event = torch.cuda.Event(enable_timing=False)
                    pending.selector_done_event = selector_done_event

                def _enqueue_body() -> bool:
                    profile_enabled = bool(self._refresh_profile_enabled())
                    profile_detail_enabled = bool(
                        profile_enabled
                        and self._refresh_profile_detail_enabled()
                        and _ASYNC_PRODUCER_GPU_PROFILE_CACHED
                    )
                    body_t0_ns = time.perf_counter_ns() if profile_enabled else 0
                    if profile_detail_enabled:
                        body_evt_pair = (
                            torch.cuda.Event(enable_timing=True),
                            torch.cuda.Event(enable_timing=True),
                        )
                        body_evt_pair[0].record(torch.cuda.current_stream(device=device))

                        def _append_async_evt_pair(field: str, pair: object) -> None:
                            if pair is None:
                                return
                            try:
                                evt0, evt1 = pair
                            except Exception:
                                return
                            if evt0 is None or evt1 is None:
                                return
                            stage = field
                            if stage.startswith("async_producer_"):
                                stage = stage[len("async_producer_") :]
                            if stage.endswith("_evt_pairs"):
                                stage = stage[: -len("_evt_pairs")]
                            self._record_deadline_async_producer_gpu_event_pair(
                                stage,
                                (evt0, evt1),
                            )
                            if profile_accum is not None:
                                getattr(profile_accum, field).append((evt0, evt1))

                        def _copy_selector_event_pairs(sel: object) -> None:
                            for field, evt0_name, evt1_name in (
                                (
                                    "async_producer_seq_full_evt_pairs",
                                    "profile_seq_full_evt0",
                                    "profile_seq_full_evt1",
                                ),
                                (
                                    "async_producer_pure_preproc_evt_pairs",
                                    "profile_pure_preproc_evt0",
                                    "profile_pure_preproc_evt1",
                                ),
                                (
                                    "async_producer_selector_bounds_evt_pairs",
                                    "profile_selector_bounds_evt0",
                                    "profile_selector_bounds_evt1",
                                ),
                                (
                                    "async_producer_selector_pipeline_evt_pairs",
                                    "profile_selector_pipeline_evt0",
                                    "profile_selector_pipeline_evt1",
                                ),
                                (
                                    "async_producer_key_norms_preproc_evt_pairs",
                                    "profile_key_norms_preproc_evt0",
                                    "profile_key_norms_preproc_evt1",
                                ),
                                (
                                    "async_producer_key_norms_evt_pairs",
                                    "profile_key_norms_evt0",
                                    "profile_key_norms_evt1",
                                ),
                                (
                                    "async_producer_key_norms_h2d_evt_pairs",
                                    "profile_key_norms_h2d_evt0",
                                    "profile_key_norms_h2d_evt1",
                                ),
                                (
                                    "async_producer_key_norms_delta_evt_pairs",
                                    "profile_key_norms_delta_evt0",
                                    "profile_key_norms_delta_evt1",
                                ),
                                (
                                    "async_producer_key_norms_pack_evt_pairs",
                                    "profile_key_norms_pack_evt0",
                                    "profile_key_norms_pack_evt1",
                                ),
                                (
                                    "async_producer_log_s_evt_pairs",
                                    "profile_log_s_evt0",
                                    "profile_log_s_evt1",
                                ),
                                (
                                    "async_producer_topk_evt_pairs",
                                    "profile_topk_evt0",
                                    "profile_topk_evt1",
                                ),
                            ):
                                _append_async_evt_pair(
                                    field,
                                    (
                                        getattr(sel, evt0_name, None),
                                        getattr(sel, evt1_name, None),
                                    ),
                                )
                    else:
                        body_evt_pair = None
                        _append_async_evt_pair = None
                        _copy_selector_event_pairs = None

                    try:
                        if pending.result is not None:
                            self._record_deadline_async_producer_count(
                                "result_precomputed"
                            )
                        selector_t0_ns = (
                            time.perf_counter_ns() if profile_enabled else 0
                        )
                        selector_evt_pair = None
                        if profile_detail_enabled:
                            selector_evt_pair = (
                                torch.cuda.Event(enable_timing=True),
                                torch.cuda.Event(enable_timing=True),
                            )
                            selector_evt_pair[0].record(
                                torch.cuda.current_stream(device=device)
                            )
                        try:
                            selected_indices_out_prev = getattr(
                                self,
                                "_selector_selected_indices_out_override",
                                None,
                            )
                            decode_bounds_prev = getattr(
                                self,
                                "_selector_decode_bounds_buffers_override",
                                None,
                            )
                            key_norms_prev = getattr(
                                self,
                                "_selector_key_norms_all_cache_override",
                                None,
                            )
                            key_norms_valid_prev = getattr(
                                self,
                                "_selector_key_norms_all_valid_cache_keys_override",
                                None,
                            )
                            key_norms_delta_prev = getattr(
                                self,
                                "_selector_key_norms_delta_buffer_override",
                                None,
                            )
                            log_f_workspace_prev = getattr(
                                self,
                                "_log_f_scratch_workspace_override",
                                None,
                            )
                            selector_pipeline_workspace_prev = getattr(
                                self,
                                "_selector_pipeline_workspace_override",
                                None,
                            )
                            self._selector_selected_indices_out_override = {}
                            self._selector_decode_bounds_buffers_override = {}
                            self._selector_key_norms_all_cache_override = {}
                            self._selector_key_norms_all_valid_cache_keys_override = set()
                            self._selector_key_norms_delta_buffer_override = {}
                            self._log_f_scratch_workspace_override = {}
                            self._selector_pipeline_workspace_override = {}
                            sel = self._resolve_pending_refresh_rebuild_result(
                                pending,
                                profile_deferred=bool(
                                    _DEFERRED_SELECTOR_PROFILE_DETAIL_CACHED
                                ),
                            )
                            self._retain_pending_refresh_rebuild_selector_scratch(
                                pending,
                                self.refresh_stream,
                                self._selector_selected_indices_out_override,
                                self._selector_decode_bounds_buffers_override,
                                self._selector_key_norms_all_cache_override,
                                self._selector_key_norms_delta_buffer_override,
                                self._log_f_scratch_workspace_override,
                                self._selector_pipeline_workspace_override,
                            )
                        finally:
                            self._selector_selected_indices_out_override = (
                                selected_indices_out_prev
                            )
                            self._selector_decode_bounds_buffers_override = (
                                decode_bounds_prev
                            )
                            self._selector_key_norms_all_cache_override = (
                                key_norms_prev
                            )
                            self._selector_key_norms_all_valid_cache_keys_override = (
                                key_norms_valid_prev
                            )
                            self._selector_key_norms_delta_buffer_override = (
                                key_norms_delta_prev
                            )
                            self._log_f_scratch_workspace_override = (
                                log_f_workspace_prev
                            )
                            self._selector_pipeline_workspace_override = (
                                selector_pipeline_workspace_prev
                            )
                            if (
                                selector_evt_pair is not None
                                and _append_async_evt_pair is not None
                            ):
                                selector_evt_pair[1].record(
                                    torch.cuda.current_stream(device=device)
                                )
                                _append_async_evt_pair(
                                    "async_producer_selector_evt_pairs",
                                    selector_evt_pair,
                                )
                            if profile_enabled:
                                self._record_deadline_async_producer_stage(
                                    "selector",
                                    elapsed_cpu_us=(
                                        time.perf_counter_ns() - selector_t0_ns
                                    )
                                    / 1000.0,
                                )
                        if sel is None:
                            raise RuntimeError(
                                "off-loop refresh enqueue: selector returned None "
                                f"(pending_id={int(pending.pending_id)}, "
                                f"layers={int(pending.target_layer_start)}-"
                                f"{int(pending.target_layer_end)})"
                            )
                        if selector_done_event is not None:
                            selector_done_event.record(
                                torch.cuda.current_stream(device=device)
                            )
                            pending.selector_done_event_recorded = True
                        self._record_pending_refresh_rebuild_stream_lifetime(
                            pending,
                            self.refresh_stream,
                            payload_tensor_scope="none",
                            include_result=True,
                        )
                        if _copy_selector_event_pairs is not None:
                            _copy_selector_event_pairs(sel)
                        if profile_enabled:
                            self._record_deadline_async_producer_key_norms_delta(sel)
                        if split_writer_release:
                            return True
                        writer_t0_ns = (
                            time.perf_counter_ns() if profile_enabled else 0
                        )
                        writer_evt_pair = None
                        if profile_detail_enabled:
                            writer_evt_pair = (
                                torch.cuda.Event(enable_timing=True),
                                torch.cuda.Event(enable_timing=True),
                            )
                            writer_evt_pair[0].record(
                                torch.cuda.current_stream(device=device)
                            )
                        try:
                            pending.compact_meta_defer_publish = True
                            self._run_pending_refresh_rebuild_compact_writer(
                                pending,
                                sel,
                            )
                        finally:
                            if (
                                writer_evt_pair is not None
                                and _append_async_evt_pair is not None
                            ):
                                writer_evt_pair[1].record(
                                    torch.cuda.current_stream(device=device)
                                )
                                _append_async_evt_pair(
                                    "async_producer_writer_evt_pairs",
                                    writer_evt_pair,
                                )
                            if profile_enabled:
                                self._record_deadline_async_producer_stage(
                                    "writer",
                                    elapsed_cpu_us=(
                                        time.perf_counter_ns() - writer_t0_ns
                                    )
                                    / 1000.0,
                                )
                        return True
                    finally:
                        if body_evt_pair is not None and _append_async_evt_pair is not None:
                            body_evt_pair[1].record(
                                torch.cuda.current_stream(device=device)
                            )
                            _append_async_evt_pair(
                                "async_producer_body_evt_pairs",
                                body_evt_pair,
                            )
                        if profile_enabled:
                            self._record_deadline_async_producer_stage(
                                "body",
                                elapsed_cpu_us=(
                                    time.perf_counter_ns() - body_t0_ns
                                )
                                / 1000.0,
                            )

                release_after_handle_id = -1
                if (
                    str(producer_kind) == "full_cudagraph_replay_refresh"
                    and int(max_delay) > 1
                    and int(current_handle_id) > 0
                ):
                    # [WRITER-RELEASE-STAGGER RETIRED 2026-07-06] A1 错峰旋钮
                    # 负结果已定谳(慢步 9→76 反向;归因量具使命完成),且双代
                    # 侦察证明它反把尾 chunk off-rail 窗拉长——按"不留调试
                    # 废墟"纪律删除。归档:commit 5fd6e8d + staggered-flush 记忆。
                    release_after_handle_id = int(current_handle_id) + 1
                outer_release_after_handle_id = int(release_after_handle_id)
                replay_refresh_requires_split_release = (
                    str(producer_kind) == "full_cudagraph_replay_refresh"
                    and int(release_after_handle_id) > 0
                )
                split_writer_release = False
                if replay_refresh_requires_split_release:
                    self._refresh_producer_register_writer_release(
                        release_after_handle_id=int(release_after_handle_id),
                    )
                    split_writer_release = True
                elif self._refresh_producer_should_split_selector_writer_release(
                        release_after_handle_id=int(release_after_handle_id),
                        target_layer_start=int(
                            getattr(pending, "target_layer_start", -1)
                        ),
                    ):
                    split_writer_release = True
                if split_writer_release:
                    pending.writer_release_after_handle_id = int(release_after_handle_id)
                    outer_release_after_handle_id = -1
                record_async_start_ns = _detail_start_ns()
                grouped_async_records = self._pending_refresh_grouped_async_records
                if (
                    grouped_async_records is not None
                    and str(producer_kind) == "full_cudagraph_replay_refresh"
                ):
                    def _abort_grouped_async_pending(reason: str) -> None:
                        self._abort_pending_refresh_rebuild_async_enqueue(
                            pending,
                            prior_buf_state=pending_buf_prior_state,
                            reason=str(reason),
                            req_mapping_snapshot=req_mapping_snapshot,
                        )

                    grouped_async_records.append(
                        {
                            "pending": pending,
                            "device": device,
                            "buf_ids": tuple(pending_buf_ids),
                            "prior_buf_state": dict(pending_buf_prior_state),
                            "submitted_epoch": int(self.step_context_epoch),
                            "release_after_handle_id": int(
                                outer_release_after_handle_id
                            ),
                            "body": _enqueue_body,
                            "selector_only": bool(split_writer_release),
                            "abort": _abort_grouped_async_pending,
                            "post_submit": _coalesce_superseded_now,
                        }
                    )
                    coalesce_after_group_submit = True
                else:
                    if split_writer_release:
                        try:
                            selector_launched = self._record_async_refresh_selector_work(
                                device=device,
                                buf_ids=pending_buf_ids,
                                body=_enqueue_body,
                            )
                        except Exception:
                            self._abort_pending_refresh_rebuild_async_enqueue(
                                pending,
                                prior_buf_state=pending_buf_prior_state,
                                reason="pending_rebuild_async_enqueue_failed",
                                req_mapping_snapshot=req_mapping_snapshot,
                            )
                            raise
                        if (
                            not selector_launched
                            or not pending.selector_done_event_recorded
                            or pending.result is None
                        ):
                            self._abort_pending_refresh_rebuild_async_enqueue(
                                pending,
                                prior_buf_state=pending_buf_prior_state,
                                reason="pending_rebuild_async_enqueue_not_launched",
                                req_mapping_snapshot=req_mapping_snapshot,
                            )
                            return
                    else:
                        try:
                            pending.writer_done_event = self._record_async_refresh_work(
                                device=device,
                                buf_ids=pending_buf_ids,
                                release_after_handle_id=outer_release_after_handle_id,
                                body=_enqueue_body,
                            )
                        except Exception:
                            self._abort_pending_refresh_rebuild_async_enqueue(
                                pending,
                                prior_buf_state=pending_buf_prior_state,
                                reason="pending_rebuild_async_enqueue_failed",
                                req_mapping_snapshot=req_mapping_snapshot,
                            )
                            raise
                        if pending.writer_done_event is None:
                            self._abort_pending_refresh_rebuild_async_enqueue(
                                pending,
                                prior_buf_state=pending_buf_prior_state,
                                reason="pending_rebuild_async_enqueue_not_launched",
                                req_mapping_snapshot=req_mapping_snapshot,
                            )
                            return
                _detail_add_elapsed(
                    "pending_group_enqueue_record_async_work_us",
                    record_async_start_ns,
                )
        # 释放 selector 阶段的大张量，避免 pending backlog 持有巨量 GPU 显存
        # （仅保留 rebuild 所需的最小字段：slot_list/row_list/seq_lens_cpu/key_cache/value_cache/block_table 等）。
        if pending.result is not None:
            compact_start_ns = _detail_start_ns()
            if profile_accum is not None:
                compact_profile_start_ns = time.perf_counter_ns()
                self._compact_pending_refresh_payloads(payloads)
                profile_accum.refresh_rebuild_compact_cpu_us += (
                    time.perf_counter_ns() - compact_profile_start_ns
                ) / 1000.0
            else:
                self._compact_pending_refresh_payloads(payloads)
            _detail_add_elapsed("pending_group_enqueue_compact_us", compact_start_ns)
        if bypass_submit_now:
            _log.warning(
                "pending refresh rebuild queue overflow: "
                f"size={len(self._pending_refresh_rebuilds)}, "
                f"limit={int(_PENDING_REBUILD_MAX_QUEUE_CACHED)}; "
                "bypass enqueue and submit now"
            )
            _coalesce_superseded_now()
            self._submit_pending_refresh_rebuild(pending)
            self._pending_refresh_rebuild_clear(pending)
            return
        if not coalesce_after_group_submit:
            _coalesce_superseded_now()
        queue_insert_start_ns = _detail_start_ns()
        self._pending_refresh_rebuild_insert_deadline_ordered(pending)
        _detail_add_elapsed(
            "pending_group_enqueue_queue_insert_us",
            queue_insert_start_ns,
        )

    def _compact_pending_refresh_payloads(self, payloads: Sequence[SelectorBatchPayload]) -> None:
        """缩减 pending refresh payload 的占用，避免 backlog 堆积导致显存增长。"""
        if not payloads:
            return
        for payload in payloads:
            device = payload.key_cache.device
            capture_scores = getattr(payload, "capture_scores", None)
            capture_dtype = (
                capture_scores.dtype
                if isinstance(capture_scores, torch.Tensor)
                else torch.float32
            )
            # 大体积 capture tensors（selector 已完成）
            payload.capture_scores = self._pending_refresh_empty_tensor(
                device=device,
                dtype=capture_dtype,
            )
            payload.log_f_denoms = None
            payload.kv_lengths = self._pending_refresh_empty_tensor(
                device=device,
                dtype=torch.long,
            )
            payload.seq_lens_batch = self._pending_refresh_empty_tensor(
                device=device,
                dtype=torch.long,
            )
            payload.seq_lens_batch_i32 = self._pending_refresh_empty_tensor(
                device=device,
                dtype=torch.int32,
            )
            payload.kv_len_per_row_i32 = None
            # Large tensors are dropped before enqueue; small indexing tensors below stay.
            payload.q = None
            payload.cu_seqlens_q = None
            payload.refresh_rows_long = None
            payload.refresh_block_table_sub = None
            payload.refresh_seq_lens_i32 = None
            # (2026-07-03 晚) slot_tensor/row_tensor 系 live 视图字段的值消费者
            # 已全部退休(DETERMINISTIC-SLOT-SOURCE:writer/delta 恒从 host list
            # 构造),按防回潮方针一并清空——留着即诱惑。
            payload.slot_tensor = None
            payload.slot_tensor_i32 = None
            payload.row_tensor = None
            payload.row_tensor_i32 = None

    def _pending_refresh_empty_tensor(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return a cached zero-length tensor carrier for pending payload compaction."""
        cache = getattr(self, "_pending_refresh_empty_tensor_cache", None)
        if cache is None:
            cache = {}
            self._pending_refresh_empty_tensor_cache = cache
        key = (str(device), str(dtype))
        cached = cache.get(key)
        if (
            isinstance(cached, torch.Tensor)
            and cached.device == device
            and cached.dtype == dtype
        ):
            return cached
        tensor = torch.empty((0,), device=device, dtype=dtype)
        cache[key] = tensor
        return tensor

    # ------------------------------------------------------------------
    # Step refresh helpers
    # ------------------------------------------------------------------

    def _step_refresh_handle_ledger_ensure(self) -> None:
        ring_size = self._step_handle_ring_size
        if ring_size <= 0:
            ring_size = 128
        ledger_size = self._step_refresh_handle_ledger_size
        ledger = self._step_refresh_handle_ledger
        if (
            isinstance(ledger, list)
            and ledger_size == ring_size
            and len(ledger) == ring_size
        ):
            return
        self._step_refresh_handle_ledger_size = ring_size
        self._step_refresh_handle_ledger = [
            [-1, -1, 0, 0, 0, 0, 0] for _ in range(ring_size)
        ]

    def _step_refresh_handle_ledger_clear_slot(self, *, slot: int) -> None:
        ledger = self._step_refresh_handle_ledger
        if slot < 0 or slot >= len(ledger):
            return
        entry = ledger[slot]
        entry[0] = -1
        entry[1] = -1
        entry[2] = 0
        entry[3] = 0
        entry[4] = 0
        entry[5] = 0
        entry[6] = 0

    def _step_refresh_handle_ledger_get(
        self,
        *,
        handle_id: int,
        handle_generation: int,
    ) -> Optional[List[int]]:
        if handle_id <= 0 or handle_generation <= 0:
            return None
        self._step_refresh_handle_ledger_ensure()
        ring_size = self._step_refresh_handle_ledger_size
        slot = handle_id % ring_size
        entry = self._step_refresh_handle_ledger[slot]
        if entry[0] != handle_id or entry[1] != handle_generation:
            return None
        return entry

    def _step_refresh_commit_assert_prev_enqueued(
        self,
        *,
        next_handle_hint: int,
        stage: str,
    ) -> None:
        self._step_refresh_handle_ledger_ensure()
        close_before_handle = next_handle_hint - 1
        if close_before_handle <= 0:
            return
        noop_empty = self._step_profile_refresh_noop_empty
        payload_none = self._step_profile_refresh_payload_none
        slot_empty = self._step_profile_refresh_slot_empty
        for slot, entry in enumerate(self._step_refresh_handle_ledger):
            handle_id = entry[0]
            if handle_id <= 0:
                continue
            # 给上一 step 一个 prepare 周期的缓冲，避免"先 prepare 后执行"的时序误报。
            if handle_id >= close_before_handle:
                continue
            handle_generation = entry[1]
            planned = entry[2]
            planned_rows = entry[3]
            num_actual_tokens = entry[4]
            post_kernel_calls = entry[5]
            enqueued = entry[6]
            # 单真源：仅以同 handle 的实证执行统计判定，不混入 profile 派生字段。
            has_confirmed_refresh_path = post_kernel_calls > 0
            if (
                planned > 0
                and num_actual_tokens > 0
                and enqueued == 0
                and has_confirmed_refresh_path
            ):
                raise RuntimeError(
                    "refresh commit invariant violated: planned refresh reqs but zero payload enqueues "
                    f"(handle_id={handle_id}, generation={handle_generation}, "
                    f"planned={planned}, planned_rows={planned_rows}, "
                    f"num_actual_tokens={num_actual_tokens}, post_kernel_calls={post_kernel_calls}, "
                    f"stage={stage}, noop_empty={noop_empty}, payload_none={payload_none}, "
                    f"slot_empty={slot_empty})"
                )
            self._step_refresh_handle_ledger_clear_slot(slot=slot)

    def _step_refresh_commit_begin(
        self,
        *,
        handle_id: int,
        handle_generation: int,
        planned_reqs: int,
        planned_rows: int,
        num_actual_tokens: int,
    ) -> int:
        commit_handle_id = handle_id
        commit_handle_generation = handle_generation
        if commit_handle_id <= 0 or commit_handle_generation <= 0:
            raise RuntimeError(
                "refresh commit begin missing handle identity: "
                f"handle_id={commit_handle_id} generation={commit_handle_generation}"
            )
        self._step_refresh_commit_assert_prev_enqueued(
            next_handle_hint=commit_handle_id,
            stage="refresh_commit_begin",
        )
        self._step_refresh_commit_id += 1
        self._step_refresh_commit_handle_id = commit_handle_id
        self._step_refresh_commit_handle_generation = commit_handle_generation
        self._step_refresh_commit_planned_reqs = max(0, planned_reqs)
        self._step_refresh_commit_planned_rows = max(0, planned_rows)
        self._step_refresh_commit_num_actual_tokens = max(0, num_actual_tokens)
        self._step_refresh_commit_payload_enqueues = 0
        self._step_refresh_commit_post_kernel_calls = 0
        self._step_refresh_commit_written_handle_id = commit_handle_id
        self._step_refresh_commit_written_handle_generation = commit_handle_generation
        self._step_refresh_commit_written_req_ids.clear()
        self._step_refresh_handle_ledger_ensure()
        ring_size = self._step_refresh_handle_ledger_size
        slot = commit_handle_id % ring_size
        entry = self._step_refresh_handle_ledger[slot]
        entry[0] = commit_handle_id
        entry[1] = commit_handle_generation
        entry[2] = self._step_refresh_commit_planned_reqs
        entry[3] = self._step_refresh_commit_planned_rows
        entry[4] = self._step_refresh_commit_num_actual_tokens
        entry[5] = 0
        entry[6] = 0
        return self._step_refresh_commit_id

    def _step_refresh_commit_note_enqueue(
        self,
        *,
        handle_id: int,
        handle_generation: int,
    ) -> None:
        self._step_refresh_commit_note_enqueues(
            handle_id=handle_id,
            handle_generation=handle_generation,
            count=1,
        )

    def _step_refresh_commit_note_enqueues(
        self,
        *,
        handle_id: int,
        handle_generation: int,
        count: int,
    ) -> None:
        count_i = int(count)
        if count_i <= 0:
            return
        commit_handle_id = handle_id
        commit_handle_generation = handle_generation
        entry = self._step_refresh_handle_ledger_get(
            handle_id=commit_handle_id,
            handle_generation=commit_handle_generation,
        )
        if entry is None:
            raise RuntimeError(
                "refresh commit enqueue missing handle-ledger entry: "
                f"handle_id={commit_handle_id} generation={commit_handle_generation}"
            )
        entry[6] = entry[6] + count_i
        if (
            self._step_refresh_commit_handle_id == commit_handle_id
            and self._step_refresh_commit_handle_generation == commit_handle_generation
        ):
            self._step_refresh_commit_payload_enqueues += count_i

    def _step_refresh_commit_note_post_kernel(
        self,
        *,
        handle_id: int,
        handle_generation: int,
    ) -> None:
        commit_handle_id = handle_id
        commit_handle_generation = handle_generation
        entry = self._step_refresh_handle_ledger_get(
            handle_id=commit_handle_id,
            handle_generation=commit_handle_generation,
        )
        if entry is None:
            raise RuntimeError(
                "refresh commit post-kernel missing handle-ledger entry: "
                f"handle_id={commit_handle_id} generation={commit_handle_generation}"
            )
        entry[5] = entry[5] + 1
        if (
            self._step_refresh_commit_handle_id == commit_handle_id
            and self._step_refresh_commit_handle_generation == commit_handle_generation
        ):
            self._step_refresh_commit_post_kernel_calls += 1

    def _step_refresh_commit_note_inflight_from_payload(self, payload: "SelectorBatchPayload") -> None:
        """Commit-phase single writer for scheduled refresh markers.

        Planner must stay pure (no inflight write). We only mark scheduled_* after
        payload enqueue succeeds to avoid silent trigger loss on ghost plans.
        """
        state = getattr(payload, "state", None)
        if state is None:
            return
        payload_handle_id = int(getattr(payload, "capture_handle_id", -1))
        payload_handle_generation = int(getattr(payload, "capture_handle_generation", -1))
        if payload_handle_id <= 0 or payload_handle_generation <= 0:
            raise RuntimeError(
                "refresh commit inflight update missing payload handle identity: "
                f"handle_id={payload_handle_id} generation={payload_handle_generation}"
            )
        if (
            self._step_refresh_commit_written_handle_id != payload_handle_id
            or self._step_refresh_commit_written_handle_generation
            != payload_handle_generation
        ):
            self._step_refresh_commit_written_handle_id = payload_handle_id
            self._step_refresh_commit_written_handle_generation = payload_handle_generation
            self._step_refresh_commit_written_req_ids.clear()
        batch_req_ids = getattr(state, "batch_request_ids", None)
        if batch_req_ids is None:
            raise RuntimeError("refresh commit requires state.batch_request_ids snapshot")
        slot_list = getattr(payload, "slot_list", None)
        if not slot_list:
            return
        raw_slot_req_ids = getattr(payload, "slot_req_ids", None)
        slot_req_ids: Optional[Tuple[str, ...]] = None
        if raw_slot_req_ids is not None:
            slot_req_ids = self._normalize_refresh_req_ids(raw_slot_req_ids)
            if len(slot_req_ids) != len(slot_list):
                raise RuntimeError(
                    "refresh commit slot_req_ids/slot_list length mismatch: "
                    f"slot_req_ids={len(slot_req_ids)} slot_list={len(slot_list)}"
                )
        request_states = self.request_states
        tickets = self._request_intent_tickets
        for idx, slot in enumerate(slot_list):
            if slot_req_ids is not None:
                req_id = slot_req_ids[idx]
            else:
                slot_idx = slot
                if slot_idx < 0 or slot_idx >= len(batch_req_ids):
                    raise RuntimeError(
                        "refresh commit slot index out of range: "
                        f"slot={slot_idx} batch={len(batch_req_ids)}"
                    )
                req_id = batch_req_ids[slot_idx]
            if not req_id or _is_free_slot_id(req_id):
                continue
            if req_id in self._step_refresh_commit_written_req_ids:
                continue
            tracking = request_states.get(req_id)
            ticket = tickets.get(req_id)
            if tracking is None or ticket is None or (not ticket.pending_refresh):
                continue
            pending_step = ticket.pending_decode_step
            if pending_step < 0:
                pending_step = int(tracking.decode_step) if tracking.decode_step is not None else -1
            pending_ctrl_step = ticket.pending_ctrl_step
            if pending_ctrl_step < 0:
                pending_ctrl_step = self.step_context_epoch
            tracking.scheduled_decode_refresh_step = pending_step
            tracking.scheduled_refresh_ctrl_step = pending_ctrl_step
            # [TP-DET-TRIGGER 2026-07-07] 决策终局提交点化(TP>1 NCCL 发散根修):
            # 触发线的全部决策状态在 enqueue 成功的 commit 点一次性终局——该点
            # 是纯 host 同步路径(方案 B single-writer),对所有 TP rank 逐 step
            # 决定论。原先 last 推进/trigger 计时归零/清票发生在 selector publish
            # 或 covered-by-ready-compact(依赖 GPU writer 完成时机,per-rank
            # 异步),使 sentence/interval 触发决策跨 rank 发散 → 集合发散挂死。
            # GPU 完成从此只服务读侧路由(off-rail/compact),不进决策。
            pending_policy_commit = int(ticket.pending_policy)
            pending_reason_commit = int(ticket.pending_reason_code)
            # 读侧在飞镜像:reason/policy 存续到 publish final(读侧闸消费:
            # dense-consume 防 torn-read + short_dense crossing 保护)。
            tracking.inflight_reason_code = pending_reason_commit
            tracking.inflight_policy = pending_policy_commit
            tracking.last_refresh_step = self.step_context_epoch
            tracking.last_decode_refresh_step = int(pending_step)
            trigger = getattr(tracking, "trigger", None)
            if trigger is not None:
                trigger.state.steps_since_refresh = 0
            if (
                pending_policy_commit == int(PendingPolicy.FORCE_NOW)
                or pending_reason_commit
                == int(PendingReasonCode.COMPACT_THRESHOLD_CROSSED)
            ):
                tracking._was_short_dense = False
            # 票转 consumed:决策面生命周期在提交点闭合(ready_compact=False
            # 不触碰刚写入的 scheduled_*;scheduled 转为读侧/观测语义,其读侧
            # 清除仍在 publish final——读路由允许 per-rank 时序,近似语义)。
            self._clear_request_pending_refresh(
                request_id=req_id,
                ready_compact=False,
            )
            self._step_refresh_commit_written_req_ids.add(req_id)

    def _step_refresh_nonempty(self) -> bool:
        step_ctx = self.step_context
        if step_ctx is None:
            return False
        step_authority = step_ctx.step_authority
        if step_authority is None:
            raise RuntimeError("step_context missing step_authority")
        if step_authority.epoch != step_ctx.epoch:
            raise RuntimeError(
                "step_authority epoch mismatch: "
                f"authority={step_authority.epoch} ctx={step_ctx.epoch}"
            )
        refresh_rows = step_authority.layer_effective_refresh_by_row
        if len(refresh_rows) < step_ctx.num_reqs:
            raise RuntimeError(
                "step_authority.layer_effective_refresh_by_row length mismatch: "
                f"rows={len(refresh_rows)} reqs={step_ctx.num_reqs}"
            )
        return any(refresh_rows[:step_ctx.num_reqs])

    # ------------------------------------------------------------------
    # Utility: normalize / resolve
    # ------------------------------------------------------------------

    def _normalize_refresh_req_ids(self, req_ids: Optional[Sequence[object]]) -> Tuple[str, ...]:
        if not req_ids:
            return tuple()
        out: List[str] = []
        seen: Set[str] = set()
        for rid in req_ids:
            if not rid:
                continue
            rid_str = str(rid)
            if rid_str in seen:
                continue
            seen.add(rid_str)
            out.append(rid_str)
        return tuple(out)

    def _resolve_refresh_lease(
        self,
        *,
        req_ids: Sequence[str],
        reason: str,
    ) -> None:
        normalized_ids = self._normalize_refresh_req_ids(tuple(req_ids) if req_ids is not None else tuple())
        if not normalized_ids:
            return
        changed = False
        reason_str = reason
        for rid in normalized_ids:
            tracking = self.request_states.get(rid)
            if tracking is None:
                continue
            tracking.scheduled_refresh_ctrl_step = -1
            tracking.scheduled_decode_refresh_step = -1
            # [TP-DET-TRIGGER] lease 重排=世代作废,读侧在飞镜像同步清除。
            tracking.inflight_reason_code = -1
            tracking.inflight_policy = -1
            try:
                cur_decode = int(tracking.decode_step) if tracking.decode_step is not None else -1
            except Exception:
                _log.warning("failed to read decode_step from tracking for rid=%s", rid)
                raise
            changed = self._set_request_lease_rearm(
                request_id=rid,
                reason=reason_str,
                decode_step=cur_decode,
            ) or changed
        if changed:
            self._bump_refresh_nonce()

    # ------------------------------------------------------------------
    # Refresh stream lifecycle
    # ------------------------------------------------------------------

    def _ensure_refresh_stream(self, device: torch.device) -> None:
        """Lazy init refresh_stream and per-buf events. Safe to call repeatedly."""
        if (
            self.refresh_stream is not None
            and self._refresh_stream_device == device
            and self.chunk_done_evt
            and self.prefill_done_evt
            and self.refresh_done_evt
        ):
            return
        if not torch.cuda.is_available():
            return
        # cudagraph capture 期间不允许创建 stream/event；同时 async 路径本就应被禁用。
        if _is_stream_capturing_or_raise(stage="ensure_refresh_stream"):
            return
        # 初始化 stream/event 时避免隐式同步；仅创建对象与记录初始 done 事件。
        # refresh_stream 默认用低优先级，避免在 decode 热路径上引入额外 GPU 争用。
        refresh_priority = int(_REFRESH_STREAM_PRIORITY_CACHED)
        # clamp priority 到当前环境支持范围，避免无效值静默被 clamp（或触发异常）。
        try:
            pr_least, pr_greatest = torch.cuda.Stream.priority_range()
            # CUDA 语义：数值越小优先级越高；通常 pr_greatest 为负数，pr_least 为 0。
            refresh_priority = max(int(pr_greatest), min(int(pr_least), int(refresh_priority)))
        except Exception:
            _log.warning("failed to query CUDA stream priority range")
            raise
        try:
            self.refresh_stream = torch.cuda.Stream(device=device, priority=refresh_priority)
        except Exception:
            _log.warning("failed to create CUDA stream with priority=%d", refresh_priority)
            raise
        self._refresh_stream_device = device
        self.chunk_ready_evt = [torch.cuda.Event(enable_timing=False) for _ in range(_CAPTURE_IN_FLIGHT)]
        self.chunk_done_evt = [torch.cuda.Event(enable_timing=False) for _ in range(_CAPTURE_IN_FLIGHT)]
        # 细粒度 done 事件：默认与 chunk_done_evt 同步推进；未来可用于更细粒度等待。
        self.prefill_done_evt = [torch.cuda.Event(enable_timing=False) for _ in range(_CAPTURE_IN_FLIGHT)]
        self.refresh_done_evt = [torch.cuda.Event(enable_timing=False) for _ in range(_CAPTURE_IN_FLIGHT)]
        # 重置 split-wait flags，避免跨 device/重建的陈旧状态误导等待策略。
        self._pending_work_reset()
        main_stream = torch.cuda.current_stream(device=device)
        for evt in self.chunk_done_evt:
            evt.record(main_stream)
        for evt in self.prefill_done_evt:
            evt.record(main_stream)
        for evt in self.refresh_done_evt:
            evt.record(main_stream)

    # ------------------------------------------------------------------
    # Enqueue refresh capture
    # ------------------------------------------------------------------

    def _enqueue_refresh_capture(self, payload: SelectorBatchPayload) -> None:
        payload_handle_id = payload.capture_handle_id
        payload_handle_generation = payload.capture_handle_generation
        if payload_handle_id <= 0 or payload_handle_generation <= 0:
            raise RuntimeError(
                "refresh capture enqueue missing payload handle identity: "
                f"handle_id={payload_handle_id} generation={payload_handle_generation}"
            )
        self._step_refresh_commit_note_enqueue(
            handle_id=payload_handle_id,
            handle_generation=payload_handle_generation,
        )
        # [DETERMINISTIC-REQIDS-SNAPSHOT 2026-07-03] slot_req_ids 提交步物化:
        # 多数构造点不填该字段,deferred 消费点曾 fallback 读 live 的
        # state.batch_request_ids(晚读时 batch 成员可能已变,与 slot_list 不再
        # 自洽)。入队时刻 slot_list↔batch_request_ids 天然自洽,单点物化为不可
        # 变 tuple 覆盖全部构造路径;每 refresh payload 一次纯 host tuple 构造。
        if payload.slot_req_ids is None:
            _sl = payload.slot_list
            _brids = getattr(payload.state, "batch_request_ids", None)
            if _sl is not None and _brids is not None:
                payload.slot_req_ids = tuple(
                    str(_brids[int(_s)]) for _s in _sl if int(_s) >= 0
                )
        self._record_sentence_materialized_refresh_payload(
            slot_req_ids=payload.slot_req_ids
        )
        self._step_refresh_commit_note_inflight_from_payload(payload)
        if self.step_refresh_epoch != self.step_context_epoch:
            self.step_refresh_epoch = self.step_context_epoch
            for buf_id, bucket in enumerate(self.step_refresh_chunk_payloads):
                for idx in range(_CAPTURE_CHUNK):
                    bucket[idx] = None
                self.step_refresh_chunk_mask[buf_id] = 0
        layer_index = self.layer_index_by_cache_key.get(payload.cache_key, -1)
        if layer_index < 0:
            raise RuntimeError("refresh capture enqueue missing layer_index")
        chunk_id, buf_id, slot_in_chunk = self._map_global_layer_to_capture_slot(layer_index)
        if slot_in_chunk < 0 or slot_in_chunk >= _CAPTURE_CHUNK:
            raise RuntimeError("refresh capture enqueue slot_in_chunk out of range")

        bucket = self.step_refresh_chunk_payloads[buf_id]
        mask = self.step_refresh_chunk_mask[buf_id]
        bit = 1 << slot_in_chunk
        if (mask & bit) != 0 or bucket[slot_in_chunk] is not None:
            raise RuntimeError(
                f"duplicate refresh payload for buf_id={buf_id} slot_in_chunk={slot_in_chunk}"
            )
        # payload.layer_index 约定为 chunk 内索引（slot_in_chunk），在构造时已固定，避免入队时写字段
        bucket[slot_in_chunk] = payload
        self.step_refresh_chunk_mask[buf_id] = mask | bit

    # ------------------------------------------------------------------
    # Nonce and prefill release
    # ------------------------------------------------------------------

    def _bump_refresh_nonce(self) -> None:
        self._refresh_nonce += 1

    def _maybe_release_prefill_state(self, state: LayerState) -> None:
        if int(state.prefill_active_count) > 0:
            return
        state.prefill_kv_lengths = None
        state.prefill_kv_len_per_row_i32 = None
        state.prefill_active_mask = None
        state.prefill_active_count = 0
        state.prefill_done_mask = None
        state.prefill_fifo_counts_cpu = None
        state.prefill_total_chunks = None
        state.prefill_chunks_seen = None
        state.bootstrap_done = self._all_slots_bootstrapped(state)
