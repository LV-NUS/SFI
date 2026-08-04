"""
patches/controller_mixins/profile_mixin.py — Step-level and refresh-level profiling for VLLMSparseController.

OWNS:
  - _init_profile_state(): profiling state initialization
  - _step_profile_begin / _step_profile_end_if_needed(): step profiling lifecycle
  - _step_profile_record_plan(): record dispatch plan mode
  - _step_profile_record_refresh_payload / _step_profile_record_refresh_noop(): refresh payload tracking
  - _refresh_profile_should_sample(): refresh profiling sampling gate
  - _refresh_profile_try_flush_pending(): flush pending refresh profile entries

DEPENDS_ON:
  - patches.sparse_constants (profiling env-cached flags)
  - patches.sparse_types.StepMeta, _RefreshProfilePending

ENTRY_POINTS:
  - _init_profile_state(): called from VLLMSparseController.__init__
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import fields
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple

_log = logging.getLogger(__name__)

import torch

from patches.sparse_constants import (
    _CAPTURE_IN_FLIGHT,
    _REFRESH_PROFILE_CACHED,
    _REFRESH_PROFILE_CALL_MIN_CACHED,
    _REFRESH_PROFILE_DETAIL_CACHED,
    _REFRESH_PROFILE_EVERY_CACHED,
    _REFRESH_PROFILE_LOG_CACHED,
    _STEP_PROFILE_CACHED,
    _STEP_PROFILE_DETAIL_CACHED,
    _STEP_PROFILE_EVERY_CACHED,
    _STEP_PROFILE_LOG_CACHED,
)
from patches.sparse_types import (
    ASYNC_PRODUCER_GPU_PROFILE_STAGES,
    _RefreshProfilePending,
)
from patches.request_intent_ticket import pending_reason_code_to_text

if TYPE_CHECKING:
    from patches.sparse_types import StepMeta

# Module-level mutable counter (only used by _refresh_profile_should_sample)
_REFRESH_PROFILE_CALLS: int = 0


class ProfileMixin:
    """Step-level and refresh-level profiling for VLLMSparseController.

    All profiling state is initialised via ``_init_profile_state()`` which
    must be called from the controller's ``__init__``.
    """

    _MIXIN_REQUIRES: tuple = ()

    # ------------------------------------------------------------------
    # State initialisation
    # ------------------------------------------------------------------

    def _init_profile_state(self) -> None:
        # --- step profile (reset each step in _step_profile_begin) ---
        self._step_profile_epoch: int = -1
        self._step_profile_logged_epoch: int = -1
        self._step_profile_plan_mode: str = "none"
        self._step_profile_plan_reason: str = "none"
        self._step_profile_refresh_reason: str = "none"
        self._step_profile_refresh_reqs: int = 0
        self._step_profile_refresh_nonempty: bool = False
        self._step_profile_refresh_payloads: int = 0
        self._step_profile_refresh_noop_empty: int = 0
        self._step_profile_refresh_payload_none: int = 0
        self._step_profile_refresh_slot_empty: int = 0
        # detail trace（仅在 STEP_PROFILE_DETAIL=1 时写出）
        self._step_profile_req_ids: Tuple[str, ...] = tuple()
        self._step_profile_refresh_req_ids: Tuple[str, ...] = tuple()
        self._step_profile_refresh_rows: Tuple[int, ...] = tuple()
        self._step_profile_req_decode_steps: Tuple[int, ...] = tuple()
        self._step_profile_req_last_decode_refresh_steps: Tuple[int, ...] = tuple()
        self._step_profile_req_interval_deltas: Tuple[int, ...] = tuple()
        self._step_profile_req_ticket_pending: Tuple[bool, ...] = tuple()
        self._step_profile_req_ticket_pending_steps: Tuple[int, ...] = tuple()
        self._step_profile_req_ticket_pending_reasons: Tuple[str, ...] = tuple()
        # step plan 使用统计（按 step 记一次）
        self._step_plan_stats_epoch: int = -1
        self._step_plan_stats_hits: int = 0
        self._step_plan_stats_misses: int = 0
        # refresh profile：按 buf 记录"上一次 flush 的 pending 事件"
        self._refresh_profile_pending_by_buf: List[Optional[_RefreshProfilePending]] = [
            None for _ in range(int(_CAPTURE_IN_FLIGHT))
        ]
        self._refresh_profile_active: bool = False
        self._sentence_trigger_intents_total: int = 0
        self._sentence_trigger_intents_profile_emitted: int = 0
        self._sentence_trigger_materialized_refresh_count: int = 0
        self._sentence_trigger_materialized_refresh_epoch: int = -1
        self._sentence_trigger_materialized_payload_epoch: int = -1
        self._sentence_trigger_materialized_profile_emitted_count: int = 0
        self._sentence_trigger_admission_coalesced_total: int = 0
        self._sentence_trigger_admission_coalesced_interval_pending_total: int = 0
        self._sentence_trigger_admission_dropped_finished_total: int = 0
        self._refresh_coalesce_skipped_existing_pending_total: int = 0

    def _selector_log_f_reduce_route_snapshot(self) -> dict[str, object]:
        """Return the existing postprocess route counters as JSON-safe data."""
        raw_counts = getattr(self, "_selector_log_f_reduce_route_counts", None)
        counts = (
            {
                str(route): int(count)
                for route, count in sorted(raw_counts.items())
            }
            if isinstance(raw_counts, dict)
            else {}
        )
        return {
            "last_route": str(
                getattr(self, "_selector_log_f_reduce_last_route", "none")
            ),
            "counts": counts,
        }

    # ------------------------------------------------------------------
    # Step profiling
    # ------------------------------------------------------------------

    @staticmethod
    def _step_profile_enabled() -> bool:
        return _STEP_PROFILE_CACHED

    @staticmethod
    def _step_profile_detail_enabled() -> bool:
        return _STEP_PROFILE_DETAIL_CACHED

    @staticmethod
    def _step_profile_every() -> int:
        return int(_STEP_PROFILE_EVERY_CACHED)

    @staticmethod
    def _step_profile_log_path() -> str:
        return str(_STEP_PROFILE_LOG_CACHED)

    def _record_sentence_materialized_refresh_payload(
        self,
        *,
        slot_req_ids: Sequence[str] | None = None,
    ) -> None:
        epoch = int(getattr(self, "step_context_epoch", self._step_profile_epoch))
        if epoch < 0:
            return
        if int(getattr(self, "_sentence_trigger_materialized_payload_epoch", -1)) == epoch:
            return

        reason_hit = "sentence" in str(
            getattr(self, "_step_profile_refresh_reason", "")
        )
        if not reason_hit and slot_req_ids:
            tickets = getattr(self, "_request_intent_tickets", {}) or {}
            for req_id in slot_req_ids:
                ticket = tickets.get(str(req_id))
                reason_code = int(getattr(ticket, "pending_reason_code", 0))
                if pending_reason_code_to_text(reason_code) == "sentence":
                    reason_hit = True
                    break
        if not reason_hit:
            return

        self._sentence_trigger_materialized_refresh_count = (
            int(getattr(self, "_sentence_trigger_materialized_refresh_count", 0)) + 1
        )
        self._sentence_trigger_materialized_refresh_epoch = epoch
        self._sentence_trigger_materialized_payload_epoch = epoch

    def _step_profile_begin(
        self,
        *,
        step_meta: "StepMeta",
        refresh_reason: str,
        refresh_reqs: Sequence[str],
    ) -> None:
        self._step_profile_epoch = step_meta.epoch
        self._step_profile_refresh_reason = str(refresh_reason)
        self._step_profile_refresh_reqs = int(len(refresh_reqs)) if refresh_reqs is not None else 0
        self._step_profile_refresh_nonempty = bool(self._step_profile_refresh_reqs)
        if not self._step_profile_enabled():
            return
        self._step_profile_logged_epoch = -1
        self._step_profile_plan_mode = "none"
        self._step_profile_plan_reason = "none"
        self._step_profile_refresh_payloads = 0
        self._step_profile_refresh_noop_empty = 0
        self._step_profile_refresh_payload_none = 0
        self._step_profile_refresh_slot_empty = 0
        self._step_profile_req_ids = tuple()
        self._step_profile_refresh_req_ids = tuple()
        self._step_profile_refresh_rows = tuple()
        self._step_profile_req_decode_steps = tuple()
        self._step_profile_req_last_decode_refresh_steps = tuple()
        self._step_profile_req_interval_deltas = tuple()
        self._step_profile_req_ticket_pending = tuple()
        self._step_profile_req_ticket_pending_steps = tuple()
        self._step_profile_req_ticket_pending_reasons = tuple()
        if self._step_profile_detail_enabled():
            req_ids = tuple(str(rid) for rid in (step_meta.req_ids or tuple()))
            refresh_req_ids = (
                tuple(str(rid) for rid in refresh_reqs if isinstance(rid, str) and rid)
                if refresh_reqs is not None
                else tuple()
            )
            refresh_rows: Tuple[int, ...] = tuple()
            if req_ids and refresh_req_ids:
                req_idx = {rid: idx for idx, rid in enumerate(req_ids)}
                refresh_rows = tuple(
                    int(req_idx[rid])
                    for rid in refresh_req_ids
                    if rid in req_idx
                )
            self._step_profile_req_ids = req_ids
            self._step_profile_refresh_req_ids = refresh_req_ids
            self._step_profile_refresh_rows = refresh_rows
            request_states = getattr(self, "request_states", None)
            tickets = getattr(self, "_request_intent_tickets", None)
            decode_steps: List[int] = []
            last_decode_refresh_steps: List[int] = []
            interval_deltas: List[int] = []
            ticket_pending: List[bool] = []
            ticket_pending_steps: List[int] = []
            ticket_pending_reasons: List[str] = []
            for rid in req_ids:
                tracking = (
                    request_states.get(rid)
                    if isinstance(request_states, dict)
                    else None
                )
                decode_step = int(getattr(tracking, "decode_step", -1))
                last_decode_refresh = int(
                    getattr(tracking, "last_decode_refresh_step", -1)
                )
                interval_delta = -1
                if decode_step >= 0 and last_decode_refresh >= 0:
                    interval_delta = int(decode_step - last_decode_refresh)
                decode_steps.append(int(decode_step))
                last_decode_refresh_steps.append(int(last_decode_refresh))
                interval_deltas.append(int(interval_delta))

                ticket = (
                    tickets.get(rid)
                    if isinstance(tickets, dict)
                    else None
                )
                pending_flag = bool(getattr(ticket, "pending_refresh", False))
                pending_step = int(getattr(ticket, "pending_decode_step", -1))
                pending_reason_code = int(getattr(ticket, "pending_reason_code", 0))
                ticket_pending.append(bool(pending_flag))
                ticket_pending_steps.append(int(pending_step))
                ticket_pending_reasons.append(
                    str(pending_reason_code_to_text(pending_reason_code))
                )
            self._step_profile_req_decode_steps = tuple(decode_steps)
            self._step_profile_req_last_decode_refresh_steps = tuple(
                last_decode_refresh_steps
            )
            self._step_profile_req_interval_deltas = tuple(interval_deltas)
            self._step_profile_req_ticket_pending = tuple(ticket_pending)
            self._step_profile_req_ticket_pending_steps = tuple(
                ticket_pending_steps
            )
            self._step_profile_req_ticket_pending_reasons = tuple(
                ticket_pending_reasons
            )
        # per-step 计划复用计数（按 layer 统计）
        self._step_plan_stats_hits = 0
        self._step_plan_stats_misses = 0
        self._step_plan_stats_epoch = int(step_meta.epoch)

    def _step_profile_record_plan(self, *, mode: str, reason: Optional[str]) -> None:
        if not self._step_profile_enabled():
            return
        self._step_profile_plan_mode = str(mode)
        self._step_profile_plan_reason = str(reason) if reason is not None else "none"

    def _step_profile_record_refresh_payload(self) -> None:
        self._record_sentence_materialized_refresh_payload()
        if not self._step_profile_enabled():
            return
        self._step_profile_refresh_payloads += 1

    def _step_profile_record_refresh_payloads(
        self,
        *,
        count: int,
        slot_req_ids: Sequence[str] | None = None,
    ) -> None:
        count_i = int(count)
        if count_i <= 0:
            return
        self._record_sentence_materialized_refresh_payload(
            slot_req_ids=slot_req_ids
        )
        if not self._step_profile_enabled():
            return
        self._step_profile_refresh_payloads += count_i

    def _step_profile_record_refresh_noop(self, kind: str) -> None:
        if not self._step_profile_enabled():
            return
        if kind == "empty_refresh_set":
            self._step_profile_refresh_noop_empty += 1
        elif kind == "payload_none":
            self._step_profile_refresh_payload_none += 1
        elif kind == "empty_slots":
            self._step_profile_refresh_slot_empty += 1

    def _step_profile_end_if_needed(self, *, epoch: int) -> None:
        if not self._step_profile_enabled():
            return
        if self._step_profile_logged_epoch == epoch:
            return
        if self._step_profile_epoch != epoch:
            return
        step_every = self._step_profile_every()
        should_log = False
        if step_every > 0 and self.step_context_epoch % step_every == 0:
            should_log = True
        if (
            self._step_profile_refresh_noop_empty > 0
            or self._step_profile_refresh_payload_none > 0
            or self._step_profile_refresh_slot_empty > 0
        ):
            should_log = True
        if not should_log:
            self._step_profile_logged_epoch = epoch
            return
        selector_log_f_route = self._selector_log_f_reduce_route_snapshot()
        rec = {
            "pid": int(os.getpid()),
            "step": self.step_context_epoch,
            "epoch": epoch,
            "decode_only": bool(self.step_authority.is_decode_only) if self.step_authority else False,
            "refresh_reason": str(self._step_profile_refresh_reason),
            "refresh_reqs": int(self._step_profile_refresh_reqs),
            "refresh_nonempty": bool(self._step_profile_refresh_nonempty),
            "plan_mode": str(self._step_profile_plan_mode),
            "plan_reason": str(self._step_profile_plan_reason),
            "step_plan_hits": int(self._step_plan_stats_hits),
            "step_plan_misses": int(self._step_plan_stats_misses),
            "refresh_payloads": int(self._step_profile_refresh_payloads),
            "refresh_noop_empty": int(self._step_profile_refresh_noop_empty),
            "refresh_payload_none": int(self._step_profile_refresh_payload_none),
            "refresh_slot_empty": int(self._step_profile_refresh_slot_empty),
            "sentence_trigger_admission_coalesced_total": int(
                getattr(self, "_sentence_trigger_admission_coalesced_total", 0)
            ),
            "sentence_trigger_admission_coalesced_interval_pending_total": int(
                getattr(
                    self,
                    "_sentence_trigger_admission_coalesced_interval_pending_total",
                    0,
                )
            ),
            "sentence_trigger_admission_dropped_finished_total": int(
                getattr(
                    self,
                    "_sentence_trigger_admission_dropped_finished_total",
                    0,
                )
            ),
            "refresh_coalesce_skipped_existing_pending_total": int(
                getattr(
                    self,
                    "_refresh_coalesce_skipped_existing_pending_total",
                    0,
                )
            ),
            "selector_log_f_reduce_last_route": selector_log_f_route[
                "last_route"
            ],
            "selector_log_f_reduce_route_counts": selector_log_f_route["counts"],
            "layer_group_enabled": bool(self._refresh_layer_group_enabled),
            "layer_group_active": self._refresh_layer_group_active,
        }
        if self.step_context is not None:
            _auth = getattr(self, "step_authority", None)
            if _auth is not None and int(getattr(_auth, "epoch", -1)) == int(getattr(self.step_context, "epoch", -1)):
                logits_last_n_by_row = tuple(
                    int(v) for v in _auth.logits_last_n_by_row
                )
                logits_capacity_by_row = tuple(
                    int(v) for v in _auth.logits_capacity_by_row
                )
                rec["logits_max_last_n"] = max(logits_last_n_by_row) if logits_last_n_by_row else 0
                rec["logits_max_kv"] = max(logits_capacity_by_row) if logits_capacity_by_row else 0
            else:
                rec["logits_max_last_n"] = 0
                rec["logits_max_kv"] = 0
            rec["logits_ready_epoch"] = int(getattr(self, "_step_logits_ready_token", -1))
            step_envelope = getattr(self.step_context, "step_envelope_v2", None)
            if step_envelope is not None:
                layer_refresh = getattr(
                    step_envelope, "layer_effective_refresh_by_row", tuple()
                )
                rec["intent_refresh_true"] = int(sum(1 for v in layer_refresh if bool(v)))
                rec["intent_refresh_rows"] = int(len(layer_refresh))
        if self._step_profile_detail_enabled():
            rec["req_ids"] = list(self._step_profile_req_ids)
            rec["refresh_req_ids"] = list(self._step_profile_refresh_req_ids)
            rec["refresh_rows"] = list(self._step_profile_refresh_rows)
            rec["req_decode_steps"] = list(self._step_profile_req_decode_steps)
            rec["req_last_decode_refresh_steps"] = list(
                self._step_profile_req_last_decode_refresh_steps
            )
            rec["req_interval_deltas"] = list(self._step_profile_req_interval_deltas)
            rec["req_ticket_pending"] = list(self._step_profile_req_ticket_pending)
            rec["req_ticket_pending_steps"] = list(
                self._step_profile_req_ticket_pending_steps
            )
            rec["req_ticket_pending_reasons"] = list(
                self._step_profile_req_ticket_pending_reasons
            )
        try:
            with open(self._step_profile_log_path(), "a", encoding="utf-8") as fp:
                fp.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
        except Exception:
            _log.warning("Failed to write step profile log to %s", self._step_profile_log_path(), exc_info=True)
            raise
        self._step_profile_logged_epoch = epoch

    # ------------------------------------------------------------------
    # Refresh profiling
    # ------------------------------------------------------------------

    @staticmethod
    def _refresh_profile_enabled() -> bool:
        return _REFRESH_PROFILE_CACHED

    @staticmethod
    def _refresh_profile_params() -> Tuple[int, int, str]:
        return _REFRESH_PROFILE_CALL_MIN_CACHED, _REFRESH_PROFILE_EVERY_CACHED, _REFRESH_PROFILE_LOG_CACHED

    @staticmethod
    def _refresh_profile_detail_enabled() -> bool:
        return bool(_REFRESH_PROFILE_DETAIL_CACHED)

    @classmethod
    def _refresh_profile_write(cls, message: str) -> None:
        if not cls._refresh_profile_enabled():
            return
        try:
            _, _, log_path = cls._refresh_profile_params()
            with open(log_path, "a", encoding="utf-8") as fp:
                fp.write(message + "\n")
        except Exception:
            _log.warning("Failed to write refresh profile log", exc_info=True)
            raise

    def _refresh_profile_should_sample(self) -> bool:
        """Return whether to sample refresh profiling for this flush call."""
        global _REFRESH_PROFILE_CALLS
        if not self._refresh_profile_enabled():
            return False
        _REFRESH_PROFILE_CALLS += 1
        call_min, every, _ = self._refresh_profile_params()
        if _REFRESH_PROFILE_CALLS < int(call_min):
            return False
        if int(every) <= 1:
            return True
        return ((_REFRESH_PROFILE_CALLS - int(call_min)) % int(every)) == 0

    def _refresh_profile_try_flush_pending(self, *, buf: int, wait_epoch: Optional[int]) -> None:
        """Flush pending refresh profile record for buf, assuming its done event has been waited."""
        if not self._refresh_profile_enabled():
            return
        if buf < 0 or buf >= len(self._refresh_profile_pending_by_buf):
            return
        pending = self._refresh_profile_pending_by_buf[buf]
        if pending is None:
            return

        def _cuda_event_ready(evt: object) -> bool:
            query = getattr(evt, "query", None)
            if not callable(query):
                return True
            try:
                return bool(query())
            except Exception:
                _log.warning("CUDA event query failed", exc_info=True)
                return True

        def _value_cuda_events_ready(value: object) -> bool:
            if value is None:
                return True
            query = getattr(value, "query", None)
            if callable(query):
                return _cuda_event_ready(value)
            if isinstance(value, tuple):
                return all(_value_cuda_events_ready(item) for item in value)
            return True

        if not all(
            _value_cuda_events_ready(getattr(pending, field.name))
            for field in fields(pending)
        ):
            return

        def _evt_ms(evt0: Optional[torch.cuda.Event], evt1: Optional[torch.cuda.Event]) -> Optional[float]:
            if evt0 is None or evt1 is None:
                return None
            try:
                for evt in (evt0, evt1):
                    query = getattr(evt, "query", None)
                    if callable(query) and not bool(query()):
                        return None
                return float(evt0.elapsed_time(evt1))
            except Exception:
                # [WARN-ONCE 2026-07-08] 纯观测计时失败(teardown 期事件对等)
                # 曾按次带全栈刷屏(单 run 实测 ×108 条 traceback 污染取证
                # 现场);首例留全栈,后续静默返 None(值语义不变)。
                if not getattr(self, "_evt_ms_warned_once", False):
                    self._evt_ms_warned_once = True
                    _log.warning("CUDA event elapsed_time failed", exc_info=True)
                return None

        def _evt_pairs_ms(
            pairs: Sequence[Tuple[torch.cuda.Event, torch.cuda.Event]],
        ) -> Optional[float]:
            values: List[float] = []
            for evt0, evt1 in pairs:
                value = _evt_ms(evt0, evt1)
                if value is not None:
                    values.append(float(value))
            return float(sum(values)) if values else None

        def _prefill_group_profile_fields(
            prefill_start_evt: Optional[torch.cuda.Event],
            pairs: Sequence[Tuple[int, torch.cuda.Event, torch.cuda.Event]],
        ) -> dict[str, float | int]:
            fields: dict[str, float | int] = {"prefill_group_count": 0}
            group_ids = sorted({int(group_id) for group_id, _, _ in pairs})
            fields["prefill_group_count"] = int(len(group_ids))
            for group_id in group_ids:
                duration_ms = 0.0
                duration_seen = False
                done_since_start_ms: Optional[float] = None
                for pair_group_id, evt0, evt1 in pairs:
                    if int(pair_group_id) != int(group_id):
                        continue
                    value = _evt_ms(evt0, evt1)
                    if value is not None:
                        duration_ms += float(value)
                        duration_seen = True
                    if prefill_start_evt is not None:
                        done_value = _evt_ms(prefill_start_evt, evt1)
                        if done_value is not None:
                            done_since_start_ms = max(
                                float(done_since_start_ms or 0.0),
                                float(done_value),
                            )
                if duration_seen:
                    fields[f"prefill_group{group_id}_gpu_ms"] = float(duration_ms)
                if done_since_start_ms is not None:
                    fields[f"prefill_group{group_id}_done_since_prefill_start_ms"] = float(
                        done_since_start_ms
                    )
            return fields

        selector_log_f_route = self._selector_log_f_reduce_route_snapshot()
        record = {
            "pid": int(os.getpid()),
            # ctrl_step：用于在 benchmark 内严格区分 warmup 与测量阶段（避免 warmup 的 pending flush 污染统计）。
            "ctrl_step": int(getattr(self, "step", -1)),
            "ts_ns": int(time.time_ns()),
            "epoch": int(pending.epoch),
            "wait_epoch": (int(wait_epoch) if wait_epoch is not None else None),
            "chunk_id": int(pending.chunk_id),
            "buf_id": int(pending.buf_id),
            "device": pending.device,
            "device_index": int(pending.device_index),
            "prefill_payloads": int(pending.prefill_payloads),
            "refresh_payloads": int(pending.refresh_payloads),
            "sentence_trigger_intents": int(pending.sentence_trigger_intents),
            "sentence_trigger_admission_coalesced_total": int(
                getattr(self, "_sentence_trigger_admission_coalesced_total", 0)
            ),
            "sentence_trigger_admission_coalesced_interval_pending_total": int(
                getattr(
                    self,
                    "_sentence_trigger_admission_coalesced_interval_pending_total",
                    0,
                )
            ),
            "sentence_trigger_admission_dropped_finished_total": int(
                getattr(
                    self,
                    "_sentence_trigger_admission_dropped_finished_total",
                    0,
                )
            ),
            "refresh_coalesce_skipped_existing_pending_total": int(
                getattr(
                    self,
                    "_refresh_coalesce_skipped_existing_pending_total",
                    0,
                )
            ),
            "selector_log_f_reduce_last_route": selector_log_f_route[
                "last_route"
            ],
            "selector_log_f_reduce_route_counts": selector_log_f_route["counts"],
            "prefill_selector_runs": int(pending.prefill_selector_runs),
            "prefill_rebuild_runs": int(pending.prefill_rebuild_runs),
            "prefill_cpu_us": float(pending.prefill_cpu_us),
            "prefill_selector_compute_cpu_us": float(
                pending.prefill_selector_compute_cpu_us
            ),
            "prefill_selector_post_cpu_us": float(
                pending.prefill_selector_post_cpu_us
            ),
            "prefill_selector_stack_cpu_us": float(
                pending.prefill_selector_stack_cpu_us
            ),
            "prefill_selector_validate_cpu_us": float(
                pending.prefill_selector_validate_cpu_us
            ),
            "prefill_selector_key_norms_cpu_us": float(
                pending.prefill_selector_key_norms_cpu_us
            ),
            "prefill_selector_key_norms_arena_cpu_us": float(
                pending.prefill_selector_key_norms_arena_cpu_us
            ),
            "prefill_selector_key_norms_direct_cpu_us": float(
                pending.prefill_selector_key_norms_direct_cpu_us
            ),
            "prefill_selector_key_norms_direct_prepare_cpu_us": float(
                pending.prefill_selector_key_norms_direct_prepare_cpu_us
            ),
            "prefill_selector_key_norms_direct_launch_cpu_us": float(
                pending.prefill_selector_key_norms_direct_launch_cpu_us
            ),
            "prefill_selector_key_norms_pack_cpu_us": float(
                pending.prefill_selector_key_norms_pack_cpu_us
            ),
            "prefill_selector_select_cpu_us": float(
                pending.prefill_selector_select_cpu_us
            ),
            "prefill_rebuild_cpu_us": float(pending.prefill_rebuild_cpu_us),
            "refresh_selector_cpu_us": float(pending.refresh_selector_cpu_us),
            "refresh_selector_apply_cpu_us": float(
                pending.refresh_selector_apply_cpu_us
            ),
            "refresh_selector_compute_cpu_us": float(pending.refresh_selector_compute_cpu_us),
            "refresh_selector_post_cpu_us": float(pending.refresh_selector_post_cpu_us),
            "refresh_selector_stack_cpu_us": float(pending.refresh_selector_stack_cpu_us),
            "refresh_selector_key_norms_cpu_us": float(
                pending.refresh_selector_key_norms_cpu_us
            ),
            "refresh_selector_key_norms_arena_cpu_us": float(
                pending.refresh_selector_key_norms_arena_cpu_us
            ),
            "refresh_selector_key_norms_direct_cpu_us": float(
                pending.refresh_selector_key_norms_direct_cpu_us
            ),
            "refresh_selector_key_norms_direct_prepare_cpu_us": float(
                pending.refresh_selector_key_norms_direct_prepare_cpu_us
            ),
            "refresh_selector_key_norms_direct_launch_cpu_us": float(
                pending.refresh_selector_key_norms_direct_launch_cpu_us
            ),
            "refresh_selector_key_norms_pack_cpu_us": float(
                pending.refresh_selector_key_norms_pack_cpu_us
            ),
            "refresh_rebuild_cpu_us": float(pending.refresh_rebuild_cpu_us),
            "refresh_total_cpu_us": float(pending.refresh_total_cpu_us),
            "refresh_rebuild_enqueue_cpu_us": float(
                pending.refresh_rebuild_enqueue_cpu_us
            ),
            "refresh_rebuild_compact_cpu_us": float(
                pending.refresh_rebuild_compact_cpu_us
            ),
            "prefill_gpu_ms": _evt_ms(pending.prefill_evt0, pending.prefill_evt1),
            "prefill_selector_gpu_ms": _evt_pairs_ms(pending.prefill_selector_evt_pairs),
            "prefill_gather_gpu_ms": _evt_pairs_ms(pending.prefill_gather_evt_pairs),
            "prefill_key_norms_preproc_gpu_ms": _evt_pairs_ms(
                pending.prefill_key_norms_preproc_evt_pairs
            ),
            "prefill_key_norms_gpu_ms": _evt_pairs_ms(
                pending.prefill_key_norms_evt_pairs
            ),
            "prefill_key_norms_h2d_gpu_ms": _evt_pairs_ms(
                pending.prefill_key_norms_h2d_evt_pairs
            ),
            "prefill_key_norms_delta_gpu_ms": _evt_pairs_ms(
                pending.prefill_key_norms_delta_evt_pairs
            ),
            "prefill_key_norms_pack_gpu_ms": _evt_pairs_ms(
                pending.prefill_key_norms_pack_evt_pairs
            ),
            "prefill_key_norms_delta_total_tokens": int(
                pending.prefill_key_norms_delta_total_tokens
            ),
            "prefill_key_norms_delta_max_tokens": int(
                pending.prefill_key_norms_delta_max_tokens
            ),
            "prefill_key_norms_delta_layers": int(
                pending.prefill_key_norms_delta_layers
            ),
            "prefill_log_s_gpu_ms": _evt_pairs_ms(pending.prefill_log_s_evt_pairs),
            "prefill_log_s_triton_gpu_ms": _evt_pairs_ms(
                pending.prefill_log_s_triton_evt_pairs
            ),
            "prefill_log_s_mask_gpu_ms": _evt_pairs_ms(
                pending.prefill_log_s_mask_evt_pairs
            ),
            "prefill_log_s_cross_gpu_ms": _evt_pairs_ms(
                pending.prefill_log_s_cross_evt_pairs
            ),
            "prefill_topk_gpu_ms": _evt_pairs_ms(pending.prefill_topk_evt_pairs),
            "prefill_preproc_gpu_ms": _evt_pairs_ms(
                pending.prefill_preproc_evt_pairs
            ),
            "prefill_seq_full_gpu_ms": _evt_pairs_ms(
                pending.prefill_seq_full_evt_pairs
            ),
            "prefill_pure_preproc_gpu_ms": _evt_pairs_ms(
                pending.prefill_pure_preproc_evt_pairs
            ),
            "prefill_selector_bounds_gpu_ms": _evt_pairs_ms(
                pending.prefill_selector_bounds_evt_pairs
            ),
            "prefill_selector_pipeline_gpu_ms": _evt_pairs_ms(
                pending.prefill_selector_pipeline_evt_pairs
            ),
            "prefill_rebuild_gpu_ms": _evt_pairs_ms(pending.prefill_rebuild_evt_pairs),
            **_prefill_group_profile_fields(
                pending.prefill_evt0,
                pending.prefill_group_evt_pairs,
            ),
            "prefill_publish_cpu_us": float(pending.prefill_publish_cpu_us),
            "refresh_selector_gpu_ms": _evt_ms(pending.refresh_sel_evt0, pending.refresh_sel_evt1),
            "refresh_rebuild_gpu_ms": _evt_ms(pending.refresh_rebuild_evt0, pending.refresh_rebuild_evt1),
            "refresh_gather_gpu_ms": _evt_ms(pending.refresh_gather_evt0, pending.refresh_gather_evt1),
            "refresh_key_norms_preproc_gpu_ms": _evt_ms(
                pending.refresh_key_norms_preproc_evt0, pending.refresh_key_norms_preproc_evt1
            ),
            "refresh_key_norms_gpu_ms": _evt_ms(pending.refresh_key_norms_evt0, pending.refresh_key_norms_evt1),
            "refresh_key_norms_h2d_gpu_ms": _evt_ms(
                pending.refresh_key_norms_h2d_evt0, pending.refresh_key_norms_h2d_evt1
            ),
            "refresh_key_norms_delta_gpu_ms": _evt_ms(
                pending.refresh_key_norms_delta_evt0, pending.refresh_key_norms_delta_evt1
            ),
            "refresh_key_norms_pack_gpu_ms": _evt_ms(
                pending.refresh_key_norms_pack_evt0, pending.refresh_key_norms_pack_evt1
            ),
            "refresh_key_norms_pre_h2d_gap_gpu_ms": _evt_ms(
                pending.refresh_key_norms_evt0, pending.refresh_key_norms_h2d_evt0
            ),
            "refresh_key_norms_h2d_to_delta_gap_gpu_ms": _evt_ms(
                pending.refresh_key_norms_h2d_evt1,
                pending.refresh_key_norms_delta_evt0,
            ),
            "refresh_key_norms_delta_to_pack_gap_gpu_ms": _evt_ms(
                pending.refresh_key_norms_delta_evt1,
                pending.refresh_key_norms_pack_evt0,
            ),
            "refresh_key_norms_post_pack_gap_gpu_ms": _evt_ms(
                pending.refresh_key_norms_pack_evt1, pending.refresh_key_norms_evt1
            ),
            "refresh_key_norms_delta_total_tokens": int(
                pending.refresh_key_norms_delta_total_tokens
            ),
            "refresh_key_norms_delta_max_tokens": int(
                pending.refresh_key_norms_delta_max_tokens
            ),
            "refresh_key_norms_delta_layers": int(
                pending.refresh_key_norms_delta_layers
            ),
            "refresh_log_s_gpu_ms": _evt_ms(pending.refresh_log_s_evt0, pending.refresh_log_s_evt1),
            "refresh_log_s_triton_gpu_ms": _evt_ms(
                pending.refresh_log_s_triton_evt0, pending.refresh_log_s_triton_evt1
            ),
            "refresh_log_s_mask_gpu_ms": _evt_ms(
                pending.refresh_log_s_mask_evt0, pending.refresh_log_s_mask_evt1
            ),
            "refresh_log_s_cross_gpu_ms": _evt_ms(
                pending.refresh_log_s_cross_evt0, pending.refresh_log_s_cross_evt1
            ),
            "refresh_topk_gpu_ms": _evt_ms(pending.refresh_topk_evt0, pending.refresh_topk_evt1),
            "refresh_preproc_gpu_ms": _evt_ms(pending.refresh_preproc_evt0, pending.refresh_preproc_evt1),
            # 细粒度计时：分离seq_full准备和纯preproc_bounds开销
            "refresh_seq_full_gpu_ms": _evt_ms(
                pending.refresh_seq_full_evt0,
                pending.refresh_seq_full_evt1,
            ),
            "refresh_pure_preproc_gpu_ms": _evt_ms(
                pending.refresh_pure_preproc_evt0,
                pending.refresh_pure_preproc_evt1,
            ),
            "refresh_selector_bounds_gpu_ms": _evt_ms(
                pending.refresh_selector_bounds_evt0,
                pending.refresh_selector_bounds_evt1,
            ),
            "refresh_selector_pipeline_gpu_ms": _evt_ms(
                pending.refresh_selector_pipeline_evt0,
                pending.refresh_selector_pipeline_evt1,
            ),
            **{
                f"async_producer_{stage}_gpu_ms": _evt_pairs_ms(
                    getattr(pending, f"async_producer_{stage}_evt_pairs")
                )
                for stage in ASYNC_PRODUCER_GPU_PROFILE_STAGES
            },
            "rebuild_head_dim": int(pending.rebuild_head_dim or 0),
            "rebuild_kv_dtype": str(pending.rebuild_kv_dtype or ""),
            "rebuild_block_size": int(pending.rebuild_block_size or 0),
            "rebuild_stride_tokens": int(pending.rebuild_stride_tokens or 0),
            "rebuild_selected_k": int(pending.rebuild_selected_k or 0),
            "rebuild_num_kv_heads": int(pending.rebuild_num_kv_heads or 0),
            "rebuild_batch_slots": int(pending.rebuild_batch_slots or 0),
            "capture_kv_len_total": int(pending.capture_kv_len_total or 0),
            "writer_pointer_rebuild_count": int(
                pending.writer_pointer_rebuild_count or 0
            ),
            "writer_pointer_lookup_count": int(
                pending.writer_pointer_lookup_count or 0
            ),
            "writer_cached_pointer_hit_rate": float(
                pending.writer_cached_pointer_hit_rate
            ),
            "writer_cached_pointer_op_count": int(
                pending.writer_cached_pointer_op_count or 0
            ),
            "writer_vector_fallback_count": int(
                pending.writer_vector_fallback_count or 0
            ),
            "writer_kernel_variant": str(pending.writer_kernel_variant or ""),
            "writer_actual_tokens": int(pending.writer_actual_tokens or 0),
            "writer_sink_tokens": int(pending.writer_sink_tokens or 0),
            "writer_persist_tokens": int(pending.writer_persist_tokens or 0),
            "writer_sink_io_bytes": int(pending.writer_sink_io_bytes or 0),
            "writer_persist_io_bytes": int(pending.writer_persist_io_bytes or 0),
            "writer_token_tiles_estimated": int(
                pending.writer_token_tiles_estimated or 0
            ),
            "writer_active_token_tiles_estimated": int(
                pending.writer_active_token_tiles_estimated or 0
            ),
            "writer_cta_count_estimated": int(
                pending.writer_cta_count_estimated or 0
            ),
            "writer_active_cta_count_estimated": int(
                pending.writer_active_cta_count_estimated or 0
            ),
            "writer_tokens_per_cta": int(pending.writer_tokens_per_cta or 0),
            "writer_k_read_bytes": int(pending.writer_k_read_bytes or 0),
            "writer_v_read_bytes": int(pending.writer_v_read_bytes or 0),
            "writer_k_write_bytes": int(pending.writer_k_write_bytes or 0),
            "writer_v_write_bytes": int(pending.writer_v_write_bytes or 0),
            "writer_pos_write_bytes": int(pending.writer_pos_write_bytes or 0),
            "writer_total_io_bytes": int(pending.writer_total_io_bytes or 0),
            "writer_effective_io_gbps": float(pending.writer_effective_io_gbps),
            "selected_indices_materialized_bytes": int(
                pending.selected_indices_materialized_bytes or 0
            ),
            "selected_indices_io_bytes": int(pending.selected_indices_io_bytes or 0),
            "selector_writer_current_path_count": int(
                pending.selector_writer_current_path_count or 0
            ),
            "selector_writer_boundary_cpu_us": float(
                pending.selector_writer_boundary_cpu_us
            ),
            "selected_boundary_lower_bound_ms_per_group": float(
                pending.selected_boundary_lower_bound_ms_per_group
            ),
            "predicted_front_early_step_improvement_ms": float(
                pending.predicted_front_early_step_improvement_ms
            ),
            "residual_fixed_capture_control_ms": float(
                pending.residual_fixed_capture_control_ms
            ),
            "source_ready_recorded_after_pointer_publish_count": int(
                pending.source_ready_recorded_after_pointer_publish_count or 0
            ),
            "lastn1_direct_count": int(pending.lastn1_direct_count or 0),
            "gt1_reduce_count": int(pending.gt1_reduce_count or 0),
            "refresh_rebuild_budget_before": int(
                pending.refresh_rebuild_budget_before
            ),
            "refresh_rebuild_budget_after": int(
                pending.refresh_rebuild_budget_after
            ),
            "refresh_rebuild_enqueued_count": int(
                pending.refresh_rebuild_enqueued_count or 0
            ),
            "refresh_rebuild_inline_count": int(
                pending.refresh_rebuild_inline_count or 0
            ),
            "refresh_rebuild_pending_queue_size": int(
                pending.refresh_rebuild_pending_queue_size or 0
            ),
            "refresh_rebuild_coalesced_count": int(
                pending.refresh_rebuild_coalesced_count or 0
            ),
            "deadline_rebuild_drop_finished_count": int(
                pending.deadline_rebuild_drop_finished_count or 0
            ),
            "deadline_rebuild_drain_finish_count": int(
                pending.deadline_rebuild_drain_finish_count or 0
            ),
            "deadline_rebuild_partial_finish_count": int(
                pending.deadline_rebuild_partial_finish_count or 0
            ),
            "deadline_rebuild_drain_submit_count": int(
                pending.deadline_rebuild_drain_submit_count or 0
            ),
            "deadline_rebuild_drain_submit_decode_step_min": int(
                pending.deadline_rebuild_drain_submit_decode_step_min
            ),
            "deadline_rebuild_drain_submit_decode_step_max": int(
                pending.deadline_rebuild_drain_submit_decode_step_max
            ),
            "deadline_rebuild_drain_submit_decode_steps": list(
                pending.deadline_rebuild_drain_submit_decode_steps or ()
            ),
            "deadline_deferred_selector_compute_count": int(
                pending.deadline_deferred_selector_compute_count or 0
            ),
            "deadline_deferred_selector_compute_cpu_us_total": float(
                pending.deadline_deferred_selector_compute_cpu_us_total
            ),
            "deadline_deferred_selector_compute_cpu_us_max": float(
                pending.deadline_deferred_selector_compute_cpu_us_max
            ),
            "deadline_deferred_selector_inner_compute_cpu_us_total": float(
                pending.deadline_deferred_selector_inner_compute_cpu_us_total
            ),
            "deadline_deferred_selector_inner_compute_cpu_us_max": float(
                pending.deadline_deferred_selector_inner_compute_cpu_us_max
            ),
            "deadline_deferred_selector_stack_cpu_us_total": float(
                pending.deadline_deferred_selector_stack_cpu_us_total
            ),
            "deadline_deferred_selector_stack_cpu_us_max": float(
                pending.deadline_deferred_selector_stack_cpu_us_max
            ),
            "deadline_deferred_selector_validate_cpu_us_total": float(
                pending.deadline_deferred_selector_validate_cpu_us_total
            ),
            "deadline_deferred_selector_validate_cpu_us_max": float(
                pending.deadline_deferred_selector_validate_cpu_us_max
            ),
            "deadline_deferred_selector_key_norms_cpu_us_total": float(
                pending.deadline_deferred_selector_key_norms_cpu_us_total
            ),
            "deadline_deferred_selector_key_norms_cpu_us_max": float(
                pending.deadline_deferred_selector_key_norms_cpu_us_max
            ),
            "deadline_deferred_selector_key_norms_arena_cpu_us_total": float(
                pending.deadline_deferred_selector_key_norms_arena_cpu_us_total
            ),
            "deadline_deferred_selector_key_norms_arena_cpu_us_max": float(
                pending.deadline_deferred_selector_key_norms_arena_cpu_us_max
            ),
            "deadline_deferred_selector_key_norms_direct_cpu_us_total": float(
                pending.deadline_deferred_selector_key_norms_direct_cpu_us_total
            ),
            "deadline_deferred_selector_key_norms_direct_cpu_us_max": float(
                pending.deadline_deferred_selector_key_norms_direct_cpu_us_max
            ),
            "deadline_deferred_selector_key_norms_direct_prepare_cpu_us_total": float(
                pending.deadline_deferred_selector_key_norms_direct_prepare_cpu_us_total
            ),
            "deadline_deferred_selector_key_norms_direct_prepare_cpu_us_max": float(
                pending.deadline_deferred_selector_key_norms_direct_prepare_cpu_us_max
            ),
            "deadline_deferred_selector_key_norms_direct_launch_cpu_us_total": float(
                pending.deadline_deferred_selector_key_norms_direct_launch_cpu_us_total
            ),
            "deadline_deferred_selector_key_norms_direct_launch_cpu_us_max": float(
                pending.deadline_deferred_selector_key_norms_direct_launch_cpu_us_max
            ),
            "deadline_deferred_selector_key_norms_pack_cpu_us_total": float(
                pending.deadline_deferred_selector_key_norms_pack_cpu_us_total
            ),
            "deadline_deferred_selector_key_norms_pack_cpu_us_max": float(
                pending.deadline_deferred_selector_key_norms_pack_cpu_us_max
            ),
            "deadline_deferred_selector_select_cpu_us_total": float(
                pending.deadline_deferred_selector_select_cpu_us_total
            ),
            "deadline_deferred_selector_select_cpu_us_max": float(
                pending.deadline_deferred_selector_select_cpu_us_max
            ),
            "deadline_deferred_selector_post_cpu_us_total": float(
                pending.deadline_deferred_selector_post_cpu_us_total
            ),
            "deadline_deferred_selector_post_cpu_us_max": float(
                pending.deadline_deferred_selector_post_cpu_us_max
            ),
            "deadline_deferred_selector_wrapper_gap_cpu_us_total": float(
                pending.deadline_deferred_selector_wrapper_gap_cpu_us_total
            ),
            "deadline_deferred_selector_wrapper_gap_cpu_us_max": float(
                pending.deadline_deferred_selector_wrapper_gap_cpu_us_max
            ),
            "deadline_deferred_producer_detail_us": {
                str(k): float(v)
                for k, v in (
                    pending.deadline_deferred_producer_detail_us or {}
                ).items()
            },
            "deadline_async_producer_body_count": int(
                pending.deadline_async_producer_body_count or 0
            ),
            "deadline_async_producer_body_cpu_us_total": float(
                pending.deadline_async_producer_body_cpu_us_total
            ),
            "deadline_async_producer_body_cpu_us_max": float(
                pending.deadline_async_producer_body_cpu_us_max
            ),
            "deadline_async_producer_selector_count": int(
                pending.deadline_async_producer_selector_count or 0
            ),
            "deadline_async_producer_selector_cpu_us_total": float(
                pending.deadline_async_producer_selector_cpu_us_total
            ),
            "deadline_async_producer_selector_cpu_us_max": float(
                pending.deadline_async_producer_selector_cpu_us_max
            ),
            "deadline_async_producer_key_norms_delta_count": int(
                pending.deadline_async_producer_key_norms_delta_count or 0
            ),
            "deadline_async_producer_key_norms_delta_total_tokens_total": int(
                pending.deadline_async_producer_key_norms_delta_total_tokens_total
                or 0
            ),
            "deadline_async_producer_key_norms_delta_max_tokens_max": int(
                pending.deadline_async_producer_key_norms_delta_max_tokens_max
            ),
            "deadline_async_producer_key_norms_delta_layers_total": int(
                pending.deadline_async_producer_key_norms_delta_layers_total or 0
            ),
            "deadline_async_producer_writer_count": int(
                pending.deadline_async_producer_writer_count or 0
            ),
            "deadline_async_producer_writer_cpu_us_total": float(
                pending.deadline_async_producer_writer_cpu_us_total
            ),
            "deadline_async_producer_writer_cpu_us_max": float(
                pending.deadline_async_producer_writer_cpu_us_max
            ),
            "deadline_async_producer_graph_replay_count": int(
                pending.deadline_async_producer_graph_replay_count or 0
            ),
            "deadline_async_producer_graph_replay_cpu_us_total": float(
                pending.deadline_async_producer_graph_replay_cpu_us_total
            ),
            "deadline_async_producer_graph_replay_cpu_us_max": float(
                pending.deadline_async_producer_graph_replay_cpu_us_max
            ),
            "deadline_async_producer_graph_replay_stage_selector_inputs_count": int(
                pending.deadline_async_producer_graph_replay_stage_selector_inputs_count
                or 0
            ),
            "deadline_async_producer_graph_replay_stage_selector_inputs_cpu_us_total": float(
                pending.deadline_async_producer_graph_replay_stage_selector_inputs_cpu_us_total
            ),
            "deadline_async_producer_graph_replay_stage_selector_inputs_cpu_us_max": float(
                pending.deadline_async_producer_graph_replay_stage_selector_inputs_cpu_us_max
            ),
            "deadline_async_producer_graph_replay_prepare_writer_count": int(
                pending.deadline_async_producer_graph_replay_prepare_writer_count or 0
            ),
            "deadline_async_producer_graph_replay_prepare_writer_cpu_us_total": float(
                pending.deadline_async_producer_graph_replay_prepare_writer_cpu_us_total
            ),
            "deadline_async_producer_graph_replay_prepare_writer_cpu_us_max": float(
                pending.deadline_async_producer_graph_replay_prepare_writer_cpu_us_max
            ),
            "deadline_async_producer_graph_replay_stage_lens_count": int(
                pending.deadline_async_producer_graph_replay_stage_lens_count or 0
            ),
            "deadline_async_producer_graph_replay_stage_lens_cpu_us_total": float(
                pending.deadline_async_producer_graph_replay_stage_lens_cpu_us_total
            ),
            "deadline_async_producer_graph_replay_stage_lens_cpu_us_max": float(
                pending.deadline_async_producer_graph_replay_stage_lens_cpu_us_max
            ),
            "deadline_async_producer_graph_replay_prepare_events_count": int(
                pending.deadline_async_producer_graph_replay_prepare_events_count or 0
            ),
            "deadline_async_producer_graph_replay_prepare_events_cpu_us_total": float(
                pending.deadline_async_producer_graph_replay_prepare_events_cpu_us_total
            ),
            "deadline_async_producer_graph_replay_prepare_events_cpu_us_max": float(
                pending.deadline_async_producer_graph_replay_prepare_events_cpu_us_max
            ),
            "deadline_async_producer_graph_replay_graph_count": int(
                pending.deadline_async_producer_graph_replay_graph_count or 0
            ),
            "deadline_async_producer_graph_replay_graph_cpu_us_total": float(
                pending.deadline_async_producer_graph_replay_graph_cpu_us_total
            ),
            "deadline_async_producer_graph_replay_graph_cpu_us_max": float(
                pending.deadline_async_producer_graph_replay_graph_cpu_us_max
            ),
            "deadline_async_producer_graph_capture_count": int(
                pending.deadline_async_producer_graph_capture_count or 0
            ),
            "deadline_async_producer_graph_capture_cpu_us_total": float(
                pending.deadline_async_producer_graph_capture_cpu_us_total
            ),
            "deadline_async_producer_graph_capture_cpu_us_max": float(
                pending.deadline_async_producer_graph_capture_cpu_us_max
            ),
            "deadline_async_producer_result_precomputed_count": int(
                pending.deadline_async_producer_result_precomputed_count or 0
            ),
            "producer_work_target_layer_start": int(
                pending.producer_work_target_layer_start
            ),
            "producer_work_target_layer_end": int(
                pending.producer_work_target_layer_end
            ),
            "producer_work_decode_step_min": int(
                pending.producer_work_decode_step_min
            ),
            "producer_work_decode_step_max": int(
                pending.producer_work_decode_step_max
            ),
            "producer_work_ready_epoch": int(pending.producer_work_ready_epoch),
            "producer_work_deadline_epoch": int(
                pending.producer_work_deadline_epoch
            ),
            "producer_work_deadline_handle_id": int(
                pending.producer_work_deadline_handle_id
            ),
            "producer_work_deadline_slack_steps": int(
                pending.producer_work_deadline_slack_steps
            ),
            "producer_work_can_drop": int(pending.producer_work_can_drop or 0),
            "producer_work_can_coalesce": int(
                pending.producer_work_can_coalesce or 0
            ),
            "producer_work_admission_reason": str(
                pending.producer_work_admission_reason or ""
            ),
            "refresh_overlap_ratio": (
                float(pending.refresh_overlap_ratio)
                if pending.refresh_overlap_ratio is not None
                else None
            ),
            "refresh_overlap_new_k": (
                int(pending.refresh_overlap_new_k)
                if pending.refresh_overlap_new_k is not None
                else None
            ),
            "refresh_overlap_old_k": (
                int(pending.refresh_overlap_old_k)
                if pending.refresh_overlap_old_k is not None
                else None
            ),
        }
        # [SELECTED-OUT-RING v2] graph 接管判据仪器:selector topk graph 的
        # replay/capture 累计与环的 acquire/spill 计数(附加字段,消费端按
        # 键读不受影响)。
        record["selector_topk_graph_replay_count"] = int(
            getattr(self, "_selector_topk_graph_replay_count", 0)
        )
        record["selector_topk_graph_capture_count"] = int(
            getattr(self, "_selector_topk_graph_capture_count", 0)
        )
        record["selector_topk_graph_scope_replacement_count"] = int(
            getattr(self, "_selector_topk_graph_scope_replacement_count", 0)
        )
        record["selector_topk_graph_admission_deferred_count"] = int(
            getattr(self, "_selector_topk_graph_admission_deferred_count", 0)
        )
        _sor = getattr(self, "_selected_out_ring", None)
        record["selected_out_ring_acquire_count"] = (
            int(_sor.acquire_count) if _sor is not None else 0
        )
        record["selected_out_ring_spill_count"] = (
            int(_sor.spill_run_count) if _sor is not None else 0
        )
        try:
            payload = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            _log.warning("Failed to serialize refresh profile record to JSON", exc_info=True)
            raise
        self._refresh_profile_pending_by_buf[buf] = None
        self._refresh_profile_write(f"{os.getpid()}\trefresh.flush\t{payload}")
