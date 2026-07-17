"""
patches/controller_mixins/wait_decider_mixin.py — Wait decision logic, pending-work tracking, and bootstrap completion.

OWNS:
  - _init_wait_decider_state(): wait decider state initialization
  - _pending_work_reset / _pending_work_mark_submitted / _pending_work_clear_buf(): pending-work ledger
  - _pending_work_for_buf / _pending_work_blockers(): pending-work queries
  - _compute_wait_decision(): main/stream wait arbitration
  - _main_stream_wait_for_chunk_done(): CUDA stream synchronization
  - _consume_step_wait_token(): per-step wait token consumption

DEPENDS_ON:
  - patches.refresh_runtime.entry.run_refresh_step
  - patches.runtime_contracts.ExecutionBackendLedger
  - patches.sparse_constants (_CAPTURE_CHUNK, _CAPTURE_IN_FLIGHT, _DYNAMIC_ENV)
  - patches.sparse_utils._is_stream_capturing_or_raise
  - CaptureRingMixin._reclaim_retired_buffers, ProfileMixin._refresh_profile_try_flush_pending

ENTRY_POINTS:
  - _init_wait_decider_state(): called from VLLMSparseController.__init__
"""
from __future__ import annotations

import logging
import os
import time
import atexit
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import torch

from patches.refresh_runtime.entry import run_refresh_step
from patches.runtime_contracts import ExecutionBackendLedger
from patches.sparse_constants import (
    _CAPTURE_CHUNK,
    _CAPTURE_IN_FLIGHT,
    _DYNAMIC_ENV,
    _ONE_SHOT_ASYNC_BOOTSTRAP_CACHED,
)
from patches.sparse_utils import _is_stream_capturing_or_raise

_log = logging.getLogger(__name__)

_DeferredProducerJobSnapshot = Tuple[Tuple[str, int, int, object], ...]
_DeferredProducerLaunchIntent = Tuple[
    int,
    Tuple[str, ...],
    bool,
    _DeferredProducerJobSnapshot,
]

_DEFER_BOOTSTRAP_PRODUCER_CACHED = (
    os.environ.get("VLLM_SPARSE_DEFER_BOOTSTRAP_PRODUCER", "0") == "1"
)



def _bootstrap_full_kv_handoff_trace_fields() -> Dict[str, bool]:
    fields: Dict[str, bool] = {"bootstrap_full_kv_handoff": True}
    if os.environ.get("VLLM_SPARSE_DEFERRED_BRIDGE_DIAGNOSTIC", "0") == "1":
        fields["deferred_bridge_diagnostic_only"] = True
    return fields




# VLLM_SPARSE_REFRESH_CP_PROBE — passive critical-path wait probe (default OFF).
_CP_PROBE_ENABLED: bool = os.environ.get("VLLM_SPARSE_REFRESH_CP_PROBE", "") == "1"
_CP_PROBE_WAIT_COUNTS: Dict[str, Dict[str, int]] = {}
_CP_PROBE_ATEXIT_DONE: bool = False


def _cp_probe_flush() -> None:
    try:
        print(
            "[CP_PROBE] pid=" + str(os.getpid())
            + " wait_query_counts=" + str(_CP_PROBE_WAIT_COUNTS),
            flush=True,
        )
    except Exception:
        pass


def _cp_probe_record_wait(evt, site: str) -> None:
    # Non-blocking evt.query() BEFORE the stream-ordered wait_event:
    #   ready=already-signalled => the awaited GPU work overlapped (wait is free)
    #   not_ready               => decode genuinely stalls behind it (real cost)
    # In-memory accumulate + atexit flush ONCE; never a per-step NAS write.
    global _CP_PROBE_ATEXIT_DONE
    try:
        ready = bool(evt.query())
    except Exception:
        ready = None
    d = _CP_PROBE_WAIT_COUNTS.setdefault(
        str(site), {"ready": 0, "not_ready": 0, "err": 0}
    )
    if ready is True:
        d["ready"] += 1
    elif ready is False:
        d["not_ready"] += 1
    else:
        d["err"] += 1
    if not _CP_PROBE_ATEXIT_DONE:
        atexit.register(_cp_probe_flush)
        _CP_PROBE_ATEXIT_DONE = True


class WaitDeciderMixin:
    """Wait decision, pending-work ledger, and bootstrap completion tracking.

    All wait-decider state is initialised via ``_init_wait_decider_state()``
    which must be called from the controller's ``__init__``.
    """

    _MIXIN_REQUIRES: tuple = ("RefreshRebuildMixin", "ProfileMixin", "CaptureRingMixin")

    # ------------------------------------------------------------------
    # State initialisation
    # ------------------------------------------------------------------

    def _init_wait_decider_state(self) -> None:
        # split-wait: per-buf pending work type flags (bit0=prefill, bit1=refresh)
        self._pending_ledger = ExecutionBackendLedger(num_bufs=int(_CAPTURE_IN_FLIGHT))
        self._buf_pending_work_flags: List[int] = self._pending_ledger.flags
        self._buf_pending_work_epoch: List[int] = self._pending_ledger.epoch
        # wait policy: chunk (wait chunk_done) | split (wait per-flag events)
        self._wait_policy_cached: Optional[str] = None
        # main_stream wait dedup: per step per buf only wait on new chunk once
        self._main_wait_epoch: int = -1
        self._main_wait_chunk_id_by_buf: List[int] = [-1 for _ in range(int(_CAPTURE_IN_FLIGHT))]
        # hints-driven wait dedup: per step per buf only trigger once
        self._step_wait_consumed_token_by_buf: List[int] = [0 for _ in range(int(_CAPTURE_IN_FLIGHT))]
        self._step_has_compact_consumer_cache_key: Tuple[int, int, int] | None = None
        self._step_has_compact_consumer_cache_value: bool = False
        # bootstrap (prefill selector+compact build) completion confirmation
        self._bootstrap_pending_request_ids: Set[str] = set()
        # Deterministic host-side transition ledger. Exact capture-plan
        # finalizers are bound to their producer epoch; steady decode pays one
        # empty-dict check instead of rescanning the scheduler batch.
        self._bootstrap_submission_boundary_pending_epoch_by_id: Dict[str, int] = {}
        # one-shot group-ready graph waits use stable event handles. The graph
        # captures waits on these handles once; each prefill producer group
        # records the same handle for the current request before replay.
        self._one_shot_group_done_evt_by_group: Dict[int, torch.cuda.Event] = {}
        self._one_shot_group_done_evt_by_group_slot: Dict[
            Tuple[int, int],
            torch.cuda.Event,
        ] = {}
        self._one_shot_group_done_evt_serial_by_group: Dict[int, int] = {}
        self._one_shot_group_done_evt_serial_by_group_slot: Dict[
            Tuple[int, int],
            int,
        ] = {}
        self._one_shot_group_ready_wait_epoch_by_group: Dict[int, int] = {}
        self._one_shot_group_ready_wait_epoch_by_group_slot: Dict[
            Tuple[int, int],
            int,
        ] = {}
        self._one_shot_group_ready_wait_serial_by_group_slot: Dict[
            Tuple[int, int],
            int,
        ] = {}
        self._one_shot_group_ready_full_graph_wait_required: bool = False
        self._one_shot_group_ready_compact_slots_cache_key: (
            Tuple[int, int, int, int, int] | None
        ) = None
        self._one_shot_group_ready_compact_slots_cache_value: Tuple[int, ...] = (
            tuple()
        )
        # Step-boundary code only stages immutable launch intents.  The sole
        # execution owner is the post-model-forward hook, which atomically
        # detaches and drains this queue after anchor kernels are submitted.
        self._deferred_bootstrap_launch_intents: List[
            _DeferredProducerLaunchIntent
        ] = []
        self._deferred_bootstrap_launch_intent_seen_epoch: int = -1
        self._deferred_bootstrap_launch_intent_identities: Set[
            Tuple[int, Tuple[str, ...], bool]
        ] = set()

    # ------------------------------------------------------------------
    # Wait policy
    # ------------------------------------------------------------------

    def _get_wait_policy(self) -> str:
        """Return wait policy name: 'chunk' or 'split'."""
        if self._wait_policy_cached is None:
            self._wait_policy_cached = "split" if self._async_refresh_enabled() else "chunk"
        return str(self._wait_policy_cached)

    # ------------------------------------------------------------------
    # Pending work ledger
    # ------------------------------------------------------------------

    def _pending_work_reset(self) -> None:
        self._pending_ledger = ExecutionBackendLedger(num_bufs=int(_CAPTURE_IN_FLIGHT))
        self._buf_pending_work_flags = self._pending_ledger.flags
        self._buf_pending_work_epoch = self._pending_ledger.epoch

    def _pending_work_mark_submitted(
        self,
        *,
        buf_id: int,
        kind: str,
        async_mode: bool,
        epoch: Optional[int] = None,
    ) -> None:
        ep = int(self.step_context_epoch if epoch is None else epoch)
        self._pending_ledger.mark_submitted(
            buf_id=int(buf_id),
            epoch=ep,
            kind=str(kind),
            async_mode=bool(async_mode),
        )

    def _pending_work_clear_buf(self, *, buf_id: int) -> None:
        self._pending_ledger.clear(buf_id=int(buf_id))

    def _pending_work_for_buf(self, *, buf_id: int) -> bool:
        """Return whether the given buf has any pending async work (epoch-agnostic)."""
        buf = buf_id % _CAPTURE_IN_FLIGHT
        flags = self._buf_pending_work_flags
        if 0 <= buf < len(flags) and flags[buf] != 0:
            return True
        return False

    def _request_can_bridge_bootstrap_decode(self, rid: str) -> bool:
        tracking = self.request_states.get(str(rid))
        if tracking is None:
            return False
        if not bool(getattr(tracking, "bootstrap_pending", False)):
            return False
        if not bool(getattr(tracking, "bootstrap_bridge_active", False)):
            return False
        if bool(getattr(tracking, "bootstrap_done", False)):
            return False
        max_tokens = int(getattr(tracking, "bridge_max_tokens", 0) or 0)
        if max_tokens <= 0:
            return False
        used_tokens = int(getattr(tracking, "bridge_token_count", 0) or 0)
        if used_tokens < max_tokens:
            return True
        if (
            getattr(tracking, "deferred_producer_job", None) is not None
            or getattr(tracking, "producer_ready_state", None) is not None
        ):
            return True
        return not WaitDeciderMixin._bootstrap_producer_publishable_for_bridge(
            tracking=tracking,
        )

    def _request_bridge_token_budget_remaining(self, rid: str) -> bool:
        tracking = self.request_states.get(str(rid))
        if tracking is None:
            return False
        if not bool(getattr(tracking, "bootstrap_pending", False)):
            return False
        if not bool(getattr(tracking, "bootstrap_bridge_active", False)):
            return False
        if bool(getattr(tracking, "bootstrap_done", False)):
            return False
        max_tokens = int(getattr(tracking, "bridge_max_tokens", 0) or 0)
        if max_tokens <= 0:
            return False
        used_tokens = int(getattr(tracking, "bridge_token_count", 0) or 0)
        if used_tokens < max_tokens:
            return True
        return not WaitDeciderMixin._bootstrap_producer_publishable_for_bridge(
            tracking=tracking,
        )

    def _mark_bridge_decode_metadata_accepted(
        self,
        *,
        req_ids: Tuple[str, ...],
        epoch: int,
    ) -> int:
        accepted = 0
        ep = int(epoch)
        for rid in req_ids:
            tracking = self.request_states.get(str(rid))
            if tracking is None:
                continue
            if int(getattr(tracking, "bridge_last_counted_epoch", -1)) == ep:
                continue
            bridge_token_count = int(getattr(tracking, "bridge_token_count", 0) or 0) + 1
            tracking.bridge_token_count = bridge_token_count
            tracking.bridge_last_counted_epoch = ep
            prompt_tokens = int(getattr(tracking, "total_prompt_tokens", 0) or 0)
            min_position = (
                prompt_tokens + bridge_token_count - 1
                if prompt_tokens > 0
                else bridge_token_count - 1
            )
            bridge_token_position = int(getattr(tracking, "last_seq_len", -1) or -1)
            if bridge_token_position < min_position:
                bridge_token_position = int(min_position)
            bridge_positions_obj = getattr(tracking, "bridge_token_positions", None)
            if isinstance(bridge_positions_obj, list):
                bridge_positions = bridge_positions_obj
            else:
                bridge_positions = []
                tracking.bridge_token_positions = bridge_positions
            bridge_positions.append(int(bridge_token_position))
            bridge_positions_tuple = tuple(int(pos) for pos in bridge_positions)
            producer_launch_step = int(getattr(tracking, "producer_launch_step", -1))
            if os.environ.get("VLLM_SPARSE_ONE_SHOT_TIMELINE_LOG", ""):
                try:
                    from patches.refresh_runtime.one_shot_timeline import (
                        append_one_shot_timeline,
                        timeline_log_path,
                    )

                    append_one_shot_timeline(
                        path=timeline_log_path(),
                        request_id=str(rid),
                        epoch=ep,
                        chunk_id=-1,
                        buffer_id=-1,
                        phase="bridge_decode_metadata_accepted",
                        timestamp_ns=time.perf_counter_ns(),
                        duration_us=0.0,
                        aux_stream_enabled=True,
                        extra_fields={
                            "bridge_token_count": int(bridge_token_count),
                            "bridge_token_position": int(bridge_token_position),
                            "bridge_token_positions": list(bridge_positions_tuple),
                            "producer_launch_step": int(producer_launch_step),
                            **_bootstrap_full_kv_handoff_trace_fields(),
                        },
                    )
                except Exception:
                    _log.debug("failed to append bridge decode timeline", exc_info=True)
            if os.environ.get("VLLM_SPARSE_FA3_ROUTE_TRACE_LOG", ""):
                try:
                    from patches.fa3_native.install import append_fa3_route_trace

                    append_fa3_route_trace(
                        {
                            "event": "deferred_bridge_decode_metadata_accepted",
                            "phase": "bridge_phase",
                            "request_id": str(rid),
                            "epoch": ep,
                            "bridge_token_count": int(bridge_token_count),
                            "bridge_token_position": int(bridge_token_position),
                            "bridge_token_positions": list(bridge_positions_tuple),
                            "producer_launch_step": int(producer_launch_step),
                            "accepted_count": 1,
                            **_bootstrap_full_kv_handoff_trace_fields(),
                        }
                    )
                except Exception:
                    _log.debug("failed to append bridge route trace", exc_info=True)
            accepted += 1
        return accepted

    def _run_deferred_bootstrap_producer_job(
        self,
        job: object,
        *,
        max_groups_per_call: int = 0,
    ) -> bool:
        from patches.refresh_runtime.deferred_producer import (
            run_deferred_bootstrap_producer_job,
        )

        return bool(
            run_deferred_bootstrap_producer_job(
                self,
                job,
                capture_chunk=int(_CAPTURE_CHUNK),
                max_groups_per_call=int(max_groups_per_call),
            )
        )

    @staticmethod
    def _deferred_producer_groups_per_step() -> int:
        raw = os.environ.get("VLLM_SPARSE_DEFERRED_PRODUCER_GROUPS_PER_STEP", "0")
        try:
            value = int(raw)
        except ValueError as exc:
            raise RuntimeError(
                "VLLM_SPARSE_DEFERRED_PRODUCER_GROUPS_PER_STEP must be an integer, "
                f"got {raw!r}"
            ) from exc
        if value < -1:
            raise RuntimeError(
                "VLLM_SPARSE_DEFERRED_PRODUCER_GROUPS_PER_STEP must be >= -1, "
                f"got {raw!r}"
            )
        return int(value)

    @staticmethod
    def _adaptive_deferred_producer_group_budget(
        *,
        tracking: object,
        job: object,
        used_bridge_tokens: Optional[int] = None,
    ) -> int:
        payload_groups = tuple(getattr(job, "payload_groups", tuple()) or tuple())
        total_groups = len(payload_groups)
        next_group = int(getattr(job, "next_payload_group_index", 0) or 0)
        remaining_groups = max(0, int(total_groups) - int(next_group))
        if remaining_groups <= 0:
            return 0
        bridge_max_tokens = int(
            getattr(job, "bridge_max_tokens", getattr(tracking, "bridge_max_tokens", 0))
            or 0
        )
        if used_bridge_tokens is None:
            used_bridge_tokens = int(
                getattr(tracking, "bridge_token_count", 0) or 0
            )
        else:
            used_bridge_tokens = int(used_bridge_tokens)
        remaining_bridge_steps = max(1, int(bridge_max_tokens) - int(used_bridge_tokens))
        return max(
            1,
            min(
                int(remaining_groups),
                (int(remaining_groups) + int(remaining_bridge_steps) - 1)
                // int(remaining_bridge_steps),
            ),
        )

    def _adaptive_deferred_producer_groups_per_step(
        self,
        *,
        bridge_token_count_by_request: Optional[Dict[str, int]] = None,
    ) -> int:
        budget = 0
        for rid, tracking in list(self.request_states.items()):
            if (
                bridge_token_count_by_request is not None
                and str(rid) not in bridge_token_count_by_request
            ):
                continue
            job = getattr(tracking, "deferred_producer_job", None)
            if job is None or bool(getattr(job, "completed", False)):
                continue
            if bool(getattr(job, "cancelled", False)) or str(
                getattr(job, "failure_reason", "") or ""
            ):
                continue
            budget = max(
                int(budget),
                int(
                    self._adaptive_deferred_producer_group_budget(
                        tracking=tracking,
                        job=job,
                        used_bridge_tokens=(
                            bridge_token_count_by_request.get(str(rid))
                            if bridge_token_count_by_request is not None
                            else None
                        ),
                    )
                ),
            )
        return int(budget)

    @staticmethod
    def _bridge_safe_deferred_producer_group_budget(
        *,
        tracking: object,
        job: object,
        requested_groups: int,
        used_bridge_tokens: Optional[int] = None,
    ) -> int:
        payload_groups = tuple(getattr(job, "payload_groups", tuple()) or tuple())
        total_groups = len(payload_groups)
        next_group = int(getattr(job, "next_payload_group_index", 0) or 0)
        remaining_groups = max(0, int(total_groups) - int(next_group))
        if remaining_groups <= 0:
            return 0

        requested = int(requested_groups)
        if requested <= 0 or requested > remaining_groups:
            requested = remaining_groups

        bridge_max_tokens = int(
            getattr(job, "bridge_max_tokens", getattr(tracking, "bridge_max_tokens", 0))
            or 0
        )
        if used_bridge_tokens is None:
            used_bridge_tokens = int(
                getattr(tracking, "bridge_token_count", 0) or 0
            )
        else:
            used_bridge_tokens = int(used_bridge_tokens)
        if bridge_max_tokens > 1 and used_bridge_tokens < bridge_max_tokens - 1:
            # Keep the final publish event hidden until bridge positions are stable.
            requested = min(requested, max(0, remaining_groups - 1))
        return max(0, int(requested))

    def _record_deferred_producer_launch_state(
        self,
        *,
        tracking: object,
        job: object,
        epoch: int,
        first_launch: bool,
    ) -> None:
        if hasattr(job, "launched"):
            setattr(job, "launched", True)
        if first_launch:
            if hasattr(job, "launched_epoch"):
                setattr(job, "launched_epoch", int(epoch))
            setattr(tracking, "producer_launch_step", int(epoch))
        if hasattr(job, "last_launch_epoch"):
            setattr(job, "last_launch_epoch", int(epoch))

    def _append_deferred_producer_launch_timeline(
        self,
        *,
        rid: str,
        tracking: object,
        job: object,
        epoch: int,
    ) -> None:
        if not os.environ.get("VLLM_SPARSE_ONE_SHOT_TIMELINE_LOG", ""):
            return
        try:
            from patches.refresh_runtime.deferred_producer import (
                deferred_producer_job_summary,
            )
            from patches.refresh_runtime.one_shot_timeline import (
                append_one_shot_timeline,
                timeline_log_path,
            )

            extra = deferred_producer_job_summary(job)
            extra["producer_launch_step"] = int(
                getattr(tracking, "producer_launch_step", -1)
            )
            extra.update(_bootstrap_full_kv_handoff_trace_fields())
            append_one_shot_timeline(
                path=timeline_log_path(),
                request_id=str(rid),
                epoch=int(epoch),
                chunk_id=-1,
                buffer_id=-1,
                phase="deferred_producer_launch",
                timestamp_ns=time.perf_counter_ns(),
                duration_us=0.0,
                aux_stream_enabled=True,
                extra_fields=extra,
            )
        except Exception:
            _log.debug(
                "failed to append deferred producer launch timeline",
                exc_info=True,
            )

    def _run_and_record_deferred_bootstrap_producer_job(
        self,
        *,
        rid: str,
        tracking: object,
        job: object,
        epoch: int,
        max_groups_per_call: int,
    ) -> bool:
        launch = getattr(self, "_run_deferred_bootstrap_producer_job", None)
        if not callable(launch):
            raise RuntimeError("deferred bootstrap producer launcher is missing")
        first_launch = not bool(getattr(job, "launched", False))
        before_group_index = int(getattr(job, "next_payload_group_index", 0) or 0)
        try:
            completed = bool(
                launch(
                    job,
                    max_groups_per_call=int(max_groups_per_call),
                )
            )
        except Exception as exc:
            if not str(getattr(job, "failure_reason", "") or ""):
                setattr(job, "failure_reason", str(exc))
            raise
        after_group_index = int(getattr(job, "next_payload_group_index", 0) or 0)
        self._record_deferred_producer_launch_state(
            tracking=tracking,
            job=job,
            epoch=int(epoch),
            first_launch=first_launch,
        )
        self._append_deferred_producer_launch_timeline(
            rid=str(rid),
            tracking=tracking,
            job=job,
            epoch=int(epoch),
        )
        return bool(completed)

    def _stage_deferred_bootstrap_producer_jobs(
        self,
        *,
        epoch: int,
        only_request_ids: Tuple[str, ...] = (),
        allow_same_epoch: bool = False,
    ) -> int:
        """Stage one deterministic post-forward producer launch intent.

        The bridge-token snapshot preserves the exact budget decision that the
        former inline call observed.  Intent identity intentionally excludes
        that snapshot: identical same-step prepare reentry is idempotent and
        must not submit another producer group.
        """
        ep = int(epoch)
        only_ids = tuple(dict.fromkeys(str(rid) for rid in only_request_ids))
        identity = (ep, only_ids, bool(allow_same_epoch))
        intents_obj = getattr(self, "_deferred_bootstrap_launch_intents", None)
        if intents_obj is None:
            intents_obj = []
            self._deferred_bootstrap_launch_intents = intents_obj
        if not isinstance(intents_obj, list):
            raise RuntimeError("deferred producer launch intent queue is invalid")
        for existing in intents_obj:
            if int(existing[0]) != ep:
                raise RuntimeError(
                    "deferred producer launch intent epoch drift: "
                    f"staged={int(existing[0])} current={ep}"
                )
        seen_epoch = int(
            getattr(self, "_deferred_bootstrap_launch_intent_seen_epoch", -1)
        )
        seen_identities = getattr(
            self,
            "_deferred_bootstrap_launch_intent_identities",
            None,
        )
        if seen_epoch != ep:
            seen_identities = set()
            self._deferred_bootstrap_launch_intent_seen_epoch = ep
            self._deferred_bootstrap_launch_intent_identities = seen_identities
        if not isinstance(seen_identities, set):
            raise RuntimeError("deferred producer launch intent identity set is invalid")
        if identity in seen_identities:
            return 0
        job_snapshot = tuple(
            (
                str(rid),
                int(getattr(tracking, "bridge_token_count", 0) or 0),
                int(getattr(job, "producer_job_epoch", -1)),
                job,
            )
            for rid, tracking in self.request_states.items()
            if (job := getattr(tracking, "deferred_producer_job", None)) is not None
        )
        intents_obj.append(
            (ep, only_ids, bool(allow_same_epoch), job_snapshot)
        )
        seen_identities.add(identity)
        return 1

    def _fail_deferred_bootstrap_launch_intents(
        self,
        intents: Sequence[_DeferredProducerLaunchIntent],
        *,
        reason: str,
    ) -> None:
        """Retire staged work terminally; never replay it on another forward."""
        for intent in tuple(intents):
            ep = int(intent[0])
            only_ids = set(str(rid) for rid in tuple(intent[1]))
            for rid, _, job_epoch, staged_job in tuple(intent[3]):
                if only_ids and str(rid) not in only_ids:
                    continue
                if bool(getattr(staged_job, "completed", False)):
                    continue
                if int(job_epoch) >= ep:
                    continue
                if not str(getattr(staged_job, "failure_reason", "") or ""):
                    setattr(staged_job, "failure_reason", str(reason))

    def _discard_staged_deferred_bootstrap_producer_jobs(
        self,
        *,
        reason: str,
    ) -> int:
        intents = tuple(
            getattr(self, "_deferred_bootstrap_launch_intents", tuple()) or tuple()
        )
        self._deferred_bootstrap_launch_intents = []
        if intents:
            self._fail_deferred_bootstrap_launch_intents(
                intents,
                reason=str(reason),
            )
        return len(intents)

    def _drain_staged_deferred_bootstrap_producer_jobs(
        self,
        *,
        epoch: int,
    ) -> int:
        """Atomically detach and execute this forward's staged launch intents."""
        ep = int(epoch)
        intents = tuple(
            getattr(self, "_deferred_bootstrap_launch_intents", tuple()) or tuple()
        )
        self._deferred_bootstrap_launch_intents = []
        if not intents:
            return 0
        drifted = tuple(intent for intent in intents if int(intent[0]) != ep)
        if drifted:
            reason = (
                "deferred producer post-forward drain epoch drift: "
                f"staged={tuple(int(intent[0]) for intent in intents)!r} current={ep}"
            )
            self._fail_deferred_bootstrap_launch_intents(intents, reason=reason)
            raise RuntimeError(reason)
        launched = 0
        try:
            for intent_epoch, only_ids, allow_same, job_snapshot in intents:
                launched += self._launch_deferred_bootstrap_producer_jobs(
                    epoch=int(intent_epoch),
                    only_request_ids=tuple(only_ids),
                    allow_same_epoch=bool(allow_same),
                    deferred_job_snapshot=tuple(job_snapshot),
                    enforce_staged_job_snapshot=True,
                )
        except Exception as exc:
            self._fail_deferred_bootstrap_launch_intents(
                intents,
                reason=f"deferred producer post-forward drain failed: {exc}",
            )
            raise
        return int(launched)

    def _launch_deferred_bootstrap_producer_jobs(
        self,
        *,
        epoch: int,
        only_request_ids: Tuple[str, ...] = (),
        allow_same_epoch: bool = False,
        deferred_job_snapshot: _DeferredProducerJobSnapshot = (),
        enforce_staged_job_snapshot: bool = False,
    ) -> int:
        launched = 0
        only_ids = {str(rid) for rid in tuple(only_request_ids or tuple())}
        staged_job_by_request = (
            {
                str(rid): (int(count), int(job_epoch), staged_job)
                for rid, count, job_epoch, staged_job in deferred_job_snapshot
            }
            if enforce_staged_job_snapshot
            else None
        )
        bridge_token_count_by_request = (
            {
                rid: int(snapshot[0])
                for rid, snapshot in staged_job_by_request.items()
            }
            if staged_job_by_request is not None
            else None
        )
        groups_per_step = self._deferred_producer_groups_per_step()
        adaptive_budget = int(groups_per_step) < 0
        remaining_group_budget = (
            self._adaptive_deferred_producer_groups_per_step(
                bridge_token_count_by_request=bridge_token_count_by_request,
            )
            if adaptive_budget
            else int(groups_per_step)
        )
        for rid, tracking in list(self.request_states.items()):
            if (adaptive_budget or groups_per_step > 0) and remaining_group_budget <= 0:
                break
            if only_ids and str(rid) not in only_ids:
                continue
            if (
                staged_job_by_request is not None
                and str(rid) not in staged_job_by_request
            ):
                continue
            job = getattr(tracking, "deferred_producer_job", None)
            if job is None:
                continue
            if staged_job_by_request is not None:
                _, staged_job_epoch, staged_job = staged_job_by_request[str(rid)]
                if (
                    job is not staged_job
                    or int(getattr(job, "producer_job_epoch", -1))
                    != int(staged_job_epoch)
                ):
                    raise RuntimeError(
                        "deferred producer staged job identity drift: "
                        f"request_id={str(rid)!r} "
                        f"staged_epoch={int(staged_job_epoch)} "
                        f"current_epoch={int(getattr(job, 'producer_job_epoch', -1))}"
                    )
            if bool(getattr(job, "completed", False)):
                continue
            if (
                bool(getattr(job, "launched", False))
                and not hasattr(job, "next_payload_group_index")
            ):
                continue
            if (
                not bool(allow_same_epoch)
                and int(getattr(job, "last_launch_epoch", -1)) == int(epoch)
            ):
                continue
            if int(getattr(job, "producer_job_epoch", -1)) >= int(epoch):
                continue
            if bool(getattr(job, "cancelled", False)):
                raise RuntimeError("deferred producer job was cancelled")
            failure_reason = str(getattr(job, "failure_reason", "") or "")
            if failure_reason:
                raise RuntimeError(f"deferred producer job failed: {failure_reason}")
            before_group_index = int(getattr(job, "next_payload_group_index", 0) or 0)
            payload_groups = tuple(getattr(job, "payload_groups", tuple()) or tuple())
            remaining_groups = max(0, len(payload_groups) - int(before_group_index))
            requested_group_budget = (
                int(remaining_group_budget)
                if adaptive_budget or groups_per_step > 0
                else int(remaining_groups)
            )
            safe_group_budget = self._bridge_safe_deferred_producer_group_budget(
                tracking=tracking,
                job=job,
                requested_groups=int(requested_group_budget),
                used_bridge_tokens=(
                    bridge_token_count_by_request.get(str(rid))
                    if bridge_token_count_by_request is not None
                    else None
                ),
            )
            if safe_group_budget <= 0:
                continue
            launch_group_budget = (
                int(safe_group_budget)
                if (
                    adaptive_budget
                    or groups_per_step > 0
                    or int(safe_group_budget) < int(remaining_groups)
                )
                else 0
            )
            completed = self._run_and_record_deferred_bootstrap_producer_job(
                rid=str(rid),
                tracking=tracking,
                job=job,
                epoch=int(epoch),
                max_groups_per_call=int(launch_group_budget),
            )
            after_group_index = int(getattr(job, "next_payload_group_index", 0) or 0)
            if adaptive_budget or groups_per_step > 0:
                submitted_groups = max(0, int(after_group_index) - int(before_group_index))
                remaining_group_budget -= int(submitted_groups)
            if after_group_index > before_group_index or bool(
                getattr(job, "completed", False)
            ):
                launched += 1
        return launched

    def _bootstrap_pending_requires_global_wait(self) -> bool:
        pending_ids = tuple(
            str(rid)
            for rid in getattr(self, "_bootstrap_pending_request_ids", set())
        )
        if not pending_ids:
            return False
        for rid in pending_ids:
            if not self._request_can_bridge_bootstrap_decode(rid):
                return True
        return False

    def _step_has_decode_consumer(self) -> bool:
        """Whether the current step can consume compact data from decode rows."""
        authority = getattr(self, "step_authority", None)
        if authority is None:
            return True
        has_decode_row = getattr(authority, "has_decode_row", None)
        if has_decode_row is not None:
            return bool(has_decode_row)
        return bool(getattr(authority, "is_decode_only", False))

    def _step_has_compact_consumer(self) -> bool:
        """Whether the current step actually consumes compact rows."""
        authority = getattr(self, "step_authority", None)
        if authority is None:
            return True
        has_compact_row = getattr(authority, "has_compact_row", None)
        if has_compact_row is not None:
            return bool(has_compact_row)
        use_compact = getattr(authority, "use_compact_by_row", None)
        if use_compact is not None:
            cache_key: Tuple[int, int, int] | None = None
            try:
                cache_key = (id(authority), id(use_compact), len(use_compact))
            except TypeError:
                cache_key = None
            if cache_key is not None and getattr(
                self,
                "_step_has_compact_consumer_cache_key",
                None,
            ) == cache_key:
                return bool(
                    getattr(
                        self,
                        "_step_has_compact_consumer_cache_value",
                        False,
                    )
                )
            result = any(bool(value) for value in tuple(use_compact))
            if cache_key is not None:
                self._step_has_compact_consumer_cache_key = cache_key
                self._step_has_compact_consumer_cache_value = bool(result)
            return bool(result)
        return False

    def _pending_work_blockers(self, *, epoch: int) -> bool:
        """Return whether global blockers require wait even if flags are empty."""
        if epoch >= 0:
            release_epoch = self._prefill_release_pending_epoch
            if release_epoch >= 0 and epoch == release_epoch:
                return True
        if self._step_has_decode_consumer() and self._bootstrap_pending_requires_global_wait():
            return True
        return False

    def _bootstrap_required_mask(self) -> int:
        required_mask = 0
        num_layers = len(self.layer_cache_keys)
        if num_layers > 0:
            num_chunks = (num_layers + int(_CAPTURE_CHUNK) - 1) // int(_CAPTURE_CHUNK)
            for cid in range(int(num_chunks)):
                required_mask |= (1 << int(cid % int(_CAPTURE_IN_FLIGHT)))
        if required_mask == 0:
            required_mask = 1
        return int(required_mask)

    @staticmethod
    def _bootstrap_request_events_submitted(*, tracking: object) -> bool:
        """[BOOTSTRAP-PUBLISH-SUBMIT-FINAL] 事件已提交=可发布(提交即终局)。

        原 query() 形态(GPU 完成锚)是 TP-DET 同族发散源:每 rank 完成时刻
        不同 → publish 步分叉 → 批组成/collective 计划分叉 → NCCL 楔死
        (32k×TP2×双代 bootstrap 长世代窗实证)。决策只看提交态(rank 不变),
        内容正确性由发布点设备侧 wait_event 排序承担。
        """
        pending_events = getattr(tracking, "bootstrap_pending_events", None)
        if pending_events is None:
            return False
        events = tuple(pending_events)
        if not events:
            return False
        return all(evt is not None for evt in events)

    @staticmethod
    def _bootstrap_producer_publishable_for_bridge(*, tracking: object) -> bool:
        """结构判(无 GPU query):producer 链已提交且 final_event 已发布。

        [SUBMIT-FINAL] require_event_query=True 臂(final_event.query())全树
        零调用+GPU 完成锚发散源,随根修下线。
        """
        ready_state = getattr(tracking, "producer_ready_state", None)
        if ready_state is None:
            if getattr(tracking, "deferred_producer_job", None) is not None:
                return False
            return True
        return getattr(ready_state, "final_event", None) is not None

    def _mark_bootstrap_request_ready(
        self,
        *,
        rid: str,
        tracking: object,
        epoch: int = -1,
    ) -> None:
        # [BOOTSTRAP-MATERIALIZE-GATE 纵深断言 2026-07-10] ready 全层 commit
        # 前,该请求不得有在飞 refresh 世代(chunk 分轮 commit 与本 commit 交错
        # =层间 read_gen 错开=[DUAL-GEN-LAYER-PARITY] 崩)。planner 咽喉门保证
        # bridge 窗内票一律不 materialize ⇒ scheduled_* 恒 -1(scheduled 仅在
        # payload enqueue 成功的 commit 路径写入);此处 fail-fast 把门破/交错
        # 在源头暴露,替代 parity 守卫的下游兜捕。调用点全部经
        # bootstrap_pending=True 门(ready 后不重复 commit),无误炸面。
        _sched_ctrl = int(getattr(tracking, "scheduled_refresh_ctrl_step", -1))
        _sched_decode = int(getattr(tracking, "scheduled_decode_refresh_step", -1))
        if _sched_ctrl >= 0 or _sched_decode >= 0:
            raise RuntimeError(
                "[BOOTSTRAP-MATERIALIZE-GATE] bootstrap ready commit while a "
                f"refresh generation is in flight for req={rid!r} "
                f"(scheduled_refresh_ctrl_step={_sched_ctrl}, "
                f"scheduled_decode_refresh_step={_sched_decode}): chunk-round "
                "commits would interleave with the bootstrap full-layer commit "
                "(dual-gen layer parity risk); the planner materialize gate "
                "must hold tickets of bootstrap_pending requests"
            )
        ready_state = getattr(tracking, "producer_ready_state", None)
        commit_log = getattr(ready_state, "compact_meta_commit_log", None)
        if commit_log:
            commit_compact_meta = getattr(self, "_commit_compact_meta_log_entries", None)
            if callable(commit_compact_meta):
                commit_compact_meta(
                    tuple(commit_log), source=f"bootstrap_ready:{rid}"
                )
                commit_log.clear()
        tracking.bootstrap_done = True
        tracking.bootstrap_pending = False
        tracking.bootstrap_pending_epoch = -1
        if hasattr(tracking, "bootstrap_pending_events"):
            tracking.bootstrap_pending_events = []
        was_bridge_active = bool(getattr(tracking, "bootstrap_bridge_active", False))
        if was_bridge_active:
            ep = int(epoch)
            tracking.bootstrap_bridge_active = False
            if ep >= 0:
                tracking.ready_compact_epoch = ep
                tracking.active_compact_epoch = ep
            bridge_token_count = int(getattr(tracking, "bridge_token_count", 0) or 0)
            try:
                refresh_interval = int(
                    getattr(
                        getattr(self, "config", None),
                        "refresh_interval",
                        0,
                    )
                    or 0
                )
            except Exception:
                refresh_interval = 0
            if refresh_interval > 0:
                try:
                    bridge_decode_step = int(getattr(tracking, "decode_step", -1))
                except Exception:
                    bridge_decode_step = -1
                if bridge_decode_step < 0:
                    bridge_decode_step = int(bridge_token_count)
                else:
                    bridge_decode_step = max(
                        int(bridge_decode_step),
                        int(bridge_token_count),
                    )
                catchup_delay = 1
                tracking.post_bridge_refresh_due_decode_step = int(
                    bridge_decode_step + catchup_delay
                )
                # [CREDIT-RETIRE 2026-07-07] post_bridge_refresh_done credit
                # 状态机已退休;追赶票(due)机制保留。
            bridge_token_positions = [
                int(pos)
                for pos in list(getattr(tracking, "bridge_token_positions", []) or [])
            ]
            if os.environ.get("VLLM_SPARSE_ONE_SHOT_TIMELINE_LOG", ""):
                try:
                    from patches.refresh_runtime.one_shot_timeline import (
                        append_one_shot_timeline,
                        timeline_log_path,
                    )

                    append_one_shot_timeline(
                        path=timeline_log_path(),
                        request_id=str(rid),
                        epoch=ep,
                        chunk_id=-1,
                        buffer_id=-1,
                        phase="deferred_producer_ready_publish",
                        timestamp_ns=time.perf_counter_ns(),
                        duration_us=0.0,
                        aux_stream_enabled=True,
                        extra_fields={
                            "producer_ready_step": int(bridge_token_count),
                            "bridge_token_count": int(bridge_token_count),
                            "bridge_token_positions": list(bridge_token_positions),
                            "producer_launch_step": int(
                                getattr(tracking, "producer_launch_step", -1)
                            ),
                            "ready_compact_epoch": int(
                                getattr(tracking, "ready_compact_epoch", -1)
                            ),
                            "active_compact_epoch": int(
                                getattr(tracking, "active_compact_epoch", -1)
                            ),
                            **_bootstrap_full_kv_handoff_trace_fields(),
                        },
                    )
                except Exception:
                    _log.debug("failed to append deferred ready timeline", exc_info=True)
            if os.environ.get("VLLM_SPARSE_FA3_ROUTE_TRACE_LOG", ""):
                try:
                    from patches.fa3_native.install import append_fa3_route_trace

                    append_fa3_route_trace(
                        {
                            "event": "deferred_bridge_producer_ready",
                            "phase": "post_switch_phase",
                            "request_id": str(rid),
                            "epoch": ep,
                            "producer_ready_step": int(bridge_token_count),
                            "bridge_token_count": int(bridge_token_count),
                            "bridge_token_positions": list(bridge_token_positions),
                            "producer_launch_step": int(
                                getattr(tracking, "producer_launch_step", -1)
                            ),
                            **_bootstrap_full_kv_handoff_trace_fields(),
                        }
                    )
                except Exception:
                    _log.debug("failed to append deferred ready route trace", exc_info=True)
            tracking.deferred_producer_job = None
            tracking.producer_ready_state = None
        self._bootstrap_pending_request_ids.discard(rid)

    def _publish_ready_bootstrap_requests_at_step_boundary(self, *, epoch: int) -> None:
        """在 step 边界按"提交即终局"提升 bootstrap 请求(request-wise)。

        设计约束：
        - prefill 完成后的 selected-ready 是 request-wise，而不是全局 buf-wise；
        - [BOOTSTRAP-PUBLISH-SUBMIT-FINAL 2026-07-08] 发布决策只看事件已提交
          (rank 不变,TP-DET 同族合同);内容正确性=发布点主流 wait_event 对
          producer 事件排序(one-shot graph 路径同语义先例),零 host 同步;
        - 不依赖 selected materialization probe,不做 host-side rebuild。
        """
        ep = int(epoch)
        if ep < 0 or not self._bootstrap_pending_request_ids:
            return

        pending_ids = list(self._bootstrap_pending_request_ids)
        for rid in pending_ids:
            tracking = self.request_states.get(rid)
            if tracking is None:
                self._bootstrap_pending_request_ids.discard(rid)
                continue
            if not bool(tracking.bootstrap_pending):
                self._bootstrap_pending_request_ids.discard(rid)
                continue
            pending_ep = int(tracking.bootstrap_pending_epoch)
            if pending_ep < 0 or pending_ep >= ep:
                continue
            if self._bootstrap_submission_boundary_blocks_publish(rid=rid):
                continue
            if self._request_bridge_token_budget_remaining(str(rid)):
                setattr(
                    tracking,
                    "bootstrap_publish_skipped_reason",
                    "bridge_budget_not_exhausted",
                )
                continue
            if not WaitDeciderMixin._bootstrap_request_events_submitted(
                tracking=tracking
            ):
                continue
            ready_state = getattr(tracking, "producer_ready_state", None)
            if ready_state is not None:
                from patches.refresh_runtime.producer_ready import (
                    validate_producer_ready_for_publish,
                )

                try:
                    validate_producer_ready_for_publish(ready_state)
                except RuntimeError:
                    raise
                except Exception as exc:
                    raise RuntimeError(
                        "producer ready state is invalid for bootstrap publish"
                    ) from exc
            # [BOOTSTRAP-PUBLISH-SUBMIT-FINAL] 设备侧排序:主流 wait_event 对
            # 全部 producer 事件排序,后续消费 kernel(含 graph replay)天然
            # 后序——publish 不再等 GPU 完成,内容也不可能被读到半成品。
            if torch.cuda.is_available():
                if _is_stream_capturing_or_raise(
                    stage="bootstrap_publish_submit_final"
                ):
                    raise RuntimeError(
                        "bootstrap step-boundary publish entered during CUDA "
                        "graph capture; device-side ordering cannot be "
                        "recorded here"
                    )
                _publish_stream = torch.cuda.current_stream()
                for _evt in tuple(tracking.bootstrap_pending_events):
                    _publish_stream.wait_event(_evt)
            self._mark_bootstrap_request_ready(
                rid=rid,
                tracking=tracking,
                epoch=ep,
            )

    def _one_shot_graph_async_bootstrap_enabled(self) -> bool:
        if not self._async_refresh_enabled():
            return False
        _one_shot_async_on = (
            os.environ.get("VLLM_SPARSE_ONE_SHOT_ASYNC_BOOTSTRAP", "0") == "1"
            if _DYNAMIC_ENV
            else _ONE_SHOT_ASYNC_BOOTSTRAP_CACHED
        )
        if not _one_shot_async_on:
            return False
        _attn_in_cudagraph_on = bool(
            getattr(self, "_sparse_attention_in_cudagraph", False)
        )
        if not _attn_in_cudagraph_on:
            return False
        return bool(getattr(getattr(self, "config", None), "one_shot_bootstrap_only", False))

    def _deferred_bootstrap_producer_enabled(self) -> bool:
        if _DYNAMIC_ENV:
            return (
                os.environ.get("VLLM_SPARSE_DEFER_BOOTSTRAP_PRODUCER", "0")
                == "1"
            )
        return bool(_DEFER_BOOTSTRAP_PRODUCER_CACHED)

    def _bootstrap_submission_boundary_enabled(self) -> bool:
        """Return whether request-local async prefill submission is required."""

        return bool(
            self._async_refresh_enabled()
            and bool(
                getattr(
                    getattr(self, "config", None),
                    "one_shot_bootstrap_only",
                    False,
                )
            )
            and not self._deferred_bootstrap_producer_enabled()
        )

    def _bootstrap_submission_boundary_blocks_publish(self, *, rid: str) -> bool:
        """Keep publication behind the host submission transaction."""

        pending_epoch_by_id = getattr(
            self,
            "_bootstrap_submission_boundary_pending_epoch_by_id",
            None,
        )
        if not isinstance(pending_epoch_by_id, dict):
            raise RuntimeError("E_TP_BOOTSTRAP_SUBMISSION_LEDGER_CORRUPT")
        return str(rid) in pending_epoch_by_id

    def _arm_prefill_submission_boundary(
        self,
        *,
        request_ids: Sequence[str],
        epoch: int,
    ) -> int:
        """Arm exact final-chunk producers for next-step publication.

        The caller supplies ``finalize_req_ids`` from the immutable CPU capture
        plan.  This avoids inferring producer existence from scheduler prompt
        counters and leaves capture-disabled/continuous/deferred modes outside
        the ledger entirely.
        """

        if not request_ids or not self._bootstrap_submission_boundary_enabled():
            return 0
        pending_epoch_by_id = getattr(
            self,
            "_bootstrap_submission_boundary_pending_epoch_by_id",
            None,
        )
        if not isinstance(pending_epoch_by_id, dict):
            raise RuntimeError("E_TP_BOOTSTRAP_SUBMISSION_LEDGER_CORRUPT")
        arm_epoch = int(epoch)
        if arm_epoch < 0:
            raise RuntimeError("E_TP_BOOTSTRAP_SUBMISSION_LEDGER_EPOCH")
        armed_ids = tuple(dict.fromkeys(str(rid) for rid in request_ids))
        missing_ids = [rid for rid in armed_ids if rid not in self.request_states]
        if missing_ids:
            raise RuntimeError(
                "E_TP_BOOTSTRAP_SUBMISSION_LEDGER_UNKNOWN_REQUEST: "
                f"requests={missing_ids[:4]} count={len(missing_ids)}"
            )
        stale_ids = [
            (rid, int(pending_epoch_by_id[rid]))
            for rid in armed_ids
            if rid in pending_epoch_by_id
            and int(pending_epoch_by_id[rid]) != arm_epoch
        ]
        if stale_ids:
            raise RuntimeError(
                "E_TP_BOOTSTRAP_SUBMISSION_LEDGER_UNCONSUMED: "
                f"armed_epoch={arm_epoch} prior={stale_ids[:4]}"
            )
        pending_epoch_by_id.update(
            (rid, arm_epoch) for rid in armed_ids
        )
        return len(armed_ids)

    def _one_shot_group_ready_graph_wait_enabled(self) -> bool:
        if not self._one_shot_graph_async_bootstrap_enabled():
            return False
        # [GROUP-READY-ENV-RETIRED 2026-07-03] The VLLM_SPARSE_ONE_SHOT_GROUP_READY
        # force-on env knob is retired: it was a 2026-05 optional guard for the seq0
        # drift, which has since been root-fixed by STAGE-RING-FIX. A/B evidence:
        # 8/8 golden outputs identical, OFF (319.2 tps) >= ON (317.7 tps), i.e. the
        # guard contributed nothing to correctness. The READY_CHUNK < CAPTURE_CHUNK
        # progressive auto-enable path below is intentionally KEPT — it shares all
        # of the group event/wait machinery.
        if not os.environ.get("VLLM_SPARSE_ONE_SHOT_READY_CHUNK", "").strip():
            return False

        from patches.refresh_runtime.producer_ready import (
            resolve_one_shot_ready_chunk,
            validate_one_shot_ready_chunk_alignment,
        )

        capture_chunk = int(_CAPTURE_CHUNK)
        ready_chunk = int(resolve_one_shot_ready_chunk(capture_chunk=capture_chunk))
        validate_one_shot_ready_chunk_alignment(
            capture_chunk=capture_chunk,
            ready_chunk=ready_chunk,
        )
        return ready_chunk < capture_chunk

    def _maybe_wait_one_shot_group_ready_for_layer(
        self,
        *,
        layer_index: int,
        device: torch.device,
    ) -> bool:
        """Capture/replay a strict per-group producer wait for one-shot RRP."""
        if not self._one_shot_group_ready_graph_wait_enabled():
            return False
        if device.type != "cuda":
            return False
        layer_index_i = int(layer_index)
        if layer_index_i < 0:
            return False
        if not self._step_has_decode_consumer():
            return False
        if not self._step_has_compact_consumer():
            return False
        from patches.refresh_runtime.producer_ready import resolve_one_shot_ready_chunk

        ready_chunk = resolve_one_shot_ready_chunk(capture_chunk=int(_CAPTURE_CHUNK))
        if ready_chunk <= 0:
            raise RuntimeError("one-shot group-ready requires a positive ready chunk")
        group_id = layer_index_i // ready_chunk
        self._wait_one_shot_group_ready_for_group(
            group_id=int(group_id),
            device=device,
        )
        return True

    def _validate_prefill_submission_before_bootstrap_publish(
        self,
        *,
        req_ids: Sequence[str],
        epoch: int,
    ) -> int:
        """Validate the async-prefill submission boundary before publish.

        ``bootstrap_done`` is a TP-visible routing decision.  The producer's
        CUDA completion may differ across ranks, but the request must not expose
        compact state until every rank has submitted its request-local final
        event.  Submission is host-owned state: a device wait cannot create a
        missing finalize record.  The transition ledger keeps steady decode
        outside both the request scan and all CUDA work.
        """
        pending_epoch_by_id = getattr(
            self,
            "_bootstrap_submission_boundary_pending_epoch_by_id",
            None,
        )
        if not isinstance(pending_epoch_by_id, dict):
            raise RuntimeError("E_TP_BOOTSTRAP_SUBMISSION_LEDGER_CORRUPT")
        if not pending_epoch_by_id:
            return 0
        if not self._bootstrap_submission_boundary_enabled():
            raise RuntimeError(
                "E_TP_BOOTSTRAP_SUBMISSION_LEDGER_SCOPE_DRIFT"
            )

        active_ids = {str(rid_raw) for rid_raw in req_ids}
        candidates: List[Tuple[str, object, int]] = []
        current_epoch = int(epoch)
        for rid, arm_epoch_raw in tuple(pending_epoch_by_id.items()):
            arm_epoch = int(arm_epoch_raw)
            if arm_epoch > current_epoch:
                raise RuntimeError(
                    "E_TP_BOOTSTRAP_SUBMISSION_LEDGER_FUTURE_EPOCH: "
                    f"request={rid!r} armed={arm_epoch} current={current_epoch}"
                )
            if arm_epoch == current_epoch:
                # _prepare_inputs may re-enter in the producer's own dispatch.
                # Validation belongs to the first later step, after attention
                # had a chance to submit the request-local final event.
                continue
            if rid not in active_ids:
                continue
            tracking = self.request_states.get(rid)
            if tracking is None:
                raise RuntimeError(
                    "E_TP_BOOTSTRAP_SUBMISSION_LEDGER_UNKNOWN_REQUEST: "
                    f"request={rid!r}"
                )
            if bool(getattr(tracking, "bootstrap_done", False)):
                raise RuntimeError(
                    "E_TP_BOOTSTRAP_SUBMISSION_LEDGER_BYPASSED: "
                    f"request={rid!r} bootstrap_done before validation"
                )
            if bool(getattr(tracking, "_was_short_dense", False)):
                raise RuntimeError(
                    "E_TP_BOOTSTRAP_SUBMISSION_LEDGER_SCOPE_DRIFT: "
                    f"request={rid!r} is short-dense"
                )
            if bool(getattr(tracking, "bootstrap_bridge_active", False)):
                raise RuntimeError(
                    "E_TP_BOOTSTRAP_SUBMISSION_LEDGER_SCOPE_DRIFT: "
                    f"request={rid!r} entered deferred bridge"
                )
            candidates.append((rid, tracking, arm_epoch))
        if not candidates:
            return 0

        missing = []
        for rid, tracking, arm_epoch in candidates:
            if not bool(getattr(tracking, "bootstrap_pending", False)):
                missing.append(rid)
                continue
            if int(getattr(tracking, "bootstrap_pending_epoch", -1)) != arm_epoch:
                raise RuntimeError(
                    "E_TP_BOOTSTRAP_SUBMISSION_LEDGER_EPOCH_DRIFT: "
                    f"request={rid!r} armed={arm_epoch} pending="
                    f"{int(getattr(tracking, 'bootstrap_pending_epoch', -1))}"
                )
            if not WaitDeciderMixin._bootstrap_request_events_submitted(
                tracking=tracking
            ):
                missing.append(rid)
                continue
            ready_state = getattr(tracking, "producer_ready_state", None)
            if ready_state is None:
                raise RuntimeError(
                    "E_TP_BOOTSTRAP_SUBMISSION_BOUNDARY: producer ready "
                    f"state is missing for request={rid!r}"
                )
            from patches.refresh_runtime.producer_ready import (
                validate_producer_ready_for_publish,
            )

            try:
                validate_producer_ready_for_publish(ready_state)
            except Exception as exc:
                raise RuntimeError(
                    "E_TP_BOOTSTRAP_SUBMISSION_BOUNDARY: producer ready "
                    f"state is incomplete for request={rid!r}"
                ) from exc
        if missing:
            raise RuntimeError(
                "E_TP_BOOTSTRAP_SUBMISSION_BOUNDARY: long-request prefill "
                "completed without a submitted request-local bootstrap event; "
                f"epoch={int(epoch)} requests={missing[:4]} count={len(missing)}"
            )
        for rid, _tracking, _arm_epoch in candidates:
            pending_epoch_by_id.pop(rid, None)
        return len(candidates)

    def _wait_one_shot_group_ready_for_full_graph_replay(
        self,
        *,
        device: torch.device,
    ) -> int:
        """Order all ready groups before a full CUDA graph replay consumes them."""
        if not self._one_shot_group_ready_graph_wait_enabled():
            self._one_shot_group_ready_full_graph_wait_required = False
            return 0
        if device.type != "cuda":
            self._one_shot_group_ready_full_graph_wait_required = False
            return 0
        if not self._step_has_decode_consumer():
            self._one_shot_group_ready_full_graph_wait_required = False
            return 0
        if not self._step_has_compact_consumer():
            self._one_shot_group_ready_full_graph_wait_required = False
            return 0

        from patches.refresh_runtime.producer_ready import resolve_one_shot_ready_chunk

        ready_chunk = int(resolve_one_shot_ready_chunk(capture_chunk=int(_CAPTURE_CHUNK)))
        if ready_chunk <= 0:
            raise RuntimeError("one-shot group-ready requires a positive ready chunk")
        layer_cache_keys = tuple(getattr(self, "layer_cache_keys", ()) or ())
        if layer_cache_keys:
            group_ids = tuple(
                range((len(layer_cache_keys) + ready_chunk - 1) // ready_chunk)
            )
        else:
            group_ids_set: set[int] = set()
            events = getattr(self, "_one_shot_group_done_evt_by_group", None)
            if isinstance(events, dict):
                group_ids_set.update(int(key) for key in events.keys())
            slot_events = getattr(self, "_one_shot_group_done_evt_by_group_slot", None)
            if isinstance(slot_events, dict):
                group_ids_set.update(
                    int(key[0])
                    for key in slot_events.keys()
                    if isinstance(key, tuple) and len(key) == 2
                )
            group_ids = tuple(sorted(group_ids_set))
        waited = 0
        for group_id in group_ids:
            waited += int(
                self._wait_one_shot_group_ready_for_group(
                    group_id=int(group_id),
                    device=device,
                )
            )
        self._one_shot_group_ready_full_graph_wait_required = False
        return int(waited)

    def _wait_one_shot_group_ready_for_group(
        self,
        *,
        group_id: int,
        device: torch.device,
    ) -> int:
        epoch = int(getattr(self, "step_context_epoch", -1))
        waited_epoch_by_slot = getattr(
            self,
            "_one_shot_group_ready_wait_epoch_by_group_slot",
            None,
        )
        if not isinstance(waited_epoch_by_slot, dict):
            waited_epoch_by_slot = {}
            self._one_shot_group_ready_wait_epoch_by_group_slot = waited_epoch_by_slot
        waited_serial_by_slot = getattr(
            self,
            "_one_shot_group_ready_wait_serial_by_group_slot",
            None,
        )
        if not isinstance(waited_serial_by_slot, dict):
            waited_serial_by_slot = {}
            self._one_shot_group_ready_wait_serial_by_group_slot = waited_serial_by_slot
        serial_by_group = getattr(
            self,
            "_one_shot_group_done_evt_serial_by_group",
            None,
        )
        serial_by_slot = getattr(
            self,
            "_one_shot_group_done_evt_serial_by_group_slot",
            None,
        )
        slot_events = getattr(self, "_one_shot_group_done_evt_by_group_slot", None)
        wait_items: list[tuple[int, object, int]] = []
        slots = self._one_shot_group_ready_compact_slots()
        if slots and isinstance(slot_events, dict):
            missing_slots: list[int] = []
            for slot in slots:
                event = slot_events.get((int(group_id), int(slot)))
                if event is None:
                    missing_slots.append(int(slot))
                    continue
                event_serial = -1
                if isinstance(serial_by_slot, dict):
                    event_serial = int(
                        serial_by_slot.get((int(group_id), int(slot)), -1)
                    )
                wait_items.append((int(slot), event, int(event_serial)))
            if wait_items and missing_slots:
                raise RuntimeError(
                    "one-shot group-ready slot event missing for "
                    f"group {int(group_id)} slot {int(missing_slots[0])}"
                )
        events = getattr(self, "_one_shot_group_done_evt_by_group", None)
        if not wait_items and not isinstance(events, dict):
            raise RuntimeError("one-shot group-ready event registry is missing")
        if not wait_items:
            event = events.get(int(group_id))
            if event is not None:
                event_serial = -1
                if isinstance(serial_by_group, dict):
                    event_serial = int(serial_by_group.get(int(group_id), -1))
                wait_items.append((-1, event, int(event_serial)))
        if not wait_items:
            raise RuntimeError(
                f"one-shot group-ready event missing for group {int(group_id)}"
            )
        current_stream = torch.cuda.current_stream(device=device)
        stream_key = int(getattr(current_stream, "cuda_stream", id(current_stream)))
        waited = 0
        for slot, event, event_serial in wait_items:
            wait_key = (int(group_id), int(slot))
            waited_key = (int(group_id), int(slot), int(stream_key))
            if event_serial >= 0:
                if int(waited_serial_by_slot.get(waited_key, -1)) == event_serial:
                    continue
            else:
                if (
                    epoch >= 0
                    and int(waited_epoch_by_slot.get(waited_key, -1)) == epoch
                ):
                    continue
            current_stream.wait_event(event)
            waited += 1
            if event_serial >= 0:
                waited_serial_by_slot[waited_key] = event_serial
            elif epoch >= 0:
                waited_epoch_by_slot[waited_key] = epoch
        return int(waited)

    def _one_shot_group_ready_compact_slots(self) -> Tuple[int, ...]:
        authority = getattr(self, "step_authority", None)
        if authority is None:
            return tuple()
        use_compact = getattr(authority, "use_compact_by_row", None)
        slot_by_row = getattr(authority, "slot_by_row", None)
        if use_compact is None or slot_by_row is None:
            return tuple()
        cache_key: Tuple[int, int, int, int, int] | None = None
        try:
            cache_key = (
                id(authority),
                id(use_compact),
                id(slot_by_row),
                len(use_compact),
                len(slot_by_row),
            )
        except TypeError:
            cache_key = None
        if cache_key is not None and getattr(
            self,
            "_one_shot_group_ready_compact_slots_cache_key",
            None,
        ) == cache_key:
            return tuple(
                getattr(
                    self,
                    "_one_shot_group_ready_compact_slots_cache_value",
                    tuple(),
                )
            )
        try:
            use_values = tuple(use_compact)
            slot_values = tuple(slot_by_row)
        except TypeError:
            return tuple()
        slots: list[int] = []
        seen: set[int] = set()
        for row, use_value in enumerate(use_values):
            if not bool(use_value) or int(row) >= len(slot_values):
                continue
            slot = int(slot_values[int(row)])
            if slot < 0 or slot in seen:
                continue
            seen.add(slot)
            slots.append(slot)
        compact_slots = tuple(slots)
        if cache_key is not None:
            self._one_shot_group_ready_compact_slots_cache_key = cache_key
            self._one_shot_group_ready_compact_slots_cache_value = compact_slots
        return compact_slots

    def _wait_and_publish_bootstrap_requests_for_graph_decode(
        self,
        *,
        epoch: int,
        device: Optional[torch.device] = None,
    ) -> int:
        """Graph-safe one-shot bootstrap publish.

        Full CUDA graph replay cannot execute the normal Python wait path. For
        one-shot bootstrap, the only async producer is prefill selector+rebuild;
        waiting on its request-local finalize events before StepAuthority is
        built orders the subsequent graph launch without a host sync.
        """
        ep = int(epoch)
        if ep < 0 or not self._bootstrap_pending_request_ids:
            return 0
        if not self._one_shot_graph_async_bootstrap_enabled():
            return 0
        if _is_stream_capturing_or_raise(stage="one_shot_graph_async_bootstrap_wait"):
            raise RuntimeError(
                "one-shot async bootstrap wait must run before CUDA graph capture/replay"
            )
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        current_stream = torch.cuda.current_stream(device=device)

        published = 0
        pending_ids = list(self._bootstrap_pending_request_ids)
        for rid in pending_ids:
            tracking = self.request_states.get(rid)
            if tracking is None:
                self._bootstrap_pending_request_ids.discard(rid)
                continue
            if not bool(tracking.bootstrap_pending):
                self._bootstrap_pending_request_ids.discard(rid)
                continue
            pending_ep = int(tracking.bootstrap_pending_epoch)
            if pending_ep < 0 or pending_ep >= ep:
                continue
            if self._bootstrap_submission_boundary_blocks_publish(rid=rid):
                continue
            if self._request_bridge_token_budget_remaining(str(rid)):
                setattr(
                    tracking,
                    "bootstrap_publish_skipped_reason",
                    "bridge_active_without_producer_wait",
                )
                continue
            ready_state = getattr(tracking, "producer_ready_state", None)
            if ready_state is not None:
                if not self._wait_and_publish_one_shot_producer_for_graph_decode(
                    rid=str(rid),
                    tracking=tracking,
                    epoch=ep,
                    current_stream=current_stream,
                ):
                    continue
                published += 1
                continue
            if bool(
                getattr(
                    getattr(self, "config", None),
                    "one_shot_bootstrap_only",
                    False,
                )
            ):
                setattr(
                    tracking,
                    "bootstrap_publish_skipped_reason",
                    "one_shot_producer_ready_state_missing",
                )
                continue
            events = tuple(getattr(tracking, "bootstrap_pending_events", tuple()) or tuple())
            if not events or any(evt is None for evt in events):
                continue
            if os.environ.get("VLLM_SPARSE_ONE_SHOT_TIMELINE_LOG", ""):
                try:
                    from patches.refresh_runtime.one_shot_timeline import (
                        append_one_shot_timeline,
                        timeline_log_path,
                    )

                    timeline_path = timeline_log_path()
                except Exception:
                    append_one_shot_timeline = None
                    timeline_path = None
            else:
                append_one_shot_timeline = None
                timeline_path = None
            for evt in events:
                wait_start_ns = time.perf_counter_ns()
                if append_one_shot_timeline is not None:
                    append_one_shot_timeline(
                        path=timeline_path,
                        request_id=str(rid),
                        epoch=ep,
                        chunk_id=-1,
                        buffer_id=-1,
                        phase="bootstrap_wait_start",
                        timestamp_ns=wait_start_ns,
                        duration_us=0.0,
                        aux_stream_enabled=True,
                    )
                current_stream.wait_event(evt)
                wait_end_ns = time.perf_counter_ns()
                if append_one_shot_timeline is not None:
                    append_one_shot_timeline(
                        path=timeline_path,
                        request_id=str(rid),
                        epoch=ep,
                        chunk_id=-1,
                        buffer_id=-1,
                        phase="bootstrap_wait_end",
                        timestamp_ns=wait_end_ns,
                        duration_us=float(wait_end_ns - wait_start_ns) / 1000.0,
                        aux_stream_enabled=True,
                    )
            self._mark_bootstrap_request_ready(
                rid=rid,
                tracking=tracking,
                epoch=ep,
            )
            published += 1

        if published > 0 and not self._bootstrap_pending_request_ids:
            for buf in range(len(self._buf_pending_work_flags)):
                flags = int(self._buf_pending_work_flags[buf])
                if (flags & 1) != 0:
                    next_flags = flags & ~1
                    self._buf_pending_work_flags[buf] = next_flags
                    if next_flags == 0:
                        self._buf_pending_work_epoch[buf] = -1
                flush_profile = getattr(self, "_refresh_profile_try_flush_pending", None)
                if callable(flush_profile):
                    flush_profile(buf=buf, wait_epoch=ep)
        return published

    def _wait_and_publish_one_shot_producer_for_graph_decode(
        self,
        *,
        rid: str,
        tracking: object,
        epoch: int,
        current_stream: torch.cuda.Stream,
    ) -> bool:
        """Wait the one-shot RRP producer final event and publish readiness."""
        from patches.refresh_runtime.producer_ready import (
            producer_ready_summary,
            validate_producer_groups_ready_for_graph_wait,
            validate_producer_ready_for_publish,
        )

        ready_state = getattr(tracking, "producer_ready_state", None)
        if ready_state is None:
            return False
        if self._one_shot_group_ready_graph_wait_enabled():
            try:
                validate_producer_groups_ready_for_graph_wait(ready_state)
            except Exception as exc:
                raise RuntimeError(
                    "one-shot group-ready producer state is invalid"
                ) from exc
            ready_state.graph_wait_event_used = True
            if os.environ.get("VLLM_SPARSE_ONE_SHOT_TIMELINE_LOG", ""):
                try:
                    from patches.refresh_runtime.one_shot_timeline import (
                        append_one_shot_timeline,
                        timeline_log_path,
                    )

                    append_one_shot_timeline(
                        path=timeline_log_path(),
                        request_id=str(rid),
                        epoch=int(epoch),
                        chunk_id=-1,
                        buffer_id=-1,
                        phase="producer_group_ready_publish",
                        timestamp_ns=time.perf_counter_ns(),
                        duration_us=0.0,
                        aux_stream_enabled=True,
                        extra_fields=producer_ready_summary(ready_state),
                    )
                except Exception:
                    _log.debug(
                        "failed to append group-ready publish timeline",
                        exc_info=True,
                    )
            self._one_shot_group_ready_full_graph_wait_required = True
            self._mark_bootstrap_request_ready(
                rid=str(rid),
                tracking=tracking,
                epoch=int(epoch),
            )
            return True
        try:
            validate_producer_ready_for_publish(ready_state)
        except RuntimeError:
            return False
        final_event = ready_state.final_event
        if final_event is None:
            return False

        if os.environ.get("VLLM_SPARSE_ONE_SHOT_TIMELINE_LOG", ""):
            try:
                from patches.refresh_runtime.one_shot_timeline import (
                    append_one_shot_timeline,
                    timeline_log_path,
                )

                timeline_path = timeline_log_path()
            except Exception:
                append_one_shot_timeline = None
                timeline_path = None
        else:
            append_one_shot_timeline = None
            timeline_path = None

        cuda_timing_sync = os.environ.get(
            "VLLM_SPARSE_ONE_SHOT_CUDA_TIMING_SYNC", ""
        ) == "1"
        wait_evt0: Optional[torch.cuda.Event] = None
        wait_evt1: Optional[torch.cuda.Event] = None
        if cuda_timing_sync:
            wait_evt0 = torch.cuda.Event(enable_timing=True)
            wait_evt1 = torch.cuda.Event(enable_timing=True)
            wait_evt0.record(current_stream)

        wait_start_ns = time.perf_counter_ns()
        if append_one_shot_timeline is not None:
            append_one_shot_timeline(
                path=timeline_path,
                request_id=str(rid),
                epoch=int(epoch),
                chunk_id=-1,
                buffer_id=-1,
                phase="producer_final_event_wait_start",
                timestamp_ns=wait_start_ns,
                duration_us=0.0,
                aux_stream_enabled=True,
                extra_fields=producer_ready_summary(ready_state),
            )
        current_stream.wait_event(final_event)
        producer_final_event_gpu_wait_us: Optional[float] = None
        if wait_evt0 is not None and wait_evt1 is not None:
            wait_evt1.record(current_stream)
            try:
                wait_evt1.synchronize()
                producer_final_event_gpu_wait_us = float(
                    wait_evt0.elapsed_time(wait_evt1)
                ) * 1000.0
            except Exception:
                _log.warning(
                    "one-shot CUDA timing sync failed for producer final event wait",
                    exc_info=True,
                )
        ready_state.graph_wait_event_used = True
        wait_end_ns = time.perf_counter_ns()
        if append_one_shot_timeline is not None:
            wait_extra_fields = producer_ready_summary(ready_state)
            if producer_final_event_gpu_wait_us is not None:
                wait_extra_fields["producer_final_event_gpu_wait_us"] = float(
                    producer_final_event_gpu_wait_us
                )
            append_one_shot_timeline(
                path=timeline_path,
                request_id=str(rid),
                epoch=int(epoch),
                chunk_id=-1,
                buffer_id=-1,
                phase="producer_final_event_wait_end",
                timestamp_ns=wait_end_ns,
                duration_us=float(wait_end_ns - wait_start_ns) / 1000.0,
                aux_stream_enabled=True,
                extra_fields=wait_extra_fields,
            )
        self._mark_bootstrap_request_ready(
            rid=str(rid),
            tracking=tracking,
            epoch=int(epoch),
        )
        return True

    # ------------------------------------------------------------------
    # Env-cache helpers
    # ------------------------------------------------------------------



    # ------------------------------------------------------------------
    # Trace logging
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # Unified wait decision
    # ------------------------------------------------------------------

    def _compute_wait_decision(
        self,
        *,
        buf_id: int,
        epoch: int,
        path_tag: str,
        consume_step_token: bool,
    ) -> Tuple[bool, str]:
        """统一 wait 判定入口，保证 dispatcher/runtime plan 条件一致。"""
        if not self._async_refresh_enabled():
            need_wait, reason = False, "async_disabled"
            return need_wait, reason

        ep = int(epoch)
        buf = int(buf_id) % int(_CAPTURE_IN_FLIGHT)
        has_pending = self._pending_work_for_buf(buf_id=buf)
        has_blockers = self._pending_work_blockers(epoch=ep)
        # FIND #4: only .need_wait is consumed here, which run_refresh_step
        # derives as bool(has_pending or has_blockers) (should_wait_when_pending);
        # read it directly to skip the per-step RefreshDecision/[]+normalize alloc.
        if not (has_pending or has_blockers):
            need_wait, reason = False, "no_pending_no_blockers"
            return need_wait, reason
        # [E9] capturing 探测（CUDA driver API ~1-2µs）后置到零工作早退之后：
        # 稳态无 pending 的每层调用不再付探测税。capture 中恰逢无 pending 时
        # 仅 reason 字符串不同（no_pending_no_blockers），need_wait 全路径不变。
        stage = f"{str(path_tag)}_wait_probe" if path_tag else "wait_probe"
        if _is_stream_capturing_or_raise(stage=stage):
            need_wait, reason = False, "stream_capturing"
            return need_wait, reason

        flags = 0
        pending_epoch = -1
        if 0 <= buf < len(self._buf_pending_work_flags):
            flags = self._buf_pending_work_flags[buf]
        if 0 <= buf < len(self._buf_pending_work_epoch):
            pending_epoch = self._buf_pending_work_epoch[buf]
        if flags == 0:
            same_epoch_work = ep >= 0 and pending_epoch == ep
            release_epoch = int(getattr(self, "_prefill_release_pending_epoch", -1))
            need_release = ep >= 0 and release_epoch >= 0 and ep == release_epoch
            need_bootstrap = (
                self._step_has_decode_consumer()
                and bool(getattr(self, "_bootstrap_pending_request_ids", set()))
            )
            if (not same_epoch_work) and (not need_release) and (not need_bootstrap):
                need_wait, reason = False, "empty_flags_safe_skip"
                return need_wait, reason

        if not bool(consume_step_token):
            need_wait, reason = True, "pending_or_blockers"
            return need_wait, reason

        hints = self.step_exec_hints
        if hints is None or int(hints.epoch) != ep:
            need_wait, reason = True, "pending_or_blockers_no_hints"
            return need_wait, reason
        if not (0 <= buf < len(hints.wait_token_by_buf)):
            need_wait, reason = False, "hints_buf_out_of_range"
            return need_wait, reason
        token = int(hints.wait_token_by_buf[buf])
        if token <= 0:
            need_wait, reason = False, "hints_token_zero"
            return need_wait, reason
        if 0 <= buf < len(self._step_wait_consumed_token_by_buf):
            if int(self._step_wait_consumed_token_by_buf[buf]) == token:
                need_wait, reason = False, "hints_token_consumed"
                return need_wait, reason
            self._step_wait_consumed_token_by_buf[buf] = token
        need_wait, reason = True, "hints_token_wait"
        return need_wait, reason

    # ------------------------------------------------------------------
    # Main-stream wait execution
    # ------------------------------------------------------------------

    def _main_stream_wait_for_chunk_done(
        self,
        *,
        buf_id: int,
        device: torch.device,
        chunk_id: Optional[int] = None,
        epoch: Optional[int] = None,
    ) -> None:
        """Before reusing capture ring buf, wait for async work completion via event.

        设计目标：避免每层都调用 wait_event；同时兼容 layer 调用顺序不严格单调的情况。
        - 若提供 (epoch, chunk_id)：同一 step 内同一 buf_id 对同一 chunk 只 wait 一次；
        - 若未提供：退化为"每次调用都 wait_event"（仍无 CPU 同步）。
        """
        if not self._async_refresh_enabled():
            return
        if _is_stream_capturing_or_raise(stage="_main_stream_wait_for_chunk_done"):
            return
        self._reclaim_retired_buffers()
        buf = int(buf_id) % int(_CAPTURE_IN_FLIGHT)
        if epoch is not None and int(epoch) != int(self._main_wait_epoch):
            self._main_wait_epoch = int(epoch)
            for i in range(len(self._main_wait_chunk_id_by_buf)):
                self._main_wait_chunk_id_by_buf[i] = -1
        if chunk_id is not None:
            cid = int(chunk_id)
            if 0 <= buf < len(self._main_wait_chunk_id_by_buf) and self._main_wait_chunk_id_by_buf[buf] == cid:
                return
            if 0 <= buf < len(self._main_wait_chunk_id_by_buf):
                self._main_wait_chunk_id_by_buf[buf] = cid

        # split-wait：若该 buf 没有任何 pending 工作（prefill/refresh 都未提交），则无需 wait_event。
        # 这能显著降低"开启 async 但 refresh_interval 很大/refresh 很少"的稳态 decode CPU 开销。
        policy = self._get_wait_policy()
        flags = 0
        if 0 <= buf < len(self._buf_pending_work_flags):
            flags = self._buf_pending_work_flags[buf]
        pending_epoch = -1
        if 0 <= buf < len(self._buf_pending_work_epoch):
            pending_epoch = self._buf_pending_work_epoch[buf]
        if flags == 0:
            # 无 pending work 时的早退：避免无意义的 wait_event + per-chunk 管理逻辑。
            # 正确性前提：
            # - 本 buf 未在当前 epoch 记录过实际工作；
            # - 无待释放 prefill / bootstrap pending；
            same_epoch_work = (
                epoch is not None
                and int(epoch) >= 0
                and int(pending_epoch) == int(epoch)
            )
            release_epoch = int(getattr(self, "_prefill_release_pending_epoch", -1))
            need_release = bool(epoch is not None and release_epoch >= 0 and int(epoch) == release_epoch)
            need_bootstrap = (
                self._step_has_decode_consumer()
                and bool(getattr(self, "_bootstrap_pending_request_ids", set()))
            )
            if (not same_epoch_work) and (not need_release) and (not need_bootstrap):
                return
        self._ensure_refresh_stream(device)
        if not self.chunk_done_evt:
            return

        # ------------------------------------------------------------------
        # 选择性等待（split-wait）：尽量等待"最早足够"的 done 事件，减少不必要串行化。
        #
        # - chunk：等待 chunk_done_evt（保持历史行为）
        # - split：根据 buf_pending_work_flags 选择：
        #    - 有 refresh 工作：等 refresh_done_evt（>=compact 写入完成）
        #    - 仅有 prefill 工作：等 prefill_done_evt（prefill selector+rebuild 完成即可复用 ring/compact）
        #
        # 注意：prefill_done_evt/refresh_done_evt 在 refresh_stream 中会"无条件 record"，
        # 因此必须以 flags 作为"该事件是否有意义"的判据，避免误选过早事件。
        # ------------------------------------------------------------------
        evt: Optional[torch.cuda.Event] = None
        if policy == "split":
            if (flags & 2) != 0 and self.refresh_done_evt:
                evt = self.refresh_done_evt[buf]
            elif (flags & 1) != 0 and self.prefill_done_evt:
                evt = self.prefill_done_evt[buf]
        if evt is None:
            evt = self.chunk_done_evt[buf]

        if _CP_PROBE_ENABLED:
            _cp_probe_record_wait(
                evt,
                "main_refresh" if (policy == "split" and (flags & 2))
                else ("main_prefill" if (policy == "split" and (flags & 1))
                      else "main_chunk"),
            )
        torch.cuda.current_stream(device=device).wait_event(evt)
        self._reclaim_retired_buffers()
        commit_flush_compact_meta = getattr(
            self, "_commit_flush_compact_meta_for_buf", None
        )
        if callable(commit_flush_compact_meta):
            commit_flush_compact_meta(buf)
        self._refresh_profile_try_flush_pending(buf=buf, wait_epoch=epoch)
        # 等待已完成后清空 flags，避免后续误判（下一次该 buf 被 flush 会重新设置 flags）。
        self._pending_work_clear_buf(buf_id=buf)

        # decode-only prefill buffer release: wait for all required bufs then release
        release_epoch = int(getattr(self, "_prefill_release_pending_epoch", -1))
        if epoch is not None and release_epoch >= 0 and int(epoch) == release_epoch:
            self._prefill_release_waited_mask |= (1 << int(buf))
            required_mask = self._bootstrap_required_mask()
            if (int(self._prefill_release_waited_mask) & int(required_mask)) == int(required_mask):
                # ensure no pending prefill buckets remain
                pending_prefill = any(int(m) != 0 for m in self.step_prefill_chunk_mask)
                if not pending_prefill:
                    for state in self.layer_states.values():
                        self._maybe_release_prefill_state(state)
                    for idx in range(len(self.step_prefill_capture_layout_ring)):
                        self.step_prefill_capture_layout_ring[idx] = None
                    self._prefill_release_pending_epoch = -1
                    self._prefill_release_waited_mask = 0
                    self._prefill_release_done_epoch = self.step_context_epoch

        if epoch is None:
            return

    # Step-hints consumer
    # ------------------------------------------------------------------

    def _consume_step_wait_token(self, *, buf_id: int, epoch: int) -> bool:
        """按 step hints 判定是否需要对该 buf 执行 wait（并做一次性消费）。"""
        need_wait, _ = self._compute_wait_decision(
            buf_id=int(buf_id),
            epoch=int(epoch),
            path_tag="step_hints",
            consume_step_token=True,
        )
        return bool(need_wait)


    # ------------------------------------------------------------------
