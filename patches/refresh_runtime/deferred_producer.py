from __future__ import annotations

from dataclasses import dataclass
import os
import time
from typing import Any

import torch


SUPPORTED_BOOTSTRAP_BRIDGE_GRAPH_POLICIES = frozenset({"evict_recapture_once"})


@dataclass(slots=True)
class DeferredProducerJob:
    request_id: str
    producer_job_epoch: int
    building_compact_epoch: int
    compact_lease_generation: int
    compact_storage_owner: str
    expected_slot: int
    payload_count: int
    bridge_max_tokens: int
    payload_groups: tuple[Any, ...] = ()
    # Producer inputs are frozen while the prefill flush runs on the refresh
    # stream.  Each event is recorded at that ownership boundary, after the
    # corresponding capture-ready dependency.  A later producer launch must
    # wait these events, not a launch-time event on the decode stream: the
    # latter would either submit before anchor attention or, when launched
    # post-forward, serialize selector work behind the whole forward.
    source_ready_events: tuple[Any, ...] = ()
    final_event: Any | None = None
    launched: bool = False
    launched_epoch: int = -1
    last_launch_epoch: int = -1
    next_payload_group_index: int = 0
    completed: bool = False
    cancelled: bool = False
    failure_reason: str = ""


def validate_bridge_graph_policy(policy: str) -> str:
    policy_s = str(policy or "").strip()
    if policy_s not in SUPPORTED_BOOTSTRAP_BRIDGE_GRAPH_POLICIES:
        raise RuntimeError(f"unsupported bootstrap bridge graph policy: {policy_s!r}")
    return policy_s


def build_deferred_bootstrap_job(
    *,
    request_id: str,
    producer_job_epoch: int,
    compact_lease_generation: int,
    expected_slot: int,
    payload_groups: tuple[Any, ...],
    bridge_max_tokens: int,
    source_ready_event: Any | None = None,
) -> DeferredProducerJob:
    if not payload_groups:
        raise RuntimeError("deferred bootstrap producer requires payload groups")
    if int(bridge_max_tokens) <= 0:
        raise RuntimeError("bridge max tokens must be positive")
    payload_groups_t = tuple(payload_groups)
    if source_ready_event is not None and len(payload_groups_t) != 1:
        raise RuntimeError(
            "deferred bootstrap producer requires one source-ready event per "
            "payload group"
        )
    return DeferredProducerJob(
        request_id=str(request_id),
        producer_job_epoch=int(producer_job_epoch),
        building_compact_epoch=int(producer_job_epoch),
        compact_lease_generation=int(compact_lease_generation),
        compact_storage_owner="native_vllm_page_kv",
        expected_slot=int(expected_slot),
        payload_count=len(payload_groups_t),
        bridge_max_tokens=int(bridge_max_tokens),
        payload_groups=payload_groups_t,
        source_ready_events=(
            (source_ready_event,) if source_ready_event is not None else tuple()
        ),
    )


def deferred_payload_count(job: DeferredProducerJob) -> int:
    return len(tuple(job.payload_groups or tuple()))


def ordered_payload_layer_indices(payloads: tuple[Any, ...]) -> tuple[int, ...]:
    """Return the one-to-one global layer identity for a deferred payload group."""

    indices = tuple(
        int(getattr(getattr(payload, "state", None), "layer_index", -1))
        for payload in tuple(payloads)
    )
    if not indices or any(layer < 0 for layer in indices):
        raise RuntimeError(
            "deferred bootstrap producer requires one global layer index per payload"
        )
    if len(set(indices)) != len(indices):
        raise RuntimeError(
            "deferred bootstrap producer payload layer indices must be unique"
        )
    if any(right <= left for left, right in zip(indices, indices[1:])):
        raise RuntimeError(
            "deferred bootstrap producer payload layer indices must be strictly increasing"
        )
    return indices


def append_deferred_payload_group(
    job: DeferredProducerJob,
    payload_group: tuple[Any, ...],
    *,
    source_ready_event: Any | None = None,
) -> None:
    group = tuple(payload_group)
    if not group:
        raise RuntimeError("deferred bootstrap producer requires payload groups")
    source_ready_events = tuple(job.source_ready_events or tuple())
    payload_group_count = deferred_payload_count(job)
    if source_ready_events and len(source_ready_events) != payload_group_count:
        raise RuntimeError(
            "deferred producer source-ready event count does not match payload groups"
        )
    if source_ready_events and source_ready_event is None:
        raise RuntimeError(
            "deferred bootstrap producer requires one source-ready event per "
            "payload group"
        )
    if not source_ready_events and source_ready_event is not None and payload_group_count:
        raise RuntimeError(
            "deferred bootstrap producer cannot append a source-ready event to "
            "an eventless payload snapshot"
        )
    job.payload_groups = tuple(job.payload_groups or tuple()) + (group,)
    job.payload_count = deferred_payload_count(job)
    if source_ready_event is not None:
        job.source_ready_events = tuple(job.source_ready_events or tuple()) + (
            source_ready_event,
        )


def validate_deferred_producer_job_for_publish(
    job: DeferredProducerJob,
    *,
    request_id: str,
    producer_epoch: int,
    compact_lease_generation: int,
    compact_storage_owner: str,
) -> None:
    if bool(job.cancelled):
        raise RuntimeError("deferred producer job was cancelled")
    if str(job.failure_reason):
        raise RuntimeError(f"deferred producer job failed: {job.failure_reason}")
    if str(job.request_id) != str(request_id):
        raise RuntimeError("deferred producer request id mismatch")
    if int(job.producer_job_epoch) != int(producer_epoch):
        raise RuntimeError("deferred producer epoch mismatch")
    if int(job.building_compact_epoch) != int(producer_epoch):
        raise RuntimeError("building compact epoch mismatch")
    if int(job.compact_lease_generation) != int(compact_lease_generation):
        raise RuntimeError("compact lease generation mismatch")
    if str(job.compact_storage_owner) != str(compact_storage_owner):
        raise RuntimeError("compact storage owner mismatch")
    if int(job.expected_slot) < 0:
        raise RuntimeError("deferred producer expected slot is invalid")
    actual_payload_count = deferred_payload_count(job)
    if actual_payload_count <= 0:
        raise RuntimeError("deferred producer payload count is invalid")
    if int(job.payload_count) != int(actual_payload_count):
        raise RuntimeError("deferred producer payload count mismatch")
    if int(job.bridge_max_tokens) <= 0:
        raise RuntimeError("bridge max tokens must be positive")




def deferred_producer_job_summary(job: DeferredProducerJob | None) -> dict[str, object]:
    if job is None:
        return {
            "deferred_producer_job_present": False,
            "producer_job_epoch": -1,
            "building_compact_epoch": -1,
            "bridge_max_tokens": -1,
            "deferred_producer_cancelled": None,
            "deferred_producer_failure_reason": "",
        }
    return {
        "deferred_producer_job_present": True,
        "producer_job_epoch": int(job.producer_job_epoch),
        "building_compact_epoch": int(job.building_compact_epoch),
        "bridge_max_tokens": int(job.bridge_max_tokens),
        "payload_count": int(deferred_payload_count(job)),
        "source_ready_event_count": len(
            tuple(getattr(job, "source_ready_events", tuple()) or tuple())
        ),
        "expected_slot": int(job.expected_slot),
        "compact_lease_generation": int(job.compact_lease_generation),
        "compact_storage_owner": str(job.compact_storage_owner),
        "deferred_producer_launched": bool(job.launched),
        "deferred_producer_launched_epoch": int(job.launched_epoch),
        "deferred_producer_last_launch_epoch": int(job.last_launch_epoch),
        "deferred_producer_next_payload_group_index": int(
            job.next_payload_group_index
        ),
        "deferred_producer_completed": bool(job.completed),
        "deferred_producer_cancelled": bool(job.cancelled),
        "deferred_producer_failure_reason": str(job.failure_reason),
    }


def run_deferred_bootstrap_producer_job(
    controller: Any,
    job: DeferredProducerJob,
    *,
    capture_chunk: int,
    max_groups_per_call: int = 0,
) -> bool:
    from patches.refresh_runtime.producer_ready import (
        ProducerReadyState,
        build_producer_group_manifest,
        declare_expected_groups,
        expected_group_mask_for_layer_count,
        producer_group_id_for_layer,
        producer_group_layers_for_group,
        publish_final_event,
        register_group_done_event,
        register_submitted_group,
        resolve_one_shot_ready_chunk,
        validate_one_shot_ready_chunk_alignment,
        validate_producer_groups_ready_for_final_event,
    )
    from patches.fa3_native.postprocess import (
        run_capture_postprocess_jobs_for_payloads,
    )
    def _profile_deferred_producer_call(
        *,
        label: str,
        metadata: dict[str, object],
        call: Any,
    ) -> Any:
        return call()

    def _fail_deferred_producer(reason: str) -> None:
        job.failure_reason = str(reason)
        raise RuntimeError(str(reason))

    if bool(job.cancelled):
        raise RuntimeError("deferred producer job was cancelled")
    failure_reason = str(job.failure_reason or "")
    if failure_reason:
        raise RuntimeError(f"deferred producer job failed: {failure_reason}")

    if not controller._async_refresh_enabled():
        _fail_deferred_producer("deferred bootstrap producer requires async refresh")

    payload_groups = tuple(job.payload_groups or tuple())
    if not payload_groups:
        _fail_deferred_producer("deferred bootstrap producer requires payload groups")
    if bool(job.completed):
        return True
    start_group_index = int(job.next_payload_group_index)
    if start_group_index < 0 or start_group_index > len(payload_groups):
        _fail_deferred_producer("deferred producer group cursor is invalid")
    group_limit = int(max_groups_per_call)
    if group_limit <= 0:
        end_group_index = len(payload_groups)
    else:
        end_group_index = min(len(payload_groups), start_group_index + group_limit)
    if start_group_index >= end_group_index:
        return False
    first_group = tuple(payload_groups[0])
    if not first_group:
        _fail_deferred_producer("deferred bootstrap producer requires payload groups")
    device = getattr(getattr(first_group[0], "capture_scores", None), "device", None)
    ensure_refresh_stream = getattr(controller, "_ensure_refresh_stream", None)
    if callable(ensure_refresh_stream):
        ensure_refresh_stream(device)
    refresh_stream = getattr(controller, "refresh_stream", None)
    if refresh_stream is None:
        _fail_deferred_producer("deferred bootstrap producer requires refresh_stream")
    source_ready_events = tuple(
        getattr(job, "source_ready_events", tuple()) or tuple()
    )
    if not source_ready_events:
        _fail_deferred_producer(
            "deferred bootstrap producer requires frozen source-ready events"
        )
    if len(source_ready_events) != len(payload_groups):
        _fail_deferred_producer(
            "deferred producer source-ready event count does not match payload groups"
        )

    rid = str(job.request_id)
    tracking = controller.request_states.get(rid)
    if tracking is None:
        tracking = controller._ensure_request(rid)

    def _append_deferred_timeline(
        *,
        phase: str,
        timestamp_ns: int | None = None,
        duration_us: float = 0.0,
        chunk_id: int = -1,
        extra_fields: dict[str, object] | None = None,
    ) -> None:
        if not os.environ.get("VLLM_SPARSE_ONE_SHOT_TIMELINE_LOG", ""):
            return
        try:
            from patches.refresh_runtime.one_shot_timeline import (
                append_one_shot_timeline,
                timeline_log_path,
            )

            append_one_shot_timeline(
                path=timeline_log_path(),
                request_id=rid,
                epoch=int(job.producer_job_epoch),
                chunk_id=int(chunk_id),
                buffer_id=-1,
                phase=str(phase),
                timestamp_ns=(
                    time.perf_counter_ns()
                    if timestamp_ns is None
                    else int(timestamp_ns)
                ),
                duration_us=float(duration_us),
                aux_stream_enabled=True,
                extra_fields=extra_fields,
            )
        except Exception:
            pass

    ready_state = getattr(tracking, "producer_ready_state", None)
    if ready_state is None:
        ready_state = ProducerReadyState()
        tracking.producer_ready_state = ready_state
    capture_chunk_i = int(capture_chunk)
    ready_chunk = resolve_one_shot_ready_chunk(capture_chunk=capture_chunk_i)
    validate_one_shot_ready_chunk_alignment(
        capture_chunk=capture_chunk_i,
        ready_chunk=int(ready_chunk),
    )
    declare_expected_groups(
        ready_state,
        expected_group_mask=expected_group_mask_for_layer_count(
            layer_count=len(controller.layer_cache_keys),
            capture_chunk=int(ready_chunk),
        ),
    )
    writer_pointer_snapshot = None
    snapshot_writer = getattr(controller, "_writer_pointer_telemetry_snapshot", None)
    if callable(snapshot_writer):
        try:
            writer_pointer_snapshot = snapshot_writer()
        except Exception:
            writer_pointer_snapshot = None

    with torch.cuda.stream(refresh_stream):
        producer_stream = torch.cuda.current_stream(device=device)
        waited_source_events: set[int] = set()
        for source_ready_event in source_ready_events[
            start_group_index:end_group_index
        ]:
            event_id = id(source_ready_event)
            if event_id in waited_source_events:
                continue
            producer_stream.wait_event(source_ready_event)
            waited_source_events.add(event_id)
        for payload_group in payload_groups[start_group_index:end_group_index]:
            group_payloads = tuple(payload_group)
            if not group_payloads:
                _fail_deferred_producer(
                    "deferred bootstrap producer requires payload groups"
                )
            payload_group_index = int(job.next_payload_group_index)
            producer_group_metadata = {
                "request_id": rid,
                "producer_job_epoch": int(job.producer_job_epoch),
                "payload_group_index": int(payload_group_index),
                "payload_count": len(group_payloads),
            }
            try:
                layer_indices = ordered_payload_layer_indices(group_payloads)
            except RuntimeError as exc:
                _fail_deferred_producer(str(exc))
            group_id = producer_group_id_for_layer(
                layer_index=min(layer_indices),
                capture_chunk=int(ready_chunk),
            )
            if int(ready_chunk) == int(capture_chunk_i):
                ready_subgroups = [
                    (
                        int(group_id),
                        0,
                        tuple(group_payloads),
                        tuple(layer_indices),
                    )
                ]
            else:
                ready_subgroups: list[
                    tuple[int, int, tuple[Any, ...], tuple[int, ...]]
                ] = []
                current_ready_group_id = -1
                current_start = 0
                current_payloads: list[Any] = []
                current_layers: list[int] = []
                seen_ready_groups: set[int] = set()
                for local_index, payload in enumerate(group_payloads):
                    layer = int(layer_indices[int(local_index)])
                    ready_group_id = producer_group_id_for_layer(
                        layer_index=int(layer),
                        capture_chunk=int(ready_chunk),
                    )
                    if current_payloads and ready_group_id != current_ready_group_id:
                        if int(current_ready_group_id) in seen_ready_groups:
                            _fail_deferred_producer(
                                "one-shot ready chunk groups must be contiguous"
                            )
                        seen_ready_groups.add(int(current_ready_group_id))
                        ready_subgroups.append(
                            (
                                int(current_ready_group_id),
                                int(current_start),
                                tuple(current_payloads),
                                tuple(current_layers),
                            )
                        )
                        current_payloads = []
                        current_layers = []
                        current_start = int(local_index)
                    if not current_payloads:
                        current_ready_group_id = int(ready_group_id)
                        current_start = int(local_index)
                    current_payloads.append(payload)
                    current_layers.append(int(layer))
                if current_payloads:
                    if int(current_ready_group_id) in seen_ready_groups:
                        _fail_deferred_producer(
                            "one-shot ready chunk groups must be contiguous"
                        )
                    ready_subgroups.append(
                        (
                            int(current_ready_group_id),
                            int(current_start),
                            tuple(current_payloads),
                            tuple(current_layers),
                        )
                    )
            ready_group_ids = tuple(int(item[0]) for item in ready_subgroups)
            ready_groups_complete = True
            for ready_group_id, _, _, ready_group_layers in ready_subgroups:
                required_layers = producer_group_layers_for_group(
                    group_id=int(ready_group_id),
                    layer_count=len(controller.layer_cache_keys),
                    ready_chunk=int(ready_chunk),
                )
                if not set(int(v) for v in required_layers).issubset(
                    set(int(v) for v in ready_group_layers)
                ):
                    ready_groups_complete = False
                    break
            group_start_ns = time.perf_counter_ns()
            _append_deferred_timeline(
                phase="deferred_producer_group_start",
                timestamp_ns=group_start_ns,
                chunk_id=int(group_id),
                extra_fields={
                    **producer_group_metadata,
                    "layer_indices": list(layer_indices),
                    "producer_capture_chunk": int(capture_chunk_i),
                    "producer_ready_chunk": int(ready_chunk),
                    "producer_group_complete": bool(ready_groups_complete),
                    "producer_ready_group_ids": list(ready_group_ids),
                    "bootstrap_full_kv_handoff": True,
                },
            )
            capture_postprocess_job_count = sum(
                1
                for payload in group_payloads
                if getattr(payload, "capture_postprocess_job", None) is not None
            )
            if capture_postprocess_job_count:
                postprocess_metadata = {
                    **producer_group_metadata,
                    "capture_postprocess_job_count": int(
                        capture_postprocess_job_count
                    ),
                }

                def _run_capture_postprocess_group() -> int:
                    return run_capture_postprocess_jobs_for_payloads(
                        group_payloads,
                        meta_cache_owner=controller,
                    )

                capture_postprocess_ran_count = _profile_deferred_producer_call(
                    label="capture_postprocess_deferred",
                    metadata=postprocess_metadata,
                    call=_run_capture_postprocess_group,
                )
            else:
                capture_postprocess_ran_count = 0
            selector_per_ready_group = len(ready_subgroups) > 1
            prefill_result = None
            if not selector_per_ready_group:
                from patches.fa3_native.capture_cohort_tape import (
                    require_capture_cohort_selector_view,
                )

                require_capture_cohort_selector_view(group_payloads)
                prefill_result = _profile_deferred_producer_call(
                    label="deferred_prefill_selector",
                    metadata={
                        **producer_group_metadata,
                        "capture_postprocess_ran_count": int(
                            capture_postprocess_ran_count
                        ),
                    },
                    call=lambda: controller._apply_alpha_selector_batched_fused(
                        group_payloads,
                        phase="prefill",
                    ),
                )
                if prefill_result is None:
                    _fail_deferred_producer(
                        "deferred bootstrap producer selector returned no result"
                    )
            selector_run_count = (
                len(ready_subgroups) if selector_per_ready_group else 1
            )
            source_ready_generation = max(
                (
                    int(manifest.source_ready_event_generation)
                    for manifest in ready_state.manifests.values()
                ),
                default=0,
            )
            for (
                ready_group_id,
                _local_start,
                ready_group_payloads,
                ready_group_layers,
            ) in ready_subgroups:
                group_layer_indices = producer_group_layers_for_group(
                    group_id=int(ready_group_id),
                    layer_count=len(controller.layer_cache_keys),
                    ready_chunk=int(ready_chunk),
                )
                group_complete = set(int(v) for v in group_layer_indices).issubset(
                    set(int(v) for v in ready_group_layers)
                )
                writer_result = prefill_result
                if selector_per_ready_group:
                    from patches.fa3_native.capture_cohort_tape import (
                        require_capture_cohort_selector_view,
                    )

                    require_capture_cohort_selector_view(ready_group_payloads)
                    writer_result = _profile_deferred_producer_call(
                        label="deferred_prefill_selector",
                        metadata={
                            **producer_group_metadata,
                            "payload_count": len(ready_group_payloads),
                            "ready_group_id": int(ready_group_id),
                            "capture_postprocess_ran_count": int(
                                capture_postprocess_ran_count
                            ),
                        },
                        call=(
                            lambda ready_group_payloads=ready_group_payloads: controller._apply_alpha_selector_batched_fused(
                                ready_group_payloads,
                                phase="prefill",
                            )
                        ),
                    )
                if writer_result is None:
                    _fail_deferred_producer(
                        "deferred bootstrap producer selector result missing"
                    )
                fused_ok = _profile_deferred_producer_call(
                    label="deferred_prefill_rebuild",
                    metadata={
                        **producer_group_metadata,
                        "payload_count": len(ready_group_payloads),
                        "ready_group_id": int(ready_group_id),
                        "capture_postprocess_ran_count": int(
                            capture_postprocess_ran_count
                        ),
                    },
                    call=lambda ready_group_payloads=ready_group_payloads, writer_result=writer_result: controller._rebuild_compact_slots_batched_layers_from_selection(
                        ready_group_payloads,
                        writer_result.selected_indices,
                        phase="prefill",
                        bootstrap_slots_by_layer=[
                            payload.bootstrap_slots
                            for payload in ready_group_payloads
                        ],
                        defer_compact_meta_publish=True,
                        compact_meta_commit_log=ready_state.compact_meta_commit_log,
                    ),
                )
                if not fused_ok:
                    _fail_deferred_producer(
                        "prefill compact rebuild: fused gather failed"
                    )
                if not group_complete:
                    continue
                group_done_event = None
                event_registry = getattr(
                    controller,
                    "_one_shot_group_done_evt_by_group",
                    None,
                )
                if event_registry is None:
                    event_registry = {}
                    controller._one_shot_group_done_evt_by_group = event_registry
                if not isinstance(event_registry, dict):
                    _fail_deferred_producer(
                        "one-shot group-ready event registry is invalid"
                    )
                group_done_event = event_registry.get(int(ready_group_id))
                if group_done_event is None:
                    group_done_event = torch.cuda.Event(enable_timing=False)
                    event_registry[int(ready_group_id)] = group_done_event
                slot_event_registry = getattr(
                    controller,
                    "_one_shot_group_done_evt_by_group_slot",
                    None,
                )
                if slot_event_registry is None:
                    slot_event_registry = {}
                    controller._one_shot_group_done_evt_by_group_slot = (
                        slot_event_registry
                    )
                if not isinstance(slot_event_registry, dict):
                    _fail_deferred_producer(
                        "one-shot group-ready slot event registry is invalid"
                    )
                slot_event_key = (int(ready_group_id), int(job.expected_slot))
                slot_group_done_event = slot_event_registry.get(slot_event_key)
                if slot_group_done_event is None:
                    slot_group_done_event = torch.cuda.Event(enable_timing=False)
                    slot_event_registry[slot_event_key] = slot_group_done_event
                event_serial = int(
                    getattr(controller, "_one_shot_group_done_evt_serial", 0) or 0
                ) + 1
                controller._one_shot_group_done_evt_serial = int(event_serial)
                serial_registry = getattr(
                    controller,
                    "_one_shot_group_done_evt_serial_by_group",
                    None,
                )
                if not isinstance(serial_registry, dict):
                    serial_registry = {}
                    controller._one_shot_group_done_evt_serial_by_group = (
                        serial_registry
                    )
                slot_serial_registry = getattr(
                    controller,
                    "_one_shot_group_done_evt_serial_by_group_slot",
                    None,
                )
                if not isinstance(slot_serial_registry, dict):
                    slot_serial_registry = {}
                    controller._one_shot_group_done_evt_serial_by_group_slot = (
                        slot_serial_registry
                    )
                serial_registry[int(ready_group_id)] = int(event_serial)
                slot_serial_registry[slot_event_key] = int(event_serial)
                group_done_event.record(torch.cuda.current_stream(device=device))
                slot_group_done_event.record(torch.cuda.current_stream(device=device))

                source_ready_generation += 1
                manifest = build_producer_group_manifest(
                    group_id=int(ready_group_id),
                    layer_indices=group_layer_indices,
                    step_epoch=int(job.producer_job_epoch),
                    snapshot_signature=int(
                        getattr(group_payloads[0], "capture_handle_generation", 0) or 0
                    ),
                    compact_lease_generation=int(job.compact_lease_generation),
                    source_ready_event_generation=int(source_ready_generation),
                    expected_slots=(int(job.expected_slot),),
                    submitted_stream_id="refresh_stream",
                )
                register_submitted_group(ready_state, manifest)
                register_group_done_event(
                    ready_state,
                    group_id=int(ready_group_id),
                    done_event=slot_group_done_event,
                )
            from patches.fa3_native.capture_cohort_tape import (
                release_capture_cohort_payload_group,
                resolve_capture_cohort_tape_group,
            )

            # Only the prebuilt tape owns a cross-step lease.  The ring path has
            # no such lifetime to retire.  All tape reads and subsequent writes
            # use this same producer stream, so FIFO order is the reuse fence.
            if resolve_capture_cohort_tape_group(group_payloads) is not None:
                release_capture_cohort_payload_group(
                    controller=controller,
                    payloads=group_payloads,
                    consumer_stream=producer_stream,
                )
            group_end_ns = time.perf_counter_ns()
            _append_deferred_timeline(
                phase="deferred_producer_group_end",
                timestamp_ns=group_end_ns,
                duration_us=float(group_end_ns - group_start_ns) / 1000.0,
                chunk_id=int(group_id),
                extra_fields={
                    **producer_group_metadata,
                    "layer_indices": list(layer_indices),
                    "deferred_capture_postprocess_jobs": int(
                        capture_postprocess_job_count
                    ),
                    "deferred_capture_postprocess_runs": int(
                        capture_postprocess_ran_count
                    ),
                    "expected_group_mask": int(ready_state.expected_group_mask),
                    "submitted_group_mask": int(ready_state.submitted_group_mask),
                    "source_ready_event_generation": int(source_ready_generation),
                    "prefill_selector_per_ready_group": bool(
                        selector_per_ready_group
                    ),
                    "producer_capture_chunk": int(capture_chunk_i),
                    "producer_ready_chunk": int(ready_chunk),
                    "producer_group_complete": bool(ready_groups_complete),
                    "producer_ready_group_ids": list(ready_group_ids),
                    "bootstrap_full_kv_handoff": True,
                },
            )
            if ready_groups_complete and os.environ.get("VLLM_SPARSE_ONE_SHOT_TIMELINE_LOG", ""):
                try:
                    from patches.refresh_runtime.one_shot_timeline import (
                        append_one_shot_timeline,
                        timeline_log_path,
                    )

                    rebuild_run_count = (
                        len(ready_subgroups)
                    )
                    for ready_group_id, _, _, ready_group_layers in ready_subgroups:
                        append_one_shot_timeline(
                            path=timeline_log_path(),
                            request_id=rid,
                            epoch=int(job.producer_job_epoch),
                            chunk_id=int(ready_group_id),
                            buffer_id=-1,
                            phase="deferred_producer_group_submitted",
                            timestamp_ns=time.perf_counter_ns(),
                            duration_us=0.0,
                            aux_stream_enabled=True,
                            extra_fields={
                                "prefill_selector_runs": 1,
                                "prefill_rebuild_runs": 1,
                                "deferred_prefill_selector_runs": 1,
                                "deferred_prefill_rebuild_runs": 1,
                                "producer_group_prefill_selector_runs": int(
                                    selector_run_count
                                ),
                                "producer_group_prefill_rebuild_runs": int(
                                    rebuild_run_count
                                ),
                                "deferred_capture_postprocess_jobs": int(
                                    capture_postprocess_job_count
                                ),
                                "deferred_capture_postprocess_runs": int(
                                    capture_postprocess_ran_count
                                ),
                                "expected_group_mask": int(
                                    ready_state.expected_group_mask
                                ),
                                "submitted_group_mask": int(
                                    ready_state.submitted_group_mask
                                ),
                                "source_ready_event_generation": int(
                                    source_ready_generation
                                ),
                                "prefill_selector_per_ready_group": bool(
                                    selector_per_ready_group
                                ),
                                "producer_capture_chunk": int(capture_chunk_i),
                                "producer_ready_chunk": int(ready_chunk),
                                "producer_micro_layer_indices": list(layer_indices),
                                "producer_group_layer_indices": list(
                                    ready_group_layers
                                ),
                                "bootstrap_full_kv_handoff": True,
                            },
                        )
                except Exception:
                    pass
            job.next_payload_group_index = int(job.next_payload_group_index) + 1

        if int(job.next_payload_group_index) < len(payload_groups):
            return False
        try:
            validate_deferred_producer_job_for_publish(
                job,
                request_id=rid,
                producer_epoch=int(job.producer_job_epoch),
                compact_lease_generation=int(job.compact_lease_generation),
                compact_storage_owner=str(job.compact_storage_owner),
            )
            validate_producer_groups_ready_for_final_event(ready_state)
        except Exception as exc:
            missing_mask = int(ready_state.expected_group_mask) & ~int(
                ready_state.submitted_group_mask
            )
            if missing_mask:
                ready_state.failed_group_mask |= int(missing_mask)
            _fail_deferred_producer(str(exc))
        final_event = torch.cuda.Event(enable_timing=False)
        final_event.record(torch.cuda.current_stream(device=device))
    job.final_event = final_event
    job.completed = True
    publish_final_event(ready_state, final_event=final_event)
    writer_telemetry: dict[str, object] = {}
    writer_delta = getattr(controller, "_writer_pointer_telemetry_delta", None)
    if callable(writer_delta):
        try:
            (
                writer_pointer_rebuild_count,
                writer_pointer_lookup_count,
                writer_cached_pointer_hit_rate,
                writer_cached_pointer_op_count,
                writer_vector_fallback_count,
                source_ready_recorded_after_pointer_publish_count,
            ) = writer_delta(writer_pointer_snapshot)
            writer_telemetry = {
                "writer_pointer_rebuild_count": int(writer_pointer_rebuild_count),
                "writer_pointer_lookup_count": int(writer_pointer_lookup_count),
                "writer_cached_pointer_hit_rate": float(
                    writer_cached_pointer_hit_rate
                ),
                "writer_cached_pointer_op_count": int(writer_cached_pointer_op_count),
                "writer_vector_fallback_count": int(writer_vector_fallback_count),
                "source_ready_recorded_after_pointer_publish_count": int(
                    source_ready_recorded_after_pointer_publish_count
                ),
            }
        except Exception:
            writer_telemetry = {}
    if os.environ.get("VLLM_SPARSE_ONE_SHOT_TIMELINE_LOG", ""):
        try:
            from patches.refresh_runtime.one_shot_timeline import (
                append_one_shot_timeline,
                timeline_log_path,
            )

            append_one_shot_timeline(
                path=timeline_log_path(),
                request_id=rid,
                epoch=int(job.producer_job_epoch),
                chunk_id=-1,
                buffer_id=-1,
                phase="deferred_producer_final_event_record",
                timestamp_ns=time.perf_counter_ns(),
                duration_us=0.0,
                aux_stream_enabled=True,
                extra_fields={
                    "expected_group_mask": int(ready_state.expected_group_mask),
                    "submitted_group_mask": int(ready_state.submitted_group_mask),
                    "producer_final_event_recorded": True,
                    "bootstrap_full_kv_handoff": True,
                    **writer_telemetry,
                },
            )
        except Exception:
            pass
    pending_events = getattr(tracking, "bootstrap_pending_events", None)
    if pending_events is None:
        tracking.bootstrap_pending_events = [final_event]
    elif final_event not in pending_events:
        pending_events.append(final_event)
    return True
