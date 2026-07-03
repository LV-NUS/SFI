from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Any


@dataclass(frozen=True, slots=True)
class ProducerGroupManifest:
    group_id: int
    layer_start: int
    layer_end: int
    step_epoch: int
    snapshot_signature: int
    compact_lease_generation: int
    source_ready_event_generation: int
    expected_rows: int
    expected_slots: tuple[int, ...]
    submitted_stream_id: str


@dataclass(slots=True)
class ProducerReadyState:
    expected_group_mask: int = 0
    submitted_group_mask: int = 0
    completed_group_mask: int = 0
    failed_group_mask: int = 0
    final_event: Any | None = None
    final_event_generation: int = 0
    manifests: dict[int, ProducerGroupManifest] = field(default_factory=dict)
    group_done_events: dict[int, Any] = field(default_factory=dict)
    graph_wait_event_used: bool = False
    event_query_used_for_publish: bool = False
    compact_meta_commit_log: list[dict[str, object]] = field(default_factory=list)


def expected_group_mask_for_layer_count(*, layer_count: int, capture_chunk: int) -> int:
    layer_count_i = int(layer_count)
    capture_chunk_i = int(capture_chunk)
    if layer_count_i <= 0:
        raise ValueError("layer_count must be positive")
    if capture_chunk_i <= 0:
        raise ValueError("capture_chunk must be positive")
    group_count = (layer_count_i + capture_chunk_i - 1) // capture_chunk_i
    return (1 << int(group_count)) - 1


def resolve_one_shot_ready_chunk(*, capture_chunk: int) -> int:
    capture_chunk_i = int(capture_chunk)
    if capture_chunk_i <= 0:
        raise ValueError("capture_chunk must be positive")
    raw = os.environ.get("VLLM_SPARSE_ONE_SHOT_READY_CHUNK", "").strip()
    if not raw:
        return capture_chunk_i
    try:
        ready_chunk = int(raw)
    except ValueError as exc:
        raise ValueError(
            "VLLM_SPARSE_ONE_SHOT_READY_CHUNK must be an integer"
        ) from exc
    if ready_chunk <= 0:
        raise ValueError("VLLM_SPARSE_ONE_SHOT_READY_CHUNK must be positive")
    return ready_chunk


def validate_one_shot_ready_chunk_alignment(
    *,
    capture_chunk: int,
    ready_chunk: int,
) -> None:
    capture_chunk_i = int(capture_chunk)
    ready_chunk_i = int(ready_chunk)
    if capture_chunk_i <= 0:
        raise ValueError("capture_chunk must be positive")
    if ready_chunk_i <= 0:
        raise ValueError("ready_chunk must be positive")
    if ready_chunk_i < capture_chunk_i:
        if (capture_chunk_i % ready_chunk_i) == 0:
            return
        raise ValueError(
            "VLLM_SPARSE_ONE_SHOT_READY_CHUNK must divide "
            "VLLM_SPARSE_CAPTURE_CHUNK when it is smaller"
        )
    if (ready_chunk_i % capture_chunk_i) != 0:
        raise ValueError(
            "VLLM_SPARSE_ONE_SHOT_READY_CHUNK must divide or be a multiple of "
            "VLLM_SPARSE_CAPTURE_CHUNK"
        )


def producer_group_layers_for_group(
    *,
    group_id: int,
    layer_count: int,
    ready_chunk: int,
) -> tuple[int, ...]:
    group_id_i = int(group_id)
    layer_count_i = int(layer_count)
    ready_chunk_i = int(ready_chunk)
    if group_id_i < 0:
        raise ValueError("group_id must be non-negative")
    if layer_count_i <= 0:
        raise ValueError("layer_count must be positive")
    if ready_chunk_i <= 0:
        raise ValueError("ready_chunk must be positive")
    layer_start = group_id_i * ready_chunk_i
    if layer_start >= layer_count_i:
        raise ValueError("producer group starts beyond layer_count")
    layer_end = min(layer_count_i, layer_start + ready_chunk_i)
    return tuple(range(layer_start, layer_end))


def declare_expected_groups(state: ProducerReadyState, *, expected_group_mask: int) -> None:
    mask = int(expected_group_mask)
    if mask <= 0:
        raise ValueError("expected_group_mask must be positive")
    state.expected_group_mask |= mask


def producer_group_id_for_layer(*, layer_index: int, capture_chunk: int) -> int:
    capture_chunk_i = int(capture_chunk)
    if capture_chunk_i <= 0:
        raise ValueError("capture_chunk must be positive")
    layer_index_i = int(layer_index)
    if layer_index_i < 0:
        raise ValueError("layer_index must be non-negative")
    return layer_index_i // capture_chunk_i


def build_producer_group_manifest(
    *,
    group_id: int,
    layer_indices: tuple[int, ...],
    step_epoch: int,
    snapshot_signature: int,
    compact_lease_generation: int,
    source_ready_event_generation: int,
    expected_slots: tuple[int, ...],
    submitted_stream_id: str,
) -> ProducerGroupManifest:
    if not layer_indices:
        raise ValueError("layer_indices must be non-empty")
    layers = tuple(int(layer) for layer in layer_indices)
    if any(layer < 0 for layer in layers):
        raise ValueError("layer_indices must be non-negative")
    slots = tuple(int(slot) for slot in expected_slots)
    return ProducerGroupManifest(
        group_id=int(group_id),
        layer_start=min(layers),
        layer_end=max(layers) + 1,
        step_epoch=int(step_epoch),
        snapshot_signature=int(snapshot_signature),
        compact_lease_generation=int(compact_lease_generation),
        source_ready_event_generation=int(source_ready_event_generation),
        expected_rows=len(slots),
        expected_slots=slots,
        submitted_stream_id=str(submitted_stream_id),
    )


def register_submitted_group(
    state: ProducerReadyState,
    manifest: ProducerGroupManifest,
) -> None:
    group_bit = 1 << int(manifest.group_id)
    state.submitted_group_mask |= group_bit
    state.manifests[int(manifest.group_id)] = manifest


def register_group_done_event(
    state: ProducerReadyState,
    *,
    group_id: int,
    done_event: Any,
) -> None:
    if done_event is None:
        raise RuntimeError("producer group done event is required")
    group_id_i = int(group_id)
    group_bit = 1 << group_id_i
    existing = state.group_done_events.get(group_id_i)
    if existing is not None and existing is not done_event:
        raise RuntimeError("producer group done event changed for group")
    state.group_done_events[group_id_i] = done_event
    state.completed_group_mask |= group_bit


def publish_final_event(
    state: ProducerReadyState,
    *,
    final_event: Any,
) -> None:
    if final_event is None:
        raise RuntimeError("producer final event is required for graph decode publication")
    if state.final_event is not None:
        if state.final_event is final_event:
            return
        raise RuntimeError("producer final event was already published")
    state.final_event = final_event
    state.final_event_generation += 1


def validate_producer_ready_for_publish(state: ProducerReadyState) -> None:
    validate_producer_groups_ready_for_final_event(state)
    if state.final_event is None:
        raise RuntimeError("producer final event is required for graph decode publication")


def validate_producer_groups_ready_for_final_event(state: ProducerReadyState) -> None:
    if int(state.expected_group_mask) <= 0:
        raise RuntimeError("producer expected group mask is required")
    if int(state.submitted_group_mask) != int(state.expected_group_mask):
        raise RuntimeError("producer submitted group mask does not cover expected group mask")
    if int(state.failed_group_mask):
        raise RuntimeError("producer group failed before publication")


def validate_producer_groups_ready_for_graph_wait(state: ProducerReadyState) -> None:
    validate_producer_groups_ready_for_final_event(state)
    if int(state.completed_group_mask) != int(state.expected_group_mask):
        raise RuntimeError("producer completed group mask does not cover expected group mask")
    expected_mask = int(state.expected_group_mask)
    group_id = 0
    while (1 << group_id) <= expected_mask:
        if (expected_mask & (1 << group_id)) != 0:
            if state.group_done_events.get(group_id) is None:
                raise RuntimeError("producer group done event is missing")
        group_id += 1


def producer_ready_summary(state: ProducerReadyState | None) -> dict[str, object]:
    if state is None:
        return {
            "expected_group_mask": -1,
            "submitted_group_mask": -1,
            "completed_group_mask": -1,
            "producer_final_event_present": None,
            "producer_group_done_event_count": -1,
            "graph_wait_event_used": None,
            "event_query_used_for_publish": None,
            "source_ready_event_generation": -1,
        }
    source_ready_generation = max(
        (
            int(manifest.source_ready_event_generation)
            for manifest in state.manifests.values()
        ),
        default=-1,
    )
    return {
        "expected_group_mask": int(state.expected_group_mask),
        "submitted_group_mask": int(state.submitted_group_mask),
        "completed_group_mask": int(state.completed_group_mask),
        "producer_final_event_present": state.final_event is not None,
        "producer_group_done_event_count": int(len(state.group_done_events)),
        "graph_wait_event_used": bool(state.graph_wait_event_used),
        "event_query_used_for_publish": bool(state.event_query_used_for_publish),
        "source_ready_event_generation": int(source_ready_generation),
    }
