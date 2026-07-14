"""Replay-prep helpers for mixed-page CUDA graph carriers."""

from __future__ import annotations

from dataclasses import dataclass

from patches.fa_sparse_runtime.effective_page_table_producer import (
    refresh_mixed_page_resolver_carriers_for_replay,
)
from patches.fa_sparse_runtime.resolved_row_ptr_arena import build_source_counter_fields
_CARRIER_SIGNATURE_ATTRS = (
    "row_consume_mode_i32",
    "resolver_visible_seqused_k_by_head_i32",
    "selected_page_table_i32",
    "effective_row_slot_i32",
    "compact_base_page_i32",
    "compact_page_count_i32",
    "recent_first_logical_page_i32",
    "resolved_page_table_row_ptr_u64",
    "resolved_page_table_affine_i32",
)
_ROW_MODES = ("compact", "safe_full_recent", "native", "unset")
_RESOLVED_ROW_PTR_REPLAY_TENSOR_ATTR_PAIRS = (
    (
        "mixed_page_resolver_replay_row_table_i32",
        "mixed_page_resolver_replay_row_table_source_i32",
    ),
    (
        "mixed_page_resolver_replay_seqused_k_i32",
        "mixed_page_resolver_replay_seqused_k_source_i32",
    ),
)
_RESOLVED_ROW_PTR_READY_EVENT_ATTR = "mixed_page_resolver_replay_ready_event"
_RESOLVED_ROW_PTR_READY_EVENT_GENERATION_ATTR = (
    "mixed_page_resolver_replay_ready_event_generation"
)
_RESOLVED_ROW_PTR_READY_EVENT_STREAM_ATTR = (
    "mixed_page_resolver_replay_ready_event_stream"
)
_RESOLVED_ROW_PTR_SAME_STREAM_ORDERED_ATTR = (
    "mixed_page_resolver_replay_same_stream_ordered"
)


@dataclass(frozen=True, slots=True)
class ResolvedRowPtrReadyState:
    """Validated writer-to-replay publication state.

    ``generation == -1`` is reserved for an unpublished arena.  Every replay
    publication has a non-negative generation and exactly one ordering proof:
    an event, or a proven same-stream edge.
    """

    event: object | None
    generation: int
    ready_stream_id: int
    same_stream_ordered: bool
    published: bool


def validate_resolved_row_ptr_ready_state(
    *,
    event: object | None,
    generation: object,
    ready_stream: object,
    same_stream_ordered: object,
    require_published: bool,
    context: str,
) -> ResolvedRowPtrReadyState:
    """Validate the single RRP generation/event contract used by replay paths."""

    # These values cross an async ownership boundary.  Silent coercion (for
    # example, ``True -> 1`` or ``3.5 -> 3``) can alias a real generation or
    # stream and incorrectly admit replay, so the publication contract uses
    # exact Python integers only.
    if type(generation) is not int:
        raise RuntimeError(f"{context} has malformed ready-event state")
    if ready_stream is None:
        ready_stream_i = -1
    elif type(ready_stream) is int:
        ready_stream_i = ready_stream
    else:
        raise RuntimeError(f"{context} has malformed ready-event state")
    generation_i = generation
    if not isinstance(same_stream_ordered, bool):
        raise RuntimeError(f"{context} has malformed same-stream ordering state")
    ordered = same_stream_ordered
    if event is not None and ordered:
        raise RuntimeError(
            f"{context} cannot publish both a ready event and same-stream ordering"
        )
    published = event is not None or ordered
    if not published:
        if generation_i != -1 or ready_stream_i != -1:
            raise RuntimeError(
                f"{context} has an unpublished ready state with published generation metadata"
            )
        if require_published:
            raise RuntimeError(f"{context} is not replay-ready")
        return ResolvedRowPtrReadyState(
            event=None,
            generation=-1,
            ready_stream_id=-1,
            same_stream_ordered=False,
            published=False,
        )
    if generation_i < 0:
        raise RuntimeError(
            f"{context} requires a non-negative ready-event generation"
        )
    if ordered and ready_stream_i < 0:
        raise RuntimeError(
            f"{context} requires a raw CUDA stream for same-stream ordering"
        )
    return ResolvedRowPtrReadyState(
        event=event,
        generation=generation_i,
        ready_stream_id=ready_stream_i,
        same_stream_ordered=ordered,
        published=True,
    )


def _build_row_mode_distribution(row_modes: object) -> dict[str, int]:
    distribution = {mode: 0 for mode in _ROW_MODES}
    for mode in row_modes:
        if mode not in distribution:
            raise ValueError(f"unsupported row mode: {mode!r}")
        distribution[mode] += 1
    return distribution


@dataclass(frozen=True, slots=True)
class MixedPageForwardContextReplayStats:
    metadata_count: int
    updated_metadata_count: int
    carrier_update_rows: int
    carrier_update_bytes: int
    carrier_update_kernel_count: int
    row_mode_distribution: dict[str, int]
    row_source_distribution: dict[str, int]
    source_counter_schema_version: int
    expected_rows: int
    num_kv_heads: int
    source_counter_missing_fields: tuple[str, ...]
    step_id: int
    graph_key: str
    ready_event_wait_count: int = 0


def iter_attention_metadata(attn_metadata: object) -> tuple[object, ...]:
    if attn_metadata is None:
        return ()
    if isinstance(attn_metadata, dict):
        flattened: list[object] = []
        for value in attn_metadata.values():
            flattened.extend(iter_attention_metadata(value))
        return tuple(flattened)
    if isinstance(attn_metadata, (list, tuple)):
        flattened = []
        for value in attn_metadata:
            flattened.extend(iter_attention_metadata(value))
        return tuple(flattened)
    return (attn_metadata,)


def metadata_has_mixed_page_replay_update(metadata: object) -> bool:
    return any(
        getattr(metadata, attr_name, None) is not None
        for attr_name in (
            "mixed_page_resolver_replay_carrier_updates",
            "mixed_page_resolver_replay_carriers",
            "mixed_page_resolver_replay_carrier_sources",
            "mixed_page_resolver_replay_row_table_i32",
            "mixed_page_resolver_replay_row_table_source_i32",
            "mixed_page_resolver_replay_seqused_k_i32",
            "mixed_page_resolver_replay_seqused_k_source_i32",
        )
    )


def _cuda_stream_identity(stream: object | None) -> int:
    if stream is None:
        return -1
    raw = getattr(stream, "cuda_stream", None)
    if raw is not None:
        try:
            return int(raw)
        except (TypeError, ValueError):
            pass
    return int(id(stream))


def wait_mixed_page_resolver_ready_events_for_forward_context(
    forward_context: object,
    *,
    stream: object | None = None,
) -> int:
    """Order direct-bound RRP carrier writes before CUDA graph replay."""
    attn_metadata = getattr(forward_context, "attn_metadata", None)
    wait_count = 0
    wait_stream = stream
    wait_stream_id = -1
    raw_stream_identity_available = False
    stream_identity_resolved = False
    seen: set[tuple[int, int]] = set()

    def _resolve_wait_stream_identity() -> tuple[int, bool]:
        nonlocal wait_stream
        nonlocal wait_stream_id
        nonlocal raw_stream_identity_available
        nonlocal stream_identity_resolved
        if stream_identity_resolved:
            return wait_stream_id, raw_stream_identity_available
        stream_identity_resolved = True
        if wait_stream is not None:
            raw = getattr(wait_stream, "cuda_stream", None)
            if raw is not None:
                try:
                    wait_stream_id = int(raw)
                    raw_stream_identity_available = True
                    return wait_stream_id, True
                except (TypeError, ValueError):
                    pass
            wait_stream_id = _cuda_stream_identity(wait_stream)
            return wait_stream_id, False
        import torch

        try:
            get_current_raw_stream = getattr(
                getattr(torch, "_C", None),
                "_cuda_getCurrentRawStream",
                None,
            )
            current_device = getattr(torch.cuda, "current_device", None)
            if callable(get_current_raw_stream) and callable(current_device):
                wait_stream_id = int(
                    get_current_raw_stream(int(current_device()))
                )
                raw_stream_identity_available = True
                return wait_stream_id, True
        except Exception:
            pass
        wait_stream = torch.cuda.current_stream()
        raw = getattr(wait_stream, "cuda_stream", None)
        if raw is not None:
            try:
                wait_stream_id = int(raw)
                raw_stream_identity_available = True
                return wait_stream_id, True
            except (TypeError, ValueError):
                pass
        wait_stream_id = _cuda_stream_identity(wait_stream)
        return wait_stream_id, False

    for metadata in iter_attention_metadata(attn_metadata):
        holders = (
            metadata,
            getattr(metadata, "mixed_page_resolver_replay_arena", None),
            getattr(metadata, "mixed_page_resolver_source_arena", None),
        )
        for holder in holders:
            if holder is None:
                continue
            ready_state = validate_resolved_row_ptr_ready_state(
                event=getattr(holder, _RESOLVED_ROW_PTR_READY_EVENT_ATTR, None),
                generation=getattr(
                    holder, _RESOLVED_ROW_PTR_READY_EVENT_GENERATION_ATTR, -1
                ),
                ready_stream=getattr(
                    holder, _RESOLVED_ROW_PTR_READY_EVENT_STREAM_ATTR, -1
                ),
                same_stream_ordered=getattr(
                    holder, _RESOLVED_ROW_PTR_SAME_STREAM_ORDERED_ATTR, False
                ),
                require_published=False,
                context="mixed-page CUDA graph replay RRP metadata",
            )
            if not ready_state.published:
                continue
            event = ready_state.event
            generation = ready_state.generation
            ready_stream_id = ready_state.ready_stream_id
            same_stream_ordered = ready_state.same_stream_ordered
            if event is None and same_stream_ordered:
                current_stream_id, identity_proven = _resolve_wait_stream_identity()
                if (
                    not identity_proven
                    or ready_stream_id != current_stream_id
                ):
                    raise RuntimeError(
                        "mixed-page CUDA graph replay cannot consume same-stream "
                        "ordered RRP metadata without an equal proven raw CUDA stream"
                    )
                continue
            wait_key = (id(event), generation)
            if wait_key in seen:
                continue
            current_stream_id, identity_proven = _resolve_wait_stream_identity()
            if (
                identity_proven
                and ready_stream_id >= 0
                and ready_stream_id == current_stream_id
            ):
                seen.add(wait_key)
                continue
            if wait_stream is None:
                import torch

                wait_stream = torch.cuda.current_stream()
            wait_event = getattr(wait_stream, "wait_event", None)
            if not callable(wait_event):
                raise RuntimeError(
                    "mixed-page CUDA graph replay requires a stream with wait_event"
                )
            wait_event(event)
            seen.add(wait_key)
            wait_count += 1
    previous_total = int(
        getattr(
            forward_context,
            "mixed_page_resolver_replay_ready_event_wait_total",
            0,
        )
    )
    setattr(
        forward_context,
        "mixed_page_resolver_replay_ready_event_wait_count",
        wait_count,
    )
    setattr(
        forward_context,
        "mixed_page_resolver_replay_ready_event_wait_total",
        previous_total + wait_count,
    )
    return wait_count


def _metadata_row_mode_distribution(metadata: object) -> dict[str, int]:
    source_arena = getattr(metadata, "mixed_page_resolver_source_arena", None)
    row_modes = getattr(source_arena, "row_mode_by_row_head", None)
    if row_modes is not None:
        return _build_row_mode_distribution(row_modes)

    metadata_distribution = getattr(metadata, "mixed_page_row_mode_distribution", None)
    if isinstance(metadata_distribution, dict):
        return {str(key): int(value) for key, value in metadata_distribution.items()}
    return {}


def _metadata_row_source_distribution(metadata: object) -> dict[str, int]:
    source_arena = getattr(metadata, "mixed_page_resolver_source_arena", None)
    source_distribution = getattr(source_arena, "row_source_distribution", None)
    if isinstance(source_distribution, dict):
        return {str(key): int(value) for key, value in source_distribution.items()}

    metadata_distribution = getattr(metadata, "mixed_page_row_source_distribution", None)
    if isinstance(metadata_distribution, dict):
        return {str(key): int(value) for key, value in metadata_distribution.items()}
    return {}


def _metadata_source_counter_fields(metadata: object) -> dict[str, object]:
    source_arena = getattr(metadata, "mixed_page_resolver_source_arena", None)
    source_distribution = getattr(source_arena, "row_source_distribution", None)
    if isinstance(source_distribution, dict):
        return build_source_counter_fields(
            batch_size=int(getattr(source_arena, "batch_size")),
            num_kv_heads=int(getattr(source_arena, "num_kv_heads")),
            row_source_distribution=source_distribution,
        )

    if hasattr(metadata, "source_counter_schema_version"):
        fields = getattr(metadata, "source_counter_missing_fields", ())
        if isinstance(fields, str):
            missing_fields = (fields,) if fields else ()
        elif isinstance(fields, (list, tuple)):
            missing_fields = tuple(str(field) for field in fields if str(field))
        else:
            missing_fields = ()
        return {
            "source_counter_schema_version": int(
                getattr(metadata, "source_counter_schema_version")
            ),
            "expected_rows": int(getattr(metadata, "expected_rows", -1)),
            "num_kv_heads": int(getattr(metadata, "num_kv_heads", -1)),
            "source_counter_missing_fields": missing_fields,
        }
    return {}


def _merge_source_counter_fields(
    total: dict[str, object],
    fields: dict[str, object],
) -> None:
    if not fields:
        return
    missing = set(total.get("source_counter_missing_fields", ()))
    missing.update(
        str(field)
        for field in fields.get("source_counter_missing_fields", ())
        if str(field)
    )

    schema_version = int(fields.get("source_counter_schema_version", -1))
    previous_schema = total.get("source_counter_schema_version")
    if previous_schema is None:
        total["source_counter_schema_version"] = schema_version
    elif int(previous_schema) != schema_version:
        missing.add("source_counter_schema_version_mismatch")

    num_kv_heads = int(fields.get("num_kv_heads", -1))
    previous_heads = total.get("num_kv_heads")
    if previous_heads is None:
        total["num_kv_heads"] = num_kv_heads
    elif int(previous_heads) != num_kv_heads:
        missing.add("num_kv_heads_mismatch")

    expected_rows = int(fields.get("expected_rows", 0) or 0)
    total["expected_rows"] = int(total.get("expected_rows", 0) or 0) + expected_rows
    total["source_counter_missing_fields"] = tuple(sorted(missing))


def _complete_source_counter_fields(total: dict[str, object]) -> dict[str, object]:
    missing = set(total.get("source_counter_missing_fields", ()))
    for key in ("source_counter_schema_version", "expected_rows", "num_kv_heads"):
        if key not in total:
            missing.add(key)
    return {
        "source_counter_schema_version": int(
            total.get("source_counter_schema_version", -1)
        ),
        "expected_rows": int(total.get("expected_rows", -1)),
        "num_kv_heads": int(total.get("num_kv_heads", -1)),
        "source_counter_missing_fields": tuple(sorted(missing)),
    }


def _tensor_signature(tensor: object | None) -> tuple[object, ...] | None:
    if tensor is None:
        return None
    data_ptr = getattr(tensor, "data_ptr", None)
    if data_ptr is None:
        return ("object", id(tensor))
    shape = tuple(int(dim) for dim in getattr(tensor, "shape", ()))
    stride_fn = getattr(tensor, "stride", None)
    stride = tuple(int(dim) for dim in stride_fn()) if stride_fn is not None else ()
    return (
        int(data_ptr()),
        getattr(tensor, "dtype", None),
        getattr(tensor, "device", None),
        shape,
        stride,
    )


def _carrier_signature(carriers: object | None) -> tuple[object, ...] | None:
    if carriers is None:
        return None
    return tuple(
        _tensor_signature(getattr(carriers, attr_name, None))
        for attr_name in _CARRIER_SIGNATURE_ATTRS
    )


def _carrier_update_signature_pairs(metadata: object) -> tuple[tuple[object, object], ...]:
    updates = getattr(metadata, "mixed_page_resolver_replay_carrier_updates", None)
    if updates is not None:
        return tuple(
            (_carrier_signature(dst), _carrier_signature(src))
            for dst, src in updates
        )

    dst = getattr(metadata, "mixed_page_resolver_replay_carriers", None)
    src = getattr(metadata, "mixed_page_resolver_replay_carrier_sources", None)
    if dst is None and src is None:
        return ()
    return ((_carrier_signature(dst), _carrier_signature(src)),)


def _metadata_update_signature(metadata: object) -> tuple[object, ...]:
    direct_tensor_pairs = tuple(
        (
            _tensor_signature(getattr(metadata, dst_attr, None)),
            _tensor_signature(getattr(metadata, src_attr, None)),
        )
        for dst_attr, src_attr in _RESOLVED_ROW_PTR_REPLAY_TENSOR_ATTR_PAIRS
    )
    return (
        _carrier_update_signature_pairs(metadata),
        direct_tensor_pairs,
    )


def _mark_replay_carriers_updated(
    metadata: object,
    *,
    stream_key: str,
    update_stats: dict[str, int],
) -> None:
    setattr(metadata, "mixed_page_resolver_replay_carriers_updated", True)
    setattr(metadata, "mixed_page_resolver_replay_carrier_update_stream_key", str(stream_key))
    setattr(metadata, "mixed_page_resolver_replay_carrier_update_stats", dict(update_stats))


def refresh_mixed_page_carriers_for_forward_context(
    forward_context: object,
    *,
    step_id: int,
    graph_key: str,
    stream_key: str = "vllm-full-cudagraph-replay",
) -> MixedPageForwardContextReplayStats:
    attn_metadata = getattr(forward_context, "attn_metadata", None)
    metadata_items = iter_attention_metadata(attn_metadata)
    if not metadata_items:
        raise RuntimeError("mixed-page CUDA graph replay requires attention metadata")

    update_metadata = tuple(
        metadata
        for metadata in metadata_items
        if metadata_has_mixed_page_replay_update(metadata)
    )
    if not update_metadata:
        raise RuntimeError(
            "mixed-page CUDA graph replay found no mixed-page replay carrier update pairs"
        )
    wait_mixed_page_resolver_ready_events_for_forward_context(forward_context)

    rows = 0
    update_bytes = 0
    kernel_count = 0
    updated_metadata_count = 0
    updated_signatures: set[tuple[object, ...]] = set()
    row_mode_distribution: dict[str, int] = {}
    row_source_distribution: dict[str, int] = {}
    source_counter_fields: dict[str, object] = {}

    for metadata in update_metadata:
        setattr(metadata, "mixed_page_resolver_graph_replay_expected", True)
        update_signature = _metadata_update_signature(metadata)
        if update_signature in updated_signatures:
            update_stats = {
                "carrier_update_rows": 0,
                "carrier_update_bytes": 0,
                "carrier_update_kernel_count": 0,
                "carrier_update_deduplicated": 1,
            }
            _mark_replay_carriers_updated(
                metadata,
                stream_key=stream_key,
                update_stats=update_stats,
            )
        else:
            update_stats = refresh_mixed_page_resolver_carriers_for_replay(
                attn_metadata=metadata,
                stream_key=stream_key,
            )
        metadata_kernel_count = int(update_stats.get("carrier_update_kernel_count", 0))
        metadata_direct_bound = int(update_stats.get("carrier_update_direct_bound", 0))
        if (
            metadata_kernel_count <= 0
            and metadata_direct_bound <= 0
            and update_signature not in updated_signatures
        ):
            raise RuntimeError(
                "mixed-page CUDA graph replay carrier refresh produced no direct-bound update"
            )
        if metadata_kernel_count > 0 or metadata_direct_bound > 0:
            updated_signatures.add(update_signature)

        rows += int(update_stats.get("carrier_update_rows", 0))
        update_bytes += int(update_stats.get("carrier_update_bytes", 0))
        kernel_count += metadata_kernel_count
        updated_metadata_count += 1
        metadata_distribution = _metadata_row_mode_distribution(metadata)
        if metadata_distribution:
            setattr(metadata, "mixed_page_row_mode_distribution", dict(metadata_distribution))
            for key, value in metadata_distribution.items():
                row_mode_distribution[str(key)] = row_mode_distribution.get(str(key), 0) + int(value)
        metadata_source_distribution = _metadata_row_source_distribution(metadata)
        if metadata_source_distribution:
            setattr(
                metadata,
                "mixed_page_row_source_distribution",
                dict(metadata_source_distribution),
            )
            for key, value in metadata_source_distribution.items():
                row_source_distribution[str(key)] = (
                    row_source_distribution.get(str(key), 0) + int(value)
                )
        metadata_source_counter_fields = _metadata_source_counter_fields(metadata)
        if metadata_source_counter_fields:
            for key, value in metadata_source_counter_fields.items():
                setattr(metadata, key, value)
            _merge_source_counter_fields(
                source_counter_fields,
                metadata_source_counter_fields,
            )
        setattr(metadata, "mixed_page_resolver_replay_carrier_update_step_id", step_id)
        setattr(
            metadata,
            "mixed_page_resolver_replay_carrier_update_graph_key",
            graph_key,
        )

    stats = MixedPageForwardContextReplayStats(
        metadata_count=len(metadata_items),
        updated_metadata_count=updated_metadata_count,
        carrier_update_rows=rows,
        carrier_update_bytes=update_bytes,
        carrier_update_kernel_count=kernel_count,
        row_mode_distribution=row_mode_distribution,
        row_source_distribution=row_source_distribution,
        **_complete_source_counter_fields(source_counter_fields),
        step_id=step_id,
        graph_key=graph_key,
    )
    setattr(forward_context, "mixed_page_resolver_replay_last_step_id", step_id)
    setattr(forward_context, "mixed_page_resolver_replay_last_graph_key", graph_key)
    setattr(forward_context, "mixed_page_resolver_replay_last_stats", stats)
    return stats


def mark_mixed_page_direct_bound_replay_for_forward_context(
    forward_context: object,
    *,
    step_id: int,
    graph_key: str,
    stream_key: str = "vllm-full-cudagraph-replay",
) -> MixedPageForwardContextReplayStats:
    """Mark direct-bound replay metadata after a graph-level proof cache hit."""
    attn_metadata = getattr(forward_context, "attn_metadata", None)
    metadata_items = iter_attention_metadata(attn_metadata)
    if not metadata_items:
        raise RuntimeError("mixed-page CUDA graph replay requires attention metadata")

    update_metadata = tuple(
        metadata
        for metadata in metadata_items
        if metadata_has_mixed_page_replay_update(metadata)
    )
    if not update_metadata:
        raise RuntimeError(
            "mixed-page CUDA graph replay found no mixed-page replay carrier update pairs"
        )
    wait_mixed_page_resolver_ready_events_for_forward_context(forward_context)

    row_mode_distribution: dict[str, int] = {}
    row_source_distribution: dict[str, int] = {}
    source_counter_fields: dict[str, object] = {}
    update_stats = {
        "carrier_update_rows": 0,
        "carrier_update_bytes": 0,
        "carrier_update_kernel_count": 0,
        "carrier_update_direct_bound": 1,
        "carrier_update_graph_proof_cached": 1,
    }
    for metadata in update_metadata:
        setattr(metadata, "mixed_page_resolver_graph_replay_expected", True)
        _mark_replay_carriers_updated(
            metadata,
            stream_key=stream_key,
            update_stats=update_stats,
        )
        metadata_distribution = _metadata_row_mode_distribution(metadata)
        if metadata_distribution:
            setattr(metadata, "mixed_page_row_mode_distribution", dict(metadata_distribution))
            for key, value in metadata_distribution.items():
                row_mode_distribution[str(key)] = row_mode_distribution.get(str(key), 0) + int(value)
        metadata_source_distribution = _metadata_row_source_distribution(metadata)
        if metadata_source_distribution:
            setattr(
                metadata,
                "mixed_page_row_source_distribution",
                dict(metadata_source_distribution),
            )
            for key, value in metadata_source_distribution.items():
                row_source_distribution[str(key)] = (
                    row_source_distribution.get(str(key), 0) + int(value)
                )
        metadata_source_counter_fields = _metadata_source_counter_fields(metadata)
        if metadata_source_counter_fields:
            for key, value in metadata_source_counter_fields.items():
                setattr(metadata, key, value)
            _merge_source_counter_fields(
                source_counter_fields,
                metadata_source_counter_fields,
            )
        setattr(metadata, "mixed_page_resolver_replay_carrier_update_step_id", step_id)
        setattr(
            metadata,
            "mixed_page_resolver_replay_carrier_update_graph_key",
            graph_key,
        )

    stats = MixedPageForwardContextReplayStats(
        metadata_count=len(metadata_items),
        updated_metadata_count=len(update_metadata),
        carrier_update_rows=0,
        carrier_update_bytes=0,
        carrier_update_kernel_count=0,
        row_mode_distribution=row_mode_distribution,
        row_source_distribution=row_source_distribution,
        **_complete_source_counter_fields(source_counter_fields),
        step_id=step_id,
        graph_key=graph_key,
    )
    setattr(forward_context, "mixed_page_resolver_replay_last_step_id", step_id)
    setattr(forward_context, "mixed_page_resolver_replay_last_graph_key", graph_key)
    setattr(forward_context, "mixed_page_resolver_replay_last_stats", stats)
    return stats
