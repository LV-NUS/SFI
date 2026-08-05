"""Capture-forward side-output scratch prep.

Diagnostics note (2026-04-23 rev 2):
    Validation checks in this module use ``torch._assert_async`` rather than
    host-readback ``.item()`` to meet the "0 hot-path sync" rule. Trade-off:
    on GPU fault the error surfaces at the **next** CUDA sync point (not at
    the offending tensor op), and the CUDA context is trashed (process must
    restart). SRE runbook when a capture-path assert fires in production:

        1. Rerun the failing job with ``CUDA_LAUNCH_BLOCKING=1`` to force
           synchronous dispatch and pinpoint the source op.
        2. Inspect ``capture_row_index_i32`` for producer rows mapping to
           -1, or values >= layout.capture_rows.
        3. Compare ``prefill_layout.kv_max`` / ``refresh_layout.kv_max`` to
           actual ``seqused_k`` in the failing batch.
        4. If layout upper bounds look wrong, check that
           ``producer_rows_cpu`` / ``max_capture_k_cpu`` /
           ``max_capture_last_n_cpu`` arguments from the caller
           (``_run_capture_only_mixed_forward``) are derived from
           ``step_authority`` CPU fields that match the batch.
"""
from __future__ import annotations

from typing import Optional, Sequence

import json
import math
import os
import threading

import torch

from patches.cpu_gpu_staging import cached_sequence_to_device
from patches.fa3_native.capture_ownership import (
    CAPTURE_OWNERSHIP_POLICY_SCHEMA,
    CaptureOwnershipPlan,
)
from patches.fa3_native.row_plan import MixedPageRowPlan
from patches.sparse_types import CaptureForwardSideOutputs, StepCaptureLayout

# Default-off allocation provenance. Keep the path latched at import like the other
# capture envs: production hot paths pay only one false branch, with no tensor readback,
# synchronization, or file work when the probe is disabled.
_CAPTURE_SCRATCH_PROBE_LOG = str(
    os.environ.get("VLLM_SPARSE_CAPTURE_SCRATCH_PROBE_LOG", "") or ""
).strip()
_CAPTURE_SCRATCH_PROBE_SEEN: set[tuple[str, str, str]] = set()
_CAPTURE_SCRATCH_PROBE_LOCK = threading.Lock()


def _capture_scratch_scope_key(scratch_key: tuple[object, ...]) -> tuple[object, ...]:
    """Return the runtime owner scope for one exact scratch allocation key.

    The sealed capture runtime has exactly one scratch owner per device.  Ring
    and cohort are alternative policies, not simultaneous cache populations;
    geometry or policy changes replace the prior slab at the cold boundary.
    """
    if len(scratch_key) != 5:
        raise RuntimeError("E_SFI_CAPTURE_SCRATCH_INVALID_KEY")
    extra_key = scratch_key[4]
    if (
        isinstance(extra_key, tuple)
        and extra_key
        and isinstance(extra_key[0], str)
        and extra_key[0]
        in {"capture_postprocess_ring", "capture_postprocess_cohort"}
    ):
        extra_key = "capture_postprocess_owner"
    return (
        scratch_key[0],
        scratch_key[1],
        scratch_key[2],
        extra_key,
    )


def _store_capture_scratch_cache_entry(
    *,
    cache_owner: object,
    cache_map: dict[tuple[object, ...], torch.Tensor],
    scratch_key: tuple[object, ...],
    scratch_storage: torch.Tensor,
) -> None:
    """Cold-path scope replacement for prebuilt/live scratch allocations."""
    scope_keys = getattr(cache_owner, "_fa3_capture_scratch_scope_keys", None)
    if not isinstance(scope_keys, dict):
        scope_keys = {}
        setattr(cache_owner, "_fa3_capture_scratch_scope_keys", scope_keys)
    scope = _capture_scratch_scope_key(scratch_key)
    previous_key = scope_keys.get(scope)
    if previous_key is not None and previous_key != scratch_key:
        if scratch_storage.is_cuda and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "E_SFI_CAPTURE_SCRATCH_SCOPE_CHANGE_DURING_CAPTURE: "
                "prebuild the exact runtime shape"
            )
        previous = cache_map.pop(previous_key, None)
        if previous is None:
            raise RuntimeError("E_SFI_CAPTURE_SCRATCH_SCOPE_INDEX_CORRUPT")
        if previous.is_cuda:
            guard = getattr(
                cache_owner, "_uaf_guard_record_streams_before_discard", None
            )
            if callable(guard):
                guard(previous)
            else:
                streams = [torch.cuda.current_stream(device=previous.device)]
                refresh_stream = getattr(cache_owner, "refresh_stream", None)
                if refresh_stream is not None and all(
                    refresh_stream != stream for stream in streams
                ):
                    streams.append(refresh_stream)
                for stream in streams:
                    previous.record_stream(stream)
    elif previous_key == scratch_key and scratch_key not in cache_map:
        raise RuntimeError("E_SFI_CAPTURE_SCRATCH_SCOPE_INDEX_CORRUPT")
    cache_map[scratch_key] = scratch_storage
    scope_keys[scope] = scratch_key


def _cached_absent_phase_mapping(
    *,
    batch_size: int,
    device: torch.device,
    cache_owner: Optional[object],
) -> torch.Tensor:
    """Return an immutable all--1 phase mapping retained by the controller.

    The tensor is read-only capture metadata. Retaining it on the controller keeps
    its address stable for every layer and for the lifetime of any CUDA graph that
    references it. The cache is deliberately keyed only by allocation identity;
    phase/step state cannot affect an all--1 value.
    """
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    key = (
        str(device.type),
        -1 if device.index is None else int(device.index),
        int(batch_size),
    )
    cache = None
    if cache_owner is not None:
        candidate = getattr(cache_owner, "_fa3_absent_phase_mapping_i32_by_shape", None)
        if isinstance(candidate, dict):
            cache = candidate
        else:
            cache = {}
            setattr(cache_owner, "_fa3_absent_phase_mapping_i32_by_shape", cache)
        cached = cache.get(key)
        if (
            isinstance(cached, torch.Tensor)
            and cached.device == device
            and cached.dtype == torch.int32
            and cached.dim() == 1
            and int(cached.numel()) == int(batch_size)
        ):
            return cached

    mapping = torch.full(
        (int(batch_size),),
        -1,
        device=device,
        dtype=torch.int32,
    )
    if cache is not None:
        cache[key] = mapping
    return mapping


def _validate_cpu_capture_rows(
    *,
    batch_size: int,
    slot_in_chunk: int,
    producer_rows: tuple[int, ...],
    prefill_rows: tuple[int, ...],
    decode_rows: tuple[int, ...],
    row_capture_last_n: tuple[int, ...],
    seqused_k: tuple[int, ...],
    max_capture_last_n: int,
    prefill_layout: Optional[StepCaptureLayout],
    refresh_layout: Optional[StepCaptureLayout],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Validate complete CPU row truth and derive trusted phase mappings.

    This helper intentionally performs no tensor reads or device operations. A
    successful return is the sole proof that permits the caller to skip redundant
    per-layer device-side mapping assertions.
    """

    def _validate_rows(name: str, rows: tuple[int, ...]) -> set[int]:
        row_set = set(rows)
        if len(row_set) != len(rows):
            raise ValueError(f"CPU row truth {name} rows must be unique")
        if any(int(row) < 0 or int(row) >= int(batch_size) for row in rows):
            raise ValueError(f"CPU row truth {name} rows must be within the batch")
        return row_set

    producer_set = _validate_rows("producer", producer_rows)
    prefill_set = _validate_rows("prefill", prefill_rows)
    decode_set = _validate_rows("decode", decode_rows)
    if prefill_set.intersection(decode_set):
        raise ValueError("CPU row truth prefill/decode rows must be disjoint")
    if prefill_set.union(decode_set) != producer_set:
        raise ValueError(
            "CPU row truth prefill/decode row union must equal producer rows"
        )
    if len(row_capture_last_n) != int(batch_size):
        raise ValueError("CPU row truth last_n length must equal batch size")
    if len(seqused_k) != int(batch_size):
        raise ValueError("CPU row truth seqused_k length must equal batch size")
    if any(int(value) < 0 for value in row_capture_last_n):
        raise ValueError("CPU row truth last_n values must be non-negative")
    if any(int(value) < 0 for value in seqused_k):
        raise ValueError("CPU row truth seqused_k values must be non-negative")
    for row in producer_rows:
        last_n = int(row_capture_last_n[int(row)])
        if last_n < 1:
            raise ValueError("CPU row truth producer last_n must be at least one")
        if last_n > int(max_capture_last_n):
            raise ValueError("CPU row truth producer last_n exceeds scratch capacity")
        if last_n > int(seqused_k[int(row)]):
            raise ValueError("CPU row truth producer last_n exceeds seqused_k")

    def _validate_layout(
        *,
        name: str,
        phase_rows: set[int],
        layout: Optional[StepCaptureLayout],
    ) -> tuple[int, ...]:
        if layout is None:
            if phase_rows:
                raise ValueError(f"{name} producer rows require {name} output tensors")
            return tuple(-1 for _ in range(int(batch_size)))

        row_list_src = getattr(layout, "row_list_cpu", None)
        if row_list_src is None:
            raise ValueError(f"CPU row truth requires {name} layout row_list_cpu")
        row_list = tuple(int(row) for row in row_list_src)
        layout_row_set = _validate_rows(f"{name} layout", row_list)
        slot_list = tuple(int(slot) for slot in layout.slot_list)
        if len(row_list) != len(slot_list):
            raise ValueError(f"{name} layout rows must align with slot_list")
        if len(set(slot_list)) != len(slot_list):
            raise ValueError(f"{name} layout slot_list must be unique")
        if not phase_rows.issubset(layout_row_set):
            raise ValueError(f"{name} layout rows do not cover phase producers")
        if any(
            int(layout.slot_to_capture_row.get(slot, -1)) != idx
            for idx, slot in enumerate(slot_list)
        ):
            raise ValueError(f"{name} layout slot mapping is not canonical")
        if not isinstance(layout.capture_scores, torch.Tensor) or layout.capture_scores.dim() != 5:
            raise ValueError(
                f"{name} layout capture_scores must be [chunk, capture_rows, heads, 1, kv]"
            )
        if not isinstance(layout.log_f_denoms, torch.Tensor) or layout.log_f_denoms.dim() != 3:
            raise ValueError(
                f"{name} layout log_f_denoms must be [chunk, capture_rows, heads]"
            )
        if int(slot_in_chunk) < 0 or int(slot_in_chunk) >= int(layout.capture_scores.shape[0]):
            raise ValueError(f"{name} layout slot_in_chunk is out of range")
        if int(slot_in_chunk) >= int(layout.log_f_denoms.shape[0]):
            raise ValueError(f"{name} layout log_f_denoms slot_in_chunk is out of range")
        if int(layout.capture_scores.shape[1]) < len(row_list):
            raise ValueError(f"{name} layout capture score rows lack CPU row capacity")
        if int(layout.log_f_denoms.shape[1]) < len(row_list):
            raise ValueError(f"{name} layout denominator rows lack CPU row capacity")
        if int(layout.capture_scores.shape[2]) != int(layout.num_heads):
            raise ValueError(f"{name} layout capture score heads mismatch")
        if int(layout.log_f_denoms.shape[2]) != int(layout.num_heads):
            raise ValueError(f"{name} layout denominator heads mismatch")
        if int(layout.capture_scores.shape[-1]) != int(layout.kv_max):
            raise ValueError(f"{name} layout kv_max must match capture_scores width")

        mapping = [-1] * int(batch_size)
        for capture_row, batch_row in enumerate(row_list):
            mapping[int(batch_row)] = int(capture_row)
        return tuple(mapping)

    return (
        _validate_layout(
            name="prefill",
            phase_rows=prefill_set,
            layout=prefill_layout,
        ),
        _validate_layout(
            name="refresh",
            phase_rows=decode_set,
            layout=refresh_layout,
        ),
    )


def log_capture_scratch_probe(
    *,
    source: str,
    cache_key: object,
    cache_hit: bool,
    scratch_storage_shape: Sequence[int],
    scratch_dtype: torch.dtype,
    element_size_bytes: int,
    actual_rows: Optional[int],
    bucket_rows: int,
    heads: int,
    last_n: int,
    capture_k: int,
    reduce_group: Optional[int] = None,
    in_flight: Optional[int] = None,
    key_kind: str = "",
) -> None:
    """Append one allocation-shape record without affecting capture semantics.

    This is deliberately best-effort: malformed paths, serialization failures, and
    write errors are swallowed. The helper consumes CPU metadata only; callers must
    not pass values obtained through device readback.
    """
    if not _CAPTURE_SCRATCH_PROBE_LOG:
        return
    try:
        if reduce_group is None or in_flight is None:
            from patches.sparse_constants import (
                _CAPTURE_IN_FLIGHT,
                _CAPTURE_REDUCE_GROUP,
            )

            if reduce_group is None:
                reduce_group = int(_CAPTURE_REDUCE_GROUP)
            if in_flight is None:
                in_flight = int(_CAPTURE_IN_FLIGHT)

        shape = tuple(int(v) for v in scratch_storage_shape)
        key_repr = repr(cache_key)
        seen_key = (_CAPTURE_SCRATCH_PROBE_LOG, str(source), key_repr)
        with _CAPTURE_SCRATCH_PROBE_LOCK:
            if seen_key in _CAPTURE_SCRATCH_PROBE_SEEN:
                return
            # Mark before serialization/I/O so a bad diagnostic path cannot
            # add repeated work to every capture layer. The record describes
            # allocation provenance, not per-step utilization.
            _CAPTURE_SCRATCH_PROBE_SEEN.add(seen_key)
        record = {
            "schema": "sfi.capture_scratch_probe.v1",
            "source": str(source),
            "shape": list(shape),
            "dtype": str(scratch_dtype).removeprefix("torch."),
            "bytes": int(element_size_bytes) * math.prod(shape),
            "actual_rows": None if actual_rows is None else int(actual_rows),
            "bucket_rows": int(bucket_rows),
            "heads": int(heads),
            "last_n": int(last_n),
            "K": int(capture_k),
            "reduce_group": int(reduce_group),
            "in_flight": int(in_flight),
            "cache_hit": bool(cache_hit),
            "key_kind": str(key_kind),
        }
        payload = (
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        fd = os.open(
            _CAPTURE_SCRATCH_PROBE_LOG,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o644,
        )
        try:
            os.write(fd, payload)
        finally:
            os.close(fd)
    except Exception:
        return


def _prepare_phase_outputs(
    *,
    layout: Optional[StepCaptureLayout],
    batch_size: int,
    slot_in_chunk: int,
    device: torch.device | str,
    cache_owner: Optional[object] = None,
    validated_mapping: Optional[torch.Tensor] = None,
    cpu_row_truth_validated: bool = False,
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if layout is None:
        mapping = _cached_absent_phase_mapping(
            batch_size=batch_size,
            device=device,
            cache_owner=cache_owner,
        )
        return mapping, None, None
    mapping = (
        validated_mapping
        if bool(cpu_row_truth_validated)
        else getattr(layout, "active_capture_row_by_batch_row_i32", None)
    )
    if mapping is None:
        raise ValueError("capture layout must provide active_capture_row_by_batch_row_i32")
    if mapping.device != device or mapping.dtype != torch.int32 or mapping.dim() != 1:
        raise ValueError("capture layout active mapping must be activity-sized int32 on the target device")
    if int(mapping.numel()) != batch_size:
        raise ValueError("capture layout active mapping must align with batch size")
    capture_rows = int(len(layout.slot_list))
    if layout.capture_scores.dim() != 5:
        raise ValueError("capture layout capture_scores must be [chunk, capture_rows, heads, 1, kv]")
    if layout.log_f_denoms.dim() != 3:
        raise ValueError("capture layout log_f_denoms must be [chunk, capture_rows, heads]")
    if slot_in_chunk < 0 or slot_in_chunk >= int(layout.capture_scores.shape[0]):
        raise ValueError("capture layout slot_in_chunk is out of range")
    if slot_in_chunk >= int(layout.log_f_denoms.shape[0]):
        raise ValueError("capture layout log_f_denoms slot_in_chunk is out of range")
    if int(layout.capture_scores.shape[1]) < capture_rows:
        raise ValueError("capture layout capture_scores rows must cover slot_list")
    if int(layout.log_f_denoms.shape[1]) < capture_rows:
        raise ValueError("capture layout log_f_denoms rows must cover slot_list")
    if int(layout.capture_scores.shape[-1]) != int(layout.kv_max):
        raise ValueError("capture layout kv_max must match capture_scores width")
    if capture_rows > 0 and not bool(cpu_row_truth_validated):
        safe_mapping = torch.where(mapping >= 0, mapping, mapping.new_zeros(()))
        torch._assert_async(
            torch.logical_not(safe_mapping.max() >= capture_rows),
            "capture row mapping exceeds layout rows",
        )
    out_capture_scores = layout.capture_scores[slot_in_chunk, :capture_rows]
    out_log_f_denoms = layout.log_f_denoms[slot_in_chunk, :capture_rows]
    return mapping, out_capture_scores, out_log_f_denoms


def prepare_capture_forward_side_outputs(
    *,
    prefill_layout: Optional[StepCaptureLayout],
    refresh_layout: Optional[StepCaptureLayout],
    row_plan: MixedPageRowPlan,
    slot_in_chunk: int,
    ring_scratch_slot: int = -1,
    seqused_k: torch.Tensor,
    device: torch.device | str,
    # Rev 2: CPU-side producer set + upper bounds from StepAuthority / layout
    # metadata, eliminating GPU nonzero/max.item() sync on hot path.
    producer_rows_cpu: tuple[int, ...],
    max_capture_k_cpu: int,
    max_capture_last_n_cpu: int,
    prefill_producer_rows_cpu: Optional[Sequence[int]] = None,
    decode_producer_rows_cpu: Optional[Sequence[int]] = None,
    row_capture_last_n_cpu: Optional[Sequence[int]] = None,
    seqused_k_cpu: Optional[Sequence[int]] = None,
    scratch_cache_owner: Optional[object] = None,
    scratch_cache_extra_key: Optional[object] = None,
) -> CaptureForwardSideOutputs:
    batch_size = int(seqused_k.numel())
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    producer_rows_cpu_tuple = tuple(int(row) for row in producer_rows_cpu)
    prefill_rows_cpu_tuple = (
        tuple(int(row) for row in prefill_producer_rows_cpu)
        if prefill_producer_rows_cpu is not None
        else tuple()
    )
    decode_rows_cpu_tuple = (
        tuple(int(row) for row in decode_producer_rows_cpu)
        if decode_producer_rows_cpu is not None
        else tuple()
    )
    row_capture_last_n_cpu_raw = (
        tuple(int(v) for v in row_capture_last_n_cpu)
        if row_capture_last_n_cpu is not None
        else tuple()
    )
    seqused_k_cpu_raw = (
        tuple(int(v) for v in seqused_k_cpu)
        if seqused_k_cpu is not None
        else tuple()
    )
    validated_phase_mappings = None
    if (
        row_capture_last_n_cpu is not None
        and seqused_k_cpu is not None
        and (
            prefill_producer_rows_cpu is not None
            or decode_producer_rows_cpu is not None
        )
    ):
        # This validation must precede every H2D/device operation. Only its
        # successful return authorizes the no-device-assert fast path below.
        validated_phase_mappings = _validate_cpu_capture_rows(
            batch_size=batch_size,
            slot_in_chunk=int(slot_in_chunk),
            producer_rows=producer_rows_cpu_tuple,
            prefill_rows=prefill_rows_cpu_tuple,
            decode_rows=decode_rows_cpu_tuple,
            row_capture_last_n=row_capture_last_n_cpu_raw,
            seqused_k=seqused_k_cpu_raw,
            max_capture_last_n=int(max_capture_last_n_cpu),
            prefill_layout=prefill_layout,
            refresh_layout=refresh_layout,
        )
    cpu_row_truth_validated = validated_phase_mappings is not None

    seqused_k = seqused_k.to(device=device, dtype=torch.int32).reshape(batch_size)
    prefill_row_set = set(prefill_rows_cpu_tuple)
    producer_row_set = set(producer_rows_cpu_tuple)
    row_capture_last_n_cpu_tuple = tuple(
        max(0, int(value)) for value in row_capture_last_n_cpu_raw
    )
    seqused_k_cpu_tuple = tuple(max(0, int(value)) for value in seqused_k_cpu_raw)
    row_is_prefill_producer_cpu_tuple = (
        tuple(row in prefill_row_set for row in range(batch_size))
        if prefill_rows_cpu_tuple or decode_rows_cpu_tuple
        else tuple()
    )

    def _layout_row_to_capture(layout: Optional[StepCaptureLayout]) -> dict[int, int]:
        row_list = getattr(layout, "row_list_cpu", None) if layout is not None else None
        if row_list is None:
            return {}
        return {int(row): idx for idx, row in enumerate(row_list)}

    if validated_phase_mappings is not None:
        prefill_mapping_cpu, refresh_mapping_cpu = validated_phase_mappings
    else:
        prefill_row_to_capture = _layout_row_to_capture(prefill_layout)
        refresh_row_to_capture = _layout_row_to_capture(refresh_layout)
        prefill_mapping_values = [-1] * batch_size
        refresh_mapping_values = [-1] * batch_size
        for row, capture_row in prefill_row_to_capture.items():
            if 0 <= int(row) < batch_size:
                prefill_mapping_values[int(row)] = int(capture_row)
        for row, capture_row in refresh_row_to_capture.items():
            if 0 <= int(row) < batch_size:
                refresh_mapping_values[int(row)] = int(capture_row)
        prefill_mapping_cpu = tuple(prefill_mapping_values)
        refresh_mapping_cpu = tuple(refresh_mapping_values)

    active_capture_row_cpu = [-1] * batch_size
    for batch_row in prefill_rows_cpu_tuple:
        if 0 <= int(batch_row) < batch_size:
            active_capture_row_cpu[int(batch_row)] = int(
                prefill_mapping_cpu[int(batch_row)]
            )
    for batch_row in decode_rows_cpu_tuple:
        if 0 <= int(batch_row) < batch_size:
            active_capture_row_cpu[int(batch_row)] = int(
                refresh_mapping_cpu[int(batch_row)]
            )

    side_tensor_key = (
        str(device.type),
        -1 if device.index is None else int(device.index),
        batch_size,
        producer_rows_cpu_tuple,
        prefill_rows_cpu_tuple,
        decode_rows_cpu_tuple,
        row_capture_last_n_cpu_tuple[:batch_size],
        tuple(active_capture_row_cpu),
        prefill_mapping_cpu,
        refresh_mapping_cpu,
    )
    cached_side = None
    if cpu_row_truth_validated and scratch_cache_owner is not None:
        cached_key = getattr(scratch_cache_owner, "_fa3_capture_side_tensor_cache_key", None)
        cached_value = getattr(scratch_cache_owner, "_fa3_capture_side_tensor_cache", None)
        if (
            cached_key == side_tensor_key
            and isinstance(cached_value, tuple)
            and len(cached_value) == 10
        ):
            cached_side = cached_value
    if cached_side is not None:
        (
            producer_rows,
            producer_rows_i32,
            capture_row_index_i32,
            row_capture_last_n_i32,
            row_is_prefill_producer,
            active_capture_row_by_batch_row_i32,
            prefill_producer_rows,
            decode_producer_rows,
            validated_prefill_mapping_i32,
            validated_refresh_mapping_i32,
        ) = cached_side
    elif cpu_row_truth_validated:
        capture_row_index_cpu = [-1] * batch_size
        for idx, row in enumerate(producer_rows_cpu_tuple):
            capture_row_index_cpu[int(row)] = int(idx)
        row_capture_last_n_values = [
            int(row_capture_last_n_cpu_tuple[row]) if row in producer_row_set else 0
            for row in range(batch_size)
        ]
        producer_rows = cached_sequence_to_device(
            producer_rows_cpu_tuple,
            device=device,
            dtype=torch.int64,
            cache_name="capture_side_producer_rows_i64",
            cache_owner=scratch_cache_owner,
        )
        producer_rows_i32 = cached_sequence_to_device(
            producer_rows_cpu_tuple,
            device=device,
            dtype=torch.int32,
            cache_name="capture_side_producer_rows_i32",
            cache_owner=scratch_cache_owner,
        )
        capture_row_index_i32 = cached_sequence_to_device(
            capture_row_index_cpu,
            device=device,
            dtype=torch.int32,
            cache_name="capture_side_capture_row_index_i32",
            cache_owner=scratch_cache_owner,
        )
        row_capture_last_n_i32 = cached_sequence_to_device(
            row_capture_last_n_values,
            device=device,
            dtype=torch.int32,
            cache_name="capture_side_last_n_i32",
            cache_owner=scratch_cache_owner,
        )
        row_is_prefill_producer = cached_sequence_to_device(
            row_is_prefill_producer_cpu_tuple,
            device=device,
            dtype=torch.bool,
            cache_name="capture_side_is_prefill_bool",
            cache_owner=scratch_cache_owner,
        )
        active_capture_row_by_batch_row_i32 = cached_sequence_to_device(
            active_capture_row_cpu,
            device=device,
            dtype=torch.int32,
            cache_name="capture_side_active_row_i32",
            cache_owner=scratch_cache_owner,
        )
        prefill_producer_rows = cached_sequence_to_device(
            prefill_rows_cpu_tuple,
            device=device,
            dtype=torch.int64,
            cache_name="capture_side_prefill_rows_i64",
            cache_owner=scratch_cache_owner,
        )
        decode_producer_rows = cached_sequence_to_device(
            decode_rows_cpu_tuple,
            device=device,
            dtype=torch.int64,
            cache_name="capture_side_decode_rows_i64",
            cache_owner=scratch_cache_owner,
        )
        validated_prefill_mapping_i32 = (
            cached_sequence_to_device(
                prefill_mapping_cpu,
                device=device,
                dtype=torch.int32,
                cache_name="capture_side_prefill_mapping_i32",
                cache_owner=scratch_cache_owner,
            )
            if prefill_layout is not None
            else None
        )
        validated_refresh_mapping_i32 = (
            cached_sequence_to_device(
                refresh_mapping_cpu,
                device=device,
                dtype=torch.int32,
                cache_name="capture_side_refresh_mapping_i32",
                cache_owner=scratch_cache_owner,
            )
            if refresh_layout is not None
            else None
        )
        if scratch_cache_owner is not None:
            setattr(scratch_cache_owner, "_fa3_capture_side_tensor_cache_key", side_tensor_key)
            setattr(
                scratch_cache_owner,
                "_fa3_capture_side_tensor_cache",
                (
                    producer_rows,
                    producer_rows_i32,
                    capture_row_index_i32,
                    row_capture_last_n_i32,
                    row_is_prefill_producer,
                    active_capture_row_by_batch_row_i32,
                    prefill_producer_rows,
                    decode_producer_rows,
                    validated_prefill_mapping_i32,
                    validated_refresh_mapping_i32,
                ),
            )
    else:
        validated_prefill_mapping_i32 = None
        validated_refresh_mapping_i32 = None
        row_capture_last_n_i32 = row_plan.row_capture_last_n_i32.to(
            device=device,
            dtype=torch.int32,
        ).reshape(batch_size)
        row_is_prefill_producer = row_plan.row_is_prefill_producer.to(
            device=device,
            dtype=torch.bool,
        ).reshape(batch_size)
        # Rev 2: producer_rows from CPU-side list (passed by caller from
        # owner_plan.prefill_capture_rows + decode_capture_rows). Avoids
        # torch.nonzero(...) shape-dependent D2H.
        producer_rows = torch.tensor(
            list(producer_rows_cpu),
            device=device,
            dtype=torch.int64,
        )
        producer_rows_i32 = producer_rows.to(dtype=torch.int32)
        capture_row_index_i32 = torch.full(
            (batch_size,),
            -1,
            device=device,
            dtype=torch.int32,
        )
        if producer_rows.numel() > 0:
            capture_row_index_i32.scatter_(
                0,
                producer_rows,
                torch.arange(int(producer_rows.numel()), device=device, dtype=torch.int32),
            )
        active_capture_row_by_batch_row_i32 = torch.full(
            (batch_size,),
            -1,
            device=device,
            dtype=torch.int32,
        )
        prefill_producer_rows = producer_rows[row_is_prefill_producer.index_select(0, producer_rows)]
        decode_producer_rows = producer_rows[
            (~row_is_prefill_producer).index_select(0, producer_rows)
        ]

    (
        prefill_capture_row_by_batch_row_i32,
        prefill_out_capture_scores,
        prefill_out_log_f_denoms,
    ) = _prepare_phase_outputs(
        layout=prefill_layout,
        batch_size=batch_size,
        slot_in_chunk=slot_in_chunk,
        device=device,
        cache_owner=scratch_cache_owner,
        validated_mapping=validated_prefill_mapping_i32,
        cpu_row_truth_validated=cpu_row_truth_validated,
    )
    (
        refresh_capture_row_by_batch_row_i32,
        refresh_out_capture_scores,
        refresh_out_log_f_denoms,
    ) = _prepare_phase_outputs(
        layout=refresh_layout,
        batch_size=batch_size,
        slot_in_chunk=slot_in_chunk,
        device=device,
        cache_owner=scratch_cache_owner,
        validated_mapping=validated_refresh_mapping_i32,
        cpu_row_truth_validated=cpu_row_truth_validated,
    )

    def _capture_head_stride(out_capture_scores: Optional[torch.Tensor]) -> int:
        if not isinstance(out_capture_scores, torch.Tensor):
            return 0
        if out_capture_scores.dim() != 4:
            raise ValueError("phase capture outputs must be [capture_rows, heads, 1, kv]")
        return int(out_capture_scores.stride(1))

    def _validate_phase_capacity_async(
        *,
        name: str,
        phase_rows: torch.Tensor,
        capture_row_by_batch_row_i32: torch.Tensor,
        out_capture_scores: Optional[torch.Tensor],
    ) -> None:
        if phase_rows.numel() == 0:
            return
        if out_capture_scores is None:
            raise ValueError(f"{name} producer rows require {name} output tensors")
        mapped_rows = capture_row_by_batch_row_i32.index_select(0, phase_rows)
        # Rev 2: device-side asserts (no host sync).
        torch._assert_async(
            torch.logical_not(torch.any(mapped_rows < 0)),
            f"{name} producer rows require valid {name} capture row mapping",
        )
        torch._assert_async(
            torch.logical_not(torch.any(mapped_rows >= int(out_capture_scores.shape[0]))),
            f"{name} capture row mapping exceeds layout rows",
        )
        # Note: max_capture_k bound is validated by caller's CPU-side upper
        # bound (max_capture_k_cpu >= layout kv_max). If caller violates, kernel
        # will write OOB; deferred to device-side CUDA error.

    if producer_rows.numel() > 0:
        if not cpu_row_truth_validated and prefill_producer_rows.numel() > 0:
            active_capture_row_by_batch_row_i32.scatter_(
                0,
                prefill_producer_rows,
                prefill_capture_row_by_batch_row_i32.index_select(0, prefill_producer_rows),
            )
        if not cpu_row_truth_validated and decode_producer_rows.numel() > 0:
            active_capture_row_by_batch_row_i32.scatter_(
                0,
                decode_producer_rows,
                refresh_capture_row_by_batch_row_i32.index_select(0, decode_producer_rows),
            )
        if not cpu_row_truth_validated:
            _validate_phase_capacity_async(
                name="prefill",
                phase_rows=prefill_producer_rows,
                capture_row_by_batch_row_i32=prefill_capture_row_by_batch_row_i32,
                out_capture_scores=prefill_out_capture_scores,
            )
            _validate_phase_capacity_async(
                name="refresh",
                phase_rows=decode_producer_rows,
                capture_row_by_batch_row_i32=refresh_capture_row_by_batch_row_i32,
                out_capture_scores=refresh_out_capture_scores,
            )

    # Rev 2: upper bounds from CPU args (caller computes from layout/authority,
    # not GPU tensor max).
    max_capture_k = int(max_capture_k_cpu)
    max_capture_last_n = int(max_capture_last_n_cpu)
    # Merge in layout capture_head_stride which is static CPU property.
    if prefill_out_capture_scores is not None:
        max_capture_k = max(max_capture_k, _capture_head_stride(prefill_out_capture_scores))
    if refresh_out_capture_scores is not None:
        max_capture_k = max(max_capture_k, _capture_head_stride(refresh_out_capture_scores))

    if prefill_layout is not None:
        num_heads = int(prefill_layout.num_heads)
    elif refresh_layout is not None:
        num_heads = int(refresh_layout.num_heads)
    else:
        raise ValueError("at least one capture layout is required")
    # The scheduler may co-schedule any row count up to max_num_seqs. Key the
    # allocation by the sealed rows capacity so every legal live batch hits the
    # prebuilt slab; slice the returned view back to the actual producer rows.
    # Unstamped non-capture runs retain their exact live row count.
    _actual_capture_rows = int(len(producer_rows_cpu))
    _capture_rows_bucket = 0
    if scratch_cache_owner is not None:
        _capture_rows_bucket = int(
            getattr(scratch_cache_owner, "_capture_rows_bucket", 0) or 0
        )
    _alloc_capture_rows = _actual_capture_rows
    if _capture_rows_bucket > _actual_capture_rows:
        _alloc_capture_rows = _capture_rows_bucket
    scratch_shape = (
        int(_alloc_capture_rows),
        num_heads,
        max_capture_last_n,
        max_capture_k,
    )
    scratch_layer_slots = 0
    if isinstance(scratch_cache_extra_key, tuple) and scratch_cache_extra_key:
        key_kind = str(scratch_cache_extra_key[0])
        if key_kind in {
            "capture_postprocess_ring",
            "capture_postprocess_cohort",
        }:
            try:
                scratch_layer_slots = int(scratch_cache_extra_key[-1])
            except (TypeError, ValueError):
                scratch_layer_slots = 0
            scratch_layer_slots = max(int(scratch_layer_slots), (int(ring_scratch_slot) if int(ring_scratch_slot) >= 0 else int(slot_in_chunk)) + 1)
    scratch_storage_shape = (
        (int(scratch_layer_slots), *scratch_shape)
        if int(scratch_layer_slots) > 0
        else scratch_shape
    )
    # Historically FA4 CuTe capture wrote fp32 scores while SM80/FA3 kept half
    # logits; capture is now UNIFIED to fp16 across backends (see the note below).
    # Select the ABI dtype before allocation so the call path does not cast/copy.
    #
    # Unified fp16 capture (see capture_layout_worker.py): FA4 scratch is fp16
    # like FA3/SM80, consistent with the capture buffer the CuTe kernel writes.
    scratch_dtype = torch.float16
    scratch_key = (
        str(device.type),
        -1 if device.index is None else int(device.index),
        scratch_dtype,
        scratch_storage_shape,
        scratch_cache_extra_key,
    )
    sealed_cache_required = False
    if scratch_cache_owner is not None:
        ownership_plan = getattr(
            scratch_cache_owner, "_capture_ownership_plan", None
        )
        if ownership_plan is not None:
            if (
                not isinstance(ownership_plan, CaptureOwnershipPlan)
                or ownership_plan.schema != CAPTURE_OWNERSHIP_POLICY_SCHEMA
            ):
                raise RuntimeError(
                    "E_SFI_CAPTURE_OWNERSHIP_CACHE_MISS: invalid stamped "
                    "capture ownership plan"
                )
            sealed_cache_required = True
            expected_extra_key = ownership_plan.scratch_extra_key
            expected_shape = ownership_plan.scratch_storage_shape
            if (
                scratch_cache_extra_key != expected_extra_key
                or scratch_storage_shape != expected_shape
            ):
                raise RuntimeError(
                    "E_SFI_CAPTURE_OWNERSHIP_CACHE_MISS: live scratch "
                    "key/shape does not match the stamped owner; "
                    f"live_extra_key={scratch_cache_extra_key!r}, "
                    f"expected_extra_key={expected_extra_key!r}, "
                    f"live_shape={scratch_storage_shape!r}, "
                    f"expected_shape={expected_shape!r}"
                )
    scratch_storage = None
    if scratch_cache_owner is not None:
        cached_key = getattr(scratch_cache_owner, "_fa3_capture_scratch_cache_key", None)
        cached_tensor = getattr(scratch_cache_owner, "_fa3_capture_scratch_cache_tensor", None)
        if scratch_cache_extra_key is not None:
            cache_map = getattr(scratch_cache_owner, "_fa3_capture_scratch_cache_by_key", None)
            if isinstance(cache_map, dict):
                cached_tensor = cache_map.get(scratch_key)
                cached_key = scratch_key if cached_tensor is not None else None
        if (
            cached_key == scratch_key
            and isinstance(cached_tensor, torch.Tensor)
            and cached_tensor.device == device
            and cached_tensor.dtype == scratch_dtype
            and tuple(int(v) for v in cached_tensor.shape) == scratch_storage_shape
        ):
            scratch_storage = cached_tensor
    scratch_cache_hit = scratch_storage is not None
    if scratch_storage is None:
        if sealed_cache_required:
            raise RuntimeError(
                "E_SFI_CAPTURE_OWNERSHIP_CACHE_MISS: stamped owner requires "
                "an exact prebuilt scratch cache hit"
            )
        # The native capture store runs after FA masking and postprocess reads
        # only producer rows within [last_n, effective_kv_len]. Reusing an
        # uninitialized scratch buffer avoids per-layer allocator/fill work; the
        # same stream orders each layer's postprocess before the next capture.
        scratch_storage = torch.empty(
            scratch_storage_shape,
            device=device,
            dtype=scratch_dtype,
        )
        if scratch_cache_owner is not None:
            if scratch_cache_extra_key is None:
                setattr(scratch_cache_owner, "_fa3_capture_scratch_cache_key", scratch_key)
                setattr(scratch_cache_owner, "_fa3_capture_scratch_cache_tensor", scratch_storage)
            else:
                cache_map = getattr(scratch_cache_owner, "_fa3_capture_scratch_cache_by_key", None)
                if not isinstance(cache_map, dict):
                    cache_map = {}
                    setattr(scratch_cache_owner, "_fa3_capture_scratch_cache_by_key", cache_map)
                # One current exact allocation per real chunk/cohort scope.
                # Full-key hit above remains one dict lookup; scope bookkeeping
                # and UAF-safe replacement run only on a cold shape change.
                _store_capture_scratch_cache_entry(
                    cache_owner=scratch_cache_owner,
                    cache_map=cache_map,
                    scratch_key=scratch_key,
                    scratch_storage=scratch_storage,
                )
    if _CAPTURE_SCRATCH_PROBE_LOG:
        _probe_key_kind = (
            str(scratch_cache_extra_key[0])
            if isinstance(scratch_cache_extra_key, tuple)
            and scratch_cache_extra_key
            else "default"
        )
        log_capture_scratch_probe(
            source="live",
            cache_key=scratch_key,
            cache_hit=bool(scratch_cache_hit),
            scratch_storage_shape=scratch_storage_shape,
            scratch_dtype=scratch_dtype,
            element_size_bytes=int(scratch_storage.element_size()),
            actual_rows=int(_actual_capture_rows),
            bucket_rows=int(_alloc_capture_rows),
            heads=int(num_heads),
            last_n=int(max_capture_last_n),
            capture_k=int(max_capture_k),
            key_kind=_probe_key_kind,
        )
    _scr_slot = int(ring_scratch_slot) if int(ring_scratch_slot) >= 0 else int(slot_in_chunk)
    if int(scratch_layer_slots) > 0:
        if int(_scr_slot) < 0 or int(_scr_slot) >= int(scratch_layer_slots):
            raise ValueError("ring/chunk scratch slot exceeds scratch storage")
        scratch_capture_scores = scratch_storage[int(_scr_slot)]
    else:
        scratch_capture_scores = scratch_storage
    # 262k OOM fix (dim-1 row bucketing): if dim-1 was over-allocated to the
    # concurrent-prefill bucket above (for cache-key invariance), slice the VIEW back
    # to the ACTUAL producer-row count. A prefix slice along dim-0 of a contiguous
    # tensor stays contiguous with IDENTICAL inner strides (stride(1)/stride(2) read by
    # postprocess.py:429,506-507 are unchanged), so the store + postprocess see the
    # exact HEAD shape (postprocess.py:122 requires shape[0] == producer rows) -> the
    # bucket lives ONLY in the allocation/key, numerics are bit-identical.
    if int(_alloc_capture_rows) > int(_actual_capture_rows):
        scratch_capture_scores = scratch_capture_scores[: int(_actual_capture_rows)]

    prefill_kv_len_cpu_src = (
        getattr(prefill_layout, "kv_len_per_row_cpu", None)
        if prefill_layout is not None
        else None
    )
    refresh_kv_len_cpu_src = (
        getattr(refresh_layout, "kv_len_per_row_cpu", None)
        if refresh_layout is not None
        else None
    )
    prefill_kv_len_cpu = (
        tuple(int(v) for v in prefill_kv_len_cpu_src)
        if prefill_kv_len_cpu_src is not None
        else tuple()
    )
    refresh_kv_len_cpu = (
        tuple(int(v) for v in refresh_kv_len_cpu_src)
        if refresh_kv_len_cpu_src is not None
        else tuple()
    )

    return CaptureForwardSideOutputs(
        scratch_capture_scores=scratch_capture_scores,
        capture_row_index_i32=capture_row_index_i32,
        producer_rows_i32=producer_rows_i32,
        row_capture_last_n_i32=row_capture_last_n_i32,
        row_is_prefill_producer=row_is_prefill_producer,
        active_capture_row_by_batch_row_i32=active_capture_row_by_batch_row_i32,
        prefill_capture_row_by_batch_row_i32=prefill_capture_row_by_batch_row_i32,
        refresh_capture_row_by_batch_row_i32=refresh_capture_row_by_batch_row_i32,
        prefill_out_capture_scores=prefill_out_capture_scores,
        prefill_out_log_f_denoms=prefill_out_log_f_denoms,
        refresh_out_capture_scores=refresh_out_capture_scores,
        refresh_out_log_f_denoms=refresh_out_log_f_denoms,
        max_capture_k=max_capture_k,
        max_capture_last_n=max_capture_last_n,
        producer_rows_cpu=producer_rows_cpu_tuple,
        row_capture_last_n_cpu=row_capture_last_n_cpu_tuple,
        row_is_prefill_producer_cpu=row_is_prefill_producer_cpu_tuple,
        seqused_k_cpu=seqused_k_cpu_tuple,
        active_capture_row_by_batch_row_cpu=tuple(active_capture_row_cpu),
        prefill_out_kv_len_per_capture_row_cpu=prefill_kv_len_cpu,
        refresh_out_kv_len_per_capture_row_cpu=refresh_kv_len_cpu,
    )
