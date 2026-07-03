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

import os

import torch

from patches.cpu_gpu_staging import cached_sequence_to_device
from patches.fa3_native.row_plan import MixedPageRowPlan
from patches.sparse_types import CaptureForwardSideOutputs, StepCaptureLayout

# Upper bound on distinct deferred/async-postprocess capture scratch buffers retained
# at once. The deferred key embeds step_handle_id/generation (unique per step), so
# without a cap this dict grew one ~235 MB fp32 scratch per step and OOM'd under a
# continuous server load. 8 covers a few in-flight deferred steps x layer-chunk groups;
# record_stream-deferred eviction keeps it async-UAF-safe. Raise to disable bounding.
_FA3_CAPTURE_SCRATCH_CACHE_CAP = max(
    2, int(os.environ.get("VLLM_SPARSE_FA3_CAPTURE_SCRATCH_CACHE_CAP", "8") or "8")
)


def _prepare_phase_outputs(
    *,
    layout: Optional[StepCaptureLayout],
    batch_size: int,
    slot_in_chunk: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    device = torch.device(device)
    if layout is None:
        mapping = torch.full((batch_size,), -1, device=device, dtype=torch.int32)
        return mapping, None, None
    mapping = getattr(layout, "active_capture_row_by_batch_row_i32", None)
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
    if capture_rows > 0:
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
    seqused_k = seqused_k.to(device=device, dtype=torch.int32).reshape(batch_size)

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
    prefill_row_set = set(prefill_rows_cpu_tuple)
    producer_row_set = set(producer_rows_cpu_tuple)
    row_capture_last_n_cpu_tuple = (
        tuple(max(0, int(v)) for v in row_capture_last_n_cpu)
        if row_capture_last_n_cpu is not None
        else tuple()
    )
    seqused_k_cpu_tuple = (
        tuple(max(0, int(v)) for v in seqused_k_cpu)
        if seqused_k_cpu is not None
        else tuple()
    )
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

    active_capture_row_cpu = [-1] * batch_size
    prefill_row_to_capture = _layout_row_to_capture(prefill_layout)
    refresh_row_to_capture = _layout_row_to_capture(refresh_layout)
    for batch_row in prefill_rows_cpu_tuple:
        if 0 <= int(batch_row) < batch_size and int(batch_row) in prefill_row_to_capture:
            active_capture_row_cpu[int(batch_row)] = int(prefill_row_to_capture[int(batch_row)])
    for batch_row in decode_rows_cpu_tuple:
        if 0 <= int(batch_row) < batch_size and int(batch_row) in refresh_row_to_capture:
            active_capture_row_cpu[int(batch_row)] = int(refresh_row_to_capture[int(batch_row)])

    has_cpu_row_truth = (
        len(row_capture_last_n_cpu_tuple) >= batch_size
        and (bool(prefill_rows_cpu_tuple) or bool(decode_rows_cpu_tuple))
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
    )
    cached_side = None
    if scratch_cache_owner is not None:
        cached_key = getattr(scratch_cache_owner, "_fa3_capture_side_tensor_cache_key", None)
        cached_value = getattr(scratch_cache_owner, "_fa3_capture_side_tensor_cache", None)
        if cached_key == side_tensor_key and isinstance(cached_value, tuple) and len(cached_value) == 8:
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
        ) = cached_side
    elif has_cpu_row_truth:
        capture_row_index_cpu = [-1] * batch_size
        for idx, row in enumerate(producer_rows_cpu_tuple):
            if 0 <= int(row) < batch_size:
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
                ),
            )
    else:
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
        if not has_cpu_row_truth and prefill_producer_rows.numel() > 0:
            active_capture_row_by_batch_row_i32.scatter_(
                0,
                prefill_producer_rows,
                prefill_capture_row_by_batch_row_i32.index_select(0, prefill_producer_rows),
            )
        if not has_cpu_row_truth and decode_producer_rows.numel() > 0:
            active_capture_row_by_batch_row_i32.scatter_(
                0,
                decode_producer_rows,
                refresh_capture_row_by_batch_row_i32.index_select(0, decode_producer_rows),
            )
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
    # 262k OOM fix (dim-1 row bucketing; symmetric with the EDIT-1 kv / EDIT-7 last_n
    # force-form): the DEFER scratch_key embeds scratch_storage_shape, whose dim-1 is
    # the producer-row count len(producer_rows_cpu). Under --max-num-seqs N the vLLM
    # v1 0.19 scheduler may co-schedule up to N prefills in ONE forward (it does NOT
    # honor max_num_partial_prefills), so this count varies 1..N; a count != the
    # profile-prebuilt one MISSes the slab -> the ~3GB lazy torch.empty at :459 re-OOMs
    # the hot path. Round the ALLOCATED/KEYED dim-1 UP to the prebuilt
    # _capture_rows_bucket (= the prebuild's producer_rows_worst, default max_num_seqs)
    # so the key is INVARIANT and HITs. The extra rows live ONLY in the allocation +
    # cache key; the downstream VIEW (scratch_capture_scores) is sliced back to the
    # actual count below, so every consumer (postprocess.py:122 asserts shape[0] ==
    # producer rows; the store writes by capture_row) sees the EXACT HEAD shape/strides
    # -> bit-identical numerics. 0 bucket (prebuild skipped / non-capture run) => no-op
    # => exact HEAD behaviour (fail-safe). scratch_cache_owner is the controller (the
    # cache map + bucket are stamped on it by _prebuild_capture_buffers).
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
        if key_kind in {"async_postprocess_chunk", "defer_postprocess_chunk"}:
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
    if scratch_storage is None:
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
                cache_map[scratch_key] = scratch_storage
                # _fa3_capture_scratch_cache LRU bound (DEFENSIVE, now mostly inert):
                # the DEFER key is keyed by chunk_id (sync_fa4_capture_scratch_chunkid
                # _reuse), NOT the per-step nonce, so this dict holds <= num_chunks (3)
                # entries and the prebuild pre-populates them -> the cap is never hit on
                # the capture path. Retained for OTHER extra_key kinds and any
                # non-prebuilt fallback: evict oldest (FIFO) beyond the cap, record_
                # stream on current + refresh streams so the allocator defers the free
                # past any still-pending deferred read.
                if len(cache_map) > _FA3_CAPTURE_SCRATCH_CACHE_CAP:
                    _capturing = False
                    try:
                        _capturing = bool(torch.cuda.is_current_stream_capturing())
                    except Exception:
                        _capturing = False
                    if not _capturing:
                        _rs = getattr(scratch_cache_owner, "refresh_stream", None)
                        _cur = (
                            torch.cuda.current_stream(device=device)
                            if device.type == "cuda"
                            else None
                        )
                        for _ek in list(cache_map.keys()):
                            if len(cache_map) <= _FA3_CAPTURE_SCRATCH_CACHE_CAP:
                                break
                            if _ek == scratch_key:
                                continue
                            _old = cache_map.pop(_ek, None)
                            if (
                                isinstance(_old, torch.Tensor)
                                and _old.is_cuda
                            ):
                                if _cur is not None:
                                    _old.record_stream(_cur)
                                if _rs is not None:
                                    try:
                                        _old.record_stream(_rs)
                                    except Exception:
                                        pass
                            del _old
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
