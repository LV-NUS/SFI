from __future__ import annotations

import os
from typing import Any, Optional, Sequence

import torch












def _resolve_phase_output_kv_len(
    *,
    kv_len: int,
    capture_row: int,
    out_capture_scores: Optional[torch.Tensor],
    out_kv_len_per_capture_row_i32: Optional[torch.Tensor],
) -> int:
    effective_kv_len = max(0, int(kv_len))
    if (
        isinstance(out_kv_len_per_capture_row_i32, torch.Tensor)
        and 0 <= int(capture_row) < int(out_kv_len_per_capture_row_i32.numel())
    ):
        effective_kv_len = min(
            effective_kv_len,
            max(0, int(out_kv_len_per_capture_row_i32[int(capture_row)].item())),
        )
    if isinstance(out_capture_scores, torch.Tensor) and out_capture_scores.dim() >= 4:
        effective_kv_len = min(effective_kv_len, int(out_capture_scores.shape[-1]))
    return effective_kv_len


def _resolve_phase_output_kv_len_cpu(
    *,
    kv_len: int,
    capture_row: int,
    out_capture_scores: Optional[torch.Tensor],
    out_kv_len_per_capture_row_cpu: Sequence[int] | None,
) -> int:
    effective_kv_len = max(0, int(kv_len))
    if (
        out_kv_len_per_capture_row_cpu is not None
        and 0 <= int(capture_row) < len(out_kv_len_per_capture_row_cpu)
    ):
        effective_kv_len = min(
            effective_kv_len,
            max(0, int(out_kv_len_per_capture_row_cpu[int(capture_row)])),
        )
    if isinstance(out_capture_scores, torch.Tensor) and out_capture_scores.dim() >= 4:
        effective_kv_len = min(effective_kv_len, int(out_capture_scores.shape[-1]))
    return effective_kv_len




def _validate_phase_outputs(
    *,
    name: str,
    capture_row_by_batch_row_i32: torch.Tensor,
    out_capture_scores: Optional[torch.Tensor],
    out_log_f_denoms: Optional[torch.Tensor],
    batch_size: int,
    device: torch.device,
) -> None:
    if capture_row_by_batch_row_i32.dim() != 1:
        raise ValueError(f"{name} capture row mapping must be rank-1")
    if int(capture_row_by_batch_row_i32.numel()) != batch_size:
        raise ValueError(f"{name} capture row mapping must align with batch size")
    if capture_row_by_batch_row_i32.device != device:
        raise ValueError(f"{name} capture row mapping must share device")
    if out_capture_scores is None and out_log_f_denoms is None:
        return
    if out_capture_scores is None or out_log_f_denoms is None:
        raise ValueError(f"{name} outputs must be provided together")
    if out_capture_scores.dim() != 4:
        raise ValueError(f"{name} out_capture_scores must be [capture_rows, heads, 1, kv]")
    if out_log_f_denoms.dim() != 2:
        raise ValueError(f"{name} out_log_f_denoms must be [capture_rows, heads]")
    if int(out_capture_scores.shape[2]) != 1:
        raise ValueError(f"{name} out_capture_scores must have window == 1")
    if out_capture_scores.device != device or out_log_f_denoms.device != device:
        raise ValueError(f"{name} outputs must share device with scratch")
    if int(out_capture_scores.shape[0]) != int(out_log_f_denoms.shape[0]):
        raise ValueError(f"{name} output row count must align")
    if int(out_capture_scores.shape[1]) != int(out_log_f_denoms.shape[1]):
        raise ValueError(f"{name} output head count must align")


def _validate_prefill_postprocess_inputs(
    *,
    scratch_capture_scores: torch.Tensor,
    producer_rows_i32: torch.Tensor,
    row_capture_last_n_i32: torch.Tensor,
    row_is_prefill_producer: torch.Tensor,
    seqused_k: torch.Tensor,
    active_capture_row_by_batch_row_i32: torch.Tensor,
    prefill_out_capture_scores: Optional[torch.Tensor],
    prefill_out_log_f_denoms: Optional[torch.Tensor],
    refresh_out_capture_scores: Optional[torch.Tensor],
    refresh_out_log_f_denoms: Optional[torch.Tensor],
) -> None:
    if scratch_capture_scores.dim() != 4:
        raise ValueError("scratch_capture_scores must be [capture_rows, heads, tail_q, kv]")
    if producer_rows_i32.dim() != 1:
        raise ValueError("producer_rows_i32 must be rank-1")
    if row_capture_last_n_i32.dim() != 1:
        raise ValueError("row_capture_last_n_i32 must be rank-1")
    if row_is_prefill_producer.dim() != 1 or seqused_k.dim() != 1:
        raise ValueError("row_is_prefill_producer and seqused_k must be rank-1")
    batch_size = int(row_capture_last_n_i32.numel())
    if int(row_is_prefill_producer.numel()) != batch_size:
        raise ValueError("row_is_prefill_producer must align with row_capture_last_n_i32")
    if int(seqused_k.numel()) != batch_size:
        raise ValueError("seqused_k must align with row_capture_last_n_i32")
    if int(producer_rows_i32.numel()) != int(scratch_capture_scores.shape[0]):
        raise ValueError("producer_rows_i32 must align with scratch capture rows")
    if producer_rows_i32.device != scratch_capture_scores.device:
        raise ValueError("producer_rows_i32 must share device with scratch_capture_scores")

    device = scratch_capture_scores.device
    _validate_phase_outputs(
        name="active",
        capture_row_by_batch_row_i32=active_capture_row_by_batch_row_i32,
        out_capture_scores=None,
        out_log_f_denoms=None,
        batch_size=batch_size,
        device=device,
    )
    _validate_phase_outputs(
        name="prefill",
        capture_row_by_batch_row_i32=active_capture_row_by_batch_row_i32,
        out_capture_scores=prefill_out_capture_scores,
        out_log_f_denoms=prefill_out_log_f_denoms,
        batch_size=batch_size,
        device=device,
    )
    _validate_phase_outputs(
        name="refresh",
        capture_row_by_batch_row_i32=active_capture_row_by_batch_row_i32,
        out_capture_scores=refresh_out_capture_scores,
        out_log_f_denoms=refresh_out_log_f_denoms,
        batch_size=batch_size,
        device=device,
    )


def _stage_meta_rows(
    rows: Sequence[Sequence[int]],
    *,
    dtype: torch.dtype,
    device: torch.device,
    cache_owner: Optional[object],
    cache_name: str,
) -> torch.Tensor:
    row_count = len(rows)
    if row_count <= 0:
        return torch.empty((0, 0), dtype=dtype, device=device)
    col_count = len(rows[0])
    if col_count <= 0:
        raise ValueError("capture postprocess metadata rows must be non-empty")
    for row in rows:
        if len(row) != col_count:
            raise ValueError("capture postprocess metadata rows must have fixed width")

    key = (
        str(device.type),
        -1 if device.index is None else int(device.index),
        dtype,
        row_count,
        col_count,
    )
    # [STAGE-RING-FIX 2026-07-02] The old single cached (cpu_stage, gpu_stage)
    # pair had a host-side WAR race: `copy_(cpu_stage, non_blocking=True)` from
    # PINNED memory returns immediately, and the NEXT call (flush fires the
    # postprocess for ~28 layers back-to-back) rewrote the same pinned buffer
    # while the previous H2D was still in flight. The meta rows carry kv_len,
    # strides and RAW out-tensor pointers, so a torn/stale transfer made the
    # reduce/copy kernels write the wrong rows/addresses -- the source of the
    # run-to-run capture-content jitter (and OOB reads) that survived every
    # producer-side snapshot fix. Fix: a small ring of staging slots with one
    # H2D-completion event per slot; a slot is rewritten only after ITS event
    # has completed (a ring-depth of transfers earlier -- in practice already
    # done, so the sync is a ~1us no-op and the pipeline never stalls).
    ring_depth = 4
    slots_attr = f"_fa3_capture_postprocess_{cache_name}_ring"
    state = getattr(cache_owner, slots_attr, None) if cache_owner is not None else None
    if not isinstance(state, dict) or state.get("key") != key:
        state = {"key": key, "slots": [], "next": 0}
        if cache_owner is not None:
            setattr(cache_owner, slots_attr, state)
    slots = state["slots"]
    if len(slots) < ring_depth:
        try:
            cpu_stage = torch.empty((row_count, col_count), dtype=dtype, device="cpu", pin_memory=True)
        except RuntimeError:
            cpu_stage = torch.empty((row_count, col_count), dtype=dtype, device="cpu")
        gpu_stage = torch.empty((row_count, col_count), dtype=dtype, device=device)
        slot = {"cpu": cpu_stage, "gpu": gpu_stage, "evt": None}
        slots.append(slot)
    else:
        slot = slots[int(state["next"]) % ring_depth]
        state["next"] = int(state["next"]) + 1
        evt = slot.get("evt")
        if evt is not None:
            evt.synchronize()
        cpu_stage = slot["cpu"]
        gpu_stage = slot["gpu"]

    # [META-STAGE-BULK 2026-07-06] 逐元素 Python setitem（rows×cols×36 层/步）
    # 换 C 层一次构造+整块拷贝：值/位置逐位等价，纯 host 构造提速 ~10×。
    cpu_stage.copy_(torch.tensor(rows, dtype=dtype))
    gpu_stage.copy_(cpu_stage, non_blocking=True)
    if device.type == "cuda":
        evt = torch.cuda.Event(enable_timing=False)
        evt.record(torch.cuda.current_stream(device=device))
        slot["evt"] = evt
    return gpu_stage


def _select_phase_outputs(
    *,
    batch_row: int,
    row_is_prefill_producer: torch.Tensor,
    active_capture_row_by_batch_row_i32: torch.Tensor,
    prefill_out_capture_scores: Optional[torch.Tensor],
    prefill_out_log_f_denoms: Optional[torch.Tensor],
    refresh_out_capture_scores: Optional[torch.Tensor],
    refresh_out_log_f_denoms: Optional[torch.Tensor],
) -> tuple[int, torch.Tensor, torch.Tensor]:
    if bool(row_is_prefill_producer[batch_row].item()):
        if prefill_out_capture_scores is None or prefill_out_log_f_denoms is None:
            raise ValueError("prefill producer rows require prefill output tensors")
        capture_row = int(active_capture_row_by_batch_row_i32[batch_row].item())
        if capture_row < 0:
            raise ValueError("prefill producer row is missing prefill capture row mapping")
        return capture_row, prefill_out_capture_scores, prefill_out_log_f_denoms

    if refresh_out_capture_scores is None or refresh_out_log_f_denoms is None:
        raise ValueError("decode producer rows require refresh output tensors")
    capture_row = int(active_capture_row_by_batch_row_i32[batch_row].item())
    if capture_row < 0:
        raise ValueError("decode producer row is missing refresh capture row mapping")
    return capture_row, refresh_out_capture_scores, refresh_out_log_f_denoms


def postprocess_prefill_capture_scores(
    *,
    scratch_capture_scores: torch.Tensor,
    producer_rows_i32: torch.Tensor,
    row_capture_last_n_i32: torch.Tensor,
    row_is_prefill_producer: torch.Tensor,
    seqused_k: torch.Tensor,
    active_capture_row_by_batch_row_i32: torch.Tensor,
    prefill_out_capture_scores: Optional[torch.Tensor],
    prefill_out_log_f_denoms: Optional[torch.Tensor],
    refresh_out_capture_scores: Optional[torch.Tensor],
    refresh_out_log_f_denoms: Optional[torch.Tensor],
    prefill_out_kv_len_per_capture_row_i32: Optional[torch.Tensor] = None,
    refresh_out_kv_len_per_capture_row_i32: Optional[torch.Tensor] = None,
    producer_rows_cpu: Optional[Sequence[int]] = None,
    row_capture_last_n_cpu: Optional[Sequence[int]] = None,
    row_is_prefill_producer_cpu: Optional[Sequence[bool]] = None,
    seqused_k_cpu: Optional[Sequence[int]] = None,
    active_capture_row_by_batch_row_cpu: Optional[Sequence[int]] = None,
    prefill_out_kv_len_per_capture_row_cpu: Optional[Sequence[int]] = None,
    refresh_out_kv_len_per_capture_row_cpu: Optional[Sequence[int]] = None,
    skip_postprocess_rows_cpu: Optional[Sequence[int]] = None,
    row_capture_accum_prev_rows_cpu: Optional[Sequence[int]] = None,
    row_capture_accum_prev_capacity_cpu: Optional[Sequence[int]] = None,
    meta_cache_owner: Optional[object] = None,
    alpha: float,
    debug_epoch: Optional[int] = None,
    debug_layer_index: Optional[int] = None,
) -> None:
    _validate_prefill_postprocess_inputs(
        scratch_capture_scores=scratch_capture_scores,
        producer_rows_i32=producer_rows_i32,
        row_capture_last_n_i32=row_capture_last_n_i32,
        row_is_prefill_producer=row_is_prefill_producer,
        seqused_k=seqused_k,
        active_capture_row_by_batch_row_i32=active_capture_row_by_batch_row_i32,
        prefill_out_capture_scores=prefill_out_capture_scores,
        prefill_out_log_f_denoms=prefill_out_log_f_denoms,
        refresh_out_capture_scores=refresh_out_capture_scores,
        refresh_out_log_f_denoms=refresh_out_log_f_denoms,
    )
    if scratch_capture_scores.device.type != "cuda":
        raise ValueError("postprocess_prefill_capture_scores requires CUDA tensors")

    from utils import selector_log_s_ext

    device = scratch_capture_scores.device
    def _profiled_postprocess_call(
        *,
        label: str,
        metadata: dict[str, object],
        call: Any,
    ) -> Any:
        return call()

    batch_size = int(row_capture_last_n_i32.numel())
    producer_rows_i32 = producer_rows_i32.to(device=device, dtype=torch.int32).reshape(-1)
    row_capture_last_n_i32 = row_capture_last_n_i32.to(device=device, dtype=torch.int32).reshape(batch_size)
    row_is_prefill_producer = row_is_prefill_producer.to(device=device, dtype=torch.bool).reshape(batch_size)
    seqused_k = seqused_k.to(device=device, dtype=torch.int32).reshape(batch_size)
    active_capture_row_by_batch_row_i32 = active_capture_row_by_batch_row_i32.reshape(batch_size)

    decode_gt1_rows = []
    lastn1_rows: list[tuple[int, int, int, torch.Tensor, torch.Tensor]] = []
    gt1_prefill_rows: list[tuple[int, int, int, int]] = []

    producer_rows_tuple = (
        tuple(int(v) for v in producer_rows_cpu)
        if producer_rows_cpu is not None
        else tuple()
    )
    if producer_rows_tuple and len(producer_rows_tuple) != int(producer_rows_i32.numel()):
        raise ValueError("producer_rows_cpu must align with producer_rows_i32")
    row_capture_last_n_tuple = (
        tuple(max(0, int(v)) for v in row_capture_last_n_cpu)
        if row_capture_last_n_cpu is not None
        else tuple()
    )
    row_is_prefill_tuple = (
        tuple(bool(v) for v in row_is_prefill_producer_cpu)
        if row_is_prefill_producer_cpu is not None
        else tuple()
    )
    seqused_k_tuple = (
        tuple(max(0, int(v)) for v in seqused_k_cpu)
        if seqused_k_cpu is not None
        else tuple()
    )
    active_capture_row_tuple = (
        tuple(int(v) for v in active_capture_row_by_batch_row_cpu)
        if active_capture_row_by_batch_row_cpu is not None
        else tuple()
    )
    skip_postprocess_rows = (
        frozenset(int(v) for v in skip_postprocess_rows_cpu)
        if skip_postprocess_rows_cpu is not None
        else frozenset()
    )
    # [CHUNKED-CAPTURE-ACCUMULATE 2026-07-06] per-row 跨片累计元数据
    # （-1/缺省=非跨片行走原路径）。跨片行必须走 gt1 reduce（accumulate
    # 位路由），含 last_n==1 的尾片——lastn1 裸拷贝的常数平移破坏合并。
    accum_prev_rows_tuple = (
        tuple(int(v) for v in row_capture_accum_prev_rows_cpu)
        if row_capture_accum_prev_rows_cpu is not None
        else tuple()
    )
    accum_prev_capacity_tuple = (
        tuple(int(v) for v in row_capture_accum_prev_capacity_cpu)
        if row_capture_accum_prev_capacity_cpu is not None
        else tuple()
    )

    def _accum_prev_rows_for(batch_row: int) -> int:
        if batch_row < len(accum_prev_rows_tuple):
            return int(accum_prev_rows_tuple[batch_row])
        return -1

    def _accum_prev_capacity_for(batch_row: int) -> int:
        if batch_row < len(accum_prev_capacity_tuple):
            return max(0, int(accum_prev_capacity_tuple[batch_row]))
        return 0
    has_cpu_row_truth = (
        bool(producer_rows_tuple)
        and len(row_capture_last_n_tuple) >= batch_size
        and len(row_is_prefill_tuple) >= batch_size
        and len(seqused_k_tuple) >= batch_size
        and len(active_capture_row_tuple) >= batch_size
    )
    producer_rows_iter = (
        producer_rows_tuple if has_cpu_row_truth else tuple(int(v) for v in producer_rows_i32.tolist())
    )

    for scratch_row, batch_row_i32 in enumerate(producer_rows_iter):
        batch_row = int(batch_row_i32)
        if batch_row in skip_postprocess_rows:
            continue
        if batch_row < 0 or batch_row >= batch_size:
            raise ValueError("producer_rows_i32 contains out-of-range batch row")
        if has_cpu_row_truth:
            last_n = int(row_capture_last_n_tuple[batch_row])
            kv_len = int(seqused_k_tuple[batch_row])
            is_prefill_producer = bool(row_is_prefill_tuple[batch_row])
            capture_row = int(active_capture_row_tuple[batch_row])
            if is_prefill_producer:
                if prefill_out_capture_scores is None or prefill_out_log_f_denoms is None:
                    raise ValueError("prefill producer rows require prefill output tensors")
                out_capture_scores = prefill_out_capture_scores
                out_log_f_denoms = prefill_out_log_f_denoms
                out_kv_len_per_capture_row_cpu = prefill_out_kv_len_per_capture_row_cpu
                out_kv_len_per_capture_row_i32 = prefill_out_kv_len_per_capture_row_i32
            else:
                if refresh_out_capture_scores is None or refresh_out_log_f_denoms is None:
                    raise ValueError("decode producer rows require refresh output tensors")
                out_capture_scores = refresh_out_capture_scores
                out_log_f_denoms = refresh_out_log_f_denoms
                out_kv_len_per_capture_row_cpu = refresh_out_kv_len_per_capture_row_cpu
                out_kv_len_per_capture_row_i32 = refresh_out_kv_len_per_capture_row_i32
            if capture_row < 0:
                raise ValueError("producer row is missing active capture row mapping")
        else:
            last_n = int(row_capture_last_n_i32[batch_row].item())
            kv_len = int(seqused_k[batch_row].item())
            is_prefill_producer = bool(row_is_prefill_producer[batch_row].item())
            capture_row, out_capture_scores, out_log_f_denoms = _select_phase_outputs(
                batch_row=batch_row,
                row_is_prefill_producer=row_is_prefill_producer,
                active_capture_row_by_batch_row_i32=active_capture_row_by_batch_row_i32,
                prefill_out_capture_scores=prefill_out_capture_scores,
                prefill_out_log_f_denoms=prefill_out_log_f_denoms,
                refresh_out_capture_scores=refresh_out_capture_scores,
                refresh_out_log_f_denoms=refresh_out_log_f_denoms,
            )
            out_kv_len_per_capture_row_cpu = None
            out_kv_len_per_capture_row_i32 = (
                prefill_out_kv_len_per_capture_row_i32
                if is_prefill_producer
                else refresh_out_kv_len_per_capture_row_i32
            )
        if kv_len <= 0 or last_n <= 0:
            continue
        effective_kv_len = (
            _resolve_phase_output_kv_len_cpu(
                kv_len=kv_len,
                capture_row=capture_row,
                out_capture_scores=out_capture_scores,
                out_kv_len_per_capture_row_cpu=out_kv_len_per_capture_row_cpu,
            )
            if has_cpu_row_truth and out_kv_len_per_capture_row_cpu is not None
            else _resolve_phase_output_kv_len(
                kv_len=kv_len,
                capture_row=capture_row,
                out_capture_scores=out_capture_scores,
                out_kv_len_per_capture_row_i32=out_kv_len_per_capture_row_i32,
            )
        )
        if effective_kv_len <= 0:
            continue
        row_is_accum = bool(is_prefill_producer) and _accum_prev_rows_for(batch_row) >= 0
        if last_n == 1 and not row_is_accum:
            lastn1_rows.append(
                (scratch_row, capture_row, effective_kv_len, out_capture_scores, out_log_f_denoms)
            )
            continue
        if not is_prefill_producer:
            decode_gt1_rows.append(batch_row)
            continue
        gt1_prefill_rows.append((scratch_row, batch_row, capture_row, effective_kv_len))

    if decode_gt1_rows:
        raise ValueError(
            "decode producer rows must have last_n == 1 in capture postprocess"
        )
    if lastn1_rows:
        meta_i32_rows = []
        meta_i64_rows = []
        scratch_head_stride = int(scratch_capture_scores.stride(1))
        for scratch_row, capture_row, effective_kv_len, out_capture_scores, out_log_f_denoms in lastn1_rows:
            meta_i32_rows.append(
                [
                    int(effective_kv_len),
                    scratch_head_stride,
                    1,
                    0,
                    int(effective_kv_len),
                    8,
                    int(out_capture_scores.stride(1)),
                ]
            )
            meta_i64_rows.append(
                [
                    0,
                    int(scratch_capture_scores[int(scratch_row)].data_ptr()),
                    int(out_capture_scores[int(capture_row)].data_ptr()),
                    int(out_log_f_denoms[int(capture_row)].data_ptr()),
                ]
            )

        def _stage_lastn1_meta() -> tuple[torch.Tensor, torch.Tensor]:
            return (
                _stage_meta_rows(
                    meta_i32_rows,
                    dtype=torch.int32,
                    device=device,
                    cache_owner=meta_cache_owner,
                    cache_name="lastn1_i32",
                ),
                _stage_meta_rows(
                    meta_i64_rows,
                    dtype=torch.int64,
                    device=device,
                    cache_owner=meta_cache_owner,
                    cache_name="lastn1_i64",
                ),
            )

        lastn1_metadata = {
            "epoch": -1 if debug_epoch is None else int(debug_epoch),
            "layer": -1 if debug_layer_index is None else int(debug_layer_index),
            "row_count": len(meta_i32_rows),
            "num_query_heads": int(scratch_capture_scores.shape[1]),
            "effective_kv_len_max": max(
                (int(row[2]) for row in lastn1_rows),
                default=0,
            ),
            "log_f_out_fp32": bool(lastn1_rows[0][3].dtype == torch.float32),
        }
        req_meta_i32, req_meta_i64 = _profiled_postprocess_call(
            label="capture_postprocess_meta_lastn1",
            metadata=lastn1_metadata,
            call=_stage_lastn1_meta,
        )

        def _copy_lastn1() -> None:
            selector_log_s_ext.copy_log_f_lastn1_scratch_cuda(
                req_meta_i32=req_meta_i32,
                req_meta_i64=req_meta_i64,
                num_seqs=len(meta_i32_rows),
                num_query_heads=int(scratch_capture_scores.shape[1]),
                scratch_in_fp16=scratch_capture_scores.dtype == torch.float16,
                log_f_out_fp32=lastn1_rows[0][3].dtype == torch.float32,
            )

        _profiled_postprocess_call(
            label="capture_postprocess_copy_lastn1",
            metadata=lastn1_metadata,
            call=_copy_lastn1,
        )
    if not gt1_prefill_rows:
        return
    if prefill_out_capture_scores is None or prefill_out_log_f_denoms is None:
        raise ValueError("prefill gt1 rows require prefill output tensors")

    scratch_head_stride = int(scratch_capture_scores.stride(1))
    scratch_row_stride = int(scratch_capture_scores.stride(2))
    can_batch_gt1 = (
        scratch_capture_scores.is_contiguous()
        and prefill_out_capture_scores.is_contiguous()
        and prefill_out_log_f_denoms.stride(-1) == 1
        and prefill_out_capture_scores.dtype in (torch.float16, torch.float32)
    )
    if can_batch_gt1:
        meta_i32_rows: list[list[int]] = []
        meta_i64_rows: list[list[int]] = []
        for scratch_row, batch_row, capture_row, effective_kv_len in gt1_prefill_rows:
            last_n = (
                int(row_capture_last_n_tuple[batch_row])
                if has_cpu_row_truth and len(row_capture_last_n_tuple) > batch_row
                else int(row_capture_last_n_i32[batch_row].item())
            )
            # FIX: clamp the reduce's per-seq kv to the captured scratch/out kv width.
            # scratch_capture_scores is graph-frozen at capture-time max_capture_k; a long
            # not-yet-compacted prefill-capture row at replay has effective_kv_len far past
            # it. The capture store already clamps to this width, so rows beyond it were
            # never written; reduce_log_f_pre_scratch must read the same bound or it walks
            # off the frozen scratch (cuda-gdb: OOB read in reduce_log_f_pre_scratch_kernel).
            # [CHUNKED-CAPTURE-ACCUMULATE 2026-07-06] 跨片行置 flags bit4 并携带
            # cols 8/9=(前片累计行数, 前片 reduce kv 宽)；kernel 从 out 反解前片
            # sum-form 做计数加权 LSE 合并。非跨片行 cols 8/9=0 且不置位——kernel
            # 不读新列，数值路径逐位不变（meta 恒 10 列保 staging ring 键稳定）。
            accum_prev = _accum_prev_rows_for(int(batch_row))
            flags = 8 | (16 if accum_prev >= 0 else 0)
            meta_i32_rows.append(
                [
                    int(effective_kv_len),
                    scratch_head_stride,
                    int(last_n),
                    0,
                    int(effective_kv_len),
                    int(flags),
                    int(prefill_out_capture_scores.stride(1)),
                    scratch_row_stride,
                    max(0, int(accum_prev)),
                    _accum_prev_capacity_for(int(batch_row)) if accum_prev >= 0 else 0,
                ]
            )
            meta_i64_rows.append(
                [
                    0,
                    int(scratch_capture_scores[int(scratch_row)].data_ptr()),
                    int(prefill_out_capture_scores[int(capture_row)].data_ptr()),
                    int(prefill_out_log_f_denoms[int(capture_row)].data_ptr()),
                ]
            )

        def _stage_gt1_meta() -> tuple[torch.Tensor, torch.Tensor]:
            return (
                _stage_meta_rows(
                    meta_i32_rows,
                    dtype=torch.int32,
                    device=device,
                    cache_owner=meta_cache_owner,
                    cache_name="gt1_i32",
                ),
                _stage_meta_rows(
                    meta_i64_rows,
                    dtype=torch.int64,
                    device=device,
                    cache_owner=meta_cache_owner,
                    cache_name="gt1_i64",
                ),
            )

        gt1_metadata = {
            "epoch": -1 if debug_epoch is None else int(debug_epoch),
            "layer": -1 if debug_layer_index is None else int(debug_layer_index),
            "row_count": len(meta_i32_rows),
            "num_query_heads": int(scratch_capture_scores.shape[1]),
            "effective_kv_len_max": max(
                (int(row[0]) for row in meta_i32_rows),
                default=0,
            ),
            "last_n_max": max(
                (int(row[2]) for row in meta_i32_rows),
                default=0,
            ),
            "log_f_out_fp32": bool(prefill_out_capture_scores.dtype == torch.float32),
            "alpha": float(alpha),
        }
        req_meta_i32, req_meta_i64 = _profiled_postprocess_call(
            label="capture_postprocess_meta_gt1",
            metadata=gt1_metadata,
            call=_stage_gt1_meta,
        )

        def _reduce_gt1() -> None:
            selector_log_s_ext.reduce_log_f_pre_scratch_cuda(
                req_meta_i32=req_meta_i32,
                req_meta_i64=req_meta_i64,
                num_seqs=len(meta_i32_rows),
                num_query_heads=int(scratch_capture_scores.shape[1]),
                scratch_in_fp16=scratch_capture_scores.dtype == torch.float16,
                log_f_out_fp32=prefill_out_capture_scores.dtype == torch.float32,
                alpha=float(alpha),
            )

        _profiled_postprocess_call(
            label="capture_postprocess_reduce_gt1",
            metadata=gt1_metadata,
            call=_reduce_gt1,
        )
    else:
        raise RuntimeError(
            "production gt1 capture postprocess requires batched reduce; "
            "scalar fallback is diagnostic-only"
        )


def run_prefill_capture_postprocess_if_needed(
    *,
    direct_capture_phase: str = "",
    **kwargs: object,
) -> bool:
    """Run scratch postprocess only when direct capture did not fill the tape."""
    producer_rows_cpu = kwargs.get("producer_rows_cpu")
    row_capture_last_n_cpu = kwargs.get("row_capture_last_n_cpu")
    producer_rows = (
        tuple(int(v) for v in producer_rows_cpu)  # type: ignore[union-attr]
        if producer_rows_cpu is not None
        else tuple()
    )
    last_n_by_row = (
        tuple(max(0, int(v)) for v in row_capture_last_n_cpu)  # type: ignore[union-attr]
        if row_capture_last_n_cpu is not None
        else tuple()
    )
    has_cpu_truth = bool(producer_rows) and len(last_n_by_row) > max(producer_rows, default=-1)
    has_gt1 = bool(has_cpu_truth and any(int(last_n_by_row[row]) > 1 for row in producer_rows))
    kwargs_mut = dict(kwargs)
    # [RING-LASTN1-DRAIN 2026-07-03] under the per-G scratch RING the chunk-tail
    # flush must not read raw ring scratch (the slot is rewritten G*in_flight
    # layers later, depth << chunk). When the caller sets drain_lastn1_rows,
    # last_n==1 rows are NOT skipped: the dedicated lastn1 copy arm drains them
    # into the phase out tensors here (under the WAR fence in early mode / by
    # same-stream order inline) and the payload carries no raw scratch view.
    # Direct-capture prefill phases never wrote scratch, so draining there would
    # copy garbage -- the exclusion below keeps them on the tape-direct contract.
    drain_lastn1_rows = bool(kwargs_mut.pop("drain_lastn1_rows", False)) and (
        str(direct_capture_phase) != "prefill"
    )
    has_lastn1 = bool(
        has_cpu_truth and any(int(last_n_by_row[row]) == 1 for row in producer_rows)
    )
    # [CHUNKED-CAPTURE-ACCUMULATE 2026-07-06] 跨片行（accum_prev_rows>=0）必须
    # 进 gt1 reduce（含 last_n==1 尾片）：不能被 lastn1 skip 剔除，且单独构成
    # scratch_needed（如尾片 q_len==1 且批内无其它 gt1 行的步）。
    accum_rows_cpu = kwargs.get("row_capture_accum_prev_rows_cpu")
    accum_by_row = (
        tuple(int(v) for v in accum_rows_cpu)  # type: ignore[union-attr]
        if accum_rows_cpu is not None
        else tuple()
    )

    def _row_is_accum(row: int) -> bool:
        return row < len(accum_by_row) and int(accum_by_row[row]) >= 0

    has_accum = bool(
        has_cpu_truth and any(_row_is_accum(int(row)) for row in producer_rows)
    )
    if has_cpu_truth and not drain_lastn1_rows:
        kwargs_mut["skip_postprocess_rows_cpu"] = tuple(
            int(row)
            for row in producer_rows
            if int(last_n_by_row[row]) == 1 and not _row_is_accum(int(row))
        )
    scratch_needed = bool(
        has_gt1
        or has_accum
        or (drain_lastn1_rows and has_lastn1)
        or not has_cpu_truth
    )
    if not scratch_needed:
        return False
    postprocess_prefill_capture_scores(**kwargs_mut)  # type: ignore[arg-type]
    return True


def _record_capture_postprocess_tensor_on_current_stream(tensor: object) -> None:
    if not isinstance(tensor, torch.Tensor):
        return
    try:
        tensor.record_stream(torch.cuda.current_stream(device=tensor.device))
    except Exception:
        # record_stream is a lifetime guard; unsupported fake tensors in unit
        # tests should not affect the semantic job contract.
        if tensor.device.type == "cuda":
            raise


def _wait_capture_postprocess_completion_event_if_needed(job: Any) -> bool:
    event = getattr(job, "completion_event", None)
    if event is None or bool(getattr(job, "waited_completion_event", False)):
        return False
    scratch = getattr(job, "scratch_capture_scores", None)
    if isinstance(scratch, torch.Tensor) and scratch.device.type == "cuda":
        stream = torch.cuda.current_stream(device=scratch.device)
    else:
        stream = torch.cuda.current_stream()
    stream.wait_event(event)
    setattr(job, "waited_completion_event", True)
    return True


def run_capture_postprocess_job_if_needed(
    job: Any,
    *,
    meta_cache_owner: Optional[object],
) -> bool:
    """Run one deferred capture postprocess job on the current CUDA stream."""
    if job is None:
        return False
    if bool(getattr(job, "completed", False)):
        _wait_capture_postprocess_completion_event_if_needed(job)
        return False
    scratch = getattr(job, "scratch_capture_scores", None)
    if not isinstance(scratch, torch.Tensor):
        raise ValueError("capture postprocess job requires scratch_capture_scores")
    ready_event = getattr(job, "ready_event", None)
    if ready_event is not None:
        torch.cuda.current_stream(device=scratch.device).wait_event(ready_event)
        setattr(job, "waited_ready_event", True)
    # [DETERMINISTIC-TAPE-WAW 2026-07-03] 与 flush 的 retarget 检查互斥
    # (TOCTOU):锁内置 launched 并快照输出目标——flush 若先进锁完成 retarget,
    # 此处读到 tape 目标+stack 完成事件;flush 若后进锁,读到 launched=True 走
    # 已发射分支(流级屏障保证其 stack 排在本 job kernels 之后)。
    _lifecycle_lock = getattr(job, "lifecycle_lock", None)
    if _lifecycle_lock is not None:
        _lifecycle_lock.acquire()
    try:
        setattr(job, "launched", True)
        tape_stack_evt = getattr(job, "tape_stack_evt", None)
        prefill_out_capture_local = getattr(job, "prefill_out_capture_scores", None)
        prefill_out_denoms_local = getattr(job, "prefill_out_log_f_denoms", None)
    finally:
        if _lifecycle_lock is not None:
            _lifecycle_lock.release()
    # job 输出已被 retarget 到 flush 的私有 tape 时,必须排在 tape 的 baseline
    # stack(提交流)之后写,否则会被 stack 的 arena 旧值覆盖(WAW)。GPU 侧
    # no-op 若 stack 已完成,零热路径开销。
    if tape_stack_evt is not None:
        torch.cuda.current_stream(device=scratch.device).wait_event(tape_stack_evt)
        setattr(job, "tape_stack_evt", None)
    for tensor in (
        scratch,
        getattr(job, "producer_rows_i32", None),
        getattr(job, "row_capture_last_n_i32", None),
        getattr(job, "row_is_prefill_producer", None),
        getattr(job, "seqused_k", None),
        getattr(job, "active_capture_row_by_batch_row_i32", None),
        prefill_out_capture_local,
        prefill_out_denoms_local,
        getattr(job, "refresh_out_capture_scores", None),
        getattr(job, "refresh_out_log_f_denoms", None),
        getattr(job, "prefill_out_kv_len_per_capture_row_i32", None),
        getattr(job, "refresh_out_kv_len_per_capture_row_i32", None),
    ):
        _record_capture_postprocess_tensor_on_current_stream(tensor)
    ran = run_prefill_capture_postprocess_if_needed(
        direct_capture_phase=str(getattr(job, "direct_capture_phase", "")),
        scratch_capture_scores=scratch,
        producer_rows_i32=getattr(job, "producer_rows_i32"),
        row_capture_last_n_i32=getattr(job, "row_capture_last_n_i32"),
        row_is_prefill_producer=getattr(job, "row_is_prefill_producer"),
        seqused_k=getattr(job, "seqused_k"),
        active_capture_row_by_batch_row_i32=getattr(
            job, "active_capture_row_by_batch_row_i32"
        ),
        prefill_out_capture_scores=prefill_out_capture_local,
        prefill_out_log_f_denoms=prefill_out_denoms_local,
        refresh_out_capture_scores=getattr(job, "refresh_out_capture_scores", None),
        refresh_out_log_f_denoms=getattr(job, "refresh_out_log_f_denoms", None),
        prefill_out_kv_len_per_capture_row_i32=getattr(
            job, "prefill_out_kv_len_per_capture_row_i32", None
        ),
        refresh_out_kv_len_per_capture_row_i32=getattr(
            job, "refresh_out_kv_len_per_capture_row_i32", None
        ),
        producer_rows_cpu=getattr(job, "producer_rows_cpu", None),
        row_capture_last_n_cpu=getattr(job, "row_capture_last_n_cpu", None),
        row_is_prefill_producer_cpu=getattr(
            job, "row_is_prefill_producer_cpu", None
        ),
        seqused_k_cpu=getattr(job, "seqused_k_cpu", None),
        active_capture_row_by_batch_row_cpu=getattr(
            job, "active_capture_row_by_batch_row_cpu", None
        ),
        prefill_out_kv_len_per_capture_row_cpu=getattr(
            job, "prefill_out_kv_len_per_capture_row_cpu", None
        ),
        refresh_out_kv_len_per_capture_row_cpu=getattr(
            job, "refresh_out_kv_len_per_capture_row_cpu", None
        ),
        skip_postprocess_rows_cpu=getattr(
            job, "skip_postprocess_rows_cpu", tuple()
        ),
        row_capture_accum_prev_rows_cpu=getattr(
            job, "row_capture_accum_prev_rows_cpu", None
        ),
        row_capture_accum_prev_capacity_cpu=getattr(
            job, "row_capture_accum_prev_capacity_cpu", None
        ),
        drain_lastn1_rows=bool(getattr(job, "drain_lastn1_rows", False)),
        meta_cache_owner=meta_cache_owner,
        alpha=float(getattr(job, "alpha", 0.0) or 0.0),
        debug_epoch=int(getattr(job, "debug_epoch", -1)),
        debug_layer_index=int(getattr(job, "debug_layer_index", -1)),
    )
    setattr(job, "ran_postprocess", bool(ran))
    setattr(job, "completed", True)
    return bool(ran)


def run_capture_postprocess_jobs_for_payloads(
    payloads: Sequence[object],
    *,
    meta_cache_owner: Optional[object],
) -> int:
    """Run unique deferred capture postprocess jobs referenced by payloads."""
    ran_count = 0
    seen: set[int] = set()
    for payload in tuple(payloads or tuple()):
        job = getattr(payload, "capture_postprocess_job", None)
        if job is None:
            continue
        job_id = id(job)
        if job_id in seen:
            continue
        seen.add(job_id)
        if run_capture_postprocess_job_if_needed(
            job,
            meta_cache_owner=meta_cache_owner,
        ):
            ran_count += 1
    return int(ran_count)
