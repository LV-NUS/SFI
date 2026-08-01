from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from collections import OrderedDict

import torch

from patches.cpu_gpu_staging import cached_sequence_to_device
from patches.sparse_constants import (
    _CAPTURE_CHUNK,
    _CAPTURE_IN_FLIGHT,
    _CAPTURE_KV_BUCKET_CACHED,
)
from patches.sparse_types import StepCaptureLayout
from patches.sparse_utils import _align_up_int


def _capture_tensor_bytes(
    *,
    capture_chunk: int,
    slots_cap_bucket: int,
    num_heads: int,
    window: int,
    kv_max_bucket: int,
    capture_element_size: int,
    denom_element_size: int = 4,
) -> tuple[int, int]:
    """Return score/denominator storage from the arena's canonical shape."""
    prefix = int(capture_chunk) * int(slots_cap_bucket) * int(num_heads)
    capture_scores_bytes = (
        prefix
        * int(window)
        * int(kv_max_bucket)
        * int(capture_element_size)
    )
    denom_bytes = prefix * int(denom_element_size)
    return int(capture_scores_bytes), int(denom_bytes)


ARENA_PHASE1_QWEN06_BS2_BUDGET_BYTES = 335544320
# 262k OOM fix: size the default budget from the same chunk/in-flight sources
# as the arena itself. A fixed 4 GiB covered chunk14 (~3.5 GiB) but silently
# became too small after the proven default moved to chunk18 (~4.5 GiB), making
# every 262k reservation fail before allocation. Round the exact two-buffer
# fp16 scores+fp32-denom projection up to the next GiB; operators may still
# override VLLM_SPARSE_CAPTURE_ARENA_BUDGET_BYTES explicitly.
_CAP262K_BUDGET_GRAIN_BYTES = 1024 * 1024 * 1024
_CAP262K_SLOTS_CAP_BUCKET = 8
_CAP262K_NUM_HEADS = 32
_CAP262K_WINDOW = 1
_CAP262K_KV_MAX_BUCKET = 262144
_CAP262K_CAPTURE_BYTES, _CAP262K_DENOM_BYTES = _capture_tensor_bytes(
    capture_chunk=_CAPTURE_CHUNK,
    slots_cap_bucket=_CAP262K_SLOTS_CAP_BUCKET,
    num_heads=_CAP262K_NUM_HEADS,
    window=_CAP262K_WINDOW,
    kv_max_bucket=_CAP262K_KV_MAX_BUCKET,
    capture_element_size=2,
)
_CAP262K_PROJECTED_BYTES = int(_CAPTURE_IN_FLIGHT) * (
    _CAP262K_CAPTURE_BYTES + _CAP262K_DENOM_BYTES
)
ARENA_PHASE1_QWEN06_BS2_CAP262K_BUDGET_BYTES = _align_up_int(
    _CAP262K_PROJECTED_BYTES,
    _CAP262K_BUDGET_GRAIN_BYTES,
)
class CaptureArenaIntent(str, Enum):
    ONE_SHOT_BOOTSTRAP = "one_shot_bootstrap"
    REFRESH = "refresh"
    SENTENCE_TRIGGER = "sentence_trigger"


class ArenaReservationStatus(str, Enum):
    MISSING = "missing"
    READY = "ready"
    FAILED = "failed"


class ArenaBindStatus(str, Enum):
    PREPARED_BIND = "prepared_bind"
    SYNC_EXPANSION_MISS = "sync_expansion_miss"
    DIAGNOSTIC_FAIL = "diagnostic_fail"
    ARENA_BUDGET_EXCEEDED = "arena_budget_exceeded"


@dataclass(frozen=True, slots=True)
class ArenaBucketKey:
    phase: str
    buf_id: int
    capture_chunk: int
    slots_cap_bucket: int
    num_heads: int
    window: int
    kv_max_bucket: int
    dtype: str
    intent: CaptureArenaIntent
    device: str = ""


@dataclass(frozen=True, slots=True)
class CaptureBucketBytes:
    capture_scores_bytes: int
    denom_bytes: int
    metadata_bytes: int = 0

    @property
    def total_bytes(self) -> int:
        return int(self.capture_scores_bytes + self.denom_bytes + self.metadata_bytes)


@dataclass(slots=True)
class ArenaReservation:
    bucket_key: Optional[ArenaBucketKey]
    step_epoch: int
    step_handle_id: int
    step_handle_generation: int
    intent: CaptureArenaIntent
    status: ArenaReservationStatus
    ready_event: object = None
    error_reason: str = ""
    alloc_bytes: int = 0
    prepared_layout_ids: tuple[int, ...] = tuple()
    bucket_keys: tuple[ArenaBucketKey, ...] = tuple()
    arena_bind_status: ArenaBindStatus = ArenaBindStatus.DIAGNOSTIC_FAIL
    arena_prepare_wait_us: float = 0.0

    def is_ready_for(
        self,
        *,
        step_epoch: int,
        step_handle_id: int,
        step_handle_generation: int,
        intent: CaptureArenaIntent,
    ) -> bool:
        return bool(
            self.status is ArenaReservationStatus.READY
            and self.intent is intent
            and int(self.step_epoch) == int(step_epoch)
            and int(self.step_handle_id) == int(step_handle_id)
            and int(self.step_handle_generation) == int(step_handle_generation)
            and (self.bucket_key is not None or bool(self.bucket_keys))
            and bool(self.prepared_layout_ids)
        )


@dataclass(slots=True)
class ArenaMetrics:
    arena_reserved_bytes: int = 0
    arena_peak_bytes: int = 0
    arena_bucket_bytes: int = 0
    arena_bucket_count: int = 0
    arena_largest_bucket_bytes: int = 0
    arena_expansion_bytes: int = 0
    arena_budget_exceeded: bool = False
    arena_prepare_miss_count: int = 0
    capture_layout_hot_path_alloc_count: int = 0
    capture_layout_new_count: int = 0
    hot_path_d2h_count: int = 0
    hot_path_cuda_sync_count: int = 0
    tail_path_item_cpu_count: int = 0
    arena_ready_before_tail: bool = False
    arena_prepare_wait_us: float = 0.0
    arena_bind_status: str = ""
    arena_reservation_status: str = ""

    def as_event_fields(self) -> dict[str, object]:
        return {
            "arena_reserved_bytes": int(self.arena_reserved_bytes),
            "arena_peak_bytes": int(self.arena_peak_bytes),
            "arena_bucket_bytes": int(self.arena_bucket_bytes),
            "arena_bucket_count": int(self.arena_bucket_count),
            "arena_largest_bucket_bytes": int(self.arena_largest_bucket_bytes),
            "arena_expansion_bytes": int(self.arena_expansion_bytes),
            "arena_budget_exceeded": bool(self.arena_budget_exceeded),
            "arena_prepare_miss_count": int(self.arena_prepare_miss_count),
            "capture_layout_hot_path_alloc_count": int(self.capture_layout_hot_path_alloc_count),
            "capture_layout_new_count": int(self.capture_layout_new_count),
            "hot_path_d2h_count": int(self.hot_path_d2h_count),
            "hot_path_cuda_sync_count": int(self.hot_path_cuda_sync_count),
            "tail_path_item_cpu_count": int(self.tail_path_item_cpu_count),
            "arena_ready_before_tail": bool(self.arena_ready_before_tail),
            "arena_prepare_wait_us": float(self.arena_prepare_wait_us),
            "arena_bind_status": str(self.arena_bind_status),
            "arena_reservation_status": str(self.arena_reservation_status),
        }


def _dtype_size(dtype: torch.dtype) -> int:
    return int(torch.empty((), dtype=dtype).element_size())


def capture_bucket_bytes(
    *,
    capture_chunk: int,
    slots_cap_bucket: int,
    num_heads: int,
    window: int,
    kv_max_bucket: int,
    capture_dtype: torch.dtype,
    metadata_bytes: int = 0,
) -> CaptureBucketBytes:
    capture_scores_bytes, denom_bytes = _capture_tensor_bytes(
        capture_chunk=capture_chunk,
        slots_cap_bucket=slots_cap_bucket,
        num_heads=num_heads,
        window=window,
        kv_max_bucket=kv_max_bucket,
        capture_element_size=_dtype_size(capture_dtype),
        denom_element_size=_dtype_size(torch.float32),
    )
    return CaptureBucketBytes(
        capture_scores_bytes=int(capture_scores_bytes),
        denom_bytes=int(denom_bytes),
        metadata_bytes=int(metadata_bytes),
    )


@dataclass(slots=True)
class SparseCaptureMetaArena:
    budget_bytes: int = ARENA_PHASE1_QWEN06_BS2_BUDGET_BYTES
    kv_max_cap_bucket: int = 0  # 0 == no explicit cap; otherwise a hard upper
                                # bound (e.g. max_model_len) for the grid-aligned
                                # kv_max, rounded up onto the kv_min grid.
    reservations_by_identity: "OrderedDict[tuple[int, int, int, CaptureArenaIntent], ArenaReservation]" = field(
        default_factory=OrderedDict
    )
    bucket_bytes_by_key: dict[ArenaBucketKey, CaptureBucketBytes] = field(default_factory=dict)
    layouts_by_key: dict[ArenaBucketKey, object] = field(default_factory=dict)
    metrics: ArenaMetrics = field(default_factory=ArenaMetrics)

    def _store_reservation(self, identity, reservation) -> None:
        # Reservation lookup is exact-step scoped and all consumers bind within
        # that step. Retire prior-step metadata when the authoritative identity
        # advances; retain every intent of the current step. This derives the
        # live set from lifecycle rather than an empirical FIFO population cap.
        d = self.reservations_by_identity
        step_identity = tuple(identity[:3])
        stale_identities = [
            existing
            for existing in d
            if tuple(existing[:3]) != step_identity
        ]
        for stale_identity in stale_identities:
            d.pop(stale_identity, None)
        d[identity] = reservation

    def reset_step_metrics(self) -> None:
        reserved_bytes = int(self.metrics.arena_reserved_bytes)
        peak_bytes = max(int(self.metrics.arena_peak_bytes), reserved_bytes)
        largest_bucket = max(
            (record.total_bytes for record in self.bucket_bytes_by_key.values()),
            default=0,
        )
        self.metrics = ArenaMetrics(
            arena_reserved_bytes=reserved_bytes,
            arena_peak_bytes=peak_bytes,
            arena_bucket_count=0,
            arena_largest_bucket_bytes=int(largest_bucket),
            arena_budget_exceeded=(
                reserved_bytes > int(self.budget_bytes)
            ),
        )

    def reset_engine_lifecycle(self) -> None:
        """Retire request/step proofs while retaining generic large buffers."""
        self.reservations_by_identity.clear()
        for layout in self.layouts_by_key.values():
            if not isinstance(layout, StepCaptureLayout):
                continue
            layout.epoch = -1
            layout.step_handle_id = -1
            layout.step_handle_generation = -1
            layout.step_memo_token = None
            layout.prefill_step_layout_memo = None
            layout.slot_row_map_key = None
            layout.live_lengths_key = None
            layout.refresh_payload_views_key = None
            layout.refresh_payload_views_fast_ident = None
            layout.refresh_scores_subviews.clear()
            layout.slot_tensor_cpu = None
            layout.seq_lens_tensor_cpu = None
            layout.seq_lens_cpu = None
            layout.kv_len_per_row_cpu = None
            layout.active_capture_row_by_batch_row_i32 = None
            layout.small_tensor_stage.clear()
        self.reset_step_metrics()

    def missing_reservation(
        self,
        *,
        step_epoch: int,
        step_handle_id: int,
        step_handle_generation: int,
        intent: CaptureArenaIntent,
        reason: str,
        count_miss: bool = True,
    ) -> ArenaReservation:
        reservation = ArenaReservation(
            bucket_key=None,
            step_epoch=int(step_epoch),
            step_handle_id=int(step_handle_id),
            step_handle_generation=int(step_handle_generation),
            intent=intent,
            status=ArenaReservationStatus.MISSING,
            error_reason=str(reason),
            arena_bind_status=ArenaBindStatus.DIAGNOSTIC_FAIL,
        )
        if count_miss:
            self.metrics.arena_prepare_miss_count += 1
        self.metrics.arena_reservation_status = reservation.status.value
        self.metrics.arena_bind_status = reservation.arena_bind_status.value
        return reservation

    def lookup_reservation(
        self,
        *,
        step_epoch: int,
        step_handle_id: int,
        step_handle_generation: int,
        intent: CaptureArenaIntent,
    ) -> ArenaReservation:
        identity = (
            int(step_epoch),
            int(step_handle_id),
            int(step_handle_generation),
            intent,
        )
        reservation = self.reservations_by_identity.get(identity)
        if reservation is None:
            return self.missing_reservation(
                step_epoch=step_epoch,
                step_handle_id=step_handle_id,
                step_handle_generation=step_handle_generation,
                intent=intent,
                reason="reservation_missing",
            )
        self.metrics.arena_reservation_status = reservation.status.value
        self.metrics.arena_bind_status = reservation.arena_bind_status.value
        return reservation

    def reserve_bucket_accounting(
        self,
        *,
        bucket_key: ArenaBucketKey,
        step_epoch: int,
        step_handle_id: int,
        step_handle_generation: int,
        bucket_bytes: CaptureBucketBytes,
    ) -> ArenaReservation:
        if bucket_key.intent is not CaptureArenaIntent.ONE_SHOT_BOOTSTRAP:
            raise ValueError(
                "Phase 1 reserves one_shot_bootstrap buckets only; "
            f"got {bucket_key.intent.value}"
            )

        identity = (
            int(step_epoch),
            int(step_handle_id),
            int(step_handle_generation),
            bucket_key.intent,
        )
        was_new_bucket = bucket_key not in self.bucket_bytes_by_key
        if was_new_bucket:
            self.bucket_bytes_by_key[bucket_key] = bucket_bytes
            self.metrics.arena_expansion_bytes += int(bucket_bytes.total_bytes)
            self.metrics.arena_reserved_bytes += int(bucket_bytes.total_bytes)
        else:
            existing_bytes = self.bucket_bytes_by_key[bucket_key]
            if int(existing_bytes.total_bytes) != int(bucket_bytes.total_bytes):
                raise ValueError("arena bucket key reused with different byte size")

        self.metrics.arena_bucket_bytes = int(bucket_bytes.total_bytes)
        previous = self.reservations_by_identity.get(identity)
        current_bucket_keys = tuple(
            dict.fromkeys(
                tuple(previous.bucket_keys if previous is not None else tuple())
                + (bucket_key,)
            )
        )
        self.metrics.arena_bucket_count = int(len(current_bucket_keys))
        self.metrics.arena_largest_bucket_bytes = max(
            (
                self.bucket_bytes_by_key[key].total_bytes
                for key in current_bucket_keys
                if key in self.bucket_bytes_by_key
            ),
            default=0,
        )
        self.metrics.arena_peak_bytes = max(
            int(self.metrics.arena_peak_bytes),
            int(self.metrics.arena_reserved_bytes),
        )
        self.metrics.arena_budget_exceeded = (
            int(self.metrics.arena_reserved_bytes) > int(self.budget_bytes)
        )

        status = (
            ArenaReservationStatus.FAILED
            if self.metrics.arena_budget_exceeded
            or int(self.metrics.arena_bucket_count) > int(_CAPTURE_IN_FLIGHT)
            else ArenaReservationStatus.READY
        )
        if int(self.metrics.arena_bucket_count) > int(_CAPTURE_IN_FLIGHT):
            self.metrics.arena_budget_exceeded = True
        bind_status = (
            ArenaBindStatus.ARENA_BUDGET_EXCEEDED
            if status is ArenaReservationStatus.FAILED
            else ArenaBindStatus.PREPARED_BIND
        )
        reservation = ArenaReservation(
            bucket_key=bucket_key,
            step_epoch=int(step_epoch),
            step_handle_id=int(step_handle_id),
            step_handle_generation=int(step_handle_generation),
            intent=bucket_key.intent,
            status=status,
            error_reason="arena_budget_exceeded" if status is ArenaReservationStatus.FAILED else "",
            alloc_bytes=int(bucket_bytes.total_bytes) if was_new_bucket else 0,
            prepared_layout_ids=(int(bucket_key.buf_id),),
            bucket_keys=(bucket_key,),
            arena_bind_status=bind_status,
        )
        if previous is not None:
            prepared_layout_ids = tuple(
                sorted(set(tuple(previous.prepared_layout_ids) + (int(bucket_key.buf_id),)))
            )
            reservation = ArenaReservation(
                bucket_key=bucket_key,
                step_epoch=int(step_epoch),
                step_handle_id=int(step_handle_id),
                step_handle_generation=int(step_handle_generation),
                intent=bucket_key.intent,
                status=status,
                error_reason=(
                    "arena_budget_exceeded" if status is ArenaReservationStatus.FAILED else ""
                ),
                alloc_bytes=int(bucket_bytes.total_bytes) if was_new_bucket else 0,
                prepared_layout_ids=prepared_layout_ids,
                bucket_keys=current_bucket_keys,
                arena_bind_status=bind_status,
            )
        self._store_reservation(identity, reservation)
        self.metrics.arena_ready_before_tail = status is ArenaReservationStatus.READY
        self.metrics.arena_reservation_status = reservation.status.value
        self.metrics.arena_bind_status = reservation.arena_bind_status.value
        return reservation

    def reserve_prefill_layouts(
        self,
        *,
        step_epoch: int,
        step_handle_id: int,
        step_handle_generation: int,
        intent: CaptureArenaIntent,
        slot_list: tuple[int, ...],
        row_list: tuple[int, ...],
        batch_size: int,
        num_heads: int,
        kv_needed: int,
        device: torch.device,
        seq_lens_by_row: tuple[int, ...] = tuple(),
        q_lens_by_row: tuple[int, ...] = tuple(),
        context_kv_len_by_row: tuple[int, ...] = tuple(),
        logits_capacity_by_row: tuple[int, ...] = tuple(),
    ) -> ArenaReservation:
        if intent is not CaptureArenaIntent.ONE_SHOT_BOOTSTRAP:
            raise ValueError(
                "Phase 1 reserves one_shot_bootstrap buckets only; "
                f"got {intent.value}"
            )
        if not slot_list or int(num_heads) <= 0 or int(kv_needed) <= 0:
            return self.missing_reservation(
                step_epoch=step_epoch,
                step_handle_id=step_handle_id,
                step_handle_generation=step_handle_generation,
                intent=intent,
                reason="empty_prefill_layout_reservation",
            )

        slots_cap = _align_up_int(max(1, len(slot_list)), 8)
        kv_min = int(_CAPTURE_KV_BUCKET_CACHED)
        if kv_min <= 0:
            kv_min = 256
        kv_min = max(256, int(kv_min))
        _need = max(1, int(kv_needed))
        # Grid-aligned kv_max bucketing on the SAME grain (kv_min ==
        # max(256, _CAPTURE_KV_BUCKET_CACHED)) as the live-build path in
        # refresh_runtime/capture_layout_worker.py (kv_max = _align_up_int(
        # kv_needed, kv_bucket)), so reserve and live-build agree on bucket
        # width (no false grow on the first live refresh). Linear over-alloc
        # (< one grid, vs up to ~2x for next_pow2) keeps a 262145-length
        # context off a 524288-wide buffer that blows the arena budget gate.
        # Downstream (_growfit_reuse / bind smallest-fit / prune high-water)
        # is grain-agnostic (compares kv_max_bucket via >= / argmin), so this
        # is a pure sizing change; kv_max is buffer width only (never a
        # selection/recent/compact threshold).
        kv_max = max(_align_up_int(_need, kv_min), kv_min)
        # Hard cap at the next-aligned(max_model_len) bound exposed via
        # kv_max_cap_bucket (0 == no explicit cap): align the cap onto the
        # same grid so kv_max never exceeds the real model bound rounded up
        # to one grid step.
        _cap = int(getattr(self, "kv_max_cap_bucket", 0))
        if _cap > 0:
            kv_max = max(kv_min, min(kv_max, _align_up_int(_cap, kv_min)))
        # Unified fp16 capture (see capture_layout_worker.py): FA4 stores fp16
        # like FA3/SM80; halves the arena bucket footprint.
        capture_dtype = torch.float16
        prepared_buf_ids: list[int] = []
        current_bucket_keys: list[ArenaBucketKey] = []
        current_bucket_bytes: list[CaptureBucketBytes] = []
        reservation_bytes = 0

        _dtype_str = str(capture_dtype).replace("torch.", "")
        for buf_id in range(int(_CAPTURE_IN_FLIGHT)):
            # _growfit_reuse: reuse an existing high-water buffer that already fits
            # (kv_max_bucket>=needed, slots_cap_bucket>=), like bind() -> no alloc for
            # shorter prompts. Only a longer-than-ever prompt allocates (grow); the
            # superseded buffer is released async-safely by the controller prune.
            reuse_key = None
            for _k, _lay in self.layouts_by_key.items():
                if (
                    isinstance(_lay, StepCaptureLayout)
                    and _k.phase == "prefill"
                    and int(_k.buf_id) == int(buf_id)
                    and _k.intent is intent
                    and int(_k.num_heads) == int(num_heads)
                    and _k.dtype == _dtype_str
                    and _k.device == str(device)
                    and int(_k.kv_max_bucket) >= int(kv_max)
                    and int(_k.slots_cap_bucket) >= int(slots_cap)
                ):
                    if reuse_key is None or int(_k.kv_max_bucket) < int(reuse_key.kv_max_bucket):
                        reuse_key = _k
            if reuse_key is not None:
                key = reuse_key
                byte_record = self.bucket_bytes_by_key[key]
            else:
                key = ArenaBucketKey(
                    phase="prefill",
                    buf_id=int(buf_id),
                    capture_chunk=int(_CAPTURE_CHUNK),
                    slots_cap_bucket=int(slots_cap),
                    num_heads=int(num_heads),
                    window=1,
                    kv_max_bucket=int(kv_max),
                    dtype=_dtype_str,
                    intent=intent,
                    device=str(device),
                )
                byte_record = capture_bucket_bytes(
                    capture_chunk=int(_CAPTURE_CHUNK),
                    slots_cap_bucket=int(slots_cap),
                    num_heads=int(num_heads),
                    window=1,
                    kv_max_bucket=int(kv_max),
                    capture_dtype=capture_dtype,
                )
            current_bucket_keys.append(key)
            current_bucket_bytes.append(byte_record)
            existing = self.layouts_by_key.get(key)
            if (
                key in self.bucket_bytes_by_key
                and int(self.bucket_bytes_by_key[key].total_bytes)
                != int(byte_record.total_bytes)
            ):
                raise ValueError("arena layout key reused with different byte size")
            if not isinstance(existing, StepCaptureLayout):
                reservation_bytes += int(byte_record.total_bytes)
            prepared_buf_ids.append(int(buf_id))

        projected_reserved_bytes = int(self.metrics.arena_reserved_bytes) + int(
            reservation_bytes
        )
        if (
            projected_reserved_bytes > int(self.budget_bytes)
            or len(current_bucket_keys) > int(_CAPTURE_IN_FLIGHT)
        ):
            self.metrics.arena_bucket_count = int(len(current_bucket_keys))
            self.metrics.arena_bucket_bytes = max(
                (int(record.total_bytes) for record in current_bucket_bytes),
                default=0,
            )
            self.metrics.arena_largest_bucket_bytes = int(self.metrics.arena_bucket_bytes)
            self.metrics.arena_budget_exceeded = True
            self.metrics.arena_reservation_status = ArenaReservationStatus.FAILED.value
            self.metrics.arena_bind_status = ArenaBindStatus.ARENA_BUDGET_EXCEEDED.value
            reservation = ArenaReservation(
                bucket_key=None,
                step_epoch=int(step_epoch),
                step_handle_id=int(step_handle_id),
                step_handle_generation=int(step_handle_generation),
                intent=intent,
                status=ArenaReservationStatus.FAILED,
                error_reason="arena_budget_exceeded",
                alloc_bytes=0,
                prepared_layout_ids=tuple(),
                bucket_keys=tuple(current_bucket_keys),
                arena_bind_status=ArenaBindStatus.ARENA_BUDGET_EXCEEDED,
            )
            identity = (
                int(step_epoch),
                int(step_handle_id),
                int(step_handle_generation),
                intent,
            )
            self._store_reservation(identity, reservation)
            self.metrics.arena_ready_before_tail = False
            return reservation

        for key, byte_record in zip(current_bucket_keys, current_bucket_bytes):
            existing = self.layouts_by_key.get(key)
            if not isinstance(existing, StepCaptureLayout):
                # [DETERMINISTIC-CAPTURE-ZERO-INIT 2026-07-03] 同 capture_layout_worker
                # 的 live-build 路径:必须 zeros 不能 empty——FIXED_K 整 bucket 切片把
                # 含 padding 的区间递给 selector,cudaMalloc 垃圾页(共享卡被污染,每
                # run 不同)参与 topk 即输出漂移;denoms(下方 zeros)从未漂与 capture
                # (曾 empty)必漂的判别对是实证铁证。arena 新块低频,一次 memset。
                capture_scores = torch.zeros(
                    (
                        int(_CAPTURE_CHUNK),
                        int(slots_cap),
                        int(num_heads),
                        1,
                        int(kv_max),
                    ),
                    device=device,
                    dtype=capture_dtype,
                )
                log_f_denoms = torch.zeros(
                    (int(_CAPTURE_CHUNK), int(slots_cap), int(num_heads)),
                    device=device,
                    dtype=torch.float32,
                )
                slot_tensor = torch.as_tensor(slot_list, device=device, dtype=torch.long)
                slot_tensor_i32 = torch.as_tensor(slot_list, device=device, dtype=torch.int32)
                row_tensor = torch.as_tensor(row_list, device=device, dtype=torch.long)
                row_tensor_i32 = torch.as_tensor(row_list, device=device, dtype=torch.int32)
                capture_row_by_batch_row_cpu = [-1] * max(0, int(batch_size))
                for capture_row, batch_row in enumerate(row_list):
                    row_i = int(batch_row)
                    if 0 <= row_i < len(capture_row_by_batch_row_cpu):
                        capture_row_by_batch_row_cpu[row_i] = int(capture_row)
                capture_row_by_batch_row_i32 = torch.as_tensor(
                    capture_row_by_batch_row_cpu,
                    device=device,
                    dtype=torch.int32,
                )
                seq_lens_batch = torch.empty((0,), device=device, dtype=torch.long)
                kv_len_per_row_i32 = torch.empty((0,), device=device, dtype=torch.int32)
                kv_lengths = torch.empty((0, int(num_heads)), device=device, dtype=torch.long)
                layout = StepCaptureLayout(
                    epoch=int(step_epoch),
                    step_handle_id=int(step_handle_id),
                    step_handle_generation=int(step_handle_generation),
                    slot_list=list(slot_list),
                    slot_tensor=slot_tensor,
                    slot_tensor_i32=slot_tensor_i32,
                    slot_to_capture_row={int(slot): idx for idx, slot in enumerate(slot_list)},
                    row_tensor=row_tensor,
                    row_tensor_i32=row_tensor_i32,
                    capture_row_by_batch_row_i32=capture_row_by_batch_row_i32,
                    kv_lengths=kv_lengths,
                    kv_len_per_row_i32=kv_len_per_row_i32,
                    chunk_lengths=None,
                    num_heads=int(num_heads),
                    window=1,
                    kv_max=int(kv_max),
                    capture_scores=capture_scores,
                    log_f_denoms=log_f_denoms,
                    row_list_cpu=[int(v) for v in row_list],
                    seq_lens_batch=seq_lens_batch,
                    active_capture_row_by_batch_row_i32=(
                        capture_row_by_batch_row_i32[: max(0, int(batch_size))]
                    ),
                    slot_row_map_key=None,
                    buf_id=int(key.buf_id),
                    small_tensor_stage={},
                )
                self.layouts_by_key[key] = layout
                if key not in self.bucket_bytes_by_key:
                    self.bucket_bytes_by_key[key] = byte_record
                    self.metrics.arena_expansion_bytes += int(byte_record.total_bytes)
                    self.metrics.arena_reserved_bytes += int(byte_record.total_bytes)
            elif key not in self.bucket_bytes_by_key:
                self.bucket_bytes_by_key[key] = byte_record
            layout_for_key = self.layouts_by_key.get(key)
            if isinstance(layout_for_key, StepCaptureLayout):
                self._prepare_prefill_live_metadata(
                    layout=layout_for_key,
                    step_epoch=int(step_epoch),
                    step_handle_id=int(step_handle_id),
                    step_handle_generation=int(step_handle_generation),
                    slot_list=slot_list,
                    row_list=row_list,
                    batch_size=int(batch_size),
                    num_heads=int(num_heads),
                    device=device,
                    seq_lens_by_row=seq_lens_by_row,
                    q_lens_by_row=q_lens_by_row,
                    context_kv_len_by_row=context_kv_len_by_row,
                    logits_capacity_by_row=logits_capacity_by_row,
                )
            self.metrics.arena_bucket_bytes = max(
                int(self.metrics.arena_bucket_bytes),
                int(byte_record.total_bytes),
            )

        self.metrics.arena_bucket_count = int(len(current_bucket_keys))
        self.metrics.arena_largest_bucket_bytes = max(
            (int(record.total_bytes) for record in current_bucket_bytes),
            default=0,
        )
        self.metrics.arena_peak_bytes = max(
            int(self.metrics.arena_peak_bytes),
            int(self.metrics.arena_reserved_bytes),
        )
        self.metrics.arena_budget_exceeded = (
            int(self.metrics.arena_reserved_bytes) > int(self.budget_bytes)
        )
        status = (
            ArenaReservationStatus.FAILED
            if self.metrics.arena_budget_exceeded
            else ArenaReservationStatus.READY
        )
        bind_status = (
            ArenaBindStatus.ARENA_BUDGET_EXCEEDED
            if status is ArenaReservationStatus.FAILED
            else ArenaBindStatus.PREPARED_BIND
        )
        reservation = ArenaReservation(
            bucket_key=None,
            step_epoch=int(step_epoch),
            step_handle_id=int(step_handle_id),
            step_handle_generation=int(step_handle_generation),
            intent=intent,
            status=status,
            error_reason="arena_budget_exceeded" if status is ArenaReservationStatus.FAILED else "",
            # alloc_bytes == bytes of buffers NEWLY built THIS call; 0 == pure reuse.
            # The controller grow-gated prune depends on this contract.
            alloc_bytes=int(reservation_bytes),
            prepared_layout_ids=tuple(prepared_buf_ids),
            bucket_keys=tuple(current_bucket_keys),
            arena_bind_status=bind_status,
        )
        identity = (
            int(step_epoch),
            int(step_handle_id),
            int(step_handle_generation),
            intent,
        )
        self._store_reservation(identity, reservation)
        self.metrics.arena_ready_before_tail = status is ArenaReservationStatus.READY
        self.metrics.arena_reservation_status = reservation.status.value
        self.metrics.arena_bind_status = reservation.arena_bind_status.value
        return reservation

    def _prepare_prefill_live_metadata(
        self,
        *,
        layout: StepCaptureLayout,
        step_epoch: int,
        step_handle_id: int,
        step_handle_generation: int,
        slot_list: tuple[int, ...],
        row_list: tuple[int, ...],
        batch_size: int,
        num_heads: int,
        device: torch.device,
        seq_lens_by_row: tuple[int, ...],
        q_lens_by_row: tuple[int, ...],
        context_kv_len_by_row: tuple[int, ...],
        logits_capacity_by_row: tuple[int, ...],
    ) -> None:
        if not (
            seq_lens_by_row
            and q_lens_by_row
            and context_kv_len_by_row
            and logits_capacity_by_row
        ):
            return
        row_key = tuple(int(row) for row in row_list)
        if not all(
            0 <= int(row) < len(context_kv_len_by_row)
            and 0 <= int(row) < len(logits_capacity_by_row)
            and 0 <= int(row) < len(q_lens_by_row)
            for row in row_key
        ):
            return

        layout.epoch = int(step_epoch)
        layout.step_handle_id = int(step_handle_id)
        layout.step_handle_generation = int(step_handle_generation)
        layout.slot_list = [int(slot) for slot in slot_list]
        layout.slot_to_capture_row = {
            int(slot): idx for idx, slot in enumerate(layout.slot_list)
        }
        layout.slot_tensor = cached_sequence_to_device(
            layout.slot_list,
            dtype=torch.long,
            device=device,
            cache_name="slot_i64",
            stage_cache=layout.small_tensor_stage,
            out=layout.slot_tensor,
        )
        layout.slot_tensor_i32 = cached_sequence_to_device(
            layout.slot_list,
            dtype=torch.int32,
            device=device,
            cache_name="slot_i32",
            stage_cache=layout.small_tensor_stage,
            out=layout.slot_tensor_i32,
        )
        layout.row_tensor = cached_sequence_to_device(
            row_key,
            dtype=torch.long,
            device=device,
            cache_name="row_i64",
            stage_cache=layout.small_tensor_stage,
            out=layout.row_tensor,
        )
        layout.row_tensor_i32 = cached_sequence_to_device(
            row_key,
            dtype=torch.int32,
            device=device,
            cache_name="row_i32",
            stage_cache=layout.small_tensor_stage,
            out=layout.row_tensor_i32,
        )
        layout.row_list_cpu = [int(row) for row in row_key]
        layout.slot_row_map_key = row_key

        capture_row_by_batch_row_cpu = [-1] * max(0, int(batch_size))
        for capture_row, batch_row in enumerate(row_key):
            row_i = int(batch_row)
            if 0 <= row_i < len(capture_row_by_batch_row_cpu):
                capture_row_by_batch_row_cpu[row_i] = int(capture_row)
        layout.capture_row_by_batch_row_i32 = cached_sequence_to_device(
            tuple(capture_row_by_batch_row_cpu),
            dtype=torch.int32,
            device=device,
            cache_name="capture_row_by_batch_row_i32",
            stage_cache=layout.small_tensor_stage,
            out=layout.capture_row_by_batch_row_i32,
        )
        layout.active_capture_row_by_batch_row_i32 = (
            layout.capture_row_by_batch_row_i32[: max(0, int(batch_size))]
        )

        context_kv_cpu = tuple(int(context_kv_len_by_row[row]) for row in row_key)
        kv_len_per_row_cpu = tuple(
            max(1, min(int(context_kv_cpu[idx]), int(logits_capacity_by_row[row])))
            for idx, row in enumerate(row_key)
        )
        seq_lens_cpu = tuple(
            int(seq_lens_by_row[row]) if 0 <= int(row) < len(seq_lens_by_row) else 0
            for row in row_key
        )
        layout.seq_lens_batch = cached_sequence_to_device(
            context_kv_cpu,
            dtype=torch.long,
            device=device,
            cache_name="live_seq_full_i64",
            stage_cache=layout.small_tensor_stage,
            out=layout.seq_lens_batch,
        )
        layout.seq_lens_batch_i32 = cached_sequence_to_device(
            context_kv_cpu,
            dtype=torch.int32,
            device=device,
            cache_name="live_seq_full_i32",
            stage_cache=layout.small_tensor_stage,
            out=layout.seq_lens_batch_i32,
        )
        kv_len = cached_sequence_to_device(
            kv_len_per_row_cpu,
            dtype=torch.long,
            device=device,
            cache_name="live_kv_len_i64",
            stage_cache=layout.small_tensor_stage,
        )
        layout.seq_lens_cpu = seq_lens_cpu
        layout.kv_len_per_row_cpu = kv_len_per_row_cpu
        layout.seq_lens_tensor_cpu = None
        layout.kv_len_per_row_i32 = cached_sequence_to_device(
            kv_len_per_row_cpu,
            dtype=torch.int32,
            device=device,
            cache_name="live_kv_len_i32",
            stage_cache=layout.small_tensor_stage,
            out=layout.kv_len_per_row_i32,
        )
        layout.kv_lengths = kv_len.unsqueeze(1).expand(-1, int(num_heads))
        layout.chunk_lengths = cached_sequence_to_device(
            tuple(int(q_lens_by_row[row]) for row in row_key),
            dtype=torch.long,
            device=device,
            cache_name="live_chunk_i64",
            stage_cache=layout.small_tensor_stage,
            out=layout.chunk_lengths,
        )
        layout.live_lengths_key = (
            int(step_epoch),
            int(step_handle_id),
            int(step_handle_generation),
            int(len(row_key)),
            row_key,
            tuple(int(v) for v in seq_lens_by_row),
            tuple(int(v) for v in context_kv_len_by_row),
            tuple(int(v) for v in logits_capacity_by_row),
            int(num_heads),
            ("cpu_context_kv",),
            ("cpu_plan_cap",),
            ("cpu_q_lens", tuple(int(v) for v in q_lens_by_row)),
        )

    def bind_prepared_prefill_capture_layout(
        self,
        *,
        reservation: ArenaReservation,
        phase: str,
        buf_id: int,
        step_epoch: int,
        step_handle_id: int,
        step_handle_generation: int,
        slot_list: tuple[int, ...],
        num_heads: int,
        kv_needed: int,
        device: torch.device,
    ) -> Optional[StepCaptureLayout]:
        if phase != "prefill":
            self.metrics.arena_prepare_miss_count += 1
            self.metrics.arena_bind_status = ArenaBindStatus.DIAGNOSTIC_FAIL.value
            return None
        if not reservation.is_ready_for(
            step_epoch=step_epoch,
            step_handle_id=step_handle_id,
            step_handle_generation=step_handle_generation,
            intent=CaptureArenaIntent.ONE_SHOT_BOOTSTRAP,
        ):
            self.metrics.arena_prepare_miss_count += 1
            self.metrics.arena_ready_before_tail = False
            self.metrics.arena_bind_status = ArenaBindStatus.SYNC_EXPANSION_MISS.value
            self.metrics.arena_reservation_status = reservation.status.value
            return None
        # Single-pass smallest-fit (no intermediate list, no per-candidate tensor
        # touch: key.device == str(device) at insert time). Candidate set is bounded
        # to <=_CAPTURE_IN_FLIGHT by the grow-to-fit prune.
        want_device = str(device)
        slots_needed = len(slot_list)
        key = None
        layout = None
        best_rank = None
        for _k, _lay in self.layouts_by_key.items():
            if not (
                isinstance(_lay, StepCaptureLayout)
                and _k.phase == "prefill"
                and int(_k.buf_id) == int(buf_id)
                and _k.intent is CaptureArenaIntent.ONE_SHOT_BOOTSTRAP
                and int(_k.num_heads) == int(num_heads)
                and int(_k.kv_max_bucket) >= int(kv_needed)
                and int(_k.slots_cap_bucket) >= slots_needed
                and _k.device == want_device
            ):
                continue
            rank = (int(_k.kv_max_bucket), int(_k.slots_cap_bucket))
            if best_rank is None or rank < best_rank:
                best_rank = rank
                key = _k
                layout = _lay
        if key is None:
            self.metrics.arena_prepare_miss_count += 1
            self.metrics.arena_ready_before_tail = False
            self.metrics.arena_bind_status = ArenaBindStatus.SYNC_EXPANSION_MISS.value
            self.metrics.arena_reservation_status = reservation.status.value
            return None
        layout.buf_id = int(buf_id)
        layout.num_heads = int(num_heads)
        layout.kv_max = int(key.kv_max_bucket)
        self.metrics.arena_ready_before_tail = True
        self.metrics.arena_bind_status = ArenaBindStatus.PREPARED_BIND.value
        self.metrics.arena_reservation_status = reservation.status.value
        return layout
