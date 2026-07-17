from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import Any, Optional


_SHA256_RE = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class CaptureCohortEntry:
    job: Any
    global_layer_index: int
    scratch_slot: int


@dataclass(frozen=True, slots=True)
class ReadyCaptureCohort:
    owner_key: tuple[int, int, int]
    plan_signature: str
    chunk_id: int
    cohort_size: int
    selected_depth: int
    entries: tuple[CaptureCohortEntry, ...]

    @property
    def jobs(self) -> tuple[Any, ...]:
        return tuple(entry.job for entry in self.entries)

    @property
    def scratch_slots(self) -> tuple[int, ...]:
        return tuple(entry.scratch_slot for entry in self.entries)


@dataclass(slots=True)
class _PendingCaptureCohort:
    owner_key: tuple[int, int, int]
    plan_signature: str
    chunk_id: int
    cohort_size: int
    selected_depth: int
    entries: list[CaptureCohortEntry]


class CaptureCohortCoordinator:
    """Own one strictly ordered capture cohort until its launch boundary."""

    __slots__ = ("_lock", "_pending")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: Optional[_PendingCaptureCohort] = None

    def submit(
        self,
        *,
        job: Any,
        handle_id: int,
        handle_generation: int,
        epoch: int,
        plan_signature: str,
        global_layer_index: int,
        total_layers: int,
        chunk_id: int,
        slot_in_chunk: int,
        scratch_slot: int,
        cohort_size: int,
        selected_depth: int,
    ) -> Optional[ReadyCaptureCohort]:
        values = {
            "handle_id": handle_id,
            "handle_generation": handle_generation,
            "epoch": epoch,
            "global_layer_index": global_layer_index,
            "total_layers": total_layers,
            "chunk_id": chunk_id,
            "slot_in_chunk": slot_in_chunk,
            "scratch_slot": scratch_slot,
            "cohort_size": cohort_size,
            "selected_depth": selected_depth,
        }
        if any(type(value) is not int for value in values.values()):
            raise TypeError("capture cohort integer fields must be exact ints")
        if (
            handle_id < 0
            or handle_generation < 0
            or epoch < 0
            or global_layer_index < 0
            or total_layers <= 0
            or chunk_id < 0
            or cohort_size <= 0
            or selected_depth <= 0
        ):
            raise ValueError("capture cohort identity and geometry must be positive")
        if global_layer_index >= total_layers:
            raise ValueError("capture cohort layer exceeds the model layer count")
        if selected_depth % cohort_size != 0:
            raise ValueError("capture cohort depth must contain whole cohorts")
        if chunk_id != global_layer_index // cohort_size:
            raise ValueError("capture cohort chunk does not match the global layer")
        if slot_in_chunk != global_layer_index % cohort_size:
            raise ValueError("capture cohort slot does not match the global layer")
        if not 0 <= scratch_slot < selected_depth:
            raise ValueError("capture cohort scratch slot exceeds the selected depth")
        if scratch_slot % cohort_size != slot_in_chunk:
            raise ValueError("capture cohort scratch slot is in the wrong cohort lane")
        if not isinstance(plan_signature, str) or _SHA256_RE.fullmatch(
            plan_signature
        ) is None:
            raise ValueError("capture cohort plan signature must be lowercase sha256")
        if job is None:
            raise ValueError("capture cohort job is required")

        owner_key = (handle_id, handle_generation, epoch)
        expected_job_key = (handle_id, handle_generation, global_layer_index)
        if getattr(job, "job_key", None) != expected_job_key:
            raise ValueError("capture cohort job identity does not match its owner")
        if int(getattr(job, "debug_epoch", -1)) != epoch:
            raise ValueError("capture cohort job epoch does not match its owner")

        entry = CaptureCohortEntry(
            job=job,
            global_layer_index=global_layer_index,
            scratch_slot=scratch_slot,
        )
        with self._lock:
            pending = self._pending
            if pending is None:
                if slot_in_chunk != 0:
                    raise RuntimeError(
                        "capture cohort cannot start after the first chunk slot"
                    )
                pending = _PendingCaptureCohort(
                    owner_key=owner_key,
                    plan_signature=plan_signature,
                    chunk_id=chunk_id,
                    cohort_size=cohort_size,
                    selected_depth=selected_depth,
                    entries=[],
                )
                self._pending = pending
            else:
                pending_identity = (
                    pending.owner_key,
                    pending.plan_signature,
                    pending.chunk_id,
                    pending.cohort_size,
                    pending.selected_depth,
                )
                submitted_identity = (
                    owner_key,
                    plan_signature,
                    chunk_id,
                    cohort_size,
                    selected_depth,
                )
                if pending_identity != submitted_identity:
                    raise RuntimeError(
                        "capture cohort owner or immutable plan changed before launch"
                    )
            expected_slot = len(pending.entries)
            if slot_in_chunk != expected_slot:
                raise RuntimeError("capture cohort layers must arrive exactly once in order")
            if any(entry.job is job for entry in pending.entries):
                raise RuntimeError("capture cohort job was submitted more than once")
            pending.entries.append(entry)

            is_chunk_tail = slot_in_chunk == cohort_size - 1
            is_model_tail = global_layer_index == total_layers - 1
            if not (is_chunk_tail or is_model_tail):
                return None

            ready = ReadyCaptureCohort(
                owner_key=pending.owner_key,
                plan_signature=pending.plan_signature,
                chunk_id=pending.chunk_id,
                cohort_size=pending.cohort_size,
                selected_depth=pending.selected_depth,
                entries=tuple(pending.entries),
            )
            self._pending = None
            return ready

    def assert_idle(self) -> None:
        with self._lock:
            if self._pending is not None:
                raise RuntimeError("capture cohort still owns unlaunched jobs")

    @property
    def pending_count(self) -> int:
        with self._lock:
            return 0 if self._pending is None else len(self._pending.entries)


def validate_capture_cohort_completion(
    cohort: ReadyCaptureCohort,
    *,
    ran_count: int,
    terminal_event: Any,
) -> Any:
    """Validate and return the ordered cohort's terminal publication token."""

    if not isinstance(cohort, ReadyCaptureCohort):
        raise TypeError("capture cohort completion requires a ready cohort")
    if (
        type(ran_count) is not int
        or not cohort.entries
        or ran_count != len(cohort.entries)
    ):
        raise RuntimeError(
            "E_SFI_CAPTURE_COHORT_LAUNCH: strict tiled owner did not consume "
            "every job"
        )
    if any(
        not bool(getattr(entry.job, "completed", False))
        or not bool(getattr(entry.job, "ran_postprocess", False))
        or getattr(entry.job, "completion_event", None) is None
        for entry in cohort.entries
    ):
        raise RuntimeError(
            "E_SFI_CAPTURE_COHORT_LIFECYCLE: cohort jobs were not completed "
            "by the strict owner"
        )
    if terminal_event is None or getattr(
        cohort.entries[-1].job, "completion_event", None
    ) is not terminal_event:
        raise RuntimeError(
            "E_SFI_CAPTURE_COHORT_TERMINAL_EVENT: strict owner must publish "
            "the final ordered job event"
        )
    return terminal_event


def publish_capture_cohort_completion(
    cohort: ReadyCaptureCohort,
    *,
    ran_count: int,
    terminal_event: Any,
    fence: Any,
) -> Any:
    """Publish one ordered terminal event to every cohort scratch slot."""

    event = validate_capture_cohort_completion(
        cohort,
        ran_count=ran_count,
        terminal_event=terminal_event,
    )
    on_reduce = getattr(fence, "on_reduce", None)
    if not callable(on_reduce):
        raise RuntimeError(
            "E_SFI_CAPTURE_COHORT_FENCE: reusable scratch has no WAR fence owner"
        )
    for scratch_slot in cohort.scratch_slots:
        on_reduce(int(scratch_slot), event, True)
    return event
