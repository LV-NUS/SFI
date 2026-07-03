from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, Optional, Set, Tuple


class LeaseKind(Enum):
    KEY_NORMS = auto()
    COMPACT = auto()
    CAPTURE_RING = auto()


class LeaseState(Enum):
    ACTIVE = auto()
    RETIRED = auto()


@dataclass(frozen=True, slots=True)
class BufferLease:
    kind: LeaseKind
    slot: int
    generation: int
    capacity: int
    epoch: int


@dataclass(slots=True)
class _LeaseRecord:
    generation: int
    capacity: int
    epoch: int
    state: LeaseState = LeaseState.ACTIVE
    retire_event_id: Optional[str] = None


class BufferLeaseRegistry:
    """Small lifecycle registry for buffer reuse contracts."""

    def __init__(self, num_slots: int) -> None:
        n = int(num_slots)
        if n <= 0:
            raise ValueError(f"num_slots must be > 0, got {n}")
        self._num_slots = n
        self._records: Dict[Tuple[LeaseKind, int], _LeaseRecord] = {}

    def acquire(
        self,
        *,
        kind: LeaseKind,
        slot: int,
        min_capacity: int,
        epoch: int,
    ) -> BufferLease:
        s = int(slot) % self._num_slots
        cap = max(1, int(min_capacity))
        key = (kind, s)
        old = self._records.get(key)
        if old is not None and old.state == LeaseState.ACTIVE:
            if cap > old.capacity:
                old.capacity = cap
            old.epoch = int(epoch)
            return BufferLease(
                kind=kind,
                slot=s,
                generation=int(old.generation),
                capacity=int(old.capacity),
                epoch=int(old.epoch),
            )

        generation = 0 if old is None else int(old.generation) + 1
        rec = _LeaseRecord(
            generation=generation,
            capacity=cap if old is None else max(cap, int(old.capacity)),
            epoch=int(epoch),
            state=LeaseState.ACTIVE,
            retire_event_id=None,
        )
        self._records[key] = rec
        return BufferLease(
            kind=kind,
            slot=s,
            generation=int(rec.generation),
            capacity=int(rec.capacity),
            epoch=int(rec.epoch),
        )

    def retire(self, *, lease: BufferLease, event_id: str) -> None:
        key = (lease.kind, int(lease.slot) % self._num_slots)
        rec = self._records.get(key)
        if rec is None:
            raise RuntimeError("retire called for unknown lease")
        if int(rec.generation) != int(lease.generation):
            raise RuntimeError(
                f"lease generation mismatch: registry={rec.generation} lease={lease.generation}"
            )
        if rec.state != LeaseState.ACTIVE:
            raise RuntimeError("retire called for non-active lease")
        rec.state = LeaseState.RETIRED
        rec.retire_event_id = str(event_id)

    def reclaim(self, *, ready_event_ids: Set[str]) -> int:
        if not ready_event_ids:
            return 0
        ready = set(str(eid) for eid in ready_event_ids)
        reclaimed = 0
        for rec in self._records.values():
            if rec.state != LeaseState.RETIRED:
                continue
            if rec.retire_event_id not in ready:
                continue
            rec.state = LeaseState.ACTIVE
            rec.retire_event_id = None
            reclaimed += 1
        return reclaimed

    def pending_retired(self) -> int:
        return sum(1 for rec in self._records.values() if rec.state == LeaseState.RETIRED)

