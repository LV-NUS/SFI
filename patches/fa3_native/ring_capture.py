"""Pure, unit-testable host-side logic for the per-G capture RING + WAR fence (FIX-1/FIX-2)."""
from __future__ import annotations

# Re-export the cold configuration validator; live slot arithmetic stays a
# branch-free expression over already validated geometry.
from patches.runtime_contracts import validate_reduce_group

__all__ = ["validate_reduce_group", "ring_depth", "ring_scratch_slot", "RingWarFence"]


def ring_depth(reduce_group: int, in_flight: int) -> int:
    return int(reduce_group) * int(in_flight)


def ring_scratch_slot(global_layer_index: int, reduce_group: int, in_flight: int) -> int:
    return int(global_layer_index) % (int(reduce_group) * int(in_flight))


class RingWarFence:
    __slots__ = ("_events",)

    def __init__(self, capacity: int) -> None:
        capacity_i = int(capacity)
        if capacity_i <= 0:
            raise ValueError("ring WAR fence capacity must be positive")
        self._events = [None] * capacity_i

    @property
    def capacity(self) -> int:
        return len(self._events)

    def war_event_before_capture(self, ring_slot: int):
        s = int(ring_slot)
        prev = self._events[s]
        self._events[s] = None
        return prev

    def on_reduce(self, ring_slot: int, event, ran_postprocess: bool) -> None:
        if bool(ran_postprocess):
            self._events[int(ring_slot)] = event

    def reset(self) -> None:
        for index in range(len(self._events)):
            self._events[index] = None
