"""Pure, unit-testable host-side logic for the per-G capture RING + WAR fence (FIX-1/FIX-2)."""
from __future__ import annotations

__all__ = ["validate_reduce_group", "ring_depth", "ring_scratch_slot", "tape_slot", "RingWarFence"]


def validate_reduce_group(reduce_group: int, capture_chunk: int) -> int:
    g = int(reduce_group)
    if g not in (0, 1, 2, 4):
        return 0
    if g > 0 and (int(capture_chunk) % g) != 0:
        return 0
    return g


def ring_depth(reduce_group: int, in_flight: int) -> int:
    g = int(reduce_group)
    return g * int(in_flight) if g > 0 else 0


def ring_scratch_slot(global_layer_index: int, reduce_group: int, in_flight: int) -> int:
    g = int(reduce_group)
    if g <= 0:
        return -1
    return int(global_layer_index) % (g * int(in_flight))


def tape_slot(global_layer_index: int, capture_chunk: int) -> int:
    return int(global_layer_index) % int(capture_chunk)


class RingWarFence:
    __slots__ = ("_evts",)

    def __init__(self) -> None:
        self._evts = {}

    def war_event_before_capture(self, ring_slot: int):
        s = int(ring_slot)
        prev = self._evts.get(s)
        self._evts[s] = None
        return prev

    def on_reduce(self, ring_slot: int, event, ran_postprocess: bool) -> None:
        if bool(ran_postprocess):
            self._evts[int(ring_slot)] = event

    def reset(self) -> None:
        self._evts.clear()
