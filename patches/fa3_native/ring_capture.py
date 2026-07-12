"""Pure, unit-testable host-side logic for the per-G capture RING + WAR fence (FIX-1/FIX-2)."""
from __future__ import annotations

# [REDUCE-GROUP-SINGLE-SOURCE 2026-07-11 EXT审计·随手批] 本文件旧携第二份
# validate_reduce_group（静默 return 0 = fallback 形态），与生产（sparse_
# constants 的 raise 判定）语义漂移 = 双真源。真源收敛到 runtime_contracts
# （纯 stdlib 合同件），此处 re-export 保 __all__/调用面不变。
from patches.runtime_contracts import validate_reduce_group

__all__ = ["validate_reduce_group", "ring_depth", "ring_scratch_slot", "tape_slot", "RingWarFence"]


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
