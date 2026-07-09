from __future__ import annotations

import heapq
from typing import Dict, Optional, Tuple


def _validate_request_id(request_id: str) -> str:
    if not isinstance(request_id, str):
        raise TypeError(f"request_id must be str, got {type(request_id).__name__}")
    rid = request_id.strip()
    if not rid:
        raise ValueError("request_id must be non-empty")
    return rid


class GlobalSlotAllocator:
    """Controller 级 request->slot 稳定分配器（带 generation 版本号）。"""

    def __init__(self, capacity: Optional[int] = None) -> None:
        if capacity is not None:
            if isinstance(capacity, bool) or not isinstance(capacity, int):
                raise TypeError(f"capacity must be int or None, got {type(capacity).__name__}")
            if capacity <= 0:
                raise ValueError(f"capacity must be > 0, got {capacity}")
        self._capacity: Optional[int] = int(capacity) if capacity is not None else None
        self._request_to_slot: Dict[str, int] = {}
        self._slot_to_request: Dict[int, str] = {}
        self._free_slots: list[int] = []
        self._next_slot: int = 0
        self._slot_generation: Dict[int, int] = {}

    def _allocate_slot(self) -> int:
        if self._free_slots:
            return int(heapq.heappop(self._free_slots))
        if self._capacity is not None and self._next_slot >= self._capacity:
            raise RuntimeError(
                f"global slot allocator capacity exceeded: "
                f"next_slot={self._next_slot}, capacity={self._capacity}. "
                "Config root cause: the step batch carries more live sparse "
                "requests than max_live_sparse_slots — raise it to >= the "
                "deployment's real concurrency (serve: --max-num-seqs; the "
                "compact lease grows by slots x blocks x 16 x KV-bytes/token "
                "x gen_count), or cap scheduler concurrency to the slot "
                "budget. The startup preflight logged "
                "W_SPARSE_SLOTS_LT_MAX_NUM_SEQS when this hazard was present."
            )
        slot = int(self._next_slot)
        self._next_slot += 1
        return slot

    def acquire(self, request_id: str) -> int:
        rid = _validate_request_id(request_id)
        slot = self._request_to_slot.get(rid)
        if slot is not None:
            return int(slot)
        slot = self._allocate_slot()
        self._request_to_slot[rid] = int(slot)
        self._slot_to_request[int(slot)] = rid
        self._slot_generation.setdefault(int(slot), 0)
        return int(slot)

    def acquire_with_generation(self, request_id: str) -> Tuple[int, int]:
        slot = self.acquire(request_id)
        return int(slot), int(self._slot_generation.get(int(slot), 0))

    def release(self, request_id: str) -> Optional[int]:
        rid = _validate_request_id(request_id)
        slot = self._request_to_slot.pop(rid, None)
        if slot is None:
            return None
        owner = self._slot_to_request.get(int(slot))
        if owner == rid:
            del self._slot_to_request[int(slot)]
        self._slot_generation[int(slot)] = int(self._slot_generation.get(int(slot), 0)) + 1
        heapq.heappush(self._free_slots, int(slot))
        return int(slot)

    def slot_of(self, request_id: str) -> Optional[int]:
        rid = _validate_request_id(request_id)
        slot = self._request_to_slot.get(rid)
        return int(slot) if slot is not None else None

    def generation_of_slot(self, slot: int) -> int:
        if isinstance(slot, bool) or not isinstance(slot, int):
            raise TypeError(f"slot must be int, got {type(slot).__name__}")
        if slot < 0:
            raise ValueError(f"slot must be >= 0, got {slot}")
        return int(self._slot_generation.get(int(slot), 0))
