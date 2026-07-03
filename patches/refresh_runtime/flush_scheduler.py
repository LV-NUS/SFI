from __future__ import annotations

from typing import Iterable


def normalize_refresh_slot_list(slots: Iterable[int]) -> tuple[int, ...]:
    return tuple(sorted({int(slot) for slot in slots}))

