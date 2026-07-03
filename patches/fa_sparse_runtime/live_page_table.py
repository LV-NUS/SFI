"""Host-side live page-row derivation for the Leg-B in-place boundary writer.
Pure Python: compact pages come from the residency lease (passed in), recent pages
from the live block table. Mirrors the heavy-bind derivation
(resolved_row_ptr_arena.py:1895-1924) without rebuilding StepBoundMeta / the launch
plan. No torch, no GPU."""
from __future__ import annotations
from typing import Iterable, Mapping, Sequence


def derive_live_page_rows(
    *,
    active_rows: Iterable[int],
    block_table_cpu: Sequence[Sequence[int]],
    compact_pages_by_row: Mapping[int, tuple[int, ...]],
    recent_first_page_by_row: Mapping[int, int],
    recent_page_count_by_row: Mapping[int, int],
) -> dict[int, tuple[int, ...]]:
    out: dict[int, tuple[int, ...]] = {}
    for row in active_rows:
        compact = tuple(int(p) for p in compact_pages_by_row.get(row, ()))
        first = int(recent_first_page_by_row[row])
        count = int(recent_page_count_by_row[row])
        table_row = block_table_cpu[row]
        if first < 0 or first + count > len(table_row):
            raise ValueError(
                f"recent slice [{first}:{first+count}] out of range for row {row}"
            )
        recent = tuple(int(table_row[first + i]) for i in range(count))
        out[int(row)] = compact + recent
    return out
