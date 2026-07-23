"""CPU-only compact_recent rail routing decisions."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Sequence


class CompactRecentRailMode(Enum):
    """Host-side compact_recent rail mode for one decode step.

    Rev 2: MIXED_ACTIVE_WORKLIST removed. Merged into ALL_ACTIVE_DIRECT.
    """

    NO_COMPACT = auto()
    ALL_ACTIVE_DIRECT = auto()
    DECODE_CAPTURE = auto()


@dataclass(frozen=True)
class CompactRecentRailDecision:
    mode: CompactRecentRailMode
    batch_size: int
    compact_rows: tuple[int, ...]
    inactive_rows: tuple[int, ...]
    capture_rows: tuple[int, ...]
    compact_rail_only: bool
    requires_active_worklist: bool
    requires_decode_capture: bool


def _exact_bool(
    values: Sequence[object],
    rows: int,
    label: str,
) -> tuple[bool, ...]:
    if len(values) != rows:
        raise ValueError(
            f"{label} must exactly match batch_size: {len(values)} != {rows}"
        )
    if isinstance(values, tuple):
        return values
    return tuple(bool(values[idx]) for idx in range(rows))


def _exact_int(
    values: Sequence[object],
    rows: int,
    label: str,
) -> tuple[int, ...]:
    if len(values) != rows:
        raise ValueError(
            f"{label} must exactly match batch_size: {len(values)} != {rows}"
        )
    if isinstance(values, tuple):
        return values
    return tuple(int(values[idx]) for idx in range(rows))


def resolve_compact_recent_rail_mode(step_authority: object) -> CompactRecentRailDecision:
    """Resolve compact_recent rail ownership from StepAuthority CPU fields only.

    This function intentionally does not import torch and must not inspect GPU
    tensors. It is the host route fact used before launching compact_recent.
    """
    raw_use_compact = step_authority.use_compact_by_row
    batch_size = int(step_authority.batch_size)
    if batch_size < 0:
        raise ValueError("batch_size must be non-negative")
    use_compact = _exact_bool(
        raw_use_compact,
        batch_size,
        "use_compact_by_row",
    )
    needs_logits = _exact_bool(
        step_authority.needs_logits_by_row,
        batch_size,
        "needs_logits_by_row",
    )
    logits_last_n = _exact_int(
        step_authority.logits_last_n_by_row,
        batch_size,
        "logits_last_n_by_row",
    )
    is_prefill = _exact_bool(
        step_authority.is_prefill_by_row,
        batch_size,
        "is_prefill_by_row",
    )

    decode_capture_rows = tuple(
        idx
        for idx in range(batch_size)
        if (
            needs_logits[idx]
            and logits_last_n[idx] > 0
            and not is_prefill[idx]
        )
    )

    # Decode capture is owned by compact_recent, but it must observe the full
    # KV domain.  Treat those rows as full-recent rows in the launch plan even
    # when the normal decode path would have consumed compact+recent.
    decode_capture_row_set = set(decode_capture_rows)
    compact_rows = tuple(
        idx
        for idx, enabled in enumerate(use_compact)
        if enabled and not is_prefill[idx] and idx not in decode_capture_row_set
    )
    compact_row_set = set(compact_rows)
    inactive_rows = tuple(idx for idx in range(batch_size) if idx not in compact_row_set)

    # Rev 2: merged MIXED_ACTIVE_WORKLIST into ALL_ACTIVE_DIRECT. Pure decode
    # with mixed compact/non-compact rows handled by compact_recent kernel
    # via per-row compact_valid_tokens_i32=0 degenerate recent-only path, not
    # active_worklist. active_worklist is only enabled when this rail is
    # co-launched with a prefill rail (caller uses force_active_worklist=True).
    if decode_capture_rows:
        mode = CompactRecentRailMode.DECODE_CAPTURE
    elif compact_rows:
        mode = CompactRecentRailMode.ALL_ACTIVE_DIRECT
    else:
        mode = CompactRecentRailMode.NO_COMPACT

    return CompactRecentRailDecision(
        mode=mode,
        batch_size=batch_size,
        compact_rows=compact_rows,
        inactive_rows=inactive_rows,
        capture_rows=decode_capture_rows,
        compact_rail_only=True,
        # Rev 2: default False. Only caller (mixed prefill+decode dispatcher)
        # sets force_active_worklist=True when co-launching with prefill rail.
        requires_active_worklist=False,
        requires_decode_capture=mode is CompactRecentRailMode.DECODE_CAPTURE,
    )
