from __future__ import annotations

from typing import Iterable

from patches.refresh_runtime.contracts import (
    REASON_NO_PENDING_NO_BLOCKERS,
    REASON_PENDING_OR_BLOCKERS,
    RefreshDecision,
)
from patches.refresh_runtime.flush_scheduler import normalize_refresh_slot_list
from patches.refresh_runtime.wait_decider import should_wait_when_pending


def run_refresh_step(
    *,
    has_pending: bool,
    has_blockers: bool,
    refresh_slots: Iterable[int],
) -> RefreshDecision:
    need_wait = should_wait_when_pending(
        has_pending=has_pending,
        has_blockers=has_blockers,
    )
    reason = REASON_PENDING_OR_BLOCKERS if need_wait else REASON_NO_PENDING_NO_BLOCKERS
    normalized_slots = normalize_refresh_slot_list(refresh_slots)
    return RefreshDecision(
        need_wait=need_wait,
        reason=reason,
        refresh_slots=normalized_slots,
    )
