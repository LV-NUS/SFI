from __future__ import annotations

from dataclasses import dataclass


REASON_PENDING_OR_BLOCKERS = "pending_or_blockers"
REASON_NO_PENDING_NO_BLOCKERS = "no_pending_no_blockers"


@dataclass(frozen=True, slots=True)
class RefreshDecision:
    need_wait: bool
    reason: str
    refresh_slots: tuple[int, ...]

    @property
    def wait_reason(self) -> str:
        return self.reason
