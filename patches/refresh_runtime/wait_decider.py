from __future__ import annotations


def should_wait_when_pending(*, has_pending: bool, has_blockers: bool) -> bool:
    return bool(has_pending or has_blockers)

