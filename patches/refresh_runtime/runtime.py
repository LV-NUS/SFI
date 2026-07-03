from __future__ import annotations


class RefreshRuntime:
    def __init__(self) -> None:
        self.pending_refresh_nonempty_count: int = 0

