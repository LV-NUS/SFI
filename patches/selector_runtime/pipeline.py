from __future__ import annotations


def should_use_selector_path(*, enabled: bool, force_dense: bool) -> bool:
    return bool(enabled and (not force_dense))

