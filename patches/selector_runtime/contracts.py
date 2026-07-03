from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SelectorDecision:
    selector_enabled: bool
    resize_needed: bool
    shape_signature: tuple[object, ...]
