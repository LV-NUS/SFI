from __future__ import annotations

from patches.selector_runtime.buffer_gate import should_resize_for_batch
from patches.selector_runtime.contracts import SelectorDecision
from patches.selector_runtime.pipeline import should_use_selector_path


def run_selector_step(
    *,
    enabled: bool,
    force_dense: bool,
    capacity: int,
    batch_size: int,
    shape_signature: tuple[object, ...],
) -> SelectorDecision:
    selector_enabled = should_use_selector_path(enabled=enabled, force_dense=force_dense)
    resize_needed = should_resize_for_batch(capacity=capacity, batch_size=batch_size)
    return SelectorDecision(
        selector_enabled=selector_enabled,
        resize_needed=resize_needed,
        shape_signature=tuple(shape_signature),
    )
