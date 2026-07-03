from __future__ import annotations

from typing import Any

from patches.runtime_state import StepRuntimeState


def _ensure_runtime_state(controller: Any) -> StepRuntimeState:
    runtime_state = getattr(controller, "_runtime_state", None)
    if runtime_state is None:
        runtime_state = StepRuntimeState()
        controller._runtime_state = runtime_state
    return runtime_state


def reset_runtime_plan_state(controller: Any, *, epoch: int) -> None:
    runtime_state = _ensure_runtime_state(controller)
    runtime_state.reset_ordered_layer_plan(epoch=epoch)


def reset_runtime_step_cursor_state(controller: Any, *, epoch: int) -> None:
    runtime_state = _ensure_runtime_state(controller)
    runtime_state.reset_ordered_layer_cursor_for_step(epoch=epoch)


def apply_ordered_layer_plan_state(
    controller: Any,
    *,
    epoch: int,
    layer_count: int,
    batch_size: int,
    cache_key: tuple[object, ...] | None,
) -> None:
    runtime_state = _ensure_runtime_state(controller)
    runtime_state.apply_ordered_layer_plan(
        epoch=epoch,
        layer_count=layer_count,
        batch_size=batch_size,
        cache_key=cache_key,
    )


def apply_reuse_ordered_plan_state(
    controller: Any, *, epoch: int, batch_size: int
) -> None:
    plan = controller.step_dispatch_plan
    layer_data_list = plan.layer_data_list_ordered if plan else None
    layer_count = len(layer_data_list) if layer_data_list is not None else 0

    apply_ordered_layer_plan_state(
        controller,
        epoch=epoch,
        layer_count=layer_count,
        batch_size=batch_size,
        cache_key=getattr(plan, "cache_key", None),
    )


