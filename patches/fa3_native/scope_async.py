from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .contracts import TargetSelectedScopeKey
from .snapshot_binding import ScopeWaitHandle, make_scope_wait_handle


class NotReadyError(RuntimeError):
    pass












def allocate_scope_wait_handle(
    target_selected_scope_key: TargetSelectedScopeKey,
    expected_layers: Iterable[int] | None = None,
) -> ScopeWaitHandle:
    return make_scope_wait_handle(
        target_selected_scope_key.consumer_step_id,
        target_selected_scope_key.layer_group_id,
        expected_layers=tuple(expected_layers or ()),
    )






def mark_layer_commit_terminal(
    handle: ScopeWaitHandle,
    *,
    layer_id: int,
    status: str,
) -> ScopeWaitHandle:
    handle.terminal_layer_status[int(layer_id)] = str(status)
    if handle.expected_layers and all(
        layer in handle.terminal_layer_status for layer in handle.expected_layers
    ):
        handle.is_ready = True
    return handle


def consume_selected_scope(snapshot: object) -> str:
    handle = getattr(snapshot, "consume_selected_scope_wait_handle", None)
    if handle is None:
        raise RuntimeError("snapshot.consume_selected_scope_wait_handle is required")
    if not bool(getattr(handle, "is_ready", False)):
        expected_layers = tuple(int(v) for v in getattr(handle, "expected_layers", tuple()))
        committed_layers = tuple(
            sorted(int(v) for v in getattr(handle, "terminal_layer_status", {}).keys())
        )
        raise NotReadyError(
            "selected scope wait handle is not ready "
            f"(consumer_step_id={int(getattr(handle, 'consumer_step_id', -1))}, "
            f"layer_group_id={int(getattr(handle, 'layer_group_id', -1))}, "
            f"expected_layers={expected_layers}, committed_layers={committed_layers})"
        )
    return "ready_for_final_launch_scratch"
