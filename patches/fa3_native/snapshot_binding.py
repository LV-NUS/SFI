from __future__ import annotations

from dataclasses import dataclass, field

from .contracts import TargetSelectedScopeKey


@dataclass(frozen=True)
class SelectedScopeKey:
    consumer_step_id: int
    layer_group_id: int
    chunk_id: int


@dataclass(slots=True)
class ScopeWaitHandle:
    consumer_step_id: int
    layer_group_id: int
    is_ready: bool = False
    expected_layers: tuple[int, ...] = ()
    terminal_layer_status: dict[int, str] = field(default_factory=dict)


@dataclass(frozen=True)
class LaunchLocalSnapshot:
    target_selected_scope_key: TargetSelectedScopeKey
    selected_scope_key: SelectedScopeKey
    selected_scope_wait_handle: ScopeWaitHandle
    mode_signature: tuple[str, str]
    has_selected_consume: bool
    has_capture: bool
    consume_selected_scope_key: SelectedScopeKey | None = None
    consume_selected_scope_wait_handle: ScopeWaitHandle | None = None


def make_scope_wait_handle(
    consumer_step_id: int,
    layer_group_id: int,
    *,
    expected_layers: tuple[int, ...] = (),
) -> ScopeWaitHandle:
    return ScopeWaitHandle(
        consumer_step_id=int(consumer_step_id),
        layer_group_id=int(layer_group_id),
        expected_layers=tuple(int(layer_id) for layer_id in expected_layers),
    )


def bind_snapshot(attn_metadata: object, *, snapshot: LaunchLocalSnapshot) -> object:
    setattr(attn_metadata, "fa3_native_snapshot", snapshot)
    return attn_metadata


def get_bound_snapshot(attn_metadata: object) -> LaunchLocalSnapshot:
    return getattr(attn_metadata, "fa3_native_snapshot")


def invalidate_snapshot_if_signature_changes(
    snapshot: LaunchLocalSnapshot,
    new_signature: tuple[str, str],
) -> LaunchLocalSnapshot | None:
    if snapshot.mode_signature != new_signature:
        return None
    return snapshot
