from __future__ import annotations

from collections.abc import Callable

from patches.sparse_constants import (
    _FORCE_DENSE_CACHED,
    _FORCE_COMPACT_OFF_CACHED,
    _LOGF_PRODUCER_ATTN,
    _LOGF_PRODUCER_NONE,
    _ROW_MODE_COMPACT,
    _ROW_MODE_DENSE,
    _ROW_MODE_LOG_F_REFRESH,
)


def one_shot_row_route_phase(
    *,
    bootstrap_bridge_active: bool,
    compact_ready: bool,
) -> str:
    if bool(bootstrap_bridge_active):
        return "bridge_phase"
    if bool(compact_ready):
        return "post_switch_phase"
    return "blocked_not_ready"


def classify_one_shot_bootstrap_decode_guard(
    req_ids: tuple[str, ...],
    *,
    compact_ready_all_layers: Callable[[str], object],
    can_bridge_bootstrap_decode: Callable[[str], object],
) -> tuple[tuple[str, ...], tuple[str, ...], str]:
    bridge_not_ready: list[str] = []
    blocked_not_ready: list[str] = []
    has_bridge_phase = False
    has_blocked_phase = False
    for rid in req_ids:
        can_bridge = bool(can_bridge_bootstrap_decode(rid))
        compact_ready = bool(compact_ready_all_layers(rid))
        if can_bridge:
            bridge_not_ready.append(rid)
        elif not compact_ready:
            blocked_not_ready.append(rid)
        row_phase = one_shot_row_route_phase(
            bootstrap_bridge_active=can_bridge,
            compact_ready=compact_ready,
        )
        has_bridge_phase = has_bridge_phase or row_phase == "bridge_phase"
        has_blocked_phase = has_blocked_phase or row_phase == "blocked_not_ready"

    if has_blocked_phase:
        decode_phase = "blocked_not_ready"
    elif has_bridge_phase:
        decode_phase = "bridge_active_without_compact_ready"
    else:
        decode_phase = "post_switch_phase"
    return tuple(bridge_not_ready), tuple(blocked_not_ready), decode_phase


def resolve_decode_row_policy(
    *,
    is_prefill_row: bool,
    is_refresh_row: bool,
    is_short_dense_row: bool,
    bootstrap_done_row: bool,
    force_dense_for_pending_refresh: bool = False,
) -> tuple[int, int]:
    """集中解析 decode 阶段每行的 row_mode 与 log_f producer。"""
    if is_prefill_row:
        return int(_ROW_MODE_DENSE), int(_LOGF_PRODUCER_NONE)

    if is_refresh_row:
        return int(_ROW_MODE_LOG_F_REFRESH), int(_LOGF_PRODUCER_ATTN)

    if is_short_dense_row or (not bootstrap_done_row):
        return int(_ROW_MODE_DENSE), int(_LOGF_PRODUCER_NONE)
    if force_dense_for_pending_refresh:
        return int(_ROW_MODE_DENSE), int(_LOGF_PRODUCER_NONE)
    if _FORCE_DENSE_CACHED or _FORCE_COMPACT_OFF_CACHED:
        return int(_ROW_MODE_DENSE), int(_LOGF_PRODUCER_NONE)
    return int(_ROW_MODE_COMPACT), int(_LOGF_PRODUCER_NONE)
