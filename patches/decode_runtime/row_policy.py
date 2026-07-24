from __future__ import annotations

from collections.abc import Callable, Mapping

from patches.sparse_constants import (
    _FORCE_DENSE_CACHED,
    _FORCE_COMPACT_OFF_CACHED,
    _LOGF_PRODUCER_ATTN,
    _LOGF_PRODUCER_NONE,
    _ROW_MODE_COMPACT,
    _ROW_MODE_DENSE,
    _ROW_MODE_LOG_F_REFRESH,
)


def classify_one_shot_decode_admission(
    req_ids: tuple[str, ...],
    *,
    row_policy_ready_by_row: tuple[bool, ...],
    request_states: Mapping[str, object],
    can_bridge_bootstrap_decode: Callable[[str], object],
    diagnose_compact_ready_all_layers: Callable[[str], object],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if len(row_policy_ready_by_row) != len(req_ids):
        raise RuntimeError(
            "one-shot decode admission row coverage mismatch: "
            f"ready={len(row_policy_ready_by_row)} requests={len(req_ids)}"
        )

    bridge_not_ready: list[str] = []
    blocked_not_ready: list[str] = []
    for row, rid in enumerate(req_ids):
        # StepAuthority already resolved both legal steady states:
        # compact-readable rows and intentional dense rows (short/crossing).
        # Rechecking compact state here used to reject the latter and repeated
        # an all-layer scan on every decode step.
        if bool(row_policy_ready_by_row[row]):
            continue
        if bool(can_bridge_bootstrap_decode(rid)):
            bridge_not_ready.append(rid)
            continue
        # Attribution can deliberately delay compact consumption after the
        # request lifecycle is terminal. Admit that cold diagnostic state only
        # when compact is already physically readable; raw lifecycle state
        # alone must never hide an early-publication bug.
        tracking = request_states.get(rid)
        if (
            bool(getattr(tracking, "bootstrap_done", False))
            and bool(diagnose_compact_ready_all_layers(rid))
        ):
            continue
        blocked_not_ready.append(rid)
    return tuple(bridge_not_ready), tuple(blocked_not_ready)


def resolve_decode_row_policy(
    *,
    is_prefill_row: bool,
    is_refresh_row: bool,
    dense_protection_active: bool,
    row_policy_ready: bool,
    force_dense_for_pending_refresh: bool = False,
) -> tuple[int, int]:
    """集中解析 decode 阶段每行的 row_mode 与 log_f producer。"""
    if is_prefill_row:
        return int(_ROW_MODE_DENSE), int(_LOGF_PRODUCER_NONE)

    if is_refresh_row:
        return int(_ROW_MODE_LOG_F_REFRESH), int(_LOGF_PRODUCER_ATTN)

    if dense_protection_active or (not row_policy_ready):
        return int(_ROW_MODE_DENSE), int(_LOGF_PRODUCER_NONE)
    if force_dense_for_pending_refresh:
        return int(_ROW_MODE_DENSE), int(_LOGF_PRODUCER_NONE)
    if _FORCE_DENSE_CACHED or _FORCE_COMPACT_OFF_CACHED:
        return int(_ROW_MODE_DENSE), int(_LOGF_PRODUCER_NONE)
    return int(_ROW_MODE_COMPACT), int(_LOGF_PRODUCER_NONE)
