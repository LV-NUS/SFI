from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Mapping, Sequence, Tuple


class PendingPolicy(IntEnum):
    COALESCEABLE = 0
    FORCE_NOW = 1


class PendingReasonCode(IntEnum):
    NONE = 0
    TRIGGER = 1
    SENTENCE = 2
    INTERVAL = 3
    COMPACT_THRESHOLD_CROSSED = 4
    LEASE_REARM = 5
    REFRESH = 6
    COMPACT_NOT_READY = 7


def pending_reason_to_code(reason: str) -> int:
    reason_norm = str(reason or "none").strip().lower()
    if "," in reason_norm:
        best = int(PendingReasonCode.NONE)
        for part in reason_norm.split(","):
            code = pending_reason_to_code(part)
            if code == int(PendingReasonCode.SENTENCE):
                return code
            if code == int(PendingReasonCode.INTERVAL):
                best = code
            elif code != int(PendingReasonCode.NONE) and best == int(PendingReasonCode.NONE):
                best = code
        return best
    if reason_norm in {"", "none"}:
        return int(PendingReasonCode.NONE)
    if reason_norm == "compact_threshold_crossed":
        return int(PendingReasonCode.COMPACT_THRESHOLD_CROSSED)
    if reason_norm == "lease_rearm":
        return int(PendingReasonCode.LEASE_REARM)
    if reason_norm == "compact_not_ready":
        return int(PendingReasonCode.COMPACT_NOT_READY)
    if reason_norm == "sentence":
        return int(PendingReasonCode.SENTENCE)
    if reason_norm == "interval":
        return int(PendingReasonCode.INTERVAL)
    if reason_norm.startswith("sentence_"):
        return int(PendingReasonCode.SENTENCE)
    if reason_norm.startswith("interval_"):
        return int(PendingReasonCode.INTERVAL)
    if reason_norm == "refresh":
        return int(PendingReasonCode.REFRESH)
    return int(PendingReasonCode.TRIGGER)


def pending_reason_code_to_text(reason_code: int) -> str:
    try:
        code = PendingReasonCode(int(reason_code))
    except (ValueError, TypeError):
        return "trigger"
    if code == PendingReasonCode.NONE:
        return "none"
    if code == PendingReasonCode.TRIGGER:
        return "trigger"
    if code == PendingReasonCode.SENTENCE:
        return "sentence"
    if code == PendingReasonCode.INTERVAL:
        return "interval"
    if code == PendingReasonCode.COMPACT_THRESHOLD_CROSSED:
        return "compact_threshold_crossed"
    if code == PendingReasonCode.LEASE_REARM:
        return "lease_rearm"
    if code == PendingReasonCode.REFRESH:
        return "refresh"
    if code == PendingReasonCode.COMPACT_NOT_READY:
        return "compact_not_ready"
    return "trigger"


@dataclass(slots=True)
class RequestIntentTicket:
    pending_refresh: bool = False
    pending_reason_code: int = int(PendingReasonCode.NONE)
    pending_decode_step: int = -1
    pending_ctrl_step: int = -1
    pending_policy: int = int(PendingPolicy.COALESCEABLE)


def _validate_decode_step(decode_step: int) -> int:
    if isinstance(decode_step, bool) or not isinstance(decode_step, int):
        raise TypeError(f"decode_step must be int, got {type(decode_step).__name__}")
    if decode_step < 0:
        raise ValueError(f"decode_step must be >= 0, got {decode_step}")
    return decode_step


def _validate_coalesce_window(coalesce_window: int) -> int:
    if isinstance(coalesce_window, bool) or not isinstance(coalesce_window, int):
        raise TypeError(
            f"coalesce_window must be int, got {type(coalesce_window).__name__}"
        )
    if coalesce_window < 0:
        raise ValueError(f"coalesce_window must be >= 0, got {coalesce_window}")
    return coalesce_window


def mark_threshold_crossing(ticket: RequestIntentTicket, decode_step: int) -> None:
    step = _validate_decode_step(decode_step)
    ticket.pending_refresh = True
    ticket.pending_reason_code = int(PendingReasonCode.COMPACT_THRESHOLD_CROSSED)
    ticket.pending_decode_step = step
    ticket.pending_policy = int(PendingPolicy.FORCE_NOW)


def materialize_refresh_reqs(
    request_ids: Sequence[str],
    tickets_by_req: Mapping[str, RequestIntentTicket],
    coalesce_window: int,
) -> Tuple[str, ...]:
    window = _validate_coalesce_window(coalesce_window)

    coalesce_pending_by_req: dict[str, int] = {}
    force_now_set: set[str] = set()
    earliest_pending_decode: int | None = None

    for rid in request_ids:
        ticket = tickets_by_req.get(rid)
        if ticket is None or not ticket.pending_refresh:
            continue

        pending_step = int(ticket.pending_decode_step)
        if pending_step < 0:
            raise RuntimeError(
                f"pending_refresh ticket missing valid pending_decode_step for request {rid!r}"
            )

        policy = int(ticket.pending_policy)
        if policy == int(PendingPolicy.FORCE_NOW):
            force_now_set.add(rid)
            continue

        coalesce_pending_by_req[rid] = pending_step
        if earliest_pending_decode is None or pending_step < earliest_pending_decode:
            earliest_pending_decode = pending_step

    selected: set[str] = set(force_now_set)

    if earliest_pending_decode is not None:
        target = int(earliest_pending_decode)
        for rid in request_ids:
            if rid not in coalesce_pending_by_req:
                continue
            if abs(int(coalesce_pending_by_req[rid]) - target) <= window:
                selected.add(rid)

    return tuple(rid for rid in request_ids if rid in selected)
