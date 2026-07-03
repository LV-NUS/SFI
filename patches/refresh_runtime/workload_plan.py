from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


WORKLOAD_PLAN_SCHEMA = "sm80_refresh_workload_plan_v1"
WORKLOAD_PLAN_REPLAY_ENV = "VLLM_SPARSE_REFRESH_WORKLOAD_PLAN_REPLAY"

_BENCH_REQUEST_ID_RE = re.compile(r"^bench-(\d+)-")


def _stable_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _digest_payload(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": plan.get("schema"),
        "config": plan.get("config", {}),
        "events": plan.get("events", []),
    }


def workload_plan_digest(plan: Mapping[str, Any]) -> str:
    return hashlib.sha256(_stable_json(_digest_payload(plan)).encode("utf-8")).hexdigest()


def attach_workload_plan_digest(plan: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(plan)
    result["workload_plan_digest"] = workload_plan_digest(result)
    return result


def _request_ordinal_from_id(request_id: str) -> int:
    match = _BENCH_REQUEST_ID_RE.match(str(request_id))
    if match is None:
        raise ValueError(
            "workload plan export requires benchmark request ids shaped "
            f"'bench-<ordinal>-<suffix>', got {request_id!r}"
        )
    return int(match.group(1))


def _reason_text(value: object) -> str:
    text = str(value or "refresh").strip().lower()
    if "sentence" in text:
        return "sentence"
    if "interval" in text:
        return "interval"
    if text in {"", "none"}:
        return "refresh"
    return text


def _pending_policy_text(value: object, *, reason: str) -> str:
    try:
        policy = int(value)
    except (TypeError, ValueError):
        policy_text = str(value or "").strip().lower()
        if policy_text in {"force_now", "force-now", "force"}:
            return "force_now"
        if policy_text in {"coalesceable", "coalescable", "defer"}:
            return "coalesceable"
        return "force_now" if reason == "sentence" else "coalesceable"
    return "force_now" if policy == 1 else "coalesceable"


def _debug_by_request_id(event: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    debug = event.get("refresh_intent_debug", [])
    if not isinstance(debug, Sequence):
        return {}
    result: dict[str, Mapping[str, Any]] = {}
    for item in debug:
        if not isinstance(item, Mapping):
            continue
        req_id = item.get("req_id")
        if req_id is None:
            continue
        result[str(req_id)] = item
    return result


def build_workload_plan_from_route_events(
    events: Iterable[Mapping[str, Any]],
    *,
    source_summary: str,
    source_route_trace: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    plan_events: list[dict[str, Any]] = []
    for event in events:
        if event.get("event") != "mixed_page_full_cudagraph_replay_refresh_payload_enqueue":
            continue
        reason = _reason_text(event.get("refresh_reason"))
        request_ids = event.get("refresh_intent_req_ids", [])
        if not isinstance(request_ids, Sequence) or isinstance(request_ids, (str, bytes)):
            continue
        debug_by_req = _debug_by_request_id(event)
        for request_id_raw in request_ids:
            request_id = str(request_id_raw)
            debug = debug_by_req.get(request_id, {})
            decode_step = debug.get("decode_step", event.get("step_id", -1))
            try:
                decode_step_int = int(decode_step)
            except (TypeError, ValueError):
                decode_step_int = -1
            if decode_step_int < 0:
                continue
            plan_events.append(
                {
                    "decode_step": decode_step_int,
                    "request_ordinal": _request_ordinal_from_id(request_id),
                    "reason": reason,
                    "pending_policy": _pending_policy_text(
                        debug.get("pending_policy"),
                        reason=reason,
                    ),
                    "refresh_slot_count": int(event.get("refresh_slot_count", 0) or 0),
                    "payload_count": int(event.get("payload_count", 0) or 0),
                }
            )

    return attach_workload_plan_digest(
        {
            "schema": WORKLOAD_PLAN_SCHEMA,
            "source_summary": str(source_summary),
            "source_route_trace": str(source_route_trace),
            "config": dict(config),
            "events": plan_events,
        }
    )


def load_workload_plan(path: str | os.PathLike[str]) -> dict[str, Any]:
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(plan, dict):
        raise ValueError("workload plan must be a JSON object")
    if plan.get("schema") != WORKLOAD_PLAN_SCHEMA:
        raise ValueError(
            f"unsupported workload plan schema: {plan.get('schema')!r}"
        )
    events = plan.get("events", [])
    if not isinstance(events, list):
        raise ValueError("workload plan events must be a list")
    return attach_workload_plan_digest(plan)


def load_workload_plan_from_env() -> dict[str, Any] | None:
    path = os.environ.get(WORKLOAD_PLAN_REPLAY_ENV, "").strip()
    if not path:
        return None
    return load_workload_plan(path)


def index_workload_plan_events(
    plan: Mapping[str, Any] | None,
) -> dict[tuple[int, int], tuple[Mapping[str, Any], ...]]:
    if not plan:
        return {}
    events = plan.get("events", [])
    if not isinstance(events, Sequence):
        return {}
    grouped: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
    for event in events:
        if not isinstance(event, Mapping):
            continue
        key = (
            int(event.get("request_ordinal", -1)),
            int(event.get("decode_step", -1)),
        )
        if key[0] < 0 or key[1] < 0:
            continue
        grouped.setdefault(key, []).append(event)
    return {key: tuple(value) for key, value in grouped.items()}


def workload_plan_events_for_request_step(
    plan: Mapping[str, Any] | None,
    *,
    request_ordinal: int,
    decode_step: int,
) -> tuple[Mapping[str, Any], ...]:
    if not plan:
        return tuple()
    events = plan.get("events", [])
    if not isinstance(events, Sequence):
        return tuple()
    matched: list[Mapping[str, Any]] = []
    for event in events:
        if not isinstance(event, Mapping):
            continue
        if int(event.get("request_ordinal", -1)) != int(request_ordinal):
            continue
        if int(event.get("decode_step", -1)) != int(decode_step):
            continue
        matched.append(event)
    return tuple(matched)


def indexed_workload_plan_events_for_request_step(
    indexed_events: Mapping[tuple[int, int], tuple[Mapping[str, Any], ...]] | None,
    *,
    request_ordinal: int,
    decode_step: int,
) -> tuple[Mapping[str, Any], ...]:
    if not indexed_events:
        return tuple()
    return indexed_events.get((int(request_ordinal), int(decode_step)), tuple())


def workload_plan_pending_policy(event: Mapping[str, Any]) -> int:
    return 1 if str(event.get("pending_policy", "")).lower() == "force_now" else 0
