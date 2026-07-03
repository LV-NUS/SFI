from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping


TIMELINE_ENV = "VLLM_SPARSE_ONE_SHOT_TIMELINE_LOG"


def timeline_log_path() -> Path | None:
    raw = os.environ.get(TIMELINE_ENV, "")
    return Path(raw) if raw else None


def append_one_shot_timeline(
    *,
    path: Path | None,
    request_id: str,
    epoch: int,
    chunk_id: int,
    buffer_id: int,
    phase: str,
    timestamp_ns: int,
    duration_us: float,
    aux_stream_enabled: bool | None = None,
    extra_fields: Mapping[str, object] | None = None,
) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "event": "one_shot_timeline",
        "request_id": str(request_id),
        "epoch": int(epoch),
        "chunk_id": int(chunk_id),
        "buffer_id": int(buffer_id),
        "phase": str(phase),
        "timestamp_ns": int(timestamp_ns),
        "duration_us": float(duration_us),
    }
    if aux_stream_enabled is not None:
        payload["aux_stream_enabled"] = bool(aux_stream_enabled)
    if extra_fields:
        for key, value in extra_fields.items():
            payload[str(key)] = value
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
