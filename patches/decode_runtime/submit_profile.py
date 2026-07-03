from __future__ import annotations

import atexit
import json
import os
import time
from collections import defaultdict
from typing import DefaultDict, List, Tuple

_PROFILE_ENV = "VLLM_SPARSE_SUBMIT_PROFILE_LOG"
_TIMING_ENV = "VLLM_SPARSE_SUBMIT_TIMING_LOG"
_FLUSH_EVERY_ENV = "VLLM_SPARSE_SUBMIT_TIMING_FLUSH_EVERY"


def _profile_path() -> str:
    return (os.environ.get(_TIMING_ENV, "") or os.environ.get(_PROFILE_ENV, "")).strip()


_ENABLED = bool(_profile_path())
_PID = int(os.getpid())
_VALUES: DefaultDict[Tuple[int, str, int], List[float]] = defaultdict(list)
_RECORD_COUNT = 0
_FLUSH_SEQ = 0
_FIRST_MARKER_WRITTEN = False


def sparse_submit_profile_enabled() -> bool:
    return _ENABLED


def sparse_submit_profile_time_ns() -> int:
    return time.perf_counter_ns()


def record_sparse_submit_profile(name: str, layer_index: int, elapsed_ns: int) -> None:
    global _FIRST_MARKER_WRITTEN, _RECORD_COUNT
    if not _ENABLED:
        return
    elapsed = int(elapsed_ns)
    if elapsed < 0:
        return
    if not _FIRST_MARKER_WRITTEN:
        _write_sparse_submit_profile_rows(
            [
                {
                    "event": "sparse_submit_profile_start",
                    "pid": int(_PID),
                    "path": _profile_path(),
                }
            ]
        )
        _FIRST_MARKER_WRITTEN = True
    _VALUES[(_PID, str(name), int(layer_index))].append(float(elapsed) / 1000.0)
    _RECORD_COUNT += 1
    flush_every = int(os.environ.get(_FLUSH_EVERY_ENV, "4096") or "4096")
    if flush_every > 0 and (_RECORD_COUNT % flush_every) == 0:
        _flush_sparse_submit_profile(final=False)


def _pct(values: List[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return float(ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo))


def _write_sparse_submit_profile_rows(rows: List[dict]) -> None:
    path = _profile_path()
    if not path or not rows:
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, sort_keys=True) + "\n")


def _flush_sparse_submit_profile(final: bool = True) -> None:
    global _FLUSH_SEQ
    if not _VALUES:
        return
    _FLUSH_SEQ += 1
    rows = []
    record_count_total = int(_RECORD_COUNT)
    flush_seq = int(_FLUSH_SEQ)
    is_final = bool(final)
    for (pid, name, layer_index), values in sorted(_VALUES.items()):
        row = {
            "event": "sparse_submit_profile_summary",
            "final": is_final,
            "flush_seq": flush_seq,
            "pid": int(pid),
            "name": str(name),
            "layer_index": int(layer_index),
            "count": int(len(values)),
            "p50_us": _pct(values, 0.5),
            "p90_us": _pct(values, 0.9),
            "record_count_total": record_count_total,
            "sum_us": float(sum(values)),
        }
        rows.append(row)
    _write_sparse_submit_profile_rows(rows)


atexit.register(_flush_sparse_submit_profile)
