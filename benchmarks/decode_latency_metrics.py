from __future__ import annotations

import math
from typing import Dict, Sequence


def _linear_percentile(sorted_values: Sequence[float], ratio: float) -> float:
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    clamped = min(1.0, max(0.0, float(ratio)))
    rank = clamped * float(len(sorted_values) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return float(sorted_values[lo])
    weight = rank - float(lo)
    low = float(sorted_values[lo])
    high = float(sorted_values[hi])
    return low * (1.0 - weight) + high * weight


def summarize_decode_metrics(
    *,
    decode_step_durations_s: Sequence[float],
    decode_tokens: int,
    decode_elapsed_s: float,
) -> Dict[str, float]:
    values_s = sorted(float(v) for v in decode_step_durations_s if float(v) >= 0.0)
    p50_us = _linear_percentile(values_s, 0.50) * 1e6
    p95_us = _linear_percentile(values_s, 0.95) * 1e6
    p99_us = _linear_percentile(values_s, 0.99) * 1e6
    decode_tps = (
        float(decode_tokens) / float(decode_elapsed_s)
        if float(decode_elapsed_s) > 0.0
        else float("nan")
    )
    return {
        "decode_p50_us": float(p50_us),
        "decode_p95_us": float(p95_us),
        "decode_p99_us": float(p99_us),
        "decode_tokens_per_s": float(decode_tps),
    }


def decode_step_durations_us(decode_step_durations_s: Sequence[float]) -> list[float]:
    return [
        float(duration_s) * 1e6
        for duration_s in decode_step_durations_s
        if float(duration_s) >= 0.0
    ]


def format_decode_metrics_line(metrics: Dict[str, float]) -> str:
    return (
        f"decode_p50_us={float(metrics['decode_p50_us']):.2f} "
        f"decode_p95_us={float(metrics['decode_p95_us']):.2f} "
        f"decode_p99_us={float(metrics['decode_p99_us']):.2f} "
        f"decode_tokens_per_s={float(metrics['decode_tokens_per_s']):.2f}"
    )
