from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

# [RING-DEPTH-3 2026-07-08] depth 3 admitted for the refresh-pipeline dwell
# cut (per-generation 3-chunk cadence no longer forces buf reuse inside one
# generation). Consumers are fully parameterized on _CAPTURE_IN_FLIGHT (ring
# mixin, wait ledger, meta arena); memory account: capture scratch reserve
# projects ~1.75 GiB per buf. Default stays 2 — flip via
# VLLM_SPARSE_CAPTURE_IN_FLIGHT=3 after the speed A/B on the target card.
_SUPPORTED_CAPTURE_IN_FLIGHT: Tuple[int, ...] = (1, 2, 3)


def validate_capture_inflight(value: int) -> int:
    v = int(value)
    if v not in _SUPPORTED_CAPTURE_IN_FLIGHT:
        raise ValueError(
            f"_CAPTURE_IN_FLIGHT={v} is unsupported. allowed={_SUPPORTED_CAPTURE_IN_FLIGHT}"
        )
    return v


@dataclass(frozen=True, slots=True)
class StepSemanticSnapshot:
    epoch: int
    alpha_log_f: float
    sink_tokens: int
    recent_tokens: int
    capture_inflight: int


class ExecutionBackendLedger:
    def __init__(self, num_bufs: int) -> None:
        n = int(num_bufs)
        if n <= 0:
            raise ValueError(f"num_bufs must be > 0, got {n}")
        self.flags = [0 for _ in range(n)]
        self.epoch = [-1 for _ in range(n)]

    def mark_submitted(
        self,
        *,
        buf_id: int,
        epoch: int,
        kind: str,
        async_mode: bool,
    ) -> None:
        if not async_mode:
            return
        if kind == "prefill":
            bit = 1
        elif kind == "refresh":
            bit = 2
        else:
            raise ValueError(f"kind must be 'prefill' or 'refresh', got {kind!r}")
        b = int(buf_id) % len(self.flags)
        self.flags[b] |= bit
        self.epoch[b] = int(epoch)

    def clear(self, *, buf_id: int) -> None:
        b = int(buf_id) % len(self.flags)
        self.flags[b] = 0
        self.epoch[b] = -1
