from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

# [RING-DEPTH-3 收窄 2026-07-11] depth 3 was admitted 2026-07-08 on the claim
# "consumers are fully parameterized on _CAPTURE_IN_FLIGHT" — code archaeology
# proved that claim FALSE at the pack-kernel layer: the out_ptr fast path
# routes capture buffers with a BINARY tl.where(buf_id == 0, buf0, buf1)
# (triton_kernel/flash_attn_score_dump_fwd.py, decode + prefill arms) and the
# pack call sites bake exactly TWO scores base pointers. Under depth 3 every
# buf_id == 2 layer group would silently write into ring[1] while chunk1 is
# still in flight — cross-generation capture corruption with NO error (wrong
# top-k, quality rot). Contract narrowed to what the kernel actually
# implements so the bad value can never be produced (fail-fast at env
# validation, 无 fallback 纪律). Re-admit 3 ONLY after the kernel's buf
# routing is generalized to an indexed base-pointer array and a depth-3
# golden-anchor run passes.
_SUPPORTED_CAPTURE_IN_FLIGHT: Tuple[int, ...] = (1, 2)


def validate_capture_inflight(value: int) -> int:
    v = int(value)
    if v not in _SUPPORTED_CAPTURE_IN_FLIGHT:
        raise ValueError(
            f"_CAPTURE_IN_FLIGHT={v} is unsupported. allowed={_SUPPORTED_CAPTURE_IN_FLIGHT}"
        )
    return v


_SUPPORTED_CAPTURE_REDUCE_GROUPS: Tuple[int, ...] = (0, 1, 2, 4)


def validate_reduce_group(reduce_group: int, capture_chunk: int) -> int:
    """[REDUCE-GROUP-SINGLE-SOURCE 2026-07-11 EXT审计·随手批] 域校验唯一真源。

    此前该域有两份实现：生产走 sparse_constants 内联判定（[REDUCE-GROUP-DOMAIN
    收窄 2026-07-11] 已改 raise），测试走 fa3_native/ring_capture.py 的副本
    （仍是旧的静默 return 0 = fallback 形态）——语义已漂移，测试测的不是生产
    路径。收敛为本纯合同函数（raise 语义，与生产逐位同判）；ring_capture 与
    sparse_constants 均消费此处。
    """
    g = int(reduce_group)
    chunk = int(capture_chunk)
    if g not in _SUPPORTED_CAPTURE_REDUCE_GROUPS:
        raise ValueError(
            f"capture reduce_group={g} unsupported; "
            f"allowed={_SUPPORTED_CAPTURE_REDUCE_GROUPS}"
        )
    if g > 0 and chunk % g != 0:
        raise ValueError(
            f"capture reduce_group={g} does not divide capture_chunk={chunk}; "
            "the per-G raw ring requires capture_chunk % G == 0 (set a "
            "dividing G, or G=0 for the explicit chunk-deep diagnostic mode)"
        )
    return g


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
