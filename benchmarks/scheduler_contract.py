"""Shared scheduler-cap contract for benchmark parents and children."""

from __future__ import annotations


def resolve_benchmark_max_num_seqs(
    *,
    batch_size: int,
    configured: int = 0,
) -> int:
    """Resolve 0 to batch size and reject caps that serialize the workload."""
    batch = int(batch_size)
    requested = int(configured)
    if batch <= 0:
        raise ValueError("--batch-size must be > 0")
    if requested < 0:
        raise ValueError("--max-num-seqs must be >= 0")
    if 0 < requested < batch:
        raise ValueError("--max-num-seqs must be 0 or >= --batch-size")
    return requested if requested > 0 else batch
