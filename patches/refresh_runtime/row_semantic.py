from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch

_IS_COMPACT_CACHE: Dict[Tuple[str, int, Tuple[int, ...]], torch.Tensor] = {}


def build_is_compact_i32_from_use_compact(
    *,
    use_compact_by_row: Sequence[bool],
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    bits = tuple(1 if bool(use_compact_by_row[idx]) else 0 for idx in range(int(batch_size)))
    dev_index = int(device.index) if device.index is not None else -1
    key = (str(device.type), dev_index, bits)
    cached = _IS_COMPACT_CACHE.get(key)
    if (
        cached is None
        or cached.device != device
        or cached.dtype != torch.int32
        or cached.numel() != int(batch_size)
    ):
        cached = torch.tensor(bits, dtype=torch.int32, device=device)
        if len(_IS_COMPACT_CACHE) > 256:
            _IS_COMPACT_CACHE.clear()
        _IS_COMPACT_CACHE[key] = cached
    return cached

