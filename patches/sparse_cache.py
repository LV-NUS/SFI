"""
Lightweight tensor cache utilities for the sparse attention engine.

No dependency on vllm_sparse_patch.py, safe to import from any module.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch

__all__ = [
    "_get_cached_empty_tensor",
]

_EMPTY_TENSORS_BY_KEY: Dict[Tuple[str, str, str], torch.Tensor] = {}


def _get_cached_empty_tensor(
    *, device: torch.device, dtype: torch.dtype, shape: Tuple[int, ...]
) -> torch.Tensor:
    """Return a reusable empty tensor (read-only semantics)."""
    key = (str(device), str(dtype), "x".join(str(int(s)) for s in shape))
    cached = _EMPTY_TENSORS_BY_KEY.get(key)
    if (
        cached is None
        or cached.device != device
        or cached.dtype != dtype
        or tuple(cached.shape) != tuple(shape)
    ):
        cached = torch.empty(shape, device=device, dtype=dtype)
        _EMPTY_TENSORS_BY_KEY[key] = cached
    return cached
