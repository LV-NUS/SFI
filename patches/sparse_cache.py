"""
Lightweight tensor cache utilities for the sparse attention engine.

No dependency on vllm_sparse_patch.py, safe to import from any module.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch

__all__ = [
    "_get_cached_empty_tensor",
    "_ROW_INDEX_TENSOR_CACHE_GLOBAL",
    "_LOGITS_PATCH_STEPWISE_CACHE_GLOBAL",
]

_EMPTY_TENSORS_BY_KEY: Dict[Tuple[str, str, str], torch.Tensor] = {}

# 备用 row_index cache（controller=None 时使用）。正常 sparse 路径 controller 总是存在；
# 该 cache 仅用于保证非常规路径也不额外引入 per-layer small alloc。
_ROW_INDEX_TENSOR_CACHE_GLOBAL: Dict[Tuple[str, int, Tuple[int, ...]], torch.Tensor] = {}
_LOGITS_PATCH_STEPWISE_CACHE_GLOBAL: Dict[
    Tuple[int, str, int, int, int, int, int, int, Tuple[object, ...], Tuple[int, ...]],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
] = {}


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
