from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional, Sequence

import torch

_log = logging.getLogger(__name__)


@dataclass
class AllocatorBackend:
    name: str

    def alloc_empty(
        self,
        shape: Sequence[int],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        raise NotImplementedError

    def alloc_full(
        self,
        shape: Sequence[int],
        *,
        fill_value: int | float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        raise NotImplementedError


class TorchAllocator(AllocatorBackend):
    def __init__(self) -> None:
        super().__init__(name="torch")

    def alloc_empty(
        self,
        shape: Sequence[int],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.empty(tuple(int(x) for x in shape), device=device, dtype=dtype)

    def alloc_full(
        self,
        shape: Sequence[int],
        *,
        fill_value: int | float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.full(tuple(int(x) for x in shape), fill_value, device=device, dtype=dtype)


class VMMAllocator(TorchAllocator):
    """VMM backend hook.

    Current implementation keeps torch allocation semantics for compatibility and
    uses the backend name to provide explicit runtime visibility.
    """

    def __init__(self) -> None:
        super().__init__()
        self.name = "vmm"


_BACKEND_CACHE: Optional[AllocatorBackend] = None
_BACKEND_CACHE_KEY: Optional[str] = None


def resolve_allocator_backend() -> AllocatorBackend:
    global _BACKEND_CACHE, _BACKEND_CACHE_KEY
    requested = os.environ.get("VLLM_SPARSE_ALLOC_BACKEND", "torch").strip().lower()
    key = requested or "torch"
    if _BACKEND_CACHE is not None and _BACKEND_CACHE_KEY == key:
        return _BACKEND_CACHE

    backend: AllocatorBackend
    if key == "vmm":
        # 实际 VMM 能力探测留给后续迭代；当前版本先以独立 backend name 打通协议层。
        # 在不支持 VMM 的设备上退化到 torch 语义，但保持可观测性。
        try:
            backend = VMMAllocator()
        except Exception:
            _log.warning("VMMAllocator creation failed, raising", exc_info=True)
            raise
    else:
        backend = TorchAllocator()

    _BACKEND_CACHE = backend
    _BACKEND_CACHE_KEY = key
    return backend

