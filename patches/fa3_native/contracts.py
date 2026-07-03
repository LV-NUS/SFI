from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TargetSelectedScopeKey:
    consumer_step_id: int
    layer_group_id: int


@dataclass(frozen=True)
class FA3NativeContracts:
    patch_root: str = "patches/fa3_native"
    kernel_root: str = "third_party_upstreams/vllm-project-flash-attention"
