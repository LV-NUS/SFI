"""LaunchTemplate owner for compact_recent dynamic descriptor rows."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

from patches.sparse_types import CompactRecentLaunchPlan


_DESCRIPTOR_ROWS = 6


@dataclass(slots=True)
class LaunchTemplate:
    plan: CompactRecentLaunchPlan
    descriptor_cpu_i32: torch.Tensor
    descriptor_gpu_i32: torch.Tensor
    slot_signature: Tuple[int, ...]
    use_compact_signature: Tuple[int, ...]
    compact_meta_epoch: int
    max_seqlen_k_capacity: int


@dataclass(frozen=True, slots=True)
class LaunchTemplateUpdateResult:
    requires_recompile: bool
    reason: str
    updated_rows: Tuple[int, ...]
    carrier_update_kernel_count: int


def _validate_descriptor(name: str, descriptor: torch.Tensor, batch_size: int) -> None:
    if not isinstance(descriptor, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    if descriptor.dtype is not torch.int32:
        raise ValueError(f"{name} must have dtype torch.int32")
    if descriptor.ndim != 2:
        raise ValueError(f"{name} must be a 2D tensor")
    if descriptor.shape[0] < _DESCRIPTOR_ROWS or descriptor.shape[1] < batch_size:
        raise ValueError(
            f"{name} shape must cover [{_DESCRIPTOR_ROWS}, {batch_size}]"
        )


def _tuple_from_rows(name: str, values: Tuple[int, ...], batch_size: int) -> Tuple[int, ...]:
    if len(values) < batch_size:
        raise ValueError(f"{name} must cover batch_size={batch_size}")
    return tuple(int(v) for v in values[:batch_size])


def _row_changed(prior: Tuple[int, ...], row: int, value: int) -> bool:
    return row >= len(prior) or int(prior[row]) != int(value)


def _fresh_descriptor_cpu_mirror(
    reference: torch.Tensor,
    *,
    pin_memory: bool,
) -> torch.Tensor:
    """[ARM-WAR-R1-PINNED-INDEPENDENT 2026-07-12] fresh pinned CPU 镜像克隆。

    E1 后向 WAR 根修(组X 案):常驻 pinned 镜像的全量异步 H2D 可能仍未决
    (整建步 enqueue 深埋在在途 forward 之后,未决窗 ms 级),下一步 delta 的
    无条件 host 行写若打在同一缓冲=在途 H2D 读到"未来值"。每次 delta 改写
    前克隆到独立 fresh pinned 分配,旧块由 torch CachingHostAllocator 事件
    护栏保到未决 H2D 完成,WAR 窗物理消灭。clone 同时保全 CPU 镜像残留语义
    (delta 只写行 2-5,行 0/1 靠整建残留)。代价=一次尺寸桶分配+≤768B
    CPU memcpy(µs 级)。
    """
    if pin_memory:
        try:
            fresh = torch.empty(
                tuple(reference.shape),
                device="cpu",
                dtype=reference.dtype,
                pin_memory=True,
            )
        except RuntimeError:
            fresh = torch.empty(
                tuple(reference.shape), device="cpu", dtype=reference.dtype
            )
    else:
        fresh = torch.empty(
            tuple(reference.shape), device="cpu", dtype=reference.dtype
        )
    fresh.copy_(reference)
    return fresh


def compile_launch_template(
    plan: CompactRecentLaunchPlan,
    *,
    descriptor_cpu_i32: torch.Tensor,
    descriptor_gpu_i32: torch.Tensor,
) -> LaunchTemplate:
    batch_size = int(plan.batch_size)
    _validate_descriptor("descriptor_cpu_i32", descriptor_cpu_i32, batch_size)
    _validate_descriptor("descriptor_gpu_i32", descriptor_gpu_i32, batch_size)
    return LaunchTemplate(
        plan=plan,
        descriptor_cpu_i32=descriptor_cpu_i32,
        descriptor_gpu_i32=descriptor_gpu_i32,
        slot_signature=tuple(int(v) for v in plan.slot_signature[:batch_size]),
        use_compact_signature=tuple(
            int(v) for v in plan.use_compact_signature[:batch_size]
        ),
        compact_meta_epoch=int(plan.compact_meta_epoch),
        max_seqlen_k_capacity=int(plan.max_seqlen_k),
    )


def apply_launch_template_row_delta(
    template: LaunchTemplate,
    *,
    request_recent_len_by_row: Tuple[int, ...],
    launch_effective_k_by_row: Tuple[int, ...],
    recent_first_page_by_row: Tuple[int, ...],
    recent_page_count_by_row: Tuple[int, ...],
    update_gpu: bool = True,
    force_gpu_refresh: bool = False,
    controller: object | None = None,
) -> LaunchTemplateUpdateResult:
    plan = template.plan
    batch_size = int(plan.batch_size)
    request_recent_len = _tuple_from_rows(
        "request_recent_len_by_row", request_recent_len_by_row, batch_size
    )
    launch_effective_k = _tuple_from_rows(
        "launch_effective_k_by_row", launch_effective_k_by_row, batch_size
    )
    recent_first = _tuple_from_rows(
        "recent_first_page_by_row", recent_first_page_by_row, batch_size
    )
    recent_count = _tuple_from_rows(
        "recent_page_count_by_row", recent_page_count_by_row, batch_size
    )

    if max(launch_effective_k, default=0) > int(template.max_seqlen_k_capacity):
        return LaunchTemplateUpdateResult(
            requires_recompile=True,
            reason="max_seqlen_k_capacity_exceeded",
            updated_rows=tuple(),
            carrier_update_kernel_count=0,
        )

    updated_rows = tuple(
        row
        for row in range(batch_size)
        if (
            _row_changed(plan.request_recent_len_cpu, row, request_recent_len[row])
            or _row_changed(plan.launch_effective_k_len_cpu, row, launch_effective_k[row])
            or _row_changed(plan.recent_first_cpu, row, recent_first[row])
            or _row_changed(plan.recent_count_cpu, row, recent_count[row])
        )
    )
    gpu_refresh_requested = bool(update_gpu) and bool(force_gpu_refresh)
    if not updated_rows and not gpu_refresh_requested:
        return LaunchTemplateUpdateResult(
            requires_recompile=False,
            reason="hit",
            updated_rows=tuple(),
            carrier_update_kernel_count=0,
        )

    recent_window_changed = any(
        _row_changed(plan.recent_first_cpu, row, recent_first[row])
        or _row_changed(plan.recent_count_cpu, row, recent_count[row])
        for row in range(batch_size)
    )
    non_blocking = template.descriptor_gpu_i32.device.type == "cuda"
    # [ARM-WAR-R1-PINNED-INDEPENDENT 2026-07-12] host 行写打在独立 fresh
    # 克隆上而非常驻镜像(E1 后向 WAR 根修,机理见 _fresh_descriptor_cpu_mirror)。
    descriptor_cpu = _fresh_descriptor_cpu_mirror(
        template.descriptor_cpu_i32,
        pin_memory=non_blocking,
    )
    if bool(recent_window_changed):
        for row in range(batch_size):
            descriptor_cpu[2, row] = recent_first[row]
            descriptor_cpu[3, row] = recent_count[row]
            descriptor_cpu[4, row] = request_recent_len[row]
            descriptor_cpu[5, row] = launch_effective_k[row]
    else:
        for row in range(batch_size):
            descriptor_cpu[4, row] = request_recent_len[row]
            descriptor_cpu[5, row] = launch_effective_k[row]

    if bool(update_gpu):
        descriptor_row_start = 2 if bool(recent_window_changed) else 4
        template.descriptor_gpu_i32[descriptor_row_start:6, :batch_size].copy_(
            descriptor_cpu[descriptor_row_start:6, :batch_size],
            non_blocking=non_blocking,
        )

    # 双引用替换:template 镜像与 controller 常驻属性同步指向 fresh,保持
    # 「controller 属性=活镜像」不变量(metadata_builder 收尾处的模板身份
    # 三判 launch_template.descriptor_cpu_i32 is descriptor_cpu_i32 依赖它,
    # 否则每 delta 步强制 recompile 并装回陈旧镜像;取证 dump 的 host 面
    # 快照也读 controller 属性)。
    template.descriptor_cpu_i32 = descriptor_cpu
    if controller is not None:
        setattr(
            controller,
            "_compact_recent_launch_plan_descriptor_cpu_i32",
            descriptor_cpu,
        )

    plan.request_recent_len_cpu = request_recent_len
    plan.recent_first_cpu = recent_first
    plan.recent_count_cpu = recent_count
    plan.launch_effective_k_len_cpu = launch_effective_k

    return LaunchTemplateUpdateResult(
        requires_recompile=False,
        reason=(
            "row_delta_applied"
            if bool(updated_rows) and bool(update_gpu)
            else "row_delta_applied_cpu_only"
            if bool(updated_rows)
            else "gpu_refreshed"
        ),
        updated_rows=updated_rows,
        carrier_update_kernel_count=1 if bool(update_gpu) and non_blocking else 0,
    )
