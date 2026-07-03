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
    descriptor_cpu = template.descriptor_cpu_i32
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

    non_blocking = template.descriptor_gpu_i32.device.type == "cuda"
    if bool(update_gpu):
        descriptor_row_start = 2 if bool(recent_window_changed) else 4
        template.descriptor_gpu_i32[descriptor_row_start:6, :batch_size].copy_(
            descriptor_cpu[descriptor_row_start:6, :batch_size],
            non_blocking=non_blocking,
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
