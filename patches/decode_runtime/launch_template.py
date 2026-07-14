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
    # ``update_gpu=False`` advances only the plan's canonical CPU tuples.  The
    # pinned descriptor is an H2D staging image, not a second source of truth;
    # leave it untouched while a prior async copy may still consume it and
    # rebuild all dynamic rows on the next requested GPU publication.
    descriptor_dynamic_rows_stale: bool = False


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
    (整建步 enqueue 深埋在在途 forward 之后,未决窗 ms 级),下一次真正发布
    GPU descriptor 时若原地 host 行写=在途 H2D 读到"未来值"。因此每次
    GPU-publishing delta 先克隆到独立 fresh pinned 分配,旧块由 torch
    CachingHostAllocator 事件护栏保到未决 H2D 完成。CPU-only delta 不触碰
    staging mirror，只更新 plan tuple truth，并由 stale 标志要求下次发布重建
    rows 2-5。clone 同时保全静态行 0/1。代价只由真正 H2D 的步骤承担。
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
    # A CPU-only delta deliberately leaves the pinned staging descriptor
    # untouched.  Therefore ``stale`` is itself a pending GPU publication,
    # even when the next caller supplies values already equal to plan truth
    # and does not need to force an otherwise redundant refresh.
    gpu_refresh_requested = bool(update_gpu) and bool(
        force_gpu_refresh or template.descriptor_dynamic_rows_stale
    )
    if not updated_rows and not gpu_refresh_requested:
        return LaunchTemplateUpdateResult(
            requires_recompile=False,
            reason="hit",
            updated_rows=tuple(),
            carrier_update_kernel_count=0,
        )

    if not bool(update_gpu):
        # The RRP ``arena_batch_seqused`` route does not consume the launch
        # descriptor on same-page steps.  Updating its pinned mirror here used
        # to allocate and clone a fresh buffer every step even though no H2D
        # followed.  Keep the four CPU tuples as the single canonical truth
        # and preserve the staging image byte-for-byte; this is both cheaper
        # and stronger than manufacturing another buffer to avoid host/H2D
        # write-after-read.  A later GPU publication reconstructs rows 2..5.
        plan.request_recent_len_cpu = request_recent_len
        plan.recent_first_cpu = recent_first
        plan.recent_count_cpu = recent_count
        plan.launch_effective_k_len_cpu = launch_effective_k
        template.descriptor_dynamic_rows_stale = True
        return LaunchTemplateUpdateResult(
            requires_recompile=False,
            reason="row_delta_applied_cpu_only",
            updated_rows=updated_rows,
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
    # [U14-ZERO-ALLOC-2-1 2026-07-12] 逐行 torch 标量 setitem(每 delta 步
    # 2·bs~4·bs 次,实测 ~4.75µs/次=76µs/步 @bs8,含 replay 步) → numpy 视图
    # 整行向量化赋值(~sub-µs)。descriptor_cpu 是 fresh pinned int32 连续镜像,
    # .numpy() 返回共享存储视图;下方 :192 H2D copy_ 读同一内存 → 语义逐位不变。
    # tuple→int32 slice 赋值域与旧标量路径同(request_recent_len 等已由
    # _tuple_from_rows 走 int() 归一,行/列切片长度均 == batch_size)。
    descriptor_np = descriptor_cpu.numpy()
    refresh_all_dynamic_rows = bool(
        recent_window_changed or template.descriptor_dynamic_rows_stale
    )
    if refresh_all_dynamic_rows:
        descriptor_np[2, :batch_size] = recent_first
        descriptor_np[3, :batch_size] = recent_count
        descriptor_np[4, :batch_size] = request_recent_len
        descriptor_np[5, :batch_size] = launch_effective_k
    else:
        descriptor_np[4, :batch_size] = request_recent_len
        descriptor_np[5, :batch_size] = launch_effective_k

    descriptor_row_start = 2 if refresh_all_dynamic_rows else 4
    template.descriptor_gpu_i32[descriptor_row_start:6, :batch_size].copy_(
        descriptor_cpu[descriptor_row_start:6, :batch_size],
        non_blocking=non_blocking,
    )
    template.descriptor_dynamic_rows_stale = False

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
            if bool(updated_rows)
            else "gpu_refreshed"
        ),
        updated_rows=updated_rows,
        carrier_update_kernel_count=1 if non_blocking else 0,
    )
