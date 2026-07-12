"""Producer for CompactRecentLaunchPlan.

OWNS: step-level single-shot construction of the 10 compact_recent launch
descriptors. Invoked once per decode step from
maybe_build_step_decode_data_from_metadata_impl, after StepBoundMeta is
populated. All inputs are step-level; outputs are shared across all layers.

DEPENDS_ON:
  - patches.sparse_types.CompactRecentLaunchPlan
  - patches.fa_sparse_runtime.compact_recent_route_authority
      .resolve_compact_recent_rail_mode
  - patches.fa3_native.compact_recent_contract.build_compact_recent_host_plan_i32

See docs/superpowers/specs/2026-04-24-compact-recent-launch-plan-design.md §4.3.
"""
from __future__ import annotations

from typing import Optional

import torch

from patches.sparse_types import CompactRecentLaunchPlan


_COMPACT_RECENT_K_BLOCK_N = 112  # FA3 sm80 compact_recent kernel contract.
_LAUNCH_EFFECTIVE_K_LEN_DESCRIPTOR_ROW = 5


def _compact_valid_alignment_tokens(
    *,
    canonical_state: object,
    page_size: int,
) -> int:
    if getattr(canonical_state, "compact_page_residency", None) is not None:
        return int(page_size)
    return _COMPACT_RECENT_K_BLOCK_N


def _int_attr_or_default(obj: object, name: str, default: int) -> int:
    value = getattr(obj, name, default)
    return int(value) if isinstance(value, int) else int(default)


def _new_cpu_tensor(
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    pin_memory: bool,
) -> torch.Tensor:
    if pin_memory:
        try:
            return torch.empty(shape, device="cpu", dtype=dtype, pin_memory=True)
        except RuntimeError:
            pass
    return torch.empty(shape, device="cpu", dtype=dtype)


def _ensure_cached_plan_buffers(
    controller: object,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return reusable launch-plan buffers.

    Tiny per-step descriptor tensors sit on the decode hot path. Reusing their
    storage avoids cudaMalloc / tensor construction overhead while keeping the
    existing CompactRecentLaunchPlan surface unchanged.
    """
    batch_size_i = max(1, int(batch_size))
    capacity = int(getattr(controller, "_compact_recent_launch_plan_capacity", 0) or 0)
    if capacity < batch_size_i:
        capacity = max(batch_size_i, capacity * 2, 8)
        setattr(controller, "_compact_recent_launch_plan_capacity", int(capacity))

    pin_memory = str(device.type) == "cuda"

    descriptor_cpu = getattr(controller, "_compact_recent_launch_plan_descriptor_cpu_i32", None)
    if (
        not isinstance(descriptor_cpu, torch.Tensor)
        or descriptor_cpu.device.type != "cpu"
        or descriptor_cpu.dtype != torch.int32
        or tuple(descriptor_cpu.shape) != (6, capacity)
    ):
        descriptor_cpu = _new_cpu_tensor(
            (6, capacity),
            dtype=torch.int32,
            pin_memory=pin_memory,
        )
        setattr(controller, "_compact_recent_launch_plan_descriptor_cpu_i32", descriptor_cpu)

    host_cpu = getattr(controller, "_compact_recent_launch_plan_host_cpu_i32", None)
    if (
        not isinstance(host_cpu, torch.Tensor)
        or host_cpu.device.type != "cpu"
        or host_cpu.dtype != torch.int32
        or tuple(host_cpu.shape) != (capacity, 5)
    ):
        host_cpu = _new_cpu_tensor(
            (capacity, 5),
            dtype=torch.int32,
            pin_memory=False,
        )
        setattr(controller, "_compact_recent_launch_plan_host_cpu_i32", host_cpu)

    offset_cpu = getattr(controller, "_compact_recent_launch_plan_offset_cpu_i64", None)
    if (
        not isinstance(offset_cpu, torch.Tensor)
        or offset_cpu.device.type != "cpu"
        or offset_cpu.dtype != torch.int64
        or tuple(offset_cpu.shape) != (capacity,)
    ):
        offset_cpu = _new_cpu_tensor(
            (capacity,),
            dtype=torch.int64,
            pin_memory=pin_memory,
        )
        setattr(controller, "_compact_recent_launch_plan_offset_cpu_i64", offset_cpu)

    descriptor_gpu = getattr(controller, "_compact_recent_launch_plan_descriptor_i32", None)
    if (
        not isinstance(descriptor_gpu, torch.Tensor)
        or descriptor_gpu.device != device
        or descriptor_gpu.dtype != torch.int32
        or tuple(descriptor_gpu.shape) != (6, capacity)
    ):
        descriptor_gpu = torch.empty((6, capacity), device=device, dtype=torch.int32)
        setattr(controller, "_compact_recent_launch_plan_descriptor_i32", descriptor_gpu)

    offset_gpu = getattr(controller, "_compact_recent_launch_plan_offset_i64", None)
    if (
        not isinstance(offset_gpu, torch.Tensor)
        or offset_gpu.device != device
        or offset_gpu.dtype != torch.int64
        or tuple(offset_gpu.shape) != (capacity,)
    ):
        offset_gpu = torch.empty((capacity,), device=device, dtype=torch.int64)
        setattr(controller, "_compact_recent_launch_plan_offset_i64", offset_gpu)

    return descriptor_cpu, host_cpu, offset_cpu, descriptor_gpu, offset_gpu


def ensure_cached_launch_effective_k_len_i32(
    controller: object,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Return graph-stable launch-effective K length storage for a batch."""
    (
        _descriptor_cpu,
        _host_cpu,
        _offset_cpu,
        descriptor_gpu,
        _offset_gpu,
    ) = _ensure_cached_plan_buffers(
        controller,
        batch_size=batch_size,
        device=device,
    )
    return descriptor_gpu[
        _LAUNCH_EFFECTIVE_K_LEN_DESCRIPTOR_ROW,
        : int(batch_size),
    ]


def build_compact_recent_launch_plan(
    *,
    controller: object,
    step_authority: object,
    step_bound_meta: object,
    page_size: int,
    device: torch.device,
    canonical_state: object | None = None,
    allow_full_kv_handoff: bool = False,
) -> Optional[CompactRecentLaunchPlan]:
    """Construct the step-level launch plan.

    Returns None if batch_size == 0 or required inputs are absent. Returns a
    plan with valid=False when the rail mode is NO_COMPACT, unless the caller
    explicitly requests the full-KV handoff route. That handoff route still
    uses the mixed-page/RRP carrier family; it just encodes every row as a
    full-recent row with compact_valid_tokens=0.
    """
    # Late imports to avoid circular deps at module load time.
    from patches.fa_sparse_runtime.compact_recent_route_authority import (
        CompactRecentRailMode,
        resolve_compact_recent_rail_mode,
    )

    batch_size = int(getattr(step_authority, "batch_size", 0))
    if batch_size <= 0:
        return None

    # --- NO_COMPACT rail early exit ---
    decision = resolve_compact_recent_rail_mode(step_authority)
    if (
        decision.mode is CompactRecentRailMode.NO_COMPACT
        and not bool(allow_full_kv_handoff)
    ):
        return _no_compact_plan(batch_size=batch_size, device=device)
    capture_row_set = set(int(row) for row in getattr(decision, "capture_rows", tuple()))

    # --- page_size passed in directly (scheduler stage has no kv_cache tensor) ---
    page_size = int(page_size)
    if page_size <= 0:
        return None

    # --- Canonical layer 0 (spec §4.3 v1.1 decision): cross-layer content is
    # identical, layer 0 is the simplest stable choice; regression test
    # test_plan_values_match_legacy_per_layer_dispatch_math guards the invariant.
    if canonical_state is None:
        layer_states = getattr(controller, "layer_states", None)
        if not layer_states:
            return None
        try:
            first_key = sorted(layer_states.keys())[0]
        except Exception:
            return None
        canonical_state = layer_states[first_key]

    # --- M5 Part B1: read from authoritative per-slot lists instead of
    # per-row CPU tensor carriers (which Part C deletes). Slot mapping
    # resolved via step_authority.slot_by_row (single source of truth). ---
    compact_kv_len_list = getattr(canonical_state, "compact_kv_len", None)
    compact_offset_tokens_list = getattr(canonical_state, "compact_offset_tokens", None)
    if not isinstance(compact_kv_len_list, list):
        compact_kv_len_list = []
    if not isinstance(compact_offset_tokens_list, list):
        compact_offset_tokens_list = []

    # --- Step-level inputs from step_authority / step_bound_meta ---
    use_compact_by_row = tuple(
        bool(v) for v in getattr(step_authority, "use_compact_by_row", ())
    )
    context_kv_len_by_row = tuple(
        int(v) for v in getattr(step_authority, "context_kv_len_by_row", ())
    )
    canonical_real_kv_len_cpu = tuple(
        int(v) for v in getattr(step_bound_meta, "canonical_real_kv_len_cpu", ())
    )
    real_kv_len_by_row = (
        canonical_real_kv_len_cpu
        if len(canonical_real_kv_len_cpu) >= batch_size
        else context_kv_len_by_row
    )
    slot_by_row = tuple(
        int(v) for v in getattr(step_authority, "slot_by_row", ())
    )
    is_prefill_by_row = tuple(
        bool(v) for v in getattr(step_authority, "is_prefill_by_row", ())
    )
    recent_first_tuple = tuple(
        int(v) for v in getattr(step_bound_meta, "request_recent_first_logical_page", ())
    )
    recent_count_tuple = tuple(
        int(v) for v in getattr(step_bound_meta, "request_recent_page_count", ())
    )
    if len(recent_first_tuple) < batch_size or len(recent_count_tuple) < batch_size:
        from patches.fa_sparse_runtime.runtime_cache import ensure_step_recent_descriptors

        layer_effective_refresh_by_row = tuple(
            bool(v)
            for v in getattr(step_authority, "layer_effective_refresh_by_row", ())
        )
        if len(layer_effective_refresh_by_row) < batch_size:
            layer_effective_refresh_by_row = (False,) * batch_size
        ensure_step_recent_descriptors(
            step_meta=step_bound_meta,
            page_size=page_size,
            layer_effective_refresh_by_row=layer_effective_refresh_by_row,
        )
        recent_first_tuple = tuple(
            int(v) for v in getattr(step_bound_meta, "request_recent_first_logical_page", ())
        )
        recent_count_tuple = tuple(
            int(v) for v in getattr(step_bound_meta, "request_recent_page_count", ())
        )

    if (
        len(use_compact_by_row) < batch_size
        or len(real_kv_len_by_row) < batch_size
        or len(slot_by_row) < batch_size
        or len(recent_first_tuple) < batch_size
        or len(recent_count_tuple) < batch_size
    ):
        return None
    compact_alignment = _compact_valid_alignment_tokens(
        canonical_state=canonical_state,
        page_size=page_size,
    )

    # --- Build CPU arrays on the fly (1 pass over batch) ---
    compact_base_block_list: list[int] = []
    compact_valid_tokens_list: list[int] = []
    compact_offset_tokens_gathered: list[int] = []  # int64 raw token offsets
    recent_first_list: list[int] = []
    recent_count_list: list[int] = []
    request_recent_len_list: list[int] = []
    launch_effective_k_len_list: list[int] = []
    # Host plan field 4 is a scheduler/policy upper bound over the covered
    # recent pages, not the exact launch seqused_k. The kernel's logical K is
    # compact_valid_tokens_i32 + request_recent_len_i32.
    host_static_k_bound_list: list[int] = []
    requires_compact_kv = False

    n_slots_kv = len(compact_kv_len_list)
    n_slots_off = len(compact_offset_tokens_list)
    # [T3-PLAN-TAIL-DIET 2026-07-10] 主循环内 int(page_size) 每行重转 ×5 提环外。
    page_size_i = int(page_size)

    for row in range(batch_size):
        real = max(0, int(real_kv_len_by_row[row]))
        first = max(0, int(recent_first_tuple[row]))
        count = max(0, int(recent_count_tuple[row]))
        slot = int(slot_by_row[row])
        is_prefill_row = bool(is_prefill_by_row[row]) if row < len(is_prefill_by_row) else False
        compact_row = (
            bool(use_compact_by_row[row])
            and not is_prefill_row
            and row not in capture_row_set
        )
        if compact_row:
            if not (0 <= slot < n_slots_kv) or slot >= n_slots_off:
                raise RuntimeError(
                    "compact_recent launch plan selected compact row without "
                    f"compact slot metadata: row={row} slot={slot} "
                    f"slots={n_slots_kv} offsets={n_slots_off}"
                )
            kv_len_raw = max(0, int(compact_kv_len_list[slot]))
            kv_len_aligned = (kv_len_raw // compact_alignment) * compact_alignment
            if kv_len_aligned > 0 and slot < n_slots_off:
                requires_compact_kv = True
                offset_tokens = max(0, int(compact_offset_tokens_list[slot]))
                base_block = offset_tokens // page_size_i
                visible_recent = min(
                    max(0, real - first * page_size_i),
                    count * page_size_i,
                )
                host_static_k_bound = kv_len_aligned + count * page_size_i
            else:
                if kv_len_raw <= 0:
                    raise RuntimeError(
                        "compact_recent launch plan compact row not ready: "
                        f"row={row} slot={slot} compact_kv_len={kv_len_raw} "
                        f"aligned={kv_len_aligned}"
                    )
                compact_row = False
        if not compact_row:
            # Full-KV row: recent covers the entire real_kv_len. This also
            # covers compact rows whose slot is not ready yet; the consumer
            # must keep making progress with a full-recent launch.
            kv_len_aligned = 0
            base_block = 0
            offset_tokens = 0
            first = 0
            count = (real + page_size_i - 1) // page_size_i
            visible_recent = real
            host_static_k_bound = real

        compact_base_block_list.append(base_block)
        compact_valid_tokens_list.append(kv_len_aligned)
        compact_offset_tokens_gathered.append(offset_tokens)
        recent_first_list.append(first)
        recent_count_list.append(count)
        request_recent_len_list.append(visible_recent)
        launch_effective_k_len_list.append(kv_len_aligned + visible_recent)
        host_static_k_bound_list.append(host_static_k_bound)

    (
        descriptor_cpu_storage,
        host_cpu_storage,
        offset_cpu_storage,
        descriptor_gpu_storage,
        offset_gpu_storage,
    ) = _ensure_cached_plan_buffers(
        controller,
        batch_size=batch_size,
        device=device,
    )
    # [ARM-WAR-R1-PINNED-INDEPENDENT 2026-07-12] E1/E2/E3 根修(组X 案定谳:
    # 时序敏感结构性异步缺边)。controller 常驻 pinned 单例 descriptor/offset
    # 存在「异步 H2D 未决 ←→ 相邻步 host 复写」WAR 族:cudaMemcpyAsync 读源
    # 发生在 GPU 执行时刻而非 enqueue 时刻,host 领先窗内对同一 pinned 源的
    # 复写会让在途 H2D 读到"未来值"。改为每次整建独立 fresh pinned 分配并
    # 替换常驻引用:torch CachingHostAllocator 在 pinned 块释放时按使用流
    # record event、复用前查询完成——旧块在其未决 H2D 完成前不会被发回,
    # WAR 窗物理消灭。零事件零 sync 零旋钮;分配走尺寸桶稳态命中(µs 级),
    # memcpy ≤2KB。先例=rrp_row_table_manager [SEQUSED-STAGING-INDEPENDENT
    # 2026-07-07](同族第六处)。host_plan(pageable、无 H2D)不在族内。
    # 形状合同不变:(6,capacity)/(capacity,) 同 _ensure_cached_plan_buffers。
    pin_memory = str(device.type) == "cuda"
    descriptor_cpu_storage = _new_cpu_tensor(
        tuple(descriptor_cpu_storage.shape),
        dtype=torch.int32,
        pin_memory=pin_memory,
    )
    offset_cpu_storage = _new_cpu_tensor(
        tuple(offset_cpu_storage.shape),
        dtype=torch.int64,
        pin_memory=pin_memory,
    )
    setattr(
        controller,
        "_compact_recent_launch_plan_descriptor_cpu_i32",
        descriptor_cpu_storage,
    )
    setattr(
        controller,
        "_compact_recent_launch_plan_offset_cpu_i64",
        offset_cpu_storage,
    )
    descriptor_cpu = descriptor_cpu_storage[:, :batch_size]
    host_plan_i32 = host_cpu_storage[:batch_size, :]
    compact_offset_cpu = offset_cpu_storage[:batch_size]
    # [LAUNCH-PLAN-BATCH-WRITE 2026-07-09] 原逐行×逐字段 tensor __setitem__
    # (11 标量写/行,bs8 实测 416µs=launch_plan_build 0.67ms 的主项)改为
    # list→tensor 批量物化+copy_(实测 23µs)。上方第一趟循环出口已保证全部
    # 值非负(compact 臂 max(0,·)/对齐截断,full 臂由 real/count 推导),原
    # 第二趟的 max(0,·) 为冗余防御,批量化语义逐位等价;int32 溢出时
    # torch.tensor 与原标量写同样抛错,无静默截断通道。
    descriptor_src = torch.tensor(
        [
            compact_base_block_list,
            compact_valid_tokens_list,
            recent_first_list,
            recent_count_list,
            request_recent_len_list,
            launch_effective_k_len_list,
        ],
        dtype=torch.int32,
    )
    descriptor_cpu.copy_(descriptor_src)
    host_plan_i32[:, 0:4].copy_(descriptor_src[0:4].t())
    host_plan_i32[:, 4].copy_(
        torch.tensor(host_static_k_bound_list, dtype=torch.int32)
    )
    compact_offset_cpu.copy_(
        torch.tensor(compact_offset_tokens_gathered, dtype=torch.int64)
    )

    non_blocking = str(device.type) == "cuda"
    descriptor_i32 = descriptor_gpu_storage[:, :batch_size]
    descriptor_i32.copy_(descriptor_cpu, non_blocking=non_blocking)
    compact_offset_tokens_i64 = offset_gpu_storage[:batch_size]
    compact_offset_tokens_i64.copy_(compact_offset_cpu, non_blocking=non_blocking)
    compact_base_block_i32 = descriptor_i32[0]
    compact_valid_tokens_i32 = descriptor_i32[1]
    recent_first_i32 = descriptor_i32[2]
    recent_count_i32 = descriptor_i32[3]
    request_recent_len_i32 = descriptor_i32[4]
    launch_effective_k_len_i32 = descriptor_i32[_LAUNCH_EFFECTIVE_K_LEN_DESCRIPTOR_ROW]

    # --- max_seqlen_k: compact_recent scheduler shell uses the static K bound ---
    max_seqlen_k = max(1, max(host_static_k_bound_list, default=1))

    plan = CompactRecentLaunchPlan(
        compact_base_block_i32=compact_base_block_i32,
        compact_valid_tokens_i32=compact_valid_tokens_i32,
        recent_first_i32=recent_first_i32,
        recent_count_i32=recent_count_i32,
        request_recent_len_i32=request_recent_len_i32,
        launch_effective_k_len_i32=launch_effective_k_len_i32,
        compact_offset_tokens_i64=compact_offset_tokens_i64,
        host_plan_i32=host_plan_i32,
        page_size=page_size,
        batch_size=batch_size,
        max_seqlen_k=max_seqlen_k,
        valid=True,
        step_identity_token=_int_attr_or_default(step_bound_meta, "step_identity_token", 0),
        req_set_hash=_int_attr_or_default(step_bound_meta, "req_set_hash", 0),
        row_phase_hash=_int_attr_or_default(step_bound_meta, "row_phase_hash", 0),
        # [T3-PLAN-TAIL-DIET 2026-07-10] 尾段 12 趟 per-row 逐元素重扫全为同值
        # 重做:主循环出口的 8 个 list 元素已是纯 Python int(int()/整型算术
        # 产出,LAUNCH-PLAN-BATCH-WRITE 注同一论证),头部输入元组已 int/bool
        # 规范化且长度恰为主循环行数——tuple(list) 一次 C 层拷贝逐位等价。
        slot_signature=slot_by_row[:batch_size],
        use_compact_signature=tuple(
            1 if v else 0 for v in use_compact_by_row[:batch_size]
        ),
        compact_meta_epoch=_int_attr_or_default(canonical_state, "compact_meta_epoch", -1),
        compact_valid_tokens_cpu=tuple(compact_valid_tokens_list),
        compact_offset_tokens_cpu=tuple(compact_offset_tokens_gathered),
        recent_first_cpu=tuple(recent_first_list),
        recent_count_cpu=tuple(recent_count_list),
        request_recent_len_cpu=tuple(request_recent_len_list),
        launch_effective_k_len_cpu=tuple(launch_effective_k_len_list),
        full_recent_only=all(
            v == 0 and f == 0
            for v, f in zip(compact_valid_tokens_list, recent_first_list)
        ),
        requires_compact_kv=bool(requires_compact_kv),
    )
    plan.real_kv_len_cpu = tuple(real_kv_len_by_row[:batch_size])
    plan._descriptor_i32 = descriptor_i32
    return plan


def _no_compact_plan(*, batch_size: int, device: torch.device) -> CompactRecentLaunchPlan:
    """Return a plan with valid=False. Consumer short-circuits on `not plan.valid`.

    All tensor fields get zero-sized placeholders to satisfy the dataclass
    typing; they are never read when valid=False.
    """
    empty_i32 = torch.zeros((batch_size,), dtype=torch.int32, device=device)
    empty_i64 = torch.zeros((batch_size,), dtype=torch.int64, device=device)
    empty_host = torch.zeros((batch_size, 1), dtype=torch.int32)  # CPU
    return CompactRecentLaunchPlan(
        compact_base_block_i32=empty_i32,
        compact_valid_tokens_i32=empty_i32,
        recent_first_i32=empty_i32,
        recent_count_i32=empty_i32,
        request_recent_len_i32=empty_i32,
        launch_effective_k_len_i32=empty_i32,
        compact_offset_tokens_i64=empty_i64,
        host_plan_i32=empty_host,
        page_size=0,
        batch_size=batch_size,
        max_seqlen_k=0,
        valid=False,
        compact_valid_tokens_cpu=tuple(),
        recent_first_cpu=tuple(),
        recent_count_cpu=tuple(),
        request_recent_len_cpu=tuple(),
        launch_effective_k_len_cpu=tuple(),
        requires_compact_kv=False,
    )
