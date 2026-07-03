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
                base_block = offset_tokens // int(page_size)
                visible_recent = min(
                    max(0, real - first * int(page_size)),
                    count * int(page_size),
                )
                host_static_k_bound = kv_len_aligned + count * int(page_size)
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
            count = (real + int(page_size) - 1) // int(page_size)
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
    descriptor_cpu = descriptor_cpu_storage[:, :batch_size]
    host_plan_i32 = host_cpu_storage[:batch_size, :]
    compact_offset_cpu = offset_cpu_storage[:batch_size]
    for row in range(batch_size):
        compact_base = int(compact_base_block_list[row])
        compact_tokens = max(0, int(compact_valid_tokens_list[row]))
        recent_first = max(0, int(recent_first_list[row]))
        recent_count = max(0, int(recent_count_list[row]))
        request_recent_len = max(0, int(request_recent_len_list[row]))
        launch_effective_k = max(0, int(launch_effective_k_len_list[row]))
        host_effective_k = max(0, int(host_static_k_bound_list[row]))
        descriptor_cpu[0, row] = compact_base
        descriptor_cpu[1, row] = compact_tokens
        descriptor_cpu[2, row] = recent_first
        descriptor_cpu[3, row] = recent_count
        descriptor_cpu[4, row] = request_recent_len
        descriptor_cpu[_LAUNCH_EFFECTIVE_K_LEN_DESCRIPTOR_ROW, row] = launch_effective_k
        host_plan_i32[row, 0] = compact_base
        host_plan_i32[row, 1] = compact_tokens
        host_plan_i32[row, 2] = recent_first
        host_plan_i32[row, 3] = recent_count
        host_plan_i32[row, 4] = host_effective_k
        compact_offset_cpu[row] = max(0, int(compact_offset_tokens_gathered[row]))

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
        slot_signature=tuple(int(slot_by_row[row]) for row in range(batch_size)),
        use_compact_signature=tuple(
            1 if bool(use_compact_by_row[row]) else 0 for row in range(batch_size)
        ),
        compact_meta_epoch=_int_attr_or_default(canonical_state, "compact_meta_epoch", -1),
        compact_valid_tokens_cpu=tuple(
            int(v) for v in compact_valid_tokens_list[:batch_size]
        ),
        compact_offset_tokens_cpu=tuple(
            int(v) for v in compact_offset_tokens_gathered[:batch_size]
        ),
        recent_first_cpu=tuple(int(v) for v in recent_first_list[:batch_size]),
        recent_count_cpu=tuple(int(v) for v in recent_count_list[:batch_size]),
        request_recent_len_cpu=tuple(
            int(v) for v in request_recent_len_list[:batch_size]
        ),
        launch_effective_k_len_cpu=tuple(
            int(v) for v in launch_effective_k_len_list[:batch_size]
        ),
        full_recent_only=all(
            int(compact_valid_tokens_list[row]) == 0 and int(recent_first_list[row]) == 0
            for row in range(batch_size)
        ),
        requires_compact_kv=bool(requires_compact_kv),
    )
    plan.real_kv_len_cpu = tuple(int(v) for v in real_kv_len_by_row[:batch_size])
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
