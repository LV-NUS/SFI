"""Route compact-page overlay launches through native mixed-page attention."""

from __future__ import annotations

import os
from collections.abc import Iterable

import torch

from patches.fa3_native.mixed_page_graph_descriptor import (
    MixedPageResolverCarrierSet,
    PageResolverKind,
    PageResolverSubkind,
    ResolverGraphDescriptor,
)
from patches.fa3_native.runtime_bridge import validate_mixed_page_resolver_replay
from patches.sparse_constants import _DYNAMIC_ENV
from patches.fa_sparse_runtime.compact_mixed_page_overlay import (
    CompactMixedPageOverlay,
)
from patches.fa_sparse_runtime.resolved_row_ptr_arena import (
    ResolvedRowPtrArena,
    build_planned_compact_row_layout,
)

_ROUTE_TRACE_ENV = "VLLM_SPARSE_FA3_ROUTE_TRACE_LOG"
_OVERLAY_TRACE_ENV = "VLLM_SPARSE_COMPACT_MIXED_PAGE_OVERLAY_TRACE"


def _contains_tensor_value(value: object) -> bool:
    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, dict):
        return any(
            _contains_tensor_value(key) or _contains_tensor_value(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_tensor_value(item) for item in value)
    return False


def _reject_tensor_value_slow(name: str, value: object) -> None:
    if isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be CPU-owned metadata, not torch.Tensor")
    if isinstance(value, dict):
        for index, (key, item) in enumerate(value.items()):
            _reject_tensor_value_slow(f"{name}.key[{index}]", key)
            _reject_tensor_value_slow(f"{name}.value[{index}]", item)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_tensor_value_slow(f"{name}[{index}]", item)


def _reject_tensor_value(name: str, value: object) -> None:
    # [E2 残余] f-string 惰性化：成功路径（每 layer 大量元素）零字符串构造，
    # 仅命中 tensor 时重扫一遍生成与旧实现逐字相同的精确路径报错。
    if _contains_tensor_value(value):
        _reject_tensor_value_slow(name, value)


def _as_int(name: str, value: object) -> int:
    _reject_tensor_value(name, value)
    return int(value)


def _as_float(name: str, value: object) -> float:
    _reject_tensor_value(name, value)
    return float(value)


def _metadata_tuple(name: str, values: object) -> tuple[object, ...]:
    if values is None:
        return ()
    _reject_tensor_value(name, values)
    materialized = tuple(values)
    _reject_tensor_value(name, materialized)
    return materialized


def _int_tuple(source: object, name: str, batch_size: int) -> tuple[int, ...]:
    metadata = _metadata_tuple(name, getattr(source, name, ()))
    if len(metadata) < batch_size:
        raise RuntimeError(
            f"{name} coverage is insufficient for compact mixed-page overlay"
        )
    return tuple(_as_int(name, value) for value in metadata[:batch_size])


def _as_bool(name: str, value: object) -> bool:
    _reject_tensor_value(name, value)
    if isinstance(value, (dict, list, tuple)):
        raise ValueError(f"{name} must contain scalar CPU-owned metadata")
    return bool(value)


def _optional_int(source: object, name: str, default: int) -> int:
    value = getattr(source, name, None)
    if value is None:
        return int(default)
    return _as_int(name, value)


def _layer_index(state: object) -> int:
    return _optional_int(state, "layer_index", 0)


def _lease(state: object) -> object:
    residency = getattr(state, "compact_page_residency", None)
    lease = getattr(residency, "lease", None)
    if lease is None:
        raise RuntimeError("compact mixed-page overlay requires compact page residency")
    return lease


# [RESERVED-IDS-CACHE] lease.reserved_manager_block_ids 在 lease 生命周期内不可变
# （frozen dataclass，创建时已强制逐元素 int/正/连续），但旧实现每 layer×每 eager
# 步做三遍逐元素递归校验（slots×blocks=1152 元素实测 ~1ms/层×36 层≈47ms/eager 步，
# 是 refresh 等待残差的最大已定位切片）。缓存持有已校验 tuple 的强引用（id 在持有
# 引用期间恒有效），命中 = 一次 dict 查找 + 一次身份比较，O(1)。
_RESERVED_IDS_CACHE: dict[int, tuple[int, ...]] = {}


def _compact_reserved_ids(state: object) -> tuple[int, ...]:
    ids = getattr(_lease(state), "reserved_manager_block_ids", None)
    if ids is None:
        raise RuntimeError("compact mixed-page overlay requires reserved manager blocks")
    cached = _RESERVED_IDS_CACHE.get(id(ids))
    if cached is not None and cached is ids:
        return cached
    validated = tuple(
        _as_int("reserved_manager_block_ids", value)
        for value in _metadata_tuple("reserved_manager_block_ids", ids)
    )
    if type(ids) is tuple and validated == ids:
        # 只缓存与校验结果逐位一致的原 tuple（保引用 → id 稳定）；异类容器每次重验。
        if len(_RESERVED_IDS_CACHE) >= 64:
            _RESERVED_IDS_CACHE.clear()  # lease 数量级为个位数；防御性上界
        _RESERVED_IDS_CACHE[id(ids)] = ids
        return ids
    return validated


def _compact_capacity_pages(state: object) -> int:
    value = _optional_int(_lease(state), "compact_blocks_per_slot", 0)
    if value <= 0:
        raise RuntimeError("compact mixed-page overlay requires compact_blocks_per_slot")
    return value


def _cached_resolved_row_ptr_arena(
    *,
    step_bound_meta: object,
    layer_index: int,
    batch_size: int,
    num_kv_heads: int,
    max_pages_per_row: int,
    device: torch.device,
) -> ResolvedRowPtrArena:
    cache = getattr(step_bound_meta, "resolved_row_ptr_arena_by_layer", None)
    if not isinstance(cache, dict):
        cache = {}
        setattr(step_bound_meta, "resolved_row_ptr_arena_by_layer", cache)
    key_cache = getattr(step_bound_meta, "resolved_row_ptr_arena_key_by_layer", None)
    if not isinstance(key_cache, dict):
        key_cache = {}
        setattr(step_bound_meta, "resolved_row_ptr_arena_key_by_layer", key_cache)

    key = (
        int(batch_size),
        int(num_kv_heads),
        int(max_pages_per_row),
        str(device),
    )
    cached = cache.get(int(layer_index))
    if key_cache.get(int(layer_index)) == key and isinstance(cached, ResolvedRowPtrArena):
        return cached

    arena = ResolvedRowPtrArena.allocate(
        batch_size=int(batch_size),
        num_kv_heads=int(num_kv_heads),
        max_pages_per_row=int(max_pages_per_row),
        device=device,
    )
    cache[int(layer_index)] = arena
    key_cache[int(layer_index)] = key
    return arena


def _env_enabled(name: str) -> bool:
    return bool(os.environ.get(name, "").strip())


# [E5] trace 开关生产默认关且进程内不变：import-time 缓存免每 layer 2 次
# os.environ.get（~0.4µs/层）；pytest/显式动态档走 live 读（tests 有 monkeypatch）。
_OVERLAY_TRACE_ENABLED_CACHED = bool(
    os.environ.get(_OVERLAY_TRACE_ENV, "").strip()
)
_ROUTE_TRACE_ENABLED_CACHED = bool(os.environ.get(_ROUTE_TRACE_ENV, "").strip())


def _overlay_trace_enabled() -> bool:
    if _DYNAMIC_ENV:
        return _env_enabled(_OVERLAY_TRACE_ENV)
    return _OVERLAY_TRACE_ENABLED_CACHED


def _route_trace_enabled() -> bool:
    if _DYNAMIC_ENV:
        return _env_enabled(_ROUTE_TRACE_ENV)
    return _ROUTE_TRACE_ENABLED_CACHED


def _append_trace(step_bound_meta: object, event: dict[str, object]) -> None:
    if not _overlay_trace_enabled():
        return
    old = getattr(step_bound_meta, "compact_mixed_page_overlay_trace", ())
    _reject_tensor_value("compact_mixed_page_overlay_trace", old)
    setattr(
        step_bound_meta,
        "compact_mixed_page_overlay_trace",
        tuple(old) + (dict(event),),
    )


def _append_route_trace_if_enabled(event: dict[str, object]) -> None:
    if not _route_trace_enabled():
        return
    try:
        from patches.fa3_native.install import append_fa3_route_trace

        append_fa3_route_trace(event)
    except Exception:
        pass


def _coerce_flash_attn_version(value: object) -> int | None:
    if value is None:
        return None
    try:
        version = int(value)
    except (TypeError, ValueError):
        return None
    if version <= 0:
        return None
    return version


def _call_flash_attn_version_getter(getter: object) -> int | None:
    if not callable(getter):
        return None
    try:
        return _coerce_flash_attn_version(getter())
    except Exception:
        return None


def _bridge_flash_attn_version(bridge: object) -> int:
    owners = (
        bridge,
        getattr(bridge, "package", None),
        getattr(bridge, "interface_module", None),
    )
    for owner in owners:
        if owner is None:
            continue
        for attr in ("flash_attn_version", "vllm_flash_attn_version", "fa_version"):
            version = _coerce_flash_attn_version(getattr(owner, attr, None))
            if version is not None:
                return version
        version = _call_flash_attn_version_getter(
            getattr(owner, "get_flash_attn_version", None)
        )
        if version is not None:
            return version
    env_version = _coerce_flash_attn_version(os.environ.get("VLLM_FLASH_ATTN_VERSION"))
    if env_version is not None:
        return env_version
    return 3


def _mixed_page_backend_label(fa_version: int) -> str:
    if fa_version == 4:
        return "fa4_cute_sm100"
    if fa_version == 3:
        return "fa3_native"
    return f"fa{fa_version}"


def _wait_for_compact_arena_if_needed(
    *,
    controller: object,
    state: object,
    device: torch.device,
) -> None:
    # The compact-ready generation is a correctness fence, not an optional
    # optimization.  Once a producer advertises a newer generation, failure to
    # establish its stream dependency must abort the launch instead of letting
    # FA3 consume stale arena contents.  This also keeps every TP rank on the
    # same fail-closed contract.
    if getattr(device, "type", "") == "cuda":
        _car_evt = getattr(controller, "_compact_arena_ready_evt", None)
        _car_gen = int(getattr(controller, "_compact_arena_ready_gen", 0))
        _car_waited = int(getattr(controller, "_compact_arena_ready_waited_gen", -1))
        if _car_gen < 0 or _car_waited < -1 or _car_waited > _car_gen:
            raise RuntimeError("compact-ready generation state is malformed")
        if _car_gen > 0 and _car_evt is None:
            raise RuntimeError(
                "compact-ready generation requires its producer event"
            )
        if (
            _car_evt is not None
            and _car_gen > max(0, _car_waited)
            and not torch.cuda.is_current_stream_capturing()
        ):
            torch.cuda.current_stream(device=device).wait_event(_car_evt)
            controller._compact_arena_ready_waited_gen = _car_gen

    from patches.fa_sparse_runtime.compact_recent_dispatch import (
        _maybe_wait_for_async_compact_arena,
    )

    _maybe_wait_for_async_compact_arena(
        controller=controller,
        state=state,
        device=device,
    )


def _recent_capacity_pages(
    recent_page_count_by_row: tuple[int, ...],
    page_size: int,
    max_seqlen_k: int,
    compact_capacity_pages: int,
) -> int:
    recent_count_pages = max(recent_page_count_by_row, default=0)
    max_pages = (max_seqlen_k + page_size - 1) // page_size
    launch_hint_pages = max(0, max_pages - compact_capacity_pages)
    return max(launch_hint_pages, recent_count_pages)


def _window_size(value: object) -> list[int]:
    values = _metadata_tuple("window_size", value)
    if len(values) != 2:
        raise ValueError("window_size must contain exactly two entries")
    return [_as_int("window_size", values[0]), _as_int("window_size", values[1])]


def _batch_effective_seqused_tensor(
    *,
    resolver_seqused_k: torch.Tensor | None,
    step_bound_meta: object,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    candidates = (
        resolver_seqused_k,
        getattr(
            getattr(step_bound_meta, "compact_recent_launch_plan", None),
            "launch_effective_k_len_i32",
            None,
        ),
    )
    for candidate in candidates:
        if not isinstance(candidate, torch.Tensor):
            continue
        if candidate.dtype != torch.int32:
            continue
        if candidate.device != device:
            continue
        if candidate.dim() != 1:
            continue
        if int(candidate.numel()) != int(batch_size):
            continue
        if not candidate.is_contiguous():
            continue
        return candidate
    raise RuntimeError(
        "ResolvedRowPtr mixed-page route requires batch effective seqused_k tensor"
    )


def _descriptor_requests_compact_native(
    descriptor: ResolverGraphDescriptor | None,
) -> bool:
    return (
        descriptor is not None
        and int(descriptor.resolver_kind) == int(PageResolverKind.COMPACT_RECENT_NATIVE)
    )


def _descriptor_requests_resolved_row_ptr(
    descriptor: ResolverGraphDescriptor | None,
) -> bool:
    return (
        descriptor is not None
        and int(descriptor.resolver_kind) == int(PageResolverKind.RESOLVED_ROW_PTR)
    )


def _descriptor_requests_selected_table(
    descriptor: ResolverGraphDescriptor | None,
) -> bool:
    return (
        descriptor is not None
        and int(descriptor.resolver_kind) == int(PageResolverKind.SELECTED_TABLE)
    )


def _descriptor_requests_native_materialized(
    descriptor: ResolverGraphDescriptor | None,
) -> bool:
    return (
        descriptor is not None
        and int(descriptor.resolver_kind) == int(PageResolverKind.NATIVE)
    )


def _descriptor_requests_retired_effective_table(
    descriptor: ResolverGraphDescriptor | None,
) -> bool:
    return (
        descriptor is not None
        and int(descriptor.resolver_kind) == int(PageResolverKind.EFFECTIVE_ROW_TABLE)
    )


def _maybe_cpu_int_tuple_value(value: object, *, label: str) -> tuple[int, ...] | None:
    if value is None or isinstance(value, torch.Tensor):
        return None
    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value)
    try:
        return tuple(int(v) for v in value)  # type: ignore[union-attr]
    except (TypeError, ValueError):
        pass
    raise RuntimeError(f"{label} must be a CPU integer sequence")


def _first_available_cpu_int_tuple_value(
    *,
    values: tuple[object | None, ...],
    label: str,
    batch_size: int,
) -> tuple[int, ...] | None:
    for value in values:
        parsed = _maybe_cpu_int_tuple_value(value, label=label)
        if parsed is not None and len(parsed) >= int(batch_size):
            return tuple(int(v) for v in parsed[: int(batch_size)])
    return None


def _first_int_attr(
    *,
    sources: tuple[object | None, ...],
    names: tuple[str, ...],
    default: int = 0,
) -> int:
    for source in sources:
        if source is None:
            continue
        for name in names:
            try:
                value = getattr(source, name)
            except Exception:
                continue
            if isinstance(value, bool):
                return int(value)
            if isinstance(value, int):
                return int(value)
    return int(default)


def resolve_compact_mixed_page_overlay_cpu_geometry(
    *,
    launch_plan: object,
    step_bound_meta: object,
    step_authority: object,
    real_kv_len_hint: object,
    page_size: int,
    batch_size: int,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    from patches.fa_sparse_runtime.materialize import derive_page_aligned_recent_window

    if int(page_size) <= 0:
        raise RuntimeError("compact mixed-page overlay requires positive page_size")
    compact_valid = tuple(
        int(v) for v in getattr(launch_plan, "compact_valid_tokens_cpu", tuple())
    )
    if len(compact_valid) < int(batch_size):
        raise RuntimeError(
            "compact mixed-page overlay requires CPU compact_valid_tokens coverage"
        )

    real_kv_len = _first_available_cpu_int_tuple_value(
        values=(
            real_kv_len_hint,
            getattr(step_bound_meta, "canonical_real_kv_len_cpu", None),
            getattr(step_authority, "context_kv_len_by_row", None),
        ),
        label="compact_mixed_page_overlay.real_kv_len",
        batch_size=int(batch_size),
    )
    if real_kv_len is None:
        raise RuntimeError(
            "compact mixed-page overlay requires CPU real_kv_len coverage; "
            "refusing implicit GPU sync"
        )

    row_is_compact = tuple(
        bool(v) for v in tuple(getattr(step_authority, "use_compact_by_row", tuple()))
    )
    if len(row_is_compact) < int(batch_size):
        raise RuntimeError(
            "compact mixed-page overlay requires use_compact_by_row coverage"
        )

    recent_cap = _first_int_attr(
        sources=(step_authority, step_bound_meta),
        names=("recent_cap",),
        default=0,
    )
    plan_recent_first = tuple(
        int(v) for v in getattr(launch_plan, "recent_first_cpu", tuple())
    )
    plan_recent_count = tuple(
        int(v) for v in getattr(launch_plan, "recent_count_cpu", tuple())
    )

    recent_first: list[int] = []
    recent_count: list[int] = []
    effective_k: list[int] = []
    for row in range(int(batch_size)):
        real = max(0, int(real_kv_len[row]))
        compact_tokens = max(0, int(compact_valid[row]))
        if bool(row_is_compact[row]) and compact_tokens > 0 and int(recent_cap) > 0:
            window = derive_page_aligned_recent_window(
                real_kv_len=real,
                page_size=int(page_size),
                recent_tokens=int(recent_cap),
            )
            recent_first.append(int(window.first_logical_page))
            recent_count.append(int(window.page_count))
            effective_k.append(compact_tokens + int(window.visible_tokens))
            continue
        if bool(row_is_compact[row]) and compact_tokens > 0:
            if len(plan_recent_first) < int(batch_size) or len(plan_recent_count) < int(batch_size):
                raise RuntimeError(
                    "compact mixed-page overlay requires CPU recent geometry coverage"
                )
            first = max(0, int(plan_recent_first[row]))
            count = max(0, int(plan_recent_count[row]))
            visible_recent = min(
                max(0, real - first * int(page_size)),
                count * int(page_size),
            )
            recent_first.append(first)
            recent_count.append(count)
            effective_k.append(compact_tokens + visible_recent)
            continue
        recent_first.append(0)
        recent_count.append((real + int(page_size) - 1) // int(page_size))
        effective_k.append(real)

    return (
        tuple(int(v) for v in compact_valid[: int(batch_size)]),
        tuple(recent_first),
        tuple(recent_count),
        tuple(effective_k),
    )


def run_compact_mixed_page_overlay_route(
    *,
    bridge: object,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    seqused_k: torch.Tensor,
    max_seqlen_k: int,
    softmax_scale: float | None,
    window_size: tuple[int, int],
    softcap: float,
    block_table: torch.Tensor,
    controller: object,
    state: object,
    step_authority: object,
    step_bound_meta: object,
    page_size: int,
    num_kv_heads: int,
    compact_valid_tokens_by_row: Iterable[int],
    compact_offset_tokens_by_row: Iterable[int],
    recent_first_page_by_row: Iterable[int],
    recent_page_count_by_row: Iterable[int],
    row_effective_k_by_row: Iterable[int],
    safe_page_id: int,
    num_splits: int,
    cp_world_size: int,
    cp_rank: int,
    cp_tot_seqused_k: torch.Tensor | None,
    q_v: torch.Tensor | None,
    q_descale: torch.Tensor | None,
    k_descale: torch.Tensor | None,
    v_descale: torch.Tensor | None,
    s_aux: torch.Tensor | None,
    capture_scores: torch.Tensor | None = None,
    capture_row_index_i32: torch.Tensor | None = None,
    row_capture_last_n_i32: torch.Tensor | None = None,
    scheduler_metadata: object | None = None,
    prefill_active_worklist: bool = False,
    prefill_active_count: int = 0,
    resolver_descriptor: ResolverGraphDescriptor | None = None,
    resolver_carriers: MixedPageResolverCarrierSet | None = None,
    resolver_seqused_k: torch.Tensor | None = None,
    captured_resolver_descriptor: ResolverGraphDescriptor | None = None,
    captured_resolver_pointer_signature: tuple[int | None, ...] | None = None,
    graph_replay_carriers: bool = False,
) -> torch.Tensor:
    if block_table.ndim != 2:
        raise ValueError("block_table must be 2D for compact mixed-page overlay")

    batch_size = int(block_table.shape[0])
    _wait_for_compact_arena_if_needed(
        controller=controller,
        state=state,
        device=q.device,
    )
    page_size_i = _as_int("page_size", page_size)
    num_kv_heads_i = _as_int("num_kv_heads", num_kv_heads)
    max_seqlen_k_i = _as_int("max_seqlen_k", max_seqlen_k)
    compact_capacity_pages = _compact_capacity_pages(state)
    layer_index = _layer_index(state)
    planned_layout = build_planned_compact_row_layout(
        batch_size=batch_size,
        compact_ready_by_batch_row=getattr(step_authority, "use_compact_by_row", ()),
        slot_by_row=getattr(step_authority, "slot_by_row", ()),
        compact_valid_tokens_by_row=compact_valid_tokens_by_row,
        compact_offset_tokens_by_row=compact_offset_tokens_by_row,
        recent_first_page_by_row=recent_first_page_by_row,
        recent_page_count_by_row=recent_page_count_by_row,
        row_effective_k_by_row=row_effective_k_by_row,
    )
    slot_by_row = planned_layout.slot_by_row
    row_is_compact = planned_layout.compact_ready_by_batch_row
    compact_valid_tokens_tuple = planned_layout.compact_valid_tokens_by_row
    compact_offset_tokens_tuple = planned_layout.compact_offset_tokens_by_row
    recent_first_page_tuple = planned_layout.recent_first_page_by_row
    recent_page_count_tuple = planned_layout.recent_page_count_by_row
    row_effective_k_tuple = planned_layout.row_effective_k_by_row
    effective_max_seqlen_k_i = max(1, max(row_effective_k_tuple, default=0))
    safe_page_id_i = _as_int("safe_page_id", safe_page_id)
    recent_capacity_pages_i = _recent_capacity_pages(
        recent_page_count_tuple,
        page_size_i,
        max_seqlen_k_i,
        compact_capacity_pages,
    )

    use_compact_native_resolver = _descriptor_requests_compact_native(resolver_descriptor)
    use_resolved_row_ptr_resolver = _descriptor_requests_resolved_row_ptr(resolver_descriptor)
    use_retired_selected_table_resolver = _descriptor_requests_selected_table(resolver_descriptor)
    use_native_materialized_resolver = _descriptor_requests_native_materialized(resolver_descriptor)
    use_retired_effective_table_resolver = _descriptor_requests_retired_effective_table(resolver_descriptor)
    if use_compact_native_resolver:
        raise ValueError("CompactRecentNative production descriptor is retired; use ResolvedRowPtr row pointers")
    if use_retired_selected_table_resolver:
        raise ValueError("SelectedTable production descriptor is retired; use ResolvedRowPtr row pointers")
    if use_native_materialized_resolver or use_retired_effective_table_resolver:
        raise ValueError("kind3/native materialized production descriptor is retired; use ResolvedRowPtr row pointers")
    if (
        (resolver_carriers is not None or graph_replay_carriers)
        and not use_resolved_row_ptr_resolver
    ):
        raise ValueError(
            "resolver carriers require ResolvedRowPtr production descriptor"
        )
    launch_block_table = block_table
    launch_effective_seqused_k = _batch_effective_seqused_tensor(
        resolver_seqused_k=resolver_seqused_k,
        step_bound_meta=step_bound_meta,
        batch_size=batch_size,
        device=q.device,
    )
    if use_resolved_row_ptr_resolver:
        if resolver_carriers is None:
            raise ValueError(
                "ResolvedRowPtr mixed-page resolver requires resolver_carriers"
            )
        validate_mixed_page_resolver_replay(
            enabled=bool(graph_replay_carriers),
            captured_descriptor=captured_resolver_descriptor,
            replay_descriptor=resolver_descriptor,
            captured_pointer_signature=captured_resolver_pointer_signature,
            replay_carriers=resolver_carriers,
        )
        if cp_world_size != 1:
            raise ValueError("ResolvedRowPtr mixed-page resolver does not support CP in this landing")
        affine_i32 = resolver_carriers.resolved_page_table_affine_i32
        affine_base = resolver_carriers.resolved_page_table_affine_base
        affine_stride = resolver_carriers.resolved_page_table_affine_stride
        affine_direct = bool(resolver_carriers.resolved_page_table_affine_direct)
        affine_cols = int(resolver_carriers.resolved_page_table_affine_cols)
        affine_head_stride = int(resolver_carriers.resolved_page_table_affine_head_stride)
        has_affine_const = affine_base is not None or affine_stride is not None
        has_affine = affine_i32 is not None or has_affine_const
        if affine_i32 is not None:
            # [AFFINE-TENSOR-RETIRED 2026-07-02] see runtime_bridge.py: the device arm
            # is TORCH_CHECK-retired by TU-INSTANCE-DIET; fail fast on the host.
            raise ValueError(
                "AFFINE_TENSOR resolver subkind is retired (TU-INSTANCE-DIET); "
                "production carriers must not set resolved_page_table_affine_i32"
            )
        elif has_affine_const:
            resolver_subkind_i = int(
                PageResolverSubkind.AFFINE_CONST_DIRECT
                if affine_direct
                else PageResolverSubkind.AFFINE_CONST
            )
        else:
            resolver_subkind_i = int(PageResolverSubkind.ROWPTR)
        if has_affine_const and (affine_base is None or affine_stride is None):
            raise ValueError("ResolvedRowPtr mixed-page affine resolver requires base and stride together")
        if affine_direct and not has_affine_const:
            raise ValueError("ResolvedRowPtr mixed-page direct affine resolver requires const base/stride")
        if resolver_carriers.resolved_page_table_row_ptr_u64 is None and not has_affine:
            raise ValueError("ResolvedRowPtr mixed-page resolver requires row-pointer or affine carrier")
        if has_affine_const and resolver_carriers.resolved_page_table_row_ptr_u64 is not None:
            raise ValueError("ResolvedRowPtr mixed-page affine base/stride resolver must not also set row-pointer carrier")
        if resolver_carriers.resolver_visible_seqused_k_by_head_i32 is None:
            raise ValueError("ResolvedRowPtr mixed-page resolver requires resolved visible lengths")
        if resolver_carriers.selected_page_table_i32 is not None:
            raise ValueError("ResolvedRowPtr mixed-page resolver must not set selected_page_table_i32")
        if resolver_carriers.row_consume_mode_i32 is not None and not has_affine:
            raise ValueError("ResolvedRowPtr mixed-page row_consume_mode_i32 requires affine carriers")
        resolver_kwargs = {
            "selected_page_table_i32": None,
            "effective_row_slot_i32": None,
            "resolved_page_table_row_ptr_u64": resolver_carriers.resolved_page_table_row_ptr_u64,
            "resolved_page_table_affine_i32": resolver_carriers.resolved_page_table_affine_i32,
            "resolved_page_table_affine_base": resolver_carriers.resolved_page_table_affine_base,
            "resolved_page_table_affine_stride": resolver_carriers.resolved_page_table_affine_stride,
            "resolved_page_table_affine_segment_pages": resolver_carriers.resolved_page_table_affine_segment_pages,
            "resolved_page_table_affine_second_base": resolver_carriers.resolved_page_table_affine_second_base,
            "resolved_page_table_affine_second_stride": resolver_carriers.resolved_page_table_affine_second_stride,
            "resolved_page_table_affine_batch_stride": resolver_carriers.resolved_page_table_affine_batch_stride,
            "resolved_page_table_affine_head_stride": affine_head_stride,
            "resolved_page_table_affine_direct": affine_direct,
            "resolved_page_table_affine_cols": affine_cols,
            "resolved_seqused_k_by_head_i32": resolver_carriers.resolver_visible_seqused_k_by_head_i32,
            "row_consume_mode_i32": resolver_carriers.row_consume_mode_i32 if has_affine else None,
            "selected_seqused_k_by_head_i32": None,
            "page_resolver_kind": int(PageResolverKind.RESOLVED_ROW_PTR),
            "page_resolver_subkind": resolver_subkind_i,
            "compact_base_page_i32": None,
            "compact_page_count_i32": None,
            "recent_first_logical_page_i32": None,
            "graph_replay_carriers": bool(graph_replay_carriers),
        }
        launch_block_table = block_table
        launch_seqused_k = launch_effective_seqused_k
        max_seqlen_k_i = effective_max_seqlen_k_i
        route_mode = "resolved_row_ptr"
        overlay_width = 0
        page_table_rows_rewritten = ()
        length_rows_rewritten = tuple(range(batch_size))
    else:
        arena = _cached_resolved_row_ptr_arena(
            step_bound_meta=step_bound_meta,
            layer_index=layer_index,
            batch_size=batch_size,
            num_kv_heads=num_kv_heads_i,
            max_pages_per_row=int(block_table.shape[1]),
            device=q.device,
        )
        arena.bind_production_row_table(
            canonical_block_table=block_table,
            compact_ready_by_batch_row=row_is_compact,
            row_effective_k_by_row=row_effective_k_tuple,
            page_size=page_size_i,
            reserved_manager_block_ids=_compact_reserved_ids(state),
            slot_by_row=slot_by_row,
            compact_valid_tokens_by_row=compact_valid_tokens_tuple,
            compact_offset_tokens_by_row=compact_offset_tokens_tuple,
            recent_first_page_by_row=recent_first_page_tuple,
            safe_page_id=safe_page_id_i,
            compact_capacity_pages=compact_capacity_pages,
            planned_layout=planned_layout,
        )
        resolver_kwargs = {
            "selected_page_table_i32": None,
            "effective_row_slot_i32": None,
            "resolved_page_table_row_ptr_u64": arena.carrier_u64,
            "resolved_page_table_affine_i32": None,
            "resolved_page_table_affine_base": None,
            "resolved_page_table_affine_stride": None,
            "resolved_page_table_affine_segment_pages": None,
            "resolved_page_table_affine_second_base": None,
            "resolved_page_table_affine_second_stride": None,
            "resolved_page_table_affine_batch_stride": None,
            "resolved_page_table_affine_head_stride": 0,
            "resolved_page_table_affine_direct": False,
            "resolved_page_table_affine_cols": 0,
            "resolved_seqused_k_by_head_i32": arena.batch_seqused_k_i32,
            "row_consume_mode_i32": None,
            "selected_seqused_k_by_head_i32": None,
            "page_resolver_kind": int(PageResolverKind.RESOLVED_ROW_PTR),
            "page_resolver_subkind": int(PageResolverSubkind.ROWPTR),
            "compact_base_page_i32": None,
            "compact_page_count_i32": None,
            "recent_first_logical_page_i32": None,
            "graph_replay_carriers": False,
        }
        launch_block_table = block_table
        launch_seqused_k = launch_effective_seqused_k
        max_seqlen_k_i = effective_max_seqlen_k_i
        route_mode = "resolved_row_ptr"
        overlay_width = 0
        page_table_rows_rewritten = ()
        length_rows_rewritten = tuple(range(batch_size))

    fa_version = _bridge_flash_attn_version(bridge)
    mixed_page_backend = _mixed_page_backend_label(fa_version)

    # [E5] 调用点先查门控：trace 关闭（生产默认）时省掉每 layer 11 键 dict 构造。
    if _overlay_trace_enabled():
        _append_trace(
            step_bound_meta,
            {
                "route": "compact_mixed_page_overlay",
                "layer": layer_index,
                "mode": route_mode,
                "fa_version": int(fa_version),
                "backend": mixed_page_backend,
                "overlay_width": overlay_width,
                "page_table_rows_rewritten": page_table_rows_rewritten,
                "length_rows_rewritten": length_rows_rewritten,
                "page_resolver_kind": int(resolver_kwargs["page_resolver_kind"]),
                "graph_replay_carriers": bool(
                    resolver_kwargs.get("graph_replay_carriers", False)
                ),
            },
        )

    result = bridge.mixed_page_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        max_seqlen_q=_as_int("max_seqlen_q", max_seqlen_q),
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_k=max_seqlen_k_i,
        seqused_k=launch_seqused_k,
        q_v=q_v,
        softmax_scale=None
        if softmax_scale is None
        else _as_float("softmax_scale", softmax_scale),
        causal=True,
        window_size=_window_size(window_size),
        softcap=_as_float("softcap", softcap),
        block_table=launch_block_table,
        **resolver_kwargs,
        return_softmax_lse=False,
        out=out,
        scheduler_metadata=scheduler_metadata,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        num_splits=_as_int("num_splits", num_splits),
        s_aux=s_aux,
        cp_world_size=_as_int("cp_world_size", cp_world_size),
        cp_rank=_as_int("cp_rank", cp_rank),
        cp_tot_seqused_k=cp_tot_seqused_k,
        capture_scores=capture_scores,
        capture_row_index_i32=capture_row_index_i32,
        row_capture_last_n_i32=row_capture_last_n_i32,
        prefill_active_worklist=bool(prefill_active_worklist),
        prefill_active_count=_as_int(
            "prefill_active_count", prefill_active_count
        ),
    )
    if _route_trace_enabled():
        resolved_seqused = resolver_kwargs.get("resolved_seqused_k_by_head_i32")
        if isinstance(resolved_seqused, torch.Tensor):
            resolved_count = int(resolved_seqused.numel())
            if resolved_count == int(batch_size):
                selected_k_granularity = "batch_row"
            elif resolved_count == int(batch_size) * int(num_kv_heads_i):
                selected_k_granularity = "per_head"
            else:
                selected_k_granularity = "invalid"
        else:
            selected_k_granularity = "per_head"
        _append_route_trace_if_enabled(
            {
                "event": "mixed_page_call",
                "callable": "mixed_page_attn_varlen_func",
                "mode": route_mode,
                "route": "compact_mixed_page_overlay",
                "fa_version": int(fa_version),
                "backend": mixed_page_backend,
                "page_abi_version": 4,
                "page_resolver_kind": int(resolver_kwargs["page_resolver_kind"]),
                "graph_replay_carriers": bool(
                    resolver_kwargs.get("graph_replay_carriers", False)
                ),
                "has_resolved_affine": bool(
                    resolver_kwargs.get("resolved_page_table_affine_i32") is not None
                    or resolver_kwargs.get("resolved_page_table_affine_base") is not None
                    or resolver_kwargs.get("resolved_page_table_affine_stride") is not None
                ),
                "has_resolved_affine_direct": bool(
                    resolver_kwargs.get("resolved_page_table_affine_direct", False)
                ),
                "selected_k_granularity": selected_k_granularity,
                "selected_per_head_tensor_present": resolver_kwargs.get("selected_page_table_i32") is not None,
                "selected_row_count": int(sum(1 for value in row_is_compact if value)),
                "capture_row_count": int(capture_row_index_i32 is not None),
                "batch_size": int(batch_size),
                "num_kv_heads": int(num_kv_heads_i),
                "overlay_width": int(overlay_width),
                "slot_by_row": tuple(int(v) for v in slot_by_row),
                "row_is_compact": tuple(bool(v) for v in row_is_compact),
                "compact_valid_tokens": compact_valid_tokens_tuple,
                "compact_offset_tokens": compact_offset_tokens_tuple,
                "recent_first_page": recent_first_page_tuple,
                "recent_page_count": recent_page_count_tuple,
                "row_effective_k": row_effective_k_tuple,
                "page_table_rows_rewritten": page_table_rows_rewritten,
                "length_rows_rewritten": length_rows_rewritten,
                "prefill_active_worklist": bool(prefill_active_worklist),
            }
        )
    return result
