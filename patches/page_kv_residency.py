"""Compact page KV residency data model.

This module intentionally accepts duck-typed vLLM-like config objects so unit
tests can validate the contract without importing vLLM.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Optional, Tuple

import torch

from patches.sparse_constants import compact_gen_count


@dataclass(frozen=True, slots=True)
class CompactPageLease:
    kv_cache_group_id: int
    reserved_manager_block_ids: Tuple[int, ...]
    reserve_epoch: int
    manager_block_size: int
    kernel_page_size: int
    kv_cache_shape_signature: Tuple[Any, ...]
    compact_blocks_per_slot: int
    max_live_sparse_slots: int

    def __post_init__(self) -> None:
        for name in (
            "manager_block_size",
            "kernel_page_size",
            "compact_blocks_per_slot",
            "max_live_sparse_slots",
        ):
            _require_positive_int(getattr(self, name), name)

        reserved_ids = self.reserved_manager_block_ids
        if not isinstance(reserved_ids, (tuple, list)):
            raise ValueError("reserved_manager_block_ids must be a tuple or list")
        reserved_ids = tuple(reserved_ids)
        # [DUAL-GEN-L2] 双代开时 lease 为每 slot 两个半区(sub-slot 域=2×slots),
        # 关时 factor=1 与历史逐位一致。
        expected_blocks = (
            self.max_live_sparse_slots
            * self.compact_blocks_per_slot
            * compact_gen_count()
        )
        if len(reserved_ids) != expected_blocks:
            raise ValueError(
                "reserved_manager_block_ids length must equal "
                "max_live_sparse_slots * compact_blocks_per_slot * gen_count"
            )
        for block_id in reserved_ids:
            if type(block_id) is not int or block_id <= 0:
                raise ValueError("reserved manager block ids must be positive ints")
        for prev_block_id, block_id in zip(reserved_ids, reserved_ids[1:]):
            if block_id != prev_block_id + 1:
                raise ValueError("reserved manager block ids must be strictly contiguous")
        object.__setattr__(self, "reserved_manager_block_ids", reserved_ids)

    def slot_manager_block_ids(self, slot: int) -> Tuple[int, ...]:
        return slot_manager_block_ids(self, slot)


@dataclass(frozen=True, slots=True)
class CompactPageResidency:
    lease: CompactPageLease
    manager_block_ids: Tuple[int, ...]
    kernel_page_ids: Tuple[int, ...]
    num_pages: int
    page_size: int
    num_heads: int
    head_dim: int

    def slot_manager_block_ids(self, slot: int) -> Tuple[int, ...]:
        return slot_manager_block_ids(self.lease, slot)


@dataclass(slots=True)
class CompactMetadataBuffers:
    compact_arena_pos: Any = None
    capacity_slots: int = 0
    compact_blocks_per_slot: int = 0
    page_size: int = 0
    kv_storage_data_ptr: int = 0
    compact_k_data_ptr: int = 0
    compact_v_data_ptr: int = 0


_RESERVED_IDS_ATTR = "_sfi_compact_reserved_block_ids"
_LEASE_ATTR = "_sfi_compact_page_lease"
_LEASE_MANIFEST_ATTR = "_sfi_compact_page_lease_manifest"
_LEASE_MANIFEST_ENV = "VLLM_SPARSE_COMPACT_PAGE_LEASE_MANIFEST"
_PATCHED_ATTR = "_sfi_compact_block_pool_patched"
_ORIGINALS_ATTR = "_sfi_compact_block_pool_originals"
_EMPTY_RESERVED_IDS: frozenset[int] = frozenset()
_LEASE_BY_CONFIG_ID: dict[int, tuple[Any, CompactPageLease]] = {}
ENGINE_CORE_BLOCK_POOL_STATE_UTILITY = "sfi_compact_page_block_pool_state"


def _require_positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive int")
    return value


def require_native_compact_residency_for_layers(
    states: Mapping[int, Any],
    expected_layers: Iterable[int],
) -> None:
    missing: list[int] = []
    for layer in expected_layers:
        layer_i = int(layer)
        state = states.get(layer_i)
        if state is None or getattr(state, "compact_page_residency", None) is None:
            missing.append(layer_i)
    if missing:
        raise RuntimeError(
            "one-shot RRP compact+recent requires native compact_page_residency "
            f"for every expected layer; missing_layers={missing}"
        )


def validate_compact_page_residency_config(config: Any) -> Optional[int]:
    if not getattr(config, "compact_page_residency_enabled", False):
        return None
    max_live_sparse_slots = _require_positive_int(
        getattr(config, "max_live_sparse_slots", None),
        "max_live_sparse_slots",
    )
    compact_blocks_per_slot = _require_positive_int(
        getattr(config, "compact_blocks_per_slot", None),
        "compact_blocks_per_slot",
    )
    # [DUAL-GEN-L2] reserve 总量含双代半区(开关关=×1 逐位)。
    return max_live_sparse_slots * compact_blocks_per_slot * compact_gen_count()


def validate_compact_page_stride_capacity(
    config: Any,
    manager_block_size: int,
) -> None:
    if not getattr(config, "compact_page_residency_enabled", False):
        return
    block_size = _require_positive_int(manager_block_size, "manager_block_size")
    blocks_per_slot = _require_positive_int(
        getattr(config, "compact_blocks_per_slot", None),
        "compact_blocks_per_slot",
    )
    sink_tokens = max(0, int(getattr(config, "sink", 0) or 0))
    alpha_fair = getattr(config, "alpha_fair", None)
    raw_k_head = getattr(alpha_fair, "k_head", 0) if alpha_fair is not None else 0
    k_head = 0 if raw_k_head is None else max(0, int(raw_k_head))
    try:
        from patches.fa_sparse_runtime.compact_recent_alignment import (
            compact_recent_effective_k_head,
        )

        k_head = compact_recent_effective_k_head(
            k_head=k_head,
            sink_tokens=sink_tokens,
            attn_mode=str(getattr(config, "attn_mode", "compact_recent")),
        )
    except Exception:
        pass
    required_tokens = sink_tokens + k_head
    capacity_tokens = blocks_per_slot * block_size
    if required_tokens > capacity_tokens:
        raise ValueError(
            "compact page stride capacity is smaller than selector budget: "
            f"capacity_tokens={capacity_tokens}, required_tokens={required_tokens}, "
            f"compact_blocks_per_slot={blocks_per_slot}, block_size={block_size}, "
            f"sink={sink_tokens}, effective_k_head={k_head}"
        )


def compute_reserve_block_ids(config: Any, num_blocks: int) -> Optional[Tuple[int, ...]]:
    reserve_blocks = validate_compact_page_residency_config(config)
    if reserve_blocks is None:
        return None
    _require_positive_int(num_blocks, "num_blocks")
    available_non_null_blocks = num_blocks - 1
    if reserve_blocks > available_non_null_blocks:
        raise ValueError(
            "compact page reserve exceeds available non-null blocks: "
            f"reserve={reserve_blocks}, available={available_non_null_blocks}"
        )
    start = num_blocks - reserve_blocks
    return tuple(range(start, num_blocks))


def _kv_cache_shape_signature(
    kv_cache_config: Any,
    kv_cache_spec: Any,
    kv_cache_group_id: int,
    kernel_page_size: int,
) -> Tuple[Any, ...]:
    spec_type = type(kv_cache_spec)
    dtype = getattr(kv_cache_spec, "dtype", None)
    return (
        ("kv_cache_spec_module", spec_type.__module__),
        ("kv_cache_spec_qualname", spec_type.__qualname__),
        ("num_blocks", getattr(kv_cache_config, "num_blocks")),
        ("kv_cache_group_id", kv_cache_group_id),
        ("type_id", getattr(kv_cache_spec, "type_id", None)),
        ("block_size", getattr(kv_cache_spec, "block_size", None)),
        ("kernel_page_size", kernel_page_size),
        ("page_size_bytes", getattr(kv_cache_spec, "page_size_bytes", None)),
        ("num_kv_heads", getattr(kv_cache_spec, "num_kv_heads", None)),
        ("head_size", getattr(kv_cache_spec, "head_size", None)),
        ("dtype", None if dtype is None else str(dtype)),
    )


def build_tail_compact_page_lease(
    kv_cache_config: Any,
    config: Any,
    kv_cache_group_id: int = 0,
    reserve_epoch: int = 0,
    kernel_page_size: Optional[int] = None,
) -> Optional[CompactPageLease]:
    reserved_manager_block_ids = compute_reserve_block_ids(
        config,
        getattr(kv_cache_config, "num_blocks"),
    )
    if reserved_manager_block_ids is None:
        return None

    groups = getattr(kv_cache_config, "kv_cache_groups")
    if len(groups) != 1:
        raise ValueError(
            "compact page residency currently requires exactly one KV cache group"
        )
    if not isinstance(kv_cache_group_id, int):
        raise TypeError("kv_cache_group_id must be int")
    if kv_cache_group_id < 0 or kv_cache_group_id >= len(groups):
        raise IndexError(
            f"kv_cache_group_id {kv_cache_group_id} is out of range for {len(groups)} groups"
        )

    group = groups[kv_cache_group_id]
    kv_cache_spec = getattr(group, "kv_cache_spec")
    manager_block_size = _require_positive_int(
        getattr(kv_cache_spec, "block_size"),
        "kv_cache_spec.block_size",
    )
    validate_compact_page_stride_capacity(config, manager_block_size)
    resolved_kernel_page_size = (
        manager_block_size
        if kernel_page_size is None
        else _require_positive_int(kernel_page_size, "kernel_page_size")
    )

    return CompactPageLease(
        kv_cache_group_id=kv_cache_group_id,
        reserved_manager_block_ids=reserved_manager_block_ids,
        reserve_epoch=reserve_epoch,
        manager_block_size=manager_block_size,
        kernel_page_size=resolved_kernel_page_size,
        kv_cache_shape_signature=_kv_cache_shape_signature(
            kv_cache_config,
            kv_cache_spec,
            kv_cache_group_id,
            resolved_kernel_page_size,
        ),
        compact_blocks_per_slot=getattr(config, "compact_blocks_per_slot"),
        max_live_sparse_slots=getattr(config, "max_live_sparse_slots"),
    )


def slot_manager_block_ids(lease: CompactPageLease, slot: int) -> Tuple[int, ...]:
    if type(slot) is not int:
        raise ValueError("slot must be an int")
    # [DUAL-GEN-L2] slot 参数为 sub-slot(gen*max_live+slot);关时域=max_live 逐位。
    sub_slot_capacity = lease.max_live_sparse_slots * compact_gen_count()
    if slot < 0 or slot >= sub_slot_capacity:
        raise ValueError(
            f"slot {slot} is out of range for compact page capacity {sub_slot_capacity}"
        )
    start = slot * lease.compact_blocks_per_slot
    end = start + lease.compact_blocks_per_slot
    return lease.reserved_manager_block_ids[start:end]


def _block_id(block: Any) -> int:
    value = getattr(block, "block_id")
    if type(value) is not int:
        raise TypeError("KV cache block.block_id must be an int")
    return value


def _reserved_ids(block_pool: Any) -> frozenset[int]:
    ids = getattr(block_pool, _RESERVED_IDS_ATTR, None)
    if not ids:
        return _EMPTY_RESERVED_IDS
    if isinstance(ids, frozenset):
        return ids
    normalized = frozenset(int(block_id) for block_id in ids)
    setattr(block_pool, _RESERVED_IDS_ATTR, normalized)
    return normalized


def _free_block_ids(block_pool: Any) -> tuple[int, ...]:
    """Snapshot the native vLLM 0.19 free queue without mutating it."""
    queue = getattr(block_pool, "free_block_queue")
    head = getattr(queue, "fake_free_list_head", None)
    tail = getattr(queue, "fake_free_list_tail", None)
    if head is not None and tail is not None:
        cursor = getattr(head, "next_free_block", None)
        result: list[int] = []
        seen: set[int] = set()
        limit = int(getattr(block_pool, "num_gpu_blocks")) + 1
        while cursor is not tail:
            if cursor is None or len(result) >= limit:
                raise RuntimeError("invalid native KV free-block linked list")
            block_id = _block_id(cursor)
            if block_id in seen:
                raise RuntimeError("cycle in native KV free-block linked list")
            seen.add(block_id)
            result.append(block_id)
            cursor = getattr(cursor, "next_free_block", None)
        if len(result) != int(getattr(queue, "num_free_blocks")):
            raise RuntimeError(
                "native KV free-block count does not match linked list"
            )
        return tuple(result)
    ids = getattr(queue, "ids", None)
    if callable(ids):
        return tuple(int(block_id) for block_id in ids())
    raise RuntimeError("unsupported KV free-block queue shape")


def _compact_lease_state(lease: Any) -> Optional[dict[str, Any]]:
    if lease is None:
        return None
    if not isinstance(lease, CompactPageLease):
        raise RuntimeError(
            "EngineCore compact lease has unexpected type: "
            f"{type(lease).__name__}"
        )
    return {
        "kv_cache_group_id": int(lease.kv_cache_group_id),
        "reserved_manager_block_ids": list(
            lease.reserved_manager_block_ids
        ),
        "reserve_epoch": int(lease.reserve_epoch),
        "manager_block_size": int(lease.manager_block_size),
        "kernel_page_size": int(lease.kernel_page_size),
        "compact_blocks_per_slot": int(lease.compact_blocks_per_slot),
        "max_live_sparse_slots": int(lease.max_live_sparse_slots),
    }


def snapshot_compact_page_block_pool_state(engine_core: Any) -> dict[str, Any]:
    """Return actual scheduler BlockPool ownership, not a transported manifest."""
    scheduler = getattr(engine_core, "scheduler", None)
    manager = getattr(scheduler, "kv_cache_manager", None)
    block_pool = getattr(manager, "block_pool", None)
    if manager is None or block_pool is None:
        raise RuntimeError("EngineCore KVCacheManager/BlockPool unavailable")

    num_gpu_blocks = int(getattr(block_pool, "num_gpu_blocks"))
    blocks = list(getattr(block_pool, "blocks"))
    if len(blocks) != num_gpu_blocks:
        raise RuntimeError(
            "EngineCore BlockPool blocks length does not match num_gpu_blocks"
        )
    free_ids = _free_block_ids(block_pool)
    free_id_set = set(free_ids)
    reserved_ids = tuple(sorted(_reserved_ids(block_pool)))
    reserved_states = []
    for block_id in reserved_ids:
        if block_id < 0 or block_id >= num_gpu_blocks:
            raise RuntimeError(
                f"EngineCore reserved block id out of range: {block_id}"
            )
        block = blocks[block_id]
        reserved_states.append(
            {
                "block_id": block_id,
                "ref_cnt": int(getattr(block, "ref_cnt", 0)),
                "is_null": bool(getattr(block, "is_null", False)),
                "in_free_queue": block_id in free_id_set,
            }
        )
    null_block = getattr(block_pool, "null_block", None)
    null_block_id = _block_id(null_block) if null_block is not None else -1
    manager_lease = getattr(manager, _LEASE_ATTR, None)
    pool_lease = getattr(block_pool, _LEASE_ATTR, None)
    return {
        "schema": "sfi.engine_core_block_pool_state.v1",
        "engine_core_class": type(engine_core).__name__,
        "scheduler_class": type(scheduler).__name__,
        "kv_cache_manager_class": type(manager).__name__,
        "block_pool_class": type(block_pool).__name__,
        "num_gpu_blocks": num_gpu_blocks,
        "num_free_blocks": len(free_ids),
        "free_queue_reported_count": int(
            getattr(block_pool.free_block_queue, "num_free_blocks")
        ),
        "null_block_id": null_block_id,
        "null_block_is_null": bool(getattr(null_block, "is_null", False)),
        "null_block_in_free_queue": null_block_id in free_id_set,
        "reserved_block_ids": list(reserved_ids),
        "reserved_block_count": len(reserved_ids),
        "reserved_ids_tail_contiguous": reserved_ids
        == tuple(range(num_gpu_blocks - len(reserved_ids), num_gpu_blocks)),
        "reserved_blocks": reserved_states,
        "reserved_ids_in_free_queue": sorted(
            set(reserved_ids).intersection(free_id_set)
        ),
        "manager_lease": _compact_lease_state(manager_lease),
        "pool_lease": _compact_lease_state(pool_lease),
        "manager_pool_lease_same_object": manager_lease is pool_lease,
    }


def install_engine_core_block_pool_state_utility() -> None:
    """Install one named exact-run utility on vLLM EngineCore."""
    from vllm.v1.engine.core import EngineCore

    existing = getattr(EngineCore, ENGINE_CORE_BLOCK_POOL_STATE_UTILITY, None)
    if existing is not None:
        if getattr(existing, "_sfi_engine_core_block_pool_state", False):
            return
        raise RuntimeError(
            f"EngineCore utility name collision: {ENGINE_CORE_BLOCK_POOL_STATE_UTILITY}"
        )

    def _state(self: Any) -> dict[str, Any]:
        return snapshot_compact_page_block_pool_state(self)

    _state._sfi_engine_core_block_pool_state = True  # type: ignore[attr-defined]
    setattr(EngineCore, ENGINE_CORE_BLOCK_POOL_STATE_UTILITY, _state)


def _is_reserved_block(block_pool: Any, block: Any) -> bool:
    reserved_ids = getattr(block_pool, _RESERVED_IDS_ATTR, None)
    return bool(reserved_ids) and _block_id(block) in reserved_ids


def _is_block_like(value: Any) -> bool:
    return hasattr(value, "block_id")


class CompactBlockPoolAdapter:
    """Small native BlockPool adapter for compact page ownership.

    Keep vLLM object-shape details here so residency logic does not grow
    branchy compatibility code. KVCacheBlock exposes ``ref_cnt`` as a field
    and FreeKVCacheBlockQueue owns the free-list mutation.
    """

    def __init__(self, block_pool: Any) -> None:
        self.block_pool = block_pool

    @property
    def blocks(self) -> Any:
        return getattr(self.block_pool, "blocks")

    @property
    def free_block_queue(self) -> Any:
        return getattr(self.block_pool, "free_block_queue")

    def reserve_block(self, block: Any) -> None:
        if bool(getattr(block, "is_null", False)):
            raise RuntimeError("compact page reservation cannot own the null block")
        if int(getattr(block, "ref_cnt", 0)) != 0:
            raise RuntimeError(
                f"compact page reserved block {_block_id(block)} is already in use"
            )
        self.free_block_queue.remove(block)
        block.ref_cnt = int(getattr(block, "ref_cnt", 0)) + 1

    def ordinary_blocks(self, blocks: Iterable[Any]) -> list[Any]:
        reserved_ids = _reserved_ids(self.block_pool)
        if not reserved_ids:
            return list(blocks)
        return [block for block in blocks if _block_id(block) not in reserved_ids]

    def compact_reserved_cache_only(self, cache: Any, reserved_ids: frozenset[int]) -> Any:
        if cache is None:
            return None
        if isinstance(cache, dict):
            kept_cache: defaultdict[Any, dict[int, Any]] = defaultdict(dict)
            for block_hash, blocks_by_id in cache.items():
                if isinstance(blocks_by_id, Mapping):
                    for block_id, block in blocks_by_id.items():
                        if int(block_id) in reserved_ids:
                            kept_cache[block_hash][int(block_id)] = block
                elif _is_block_like(blocks_by_id):
                    block_id = _block_id(blocks_by_id)
                    if block_id in reserved_ids:
                        kept_cache[block_hash][block_id] = blocks_by_id
            return kept_cache
        raw_cache = getattr(cache, "_cache", None)
        if isinstance(raw_cache, dict):
            kept = type(cache)()
            kept_raw = getattr(kept, "_cache")
            for block_hash, blocks_value in raw_cache.items():
                if isinstance(blocks_value, Mapping):
                    ordinary = {
                        int(block_id): block
                        for block_id, block in blocks_value.items()
                        if int(block_id) in reserved_ids
                    }
                    if ordinary:
                        kept_raw[block_hash] = ordinary
                elif _is_block_like(blocks_value):
                    if _block_id(blocks_value) in reserved_ids:
                        kept_raw[block_hash] = blocks_value
            return kept
        try:
            return type(cache)()
        except Exception:
            return cache


def _same_lease(left: CompactPageLease, right: CompactPageLease) -> bool:
    return left == right


def _lease_to_manifest(lease: CompactPageLease) -> dict[str, Any]:
    return {
        "kv_cache_group_id": lease.kv_cache_group_id,
        "reserved_manager_block_ids": tuple(lease.reserved_manager_block_ids),
        "reserve_epoch": lease.reserve_epoch,
        "manager_block_size": lease.manager_block_size,
        "kernel_page_size": lease.kernel_page_size,
        "kv_cache_shape_signature": tuple(lease.kv_cache_shape_signature),
        "compact_blocks_per_slot": lease.compact_blocks_per_slot,
        "max_live_sparse_slots": lease.max_live_sparse_slots,
    }


def _lease_from_manifest(value: Mapping[str, Any]) -> CompactPageLease:
    raw_signature = value["kv_cache_shape_signature"]
    kv_cache_shape_signature = tuple(
        tuple(item) if isinstance(item, (list, tuple)) else item
        for item in raw_signature
    )
    return CompactPageLease(
        kv_cache_group_id=int(value["kv_cache_group_id"]),
        reserved_manager_block_ids=tuple(
            int(block_id) for block_id in value["reserved_manager_block_ids"]
        ),
        reserve_epoch=int(value["reserve_epoch"]),
        manager_block_size=int(value["manager_block_size"]),
        kernel_page_size=int(value["kernel_page_size"]),
        kv_cache_shape_signature=kv_cache_shape_signature,
        compact_blocks_per_slot=int(value["compact_blocks_per_slot"]),
        max_live_sparse_slots=int(value["max_live_sparse_slots"]),
    )


def attach_compact_page_lease(kv_cache_config: Any, lease: CompactPageLease) -> None:
    if not isinstance(lease, CompactPageLease):
        raise TypeError("lease must be CompactPageLease")
    _LEASE_BY_CONFIG_ID[id(kv_cache_config)] = (kv_cache_config, lease)
    manifest = _lease_to_manifest(lease)
    os.environ[_LEASE_MANIFEST_ENV] = json.dumps(manifest, sort_keys=True)
    try:
        setattr(kv_cache_config, _LEASE_ATTR, lease)
        setattr(kv_cache_config, _LEASE_MANIFEST_ATTR, manifest)
    except AttributeError:
        pass


def clear_compact_page_lease_transport() -> None:
    _LEASE_BY_CONFIG_ID.clear()
    os.environ.pop(_LEASE_MANIFEST_ENV, None)


def _get_transported_lease(
    kv_cache_config: Any,
    block_pool: Any = None,
    *,
    include_env: bool = True,
) -> Any:
    if kv_cache_config is not None:
        stored = _LEASE_BY_CONFIG_ID.get(id(kv_cache_config))
        if stored is not None:
            stored_config, lease = stored
            if stored_config is kv_cache_config:
                return lease
    for source in (kv_cache_config, block_pool):
        if source is None:
            continue
        lease = getattr(source, _LEASE_ATTR, None)
        if lease is not None:
            return lease
        manifest = getattr(source, _LEASE_MANIFEST_ATTR, None)
        if manifest is not None:
            return manifest
    if include_env:
        manifest_payload = os.environ.get(_LEASE_MANIFEST_ENV)
        if manifest_payload:
            try:
                return json.loads(manifest_payload)
            except json.JSONDecodeError as exc:
                raise RuntimeError("compact page lease manifest env is not valid JSON") from exc
    return None


def _coerce_compact_page_lease(value: Any) -> CompactPageLease:
    if isinstance(value, CompactPageLease):
        return value
    if isinstance(value, Mapping):
        return _lease_from_manifest(value)
    raise TypeError("compact page lease must be CompactPageLease or manifest mapping")


def ensure_compact_page_lease_transport(
    kv_cache_config: Any,
    config: Any,
    *,
    kv_cache_group_id: int = 0,
    reserve_epoch: int = 0,
    kernel_page_size: Optional[int] = None,
) -> Optional[CompactPageLease]:
    """Attach a deterministic lease manifest before BlockPool reservation exists."""
    if not getattr(config, "compact_page_residency_enabled", False):
        return None
    raw_lease = _get_transported_lease(kv_cache_config, include_env=False)
    if raw_lease is not None:
        return resolve_compact_page_lease(kv_cache_config, config)
    lease = build_tail_compact_page_lease(
        kv_cache_config,
        config,
        kv_cache_group_id=kv_cache_group_id,
        reserve_epoch=reserve_epoch,
        kernel_page_size=kernel_page_size,
    )
    if lease is None:
        return None
    attach_compact_page_lease(kv_cache_config, lease)
    return lease


def _validate_transported_lease(
    lease: CompactPageLease,
    expected: CompactPageLease,
) -> None:
    checks = (
        ("reserved_manager_block_ids", lease.reserved_manager_block_ids, expected.reserved_manager_block_ids),
        ("kv_cache_group_id", lease.kv_cache_group_id, expected.kv_cache_group_id),
        ("manager_block_size", lease.manager_block_size, expected.manager_block_size),
        ("kernel_page_size", lease.kernel_page_size, expected.kernel_page_size),
        ("kv_cache_shape_signature", lease.kv_cache_shape_signature, expected.kv_cache_shape_signature),
        ("compact_blocks_per_slot", lease.compact_blocks_per_slot, expected.compact_blocks_per_slot),
        ("max_live_sparse_slots", lease.max_live_sparse_slots, expected.max_live_sparse_slots),
    )
    for name, actual, required in checks:
        if actual != required:
            raise RuntimeError(
                f"compact page lease mismatch for {name}: "
                f"transported={actual!r}, expected={required!r}"
            )


def _signature_value(signature: Tuple[Any, ...], key: str) -> Any:
    for item in signature:
        if (
            isinstance(item, (tuple, list))
            and len(item) == 2
            and item[0] == key
        ):
            return item[1]
    return None


def _validate_lease_against_worker_kv_cache(
    lease: CompactPageLease,
    kv_cache_config: Any,
    config: Any,
) -> None:
    """Validate a scheduler-owned compact lease in a worker KV cache context.

    The scheduler BlockPool may use a bounded manager block count while the
    worker KV tensor exposes a larger physical block capacity. The transported
    lease is the ownership truth; the worker only verifies that it can map
    those manager block ids into the current KV cache layout.
    """
    expected_blocks = validate_compact_page_residency_config(config)
    if expected_blocks is None:
        raise RuntimeError("compact page lease was transported while config is disabled")
    if len(lease.reserved_manager_block_ids) != expected_blocks:
        raise RuntimeError(
            "compact page lease mismatch for reserved_manager_block_ids length: "
            f"transported={len(lease.reserved_manager_block_ids)!r}, expected={expected_blocks!r}"
        )
    if int(lease.compact_blocks_per_slot) != int(getattr(config, "compact_blocks_per_slot")):
        raise RuntimeError(
            "compact page lease mismatch for compact_blocks_per_slot: "
            f"transported={lease.compact_blocks_per_slot!r}, "
            f"expected={getattr(config, 'compact_blocks_per_slot')!r}"
        )
    if int(lease.max_live_sparse_slots) != int(getattr(config, "max_live_sparse_slots")):
        raise RuntimeError(
            "compact page lease mismatch for max_live_sparse_slots: "
            f"transported={lease.max_live_sparse_slots!r}, "
            f"expected={getattr(config, 'max_live_sparse_slots')!r}"
        )

    groups = getattr(kv_cache_config, "kv_cache_groups")
    if len(groups) != 1:
        raise RuntimeError(
            "compact page residency currently requires exactly one KV cache group"
        )
    if int(lease.kv_cache_group_id) < 0 or int(lease.kv_cache_group_id) >= len(groups):
        raise RuntimeError(
            "compact page lease kv_cache_group_id is outside worker KV cache groups"
        )
    kv_cache_spec = getattr(groups[int(lease.kv_cache_group_id)], "kv_cache_spec")
    manager_block_size = _require_positive_int(
        getattr(kv_cache_spec, "block_size"),
        "kv_cache_spec.block_size",
    )
    if int(lease.manager_block_size) != manager_block_size:
        raise RuntimeError(
            "compact page lease mismatch for manager_block_size: "
            f"transported={lease.manager_block_size!r}, expected={manager_block_size!r}"
        )
    if int(lease.kernel_page_size) != manager_block_size:
        raise RuntimeError(
            "compact page lease mismatch for kernel_page_size: "
            f"transported={lease.kernel_page_size!r}, expected={manager_block_size!r}"
        )

    current_signature = _kv_cache_shape_signature(
        kv_cache_config,
        kv_cache_spec,
        int(lease.kv_cache_group_id),
        int(lease.kernel_page_size),
    )
    for key in (
        "kv_cache_spec_module",
        "kv_cache_spec_qualname",
        "kv_cache_group_id",
        "type_id",
        "block_size",
        "kernel_page_size",
        "page_size_bytes",
        "num_kv_heads",
        "head_size",
        "dtype",
    ):
        transported = _signature_value(lease.kv_cache_shape_signature, key)
        current = _signature_value(current_signature, key)
        if transported != current:
            raise RuntimeError(
                f"compact page lease mismatch for kv_cache_shape_signature.{key}: "
                f"transported={transported!r}, expected={current!r}"
            )

    num_blocks = _require_positive_int(getattr(kv_cache_config, "num_blocks"), "num_blocks")
    max_reserved = max(lease.reserved_manager_block_ids)
    if max_reserved >= num_blocks:
        raise RuntimeError(
            "compact page lease reserved block id outside worker KV cache: "
            f"max_reserved={max_reserved}, num_blocks={num_blocks}"
        )


def resolve_compact_page_lease(
    kv_cache_config: Any,
    config: Any,
    block_pool: Any = None,
) -> Optional[CompactPageLease]:
    if not getattr(config, "compact_page_residency_enabled", False):
        return None

    raw_lease = _get_transported_lease(kv_cache_config, block_pool)
    if raw_lease is None:
        raise RuntimeError(
            "compact page lease is missing from kv_cache_config/block_pool; "
            "worker must not rebuild unchecked tail residency"
        )
    lease = _coerce_compact_page_lease(raw_lease)
    _validate_lease_against_worker_kv_cache(lease, kv_cache_config, config)
    return lease


def _storage_data_ptr(tensor: torch.Tensor) -> int:
    if hasattr(tensor, "untyped_storage"):
        return int(tensor.untyped_storage().data_ptr())
    return int(tensor.storage().data_ptr())


def _tensor_byte_span(tensor: torch.Tensor) -> tuple[int, int]:
    start = int(tensor.data_ptr())
    end = start + int(tensor.numel()) * int(tensor.element_size())
    return start, end


def _tensor_data_inside_span(tensor: torch.Tensor, span: tuple[int, int]) -> bool:
    start, end = _tensor_byte_span(tensor)
    span_start, span_end = span
    return bool(start >= span_start and end <= span_end and start < span_end)


def _append_compact_page_residency_trace(event: Mapping[str, Any]) -> None:
    raw_path = os.environ.get("VLLM_SPARSE_FA3_ROUTE_TRACE_LOG")
    if not raw_path:
        return
    try:
        with open(raw_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(dict(event), sort_keys=True) + "\n")
    except Exception:
        return


def bind_compact_page_residency_to_layer(
    state: Any,
    kv_cache: torch.Tensor,
    kv_cache_config: Any,
    config: Any,
    lease: Optional[CompactPageLease] = None,
) -> Optional[CompactPageResidency]:
    resolved_lease = lease
    if resolved_lease is None:
        resolved_lease = resolve_compact_page_lease(kv_cache_config, config)
    if resolved_lease is None:
        return None

    if not isinstance(kv_cache, torch.Tensor):
        raise TypeError("kv_cache must be a torch.Tensor")
    if kv_cache.dim() != 5 or int(kv_cache.shape[0]) != 2:
        raise RuntimeError(
            "compact page residency requires KV cache shape "
            "[2, num_blocks, block_size, num_kv_heads, head_dim]; "
            f"got shape={tuple(kv_cache.shape)} stride={tuple(kv_cache.stride())} "
            f"dtype={kv_cache.dtype} device={kv_cache.device}"
        )

    num_blocks = int(kv_cache.shape[1])
    block_size = int(kv_cache.shape[2])
    num_kv_heads = int(kv_cache.shape[3])
    head_dim = int(kv_cache.shape[4])
    if block_size != int(resolved_lease.manager_block_size):
        raise RuntimeError(
            "compact page lease block size mismatch: "
            f"lease={resolved_lease.manager_block_size}, kv_cache={block_size}"
        )
    reserved_ids = tuple(int(block_id) for block_id in resolved_lease.reserved_manager_block_ids)
    if not reserved_ids:
        raise RuntimeError("compact page lease has no reserved blocks")
    start_block = reserved_ids[0]
    block_count = len(reserved_ids)
    end_block = reserved_ids[-1] + 1
    if start_block < 0 or end_block > num_blocks:
        raise RuntimeError(
            "compact page reserved block id outside layer KV cache: "
            f"range=[{start_block}, {end_block}), num_blocks={num_blocks}"
        )
    if reserved_ids != tuple(range(start_block, end_block)):
        raise RuntimeError("compact page reserved blocks must be contiguous")

    key_blocks = kv_cache[0].narrow(0, start_block, block_count)
    value_blocks = kv_cache[1].narrow(0, start_block, block_count)
    try:
        compact_k = key_blocks.view(-1, num_kv_heads, head_dim)
        compact_v = value_blocks.view(-1, num_kv_heads, head_dim)
    except RuntimeError as exc:
        raise RuntimeError(
            "compact page reserved KV range is not viewable without copy"
        ) from exc
    base_ptr = _storage_data_ptr(kv_cache)
    k_storage_matches = _storage_data_ptr(compact_k) == base_ptr
    v_storage_matches = _storage_data_ptr(compact_v) == base_ptr
    if not k_storage_matches or not v_storage_matches:
        raise RuntimeError("compact page arena view does not share KV cache storage")
    k_data_in_reserved_span = _tensor_data_inside_span(
        compact_k,
        _tensor_byte_span(key_blocks),
    )
    v_data_in_reserved_span = _tensor_data_inside_span(
        compact_v,
        _tensor_byte_span(value_blocks),
    )
    if not k_data_in_reserved_span or not v_data_in_reserved_span:
        raise RuntimeError("compact page arena view is outside reserved KV span")

    residency = CompactPageResidency(
        lease=resolved_lease,
        manager_block_ids=reserved_ids,
        kernel_page_ids=reserved_ids,
        num_pages=block_count,
        page_size=block_size,
        num_heads=num_kv_heads,
        head_dim=head_dim,
    )
    pos_tokens = block_count * block_size
    existing_pos = getattr(state, "compact_arena_pos", None)
    if (
        isinstance(existing_pos, torch.Tensor)
        and tuple(existing_pos.shape) == (num_kv_heads, pos_tokens)
        and existing_pos.dtype == torch.int32
        and existing_pos.device == kv_cache.device
    ):
        compact_pos = existing_pos
        compact_pos.fill_(-1)
    else:
        compact_pos = torch.full(
            (num_kv_heads, pos_tokens),
            -1,
            device=kv_cache.device,
            dtype=torch.int32,
        )
    metadata = CompactMetadataBuffers(
        compact_arena_pos=compact_pos,
        capacity_slots=resolved_lease.max_live_sparse_slots,
        compact_blocks_per_slot=resolved_lease.compact_blocks_per_slot,
        page_size=block_size,
        kv_storage_data_ptr=base_ptr,
        compact_k_data_ptr=int(compact_k.data_ptr()),
        compact_v_data_ptr=int(compact_v.data_ptr()),
    )

    state.compact_arena_k = compact_k
    state.compact_arena_v = compact_v
    state.compact_arena_pos = compact_pos
    state.compact_arena_capacity_tokens = block_count * block_size
    state.compact_stride_tokens = resolved_lease.compact_blocks_per_slot * block_size
    state.compact_stride_blocks = resolved_lease.compact_blocks_per_slot
    state.compact_stride_block_size = block_size
    state.compact_page_residency = residency
    state.compact_metadata_buffers = metadata
    state.compact_page_residency_signature = resolved_lease.kv_cache_shape_signature
    state.compact_page_residency_generation = int(
        getattr(state, "compact_page_residency_generation", 0)
    ) + 1
    state.compact_views_bound_slots = 0
    state.compact_layout_key_view = None
    state.compact_layout_value_view = None
    state.compact_layout_token_positions_view = None
    state._compact_layout_view_key = None
    state._compact_layout_token_pos_view_key = None
    _append_compact_page_residency_trace(
        {
            "event": "compact_page_residency_bind",
            "layer_index": int(getattr(state, "layer_index", -1)),
            "reserved_manager_block_start": int(start_block),
            "reserved_manager_block_count": int(block_count),
            "slot_capacity_tokens": int(resolved_lease.compact_blocks_per_slot * block_size),
            "capacity_slots": int(resolved_lease.max_live_sparse_slots),
            "page_size": int(block_size),
            "num_kv_heads": int(num_kv_heads),
            "head_dim": int(head_dim),
            "compact_kv_storage_owner": "native_vllm_page_kv",
            "compact_arena_k_storage_matches_layer_kv_cache": bool(k_storage_matches),
            "compact_arena_v_storage_matches_layer_kv_cache": bool(v_storage_matches),
            "compact_arena_k_data_ptr_in_reserved_span": bool(k_data_in_reserved_span),
            "compact_arena_v_data_ptr_in_reserved_span": bool(v_data_in_reserved_span),
            "compact_to_compact_copy_bytes": 0,
            "arena_storage_matches_kv": True,
        }
    )
    return residency


def reserve_compact_page_blocks(
    block_pool: Any,
    kv_cache_config: Any,
    config: Any,
    *,
    kv_cache_group_id: int = 0,
    reserve_epoch: int = 0,
    kernel_page_size: Optional[int] = None,
) -> Optional[CompactPageLease]:
    """Reserve tail manager blocks for compact-page residency.

    The reservation is represented as first-class owner metadata on the block
    pool. The block ref-count is kept non-zero only as a secondary invariant so
    ordinary BlockPool paths do not mistake reserved blocks for free capacity.
    """
    expected_lease = build_tail_compact_page_lease(
        kv_cache_config,
        config,
        kv_cache_group_id=kv_cache_group_id,
        reserve_epoch=reserve_epoch,
        kernel_page_size=kernel_page_size,
    )
    if expected_lease is None:
        return None
    transported = _get_transported_lease(kv_cache_config, include_env=False)
    if transported is None:
        lease = expected_lease
    else:
        lease = _coerce_compact_page_lease(transported)
        _validate_transported_lease(lease, expected_lease)

    patch_compact_page_block_pool_methods(block_pool.__class__)

    existing_lease = getattr(block_pool, _LEASE_ATTR, None)
    if existing_lease is not None:
        if _same_lease(existing_lease, lease):
            return existing_lease
        raise RuntimeError("different compact page reservation already installed")

    existing_reserved_ids = getattr(block_pool, _RESERVED_IDS_ATTR, None)
    if existing_reserved_ids:
        raise RuntimeError("different compact page reservation already installed")

    pool = CompactBlockPoolAdapter(block_pool)
    blocks = pool.blocks
    reserved_ids = frozenset(lease.reserved_manager_block_ids)
    for block_id in lease.reserved_manager_block_ids:
        block = blocks[block_id]
        pool.reserve_block(block)

    setattr(block_pool, _RESERVED_IDS_ATTR, reserved_ids)
    setattr(block_pool, _LEASE_ATTR, lease)
    return lease


def patch_compact_page_block_pool_methods(block_pool_cls: type[Any]) -> dict[str, Callable[..., Any]]:
    """Patch BlockPool methods so reserved blocks stay outside normal ownership."""
    originals = getattr(block_pool_cls, _ORIGINALS_ATTR, None)
    if getattr(block_pool_cls, _PATCHED_ATTR, False):
        return originals

    originals = {
        "get_new_blocks": block_pool_cls.get_new_blocks,
        "free_blocks": block_pool_cls.free_blocks,
        "touch": block_pool_cls.touch,
        "_maybe_evict_cached_block": block_pool_cls._maybe_evict_cached_block,
        "reset_prefix_cache": block_pool_cls.reset_prefix_cache,
    }

    def _patched_get_new_blocks(self: Any, num_blocks: int) -> list[Any]:
        result = originals["get_new_blocks"](self, num_blocks)
        reserved_ids = _reserved_ids(self)
        if reserved_ids and any(_block_id(block) in reserved_ids for block in result):
            raise RuntimeError("BlockPool returned compact reserved blocks")
        return result

    def _patched_free_blocks(self: Any, ordered_blocks: Iterable[Any]) -> None:
        reserved_ids = _reserved_ids(self)
        if not reserved_ids:
            return originals["free_blocks"](self, ordered_blocks)
        if isinstance(ordered_blocks, (list, tuple)):
            has_reserved = any(
                _block_id(block) in reserved_ids for block in ordered_blocks
            )
            if not has_reserved:
                return originals["free_blocks"](self, ordered_blocks)
            ordinary_blocks = [
                block for block in ordered_blocks if _block_id(block) not in reserved_ids
            ]
        else:
            ordinary_iter = (
                block for block in ordered_blocks if _block_id(block) not in reserved_ids
            )
            return originals["free_blocks"](self, ordinary_iter)
        if ordinary_blocks:
            return originals["free_blocks"](self, ordinary_blocks)
        return None

    def _patched_touch(self: Any, blocks: Any) -> None:
        reserved_ids = _reserved_ids(self)
        if not reserved_ids:
            return originals["touch"](self, blocks)

        if not isinstance(blocks, (list, tuple)):
            blocks = list(blocks)
        ordinary_blocks = [
            block for block in blocks if _block_id(block) not in reserved_ids
        ]
        if len(ordinary_blocks) == len(blocks):
            return originals["touch"](self, blocks)
        if ordinary_blocks:
            return originals["touch"](self, ordinary_blocks)
        return None

    def _patched_maybe_evict_cached_block(self: Any, block: Any) -> bool:
        if _is_reserved_block(self, block):
            return False
        return originals["_maybe_evict_cached_block"](self, block)

    def _append_all_blocks_cleared_event(self: Any) -> None:
        if not bool(getattr(self, "enable_kv_cache_events", False)):
            return
        try:
            from vllm.distributed.kv_events import AllBlocksCleared  # type: ignore[import]
        except Exception as exc:
            raise RuntimeError(
                "compact page reset_prefix_cache requires AllBlocksCleared event"
            ) from exc
        getattr(self, "kv_event_queue").append(AllBlocksCleared())

    def _reset_sparse_completed_request_run() -> None:
        import sys

        controller = getattr(
            sys.modules.get("patches.vllm_sparse_patch"),
            "_GLOBAL_CONTROLLER",
            None,
        )
        if controller is None:
            return
        reset_run = getattr(controller, "reset_for_completed_request_run", None)
        if not callable(reset_run):
            raise RuntimeError(
                "sparse controller is missing completed-request reset owner"
            )
        # reset_prefix_cache is a lifecycle boundary. Continuing after an
        # async sparse-state drain failure can expose stale leases to the next
        # request run, so interface drift and reset failure both fail closed.
        reset_run()

    def _patched_reset_prefix_cache(self: Any) -> bool:
        reserved_ids = _reserved_ids(self)
        num_used_blocks = int(getattr(self, "num_gpu_blocks")) - int(
            self.get_num_free_blocks()
        )
        # Native BlockPool reset is conditional. Check that read-only boundary
        # before clearing request-local sparse state; otherwise a failed reset
        # leaves live native ownership paired with an empty sparse controller.
        if num_used_blocks != 1 + len(reserved_ids):
            return False

        _reset_sparse_completed_request_run()
        if not reserved_ids:
            # Keep the ordinary path owned by native vLLM after the shared
            # eligibility preflight; do not duplicate its hash/map/metrics/event
            # reset implementation here.
            reset_result = originals["reset_prefix_cache"](self)
            if reset_result is not True:
                raise RuntimeError(
                    "native reset_prefix_cache contradicted eligibility preflight"
                )
            return True

        cached_block_hash_to_block = getattr(self, "cached_block_hash_to_block", None)
        if cached_block_hash_to_block is not None:
            self.cached_block_hash_to_block = CompactBlockPoolAdapter(
                self
            ).compact_reserved_cache_only(cached_block_hash_to_block, reserved_ids)

        for block in getattr(self, "blocks"):
            if _block_id(block) not in reserved_ids:
                block.reset_hash()
        metrics_collector = getattr(self, "metrics_collector", None)
        if metrics_collector:
            # Parity with native BlockPool.reset_prefix_cache(): benchmark and
            # RLHF reset boundaries must not retain prefix-cache metrics from
            # the preceding request run.
            metrics_collector.reset()
        _append_all_blocks_cleared_event(self)
        return True

    block_pool_cls.get_new_blocks = _patched_get_new_blocks  # type: ignore[assignment]
    block_pool_cls.free_blocks = _patched_free_blocks  # type: ignore[assignment]
    block_pool_cls.touch = _patched_touch  # type: ignore[assignment]
    block_pool_cls._maybe_evict_cached_block = _patched_maybe_evict_cached_block  # type: ignore[assignment]
    block_pool_cls.reset_prefix_cache = _patched_reset_prefix_cache  # type: ignore[assignment]
    setattr(block_pool_cls, _ORIGINALS_ATTR, originals)
    setattr(block_pool_cls, _PATCHED_ATTR, True)
    return originals


def restore_compact_page_block_pool_methods(
    block_pool_cls: type[Any],
    originals: Optional[dict[str, Callable[..., Any]]] = None,
) -> None:
    """Restore BlockPool methods patched by compact-page residency."""
    saved_originals = originals or getattr(block_pool_cls, _ORIGINALS_ATTR, None)
    if not saved_originals:
        return
    for name, original in saved_originals.items():
        setattr(block_pool_cls, name, original)
    setattr(block_pool_cls, _PATCHED_ATTR, False)
    setattr(block_pool_cls, _ORIGINALS_ATTR, None)
