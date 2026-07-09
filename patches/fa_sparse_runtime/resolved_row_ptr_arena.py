"""Graph-stable row-pointer arena for the ResolvedRowPtr resolver.

The i32 row table remains the owned backing storage for compact+recent page
ids. The production graph ABI captures a stable u64 pointer vector plus visible
lengths, so replay can redirect rows without copying a SelectedTable carrier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import os
import time
import torch

from patches.fa3_native.mixed_page_graph_descriptor import (
    MixedPageResolverCarrierSet,
    PageResolverKind,
    ResolverGraphDescriptor,
)
from patches.fa3_native.row_consume_modes import (
    ROW_CONSUME_MODE_FULL_I32,
    ROW_CONSUME_MODE_SELECTED_I32,
)
from patches.sparse_constants import _DYNAMIC_ENV, compact_gen_count
# C++ host-derivation fast path for bind_production_row_table (env-gated,
# default OFF). The ext is built lazily on first enabled() use; importing
# the wrapper module here is cheap and never compiles anything on an OFF run.
from utils import rrp_bind_host_ext

_ROW_MODES = ("compact", "safe_full_recent", "native", "unset")
_WRITABLE_ROW_MODES = frozenset(_ROW_MODES[:-1])
_ROW_SOURCE_KEYS = (
    "compact_rows",
    "native_rows",
    "compact_reserved_pages",
    "recent_canonical_pages",
    "middle_native_canonical_pages",
    "native_canonical_pages",
    "compact_rows_with_reserved_pages",
    "compact_rows_with_recent_pages",
    "compact_full_native_fallback_rows",
)
SOURCE_COUNTER_SCHEMA_VERSION = 1
_AFFINE_ROW_PTR_FALLBACK_SEGMENT_PAGES = -1
# [ARENA-AFFINE-DEVICE-RETIRED 2026-07-03] The zero-consumer device tensors
# affine_i32 / affine_row_consume_mode_i32 and everything that ONLY served
# their maintenance writes were removed:
#   - B2 affine-publish env caches (_AFFINE_PUBLISH_CACHED /
#     _AFFINE_PUBLISH_LIVE_CACHED / _AFFINE_PUBLISH_TRACE_CACHED,
#     VLLM_SPARSE_PAGE_ADD_AFFINE_PUBLISH[_TRACE]) and the
#     _b2_affine_fire_trace diagnostic counter,
#   - the #12 v6 affine-clean write-skip cache
#     (_RRP_AFFINE_CLEAN_NARROW_CACHED / _rrp_affine_clean_narrow_enabled /
#     _affine_clean_last_key_by_row, VLLM_SPARSE_RRP_AFFINE_CLEAN_NARROW).
# The production chain never read either tensor (binder branches publish
# affine_i32=None; C++ row_consume_mode is always nullptr; AFFINE_TENSOR
# subkind fail-fast retired 2026-07-02). Host-side scalar affine bookkeeping
# (_direct_affine_scalar_* / _direct_affine_batch_bases /
# affine_batch_base_i32, the AffineConstDirect live path) is unaffected.
# Recover the original bodies from git history if ever needed.
# WF-4 env-cache: RRP direct-affine disable is a process-lifetime kill switch (never
# changes mid-run). Cache the env read once at import so the RRP bind hot path does not
# re-read os.environ every call. Same name/default/comparison -> byte-identical routing.
_RRP_DISABLE_DIRECT_AFFINE_CACHED = os.environ.get(
    "VLLM_SPARSE_RRP_DISABLE_DIRECT_AFFINE", "0"
) == "1"
_SOURCE_COUNTER_REQUIRED_ROW_SOURCE_KEYS = (
    "compact_rows",
    "native_canonical_pages",
    "compact_rows_with_reserved_pages",
    "compact_rows_with_recent_pages",
    "compact_reserved_pages",
    "recent_canonical_pages",
    "middle_native_canonical_pages",
    "native_rows",
    "compact_full_native_fallback_rows",
)


@dataclass(frozen=True, slots=True)
class PlannedCompactRowLayout:
    compact_ready_by_batch_row: tuple[bool, ...]
    slot_by_row: tuple[int, ...]
    compact_valid_tokens_by_row: tuple[int, ...]
    compact_offset_tokens_by_row: tuple[int, ...]
    recent_first_page_by_row: tuple[int, ...]
    recent_page_count_by_row: tuple[int, ...]
    row_effective_k_by_row: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ResolvedRowPtrDescriptorPayload:
    affine_descriptor_by_row: tuple[object, ...]
    row_table_pages_by_row: tuple[tuple[int, ...], ...]
    segment_pages_by_row: tuple[int, ...]


def _reject_tensor_metadata(value: object, *, label: str) -> None:
    if isinstance(value, torch.Tensor):
        raise TypeError(f"{label} must be CPU-owned metadata, not torch.Tensor")
    if isinstance(value, dict):
        for index, (key, item) in enumerate(value.items()):
            _reject_tensor_metadata(key, label=f"{label}.key[{index}]")
            _reject_tensor_metadata(item, label=f"{label}.value[{index}]")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_tensor_metadata(item, label=f"{label}[{index}]")


def _layout_tuple(values: Iterable[object], *, label: str) -> tuple[object, ...]:
    _reject_tensor_metadata(values, label=label)
    materialized = tuple(values)
    _reject_tensor_metadata(materialized, label=label)
    return materialized


def _affine_base_stride_from_cpu_sequence(
    values: Sequence[object],
    *,
    start: int,
    count: int,
) -> tuple[int, int] | None:
    start_i = int(start)
    count_i = int(count)
    if start_i < 0 or count_i <= 0 or start_i + count_i > len(values):
        return None
    base = int(values[start_i])
    if count_i == 1:
        return base, 1
    stride = int(values[start_i + 1]) - base
    for offset in range(2, count_i):
        if int(values[start_i + offset]) != base + stride * offset:
            return None
    return base, stride


def _prove_scalar_batch_affine(
    affine_rows_by_batch: Sequence[tuple[int, int, int, int, int]],
    affine_row_modes_by_batch: Sequence[int],
    *,
    batch_size: int,
) -> tuple[int, int, int] | None:
    """Host-only proof that per-batch 5-tuple affine descriptors collapse to ONE
    scalar batch-affine pattern eligible for AFFINE_CONST_DIRECT. Returns
    (base0, stride, batch_stride) or None (fail-closed -> keep AFFINE_TENSOR).
    Mirrors runtime_bridge has_affine_const_direct on the host: single-segment
    (recent segment continues the first stride), constant first stride, constant
    inter-batch base delta, every batch a genuine SELECTED single-segment row.
    Consumes only host int lists -- no CUDA reads."""
    if batch_size < 1 or len(affine_rows_by_batch) != batch_size:  # fa3_sm90_bs1_affine: was <2
        return None
    if len(affine_row_modes_by_batch) != batch_size:
        return None
    base0, stride0 = int(affine_rows_by_batch[0][0]), int(affine_rows_by_batch[0][1])
    bases: list[int] = []
    for batch in range(batch_size):
        b, st, seg, sb, ss = (int(v) for v in affine_rows_by_batch[batch])
        if int(affine_row_modes_by_batch[batch]) != ROW_CONSUME_MODE_SELECTED_I32:
            return None
        if seg == _AFFINE_ROW_PTR_FALLBACK_SEGMENT_PAGES or seg <= 0:
            return None
        if st != stride0:
            return None
        # single-segment <=> recent segment continues the first stride contiguously
        if ss != st or sb != b + st * seg:
            return None
        bases.append(b)
    if batch_size == 1:  # fa3_sm90_bs1_affine
        return base0, stride0, 0, None  # fa3_sm90_perbatch_base: 4-tuple
    batch_stride = bases[1] - bases[0]
    delta_const = all(
        bases[batch] - bases[batch - 1] == batch_stride for batch in range(1, batch_size)
    )
    if delta_const:
        return base0, stride0, batch_stride, None  # fa3_sm90_perbatch_base: scalar path
    # fa3_sm90_perbatch_base (gap C): constant intra-stride, non-constant inter-base
    # delta -> per-batch base array instead of rejecting to AFFINE_TENSOR.
    return base0, stride0, None, tuple(bases)


def _cpu_block_table_row_source(
    block_table_cpu: object | None,
    row: int,
) -> Sequence[object] | None:
    if block_table_cpu is None:
        return None
    row_i = int(row)
    if row_i < 0:
        return None
    if isinstance(block_table_cpu, torch.Tensor):
        if block_table_cpu.device.type != "cpu" or block_table_cpu.dim() != 2:
            return None
        if int(block_table_cpu.shape[0]) <= row_i:
            return None
        return block_table_cpu[row_i]
    try:
        row_count = len(block_table_cpu)  # type: ignore[arg-type]
    except TypeError:
        return None
    if int(row_count) <= row_i:
        return None
    try:
        row_values = block_table_cpu[row_i]  # type: ignore[index]
    except Exception:
        return None
    if isinstance(row_values, torch.Tensor) and row_values.device.type != "cpu":
        return None
    return row_values


def apply_arena_writes_from_descriptors(
    arena,
    derived: dict,
    *,
    reserved: Sequence[int] | tuple[()] = (),
    reserved_tensor: torch.Tensor | None = None,
    canonical_i32: torch.Tensor,
    safe_page_id: int | None = None,
    page_size: int | None = None,
    bind_row_pointers: bool = True,
) -> None:
    """Reproduce bind_production_row_table CUDA writes from a derivation dict.

    Byte-exact mirror of resolved_row_ptr_arena.py:2008-2264, driven by the
    per-batch ``write_descriptors`` in ``derived`` -- NO geometry re-derivation.
    Used by the env-gated C++ host-derivation fast path in
    ``bind_production_row_table``. ``bind_row_pointers`` is False when the caller
    already ran ``_bind_own_row_table_pointers`` (the fast path does).
    """
    del page_size  # geometry is fully carried by the descriptors

    num_kv_heads = int(arena.num_kv_heads)

    if bind_row_pointers:
        arena._bind_own_row_table_pointers()

    # SEQUSED_VECTORIZE: hoist the per-batch seqused publish to TWO
    # vectorized copies (byte-exact; same pattern as the direct-affine path).
    # Defensive: only take this path when the host emitted the
    # ``effective_k_seq`` key (newer .so); otherwise fall back to the
    # original per-descriptor ``.fill_`` below (no KeyError on old .so).
    _seqused_effective_k_seq = derived.get("effective_k_seq")
    if _seqused_effective_k_seq is not None:
        _seqused_batch_size = int(arena.batch_size)
        _seqused_re = arena.batch_seqused_k_i32.new_tensor(
            _seqused_effective_k_seq
        )
        arena.batch_seqused_k_i32.copy_(_seqused_re, non_blocking=True)
        arena.seqused_k_i32.view(_seqused_batch_size, num_kv_heads).copy_(
            _seqused_re.reshape(_seqused_batch_size, 1).expand(
                _seqused_batch_size, num_kv_heads
            ),
            non_blocking=True,
        )

    for desc in derived["write_descriptors"]:
        branch = desc["branch"]
        row_start = int(desc["row_start"])
        row_stop = int(desc["row_stop"])
        row_slice = slice(row_start, row_stop)
        effective_k = int(desc["effective_k"])
        batch = row_start // num_kv_heads

        # SEQUSED_VECTORIZE: per-row seqused fills retained ONLY as the
        # fallback for old host .so missing ``effective_k_seq``. When the
        # vectorized write above ran, skip these (byte-identical result).
        if _seqused_effective_k_seq is None:
            arena.batch_seqused_k_i32[batch].fill_(effective_k)
            arena.seqused_k_i32[row_slice].fill_(effective_k)

        if branch == "inactive":
            sp = (
                int(desc["safe_page_id"])
                if safe_page_id is None
                else int(safe_page_id)
            )
            arena.row_table_i32[row_slice].fill_(sp)
            continue

        if branch == "compact":
            compact_pages = int(desc["compact_pages"])
            slot_start = int(desc["slot_start"])
            slot_end = int(desc["slot_end"])
            recent_visible_pages = int(desc["recent_visible_pages"])
            recent_dst = int(desc["recent_dst"])
            recent_first = int(desc["recent_first"])
            canonical_row = int(desc["canonical_row"])
            if compact_pages:
                if reserved_tensor is None:
                    compact_src = arena.row_table_i32.new_tensor(
                        reserved[slot_start:slot_end]
                    )
                else:
                    compact_src = reserved_tensor.narrow(
                        0, slot_start, compact_pages
                    )
                arena.row_table_i32[row_slice, :compact_pages].copy_(
                    compact_src.reshape(1, compact_pages).expand(
                        num_kv_heads,
                        compact_pages,
                    ),
                    non_blocking=True,
                )
            if recent_visible_pages:
                recent_src = canonical_i32[
                    canonical_row,
                    recent_first : recent_first + recent_visible_pages,
                ]
                arena.row_table_i32[
                    row_slice,
                    recent_dst : recent_dst + recent_visible_pages,
                ].copy_(
                    recent_src.reshape(1, recent_visible_pages).expand(
                        num_kv_heads,
                        recent_visible_pages,
                    ),
                    non_blocking=True,
                )
            continue

        if branch == "native":
            visible_pages = int(desc["visible_pages"])
            canonical_row = int(desc["canonical_row"])
            if visible_pages:
                native_src = canonical_i32[canonical_row, :visible_pages]
                arena.row_table_i32[row_slice, :visible_pages].copy_(
                    native_src.reshape(1, visible_pages).expand(
                        num_kv_heads,
                        visible_pages,
                    ),
                    non_blocking=True,
                )
            continue

        raise ValueError(f"unknown write descriptor branch: {branch!r}")

    arena.coverage_count_by_row_head = tuple(derived["coverage"])
    arena.compact_ready_by_row_head = tuple(
        bool(v) for v in derived["compact_ready"]
    )
    arena.row_mode_by_row_head = tuple(str(v) for v in derived["row_modes"])
    arena.row_source_distribution = dict(derived["row_sources"])

    arena._direct_affine_scalar_base = None
    arena._direct_affine_scalar_stride = None
    arena._direct_affine_scalar_batch_stride = None
    scalar_affine = derived["scalar_collapse"]
    if scalar_affine is not None:
        arena._direct_affine_scalar_base = int(scalar_affine[0])
        arena._direct_affine_scalar_stride = int(scalar_affine[1])
        arena._direct_affine_scalar_batch_stride = (  # fa3_sm90_perbatch_base
            int(scalar_affine[2]) if scalar_affine[2] is not None else None)
        arena._direct_affine_batch_bases = (
            tuple(int(x) for x in scalar_affine[3])
            if len(scalar_affine) > 3 and scalar_affine[3] is not None else None)
        if arena._direct_affine_batch_bases is not None:
            arena.affine_batch_base_i32.copy_(
                arena.affine_batch_base_i32.new_tensor(arena._direct_affine_batch_bases),
                non_blocking=True)

    # [ARENA-AFFINE-DEVICE-RETIRED 2026-07-03] The whole-arena affine_i32 /
    # affine_row_consume_mode_i32 device rewrites (and their v6 write-skip cache
    # clear) were removed; the host-derived affine rows now only drive the
    # readiness boolean below.
    arena._direct_affine_ready = bool(derived["affine_rows_by_batch"])
    arena._direct_affine_row_mode_required = False

    _b1_last_pages_by_row = getattr(arena, "_b1_last_pages_by_row", None)
    if _b1_last_pages_by_row is not None:
        _b1_last_pages_by_row.clear()

    arena.generation += 1


def _planned_bool_tuple(
    values: Iterable[object],
    *,
    label: str,
    batch_size: int,
) -> tuple[bool, ...]:
    materialized = _layout_tuple(values, label=label)
    if len(materialized) != batch_size:
        raise ValueError(f"{label} length must equal batch_size")
    return tuple(bool(value) for value in materialized)


def _planned_int_tuple(
    values: Iterable[object],
    *,
    label: str,
    batch_size: int,
) -> tuple[int, ...]:
    materialized = _layout_tuple(values, label=label)
    if len(materialized) != batch_size:
        raise ValueError(f"{label} length must equal batch_size")
    return tuple(int(value) for value in materialized)


def build_planned_compact_row_layout(
    *,
    batch_size: int,
    compact_ready_by_batch_row: Iterable[object],
    slot_by_row: Iterable[object],
    compact_valid_tokens_by_row: Iterable[object],
    compact_offset_tokens_by_row: Iterable[object],
    recent_first_page_by_row: Iterable[object],
    recent_page_count_by_row: Iterable[object],
    row_effective_k_by_row: Iterable[object],
) -> PlannedCompactRowLayout:
    batch_size_i = _validate_positive(batch_size, label="batch_size")
    layout = PlannedCompactRowLayout(
        compact_ready_by_batch_row=_planned_bool_tuple(
            compact_ready_by_batch_row,
            label="compact_ready_by_batch_row",
            batch_size=batch_size_i,
        ),
        slot_by_row=_planned_int_tuple(
            slot_by_row,
            label="slot_by_row",
            batch_size=batch_size_i,
        ),
        compact_valid_tokens_by_row=_planned_int_tuple(
            compact_valid_tokens_by_row,
            label="compact_valid_tokens_by_row",
            batch_size=batch_size_i,
        ),
        compact_offset_tokens_by_row=_planned_int_tuple(
            compact_offset_tokens_by_row,
            label="compact_offset_tokens_by_row",
            batch_size=batch_size_i,
        ),
        recent_first_page_by_row=_planned_int_tuple(
            recent_first_page_by_row,
            label="recent_first_page_by_row",
            batch_size=batch_size_i,
        ),
        recent_page_count_by_row=_planned_int_tuple(
            recent_page_count_by_row,
            label="recent_page_count_by_row",
            batch_size=batch_size_i,
        ),
        row_effective_k_by_row=_planned_int_tuple(
            row_effective_k_by_row,
            label="row_effective_k_by_row",
            batch_size=batch_size_i,
        ),
    )
    assert len(layout.compact_ready_by_batch_row) == batch_size_i
    assert len(layout.slot_by_row) == batch_size_i
    assert len(layout.compact_valid_tokens_by_row) == batch_size_i
    assert len(layout.compact_offset_tokens_by_row) == batch_size_i
    assert len(layout.recent_first_page_by_row) == batch_size_i
    assert len(layout.recent_page_count_by_row) == batch_size_i
    assert len(layout.row_effective_k_by_row) == batch_size_i
    return layout


def build_resolved_row_ptr_descriptor_payload(
    *,
    batch_size: int,
    canonical_block_table: object | None = None,
    compact_ready_by_batch_row: Iterable[object],
    row_effective_k_by_row: Iterable[object],
    page_size: int,
    reserved_manager_block_ids: Iterable[int] = (),
    slot_by_row: Iterable[int] = (),
    compact_valid_tokens_by_row: Iterable[int] = (),
    compact_offset_tokens_by_row: Iterable[int] = (),
    recent_first_page_by_row: Iterable[int] = (),
    canonical_row_index_by_batch_row: Iterable[int] = (),
    safe_page_id: int = 0,
    compact_capacity_pages: int = 0,
    max_pages_per_row: int = 0,
    planned_layout: PlannedCompactRowLayout | None = None,
    reserved_manager_block_ids_cpu: Iterable[int] | None = None,
    canonical_block_table_cpu: object | None = None,
) -> ResolvedRowPtrDescriptorPayload:
    batch_size_i = _validate_positive(batch_size, label="batch_size")
    page_size_i = _validate_positive(page_size, label="page_size")
    max_pages_i = _validate_positive(max_pages_per_row, label="max_pages_per_row")
    safe_page_id_i = _as_int(safe_page_id, label="safe_page_id")
    if safe_page_id_i < 0:
        raise ValueError("safe_page_id must be non-negative")

    canonical_row_source = _layout_tuple(
        canonical_row_index_by_batch_row,
        label="canonical_row_index_by_batch_row",
    )
    if canonical_row_source:
        if len(canonical_row_source) != batch_size_i:
            raise ValueError("canonical_row_index_by_batch_row length must match batch_size")
        canonical_row_by_batch = tuple(int(row) for row in canonical_row_source)
    else:
        canonical_row_by_batch = tuple(range(batch_size_i))

    layout = planned_layout
    if layout is None:
        slot_source = _layout_tuple(slot_by_row, label="slot_by_row")
        compact_valid_source = _layout_tuple(
            compact_valid_tokens_by_row,
            label="compact_valid_tokens_by_row",
        )
        compact_offset_source = _layout_tuple(
            compact_offset_tokens_by_row,
            label="compact_offset_tokens_by_row",
        )
        recent_first_source = _layout_tuple(
            recent_first_page_by_row,
            label="recent_first_page_by_row",
        )
        layout = build_planned_compact_row_layout(
            batch_size=batch_size_i,
            compact_ready_by_batch_row=compact_ready_by_batch_row,
            slot_by_row=slot_source if slot_source else range(batch_size_i),
            compact_valid_tokens_by_row=compact_valid_source
            if compact_valid_source
            else (0,) * batch_size_i,
            compact_offset_tokens_by_row=compact_offset_source
            if compact_offset_source
            else (0,) * batch_size_i,
            recent_first_page_by_row=recent_first_source
            if recent_first_source
            else (0,) * batch_size_i,
            recent_page_count_by_row=(0,) * batch_size_i,
            row_effective_k_by_row=row_effective_k_by_row,
        )
    else:
        if len(layout.compact_ready_by_batch_row) != batch_size_i:
            raise ValueError("planned_layout compact_ready_by_batch_row length must match batch_size")
        if len(layout.slot_by_row) != batch_size_i:
            raise ValueError("planned_layout slot_by_row length must match batch_size")
        if len(layout.compact_valid_tokens_by_row) != batch_size_i:
            raise ValueError("planned_layout compact_valid_tokens_by_row length must match batch_size")
        if len(layout.compact_offset_tokens_by_row) != batch_size_i:
            raise ValueError("planned_layout compact_offset_tokens_by_row length must match batch_size")
        if len(layout.recent_first_page_by_row) != batch_size_i:
            raise ValueError("planned_layout recent_first_page_by_row length must match batch_size")
        if len(layout.row_effective_k_by_row) != batch_size_i:
            raise ValueError("planned_layout row_effective_k_by_row length must match batch_size")

    if reserved_manager_block_ids_cpu is not None:
        reserved_cpu_values = tuple(int(v) for v in reserved_manager_block_ids_cpu)
    elif isinstance(reserved_manager_block_ids, torch.Tensor):
        if reserved_manager_block_ids.device.type != "cpu":
            reserved_cpu_values = None
        else:
            reserved_cpu_values = tuple(int(v) for v in reserved_manager_block_ids.tolist())
    else:
        reserved_cpu_values = tuple(int(v) for v in reserved_manager_block_ids)
    reserved_len = 0 if reserved_cpu_values is None else len(reserved_cpu_values)

    canonical_cpu_source = (
        canonical_block_table
        if isinstance(canonical_block_table, torch.Tensor)
        and canonical_block_table.device.type == "cpu"
        else canonical_block_table_cpu
    )
    batch_has_compact_row = any(bool(v) for v in layout.compact_ready_by_batch_row)
    compact_capacity_i = int(compact_capacity_pages)
    affine_descriptors: list[object] = []
    row_table_pages: list[tuple[int, ...]] = []
    segment_pages_values: list[int] = []

    def _fallback_segment(pages: tuple[int, ...]) -> int:
        return _AFFINE_ROW_PTR_FALLBACK_SEGMENT_PAGES if pages else 0

    for batch in range(batch_size_i):
        effective_k = max(0, int(layout.row_effective_k_by_row[batch]))
        canonical_row = int(canonical_row_by_batch[batch])
        if canonical_row < 0:
            pages = (safe_page_id_i,) if effective_k > 0 else tuple()
            affine_descriptors.append("row_table_fallback")
            row_table_pages.append(pages)
            segment_pages_values.append(_fallback_segment(pages))
            continue

        canonical_cpu_row = _cpu_block_table_row_source(canonical_cpu_source, canonical_row)
        if canonical_cpu_row is None:
            raise ValueError("canonical_block_table_cpu must cover descriptor payload rows")

        if bool(layout.compact_ready_by_batch_row[batch]):
            compact_tokens = max(0, int(layout.compact_valid_tokens_by_row[batch]))
            if compact_tokens % page_size_i != 0:
                raise ValueError("compact_valid_tokens must be page-aligned")
            compact_pages = compact_tokens // page_size_i
            compact_offset_tokens_i = int(layout.compact_offset_tokens_by_row[batch])
            if compact_offset_tokens_i < 0 or compact_offset_tokens_i % page_size_i != 0:
                raise ValueError("compact_offset_tokens must be non-negative and page-aligned")
            compact_offset_pages = compact_offset_tokens_i // page_size_i
            if compact_pages > compact_capacity_i:
                raise ValueError("compact page count exceeds compact_capacity_pages")
            if effective_k < compact_tokens:
                raise ValueError("row_effective_k is smaller than compact tokens")
            recent_visible_tokens = effective_k - compact_tokens
            recent_visible_pages = (
                _ceil_div(recent_visible_tokens, page_size_i)
                if recent_visible_tokens > 0
                else 0
            )
            visible_pages = compact_pages + recent_visible_pages
            if visible_pages > max_pages_i:
                raise ValueError("visible compact/recent pages exceed row width")
            slot = int(layout.slot_by_row[batch])
            # [DUAL-GEN-L2a] 双代下 slot 的合法 span 为 gen0/gen1 两个候选半区
            # (gen1 基址=半区页数=reserved_len/factor);开关关时 factor=1、
            # 候选只剩 gen0=历史判定逐位。offset 由读侧 state 权威产生,此处
            # 按其落点选半区再做同款窗校验。
            _gen_count = compact_gen_count()
            _gen_stride_pages = reserved_len // _gen_count if _gen_count > 0 else 0
            _gen_of_offset = (
                1
                if (_gen_count > 1 and compact_offset_pages >= _gen_stride_pages)
                else 0
            )
            expected_slot_start = (
                _gen_of_offset * _gen_stride_pages + slot * compact_capacity_i
            )
            expected_slot_end = expected_slot_start + compact_capacity_i
            if (
                reserved_cpu_values is None
                or slot < 0
                or expected_slot_start < 0
                or expected_slot_end > reserved_len
            ):
                raise ValueError("reserved_manager_block_ids_cpu must cover compact slots")
            slot_start = compact_offset_pages
            slot_end = slot_start + compact_pages
            if slot_start < expected_slot_start or slot_end > expected_slot_end:
                raise ValueError("compact offset plus page count exceeds slot compact span")
            if slot_start < 0 or slot_end > reserved_len:
                raise ValueError("compact offset plus page count exceeds reserved_manager_block_ids")
            recent_first = int(layout.recent_first_page_by_row[batch])
            if recent_first < 0 or recent_first + recent_visible_pages > len(canonical_cpu_row):
                raise ValueError("recent page range exceeds canonical block table")
            compact_page_values = (
                tuple(int(v) for v in reserved_cpu_values[slot_start:slot_end])
                if compact_pages
                else tuple()
            )
            recent_page_values = (
                tuple(
                    int(canonical_cpu_row[recent_first + offset])
                    for offset in range(recent_visible_pages)
                )
                if recent_visible_pages
                else tuple()
            )
            pages = compact_page_values + recent_page_values
            selected_affine: tuple[int, int, int, int, int] | None = None
            if compact_pages > 0:
                compact_affine = _affine_base_stride_from_cpu_sequence(
                    reserved_cpu_values,
                    start=slot_start,
                    count=compact_pages,
                )
                if compact_affine is not None:
                    if recent_visible_pages > 0:
                        recent_affine = _affine_base_stride_from_cpu_sequence(
                            canonical_cpu_row,
                            start=recent_first,
                            count=recent_visible_pages,
                        )
                    else:
                        recent_affine = (
                            compact_affine[0] + compact_affine[1] * (compact_pages - 1),
                            1,
                        )
                    if recent_affine is not None:
                        selected_affine = (
                            compact_affine[0],
                            compact_affine[1],
                            compact_pages,
                            recent_affine[0],
                            recent_affine[1],
                        )
            if selected_affine is None:
                affine_descriptors.append("row_table_fallback")
                segment_pages_values.append(_fallback_segment(pages))
            else:
                affine_descriptors.append(selected_affine)
                segment_pages_values.append(int(compact_pages))
            row_table_pages.append(pages)
            continue

        visible_pages = min(
            _ceil_div(effective_k, page_size_i) if effective_k > 0 else 0,
            len(canonical_cpu_row),
            max_pages_i,
        )
        pages = tuple(int(canonical_cpu_row[offset]) for offset in range(visible_pages))
        if batch_has_compact_row:
            affine_descriptors.append("row_table_fallback")
            segment_pages_values.append(_fallback_segment(pages))
        else:
            native_affine = (
                _affine_base_stride_from_cpu_sequence(
                    canonical_cpu_row,
                    start=0,
                    count=visible_pages,
                )
                if visible_pages > 0
                else None
            )
            if native_affine is None:
                affine_descriptors.append("row_table_fallback")
                segment_pages_values.append(_fallback_segment(pages))
            else:
                native_base, native_stride = native_affine
                affine_descriptors.append(
                    (
                        native_base,
                        native_stride,
                        visible_pages,
                        native_base + native_stride * max(visible_pages - 1, 0),
                        1,
                    )
                )
                segment_pages_values.append(int(visible_pages))
        row_table_pages.append(pages)

    return ResolvedRowPtrDescriptorPayload(
        affine_descriptor_by_row=tuple(affine_descriptors),
        row_table_pages_by_row=tuple(row_table_pages),
        segment_pages_by_row=tuple(int(v) for v in segment_pages_values),
    )


def build_row_mode_distribution(row_modes: Iterable[str]) -> dict[str, int]:
    distribution = {mode: 0 for mode in _ROW_MODES}
    for mode in row_modes:
        if mode not in distribution:
            raise ValueError(f"unsupported row mode: {mode!r}")
        distribution[mode] += 1
    return distribution


def empty_row_source_distribution() -> dict[str, int]:
    return {key: 0 for key in _ROW_SOURCE_KEYS}


def build_source_counter_fields(
    *,
    batch_size: int,
    num_kv_heads: int,
    row_source_distribution: dict[str, int] | None,
) -> dict[str, object]:
    distribution = row_source_distribution if isinstance(row_source_distribution, dict) else {}
    missing_fields = tuple(
        key for key in _SOURCE_COUNTER_REQUIRED_ROW_SOURCE_KEYS if key not in distribution
    )
    return {
        "source_counter_schema_version": SOURCE_COUNTER_SCHEMA_VERSION,
        "expected_rows": _validate_positive(batch_size, label="batch_size"),
        "num_kv_heads": _validate_positive(num_kv_heads, label="num_kv_heads"),
        "source_counter_missing_fields": missing_fields,
    }


def _as_int(value: int, *, label: str) -> int:
    value_i = int(value)
    if value_i != value:
        raise ValueError(f"{label} must be an integer")
    return value_i


def _validate_positive(value: int, *, label: str) -> int:
    value_i = _as_int(value, label=label)
    if value_i <= 0:
        raise ValueError(f"{label} must be positive")
    return value_i


# Grid granularity (in query rows) used to bucket max_seqlen_q before it is
# baked into the resolver-graph replay key. The captured kernel grid is tiled on
# max_seqlen_q; bucketing UP to the next multiple of this small grid means the
# captured bucket is always >= any live max_seqlen_q for the same bucket, so a
# replay can never under-size the captured grid (OOB mis-tile). A coarser bucket
# only forces an occasional benign recapture.
_MAX_SEQLEN_Q_REPLAY_BUCKET_GRID = 8


def _bucket_max_seqlen_q(max_seqlen_q: int) -> int:
    value_i = int(max_seqlen_q)
    if value_i <= 1:
        return 1
    grid = _MAX_SEQLEN_Q_REPLAY_BUCKET_GRID
    return ((value_i + grid - 1) // grid) * grid


def _bucket_max_seqlen_k(max_seqlen_k: int) -> int:
    # K-side mirror of _bucket_max_seqlen_q. The KV extent ranges from the sparse
    # compact-recent bound up to a demoted-dense full context, so use a coarse
    # power-of-2 bucket (few recaptures). Rounding UP guarantees the captured
    # SplitKV scratch (num_splits/out_partial sized from this bound) is always >=
    # any live KV extent in the same bucket -> replay can never under-size it.
    value_i = int(max_seqlen_k)
    if value_i <= 1:
        return 1
    bucket = 1
    while bucket < value_i:
        bucket <<= 1
    return bucket


def _validate_flat_row(flat_row: int, *, total_rows: int) -> int:
    flat_row_i = _as_int(flat_row, label="row_head_index")
    if flat_row_i < 0 or flat_row_i >= total_rows:
        raise IndexError("row_head_index out of range")
    return flat_row_i


def _ceil_div(value: int, divisor: int) -> int:
    return (int(value) + int(divisor) - 1) // int(divisor)


def _replace_tuple_value(values: tuple, index: int, value) -> tuple:
    return values[:index] + (value,) + values[index + 1 :]


def _validate_arena_contract(
    arena: "ResolvedRowPtrArena",
    *,
    label: str,
    batch_size: int,
    num_kv_heads: int,
    max_pages_per_row: int,
) -> None:
    if arena.batch_size != batch_size:
        raise ValueError(f"{label} batch_size must match batch_size")
    if arena.num_kv_heads != num_kv_heads:
        raise ValueError(f"{label} num_kv_heads must match num_kv_heads")
    if arena.max_pages_per_row != max_pages_per_row:
        raise ValueError(f"{label} max_pages_per_row must match max_pages_per_row")

    total_rows = batch_size * num_kv_heads
    if tuple(arena.row_table_i32.shape) != (total_rows, max_pages_per_row):
        raise ValueError(f"{label} row_table_i32 shape mismatch")
    if tuple(arena.carrier_u64.shape) != (total_rows,):
        raise ValueError(f"{label} carrier_u64 shape mismatch")
    if tuple(getattr(arena, "batch_seqused_k_i32", ()).shape) != (batch_size,):
        raise ValueError(f"{label} batch_seqused_k_i32 shape mismatch")


@dataclass(frozen=True, slots=True)
class ResolvedRowPtrReplayMetadataBinding:
    descriptor: ResolverGraphDescriptor
    replay_arena: "ResolvedRowPtrArena"
    source_arena: "ResolvedRowPtrArena"
    replay_carriers: MixedPageResolverCarrierSet
    source_carriers: MixedPageResolverCarrierSet
    pointer_signature: tuple[int | None, ...]
    row_mode_distribution: dict[str, int]
    row_source_distribution: dict[str, int]
    source_counter_schema_version: int
    expected_rows: int
    num_kv_heads: int
    source_counter_missing_fields: tuple[str, ...]


def validate_block_table_row_pointer_source(
    *,
    block_table: object,
    batch_size: int,
    max_pages_per_row: int,
    device: torch.device | str,
) -> torch.Tensor:
    batch_size_i = _validate_positive(batch_size, label="batch_size")
    max_pages_i = _validate_positive(max_pages_per_row, label="max_pages_per_row")
    torch_device = torch.device(device)
    if not isinstance(block_table, torch.Tensor):
        raise TypeError("block_table must be a torch.Tensor")
    if block_table.dtype != torch.int32:
        raise ValueError("block_table must have dtype torch.int32")
    if block_table.dim() != 2:
        raise ValueError("block_table must be rank 2")
    if block_table.device != torch_device:
        raise ValueError("block_table device must match arena device")
    if int(block_table.shape[0]) < batch_size_i:
        raise ValueError("block_table rows must cover batch_size")
    if int(block_table.shape[1]) < max_pages_i:
        raise ValueError("block_table cols must cover max_pages_per_row")
    if int(block_table.stride(1)) != 1:
        raise ValueError("block_table last-dim stride must be 1")
    return block_table


@dataclass(slots=True)
class ResolvedRowPtrArena:
    batch_size: int
    num_kv_heads: int
    max_pages_per_row: int
    row_table_i32: torch.Tensor
    carrier_u64: torch.Tensor
    # [ARENA-AFFINE-DEVICE-RETIRED 2026-07-03] The zero-consumer device tensors
    # ``affine_i32`` ([total_rows, 5] five-tuples) and
    # ``affine_row_consume_mode_i32`` ([batch_size]) used to live here. No
    # production reader existed (binder publishes affine_i32=None; C++
    # row_consume_mode is always nullptr), so the fields, their allocation and
    # every maintenance write were removed. Recover from git history if needed.
    affine_batch_base_i32: torch.Tensor  # fa3_sm90_perbatch_base (gap C)
    seqused_k_i32: torch.Tensor
    batch_seqused_k_i32: torch.Tensor
    coverage_count_by_row_head: tuple[int, ...]
    compact_ready_by_row_head: tuple[bool, ...]
    row_mode_by_row_head: tuple[str, ...]
    row_source_distribution: dict[str, int] = field(default_factory=empty_row_source_distribution)
    generation: int = 0
    mixed_page_resolver_replay_ready_event: object | None = None
    mixed_page_resolver_replay_ready_event_generation: int = -1
    mixed_page_resolver_replay_ready_event_waited_generation: int = -1
    mixed_page_resolver_replay_ready_event_stream: int = -1
    _carriers: MixedPageResolverCarrierSet = field(init=False, repr=False)
    _row_batch_index_i64: torch.Tensor = field(init=False, repr=False)
    _row_head_index_i64: torch.Tensor = field(init=False, repr=False)
    _direct_affine_ready: bool = field(default=False, init=False, repr=False)
    _direct_affine_row_mode_required: bool = field(default=False, init=False, repr=False)
    _direct_affine_binding_active: bool = field(default=False, init=False, repr=False)
    # Whole-arena scalar batch-affine collapse proven on host in bind_production_row_table;
    # None => not provable => keep AFFINE_TENSOR (fail-closed). Plain ints, never tensors.
    _direct_affine_scalar_base: int | None = field(default=None, init=False, repr=False)
    _direct_affine_scalar_stride: int | None = field(default=None, init=False, repr=False)
    _direct_affine_scalar_batch_stride: int | None = field(default=None, init=False, repr=False)
    _direct_affine_batch_bases: tuple[int, ...] | None = field(default=None, init=False, repr=False)  # fa3_sm90_perbatch_base
    _carrier_points_to_own_row_table: bool = field(default=True, init=False, repr=False)
    _last_debug_pointer_signature: tuple[int | None, ...] = field(default=(), init=False, repr=False)
    _resolved_seqused_source_kind: str = field(default="arena_batch_seqused", init=False, repr=False)
    # B1 incremental-column-write per-batch-row baseline (page ids last
    # written into row_table_i32). None => B1 never ran / off. Declared as
    # a slot field because the arena is @dataclass(slots=True) with no
    # __dict__; without it B1 cannot store its baseline.
    _b1_last_pages_by_row: object = field(default=None, init=False, repr=False)
    # [LIVE-PAGES-PINNED-ASYNC 2026-07-08] publish_live_rows pinned staging
    # ring + persistent GPU staging (slots-declared for the same reason).
    _live_pages_pinned_ring: object = field(default=None, init=False, repr=False)
    _live_pages_pinned_ring_idx: int = field(default=0, init=False, repr=False)
    _live_pages_staging_gpu: object = field(default=None, init=False, repr=False)
    # [ARENA-AFFINE-DEVICE-RETIRED 2026-07-03] The #12 v6 affine-clean
    # write-skip cache slot ``_affine_clean_last_key_by_row`` was removed with
    # the affine device tensors it gated (it only ever skipped those two
    # device writes). Recover from git history if needed.

    def __post_init__(self) -> None:
        self._carriers = MixedPageResolverCarrierSet(
            row_consume_mode_i32=None,
            resolver_visible_seqused_k_by_head_i32=self.batch_seqused_k_i32,
            selected_page_table_i32=None,
            effective_row_slot_i32=None,
            compact_base_page_i32=None,
            compact_page_count_i32=None,
            recent_first_logical_page_i32=None,
            resolved_page_table_row_ptr_u64=self.carrier_u64,
        )
        self._row_batch_index_i64 = torch.arange(
            self.batch_size,
            dtype=torch.int64,
            device=self.carrier_u64.device,
        ).repeat_interleave(self.num_kv_heads)
        self._row_head_index_i64 = torch.arange(
            self.total_rows,
            dtype=torch.int64,
            device=self.carrier_u64.device,
        )

    @classmethod
    def allocate(
        cls,
        *,
        batch_size: int,
        num_kv_heads: int,
        max_pages_per_row: int,
        device: torch.device | str | None = None,
    ) -> "ResolvedRowPtrArena":
        batch_size_i = _validate_positive(batch_size, label="batch_size")
        num_kv_heads_i = _validate_positive(num_kv_heads, label="num_kv_heads")
        max_pages_i = _validate_positive(max_pages_per_row, label="max_pages_per_row")
        total_rows = batch_size_i * num_kv_heads_i
        torch_device = torch.device("cpu") if device is None else torch.device(device)

        row_table_i32 = torch.full(
            (total_rows, max_pages_i),
            -1,
            dtype=torch.int32,
            device=torch_device,
        )
        row_stride_bytes = int(row_table_i32.stride(0)) * int(
            row_table_i32.element_size()
        )
        carrier_u64 = int(row_table_i32.data_ptr()) + (
            torch.arange(total_rows, dtype=torch.int64, device=torch_device)
            * row_stride_bytes
        )
        # [ARENA-AFFINE-DEVICE-RETIRED 2026-07-03] affine_i32 /
        # affine_row_consume_mode_i32 allocations removed.
        affine_batch_base_i32 = torch.zeros(  # fa3_sm90_perbatch_base (gap C)
            (batch_size_i,), dtype=torch.int32, device=torch_device,
        )
        seqused_k_i32 = torch.zeros(
            (total_rows,),
            dtype=torch.int32,
            device=torch_device,
        )
        batch_seqused_k_i32 = torch.zeros(
            (batch_size_i,),
            dtype=torch.int32,
            device=torch_device,
        )
        return cls(
            batch_size=batch_size_i,
            num_kv_heads=num_kv_heads_i,
            max_pages_per_row=max_pages_i,
            row_table_i32=row_table_i32,
            carrier_u64=carrier_u64,
            affine_batch_base_i32=affine_batch_base_i32,  # fa3_sm90_perbatch_base
            seqused_k_i32=seqused_k_i32,
            batch_seqused_k_i32=batch_seqused_k_i32,
            coverage_count_by_row_head=(0,) * total_rows,
            compact_ready_by_row_head=(False,) * total_rows,
            row_mode_by_row_head=("unset",) * total_rows,
        )

    @property
    def total_rows(self) -> int:
        return self.batch_size * self.num_kv_heads

    @property
    def carriers(self) -> MixedPageResolverCarrierSet:
        return self._carriers

    @property
    def direct_affine_ready(self) -> bool:
        return bool(self._direct_affine_ready)

    @property
    def direct_affine_row_mode_required(self) -> bool:
        return bool(self._direct_affine_row_mode_required)

    @property
    def direct_affine_scalar(self) -> tuple[int, int, int] | None:
        if (
            self._direct_affine_scalar_base is None
            or self._direct_affine_scalar_stride is None
            or self._direct_affine_scalar_batch_stride is None
        ):
            return None
        return (
            int(self._direct_affine_scalar_base),
            int(self._direct_affine_scalar_stride),
            int(self._direct_affine_scalar_batch_stride),
        )

    @property
    def direct_affine_batch_bases(self) -> tuple[int, tuple[int, ...]] | None:
        # fa3_sm90_perbatch_base: (stride, per-batch bases) when the scalar collapse
        # held a constant intra-stride but non-constant inter-batch base -> per-batch array.
        if self._direct_affine_batch_bases is None or self._direct_affine_scalar_stride is None:
            return None
        return (
            int(self._direct_affine_scalar_stride),
            tuple(int(x) for x in self._direct_affine_batch_bases),
        )

    def bind_resolved_seqused_source(
        self,
        seqused_k_i32: torch.Tensor,
        *,
        source_kind: str | None = None,
    ) -> None:
        if not isinstance(seqused_k_i32, torch.Tensor):
            raise TypeError("seqused_k_i32 must be a torch.Tensor")
        if seqused_k_i32.dtype != torch.int32:
            raise ValueError("seqused_k_i32 must have dtype torch.int32")
        if seqused_k_i32.device != self.seqused_k_i32.device:
            raise ValueError("seqused_k_i32 device must match arena device")
        if seqused_k_i32.dim() != 1:
            raise ValueError("seqused_k_i32 must be rank 1")
        seqused_count = int(seqused_k_i32.numel())
        if seqused_count not in (int(self.batch_size), int(self.total_rows)):
            raise ValueError(
                "seqused_k_i32 must provide one entry per batch row or one "
                "entry per batch row/head; "
                f"got={seqused_count} batch_size={self.batch_size} "
                f"total_rows={self.total_rows}"
            )
        if not seqused_k_i32.is_contiguous():
            raise ValueError("seqused_k_i32 must be contiguous")
        self._carriers.resolver_visible_seqused_k_by_head_i32 = seqused_k_i32
        if source_kind is not None:
            self._resolved_seqused_source_kind = str(source_kind)
        elif int(seqused_k_i32.data_ptr()) == int(self.seqused_k_i32.data_ptr()):
            self._resolved_seqused_source_kind = "arena_seqused"
        elif int(seqused_k_i32.data_ptr()) == int(self.batch_seqused_k_i32.data_ptr()):
            self._resolved_seqused_source_kind = "arena_batch_seqused"
        else:
            self._resolved_seqused_source_kind = "external"

    def _bind_own_row_table_pointers(self) -> None:
        if self._carrier_points_to_own_row_table:
            return
        row_stride_bytes = int(self.row_table_i32.stride(0)) * int(
            self.row_table_i32.element_size()
        )
        self.carrier_u64.copy_(self._row_head_index_i64, non_blocking=True)
        self.carrier_u64.mul_(row_stride_bytes)
        self.carrier_u64.add_(int(self.row_table_i32.data_ptr()))
        self._carrier_points_to_own_row_table = True

    def _fallback_direct_affine_to_row_ptr(self) -> None:
        # [ARENA-AFFINE-DEVICE-RETIRED 2026-07-03] Used to reset the retired
        # affine_i32 / affine_row_consume_mode_i32 device tensors to the
        # row-ptr fallback sentinel; only the host booleans remain. The
        # ``_clear_direct_affine`` / ``_refresh_direct_affine_from_row_table``
        # device-scan helpers (zero production callers) were removed outright.
        self._direct_affine_ready = True
        self._direct_affine_row_mode_required = False

    def write_row(
        self,
        *,
        batch: int,
        head: int,
        pages: torch.Tensor,
        coverage_count: int,
        compact_ready: bool,
        row_mode: str,
    ) -> None:
        batch_i = _as_int(batch, label="batch")
        head_i = _as_int(head, label="head")
        if batch_i < 0 or batch_i >= self.batch_size:
            raise IndexError("batch index out of range")
        if head_i < 0 or head_i >= self.num_kv_heads:
            raise IndexError("head index out of range")
        self.write_flat_row(
            row_head_index=batch_i * self.num_kv_heads + head_i,
            pages=pages,
            coverage_count=coverage_count,
            compact_ready=compact_ready,
            row_mode=row_mode,
        )

    def write_flat_row(
        self,
        *,
        row_head_index: int,
        pages: torch.Tensor,
        coverage_count: int,
        compact_ready: bool,
        row_mode: str,
    ) -> None:
        flat_row_i = _validate_flat_row(row_head_index, total_rows=self.total_rows)
        if row_mode not in _WRITABLE_ROW_MODES:
            raise ValueError(f"unsupported row mode for write: {row_mode!r}")
        if row_mode == "safe_full_recent" and compact_ready is True:
            raise RuntimeError("compact-designated row entered safe full-recent mode")
        if not isinstance(pages, torch.Tensor):
            raise TypeError("pages must be a torch.Tensor")
        if pages.dtype != torch.int32:
            raise ValueError("pages must have dtype torch.int32")
        if pages.device != self.row_table_i32.device:
            raise ValueError("pages device must match row_table_i32 device")
        if tuple(pages.shape) != (self.max_pages_per_row,):
            raise ValueError("pages must have shape (max_pages_per_row,)")

        coverage_count_i = _as_int(coverage_count, label="coverage_count")
        if coverage_count_i < 0 or coverage_count_i > self.max_pages_per_row:
            raise ValueError("coverage_count must be in 0..max_pages_per_row")

        self.row_table_i32[flat_row_i].copy_(pages, non_blocking=True)
        # [ARENA-AFFINE-DEVICE-RETIRED 2026-07-03] per-flat-row affine_i32
        # fallback-sentinel device write removed.
        self._direct_affine_ready = bool(self._direct_affine_binding_active)
        self._direct_affine_row_mode_required = False
        self.coverage_count_by_row_head = _replace_tuple_value(
            self.coverage_count_by_row_head,
            flat_row_i,
            coverage_count_i,
        )
        self.compact_ready_by_row_head = _replace_tuple_value(
            self.compact_ready_by_row_head,
            flat_row_i,
            bool(compact_ready),
        )
        self.row_mode_by_row_head = _replace_tuple_value(
            self.row_mode_by_row_head,
            flat_row_i,
            row_mode,
        )
        # B1 stale-baseline guard: write_flat_row rewrote one flat
        # row-head directly. B1 keys its baseline by BATCH row, so drop
        # the baseline for the batch that owns this flat row. No-op when
        # B1 is off.
        _b1_last_pages_by_row = getattr(self, "_b1_last_pages_by_row", None)
        if _b1_last_pages_by_row is not None:
            _b1_last_pages_by_row.pop(flat_row_i // self.num_kv_heads, None)
        # [ARENA-AFFINE-DEVICE-RETIRED 2026-07-03] v6 affine-clean cache pop
        # removed with the cache itself.
        self.generation += 1

    def publish_descriptor_rows(
        self,
        *,
        snapshot: object,
        dirty_rows: Iterable[int] | None = None,
    ) -> bool:
        if not bool(getattr(snapshot, "valid", False)):
            return False
        try:
            snapshot_epoch = int(getattr(snapshot, "epoch"))
            rows_by_graph_row = tuple(getattr(snapshot, "rows_by_graph_row"))
        except (AttributeError, TypeError, ValueError):
            return False
        if snapshot_epoch < 0:
            return False
        if dirty_rows is None:
            target_rows = tuple(
                row for row, descriptor in enumerate(rows_by_graph_row)
                if descriptor is not None
            )
        else:
            try:
                target_rows = tuple(int(row) for row in dirty_rows)
            except (TypeError, ValueError):
                return False
        if not target_rows:
            return False
        if len(set(target_rows)) != len(target_rows):
            return False
        if any(row < 0 or row >= self.batch_size for row in target_rows):
            return False
        if len(rows_by_graph_row) < self.batch_size:
            return False

        self._bind_own_row_table_pointers()
        # The per-row seqused fills below are DEAD writes when the bound visible seqused
        # carrier is owned externally (e.g. the sparse_dynamic_state native lifecycle):
        # the FA4 kernel reads carriers.resolver_visible_seqused_k_by_head_i32
        # (compact_mixed_page_route.py resolver_kwargs), NOT these arena tensors. Skip
        # them there; keep them where batch_seqused_k_i32 IS the bound visible carrier
        # (arena_batch_seqused, i.e. the visible carrier aliases it). Computed once.
        _visible_carrier = getattr(
            getattr(self, "carriers", None),
            "resolver_visible_seqused_k_by_head_i32",
            None,
        )
        _seqused_fill_is_live = not (
            isinstance(_visible_carrier, torch.Tensor)
            and isinstance(self.seqused_k_i32, torch.Tensor)
            and isinstance(self.batch_seqused_k_i32, torch.Tensor)
            and int(_visible_carrier.data_ptr()) != int(self.seqused_k_i32.data_ptr())
            and int(_visible_carrier.data_ptr()) != int(self.batch_seqused_k_i32.data_ptr())
        )
        coverage = list(self.coverage_count_by_row_head)
        compact_ready = list(self.compact_ready_by_row_head)
        row_modes = list(self.row_mode_by_row_head)
        for batch in target_rows:
            descriptor = rows_by_graph_row[batch]
            if descriptor is None:
                return False
            if not self._descriptor_row_epoch_current(
                descriptor,
                batch=batch,
                snapshot_epoch=snapshot_epoch,
            ):
                return False
            try:
                row_effective = int(getattr(descriptor, "row_effective"))
                visible_k = int(getattr(descriptor, "visible_k"))
                row_mode_value = str(getattr(descriptor, "row_mode"))
                pages = tuple(int(v) for v in getattr(descriptor, "row_table_pages"))
                segment_pages = int(getattr(descriptor, "segment_pages"))
                affine_descriptor = getattr(descriptor, "affine_descriptor")
            except (AttributeError, TypeError, ValueError):
                return False
            if row_effective < 0 or visible_k != row_effective:
                return False
            page_count = len(pages)
            if page_count < 0 or page_count > self.max_pages_per_row:
                return False

            row_start = int(batch) * self.num_kv_heads
            row_stop = row_start + self.num_kv_heads
            row_slice = slice(row_start, row_stop)
            if _seqused_fill_is_live:
                self.batch_seqused_k_i32[batch].fill_(row_effective)
                self.seqused_k_i32[row_slice].fill_(row_effective)
            # --- CUT B1: incremental per-column page write -------------------
            # VLLM_SPARSE_B1_INCREMENTAL_PAGES: only re-upload the columns whose
            # page id actually changed since the last publish for this row. The
            # untouched columns already hold the prior (== current) page id, so
            # the resulting row_table_i32[row_slice] is byte-identical to the
            # full fill_(-1)+upload path. row_table_i32 stays graph-stable
            # (same tensor, in-place writes only).
            _b1_on = __import__("os").environ.get(
                "VLLM_SPARSE_B1_INCREMENTAL_PAGES", "1"
            ) == "1"
            _b1_assert = __import__("os").environ.get(
                "VLLM_SPARSE_B1_ASSERT"
            ) == "1"
            _b1_last_pages_by_row = getattr(self, "_b1_last_pages_by_row", None)
            if _b1_on and _b1_last_pages_by_row is None:
                _b1_last_pages_by_row = {}
                self._b1_last_pages_by_row = _b1_last_pages_by_row
            _b1_prev_pages = (
                _b1_last_pages_by_row.get(batch)
                if (_b1_on and _b1_last_pages_by_row is not None)
                else None
            )
            _b1_changed_cols = None
            _b1_incremental_eligible = (
                _b1_on
                and _b1_prev_pages is not None
                and page_count > 0
                and len(_b1_prev_pages) == page_count
            )
            if _b1_incremental_eligible:
                _b1_changed_cols = [
                    _i
                    for _i, (_a, _b) in enumerate(zip(_b1_prev_pages, pages))
                    if _a != _b
                ]
            if _b1_incremental_eligible:
                # ARENA-equality gate: snapshot BEFORE the incremental write so we
                # can recompute the FULL path into a scratch clone and compare.
                if _b1_assert:
                    _b1_before = self.row_table_i32[row_slice].clone()
                # Incremental write: ONLY the changed columns, no fill_(-1).
                if _b1_changed_cols:
                    _b1_col_idx = torch.as_tensor(
                        _b1_changed_cols,
                        dtype=torch.long,
                        device=self.row_table_i32.device,
                    )
                    _b1_vals = self.row_table_i32.new_tensor(
                        [pages[_c] for _c in _b1_changed_cols]
                    )
                    _b1_ncols = _b1_col_idx.numel()
                    self.row_table_i32[row_slice].index_copy_(
                        1,
                        _b1_col_idx,
                        _b1_vals.reshape(1, _b1_ncols).expand(
                            self.num_kv_heads,
                            _b1_ncols,
                        ),
                    )
                if _b1_assert:
                    _b1_incr = self.row_table_i32[row_slice].clone()
                    _b1_full = _b1_before.clone()
                    _b1_full.fill_(-1)
                    if page_count:
                        _b1_full_pages = _b1_full.new_tensor(pages)
                        _b1_full[:, :page_count].copy_(
                            _b1_full_pages.reshape(1, page_count).expand(
                                self.num_kv_heads,
                                page_count,
                            )
                        )
                    if not torch.equal(_b1_incr, _b1_full):
                        _b1_neq = (_b1_incr != _b1_full).any(dim=0)
                        _b1_first = int(_b1_neq.nonzero()[0].item())
                        raise AssertionError(
                            "VLLM_SPARSE_B1 arena mismatch row="
                            + str(int(batch))
                            + " first_diff_col="
                            + str(_b1_first)
                            + " incr="
                            + str(_b1_incr[:, _b1_first].tolist())
                            + " full="
                            + str(_b1_full[:, _b1_first].tolist())
                        )
            else:
                # EXACT existing full path (verbatim): fill_(-1) + full upload.
                # [WAR-SAFE-FILL-RETIRED 2026-07-07] 尾列填 pages[0] 的兜底下线:
                # 主凶(condense 行迁移×票快照过期)已终局收案,序保证由 WAR
                # fence+chunk_done 门承担;越界读回 -1 fail-fast,不静默读合法页。
                self.row_table_i32[row_slice].fill_(-1)
                if page_count:
                    page_tensor = self.row_table_i32.new_tensor(pages)
                    self.row_table_i32[row_slice, :page_count].copy_(
                        page_tensor.reshape(1, page_count).expand(
                            self.num_kv_heads,
                            page_count,
                        ),
                        non_blocking=True,
                    )
            if _b1_on and _b1_last_pages_by_row is not None:
                _b1_last_pages_by_row[batch] = pages
            # --- end CUT B1 -------------------------------------------------

            # [ARENA-AFFINE-DEVICE-RETIRED 2026-07-03] The per-row affine_i32 /
            # affine_row_consume_mode_i32 device writes and the #12 v6
            # affine-clean write-skip cache that gated them were removed. The
            # host-side descriptor-affine validity gate below is KEPT: a
            # non-fallback descriptor whose affine tuple does not rebuild its
            # pages still fails the publish closed.
            uses_fallback = (
                affine_descriptor == "row_table_fallback"
                or int(segment_pages) < 0
            )
            if not uses_fallback:
                try:
                    affine_tuple = tuple(int(v) for v in affine_descriptor)
                except (TypeError, ValueError):
                    return False
                if len(affine_tuple) != 5:
                    return False
                if not self._descriptor_affine_matches_pages(
                    affine=affine_tuple,
                    pages=pages,
                ):
                    return False
            compact_row = row_mode_value.lower() == "compact" or row_mode_value == "1"
            arena_row_mode = "compact" if compact_row else "native"
            for flat_row in range(row_start, row_stop):
                coverage[flat_row] = page_count
                compact_ready[flat_row] = compact_row
                row_modes[flat_row] = arena_row_mode

        self.coverage_count_by_row_head = tuple(int(v) for v in coverage)
        self.compact_ready_by_row_head = tuple(bool(v) for v in compact_ready)
        self.row_mode_by_row_head = tuple(str(v) for v in row_modes)
        self._direct_affine_ready = True
        self._direct_affine_row_mode_required = False
        self.generation += 1
        return True

    def publish_live_rows(
        self,
        *,
        pages_by_row: dict[int, tuple[int, ...]],
        dirty_rows: Iterable[int],
        compact_pages_by_row: dict[int, tuple[int, ...]] | None = None,
    ) -> None:
        """Cheap in-place live-table page writer for recent-window boundary rows.

        Writes ``row_table_i32`` for the given ``dirty_rows`` only, fanned
        across all kv heads. All writes are in place (graph-stable):
        ``row_table_i32`` is never reassigned.

        ``dirty_rows`` are BATCH row indices (0..batch_size-1), not flat rows.

        [ARENA-AFFINE-DEVICE-RETIRED 2026-07-03] ``compact_pages_by_row`` only
        fed the removed B2 affine-publish device writes into affine_i32 /
        affine_row_consume_mode_i32; it is accepted-and-ignored so existing
        callers keep working.
        """
        del compact_pages_by_row  # [ARENA-AFFINE-DEVICE-RETIRED 2026-07-03]
        # [LIVE-PAGES-PINNED-ASYNC 2026-07-08] the old per-row
        # ``new_tensor(list(pages))`` built a CUDA tensor from a Python list =
        # a SYNCHRONOUS pageable H2D that stalls the host until the main
        # stream drains — on TP2 async scheduling that is the previous step's
        # in-flight forward (~8ms), and page_add ran it every step: the
        # z_rrp_call 7.9ms/step plateau (96% of the TP2 steady metadata
        # slice). Replace with a pinned staging RING (host memcpy is instant,
        # H2D is truly async) + one persistent GPU staging tensor. Same-stream
        # ordering makes the GPU staging reuse safe (next step's H2D is queued
        # after this step's row_table copies); the pinned side needs the ring
        # because host rewrites are NOT stream-ordered (SLOT-STAGING-UAF-FIX
        # precedent). Values/write-set are byte-identical to the old path.
        dirty = [int(r) for r in dirty_rows]
        if not dirty:
            return
        max_pages = int(self.max_pages_per_row)
        ring = getattr(self, "_live_pages_pinned_ring", None)
        if (
            ring is None
            or ring[0].shape[0] < len(dirty)
            or ring[0].shape[1] != max_pages
        ):
            if ring is not None:
                # Cold-path regeneration guard (batch growth only): the old
                # pinned buffers may still feed an in-flight H2D; dropping the
                # refs hands them to GC mid-copy (SLOT-STAGING-UAF family).
                torch.cuda.current_stream(self.row_table_i32.device).synchronize()
            depth = 4
            rows_cap = max(len(dirty), int(self.batch_size))
            ring = [
                torch.full(
                    (rows_cap, max_pages), -1, dtype=torch.int32, pin_memory=True
                )
                for _ in range(depth)
            ]
            self._live_pages_pinned_ring = ring
            self._live_pages_pinned_ring_idx = 0
            self._live_pages_staging_gpu = torch.full(
                (rows_cap, max_pages),
                -1,
                dtype=torch.int32,
                device=self.row_table_i32.device,
            )
        idx = int(getattr(self, "_live_pages_pinned_ring_idx", 0))
        pinned = ring[idx]
        self._live_pages_pinned_ring_idx = (idx + 1) % len(ring)
        staging = self._live_pages_staging_gpu
        counts: list[int] = []
        for i, batch_row in enumerate(dirty):
            pages = pages_by_row.get(batch_row, ())
            page_count = len(pages)
            if page_count > max_pages:
                raise ValueError(
                    f"pages length {page_count} exceeds max_pages_per_row "
                    f"{max_pages} for batch_row {batch_row}"
                )
            row = pinned[i]
            row[:page_count] = torch.as_tensor(
                pages, dtype=torch.int32
            )  # host->pinned memcpy (no device traffic)
            if page_count < max_pages:
                row[page_count:] = -1
            counts.append(page_count)
        n = len(dirty)
        staging[:n].copy_(pinned[:n], non_blocking=True)
        for i, batch_row in enumerate(dirty):
            page_count = counts[i]
            row_start = batch_row * self.num_kv_heads
            row_slice = slice(row_start, row_start + self.num_kv_heads)
            # [WAR-SAFE-FILL-RETIRED 2026-07-07] 兜底下线,同上臂:恢复 fill_(-1)
            # fail-fast 语义。
            self.row_table_i32[row_slice].fill_(-1)
            if page_count:
                self.row_table_i32[row_slice, :page_count].copy_(
                    staging[i, :page_count]
                    .reshape(1, page_count)
                    .expand(self.num_kv_heads, page_count),
                    non_blocking=True,
                )
            # B1 stale-baseline guard: this path rewrote row_table_i32 for
            # batch_row outside B1's own writer, so the cached last-written
            # pages are no longer valid for it. Drop the per-row baseline so
            # the next B1 publish does a full (non-incremental) upload for
            # this row. No-op when B1 is off.
            _b1_last_pages_by_row = getattr(
                self, "_b1_last_pages_by_row", None
            )
            if _b1_last_pages_by_row is not None:
                _b1_last_pages_by_row.pop(batch_row, None)


    @staticmethod
    def _descriptor_row_epoch_current(
        descriptor: object,
        *,
        batch: int,
        snapshot_epoch: int,
    ) -> bool:
        try:
            return (
                int(getattr(descriptor, "epoch")) == int(snapshot_epoch)
                and int(getattr(descriptor, "visible_epoch")) == int(snapshot_epoch)
                and int(getattr(descriptor, "row_mode_epoch")) == int(snapshot_epoch)
                and int(getattr(descriptor, "graph_row_index")) == int(batch)
            )
        except (AttributeError, TypeError, ValueError):
            return False

    @staticmethod
    def _descriptor_affine_matches_pages(
        *,
        affine: tuple[int, int, int, int, int],
        pages: tuple[int, ...],
    ) -> bool:
        base, stride, segment_pages, second_base, second_stride = (
            int(v) for v in affine
        )
        if segment_pages < 0 or segment_pages > len(pages):
            return False
        rebuilt: list[int] = []
        for page_idx in range(len(pages)):
            if page_idx < segment_pages:
                rebuilt.append(base + stride * page_idx)
            else:
                rebuilt.append(
                    second_base + second_stride * (page_idx - segment_pages)
                )
        return tuple(rebuilt) == tuple(int(v) for v in pages)

    def bind_block_table_row_pointers(
        self,
        *,
        block_table: object,
        compact_ready_by_batch_row: Iterable[bool],
        row_effective_k_by_row: Iterable[int] = (),
        page_size: int = 1,
    ) -> None:
        block_table_i32 = validate_block_table_row_pointer_source(
            block_table=block_table,
            batch_size=self.batch_size,
            max_pages_per_row=self.max_pages_per_row,
            device=self.carrier_u64.device,
        )
        if isinstance(compact_ready_by_batch_row, torch.Tensor):
            raise TypeError("compact_ready_by_batch_row must be a CPU sequence")

        compact_ready_by_batch = tuple(
            bool(value) for value in compact_ready_by_batch_row
        )
        if len(compact_ready_by_batch) < self.batch_size:
            raise ValueError("compact_ready_by_batch_row must cover batch_size")
        row_effective_k = tuple(int(v) for v in row_effective_k_by_row)
        has_row_effective_k = bool(row_effective_k)
        if has_row_effective_k and len(row_effective_k) < self.batch_size:
            raise ValueError("row_effective_k_by_row must cover batch_size")
        page_size_i = _validate_positive(page_size, label="page_size")

        row_stride_bytes = int(block_table_i32.stride(0)) * int(
            block_table_i32.element_size()
        )
        self.carrier_u64.copy_(self._row_batch_index_i64, non_blocking=True)
        self.carrier_u64.mul_(row_stride_bytes)
        self.carrier_u64.add_(int(block_table_i32.data_ptr()))
        self._carrier_points_to_own_row_table = False

        coverage: list[int] = []
        compact_ready: list[bool] = []
        row_modes: list[str] = []
        row_sources = empty_row_source_distribution()
        for batch in range(self.batch_size):
            is_compact = compact_ready_by_batch[batch]
            mode = "compact" if is_compact else "native"
            if has_row_effective_k:
                visible_pages = min(
                    _ceil_div(max(0, int(row_effective_k[batch])), page_size_i),
                    self.max_pages_per_row,
                )
            else:
                visible_pages = self.max_pages_per_row
            if is_compact:
                row_sources["compact_rows"] += self.num_kv_heads
            else:
                row_sources["native_rows"] += self.num_kv_heads
                row_sources["native_canonical_pages"] += (
                    visible_pages * self.num_kv_heads
                )
            for _head in range(self.num_kv_heads):
                coverage.append(visible_pages)
                compact_ready.append(is_compact)
                row_modes.append(mode)
        if has_row_effective_k:
            values = torch.tensor(
                row_effective_k[: self.batch_size],
                dtype=torch.int32,
                device=self.seqused_k_i32.device,
            )
            self.batch_seqused_k_i32.copy_(values, non_blocking=True)
            visible = self.carriers.resolver_visible_seqused_k_by_head_i32
            if (
                isinstance(visible, torch.Tensor)
                and int(visible.data_ptr()) == int(self.seqused_k_i32.data_ptr())
            ):
                target = self.seqused_k_i32.view(self.batch_size, self.num_kv_heads)
                target.copy_(
                    values.reshape(self.batch_size, 1).expand(
                        self.batch_size,
                        self.num_kv_heads,
                    ),
                    non_blocking=True,
                )
        self.coverage_count_by_row_head = tuple(coverage)
        self.compact_ready_by_row_head = tuple(compact_ready)
        self.row_mode_by_row_head = tuple(row_modes)
        self.row_source_distribution = row_sources
        self._fallback_direct_affine_to_row_ptr()
        # B1 stale-baseline guard: this heavy re-bind repointed the
        # carrier away from the own row table and rewrote every row, so
        # all cached B1 baselines are stale. Clear the whole baseline.
        # No-op when B1 is off.
        _b1_last_pages_by_row = getattr(self, "_b1_last_pages_by_row", None)
        if _b1_last_pages_by_row is not None:
            _b1_last_pages_by_row.clear()
        # [ARENA-AFFINE-DEVICE-RETIRED 2026-07-03] v6 affine-clean cache clear
        # removed with the cache itself.
        self.generation += 1

    def try_bind_production_direct_affine(
        self,
        *,
        canonical_block_table: object,
        compact_ready_by_batch_row: Iterable[bool],
        row_effective_k_by_row: Iterable[int],
        page_size: int,
        reserved_manager_block_ids: Iterable[int] = (),
        slot_by_row: Iterable[int] = (),
        compact_valid_tokens_by_row: Iterable[int] = (),
        compact_offset_tokens_by_row: Iterable[int] = (),
        recent_first_page_by_row: Iterable[int] = (),
        canonical_row_index_by_batch_row: Iterable[int] = (),
        compact_capacity_pages: int = 0,
        profile_phase_us: dict[str, float] | None = None,
        planned_layout: PlannedCompactRowLayout | None = None,
        reserved_manager_block_ids_cpu: Iterable[int] | None = None,
        canonical_block_table_cpu: object | None = None,
    ) -> bool:
        profile_start_ns = time.perf_counter_ns() if profile_phase_us is not None else 0

        def _mark_direct_affine_phase(name: str) -> None:
            nonlocal profile_start_ns
            if profile_phase_us is None:
                return
            now_ns = time.perf_counter_ns()
            profile_phase_us[name] = float(now_ns - profile_start_ns) / 1000.0
            profile_start_ns = now_ns

        canonical_row_source = _layout_tuple(
            canonical_row_index_by_batch_row,
            label="canonical_row_index_by_batch_row",
        )
        if canonical_row_source:
            if len(canonical_row_source) != self.batch_size:
                return False
            canonical_row_by_batch = tuple(int(row) for row in canonical_row_source)
        else:
            canonical_row_by_batch = tuple(range(self.batch_size))
        if any(row < 0 for row in canonical_row_by_batch):
            return False
        max_canonical_row = max(canonical_row_by_batch, default=0)
        canonical_i32 = validate_block_table_row_pointer_source(
            block_table=canonical_block_table,
            batch_size=max_canonical_row + 1,
            max_pages_per_row=1,
            device=self.row_table_i32.device,
        )
        layout = planned_layout
        if layout is None:
            layout = build_planned_compact_row_layout(
                batch_size=self.batch_size,
                compact_ready_by_batch_row=compact_ready_by_batch_row,
                slot_by_row=_layout_tuple(slot_by_row, label="slot_by_row")
                or range(self.batch_size),
                compact_valid_tokens_by_row=_layout_tuple(
                    compact_valid_tokens_by_row,
                    label="compact_valid_tokens_by_row",
                )
                or (0,) * self.batch_size,
                compact_offset_tokens_by_row=_layout_tuple(
                    compact_offset_tokens_by_row,
                    label="compact_offset_tokens_by_row",
                )
                or (0,) * self.batch_size,
                recent_first_page_by_row=_layout_tuple(
                    recent_first_page_by_row,
                    label="recent_first_page_by_row",
                )
                or (0,) * self.batch_size,
                recent_page_count_by_row=(0,) * self.batch_size,
                row_effective_k_by_row=row_effective_k_by_row,
            )
        elif (
            len(layout.compact_ready_by_batch_row) != self.batch_size
            or len(layout.slot_by_row) != self.batch_size
            or len(layout.compact_valid_tokens_by_row) != self.batch_size
            or len(layout.compact_offset_tokens_by_row) != self.batch_size
            or len(layout.recent_first_page_by_row) != self.batch_size
            or len(layout.recent_page_count_by_row) != self.batch_size
            or len(layout.row_effective_k_by_row) != self.batch_size
        ):
            return False
        if not all(bool(v) for v in layout.compact_ready_by_batch_row):
            return False
        page_size_i = _validate_positive(page_size, label="page_size")
        compact_capacity_i = int(compact_capacity_pages)
        if compact_capacity_i <= 0:
            return False

        if isinstance(reserved_manager_block_ids, torch.Tensor):
            if reserved_manager_block_ids.dtype != torch.int32:
                return False
            if reserved_manager_block_ids.dim() != 1:
                return False
            if reserved_manager_block_ids.device != self.row_table_i32.device:
                return False
            if int(reserved_manager_block_ids.stride(0)) != 1:
                return False
            reserved_tensor = reserved_manager_block_ids
        else:
            reserved = tuple(int(v) for v in reserved_manager_block_ids)
            if not reserved:
                return False
            reserved_tensor = self.row_table_i32.new_tensor(reserved)
        reserved_len = int(reserved_tensor.numel())
        if reserved_manager_block_ids_cpu is not None:
            reserved_cpu_values = tuple(int(v) for v in reserved_manager_block_ids_cpu)
        elif isinstance(reserved_manager_block_ids, torch.Tensor):
            if reserved_manager_block_ids.device.type == "cpu":
                reserved_cpu_values = tuple(int(v) for v in reserved_manager_block_ids.tolist())
            else:
                reserved_cpu_values = None
        else:
            reserved_cpu_values = reserved
        if reserved_cpu_values is not None and len(reserved_cpu_values) != reserved_len:
            return False
        canonical_cpu_source = (
            canonical_i32 if canonical_i32.device.type == "cpu" else canonical_block_table_cpu
        )
        _mark_direct_affine_phase("rrp_direct_affine_validate")

        # [ARENA-AFFINE-DEVICE-RETIRED 2026-07-03] The per-batch device scalar
        # gathering (first/stride/segment/second tensors) and the packed
        # affine_i32 / affine_row_consume_mode_i32 device writes were removed;
        # only the host-side affine proofs and eligibility gates remain.
        host_affine_rows_by_batch: list[tuple[int, int, int, int, int]] = []
        host_affine_row_modes_by_batch: list[int] = []
        coverage: list[int] = []
        row_sources = empty_row_source_distribution()

        for batch in range(self.batch_size):
            effective_k = max(0, int(layout.row_effective_k_by_row[batch]))
            compact_tokens = max(0, int(layout.compact_valid_tokens_by_row[batch]))
            if compact_tokens <= 0 or compact_tokens % page_size_i != 0:
                return False
            compact_offset_tokens_i = int(layout.compact_offset_tokens_by_row[batch])
            if compact_offset_tokens_i < 0 or compact_offset_tokens_i % page_size_i != 0:
                return False
            compact_pages = compact_tokens // page_size_i
            compact_offset_pages = compact_offset_tokens_i // page_size_i
            if compact_pages <= 0 or compact_pages > compact_capacity_i:
                return False
            if effective_k < compact_tokens:
                return False
            recent_visible_tokens = effective_k - compact_tokens
            recent_pages = _ceil_div(recent_visible_tokens, page_size_i) if recent_visible_tokens > 0 else 0
            visible_pages = compact_pages + recent_pages
            if visible_pages <= 0 or visible_pages > self.max_pages_per_row:
                return False
            slot = int(layout.slot_by_row[batch])
            # [DUAL-GEN-L2a] 同款半区选择(此路径为 bool 快验非 raise);
            # 关态 factor=1=历史判定逐位。
            _gen_count = compact_gen_count()
            _gen_stride_pages = reserved_len // _gen_count if _gen_count > 0 else 0
            _gen_of_offset = (
                1
                if (_gen_count > 1 and compact_offset_pages >= _gen_stride_pages)
                else 0
            )
            expected_slot_start = (
                _gen_of_offset * _gen_stride_pages + slot * compact_capacity_i
            )
            expected_slot_end = expected_slot_start + compact_capacity_i
            slot_start = compact_offset_pages
            slot_end = slot_start + compact_pages
            if (
                slot < 0
                or expected_slot_start < 0
                or expected_slot_end > reserved_len
                or slot_start < expected_slot_start
                or slot_end > expected_slot_end
                or slot_end > reserved_len
            ):
                return False
            recent_first = int(layout.recent_first_page_by_row[batch])
            canonical_row = int(canonical_row_by_batch[batch])
            canonical_cpu_row = _cpu_block_table_row_source(
                canonical_cpu_source,
                canonical_row,
            )
            if recent_pages > 0 and (
                recent_first < 0
                or recent_first + recent_pages > int(canonical_i32.shape[1])
            ):
                return False

            # [ARENA-AFFINE-DEVICE-RETIRED 2026-07-03] Host affine eligibility
            # gates kept verbatim; the device compact_src/recent_src scalar
            # reads that fed the removed affine_i32 pack are gone.
            if reserved_cpu_values is not None:
                compact_affine = _affine_base_stride_from_cpu_sequence(
                    reserved_cpu_values,
                    start=slot_start,
                    count=compact_pages,
                )
                if compact_affine is None:
                    return False
            elif compact_pages > 2:
                return False
            if recent_pages > 0:
                if canonical_cpu_row is not None:
                    recent_affine = _affine_base_stride_from_cpu_sequence(
                        canonical_cpu_row,
                        start=recent_first,
                        count=recent_pages,
                    )
                    if recent_affine is None:
                        return False
                elif recent_pages > 2:
                    return False

            # Host-only affine 5-tuple mirror, so the whole-arena scalar
            # collapse can be proven without reading back any CUDA tensor
            # (compact_affine / recent_affine are host ints when the cpu
            # sources are present; absent => no scalar proof).
            if reserved_cpu_values is not None and compact_affine is not None:
                if recent_pages > 0:
                    _host_recent_affine = recent_affine if canonical_cpu_row is not None else None
                else:
                    _host_recent_affine = (
                        compact_affine[0] + compact_affine[1] * (compact_pages - 1),
                        1,
                    )
                if _host_recent_affine is not None:
                    host_affine_rows_by_batch.append(
                        (
                            int(compact_affine[0]),
                            int(compact_affine[1]),
                            int(compact_pages),
                            int(_host_recent_affine[0]),
                            int(_host_recent_affine[1]),
                        )
                    )
                    host_affine_row_modes_by_batch.append(ROW_CONSUME_MODE_SELECTED_I32)
            coverage.append(visible_pages)
            row_sources["compact_rows"] += self.num_kv_heads
            row_sources["compact_reserved_pages"] += compact_pages * self.num_kv_heads
            row_sources["recent_canonical_pages"] += recent_pages * self.num_kv_heads
            row_sources["compact_rows_with_reserved_pages"] += self.num_kv_heads
            if recent_pages > 0:
                row_sources["compact_rows_with_recent_pages"] += self.num_kv_heads
        _mark_direct_affine_phase("rrp_direct_affine_gather")

        # [ARENA-AFFINE-DEVICE-RETIRED 2026-07-03] The whole-arena affine_i32
        # pack + copy and the affine_row_consume_mode_i32 fill (plus the v6
        # cache clear) were removed. The phase marks are kept so profiling
        # consumers see the same key set (both now measure ~0).
        _mark_direct_affine_phase("rrp_direct_affine_pack")
        _mark_direct_affine_phase("rrp_direct_affine_carrier_copy")

        row_effective = self.batch_seqused_k_i32.new_tensor(
            [int(v) for v in layout.row_effective_k_by_row]
        )
        self.batch_seqused_k_i32.copy_(row_effective, non_blocking=True)
        self.seqused_k_i32.view(self.batch_size, self.num_kv_heads).copy_(
            row_effective.reshape(self.batch_size, 1).expand(self.batch_size, self.num_kv_heads),
            non_blocking=True,
        )
        _mark_direct_affine_phase("rrp_direct_affine_seqused_copy")
        self.coverage_count_by_row_head = tuple(
            int(v) for v in coverage for _ in range(self.num_kv_heads)
        )
        self.compact_ready_by_row_head = (True,) * self.total_rows
        self.row_mode_by_row_head = ("compact",) * self.total_rows
        self.row_source_distribution = row_sources
        # Refresh the whole-arena scalar batch-affine collapse from the host-only
        # rows assembled above (never a CUDA read-back). Without this, a direct
        # bind would leave _direct_affine_scalar_* stale/None while marking the
        # arena ready, so the trimmed const-direct rebind could assert-compare
        # against a scalar this bind never published.
        #
        # IMPORTANT: a None proof is NOT a failure. _prove_scalar_batch_affine
        # returns None whenever the whole-arena layout does not collapse to ONE
        # scalar -- which is the COMMON multi-request case (batch_size>=2 with
        # heterogeneous per-row stride/base) -- and the clean bind legitimately
        # returns True there with the per-batch AFFINE_TENSOR carrier (no scalar
        # needed). It is also None when the proof is simply inapplicable
        # (batch_size<2, or a row lacked a cpu source so host_affine_rows is
        # short). In EVERY None case we keep the three scalar fields cleared (so
        # ``direct_affine_scalar`` reports None and the trimmed rebind + the
        # const-direct binder both fail closed naturally) and FALL THROUGH to the
        # legacy ready path -- never ``return False``, which would force the
        # caller onto the heavy bind_production_row_table this fast path exists to
        # avoid. Only a genuine scalar proof publishes the three fields.
        self._direct_affine_scalar_base = None
        self._direct_affine_scalar_stride = None
        self._direct_affine_scalar_batch_stride = None
        if (
            self.batch_size >= 1  # fa3_sm90_bs1_affine: was >=2
            and len(host_affine_rows_by_batch) == self.batch_size
            and len(host_affine_row_modes_by_batch) == self.batch_size
        ):
            _direct_scalar_affine = _prove_scalar_batch_affine(
                host_affine_rows_by_batch,
                host_affine_row_modes_by_batch,
                batch_size=self.batch_size,
            )
            if _direct_scalar_affine is not None:
                self._direct_affine_scalar_base = int(_direct_scalar_affine[0])
                self._direct_affine_scalar_stride = int(_direct_scalar_affine[1])
                self._direct_affine_scalar_batch_stride = (  # fa3_sm90_perbatch_base
                    int(_direct_scalar_affine[2]) if _direct_scalar_affine[2] is not None else None)
                self._direct_affine_batch_bases = (
                    tuple(int(x) for x in _direct_scalar_affine[3])
                    if len(_direct_scalar_affine) > 3 and _direct_scalar_affine[3] is not None else None)
                if self._direct_affine_batch_bases is not None:
                    self.affine_batch_base_i32.copy_(
                        self.affine_batch_base_i32.new_tensor(self._direct_affine_batch_bases),
                        non_blocking=True)
        self._direct_affine_ready = True
        self._direct_affine_row_mode_required = False
        self.generation += 1
        _mark_direct_affine_phase("rrp_direct_affine_state_publish")
        return True

    def bind_production_row_table(
        self,
        *,
        canonical_block_table: object,
        compact_ready_by_batch_row: Iterable[bool],
        row_effective_k_by_row: Iterable[int],
        page_size: int,
        reserved_manager_block_ids: Iterable[int] = (),
        slot_by_row: Iterable[int] = (),
        compact_valid_tokens_by_row: Iterable[int] = (),
        compact_offset_tokens_by_row: Iterable[int] = (),
        recent_first_page_by_row: Iterable[int] = (),
        canonical_row_index_by_batch_row: Iterable[int] = (),
        safe_page_id: int = 0,
        compact_capacity_pages: int = 0,
        planned_layout: PlannedCompactRowLayout | None = None,
        profile_phase_us: dict[str, float] | None = None,
        reserved_manager_block_ids_cpu: Iterable[int] | None = None,
        canonical_block_table_cpu: object | None = None,
    ) -> None:
        profile_start_ns = time.perf_counter_ns() if profile_phase_us is not None else 0

        def _mark_row_table_phase(name: str) -> None:
            nonlocal profile_start_ns
            if profile_phase_us is None:
                return
            now_ns = time.perf_counter_ns()
            profile_phase_us[name] = float(now_ns - profile_start_ns) / 1000.0
            profile_start_ns = now_ns

        canonical_row_source = _layout_tuple(
            canonical_row_index_by_batch_row,
            label="canonical_row_index_by_batch_row",
        )
        if canonical_row_source:
            if len(canonical_row_source) != self.batch_size:
                raise ValueError(
                    "canonical_row_index_by_batch_row length must match batch_size"
                )
            canonical_row_by_batch = tuple(int(row) for row in canonical_row_source)
        else:
            canonical_row_by_batch = tuple(range(self.batch_size))
        max_canonical_row = max(
            (row for row in canonical_row_by_batch if row >= 0),
            default=0,
        )
        canonical_i32 = validate_block_table_row_pointer_source(
            block_table=canonical_block_table,
            batch_size=max_canonical_row + 1,
            max_pages_per_row=1,
            device=self.row_table_i32.device,
        )
        if int(canonical_i32.shape[1]) > self.max_pages_per_row:
            raise ValueError(
                "canonical_block_table width exceeds arena max_pages_per_row"
            )
        _mark_row_table_phase("rrp_row_table_validate")

        layout = planned_layout
        if layout is None:
            slot_source = _layout_tuple(slot_by_row, label="slot_by_row")
            compact_valid_source = _layout_tuple(
                compact_valid_tokens_by_row,
                label="compact_valid_tokens_by_row",
            )
            compact_offset_source = _layout_tuple(
                compact_offset_tokens_by_row,
                label="compact_offset_tokens_by_row",
            )
            recent_first_source = _layout_tuple(
                recent_first_page_by_row,
                label="recent_first_page_by_row",
            )
            layout = build_planned_compact_row_layout(
                batch_size=self.batch_size,
                compact_ready_by_batch_row=compact_ready_by_batch_row,
                slot_by_row=slot_source if slot_source else range(self.batch_size),
                compact_valid_tokens_by_row=compact_valid_source
                if compact_valid_source
                else (0,) * self.batch_size,
                compact_offset_tokens_by_row=compact_offset_source
                if compact_offset_source
                else (0,) * self.batch_size,
                recent_first_page_by_row=recent_first_source
                if recent_first_source
                else (0,) * self.batch_size,
                recent_page_count_by_row=(0,) * self.batch_size,
                row_effective_k_by_row=row_effective_k_by_row,
            )
        else:
            if len(layout.compact_ready_by_batch_row) != self.batch_size:
                raise ValueError("planned_layout compact_ready_by_batch_row length must match batch_size")
            if len(layout.slot_by_row) != self.batch_size:
                raise ValueError("planned_layout slot_by_row length must match batch_size")
            if len(layout.compact_valid_tokens_by_row) != self.batch_size:
                raise ValueError("planned_layout compact_valid_tokens_by_row length must match batch_size")
            if len(layout.compact_offset_tokens_by_row) != self.batch_size:
                raise ValueError("planned_layout compact_offset_tokens_by_row length must match batch_size")
            if len(layout.recent_first_page_by_row) != self.batch_size:
                raise ValueError("planned_layout recent_first_page_by_row length must match batch_size")
            if len(layout.recent_page_count_by_row) != self.batch_size:
                raise ValueError("planned_layout recent_page_count_by_row length must match batch_size")
            if len(layout.row_effective_k_by_row) != self.batch_size:
                raise ValueError("planned_layout row_effective_k_by_row length must match batch_size")

        compact_ready_by_batch = layout.compact_ready_by_batch_row
        batch_has_compact_row = any(bool(v) for v in compact_ready_by_batch)
        row_effective_k = layout.row_effective_k_by_row
        page_size_i = _validate_positive(page_size, label="page_size")
        safe_page_id_i = _as_int(safe_page_id, label="safe_page_id")
        if safe_page_id_i < 0:
            raise ValueError("safe_page_id must be non-negative")
        _mark_row_table_phase("rrp_row_table_layout")

        reserved_tensor: torch.Tensor | None = None
        if isinstance(reserved_manager_block_ids, torch.Tensor):
            if reserved_manager_block_ids.dtype != torch.int32:
                raise ValueError("reserved_manager_block_ids tensor must have dtype torch.int32")
            if reserved_manager_block_ids.dim() != 1:
                raise ValueError("reserved_manager_block_ids tensor must be rank 1")
            if reserved_manager_block_ids.device != self.row_table_i32.device:
                raise ValueError("reserved_manager_block_ids tensor device must match row_table_i32")
            if int(reserved_manager_block_ids.stride(0)) != 1:
                raise ValueError("reserved_manager_block_ids tensor must be contiguous")
            reserved_tensor = reserved_manager_block_ids
            reserved = ()
            reserved_len = int(reserved_tensor.numel())
        else:
            reserved = tuple(int(v) for v in reserved_manager_block_ids)
            reserved_len = len(reserved)
        if reserved_manager_block_ids_cpu is not None:
            reserved_cpu_values = tuple(int(v) for v in reserved_manager_block_ids_cpu)
            if len(reserved_cpu_values) != reserved_len:
                raise ValueError("reserved_manager_block_ids_cpu length mismatch")
        elif isinstance(reserved_manager_block_ids, torch.Tensor):
            if reserved_manager_block_ids.device.type == "cpu":
                reserved_cpu_values = tuple(int(v) for v in reserved_manager_block_ids.tolist())
            else:
                reserved_cpu_values = None
        else:
            reserved_cpu_values = reserved
        canonical_cpu_source = (
            canonical_i32 if canonical_i32.device.type == "cpu" else canonical_block_table_cpu
        )
        slot_by_batch = layout.slot_by_row
        compact_valid_tokens = layout.compact_valid_tokens_by_row
        compact_offset_tokens = layout.compact_offset_tokens_by_row
        recent_first_pages = layout.recent_first_page_by_row
        compact_capacity_i = int(compact_capacity_pages)
        _mark_row_table_phase("rrp_row_table_reserved")

        self._bind_own_row_table_pointers()
        _mark_row_table_phase("rrp_row_table_pointer_bind")
        # ---- env-gated C++ host-derivation FAST PATH (default OFF) --------
        # When enabled, derive ALL host integers/affine/row_sources in one C++
        # call (v2 BUFFER ABI: canonical_cpu passed as a TENSOR for the wrapper
        # to read via data_ptr), then reproduce the exact CUDA writes + state
        # publish from the returned per-batch write_descriptors. Default OFF =>
        # this block is skipped and the original Python loop below runs
        # byte-identically. ANY exception (ext build failure, the wrapper
        # refusing a width-mismatched / non-CPU canonical_cpu, a partial write)
        # falls through to the original loop, which re-runs the full derivation
        # + tensor writes from scratch. The pointer bind at :1969 already ran
        # once, so the helper is called bind_row_pointers=False and the fallback
        # loop relies on that single bind.
        if rrp_bind_host_ext.enabled():
            try:
                _bind_host_cpp_derived = rrp_bind_host_ext.bind_host_derive_cpp(
                    batch_size=self.batch_size,
                    num_kv_heads=self.num_kv_heads,
                    max_pages_per_row=self.max_pages_per_row,
                    page_size_i=page_size_i,
                    safe_page_id_i=safe_page_id_i,
                    compact_capacity_i=compact_capacity_i,
                    canonical_width=int(canonical_i32.shape[1]),
                    canonical_rows=max_canonical_row + 1,
                    row_effective_k_by_row=row_effective_k,
                    compact_ready_by_batch=compact_ready_by_batch,
                    slot_by_row=slot_by_batch,
                    compact_valid_tokens_by_row=compact_valid_tokens,
                    compact_offset_tokens_by_row=compact_offset_tokens,
                    recent_first_page_by_row=recent_first_pages,
                    canonical_row_by_batch=canonical_row_by_batch,
                    reserved_is_tensor=(reserved_tensor is not None),
                    reserved_len=reserved_len,
                    reserved_cpu_values=reserved_cpu_values,
                    canonical_cpu=canonical_cpu_source,
                )
                apply_arena_writes_from_descriptors(
                    self,
                    _bind_host_cpp_derived,
                    reserved=reserved,
                    reserved_tensor=reserved_tensor,
                    canonical_i32=canonical_i32,
                    safe_page_id=safe_page_id_i,
                    page_size=page_size_i,
                    bind_row_pointers=False,
                )
                _mark_row_table_phase("rrp_row_table_rows_copy")
                _mark_row_table_phase("rrp_row_table_affine_refresh")
                _mark_row_table_phase("rrp_row_table_state_publish")
                return
            except Exception as _cpp_exc:
                # Fall through to the unchanged original Python loop below.
                rrp_bind_host_ext.trace_cpp_fallthrough(_cpp_exc)
        # ---- end fast path -----------------------------------------------

        coverage: list[int] = []
        compact_ready: list[bool] = []
        row_modes: list[str] = []
        affine_rows_by_batch: list[tuple[int, int, int, int, int]] = []
        affine_row_modes_by_batch: list[int] = []

        def _append_affine_batch_row(
            row_mode: int,
            base: int = 0,
            stride: int = 1,
            segment_pages: int = _AFFINE_ROW_PTR_FALLBACK_SEGMENT_PAGES,
            second_base: int = 0,
            second_stride: int = 1,
        ) -> None:
            affine_row_modes_by_batch.append(int(row_mode))
            affine_rows_by_batch.append(
                (
                    int(base),
                    int(stride),
                    int(segment_pages),
                    int(second_base),
                    int(second_stride),
                )
            )

        row_sources = empty_row_source_distribution()
        # SEQUSED_VECTORIZE: hoist the per-batch seqused publish to TWO
        # vectorized copies (byte-exact; same value every branch). The
        # per-row local ``effective_k`` below is kept for branch geometry.
        _seqused_effective_k_seq = [
            max(0, int(v)) for v in row_effective_k[: self.batch_size]
        ]
        _seqused_re = self.batch_seqused_k_i32.new_tensor(
            _seqused_effective_k_seq
        )
        self.batch_seqused_k_i32.copy_(_seqused_re, non_blocking=True)
        self.seqused_k_i32.view(self.batch_size, self.num_kv_heads).copy_(
            _seqused_re.reshape(self.batch_size, 1).expand(
                self.batch_size, self.num_kv_heads
            ),
            non_blocking=True,
        )
        for batch in range(self.batch_size):
            effective_k = max(0, int(row_effective_k[batch]))
            is_compact = bool(compact_ready_by_batch[batch])
            mode = "compact" if is_compact else "native"
            canonical_row = int(canonical_row_by_batch[batch])
            if canonical_row < 0:
                if effective_k > page_size_i or is_compact:
                    raise ValueError(
                        "inactive canonical rows must be non-compact with at most one safe page"
                    )
                row_start = batch * self.num_kv_heads
                row_stop = row_start + self.num_kv_heads
                row_slice = slice(row_start, row_stop)
                self.row_table_i32[row_slice].fill_(safe_page_id_i)
                row_sources["inactive_rows"] = (
                    int(row_sources.get("inactive_rows", 0)) + self.num_kv_heads
                )
                safe_pages = 1 if effective_k > 0 else 0
                coverage.extend([safe_pages] * self.num_kv_heads)
                compact_ready.extend([False] * self.num_kv_heads)
                row_modes.extend(["unset"] * self.num_kv_heads)
                _append_affine_batch_row(ROW_CONSUME_MODE_FULL_I32)
                continue

            if is_compact:
                compact_tokens = max(0, int(compact_valid_tokens[batch]))
                if compact_tokens % page_size_i != 0:
                    raise ValueError("compact_valid_tokens must be page-aligned")
                compact_offset_tokens_i = int(compact_offset_tokens[batch])
                if compact_offset_tokens_i < 0:
                    raise ValueError("compact_offset_tokens must be non-negative")
                if compact_offset_tokens_i % page_size_i != 0:
                    raise ValueError("compact_offset_tokens must be page-aligned")
                compact_pages = compact_tokens // page_size_i
                compact_offset_pages = compact_offset_tokens_i // page_size_i
                if compact_pages > compact_capacity_i:
                    raise ValueError("compact page count exceeds compact_capacity_pages")
                if effective_k < compact_tokens:
                    raise ValueError("row_effective_k is smaller than compact tokens")
                recent_visible_tokens = effective_k - compact_tokens
                recent_visible_pages = (
                    _ceil_div(recent_visible_tokens, page_size_i)
                    if recent_visible_tokens > 0
                    else 0
                )
                visible_pages = compact_pages + recent_visible_pages
                if visible_pages > self.max_pages_per_row:
                    raise ValueError("visible compact/recent pages exceed row width")
                slot = int(slot_by_batch[batch])
                # [DUAL-GEN-L2a] 与首处校验同款:offset 落点选 gen 半区后做
                # slot 窗校验(关态 factor=1=历史判定逐位)。
                _gen_count = compact_gen_count()
                _gen_stride_pages = (
                    reserved_len // _gen_count if _gen_count > 0 else 0
                )
                _gen_of_offset = (
                    1
                    if (
                        _gen_count > 1
                        and compact_offset_pages >= _gen_stride_pages
                    )
                    else 0
                )
                expected_slot_start = (
                    _gen_of_offset * _gen_stride_pages + slot * compact_capacity_i
                )
                expected_slot_end = expected_slot_start + compact_capacity_i
                if (
                    slot < 0
                    or expected_slot_start < 0
                    or expected_slot_end > reserved_len
                ):
                    raise ValueError("slot exceeds reserved_manager_block_ids")
                slot_start = compact_offset_pages
                slot_end = slot_start + compact_pages
                if (
                    slot_start < expected_slot_start
                    or slot_end > expected_slot_end
                ):
                    raise ValueError(
                        "compact offset plus page count exceeds slot compact span"
                    )
                if slot_start < 0 or slot_end > reserved_len:
                    raise ValueError(
                        "compact offset plus page count exceeds reserved_manager_block_ids"
                    )
                recent_first = int(recent_first_pages[batch])
                if (
                    recent_first < 0
                    or recent_first + recent_visible_pages > int(canonical_i32.shape[1])
                ):
                    raise ValueError("recent page range exceeds canonical block table")
                row_sources["compact_rows"] += self.num_kv_heads
                row_sources["compact_reserved_pages"] += compact_pages * self.num_kv_heads
                row_sources["recent_canonical_pages"] += recent_visible_pages * self.num_kv_heads
                if compact_pages > 0:
                    row_sources["compact_rows_with_reserved_pages"] += self.num_kv_heads
                if recent_visible_pages > 0:
                    row_sources["compact_rows_with_recent_pages"] += self.num_kv_heads
                if compact_pages == 0 and recent_visible_pages > 0:
                    row_sources["compact_full_native_fallback_rows"] += self.num_kv_heads
                row_start = batch * self.num_kv_heads
                row_stop = row_start + self.num_kv_heads
                row_slice = slice(row_start, row_stop)
                if compact_pages:
                    if reserved_tensor is None:
                        compact_src = self.row_table_i32.new_tensor(
                            reserved[slot_start:slot_end]
                        )
                    else:
                        compact_src = reserved_tensor.narrow(0, slot_start, compact_pages)
                    self.row_table_i32[row_slice, :compact_pages].copy_(
                        compact_src.reshape(1, compact_pages).expand(
                            self.num_kv_heads,
                            compact_pages,
                        ),
                        non_blocking=True,
                    )
                if recent_visible_pages:
                    recent_dst = compact_pages
                    recent_src = canonical_i32[
                        canonical_row,
                        recent_first : recent_first + recent_visible_pages,
                    ]
                    self.row_table_i32[
                        row_slice,
                        recent_dst : recent_dst + recent_visible_pages,
                    ].copy_(
                        recent_src.reshape(1, recent_visible_pages).expand(
                            self.num_kv_heads,
                            recent_visible_pages,
                        ),
                        non_blocking=True,
                    )
                canonical_cpu_row = _cpu_block_table_row_source(
                    canonical_cpu_source,
                    canonical_row,
                )
                selected_affine: tuple[int, int, int, int, int] | None = None
                if compact_pages > 0 and reserved_cpu_values is not None:
                    compact_affine = _affine_base_stride_from_cpu_sequence(
                        reserved_cpu_values,
                        start=slot_start,
                        count=compact_pages,
                    )
                    if compact_affine is not None:
                        if recent_visible_pages > 0:
                            recent_affine = (
                                _affine_base_stride_from_cpu_sequence(
                                    canonical_cpu_row,
                                    start=recent_first,
                                    count=recent_visible_pages,
                                )
                                if canonical_cpu_row is not None
                                else None
                            )
                        else:
                            recent_affine = (
                                compact_affine[0] + compact_affine[1] * (compact_pages - 1),
                                1,
                            )
                        if recent_affine is not None:
                            selected_affine = (
                                compact_affine[0],
                                compact_affine[1],
                                compact_pages,
                                recent_affine[0],
                                recent_affine[1],
                            )
                coverage.extend([visible_pages] * self.num_kv_heads)
                compact_ready.extend([True] * self.num_kv_heads)
                row_modes.extend([mode] * self.num_kv_heads)
                first_segment_pages = compact_pages if compact_pages > 0 else visible_pages
                if selected_affine is None:
                    _append_affine_batch_row(ROW_CONSUME_MODE_FULL_I32)
                else:
                    _append_affine_batch_row(ROW_CONSUME_MODE_SELECTED_I32, *selected_affine)
                continue

            visible_pages = min(
                _ceil_div(effective_k, page_size_i) if effective_k > 0 else 0,
                int(canonical_i32.shape[1]),
                self.max_pages_per_row,
            )
            row_sources["native_rows"] += self.num_kv_heads
            row_sources["native_canonical_pages"] += visible_pages * self.num_kv_heads
            row_start = batch * self.num_kv_heads
            row_stop = row_start + self.num_kv_heads
            row_slice = slice(row_start, row_stop)
            if visible_pages:
                native_src = canonical_i32[canonical_row, :visible_pages]
                self.row_table_i32[row_slice, :visible_pages].copy_(
                    native_src.reshape(1, visible_pages).expand(
                        self.num_kv_heads,
                        visible_pages,
                    ),
                    non_blocking=True,
                )
            coverage.extend([visible_pages] * self.num_kv_heads)
            compact_ready.extend([False] * self.num_kv_heads)
            row_modes.extend([mode] * self.num_kv_heads)
            # PHASE2B regression fix (d2f82bc activated DIRECT affine): the FULL/native
            # row -- in MIXED batches too -- must carry its REAL physical canonical affine,
            # NOT the bare (base=0, seg=-1) sentinel. DIRECT reads base/stride as ABSOLUTE
            # physical pool pages; base=0 -> illegal pool block -> OOB. native_base/stride
            # makes the DIRECT page == the row's true canonical page (== validated INDIRECT);
            # gappy rows keep seg=-1 so the kernel's row-ptr fallback reads row_table_i32.
            canonical_cpu_row = _cpu_block_table_row_source(
                canonical_cpu_source,
                canonical_row,
            )
            native_affine = (
                _affine_base_stride_from_cpu_sequence(
                    canonical_cpu_row,
                    start=0,
                    count=visible_pages,
                )
                if canonical_cpu_row is not None and visible_pages > 0
                else None
            )
            if native_affine is None:
                _append_affine_batch_row(ROW_CONSUME_MODE_FULL_I32)
            else:
                native_base, native_stride = native_affine
                _append_affine_batch_row(
                    ROW_CONSUME_MODE_FULL_I32,
                    native_base,
                    native_stride,
                    visible_pages,
                    native_base + native_stride * max(visible_pages - 1, 0),
                    1,
                )

        _mark_row_table_phase("rrp_row_table_rows_copy")
        self.coverage_count_by_row_head = tuple(coverage)
        self.compact_ready_by_row_head = tuple(compact_ready)
        self.row_mode_by_row_head = tuple(row_modes)
        self.row_source_distribution = row_sources
        # Host-only whole-arena scalar-affine collapse proof (no CUDA read). When it
        # holds, the replay binder lowers the whole arena to AFFINE_CONST_DIRECT (free
        # direct-physical path) instead of the heavy AFFINE_TENSOR carrier.
        self._direct_affine_scalar_base = None
        self._direct_affine_scalar_stride = None
        self._direct_affine_scalar_batch_stride = None
        _scalar_affine = _prove_scalar_batch_affine(
            affine_rows_by_batch,
            affine_row_modes_by_batch,
            batch_size=self.batch_size,
        )
        if _scalar_affine is not None:
            self._direct_affine_scalar_base = int(_scalar_affine[0])
            self._direct_affine_scalar_stride = int(_scalar_affine[1])
            self._direct_affine_scalar_batch_stride = (  # fa3_sm90_perbatch_base
                int(_scalar_affine[2]) if _scalar_affine[2] is not None else None)
            self._direct_affine_batch_bases = (
                tuple(int(x) for x in _scalar_affine[3])
                if len(_scalar_affine) > 3 and _scalar_affine[3] is not None else None)
            if self._direct_affine_batch_bases is not None:
                self.affine_batch_base_i32.copy_(
                    self.affine_batch_base_i32.new_tensor(self._direct_affine_batch_bases),
                    non_blocking=True)
        # [ARENA-AFFINE-DEVICE-RETIRED 2026-07-03] The whole-arena affine_i32 /
        # affine_row_consume_mode_i32 device uploads were removed; the host
        # affine_rows_by_batch list still drives the scalar-collapse proof
        # above and the readiness boolean below.
        self._direct_affine_ready = bool(affine_rows_by_batch)
        self._direct_affine_row_mode_required = False
        _mark_row_table_phase("rrp_row_table_affine_refresh")
        # B1 stale-baseline guard: this heavy re-bind rewrote the whole
        # row_table_i32 from the canonical block table, so every cached
        # B1 baseline is stale. Clear the whole baseline. No-op when B1
        # is off.
        _b1_last_pages_by_row = getattr(self, "_b1_last_pages_by_row", None)
        if _b1_last_pages_by_row is not None:
            _b1_last_pages_by_row.clear()
        # [ARENA-AFFINE-DEVICE-RETIRED 2026-07-03] v6 affine-clean cache clear
        # removed with the cache itself.
        # PHASE2B REAL FIX: cap published seqused to written page coverage per row
        # (n_block_max can never exceed the materialized pages -> no -1 tail walk).
        try:
            if len(coverage) >= self.batch_size * self.num_kv_heads:
                _cap_seq = [
                    min(
                        max(0, int(row_effective_k[_cb])),
                        int(coverage[_cb * self.num_kv_heads]) * page_size_i,
                    )
                    for _cb in range(self.batch_size)
                ]
                _cap_t = self.batch_seqused_k_i32.new_tensor(_cap_seq)
                self.batch_seqused_k_i32.copy_(_cap_t, non_blocking=True)
                self.seqused_k_i32.view(self.batch_size, self.num_kv_heads).copy_(
                    _cap_t.reshape(self.batch_size, 1).expand(
                        self.batch_size, self.num_kv_heads
                    ),
                    non_blocking=True,
                )
        except Exception:
            pass
        self.generation += 1
        _mark_row_table_phase("rrp_row_table_state_publish")



def bind_resolved_row_ptr_replay_metadata(
    *,
    attn_metadata: object,
    replay_arena: ResolvedRowPtrArena,
    source_arena: ResolvedRowPtrArena,
    batch_size: int,
    num_kv_heads: int,
    page_block_size: int,
    max_pages_per_row: int,
    q_layout_key: str,
    max_seqlen_q: int = 0,
    max_seqlen_k: int = 0,
    capture_buffer_shape_key: str = "resolved_row_ptr:none",
    producer_stream_key: str = "resolved-row-ptr-arena",
) -> ResolvedRowPtrReplayMetadataBinding:
    batch_size_i = _validate_positive(batch_size, label="batch_size")
    num_kv_heads_i = _validate_positive(num_kv_heads, label="num_kv_heads")
    page_block_size_i = _validate_positive(page_block_size, label="page_block_size")
    max_pages_i = _validate_positive(max_pages_per_row, label="max_pages_per_row")
    for label, arena in (
        ("replay_arena", replay_arena),
        ("source_arena", source_arena),
    ):
        _validate_arena_contract(
            arena,
            label=label,
            batch_size=batch_size_i,
            num_kv_heads=num_kv_heads_i,
            max_pages_per_row=max_pages_i,
        )

    descriptor = ResolverGraphDescriptor(
        resolver_kind=PageResolverKind.RESOLVED_ROW_PTR,
        batch=batch_size_i,
        num_heads=num_kv_heads_i,
        page_block_size=page_block_size_i,
        max_pages_per_row=max_pages_i,
        max_selected_pages_per_row=max_pages_i,
        max_capture_rows=0,
        q_layout_key=str(q_layout_key),
        kv_cache_addr=0,
        capture_buffer_shape_key=str(capture_buffer_shape_key),
        producer_stream_key=str(producer_stream_key),
        max_seqlen_q_bucket=_bucket_max_seqlen_q(max_seqlen_q),
        max_seqlen_k_bucket=_bucket_max_seqlen_k(max_seqlen_k),
    )
    replay_visible = replay_arena.carriers.resolver_visible_seqused_k_by_head_i32
    if not isinstance(replay_visible, torch.Tensor):
        raise RuntimeError("ResolvedRowPtr binding requires visible seqused tensor")
    disable_direct_affine = _RRP_DISABLE_DIRECT_AFFINE_CACHED
    use_direct_affine = (
        not disable_direct_affine
        and replay_arena.direct_affine_ready
        and not replay_arena.direct_affine_row_mode_required
    )
    scalar_affine = replay_arena.direct_affine_scalar if use_direct_affine else None
    if scalar_affine is not None:
        # AFFINE_CONST_DIRECT: scalar base/stride/batch_stride only. row_ptr_u64 and
        # affine_i32 MUST be None (carrier exclusivity); head_stride 0 and segment/second
        # cleared so runtime_bridge has_affine_const_direct passes -> the free resolve.
        scalar_base, scalar_stride, scalar_batch_stride = scalar_affine
        replay_carriers = MixedPageResolverCarrierSet(
            row_consume_mode_i32=None,
            resolver_visible_seqused_k_by_head_i32=replay_visible,
            selected_page_table_i32=None,
            effective_row_slot_i32=None,
            compact_base_page_i32=None,
            compact_page_count_i32=None,
            recent_first_logical_page_i32=None,
            resolved_page_table_row_ptr_u64=None,
            resolved_page_table_affine_i32=None,
            resolved_page_table_affine_base=int(scalar_base),
            resolved_page_table_affine_stride=int(scalar_stride),
            resolved_page_table_affine_batch_stride=int(scalar_batch_stride),
            resolved_page_table_affine_segment_pages=None,
            resolved_page_table_affine_second_base=None,
            resolved_page_table_affine_second_stride=None,
            resolved_page_table_affine_head_stride=0,
            resolved_page_table_affine_direct=True,
        )
    elif use_direct_affine and replay_arena.direct_affine_batch_bases is not None:
        # fa3_sm90_perbatch_base (gap C): per-batch AFFINE_CONST_DIRECT carrier. The
        # per-batch absolute bases live in the affine_batch_base_i32 device tensor;
        # device reads base[resolver_bidb] (C2). affine_base unused (0), batch_stride
        # None, affine_direct True -> has_affine_const_direct passes via affine_direct.
        pb_stride, _pb_bases = replay_arena.direct_affine_batch_bases
        replay_carriers = MixedPageResolverCarrierSet(
            row_consume_mode_i32=None,
            resolver_visible_seqused_k_by_head_i32=replay_visible,
            selected_page_table_i32=None,
            effective_row_slot_i32=None,
            compact_base_page_i32=None,
            compact_page_count_i32=None,
            recent_first_logical_page_i32=None,
            resolved_page_table_row_ptr_u64=None,
            resolved_page_table_affine_i32=None,
            resolved_page_table_affine_base=0,
            resolved_page_table_affine_stride=int(pb_stride),
            resolved_page_table_affine_batch_stride=None,
            resolved_page_table_affine_segment_pages=None,
            resolved_page_table_affine_second_base=None,
            resolved_page_table_affine_second_stride=None,
            resolved_page_table_affine_head_stride=0,
            resolved_page_table_affine_direct=True,
            resolved_page_table_affine_per_batch_base_i32=replay_arena.affine_batch_base_i32,
        )
    else:
        replay_carriers = MixedPageResolverCarrierSet(
            row_consume_mode_i32=None,
            resolver_visible_seqused_k_by_head_i32=replay_visible,
            selected_page_table_i32=None,
            effective_row_slot_i32=None,
            compact_base_page_i32=None,
            compact_page_count_i32=None,
            recent_first_logical_page_i32=None,
            # [AFFINE-BS>1-FIX 2026-07-01] Route the non-scalar-collapse affine case
            # through the proven RowPtr compact carrier instead of AFFINE_TENSOR.
            # ROOT CAUSE: the SM80 AFFINE_TENSOR device resolve
            # (paged_kv.h AffineTensor2Plan: resolved_page = ptr_page_table_row[
            # affine_base + affine_stride*page_idx]) indexes a *base page table*, but
            # carrier exclusivity (Python validate_mixed_page_optional_blocks + the C++
            # "AffineTensor subkind does not accept rowptr" check) forbids the RowPtr
            # carrier, and the sparse compact path does not supply a valid base
            # ptr_page_table_row for AFFINE_TENSOR -> it read wrong/garbage pages at
            # bs>1 (empirically: gate green but output garbage vs affine-OFF). RowPtr is
            # bit-identical to parent AND is itself the compact carrier, so this keeps
            # both correctness and compaction; only the affine-arithmetic micro-opt is
            # dropped for the bs>1 non-collapsible case. affine-on bs=1/uniform still
            # takes the correct scalar AFFINE_CONST_DIRECT branch above.
            resolved_page_table_row_ptr_u64=replay_arena.carrier_u64,
            resolved_page_table_affine_i32=None,
            resolved_page_table_affine_batch_stride=None,
        )
    replay_arena._direct_affine_binding_active = bool(use_direct_affine)
    binding = ResolvedRowPtrReplayMetadataBinding(
        descriptor=descriptor,
        replay_arena=replay_arena,
        source_arena=source_arena,
        replay_carriers=replay_carriers,
        source_carriers=replay_carriers,
        pointer_signature=replay_carriers.pointer_signature(),
        row_mode_distribution=build_row_mode_distribution(
            source_arena.row_mode_by_row_head
        ),
        row_source_distribution=dict(source_arena.row_source_distribution),
        **build_source_counter_fields(
            batch_size=batch_size_i,
            num_kv_heads=num_kv_heads_i,
            row_source_distribution=source_arena.row_source_distribution,
        ),
    )
    replay_arena._last_debug_pointer_signature = binding.pointer_signature
    attach_resolved_row_ptr_replay_metadata(
        attn_metadata=attn_metadata,
        binding=binding,
    )
    return binding


def attach_resolved_row_ptr_replay_metadata(
    *,
    attn_metadata: object,
    binding: ResolvedRowPtrReplayMetadataBinding,
) -> None:
    descriptor = binding.descriptor
    replay_arena = binding.replay_arena
    source_arena = binding.source_arena
    replay_carriers = binding.replay_carriers
    source_carriers = binding.source_carriers
    replay_seqused_k_i32 = replay_carriers.resolver_visible_seqused_k_by_head_i32
    source_seqused_k_i32 = source_carriers.resolver_visible_seqused_k_by_head_i32
    setattr(attn_metadata, "mixed_page_resolver_replay_metadata_binding", binding)
    setattr(attn_metadata, "mixed_page_resolver_descriptor", descriptor)
    setattr(attn_metadata, "mixed_page_resolver_captured_descriptor", descriptor)
    setattr(attn_metadata, "mixed_page_resolver_replay_arena", replay_arena)
    setattr(attn_metadata, "mixed_page_resolver_source_arena", source_arena)
    setattr(attn_metadata, "mixed_page_resolver_replay_carriers", replay_carriers)
    setattr(
        attn_metadata,
        "mixed_page_resolver_replay_carrier_sources",
        source_carriers,
    )
    setattr(attn_metadata, "mixed_page_resolver_replay_row_table_i32", replay_arena.row_table_i32)
    setattr(
        attn_metadata,
        "mixed_page_resolver_replay_row_table_source_i32",
        replay_arena.row_table_i32,
    )
    setattr(attn_metadata, "mixed_page_resolver_replay_row_ptr_u64", replay_carriers.resolved_page_table_row_ptr_u64)
    setattr(
        attn_metadata,
        "mixed_page_resolver_replay_row_ptr_source_u64",
        source_carriers.resolved_page_table_row_ptr_u64,
    )
    setattr(attn_metadata, "mixed_page_resolver_replay_affine_i32", replay_carriers.resolved_page_table_affine_i32)
    setattr(
        attn_metadata,
        "mixed_page_resolver_replay_affine_source_i32",
        source_carriers.resolved_page_table_affine_i32,
    )
    setattr(attn_metadata, "mixed_page_resolver_replay_seqused_k_i32", replay_seqused_k_i32)
    setattr(
        attn_metadata,
        "mixed_page_resolver_replay_seqused_k_source_i32",
        replay_seqused_k_i32,
    )
    setattr(
        attn_metadata,
        "mixed_page_resolver_captured_pointer_signature",
        binding.pointer_signature,
    )
    setattr(
        attn_metadata,
        "mixed_page_row_mode_distribution",
        dict(binding.row_mode_distribution),
    )
    setattr(
        attn_metadata,
        "mixed_page_row_source_distribution",
        dict(binding.row_source_distribution),
    )
    setattr(
        attn_metadata,
        "source_counter_schema_version",
        int(binding.source_counter_schema_version),
    )
    setattr(attn_metadata, "expected_rows", int(binding.expected_rows))
    setattr(attn_metadata, "num_kv_heads", int(binding.num_kv_heads))
    setattr(
        attn_metadata,
        "source_counter_missing_fields",
        list(binding.source_counter_missing_fields),
    )
