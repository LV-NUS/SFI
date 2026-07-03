from __future__ import annotations

from typing import Tuple

import torch

from patches.sparse_constants import _ROW_MODE_COMPACT


def sparse_native_visible_values_from_step(
    *,
    controller: object,
    step_authority: object,
    step_bound_meta: object | None,
    page_size: int | None = None,
) -> tuple[Tuple[int, ...], Tuple[int, ...]] | None:
    _svv_cache_on = __import__("os").environ.get("VLLM_SPARSE_CACHE_VISIBLE_VALUES") == "1"
    _svv_key = None
    if _svv_cache_on:
        _svv_key = (id(step_authority), int(getattr(step_authority, "epoch", -1)), id(step_bound_meta), page_size)
        _svv_c = getattr(controller, "_svv_values_cache", None)
        if _svv_c is not None and _svv_c[0] == _svv_key:
            return _svv_c[1]
    try:
        batch_size = int(getattr(step_authority, "batch_size"))
    except (TypeError, ValueError):
        return None
    if batch_size <= 0:
        return None

    context_kv_len = _first_cpu_int_tuple(
        (
            getattr(step_bound_meta, "canonical_real_kv_len_cpu", None)
            if step_bound_meta is not None
            else None
        ),
        getattr(step_authority, "context_kv_len_by_row", None),
        batch_size=batch_size,
    )
    compact_ready = _compact_ready_tuple(step_authority, batch_size=batch_size)
    if compact_ready is None:
        return None
    if context_kv_len is None and not any(compact_ready):
        return None

    launch_plan = (
        getattr(step_bound_meta, "compact_recent_launch_plan", None)
        if step_bound_meta is not None
        else None
    )
    if any(compact_ready):
        if launch_plan is None or not bool(getattr(launch_plan, "valid", False)):
            return None
        page_size_i = _resolve_page_size(controller, page_size)
        if page_size_i is None:
            return None
        try:
            from patches.fa_sparse_runtime.compact_mixed_page_route import (
                resolve_compact_mixed_page_overlay_cpu_geometry,
            )

            (
                _compact_valid,
                _recent_first,
                _recent_count,
                row_effective,
            ) = resolve_compact_mixed_page_overlay_cpu_geometry(
                launch_plan=launch_plan,
                step_bound_meta=step_bound_meta,
                step_authority=step_authority,
                real_kv_len_hint=context_kv_len,
                page_size=int(page_size_i),
                batch_size=batch_size,
            )
        except Exception:
            return None
        row_values = tuple(int(v) for v in row_effective[:batch_size])
    else:
        row_values = tuple(int(v) for v in context_kv_len[:batch_size])

    if len(row_values) != batch_size:
        return None
    launch_values = _cpu_int_tuple(
        getattr(launch_plan, "launch_effective_k_len_cpu", None)
        if launch_plan is not None
        else None,
        batch_size=batch_size,
    )
    if launch_values is None:
        launch_values = row_values
    if _svv_key is not None:
        try:
            setattr(controller, "_svv_values_cache", (_svv_key, (row_values, launch_values)))
        except Exception:
            pass
    return row_values, launch_values


def _first_cpu_int_tuple(*values: object, batch_size: int) -> tuple[int, ...] | None:
    for value in values:
        result = _cpu_int_tuple(value, batch_size=batch_size)
        if result is not None:
            return result
    return None


def _cpu_int_tuple(value: object, *, batch_size: int) -> tuple[int, ...] | None:
    if value is None or isinstance(value, torch.Tensor):
        return None
    try:
        result = tuple(int(v) for v in tuple(value)[: int(batch_size)])
    except Exception:
        return None
    if len(result) != int(batch_size):
        return None
    return result


def _compact_ready_tuple(
    step_authority: object,
    *,
    batch_size: int,
) -> tuple[bool, ...] | None:
    row_mode_by_row = getattr(step_authority, "row_mode_by_row", None)
    if row_mode_by_row is not None and not isinstance(row_mode_by_row, torch.Tensor):
        try:
            row_modes = tuple(row_mode_by_row)[: int(batch_size)]
        except Exception:
            row_modes = tuple()
        if len(row_modes) == int(batch_size):
            try:
                return tuple(int(v) == int(_ROW_MODE_COMPACT) for v in row_modes)
            except Exception:
                return tuple(str(v).strip().lower() == "compact" for v in row_modes)

    use_compact_by_row = getattr(step_authority, "use_compact_by_row", None)
    if use_compact_by_row is None or isinstance(use_compact_by_row, torch.Tensor):
        return None
    try:
        values = tuple(use_compact_by_row)[: int(batch_size)]
    except Exception:
        return None
    if len(values) != int(batch_size):
        return None
    return tuple(bool(v) for v in values)


def _resolve_page_size(controller: object, page_size: int | None) -> int | None:
    if page_size is not None and int(page_size) > 0:
        return int(page_size)
    sparse_state = getattr(controller, "_sparse_native_batch_state", None)
    structural_key = getattr(sparse_state, "structural_key", None)
    if callable(structural_key):
        try:
            key = structural_key()
            value = int(getattr(key, "page_size"))
            if value > 0:
                return value
        except Exception:
            pass
    capacity_key = getattr(controller, "_sparse_native_capacity_key", None)
    if isinstance(capacity_key, tuple) and len(capacity_key) >= 3:
        try:
            value = int(capacity_key[2])
            if value > 0:
                return value
        except Exception:
            pass
    return None
