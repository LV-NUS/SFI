"""
Pure utility functions for the sparse attention engine.

All functions here are stateless (no module-level mutable state references),
making them safe to import from any module without circular dependency risk.
"""
from __future__ import annotations

import logging
import os
from contextlib import nullcontext
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple

import torch

_log = logging.getLogger(__name__)

from patches.sparse_constants import (
    _DYNAMIC_ENV,
    _ROW_MODE_COMPACT,
    _ROW_MODE_DENSE,
    _ROW_MODE_LOG_F_PREFILL,
    _ROW_MODE_LOG_F_REFRESH,
    _SELECTOR_FIXED_K_CACHED,
    _is_free_slot_id,
)

if TYPE_CHECKING:
    from patches.layer_state import LayerState
    from patches.step_authority import StepAuthority


# ---------------------------------------------------------------------------
# Alignment / math
# ---------------------------------------------------------------------------

def _align_up_int(value: int, align: int) -> int:
    if align <= 0:
        return int(value)
    v = int(value)
    a = int(align)
    return ((v + a - 1) // a) * a


# ---------------------------------------------------------------------------
# Selector / row-mode helpers
# ---------------------------------------------------------------------------

def _selector_fixed_k_enabled() -> bool:
    if _DYNAMIC_ENV:
        return os.environ.get("VLLM_SPARSE_SELECTOR_FIXED_K", "1") == "1"
    return bool(_SELECTOR_FIXED_K_CACHED)


def _submission_slot_owner_snapshot(
    payloads: Sequence[object],
    slot_list: Sequence[int],
    *,
    stage: str,
) -> Tuple[Tuple[int, str], ...]:
    """Validate and return the immutable slot owners carried by payloads."""

    if not payloads:
        return tuple()
    slots = tuple(int(slot) for slot in slot_list)
    if len(set(slots)) != len(slots):
        raise RuntimeError(f"{stage} slot_list contains duplicate slots")

    first_req_ids_raw = getattr(payloads[0], "slot_req_ids", None)
    if first_req_ids_raw is None:
        raise RuntimeError(
            f"{stage} missing submission-step slot_req_ids "
            "(live batch_request_ids fallback retired)"
        )

    def _normalize(raw_req_ids: Sequence[object]) -> Tuple[str, ...]:
        req_ids = []
        for req_id in raw_req_ids:
            if req_id is None or _is_free_slot_id(req_id):
                raise RuntimeError(
                    f"{stage} contains an invalid submission-step request identity"
                )
            req_ids.append(str(req_id))
        return tuple(req_ids)

    req_ids = _normalize(first_req_ids_raw)
    if len(req_ids) != len(slots):
        raise RuntimeError(
            f"{stage} slot_req_ids/slot_list length mismatch: "
            f"slot_req_ids={len(req_ids)} slot_list={len(slots)}"
        )
    for payload in payloads[1:]:
        payload_slots = tuple(int(slot) for slot in getattr(payload, "slot_list", ()))
        payload_req_ids_raw = getattr(payload, "slot_req_ids", None)
        if payload_req_ids_raw is None:
            raise RuntimeError(
                f"{stage} missing submission-step slot_req_ids "
                "(live batch_request_ids fallback retired)"
            )
        if payload_slots != slots or _normalize(payload_req_ids_raw) != req_ids:
            raise RuntimeError(
                f"{stage} submission identity mismatch across layers"
            )
    return tuple(zip(slots, req_ids))


def _resolve_compact_row_gate(
    *,
    requested_compact: bool,
    pending_refresh: bool,
    refresh_this: bool,
    compact_kv_len: int,
) -> Tuple[bool, str]:
    """Resolve whether a row may use compact path in current layer.

    Returns:
        (allow_compact, reason)
    """
    if not bool(requested_compact):
        return False, "not_requested"
    if bool(pending_refresh) and (not bool(refresh_this)):
        return False, "pending_refresh_not_cleared"
    if int(compact_kv_len) <= 0:
        return False, "compact_not_ready"
    return True, "ready"


# ---------------------------------------------------------------------------
# Cache key constructors
# ---------------------------------------------------------------------------

ALL_FALSE_SIGNATURE: Tuple[str, ...] = ("ALL_FALSE_SIGNATURE",)


def _make_decode_plan_version(
    *,
    step_handle_id: int,
    step_handle_generation: int,
) -> int:
    """Build a stable step-local decode plan version (int-only)."""
    handle_id_i = int(step_handle_id)
    handle_gen_i = int(step_handle_generation)
    if handle_id_i <= 0 or handle_gen_i <= 0:
        return -1
    # High bits carry generation; low bits carry handle id.
    return (handle_gen_i << 32) | (handle_id_i & 0xFFFFFFFF)


def normalize_layer_effective_refresh_signature(
    layer_effective_refresh_by_row: Sequence[bool],
    *,
    batch_size: int,
) -> Tuple[object, ...]:
    if batch_size <= 0:
        return tuple()
    if len(layer_effective_refresh_by_row) != batch_size:
        raise RuntimeError(
            "layer_effective_refresh_by_row must exactly match batch size: "
            f"rows={len(layer_effective_refresh_by_row)} batch={batch_size}"
        )
    signature = tuple(bool(layer_effective_refresh_by_row[i]) for i in range(batch_size))
    if not any(signature):
        return ALL_FALSE_SIGNATURE
    return signature


def build_step_plan_signature(
    *,
    req_ids: Sequence[str],
    slot_by_row: Sequence[int],
    row_mode_by_row: Sequence[int],
    layer_effective_refresh_by_row: Sequence[bool],
    row_policy_ready_by_row: Sequence[bool],
) -> Tuple[object, ...]:
    """Build stable signatures for step plan caches."""
    batch_size = len(req_ids)
    if len(slot_by_row) != batch_size:
        raise RuntimeError(
            "slot_by_row must exactly match req_ids: "
            f"slots={len(slot_by_row)} reqs={batch_size}"
        )
    if len(row_mode_by_row) != batch_size:
        raise RuntimeError(
            "row_mode_by_row must exactly match req_ids: "
            f"modes={len(row_mode_by_row)} reqs={batch_size}"
        )
    if len(layer_effective_refresh_by_row) != batch_size:
        raise RuntimeError(
            "layer_effective_refresh_by_row must exactly match req_ids: "
            f"rows={len(layer_effective_refresh_by_row)} reqs={batch_size}"
        )
    layer_effective_refresh_signature = normalize_layer_effective_refresh_signature(
        layer_effective_refresh_by_row,
        batch_size=batch_size,
    )
    if len(row_policy_ready_by_row) != batch_size:
        raise RuntimeError(
            "row_policy_ready_by_row must exactly match req_ids: "
            f"rows={len(row_policy_ready_by_row)} reqs={batch_size}"
        )
    bootstrap_refresh = tuple(
        bool(layer_effective_refresh_by_row[row])
        and (not bool(row_policy_ready_by_row[row]))
        for row in range(batch_size)
    )
    bootstrap_refresh_signature = (
        bootstrap_refresh if any(bootstrap_refresh) else ALL_FALSE_SIGNATURE
    )
    if isinstance(req_ids, tuple) and len(req_ids) == batch_size:
        req_sig = req_ids
    else:
        req_sig = tuple(str(req_ids[row]) for row in range(batch_size))
    if isinstance(slot_by_row, tuple) and len(slot_by_row) == batch_size:
        slot_sig = slot_by_row
    else:
        slot_sig = tuple(int(slot_by_row[row]) for row in range(batch_size))
    if isinstance(row_mode_by_row, tuple) and len(row_mode_by_row) == batch_size:
        row_mode_sig = row_mode_by_row
    else:
        row_mode_sig = tuple(int(row_mode_by_row[row]) for row in range(batch_size))
    plan_signature: Tuple[object, ...] = (
        req_sig,
        slot_sig,
        row_mode_sig,
        layer_effective_refresh_signature,
        bootstrap_refresh_signature,
    )
    return plan_signature


def _make_step_cache_key(
    step_authority: "StepAuthority",
    state: LayerState,
    *,
    force_dense: bool,
    force_compact_off: bool,
    layer_effective_refresh_by_row: Sequence[bool],
) -> Tuple[object, ...]:
    """构造 per-layer step cache 的稳定 key（避免每步重建）。"""
    batch_size = int(step_authority.batch_size)
    refresh_row_signature = normalize_layer_effective_refresh_signature(
        layer_effective_refresh_by_row,
        batch_size=batch_size,
    )
    if len(step_authority.slot_by_row) != batch_size:
        raise RuntimeError(
            "slot_by_row must exactly match batch size: "
            f"slots={len(step_authority.slot_by_row)} batch={batch_size}"
        )
    if len(step_authority.row_mode_by_row) != batch_size:
        raise RuntimeError(
            "row_mode_by_row must exactly match batch size: "
            f"modes={len(step_authority.row_mode_by_row)} batch={batch_size}"
        )
    slot_by_row = step_authority.slot_by_row
    row_mode_by_row = step_authority.row_mode_by_row
    slot_signature = (
        slot_by_row
        if isinstance(slot_by_row, tuple) and len(slot_by_row) == batch_size
        else tuple(int(slot_by_row[row]) for row in range(batch_size))
    )
    row_mode_signature = (
        row_mode_by_row
        if isinstance(row_mode_by_row, tuple) and len(row_mode_by_row) == batch_size
        else tuple(int(row_mode_by_row[row]) for row in range(batch_size))
    )
    return (
        step_authority.req_ids,
        slot_signature,
        row_mode_signature,
        step_authority.row_policy_ready_by_row,
        step_authority.short_dense_by_row,
        refresh_row_signature,
        state.compact_meta_epoch,
        force_dense,
        force_compact_off,
    )


def _make_step_decode_cache_key(
    step_authority: "StepAuthority",
    *,
    layer_cache_keys: Sequence[int],
    layer_effective_refresh_by_row: Sequence[bool],
    compact_layout_generation: int,
    refresh_signature_override: Optional[Tuple[object, ...]] = None,
) -> Tuple[object, ...]:
    """构造 step 级稳定 cache key，用于复用 StepDecodeData/Plan。"""
    batch_size = int(step_authority.batch_size)
    if refresh_signature_override is not None:
        refresh_row_signature = refresh_signature_override
    else:
        refresh_row_signature = normalize_layer_effective_refresh_signature(
            layer_effective_refresh_by_row,
            batch_size=batch_size,
        )
    if len(step_authority.slot_by_row) != batch_size:
        raise RuntimeError(
            "slot_by_row must exactly match batch size: "
            f"slots={len(step_authority.slot_by_row)} batch={batch_size}"
        )
    if len(step_authority.row_mode_by_row) != batch_size:
        raise RuntimeError(
            "row_mode_by_row must exactly match batch size: "
            f"modes={len(step_authority.row_mode_by_row)} batch={batch_size}"
        )
    slot_by_row = step_authority.slot_by_row
    row_mode_by_row = step_authority.row_mode_by_row
    slot_signature = (
        slot_by_row
        if isinstance(slot_by_row, tuple) and len(slot_by_row) == batch_size
        else tuple(int(slot_by_row[row]) for row in range(batch_size))
    )
    row_mode_signature = (
        row_mode_by_row
        if isinstance(row_mode_by_row, tuple) and len(row_mode_by_row) == batch_size
        else tuple(int(row_mode_by_row[row]) for row in range(batch_size))
    )
    return (
        step_authority.req_ids,
        slot_signature,
        row_mode_signature,
        step_authority.row_policy_ready_by_row,
        step_authority.short_dense_by_row,
        tuple(layer_cache_keys),
        int(compact_layout_generation),
        refresh_row_signature,
    )


def assert_cleanup_ledgers_drained_for_step_build(owner: object, *, stage: str) -> None:
    pending_finished = getattr(owner, "_finished_req_ids_step", None)
    if pending_finished:
        raise RuntimeError(
            f"{stage} observed non-empty finished cleanup ledger"
        )
    pending_global_slot_releases = getattr(owner, "_pending_global_slot_releases", None)
    if pending_global_slot_releases:
        raise RuntimeError(
            f"{stage} observed non-empty pending global slot releases"
        )


# ---------------------------------------------------------------------------
# Profiling / NVTX placeholder
# ---------------------------------------------------------------------------

def _range(name: str):
    # 旧的 NVTX / record_function profile 体系已清理；保留调用点作为"无开销"占位。
    # 统一依赖 kernel-level probe（见 triton kernel 侧），避免 Python 热路径被 profile 逻辑污染。
    return nullcontext()


# ---------------------------------------------------------------------------
# Block-table / tensor helpers
# ---------------------------------------------------------------------------

def _row_keys(block_table: torch.Tensor, count: int) -> List[int]:
    """Return unique identifiers for block-table rows."""
    if count <= 0:
        return []
    stride0 = block_table.stride(0)
    elem_size = block_table.element_size()
    base_ptr = block_table.data_ptr()
    return [base_ptr + idx * stride0 * elem_size for idx in range(count)]


def _compute_recent_window(context_kv_len: int, recent_cap: int, block_size: int) -> Tuple[int, int]:
    """按照 block 对齐计算 recent 起点与长度。

    Args:
        context_kv_len: 当前上下文可见的 KV 长度（seqused_k）
        recent_cap: recent 窗口上限（未对齐）
        block_size: KV cache 的 block 大小

    Returns:
        (recent_start, recent_len)，当 recent_cap<=0 或 block_size<=0 时返回 (context_kv_len, 0)。
    """
    if context_kv_len <= 0 or recent_cap <= 0 or block_size <= 0:
        return context_kv_len, 0
    cap = min(recent_cap, context_kv_len)
    recent_start = ((context_kv_len - cap) // block_size) * block_size
    recent_start = max(0, recent_start)
    recent_len = max(0, context_kv_len - recent_start)
    return recent_start, recent_len


def _make_selector_fast_signature(
    *,
    capture_scores: torch.Tensor,
    log_f_denoms: Optional[torch.Tensor] = None,
    kv_lengths: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
) -> Optional[Tuple[object, ...]]:
    """Build a lightweight payload signature for cached-layout validation."""
    if not isinstance(capture_scores, torch.Tensor):
        return None
    try:
        denoms_ptr = (
            int(log_f_denoms.data_ptr())
            if isinstance(log_f_denoms, torch.Tensor) and log_f_denoms.numel() > 0
            else 0
        )
        kv_ptr = (
            int(kv_lengths.data_ptr())
            if isinstance(kv_lengths, torch.Tensor) and kv_lengths.numel() > 0
            else 0
        )
        block_ptr = (
            int(block_table.data_ptr())
            if isinstance(block_table, torch.Tensor) and block_table.numel() > 0
            else 0
        )
        return (
            tuple(int(v) for v in capture_scores.shape),
            tuple(int(v) for v in capture_scores.stride()),
            int(capture_scores.data_ptr()),
            str(capture_scores.device),
            str(capture_scores.dtype),
            denoms_ptr,
            kv_ptr,
            block_ptr,
        )
    except Exception:
        _log.warning("Failed to build selector fast signature", exc_info=True)
        raise


def _is_stream_capturing_or_raise(*, stage: str) -> bool:
    """Return ``True`` when the current CUDA stream is capturing a graph."""
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception as exc:
        raise RuntimeError(
            f"{stage}: failed to query capture-state; refuse silent skip"
        ) from exc


_SELECTOR_BATCH_EXT_MODULE: Optional[object] = None
_SELECTOR_BATCH_EXT_LOADED: bool = False


def _get_selector_batch_ext():
    """Lazily load selector_batch_ext module, cached after first call."""
    global _SELECTOR_BATCH_EXT_MODULE, _SELECTOR_BATCH_EXT_LOADED
    if not _SELECTOR_BATCH_EXT_LOADED:
        try:
            from utils import selector_batch_ext as _mod
            _SELECTOR_BATCH_EXT_MODULE = _mod
        except Exception:
            _log.warning("Failed to load selector_batch_ext module", exc_info=True)
            raise
        _SELECTOR_BATCH_EXT_LOADED = True
    return _SELECTOR_BATCH_EXT_MODULE
