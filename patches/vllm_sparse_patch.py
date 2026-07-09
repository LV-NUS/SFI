"""
patches/vllm_sparse_patch.py — Sparse Controller 运行时核心

OWNS:
  - VLLMSparseController: 稀疏注意力主控制器（调度、生命周期、step 管理）
  - Lazy loaders (_load_*): 运行时 worker 模块延迟导入
  - Bridge methods: 连接 controller 与 runtime workers 的桥接调用
  - Module-level utility functions: 缓存、行模式解析、payload 准备

DEPENDS_ON:
  - patches/sparse_types.py: 所有数据类 (SparseControllerConfig, StepMeta, ...)
  - patches/layer_state.py: LayerState (per-layer 可变状态)
  - patches/controller_mixins/*: 6 个行为 mixin
  - patches/*_runtime/: decode/refresh/selector 运行时 worker
  - patches/patch_installer.py: patch 安装/卸载（启动时）

ENTRY_POINTS:
  - VLLMSparseController.__init__(config): 控制器初始化
  - _patched_unified_attention(): 正式 unified_attention 入口桥
  - apply_vllm_sparse_patch(config): 安装 sparse patch（委托 patch_installer）
  - ensure_vllm_sparse_patch_from_env(): 从环境变量自动安装（委托 patch_installer）

Mixin Cross-Domain Contract:
  ProfileMixin        → reads: layer_states, config
                      → owns: _refresh_profile_*, _flush_micro_profile_*
  CaptureRingMixin    → reads: layer_states, config
                      → owns: _capture_ring_*, _map_global_layer_to_capture_slot
  CompactKVMixin      → reads: layer_states, _global_slot_allocator
                      → owns: _compact_*, _ensure_compact_*
  WaitDeciderMixin    → reads: step_exec_hints, config
                      → owns: _wait_decider_*, _should_refresh_*
  RefreshRebuildMixin → calls: _ensure_capture_layout_cpu_tensors [CaptureRingMixin]
                      → owns: _rebuild_*, _refresh_rebuild_*
  SelectorComputeMixin → calls: _get_step_capture_layout [main bridge]
                       → calls: _rebuild_compact_slots_batched_layers_from_selection [main bridge]
                       → owns: _compute_alpha_*, _select_alpha_*, _ensure_selector_*
"""

from __future__ import annotations

import heapq
import json
import logging
import os
import sys
from collections import deque
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch

from patches.refresh_runtime.capture_live_lengths import (
    get_step_plan_cap_tensor,
    refresh_capture_layout_live_lengths,
)
from patches.cpu_gpu_staging import cached_cpu_tensor_to_device, cached_sequence_to_device
from utils.selector_key_norms_ext import compute_key_norms_paged_batched_layers_delta_cuda
from utils.sentence_triggers import DEFAULT_MIN_REFRESH_GAP, RefreshTrigger

from patches.runtime_contracts import StepSemanticSnapshot
from patches.runtime_state import StepRuntimeState
from patches.step_decode_pipeline import (
    apply_ordered_layer_plan_state,
    reset_runtime_plan_state,
)
from patches.decode_runtime.plan_builder import build_decode_reuse_order_decision


# [F2] trace path 进程内不变：import-time 缓存；pytest/显式动态档 live 读
# （_DYNAMIC_ENV 于下方 sparse_constants 块引入，运行时解析）。
_SELECTED_READY_TRACE_PATH_CACHED = os.environ.get(
    "VLLM_SPARSE_SELECTED_READY_TRACE_LOG", ""
).strip()


def _selected_ready_trace_path() -> str:
    if _DYNAMIC_ENV:
        return os.environ.get("VLLM_SPARSE_SELECTED_READY_TRACE_LOG", "").strip()
    return _SELECTED_READY_TRACE_PATH_CACHED


def _append_selected_ready_trace(event: dict[str, object]) -> None:
    path = _selected_ready_trace_path()
    if not path:
        return
    record = dict(event)
    record.setdefault("pid", os.getpid())
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")

from patches.refresh_runtime.flush_worker import flush_prefill_batches_impl

from patches.buffer_allocator_backends import AllocatorBackend, resolve_allocator_backend
from patches.global_slot_allocator import GlobalSlotAllocator
from patches.request_intent_ticket import (
    PendingPolicy,
    PendingReasonCode,
    RequestIntentTicket,
    TicketState,
    materialize_refresh_reqs,
    pending_reason_code_to_text,
    pending_reason_to_code,
)
from patches.refresh_runtime.workload_plan import (
    index_workload_plan_events,
    indexed_workload_plan_events_for_request_step,
    load_workload_plan_from_env,
    workload_plan_pending_policy,
)
from patches.prefill_capture_meta_arena import (
    ARENA_PHASE1_QWEN06_BS2_BUDGET_BYTES,
    ARENA_PHASE1_QWEN06_BS2_CAP262K_BUDGET_BYTES,
    ArenaReservationStatus,
    CaptureArenaIntent,
    SparseCaptureMetaArena,
)
from patches.sparse_types import (
    ActiveStepSnapshot,
    StepBoundMeta,
    RequestTracking,
    SelectorBatchPayload,
    SparseControllerConfig,
    StepCaptureLayout,
    StepContext,
    StepDecodeData,
    StepDispatchPlan,
    StepExecHints,
    StepHandle,
    StepRefreshMode,
    StepRefreshPlan,
    StepTicket,
    StepMeta,
    LayerDecodeData,
    _FlushProfileAccum,
    _RefreshProfilePending,
    continuous_producer_enabled,
)
if TYPE_CHECKING:
    from patches.sparse_types import BoundLayerMeta
from patches.step_authority import StepAuthority
from patches.fa_sparse_runtime.compact_recent_alignment import (
    compact_slot_offset_tokens,
)
from patches.layer_state import LayerState
from patches.sparse_constants import (
    _CAPTURE_CHUNK,
    _CAPTURE_IN_FLIGHT,
    _DYNAMIC_ENV,
    compact_gen_count,
    _FORCE_COMPACT_OFF_CACHED,
    _FORCE_DENSE_CACHED,
    _FREE_SLOT_ID,
    _REBUILD_PHYSICAL_BLOCK_SORT_CACHED,
    _REFRESH_MICRO_PROFILE_CACHED,
    _RELEASE_ON_IDLE_CACHED,
    _is_free_slot_id,
)
from patches.sparse_utils import (
    _align_up_int,
    _is_stream_capturing_or_raise,
    _make_selector_fast_signature,
    assert_cleanup_ledgers_drained_for_step_build,
)
from patches.controller_mixins import (
    CaptureRingMixin,
    CompactKVMixin,
    ProfileMixin,
    RefreshRebuildMixin,
    SelectorComputeMixin,
    WaitDeciderMixin,
)

try:
    from vllm.logger import init_logger
    _log = init_logger(__name__)
except Exception:
    import logging
_log = logging.getLogger(__name__)
# -----------------------------------------------------------------------------
# Mutable module-level state
# -----------------------------------------------------------------------------




# Micro-profiling ring buffer (64-sample, auto-flush P50/P95/Max to stderr)
_REFRESH_MICRO_PROFILE_BUF: deque = deque(maxlen=64)
_REFRESH_MICRO_PROFILE_COUNT = [0]  # mutable int wrapper; avoids 'global' in methods/closures


def _flush_micro_profile_summary() -> None:
    """Print P50/P95/Max of micro-profiling ring buffer to stderr."""
    buf = _REFRESH_MICRO_PROFILE_BUF
    n = len(buf)
    if n == 0:
        return
    # Each entry: (record_us, selector_us, rebuild_us, overhead_us, total_us)
    names = ("record", "selector", "rebuild", "overhead", "total")
    parts: list = []
    for i, name in enumerate(names):
        vals = sorted(s[i] for s in buf)
        p50 = vals[len(vals) * 50 // 100] if vals else 0.0
        p95 = vals[min(len(vals) * 95 // 100, len(vals) - 1)] if vals else 0.0
        mx = vals[-1] if vals else 0.0
        parts.append(f"{name}={p50:.0f}/{p95:.0f}/{mx:.0f}")
    sys.stderr.write(f"[REFRESH_MICRO] {' '.join(parts)} \u03bcs (P50/P95/Max, N={n})\n")
    sys.stderr.flush()


_CURRENT_UNIFIED_ATTENTION_MODE = "default"












# -----------------------------------------------------------------------------
# 极简 Decode 快速路径数据结构
# 目标：消除 per-layer Python 开销，使 decode 稳态只需"读缓存 + 调 kernel"
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Helper utilities

def _normalize_capture_layout_views(
    *,
    layout,
    slot_list,
    seqused_k,
    device,
    step_context,
    controller,
    state,
):
    """Shared normalization for capture layout views used by both prefill and refresh."""
    rows = layout.row_tensor

    row_list_cpu = layout.row_list_cpu
    if row_list_cpu is None:
        row_list_cpu = controller._slots_to_rows_for_step_context(
            step_context=step_context,
            state=state,
            slot_list=layout.slot_list,
        )
        layout.row_list_cpu = row_list_cpu

    if not step_context.seq_lens:
        raise RuntimeError(
            "capture payload missing seq_lens in strict path; "
            f"(epoch={int(step_context.epoch)}, slots={len(slot_list)})"
        )
    bound_meta = controller._require_step_bound_meta(
        step_context=step_context,
        stage="capture payload normalization",
    )
    cap_tensor = get_step_plan_cap_tensor(
        controller=controller,
        bound_meta=bound_meta,
        plan_cap_by_row=bound_meta.logits_capacity_by_row,
        device=device,
    )
    refresh_capture_layout_live_lengths(
        layout=layout,
        row_tensor=rows,
        row_list=row_list_cpu,
        seqused_k=seqused_k,
        cap_tensor=cap_tensor,
        step_context=step_context,
        device=device,
        num_heads=int(layout.num_heads),
        chunk_query_lengths=None,
        plan_cap_by_row_cpu=bound_meta.logits_capacity_by_row,
        context_kv_len_by_row_cpu=bound_meta.context_kv_len_by_row,
    )
    kv_lengths_tensor = layout.kv_lengths
    seq_lens_batch = layout.seq_lens_batch
    seq_lens_batch_i32 = getattr(layout, "seq_lens_batch_i32", None)
    if (
        seq_lens_batch is not None
        and (
            seq_lens_batch_i32 is None
            or not isinstance(seq_lens_batch_i32, torch.Tensor)
            or seq_lens_batch_i32.device != device
            or seq_lens_batch_i32.dtype != torch.int32
            or seq_lens_batch_i32.numel() < len(layout.slot_list)
        )
    ):
        seq_lens_batch_i32 = seq_lens_batch.to(device=device, dtype=torch.int32)
        layout.seq_lens_batch_i32 = seq_lens_batch_i32
    kv_len_per_row_i32 = layout.kv_len_per_row_i32
    seq_lens_cpu = layout.seq_lens_cpu

    slot_tensor_i32 = layout.slot_tensor_i32
    if (
        slot_tensor_i32 is None
        or slot_tensor_i32.device != device
        or slot_tensor_i32.dtype != torch.int32
        or slot_tensor_i32.numel() < len(layout.slot_list)
    ):
        slot_tensor_i32 = layout.slot_tensor.to(device=device, dtype=torch.int32)
        layout.slot_tensor_i32 = slot_tensor_i32
    row_tensor_i32 = layout.row_tensor_i32
    if (
        row_tensor_i32 is None
        or row_tensor_i32.device != device
        or row_tensor_i32.dtype != torch.int32
        or row_tensor_i32.numel() < len(layout.slot_list)
    ):
        row_tensor_i32 = layout.row_tensor.to(device=device, dtype=torch.int32)
        layout.row_tensor_i32 = row_tensor_i32

    self_slot_tensor_cpu = layout.slot_tensor_cpu
    self_seq_lens_tensor_cpu = layout.seq_lens_tensor_cpu
    if self_slot_tensor_cpu is None or self_slot_tensor_cpu.numel() != len(layout.slot_list):
        self_slot_tensor_cpu = torch.tensor(layout.slot_list, dtype=torch.long)
        layout.slot_tensor_cpu = self_slot_tensor_cpu
    if seq_lens_cpu is not None:
        if (
            self_seq_lens_tensor_cpu is None
            or self_seq_lens_tensor_cpu.numel() != len(seq_lens_cpu)
        ):
            self_seq_lens_tensor_cpu = torch.tensor(
                [max(0, int(s)) for s in seq_lens_cpu], dtype=torch.long
            )
            layout.seq_lens_tensor_cpu = self_seq_lens_tensor_cpu

    return (
        kv_lengths_tensor,
        seq_lens_batch,
        kv_len_per_row_i32,
        row_list_cpu,
        seq_lens_cpu,
        slot_tensor_i32,
        self_slot_tensor_cpu,
        self_seq_lens_tensor_cpu,
    )


def _prepare_prefill_capture_payload(
    *,
    controller: "VLLMSparseController",
    cache_key: int,
    state: LayerState,
    step_context: StepContext,
    q: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqused_k: torch.Tensor,
    chunk_query_lengths: torch.Tensor,
    num_heads: int,
    device: torch.device,
    capture_plan: Optional[Dict[int, int]] = None,
) -> Optional[
    Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        List[int],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        List[int],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Tuple[int, ...],
        Optional[torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
    ]
]:
    prefill_impl, _ = _load_payload_builder_impls()
    return prefill_impl(
        controller=controller,
        cache_key=cache_key,
        state=state,
        step_context=step_context,
        q=q,
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=seqused_k,
        chunk_query_lengths=chunk_query_lengths,
        num_heads=num_heads,
        device=device,
        capture_plan=capture_plan,
    )


def _prepare_refresh_capture_payload(
    *,
    controller: "VLLMSparseController",
    cache_key: int,
    state: LayerState,
    step_context: StepContext,
    q: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqused_k: torch.Tensor,
    slots_filter: Optional[Sequence[int]] = None,
    slots_filter_sorted: bool = False,
    layout: Optional["StepCaptureLayout"] = None,
    ) -> Optional[
    Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        List[int],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        List[int],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Tuple[int, ...],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
    ]
]:
    _, refresh_impl = _load_payload_builder_impls()
    return refresh_impl(
        controller=controller,
        cache_key=cache_key,
        state=state,
        step_context=step_context,
        q=q,
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=seqused_k,
        slots_filter=slots_filter,
        slots_filter_sorted=slots_filter_sorted,
        layout=layout,
    )


def _revoke_slot_selected_truth(state: LayerState, *, slot: int) -> None:
    slot_i = int(slot)
    if slot_i < 0:
        return
    if slot_i < len(state.sparse_selected_middle_pages):
        state.sparse_selected_middle_pages[slot_i] = torch.empty(
            0,
            dtype=torch.int32,
            device=state.device,
        )
    if slot_i < len(state.sparse_selected_middle_counts):
        state.sparse_selected_middle_counts[slot_i] = torch.empty(
            0,
            dtype=torch.int32,
            device=state.device,
        )
    if slot_i < len(state.sparse_request_refresh_generation):
        state.sparse_request_refresh_generation[slot_i] = 0
    if slot_i < len(state.sparse_selected_middle_uniform_count_cpu):
        state.sparse_selected_middle_uniform_count_cpu[slot_i] = 0
    if slot_i < len(state.sparse_selected_middle_min_logical_page_cpu):
        state.sparse_selected_middle_min_logical_page_cpu[slot_i] = -1
    if slot_i < len(state.sparse_selected_middle_max_logical_page_cpu):
        state.sparse_selected_middle_max_logical_page_cpu[slot_i] = -1
    dense_pages = state.sparse_selected_middle_pages_dense_i32
    if dense_pages is not None and slot_i < int(dense_pages.shape[0]):
        dense_pages[slot_i].fill_(-1)
    dense_counts = state.sparse_selected_middle_counts_dense_i32
    if dense_counts is not None and slot_i < int(dense_counts.shape[0]):
        dense_counts[slot_i].zero_()


def _invalidate_page_sparse_step_cache_truth(state: LayerState) -> None:
    state.step_cache_epoch = -1
    state.step_cache_key = None
    state.step_cache_plan_version = -1
    state.step_cache_meta_packed = False
    state.step_cache_selected_static_pages_i32 = None
    state.step_cache_selected_static_seqused_k_by_head_i32 = None
    state.step_cache_selected_static_schema_version = 0
    state.step_cache_page_table_i32 = None
    state.step_cache_selected_seqused_k_by_head_i32 = None
    state.step_cache_cp_selected_seqused_k_by_head_i32 = None
    state.step_cache_real_kv_len_i32 = None
    state.step_cache_kv_batch_idx_i32 = None
    state.step_cache_page_table_layout = -1
    state.step_cache_has_page_sparse = False
    state.step_cache_all_page_sparse = False
    state.step_cache_applied_recent_epoch_i32 = None
    state.step_cache_applied_refresh_generation_i32 = None
    state.step_cache_materialize_status_i32 = None
    state.step_cache_patch_status_i32 = None
    state.step_cache_applied_recent_epoch_value = -1
    state.step_cache_requested_refresh_generation_signature = None
    state.step_cache_cached_lengths_ok = False
    state.step_cache_cached_kv_batch_idx_identity = False
    state.step_cache_cached_status_ok = False
    state.step_cache_cached_freshness_ok = False
    state.step_cache_cached_launch_ready = False


def _cleanup_inactive_slots(state: LayerState, controller: Optional['VLLMSparseController']) -> None:
    """清理已结束 request 对应的 slot，释放其 compact 占位。"""
    if controller is None or not state.batch_request_ids:
        return
    changed = False

    # 优先使用 vLLM 的 finished_req_ids 精确清理，避免在多 request 下：
    # - 误把“本步未调度”的请求当作结束；
    # - 每层构造 active_ids(set) 带来的额外开销。
    finished_ids = controller._finished_req_ids_step
    if finished_ids:
        for rid in finished_ids:
            slot = state.request_id_to_slot.get(rid)
            if slot is None:
                continue
            if slot < 0 or slot >= len(state.batch_request_ids):
                continue
            if _is_free_slot_id(state.batch_request_ids[slot]):
                continue
            changed = True
            if rid in state.request_id_to_slot:
                del state.request_id_to_slot[rid]
            state.batch_request_ids[slot] = _FREE_SLOT_ID
            # batch_request_ids 已标记为 FREE_SLOT_ID，本轮/后续 cleanup 不会重复命中该 slot；
            # 因此无需 O(n) 的 membership 检查。
            heapq.heappush(state.free_slots, int(slot))
            if slot < len(state.compact_sink_len):
                state.compact_sink_len[slot] = 0
            if slot < len(state.compact_persist_len):
                state.compact_persist_len[slot] = 0
            if slot < len(state.compact_kv_len):
                state.compact_kv_len[slot] = 0
            if slot < state.key_norms_len.size(0):
                state.key_norms_len[slot] = 0
            if slot < len(state.key_norms_capacity):
                state.key_norms_capacity[slot] = 0
            if state.prefill_done_mask is not None and slot < state.prefill_done_mask.shape[0]:
                state.prefill_done_mask[slot] = False
            if state.prefill_active_mask is not None and slot < state.prefill_active_mask.shape[0]:
                state.prefill_active_mask[slot] = False
            if state.prefill_fifo_counts_cpu is not None and slot < len(state.prefill_fifo_counts_cpu):
                state.prefill_fifo_counts_cpu[slot] = 0
            if state.prefill_kv_len_per_row_i32 is not None and slot < state.prefill_kv_len_per_row_i32.shape[0]:
                state.prefill_kv_len_per_row_i32[slot] = 0
            if state.prefill_kv_lengths is not None and slot < state.prefill_kv_lengths.shape[0]:
                state.prefill_kv_lengths[slot].zero_()
            state.slot_batch_rows = None
            if state.slot_batch_rows_cpu is not None and slot < len(state.slot_batch_rows_cpu):
                state.slot_batch_rows_cpu[slot] = -1
            steps_cpu = state.last_refresh_step_per_slot_cpu
            if steps_cpu is not None and slot < len(steps_cpu):
                steps_cpu[slot] = -1
            steps_decode_cpu = state.last_refresh_decode_per_slot_cpu
            if steps_decode_cpu is not None and slot < len(steps_decode_cpu):
                steps_decode_cpu[slot] = -1
            # skip_unchanged 模式下，必须清除 compact_pos 避免新请求的 sink token
            # 与旧 slot 的 pos 恰好相等导致跳过 copy 的静默错误。
            if state.compact_pos is not None and slot < len(state.compact_pos):
                pos_buf = state.compact_pos[slot]
                if pos_buf is not None:
                    pos_buf.fill_(-1)
                # [DUAL-GEN-RELEASE-POS-ALLGENS 2026-07-07] 上面的视图只覆盖
                # 当前 read_gen 半区;双代下 writer 物理写目标是写代半区
                # sub-slot,其 pos 残留(旧请求)不被清 → 新请求首次 flush 的
                # sink 区 old_pos==t==new pos → skip_unchanged 跳拷 → 读到旧
                # 请求的 sink KV(跨请求静默泄漏)。修=释放时清该 slot 的全部
                # gen 半区(gen_stride 由 arena shape 自洽推导;单代路径
                # gen_count==1 不进此分支,行为逐位不变)。
                _gen_count = compact_gen_count()
                _arena_pos = getattr(state, "compact_arena_pos", None)
                _stride = int(getattr(state, "compact_stride_tokens", 0) or 0)
                if _gen_count > 1 and _arena_pos is not None and _stride > 0:
                    _gen_stride = int(_arena_pos.shape[1]) // int(_gen_count)
                    for _gen in range(int(_gen_count)):
                        _off = compact_slot_offset_tokens(
                            slot=int(slot),
                            stride_tokens=_stride,
                            read_gen=int(_gen),
                            gen_stride_tokens=_gen_stride,
                        )
                        _arena_pos.narrow(1, _off, _stride).fill_(-1)
            # F10: 清零 compact 元数据，防止新请求复用 slot 时继承残留值
            if slot < len(state.compact_capacity):
                state.compact_capacity[slot] = 0
            if slot < len(state.compact_offset_tokens):
                state.compact_offset_tokens[slot] = 0
            if slot < len(state.compact_pad_zeroed_len):
                state.compact_pad_zeroed_len[slot] = -1
            # Task 10: cleanup 必须撤销 slot 绑定的 selected truth，避免同一步 slot 复用
            # 继承旧请求的 selected middle / refresh generation。
            # [F2] 调用点先查门控：trace 关闭时省掉 sorted+str 列表与 dict 构造。
            if _selected_ready_trace_path():
                _append_selected_ready_trace(
                    {
                        "event": "revoke_slot_selected_truth",
                        "request_id": str(rid),
                        "slot": int(slot),
                        "layer_index": int(getattr(state, "layer_index", -1)),
                        "finished_ids": [str(v) for v in sorted(finished_ids)],
                    }
                )
            _revoke_slot_selected_truth(state, slot=int(slot))
        if changed:
            # Task 10: cleanup 之后，旧请求的 selected/launch truth 不能继续参与后续 decode route。
            _invalidate_page_sparse_step_cache_truth(state)
            state.bump_compact_meta_epoch()
            # finished slot 会把 CPU slot->row mirror 置为 -1；必须失效缓存键，
            # 否则同 key 复用时 _maybe_update_slot_rows 可能跳过重建，遗留旧行号。
            state._slot_row_map_key = None
            state._slot_row_map_epoch = -1
            prev_active = state.last_active_request_ids
            if prev_active is not None:
                active_ids = tuple(rid for rid in prev_active if rid in state.request_id_to_slot)
            else:
                active_ids = tuple(
                    rid
                    for rid in state.batch_request_ids
                    if not _is_free_slot_id(rid)
                )
            state.last_active_request_ids = active_ids
            _auth_hint = controller.step_authority
            epoch_hint = _auth_hint.epoch if _auth_hint is not None else -1
            state._refresh_slot_signature(
                active_request_ids=active_ids,
                epoch=epoch_hint,
            )
        return


_METADATA_BUILDER_IMPLS = None


_RUNTIME_WORKER_DEPS_BOUND = False

# Symbols still injected via require_runtime_dep() to workers.
# Workers directly import types/utils/constants from their source modules now.
# Only symbols genuinely defined in this module (or re-exported here) remain.
_RUNTIME_WORKER_DEP_NAMES: Tuple[str, ...] = (
    # --- main-module internal functions ---
    "_build_layer_step_cache",
    "_cleanup_inactive_slots",
    "_normalize_capture_layout_views",
    "_prepare_prefill_capture_payload",
    "_prepare_refresh_capture_payload",
)

def _bind_runtime_worker_deps() -> None:
    global _RUNTIME_WORKER_DEPS_BOUND
    if _RUNTIME_WORKER_DEPS_BOUND:
        return
    missing = [name for name in _RUNTIME_WORKER_DEP_NAMES if name not in globals()]
    if missing:
        raise RuntimeError(
            "runtime worker dependency binding failed; missing symbols: "
            + ", ".join(sorted(missing))
        )
    from patches.runtime_deps import bind_decode_deps, bind_refresh_deps, bind_selector_deps

    bind_decode_deps({name: globals()[name] for name in _RUNTIME_WORKER_DEP_NAMES})
    bind_refresh_deps({name: globals()[name] for name in _RUNTIME_WORKER_DEP_NAMES})
    bind_selector_deps({name: globals()[name] for name in _RUNTIME_WORKER_DEP_NAMES})
    _RUNTIME_WORKER_DEPS_BOUND = True




_PAYLOAD_BUILDER_IMPLS = None


def _load_payload_builder_impls():
    global _PAYLOAD_BUILDER_IMPLS
    if _PAYLOAD_BUILDER_IMPLS is None:
        _bind_runtime_worker_deps()
        from patches.refresh_runtime.payload_worker import (
            prepare_prefill_capture_payload_impl,
            prepare_refresh_capture_payload_impl,
        )

        _PAYLOAD_BUILDER_IMPLS = (
            prepare_prefill_capture_payload_impl,
            prepare_refresh_capture_payload_impl,
        )
    return _PAYLOAD_BUILDER_IMPLS


_STEP_DECODE_DATA_IMPL = None


def _load_step_decode_data_impl():
    global _STEP_DECODE_DATA_IMPL
    if _STEP_DECODE_DATA_IMPL is None:
        _bind_runtime_worker_deps()
        from patches.decode_runtime.step_decode_data_worker import build_step_decode_data_impl

        _STEP_DECODE_DATA_IMPL = build_step_decode_data_impl
    return _STEP_DECODE_DATA_IMPL


_STEP_CONTEXT_IMPL = None


def _load_step_context_impl():
    global _STEP_CONTEXT_IMPL
    if _STEP_CONTEXT_IMPL is None:
        _bind_runtime_worker_deps()
        from patches.decode_runtime.step_context_worker import prepare_step_context_impl

        _STEP_CONTEXT_IMPL = prepare_step_context_impl
    return _STEP_CONTEXT_IMPL


_POST_KERNEL_IMPLS = None


def _load_post_kernel_impls():
    global _POST_KERNEL_IMPLS
    if _POST_KERNEL_IMPLS is None:
        _bind_runtime_worker_deps()
        from patches.refresh_runtime.post_kernel_worker import (
            build_layer_step_cache_impl,
        )

        _POST_KERNEL_IMPLS = build_layer_step_cache_impl
    return _POST_KERNEL_IMPLS


_CAPTURE_LAYOUT_IMPL = None


def _load_capture_layout_impl():
    global _CAPTURE_LAYOUT_IMPL
    if _CAPTURE_LAYOUT_IMPL is None:
        _bind_runtime_worker_deps()
        from patches.refresh_runtime.capture_layout_worker import get_step_capture_layout_impl

        _CAPTURE_LAYOUT_IMPL = get_step_capture_layout_impl
    return _CAPTURE_LAYOUT_IMPL


_SELECTOR_SELECTION_IMPLS = None


def _load_selector_selection_impls():
    global _SELECTOR_SELECTION_IMPLS
    if _SELECTOR_SELECTION_IMPLS is None:
        _bind_runtime_worker_deps()
        from patches.selector_runtime.selection_worker import (
            rebuild_compact_slots_batched_layers_from_selection_impl,
            compute_alpha_selection_pipeline_unified_impl,
        )

        _SELECTOR_SELECTION_IMPLS = (
            rebuild_compact_slots_batched_layers_from_selection_impl,
            compute_alpha_selection_pipeline_unified_impl,
        )
    return _SELECTOR_SELECTION_IMPLS


def _load_metadata_builder_impls():
    global _METADATA_BUILDER_IMPLS
    if _METADATA_BUILDER_IMPLS is None:
        _bind_runtime_worker_deps()
        from patches.decode_runtime.metadata_builder import (
            build_step_bound_meta_from_metadata_impl,
            maybe_build_step_decode_data_from_metadata_impl,
            maybe_build_step_prefill_global_meta_from_metadata_impl,
        )

        _METADATA_BUILDER_IMPLS = (
            maybe_build_step_decode_data_from_metadata_impl,
            maybe_build_step_prefill_global_meta_from_metadata_impl,
            build_step_bound_meta_from_metadata_impl,
        )
    return _METADATA_BUILDER_IMPLS







# -----------------------------------------------------------------------------
# Debug helpers
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Debug helpers (runtime-safe for torch.compile/cudagraph)
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# 两阶段缓存：per-layer step cache 构建
# -----------------------------------------------------------------------------

def _build_layer_step_cache(
    state: LayerState,
    step_meta: "StepMeta",
    step_authority: "StepAuthority",
    block_table: Optional[torch.Tensor],
    device: torch.device,
    force_dense: bool = False,
    force_compact_off: bool = False,
    *,
    skip_meta_pack: bool = False,
    layer_effective_refresh_by_row: Tuple[bool, ...],
    step_bound_meta: Optional["StepBoundMeta"] = None,
    precomputed_cache_key: Optional[Tuple[object, ...]] = None,
) -> None:
    # M5 Part B2 (2026-04-24): plumb step_bound_meta to the impl so the
    # per-layer bail-out can consult CompactRecentLaunchPlan.valid.
    # P12 lever-2 (2026-06-12): plumb precomputed_cache_key the same way.
    build_impl = _load_post_kernel_impls()
    return build_impl(
        state=state,
        step_meta=step_meta,
        step_authority=step_authority,
        block_table=block_table,
        device=device,
        force_dense=force_dense,
        force_compact_off=force_compact_off,
        skip_meta_pack=skip_meta_pack,
        layer_effective_refresh_by_row=layer_effective_refresh_by_row,
        step_bound_meta=step_bound_meta,
        precomputed_cache_key=precomputed_cache_key,
    )
# -----------------------------------------------------------------------------
# Controller keyed by cache pointer
# -----------------------------------------------------------------------------

class VLLMSparseController(
    ProfileMixin,
    CaptureRingMixin,
    CompactKVMixin,
    WaitDeciderMixin,
    RefreshRebuildMixin,
    SelectorComputeMixin,
):
    # 默认 max_batch_size 用于预分配 buffer，避免热路径 slice
    # 设为 32 平衡内存开销和常见场景覆盖（会在运行时动态扩展）
    DEFAULT_MAX_BATCH_SIZE: int = 32
    STEP_HANDLE_RING_SIZE: int = 64

    def __init__(self, config: SparseControllerConfig) -> None:
        self.config = config
        interval_merge_policy = str(
            getattr(config, "interval_merge_policy", "delta1") or "delta1"
        ).strip().lower()
        if interval_merge_policy not in {"delta1", "off"}:
            raise ValueError(
                "interval_merge_policy must be 'delta1' or 'off', "
                f"got {interval_merge_policy!r}"
            )
        self._interval_merge_policy: str = interval_merge_policy
        self._workload_plan_replay = load_workload_plan_from_env()
        self._workload_plan_replay_events_by_key = index_workload_plan_events(
            self._workload_plan_replay
        )
        self._workload_plan_replay_injected_count: int = 0
        # -- validate mixin dependency contracts --
        mro_names = {cls.__name__ for cls in type(self).__mro__}
        for cls in type(self).__mro__:
            for req in getattr(cls, "_MIXIN_REQUIRES", ()):
                if req not in mro_names:
                    raise TypeError(
                        f"{cls.__name__} requires {req} in MRO but it is missing"
                    )
        # -- mixin state initialization --
        self._init_profile_state()
        self._init_capture_ring_state()
        self._init_compact_kv_state()
        self._init_wait_decider_state()
        self._init_refresh_rebuild_state()
        self._init_selector_compute_state()
        self._allocator_backend: AllocatorBackend = resolve_allocator_backend()
        self.layer_states: Dict[int, LayerState] = {}
        # cache_key -> layer_index，用于 StepDecodeData 的 list 索引
        self.layer_index_by_cache_key: Dict[int, int] = {}
        self.layer_cache_keys: List[int] = []
        # layer_index 缓存 epoch（仅在全量重建映射时递增）
        self._layer_index_cache_epoch: int = 0

        self.base_cache_key: Optional[int] = None
        self.base_last_seq_len: int = -1
        self._global_slot_allocator: GlobalSlotAllocator = GlobalSlotAllocator(
            capacity=self._global_slot_allocator_capacity()
        )
        self._pending_global_slot_releases: Set[str] = set()
        self._step_global_slot_map_epoch: int = -1
        self._step_global_slot_map_req_ids: Tuple[str, ...] = tuple()
        self._step_global_slot_map: Dict[str, int] = {}
        self.request_states: Dict[str, RequestTracking] = {}
        self._request_intent_tickets: Dict[str, RequestIntentTicket] = {}
        self.tp_size: int = 1
        # TP size at init time, read from env for EngineCore-safe guard.
        # Unlike self.tp_size (which Workers overwrite at each step), _init_tp_size
        # is set once and never changes — safe for EngineCore's _patched_append.
        _env_tp = os.environ.get("VLLM_TENSOR_PARALLEL_SIZE", "")
        self._init_tp_size: int = int(_env_tp) if _env_tp.isdigit() and int(_env_tp) > 0 else 1
        self.step_context: Optional[StepContext] = None
        self.step_exec_hints: Optional[StepExecHints] = None
        self.step_context_epoch: int = 0
        self._current_decode_step_by_req_epoch: int = -1
        self._current_decode_step_by_req: Dict[str, int] = {}
        # kernel_dispatch step-local caches（避免热路径重复构建）
        self._step_dispatcher_slot_row_map_token: int = -1
        self._step_dispatcher_slot_row_map_key: Tuple[int, ...] = tuple()
        self._step_dispatcher_slot_row_map: Optional[Dict[int, int]] = None
        self._step_dispatcher_refresh_dirty_token: int = -1
        self._step_dispatcher_refresh_dirty_slot_key: Tuple[int, ...] = tuple()
        self._step_dispatcher_refresh_dirty_payload_slots: Tuple[int, ...] = tuple()
        self._step_dispatcher_refresh_dirty_any_decode: bool = False
        self._fa3_live_route_token: int = -1
        self._fa3_live_route_batch_size: int = -1
        self._fa3_live_route_has_selected_consume: bool = False
        self._fa3_live_route_has_capture: bool = False
        self._fa3_live_route: Optional[str] = None
        # step-local local-pack compact 缓存（按 step identity token + layer + compact meta 绑定）
        self._step_local_pack_compact_token: int = -1
        self._step_local_pack_compact_layer_index: int = -1
        self._step_local_pack_compact_meta_epoch: int = -1
        self._step_local_pack_compact_kv_len_by_row: Tuple[int, ...] = tuple()
        self._step_local_pack_compact_offsets_by_row: Tuple[int, ...] = tuple()
        self._step_local_pack_compact_kv_len_max: int = 0
        self._step_handle_ring_size: int = int(self.STEP_HANDLE_RING_SIZE)
        self._step_handle_next_id: int = 0
        self._step_handle_by_slot: List[Optional[StepHandle]] = [
            None for _ in range(self._step_handle_ring_size)
        ]
        self._step_context_by_handle_slot: List[Optional[StepContext]] = [
            None for _ in range(self._step_handle_ring_size)
        ]
        self._step_handle_generation_by_slot: List[int] = [
            0 for _ in range(self._step_handle_ring_size)
        ]
        self._current_step_handle_id: int = -1
        self._current_step_handle_generation: int = -1
        # step 级语义快照：在 prepare_step_context 开始处构建，后续热路径只读。
        self._step_semantic_snapshot: Optional[StepSemanticSnapshot] = None
        self._step_authority_builder_scratch = None
        # decode log_f per-row state cache（用于 step 边界 dirty-row 计算）。
        self._prev_decode_logf_row_state: Optional[Tuple[Tuple[int, int, int], ...]] = None
        # vLLM V1 某些配置下可能在同一个 ctrl_step 内重复调用一次 _prepare_inputs。
        # 为避免重复递增 step_context_epoch / 重复计划 refresh / 无意义重建缓存，这里记录最近一次 prepare_step_context 的签名用于去重。
        # StepIdentity: (target_epoch, source_signature, scheduler_token, refresh_nonce_key)
        # 作为 prepare_step_context 的复用主键；字段均为 int，比较开销极低。
        self._prepared_step_identity: Tuple[int, int, int, int] = (-1, -1, -1, -1)
        self._prepared_num_actual_tokens: int = -1
        # sentence trigger 可能在同一 ctrl_step 内更新 refresh 状态：用 nonce 避免误复用 StepContext。
        self._prepared_refresh_nonce: int = -1
        # TP>1 worker token source（step 级单源）：由 _prepare_inputs 每步绑定。
        self._worker_token_ids_cpu = None
        self._worker_batch_id_to_idx = None
        # [TP-ASYNC-HARVEST] async 调度下搭车收割的采样 token 副本 FIFO
        # （(cpu_copy, ready_event, prev_req_id_to_index) 三元组；跨步存活，
        # 由 _prepare_inputs 的修复通道排空）。非 None 同时充当 async 模式哨兵。
        self._async_sampled_stash = None
        # 轻量一致性哨兵：不做 tensor clone，检测 source 行数是否在本步内被意外改写。
        # （version/data_ptr 两支已退休：numpy 源无这两个属性，检查自设计起恒短路。）
        self._worker_token_source_rows: int = -1
        self.step_meta: Optional[StepMeta] = None
        self.step_authority: Optional["StepAuthority"] = None
        self.device: Optional[torch.device] = None  # 在首次调用 dispatcher 时设置
        # pending rebuild latest-wins：每个 req 仅保留最新 pending_id
        # prefill buffer release tracking (decode-only steps)
        self._prefill_release_pending_epoch: int = -1
        self._prefill_release_waited_mask: int = 0
        # 记录最近一次 prefill enqueue 的 epoch（用于回退释放判据）
        self._prefill_last_enqueue_epoch: int = -1
        # 最近一次完成 prefill release 的 epoch（用于避免重复 release 触发）
        self._prefill_release_done_epoch: int = -1
        # refresh layer-group gating（默认关闭）
        self._refresh_layer_group_event_idx: int = 0
        self._refresh_layer_group_active: int = 0
        self._refresh_layer_group_enabled: bool = False
        self._refresh_layer_group_epoch: int = -1
        self._refresh_layer_group_any_refresh: bool = False
        # step-level profiling counters (optional)
        # per-step cached slot->request_id snapshot (avoid per-layer tuple rebuild)
        self._step_refresh_slot_req_ids_epoch: int = -1
        self._step_refresh_slot_req_ids_handle_id: int = -1
        self._step_refresh_slot_req_ids_handle_generation: int = -1
        self._step_refresh_slot_req_ids_cache: Dict[
            Tuple[int, int, int, Tuple[int, ...]],
            Tuple[str, ...],
        ] = {}
        # refresh CPU-side cache (per step): slot_list -> slot_tensor_cpu/seq_lens_tensor_cpu
        # capture_row_by_batch_row cache: (epoch, ptr, device, rows_tuple) -> capture_rows_i64
        # per-step refresh 决策缓存：同一步内各层的决策一致，避免每层重复扫描 request_states
        self._should_refresh_cache_step: int = -1
        self._should_refresh_cache_nonce: int = -1
        self._should_refresh_cache: Dict[Tuple[Tuple[str, ...], str], StepRefreshPlan] = {}
        # 预分配 buffer 的最大 batch 大小（避免热路径 slice）
        # 会在运行时动态扩展（如果实际 batch_size 超过此值）
        self.max_batch_size: int = self.DEFAULT_MAX_BATCH_SIZE
        # KV cache 规格（从 TritonAttentionMetadataBuilder 提取）
        self.kv_cache_block_size: Optional[int] = None
        self.kv_cache_num_kv_heads: Optional[int] = None
        self.kv_cache_head_dim: Optional[int] = None
        self.kv_cache_dtype: Optional[torch.dtype] = None
        self._step_decode_spec_key: Optional[Tuple[int, int, int, torch.dtype]] = None

        # ============ Step 级 ordered layer reuse ============
        # 每 step 构建一次，包含所有层的预构建数据
        # 使 dispatcher 入口只需"读缓存 + 调 kernel"
        self.step_decode_data: Optional[StepDecodeData] = None
        # step 级稳定 cache key（用于复用 StepDecodeData/Plan）
        self.step_decode_cache_key: Optional[Tuple[object, ...]] = None
        self.step_decode_plan_version: int = -1
        # graph-stable ResolvedRowPtr arena. replay/source intentionally alias:
        # the production writer updates the tensor addresses captured by CUDA graph.
        self._resolved_row_ptr_replay_arena: Optional[object] = None
        self._resolved_row_ptr_source_arena: Optional[object] = None
        self._resolved_row_ptr_arena_key: Optional[Tuple[object, ...]] = None
        # step 级调度计划（用于 per-layer 入口零逻辑）
        self.step_dispatch_plan: Optional[StepDispatchPlan] = None
        self._runtime_state: StepRuntimeState = StepRuntimeState()

        # ordered layer reuse 预绑定数据（避免 per-layer 查表）
        self.ordered_layer_data_list: Optional[List[LayerDecodeData]] = None
        self.ordered_step_decode_data: Optional[StepDecodeData] = None
        # ordered layer reuse 命中统计（按 step 记一次）
        self._ordered_reuse_hit_epoch: int = -1
        # ordered layer reuse miss 统计（按 step 记一次）
        self._ordered_reuse_miss_epoch: int = -1
        # layer dispatch cursor：按调用顺序计数，避免 layer_cache_keys 顺序不一致导致提前 flush
        self.layer_dispatch_cursor: int = 0
        self.layer_dispatch_epoch: int = -1
        self.layer_dispatch_layer_count: int = 0
        # step 级 prefill 捕获批次（按层聚合，减少 per-layer selector 开销）
        self.step_prefill_epoch: int = -1
        # chunk-batched：按 buf_id/slot_in_chunk 入队，flush 时只处理当前 chunk。
        # 热路径避免 dict/hash：固定长度 list[CHUNK]，None 表示空槽位。
        # prefill 也复用 SelectorBatchPayload：chunk flush 时直接 selector+rebuild（支持 refresh_stream 异步）
        self.step_prefill_chunk_payloads: List[List[Optional[SelectorBatchPayload]]] = [
            [None for _ in range(int(_CAPTURE_CHUNK))] for _ in range(int(_CAPTURE_IN_FLIGHT))
        ]
        # per-buf payload 存在性 bitmask（低位=slot_in_chunk），避免 flush 前 any(...) 扫描
        self.step_prefill_chunk_mask: List[int] = [0 for _ in range(int(_CAPTURE_IN_FLIGHT))]
        # step 级 prefill 计划（按 req_id），全层共享
        # step 级 prefill capture last_n（按 row 顺序），避免 per-layer 反复 dict 查找
        # prefill subset 索引缓存（CPU pinned，小集合 LRU）
        # step 级 refresh 捕获批次（按层聚合）
        self.step_refresh_epoch: int = -1
        # chunk-batched：按 buf_id/slot_in_chunk 入队，flush 时只处理当前 chunk。
        # 热路径避免 dict/hash：固定长度 list[CHUNK]，None 表示空槽位。
        self.step_refresh_chunk_payloads: List[List[Optional[SelectorBatchPayload]]] = [
            [None for _ in range(int(_CAPTURE_CHUNK))] for _ in range(int(_CAPTURE_IN_FLIGHT))
        ]
        # per-buf payload 存在性 bitmask（低位=slot_in_chunk），避免 flush 前 any(...) 扫描
        self.step_refresh_chunk_mask: List[int] = [0 for _ in range(int(_CAPTURE_IN_FLIGHT))]
        # step 级 capture layout ring（prefill/refresh），每个 buf 为 [CHUNK,...]
        self.step_prefill_capture_layout_ring: List[Optional[StepCaptureLayout]] = [
            None for _ in range(_CAPTURE_IN_FLIGHT)
        ]
        self.step_refresh_capture_layout_ring: List[Optional[StepCaptureLayout]] = [
            None for _ in range(_CAPTURE_IN_FLIGHT)
        ]
        # per-step meta packing（跨层一次性）
        self.step_decode_req_meta_i32_all: Optional[torch.Tensor] = None
        self.step_decode_req_meta_i64_all: Optional[torch.Tensor] = None
        self.step_decode_is_compact_all: Optional[torch.Tensor] = None
        self.step_decode_compact_kv_len_all: Optional[torch.Tensor] = None
        self.step_decode_compact_offset_all: Optional[torch.Tensor] = None
        self.step_bound_meta: Optional[StepBoundMeta] = None
        # StepBoundMeta 轻量一致性探针（每 step 仅校验一次）。
        self._step_bound_meta_probe_checked_epoch: int = -1
        self._step_bound_meta_probe_checked_handle_id: int = -1
        self._step_bound_meta_probe_checked_handle_generation: int = -1
        self._step_bound_meta_probe_warned_missing_authority: bool = False
        # RefreshStepBundle 验证缓存（每 step 仅校验一次）。
        self._refresh_bundle_validated_token: int = -1
        # decode log_f / logits stage cache: 必须绑定 step identity，避免同 epoch 重入误复用。
        self._step_logits_ready_token: int = -1
        self._step_logits_ready_input_signature: Optional[Tuple[object, ...]] = None
        self._step_logits_ready_bound_signature: Optional[Tuple[object, ...]] = None
        self._decode_logf_stage_token: int = -1
        self._decode_logf_stage_signature: Optional[Tuple[object, ...]] = None
        self._decode_logf_stage_bound_signature: Optional[Tuple[object, ...]] = None
        self._step_cql_epoch: int = -1
        self._step_cql_handle_id: int = -1
        self._step_cql_handle_generation: int = -1
        self._step_cql_tensor: Optional[torch.Tensor] = None
        # selector 复用缓冲（避免每步大分配）
        # key_norms delta 复用缓冲（减少 per-layer 小分配与 HtoD 抖动）
        # log_f(last_n>1) workspace（scratch + denom，step-wise 复用以避免频繁分配）
        # refresh_stream 异步流水线（chunk-batched）
        self.refresh_stream: Optional[torch.cuda.Stream] = None
        self._refresh_stream_device: Optional[torch.device] = None
        self.chunk_ready_evt: List[torch.cuda.Event] = []
        self.chunk_done_evt: List[torch.cuda.Event] = []
        # 事件拆分（性能优化/解耦）：prefill 与 refresh 可分别打点 done。
        # - chunk_done_evt 仍代表“该 buf 的异步流水线全部完成”（默认等待点，保持正确性）。
        # - prefill_done_evt / refresh_done_evt 为更细粒度的完成事件，便于后续进一步减少不必要的等待/串行化。
        self.prefill_done_evt: List[torch.cuda.Event] = []
        self.refresh_done_evt: List[torch.cuda.Event] = []
        # split-wait：每个 in-flight buf 的“待完成工作类型”标记（用于选择等待事件）。
        # bit0=prefill(selector+rebuild), bit1=refresh(selector+rebuild)
        # split-wait：记录该 buf 最近一次提交工作的 epoch（防止 flags 漏置导致过早复用）
        # wait 策略：chunk（等 chunk_done）| split（按 flags 等 prefill/refresh done）
        # main_stream 等待去重：同一 step 内每个 buf_id 只在“新 chunk”首次触发 wait_event
        # hints 驱动的 wait 去重：同一 step 同一 buf 只触发一次 wait。
        # refresh profile：按 buf 记录“上一次 flush 的 pending 事件”，在下一次 wait_event 后写日志。
        # refresh profile（detail）仅在“本次 flush 采样”期间开启，避免对非采样 step 引入额外 event/launch。
        # bootstrap（prefill selector+compact build）完成确认：基于 main_stream 对 chunk_done_evt 的 wait_event
        # vLLM scheduler 提供的“已完成请求”集合（用于跨层清理 slot/释放 compact）
        # 语义：SchedulerOutput.finished_req_ids（上一步 -> 本步之间完成的请求）
        # cleanup ledger（仅清理路径消费；与 snapshot ledger 解耦）
        self._finished_req_ids_step: Set[str] = set()
        # snapshot ledger（仅 snapshot builder 消费）
        self._snapshot_finished_req_ids: Set[str] = set()
        # finished 事件代际：仅在新增 finished request 时递增。
        # prepare_inputs 可用它作为 active compaction cache 的失效源，避免每步重算。
        self._finished_generation: int = 0
        # worker 边界 finished 消费去重：记录最近一次已消费的 step token。
        self._finished_boundary_step_token: Optional[object] = None
        # step 快照（单写者：step 边界构建，热路径只读）
        self._active_step_snapshot: Optional[ActiveStepSnapshot] = None
        self._active_step_snapshot_epoch: int = -1
        self._active_step_ticket: Optional[StepTicket] = None
        self._active_step_source_signature: int = -1
        # selector layer index tensor cache（避免每步构造）
        # compact gather 复用缓冲（避免频繁分配 src/dst/pos/lengths）
        # positions cache（避免 refresh 高频构造 torch.arange(kv_len)）
        # window_idx / tail_offsets 缓存（selector preproc 高频使用的常量 tensor）
        # row_index cache：避免在 refresh/prefill patch 路径频繁构造 torch.tensor([rows], device=cuda)
        # rebuild gather 指针数组缓存：避免每次 rebuild 构造/逐元素写入 GPU int64 tensor
        # refresh/log_f meta patch 的 step-epoch 缓存：复用 last_n/offset/cap 这类“跨层不变”的小张量
        # decode step-wise log_f mask（用于 pack_req_meta_decode_fast_layers 的动态 override）
        # decode：跨层 pack 时用于 out_ptr(meta64[2]) 计算的 per-layer 映射（只依赖 num_layers）
        self._decode_buf_id_by_layer_i32: Optional[torch.Tensor] = None
        self._decode_slot_in_chunk_by_layer_i32: Optional[torch.Tensor] = None
        self._decode_layer_logf_enable_i32: Optional[torch.Tensor] = None
        self._decode_layer_map_num_layers: int = 0
        # decode：无 refresh 时给 pack 传入的 dummy 映射，避免每步分配小张量
        self._decode_dummy_capture_row_by_batch_row_i32: Optional[torch.Tensor] = None
        self._decode_dummy_capture_row_device: Optional[torch.device] = None
        self._decode_dummy_capture_row_cap: int = 0

        # prefill：跨层 req_meta（一次 pack）
        self.step_prefill_req_meta_i32_all: Optional[torch.Tensor] = None  # [layers, max_batch, 7]
        self.step_prefill_req_meta_i64_all: Optional[torch.Tensor] = None  # [layers, max_batch, 4]
        self.prefill_global_meta_epoch: int = -1
        self.prefill_global_meta_handle_id: int = -1
        self.prefill_global_meta_handle_generation: int = -1
        self._prefill_last_n_i32: Optional[torch.Tensor] = None
        self._prefill_cap_i32: Optional[torch.Tensor] = None
        self._prefill_i32_epoch: int = -1
        self._prefill_i32_handle_id: int = -1
        self._prefill_i32_handle_generation: int = -1
        self._prefill_log_f_stride_head: int = 0
        self._prefill_log_f_stride_epoch: int = -1
        self._prefill_log_f_stride_handle_id: int = -1
        self._prefill_log_f_stride_handle_generation: int = -1
        # 262k OOM fix: cap the arena kv_max at the model bound and raise the
        # default budget so the profile-time prebuild (_prebuild_capture_buffers)
        # can reserve the max_model_len(262144)-bucket window=1 arena past the
        # budget gate (prefill_capture_meta_arena.py:519-522). kv_max_cap_bucket
        # starts 0 (no cap == HEAD behaviour for short ctx) and is set
        # AUTHORITATIVELY at profile time by the prebuild (controller __init__
        # runs before profile_run, so max_model_len is not yet known here). The
        # cap is consumed at prefill_capture_meta_arena.py:447-449 -> a 262k
        # prefill builds kv_max = _align_up_int(max_model_len, kv_min) = 262144
        # -> arena head-stride == the EDIT-1 bucketed planned_max_capture_k ->
        # the DEFER scratch last dim collapses to ONE value (byte-match).
        self.prefill_capture_meta_arena: SparseCaptureMetaArena = SparseCaptureMetaArena(
            budget_bytes=int(
                os.environ.get(
                    "VLLM_SPARSE_CAPTURE_ARENA_BUDGET_BYTES",
                    ARENA_PHASE1_QWEN06_BS2_CAP262K_BUDGET_BYTES,
                )
            ),
            kv_max_cap_bucket=int(
                os.environ.get("VLLM_SPARSE_CAPTURE_ARENA_KV_MAX_CAP", "0") or "0"
            ),
        )
        self._prefill_capture_arena_intent: CaptureArenaIntent = (
            CaptureArenaIntent.ONE_SHOT_BOOTSTRAP
        )
        self._prefill_capture_meta_arena_enabled: bool = (
            os.getenv("VLLM_SPARSE_CAPTURE_META_ARENA", "1") != "0"
        )
        self._last_seen_device: Optional[torch.device] = None

        # async trace（默认关闭）：用于定位“异步更慢/无法 hide”的根因（等待是否频繁、flush 是否切碎等）。


    # ------------------------------------------------------------------
    # Prefill capture/meta arena helpers
    # ------------------------------------------------------------------

    def reset_prefill_capture_arena_step_metrics(self) -> None:
        if not bool(getattr(self, "_prefill_capture_meta_arena_enabled", False)):
            return
        self.prefill_capture_meta_arena.reset_step_metrics()

    def get_prefill_capture_arena_metrics(self) -> dict[str, object]:
        return self.prefill_capture_meta_arena.metrics.as_event_fields()

    def reserve_prefill_capture_bucket(
        self,
        *,
        step_context: StepContext,
        step_authority: StepAuthority,
        capture_intent: CaptureArenaIntent,
        slot_by_row: Tuple[int, ...],
    ) -> object:
        if not bool(getattr(self, "_prefill_capture_meta_arena_enabled", False)):
            return self.prefill_capture_meta_arena.missing_reservation(
                step_epoch=int(step_context.epoch),
                step_handle_id=int(getattr(step_context, "step_handle_id", -1)),
                step_handle_generation=int(
                    getattr(step_context, "step_handle_generation", -1)
                ),
                intent=capture_intent,
                reason="arena_disabled",
                count_miss=False,
            )
        configured = (
            int(getattr(self.config, "prefill_last_n_query", 0) or 0)
            if self.config is not None
            else 0
        )
        if configured <= 0:
            return self.prefill_capture_meta_arena.missing_reservation(
                step_epoch=int(step_context.epoch),
                step_handle_id=int(getattr(step_context, "step_handle_id", -1)),
                step_handle_generation=int(
                    getattr(step_context, "step_handle_generation", -1)
                ),
                intent=capture_intent,
                reason="prefill_capture_disabled",
                count_miss=False,
            )
        first_state = next(iter(self.layer_states.values()), None)
        if first_state is None:
            return self.prefill_capture_meta_arena.missing_reservation(
                step_epoch=int(step_context.epoch),
                step_handle_id=int(getattr(step_context, "step_handle_id", -1)),
                step_handle_generation=int(
                    getattr(step_context, "step_handle_generation", -1)
                ),
                intent=capture_intent,
                reason="num_heads_unavailable",
            )
        num_heads = int(getattr(first_state, "num_heads", 0) or 0)
        device = getattr(first_state, "device", None) or getattr(
            self, "_last_seen_device", None
        )
        if num_heads <= 0 or device is None:
            return self.prefill_capture_meta_arena.missing_reservation(
                step_epoch=int(step_context.epoch),
                step_handle_id=int(getattr(step_context, "step_handle_id", -1)),
                step_handle_generation=int(
                    getattr(step_context, "step_handle_generation", -1)
                ),
                intent=capture_intent,
                reason="arena_device_unavailable",
            )

        capture_plan_by_req, _finalize_req_ids = self.get_step_prefill_plan_by_req(
            step_context=step_context
        )
        active_capture_by_req = {
            str(rid): int(last_n)
            for rid, last_n in (capture_plan_by_req or {}).items()
            if int(last_n or 0) > 0
        }
        lookahead_only = not active_capture_by_req
        compact_threshold = self._compact_threshold_tokens() if lookahead_only else 0
        candidates: list[tuple[int, int]] = []
        kv_needed = 0
        is_prefill_by_row = tuple(bool(v) for v in step_authority.is_prefill_by_row)
        for row, rid in enumerate(step_authority.req_ids[: step_authority.batch_size]):
            if row >= len(is_prefill_by_row) or not bool(is_prefill_by_row[row]):
                continue
            if not lookahead_only and int(active_capture_by_req.get(str(rid), 0) or 0) <= 0:
                continue
            tracking = self.request_states.get(str(rid))
            if tracking is None or bool(getattr(tracking, "bootstrap_done", False)):
                continue
            used_tokens = (
                int(step_authority.context_kv_len_by_row[row])
                if row < len(step_authority.context_kv_len_by_row)
                else 0
            )
            used_tokens = max(0, int(used_tokens))
            if used_tokens <= 0:
                continue
            if lookahead_only:
                total_req = int(getattr(tracking, "total_prompt_tokens", 0) or 0)
                if (
                    total_req <= 0
                    and step_context.prompt_lens is not None
                    and row < len(step_context.prompt_lens)
                ):
                    total_req = int(step_context.prompt_lens[row])
                used_tokens = max(int(total_req), int(used_tokens))
                if compact_threshold > 0 and used_tokens <= int(compact_threshold):
                    continue
            slot = int(slot_by_row[row]) if row < len(slot_by_row) else -1
            if slot < 0:
                continue
            candidates.append((int(slot), int(row)))
            kv_needed = max(int(kv_needed), int(used_tokens))

        if not candidates:
            return self.prefill_capture_meta_arena.missing_reservation(
                step_epoch=int(step_context.epoch),
                step_handle_id=int(getattr(step_context, "step_handle_id", -1)),
                step_handle_generation=int(
                    getattr(step_context, "step_handle_generation", -1)
                ),
                intent=capture_intent,
                reason="no_prefill_capture_lookahead_candidate",
                count_miss=False,
            )

        row_by_slot: dict[int, int] = {}
        for slot, row in candidates:
            row_by_slot.setdefault(int(slot), int(row))
        pairs_by_slot = tuple((slot, row_by_slot[slot]) for slot in sorted(row_by_slot))
        _reservation = self.prefill_capture_meta_arena.reserve_prefill_layouts(
            step_epoch=int(step_context.epoch),
            step_handle_id=int(getattr(step_context, "step_handle_id", -1)),
            step_handle_generation=int(getattr(step_context, "step_handle_generation", -1)),
            intent=capture_intent,
            slot_list=tuple(int(slot) for slot, _row in pairs_by_slot),
            row_list=tuple(int(row) for _slot, row in pairs_by_slot),
            batch_size=int(step_authority.batch_size),
            num_heads=int(num_heads),
            kv_needed=max(1, int(kv_needed)),
            device=torch.device(device),
            seq_lens_by_row=tuple(int(v) for v in getattr(step_context, "seq_lens", tuple())),
            q_lens_by_row=tuple(int(v) for v in getattr(step_context, "q_lens", tuple())),
            context_kv_len_by_row=tuple(
                int(v) for v in step_authority.context_kv_len_by_row
            ),
            logits_capacity_by_row=tuple(
                int(v) for v in step_authority.context_kv_len_by_row
            ),
        )
        try:
            if getattr(_reservation, "status", None) is ArenaReservationStatus.READY:
                if int(getattr(_reservation, "alloc_bytes", 0) or 0) > 0:
                    # PRODUCER gated on a real grow: a new high-water buffer was built
                    # this step, so a smaller bucket may now be superseded. Prune scans
                    # only here (rare). On pure-reuse steps (alloc_bytes==0) NOTHING is
                    # newly superseded -> zero release work in steady state.
                    self._prune_superseded_arena_buckets()
                elif self._arena_retired_buckets:
                    # CONSUMER stays guarded-eager: drain buckets retired on a prior grow
                    # once their async release event fires (O(1) when the deque is empty).
                    self._reclaim_retired_arena_buckets()
        except Exception:
            _log.warning("arena grow-to-fit prune failed", exc_info=True)
        return _reservation

    # ------------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------------


    def _release_on_idle_enabled(self) -> bool:
        return bool(_RELEASE_ON_IDLE_CACHED)

    def release_idle_buffers(self) -> None:
        """释放 request 生命周期相关的缓存，允许 idle 时显存回落。"""
        # 尽量确保 refresh_stream 的异步写入已完成，避免释放仍在使用的 arena。
        if self.chunk_done_evt:
            pending = any(not evt.query() for evt in self.chunk_done_evt)
            if pending:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
        if self._pending_refresh_rebuilds:
            for pending in self._pending_refresh_rebuilds:
                # [SELECTED-OUT-RING] 整批丢弃绕过 clear 漏斗:逐个释放环槽。
                self._selected_out_ring_release_pending_slot(pending)
                req_ids = self._normalize_refresh_req_ids(pending.req_ids)
                if req_ids:
                    self._resolve_refresh_lease(
                        req_ids=req_ids,
                        reason="idle_pending_rebuild_release",
                    )
                elif pending.payloads:
                    raise RuntimeError("pending refresh rebuild missing req_ids during idle release")
            self._pending_refresh_rebuilds = deque()
        self._pending_refresh_rebuild_by_req.clear()
        self._pending_refresh_rebuild_id = 0

        # idle 时清空 request 相关状态（避免 finished_req_ids 缺失导致的粘性增长）
        self.request_states.clear()
        self._request_intent_tickets.clear()
        self._bootstrap_pending_request_ids.clear()
        # idle 周期边界：重置 global slot 相关状态，避免 slot 跨 idle 滞留。
        self._global_slot_allocator = GlobalSlotAllocator(
            capacity=self._global_slot_allocator_capacity()
        )
        self._pending_global_slot_releases = set()
        self._step_global_slot_map_epoch = -1
        self._step_global_slot_map_req_ids = tuple()
        self._step_global_slot_map = {}
        self._finished_req_ids_step = set()
        self._snapshot_finished_req_ids = set()
        self._finished_boundary_step_token = None
        self._active_step_snapshot = None
        self._active_step_snapshot_epoch = -1
        self._active_step_ticket = None
        self._active_step_source_signature = -1
        self._should_refresh_cache_step = -1
        self._should_refresh_cache_nonce = -1
        self._should_refresh_cache.clear()
        self._step_dispatcher_slot_row_map_token = -1
        self._step_dispatcher_slot_row_map_key = tuple()
        self._step_dispatcher_slot_row_map = None
        self._step_dispatcher_refresh_dirty_token = -1
        self._step_dispatcher_refresh_dirty_slot_key = tuple()
        self._step_dispatcher_refresh_dirty_payload_slots = tuple()
        self._step_dispatcher_refresh_dirty_any_decode = False
        self._fa3_live_route_token = -1
        self._fa3_live_route_batch_size = -1
        self._fa3_live_route_has_selected_consume = False
        self._fa3_live_route_has_capture = False
        self._fa3_live_route = None
        self._step_context_slot_row_map_token = -1
        self._step_context_slot_row_map_key = tuple()
        self._step_context_slot_row_map = None
        self._step_local_pack_compact_token = -1
        self._step_local_pack_compact_layer_index = -1
        self._step_local_pack_compact_meta_epoch = -1
        self._step_local_pack_compact_kv_len_by_row = tuple()
        self._step_local_pack_compact_offsets_by_row = tuple()
        self._step_local_pack_compact_kv_len_max = 0
        self._refresh_bundle_validated_token = -1
        self._step_refresh_slot_req_ids_epoch = -1
        self._step_refresh_slot_req_ids_handle_id = -1
        self._step_refresh_slot_req_ids_handle_generation = -1
        self._step_refresh_slot_req_ids_cache.clear()
        self._step_logits_ready_token = -1
        self._step_logits_ready_input_signature = None
        self._step_logits_ready_bound_signature = None
        self._decode_logf_stage_token = -1
        self._decode_logf_stage_signature = None
        self._decode_logf_stage_bound_signature = None
        self._step_cql_epoch = -1
        self._step_cql_handle_id = -1
        self._step_cql_handle_generation = -1
        self._step_cql_tensor = None
        self._step_refresh_commit_handle_id = -1
        self._step_refresh_commit_handle_generation = -1
        self._step_refresh_commit_written_handle_id = -1
        self._step_refresh_commit_written_handle_generation = -1
        if hasattr(self, "_step_refresh_commit_written_req_ids"):
            self._step_refresh_commit_written_req_ids.clear()
        self.step_exec_hints = None
        self._current_step_handle_id = -1
        self._current_step_handle_generation = -1
        for i in range(len(self._step_wait_consumed_token_by_buf)):
            self._step_wait_consumed_token_by_buf[i] = 0
        self._refresh_layer_group_event_idx = 0

        # 清空 capture ring 与 step buckets
        for buf_id in range(len(self.step_prefill_chunk_payloads)):
            _buf = self.step_prefill_chunk_payloads[buf_id]
            for _i in range(len(_buf)):
                _buf[_i] = None
            self.step_prefill_chunk_mask[buf_id] = 0
        for buf_id in range(len(self.step_refresh_chunk_payloads)):
            _buf = self.step_refresh_chunk_payloads[buf_id]
            for _i in range(len(_buf)):
                _buf[_i] = None
            self.step_refresh_chunk_mask[buf_id] = 0
        for idx in range(len(self.step_prefill_capture_layout_ring)):
            self.step_prefill_capture_layout_ring[idx] = None
        for idx in range(len(self.step_refresh_capture_layout_ring)):
            self.step_refresh_capture_layout_ring[idx] = None
        for idx in range(len(self._capture_ring_active_lease_by_buf)):
            self._capture_ring_active_lease_by_buf[idx] = None
        self._capture_ring_retired_events.clear()
        self._lease_stats["pending"] = 0

        # 释放 controller 级缓冲
        self._selector_key_norms_all = None
        self._selector_key_norms_shape = None
        _kn_cache = getattr(self, "_selector_key_norms_all_cache", None)
        if isinstance(_kn_cache, dict):
            _kn_cache.clear()  # idle: free retained key_norms_all GPU buffers
        self._selector_capture_scores_all = None
        self._selector_kv_lengths_all = None
        self._selector_log_f_denoms_all = None
        self._selector_key_norms_delta_start_cpu = None
        self._selector_key_norms_delta_end_cpu = None
        self._selector_key_norms_delta_start_gpu = None
        self._selector_key_norms_delta_end_gpu = None
        self._selector_layer_index_cache_key = None
        self._selector_layer_index_cache_device = None
        self._selector_layer_index_cache_tensor = None
        self.step_decode_req_meta_i32_all = None
        self.step_decode_req_meta_i64_all = None
        self.step_decode_is_compact_all = None
        self.step_bound_meta = None
        self._step_bound_meta_probe_checked_epoch = -1
        self._step_bound_meta_probe_checked_handle_id = -1
        self._step_bound_meta_probe_checked_handle_generation = -1
        self._step_bound_meta_probe_warned_missing_authority = False
        self.step_prefill_req_meta_i32_all = None
        self.step_prefill_req_meta_i64_all = None

        # 释放 per-layer buffers（保留层规格与 cache_key 映射）
        for state in self.layer_states.values():
            state.reset_idle_buffers()
        # idle release 只做状态清理，不应触发任何 flush 调度。


    # ------------------------------------------------------------------
    # Step context helpers (no behavior change)
    # ------------------------------------------------------------------

    def _build_step_semantic_snapshot(self) -> StepSemanticSnapshot:
        cfg = self.config
        alpha_log_f = 0.5
        sink_tokens = 0
        recent_tokens = 0
        if cfg is not None:
            sink_tokens = cfg.sink
            recent_tokens = cfg.recent
            if cfg.alpha_fair is not None:
                alpha_log_f = cfg.alpha_fair.alpha
        return StepSemanticSnapshot(
            epoch=self.step_context_epoch,
            alpha_log_f=alpha_log_f,
            sink_tokens=sink_tokens,
            recent_tokens=recent_tokens,
            capture_inflight=_CAPTURE_IN_FLIGHT,
        )

    def _get_step_semantic_snapshot(self) -> StepSemanticSnapshot:
        snap = self._step_semantic_snapshot
        if snap is None or snap.epoch != self.step_context_epoch:
            snap = self._build_step_semantic_snapshot()
            self._step_semantic_snapshot = snap
        return snap

    def _build_step_exec_hints(
        self,
        *,
        batch_size: int,
    ) -> StepExecHints:
        epoch = self.step_context_epoch
        # Only wait_token_by_buf is consumed by wait_decider.
        blockers = bool(self._pending_work_blockers(epoch=epoch))
        wait_token = max(1, epoch)
        wait_token_by_buf_list: List[int] = []
        for buf in range(_CAPTURE_IN_FLIGHT):
            has_pending = False
            if 0 <= buf < len(self._buf_pending_work_flags):
                has_pending = self._buf_pending_work_flags[buf] != 0
            wait_token_by_buf_list.append(wait_token if (blockers or has_pending) else 0)
        return StepExecHints(
            epoch=epoch,
            batch_size=batch_size,
            wait_token_by_buf=tuple(wait_token_by_buf_list),
        )

    def prepare_step_context(
        self,
        *,
        req_ids: Sequence[str],
        num_scheduled_tokens: Sequence[int],
        q_start_loc: Optional[Sequence[int]] = None,
        query_start_loc: Optional[Sequence[int]] = None,
        seq_lens: Sequence[int],
        num_actual_tokens: int,
        prompt_lengths: Optional[Sequence[int]] = None,
        num_computed_tokens: Optional[Sequence[int]] = None,
        step_ticket: StepTicket,
    ) -> Optional[StepContext]:
        impl = _load_step_context_impl()
        q_start_loc_src: Optional[Sequence[int]] = q_start_loc
        if q_start_loc_src is None:
            q_start_loc_src = query_start_loc
        if q_start_loc_src is None:
            raise RuntimeError("prepare_step_context requires q_start_loc")
        return impl(
            self,
            req_ids=req_ids,
            num_scheduled_tokens=num_scheduled_tokens,
            q_start_loc=q_start_loc_src,
            seq_lens=seq_lens,
            num_actual_tokens=num_actual_tokens,
            prompt_lengths=prompt_lengths,
            num_computed_tokens=num_computed_tokens,
            step_ticket=step_ticket,
        )

    def _require_step_bound_meta(
        self,
        *,
        step_context: StepContext,
        stage: str,
    ) -> StepBoundMeta:
        step_ctx_epoch = step_context.epoch
        step_ctx_handle_id = step_context.step_handle_id
        step_ctx_handle_generation = step_context.step_handle_generation
        step_ctx_num_reqs = step_context.num_reqs
        step_bound_meta = self.step_bound_meta
        if step_bound_meta is None:
            raise RuntimeError(
                f"{stage}: missing step_bound_meta "
                f"(epoch={step_ctx_epoch})"
            )
        bound_epoch = step_bound_meta.epoch
        if bound_epoch != step_ctx_epoch:
            raise RuntimeError(
                f"{stage}: step_bound_meta epoch mismatch "
                f"(bound_epoch={bound_epoch} "
                f"ctx_epoch={step_ctx_epoch})"
            )
        bound_handle_id = step_bound_meta.step_handle_id
        bound_handle_generation = step_bound_meta.step_handle_generation
        if (
            bound_handle_id != step_ctx_handle_id
            or bound_handle_generation != step_ctx_handle_generation
        ):
            raise RuntimeError(
                f"{stage}: step_bound_meta handle mismatch "
                f"(bound=({bound_handle_id},"
                f" {bound_handle_generation}) "
                f"ctx=({step_ctx_handle_id},"
                f" {step_ctx_handle_generation}))"
            )
        probe_checked_same_step = (
            self._step_bound_meta_probe_checked_epoch == bound_epoch
            and self._step_bound_meta_probe_checked_handle_id == bound_handle_id
            and self._step_bound_meta_probe_checked_handle_generation
            == bound_handle_generation
        )
        if probe_checked_same_step:
            # 同一 step 内首层已验证通过，后续层直接返回。
            # batch_size / q_start_loc / logits 覆盖在 step 内不变。
            return step_bound_meta
        if True:
            step_authority = self.step_authority
            if step_authority is None:
                if not bool(self._step_bound_meta_probe_warned_missing_authority):
                    _log.warning(
                        "%s: step_bound_meta probe skipped once (step_authority missing)",
                        stage,
                    )
                    self._step_bound_meta_probe_warned_missing_authority = True
            else:
                auth_req_set_hash = step_authority.req_set_hash
                auth_row_phase_hash = step_authority.row_phase_hash
                bound_req_set_hash = step_bound_meta.req_set_hash
                bound_row_phase_hash = step_bound_meta.row_phase_hash
                if (
                    bound_req_set_hash != auth_req_set_hash
                    or bound_row_phase_hash != auth_row_phase_hash
                ):
                    raise RuntimeError(
                        f"{stage}: step_bound_meta probe signature drift "
                        f"(bound_req_set_hash={bound_req_set_hash} "
                        f"bound_row_phase_hash={bound_row_phase_hash} "
                        f"auth_req_set_hash={auth_req_set_hash} "
                        f"auth_row_phase_hash={auth_row_phase_hash})"
                    )
            self._step_bound_meta_probe_checked_epoch = bound_epoch
            self._step_bound_meta_probe_checked_handle_id = bound_handle_id
            self._step_bound_meta_probe_checked_handle_generation = (
                bound_handle_generation
            )
        bound_batch_size = step_bound_meta.batch_size
        if bound_batch_size < step_ctx_num_reqs:
            raise RuntimeError(
                f"{stage}: step_bound_meta batch_size too small "
                f"(bound_batch={bound_batch_size} num_reqs={step_ctx_num_reqs})"
            )
        if len(step_bound_meta.q_start_loc) < step_ctx_num_reqs + 1:
            raise RuntimeError(
                f"{stage}: step_bound_meta q_start_loc coverage mismatch "
                f"(q_start={len(step_bound_meta.q_start_loc)} num_reqs={step_ctx_num_reqs})"
            )
        if len(step_bound_meta.logits_last_n_by_row) < step_ctx_num_reqs:
            raise RuntimeError(
                f"{stage}: step_bound_meta logits_last_n coverage mismatch "
                f"(rows={len(step_bound_meta.logits_last_n_by_row)} num_reqs={step_ctx_num_reqs})"
            )
        if len(step_bound_meta.logits_capacity_by_row) < step_ctx_num_reqs:
            raise RuntimeError(
                f"{stage}: step_bound_meta logits_capacity coverage mismatch "
                f"(rows={len(step_bound_meta.logits_capacity_by_row)} num_reqs={step_ctx_num_reqs})"
            )
        return step_bound_meta

    def maybe_build_step_decode_data_from_metadata(
        self,
        *,
        attn_metadata: object,
        kv_cache_spec: Optional[object],
    ) -> None:
        decode_impl, _, _ = _load_metadata_builder_impls()
        return decode_impl(
            self,
            attn_metadata=attn_metadata,
            kv_cache_spec=kv_cache_spec,
        )

    def _build_step_decode_data(
        self,
        step_meta: StepMeta,
        batch_size: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        kv_cache_dtype: torch.dtype,
        cache_key: Tuple[object, ...],
        decode_plan_version: int = -1,
    ) -> None:
        impl = _load_step_decode_data_impl()
        return impl(
            self,
            step_meta=step_meta,
            batch_size=batch_size,
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            kv_cache_dtype=kv_cache_dtype,
            cache_key=cache_key,
            decode_plan_version=decode_plan_version,
        )

    def _build_step_dispatch_plan(
        self,
        step_meta: StepMeta,
        *,
        cache_key: Optional[Tuple[object, ...]] = None,
    ) -> None:
        """构建 step 级调度计划，避免 per-layer 重复判断。"""
        step_decode_data = self.step_decode_data
        step_authority = self.step_authority
        reason: Optional[str] = None
        reuse_ordered_layers = False
        cache_key_use = cache_key if cache_key is not None else self.step_decode_cache_key
        decode_plan_version = int(getattr(self, "step_decode_plan_version", -1))
        if int(getattr(step_meta, "decode_plan_version", -1)) >= 0:
            decode_plan_version = int(step_meta.decode_plan_version)
        self.step_decode_plan_version = int(decode_plan_version)

        force_dense = _FORCE_DENSE_CACHED
        force_compact_off = _FORCE_COMPACT_OFF_CACHED
        if self.config is None or not self.config.enabled:
            reason = "disabled"
        elif step_authority is None:
            raise RuntimeError("step dispatch plan requires step_authority single source")
        elif not bool(step_authority.is_decode_only):
            reason = "not_decode"
        elif self._step_refresh_nonempty():
            # refresh step: allow ordered reuse only when layer-group gating is enabled
            # (inactive layers can reuse ordered layer data; active layers rebuild step data).
            if not self._refresh_layer_group_enabled:
                reason = "refresh_step"
        elif force_dense:
            reason = "force_dense"
        elif force_compact_off:
            reason = "force_compact_off"
        else:
            reuse_ordered_layers = True
            if step_decode_data is None:
                reason = "no_step_data"
            elif cache_key_use is None:
                reason = "no_cache_key"
            elif decode_plan_version < 0:
                reason = "no_plan_version"
            elif step_decode_data.cache_key != cache_key_use:
                reason = "cache_key_mismatch"
            elif int(getattr(step_decode_data, "decode_plan_version", -1)) != decode_plan_version:
                reason = "plan_version_mismatch"
            elif step_meta.seqused_k_gpu is None:
                reason = "seqused_none"
            elif step_meta.seqused_k_gpu.numel() < step_meta.batch_size:
                reason = "seqused_short"

        layer_data_list_ordered: Optional[List[LayerDecodeData]] = None
        if reuse_ordered_layers and reason is None and step_decode_data is not None:
            reuse_order = build_decode_reuse_order_decision(
                has_cached_layer_list=(self.ordered_layer_data_list is not None),
                cached_step_data_identity=(self.ordered_step_decode_data is step_decode_data),
                cached_order_cache_key=self._runtime_state.ordered_cache_key,
                current_cache_key=cache_key_use,
            )
            if reuse_order:
                layer_data_list_ordered = self.ordered_layer_data_list
            else:
                layer_data_list_ordered = []
                for ck in self.layer_cache_keys:
                    layer_index = self.layer_index_by_cache_key.get(ck, -1)
                    if layer_index < 0 or layer_index >= len(step_decode_data.layer_data):
                        reason = "layer_index_oob"
                        break
                    layer_data = step_decode_data.layer_data[layer_index]
                    if layer_data is None:
                        reason = "no_layer_data"
                        break
                    layer_data_list_ordered.append(layer_data)
            if reason is None and not layer_data_list_ordered:
                reason = "no_layer_data"
            # Compact readiness guard: when any row has left short-dense
            # (expects compact mode), verify every layer has compact data
            # before allowing ordered reuse.  Pure CPU check — no GPU sync.
            if reason is None and layer_data_list_ordered:
                _needs_compact = (
                    step_authority is not None
                    and any(
                        not sd
                        for sd in step_authority.short_dense_by_row[
                            : step_meta.batch_size
                        ]
                    )
                )
                if _needs_compact:
                    for _ld in layer_data_list_ordered:
                        if _ld.compact_kv_len_max <= 0:
                            reason = "compact_not_ready"
                            break

        if reuse_ordered_layers and reason is None:
            if self._ordered_reuse_hit_epoch != step_meta.epoch:
                self._ordered_reuse_hit_epoch = step_meta.epoch

        if reason is not None:
            if reuse_ordered_layers:
                if self._ordered_reuse_miss_epoch != step_meta.epoch:
                    self._ordered_reuse_miss_epoch = step_meta.epoch
                # step profile：记录 ordered layer reuse 的失败原因。
                self._step_profile_record_plan(mode="ordered_reuse", reason=str(reason))
            # ordered reuse 失败时不构建计划，避免静默回退
            if reuse_ordered_layers:
                self.step_dispatch_plan = None
                self.ordered_layer_data_list = None
                self.ordered_step_decode_data = None
                reset_runtime_plan_state(self, epoch=int(step_meta.epoch))
                self._set_unified_attention_mode("default")
                return
            # 默认路径允许计划存在（refresh/prefill 等场景）

        self.step_dispatch_plan = StepDispatchPlan(
            cache_key=cache_key_use,
            epoch=step_meta.epoch,
            batch_size=step_meta.batch_size,
            step_decode_data=step_decode_data if reuse_ordered_layers else None,
            decode_plan_version=decode_plan_version,
            layer_data_list_ordered=layer_data_list_ordered if reuse_ordered_layers else None,
        )
        # step profile：记录本步计划状态。
        self._step_profile_record_plan(
            mode=("ordered_reuse" if reuse_ordered_layers else "default"),
            reason=(str(reason) if reason else None),
        )
        if reuse_ordered_layers:
            self.ordered_layer_data_list = layer_data_list_ordered
            self.ordered_step_decode_data = step_decode_data
            apply_ordered_layer_plan_state(
                self,
                epoch=int(step_meta.epoch),
                layer_count=(
                    len(layer_data_list_ordered)
                    if layer_data_list_ordered is not None
                    else 0
                ),
                batch_size=int(step_meta.batch_size),
                cache_key=cache_key_use,
            )
            self._set_unified_attention_mode("ordered_reuse")
        else:
            self.ordered_layer_data_list = None
            self.ordered_step_decode_data = None
            reset_runtime_plan_state(self, epoch=int(step_meta.epoch))
            self._set_unified_attention_mode("default")

    def _enqueue_prefill_capture(
        self,
        payload: SelectorBatchPayload,
    ) -> None:
        if self.step_prefill_epoch != self.step_context_epoch:
            self.step_prefill_epoch = self.step_context_epoch
            for buf_id, bucket in enumerate(self.step_prefill_chunk_payloads):
                for idx in range(int(_CAPTURE_CHUNK)):
                    bucket[idx] = None
                self.step_prefill_chunk_mask[buf_id] = 0
        layer_index = self.layer_index_by_cache_key.get(payload.cache_key, -1)
        if layer_index < 0:
            raise RuntimeError("prefill capture enqueue missing layer_index")
        _, buf_id, slot_in_chunk = self._map_global_layer_to_capture_slot(layer_index)
        if slot_in_chunk < 0 or slot_in_chunk >= int(_CAPTURE_CHUNK):
            raise RuntimeError("prefill capture enqueue slot_in_chunk out of range")
        bucket = self.step_prefill_chunk_payloads[buf_id]
        mask = int(self.step_prefill_chunk_mask[buf_id])
        bit = 1 << int(slot_in_chunk)
        if (mask & bit) != 0 or bucket[slot_in_chunk] is not None:
            raise RuntimeError(
                f"duplicate prefill payload for buf_id={buf_id} slot_in_chunk={slot_in_chunk}"
            )
        bucket[slot_in_chunk] = payload
        self.step_prefill_chunk_mask[buf_id] = mask | bit
        # 记录最近一次 prefill enqueue epoch，用于无 prompt/computed 时的释放回退判据
        self._prefill_last_enqueue_epoch = self.step_context_epoch
        self._prefill_release_done_epoch = -1


    def _flush_prefill_batches(
        self,
        *,
        buf_id: int,
        chunk_id: int,
        chunk_size: int,
        is_last_layer: bool,
    ) -> None:
        return flush_prefill_batches_impl(
            self,
            buf_id=int(buf_id),
            chunk_id=int(chunk_id),
            chunk_size=int(chunk_size),
            is_last_layer=bool(is_last_layer),
            capture_in_flight=int(_CAPTURE_IN_FLIGHT),
            capture_chunk=int(_CAPTURE_CHUNK),
            is_stream_capturing_or_raise_fn=_is_stream_capturing_or_raise,
            flush_profile_accum_cls=_FlushProfileAccum,
            refresh_micro_profile_cached=bool(_REFRESH_MICRO_PROFILE_CACHED),
            refresh_micro_profile_buf=_REFRESH_MICRO_PROFILE_BUF,
            refresh_micro_profile_count=_REFRESH_MICRO_PROFILE_COUNT,
            flush_micro_profile_summary_fn=_flush_micro_profile_summary,
            refresh_profile_pending_cls=_RefreshProfilePending,
            make_selector_fast_signature_fn=_make_selector_fast_signature,
            rebuild_physical_block_sort_cached=bool(_REBUILD_PHYSICAL_BLOCK_SORT_CACHED),
        )

    def allocate_step_handle(
        self,
        *,
        epoch: int,
        req_ids: Sequence[str],
        num_actual_tokens: int,
        refresh_plan_signature: Sequence[object],
    ) -> StepHandle:
        """为当前 step 分配稳定句柄（ring + generation 双因子）。"""
        handle_id = int(self._step_handle_next_id) + 1
        self._step_handle_next_id = int(handle_id)
        ring_size = int(self._step_handle_ring_size)
        slot = int(handle_id % ring_size)
        generation = int(self._step_handle_generation_by_slot[slot]) + 1
        self._step_handle_generation_by_slot[slot] = int(generation)
        req_ids_tuple = tuple(str(rid) for rid in req_ids)
        req_ids_signature = int(hash(req_ids_tuple) & 0x7FFFFFFFFFFFFFFF)
        return StepHandle(
            handle_id=int(handle_id),
            epoch=int(epoch),
            generation=int(generation),
            req_ids_signature=int(req_ids_signature),
            num_actual_tokens=int(num_actual_tokens),
            refresh_plan_signature=tuple(refresh_plan_signature),
        )

    def bind_step_handle_context(
        self,
        *,
        step_handle: StepHandle,
        step_context: StepContext,
    ) -> None:
        """把 StepHandle 与 StepContext 绑定到 ring（单真源注册）。"""
        if int(step_context.epoch) != int(step_handle.epoch):
            raise RuntimeError(
                "step handle/context epoch mismatch during bind: "
                f"handle_epoch={int(step_handle.epoch)} ctx_epoch={int(step_context.epoch)}"
            )
        slot = int(step_handle.handle_id % int(self._step_handle_ring_size))
        if int(self._step_handle_generation_by_slot[slot]) != int(step_handle.generation):
            raise RuntimeError(
                "step handle generation mismatch during bind: "
                f"slot={slot} expected={int(self._step_handle_generation_by_slot[slot])} "
                f"actual={int(step_handle.generation)}"
            )
        self._step_handle_by_slot[slot] = step_handle
        self._step_context_by_handle_slot[slot] = step_context
        self._current_step_handle_id = int(step_handle.handle_id)
        self._current_step_handle_generation = int(step_handle.generation)

    def get_step_handle(
        self,
        *,
        step_handle_id: int,
        expected_generation: Optional[int] = None,
    ) -> Optional[StepHandle]:
        if step_handle_id <= 0:
            return None
        slot = step_handle_id % self._step_handle_ring_size
        handle = self._step_handle_by_slot[slot]
        if handle is None:
            return None
        if handle.handle_id != step_handle_id:
            return None
        if expected_generation is not None:
            if expected_generation <= 0:
                return None
            if handle.generation != expected_generation:
                return None
        return handle

    def get_step_context_for_handle(
        self,
        *,
        step_handle_id: int,
        expected_generation: Optional[int] = None,
    ) -> Optional[StepContext]:
        handle = self.get_step_handle(
            step_handle_id=step_handle_id,
            expected_generation=expected_generation,
        )
        if handle is None:
            return None
        slot = handle.handle_id % self._step_handle_ring_size
        ctx = self._step_context_by_handle_slot[slot]
        if ctx is None:
            return None
        if ctx.step_handle_id != handle.handle_id:
            return None
        if ctx.step_handle_generation != handle.generation:
            return None
        return ctx


    def _get_step_capture_layout(
        self,
        *,
        phase: str,
        state: LayerState,
        step_context: StepContext,
        global_layer_index: int,
        slot_list: List[int],
        seqused_k: torch.Tensor,
        num_heads: int,
        device: torch.device,
        chunk_query_lengths: Optional[torch.Tensor] = None,
        prepared_only: bool = False,
        skip_live_lengths: bool = False,
    ) -> Optional[StepCaptureLayout]:
        # [CU-SEQLENS-DEAD-PARAM-RETIRE 2026-07-09] cu_seqlens_q 形参下线:
        # layout impl 全程不读(死形参),而 metadata_builder 触发步为喂它
        # 每次做 numpy cumsum+pageable 同步 H2D(decode_out_ptr_prep 2-8ms
        # 族根源之一)。参数链连根拔除;payload_worker 入参链的残余死参随
        # installer B 级清理场收尾。
        impl = _load_capture_layout_impl()
        return impl(
            self,
            phase=phase,
            state=state,
            step_context=step_context,
            global_layer_index=global_layer_index,
            slot_list=slot_list,
            seqused_k=seqused_k,
            num_heads=num_heads,
            device=device,
            chunk_query_lengths=chunk_query_lengths,
            prepared_only=prepared_only,
            skip_live_lengths=skip_live_lengths,
        )


    # ------------------------------------------------------------------
    # Compact KV helpers
    # ------------------------------------------------------------------


    # NOTE: _rebuild_compact_slot (single-slot, non-batched) 已删除——
    # 所有调用点已迁移到 _rebuild_compact_slots_batched_layers_from_selection。

    def _rebuild_compact_slots_batched_layers_from_selection(
        self,
        payloads: Sequence[SelectorBatchPayload],
        selected_indices: torch.Tensor,
        *,
        phase: str,
        bootstrap_slots_by_layer: Optional[Sequence[Set[int]]] = None,
        prepare_for_capture_only: bool = False,
        defer_compact_meta_publish: bool = False,
        compact_meta_commit_log: Optional[List[Dict[str, object]]] = None,
    ) -> bool:
        rebuild_impl, _ = _load_selector_selection_impls()
        return rebuild_impl(
            self,
            payloads=payloads,
            selected_indices=selected_indices,
            phase=phase,
            bootstrap_slots_by_layer=bootstrap_slots_by_layer,
            prepare_for_capture_only=prepare_for_capture_only,
            defer_compact_meta_publish=defer_compact_meta_publish,
            compact_meta_commit_log=compact_meta_commit_log,
        )

    def _register_layer(
        self,
        cache_key: int,
        num_heads: int,
        num_kv_heads: int,
        device: torch.device,
        head_dim: Optional[int] = None,
        kv_cache_dtype: Optional[torch.dtype] = None,
    ) -> LayerState:
        state = LayerState(num_heads, num_kv_heads, device)
        if head_dim is not None:
            state.head_dim = int(head_dim)
        if kv_cache_dtype is not None:
            state.kv_cache_dtype = kv_cache_dtype
        state.k_max_current = self.config.k_max

        self.layer_states[cache_key] = state
        if cache_key not in self.layer_index_by_cache_key:
            self.layer_index_by_cache_key[cache_key] = len(self.layer_cache_keys)
            self.layer_cache_keys.append(cache_key)
            # 新层加入时，下一步需要重建 StepDecodeData
            self.step_decode_data = None
            self.step_decode_cache_key = None
            self.step_decode_plan_version = -1
            self._step_decode_spec_key = None
            self.step_dispatch_plan = None
        if self.base_cache_key is None:
            self.base_cache_key = cache_key
            self.base_last_seq_len = -1
        # 记录稳定的“真实层序号”，用于 refresh layer-group gating。
        state.layer_index = int(self.layer_index_by_cache_key.get(cache_key, -1))
        # 缓存 per-layer capture slot 映射（避免热路径重复计算）
        if state.layer_index >= 0:
            chunk_id, buf_id, slot_in_chunk = self._map_global_layer_to_capture_slot(state.layer_index)
            state.capture_chunk_id = int(chunk_id)
            state.capture_buf_id = int(buf_id)
            state.capture_slot_in_chunk = int(slot_in_chunk)
            state.layer_index_epoch = int(self._layer_index_cache_epoch)
        else:
            state.capture_chunk_id = -1
            state.capture_buf_id = -1
            state.capture_slot_in_chunk = -1
            state.layer_index_epoch = int(self._layer_index_cache_epoch)
        return state

    def _refresh_layer_index_cache(self) -> None:
        """刷新 layer_index + capture slot 映射缓存（全量更新）。"""
        self._layer_index_cache_epoch += 1
        epoch = int(self._layer_index_cache_epoch)
        for cache_key, state in self.layer_states.items():
            idx = int(self.layer_index_by_cache_key.get(cache_key, -1))
            state.layer_index = idx
            if idx >= 0:
                chunk_id, buf_id, slot_in_chunk = self._map_global_layer_to_capture_slot(idx)
                state.capture_chunk_id = int(chunk_id)
                state.capture_buf_id = int(buf_id)
                state.capture_slot_in_chunk = int(slot_in_chunk)
            else:
                state.capture_chunk_id = -1
                state.capture_buf_id = -1
                state.capture_slot_in_chunk = -1
            state.layer_index_epoch = epoch

    def _ensure_request(self, request_id: str) -> RequestTracking:
        tracking = self.request_states.get(request_id)
        if tracking is None:
            tracking = RequestTracking()
            if self.config.trigger is not None and self._workload_plan_replay is None:
                tracking.trigger = RefreshTrigger(self.config.trigger)
            self.request_states[request_id] = tracking
        self._ensure_request_ticket(request_id)
        return tracking

    def _reset_request_sparse_state_for_resume(
        self, request_id: str, tracking: "RequestTracking"
    ) -> None:
        # [RESUME-STATE-RESET 2026-07-03] vLLM 抢占(KV 逐出,RECOMPUTE)不进
        # finished_req_ids,sparse 清理链对其完全 no-op:resume 拿到同 slot 零重
        # 置——key_norms_len 从抢占前旧值继续累计(重吞已生成段→~2× 溢计,选页
        # 依据污染=静默错数,审计实证),旧 compact 缓冲被 compact_ready=True 放
        # 行且无长度对账。resume 语义=从头重算=重新 bootstrap,旧压缩状态必须
        # 清零。检测点(step_context_worker 分类循环)判据:已 bootstrap 的请求
        # 回到 prompt 中段(computed<prompt_len ∧ tracking.bootstrap_done),纯
        # host 比较零开销;本函数置 bootstrap_done=False 后判据自灭,天然 once。
        # slot 身份保留(请求仍活,不 release/不动 free_slots),仅清压缩状态。
        for state in self.layer_states.values():
            slot = state.request_id_to_slot.get(request_id)
            if slot is None or slot < 0 or slot >= state.batch_size:
                continue
            self._reset_compact_slot(state, int(slot))
            if slot < state.key_norms_len.size(0):
                state.key_norms_len[slot] = 0
            if slot < len(state.key_norms_capacity):
                state.key_norms_capacity[slot] = 0
            if state.prefill_done_mask is not None and slot < state.prefill_done_mask.shape[0]:
                state.prefill_done_mask[slot] = False
            if state.prefill_active_mask is not None and slot < state.prefill_active_mask.shape[0]:
                state.prefill_active_mask[slot] = False
            if state.prefill_fifo_counts_cpu is not None and slot < len(state.prefill_fifo_counts_cpu):
                state.prefill_fifo_counts_cpu[slot] = 0
            if state.prefill_kv_len_per_row_i32 is not None and slot < state.prefill_kv_len_per_row_i32.shape[0]:
                state.prefill_kv_len_per_row_i32[slot] = 0
            if state.prefill_kv_lengths is not None and slot < state.prefill_kv_lengths.shape[0]:
                state.prefill_kv_lengths[slot].zero_()
            steps_cpu = state.last_refresh_step_per_slot_cpu
            if steps_cpu is not None and slot < len(steps_cpu):
                steps_cpu[slot] = -1
            steps_decode_cpu = state.last_refresh_decode_per_slot_cpu
            if steps_decode_cpu is not None and slot < len(steps_decode_cpu):
                steps_decode_cpu[slot] = -1
            _revoke_slot_selected_truth(state, slot=int(slot))
            _invalidate_page_sparse_step_cache_truth(state)
            state._slot_row_map_key = None
        tracking.bootstrap_done = False
        # [CHUNKED-CAPTURE-ACCUMULATE 2026-07-06] resume=从头重算=捕获窗重新
        # 分片；同长 prompt 不触发 record_prompt_tokens 的 != 归零，跨片累计
        # 必须在此清零，否则 finalize 对账把旧片计入 fail-fast 误伤。
        tracking.prefill_chunks_seen = 0
        tracking.prefill_capture_rows_accum = 0
        tracking.prefill_capture_prev_capacity = 0
        tracking.prefill_capture_accum_prev_rows = 0
        tracking.prefill_capture_accum_prev_capacity = 0
        tracking.prefill_capture_accum_sit = -1
        _log.warning(
            "sparse resume detected for request %s: compact/key_norms state reset, "
            "re-bootstrapping from scratch",
            request_id,
        )

    def _ensure_request_ticket(self, request_id: str) -> RequestIntentTicket:
        ticket = self._request_intent_tickets.get(request_id)
        if ticket is None:
            ticket = RequestIntentTicket()
            self._request_intent_tickets[request_id] = ticket
        return ticket

    def _set_request_trigger_intent(
        self,
        *,
        request_id: str,
        reason: str,
        decode_step: int,
        force_now: bool,
    ) -> bool:
        step_int = decode_step
        if step_int < 0:
            return False
        reason_str = reason or "trigger"
        finished_ids = getattr(self, "_finished_req_ids_step", set())
        snapshot_finished_ids = getattr(self, "_snapshot_finished_req_ids", set())
        if request_id in finished_ids or request_id in snapshot_finished_ids:
            if "sentence" in reason_str:
                self._sentence_trigger_admission_dropped_finished_total = (
                    int(
                        getattr(
                            self,
                            "_sentence_trigger_admission_dropped_finished_total",
                            0,
                        )
                    )
                    + 1
                )
            return False
        tracking = self._ensure_request(request_id)
        prev_step = tracking.trigger_intent_decode_step
        prev_reason = tracking.trigger_intent_reason or "none"
        prev_force = tracking.trigger_intent_force_now

        changed = False
        if prev_step < 0 or step_int < prev_step:
            tracking.trigger_intent_decode_step = step_int
            tracking.trigger_intent_reason = reason_str
            changed = True
        if prev_reason == "none":
            tracking.trigger_intent_reason = reason_str
            changed = True
        if force_now and (not prev_force):
            tracking.trigger_intent_force_now = True
            tracking.trigger_intent_reason = reason_str
            changed = True
        if changed and "sentence" in reason_str:
            self._sentence_trigger_intents_total = (
                int(getattr(self, "_sentence_trigger_intents_total", 0)) + 1
            )
        return changed

    def _clear_request_trigger_intent(self, *, request_id: str) -> bool:
        tracking = self.request_states.get(request_id)
        if tracking is None:
            return False
        changed = (
            tracking.trigger_intent_decode_step >= 0
            or (tracking.trigger_intent_reason or "none") != "none"
            or tracking.trigger_intent_force_now
        )
        tracking.trigger_intent_decode_step = -1
        tracking.trigger_intent_reason = "none"
        tracking.trigger_intent_force_now = False
        return changed

    def _set_request_lease_rearm(
        self,
        *,
        request_id: str,
        reason: str,
        decode_step: int,
    ) -> bool:
        tracking = self._ensure_request(request_id)
        step_int = decode_step
        reason_str = reason or "lease_rearm"
        prev_flag = tracking.lease_rearm
        prev_reason = tracking.lease_rearm_reason or "none"
        prev_step = tracking.lease_rearm_decode_step
        changed = (
            (not prev_flag)
            or prev_reason != reason_str
            or prev_step != step_int
        )
        tracking.lease_rearm = True
        tracking.lease_rearm_reason = reason_str
        tracking.lease_rearm_decode_step = step_int
        return changed

    def _clear_request_lease_rearm(self, *, request_id: str) -> bool:
        tracking = self.request_states.get(request_id)
        if tracking is None:
            return False
        changed = (
            tracking.lease_rearm
            or (tracking.lease_rearm_reason or "none") != "none"
            or tracking.lease_rearm_decode_step >= 0
        )
        tracking.lease_rearm = False
        tracking.lease_rearm_reason = "none"
        tracking.lease_rearm_decode_step = -1
        return changed

    def _set_request_pending_refresh(
        self,
        *,
        request_id: str,
        reason: Optional[str] = None,
        reason_code: Optional[int] = None,
        decode_step: int,
        pending_policy: Optional[int] = None,
        pending_ctrl_step: Optional[int] = None,
    ) -> bool:
        self._ensure_request(request_id)
        ticket = self._ensure_request_ticket(request_id)
        if reason_code is None:
            reason_norm = reason or "refresh"
            reason_code_int = pending_reason_to_code(reason_norm)
        else:
            reason_code_int = int(reason_code)
        step_int = decode_step
        if pending_policy is None:
            policy_int = ticket.pending_policy
        else:
            policy_int = pending_policy
        ctrl_step_cur = ticket.pending_ctrl_step
        if pending_ctrl_step is None:
            if ticket.pending_refresh and ctrl_step_cur >= 0:
                ctrl_step_int = ctrl_step_cur
            else:
                ctrl_step_int = self.step_context_epoch
        else:
            ctrl_step_int = pending_ctrl_step
        changed = (
            (not ticket.pending_refresh)
            or ticket.pending_reason_code != reason_code_int
            or ticket.pending_decode_step != step_int
            or ticket.pending_ctrl_step != ctrl_step_int
            or ticket.pending_policy != policy_int
            or ticket.state != TicketState.PENDING_REFRESH
        )
        ticket.pending_refresh = True
        ticket.pending_reason_code = reason_code_int
        ticket.pending_decode_step = step_int
        ticket.pending_ctrl_step = ctrl_step_int
        ticket.pending_policy = policy_int
        ticket.state = TicketState.PENDING_REFRESH
        return changed

    def _clear_request_pending_refresh(
        self,
        *,
        request_id: str,
        ready_compact: bool,
    ) -> bool:
        tracking = self._ensure_request(request_id)
        ticket = self._ensure_request_ticket(request_id)
        next_state = TicketState.READY_COMPACT if ready_compact else TicketState.NOT_READY
        changed = (
            ticket.pending_refresh
            or ticket.pending_reason_code
            != PendingReasonCode.NONE
            or ticket.pending_decode_step != -1
            or ticket.pending_ctrl_step != -1
            or ticket.pending_policy != PendingPolicy.COALESCEABLE
            or ticket.state != next_state
        )
        ticket.pending_refresh = False
        ticket.pending_reason_code = PendingReasonCode.NONE
        ticket.pending_decode_step = -1
        ticket.pending_ctrl_step = -1
        ticket.pending_policy = PendingPolicy.COALESCEABLE
        ticket.state = next_state
        if ready_compact:
            # [SCHED-RESIDUE-FIX 2026-07-07] ready_compact 清除 = 世代对该
            # request 终局;scheduled_* 必须一并终局,否则 publish 时刻 GPU
            # writer 未完(_update_selection_tracking 走 partial 分支置
            # scheduled)、随后 plan 以 covered-by-ready-compact 清 ticket 的
            # 请求会永久滞留 scheduled → inflight_refresh 恒真 → 该 request
            # 的 sentence/interval 触发从此全部静默(bs8x12k 实测每请求首个
            # sentence 世代后 interval_trigger_intents 恒 0)。正常完成路径
            # (_update_selection_tracking final 分支)本就先清 scheduled 再
            # 调本函数,此处幂等。
            if (
                int(getattr(tracking, "scheduled_decode_refresh_step", -1)) >= 0
                or int(getattr(tracking, "scheduled_refresh_ctrl_step", -1)) >= 0
            ):
                tracking.scheduled_decode_refresh_step = -1
                tracking.scheduled_refresh_ctrl_step = -1
                changed = True
            # [TP-DET-TRIGGER] 读侧在飞镜像随终局清除。
            tracking.inflight_reason_code = -1
            tracking.inflight_policy = -1
        return changed

    def acquire_global_slot(self, request_id: str) -> int:
        if not isinstance(request_id, str):
            raise TypeError(f"request_id must be str, got {type(request_id).__name__}")
        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        self._ensure_request(rid)
        return int(self._global_slot_allocator.acquire(rid))

    def _global_slot_allocator_capacity(self) -> Optional[int]:
        if not bool(getattr(self.config, "compact_page_residency_enabled", False)):
            return None
        return int(getattr(self.config, "max_live_sparse_slots"))

    def release_global_slot(self, request_id: str) -> Optional[int]:
        if not isinstance(request_id, str):
            raise TypeError(f"request_id must be str, got {type(request_id).__name__}")
        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        return self._global_slot_allocator.release(rid)

    def get_step_global_slot_map(self, request_ids: Sequence[str]) -> Dict[str, int]:
        assert_cleanup_ledgers_drained_for_step_build(
            self,
            stage="get_step_global_slot_map",
        )
        req_ids_list: List[str] = []
        for rid_raw in request_ids:
            if not isinstance(rid_raw, str):
                raise TypeError(
                    f"request_id must be str, got {type(rid_raw).__name__}"
                )
            rid = rid_raw.strip()
            if not rid:
                raise ValueError("request_id must be non-empty")
            req_ids_list.append(rid)
        req_ids_tuple = tuple(req_ids_list)
        epoch = self.step_context_epoch
        if (
            self._step_global_slot_map_epoch == epoch
            and self._step_global_slot_map_req_ids == req_ids_tuple
            and self._step_global_slot_map
        ):
            return self._step_global_slot_map
        slot_map: Dict[str, int] = {}
        for rid in req_ids_tuple:
            slot_map[rid] = int(self.acquire_global_slot(rid))
        self._step_global_slot_map_epoch = epoch
        self._step_global_slot_map_req_ids = req_ids_tuple
        self._step_global_slot_map = slot_map
        return slot_map

    def _drain_async_work_at_request_run_boundary(self) -> None:
        if not self._async_refresh_enabled():
            return
        flags = [int(v) for v in getattr(self, "_buf_pending_work_flags", ()) or ()]
        if not any(flags):
            return
        cleanup_device = self.device
        if cleanup_device is None:
            for state in self.layer_states.values():
                device = getattr(state, "device", None)
                if isinstance(device, torch.device):
                    cleanup_device = device
                    break
        if cleanup_device is None:
            raise RuntimeError(
                "request-run reset cannot drain async refresh work without device"
            )
        for buf, flag in enumerate(flags[:_CAPTURE_IN_FLIGHT]):
            if flag:
                self._main_stream_wait_for_chunk_done(
                    buf_id=buf,
                    device=cleanup_device,
                    epoch=self.step_context_epoch,
                )
        remaining = [
            idx
            for idx, flag in enumerate(getattr(self, "_buf_pending_work_flags", ()))
            if int(flag) != 0
        ]
        if remaining:
            raise RuntimeError(
                "request-run reset failed to drain pending async refresh work: "
                f"remaining_bufs={remaining}"
            )

    def _invalidate_request_run_selector_progress(self) -> None:
        valid_keys = getattr(self, "_selector_key_norms_all_valid_cache_keys", None)
        if isinstance(valid_keys, set):
            valid_keys.clear()
        self._selector_key_norms_all_active_cache_key = None
        for state in self.layer_states.values():
            key_norms_len = getattr(state, "key_norms_len", None)
            if isinstance(key_norms_len, torch.Tensor) and key_norms_len.numel() > 0:
                key_norms_len.zero_()

    def _clear_request_run_step_buffers(self) -> None:
        for buf_id in range(len(self.step_prefill_chunk_payloads)):
            bucket = self.step_prefill_chunk_payloads[buf_id]
            for idx in range(len(bucket)):
                bucket[idx] = None
            self.step_prefill_chunk_mask[buf_id] = 0
        for buf_id in range(len(self.step_refresh_chunk_payloads)):
            bucket = self.step_refresh_chunk_payloads[buf_id]
            for idx in range(len(bucket)):
                bucket[idx] = None
            self.step_refresh_chunk_mask[buf_id] = 0
        for idx in range(len(self.step_prefill_capture_layout_ring)):
            self.step_prefill_capture_layout_ring[idx] = None
        for idx in range(len(self.step_refresh_capture_layout_ring)):
            self.step_refresh_capture_layout_ring[idx] = None
        for idx in range(len(self._capture_ring_active_lease_by_buf)):
            self._capture_ring_active_lease_by_buf[idx] = None
        self._capture_ring_retired_events.clear()
        self._pending_work_reset()
        self._main_wait_epoch = -1
        for idx in range(len(self._main_wait_chunk_id_by_buf)):
            self._main_wait_chunk_id_by_buf[idx] = -1
        for idx in range(len(self._step_wait_consumed_token_by_buf)):
            self._step_wait_consumed_token_by_buf[idx] = 0

    def reset_for_completed_request_run(self) -> None:
        """清理一次完整 generate 结束后遗留的 request-local 状态。

        只在显式 reset_prefix_cache / benchmark run 边界调用；不清 CUDA graph、
        layer 注册、compact reserved pages 或 RRP arena，避免破坏 warm graph。
        """
        self._drain_async_work_at_request_run_boundary()
        finished_ids: Set[str] = set()
        for rid in getattr(self, "request_states", {}).keys():
            if isinstance(rid, str) and rid and not _is_free_slot_id(rid):
                finished_ids.add(rid)
        for state in self.layer_states.values():
            for rid in getattr(state, "batch_request_ids", ()) or ():
                if isinstance(rid, str) and rid and not _is_free_slot_id(rid):
                    finished_ids.add(rid)
        allocator = getattr(self, "_global_slot_allocator", None)
        request_to_slot = getattr(allocator, "_request_to_slot", None)
        if isinstance(request_to_slot, dict):
            for rid in request_to_slot.keys():
                if isinstance(rid, str) and rid and not _is_free_slot_id(rid):
                    finished_ids.add(rid)

        if finished_ids:
            finished_set = self._normalize_finished_request_ids(finished_ids)
            pending_new = finished_set - self._snapshot_finished_req_ids
            if pending_new:
                self._snapshot_finished_req_ids.update(pending_new)
                self._finished_generation += 1
            self._finished_req_ids_step.update(finished_set)
            self._pending_global_slot_releases.update(finished_set)
            for rid in finished_set:
                self._bootstrap_pending_request_ids.discard(rid)
                self.request_states.pop(rid, None)
                self._request_intent_tickets.pop(rid, None)
            self._active_step_snapshot = None
            self._active_step_snapshot_epoch = -1
            self._active_step_ticket = None
            self._active_step_source_signature = -1
            self._step_global_slot_map_epoch = -1
            self._step_global_slot_map_req_ids = tuple()
            self._step_global_slot_map = {}
            self._run_finished_cleanup_at_step_boundary()

        self._invalidate_request_run_selector_progress()
        self._clear_request_run_step_buffers()
        self.step_context = None
        self.step_exec_hints = None
        self._current_step_handle_id = -1
        self._current_step_handle_generation = -1
        self._prepared_step_identity = (-1, -1, -1, -1)
        self._prepared_num_actual_tokens = -1
        self._prepared_refresh_nonce = -1
        self._step_semantic_snapshot = None
        self._prev_decode_logf_row_state = None
        self.step_meta = None
        self.step_authority = None
        self.step_decode_data = None
        self.step_decode_cache_key = None
        self.step_decode_plan_version = -1
        self.step_dispatch_plan = None
        reset_runtime_plan_state(self, epoch=-1)
        self.ordered_layer_data_list = None
        self.ordered_step_decode_data = None
        self.layer_dispatch_cursor = 0
        self.layer_dispatch_epoch = -1
        self.layer_dispatch_layer_count = 0
        self._step_refresh_slot_req_ids_epoch = -1
        self._step_refresh_slot_req_ids_handle_id = -1
        self._step_refresh_slot_req_ids_handle_generation = -1
        self._step_refresh_slot_req_ids_cache.clear()
        self._step_dispatcher_refresh_dirty_token = -1
        self._step_dispatcher_refresh_dirty_slot_key = tuple()
        self._step_dispatcher_refresh_dirty_payload_slots = tuple()
        self._step_dispatcher_refresh_dirty_any_decode = False
        self._fa3_live_route_token = -1
        self._fa3_live_route_batch_size = -1
        self._fa3_live_route_has_selected_consume = False
        self._fa3_live_route_has_capture = False
        self._fa3_live_route = None
        self._should_refresh_cache_step = -1
        self._should_refresh_cache_nonce = -1
        self._should_refresh_cache.clear()

    def reset_for_new_engine(self) -> None:
        """重置与 engine/kv-cache 绑定的状态，避免跨 engine 污染。"""
        self.layer_states.clear()
        self.layer_cache_keys.clear()
        self.layer_index_by_cache_key.clear()
        self._layer_index_cache_epoch = 0
        self.step_context = None
        self.step_exec_hints = None
        self._step_handle_next_id = 0
        for i in range(self._step_handle_ring_size):
            self._step_handle_by_slot[i] = None
            self._step_context_by_handle_slot[i] = None
            self._step_handle_generation_by_slot[i] = 0
        self._current_step_handle_id = -1
        self._current_step_handle_generation = -1
        self._worker_token_ids_cpu = None
        self._worker_batch_id_to_idx = None
        self._async_sampled_stash = None
        self._worker_token_source_rows = -1
        self._has_enginecore_sentence_hook = False
        self._step_semantic_snapshot = None
        self._step_authority_builder_scratch = None
        self._prev_decode_logf_row_state = None
        self.step_meta = None
        self.step_authority = None
        self.step_decode_data = None
        self.step_decode_cache_key = None
        self.step_decode_plan_version = -1
        self.step_dispatch_plan = None
        reset_runtime_plan_state(self, epoch=-1)
        self.ordered_layer_data_list = None
        self.ordered_step_decode_data = None
        self.layer_dispatch_cursor = 0
        self.layer_dispatch_epoch = -1
        self.layer_dispatch_layer_count = 0
        self.base_cache_key = None
        self.base_last_seq_len = -1
        self._global_slot_allocator = GlobalSlotAllocator(
            capacity=self._global_slot_allocator_capacity()
        )
        self._pending_global_slot_releases = set()
        self._step_global_slot_map_epoch = -1
        self._step_global_slot_map_req_ids = tuple()
        self._step_global_slot_map = {}
        self._step_dispatcher_slot_row_map_token = -1
        self._step_dispatcher_slot_row_map_key = tuple()
        self._step_dispatcher_slot_row_map = None
        self._step_dispatcher_refresh_dirty_token = -1
        self._step_dispatcher_refresh_dirty_slot_key = tuple()
        self._step_dispatcher_refresh_dirty_payload_slots = tuple()
        self._step_dispatcher_refresh_dirty_any_decode = False
        self._fa3_live_route_token = -1
        self._fa3_live_route_batch_size = -1
        self._fa3_live_route_has_selected_consume = False
        self._fa3_live_route_has_capture = False
        self._fa3_live_route = None
        self._step_context_slot_row_map_token = -1
        self._step_context_slot_row_map_key = tuple()
        self._step_context_slot_row_map = None
        self._step_refresh_slot_req_ids_epoch = -1
        self._step_refresh_slot_req_ids_handle_id = -1
        self._step_refresh_slot_req_ids_handle_generation = -1
        self._step_refresh_slot_req_ids_cache.clear()
        self._step_logits_ready_token = -1
        self._step_logits_ready_input_signature = None
        self._step_logits_ready_bound_signature = None
        self._decode_logf_stage_token = -1
        self._decode_logf_stage_signature = None
        self._decode_logf_stage_bound_signature = None
        self._step_cql_epoch = -1
        self._step_cql_handle_id = -1
        self._step_cql_handle_generation = -1
        self._step_cql_tensor = None

        self.request_states.clear()
        self._request_intent_tickets.clear()
        self._bootstrap_pending_request_ids.clear()
        self._finished_req_ids_step = set()
        self._snapshot_finished_req_ids = set()
        self._finished_generation = 0
        self._finished_boundary_step_token = None
        self._active_step_snapshot = None
        self._active_step_snapshot_epoch = -1
        self._active_step_ticket = None
        self._active_step_source_signature = -1
        self._should_refresh_cache_step = -1
        self._should_refresh_cache_nonce = -1
        self._should_refresh_cache.clear()
        # [SELECTED-OUT-RING] 全量 reset 绕过 clear 漏斗:逐个释放环槽。
        for _srr_pending in self._pending_refresh_rebuilds:
            self._selected_out_ring_release_pending_slot(_srr_pending)
        self._pending_refresh_rebuilds = deque()
        self._pending_refresh_rebuild_by_req.clear()
        self._pending_refresh_rebuild_id = 0
        self._step_refresh_commit_handle_id = -1
        self._step_refresh_commit_handle_generation = -1
        self._step_refresh_commit_written_handle_id = -1
        self._step_refresh_commit_written_handle_generation = -1
        if hasattr(self, "_step_refresh_commit_written_req_ids"):
            self._step_refresh_commit_written_req_ids.clear()
        self._step_refresh_handle_ledger = []
        self._step_refresh_handle_ledger_size = 0
        self._refresh_rebuild_delay_max = 0
        self._refresh_rebuild_delay_max_epoch = -1
        self.step_prefill_epoch = -1
        for buf_id in range(len(self.step_prefill_chunk_payloads)):
            _buf = self.step_prefill_chunk_payloads[buf_id]
            for _i in range(len(_buf)):
                _buf[_i] = None
            self.step_prefill_chunk_mask[buf_id] = 0
        self.step_refresh_epoch = -1
        for buf_id in range(len(self.step_refresh_chunk_payloads)):
            _buf = self.step_refresh_chunk_payloads[buf_id]
            for _i in range(len(_buf)):
                _buf[_i] = None
            self.step_refresh_chunk_mask[buf_id] = 0
        for idx in range(len(self.step_prefill_capture_layout_ring)):
            self.step_prefill_capture_layout_ring[idx] = None
        for idx in range(len(self.step_refresh_capture_layout_ring)):
            self.step_refresh_capture_layout_ring[idx] = None
        for idx in range(len(self._capture_ring_active_lease_by_buf)):
            self._capture_ring_active_lease_by_buf[idx] = None
        self._capture_ring_retired_events.clear()
        self.refresh_stream = None
        self._refresh_stream_device = None
        self.chunk_ready_evt = []
        self.chunk_done_evt = []
        self.prefill_done_evt = []
        self.refresh_done_evt = []
        self._pending_work_reset()
        self._main_wait_epoch = -1
        for i in range(len(self._main_wait_chunk_id_by_buf)):
            self._main_wait_chunk_id_by_buf[i] = -1
        for i in range(len(self._step_wait_consumed_token_by_buf)):
            self._step_wait_consumed_token_by_buf[i] = 0
        self.step_prefill_plan_epoch = -1
        self.step_prefill_capture_plan_by_req = {}
        self.step_prefill_finalize_req_ids = tuple()
        self.step_context_epoch = 0
        self.step_decode_req_meta_i32_all = None
        self.step_decode_req_meta_i64_all = None
        self.step_decode_is_compact_all = None
        self.step_bound_meta = None
        self._step_bound_meta_probe_checked_epoch = -1
        self._step_bound_meta_probe_checked_handle_id = -1
        self._step_bound_meta_probe_checked_handle_generation = -1
        self._step_bound_meta_probe_warned_missing_authority = False
        self.step_decode_compact_kv_len_all = None
        self.step_decode_compact_offset_all = None
        self.step_prefill_req_meta_i32_all = None
        self.step_prefill_req_meta_i64_all = None
        self.prefill_global_meta_epoch = -1
        self.prefill_global_meta_handle_id = -1
        self.prefill_global_meta_handle_generation = -1
        self._prefill_last_n_i32 = None
        self._prefill_cap_i32 = None
        self._prefill_i32_epoch = -1
        self._prefill_i32_handle_id = -1
        self._prefill_i32_handle_generation = -1
        self._prefill_log_f_stride_head = 0
        self._prefill_log_f_stride_epoch = -1
        self._prefill_log_f_stride_handle_id = -1
        self._prefill_log_f_stride_handle_generation = -1
        self._set_unified_attention_mode("default")

    def _maybe_build_step_prefill_global_meta_from_metadata(
        self,
        *,
        attn_metadata: object,
        kv_cache_spec: Optional[object],
    ) -> None:
        _, prefill_impl, _ = _load_metadata_builder_impls()
        return prefill_impl(
            self,
            attn_metadata=attn_metadata,
            kv_cache_spec=kv_cache_spec,
        )

    def _set_unified_attention_mode(self, mode: str) -> None:
        old_mode = _CURRENT_UNIFIED_ATTENTION_MODE
        _set_unified_attention_mode(mode)
        if mode == old_mode:
            return
        self._selector_key_norms_all = None
        self._selector_key_norms_shape = None
        self._selector_capture_scores_all = None
        self._selector_kv_lengths_all = None
        self._selector_key_norms_delta_start_cpu = None
        self._selector_key_norms_delta_end_cpu = None
        self._selector_key_norms_delta_start_gpu = None
        self._selector_key_norms_delta_end_gpu = None
        self._selector_layer_index_cache_key = None
        self._selector_layer_index_cache_device = None
        self._selector_layer_index_cache_tensor = None
        self.kv_cache_block_size = None
        self.kv_cache_num_kv_heads = None
        self.kv_cache_head_dim = None
        self.kv_cache_dtype = None
        self._step_decode_spec_key = None

    def _all_slots_bootstrapped(self, state: LayerState) -> bool:
        if not state.batch_request_ids:
            return False
        for req_id in state.batch_request_ids:
            if _is_free_slot_id(req_id):
                continue
            tracking = self.request_states.get(req_id)
            if tracking is None or (not tracking.bootstrap_done):
                return False
        return True

    def _request_compact_ready_all_layers(self, req_id: str) -> bool:
        """Whether request may consume compact buffers on every registered layer."""
        # [PENDING-FUNNEL-DEBUG 2026-07-09 临时取证探针,破案后拆] 读门 verdict:
        # False 采样 1/32 记首个失败层(reason),True 必记(稀有转折事件)。
        _funnel_dbg = os.environ.get("VLLM_SPARSE_PENDING_FUNNEL_DEBUG_LOG", "")

        def _funnel_note(verdict: bool, reason: str) -> bool:
            if _funnel_dbg:
                n = getattr(self, "_funnel_ready_probe_n", 0) + 1
                self._funnel_ready_probe_n = n
                if verdict or (n % 32 == 1):
                    try:
                        with open(_funnel_dbg, "a") as _fh:
                            _fh.write(
                                f"ready\treq={req_id}\tverdict={verdict}\t"
                                f"reason={reason}\t"
                                f"epoch={int(getattr(self, 'step_context_epoch', -1))}\n"
                            )
                    except OSError:
                        pass
            return verdict

        if _is_free_slot_id(req_id):
            return _funnel_note(False, "free_slot_id")
        tracking = self.request_states.get(req_id)
        if (
            tracking is not None
            and bool(getattr(tracking, "bootstrap_pending", False))
            and bool(getattr(tracking, "bootstrap_bridge_active", False))
            and not bool(getattr(tracking, "bootstrap_done", False))
        ):
            return _funnel_note(False, "bootstrap_bridge_pending")
        if not self.layer_states:
            return _funnel_note(False, "no_layer_states")
        checked_layers = 0
        for state in self.layer_states.values():
            slot = int(state.request_id_to_slot.get(req_id, -1))
            if slot < 0:
                return _funnel_note(
                    False, f"no_slot@L{int(getattr(state, 'layer_index', -1))}"
                )
            if slot >= len(state.compact_kv_len):
                return _funnel_note(
                    False, f"slot_oob@L{int(getattr(state, 'layer_index', -1))}"
                )
            if int(state.compact_kv_len[slot]) <= 0:
                return _funnel_note(
                    False,
                    f"kv_len0@L{int(getattr(state, 'layer_index', -1))}:slot{slot}",
                )
            checked_layers += 1
        return _funnel_note(checked_layers > 0, f"ok_layers={checked_layers}")

    def record_prompt_tokens(self, request_ids: Iterable[str], prompt_lengths: Iterable[int]) -> None:
        for req_id, length in zip(request_ids, prompt_lengths):
            if length is None:
                continue
            length_int = int(length)
            if length_int <= 0:
                continue
            tracking = self._ensure_request(req_id)
            if tracking.total_prompt_tokens != length_int:
                tracking.total_prompt_tokens = length_int
                tracking.last_seq_len = length_int
                tracking.prefill_chunks_seen = 0
                tracking.prompt_chunk_size = 0
                tracking.prefill_capture_rows_accum = 0
                tracking.prefill_capture_prev_capacity = 0
                tracking.prefill_capture_accum_prev_rows = 0
                tracking.prefill_capture_accum_prev_capacity = 0
                tracking.prefill_capture_accum_sit = -1

    def get_or_build_active_step_snapshot(
        self,
        *,
        req_ids: Sequence[str],
        num_scheduled_tokens: Dict[str, int],
        finished_req_ids: Optional[Iterable[str]] = None,
        step_ticket: StepTicket,
    ) -> ActiveStepSnapshot:
        """Step 边界的 active 快照单写者。

        约束：
        1. finished 仅在此入口写入；
        2. 同一 step(tgt epoch + ticket) 复入必须复用同一快照；
        3. 同步骤复入签名漂移直接 fail-fast。
        """
        req_ids_tuple = tuple(req_ids)
        finished_set = self._normalize_finished_request_ids(
            finished_req_ids if finished_req_ids is not None else tuple()
        )
        req_ids_signature_actual = int(hash(req_ids_tuple) & 0x7FFFFFFFFFFFFFFF)

        req_ids_signature = int(step_ticket.req_ids_signature)
        scheduled_signature = int(step_ticket.scheduled_signature)
        finished_signature = int(step_ticket.finished_signature)
        source_signature = int(step_ticket.source_signature)
        if source_signature < 0:
            raise RuntimeError(
                "E_STEP_TICKET_REQUIRED: "
                f"step_ticket.source_signature must be non-negative, got={source_signature}"
            )
        if req_ids_signature != req_ids_signature_actual:
            raise RuntimeError(
                "step ticket signature mismatch: "
                f"ticket=(req={req_ids_signature},sched={scheduled_signature},"
                f"finished={finished_signature}), "
                f"incoming=(req={req_ids_signature_actual},sched=*,finished=*)"
            )

        step_epoch = int(step_ticket.target_epoch)
        cached = self._active_step_snapshot
        if cached is not None and int(cached.step_epoch) == step_epoch:
            cached_ticket = self._active_step_ticket
            cached_source_signature = int(self._active_step_source_signature)
            if cached_ticket != step_ticket or cached_source_signature != source_signature:
                committed_epoch = (
                    self.step_context.epoch
                    if self.step_context is not None
                    else -1
                )
                if committed_epoch >= step_epoch:
                    raise RuntimeError(
                        "same-step snapshot mismatch: "
                        f"expected=(epoch={int(cached.step_epoch)},ticket={cached_ticket},"
                        f"source={cached_source_signature}), "
                        f"got=(epoch={step_epoch},ticket={step_ticket},source={source_signature})"
                    )
                # 同一 target_epoch 但 StepContext 尚未提交：允许覆盖 provisional snapshot。
                self._active_step_snapshot = None
                self._active_step_snapshot_epoch = -1
                self._active_step_ticket = None
                self._active_step_source_signature = -1
                cached = None
            else:
                if self._finished_req_ids_step:
                    raise RuntimeError(
                        "same-step snapshot reentry with non-empty finished cleanup ledger"
                    )
                if self._pending_global_slot_releases:
                    raise RuntimeError(
                        "same-step snapshot reentry with non-empty pending global slot releases"
                    )
                return cached

        expected_epoch = self.step_context_epoch + 1
        if step_epoch != expected_epoch:
            raise RuntimeError(
                "step ticket epoch mismatch: "
                f"ticket_epoch={step_epoch}, expected_epoch={expected_epoch}"
            )

        if finished_set:
            self.mark_finished_requests(finished_set)
        self._run_finished_cleanup_at_step_boundary()

        if self._snapshot_finished_req_ids:
            keep_finished = set(req_ids_tuple)
            if finished_set:
                keep_finished.update(finished_set)
            stale_finished = self._snapshot_finished_req_ids - keep_finished
            if stale_finished:
                self._snapshot_finished_req_ids.difference_update(stale_finished)

        active_row_indices = tuple(
            idx
            for idx, rid in enumerate(req_ids_tuple)
            if rid not in self._snapshot_finished_req_ids
        )
        active_req_ids = tuple(req_ids_tuple[idx] for idx in active_row_indices)
        scheduled_active = tuple(
            int(num_scheduled_tokens.get(rid, 0)) for rid in active_req_ids
        )

        q_start_loc_list: List[int] = [0]
        for scheduled in scheduled_active:
            q_start_loc_list.append(q_start_loc_list[-1] + int(scheduled))
        q_start_loc = tuple(q_start_loc_list)

        req_signature = int(
            hash((active_req_ids, active_row_indices, scheduled_active))
            & 0x7FFFFFFFFFFFFFFF
        )
        snapshot_signature = (
            int(step_epoch),
            int(self._finished_generation),
            int(req_signature),
        )
        snapshot = ActiveStepSnapshot(
            step_epoch=int(step_epoch),
            finished_generation=int(self._finished_generation),
            active_req_ids=active_req_ids,
            active_row_indices=active_row_indices,
            q_start_loc=q_start_loc,
            num_scheduled_tokens=scheduled_active,
            snapshot_signature=snapshot_signature,
        )
        self._active_step_snapshot = snapshot
        self._active_step_snapshot_epoch = int(step_epoch)
        self._active_step_ticket = step_ticket
        self._active_step_source_signature = int(source_signature)
        return snapshot

    def register_block_tables(
        self,
        block_tables: Iterable[torch.Tensor],
        request_ids: Iterable[str],
        *,
        finished_req_ids: Optional[Iterable[str]] = None,
        block_tables_cpu: Optional[Iterable[object]] = None,
    ) -> None:
        _ = finished_req_ids

        ids = list(request_ids)
        block_tables_tuple = tuple(table for table in block_tables if table is not None)
        block_tables_cpu_tuple = (
            tuple(block_tables_cpu) if block_tables_cpu is not None else tuple()
        )
        self._worker_block_table = (
            block_tables_tuple[0] if block_tables_tuple else None
        )
        self._worker_block_table_cpu = (
            block_tables_cpu_tuple[0] if block_tables_cpu_tuple else None
        )
        if not ids:
            return

        # [BLOCK-ROW-MAP-RETIRED 2026-07-05] block_row_to_request（行基址→req_id
        # 映射）已整体退休：全 release 树零生产读者（写 7 处/读仅测试断言 + 自身
        # stale 扫描），compact_recent 退休后的死状态维护——每步白付 _row_keys
        # 基址算术 + active dict 重建 + 全表 stale 扫描（实测 ~5.5µs/步@bs8）。
        # 连带退休 VLLM_SPARSE_SKIP_REDUNDANT_BLOCK_ROW_MAP 旋钮（其精确失效键
        # 只服务于该映射）。保留 _ensure_request（三重覆盖的末重防御：
        # record_prompt_tokens 跳过 length<=0 行、prepare 的 ensure 在
        # has_prompt_counter 分支内——此处兜底所有 ids）。
        for req_id in ids:
            self._ensure_request(req_id)

    def _normalize_finished_request_ids(
        self,
        finished_req_ids: Iterable[str],
    ) -> Set[str]:
        finished_set: Set[str] = set()
        for rid_raw in finished_req_ids:
            if not isinstance(rid_raw, str):
                raise TypeError(
                    f"finished request_id must be str, got {type(rid_raw).__name__}"
                )
            rid = rid_raw.strip()
            if not rid:
                raise ValueError("finished request_id must be non-empty")
            finished_set.add(rid)
        return finished_set

    def consume_finished_at_worker_boundary(
        self,
        *,
        step_token: object,
        finished_req_ids: Optional[Iterable[str]] = None,
    ) -> None:
        """worker 边界 finished 消费入口（单写者）。"""
        finished_set = self._normalize_finished_request_ids(
            finished_req_ids if finished_req_ids is not None else tuple()
        )
        if self._finished_boundary_step_token == step_token:
            if finished_set:
                raise RuntimeError(
                    "duplicate worker-boundary finished consume for same step_token: "
                    f"step_token={step_token!r}, finished_count={len(finished_set)}"
                )
            return
        if finished_set:
            self.mark_finished_requests(finished_set)
        self._run_finished_cleanup_at_step_boundary()
        self._finished_boundary_step_token = step_token

    def mark_finished_requests(self, finished_req_ids: Iterable[str]) -> None:
        """根据 vLLM SchedulerOutput.finished_req_ids 清理已完成请求的状态。

        - finished_req_ids 是“确定已结束”的信号，可靠；
        - 不应根据“本步未调度”推断结束（多 request 下可能只是暂时没被调度）。
        """
        finished_set = self._normalize_finished_request_ids(finished_req_ids)
        if not finished_set:
            return
        # snapshot ledger：仅用于 step snapshot 语义，不受 cleanup 路径影响。
        pending_new = finished_set - self._snapshot_finished_req_ids
        if not pending_new:
            return
        self._snapshot_finished_req_ids.update(pending_new)
        self._finished_generation += 1
        self._active_step_snapshot = None
        self._active_step_snapshot_epoch = -1
        self._active_step_ticket = None
        self._active_step_source_signature = -1
        # cleanup ledger：仅用于跨层 slot 回收消费。
        self._finished_req_ids_step.update(pending_new)
        self._pending_global_slot_releases.update(pending_new)
        self._step_global_slot_map_epoch = -1
        self._step_global_slot_map_req_ids = tuple()
        self._step_global_slot_map = {}
        for rid in pending_new:
            self._bootstrap_pending_request_ids.discard(rid)
            if rid in self.request_states:
                del self.request_states[rid]
            if rid in self._request_intent_tickets:
                del self._request_intent_tickets[rid]

    def _run_finished_cleanup_at_step_boundary(self) -> None:
        """Step 边界消费 finished cleanup ledger（单写者）。"""
        finished_ids = set(self._finished_req_ids_step)
        step_handle_id = int(getattr(self, "_step_refresh_commit_handle_id", -1))
        step_handle_generation = int(
            getattr(self, "_step_refresh_commit_handle_generation", -1)
        )
        step_epoch = int(getattr(self, "step_context_epoch", -1))
        if not finished_ids:
            if self._pending_global_slot_releases:
                raise RuntimeError(
                    "finished cleanup boundary invariant violated: "
                    "pending_global_slot_releases is non-empty while finished ledger is empty"
                )
            return

        if _is_stream_capturing_or_raise(stage="finished_cleanup_step_boundary"):
            raise RuntimeError(
                "finished cleanup cannot run during CUDA graph capture at step boundary"
            )

        self._pending_refresh_rebuild_finish_boundary_drain(
            finished_ids=finished_ids,
        )

        finished_pending = [
            rid
            for rid in finished_ids
            if self._pending_refresh_rebuild_has_req(rid)
        ]
        if finished_pending:
            raise RuntimeError(
                "E_PENDING_REBUILD_ON_FINISH: "
                "finished cleanup boundary sees pending refresh-rebuild reqs: "
                f"step_handle_id={step_handle_id} "
                f"generation={step_handle_generation} "
                f"epoch={step_epoch} "
                f"count={len(finished_pending)} sample={finished_pending[:4]}"
            )

        if self._async_refresh_enabled():
            if self.chunk_done_evt:
                cleanup_device = self.device
                if cleanup_device is None:
                    for state in self.layer_states.values():
                        device = state.device
                        if isinstance(device, torch.device):
                            cleanup_device = device
                            break
                if cleanup_device is None:
                    raise RuntimeError(
                        "E_PENDING_REBUILD_ON_FINISH: "
                        "finished cleanup boundary missing device while async events exist: "
                        f"step_handle_id={step_handle_id} "
                        f"generation={step_handle_generation} "
                        f"epoch={step_epoch}"
                    )
                for buf in range(_CAPTURE_IN_FLIGHT):
                    self._main_stream_wait_for_chunk_done(
                        buf_id=buf,
                        device=cleanup_device,
                        epoch=self.step_context_epoch,
                    )
                pending_flags = [int(v) for v in self._buf_pending_work_flags]
                remaining_bufs = [
                    idx for idx, flag in enumerate(pending_flags) if int(flag) != 0
                ]
                if remaining_bufs:
                    raise RuntimeError(
                        "E_PENDING_REBUILD_ON_FINISH: "
                        "finished cleanup boundary wait did not clear pending work flags: "
                        f"step_handle_id={step_handle_id} "
                        f"generation={step_handle_generation} "
                        f"epoch={step_epoch} "
                        f"remaining_bufs={remaining_bufs} flags={pending_flags}"
                    )

        for state in self.layer_states.values():
            _cleanup_inactive_slots(state, self)

        for rid in list(self._pending_global_slot_releases):
            self.release_global_slot(rid)
            self._pending_global_slot_releases.discard(rid)

        if self._pending_global_slot_releases:
            raise RuntimeError(
                "finished cleanup boundary failed: pending_global_slot_releases not empty"
            )
        leaks = [
            rid for rid in finished_ids
            if self._global_slot_allocator.slot_of(rid) is not None
        ]
        if leaks:
            raise RuntimeError(
                "finished cleanup boundary failed to release global slots: "
                f"sample={leaks[:4]} count={len(leaks)}"
            )
        self._finished_req_ids_step = set()


    def evaluate_tp_sentence_token_window(
        self,
        request_id: str,
        token_window: Tuple[torch.Tensor, int, int, int],
    ) -> int:
        token_ids_cpu, row_idx, start, end = token_window
        start = int(start)
        end = int(end)
        row_idx = int(row_idx)
        total_rows = int(token_ids_cpu.shape[0]) if hasattr(token_ids_cpu, "shape") else len(token_ids_cpu)
        expected_rows = int(getattr(self, "_worker_token_source_rows", -1))
        if expected_rows >= 0 and total_rows != expected_rows:
            raise RuntimeError(
                "E_TP2_TOKEN_SOURCE_INCOMPLETE: worker token source row-count changed in-step "
                f"for request {request_id!r}, expected_rows={expected_rows}, rows={total_rows}"
            )
        # [DEAD-SENTINEL-RETIRE] version/data_ptr 两支已删除：token 源是 numpy 视图
        # （无 _version/data_ptr），expected 恒 -1 使两门自设计起恒短路——零保护的
        # 虚假合同。活保护 = 上面的 rows 哨兵 + 下方实值前缀/FIFO 序校验（值域层面）。
        if row_idx < 0 or row_idx >= total_rows:
            raise RuntimeError(
                "E_TP2_TOKEN_SOURCE_INCOMPLETE: worker row index out of range "
                f"for request {request_id!r}, row_idx={row_idx}, rows={total_rows}"
            )
        row_tokens = token_ids_cpu[row_idx]
        row_len = int(row_tokens.shape[0]) if hasattr(row_tokens, "shape") else len(row_tokens)
        if row_len <= 0:
            return 0
        if start < 0:
            start = 0
        if end > row_len:
            end = row_len
        if end <= start:
            return 0
        tokens = [int(row_tokens[pos]) for pos in range(start, end)]
        # [TP-ASYNC-HARVEST] 实值前缀截断：async 调度下窗口尾部允许还未被修复
        # 通道回填的 -1 占位符——喂入实值前缀，水位只推进已喂入部分，余量下一步
        # 接续（与 TP=1 enginecore hook 的事后补喂同构，1 步滞后）。合同保牙：
        # ①sync 模式（无 stash）下 TP>1 出现占位符仍然 fail-fast（原合同）；
        # ②占位符之后出现实值 = FIFO 修复序被破坏，无条件 fail-fast。
        n_real = 0
        for token in tokens:
            if token < 0:
                break
            n_real += 1
        if n_real < len(tokens):
            if any(token >= 0 for token in tokens[n_real + 1 :]):
                raise RuntimeError(
                    "E_TP2_TOKEN_SOURCE_INCOMPLETE: worker token source has a real "
                    f"token after a placeholder for request {request_id!r} "
                    f"(window=[{start},{end}), n_real={n_real}); repair order corrupted"
                )
            _async_mode = getattr(self, "_async_sampled_stash", None) is not None
            if not _async_mode and int(getattr(self, "tp_size", 1) or 1) > 1:
                raise RuntimeError(
                    "E_TP2_TOKEN_SOURCE_INCOMPLETE: worker token source contains "
                    f"placeholder decode token(s) for request {request_id!r} "
                    "under synchronous scheduling"
                )
        if n_real == 0:
            return 0
        # 由 worker 传窗口描述符；触发判定仍复用 record_generated_tokens 语义。
        self.record_generated_tokens(request_id, tokens[:n_real])
        return n_real


    def record_generated_tokens(self, request_id: str, token_ids: Iterable[int]) -> None:
        tracking = self._ensure_request(request_id)
        tokens = token_ids if isinstance(token_ids, list) else list(token_ids)
        if not tokens:
            return
        intent_changed = False
        # 确保 decode 位置基于“完整上下文长度”（prompt + 已生成 token）。
        # 若 prefill 路径未能写入 last_seq_len，则回退到已知的 prompt 长度，
        # 否则 recent/persist 只会覆盖 0..len_decode 范围，导致 compact 与 dense 严重偏移。
        if tracking.total_prompt_tokens > 0 and tracking.last_seq_len < tracking.total_prompt_tokens:
            _log.warning(
                "last_seq_len(%d) < total_prompt_tokens(%d) for %s; "
                "prefill path may not have written last_seq_len correctly",
                tracking.last_seq_len, tracking.total_prompt_tokens, request_id,
            )
            tracking.last_seq_len = tracking.total_prompt_tokens

        # 仅统计 decode 阶段的步数，避免 prompt 长度影响 refresh 间隔
        prev_step = int(tracking.decode_step) if tracking.decode_step is not None else -1
        start_step = prev_step + 1
        trigger = tracking.trigger
        for idx, token in enumerate(tokens):
            if trigger is not None:
                step = start_step + idx
                # TP SAFETY: sentence trigger decisions must be deterministic across
                # TP ranks. prepare_step_context ensures TP-consistent token feed via
                # _worker_token_ids_cpu. This function must be called AFTER token sync
                # to avoid different ranks producing different _refresh_nonce values.
                should, reason = trigger.should_refresh(
                    step=step,
                    cache_position=tracking.last_seq_len,
                    token_id=int(token),
                )
                if should:
                    reason_str = str(reason or "trigger")
                    reason_code = int(pending_reason_to_code(reason_str))
                    sentence_reason: Optional[str] = (
                        reason_str
                        if reason_code == int(PendingReasonCode.SENTENCE)
                        else None
                    )
                    # token-time 仅记录轻量 intent；pending ticket 统一由 planner 在 step 边界落票。
                    if sentence_reason is not None:
                        intent_changed = self._set_request_trigger_intent(
                            request_id=request_id,
                            reason=sentence_reason,
                            decode_step=int(step),
                            force_now=True,
                        ) or bool(intent_changed)
        tracking.decode_step = start_step + len(tokens) - 1
        if intent_changed:
            self._bump_refresh_nonce()

        prev_len = tracking.last_seq_len if tracking.last_seq_len is not None else 0
        curr_len = prev_len + len(tokens)
        tracking.last_seq_len = curr_len

    def get_state(
        self,
        cache_key: int,
        num_heads: int,
        num_kv_heads: int,
        device: torch.device,
        head_dim: Optional[int] = None,
        kv_cache_dtype: Optional[torch.dtype] = None,
    ) -> LayerState:
        self._last_seen_device = device
        state = self.layer_states.get(cache_key)
        if state is None:
            state = self._register_layer(
                cache_key,
                num_heads,
                num_kv_heads,
                device,
                head_dim=head_dim,
                kv_cache_dtype=kv_cache_dtype,
            )
        else:
            if head_dim is not None and state.head_dim is None:
                state.head_dim = int(head_dim)
            if kv_cache_dtype is not None and state.kv_cache_dtype is None:
                state.kv_cache_dtype = kv_cache_dtype
        return state

    def begin_step_if_needed(self, cache_key: int, seq_len: int) -> None:
        if self.base_cache_key is None:
            self.base_cache_key = cache_key
            self.base_last_seq_len = seq_len
            self._should_refresh_cache_step = self.step_context_epoch
            self._should_refresh_cache_nonce = self._refresh_nonce
            self._should_refresh_cache.clear()
            return
        if cache_key != self.base_cache_key:
            return
        if seq_len != self.base_last_seq_len:
            self.base_last_seq_len = seq_len
            self._should_refresh_cache_step = self.step_context_epoch
            self._should_refresh_cache_nonce = self._refresh_nonce
            self._should_refresh_cache.clear()

    def plan_refresh_requests(
        self,
        request_ids: Tuple[str, ...],
        *,
        use_cache: bool = True,
        update_state: bool = True,
        allow_materialize: bool = True,
    ) -> StepRefreshPlan:
        """按 request_id 计划本步 refresh 请求列表（step-wise, single-source plan）。"""
        execution_class = "exec" if (update_state and allow_materialize) else "non_exec"
        cache_key = (request_ids, execution_class)
        if use_cache:
            refresh_nonce = self._refresh_nonce
            if (
                self._should_refresh_cache_step != self.step_context_epoch
                or self._should_refresh_cache_nonce != refresh_nonce
            ):
                self._should_refresh_cache_step = self.step_context_epoch
                self._should_refresh_cache_nonce = refresh_nonce
                self._should_refresh_cache.clear()
            cached = self._should_refresh_cache.get(cache_key)
            if cached is not None:
                return cached

        if not request_ids:
            result = StepRefreshPlan(
                epoch=self.step_context_epoch,
                req_ids=tuple(),
                mode_by_row=tuple(),
                refresh_rows=tuple(),
                refresh_reqs=tuple(),
                refresh_reason="compact",
                bootstrap_done=False,
                plan_signature=(
                    self.step_context_epoch,
                    tuple(),
                    tuple(),
                    tuple(),
                    "compact",
                    False,
                ),
            )
            self._should_refresh_cache[cache_key] = result
            return result

        tracking_by_req: Dict[str, RequestTracking] = {
            rid: self._ensure_request(rid) for rid in request_ids
        }
        tickets_by_req: Dict[str, RequestIntentTicket] = {
            rid: self._ensure_request_ticket(rid) for rid in request_ids
        }
        tickets_plan_by_req: Dict[str, RequestIntentTicket]
        if update_state:
            tickets_plan_by_req = tickets_by_req
        else:
            tickets_plan_by_req = {
                rid: RequestIntentTicket(
                    state=ticket.state,
                    pending_refresh=ticket.pending_refresh,
                    pending_reason_code=ticket.pending_reason_code,
                    pending_decode_step=ticket.pending_decode_step,
                    pending_ctrl_step=ticket.pending_ctrl_step,
                    pending_policy=ticket.pending_policy,
                )
                for rid, ticket in tickets_by_req.items()
            }
        pending_updates: Dict[str, tuple[int, int, int, int, bool]] = {}

        def _queue_set_pending_refresh(
            *,
            request_id: str,
            reason_code: Optional[int] = None,
            reason: Optional[str] = None,
            decode_step: int,
            pending_policy: Optional[int] = None,
            pending_ctrl_step: Optional[int] = None,
        ) -> bool:
            ticket_local = tickets_plan_by_req[request_id]
            if reason_code is None:
                reason_code_local = pending_reason_to_code(reason or "refresh")
            else:
                reason_code_local = int(reason_code)
            step_local = decode_step
            if pending_policy is None:
                policy_local = ticket_local.pending_policy
            else:
                policy_local = pending_policy
            ctrl_step_cur_local = ticket_local.pending_ctrl_step
            if pending_ctrl_step is None:
                if ticket_local.pending_refresh and ctrl_step_cur_local >= 0:
                    ctrl_step_local = ctrl_step_cur_local
                else:
                    ctrl_step_local = self.step_context_epoch
            else:
                ctrl_step_local = pending_ctrl_step
            changed_local = (
                (not ticket_local.pending_refresh)
                or ticket_local.pending_reason_code != reason_code_local
                or ticket_local.pending_decode_step != step_local
                or ticket_local.pending_ctrl_step != ctrl_step_local
                or ticket_local.pending_policy != policy_local
                or ticket_local.state != TicketState.PENDING_REFRESH
            )
            ticket_local.pending_refresh = True
            ticket_local.pending_reason_code = reason_code_local
            ticket_local.pending_decode_step = step_local
            ticket_local.pending_ctrl_step = ctrl_step_local
            ticket_local.pending_policy = policy_local
            ticket_local.state = TicketState.PENDING_REFRESH
            if update_state:
                pending_updates[request_id] = (
                    reason_code_local,
                    step_local,
                    policy_local,
                    ctrl_step_local,
                    False,
                )
            return changed_local

        def _queue_clear_pending_refresh(*, request_id: str, ready_compact: bool) -> bool:
            ticket_local = tickets_plan_by_req[request_id]
            next_state_local = (
                TicketState.READY_COMPACT if ready_compact else TicketState.NOT_READY
            )
            changed_local = (
                ticket_local.pending_refresh
                or ticket_local.pending_reason_code != PendingReasonCode.NONE
                or ticket_local.pending_decode_step != -1
                or ticket_local.pending_ctrl_step != -1
                or ticket_local.pending_policy != PendingPolicy.COALESCEABLE
                or ticket_local.state != next_state_local
            )
            ticket_local.pending_refresh = False
            ticket_local.pending_reason_code = PendingReasonCode.NONE
            ticket_local.pending_decode_step = -1
            ticket_local.pending_ctrl_step = -1
            ticket_local.pending_policy = PendingPolicy.COALESCEABLE
            ticket_local.state = next_state_local
            if update_state:
                pending_updates[request_id] = (
                    int(PendingReasonCode.NONE),
                    -1,
                    PendingPolicy.COALESCEABLE,
                    -1,
                    ready_compact,
                )
            return changed_local

        bootstrap_done = True
        for rid in request_ids:
            tracking = tracking_by_req[rid]
            if not tracking.bootstrap_done:
                bootstrap_done = False
                break

        if (
            bool(getattr(self.config, "one_shot_bootstrap_only", False))
            and bootstrap_done
            and not continuous_producer_enabled(self.config)
        ):
            mode_by_row = tuple(StepRefreshMode.NONE for _ in request_ids)
            pending_cleared = False
            for rid in request_ids:
                pending_cleared = _queue_clear_pending_refresh(
                    request_id=rid,
                    ready_compact=True,
                ) or pending_cleared
            if update_state:
                for rid, update in pending_updates.items():
                    (
                        reason_code_update,
                        step_update,
                        policy_update,
                        ctrl_update,
                        clear_ready_compact,
                    ) = update
                    if clear_ready_compact:
                        self._clear_request_pending_refresh(
                            request_id=rid,
                            ready_compact=True,
                        )
                    else:
                        self._set_request_pending_refresh(
                            request_id=rid,
                            reason_code=reason_code_update,
                            decode_step=step_update,
                            pending_policy=policy_update,
                            pending_ctrl_step=ctrl_update,
                        )
            result = StepRefreshPlan(
                epoch=self.step_context_epoch,
                req_ids=request_ids,
                mode_by_row=mode_by_row,
                refresh_rows=tuple(),
                refresh_reqs=tuple(),
                refresh_reason="one_shot_bootstrap_only",
                bootstrap_done=True,
                plan_signature=(
                    self.step_context_epoch,
                    request_ids,
                    mode_by_row,
                    tuple(),
                    tuple(),
                    "one_shot_bootstrap_only",
                    True,
                ),
            )
            self._should_refresh_cache[cache_key] = result
            if update_state and pending_cleared:
                self._bump_refresh_nonce()
            return result

        interval = self.config.refresh_interval
        refresh_set: Set[str] = set()
        replay_forced_refresh_set: Set[str] = set()
        decode_step_by_req: Dict[str, int] = {}
        inflight_by_req: Dict[str, bool] = {}
        pending_rebuild_inflight_by_req: Dict[str, bool] = {}
        pending_reason_code: Optional[int] = None
        interval_reason_code: Optional[int] = None
        earliest_pending_decode: Optional[int] = None
        last_reason: str = "none"
        pending_cleared = False
        coalesced = False
        inflight_dense_consume_set: Set[str] = set()
        workload_plan_replay_active = self._workload_plan_replay is not None

        def _is_threshold_crossing_force_pending(ticket_local: RequestIntentTicket) -> bool:
            return bool(
                ticket_local.pending_refresh
                and int(ticket_local.pending_policy) == int(PendingPolicy.FORCE_NOW)
                and int(ticket_local.pending_reason_code)
                == int(PendingReasonCode.COMPACT_THRESHOLD_CROSSED)
            )

        def _is_short_dense_blocked(rid_local: str) -> bool:
            tracking_local = tracking_by_req[rid_local]
            if not bool(getattr(tracking_local, "_was_short_dense", False)):
                return False
            ticket_local = tickets_plan_by_req[rid_local]
            # short 阶段仅放行 crossing 的 FORCE_NOW refresh。
            return not _is_threshold_crossing_force_pending(ticket_local)

        def _refresh_gap_blocked(
            tracking_local: RequestTracking,
            decode_step_local: int,
        ) -> bool:
            # [TP-DET-TRIGGER 2026-07-07] 决定论触发挡板:自上次提交(enqueue
            # commit 点推进 last_decode_refresh_step,票面计划步)起
            # min_refresh_gap 步内不触发/不拉入——替代原 inflight(scheduled/
            # GPU-writer 完成时序)挡板。输入全部 TP-rank 一致(step/提交历史),
            # 与用户设计合同同构(任意两次 refresh ≥ min_refresh_gap,跨 reason)。
            trigger_local = getattr(tracking_local, "trigger", None)
            if trigger_local is not None:
                gap_local = max(
                    0,
                    int(
                        getattr(
                            trigger_local.config,
                            "min_refresh_gap",
                            DEFAULT_MIN_REFRESH_GAP,
                        )
                        or 0
                    ),
                )
            else:
                # refresh-on(纯 interval)形态 trigger 缺席:挡板不可失效,
                # fallback 全局默认(lease/coalesce 线依赖它防重复提交)。
                gap_local = DEFAULT_MIN_REFRESH_GAP
            if gap_local <= 0 or decode_step_local < 0:
                return False
            last_local = int(
                getattr(tracking_local, "last_decode_refresh_step", -1) or -1
            )
            if last_local < 0:
                return False
            return (decode_step_local - last_local) < gap_local

        def _sentence_intent_covered_by_existing_pending(
            rid_local: str,
            tracking_local: RequestTracking,
            ticket_local: RequestIntentTicket,
        ) -> bool:
            if tracking_local.trigger_intent_decode_step < 0:
                return False
            if (
                pending_reason_to_code(tracking_local.trigger_intent_reason or "none")
                != int(PendingReasonCode.SENTENCE)
            ):
                return False
            if not ticket_local.pending_refresh:
                return False
            if int(ticket_local.pending_reason_code) != int(PendingReasonCode.INTERVAL):
                return False
            # [TP-DET-TRIGGER] 票在提交点即转 consumed,存在的 INTERVAL 票必为
            # defer 残留(未提交)——sentence 一律并入;原判定读 GPU writer 完成
            # 态(per-rank 异步,决策禁用)已移除。
            return True

        def _has_interval_pending_not_ready(
            rid_local: str,
            ticket_local: RequestIntentTicket,
        ) -> bool:
            del rid_local
            if not ticket_local.pending_refresh:
                return False
            # [TP-DET-TRIGGER] 同上:有 INTERVAL 票(defer 残留)即跳过 coalesce
            # 拉入,不查 GPU ready。
            return int(ticket_local.pending_reason_code) == int(
                PendingReasonCode.INTERVAL
            )

        def _has_sentence_refresh_due_at_target(
            rid_local: str,
            target_step: int,
        ) -> bool:
            if target_step < 0:
                return False
            ticket_local = tickets_plan_by_req[rid_local]
            if (
                ticket_local.pending_refresh
                and int(ticket_local.pending_reason_code)
                == int(PendingReasonCode.SENTENCE)
                and int(ticket_local.pending_decode_step) == target_step
            ):
                return True
            tracking_local = tracking_by_req[rid_local]
            return (
                int(getattr(tracking_local, "trigger_intent_decode_step", -1))
                == target_step
                and pending_reason_to_code(
                    getattr(tracking_local, "trigger_intent_reason", "none")
                    or "none"
                )
                == int(PendingReasonCode.SENTENCE)
            )

        def _post_bridge_sentence_due_near_target(
            rid_local: str,
            *,
            decode_step_local: int,
            target_step: int,
            window: int,
        ) -> int:
            if target_step < 0 or decode_step_local < 0:
                return -1
            tracking_local = tracking_by_req[rid_local]
            due_step = int(
                getattr(tracking_local, "post_bridge_refresh_due_decode_step", -1)
            )
            if due_step < 0:
                return -1
            # [TP-DET-TRIGGER] 原此处读 GPU writer 完成态作为拉入前置(per-rank
            # 异步,决策禁用)→ 换决定论 gap 挡板(提交历史):gap 内不拉入。
            if _refresh_gap_blocked(tracking_local, decode_step_local):
                return -1
            last_decode_refresh_local = int(
                getattr(tracking_local, "last_decode_refresh_step", -1)
            )
            if last_decode_refresh_local >= due_step:
                return -1
            coalesce_window_i = max(0, int(window))
            if abs(due_step - int(target_step)) > coalesce_window_i:
                return -1
            # A post-bridge refresh is launched after the current graph replay.
            # Allow at most one pre-call decode-step of lookahead so the
            # current post-call can materialize the due step without creating
            # an earlier semantic commit.
            if due_step - int(decode_step_local) > coalesce_window_i:
                return -1
            return int(due_step)

        def _post_bridge_sentence_near_target_coalesce_enabled() -> bool:
            if pending_reason_code != int(PendingReasonCode.SENTENCE):
                return False
            layer_count = len(getattr(self, "layer_cache_keys", ()) or ())
            if layer_count <= 0 or int(_CAPTURE_CHUNK) < int(layer_count):
                return False
            if (
                os.environ.get(
                    "VLLM_SPARSE_REPLAY_REFRESH_PROGRESSIVE_CONSUME",
                    "0",
                )
                != "1"
            ):
                return False
            try:
                from patches.refresh_runtime.producer_ready import (
                    resolve_one_shot_ready_chunk,
                    validate_one_shot_ready_chunk_alignment,
                )

                ready_chunk = resolve_one_shot_ready_chunk(
                    capture_chunk=int(_CAPTURE_CHUNK),
                )
                validate_one_shot_ready_chunk_alignment(
                    capture_chunk=int(_CAPTURE_CHUNK),
                    ready_chunk=int(ready_chunk),
                )
            except Exception:
                raise
            return int(ready_chunk) < int(_CAPTURE_CHUNK)

        # [CREDIT-RETIRE 2026-07-07] _post_bridge_refresh_credit_active/_due
        # 与 _reset_sentence_trigger_refresh_gap 已随 credit 状态机整机退休
        # (见 sentence intent 落票处注记);gap 归零由世代完成路径
        # (selector_compute_mixin 更新 tracking 时)统一执行。

        # [TP-DET-TRIGGER 2026-07-07] _pending_refresh_covered_by_ready_compact
        # 已随 covered 清票段整体退休:票在 enqueue commit 点转 consumed,不再
        # 存在"挂着等 GPU writer 完成"的票形态(其判定读 GPU 完成态,是 TP>1
        # 决策发散根之一)。

        def _pending_rebuild_ticket_decode_step(
            rid_local: str,
            ticket_local: RequestIntentTicket,
        ) -> int:
            if not self._pending_refresh_rebuild_has_req(rid_local):
                return -1
            pending_decode = int(ticket_local.pending_decode_step)
            if pending_decode < 0:
                return -1
            pending_ctrl = int(ticket_local.pending_ctrl_step)
            if pending_ctrl >= 0 and pending_ctrl > self.step_context_epoch:
                return -1
            return pending_decode

        def _sentence_intent_covered_by_inflight_refresh(
            rid_local: str,
            tracking_local: RequestTracking,
            ticket_local: RequestIntentTicket,
            pending_rebuild_inflight_local: bool,
        ) -> bool:
            intent_step = int(tracking_local.trigger_intent_decode_step)
            if intent_step < 0:
                return False
            if (
                pending_reason_to_code(tracking_local.trigger_intent_reason or "none")
                != int(PendingReasonCode.SENTENCE)
            ):
                return False
            scheduled_decode = int(
                getattr(tracking_local, "scheduled_decode_refresh_step", -1)
            )
            if scheduled_decode < 0 and pending_rebuild_inflight_local:
                scheduled_decode = _pending_rebuild_ticket_decode_step(
                    rid_local,
                    ticket_local,
                )
            if scheduled_decode < 0 and pending_rebuild_inflight_local:
                return True
            return scheduled_decode >= intent_step

        def _pending_dense_consume_guard_active() -> bool:
            layer_count = len(getattr(self, "layer_cache_keys", ()) or ())
            if layer_count <= 0:
                return False
            try:
                from patches.refresh_runtime.producer_ready import (
                    resolve_one_shot_ready_chunk,
                    validate_one_shot_ready_chunk_alignment,
                )

                ready_chunk = resolve_one_shot_ready_chunk(
                    capture_chunk=int(_CAPTURE_CHUNK),
                )
                validate_one_shot_ready_chunk_alignment(
                    capture_chunk=int(_CAPTURE_CHUNK),
                    ready_chunk=int(ready_chunk),
                )
                if int(ready_chunk) < int(_CAPTURE_CHUNK):
                    return False
            except Exception:
                raise
            if (
                os.environ.get(
                    "VLLM_SPARSE_REPLAY_REFRESH_PROGRESSIVE_CONSUME",
                    "0",
                )
                == "1"
            ):
                return False
            return int(_CAPTURE_CHUNK) >= int(layer_count)

        def _dual_gen_inflight_compact_readable(
            tracking_local: RequestTracking,
        ) -> bool:
            # [DUAL-GEN-L2b] 双代下 INFLIGHT 免 dense 的前提=行有可读旧代:
            # writer 写备用半区、读侧继续消费旧代 compact,writer_done 后原子
            # 切代——in-place torn-read 窗(dense 闸的存在理由)不复存在。
            # bootstrap 首刷(从未有过 compact 内容)没有旧代,必须保留 dense。
            if compact_gen_count() <= 1:
                return False
            return bool(tracking_local.bootstrap_done)

        def _pending_rebuild_requires_dense_consume(
            rid_local: str,
            tracking_local: RequestTracking,
            ticket_local: RequestIntentTicket,
            pending_rebuild_inflight_local: bool,
        ) -> bool:
            if not pending_rebuild_inflight_local:
                return False
            # [TP-DET-TRIGGER] 票在 commit 转 consumed;在飞窗的 reason/policy
            # 由读侧镜像(commit 写/publish final 清)承载——防 torn-read 的
            # dense 闸不得因票提前清除而失效(4B 实测 illegal address 教训)。
            if ticket_local.pending_refresh:
                reason_code_local = int(ticket_local.pending_reason_code)
                policy_local = int(ticket_local.pending_policy)
            else:
                reason_code_local = int(
                    getattr(tracking_local, "inflight_reason_code", -1)
                )
                policy_local = int(getattr(tracking_local, "inflight_policy", -1))
            if reason_code_local >= 0 or policy_local >= 0:
                if reason_code_local == int(PendingReasonCode.LEASE_REARM):
                    # [DUAL-GEN-L2b] 容量重排可能整体 reset 旧代内容,
                    # 双代不豁免。
                    return True
                if policy_local == int(PendingPolicy.FORCE_NOW):
                    if _dual_gen_inflight_compact_readable(tracking_local):
                        return False
                    return True
                if reason_code_local in (
                    int(PendingReasonCode.SENTENCE),
                    int(PendingReasonCode.COMPACT_THRESHOLD_CROSSED),
                ):
                    if _dual_gen_inflight_compact_readable(tracking_local):
                        return False
                    return True
            if _sentence_intent_covered_by_inflight_refresh(
                rid_local,
                tracking_local,
                ticket_local,
                pending_rebuild_inflight_local,
            ):
                if _dual_gen_inflight_compact_readable(tracking_local):
                    return False
                return True
            return False

        def _has_pending_refresh_work_ledger() -> bool:
            flags_local = getattr(self, "_buf_pending_work_flags", ())
            for flag in flags_local:
                try:
                    if (int(flag) & 2) != 0:
                        return True
                except Exception:
                    continue
            return False

        def _request_has_pending_rebuild_inflight(
            rid_local: str,
            tracking_local: RequestTracking,
            ticket_local: RequestIntentTicket,
        ) -> bool:
            if not self._pending_refresh_rebuild_has_req(rid_local):
                return bool(
                    ticket_local.pending_refresh
                    and _has_pending_refresh_work_ledger()
                )
            scheduled_ctrl = int(
                getattr(tracking_local, "scheduled_refresh_ctrl_step", -1)
            )
            scheduled_decode = int(
                getattr(tracking_local, "scheduled_decode_refresh_step", -1)
            )
            if scheduled_decode < 0:
                scheduled_decode = _pending_rebuild_ticket_decode_step(
                    rid_local,
                    ticket_local,
                )
            if scheduled_decode < 0:
                return scheduled_ctrl < 0 or scheduled_ctrl <= self.step_context_epoch
            return (
                (scheduled_ctrl < 0 or scheduled_ctrl <= self.step_context_epoch)
                and scheduled_decode >= 0
            )

        def _record_sentence_trigger_admission_coalesced(
            detail_counter_attr: str,
        ) -> None:
            self._sentence_trigger_admission_coalesced_total = (
                int(
                    getattr(
                        self,
                        "_sentence_trigger_admission_coalesced_total",
                        0,
                    )
                )
                + 1
            )
            setattr(
                self,
                detail_counter_attr,
                int(getattr(self, detail_counter_attr, 0)) + 1,
            )

        for request_ordinal, rid in enumerate(request_ids):
            tracking = tracking_by_req[rid]
            ticket = tickets_plan_by_req[rid]
            decode_step = int(tracking.decode_step) if tracking.decode_step is not None else -1
            decode_step_by_req[rid] = decode_step
            scheduled_ctrl = tracking.scheduled_refresh_ctrl_step
            pending_rebuild_inflight = _request_has_pending_rebuild_inflight(
                rid,
                tracking,
                ticket,
            )
            pending_rebuild_inflight_by_req[rid] = pending_rebuild_inflight
            inflight_refresh = (
                scheduled_ctrl >= 0 and scheduled_ctrl < self.step_context_epoch
            ) or pending_rebuild_inflight
            inflight_by_req[rid] = inflight_refresh
            if _pending_rebuild_requires_dense_consume(
                rid,
                tracking,
                ticket,
                pending_rebuild_inflight,
            ):
                inflight_dense_consume_set.add(rid)
            last_decode_refresh = (
                int(tracking.last_decode_refresh_step)
                if tracking.last_decode_refresh_step is not None
                else -1
            )
            # 第一次 decode 时，避免用 prompt 长度触发 interval：若从未 refresh，视为“已在本步 refresh”
            if last_decode_refresh < 0 and decode_step >= 0:
                # prepare_step_context 已负责初始化 tracking.last_decode_refresh_step；
                # 此处仅用局部变量兜底本函数内的 interval 计算。
                last_decode_refresh = decode_step

            # short 阶段：禁止 refresh/compact 构建相关计划，仅保留 crossing 的 FORCE_NOW。
            if _is_short_dense_blocked(rid):
                if ticket.pending_refresh:
                    _queue_clear_pending_refresh(
                        request_id=rid,
                        ready_compact=False,
                    )
                    ticket = tickets_plan_by_req[rid]
                if update_state and tracking.trigger_intent_decode_step >= 0:
                    self._clear_request_trigger_intent(request_id=rid)
                if update_state and tracking.lease_rearm:
                    self._clear_request_lease_rearm(request_id=rid)
                continue

            if workload_plan_replay_active:
                # Replay is benchmark control data: planned events must fire
                # even if a candidate keeps producer work in flight longer than
                # the recording run. Live mode still uses the inflight/coalesce
                # guards below.
                for planned_event in indexed_workload_plan_events_for_request_step(
                    self._workload_plan_replay_events_by_key,
                    request_ordinal=int(request_ordinal),
                    decode_step=int(decode_step),
                ):
                    planned_reason = str(planned_event.get("reason", "refresh") or "refresh")
                    planned_policy = workload_plan_pending_policy(planned_event)
                    _queue_set_pending_refresh(
                        request_id=rid,
                        reason=planned_reason,
                        decode_step=int(decode_step),
                        pending_policy=planned_policy,
                        pending_ctrl_step=self.step_context_epoch,
                    )
                    ticket = tickets_plan_by_req[rid]
                    refresh_set.add(rid)
                    replay_forced_refresh_set.add(rid)
                    self._workload_plan_replay_injected_count += 1
                    if earliest_pending_decode is None or decode_step < earliest_pending_decode:
                        earliest_pending_decode = int(decode_step)
                    if pending_reason_code is None:
                        pending_reason_code = int(ticket.pending_reason_code)

            # token-time trigger intent 在 step 边界统一落票，避免 token-time 直接写 ticket。
            # [TP-DET-TRIGGER 2026-07-07] 旧 intent 吸收判定决定论化:原判定
            # `inflight ∧ scheduled_decode ≥ intent_step`(GPU 完成时序决定
            # inflight/scheduled 存续,per-rank 异步)→ 等价替换为
            # `intent_step ≤ last_decode_refresh_step`(last 在提交点=票面计划
            # 步,与原 scheduled 同值,但推进时刻决定论)。语义不变:已提交世代
            # 的计划步覆盖了不晚于它的句边界。
            _intent_step_probe = int(tracking.trigger_intent_decode_step)
            if (
                (not workload_plan_replay_active)
                and _intent_step_probe >= 0
                and pending_reason_to_code(tracking.trigger_intent_reason or "none")
                == int(PendingReasonCode.SENTENCE)
                and last_decode_refresh >= _intent_step_probe
            ):
                if update_state:
                    self._clear_request_trigger_intent(request_id=rid)
                    detail_counter_attr = (
                        "_sentence_trigger_admission_coalesced_pending_rebuild_total"
                        if pending_rebuild_inflight_by_req.get(rid, False)
                        else "_sentence_trigger_admission_coalesced_inflight_total"
                    )
                    _record_sentence_trigger_admission_coalesced(detail_counter_attr)
            if not workload_plan_replay_active:
                intent_step = tracking.trigger_intent_decode_step
                if intent_step >= 0:
                    if _sentence_intent_covered_by_existing_pending(
                        rid,
                        tracking,
                        ticket,
                    ):
                        if update_state:
                            self._clear_request_trigger_intent(request_id=rid)
                            _record_sentence_trigger_admission_coalesced(
                                "_sentence_trigger_admission_coalesced_interval_pending_total"
                            )
                        continue
                    intent_reason = tracking.trigger_intent_reason or "trigger"
                    intent_reason_code = pending_reason_to_code(intent_reason)
                    # [CREDIT-RETIRE 2026-07-07] post_bridge credit 状态机下线:
                    # 原 credit_active/credit_due 两分支在此吸收/消费 sentence
                    # intent。credit 的"计时校准"动机已由世代完成时的
                    # last_decode_refresh 推进天然覆盖;其 done 标志被任何
                    # FORCE_NOW+SENTENCE 世代完成误置(不限 post-bridge 追赶
                    # 世代),使普通句世代后的 interval 到点被静默吞掉——隐式
                    # 状态机横跨 4 文件,出错无法定位,按"兜底下线"方针整机退休。
                    intent_force_now = tracking.trigger_intent_force_now
                    # [SENTENCE-INTERVAL-SPACING 2026-07-08 用户拍板 v2"冲突的
                    # 部分要丢弃,refresh 不是无限排队"] 间距冲突的句边界意图
                    # =当场丢弃(清意图+计数),不顺延不排队:下一个满足间距的
                    # **真句边界**才触发,interval 陈旧上界仍是兜底节拍。v1 的
                    # 顺延语义会让旧边界在间距到期补发=节奏钉在间距地板上且
                    # 票面步陈旧,已按用户口径废弃。注:token-time 句边界意图
                    # 一律带 force_now=True(=物化优先级,非补票标记),间距闸不
                    # 豁免 force_now;bridge 追赶票走独立 post_bridge 臂不经
                    # 此闸;"并入既有 pending"(零新增世代)不受限。输入全为
                    # 决定论量(步数/提交历史/config),TP-rank 不变。
                    _sentence_drop = False
                    if (
                        (not ticket.pending_refresh)
                        and intent_reason_code == int(PendingReasonCode.SENTENCE)
                        and decode_step >= 0
                        and last_decode_refresh >= 0
                    ):
                        _trigger_cfg_local = getattr(tracking, "trigger", None)
                        _gap_floor = int(
                            getattr(
                                getattr(_trigger_cfg_local, "config", None),
                                "min_refresh_gap",
                                DEFAULT_MIN_REFRESH_GAP,
                            )
                            or 0
                        )
                        _sentence_spacing = max(
                            _gap_floor,
                            (interval // 2) if interval > 0 else 0,
                        )
                        if (
                            _sentence_spacing > 0
                            and (decode_step - last_decode_refresh)
                            < _sentence_spacing
                        ):
                            _sentence_drop = True
                            if update_state:
                                _record_sentence_trigger_admission_coalesced(
                                    "_sentence_trigger_admission_dropped_spacing_total"
                                )
                    if ticket.pending_refresh:
                        pending_step_cur = ticket.pending_decode_step
                        if pending_step_cur < 0:
                            pending_step_cur = decode_step
                        merged_step = intent_step
                        if pending_step_cur >= 0:
                            merged_step = min(pending_step_cur, intent_step)
                        merged_reason_code = ticket.pending_reason_code
                        if merged_reason_code == PendingReasonCode.NONE:
                            merged_reason_code = intent_reason_code
                        merged_policy = ticket.pending_policy
                        if intent_force_now and merged_policy != PendingPolicy.FORCE_NOW:
                            merged_policy = PendingPolicy.FORCE_NOW
                            merged_reason_code = intent_reason_code
                        _queue_set_pending_refresh(
                            request_id=rid,
                            reason_code=merged_reason_code,
                            decode_step=merged_step,
                            pending_policy=merged_policy,
                            pending_ctrl_step=(
                                ticket.pending_ctrl_step
                                if ticket.pending_ctrl_step >= 0
                                else self.step_context_epoch
                            ),
                        )
                    elif not _sentence_drop:
                        # [SENTENCE-INTERVAL-SPACING 2026-07-08 用户拍板"sentence/
                        # interval 配合 min_gap 不能打架"] 新建 SENTENCE pending
                        # 的专属间距=max(min_gap, interval//2):此前 intent 入队
                        # 无间距检查、pending 在全局 gap 到期即物化→句号密集语料
                        # 下 sentence 以 min_gap 地板节奏开火(12k 实测 126 发/512
                        # 步≈每 32 步一世代,触发密度×每世代成本=TP>1 倒挂主因),
                        # 对 acc 无增益。派生间距语义:sentence 世代已在提交点复位
                        # interval 计时(last_decode_refresh 跨 reason 推进),故
                        # sentence 只作为"至多把周期刷新提前一倍频"的机会主义
                        # 刷新;interval 陈旧上界与 min_gap 反爆发地板合同不变。
                        # 顺延=意图保留不清除(下步自然重评,更新的句边界覆盖旧
                        # 意图),该 rid 的 interval/lease_rearm 兜底节拍照常;
                        # FORCE_NOW(bridge 补票)与"并入既有 pending"(零新增世代
                        # 成本)不受此限。输入全为决定论量,TP-rank 不变。
                        _queue_set_pending_refresh(
                            request_id=rid,
                            reason_code=intent_reason_code,
                            decode_step=intent_step,
                            pending_policy=(
                                PendingPolicy.FORCE_NOW
                                if intent_force_now
                                else PendingPolicy.COALESCEABLE
                            ),
                            pending_ctrl_step=self.step_context_epoch,
                        )
                    if not _sentence_drop:
                        ticket = tickets_plan_by_req[rid]
                    # [SPACING v2] 丢弃与准入一律清意图(丢弃=作废该句边界,
                    # 无排队;下一个满足间距的真句边界重新置意图)。
                    if update_state:
                        self._clear_request_trigger_intent(request_id=rid)

            # lease 异常恢复意图由 planner 统一落票（单写者）：默认 FORCE_NOW。
            # [TP-DET-TRIGGER] gap 挡板替代 inflight 挡:顺手斩断远端 64k
            # lease_rearm↔interval 拉锯风暴(gap 内不再重复 rearm 提交)。
            if (not _refresh_gap_blocked(tracking, decode_step)) and tracking.lease_rearm:
                lease_step = tracking.lease_rearm_decode_step
                if lease_step < 0:
                    lease_step = decode_step
                if lease_step >= 0:
                    lease_reason_code = pending_reason_to_code(
                        tracking.lease_rearm_reason or "lease_rearm"
                    )
                    if ticket.pending_refresh:
                        pending_step_cur = ticket.pending_decode_step
                        if pending_step_cur < 0:
                            pending_step_cur = decode_step
                        merged_step = lease_step
                        if pending_step_cur >= 0:
                            merged_step = min(pending_step_cur, lease_step)
                        _queue_set_pending_refresh(
                            request_id=rid,
                            reason_code=lease_reason_code,
                            decode_step=merged_step,
                            pending_policy=PendingPolicy.FORCE_NOW,
                            pending_ctrl_step=(
                                ticket.pending_ctrl_step
                                if ticket.pending_ctrl_step >= 0
                                else self.step_context_epoch
                            ),
                        )
                    else:
                        _queue_set_pending_refresh(
                            request_id=rid,
                            reason_code=lease_reason_code,
                            decode_step=lease_step,
                            pending_policy=PendingPolicy.FORCE_NOW,
                            pending_ctrl_step=self.step_context_epoch,
                        )
                    ticket = tickets_plan_by_req[rid]
                    if update_state:
                        self._clear_request_lease_rearm(request_id=rid)

            post_bridge_due = int(
                getattr(tracking, "post_bridge_refresh_due_decode_step", -1)
            )
            if (
                (not workload_plan_replay_active)
                and post_bridge_due >= 0
                and last_decode_refresh >= post_bridge_due
            ):
                # [CREDIT-RETIRE 2026-07-07] 追赶已由其它世代覆盖:只清 due,
                # 不再发 credit(post_bridge_refresh_done 状态机已退休)。
                if update_state:
                    tracking.post_bridge_refresh_due_decode_step = -1
            elif (
                (not workload_plan_replay_active)
                and (not _refresh_gap_blocked(tracking, decode_step))
                and post_bridge_due >= 0
                and decode_step >= post_bridge_due
                and not ticket.pending_refresh
            ):
                _queue_set_pending_refresh(
                    request_id=rid,
                    reason_code=PendingReasonCode.SENTENCE,
                    decode_step=decode_step,
                    pending_policy=PendingPolicy.FORCE_NOW,
                    pending_ctrl_step=self.step_context_epoch,
                )
                ticket = tickets_plan_by_req[rid]
                if update_state:
                    tracking.post_bridge_refresh_due_decode_step = -1

            # [TP-DET-TRIGGER 2026-07-07] 票段决定论化:票在 enqueue commit 点
            # 即转 consumed(不再挂着等 GPU writer 完成)。此处仍见到票 = defer
            # 残留(allow_materialize=False 的空转步落票)或本步刚落——一律按
            # 决定论 gap 挡板决定是否 materialize;原 covered-by-ready-compact
            # 清票与 INTERVAL not-ready 保票三态(均读 GPU 完成态,per-rank
            # 异步,TP>1 决策发散根)整段退休。
            if ticket.pending_refresh:
                if not _refresh_gap_blocked(tracking, decode_step):
                    req_pending_step = (
                        ticket.pending_decode_step
                        if ticket.pending_decode_step >= 0
                        else decode_step
                    )
                    ticket_reason_code = ticket.pending_reason_code
                    if ticket_reason_code == PendingReasonCode.NONE:
                        ticket_reason_code = PendingReasonCode.TRIGGER
                    ticket_policy = ticket.pending_policy
                    pending_ctrl_step = ticket.pending_ctrl_step
                    if pending_ctrl_step < 0 and req_pending_step >= 0:
                        _queue_set_pending_refresh(
                            request_id=rid,
                            reason_code=ticket_reason_code,
                            decode_step=req_pending_step,
                            pending_policy=ticket_policy,
                            pending_ctrl_step=self.step_context_epoch,
                        )
                        ticket = tickets_plan_by_req[rid]
                        pending_ctrl_step = ticket.pending_ctrl_step
                        ticket_policy = ticket.pending_policy
                    if req_pending_step >= 0 and ticket.pending_decode_step < 0:
                        _queue_set_pending_refresh(
                            request_id=rid,
                            reason_code=ticket_reason_code,
                            decode_step=req_pending_step,
                        )
                        ticket = tickets_plan_by_req[rid]
                    if (
                        ticket.pending_reason_code
                        == PendingReasonCode.SENTENCE
                        and ticket_policy != PendingPolicy.FORCE_NOW
                    ):
                        _queue_set_pending_refresh(
                            request_id=rid,
                            reason_code=ticket_reason_code,
                            decode_step=req_pending_step,
                            pending_policy=PendingPolicy.FORCE_NOW,
                            pending_ctrl_step=(
                                pending_ctrl_step
                                if pending_ctrl_step >= 0
                                else self.step_context_epoch
                            ),
                        )
                        ticket = tickets_plan_by_req[rid]
                    if earliest_pending_decode is None or (
                        req_pending_step >= 0 and req_pending_step < earliest_pending_decode
                    ):
                        earliest_pending_decode = req_pending_step
                    refresh_set.add(rid)
                    if pending_reason_code is None:
                        pending_reason_code = ticket.pending_reason_code

            # interval 基于 request 自身的 decode 步数，避免跨 request 污染。
            # [TP-DET-TRIGGER] inflight 门删除:判定自身(decode_step-last≥
            # interval,last=提交点票面步)已决定论且蕴含 min_gap(interval≥gap)。
            if (
                (not workload_plan_replay_active)
                and interval > 0
                and decode_step >= 0
                and (decode_step - last_decode_refresh) >= interval
            ):
                # [CREDIT-RETIRE 2026-07-07] 原 post_bridge credit 在此吞掉
                # interval 到点(推 last+continue),已随状态机退休——interval
                # 到点一律按票面语义落票。
                has_pending = bool(ticket.pending_refresh)
                reason_cur = int(ticket.pending_reason_code)
                pending_is_interval_cur = reason_cur == int(PendingReasonCode.INTERVAL)
                # interval 不抢占更强 pending（如 sentence/lease_rearm）。
                if has_pending and (not pending_is_interval_cur):
                    continue

                pending_step_cur = (
                    ticket.pending_decode_step
                    if has_pending and ticket.pending_decode_step >= 0
                    else decode_step
                )
                refresh_set.add(rid)
                _queue_set_pending_refresh(
                    request_id=rid,
                    reason_code=PendingReasonCode.INTERVAL,
                    decode_step=pending_step_cur,
                    pending_policy=PendingPolicy.COALESCEABLE,
                    pending_ctrl_step=(
                        ticket.pending_ctrl_step
                        if has_pending and ticket.pending_ctrl_step >= 0
                        else self.step_context_epoch
                    ),
                )
                if interval_reason_code is None:
                    interval_reason_code = PendingReasonCode.INTERVAL

        interval_merge_policy = getattr(self, "_interval_merge_policy", "delta1")
        # [INTERVAL-RIDE-ALONG 2026-07-09] interval 拍点世代拉齐（整块替换原
        # delta1"全有或全无"合并）。原语义要求批内全部未触发请求恰好
        # delta==interval-1 才整批并入——chunked prefill/serve 错峰批下几乎
        # 永不成立 → 每请求各开独立世代，世代固定成本（selector/writer 链墙
        # +bind）×BS 放大（TP8×64k bs32 取证 interval 事件=朴素预期 4.3×，
        # 0.77× 倒挂的结构根源）。
        # 新语义：interval 拍点开世代时，决定论 gap 挡板放行、无既有 pending
        # 票、且已进入稳态 decode 的请求，就地以**当前 decode_step 为票面**
        # 正常触发搭同一班车（新鲜信号，无旧票堆积/延迟攒批）；提交即终局
        # 推进时钟 → 搭车批自锁进同一节拍，稳态恢复每请求 1/interval，世代数
        # 坍缩到节拍数。sentence/lease 语义不动（有票不抢占）；min_refresh_gap
        # 全局合同不变（挡板原样）；输入全为 host 决定论量（[TP-DET-TRIGGER]
        # 合同保持）。escape=interval_merge_policy="off"（既有旋钮，零新增）。
        if (
            interval_merge_policy == "delta1"
            and (not workload_plan_replay_active)
            and interval > 0
            and interval_reason_code is not None
            and refresh_set
            and len(refresh_set) < len(request_ids)
        ):
            ride_along_joined = 0
            for rid in request_ids:
                if rid in refresh_set:
                    continue
                ticket = tickets_plan_by_req.get(rid)
                if ticket is not None and bool(ticket.pending_refresh):
                    # 已有票（sentence/lease/interval pending）：不抢占不改票。
                    continue
                tracking = tracking_by_req[rid]
                # [TP-DET-TRIGGER] inflight 过滤 → 决定论 gap 过滤。
                if _refresh_gap_blocked(tracking, decode_step_by_req.get(rid, -1)):
                    continue
                if _is_short_dense_blocked(rid):
                    continue
                decode_step = int(tracking.decode_step) if tracking.decode_step is not None else -1
                last_decode_refresh = int(tracking.last_decode_refresh_step) if tracking.last_decode_refresh_step is not None else -1
                if decode_step < 0 or last_decode_refresh < 0:
                    # bootstrap/prefill 窗（decode 时钟未立）：不拉。
                    continue
                refresh_set.add(rid)
                _queue_set_pending_refresh(
                    request_id=rid,
                    reason_code=PendingReasonCode.INTERVAL,
                    decode_step=decode_step,
                    pending_policy=PendingPolicy.COALESCEABLE,
                    pending_ctrl_step=self.step_context_epoch,
                )
                ride_along_joined += 1
            if ride_along_joined:
                coalesced = True
                if update_state:
                    self._interval_ride_along_joined_total = (
                        int(
                            getattr(
                                self,
                                "_interval_ride_along_joined_total",
                                0,
                            )
                        )
                        + ride_along_joined
                    )

        self._current_decode_step_by_req_epoch = int(self.step_context_epoch)
        self._current_decode_step_by_req = dict(decode_step_by_req)

        # refresh coalescing：允许在小窗口内合并相邻 request 的 refresh，避免形成极小 batch。
        coalesce_window = self.config.refresh_coalesce_window
        post_bridge_near_target_coalesce = (
            _post_bridge_sentence_near_target_coalesce_enabled()
        )
        sentence_effective_coalesce_window = (
            max(1, int(coalesce_window))
            if post_bridge_near_target_coalesce
            else int(coalesce_window)
        )
        if (
            sentence_effective_coalesce_window > 0
            and (not workload_plan_replay_active)
            and refresh_set
            and len(refresh_set) < len(request_ids)
        ):
            target: Optional[int] = None
            if earliest_pending_decode is not None and earliest_pending_decode >= 0:
                target = earliest_pending_decode
            else:
                # interval-only 或非 trigger 场景：以已触发 refresh 的最小 decode_step 为对齐基准
                for rid in refresh_set:
                    decode_step = decode_step_by_req.get(rid, -1)
                    if decode_step < 0:
                        continue
                    if target is None or decode_step < target:
                        target = decode_step
            if target is None:
                target = -1
            for rid in request_ids:
                if rid in refresh_set:
                    continue
                # [TP-DET-TRIGGER] inflight 过滤 → 决定论 gap 过滤。
                if _refresh_gap_blocked(
                    tracking_by_req[rid], decode_step_by_req.get(rid, -1)
                ):
                    continue
                if _is_short_dense_blocked(rid):
                    continue
                decode_step = decode_step_by_req.get(rid, -1)
                if decode_step < 0:
                    continue
                if abs(decode_step - target) <= sentence_effective_coalesce_window:
                    if _has_interval_pending_not_ready(
                        rid,
                        tickets_plan_by_req[rid],
                    ):
                        if update_state:
                            self._refresh_coalesce_skipped_existing_pending_total = (
                                int(
                                    getattr(
                                        self,
                                        "_refresh_coalesce_skipped_existing_pending_total",
                                        0,
                                    )
                                )
                                + 1
                            )
                        continue
                    if (
                        pending_reason_code == int(PendingReasonCode.SENTENCE)
                        and not _has_sentence_refresh_due_at_target(rid, target)
                    ):
                        if not post_bridge_near_target_coalesce:
                            continue
                        post_bridge_due = _post_bridge_sentence_due_near_target(
                            rid,
                            decode_step_local=int(decode_step),
                            target_step=int(target),
                            window=int(sentence_effective_coalesce_window),
                        )
                        if post_bridge_due < 0:
                            continue
                        _queue_set_pending_refresh(
                            request_id=rid,
                            reason_code=PendingReasonCode.SENTENCE,
                            decode_step=int(post_bridge_due),
                            pending_policy=PendingPolicy.FORCE_NOW,
                            pending_ctrl_step=self.step_context_epoch,
                        )
                        if update_state:
                            tracking_by_req[
                                rid
                            ].post_bridge_refresh_due_decode_step = -1
                    refresh_set.add(rid)
                    coalesced = True

        for rid in tuple(refresh_set):
            ticket = tickets_plan_by_req[rid]
            pending_step = ticket.pending_decode_step
            if pending_step >= 0:
                continue
            inferred_pending_step = decode_step_by_req.get(rid, -1)
            if inferred_pending_step < 0:
                raise RuntimeError(
                    f"pending refresh ticket missing valid pending_decode_step for request {rid!r}"
                )
            reason_code_cur = ticket.pending_reason_code
            if reason_code_cur != PendingReasonCode.NONE:
                inferred_reason_code = reason_code_cur
            elif pending_reason_code is not None:
                inferred_reason_code = pending_reason_code
            elif interval_reason_code is not None:
                inferred_reason_code = interval_reason_code
            else:
                inferred_reason_code = PendingReasonCode.REFRESH
            _queue_set_pending_refresh(
                request_id=rid,
                reason_code=inferred_reason_code,
                decode_step=inferred_pending_step,
            )

        if update_state:
            for rid, update in pending_updates.items():
                (
                    reason_code_update,
                    step_update,
                    policy_update,
                    ctrl_update,
                    clear_ready_compact,
                ) = update
                if clear_ready_compact:
                    self._clear_request_pending_refresh(
                        request_id=rid,
                        ready_compact=True,
                    )
                else:
                    self._set_request_pending_refresh(
                        request_id=rid,
                        reason_code=reason_code_update,
                        decode_step=step_update,
                        pending_policy=policy_update,
                        pending_ctrl_step=ctrl_update,
                    )

        if refresh_set:
            # [TP-DET-TRIGGER] 终审过滤 inflight → 决定论 gap。
            refresh_set = {
                rid
                for rid in refresh_set
                if (
                    not _refresh_gap_blocked(
                        tracking_by_req[rid], decode_step_by_req.get(rid, -1)
                    )
                )
                or (
                    workload_plan_replay_active
                    and rid in replay_forced_refresh_set
                )
            }
        allowed = bool(refresh_set)
        can_materialize = allowed and allow_materialize
        refresh_reqs: Tuple[str, ...] = tuple()
        if can_materialize:
            pending_window = coalesce_window
            if (
                pending_reason_code is not None
                and earliest_pending_decode is not None
                and earliest_pending_decode >= 0
                and (not coalesced)
            ):
                if (
                    pending_reason_code == PendingReasonCode.SENTENCE
                    and pending_window < 1
                ):
                    pending_window = 1
            else:
                pending_window = max(pending_window, max(0, interval))
            refresh_candidates = tuple(
                rid
                for rid in request_ids
                if rid in refresh_set
                and (
                    (
                        not _refresh_gap_blocked(
                            tracking_by_req[rid], decode_step_by_req.get(rid, -1)
                        )
                    )
                    or (
                        workload_plan_replay_active
                        and rid in replay_forced_refresh_set
                    )
                )
            )
            refresh_reqs = materialize_refresh_reqs(
                request_ids=refresh_candidates,
                tickets_by_req=tickets_plan_by_req,
                coalesce_window=max(0, pending_window),
            )
            if refresh_set and not refresh_reqs:
                raise RuntimeError(
                    "plan_refresh_requests materialization failed: "
                    "refresh_set non-empty but refresh_reqs empty"
                )
            if pending_reason_code is not None:
                last_reason = pending_reason_code_to_text(pending_reason_code)
            elif interval_reason_code is not None:
                last_reason = pending_reason_code_to_text(interval_reason_code)
            else:
                last_reason = last_reason or "refresh"
            # 方案 B（single-source commit）：planner 只产出计划，不写 inflight/scheduled。
            # scheduled_* 仅在 payload 真正 enqueue 成功时由 commit 路径写入，
            # 避免“计划成功但未提交”被错误标记为 inflight，导致 sentence/interval 静默失效。
        elif allowed and (not allow_materialize):
            # 空转步（无 kernel 执行）只允许保留 pending 语义，不允许 materialize 成本步 refresh_reqs；
            # 否则会出现“计划触发但执行路径不存在”的 ghost plan。
            last_reason = "defer_no_kernel_work"
        else:
            last_reason = "compact"

        refresh_reqs_set = set(refresh_reqs)
        refresh_rows = tuple(idx for idx, rid in enumerate(request_ids) if rid in refresh_reqs_set)
        mode_by_row_list: List[int] = []
        force_dense_while_inflight_by_row_list: List[bool] = []
        for rid in request_ids:
            ticket = tickets_plan_by_req[rid]
            is_inflight = inflight_by_req.get(rid, False)
            if rid in refresh_reqs_set:
                mode = (
                    StepRefreshMode.MUST_NOW
                    if ticket.pending_policy == PendingPolicy.FORCE_NOW
                    else StepRefreshMode.DEFER
                )
            elif is_inflight:
                mode = StepRefreshMode.INFLIGHT
            elif ticket.pending_refresh:
                mode = StepRefreshMode.DEFER
            else:
                mode = StepRefreshMode.NONE
            mode_by_row_list.append(mode)
            force_dense_while_inflight_by_row_list.append(
                mode == StepRefreshMode.INFLIGHT
                and rid in inflight_dense_consume_set
            )
            # [PENDING-FUNNEL-DEBUG 2026-07-09 临时取证探针,破案后拆] 行判决快照。
            _funnel_dbg = os.environ.get("VLLM_SPARSE_PENDING_FUNNEL_DEBUG_LOG", "")
            if _funnel_dbg:
                _fd_n = getattr(self, "_funnel_plan_probe_n", 0) + 1
                self._funnel_plan_probe_n = _fd_n
                if _fd_n % 16 == 1:
                    try:
                        _fd_tr = tracking_by_req[rid]
                        with open(_funnel_dbg, "a") as _fd_fh:
                            _fd_fh.write(
                                f"plan\treq={rid}\tmode={mode}\t"
                                f"inflight={inflight_by_req.get(rid, False)}\t"
                                f"prinflight={pending_rebuild_inflight_by_req.get(rid, False)}\t"
                                f"sched_ctrl={int(getattr(_fd_tr, 'scheduled_refresh_ctrl_step', -1))}\t"
                                f"sched_dec={int(getattr(_fd_tr, 'scheduled_decode_refresh_step', -1))}\t"
                                f"dense_consume={rid in inflight_dense_consume_set}\t"
                                f"dg_readable={_dual_gen_inflight_compact_readable(_fd_tr)}\t"
                                f"tk_pend={bool(tickets_plan_by_req[rid].pending_refresh)}\t"
                                f"infl_reason={int(getattr(_fd_tr, 'inflight_reason_code', -1))}\t"
                                f"epoch={self.step_context_epoch}\n"
                            )
                    except OSError:
                        pass

        for idx, rid in enumerate(request_ids):
            mode = mode_by_row_list[idx]
            if mode == StepRefreshMode.MUST_NOW and rid not in refresh_reqs_set:
                raise RuntimeError(
                    "refresh plan invariant violated: MUST_NOW row missing in refresh_reqs "
                    f"(req={rid!r}, step={self.step_context_epoch})"
                )
            if mode == StepRefreshMode.INFLIGHT and rid in refresh_reqs_set:
                raise RuntimeError(
                    "refresh plan invariant violated: INFLIGHT row appears in refresh_reqs "
                    f"(req={rid!r}, step={self.step_context_epoch})"
                )
            ticket = tickets_plan_by_req[rid]
            if (
                (not inflight_by_req.get(rid, False))
                and ticket.pending_refresh
                and ticket.pending_policy == PendingPolicy.FORCE_NOW
                and rid not in refresh_reqs_set
                and allow_materialize
            ):
                raise RuntimeError(
                    "FORCE_NOW pending request missing from refresh_reqs: "
                    f"req={rid!r} step={self.step_context_epoch}"
                )

        result = StepRefreshPlan(
            epoch=self.step_context_epoch,
            req_ids=request_ids,
            mode_by_row=tuple(mode_by_row_list),
            refresh_rows=refresh_rows,
            refresh_reqs=refresh_reqs,
            refresh_reason=last_reason,
            bootstrap_done=bootstrap_done,
            plan_signature=(
                self.step_context_epoch,
                request_ids,
                tuple(mode_by_row_list),
                tuple(force_dense_while_inflight_by_row_list),
                refresh_rows,
                refresh_reqs,
                last_reason,
                bootstrap_done,
            ),
            force_dense_while_inflight_by_row=tuple(
                force_dense_while_inflight_by_row_list
            ),
        )
        if update_state:
            self._should_refresh_cache[cache_key] = result
            if pending_cleared:
                self._bump_refresh_nonce()
        return result

    def should_refresh(self, state: LayerState) -> Tuple[bool, List[int]]:
        """遗留分支：已下线，refresh 由 dispatcher StepAuthority 单源路径统一处理。"""
        del state
        raise RuntimeError(
            "should_refresh direct branch is disabled; use dispatcher StepAuthority path"
        )

    def layer_stats(self) -> Dict[int, Dict[str, object]]:
        stats: Dict[int, Dict[str, object]] = {}
        for cache_key, state in self.layer_states.items():
            lengths = [int(x) for x in state.compact_kv_len]
            coverage = []
            if state.last_coverage is not None:
                coverage = [float(x) for x in state.last_coverage.detach().cpu().reshape(-1).tolist()]
            capped = []
            if state.last_capped is not None:
                capped = [bool(x) for x in state.last_capped.detach().cpu().reshape(-1).tolist()]
            last_refresh = state.last_refresh_step
            steps_cpu = state.last_refresh_step_per_slot_cpu
            if steps_cpu:
                last_refresh = int(max(int(x) for x in steps_cpu))
            stats[cache_key] = {
                "lengths": lengths,
                "coverage": coverage,
                "capped": capped,
                "last_reason": state.last_reason,
                "last_refresh_step": last_refresh,
                "k_max_current": state.k_max_current,
            }
        return stats


    def _compute_alpha_selection_pipeline_unified(
        self,
        *,
        capture_scores: torch.Tensor,
        log_f_denoms: Optional[torch.Tensor],
        kv_lengths: torch.Tensor,
        key_norms_full: torch.Tensor,
        num_kv_heads: int,
        num_queries_per_kv: int,
        block_size: int,
        topk_slice_start: Optional[int] = None,
        topk_slice_end: Optional[int] = None,
        seq_lens_full: Optional[torch.Tensor] = None,
        seq_lens_cpu: Optional[Sequence[int]] = None,
        seq_lens_tensor_cpu: Optional[torch.Tensor] = None,
        profile_detail: bool = False,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[Dict[str, Tuple[Optional[torch.cuda.Event], Optional[torch.cuda.Event]]]],
    ]:
        _, pipeline_impl = _load_selector_selection_impls()
        return pipeline_impl(
            self,
            capture_scores=capture_scores,
            log_f_denoms=log_f_denoms,
            kv_lengths=kv_lengths,
            key_norms_full=key_norms_full,
            num_kv_heads=num_kv_heads,
            num_queries_per_kv=num_queries_per_kv,
            block_size=block_size,
            topk_slice_start=topk_slice_start,
            topk_slice_end=topk_slice_end,
            seq_lens_full=seq_lens_full,
            seq_lens_cpu=seq_lens_cpu,
            seq_lens_tensor_cpu=seq_lens_tensor_cpu,
            profile_detail=profile_detail,
        )

# ---------------------------------------------------------------------------
# Patch installation (delegated to patch_installer.py)
# ---------------------------------------------------------------------------
from patches.patch_installer import (  # noqa: E402
    _set_unified_attention_mode,
    apply_vllm_sparse_patch,
    disable_vllm_sparse_patch,
    ensure_vllm_sparse_patch_from_env,
)
