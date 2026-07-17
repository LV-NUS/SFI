from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch

from patches.fa_sparse_runtime.compact_recent_alignment import (
    compact_recent_effective_k_head,
    compact_write_sub_slot,
)
from patches.sparse_constants import compact_gen_count

_log = logging.getLogger(__name__)
from patches.sparse_constants import (
    _DYNAMIC_ENV,
    _SELECTOR_LOGS_CACHE_R_CACHED,
    _ASYNC_PRODUCER_WRITER_GRAPH_CACHED,  # ASYNC_PRODUCER_WRITER_GRAPH
    _DECODE_BOUNDS_KERNEL_CACHED,
    _DEFERRED_SELECTOR_PROFILE_DETAIL_CACHED,
    _REBUILD_PTRS_PINNED_CACHED,
    _SELECTOR_SELECTED_INDICES_OUT_CACHED,
    _SELECTOR_PIPELINE_WORKSPACE_CACHED,
    _SELECTOR_FIXED_SHAPE_TOPK_CACHED,
    _SELECTOR_TOPK_GRAPH_CACHED,  # #13 STAGE-0 captured selector graph (default OFF)
    _WRITER_TOKEN_TILE_CACHED,
    _WRITER_INPUT_BTABLE_CHECK_CACHED,
)
from patches.selector_runtime.selected_out_ring import SelectedOutRing, SlotStableOverrides
from patches.sparse_types import SelectorBatchPayload, continuous_producer_enabled
from patches.sparse_utils import _submission_slot_owner_snapshot
from utils.selector_pipeline_identity import SELECTOR_PIPELINE_SEMANTIC_VERSION
import time as _sw_time


def _pipeline_pack_order_is_verified(pipeline_fn: object) -> bool:
    marker = getattr(pipeline_fn, "_sfi_pack_order_canonical_semantic", None)
    return type(marker) is int and marker == SELECTOR_PIPELINE_SEMANTIC_VERSION


def _producer_detail_marker(controller):
    """[S7-FORENSIC 2026-07-10] off-loop selector/writer impl host 分相记账器。

    detail 门(VLLM_SPARSE_DEFERRED_SELECTOR_PROFILE_DETAIL)关=返回 None 零税;
    开=返回 mark(name) 闭包,dict 载体挂 controller,flush_worker 快照带出
    refresh_profile.log(deadline_deferred_producer_detail_us)。
    """
    if not _DEFERRED_SELECTOR_PROFILE_DETAIL_CACHED:
        return None
    detail = getattr(controller, "_deadline_deferred_producer_detail_us", None)
    if detail is None:
        detail = {}
        controller._deadline_deferred_producer_detail_us = detail
    state = [_sw_time.perf_counter_ns()]

    def _mark(name: str) -> None:
        now_ns = _sw_time.perf_counter_ns()
        detail[name] = float(detail.get(name, 0.0) or 0.0) + (
            now_ns - state[0]
        ) / 1000.0
        state[0] = now_ns

    return _mark

# [REBUILD-H2D-STAGING 2026-07-05] rebuild 提交路径的三个小张量原用
# `torch.tensor(list, device=cuda)` 构造——pageable 源的 H2D 是同步拷贝
# （cudaStreamSynchronize 当前流）,而 rebuild 恰在 selector 刚向 refresh_stream
# 排完 kernel 之后执行,每次构造都把 host 阻塞到流排空（实测慢步主体,
# ~0.7ms/chunk 隐藏等待）。改用 cached_sequence_to_device（pinned staging +
# 值键 reuse + stage 事件 WAR 护栏）:值同直接命中（世代内 3 chunk 逐位同值,
# slot/row 稳态跨世代恒定）,miss 走 pinned 非阻塞。DETERMINISTIC-SLOT-SOURCE
# 合同不变:值仍恒取自 host 权威快照（slot_list/row_list/seq_lens_cpu_ref）,
# 且 graph ON 下 writer 消费的是 _ensure_selector_writer_* stable buffer,
# 这些张量只是同流 copy_ 的瞬时 src——复用无 deferred 消费者腐蚀面。
# row_list 错峰下可按层不同。staging 由 LayerState 的真实生命周期持有，
# 每层只保留当前权威值；同一 fused 调用中的同值 layer 共享一个 tensor。
# 稳态仍按值命中免 H2D，且不再累计历史组合或依赖经验 population 上限。
def _compact_gather_stride_tokens(owner: object, state: object, block_size: int) -> int:
    if getattr(state, "compact_page_residency", None) is not None:
        if int(getattr(state, "compact_stride_block_size", block_size)) != int(block_size):
            return 0
        return int(getattr(state, "compact_stride_tokens", 0))
    return int(owner._compact_stride_tokens(block_size))


def _rebuild_ptr_buffer_layer_suffix(
    payloads: Sequence[SelectorBatchPayload],
) -> str:
    """[W2a 2026-07-09] 层段后缀一次推导(六个 ptr buffer 名共享同一 payloads,
    原实现每个名字重复遍历 payloads 推导同一后缀)。返回 ":layers_..." 或 ""。"""
    state_layer_indices: List[int] = []
    for payload_index, payload in enumerate(payloads):
        state = getattr(payload, "state", None)
        state_layer_index = int(getattr(state, "layer_index", -1))
        if state_layer_index < 0:
            state_layer_index = int(getattr(payload, "layer_index", payload_index))
        state_layer_indices.append(int(state_layer_index))
    if not state_layer_indices:
        return ""

    first_layer = int(state_layer_indices[0])
    contiguous = all(
        int(layer_index) == int(first_layer) + int(offset)
        for offset, layer_index in enumerate(state_layer_indices)
    )
    if contiguous:
        last_layer = int(state_layer_indices[-1])
        suffix = f"layers_{first_layer}_{last_layer}_n{len(state_layer_indices)}"
    else:
        suffix = "layers_" + "_".join(str(int(v)) for v in state_layer_indices)
    return f":{suffix}"


def _rebuild_ptr_buffer_name(
    base_name: str,
    payloads: Sequence[SelectorBatchPayload],
) -> str:
    """Scope cached pointer arrays by layer segment to avoid ready-group ping-pong."""
    return f"{base_name}{_rebuild_ptr_buffer_layer_suffix(payloads)}"
















def _new_pinned_i32_tensor(values: tuple) -> torch.Tensor:
    """[SEQLENS-STAGING-UAF-FIX] 每世代一次的独立 pinned 小张量(bs×4B),
    供 deferred writer 路径做非阻塞 H2D 源——不入任何共享 staging 缓存。"""
    t = torch.tensor(values, dtype=torch.int32)
    try:
        return t.pin_memory()
    except RuntimeError:
        return t


def _new_pinned_i64_tensor(values: tuple) -> torch.Tensor:
    """[SLOT-STAGING-UAF-FIX] 同上 i32 版:每世代一次的独立 pinned 小张量
    (bs×8B),供 deferred writer 路径的 slot 输入做非阻塞 H2D 源——不入任何
    共享 staging 缓存。"""
    t = torch.tensor(values, dtype=torch.long)
    try:
        return t.pin_memory()
    except RuntimeError:
        return t


def rebuild_compact_slots_batched_layers_from_selection_impl(
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
    """跨层 fused gather 重建 compact。成功返回 True，失败回退返回 False。"""
    _pd_mark = _producer_detail_marker(self)
    if _pd_mark is not None:
        # [B6 2026-07-11] wr_* 分相键为全调用方累计(bootstrap/prefill/refresh
        # 同一载体)——per-call 判读除以 writer_count 分母错。加显式调用计数键,
        # per-call = wr_*/wr_calls 才可信(相对占比不受影响)。
        _pd_wd = self._deadline_deferred_producer_detail_us
        _pd_wd["wr_calls"] = float(_pd_wd.get("wr_calls", 0.0) or 0.0) + 1.0
    if defer_compact_meta_publish and compact_meta_commit_log is None:
        raise RuntimeError("deferred compact metadata publish requires commit log")
    if not payloads:
        return True
    cfg = getattr(self, "config", None)
    strict_refresh_rebuild = bool(
        getattr(cfg, "one_shot_bootstrap_only", False)
    ) and continuous_producer_enabled(cfg) and phase == "refresh"

    def _fail_rebuild_contract(reason: str) -> bool:
        if strict_refresh_rebuild:
            raise RuntimeError(f"refresh compact rebuild contract failed: {reason}")
        return False

    if (
        bool(getattr(cfg, "one_shot_bootstrap_only", False))
        and not continuous_producer_enabled(cfg)
        and phase == "refresh"
    ):
        raise RuntimeError("one-shot bootstrap-only mode forbids decode refresh rebuild")
    layers = len(payloads)
    if selected_indices.dim() != 4 or selected_indices.shape[0] != layers:
        return _fail_rebuild_contract(
            f"selected_indices_shape_mismatch shape={tuple(selected_indices.shape)} layers={layers}"
        )

    first = payloads[0]
    slot_list = [int(slot) for slot in first.slot_list]
    batch = len(slot_list)
    if batch == 0:
        return True
    slot_owner_snapshot: Tuple[Tuple[int, str], ...] = tuple()
    if defer_compact_meta_publish:
        slot_owner_snapshot = _submission_slot_owner_snapshot(
            payloads,
            slot_list,
            stage=f"deferred {phase} compact metadata",
        )
    device = first.key_cache.device
    if selected_indices.device != device:
        return _fail_rebuild_contract("selected_indices_device_mismatch")
    num_kv_heads = int(first.state.num_kv_heads)
    if num_kv_heads <= 0:
        return _fail_rebuild_contract("num_kv_heads_nonpositive")
    if first.key_cache.dim() < 2:
        return _fail_rebuild_contract("key_cache_rank_lt_2")
    block_size_ref = int(first.key_cache.shape[1])
    head_dim_ref = int(first.key_cache.shape[-1])
    kv_dtype_ref = first.key_cache.dtype
    if block_size_ref <= 0 or head_dim_ref <= 0:
        return _fail_rebuild_contract("invalid_block_size_or_head_dim")

    if bootstrap_slots_by_layer is None:
        bootstrap_slots_by_layer = [payload.bootstrap_slots for payload in payloads]
    if len(bootstrap_slots_by_layer) != layers:
        return _fail_rebuild_contract("bootstrap_slots_layer_count_mismatch")
    base_slot_epoch = int(getattr(first.state, "slot_epoch", -1))
    base_slot_sig = int(getattr(first.state, "slot_signature64", -1))
    if layers > 1:
        for layer_idx, payload in enumerate(payloads[1:], start=1):
            state_cur = payload.state
            cur_slot_epoch = int(getattr(state_cur, "slot_epoch", -1))
            cur_slot_sig = int(getattr(state_cur, "slot_signature64", -1))
            if cur_slot_epoch != base_slot_epoch or cur_slot_sig != base_slot_sig:
                raise RuntimeError(
                    "slot signature mismatch before fused rebuild: "
                    f"base_layer=0 base_epoch={base_slot_epoch} base_sig={base_slot_sig} "
                    f"layer_index={layer_idx} layer_epoch={cur_slot_epoch} "
                    f"layer_sig={cur_slot_sig}"
                )

    # 诊断状态不得抢在输入契约之前写入：fail-fast 调用方可能只提供一个
    # 最小 self 来验证跨层 slot 一致性。通过全部结构校验后，真实 controller
    # 仍在原有有效路径上恰好重置一次该字段，零额外热路径分支。
    setattr(self, "_last_writer_kernel_variant", "")
    threshold = self._compact_threshold_tokens()
    cfg = self.config
    semantic_snapshot = self._get_step_semantic_snapshot() if cfg is not None else None
    sink_cap = max(0, int(semantic_snapshot.sink_tokens)) if semantic_snapshot is not None else 0
    recent_cfg = max(0, int(semantic_snapshot.recent_tokens)) if semantic_snapshot is not None else 0
    alpha_fair_cfg = getattr(cfg, "alpha_fair", None) if cfg is not None else None
    k_head_cfg = max(0, int(getattr(alpha_fair_cfg, "k_head", 0) or 0)) if alpha_fair_cfg is not None else 0
    k_head_cfg = compact_recent_effective_k_head(
        k_head=k_head_cfg,
        sink_tokens=sink_cap,
        attn_mode=str(getattr(cfg, "attn_mode", "compact_recent")),
    )
    use_cuda_ptr_arrays = bool(_REBUILD_PTRS_PINNED_CACHED)
    use_ptr_arrays = bool(use_cuda_ptr_arrays)
    use_pinned_ptrs = bool(_REBUILD_PTRS_PINNED_CACHED and use_ptr_arrays)
    # [W2a 2026-07-09] 后缀一次推导,六名共享(原六次重复遍历 payloads)。
    _ptr_layer_suffix = _rebuild_ptr_buffer_layer_suffix(payloads)
    flat_k_ptrs_name = f"flat_k_ptrs{_ptr_layer_suffix}"
    flat_v_ptrs_name = f"flat_v_ptrs{_ptr_layer_suffix}"
    compact_k_ptrs_name = f"compact_k_ptrs{_ptr_layer_suffix}"
    compact_v_ptrs_name = f"compact_v_ptrs{_ptr_layer_suffix}"
    compact_pos_ptrs_name = f"compact_pos_ptrs{_ptr_layer_suffix}"
    block_table_ptrs_name = f"block_table_ptrs{_ptr_layer_suffix}"

    # ------------------------------------------------------------------
    # A 路线：CPU 元数据写回优化
    # fused gather 之后每层都需要更新 compact_{sink,persist,kv}_len；
    # 这些值按 slot 只依赖 (seq_len, threshold, sink, recent, block_size, k_head) 且跨层相同。
    # 因此这里按 batch 预计算一次，后续 per-layer 只做赋值/必要 reset。
    # ------------------------------------------------------------------
    seq_lens_cpu_ref = payloads[0].seq_lens_cpu
    if seq_lens_cpu_ref is None or len(seq_lens_cpu_ref) != batch:
        return _fail_rebuild_contract("seq_lens_cpu_ref_missing_or_wrong_batch")
    if threshold > 0:
        rebuild_mask_cpu_ref = [int(s) >= int(threshold) for s in seq_lens_cpu_ref]
    else:
        rebuild_mask_cpu_ref = [int(s) > 0 for s in seq_lens_cpu_ref]
    has_rebuild_any_ref = any(rebuild_mask_cpu_ref)
    reset_slots_ref = [int(slot_list[i]) for i, flag in enumerate(rebuild_mask_cpu_ref) if not flag]
    rebuild_pos_ref = [int(i) for i, flag in enumerate(rebuild_mask_cpu_ref) if flag]
    rebuild_slots_ref = [int(slot_list[i]) for i in rebuild_pos_ref]

    sink_lens_ref: List[int] = []
    persist_lens_ref: List[int] = []
    kv_lens_ref: List[int] = []
    # 预计算 seq_lens / rebuild_mask / sink_len（跨层共享，避免每层重复构造）
    # [DETERMINISTIC-AUTOLEN 2026-07-02] `first.seq_lens_batch[:batch]` is a VIEW of
    # the live per-step seqused buffer, which the decode main stream advances every
    # step. The gather kernel runs on refresh_stream and its actual execution time
    # floats relative to the main stream (the submission handoff_event only
    # lower-bounds it), so an autolen read through the live view copies a token
    # count that depends on WHEN the gather happens to run -> the compact KV
    # contents (and therefore the decoded tokens) differed run-to-run inside the
    # bootstrap/refresh switchover window. Always materialize seq_lens from the
    # host-side snapshot taken at submission time (`seq_lens_cpu_ref`) -- the same
    # submission-step value the rest of this rebuild is planned against.
    # [REBUILD-H2D-STAGING] 原 `torch.tensor(list, device)`（pageable H2D）会同步
    # 等当前流排空（此处 refresh_stream 刚被 selector 塞满）——并非注释旧称的
    # "no hot-path cost"。staging 化:同世代 3 chunk 值逐位相同,chunk2/3 直接
    # 命中;跨世代 miss 走 pinned 非阻塞。值语义与快照来源不变。
    # ★ 仅 writer-graph ON（默认）启用:ON 下 writer 消费 _ensure_selector_writer_*
    # stable buffer,本张量只是同流 copy_ 的瞬时 src（copy 后无 deferred 读者,
    # 跨世代复写受流序保护）。OFF 逃生旋钮下 writer_launch_args 直接持有该张量
    # 且 split-release 使 writer 延到次步 launch——staging 复用=跨世代脏读 race,
    # 故 OFF 保持原每世代独立分配（正确性优先,该旋钮不追性能）。
    if _ASYNC_PRODUCER_WRITER_GRAPH_CACHED:
        # [SEQLENS-STAGING-UAF-FIX 2026-07-07] 原共享 staging(cache_name=
        # "swk_rebuild_seq_lens_ref")的安全前提"copy 后无 deferred 读者"已被
        # deferred-writer 架构打破:本函数整体在 deferred drain(主流)执行,
        # 而共享 GPU stage 的容量换代直接替换引用(旧 storage 无 record_stream
        # 即被 GC),单 WAR 事件只护 H2D 不护跨流消费——4B bs8x12k bootstrap
        # 错峰世代密集时 deferred 窗跨换代点 → 读已释放 storage = illegal
        # address(CUDA_LAUNCH_BLOCKING 血栈钉在下游 view.copy_(src))。
        # 修:本站点弃共享 staging,独立 pinned 小分配(bs×4B/世代,µs 级)
        # 非阻塞 H2D。src/GPU 张量均由本函数局部引用持有到全部消费(persistent
        # copy_)入队之后;释放后 caching allocator 的 stream-aware 复用保证
        # 同流序安全,无跨流读者——零共享零 WAR 窗,值语义与快照来源不变。
        _seq_cpu_pin = _new_pinned_i32_tensor(
            tuple(max(0, int(seq_lens_cpu_ref[i])) for i in range(batch))
        )
        seq_lens_tensor_ref = torch.empty(
            (batch,), device=device, dtype=torch.int32
        )
        seq_lens_tensor_ref.copy_(_seq_cpu_pin, non_blocking=True)
    else:
        seq_lens_tensor_ref = torch.tensor(
            [max(0, int(seq_lens_cpu_ref[i])) for i in range(batch)],
            device=device,
            dtype=torch.int32,
        )
    # [DEAD-KERNEL-RETIRED 2026-07-05] 原此处的 rebuild_mask_tensor_ref（ge 比较）
    # 与 sink_len_tensor_ref（clamp）两个 GPU kernel 已删除：输出仅赋给局部变量
    # 再无任何消费者（mask/sink/persist 由 CUDA gather kernel 内部计算，见下方
    # kv_len 推导注释）——每 chunk 白付 2 次 launch 占 refresh_stream。

    if rebuild_pos_ref:
        for i, slot in enumerate(slot_list):
            if not rebuild_mask_cpu_ref[i]:
                continue
            seq_len = int(seq_lens_cpu_ref[i])
            seq_len_safe = max(seq_len, 0)
            sink_len = min(seq_len_safe, sink_cap)
            if recent_cfg > 0 and block_size_ref > 0:
                cap = min(seq_len_safe, recent_cfg)
                recent_start = ((seq_len_safe - cap) // block_size_ref) * block_size_ref
            else:
                recent_start = 0
            allowed_len = max(0, recent_start - sink_len)
            persist_len = min(k_head_cfg, allowed_len, int(selected_indices.shape[-1]))
            kv_len = sink_len + persist_len
            sink_lens_ref.append(int(sink_len))
            persist_lens_ref.append(int(persist_len))
            kv_lens_ref.append(int(kv_len))
    # [WRITER-TELEMETRY-GATE 2026-07-09] 遥测聚合(3×sum+字节数学+17 setattr,
    # cProfile 实测 ~200µs/chunk)唯一读者=flush_worker
    # _capture_writer_kernel_variant,且只在 profile accum 档被调——非取证档
    # 纯浪费,按同一 latch 门控。sink/persist/kv_lens_ref 列表是 per-layer
    # meta 写回的功能输入,保持无条件计算。
    if getattr(self, "_active_flush_profile_accum", None) is not None:
        writer_sink_tokens = int(sum(sink_lens_ref)) * int(layers) * int(num_kv_heads)
        writer_persist_tokens = int(sum(persist_lens_ref)) * int(layers) * int(num_kv_heads)
        writer_actual_tokens = int(sum(kv_lens_ref)) * int(layers) * int(num_kv_heads)
        writer_dtype_bytes = int(first.key_cache.element_size()) if first.key_cache is not None else 0
        writer_bytes_per_token = int(head_dim_ref) * int(writer_dtype_bytes) * 4 + 4
        writer_sink_io_bytes = int(writer_sink_tokens) * int(writer_bytes_per_token)
        writer_persist_io_bytes = int(writer_persist_tokens) * int(writer_bytes_per_token)
        writer_k_read_bytes = int(writer_actual_tokens) * int(head_dim_ref) * int(writer_dtype_bytes)
        writer_v_read_bytes = int(writer_k_read_bytes)
        writer_k_write_bytes = int(writer_k_read_bytes)
        writer_v_write_bytes = int(writer_k_read_bytes)
        writer_pos_write_bytes = int(writer_actual_tokens) * 4
        writer_total_io_bytes = (
            int(writer_k_read_bytes)
            + int(writer_v_read_bytes)
            + int(writer_k_write_bytes)
            + int(writer_v_write_bytes)
            + int(writer_pos_write_bytes)
        )
        setattr(self, "_last_writer_sink_tokens", int(writer_sink_tokens))
        setattr(self, "_last_writer_persist_tokens", int(writer_persist_tokens))
        setattr(self, "_last_writer_sink_io_bytes", int(writer_sink_io_bytes))
        setattr(self, "_last_writer_persist_io_bytes", int(writer_persist_io_bytes))
        setattr(self, "_last_writer_token_tiles_estimated", 0)
        setattr(self, "_last_writer_active_token_tiles_estimated", 0)
        setattr(self, "_last_writer_cta_count_estimated", 0)
        setattr(self, "_last_writer_active_cta_count_estimated", 0)
        setattr(self, "_last_writer_tokens_per_cta", 0)
        setattr(self, "_last_writer_actual_tokens", int(writer_actual_tokens))
        setattr(self, "_last_writer_k_read_bytes", int(writer_k_read_bytes))
        setattr(self, "_last_writer_v_read_bytes", int(writer_v_read_bytes))
        setattr(self, "_last_writer_k_write_bytes", int(writer_k_write_bytes))
        setattr(self, "_last_writer_v_write_bytes", int(writer_v_write_bytes))
        setattr(self, "_last_writer_pos_write_bytes", int(writer_pos_write_bytes))
        setattr(self, "_last_writer_total_io_bytes", int(writer_total_io_bytes))
        setattr(self, "_last_writer_effective_io_gbps", -1.0)
    layer_infos: List[Tuple] = []  # tuple: (payload, has_rebuild, row_tensor, reset_slot_commits)
    # [DETERMINISTIC-SLOT-SOURCE 2026-07-03] row_list 值→device 张量缓存。
    # writer-graph ON 时，每个 LayerState 持有其当前 row 值的 staging tensor；
    # 本调用的 dict 只负责让同值 layer 共享同一 tensor。值不变时没有 H2D、
    # rebind 或 graph capture；值变化时仅换代对应 layer，旧 storage 在冷路径
    # 做跨流 UAF 守卫。缓存规模由实际 LayerState 数量天然限定，不保存历史
    # row 组合，也没有 clear-all。OFF 逃生旋钮仍保持 per-call 独立张量语义。
    _row_tensor_by_key: Dict[Tuple[int, ...], torch.Tensor] = {}
    _row_h2d_sources: List[torch.Tensor] = []
    _row_guarded_old_tensor_ids: Set[int] = set()
    max_len = 0
    flat_k_strides: Optional[Tuple[int, int, int]] = None
    flat_v_strides: Optional[Tuple[int, int, int]] = None
    compact_k_strides: Optional[Tuple[int, int, int]] = None
    compact_v_strides: Optional[Tuple[int, int, int]] = None
    compact_pos_strides: Optional[Tuple[int, int]] = None

    if use_pinned_ptrs:
        # Build signatures first; acquire pinned CPU staging only on a cache miss.
        flat_k_ptrs_cpu = None
        flat_v_ptrs_cpu = None
        compact_k_ptrs_cpu = None
        compact_v_ptrs_cpu = None
        compact_pos_ptrs_cpu = None
        flat_k_ptrs = None
        flat_v_ptrs = None
        compact_k_ptrs = None
        compact_v_ptrs = None
        compact_pos_ptrs = None
    elif use_ptr_arrays:
        flat_k_ptrs_cpu = None
        flat_v_ptrs_cpu = None
        compact_k_ptrs_cpu = None
        compact_v_ptrs_cpu = None
        compact_pos_ptrs_cpu = None
        flat_k_ptrs = torch.empty((layers,), device=device, dtype=torch.int64)
        flat_v_ptrs = torch.empty((layers,), device=device, dtype=torch.int64)
        compact_k_ptrs = torch.empty((layers,), device=device, dtype=torch.int64)
        compact_v_ptrs = torch.empty((layers,), device=device, dtype=torch.int64)
        compact_pos_ptrs = torch.empty((layers,), device=device, dtype=torch.int64)
    else:
        flat_k_ptrs_cpu = None
        flat_v_ptrs_cpu = None
        compact_k_ptrs_cpu = None
        compact_v_ptrs_cpu = None
        compact_pos_ptrs_cpu = None
        flat_k_ptrs = None
        flat_v_ptrs = None
        compact_k_ptrs = None
        compact_v_ptrs = None
        compact_pos_ptrs = None
    flat_k_ptr_sig: List[object] = []
    flat_v_ptr_sig: List[object] = []
    compact_k_ptr_sig: List[object] = []
    compact_v_ptr_sig: List[object] = []
    compact_pos_ptr_sig: List[object] = []
    block_table_ptr_sig: List[object] = []
    flat_k_ptr_values: List[int] = []
    flat_v_ptr_values: List[int] = []
    compact_k_ptr_values: List[int] = []
    compact_v_ptr_values: List[int] = []
    compact_pos_ptr_values: List[int] = []
    block_table_ptr_values: List[int] = []
    def _layer_cache_signature(
        payload: SelectorBatchPayload,
        state: object,
        layer_idx: int,
    ) -> tuple[object, ...]:
        residency_sig = getattr(state, "compact_page_residency_signature", None)
        if isinstance(residency_sig, (list, tuple)):
            residency_sig_key = tuple(residency_sig)
        else:
            residency_sig_key = residency_sig
        # [PTR-SIG-DEGEN 2026-07-09] 删两个恒增世代计数:compact_generation/
        # compact_page_residency_generation 每 refresh 必增,把六组 ptr buffer
        # 的签名打成每 chunk 恒 miss→全量 republish(6×H2D+event/chunk,cProfile
        # 实测 ~490µs/chunk,连带 uaf 守卫/current_stream/env 读 6×)。ptr buffer
        # 内容=per-layer 基址,与世代无关:双代写偏移走 slot_tensor 的 sub-slot
        # 编码(DUAL-GEN-L2a-B),翻代只改 narrow 偏移不改基址(取证定谳)。
        # 失效链不减弱:六签名均为 (data_ptr, layer_cache_sig) 对——realloc/
        # rebind 由 data_ptr fail-close,物理页迁移由 residency_sig 承担。
        return (
            getattr(payload, "cache_key", None),
            int(getattr(payload, "layer_index", layer_idx)),
            int(getattr(state, "layer_index", layer_idx)),
            residency_sig_key,
        )

    # [DETERMINISTIC-SLOT-SOURCE 2026-07-03] rebuild/gather writer 的 slot 输入
    # 恒从 slot_list(host 权威值快照)构造,勿改回 payload 的
    # slot_tensor/slot_tensor_i32(per-batch 复用 buffer 的 live 视图,错峰
    # bootstrap 下一行 flush 原位覆写;writer 为 deferred 消费者,脏 slot=
    # compact KV 写进错误 slot 的 arena 行,跨序列静默腐蚀)。
    # [SLOT-STAGING-UAF-FIX 2026-07-07] 原 [REBUILD-H2D-STAGING] 共享 staging
    # (cache_name="swk_rebuild_slot_tensor")与 seq_lens 同族同拆:其安全前提
    # "copy 后无 deferred 读者"同样被 deferred-drain 主流化打破——共享 GPU
    # stage 容量换代直接替换引用(旧 storage 无 record_stream 即 GC),单 WAR
    # 事件只护 H2D 不护跨流消费;4B bs8x12k bootstrap 错峰 batch 翻倍换代窗
    # 与 seq_lens 雷同窗,且毒 slot 直接喂 gather kernel 的 dst 寻址
    # (dst_token=slot*stride+t 写 + dst_pos OOB 读)=illegal address 嫌疑#1。
    # 修同款:独立 pinned 小分配(bs×8B/世代)非阻塞 H2D,局部引用持有至
    # stable copy_ 入队,零共享零 WAR 窗;值语义不变(仍恒取自 slot_list
    # 快照)。graph ON 下 writer 消费 _ensure_selector_writer_slot_tensor_all
    # stable buffer,本张量只是同流 copy_ 的瞬时 src。OFF 逃生旋钮回退原
    # 独立分配（OFF 下游 long→int32 cast 虽恒新分配,仍同姿态防御）。
    # [DUAL-GEN-L2a-B] gather writer 的物理写目标=备用半区 sub-slot(单代=
    # slot 逐位)。slot_tensor 仅有两个消费者(ptr_gather_args 写偏移+profile
    # data_ptr),逻辑记账全走 rebuild_slots_ref,故构造点值变换即安全。
    # write_gen 取 first.state(commit 按 pending 全层同批翻代,层间一致)。
    writer_slot_values = slot_list
    if compact_gen_count() > 1:
        # [DUAL-GEN-REBIND-NORMALIZE 2026-07-08] 楔死案根修(出生点):slot 重绑
        # (bootstrap)时先把该 slot 的 read_gen 在本批全部层归一为 0,再算写
        # 偏移。跨 req 生命周期的"部分世代落地"(req 完成→在飞世代部分 chunk
        # 票已 commit 翻代、余票被 is_latest 正确作废)让层间 read_gen 错开;
        # 旧 req 退场后该错开无内容语义,但新 req 的 bootstrap 若继承它:写
        # 偏移取 first.state(下方注释明言依赖"全层同批翻代,层间一致")而
        # commit 按各层现值翻代→32k×TP2 取证实锤 28/8 chunk 边界错代+parity
        # 断言单 rank 开火(非 output rank 异常被 vLLM 吞→TP collective
        # desync 楔死)。重绑=世代计数重启:归一后全层写 half1、commit 翻至
        # 1,决定论且 rank 无关;首次 bootstrap 全 0=无操作,黄金/双代 12k
        # 判据零扰动。生命周期内的部分落地(下代按层追平)不经此路,语义照旧。
        _rebind_slots = set()
        for _bs in bootstrap_slots_by_layer:
            if _bs:
                _rebind_slots.update(int(v) for v in _bs)
        if _rebind_slots:
            for _p in payloads:
                _rg_norm = getattr(_p.state, "compact_read_gen", None)
                if _rg_norm is None:
                    continue
                for _rebind_slot in _rebind_slots:
                    if 0 <= _rebind_slot < len(_rg_norm):
                        _rg_norm[_rebind_slot] = 0
        _dg_residency = getattr(first.state, "compact_page_residency", None)
        if _dg_residency is None:
            raise RuntimeError(
                "compact dual-gen requires page residency (legacy arena unsupported)"
            )
        _dg_max_live = int(_dg_residency.lease.max_live_sparse_slots)
        _dg_read_gen = first.state.compact_read_gen
        writer_slot_values = [
            compact_write_sub_slot(
                slot=int(s),
                read_gen=int(_dg_read_gen[int(s)]) if int(s) < len(_dg_read_gen) else 0,
                gen_count=compact_gen_count(),
                max_live_slots=_dg_max_live,
            )
            for s in slot_list
        ]
    if _ASYNC_PRODUCER_WRITER_GRAPH_CACHED:
        _slot_cpu_pin = _new_pinned_i64_tensor(
            tuple(int(s) for s in writer_slot_values)
        )
        slot_tensor = torch.empty(
            (len(writer_slot_values),), device=device, dtype=torch.long
        )
        slot_tensor.copy_(_slot_cpu_pin, non_blocking=True)
    else:
        slot_tensor = torch.tensor(writer_slot_values, device=device, dtype=torch.long)
    slot_tensor_i32_ref = None
    max_slot = max(slot_list) if slot_list else -1

    # R3+ 优化：首层做完整验证 + stride 记录，后续层跳过所有冗余检查和 .stride() 调用
    # block_size/head_dim/dtype/strides 在模型加载后固定，跨层一致
    stride_tokens_cached = 0
    has_rebuild = bool(has_rebuild_any_ref)

    for layer_idx, payload in enumerate(payloads):
        block_size = block_size_ref
        head_dim = head_dim_ref
        state = payload.state

        if layer_idx == 0:
            # 首层：完整检查
            if [int(slot) for slot in payload.slot_list] != slot_list:
                return _fail_rebuild_contract("slot_list_mismatch")
            if state.num_kv_heads != num_kv_heads:
                return _fail_rebuild_contract("num_kv_heads_mismatch")
            if payload.key_cache.device != device or payload.value_cache.device != device:
                return _fail_rebuild_contract("kv_cache_device_mismatch")
            if payload.key_cache.dtype != kv_dtype_ref or payload.value_cache.dtype != kv_dtype_ref:
                return _fail_rebuild_contract("kv_cache_dtype_mismatch")
            if payload.key_cache.dim() < 2:
                return _fail_rebuild_contract("layer_key_cache_rank_lt_2")
            if int(payload.key_cache.shape[1]) != block_size_ref or int(payload.key_cache.shape[-1]) != head_dim_ref:
                return _fail_rebuild_contract("key_cache_shape_mismatch")
            if len(payload.row_list) != batch:
                return _fail_rebuild_contract("row_list_batch_mismatch")
            seq_lens_cpu = payload.seq_lens_cpu
            if seq_lens_cpu is None or len(seq_lens_cpu) != batch:
                return _fail_rebuild_contract("layer_seq_lens_cpu_missing_or_wrong_batch")
            if seq_lens_cpu != seq_lens_cpu_ref:
                return _fail_rebuild_contract("layer_seq_lens_cpu_mismatch")
            stride_tokens_cached = _compact_gather_stride_tokens(
                self,
                state,
                block_size,
            )
            if stride_tokens_cached <= 0:
                return _fail_rebuild_contract("compact_stride_tokens_nonpositive")
            max_len = stride_tokens_cached

        if max_slot >= 0:
            state.ensure_batch(max_slot + 1)
        self._ensure_compact_capacity(
            state, max_slot, stride_tokens_cached, state.num_kv_heads,
            head_dim, payload.key_cache.dtype, block_size,
        )

        reset_slot_commits: Tuple[int, ...] = tuple()
        if reset_slots_ref:
            if defer_compact_meta_publish:
                reset_slot_commits = tuple(int(slot) for slot in reset_slots_ref)
            else:
                for slot in reset_slots_ref:
                    self._reset_compact_slot(state, slot)

        flat_tokens = payload.key_cache.shape[0] * block_size
        flat_k = payload.key_cache.view(flat_tokens, num_kv_heads, head_dim)
        flat_v = payload.value_cache.view(flat_tokens, num_kv_heads, head_dim)
        compact_k = state.compact_arena_k
        compact_v = state.compact_arena_v
        compact_pos = state.compact_arena_pos
        layer_cache_sig = _layer_cache_signature(payload, state, layer_idx)

        if layer_idx == 0:
            # 首层：记录 strides + 完整设备/类型/stride 校验
            flat_k_strides = (int(flat_k.stride(0)), int(flat_k.stride(1)), int(flat_k.stride(2)))
            flat_v_strides = (int(flat_v.stride(0)), int(flat_v.stride(1)), int(flat_v.stride(2)))
            if compact_k.device != device or compact_v.device != device or compact_pos.device != device:
                return _fail_rebuild_contract("compact_arena_device_mismatch")
            if compact_k.dtype != kv_dtype_ref or compact_v.dtype != kv_dtype_ref:
                return _fail_rebuild_contract("compact_arena_dtype_mismatch")
            compact_k_strides = (int(compact_k.stride(0)), int(compact_k.stride(1)), int(compact_k.stride(2)))
            compact_v_strides = (int(compact_v.stride(0)), int(compact_v.stride(1)), int(compact_v.stride(2)))
            compact_pos_strides = (int(compact_pos.stride(0)), int(compact_pos.stride(1)))
        # 后续层：跳过 stride 比较（模型架构不变 → strides 不变），只收集 data_ptr

        if use_pinned_ptrs:
            flat_k_ptr = int(flat_k.data_ptr())
            flat_v_ptr = int(flat_v.data_ptr())
            compact_k_ptr = int(compact_k.data_ptr())
            compact_v_ptr = int(compact_v.data_ptr())
            compact_pos_ptr = int(compact_pos.data_ptr())
            flat_k_ptr_values.append(flat_k_ptr)
            flat_v_ptr_values.append(flat_v_ptr)
            compact_k_ptr_values.append(compact_k_ptr)
            compact_v_ptr_values.append(compact_v_ptr)
            compact_pos_ptr_values.append(compact_pos_ptr)
            flat_k_ptr_sig.append((flat_k_ptr, layer_cache_sig))
            flat_v_ptr_sig.append((flat_v_ptr, layer_cache_sig))
            compact_k_ptr_sig.append((compact_k_ptr, layer_cache_sig))
            compact_v_ptr_sig.append((compact_v_ptr, layer_cache_sig))
            compact_pos_ptr_sig.append((compact_pos_ptr, layer_cache_sig))
        elif use_ptr_arrays:
            assert flat_k_ptrs is not None
            assert flat_v_ptrs is not None
            assert compact_k_ptrs is not None
            assert compact_v_ptrs is not None
            assert compact_pos_ptrs is not None
            flat_k_ptrs[layer_idx] = int(flat_k.data_ptr())
            flat_v_ptrs[layer_idx] = int(flat_v.data_ptr())
            compact_k_ptrs[layer_idx] = int(compact_k.data_ptr())
            compact_v_ptrs[layer_idx] = int(compact_v.data_ptr())
            compact_pos_ptrs[layer_idx] = int(compact_pos.data_ptr())

        # [DETERMINISTIC-SLOT-SOURCE 2026-07-03] row 输入恒从 row_list(host 权威
        # 值快照)构造,勿改回 payload 的 row_tensor(_i32)(复用 buffer live 视图,
        # 错峰下被覆写;writer 为 deferred 消费者,脏 row=从错误序列的物理块
        # gather KV 进 compact arena,跨序列静默腐蚀)。按值缓存:各层 row_list
        # 相同时只发一次 H2D。
        _row_key = tuple(int(r) for r in payload.row_list)
        _row_tensor_layer = _row_tensor_by_key.get(_row_key)
        if _ASYNC_PRODUCER_WRITER_GRAPH_CACHED:
            _state_row_key = getattr(state, "_rebuild_row_tensor_cache_key", None)
            _state_row_tensor = getattr(state, "_rebuild_row_tensor_cache", None)
            if _row_tensor_layer is None:
                if _state_row_key == _row_key and isinstance(_state_row_tensor, torch.Tensor):
                    _row_tensor_layer = _state_row_tensor
                else:
                    _row_cpu_pin = _new_pinned_i32_tensor(_row_key)
                    _row_tensor_layer = torch.empty(
                        (len(_row_key),), device=device, dtype=torch.int32
                    )
                    _row_tensor_layer.copy_(_row_cpu_pin, non_blocking=True)
                    # Hold the pinned source through all stable-buffer copies
                    # enqueued by this call.
                    _row_h2d_sources.append(_row_cpu_pin)
                _row_tensor_by_key[_row_key] = _row_tensor_layer
            if _state_row_tensor is not _row_tensor_layer:
                if (
                    isinstance(_state_row_tensor, torch.Tensor)
                    and _state_row_tensor.is_cuda
                    and id(_state_row_tensor) not in _row_guarded_old_tensor_ids
                ):
                    self._uaf_guard_record_streams_before_discard(_state_row_tensor)
                    _row_guarded_old_tensor_ids.add(id(_state_row_tensor))
                state._rebuild_row_tensor_cache_key = _row_key
                state._rebuild_row_tensor_cache = _row_tensor_layer
        elif _row_tensor_layer is None:
            _row_tensor_layer = torch.tensor(
                payload.row_list, device=device, dtype=torch.int32
            )
            _row_tensor_by_key[_row_key] = _row_tensor_layer
        # info[4]=layer_cache_sig：本层签名只算一次，第二遍 per-layer 校验循环复用。
        layer_infos.append((payload, has_rebuild, _row_tensor_layer, reset_slot_commits, layer_cache_sig))

    if _pd_mark is not None:
        _pd_mark("wr_setup_layers")
    if max_len <= 0:
        return _fail_rebuild_contract("max_len_nonpositive")
    if not any(info[1] for info in layer_infos):  # info[1] = has_rebuild
        if defer_compact_meta_publish and reset_slots_ref:
            assert compact_meta_commit_log is not None
            for layer_idx, info in enumerate(layer_infos):
                compact_meta_commit_log.append(
                    {
                        "state": info[0].state,
                        "phase": str(phase),
                        "reset_slot_commits": info[3],
                        "slot_meta_commits": tuple(),
                        "pad_marker_commits": tuple(),
                        "bootstrap_slots": tuple(
                            int(v)
                            for v in tuple(
                                bootstrap_slots_by_layer[layer_idx] or tuple()
                            )
                        ),
                        "slot_owner_snapshot": slot_owner_snapshot,
                    }
                )
        return True

    stride_tokens_ref = stride_tokens_cached
    writer_active_max_tokens = (
        min(int(stride_tokens_ref), max(int(kv_len) for kv_len in kv_lens_ref))
        if kv_lens_ref
        else 0
    )
    # [WRITER-TELEMETRY-GATE 2026-07-09] tile/CTA 估算段=纯遥测(唯一读者
    # =_capture_writer_kernel_variant,profile accum 档),同 latch 门控;
    # writer_active_max_tokens 是 launch 功能输入,保持在门控外。
    if getattr(self, "_active_flush_profile_accum", None) is not None:
        writer_tokens_per_cta = 0
        writer_token_tiles_estimated = 0
        writer_active_token_tiles_estimated = 0
        writer_cta_count_estimated = 0
        writer_active_cta_count_estimated = 0
        if int(stride_tokens_ref) > 0:
            requested_tile_tokens = int(_WRITER_TOKEN_TILE_CACHED)
            if requested_tile_tokens > 0:
                writer_grid_tokens = max(1, int(writer_active_max_tokens))
                max_grid_z = 65535
                min_tile_for_grid_z = (int(writer_grid_tokens) + max_grid_z - 1) // max_grid_z
                writer_tokens_per_cta = max(1, int(requested_tile_tokens), int(min_tile_for_grid_z))
                writer_token_tiles_estimated = max(
                    1,
                    (int(writer_grid_tokens) + int(writer_tokens_per_cta) - 1)
                    // int(writer_tokens_per_cta),
                )
                writer_active_token_tiles_estimated = sum(
                    (min(max(0, int(kv_len)), int(stride_tokens_ref)) + int(writer_tokens_per_cta) - 1)
                    // int(writer_tokens_per_cta)
                    for kv_len in kv_lens_ref
                    if int(kv_len) > 0
                )
            else:
                writer_tokens_per_cta = int(stride_tokens_ref)
                writer_token_tiles_estimated = 1 if batch > 0 else 0
                writer_active_token_tiles_estimated = sum(1 for kv_len in kv_lens_ref if int(kv_len) > 0)
            writer_cta_count_estimated = (
                int(layers) * int(batch) * int(num_kv_heads) * int(writer_token_tiles_estimated)
            )
            writer_active_cta_count_estimated = (
                int(layers) * int(num_kv_heads) * int(writer_active_token_tiles_estimated)
            )
        setattr(self, "_last_writer_token_tiles_estimated", int(writer_token_tiles_estimated))
        setattr(self, "_last_writer_active_token_tiles_estimated", int(writer_active_token_tiles_estimated))
        setattr(self, "_last_writer_cta_count_estimated", int(writer_cta_count_estimated))
        setattr(self, "_last_writer_active_cta_count_estimated", int(writer_active_cta_count_estimated))
        setattr(self, "_last_writer_tokens_per_cta", int(writer_tokens_per_cta))

    block_table_ref = payloads[0].block_table
    if block_table_ref.dtype != torch.int32:
        return _fail_rebuild_contract("block_table_dtype_not_int32")
    if block_table_ref.dim() != 2 or block_table_ref.shape[0] < batch:
        return _fail_rebuild_contract("block_table_shape_invalid")
    block_table_cols = int(block_table_ref.shape[1])
    if block_table_cols <= 0:
        return _fail_rebuild_contract("block_table_cols_nonpositive")
    block_table_strides = (int(block_table_ref.stride(0)), int(block_table_ref.stride(1)))

    if use_pinned_ptrs:
        block_table_ptrs_cpu = None
        block_table_ptrs = None
    elif use_ptr_arrays:
        block_table_ptrs_cpu = None
        block_table_ptrs = torch.empty((layers,), device=device, dtype=torch.int64)
    else:
        block_table_ptrs_cpu = None
        block_table_ptrs = None
    # --------------------------------------------------------------
    # A 路线优化：sink_len/persist_len/row_tensor 在不同 layer 之间通常完全一致（同一 batch 的同一批 slot）。
    # 这里尽量用一次性张量计算替代 per-layer 多次 clamp/算术（减少 kernel launch）。
    # --------------------------------------------------------------
    persist_len_max = int(selected_indices.shape[-1])
    # 用 CPU seq_lens 推导上界，避免任何 `.max().item()` 同步点。
    seq_lens_cpu_ref = payloads[0].seq_lens_cpu
    if seq_lens_cpu_ref is None or len(seq_lens_cpu_ref) != batch:
        return _fail_rebuild_contract("seq_lens_cpu_ref_invalid_before_bounds")
    max_sink_cpu = min(sink_cap, max(seq_lens_cpu_ref)) if seq_lens_cpu_ref else 0
    if max_sink_cpu + persist_len_max > max_len:
        return _fail_rebuild_contract(
            f"compact_bounds_exceed_capacity max_sink={max_sink_cpu} persist={persist_len_max} max_len={max_len}"
        )

    # block_table 指针与 selected shape 校验仍需 per-layer 走一遍（保证安全）。
    for layer_idx, info in enumerate(layer_infos):
        payload = info[0]  # info[0] = payload
        state = payload.state
        layer_cache_sig = info[4]  # loop1 已算，复用（同 payload 同 state 同步内不变）
        selected_layer = selected_indices[layer_idx]
        if selected_layer.shape[0] != batch or selected_layer.shape[1] != num_kv_heads:
            return _fail_rebuild_contract("selected_layer_shape_mismatch")
        block_table = payload.block_table
        if (
            block_table.dtype != block_table_ref.dtype
            or block_table.shape != block_table_ref.shape
            or block_table.stride(0) != block_table_strides[0]
            or block_table.stride(1) != block_table_strides[1]
        ):
            return _fail_rebuild_contract("block_table_layout_mismatch")
        if use_pinned_ptrs:
            block_table_ptr = int(block_table.data_ptr())
            block_table_ptr_values.append(block_table_ptr)
            block_table_ptr_sig.append((block_table_ptr, layer_cache_sig))
        elif use_ptr_arrays:
            assert block_table_ptrs is not None
            block_table_ptrs[layer_idx] = int(block_table.data_ptr())

    if use_pinned_ptrs:
        def _get_or_publish_ptr_buffer(
            *,
            name: str,
            values: Sequence[int],
            signature: Sequence[object],
        ) -> Tuple[torch.Tensor, bool]:
            if len(values) != layers:
                raise RuntimeError(
                    f"rebuild pointer value count mismatch name={name} "
                    f"values={len(values)} layers={layers}"
                )
            signature_tuple = tuple(signature)
            cached_gpu = self._get_rebuild_ptr_gpu_if_signature_ready(
                name=name,
                device=device,
                size=layers,
                signature=signature_tuple,
            )
            if cached_gpu is not None:
                return cached_gpu, False

            cpu, gpu = self._get_rebuild_ptr_buffers(
                name=name,
                device=device,
                size=layers,
            )
            for ptr_idx, ptr_value in enumerate(values):
                cpu[ptr_idx] = int(ptr_value)
            published = self._publish_rebuild_ptr_buffer_if_needed(
                name=name,
                cpu=cpu,
                gpu=gpu,
                signature=signature_tuple,
            )
            return gpu, bool(published)

        pointer_rebuild_miss = False
        flat_k_ptrs, _miss = _get_or_publish_ptr_buffer(
            name=flat_k_ptrs_name,
            values=flat_k_ptr_values,
            signature=flat_k_ptr_sig,
        )
        pointer_rebuild_miss |= _miss
        flat_v_ptrs, _miss = _get_or_publish_ptr_buffer(
            name=flat_v_ptrs_name,
            values=flat_v_ptr_values,
            signature=flat_v_ptr_sig,
        )
        pointer_rebuild_miss |= _miss
        compact_k_ptrs, _miss = _get_or_publish_ptr_buffer(
            name=compact_k_ptrs_name,
            values=compact_k_ptr_values,
            signature=compact_k_ptr_sig,
        )
        pointer_rebuild_miss |= _miss
        compact_v_ptrs, _miss = _get_or_publish_ptr_buffer(
            name=compact_v_ptrs_name,
            values=compact_v_ptr_values,
            signature=compact_v_ptr_sig,
        )
        pointer_rebuild_miss |= _miss
        compact_pos_ptrs, _miss = _get_or_publish_ptr_buffer(
            name=compact_pos_ptrs_name,
            values=compact_pos_ptr_values,
            signature=compact_pos_ptr_sig,
        )
        pointer_rebuild_miss |= _miss
        block_table_ptrs, _miss = _get_or_publish_ptr_buffer(
            name=block_table_ptrs_name,
            values=block_table_ptr_values,
            signature=block_table_ptr_sig,
        )
        pointer_rebuild_miss |= _miss
    else:
        pointer_rebuild_miss = bool(use_ptr_arrays)
    if _pd_mark is not None:
        _pd_mark("wr_ptr_publish")

    # CUDA path: the op consumes logical selected_indices + row_tensor +
    # slot_tensor + block_tables and performs block_table lookup internally.
    # No Python-side logical->absolute conversion is needed.
    from utils.fa_sparse_runtime_ext import _load_ext as _load_fa_sparse_ext

    ext = _load_fa_sparse_ext()
    if ext is None or not hasattr(ext, "gather_compact_kv_into_arena"):
        raise RuntimeError("compact writer requires gather_compact_kv_into_arena")

    # seq_lens / row_tensor: [L,B]
    # 注意：rebuild_mask/sink_len/persist_len 改为由 CUDA gather kernel 内部计算（减少额外 kernel launch）。
    # R2 优化：seq_lens_tensor_ref 跨层共享，只做一次 to() 调用，避免 L 次冗余转换。
    seq_lens_base = seq_lens_tensor_ref[:batch]
    if seq_lens_base.device != device or seq_lens_base.dtype != torch.int32:
        seq_lens_base = seq_lens_base.to(device=device, dtype=torch.int32)

    def _to_i32_on_device(tensor: torch.Tensor) -> torch.Tensor:
        out = tensor[:batch]
        if out.device != device:
            out = out.to(device=device)
        if out.dtype != torch.int32:
            out = out.to(dtype=torch.int32)
        return out

    # R2 优化：检查 row_tensor 是否跨层共享（同一 batch 通常共享）
    row_tensor_first = _to_i32_on_device(layer_infos[0][2])  # info[2] = row_tensor
    can_broadcast_row = all(
        info[2].data_ptr() == layer_infos[0][2].data_ptr()
        for info in layer_infos[1:]
    )
    if can_broadcast_row:
        row_tensors = [row_tensor_first] * layers
    else:
        row_tensors = [_to_i32_on_device(info[2]) for info in layer_infos]

    row_tensor_all = (
        row_tensor_first.unsqueeze(0).expand(layers, -1)
        if can_broadcast_row
        else torch.stack(row_tensors, dim=0)
    )

    if _ASYNC_PRODUCER_WRITER_GRAPH_CACHED:
        # #9-KEY: copy into a persistent slice-view so the captured writer
        # replays a STABLE seq_lens data_ptr. Byte layout matches the OFF path.
        seq_lens_cuda = self._ensure_selector_writer_seq_lens_all(
            batch=batch,
            device=device,
            src=seq_lens_base,
        )
    else:
        seq_lens_cuda = (
            seq_lens_base.contiguous()
            if seq_lens_base.dtype == torch.int32
            else seq_lens_base.to(dtype=torch.int32).contiguous()
        )
    # Ensure row_tensor / slot_tensor are contiguous int32 for the CUDA writer.
    # ASYNC_PRODUCER_WRITER_GRAPH (defect #6): the eager path freshly allocates row_tensor_all via
    # expand().contiguous()/stack() (unstable data_ptr). When the writer graph
    # is ON, copy into a persistent slice-view so the captured launch replays a
    # stable pointer. OFF path is byte-identical to today.
    if _ASYNC_PRODUCER_WRITER_GRAPH_CACHED:
        row_tensor_cuda = self._ensure_selector_writer_row_tensor_all(
            layers=layers,
            batch=batch,
            device=device,
            row_tensor_first=row_tensor_first,
            per_layer_rows=None if can_broadcast_row else row_tensors,
        )
    else:
        row_tensor_cuda = (
            row_tensor_all.contiguous()
            if row_tensor_all.dtype == torch.int32
            else row_tensor_all.to(dtype=torch.int32).contiguous()
        )
    if _ASYNC_PRODUCER_WRITER_GRAPH_CACHED:
        # #9-KEY v2: copy slot_tensor into a persistent slice-view so the captured
        # writer replays a STABLE slot_tensor data_ptr. Byte layout matches OFF.
        slot_tensor_src = (
            slot_tensor_i32_ref
            if slot_tensor_i32_ref is not None
            else slot_tensor
        )
        slot_tensor_cuda = self._ensure_selector_writer_slot_tensor_all(
            batch=batch,
            device=device,
            src=slot_tensor_src,
        )
    elif slot_tensor_i32_ref is not None:
        slot_tensor_cuda = slot_tensor_i32_ref.contiguous()
    else:
        slot_tensor_cuda = (
            slot_tensor.to(dtype=torch.int32).contiguous()
            if slot_tensor.dtype != torch.int32
            else slot_tensor.contiguous()
        )

    def _record_selected_indices_materialization(num_bytes: int) -> None:
        try:
            prof = getattr(self, "_active_flush_profile_accum", None)
            if prof is None:
                return
            prof.selected_indices_materialized_bytes += int(num_bytes)
            prof.selected_indices_io_bytes += int(num_bytes) * 2
            prof.selector_writer_current_path_count += 1
        except Exception:
            _log.warning(
                "lastn1 fused producer attribution: selected index accounting failed",
                exc_info=True,
            )

    if _ASYNC_PRODUCER_WRITER_GRAPH_CACHED:
        # #9-KEY v2: copy selected_indices into a persistent slice-view so the
        # captured writer replays a STABLE selected data_ptr even when the
        # post-topk out-buffer reallocs on k_head/batch jitter. Byte layout
        # (contiguous row-major [L,B,H,k]) matches the OFF .contiguous() path.
        selected_cuda = self._ensure_selector_writer_selected_all(
            layers=layers,
            batch=batch,
            num_kv_heads=num_kv_heads,
            k_head=int(selected_indices.shape[-1]),
            device=device,
            src=selected_indices,
        )
        # #9-KEY v3: latch the three writer-input copies on the producing
        # stream (defensive same-stream no-op; drained before replay).
        self._record_writer_input_ready(device=device)
    else:
        selected_indices_is_contiguous = bool(selected_indices.is_contiguous())
        if selected_indices_is_contiguous:
            selected_cuda = selected_indices
        else:
            selected_cuda = selected_indices.contiguous()
            if getattr(self, "_active_flush_profile_accum", None) is not None:
                selected_indices_bytes = int(selected_indices.numel()) * int(
                    selected_indices.element_size()
                )
                _record_selected_indices_materialization(selected_indices_bytes)
    use_cuda_ptr_op = bool(
        use_pinned_ptrs
        and flat_k_ptrs is not None
        and flat_v_ptrs is not None
        and compact_k_ptrs is not None
        and compact_v_ptrs is not None
        and compact_pos_ptrs is not None
        and block_table_ptrs is not None
    )
    if not use_cuda_ptr_op:
        raise RuntimeError(
            "compact writer requires cached pointer autolen gather op; "
            "vector fallback path is retired"
        )

    writer_token_tile = int(_WRITER_TOKEN_TILE_CACHED)
    # [WRITER-ENQUEUE-DIET 2026-07-12] ext 入口存在性按 ext 对象身份验一次
    # (旧=每 dispatch hasattr);缺口 raise 语义逐字保留。
    if getattr(self, "_writer_ext_entry_checked_for", None) is not ext:
        if not hasattr(ext, "gather_compact_kv_into_arena_ptrs_tiled_autolen_skip_unchanged"):
            raise RuntimeError(
                "CUDA compact writer requires "
                "gather_compact_kv_into_arena_ptrs_tiled_autolen_skip_unchanged"
            )
        self._writer_ext_entry_checked_for = ext
    writer_kernel_variant = (
        "ptr_tiled_autolen_skip_unchanged"
        if writer_token_tile > 0
        else "ptr_autolen_skip_unchanged"
    )
    setattr(self, "_last_writer_kernel_variant", writer_kernel_variant)
    self._record_writer_pointer_lookup(
        pointer_rebuild=bool(pointer_rebuild_miss),
        cached_pointer_op=True,
        vector_fallback=False,
    )
    if prepare_for_capture_only:
        setattr(self, "_last_writer_kernel_variant", f"{writer_kernel_variant}_prepare_only")
        return True

    compact_pos_per_layer_tokens = int(layer_infos[0][0].state.compact_arena_pos.shape[1])
    ptr_gather_args = (
        flat_k_ptrs,
        flat_v_ptrs,
        compact_k_ptrs,
        compact_v_ptrs,
        compact_pos_ptrs,
        block_table_ptrs,
        row_tensor_cuda,
        slot_tensor_cuda,
        selected_cuda,
        seq_lens_cuda,
        int(block_size_ref),
        int(stride_tokens_ref),
        int(block_table_cols),
        int(head_dim_ref),
        int(compact_pos_per_layer_tokens),
        int((flat_k_strides or (0, 0, 0))[0]),
        int((flat_k_strides or (0, 0, 0))[1]),
        int((flat_k_strides or (0, 0, 0))[2]),
        int((flat_v_strides or (0, 0, 0))[0]),
        int((flat_v_strides or (0, 0, 0))[1]),
        int((flat_v_strides or (0, 0, 0))[2]),
        int((compact_k_strides or (0, 0, 0))[0]),
        int((compact_k_strides or (0, 0, 0))[1]),
        int((compact_k_strides or (0, 0, 0))[2]),
        int((compact_v_strides or (0, 0, 0))[0]),
        int((compact_v_strides or (0, 0, 0))[1]),
        int((compact_v_strides or (0, 0, 0))[2]),
        int((compact_pos_strides or (0, 0))[0]),
        int((compact_pos_strides or (0, 0))[1]),
        int(block_table_strides[0]),
        int(block_table_strides[1]),
    )
    # ASYNC_PRODUCER_WRITER_GRAPH (task #9): the writer launch tail. ``writer_launch_args`` is the EXACT
    # positional arg list of the single fused ext op. Eager launches use the
    # skip_unchanged variant; the captured graph re-issues the STATELESS
    # ``_tiled_autolen`` full-copy variant with the SAME args (#9-KEY v4(b)) so
    # replay is immune to compact_pos arena mutation — final arena bytes are
    # identical across eager / cold capture / hot replay.
    writer_token_tile_arg = (
        int(writer_token_tile) if int(writer_token_tile) > 0 else int(stride_tokens_ref)
    )
    # [WRITER-GRAPH-KEY v6] Canonicalize the host-only max_total_tokens scalar
    # to the CUDA launch geometry it actually produces. The extension uses this
    # scalar only to derive safe_tile_tokens and grid.z; it is NOT a kernel
    # argument. Therefore all raw prompt lengths in the same
    # (safe_tile_tokens, token_tiles) class are one exact CUDA graph identity.
    # Passing the class ceiling below emits the identical grid and identical
    # kernel arguments: no extra CTA, kernel, synchronization, or tensor work.
    _writer_requested_tile_tokens = max(1, int(writer_token_tile_arg))
    _writer_stride_tokens_i32 = max(1, int(stride_tokens_ref))
    _writer_active_tokens_i32 = min(
        _writer_stride_tokens_i32,
        max(1, int(writer_active_max_tokens)),
    )
    _writer_max_grid_z = 65535
    _writer_min_tile_for_grid_z = (
        _writer_active_tokens_i32 + _writer_max_grid_z - 1
    ) // _writer_max_grid_z
    writer_safe_tile_tokens = max(
        _writer_requested_tile_tokens,
        _writer_min_tile_for_grid_z,
    )
    writer_graph_token_tiles = max(
        1,
        (
            _writer_active_tokens_i32
            + int(writer_safe_tile_tokens)
            - 1
        )
        // int(writer_safe_tile_tokens),
    )
    writer_graph_max_tokens = min(
        _writer_stride_tokens_i32,
        int(writer_graph_token_tiles) * int(writer_safe_tile_tokens),
    )
    if _pd_mark is not None:
        _pd_mark("wr_stable_copies")
    writer_launch_args = (
        *ptr_gather_args,
        int(writer_token_tile_arg),
        int(writer_graph_max_tokens),
        int(sink_cap),
        int(recent_cfg),
        int(k_head_cfg),
        int(threshold),
    )

    def _eager_writer_launch() -> None:
        ext.gather_compact_kv_into_arena_ptrs_tiled_autolen_skip_unchanged(
            *writer_launch_args
        )

    if (
        os.environ.get("VLLM_SPARSE_WRITER_INPUT_BTABLE_CHECK") == "1"
        if _DYNAMIC_ENV
        else _WRITER_INPUT_BTABLE_CHECK_CACHED
    ):
        # [诊断档,默认关] launch 前 host 侧断言 btable 有效区非负。graph/eager
        # trap 后 CUDA printf 缓冲丢失拿不到坐标,这里用 python 栈+完整值取证;
        # 若本检查通过而 kernel 仍 trap = 毒写发生在 launch~执行窗内(跨流并发
        # 写实锤)。同步 D2H 仅诊断档开销。
        # [WRITER-ENQUEUE-DIET 2026-07-12] 热路径 env 读迁 import 期缓存
        # (_DYNAMIC_ENV 双臂惯例,pytest 域语义不变)。
        for _ci, _info in enumerate(layer_infos):
            _pl = _info[0]
            _rows = [int(r) for r in _pl.row_list]
            _bt_host = block_table_ref[_rows].cpu()
            for _bi, _r in enumerate(_rows):
                _need = (
                    max(0, int(seq_lens_cpu_ref[_bi])) + int(block_size_ref) - 1
                ) // int(block_size_ref)
                _need = min(_need, int(block_table_cols))
                if _need <= 0:
                    continue
                _vals = _bt_host[_bi, :_need]
                _mn = int(_vals.min().item())
                if _mn < 0:
                    _col = int((_vals < 0).nonzero()[0].item())
                    _wt = getattr(self, "_worker_block_table", None)
                    _wt_ptr = (
                        int(_wt.data_ptr()) if isinstance(_wt, torch.Tensor) else 0
                    )
                    _wt_row_head = (
                        _wt[_r, :4].cpu().tolist()
                        if isinstance(_wt, torch.Tensor) and _r < int(_wt.shape[0])
                        else None
                    )
                    raise RuntimeError(
                        "SFI_BTABLE_PRECHECK layer_pos=%d layer_idx=%s b=%d row=%d "
                        "col=%d blkid=%d need=%d seq=%d bt_ptr=0x%x wt_ptr=0x%x "
                        "same_table=%s bt_row_head=%s wt_row_head=%s bt_shape=%s"
                        % (
                            _ci,
                            getattr(getattr(_pl, "state", None), "layer_index", -1),
                            _bi,
                            _r,
                            _col,
                            _mn,
                            _need,
                            int(seq_lens_cpu_ref[_bi]),
                            int(block_table_ref.data_ptr()),
                            _wt_ptr,
                            bool(int(block_table_ref.data_ptr()) == _wt_ptr),
                            _bt_host[_bi, :4].tolist(),
                            _wt_row_head,
                            tuple(block_table_ref.shape),
                        )
                    )

    if not _ASYNC_PRODUCER_WRITER_GRAPH_CACHED:
        _eager_writer_launch()
    else:
        # Hot replay / cold capture / eager-fallback dispatcher (defects #1-#7).
        # The eager_fn re-issues the SAME bytes; the key is a single source of
        # truth driven by pointer_rebuild_miss + the 4 input data_ptrs + scalars.
        # payload-compaction (refresh_rebuild_mixin _compact_pending_refresh_payloads)
        # replaces seq_lens_batch with an empty carrier, so seq_lens reconstructs
        # fresh (see selection_worker seq_lens_tensor_ref branch) and its data_ptr
        # naturally changes -> key mismatch -> eager. No extra code needed.
        # #9-KEY v5: layer-group identity for the key. Refreshes flush per
        # _CAPTURE_CHUNK layer chunk -> TWO writer pendings (e.g. layers 0-13
        # and 14-27) with identical shapes and (post-v3 stable buffers)
        # identical input data_ptrs. Without group fields their keys COLLIDE
        # and the cached graph captured for one group replays for the other -
        # the other group's layers are then never gathered (the ROW1 char-104
        # divergence).
        _writer_group_layer_ids = tuple(
            int(getattr(getattr(_li[0], "state", None), "layer_index", -1))
            for _li in layer_infos
        )
        self._writer_graph_dispatch_launch(
            eager_fn=_eager_writer_launch,
            ext=ext,
            launch_args=writer_launch_args,
            device=device,
            # [PTR-REPUBLISH-REPLAY-SAFE] pure republish signal (forensics /
            # telemetry; no longer blocks replay/capture — the graph bakes the
            # pointer-array BUFFER addresses, contents update in place).
            pointer_rebuild_miss=bool(pointer_rebuild_miss),
            # #9-KEY v5: unresolved layer identity (-1) would collapse both
            # layer groups to one key - the collision regime. Never
            # capture/replay under unresolved identity (eager only).
            key_unresolved=bool(
                bool(_writer_group_layer_ids)
                and min(_writer_group_layer_ids) < 0
            ),
            key_fields=(
                # [WRITER-GRAPH-KEY v6] bounded ownership scope prefix.
                # _capture_writer_graph keeps one current exact graph per
                # (batch, layer-group), so BS8 x three legal groups is bounded
                # by construction instead of a guessed global cache size.
                int(batch),
                int(_writer_group_layer_ids[0] if _writer_group_layer_ids else -1),
                int(_writer_group_layer_ids[-1] if _writer_group_layer_ids else -1),
                int(hash(_writer_group_layer_ids) & 0x7FFFFFFFFFFFFFFF),
                # Exact CUDA launch geometry (raw prompt length is host-only).
                int(writer_safe_tile_tokens),
                int(writer_graph_token_tiles),
                int(selected_cuda.data_ptr()),
                int(seq_lens_cuda.data_ptr()),
                int(row_tensor_cuda.data_ptr()),
                int(slot_tensor_cuda.data_ptr()),
                int(layers),
                int(num_kv_heads),
                int(sink_cap),
                int(recent_cfg),
                int(k_head_cfg),
                int(threshold),
                int(stride_tokens_ref),
                int(block_size_ref),
                int(head_dim_ref),
                int(block_table_cols),
                int(block_table_strides[0]),
                int(block_table_strides[1]),
            ),
        )

    # [WRITER-STABLE-OVERWRITE-WAR-FIX 2026-07-07] R3 record 端:writer 发射
    # (eager/replay/capture 任一臂)后在当前流记录 dispatch-done 事件;下一次
    # rebuild(可能在另一流上下文)覆写 stable 四单例前 wait 此事件,补上
    # "上次 replay→本次 copy" 的反向序(#9-KEY v3 latch 只护正向序)。
    # 4B illegal 主凶终审:sanitizer 100/100 Invalid read 于 replay 内 gather
    # 的 src 读(btable 列 -1 负寻址)=撕裂 stable 输入所致。
    _dispatch_done_evt = getattr(self, "_writer_dispatch_done_evt", None)
    if _dispatch_done_evt is None:
        _dispatch_done_evt = torch.cuda.Event()
        self._writer_dispatch_done_evt = _dispatch_done_evt
    _dispatch_done_evt.record(torch.cuda.current_stream())
    if _pd_mark is not None:
        _pd_mark("wr_kernel_dispatch")

    # 方案 G2 优化：stride_tokens_int 跨层一致，在循环外预计算
    stride_tokens_int = stride_tokens_ref

    def _plan_pad_cleanup_commit(
        state: object,
        slot: int,
        kv_len: int,
        stride_tokens: int,
    ) -> Tuple[Optional[int], Optional[Tuple[int, int, int]]]:
        if slot < 0 or slot >= len(state.compact_k):
            return None, None
        if stride_tokens <= 0:
            return None, None
        if slot >= len(state.compact_pad_zeroed_len):
            return None, None
        if kv_len >= stride_tokens:
            marker = int(stride_tokens)
            if int(state.compact_pad_zeroed_len[slot]) == marker:
                return None, None
            return marker, None
        prev = int(state.compact_pad_zeroed_len[slot])
        if prev == kv_len:
            return None, None
        if prev < 0:
            pad_start = int(kv_len)
            pad_end = int(stride_tokens)
        elif kv_len < prev:
            pad_start = int(kv_len)
            pad_end = int(prev)
        else:
            return int(kv_len), None
        if pad_end <= pad_start:
            return int(kv_len), None
        return int(kv_len), (int(slot), int(pad_start), int(pad_end))

    for layer_idx, info in enumerate(layer_infos):
        payload = info[0]  # info[0] = payload
        state = payload.state

        if not info[1]:  # info[1] = has_rebuild
            continue
        if phase == "refresh" and not defer_compact_meta_publish:
            state.last_reason = "refresh"

        # CPU 更新元数据：只做赋值（避免每层重复做 allowed_len/sink_len 的算术与分支）
        if not rebuild_slots_ref:
            continue

        pad_cleanup_tasks = []  # List of (slot, pad_start, pad_end)
        slot_meta_commits = [] if defer_compact_meta_publish else None
        pad_marker_commits = [] if defer_compact_meta_publish else None
        # [DUAL-GEN-L2a] 开态窄门:切代依赖 writer_done 后的 defer commit
        # (立即模式无原子切换点);且 pad marker 单值只描述当前读代半区,
        # 开态下对写代恒不可信(matched 恒 False→每次重清写半区+重提 marker,
        # flip 后 marker 即新读代真值,自洽)。
        _dg_on = compact_gen_count() > 1
        if _dg_on and not defer_compact_meta_publish:
            raise RuntimeError(
                "compact dual-gen requires deferred compact-meta publish path"
            )
        for idx, slot in enumerate(rebuild_slots_ref):
            sink_len = sink_lens_ref[idx]
            persist_len = persist_lens_ref[idx]
            kv_len = kv_lens_ref[idx]
            same_lengths = (
                int(state.compact_sink_len[slot]) == int(sink_len)
                and int(state.compact_persist_len[slot]) == int(persist_len)
                and int(state.compact_kv_len[slot]) == int(kv_len)
            )
            pad_marker_matched = (not _dg_on) and (
                slot < len(state.compact_pad_zeroed_len)
                and int(state.compact_pad_zeroed_len[slot]) == int(kv_len)
            )
            if defer_compact_meta_publish:
                # Deferred publication crosses a mutable slot-state boundary.
                # Carry a complete transaction instead of a live-state diff;
                # cleanup/rebind between staging and commit must not erase an
                # "unchanged" field that the writer has just made authoritative.
                assert slot_meta_commits is not None
                slot_meta_commits.append(
                    (int(slot), int(sink_len), int(persist_len), int(kv_len))
                )
                if not (same_lengths and pad_marker_matched):
                    marker, task = _plan_pad_cleanup_commit(
                        state, int(slot), int(kv_len), stride_tokens_int
                    )
                    if marker is not None:
                        assert pad_marker_commits is not None
                        pad_marker_commits.append((int(slot), int(marker)))
                    if task is not None:
                        pad_cleanup_tasks.append(task)
                continue
            if same_lengths and pad_marker_matched:
                continue
            if not same_lengths:
                state.compact_sink_len[slot] = sink_len
                state.compact_persist_len[slot] = persist_len
                state.compact_kv_len[slot] = kv_len
            # 收集需要清理的任务
            task = self._check_pad_cleanup_needed(
                state, slot, int(kv_len), stride_tokens_int
            )
            if task is not None:
                pad_cleanup_tasks.append((slot, task[0], task[1]))
        # 批量执行 pad 清理
        if defer_compact_meta_publish:
            assert compact_meta_commit_log is not None
            compact_meta_commit_log.append(
                {
                    "state": state,
                    "phase": str(phase),
                    "reset_slot_commits": reset_slot_commits,
                    "slot_meta_commits": tuple(slot_meta_commits or ()),
                    "pad_marker_commits": tuple(pad_marker_commits or ()),
                    "bootstrap_slots": tuple(
                        int(v) for v in tuple(bootstrap_slots_by_layer[layer_idx] or tuple())
                    ),
                    # [DUAL-GEN-L2a] commit 时对本批 writer 写过的全部 slot
                    # 原子切读代(关态恒空 tuple=零行为)。
                    "dual_gen_flip_slots": (
                        tuple(int(s) for s in rebuild_slots_ref) if _dg_on else tuple()
                    ),
                    "slot_owner_snapshot": slot_owner_snapshot,
                }
            )
            for slot, pad_start, pad_end in pad_cleanup_tasks:
                # [DUAL-GEN-L2a] 开态 pad 清理作用于 writer 半区(读侧视图
                # 仍指旧代);关态 None=历史视图路径逐位。
                _dg_wo = None
                if _dg_on:
                    _dg_wo = compact_write_sub_slot(
                        slot=int(slot),
                        read_gen=int(state.compact_read_gen[int(slot)])
                        if int(slot) < len(state.compact_read_gen)
                        else 0,
                        gen_count=compact_gen_count(),
                        max_live_slots=int(
                            state.compact_page_residency.lease.max_live_sparse_slots
                        ),
                    ) * int(state.compact_stride_tokens)
                self._execute_pad_cleanup(
                    state, slot, pad_start, pad_end, write_offset_tokens=_dg_wo
                )
        else:
            for slot, pad_start, pad_end in pad_cleanup_tasks:
                self._execute_pad_cleanup(state, slot, pad_start, pad_end)

            bootstrap_slots = bootstrap_slots_by_layer[layer_idx]
            if bootstrap_slots:
                for slot in bootstrap_slots:
                    if slot >= len(state.compact_kv_len) or int(state.compact_kv_len[slot]) <= 0:
                        raise RuntimeError(
                            f"bootstrap slot {slot} has empty compact buffer after rebuild"
                        )
            state.bump_compact_meta_epoch()

    if _pd_mark is not None:
        _pd_mark("wr_meta_commit")
    return True

def compute_alpha_selection_pipeline_unified_impl(
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
    """统一 pipeline：单次 Python→C++ 调用完成 preproc_bounds + log_f → soft-nms → cross-head → topk。

    优化：将 preproc_bounds 融合进 C++ 调用，减少 Python→C++ 调用开销。
    仅在 cross_head_power==0 时调用此方法（由调用方保证）。
    返回值与 _compute_alpha_selection_batched_layers 保持一致。

    """
    # NOTE: bounds_kernel 和 pipeline_with_bounds 的 import 延迟到确定需要使用时
    # 这避免了不必要的 import 副作用

    if self.config is None or self.config.alpha_fair is None:
        raise RuntimeError("alpha selector requires config.alpha_fair")

    cfg = self.config.alpha_fair
    layers, batch_size, num_heads, window, kv_len_total = capture_scores.shape
    if num_heads != num_kv_heads * num_queries_per_kv:
        raise ValueError("num_heads mismatch with kv heads")

    device = capture_scores.device

    # 细粒度计时事件初始化
    seq_full_evt0: Optional[torch.cuda.Event] = None
    seq_full_evt1: Optional[torch.cuda.Event] = None
    pure_preproc_evt0: Optional[torch.cuda.Event] = None
    pure_preproc_evt1: Optional[torch.cuda.Event] = None
    bounds_evt0: Optional[torch.cuda.Event] = None
    bounds_evt1: Optional[torch.cuda.Event] = None
    pipeline_evt0: Optional[torch.cuda.Event] = None
    pipeline_evt1: Optional[torch.cuda.Event] = None

    _pd_mark = _producer_detail_marker(self)
    # 记录 seq_full 准备开始时间点
    if profile_detail:
        seq_full_evt0 = torch.cuda.Event(enable_timing=True)
        seq_full_evt0.record()

    # 1. 准备 seq_full 张量（复用原有逻辑，避免重复创建）
    semantic_snapshot = self._get_step_semantic_snapshot()
    sink_cfg = int(semantic_snapshot.sink_tokens)
    recent_cfg = int(semantic_snapshot.recent_tokens)
    seq_full: Optional[torch.Tensor] = None
    if block_size > 0 and recent_cfg > 0:
        if (
            seq_lens_full is not None
            and isinstance(seq_lens_full, torch.Tensor)
            and seq_lens_full.numel() >= batch_size
        ):
            seq_full = seq_lens_full[:batch_size]
            if seq_full.device != device:
                seq_full = seq_full.to(device=device)
            if seq_full.dtype != torch.int32:
                seq_full = seq_full.to(dtype=torch.int32)
            seq_full = seq_full.view(1, batch_size, 1).expand(layers, batch_size, num_kv_heads)
        elif seq_lens_tensor_cpu is not None and seq_lens_tensor_cpu.numel() >= batch_size:
            seq_full = (
                seq_lens_tensor_cpu[:batch_size]
                .to(device=device, dtype=torch.int32)
                .view(1, batch_size, 1)
                .expand(layers, batch_size, num_kv_heads)
            )
        elif seq_lens_cpu is not None and len(seq_lens_cpu) >= batch_size:
            seq_full = torch.tensor(
                [max(0, int(seq_lens_cpu[i])) for i in range(batch_size)],
                device=device,
                dtype=torch.int32,
            ).view(1, batch_size, 1).expand(layers, batch_size, num_kv_heads)

    # 记录 seq_full 准备结束时间点
    if profile_detail:
        seq_full_evt1 = torch.cuda.Event(enable_timing=True)
        seq_full_evt1.record()

    if _pd_mark is not None:
        _pd_mark("sel_seq_full")
    # 2. 准备 pipeline 参数（使用缓存避免每次 float() 转换）
    k_head = compact_recent_effective_k_head(
        k_head=int(cfg.k_head or 0),
        sink_tokens=sink_cfg,
        attn_mode=str(getattr(self.config, "attn_mode", "compact_recent")),
    )
    slice_start = max(0, int(topk_slice_start or 0))
    slice_end = min(kv_len_total, int(topk_slice_end or kv_len_total))
    if slice_end <= 0:
        slice_end = kv_len_total
    # [T3-LOG-R-WS-B 2026-07-08] log_r_cache default ON, borrowing ws_b.
    # OFF meant fused_log_f_prior_kernel recomputed log_r_raw (L2 + positional
    # prior, full-K logf/expf) THREE more times in passes 2/3/4 — the cost the
    # cache was built to remove, left off only because its dedicated (M,K) f32
    # buffer doubled selector scratch. ws_b is run_soft_nms's OUTPUT scratch:
    # dead while run_log_s runs (log_r_cache's entire lifetime) and overwritten
    # as pure output afterwards — disjoint lifetimes, zero extra VRAM, zero
    # alloc. Bitwise-identical numerics: cached reads return the exact f32
    # pass 1 stored; the recompute branch evaluates the same expression on the
    # same inputs. Escape: VLLM_SPARSE_SELECTOR_LOGS_CACHE_R=0 (recompute
    # path). The dedicated-buffer helper (_ensure_selector_log_r_cache_buffer
    # + realloc-override machinery) is retired with this.
    log_r_cache = None
    _use_log_r_cache = (
        os.environ.get("VLLM_SPARSE_SELECTOR_LOGS_CACHE_R", "1").strip() != "0"
        if _DYNAMIC_ENV
        else _SELECTOR_LOGS_CACHE_R_CACHED
    )
    if _use_log_r_cache and _SELECTOR_PIPELINE_WORKSPACE_CACHED:
        _lrc_ws = self._ensure_selector_pipeline_workspaces(
            layers=layers,
            batch=batch_size,
            num_kv_heads=num_kv_heads,
            kv_max=kv_len_total,
            device=device,
        )
        if _lrc_ws is not None:
            log_r_cache = _lrc_ws[1]

    # 缓存 cfg 参数转换结果。cfg(AlphaFairSelectorConfig, slots=True) 构造一次、
    # 全程不原地改字段、不重绑同槽（已 rg 复核：无 .alpha_fair= / 无字段赋值 /
    # 无 setattr/replace；self.config 仅赋值一次），故对象身份唯一决定其不变签名，
    # 且该对象存活期不被释放 -> id 不复用 -> 无 stale-id。用 id(cfg) 短路：
    # id 命中则复用旧 cache 跳过 17 次转换；仅新 cfg 对象才重建值签名与 cache。
    # 保持 cache tuple 布局不变（下游解包不受影响）。
    _cfg_cache = getattr(self, "_pipeline_cfg_cache", None)
    _cfg_cache_id = getattr(self, "_pipeline_cfg_cache_id", None)
    _cfg_id = id(cfg)
    if _cfg_cache is None or _cfg_cache_id != _cfg_id:
        _cfg_sig = self._build_pipeline_cfg_signature(cfg)
        _cfg_cache = (
            _cfg_sig,
            float(cfg.alpha),
            float(cfg.eps),
            float(cfg.gamma),
            float(cfg.prior_weight_l2),
            float(cfg.prior_weight_pos),
            float(cfg.prior_pos_power),
            float(cfg.prior_pos_eta),
            float(cfg.beta()),
            float(cfg.lambda_clip_single),
            float(cfg.lambda_clip_multi),
            float(cfg.lambda_tail_kappa),
            float(cfg.lambda_tail_pivot),
            bool(cfg.lambda_soft),
            int(cfg.nms_window),
            float(cfg.soft_alpha),
            float(cfg.cross_head_alpha),
            float(cfg.cross_head_temperature),
        )
        self._pipeline_cfg_cache = _cfg_cache
        self._pipeline_cfg_cache_id = _cfg_id
    (
        _, _alpha, _eps, _gamma, _prior_l2, _prior_pos, _prior_pow, _prior_eta,
        _beta, _lam_single, _lam_multi, _lam_kappa, _lam_pivot, _lam_soft,
        _nms_win, _soft_alpha, _alpha_cross, _temperature,
    ) = _cfg_cache

    if _pd_mark is not None:
        _pd_mark("sel_params_cfg")
    # 3. 调用 pipeline
    # 记录 pure_preproc 开始时间点
    if profile_detail:
        pure_preproc_evt0 = torch.cuda.Event(enable_timing=True)
        pure_preproc_evt0.record()

    # Current unified selector is bounds-first. Unsupported shapes fail closed.
    _use_decode_bounds_opt = window == 1
    _use_prefill_bounds_opt = (
        window > 1
        and log_f_denoms is None
    )

    pack_order_canonical = False
    if _use_decode_bounds_opt:
        if not _DECODE_BOUNDS_KERNEL_CACHED:
            raise RuntimeError(
                "Decode bounds CUDA path is required for unified selector; "
                "unified selector does not fall back"
            )
        # Decode 优化路径：使用 CUDA kernel 预计算 bounds
        try:
            # 缓存 import 的函数引用，避免每次调用都 import
            _compute_bounds_decode = getattr(self, "_cached_compute_bounds_decode", None)
            _pipeline_with_bounds = getattr(self, "_cached_pipeline_with_bounds", None)
            if _compute_bounds_decode is None or _pipeline_with_bounds is None:
                from utils.bounds_kernel_ext import compute_bounds_decode as _cbd
                from utils.selector_pipeline_ext import pipeline_logits_topk_with_bounds as _pwb
                self._cached_compute_bounds_decode = _cbd
                self._cached_pipeline_with_bounds = _pwb
                _compute_bounds_decode = _cbd
                _pipeline_with_bounds = _pwb
            pack_order_canonical = _pipeline_pack_order_is_verified(
                _pipeline_with_bounds
            )

            # 计算 bounds（使用 CUDA kernel，避免 ATen ops）
            if profile_detail:
                bounds_evt0 = torch.cuda.Event(enable_timing=True)
                bounds_evt0.record()
            decode_bounds_out = self._ensure_selector_decode_bounds_buffers(
                layers=layers,
                batch=batch_size,
                num_kv_heads=num_kv_heads,
                num_queries_per_kv=num_queries_per_kv,
                device=device,
            )
            kv_len_head, head_sink, recent_start, allowed_lengths, row_lo, row_hi = _compute_bounds_decode(
                kv_lengths,
                seq_full,
                num_kv_heads=num_kv_heads,
                num_queries_per_kv=num_queries_per_kv,
                kv_len_total=kv_len_total,
                sink_cfg=sink_cfg,
                recent_cfg=recent_cfg,
                block_size=block_size,
                out=decode_bounds_out,
            )
            if profile_detail:
                bounds_evt1 = torch.cuda.Event(enable_timing=True)
                bounds_evt1.record()

            if _pd_mark is not None:
                _pd_mark("sel_bounds")
            # 调用 with_bounds 版本（跳过 C++ 中的 ATen bounds 计算）
            if profile_detail:
                pipeline_evt0 = torch.cuda.Event(enable_timing=True)
                pipeline_evt0.record()
            if log_f_denoms is None:
                selected_indices_out = (
                    self._ensure_selector_selected_indices_out(
                        layers=layers,
                        batch=batch_size,
                        num_kv_heads=num_kv_heads,
                        k_head=k_head,
                        device=device,
                    )
                    if _SELECTOR_SELECTED_INDICES_OUT_CACHED
                    else None
                )
                # #13 STAGE-0: hoist the pipeline-workspace ensure above the
                # graph branch so its stable data_ptrs can be folded into the
                # captured-graph key. Identical value/ordering vs the inline form
                # on the OFF path (byte-identical bytes into _pipeline_with_bounds).
                pipeline_workspaces = (
                    self._ensure_selector_pipeline_workspaces(
                        layers=layers,
                        batch=batch_size,
                        num_kv_heads=num_kv_heads,
                        kv_max=kv_len_total,
                        device=device,
                    )
                    if _SELECTOR_PIPELINE_WORKSPACE_CACHED
                    else None
                )

                if _pd_mark is not None:
                    _pd_mark("sel_ensure")

                def _eager_topk_pipeline():
                    return _pipeline_with_bounds(
                        capture_scores,
                        row_lo,
                        row_hi,
                        key_norms_full,
                        head_sink,
                        recent_start,
                        num_kv_heads=num_kv_heads,
                        num_queries_per_kv=num_queries_per_kv,
                        k_head=k_head,
                        alpha=_alpha,
                        eps=_eps,
                        gamma=_gamma,
                        prior_weight_l2=_prior_l2,
                        prior_weight_pos=_prior_pos,
                        prior_pos_power=_prior_pow,
                        prior_pos_eta=_prior_eta,
                        beta=_beta,
                        lambda_clip_single=_lam_single,
                        lambda_clip_multi=_lam_multi,
                        lambda_tail_kappa=_lam_kappa,
                        lambda_tail_pivot=_lam_pivot,
                        lambda_soft=_lam_soft,
                        nms_window=_nms_win,
                        soft_alpha=_soft_alpha,
                        alpha_cross=_alpha_cross,
                        temperature=_temperature,
                        cross_eps=_eps,
                        slice_start=slice_start,
                        slice_end=slice_end,
                        log_r_cache=log_r_cache,
                        pipeline_workspaces=pipeline_workspaces,
                        selected_indices_out=selected_indices_out,
                        fixed_shape_topk=_SELECTOR_FIXED_SHAPE_TOPK_CACHED,
                    )

                # [SELECTED-OUT-RING v2·门控] 只有环槽 run(指针槽内稳定)才进
                # dispatch;sync/bootstrap 路径用基础单槽 buffer,14/8 层组交替
                # 即 realloc=key 逐 run 全新——放进 dispatch 只会以 churn 风暴
                # 冲刷 8-graph 图库并打满 thrash 窗(rv7 keys 取证:64 捕中
                # 40 发为 sync 路径 churn)。spill run 同理(瞬时指针)。
                _sel_ring = getattr(self, "_selected_out_ring", None)
                _topk_ring_run = (
                    isinstance(_sel_ring, SelectedOutRing)
                    and _sel_ring.run_open
                    and not _sel_ring.current_run_spilled
                )
                if _pd_mark is not None:
                    _pd_g = self._deadline_deferred_producer_detail_us
                    _pd_g["sel_gate_env_on"] = float(bool(_SELECTOR_TOPK_GRAPH_CACHED))
                    _gk = (
                        "sel_gate_ring_true"
                        if _topk_ring_run
                        else (
                            "sel_gate_ring_none"
                            if _sel_ring is None
                            else (
                                "sel_gate_ring_closed"
                                if not getattr(_sel_ring, "run_open", False)
                                else "sel_gate_ring_spilled"
                            )
                        )
                    )
                    _pd_g[_gk] = float(_pd_g.get(_gk, 0.0) or 0.0) + 1.0
                    _sa = True
                    for _sa_attr in (
                        "_selector_selected_indices_out_override",
                        "_selector_decode_bounds_buffers_override",
                        "_selector_pipeline_workspace_override",
                    ):
                        _sa_ov = getattr(self, _sa_attr, None)
                        if isinstance(_sa_ov, dict) and not isinstance(
                            _sa_ov, SlotStableOverrides
                        ):
                            _sa = False
                            break
                    _sk = "sel_gate_stable_true" if _sa else "sel_gate_stable_false"
                    _pd_g[_sk] = float(_pd_g.get(_sk, 0.0) or 0.0) + 1.0
                if _SELECTOR_TOPK_GRAPH_CACHED and _topk_ring_run:
                    # Key on shapes + EVERY consumed/produced data_ptr so a moved
                    # storage or a regime change (override realloc, kbucket
                    # clamp-fallback, slice pad, or explicit fixed-shape mode
                    # changing the scan domain) misses and falls to eager.
                    def _dp(t):
                        try:
                            return int(t.data_ptr()) if t is not None else 0
                        except Exception:
                            return 0

                    _ws_a_ptr = _dp(pipeline_workspaces[0]) if pipeline_workspaces else 0
                    _ws_b_ptr = _dp(pipeline_workspaces[1]) if pipeline_workspaces else 0
                    _topk_graph_key = (
                        int(layers),
                        int(batch_size),
                        int(num_kv_heads),
                        int(num_queries_per_kv),
                        int(k_head),
                        int(kv_len_total),
                        int(slice_start),
                        int(slice_end),
                        bool(_SELECTOR_FIXED_SHAPE_TOPK_CACHED),
                        _dp(capture_scores),
                        _dp(row_lo),
                        _dp(row_hi),
                        _dp(key_norms_full),
                        _dp(head_sink),
                        _dp(recent_start),
                        _dp(log_r_cache),
                        _ws_a_ptr,
                        _ws_b_ptr,
                        _dp(selected_indices_out),
                    )
                    selected_indices = self._selector_topk_graph_dispatch(
                        eager_fn=_eager_topk_pipeline,
                        key_fields=_topk_graph_key,
                    )
                else:
                    selected_indices = _eager_topk_pipeline()
            else:
                _ppwb = getattr(self, "_cached_pipeline_pre_denom", None)
                if _ppwb is None:
                    from utils.selector_pipeline_ext import pipeline_pre_denom_topk_with_bounds as _ppwb
                    self._cached_pipeline_pre_denom = _ppwb
                pack_order_canonical = _pipeline_pack_order_is_verified(_ppwb)

                # [T3-B] pre_denom arm gets the exact logits-arm treatment
                # (#13 STAGE-0): hoist the two _ensure_* producers above the
                # graph branch so their stable data_ptrs can fold into the
                # captured-graph key. Identical value/ordering vs the inline
                # form on the OFF path (byte-identical bytes into _ppwb).
                selected_indices_out = (
                    self._ensure_selector_selected_indices_out(
                        layers=layers,
                        batch=batch_size,
                        num_kv_heads=num_kv_heads,
                        k_head=k_head,
                        device=device,
                    )
                    if _SELECTOR_SELECTED_INDICES_OUT_CACHED
                    else None
                )
                pipeline_workspaces = (
                    self._ensure_selector_pipeline_workspaces(
                        layers=layers,
                        batch=batch_size,
                        num_kv_heads=num_kv_heads,
                        kv_max=kv_len_total,
                        device=device,
                    )
                    if _SELECTOR_PIPELINE_WORKSPACE_CACHED
                    else None
                )

                if _pd_mark is not None:
                    _pd_mark("sel_ensure")

                def _eager_topk_pipeline_pre_denom():
                    return _ppwb(
                        capture_scores,
                        log_f_denoms,
                        row_lo,
                        row_hi,
                        key_norms_full,
                        head_sink,
                        recent_start,
                        num_kv_heads=num_kv_heads,
                        num_queries_per_kv=num_queries_per_kv,
                        k_head=k_head,
                        alpha=_alpha,
                        eps=_eps,
                        gamma=_gamma,
                        prior_weight_l2=_prior_l2,
                        prior_weight_pos=_prior_pos,
                        prior_pos_power=_prior_pow,
                        prior_pos_eta=_prior_eta,
                        beta=_beta,
                        lambda_clip_single=_lam_single,
                        lambda_clip_multi=_lam_multi,
                        lambda_tail_kappa=_lam_kappa,
                        lambda_tail_pivot=_lam_pivot,
                        lambda_soft=_lam_soft,
                        nms_window=_nms_win,
                        soft_alpha=_soft_alpha,
                        alpha_cross=_alpha_cross,
                        temperature=_temperature,
                        cross_eps=_eps,
                        slice_start=slice_start,
                        slice_end=slice_end,
                        log_r_cache=log_r_cache,
                        pipeline_workspaces=pipeline_workspaces,
                        selected_indices_out=selected_indices_out,
                        fixed_shape_topk=_SELECTOR_FIXED_SHAPE_TOPK_CACHED,
                    )

                # [SELECTED-OUT-RING v2·门控] 同 logits 臂:仅环槽 run 进 dispatch。
                _sel_ring = getattr(self, "_selected_out_ring", None)
                _topk_ring_run = (
                    isinstance(_sel_ring, SelectedOutRing)
                    and _sel_ring.run_open
                    and not _sel_ring.current_run_spilled
                )
                if _SELECTOR_TOPK_GRAPH_CACHED and _topk_ring_run:
                    # Same key discipline as the logits arm: shapes + EVERY
                    # consumed/produced data_ptr. One extra field vs the
                    # 19-field logits key — log_f_denoms' ptr — which also
                    # structurally separates pre_denom graphs from logits
                    # graphs in the shared cache (different tuple arity never
                    # compares equal).
                    def _dp(t):
                        try:
                            return int(t.data_ptr()) if t is not None else 0
                        except Exception:
                            return 0

                    _ws_a_ptr = _dp(pipeline_workspaces[0]) if pipeline_workspaces else 0
                    _ws_b_ptr = _dp(pipeline_workspaces[1]) if pipeline_workspaces else 0
                    _topk_graph_key = (
                        int(layers),
                        int(batch_size),
                        int(num_kv_heads),
                        int(num_queries_per_kv),
                        int(k_head),
                        int(kv_len_total),
                        int(slice_start),
                        int(slice_end),
                        bool(_SELECTOR_FIXED_SHAPE_TOPK_CACHED),
                        _dp(capture_scores),
                        _dp(log_f_denoms),
                        _dp(row_lo),
                        _dp(row_hi),
                        _dp(key_norms_full),
                        _dp(head_sink),
                        _dp(recent_start),
                        _dp(log_r_cache),
                        _ws_a_ptr,
                        _ws_b_ptr,
                        _dp(selected_indices_out),
                    )
                    selected_indices = self._selector_topk_graph_dispatch(
                        eager_fn=_eager_topk_pipeline_pre_denom,
                        key_fields=_topk_graph_key,
                    )
                else:
                    selected_indices = _eager_topk_pipeline_pre_denom()
            if _pd_mark is not None:
                _pd_mark("sel_dispatch")
                _pd_d = self._deadline_deferred_producer_detail_us
                _pd_d["sel_dispatch_calls"] = float(
                    _pd_d.get("sel_dispatch_calls", 0.0) or 0.0
                ) + 1.0
                _pd_d["sel_graph_replay_count_last"] = float(
                    getattr(self, "_selector_topk_graph_replay_count", 0)
                )
            if profile_detail:
                pipeline_evt1 = torch.cuda.Event(enable_timing=True)
                pipeline_evt1.record()
            # selected_indices 直接返回，head_sink, recent_start, kv_len_head, allowed_lengths 已从 compute_bounds_decode 获得

        except Exception as e:
            raise RuntimeError(
                "Decode bounds CUDA path failed; unified selector does not fall back"
            ) from e

    elif _use_prefill_bounds_opt:
        # Prefill 优化路径：W>1 使用 CUDA kernel 预计算 bounds，跳过 ATen-heavy preproc。
        try:
            _compute_bounds_prefill = getattr(self, "_cached_compute_bounds_prefill", None)
            _pipeline_with_bounds = getattr(self, "_cached_pipeline_prefill_with_bounds", None)
            if _compute_bounds_prefill is None or _pipeline_with_bounds is None:
                from utils.bounds_prefill_kernel_ext import compute_bounds_prefill as _cbp
                from utils.selector_pipeline_ext import pipeline_logits_topk_with_bounds as _pwb
                self._cached_compute_bounds_prefill = _cbp
                self._cached_pipeline_prefill_with_bounds = _pwb
                _compute_bounds_prefill = _cbp
                _pipeline_with_bounds = _pwb
            pack_order_canonical = _pipeline_pack_order_is_verified(
                _pipeline_with_bounds
            )

            if profile_detail:
                bounds_evt0 = torch.cuda.Event(enable_timing=True)
                bounds_evt0.record()
            kv_len_head, head_sink, recent_start, allowed_lengths, row_lo, row_hi = _compute_bounds_prefill(
                kv_lengths,
                seq_full,
                num_kv_heads=num_kv_heads,
                num_queries_per_kv=num_queries_per_kv,
                window=window,
                kv_len_total=kv_len_total,
                sink_cfg=sink_cfg,
                recent_cfg=recent_cfg,
                block_size=block_size,
            )
            if profile_detail:
                bounds_evt1 = torch.cuda.Event(enable_timing=True)
                bounds_evt1.record()
                pipeline_evt0 = torch.cuda.Event(enable_timing=True)
                pipeline_evt0.record()
            selected_indices = _pipeline_with_bounds(
                capture_scores,
                row_lo,
                row_hi,
                key_norms_full,
                head_sink,
                recent_start,
                num_kv_heads=num_kv_heads,
                num_queries_per_kv=num_queries_per_kv,
                k_head=k_head,
                alpha=_alpha,
                eps=_eps,
                gamma=_gamma,
                prior_weight_l2=_prior_l2,
                prior_weight_pos=_prior_pos,
                prior_pos_power=_prior_pow,
                prior_pos_eta=_prior_eta,
                beta=_beta,
                lambda_clip_single=_lam_single,
                lambda_clip_multi=_lam_multi,
                lambda_tail_kappa=_lam_kappa,
                lambda_tail_pivot=_lam_pivot,
                lambda_soft=_lam_soft,
                nms_window=_nms_win,
                soft_alpha=_soft_alpha,
                alpha_cross=_alpha_cross,
                temperature=_temperature,
                cross_eps=_eps,
                slice_start=slice_start,
                slice_end=slice_end,
                log_r_cache=log_r_cache,
                pipeline_workspaces=(
                    self._ensure_selector_pipeline_workspaces(
                        layers=layers,
                        batch=batch_size,
                        num_kv_heads=num_kv_heads,
                        kv_max=kv_len_total,
                        device=device,
                    )
                    if _SELECTOR_PIPELINE_WORKSPACE_CACHED
                    else None
                ),
                selected_indices_out=(
                    self._ensure_selector_selected_indices_out(
                        layers=layers,
                        batch=batch_size,
                        num_kv_heads=num_kv_heads,
                        k_head=k_head,
                        device=device,
                    )
                    if _SELECTOR_SELECTED_INDICES_OUT_CACHED
                    else None
                ),
                fixed_shape_topk=_SELECTOR_FIXED_SHAPE_TOPK_CACHED,
            )
            if profile_detail:
                pipeline_evt1 = torch.cuda.Event(enable_timing=True)
                pipeline_evt1.record()
        except Exception as e:
            raise RuntimeError(
                "Prefill bounds CUDA path failed"
            ) from e

    else:
        raise RuntimeError(
            "Unified selector does not support prefill pre-denom path; "
            "unified selector does not fall back"
        )

    # 记录 pure_preproc 结束时间点
    if profile_detail:
        pure_preproc_evt1 = torch.cuda.Event(enable_timing=True)
        pure_preproc_evt1.record()

    # 4. 后处理：dtype 转换
    # pipeline 的 post_topk 已经将不足 k_head 的位置设为 -1
    # log_s 计算时边界外已设为 -inf，topk 选出的 indices 保证在有效范围内
    # 无需额外边界检查（避免 .any() 等触发 GPU→CPU 同步）
    # selected_indices: [L, B, H_kv, k_head]
    if selected_indices.dtype != torch.int32:
        selected_indices = selected_indices.to(dtype=torch.int32)

    # 构建 profile_events 字典
    profile_events: Optional[Dict[str, Tuple[Optional[torch.cuda.Event], Optional[torch.cuda.Event]]]] = None
    if profile_detail:
        profile_events = {
            "seq_full": (seq_full_evt0, seq_full_evt1),
            "pure_preproc": (pure_preproc_evt0, pure_preproc_evt1),
            "bounds": (bounds_evt0, bounds_evt1),
            "pipeline": (pipeline_evt0, pipeline_evt1),
        }

    if _pd_mark is not None:
        _pd_mark("sel_post")
    # Exact-semantic unified extension output is already canonical: ascending
    # valid int32 prefix followed by -1.  The explicit marker lets downstream
    # skip L1 only for this verified producer; legacy/mocked tuples default
    # fail-closed to False in _normalize_selection_layers_result.
    return (
        selected_indices,
        head_sink,
        recent_start,
        kv_len_head,
        allowed_lengths,
        profile_events,
        pack_order_canonical,
    )
