from __future__ import annotations

import json
import os
import time
from typing import Dict, Optional, Sequence, Set, Tuple

import torch

from patches.fa3_native.page_materialize import (
    SelectedStaticCarrier,
    build_final_launch_scratch_runtime,
    static_materialize_selected_pages_runtime,
)
from patches.fa_sparse_runtime.runtime_cache import (
    build_layer_page_sparse_metadata,
    ensure_step_request_sparse_tensors,
)
from patches.runtime_deps import require_runtime_dep
from patches.sparse_utils import (
    _make_step_cache_key,
)

from triton_kernel.flash_attn_score_dump_fwd import pack_req_meta_decode_fast

# P12 lever-2 (2026-06-12): one-shot equivalence-assert gate for the
# layer_state_refresh loop hoists (precomputed cache_key / active_slots).
# ON => every hoisted value is re-derived per layer and compared; any
# divergence raises (verification round: zero triggers, then default OFF).
# Read once at import: the launch scripts export env before process start.
_P12_LSR_VALIDATE_HOIST = (
    os.environ.get("VLLM_SPARSE_VALIDATE_LSR_CACHE_KEY", "0").strip() == "1"
)


def _selected_ready_trace_path() -> str:
    return os.environ.get("VLLM_SPARSE_SELECTED_READY_TRACE_LOG", "").strip()


class StepCacheInvariants:
    """[T2-HOST-DIET 2026-07-10] build_layer_step_cache_impl 的步级不变量包。

    世代 commit 步 36 层 cache 全 miss 时,预备段(标量提取/合同校验/plan
    取用/seqused 合同/veto 行分类/page-sparse 门/trace 开关)只依赖 step 级
    输入,却曾被每层重算 36 遍(cProfile 定谳:controller 三连+函数内
    import+env 读 46 次/步)。由 build_step_cache_invariants 每步构建一次并
    thread 进 impl(P12 precomputed_cache_key/active_slots 同款惯用法);
    无 bundle 的调用方(测试/view-refresh 腿)由 impl 自建,单实现零漂移。

    slot_by_row 取自首个 post-align miss 层:跨层 slot 一致性在
    metadata-builder 循环前已由 _validate_layer_slot_signature_consistency
    验证,align 对共享步输入确定(与 P12 active_slots hoist 同一论证)。
    """

    __slots__ = (
        "force_dense",
        "force_compact_off",
        "batch_size",
        "max_batch_size",
        "block_size",
        "decode_plan_version",
        "plan",
        "plan_valid",
        "seqused_k_gpu",
        "slot_by_row",
        "want_compact_by_row",
        "request_selected_rows",
        "any_selected",
        "page_sparse_gate_on",
        "trace_enabled",
    )

    def __init__(
        self,
        *,
        force_dense: bool,
        force_compact_off: bool,
        batch_size: int,
        max_batch_size: int,
        block_size: int,
        decode_plan_version: int,
        plan: object,
        plan_valid: bool,
        seqused_k_gpu: torch.Tensor,
        slot_by_row: Tuple[int, ...],
        want_compact_by_row: Tuple[bool, ...],
        request_selected_rows: Tuple[bool, ...],
        any_selected: bool,
        page_sparse_gate_on: bool,
        trace_enabled: bool,
    ) -> None:
        self.force_dense = force_dense
        self.force_compact_off = force_compact_off
        self.batch_size = batch_size
        self.max_batch_size = max_batch_size
        self.block_size = block_size
        self.decode_plan_version = decode_plan_version
        self.plan = plan
        self.plan_valid = plan_valid
        self.seqused_k_gpu = seqused_k_gpu
        self.slot_by_row = slot_by_row
        self.want_compact_by_row = want_compact_by_row
        self.request_selected_rows = request_selected_rows
        self.any_selected = any_selected
        self.page_sparse_gate_on = page_sparse_gate_on
        self.trace_enabled = trace_enabled


def build_step_cache_invariants(
    *,
    state: LayerState,
    step_meta: "StepMeta",
    step_authority: "StepAuthority",
    step_bound_meta: Optional["StepBoundMeta"],
    device: torch.device,
    force_dense: bool,
    force_compact_off: bool,
    layer_effective_refresh_by_row: Tuple[bool, ...],
) -> StepCacheInvariants:
    """步级不变量构建(原 impl 预备段原样搬迁,校验 raise 文案不变)。"""
    batch_size = int(step_authority.batch_size)
    max_batch_size = int(step_authority.max_batch_size)
    block_size = int(step_meta.block_size)
    compact_threshold = int(step_authority.compact_bootstrap_threshold)
    req_ids = step_authority.req_ids
    decode_plan_version = int(getattr(step_authority, "decode_plan_version", -1))
    context_kv_len_cpu = step_authority.context_kv_len_by_row
    if int(step_meta.epoch) != int(step_authority.epoch):
        raise RuntimeError(
            "step cache carrier epoch mismatch between StepMeta and StepAuthority; "
            f"meta_epoch={int(step_meta.epoch)} auth_epoch={int(step_authority.epoch)}"
        )
    if int(step_meta.batch_size) != batch_size:
        raise RuntimeError(
            "step cache carrier batch mismatch between StepMeta and StepAuthority; "
            f"meta_batch={int(step_meta.batch_size)} auth_batch={batch_size}"
        )
    if (
        len(req_ids) < batch_size
        or len(context_kv_len_cpu) < batch_size
        or len(step_authority.bootstrap_done_by_row) < batch_size
        or len(step_authority.q_lens_by_row) < batch_size
        or len(step_authority.is_prefill_by_row) < batch_size
        or len(step_authority.short_dense_by_row) < batch_size
    ):
        raise RuntimeError(
            "step cache requires full StepAuthority row coverage "
            f"(batch={batch_size}, req={len(req_ids)}, context={len(context_kv_len_cpu)}, "
            f"bootstrap={len(step_authority.bootstrap_done_by_row)}, "
            f"q_lens={len(step_authority.q_lens_by_row)}, "
            f"is_prefill={len(step_authority.is_prefill_by_row)}, "
            f"short_dense={len(step_authority.short_dense_by_row)})"
        )
    if len(layer_effective_refresh_by_row) < batch_size:
        raise RuntimeError(
            "step cache missing layer_effective_refresh_by_row rows; "
            f"rows={len(layer_effective_refresh_by_row)} batch={batch_size}"
        )
    plan = (
        getattr(step_bound_meta, "compact_recent_launch_plan", None)
        if step_bound_meta is not None
        else None
    )
    plan_valid = plan is not None and bool(getattr(plan, "valid", False))
    seqused_k_gpu = step_meta.canonical_real_kv_len_i32_gpu
    if (
        not isinstance(seqused_k_gpu, torch.Tensor)
        or seqused_k_gpu.device != device
        or seqused_k_gpu.dtype != torch.int32
        or seqused_k_gpu.dim() != 1
        or int(seqused_k_gpu.numel()) != int(batch_size)
    ):
        raise RuntimeError(
            "post_kernel_worker requires canonical_real_kv_len_i32_gpu to satisfy contract; "
            f"batch_size={int(batch_size)}"
        )
    # veto 行分类:除 compact_ready(经 state.compact_kv_len,真 per-layer)外,
    # use_compact 判定链全为步级输入。原 per-layer 循环的 q_len 读取是死读
    # (不参与判定),不再保留。
    slot_by_row = tuple(
        int(state.request_id_to_slot.get(req_id, -1)) for req_id in req_ids
    )
    want_compact_list = []
    for row in range(len(req_ids)):
        bootstrap_done = bool(step_authority.bootstrap_done_by_row[row])
        is_prefill = bool(step_authority.is_prefill_by_row[row])
        is_refresh = bool(layer_effective_refresh_by_row[row])
        context_kv_len = int(context_kv_len_cpu[row])
        short_dense = bool(step_authority.short_dense_by_row[row])
        if (not short_dense) and compact_threshold > 0 and context_kv_len <= compact_threshold:
            # Keep semantic parity with previous threshold-derived behavior.
            short_dense = True
        want_compact_list.append(
            not (
                force_dense
                or force_compact_off
                or is_prefill
                or (not bootstrap_done)
                or short_dense
                or is_refresh
            )
        )
    request_selected_rows = tuple(
        bool(v) for v in getattr(step_authority, "use_compact_by_row", tuple())[:batch_size]
    )
    any_selected = any(request_selected_rows)
    # page-sparse 门(操作数全步级;import 从 per-layer 函数体提为 per-step,
    # 保持函数内 import 以避免模块环)。
    from patches.sparse_constants import should_skip_page_sparse_state
    from patches.patch_installer import _get_global_controller

    _ctrl_peripheral = _get_global_controller()
    _attn_mode_peripheral = str(
        getattr(
            getattr(_ctrl_peripheral, "config", None),
            "attn_mode",
            "compact_recent",
        )
    )
    page_sparse_gate_on = (
        not should_skip_page_sparse_state(_attn_mode_peripheral)
        and not force_dense
        and int(batch_size) > 0
        and bool(getattr(step_meta, "has_decode_row", False))
        and any_selected
    )
    return StepCacheInvariants(
        force_dense=bool(force_dense),
        force_compact_off=bool(force_compact_off),
        batch_size=batch_size,
        max_batch_size=max_batch_size,
        block_size=block_size,
        decode_plan_version=decode_plan_version,
        plan=plan,
        plan_valid=plan_valid,
        seqused_k_gpu=seqused_k_gpu,
        slot_by_row=slot_by_row,
        want_compact_by_row=tuple(want_compact_list),
        request_selected_rows=request_selected_rows,
        any_selected=any_selected,
        page_sparse_gate_on=page_sparse_gate_on,
        trace_enabled=bool(_selected_ready_trace_path()),
    )


def _append_selected_ready_trace(event: dict[str, object]) -> None:
    path = _selected_ready_trace_path()
    if not path:
        return
    record = dict(event)
    record.setdefault("pid", os.getpid())
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def refresh_static_materialize_cuda(
    *,
    request_refresh_generation_i32: torch.Tensor,
    **_: object,
) -> dict[str, torch.Tensor]:
    request_refresh_generation_i32 = request_refresh_generation_i32.to(dtype=torch.int32)
    batch_size = int(request_refresh_generation_i32.numel())
    device = request_refresh_generation_i32.device
    return {
        "materialize_status_i32": torch.zeros(
            (batch_size,),
            dtype=torch.int32,
            device=device,
        ),
        "applied_refresh_generation_i32": request_refresh_generation_i32.reshape(batch_size).clone(),
    }




def publish_selected_scope_launch_ready_if_needed(
    *,
    state: LayerState,
    step_authority: "StepAuthority",
) -> bool:
    if not bool(getattr(state, "step_cache_cached_launch_ready", False)):
        _append_selected_ready_trace(
            {
                "event": "selected_scope_publish_skipped",
                "reason": "cached_launch_ready_false",
                "epoch": int(getattr(step_authority, "epoch", -1)),
                "layer_index": int(getattr(state, "layer_index", -1)),
                "handle_id": id(getattr(step_authority, "selected_scope_wait_handle", None)),
            }
        )
        return False
    return _mark_selected_scope_terminal_if_needed(
        state=state,
        step_authority=step_authority,
        status="accept",
        reason="launch_ready",
    )


def refresh_selected_launch_view_for_current_step(
    *,
    state: LayerState,
    step_meta: "StepMeta",
    step_authority: "StepAuthority",
    block_table: Optional[torch.Tensor],
    device: torch.device,
    layer_effective_refresh_by_row: Tuple[bool, ...],
    step_bound_meta: Optional["StepBoundMeta"] = None,
    precomputed_active_slots: Optional[Tuple[int, ...]] = None,
    precomputed_cache_key: Optional[Tuple[object, ...]] = None,
) -> None:
    batch_size = int(step_authority.batch_size)
    # P12 lever-2 (2026-06-12): once the metadata-builder loop has aligned
    # every layer's slot map to the step-global map, active_slots is identical
    # across layers (alignment is deterministic from the shared (req_ids,
    # epoch, global_slot_map) inputs, and cross-layer consistency is validated
    # BEFORE that loop). The caller computes it once from the first post-align
    # layer and threads it down; None (any other caller / test) keeps the
    # legacy per-layer computation below.
    if precomputed_active_slots is not None:
        active_slots = precomputed_active_slots
        if _P12_LSR_VALIDATE_HOIST:
            _recomputed_slots = tuple(
                int(state.request_id_to_slot.get(req_id, -1))
                for req_id in step_meta.req_ids[:batch_size]
            )
            if _recomputed_slots != active_slots:
                raise RuntimeError(
                    "P12 lever-2 hoisted active_slots diverged from per-layer "
                    "truth; "
                    f"layer_index={int(getattr(state, 'layer_index', -1))} "
                    f"hoisted={active_slots} recomputed={_recomputed_slots}"
                )
    else:
        active_slots = tuple(
            int(state.request_id_to_slot.get(req_id, -1))
            for req_id in step_meta.req_ids[:batch_size]
        )
    current_refresh_signature = tuple(
        int(state.sparse_request_refresh_generation[slot]) if int(slot) >= 0 else 0
        for slot in active_slots
    )
    cached_static_slots = getattr(state, "_step_cache_selected_static_slots", None)
    if cached_static_slots is None and state.step_cache_selected_static_pages_i32 is not None:
        setattr(state, "_step_cache_selected_static_slots", active_slots)
    elif cached_static_slots != active_slots:
        state.step_cache_selected_static_pages_i32 = None
        state.step_cache_selected_static_seqused_k_by_head_i32 = None
        state.step_cache_selected_static_schema_version = 0
    has_effective_refresh = any(
        bool(v) for v in layer_effective_refresh_by_row[:batch_size]
    )
    preserve_static_carrier = bool(
        has_effective_refresh
        and
        state.step_cache_selected_static_pages_i32 is not None
        and state.step_cache_selected_static_seqused_k_by_head_i32 is not None
        and int(state.step_cache_selected_static_schema_version) == 2
        and getattr(state, "_step_cache_selected_static_slots", None) == active_slots
        and state.step_cache_requested_refresh_generation_signature
        == current_refresh_signature
    )
    # Force final launch scratch to be recomposed from the current step truth
    # while preserving reusable compact/static buffers whenever possible.
    state.step_cache_applied_recent_epoch_value = -1
    setattr(state, "_preserve_selected_static_carrier", preserve_static_carrier)
    try:
        build_layer_step_cache_impl(
            state=state,
            step_meta=step_meta,
            step_authority=step_authority,
            block_table=block_table,
            device=device,
            force_dense=False,
            force_compact_off=False,
            skip_meta_pack=True,
            layer_effective_refresh_by_row=layer_effective_refresh_by_row,
            # M5 Part B2 (2026-04-24): forward step_bound_meta so the
            # inner bail-out / compact_cache_valid gate can consult
            # plan.valid. None is safe — the inner callsite guards it.
            step_bound_meta=step_bound_meta,
            # P12 lever-2: thread the caller-proven step cache key down so the
            # impl skips the duplicate _make_step_cache_key 9-tuple rebuild
            # (the metadata-builder else-leg only fires after proving
            # state.step_cache_key == cache_key). None keeps the local
            # computation.
            precomputed_cache_key=precomputed_cache_key,
        )
    finally:
        setattr(state, "_preserve_selected_static_carrier", False)


def _mark_selected_scope_terminal_if_needed(
    *,
    state: LayerState,
    step_authority: "StepAuthority",
    status: str,
    reason: str,
) -> bool:
    handle = getattr(step_authority, "selected_scope_wait_handle", None)
    if handle is None or bool(getattr(handle, "is_ready", False)):
        # [T2-HOST-DIET] trace 关闭时不再白构造事件 dict(id/getattr/int 全做
        # 再被 append 丢弃;cProfile 定谳 360 次/取证轮)。
        if _selected_ready_trace_path():
            _append_selected_ready_trace(
                {
                    "event": "selected_scope_publish_skipped",
                    "reason": "handle_missing_or_ready",
                    "epoch": int(getattr(step_authority, "epoch", -1)),
                    "layer_index": int(getattr(state, "layer_index", -1)),
                    "handle_id": id(handle),
                    "handle_ready": bool(getattr(handle, "is_ready", False)) if handle is not None else None,
                }
            )
        return False

    from patches.fa3_native.scope_async import mark_layer_commit_terminal

    expected_layers = tuple(getattr(handle, "expected_layers", ()))
    if not expected_layers:
        raise RuntimeError(
            "selected scope wait handle requires expected_layers before launch-ready publish"
        )
    layer_index = int(getattr(state, "layer_index", -1))
    if layer_index < 0:
        raise RuntimeError(
            "selected scope wait handle requires non-negative layer_index before launch-ready publish"
        )
    mark_layer_commit_terminal(handle, layer_id=layer_index, status=str(status))
    _append_selected_ready_trace(
        {
            "event": "selected_scope_publish_commit",
            "epoch": int(getattr(step_authority, "epoch", -1)),
            "layer_index": layer_index,
            "status": str(status),
            "reason": str(reason),
            "handle_id": id(handle),
            "consumer_step_id": int(getattr(getattr(handle, "target_selected_scope_key", None), "consumer_step_id", -1)),
            "expected_layers": list(int(v) for v in expected_layers),
            "committed_layers": list(int(v) for v in tuple(getattr(handle, "committed_layers", tuple()))),
            "is_ready": bool(getattr(handle, "is_ready", False)),
        }
    )
    return True


def build_layer_step_cache_impl(
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
    step_invariants: Optional[StepCacheInvariants] = None,
) -> None:
    """为该层构建 step 缓存（每层每 step 只执行一次）。

    从 StepAuthority（语义单源）和 StepMeta（张量载体）中提取数据，
    构建该层所需的 GPU tensors，供 dispatcher 直接使用。

    [T2-HOST-DIET 2026-07-10] slot map 在 metadata-builder 循环内已对齐到
    步级全局 map 且跨层一致性预验证,故 slot_by_row 与 use_compact 的 veto
    链均为步级不变量,走 step_invariants(调用方 thread 或本函数自建)。
    真 per-layer 的只剩 compact_ready(state.compact_kv_len)与 state 字段写。

    ⚠️ 性能优化：compact 相关的 2 个 GPU tensor (compact_kv_len, compact_offset)
    只在 compact_meta_epoch 或 req_ids 变化时重建。在 decode 稳态，这些值是稳定的，
    可以跨多步复用，大幅减少 torch.tensor() 调用（从每步每层 2 次→仅在变化时）。
    """
    # ------------------------------------------------------------------
    # Top-level bail-out (2026-04-24 spec v1.2 §4.4).
    # Judged by CONTENT signature (_make_step_cache_key), not the per-step
    # counter decode_plan_version which advances every step and would never
    # let the bail-out hit. The cache_key construction is ~5us Python tuple
    # boxing (batch_size is 2-4) — cheap relative to the 127us slow path we
    # skip on hit. Slow path also consumes the same cache_key (persisted
    # onto state at the end), so zero duplicate work.
    # ------------------------------------------------------------------
    # P12 lever-2 (2026-06-12): the metadata-builder loop already assembles
    # the IDENTICAL 9-tuple once per step (step-constant prefix + per-layer
    # compact_meta_epoch, force flags constant-False on that path) and threads
    # it down; recomputing it here per layer is pure duplicate work. None (any
    # other caller) keeps the local computation. With
    # VLLM_SPARSE_VALIDATE_LSR_CACHE_KEY=1 the key is re-derived and compared
    # for the one-shot verification round; divergence raises.
    if precomputed_cache_key is not None and not _P12_LSR_VALIDATE_HOIST:
        cache_key = precomputed_cache_key
    else:
        cache_key = _make_step_cache_key(
            step_authority,
            state,
            force_dense=force_dense,
            force_compact_off=force_compact_off,
            layer_effective_refresh_by_row=layer_effective_refresh_by_row[:int(step_authority.batch_size)],
        )
        if precomputed_cache_key is not None and precomputed_cache_key != cache_key:
            raise RuntimeError(
                "P12 lever-2 precomputed step cache key diverged from "
                "_make_step_cache_key truth; "
                f"layer_index={int(getattr(state, 'layer_index', -1))} "
                f"precomputed={precomputed_cache_key!r} recomputed={cache_key!r}"
            )
    # M5 Part B2 (2026-04-24): the fast-path existence/shape gate now uses the
    # step-level CompactRecentLaunchPlan (single source of truth). Downstream
    # readers all consume plan.* — checking `plan is not None and plan.valid`
    # is the semantically equivalent way to confirm the compact descriptors
    # are ready, without having to materialize (and later read) per-layer
    # state.step_cache_compact_*_gpu tensors.
    _bailout_plan = (
        getattr(step_bound_meta, "compact_recent_launch_plan", None)
        if step_bound_meta is not None
        else None
    )
    if (
        state.step_cache_key == cache_key
        and _bailout_plan is not None
        and bool(getattr(_bailout_plan, "valid", False))
        and not force_dense
        and not force_compact_off
        and not bool(getattr(state, "_preserve_selected_static_carrier", False))
        and (skip_meta_pack or bool(getattr(state, "step_cache_meta_packed", False)))
    ):
        return  # bail-out: no validation, no pack_req_meta, no page_sparse branch, no sync

    # [T2-HOST-DIET 2026-07-10] 步级不变量:调用方 thread 则直接用(commit 步
    # 36 层共享一份),否则自建(测试/view-refresh 腿,单实现零漂移)。
    # force 标志护栏 fail-close:bundle 与本调用不一致=错传,响亮 raise。
    if step_invariants is None:
        step_invariants = build_step_cache_invariants(
            state=state,
            step_meta=step_meta,
            step_authority=step_authority,
            step_bound_meta=step_bound_meta,
            device=device,
            force_dense=force_dense,
            force_compact_off=force_compact_off,
            layer_effective_refresh_by_row=layer_effective_refresh_by_row,
        )
    elif (
        bool(step_invariants.force_dense) != bool(force_dense)
        or bool(step_invariants.force_compact_off) != bool(force_compact_off)
    ):
        raise RuntimeError(
            "step_invariants force flags diverge from call flags; "
            f"inv=({step_invariants.force_dense},{step_invariants.force_compact_off}) "
            f"call=({force_dense},{force_compact_off})"
        )
    inv = step_invariants
    batch_size = inv.batch_size
    max_batch_size = inv.max_batch_size
    block_size = inv.block_size
    decode_plan_version = inv.decode_plan_version

    # cache_key already computed at the top-level bail-out (content signature).
    # Slow path reuses it for persist at the end; no duplicate compute.

    # M5 Part B2 (2026-04-24): `compact_cache_valid` now gates on plan
    # availability — the step-level CompactRecentLaunchPlan is the single
    # source of truth for compact descriptors. We still keep cache_key +
    # plan_version so the per-layer bail-out remains content-sensitive, but
    # the tensor-existence check is delegated to `plan.valid`.
    _slow_plan = inv.plan
    compact_cache_valid = (
        state.step_cache_key == cache_key
        and int(getattr(state, "step_cache_plan_version", -1)) == decode_plan_version
        and inv.plan_valid
    )

    if compact_cache_valid:
        # ✅ 复用已缓存的 has_compact/all_compact（use_compact_list=None 跳过重算）。
        # compact 布局张量不再 per-layer 持有，已迁移到 plan.compact_valid_tokens_i32
        # / plan.compact_offset_tokens_i64（step 级单源，见 M5 Part B2/C）。
        use_compact_list = None
    else:
        # ❌ cache miss: 仍按 per-row 规则计算 use_compact 以维护 step_cache_has_compact
        # / step_cache_all_compact 预计算值。veto 链为步级不变量(inv.want_compact_by_row),
        # 真 per-layer 输入只剩 compact_ready(state.compact_kv_len)。
        _want_compact = inv.want_compact_by_row
        _slot_by_row = inv.slot_by_row
        _compact_kv_len = state.compact_kv_len
        _ckl_len = len(_compact_kv_len)
        use_compact_list = []
        for row in range(len(_want_compact)):
            slot = _slot_by_row[row]
            use_compact_list.append(
                1
                if (
                    _want_compact[row]
                    and 0 <= slot < _ckl_len
                    and _compact_kv_len[slot] > 0
                )
                else 0
            )
        state._cached_is_compact_gpu = None

    # ⚠️ 性能优化：使用 max_batch_size 预分配 buffer，避免热路径 slice
    # 这样 dispatcher 可以直接使用整个 buffer + num_seqs 参数
    if state.step_cache_req_meta_i32 is None or state.step_cache_req_meta_i32.shape[0] < max_batch_size:
        state.step_cache_req_meta_i32 = torch.empty((max_batch_size, 7), dtype=torch.int32, device=device)
    if state.step_cache_req_meta_i64 is None or state.step_cache_req_meta_i64.shape[0] < max_batch_size:
        state.step_cache_req_meta_i64 = torch.empty((max_batch_size, 4), dtype=torch.int64, device=device)

    seqused_k_gpu = inv.seqused_k_gpu

    # ⚠️ 调用 GPU kernel 打包 meta（传入整个 buffer + num_seqs，避免 slice）
    if not skip_meta_pack:
        # M5 Part C (2026-04-24): feed pack_req_meta_decode_fast exclusively
        # from the step-level CompactRecentLaunchPlan. Plan tensors are
        # produced once per step (single H2D) and shared across all layers;
        # the 4 per-layer `state.step_cache_compact_*` carriers have been
        # deleted. Missing plan is a caller contract violation; tests that need
        # all-dense behavior should pass an explicit zero plan.
        if _slow_plan is None:
            raise RuntimeError(
                "build_layer_step_cache_impl requires compact_recent_launch_plan "
                "when skip_meta_pack is false"
            )
        _pack_compact_kv_len_i32 = _slow_plan.compact_valid_tokens_i32
        _pack_compact_offset_tokens_i64 = _slow_plan.compact_offset_tokens_i64
        # 单一语义真源：is_compact 由 compact_kv_len 布局派生，不再读取 legacy compact mask 缓存。
        if compact_cache_valid and hasattr(state, '_cached_is_compact_gpu') and state._cached_is_compact_gpu is not None:
            is_compact_gpu = state._cached_is_compact_gpu
        else:
            is_compact_gpu = _pack_compact_kv_len_i32[:batch_size].gt(0).to(torch.int32)
            state._cached_is_compact_gpu = is_compact_gpu
        pack_req_meta_decode_fast(
            seqused_k=seqused_k_gpu,
            is_compact_i32=is_compact_gpu,
            compact_kv_len_i32=_pack_compact_kv_len_i32,
            compact_offset_tokens_i64=_pack_compact_offset_tokens_i64,
            req_meta_i32=state.step_cache_req_meta_i32,
            req_meta_i64=state.step_cache_req_meta_i64,
            block_size=block_size,
            recent_cap=int(step_authority.recent_cap),
            sink_tokens=int(step_authority.sink_tokens),
            num_seqs=batch_size,  # ⚠️ 关键：传入实际行数，kernel 只处理前 num_seqs 行
        )
        state.step_cache_meta_packed = True
    else:
        state.step_cache_meta_packed = False

    # 缓存到 LayerState
    # M5 Part C (2026-04-24): step_cache_compact_kv_len_gpu /
    # step_cache_compact_offset_gpu 已随 4 个字段一并删除；compact 布局真源
    # 现在是 step_bound_meta.compact_recent_launch_plan 的 GPU 载体。
    state.step_cache_epoch = int(step_authority.epoch)
    state.step_cache_key = cache_key
    state.step_cache_plan_version = decode_plan_version
    # ⚠️ P1-2 优化：预计算 has_compact 布尔值，避免 dispatcher 热路径的 .any() GPU→CPU 同步
    # 注意：当 compact_cache_valid 时 use_compact_list 为 None，此时 has_compact 不变
    if use_compact_list is not None:
        if use_compact_list:
            state.step_cache_has_compact = any(use_compact_list)
            state.step_cache_all_compact = all(use_compact_list)
        else:
            state.step_cache_has_compact = False
            state.step_cache_all_compact = False

    page_sparse_metadata: Optional[dict[str, object]] = None
    page_sparse_enabled = False
    request_selected_rows = inv.request_selected_rows
    # ---------------------------------------------------------------------
    # SCAFFOLDING skip block B (peripheral companion §5.2.B)
    # Gate on 时跳过 page-sparse metadata 构建;既有 line 742+ disabled-branch
    # 自动处理 state 清理 + early return.
    # 默认 off: 行为 bit-identical. 切 on 见 spec §5.4.
    # Phase 6 无 gate 删除时一并移除.
    # [T2-HOST-DIET] 门操作数全步级 → inv.page_sparse_gate_on 一次求值。
    # ---------------------------------------------------------------------
    if inv.page_sparse_gate_on:
        page_sparse_metadata = build_layer_page_sparse_metadata(
            step_meta=step_meta,
            request_id_to_slot=state.request_id_to_slot,
            selected_middle_pages_by_slot=state.sparse_selected_middle_pages,
            selected_middle_counts_by_slot=state.sparse_selected_middle_counts,
            refresh_generation_by_slot=state.sparse_request_refresh_generation,
            refresh_by_row=layer_effective_refresh_by_row,
            selected_middle_uniform_counts_cpu=state.sparse_selected_middle_uniform_count_cpu,
            selected_middle_min_logical_page_cpu=state.sparse_selected_middle_min_logical_page_cpu,
            selected_middle_max_logical_page_cpu=state.sparse_selected_middle_max_logical_page_cpu,
            request_selected_rows=request_selected_rows,
            num_kv_heads=int(state.num_kv_heads),
            page_size=int(block_size),
            device=device,
        )
        page_sparse_enabled = bool(page_sparse_metadata["use_sparse"])
        if inv.trace_enabled:
            _append_selected_ready_trace(
                {
                    "event": "page_sparse_metadata_built",
                    "epoch": int(getattr(step_authority, "epoch", -1)),
                    "layer_index": int(getattr(state, "layer_index", -1)),
                    "req_ids": list(str(v) for v in step_meta.req_ids[:batch_size]),
                    "request_selected_rows": [bool(v) for v in request_selected_rows],
                    "request_sparse_eligible": list(
                        bool(v) for v in tuple(page_sparse_metadata.get("request_sparse_eligible", tuple()))
                    ),
                    "selected_page_count": [
                        int(v)
                        for v in page_sparse_metadata["selected_page_count_summary_i32"]
                        .detach()
                        .to("cpu")
                        .tolist()
                    ],
                    "visible_kv_len": [
                        int(v)
                        for v in page_sparse_metadata["selected_summary_k_i32"]
                        .detach()
                        .to("cpu")
                        .tolist()
                    ],
                    "refresh_generation_signature": list(
                        int(v)
                        for v in tuple(page_sparse_metadata.get("request_refresh_generation_signature", tuple()))
                    ),
                    "request_sparse_debug_rows": [
                        dict(row)
                        for row in tuple(page_sparse_metadata.get("request_sparse_debug_rows", tuple()))
                    ],
                    "use_sparse": bool(page_sparse_enabled),
                    "handle_id": id(getattr(step_authority, "selected_scope_wait_handle", None)),
                }
            )

    if not page_sparse_enabled or page_sparse_metadata is None or block_table is None:
        state.step_cache_selected_static_pages_i32 = None
        state.step_cache_selected_static_seqused_k_by_head_i32 = None
        state.step_cache_selected_static_schema_version = 0
        setattr(state, "_step_cache_selected_static_slots", None)
        state.step_cache_page_table_i32 = None
        state.step_cache_selected_seqused_k_by_head_i32 = None
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
        if inv.trace_enabled:
            _append_selected_ready_trace(
                {
                    "event": "page_sparse_disabled",
                    "epoch": int(getattr(step_authority, "epoch", -1)),
                    "layer_index": int(getattr(state, "layer_index", -1)),
                    "page_sparse_metadata_present": bool(page_sparse_metadata is not None),
                    "page_sparse_enabled": bool(page_sparse_enabled),
                    "block_table_present": bool(block_table is not None),
                    "request_selected_rows": [bool(v) for v in request_selected_rows],
                    "handle_id": id(getattr(step_authority, "selected_scope_wait_handle", None)),
                }
            )
        if not inv.any_selected:
            _mark_selected_scope_terminal_if_needed(
                state=state,
                step_authority=step_authority,
                status="no_op",
                reason="no_selected_rows",
            )
        return

    if block_table.dim() != 2 or block_table.shape[0] < batch_size:
        raise ValueError("step-cache page-sparse materialize requires block_table to cover batch rows")

    block_table_i32 = block_table.to(device=device, dtype=torch.int32)
    real_kv_len_i32 = page_sparse_metadata["real_kv_len_i32"]
    selected_seqused_k_by_head_i32 = page_sparse_metadata["selected_seqused_k_by_head_i32"].to(
        device=device,
        dtype=torch.int32,
    )
    kv_batch_idx_i32 = page_sparse_metadata["kv_batch_idx_i32"]
    request_refresh_generation_i32 = page_sparse_metadata["applied_refresh_generation_i32"]
    request_sparse_tensors = ensure_step_request_sparse_tensors(
        step_meta=step_meta,
        page_size=int(block_size),
        device=device,
        layer_effective_refresh_by_row=layer_effective_refresh_by_row,
    )
    request_recent_epoch_i32 = request_sparse_tensors["request_recent_epoch_i32"]
    request_recent_first_i32 = request_sparse_tensors["request_recent_first_logical_page_i32"]
    request_recent_count_i32 = request_sparse_tensors["request_recent_page_count_i32"]

    max_page_count = int(page_sparse_metadata["max_page_count"])
    expected_rows = int(batch_size) * int(state.num_kv_heads)
    if (
        state.step_cache_page_table_i32 is None
        or state.step_cache_page_table_i32.device != device
        or state.step_cache_page_table_i32.shape[0] != expected_rows
        or state.step_cache_page_table_i32.shape[1] != int(max_page_count)
    ):
        state.step_cache_page_table_i32 = torch.empty(
            (expected_rows, int(max_page_count)),
            dtype=torch.int32,
            device=device,
        )
        state.step_cache_selected_seqused_k_by_head_i32 = None
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

    state.step_cache_selected_seqused_k_by_head_i32 = selected_seqused_k_by_head_i32.reshape(
        expected_rows
    ).clone()
    state.step_cache_real_kv_len_i32 = real_kv_len_i32
    state.step_cache_kv_batch_idx_i32 = kv_batch_idx_i32
    state.step_cache_page_table_layout = int(page_sparse_metadata["page_table_layout"])
    state.step_cache_has_page_sparse = True
    state.step_cache_all_page_sparse = True
    state.step_cache_cached_lengths_ok = bool(page_sparse_metadata["cached_lengths_ok"])
    state.step_cache_cached_kv_batch_idx_identity = bool(
        page_sparse_metadata["cached_kv_batch_idx_identity"]
    )
    state.step_cache_cached_status_ok = False
    state.step_cache_cached_freshness_ok = False
    state.step_cache_cached_launch_ready = False

    request_refresh_generation_signature = tuple(
        int(v) for v in page_sparse_metadata["request_refresh_generation_signature"]
    )
    refresh_matches = (
        state.step_cache_requested_refresh_generation_signature
        == request_refresh_generation_signature
    )
    request_recent_epoch_value = int(page_sparse_metadata["request_recent_epoch_value"])
    recent_matches = (
        int(state.step_cache_applied_recent_epoch_value)
        == int(request_recent_epoch_value)
    )

    active_slots = tuple(
        int(state.request_id_to_slot.get(req_id, -1))
        for req_id in step_meta.req_ids[:batch_size]
    )
    carrier_missing = (
        state.step_cache_selected_static_pages_i32 is None
        or state.step_cache_selected_static_seqused_k_by_head_i32 is None
        or int(state.step_cache_selected_static_schema_version)
        != 2
    )
    preserve_static_carrier = bool(getattr(state, "_preserve_selected_static_carrier", False))

    if carrier_missing or (not refresh_matches and not preserve_static_carrier):
        if (
            state.sparse_selected_middle_pages_dense_i32 is None
            or state.sparse_selected_middle_counts_dense_i32 is None
        ):
            state._sync_sparse_selected_middle_dense_for_slots(active_slots)
        active_slots_i32 = torch.tensor(
            active_slots,
            device=device,
            dtype=torch.int32,
        )
        selected_static_carrier = static_materialize_selected_pages_runtime(
            selected_static_pages_by_slot_i32=state.sparse_selected_middle_pages_dense_i32,
            selected_static_page_count_by_slot_i32=state.sparse_selected_middle_counts_dense_i32,
            request_slot_rows_i32=active_slots_i32,
            page_size=int(block_size),
        )
        state.step_cache_selected_static_pages_i32 = (
            selected_static_carrier.selected_static_pages_i32
        )
        state.step_cache_selected_static_seqused_k_by_head_i32 = (
            selected_static_carrier.selected_static_seqused_k_by_head_i32
        )
        state.step_cache_selected_static_schema_version = int(
            selected_static_carrier.schema_version
        )
        setattr(state, "_step_cache_selected_static_slots", active_slots)
    if not refresh_matches and not preserve_static_carrier:
        materialize_out = refresh_static_materialize_cuda(
            final_page_table_i32=state.step_cache_page_table_i32,
            selected_middle_pages_by_slot_i32=state.sparse_selected_middle_pages_dense_i32,
            selected_middle_counts_by_slot_i32=state.sparse_selected_middle_counts_dense_i32,
            request_slot_rows_i32=torch.tensor(
                active_slots,
                device=device,
                dtype=torch.int32,
            ),
            request_block_table_i32=block_table_i32[:batch_size],
            sink_page_slots=int(step_meta.sink_tokens + int(block_size) - 1) // int(block_size),
            request_refresh_generation_i32=request_refresh_generation_i32.to(
                device=device,
                dtype=torch.int32,
            ),
        )
        state.step_cache_materialize_status_i32 = materialize_out["materialize_status_i32"]
        state.step_cache_applied_refresh_generation_i32 = materialize_out[
            "applied_refresh_generation_i32"
        ]
        state.step_cache_requested_refresh_generation_signature = (
            request_refresh_generation_signature
        )
    final_truth_missing = getattr(state, "step_cache_selected_seqused_k_by_head_i32", None) is None
    if not refresh_matches or not recent_matches or final_truth_missing:
        selected_static_carrier = SelectedStaticCarrier(
            schema_version=int(state.step_cache_selected_static_schema_version),
            selected_static_pages_i32=state.step_cache_selected_static_pages_i32,
            selected_static_seqused_k_by_head_i32=(
                state.step_cache_selected_static_seqused_k_by_head_i32
            ),
        )
        final_launch_scratch = build_final_launch_scratch_runtime(
            selected_static_carrier,
            page_size=int(block_size),
            request_block_table_i32=block_table_i32[:batch_size],
            sink_page_slots=int(step_meta.sink_tokens + int(block_size) - 1) // int(block_size),
            request_recent_first_logical_page_i32=request_recent_first_i32,
            request_recent_page_count_i32=request_recent_count_i32,
            request_recent_epoch_i32=request_recent_epoch_i32,
            request_refresh_generation_i32=request_refresh_generation_i32.to(
                device=device,
                dtype=torch.int32,
            ),
            request_real_kv_len_i32=real_kv_len_i32.to(
                device=device,
                dtype=torch.int32,
            ),
            request_selected_rows=torch.tensor(
                request_selected_rows,
                device=device,
                dtype=torch.bool,
            ),
            num_kv_heads=int(state.num_kv_heads),
            out_selected_page_table_i32=state.step_cache_page_table_i32,
        )
        state.step_cache_page_table_i32 = final_launch_scratch.selected_page_table_i32
        state.step_cache_selected_seqused_k_by_head_i32 = (
            final_launch_scratch.selected_seqused_k_by_head_i32
        )
        state.step_cache_materialize_status_i32 = final_launch_scratch.materialize_status_i32
        state.step_cache_patch_status_i32 = final_launch_scratch.patch_status_i32
        state.step_cache_applied_refresh_generation_i32 = (
            final_launch_scratch.applied_refresh_generation_i32
        )
        state.step_cache_applied_recent_epoch_i32 = (
            final_launch_scratch.applied_recent_epoch_i32
        )
        state.step_cache_requested_refresh_generation_signature = (
            request_refresh_generation_signature
        )
        state.step_cache_applied_recent_epoch_value = int(request_recent_epoch_value)

    if state.step_cache_materialize_status_i32 is None:
        state.step_cache_materialize_status_i32 = torch.zeros(
            (batch_size,),
            dtype=torch.int32,
            device=device,
        )
    if state.step_cache_patch_status_i32 is None:
        state.step_cache_patch_status_i32 = torch.zeros(
            (batch_size,),
            dtype=torch.int32,
            device=device,
        )
    state.step_cache_cached_status_ok = True
    state.step_cache_cached_freshness_ok = True
    state.step_cache_cached_launch_ready = bool(
        state.step_cache_cached_lengths_ok
        and state.step_cache_cached_kv_batch_idx_identity
        and state.step_cache_cached_status_ok
        and state.step_cache_cached_freshness_ok
        and state.step_cache_page_table_i32 is not None
        and state.step_cache_selected_seqused_k_by_head_i32 is not None
        and state.step_cache_real_kv_len_i32 is not None
        and state.step_cache_kv_batch_idx_i32 is not None
    )
    if _selected_ready_trace_path():
        _append_selected_ready_trace(
            {
                "event": "page_sparse_launch_ready",
                "epoch": int(getattr(step_authority, "epoch", -1)),
                "layer_index": int(getattr(state, "layer_index", -1)),
                "handle_id": id(getattr(step_authority, "selected_scope_wait_handle", None)),
                "cached_launch_ready": bool(state.step_cache_cached_launch_ready),
                "cached_lengths_ok": bool(state.step_cache_cached_lengths_ok),
                "cached_kv_batch_idx_identity": bool(
                    state.step_cache_cached_kv_batch_idx_identity
                ),
                "cached_status_ok": bool(state.step_cache_cached_status_ok),
                "cached_freshness_ok": bool(state.step_cache_cached_freshness_ok),
            }
        )
    publish_selected_scope_launch_ready_if_needed(
        state=state,
        step_authority=step_authority,
    )
