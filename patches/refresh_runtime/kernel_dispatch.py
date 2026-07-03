from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Set, Tuple

import torch

from patches.runtime_deps import require_runtime_dep
from patches.sparse_constants import (
    _DYNAMIC_ENV,
    _FORCE_COMPACT_OFF_CACHED,
    _FORCE_DENSE_CACHED,
    _LOGF_PRODUCER_NONE,
    _LOGF_OUT_FP32_CACHED,
    _ROW_MODE_COMPACT,
    _ROW_MODE_DENSE,
    _VALIDATE_COMPACT_META_CACHED,
    _VALIDATE_META_CONTRACT_CACHED,
)
from patches.sparse_types import (
    LogitSpec,
    StepCaptureLayout,
    StepContext,
)
from patches.refresh_runtime.meta_pack import (
    _build_logits_capture_specs,
    _collect_refresh_dynamic_dirty_rows,
    _pack_dispatch_meta,
    _pack_meta_from_local_plan,
)
from patches.sparse_utils import (
    _range,
)
from patches.decode_runtime.submit_profile import (
    record_sparse_submit_profile as _record_submit_profile,
    sparse_submit_profile_enabled as _submit_profile_enabled,
    sparse_submit_profile_time_ns as _submit_profile_time_ns,
)

from patches.layer_state import LayerState

if TYPE_CHECKING:
    from patches.vllm_sparse_patch import VLLMSparseController
    from patches.step_authority import StepAuthority

import logging
_log = logging.getLogger("vllm_sparse.kernel_dispatch")

_prefill_update_key_norms = require_runtime_dep("_prefill_update_key_norms")
_get_global_decode_bound_layer = require_runtime_dep("_get_global_decode_bound_layer")
from triton_kernel.flash_attn_score_dump_fwd import flash_attn_score_dump_fwd_unified
from triton_kernel.req_meta_flag_codec import validate_sink_tokens

DispatchResult = Tuple[
    Sequence[int],          # refresh_slots
    Set[int],               # bootstrap_slots_set
    bool,                   # profile_enabled
    Optional[Dict[str, float]],  # profile_stats
    Sequence[int],          # refresh_slot_list_for_payload
    Optional["StepCaptureLayout"],  # refresh_layout
]



_EMPTY_TUPLE: tuple = ()
_EMPTY_FROZENSET: frozenset = frozenset()
_META_SOURCE_LOCAL_PACK = 1
_META_SOURCE_BOUND = 2
_META_SOURCE_RUNTIME_PACK = 3

def _validate_meta_contract_enabled() -> bool:
    if _DYNAMIC_ENV:
        return os.environ.get("VLLM_SPARSE_VALIDATE_META_CONTRACT", "0") == "1"
    return _VALIDATE_META_CONTRACT_CACHED


def _choose_meta_source(
    *,
    is_decode: bool,
    has_decode_row: bool,
    has_prefill_row: bool,
    use_local_pack: bool,
    allow_bound_meta: bool,
) -> int:
    if use_local_pack:
        return _META_SOURCE_LOCAL_PACK
    if allow_bound_meta and is_decode and has_decode_row and (not has_prefill_row):
        return _META_SOURCE_BOUND
    return _META_SOURCE_RUNTIME_PACK


def _resolve_dispatch_meta_source(
    *,
    is_decode: bool,
    has_decode_row: bool,
    has_prefill_row: bool,
    has_request_phase_mix: bool,
    layer_refresh_active: bool,
    refresh_dynamic_decode_dirty_any: bool,
    layer_group_inactive: bool,
) -> Tuple[int, bool, bool]:
    # R.2: 仅在 request 级真实混合（mask==0b11）时触发 mix local-pack。
    use_local_pack_in_mix = bool(has_request_phase_mix)
    use_local_pack_for_refresh_dirty = bool(
        is_decode
        and has_decode_row
        and bool(layer_refresh_active)
        and bool(refresh_dynamic_decode_dirty_any)
    )
    use_local_pack = bool(use_local_pack_in_mix or use_local_pack_for_refresh_dirty)
    meta_source = _choose_meta_source(
        is_decode=bool(is_decode),
        has_decode_row=bool(has_decode_row),
        has_prefill_row=bool(has_prefill_row),
        use_local_pack=bool(use_local_pack),
        allow_bound_meta=(not bool(layer_group_inactive)),
    )

    return int(meta_source), bool(use_local_pack_in_mix), bool(use_local_pack_for_refresh_dirty)

def _maybe_update_slot_rows(
    *,
    state: LayerState,
    slot_row_map: Dict[int, int],
    slot_by_row_key: Tuple[int, ...],
    epoch_hint: int,
) -> None:
    """仅在 slot->row 映射变化/缓存缺失时刷新 CPU slot->row mirror。"""
    need_update = (
        state.slot_batch_rows_cpu is None
        or len(state.slot_batch_rows_cpu) != state.batch_size
        or state._slot_row_map_key != slot_by_row_key
    )
    if need_update:
        state.update_slot_rows(slot_row_map)
        state._slot_row_map_key = slot_by_row_key
    state._slot_row_map_epoch = int(epoch_hint)


def _maybe_wait_for_async_refresh_and_cleanup(
    controller: "VLLMSparseController",
    state: LayerState,
    cache_key: int,
    key_cache: torch.Tensor,
    device: torch.device,
) -> None:
    if controller is None:
        return
    layer_index_global = state.layer_index
    if layer_index_global < 0:
        layer_index_global = controller.layer_index_by_cache_key.get(key_cache.data_ptr(), -1)
        if layer_index_global >= 0:
            state.layer_index = layer_index_global
    if layer_index_global >= 0:
        # 复用缓存的 layer->chunk/buf 映射；若 epoch 不一致则回退重算并刷新缓存
        chunk_id = state.capture_chunk_id
        buf_id = state.capture_buf_id
        if (
            state.layer_index_epoch != controller._layer_index_cache_epoch
            or chunk_id < 0
            or buf_id < 0
        ):
            chunk_id, buf_id, slot_in_chunk = controller._map_global_layer_to_capture_slot(layer_index_global)
            state.capture_chunk_id = chunk_id
            state.capture_buf_id = buf_id
            state.capture_slot_in_chunk = slot_in_chunk
            state.layer_index_epoch = controller._layer_index_cache_epoch

        need_wait, _ = controller._compute_wait_decision(
            buf_id=buf_id,
            epoch=controller.step_context_epoch,
            path_tag="decode_layer",
            consume_step_token=False,
        )
        if need_wait:
            controller._main_stream_wait_for_chunk_done(
                buf_id=buf_id,
                device=device,
                chunk_id=chunk_id,
                epoch=controller.step_context_epoch,
            )

def _validate_dispatch_meta_contract(
    *,
    batch_size: int,
    has_compact_row: bool,
    req_meta_i32: torch.Tensor,
    req_meta_i64: torch.Tensor,
    req_ids: Sequence[str],
    slot_by_row: List[int],
    block_table: torch.Tensor,
    block_size: int,
    key_compact: torch.Tensor,
    state: 'LayerState',
) -> None:
    """Validate packed dispatch metadata (compact meta + meta contract).

    Raises RuntimeError on any invariant violation.  Called only when
    VLLM_SPARSE_VALIDATE_COMPACT_META or VLLM_SPARSE_VALIDATE_META_CONTRACT
    is enabled.
    """
    validate_compact_meta = (
        os.environ.get("VLLM_SPARSE_VALIDATE_COMPACT_META", "0") == "1"
    ) if _DYNAMIC_ENV else _VALIDATE_COMPACT_META_CACHED
    validate_meta_contract = _validate_meta_contract_enabled()

    meta_i32_cpu = req_meta_i32[:batch_size, :].to("cpu")
    meta_i64_cpu = req_meta_i64[:batch_size, :].to("cpu")

    if validate_compact_meta and has_compact_row:
        compact_total_blocks = key_compact.shape[0] if key_compact.dim() > 0 else 0
        compact_total_tokens = state.compact_arena_pos.shape[1] if state.compact_arena_pos.dim() == 2 else 0
        compact_stride_tokens = max(0, state.compact_stride_tokens)
        for row in range(batch_size):
            flags = int(meta_i32_cpu[row, 5].item())
            use_compact_row = (flags & 1) != 0
            if not use_compact_row:
                continue
            req_id_dbg = req_ids[row] if row < len(req_ids) else f"row:{row}"
            slot_dbg = slot_by_row[row] if row < len(slot_by_row) else -1
            compact_kv_len_dbg = int(meta_i32_cpu[row, 2].item())
            compact_block_cnt_dbg = int(meta_i32_cpu[row, 1].item())
            compact_base_block_dbg = int(meta_i64_cpu[row, 1].item())
            token_row_base_dbg = int(meta_i64_cpu[row, 3].item())
            if compact_kv_len_dbg <= 0 or compact_block_cnt_dbg <= 0:
                raise RuntimeError(
                    "invalid compact meta(non-positive length): "
                    f"row={row} req={req_id_dbg} slot={slot_dbg} "
                    f"kv_len={compact_kv_len_dbg} block_cnt={compact_block_cnt_dbg}"
                )
            if compact_stride_tokens > 0 and compact_kv_len_dbg > compact_stride_tokens:
                raise RuntimeError(
                    "invalid compact meta(kv_len exceeds stride): "
                    f"row={row} req={req_id_dbg} slot={slot_dbg} "
                    f"kv_len={compact_kv_len_dbg} stride_tokens={compact_stride_tokens}"
                )
            if compact_base_block_dbg < 0 or (compact_base_block_dbg + compact_block_cnt_dbg) > compact_total_blocks:
                raise RuntimeError(
                    "invalid compact meta(block range overflow): "
                    f"row={row} req={req_id_dbg} slot={slot_dbg} "
                    f"base_block={compact_base_block_dbg} block_cnt={compact_block_cnt_dbg} "
                    f"total_blocks={compact_total_blocks}"
                )
            if token_row_base_dbg < 0 or token_row_base_dbg != compact_base_block_dbg * block_size:
                raise RuntimeError(
                    "invalid compact meta(token base mismatch): "
                    f"row={row} req={req_id_dbg} slot={slot_dbg} "
                    f"token_row_base={token_row_base_dbg} "
                    f"expected={compact_base_block_dbg * block_size}"
                )
            if (token_row_base_dbg + compact_kv_len_dbg) > compact_total_tokens:
                raise RuntimeError(
                    "invalid compact meta(token range overflow): "
                    f"row={row} req={req_id_dbg} slot={slot_dbg} "
                    f"token_base={token_row_base_dbg} kv_len={compact_kv_len_dbg} "
                    f"total_tokens={compact_total_tokens}"
                )

    if validate_meta_contract:
        block_table_rows = block_table.shape[0] if block_table.dim() >= 1 else 0
        for row in range(batch_size):
            req_id_dbg = req_ids[row] if row < len(req_ids) else f"row:{row}"
            slot_dbg = slot_by_row[row] if row < len(slot_by_row) else -1

            meta0 = int(meta_i64_cpu[row, 0].item())
            if meta0 < 0 or meta0 >= block_table_rows:
                raise RuntimeError(
                    "invalid req_meta_i64 block_row_base_paged: "
                    f"row={row} req={req_id_dbg} slot={slot_dbg} "
                    f"block_row_base_paged={meta0} block_table_rows={block_table_rows}"
                )

            flags = int(meta_i32_cpu[row, 5].item())
            use_compact_row = (flags & 1) != 0
            use_log_f_row = (flags & 8) != 0
            if use_compact_row and use_log_f_row:
                raise RuntimeError(
                    "invalid req_meta flags(compact+log_f both enabled): "
                    f"row={row} req={req_id_dbg} slot={slot_dbg} flags={flags}"
                )

            if use_compact_row:
                logits_base_ptr = int(meta_i64_cpu[row, 2].item())
                if logits_base_ptr != 0:
                    raise RuntimeError(
                        "invalid compact meta(logits base ptr must be zero): "
                        f"row={row} req={req_id_dbg} slot={slot_dbg} logits_base_ptr={logits_base_ptr}"
                    )
                continue

            if not use_log_f_row:
                continue

            log_f_stride_head = int(meta_i32_cpu[row, 1].item())
            logits_last_n = int(meta_i32_cpu[row, 2].item())
            logits_capacity = int(meta_i32_cpu[row, 4].item())
            scratch_ptr = int(meta_i64_cpu[row, 1].item())
            logits_base_ptr = int(meta_i64_cpu[row, 2].item())
            denom_ptr = int(meta_i64_cpu[row, 3].item())
            if log_f_stride_head <= 0 or logits_last_n <= 0 or logits_capacity <= 0:
                raise RuntimeError(
                    "invalid dense log_f meta(non-positive i32 fields): "
                    f"row={row} req={req_id_dbg} slot={slot_dbg} "
                    f"log_f_stride_head={log_f_stride_head} "
                    f"logits_last_n={logits_last_n} logits_capacity={logits_capacity}"
                )
            if logits_base_ptr <= 0:
                raise RuntimeError(
                    "invalid dense log_f meta(missing logits base ptr): "
                    f"row={row} req={req_id_dbg} slot={slot_dbg} logits_base_ptr={logits_base_ptr}"
                )
            if logits_last_n > 1 and (scratch_ptr <= 0 or denom_ptr <= 0):
                raise RuntimeError(
                    "invalid dense log_f meta(last_n>1 missing scratch/denom ptr): "
                    f"row={row} req={req_id_dbg} slot={slot_dbg} "
                    f"logits_last_n={logits_last_n} scratch_ptr={scratch_ptr} denom_ptr={denom_ptr}"
                )




def run_unified_attention_dispatcher_impl(
    controller: 'VLLMSparseController',
    state: LayerState,
    cache_key: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    seqused_k: torch.Tensor,
    max_seqlen_k: int,
    softmax_scale: float,
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    window_size: Optional[Tuple[int, int]] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    request_ids: Sequence[str] = (),
    step_context: Optional[StepContext] = None,
    prefill_capture_plan: Optional[Dict[int, int]] = None,
    chunk_query_lengths: Optional[torch.Tensor] = None,
) -> DispatchResult:
    profile_enabled = False
    profile_stats: Optional[Dict[str, float]] = None

    device = q.device
    num_heads = q.shape[1]
    head_dim = q.shape[2]
    num_kv_heads = k.shape[2]
    block_size = k.shape[1]
    _profile_enabled = _submit_profile_enabled()
    _profile_layer_index = int(getattr(state, "layer_index", -1))
    _profile_total_start_ns = _submit_profile_time_ns() if _profile_enabled else 0
    _profile_segment_start_ns = _profile_total_start_ns

    # --------------------------------------------------------------
    # 异步 refresh 正确性屏障（关键）
    #
    # refresh_stream 会异步写入：
    # 1) step capture ring（给 selector 消费）
    # 2) compact arena（给后续 step/layer 的 kernel 读取）
    #
    # 若仅在 needs_logits 时才 wait_event，会出现：
    # - 下一步无 logits capture，但 kernel 仍可能读取 compact（读到未完成写入的数据）
    # - 下一步清理/复用 slot 时与 refresh_stream 写 compact 发生跨 stream 竞态
    #
    # 因此这里在“chunk 起始层”（slot_in_chunk==0）插入 wait_event：
    # - 保证复用 ring buf 前，上一轮异步 selector+rebuild 已完成
    # - 避免每层重复 wait_event 的额外开销
    # --------------------------------------------------------------
    _maybe_wait_for_async_refresh_and_cleanup(
        controller=controller,
        state=state,
        cache_key=cache_key,
        key_cache=k,
        device=device,
    )
    if _profile_enabled:
        _profile_now_ns = _submit_profile_time_ns()
        _record_submit_profile(
            "dispatch.wait_cleanup", _profile_layer_index, _profile_now_ns - _profile_segment_start_ns
        )
        _profile_segment_start_ns = _profile_now_ns

    _auth_decode = controller.step_authority
    if _auth_decode is None:
        raise RuntimeError("dispatcher requires step_authority single-source decode gate")
    is_decode = bool(_auth_decode.is_decode_only)

    # 预先对齐 slots，确保 refresh 判定使用稳定的 slot 索引而非本帧行号。
    with _range("sparse.dispatch.req_ids"):
        batch_size = cu_seqlens_q.shape[0] - 1
        req_ids = request_ids
        if batch_size > 0 and (not req_ids):
            raise RuntimeError(
                "dispatcher requires request_ids single-source input from worker entry"
            )

    # Step 1: Build step-wise plan (no RequestOp)
    layer_refresh_active = False
    refresh_slots: Sequence[int] = []
    ctx_used = False
    if step_context is not None:
        if step_context.num_reqs == batch_size and tuple(req_ids) == step_context.req_ids:
            ctx_used = True
    if not ctx_used:
        raise RuntimeError("StepContext missing or req_ids mismatch; step-wise path required")
    # envelope_cached_snapshot: 复用 prepare_step_context 预编译快照，避免热路径重复 tuple(int(...)) 组装。
    step_envelope_v2 = step_context.step_envelope_v2
    if step_envelope_v2 is None:
        raise RuntimeError("StepContext step_envelope_v2 missing; single-source refresh signals required")
    if step_envelope_v2.req_ids != step_context.req_ids:
        raise RuntimeError("StepContext step_envelope_v2 req_ids mismatch")
    if (
        step_envelope_v2.handle_id != step_context.step_handle_id
        or step_envelope_v2.handle_generation != step_context.step_handle_generation
    ):
        raise RuntimeError(
            "step_envelope_v2 handle mismatch; "
            f"env=({step_envelope_v2.handle_id}, {step_envelope_v2.handle_generation}) "
            f"ctx=({step_context.step_handle_id}, {step_context.step_handle_generation})"
        )
    layer_effective_refresh_by_row_step: Tuple[bool, ...] = tuple()
    # step-wise 路径：slot 对齐必须来自 prepare_step_context 产出的 snapshot，
    # 避免 dispatcher 热路径重复构造 request->slot 映射 dict。
    if state.last_active_request_ids != step_context.req_ids:
        state.align_slots_from_snapshot(
            request_ids=step_context.req_ids,
            slot_by_row=step_envelope_v2.slot_by_row,
            epoch=step_envelope_v2.epoch,
        )
    # step-static semantic snapshot: cache once per step instead of per-layer.
    _cached_semantic_snapshot = controller._get_step_semantic_snapshot()
    # StepAuthority 跨层共享缓存（提前判定，用于后续 refresh/prefill 低开销分支）
    _auth = _auth_decode
    use_step_meta = False
    q_lens_step: Tuple[int, ...] = tuple()
    seq_lens_step: Tuple[int, ...] = tuple()
    is_prefill_step: Tuple[bool, ...] = tuple()
    bootstrap_done_step: Tuple[bool, ...] = tuple()
    if (
        step_context is not None
        and _auth.epoch == step_context.epoch
        and int(_auth.step_handle_id) == int(step_context.step_handle_id)
        and int(_auth.step_handle_generation)
        == int(step_context.step_handle_generation)
    ):
        try:
            if _auth.req_ids == step_context.req_ids and _auth.batch_size >= batch_size:
                is_prefill_cached = _auth.is_prefill_by_row
                if is_prefill_cached is not None and len(is_prefill_cached) >= batch_size:
                    use_step_meta = True
                    q_lens_step = _auth.q_lens_by_row
                    seq_lens_step = _auth.context_kv_len_by_row
                    is_prefill_step = is_prefill_cached
                    bootstrap_done_step = _auth.bootstrap_done_by_row
        except (AttributeError, KeyError, IndexError, TypeError) as e:
            _log.error("step_authority cache read failed: %s; disabling step_meta this layer", e)
            use_step_meta = False
    _sit = int(step_context.step_identity_token) if step_context is not None else -1
    if use_step_meta:
        auth_layer_effective_refresh_by_row: Tuple[bool, ...] = (
            _auth.layer_effective_refresh_by_row
            if _auth.batch_size == batch_size
            else _auth.layer_effective_refresh_by_row[:batch_size]
        )
        if len(auth_layer_effective_refresh_by_row) < batch_size:
            raise RuntimeError(
                "step_authority.layer_effective_refresh_by_row length mismatch; "
                f"rows={len(auth_layer_effective_refresh_by_row)} batch={batch_size}"
            )
        # R.4: refresh 判定主源使用 StepAuthority。
        layer_effective_refresh_by_row_step = auth_layer_effective_refresh_by_row

    # layer_refresh_active：layer-gated 版本，决定当层是否实际执行 refresh。
    # layer-group 仅作为执行域 gate，不再派生第二套 dispatcher / metadata 语义分支。
    layer_refresh_active = bool(_auth.has_refresh_row) if use_step_meta else bool(any(layer_effective_refresh_by_row_step))
    layer_group_inactive = False
    if layer_refresh_active:
        if step_envelope_v2.layer_group_enabled:
            layer_index = state.layer_index
            if layer_index >= 0 and (layer_index & 1) != step_envelope_v2.layer_group_active:
                layer_group_inactive = True
                layer_refresh_active = False
    state.bootstrap_done = step_envelope_v2.bootstrap_done
    state.last_reason = step_envelope_v2.refresh_reason
    force_dense = _FORCE_DENSE_CACHED
    force_compact_off = _FORCE_COMPACT_OFF_CACHED

    step_bound_meta = controller._require_step_bound_meta(
        step_context=step_context,
        stage="kernel dispatch",
    )
    refresh_bundle = step_bound_meta.refresh_bundle
    refresh_nonempty_step = (
        int(getattr(_auth, "refresh_decode_count", 0))
        + int(getattr(_auth, "refresh_prefill_count", 0))
    ) > 0
    # 缓存门控：同一 step 内只验证一次 refresh bundle 签名。
    _bundle_already_validated = (controller._refresh_bundle_validated_token == _sit)
    if not _bundle_already_validated:
        if refresh_nonempty_step:
            if refresh_bundle is None:
                raise RuntimeError(
                    "refresh step requires step_bound_meta.refresh_bundle in strict mode"
                )
            bundle_signature = tuple(int(v) for v in refresh_bundle.signature)
            expected_signature = (
                int(step_context.epoch),
                int(step_context.step_handle_id),
                int(step_context.step_handle_generation),
                int(_auth.req_set_hash),
                int(_auth.row_phase_hash),
            )
            if bundle_signature != expected_signature:
                raise RuntimeError(
                    "refresh bundle signature mismatch; "
                    f"bundle={bundle_signature} expected={expected_signature}"
                )
            if int(refresh_bundle.enabled_i32) != 1:
                raise RuntimeError("refresh bundle disabled on refresh step")
            if int(refresh_bundle.req_meta_ready_i32) != 1:
                raise RuntimeError("refresh bundle req_meta_ready_i32 != 1 on refresh step")
            if len(refresh_bundle.logf_mask_by_row) < batch_size:
                raise RuntimeError(
                    "refresh bundle logf_mask coverage mismatch; "
                    f"rows={len(refresh_bundle.logf_mask_by_row)} batch={batch_size}"
                )
            if len(refresh_bundle.logits_last_n_by_row) < batch_size:
                raise RuntimeError(
                    "refresh bundle logits_last_n coverage mismatch; "
                    f"rows={len(refresh_bundle.logits_last_n_by_row)} batch={batch_size}"
                )
        elif refresh_bundle is not None:
            raise RuntimeError(
                "non-refresh step must not carry refresh bundle in strict mode"
            )
        controller._refresh_bundle_validated_token = _sit
    if _profile_enabled:
        _profile_now_ns = _submit_profile_time_ns()
        _record_submit_profile(
            "dispatch.authority_bundle", _profile_layer_index, _profile_now_ns - _profile_segment_start_ns
        )
        _profile_segment_start_ns = _profile_now_ns

    # StepAuthority 单源消费：每层仅做 O(1) 分组选择 + 参数读取。
    if not use_step_meta or _auth is None:
        raise RuntimeError(
            "dispatcher requires step_authority single-source metadata; "
            f"epoch={int(step_context.epoch)} batch={int(batch_size)} layer={int(state.layer_index)}"
        )

    # 关键契约：默认路径必须由 StepContext 提供完整 q_lens/seq_lens。
    q_lens_ctx = step_context.q_lens
    seq_lens_ctx = step_context.seq_lens
    if len(q_lens_ctx) < batch_size or len(seq_lens_ctx) < batch_size:
        raise RuntimeError(
            "kernel dispatch missing q_lens/seq_lens in default path; "
            f"epoch={int(step_context.epoch)} batch={int(batch_size)} layer={int(state.layer_index)} "
            f"q_lens={len(q_lens_ctx)} seq_lens={len(seq_lens_ctx)}"
        )

    _bs_match = (_auth.batch_size == batch_size)
    if _bs_match:
        base_use_compact_by_row = _auth.use_compact_by_row
        base_logf_producer_by_row = _auth.dispatch_logf_producer_by_row
        base_needs_logits_by_row = _auth.needs_logits_by_row
        base_logits_last_n_by_row = _auth.logits_last_n_by_row
        base_row_mode_signature = _auth.row_mode_by_row
        base_any_needs_logits = bool(_auth.any_needs_logits)
        base_prefill_needs_logits = bool(_auth.prefill_needs_logits)
        base_refresh_needs_logits = bool(_auth.refresh_needs_logits)
        base_hint_log_f_eq1 = bool(_auth.hint_log_f_eq1)
        base_hint_log_f_gt1 = bool(_auth.hint_log_f_gt1)
        base_logits_rows = _auth.logits_rows
        base_logits_rows_gt1 = _auth.logits_rows_gt1
        base_has_compact_row = bool(_auth.has_compact_row)
        base_all_compact_rows = bool(_auth.hint_all_compact)
    else:
        base_use_compact_by_row = _auth.use_compact_by_row[:batch_size]
        base_logf_producer_by_row = _auth.dispatch_logf_producer_by_row[:batch_size]
        base_needs_logits_by_row = _auth.needs_logits_by_row[:batch_size]
        base_logits_last_n_by_row = _auth.logits_last_n_by_row[:batch_size]
        base_row_mode_signature = _auth.row_mode_by_row[:batch_size]
        base_any_needs_logits = any(bool(v) for v in base_needs_logits_by_row)
        base_prefill_needs_logits = any(
            bool(base_needs_logits_by_row[row]) and bool(_auth.is_prefill_by_row[row])
            for row in range(batch_size)
        )
        base_refresh_needs_logits = any(
            bool(base_needs_logits_by_row[row]) and bool(_auth.layer_effective_refresh_by_row[row])
            for row in range(batch_size)
        )
        base_logits_rows = tuple(
            row
            for row in range(batch_size)
            if bool(base_needs_logits_by_row[row]) and int(base_logits_last_n_by_row[row]) > 0
        )
        base_logits_rows_gt1 = tuple(
            row for row in base_logits_rows if int(base_logits_last_n_by_row[row]) > 1
        )
        base_hint_log_f_eq1 = any(
            int(base_logits_last_n_by_row[row]) == 1 for row in base_logits_rows
        )
        base_hint_log_f_gt1 = any(
            int(base_logits_last_n_by_row[row]) > 1 for row in base_logits_rows
        )
        base_has_compact_row = any(bool(v) for v in base_use_compact_by_row)
        base_all_compact_rows = all(bool(v) for v in base_use_compact_by_row) if batch_size > 0 else False

    if layer_refresh_active:
        if refresh_bundle is None:
            raise RuntimeError(
                "active refresh layer requires refresh bundle in strict mode"
            )
        use_compact_by_row = base_use_compact_by_row
        logf_producer_by_row = base_logf_producer_by_row
        group_needs_logits = base_needs_logits_by_row
        group_logits_last_n = (
            refresh_bundle.logits_last_n_by_row
            if _bs_match
            else refresh_bundle.logits_last_n_by_row[:batch_size]
        )
        row_mode_signature = base_row_mode_signature
        any_needs_logits = base_any_needs_logits
        prefill_needs_logits = base_prefill_needs_logits
        refresh_needs_logits = base_refresh_needs_logits
        dense_log_f_has_last_n_eq1 = base_hint_log_f_eq1
        dense_log_f_has_last_n_gt1 = base_hint_log_f_gt1
        logits_rows_cached = base_logits_rows
        logits_rows_gt1_cached = base_logits_rows_gt1
        refresh_slots_sorted_tuple = refresh_bundle.refresh_slots
        refresh_slot_list_for_payload_plan = refresh_bundle.refresh_slots
        bootstrap_slots_sorted_tuple = _auth.bootstrap_slots
        has_compact_row = base_has_compact_row
        all_compact_rows = base_all_compact_rows
    elif layer_group_inactive:
        use_compact_list = [bool(v) for v in base_use_compact_by_row]
        logf_producer_list = [int(v) for v in base_logf_producer_by_row]
        needs_logits_list = [bool(v) for v in base_needs_logits_by_row]
        logits_last_n_list = [int(v) for v in base_logits_last_n_by_row]
        row_mode_list = [int(v) for v in base_row_mode_signature]
        is_prefill_slice = (
            _auth.is_prefill_by_row if _bs_match else _auth.is_prefill_by_row[:batch_size]
        )
        layer_refresh_slice = (
            _auth.layer_effective_refresh_by_row
            if _bs_match
            else _auth.layer_effective_refresh_by_row[:batch_size]
        )
        bootstrap_done_slice = (
            _auth.bootstrap_done_by_row
            if _bs_match
            else _auth.bootstrap_done_by_row[:batch_size]
        )
        short_dense_slice = (
            _auth.short_dense_by_row
            if _bs_match
            else _auth.short_dense_by_row[:batch_size]
        )
        for row in range(batch_size):
            if not bool(layer_refresh_slice[row]) or bool(is_prefill_slice[row]):
                continue
            downgraded_use_compact = (
                (not bool(short_dense_slice[row]))
                and bool(bootstrap_done_slice[row])
                and (not force_dense)
            )
            use_compact_list[row] = bool(downgraded_use_compact)
            logf_producer_list[row] = int(_LOGF_PRODUCER_NONE)
            needs_logits_list[row] = False
            logits_last_n_list[row] = 0
            row_mode_list[row] = int(
                _ROW_MODE_COMPACT if downgraded_use_compact else _ROW_MODE_DENSE
            )

        use_compact_by_row = tuple(use_compact_list)
        logf_producer_by_row = tuple(logf_producer_list)
        group_needs_logits = tuple(needs_logits_list)
        group_logits_last_n = tuple(logits_last_n_list)
        row_mode_signature = tuple(row_mode_list)
        logits_rows_cached = tuple(
            row
            for row in range(batch_size)
            if bool(group_needs_logits[row]) and int(group_logits_last_n[row]) > 0
        )
        logits_rows_gt1_cached = tuple(
            row for row in logits_rows_cached if int(group_logits_last_n[row]) > 1
        )
        any_needs_logits = any(bool(v) for v in group_needs_logits)
        prefill_needs_logits = any(
            bool(group_needs_logits[row]) and bool(is_prefill_slice[row])
            for row in range(batch_size)
        )
        refresh_needs_logits = any(
            bool(group_needs_logits[row]) and bool(layer_refresh_slice[row])
            for row in range(batch_size)
        )
        dense_log_f_has_last_n_eq1 = any(
            int(group_logits_last_n[row]) == 1 for row in logits_rows_cached
        )
        dense_log_f_has_last_n_gt1 = any(
            int(group_logits_last_n[row]) > 1 for row in logits_rows_cached
        )
        refresh_slots_sorted_tuple = _EMPTY_TUPLE
        bootstrap_slots_sorted_tuple = _EMPTY_TUPLE
        refresh_slot_list_for_payload_plan = _EMPTY_TUPLE
        has_compact_row = any(bool(v) for v in use_compact_by_row) if batch_size > 0 else False
        all_compact_rows = all(bool(v) for v in use_compact_by_row) if batch_size > 0 else False
    else:
        use_compact_by_row = base_use_compact_by_row
        logf_producer_by_row = base_logf_producer_by_row
        group_needs_logits = base_needs_logits_by_row
        group_logits_last_n = base_logits_last_n_by_row
        row_mode_signature = base_row_mode_signature
        any_needs_logits = base_any_needs_logits
        prefill_needs_logits = base_prefill_needs_logits
        refresh_needs_logits = base_refresh_needs_logits
        dense_log_f_has_last_n_eq1 = base_hint_log_f_eq1
        dense_log_f_has_last_n_gt1 = base_hint_log_f_gt1
        logits_rows_cached = base_logits_rows
        logits_rows_gt1_cached = base_logits_rows_gt1
        refresh_slots_sorted_tuple = _EMPTY_TUPLE
        bootstrap_slots_sorted_tuple = _EMPTY_TUPLE
        refresh_slot_list_for_payload_plan = _EMPTY_TUPLE
        has_compact_row = base_has_compact_row
        all_compact_rows = base_all_compact_rows

    slot_by_row_seq: Sequence[int] = (
        _auth.slot_by_row if _bs_match else _auth.slot_by_row[:batch_size]
    )
    if len(slot_by_row_seq) < batch_size:
        raise RuntimeError(
            "step_authority.slot_by_row length mismatch; "
            f"rows={len(slot_by_row_seq)} batch={batch_size}"
        )
    slot_by_row: Sequence[int] = slot_by_row_seq
    slot_by_row_has_negative = (
        bool(_auth.slot_by_row_has_negative)
        if _bs_match
        else any(slot < 0 for slot in slot_by_row_seq)
    )
    if slot_by_row_has_negative:
        state.align_slots_from_snapshot(
            request_ids=step_context.req_ids,
            slot_by_row=step_envelope_v2.slot_by_row,
            epoch=step_envelope_v2.epoch,
        )
        req_id_to_slot = state.request_id_to_slot
        slot_by_row_list = list(slot_by_row_seq)
        for row in range(batch_size):
            slot_by_row_list[row] = int(req_id_to_slot.get(req_ids[row], -1))
        slot_by_row = slot_by_row_list
    slot_by_row_tuple = (
        slot_by_row_seq if slot_by_row is slot_by_row_seq else tuple(slot_by_row)
    )

    is_prefill_by_row = (
        _auth.is_prefill_by_row if _bs_match else _auth.is_prefill_by_row[:batch_size]
    )
    q_len_by_row: Sequence[int] = q_lens_step if q_lens_step else tuple(0 for _ in range(batch_size))
    context_kv_len_by_row: Sequence[int] = (
        seq_lens_step if seq_lens_step else tuple(0 for _ in range(batch_size))
    )
    logits_spec_by_row: Optional[List[Optional[LogitSpec]]] = None
    log_f_denom_ptr_by_row: Optional[List[int]] = None
    compact_kv_len_by_row: Optional[List[int]] = None
    compact_kv_len_max = 0
    # decode 路径下用于布局化生成 logits/log_f 指针（避免每层 torch.tensor(list)）
    refresh_layout_for_log_f_capture: Optional[StepCaptureLayout] = None
    slot_in_chunk_for_log_f_capture: int = 0

    prefill_last_n_by_row: Optional[Tuple[int, ...]] = None
    if controller is not None:
        prefill_last_n_epoch = int(controller.step_prefill_capture_last_n_epoch)
        prefill_last_n_handle_id = int(controller.step_prefill_capture_last_n_handle_id)
        prefill_last_n_handle_generation = int(controller.step_prefill_capture_last_n_handle_generation)
        if (
            prefill_last_n_epoch == int(_auth.epoch)
            and prefill_last_n_handle_id == int(step_context.step_handle_id)
            and prefill_last_n_handle_generation
            == int(step_context.step_handle_generation)
        ):
            prefill_last_n_by_row = controller.step_prefill_capture_last_n_by_row
            if prefill_last_n_by_row is not None and len(prefill_last_n_by_row) < batch_size:
                prefill_last_n_by_row = None

    bootstrap_slots_set = (
        frozenset(bootstrap_slots_sorted_tuple)
        if bootstrap_slots_sorted_tuple
        else _EMPTY_FROZENSET
    )
    refresh_slots: Sequence[int] = refresh_slots_sorted_tuple
    refresh_slot_list_for_payload: Sequence[int] = refresh_slot_list_for_payload_plan
    if (refresh_slots or bootstrap_slots_set) and (not refresh_slot_list_for_payload):
        raise RuntimeError(
            "refresh payload slot_list is empty while refresh/bootstrap is pending"
        )
    if refresh_slot_list_for_payload and (not refresh_slots) and (not bootstrap_slots_set):
        raise RuntimeError(
            "refresh payload slot_list is non-empty while refresh/bootstrap intent is empty"
        )
    if _bs_match:
        has_decode_row = bool(_auth.has_decode_row)
        has_prefill_row = bool(_auth.has_prefill_row)
        has_request_phase_mix = bool(_auth.has_request_phase_mix)
    else:
        has_prefill_row = any(bool(v) for v in is_prefill_by_row)
        has_decode_row = any((not bool(v)) for v in is_prefill_by_row)
        has_request_phase_mix = bool(has_prefill_row and has_decode_row)

    # slot->row 映射在同一步内稳定；避免每层重复清空/填充 slot_batch_rows
    slot_by_row_key = slot_by_row_tuple  # 复用已构建的 tuple
    slot_row_map: Dict[int, int]
    _slot_map_cache_val = controller._step_dispatcher_slot_row_map
    if (
        controller._step_dispatcher_slot_row_map_token == _sit
        and controller._step_dispatcher_slot_row_map_key == slot_by_row_key
        and isinstance(_slot_map_cache_val, dict)
    ):
        slot_row_map = _slot_map_cache_val
    else:
        slot_row_map = {
            slot_by_row[row]: row
            for row in range(batch_size)
            if slot_by_row[row] >= 0
        }
        controller._step_dispatcher_slot_row_map_token = _sit
        controller._step_dispatcher_slot_row_map_key = slot_by_row_key
        controller._step_dispatcher_slot_row_map = slot_row_map
    _maybe_update_slot_rows(
        state=state,
        slot_row_map=slot_row_map,
        slot_by_row_key=slot_by_row_key,
        epoch_hint=int(controller.step_context_epoch),
    )
    if _profile_enabled:
        _profile_now_ns = _submit_profile_time_ns()
        _record_submit_profile(
            "dispatch.row_slot_map", _profile_layer_index, _profile_now_ns - _profile_segment_start_ns
        )
        _profile_segment_start_ns = _profile_now_ns

    # Step 级缓存：refresh-dynamic 是否触达 decode row（layer 无关）。
    refresh_dynamic_decode_dirty_any = False
    refresh_dynamic_slots_key: Tuple[int, ...] = _EMPTY_TUPLE
    refresh_req_set_hash = int(_auth.req_set_hash)
    refresh_row_phase_hash = int(_auth.row_phase_hash)
    if has_decode_row and refresh_slot_list_for_payload:
        if not isinstance(refresh_slot_list_for_payload, tuple):
            raise RuntimeError(
                "refresh payload slot_list must be tuple from step_authority single-source snapshot"
            )
        refresh_dynamic_slots_key = refresh_slot_list_for_payload
    if has_decode_row and refresh_dynamic_slots_key:
        if (
            controller._step_dispatcher_refresh_dirty_token == _sit
            and controller._step_dispatcher_refresh_dirty_slot_key == slot_by_row_key
            and controller._step_dispatcher_refresh_dirty_payload_slots == refresh_dynamic_slots_key
        ):
            refresh_dynamic_decode_dirty_any = bool(controller._step_dispatcher_refresh_dirty_any_decode)
        else:
            dirty_any = False
            for _slot in refresh_dynamic_slots_key:
                _row = int(slot_row_map.get(int(_slot), -1))
                if _row < 0:
                    continue
                if _row < len(is_prefill_by_row) and bool(is_prefill_by_row[_row]):
                    continue
                dirty_any = True
                break
            if not dirty_any:
                if refresh_dynamic_slots_key != refresh_slot_list_for_payload:
                    raise RuntimeError(
                        "refresh dirty slot_set must come from payload slot_list single-source snapshot"
                    )
                _slot_set = _auth.refresh_capture_slot_set
                dirty_any = bool(
                    _collect_refresh_dynamic_dirty_rows(
                        batch_size=int(batch_size),
                        slot_by_row=slot_by_row,
                        is_prefill_by_row=is_prefill_by_row,
                        refresh_dynamic_slot_set=_slot_set,
                    )
                )
            refresh_dynamic_decode_dirty_any = bool(dirty_any)
            controller._step_dispatcher_refresh_dirty_token = _sit
            controller._step_dispatcher_refresh_dirty_slot_key = slot_by_row_key
            controller._step_dispatcher_refresh_dirty_payload_slots = refresh_dynamic_slots_key
            controller._step_dispatcher_refresh_dirty_any_decode = refresh_dynamic_decode_dirty_any

    if _profile_enabled:
        _profile_now_ns = _submit_profile_time_ns()
        _record_submit_profile(
            "dispatch.refresh_dirty", _profile_layer_index, _profile_now_ns - _profile_segment_start_ns
        )
        _profile_segment_start_ns = _profile_now_ns

    # log_f mixed last_n 支持：允许同一 batch 内同时存在 last_n==1 与 last_n>1 的 log_f slot。
    # kernel 会在一次 launch 内同时启用两条 store 路径（last_n==1 写 fp16 logits；last_n>1 写 fp32 scratch）。

    # Step 1.5: 生成 logits 写入布局（step-wise capture arena）
    if any_needs_logits:
        logits_spec_by_row = [None] * batch_size
        log_f_denom_ptr_by_row = [0] * batch_size
        refresh_layout_for_log_f_capture, slot_in_chunk_for_log_f_capture = _build_logits_capture_specs(
            controller=controller,
            state=state,
            step_context=step_context,
            prefill_capture_plan=prefill_capture_plan,
            needs_logits_by_row=group_needs_logits,
            is_prefill_by_row=is_prefill_by_row,
            slot_by_row=slot_by_row,
            logits_last_n_by_row=group_logits_last_n,
            q_len_by_row=q_len_by_row,
            context_kv_len_by_row=context_kv_len_by_row,
            logits_spec_by_row=logits_spec_by_row,
            log_f_denom_ptr_by_row=log_f_denom_ptr_by_row,
            refresh_slot_list_for_payload_plan=refresh_slot_list_for_payload_plan,
            key_cache=k,
            cu_seqlens_q=cu_seqlens_q,
            seqused_k=seqused_k,
            num_heads=num_heads,
            device=device,
            chunk_query_lengths=chunk_query_lengths,
            batch_size=batch_size,
        )
    if _profile_enabled:
        _profile_now_ns = _submit_profile_time_ns()
        _record_submit_profile(
            "dispatch.logits_specs", _profile_layer_index, _profile_now_ns - _profile_segment_start_ns
        )
        _profile_segment_start_ns = _profile_now_ns
    bound_layer_meta = _get_global_decode_bound_layer(controller, cache_key)
    if bound_layer_meta is None:
        raise RuntimeError(
            "dispatcher requires step-bound layer meta; "
            f"epoch={step_context.epoch} layer_index={state.layer_index} cache_key={int(cache_key)}"
        )
    bound_req_meta_i32 = bound_layer_meta.req_meta_i32
    bound_req_meta_i64 = bound_layer_meta.req_meta_i64
    bound_compact_kv_len_max = int(bound_layer_meta.compact_kv_len_max)
    bound_all_compact = bool(bound_layer_meta.all_compact)
    bound_hint_has_log_f = bool(bound_layer_meta.hint_has_log_f)
    bound_hint_log_f_eq1 = bool(bound_layer_meta.hint_log_f_eq1)
    bound_hint_log_f_gt1 = bool(bound_layer_meta.hint_log_f_gt1)
    runtime_has_compact_row = bool(has_compact_row)
    runtime_all_compact_rows = bool(all_compact_rows)
    runtime_compact_kv_len_max = int(compact_kv_len_max)
    bound_has_compact_row = bool(bound_compact_kv_len_max > 0)
    if _profile_enabled:
        _profile_now_ns = _submit_profile_time_ns()
        _record_submit_profile(
            "dispatch.bound_layer_meta", _profile_layer_index, _profile_now_ns - _profile_segment_start_ns
        )
        _profile_segment_start_ns = _profile_now_ns

    use_fast_pack = is_decode
    # Step 2+3: Prepare compact arena view + launch buffers
    with _range("sparse.dispatch.build_compact"):
        key_compact = bound_layer_meta.k_compact
        value_compact = bound_layer_meta.v_compact
        compact_offsets = None
        use_step_cache_pack_cached = True
        logical_token_indices = bound_layer_meta.token_positions
    if _profile_enabled:
        _profile_now_ns = _submit_profile_time_ns()
        _record_submit_profile(
            "dispatch.build_compact", _profile_layer_index, _profile_now_ns - _profile_segment_start_ns
        )
        _profile_segment_start_ns = _profile_now_ns

    with _range("sparse.dispatch.pack_meta"):
        # 单源判定：req_meta/hint/compact 校验均由同一 source 派生，避免后续改动引入混源。
        meta_source, use_local_pack_in_mix, use_local_pack_for_refresh_dirty = _resolve_dispatch_meta_source(
            is_decode=bool(is_decode),
            has_decode_row=bool(has_decode_row),
            has_prefill_row=bool(has_prefill_row),
            has_request_phase_mix=bool(has_request_phase_mix),
            layer_refresh_active=bool(layer_refresh_active),
            refresh_dynamic_decode_dirty_any=bool(refresh_dynamic_decode_dirty_any),
            layer_group_inactive=bool(layer_group_inactive),
        )
        use_local_pack = bool(meta_source == _META_SOURCE_LOCAL_PACK)
        meta_source_is_bound = bool(meta_source == _META_SOURCE_BOUND and (not layer_group_inactive))
        if use_local_pack and controller is not None and controller._step_profile_enabled():
            if use_local_pack_in_mix:
                controller._step_profile_record_local_pack("mix")
            if use_local_pack_for_refresh_dirty:
                controller._step_profile_record_local_pack("refresh_dynamic")
        if use_local_pack and controller is not None:
            if not hasattr(controller, "_step_local_pack_compact_batch_size"):
                controller._step_local_pack_compact_batch_size = -1
            if not hasattr(controller, "_step_local_pack_compact_stride_blocks"):
                controller._step_local_pack_compact_stride_blocks = -1
            if not hasattr(controller, "_step_local_pack_compact_slot_signature"):
                controller._step_local_pack_compact_slot_signature = tuple()
            if not hasattr(controller, "_step_local_pack_compact_use_compact_signature"):
                controller._step_local_pack_compact_use_compact_signature = tuple()
            compact_stride_blocks = int(state.compact_stride_blocks)
            local_pack_compact_slot_signature = tuple(
                int(slot_by_row[row]) for row in range(batch_size)
            )
            local_pack_compact_use_compact_signature = tuple(
                1 if bool(use_compact_by_row[row]) else 0 for row in range(batch_size)
            )
            # [DUAL-GEN-L2a] offset 签名进 memo 键:双代切代改变 offset 而
            # slot/epoch 可能不变,旧键会回放错半区 offsets;关态 offset 恒
            # slot*stride=签名恒定,memo 行为逐位不变。
            local_pack_compact_offset_signature = tuple(
                int(state.compact_offset_tokens[int(slot_by_row[row])])
                if bool(use_compact_by_row[row])
                and 0 <= int(slot_by_row[row]) < len(state.compact_offset_tokens)
                else -1
                for row in range(batch_size)
            )
            cache_hit = (
                controller._step_local_pack_compact_token == _sit
                and controller._step_local_pack_compact_layer_index == int(state.layer_index)
                and controller._step_local_pack_compact_meta_epoch == int(state.compact_meta_epoch)
                and controller._step_local_pack_compact_batch_size == int(batch_size)
                and controller._step_local_pack_compact_stride_blocks == int(compact_stride_blocks)
                and controller._step_local_pack_compact_slot_signature == local_pack_compact_slot_signature
                and (
                    controller._step_local_pack_compact_use_compact_signature
                    == local_pack_compact_use_compact_signature
                )
                and (
                    getattr(
                        controller,
                        "_step_local_pack_compact_offset_signature",
                        None,
                    )
                    == local_pack_compact_offset_signature
                )
            )
            if cache_hit:
                compact_kv_len_by_row = controller._step_local_pack_compact_kv_len_by_row
                compact_offsets = controller._step_local_pack_compact_offsets_by_row
                compact_kv_len_max = int(controller._step_local_pack_compact_kv_len_max)
            else:
                local_pack_compact_kv_len_by_row: List[int] = [0] * batch_size
                local_pack_compact_offsets_by_row: List[int] = [0] * batch_size
                compact_kv_len_max = 0
                compact_kv_len_list = state.compact_kv_len
                compact_kv_len_len = len(compact_kv_len_list)
                compact_offset_tokens_len = len(state.compact_offset_tokens)
                for row in range(batch_size):
                    if not bool(use_compact_by_row[row]):
                        continue
                    slot = int(slot_by_row[row])
                    if slot < 0 or slot >= compact_offset_tokens_len:
                        raise RuntimeError(f"compact slot missing for row={row} slot={slot}")
                    kv_len = int(compact_kv_len_list[slot]) if slot < compact_kv_len_len else 0
                    if kv_len <= 0:
                        req_id_err = req_ids[row] if row < len(req_ids) else f"batch:{row}"
                        raise RuntimeError(
                            "local-pack compact row not ready: "
                            f"row={row} req={req_id_err} slot={slot} layer={int(state.layer_index)}"
                        )
                    local_pack_compact_kv_len_by_row[row] = int(kv_len)
                    # [DUAL-GEN-L1] offset 从 state 权威载体读(历史硬编码
                    # slot*stride 在双代切换后会读错半区);gen0 下逐位同值。
                    _offset_tokens = int(state.compact_offset_tokens[slot])
                    _stride_tokens_slot = int(state.compact_capacity[slot])
                    if _stride_tokens_slot <= 0 or (
                        (_offset_tokens * compact_stride_blocks) % _stride_tokens_slot != 0
                    ):
                        raise RuntimeError(
                            "local-pack compact offset not page aligned: "
                            f"slot={slot} offset_tokens={_offset_tokens} "
                            f"stride_tokens={_stride_tokens_slot}"
                        )
                    local_pack_compact_offsets_by_row[row] = (
                        _offset_tokens * compact_stride_blocks // _stride_tokens_slot
                    )
                    compact_kv_len_max = max(compact_kv_len_max, int(kv_len))
                compact_kv_len_by_row = tuple(local_pack_compact_kv_len_by_row)
                compact_offsets = tuple(local_pack_compact_offsets_by_row)
                controller._step_local_pack_compact_token = _sit
                controller._step_local_pack_compact_layer_index = int(state.layer_index)
                controller._step_local_pack_compact_meta_epoch = int(state.compact_meta_epoch)
                controller._step_local_pack_compact_batch_size = int(batch_size)
                controller._step_local_pack_compact_stride_blocks = int(compact_stride_blocks)
                controller._step_local_pack_compact_slot_signature = local_pack_compact_slot_signature
                controller._step_local_pack_compact_offset_signature = local_pack_compact_offset_signature
                controller._step_local_pack_compact_use_compact_signature = (
                    local_pack_compact_use_compact_signature
                )
                controller._step_local_pack_compact_kv_len_by_row = compact_kv_len_by_row
                controller._step_local_pack_compact_offsets_by_row = compact_offsets
                controller._step_local_pack_compact_kv_len_max = int(compact_kv_len_max)
            runtime_compact_kv_len_max = int(compact_kv_len_max)
        elif not meta_source_is_bound:
            if compact_kv_len_by_row is None:
                compact_kv_len_by_row = [0] * batch_size
            compact_kv_len_max = 0
            compact_kv_len_list = state.compact_kv_len
            compact_kv_len_len = len(compact_kv_len_list)
            for row in range(batch_size):
                compact_kv_len_by_row[row] = 0
                if not bool(use_compact_by_row[row]):
                    continue
                slot = int(slot_by_row[row])
                kv_len = int(compact_kv_len_list[slot]) if 0 <= slot < compact_kv_len_len else 0
                compact_kv_len_by_row[row] = kv_len
                compact_kv_len_max = max(compact_kv_len_max, kv_len)
            runtime_compact_kv_len_max = int(compact_kv_len_max)
        if meta_source == _META_SOURCE_LOCAL_PACK:
            req_meta_i32, req_meta_i64 = _pack_meta_from_local_plan(
                controller=controller,
                state=state,
                step_context=step_context,
                step_envelope=step_envelope_v2,
                use_fast_pack=bool(use_fast_pack),
                # local pack 必须消费当步布局，禁止复用 step_cache 的 compact 布局。
                use_step_cache_pack_cached=False,
                has_decode_row=bool(has_decode_row),
                has_prefill_row=bool(has_prefill_row),
                is_prefill_by_row=is_prefill_by_row,
                batch_size=int(batch_size),
                block_size=int(block_size),
                num_heads=int(num_heads),
                device=device,
                seqused_k=seqused_k,
                cu_seqlens_q=cu_seqlens_q,
                needs_logits_by_row=group_needs_logits,
                use_compact_by_row=use_compact_by_row,
                logf_producer_by_row=logf_producer_by_row,
                logits_last_n_by_row=group_logits_last_n,
                logits_spec_by_row=(
                    logits_spec_by_row if logits_spec_by_row is not None else []
                ),
                log_f_denom_ptr_by_row=(
                    log_f_denom_ptr_by_row if log_f_denom_ptr_by_row is not None else []
                ),
                compact_kv_len_by_row=(
                    compact_kv_len_by_row if compact_kv_len_by_row is not None else [0] * batch_size
                ),
                compact_offsets=compact_offsets,
                slot_by_row=slot_by_row,
                row_mode_signature=row_mode_signature,
                logits_rows_cached=logits_rows_cached,
                logits_rows_gt1_cached=logits_rows_gt1_cached,
                refresh_layout_for_log_f_capture=refresh_layout_for_log_f_capture,
                slot_in_chunk_for_log_f_capture=int(slot_in_chunk_for_log_f_capture),
                q_len_by_row=q_len_by_row,
                context_kv_len_by_row=context_kv_len_by_row,
            )
            if (
                req_meta_i32.data_ptr() == bound_req_meta_i32.data_ptr()
                or req_meta_i64.data_ptr() == bound_req_meta_i64.data_ptr()
            ):
                epoch_value = int(step_context.epoch) if step_context is not None else -1
                raise RuntimeError(
                    "local-pack contract violated: local req_meta aliases bound global req_meta; "
                    f"epoch={epoch_value} layer={int(state.layer_index)} cache_key={int(cache_key)}"
                )
        elif meta_source_is_bound:
            # 稳态 decode 保持复用 step-bound meta，避免热路径额外 pack 开销。
            req_meta_i32 = bound_req_meta_i32
            req_meta_i64 = bound_req_meta_i64
        else:
            req_meta_i32, req_meta_i64, _ = _pack_dispatch_meta(
                controller=controller,
                state=state,
                step_context=step_context,
                step_envelope=step_envelope_v2,
                is_decode=is_decode,
                use_fast_pack=bool(use_fast_pack),
                use_step_cache_pack_cached=bool(use_step_cache_pack_cached),
                has_decode_row=bool(has_decode_row),
                has_prefill_row=bool(has_prefill_row),
                has_compact_row=bool(runtime_has_compact_row),
                refresh_slots=refresh_slots,
                is_prefill_by_row=is_prefill_by_row,
                needs_logits_by_row=group_needs_logits,
                use_compact_by_row=use_compact_by_row,
                logf_producer_by_row=logf_producer_by_row,
                logits_last_n_by_row=group_logits_last_n,
                logits_spec_by_row=(
                    logits_spec_by_row if logits_spec_by_row is not None else []
                ),
                log_f_denom_ptr_by_row=(
                    log_f_denom_ptr_by_row if log_f_denom_ptr_by_row is not None else []
                ),
                compact_kv_len_by_row=(
                    compact_kv_len_by_row if compact_kv_len_by_row is not None else [0] * batch_size
                ),
                compact_offsets=compact_offsets,
                slot_by_row=slot_by_row,
                row_mode_signature=row_mode_signature,
                logits_rows_cached=logits_rows_cached,
                logits_rows_gt1_cached=logits_rows_gt1_cached,
                refresh_layout_for_log_f_capture=refresh_layout_for_log_f_capture,
                slot_in_chunk_for_log_f_capture=int(slot_in_chunk_for_log_f_capture),
                q_len_by_row=q_len_by_row,
                context_kv_len_by_row=context_kv_len_by_row,
                batch_size=int(batch_size),
                block_size=int(block_size),
                num_heads=int(num_heads),
                device=device,
                key_cache=k,
                seqused_k=seqused_k,
                cu_seqlens_q=cu_seqlens_q,
                dense_log_f_has_last_n_gt1=bool(dense_log_f_has_last_n_gt1),
            )
    if _profile_enabled:
        _profile_now_ns = _submit_profile_time_ns()
        _record_submit_profile(
            "dispatch.pack_meta", _profile_layer_index, _profile_now_ns - _profile_segment_start_ns
        )
        _profile_segment_start_ns = _profile_now_ns

    # hint 同源对齐：hint 的来源必须与 req_meta 来源一致，避免“meta/hint 混源”。
    if meta_source_is_bound:
        hint_has_log_f = bool(bound_hint_has_log_f)
        hint_log_f_eq1 = bool(bound_hint_log_f_eq1)
        hint_log_f_gt1 = bool(bound_hint_log_f_gt1)
        hint_all_compact = bool(bound_all_compact)
        hint_compact_kv_len_max = int(bound_compact_kv_len_max)
    else:
        hint_log_f_eq1 = bool(dense_log_f_has_last_n_eq1)
        hint_log_f_gt1 = bool(dense_log_f_has_last_n_gt1)
        hint_has_log_f = bool(hint_log_f_eq1 or hint_log_f_gt1)
        hint_all_compact = bool(runtime_all_compact_rows)
        hint_compact_kv_len_max = int(runtime_compact_kv_len_max)

    # Step 5: Call unified kernel
    # 仅传递本批有效行，避免冗余数据
    # 预计算 hints，避免 kernel 内部 GPU-CPU 同步
    # - has_log_f: 任一 op 需要 log_f（dense capture）
    # - compact_only: 所有 op 都是 compact 且无 logits/log_f
    # - compact_kv_len_max: compact-only 时最大 compact_kv_len（用于小网格 2D/3D 自适应）
    #
    # 【单源化】hint 直接从 dispatcher 当层持有的行语义变量派生，
    # 而非从 plan_stats/bound_meta 读取。这保证 hint 与实际 req_meta 一致：
    # - logf_producer_by_row / use_compact_by_row / logits_last_n_by_row
    #   是 StepAuthority 单源输出，也是 req_meta 编码的数据来源（local pack 或 bound meta）。
    # - 消除 plan_stats.dense_log_f_has_last_n_* 滞后导致的 hint 与 req_meta flags 分裂。
    _hint_has_log_f = hint_has_log_f
    _hint_has_logits = False
    _hint_log_f_has_last_n_eq1 = hint_log_f_eq1 if _hint_has_log_f else None
    _hint_log_f_has_last_n_gt1 = hint_log_f_gt1 if _hint_has_log_f else None
    _hint_all_compact = bool(hint_all_compact)
    _hint_compact_only = _hint_all_compact and (not _hint_has_log_f)
    _hint_compact_kv_len_max = hint_compact_kv_len_max if _hint_compact_only else None
    _hint_alpha_log_f: Optional[float] = None
    _hint_sink_tokens: Optional[int] = None
    if _hint_has_log_f:
        if controller is None:
            raise RuntimeError("log_f launch requires controller semantic snapshot")
        semantic_snapshot = _cached_semantic_snapshot
        _hint_alpha_log_f = float(semantic_snapshot.alpha_log_f)
        if not math.isfinite(_hint_alpha_log_f):
            raise ValueError(f"Invalid alpha_log_f from snapshot: {_hint_alpha_log_f}")
        _hint_sink_tokens = validate_sink_tokens(int(semantic_snapshot.sink_tokens))
    log_f_out_fp32 = _LOGF_OUT_FP32_CACHED

    meta_has_compact_row = bool(bound_has_compact_row if meta_source_is_bound else runtime_has_compact_row)
    if batch_size > 0 and (
        _validate_meta_contract_enabled()
        or ((_VALIDATE_COMPACT_META_CACHED if not _DYNAMIC_ENV
             else os.environ.get("VLLM_SPARSE_VALIDATE_COMPACT_META", "0") == "1")
            and meta_has_compact_row)
    ):
        _validate_dispatch_meta_contract(
            batch_size=batch_size,
            has_compact_row=meta_has_compact_row,
            req_meta_i32=req_meta_i32,
            req_meta_i64=req_meta_i64,
            req_ids=req_ids,
            slot_by_row=slot_by_row,
            block_table=block_table,
            block_size=block_size,
            key_compact=key_compact,
            state=state,
        )
    if _profile_enabled:
        _profile_now_ns = _submit_profile_time_ns()
        _record_submit_profile(
            "dispatch.hints_validate", _profile_layer_index, _profile_now_ns - _profile_segment_start_ns
        )
        _profile_segment_start_ns = _profile_now_ns

    with _range("sparse.dispatch.kernel"):
        flash_attn_score_dump_fwd_unified(
            q,
            out,
            k,
            v,
            block_table,
            key_compact,
            value_compact,
            logical_token_indices,
            req_meta_i32,
            req_meta_i64,
            cu_seqlens_q,
            max_seqlen_q,
            seqused_k,
            max_seqlen_k,
            softmax_scale,
            softcap,
            alibi_slopes,
            window_size,
            k_descale,
            v_descale,
            _hint_has_logits=_hint_has_logits,
            _hint_has_log_f=_hint_has_log_f,
            _hint_log_f_has_last_n_eq1=_hint_log_f_has_last_n_eq1,
            _hint_log_f_has_last_n_gt1=_hint_log_f_has_last_n_gt1,
            _hint_log_f_out_fp32=log_f_out_fp32,
            _hint_alpha_log_f=_hint_alpha_log_f,
            _hint_sink_tokens=_hint_sink_tokens,
            _hint_compact_only=_hint_compact_only,
            _hint_strict_no_sync_fallback=True,
            _hint_compact_kv_len_max=_hint_compact_kv_len_max,
            _num_seqs=batch_size,
        )
    if _profile_enabled:
        _profile_now_ns = _submit_profile_time_ns()
        _record_submit_profile(
            "dispatch.kernel_submit", _profile_layer_index, _profile_now_ns - _profile_segment_start_ns
        )
        _profile_segment_start_ns = _profile_now_ns
    # prefill：kernel 完成后再做 key_norms 增量预热，确保 KV 已写入。
    if controller is not None and has_prefill_row:
        _prefill_update_key_norms(
            controller=controller,
            state=state,
            key_cache=k,
            block_table=block_table,
            slot_by_row=slot_by_row,
            is_prefill_by_row=is_prefill_by_row,
            chunk_len_by_row=q_len_by_row,
            context_kv_len_by_row=context_kv_len_by_row,
            num_kv_heads=num_kv_heads,
        )
    if _profile_enabled:
        _profile_now_ns = _submit_profile_time_ns()
        _record_submit_profile(
            "dispatch.prefill_key_norms", _profile_layer_index, _profile_now_ns - _profile_segment_start_ns
        )
        _record_submit_profile(
            "dispatch.total", _profile_layer_index, _profile_now_ns - _profile_total_start_ns
        )

    return (
        refresh_slots,
        bootstrap_slots_set,
        profile_enabled,
        profile_stats,
        refresh_slot_list_for_payload,
        refresh_layout_for_log_f_capture,
    )
