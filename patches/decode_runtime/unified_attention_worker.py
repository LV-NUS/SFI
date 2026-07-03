from __future__ import annotations

import logging
import os
from typing import Dict, Optional, Tuple

import torch

from patches.patch_installer import (
    _get_global_controller,
    _ORIGINAL_UNIFIED_ATTENTION,
    _cache_full_cudagraph_replay_payload_refs,
)
from patches.runtime_deps import require_runtime_dep
from patches.sparse_constants import (
    _CAPTURE_CHUNK,
    _DYNAMIC_ENV,
    _FORCE_DENSE_CACHED,
)
from patches.sparse_types import SelectorBatchPayload
from patches.sparse_utils import (
    _make_selector_fast_signature,
)
from patches.decode_runtime.submit_profile import (
    record_sparse_submit_profile as _record_submit_profile,
    sparse_submit_profile_enabled as _submit_profile_enabled,
    sparse_submit_profile_time_ns as _submit_profile_time_ns,
)

_log = logging.getLogger(__name__)

_execute_refresh_post_kernel = require_runtime_dep("_execute_refresh_post_kernel")
_prepare_prefill_capture_payload = require_runtime_dep("_prepare_prefill_capture_payload")
_run_unified_attention_dispatcher = require_runtime_dep("_run_unified_attention_dispatcher")


def patched_unified_attention_impl(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    max_seqlen_q,
    seqused_k,
    max_seqlen_k,
    softmax_scale,
    causal,
    window_size,
    block_table,
    softcap,
    q_descale,
    k_descale,
    v_descale,
    alibi_slopes=None,
):
    # 全局强制 dense：直接走原生 unified_attention，绕过所有稀疏调度逻辑。
    force_dense = (os.environ.get("VLLM_SPARSE_FORCE_DENSE") == "1") if _DYNAMIC_ENV else _FORCE_DENSE_CACHED
    controller = _get_global_controller()
    _profile_enabled = _submit_profile_enabled()
    _profile_total_start_ns = _submit_profile_time_ns() if _profile_enabled else 0
    _profile_segment_start_ns = _profile_total_start_ns

    # Step-level gate: selector enable was decided in prepare_step_context.
    # force_dense overrides at runtime via env var.
    if force_dense or not controller.config.enabled:
        return _ORIGINAL_UNIFIED_ATTENTION(
            q, k, v, out, cu_seqlens_q, max_seqlen_q, seqused_k,
            max_seqlen_k, softmax_scale, causal, window_size,
            block_table, softcap, q_descale, k_descale, v_descale, alibi_slopes,
        )

    step_authority = controller.step_authority
    if step_authority is None:
        raise RuntimeError(
            "unified attention requires step_authority single-source metadata"
        )

    has_prefill = not bool(step_authority.is_decode_only)

    # Sanity checks for unified attention metadata: these should always hold
    if cu_seqlens_q.dim() != 1:
        raise ValueError(f"cu_seqlens_q must be 1D prefix-sum, got dim={cu_seqlens_q.dim()}")

    num_seqs = cu_seqlens_q.shape[0] - 1
    if num_seqs <= 0:
        raise ValueError(f"cu_seqlens_q must describe at least one sequence, got {cu_seqlens_q.shape}")

    if seqused_k.dim() == 0:
        seqused_k = seqused_k.view(1)
    if seqused_k.shape[0] != num_seqs:
        raise ValueError(
            f"seqused_k length {seqused_k.shape[0]} does not match num_seqs {num_seqs}"
        )
    if block_table.shape[0] != num_seqs:
        raise ValueError(
            f"block_table batch dimension {block_table.shape[0]} does not match num_seqs {num_seqs}"
        )

    # cu_seqlens_q[-1].item() 会触发 DtoH；该一致性校验只在 debug 时启用。

    cache_key = k.data_ptr()
    state = controller.get_state(
        cache_key,
        q.shape[1],
        k.shape[2],
        q.device,
        head_dim=q.shape[2],
        kv_cache_dtype=k.dtype,
    )
    _cache_full_cudagraph_replay_payload_refs(
        state=state,
        key_cache=k,
        value_cache=v,
        block_table=block_table,
        q=q,
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=seqused_k,
        softmax_scale=softmax_scale,
        softcap=softcap if softcap is not None else 0.0,
        window_size=window_size,
        alibi_slopes=alibi_slopes,
        k_descale=k_descale,
    )
    # batch size equals number of sequences described by cu_seqlens_q
    batch = num_seqs
    # 避免 seqused_k.max().item() 触发 DtoH 同步：直接使用 vLLM 传入的 max_seqlen_k（Python int）。
    seq_len_max = max_seqlen_k
    controller.begin_step_if_needed(cache_key, seq_len_max)
    if controller.layer_dispatch_epoch != controller.step_context_epoch:
        controller.layer_dispatch_epoch = controller.step_context_epoch
        controller.layer_dispatch_cursor = 0
        controller.layer_dispatch_layer_count = len(controller.layer_cache_keys)
    controller.layer_dispatch_cursor += 1
    is_last_layer = (
        controller.layer_dispatch_layer_count > 0
        and controller.layer_dispatch_cursor == controller.layer_dispatch_layer_count
    )

    # 单真源硬切：执行入口只认 controller.step_context，不再依赖 LayerState 绑定缓存。
    step_ctx = controller.step_context
    if step_ctx is None:
        raise RuntimeError("StepContext missing before prefill plan")
    request_ids = step_ctx.req_ids
    step_envelope = step_ctx.step_envelope_v2
    # Step-invariant validation: step_ctx/step_envelope are the SAME objects
    # across all layers of a step, so re-validating every layer is redundant.
    # Gate to the first layer of the step (cursor == 1 right after the
    # unconditional +1 above). The bindings above stay per-layer because they
    # are consumed downstream every layer (request_ids -> dispatcher;
    # step_envelope -> align_slots_from_snapshot).
    if controller.layer_dispatch_cursor == 1:
        if len(request_ids) != batch:
            raise RuntimeError(
                "StepContext req_ids length mismatch before prefill plan "
                f"(len={len(request_ids)}, batch={batch})"
            )
        step_handle_id = int(step_ctx.step_handle_id)
        step_handle_generation = int(step_ctx.step_handle_generation)
        if step_handle_id <= 0 or step_handle_generation <= 0:
            raise RuntimeError(
                "StepContext missing step handle identity before prefill plan "
                f"(handle_id={step_handle_id}, generation={step_handle_generation}, "
                f"layer_index={int(getattr(state, 'layer_index', -1))})"
            )
        if step_ctx.epoch != controller.step_context_epoch:
            raise RuntimeError(
                "StepContext epoch mismatch with controller before prefill plan "
                f"(ctx_epoch={step_ctx.epoch}, ctrl_epoch={controller.step_context_epoch}, "
                f"handle_id={step_handle_id}, generation={step_handle_generation})"
            )
        if step_envelope is None:
            raise RuntimeError("StepContext step_envelope_v2 missing before prefill plan")
        if step_envelope.req_ids != step_ctx.req_ids:
            raise RuntimeError("StepContext step_envelope_v2 req_ids mismatch before prefill plan")

    # slot 对齐仅依赖 step snapshot，避免每层读取 request->slot map 热路径。
    state.align_slots_from_snapshot(
        request_ids=step_ctx.req_ids,
        slot_by_row=step_envelope.slot_by_row,
        epoch=int(step_envelope.epoch),
    )

    capture_plan_by_req: Dict[str, int] = {}
    finalize_req_ids: Tuple[str, ...] = tuple()
    if has_prefill:
        capture_plan_by_req, finalize_req_ids = controller.get_step_prefill_plan_by_req(step_context=step_ctx)

    capture_plan_active_by_req: Dict[str, int] = {}
    if capture_plan_by_req:
        for rid, last_n_raw in capture_plan_by_req.items():
            last_n = int(last_n_raw or 0)
            if last_n > 0:
                capture_plan_active_by_req[str(rid)] = int(last_n)

    capture_plan_active_by_slot: Dict[int, int] = {}
    if capture_plan_active_by_req:
        for rid, last_n in capture_plan_active_by_req.items():
            slot_idx = state.request_id_to_slot.get(rid, -1)
            if slot_idx < 0:
                raise RuntimeError(f"prefill capture plan: req_id={rid} missing LayerState slot")
            capture_plan_active_by_slot[int(slot_idx)] = int(last_n)

    # step_ctx 已在 prefill plan 阶段获取并校验

    # chunk_query_lengths 仅用于 prefill capture 的 chunk 长度视图/调试；构建时禁止任何 DtoH 同步。
    # Step-level cache: 同一 step 内所有层共享相同 cu_seqlens_q，只需计算一次。
    chunk_query_lengths: Optional[torch.Tensor] = None
    if has_prefill:
        _cql_epoch = controller._step_cql_epoch
        _cql_handle_id = int(controller._step_cql_handle_id)
        _cql_handle_generation = int(controller._step_cql_handle_generation)
        _step_handle_id = int(step_ctx.step_handle_id)
        _step_handle_generation = int(step_ctx.step_handle_generation)
        if (
            _cql_epoch == controller.step_context_epoch
            and _cql_handle_id == _step_handle_id
            and _cql_handle_generation == _step_handle_generation
        ):
            chunk_query_lengths = controller._step_cql_tensor
        else:
            chunk_query_lengths = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).to(device=q.device, dtype=torch.long)
            controller._step_cql_epoch = controller.step_context_epoch
            controller._step_cql_handle_id = _step_handle_id
            controller._step_cql_handle_generation = _step_handle_generation
            controller._step_cql_tensor = chunk_query_lengths

    controller.prepare_step_logits_buffers(
        state=state,
        step_context=step_ctx,
        capture_plan_by_req=capture_plan_active_by_req if capture_plan_active_by_req else None,
        seqused_k=seqused_k,
        max_seqlen_k=max_seqlen_k,
        block_size=int(k.shape[1]) if k is not None and len(k.shape) >= 2 else 0,
        num_heads=q.shape[1],
        device=q.device,
    )
    _profile_layer_index = int(getattr(state, "layer_index", -1))
    if _profile_enabled:
        _profile_now_ns = _submit_profile_time_ns()
        _record_submit_profile(
            "entry.pre_dispatch", _profile_layer_index, _profile_now_ns - _profile_segment_start_ns
        )
        _profile_segment_start_ns = _profile_now_ns

    # Use unified dispatcher for decode phase
    # This replaces the old dense/sparse path branching
    (
        refresh_slots,
        bootstrap_slots_set,
        profile_enabled,
        profile_stats,
        refresh_slot_list_for_payload,
        refresh_layout_for_payload,
    ) = _run_unified_attention_dispatcher(
        controller=controller,
        state=state,
        cache_key=cache_key,
        request_ids=request_ids,
        q=q,
        k=k,
        v=v,
        out=out,
        block_table=block_table,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=max_seqlen_q,
        seqused_k=seqused_k,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=softmax_scale,
        softcap=softcap if softcap is not None else 0.0,
        alibi_slopes=alibi_slopes,
        window_size=window_size,
        k_descale=k_descale,
        v_descale=v_descale,
        step_context=step_ctx,
        prefill_capture_plan=capture_plan_active_by_slot if capture_plan_active_by_slot else None,
        chunk_query_lengths=chunk_query_lengths,
    )
    if _profile_enabled:
        _profile_now_ns = _submit_profile_time_ns()
        _record_submit_profile(
            "entry.dispatch", _profile_layer_index, _profile_now_ns - _profile_segment_start_ns
        )
        _profile_segment_start_ns = _profile_now_ns

    _execute_refresh_post_kernel(
        controller=controller,
        state=state,
        q=q,
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=seqused_k,
        key_cache=k,
        value_cache=v,
        block_table=block_table,
        refresh_slots=refresh_slots,
        bootstrap_slots_set=bootstrap_slots_set,
        refresh_slot_list=refresh_slot_list_for_payload,
        refresh_layout=refresh_layout_for_payload,
        step_context=step_ctx,
        softmax_scale=float(softmax_scale),
        softcap=float(softcap if softcap is not None else 0.0),
        window_size=window_size,
        alibi_slopes=alibi_slopes,
        k_descale=k_descale,
        profile_enabled=profile_enabled,
        profile_stats=profile_stats,
    )
    if _profile_enabled:
        _profile_now_ns = _submit_profile_time_ns()
        _record_submit_profile(
            "entry.post_kernel", _profile_layer_index, _profile_now_ns - _profile_segment_start_ns
        )
        _profile_segment_start_ns = _profile_now_ns

    if capture_plan_active_by_slot:
        payload = _prepare_prefill_capture_payload(
            controller=controller,
            cache_key=cache_key,
            state=state,
            step_context=step_ctx,
            q=q,
            cu_seqlens_q=cu_seqlens_q,
            seqused_k=seqused_k,
            chunk_query_lengths=chunk_query_lengths,
            num_heads=q.shape[1],
            device=q.device,
            capture_plan=capture_plan_active_by_slot,
        )
        if payload is None:
            raise RuntimeError(
                "prefill capture payload build failed; refuse silent skip "
                f"(cache_key={int(cache_key)}, epoch={step_ctx.epoch})"
            )
        (
            capture_scores,
            log_f_denoms,
            kv_lengths,
            seq_lens_batch,
            seq_lens_batch_i32,
            chunk_lengths,
            slot_list,
            slot_tensor,
            slot_tensor_i32,
            slot_tensor_cpu,
            row_list_cpu,
            row_tensor,
            row_tensor_i32,
            kv_len_per_row_i32,
            seq_lens_cpu,
            seq_lens_tensor_cpu,
            layer_index_in_chunk,
        ) = payload
        row_limit = int(block_table.shape[0])
        if any((int(row) < 0 or int(row) >= row_limit) for row in row_list_cpu):
            raise RuntimeError("prefill capture requires valid batch row indices for all slots")
        if not controller.layer_cache_keys:
            raise RuntimeError("prefill capture requires layer_cache_keys")
        controller._enqueue_prefill_capture(
            SelectorBatchPayload(
                cache_key=cache_key,
                state=state,
                capture_scores=capture_scores,
                log_f_denoms=log_f_denoms,
                kv_lengths=kv_lengths,
                slot_list=slot_list,
                row_list=row_list_cpu,
                key_cache=k,
                value_cache=v,
                block_table=block_table,
                bootstrap_slots=set(slot_list),
                layer_index=int(layer_index_in_chunk),
                seq_lens_batch=seq_lens_batch,
                seq_lens_batch_i32=seq_lens_batch_i32,
                slot_tensor=slot_tensor,
                slot_tensor_i32=slot_tensor_i32,
                slot_tensor_cpu=slot_tensor_cpu,
                row_tensor=row_tensor,
                row_tensor_i32=row_tensor_i32,
                kv_len_per_row_i32=kv_len_per_row_i32,
                seq_lens_cpu=seq_lens_cpu if seq_lens_cpu is not None else tuple(),
                seq_lens_tensor_cpu=seq_lens_tensor_cpu,
                fast_signature=_make_selector_fast_signature(
                    capture_scores=capture_scores,
                    log_f_denoms=log_f_denoms,
                    kv_lengths=kv_lengths,
                    block_table=block_table,
                ),
            ),
        )
        # prefill 的 request-facing bootstrap 提交统一延后到 flush 成功后的 finalize boundary；
        # 这里仅负责 enqueue transport payload，不直接推进 pending/done 状态。

    # chunk-batched flush：slot_in_chunk==CHUNK-1 或最后一层（最后 chunk 可能不满）
    layer_index_global = int(state.layer_index)
    if layer_index_global < 0:
        layer_index_global = int(controller.layer_index_by_cache_key.get(cache_key, -1))
    if layer_index_global >= 0:
        chunk_id, buf_id, slot_in_chunk = controller._map_global_layer_to_capture_slot(layer_index_global)
        if (int(slot_in_chunk) == int(_CAPTURE_CHUNK) - 1) or bool(is_last_layer):
            chunk_size = (int(slot_in_chunk) + 1) if bool(is_last_layer) else int(_CAPTURE_CHUNK)
            controller._flush_prefill_batches(
                buf_id=int(buf_id),
                chunk_id=int(chunk_id),
                chunk_size=int(chunk_size),
                is_last_layer=bool(is_last_layer),
            )

    if _profile_enabled:
        _profile_now_ns = _submit_profile_time_ns()
        _record_submit_profile(
            "entry.prefill_capture_flush", _profile_layer_index, _profile_now_ns - _profile_segment_start_ns
        )
        _record_submit_profile(
            "entry.total", _profile_layer_index, _profile_now_ns - _profile_total_start_ns
        )
    return None
