from __future__ import annotations

import logging
import os
from typing import List, Optional, Sequence, Tuple

_log = logging.getLogger(__name__)

from patches.step_decode_pipeline import reset_runtime_step_cursor_state
from patches.step_faults import StepFault
from patches.request_intent_ticket import PendingPolicy, mark_threshold_crossing
from patches.prefill_capture_meta_arena import CaptureArenaIntent
from triton_kernel.req_meta_flag_codec import validate_sink_tokens
from patches.decode_runtime.row_policy import resolve_decode_row_policy
from patches.refresh_runtime.flush_scheduler import normalize_refresh_slot_list
from patches.sparse_utils import (
    _align_up_int,
    _make_decode_plan_version,
    assert_cleanup_ledgers_drained_for_step_build,
    build_step_plan_signature,
)
from patches.sparse_constants import (
    _CAPTURE_CHUNK,
    _CAPTURE_KV_BUCKET_CACHED,
    _DYNAMIC_ENV,
    _LOGF_PRODUCER_ATTN,
    _LOGF_PRODUCER_NONE,
    _PREFILL_RELEASE_GRACE_STEPS,
    _ROW_MODE_COMPACT,
)
from patches.sparse_types import (
    StepContext,
    StepEnvelopeV2,
    StepMeta,
    StepRefreshMode,
    StepTicket,
)
from patches.step_authority import StepAuthority
from patches.fa3_native.install import (
    append_fa3_step_trace,
    build_fa3_step_trace_event,
    fa3_step_trace_enabled,
)
from patches.fa3_native.contracts import TargetSelectedScopeKey
from patches.fa3_native.snapshot_binding import SelectedScopeKey
from patches.fa3_native.scope_async import allocate_scope_wait_handle

_PERSISTENT_BATCH_ENABLED = os.environ.get("VLLM_SPARSE_PERSISTENT_BATCH", "0") == "1"

# === VLLM_SPARSE_PSC_BSKIP: steady-step pure-derivation (B) reuse gate ===
# Skip the 4 pure-derivation B blocks of prepare_step_context_impl on a
# no-change decode step, reuse cached B outputs, re-stamp currency into the
# reused dataclasses. All side-effecting A statements still run. Gate OFF by
# default; OFF => verbatim full build (these reads are a single bool each).
_PSC_BSKIP_ENABLED = os.environ.get("VLLM_SPARSE_PSC_BSKIP", "0") == "1"
_PSC_BSKIP_ASSERT = os.environ.get("VLLM_SPARSE_PSC_BSKIP_ASSERT", "0") == "1"

# === Phased env cache for OFF-by-default per-step diagnostic reads ===
# Production hot path reads the cached constant; PyTest / explicit dynamic
# runs (_DYNAMIC_ENV) read live os.environ at each site (see
# unified_attention_worker.py:57 for the canonical idiom).
_DEFER_BOOTSTRAP_PRODUCER_CACHED = (
    os.environ.get("VLLM_SPARSE_DEFER_BOOTSTRAP_PRODUCER", "0") == "1"
)
# SKIP keeps NO default to stay byte-exact with the original
# os.environ.get(...) == "1" (None on absence).
_SKIP_DECODE_PREFILL_RESERVE_CACHED = (
    os.environ.get("VLLM_SPARSE_SKIP_DECODE_PREFILL_RESERVE") == "1"
)


def _psc_bskip_restamp_authority(auth, *, epoch, handle_id, handle_generation,
                                 decode_plan_version, consume_key, consume_handle,
                                 target_key, wait_handle,
                                 context_kv_len_by_row, q_lens_by_row):
    # Re-stamp per-step currency + scope keys onto a REUSED StepAuthority
    # (dataclass slots=True, non-frozen). object.__setattr__ keeps it robust
    # to any future frozen flip. step_identity_token / step_fast_identity are
    # the __post_init__-derived fields and must be recomputed here.
    _sa = object.__setattr__
    _sa(auth, 'epoch', int(epoch))
    _sa(auth, 'step_handle_id', int(handle_id))
    _sa(auth, 'step_handle_generation', int(handle_generation))
    _sa(auth, 'decode_plan_version', int(decode_plan_version))
    _sa(auth, 'consume_selected_scope_key', consume_key)
    _sa(auth, 'consume_selected_scope_wait_handle', consume_handle)
    _sa(auth, 'target_selected_scope_key', target_key)
    _sa(auth, 'selected_scope_wait_handle', wait_handle)
    _sa(auth, 'context_kv_len_by_row', context_kv_len_by_row)
    _sa(auth, 'q_lens_by_row', q_lens_by_row)
    _tok = (int(epoch) * 1_000_000_000
            + int(handle_id) * 1_000_000
            + int(handle_generation))
    _sa(auth, 'step_identity_token', _tok)
    _sa(auth, 'step_fast_identity', (
        int(epoch), int(handle_id), int(handle_generation), int(_tok),
        int(auth.req_set_hash), int(auth.row_phase_hash),
    ))
    return auth


def _psc_bskip_restamp_envelope(env, *, epoch, handle_id, handle_generation,
                                interval_merge_policy, req_ids_tuple, slot_by_row,
                                row_mode_signature, refresh_reqs, refresh_rows,
                                logf_producer_by_row, needs_logits_by_row,
                                logits_last_n_by_row):
    # StepEnvelopeV2 is frozen=True, slots=True. object.__setattr__ bypasses
    # the frozen guard (it is never hashed/used as a dict key — verified).
    # cache_signature embeds the epoch, so it is rebuilt with current epoch.
    _sa = object.__setattr__
    _sa(env, 'epoch', int(epoch))
    _sa(env, 'handle_id', int(handle_id))
    _sa(env, 'handle_generation', int(handle_generation))
    _sa(env, 'cache_signature', (
        int(epoch), interval_merge_policy, req_ids_tuple, slot_by_row,
        row_mode_signature, refresh_reqs, refresh_rows, logf_producer_by_row,
        needs_logits_by_row, logits_last_n_by_row,
    ))
    return env


def _psc_bskip_restamp_context(ctx, *, epoch, handle_id, handle_generation,
                               step_handle, num_actual_tokens, q_start_loc, q_lens,
                               seq_lens, max_query_len, max_seq_len, req_id_to_index,
                               prompt_lens, num_computed_tokens, step_envelope_v2):
    # StepContext is slots=True, non-frozen. Re-stamp the per-step fields that
    # advance even on a steady step (currency + monotonic seq/q geometry) and
    # recompute the __post_init__ identity token.
    _sa = object.__setattr__
    _sa(ctx, 'epoch', int(epoch))
    _sa(ctx, 'step_handle_id', int(handle_id))
    _sa(ctx, 'step_handle_generation', int(handle_generation))
    _sa(ctx, 'step_handle', step_handle)
    _sa(ctx, 'num_actual_tokens', int(num_actual_tokens))
    _sa(ctx, 'q_start_loc', q_start_loc)
    _sa(ctx, 'q_lens', q_lens)
    _sa(ctx, 'seq_lens', seq_lens)
    _sa(ctx, 'max_query_len', int(max_query_len))
    _sa(ctx, 'max_seq_len', int(max_seq_len))
    _sa(ctx, 'req_id_to_index', req_id_to_index)
    _sa(ctx, 'prompt_lens', prompt_lens)
    _sa(ctx, 'num_computed_tokens', num_computed_tokens)
    _sa(ctx, 'step_envelope_v2', step_envelope_v2)
    _sa(ctx, 'step_identity_token', (
        int(epoch) * 1_000_000_000
        + int(handle_id) * 1_000_000
        + int(handle_generation)))
    return ctx
# === end VLLM_SPARSE_PSC_BSKIP machinery ===
StepIdentity = Tuple[int, int, int, int]
_STEP_IDENTITY_EMPTY: StepIdentity = (-1, -1, -1, -1)
_HASH_MASK_I64 = 0x7FFFFFFFFFFFFFFF




def _compute_step_request_hashes(
    *,
    req_ids_tuple: Tuple[str, ...],
    is_prefill_by_row: Sequence[bool],
    num_reqs: int,
) -> Tuple[int, int]:
    """Build order-insensitive req-set hash + order-sensitive row-phase hash."""
    req_set_hash = hash(req_ids_tuple) & _HASH_MASK_I64
    is_prefill_tuple = tuple(bool(is_prefill_by_row[i]) for i in range(num_reqs))
    row_phase_hash = hash((req_ids_tuple, is_prefill_tuple)) & _HASH_MASK_I64
    return int(req_set_hash), int(row_phase_hash)


def _parse_compact_consume_delay_steps(raw: str) -> int:
    try:
        return max(0, int(raw))
    except ValueError as exc:
        raise RuntimeError(
            "VLLM_SPARSE_ATTRIB_COMPACT_CONSUME_DELAY_STEPS "
            f"must be an integer, got {raw!r}"
        ) from exc


_COMPACT_CONSUME_DELAY_STEPS_CACHED = _parse_compact_consume_delay_steps(
    os.environ.get("VLLM_SPARSE_ATTRIB_COMPACT_CONSUME_DELAY_STEPS", "0")
)


def _attrib_compact_consume_delay_steps() -> int:
    if _DYNAMIC_ENV:
        return _parse_compact_consume_delay_steps(
            os.environ.get("VLLM_SPARSE_ATTRIB_COMPACT_CONSUME_DELAY_STEPS", "0")
        )
    return _COMPACT_CONSUME_DELAY_STEPS_CACHED


def _bootstrap_done_for_row_policy(
    *,
    config: object,
    tracking: object,
    compact_consume_delay_steps: int,
) -> bool:
    boot = bool(getattr(tracking, "bootstrap_done", False))
    if not boot or compact_consume_delay_steps <= 0:
        return boot
    if not bool(getattr(config, "one_shot_bootstrap_only", False)):
        return boot
    raw_decode_step = getattr(tracking, "decode_step", -1)
    decode_step = int(raw_decode_step) if raw_decode_step is not None else -1
    # 诊断开关：只延迟 row-policy 视角的 compact consume，不回滚 request
    # 级 bootstrap_done。compact_ready / producer 状态仍保持真实，便于归因。
    if 0 <= decode_step < compact_consume_delay_steps:
        return False
    return boot


def _publish_bootstrap_readiness_before_step_authority(
    self,
    *,
    req_ids_tuple: Tuple[str, ...],
) -> None:
    def _bridge_bootstrap_ids() -> Tuple[str, ...]:
        return tuple(
            str(rid)
            for rid in req_ids_tuple
            if self._request_can_bridge_bootstrap_decode(str(rid))
        )

    defer_bootstrap_producer = (
        (os.environ.get("VLLM_SPARSE_DEFER_BOOTSTRAP_PRODUCER", "0") == "1")
        if _DYNAMIC_ENV
        else _DEFER_BOOTSTRAP_PRODUCER_CACHED
    )
    bridge_bootstrap_ids = _bridge_bootstrap_ids()
    if defer_bootstrap_producer:
        self._launch_deferred_bootstrap_producer_jobs(
            epoch=self.step_context_epoch,
        )
        # A request can finish its deferred producer while another request still
        # needs bridge full-KV. Publish completed producers before the metadata
        # guard classifies rows; otherwise the completed request is neither
        # bridge-eligible nor compact-visible in this step.
        self._wait_and_publish_bootstrap_requests_for_graph_decode(
            epoch=self.step_context_epoch
        )
        bridge_bootstrap_ids = _bridge_bootstrap_ids()

    blocked_pending_before_bridge = tuple(
        str(rid)
        for rid in getattr(self, "_bootstrap_pending_request_ids", set())
        if str(rid) not in bridge_bootstrap_ids
        and not self._request_can_bridge_bootstrap_decode(str(rid))
    )
    if bridge_bootstrap_ids:
        self._mark_bridge_decode_metadata_accepted(
            req_ids=bridge_bootstrap_ids,
            epoch=self.step_context_epoch,
        )
        if not blocked_pending_before_bridge:
            if defer_bootstrap_producer:
                tail_launch_ids = []
                for rid in bridge_bootstrap_ids:
                    tracking = self.request_states.get(str(rid))
                    if tracking is None:
                        continue
                    bridge_max_tokens = int(
                        getattr(tracking, "bridge_max_tokens", 0) or 0
                    )
                    if bridge_max_tokens <= 0:
                        continue
                    bridge_token_count = int(
                        getattr(tracking, "bridge_token_count", 0) or 0
                    )
                    launch_threshold = int(bridge_max_tokens) - 1
                    if bridge_token_count >= launch_threshold:
                        tail_launch_ids.append(str(rid))
                if tail_launch_ids:
                    self._launch_deferred_bootstrap_producer_jobs(
                        epoch=self.step_context_epoch,
                        only_request_ids=tuple(tail_launch_ids),
                        allow_same_epoch=True,
                    )
            return
    if not self._bootstrap_pending_requires_global_wait():
        self._publish_ready_bootstrap_requests_at_step_boundary(
            epoch=self.step_context_epoch
        )
    else:
        # full-cudagraph one-shot async bootstrap：在 StepAuthority 构建前把
        # request-local finalize events 排进主流，随后可安全发布 bootstrap_done。
        self._wait_and_publish_bootstrap_requests_for_graph_decode(
            epoch=self.step_context_epoch
        )
        # step 边界先用轻量 event-query 提升真正 ready 的 bootstrap 请求。
        # 这样后续 StepAuthority 只消费严格语义的 bootstrap_done，
        # 不会把 bootstrap_pending 过早物化为 selected-ready。
        self._publish_ready_bootstrap_requests_at_step_boundary(
            epoch=self.step_context_epoch
        )


def _resolve_consume_selected_scope_binding(
    *,
    previous_step_authority: object | None,
    current_epoch: int,
    target_epoch: int,
) -> tuple[object | None, object | None]:
    if previous_step_authority is None:
        return None, None

    consume_key = getattr(previous_step_authority, "consume_selected_scope_key", None)
    consume_handle = getattr(
        previous_step_authority, "consume_selected_scope_wait_handle", None
    )

    if int(target_epoch) > int(current_epoch):
        previous_has_selected_consume = any(
            bool(v)
            for v in tuple(getattr(previous_step_authority, "use_compact_by_row", tuple()))
        )
        if not previous_has_selected_consume:
            return consume_key, consume_handle
        previous_publish_key = getattr(
            previous_step_authority, "target_selected_scope_key", None
        )
        consume_key = None
        if previous_publish_key is not None:
            consumer_step_id = getattr(previous_publish_key, "consumer_step_id", None)
            layer_group_id = getattr(previous_publish_key, "layer_group_id", None)
            if consumer_step_id is not None and layer_group_id is not None:
                consume_key = SelectedScopeKey(
                    consumer_step_id=int(consumer_step_id),
                    layer_group_id=int(layer_group_id),
                    chunk_id=0,
                )
        consume_handle = getattr(previous_step_authority, "selected_scope_wait_handle", None)
        return consume_key, consume_handle

    return consume_key, consume_handle


def consume_step_faults_impl(self) -> Tuple[StepFault, ...]:
    faults = tuple(getattr(self, "_step_faults", ()) or ())
    setattr(self, "_step_faults", [])
    return faults


def _ensure_step_fault_container(self) -> None:
    if getattr(self, "_step_faults", None) is None:
        setattr(self, "_step_faults", [])


class _StepAuthorityBuilderScratch:
    """Reusable CPU containers for prepare_step_context step-authority build."""

    __slots__ = (
        "is_prefill_by_row",
        "prefill_rows",
        "bootstrap_done",
        "row_mode_by_row",
        "logf_producer_by_row",
        "needs_logits_by_row",
        "logits_last_n_by_row",
        "decode_logf_capacity_by_row",
        "decode_logf_mask_by_row",
        "decode_logf_q_lens_by_row",
        "decode_logf_attn_rows",
        "refresh_row_mask",
    )

    def __init__(self) -> None:
        self.is_prefill_by_row: List[bool] = []
        self.prefill_rows: List[int] = []
        self.bootstrap_done: List[bool] = []
        self.row_mode_by_row: List[int] = []
        self.logf_producer_by_row: List[int] = []
        self.needs_logits_by_row: List[bool] = []
        self.logits_last_n_by_row: List[int] = []
        self.decode_logf_capacity_by_row: List[int] = []
        self.decode_logf_mask_by_row: List[int] = []
        self.decode_logf_q_lens_by_row: List[int] = []
        self.decode_logf_attn_rows: List[int] = []
        self.refresh_row_mask: List[bool] = []

    def reset(self, *, batch_size: int) -> None:
        self.is_prefill_by_row.clear()
        self.prefill_rows.clear()
        self.bootstrap_done.clear()
        self.row_mode_by_row.clear()
        self.logf_producer_by_row.clear()
        self.needs_logits_by_row.clear()
        self.logits_last_n_by_row.clear()
        self.decode_logf_capacity_by_row.clear()
        self.decode_logf_mask_by_row.clear()
        self.decode_logf_q_lens_by_row.clear()
        self.decode_logf_attn_rows.clear()
        mask = self.refresh_row_mask
        mask.clear()
        if batch_size > 0:
            mask.extend([False] * batch_size)


def _get_step_authority_builder_scratch(self) -> _StepAuthorityBuilderScratch:
    scratch = getattr(self, "_step_authority_builder_scratch", None)
    if scratch is None:
        scratch = _StepAuthorityBuilderScratch()
        setattr(self, "_step_authority_builder_scratch", scratch)
    return scratch










def prepare_step_context_impl(
    self,
    *,
    req_ids: Sequence[str],
    num_scheduled_tokens: Sequence[int],
    q_start_loc: Sequence[int],
    seq_lens: Sequence[int],
    num_actual_tokens: int,
    prompt_lengths: Optional[Sequence[int]] = None,
    num_computed_tokens: Optional[Sequence[int]] = None,
    step_ticket: StepTicket,
) -> Optional[StepContext]:
    _ensure_step_fault_container(self)
    _faults = consume_step_faults_impl(self)
    if _faults:
        for _f in _faults:
            if not _f.recoverable:
                raise RuntimeError(
                    f"step fault (non-recoverable): code={_f.code} message={_f.message}"
                )
            _log.warning("step fault (recoverable): code=%s message=%s", _f.code, _f.message)

    if not req_ids:
        self._step_refresh_commit_assert_prev_enqueued(
            next_handle_hint=int(getattr(self, "_step_handle_next_id", 0)) + 1,
            stage="prepare_step_context_idle",
        )
        self.step_context = None
        self.step_exec_hints = None
        self._current_step_handle_id = -1
        self._current_step_handle_generation = -1
        self._prepared_step_identity = _STEP_IDENTITY_EMPTY
        self._prepared_refresh_nonce = -1
        self._step_semantic_snapshot = None
        self._prev_decode_logf_row_state = None
        self.step_decode_plan_version = -1
        for i in range(len(self._step_wait_consumed_token_by_buf)):
            self._step_wait_consumed_token_by_buf[i] = 0
        if self._release_on_idle_enabled():
            self.release_idle_buffers()
        return None

    assert_cleanup_ledgers_drained_for_step_build(
        self,
        stage="prepare_step_context",
    )

    self._reclaim_retired_buffers()

    req_ids_tuple = tuple(req_ids)
    num_reqs = len(req_ids_tuple)
    step_token = int(step_ticket.target_epoch)
    step_source_signature = int(step_ticket.source_signature)
    step_scheduler_token = int(step_ticket.scheduler_token)
    refresh_nonce = self._refresh_nonce
    trigger_cfg = self.config.trigger
    use_refresh_nonce = trigger_cfg.enable_sentence_triggers
    refresh_nonce_key = refresh_nonce if use_refresh_nonce else -1
    step_identity: StepIdentity = (
        step_token,
        step_source_signature,
        step_scheduler_token,
        refresh_nonce_key,
    )
    tp_size = getattr(self, "_cached_tp_size", None)
    if tp_size is None:
        tp_size = int(getattr(self, "tp_size", 1) or 1)
        if tp_size <= 1:
            for env_name in ("VLLM_TENSOR_PARALLEL_SIZE", "VLLM_TP_SIZE", "TP_SIZE"):
                env_val = os.environ.get(env_name)
                if env_val is None:
                    continue
                try:
                    parsed = int(env_val)
                except ValueError:
                    continue
                if parsed > 0:
                    tp_size = parsed
                    break
        self._cached_tp_size = tp_size
    if (
        use_refresh_nonce
        and tp_size > 1
        and getattr(self, "_has_enginecore_sentence_hook", False)
    ):
        raise RuntimeError("tp>1 must not rely on enginecore sentence hook")
    # Sentence trigger: prefer worker-side token feed so intents are visible at
    # the step boundary.  _patched_append remains a TP=1 fallback only when the
    # worker source is unavailable.
    _wts_src = None  # type: tuple | None  # (token_ids_cpu, batch_id_to_idx)
    if use_refresh_nonce:
        _tk = getattr(self, '_worker_token_ids_cpu', None)
        _idx = getattr(self, '_worker_batch_id_to_idx', None)
        if _tk is not None and _idx is not None:
            _wts_src = (_tk, _idx)
    # vLLM V1 某些配置下可能在同一个 ctrl_step 内重复调用 prepare_step_context。
    # 若签名完全一致，则直接复用上一次构建的 StepContext，避免无意义重建（并防止重复 refresh 计划）。
    if (
        self.step_context is not None
        and self._prepared_step_identity == step_identity
        and self._prepared_refresh_nonce == refresh_nonce
        and self._prepared_num_actual_tokens == num_actual_tokens
        and self.step_context.req_ids == req_ids_tuple
    ):
        return self.step_context
    self._step_refresh_commit_assert_prev_enqueued(
        next_handle_hint=int(getattr(self, "_step_handle_next_id", 0)) + 1,
        stage="prepare_step_context_next_step",
    )
    q_lens: Tuple[int, ...] = tuple(num_scheduled_tokens[:num_reqs])

    if len(q_start_loc) >= num_reqs + 1:
        q_start_loc = tuple(q_start_loc[: num_reqs + 1])
        if not q_lens:
            q_lens = tuple(
                max(0, q_start_loc[i + 1] - q_start_loc[i])
                for i in range(num_reqs)
            )
    else:
        q_start_loc_list = [0]
        for q_len in q_lens:
            q_start_loc_list.append(q_start_loc_list[-1] + q_len)
        q_start_loc = tuple(q_start_loc_list)

    if len(seq_lens) < num_reqs:
        raise ValueError(
            f"seq_lens length {len(seq_lens)} < num_reqs {num_reqs}; "
            f"caller must provide seq_lens for all requests"
        )
    seq_lens_tuple = tuple(seq_lens[:num_reqs])
    max_query_len = max(q_lens) if q_lens else 0
    max_seq_len = max(seq_lens_tuple) if seq_lens_tuple else 0

    # --- Persistent batch: incremental diff tracking (VLLM_SPARSE_PERSISTENT_BATCH=1) ---
    pb = None
    if _PERSISTENT_BATCH_ENABLED:
        pb = getattr(self, '_persistent_batch', None)
        if pb is None:
            from patches.persistent_batch import PersistentStepBatch
            pb = PersistentStepBatch.create(max(128, num_reqs * 2))
            self._persistent_batch = pb
        pb.apply_diff(req_ids_tuple, seq_lens_tuple, q_lens)

    # async trace：每步输出上一 epoch 的统计（如果启用）。
    # step profile：输出上一 epoch 的统计（如果启用）。
    current_epoch = self.step_context_epoch
    target_epoch = int(step_ticket.target_epoch)
    if target_epoch < current_epoch:
        raise RuntimeError(
            "prepare_step_context ticket epoch regressed: "
            f"ticket_epoch={target_epoch}, current_epoch={current_epoch}"
        )
    if target_epoch > current_epoch:
        self._step_profile_end_if_needed(epoch=current_epoch)
        self.step_context_epoch = target_epoch
    else:
        # same-step reentry with identical ticket: keep epoch stable.
        self.step_context_epoch = current_epoch

    self._step_semantic_snapshot = self._build_step_semantic_snapshot()
    # 每步重置 slot debug 采样信息（避免沿用上一步数据）
    # async trace：为新 epoch 重置计数器（如果启用）。
    req_id_to_index = (
        pb.export_req_id_to_index() if pb is not None
        else {rid: idx for idx, rid in enumerate(req_ids_tuple)}
    )

    prompt_len_list: Tuple[int, ...] = (
        tuple(prompt_lengths[:num_reqs])
        if prompt_lengths is not None else tuple()
    )
    # TP prompt 合同在 prepare_inputs 的单源入口已校验（validate_tp_input_contract +
    # ensure_tp_prompt_lengths）；这里不重复校验，避免每步热路径重复 gate。
    computed_list: Tuple[int, ...] = (
        tuple(num_computed_tokens[:num_reqs])
        if num_computed_tokens is not None else tuple()
    )
    has_prompt_counter = bool(
        prompt_len_list
        and computed_list
        and len(prompt_len_list) == num_reqs
        and len(computed_list) == num_reqs
    )
    scratch = _get_step_authority_builder_scratch(self)
    scratch.reset(batch_size=num_reqs)
    prefill_active: Optional[bool] = False if has_prompt_counter else None
    has_prefill_by_prompt = False
    _publish_bootstrap_readiness_before_step_authority(
        self,
        req_ids_tuple=req_ids_tuple,
    )

    # 预先构建 per-row 阶段判定与 bootstrap_done 列表，减少后续重复遍历。
    is_prefill_by_row_list = scratch.is_prefill_by_row
    has_prefill_row = False
    has_decode_row = False
    prefill_rows_list = scratch.prefill_rows
    bootstrap_done_list = scratch.bootstrap_done
    compact_consume_delay_steps = _attrib_compact_consume_delay_steps()

    compact_ready_for_row_policy_by_req: dict[str, bool] = {}

    def _bootstrap_done_for_compact_row_policy(rid: str, tracking: object) -> bool:
        boot = _bootstrap_done_for_row_policy(
            config=self.config,
            tracking=tracking,
            compact_consume_delay_steps=compact_consume_delay_steps,
        )
        if not boot:
            return False
        if bool(getattr(tracking, "_was_short_dense", False)):
            return True
        cached = compact_ready_for_row_policy_by_req.get(rid)
        if cached is None:
            cached = bool(self._request_compact_ready_all_layers(rid))
            compact_ready_for_row_policy_by_req[rid] = cached
        return bool(cached)

    # Step-wise 更新 decode_step：使用 vLLM 提供的 num_computed_tokens_cpu 与 num_prompt_tokens，
    # 避免依赖 append_output_token_ids 的回调时序（multi_step_stream_outputs 下可能滞后）。
    if has_prompt_counter:
        compact_threshold = self._compact_threshold_tokens()
        for idx, rid in enumerate(req_ids_tuple):
            prompt_len = prompt_len_list[idx]
            computed = computed_list[idx]
            tracking = self._ensure_request(rid)
            if prompt_len > 0:
                if computed >= prompt_len:
                    # 关键：num_computed_tokens_cpu 在 multi_step_stream_outputs 下可能滞后，
                    # 若这里无条件覆盖，会把 record_generated_tokens() 推进的 decode_step 回滚，
                    # 导致 interval refresh 永远触发不了（或触发抖动）。
                    observed = max(0, computed - prompt_len)
                    # --- sentence trigger: feed latest decode token ---
                    # The token at position (prompt_len + observed) in
                    # token_ids_cpu is the latest sampled output token.
                    # record_generated_tokens checks it against sentence-end
                    # patterns and may set pending_refresh (FORCE_NOW).
                    # TP=1 also uses this path when worker token source is
                    # available; _patched_append is only a late fallback.
                    if _wts_src is not None:
                        # _trig_fed 语义：已 feed 给 trigger 的 decode token 数量（初始 0）。
                        _fed = int(tracking._trig_fed)
                        if observed > _fed:
                            _tk_cpu, _id2idx = _wts_src
                            _bidx = _id2idx.get(rid)
                            if _bidx is None:
                                if tp_size > 1:
                                    raise RuntimeError(
                                        "tp>1 sentence trigger missing worker row mapping for request "
                                        f"{rid!r}"
                                    )
                            else:
                                # feed 区间使用半开区间 [start, end)；
                                # observed=0 表示尚无真实 decode token，不应触发任何 feed。
                                feed_start = prompt_len + _fed
                                feed_end = prompt_len + observed
                                _n = feed_end - feed_start
                                if _n > 0:
                                    decode_step_before_trigger = (
                                        int(tracking.decode_step)
                                        if tracking.decode_step is not None
                                        else -1
                                    )
                                    # 热路径仅传窗口描述符，避免在 worker 侧构建 Python token list。
                                    token_window = (_tk_cpu, _bidx, feed_start, feed_end)
                                    recorded = self.evaluate_tp_sentence_token_window(
                                        rid, token_window
                                    )
                                    # decode_step 在本函数中单写：trigger feed 只负责落 trigger intent。
                                    tracking.decode_step = decode_step_before_trigger
                                    if recorded:
                                        # [TP-ASYNC-HARVEST] evaluate 返回实际喂入
                                        # 的 token 数（async 下窗口尾部占位符被前缀
                                        # 截断）：水位只推进已喂入部分，余量下一步
                                        # 接续。sync 模式全窗喂入时 _fed+recorded
                                        # == observed，与旧语义逐位一致。
                                        tracking._trig_fed = _fed + int(recorded)
                    elif tp_size > 1 and use_refresh_nonce:
                        _fed = int(tracking._trig_fed)
                        if observed > _fed:
                            raise RuntimeError(
                                "tp>1 sentence trigger missing worker token source during decode feed: "
                                f"request={rid!r} observed={observed} fed={_fed} "
                                f"step={int(step_ticket.target_epoch)}"
                            )
                    elif use_refresh_nonce:
                        _fed = int(tracking._trig_fed)
                    prev = int(tracking.decode_step) if tracking.decode_step is not None else -1
                    tracking.decode_step = max(prev, observed)
                else:
                    # prompt 仍未完成：只在 decode_step 尚未建立时写 -1，避免回滚已建立的 decode_step。
                    if tracking.decode_step is None or int(tracking.decode_step) < 0:
                        tracking.decode_step = -1
            else:
                # prompt_len 不可用/为 0 时不要覆盖已有的 decode_step：
                # - Request.append_output_token_ids 仍会更新 decode_step（token 计数）；
                # - 若此处强制写 -1，会导致 interval refresh 永远不触发（decode_step 被每步抹掉）。
                if tracking.decode_step is None:
                    tracking.decode_step = -1
            # interval 刷新计数基准：首次进入 decode 时，把 last_decode_refresh_step 置为当前 decode_step。
            # 否则 plan_refresh_requests 每步都会把 last_decode_refresh 当作“本步 decode_step”，导致 interval 永不触发。
            # ⚠️ 注意：last_decode_refresh_step=0 是合法值（首个 decode token 后），不能用 `or -1` 判空。
            if tracking.decode_step >= 0 and tracking.last_decode_refresh_step < 0:
                tracking.last_decode_refresh_step = tracking.decode_step
            # 方案 1：request-wise 的 prefill_done 仅代表 prompt ingest 是否完成（阶段信号）
            if prompt_len > 0:
                tracking.prefill_done = bool(computed >= prompt_len)
            # 短上下文（无需 compact bootstrap）：一旦 prompt 完成即可直接视为“已 bootstrapped”
            if tracking.prefill_done and not tracking.bootstrap_done:
                ctx_len = seq_lens_tuple[idx]
                # 约定：只有“真正超过阈值”才离开 short（ctx_len > threshold）。
                # 因此在 ctx_len == threshold 时仍视为 short。
                if compact_threshold <= 0 or (ctx_len > 0 and ctx_len <= compact_threshold):
                    tracking.bootstrap_done = True
                    tracking.bootstrap_pending = False
                    tracking.bootstrap_pending_epoch = -1
                    tracking.bootstrap_pending_events = []
                    self._bootstrap_pending_request_ids.discard(rid)
                    tracking._was_short_dense = True

            # Threshold crossing 检测：short_dense → not short_dense 时强行触发 decode refresh
            # 构建 compact cache（last_n=1 的 decode refresh），避免 compact_only kernel 对
            # 该行仍以 FULL_CONTEXT_FLAG 遍历全部 paged blocks（效率低）。
            if tracking.prefill_done and tracking.bootstrap_done:
                ctx_len = seq_lens_tuple[idx]
                was_short = tracking._was_short_dense
                # 与上面的 short 语义保持一致：<= threshold 仍为 short。
                is_short = compact_threshold > 0 and ctx_len > 0 and ctx_len <= compact_threshold
                if was_short and not is_short:
                    crossing_step = int(tracking.decode_step) if tracking.decode_step is not None else -1
                    if crossing_step < 0:
                        raise RuntimeError(
                            f"threshold crossing pending requires decode_step>=0 for request {rid!r}"
                        )
                    ticket = self._ensure_request_ticket(rid)
                    if (
                        (not ticket.pending_refresh)
                        or ticket.pending_policy
                        != PendingPolicy.FORCE_NOW
                        or ticket.pending_decode_step < 0
                    ):
                        mark_threshold_crossing(ticket=ticket, decode_step=crossing_step)
                    # crossing 命中后保持 short-dense 保护，直到 refresh ack 真正清票。
                    tracking._was_short_dense = True
                else:
                    if was_short:
                        ticket = self._request_intent_tickets.get(rid)
                        if (
                            ticket is not None
                            and ticket.pending_refresh
                            and ticket.pending_policy
                            == PendingPolicy.FORCE_NOW
                        ):
                            tracking._was_short_dense = True
                        else:
                            tracking._was_short_dense = is_short
                    else:
                        tracking._was_short_dense = is_short

            # 记录 prompt 判定（用于 is_decode_only 与 release 判定）
            if prompt_len > 0 and computed < prompt_len:
                prefill_active = True
                has_prefill_by_prompt = True

            # [RESUME-STATE-RESET 2026-07-03] 已 bootstrap 的请求回到 prompt 中
            # 段=preempt 后 RECOMPUTE 重算(抢占不进 finished_req_ids,清理链对
            # 其 no-op,旧压缩状态原样留在同 slot)。在 boot 读取前重置:压缩状
            # 态清零+bootstrap_done=False,重算即重新 bootstrap;判据自灭天然
            # once,黄金档(无抢占)不可达零扰动。
            if (
                tracking is not None
                and bool(getattr(tracking, "bootstrap_done", False))
                and prompt_len > 0
                and computed < prompt_len
            ):
                self._reset_request_sparse_state_for_resume(rid, tracking)

            boot = _bootstrap_done_for_compact_row_policy(rid, tracking)
            bootstrap_done_list.append(boot)
            if prompt_len > 0:
                is_prefill = computed < prompt_len
            else:
                is_prefill = not boot
            # q_len(=num_scheduled_tokens) 是调度器权威、owner 矩阵的派发依据:
            # q_len>1 的行必须归 prefill(多 token),即使 prompt 边界计数已
            # computed>=prompt_len——resume/preempt 重算续吞已生成 token 段、或
            # 错峰 chunked prefill 越过 prompt 边界的续段(4B bs8 实证 row
            # q_len=183 被判 decode 后炸 owner 的 q_len==1 不变式)。单向对账,
            # owner 侧 fail-fast 原样保留守新形态;黄金档(整段 prefill/纯
            # decode)该条件不可满足,零影响。
            if not is_prefill and idx < len(q_lens) and int(q_lens[idx]) > 1:
                is_prefill = True
            is_prefill_by_row_list.append(is_prefill)
            if is_prefill:
                has_prefill_row = True
                prefill_rows_list.append(idx)
            else:
                has_decode_row = True

    else:
        for idx, rid in enumerate(req_ids_tuple):
            tracking = self.request_states.get(rid)
            boot = bool(tracking.bootstrap_done) if tracking is not None else False
            bootstrap_done_list.append(boot)
            is_prefill = not boot
            if not is_prefill and idx < len(q_lens) and int(q_lens[idx]) > 1:
                is_prefill = True
            is_prefill_by_row_list.append(is_prefill)
            if is_prefill:
                has_prefill_row = True
                prefill_rows_list.append(idx)
            else:
                has_decode_row = True

    # mixchunk 判定：仅使用“当前 step 事实”，不依赖跨步粘连状态。
    # 当同一步内同时存在 prefill row 和 decode row 时，视为 mixed-phase。
    has_request_phase_mix = bool(has_prefill_row and has_decode_row)
    # step-wise hash 签名：缓存稳态 decode（req_ids + phase 不变时跳过 hash）。
    _hash_cache_key = (req_ids_tuple, has_prefill_row)
    _prev = getattr(self, "_step_hash_cache", None)
    # [PERF-07] 原 `is` 身份比较结构性恒 miss（req_ids_tuple 每步经 list→tuple
    # 新建对象），"稳态跳过 hash"设计从未生效；等值比较（str tuple，字符串驻留
    # 下近 O(1)/元素）使缓存按设计命中。
    if _prev is not None and _prev[0] == _hash_cache_key[0] and _prev[1] == _hash_cache_key[1]:
        req_set_hash, row_phase_hash = _prev[2], _prev[3]
    else:
        req_set_hash, row_phase_hash = _compute_step_request_hashes(
            req_ids_tuple=req_ids_tuple,
            is_prefill_by_row=is_prefill_by_row_list,
            num_reqs=num_reqs,
        )
        self._step_hash_cache = (req_ids_tuple, has_prefill_row, req_set_hash, row_phase_hash)
    # Sync derived phase state into persistent batch (enables cross-step caching)
    if pb is not None:
        pb.sync_derived_arrays(
            is_prefill_by_row_list, bootstrap_done_list,
            has_prefill_row, has_decode_row,
        )

    # 回退判据：当上游不给 prompt/computed 时，基于"近期是否有 prefill enqueue"判断是否可释放。
    if prefill_active is None:
        no_prefill_pending = True
        for mask in self.step_prefill_chunk_mask:
            if mask != 0:
                no_prefill_pending = False
                break
        if no_prefill_pending:
            last_enqueue = self._prefill_last_enqueue_epoch
            grace = max(1, _PREFILL_RELEASE_GRACE_STEPS)
            if last_enqueue < 0 or (self.step_context_epoch - last_enqueue) >= grace:
                prefill_active = False

    # decode-only: arm prefill buffer release (after wait_event on all required bufs)
    if prefill_active is False:
        if self._prefill_release_pending_epoch < 0:
            last_enqueue = self._prefill_last_enqueue_epoch
            done_epoch = self._prefill_release_done_epoch
            if last_enqueue >= 0 and (done_epoch < 0 or done_epoch < last_enqueue):
                self._prefill_release_pending_epoch = self.step_context_epoch
                self._prefill_release_waited_mask = 0
    elif prefill_active is True:
        self._prefill_release_pending_epoch = -1
        self._prefill_release_waited_mask = 0
        self._prefill_release_done_epoch = -1

    # execution-authoritative gating:
    # multi_step_stream_outputs 下可能出现“prepare_step_context 已推进、但本轮无 kernel 执行”的空转步。
    # 空转步必须保持 refresh 票据语义，但不能 materialize 成本步 refresh_reqs（否则会产生 ghost plan）。
    step_has_kernel_work = bool(
        num_actual_tokens > 0 and any(q_len > 0 for q_len in q_lens)
    )

    refresh_plan = self.plan_refresh_requests(
        req_ids_tuple,
        use_cache=True,
        update_state=bool(step_has_kernel_work),
        allow_materialize=bool(step_has_kernel_work),
    )
    step_refresh_nonempty = bool(refresh_plan.has_refresh_reqs)
    refresh_reqs = tuple(refresh_plan.refresh_reqs)
    refresh_reason = str(refresh_plan.refresh_reason)
    bootstrap_done = bool(refresh_plan.bootstrap_done)
    refresh_mode_by_row = tuple(refresh_plan.mode_by_row)
    if len(refresh_mode_by_row) != num_reqs:
        raise RuntimeError(
            "refresh plan row-size mismatch: "
            f"plan={len(refresh_mode_by_row)} batch={num_reqs} step={self.step_context_epoch}"
        )
    refresh_rows = tuple(refresh_plan.refresh_rows)
    force_dense_while_inflight_by_row = tuple(
        bool(v)
        for v in getattr(
            refresh_plan,
            "force_dense_while_inflight_by_row",
            tuple(),
        )
    )
    refresh_row_mask_list = scratch.refresh_row_mask
    for row_idx in refresh_rows:
        if row_idx < 0 or row_idx >= num_reqs:
            raise RuntimeError(
                "refresh plan row index out of range: "
                f"row={row_idx} batch={num_reqs} step={self.step_context_epoch}"
            )
        refresh_row_mask_list[row_idx] = True
    refresh_row_mask = tuple(refresh_row_mask_list)
    refresh_row_count = sum(1 for flag in refresh_row_mask if flag)
    if refresh_row_count != len(refresh_reqs):
        raise RuntimeError(
            "refresh plan req/row cardinality mismatch: "
            f"rows={refresh_row_count} reqs={len(refresh_reqs)} "
            f"step={self.step_context_epoch}"
        )
    slot_map = self.get_step_global_slot_map(req_ids_tuple)
    if pb is not None:
        for _i in range(num_reqs):
            pb.slot_by_row[_i] = int(slot_map.get(req_ids_tuple[_i], -1))
        slot_by_row = pb.export_slot_by_row()
    else:
        slot_by_row = tuple(int(slot_map.get(rid, -1)) for rid in req_ids_tuple)
    # === VLLM_SPARSE_PSC_BSKIP: content-key + steady decision ===
    _psc_full_build = True
    _psc_bskip_use = False
    _psc_key = None
    if _PSC_BSKIP_ENABLED:
        try:
            _psc_rows = tuple(
                (
                    bool(is_prefill_by_row_list[_r]),
                    bool(bootstrap_done_list[_r]) if _r < len(bootstrap_done_list) else False,
                    bool((self.request_states.get(req_ids_tuple[_r]) or None) is not None
                         and getattr(self.request_states.get(req_ids_tuple[_r]),
                                     '_was_short_dense', False)),
                )
                for _r in range(num_reqs)
            )
            _psc_key = (
                req_ids_tuple,
                _psc_rows,
                q_lens,
                tuple(bool(v) for v in refresh_row_mask),
                tuple(int(v) for v in refresh_mode_by_row),
                refresh_rows,
                bool(step_refresh_nonempty),
                bool(bootstrap_done),
                force_dense_while_inflight_by_row,
                refresh_reqs,
                slot_by_row,
                int(self._refresh_layer_group_event_idx & 1),
                getattr(self, '_interval_merge_policy', 'delta1'),
                tuple(int(_sl) // 8 for _sl in seq_lens_tuple),
            )
            _psc_cached = getattr(self, '_psc_bskip_cache', None)
            # Reuse only when the key matches AND the cached step was a pure
            # steady decode: no refresh AND no logf capture (=> B's seq-len-
            # dependent outputs are all empty/zero, so reuse stays exact while
            # seq_lens grow). See DESIGN NOTE 1.
            if (
                _psc_cached is not None
                and _psc_cached.get('key') == _psc_key
                and _psc_cached.get('steady_eligible') is True
            ):
                _psc_bskip_use = True
                if _PSC_BSKIP_ASSERT:
                    _psc_full_build = True
                elif os.environ.get("VLLM_SPARSE_PSC_BSKIP_REUSE") == "1":
                    _psc_full_build = False
                else:
                    _psc_full_build = not _PSC_BSKIP_ASSERT
        except Exception:
            _psc_full_build = True
            _psc_bskip_use = False
    # === end content-key ===
    if _psc_full_build:
        row_mode_by_row_list = scratch.row_mode_by_row
        logf_producer_by_row_list = scratch.logf_producer_by_row
        needs_logits_by_row_list = scratch.needs_logits_by_row
        logits_last_n_by_row_list = scratch.logits_last_n_by_row
        for idx, rid in enumerate(req_ids_tuple):
            tracking = self.request_states.get(rid)
            is_prefill_row = is_prefill_by_row_list[idx]
            short_dense_row = tracking._was_short_dense if tracking is not None else False
            boot_done_row = bootstrap_done_list[idx] if idx < len(bootstrap_done_list) else False
            is_refresh_row = refresh_row_mask[idx]
            force_dense_for_pending_refresh = (
                idx < len(force_dense_while_inflight_by_row)
                and force_dense_while_inflight_by_row[idx]
            )
            row_mode, logf_producer = resolve_decode_row_policy(
                is_prefill_row=is_prefill_row,
                is_refresh_row=is_refresh_row,
                is_short_dense_row=short_dense_row,
                bootstrap_done_row=boot_done_row,
                force_dense_for_pending_refresh=force_dense_for_pending_refresh,
            )
            if row_mode == _ROW_MODE_COMPACT and logf_producer == _LOGF_PRODUCER_ATTN:
                raise RuntimeError("compact row cannot carry ATTN log_f producer")
            row_mode_by_row_list.append(row_mode)
            logf_producer_by_row_list.append(logf_producer)
            plan_mode = refresh_mode_by_row[idx]
            if (
                plan_mode == StepRefreshMode.MUST_NOW
                and (not is_refresh_row)
            ):
                raise RuntimeError(
                    "refresh plan invariant violated: MUST_NOW missing from refresh_reqs: "
                    f"req={rid!r} step={self.step_context_epoch}"
                )
            if (
                plan_mode == StepRefreshMode.INFLIGHT
                and is_refresh_row
            ):
                raise RuntimeError(
                    "refresh plan invariant violated: INFLIGHT row appears in refresh_reqs: "
                    f"req={rid!r} step={self.step_context_epoch}"
                )
            needs_logits = logf_producer == _LOGF_PRODUCER_ATTN
            needs_logits_by_row_list.append(needs_logits)
            logits_last_n_by_row_list.append(1 if needs_logits else 0)
        if pb is not None:
            for _i in range(num_reqs):
                pb.row_mode[_i] = row_mode_by_row_list[_i]
            row_mode_by_row = pb.export_row_mode()
        else:
            row_mode_by_row = tuple(row_mode_by_row_list)
        row_mode_signature = row_mode_by_row
        logf_producer_by_row = tuple(logf_producer_by_row_list)
        needs_logits_by_row = tuple(needs_logits_by_row_list)
        logits_last_n_by_row = tuple(logits_last_n_by_row_list)
        layer_effective_refresh_by_row = tuple(
            bool(step_refresh_nonempty and refresh_row_mask[idx]) for idx in range(num_reqs)
        )

        # ── 预计算 refresh/bootstrap slot 集合（单源）──
        _refresh_slot_set: set[int] = set()
        _bootstrap_slot_set: set[int] = set()
        for _slot_idx in range(num_reqs):
            if not refresh_row_mask[_slot_idx]:
                continue
            if _slot_idx < len(is_prefill_by_row_list) and bool(is_prefill_by_row_list[_slot_idx]):
                continue
            _slot = int(slot_by_row[_slot_idx]) if _slot_idx < len(slot_by_row) else -1
            if _slot < 0:
                continue
            _refresh_slot_set.add(_slot)
            if _slot_idx < len(bootstrap_done_list) and not bool(bootstrap_done_list[_slot_idx]):
                _bootstrap_slot_set.add(_slot)
        _refresh_slots = normalize_refresh_slot_list(_refresh_slot_set)
        _bootstrap_slots = normalize_refresh_slot_list(_bootstrap_slot_set)
        _payload_slots = set(_refresh_slot_set)
        _payload_slots.update(_bootstrap_slot_set)
        _refresh_capture_slot_list = normalize_refresh_slot_list(_payload_slots)
    else:
        _c = _psc_cached
        row_mode_by_row_list = list(_c['row_mode_by_row'])
        logf_producer_by_row_list = list(_c['logf_producer_by_row'])
        logits_last_n_by_row_list = list(_c['logits_last_n_by_row'])
        needs_logits_by_row_list = list(_c['needs_logits_by_row'])
        row_mode_by_row = _c['row_mode_by_row']
        row_mode_signature = _c['row_mode_signature']
        logf_producer_by_row = _c['logf_producer_by_row']
        needs_logits_by_row = _c['needs_logits_by_row']
        logits_last_n_by_row = _c['logits_last_n_by_row']
        layer_effective_refresh_by_row = _c['layer_effective_refresh_by_row']
        _refresh_slots = _c['refresh_slots']
        _bootstrap_slots = _c['bootstrap_slots']
        _refresh_capture_slot_list = _c['refresh_capture_slot_list']
    step_handle = self.allocate_step_handle(
        epoch=self.step_context_epoch,
        req_ids=req_ids_tuple,
        num_actual_tokens=num_actual_tokens,
        refresh_plan_signature=tuple(refresh_plan.plan_signature),
    )
    decode_plan_version = _make_decode_plan_version(
        step_handle_id=int(step_handle.handle_id),
        step_handle_generation=int(step_handle.generation),
    )
    self.step_decode_plan_version = int(decode_plan_version)
    self._step_refresh_commit_begin(
        handle_id=step_handle.handle_id,
        handle_generation=step_handle.generation,
        planned_reqs=len(refresh_reqs),
        planned_rows=len(refresh_rows),
        num_actual_tokens=num_actual_tokens,
    )
    if _psc_full_build:
        plan_signature = build_step_plan_signature(
            req_ids=req_ids_tuple,
            slot_by_row=slot_by_row,
            row_mode_by_row=row_mode_by_row,
            layer_effective_refresh_by_row=layer_effective_refresh_by_row,
            bootstrap_done_by_row=tuple(bootstrap_done_list),
        )
        # layer-group gating: inline computation for step envelope，避免 per-layer
        # controller attribute reads 与 dispatcher 侧回退判定。
        _lg_groups = self.config.refresh_layer_groups or 1
        if _lg_groups < 1:
            _lg_groups = 1
        _lg_decode_bd = bool(bootstrap_done)
        if _lg_groups == 2 and is_prefill_by_row_list:
            _lg_decode_bd = True
            for _lg_idx, _lg_pf in enumerate(is_prefill_by_row_list):
                if _lg_pf:
                    continue
                if _lg_idx >= len(bootstrap_done_list) or not bootstrap_done_list[_lg_idx]:
                    _lg_decode_bd = False
                    break
        _lg_enabled = _lg_groups == 2 and _lg_decode_bd and has_decode_row
        _lg_active = (self._refresh_layer_group_event_idx & 1) if _lg_enabled else -1
        max_seq_bound = max_seq_len if max_seq_len > 0 else 0
        decode_logf_capacity_by_row_list = scratch.decode_logf_capacity_by_row
        decode_logf_mask_by_row_list = scratch.decode_logf_mask_by_row
        decode_logf_q_lens_by_row_list = scratch.decode_logf_q_lens_by_row
        decode_logf_attn_rows_list = scratch.decode_logf_attn_rows
        decode_logf_max_kv = 0
        for idx in range(num_reqs):
            q_len = q_lens[idx] if idx < len(q_lens) else 1
            seq_len = seq_lens_tuple[idx] if idx < len(seq_lens_tuple) else 0
            producer = (
                logf_producer_by_row[idx]
                if idx < len(logf_producer_by_row)
                else _LOGF_PRODUCER_NONE
            )
            capture_needed = producer != _LOGF_PRODUCER_NONE
            cap = 0
            if capture_needed:
                cap = max(1, min(max(0, seq_len), max(1, max_seq_bound)))
                decode_logf_max_kv = max(decode_logf_max_kv, cap)
            mask = 1 if producer == _LOGF_PRODUCER_ATTN else 0
            decode_logf_capacity_by_row_list.append(cap)
            decode_logf_q_lens_by_row_list.append(max(0, q_len))
            decode_logf_mask_by_row_list.append(mask)
            if mask == 1:
                decode_logf_attn_rows_list.append(idx)
        kv_bucket = _CAPTURE_KV_BUCKET_CACHED
        if kv_bucket <= 0:
            kv_bucket = 256
        kv_bucket = max(256, kv_bucket)
        decode_logf_stride_head = (
            _align_up_int(decode_logf_max_kv, kv_bucket)
            if decode_logf_max_kv > 0
            else 0
        )
        decode_logf_capacity_by_row = tuple(int(v) for v in decode_logf_capacity_by_row_list)
        decode_logf_q_lens_by_row = tuple(int(v) for v in decode_logf_q_lens_by_row_list)
        decode_logf_mask_by_row = tuple(int(v) for v in decode_logf_mask_by_row_list)
        decode_logf_attn_rows = tuple(int(v) for v in decode_logf_attn_rows_list)
        decode_logf_row_state = tuple(
            zip(
                req_ids_tuple,
                decode_logf_capacity_by_row,
                decode_logf_q_lens_by_row,
                decode_logf_mask_by_row,
            )
        )
        prev_decode_logf_row_state = getattr(self, "_prev_decode_logf_row_state", None)
        if (
            prev_decode_logf_row_state is None
            or len(prev_decode_logf_row_state) != len(decode_logf_row_state)
        ):
            decode_logf_dirty_rows = tuple(range(num_reqs))
        else:
            decode_logf_dirty_rows = tuple(
                idx
                for idx, row_state in enumerate(decode_logf_row_state)
                if row_state != prev_decode_logf_row_state[idx]
            )
        self._prev_decode_logf_row_state = decode_logf_row_state
        refresh_signals_by_row = tuple(int(mode) for mode in refresh_mode_by_row)
        envelope_cache_signature: Tuple[object, ...] = (
            self.step_context_epoch,
            getattr(self, "_interval_merge_policy", "delta1"),
            req_ids_tuple,
            slot_by_row,
            row_mode_signature,
            refresh_reqs,
            refresh_rows,
            logf_producer_by_row,
            needs_logits_by_row,
            logits_last_n_by_row,
        )
        step_envelope_v2 = StepEnvelopeV2(
            epoch=self.step_context_epoch,
            handle_id=step_handle.handle_id,
            handle_generation=step_handle.generation,
            req_ids=req_ids_tuple,
            slot_by_row=slot_by_row,
            row_mode_by_row=row_mode_by_row,
            refresh_signals=refresh_signals_by_row,
            refresh_rows=refresh_rows,
            refresh_reqs=refresh_reqs,
            layer_effective_refresh_by_row=layer_effective_refresh_by_row,
            refresh_reason=refresh_reason,
            bootstrap_done=bootstrap_done,
            plan_signature=plan_signature,
            layer_group_active=_lg_active,
            layer_group_enabled=_lg_enabled,
            cache_signature=envelope_cache_signature,
        )
        ctx = StepContext(
            req_ids=req_ids_tuple,
            num_reqs=num_reqs,
            num_actual_tokens=num_actual_tokens,
            q_start_loc=q_start_loc,
            q_lens=q_lens,
            seq_lens=seq_lens_tuple,
            max_query_len=max_query_len,
            max_seq_len=max_seq_len,
            epoch=self.step_context_epoch,
            req_id_to_index=req_id_to_index,
            step_handle_id=step_handle.handle_id,
            step_handle_generation=step_handle.generation,
            step_handle=step_handle,
            prompt_lens=prompt_len_list if prompt_len_list else None,
            num_computed_tokens=computed_list if computed_list else None,
            step_envelope_v2=step_envelope_v2,
        )
    else:
        _c = _psc_cached
        plan_signature = _c['plan_signature']
        _lg_enabled = _c['lg_enabled']
        _lg_active = _c['lg_active']
        decode_logf_capacity_by_row = _c['decode_logf_capacity_by_row']
        decode_logf_q_lens_by_row = _c['decode_logf_q_lens_by_row']
        decode_logf_mask_by_row = _c['decode_logf_mask_by_row']
        decode_logf_attn_rows = _c['decode_logf_attn_rows']
        decode_logf_stride_head = _c['decode_logf_stride_head']
        decode_logf_dirty_rows = _c['decode_logf_dirty_rows']
        decode_logf_row_state = _c['decode_logf_row_state']
        refresh_signals_by_row = _c['refresh_signals_by_row']
        # _prev_decode_logf_row_state side-effect must still advance.
        self._prev_decode_logf_row_state = decode_logf_row_state
        step_envelope_v2 = _psc_bskip_restamp_envelope(
            _c['step_envelope_v2'],
            epoch=self.step_context_epoch,
            handle_id=step_handle.handle_id,
            handle_generation=step_handle.generation,
            interval_merge_policy=getattr(self, '_interval_merge_policy', 'delta1'),
            req_ids_tuple=req_ids_tuple,
            slot_by_row=slot_by_row,
            row_mode_signature=row_mode_signature,
            refresh_reqs=refresh_reqs,
            refresh_rows=refresh_rows,
            logf_producer_by_row=logf_producer_by_row,
            needs_logits_by_row=needs_logits_by_row,
            logits_last_n_by_row=logits_last_n_by_row,
        )
        ctx = _psc_bskip_restamp_context(
            _c['ctx'],
            epoch=self.step_context_epoch,
            handle_id=step_handle.handle_id,
            handle_generation=step_handle.generation,
            step_handle=step_handle,
            num_actual_tokens=num_actual_tokens,
            q_start_loc=q_start_loc,
            q_lens=q_lens,
            seq_lens=seq_lens_tuple,
            max_query_len=max_query_len,
            max_seq_len=max_seq_len,
            req_id_to_index=req_id_to_index,
            prompt_lens=prompt_len_list if prompt_len_list else None,
            num_computed_tokens=computed_list if computed_list else None,
            step_envelope_v2=step_envelope_v2,
        )
    # 清理 prefill 批次（按 step epoch）
    self.step_prefill_epoch = self.step_context_epoch
    for buf_id, bucket in enumerate(self.step_prefill_chunk_payloads):
        for idx in range(_CAPTURE_CHUNK):
            bucket[idx] = None
        self.step_prefill_chunk_mask[buf_id] = 0
    # layer dispatch 进度归零（防止上一步残留）
    self.layer_dispatch_epoch = self.step_context_epoch
    self.layer_dispatch_cursor = 0
    self.layer_dispatch_layer_count = len(self.layer_cache_keys)
    self.step_refresh_epoch = self.step_context_epoch
    for buf_id, bucket in enumerate(self.step_refresh_chunk_payloads):
        for idx in range(_CAPTURE_CHUNK):
            bucket[idx] = None
        self.step_refresh_chunk_mask[buf_id] = 0
    # ⚠️ 重要：不要在每个 step 开始时清空 capture_layout_ring。
    # 原因：
    # - refresh_stream 异步流水线可能仍在消费上一 step 的 capture buffers；
    # - 若此处把 ring 置 None，会提前释放/复用底层 storage，导致跨 stream 的静默数据竞争；
    # - _get_step_capture_layout 已支持跨 step 复用并在 epoch 变化时更新动态元数据。
    self.step_prefill_plan_epoch = -1
    self.step_prefill_plan_handle_id = -1
    self.step_prefill_plan_handle_generation = -1
    self.step_prefill_capture_plan_by_req = {}
    self.step_prefill_finalize_req_ids = tuple()
    self.step_prefill_capture_last_n_by_row = None
    self.step_prefill_capture_last_n_epoch = -1
    self.step_prefill_capture_last_n_handle_id = -1
    self.step_prefill_capture_last_n_handle_generation = -1

    # ============ 构建 StepMeta（两阶段缓存的第一阶段）============
    # 收集跨层共享的信息，供 per-layer cache 使用
    semantic_snapshot = self._get_step_semantic_snapshot()
    recent_cap = max(0, semantic_snapshot.recent_tokens)
    # block_size 在此时不可知（需要从 KV cache 获取），使用默认值 16
    # 实际 block_size 会在 dispatcher 中通过 block_table 推断
    block_size = 16

    # 判断是否全部是 decode：
    # - 使用 prompt_len 与 num_computed_tokens 判断是否越过 prompt
    # - 兼容 multi-step decode（q_len > 1）
    # - 兜底：decode_step>=0 或 q_len==1
    if q_lens:
        is_decode_only = True
        for idx, (rid, q_len) in enumerate(zip(req_ids_tuple, q_lens)):
            tracking = self.request_states.get(rid)
            decode_started = (
                tracking is not None
                and tracking.decode_step is not None
                and tracking.decode_step >= 0
            )
            prompt_len = prompt_len_list[idx] if idx < len(prompt_len_list) else 0
            computed = computed_list[idx] if idx < len(computed_list) else -1
            if prompt_len > 0:
                # 关键：prefill 的最后一个 chunk 可能 q_len==1，但仍应视为 prefill（否则会错误走 decode-only 路径）
                decode_by_prompt = bool(computed >= prompt_len)
                if decode_by_prompt or decode_started:
                    continue
            else:
                # prompt_len 不可用时，使用 decode_started / q_len 作为弱判据（保持旧行为）
                if decode_started or q_len == 1:
                    continue
            is_decode_only = False
            break
    else:
        is_decode_only = False
        has_prefill_by_prompt = False

    # 动态扩展 max_batch_size（如果当前 batch 超过已有值）
    if num_reqs > self.max_batch_size:
        self.max_batch_size = num_reqs

    compact_bootstrap_threshold = self._compact_threshold_tokens()
    if pb is not None:
        _sd_list = [
            compact_bootstrap_threshold > 0 and seq_len > 0 and seq_len <= compact_bootstrap_threshold
            for seq_len in seq_lens_tuple
        ]
        pb.sync_short_dense(_sd_list)
        short_dense_by_row = pb.export_short_dense()
    else:
        short_dense_by_row = tuple(
            (compact_bootstrap_threshold > 0 and seq_len > 0 and seq_len <= compact_bootstrap_threshold)
            for seq_len in seq_lens_tuple
        )

    self.step_meta = StepMeta(
        epoch=self.step_context_epoch,
        batch_size=num_reqs,
        max_batch_size=self.max_batch_size,
        req_ids=req_ids_tuple,
        req_id_to_index=req_id_to_index,
        context_kv_len=seq_lens_tuple,
        seqused_k_gpu=None,  # 延迟构建：在 preheat 开始时构建一次
        recent_cap=recent_cap,
        sink_tokens=validate_sink_tokens(semantic_snapshot.sink_tokens),
        block_size=block_size,
        compact_bootstrap_threshold=compact_bootstrap_threshold,
        bootstrap_done_by_row=pb.export_bootstrap_done() if pb is not None else tuple(bootstrap_done_list),
        q_lens=q_lens,
        is_prefill_by_row=pb.export_is_prefill() if pb is not None else tuple(is_prefill_by_row_list),
        has_prefill_row=has_prefill_row,
        has_decode_row=has_decode_row,
        prefill_rows=pb.get_prefill_rows() if pb is not None else tuple(prefill_rows_list),
        is_decode_only=is_decode_only,
        has_prefill_by_prompt=bool(has_prefill_by_prompt),
        short_dense_by_row=short_dense_by_row,
        decode_plan_version=int(decode_plan_version),
        request_kv_rows=tuple(range(num_reqs)),
    )

    # ── 预计算 canonical kernel hints ──
    if _psc_full_build:
        _use_compact = tuple(m == _ROW_MODE_COMPACT for m in row_mode_by_row_list)
        _dispatch = tuple(
            int(_LOGF_PRODUCER_NONE) if _use_compact[i] else int(logf_producer_by_row_list[i])
            for i in range(num_reqs)
        )
        _needs = tuple(
            int(_dispatch[i]) == int(_LOGF_PRODUCER_ATTN) for i in range(num_reqs)
        )
        _any_needs = any(_needs)
        _refresh_needs_logits = any(
            _needs[i] and layer_effective_refresh_by_row[i]
            for i in range(num_reqs)
        )
        _logits_rows = tuple(
            i for i in range(num_reqs)
            if _needs[i] and logits_last_n_by_row_list[i] > 0
        )
        _logits_rows_gt1 = tuple(
            i for i in _logits_rows if logits_last_n_by_row_list[i] > 1
        )
        _hint_has_log_f = False
        _hint_log_f_eq1 = False
        _hint_log_f_gt1 = False
        _hint_all_compact = all(_use_compact) if num_reqs > 0 else False
        _has_compact_row = any(_use_compact) if num_reqs > 0 else False
        # refresh 计数合并到 hint 循环中，避免独立 for 循环。
        refresh_decode_count = 0
        refresh_non_last_n1_count = 0
        refresh_prefill_count = 0
        for _hi in range(num_reqs):
            _prod = logf_producer_by_row_list[_hi]
            _compact = _use_compact[_hi]
            if _prod == _LOGF_PRODUCER_ATTN and not _compact:
                _ln = logits_last_n_by_row_list[_hi]
                if _ln == 1:
                    _hint_log_f_eq1 = True
                elif _ln > 1:
                    _hint_log_f_gt1 = True
                _hint_has_log_f = True
            # refresh 计数（合并）
            if bool(refresh_row_mask[_hi]):
                if bool(is_prefill_by_row_list[_hi]):
                    refresh_prefill_count += 1
                else:
                    refresh_decode_count += 1
                if int(logits_last_n_by_row_list[_hi]) != 1:
                    refresh_non_last_n1_count += 1
    else:
        _c = _psc_cached
        _use_compact = _c['use_compact']
        _dispatch = _c['dispatch']
        _needs = _c['needs']
        _any_needs = _c['any_needs']
        _refresh_needs_logits = _c['refresh_needs_logits']
        _logits_rows = _c['logits_rows']
        _logits_rows_gt1 = _c['logits_rows_gt1']
        _hint_has_log_f = _c['hint_has_log_f']
        _hint_log_f_eq1 = _c['hint_log_f_eq1']
        _hint_log_f_gt1 = _c['hint_log_f_gt1']
        _hint_all_compact = _c['hint_all_compact']
        _has_compact_row = _c['has_compact_row']
        refresh_decode_count = _c['refresh_decode_count']
        refresh_non_last_n1_count = _c['refresh_non_last_n1_count']
        refresh_prefill_count = _c['refresh_prefill_count']

    has_request_phase_mix_i32 = 1 if has_request_phase_mix else 0
    previous_step_authority = getattr(self, "step_authority", None)
    consume_selected_scope_key, consume_selected_scope_wait_handle = (
        _resolve_consume_selected_scope_binding(
            previous_step_authority=previous_step_authority,
            current_epoch=int(current_epoch),
            target_epoch=int(target_epoch),
        )
    )
    selected_scope_layer_group_id = int(_lg_active) if _lg_enabled and _lg_active >= 0 else 0
    target_selected_scope_key = TargetSelectedScopeKey(
        consumer_step_id=int(self.step_context_epoch),
        layer_group_id=int(selected_scope_layer_group_id),
    )
    expected_scope_layers = tuple(
        layer_idx
        for layer_idx in range(len(self.layer_cache_keys))
        if (not _lg_enabled) or ((int(layer_idx) & 1) == int(selected_scope_layer_group_id))
    )
    selected_scope_wait_handle = allocate_scope_wait_handle(
        target_selected_scope_key,
        expected_layers=expected_scope_layers,
    )
    # ============ 构建 StepAuthority（单源）============
    if _psc_full_build:
        step_authority = StepAuthority(
            epoch=self.step_context_epoch,
            step_handle_id=step_handle.handle_id,
            step_handle_generation=step_handle.generation,
            decode_plan_version=int(decode_plan_version),
            batch_size=num_reqs,
            max_batch_size=self.max_batch_size,
            req_ids=req_ids_tuple,
            req_id_to_index=req_id_to_index,
            q_lens_by_row=q_lens,
            context_kv_len_by_row=seq_lens_tuple,
            q_start_loc=q_start_loc,
            is_prefill_by_row=pb.export_is_prefill() if pb is not None else tuple(is_prefill_by_row_list),
            has_prefill_row=has_prefill_row,
            has_decode_row=has_decode_row,
            prefill_rows=pb.get_prefill_rows() if pb is not None else tuple(prefill_rows_list),
            is_decode_only=is_decode_only,
            has_prefill_by_prompt=bool(has_prefill_by_prompt),
            bootstrap_done_by_row=pb.export_bootstrap_done() if pb is not None else tuple(bootstrap_done_list),
            short_dense_by_row=short_dense_by_row,
            slot_by_row=slot_by_row,
            row_mode_by_row=row_mode_by_row,
            refresh_mode_by_row=refresh_mode_by_row,
            layer_effective_refresh_by_row=layer_effective_refresh_by_row,
            logf_producer_by_row=logf_producer_by_row,
            logf_attn_rows=decode_logf_attn_rows,
            logf_mask_by_row=decode_logf_mask_by_row,
            logf_stride_head=int(decode_logf_stride_head),
            logf_dirty_rows=decode_logf_dirty_rows,
            logits_last_n_by_row=logits_last_n_by_row,
            logits_capacity_by_row=decode_logf_capacity_by_row,
            use_compact_by_row=tuple(_use_compact),
            slot_by_row_has_negative=any(_slot < 0 for _slot in slot_by_row),
            hint_has_log_f=_hint_has_log_f,
            hint_all_compact=_hint_all_compact,
            hint_log_f_eq1=_hint_log_f_eq1,
            hint_log_f_gt1=_hint_log_f_gt1,
            has_compact_row=_has_compact_row,
            recent_cap=recent_cap,
            sink_tokens=validate_sink_tokens(semantic_snapshot.sink_tokens),
            compact_bootstrap_threshold=compact_bootstrap_threshold,
            plan_signature=plan_signature,
            req_set_hash=req_set_hash,
            row_phase_hash=row_phase_hash,
            has_request_phase_mix=bool(has_request_phase_mix),
            has_request_phase_mix_i32=has_request_phase_mix_i32,
            refresh_decode_count=refresh_decode_count,
            refresh_non_last_n1_count=refresh_non_last_n1_count,
            refresh_prefill_count=refresh_prefill_count,
            refresh_slots=_refresh_slots,
            bootstrap_slots=_bootstrap_slots,
            refresh_capture_slot_list=_refresh_capture_slot_list,
            refresh_capture_slot_set=frozenset(_refresh_capture_slot_list),
            consume_selected_scope_key=consume_selected_scope_key,
            consume_selected_scope_wait_handle=consume_selected_scope_wait_handle,
            target_selected_scope_key=target_selected_scope_key,
            selected_scope_wait_handle=selected_scope_wait_handle,
            needs_logits_by_row=_needs,
            any_needs_logits=_any_needs,
            prefill_needs_logits=False,
            refresh_needs_logits=_refresh_needs_logits,
            logits_rows=_logits_rows,
            logits_rows_gt1=_logits_rows_gt1,
            dispatch_logf_producer_by_row=_dispatch,
            has_refresh_row=bool(any(layer_effective_refresh_by_row)),
        )
    else:
        step_authority = _psc_bskip_restamp_authority(
            _psc_cached['step_authority'],
            epoch=self.step_context_epoch,
            handle_id=step_handle.handle_id,
            handle_generation=step_handle.generation,
            decode_plan_version=int(decode_plan_version),
            consume_key=consume_selected_scope_key,
            consume_handle=consume_selected_scope_wait_handle,
            target_key=target_selected_scope_key,
            wait_handle=selected_scope_wait_handle,
            context_kv_len_by_row=seq_lens_tuple,
            q_lens_by_row=q_lens,
        )
    ctx.step_authority = step_authority
    self.step_authority = step_authority
    self.reset_prefill_capture_arena_step_metrics()
    if not (
        (
            (os.environ.get("VLLM_SPARSE_SKIP_DECODE_PREFILL_RESERVE") == "1")
            if _DYNAMIC_ENV
            else _SKIP_DECODE_PREFILL_RESERVE_CACHED
        )
        and bool(step_authority.is_decode_only)
        and not bool(step_authority.has_prefill_row)
    ):
        self.reserve_prefill_capture_bucket(
            step_context=ctx,
            step_authority=step_authority,
            capture_intent=CaptureArenaIntent.ONE_SHOT_BOOTSTRAP,
            slot_by_row=slot_by_row,
        )
    self.step_context = ctx
    self.bind_step_handle_context(step_handle=step_handle, step_context=ctx)
    self._prepared_step_identity = step_identity
    self._prepared_num_actual_tokens = num_actual_tokens
    self._prepared_refresh_nonce = refresh_nonce
    if fa3_step_trace_enabled():
        append_fa3_step_trace(
            build_fa3_step_trace_event(
                step_authority=step_authority,
                step_context=ctx,
                source="prepare_step_context",
            )
        )

    # step profile：记录本步 refresh 规划信息（如果启用）。
    self._step_profile_begin(
        step_meta=self.step_meta,
        refresh_reason=str(refresh_reason),
        refresh_reqs=refresh_reqs,
    )
    # refresh layer-group gating：从 step_envelope_v2 读取（执行链单源）。
    self._refresh_layer_group_epoch = self.step_context_epoch
    self._refresh_layer_group_any_refresh = False
    self._refresh_layer_group_enabled = step_envelope_v2.layer_group_enabled
    self._refresh_layer_group_active = (
        step_envelope_v2.layer_group_active
        if step_envelope_v2.layer_group_active >= 0
        else 0
    )

    self.step_exec_hints = self._build_step_exec_hints(
        batch_size=len(req_ids_tuple),
    )
    for i in range(len(self._step_wait_consumed_token_by_buf)):
        self._step_wait_consumed_token_by_buf[i] = 0

    reset_runtime_step_cursor_state(self, epoch=self.step_context_epoch)
    self._set_unified_attention_mode("default")
    # StepDecodeData 与 step_cache 的预热转移到 Triton metadata builder，
    # 以便使用真实 block_size/seq_lens 构建（避免热路径修正）。

    # === VLLM_SPARSE_PSC_BSKIP: cache-write + assert + hit-counter ===
    if _PSC_BSKIP_ENABLED:
        # Steady-eligible iff this step had no refresh and no logf capture:
        # then B's seq-len-dependent outputs are empty/zero -> reuse is exact
        # while seq_lens grow. (DESIGN NOTE 1.)
        if (
            (not _psc_full_build)
            and _psc_bskip_use
            and not _PSC_BSKIP_ASSERT
            and os.environ.get("VLLM_SPARSE_PSC_BSKIP_FASTHIT") == "1"
        ):
            # HIT: cached objects already restamped in place + steady tuples unchanged
            # => cache is current; skip the 40-key dict rebuild (the bookkeeping that
            # negated B-skip's savings). The tail is `cache=_psc_fresh; return ctx`.
            self._psc_bskip_hits = int(getattr(self, "_psc_bskip_hits", 0)) + 1
            if os.environ.get("VLLM_SPARSE_PSC_BSKIP_HITS_LOG") == "1":
                try:
                    open("logs/bskip_hits.txt","w").write(str(self._psc_bskip_hits))
                except Exception:
                    pass
            return ctx
        _psc_steady = (not bool(step_refresh_nonempty)) and (int(decode_logf_stride_head) == 0 or os.environ.get("VLLM_SPARSE_PSC_BSKIP_LOGF") == "1")
        _psc_fresh = {
            'key': _psc_key,
            'steady_eligible': bool(_psc_steady),
            'row_mode_by_row': row_mode_by_row,
            'row_mode_signature': row_mode_signature,
            'logf_producer_by_row': logf_producer_by_row,
            'needs_logits_by_row': needs_logits_by_row,
            'logits_last_n_by_row': logits_last_n_by_row,
            'layer_effective_refresh_by_row': layer_effective_refresh_by_row,
            'refresh_slots': _refresh_slots,
            'bootstrap_slots': _bootstrap_slots,
            'refresh_capture_slot_list': _refresh_capture_slot_list,
            'plan_signature': plan_signature,
            'lg_enabled': _lg_enabled,
            'lg_active': _lg_active,
            'decode_logf_capacity_by_row': decode_logf_capacity_by_row,
            'decode_logf_q_lens_by_row': decode_logf_q_lens_by_row,
            'decode_logf_mask_by_row': decode_logf_mask_by_row,
            'decode_logf_attn_rows': decode_logf_attn_rows,
            'decode_logf_stride_head': decode_logf_stride_head,
            'decode_logf_dirty_rows': decode_logf_dirty_rows,
            'decode_logf_row_state': decode_logf_row_state,
            'refresh_signals_by_row': refresh_signals_by_row,
            'use_compact': _use_compact,
            'dispatch': _dispatch,
            'needs': _needs,
            'any_needs': _any_needs,
            'refresh_needs_logits': _refresh_needs_logits,
            'logits_rows': _logits_rows,
            'logits_rows_gt1': _logits_rows_gt1,
            'hint_has_log_f': _hint_has_log_f,
            'hint_log_f_eq1': _hint_log_f_eq1,
            'hint_log_f_gt1': _hint_log_f_gt1,
            'hint_all_compact': _hint_all_compact,
            'has_compact_row': _has_compact_row,
            'refresh_decode_count': refresh_decode_count,
            'refresh_non_last_n1_count': refresh_non_last_n1_count,
            'refresh_prefill_count': refresh_prefill_count,
            'step_envelope_v2': step_envelope_v2,
            'ctx': ctx,
            'step_authority': step_authority,
        }
        if _PSC_BSKIP_ASSERT and _psc_bskip_use and _psc_cached is not None:
            # Shadow-assert: cached B outputs must equal this fresh full build.
            _akeys_tuple = (
                'row_mode_by_row', 'row_mode_signature', 'logf_producer_by_row',
                'needs_logits_by_row', 'logits_last_n_by_row',
                'layer_effective_refresh_by_row', 'refresh_slots', 'bootstrap_slots',
                'refresh_capture_slot_list', 'plan_signature',
                'decode_logf_capacity_by_row', 'decode_logf_q_lens_by_row',
                'decode_logf_mask_by_row', 'decode_logf_attn_rows',
                'decode_logf_dirty_rows', 'decode_logf_row_state',
                'refresh_signals_by_row', 'use_compact', 'dispatch', 'needs',
                'logits_rows', 'logits_rows_gt1',
            )
            _ascalar = (
                'lg_enabled', 'lg_active', 'decode_logf_stride_head', 'any_needs',
                'refresh_needs_logits', 'hint_has_log_f', 'hint_log_f_eq1',
                'hint_log_f_gt1', 'hint_all_compact', 'has_compact_row',
                'refresh_decode_count', 'refresh_non_last_n1_count',
                'refresh_prefill_count', 'steady_eligible',
            )
            for _ak in _akeys_tuple + _ascalar:
                if _psc_cached.get(_ak) != _psc_fresh.get(_ak):
                    try:
                        open('logs/bskip_mismatch.txt','a').write('ak:'+_ak+'\n')
                    except Exception:
                        pass
            # Dataclass field-by-field (incl recomputed identity tokens). The
            # cached objects were restamped to CURRENT currency before this
            # fresh build, so compare on the currency-independent fields plus
            # the recomputed identity token.
            _ce = _psc_cached['step_envelope_v2']
            for _f in ('req_ids', 'slot_by_row', 'row_mode_by_row',
                       'refresh_signals', 'refresh_rows', 'refresh_reqs',
                       'layer_effective_refresh_by_row', 'refresh_reason',
                       'bootstrap_done', 'plan_signature', 'layer_group_active',
                       'layer_group_enabled', 'cache_signature'):
                if getattr(_ce, _f) != getattr(step_envelope_v2, _f):
                    try:
                        open('logs/bskip_mismatch.txt','a').write('env:'+_f+'\n')
                    except Exception:
                        pass
            _ca = _psc_cached['step_authority']
            for _f in ('epoch', 'step_handle_id', 'step_handle_generation',
                       'decode_plan_version', 'step_identity_token',
                       'step_fast_identity', 'req_ids', 'slot_by_row',
                       'row_mode_by_row', 'use_compact_by_row', 'plan_signature',
                       'logits_capacity_by_row', 'logf_stride_head',
                       'logf_dirty_rows', 'refresh_slots', 'bootstrap_slots',
                       'needs_logits_by_row', 'dispatch_logf_producer_by_row'):
                if getattr(_ca, _f) != getattr(step_authority, _f):
                    try:
                        open('logs/bskip_mismatch.txt','a').write('auth:'+_f+'\n')
                    except Exception:
                        pass
        if (not _psc_full_build) and _psc_bskip_use:
            self._psc_bskip_hits = int(getattr(self, '_psc_bskip_hits', 0)) + 1
        self._psc_bskip_cache = _psc_fresh
    # === end VLLM_SPARSE_PSC_BSKIP cache-write ===
    return ctx
