from __future__ import annotations

from dataclasses import replace
import json
import logging
import os
import time
from typing import (
    Any,
    Callable,
    Dict,
    List,
    MutableSequence,
    Optional,
    Sequence,
    Set,
    Tuple,
)

_log = logging.getLogger(__name__)

# [T1-FORENSIC 2026-07-09] refresh 链 host 分相取证:目录非空时对每次 refresh
# flush 的 host 段做 cProfile 采样并落 pstats(取证发专用;默认空=零判速税)。
_REFRESH_CPROFILE_DIR = os.environ.get("VLLM_SPARSE_REFRESH_CPROFILE_DIR", "")

import torch
from patches.sparse_constants import (
    _is_free_slot_id,
)
from patches.refresh_runtime.producer_workspace import (
    build_refresh_producer_work_item,
    get_refresh_producer_workspace,
)
from patches.sparse_types import (
    ASYNC_PRODUCER_GPU_PROFILE_STAGES,
    SelectorBatchPayload,
)
from patches.sparse_utils import _submission_slot_owner_snapshot


def _prefill_submission_req_ids_by_slot(
    payloads: Sequence[SelectorBatchPayload],
    slot_list: Sequence[int],
) -> Dict[int, str]:
    """Resolve immutable submission-step request identities for a prefill flush."""
    return dict(
        _submission_slot_owner_snapshot(
            payloads,
            slot_list,
            stage="prefill payload",
        )
    )


def _sentence_trigger_intents_from_refresh_payloads(
    refresh_payloads: Sequence[SelectorBatchPayload] | None,
) -> int:
    if not refresh_payloads:
        return 0
    best_count = 0
    reason_hit = False
    for payload in refresh_payloads:
        reason = str(getattr(payload, "refresh_reason", ""))
        if "sentence" not in reason:
            continue
        reason_hit = True
        intent_req_ids = tuple(
            str(req_id)
            for req_id in (getattr(payload, "refresh_intent_req_ids", ()) or ())
        )
        if intent_req_ids:
            best_count = max(best_count, len(set(intent_req_ids)))
    if best_count > 0:
        return int(best_count)
    return 1 if reason_hit else 0


def _sentence_trigger_intents_for_refresh_profile(
    controller: Any,
    *,
    refresh_payload_count: int,
    refresh_payloads: Sequence[SelectorBatchPayload] | None = None,
) -> int:
    """Return sentence-trigger evidence for the refresh profile record."""

    sentence_trigger_intents_total = int(
        getattr(controller, "_sentence_trigger_intents_total", 0)
    )
    sentence_trigger_intents_prev = int(
        getattr(controller, "_sentence_trigger_intents_profile_emitted", 0)
    )
    sentence_trigger_intents = max(
        0,
        sentence_trigger_intents_total - sentence_trigger_intents_prev,
    )
    controller._sentence_trigger_intents_profile_emitted = (
        sentence_trigger_intents_total
    )
    epoch = int(getattr(controller, "step_context_epoch", -1))
    if sentence_trigger_intents > 0:
        controller._sentence_trigger_materialized_profile_emitted_epoch = epoch
        return int(sentence_trigger_intents)
    if int(refresh_payload_count) <= 0:
        return 0

    materialized_count = int(
        getattr(controller, "_sentence_trigger_materialized_refresh_count", 0)
    )
    materialized_prev = int(
        getattr(controller, "_sentence_trigger_materialized_profile_emitted_count", 0)
    )
    materialized_epoch = int(
        getattr(controller, "_sentence_trigger_materialized_refresh_epoch", -1)
    )
    if materialized_count > materialized_prev and (
        epoch < 0 or materialized_epoch == epoch
    ):
        controller._sentence_trigger_materialized_profile_emitted_count = (
            materialized_count
        )
        controller._sentence_trigger_materialized_profile_emitted_epoch = epoch
        return int(materialized_count - materialized_prev)

    payload_sentence_intents = _sentence_trigger_intents_from_refresh_payloads(
        refresh_payloads
    )
    if payload_sentence_intents > 0:
        last_epoch = int(
            getattr(
                controller,
                "_sentence_trigger_materialized_profile_emitted_epoch",
                -1,
            )
        )
        if epoch >= 0 and last_epoch == epoch:
            return 0
        controller._sentence_trigger_materialized_profile_emitted_epoch = epoch
        return int(payload_sentence_intents)

    step_context = getattr(controller, "step_context", None)
    step_envelope = getattr(step_context, "step_envelope_v2", None)
    reasons = [
        getattr(step_envelope, "refresh_reason", ""),
        getattr(controller, "_step_profile_refresh_reason", ""),
    ]
    reasons.extend(getattr(controller, "_step_profile_req_ticket_pending_reasons", ()) or ())
    if not any("sentence" in str(reason) for reason in reasons):
        return 0

    last_epoch = int(
        getattr(controller, "_sentence_trigger_materialized_profile_emitted_epoch", -1)
    )
    if epoch >= 0 and last_epoch == epoch:
        return 0
    controller._sentence_trigger_materialized_profile_emitted_epoch = epoch
    return 1


def flush_prefill_batches_impl(
    self: Any,
    *,
    buf_id: int,
    chunk_id: int,
    chunk_size: int,
    is_last_layer: bool,
    capture_in_flight: int,
    capture_chunk: int,
    is_stream_capturing_or_raise_fn: Callable[..., bool],
    flush_profile_accum_cls: type,
    refresh_micro_profile_cached: bool,
    refresh_micro_profile_buf: MutableSequence[Any],
    refresh_micro_profile_count: MutableSequence[int],
    flush_micro_profile_summary_fn: Callable[[], None],
    refresh_profile_pending_cls: type,
    make_selector_fast_signature_fn: Callable[..., object],
) -> None:
    _CAPTURE_IN_FLIGHT = capture_in_flight
    _CAPTURE_CHUNK = capture_chunk
    _is_stream_capturing_or_raise = is_stream_capturing_or_raise_fn
    _FlushProfileAccum = flush_profile_accum_cls
    _REFRESH_MICRO_PROFILE_CACHED = refresh_micro_profile_cached
    _REFRESH_MICRO_PROFILE_BUF = refresh_micro_profile_buf
    _REFRESH_MICRO_PROFILE_COUNT = refresh_micro_profile_count
    _flush_micro_profile_summary = flush_micro_profile_summary_fn
    _REFRESH_MICRO_PROFILE_EVERY = 64
    if _REFRESH_MICRO_PROFILE_CACHED:
        try:
            _REFRESH_MICRO_PROFILE_EVERY = max(
                1,
                int(os.environ.get("VLLM_SPARSE_REFRESH_MICRO_PROFILE_EVERY", "64")),
            )
        except ValueError:
            _REFRESH_MICRO_PROFILE_EVERY = 64
    _RefreshProfilePending = refresh_profile_pending_cls
    _make_selector_fast_signature = make_selector_fast_signature_fn
    """Chunk-batched flush.

    - per-chunk：prefill/refresh 都以 SelectorBatchPayload 入队；
      flush 时在 refresh_stream 中执行 selector + fused rebuild（无全局同步）。
    """
    buf = buf_id % _CAPTURE_IN_FLIGHT
    size = max(0, min(chunk_size, _CAPTURE_CHUNK))
    prefill_bucket = self.step_prefill_chunk_payloads[buf]
    refresh_bucket = self.step_refresh_chunk_payloads[buf]
    active_mask = (1 << size) - 1 if size > 0 else 0
    has_prefill_bucket = (self.step_prefill_chunk_mask[buf] & active_mask) != 0
    has_refresh_bucket = (self.step_refresh_chunk_mask[buf] & active_mask) != 0
    if (not has_prefill_bucket) and (not has_refresh_bucket) and (not is_last_layer):
        return

    # --------------------------------------------------------------
    # 1) Prefill：仅 flush 当前 chunk（selector+compact rebuild）
    # --------------------------------------------------------------
    producer_workspace = get_refresh_producer_workspace(self)
    if has_prefill_bucket:
        prefill_payloads, self.step_prefill_chunk_mask[buf] = (
            producer_workspace.drain_payload_bucket(
                kind="prefill",
                bucket=prefill_bucket,
                mask=self.step_prefill_chunk_mask[buf],
                size=size,
            )
        )
    else:
        prefill_payloads = producer_workspace.drain_payload_bucket(
            kind="prefill",
            bucket=prefill_bucket,
            mask=self.step_prefill_chunk_mask[buf],
            size=0,
        )[0]

    # --------------------------------------------------------------
    # 2) Refresh：仅 flush 当前 chunk（selector+compact rebuild）
    # --------------------------------------------------------------
    if has_refresh_bucket:
        refresh_payloads, self.step_refresh_chunk_mask[buf] = (
            producer_workspace.drain_payload_bucket(
                kind="refresh",
                bucket=refresh_bucket,
                mask=self.step_refresh_chunk_mask[buf],
                size=size,
            )
        )
    else:
        refresh_payloads = producer_workspace.drain_payload_bucket(
            kind="refresh",
            bucket=refresh_bucket,
            mask=self.step_refresh_chunk_mask[buf],
            size=0,
        )[0]
    refresh_carrier = producer_workspace.prepare_refresh_carrier(refresh_payloads)
    # layer-group gating：记录本 step 是否真的执行了 refresh（用于在 step 末尾推进 event_idx）。
    if refresh_payloads and self._refresh_layer_group_enabled:
        self._refresh_layer_group_any_refresh = True

    # --------------------------------------------------------------
    # 3) Chunk async pipeline：prefill + refresh 统一放到 refresh_stream
    # --------------------------------------------------------------
    work_payloads = prefill_payloads or refresh_payloads
    # 快照循环必须覆盖两组 payloads:`or` 语义下混合 flush(错峰 bootstrap,
    # prefill 与 refresh 同 flush 共存)时 work_payloads 只含 prefill,refresh
    # payloads 会漏掉长度族/capture 快照(窗口 b)。by-ptr 缓存使并集遍历对共享
    # buffer 零额外 clone。
    snap_payloads = list(prefill_payloads or ()) + list(refresh_payloads or ())
    if work_payloads:
        from patches.fa3_native.capture_ownership import (
            CHUNK_COHORT,
            CaptureOwnershipPlan,
        )

        capture_ownership_plan = getattr(self, "_capture_ownership_plan", None)
        chunk_cohort_stamped = bool(
            isinstance(capture_ownership_plan, CaptureOwnershipPlan)
            and str(capture_ownership_plan.mode) == CHUNK_COHORT
        )
        device = work_payloads[0].capture_scores.device
        # [DETERMINISTIC-CAPTURE-SNAPSHOT 2026-07-02] flush is entered on the
        # decode thread BEFORE the refresh_stream context is created, so the
        # stream current here is the submission-step stream: copies enqueued on
        # it land at the deterministic submission position, serialized against
        # the decode-side capture arena rewrites.
        submission_stream = torch.cuda.current_stream(device=device)
        # [DETERMINISTIC-LEN-SNAPSHOT 2026-07-03] 长度族提交步快照(黄金重锚定批):
        # payload.kv_lengths 是全局 live 长度 buffer 的 expand 视图(每步原位复
        # 写),payload.seq_lens_tensor_cpu 是每步被 capture_live_lengths 原位
        # copy_ 推进的 CPU buffer——deferred selector 读到的是"执行时刻"的实时
        # 长度而非提交步快照,输出随 host 时序漂移(违反 race 合同;旧黄金基线
        # 5b2f4444 建立在该实时读的稳定巧合上,任何 host 路径扰动即漂移)。
        # by-ptr 分组 clone:同一 buffer 的所有层共享同一 snap 对象,保住
        # selector direct-ref 的 data_ptr 相等性判定;prefill/refresh 两相统一
        # 在此收口(子集路径的既有 by-ptr 快照与此叠加无害)。
        _len_snap_by_ptr: Dict[int, torch.Tensor] = {}

        _rs_for_snap = getattr(self, "refresh_stream", None)

        def _snap_len_field(_t: object, *, gpu: bool) -> object:
            if not isinstance(_t, torch.Tensor):
                return _t
            _key = int(_t.data_ptr())
            _snap = _len_snap_by_ptr.get(_key)
            if _snap is None:
                if gpu:
                    with torch.cuda.stream(submission_stream):
                        _snap = _t.clone()
                    # UAF 护栏:snap 在 submission_stream 上分配/写,消费在
                    # refresh_stream(async);payload 释放后 allocator 若在消费
                    # kernel 完成前重用该块给主流即撕裂。refresh_stream 尚未创
                    # 建时本 flush 的消费必在主流,天然无需护栏。
                    if _rs_for_snap is not None:
                        _snap.record_stream(_rs_for_snap)
                else:
                    _snap = _t.clone()
                _len_snap_by_ptr[_key] = _snap
            return _snap

        for _p in snap_payloads:
            _p.kv_lengths = _snap_len_field(_p.kv_lengths, gpu=True)
            _p.kv_len_per_row_i32 = _snap_len_field(
                getattr(_p, "kv_len_per_row_i32", None), gpu=True
            )
            _p.seq_lens_batch = _snap_len_field(
                getattr(_p, "seq_lens_batch", None), gpu=True
            )
            _p.seq_lens_batch_i32 = _snap_len_field(
                getattr(_p, "seq_lens_batch_i32", None), gpu=True
            )
            _p.seq_lens_tensor_cpu = _snap_len_field(
                getattr(_p, "seq_lens_tensor_cpu", None), gpu=False
            )
            # [DETERMINISTIC-BLOCKTABLE-SNAPSHOT 2026-07-03] block_table 的行内
            # 容是主流每步改写的复用 buffer(行轮换时旧 slot 条目被新请求覆写),
            # slot/row 索引同源化只快照了索引、未快照索引所指的表——deferred
            # selector delta kernel 与 compact gather writer 晚读即得脏行。整
            # 表 clone(int32 块索引,bs8×40k 约 80KB)by-ptr 去重后每 flush 一
            # 次,微秒级;保形快照使消费端行号语义不变。
            _p.block_table = _snap_len_field(
                getattr(_p, "block_table", None), gpu=True
            )

        # [RETIRED 2026-07-03 晚] refresh 相 capture/denoms 的提交步快照已下线:
        # 该快照是对"refresh 相 capture 漂移"的过度修复——漂移真根因后来定罪为
        # capture buffer torch.empty 未初始化(kbucket padding 垃圾参与 topk,
        # 已由 [DETERMINISTIC-CAPTURE-ZERO-INIT] 根修)。覆写序本身由既有
        # BUF-keyed 机制保护:refresh 工作提交时登记 buf pending(同 host 线程,
        # 程序序可见),主流 decode 层 dispatch 触同 buf 时经 _compute_wait_decision
        # → _main_stream_wait_for_chunk_done 等待 selector 完成后才覆写(快照
        # 出现前的历史 ×12 全同即该机制的反复实证)。快照的整-storage clone 在
        # 大 ctx/bs 下不可扩展(4B bs8×12k 单份 1.25GiB,OOM 实证),零拷贝的
        # 事件边才是正确形态。长度族快照(上方)保留:长度 buffer 不经 capture
        # ring,无任何 wait 保护,live-read 语义证据独立成立。
        do_profile = self._refresh_profile_should_sample()
        do_async = self._async_refresh_enabled()
        if do_async or do_profile:
            is_capturing = _is_stream_capturing_or_raise(stage="flush_prefill_batches_capture_gate")
            if do_async and is_capturing:
                do_async = False
            if do_profile and is_capturing:
                do_profile = False
        if do_async:
            self._ensure_refresh_stream(device)
            if self.refresh_stream is None or not self.chunk_ready_evt or not self.chunk_done_evt:
                do_async = False
        if chunk_cohort_stamped and not do_async:
            raise RuntimeError(
                "E_SFI_CAPTURE_COHORT_ASYNC_OWNER: stamped chunk cohort lost "
                "its refresh-stream owner"
            )

        # split-wait flags：只在“最终确定异步提交”时置位。
        # 这样可避免 async 被动态关闭时残留 flags（例如 capture 中强制同步）。
        if prefill_payloads:
            self._pending_work_mark_submitted(
                buf_id=buf,
                kind="prefill",
                async_mode=bool(do_async),
                epoch=self.step_context_epoch,
            )
        if refresh_payloads:
            self._pending_work_mark_submitted(
                buf_id=buf,
                kind="refresh",
                async_mode=bool(do_async),
                epoch=self.step_context_epoch,
            )

        # Profiling accumulator：仅在 profiling 活跃时创建实例，
        # steady-state 热路径零开销。
        if do_profile or _REFRESH_MICRO_PROFILE_CACHED:
            prof = _FlushProfileAccum()
        else:
            prof = None
        if do_profile:
            if prefill_payloads:
                prof.prefill_evt0 = torch.cuda.Event(enable_timing=True)
                prof.prefill_evt1 = torch.cuda.Event(enable_timing=True)
            if refresh_payloads:
                prof.refresh_sel_evt0 = torch.cuda.Event(enable_timing=True)
                prof.refresh_sel_evt1 = torch.cuda.Event(enable_timing=True)
                prof.refresh_rebuild_evt0 = torch.cuda.Event(enable_timing=True)
                prof.refresh_rebuild_evt1 = torch.cuda.Event(enable_timing=True)
        writer_pointer_snapshot = (
            self._writer_pointer_telemetry_snapshot() if do_profile else None
        )
        flush_compact_meta_commit_log: Optional[List[dict[str, object]]] = (
            [] if bool(do_async and (prefill_payloads or refresh_payloads)) else None
        )

        def _stage_flush_compact_meta_commit_log() -> None:
            if flush_compact_meta_commit_log is None:
                return
            commit_logs = getattr(self, "_flush_compact_meta_commit_log_by_buf", None)
            target_size = max(int(buf) + 1, len(getattr(self, "chunk_done_evt", ()) or ()))
            if not isinstance(commit_logs, list):
                commit_logs = [[] for _ in range(target_size)]
                self._flush_compact_meta_commit_log_by_buf = commit_logs
            elif len(commit_logs) < target_size:
                commit_logs.extend([] for _ in range(target_size - len(commit_logs)))
            # [FLUSH-META-LOG-QUEUE 2026-07-08] 槽内容从"单轮(覆写)"改为
            # "轮队列(追加)"。旧形态下,同 buf 在上一轮尚未被 step-prep 消费
            # 前再次 stage(环深 2<每世代 3 chunk 的复用节奏下可达)会静默丢
            # 整轮 slot_meta/pad/翻代提交:单代=旧 kv_len 配新内容的 torn
            # 世代;双代=部分层永不翻代(楔死案同族毒源,层间 read_gen 错开)。
            # 不能 merge 成一轮:同层同 slot 双翻会误触 [DUAL-GEN-LAYER-PARITY]
            # 守卫;消费端按轮序分批 commit,GPU 序由"消费前 wait 最新
            # chunk_done"覆盖(更早轮同流更早完成)。
            commit_logs[int(buf)].append(list(flush_compact_meta_commit_log))

        lastn1_direct_count = 0
        gt1_reduce_count = 0
        gt1_scalar_fallback_count = 0
        cohort_private_tape_used = False
        cohort_tape_deferred_producer_used = False
        cohort_snapshot_copy_event = None
        cohort_tape_tokens_by_slot: Dict[int, object] = {}

        def _producer_work_int(name: str, default: int = -1) -> int:
            prof_value = (
                getattr(prof, f"producer_work_{name}", default)
                if prof is not None
                else default
            )
            try:
                prof_int = int(prof_value)
            except (TypeError, ValueError):
                prof_int = int(default)
            if default < 0:
                if prof_int >= 0:
                    return prof_int
            elif prof_int != default:
                return prof_int
            return int(getattr(self, f"_deadline_producer_work_{name}", default))

        def _producer_work_str(name: str) -> str:
            prof_value = (
                str(getattr(prof, f"producer_work_{name}", "") or "")
                if prof is not None
                else ""
            )
            if prof_value:
                return prof_value
            return str(getattr(self, f"_deadline_producer_work_{name}", "") or "")

        def _capture_writer_kernel_variant(accum: _FlushProfileAccum) -> None:
            variant = str(getattr(self, "_last_writer_kernel_variant", "") or "")
            if variant and not accum.writer_kernel_variant:
                accum.writer_kernel_variant = variant
            for field in (
                "writer_actual_tokens",
                "writer_sink_tokens",
                "writer_persist_tokens",
                "writer_sink_io_bytes",
                "writer_persist_io_bytes",
                "writer_token_tiles_estimated",
                "writer_active_token_tiles_estimated",
                "writer_cta_count_estimated",
                "writer_active_cta_count_estimated",
                "writer_tokens_per_cta",
                "writer_k_read_bytes",
                "writer_v_read_bytes",
                "writer_k_write_bytes",
                "writer_v_write_bytes",
                "writer_pos_write_bytes",
                "writer_total_io_bytes",
            ):
                setattr(accum, field, int(getattr(self, f"_last_{field}", 0) or 0))
            accum.writer_effective_io_gbps = float(
                getattr(self, "_last_writer_effective_io_gbps", -1.0) or -1.0
            )

        def _capture_prefill_selector_detail(
            accum: _FlushProfileAccum,
            result: Any,
        ) -> None:
            accum.prefill_gather_evt_pairs.append(
                (result.profile_gather_evt0, result.profile_gather_evt1)
            )
            accum.prefill_key_norms_preproc_evt_pairs.append(
                (
                    result.profile_key_norms_preproc_evt0,
                    result.profile_key_norms_preproc_evt1,
                )
            )
            accum.prefill_key_norms_evt_pairs.append(
                (result.profile_key_norms_evt0, result.profile_key_norms_evt1)
            )
            accum.prefill_key_norms_h2d_evt_pairs.append(
                (result.profile_key_norms_h2d_evt0, result.profile_key_norms_h2d_evt1)
            )
            accum.prefill_key_norms_delta_evt_pairs.append(
                (
                    result.profile_key_norms_delta_evt0,
                    result.profile_key_norms_delta_evt1,
                )
            )
            accum.prefill_key_norms_pack_evt_pairs.append(
                (result.profile_key_norms_pack_evt0, result.profile_key_norms_pack_evt1)
            )
            accum.prefill_key_norms_delta_total_tokens += int(
                result.profile_key_norms_delta_total_tokens or 0
            )
            accum.prefill_key_norms_delta_max_tokens = max(
                int(accum.prefill_key_norms_delta_max_tokens),
                int(result.profile_key_norms_delta_max_tokens or -1),
            )
            accum.prefill_key_norms_delta_layers += int(
                result.profile_key_norms_delta_layers or 0
            )
            accum.prefill_log_s_evt_pairs.append(
                (result.profile_log_s_evt0, result.profile_log_s_evt1)
            )
            accum.prefill_log_s_triton_evt_pairs.append(
                (
                    result.profile_log_s_triton_evt0,
                    result.profile_log_s_triton_evt1,
                )
            )
            accum.prefill_log_s_mask_evt_pairs.append(
                (result.profile_log_s_mask_evt0, result.profile_log_s_mask_evt1)
            )
            accum.prefill_log_s_cross_evt_pairs.append(
                (result.profile_log_s_cross_evt0, result.profile_log_s_cross_evt1)
            )
            accum.prefill_topk_evt_pairs.append(
                (result.profile_topk_evt0, result.profile_topk_evt1)
            )
            accum.prefill_preproc_evt_pairs.append(
                (result.profile_preproc_evt0, result.profile_preproc_evt1)
            )
            accum.prefill_seq_full_evt_pairs.append(
                (result.profile_seq_full_evt0, result.profile_seq_full_evt1)
            )
            accum.prefill_pure_preproc_evt_pairs.append(
                (result.profile_pure_preproc_evt0, result.profile_pure_preproc_evt1)
            )
            accum.prefill_selector_bounds_evt_pairs.append(
                (result.profile_selector_bounds_evt0, result.profile_selector_bounds_evt1)
            )
            accum.prefill_selector_pipeline_evt_pairs.append(
                (
                    result.profile_selector_pipeline_evt0,
                    result.profile_selector_pipeline_evt1,
                )
            )
            accum.prefill_selector_compute_cpu_us += float(
                result.profile_cpu_compute_us or 0.0
            )
            accum.prefill_selector_post_cpu_us += float(
                result.profile_cpu_post_us or 0.0
            )
            accum.prefill_selector_stack_cpu_us += float(
                result.profile_cpu_stack_us or 0.0
            )
            accum.prefill_selector_validate_cpu_us += float(
                result.profile_cpu_validate_us or 0.0
            )
            accum.prefill_selector_key_norms_cpu_us += float(
                result.profile_cpu_key_norms_us or 0.0
            )
            accum.prefill_selector_key_norms_arena_cpu_us += float(
                result.profile_cpu_key_norms_arena_us or 0.0
            )
            accum.prefill_selector_key_norms_direct_cpu_us += float(
                result.profile_cpu_key_norms_direct_us or 0.0
            )
            accum.prefill_selector_key_norms_direct_prepare_cpu_us += float(
                result.profile_cpu_key_norms_direct_prepare_us or 0.0
            )
            accum.prefill_selector_key_norms_direct_launch_cpu_us += float(
                result.profile_cpu_key_norms_direct_launch_us or 0.0
            )
            accum.prefill_selector_key_norms_pack_cpu_us += float(
                result.profile_cpu_key_norms_pack_us or 0.0
            )
            accum.prefill_selector_select_cpu_us += float(
                result.profile_cpu_select_us or 0.0
            )

        def _record_stream_for_capture_bases(
            *payload_groups: Sequence[SelectorBatchPayload],
        ) -> None:
            """确保 capture ring 的底层 storage 在 refresh_stream 完成前不被复用。

            背景：refresh_stream 中的 selector/rebuild 是异步执行的；在 Python 侧退出本函数后，
            payload 及其关联的张量 view 可能被释放并进入缓存分配器的复用池。
            这里通过 record_stream 把底层 storage 的生命周期绑定到 refresh_stream，避免潜在的跨 stream
            UAF/复用竞态（尤其在多 request / 高并发下更容易触发）。
            """
            if not do_async or self.refresh_stream is None:
                return
            if not any(payloads_in for payloads_in in payload_groups):
                return
            record_ptr_set: Set[int] = set()

            def _record(t: Optional[torch.Tensor]) -> None:
                if not isinstance(t, torch.Tensor):
                    return
                try:
                    ptr = int(t.untyped_storage().data_ptr())
                except Exception:
                    _log.warning("record_stream: untyped_storage().data_ptr() failed", exc_info=True)
                    try:
                        ptr = int(t.data_ptr())
                    except Exception:
                        _log.warning("record_stream: data_ptr() fallback also failed", exc_info=True)
                        raise
                if not ptr or ptr in record_ptr_set:
                    return
                try:
                    t.record_stream(self.refresh_stream)
                except Exception:
                    _log.warning("record_stream: t.record_stream() failed", exc_info=True)
                    raise
                record_ptr_set.add(ptr)

            for payloads_in in payload_groups:
                for payload in payloads_in:
                    base_scores = getattr(payload.capture_scores, "_base", None)
                    if isinstance(base_scores, torch.Tensor):
                        _record(base_scores)
                    else:
                        _record(payload.capture_scores if isinstance(payload.capture_scores, torch.Tensor) else None)
                    lastn1_scores = getattr(payload, "lastn1_capture_scores", None)
                    if isinstance(lastn1_scores, torch.Tensor):
                        base_lastn1 = getattr(lastn1_scores, "_base", None)
                        if isinstance(base_lastn1, torch.Tensor):
                            _record(base_lastn1)
                        else:
                            _record(lastn1_scores)

                    base_denoms = (
                        getattr(payload.log_f_denoms, "_base", None)
                        if payload.log_f_denoms is not None
                        else None
                    )
                    if isinstance(base_denoms, torch.Tensor):
                        _record(base_denoms)
                    else:
                        _record(payload.log_f_denoms if isinstance(payload.log_f_denoms, torch.Tensor) else None)

                    # refresh_stream 会读取这些张量；逐 payload 扫描 + ptr 去重可覆盖双源 payload 差异。
                    _record(payload.seq_lens_batch)
                    _record(getattr(payload, "seq_lens_batch_i32", None))
                    _record(payload.kv_lengths)
                    _record(payload.kv_len_per_row_i32)
                    _record(payload.row_tensor_i32)
                    _record(payload.row_tensor)
                    _record(payload.refresh_rows_long)
                    _record(payload.refresh_block_table_sub)
                    _record(payload.refresh_seq_lens_i32)
                    _record(payload.slot_tensor)
                    _record(payload.slot_tensor_i32)
                    _record(payload.cu_seqlens_q)
                    _record(payload.alibi_slopes)
                    _record(payload.k_descale)
                    _record(payload.block_table)

        def _record_refresh_producer_work_metadata(
            *,
            admission_reason: str,
            can_drop: bool,
            can_coalesce: bool,
        ) -> None:
            if not do_profile or not refresh_payloads:
                return
            layer_cache_keys = getattr(self, "layer_cache_keys", ())
            if layer_cache_keys:
                num_chunks = (len(layer_cache_keys) + _CAPTURE_CHUNK - 1) // _CAPTURE_CHUNK
            else:
                num_chunks = 1
            max_delay = self._refresh_rebuild_max_delay_steps(num_chunks)
            request_states = getattr(self, "request_states", None)
            step_ctx_for_handle = getattr(self, "step_context", None)
            step_authority_for_handle = getattr(self, "step_authority", None)
            if step_authority_for_handle is None and step_ctx_for_handle is not None:
                step_authority_for_handle = getattr(
                    step_ctx_for_handle, "step_authority", None
                )
            current_handle_id = int(
                getattr(
                    step_authority_for_handle,
                    "step_handle_id",
                    getattr(step_ctx_for_handle, "step_handle_id", -1),
                )
                or -1
            )
            work_item = build_refresh_producer_work_item(
                payloads=refresh_payloads,
                request_states=request_states if isinstance(request_states, dict) else None,
                layer_indices=refresh_carrier.layer_indices,
                max_delay_steps=int(max_delay),
                current_epoch=int(getattr(self, "step_context_epoch", -1)),
                current_handle_id=int(current_handle_id),
                admission_reason=str(admission_reason or ""),
                can_drop=bool(can_drop),
                can_coalesce=bool(can_coalesce),
            )
            work_item.apply_to_profile(prof)

        def _publish_refresh_writer(
            refresh_result: Any,
            *,
            profile_accum: Optional[Any] = None,
        ) -> bool:
            fused_ok = self._rebuild_compact_slots_batched_layers_from_selection(
                refresh_payloads,
                refresh_result.selected_indices,
                phase="refresh",
                bootstrap_slots_by_layer=refresh_carrier.bootstrap_slots_by_layer,
                defer_compact_meta_publish=flush_compact_meta_commit_log is not None,
                compact_meta_commit_log=flush_compact_meta_commit_log,
            )
            if profile_accum is not None:
                _capture_writer_kernel_variant(profile_accum)
            return bool(fused_ok)

        def _run_refresh_body() -> None:
            if not refresh_payloads:
                return
            if do_profile:
                t_total0_ns = time.perf_counter_ns()
                if prof.refresh_sel_evt0 is not None:
                    prof.refresh_sel_evt0.record(torch.cuda.current_stream(device=device))
                t_sel0_ns = time.perf_counter_ns()
                refresh_result = self._apply_alpha_selector_batched_fused(
                    refresh_payloads,
                    phase="decode",
                    update_tracking=True,
                )
                t_sel1_ns = time.perf_counter_ns()
                prof.refresh_selector_cpu_us = (t_sel1_ns - t_sel0_ns) / 1000.0
                prof.refresh_selector_apply_cpu_us = float(prof.refresh_selector_cpu_us)
                if _REFRESH_MICRO_PROFILE_CACHED:
                    prof.micro_selector_ns = t_sel1_ns - t_sel0_ns
                if prof.refresh_sel_evt1 is not None:
                    prof.refresh_sel_evt1.record(torch.cuda.current_stream(device=device))
                if refresh_result is None:
                    prof.refresh_total_cpu_us = (time.perf_counter_ns() - t_total0_ns) / 1000.0
                    return
                selector_done_ns = t_sel1_ns
                # 仅在 profiling detail 中做集合重合度采样（避免影响热路径）。
                if self._refresh_profile_detail_enabled():
                    try:
                        p0 = refresh_payloads[0]
                        seq_lens_cpu = getattr(p0, "seq_lens_cpu", None)
                        slot_list_local = list(getattr(p0, "slot_list", []) or [])
                        if seq_lens_cpu is not None and slot_list_local:
                            threshold = int(self._compact_threshold_tokens())
                            pick_idx = -1
                            for i, seq_len in enumerate(seq_lens_cpu):
                                seq_len_i = int(seq_len)
                                if threshold <= 0 or seq_len_i >= threshold:
                                    pick_idx = int(i)
                                    break
                            if pick_idx >= 0 and pick_idx < len(slot_list_local):
                                slot = int(slot_list_local[pick_idx])
                                st0 = p0.state
                                if 0 <= slot < len(st0.compact_pos):
                                    sink_len_old = (
                                        int(st0.compact_sink_len[slot])
                                        if slot < len(st0.compact_sink_len)
                                        else 0
                                    )
                                    persist_len_old = (
                                        int(st0.compact_persist_len[slot])
                                        if slot < len(st0.compact_persist_len)
                                        else 0
                                    )
                                    old_pos_slot = st0.compact_pos[slot]
                                    if old_pos_slot.dim() < 2 or old_pos_slot.shape[0] <= 0:
                                        old_pos_head0 = old_pos_slot.new_empty((0,), dtype=old_pos_slot.dtype)
                                    else:
                                        old_pos_head0 = old_pos_slot[0]
                                    if old_pos_head0.numel() > 0:
                                        if (
                                            persist_len_old > 0
                                            and old_pos_head0.numel() >= sink_len_old + persist_len_old
                                        ):
                                            old_persist = old_pos_head0.narrow(0, sink_len_old, persist_len_old)
                                        else:
                                            old_persist = old_pos_head0[sink_len_old:]
                                        if old_persist.numel() > 0:
                                            old_persist = old_persist[old_persist >= 0]
                                    else:
                                        old_persist = old_pos_head0
                                    new_tokens = refresh_result.selected_indices[0, pick_idx, 0, :]
                                    if old_persist.numel() > 0 and new_tokens.numel() > 0:
                                        if old_persist.dtype != new_tokens.dtype:
                                            old_persist = old_persist.to(dtype=new_tokens.dtype)
                                        # Rev 2 (2026-04-23): keep as GPU int
                                        # scalar tensor; .item() only at the
                                        # profiler JSON emit boundary (L1065/L1165
                                        # below). Eliminates per-overlap-sample
                                        # sync from this refresh-path location.
                                        overlap_cnt_tensor = torch.isin(new_tokens, old_persist).sum()
                                        prof.refresh_overlap_new_k = int(new_tokens.numel())
                                        prof.refresh_overlap_old_k = int(old_persist.numel())
                                        prof.refresh_overlap_count_tensor = overlap_cnt_tensor
                                        # Ratio computed from tensor at emit.
                                        prof.refresh_overlap_ratio = None
                    except Exception:
                        _log.warning("refresh profiling: overlap ratio sampling failed", exc_info=True)
                        # detail 采样失败不应影响主流程；仅保留 warning
                prof.refresh_gather_evt0 = refresh_result.profile_gather_evt0
                prof.refresh_gather_evt1 = refresh_result.profile_gather_evt1
                prof.refresh_selector_compute_cpu_us = float(refresh_result.profile_cpu_compute_us or 0.0)
                prof.refresh_selector_post_cpu_us = float(refresh_result.profile_cpu_post_us or 0.0)
                prof.refresh_selector_stack_cpu_us = float(refresh_result.profile_cpu_stack_us or 0.0)
                prof.refresh_selector_key_norms_cpu_us = float(
                    refresh_result.profile_cpu_key_norms_us or 0.0
                )
                prof.refresh_selector_key_norms_arena_cpu_us = float(
                    refresh_result.profile_cpu_key_norms_arena_us or 0.0
                )
                prof.refresh_selector_key_norms_direct_cpu_us = float(
                    refresh_result.profile_cpu_key_norms_direct_us or 0.0
                )
                prof.refresh_selector_key_norms_direct_prepare_cpu_us = float(
                    refresh_result.profile_cpu_key_norms_direct_prepare_us or 0.0
                )
                prof.refresh_selector_key_norms_direct_launch_cpu_us = float(
                    refresh_result.profile_cpu_key_norms_direct_launch_us or 0.0
                )
                prof.refresh_selector_key_norms_pack_cpu_us = float(
                    refresh_result.profile_cpu_key_norms_pack_us or 0.0
                )
                prof.refresh_key_norms_preproc_evt0 = refresh_result.profile_key_norms_preproc_evt0
                prof.refresh_key_norms_preproc_evt1 = refresh_result.profile_key_norms_preproc_evt1
                prof.refresh_key_norms_evt0 = refresh_result.profile_key_norms_evt0
                prof.refresh_key_norms_evt1 = refresh_result.profile_key_norms_evt1
                prof.refresh_key_norms_h2d_evt0 = refresh_result.profile_key_norms_h2d_evt0
                prof.refresh_key_norms_h2d_evt1 = refresh_result.profile_key_norms_h2d_evt1
                prof.refresh_key_norms_delta_evt0 = refresh_result.profile_key_norms_delta_evt0
                prof.refresh_key_norms_delta_evt1 = refresh_result.profile_key_norms_delta_evt1
                prof.refresh_key_norms_pack_evt0 = refresh_result.profile_key_norms_pack_evt0
                prof.refresh_key_norms_pack_evt1 = refresh_result.profile_key_norms_pack_evt1
                prof.refresh_key_norms_delta_total_tokens = int(
                    refresh_result.profile_key_norms_delta_total_tokens or 0
                )
                prof.refresh_key_norms_delta_max_tokens = int(
                    refresh_result.profile_key_norms_delta_max_tokens or -1
                )
                prof.refresh_key_norms_delta_layers = int(
                    refresh_result.profile_key_norms_delta_layers or 0
                )
                prof.refresh_log_s_evt0 = refresh_result.profile_log_s_evt0
                prof.refresh_log_s_evt1 = refresh_result.profile_log_s_evt1
                prof.refresh_log_s_triton_evt0 = refresh_result.profile_log_s_triton_evt0
                prof.refresh_log_s_triton_evt1 = refresh_result.profile_log_s_triton_evt1
                prof.refresh_log_s_mask_evt0 = refresh_result.profile_log_s_mask_evt0
                prof.refresh_log_s_mask_evt1 = refresh_result.profile_log_s_mask_evt1
                prof.refresh_log_s_cross_evt0 = refresh_result.profile_log_s_cross_evt0
                prof.refresh_log_s_cross_evt1 = refresh_result.profile_log_s_cross_evt1
                prof.refresh_topk_evt0 = refresh_result.profile_topk_evt0
                prof.refresh_topk_evt1 = refresh_result.profile_topk_evt1
                prof.refresh_preproc_evt0 = refresh_result.profile_preproc_evt0
                prof.refresh_preproc_evt1 = refresh_result.profile_preproc_evt1
                prof.refresh_seq_full_evt0 = refresh_result.profile_seq_full_evt0
                prof.refresh_seq_full_evt1 = refresh_result.profile_seq_full_evt1
                prof.refresh_pure_preproc_evt0 = refresh_result.profile_pure_preproc_evt0
                prof.refresh_pure_preproc_evt1 = refresh_result.profile_pure_preproc_evt1
                prof.refresh_selector_bounds_evt0 = (
                    refresh_result.profile_selector_bounds_evt0
                )
                prof.refresh_selector_bounds_evt1 = (
                    refresh_result.profile_selector_bounds_evt1
                )
                prof.refresh_selector_pipeline_evt0 = (
                    refresh_result.profile_selector_pipeline_evt0
                )
                prof.refresh_selector_pipeline_evt1 = (
                    refresh_result.profile_selector_pipeline_evt1
                )
                # Keep profiling-only detail sampling and bookkeeping out of
                # the selector->writer boundary attribution. This boundary is
                # meant to classify publish/enqueue/rebuild exposure, not
                # optional overlap diagnostics.
                selector_done_ns = time.perf_counter_ns()
                _record_refresh_producer_work_metadata(
                    admission_reason="inline_refresh",
                    can_drop=True,
                    can_coalesce=True,
                )
                if prof.refresh_rebuild_evt0 is not None:
                    prof.refresh_rebuild_evt0.record(torch.cuda.current_stream(device=device))
                t_rebuild0_ns = time.perf_counter_ns()
                try:
                    p0 = refresh_payloads[0]
                    prof.rebuild_head_dim = int(p0.key_cache.shape[-1]) if p0.key_cache is not None else 0
                    prof.rebuild_kv_dtype = str(p0.key_cache.dtype) if p0.key_cache is not None else ""
                    prof.rebuild_block_size = (
                        int(p0.key_cache.shape[1]) if p0.key_cache is not None and p0.key_cache.dim() >= 2 else 0
                    )
                    prof.rebuild_num_kv_heads = int(getattr(p0.state, "num_kv_heads", 0) or 0) if p0 is not None else 0
                    prof.rebuild_batch_slots = int(len(getattr(p0, "slot_list", []) or []))
                    prof.capture_kv_len_total = int(p0.capture_scores.shape[-1]) if p0.capture_scores is not None else 0
                    prof.rebuild_stride_tokens = (
                        int(self._compact_stride_tokens(int(prof.rebuild_block_size))) if int(prof.rebuild_block_size) > 0 else 0
                    )
                except Exception:
                    _log.warning("refresh profiling: rebuild metadata extraction failed", exc_info=True)
                    prof.rebuild_head_dim = 0
                    prof.rebuild_kv_dtype = ""
                    prof.rebuild_block_size = 0
                    prof.rebuild_num_kv_heads = 0
                    prof.rebuild_batch_slots = 0
                    prof.capture_kv_len_total = 0
                    prof.rebuild_stride_tokens = 0
                fused_ok = _publish_refresh_writer(
                    refresh_result,
                    profile_accum=prof,
                )
                prof.selector_writer_boundary_cpu_us += (
                    time.perf_counter_ns() - selector_done_ns
                ) / 1000.0
                try:
                    prof.rebuild_selected_k = int(refresh_result.selected_indices.shape[-1])
                except Exception:
                    _log.warning("refresh profiling: rebuild_selected_k extraction failed", exc_info=True)
                    prof.rebuild_selected_k = 0
                t_rebuild1_ns = time.perf_counter_ns()
                prof.refresh_rebuild_cpu_us = (t_rebuild1_ns - t_rebuild0_ns) / 1000.0
                if _REFRESH_MICRO_PROFILE_CACHED:
                    prof.micro_rebuild_ns = t_rebuild1_ns - t_rebuild0_ns
                if prof.refresh_rebuild_evt1 is not None:
                    prof.refresh_rebuild_evt1.record(torch.cuda.current_stream(device=device))
                if not fused_ok:
                    raise RuntimeError("refresh compact rebuild: fused gather failed")
                prof.refresh_total_cpu_us = (time.perf_counter_ns() - t_total0_ns) / 1000.0
                return

            # do_profile==False：保持热路径最小开销（不创建/读 timing events，不做任何 synchronize）
            _micro = _REFRESH_MICRO_PROFILE_CACHED
            if _micro:
                _mt0 = time.perf_counter_ns()
            refresh_result = self._apply_alpha_selector_batched_fused(
                refresh_payloads,
                phase="decode",
                update_tracking=True,
            )
            if _micro:
                _mt1 = time.perf_counter_ns()
            if refresh_result is None:
                if _micro:
                    prof.micro_selector_ns = _mt1 - _mt0
                return
            fused_ok = _publish_refresh_writer(
                refresh_result,
                profile_accum=prof,
            )
            if _micro:
                _mt3 = time.perf_counter_ns()
                prof.micro_selector_ns = _mt1 - _mt0
                prof.micro_rebuild_ns = _mt3 - _mt1
            if not fused_ok:
                raise RuntimeError("refresh compact rebuild: fused gather failed")

        def _run_refresh() -> None:
            # [T1-FORENSIC 2026-07-09] 取证发专用 host 分相采样(默认目录空,
            # 走 else 原路径零开销);cProfile 扭曲判速,取证发与判速发分离。
            if not (_REFRESH_CPROFILE_DIR and refresh_payloads):
                _run_refresh_body()
                return
            import cProfile

            profiler = cProfile.Profile()
            profiler.enable()
            try:
                _run_refresh_body()
            finally:
                profiler.disable()
                profiler.dump_stats(
                    os.path.join(
                        _REFRESH_CPROFILE_DIR,
                        f"refresh_host_{os.getpid()}_{time.time_ns()}.pstats",
                    )
                )

        def _run_prefill() -> None:
            if not prefill_payloads:
                return
            t0_ns: Optional[int] = None
            if do_profile:
                if prof.prefill_evt0 is not None:
                    prof.prefill_evt0.record(torch.cuda.current_stream(device=device))
                t0_ns = time.perf_counter_ns()
            first = prefill_payloads[0]
            slot_list_full: Tuple[int, ...]
            if isinstance(first.slot_list, tuple):
                slot_list_full = first.slot_list
            else:
                slot_list_full = tuple(int(s) for s in first.slot_list)
            if not slot_list_full:
                if do_profile and prof.prefill_evt1 is not None:
                    prof.prefill_evt1.record(torch.cuda.current_stream(device=device))
                return

            # 重要：按“有效 last_n”分组（已被 q_len clamp），避免 prefill 最后一步
            # q_len==1 时仍错误走 log_f_pre+denom。
            plan_by_req = self.step_prefill_capture_plan_by_req or {}
            finalize_req_ids = tuple(
                str(rid)
                for rid in (getattr(self, "step_prefill_finalize_req_ids", tuple()) or tuple())
                if rid
            )
            finalize_req_id_set = set(finalize_req_ids)
            finalize_slot_set: set[int] = set()
            submission_req_id_by_slot = _prefill_submission_req_ids_by_slot(
                prefill_payloads,
                slot_list_full,
            )
            if finalize_req_id_set:
                for slot_i in slot_list_full:
                    req_id = submission_req_id_by_slot.get(int(slot_i))
                    if (
                        req_id in finalize_req_id_set
                        and (not _is_free_slot_id(req_id))
                    ):
                        finalize_slot_set.add(int(slot_i))
            if bool(getattr(getattr(self, "config", None), "one_shot_bootstrap_only", False)):
                selector_slot_list = tuple(sorted(finalize_slot_set))
            else:
                selector_slot_list = slot_list_full
            logits_last_n_by_row: tuple[int, ...] = tuple()
            q_start_loc_bound: tuple[int, ...] = tuple()
            if self.step_context is not None:
                step_bound_meta = self._require_step_bound_meta(
                    step_context=self.step_context,
                    stage="flush prefill",
                )
                logits_last_n_by_row = tuple(
                    int(v) for v in step_bound_meta.logits_last_n_by_row
                )
                q_start_loc_bound = tuple(int(v) for v in step_bound_meta.q_start_loc)
            slots_lastn1: List[int] = []
            slots_lastn_gt1: List[int] = []
            row_list_full: Tuple[int, ...] = (
                tuple(int(r) for r in first.row_list)
                if first.row_list is not None
                else tuple()
            )
            slot_pos_by_slot = {int(s): idx for idx, s in enumerate(slot_list_full)}
            for s in selector_slot_list:
                slot_i = int(s)
                idx = slot_pos_by_slot.get(slot_i, -1)
                if idx < 0:
                    continue
                ln = 0
                if idx < len(row_list_full) and logits_last_n_by_row:
                    row = int(row_list_full[idx])
                    if 0 <= row < len(logits_last_n_by_row):
                        ln = int(logits_last_n_by_row[row])
                if ln <= 0:
                    req_id = submission_req_id_by_slot.get(slot_i)
                    if req_id is None or _is_free_slot_id(req_id):
                        continue
                    ln = int(plan_by_req.get(str(req_id), 0) or 0)
                    # Bug-5 fix: plan_by_req 可能包含过期值（基于旧 q_len），
                    # 必须用当前 step_context 的 q_len 重新 clamp。
                    if ln > 1 and self.step_context is not None:
                        if q_start_loc_bound and idx < len(row_list_full):
                            row = int(row_list_full[idx])
                            if row + 1 < len(q_start_loc_bound):
                                q_len = int(q_start_loc_bound[row + 1]) - int(
                                    q_start_loc_bound[row]
                                )
                                if q_len > 0:
                                    ln = min(ln, q_len)
                if ln == 1:
                    slots_lastn1.append(slot_i)
                elif ln > 1:
                    slots_lastn_gt1.append(slot_i)

            # subset 的 slot→pos 映射在同一 chunk 内可复用（最多拆 2 组），避免重复构造 dict
            slot_list_ref = slot_list_full
            pos_map = {int(s): idx for idx, s in enumerate(slot_list_ref)}
            if chunk_cohort_stamped:
                nonlocal cohort_private_tape_used
                nonlocal cohort_snapshot_copy_event
                nonlocal cohort_tape_tokens_by_slot
                if not bool(
                    getattr(
                        getattr(self, "config", None),
                        "one_shot_bootstrap_only",
                        False,
                    )
                ):
                    raise RuntimeError(
                        "E_SFI_CAPTURE_COHORT_FINALIZE: chunk cohort requires "
                        "one-shot finalization"
                    )
                if not finalize_slot_set:
                    return
                from patches.fa3_native.capture_cohort_tape import (
                    CaptureCohortTapeLeaseTracker,
                    CaptureCohortTapeState,
                    capture_cohort_consumer_token,
                    capture_cohort_tape_owner_key,
                    require_capture_cohort_consumer_coverage,
                    snapshot_capture_cohort_payload_group,
                )

                tape_state = getattr(self, "_capture_cohort_tape_state", None)
                tape_tracker = getattr(
                    self, "_capture_cohort_tape_lease_tracker", None
                )
                if not isinstance(tape_state, CaptureCohortTapeState) or not isinstance(
                    tape_tracker, CaptureCohortTapeLeaseTracker
                ):
                    raise RuntimeError(
                        "E_SFI_CAPTURE_COHORT_TAPE_PREBUILD: stamped chunk cohort "
                        "has no prebuilt tape state"
                    )
                if (
                    str(tape_state.plan_signature)
                    != str(capture_ownership_plan.signature_sha256)
                    or int(tape_state.report.logical_k_capacity)
                    != int(first.capture_scores.shape[-1])
                ):
                    raise RuntimeError(
                        "E_SFI_CAPTURE_COHORT_FIXED_K: live payload disagrees "
                        "with the immutable tape plan"
                    )
                owner_key = capture_cohort_tape_owner_key(
                    tape_state, prefill_payloads
                )
                # Registering a request token without a corresponding deferred
                # producer would poison the bounded generation bank until a
                # later request reports BANKS_EXHAUSTED.  Prove the one-to-one
                # consumer set before snapshot acquires any bank ownership.
                registered_consumer_slots = (
                    require_capture_cohort_consumer_coverage(
                        registered_slots=tuple(sorted(finalize_slot_set)),
                        producer_slots=tuple(slots_lastn1) + tuple(slots_lastn_gt1),
                    )
                )
                cohort_tape_tokens_by_slot = {}
                for slot_i in registered_consumer_slots:
                    if slot_i not in submission_req_id_by_slot:
                        raise RuntimeError(
                            "E_SFI_CAPTURE_COHORT_CONSUMER: finalize slot is "
                            "outside the submission-step payload"
                        )
                    request_id = submission_req_id_by_slot[slot_i]
                    if request_id is None or _is_free_slot_id(request_id):
                        raise RuntimeError(
                            "E_SFI_CAPTURE_COHORT_CONSUMER: finalize slot has "
                            "no submission-step request identity"
                        )
                    cohort_tape_tokens_by_slot[slot_i] = (
                        capture_cohort_consumer_token(
                            owner_key,
                            slot=slot_i,
                            request_id=str(request_id),
                        )
                    )
                lastn1_row_positions = tuple(
                    sorted(pos_map[int(slot)] for slot in slots_lastn1)
                )
                # Finalization is the single source boundary: drain any still
                # pending postprocess jobs on this same refresh stream before
                # snapshotting.  Completed early-cohort jobs are deduplicated
                # no-ops; no wait is submitted against an unrecorded event.
                from patches.fa3_native.postprocess import (
                    run_capture_postprocess_jobs_for_payloads,
                )

                run_capture_postprocess_jobs_for_payloads(
                    prefill_payloads,
                    meta_cache_owner=self,
                )
                cohort_snapshot = snapshot_capture_cohort_payload_group(
                    state=tape_state,
                    tracker=tape_tracker,
                    payloads=prefill_payloads,
                    consumer_tokens=tuple(cohort_tape_tokens_by_slot.values()),
                    lastn1_row_positions=lastn1_row_positions,
                    # The prebuilt tape owns a generation bank on the dedicated
                    # refresh stream.  CUDAGraph execution can make the ambient
                    # current stream differ even inside this flush, so submitting
                    # the snapshot there violates the tape lease contract.
                    stream=self.refresh_stream,
                )
                cohort_snapshot_copy_event = (
                    cohort_snapshot.copy_completion_event
                )
                fence = getattr(self, "_ring_war_fence", None)
                on_reduce = getattr(fence, "on_reduce", None)
                if not callable(on_reduce):
                    raise RuntimeError(
                        "E_SFI_CAPTURE_COHORT_FENCE: snapshot publication has "
                        "no scratch WAR owner"
                    )
                cohort_size = int(capture_ownership_plan.cohort_size)
                in_flight = int(capture_ownership_plan.in_flight)
                for global_layer in cohort_snapshot.group.layer_slots:
                    cohort_origin = (
                        int(global_layer) // cohort_size
                    ) * cohort_size
                    cohort_lane = (
                        cohort_origin // cohort_size
                    ) % in_flight
                    scratch_slot = (
                        cohort_lane * cohort_size
                        + int(global_layer) % cohort_size
                    )
                    on_reduce(
                        int(scratch_slot),
                        cohort_snapshot_copy_event,
                        True,
                    )
                cohort_private_tape_used = True
            # [DETERMINISTIC-CAPTURE-SNAPSHOT 2026-07-02] one private tape per
            # flush per source kind (True=capture+denoms, False=lastn1), shared
            # by every subset of this flush so all per-slot payloads and the
            # postprocess retarget agree on a single buffer.
            subset_tape_cache: Dict[object, object] = {}
            retargeted_postprocess_job_ids: set = set()

            def _subset_payloads_for_slots(
                payloads_in: Sequence[SelectorBatchPayload],
                slots_use: Sequence[int],
                *,
                use_denoms: bool,
            ) -> List[SelectorBatchPayload]:
                if not slots_use:
                    return []
                first_in = payloads_in[0]

                slots_use_values = [int(s) for s in slots_use]
                if not slots_use_values:
                    return []
                try:
                    idx_list = [pos_map[s] for s in slots_use_values]
                except KeyError as exc:
                    raise RuntimeError(
                        f"prefill subset: slot {int(exc.args[0])} missing from payload slot_list"
                    ) from None
                if not idx_list:
                    return []

                kv_len_total = int(first_in.capture_scores.shape[-1])
                batch_sub = len(idx_list)

                start = idx_list[0]
                end = idx_list[-1]
                contiguous = (end - start + 1) == int(batch_sub)
                slots_use_ordered: Sequence[int] = slot_list_ref[start : start + batch_sub]
                if not contiguous:
                    slots_use_sorted = sorted(slots_use_values)
                    try:
                        idx_list = [pos_map[s] for s in slots_use_sorted]
                    except KeyError as exc:
                        raise RuntimeError(
                            f"prefill subset: slot {int(exc.args[0])} missing from payload slot_list"
                        ) from None
                    if not idx_list:
                        return []
                    start = idx_list[0]
                    end = idx_list[-1]
                    contiguous = (end - start + 1) == int(batch_sub)
                    slots_use_ordered = slots_use_sorted
                if not contiguous:
                    raise RuntimeError(
                        "prefill subset requires contiguous slot positions; "
                        "split last_n groups before selector"
                    )
                # [SUBSET-CONTIGUOUS-ONLY 2026-07-02] past the guard above every
                # subset is a contiguous [start, start+batch_sub) slice; the old
                # index_select/non-contiguous machinery was structurally dead (the
                # guard raises first) and incompatible with the tape slicing below,
                # so it was removed. A future non-contiguous subset needs a fresh
                # design (tape rows included), not a revert of that code.

                def _slice_payload_batch_dim(tensor: torch.Tensor, *, label: str, expected_dim: int) -> torch.Tensor:
                    if not isinstance(tensor, torch.Tensor) or tensor.dim() != expected_dim:
                        raise RuntimeError(f"prefill subset: missing {label}")
                    return tensor[start : start + batch_sub]

                # [DETERMINISTIC-CAPTURE-SNAPSHOT 2026-07-02] length-tensor snapshots.
                # layout.kv_len_per_row_i32 is refreshed each step via
                # cached_sequence_to_device(..., out=<same buffer>) and
                # layout.kv_lengths is an expand view of the same live length
                # source (capture_live_lengths.py), so the payload views float
                # while the deferred producer runs -- the selector's per-head
                # topk bounds then depend on WHEN it executed. Clone every
                # length tensor ONCE per flush on the submission stream (a few
                # KB, bootstrap-only) so all subsets see submission-step values.
                len_snap = subset_tape_cache.get("len_snap")
                if len_snap is None and chunk_cohort_stamped:
                    _sl_snap = first_in.seq_lens_batch
                    _sl32_snap = getattr(first_in, "seq_lens_batch_i32", None)
                    _kvpr_snap = first_in.kv_len_per_row_i32
                    if not all(
                        isinstance(tensor, torch.Tensor)
                        for tensor in (_sl_snap, _sl32_snap, _kvpr_snap)
                    ):
                        raise RuntimeError(
                            "E_SFI_CAPTURE_COHORT_METADATA: stamped payload "
                            "lost its submission-step length snapshots"
                        )
                    _kv_ptr_snaps = {}
                    for _p in payloads_in:
                        _kv = _p.kv_lengths
                        if not isinstance(_kv, torch.Tensor):
                            raise RuntimeError(
                                "E_SFI_CAPTURE_COHORT_METADATA: stamped payload "
                                "lost kv_lengths"
                            )
                        _kv_ptr_snaps[int(_kv.data_ptr())] = _kv
                    len_snap = (
                        _sl_snap,
                        _sl32_snap,
                        _kvpr_snap,
                        _kv_ptr_snaps,
                    )
                    subset_tape_cache["len_snap"] = len_snap
                if len_snap is None:
                    _kv_ptr_snaps: Dict[int, torch.Tensor] = {}
                    with torch.cuda.stream(submission_stream):
                        _sl_snap = (
                            first_in.seq_lens_batch.clone()
                            if isinstance(first_in.seq_lens_batch, torch.Tensor)
                            else first_in.seq_lens_batch
                        )
                        _sl32_src = getattr(first_in, "seq_lens_batch_i32", None)
                        _sl32_snap = (
                            _sl32_src.clone()
                            if isinstance(_sl32_src, torch.Tensor)
                            else _sl32_src
                        )
                        _kvpr_src = first_in.kv_len_per_row_i32
                        _kvpr_snap = (
                            _kvpr_src.clone()
                            if isinstance(_kvpr_src, torch.Tensor)
                            else _kvpr_src
                        )
                        for _p in payloads_in:
                            _kv = _p.kv_lengths
                            if (
                                isinstance(_kv, torch.Tensor)
                                and int(_kv.data_ptr()) not in _kv_ptr_snaps
                            ):
                                _kv_ptr_snaps[int(_kv.data_ptr())] = _kv.clone()
                    _len_consume = torch.cuda.current_stream(device=device)
                    if _len_consume.cuda_stream != submission_stream.cuda_stream:
                        _len_evt = torch.cuda.Event()
                        _len_evt.record(submission_stream)
                        _len_consume.wait_event(_len_evt)
                        for _t in (_sl_snap, _sl32_snap, _kvpr_snap, *_kv_ptr_snaps.values()):
                            if isinstance(_t, torch.Tensor):
                                _t.record_stream(_len_consume)
                    len_snap = (_sl_snap, _sl32_snap, _kvpr_snap, _kv_ptr_snaps)
                    subset_tape_cache["len_snap"] = len_snap
                (
                    seq_lens_batch_src,
                    seq_lens_batch_i32_src,
                    kv_len_per_row_i32_src,
                    kv_lengths_snap_by_ptr,
                ) = len_snap

                if not isinstance(seq_lens_batch_src, torch.Tensor):
                    raise RuntimeError("prefill subset: missing seq_lens_batch")
                seq_lens_batch_sub = seq_lens_batch_src[start : start + batch_sub]
                seq_lens_batch_i32_sub = seq_lens_batch_i32_src
                if (
                    seq_lens_batch_i32_sub is not None
                    and isinstance(seq_lens_batch_i32_sub, torch.Tensor)
                    and seq_lens_batch_i32_sub.device == device
                    and seq_lens_batch_i32_sub.dtype == torch.int32
                    and seq_lens_batch_i32_sub.shape[0] >= len(slot_list_ref)
                ):
                    seq_lens_batch_i32_sub = seq_lens_batch_i32_sub[start : start + batch_sub]
                else:
                    if chunk_cohort_stamped:
                        raise RuntimeError(
                            "E_SFI_CAPTURE_COHORT_METADATA: missing prebuilt "
                            "seq_lens_batch_i32"
                        )
                    seq_lens_batch_i32_sub = seq_lens_batch_sub.to(
                        device=device,
                        dtype=torch.int32,
                    )

                if not isinstance(first_in.row_tensor, torch.Tensor):
                    raise RuntimeError("prefill subset: missing row_tensor")
                row_tensor_sub = first_in.row_tensor[start : start + batch_sub]
                # row_tensor_i32：优先复用 layout 内已有的 int32 view，避免额外的 dtype 转换 kernel。
                row_tensor_i32_src = first_in.row_tensor_i32
                if (
                    row_tensor_i32_src is not None
                    and row_tensor_i32_src.device == device
                    and row_tensor_i32_src.dtype == torch.int32
                    and row_tensor_i32_src.shape[0] >= len(slot_list_ref)
                ):
                    row_tensor_i32_sub = row_tensor_i32_src[start : start + batch_sub]
                else:
                    if chunk_cohort_stamped:
                        raise RuntimeError(
                            "E_SFI_CAPTURE_COHORT_METADATA: missing prebuilt row_tensor_i32"
                        )
                    row_tensor_i32_sub = row_tensor_sub.to(device=device, dtype=torch.int32)

                kv_len_per_row_i32_sub = kv_len_per_row_i32_src
                if kv_len_per_row_i32_sub is not None:
                    if not isinstance(kv_len_per_row_i32_sub, torch.Tensor):
                        raise RuntimeError(
                            "prefill subset: invalid kv_len_per_row_i32"
                        )
                    kv_len_per_row_i32_sub = kv_len_per_row_i32_sub[start : start + batch_sub]
                    if kv_len_per_row_i32_sub.dtype != torch.int32:
                        if chunk_cohort_stamped:
                            raise RuntimeError(
                                "E_SFI_CAPTURE_COHORT_METADATA: kv_len_per_row_i32 "
                                "dtype drift"
                            )
                        kv_len_per_row_i32_sub = kv_len_per_row_i32_sub.to(device=device, dtype=torch.int32)
                else:
                    if chunk_cohort_stamped:
                        raise RuntimeError(
                            "E_SFI_CAPTURE_COHORT_METADATA: missing authoritative "
                            "kv_len_per_row_i32"
                        )
                    # 保底：确保 rebuild/key_norms 路线始终有 int32 的 kv_len_per_row
                    kv_len_per_row_i32_sub = seq_lens_batch_sub.to(device=device, dtype=torch.int32)

                seq_lens_cpu_full = first_in.seq_lens_cpu
                if seq_lens_cpu_full is None:
                    raise RuntimeError("prefill subset: missing seq_lens_cpu")
                row_list_sub = first_in.row_list[start : start + batch_sub]
                seq_lens_cpu_sub = seq_lens_cpu_full[start : start + batch_sub]
                slots_use_ordered = slot_list_ref[start : start + batch_sub]

                slot_tensor_sub = first_in.slot_tensor
                if (
                    slot_tensor_sub is not None
                    and slot_tensor_sub.device == device
                    and slot_tensor_sub.dtype == torch.long
                    and slot_tensor_sub.shape[0] >= len(slot_list_ref)
                ):
                    slot_tensor_sub = slot_tensor_sub[start : start + batch_sub]
                else:
                    if chunk_cohort_stamped:
                        raise RuntimeError(
                            "E_SFI_CAPTURE_COHORT_METADATA: missing prebuilt slot_tensor"
                        )
                    slot_tensor_sub = torch.tensor(slots_use_ordered, device=device, dtype=torch.long)

                slot_tensor_i32_sub = getattr(first_in, "slot_tensor_i32", None)
                if (
                    slot_tensor_i32_sub is not None
                    and isinstance(slot_tensor_i32_sub, torch.Tensor)
                    and slot_tensor_i32_sub.device == device
                    and slot_tensor_i32_sub.dtype == torch.int32
                    and slot_tensor_i32_sub.shape[0] >= len(slot_list_ref)
                ):
                    slot_tensor_i32_sub = slot_tensor_i32_sub[start : start + batch_sub]
                else:
                    if chunk_cohort_stamped:
                        raise RuntimeError(
                            "E_SFI_CAPTURE_COHORT_METADATA: missing prebuilt slot_tensor_i32"
                        )
                    slot_tensor_i32_sub = slot_tensor_sub.to(device=device, dtype=torch.int32)

                slot_tensor_cpu_sub: Optional[torch.Tensor] = getattr(first_in, "slot_tensor_cpu", None)
                if (
                    slot_tensor_cpu_sub is not None
                    and isinstance(slot_tensor_cpu_sub, torch.Tensor)
                    and slot_tensor_cpu_sub.device.type == "cpu"
                    and slot_tensor_cpu_sub.dtype == torch.long
                    and slot_tensor_cpu_sub.numel() >= len(slot_list_ref)
                ):
                    slot_tensor_cpu_sub = slot_tensor_cpu_sub[start : start + batch_sub]
                else:
                    if chunk_cohort_stamped:
                        raise RuntimeError(
                            "E_SFI_CAPTURE_COHORT_METADATA: missing CPU slot snapshot"
                        )
                    slot_tensor_cpu_sub = torch.tensor(slots_use_ordered, dtype=torch.long)

                seq_lens_tensor_cpu_sub: Optional[torch.Tensor] = getattr(first_in, "seq_lens_tensor_cpu", None)
                if (
                    seq_lens_tensor_cpu_sub is not None
                    and isinstance(seq_lens_tensor_cpu_sub, torch.Tensor)
                    and seq_lens_tensor_cpu_sub.device.type == "cpu"
                    and seq_lens_tensor_cpu_sub.dtype == torch.long
                    and seq_lens_tensor_cpu_sub.numel() >= len(slot_list_ref)
                ):
                    seq_lens_tensor_cpu_sub = seq_lens_tensor_cpu_sub[start : start + batch_sub]
                else:
                    if chunk_cohort_stamped:
                        raise RuntimeError(
                            "E_SFI_CAPTURE_COHORT_METADATA: missing CPU length snapshot"
                        )
                    seq_lens_tensor_cpu_sub = torch.tensor(
                        [max(0, int(s)) for s in seq_lens_cpu_sub], dtype=torch.long
                    )

                # [DETERMINISTIC-CAPTURE-SNAPSHOT 2026-07-02] Third member of the
                # producer live-read race family (with [DETERMINISTIC-AUTOLEN] and
                # [DETERMINISTIC-SELECTOR-BOUNDS]). The payload capture views point
                # into the per-(state,chunk) capture arena, which the decode main
                # stream keeps rewriting every step (the rolling capture hook),
                # while the deferred bootstrap selector reads them on refresh_stream
                # at a floating time -> the topk input depended on WHEN the producer
                # executed (run-to-run output drift). Fix: take the shared arena out
                # of the deferred dataflow entirely via a private per-flush tape.
                #   1. Baseline: one stacked D2D copy of the full per-layer views,
                #      enqueued on the SUBMISSION stream (deterministic position,
                #      serialized against decode's arena rewrites). Covers direct
                #      capture / last_n==1 rows, whose arena content is final at
                #      submission.
                #   2. gt1 rows: the pending capture-postprocess jobs (stable
                #      scratch input + ready_event) are retargeted to write the
                #      tape instead of the arena -- the authoritative content lands
                #      in the tape at deferred time with no arena involvement and
                #      zero extra bandwidth (the postprocess write happens anyway).
                #   3. All subset payloads view the tape; tape[i] views share one
                #      5-dim base, satisfying the selector's direct-tape contract
                #      (per-layer clones satisfy neither probe and trip its
                #      require_base guard).
                # Decode keeps exclusive ownership of the arena; the deferred
                # producer keeps floating (async overlap preserved); bootstrap-only
                # path, no steady-state cost.
                if chunk_cohort_stamped:
                    from patches.fa3_native.capture_cohort_tape import (
                        resolve_capture_cohort_tape_group,
                    )

                    tape_group = resolve_capture_cohort_tape_group(payloads_in)
                    if tape_group is None:
                        raise RuntimeError(
                            "E_SFI_CAPTURE_COHORT_TAPE_PUBLICATION_MISSING: "
                            "stamped chunk cohort cannot fall back to an arena stack"
                        )
                    tape_denoms = tape_group.denoms
                    if use_denoms and tape_denoms is None:
                        raise RuntimeError(
                            "E_SFI_CAPTURE_COHORT_TAPE_DENOM_MISSING"
                        )
                    subset_tape_cache[bool(use_denoms)] = (
                        tape_group.scores,
                        tape_denoms if use_denoms else None,
                    )
                tape_entry = subset_tape_cache.get(bool(use_denoms))
                if tape_entry is None:
                    capture_views_all: List[torch.Tensor] = []
                    denoms_views_all: List[torch.Tensor] = []
                    for p in payloads_in:
                        capture_source = p.capture_scores
                        if not use_denoms:
                            lastn1_source = getattr(p, "lastn1_capture_scores", None)
                            if isinstance(lastn1_source, torch.Tensor):
                                capture_source = lastn1_source
                        if not isinstance(capture_source, torch.Tensor) or capture_source.dim() != 4:
                            raise RuntimeError("prefill subset: missing capture_scores")
                        capture_views_all.append(capture_source)
                        if use_denoms:
                            if p.log_f_denoms is None:
                                raise RuntimeError("prefill subset: missing log_f_denoms for denom path")
                            denoms_views_all.append(p.log_f_denoms)
                    # [DETERMINISTIC-TAPE-RAW 2026-07-03] 已发射 job 的
                    # reduce/denoms kernel 全在 refresh_stream 上,而 launched/
                    # completed 只是发射前后立即置位的 host 标志(kernel 可能仍在
                    # 飞,completion_event 也可能尚未挂上)。流级屏障:单事件把
                    # baseline stack(提交流,读 arena)排到 refresh_stream 当前
                    # 全部工作之后——一条边覆盖所有已发射 job,含 event-None 的
                    # host 窗口(逐 job wait 无法覆盖)。GPU 侧 no-op 若已完成,
                    # 零 host 阻塞,bootstrap-only。
                    _rs = getattr(self, "refresh_stream", None)
                    if _rs is not None:
                        _rs_evt = torch.cuda.Event()
                        _rs_evt.record(_rs)
                        submission_stream.wait_event(_rs_evt)
                    with torch.cuda.stream(submission_stream):
                        capture_tape = torch.stack(capture_views_all, dim=0)
                        denoms_tape = (
                            torch.stack(denoms_views_all, dim=0)
                            if denoms_views_all
                            else None
                        )
                    consume_stream = torch.cuda.current_stream(device=device)
                    if consume_stream.cuda_stream != submission_stream.cuda_stream:
                        snap_evt = torch.cuda.Event()
                        snap_evt.record(submission_stream)
                        consume_stream.wait_event(snap_evt)
                        capture_tape.record_stream(consume_stream)
                        if denoms_tape is not None:
                            denoms_tape.record_stream(consume_stream)
                    if use_denoms:
                        # [DETERMINISTIC-TAPE-WAW 2026-07-03] tape 有两个写者:
                        # 上面的 baseline stack(提交流)与下面 retarget 的
                        # deferred job(refresh_stream)。job 若早于 stack 执行,
                        # stack 会用 arena 旧值覆盖 job 的 tape 输出——错峰行
                        # 的 capture/denoms 内容随两流实际时序二态分叉(指纹
                        # 取证:job 输入 scratch 恒稳定、caller/执行序全同,
                        # 唯 tape 内容漂;任何 host 同步探针都会把 job 推迟到
                        # stack 后而"治好"=heisenbug 铁证)。补 WAW 边:stack
                        # 完成事件挂到 job,job 消费前 wait(GPU 侧 no-op 若已
                        # 完成;bootstrap-only,零热路径开销)。
                        tape_stack_evt = torch.cuda.Event()
                        tape_stack_evt.record(submission_stream)
                        for layer_pos, p in enumerate(payloads_in):
                            job = getattr(p, "capture_postprocess_job", None)
                            if job is None or id(job) in retargeted_postprocess_job_ids:
                                continue
                            retargeted_postprocess_job_ids.add(id(job))
                            # TOCTOU 互斥:launched 检查与 deferred drain 线程的
                            # launch 并发;锁内完成"检查+retarget/挂 evt",执行侧
                            # 锁内完成"置 launched+读输出目标"。
                            _lk = getattr(job, "lifecycle_lock", None)
                            if _lk is not None:
                                _lk.acquire()
                            try:
                                if bool(getattr(job, "completed", False)) or bool(
                                    getattr(job, "launched", False)
                                ):
                                    # 已发射的 job 输出在 arena:上方流级屏障已把
                                    # baseline stack 排到其 kernels 之后,拷贝即
                                    # 持有其终值。
                                    continue
                                if getattr(job, "prefill_out_capture_scores", None) is not None:
                                    job.prefill_out_capture_scores = capture_tape[layer_pos]
                                if (
                                    denoms_tape is not None
                                    and getattr(job, "prefill_out_log_f_denoms", None) is not None
                                ):
                                    job.prefill_out_log_f_denoms = denoms_tape[layer_pos]
                                job.tape_stack_evt = tape_stack_evt
                            finally:
                                if _lk is not None:
                                    _lk.release()
                    tape_entry = (capture_tape, denoms_tape)
                    subset_tape_cache[bool(use_denoms)] = tape_entry
                capture_tape, denoms_tape = tape_entry
                cohort_consumer_tokens_sub = tuple()
                if chunk_cohort_stamped:
                    try:
                        cohort_consumer_tokens_sub = tuple(
                            cohort_tape_tokens_by_slot[int(slot)]
                            for slot in slots_use_ordered
                        )
                    except KeyError as exc:
                        raise RuntimeError(
                            "E_SFI_CAPTURE_COHORT_CONSUMER: subset contains a "
                            f"non-finalized slot {int(exc.args[0])}"
                        ) from None

                payloads_out = []
                for layer_pos, p in enumerate(payloads_in):
                    capture_view = capture_tape[layer_pos][
                        start : start + batch_sub, :, :, :kv_len_total
                    ]
                    denoms_view = None
                    if use_denoms:
                        assert denoms_tape is not None
                        denoms_view = denoms_tape[layer_pos][start : start + batch_sub]
                    kv_lengths_src = p.kv_lengths
                    if isinstance(kv_lengths_src, torch.Tensor):
                        kv_lengths_src = kv_lengths_snap_by_ptr.get(
                            int(p.kv_lengths.data_ptr()), kv_lengths_src
                        )
                    kv_lengths_sub = _slice_payload_batch_dim(
                        kv_lengths_src,
                        label="kv_lengths",
                        expected_dim=2,
                    )
                    slot_req_ids_sub = None
                    if p.slot_req_ids is not None:
                        if len(p.slot_req_ids) != len(slot_list_ref):
                            raise RuntimeError(
                                "prefill subset: slot_req_ids must align with slot_list"
                            )
                        slot_req_ids_sub = tuple(
                            p.slot_req_ids[start : start + batch_sub]
                        )
                    payloads_out.append(
                        replace(
                            p,
                            capture_scores=capture_view,
                            log_f_denoms=denoms_view,
                            kv_lengths=kv_lengths_sub,
                            kv_len_per_row_i32=kv_len_per_row_i32_sub,
                            seq_lens_batch=seq_lens_batch_sub,
                            seq_lens_batch_i32=seq_lens_batch_i32_sub,
                            slot_list=list(slots_use_ordered),
                            row_list=row_list_sub,
                            slot_req_ids=slot_req_ids_sub,
                            # Query-side and refresh-only views are not aligned to
                            # this request-local slot subset.  The historical
                            # constructor intentionally dropped them; keep that
                            # semantic boundary explicit while replace preserves
                            # step identity/provenance fields automatically.
                            q=None,
                            cu_seqlens_q=None,
                            softmax_scale=0.0,
                            softcap=0.0,
                            window_size=None,
                            alibi_slopes=None,
                            k_descale=None,
                            slot_tensor=slot_tensor_sub,
                            slot_tensor_i32=slot_tensor_i32_sub,
                            slot_tensor_cpu=slot_tensor_cpu_sub,
                            row_tensor=row_tensor_sub,
                            row_tensor_i32=row_tensor_i32_sub,
                            refresh_rows_long=None,
                            refresh_block_table_sub=None,
                            refresh_seq_lens_i32=None,
                            bootstrap_slots=set(slots_use_ordered),
                            lastn1_capture_scores=None,
                            seq_lens_cpu=seq_lens_cpu_sub,
                            seq_lens_tensor_cpu=seq_lens_tensor_cpu_sub,
                            refresh_reason="",
                            refresh_intent_req_ids=tuple(),
                            stagger_layer_index=-1,
                            q_is_sub=False,
                            fast_signature=_make_selector_fast_signature(
                                capture_scores=capture_view,
                                log_f_denoms=denoms_view,
                                kv_lengths=kv_lengths_sub,
                                block_table=p.block_table,
                            ),
                            cohort_tape_consumer_tokens=(
                                cohort_consumer_tokens_sub
                                if chunk_cohort_stamped
                                else p.cohort_tape_consumer_tokens
                            ),
                            cohort_tape_row_start=(
                                int(start)
                                if chunk_cohort_stamped
                                else p.cohort_tape_row_start
                            ),
                        )
                    )
                return payloads_out

            def _run_prefill_group(slots_use: List[int], *, use_denoms: bool) -> None:
                nonlocal lastn1_direct_count, gt1_reduce_count, gt1_scalar_fallback_count
                nonlocal cohort_tape_deferred_producer_used
                if not slots_use:
                    return
                group_payloads = _subset_payloads_for_slots(prefill_payloads, slots_use, use_denoms=use_denoms)
                if not group_payloads:
                    return
                if use_denoms:
                    gt1_reduce_count += len(slots_use)
                else:
                    lastn1_direct_count += len(slots_use)
                gt1_scalar_fallback_count = 0
                one_shot_bootstrap_only = bool(
                    getattr(getattr(self, "config", None), "one_shot_bootstrap_only", False)
                )
                layer_indices = tuple(
                    int(getattr(getattr(payload, "state", None), "layer_index", -1))
                    for payload in group_payloads
                )
                layer_indices = tuple(layer for layer in layer_indices if layer >= 0)
                group_id_for_profile = (
                    int(min(layer_indices) // int(_CAPTURE_CHUNK))
                    if layer_indices and int(_CAPTURE_CHUNK) > 0
                    else -1
                )

                def _append_prefill_producer_timeline(
                    *,
                    phase: str,
                    timestamp_ns: int | None = None,
                    duration_us: float = 0.0,
                    extra_fields: Optional[Dict[str, object]] = None,
                ) -> None:
                    if not os.environ.get("VLLM_SPARSE_ONE_SHOT_TIMELINE_LOG", ""):
                        return
                    try:
                        from patches.refresh_runtime.one_shot_timeline import (
                            append_one_shot_timeline,
                            timeline_log_path,
                        )

                        ts_ns = (
                            time.perf_counter_ns()
                            if timestamp_ns is None
                            else int(timestamp_ns)
                        )
                        for slot in slots_use:
                            slot_i = int(slot)
                            req_id = submission_req_id_by_slot.get(slot_i)
                            if req_id is None or _is_free_slot_id(req_id):
                                continue
                            append_one_shot_timeline(
                                path=timeline_log_path(),
                                request_id=str(req_id),
                                epoch=int(self.step_context_epoch),
                                chunk_id=int(group_id_for_profile),
                                buffer_id=int(buf),
                                phase=str(phase),
                                timestamp_ns=ts_ns,
                                duration_us=float(duration_us),
                                aux_stream_enabled=bool(do_async),
                                extra_fields=extra_fields,
                            )
                    except Exception:
                        _log.debug("prefill producer timeline append failed", exc_info=True)

                def _mark_attribution_bootstrap_ready(*, mode: str) -> None:
                    if os.environ.get("VLLM_SPARSE_FORCE_COMPACT_OFF", "0") != "1":
                        raise RuntimeError(
                            "VLLM_SPARSE_ATTRIB_PREFILL_PRODUCER requires "
                            "VLLM_SPARSE_FORCE_COMPACT_OFF=1"
                        )
                    for payload in group_payloads:
                        state = getattr(payload, "state", None)
                        compact_kv_len = getattr(state, "compact_kv_len", None)
                        if not isinstance(compact_kv_len, list):
                            continue
                        for slot in slots_use:
                            slot_i = int(slot)
                            if 0 <= slot_i < len(compact_kv_len):
                                compact_kv_len[slot_i] = max(
                                    1,
                                    int(compact_kv_len[slot_i] or 0),
                                )
                    for slot in slots_use:
                        slot_i = int(slot)
                        req_id = submission_req_id_by_slot.get(slot_i)
                        if req_id is None or _is_free_slot_id(req_id):
                            continue
                        tracking = self._ensure_request(req_id)
                        tracking.bootstrap_done = True
                        tracking.bootstrap_pending = False
                        tracking.bootstrap_pending_epoch = -1
                        if hasattr(tracking, "bootstrap_pending_events"):
                            tracking.bootstrap_pending_events = []
                        tracking.bootstrap_publish_skipped_reason = (
                            f"attribution_{mode}_force_compact_off"
                        )
                        self._bootstrap_pending_request_ids.discard(req_id)

                if one_shot_bootstrap_only:
                    if not bool(
                        getattr(
                            getattr(self, "config", None),
                            "compact_page_residency_enabled",
                            False,
                        )
                    ):
                        raise RuntimeError(
                            "one-shot RRP compact+recent requires native compact_page_residency "
                            "before producer enqueue"
                        )
                    from patches.page_kv_residency import (
                        require_native_compact_residency_for_layers,
                    )

                    expected_layer_states = {}
                    for payload in group_payloads:
                        state = getattr(payload, "state", None)
                        layer_index = int(getattr(state, "layer_index", -1))
                        if layer_index >= 0:
                            expected_layer_states[layer_index] = state
                    require_native_compact_residency_for_layers(
                        expected_layer_states,
                        tuple(expected_layer_states),
                    )
                defer_bootstrap_producer_requested = (
                    os.environ.get("VLLM_SPARSE_DEFER_BOOTSTRAP_PRODUCER", "0")
                    == "1"
                )
                if defer_bootstrap_producer_requested and not one_shot_bootstrap_only:
                    raise RuntimeError(
                        "deferred bootstrap producer requires one_shot_bootstrap_only"
                    )
                defer_bootstrap_producer = defer_bootstrap_producer_requested
                if chunk_cohort_stamped and not defer_bootstrap_producer:
                    raise RuntimeError(
                        "E_SFI_CAPTURE_COHORT_DEFERRED_OWNER: stamped chunk "
                        "cohort requires the deferred bootstrap consumer"
                    )
                if defer_bootstrap_producer:
                    from patches.refresh_runtime.deferred_producer import (
                        DeferredProducerJob,
                        append_deferred_payload_group,
                        build_deferred_bootstrap_job,
                        validate_bridge_graph_policy,
                    )

                    bridge_max_tokens = int(
                        os.environ.get(
                            "VLLM_SPARSE_BOOTSTRAP_BRIDGE_MAX_TOKENS",
                            "2",
                        )
                    )
                    bridge_graph_policy = validate_bridge_graph_policy(
                        os.environ.get(
                            "VLLM_SPARSE_BOOTSTRAP_BRIDGE_GRAPH_POLICY",
                            "evict_recapture_once",
                        )
                    )
                    for slot in slots_use:
                        slot_i = int(slot)
                        req_id = submission_req_id_by_slot.get(slot_i)
                        if req_id is None or _is_free_slot_id(req_id):
                            continue
                        # [CHUNKED-CAPTURE-ACCUMULATE 2026-07-06] 捕获窗跨 chunk
                        # 时首/中间片也产生 capture payload（分片 accumulate 进
                        # arena），但 bootstrap producer job 只能在 finalize 片
                        # （remaining==0）构建——selection 消费的是 arena 的最终
                        # 合并态，半窗 job 是无用功，且首片步界 launch 后尾片
                        # 再进组会撞 "already launched" fail-fast。非跨片路径
                        # 首/中间 chunk 没有 payload（should_capture=False）
                        # 到不了这里——构造性零影响。
                        if slot_i not in finalize_slot_set:
                            continue
                        per_slot_payloads = _subset_payloads_for_slots(
                            prefill_payloads,
                            [slot_i],
                            use_denoms=use_denoms,
                        )
                        if not per_slot_payloads:
                            continue
                        # This is the source-ownership boundary for this
                        # request-local frozen payload group.  _run_prefill
                        # executes after chunk_ready on the submission stream;
                        # recording after _subset_payloads_for_slots also
                        # covers any request-local snapshot copies it queued.
                        # Use one event per group so later slot snapshots cannot
                        # accidentally fall after a shared earlier event.
                        deferred_source_ready_event = torch.cuda.Event(
                            enable_timing=False
                        )
                        deferred_source_ready_event.record(
                            torch.cuda.current_stream(device=device)
                        )
                        compact_lease_generation = max(
                            (
                                int(
                                    getattr(
                                        getattr(payload, "state", None),
                                        "compact_page_residency_generation",
                                        0,
                                    )
                                    or 0
                                )
                                for payload in per_slot_payloads
                            ),
                            default=0,
                        )
                        payload_group_snapshot = tuple(per_slot_payloads)
                        tracking = self._ensure_request(req_id)
                        job = getattr(tracking, "deferred_producer_job", None)
                        if job is None:
                            job = build_deferred_bootstrap_job(
                                request_id=str(req_id),
                                producer_job_epoch=int(self.step_context_epoch),
                                compact_lease_generation=int(compact_lease_generation),
                                expected_slot=int(slot_i),
                                payload_groups=(payload_group_snapshot,),
                                bridge_max_tokens=bridge_max_tokens,
                                source_ready_event=deferred_source_ready_event,
                            )
                            tracking.deferred_producer_job = job
                        else:
                            if bool(getattr(job, "launched", False)):
                                raise RuntimeError(
                                    "deferred bootstrap producer job already launched"
                                )
                            append_deferred_payload_group(
                                job,
                                payload_group_snapshot,
                                source_ready_event=deferred_source_ready_event,
                            )
                        job_typed: DeferredProducerJob = job
                        tracking.bootstrap_bridge_active = True
                        tracking.bridge_max_tokens = int(job_typed.bridge_max_tokens)
                        tracking.bridge_token_count = 0
                        tracking.bridge_token_positions = []
                        tracking.bridge_last_counted_epoch = -1
                        tracking.producer_job_epoch = int(job_typed.producer_job_epoch)
                        tracking.producer_launch_step = -1
                        tracking.building_compact_epoch = int(
                            job_typed.building_compact_epoch
                        )
                        tracking.ready_compact_epoch = -1
                        tracking.active_compact_epoch = -1
                        tracking.bridge_graph_policy = str(bridge_graph_policy)
                        if slot_i in finalize_slot_set and bool(is_last_layer):
                            tracking.prefill_capture_ready = False
                            tracking.prefill_capture_last_n = 0
                            tracking.bootstrap_pending = True
                            tracking.bootstrap_pending_epoch = int(
                                self.step_context_epoch
                            )
                            tracking.bootstrap_pending_events = []
                            self._bootstrap_pending_request_ids.add(req_id)
                    _append_prefill_producer_timeline(
                        phase="deferred_producer_job_created",
                        extra_fields={
                            "layer_indices": list(layer_indices),
                            "lastn1_direct_count": int(lastn1_direct_count),
                            "gt1_reduce_count": int(gt1_reduce_count),
                            "gt1_scalar_fallback_count": int(
                                gt1_scalar_fallback_count
                            ),
                            "bootstrap_full_kv_handoff": True,
                        },
                    )
                    # Real deferred path: the producer has not run yet. The step
                    # boundary calls _launch_deferred_bootstrap_producer_jobs.
                    if chunk_cohort_stamped:
                        cohort_tape_deferred_producer_used = True
                    return
                from patches.fa3_native.postprocess import (
                    run_capture_postprocess_jobs_for_payloads,
                )

                producer_group_start_ns = time.perf_counter_ns()
                _append_prefill_producer_timeline(
                    phase="producer_group_start",
                    timestamp_ns=producer_group_start_ns,
                    extra_fields={
                        "layer_indices": list(layer_indices),
                        "lastn1_direct_count": int(lastn1_direct_count),
                        "gt1_reduce_count": int(gt1_reduce_count),
                        "gt1_scalar_fallback_count": int(gt1_scalar_fallback_count),
                    },
                )
                producer_attrib_mode = ""
                if one_shot_bootstrap_only:
                    producer_attrib_mode = (
                        os.environ.get("VLLM_SPARSE_ATTRIB_PREFILL_PRODUCER", "")
                        .strip()
                        .lower()
                    )
                    if producer_attrib_mode not in {"", "normal", "none", "selector_only"}:
                        producer_attrib_mode = ""
                    if producer_attrib_mode == "normal":
                        producer_attrib_mode = ""
                if producer_attrib_mode == "none":
                    _mark_attribution_bootstrap_ready(mode="none")
                    producer_group_end_ns = time.perf_counter_ns()
                    _append_prefill_producer_timeline(
                        phase="producer_attrib_none",
                        timestamp_ns=producer_group_end_ns,
                        duration_us=float(
                            producer_group_end_ns - producer_group_start_ns
                        )
                        / 1000.0,
                        extra_fields={
                            "layer_indices": list(layer_indices),
                            "producer_attribution_diagnostic": True,
                            "producer_attrib_mode": "none",
                            "lastn1_direct_count": int(lastn1_direct_count),
                            "gt1_reduce_count": int(gt1_reduce_count),
                            "gt1_scalar_fallback_count": int(gt1_scalar_fallback_count),
                        },
                    )
                    return
                ready_chunk_for_rebuild = int(_CAPTURE_CHUNK)
                ready_group_done_events: dict[int, torch.cuda.Event] = {}
                if one_shot_bootstrap_only:
                    from patches.refresh_runtime.producer_ready import (
                        producer_group_id_for_layer,
                        producer_group_layers_for_group,
                        resolve_one_shot_ready_chunk,
                        validate_one_shot_ready_chunk_alignment,
                    )

                    ready_chunk_for_rebuild = resolve_one_shot_ready_chunk(
                        capture_chunk=int(_CAPTURE_CHUNK),
                    )
                    validate_one_shot_ready_chunk_alignment(
                        capture_chunk=int(_CAPTURE_CHUNK),
                        ready_chunk=int(ready_chunk_for_rebuild),
                    )

                def _ready_payload_segments() -> List[
                    Tuple[Optional[int], Sequence[SelectorBatchPayload], Sequence[Set[int]]]
                ]:
                    if (
                        one_shot_bootstrap_only
                        and int(ready_chunk_for_rebuild) < int(_CAPTURE_CHUNK)
                    ):
                        segments: List[
                            Tuple[
                                Optional[int],
                                Sequence[SelectorBatchPayload],
                                Sequence[Set[int]],
                            ]
                        ] = []
                        start = 0
                        while start < len(group_payloads):
                            first_layer = int(
                                getattr(
                                    getattr(group_payloads[start], "state", None),
                                    "layer_index",
                                    -1,
                                )
                            )
                            if first_layer < 0:
                                raise RuntimeError(
                                    "one-shot writer-ready segment requires layer_index"
                                )
                            group_id_i = producer_group_id_for_layer(
                                layer_index=int(first_layer),
                                capture_chunk=int(ready_chunk_for_rebuild),
                            )
                            end = start + 1
                            while end < len(group_payloads):
                                layer_i = int(
                                    getattr(
                                        getattr(group_payloads[end], "state", None),
                                        "layer_index",
                                        -1,
                                    )
                                )
                                if producer_group_id_for_layer(
                                    layer_index=int(layer_i),
                                    capture_chunk=int(ready_chunk_for_rebuild),
                                ) != int(group_id_i):
                                    break
                                end += 1
                            segment_layers = tuple(
                                int(
                                    getattr(
                                        getattr(payload, "state", None),
                                        "layer_index",
                                        -1,
                                    )
                                )
                                for payload in group_payloads[start:end]
                            )
                            expected_layers = producer_group_layers_for_group(
                                group_id=int(group_id_i),
                                layer_count=len(self.layer_cache_keys),
                                ready_chunk=int(ready_chunk_for_rebuild),
                            )
                            if tuple(segment_layers) != tuple(expected_layers):
                                raise RuntimeError(
                                    "one-shot writer-ready segment crosses selector chunk boundary"
                                )
                            segments.append(
                                (
                                    group_id_i,
                                    group_payloads[start:end],
                                    [
                                        payload.bootstrap_slots
                                        for payload in group_payloads[start:end]
                                    ],
                                )
                            )
                            start = end
                        return segments
                    return [
                        (
                            None,
                            group_payloads,
                            [payload.bootstrap_slots for payload in group_payloads],
                        )
                    ]

                ready_payload_segments = _ready_payload_segments()

                def _record_prefill_group_profile_start(
                    segment_group_id: Optional[int],
                ) -> Optional[Tuple[int, torch.cuda.Event, torch.cuda.Event]]:
                    profile_group_id = (
                        int(segment_group_id)
                        if segment_group_id is not None
                        else int(group_id_for_profile)
                    )
                    if do_profile and prof is not None and profile_group_id >= 0:
                        group_evt_pair = (
                            int(profile_group_id),
                            torch.cuda.Event(enable_timing=True),
                            torch.cuda.Event(enable_timing=True),
                        )
                        group_evt_pair[1].record(torch.cuda.current_stream(device=device))
                        return group_evt_pair
                    return None

                def _record_prefill_group_profile_end(
                    group_evt_pair: Optional[
                        Tuple[int, torch.cuda.Event, torch.cuda.Event]
                    ],
                ) -> None:
                    if group_evt_pair is None or prof is None:
                        return
                    group_evt_pair[2].record(torch.cuda.current_stream(device=device))
                    prof.prefill_group_evt_pairs.append(group_evt_pair)

                def _run_selector_for_segment(
                    segment_payloads: Sequence[SelectorBatchPayload],
                ) -> Optional[Any]:
                    if chunk_cohort_stamped:
                        from patches.fa3_native.capture_cohort_tape import (
                            require_capture_cohort_selector_view,
                        )

                        if require_capture_cohort_selector_view(segment_payloads) is None:
                            raise RuntimeError(
                                "E_SFI_CAPTURE_COHORT_SELECTOR_VIEW: stamped "
                                "selector payload lost tape provenance"
                            )
                    selector_evt_pair: Optional[
                        Tuple[torch.cuda.Event, torch.cuda.Event]
                    ] = None
                    if do_profile and prof is not None:
                        selector_evt_pair = (
                            torch.cuda.Event(enable_timing=True),
                            torch.cuda.Event(enable_timing=True),
                        )
                        selector_evt_pair[0].record(torch.cuda.current_stream(device=device))
                    segment_result = self._apply_alpha_selector_batched_fused(
                        segment_payloads,
                        phase="prefill",
                    )
                    if selector_evt_pair is not None:
                        selector_evt_pair[1].record(torch.cuda.current_stream(device=device))
                    if segment_result is None:
                        return None
                    if prof is not None:
                        if selector_evt_pair is not None:
                            prof.prefill_selector_evt_pairs.append(selector_evt_pair)
                        _capture_prefill_selector_detail(prof, segment_result)
                        prof.prefill_selector_runs += 1
                    return segment_result

                fused_ok = True
                _append_prefill_producer_timeline(
                    phase="capture_postprocess_start",
                    extra_fields={"layer_indices": list(layer_indices)},
                )
                run_capture_postprocess_jobs_for_payloads(
                    group_payloads,
                    meta_cache_owner=self,
                )
                _append_prefill_producer_timeline(
                    phase="capture_postprocess_end",
                    extra_fields={"layer_indices": list(layer_indices)},
                )
                _append_prefill_producer_timeline(
                    phase="selector_start",
                    extra_fields={"layer_indices": list(layer_indices)},
                )
                group_evt_pair = _record_prefill_group_profile_start(None)
                prefill_result = _run_selector_for_segment(group_payloads)
                if prefill_result is None:
                    return
                _append_prefill_producer_timeline(
                    phase="selector_end",
                    extra_fields={"layer_indices": list(layer_indices)},
                )
                selector_done_ns: Optional[int] = (
                    time.perf_counter_ns() if do_profile and prof is not None else None
                )
                if producer_attrib_mode == "selector_only":
                    _mark_attribution_bootstrap_ready(mode="selector_only")
                    producer_group_end_ns = time.perf_counter_ns()
                    _record_prefill_group_profile_end(group_evt_pair)
                    _append_prefill_producer_timeline(
                        phase="producer_attrib_selector_only",
                        timestamp_ns=producer_group_end_ns,
                        duration_us=float(
                            producer_group_end_ns - producer_group_start_ns
                        )
                        / 1000.0,
                        extra_fields={
                            "layer_indices": list(layer_indices),
                            "producer_attribution_diagnostic": True,
                            "producer_attrib_mode": "selector_only",
                            "lastn1_direct_count": int(lastn1_direct_count),
                            "gt1_reduce_count": int(gt1_reduce_count),
                            "gt1_scalar_fallback_count": int(gt1_scalar_fallback_count),
                        },
                    )
                    return

                rebuild_segments = []
                if (
                    one_shot_bootstrap_only
                    and int(ready_chunk_for_rebuild) < int(_CAPTURE_CHUNK)
                ):
                    selected_offset = 0
                    for (
                        segment_group_id,
                        segment_payloads,
                        segment_bootstrap,
                    ) in ready_payload_segments:
                        selected_next = selected_offset + len(segment_payloads)
                        rebuild_segments.append(
                            (
                                segment_group_id,
                                segment_payloads,
                                prefill_result.selected_indices[
                                    selected_offset:selected_next
                                ].contiguous(),
                                segment_bootstrap,
                            )
                        )
                        selected_offset = selected_next
                else:
                    rebuild_segments.append(
                        (
                            None,
                            group_payloads,
                            prefill_result.selected_indices,
                            [payload.bootstrap_slots for payload in group_payloads],
                        )
                    )

                rebuild_cpu_t0_ns: Optional[int] = (
                    time.perf_counter_ns() if do_profile and prof is not None else None
                )
                boundary_recorded = False
                _append_prefill_producer_timeline(
                    phase="rebuild_start",
                    extra_fields={"layer_indices": list(layer_indices)},
                )

                def _mark_one_shot_group_event_serial(
                    *,
                    group_id: int,
                    slot: Optional[int] = None,
                ) -> int:
                    event_serial = int(
                        getattr(self, "_one_shot_group_done_evt_serial", 0) or 0
                    ) + 1
                    self._one_shot_group_done_evt_serial = int(event_serial)
                    serial_registry = getattr(
                        self,
                        "_one_shot_group_done_evt_serial_by_group",
                        None,
                    )
                    if not isinstance(serial_registry, dict):
                        serial_registry = {}
                        self._one_shot_group_done_evt_serial_by_group = (
                            serial_registry
                        )
                    serial_registry[int(group_id)] = int(event_serial)
                    if slot is not None:
                        slot_serial_registry = getattr(
                            self,
                            "_one_shot_group_done_evt_serial_by_group_slot",
                            None,
                        )
                        if not isinstance(slot_serial_registry, dict):
                            slot_serial_registry = {}
                            self._one_shot_group_done_evt_serial_by_group_slot = (
                                slot_serial_registry
                            )
                        slot_serial_registry[(int(group_id), int(slot))] = int(
                            event_serial
                        )
                    return int(event_serial)

                for segment_group_id, segment_payloads, segment_selected, segment_bootstrap in rebuild_segments:
                    rebuild_evt_pair: Optional[Tuple[torch.cuda.Event, torch.cuda.Event]] = None
                    if do_profile and prof is not None:
                        rebuild_evt_pair = (
                            torch.cuda.Event(enable_timing=True),
                            torch.cuda.Event(enable_timing=True),
                        )
                        rebuild_evt_pair[0].record(torch.cuda.current_stream(device=device))
                    segment_ok = self._rebuild_compact_slots_batched_layers_from_selection(
                        segment_payloads,
                        segment_selected,
                        phase="prefill",
                        bootstrap_slots_by_layer=segment_bootstrap,
                        defer_compact_meta_publish=flush_compact_meta_commit_log is not None,
                        compact_meta_commit_log=flush_compact_meta_commit_log,
                    )
                    fused_ok = bool(fused_ok and segment_ok)
                    if selector_done_ns is not None and prof is not None and not boundary_recorded:
                        prof.selector_writer_boundary_cpu_us += (
                            time.perf_counter_ns() - selector_done_ns
                        ) / 1000.0
                        boundary_recorded = True
                    if prof is not None:
                        _capture_writer_kernel_variant(prof)
                    if rebuild_evt_pair is not None:
                        rebuild_evt_pair[1].record(torch.cuda.current_stream(device=device))
                    if not segment_ok:
                        break
                    if prof is not None:
                        if rebuild_evt_pair is not None:
                            prof.prefill_rebuild_evt_pairs.append(rebuild_evt_pair)
                        prof.prefill_rebuild_runs += 1
                    if segment_group_id is not None:
                        event_registry = getattr(
                            self,
                            "_one_shot_group_done_evt_by_group",
                            None,
                        )
                        if event_registry is None:
                            event_registry = {}
                            self._one_shot_group_done_evt_by_group = event_registry
                        if not isinstance(event_registry, dict):
                            raise RuntimeError(
                                "one-shot group-ready event registry is invalid"
                            )
                        group_done_event = event_registry.get(int(segment_group_id))
                        if group_done_event is None:
                            group_done_event = torch.cuda.Event(enable_timing=False)
                            event_registry[int(segment_group_id)] = group_done_event
                        _mark_one_shot_group_event_serial(
                            group_id=int(segment_group_id),
                        )
                        group_done_event.record(torch.cuda.current_stream(device=device))
                        event_registry[int(segment_group_id)] = group_done_event
                        ready_group_done_events[int(segment_group_id)] = group_done_event
                _append_prefill_producer_timeline(
                    phase="rebuild_end",
                    extra_fields={"layer_indices": list(layer_indices)},
                )
                if rebuild_cpu_t0_ns is not None and prof is not None:
                    prof.prefill_rebuild_cpu_us += (
                        time.perf_counter_ns() - rebuild_cpu_t0_ns
                    ) / 1000.0
                _record_prefill_group_profile_end(group_evt_pair)
                if not fused_ok:
                    raise RuntimeError("prefill compact rebuild: fused gather failed")
                publish_t0_ns: Optional[int] = (
                    time.perf_counter_ns() if do_profile and prof is not None else None
                )
                if one_shot_bootstrap_only:
                    from patches.refresh_runtime.producer_ready import (
                        ProducerReadyState,
                        build_producer_group_manifest,
                        declare_expected_groups,
                        expected_group_mask_for_layer_count,
                        producer_group_id_for_layer,
                        producer_group_layers_for_group,
                        producer_ready_summary,
                        register_group_done_event,
                        register_submitted_group,
                        resolve_one_shot_ready_chunk,
                        validate_one_shot_ready_chunk_alignment,
                    )

                    if layer_indices:
                        capture_chunk_i = int(_CAPTURE_CHUNK)
                        ready_chunk = resolve_one_shot_ready_chunk(
                            capture_chunk=capture_chunk_i,
                        )
                        validate_one_shot_ready_chunk_alignment(
                            capture_chunk=capture_chunk_i,
                            ready_chunk=int(ready_chunk),
                        )
                        group_ids = tuple(
                            sorted(
                                {
                                    producer_group_id_for_layer(
                                        layer_index=int(layer),
                                        capture_chunk=int(ready_chunk),
                                    )
                                    for layer in layer_indices
                                }
                            )
                        )
                        event_registry = getattr(
                            self,
                            "_one_shot_group_done_evt_by_group",
                            None,
                        )
                        if event_registry is None:
                            event_registry = {}
                            self._one_shot_group_done_evt_by_group = event_registry
                        if not isinstance(event_registry, dict):
                            raise RuntimeError(
                                "one-shot group-ready event registry is invalid"
                            )
                        compact_lease_generation = max(
                            (
                                int(
                                    getattr(
                                        getattr(payload, "state", None),
                                        "compact_page_residency_generation",
                                        0,
                                    )
                                    or 0
                                )
                                for payload in group_payloads
                            ),
                            default=0,
                        )
                        snapshot_signature = int(
                            getattr(
                                group_payloads[0],
                                "capture_handle_generation",
                                0,
                            )
                            or 0
                        )
                        for group_id in group_ids:
                            group_layer_indices = producer_group_layers_for_group(
                                group_id=int(group_id),
                                layer_count=len(self.layer_cache_keys),
                                ready_chunk=int(ready_chunk),
                            )
                            if not set(group_layer_indices).issubset(set(layer_indices)):
                                continue
                            group_done_event = ready_group_done_events.get(int(group_id))
                            if group_done_event is None:
                                group_done_event = event_registry.get(int(group_id))
                            if group_done_event is None:
                                group_done_event = torch.cuda.Event(
                                    enable_timing=False
                                )
                            if int(group_id) not in ready_group_done_events:
                                _mark_one_shot_group_event_serial(
                                    group_id=int(group_id),
                                )
                                group_done_event.record(
                                    torch.cuda.current_stream(device=device)
                                )
                            event_registry[int(group_id)] = group_done_event
                            slot_event_registry = getattr(
                                self,
                                "_one_shot_group_done_evt_by_group_slot",
                                None,
                            )
                            if slot_event_registry is None:
                                slot_event_registry = {}
                                self._one_shot_group_done_evt_by_group_slot = (
                                    slot_event_registry
                                )
                            if not isinstance(slot_event_registry, dict):
                                raise RuntimeError(
                                    "one-shot group-ready slot event registry is invalid"
                                )
                            for slot in slots_use:
                                slot_i = int(slot)
                                req_id = submission_req_id_by_slot.get(slot_i)
                                if req_id is None or _is_free_slot_id(req_id):
                                    continue
                                tracking = self._ensure_request(req_id)
                                ready_state = getattr(
                                    tracking,
                                    "producer_ready_state",
                                    None,
                                )
                                if ready_state is None:
                                    ready_state = ProducerReadyState()
                                    tracking.producer_ready_state = ready_state
                                slot_event_key = (int(group_id), int(slot_i))
                                slot_group_done_event = slot_event_registry.get(
                                    slot_event_key
                                )
                                if slot_group_done_event is None:
                                    slot_group_done_event = torch.cuda.Event(
                                        enable_timing=False
                                    )
                                    slot_event_registry[slot_event_key] = (
                                        slot_group_done_event
                                    )
                                _mark_one_shot_group_event_serial(
                                    group_id=int(group_id),
                                    slot=slot_i,
                                )
                                slot_group_done_event.record(
                                    torch.cuda.current_stream(device=device)
                                )
                                declare_expected_groups(
                                    ready_state,
                                    expected_group_mask=expected_group_mask_for_layer_count(
                                        layer_count=len(self.layer_cache_keys),
                                        capture_chunk=int(ready_chunk),
                                    ),
                                )
                                source_ready_generation = max(
                                    (
                                        int(manifest.source_ready_event_generation)
                                        for manifest in ready_state.manifests.values()
                                    ),
                                    default=0,
                                ) + 1
                                manifest = build_producer_group_manifest(
                                    group_id=int(group_id),
                                    layer_indices=group_layer_indices,
                                    step_epoch=int(self.step_context_epoch),
                                    snapshot_signature=int(snapshot_signature),
                                    compact_lease_generation=int(
                                        compact_lease_generation
                                    ),
                                    source_ready_event_generation=int(
                                        source_ready_generation
                                    ),
                                    expected_slots=(slot_i,),
                                    submitted_stream_id=(
                                        "refresh_stream"
                                        if bool(do_async)
                                        else "current_stream"
                                    ),
                                )
                                register_submitted_group(ready_state, manifest)
                                register_group_done_event(
                                    ready_state,
                                    group_id=int(group_id),
                                    done_event=slot_group_done_event,
                                )
                                if os.environ.get(
                                    "VLLM_SPARSE_ONE_SHOT_TIMELINE_LOG",
                                    "",
                                ):
                                    try:
                                        from patches.refresh_runtime.one_shot_timeline import (
                                            append_one_shot_timeline,
                                            timeline_log_path,
                                        )

                                        append_one_shot_timeline(
                                            path=timeline_log_path(),
                                            request_id=str(req_id),
                                            epoch=int(self.step_context_epoch),
                                            chunk_id=int(group_id),
                                            buffer_id=int(
                                                getattr(
                                                    group_payloads[0],
                                                    "buf_id",
                                                    0,
                                                )
                                                or 0
                                            ),
                                            phase="producer_group_submitted",
                                            timestamp_ns=time.perf_counter_ns(),
                                            duration_us=0.0,
                                            aux_stream_enabled=bool(do_async),
                                            extra_fields={
                                                **producer_ready_summary(ready_state),
                                                "producer_group_id": int(group_id),
                                                "producer_capture_chunk": int(
                                                    capture_chunk_i
                                                ),
                                                "producer_ready_chunk": int(
                                                    ready_chunk
                                                ),
                                                "producer_micro_layer_indices": list(
                                                    layer_indices
                                                ),
                                                "producer_group_layer_indices": list(
                                                    group_layer_indices
                                                ),
                                                "producer_submitted_stream_id": (
                                                    manifest.submitted_stream_id
                                                ),
                                                "lastn1_direct_count": int(
                                                    lastn1_direct_count
                                                ),
                                                "gt1_reduce_count": int(
                                                    gt1_reduce_count
                                                ),
                                                "gt1_scalar_fallback_count": int(
                                                    gt1_scalar_fallback_count
                                                ),
                                                "capture_tap_visible_ms_or_unavailable_reason": (
                                                    "unavailable:not_measured_in_hot_path"
                                                ),
                                            },
                                        )
                                    except Exception:
                                        _log.warning(
                                            "producer-ready timeline append failed",
                                            exc_info=True,
                                        )
                producer_group_end_ns = time.perf_counter_ns()
                _append_prefill_producer_timeline(
                    phase="producer_group_end",
                    timestamp_ns=producer_group_end_ns,
                    duration_us=float(
                        producer_group_end_ns - producer_group_start_ns
                    )
                    / 1000.0,
                    extra_fields={
                        "layer_indices": list(layer_indices),
                        "lastn1_direct_count": int(lastn1_direct_count),
                        "gt1_reduce_count": int(gt1_reduce_count),
                        "gt1_scalar_fallback_count": int(gt1_scalar_fallback_count),
                    },
                )
                if finalize_slot_set and ((not one_shot_bootstrap_only) or bool(is_last_layer)):
                    finalize_event = None
                    if self.chunk_done_evt and 0 <= int(buf) < len(self.chunk_done_evt):
                        # Reuse the chunk completion event already recorded after
                        # all prefill/refresh work on the aux stream. One-shot
                        # graph publication additionally exposes this as the
                        # producer final event below.
                        finalize_event = self.chunk_done_evt[int(buf)]
                    if finalize_event is None:
                        finalize_event = torch.cuda.Event(enable_timing=False)
                        finalize_event.record(torch.cuda.current_stream(device=device))
                    if os.environ.get("VLLM_SPARSE_ONE_SHOT_TIMELINE_LOG", ""):
                        try:
                            from patches.refresh_runtime.one_shot_timeline import (
                                append_one_shot_timeline,
                                timeline_log_path,
                            )

                            _timeline_path = timeline_log_path()
                        except Exception:
                            append_one_shot_timeline = None
                            _timeline_path = None
                    else:
                        append_one_shot_timeline = None
                        _timeline_path = None

                    for slot in slots_use:
                        slot_i = int(slot)
                        if slot_i not in finalize_slot_set:
                            continue
                        req_id = submission_req_id_by_slot.get(slot_i)
                        if req_id is None or _is_free_slot_id(req_id):
                            continue
                        tracking = self._ensure_request(req_id)
                        tracking.prefill_capture_ready = False
                        tracking.prefill_capture_last_n = 0
                        pending_epoch = int(getattr(tracking, "bootstrap_pending_epoch", -1))
                        if pending_epoch != int(self.step_context_epoch):
                            tracking.bootstrap_pending_events = []
                        tracking.bootstrap_pending = True
                        tracking.bootstrap_pending_epoch = int(self.step_context_epoch)
                        if finalize_event not in tracking.bootstrap_pending_events:
                            tracking.bootstrap_pending_events.append(finalize_event)
                        if one_shot_bootstrap_only:
                            from patches.refresh_runtime.producer_ready import (
                                ProducerReadyState,
                                publish_final_event,
                                producer_ready_summary,
                            )

                            ready_state = getattr(tracking, "producer_ready_state", None)
                            if ready_state is None:
                                ready_state = ProducerReadyState()
                                tracking.producer_ready_state = ready_state
                            publish_final_event(ready_state, final_event=finalize_event)
                            timeline_extra_fields = producer_ready_summary(ready_state)
                        else:
                            timeline_extra_fields = {}
                        timeline_extra_fields.update(
                            {
                                "lastn1_direct_count": int(lastn1_direct_count),
                                "gt1_reduce_count": int(gt1_reduce_count),
                                "gt1_scalar_fallback_count": int(gt1_scalar_fallback_count),
                                "capture_tap_visible_ms_or_unavailable_reason": (
                                    "unavailable:not_measured_in_hot_path"
                                ),
                                "producer_final_event_recorded": True,
                            }
                        )
                        self._bootstrap_pending_request_ids.add(req_id)
                        if append_one_shot_timeline is not None:
                            append_one_shot_timeline(
                                path=_timeline_path,
                                request_id=str(req_id),
                                epoch=int(self.step_context_epoch),
                                chunk_id=int(getattr(group_payloads[0], "chunk_id", 0) or 0),
                                buffer_id=int(getattr(group_payloads[0], "buf_id", 0) or 0),
                                phase=(
                                    "finalize_event_deferred_to_chunk_done"
                                    if (
                                        self.chunk_done_evt
                                        and 0 <= int(buf) < len(self.chunk_done_evt)
                                        and finalize_event is self.chunk_done_evt[int(buf)]
                                    )
                                    else "finalize_event_record"
                                ),
                                timestamp_ns=time.perf_counter_ns(),
                                duration_us=0.0,
                                aux_stream_enabled=bool(do_async),
                                extra_fields=timeline_extra_fields,
                            )
                # request-wise 的 prefill_done（prompt ingest 完成）由 prepare_step_context 决定；
                # 这里是 chunk 级异步流水线，不应写入 request 全局阶段标记，避免误用。
                if publish_t0_ns is not None and prof is not None:
                    prof.prefill_publish_cpu_us += (
                        time.perf_counter_ns() - publish_t0_ns
                    ) / 1000.0

            def _run_prefill_group_contiguous_runs(
                slots_use: List[int],
                *,
                use_denoms: bool,
            ) -> None:
                if not slots_use:
                    return
                run_slots: List[int] = []
                prev_pos: Optional[int] = None
                ordered_fast_path = True
                for slot in slots_use:
                    slot_i = int(slot)
                    pos_i = pos_map.get(slot_i)
                    if pos_i is None:
                        ordered_fast_path = False
                        break
                    if prev_pos is not None and int(pos_i) <= int(prev_pos):
                        ordered_fast_path = False
                        break
                    if prev_pos is not None and int(pos_i) != int(prev_pos) + 1:
                        _run_prefill_group(run_slots, use_denoms=use_denoms)
                        run_slots = []
                    run_slots.append(int(slot_i))
                    prev_pos = int(pos_i)
                if ordered_fast_path:
                    if run_slots:
                        _run_prefill_group(run_slots, use_denoms=use_denoms)
                    return

                ordered = sorted(
                    (pos_map[int(slot)], int(slot))
                    for slot in slots_use
                    if int(slot) in pos_map
                )
                if not ordered:
                    return
                run_slots = []
                prev_pos = None
                for pos_i, slot_i in ordered:
                    if prev_pos is not None and int(pos_i) != int(prev_pos) + 1:
                        _run_prefill_group(run_slots, use_denoms=use_denoms)
                        run_slots = []
                    run_slots.append(int(slot_i))
                    prev_pos = int(pos_i)
                if run_slots:
                    _run_prefill_group(run_slots, use_denoms=use_denoms)

            if slots_lastn1 and slots_lastn_gt1:
                _run_prefill_group_contiguous_runs(slots_lastn1, use_denoms=False)
                _run_prefill_group_contiguous_runs(slots_lastn_gt1, use_denoms=True)
            else:
                use_denoms = bool(slots_lastn_gt1)
                # In one-shot mode selector_slot_list contains only requests
                # whose prompt finalizes in this flush.  Falling back to the
                # full payload here would rebuild partial chunk-prefill rows;
                # those rows are deliberately below the compact threshold and
                # have no publishable bootstrap buffer yet.  Non-one-shot mode
                # already defines selector_slot_list as slot_list_full.
                _run_prefill_group(selector_slot_list, use_denoms=use_denoms)
            if do_profile and t0_ns is not None:
                prof.prefill_cpu_us = (time.perf_counter_ns() - t0_ns) / 1000.0
                if prof.prefill_evt1 is not None:
                    prof.prefill_evt1.record(torch.cuda.current_stream(device=device))

        if do_async:
            main_stream = torch.cuda.current_stream(device=device)
            self.chunk_ready_evt[buf].record(main_stream)
            with torch.cuda.stream(self.refresh_stream):
                torch.cuda.current_stream(device=device).wait_event(self.chunk_ready_evt[buf])
                prev_profile_active = bool(getattr(self, "_refresh_profile_active", False))
                prev_active_flush_profile_accum = getattr(
                    self,
                    "_active_flush_profile_accum",
                    None,
                )
                if do_profile:
                    self._refresh_profile_active = True
                    self._active_flush_profile_accum = prof
                try:
                    _micro = _REFRESH_MICRO_PROFILE_CACHED and bool(refresh_payloads)
                    # 先绑定 capture ring storage 的生命周期，避免下一 step 清理 ring 时触发复用竞态。
                    if _micro:
                        _mt_rec0 = time.perf_counter_ns()
                    if prefill_payloads or refresh_payloads:
                        _record_stream_for_capture_bases(prefill_payloads, refresh_payloads)
                    if _micro:
                        _mt_rec1 = time.perf_counter_ns()
                    _run_prefill()
                    # prefill 细粒度 done（用于未来进一步减少不必要的等待/串行化）
                    if self.prefill_done_evt and prefill_payloads:
                        self.prefill_done_evt[buf].record(torch.cuda.current_stream(device=device))
                    if _micro:
                        _mt_ref0 = time.perf_counter_ns()
                    _run_refresh()
                    _stage_flush_compact_meta_commit_log()
                    if _micro:
                        _mt_ref1 = time.perf_counter_ns()
                        _record_us = (_mt_rec1 - _mt_rec0) / 1000.0
                        _selector_us = prof.micro_selector_ns / 1000.0
                        _rebuild_us = prof.micro_rebuild_ns / 1000.0
                        _total_us = _record_us + (_mt_ref1 - _mt_ref0) / 1000.0
                        _overhead_us = _total_us - _record_us - _selector_us - _rebuild_us
                        _REFRESH_MICRO_PROFILE_BUF.append(
                            (_record_us, _selector_us, _rebuild_us, _overhead_us, _total_us)
                        )
                        _REFRESH_MICRO_PROFILE_COUNT[0] += 1
                        if _REFRESH_MICRO_PROFILE_COUNT[0] % _REFRESH_MICRO_PROFILE_EVERY == 0:
                            _flush_micro_profile_summary()
                    # refresh 细粒度 done
                    if self.refresh_done_evt and refresh_payloads:
                        self.refresh_done_evt[buf].record(torch.cuda.current_stream(device=device))
                    self.chunk_done_evt[buf].record(torch.cuda.current_stream(device=device))
                    # FIX (bs8 async-prefill cross-stream RAW race): this async flush writes the
                    # step's resolved descriptors (row_table / seqused / affine) on refresh_stream,
                    # and the NEXT step's prefill reads them on main_stream. Without ordering, main
                    # can dereference a half-written page table (the fill_(-1) window) and hit
                    # cudaErrorIllegalAddress. `prefill_done_evt` is intentionally recorded before
                    # `_run_refresh`, so it cannot fence descriptor or compact writes published by
                    # that phase. `chunk_done_evt` is recorded after `_run_refresh` and compact-meta
                    # staging; it is the precise GPU-side completion fence for this RAW boundary.
                    # Only prefill flushes carry this: decode refresh steps and the decode hot path
                    # are untouched, and prefill stays ASYNC (not force-synced).
                    if prefill_payloads:
                        cohort_deferred_overlap_proven = bool(
                            chunk_cohort_stamped
                            and cohort_private_tape_used
                            and cohort_tape_deferred_producer_used
                            and not refresh_payloads
                        )
                        if cohort_deferred_overlap_proven:
                            if cohort_snapshot_copy_event is None:
                                raise RuntimeError(
                                    "E_SFI_CAPTURE_COHORT_COPY_EVENT: private "
                                    "tape publication has no arena-release event"
                                )
                            # The only main-stream dependency is source lifetime:
                            # once the R-stream snapshot copy completes, the arena
                            # may be rewritten. Selector/rebuild remain deferred and
                            # therefore overlap the next main-stream forward.
                            main_stream.wait_event(cohort_snapshot_copy_event)
                        else:
                            # The event was recorded on refresh_stream immediately above after all
                            # prefill/refresh work for this buf. This enqueues an ordering edge only;
                            # it neither blocks the host nor drains later work on refresh_stream.
                            main_stream.wait_event(self.chunk_done_evt[buf])
                    # WAR FIX(bs8 async-prefill compact arena): compact_arena_k/v/pos is
                    # SLOT-keyed but chunk_done_evt/_buf_pending_work_flags are BUF-keyed, so the
                    # consumer's flags==0 early-return can skip the wait -> torn compact read ->
                    # illegal address. Record a dedicated monotonic-gen-keyed compact-ready event
                    # the consumer ALWAYS waits (never buf/refresh_stream gated) before reading it.
                    if getattr(self, "_compact_arena_ready_evt", None) is None:
                        self._compact_arena_ready_evt = torch.cuda.Event(enable_timing=False)
                    self._compact_arena_ready_evt.record(torch.cuda.current_stream(device=device))
                    self._compact_arena_ready_gen = int(getattr(self, "_compact_arena_ready_gen", 0)) + 1
                    # 注意：不能在这里清空 _buf_pending_work_flags/_epoch。
                    # 这些标记用于 main_stream 的 wait_event 早退判定；若在异步提交后立即清零，
                    # 可能导致下一层/下一步错误跳过等待，从而在 ring 复用时发生竞态。
                    # 正确的清理位置是 _main_stream_wait_for_chunk_done（完成等待后）。
                finally:
                    if do_profile:
                        self._active_flush_profile_accum = (
                            prev_active_flush_profile_accum
                        )
                        self._refresh_profile_active = prev_profile_active
            if do_profile:
                # Lower overlap count tensor to float ratio at serializer
                # emit boundary (Rev 2, M1.5). .item() here is the single
                # unavoidable host sync for profile data; it occurs inside
                # `if do_profile:` so non-profile path pays nothing.
                if prof.refresh_overlap_count_tensor is not None and prof.refresh_overlap_ratio is None:
                    _overlap_cnt = int(prof.refresh_overlap_count_tensor.item())
                    _new_k = max(1, int(prof.refresh_overlap_new_k or 0))
                    prof.refresh_overlap_ratio = float(_overlap_cnt) / float(_new_k)
                (
                    prof.writer_pointer_rebuild_count,
                    prof.writer_pointer_lookup_count,
                    prof.writer_cached_pointer_hit_rate,
                    prof.writer_cached_pointer_op_count,
                    prof.writer_vector_fallback_count,
                    prof.source_ready_recorded_after_pointer_publish_count,
                ) = self._writer_pointer_telemetry_delta(writer_pointer_snapshot)
                sentence_trigger_intents = _sentence_trigger_intents_for_refresh_profile(
                    self,
                    refresh_payload_count=len(refresh_payloads),
                    refresh_payloads=refresh_payloads,
                )
                dev_index = int(device.index) if device.index is not None else -1
                self._refresh_profile_pending_by_buf[buf] = _RefreshProfilePending(
                    epoch=self.step_context_epoch,
                    chunk_id=int(chunk_id),
                    buf_id=int(buf),
                    device=str(device.type),
                    device_index=int(dev_index),
                    prefill_payloads=int(len(prefill_payloads)),
                    refresh_payloads=int(len(refresh_payloads)),
                    sentence_trigger_intents=int(sentence_trigger_intents),
                    prefill_selector_runs=int(prof.prefill_selector_runs),
                    prefill_rebuild_runs=int(prof.prefill_rebuild_runs),
                    prefill_cpu_us=float(prof.prefill_cpu_us),
                    prefill_selector_compute_cpu_us=float(
                        prof.prefill_selector_compute_cpu_us
                    ),
                    prefill_selector_post_cpu_us=float(
                        prof.prefill_selector_post_cpu_us
                    ),
                    prefill_selector_stack_cpu_us=float(
                        prof.prefill_selector_stack_cpu_us
                    ),
                    prefill_selector_validate_cpu_us=float(
                        prof.prefill_selector_validate_cpu_us
                    ),
                    prefill_selector_key_norms_cpu_us=float(
                        prof.prefill_selector_key_norms_cpu_us
                    ),
                    prefill_selector_key_norms_arena_cpu_us=float(
                        prof.prefill_selector_key_norms_arena_cpu_us
                    ),
                    prefill_selector_key_norms_direct_cpu_us=float(
                        prof.prefill_selector_key_norms_direct_cpu_us
                    ),
                    prefill_selector_key_norms_direct_prepare_cpu_us=float(
                        prof.prefill_selector_key_norms_direct_prepare_cpu_us
                    ),
                    prefill_selector_key_norms_direct_launch_cpu_us=float(
                        prof.prefill_selector_key_norms_direct_launch_cpu_us
                    ),
                    prefill_selector_key_norms_pack_cpu_us=float(
                        prof.prefill_selector_key_norms_pack_cpu_us
                    ),
                    prefill_selector_select_cpu_us=float(
                        prof.prefill_selector_select_cpu_us
                    ),
                    prefill_rebuild_cpu_us=float(prof.prefill_rebuild_cpu_us),
                    refresh_selector_cpu_us=float(prof.refresh_selector_cpu_us),
                    refresh_selector_apply_cpu_us=float(
                        prof.refresh_selector_apply_cpu_us
                    ),
                    refresh_selector_compute_cpu_us=float(prof.refresh_selector_compute_cpu_us),
                    refresh_selector_post_cpu_us=float(prof.refresh_selector_post_cpu_us),
                    refresh_selector_stack_cpu_us=float(prof.refresh_selector_stack_cpu_us),
                    refresh_selector_key_norms_cpu_us=float(
                        prof.refresh_selector_key_norms_cpu_us
                    ),
                    refresh_selector_key_norms_arena_cpu_us=float(
                        prof.refresh_selector_key_norms_arena_cpu_us
                    ),
                    refresh_selector_key_norms_direct_cpu_us=float(
                        prof.refresh_selector_key_norms_direct_cpu_us
                    ),
                    refresh_selector_key_norms_direct_prepare_cpu_us=float(
                        prof.refresh_selector_key_norms_direct_prepare_cpu_us
                    ),
                    refresh_selector_key_norms_direct_launch_cpu_us=float(
                        prof.refresh_selector_key_norms_direct_launch_cpu_us
                    ),
                    refresh_selector_key_norms_pack_cpu_us=float(
                        prof.refresh_selector_key_norms_pack_cpu_us
                    ),
                    refresh_rebuild_cpu_us=float(prof.refresh_rebuild_cpu_us),
                    refresh_total_cpu_us=float(prof.refresh_total_cpu_us),
                    refresh_rebuild_enqueue_cpu_us=float(
                        prof.refresh_rebuild_enqueue_cpu_us
                    ),
                    refresh_rebuild_compact_cpu_us=float(
                        prof.refresh_rebuild_compact_cpu_us
                    ),
                    prefill_evt0=prof.prefill_evt0,
                    prefill_evt1=prof.prefill_evt1,
                    prefill_selector_evt_pairs=tuple(prof.prefill_selector_evt_pairs),
                    prefill_gather_evt_pairs=tuple(prof.prefill_gather_evt_pairs),
                    prefill_key_norms_preproc_evt_pairs=tuple(
                        prof.prefill_key_norms_preproc_evt_pairs
                    ),
                    prefill_key_norms_evt_pairs=tuple(prof.prefill_key_norms_evt_pairs),
                    prefill_key_norms_h2d_evt_pairs=tuple(
                        prof.prefill_key_norms_h2d_evt_pairs
                    ),
                    prefill_key_norms_delta_evt_pairs=tuple(
                        prof.prefill_key_norms_delta_evt_pairs
                    ),
                    prefill_key_norms_pack_evt_pairs=tuple(
                        prof.prefill_key_norms_pack_evt_pairs
                    ),
                    prefill_log_s_evt_pairs=tuple(prof.prefill_log_s_evt_pairs),
                    prefill_log_s_triton_evt_pairs=tuple(
                        prof.prefill_log_s_triton_evt_pairs
                    ),
                    prefill_log_s_mask_evt_pairs=tuple(
                        prof.prefill_log_s_mask_evt_pairs
                    ),
                    prefill_log_s_cross_evt_pairs=tuple(
                        prof.prefill_log_s_cross_evt_pairs
                    ),
                    prefill_topk_evt_pairs=tuple(prof.prefill_topk_evt_pairs),
                    prefill_preproc_evt_pairs=tuple(prof.prefill_preproc_evt_pairs),
                    prefill_seq_full_evt_pairs=tuple(prof.prefill_seq_full_evt_pairs),
                    prefill_pure_preproc_evt_pairs=tuple(
                        prof.prefill_pure_preproc_evt_pairs
                    ),
                    prefill_selector_bounds_evt_pairs=tuple(
                        prof.prefill_selector_bounds_evt_pairs
                    ),
                    prefill_selector_pipeline_evt_pairs=tuple(
                        prof.prefill_selector_pipeline_evt_pairs
                    ),
                    prefill_rebuild_evt_pairs=tuple(prof.prefill_rebuild_evt_pairs),
                    prefill_group_evt_pairs=tuple(prof.prefill_group_evt_pairs),
                    prefill_publish_cpu_us=float(prof.prefill_publish_cpu_us),
                    prefill_key_norms_delta_total_tokens=int(
                        prof.prefill_key_norms_delta_total_tokens
                    ),
                    prefill_key_norms_delta_max_tokens=int(
                        prof.prefill_key_norms_delta_max_tokens
                    ),
                    prefill_key_norms_delta_layers=int(
                        prof.prefill_key_norms_delta_layers
                    ),
                    refresh_sel_evt0=prof.refresh_sel_evt0,
                    refresh_sel_evt1=prof.refresh_sel_evt1,
                    refresh_rebuild_evt0=prof.refresh_rebuild_evt0,
                    refresh_rebuild_evt1=prof.refresh_rebuild_evt1,
                    refresh_gather_evt0=prof.refresh_gather_evt0,
                    refresh_gather_evt1=prof.refresh_gather_evt1,
                    refresh_key_norms_preproc_evt0=prof.refresh_key_norms_preproc_evt0,
                    refresh_key_norms_preproc_evt1=prof.refresh_key_norms_preproc_evt1,
                    refresh_key_norms_evt0=prof.refresh_key_norms_evt0,
                    refresh_key_norms_evt1=prof.refresh_key_norms_evt1,
                    refresh_key_norms_h2d_evt0=prof.refresh_key_norms_h2d_evt0,
                    refresh_key_norms_h2d_evt1=prof.refresh_key_norms_h2d_evt1,
                    refresh_key_norms_delta_evt0=prof.refresh_key_norms_delta_evt0,
                    refresh_key_norms_delta_evt1=prof.refresh_key_norms_delta_evt1,
                    refresh_key_norms_pack_evt0=prof.refresh_key_norms_pack_evt0,
                    refresh_key_norms_pack_evt1=prof.refresh_key_norms_pack_evt1,
                    refresh_key_norms_delta_total_tokens=int(
                        prof.refresh_key_norms_delta_total_tokens
                    ),
                    refresh_key_norms_delta_max_tokens=int(
                        prof.refresh_key_norms_delta_max_tokens
                    ),
                    refresh_key_norms_delta_layers=int(
                        prof.refresh_key_norms_delta_layers
                    ),
                    refresh_log_s_evt0=prof.refresh_log_s_evt0,
                    refresh_log_s_evt1=prof.refresh_log_s_evt1,
                    refresh_log_s_triton_evt0=prof.refresh_log_s_triton_evt0,
                    refresh_log_s_triton_evt1=prof.refresh_log_s_triton_evt1,
                    refresh_log_s_mask_evt0=prof.refresh_log_s_mask_evt0,
                    refresh_log_s_mask_evt1=prof.refresh_log_s_mask_evt1,
                    refresh_log_s_cross_evt0=prof.refresh_log_s_cross_evt0,
                    refresh_log_s_cross_evt1=prof.refresh_log_s_cross_evt1,
                    refresh_topk_evt0=prof.refresh_topk_evt0,
                    refresh_topk_evt1=prof.refresh_topk_evt1,
                    refresh_preproc_evt0=prof.refresh_preproc_evt0,
                    refresh_preproc_evt1=prof.refresh_preproc_evt1,
                    refresh_seq_full_evt0=prof.refresh_seq_full_evt0,
                    refresh_seq_full_evt1=prof.refresh_seq_full_evt1,
                    refresh_pure_preproc_evt0=prof.refresh_pure_preproc_evt0,
                    refresh_pure_preproc_evt1=prof.refresh_pure_preproc_evt1,
                    refresh_selector_bounds_evt0=prof.refresh_selector_bounds_evt0,
                    refresh_selector_bounds_evt1=prof.refresh_selector_bounds_evt1,
                    refresh_selector_pipeline_evt0=prof.refresh_selector_pipeline_evt0,
                    refresh_selector_pipeline_evt1=prof.refresh_selector_pipeline_evt1,
                    **{
                        f"async_producer_{stage}_evt_pairs": tuple(
                            getattr(prof, f"async_producer_{stage}_evt_pairs")
                        )
                        for stage in ASYNC_PRODUCER_GPU_PROFILE_STAGES
                    },
                    rebuild_head_dim=int(prof.rebuild_head_dim),
                    rebuild_kv_dtype=str(prof.rebuild_kv_dtype),
                    rebuild_block_size=int(prof.rebuild_block_size),
                    rebuild_stride_tokens=int(prof.rebuild_stride_tokens),
                    rebuild_selected_k=int(prof.rebuild_selected_k),
                    rebuild_num_kv_heads=int(prof.rebuild_num_kv_heads),
                    rebuild_batch_slots=int(prof.rebuild_batch_slots),
                    capture_kv_len_total=int(prof.capture_kv_len_total),
                    writer_pointer_rebuild_count=int(prof.writer_pointer_rebuild_count),
                    writer_pointer_lookup_count=int(prof.writer_pointer_lookup_count),
                    writer_cached_pointer_hit_rate=float(prof.writer_cached_pointer_hit_rate),
                    writer_cached_pointer_op_count=int(prof.writer_cached_pointer_op_count),
                    writer_vector_fallback_count=int(prof.writer_vector_fallback_count),
                    writer_kernel_variant=str(prof.writer_kernel_variant),
                    writer_actual_tokens=int(prof.writer_actual_tokens),
                    writer_sink_tokens=int(prof.writer_sink_tokens),
                    writer_persist_tokens=int(prof.writer_persist_tokens),
                    writer_sink_io_bytes=int(prof.writer_sink_io_bytes),
                    writer_persist_io_bytes=int(prof.writer_persist_io_bytes),
                    writer_token_tiles_estimated=int(prof.writer_token_tiles_estimated),
                    writer_active_token_tiles_estimated=int(prof.writer_active_token_tiles_estimated),
                    writer_cta_count_estimated=int(prof.writer_cta_count_estimated),
                    writer_active_cta_count_estimated=int(prof.writer_active_cta_count_estimated),
                    writer_tokens_per_cta=int(prof.writer_tokens_per_cta),
                    writer_k_read_bytes=int(prof.writer_k_read_bytes),
                    writer_v_read_bytes=int(prof.writer_v_read_bytes),
                    writer_k_write_bytes=int(prof.writer_k_write_bytes),
                    writer_v_write_bytes=int(prof.writer_v_write_bytes),
                    writer_pos_write_bytes=int(prof.writer_pos_write_bytes),
                    writer_total_io_bytes=int(prof.writer_total_io_bytes),
                    writer_effective_io_gbps=float(prof.writer_effective_io_gbps),
                    selected_indices_materialized_bytes=int(
                        prof.selected_indices_materialized_bytes
                    ),
                    selected_indices_io_bytes=int(prof.selected_indices_io_bytes),
                    selector_writer_current_path_count=int(
                        prof.selector_writer_current_path_count
                    ),
                    selector_writer_boundary_cpu_us=float(
                        prof.selector_writer_boundary_cpu_us
                    ),
                    selected_boundary_lower_bound_ms_per_group=float(
                        prof.selected_boundary_lower_bound_ms_per_group
                    ),
                    predicted_front_early_step_improvement_ms=float(
                        prof.predicted_front_early_step_improvement_ms
                    ),
                    residual_fixed_capture_control_ms=float(
                        prof.residual_fixed_capture_control_ms
                    ),
                    source_ready_recorded_after_pointer_publish_count=int(
                        prof.source_ready_recorded_after_pointer_publish_count
                    ),
                    lastn1_direct_count=int(lastn1_direct_count),
                    gt1_reduce_count=int(gt1_reduce_count),
                    gt1_scalar_fallback_count=int(gt1_scalar_fallback_count),
                    refresh_rebuild_budget_before=int(
                        prof.refresh_rebuild_budget_before
                    ),
                    refresh_rebuild_budget_after=int(
                        prof.refresh_rebuild_budget_after
                    ),
                    refresh_rebuild_enqueued_count=int(
                        prof.refresh_rebuild_enqueued_count
                    ),
                    refresh_rebuild_inline_count=int(
                        prof.refresh_rebuild_inline_count
                    ),
                    refresh_rebuild_pending_queue_size=int(
                        prof.refresh_rebuild_pending_queue_size
                    ),
                    refresh_rebuild_coalesced_count=int(
                        prof.refresh_rebuild_coalesced_count
                    ),
                    deadline_rebuild_drop_finished_count=int(
                        getattr(self, "_deadline_rebuild_drop_finished_count", 0)
                    ),
                    deadline_rebuild_drain_finish_count=int(
                        getattr(self, "_deadline_rebuild_drain_finish_count", 0)
                    ),
                    deadline_rebuild_partial_finish_count=int(
                        getattr(self, "_deadline_rebuild_partial_finish_count", 0)
                    ),
                    deadline_rebuild_drain_submit_count=int(
                        getattr(self, "_deadline_rebuild_drain_submit_count", 0)
                    ),
                    deadline_rebuild_drain_submit_decode_step_min=int(
                        getattr(
                            self,
                            "_deadline_rebuild_drain_submit_decode_step_min",
                            -1,
                        )
                    ),
                    deadline_rebuild_drain_submit_decode_step_max=int(
                        getattr(
                            self,
                            "_deadline_rebuild_drain_submit_decode_step_max",
                            -1,
                        )
                    ),
                    deadline_rebuild_drain_submit_decode_steps=(
                        self._take_deadline_rebuild_drain_submit_decode_steps()
                    ),
                    deadline_deferred_selector_compute_count=int(
                        getattr(
                            self,
                            "_deadline_deferred_selector_compute_count",
                            0,
                        )
                    ),
                    deadline_deferred_selector_compute_cpu_us_total=float(
                        getattr(
                            self,
                            "_deadline_deferred_selector_compute_cpu_us_total",
                            0.0,
                        )
                    ),
                    deadline_deferred_selector_compute_cpu_us_max=float(
                        getattr(
                            self,
                            "_deadline_deferred_selector_compute_cpu_us_max",
                            0.0,
                        )
                    ),
                    deadline_deferred_selector_inner_compute_cpu_us_total=float(
                        getattr(
                            self,
                            "_deadline_deferred_selector_inner_compute_cpu_us_total",
                            0.0,
                        )
                    ),
                    deadline_deferred_selector_inner_compute_cpu_us_max=float(
                        getattr(
                            self,
                            "_deadline_deferred_selector_inner_compute_cpu_us_max",
                            0.0,
                        )
                    ),
                    deadline_deferred_selector_stack_cpu_us_total=float(
                        getattr(
                            self,
                            "_deadline_deferred_selector_stack_cpu_us_total",
                            0.0,
                        )
                    ),
                    deadline_deferred_selector_stack_cpu_us_max=float(
                        getattr(
                            self,
                            "_deadline_deferred_selector_stack_cpu_us_max",
                            0.0,
                        )
                    ),
                    deadline_deferred_selector_validate_cpu_us_total=float(
                        getattr(
                            self,
                            "_deadline_deferred_selector_validate_cpu_us_total",
                            0.0,
                        )
                    ),
                    deadline_deferred_selector_validate_cpu_us_max=float(
                        getattr(
                            self,
                            "_deadline_deferred_selector_validate_cpu_us_max",
                            0.0,
                        )
                    ),
                    deadline_deferred_selector_key_norms_cpu_us_total=float(
                        getattr(
                            self,
                            "_deadline_deferred_selector_key_norms_cpu_us_total",
                            0.0,
                        )
                    ),
                    deadline_deferred_selector_key_norms_cpu_us_max=float(
                        getattr(
                            self,
                            "_deadline_deferred_selector_key_norms_cpu_us_max",
                            0.0,
                        )
                    ),
                    deadline_deferred_selector_key_norms_arena_cpu_us_total=float(
                        getattr(
                            self,
                            "_deadline_deferred_selector_key_norms_arena_cpu_us_total",
                            0.0,
                        )
                    ),
                    deadline_deferred_selector_key_norms_arena_cpu_us_max=float(
                        getattr(
                            self,
                            "_deadline_deferred_selector_key_norms_arena_cpu_us_max",
                            0.0,
                        )
                    ),
                    deadline_deferred_selector_key_norms_direct_cpu_us_total=float(
                        getattr(
                            self,
                            "_deadline_deferred_selector_key_norms_direct_cpu_us_total",
                            0.0,
                        )
                    ),
                    deadline_deferred_selector_key_norms_direct_cpu_us_max=float(
                        getattr(
                            self,
                            "_deadline_deferred_selector_key_norms_direct_cpu_us_max",
                            0.0,
                        )
                    ),
                    deadline_deferred_selector_key_norms_direct_prepare_cpu_us_total=float(
                        getattr(
                            self,
                            "_deadline_deferred_selector_key_norms_direct_prepare_cpu_us_total",
                            0.0,
                        )
                    ),
                    deadline_deferred_selector_key_norms_direct_prepare_cpu_us_max=float(
                        getattr(
                            self,
                            "_deadline_deferred_selector_key_norms_direct_prepare_cpu_us_max",
                            0.0,
                        )
                    ),
                    deadline_deferred_selector_key_norms_direct_launch_cpu_us_total=float(
                        getattr(
                            self,
                            "_deadline_deferred_selector_key_norms_direct_launch_cpu_us_total",
                            0.0,
                        )
                    ),
                    deadline_deferred_selector_key_norms_direct_launch_cpu_us_max=float(
                        getattr(
                            self,
                            "_deadline_deferred_selector_key_norms_direct_launch_cpu_us_max",
                            0.0,
                        )
                    ),
                    deadline_deferred_selector_key_norms_pack_cpu_us_total=float(
                        getattr(
                            self,
                            "_deadline_deferred_selector_key_norms_pack_cpu_us_total",
                            0.0,
                        )
                    ),
                    deadline_deferred_selector_key_norms_pack_cpu_us_max=float(
                        getattr(
                            self,
                            "_deadline_deferred_selector_key_norms_pack_cpu_us_max",
                            0.0,
                        )
                    ),
                    deadline_deferred_selector_select_cpu_us_total=float(
                        getattr(
                            self,
                            "_deadline_deferred_selector_select_cpu_us_total",
                            0.0,
                        )
                    ),
                    deadline_deferred_selector_select_cpu_us_max=float(
                        getattr(
                            self,
                            "_deadline_deferred_selector_select_cpu_us_max",
                            0.0,
                        )
                    ),
                    deadline_deferred_selector_post_cpu_us_total=float(
                        getattr(
                            self,
                            "_deadline_deferred_selector_post_cpu_us_total",
                            0.0,
                        )
                    ),
                    deadline_deferred_selector_post_cpu_us_max=float(
                        getattr(
                            self,
                            "_deadline_deferred_selector_post_cpu_us_max",
                            0.0,
                        )
                    ),
                    deadline_deferred_selector_wrapper_gap_cpu_us_total=float(
                        getattr(
                            self,
                            "_deadline_deferred_selector_wrapper_gap_cpu_us_total",
                            0.0,
                        )
                    ),
                    deadline_deferred_selector_wrapper_gap_cpu_us_max=float(
                        getattr(
                            self,
                            "_deadline_deferred_selector_wrapper_gap_cpu_us_max",
                            0.0,
                        )
                    ),
                    deadline_deferred_producer_detail_us=dict(
                        getattr(
                            self,
                            "_deadline_deferred_producer_detail_us",
                            None,
                        )
                        or {}
                    ),
                    deadline_async_producer_body_count=int(
                        getattr(self, "_deadline_async_producer_body_count", 0)
                    ),
                    deadline_async_producer_body_cpu_us_total=float(
                        getattr(
                            self,
                            "_deadline_async_producer_body_cpu_us_total",
                            0.0,
                        )
                    ),
                    deadline_async_producer_body_cpu_us_max=float(
                        getattr(
                            self,
                            "_deadline_async_producer_body_cpu_us_max",
                            0.0,
                        )
                    ),
                    deadline_async_producer_selector_count=int(
                        getattr(self, "_deadline_async_producer_selector_count", 0)
                    ),
                    deadline_async_producer_selector_cpu_us_total=float(
                        getattr(
                            self,
                            "_deadline_async_producer_selector_cpu_us_total",
                            0.0,
                        )
                    ),
                    deadline_async_producer_selector_cpu_us_max=float(
                        getattr(
                            self,
                            "_deadline_async_producer_selector_cpu_us_max",
                            0.0,
                        )
                    ),
                    deadline_async_producer_key_norms_delta_count=int(
                        getattr(
                            self,
                            "_deadline_async_producer_key_norms_delta_count",
                            0,
                        )
                    ),
                    deadline_async_producer_key_norms_delta_total_tokens_total=int(
                        getattr(
                            self,
                            "_deadline_async_producer_key_norms_delta_total_tokens_total",
                            0,
                        )
                    ),
                    deadline_async_producer_key_norms_delta_max_tokens_max=int(
                        getattr(
                            self,
                            "_deadline_async_producer_key_norms_delta_max_tokens_max",
                            -1,
                        )
                    ),
                    deadline_async_producer_key_norms_delta_layers_total=int(
                        getattr(
                            self,
                            "_deadline_async_producer_key_norms_delta_layers_total",
                            0,
                        )
                    ),
                    deadline_async_producer_writer_count=int(
                        getattr(self, "_deadline_async_producer_writer_count", 0)
                    ),
                    deadline_async_producer_writer_cpu_us_total=float(
                        getattr(
                            self,
                            "_deadline_async_producer_writer_cpu_us_total",
                            0.0,
                        )
                    ),
                    deadline_async_producer_writer_cpu_us_max=float(
                        getattr(
                            self,
                            "_deadline_async_producer_writer_cpu_us_max",
                            0.0,
                        )
                    ),
                    deadline_async_producer_graph_replay_count=int(
                        getattr(
                            self,
                            "_deadline_async_producer_graph_replay_count",
                            0,
                        )
                    ),
                    deadline_async_producer_graph_replay_cpu_us_total=float(
                        getattr(
                            self,
                            "_deadline_async_producer_graph_replay_cpu_us_total",
                            0.0,
                        )
                    ),
                    deadline_async_producer_graph_replay_cpu_us_max=float(
                        getattr(
                            self,
                            "_deadline_async_producer_graph_replay_cpu_us_max",
                            0.0,
                        )
                    ),
                    deadline_async_producer_graph_replay_stage_selector_inputs_count=int(
                        getattr(
                            self,
                            "_deadline_async_producer_graph_replay_stage_selector_inputs_count",
                            0,
                        )
                    ),
                    deadline_async_producer_graph_replay_stage_selector_inputs_cpu_us_total=float(
                        getattr(
                            self,
                            "_deadline_async_producer_graph_replay_stage_selector_inputs_cpu_us_total",
                            0.0,
                        )
                    ),
                    deadline_async_producer_graph_replay_stage_selector_inputs_cpu_us_max=float(
                        getattr(
                            self,
                            "_deadline_async_producer_graph_replay_stage_selector_inputs_cpu_us_max",
                            0.0,
                        )
                    ),
                    deadline_async_producer_graph_replay_prepare_writer_count=int(
                        getattr(
                            self,
                            "_deadline_async_producer_graph_replay_prepare_writer_count",
                            0,
                        )
                    ),
                    deadline_async_producer_graph_replay_prepare_writer_cpu_us_total=float(
                        getattr(
                            self,
                            "_deadline_async_producer_graph_replay_prepare_writer_cpu_us_total",
                            0.0,
                        )
                    ),
                    deadline_async_producer_graph_replay_prepare_writer_cpu_us_max=float(
                        getattr(
                            self,
                            "_deadline_async_producer_graph_replay_prepare_writer_cpu_us_max",
                            0.0,
                        )
                    ),
                    deadline_async_producer_graph_replay_stage_lens_count=int(
                        getattr(
                            self,
                            "_deadline_async_producer_graph_replay_stage_lens_count",
                            0,
                        )
                    ),
                    deadline_async_producer_graph_replay_stage_lens_cpu_us_total=float(
                        getattr(
                            self,
                            "_deadline_async_producer_graph_replay_stage_lens_cpu_us_total",
                            0.0,
                        )
                    ),
                    deadline_async_producer_graph_replay_stage_lens_cpu_us_max=float(
                        getattr(
                            self,
                            "_deadline_async_producer_graph_replay_stage_lens_cpu_us_max",
                            0.0,
                        )
                    ),
                    deadline_async_producer_graph_replay_prepare_events_count=int(
                        getattr(
                            self,
                            "_deadline_async_producer_graph_replay_prepare_events_count",
                            0,
                        )
                    ),
                    deadline_async_producer_graph_replay_prepare_events_cpu_us_total=float(
                        getattr(
                            self,
                            "_deadline_async_producer_graph_replay_prepare_events_cpu_us_total",
                            0.0,
                        )
                    ),
                    deadline_async_producer_graph_replay_prepare_events_cpu_us_max=float(
                        getattr(
                            self,
                            "_deadline_async_producer_graph_replay_prepare_events_cpu_us_max",
                            0.0,
                        )
                    ),
                    deadline_async_producer_graph_replay_graph_count=int(
                        getattr(
                            self,
                            "_deadline_async_producer_graph_replay_graph_count",
                            0,
                        )
                    ),
                    deadline_async_producer_graph_replay_graph_cpu_us_total=float(
                        getattr(
                            self,
                            "_deadline_async_producer_graph_replay_graph_cpu_us_total",
                            0.0,
                        )
                    ),
                    deadline_async_producer_graph_replay_graph_cpu_us_max=float(
                        getattr(
                            self,
                            "_deadline_async_producer_graph_replay_graph_cpu_us_max",
                            0.0,
                        )
                    ),
                    deadline_async_producer_graph_capture_count=int(
                        getattr(
                            self,
                            "_deadline_async_producer_graph_capture_count",
                            0,
                        )
                    ),
                    deadline_async_producer_graph_capture_cpu_us_total=float(
                        getattr(
                            self,
                            "_deadline_async_producer_graph_capture_cpu_us_total",
                            0.0,
                        )
                    ),
                    deadline_async_producer_graph_capture_cpu_us_max=float(
                        getattr(
                            self,
                            "_deadline_async_producer_graph_capture_cpu_us_max",
                            0.0,
                        )
                    ),
                    deadline_async_producer_result_precomputed_count=int(
                        getattr(
                            self,
                            "_deadline_async_producer_result_precomputed_count",
                            0,
                        )
                    ),
                    refresh_rebuild_delay_max=int(
                        getattr(self, "_refresh_rebuild_delay_max", 0)
                    ),
                    producer_work_target_layer_start=int(
                        _producer_work_int("target_layer_start")
                    ),
                    producer_work_target_layer_end=int(
                        _producer_work_int("target_layer_end")
                    ),
                    producer_work_decode_step_min=int(
                        _producer_work_int("decode_step_min")
                    ),
                    producer_work_decode_step_max=int(
                        _producer_work_int("decode_step_max")
                    ),
                    producer_work_ready_epoch=int(_producer_work_int("ready_epoch")),
                    producer_work_deadline_epoch=int(
                        _producer_work_int("deadline_epoch")
                    ),
                    producer_work_deadline_handle_id=int(
                        _producer_work_int("deadline_handle_id")
                    ),
                    producer_work_deadline_slack_steps=int(
                        _producer_work_int("deadline_slack_steps")
                    ),
                    producer_work_can_drop=int(_producer_work_int("can_drop", 0)),
                    producer_work_can_coalesce=int(
                        _producer_work_int("can_coalesce", 0)
                    ),
                    producer_work_admission_reason=str(
                        _producer_work_str("admission_reason")
                    ),
                    refresh_overlap_ratio=prof.refresh_overlap_ratio,
                    refresh_overlap_new_k=prof.refresh_overlap_new_k,
                    refresh_overlap_old_k=prof.refresh_overlap_old_k,
                )
        else:
            prev_profile_active = bool(getattr(self, "_refresh_profile_active", False))
            prev_active_flush_profile_accum = getattr(
                self,
                "_active_flush_profile_accum",
                None,
            )
            if do_profile:
                self._refresh_profile_active = True
                self._active_flush_profile_accum = prof
            try:
                _run_prefill()
                _micro = _REFRESH_MICRO_PROFILE_CACHED and bool(refresh_payloads)
                if _micro:
                    _mt_ref0 = time.perf_counter_ns()
                _run_refresh()
                if _micro:
                    _mt_ref1 = time.perf_counter_ns()
                    _selector_us = prof.micro_selector_ns / 1000.0
                    _rebuild_us = prof.micro_rebuild_ns / 1000.0
                    _total_us = (_mt_ref1 - _mt_ref0) / 1000.0
                    _overhead_us = _total_us - _selector_us - _rebuild_us
                    _REFRESH_MICRO_PROFILE_BUF.append(
                        (0.0, _selector_us, _rebuild_us, _overhead_us, _total_us)
                    )
                    _REFRESH_MICRO_PROFILE_COUNT[0] += 1
                    if _REFRESH_MICRO_PROFILE_COUNT[0] % _REFRESH_MICRO_PROFILE_EVERY == 0:
                        _flush_micro_profile_summary()
            finally:
                if do_profile:
                    self._active_flush_profile_accum = (
                        prev_active_flush_profile_accum
                    )
                    self._refresh_profile_active = prev_profile_active
            if do_profile:
                dev_index = int(device.index) if device.index is not None else -1
                # 同步 profiling：为拿到 CUDA event timing，这里必须显式 synchronize（仅诊断用）。
                try:
                    torch.cuda.current_stream(device=device).synchronize()
                except Exception:
                    _log.warning("sync profiling: cuda stream synchronize failed", exc_info=True)
                    # profiling only: keep decode path alive even if timing sync fails
                    pass
                # Lower overlap count tensor to float ratio at serializer
                # emit boundary (Rev 2, M1.5). Post-synchronize so .item() is
                # a host read from an already-completed computation.
                if prof.refresh_overlap_count_tensor is not None and prof.refresh_overlap_ratio is None:
                    _overlap_cnt = int(prof.refresh_overlap_count_tensor.item())
                    _new_k = max(1, int(prof.refresh_overlap_new_k or 0))
                    prof.refresh_overlap_ratio = float(_overlap_cnt) / float(_new_k)
                (
                    prof.writer_pointer_rebuild_count,
                    prof.writer_pointer_lookup_count,
                    prof.writer_cached_pointer_hit_rate,
                    prof.writer_cached_pointer_op_count,
                    prof.writer_vector_fallback_count,
                    prof.source_ready_recorded_after_pointer_publish_count,
                ) = self._writer_pointer_telemetry_delta(writer_pointer_snapshot)

                def _evt_ms(evt0: Optional[torch.cuda.Event], evt1: Optional[torch.cuda.Event]) -> Optional[float]:
                    if evt0 is None or evt1 is None:
                        return None
                    try:
                        return float(evt0.elapsed_time(evt1))
                    except Exception:
                        _log.warning("sync profiling: elapsed_time() failed", exc_info=True)
                        return None

                def _evt_pairs_ms(
                    pairs: Sequence[Tuple[torch.cuda.Event, torch.cuda.Event]],
                ) -> Optional[float]:
                    values: List[float] = []
                    for evt0, evt1 in pairs:
                        value = _evt_ms(evt0, evt1)
                        if value is not None:
                            values.append(float(value))
                    return float(sum(values)) if values else None

                def _prefill_group_profile_fields(
                    prefill_start_evt: Optional[torch.cuda.Event],
                    pairs: Sequence[Tuple[int, torch.cuda.Event, torch.cuda.Event]],
                ) -> dict[str, float | int]:
                    fields: dict[str, float | int] = {"prefill_group_count": 0}
                    group_ids = sorted({int(group_id) for group_id, _, _ in pairs})
                    fields["prefill_group_count"] = int(len(group_ids))
                    for group_id in group_ids:
                        duration_ms = 0.0
                        duration_seen = False
                        done_since_start_ms: Optional[float] = None
                        for pair_group_id, evt0, evt1 in pairs:
                            if int(pair_group_id) != int(group_id):
                                continue
                            value = _evt_ms(evt0, evt1)
                            if value is not None:
                                duration_ms += float(value)
                                duration_seen = True
                            if prefill_start_evt is not None:
                                done_value = _evt_ms(prefill_start_evt, evt1)
                                if done_value is not None:
                                    done_since_start_ms = max(
                                        float(done_since_start_ms or 0.0),
                                        float(done_value),
                                    )
                        if duration_seen:
                            fields[f"prefill_group{group_id}_gpu_ms"] = float(duration_ms)
                        if done_since_start_ms is not None:
                            fields[
                                f"prefill_group{group_id}_done_since_prefill_start_ms"
                            ] = float(done_since_start_ms)
                    return fields

                record = {
                    "pid": int(os.getpid()),
                    "ctrl_step": int(getattr(self, "step", -1)),
                    "ts_ns": int(time.time_ns()),
                    "epoch": self.step_context_epoch,
                    "wait_epoch": None,
                    "chunk_id": int(chunk_id),
                    "buf_id": int(buf),
                    "device": str(device.type),
                    "device_index": int(dev_index),
                    "prefill_payloads": int(len(prefill_payloads)),
                    "refresh_payloads": int(len(refresh_payloads)),
                    "prefill_selector_runs": int(prof.prefill_selector_runs),
                    "prefill_rebuild_runs": int(prof.prefill_rebuild_runs),
                    "prefill_cpu_us": float(prof.prefill_cpu_us),
                    "prefill_selector_compute_cpu_us": float(
                        prof.prefill_selector_compute_cpu_us
                    ),
                    "prefill_selector_post_cpu_us": float(
                        prof.prefill_selector_post_cpu_us
                    ),
                    "prefill_selector_stack_cpu_us": float(
                        prof.prefill_selector_stack_cpu_us
                    ),
                    "prefill_selector_validate_cpu_us": float(
                        prof.prefill_selector_validate_cpu_us
                    ),
                    "prefill_selector_key_norms_cpu_us": float(
                        prof.prefill_selector_key_norms_cpu_us
                    ),
                    "prefill_selector_key_norms_arena_cpu_us": float(
                        prof.prefill_selector_key_norms_arena_cpu_us
                    ),
                    "prefill_selector_key_norms_direct_cpu_us": float(
                        prof.prefill_selector_key_norms_direct_cpu_us
                    ),
                    "prefill_selector_key_norms_direct_prepare_cpu_us": float(
                        prof.prefill_selector_key_norms_direct_prepare_cpu_us
                    ),
                    "prefill_selector_key_norms_direct_launch_cpu_us": float(
                        prof.prefill_selector_key_norms_direct_launch_cpu_us
                    ),
                    "prefill_selector_key_norms_pack_cpu_us": float(
                        prof.prefill_selector_key_norms_pack_cpu_us
                    ),
                    "prefill_selector_select_cpu_us": float(
                        prof.prefill_selector_select_cpu_us
                    ),
                    "prefill_rebuild_cpu_us": float(prof.prefill_rebuild_cpu_us),
                    "refresh_selector_cpu_us": float(prof.refresh_selector_cpu_us),
                    "refresh_selector_apply_cpu_us": float(
                        prof.refresh_selector_apply_cpu_us
                    ),
                    "refresh_selector_stack_cpu_us": float(prof.refresh_selector_stack_cpu_us),
                    "refresh_selector_key_norms_cpu_us": float(
                        prof.refresh_selector_key_norms_cpu_us
                    ),
                    "refresh_selector_key_norms_arena_cpu_us": float(
                        prof.refresh_selector_key_norms_arena_cpu_us
                    ),
                    "refresh_selector_key_norms_direct_cpu_us": float(
                        prof.refresh_selector_key_norms_direct_cpu_us
                    ),
                    "refresh_selector_key_norms_direct_prepare_cpu_us": float(
                        prof.refresh_selector_key_norms_direct_prepare_cpu_us
                    ),
                    "refresh_selector_key_norms_direct_launch_cpu_us": float(
                        prof.refresh_selector_key_norms_direct_launch_cpu_us
                    ),
                    "refresh_selector_key_norms_pack_cpu_us": float(
                        prof.refresh_selector_key_norms_pack_cpu_us
                    ),
                    "refresh_rebuild_cpu_us": float(prof.refresh_rebuild_cpu_us),
                    "refresh_total_cpu_us": float(prof.refresh_total_cpu_us),
                    "refresh_rebuild_enqueue_cpu_us": float(
                        prof.refresh_rebuild_enqueue_cpu_us
                    ),
                    "refresh_rebuild_compact_cpu_us": float(
                        prof.refresh_rebuild_compact_cpu_us
                    ),
                    "prefill_gpu_ms": _evt_ms(prof.prefill_evt0, prof.prefill_evt1),
                    "prefill_selector_gpu_ms": _evt_pairs_ms(prof.prefill_selector_evt_pairs),
                    "prefill_gather_gpu_ms": _evt_pairs_ms(prof.prefill_gather_evt_pairs),
                    "prefill_key_norms_preproc_gpu_ms": _evt_pairs_ms(
                        prof.prefill_key_norms_preproc_evt_pairs
                    ),
                    "prefill_key_norms_gpu_ms": _evt_pairs_ms(
                        prof.prefill_key_norms_evt_pairs
                    ),
                    "prefill_key_norms_h2d_gpu_ms": _evt_pairs_ms(
                        prof.prefill_key_norms_h2d_evt_pairs
                    ),
                    "prefill_key_norms_delta_gpu_ms": _evt_pairs_ms(
                        prof.prefill_key_norms_delta_evt_pairs
                    ),
                    "prefill_key_norms_pack_gpu_ms": _evt_pairs_ms(
                        prof.prefill_key_norms_pack_evt_pairs
                    ),
                    "prefill_log_s_gpu_ms": _evt_pairs_ms(prof.prefill_log_s_evt_pairs),
                    "prefill_log_s_triton_gpu_ms": _evt_pairs_ms(
                        prof.prefill_log_s_triton_evt_pairs
                    ),
                    "prefill_log_s_mask_gpu_ms": _evt_pairs_ms(
                        prof.prefill_log_s_mask_evt_pairs
                    ),
                    "prefill_log_s_cross_gpu_ms": _evt_pairs_ms(
                        prof.prefill_log_s_cross_evt_pairs
                    ),
                    "prefill_topk_gpu_ms": _evt_pairs_ms(prof.prefill_topk_evt_pairs),
                    "prefill_preproc_gpu_ms": _evt_pairs_ms(
                        prof.prefill_preproc_evt_pairs
                    ),
                    "prefill_seq_full_gpu_ms": _evt_pairs_ms(
                        prof.prefill_seq_full_evt_pairs
                    ),
                    "prefill_pure_preproc_gpu_ms": _evt_pairs_ms(
                        prof.prefill_pure_preproc_evt_pairs
                    ),
                    "prefill_selector_bounds_gpu_ms": _evt_pairs_ms(
                        prof.prefill_selector_bounds_evt_pairs
                    ),
                    "prefill_selector_pipeline_gpu_ms": _evt_pairs_ms(
                        prof.prefill_selector_pipeline_evt_pairs
                    ),
                    "prefill_rebuild_gpu_ms": _evt_pairs_ms(prof.prefill_rebuild_evt_pairs),
                    **_prefill_group_profile_fields(
                        prof.prefill_evt0,
                        prof.prefill_group_evt_pairs,
                    ),
                    "prefill_publish_cpu_us": float(prof.prefill_publish_cpu_us),
                    "prefill_key_norms_delta_total_tokens": int(
                        prof.prefill_key_norms_delta_total_tokens
                    ),
                    "prefill_key_norms_delta_max_tokens": int(
                        prof.prefill_key_norms_delta_max_tokens
                    ),
                    "prefill_key_norms_delta_layers": int(
                        prof.prefill_key_norms_delta_layers
                    ),
                    "refresh_selector_gpu_ms": _evt_ms(prof.refresh_sel_evt0, prof.refresh_sel_evt1),
                    "refresh_rebuild_gpu_ms": _evt_ms(prof.refresh_rebuild_evt0, prof.refresh_rebuild_evt1),
                    # 细分 evt（selector 内部可选字段）在 sync 模式同样可读
                    "refresh_gather_gpu_ms": _evt_ms(prof.refresh_gather_evt0, prof.refresh_gather_evt1),
                    "refresh_key_norms_preproc_gpu_ms": _evt_ms(
                        prof.refresh_key_norms_preproc_evt0, prof.refresh_key_norms_preproc_evt1
                    ),
                    "refresh_key_norms_gpu_ms": _evt_ms(prof.refresh_key_norms_evt0, prof.refresh_key_norms_evt1),
                    "refresh_key_norms_h2d_gpu_ms": _evt_ms(
                        prof.refresh_key_norms_h2d_evt0, prof.refresh_key_norms_h2d_evt1
                    ),
                    "refresh_key_norms_delta_gpu_ms": _evt_ms(
                        prof.refresh_key_norms_delta_evt0, prof.refresh_key_norms_delta_evt1
                    ),
                    "refresh_key_norms_pack_gpu_ms": _evt_ms(
                        prof.refresh_key_norms_pack_evt0, prof.refresh_key_norms_pack_evt1
                    ),
                    "refresh_key_norms_pre_h2d_gap_gpu_ms": _evt_ms(
                        prof.refresh_key_norms_evt0, prof.refresh_key_norms_h2d_evt0
                    ),
                    "refresh_key_norms_h2d_to_delta_gap_gpu_ms": _evt_ms(
                        prof.refresh_key_norms_h2d_evt1,
                        prof.refresh_key_norms_delta_evt0,
                    ),
                    "refresh_key_norms_delta_to_pack_gap_gpu_ms": _evt_ms(
                        prof.refresh_key_norms_delta_evt1,
                        prof.refresh_key_norms_pack_evt0,
                    ),
                    "refresh_key_norms_post_pack_gap_gpu_ms": _evt_ms(
                        prof.refresh_key_norms_pack_evt1, prof.refresh_key_norms_evt1
                    ),
                    "refresh_key_norms_delta_total_tokens": int(
                        prof.refresh_key_norms_delta_total_tokens
                    ),
                    "refresh_key_norms_delta_max_tokens": int(
                        prof.refresh_key_norms_delta_max_tokens
                    ),
                    "refresh_key_norms_delta_layers": int(
                        prof.refresh_key_norms_delta_layers
                    ),
                    "refresh_log_s_gpu_ms": _evt_ms(prof.refresh_log_s_evt0, prof.refresh_log_s_evt1),
                    "refresh_log_s_triton_gpu_ms": _evt_ms(
                        prof.refresh_log_s_triton_evt0, prof.refresh_log_s_triton_evt1
                    ),
                    "refresh_log_s_mask_gpu_ms": _evt_ms(prof.refresh_log_s_mask_evt0, prof.refresh_log_s_mask_evt1),
                    "refresh_log_s_cross_gpu_ms": _evt_ms(prof.refresh_log_s_cross_evt0, prof.refresh_log_s_cross_evt1),
                    "refresh_topk_gpu_ms": _evt_ms(prof.refresh_topk_evt0, prof.refresh_topk_evt1),
                    "refresh_preproc_gpu_ms": _evt_ms(prof.refresh_preproc_evt0, prof.refresh_preproc_evt1),
                    "refresh_seq_full_gpu_ms": _evt_ms(prof.refresh_seq_full_evt0, prof.refresh_seq_full_evt1),
                    "refresh_pure_preproc_gpu_ms": _evt_ms(prof.refresh_pure_preproc_evt0, prof.refresh_pure_preproc_evt1),
                    "refresh_selector_bounds_gpu_ms": _evt_ms(
                        prof.refresh_selector_bounds_evt0,
                        prof.refresh_selector_bounds_evt1,
                    ),
                    "refresh_selector_pipeline_gpu_ms": _evt_ms(
                        prof.refresh_selector_pipeline_evt0,
                        prof.refresh_selector_pipeline_evt1,
                    ),
                    **{
                        f"async_producer_{stage}_gpu_ms": _evt_pairs_ms(
                            getattr(prof, f"async_producer_{stage}_evt_pairs")
                        )
                        for stage in ASYNC_PRODUCER_GPU_PROFILE_STAGES
                    },
                    "rebuild_head_dim": int(prof.rebuild_head_dim),
                    "rebuild_kv_dtype": str(prof.rebuild_kv_dtype),
                    "rebuild_block_size": int(prof.rebuild_block_size),
                    "rebuild_stride_tokens": int(prof.rebuild_stride_tokens),
                    "rebuild_selected_k": int(prof.rebuild_selected_k),
                    "rebuild_num_kv_heads": int(prof.rebuild_num_kv_heads),
                    "rebuild_batch_slots": int(prof.rebuild_batch_slots),
                    "capture_kv_len_total": int(prof.capture_kv_len_total),
                    "writer_pointer_rebuild_count": int(prof.writer_pointer_rebuild_count),
                    "writer_pointer_lookup_count": int(prof.writer_pointer_lookup_count),
                    "writer_cached_pointer_hit_rate": float(prof.writer_cached_pointer_hit_rate),
                    "writer_cached_pointer_op_count": int(prof.writer_cached_pointer_op_count),
                    "writer_vector_fallback_count": int(prof.writer_vector_fallback_count),
                    "writer_kernel_variant": str(prof.writer_kernel_variant),
                    "writer_actual_tokens": int(prof.writer_actual_tokens),
                    "writer_sink_tokens": int(prof.writer_sink_tokens),
                    "writer_persist_tokens": int(prof.writer_persist_tokens),
                    "writer_sink_io_bytes": int(prof.writer_sink_io_bytes),
                    "writer_persist_io_bytes": int(prof.writer_persist_io_bytes),
                    "writer_token_tiles_estimated": int(prof.writer_token_tiles_estimated),
                    "writer_active_token_tiles_estimated": int(prof.writer_active_token_tiles_estimated),
                    "writer_cta_count_estimated": int(prof.writer_cta_count_estimated),
                    "writer_active_cta_count_estimated": int(prof.writer_active_cta_count_estimated),
                    "writer_tokens_per_cta": int(prof.writer_tokens_per_cta),
                    "writer_k_read_bytes": int(prof.writer_k_read_bytes),
                    "writer_v_read_bytes": int(prof.writer_v_read_bytes),
                    "writer_k_write_bytes": int(prof.writer_k_write_bytes),
                    "writer_v_write_bytes": int(prof.writer_v_write_bytes),
                    "writer_pos_write_bytes": int(prof.writer_pos_write_bytes),
                    "writer_total_io_bytes": int(prof.writer_total_io_bytes),
                    "writer_effective_io_gbps": float(prof.writer_effective_io_gbps),
                    "selected_indices_materialized_bytes": int(
                        prof.selected_indices_materialized_bytes
                    ),
                    "selected_indices_io_bytes": int(prof.selected_indices_io_bytes),
                    "selector_writer_current_path_count": int(
                        prof.selector_writer_current_path_count
                    ),
                    "selector_writer_boundary_cpu_us": float(
                        prof.selector_writer_boundary_cpu_us
                    ),
                    "selected_boundary_lower_bound_ms_per_group": float(
                        prof.selected_boundary_lower_bound_ms_per_group
                    ),
                    "predicted_front_early_step_improvement_ms": float(
                        prof.predicted_front_early_step_improvement_ms
                    ),
                    "residual_fixed_capture_control_ms": float(
                        prof.residual_fixed_capture_control_ms
                    ),
                    "source_ready_recorded_after_pointer_publish_count": int(
                        prof.source_ready_recorded_after_pointer_publish_count
                    ),
                    "lastn1_direct_count": int(lastn1_direct_count),
                    "gt1_reduce_count": int(gt1_reduce_count),
                    "gt1_scalar_fallback_count": int(gt1_scalar_fallback_count),
                    "refresh_rebuild_budget_before": int(
                        prof.refresh_rebuild_budget_before
                    ),
                    "refresh_rebuild_budget_after": int(
                        prof.refresh_rebuild_budget_after
                    ),
                    "refresh_rebuild_enqueued_count": int(
                        prof.refresh_rebuild_enqueued_count
                    ),
                    "refresh_rebuild_inline_count": int(
                        prof.refresh_rebuild_inline_count
                    ),
                    "refresh_rebuild_pending_queue_size": int(
                        prof.refresh_rebuild_pending_queue_size
                    ),
                    "refresh_rebuild_coalesced_count": int(
                        prof.refresh_rebuild_coalesced_count
                    ),
                    "deadline_rebuild_drop_finished_count": int(
                        getattr(self, "_deadline_rebuild_drop_finished_count", 0)
                    ),
                    "deadline_rebuild_drain_finish_count": int(
                        getattr(self, "_deadline_rebuild_drain_finish_count", 0)
                    ),
                    "deadline_rebuild_partial_finish_count": int(
                        getattr(self, "_deadline_rebuild_partial_finish_count", 0)
                    ),
                    "deadline_rebuild_drain_submit_count": int(
                        getattr(self, "_deadline_rebuild_drain_submit_count", 0)
                    ),
                    "deadline_rebuild_drain_submit_decode_step_min": int(
                        getattr(
                            self,
                            "_deadline_rebuild_drain_submit_decode_step_min",
                            -1,
                        )
                    ),
                    "deadline_rebuild_drain_submit_decode_step_max": int(
                        getattr(
                            self,
                            "_deadline_rebuild_drain_submit_decode_step_max",
                            -1,
                        )
                    ),
                    "deadline_rebuild_drain_submit_decode_steps": list(
                        self._take_deadline_rebuild_drain_submit_decode_steps()
                    ),
                    "deadline_async_producer_body_count": int(
                        getattr(self, "_deadline_async_producer_body_count", 0)
                    ),
                    "deadline_async_producer_body_cpu_us_total": float(
                        getattr(
                            self,
                            "_deadline_async_producer_body_cpu_us_total",
                            0.0,
                        )
                    ),
                    "deadline_async_producer_body_cpu_us_max": float(
                        getattr(
                            self,
                            "_deadline_async_producer_body_cpu_us_max",
                            0.0,
                        )
                    ),
                    "deadline_async_producer_selector_count": int(
                        getattr(self, "_deadline_async_producer_selector_count", 0)
                    ),
                    "deadline_async_producer_selector_cpu_us_total": float(
                        getattr(
                            self,
                            "_deadline_async_producer_selector_cpu_us_total",
                            0.0,
                        )
                    ),
                    "deadline_async_producer_selector_cpu_us_max": float(
                        getattr(
                            self,
                            "_deadline_async_producer_selector_cpu_us_max",
                            0.0,
                        )
                    ),
                    "deadline_async_producer_key_norms_delta_count": int(
                        getattr(
                            self,
                            "_deadline_async_producer_key_norms_delta_count",
                            0,
                        )
                    ),
                    "deadline_async_producer_key_norms_delta_total_tokens_total": int(
                        getattr(
                            self,
                            "_deadline_async_producer_key_norms_delta_total_tokens_total",
                            0,
                        )
                    ),
                    "deadline_async_producer_key_norms_delta_max_tokens_max": int(
                        getattr(
                            self,
                            "_deadline_async_producer_key_norms_delta_max_tokens_max",
                            -1,
                        )
                    ),
                    "deadline_async_producer_key_norms_delta_layers_total": int(
                        getattr(
                            self,
                            "_deadline_async_producer_key_norms_delta_layers_total",
                            0,
                        )
                    ),
                    "deadline_async_producer_writer_count": int(
                        getattr(self, "_deadline_async_producer_writer_count", 0)
                    ),
                    "deadline_async_producer_writer_cpu_us_total": float(
                        getattr(
                            self,
                            "_deadline_async_producer_writer_cpu_us_total",
                            0.0,
                        )
                    ),
                    "deadline_async_producer_writer_cpu_us_max": float(
                        getattr(
                            self,
                            "_deadline_async_producer_writer_cpu_us_max",
                            0.0,
                        )
                    ),
                    "deadline_async_producer_graph_replay_count": int(
                        getattr(
                            self,
                            "_deadline_async_producer_graph_replay_count",
                            0,
                        )
                    ),
                    "deadline_async_producer_graph_replay_cpu_us_total": float(
                        getattr(
                            self,
                            "_deadline_async_producer_graph_replay_cpu_us_total",
                            0.0,
                        )
                    ),
                    "deadline_async_producer_graph_replay_cpu_us_max": float(
                        getattr(
                            self,
                            "_deadline_async_producer_graph_replay_cpu_us_max",
                            0.0,
                        )
                    ),
                    "deadline_async_producer_graph_replay_stage_selector_inputs_count": int(
                        getattr(
                            self,
                            "_deadline_async_producer_graph_replay_stage_selector_inputs_count",
                            0,
                        )
                    ),
                    "deadline_async_producer_graph_replay_stage_selector_inputs_cpu_us_total": float(
                        getattr(
                            self,
                            "_deadline_async_producer_graph_replay_stage_selector_inputs_cpu_us_total",
                            0.0,
                        )
                    ),
                    "deadline_async_producer_graph_replay_stage_selector_inputs_cpu_us_max": float(
                        getattr(
                            self,
                            "_deadline_async_producer_graph_replay_stage_selector_inputs_cpu_us_max",
                            0.0,
                        )
                    ),
                    "deadline_async_producer_graph_replay_prepare_writer_count": int(
                        getattr(
                            self,
                            "_deadline_async_producer_graph_replay_prepare_writer_count",
                            0,
                        )
                    ),
                    "deadline_async_producer_graph_replay_prepare_writer_cpu_us_total": float(
                        getattr(
                            self,
                            "_deadline_async_producer_graph_replay_prepare_writer_cpu_us_total",
                            0.0,
                        )
                    ),
                    "deadline_async_producer_graph_replay_prepare_writer_cpu_us_max": float(
                        getattr(
                            self,
                            "_deadline_async_producer_graph_replay_prepare_writer_cpu_us_max",
                            0.0,
                        )
                    ),
                    "deadline_async_producer_graph_replay_stage_lens_count": int(
                        getattr(
                            self,
                            "_deadline_async_producer_graph_replay_stage_lens_count",
                            0,
                        )
                    ),
                    "deadline_async_producer_graph_replay_stage_lens_cpu_us_total": float(
                        getattr(
                            self,
                            "_deadline_async_producer_graph_replay_stage_lens_cpu_us_total",
                            0.0,
                        )
                    ),
                    "deadline_async_producer_graph_replay_stage_lens_cpu_us_max": float(
                        getattr(
                            self,
                            "_deadline_async_producer_graph_replay_stage_lens_cpu_us_max",
                            0.0,
                        )
                    ),
                    "deadline_async_producer_graph_replay_prepare_events_count": int(
                        getattr(
                            self,
                            "_deadline_async_producer_graph_replay_prepare_events_count",
                            0,
                        )
                    ),
                    "deadline_async_producer_graph_replay_prepare_events_cpu_us_total": float(
                        getattr(
                            self,
                            "_deadline_async_producer_graph_replay_prepare_events_cpu_us_total",
                            0.0,
                        )
                    ),
                    "deadline_async_producer_graph_replay_prepare_events_cpu_us_max": float(
                        getattr(
                            self,
                            "_deadline_async_producer_graph_replay_prepare_events_cpu_us_max",
                            0.0,
                        )
                    ),
                    "deadline_async_producer_graph_replay_graph_count": int(
                        getattr(
                            self,
                            "_deadline_async_producer_graph_replay_graph_count",
                            0,
                        )
                    ),
                    "deadline_async_producer_graph_replay_graph_cpu_us_total": float(
                        getattr(
                            self,
                            "_deadline_async_producer_graph_replay_graph_cpu_us_total",
                            0.0,
                        )
                    ),
                    "deadline_async_producer_graph_replay_graph_cpu_us_max": float(
                        getattr(
                            self,
                            "_deadline_async_producer_graph_replay_graph_cpu_us_max",
                            0.0,
                        )
                    ),
                    "deadline_async_producer_graph_capture_count": int(
                        getattr(
                            self,
                            "_deadline_async_producer_graph_capture_count",
                            0,
                        )
                    ),
                    "deadline_async_producer_graph_capture_cpu_us_total": float(
                        getattr(
                            self,
                            "_deadline_async_producer_graph_capture_cpu_us_total",
                            0.0,
                        )
                    ),
                    "deadline_async_producer_graph_capture_cpu_us_max": float(
                        getattr(
                            self,
                            "_deadline_async_producer_graph_capture_cpu_us_max",
                            0.0,
                        )
                    ),
                    "deadline_async_producer_result_precomputed_count": int(
                        getattr(
                            self,
                            "_deadline_async_producer_result_precomputed_count",
                            0,
                        )
                    ),
                    "refresh_rebuild_delay_max": int(
                        getattr(self, "_refresh_rebuild_delay_max", 0)
                    ),
                    "producer_work_target_layer_start": int(
                        _producer_work_int("target_layer_start")
                    ),
                    "producer_work_target_layer_end": int(
                        _producer_work_int("target_layer_end")
                    ),
                    "producer_work_decode_step_min": int(
                        _producer_work_int("decode_step_min")
                    ),
                    "producer_work_decode_step_max": int(
                        _producer_work_int("decode_step_max")
                    ),
                    "producer_work_ready_epoch": int(_producer_work_int("ready_epoch")),
                    "producer_work_deadline_epoch": int(
                        _producer_work_int("deadline_epoch")
                    ),
                    "producer_work_deadline_handle_id": int(
                        _producer_work_int("deadline_handle_id")
                    ),
                    "producer_work_deadline_slack_steps": int(
                        _producer_work_int("deadline_slack_steps")
                    ),
                    "producer_work_can_drop": int(_producer_work_int("can_drop", 0)),
                    "producer_work_can_coalesce": int(
                        _producer_work_int("can_coalesce", 0)
                    ),
                    "producer_work_admission_reason": str(
                        _producer_work_str("admission_reason")
                    ),
                    "refresh_overlap_ratio": prof.refresh_overlap_ratio,
                    "refresh_overlap_new_k": prof.refresh_overlap_new_k,
                    "refresh_overlap_old_k": prof.refresh_overlap_old_k,
                    "note": "sync_profile_debug_only",
                }
                # [LITE-P0 J3 仪器 2026-07-11] SIG_RETURN 臂命中/降级计数快照
                # ——J1 红案取证发现 DecodeRuntimeCounters 全族均无遥测通道
                # ("计数恒 0"实为无此键假象)。record 为 dict,此处直塞与既有
                # 动态键同型;-1=runtime state 缺席哨兵。
                _lite_counters = getattr(
                    getattr(self, "_decode_runtime_state", None), "counters", None
                )
                record["lite_sig_return_step_count"] = int(
                    getattr(_lite_counters, "sig_return_step_count", -1)
                )
                record["lite_fallback_count"] = int(
                    getattr(_lite_counters, "lite_fallback_count", -1)
                )
                record["lite_last_fallback_reason"] = str(
                    getattr(self, "_lite_sig_return_last_fallback_reason", "")
                )
                try:
                    payload = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                except Exception:
                    _log.warning("sync profiling: json.dumps failed, falling back to str()", exc_info=True)
                    payload = str(record)
                self._refresh_profile_write(f"{os.getpid()}\trefresh.flush\t{payload}")
            if self._async_refresh_enabled():
                # do_async=False 的同步模式也需要更新 chunk_done_evt，避免 main_stream 的 wait_event
                # 误等待旧事件；但在 cudagraph capture 期间必须跳过 record_event。
                if not _is_stream_capturing_or_raise(stage="flush_prefill_batches_sync_record_event"):
                    self._ensure_refresh_stream(device)
                    if self.chunk_done_evt:
                        torch.cuda.current_stream(device=device).record_event(self.chunk_done_evt[buf])
                    # 同步模式下，也推进细粒度 done 事件，保持其单调更新
                    if self.prefill_done_evt:
                        torch.cuda.current_stream(device=device).record_event(self.prefill_done_evt[buf])
                    if self.refresh_done_evt:
                        torch.cuda.current_stream(device=device).record_event(self.refresh_done_evt[buf])
                    self._pending_work_clear_buf(buf_id=buf)

    if not is_last_layer:
        return

    # prefill/refresh 已在 chunk 边界完成（支持 refresh_stream 异步），这里不再做 last-layer finalize。

    # 清理本 step 的队列/状态（ring layout 保留以便复用，避免频繁分配）
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

    # refresh layer-group gating：只在本 step 确实发生 refresh 时推进 event_idx（避免 ctrl_step parity 饿死一半层）。
    if self._refresh_layer_group_enabled and self._refresh_layer_group_any_refresh:
        try:
            self._refresh_layer_group_event_idx += 1
        except Exception:
            _log.warning("refresh layer-group: event_idx increment failed, resetting to 0", exc_info=True)
            self._refresh_layer_group_event_idx = 0
            raise
    self._refresh_layer_group_any_refresh = False
