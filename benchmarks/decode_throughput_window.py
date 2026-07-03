from __future__ import annotations

import os
import sys
import time
import types
from dataclasses import dataclass, field
from typing import Any, Iterable, Tuple


@dataclass
class DecodeWindowMeter:
    """Measure decode throughput in the window [first_emit_ts, last_emit_ts].

    When *batch_size* is provided, an additional **all-decode window** is
    tracked: it starts from the first step where ``new_tokens >= batch_size``
    (i.e. all requests have entered decode) and ends at the last emit.  This
    avoids counting the slow prefill-interleaved steps that occur when
    different-length prompts finish chunked prefill at different times.
    """

    batch_size: int = 0
    first_emit_ts: float | None = None
    last_emit_ts: float | None = None
    total_tokens: int = 0
    first_emit_tokens: int = 0
    _prev_emit_ts: float | None = None
    decode_step_durations_s: list[float] = field(default_factory=list)
    # all-decode window (from first full-batch decode step)
    _ad_start_ts: float | None = None
    _ad_start_tokens: int = 0
    _ad_steps: int = 0

    def observe(self, ts: float, new_tokens: int) -> None:
        tokens = int(new_tokens)
        if tokens <= 0:
            return
        timestamp = float(ts)
        self.total_tokens += tokens
        if self.first_emit_ts is None:
            self.first_emit_ts = timestamp
            self.last_emit_ts = timestamp
            self._prev_emit_ts = timestamp
            self.first_emit_tokens = tokens
        else:
            prev_ts = self._prev_emit_ts if self._prev_emit_ts is not None else timestamp
            dt = timestamp - prev_ts
            if dt < 0.0:
                dt = 0.0
            self.decode_step_durations_s.append(float(dt))
            self._prev_emit_ts = timestamp
            self.last_emit_ts = timestamp

        # all-decode window: starts when tok/step reaches batch_size
        if self.batch_size > 0 and self._ad_start_ts is None and tokens >= self.batch_size:
            self._ad_start_ts = timestamp
            self._ad_start_tokens = self.total_tokens
        if self._ad_start_ts is not None:
            self._ad_steps += 1

    def finalize(self) -> Tuple[float, int, float, list[float]]:
        window_tokens = max(0, int(self.total_tokens) - int(self.first_emit_tokens))
        if self.first_emit_ts is None or self.last_emit_ts is None:
            return 0.0, int(window_tokens), float("nan"), list(self.decode_step_durations_s)

        decode_elapsed_s = float(self.last_emit_ts - self.first_emit_ts)
        if decode_elapsed_s <= 0.0:
            return 0.0, int(window_tokens), float("nan"), list(self.decode_step_durations_s)

        decode_tps = float(window_tokens) / float(decode_elapsed_s)
        return float(decode_elapsed_s), int(window_tokens), float(decode_tps), list(
            self.decode_step_durations_s
        )

    def boundary_delays(self, total_start_s: float, total_end_s: float) -> Tuple[float, float]:
        if self.first_emit_ts is None or self.last_emit_ts is None:
            return float("nan"), float("nan")
        first_emit_delay_s = max(0.0, float(self.first_emit_ts) - float(total_start_s))
        post_decode_tail_s = max(0.0, float(total_end_s) - float(self.last_emit_ts))
        return first_emit_delay_s, post_decode_tail_s

    def finalize_all_decode(self) -> Tuple[float, int, float, int]:
        """Return (elapsed_s, tokens, tok_per_s, steps) for the all-decode window.

        Falls back to the full window if batch_size was not set or all-decode
        was never entered.
        """
        if self._ad_start_ts is not None and self.last_emit_ts is not None:
            ad_elapsed = float(self.last_emit_ts - self._ad_start_ts)
            ad_tokens = max(0, int(self.total_tokens) - int(self._ad_start_tokens))
            ad_tps = float(ad_tokens) / float(ad_elapsed) if ad_elapsed > 0 else float("nan")
            return ad_elapsed, ad_tokens, ad_tps, int(self._ad_steps)
        # fallback: use the full first-emit window
        elapsed, tokens, tps, _ = self.finalize()
        return elapsed, tokens, tps, len(self.decode_step_durations_s)


def count_new_tokens(request_outputs: Iterable[Any], prev_len: dict[str, int]) -> int:
    step_new_tokens = 0
    for item in request_outputs:
        rid = getattr(item, "request_id", None)
        if rid is None:
            continue
        outputs = getattr(item, "outputs", None)
        if not outputs:
            continue
        token_ids = getattr(outputs[0], "token_ids", None)
        if token_ids is None:
            continue
        curr = int(len(token_ids))
        prev = int(prev_len.get(rid, 0))
        if curr > prev:
            step_new_tokens += curr - prev
            prev_len[rid] = curr
    return int(step_new_tokens)


_INNER_TIMING_FALSE_VALUES = {"", "0", "false", "off", "no"}


def _engine_core_inner_timing_enabled() -> bool:
    value = str(os.environ.get("VLLM_DECODE_ENGINE_CORE_INNER_TIMING", "") or "")
    return value.strip().lower() not in _INNER_TIMING_FALSE_VALUES


def _elapsed_us(start_s: float) -> float:
    return float((time.perf_counter() - start_s) * 1_000_000.0)


def _record_engine_core_timing(engine_core: Any, timing: dict[str, float]) -> None:
    try:
        setattr(engine_core, "_decode_inner_timing_last_us", dict(timing))
    except Exception:
        return


def _maybe_install_model_runner_segment_timing(engine_core: Any) -> dict[str, float]:
    """[MR-SEGMENT-TIMING] in-proc 下再往下包一层 model_runner 关键段。

    ec_execute_model_submit 是黑盒总量（in-proc=同步整个 execute_model）；这里
    对 runner 的 _prepare_inputs / execute_model / sample_tokens 各包一层
    perf_counter，写进共享 dict，由 timed_step 并入 inner（mr_* 键）随
    step_engine_core_timing_all 落盘。诊断专用（INNER_TIMING 门控内），
    monkey-patch 只装一次，失败静默降级（返回空 dict 不影响原计时）。
    """
    seg: dict[str, float] = {}
    try:
        wrapper = engine_core.model_executor.driver_worker
        # UniProcExecutor.driver_worker 是 WorkerWrapperBase，真 Worker 在 .worker
        worker = getattr(wrapper, "worker", None) or wrapper
        runner = worker.model_runner
    except Exception as exc:
        print(f"[mr-seg-timing] install skipped: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return seg
    print(f"[mr-seg-timing] installed on {type(runner).__name__}", file=sys.stderr, flush=True)
    if bool(getattr(runner, "_mr_segment_timing_installed", False)):
        return getattr(runner, "_mr_segment_timing_shared", seg)

    def _wrap(name: str) -> bool:
        orig = getattr(runner, name, None)
        if not callable(orig):
            return False

        def timed(*args: Any, _orig: Any = orig, _key: str = f"mr_{name}_us", **kwargs: Any):
            t0 = time.perf_counter()
            try:
                return _orig(*args, **kwargs)
            finally:
                seg[_key] = seg.get(_key, 0.0) + _elapsed_us(t0)

        setattr(runner, name, timed)
        return True

    for method in ("_prepare_inputs", "sample_tokens"):
        _wrap(method)

    # [MR-GPU-CLOCK] execute_model 额外做 CUDA event 对钟，回答 host/GPU 重叠：
    # mr_gpu_span_us = 主流上本步首尾 event 跨度（含流内空隙）；
    # mr_gpu_done_at_return = host 返回时刻 GPU 是否已完成（1/0）。
    # 判读：span≈host 且 done=0 ⇒ GPU 主导（泡小）；span<<host 且 done=1 ⇒
    # host 尾部纯 CPU 段=可重叠泡。event 对轮转成对（上一步的 elapsed 在
    # 下一步开头收割，完成才读，零阻塞）。诊断门（INNER_TIMING）内。
    def _wrap_execute_with_gpu_clock() -> bool:
        orig = getattr(runner, "execute_model", None)
        if not callable(orig):
            return False
        try:
            import torch as _torch
        except Exception:
            return False
        if not _torch.cuda.is_available():
            return _wrap("execute_model")
        ring = [
            (_torch.cuda.Event(enable_timing=True), _torch.cuda.Event(enable_timing=True))
            for _ in range(2)
        ]
        slot_state = {"next": 0, "pending": None}

        def timed(*args: Any, **kwargs: Any):
            pend = slot_state["pending"]
            if pend is not None:
                p_start, p_end = pend
                if p_end.query():
                    try:
                        seg["mr_gpu_span_prev_us"] = float(
                            p_start.elapsed_time(p_end) * 1000.0
                        )
                    except Exception:
                        pass
                    slot_state["pending"] = None
            s_evt, e_evt = ring[slot_state["next"] % 2]
            slot_state["next"] += 1
            t0 = time.perf_counter()
            s_evt.record()
            try:
                return orig(*args, **kwargs)
            finally:
                e_evt.record()
                seg["mr_execute_model_us"] = (
                    seg.get("mr_execute_model_us", 0.0) + _elapsed_us(t0)
                )
                seg["mr_gpu_done_at_return"] = 1.0 if e_evt.query() else 0.0
                slot_state["pending"] = (s_evt, e_evt)

        setattr(runner, "execute_model", timed)
        return True

    _wrap_execute_with_gpu_clock()
    runner._mr_segment_timing_installed = True
    runner._mr_segment_timing_shared = seg
    return seg


def _maybe_install_engine_core_inner_timing(llm_engine: Any) -> None:
    if not _engine_core_inner_timing_enabled():
        return
    client = getattr(llm_engine, "engine_core", None)
    engine_core = getattr(client, "engine_core", None)
    if engine_core is None or bool(
        getattr(engine_core, "_decode_inner_timing_installed", False)
    ):
        return
    mr_seg = _maybe_install_model_runner_segment_timing(engine_core)

    def timed_step(self: Any):
        inner: dict[str, float] = {}
        total0 = time.perf_counter()
        try:
            has0 = time.perf_counter()
            has_requests = bool(self.scheduler.has_requests())
            inner["ec_has_requests_us"] = _elapsed_us(has0)
            if not has_requests:
                inner["ec_total_us"] = _elapsed_us(total0)
                _record_engine_core_timing(self, inner)
                return {}, False

            t0 = time.perf_counter()
            scheduler_output = self.scheduler.schedule()
            inner["ec_schedule_us"] = _elapsed_us(t0)

            t0 = time.perf_counter()
            future = self.model_executor.execute_model(
                scheduler_output, non_block=True
            )
            inner["ec_execute_model_submit_us"] = _elapsed_us(t0)

            t0 = time.perf_counter()
            grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
            inner["ec_grammar_bitmask_us"] = _elapsed_us(t0)

            with (
                self.log_error_detail(scheduler_output),
                self.log_iteration_details(scheduler_output),
            ):
                t0 = time.perf_counter()
                model_output = future.result()
                inner["ec_future_result_us"] = _elapsed_us(t0)
                if model_output is None:
                    t0 = time.perf_counter()
                    model_output = self.model_executor.sample_tokens(grammar_output)
                    inner["ec_sample_tokens_us"] = _elapsed_us(t0)
                else:
                    inner["ec_sample_tokens_us"] = 0.0

            t0 = time.perf_counter()
            self._process_aborts_queue()
            inner["ec_process_aborts_queue_us"] = _elapsed_us(t0)

            t0 = time.perf_counter()
            engine_core_outputs = self.scheduler.update_from_output(
                scheduler_output, model_output
            )
            inner["ec_update_from_output_us"] = _elapsed_us(t0)
            if mr_seg:
                inner.update(mr_seg)
                mr_seg.clear()
            inner["ec_total_us"] = _elapsed_us(total0)
            _record_engine_core_timing(self, inner)
            return (
                engine_core_outputs,
                scheduler_output.total_num_scheduled_tokens > 0,
            )
        except Exception:
            inner["ec_total_us"] = _elapsed_us(total0)
            _record_engine_core_timing(self, inner)
            raise

    def timed_step_with_batch_queue(self: Any):
        inner: dict[str, float] = {}
        total0 = time.perf_counter()
        try:
            batch_queue = self.batch_queue
            assert batch_queue is not None
            assert len(batch_queue) < self.batch_queue_size

            model_executed = False
            deferred_scheduler_output = None
            t0 = time.perf_counter()
            has_requests = bool(self.scheduler.has_requests())
            inner["ec_has_requests_us"] = _elapsed_us(t0)
            if has_requests:
                t0 = time.perf_counter()
                scheduler_output = self.scheduler.schedule()
                inner["ec_schedule_us"] = _elapsed_us(t0)
                with self.log_error_detail(scheduler_output):
                    t0 = time.perf_counter()
                    exec_future = self.model_executor.execute_model(
                        scheduler_output, non_block=True
                    )
                    inner["ec_execute_model_submit_us"] = _elapsed_us(t0)
                if self.is_ec_consumer:
                    model_executed = scheduler_output.total_num_scheduled_tokens > 0

                if self.is_pooling_model or not model_executed:
                    future = exec_future
                    inner["ec_grammar_bitmask_us"] = 0.0
                    inner["ec_sample_tokens_submit_us"] = 0.0
                else:
                    if not scheduler_output.pending_structured_output_tokens:
                        t0 = time.perf_counter()
                        grammar_output = self.scheduler.get_grammar_bitmask(
                            scheduler_output
                        )
                        inner["ec_grammar_bitmask_us"] = _elapsed_us(t0)
                        t0 = time.perf_counter()
                        future = self.model_executor.sample_tokens(
                            grammar_output, non_block=True
                        )
                        inner["ec_sample_tokens_submit_us"] = _elapsed_us(t0)
                    else:
                        deferred_scheduler_output = scheduler_output

                if not deferred_scheduler_output:
                    batch_queue.appendleft((future, scheduler_output, exec_future))
                    if (
                        model_executed
                        and len(batch_queue) < self.batch_queue_size
                        and not batch_queue[-1][0].done()
                    ):
                        inner["ec_returned_none_us"] = _elapsed_us(total0)
                        inner["ec_total_us"] = _elapsed_us(total0)
                        _record_engine_core_timing(self, inner)
                        return None, True

            elif not batch_queue:
                inner["ec_total_us"] = _elapsed_us(total0)
                _record_engine_core_timing(self, inner)
                return None, False

            t0 = time.perf_counter()
            future, scheduler_output, exec_model_fut = batch_queue.pop()
            inner["ec_batch_queue_pop_us"] = _elapsed_us(t0)
            with (
                self.log_error_detail(scheduler_output),
                self.log_iteration_details(scheduler_output),
            ):
                t0 = time.perf_counter()
                model_output = future.result()
                inner["ec_future_result_us"] = _elapsed_us(t0)
                if model_output is None:
                    t0 = time.perf_counter()
                    exec_model_fut.result()
                    inner["ec_exec_model_future_result_us"] = _elapsed_us(t0)
                    raise RuntimeError("unexpected error")

            t0 = time.perf_counter()
            self._process_aborts_queue()
            inner["ec_process_aborts_queue_us"] = _elapsed_us(t0)

            t0 = time.perf_counter()
            engine_core_outputs = self.scheduler.update_from_output(
                scheduler_output, model_output
            )
            inner["ec_update_from_output_us"] = _elapsed_us(t0)
            if mr_seg:
                inner.update(mr_seg)
                mr_seg.clear()

            if deferred_scheduler_output:
                if self.use_spec_decode:
                    t0 = time.perf_counter()
                    draft_token_ids = self.model_executor.take_draft_token_ids()
                    inner["ec_take_draft_token_ids_us"] = _elapsed_us(t0)
                    assert draft_token_ids is not None
                    t0 = time.perf_counter()
                    self.scheduler.update_draft_token_ids_in_output(
                        draft_token_ids, deferred_scheduler_output
                    )
                    inner["ec_update_draft_token_ids_us"] = _elapsed_us(t0)
                t0 = time.perf_counter()
                grammar_output = self.scheduler.get_grammar_bitmask(
                    deferred_scheduler_output
                )
                inner["ec_deferred_grammar_bitmask_us"] = _elapsed_us(t0)
                t0 = time.perf_counter()
                future = self.model_executor.sample_tokens(grammar_output, non_block=True)
                inner["ec_deferred_sample_tokens_submit_us"] = _elapsed_us(t0)
                batch_queue.appendleft(
                    (future, deferred_scheduler_output, exec_future)
                )

            inner["ec_total_us"] = _elapsed_us(total0)
            _record_engine_core_timing(self, inner)
            return engine_core_outputs, model_executed
        except Exception:
            inner["ec_total_us"] = _elapsed_us(total0)
            _record_engine_core_timing(self, inner)
            raise

    original_post_step = engine_core.post_step

    def timed_post_step(self: Any, model_executed: bool) -> None:
        t0 = time.perf_counter()
        try:
            return original_post_step(model_executed)
        finally:
            inner = dict(getattr(self, "_decode_inner_timing_last_us", {}) or {})
            inner["ec_post_step_us"] = _elapsed_us(t0)
            inner["ec_total_plus_post_us"] = float(
                inner.get("ec_total_us", 0.0) + inner["ec_post_step_us"]
            )
            _record_engine_core_timing(self, inner)

    engine_core.step = types.MethodType(timed_step, engine_core)
    engine_core.step_with_batch_queue = types.MethodType(
        timed_step_with_batch_queue, engine_core
    )
    engine_core.post_step = types.MethodType(timed_post_step, engine_core)
    engine_core.step_fn = (
        engine_core.step
        if getattr(engine_core, "batch_queue", None) is None
        else engine_core.step_with_batch_queue
    )
    setattr(engine_core, "_decode_inner_timing_installed", True)


def pull_step_outputs_with_timing(llm_engine) -> tuple[list[Any], float, dict[str, float]]:
    timing = {
        "dummy_batch_us": 0.0,
        "get_output_us": 0.0,
        "process_outputs_us": 0.0,
        "abort_requests_us": 0.0,
    }
    # Keep parity with LLMEngine.step(): execute dummy batch once if requested.
    if bool(getattr(llm_engine, "should_execute_dummy_batch", False)):
        llm_engine.should_execute_dummy_batch = False
        t0 = time.perf_counter()
        llm_engine.engine_core.execute_dummy_batch()
        t1 = time.perf_counter()
        timing["dummy_batch_us"] = float((t1 - t0) * 1_000_000.0)
        return [], float("nan"), timing

    _maybe_install_engine_core_inner_timing(llm_engine)
    t0 = time.perf_counter()
    outputs = llm_engine.engine_core.get_output()
    t1 = time.perf_counter()
    engine_core = getattr(llm_engine.engine_core, "engine_core", None)
    inner_timing = getattr(engine_core, "_decode_inner_timing_last_us", None)
    if isinstance(inner_timing, dict):
        for key, value in inner_timing.items():
            if key.startswith("ec_") or key.startswith("mr_"):
                timing[str(key)] = float(value)
    processed_outputs = llm_engine.output_processor.process_outputs(
        outputs.outputs,
        engine_core_timestamp=float(outputs.timestamp),
        iteration_stats=None,
    )
    t2 = time.perf_counter()
    llm_engine.engine_core.abort_requests(processed_outputs.reqs_to_abort)
    t3 = time.perf_counter()
    timing["get_output_us"] = float((t1 - t0) * 1_000_000.0)
    timing["process_outputs_us"] = float((t2 - t1) * 1_000_000.0)
    timing["abort_requests_us"] = float((t3 - t2) * 1_000_000.0)
    return list(processed_outputs.request_outputs), float(outputs.timestamp), timing


def pull_step_outputs_with_timestamp(llm_engine) -> tuple[list[Any], float]:
    request_outputs, timestamp, _timing = pull_step_outputs_with_timing(llm_engine)
    return request_outputs, timestamp
