"""
Python automatically imports ``sitecustomize`` (if available) during start-up.

Worker processes spawned by vLLM therefore run this module before importing
vLLM internals, which lets us re-apply the sparse attention patch simply by
setting ``VLLM_SPARSE_CONTROLLER_JSON`` in the environment.
"""

import os
import sys

# 确保仓库根目录在 sys.path，避免 worker 进程找不到 patches 模块。
_REPO_ROOT = os.path.dirname(__file__)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ---------------------------------------------------------------------------
# Compatibility shim: newer `transformers` versions may already register some
# model types that older vLLM versions try to register again (e.g. "aimv2"),
# which raises at import time and prevents running benchmarks.
# Keep behavior deterministic by ignoring duplicate registrations.
# ---------------------------------------------------------------------------
if os.environ.get("VLLM_IGNORE_DUPLICATE_TRANSFORMERS_CONFIGS", "1") == "1":
    try:
        from transformers import AutoConfig  # type: ignore

        _orig_register = AutoConfig.register

        def _safe_register(model_type, config, exist_ok: bool = False):  # type: ignore[no-redef]
            try:
                return _orig_register(model_type, config, exist_ok=True)
            except ValueError:
                # Duplicate key in CONFIG_MAPPING: keep the existing mapping.
                return None

        AutoConfig.register = _safe_register  # type: ignore[assignment]
    except Exception:
        pass

# [WEDGE-FORENSICS 2026-07-08] env 门控诊断钩(铁律允许的诊断期取证工具,
# 默认关):设 VLLM_SPARSE_FAULTHANDLER_DIR=<dir> 后,每个 python 进程(含
# vLLM worker)注册 SIGUSR1→faulthandler 全线程栈 dump 到 <dir>/<pid>.stack
# (追加式,可多次采样),不杀进程——用于 ptrace 被禁机器上的活体楔死取证。
# [WEDGE-FORENSICS] 姊妹钩:VLLM_SPARSE_STDERR_TEE_DIR=<dir> → 把本进程
# fd2(stderr,含 C 层/NCCL/device printf/__trap 指纹)dup 到 <dir>/<pid>.err
# 落盘——bench 管道在楔死强杀时会吞掉 child stderr,此钩保证原发异常必留痕。
_tee_dir = os.environ.get("VLLM_SPARSE_STDERR_TEE_DIR", "")
if _tee_dir:
    os.makedirs(_tee_dir, exist_ok=True)
    _tee_fd = os.open(
        os.path.join(_tee_dir, f"{os.getpid()}.err"),
        os.O_WRONLY | os.O_CREAT | os.O_APPEND,
        0o644,
    )
    # 只重定向 fd2:异常 traceback 与 NCCL/C 层错误走 stderr。fd1(stdout)
    # 承载 EngineCore↔client 的 route_counters/summary IPC,不能动(dup 走会
    # 令 gate 恒报 continuous_refresh_reqs_missing)。
    # [2026-07-08 修] 此前只 open 未 dup2,钩子从未生效(.err 恒 0 字节,
    # 由此得出的"楔死态无异常"结论作废)。dup2 后 fd2 全量落盘。
    os.dup2(_tee_fd, 2)

_fh_dir = os.environ.get("VLLM_SPARSE_FAULTHANDLER_DIR", "")
if _fh_dir:
    import faulthandler
    import signal

    os.makedirs(_fh_dir, exist_ok=True)
    _fh_file = open(
        os.path.join(_fh_dir, f"{os.getpid()}.stack"), "a", buffering=1
    )
    _fh_file.write(f"=== pid={os.getpid()} argv={sys.argv[:3]} ===\n")
    faulthandler.enable(file=_fh_file, all_threads=True)
    if hasattr(signal, "SIGUSR1"):
        faulthandler.register(
            signal.SIGUSR1, file=_fh_file, all_threads=True, chain=False
        )

# [WEDGE-FORENSICS 2026-07-08] 三号钩:VLLM_SPARSE_STEP_LEDGER_DIR=<dir> →
# 每个 vLLM worker 进程写 <dir>/<pid>.ledger 逐步账本(execute_model BEGIN/END/EXC
# + eager all-reduce 计数器 + cudagraph replay 计数器 + busy_loop 退出/shutdown 行,
# 行缓冲落盘)。用途:TP 楔死取证——diff 两 rank 账本找首个分叉步;账本 mtime 兼作
# 活性信号(区分"慢"与"楔死")。同时给 vllm logger 挂 <dir>/<pid>.vllmlog 文件
# handler:vLLM 默认日志走 stdout(被 bench IPC 管道吞),worker_busy_loop 的
# "WorkerProc hit an exception."(非 output rank 异常被静默 continue,是 TP 集合
# 序列 desync 的直接机器)必须留痕。诊断钩默认关,不改任何行为。
_ledger_dir = os.environ.get("VLLM_SPARSE_STEP_LEDGER_DIR", "")
if _ledger_dir:
    try:
        import json as _sfi_lg_json
        import logging as _sfi_lg_logging
        import time as _sfi_lg_time
        import traceback as _sfi_lg_traceback

        os.makedirs(_ledger_dir, exist_ok=True)
        _sfi_lg_file = open(
            os.path.join(_ledger_dir, f"{os.getpid()}.ledger"), "a", buffering=1
        )
        _sfi_lg_counters = {"ar_total": 0, "ar_nccl": 0, "cgw_replay": 0, "cgw_other": 0}
        _sfi_lg_seq = {"n": 0}

        def _sfi_lg_write(payload):
            payload["ts"] = _sfi_lg_time.time()
            payload["pid"] = os.getpid()
            _sfi_lg_file.write(
                _sfi_lg_json.dumps(payload, sort_keys=True, default=str) + "\n"
            )

        _sfi_lg_write({"event": "ledger_open", "argv": sys.argv[:3]})

        # vllm logger → 文件 handler(逮 stdout 日志里的吞异常)
        _sfi_lg_handler = _sfi_lg_logging.FileHandler(
            os.path.join(_ledger_dir, f"{os.getpid()}.vllmlog")
        )
        _sfi_lg_handler.setLevel(_sfi_lg_logging.INFO)
        _sfi_lg_handler.setFormatter(
            _sfi_lg_logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        _sfi_lg_logging.getLogger("vllm").addHandler(_sfi_lg_handler)

        from vllm.compilation.cuda_graph import CUDAGraphWrapper as _sfi_lg_CGW
        from vllm.config import CUDAGraphMode as _sfi_lg_CGMode
        from vllm.distributed.device_communicators.cuda_communicator import (
            CudaCommunicator as _sfi_lg_CudaComm,
        )
        from vllm.distributed.device_communicators.pynccl import (
            PyNcclCommunicator as _sfi_lg_PyNccl,
        )
        from vllm.forward_context import (
            get_forward_context as _sfi_lg_get_fc,
            is_forward_context_available as _sfi_lg_fc_avail,
        )
        from vllm.v1.executor import multiproc_executor as _sfi_lg_mpe
        from vllm.v1.worker import worker_base as _sfi_lg_wb

        if not getattr(_sfi_lg_CudaComm.all_reduce, "_sfi_lg_patch", False):
            _sfi_lg_orig_cc_ar = _sfi_lg_CudaComm.all_reduce

            def _sfi_lg_cc_ar(self, input_):
                _sfi_lg_counters["ar_total"] += 1
                return _sfi_lg_orig_cc_ar(self, input_)

            _sfi_lg_cc_ar._sfi_lg_patch = True
            _sfi_lg_CudaComm.all_reduce = _sfi_lg_cc_ar

        if not getattr(_sfi_lg_PyNccl.all_reduce, "_sfi_lg_patch", False):
            _sfi_lg_orig_nccl_ar = _sfi_lg_PyNccl.all_reduce

            def _sfi_lg_nccl_ar(self, in_tensor, *args, **kwargs):
                _sfi_lg_counters["ar_nccl"] += 1
                return _sfi_lg_orig_nccl_ar(self, in_tensor, *args, **kwargs)

            _sfi_lg_nccl_ar._sfi_lg_patch = True
            _sfi_lg_PyNccl.all_reduce = _sfi_lg_nccl_ar

        if not getattr(_sfi_lg_CGW.__call__, "_sfi_lg_patch", False):
            _sfi_lg_orig_cgw_call = _sfi_lg_CGW.__call__

            def _sfi_lg_cgw_call(self, *args, **kwargs):
                is_replay = False
                try:
                    if _sfi_lg_fc_avail():
                        fc = _sfi_lg_get_fc()
                        if (
                            getattr(fc, "cudagraph_runtime_mode", None)
                            == _sfi_lg_CGMode.FULL
                            and getattr(self, "runtime_mode", None)
                            == _sfi_lg_CGMode.FULL
                        ):
                            entries = getattr(self, "concrete_cudagraph_entries", {})
                            bd = getattr(fc, "batch_descriptor", None)
                            entry = (
                                entries.get(bd)
                                if bd is not None and hasattr(entries, "get")
                                else None
                            )
                            is_replay = bool(
                                entry is not None
                                and getattr(entry, "cudagraph", None) is not None
                            )
                except Exception:
                    pass
                _sfi_lg_counters["cgw_replay" if is_replay else "cgw_other"] += 1
                return _sfi_lg_orig_cgw_call(self, *args, **kwargs)

            _sfi_lg_cgw_call._sfi_lg_patch = True
            _sfi_lg_CGW.__call__ = _sfi_lg_cgw_call

        if not getattr(_sfi_lg_wb.WorkerWrapperBase.execute_model, "_sfi_lg_patch", False):
            _sfi_lg_orig_wb_exec = _sfi_lg_wb.WorkerWrapperBase.execute_model

            def _sfi_lg_sched_summary(sched):
                out = {}
                try:
                    v = getattr(sched, "total_num_scheduled_tokens", None)
                    if v is not None:
                        out["tokens"] = int(v)
                    nst = getattr(sched, "num_scheduled_tokens", None)
                    if isinstance(nst, dict):
                        out["reqs"] = len(nst)
                    fin = getattr(sched, "finished_req_ids", None)
                    if fin is not None:
                        out["finished"] = len(fin)
                except Exception:
                    pass
                return out

            def _sfi_lg_wb_exec(self, scheduler_output, *args, **kwargs):
                seq = _sfi_lg_seq["n"]
                _sfi_lg_seq["n"] += 1
                rec = {
                    "event": "exec_begin",
                    "seq": seq,
                    "rank": int(getattr(self, "rpc_rank", -1)),
                }
                rec.update(_sfi_lg_sched_summary(scheduler_output))
                rec.update(_sfi_lg_counters)
                _sfi_lg_write(rec)
                t0 = _sfi_lg_time.perf_counter_ns()
                try:
                    result = _sfi_lg_orig_wb_exec(self, scheduler_output, *args, **kwargs)
                except BaseException as exc:
                    rec = {
                        "event": "exec_exc",
                        "seq": seq,
                        "exc_type": type(exc).__name__,
                        "exc_msg": str(exc)[:2000],
                        "tb_tail": _sfi_lg_traceback.format_exc()[-4000:],
                        "dur_us": (_sfi_lg_time.perf_counter_ns() - t0) / 1000.0,
                    }
                    rec.update(_sfi_lg_counters)
                    _sfi_lg_write(rec)
                    raise
                rec = {
                    "event": "exec_end",
                    "seq": seq,
                    "dur_us": (_sfi_lg_time.perf_counter_ns() - t0) / 1000.0,
                    "output_type": type(result).__name__,
                }
                rec.update(_sfi_lg_counters)
                _sfi_lg_write(rec)
                return result

            _sfi_lg_wb_exec._sfi_lg_patch = True
            _sfi_lg_wb.WorkerWrapperBase.execute_model = _sfi_lg_wb_exec

        if not getattr(_sfi_lg_mpe.WorkerProc.worker_busy_loop, "_sfi_lg_patch", False):
            _sfi_lg_orig_busy = _sfi_lg_mpe.WorkerProc.worker_busy_loop

            def _sfi_lg_busy(self):
                _sfi_lg_write(
                    {"event": "busy_loop_enter", "rank": int(getattr(self, "rank", -1))}
                )
                try:
                    return _sfi_lg_orig_busy(self)
                except BaseException as exc:
                    _sfi_lg_write(
                        {
                            "event": "busy_loop_exit",
                            "rank": int(getattr(self, "rank", -1)),
                            "exc_type": type(exc).__name__,
                            "exc_msg": str(exc)[:500],
                        }
                    )
                    raise
                finally:
                    _sfi_lg_file.flush()

            _sfi_lg_busy._sfi_lg_patch = True
            _sfi_lg_mpe.WorkerProc.worker_busy_loop = _sfi_lg_busy

        if not getattr(_sfi_lg_mpe.WorkerProc.shutdown, "_sfi_lg_patch", False):
            _sfi_lg_orig_wp_shutdown = _sfi_lg_mpe.WorkerProc.shutdown

            def _sfi_lg_wp_shutdown(self):
                rec = {
                    "event": "worker_shutdown",
                    "rank": int(getattr(self, "rank", -1)),
                }
                rec.update(_sfi_lg_counters)
                _sfi_lg_write(rec)
                return _sfi_lg_orig_wp_shutdown(self)

            _sfi_lg_wp_shutdown._sfi_lg_patch = True
            _sfi_lg_mpe.WorkerProc.shutdown = _sfi_lg_wp_shutdown
    except Exception as exc:
        raise SystemExit(
            f"sitecustomize step-ledger diagnostic failed: {exc}"
        ) from exc

site_log_enabled = os.environ.get("VLLM_SPARSE_SITE_LOG", "0") == "1"
fa4_dense_gateway_requested = (
    os.environ.get("VLLM_SPARSE_FA4_DENSE_GATEWAY") == "1"
    and os.environ.get("VLLM_FLASH_ATTN_VERSION") == "4"
    and "VLLM_SPARSE_CONTROLLER_JSON" not in os.environ
)

vendored_probe_requested = (
    os.environ.get("VLLM_ATTENTION_BACKEND") == "FLASH_ATTN_VLLM_V1"
    and os.environ.get("VLLM_FLASH_ATTN_VERSION") in ("3", "4")
)

try:
    from patches.fa3_native.install import (
        install_dense_fa3_route_trace_probe_patch,
        install_vendored_flash_attn_probe_patch,
        should_install_vendored_flash_attn_probe_patch,
    )

    vendored_probe_requested = should_install_vendored_flash_attn_probe_patch()
    vendored_probe_summary = install_vendored_flash_attn_probe_patch(
        repo_root=_REPO_ROOT,
    )
    dense_route_probe_requested = (
        vendored_probe_requested
        and bool(os.environ.get("VLLM_SPARSE_FA3_ROUTE_TRACE_LOG"))
        and "VLLM_SPARSE_CONTROLLER_JSON" not in os.environ
    )
    dense_route_probe_summary = install_dense_fa3_route_trace_probe_patch()
    if site_log_enabled and vendored_probe_summary.get("applied"):
        with open("/tmp/vllm_sparse_site.log", "a", encoding="utf-8") as _log:
            _log.write(
                "sitecustomize installed vendored flash-attn probe patch, "
                f"PID={os.getpid()}, summary={vendored_probe_summary}\n"
            )
    if site_log_enabled and dense_route_probe_summary.get("applied"):
        with open("/tmp/vllm_sparse_site.log", "a", encoding="utf-8") as _log:
            _log.write(
                "sitecustomize installed dense fa3 route trace probe, "
                f"PID={os.getpid()}, summary={dense_route_probe_summary}\n"
            )
    if dense_route_probe_requested and not dense_route_probe_summary.get("applied"):
        raise RuntimeError(
            "dense fa3 route trace probe requested but not installed: "
            f"{dense_route_probe_summary}"
        )
    if fa4_dense_gateway_requested:
        from patches.patch_installer import install_fa4_dense_fallback_gateway

        install_fa4_dense_fallback_gateway()
        if site_log_enabled:
            with open("/tmp/vllm_sparse_site.log", "a", encoding="utf-8") as _log:
                _log.write(
                    "sitecustomize installed fa4 dense fallback gateway, "
                    f"PID={os.getpid()}\n"
                )
except Exception as exc:
    if site_log_enabled:
        with open("/tmp/vllm_sparse_site.log", "a", encoding="utf-8") as _log:
            _log.write(
                "sitecustomize vendored flash-attn probe patch failed, "
                f"PID={os.getpid()}, error={exc}\n"
            )
    if vendored_probe_requested or fa4_dense_gateway_requested:
        raise SystemExit(
            f"sitecustomize vendored flash-attn probe patch failed: {exc}"
        ) from exc

# (EngineCore step-timing diagnostic hook retired: its installer
# install_engine_core_step_timing_diagnostic was deleted from patch_installer
# in an earlier sweep, leaving this entry a hard SystemExit for anyone setting
# VLLM_DECODE_ENGINE_CORE_STEP_LOG. Orphan state cluster removed with it.)


# Native CUDAGraphWrapper replay CUDA-event diagnostic.
# Default off. Enable with VLLM_NATIVE_FULLGRAPH_REPLAY_CUDA_EVENT_LOG.
# This is intentionally installed before the sparse patch so sparse wrapper
# original_call measures raw native CUDAGraphWrapper replay separately.
if os.environ.get("VLLM_NATIVE_FULLGRAPH_REPLAY_CUDA_EVENT_LOG"):
    try:
        import json
        import time

        import torch
        from vllm.compilation.cuda_graph import CUDAGraphWrapper
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import get_forward_context, is_forward_context_available

        _sfi_native_cgw_replay_log = os.environ.get(
            "VLLM_NATIVE_FULLGRAPH_REPLAY_CUDA_EVENT_LOG", ""
        )
        _sfi_native_cgw_replay_index = 0
        _sfi_original_cgw_call = CUDAGraphWrapper.__call__

        def _sfi_write_native_cgw_replay_record(payload):
            if not _sfi_native_cgw_replay_log:
                return
            parent = os.path.dirname(_sfi_native_cgw_replay_log)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(_sfi_native_cgw_replay_log, "a", encoding="utf-8") as handle:
                json.dump(payload, handle, sort_keys=True)
                handle.write("\n")

        if not getattr(CUDAGraphWrapper.__call__, "_sfi_native_cgw_event_patch", False):

            def _sfi_native_cgw_call(self, *args, **kwargs):
                global _sfi_native_cgw_replay_index
                should_time = False
                payload = {
                    "event": "native_cudagraph_wrapper_replay_cuda_event",
                    "pid": int(os.getpid()),
                }
                if _sfi_native_cgw_replay_log and is_forward_context_available():
                    fc = get_forward_context()
                    runtime_mode = getattr(fc, "cudagraph_runtime_mode", None)
                    wrapper_mode = getattr(self, "runtime_mode", None)
                    batch_descriptor = getattr(fc, "batch_descriptor", None)
                    entries = getattr(self, "concrete_cudagraph_entries", {})
                    entry = (
                        entries.get(batch_descriptor)
                        if batch_descriptor is not None and hasattr(entries, "get")
                        else None
                    )
                    cudagraph_present = bool(
                        entry is not None and getattr(entry, "cudagraph", None) is not None
                    )
                    should_time = bool(
                        runtime_mode == CUDAGraphMode.FULL
                        and wrapper_mode == CUDAGraphMode.FULL
                        and cudagraph_present
                    )
                    payload.update(
                        {
                            "forward_runtime_mode": getattr(runtime_mode, "name", str(runtime_mode)),
                            "wrapper_runtime_mode": getattr(wrapper_mode, "name", str(wrapper_mode)),
                            "batch_descriptor": str(batch_descriptor),
                            "entry_present": entry is not None,
                            "cudagraph_present": cudagraph_present,
                            "graph_entry_count": len(entries) if hasattr(entries, "__len__") else None,
                        }
                    )
                if not should_time:
                    return _sfi_original_cgw_call(self, *args, **kwargs)
                replay_index = int(_sfi_native_cgw_replay_index)
                _sfi_native_cgw_replay_index += 1
                host_begin_ns = time.time_ns()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                result = _sfi_original_cgw_call(self, *args, **kwargs)
                end.record()
                end.synchronize()
                host_end_ns = time.time_ns()
                try:
                    cuda_ms = float(start.elapsed_time(end))
                except Exception:
                    cuda_ms = -1.0
                payload.update(
                    {
                        "replay_index": replay_index,
                        "host_begin_ns": int(host_begin_ns),
                        "host_end_ns": int(host_end_ns),
                        "host_wall_us": (host_end_ns - host_begin_ns) / 1000.0,
                        "cuda_ms": cuda_ms,
                    }
                )
                _sfi_write_native_cgw_replay_record(payload)
                return result

            _sfi_native_cgw_call._sfi_native_cgw_event_patch = True
            CUDAGraphWrapper.__call__ = _sfi_native_cgw_call
            if site_log_enabled:
                with open("/tmp/vllm_sparse_site.log", "a", encoding="utf-8") as _log:
                    _log.write(
                        "sitecustomize installed native CUDAGraphWrapper replay diagnostic, "
                        f"PID={os.getpid()}\n"
                    )
    except Exception as exc:
        if site_log_enabled:
            with open("/tmp/vllm_sparse_site.log", "a", encoding="utf-8") as _log:
                _log.write(
                    "sitecustomize native CUDAGraphWrapper replay diagnostic failed, "
                    f"PID={os.getpid()}, error={exc}\n"
                )
        raise SystemExit(
            f"sitecustomize native CUDAGraphWrapper replay diagnostic failed: {exc}"
        ) from exc


# [K6-FORENSICS 2026-07-12] Replay-counted kineto window (env-gated diagnostic
# hook, default off — same family as the blocks above). Set
# VLLM_SPARSE_KINETO_REPLAY_WINDOW_OUT=<chrome_trace.json> to hook
# torch.cuda.CUDAGraph.replay: at replay #VLLM_SPARSE_KINETO_REPLAY_START
# (default 150) it enters a CUDA-activity-only torch profiler and
# VLLM_SPARSE_KINETO_REPLAY_SPAN (default 40) replays later dumps the trace.
# Purpose: count in-graph prepare launches per decode step (K6 36->1 refit).
# Observation only; production numbers must come from profiler-free runs.
if os.environ.get("VLLM_SPARSE_KINETO_REPLAY_WINDOW_OUT"):
    try:
        import threading as _sfi_k6_threading

        import torch as _sfi_k6_torch
        from torch.profiler import (
            ProfilerActivity as _sfi_k6_PA,
            profile as _sfi_k6_profile,
        )

        _sfi_k6_state = {"n": 0, "prof": None, "done": False}
        _sfi_k6_lock = _sfi_k6_threading.Lock()
        _sfi_k6_orig_replay = _sfi_k6_torch.cuda.CUDAGraph.replay
        _sfi_k6_start = int(
            os.environ.get("VLLM_SPARSE_KINETO_REPLAY_START", "150")
        )
        _sfi_k6_span = int(os.environ.get("VLLM_SPARSE_KINETO_REPLAY_SPAN", "40"))
        _sfi_k6_out = os.environ["VLLM_SPARSE_KINETO_REPLAY_WINDOW_OUT"]

        def _sfi_k6_replay_hook(self, *args, **kwargs):
            do_start = do_stop = False
            with _sfi_k6_lock:
                _sfi_k6_state["n"] += 1
                n = _sfi_k6_state["n"]
                if (
                    n == _sfi_k6_start
                    and _sfi_k6_state["prof"] is None
                    and not _sfi_k6_state["done"]
                ):
                    do_start = True
                elif (
                    _sfi_k6_state["prof"] is not None
                    and not _sfi_k6_state["done"]
                    and n >= _sfi_k6_start + _sfi_k6_span
                ):
                    do_stop = True
            if do_start:
                try:
                    p = _sfi_k6_profile(activities=[_sfi_k6_PA.CUDA])
                    p.__enter__()
                    with _sfi_k6_lock:
                        _sfi_k6_state["prof"] = p
                    print(f"[k6-kineto] profiler ON at replay {n}", flush=True)
                except Exception as exc:
                    print(f"[k6-kineto] profiler start failed: {exc!r}", flush=True)
                    with _sfi_k6_lock:
                        _sfi_k6_state["done"] = True
            if do_stop:
                with _sfi_k6_lock:
                    p = _sfi_k6_state["prof"]
                    _sfi_k6_state["prof"] = None
                    _sfi_k6_state["done"] = True
                try:
                    p.__exit__(None, None, None)
                    p.export_chrome_trace(_sfi_k6_out)
                    print(
                        f"[k6-kineto] profiler OFF at replay {_sfi_k6_state['n']}, "
                        f"dumped {_sfi_k6_out}",
                        flush=True,
                    )
                except Exception as exc:
                    print(f"[k6-kineto] profiler stop failed: {exc!r}", flush=True)
            return _sfi_k6_orig_replay(self, *args, **kwargs)

        if not getattr(_sfi_k6_torch.cuda.CUDAGraph.replay, "_sfi_k6_kineto", False):
            _sfi_k6_replay_hook._sfi_k6_kineto = True
            _sfi_k6_torch.cuda.CUDAGraph.replay = _sfi_k6_replay_hook
            print(
                f"[k6-kineto] replay hook armed (start={_sfi_k6_start}, "
                f"span={_sfi_k6_span})",
                flush=True,
            )
    except Exception as exc:
        raise SystemExit(
            f"sitecustomize K6 kineto replay-window hook failed: {exc}"
        ) from exc


# Executor/MQ timing diagnostic.
# Default off. Enable with VLLM_EXECUTOR_RPC_TIMING_LOG=/path/to/jsonl.
# This splits MultiprocExecutor.collective_rpc(non_block=True) into host-side
# RPC enqueue, MessageQueue pickle/ring-buffer phases, worker method runtime,
# response enqueue, and FutureWrapper.result drain time.
if os.environ.get("VLLM_EXECUTOR_RPC_TIMING_LOG"):
    try:
        import json as _sfi_json
        import threading as _sfi_threading
        import time as _sfi_time
        import traceback as _sfi_traceback
        from functools import partial as _sfi_partial
        from pickle import PickleBuffer as _sfi_PickleBuffer
        import pickle as _sfi_pickle

        import cloudpickle as _sfi_cloudpickle
        from vllm.distributed.device_communicators import shm_broadcast as _sfi_shm
        from vllm.v1.executor import multiproc_executor as _sfi_mpe

        _sfi_rpc_timing_log = os.environ.get("VLLM_EXECUTOR_RPC_TIMING_LOG", "")
        _sfi_rpc_tls = _sfi_threading.local()
        _sfi_rpc_seq_lock = _sfi_threading.Lock()
        _sfi_rpc_seq = 0

        def _sfi_rpc_now_ns():
            return _sfi_time.perf_counter_ns()

        def _sfi_rpc_us(start_ns, end_ns):
            return (int(end_ns) - int(start_ns)) / 1000.0

        def _sfi_next_rpc_seq():
            global _sfi_rpc_seq
            with _sfi_rpc_seq_lock:
                seq = _sfi_rpc_seq
                _sfi_rpc_seq += 1
            return int(seq)

        def _sfi_rpc_write(payload):
            if not _sfi_rpc_timing_log:
                return
            try:
                payload.setdefault("pid", int(os.getpid()))
                payload.setdefault("ts_ns", int(_sfi_time.time_ns()))
                parent = os.path.dirname(_sfi_rpc_timing_log)
                if parent:
                    os.makedirs(parent, exist_ok=True)
                with open(_sfi_rpc_timing_log, "a", encoding="utf-8") as handle:
                    handle.write(_sfi_json.dumps(payload, sort_keys=True, default=str))
                    handle.write("\n")
            except Exception:
                pass

        def _sfi_len_or_none(value):
            try:
                return len(value)
            except Exception:
                return None

        def _sfi_shape_or_none(value):
            shape = getattr(value, "shape", None)
            if shape is None:
                return None
            try:
                return [int(x) for x in shape]
            except Exception:
                return str(shape)

        def _sfi_scheduler_summary(obj, prefix):
            out = {f"{prefix}_type": type(obj).__name__}
            for name in (
                "total_num_scheduled_tokens",
                "num_scheduled_tokens",
                "scheduled_new_reqs",
                "scheduled_cached_reqs",
                "num_common_prefix_blocks",
                "finished_req_ids",
                "grammar_bitmask",
            ):
                if not hasattr(obj, name):
                    continue
                value = getattr(obj, name)
                shape = _sfi_shape_or_none(value)
                if shape is not None:
                    out[f"{prefix}_{name}_shape"] = shape
                    continue
                if isinstance(value, dict):
                    out[f"{prefix}_{name}_len"] = len(value)
                    if name == "num_scheduled_tokens":
                        try:
                            out[f"{prefix}_{name}_sum"] = int(sum(int(v) for v in value.values()))
                        except Exception:
                            pass
                    continue
                if isinstance(value, (list, tuple, set, frozenset)):
                    out[f"{prefix}_{name}_len"] = len(value)
                    continue
                if isinstance(value, (int, float, bool, str)) or value is None:
                    out[f"{prefix}_{name}"] = value
                else:
                    out[f"{prefix}_{name}_type"] = type(value).__name__
                    out[f"{prefix}_{name}_len"] = _sfi_len_or_none(value)
            return out

        def _sfi_payload_summary(obj):
            summary = {"payload_type": type(obj).__name__}
            if isinstance(obj, tuple) and len(obj) == 4:
                method, args, kwargs, output_rank = obj
                summary.update({
                    "payload_kind": "rpc",
                    "rpc_method": method if isinstance(method, str) else "<callable_bytes>",
                    "arg_count": len(args) if isinstance(args, tuple) else None,
                    "kwarg_count": len(kwargs) if isinstance(kwargs, dict) else None,
                    "output_rank": output_rank,
                })
                if isinstance(args, tuple) and args:
                    summary.update(_sfi_scheduler_summary(args[0], "arg0"))
                return summary
            if isinstance(obj, tuple) and len(obj) == 2 and hasattr(obj[0], "name"):
                summary.update({
                    "payload_kind": "response",
                    "response_status": getattr(obj[0], "name", str(obj[0])),
                    "response_type": type(obj[1]).__name__,
                    "worker_rpc_method": getattr(_sfi_rpc_tls, "worker_rpc_method", None),
                })
                return summary
            summary["payload_kind"] = "other"
            return summary

        if not getattr(_sfi_shm.MessageQueue.enqueue, "_sfi_rpc_timing_patch", False):

            def _sfi_timed_mq_enqueue(self, obj, timeout=None):
                assert self._is_writer, "Only writers can enqueue"
                all_buffers = [b""]
                total_bytes = 6

                def oob_callback(buf: _sfi_PickleBuffer) -> bool:
                    raw_buf = buf.raw()
                    if len(raw_buf) < 1024 * 1024:
                        return True
                    all_buffers.append(raw_buf)
                    nonlocal total_bytes
                    total_bytes += len(raw_buf) + 4
                    return False

                t0 = _sfi_rpc_now_ns()
                all_buffers[0] = _sfi_pickle.dumps(
                    obj, protocol=_sfi_pickle.HIGHEST_PROTOCOL, buffer_callback=oob_callback
                )
                t_pickle = _sfi_rpc_now_ns()
                local_acquire_us = None
                local_copy_us = None
                local_exit_us = None
                local_overflow_send_us = None
                remote_send_us = None
                overflow = None
                main_buffer_bytes = len(all_buffers[0])
                oob_buffer_count = max(0, len(all_buffers) - 1)
                oob_total_bytes = sum(len(buffer) for buffer in all_buffers[1:])

                if self.n_local_reader > 0:
                    overflow = total_bytes + main_buffer_bytes >= self.buffer.max_chunk_bytes
                    if overflow:
                        ta0 = _sfi_rpc_now_ns()
                        with self.acquire_write(timeout) as buf:
                            ta1 = _sfi_rpc_now_ns()
                            buf[0] = 1
                            tc1 = _sfi_rpc_now_ns()
                        ta2 = _sfi_rpc_now_ns()
                        ts0 = _sfi_rpc_now_ns()
                        self.local_socket.send_multipart(all_buffers, copy=False)
                        ts1 = _sfi_rpc_now_ns()
                        local_overflow_send_us = _sfi_rpc_us(ts0, ts1)
                    else:
                        ta0 = _sfi_rpc_now_ns()
                        with self.acquire_write(timeout) as buf:
                            ta1 = _sfi_rpc_now_ns()
                            buf[0] = 0
                            offset = 3
                            buf[1:offset] = _sfi_shm.to_bytes_big(len(all_buffers), 2)
                            for buffer in all_buffers:
                                buf_len = len(buffer)
                                buf_offset = offset + 4
                                buf[offset:buf_offset] = _sfi_shm.to_bytes_big(buf_len, 4)
                                buf[buf_offset : (offset := buf_offset + buf_len)] = buffer
                            tc1 = _sfi_rpc_now_ns()
                        ta2 = _sfi_rpc_now_ns()
                    local_acquire_us = _sfi_rpc_us(ta0, ta1)
                    local_copy_us = _sfi_rpc_us(ta1, tc1)
                    local_exit_us = _sfi_rpc_us(tc1, ta2)
                    self._spin_condition.notify()

                if self.n_remote_reader > 0:
                    tr0 = _sfi_rpc_now_ns()
                    self.remote_socket.send_multipart(all_buffers, copy=False)
                    tr1 = _sfi_rpc_now_ns()
                    remote_send_us = _sfi_rpc_us(tr0, tr1)

                t1 = _sfi_rpc_now_ns()
                record = {
                    "event": "message_queue_enqueue",
                    "queue_id": int(id(self)),
                    "current_idx_after": getattr(self, "current_idx", None),
                    "n_local_reader": int(getattr(self, "n_local_reader", 0)),
                    "n_remote_reader": int(getattr(self, "n_remote_reader", 0)),
                    "pickle_us": _sfi_rpc_us(t0, t_pickle),
                    "local_acquire_us": local_acquire_us,
                    "local_copy_us": local_copy_us,
                    "local_exit_us": local_exit_us,
                    "local_overflow_send_us": local_overflow_send_us,
                    "remote_send_us": remote_send_us,
                    "total_us": _sfi_rpc_us(t0, t1),
                    "main_buffer_bytes": int(main_buffer_bytes),
                    "oob_buffer_count": int(oob_buffer_count),
                    "oob_total_bytes": int(oob_total_bytes),
                    "overflow": overflow,
                    "main_rpc_method": getattr(_sfi_rpc_tls, "main_rpc_method", None),
                    "main_rpc_seq": getattr(_sfi_rpc_tls, "main_rpc_seq", None),
                    "worker_rpc_method": getattr(_sfi_rpc_tls, "worker_rpc_method", None),
                }
                record.update(_sfi_payload_summary(obj))
                _sfi_rpc_write(record)

            _sfi_timed_mq_enqueue._sfi_rpc_timing_patch = True
            _sfi_shm.MessageQueue.enqueue = _sfi_timed_mq_enqueue

        if not getattr(_sfi_mpe.FutureWrapper.result, "_sfi_rpc_timing_patch", False):
            _sfi_orig_future_result = _sfi_mpe.FutureWrapper.result
            _sfi_orig_wait_for_response = _sfi_mpe.FutureWrapper.wait_for_response

            def _sfi_timed_future_result(self, timeout=None):
                t0 = _sfi_rpc_now_ns()
                q_before = _sfi_len_or_none(self.futures_queue)
                try:
                    return _sfi_orig_future_result(self, timeout=timeout)
                finally:
                    t1 = _sfi_rpc_now_ns()
                    _sfi_rpc_write({
                        "event": "future_result",
                        "rpc_method": getattr(self, "_sfi_rpc_method", None),
                        "rpc_seq": getattr(self, "_sfi_rpc_seq", None),
                        "queue_len_before": q_before,
                        "queue_len_after": _sfi_len_or_none(self.futures_queue),
                        "done": bool(self.done()),
                        "total_us": _sfi_rpc_us(t0, t1),
                    })

            def _sfi_timed_wait_for_response(self, get_response):
                t0 = _sfi_rpc_now_ns()
                _sfi_orig_wait_for_response(self, get_response)
                t1 = _sfi_rpc_now_ns()
                exc_type = None
                try:
                    exc = self.exception() if self.done() else None
                    exc_type = type(exc).__name__ if exc is not None else None
                except Exception:
                    exc_type = "<exception_check_failed>"
                _sfi_rpc_write({
                    "event": "future_wait_for_response",
                    "rpc_method": getattr(self, "_sfi_rpc_method", None),
                    "rpc_seq": getattr(self, "_sfi_rpc_seq", None),
                    "done": bool(self.done()),
                    "exception_type": exc_type,
                    "wait_us": _sfi_rpc_us(t0, t1),
                })

            _sfi_timed_future_result._sfi_rpc_timing_patch = True
            _sfi_timed_wait_for_response._sfi_rpc_timing_patch = True
            _sfi_mpe.FutureWrapper.result = _sfi_timed_future_result
            _sfi_mpe.FutureWrapper.wait_for_response = _sfi_timed_wait_for_response

        if not getattr(_sfi_mpe.MultiprocExecutor.collective_rpc, "_sfi_rpc_timing_patch", False):

            def _sfi_timed_collective_rpc(
                self,
                method,
                timeout=None,
                args=(),
                kwargs=None,
                non_block=False,
                unique_reply_rank=None,
                kv_output_aggregator=None,
            ):
                assert self.rpc_broadcast_mq is not None, (
                    "collective_rpc should not be called on follower node"
                )
                if self.is_failed:
                    raise RuntimeError("Executor failed.")

                t0 = _sfi_rpc_now_ns()
                deadline = None if timeout is None else _sfi_time.monotonic() + timeout
                kwargs = kwargs or {}
                if kv_output_aggregator is not None:
                    output_rank = None
                    aggregate = _sfi_partial(
                        kv_output_aggregator.aggregate,
                        output_rank=unique_reply_rank or 0,
                    )
                else:
                    output_rank = unique_reply_rank
                    aggregate = lambda x: x

                t_ser0 = _sfi_rpc_now_ns()
                if isinstance(method, str):
                    send_method = method
                    method_name = method
                else:
                    send_method = _sfi_cloudpickle.dumps(
                        method, protocol=_sfi_pickle.HIGHEST_PROTOCOL
                    )
                    method_name = "<callable>"
                t_ser1 = _sfi_rpc_now_ns()

                seq = _sfi_next_rpc_seq()
                q_before = _sfi_len_or_none(self.futures_queue)
                _sfi_rpc_tls.main_rpc_method = method_name
                _sfi_rpc_tls.main_rpc_seq = seq
                t_enq0 = _sfi_rpc_now_ns()
                try:
                    self.rpc_broadcast_mq.enqueue((send_method, args, kwargs, output_rank))
                finally:
                    _sfi_rpc_tls.main_rpc_method = None
                    _sfi_rpc_tls.main_rpc_seq = None
                t_enq1 = _sfi_rpc_now_ns()

                response_mqs = self.response_mqs
                if output_rank is not None:
                    response_mqs = (response_mqs[output_rank],)

                def get_response():
                    responses = []
                    for mq in response_mqs:
                        dequeue_timeout = None if deadline is None else (deadline - _sfi_time.monotonic())
                        try:
                            status, result = mq.dequeue(timeout=dequeue_timeout)
                        except TimeoutError as e:
                            raise TimeoutError(f"RPC call to {method} timed out.") from e
                        if status != _sfi_mpe.WorkerProc.ResponseStatus.SUCCESS:
                            raise RuntimeError(
                                f"Worker failed with error '{result}', please check the"
                                " stack trace above for the root cause"
                            )
                        responses.append(result)
                    return responses[0] if output_rank is not None else responses

                if non_block:
                    tf0 = _sfi_rpc_now_ns()
                    future = _sfi_mpe.FutureWrapper(self.futures_queue, aggregate=aggregate)
                    future._sfi_rpc_method = method_name
                    future._sfi_rpc_seq = seq
                    self.futures_queue.appendleft((future, get_response))
                    tf1 = _sfi_rpc_now_ns()
                    record = {
                        "event": "collective_rpc",
                        "rpc_method": method_name,
                        "rpc_seq": seq,
                        "non_block": bool(non_block),
                        "output_rank": output_rank,
                        "unique_reply_rank": unique_reply_rank,
                        "response_mq_count": len(response_mqs),
                        "send_method_serialize_us": _sfi_rpc_us(t_ser0, t_ser1),
                        "enqueue_us": _sfi_rpc_us(t_enq0, t_enq1),
                        "future_create_us": _sfi_rpc_us(tf0, tf1),
                        "total_us": _sfi_rpc_us(t0, tf1),
                        "futures_queue_len_before": q_before,
                        "futures_queue_len_after": _sfi_len_or_none(self.futures_queue),
                    }
                    if args:
                        record.update(_sfi_scheduler_summary(args[0], "arg0"))
                    _sfi_rpc_write(record)
                    return future

                while self.futures_queue:
                    future, get_fut_response = self.futures_queue.pop()
                    future.wait_for_response(get_fut_response)
                tg0 = _sfi_rpc_now_ns()
                result = aggregate(get_response())
                tg1 = _sfi_rpc_now_ns()
                _sfi_rpc_write({
                    "event": "collective_rpc",
                    "rpc_method": method_name,
                    "rpc_seq": seq,
                    "non_block": bool(non_block),
                    "output_rank": output_rank,
                    "unique_reply_rank": unique_reply_rank,
                    "response_mq_count": len(response_mqs),
                    "send_method_serialize_us": _sfi_rpc_us(t_ser0, t_ser1),
                    "enqueue_us": _sfi_rpc_us(t_enq0, t_enq1),
                    "sync_get_response_us": _sfi_rpc_us(tg0, tg1),
                    "total_us": _sfi_rpc_us(t0, tg1),
                    "futures_queue_len_before": q_before,
                    "futures_queue_len_after": _sfi_len_or_none(self.futures_queue),
                })
                return result

            _sfi_timed_collective_rpc._sfi_rpc_timing_patch = True
            _sfi_mpe.MultiprocExecutor.collective_rpc = _sfi_timed_collective_rpc

        if not getattr(_sfi_mpe.WorkerProc.worker_busy_loop, "_sfi_rpc_timing_patch", False):

            def _sfi_timed_worker_busy_loop(self):
                assert self.rpc_broadcast_mq is not None
                while True:
                    td0 = _sfi_rpc_now_ns()
                    method, args, kwargs, output_rank = self.rpc_broadcast_mq.dequeue(indefinite=True)
                    td1 = _sfi_rpc_now_ns()
                    method_name = method if isinstance(method, str) else "<callable_bytes>"
                    handle_us = None
                    func_us = None
                    output_type = None
                    try:
                        if isinstance(method, str):
                            func = getattr(self.worker, method)
                        elif isinstance(method, bytes):
                            func = _sfi_partial(_sfi_cloudpickle.loads(method), self.worker)
                        tf0 = _sfi_rpc_now_ns()
                        output = func(*args, **kwargs)
                        tf1 = _sfi_rpc_now_ns()
                        func_us = _sfi_rpc_us(tf0, tf1)
                        output_type = type(output).__name__
                    except Exception as e:
                        if hasattr(e, "add_note"):
                            e.add_note(_sfi_traceback.format_exc())
                        _sfi_mpe.logger.exception("WorkerProc hit an exception.")
                        if output_rank is None or self.rank == output_rank:
                            _sfi_rpc_tls.worker_rpc_method = method_name
                            th0 = _sfi_rpc_now_ns()
                            try:
                                self.handle_output(e)
                            finally:
                                _sfi_rpc_tls.worker_rpc_method = None
                            th1 = _sfi_rpc_now_ns()
                            handle_us = _sfi_rpc_us(th0, th1)
                        _sfi_rpc_write({
                            "event": "worker_rpc",
                            "rank": int(getattr(self, "rank", -1)),
                            "rpc_method": method_name,
                            "output_rank": output_rank,
                            "dequeue_wait_us": _sfi_rpc_us(td0, td1),
                            "func_us": func_us,
                            "handle_output_us": handle_us,
                            "output_type": "exception",
                        })
                        continue

                    if output_rank is None or self.rank == output_rank:
                        _sfi_rpc_tls.worker_rpc_method = method_name
                        th0 = _sfi_rpc_now_ns()
                        try:
                            self.handle_output(output)
                        finally:
                            _sfi_rpc_tls.worker_rpc_method = None
                        th1 = _sfi_rpc_now_ns()
                        handle_us = _sfi_rpc_us(th0, th1)
                    _sfi_rpc_write({
                        "event": "worker_rpc",
                        "rank": int(getattr(self, "rank", -1)),
                        "rpc_method": method_name,
                        "output_rank": output_rank,
                        "dequeue_wait_us": _sfi_rpc_us(td0, td1),
                        "func_us": func_us,
                        "handle_output_us": handle_us,
                        "output_type": output_type,
                    })

            _sfi_timed_worker_busy_loop._sfi_rpc_timing_patch = True
            _sfi_mpe.WorkerProc.worker_busy_loop = _sfi_timed_worker_busy_loop

        if site_log_enabled:
            with open("/tmp/vllm_sparse_site.log", "a", encoding="utf-8") as _log:
                _log.write(
                    "sitecustomize installed executor RPC timing diagnostic, "
                    f"PID={os.getpid()}\n"
                )
    except Exception as exc:
        if site_log_enabled:
            with open("/tmp/vllm_sparse_site.log", "a", encoding="utf-8") as _log:
                _log.write(
                    "sitecustomize executor RPC timing diagnostic failed, "
                    f"PID={os.getpid()}, error={exc}\n"
                )
        raise SystemExit(
            f"sitecustomize executor RPC timing diagnostic failed: {exc}"
        ) from exc


# UniProc executor timing diagnostic.
# Default off. Shares VLLM_EXECUTOR_RPC_TIMING_LOG with the executor/MQ block.
# Single-GPU vLLM often uses UniProcExecutor, where non_block=True still runs
# driver_worker.execute_model synchronously and only get_output may be deferred.
if os.environ.get("VLLM_EXECUTOR_RPC_TIMING_LOG"):
    try:
        import json as _sfi_uni_json
        import time as _sfi_uni_time
        from concurrent.futures import Future as _sfi_UniFuture

        from vllm.v1.executor import uniproc_executor as _sfi_uni

        if "_sfi_rpc_write" not in globals():
            _sfi_rpc_timing_log = os.environ.get("VLLM_EXECUTOR_RPC_TIMING_LOG", "")

            def _sfi_rpc_write(payload):
                if not _sfi_rpc_timing_log:
                    return
                try:
                    payload.setdefault("pid", int(os.getpid()))
                    payload.setdefault("ts_ns", int(_sfi_uni_time.time_ns()))
                    parent = os.path.dirname(_sfi_rpc_timing_log)
                    if parent:
                        os.makedirs(parent, exist_ok=True)
                    with open(_sfi_rpc_timing_log, "a", encoding="utf-8") as handle:
                        handle.write(_sfi_uni_json.dumps(payload, sort_keys=True, default=str))
                        handle.write("\n")
                except Exception:
                    pass

        if "_sfi_rpc_now_ns" not in globals():
            def _sfi_rpc_now_ns():
                return _sfi_uni_time.perf_counter_ns()

        if "_sfi_rpc_us" not in globals():
            def _sfi_rpc_us(start_ns, end_ns):
                return (int(end_ns) - int(start_ns)) / 1000.0

        if "_sfi_next_rpc_seq" not in globals():
            _sfi_uni_seq = 0

            def _sfi_next_rpc_seq():
                global _sfi_uni_seq
                seq = _sfi_uni_seq
                _sfi_uni_seq += 1
                return int(seq)

        if "_sfi_scheduler_summary" not in globals():
            def _sfi_scheduler_summary(obj, prefix):
                out = {f"{prefix}_type": type(obj).__name__}
                for name in (
                    "total_num_scheduled_tokens",
                    "num_scheduled_tokens",
                    "scheduled_new_reqs",
                    "scheduled_cached_reqs",
                    "finished_req_ids",
                    "grammar_bitmask",
                ):
                    if not hasattr(obj, name):
                        continue
                    value = getattr(obj, name)
                    if isinstance(value, dict):
                        out[f"{prefix}_{name}_len"] = len(value)
                        if name == "num_scheduled_tokens":
                            try:
                                out[f"{prefix}_{name}_sum"] = int(sum(int(v) for v in value.values()))
                            except Exception:
                                pass
                    elif isinstance(value, (list, tuple, set, frozenset)):
                        out[f"{prefix}_{name}_len"] = len(value)
                    elif isinstance(value, (int, float, bool, str)) or value is None:
                        out[f"{prefix}_{name}"] = value
                    else:
                        out[f"{prefix}_{name}_type"] = type(value).__name__
                return out

        def _sfi_tag_uniproc_future(future, method_name, seq, source):
            if getattr(future, "_sfi_uniproc_result_wrapped", False):
                return future
            orig_result = future.result

            def _sfi_timed_uniproc_future_result(timeout=None):
                t0 = _sfi_rpc_now_ns()
                try:
                    return orig_result(timeout=timeout)
                finally:
                    t1 = _sfi_rpc_now_ns()
                    _sfi_rpc_write({
                        "event": "uniproc_future_result",
                        "rpc_method": method_name,
                        "rpc_seq": seq,
                        "future_source": source,
                        "future_done_after": bool(future.done()),
                        "result_us": _sfi_rpc_us(t0, t1),
                    })

            future.result = _sfi_timed_uniproc_future_result
            future._sfi_uniproc_result_wrapped = True
            future._sfi_rpc_method = method_name
            future._sfi_rpc_seq = seq
            return future

        if not getattr(_sfi_uni.UniProcExecutor.collective_rpc, "_sfi_uniproc_timing_patch", False):

            def _sfi_timed_uniproc_collective_rpc(
                self,
                method,
                timeout=None,
                args=(),
                kwargs=None,
                non_block=False,
                single_value=False,
            ):
                del timeout
                if kwargs is None:
                    kwargs = {}
                method_name = method if isinstance(method, str) else "<callable>"
                seq = _sfi_next_rpc_seq()
                record = {
                    "event": "uniproc_collective_rpc",
                    "rpc_method": method_name,
                    "rpc_seq": seq,
                    "non_block": bool(non_block),
                    "single_value": bool(single_value),
                    "async_output_thread_present": self.async_output_thread is not None,
                    "max_concurrent_batches": int(getattr(self, "max_concurrent_batches", -1)),
                }
                if args:
                    record.update(_sfi_scheduler_summary(args[0], "arg0"))

                if not non_block:
                    t0 = _sfi_rpc_now_ns()
                    result = _sfi_uni.run_method(self.driver_worker, method, args, kwargs)
                    t1 = _sfi_rpc_now_ns()
                    record.update({
                        "result_type": type(result).__name__,
                        "run_method_us": _sfi_rpc_us(t0, t1),
                        "total_us": _sfi_rpc_us(t0, t1),
                        "future_source": "sync_return",
                    })
                    _sfi_rpc_write(record)
                    return result if single_value else [result]

                t0 = _sfi_rpc_now_ns()
                try:
                    result = _sfi_uni.run_method(self.driver_worker, method, args, kwargs)
                    t_run = _sfi_rpc_now_ns()
                    record.update({
                        "result_type": type(result).__name__,
                        "run_method_us": _sfi_rpc_us(t0, t_run),
                    })
                    if isinstance(result, _sfi_uni.AsyncModelRunnerOutput):
                        record["is_async_model_runner_output"] = True
                        if (async_thread := self.async_output_thread) is not None:
                            t_submit0 = _sfi_rpc_now_ns()
                            if single_value:
                                future = async_thread.submit(result.get_output)
                                future_source = "async_thread_get_output"
                            else:
                                def get_output_list():
                                    return [result.get_output()]
                                future = async_thread.submit(get_output_list)
                                future_source = "async_thread_get_output_list"
                            t_submit1 = _sfi_rpc_now_ns()
                            record.update({
                                "async_submit_us": _sfi_rpc_us(t_submit0, t_submit1),
                                "inline_get_output_us": None,
                                "future_done_at_return": bool(future.done()),
                                "future_source": future_source,
                                "total_us": _sfi_rpc_us(t0, t_submit1),
                            })
                            _sfi_rpc_write(record)
                            return _sfi_tag_uniproc_future(future, method_name, seq, future_source)

                        t_get0 = _sfi_rpc_now_ns()
                        result = result.get_output()
                        t_get1 = _sfi_rpc_now_ns()
                        record.update({
                            "inline_get_output_us": _sfi_rpc_us(t_get0, t_get1),
                            "future_source": "inline_get_output_done_future",
                        })
                    else:
                        record["is_async_model_runner_output"] = False
                        record["inline_get_output_us"] = None
                        record["future_source"] = "immediate_done_future"

                    future = _sfi_UniFuture()
                    future.set_result(result if single_value else [result])
                    t_done = _sfi_rpc_now_ns()
                    record.update({
                        "future_done_at_return": bool(future.done()),
                        "result_type_after_get_output": type(result).__name__,
                        "total_us": _sfi_rpc_us(t0, t_done),
                    })
                    _sfi_rpc_write(record)
                    return _sfi_tag_uniproc_future(future, method_name, seq, record["future_source"])
                except Exception as e:
                    future = _sfi_UniFuture()
                    future.set_exception(e)
                    t_err = _sfi_rpc_now_ns()
                    record.update({
                        "exception_type": type(e).__name__,
                        "future_done_at_return": bool(future.done()),
                        "future_source": "exception_done_future",
                        "total_us": _sfi_rpc_us(t0, t_err),
                    })
                    _sfi_rpc_write(record)
                    return _sfi_tag_uniproc_future(future, method_name, seq, "exception_done_future")

            _sfi_timed_uniproc_collective_rpc._sfi_uniproc_timing_patch = True
            _sfi_uni.UniProcExecutor.collective_rpc = _sfi_timed_uniproc_collective_rpc

        if site_log_enabled:
            with open("/tmp/vllm_sparse_site.log", "a", encoding="utf-8") as _log:
                _log.write(
                    "sitecustomize installed uniproc executor timing diagnostic, "
                    f"PID={os.getpid()}\n"
                )
    except Exception as exc:
        if site_log_enabled:
            with open("/tmp/vllm_sparse_site.log", "a", encoding="utf-8") as _log:
                _log.write(
                    "sitecustomize uniproc executor timing diagnostic failed, "
                    f"PID={os.getpid()}, error={exc}\n"
                )
        raise SystemExit(
            f"sitecustomize uniproc executor timing diagnostic failed: {exc}"
        ) from exc


# GPUModelRunner execute_model phase timing diagnostic.
# Default off. Enable with VLLM_GPU_EXECUTE_PHASE_TIMING_LOG=/path/to/jsonl.
# Records are buffered in-process and flushed at exit to avoid moving waits
# between execute_model/sample_tokens/future.result during the measured steps.
if os.environ.get("VLLM_GPU_EXECUTE_PHASE_TIMING_LOG"):
    try:
        import atexit as _sfi_gpu_atexit
        import json as _sfi_gpu_json
        import threading as _sfi_gpu_threading
        import time as _sfi_gpu_time

        from vllm.v1.worker import gpu_model_runner as _sfi_gmr

        _sfi_gpu_phase_log = os.environ.get("VLLM_GPU_EXECUTE_PHASE_TIMING_LOG", "")
        _sfi_gpu_phase_records = []
        _sfi_gpu_phase_lock = _sfi_gpu_threading.Lock()
        _sfi_gpu_phase_tls = _sfi_gpu_threading.local()
        _sfi_gpu_execute_step = 0
        _sfi_gpu_execute_step_lock = _sfi_gpu_threading.Lock()

        def _sfi_gpu_now_ns():
            return _sfi_gpu_time.perf_counter_ns()

        def _sfi_gpu_us(start_ns, end_ns):
            return (int(end_ns) - int(start_ns)) / 1000.0

        def _sfi_gpu_next_step():
            global _sfi_gpu_execute_step
            with _sfi_gpu_execute_step_lock:
                step = _sfi_gpu_execute_step
                _sfi_gpu_execute_step += 1
            return int(step)

        def _sfi_gpu_append(record):
            if not _sfi_gpu_phase_log:
                return
            record.setdefault("pid", int(os.getpid()))
            record.setdefault("ts_ns", int(_sfi_gpu_time.time_ns()))
            flush_now = False
            with _sfi_gpu_phase_lock:
                _sfi_gpu_phase_records.append(record)
                # [2026-07-08] multiproc worker 经 os._exit 退出不跑 atexit,
                # 纯 atexit flush 在 TP>1 worker 里永远得到空文件(此前只在
                # in-proc 形态用过)。加尺寸触发批量落盘(~每 512 条一次
                # append,幅度=几十步一次,不在单步内引入等待)。
                if len(_sfi_gpu_phase_records) >= 512:
                    flush_now = True
            if flush_now:
                _sfi_gpu_flush()

        def _sfi_gpu_flush():
            if not _sfi_gpu_phase_log:
                return
            with _sfi_gpu_phase_lock:
                records = list(_sfi_gpu_phase_records)
                _sfi_gpu_phase_records.clear()
            if not records:
                return
            try:
                parent = os.path.dirname(_sfi_gpu_phase_log)
                if parent:
                    os.makedirs(parent, exist_ok=True)
                with open(_sfi_gpu_phase_log, "a", encoding="utf-8") as handle:
                    for record in records:
                        handle.write(_sfi_gpu_json.dumps(record, sort_keys=True, default=str))
                        handle.write("\n")
            except Exception:
                pass

        _sfi_gpu_atexit.register(_sfi_gpu_flush)

        def _sfi_gpu_scheduler_summary(scheduler_output):
            out = {}
            if scheduler_output is None:
                return out
            for name in ("total_num_scheduled_tokens", "num_scheduled_tokens", "scheduled_new_reqs", "scheduled_cached_reqs"):
                if not hasattr(scheduler_output, name):
                    continue
                value = getattr(scheduler_output, name)
                if isinstance(value, dict):
                    out[f"{name}_len"] = len(value)
                    if name == "num_scheduled_tokens":
                        try:
                            out[f"{name}_sum"] = int(sum(int(v) for v in value.values()))
                        except Exception:
                            pass
                elif isinstance(value, (list, tuple, set, frozenset)):
                    out[f"{name}_len"] = len(value)
                elif isinstance(value, (int, float, bool, str)) or value is None:
                    out[name] = value
                else:
                    out[f"{name}_type"] = type(value).__name__
            return out

        def _sfi_gpu_patch_compute_logits(runner):
            model = getattr(runner, "model", None)
            if model is None:
                return
            orig = getattr(model, "compute_logits", None)
            if not callable(orig) or getattr(orig, "_sfi_gpu_phase_patch", False):
                return

            def _sfi_timed_compute_logits(*args, **kwargs):
                if not getattr(_sfi_gpu_phase_tls, "in_execute_model", False):
                    return orig(*args, **kwargs)
                t0 = _sfi_gpu_now_ns()
                try:
                    return orig(*args, **kwargs)
                finally:
                    t1 = _sfi_gpu_now_ns()
                    _sfi_gpu_append({
                        "event": "gpu_execute_phase",
                        "phase": "compute_logits",
                        "step": getattr(_sfi_gpu_phase_tls, "step", -1),
                        "duration_us": _sfi_gpu_us(t0, t1),
                    })

            _sfi_timed_compute_logits._sfi_gpu_phase_patch = True
            setattr(model, "compute_logits", _sfi_timed_compute_logits)

        def _sfi_wrap_gpu_runner_method(name):
            orig = getattr(_sfi_gmr.GPUModelRunner, name, None)
            if not callable(orig) or getattr(orig, "_sfi_gpu_phase_patch", False):
                return

            def _sfi_timed_method(self, *args, **kwargs):
                if not getattr(_sfi_gpu_phase_tls, "in_execute_model", False):
                    return orig(self, *args, **kwargs)
                t0 = _sfi_gpu_now_ns()
                try:
                    return orig(self, *args, **kwargs)
                finally:
                    t1 = _sfi_gpu_now_ns()
                    _sfi_gpu_append({
                        "event": "gpu_execute_phase",
                        "phase": name,
                        "step": getattr(_sfi_gpu_phase_tls, "step", -1),
                        "duration_us": _sfi_gpu_us(t0, t1),
                    })

            _sfi_timed_method._sfi_gpu_phase_patch = True
            setattr(_sfi_gmr.GPUModelRunner, name, _sfi_timed_method)

        for _sfi_phase_name in (
            "_update_states",
            "_prepare_inputs",
            "_determine_batch_execution_and_padding",
            "_get_slot_mappings",
            "_build_attention_metadata",
            "_preprocess",
            "_model_forward",
        ):
            _sfi_wrap_gpu_runner_method(_sfi_phase_name)

        if not getattr(_sfi_gmr.GPUModelRunner.execute_model, "_sfi_gpu_phase_patch", False):
            _sfi_orig_gpu_execute_model = _sfi_gmr.GPUModelRunner.execute_model

            def _sfi_timed_gpu_execute_model(self, scheduler_output, *args, **kwargs):
                _sfi_gpu_patch_compute_logits(self)
                step = _sfi_gpu_next_step()
                prev_in_execute = getattr(_sfi_gpu_phase_tls, "in_execute_model", False)
                prev_step = getattr(_sfi_gpu_phase_tls, "step", -1)
                _sfi_gpu_phase_tls.in_execute_model = True
                _sfi_gpu_phase_tls.step = step
                t0 = _sfi_gpu_now_ns()
                result = None
                exc_type = None
                try:
                    result = _sfi_orig_gpu_execute_model(self, scheduler_output, *args, **kwargs)
                    return result
                except Exception as exc:
                    exc_type = type(exc).__name__
                    raise
                finally:
                    t1 = _sfi_gpu_now_ns()
                    payload = {
                        "event": "gpu_execute_model_outer",
                        "step": int(step),
                        "duration_us": _sfi_gpu_us(t0, t1),
                        "result_type": type(result).__name__ if exc_type is None else None,
                        "exception_type": exc_type,
                        "use_async_scheduling": bool(getattr(self, "use_async_scheduling", False)),
                        "execute_model_state_set": getattr(self, "execute_model_state", None) is not None,
                    }
                    payload.update(_sfi_gpu_scheduler_summary(scheduler_output))
                    _sfi_gpu_append(payload)
                    _sfi_gpu_phase_tls.in_execute_model = prev_in_execute
                    _sfi_gpu_phase_tls.step = prev_step

            _sfi_timed_gpu_execute_model._sfi_gpu_phase_patch = True
            _sfi_gmr.GPUModelRunner.execute_model = _sfi_timed_gpu_execute_model

        if site_log_enabled:
            with open("/tmp/vllm_sparse_site.log", "a", encoding="utf-8") as _log:
                _log.write(
                    "sitecustomize installed GPUModelRunner execute_model phase timing diagnostic, "
                    f"PID={os.getpid()}\n"
                )
    except Exception as exc:
        if site_log_enabled:
            with open("/tmp/vllm_sparse_site.log", "a", encoding="utf-8") as _log:
                _log.write(
                    "sitecustomize GPUModelRunner execute_model phase timing diagnostic failed, "
                    f"PID={os.getpid()}, error={exc}\n"
                )
        raise SystemExit(
            f"sitecustomize GPUModelRunner execute_model phase timing diagnostic failed: {exc}"
        ) from exc


# synchronize_input_prep phase timing diagnostic.
# Complements VLLM_GPU_EXECUTE_PHASE_TIMING_LOG by splitting the context-manager
# wait/record cost that surrounds GPUModelRunner.execute_model preprocessing.
if os.environ.get("VLLM_GPU_EXECUTE_PHASE_TIMING_LOG"):
    try:
        from contextlib import contextmanager as _sfi_sync_contextmanager
        from vllm.v1.worker import gpu_model_runner as _sfi_sync_gmr

        if "_sfi_gpu_append" in globals() and not getattr(
            _sfi_sync_gmr.GPUModelRunner.synchronize_input_prep,
            "_sfi_gpu_sync_phase_patch",
            False,
        ):
            _sfi_orig_sync_input_prep = _sfi_sync_gmr.GPUModelRunner.synchronize_input_prep

            @_sfi_sync_contextmanager
            def _sfi_timed_synchronize_input_prep(self):
                if not getattr(_sfi_gpu_phase_tls, "in_execute_model", False):
                    with _sfi_orig_sync_input_prep(self):
                        yield
                    return
                event = getattr(self, "prepare_inputs_event", None)
                if event is None:
                    _sfi_gpu_append({
                        "event": "gpu_execute_phase",
                        "phase": "synchronize_input_prep_wait",
                        "step": getattr(_sfi_gpu_phase_tls, "step", -1),
                        "duration_us": 0.0,
                        "event_present": False,
                    })
                    yield
                    return
                t_wait0 = _sfi_gpu_now_ns()
                event.synchronize()
                t_wait1 = _sfi_gpu_now_ns()
                _sfi_gpu_append({
                    "event": "gpu_execute_phase",
                    "phase": "synchronize_input_prep_wait",
                    "step": getattr(_sfi_gpu_phase_tls, "step", -1),
                    "duration_us": _sfi_gpu_us(t_wait0, t_wait1),
                    "event_present": True,
                })
                try:
                    yield
                finally:
                    t_rec0 = _sfi_gpu_now_ns()
                    event.record()
                    t_rec1 = _sfi_gpu_now_ns()
                    _sfi_gpu_append({
                        "event": "gpu_execute_phase",
                        "phase": "synchronize_input_prep_record",
                        "step": getattr(_sfi_gpu_phase_tls, "step", -1),
                        "duration_us": _sfi_gpu_us(t_rec0, t_rec1),
                        "event_present": True,
                    })

            _sfi_timed_synchronize_input_prep._sfi_gpu_sync_phase_patch = True
            _sfi_sync_gmr.GPUModelRunner.synchronize_input_prep = _sfi_timed_synchronize_input_prep
            if site_log_enabled:
                with open("/tmp/vllm_sparse_site.log", "a", encoding="utf-8") as _log:
                    _log.write(
                        "sitecustomize installed synchronize_input_prep phase timing diagnostic, "
                        f"PID={os.getpid()}\n"
                    )
    except Exception as exc:
        if site_log_enabled:
            with open("/tmp/vllm_sparse_site.log", "a", encoding="utf-8") as _log:
                _log.write(
                    "sitecustomize synchronize_input_prep phase timing diagnostic failed, "
                    f"PID={os.getpid()}, error={exc}\n"
                )
        raise SystemExit(
            f"sitecustomize synchronize_input_prep phase timing diagnostic failed: {exc}"
        ) from exc

if "VLLM_SPARSE_CONTROLLER_JSON" in os.environ:
    if site_log_enabled:
        with open("/tmp/vllm_sparse_site.log", "a", encoding="utf-8") as _log:
            _log.write(f"sitecustomize installing vllm sparse patch, PID={os.getpid()}\n")
    try:
        from patches.vllm_sparse_patch import ensure_vllm_sparse_patch_from_env
        controller = ensure_vllm_sparse_patch_from_env()
        if site_log_enabled:
            with open("/tmp/vllm_sparse_site.log", "a", encoding="utf-8") as _log:
                vllm_mod = sys.modules.get("vllm")
                vllm_file = getattr(vllm_mod, "__file__", None)
                sys_path_head = ",".join(str(p) for p in sys.path[:5])
                _log.write(
                    f"  patch installed successfully, controller={controller is not None}, PID={os.getpid()}, "
                    f"vllm_file={vllm_file}, sys_path_head={sys_path_head}\n"
                )
    except Exception as e:
        if site_log_enabled:
            with open("/tmp/vllm_sparse_site.log", "a", encoding="utf-8") as _log:
                _log.write(f"  patch FAILED: {e}, PID={os.getpid()}\n")
        import traceback
        if site_log_enabled:
            with open("/tmp/vllm_sparse_site.log", "a", encoding="utf-8") as _log:
                _log.write(traceback.format_exc())
        raise SystemExit(f"sitecustomize sparse patch failed: {e}") from e

# Optional deterministic / TF32 guardrails for debug runs.
# Enable via env to avoid impacting normal high-performance paths.
if os.environ.get("VLLM_FORCE_DETERMINISTIC") == "1":
    try:
        import torch

        # Disable TF32 for matmul/cudnn to reduce accum rounding drift.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        # Use full FP32 accumulation for float32 matmul paths.
        torch.set_float32_matmul_precision("highest")

        # Request deterministic cuBLAS kernels for GEMM paths.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    except Exception as exc:  # pragma: no cover - best-effort debug hook
        if site_log_enabled:
            with open("/tmp/vllm_sparse_site.log", "a", encoding="utf-8") as _log:
                _log.write(f"sitecustomize deterministic setup failed: {exc}\n")

# ---------------------------------------------------------------------------
