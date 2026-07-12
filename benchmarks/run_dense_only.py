from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

try:
    from benchmarks.decode_latency_metrics import (
        decode_step_durations_us,
        format_decode_metrics_line,
        summarize_decode_metrics,
    )
    from benchmarks.decode_throughput_window import (
        DecodeWindowMeter,
        count_new_tokens,
        pull_step_outputs_with_timing,
    )
    from benchmarks.prompt_batch_io import (
        load_prompt_batch,
        maybe_apply_chat_template,
        output_payload_from_generation,
    )
    from benchmarks.scheduler_contract import resolve_benchmark_max_num_seqs
    from benchmarks.vllm_profiler_scope import (
        build_vllm_torch_profiler_config,
        maybe_start_vllm_torch_profile,
        maybe_stop_vllm_torch_profile,
    )
except ModuleNotFoundError:
    from decode_latency_metrics import (  # type: ignore[no-redef]
        decode_step_durations_us,
        format_decode_metrics_line,
        summarize_decode_metrics,
    )
    from decode_throughput_window import (  # type: ignore[no-redef]
        DecodeWindowMeter,
        count_new_tokens,
        pull_step_outputs_with_timing,
    )
    from prompt_batch_io import (  # type: ignore[no-redef]
        load_prompt_batch,
        maybe_apply_chat_template,
        output_payload_from_generation,
    )
    from scheduler_contract import resolve_benchmark_max_num_seqs  # type: ignore[no-redef]
    from vllm_profiler_scope import (  # type: ignore[no-redef]
        build_vllm_torch_profiler_config,
        maybe_start_vllm_torch_profile,
        maybe_stop_vllm_torch_profile,
    )


def _final_outputs_with_decoded_text(
    final_outputs: dict[str, object],
    engine: object,
) -> dict[str, object]:
    """Re-decode text from cumulative token_ids before writing outputs.

    The step-driven decode loop stores per-step RequestOutput payloads whose
    ``text`` is the incremental (streaming) fragment — empty on the final
    step — while ``token_ids`` is cumulative. Same contract as the sparse
    runner's helper of the same name.
    """
    tokenizer = engine.get_tokenizer()  # type: ignore[attr-defined]
    decode = getattr(tokenizer, "decode", None)
    enriched: dict[str, object] = {}
    for rid, payload in final_outputs.items():
        if isinstance(payload, dict):
            raw_token_ids = payload.get("token_ids", [])
        else:
            raw_token_ids = payload
        token_ids = (
            [int(token_id) for token_id in raw_token_ids]
            if isinstance(raw_token_ids, list)
            else []
        )
        text = str(decode(token_ids)) if callable(decode) else ""
        enriched[str(rid)] = {
            "token_ids": token_ids,
            "text": text,
        }
    return enriched


def _install_transformers_register_shim() -> None:
    if os.environ.get("VLLM_IGNORE_DUPLICATE_TRANSFORMERS_CONFIGS", "1") != "1":
        return
    try:
        from transformers import AutoConfig  # type: ignore
    except Exception:
        return

    orig_register = AutoConfig.register

    def _safe_register(model_type, config, exist_ok: bool = False):  # type: ignore[no-redef]
        try:
            return orig_register(model_type, config, exist_ok=True)
        except ValueError:
            return None

    AutoConfig.register = _safe_register  # type: ignore[assignment]


def _full_cudagraph_compilation_config(
    capture_sizes_raw: str,
) -> dict[str, object]:
    try:
        from vllm.config import CompilationConfig  # pylint: disable=import-error
    except Exception:
        field_names: set[str] = set()
    else:
        field_names = set(
            getattr(CompilationConfig, "__dataclass_fields__", {}).keys()
        )

    if "cudagraph_mode" in field_names:
        config: dict[str, object] = {"cudagraph_mode": "FULL"}
    else:
        config = {"full_cuda_graph": True}

    if capture_sizes_raw:
        sizes = [
            int(x.strip())
            for x in str(capture_sizes_raw).split(",")
            if x.strip()
        ]
        config["cudagraph_capture_sizes"] = sizes
    return config


def _decode_run_config(args: argparse.Namespace, *, prompt_count: int) -> dict[str, object]:
    return {
        "runner": "run_dense_only.py",
        "model": str(args.model),
        "prompt": str(args.prompt),
        "prompt_count": int(prompt_count),
        "batch_size": int(args.batch_size),
        "max_num_seqs": _effective_max_num_seqs(args),
        "split_context_prompts": bool(args.split_context_prompts),
        "max_new_tokens": int(args.max_new_tokens),
        "respect_eos": bool(args.respect_eos),
        "ignore_eos": not bool(args.respect_eos),
        "measure_decode_latency": bool(args.measure_decode_latency),
        "warmup_runs": int(args.warmup_runs),
        "reset_prefix_cache": bool(args.reset_prefix_cache),
        "full_cuda_graph": bool(args.full_cuda_graph),
        "cudagraph_capture_sizes": str(args.cudagraph_capture_sizes),
        "disable_cascade_attn": bool(args.disable_cascade_attn),
        "gpu_mem_util": float(args.gpu_mem_util),
        "dtype": str(args.dtype),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "attention_backend": os.environ.get("VLLM_ATTENTION_BACKEND", ""),
        "flash_attn_version": os.environ.get("VLLM_FLASH_ATTN_VERSION", ""),
    }


def _engine_args_accepts(name: str) -> bool:
    try:
        from vllm.engine.arg_utils import EngineArgs  # pylint: disable=import-error
    except Exception:
        return True
    return name in getattr(EngineArgs, "__dataclass_fields__", {})


def _effective_max_num_seqs(args: argparse.Namespace) -> int:
    return resolve_benchmark_max_num_seqs(
        batch_size=int(args.batch_size),
        configured=int(getattr(args, "max_num_seqs", 0) or 0),
    )


def _scheduler_engine_kwargs(args: argparse.Namespace) -> dict[str, object]:
    return {"max_num_seqs": _effective_max_num_seqs(args)}


def _visible_gpus_all_have_nvlink() -> tuple[bool, str]:
    """[TP-CUSTOM-AR-PROBE] (ok, reason): ok iff every visible GPU reports an
    active NVLink (covers NVSwitch fabrics too). Handles unset
    CUDA_VISIBLE_DEVICES (probe all NVML devices) and UUID entries
    (nvmlDeviceGetHandleByUUID). Conservative: any doubt -> (False, why),
    keeping vLLM's custom all-reduce disabled; the caller logs the reason so
    NVLink boxes never lose custom AR silently."""
    try:
        import pynvml  # pylint: disable=import-error
        pynvml.nvmlInit()
    except Exception as exc:
        return False, f"NVML unavailable ({type(exc).__name__})"
    try:
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        handles = []
        if not cvd:
            count = int(pynvml.nvmlDeviceGetCount())
            if count <= 0:
                return False, "NVML reports no devices"
            handles = [
                (str(i), pynvml.nvmlDeviceGetHandleByIndex(i)) for i in range(count)
            ]
        else:
            for entry in cvd.split(","):
                entry = entry.strip()
                if not entry:
                    continue
                try:
                    if entry.isdigit():
                        handles.append(
                            (entry, pynvml.nvmlDeviceGetHandleByIndex(int(entry)))
                        )
                    else:  # UUID / MIG form
                        handles.append(
                            (
                                entry,
                                pynvml.nvmlDeviceGetHandleByUUID(entry.encode()),
                            )
                        )
                except Exception as exc:
                    return False, f"cannot resolve device {entry!r} ({type(exc).__name__})"
        if not handles:
            return False, "no visible devices"
        for name, handle in handles:
            active = False
            for link in range(18):
                try:
                    if pynvml.nvmlDeviceGetNvLinkState(handle, link):
                        active = True
                        break
                except pynvml.NVMLError:
                    break
            if not active:
                return False, f"GPU {name} has no active NVLink (PCIe-only)"
        return True, "active NVLink on all visible GPUs"
    except Exception as exc:
        return False, f"probe error ({type(exc).__name__})"
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def _install_dense_fa3_route_trace_probe() -> None:
    route_trace_log = os.environ.get("VLLM_SPARSE_FA3_ROUTE_TRACE_LOG", "")
    if not route_trace_log:
        return
    try:
        from patches.fa3_native.install import install_dense_fa3_route_trace_probe_patch
    except Exception:
        return
    install_dense_fa3_route_trace_probe_patch()


def _install_fa4_dense_fallback_gateway_if_requested() -> None:
    if os.environ.get("VLLM_FLASH_ATTN_VERSION") != "4":
        return
    try:
        from patches.patch_installer import install_fa4_dense_fallback_gateway
    except Exception:
        return
    install_fa4_dense_fallback_gateway()


def _setup_repo_imports() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    vllm_src = repo_root / "vllm"
    upstream_raw = os.environ.get("VLLM_SPARSE_FA3_UPSTREAM_ROOT", "").strip()
    fa_upstream = Path(upstream_raw) if upstream_raw else repo_root / "third_party_upstreams" / "vllm-project-flash-attention"
    pythonpath_parts: list[str] = []
    if (fa_upstream / "flash_attn").exists():
        pythonpath_parts.append(str(fa_upstream))
    if vllm_src.exists():
        pythonpath_parts.append(str(vllm_src))
    pythonpath_parts.append(str(repo_root))
    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    os.environ["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

    sys.path.insert(0, str(repo_root))
    if vllm_src.exists():
        sys.path.insert(0, str(vllm_src))
    if (fa_upstream / "flash_attn").exists():
        sys.path.insert(0, str(fa_upstream))
    return repo_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./qwen3-0.6b")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16"))
    parser.add_argument("--prompt", default="benchmarks/needle_prompt_part1.txt")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=0,
        help=(
            "vLLM scheduler concurrency cap (0 = --batch-size). Benchmark "
            "parents pass the request batch explicitly to keep dense/sparse "
            "engine capacity comparable."
        ),
    )
    parser.add_argument(
        "--split-context-prompts",
        action="store_true",
        help="Treat each Context: segment as one request instead of repeating the full file.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--respect-eos",
        action="store_true",
        help="Use vLLM default EOS stopping instead of forcing generation to max tokens.",
    )
    parser.add_argument(
        "--chat-template",
        action="store_true",
        help="Render prompts through the model chat template before generation.",
    )
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument(
        "--disable-cascade-attn",
        action="store_true",
        help="Disable vLLM V1 cascade attention (useful to avoid common-prefix optimization masking sparse benefits).",
    )
    parser.add_argument(
        "--measure-decode-latency",
        action="store_true",
        help="Report decode-only latency using vLLM request metrics (excludes prefill).",
    )
    parser.add_argument(
        "--decode-metrics-json",
        type=str,
        default="",
        help=(
            "Optional output path for decode metrics JSON "
            "(decode_p50_us/decode_p95_us/decode_p99_us/decode_tokens_per_s)."
        ),
    )
    parser.add_argument(
        "--outputs-json",
        type=str,
        default="",
        help="Optional output path for final generated token ids.",
    )
    parser.add_argument(
        "--outputs-include-text",
        action="store_true",
        help="Write text plus token_ids in --outputs-json for semantic checks.",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Number of warmup runs before measurement (only effective with --measure-decode-latency).",
    )
    parser.add_argument(
        "--reset-prefix-cache",
        action="store_true",
        help="Reset vLLM prefix cache before each generate call to avoid cross-run reuse skewing timings.",
    )
    parser.add_argument("--gpu-mem-util", type=float, default=0.6)
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=0,
        help=(
            "Optional engine max_model_len override (0 = model default). Needed for "
            "models whose max_position_embeddings exceeds the GPU KV-cache capacity "
            "(e.g. Qwen3-4B 262144 on A100-40GB)."
        ),
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="vLLM tensor_parallel_size (multi-GPU; pass matching --cuda-visible-devices).",
    )
    parser.add_argument(
        "--enable-custom-all-reduce",
        action="store_true",
        help=(
            "Re-enable vLLM's custom all-reduce for TP>1 (NVLink boxes). Default "
            "keeps it disabled: on PCIe-only topologies it crashes worker init "
            "with 'custom_all_reduce.cuh ... invalid argument'."
        ),
    )
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--full-cuda-graph", action="store_true")
    parser.add_argument("--max-seq-len-to-capture", type=int, default=16384)
    parser.add_argument(
        "--cudagraph-capture-sizes",
        type=str,
        default="",
        help="Optional comma-separated cudagraph capture sizes override, e.g. '1,2,4,8'.",
    )
    args = parser.parse_args()
    if int(args.batch_size) <= 0:
        parser.error("--batch-size must be > 0")
    try:
        _effective_max_num_seqs(args)
    except ValueError as exc:
        parser.error(str(exc))

    repo_root = _setup_repo_imports()
    _install_transformers_register_shim()

    from vllm import LLM, SamplingParams  # pylint: disable=import-error

    _install_dense_fa3_route_trace_probe()

    # Dense reference runs use the current FA3 route by default. Keep this
    # explicit so stale shell env cannot reopen the retired Triton backend path,
    # while allowing the SM100 parent runner to request FA4.
    requested_flash_attn_version = os.environ.get("VLLM_FLASH_ATTN_VERSION", "3")
    if requested_flash_attn_version not in {"3", "4"}:
        requested_flash_attn_version = "3"
    os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN_VLLM_V1"
    os.environ["VLLM_FLASH_ATTN_VERSION"] = requested_flash_attn_version
    if requested_flash_attn_version == "4":
        os.environ["VLLM_SPARSE_FA4_DENSE_GATEWAY"] = "1"
    else:
        os.environ.pop("VLLM_SPARSE_FA4_DENSE_GATEWAY", None)
    _install_fa4_dense_fallback_gateway_if_requested()

    compilation_config = (
        _full_cudagraph_compilation_config(args.cudagraph_capture_sizes)
        if bool(args.full_cuda_graph)
        else None
    )
    engine_kwargs: dict[str, object] = {
        "model": str(Path(args.model)),
        "dtype": args.dtype,
        "enforce_eager": bool(args.enforce_eager),
        "gpu_memory_utilization": float(args.gpu_mem_util),
        "tensor_parallel_size": max(1, int(getattr(args, "tensor_parallel_size", 1) or 1)),
        "compilation_config": compilation_config,
        "disable_cascade_attn": bool(args.disable_cascade_attn),
    }
    engine_kwargs.update(_scheduler_engine_kwargs(args))
    if int(engine_kwargs["tensor_parallel_size"]) > 1 and (
        os.environ.get("VLLM_SPARSE_FORCE_DISABLE_CUSTOM_AR", "") == "1"
    ):
        # [TP-CUSTOM-AR-ESCAPE] see run_sparse_only.py: operator override for
        # boxes where custom AR + FULL cudagraph crashes despite NVLink probe
        # OK. Mirrored on the dense side so A/B comparisons stay symmetric.
        engine_kwargs["disable_custom_all_reduce"] = True
        print(
            "[run_dense_only] TP>1: custom all-reduce force-disabled via "
            "VLLM_SPARSE_FORCE_DISABLE_CUSTOM_AR=1",
            flush=True,
        )
    elif int(engine_kwargs["tensor_parallel_size"]) > 1 and not bool(
        getattr(args, "enable_custom_all_reduce", False)
    ):
        # [TP-CUSTOM-AR-PROBE] see run_sparse_only.py: enable custom all-reduce
        # automatically on NVLink boxes, keep it disabled on PCIe-only ones.
        _nvlink_ok, _nvlink_reason = _visible_gpus_all_have_nvlink()
        if _nvlink_ok:
            print(
                "[run_dense_only] TP>1: %s; leaving vLLM custom all-reduce "
                "enabled" % _nvlink_reason,
                flush=True,
            )
        else:
            engine_kwargs["disable_custom_all_reduce"] = True
            print(
                "[run_dense_only] TP>1: disabling vLLM custom all-reduce "
                "(%s); pass --enable-custom-all-reduce to force it on"
                % _nvlink_reason,
                flush=True,
            )
    if int(getattr(args, "max_model_len", 0) or 0) > 0:
        engine_kwargs["max_model_len"] = int(args.max_model_len)
    # Shared-GPU workaround: fix KV cache size via env to skip vLLM memory profiling.
    _kvb = os.environ.get("VLLM_KV_CACHE_MEMORY_BYTES")
    if _kvb:
        engine_kwargs["kv_cache_memory_bytes"] = int(_kvb)
    if _engine_args_accepts("attention_config"):
        engine_kwargs["attention_config"] = {
            "backend": "FLASH_ATTN",
            "flash_attn_version": int(requested_flash_attn_version),
        }
    if _engine_args_accepts("max_seq_len_to_capture"):
        engine_kwargs["max_seq_len_to_capture"] = int(args.max_seq_len_to_capture)
    profiler_config = build_vllm_torch_profiler_config(_engine_args_accepts)
    if profiler_config is not None:
        engine_kwargs["profiler_config"] = profiler_config
    engine = LLM(**engine_kwargs)
    sp = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=int(args.max_new_tokens),
        ignore_eos=not bool(args.respect_eos),
        # [C4-DETOKENIZE 2026-07-06] speed 路径只消费 token_ids（text 结束后
        # 统一重解码），关闭前端每步增量 detokenize；非测速分支直读
        # output.text，保持默认——分支限定（与 sparse runner 同款，配对公平）。
        detokenize=not bool(args.measure_decode_latency),
    )
    prompt_path = Path(args.prompt)
    if not prompt_path.is_absolute():
        prompt_path = repo_root / prompt_path
    prompts = load_prompt_batch(
        prompt_path,
        batch_size=int(args.batch_size),
        split_context_prompts=bool(args.split_context_prompts),
    )
    prompts = maybe_apply_chat_template(
        engine,
        prompts,
        use_chat_template=bool(args.chat_template),
        enable_thinking=bool(args.enable_thinking),
    )
    final_outputs: dict[str, object] = {}

    def _reset_prefix_cache() -> None:
        try:
            engine.llm_engine.reset_prefix_cache()  # type: ignore[attr-defined]
        except Exception:
            return

    def _generate_with_engine_step() -> tuple[float, float, int, int, list[float], float, float, dict[str, object]]:
        # Use cumulative outputs, track per-request token deltas.
        request_ids = [f"bench-{i}-{time.time_ns()}" for i in range(len(prompts))]
        t_add0 = time.perf_counter()
        for rid, p in zip(request_ids, prompts):
            engine.llm_engine.add_request(rid, p, sp)  # type: ignore[attr-defined]
        t_add1 = time.perf_counter()

        prev_len: dict[str, int] = {}
        # batch_size 使能 all-decode 稳态窗（与 run_sparse_only 同款，配对公平）：
        # 排除 chunked prefill 交错段后的真实稳态 decode 吞吐。
        decode_meter = DecodeWindowMeter(batch_size=len(prompts))
        out_tokens = 0
        total_engine_steps = 0
        first_emit_step_index = -1
        pre_first_emit_step_wall_s: list[float] = []
        first_emit_step_wall_s = 0.0
        step_wall_s_first_16: list[float] = []
        step_get_output_us_first_16: list[float] = []
        step_process_outputs_us_first_16: list[float] = []
        step_abort_requests_us_first_16: list[float] = []
        step_dummy_batch_us_first_16: list[float] = []
        step_new_tokens_first_16: list[int] = []
        capture_all_step_timing = str(
            os.environ.get("VLLM_DECODE_FULL_STEP_TIMING", "") or ""
        ).strip().lower() not in {"", "0", "false", "off", "no"}
        step_wall_s_all: list[float] = []
        step_get_output_us_all: list[float] = []
        step_process_outputs_us_all: list[float] = []
        step_abort_requests_us_all: list[float] = []
        step_dummy_batch_us_all: list[float] = []
        step_new_tokens_all: list[int] = []
        step_engine_core_timing_all: list[dict[str, float]] = []
        step_host_begin_ns_all: list[int] = []
        step_host_end_ns_all: list[int] = []

        t_total0 = time.perf_counter()
        while engine.llm_engine.has_unfinished_requests():  # type: ignore[attr-defined]
            step_host_begin_ns = time.time_ns()
            t_step0 = time.perf_counter()
            step_out, core_ts, step_timing = pull_step_outputs_with_timing(
                engine.llm_engine  # type: ignore[attr-defined]
            )
            t_step1 = time.perf_counter()
            step_host_end_ns = time.time_ns()
            step_wall_s = t_step1 - t_step0
            step_new_tokens = count_new_tokens(step_out, prev_len)
            if total_engine_steps < 16:
                step_wall_s_first_16.append(float(step_wall_s))
                step_get_output_us_first_16.append(
                    float(step_timing.get("get_output_us", 0.0))
                )
                step_process_outputs_us_first_16.append(
                    float(step_timing.get("process_outputs_us", 0.0))
                )
                step_abort_requests_us_first_16.append(
                    float(step_timing.get("abort_requests_us", 0.0))
                )
                step_dummy_batch_us_first_16.append(
                    float(step_timing.get("dummy_batch_us", 0.0))
                )
                step_new_tokens_first_16.append(int(step_new_tokens))
            if capture_all_step_timing:
                step_wall_s_all.append(float(step_wall_s))
                step_get_output_us_all.append(
                    float(step_timing.get("get_output_us", 0.0))
                )
                step_process_outputs_us_all.append(
                    float(step_timing.get("process_outputs_us", 0.0))
                )
                step_abort_requests_us_all.append(
                    float(step_timing.get("abort_requests_us", 0.0))
                )
                step_dummy_batch_us_all.append(
                    float(step_timing.get("dummy_batch_us", 0.0))
                )
                step_new_tokens_all.append(int(step_new_tokens))
                step_host_begin_ns_all.append(int(step_host_begin_ns))
                step_host_end_ns_all.append(int(step_host_end_ns))
                step_engine_core_timing_all.append(
                    {
                        str(key): float(value)
                        for key, value in step_timing.items()
                        if str(key).startswith("ec_")
                    }
                )
            if args.outputs_json:
                for item in step_out:
                    rid = getattr(item, "request_id", None)
                    outputs = getattr(item, "outputs", None)
                    if rid is None or not outputs:
                        continue
                    final_outputs[str(rid)] = output_payload_from_generation(
                        item,
                        include_text=bool(args.outputs_include_text),
                    )

            if step_new_tokens > 0:
                if first_emit_step_index < 0:
                    first_emit_step_index = int(total_engine_steps)
                    first_emit_step_wall_s = float(step_wall_s)
                out_tokens += step_new_tokens
                decode_meter.observe(core_ts, step_new_tokens)
            elif first_emit_step_index < 0:
                pre_first_emit_step_wall_s.append(float(step_wall_s))
            total_engine_steps += 1
        t_total1 = time.perf_counter()
        decode_elapsed, decode_tokens, _decode_tps, decode_step_durations_s = decode_meter.finalize()
        ad_elapsed_s, ad_tokens, ad_tps, ad_steps = decode_meter.finalize_all_decode()
        first_emit_delay_s, post_decode_tail_s = decode_meter.boundary_delays(
            t_total0,
            t_total1,
        )
        boundary_diagnostics = {
            "all_decode_elapsed_s": float(ad_elapsed_s),
            "all_decode_tokens": int(ad_tokens),
            "all_decode_tok_per_s": float(ad_tps),
            "all_decode_steps": int(ad_steps),
            "process_pid": int(os.getpid()),
            "engine_core_step_log": str(
                os.environ.get("VLLM_DECODE_ENGINE_CORE_STEP_LOG", "") or ""
            ),
            "request_add_s": float(t_add1 - t_add0),
            "total_engine_steps": int(total_engine_steps),
            "pre_first_emit_step_count": int(max(0, first_emit_step_index)),
            "first_emit_step_index": int(first_emit_step_index),
            "pre_first_emit_step_wall_us": [
                float(x * 1_000_000.0) for x in pre_first_emit_step_wall_s
            ],
            "first_emit_step_wall_us": float(first_emit_step_wall_s * 1_000_000.0),
            "step_wall_us_first_16": [
                float(x * 1_000_000.0) for x in step_wall_s_first_16
            ],
            "step_get_output_us_first_16": step_get_output_us_first_16,
            "step_process_outputs_us_first_16": step_process_outputs_us_first_16,
            "step_abort_requests_us_first_16": step_abort_requests_us_first_16,
            "step_dummy_batch_us_first_16": step_dummy_batch_us_first_16,
            "step_new_tokens_first_16": step_new_tokens_first_16,
        }
        if capture_all_step_timing:
            boundary_diagnostics.update(
                {
                    "step_timing_scope": "all_engine_steps",
                    "step_wall_us_all": [
                        float(x * 1_000_000.0) for x in step_wall_s_all
                    ],
                    "step_get_output_us_all": step_get_output_us_all,
                    "step_process_outputs_us_all": step_process_outputs_us_all,
                    "step_abort_requests_us_all": step_abort_requests_us_all,
                    "step_dummy_batch_us_all": step_dummy_batch_us_all,
                    "step_new_tokens_all": step_new_tokens_all,
                    "step_host_begin_ns_all": step_host_begin_ns_all,
                    "step_host_end_ns_all": step_host_end_ns_all,
                    "step_engine_core_timing_all": step_engine_core_timing_all,
                }
            )
        return (t_total1 - t_total0, decode_elapsed, int(out_tokens),
                int(decode_tokens), decode_step_durations_s,
                first_emit_delay_s, post_decode_tail_s, boundary_diagnostics)

    if bool(args.reset_prefix_cache):
        _reset_prefix_cache()
    if bool(args.measure_decode_latency):
        for _ in range(max(0, int(args.warmup_runs))):
            _generate_with_engine_step()
    else:
        engine.generate(prompts, sp)
    final_outputs.clear()
    if bool(args.reset_prefix_cache):
        _reset_prefix_cache()
    if bool(args.measure_decode_latency):
        profiling = maybe_start_vllm_torch_profile(engine)
        try:
            (
                elapsed,
                decode_elapsed,
                out_tokens,
                decode_tokens,
                decode_step_durations_s,
                first_emit_delay_s,
                post_decode_tail_s,
                boundary_diagnostics,
            ) = _generate_with_engine_step()
        finally:
            maybe_stop_vllm_torch_profile(engine, profiling)
        tps = float(out_tokens) / elapsed if elapsed > 0 else float("nan")
        decode_tps = float(decode_tokens) / decode_elapsed if decode_elapsed > 0 else float("nan")
        decode_metrics = summarize_decode_metrics(
            decode_step_durations_s=decode_step_durations_s,
            decode_tokens=int(decode_tokens),
            decode_elapsed_s=float(decode_elapsed),
        )
        print(
            f"elapsed_s={elapsed:.6f} out_tokens={out_tokens} tok_per_s={tps:.2f} "
            f"decode_tokens={decode_tokens} decode_elapsed_s={decode_elapsed:.6f} "
            f"decode_tok_per_s={decode_tps:.2f}",
            flush=True,
        )
        print(
            "all_decode(steady window, prefill-interleave excluded): "
            f"tok_per_s={float(boundary_diagnostics.get('all_decode_tok_per_s', float('nan'))):.2f} "
            f"tokens={int(boundary_diagnostics.get('all_decode_tokens', 0))} "
            f"elapsed_s={float(boundary_diagnostics.get('all_decode_elapsed_s', 0.0)):.6f} "
            f"steps={int(boundary_diagnostics.get('all_decode_steps', 0))}",
            flush=True,
        )
        print(format_decode_metrics_line(decode_metrics), flush=True)
        if args.decode_metrics_json:
            run_config = _decode_run_config(args, prompt_count=len(prompts))
            metrics_payload = {
                "elapsed_s": float(elapsed),
                "process_pid": int(os.getpid()),
                "runner": run_config["runner"],
                "batch_size": run_config["batch_size"],
                "max_new_tokens": run_config["max_new_tokens"],
                "split_context_prompts": run_config["split_context_prompts"],
                "respect_eos": run_config["respect_eos"],
                "run_config": run_config,
                "out_tokens": int(out_tokens),
                "tok_per_s": float(tps),
                "decode_tokens": int(decode_tokens),
                "decode_elapsed_s": float(decode_elapsed),
                "decode_tok_per_s": float(decode_tps),
                "all_decode_tok_per_s": float(
                    boundary_diagnostics.get("all_decode_tok_per_s", float("nan"))
                ),
                "all_decode_tokens": int(
                    boundary_diagnostics.get("all_decode_tokens", 0)
                ),
                "all_decode_elapsed_s": float(
                    boundary_diagnostics.get("all_decode_elapsed_s", 0.0)
                ),
                "first_emit_delay_s": float(first_emit_delay_s),
                "post_decode_tail_s": float(post_decode_tail_s),
                "front_overhead_s": float(max(0.0, elapsed - decode_elapsed)),
                "decode_step_durations_us": decode_step_durations_us(
                    decode_step_durations_s
                ),
                "boundary_diagnostics": boundary_diagnostics,
            }
            metrics_payload.update(decode_metrics)
            Path(args.decode_metrics_json).write_text(
                json.dumps(metrics_payload, ensure_ascii=True, indent=2),
                encoding="utf-8",
            )
        if args.outputs_json:
            outputs_payload = (
                _final_outputs_with_decoded_text(final_outputs, engine)
                if bool(args.outputs_include_text)
                else final_outputs
            )
            Path(args.outputs_json).write_text(
                json.dumps(outputs_payload, ensure_ascii=True, indent=2, sort_keys=True),
                encoding="utf-8",
            )
    else:
        profiling = maybe_start_vllm_torch_profile(engine)
        t0 = time.perf_counter()
        try:
            out = engine.generate(prompts, sp)
            t1 = time.perf_counter()
        finally:
            maybe_stop_vllm_torch_profile(engine, profiling)
        out_tokens = sum(len(item.outputs[0].token_ids) for item in out)
        elapsed = t1 - t0
        tps = float(out_tokens) / elapsed if elapsed > 0 else float("nan")
        print(f"elapsed_s={elapsed:.6f} out_tokens={out_tokens} tok_per_s={tps:.2f}", flush=True)
        if args.outputs_json:
            payload = {
                str(getattr(item, "request_id", idx)): output_payload_from_generation(
                    item,
                    include_text=bool(args.outputs_include_text),
                )
                for idx, item in enumerate(out)
            }
            Path(args.outputs_json).write_text(
                json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True),
                encoding="utf-8",
            )


if __name__ == "__main__":
    main()
