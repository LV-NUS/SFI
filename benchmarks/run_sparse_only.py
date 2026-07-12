from __future__ import annotations

import argparse
import json
import os
import statistics
import struct
import sys
import time
from pathlib import Path

ACTIVE_SM80_GT1_RUNNER = "bench_sm80_mixed_page_one_shot_graph_e2e.py"
FA3_ROUTE_COUNTER_MMAP_BYTES = 8 * 8


def _fa3_interface_module() -> object | None:
    try:
        from patches.fa3_native import install as fa3_install

        if hasattr(fa3_install, "get_sparse_fa3_route_counters"):
            return fa3_install
    except Exception:
        pass
    for name in (
        "vllm_flash_attn.flash_attn_interface",
        "flash_attn.flash_attn_interface",
    ):
        module = sys.modules.get(name)
        if module is not None:
            return module
    return None


def _reset_fa3_route_counters_for_measurement() -> None:
    module = _fa3_interface_module()
    reset = getattr(module, "reset_sparse_fa3_route_counters", None)
    if callable(reset):
        reset()
    raw_path = os.environ.get("VLLM_SPARSE_FA3_ROUTE_COUNTER_MMAP", "")
    if raw_path:
        _prepare_fa3_route_counter_mmap(Path(raw_path))


def _prepare_fa3_route_counter_mmap(metrics_or_counter_path: str | Path) -> Path:
    path = Path(metrics_or_counter_path)
    if path.suffix != ".bin":
        path = path.with_name(path.name + ".route_counters.bin")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00" * FA3_ROUTE_COUNTER_MMAP_BYTES)
    return path


def _shared_fa3_route_counter_snapshot(path: str | Path) -> dict[str, object]:
    counter_path = Path(path)
    if not counter_path.exists():
        return {}
    data = counter_path.read_bytes()
    if len(data) < FA3_ROUTE_COUNTER_MMAP_BYTES:
        return {}
    values = struct.unpack_from("8q", data, 0)
    return {
        "actual_fwd_mixed_page_count": int(values[0]),
        "resolved_row_ptr_fwd_mixed_page_count": int(values[1]),
        "has_resolved_row_ptr_count": int(values[2]),
        "page_resolver_kind_counts": {
            "0": int(values[3]),
            "1": int(values[4]),
            "2": int(values[5]),
            "3": int(values[6]),
            "4": int(values[7]),
        },
    }


def _fa3_route_counter_snapshot() -> dict[str, object]:
    shared_path = os.environ.get("VLLM_SPARSE_FA3_ROUTE_COUNTER_MMAP", "")
    if shared_path:
        shared = _shared_fa3_route_counter_snapshot(shared_path)
        if int(shared.get("actual_fwd_mixed_page_count", 0) or 0) > 0:
            return shared
    module = _fa3_interface_module()
    get_counters = getattr(module, "get_sparse_fa3_route_counters", None)
    if not callable(get_counters):
        return {}
    try:
        counters = get_counters(reset=False)
    except Exception:
        return {}
    return dict(counters) if isinstance(counters, dict) else {}


def _route_kind_count(counters: dict[str, object], kind: int) -> int:
    raw_counts = counters.get("page_resolver_kind_counts", {})
    if not isinstance(raw_counts, dict):
        return 0
    return int(raw_counts.get(kind, raw_counts.get(str(kind), 0)) or 0)


def _fa3_route_counter_metrics(counters: dict[str, object]) -> dict[str, object]:
    actual = int(counters.get("actual_fwd_mixed_page_count", 0) or 0)
    resolved = int(counters.get("resolved_row_ptr_fwd_mixed_page_count", 0) or 0)
    has_resolved = int(counters.get("has_resolved_row_ptr_count", 0) or 0)
    kind0 = _route_kind_count(counters, 0)
    kind1 = _route_kind_count(counters, 1)
    kind2 = _route_kind_count(counters, 2)
    kind3 = _route_kind_count(counters, 3)
    kind4 = _route_kind_count(counters, 4)
    available = bool(counters and actual > 0)
    payload: dict[str, object] = {
        "speed_child_route_counter_scope": "measurement_window",
        "speed_child_route_counter_available": available,
        "speed_child_actual_fwd_mixed_page_count": actual,
        "speed_child_resolved_row_ptr_fwd_mixed_page_count": resolved,
        "speed_child_has_resolved_row_ptr_count": has_resolved,
        "speed_child_page_resolver_kind0_count": kind0,
        "speed_child_page_resolver_kind1_count": kind1,
        "speed_child_page_resolver_kind2_count": kind2,
        "speed_child_page_resolver_kind3_count": kind3,
        "speed_child_page_resolver_kind4_count": kind4,
    }
    if available:
        payload.update(
            {
                "actual_fwd_mixed_page_count": actual,
                "resolved_row_ptr_fwd_mixed_page_count": resolved,
                "page_resolver_kind4_count": kind4,
                "kind2_dispatch_count": kind2,
                "kind3_dispatch_count": kind3,
                "selected_table_publish_count": kind1,
            }
        )
    return payload


def _final_outputs_with_decoded_text(
    final_outputs: dict[str, object],
    engine: object,
) -> dict[str, object]:
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
    from vllm_profiler_scope import (  # type: ignore[no-redef]
        build_vllm_torch_profiler_config,
        maybe_start_vllm_torch_profile,
        maybe_stop_vllm_torch_profile,
    )


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


def _truthy_payload_value(value: object, *, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return bool(value)
    return str(value).strip().lower() not in {"", "0", "false", "off", "no"}


def _controller_payload_requests_sm80_gt1_final_path(
    payload: dict[str, object] | None,
) -> bool:
    if not isinstance(payload, dict):
        return False
    if not _truthy_payload_value(payload.get("enabled", True), default=True):
        return False
    try:
        prefill_last_n = int(payload.get("prefill_last_n_query", 0) or 0)
    except Exception:
        prefill_last_n = 0
    if prefill_last_n <= 1:
        return False
    try:
        refresh_interval = int(payload.get("refresh_interval", 0) or 0)
    except Exception:
        refresh_interval = 0
    trigger = payload.get("trigger")
    if refresh_interval <= 0 and isinstance(trigger, dict):
        try:
            refresh_interval = int(trigger.get("refresh_interval", 0) or 0)
        except Exception:
            refresh_interval = 0
    return refresh_interval > 0 and bool(payload.get("one_shot_bootstrap_only", False))


def _enforce_sm80_gt1_runner_contract(
    *,
    args: argparse.Namespace,
    sparse_payload: dict[str, object] | None,
) -> None:
    if not bool(getattr(args, "full_cuda_graph", False)):
        return
    if not bool(getattr(args, "measure_decode_latency", False)):
        return
    if not _controller_payload_requests_sm80_gt1_final_path(sparse_payload):
        return
    active_runner = str(os.environ.get("VLLM_SPARSE_ACTIVE_BENCH_RUNNER", "") or "")
    if active_runner == ACTIVE_SM80_GT1_RUNNER:
        return
    if os.environ.get("VLLM_SPARSE_ALLOW_DIRECT_GT1_WORKER", "0") == "1":
        print(
            "[warn] run_sparse_only.py is worker/debug only for SM80 GT1 "
            f"full-cudagraph; final perf runner is {ACTIVE_SM80_GT1_RUNNER}.",
            file=sys.stderr,
            flush=True,
        )
        return
    print(
        "[error] stale runner boundary: run_sparse_only.py must not be used as "
        "the final SM80 GT1 full-cudagraph benchmark runner. Use "
        f"benchmarks/{ACTIVE_SM80_GT1_RUNNER}, which sets "
        "VLLM_SPARSE_ACTIVE_BENCH_RUNNER for its worker child.",
        file=sys.stderr,
        flush=True,
    )
    raise SystemExit(2)


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
        "runner": "run_sparse_only.py",
        "runner_scope": "worker_child_not_sm80_gt1_final_runner",
        "active_bench_runner": os.environ.get("VLLM_SPARSE_ACTIVE_BENCH_RUNNER", ""),
        "model": str(args.model),
        "prompt": str(args.prompt),
        "prompt_count": int(prompt_count),
        "batch_size": int(args.batch_size),
        "split_context_prompts": bool(args.split_context_prompts),
        "max_new_tokens": int(args.max_new_tokens),
        "mix_max_new_tokens": (
            None
            if args.mix_max_new_tokens is None
            else int(args.mix_max_new_tokens)
        ),
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
        "async_refresh": os.environ.get("VLLM_SPARSE_ASYNC_REFRESH", ""),
        "attention_in_cudagraph": os.environ.get(
            "VLLM_SPARSE_ATTENTION_IN_CUDAGRAPH",
            "",
        ),
        "full_cudagraph_replay_refresh_batched_flush": os.environ.get(
            "VLLM_SPARSE_FULL_CUDAGRAPH_REPLAY_REFRESH_BATCHED_FLUSH",
            "",
        ),
        "full_cudagraph_replay_refresh_defer_to_deadline": os.environ.get(
            "VLLM_SPARSE_FULL_CUDAGRAPH_REPLAY_REFRESH_DEFER_TO_DEADLINE",
            "",
        ),
        "one_shot_async_bootstrap": os.environ.get(
            "VLLM_SPARSE_ONE_SHOT_ASYNC_BOOTSTRAP", ""
        ),
        "defer_bootstrap_producer": os.environ.get(
            "VLLM_SPARSE_DEFER_BOOTSTRAP_PRODUCER", ""
        ),
        "bootstrap_bridge_max_tokens": os.environ.get(
            "VLLM_SPARSE_BOOTSTRAP_BRIDGE_MAX_TOKENS", ""
        ),
        "deferred_producer_groups_per_step": os.environ.get(
            "VLLM_SPARSE_DEFERRED_PRODUCER_GROUPS_PER_STEP",
            "",
        ),
        "controller_json_present": bool(os.environ.get("VLLM_SPARSE_CONTROLLER_JSON")),
    }


def _engine_args_accepts(name: str) -> bool:
    try:
        from vllm.engine.arg_utils import EngineArgs  # pylint: disable=import-error
    except Exception:
        return True
    return name in getattr(EngineArgs, "__dataclass_fields__", {})


def _sync_scheduling_requested(args) -> bool:
    if bool(getattr(args, "sync_scheduling", False)):
        return True
    return str(os.environ.get("VLLM_SPARSE_SYNC_SCHEDULING", "")).strip().lower() in (
        "1",
        "true",
        "yes",
    )


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


DEFERRED_BRIDGE_ENV_KEYS = (
    "VLLM_SPARSE_DEFER_BOOTSTRAP_PRODUCER",
    "VLLM_SPARSE_BOOTSTRAP_DENSE_BRIDGE",
    "VLLM_SPARSE_DEFERRED_BRIDGE_DIAGNOSTIC",
    "VLLM_SPARSE_BOOTSTRAP_BRIDGE_MAX_TOKENS",
    "VLLM_SPARSE_BOOTSTRAP_BRIDGE_GRAPH_POLICY",
    "VLLM_SPARSE_DEFERRED_PRODUCER_GROUPS_PER_STEP",
)
DEFERRED_BRIDGE_GRAPH_POLICIES = ("evict_recapture_once",)


def add_deferred_bridge_arguments(parser: argparse.ArgumentParser) -> None:
    bootstrap_producer_group = parser.add_mutually_exclusive_group()
    bootstrap_producer_group.add_argument(
        "--defer-bootstrap-producer",
        dest="defer_bootstrap_producer",
        action="store_true",
        default=True,
    )
    bootstrap_producer_group.add_argument(
        "--no-defer-bootstrap-producer",
        dest="defer_bootstrap_producer",
        action="store_false",
    )
    parser.add_argument("--bootstrap-bridge-max-tokens", type=int, default=3)
    parser.add_argument("--deferred-producer-groups-per-step", type=int, default=-1)
    parser.add_argument(
        "--bootstrap-bridge-graph-policy",
        choices=DEFERRED_BRIDGE_GRAPH_POLICIES,
        default="evict_recapture_once",
    )


def validate_deferred_bridge_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    if int(args.bootstrap_bridge_max_tokens) <= 0:
        parser.error("--bootstrap-bridge-max-tokens must be > 0")
    if int(args.deferred_producer_groups_per_step) < -1:
        parser.error("--deferred-producer-groups-per-step must be >= -1")


def apply_deferred_bridge_env(args: argparse.Namespace, env) -> None:
    for key in DEFERRED_BRIDGE_ENV_KEYS:
        env.pop(key, None)
    if not bool(getattr(args, "defer_bootstrap_producer", False)):
        return
    env["VLLM_SPARSE_DEFER_BOOTSTRAP_PRODUCER"] = "1"
    env["VLLM_SPARSE_BOOTSTRAP_BRIDGE_MAX_TOKENS"] = str(
        int(args.bootstrap_bridge_max_tokens)
    )
    env["VLLM_SPARSE_BOOTSTRAP_BRIDGE_GRAPH_POLICY"] = str(
        args.bootstrap_bridge_graph_policy
    )
    groups_per_step = effective_deferred_producer_groups_per_step(args)
    if groups_per_step != 0:
        env["VLLM_SPARSE_DEFERRED_PRODUCER_GROUPS_PER_STEP"] = str(groups_per_step)


def effective_deferred_producer_groups_per_step(args: argparse.Namespace) -> int:
    raw = int(getattr(args, "deferred_producer_groups_per_step", -1))
    if raw >= 0:
        return raw
    return -1 if bool(getattr(args, "defer_bootstrap_producer", False)) else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./qwen3-0.6b")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16"))
    parser.add_argument("--prompt", default="benchmarks/needle_prompt_part1.txt")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--split-context-prompts",
        action="store_true",
        help="Treat each Context: segment as one request instead of repeating the full file.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--mix-max-new-tokens",
        type=int,
        default=None,
        help="Debug-only: assign a smaller max_new_tokens to half of the requests to simulate early finishes and slot reuse.",
    )
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
    parser.add_argument("--gpu-mem-util", type=float, default=0.9)
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
        "--sync-scheduling",
        action="store_true",
        help=(
            "Force synchronous vLLM scheduling (any TP size). TP>1 runs async "
            "scheduling by default via [TP-ASYNC-HARVEST]; env "
            "VLLM_SPARSE_SYNC_SCHEDULING=1 has the same effect (handy when "
            "driving this runner through the bench)."
        ),
    )
    parser.add_argument(
        "--enable-custom-all-reduce",
        action="store_true",
        help=(
            "Re-enable vLLM's custom all-reduce for TP>1 (NVLink boxes). Default "
            "auto-probes NVLink and keeps it disabled on PCIe-only topologies, "
            "where it crashes worker init with 'custom_all_reduce.cuh ... "
            "invalid argument'."
        ),
    )
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--full-cuda-graph", action="store_true")
    parser.add_argument("--max-seq-len-to-capture", type=int, default=32768)
    parser.add_argument(
        "--cudagraph-capture-sizes",
        type=str,
        default="",
        help="Optional comma-separated cudagraph capture sizes override, e.g. '1,2,4,8'.",
    )

    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--k-min", type=int, default=32)
    parser.add_argument("--k-max", type=str, default="none")
    parser.add_argument("--sink", type=int, default=4)
    parser.add_argument("--recent", type=int, default=256)
    parser.add_argument("--refresh-interval", type=int, default=1_000_000_000)
    parser.add_argument(
        "--refresh-layer-groups",
        type=int,
        default=None,
        help="Optional layer-group gating for refresh (e.g., 2 = odd/even layers refresh alternation).",
    )
    parser.add_argument(
        "--disable-sentence-trigger",
        action="store_true",
        help="Disable sentence-based refresh triggers (recommended for steady-state decode benchmarks).",
    )
    parser.add_argument("--alpha-k-head", type=int, default=1536)
    parser.add_argument(
        "--prefill-last-n",
        type=int,
        default=16,
        help="Prefill capture window (<=0 disables capture).",
    )
    parser.add_argument(
        "--patch-in-parent",
        action="store_true",
        help="Install sparse patch in the driver process as well (debug-only; may impact multiprocessing/CUDA init).",
    )
    parser.add_argument(
        "--enable-sparse",
        action="store_true",
        help="Write VLLM_SPARSE_CONTROLLER_JSON from CLI args (otherwise keep env as-is).",
    )
    parser.add_argument(
        "--disable-sparse",
        action="store_true",
        help="Force dense mode by unsetting VLLM_SPARSE_CONTROLLER_JSON.",
    )
    parser.add_argument(
        "--selector-fixed-k",
        action="store_true",
        help="Set VLLM_SPARSE_SELECTOR_FIXED_K=1 to reduce selector K-specialization jitter (may do extra padded work).",
    )
    parser.add_argument(
        "--wait-policy",
        type=str,
        default="split",
        choices=("", "chunk", "split"),
        help="Set VLLM_SPARSE_WAIT_POLICY (chunk=always wait per buf; split=skip wait when no pending work).",
    )
    parser.add_argument(
        "--refresh-stream-priority",
        type=int,
        default=None,
        help="Override VLLM_SPARSE_REFRESH_STREAM_PRIORITY (CUDA: smaller number => higher priority).",
    )
    parser.add_argument(
        "--refresh-profile",
        action="store_true",
        help="Enable refresh pipeline profiling (VLLM_SPARSE_REFRESH_PROFILE=1) and summarize the log.",
    )
    parser.add_argument(
        "--refresh-profile-detail",
        action="store_true",
        help="Enable detailed refresh profiling (VLLM_SPARSE_REFRESH_PROFILE_DETAIL=1).",
    )
    parser.add_argument(
        "--refresh-profile-call-min",
        type=int,
        default=0,
        help="VLLM_SPARSE_REFRESH_PROFILE_CALL_MIN: start logging after this many flushes.",
    )
    parser.add_argument(
        "--refresh-profile-every",
        type=int,
        default=1,
        help="VLLM_SPARSE_REFRESH_PROFILE_EVERY: sample every N flushes.",
    )
    parser.add_argument(
        "--refresh-profile-log",
        type=str,
        default="/tmp/vllm_sparse_refresh_profile.log",
        help="VLLM_SPARSE_REFRESH_PROFILE_LOG path.",
    )
    parser.add_argument(
        "--capture-kv-bucket",
        type=int,
        default=2048,
        help="Override VLLM_SPARSE_CAPTURE_KV_BUCKET to reduce selector/kernel specialization jitter (e.g. 2048/4096).",
    )
    parser.add_argument(
        "--trace-async",
        action="store_true",
        help="Enable async wait/flush tracing (VLLM_SPARSE_TRACE_ASYNC=1).",
    )
    parser.add_argument(
        "--trace-async-log",
        type=str,
        default="/tmp/vllm_sparse_async_trace.log",
        help="VLLM_SPARSE_TRACE_ASYNC_LOG path.",
    )
    parser.add_argument(
        "--trace-log-s-shapes",
        action="store_true",
        help="Trace refresh selector input shapes to debug K-specialization jitter (VLLM_SPARSE_TRACE_LOG_S_SHAPES=1).",
    )
    parser.add_argument(
        "--trace-log-s-shapes-log",
        type=str,
        default="/tmp/vllm_sparse_log_s_shape_trace.log",
        help="VLLM_SPARSE_TRACE_LOG_S_SHAPES_LOG path.",
    )
    add_deferred_bridge_arguments(parser)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_deferred_bridge_args(parser, args)
    return args


def main() -> None:
    args = parse_args()
    apply_deferred_bridge_env(args, os.environ)
    if args.decode_metrics_json:
        counter_path = _prepare_fa3_route_counter_mmap(args.decode_metrics_json)
        os.environ["VLLM_SPARSE_FA3_ROUTE_COUNTER_MMAP"] = str(counter_path)

    def _read_env_sparse_json() -> dict[str, object] | None:
        raw = os.environ.get("VLLM_SPARSE_CONTROLLER_JSON", "")
        if not raw:
            return None
        try:
            payload = json.loads(raw)
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        return payload

    def _argv_has_flag(name: str) -> bool:
        # argparse 本身无法区分“使用默认值”与“用户显式传参=默认值”。
        # benchmark 脚本的典型用法是：用户提前写好 VLLM_SPARSE_CONTROLLER_JSON，
        # 然后希望通过 CLI 的少量参数（如 --refresh-interval）覆盖其中的字段。
        # 因此这里用 sys.argv 做显式判定，避免在未传参时意外覆盖 env JSON。
        token = f"--{name}"
        for arg in sys.argv[1:]:
            if arg == token or arg.startswith(token + "="):
                return True
        return False

    # 默认操作：sparse + decode 测量时自动 reset prefix cache（除非用户显式关闭）。
    if (
        bool(args.measure_decode_latency)
        and ("VLLM_SPARSE_CONTROLLER_JSON" in os.environ or bool(args.enable_sparse))
        and (not _argv_has_flag("reset-prefix-cache"))
    ):
        args.reset_prefix_cache = True

    env_sparse_payload = _read_env_sparse_json()
    env_refresh_layer_groups: int | None = None
    if isinstance(env_sparse_payload, dict):
        raw_groups = env_sparse_payload.get("refresh_layer_groups")
        if raw_groups is not None:
            try:
                env_refresh_layer_groups = max(1, int(raw_groups))
            except Exception:
                env_refresh_layer_groups = None

    def _maybe_override_env_sparse_json() -> None:
        if "VLLM_SPARSE_CONTROLLER_JSON" not in os.environ:
            return
        if bool(args.enable_sparse) or bool(args.disable_sparse):
            return
        raw = os.environ.get("VLLM_SPARSE_CONTROLLER_JSON", "")
        try:
            payload = json.loads(raw)
        except Exception:
            return
        if not isinstance(payload, dict):
            return

        changed = False

        def _set(key: str, value: object) -> None:
            nonlocal changed
            if payload.get(key) != value:
                payload[key] = value
                changed = True

        def _ensure_trigger() -> dict[str, object]:
            nonlocal changed
            trig = payload.get("trigger")
            if not isinstance(trig, dict):
                trig = {}
                payload["trigger"] = trig
                changed = True
            return trig

        # 仅覆盖“用户显式传参”的字段。
        if _argv_has_flag("tau"):
            _set("tau", float(args.tau))
        if _argv_has_flag("k-min"):
            _set("k_min", int(args.k_min))
        if _argv_has_flag("k-max"):
            _set("k_max", k_max)
        if _argv_has_flag("sink"):
            _set("sink", int(args.sink))
        if _argv_has_flag("recent"):
            _set("recent", int(args.recent))
        if _argv_has_flag("refresh-layer-groups"):
            groups = args.refresh_layer_groups
            if groups is None:
                groups = 1
            _set("refresh_layer_groups", max(1, int(groups)))
        if _argv_has_flag("alpha-k-head"):
            payload.setdefault("alpha_fair", {})
            if isinstance(payload.get("alpha_fair"), dict):
                alpha_fair = payload["alpha_fair"]
                if alpha_fair.get("k_head") != int(args.alpha_k_head):
                    alpha_fair["k_head"] = int(args.alpha_k_head)
                    changed = True
        if _argv_has_flag("prefill-last-n"):
            _set("prefill_last_n_query", max(0, int(args.prefill_last_n)))

        # refresh_interval 的真实判定逻辑在 sparse patch 中使用 config.refresh_interval（顶层），
        # 因此这里同时覆盖顶层与 trigger.refresh_interval，避免出现“看似开了 refresh 实则不触发”。
        if _argv_has_flag("refresh-interval"):
            _set("refresh_interval", int(args.refresh_interval))
            trigger = _ensure_trigger()
            if trigger.get("refresh_interval") != int(args.refresh_interval):
                trigger["refresh_interval"] = int(args.refresh_interval)
                changed = True

        if _argv_has_flag("disable-sentence-trigger"):
            trigger = _ensure_trigger()
            if trigger.get("enable_sentence_triggers") != (not bool(args.disable_sentence_trigger)):
                trigger["enable_sentence_triggers"] = (not bool(args.disable_sentence_trigger))
                changed = True

        if changed:
            os.environ["VLLM_SPARSE_CONTROLLER_JSON"] = json.dumps(payload)

    def _truncate_text(path: str) -> None:
        try:
            with open(path, "w", encoding="utf-8"):
                pass
        except Exception:
            return

    def _pipeline_profile_marker(phase: str, **fields: object) -> None:
        path = os.environ.get("VLLM_SPARSE_PIPELINE_CPU_PROFILE_LOG")
        if not path:
            return
        try:
            parts = [f"phase={phase}"]
            for key, value in fields.items():
                parts.append(f"{key}={value}")
            with open(path, "a", encoding="utf-8") as handle:
                handle.write("pipeline_phase " + " ".join(parts) + "\n")
        except Exception:
            return

    def _refresh_profile_marker(phase: str, **fields: object) -> None:
        path = os.environ.get("VLLM_SPARSE_REFRESH_PROFILE_LOG")
        if not path:
            return
        try:
            payload = {"phase": str(phase)}
            payload.update(fields)
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(
                    f"{os.getpid()}\trefresh.marker\t{json.dumps(payload, sort_keys=True)}\n"
                )
        except Exception:
            return

    def _route_trace_marker(phase: str, **fields: object) -> None:
        path = os.environ.get("VLLM_SPARSE_FA3_ROUTE_TRACE_LOG")
        if not path:
            return
        try:
            payload = {"event": "bench_measure_marker", "phase": str(phase)}
            payload.update(fields)
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, sort_keys=True) + "\n")
        except Exception:
            return

    def _full_cudagraph_hook_profile_marker(
        phase: str,
        **fields: object,
    ) -> None:
        path = os.environ.get("VLLM_SPARSE_FULL_CUDAGRAPH_HOOK_PROFILE_LOG")
        if not path:
            return
        try:
            payload = {"event": "bench_measure_marker", "phase": str(phase)}
            payload.update(fields)
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, sort_keys=True) + "\n")
        except Exception:
            return

    def _summarize_refresh_profile(
        path: str,
        *,
        min_ctrl_step_exclusive: int | None = None,
        min_ts_ns_exclusive: int | None = None,
        max_ts_ns_inclusive: int | None = None,
    ) -> None:
        try:
            with open(path, "r", encoding="utf-8") as fp:
                lines = fp.readlines()
        except Exception:
            return
        records: list[dict[str, object]] = []
        for line in lines:
            parts = line.rstrip("\n").split("\t", 2)
            if len(parts) != 3:
                continue
            _, kind, payload = parts
            if kind != "refresh.flush":
                continue
            try:
                rec = json.loads(payload)
            except Exception:
                continue
            if not isinstance(rec, dict):
                continue
            if min_ctrl_step_exclusive is not None:
                try:
                    ctrl_step = int(rec.get("ctrl_step", -1))
                except Exception:
                    ctrl_step = -1
                if ctrl_step <= int(min_ctrl_step_exclusive):
                    continue
            if min_ts_ns_exclusive is not None:
                try:
                    ts_ns = int(rec.get("ts_ns", -1))
                except Exception:
                    ts_ns = -1
                if ts_ns <= int(min_ts_ns_exclusive):
                    continue
            if max_ts_ns_inclusive is not None:
                try:
                    ts_ns = int(rec.get("ts_ns", -1))
                except Exception:
                    ts_ns = -1
                if ts_ns > int(max_ts_ns_inclusive):
                    continue
            records.append(rec)
        if not records:
            return

        def _collect(key: str, *, predicate=None) -> list[float]:
            out: list[float] = []
            for rec in records:
                if predicate is not None:
                    try:
                        if not bool(predicate(rec)):
                            continue
                    except Exception:
                        continue
                val = rec.get(key)
                if val is None:
                    continue
                try:
                    out.append(float(val))
                except Exception:
                    continue
            return out

        def _p(pct: float, values: list[float]) -> float:
            if not values:
                return float("nan")
            xs = sorted(values)
            idx = int(round((len(xs) - 1) * pct))
            idx = max(0, min(idx, len(xs) - 1))
            return float(xs[idx])

        def _fmt(name: str, values: list[float], *, unit: str = "", decimals: int = 3) -> str:
            if not values:
                return f"{name}=n/a"
            mean = statistics.mean(values)
            p50 = _p(0.50, values)
            p90 = _p(0.90, values)
            if unit:
                return f"{name}_mean={mean:.{decimals}f}{unit} {name}_p50={p50:.{decimals}f}{unit} {name}_p90={p90:.{decimals}f}{unit}"
            return f"{name}_mean={mean:.{decimals}f} {name}_p50={p50:.{decimals}f} {name}_p90={p90:.{decimals}f}"

        def _as_int(value) -> int:
            try:
                return int(value)
            except Exception:
                return 0

        refresh_pred = lambda rec: _as_int(rec.get("refresh_payloads")) > 0
        prefill_pred = lambda rec: _as_int(rec.get("prefill_payloads")) > 0

        sel_ms = _collect("refresh_selector_gpu_ms", predicate=refresh_pred)
        rebuild_ms = _collect("refresh_rebuild_gpu_ms", predicate=refresh_pred)
        gather_ms = _collect("refresh_gather_gpu_ms", predicate=refresh_pred)
        key_norms_ms = _collect("refresh_key_norms_gpu_ms", predicate=refresh_pred)
        log_s_ms = _collect("refresh_log_s_gpu_ms", predicate=refresh_pred)
        log_s_triton_ms = _collect("refresh_log_s_triton_gpu_ms", predicate=refresh_pred)
        topk_ms = _collect("refresh_topk_gpu_ms", predicate=refresh_pred)
        sel_cpu_us = _collect("refresh_selector_cpu_us", predicate=refresh_pred)
        sel_compute_cpu_us = _collect("refresh_selector_compute_cpu_us", predicate=refresh_pred)
        sel_post_cpu_us = _collect("refresh_selector_post_cpu_us", predicate=refresh_pred)
        rebuild_cpu_us = _collect("refresh_rebuild_cpu_us", predicate=refresh_pred)
        total_cpu_us = _collect("refresh_total_cpu_us", predicate=refresh_pred)
        kv_len_total_vals = _collect("capture_kv_len_total", predicate=refresh_pred)
        stride_tokens_vals = _collect("rebuild_stride_tokens", predicate=refresh_pred)
        selected_k_vals = _collect("rebuild_selected_k", predicate=refresh_pred)
        block_size_vals = _collect("rebuild_block_size", predicate=refresh_pred)
        overlap_ratio_vals = _collect("refresh_overlap_ratio", predicate=refresh_pred)
        overlap_new_k_vals = _collect("refresh_overlap_new_k", predicate=refresh_pred)
        overlap_old_k_vals = _collect("refresh_overlap_old_k", predicate=refresh_pred)

        prefill_gpu_ms = _collect("prefill_gpu_ms", predicate=prefill_pred)
        prefill_cpu_us = _collect("prefill_cpu_us", predicate=prefill_pred)

        # rebuild 带宽估算依赖 copy_stats；已移除 skip_unchanged 统计后暂不输出。
        rebuild_bw_gbps: list[float] = []
        rebuild_bytes_gb: list[float] = []

        parts: list[str] = [
            f"count={len(records)}",
            f"refresh_records={sum(1 for r in records if refresh_pred(r))}",
            f"prefill_records={sum(1 for r in records if prefill_pred(r))}",
            _fmt("selector_ms", sel_ms, unit="ms"),
            _fmt("rebuild_ms", rebuild_ms, unit="ms"),
            _fmt("gather_ms", gather_ms, unit="ms"),
            _fmt("key_norms_ms", key_norms_ms, unit="ms"),
            _fmt("log_s_ms", log_s_ms, unit="ms"),
            _fmt("log_s_triton_ms", log_s_triton_ms, unit="ms"),
            _fmt("topk_ms", topk_ms, unit="ms"),
        ]
        if kv_len_total_vals:
            parts.append(_fmt("kv_len_total", kv_len_total_vals, unit="", decimals=0))
        if stride_tokens_vals:
            parts.append(_fmt("stride_tokens", stride_tokens_vals, unit="", decimals=0))
        if selected_k_vals:
            parts.append(_fmt("selected_k", selected_k_vals, unit="", decimals=0))
        if block_size_vals:
            parts.append(_fmt("block_size", block_size_vals, unit="", decimals=0))
        if overlap_ratio_vals:
            parts.append(_fmt("overlap_ratio", overlap_ratio_vals, unit="", decimals=3))
        if overlap_new_k_vals:
            parts.append(_fmt("overlap_new_k", overlap_new_k_vals, unit="", decimals=1))
        if overlap_old_k_vals:
            parts.append(_fmt("overlap_old_k", overlap_old_k_vals, unit="", decimals=1))
        if sel_cpu_us:
            parts.append(_fmt("selector_cpu_us", sel_cpu_us, unit="us", decimals=1))
        if sel_compute_cpu_us:
            parts.append(_fmt("selector_compute_cpu_us", sel_compute_cpu_us, unit="us", decimals=1))
        if sel_post_cpu_us:
            parts.append(_fmt("selector_post_cpu_us", sel_post_cpu_us, unit="us", decimals=1))
        if rebuild_cpu_us:
            parts.append(_fmt("rebuild_cpu_us", rebuild_cpu_us, unit="us", decimals=1))
        if total_cpu_us:
            parts.append(_fmt("total_cpu_us", total_cpu_us, unit="us", decimals=1))
        if rebuild_bw_gbps:
            parts.append(_fmt("rebuild_bw_gbps", rebuild_bw_gbps, unit="", decimals=1))
        if rebuild_bytes_gb:
            parts.append(_fmt("rebuild_gb", rebuild_bytes_gb, unit="", decimals=3))
        if prefill_gpu_ms:
            parts.append(_fmt("prefill_gpu_ms", prefill_gpu_ms, unit="ms"))
        if prefill_cpu_us:
            parts.append(_fmt("prefill_cpu_us", prefill_cpu_us, unit="us", decimals=1))

        print("[refresh-profile] " + " ".join(parts), flush=True)

    def _read_max_ctrl_step(path: str) -> int | None:
        try:
            with open(path, "r", encoding="utf-8") as fp:
                lines = fp.readlines()
        except Exception:
            return None
        out: int | None = None
        for line in lines:
            parts = line.rstrip("\n").split("\t", 2)
            if len(parts) != 3:
                continue
            _, kind, payload = parts
            if kind != "refresh.flush":
                continue
            try:
                rec = json.loads(payload)
            except Exception:
                continue
            if not isinstance(rec, dict):
                continue
            try:
                step = int(rec.get("ctrl_step", -1))
            except Exception:
                continue
            out = step if out is None else max(int(out), int(step))
        return out

    repo_root = _setup_repo_imports()
    _install_transformers_register_shim()

    if int(args.batch_size) > 1 and (not bool(args.disable_cascade_attn)):
        print(
            "[warn] 当前 batch 内 prompt 完全相同，且未关闭 vLLM V1 cascade attention；"
            "这会显著削弱 sparse/refresh 的相对收益，建议加上 --disable-cascade-attn 做对比。",
            file=sys.stderr,
            flush=True,
        )

    k_max: int | None
    if str(args.k_max).lower() in {"none", "null"}:
        k_max = None
    else:
        k_max = int(args.k_max)

    # Sparse config is opt-in to keep the benchmark script usable for both
    # dense and sparse runs (docs expect `unset VLLM_SPARSE_CONTROLLER_JSON`
    # to produce a dense baseline).
    cfg_payload: dict[str, object] = {
        "enabled": True,
        "tau": float(args.tau),
        "k_min": int(args.k_min),
        "k_max": k_max,
        "sink": int(args.sink),
        "recent": int(args.recent),
        "refresh_interval": int(args.refresh_interval),
        "alpha_fair": {"k_head": int(args.alpha_k_head)},
        "prefill_last_n_query": max(0, int(args.prefill_last_n)),
        "trigger": {
            "refresh_interval": int(args.refresh_interval),
            "enable_sentence_triggers": not bool(args.disable_sentence_trigger),
        },
    }
    refresh_layer_groups = args.refresh_layer_groups
    if refresh_layer_groups is None and env_refresh_layer_groups is not None:
        refresh_layer_groups = env_refresh_layer_groups

    if refresh_layer_groups is not None:
        cfg_payload["refresh_layer_groups"] = max(1, int(refresh_layer_groups))
    if bool(args.disable_sparse):
        os.environ.pop("VLLM_SPARSE_CONTROLLER_JSON", None)
    elif bool(args.enable_sparse):
        os.environ["VLLM_SPARSE_CONTROLLER_JSON"] = json.dumps(cfg_payload)
    else:
        _maybe_override_env_sparse_json()

    if "VLLM_SPARSE_CONTROLLER_JSON" in os.environ:
        mp_method = str(os.environ.get("VLLM_WORKER_MULTIPROC_METHOD", "")).strip().lower()
        if mp_method and mp_method != "spawn":
            print(
                f"[warn] VLLM_WORKER_MULTIPROC_METHOD={mp_method!r} 可能导致 sparse patch 无法在 worker 中生效；"
                "建议使用 'spawn'（例如：VLLM_WORKER_MULTIPROC_METHOD=spawn）。",
                file=sys.stderr,
                flush=True,
            )

    if "VLLM_SPARSE_CONTROLLER_JSON" in os.environ and (not _argv_has_flag("selector-fixed-k")):
        os.environ.setdefault("VLLM_SPARSE_SELECTOR_FIXED_K", "1")

    _enforce_sm80_gt1_runner_contract(
        args=args,
        sparse_payload=_read_env_sparse_json(),
    )

    if bool(args.selector_fixed_k):
        os.environ["VLLM_SPARSE_SELECTOR_FIXED_K"] = "1"
    if args.capture_kv_bucket is not None:
        os.environ["VLLM_SPARSE_CAPTURE_KV_BUCKET"] = str(int(args.capture_kv_bucket))
    if str(args.wait_policy).strip():
        os.environ["VLLM_SPARSE_WAIT_POLICY"] = str(args.wait_policy).strip()
    if args.refresh_stream_priority is not None:
        os.environ["VLLM_SPARSE_REFRESH_STREAM_PRIORITY"] = str(int(args.refresh_stream_priority))
    if bool(args.trace_async):
        os.environ["VLLM_SPARSE_TRACE_ASYNC"] = "1"
        os.environ["VLLM_SPARSE_TRACE_ASYNC_LOG"] = str(args.trace_async_log)
    if bool(args.trace_log_s_shapes):
        os.environ["VLLM_SPARSE_TRACE_LOG_S_SHAPES"] = "1"
        os.environ["VLLM_SPARSE_TRACE_LOG_S_SHAPES_LOG"] = str(args.trace_log_s_shapes_log)

    refresh_profile_log = str(args.refresh_profile_log)
    if bool(args.refresh_profile):
        os.environ["VLLM_SPARSE_REFRESH_PROFILE"] = "1"
        os.environ["VLLM_SPARSE_REFRESH_PROFILE_CALL_MIN"] = str(int(args.refresh_profile_call_min))
        os.environ["VLLM_SPARSE_REFRESH_PROFILE_EVERY"] = str(max(1, int(args.refresh_profile_every)))
        os.environ["VLLM_SPARSE_REFRESH_PROFILE_LOG"] = refresh_profile_log
        if bool(args.refresh_profile_detail):
            os.environ["VLLM_SPARSE_REFRESH_PROFILE_DETAIL"] = "1"
        _truncate_text(refresh_profile_log)

    controller = None
    if bool(args.patch_in_parent) and ("VLLM_SPARSE_CONTROLLER_JSON" in os.environ):
        from patches.vllm_sparse_patch import SparseControllerConfig, apply_vllm_sparse_patch

        cfg = SparseControllerConfig(
            tau=float(args.tau),
            k_min=int(args.k_min),
            k_max=k_max,
            sink=int(args.sink),
            recent=int(args.recent),
            refresh_interval=int(args.refresh_interval),
            enabled=True,
        )
        cfg.alpha_fair.k_head = int(args.alpha_k_head)
        cfg.prefill_last_n_query = max(0, int(args.prefill_last_n))
        cfg.trigger.refresh_interval = int(args.refresh_interval)
        cfg.trigger.enable_sentence_triggers = not bool(args.disable_sentence_trigger)
        controller = apply_vllm_sparse_patch(cfg)

    # Sparse standalone runs use the current FA3 route by default. Keep this
    # explicit so stale shell env cannot reopen the retired Triton backend path,
    # while allowing the SM100 parent runner to request FA4.
    requested_flash_attn_version = os.environ.get("VLLM_FLASH_ATTN_VERSION", "3")
    if requested_flash_attn_version not in {"3", "4"}:
        requested_flash_attn_version = "3"
    requested_attention_backend = os.environ.get("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
    if requested_attention_backend not in {"FLASH_ATTN", "FLASH_ATTN_VLLM_V1"}:
        requested_attention_backend = "FLASH_ATTN"
    if requested_flash_attn_version == "4":
        requested_attention_backend = "FLASH_ATTN_VLLM_V1"
    os.environ["VLLM_ATTENTION_BACKEND"] = requested_attention_backend
    os.environ["VLLM_FLASH_ATTN_VERSION"] = requested_flash_attn_version

    from vllm import LLM, SamplingParams  # pylint: disable=import-error

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
    if int(engine_kwargs["tensor_parallel_size"]) > 1 and (
        os.environ.get("VLLM_SPARSE_FORCE_DISABLE_CUSTOM_AR", "") == "1"
    ):
        # [TP-CUSTOM-AR-ESCAPE] NVLink probe says custom AR is safe, but some
        # boxes still crash with custom AR + FULL cudagraph (remote exp4,
        # 2026-07-06). Probe cannot see driver/graph interactions — this env is
        # the operator override; it wins over the probe and --enable flag.
        engine_kwargs["disable_custom_all_reduce"] = True
        print(
            "[run_sparse_only] TP>1: custom all-reduce force-disabled via "
            "VLLM_SPARSE_FORCE_DISABLE_CUSTOM_AR=1",
            flush=True,
        )
    elif int(engine_kwargs["tensor_parallel_size"]) > 1 and not bool(
        getattr(args, "enable_custom_all_reduce", False)
    ):
        # [TP-CUSTOM-AR-PROBE] custom all-reduce is a large win for small decode
        # messages on NVLink boxes but crashes worker init on PCIe-only
        # topologies ('custom_all_reduce.cuh ... invalid argument').
        _nvlink_ok, _nvlink_reason = _visible_gpus_all_have_nvlink()
        if _nvlink_ok:
            print(
                "[run_sparse_only] TP>1: %s; leaving vLLM custom all-reduce "
                "enabled" % _nvlink_reason,
                flush=True,
            )
        else:
            engine_kwargs["disable_custom_all_reduce"] = True
            print(
                "[run_sparse_only] TP>1: disabling vLLM custom all-reduce "
                "(%s); pass --enable-custom-all-reduce to force it on"
                % _nvlink_reason,
                flush=True,
            )
    if int(engine_kwargs["tensor_parallel_size"]) > 1:
        # [TP-ENV-PIN] propagate TP size to every spawned child (EngineCore +
        # workers): controller._init_tp_size reads this env once at init and
        # the EngineCore sentence hook self-disables on TP>1. Without it the
        # hook does full per-token trigger bookkeeping in the scheduler
        # process — pure waste under TP (refresh decisions live worker-side).
        os.environ["VLLM_TENSOR_PARALLEL_SIZE"] = str(
            int(engine_kwargs["tensor_parallel_size"])
        )
    if _sync_scheduling_requested(args) and _engine_args_accepts("async_scheduling"):
        # [TP-ASYNC-HARVEST] TP>1 async scheduling is supported: the patch
        # harvests vLLM's own async sampled-token D2H copy and repairs the -1
        # placeholders in token_ids_cpu in place (1-step lag, semantically the
        # same as the TP=1 enginecore sentence hook). This flag/env is only the
        # escape hatch back to synchronous scheduling.
        engine_kwargs["async_scheduling"] = False
    if int(getattr(args, "max_model_len", 0) or 0) > 0:
        engine_kwargs["max_model_len"] = int(args.max_model_len)
    # Shared-GPU workaround: fix KV cache size via env to skip vLLM memory
    # profiling (gpu_worker.py asserts free-memory consistency, which breaks when
    # other jobs on the same GPU release memory mid-profiling). Only used when set.
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
        # [C4-DETOKENIZE 2026-07-06] speed 路径只消费 token_ids（text 由
        # _final_outputs_with_decoded_text 结束后统一重解码），关闭 vLLM 前端
        # 每步增量 detokenize（bs×5-20µs/步纯浪费）；非测速分支直读
        # output.text，必须保持默认开启——严格分支限定。
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
        request_ids = [f"bench-{i}-{time.time_ns()}" for i in range(len(prompts))]
        mix_max = getattr(args, "mix_max_new_tokens", None)
        t_add0 = time.perf_counter()
        for i, (rid, p) in enumerate(zip(request_ids, prompts)):
            if mix_max is not None and int(mix_max) > 0 and (i % 2) == 1:
                sp_i = SamplingParams(
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=int(mix_max),
                    ignore_eos=not bool(args.respect_eos),
                    # [C4-DETOKENIZE 2026-07-06] 本函数是测速专用 engine-step
                    # 循环，恒关增量 detokenize（与 sp 同口径）。
                    detokenize=False,
                )
            else:
                sp_i = sp
            engine.llm_engine.add_request(rid, p, sp_i)  # type: ignore[attr-defined]
        t_add1 = time.perf_counter()

        prev_len: dict[str, int] = {}
        # batch_size 使能 all-decode 稳态窗（全部请求完成 chunked prefill 后的窗口）：
        # 长 ctx 大 bs 档 decode 计量窗前段与 prefill 交错（实测 8×32k 占墙钟 81%），
        # decode_tok_per_s 被稀释；all_decode_tok_per_s 反映真实稳态 decode 吞吐。
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
        final_output_items: dict[str, object] = {}

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
                        if str(key).startswith(("ec_", "mr_"))
                    }
                )
            if args.outputs_json:
                for item in step_out:
                    rid = getattr(item, "request_id", None)
                    outputs = getattr(item, "outputs", None)
                    if rid is None or not outputs:
                        continue
                    final_output_items[str(rid)] = item

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
        if args.outputs_json:
            final_outputs.clear()
            for rid, item in final_output_items.items():
                final_outputs[str(rid)] = output_payload_from_generation(
                    item,
                    include_text=False,
                )
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

    reset_after_warmup = False
    if bool(args.reset_prefix_cache):
        _reset_prefix_cache()
    if bool(args.measure_decode_latency):
        warmup_runs = max(0, int(args.warmup_runs))
        warmup_max_ctrl_step: int | None = None
        for warmup_index in range(warmup_runs):
            _pipeline_profile_marker("warmup_begin", index=warmup_index)
            _generate_with_engine_step()
            _pipeline_profile_marker("warmup_end", index=warmup_index)
            if bool(args.reset_prefix_cache):
                _reset_prefix_cache()
                reset_after_warmup = True
        if bool(args.refresh_profile):
            # 更严谨：不再 truncate 文件（避免与 worker 异步写日志发生竞态），改为记录 warmup 的 ctrl_step 边界，
            # 在统计阶段过滤 ctrl_step<=boundary 的记录（也能屏蔽 warmup 的 pending flush 迟到写入）。
            warmup_max_ctrl_step = _read_max_ctrl_step(refresh_profile_log)
        final_outputs.clear()
    else:
        engine.generate(prompts, sp)
    if bool(args.reset_prefix_cache) and not reset_after_warmup:
        _reset_prefix_cache()
    if bool(args.measure_decode_latency):
        _reset_fa3_route_counters_for_measurement()
        profiling = maybe_start_vllm_torch_profile(engine)
        warmup_measure_begin_ts_ns = time.time_ns()
        _pipeline_profile_marker("measure_begin", warmup_runs=warmup_runs)
        _refresh_profile_marker("measure_begin", ts_ns=warmup_measure_begin_ts_ns)
        _route_trace_marker(
            "measure_begin",
            ts_ns=warmup_measure_begin_ts_ns,
            warmup_runs=warmup_runs,
        )
        _full_cudagraph_hook_profile_marker(
            "measure_begin",
            ts_ns=warmup_measure_begin_ts_ns,
            warmup_runs=warmup_runs,
        )
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
            warmup_measure_end_ts_ns = time.time_ns()
            _pipeline_profile_marker("measure_end", warmup_runs=warmup_runs)
            _refresh_profile_marker("measure_end", ts_ns=warmup_measure_end_ts_ns)
            _route_trace_marker(
                "measure_end",
                ts_ns=warmup_measure_end_ts_ns,
                warmup_runs=warmup_runs,
            )
            _full_cudagraph_hook_profile_marker(
                "measure_end",
                ts_ns=warmup_measure_end_ts_ns,
                warmup_runs=warmup_runs,
            )
        tps = float(out_tokens) / elapsed if elapsed > 0 else float("nan")
        decode_tps = float(decode_tokens) / decode_elapsed if decode_elapsed > 0 else float("nan")
        route_counters = _fa3_route_counter_snapshot()
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
                "mix_max_new_tokens": run_config["mix_max_new_tokens"],
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
            metrics_payload.update(_fa3_route_counter_metrics(route_counters))
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
        if bool(args.refresh_profile):
            _summarize_refresh_profile(
                refresh_profile_log,
                min_ctrl_step_exclusive=warmup_max_ctrl_step,
                min_ts_ns_exclusive=warmup_measure_begin_ts_ns,
                max_ts_ns_inclusive=warmup_measure_end_ts_ns,
            )
        try:
            if controller is None:
                raise RuntimeError("controller not available in parent process (use --patch-in-parent to enable).")
            layer_count = len(getattr(controller, "layer_states", {}) or {})
            compact_max_per_layer: list[int] = []
            bootstrap_done_layers = 0
            for st in (getattr(controller, "layer_states", {}) or {}).values():
                if bool(getattr(st, "bootstrap_done", False)):
                    bootstrap_done_layers += 1
                kv_lens = getattr(st, "compact_kv_len", None)
                if kv_lens:
                    compact_max_per_layer.append(int(max(int(x) for x in kv_lens)))
            if compact_max_per_layer:
                compact_min = min(compact_max_per_layer)
                compact_max = max(compact_max_per_layer)
            else:
                compact_min = 0
                compact_max = 0
            print(
                f"[sparse-stats] layers={layer_count} bootstrap_done_layers={bootstrap_done_layers} "
                f"compact_kv_len_max_range=[{compact_min},{compact_max}]",
                flush=True,
            )
        except Exception:
            pass
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
