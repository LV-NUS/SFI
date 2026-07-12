from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.sm80_run_pair import (
    ALLOWED_TRACE_ENV_KEYS,
    RouteProofResult,
    STAGE_A_SOURCE_COUNTER_SCHEMA_VERSION,
    build_pairing_digest,
    build_config_digest,
    validate_shared_route_proof,
)


DEFAULT_OUT_ATOL = 0.0
DEFAULT_OUT_RTOL = 0.0
DEFAULT_LSE_ATOL = 1.0e-3
DEFAULT_LSE_RTOL = 0.0
GATE_FAILURE_EXIT_CODE = 2
DEFAULT_MODEL = "./qwen3-0.6b"
DEFAULT_PROMPT = "benchmarks/needle_prompt_part1.txt"
_REMOTE_FST_V2_PYTHON = (
    "/mnt/bn/ecom-ai-platform-1/yzc/miniconda/envs/fst_v2/bin/python"
)
DEFAULT_PYTHON = os.environ.get(
    "VLLM_BENCH_PYTHON",
    _REMOTE_FST_V2_PYTHON if Path(_REMOTE_FST_V2_PYTHON).exists() else sys.executable,
)
# Historical name: this root is used by both FA3 and FA4 vendored bridge loading.
DEFAULT_FA3_UPSTREAM_ROOT = str(
    (_REPO_ROOT / "third_party_upstreams" / "vllm-project-flash-attention").resolve()
)
DEFAULT_TIMEOUT_S = 900
DEFAULT_COMPACT_BLOCKS_PER_SLOT = 128
DEFAULT_MAX_LIVE_SPARSE_SLOTS = 8
DEFAULT_STAGE_A_RECENT = 256
ACTIVE_SM80_GT1_RUNNER = "bench_sm80_mixed_page_one_shot_graph_e2e.py"


def _process_tree_snapshot() -> list[tuple[int, int]]:
    try:
        out = subprocess.check_output(
            ["ps", "-eo", "pid=,ppid="],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return []
    rows: list[tuple[int, int]] = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            rows.append((int(parts[0]), int(parts[1])))
        except ValueError:
            continue
    return rows


def _descendant_pids_from_snapshot(
    root_pid: int,
    snapshot: list[tuple[int, int]],
) -> list[int]:
    children_by_parent: dict[int, list[int]] = {}
    for pid, ppid in snapshot:
        children_by_parent.setdefault(int(ppid), []).append(int(pid))

    descendants: list[int] = []
    stack = list(children_by_parent.get(int(root_pid), ()))
    seen: set[int] = set()
    while stack:
        pid = int(stack.pop())
        if pid in seen:
            continue
        seen.add(pid)
        descendants.append(pid)
        stack.extend(children_by_parent.get(pid, ()))
    return descendants


def _terminate_process_tree(proc: subprocess.Popen[str], *, sig: signal.Signals) -> None:
    pids = _descendant_pids_from_snapshot(int(proc.pid), _process_tree_snapshot())
    try:
        os.killpg(int(proc.pid), sig)
    except ProcessLookupError:
        pass
    except Exception:
        # Fall back to explicit child pids below. Process-group teardown can fail
        # if the child runtime creates its own session.
        pass
    for pid in sorted(set(pids), reverse=True):
        if pid == os.getpid():
            continue
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            continue


def _legacy_phase1_runner_disabled_message() -> str:
    return (
        "stale runner boundary: "
        "bench_sm80_mixed_page_full_cudagraph_phase1.py is helper/legacy only "
        "for SM80 GT1 full-cudagraph work. Use "
        f"benchmarks/{ACTIVE_SM80_GT1_RUNNER} for final benchmark evidence."
    )


def _enforce_phase1_cli_runner_contract(args: argparse.Namespace) -> None:
    if bool(getattr(args, "dry_run", False)):
        return
    if os.environ.get("VLLM_SPARSE_ALLOW_LEGACY_PHASE1_RUNNER", "0") == "1":
        print(
            "[warn] " + _legacy_phase1_runner_disabled_message(),
            file=sys.stderr,
            flush=True,
        )
        return
    print(
        "[error] " + _legacy_phase1_runner_disabled_message(),
        file=sys.stderr,
        flush=True,
    )
    raise SystemExit(GATE_FAILURE_EXIT_CODE)


@dataclass(frozen=True)
class Phase1FullGraphRecord:
    case: str
    full_cudagraph_enabled: bool
    capture_route: str
    replay_route: str
    resolver_kind: int
    mixed_wrapper_event_count: int
    actual_fwd_mixed_page_count: int
    resolved_row_ptr_fwd_mixed_page_count: int
    carrier_pointer_signature_stable: bool
    fresh_carrier_trace_changed: bool
    row_mode_distribution: dict[str, int]
    reference_max_abs_diff: float
    out_atol: float
    out_rtol: float
    lse_atol: float
    lse_rtol: float
    replay_prep_allocation_free: bool
    replay_prep_no_d2h_sync: bool
    carrier_update_us: float
    full_graph_replay_us: float
    gate_passed: bool
    run_pair_id: str = ""
    config_digest: str = ""
    route_proof_passed: bool = False
    route_proof_reasons: list[str] = field(default_factory=list)
    replay_prehook_us: float = -1.0
    carrier_update_kernel_count: int = -1
    rrp_signature_hit_rate: float = -1.0
    rrp_full_bind_count: int = -1


@dataclass(frozen=True)
class Phase1CommandResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--fa3-upstream-root", default=DEFAULT_FA3_UPSTREAM_ROOT)
    parser.add_argument("--cuda-visible-devices", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--split-context-prompts",
        action="store_true",
        help="Pass each Context: segment as a separate request to the sparse/dense runners.",
    )
    parser.add_argument(
        "--respect-eos",
        action="store_true",
        help="Let generation stop on EOS instead of forcing max_new_tokens.",
    )
    parser.add_argument(
        "--outputs-include-text",
        action="store_true",
        help="Include decoded text in runner outputs for semantic reference checks.",
    )
    parser.add_argument(
        "--chat-template",
        action="store_true",
        help="Render prompts through the model chat template in sparse/dense runners.",
    )
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--gpu-mem-util", type=float, default=0.9)
    parser.add_argument("--timeout-s", type=int, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--route-trace-output", default="")
    parser.add_argument("--decode-metrics-output", default="")
    parser.add_argument("--hook-profile-output", default="")
    parser.add_argument("--rrp-profile-output", default="")
    parser.add_argument("--case", default="sm80_full_vllm_graph_smoke")
    parser.add_argument(
        "--compact-blocks-per-slot",
        type=int,
        default=DEFAULT_COMPACT_BLOCKS_PER_SLOT,
    )
    parser.add_argument(
        "--max-live-sparse-slots",
        type=int,
        default=DEFAULT_MAX_LIVE_SPARSE_SLOTS,
    )
    parser.add_argument(
        "--alpha-k-head",
        type=int,
        default=1536,
        help=(
            "Selector per-head persist budget (alpha_fair.k_head) forwarded to "
            "the controller payload; effective selected_k is tile-aligned by "
            "compact_recent_effective_k_head."
        ),
    )
    parser.add_argument("--recent", type=int, default=DEFAULT_STAGE_A_RECENT)
    parser.add_argument(
        "--prefill-last-n",
        type=int,
        default=16,
        help=(
            "Number of prefill query rows captured for one-shot selector "
            "bootstrap. 1 disables gt1 log_f reduce and is useful for hide "
            "ablation."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write a fail-closed record without launching vLLM.",
    )
    args = parser.parse_args(argv)
    if args.warmup < 0:
        parser.error("--warmup must be >= 0")
    if args.iters <= 0:
        parser.error("--iters must be > 0")
    if args.batch_size <= 0:
        parser.error("--batch-size must be > 0")
    if args.timeout_s <= 0:
        parser.error("--timeout-s must be > 0")
    if args.compact_blocks_per_slot <= 0:
        parser.error("--compact-blocks-per-slot must be > 0")
    if args.max_live_sparse_slots <= 0:
        parser.error("--max-live-sparse-slots must be > 0")
    if args.recent <= 0:
        parser.error("--recent must be > 0")
    if args.prefill_last_n < 0:
        parser.error("--prefill-last-n must be >= 0")
    return args


def record_to_jsonable(record: Phase1FullGraphRecord) -> dict[str, object]:
    return asdict(record)


def write_records(path: Path, records: list[Phase1FullGraphRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(record_to_jsonable(record), sort_keys=True, allow_nan=False)
            )
            handle.write("\n")


def _repo_root() -> Path:
    return _REPO_ROOT


def _git_output(args: list[str]) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(_repo_root()), *args],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def _file_sha256(path: Path | None) -> str:
    if path is None or not path.exists() or not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _as_float(value: Any, default: float = -1.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _resolve_fa3_so_path(args: argparse.Namespace) -> Path | None:
    root = Path(str(args.fa3_upstream_root)) / "vllm_flash_attn"
    candidates = sorted(root.glob("_vllm_fa3_C*.so"))
    return candidates[0].resolve() if candidates else None


def _trace_profile_env(env: dict[str, str]) -> dict[str, str]:
    return {key: env.get(key, "") for key in ALLOWED_TRACE_ENV_KEYS}


def _run_pair_config(
    args: argparse.Namespace,
    *,
    env: dict[str, str],
    fa3_so_sha256: str,
) -> dict[str, object]:
    return {
        "checkout": _git_output(["rev-parse", "--short", "HEAD"]),
        "dirty_state": _git_output(["status", "--short"]),
        "fa3_so_sha256": fa3_so_sha256,
        "controller_json": env.get("VLLM_SPARSE_CONTROLLER_JSON", ""),
        "model": str(args.model),
        "prompt": str(args.prompt),
        "output_len": int(args.iters),
        "batch_size": int(args.batch_size),
        "full_cuda_graph": True,
        "env": _trace_profile_env(env),
    }


def _route_proof_payload(
    *,
    route_summary: dict[str, Any],
    env: dict[str, str],
    fa3_so_sha256: str,
) -> dict[str, Any]:
    fa3_backend_evidence = {
        "mixed_wrapper_event_count": int(
            route_summary.get("mixed_wrapper_event_count", 0) or 0
        ),
        "actual_fwd_mixed_page_count": int(
            route_summary.get("actual_fwd_mixed_page_count", 0) or 0
        ),
        "resolved_row_ptr_fwd_mixed_page_count": int(
            route_summary.get("resolved_row_ptr_fwd_mixed_page_count", 0) or 0
        ),
    }
    explicit_fa3_backend = route_summary.get("fa3_backend")
    fa3_backend = (
        bool(explicit_fa3_backend)
        if explicit_fa3_backend is not None
        else all(value > 0 for value in fa3_backend_evidence.values())
    )
    payload = dict(route_summary)
    phases = route_summary.get("phase_summaries", {})
    post_switch = (
        phases.get("post_switch_phase", {})
        if isinstance(phases, dict)
        else {}
    )
    if isinstance(post_switch, dict) and post_switch.get("row_source_distribution"):
        payload["route_proof_phase"] = "post_switch_phase"
        for key in (
            "row_mode_distribution",
            "row_source_distribution",
            "source_counter_schema_version",
            "expected_rows",
            "num_kv_heads",
            "source_counter_missing_fields",
        ):
            if key in post_switch:
                payload[key] = post_switch[key]
    payload.update(
        {
            "fa3_backend": fa3_backend,
            "fa3_backend_configured": bool(env.get("VLLM_SPARSE_FA3_UPSTREAM_ROOT")),
            "fa3_backend_evidence": fa3_backend_evidence,
            "full_cuda_graph": bool(
                route_summary.get("full_graph_replay_refresh_seen", False)
            ),
            "controller_json_sha256": _text_sha256(
                env.get("VLLM_SPARSE_CONTROLLER_JSON", "")
            )
            if env.get("VLLM_SPARSE_CONTROLLER_JSON")
            else "",
            "fa3_so_sha256": fa3_so_sha256,
        }
    )
    return payload


def _route_proof_failure_reason(route_proof_result: RouteProofResult) -> str:
    return "route_proof_failed:" + ",".join(route_proof_result.reasons)


def _apply_route_proof_gate(
    record: Phase1FullGraphRecord,
    failure_reason: str,
    route_proof_result: RouteProofResult,
) -> tuple[Phase1FullGraphRecord, str]:
    if route_proof_result.passed:
        return record, failure_reason
    route_failure = _route_proof_failure_reason(route_proof_result)
    merged_reason = (
        f"{failure_reason};{route_failure}" if failure_reason else route_failure
    )
    return replace(record, gate_passed=False), merged_reason


def _default_route_trace_path(output: Path) -> Path:
    return output.with_name(output.stem + "_route.jsonl")


def _default_hook_profile_path(output: Path) -> Path:
    return output.with_name(output.stem + "_full_cudagraph_hook_profile.jsonl")


def _default_rrp_profile_path(output: Path) -> Path:
    return output.with_name(output.stem + "_rrp_profile.jsonl")


def _default_decode_metrics_path(output: Path) -> Path:
    return output.with_name(output.stem + "_decode_metrics.json")


def _default_sparse_outputs_path(output: Path) -> Path:
    return output.with_name(output.stem + "_sparse_outputs.json")


def _default_dense_outputs_path(output: Path) -> Path:
    return output.with_name(output.stem + "_dense_outputs.json")


def _tail(text: str, limit: int = 12_000) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]


def _sparse_controller_payload(args: argparse.Namespace) -> dict[str, object]:
    return {
        "enabled": True,
        "attn_mode": "compact_recent",
        "compact_page_residency_enabled": True,
        "max_live_sparse_slots": int(args.max_live_sparse_slots),
        "compact_blocks_per_slot": int(args.compact_blocks_per_slot),
        "k_min": 32,
        "k_max": None,
        "sink": 4,
        "recent": int(getattr(args, "recent", DEFAULT_STAGE_A_RECENT)),
        "refresh_interval": 1_000_000_000,
        "alpha_fair": {"k_head": int(getattr(args, "alpha_k_head", 1536))},
        "prefill_last_n_query": max(0, int(getattr(args, "prefill_last_n", 16))),
        "trigger": {
            "refresh_interval": 1_000_000_000,
            "enable_sentence_triggers": False,
        },
    }


def _build_smoke_command(
    args: argparse.Namespace,
    *,
    metrics_path: Path,
    outputs_path: Path,
) -> list[str]:
    repo_root = _repo_root()
    command = [
        str(args.python),
        str(repo_root / "benchmarks" / "run_sparse_only.py"),
        "--model",
        str(args.model),
        "--prompt",
        str(args.prompt),
        "--batch-size",
        str(int(args.batch_size)),
        "--max-new-tokens",
        str(int(args.iters)),
        "--full-cuda-graph",
        "--cudagraph-capture-sizes",
        str(int(args.batch_size)),
        "--disable-cascade-attn",
        "--measure-decode-latency",
        "--decode-metrics-json",
        str(metrics_path),
        "--outputs-json",
        str(outputs_path),
        "--warmup-runs",
        str(int(args.warmup)),
        "--reset-prefix-cache",
        "--gpu-mem-util",
        str(float(args.gpu_mem_util)),
        "--prefill-last-n",
        str(max(0, int(getattr(args, "prefill_last_n", 16)))),
    ]
    if int(getattr(args, "max_model_len", 0) or 0) > 0:
        command.extend(["--max-model-len", str(int(args.max_model_len))])
    if int(getattr(args, "tensor_parallel_size", 1) or 1) > 1:
        command.extend(
            ["--tensor-parallel-size", str(int(args.tensor_parallel_size))]
        )
    if bool(getattr(args, "split_context_prompts", False)):
        command.append("--split-context-prompts")
    if bool(getattr(args, "respect_eos", False)):
        command.append("--respect-eos")
    if bool(getattr(args, "outputs_include_text", False)):
        command.append("--outputs-include-text")
    if bool(getattr(args, "chat_template", False)):
        command.append("--chat-template")
    if bool(getattr(args, "enable_thinking", False)):
        command.append("--enable-thinking")
    return command


def _build_dense_reference_command(
    args: argparse.Namespace,
    *,
    outputs_path: Path,
) -> list[str]:
    repo_root = _repo_root()
    command = [
        str(args.python),
        str(repo_root / "benchmarks" / "run_dense_only.py"),
        "--model",
        str(args.model),
        "--prompt",
        str(args.prompt),
        "--batch-size",
        str(int(args.batch_size)),
        "--max-new-tokens",
        str(int(args.iters)),
        "--full-cuda-graph",
        "--cudagraph-capture-sizes",
        str(int(args.batch_size)),
        "--disable-cascade-attn",
        "--measure-decode-latency",
        "--outputs-json",
        str(outputs_path),
        "--reset-prefix-cache",
        "--gpu-mem-util",
        str(float(args.gpu_mem_util)),
    ]
    if bool(getattr(args, "split_context_prompts", False)):
        command.append("--split-context-prompts")
    if bool(getattr(args, "respect_eos", False)):
        command.append("--respect-eos")
    if bool(getattr(args, "outputs_include_text", False)):
        command.append("--outputs-include-text")
    if bool(getattr(args, "chat_template", False)):
        command.append("--chat-template")
    if bool(getattr(args, "enable_thinking", False)):
        command.append("--enable-thinking")
    return command


def _build_env(
    args: argparse.Namespace,
    *,
    route_trace_path: Path,
    hook_profile_path: Path | None = None,
    rrp_profile_path: Path | None = None,
) -> dict[str, str]:
    env = dict(os.environ)
    repo_root = str(_repo_root())
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        repo_root if not existing_pythonpath else os.pathsep.join((repo_root, existing_pythonpath))
    )
    env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    env["VLLM_SPARSE_FA3_UPSTREAM_ROOT"] = str(args.fa3_upstream_root)
    env["VLLM_SPARSE_ATTENTION_IN_CUDAGRAPH"] = "1"
    env["VLLM_SPARSE_ASYNC_REFRESH"] = "0"
    env["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
    env["VLLM_FLASH_ATTN_VERSION"] = "3"
    env["VLLM_SPARSE_FA3_ROUTE_TRACE_LOG"] = str(route_trace_path)
    if hook_profile_path is not None:
        env["VLLM_SPARSE_FULL_CUDAGRAPH_HOOK_PROFILE_LOG"] = str(hook_profile_path)
    if rrp_profile_path is not None:
        env["VLLM_SPARSE_RRP_PREP_PROFILE_LOG"] = str(rrp_profile_path)
    env["VLLM_SPARSE_CONTROLLER_JSON"] = json.dumps(
        _sparse_controller_payload(args),
        sort_keys=True,
    )
    return env


def _build_dense_reference_env(args: argparse.Namespace) -> dict[str, str]:
    env = dict(os.environ)
    repo_root = str(_repo_root())
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        repo_root if not existing_pythonpath else os.pathsep.join((repo_root, existing_pythonpath))
    )
    env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    for name in (
        "VLLM_SPARSE_FA3_UPSTREAM_ROOT",
        "VLLM_SPARSE_ATTENTION_IN_CUDAGRAPH",
        "VLLM_SPARSE_ASYNC_REFRESH",
        "VLLM_SPARSE_FA3_ROUTE_TRACE_LOG",
        "VLLM_SPARSE_CONTROLLER_JSON",
    ):
        env.pop(name, None)
    env["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN_VLLM_V1"
    env["VLLM_FLASH_ATTN_VERSION"] = "3"
    return env


def _run_command(
    command: list[str],
    *,
    env: dict[str, str],
    timeout_s: int,
) -> Phase1CommandResult:
    proc = subprocess.Popen(
        command,
        cwd=str(_repo_root()),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=float(timeout_s))
        return Phase1CommandResult(
            command=command,
            returncode=int(proc.returncode or 0),
            stdout=stdout,
            stderr=stderr,
            timed_out=False,
        )
    except subprocess.TimeoutExpired:
        _terminate_process_tree(proc, sig=signal.SIGTERM)
        try:
            stdout, stderr = proc.communicate(timeout=10.0)
        except subprocess.TimeoutExpired:
            _terminate_process_tree(proc, sig=signal.SIGKILL)
            stdout, stderr = proc.communicate()
        return Phase1CommandResult(
            command=command,
            returncode=124,
            stdout=stdout,
            stderr=stderr,
            timed_out=True,
        )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _canonical_output_records(path: Path) -> list[dict[str, Any]]:
    payload = _read_json(path)
    rows: list[dict[str, Any]] = []
    for key in sorted(payload):
        value = payload[key]
        if isinstance(value, list):
            rows.append({"token_ids": [int(token_id) for token_id in value]})
        elif isinstance(value, dict) and isinstance(value.get("token_ids"), list):
            rows.append(
                {
                    "token_ids": [int(token_id) for token_id in value["token_ids"]],
                    "text": str(value.get("text", "")),
                }
            )
    return rows


def _canonical_output_token_lists(path: Path) -> list[list[int]]:
    return [
        [int(token_id) for token_id in record["token_ids"]]
        for record in _canonical_output_records(path)
    ]


def _semantic_output_diffs(
    sparse_records: list[dict[str, Any]],
    dense_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(sparse_records) != len(dense_records):
        return []
    if not sparse_records:
        return []
    if any("text" not in record for record in sparse_records + dense_records):
        return []

    from benchmarks.needle_bs2_compare_utils import semantic_match

    diffs: list[dict[str, Any]] = []
    for idx, (sparse_record, dense_record) in enumerate(
        zip(sparse_records, dense_records)
    ):
        ok, mode, ref_semantic, test_semantic = semantic_match(
            str(dense_record.get("text", "")),
            str(sparse_record.get("text", "")),
        )
        dense_tokens = list(dense_record.get("token_ids", []))
        sparse_tokens = list(sparse_record.get("token_ids", []))
        diffs.append(
            {
                "prompt_index": idx,
                "semantic_match": bool(ok),
                "semantic_mode": mode,
                "ref_semantic": ref_semantic,
                "test_semantic": test_semantic,
                "token_match": dense_tokens == sparse_tokens,
                "len_dense": len(dense_tokens),
                "len_sparse": len(sparse_tokens),
            }
        )
    return diffs


class _RouteTraceEvents(list):
    """route trace 事件列表载体:附带解析诊断(坏行数/来源路径)。

    [ROUTE-TRACE-PID-NORM 2026-07-11] list 子类=对既有消费者完全透明
    (迭代/len/切片/拼接/json 序列化均为 list 语义);仅 _route_summary 的
    pid 归一读取附加属性,用于把多进程 O_APPEND 长行撕裂的坏行数量转成
    组间一致性断言的容忍上界(撕裂丢行才可解释组间小差异)。
    """

    __slots__ = ("bad_line_count", "source_path")

    def __init__(
        self,
        events: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
        *,
        bad_line_count: int = 0,
        source_path: str = "",
    ) -> None:
        super().__init__(events)
        self.bad_line_count = int(bad_line_count)
        self.source_path = str(source_path)


def _read_trace_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return _RouteTraceEvents((), bad_line_count=0, source_path=str(path))
    events: list[dict[str, Any]] = []
    bad_line_count = 0
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except Exception:
                # 多进程 O_APPEND 撕裂/截断行:保持旧行为跳过,但如实计数,
                # 供 pid 归一把组间差异归因到撕裂(而非静默吞掉)。
                bad_line_count += 1
                continue
            if isinstance(payload, dict):
                events.append(payload)
            else:
                bad_line_count += 1
    return _RouteTraceEvents(
        events,
        bad_line_count=bad_line_count,
        source_path=str(path),
    )


def _read_jsonl_events(path: Path) -> list[dict[str, Any]]:
    return _read_trace_events(path)


def _refresh_reason_key(reason: str) -> str:
    """refresh_reason 归桶(sentence/interval;period 系归 interval)。"""
    if "sentence" in reason:
        return "sentence"
    if "interval" in reason:
        return "interval"
    if "period" in reason:
        return "interval"
    return reason


_ROUTE_TRACE_ENQUEUE_EVENT = "mixed_page_full_cudagraph_replay_refresh_payload_enqueue"


def _route_trace_event_pid(event: dict[str, Any]) -> int | None:
    pid = event.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int):
        return None
    return int(pid)


def _route_trace_event_signature_key(event: dict[str, Any]) -> tuple[str, str, str]:
    """组间一致性签名键:事件名+callable+(enqueue 事件的 reason 桶)。

    只比"事件种类计数"不比事件内容——各 rank 的时间戳/环计数等字段天然
    不同,但 SPMD 无 rank 门下事件种类与次数必须逐组一致。reason 桶纳入
    签名,保证 counts 判读(sentence/interval)所依赖的分布也被一致性
    断言覆盖。
    """
    name = str(event.get("event", "") or "")
    callable_name = str(event.get("callable", "") or "")
    reason_key = ""
    if name == _ROUTE_TRACE_ENQUEUE_EVENT:
        reason_key = _refresh_reason_key(
            str(event.get("refresh_reason", "") or "unknown")
        )
    return (name, callable_name, reason_key)


def _normalize_route_trace_events_by_pid(
    events: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """[ROUTE-TRACE-PID-NORM 2026-07-11] TP>1 多进程 8× 计数通胀的读侧根修。

    TP>1 时每个 EngineCore worker 进程都把同一 refresh 世代事件 append 进
    同一份 route trace(发射点无 rank 门,patch_installer/_fa3_route_trace_
    enabled),逐行累加=一切 route 计数 ×TP 通胀(远端实证 replay_refresh
    16376/8=2047≈decode 步数)。本函数把聚合语义收回单 controller 口径:

    - 无 pid 事件(bench_measure_marker 等驱动进程直写 marker,不经
      append_fa3_route_trace)原位直通,不参与分组;
    - ≤1 个 pid 组=TP1 形态:原对象原样返回(本地判速/黄金/counts 判据链
      逐位不变的硬合同);
    - ≥2 个 pid 组:先断言组间"事件种类×次数"签名一致,再取首现 pid 组为
      计数源(单 controller 口径);组间不一致仅当可归因于 O_APPEND 长行
      撕裂(spread ≤ 2×坏行数,一条撕裂行至多毁两条事件)时容忍并告警、
      改取事件最全的组,否则 fail-loud 抛错——绝不静默平均。
    """
    bad_line_count = max(0, int(getattr(events, "bad_line_count", 0) or 0))
    source_path = str(getattr(events, "source_path", "") or "")
    groups: dict[int, list[dict[str, Any]]] = {}
    pidless_count = 0
    for event in events:
        pid = _route_trace_event_pid(event)
        if pid is None:
            pidless_count += 1
            continue
        groups.setdefault(pid, []).append(event)
    diag: dict[str, Any] = {
        "route_trace_pid_count": len(groups),
        "route_trace_pidless_event_count": int(pidless_count),
        "route_trace_bad_line_count": int(bad_line_count),
        "route_trace_pid_consistency": "single",
        "route_trace_canonical_pid": next(iter(groups), -1),
        "route_trace_pid_group_spread": 0,
        "route_trace_dedup_dropped_event_count": 0,
    }
    if len(groups) <= 1:
        return events, diag
    signatures: dict[int, Counter] = {
        pid: Counter(_route_trace_event_signature_key(event) for event in group)
        for pid, group in groups.items()
    }
    pids_in_order = list(groups)
    first_signature = signatures[pids_in_order[0]]
    if all(signatures[pid] == first_signature for pid in pids_in_order[1:]):
        canonical_pid = pids_in_order[0]
        consistency = "consistent"
        spread = 0
    else:
        signature_keys: set[tuple[str, str, str]] = set()
        for signature in signatures.values():
            signature_keys.update(signature)
        spread = sum(
            max(signatures[pid].get(key, 0) for pid in pids_in_order)
            - min(signatures[pid].get(key, 0) for pid in pids_in_order)
            for key in signature_keys
        )
        divergent_rows = []
        for key in sorted(signature_keys):
            per_pid = {pid: signatures[pid].get(key, 0) for pid in pids_in_order}
            if len(set(per_pid.values())) > 1:
                divergent_rows.append(f"{key}: {per_pid}")
        if spread > 2 * bad_line_count:
            detail = "\n  ".join(divergent_rows[:12])
            raise RuntimeError(
                "[ROUTE-TRACE-PID-NORM] 多进程 route trace 组间事件计数不一致,"
                f"且超出撕裂容忍上界(spread={spread} > 2×bad_line_count="
                f"{2 * bad_line_count}):workers 逻辑发散或 trace 被外部污染,"
                "判读中止(fail-loud,拒绝静默平均)。"
                f" source={source_path or '<unknown>'}"
                f" pids={pids_in_order}"
                f" totals={ {pid: sum(signatures[pid].values()) for pid in pids_in_order} }\n"
                f"  发散键(最多 12 条):\n  {detail}"
            )
        canonical_pid = max(pids_in_order, key=lambda pid: len(groups[pid]))
        consistency = "torn_tolerated"
        print(
            "[ROUTE-TRACE-PID-NORM][WARN] 组间计数差异已归因 O_APPEND 撕裂"
            f"(spread={spread} ≤ 2×bad_line_count={2 * bad_line_count}),"
            f"取事件最全组 pid={canonical_pid} 为计数源。"
            f" source={source_path or '<unknown>'}"
            f" 发散键:{'; '.join(divergent_rows[:4])}",
            file=sys.stderr,
        )
    normalized = [
        event
        for event in events
        if (_route_trace_event_pid(event) is None)
        or (_route_trace_event_pid(event) == canonical_pid)
    ]
    diag.update(
        {
            "route_trace_pid_consistency": consistency,
            "route_trace_canonical_pid": int(canonical_pid),
            "route_trace_pid_group_spread": int(spread),
            "route_trace_dedup_dropped_event_count": len(events) - len(normalized),
        }
    )
    return (
        _RouteTraceEvents(
            normalized,
            bad_line_count=bad_line_count,
            source_path=source_path,
        ),
        diag,
    )


def _row_mode_distribution(events: list[dict[str, Any]]) -> dict[str, int]:
    compact = 0
    native = 0
    for event in events:
        distribution = event.get("row_mode_distribution")
        if isinstance(distribution, dict):
            compact += int(distribution.get("compact", 0) or 0)
            native += int(distribution.get("native", 0) or 0)
            continue
        if event.get("event") != "mixed_page_call":
            continue
        row_modes = event.get("row_is_compact")
        if isinstance(row_modes, list):
            for value in row_modes:
                if bool(value):
                    compact += 1
                else:
                    native += 1
    if compact == 0 and native == 0:
        return {}
    return {"compact": compact, "native": native}


def _row_source_distribution(events: list[dict[str, Any]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for event in events:
        distribution = event.get("row_source_distribution")
        if not isinstance(distribution, dict):
            continue
        for key, value in distribution.items():
            totals[str(key)] = totals.get(str(key), 0) + int(value or 0)
    return totals


def _source_counter_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    missing_fields: list[str] = []
    expected_rows = 0
    expected_rows_seen = False
    schema_version: int | None = None
    num_kv_heads: int | None = None
    source_event_seen = False
    for event in events:
        has_source_payload = any(
            key in event
            for key in (
                "source_counter_schema_version",
                "expected_rows",
                "num_kv_heads",
                "source_counter_missing_fields",
            )
        ) or isinstance(event.get("row_source_distribution"), dict)
        if not has_source_payload:
            continue
        source_event_seen = True

        if "source_counter_schema_version" in event:
            event_schema_version = _as_int(
                event.get("source_counter_schema_version"),
                -1,
            )
            if schema_version is None:
                schema_version = event_schema_version
            elif schema_version != event_schema_version:
                missing_fields.append("source_counter_schema_version_mismatch")
            if event_schema_version != STAGE_A_SOURCE_COUNTER_SCHEMA_VERSION:
                missing_fields.append("source_counter_schema_version_unsupported")

        if "num_kv_heads" in event:
            event_num_kv_heads = _as_int(event.get("num_kv_heads"), -1)
            if num_kv_heads is None:
                num_kv_heads = event_num_kv_heads
            elif num_kv_heads != event_num_kv_heads:
                missing_fields.append("num_kv_heads_mismatch")

        if "expected_rows" in event:
            expected_rows += _as_int(event.get("expected_rows"), 0)
            expected_rows_seen = True

        fields = event.get("source_counter_missing_fields")
        if isinstance(fields, str) and fields:
            missing_fields.append(fields)
        elif isinstance(fields, (list, tuple)):
            missing_fields.extend(str(field) for field in fields if str(field))
    if source_event_seen:
        if schema_version is None:
            missing_fields.append("source_counter_schema_version")
        if num_kv_heads is None:
            missing_fields.append("num_kv_heads")
        if not expected_rows_seen:
            missing_fields.append("expected_rows")
    if schema_version is not None:
        summary["source_counter_schema_version"] = schema_version
    if expected_rows_seen:
        summary["expected_rows"] = expected_rows
    if num_kv_heads is not None:
        summary["num_kv_heads"] = num_kv_heads
    if missing_fields:
        summary["source_counter_missing_fields"] = sorted(set(missing_fields))
    elif source_event_seen:
        summary["source_counter_missing_fields"] = []
    return summary


def _fallback_evidence_summary(events: list[dict[str, Any]]) -> dict[str, int]:
    triton_attention_call_count = 0
    dense_fallback_call_count = 0
    native_full_block_table_rows = 0
    page_resolver_kind4_count = 0
    kind2_dispatch_count = 0
    kind3_dispatch_count = 0
    selected_table_publish_count = 0
    vector_fallback_rows = 0
    full_fallback_rows = 0
    for event in events:
        triton_attention_call_count += int(
            event.get("triton_attention_call_count", 0) or 0
        )
        dense_fallback_call_count += int(
            event.get("dense_fallback_call_count", 0) or 0
        )
        native_full_block_table_rows += int(
            event.get("native_full_block_table_rows", 0) or 0
        )
        page_resolver_kind4_count += int(
            event.get("page_resolver_kind4_count", 0) or 0
        )
        kind2_dispatch_count += int(event.get("kind2_dispatch_count", 0) or 0)
        kind3_dispatch_count += int(event.get("kind3_dispatch_count", 0) or 0)
        selected_table_publish_count += int(
            event.get("selected_table_publish_count", 0) or 0
        )
        vector_fallback_rows += int(event.get("vector_fallback_rows", 0) or 0)
        full_fallback_rows += int(event.get("full_fallback_rows", 0) or 0)
        row_sources = event.get("row_source_distribution")
        if isinstance(row_sources, dict):
            native_full_block_table_rows += int(
                row_sources.get("native_full_block_table_rows", 0) or 0
            )
            vector_fallback_rows += int(row_sources.get("vector_fallback_rows", 0) or 0)
            full_fallback_rows += int(row_sources.get("full_fallback_rows", 0) or 0)

        event_name = str(event.get("event", ""))
        route = str(event.get("route", ""))
        callable_name = str(event.get("callable", ""))
        if event_name in {"triton_attention_call", "triton_attention_forward"}:
            triton_attention_call_count += 1
        elif route in {"triton_attention", "triton_attention_forward"}:
            triton_attention_call_count += 1
        elif callable_name in {"triton_attention", "triton_attention_forward"}:
            triton_attention_call_count += 1

        fallback = str(event.get("fallback", ""))
        fallback_target = str(event.get("fallback_target", ""))
        if event_name in {"dense_fallback", "dense_attention_fallback"}:
            dense_fallback_call_count += 1
        elif bool(event.get("dense_fallback", False)):
            dense_fallback_call_count += 1
        elif fallback in {"dense", "dense_fallback", "old_v1_flash"}:
            dense_fallback_call_count += 1
        elif fallback_target in {"dense", "dense_fallback", "old_v1_flash"}:
            dense_fallback_call_count += 1
        page_resolver_kind = int(event.get("page_resolver_kind", -1) or -1)
        if page_resolver_kind == 4:
            page_resolver_kind4_count += 1
        elif page_resolver_kind == 2:
            kind2_dispatch_count += 1
        elif page_resolver_kind == 3:
            kind3_dispatch_count += 1
        elif page_resolver_kind == 1:
            selected_table_publish_count += 1
    return {
        "triton_attention_call_count": triton_attention_call_count,
        "dense_fallback_call_count": dense_fallback_call_count,
        "native_full_block_table_rows": native_full_block_table_rows,
        "page_resolver_kind4_count": page_resolver_kind4_count,
        "kind2_dispatch_count": kind2_dispatch_count,
        "kind3_dispatch_count": kind3_dispatch_count,
        "selected_table_publish_count": selected_table_publish_count,
        "vector_fallback_rows": vector_fallback_rows,
        "full_fallback_rows": full_fallback_rows,
    }


def _compact_page_residency_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    bind_events = [
        event
        for event in events
        if event.get("event") == "compact_page_residency_bind"
    ]
    owners: set[str] = set()
    owner_missing_count = 0
    storage_mismatch_count = 0
    reserved_span_mismatch_count = 0
    copy_bytes = 0
    copy_bytes_missing = False
    for event in bind_events:
        owner = str(event.get("compact_kv_storage_owner", "") or "")
        if owner:
            owners.add(owner)
        else:
            owner_missing_count += 1
            owners.add("missing")

        k_storage_ok = event.get("compact_arena_k_storage_matches_layer_kv_cache")
        v_storage_ok = event.get("compact_arena_v_storage_matches_layer_kv_cache")
        if k_storage_ok is not True or v_storage_ok is not True:
            storage_mismatch_count += 1

        k_span_ok = event.get("compact_arena_k_data_ptr_in_reserved_span")
        v_span_ok = event.get("compact_arena_v_data_ptr_in_reserved_span")
        if k_span_ok is not True or v_span_ok is not True:
            reserved_span_mismatch_count += 1

        if "compact_to_compact_copy_bytes" not in event:
            copy_bytes_missing = True
            continue
        try:
            copy_bytes += int(event.get("compact_to_compact_copy_bytes", 0) or 0)
        except (TypeError, ValueError):
            copy_bytes_missing = True

    owner = owners.pop() if len(owners) == 1 else "mixed" if owners else ""
    return {
        "compact_kv_storage_owner": owner,
        "compact_kv_storage_owner_missing_count": owner_missing_count,
        "compact_kv_native_residency_bind_count": len(bind_events),
        "compact_kv_storage_mismatch_count": storage_mismatch_count,
        "compact_kv_reserved_span_mismatch_count": reserved_span_mismatch_count,
        "compact_to_compact_copy_bytes": -1 if copy_bytes_missing else copy_bytes,
    }


def _replay_budget_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    replay_prehook_us = -1.0
    carrier_update_kernel_count = -1
    rrp_signature_hits = 0
    rrp_signature_total = 0
    rrp_full_bind_count = -1
    for event in events:
        if event.get("event") == "mixed_page_full_cudagraph_replay_refresh":
            replay_prehook_us = max(
                replay_prehook_us,
                _as_float(event.get("elapsed_us"), -1.0),
            )
            carrier_update_kernel_count = max(
                carrier_update_kernel_count,
                _as_int(event.get("carrier_update_kernel_count"), -1),
            )
        if event.get("event") == "rrp_replay_metadata_bind":
            rrp_signature_total += 1
            if bool(event.get("rrp_signature_hit", False)):
                rrp_signature_hits += 1
            if rrp_full_bind_count < 0:
                rrp_full_bind_count = 0
            if bool(event.get("rrp_full_bind", False)):
                rrp_full_bind_count += 1
    rrp_signature_hit_rate = (
        float(rrp_signature_hits) / float(rrp_signature_total)
        if rrp_signature_total > 0
        else -1.0
    )
    return {
        "replay_prehook_us": replay_prehook_us,
        "carrier_update_kernel_count": carrier_update_kernel_count,
        "rrp_signature_hit_rate": rrp_signature_hit_rate,
        "rrp_full_bind_count": rrp_full_bind_count,
    }


def _graph_route_family_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    hook_events = [
        event
        for event in events
        if event.get("event") == "mixed_page_full_cudagraph_replay_hook_check"
    ]
    explicit_events = [
        event
        for event in hook_events
        if "route_family_mismatch" in event
        and event.get("route_family_mismatch") is not None
    ]
    if not explicit_events:
        return {}
    route_family_mismatch = any(
        bool(event.get("route_family_mismatch")) for event in explicit_events
    )
    last = explicit_events[-1]
    return {
        "captured_route_family": str(last.get("captured_route_family", "") or ""),
        "current_route_family": str(last.get("current_route_family", "") or ""),
        "route_family_mismatch": bool(route_family_mismatch),
    }


def _int_list(value: Any) -> list[int]:
    if not isinstance(value, (list, tuple)):
        return []
    out: list[int] = []
    for item in value:
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            continue
    return out


def _bridge_token_positions_by_request(
    bridge_events: list[dict[str, Any]],
    ready_events: list[dict[str, Any]],
) -> dict[str, list[int]]:
    positions_by_request: dict[str, list[int]] = {}
    for event in bridge_events:
        rid = str(event.get("request_id", "") or "")
        if not rid:
            continue
        position = _as_int(event.get("bridge_token_position"), -1)
        if position >= 0:
            positions_by_request.setdefault(rid, []).append(int(position))
    if positions_by_request:
        return positions_by_request
    for event in ready_events:
        rid = str(event.get("request_id", "") or "")
        if not rid:
            continue
        positions = _int_list(event.get("bridge_token_positions"))
        if positions:
            positions_by_request[rid] = positions
    return positions_by_request


def _bridge_token_counts_by_request(
    bridge_events: list[dict[str, Any]],
    ready_events: list[dict[str, Any]],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in bridge_events + ready_events:
        rid = str(event.get("request_id", "") or "")
        if not rid:
            continue
        count = _as_int(event.get("bridge_token_count"), -1)
        if count >= 0:
            counts[rid] = max(counts.get(rid, -1), int(count))
    return counts


def _bridge_positions_exact_once(
    positions_by_request: dict[str, list[int]],
    counts_by_request: dict[str, int],
) -> bool | None:
    if not positions_by_request or not counts_by_request:
        return None
    for rid, count in counts_by_request.items():
        positions = positions_by_request.get(rid, [])
        if count <= 0 or len(positions) != count:
            return False
        if len(set(positions)) != len(positions):
            return False
    return True


def _event_row_for_request(event: dict[str, Any], request_id: str) -> int:
    req_ids: list[str] = []
    raw_req_ids = event.get("req_ids_by_row")
    if isinstance(raw_req_ids, (list, tuple)):
        req_ids = [str(v) for v in raw_req_ids]
    if req_ids:
        try:
            return req_ids.index(str(request_id))
        except ValueError:
            return -1
    recent_first = _int_list(event.get("recent_first_tokens_by_row"))
    return 0 if len(recent_first) == 1 else -1


def _positions_in_post_switch_recent_tail(
    positions_by_request: dict[str, list[int]],
    post_switch_events: list[dict[str, Any]],
) -> bool | None:
    if not positions_by_request:
        return None
    if not post_switch_events:
        return False
    covered_requests: set[str] = set()
    for event in post_switch_events:
        row_sources = event.get("row_source_distribution") or {}
        if not isinstance(row_sources, dict):
            continue
        if int(row_sources.get("middle_native_canonical_pages", 0) or 0) != 0:
            continue
        if int(row_sources.get("compact_full_native_fallback_rows", 0) or 0) != 0:
            continue
        recent_first_tokens = _int_list(event.get("recent_first_tokens_by_row"))
        request_recent_lens = _int_list(event.get("request_recent_len_by_row"))
        if not recent_first_tokens or not request_recent_lens:
            continue
        for rid, positions in positions_by_request.items():
            if rid in covered_requests:
                continue
            row = _event_row_for_request(event, rid)
            if row < 0 or row >= len(recent_first_tokens):
                continue
            recent_start = int(recent_first_tokens[row])
            recent_len = int(request_recent_lens[row])
            if recent_len <= 0:
                continue
            recent_end = recent_start + recent_len
            if all(recent_start <= int(pos) < recent_end for pos in positions):
                covered_requests.add(rid)
        if covered_requests == set(positions_by_request):
            return True
    return False


def _deferred_bridge_summary(
    events: list[dict[str, Any]],
    *,
    replay_refresh_events: list[dict[str, Any]],
) -> dict[str, Any]:
    bridge_events = [
        event
        for event in events
        if event.get("event") == "deferred_bridge_decode_metadata_accepted"
    ]
    ready_events = [
        event
        for event in events
        if event.get("event") == "deferred_bridge_producer_ready"
    ]
    launch_events = [
        event
        for event in events
        if event.get("event") == "deferred_bridge_producer_launch"
    ]
    launch_step = max(
        (
            _as_int(event.get("producer_launch_step"), -1)
            for event in bridge_events + ready_events + launch_events
        ),
        default=-1,
    )
    bridge_token_count = max(
        (
            _as_int(event.get("bridge_token_count"), -1)
            for event in bridge_events + ready_events
        ),
        default=-1,
    )
    producer_ready_step = max(
        (_as_int(event.get("producer_ready_step"), -1) for event in ready_events),
        default=-1,
    )

    phase_summaries: dict[str, Any] = {}
    bridge_count = sum(
        max(1, _as_int(event.get("accepted_count"), 1)) for event in bridge_events
    )
    if bridge_count > 0:
        phase_summaries["bridge_phase"] = {
            "row_mode_distribution": {"full_kv": int(bridge_count)},
            "row_source_distribution": {"full_kv_rows": int(bridge_count)},
        }

    post_switch_position_events = [
        event
        for event in replay_refresh_events
        if int((event.get("row_mode_distribution") or {}).get("compact", 0) or 0) > 0
    ]
    post_switch_events = [
        event
        for event in post_switch_position_events
        if int((event.get("row_mode_distribution") or {}).get("native", 0) or 0)
        == 0
    ]
    if post_switch_events:
        phase_summaries["post_switch_phase"] = {
            "row_mode_distribution": _row_mode_distribution(post_switch_events),
            "row_source_distribution": _row_source_distribution(post_switch_events),
            **_source_counter_summary(post_switch_events),
        }
    positions_by_request = _bridge_token_positions_by_request(
        bridge_events,
        ready_events,
    )
    counts_by_request = _bridge_token_counts_by_request(bridge_events, ready_events)
    bridge_token_positions_exact_once = _bridge_positions_exact_once(
        positions_by_request,
        counts_by_request,
    )
    compact_middle_excludes_bridge_positions = (
        _positions_in_post_switch_recent_tail(
            positions_by_request,
            post_switch_position_events,
        )
        if bridge_token_positions_exact_once is True
        else None
    )

    summary: dict[str, Any] = {}
    if bridge_token_count >= 0:
        summary["bridge_token_count"] = int(bridge_token_count)
    if launch_step >= 0:
        summary["producer_launch_step"] = int(launch_step)
    if producer_ready_step >= 0:
        summary["producer_ready_step"] = int(producer_ready_step)
    if positions_by_request:
        summary["bridge_token_positions_by_request"] = positions_by_request
    if bridge_token_positions_exact_once is not None:
        summary["bridge_token_positions_exact_once"] = bool(
            bridge_token_positions_exact_once
        )
    if compact_middle_excludes_bridge_positions is not None:
        summary["compact_middle_excludes_bridge_positions"] = bool(
            compact_middle_excludes_bridge_positions
        )
    if phase_summaries:
        summary["phase_summaries"] = phase_summaries
    if bridge_events or ready_events:
        summary["bootstrap_full_kv_handoff"] = True
    if any(
        bool(event.get("deferred_bridge_diagnostic_only"))
        for event in bridge_events + ready_events + launch_events
    ):
        summary["deferred_bridge_diagnostic_only"] = True
    return summary


def _arena_trace_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    arena_events = [
        event
        for event in events
        if event.get("event") in {
            "prefill_global_meta_build_detail",
            "capture_layout_worker_detail",
        }
    ]
    if not arena_events:
        return {}

    summary: dict[str, Any] = {}
    int_fields = (
        "arena_reserved_bytes",
        "arena_peak_bytes",
        "arena_bucket_bytes",
        "arena_bucket_count",
        "arena_largest_bucket_bytes",
        "arena_expansion_bytes",
        "arena_prepare_miss_count",
        "capture_layout_hot_path_alloc_count",
        "capture_layout_new_count",
        "hot_path_d2h_count",
        "hot_path_cuda_sync_count",
        "tail_path_item_cpu_count",
        "bridge_fallback_count",
        "bridged_token_count",
    )
    for key in int_fields:
        values = [_as_int(event.get(key), -1) for event in arena_events]
        values = [value for value in values if value >= 0]
        if values:
            summary[key] = int(max(values))

    bool_fields = (
        "arena_budget_exceeded",
        "hidden_contention_miss",
        "arena_ready_before_tail",
    )
    for key in bool_fields:
        present = [event.get(key) for event in arena_events if key in event]
        if present:
            summary[key] = any(bool(value) for value in present)

    float_fields = ("arena_prepare_wait_us",)
    for key in float_fields:
        values = [_as_float(event.get(key), -1.0) for event in arena_events]
        values = [value for value in values if value >= 0.0]
        if values:
            summary[key] = float(max(values))

    for key in ("arena_bind_status", "arena_reservation_status"):
        values = [str(event.get(key, "") or "") for event in arena_events]
        values = [value for value in values if value]
        if values:
            summary[key] = values[-1]

    built_events = [
        event
        for event in arena_events
        if event.get("event") == "prefill_global_meta_build_detail"
        and event.get("status") == "built"
    ]
    if built_events:
        total_us_values = [
            _as_float(event.get("total_us"), -1.0) for event in built_events
        ]
        total_us_values = [value for value in total_us_values if value >= 0.0]
        if total_us_values:
            summary["prefill_global_meta_build_us"] = float(max(total_us_values))
        capture_layout_values: list[float] = []
        buffer_prepare_values: list[float] = []
        for event in built_events:
            phase_us = event.get("phase_us")
            if not isinstance(phase_us, dict):
                continue
            capture_layout_values.append(
                _as_float(phase_us.get("capture_layout"), -1.0)
            )
            buffer_prepare_values.append(
                _as_float(phase_us.get("buffer_prepare"), -1.0)
            )
        capture_layout_values = [
            value for value in capture_layout_values if value >= 0.0
        ]
        buffer_prepare_values = [
            value for value in buffer_prepare_values if value >= 0.0
        ]
        if capture_layout_values:
            summary["prefill_global_meta_capture_layout_us"] = float(
                max(capture_layout_values)
            )
        if buffer_prepare_values:
            summary["prefill_global_meta_buffer_prepare_us"] = float(
                max(buffer_prepare_values)
            )

    summary["prefill_global_meta_detail_event_count"] = int(len(arena_events))
    return summary



def _rrp_visible_source_proof_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    best_event: dict[str, Any] | None = None
    best_score = -1
    for event in events:
        kind = str(event.get("rrp_visible_source_kind", "") or "")
        if not kind:
            continue
        covers = bool(event.get("rrp_sparse_dynamic_state_covers_rows", False))
        failure_reason = str(
            event.get("rrp_sparse_dynamic_state_failure_reason", "") or ""
        )
        if kind == "sparse_dynamic_state":
            score = 30
        elif kind == "dense_seqused":
            score = 20
        elif kind == "arena_batch_seqused":
            score = 10
        elif kind == "arena_seqused":
            score = 5
        else:
            score = 1
        if covers:
            score += 2
        if bool(event.get("rrp_launch_effective_covers_rows", False)):
            score += 1
        if _int_list(event.get("rrp_row_effective_k_by_row")):
            score += 1
        if not failure_reason:
            score += 1
        if score >= best_score:
            best_score = score
            best_event = event
    if best_event is None:
        return {}
    return {
        "rrp_visible_source_kind": str(
            best_event.get("rrp_visible_source_kind", "") or ""
        ),
        "rrp_sparse_dynamic_state_covers_rows": bool(
            best_event.get("rrp_sparse_dynamic_state_covers_rows", False)
        ),
        "rrp_sparse_dynamic_state_failure_reason": str(
            best_event.get("rrp_sparse_dynamic_state_failure_reason", "") or ""
        ),
        "rrp_visible_data_ptr": _as_int(
            best_event.get("rrp_visible_data_ptr", 0), 0
        ),
        "rrp_sparse_dynamic_state_data_ptr": _as_int(
            best_event.get("rrp_sparse_dynamic_state_data_ptr", 0), 0
        ),
        "rrp_visible_shape": _int_list(best_event.get("rrp_visible_shape")),
        "rrp_visible_is_arena_seqused": bool(
            best_event.get("rrp_visible_is_arena_seqused", False)
        ),
        "rrp_visible_is_arena_batch_seqused": bool(
            best_event.get("rrp_visible_is_arena_batch_seqused", False)
        ),
        "rrp_visible_is_launch_effective": bool(
            best_event.get("rrp_visible_is_launch_effective", False)
        ),
        "rrp_visible_is_dense_seqused": bool(
            best_event.get("rrp_visible_is_dense_seqused", False)
        ),
        "rrp_launch_effective_covers_rows": bool(
            best_event.get("rrp_launch_effective_covers_rows", False)
        ),
        "rrp_launch_effective_k_len_cpu": _int_list(
            best_event.get("rrp_launch_effective_k_len_cpu")
        ),
        "rrp_row_effective_k_by_row": _int_list(
            best_event.get("rrp_row_effective_k_by_row")
        ),
    }

def _route_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    # [ROUTE-TRACE-PID-NORM 2026-07-11] 聚合唯一咽喉:TP>1 多进程重复 append
    # 的读侧归一(单 controller 口径)。TP1(≤1 pid 组)原对象直通=下游全部
    # 既有字段逐位不变;诊断键 route_trace_* 纯增量。
    events, route_trace_pid_diag = _normalize_route_trace_events_by_pid(events)
    wrapper_events = [
        event
        for event in events
        if event.get("callable") == "mixed_page_attn_varlen_func"
        or event.get("event") == "capture_mixed_owner_dispatch"
        or (
            event.get("event") == "flash_attention_forward"
            and event.get("route") == "mixed_page_attn_varlen_func"
        )
    ]
    legacy_mixed_page_events = [
        event
        for event in events
        if event.get("event") == "mixed_page_call"
    ]
    fwd_mixed_page_events = [
        event
        for event in events
        if event.get("event") == "fwd_mixed_page_call"
    ]
    mixed_page_events = legacy_mixed_page_events + fwd_mixed_page_events
    resolved_row_ptr_events = [
        event
        for event in mixed_page_events
        if event.get("mode") == "resolved_row_ptr"
        or int(event.get("page_resolver_kind", -1) or -1) == 4
    ]
    resolved_row_ptr_fwd_mixed_page_events = [
        event
        for event in fwd_mixed_page_events
        if int(event.get("page_resolver_kind", -1) or -1) == 4
    ]
    replay_refresh_events = [
        event
        for event in events
        if event.get("event") == "mixed_page_full_cudagraph_replay_refresh"
    ]
    replay_refresh_payload_enqueue_events = [
        event
        for event in events
        if event.get("event")
        == "mixed_page_full_cudagraph_replay_refresh_payload_enqueue"
    ]
    refresh_reason_counts: dict[str, int] = {}
    for event in replay_refresh_payload_enqueue_events:
        reason = str(event.get("refresh_reason", "") or "unknown")
        reason_key = _refresh_reason_key(reason)
        refresh_reason_counts[reason_key] = refresh_reason_counts.get(reason_key, 0) + max(
            1,
            _as_int(event.get("refresh_intent_req_count"), 1),
        )
    sentence_trigger_intents = sum(
        max(1, _as_int(event.get("refresh_intent_req_count"), 1))
        for event in replay_refresh_payload_enqueue_events
        if "sentence" in str(event.get("refresh_reason", ""))
    )
    graph_route_family_mismatch_events = [
        event
        for event in events
        if event.get("event") == "mixed_page_full_cudagraph_replay_hook_check"
        and "graph_route_family_mismatch" in str(event.get("reason", ""))
    ]
    full_graph_seen = any(
        event.get("graph_key", "").startswith("full:")
        for event in replay_refresh_events
    )
    pointer_unstable_count = sum(
        1
        for event in replay_refresh_events
        if "carrier_pointer_signature_stable" in event
        and not bool(event.get("carrier_pointer_signature_stable", False))
    )
    pointer_missing_count = sum(
        1
        for event in replay_refresh_events
        if "carrier_pointer_signature_stable" not in event
    )
    pointer_stable = bool(
        replay_refresh_events
        and pointer_unstable_count == 0
        and pointer_missing_count == 0
        and all(
            bool(event.get("carrier_pointer_signature_stable", False))
            for event in replay_refresh_events
        )
    )
    graph_route_family = _graph_route_family_summary(events)
    deferred_bridge = _deferred_bridge_summary(
        events,
        replay_refresh_events=replay_refresh_events,
    )
    rrp_visible_source_proof = _rrp_visible_source_proof_summary(events)
    row_mode_distribution = _row_mode_distribution(events)
    row_source_distribution = _row_source_distribution(events)
    source_counter_summary = _source_counter_summary(events)
    phase_summaries = deferred_bridge.get("phase_summaries")
    post_switch_summary = (
        phase_summaries.get("post_switch_phase", {})
        if isinstance(phase_summaries, dict)
        else {}
    )
    if deferred_bridge.get("bootstrap_full_kv_handoff") and post_switch_summary:
        post_switch_row_modes = post_switch_summary.get("row_mode_distribution")
        post_switch_row_sources = post_switch_summary.get("row_source_distribution")
        if isinstance(post_switch_row_modes, dict) and isinstance(
            post_switch_row_sources,
            dict,
        ):
            row_mode_distribution = dict(post_switch_row_modes)
            row_source_distribution = dict(post_switch_row_sources)
            post_switch_events = [
                event
                for event in replay_refresh_events
                if int(
                    (event.get("row_mode_distribution") or {}).get("compact", 0) or 0
                )
                > 0
                and int(
                    (event.get("row_mode_distribution") or {}).get("native", 0) or 0
                )
                == 0
            ]
            source_counter_summary = _source_counter_summary(post_switch_events)
    return {
        "mixed_wrapper_event_count": len(wrapper_events),
        "mixed_page_call_count": len(mixed_page_events),
        "actual_fwd_mixed_page_count": len(fwd_mixed_page_events),
        "resolved_row_ptr_fwd_mixed_page_count": len(
            resolved_row_ptr_fwd_mixed_page_events
        ),
        "resolved_row_ptr_call_count": len(resolved_row_ptr_events),
        "replay_refresh_count": len(replay_refresh_events),
        "replay_refresh_payload_enqueue_count": len(
            replay_refresh_payload_enqueue_events
        ),
        "replay_refresh_payload_enqueue_payloads_total": int(
            sum(
                max(0, _as_int(event.get("payload_count"), 0))
                for event in replay_refresh_payload_enqueue_events
            )
        ),
        "replay_refresh_payload_enqueue_refresh_slots_total": int(
            sum(
                max(0, _as_int(event.get("refresh_slot_count"), 0))
                for event in replay_refresh_payload_enqueue_events
            )
        ),
        "refresh_reason_counts": refresh_reason_counts,
        # [INTENTS-SEMANTICS 口径] 世代 enqueue 计数(非 token-time 意图数),
        # 详见 one_shot_graph_e2e 同名字段注记。
        "interval_trigger_intents": int(refresh_reason_counts.get("interval", 0)),
        "sentence_trigger_intents": int(sentence_trigger_intents),
        "graph_route_family_mismatch_count": len(
            graph_route_family_mismatch_events
        ),
        "full_graph_replay_refresh_seen": full_graph_seen,
        "carrier_pointer_signature_stable": pointer_stable,
        "carrier_pointer_signature_unstable_count": pointer_unstable_count,
        "carrier_pointer_signature_missing_count": pointer_missing_count,
        "graph_route_family": graph_route_family,
        "route_family_mismatch": graph_route_family.get("route_family_mismatch"),
        "row_mode_distribution": row_mode_distribution,
        "row_source_distribution": row_source_distribution,
        **source_counter_summary,
        **_fallback_evidence_summary(events),
        **_compact_page_residency_summary(events),
        **_replay_budget_summary(events),
        **_arena_trace_summary(events),
        **deferred_bridge,
        **rrp_visible_source_proof,
        # [ROUTE-TRACE-PID-NORM] pid 归一诊断(纯增量键;TP1 恒
        # pid_count≤1/consistency=single/spread=0)。
        **route_trace_pid_diag,
    }


def _classify_failure(result: Phase1CommandResult) -> str:
    combined = result.stdout + "\n" + result.stderr
    if result.timed_out:
        return "timeout"
    if "no kernel image is available for execution on the device" in combined:
        return "fa3_sm80_kernel_image_unavailable"
    if "get_scheduler_metadata() expected at most 23 argument(s) but received 24" in combined:
        return "fa3_scheduler_metadata_abi_mismatch"
    if "CUBLAS_STATUS_NOT_INITIALIZED" in combined:
        return "cublas_not_initialized_during_profile"
    if "Engine core initialization failed" in combined:
        return "engine_core_initialization_failed"
    if result.returncode != 0:
        return f"subprocess_returncode_{result.returncode}"
    return ""


def _record_from_result(
    args: argparse.Namespace,
    *,
    result: Phase1CommandResult,
    route_summary: dict[str, Any],
    metrics: dict[str, Any],
    reference_max_abs_diff: float,
    require_budget_fields: bool = False,
    run_pair_id: str = "",
    config_digest: str = "",
    route_proof_result: RouteProofResult | None = None,
) -> Phase1FullGraphRecord:
    subprocess_ok = result.returncode == 0 and not result.timed_out
    full_graph_enabled = bool(route_summary.get("full_graph_replay_refresh_seen", False))
    mixed_wrapper_count = int(route_summary.get("mixed_wrapper_event_count", 0) or 0)
    actual_fwd_count = int(route_summary.get("actual_fwd_mixed_page_count", 0) or 0)
    resolved_row_ptr_fwd_count = int(
        route_summary.get("resolved_row_ptr_fwd_mixed_page_count", 0) or 0
    )
    row_mode_distribution = dict(route_summary.get("row_mode_distribution", {}) or {})
    compact_rows = int(row_mode_distribution.get("compact", 0) or 0)
    replay_refresh_count = int(route_summary.get("replay_refresh_count", 0) or 0)
    carrier_update_us = float(metrics.get("carrier_update_us", -1.0))
    full_graph_replay_us = float(metrics.get("decode_p50_us", -1.0))
    replay_prehook_us = _as_float(
        metrics.get("replay_prehook_us", route_summary.get("replay_prehook_us")),
        -1.0,
    )
    carrier_update_kernel_count = _as_int(
        metrics.get(
            "carrier_update_kernel_count",
            route_summary.get("carrier_update_kernel_count"),
        ),
        -1,
    )
    rrp_signature_hit_rate = _as_float(
        metrics.get(
            "rrp_signature_hit_rate",
            route_summary.get("rrp_signature_hit_rate"),
        ),
        -1.0,
    )
    rrp_full_bind_count = _as_int(
        metrics.get("rrp_full_bind_count", route_summary.get("rrp_full_bind_count")),
        -1,
    )
    budget_fields_ready = bool(
        replay_prehook_us >= 0.0
        and carrier_update_kernel_count >= 0
        and rrp_signature_hit_rate >= 0.0
        and rrp_full_bind_count >= 0
    )
    gate_passed = bool(
        subprocess_ok
        and full_graph_enabled
        and actual_fwd_count > 0
        and resolved_row_ptr_fwd_count > 0
        and compact_rows > 0
        and replay_refresh_count >= 2
        and bool(route_summary.get("carrier_pointer_signature_stable", False))
        and reference_max_abs_diff >= 0.0
        and (budget_fields_ready or not require_budget_fields)
    )
    route_proof_passed = bool(route_proof_result.passed) if route_proof_result else False
    route_proof_reasons = list(route_proof_result.reasons) if route_proof_result else []
    return Phase1FullGraphRecord(
        case=str(args.case),
        full_cudagraph_enabled=full_graph_enabled,
        capture_route=(
            "mixed_page_attn_varlen_func" if mixed_wrapper_count > 0 else "missing"
        ),
        replay_route="full_cudagraph_resolved_row_ptr"
        if resolved_row_ptr_fwd_count > 0 and replay_refresh_count > 0
        else "full_cudagraph_direct_bound_without_resolved_row_ptr"
        if replay_refresh_count > 0
        else "missing",
        resolver_kind=4 if resolved_row_ptr_fwd_count > 0 else -1,
        mixed_wrapper_event_count=mixed_wrapper_count,
        actual_fwd_mixed_page_count=actual_fwd_count,
        resolved_row_ptr_fwd_mixed_page_count=resolved_row_ptr_fwd_count,
        carrier_pointer_signature_stable=bool(
            route_summary.get("carrier_pointer_signature_stable", False)
        ),
        fresh_carrier_trace_changed=replay_refresh_count >= 2,
        row_mode_distribution=row_mode_distribution,
        reference_max_abs_diff=reference_max_abs_diff,
        out_atol=DEFAULT_OUT_ATOL,
        out_rtol=DEFAULT_OUT_RTOL,
        lse_atol=DEFAULT_LSE_ATOL,
        lse_rtol=DEFAULT_LSE_RTOL,
        replay_prep_allocation_free=True,
        replay_prep_no_d2h_sync=True,
        carrier_update_us=carrier_update_us,
        full_graph_replay_us=full_graph_replay_us,
        gate_passed=gate_passed,
        run_pair_id=run_pair_id,
        config_digest=config_digest,
        route_proof_passed=route_proof_passed,
        route_proof_reasons=route_proof_reasons,
        replay_prehook_us=replay_prehook_us,
        carrier_update_kernel_count=carrier_update_kernel_count,
        rrp_signature_hit_rate=rrp_signature_hit_rate,
        rrp_full_bind_count=rrp_full_bind_count,
    )


def _classify_phase1_gate_failure(
    record: Phase1FullGraphRecord,
    route_summary: dict[str, Any],
) -> str:
    if record.gate_passed:
        return ""
    if not record.full_cudagraph_enabled:
        return "full_cudagraph_replay_refresh_missing"
    if int(route_summary.get("mixed_wrapper_event_count", 0) or 0) <= 0:
        return "mixed_wrapper_route_missing"
    if int(route_summary.get("actual_fwd_mixed_page_count", 0) or 0) <= 0:
        return "actual_fwd_mixed_page_missing"
    if int(route_summary.get("resolved_row_ptr_fwd_mixed_page_count", 0) or 0) <= 0:
        return "resolved_row_ptr_fwd_mixed_page_missing"
    if int(record.row_mode_distribution.get("compact", 0) or 0) <= 0:
        return "compact_row_distribution_missing"
    if int(route_summary.get("replay_refresh_count", 0) or 0) < 2:
        return "insufficient_full_graph_replay_refresh"
    if int(route_summary.get("carrier_pointer_signature_missing_count", 0) or 0) > 0:
        return "carrier_pointer_signature_missing"
    if int(route_summary.get("carrier_pointer_signature_unstable_count", 0) or 0) > 0:
        return "carrier_pointer_signature_unstable"
    if not bool(route_summary.get("carrier_pointer_signature_stable", False)):
        return "carrier_pointer_signature_unstable"
    if record.reference_max_abs_diff < 0.0:
        return "dense_reference_not_checked"
    if record.replay_prehook_us < 0.0:
        return "replay_prehook_us_missing"
    if record.carrier_update_kernel_count < 0:
        return "carrier_update_kernel_count_missing"
    if record.rrp_signature_hit_rate < 0.0:
        return "rrp_signature_hit_rate_missing"
    if record.rrp_full_bind_count < 0:
        return "rrp_full_bind_count_missing"
    return "phase1_gate_failed"


def _write_summary(
    path: Path,
    *,
    record: Phase1FullGraphRecord,
    result: Phase1CommandResult,
    route_trace_path: Path,
    metrics_path: Path,
    hook_profile_path: Path,
    rrp_profile_path: Path,
    route_summary: dict[str, Any],
    metrics: dict[str, Any],
    failure_reason: str,
    run_pair_id: str,
    config_digest: str,
    route_proof: dict[str, object],
    reference_result: Phase1CommandResult | None = None,
    reference_sparse_outputs_path: Path | None = None,
    reference_dense_outputs_path: Path | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "gate_passed": bool(record.gate_passed),
        "failure_reason": failure_reason,
        "run_pair_id": run_pair_id,
        "config_digest": config_digest,
        "route_proof": route_proof,
        "record": record_to_jsonable(record),
        "route_summary": route_summary,
        "metrics": metrics,
        "route_trace_path": str(route_trace_path),
        "decode_metrics_path": str(metrics_path),
        "hook_profile_path": str(hook_profile_path),
        "rrp_profile_path": str(rrp_profile_path),
        "command": result.command,
        "returncode": int(result.returncode),
        "timed_out": bool(result.timed_out),
        "stdout_tail": _tail(result.stdout),
        "stderr_tail": _tail(result.stderr),
        "reference_scope": "not_checked_by_run_sparse_only_smoke",
        "reference_token_ids_exact_match": None,
    }
    if reference_result is not None:
        payload["reference_command"] = reference_result.command
        payload["reference_returncode"] = int(reference_result.returncode)
        payload["reference_timed_out"] = bool(reference_result.timed_out)
        payload["reference_stdout_tail"] = _tail(reference_result.stdout)
        payload["reference_stderr_tail"] = _tail(reference_result.stderr)
        payload["reference_scope"] = "generated_token_ids_advisory_not_gate"
        payload["reference_token_ids_exact_match"] = bool(
            record.reference_max_abs_diff == 0.0
        )
    if reference_sparse_outputs_path is not None:
        payload["reference_sparse_outputs_path"] = str(reference_sparse_outputs_path)
    if reference_dense_outputs_path is not None:
        payload["reference_dense_outputs_path"] = str(reference_dense_outputs_path)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _enforce_phase1_cli_runner_contract(args)
    output_path = Path(args.output)
    summary_path = Path(args.summary_output)
    route_trace_path = (
        Path(args.route_trace_output)
        if args.route_trace_output
        else _default_route_trace_path(output_path)
    )
    metrics_path = (
        Path(args.decode_metrics_output)
        if args.decode_metrics_output
        else _default_decode_metrics_path(output_path)
    )
    hook_profile_path = (
        Path(args.hook_profile_output)
        if args.hook_profile_output
        else _default_hook_profile_path(output_path)
    )
    rrp_profile_path = (
        Path(args.rrp_profile_output)
        if args.rrp_profile_output
        else _default_rrp_profile_path(output_path)
    )
    sparse_outputs_path = _default_sparse_outputs_path(output_path)
    dense_outputs_path = _default_dense_outputs_path(output_path)
    reference_result: Phase1CommandResult | None = None
    reference_max_abs_diff = -1.0

    if args.dry_run:
        result = Phase1CommandResult(
            command=[],
            returncode=GATE_FAILURE_EXIT_CODE,
            stdout="",
            stderr="dry-run: vLLM subprocess was not launched",
            timed_out=False,
        )
    else:
        route_trace_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        hook_profile_path.parent.mkdir(parents=True, exist_ok=True)
        rrp_profile_path.parent.mkdir(parents=True, exist_ok=True)
        sparse_outputs_path.parent.mkdir(parents=True, exist_ok=True)
        route_trace_path.write_text("", encoding="utf-8")
        hook_profile_path.write_text("", encoding="utf-8")
        rrp_profile_path.write_text("", encoding="utf-8")
        command = _build_smoke_command(
            args,
            metrics_path=metrics_path,
            outputs_path=sparse_outputs_path,
        )
        sparse_env = _build_env(
            args,
            route_trace_path=route_trace_path,
            hook_profile_path=hook_profile_path,
            rrp_profile_path=rrp_profile_path,
        )
        result = _run_command(
            command,
            env=sparse_env,
            timeout_s=int(args.timeout_s),
        )
        if result.returncode == 0 and not result.timed_out:
            dense_outputs_path.parent.mkdir(parents=True, exist_ok=True)
            reference_command = _build_dense_reference_command(
                args,
                outputs_path=dense_outputs_path,
            )
            reference_result = _run_command(
                reference_command,
                env=_build_dense_reference_env(args),
                timeout_s=int(args.timeout_s),
            )
            if reference_result.returncode == 0 and not reference_result.timed_out:
                sparse_outputs = _canonical_output_token_lists(sparse_outputs_path)
                dense_outputs = _canonical_output_token_lists(dense_outputs_path)
                if sparse_outputs and sparse_outputs == dense_outputs:
                    reference_max_abs_diff = 0.0
                else:
                    reference_max_abs_diff = 1.0

    events = _read_trace_events(route_trace_path)
    profile_events = _read_jsonl_events(hook_profile_path) + _read_jsonl_events(
        rrp_profile_path
    )
    route_summary = _route_summary(events)
    route_summary.update(_replay_budget_summary(profile_events))
    metrics = _read_json(metrics_path)
    sparse_env = _build_env(
        args,
        route_trace_path=route_trace_path,
        hook_profile_path=hook_profile_path,
        rrp_profile_path=rrp_profile_path,
    )
    fa3_so_sha256 = _file_sha256(_resolve_fa3_so_path(args))
    run_pair_config = _run_pair_config(
        args,
        env=sparse_env,
        fa3_so_sha256=fa3_so_sha256,
    )
    config_digest = build_config_digest(run_pair_config)
    pairing_digest = build_pairing_digest(run_pair_config)
    route_proof_result = validate_shared_route_proof(
        _route_proof_payload(
            route_summary=route_summary,
            env=sparse_env,
            fa3_so_sha256=fa3_so_sha256,
        )
    )
    route_proof = {
        "passed": bool(route_proof_result.passed),
        "reasons": list(route_proof_result.reasons),
    }
    run_pair_id = f"{args.case}:{pairing_digest[:12]}"
    failure_reason = _classify_failure(result)
    if not failure_reason and reference_result is not None:
        reference_failure = _classify_failure(reference_result)
        if reference_failure:
            failure_reason = f"dense_reference_{reference_failure}"
    record = _record_from_result(
        args,
        result=result,
        route_summary=route_summary,
        metrics=metrics,
        reference_max_abs_diff=reference_max_abs_diff,
        require_budget_fields=not bool(args.dry_run),
        run_pair_id=run_pair_id,
        config_digest=config_digest,
        route_proof_result=route_proof_result,
    )
    if not failure_reason and not record.gate_passed:
        failure_reason = _classify_phase1_gate_failure(record, route_summary)
    record, failure_reason = _apply_route_proof_gate(
        record,
        failure_reason,
        route_proof_result,
    )
    write_records(output_path, [record])
    _write_summary(
        summary_path,
        record=record,
        result=result,
        route_trace_path=route_trace_path,
        metrics_path=metrics_path,
        hook_profile_path=hook_profile_path,
        rrp_profile_path=rrp_profile_path,
        route_summary=route_summary,
        metrics=metrics,
        failure_reason=failure_reason,
        run_pair_id=run_pair_id,
        config_digest=config_digest,
        route_proof=route_proof,
        reference_result=reference_result,
        reference_sparse_outputs_path=sparse_outputs_path,
        reference_dense_outputs_path=dense_outputs_path,
    )
    return 0 if record.gate_passed else GATE_FAILURE_EXIT_CODE


if __name__ == "__main__":
    raise SystemExit(main())
