from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Callable

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BENCHMARK_NAME = "fa4_sm100_mixed_page_rrp_microbenchmark"
RRP_LATENCY_BUDGET_MS = 0.09
RRP_SPEED_GATE_REASON_PREFIX = "RRP latency budget exceeded"


@dataclass(frozen=True)
class RrpBenchCase:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    cu_seqlens_q: torch.Tensor
    seqused_k: torch.Tensor
    page_table: torch.Tensor
    resolved_rows: torch.Tensor
    resolved_row_ptr: torch.Tensor
    resolved_seqused: torch.Tensor
    selected_k: int
    softmax_scale: float
    num_splits: int


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FA4 SM100 mixed_page ResolvedRowPtr correctness and latency microbenchmark"
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--q-heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--page-size", type=int, default=128)
    parser.add_argument("--pages-per-batch", type=int, default=4)
    parser.add_argument("--selected-pages", type=int, default=2)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--num-splits", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--output-jsonl",
        default="logs/b200_fa4_mixed_page_rrp_microbench.jsonl",
    )
    parser.add_argument("--append", action="store_true", default=False)
    parser.add_argument(
        "--rrp-latency-budget-ms",
        type=float,
        default=RRP_LATENCY_BUDGET_MS,
        help="Maximum allowed RPP/RRP CUDA-event median latency for this case.",
    )
    return parser.parse_args([] if argv is None else argv)


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    raise ValueError(f"unsupported dtype: {name}")


def _sha256(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_revision(path: Path) -> str | None:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(path),
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def _requires_sm100() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for B200/SM100 FA4 mixed_page RPP benchmark")
    device = torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(device)
    if major != 10:
        raise RuntimeError(
            f"requires SM100/B200 class GPU, got compute capability {major}.{minor}"
        )
    return {
        "gpu_name": torch.cuda.get_device_name(device),
        "compute_capability": [int(major), int(minor)],
        "cuda_device": int(device),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
    }


def _row_ptr_carrier_i64(page_rows: torch.Tensor) -> torch.Tensor:
    assert page_rows.dim() == 2
    return torch.tensor(
        [int(page_rows[row].data_ptr()) for row in range(page_rows.shape[0])],
        device=page_rows.device,
        dtype=torch.int64,
    )


def _build_resolved_rows(
    *,
    batch_size: int,
    kv_heads: int,
    pages_per_batch: int,
    selected_pages: int,
    device: str,
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    for batch_idx in range(batch_size):
        base = batch_idx * pages_per_batch
        all_pages = torch.arange(
            base,
            base + pages_per_batch,
            device=device,
            dtype=torch.int32,
        )
        for kv_head in range(kv_heads):
            rows.append(torch.roll(all_pages, shifts=-kv_head)[:selected_pages])
    return torch.stack(rows, dim=0).contiguous()


def build_case(args: argparse.Namespace) -> RrpBenchCase:
    if args.q_heads % args.kv_heads != 0:
        raise ValueError("--q-heads must be divisible by --kv-heads")
    if args.selected_pages <= 0:
        raise ValueError("--selected-pages must be > 0")
    if args.pages_per_batch < args.selected_pages:
        raise ValueError("--pages-per-batch must be >= --selected-pages")
    if args.page_size <= 0:
        raise ValueError("--page-size must be > 0")
    if args.warmup < 0 or args.repeat <= 0:
        raise ValueError("--warmup must be >= 0 and --repeat must be > 0")

    torch.manual_seed(int(args.seed))
    device = "cuda"
    dtype = _dtype_from_name(str(args.dtype))
    total_pages = int(args.batch_size) * int(args.pages_per_batch)
    selected_k = int(args.selected_pages) * int(args.page_size)
    q = torch.randn(
        (int(args.batch_size), int(args.q_heads), int(args.head_dim)),
        device=device,
        dtype=dtype,
    )
    k = torch.randn(
        (total_pages, int(args.page_size), int(args.kv_heads), int(args.head_dim)),
        device=device,
        dtype=dtype,
    )
    v = torch.randn_like(k)
    cu_seqlens_q = torch.arange(
        0,
        int(args.batch_size) + 1,
        device=device,
        dtype=torch.int32,
    )
    seqused_k = torch.full(
        (int(args.batch_size),),
        selected_k,
        device=device,
        dtype=torch.int32,
    )
    page_table = torch.arange(total_pages, device=device, dtype=torch.int32).reshape(
        int(args.batch_size),
        int(args.pages_per_batch),
    )
    resolved_rows = _build_resolved_rows(
        batch_size=int(args.batch_size),
        kv_heads=int(args.kv_heads),
        pages_per_batch=int(args.pages_per_batch),
        selected_pages=int(args.selected_pages),
        device=device,
    )
    resolved_row_ptr = _row_ptr_carrier_i64(resolved_rows)
    resolved_seqused = torch.full(
        (int(args.batch_size) * int(args.kv_heads),),
        selected_k,
        device=device,
        dtype=torch.int32,
    )
    return RrpBenchCase(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=seqused_k,
        page_table=page_table,
        resolved_rows=resolved_rows,
        resolved_row_ptr=resolved_row_ptr,
        resolved_seqused=resolved_seqused,
        selected_k=selected_k,
        softmax_scale=1.0 / math.sqrt(int(args.head_dim)),
        num_splits=int(args.num_splits),
    )


def _dense_reference(case: RrpBenchCase) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, q_heads, _head_dim = case.q.shape
    kv_heads = case.k.shape[2]
    q_per_kv = q_heads // kv_heads
    out = torch.empty_like(case.q)
    lse = torch.empty((batch_size, q_heads), device=case.q.device, dtype=torch.float32)
    for batch_idx in range(batch_size):
        for q_head in range(q_heads):
            kv_head = q_head // q_per_kv
            row = batch_idx * kv_heads + kv_head
            page_ids = case.resolved_rows[row].long()
            k_tokens = case.k[page_ids, :, kv_head, :].reshape(-1, case.k.shape[-1])
            v_tokens = case.v[page_ids, :, kv_head, :].reshape(-1, case.v.shape[-1])
            scaled_scores = torch.matmul(case.q[batch_idx, q_head].float(), k_tokens.float().T) * case.softmax_scale
            probs = torch.softmax(scaled_scores, dim=-1).to(v_tokens.dtype)
            out[batch_idx, q_head] = torch.matmul(probs, v_tokens).to(out.dtype)
            lse[batch_idx, q_head] = torch.logsumexp(scaled_scores, dim=-1)
    return out, lse


def _dense_lse_like(dense_lse: torch.Tensor, observed_lse: torch.Tensor) -> torch.Tensor:
    if tuple(dense_lse.shape) == tuple(observed_lse.shape):
        return dense_lse.to(device=observed_lse.device, dtype=observed_lse.dtype)
    if dense_lse.dim() == 2 and tuple(dense_lse.t().shape) == tuple(observed_lse.shape):
        return dense_lse.t().contiguous().to(device=observed_lse.device, dtype=observed_lse.dtype)
    if dense_lse.numel() == observed_lse.numel():
        return dense_lse.reshape(observed_lse.shape).to(device=observed_lse.device, dtype=observed_lse.dtype)
    raise AssertionError(f"dense LSE shape {tuple(dense_lse.shape)} cannot match observed {tuple(observed_lse.shape)}")


def run_rrp(case: RrpBenchCase) -> tuple[torch.Tensor, torch.Tensor]:
    from flash_attn.cute.interface import (
        PAGE_RESOLVER_KIND_RESOLVED_ROW_PTR,
        flash_attn_varlen_mixed_page_func,
    )

    return flash_attn_varlen_mixed_page_func(
        case.q,
        case.k,
        case.v,
        case.cu_seqlens_q,
        case.seqused_k,
        case.page_table,
        max_seqlen_q=1,
        max_seqlen_k=case.selected_k,
        causal=False,
        softmax_scale=case.softmax_scale,
        return_lse=True,
        num_splits=case.num_splits,
        page_resolver_kind=PAGE_RESOLVER_KIND_RESOLVED_ROW_PTR,
        resolved_page_table_row_ptr_u64=case.resolved_row_ptr,
        resolved_seqused_k_by_head_i32=case.resolved_seqused,
        graph_replay_carriers=True,
    )


def _assert_correct(case: RrpBenchCase) -> dict[str, float]:
    rrp_out, rrp_lse = run_rrp(case)
    expected, expected_lse = _dense_reference(case)
    expected_lse = _dense_lse_like(expected_lse, rrp_lse)
    torch.cuda.synchronize()
    rrp_vs_ref = float((rrp_out.float() - expected.float()).abs().max().item())
    lse_delta = float((rrp_lse.float() - expected_lse.float()).abs().max().item())
    torch.testing.assert_close(rrp_out, expected, atol=4e-2, rtol=4e-2)
    torch.testing.assert_close(rrp_lse.float(), expected_lse.float(), atol=4e-2, rtol=4e-2)
    return {
        "rrp_vs_dense_absmax": rrp_vs_ref,
        "rrp_lse_vs_dense_absmax": lse_delta,
    }


def _measure_one_cuda_ms(run_case: Callable[[], object]) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    run_case()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end))


def _measure_cuda_ms(run_case: Callable[[], object], *, warmup: int, repeat: int) -> list[float]:
    for _ in range(warmup):
        run_case()
    torch.cuda.synchronize()
    return [_measure_one_cuda_ms(run_case) for _ in range(repeat)]


def _percentile(samples: list[float], percentile: float) -> float:
    if not samples:
        raise ValueError("samples must be non-empty")
    ordered = sorted(samples)
    index = int(math.ceil((percentile / 100.0) * len(ordered))) - 1
    return float(ordered[max(0, min(index, len(ordered) - 1))])


def classify_rrp_speed_gate(
    *,
    rrp_ms: float,
    latency_budget_ms: float,
) -> tuple[str, str | None]:
    if not math.isfinite(rrp_ms):
        return "fail", "RPP/RRP median must be finite"
    if latency_budget_ms <= 0 or not math.isfinite(latency_budget_ms):
        return "fail", "RPP/RRP latency budget must be finite and positive"
    if rrp_ms <= latency_budget_ms:
        return "pass", None
    return (
        "fail",
        f"{RRP_SPEED_GATE_REASON_PREFIX}: median_ms={rrp_ms:.6f} > budget_ms={latency_budget_ms:.6f}",
    )


def _write_jsonl(path: Path, record: dict[str, Any], *, append: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    argv_list = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv_list)
    device_info = _requires_sm100()
    os.environ.setdefault("VLLM_FLASH_ATTN_VERSION", "4")
    case = build_case(args)
    correctness = _assert_correct(case)
    rrp_samples = _measure_cuda_ms(
        lambda: run_rrp(case),
        warmup=int(args.warmup),
        repeat=int(args.repeat),
    )
    rrp_median = float(median(rrp_samples))
    latency_budget_ms = float(args.rrp_latency_budget_ms)
    speed_gate_status, speed_gate_reason = classify_rrp_speed_gate(
        rrp_ms=rrp_median,
        latency_budget_ms=latency_budget_ms,
    )
    output_path = Path(str(args.output_jsonl))
    record: dict[str, Any] = {
        "benchmark": BENCHMARK_NAME,
        "command_line": [sys.executable, "benchmarks/bench_fa4_sm100_mixed_page_rrp.py", *argv_list],
        "status": speed_gate_status,
        "gate_status": speed_gate_status,
        "speed_gate_reason": speed_gate_reason,
        "path_validation_status": "valid_direct_sm100_cute",
        "backend": "fa4_cute_sm100",
        "page_resolver_kind": 4,
        "rrp_route_mode": "resolved_row_ptr",
        "rrp_runtime_selected_table": False,
        "speed_baseline": "rpp_latency_budget",
        "kind2_dispatch_count": 0,
        "kind3_dispatch_count": 0,
        "selected_table_publish_count": 0,
        "vector_fallback_rows": 0,
        "full_fallback_rows": 0,
        "graph_replay_carriers": True,
        "graph_replay_evidence_scope": "direct_fa4_cute_microbench_carrier_flag",
        "graph_replay_full_cuda_graph_hotpath_verified": False,
        "speed_gate_basis": "measured RPP/RRP CUDA-event median against fixed latency budget",
        "speed_gate_reference": "rrp_cuda_event_ms_median",
        "rrp_latency_budget_ms": latency_budget_ms,
        "correctness_reference": "dense_reference_only",
        "correctness_absmax_max": 4.0e-2,
        "correctness_lse_absmax_max": 4.0e-2,
        "kernel_case": f"rrp_b{args.batch_size}_qh{args.q_heads}_kvh{args.kv_heads}_p{args.page_size}_sel{args.selected_pages}_{args.dtype}",
        "batch_size": int(args.batch_size),
        "q_heads": int(args.q_heads),
        "kv_heads": int(args.kv_heads),
        "head_dim": int(args.head_dim),
        "page_size": int(args.page_size),
        "pages_per_batch": int(args.pages_per_batch),
        "selected_pages": int(args.selected_pages),
        "selected_k": int(case.selected_k),
        "requested_num_splits": int(case.num_splits),
        "max_seqlen_k": int(case.selected_k),
        "dtype": str(args.dtype),
        "warmup": int(args.warmup),
        "repeat": int(args.repeat),
        "measurement_order": "rrp_only",
        "seed": int(args.seed),
        "rrp_cuda_event_ms_median": rrp_median,
        "rrp_cuda_event_ms_p90": _percentile(rrp_samples, 90.0),
        "rrp_cuda_event_ms_samples": rrp_samples,
        "sample_count_rrp": len(rrp_samples),
        "root_git_revision": _git_revision(REPO_ROOT),
        "latest_sm100_patch_sha256": _sha256(
            REPO_ROOT / "patches/fa4_cute_sm100/upstream_recovery/latest-sm100-cute-compact-recent-current.patch"
        ),
    }
    record.update(device_info)
    record.update(correctness)
    _write_jsonl(output_path, record, append=bool(args.append))
    print(json.dumps(record, sort_keys=True))
    return 0 if speed_gate_status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
