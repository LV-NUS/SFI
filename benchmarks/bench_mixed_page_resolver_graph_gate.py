from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Callable

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_FA3_ROOT = REPO_ROOT / "third_party_upstreams" / "vllm-project-flash-attention"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(UPSTREAM_FA3_ROOT) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_FA3_ROOT))

from benchmarks.bench_mixed_page_baseline import BenchShape, build_benchmark_case  # noqa: E402
from benchmarks.mixed_page_resolver_logit_capture_check import (  # noqa: E402
    _dense_capture_reference,
)
from patches.fa3_native.install import load_vendored_flash_attn_bridge  # noqa: E402
from patches.fa3_native.mixed_page_graph_descriptor import PageResolverKind  # noqa: E402
from patches.fa3_native.row_consume_modes import (  # noqa: E402
    ROW_CONSUME_MODE_FULL_I32,
    ROW_CONSUME_MODE_SELECTED_I32,
)


GATE_FAILURE_EXIT_CODE = 2
DEFAULT_ATOL_OUT = 0.0
DEFAULT_RTOL_OUT = 0.0
DEFAULT_ATOL_LSE = 1.0e-3
DEFAULT_RTOL_LSE = 0.0
DEFAULT_PAGE_BLOCK_SIZE = 16
DEFAULT_SELECTED_PAGES = 4
PAGE_RESOLVER_SUBKIND_ROWPTR = 0
PAGE_RESOLVER_SUBKIND_DIRECT_TABLE = 1
PAGE_RESOLVER_SUBKIND_AFFINE_TENSOR = 3
FA3_EXTENSION_GLOBS = (
    "_vllm_fa3_C*.so",
    "_vllm_fa3_C*.pyd",
)


@dataclass(frozen=True)
class MixedPageResolverGateRecord:
    case: str
    resolver_kind: int
    row_modes: list[int]
    batch: int
    heads: int
    page_block_size: int
    q_tokens: int
    kv_tokens: int
    warmup: int
    iters: int
    graph_replay: bool
    graph_cache_hit: bool
    resolver_subkind: str
    tma_or_non_tma: str
    carrier_pointer_signature_before: dict[str, int]
    carrier_pointer_signature_after: dict[str, int]
    graph_recapture_count: int
    stale_carrier_negative_passed: bool
    changed_mapping_output_allclose: bool
    carrier_update_us: float
    mixed_page_us: float
    tokens_per_second: float
    reference_max_abs_diff: float
    correctness_passed: bool
    logit_capture_passed: bool
    route_proof: str
    extension_path: str
    git_sha: str
    cuda_device_name: str


@dataclass(frozen=True)
class CaseSpec:
    name: str
    resolver_kind: int
    page_resolver_subkind: int
    resolver_subkind: str
    row_modes: tuple[int, ...]
    capture: bool
    graph_replay: bool = False
    tma_or_non_tma: str = "non_tma"


@dataclass(frozen=True)
class GateSummary:
    gate_passed: bool
    case_count: int
    missing_cases: list[str]
    duplicate_cases: list[str]
    failed_cases: list[str]
    max_graph_overhead_pct: float
    resolved_row_ptr_graph_overhead_pct: float | None
    max_existing_regression_pct: float
    existing_regression_pct: float | None
    selected_table_graph_replay_us: float | None
    resolved_row_ptr_graph_replay_us: float | None
    resolved_row_ptr_carrier_update_us: float | None
    external_resolved_row_ptr_carrier_update_us: float | None
    external_resolved_row_ptr_graph_overhead_pct: float | None


EXPECTED_CASES = (
    CaseSpec(
        "native_paged_no_capture",
        int(PageResolverKind.NATIVE),
        PAGE_RESOLVER_SUBKIND_ROWPTR,
        "native",
        (ROW_CONSUME_MODE_FULL_I32,),
        False,
    ),
    CaseSpec(
        "selected_table_no_capture",
        int(PageResolverKind.SELECTED_TABLE),
        PAGE_RESOLVER_SUBKIND_ROWPTR,
        "selected",
        (ROW_CONSUME_MODE_SELECTED_I32,),
        False,
    ),
    CaseSpec(
        "mixed_native_selected_no_capture",
        int(PageResolverKind.SELECTED_TABLE),
        PAGE_RESOLVER_SUBKIND_ROWPTR,
        "selected_mixed_rows",
        (ROW_CONSUME_MODE_SELECTED_I32, ROW_CONSUME_MODE_FULL_I32),
        False,
    ),
    CaseSpec(
        "resolved_row_ptr_no_capture",
        int(PageResolverKind.RESOLVED_ROW_PTR),
        PAGE_RESOLVER_SUBKIND_ROWPTR,
        "rowptr",
        (ROW_CONSUME_MODE_SELECTED_I32,),
        False,
    ),
    CaseSpec(
        "resolved_direct_table_no_capture",
        int(PageResolverKind.RESOLVED_ROW_PTR),
        PAGE_RESOLVER_SUBKIND_DIRECT_TABLE,
        "direct_table",
        (ROW_CONSUME_MODE_SELECTED_I32,),
        False,
    ),
    CaseSpec(
        "resolved_affine_tensor_no_capture",
        int(PageResolverKind.RESOLVED_ROW_PTR),
        PAGE_RESOLVER_SUBKIND_AFFINE_TENSOR,
        "affine_tensor",
        (ROW_CONSUME_MODE_SELECTED_I32,),
        False,
    ),
    CaseSpec(
        "selected_table_capture",
        int(PageResolverKind.SELECTED_TABLE),
        PAGE_RESOLVER_SUBKIND_ROWPTR,
        "selected",
        (ROW_CONSUME_MODE_SELECTED_I32,),
        True,
    ),
    CaseSpec(
        "resolved_row_ptr_capture",
        int(PageResolverKind.RESOLVED_ROW_PTR),
        PAGE_RESOLVER_SUBKIND_ROWPTR,
        "rowptr",
        (ROW_CONSUME_MODE_SELECTED_I32,),
        True,
    ),
    CaseSpec(
        "resolved_row_ptr_graph_replay",
        int(PageResolverKind.RESOLVED_ROW_PTR),
        PAGE_RESOLVER_SUBKIND_ROWPTR,
        "rowptr",
        (ROW_CONSUME_MODE_SELECTED_I32,),
        False,
        True,
    ),
)
CASE_BY_NAME = {case.name: case for case in EXPECTED_CASES}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gate mixed_page resolver correctness, capture, and CUDA graph replay speed."
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--max-graph-overhead-pct", type=float, default=3.0)
    parser.add_argument("--max-existing-regression-pct", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--kv-len", type=int, default=128)
    parser.add_argument("--selected-pages", type=int, default=DEFAULT_SELECTED_PAGES)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    if args.warmup < 0:
        parser.error("--warmup must be >= 0")
    if args.iters <= 0:
        parser.error("--iters must be > 0")
    if args.max_graph_overhead_pct < 0.0:
        parser.error("--max-graph-overhead-pct must be >= 0")
    if args.max_existing_regression_pct < 0.0:
        parser.error("--max-existing-regression-pct must be >= 0")
    if args.batch_size < 2:
        parser.error("--batch-size must be >= 2")
    if args.kv_len <= 0 or args.kv_len % DEFAULT_PAGE_BLOCK_SIZE != 0:
        parser.error("--kv-len must be a positive multiple of 16")
    if args.selected_pages <= 0 or args.selected_pages > args.kv_len // DEFAULT_PAGE_BLOCK_SIZE:
        parser.error("--selected-pages must fit within --kv-len")
    return args


def expected_case_names() -> set[str]:
    return {case.name for case in EXPECTED_CASES}


def record_to_jsonable(record: MixedPageResolverGateRecord) -> dict[str, object]:
    return asdict(record)


def summary_to_jsonable(summary: GateSummary) -> dict[str, object]:
    return asdict(summary)


def write_records(output: Path, records: list[MixedPageResolverGateRecord]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record_to_jsonable(record), allow_nan=False, sort_keys=True))
            handle.write("\n")


def write_summary(output: Path, summary: GateSummary) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary_to_jsonable(summary), allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return result.stdout.strip()


def _extension_path() -> str:
    package_dir = UPSTREAM_FA3_ROOT / "vllm_flash_attn"
    matches: list[Path] = []
    for pattern in FA3_EXTENSION_GLOBS:
        matches.extend(package_dir.glob(pattern))
    if not matches:
        return ""
    return str(sorted(matches)[0])


def _cuda_device_name() -> str:
    if not torch.cuda.is_available():
        return ""
    return str(torch.cuda.get_device_name(torch.cuda.current_device()))


def _unavailable_records(
    *,
    reason: str,
    warmup: int,
    iters: int,
    extension_path: str,
    git_sha: str,
    cuda_device_name: str,
) -> list[MixedPageResolverGateRecord]:
    records: list[MixedPageResolverGateRecord] = []
    for case in EXPECTED_CASES:
        records.append(
            MixedPageResolverGateRecord(
                case=case.name,
                resolver_kind=case.resolver_kind,
                row_modes=[int(mode) for mode in case.row_modes],
                batch=0,
                heads=0,
                page_block_size=DEFAULT_PAGE_BLOCK_SIZE,
                q_tokens=0,
                kv_tokens=0,
                warmup=int(warmup),
                iters=int(iters),
                graph_replay=case.graph_replay,
                graph_cache_hit=False,
                resolver_subkind=case.resolver_subkind,
                tma_or_non_tma=case.tma_or_non_tma,
                carrier_pointer_signature_before={},
                carrier_pointer_signature_after={},
                graph_recapture_count=0,
                stale_carrier_negative_passed=False,
                changed_mapping_output_allclose=False,
                carrier_update_us=0.0,
                mixed_page_us=0.0,
                tokens_per_second=0.0,
                reference_max_abs_diff=1.0,
                correctness_passed=False,
                logit_capture_passed=not case.capture,
                route_proof=f"unavailable: {reason}",
                extension_path=extension_path,
                git_sha=git_sha,
                cuda_device_name=cuda_device_name,
            )
        )
    return records


def summarize_records(
    records: list[MixedPageResolverGateRecord],
    *,
    max_graph_overhead_pct: float,
    max_existing_regression_pct: float,
    selected_table_graph_replay_us: float | None = None,
    resolved_row_ptr_graph_replay_us: float | None = None,
    resolved_row_ptr_carrier_update_us: float | None = None,
    external_resolved_row_ptr_carrier_update_us: float | None = None,
    existing_regression_pct: float | None = None,
) -> GateSummary:
    actual_cases = [record.case for record in records]
    expected_cases = expected_case_names()
    missing_cases = sorted(expected_cases.difference(actual_cases))
    duplicate_cases = sorted(
        {case for case in actual_cases if actual_cases.count(case) > 1}
    )
    failed_cases = sorted(
        record.case
        for record in records
        if not record.correctness_passed or not record.logit_capture_passed
    )
    failed_graph_cases = sorted(
        record.case
        for record in records
        if record.graph_replay
        and (
            not record.graph_cache_hit
            or record.carrier_pointer_signature_before
            != record.carrier_pointer_signature_after
            or record.graph_recapture_count != 0
            or not record.stale_carrier_negative_passed
            or not record.changed_mapping_output_allclose
        )
    )
    failed_cases = sorted(set(failed_cases).union(failed_graph_cases))

    resolved_row_ptr_graph_overhead_pct: float | None = None
    external_resolved_row_ptr_graph_overhead_pct: float | None = None
    if (
        selected_table_graph_replay_us is not None
        and resolved_row_ptr_graph_replay_us is not None
        and resolved_row_ptr_carrier_update_us is not None
        and selected_table_graph_replay_us > 0.0
    ):
        resolved_row_ptr_graph_overhead_pct = (
            (resolved_row_ptr_graph_replay_us + resolved_row_ptr_carrier_update_us)
            / selected_table_graph_replay_us
            - 1.0
        ) * 100.0
    if (
        selected_table_graph_replay_us is not None
        and resolved_row_ptr_graph_replay_us is not None
        and external_resolved_row_ptr_carrier_update_us is not None
        and selected_table_graph_replay_us > 0.0
    ):
        external_resolved_row_ptr_graph_overhead_pct = (
            (resolved_row_ptr_graph_replay_us + external_resolved_row_ptr_carrier_update_us)
            / selected_table_graph_replay_us
            - 1.0
        ) * 100.0

    graph_gate_passed = (
        resolved_row_ptr_graph_overhead_pct is not None
        and resolved_row_ptr_graph_overhead_pct <= float(max_graph_overhead_pct)
    )
    existing_gate_passed = (
        existing_regression_pct is None
        or existing_regression_pct <= float(max_existing_regression_pct)
    )
    gate_passed = (
        not missing_cases
        and not duplicate_cases
        and not failed_cases
        and graph_gate_passed
        and existing_gate_passed
    )
    return GateSummary(
        gate_passed=gate_passed,
        case_count=len(records),
        missing_cases=missing_cases,
        duplicate_cases=duplicate_cases,
        failed_cases=failed_cases,
        max_graph_overhead_pct=float(max_graph_overhead_pct),
        resolved_row_ptr_graph_overhead_pct=resolved_row_ptr_graph_overhead_pct,
        max_existing_regression_pct=float(max_existing_regression_pct),
        existing_regression_pct=existing_regression_pct,
        selected_table_graph_replay_us=selected_table_graph_replay_us,
        resolved_row_ptr_graph_replay_us=resolved_row_ptr_graph_replay_us,
        resolved_row_ptr_carrier_update_us=resolved_row_ptr_carrier_update_us,
        external_resolved_row_ptr_carrier_update_us=external_resolved_row_ptr_carrier_update_us,
        external_resolved_row_ptr_graph_overhead_pct=external_resolved_row_ptr_graph_overhead_pct,
    )


def exit_code_for_summary(summary: GateSummary) -> int:
    return 0 if summary.gate_passed else GATE_FAILURE_EXIT_CODE


def _time_cuda_us(
    runner: Callable[[], object],
    *,
    warmup: int,
    iters: int,
) -> float:
    for _ in range(warmup):
        runner()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        runner()
        end.record()
        torch.cuda.synchronize()
        samples.append(float(start.elapsed_time(end)) * 1000.0)
    return float(median(samples))


def _time_copy_us(
    update: Callable[[], object],
    *,
    iters: int,
) -> float:
    samples: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        update()
        end.record()
        torch.cuda.synchronize()
        samples.append(float(start.elapsed_time(end)) * 1000.0)
    return float(median(samples))


def _make_shape(args: argparse.Namespace) -> BenchShape:
    return BenchShape(
        batch_size=int(args.batch_size),
        q_len=1,
        kv_len=int(args.kv_len),
        selected_ratio=float(args.selected_pages) / float(args.kv_len // DEFAULT_PAGE_BLOCK_SIZE),
        page_size=DEFAULT_PAGE_BLOCK_SIZE,
        num_q_heads=32,
        num_kv_heads=8,
        head_dim=128,
        dtype=torch.bfloat16,
        device="cuda",
        seed=int(args.seed),
    )


def _selected_table(case, selected_pages: int) -> torch.Tensor:
    return case.page_table_i32[:, : int(selected_pages)].contiguous()


def _selected_seqused(case, selected_k: int) -> torch.Tensor:
    return torch.full(
        (case.shape.batch_size,),
        int(selected_k),
        dtype=torch.int32,
        device=case.q.device,
    )


def _selected_by_head(case, selected_k: int, *, mixed: bool = False) -> torch.Tensor:
    values = torch.full(
        (case.shape.batch_size * case.shape.num_kv_heads,),
        int(selected_k),
        dtype=torch.int32,
        device=case.q.device,
    )
    if mixed:
        values[case.shape.num_kv_heads :] = 0
    return values


def _q_bhd(case) -> torch.Tensor:
    return case.q.view(
        case.shape.batch_size,
        case.shape.q_len,
        case.shape.num_q_heads,
        case.shape.head_dim,
    )[:, 0].contiguous()


def _run_native(
    bridge,
    case,
    *,
    page_table: torch.Tensor,
    seqused_k: torch.Tensor,
    max_seqlen_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return bridge.flash_attn_varlen_func(
        q=case.q,
        k=case.k_cache,
        v=case.v_cache,
        max_seqlen_q=case.shape.q_len,
        cu_seqlens_q=case.cu_seqlens_q_i32,
        max_seqlen_k=int(max_seqlen_k),
        seqused_k=seqused_k,
        softmax_scale=case.softmax_scale,
        causal=True,
        block_table=page_table,
        return_softmax_lse=True,
        fa_version=3,
        num_splits=0,
    )


def _run_mixed_page(
    bridge,
    case,
    *,
    selected_page_table_i32: torch.Tensor | None,
    row_consume_mode_i32: torch.Tensor | None,
    selected_seqused_k_by_head_i32: torch.Tensor | None,
    seqused_k: torch.Tensor,
    max_seqlen_k: int,
    page_resolver_kind: int,
    page_resolver_subkind: int = PAGE_RESOLVER_SUBKIND_ROWPTR,
    resolved_page_table_row_ptr_u64: torch.Tensor | None = None,
    resolved_page_table_i32: torch.Tensor | None = None,
    resolved_page_table_affine_i32: torch.Tensor | None = None,
    resolved_seqused_k_by_head_i32: torch.Tensor | None = None,
    capture_scores: torch.Tensor | None = None,
    capture_row_index_i32: torch.Tensor | None = None,
    row_capture_last_n_i32: torch.Tensor | None = None,
    block_table_i32: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return bridge.mixed_page_attn_varlen_func(
        q=case.q,
        k=case.k_cache,
        v=case.v_cache,
        max_seqlen_q=case.shape.q_len,
        cu_seqlens_q=case.cu_seqlens_q_i32,
        max_seqlen_k=int(max_seqlen_k),
        seqused_k=seqused_k,
        softmax_scale=case.softmax_scale,
        causal=True,
        block_table=case.page_table_i32 if block_table_i32 is None else block_table_i32,
        return_softmax_lse=True,
        out=out,
        selected_page_table_i32=selected_page_table_i32,
        resolved_page_table_row_ptr_u64=resolved_page_table_row_ptr_u64,
        resolved_page_table_i32=resolved_page_table_i32,
        resolved_page_table_affine_i32=resolved_page_table_affine_i32,
        resolved_page_table_affine_cols=(
            int(resolved_page_table_affine_i32.shape[-1])
            if resolved_page_table_affine_i32 is not None
            else 0
        ),
        resolved_seqused_k_by_head_i32=resolved_seqused_k_by_head_i32,
        row_consume_mode_i32=row_consume_mode_i32,
        selected_seqused_k_by_head_i32=selected_seqused_k_by_head_i32,
        cp_selected_seqused_k_by_head_i32=None,
        page_resolver_kind=int(page_resolver_kind),
        page_resolver_subkind=int(page_resolver_subkind),
        capture_scores=capture_scores,
        capture_row_index_i32=capture_row_index_i32,
        row_capture_last_n_i32=row_capture_last_n_i32,
        num_splits=0,
    )


def _assert_close_and_diff(
    *,
    actual_out: torch.Tensor,
    expected_out: torch.Tensor,
    actual_lse: torch.Tensor,
    expected_lse: torch.Tensor,
) -> tuple[bool, float]:
    max_abs_diff = float((actual_out.to(torch.float32) - expected_out.to(torch.float32)).abs().max().item())
    out_ok = torch.allclose(actual_out, expected_out, atol=DEFAULT_ATOL_OUT, rtol=DEFAULT_RTOL_OUT)
    lse_ok = torch.allclose(actual_lse, expected_lse, atol=DEFAULT_ATOL_LSE, rtol=DEFAULT_RTOL_LSE)
    return bool(out_ok and lse_ok), max_abs_diff


def _assert_mixed_rows_close_and_diff(
    *,
    actual_out: torch.Tensor,
    expected_selected: torch.Tensor,
    expected_full: torch.Tensor,
    actual_lse: torch.Tensor,
    expected_selected_lse: torch.Tensor,
    expected_full_lse: torch.Tensor,
) -> tuple[bool, float]:
    row0 = slice(0, 1)
    row1 = slice(1, 2)
    selected_diff = (
        actual_out[row0].to(torch.float32) - expected_selected[row0].to(torch.float32)
    ).abs().max()
    full_diff = (
        actual_out[row1].to(torch.float32) - expected_full[row1].to(torch.float32)
    ).abs().max()
    if actual_lse.dim() != 2:
        return False, float(torch.maximum(selected_diff, full_diff).item())
    if actual_lse.shape[0] == actual_out.shape[1]:
        actual_lse_row0 = actual_lse[:, row0]
        actual_lse_row1 = actual_lse[:, row1]
        selected_lse_row0 = expected_selected_lse[:, row0]
        full_lse_row1 = expected_full_lse[:, row1]
    elif actual_lse.shape[0] == actual_out.shape[0]:
        actual_lse_row0 = actual_lse[row0]
        actual_lse_row1 = actual_lse[row1]
        selected_lse_row0 = expected_selected_lse[row0]
        full_lse_row1 = expected_full_lse[row1]
    else:
        return False, float(torch.maximum(selected_diff, full_diff).item())
    out_ok = bool(
        torch.allclose(actual_out[row0], expected_selected[row0], atol=0.0, rtol=0.0)
        and torch.allclose(actual_out[row1], expected_full[row1], atol=0.0, rtol=0.0)
    )
    lse_ok = bool(
        torch.allclose(actual_lse_row0, selected_lse_row0, atol=DEFAULT_ATOL_LSE, rtol=0.0)
        and torch.allclose(actual_lse_row1, full_lse_row1, atol=DEFAULT_ATOL_LSE, rtol=0.0)
    )
    return bool(out_ok and lse_ok), float(torch.maximum(selected_diff, full_diff).item())


def _tokens_per_second(q_tokens: int, mixed_page_us: float) -> float:
    if mixed_page_us <= 0.0:
        return 0.0
    return float(q_tokens) / (mixed_page_us / 1_000_000.0)


def _record(
    *,
    spec: CaseSpec,
    case,
    kv_tokens: int,
    warmup: int,
    iters: int,
    graph_cache_hit: bool,
    carrier_update_us: float,
    mixed_page_us: float,
    reference_max_abs_diff: float,
    correctness_passed: bool,
    logit_capture_passed: bool,
    route_proof: str,
    extension_path: str,
    git_sha: str,
    cuda_device_name: str,
    carrier_pointer_signature_before: dict[str, int] | None = None,
    carrier_pointer_signature_after: dict[str, int] | None = None,
    graph_recapture_count: int = 0,
    stale_carrier_negative_passed: bool = True,
    changed_mapping_output_allclose: bool = True,
) -> MixedPageResolverGateRecord:
    q_tokens = int(case.shape.batch_size * case.shape.q_len)
    return MixedPageResolverGateRecord(
        case=spec.name,
        resolver_kind=spec.resolver_kind,
        row_modes=[int(mode) for mode in spec.row_modes],
        batch=int(case.shape.batch_size),
        heads=int(case.shape.num_q_heads),
        page_block_size=int(case.shape.page_size),
        q_tokens=q_tokens,
        kv_tokens=int(kv_tokens),
        warmup=int(warmup),
        iters=int(iters),
        graph_replay=spec.graph_replay,
        graph_cache_hit=graph_cache_hit,
        resolver_subkind=spec.resolver_subkind,
        tma_or_non_tma=spec.tma_or_non_tma,
        carrier_pointer_signature_before=dict(carrier_pointer_signature_before or {}),
        carrier_pointer_signature_after=dict(carrier_pointer_signature_after or {}),
        graph_recapture_count=int(graph_recapture_count),
        stale_carrier_negative_passed=bool(stale_carrier_negative_passed),
        changed_mapping_output_allclose=bool(changed_mapping_output_allclose),
        carrier_update_us=float(carrier_update_us),
        mixed_page_us=float(mixed_page_us),
        tokens_per_second=_tokens_per_second(q_tokens, mixed_page_us),
        reference_max_abs_diff=float(reference_max_abs_diff),
        correctness_passed=bool(correctness_passed),
        logit_capture_passed=bool(logit_capture_passed),
        route_proof=route_proof,
        extension_path=extension_path,
        git_sha=git_sha,
        cuda_device_name=cuda_device_name,
    )


def _measure_graph_runner(
    runner: Callable[[], tuple[torch.Tensor, torch.Tensor]],
    *,
    warmup: int,
    iters: int,
) -> tuple[float, torch.Tensor, bool, torch.cuda.CUDAGraph]:
    for _ in range(warmup):
        runner()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    graph_outputs: list[torch.Tensor] = []
    with torch.cuda.graph(graph):
        out, _ = runner()
        graph_outputs[:] = [out]
    graph.replay()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        samples.append(float(start.elapsed_time(end)) * 1000.0)
    if not graph_outputs:
        raise RuntimeError("CUDA graph capture did not expose an output tensor")
    return float(median(samples)), graph_outputs[0], True, graph


def _capture_passed(
    *,
    actual: torch.Tensor,
    expected: torch.Tensor,
    valid_k: int,
) -> bool:
    if not torch.allclose(
        actual[:, :, :, :valid_k],
        expected[:, :, :, :valid_k].to(actual.dtype),
        atol=2.0e-2,
        rtol=2.0e-2,
    ):
        return False
    if int(actual.shape[-1]) > int(valid_k):
        return bool(torch.isneginf(actual[:, :, :, valid_k:]).all())
    return True


def _resolved_row_ptr_table(case, page_rows: list[torch.Tensor]) -> torch.Tensor:
    values: list[int] = []
    for row_tensor in page_rows:
        for _ in range(int(case.shape.num_kv_heads)):
            values.append(int(row_tensor.data_ptr()))
    return torch.tensor(values, device=case.q.device, dtype=torch.int64)


def _affine_tensor_from_selected_table(selected_table: torch.Tensor) -> torch.Tensor:
    base = selected_table[:, 0]
    stride = torch.ones_like(base)
    return torch.stack((base, stride), dim=1).contiguous()


def carrier_pointer_signature(carriers: dict[str, torch.Tensor | None]) -> dict[str, int]:
    return {
        name: int(value.data_ptr())
        for name, value in carriers.items()
        if isinstance(value, torch.Tensor)
    }


def mutate_carriers_in_place(
    carriers: dict[str, torch.Tensor | None],
    *,
    page_block_size: int,
    max_page_id: int,
) -> None:
    table = carriers.get("selected_page_table_i32")
    if isinstance(table, torch.Tensor):
        if int(table.max().item()) < int(max_page_id):
            table.add_(1)
        else:
            table.sub_(1)
    direct = carriers.get("resolved_page_table_i32")
    if isinstance(direct, torch.Tensor):
        if int(direct.max().item()) < int(max_page_id):
            direct.add_(1)
        else:
            direct.sub_(1)
    affine = carriers.get("resolved_page_table_affine_i32")
    if isinstance(affine, torch.Tensor):
        if int(affine[:, 0].max().item()) < int(max_page_id):
            affine[:, 0].add_(1)
        else:
            affine[:, 0].sub_(1)
    visible = carriers.get("resolved_seqused_k_by_head_i32")
    if isinstance(visible, torch.Tensor):
        visible.copy_(torch.clamp(visible - int(page_block_size), min=int(page_block_size)))


def _run_available_gate(args: argparse.Namespace) -> tuple[list[MixedPageResolverGateRecord], GateSummary]:
    bridge = load_vendored_flash_attn_bridge()
    extension_path = _extension_path()
    git_sha = _git_sha()
    cuda_device_name = _cuda_device_name()

    case = build_benchmark_case(_make_shape(args))
    selected_pages = int(args.selected_pages)
    selected_k = selected_pages * case.shape.page_size
    selected_table = _selected_table(case, selected_pages)
    selected_seqused = _selected_seqused(case, selected_k)
    selected_by_head = _selected_by_head(case, selected_k)
    resolved_row_ptr = _resolved_row_ptr_table(
        case,
        [selected_table[row] for row in range(int(case.shape.batch_size))],
    )
    full_out, full_lse = _run_native(
        bridge,
        case,
        page_table=case.page_table_i32,
        seqused_k=case.full_seqused_k_i32,
        max_seqlen_k=case.shape.kv_len,
    )
    selected_out, selected_lse = _run_native(
        bridge,
        case,
        page_table=selected_table,
        seqused_k=selected_seqused,
        max_seqlen_k=selected_k,
    )

    records: list[MixedPageResolverGateRecord] = []
    warmup = int(args.warmup)
    iters = int(args.iters)
    common = {
        "extension_path": extension_path,
        "git_sha": git_sha,
        "cuda_device_name": cuda_device_name,
    }

    native_spec = CASE_BY_NAME["native_paged_no_capture"]
    native_runner = lambda: _run_native(
        bridge,
        case,
        page_table=case.page_table_i32,
        seqused_k=case.full_seqused_k_i32,
        max_seqlen_k=case.shape.kv_len,
    )
    native_us = _time_cuda_us(native_runner, warmup=warmup, iters=iters)
    records.append(
        _record(
            spec=native_spec,
            case=case,
            kv_tokens=case.shape.kv_len,
            warmup=warmup,
            iters=iters,
            graph_cache_hit=False,
            carrier_update_us=0.0,
            mixed_page_us=native_us,
            reference_max_abs_diff=0.0,
            correctness_passed=True,
            logit_capture_passed=True,
            route_proof="direct native paged FA3 baseline",
            **common,
        )
    )

    selected_spec = CASE_BY_NAME["selected_table_no_capture"]
    selected_runner = lambda: _run_mixed_page(
        bridge,
        case,
        selected_page_table_i32=selected_table,
        row_consume_mode_i32=None,
        selected_seqused_k_by_head_i32=None,
        seqused_k=selected_seqused,
        max_seqlen_k=selected_k,
        page_resolver_kind=int(PageResolverKind.SELECTED_TABLE),
    )
    mixed_selected_out, mixed_selected_lse = selected_runner()
    selected_correct, selected_diff = _assert_close_and_diff(
        actual_out=mixed_selected_out,
        expected_out=selected_out,
        actual_lse=mixed_selected_lse,
        expected_lse=selected_lse,
    )
    selected_us = _time_cuda_us(selected_runner, warmup=warmup, iters=iters)
    records.append(
        _record(
            spec=selected_spec,
            case=case,
            kv_tokens=selected_k,
            warmup=warmup,
            iters=iters,
            graph_cache_hit=False,
            carrier_update_us=0.0,
            mixed_page_us=selected_us,
            reference_max_abs_diff=selected_diff,
            correctness_passed=selected_correct,
            logit_capture_passed=True,
            route_proof="mixed_page selected-table no-capture",
            **common,
        )
    )

    mixed_spec = CASE_BY_NAME["mixed_native_selected_no_capture"]
    mixed_modes = torch.tensor(
        [ROW_CONSUME_MODE_SELECTED_I32, ROW_CONSUME_MODE_FULL_I32],
        dtype=torch.int32,
        device=case.q.device,
    )
    mixed_seqused = torch.tensor(
        [selected_k, case.shape.kv_len],
        dtype=torch.int32,
        device=case.q.device,
    )
    mixed_selected_by_head = _selected_by_head(case, selected_k, mixed=True)
    mixed_runner = lambda: _run_mixed_page(
        bridge,
        case,
        selected_page_table_i32=selected_table,
        row_consume_mode_i32=mixed_modes,
        selected_seqused_k_by_head_i32=mixed_selected_by_head,
        seqused_k=mixed_seqused,
        max_seqlen_k=case.shape.kv_len,
        page_resolver_kind=int(PageResolverKind.SELECTED_TABLE),
    )
    mixed_out, mixed_lse = mixed_runner()
    mixed_correct, mixed_diff = _assert_mixed_rows_close_and_diff(
        actual_out=mixed_out,
        expected_selected=selected_out,
        expected_full=full_out,
        actual_lse=mixed_lse,
        expected_selected_lse=selected_lse,
        expected_full_lse=full_lse,
    )
    mixed_us = _time_cuda_us(mixed_runner, warmup=warmup, iters=iters)
    records.append(
        _record(
            spec=mixed_spec,
            case=case,
            kv_tokens=case.shape.kv_len,
            warmup=warmup,
            iters=iters,
            graph_cache_hit=False,
            carrier_update_us=0.0,
            mixed_page_us=mixed_us,
            reference_max_abs_diff=mixed_diff,
            correctness_passed=mixed_correct,
            logit_capture_passed=True,
            route_proof="mixed_page row_mode selected+full no-capture",
            **common,
        )
    )

    resolved_rowptr_runner = lambda: _run_mixed_page(
        bridge,
        case,
        selected_page_table_i32=None,
        row_consume_mode_i32=None,
        selected_seqused_k_by_head_i32=None,
        seqused_k=selected_seqused,
        max_seqlen_k=selected_k,
        page_resolver_kind=int(PageResolverKind.RESOLVED_ROW_PTR),
        page_resolver_subkind=PAGE_RESOLVER_SUBKIND_ROWPTR,
        resolved_page_table_row_ptr_u64=resolved_row_ptr,
        resolved_seqused_k_by_head_i32=selected_by_head,
    )
    resolved_rowptr_out, resolved_rowptr_lse = resolved_rowptr_runner()
    resolved_rowptr_correct, resolved_rowptr_diff = _assert_close_and_diff(
        actual_out=resolved_rowptr_out,
        expected_out=selected_out,
        actual_lse=resolved_rowptr_lse,
        expected_lse=selected_lse,
    )
    resolved_rowptr_us = _time_cuda_us(resolved_rowptr_runner, warmup=warmup, iters=iters)
    rowptr_spec = CASE_BY_NAME["resolved_row_ptr_no_capture"]
    records.append(
        _record(
            spec=rowptr_spec,
            case=case,
            kv_tokens=selected_k,
            warmup=warmup,
            iters=iters,
            graph_cache_hit=False,
            carrier_update_us=0.0,
            mixed_page_us=resolved_rowptr_us,
            reference_max_abs_diff=resolved_rowptr_diff,
            correctness_passed=resolved_rowptr_correct,
            logit_capture_passed=True,
            route_proof="mixed_page ResolvedRowPtr no-capture",
            **common,
        )
    )

    direct_table = selected_table.clone()
    direct_runner = lambda: _run_mixed_page(
        bridge,
        case,
        selected_page_table_i32=None,
        row_consume_mode_i32=None,
        selected_seqused_k_by_head_i32=None,
        seqused_k=selected_seqused,
        max_seqlen_k=selected_k,
        page_resolver_kind=int(PageResolverKind.RESOLVED_ROW_PTR),
        page_resolver_subkind=PAGE_RESOLVER_SUBKIND_DIRECT_TABLE,
        resolved_page_table_i32=direct_table,
        resolved_seqused_k_by_head_i32=selected_by_head,
    )
    direct_out, direct_lse = direct_runner()
    direct_correct, direct_diff = _assert_close_and_diff(
        actual_out=direct_out,
        expected_out=selected_out,
        actual_lse=direct_lse,
        expected_lse=selected_lse,
    )
    direct_us = _time_cuda_us(direct_runner, warmup=warmup, iters=iters)
    direct_spec = CASE_BY_NAME["resolved_direct_table_no_capture"]
    records.append(
        _record(
            spec=direct_spec,
            case=case,
            kv_tokens=selected_k,
            warmup=warmup,
            iters=iters,
            graph_cache_hit=False,
            carrier_update_us=0.0,
            mixed_page_us=direct_us,
            reference_max_abs_diff=direct_diff,
            correctness_passed=direct_correct,
            logit_capture_passed=True,
            route_proof="mixed_page ResolvedRowPtr direct-table no-capture",
            **common,
        )
    )

    affine_tensor = _affine_tensor_from_selected_table(selected_table)
    affine_runner = lambda: _run_mixed_page(
        bridge,
        case,
        selected_page_table_i32=None,
        row_consume_mode_i32=None,
        selected_seqused_k_by_head_i32=None,
        seqused_k=selected_seqused,
        max_seqlen_k=selected_k,
        page_resolver_kind=int(PageResolverKind.RESOLVED_ROW_PTR),
        page_resolver_subkind=PAGE_RESOLVER_SUBKIND_AFFINE_TENSOR,
        resolved_page_table_affine_i32=affine_tensor,
        resolved_seqused_k_by_head_i32=selected_by_head,
    )
    affine_out, affine_lse = affine_runner()
    affine_correct, affine_diff = _assert_close_and_diff(
        actual_out=affine_out,
        expected_out=selected_out,
        actual_lse=affine_lse,
        expected_lse=selected_lse,
    )
    affine_us = _time_cuda_us(affine_runner, warmup=warmup, iters=iters)
    affine_spec = CASE_BY_NAME["resolved_affine_tensor_no_capture"]
    records.append(
        _record(
            spec=affine_spec,
            case=case,
            kv_tokens=selected_k,
            warmup=warmup,
            iters=iters,
            graph_cache_hit=False,
            carrier_update_us=0.0,
            mixed_page_us=affine_us,
            reference_max_abs_diff=affine_diff,
            correctness_passed=affine_correct,
            logit_capture_passed=True,
            route_proof="mixed_page ResolvedRowPtr affine-tensor no-capture",
            **common,
        )
    )

    capture_reference = _dense_capture_reference(
        q=_q_bhd(case),
        key_cache=case.k_cache,
        block_table=selected_table,
        seqused_k=selected_seqused,
        capacity=selected_k,
        softmax_scale=case.softmax_scale,
    )
    capture_rows = torch.arange(case.shape.batch_size, dtype=torch.int32, device=case.q.device)
    capture_last_n = torch.ones(case.shape.batch_size, dtype=torch.int32, device=case.q.device)

    selected_capture_scores = torch.full(
        (case.shape.batch_size, case.shape.num_q_heads, 1, selected_k),
        float("-inf"),
        dtype=torch.float32,
        device=case.q.device,
    )
    selected_capture_runner = lambda: _run_mixed_page(
        bridge,
        case,
        selected_page_table_i32=selected_table,
        row_consume_mode_i32=torch.full(
            (case.shape.batch_size,),
            ROW_CONSUME_MODE_SELECTED_I32,
            dtype=torch.int32,
            device=case.q.device,
        ),
        selected_seqused_k_by_head_i32=selected_by_head,
        seqused_k=selected_seqused,
        max_seqlen_k=selected_k,
        page_resolver_kind=int(PageResolverKind.SELECTED_TABLE),
        capture_scores=selected_capture_scores,
        capture_row_index_i32=capture_rows,
        row_capture_last_n_i32=capture_last_n,
    )
    selected_capture_out, selected_capture_lse = selected_capture_runner()
    selected_capture_correct, selected_capture_diff = _assert_close_and_diff(
        actual_out=selected_capture_out,
        expected_out=selected_out,
        actual_lse=selected_capture_lse,
        expected_lse=selected_lse,
    )
    selected_capture_passed = _capture_passed(
        actual=selected_capture_scores,
        expected=capture_reference,
        valid_k=selected_k,
    )
    selected_capture_us = _time_cuda_us(selected_capture_runner, warmup=warmup, iters=iters)
    selected_capture_spec = CASE_BY_NAME["selected_table_capture"]
    records.append(
        _record(
            spec=selected_capture_spec,
            case=case,
            kv_tokens=selected_k,
            warmup=warmup,
            iters=iters,
            graph_cache_hit=False,
            carrier_update_us=0.0,
            mixed_page_us=selected_capture_us,
            reference_max_abs_diff=selected_capture_diff,
            correctness_passed=selected_capture_correct,
            logit_capture_passed=selected_capture_passed,
            route_proof="mixed_page selected-table capture",
            **common,
        )
    )

    rowptr_capture_scores = torch.full_like(selected_capture_scores, float("-inf"))
    rowptr_capture_runner = lambda: _run_mixed_page(
        bridge,
        case,
        selected_page_table_i32=None,
        row_consume_mode_i32=None,
        selected_seqused_k_by_head_i32=None,
        seqused_k=selected_seqused,
        max_seqlen_k=selected_k,
        page_resolver_kind=int(PageResolverKind.RESOLVED_ROW_PTR),
        page_resolver_subkind=PAGE_RESOLVER_SUBKIND_ROWPTR,
        resolved_page_table_row_ptr_u64=resolved_row_ptr,
        resolved_seqused_k_by_head_i32=selected_by_head,
        capture_scores=rowptr_capture_scores,
        capture_row_index_i32=capture_rows,
        row_capture_last_n_i32=capture_last_n,
    )
    rowptr_capture_out, rowptr_capture_lse = rowptr_capture_runner()
    rowptr_capture_correct, rowptr_capture_diff = _assert_close_and_diff(
        actual_out=rowptr_capture_out,
        expected_out=selected_out,
        actual_lse=rowptr_capture_lse,
        expected_lse=selected_lse,
    )
    rowptr_capture_passed = _capture_passed(
        actual=rowptr_capture_scores,
        expected=capture_reference,
        valid_k=selected_k,
    )
    rowptr_capture_us = _time_cuda_us(rowptr_capture_runner, warmup=warmup, iters=iters)
    rowptr_capture_spec = CASE_BY_NAME["resolved_row_ptr_capture"]
    records.append(
        _record(
            spec=rowptr_capture_spec,
            case=case,
            kv_tokens=selected_k,
            warmup=warmup,
            iters=iters,
            graph_cache_hit=False,
            carrier_update_us=0.0,
            mixed_page_us=rowptr_capture_us,
            reference_max_abs_diff=rowptr_capture_diff,
            correctness_passed=rowptr_capture_correct,
            logit_capture_passed=rowptr_capture_passed,
            route_proof="mixed_page ResolvedRowPtr capture",
            **common,
        )
    )

    selected_graph_us, _, selected_graph_hit, _ = _measure_graph_runner(
        selected_runner,
        warmup=warmup,
        iters=iters,
    )
    rowptr_graph_us, rowptr_graph_out, rowptr_graph_hit, rowptr_graph = _measure_graph_runner(
        resolved_rowptr_runner,
        warmup=warmup,
        iters=iters,
    )
    rowptr_graph_correct = bool(
        torch.allclose(rowptr_graph_out, selected_out, atol=0.0, rtol=0.0)
    )
    graph_carriers = {
        "selected_page_table_i32": selected_table,
        "resolved_page_table_row_ptr_u64": resolved_row_ptr,
        "resolved_seqused_k_by_head_i32": selected_by_head,
    }
    rowptr_update_sources = {
        "selected_page_table_i32": selected_table.clone(),
        "resolved_seqused_k_by_head_i32": selected_by_head.clone(),
    }
    prebound_rowptr_update_us = _time_copy_us(
        lambda: (
            selected_table.copy_(rowptr_update_sources["selected_page_table_i32"]),
            selected_by_head.copy_(rowptr_update_sources["resolved_seqused_k_by_head_i32"]),
        ),
        iters=iters,
    )
    rowptr_signature_before = carrier_pointer_signature(graph_carriers)
    mutate_carriers_in_place(
        graph_carriers,
        page_block_size=int(case.shape.page_size),
        max_page_id=int(case.page_table_i32.max().item()),
    )
    rowptr_signature_after = carrier_pointer_signature(graph_carriers)
    changed_selected_seqused = selected_by_head.view(
        int(case.shape.batch_size),
        int(case.shape.num_kv_heads),
    )[:, 0].contiguous()
    changed_selected_out, _ = _run_native(
        bridge,
        case,
        page_table=selected_table,
        seqused_k=changed_selected_seqused,
        max_seqlen_k=int(changed_selected_seqused.max().item()),
    )
    rowptr_graph.replay()
    torch.cuda.synchronize()
    changed_mapping_output_allclose = bool(
        torch.allclose(rowptr_graph_out, changed_selected_out, atol=0.0, rtol=0.0)
    )
    stale_carrier_negative_passed = rowptr_signature_before == rowptr_signature_after
    external_rowptr_update_us = prebound_rowptr_update_us
    rowptr_graph_spec = CASE_BY_NAME["resolved_row_ptr_graph_replay"]
    records.append(
        _record(
            spec=rowptr_graph_spec,
            case=case,
            kv_tokens=selected_k,
            warmup=warmup,
            iters=iters,
            graph_cache_hit=rowptr_graph_hit,
            carrier_update_us=prebound_rowptr_update_us,
            mixed_page_us=rowptr_graph_us,
            reference_max_abs_diff=float(
                (rowptr_graph_out.to(torch.float32) - changed_selected_out.to(torch.float32)).abs().max().item()
            ),
            correctness_passed=rowptr_graph_correct and changed_mapping_output_allclose,
            logit_capture_passed=True,
            route_proof="CUDA graph replay: prebound ResolvedRowPtr carriers",
            carrier_pointer_signature_before=rowptr_signature_before,
            carrier_pointer_signature_after=rowptr_signature_after,
            graph_recapture_count=0,
            stale_carrier_negative_passed=stale_carrier_negative_passed,
            changed_mapping_output_allclose=changed_mapping_output_allclose,
            **common,
        )
    )

    summary = summarize_records(
        records,
        max_graph_overhead_pct=float(args.max_graph_overhead_pct),
        max_existing_regression_pct=float(args.max_existing_regression_pct),
        selected_table_graph_replay_us=selected_graph_us,
        resolved_row_ptr_graph_replay_us=rowptr_graph_us,
        resolved_row_ptr_carrier_update_us=prebound_rowptr_update_us,
        external_resolved_row_ptr_carrier_update_us=external_rowptr_update_us,
        existing_regression_pct=None,
    )
    return records, summary


def run_gate(args: argparse.Namespace) -> int:
    output = Path(str(args.output))
    summary_output = Path(str(args.summary_output))
    extension_path = _extension_path()
    git_sha = _git_sha()
    cuda_device_name = _cuda_device_name()
    if not torch.cuda.is_available():
        records = _unavailable_records(
            reason="CUDA is not available",
            warmup=int(args.warmup),
            iters=int(args.iters),
            extension_path=extension_path,
            git_sha=git_sha,
            cuda_device_name=cuda_device_name,
        )
        summary = summarize_records(
            records,
            max_graph_overhead_pct=float(args.max_graph_overhead_pct),
            max_existing_regression_pct=float(args.max_existing_regression_pct),
        )
        write_records(output, records)
        write_summary(summary_output, summary)
        return exit_code_for_summary(summary)

    try:
        records, summary = _run_available_gate(args)
    except Exception as exc:
        records = _unavailable_records(
            reason=f"runtime gate failed: {exc.__class__.__name__}: {exc}",
            warmup=int(args.warmup),
            iters=int(args.iters),
            extension_path=extension_path,
            git_sha=git_sha,
            cuda_device_name=cuda_device_name,
        )
        summary = summarize_records(
            records,
            max_graph_overhead_pct=float(args.max_graph_overhead_pct),
            max_existing_regression_pct=float(args.max_existing_regression_pct),
        )
    write_records(output, records)
    write_summary(summary_output, summary)
    return exit_code_for_summary(summary)


def main(argv: list[str] | None = None) -> int:
    return run_gate(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
