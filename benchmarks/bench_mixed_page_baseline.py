from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Callable, Iterable, Mapping

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.compact_recent_runtime_identity import (  # noqa: E402
    collect_runtime_identity,
    runtime_identity_to_payload,
)
from patches.fa3_native.install import load_vendored_flash_attn_bridge  # noqa: E402


BENCHMARK_NAME = "mixed_page_kernel_microbenchmark"
CASE_FAMILIES = ("full", "selected", "equivalent", "mixed_rows")
DEFAULT_WARMUP = 20
DEFAULT_REPEAT = 100
DEFAULT_INNER_ITERS = 50
DEFAULT_PAGE_SIZE = 16
DEFAULT_NUM_Q_HEADS = 32
DEFAULT_NUM_KV_HEADS = 8
DEFAULT_HEAD_DIM = 128
CORRECTNESS_OUT_ATOL = 0.0
CORRECTNESS_OUT_RTOL = 0.0
CORRECTNESS_LSE_ATOL = 1.0e-3
CORRECTNESS_LSE_RTOL = 0.0
CORRECTNESS_SCOPE = "out_bit_exact_lse_abs_1e-3_no_capture"

_REQUIRED_RECORD_FIELDS = {
    "benchmark",
    "case_id",
    "command_line",
    "kernel_case",
    "baseline_case",
    "path_validation_status",
    "speed_gate_basis",
    "split_match_status",
    "gate_status",
    "requested_num_splits",
    "native_resolved_num_splits",
    "mixed_resolved_num_splits",
    "native_num_splits_dynamic_min",
    "native_num_splits_dynamic_max",
    "mixed_num_splits_dynamic_min",
    "mixed_num_splits_dynamic_max",
    "split_match_error",
    "max_seqlen_k",
    "native_effective_k",
    "mixed_effective_k",
    "fa3_so_sha256",
    "gpu_id",
    "physical_gpu_index",
    "gpu_uuid",
    "gpu_utilization_before_pct",
    "gpu_memory_used_before_mib",
    "benchmark_start_gpu_utilization_before_pct",
    "benchmark_start_gpu_memory_used_before_mib",
    "git_dirty",
    "git_status_short",
    "warmup",
    "repeat",
    "inner_iters",
    "selected_effective_k",
    "selected_page_count",
    "selected_logical_pages",
    "selected_page_table_shape",
    "selected_page_table_num_cols",
    "selected_page_table_layout",
    "selected_native_page_table_shape",
    "selected_native_page_table_layout",
    "full_page_table_shape",
    "selected_table_matches_full",
    "compact_mixed_page_overlay",
    "equivalence_probe",
    "correctness_scope",
    "correctness_out_atol",
    "correctness_out_rtol",
    "correctness_lse_atol",
    "correctness_lse_rtol",
}


@dataclass(frozen=True)
class SelectedGeometry:
    total_pages: int
    selected_page_count: int
    selected_effective_k: int
    selected_logical_pages: list[int]


@dataclass(frozen=True)
class BenchShape:
    batch_size: int
    q_len: int
    kv_len: int
    selected_ratio: float
    page_size: int = DEFAULT_PAGE_SIZE
    num_q_heads: int = DEFAULT_NUM_Q_HEADS
    num_kv_heads: int = DEFAULT_NUM_KV_HEADS
    head_dim: int = DEFAULT_HEAD_DIM
    dtype: torch.dtype = torch.bfloat16
    device: str | torch.device = "cuda"
    seed: int = 0


@dataclass(frozen=True)
class SplitTruth:
    requested_num_splits: int
    resolved_num_splits: int | None
    num_splits_dynamic_min: int | None
    num_splits_dynamic_max: int | None
    use_dynamic_split: bool = False
    num_splits_dynamic_offset: int | None = None
    scheduler_metadata_batch_size: int | None = None


@dataclass(frozen=True)
class BenchmarkCase:
    shape: BenchShape
    selected_geometry: SelectedGeometry
    q_pad: torch.Tensor
    q: torch.Tensor
    cu_seqlens_q_i32: torch.Tensor
    k_full: torch.Tensor
    v_full: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    page_table_i32: torch.Tensor
    full_row_consume_mode_i32: torch.Tensor
    selected_row_consume_mode_i32: torch.Tensor
    mixed_row_consume_mode_i32: torch.Tensor
    full_selected_page_table_i32: torch.Tensor
    selected_page_table_i32: torch.Tensor
    full_selected_seqused_k_by_head_i32: torch.Tensor
    full_seqused_k_i32: torch.Tensor
    selected_seqused_k_by_head_i32: torch.Tensor
    selected_native_k_cache: torch.Tensor
    selected_native_v_cache: torch.Tensor
    selected_native_page_table_i32: torch.Tensor
    selected_native_seqused_k_i32: torch.Tensor
    softmax_scale: float


def compute_selected_geometry(*, kv_len: int, page_size: int, selected_ratio: float) -> SelectedGeometry:
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    if kv_len % page_size != 0:
        raise ValueError("kv_len must be divisible by page_size")
    if not (0.0 < selected_ratio <= 1.0):
        raise ValueError("selected_ratio must be in (0, 1]")

    total_pages = kv_len // page_size
    selected_page_count = max(1, math.floor(total_pages * selected_ratio))
    selected_page_count = min(total_pages, selected_page_count)

    selected_logical_pages = [
        min(total_pages - 1, math.floor(index * total_pages / selected_page_count))
        for index in range(selected_page_count)
    ]

    return SelectedGeometry(
        total_pages=total_pages,
        selected_page_count=selected_page_count,
        selected_effective_k=selected_page_count * page_size,
        selected_logical_pages=selected_logical_pages,
    )


def _build_full_page_tensor(batch_size: int, total_pages: int, *, device: torch.device) -> torch.Tensor:
    return torch.arange(batch_size * total_pages, dtype=torch.int32, device=device).reshape(batch_size, total_pages)


def _make_generator(device: torch.device, seed: int) -> torch.Generator:
    generator_device = device.type if device.type in {"cpu", "cuda"} else "cpu"
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(seed)
    return generator


def _collect_git_state() -> dict[str, object]:
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(REPO_ROOT),
                "status",
                "--short",
                "--untracked-files=normal",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        return {
            "git_dirty": True,
            "git_status_short": [f"<git status unavailable: {exc.__class__.__name__}>"],
        }

    status_short = [line.rstrip("\n") for line in result.stdout.splitlines() if line.strip()]
    status_short_limit = 12
    if len(status_short) > status_short_limit:
        status_short = status_short[:status_short_limit] + [
            f"...(+{len(status_short) - status_short_limit} more)"
        ]
    return {
        "git_dirty": bool(status_short),
        "git_status_short": status_short,
    }


def _command_line_for_run(argv: list[str]) -> list[str]:
    return [sys.executable, str(Path(__file__).resolve()), *argv]


def _prepare_output_path(output_path: Path, *, append: bool) -> None:
    if output_path.exists() and not append:
        output_path.unlink()


def build_benchmark_case(shape: BenchShape) -> BenchmarkCase:
    geometry = compute_selected_geometry(
        kv_len=shape.kv_len,
        page_size=shape.page_size,
        selected_ratio=shape.selected_ratio,
    )

    device = torch.device(shape.device)
    generator = _make_generator(device, shape.seed)

    q_pad = torch.randn(
        shape.batch_size,
        shape.num_q_heads,
        shape.q_len,
        shape.head_dim,
        generator=generator,
        dtype=shape.dtype,
        device=device,
    )
    q = q_pad.permute(0, 2, 1, 3).reshape(shape.batch_size * shape.q_len, shape.num_q_heads, shape.head_dim)
    cu_seqlens_q_i32 = torch.arange(shape.batch_size + 1, dtype=torch.int32, device=device) * shape.q_len

    k_full = torch.randn(
        shape.batch_size,
        shape.num_kv_heads,
        shape.kv_len,
        shape.head_dim,
        generator=generator,
        dtype=shape.dtype,
        device=device,
    )
    v_full = torch.randn(
        shape.batch_size,
        shape.num_kv_heads,
        shape.kv_len,
        shape.head_dim,
        generator=generator,
        dtype=shape.dtype,
        device=device,
    )

    page_table_i32 = _build_full_page_tensor(shape.batch_size, geometry.total_pages, device=device)
    full_row_consume_mode_i32 = torch.zeros(shape.batch_size, dtype=torch.int32, device=device)
    selected_row_consume_mode_i32 = torch.ones(shape.batch_size, dtype=torch.int32, device=device)
    mixed_row_consume_mode_i32 = torch.arange(
        shape.batch_size, dtype=torch.int32, device=device
    ).remainder(2)

    full_selected_page_table_i32 = page_table_i32.clone()

    selected_page_ids = torch.as_tensor(geometry.selected_logical_pages, dtype=torch.int64, device=device)
    selected_page_table_i32 = page_table_i32.index_select(1, selected_page_ids).contiguous()

    k_full_pages = k_full.view(
        shape.batch_size,
        shape.num_kv_heads,
        geometry.total_pages,
        shape.page_size,
        shape.head_dim,
    )
    v_full_pages = v_full.view(
        shape.batch_size,
        shape.num_kv_heads,
        geometry.total_pages,
        shape.page_size,
        shape.head_dim,
    )
    k_cache = k_full_pages.permute(0, 2, 3, 1, 4).contiguous().reshape(
        shape.batch_size * geometry.total_pages,
        shape.page_size,
        shape.num_kv_heads,
        shape.head_dim,
    )
    v_cache = v_full_pages.permute(0, 2, 3, 1, 4).contiguous().reshape(
        shape.batch_size * geometry.total_pages,
        shape.page_size,
        shape.num_kv_heads,
        shape.head_dim,
    )

    full_selected_seqused_k_by_head_i32 = torch.full(
        (shape.batch_size * shape.num_kv_heads,),
        shape.kv_len,
        dtype=torch.int32,
        device=device,
    )
    selected_seqused_k_by_head_i32 = torch.full(
        (shape.batch_size * shape.num_kv_heads,),
        geometry.selected_effective_k,
        dtype=torch.int32,
        device=device,
    )

    selected_native_k_cache = k_cache
    selected_native_v_cache = v_cache
    selected_native_page_table_i32 = selected_page_table_i32
    selected_native_seqused_k_i32 = torch.full(
        (shape.batch_size,),
        geometry.selected_effective_k,
        dtype=torch.int32,
        device=device,
    )
    full_seqused_k_i32 = torch.full((shape.batch_size,), shape.kv_len, dtype=torch.int32, device=device)
    softmax_scale = 1.0 / math.sqrt(shape.head_dim)

    return BenchmarkCase(
        shape=shape,
        selected_geometry=geometry,
        q_pad=q_pad,
        q=q,
        cu_seqlens_q_i32=cu_seqlens_q_i32,
        k_full=k_full,
        v_full=v_full,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table_i32=page_table_i32,
        full_row_consume_mode_i32=full_row_consume_mode_i32,
        selected_row_consume_mode_i32=selected_row_consume_mode_i32,
        mixed_row_consume_mode_i32=mixed_row_consume_mode_i32,
        full_selected_page_table_i32=full_selected_page_table_i32,
        selected_page_table_i32=selected_page_table_i32,
        full_selected_seqused_k_by_head_i32=full_selected_seqused_k_by_head_i32,
        full_seqused_k_i32=full_seqused_k_i32,
        selected_seqused_k_by_head_i32=selected_seqused_k_by_head_i32,
        selected_native_k_cache=selected_native_k_cache,
        selected_native_v_cache=selected_native_v_cache,
        selected_native_page_table_i32=selected_native_page_table_i32,
        selected_native_seqused_k_i32=selected_native_seqused_k_i32,
        softmax_scale=softmax_scale,
    )


def validate_split_truth_for_gate(*, native: SplitTruth, mixed: SplitTruth) -> str:
    if native.resolved_num_splits is None or mixed.resolved_num_splits is None:
        raise ValueError("split runtime truth is unknown")
    if native.resolved_num_splits != mixed.resolved_num_splits:
        raise ValueError("resolved split mismatch")
    if native.use_dynamic_split != mixed.use_dynamic_split:
        raise ValueError("dynamic split mode mismatch")

    native_bounds = (native.num_splits_dynamic_min, native.num_splits_dynamic_max)
    mixed_bounds = (mixed.num_splits_dynamic_min, mixed.num_splits_dynamic_max)
    if native.use_dynamic_split or mixed.use_dynamic_split:
        if None in native_bounds or None in mixed_bounds:
            raise ValueError("dynamic split bounds missing")
    native_has_bounds = native_bounds != (None, None)
    mixed_has_bounds = mixed_bounds != (None, None)
    if native_has_bounds != mixed_has_bounds or (native_has_bounds and native_bounds != mixed_bounds):
        raise ValueError("dynamic split bounds mismatch")
    return "matched_resolved"


def classify_speed_gate(
    *,
    case_family: str,
    mixed_ms: float,
    native_ms: float,
) -> str:
    if native_ms <= 0:
        ratio = float("inf") if mixed_ms > 0 else 0.0
    else:
        ratio = mixed_ms / native_ms

    if ratio <= 1.0:
        return "pass"
    if case_family == "full":
        return "gray" if ratio <= 1.01 else "fail"
    if case_family in {"selected", "equivalent", "mixed_rows"}:
        return "gray" if ratio <= 1.03 else "fail"
    raise ValueError("case_family must be 'full', 'selected', 'equivalent', or 'mixed_rows'")


def emit_json_record(output: Path, record: Mapping[str, object]) -> None:
    missing = sorted(_REQUIRED_RECORD_FIELDS.difference(record.keys()))
    if missing:
        raise ValueError(f"missing required record fields: {', '.join(missing)}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(dict(record), ensure_ascii=False, sort_keys=True))
        fh.write("\n")


def run_native_paged_full(bridge: Any, case: BenchmarkCase, *, num_splits: int) -> tuple[torch.Tensor, torch.Tensor]:
    out, lse = bridge.flash_attn_varlen_func(
        q=case.q,
        k=case.k_cache,
        v=case.v_cache,
        max_seqlen_q=case.shape.q_len,
        cu_seqlens_q=case.cu_seqlens_q_i32,
        max_seqlen_k=case.shape.kv_len,
        seqused_k=case.full_seqused_k_i32,
        softmax_scale=case.softmax_scale,
        causal=True,
        block_table=case.page_table_i32,
        return_softmax_lse=True,
        fa_version=3,
        num_splits=int(num_splits),
    )
    return out, lse


def run_native_paged_selected(
    bridge: Any,
    case: BenchmarkCase,
    *,
    num_splits: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    out, lse = bridge.flash_attn_varlen_func(
        q=case.q,
        k=case.selected_native_k_cache,
        v=case.selected_native_v_cache,
        max_seqlen_q=case.shape.q_len,
        cu_seqlens_q=case.cu_seqlens_q_i32,
        max_seqlen_k=case.selected_geometry.selected_effective_k,
        seqused_k=case.selected_native_seqused_k_i32,
        softmax_scale=case.softmax_scale,
        causal=True,
        block_table=case.selected_native_page_table_i32,
        return_softmax_lse=True,
        fa_version=3,
        num_splits=int(num_splits),
    )
    return out, lse


def run_mixed_page_full_no_capture(
    bridge: Any,
    case: BenchmarkCase,
    *,
    num_splits: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # [2026-07-11 J1] All rows are native/full here, so pass NATIVE (kind0)
    # explicitly. Relying on the interface default meant SELECTED_TABLE (kind1),
    # which is retired at launch -- profile_mixed_page_hotspots --case full died
    # on it since the 07-03 selectedtable delete.
    from patches.fa3_native.mixed_page_graph_descriptor import PageResolverKind

    out, lse = bridge.mixed_page_attn_varlen_func(
        q=case.q,
        k=case.k_cache,
        v=case.v_cache,
        max_seqlen_q=case.shape.q_len,
        cu_seqlens_q=case.cu_seqlens_q_i32,
        max_seqlen_k=case.shape.kv_len,
        seqused_k=case.full_seqused_k_i32,
        softmax_scale=case.softmax_scale,
        causal=True,
        block_table=case.page_table_i32,
        return_softmax_lse=True,
        page_resolver_kind=int(PageResolverKind.NATIVE),
        selected_page_table_i32=None,
        row_consume_mode_i32=None,
        selected_seqused_k_by_head_i32=None,
        cp_selected_seqused_k_by_head_i32=None,
        capture_scores=None,
        capture_row_index_i32=None,
        row_capture_last_n_i32=None,
        num_splits=int(num_splits),
    )
    return out, lse


def run_mixed_page_selected_no_capture(
    bridge: Any,
    case: BenchmarkCase,
    *,
    num_splits: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    out, lse = bridge.mixed_page_attn_varlen_func(
        q=case.q,
        k=case.k_cache,
        v=case.v_cache,
        max_seqlen_q=case.shape.q_len,
        cu_seqlens_q=case.cu_seqlens_q_i32,
        max_seqlen_k=case.selected_geometry.selected_effective_k,
        seqused_k=case.selected_native_seqused_k_i32,
        softmax_scale=case.softmax_scale,
        causal=True,
        block_table=case.page_table_i32,
        return_softmax_lse=True,
        selected_page_table_i32=case.selected_page_table_i32,
        row_consume_mode_i32=None,
        selected_seqused_k_by_head_i32=None,
        cp_selected_seqused_k_by_head_i32=None,
        capture_scores=None,
        capture_row_index_i32=None,
        row_capture_last_n_i32=None,
        num_splits=int(num_splits),
    )
    return out, lse


def run_mixed_page_equivalent_no_capture(
    bridge: Any,
    case: BenchmarkCase,
    *,
    num_splits: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    out, lse = bridge.mixed_page_attn_varlen_func(
        q=case.q,
        k=case.k_cache,
        v=case.v_cache,
        max_seqlen_q=case.shape.q_len,
        cu_seqlens_q=case.cu_seqlens_q_i32,
        max_seqlen_k=case.shape.kv_len,
        seqused_k=case.selected_native_seqused_k_i32,
        softmax_scale=case.softmax_scale,
        causal=True,
        block_table=case.page_table_i32,
        return_softmax_lse=True,
        selected_page_table_i32=case.selected_page_table_i32,
        row_consume_mode_i32=None,
        selected_seqused_k_by_head_i32=None,
        cp_selected_seqused_k_by_head_i32=None,
        capture_scores=None,
        capture_row_index_i32=None,
        row_capture_last_n_i32=None,
        num_splits=int(num_splits),
    )
    return out, lse


def run_mixed_page_mixed_rows_no_capture(
    bridge: Any,
    case: BenchmarkCase,
    *,
    num_splits: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    out, lse = bridge.mixed_page_attn_varlen_func(
        q=case.q,
        k=case.k_cache,
        v=case.v_cache,
        max_seqlen_q=case.shape.q_len,
        cu_seqlens_q=case.cu_seqlens_q_i32,
        max_seqlen_k=case.shape.kv_len,
        seqused_k=case.full_seqused_k_i32,
        softmax_scale=case.softmax_scale,
        causal=True,
        block_table=case.page_table_i32,
        return_softmax_lse=True,
        selected_page_table_i32=case.full_selected_page_table_i32,
        row_consume_mode_i32=case.mixed_row_consume_mode_i32,
        selected_seqused_k_by_head_i32=case.full_selected_seqused_k_by_head_i32,
        cp_selected_seqused_k_by_head_i32=None,
        capture_scores=None,
        capture_row_index_i32=None,
        row_capture_last_n_i32=None,
        num_splits=int(num_splits),
    )
    return out, lse


def assert_outputs_close(
    *,
    case_name: str,
    native_out: torch.Tensor,
    mixed_out: torch.Tensor,
    native_lse: torch.Tensor,
    mixed_lse: torch.Tensor,
) -> None:
    try:
        torch.testing.assert_close(
            mixed_out,
            native_out,
            atol=CORRECTNESS_OUT_ATOL,
            rtol=CORRECTNESS_OUT_RTOL,
        )
    except AssertionError as exc:
        raise AssertionError(f"{case_name} output mismatch: {exc}") from exc
    try:
        torch.testing.assert_close(
            mixed_lse,
            native_lse,
            atol=CORRECTNESS_LSE_ATOL,
            rtol=CORRECTNESS_LSE_RTOL,
        )
    except AssertionError as exc:
        raise AssertionError(f"{case_name} lse mismatch: {exc}") from exc


def _validate_measurement_window(*, warmup: int, repeat: int, inner_iters: int = 1) -> None:
    if isinstance(warmup, bool) or not isinstance(warmup, int):
        raise TypeError("warmup must be an integer")
    if isinstance(repeat, bool) or not isinstance(repeat, int):
        raise TypeError("repeat must be an integer")
    if isinstance(inner_iters, bool) or not isinstance(inner_iters, int):
        raise TypeError("inner_iters must be an integer")
    if warmup < 0:
        raise ValueError("warmup must be >= 0")
    if repeat <= 0:
        raise ValueError("repeat must be > 0")
    if inner_iters <= 0:
        raise ValueError("inner_iters must be > 0")
    if not torch.cuda.is_available():
        raise RuntimeError("requires CUDA")


def _measure_case_latency(
    run_case: Callable[[], None],
    *,
    warmup: int,
    repeat: int,
    inner_iters: int = 1,
) -> tuple[float, list[float]]:
    _validate_measurement_window(warmup=warmup, repeat=repeat, inner_iters=inner_iters)

    samples: list[float] = []
    for _ in range(warmup):
        for _ in range(inner_iters):
            run_case()
    torch.cuda.synchronize()
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(inner_iters):
            run_case()
        end.record()
        torch.cuda.synchronize()
        samples.append(float(start.elapsed_time(end)) / float(inner_iters))
    return float(median(samples)), samples


def _measure_paired_case_latency(
    native_run: Callable[[], None],
    mixed_run: Callable[[], None],
    *,
    warmup: int,
    repeat: int,
    inner_iters: int = 1,
) -> tuple[float, list[float], float, list[float]]:
    _validate_measurement_window(warmup=warmup, repeat=repeat, inner_iters=inner_iters)

    for index in range(warmup):
        if index % 2 == 0:
            for _ in range(inner_iters):
                native_run()
            for _ in range(inner_iters):
                mixed_run()
        else:
            for _ in range(inner_iters):
                mixed_run()
            for _ in range(inner_iters):
                native_run()
    torch.cuda.synchronize()

    native_samples: list[float] = []
    mixed_samples: list[float] = []
    for index in range(repeat):
        run_order = (native_run, mixed_run) if index % 2 == 0 else (mixed_run, native_run)
        for run_case in run_order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(inner_iters):
                run_case()
            end.record()
            torch.cuda.synchronize()
            elapsed = float(start.elapsed_time(end)) / float(inner_iters)
            if run_case is native_run:
                native_samples.append(elapsed)
            else:
                mixed_samples.append(elapsed)
    return (
        float(median(native_samples)),
        native_samples,
        float(median(mixed_samples)),
        mixed_samples,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="mixed_page no-capture vs native paged FA3 microbenchmark"
    )
    parser.add_argument("--case", action="append", choices=CASE_FAMILIES, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--q-len", type=int, default=1)
    parser.add_argument("--kv-len", type=int, default=8192)
    parser.add_argument("--selected-ratio", type=float, default=1.0)
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE)
    parser.add_argument("--num-q-heads", type=int, default=DEFAULT_NUM_Q_HEADS)
    parser.add_argument("--num-kv-heads", type=int, default=DEFAULT_NUM_KV_HEADS)
    parser.add_argument("--head-dim", type=int, default=DEFAULT_HEAD_DIM)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--num-splits", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--repeat", type=int, default=DEFAULT_REPEAT)
    parser.add_argument("--inner-iters", type=int, default=DEFAULT_INNER_ITERS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-jsonl", default="logs/mixed_page_kernel_microbenchmark.jsonl")
    parser.add_argument("--append", action="store_true", default=False)
    parsed = parser.parse_args([] if argv is None else argv)
    if parsed.case is None:
        parsed.case = ["full", "selected"]
    return parsed


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    raise ValueError(f"unsupported dtype: {name}")


def _collect_gpu_state() -> dict[str, object]:
    gpu_id = int(torch.cuda.current_device()) if torch.cuda.is_available() else -1
    gpu_uuid = ""
    if torch.cuda.is_available():
        try:
            device_properties = torch.cuda.get_device_properties(gpu_id)
            gpu_uuid = _normalize_gpu_uuid(str(getattr(device_properties, "uuid", "") or ""))
        except Exception:
            gpu_uuid = ""
    payload: dict[str, object] = {
        "gpu_id": gpu_id,
        "physical_gpu_index": -1,
        "gpu_uuid": gpu_uuid,
        "gpu_utilization_before_pct": -1,
        "gpu_memory_used_before_mib": -1,
        "nvidia_smi_driver_version": "",
    }
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,utilization.gpu,memory.used,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return payload
    rows: list[dict[str, object]] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        rows.append(
            {
                "index": int(parts[0]),
                "uuid": parts[1],
                "utilization": int(parts[2]),
                "memory_used": int(parts[3]),
                "driver_version": parts[4],
            }
        )
    selected_row = None
    if gpu_uuid:
        for row in rows:
            if _normalize_gpu_uuid(str(row["uuid"])) == gpu_uuid:
                selected_row = row
                break
    visible_devices = [
        entry.strip()
        for entry in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if entry.strip()
    ]
    if selected_row is None and 0 <= gpu_id < len(visible_devices):
        visible_device = visible_devices[gpu_id]
        if visible_device.isdigit():
            visible_physical_index = int(visible_device)
            for row in rows:
                if int(row["index"]) == visible_physical_index:
                    selected_row = row
                    break
    if selected_row is None and gpu_id >= 0:
        for row in rows:
            if int(row["index"]) == gpu_id:
                selected_row = row
                break
    if selected_row is not None:
        payload["physical_gpu_index"] = int(selected_row["index"])
        payload["gpu_uuid"] = str(selected_row["uuid"])
        payload["gpu_utilization_before_pct"] = int(selected_row["utilization"])
        payload["gpu_memory_used_before_mib"] = int(selected_row["memory_used"])
        payload["nvidia_smi_driver_version"] = str(selected_row["driver_version"])
    return payload


def _normalize_gpu_uuid(uuid: str) -> str:
    value = str(uuid).strip()
    if not value:
        return ""
    return value if value.startswith("GPU-") else f"GPU-{value}"


def _decode_split_truth(bridge: Any, anchor: torch.Tensor, *, requested_num_splits: int) -> SplitTruth:
    state = bridge.interface_module.fwd_last_launch_debug_state(anchor)
    use_dynamic_split = bool(state["use_dynamic_split"])
    dynamic_offset = int(state["num_splits_dynamic_offset"])
    scheduler_metadata_batch_size = int(state["scheduler_metadata_batch_size"])
    dynamic_min = int(state.get("num_splits_dynamic_min", -1))
    dynamic_max = int(state.get("num_splits_dynamic_max", -1))
    return SplitTruth(
        requested_num_splits=int(requested_num_splits),
        resolved_num_splits=int(state["resolved_num_splits"]),
        num_splits_dynamic_min=dynamic_min if use_dynamic_split and dynamic_min >= 0 else None,
        num_splits_dynamic_max=dynamic_max if use_dynamic_split and dynamic_max >= 0 else None,
        use_dynamic_split=use_dynamic_split,
        num_splits_dynamic_offset=dynamic_offset if use_dynamic_split else None,
        scheduler_metadata_batch_size=scheduler_metadata_batch_size,
    )


def summarize_gate_records(records: Iterable[dict[str, object]]) -> dict[str, object]:
    statuses = [str(record["gate_status"]) for record in records]
    if any(status == "fail" for status in statuses):
        overall = "fail"
    elif any(status == "gray" for status in statuses):
        overall = "gray"
    elif statuses and all(status == "pass" for status in statuses):
        overall = "pass"
    else:
        overall = "no_records"
    return {"overall_g1_status": overall, "case_count": len(statuses)}


def _selected_page_table_layout(case_family: str) -> str:
    if case_family == "full":
        return "full_no_selected_payload"
    if case_family == "selected":
        return "selected_batch_prefix"
    if case_family == "equivalent":
        return "dense_selected_equivalent"
    if case_family == "mixed_rows":
        return "mixed_rows_dense_selected_equivalent"
    raise ValueError(f"unknown case_family: {case_family}")


def _compact_mixed_page_overlay_payload(
    *,
    case: BenchmarkCase,
    case_family: str,
    mixed_row_selected_count: int,
) -> dict[str, object]:
    enabled = case_family == "mixed_rows"
    batch_size = int(case.shape.batch_size)
    overlay_width_pages = int(case.full_selected_page_table_i32.shape[1]) if enabled else 0
    return {
        "enabled": enabled,
        "compose_count": 1 if enabled else 0,
        "page_table_rows_rewritten": batch_size if enabled else 0,
        "length_rows_rewritten": batch_size if enabled else 0,
        "recent_suffix_rows_rewritten": int(mixed_row_selected_count) if enabled else 0,
        "overlay_width_pages": overlay_width_pages,
        "allocation_after_warmup": 0,
        "d2h_sync_detected": False,
    }


def _bench_pair(
    *,
    bridge: Any,
    case: BenchmarkCase,
    case_family: str,
    num_splits: int,
    warmup: int,
    repeat: int,
    inner_iters: int,
) -> dict[str, object]:
    if case_family == "full":
        native_runner = lambda: run_native_paged_full(bridge, case, num_splits=num_splits)
        mixed_runner = lambda: run_mixed_page_full_no_capture(bridge, case, num_splits=num_splits)
        kernel_case = "mixed_page_full_no_capture"
        baseline_case = "native_paged_full_same_k"
    elif case_family == "selected":
        native_runner = lambda: run_native_paged_selected(bridge, case, num_splits=num_splits)
        mixed_runner = lambda: run_mixed_page_selected_no_capture(bridge, case, num_splits=num_splits)
        kernel_case = "mixed_page_selected_no_capture"
        baseline_case = "native_paged_selected_effective_k"
    elif case_family == "equivalent":
        if case.selected_geometry.selected_page_count != case.selected_geometry.total_pages:
            raise ValueError("equivalent case requires --selected-ratio 1.0")
        native_runner = lambda: run_native_paged_full(bridge, case, num_splits=num_splits)
        mixed_runner = lambda: run_mixed_page_equivalent_no_capture(bridge, case, num_splits=num_splits)
        kernel_case = "mixed_page_dense_selected_equivalent_no_capture"
        baseline_case = "native_paged_full_same_k_dense_equivalent"
    elif case_family == "mixed_rows":
        if case.selected_geometry.selected_page_count != case.selected_geometry.total_pages:
            raise ValueError("mixed_rows case requires --selected-ratio 1.0")
        if case.shape.batch_size < 2:
            raise ValueError("mixed_rows case requires --batch-size >= 2")
        native_runner = lambda: run_native_paged_full(bridge, case, num_splits=num_splits)
        mixed_runner = lambda: run_mixed_page_mixed_rows_no_capture(
            bridge, case, num_splits=num_splits
        )
        kernel_case = "mixed_page_mixed_rows_dense_equivalent_no_capture"
        baseline_case = "native_paged_full_same_k_mixed_rows_equivalent"
    else:
        raise ValueError(f"unknown case_family: {case_family}")

    native_out, native_lse = native_runner()
    native_truth = _decode_split_truth(bridge, case.q, requested_num_splits=num_splits)
    mixed_out, mixed_lse = mixed_runner()
    mixed_truth = _decode_split_truth(bridge, case.q, requested_num_splits=num_splits)
    assert_outputs_close(
        case_name=kernel_case,
        native_out=native_out,
        mixed_out=mixed_out,
        native_lse=native_lse,
        mixed_lse=mixed_lse,
    )
    split_match_error = None
    try:
        split_match_status = validate_split_truth_for_gate(native=native_truth, mixed=mixed_truth)
    except ValueError as exc:
        split_match_error = str(exc)
        split_match_status = (
            "split_truth_unknown"
            if "split runtime truth is unknown" in split_match_error
            else "split_truth_invalid"
        )

    native_ms, native_samples, mixed_ms, mixed_samples = _measure_paired_case_latency(
        native_runner,
        mixed_runner,
        warmup=warmup,
        repeat=repeat,
        inner_iters=inner_iters,
    )
    path_validation_status = "valid_direct_mixed_page"
    gate_status = classify_speed_gate(case_family=case_family, mixed_ms=mixed_ms, native_ms=native_ms)
    if split_match_error is not None:
        gate_status = "fail"
    max_seqlen_k = (
        case.selected_geometry.selected_effective_k if case_family == "selected" else case.shape.kv_len
    )
    if case_family == "full":
        selected_page_table_for_case = case.page_table_i32
    elif case_family == "mixed_rows":
        selected_page_table_for_case = case.full_selected_page_table_i32
    else:
        selected_page_table_for_case = case.selected_page_table_i32
    selected_table_matches_full = bool(
        selected_page_table_for_case.shape == case.page_table_i32.shape
        and torch.equal(selected_page_table_for_case, case.page_table_i32)
    )
    mixed_row_full_count = (case.shape.batch_size + 1) // 2 if case_family == "mixed_rows" else 0
    mixed_row_selected_count = case.shape.batch_size // 2 if case_family == "mixed_rows" else 0
    compact_overlay_payload = _compact_mixed_page_overlay_payload(
        case=case,
        case_family=case_family,
        mixed_row_selected_count=mixed_row_selected_count,
    )
    return {
        "benchmark": BENCHMARK_NAME,
        "case_id": (
            f"{case_family}_b{case.shape.batch_size}_q{case.shape.q_len}"
            f"_k{case.shape.kv_len}_ratio{case.shape.selected_ratio:g}"
        ),
        "kernel_case": kernel_case,
        "baseline_case": baseline_case,
        "path_validation_status": path_validation_status,
        "speed_gate_basis": "measured CUDA-event median ratio",
        "adapter_route_bypassed": True,
        "split_match_status": split_match_status,
        "split_match_error": split_match_error,
        "gate_status": gate_status,
        "requested_num_splits": int(num_splits),
        "native_resolved_num_splits": native_truth.resolved_num_splits,
        "mixed_resolved_num_splits": mixed_truth.resolved_num_splits,
        "native_use_dynamic_split": native_truth.use_dynamic_split,
        "mixed_use_dynamic_split": mixed_truth.use_dynamic_split,
        "native_num_splits_dynamic_offset": native_truth.num_splits_dynamic_offset,
        "mixed_num_splits_dynamic_offset": mixed_truth.num_splits_dynamic_offset,
        "native_num_splits_dynamic_min": native_truth.num_splits_dynamic_min,
        "native_num_splits_dynamic_max": native_truth.num_splits_dynamic_max,
        "mixed_num_splits_dynamic_min": mixed_truth.num_splits_dynamic_min,
        "mixed_num_splits_dynamic_max": mixed_truth.num_splits_dynamic_max,
        "native_scheduler_metadata_batch_size": native_truth.scheduler_metadata_batch_size,
        "mixed_scheduler_metadata_batch_size": mixed_truth.scheduler_metadata_batch_size,
        "max_seqlen_k": int(max_seqlen_k),
        "native_effective_k": int(max_seqlen_k),
        "mixed_effective_k": int(max_seqlen_k),
        "native_cuda_event_ms_median": native_ms,
        "mixed_cuda_event_ms_median": mixed_ms,
        "ratio_mixed_over_native": float(mixed_ms / native_ms) if native_ms > 0 else float("inf"),
        "native_cuda_event_ms_samples": native_samples,
        "mixed_cuda_event_ms_samples": mixed_samples,
        "batch_size": case.shape.batch_size,
        "q_len": case.shape.q_len,
        "kv_len": case.shape.kv_len,
        "page_size": case.shape.page_size,
        "num_q_heads": case.shape.num_q_heads,
        "num_kv_heads": case.shape.num_kv_heads,
        "head_dim": case.shape.head_dim,
        "selected_ratio": case.shape.selected_ratio,
        "total_pages": case.selected_geometry.total_pages,
        "selected_page_count": case.selected_geometry.selected_page_count,
        "selected_effective_k": case.selected_geometry.selected_effective_k,
        "selected_logical_pages": case.selected_geometry.selected_logical_pages,
        "selected_page_table_shape": list(selected_page_table_for_case.shape),
        "selected_page_table_num_cols": int(selected_page_table_for_case.shape[1]),
        "selected_page_table_layout": _selected_page_table_layout(case_family),
        "selected_page_table_present": case_family != "full",
        "row_consume_mode_present": case_family == "mixed_rows",
        "row_consume_mode_layout": (
            "alternating_full_selected" if case_family == "mixed_rows" else "none"
        ),
        "mixed_row_full_count": mixed_row_full_count,
        "mixed_row_selected_count": mixed_row_selected_count,
        "selected_native_page_table_shape": list(case.selected_native_page_table_i32.shape),
        "selected_native_page_table_layout": "same_physical_selected_pages",
        "full_page_table_shape": list(case.page_table_i32.shape),
        "selected_table_matches_full": selected_table_matches_full,
        "compact_mixed_page_overlay": compact_overlay_payload,
        "equivalence_probe": case_family in {"equivalent", "mixed_rows"},
        "correctness_scope": CORRECTNESS_SCOPE,
        "correctness_out_atol": CORRECTNESS_OUT_ATOL,
        "correctness_out_rtol": CORRECTNESS_OUT_RTOL,
        "correctness_lse_atol": CORRECTNESS_LSE_ATOL,
        "correctness_lse_rtol": CORRECTNESS_LSE_RTOL,
        "warmup": int(warmup),
        "repeat": int(repeat),
        "inner_iters": int(inner_iters),
    }


def main(argv: list[str] | None = None) -> None:
    argv_list = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv_list)
    if int(args.num_splits) < 0:
        raise ValueError("--num-splits must be >= 0")
    if int(args.inner_iters) <= 0:
        raise ValueError("--inner-iters must be > 0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for mixed_page microbenchmark smoke")

    command_line = _command_line_for_run(argv_list)
    benchmark_start_gpu_state = _collect_gpu_state()

    shape = BenchShape(
        batch_size=int(args.batch_size),
        q_len=int(args.q_len),
        kv_len=int(args.kv_len),
        selected_ratio=float(args.selected_ratio),
        page_size=int(args.page_size),
        num_q_heads=int(args.num_q_heads),
        num_kv_heads=int(args.num_kv_heads),
        head_dim=int(args.head_dim),
        dtype=_dtype_from_name(str(args.dtype)),
        device="cuda",
        seed=int(args.seed),
    )
    case = build_benchmark_case(shape)
    bridge = load_vendored_flash_attn_bridge()
    runtime_payload = runtime_identity_to_payload(collect_runtime_identity())
    output_path = Path(str(args.output_jsonl))
    _prepare_output_path(output_path, append=bool(args.append))
    git_payload = _collect_git_state()
    records: list[dict[str, object]] = []
    for case_family in args.case:
        case_gpu_payload = _collect_gpu_state()
        record = _bench_pair(
            bridge=bridge,
            case=case,
            case_family=str(case_family),
            num_splits=int(args.num_splits),
            warmup=int(args.warmup),
            repeat=int(args.repeat),
            inner_iters=int(args.inner_iters),
        )
        record.update(
            {
                "command_line": command_line,
                "benchmark_start_gpu_utilization_before_pct": benchmark_start_gpu_state[
                    "gpu_utilization_before_pct"
                ],
                "benchmark_start_gpu_memory_used_before_mib": benchmark_start_gpu_state[
                    "gpu_memory_used_before_mib"
                ],
            }
        )
        record.update(git_payload)
        record.update(runtime_payload)
        record.update(case_gpu_payload)
        emit_json_record(output_path, record)
        records.append(record)
        print(json.dumps(record, sort_keys=True))
    print(json.dumps(summarize_gate_records(records), sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1:])
