from __future__ import annotations

import argparse
from collections import Counter
import fnmatch
import hashlib
import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from types import ModuleType
from typing import Callable

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_FA3_ROOT = REPO_ROOT / "third_party_upstreams" / "vllm-project-flash-attention"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(UPSTREAM_FA3_ROOT) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_FA3_ROOT))

from benchmarks.bench_mixed_page_baseline import BenchShape, build_benchmark_case  # noqa: E402
from benchmarks.bench_mixed_page_resolver_graph_gate import (  # noqa: E402
    DEFAULT_PAGE_BLOCK_SIZE,
    _assert_close_and_diff,
    _extension_path,
    _selected_seqused,
    _selected_table,
)
from patches.fa3_native.install import load_vendored_flash_attn_bridge  # noqa: E402
from patches.fa3_native.mixed_page_graph_descriptor import PageResolverKind  # noqa: E402
from patches.fa3_native.row_consume_modes import (  # noqa: E402
    ROW_CONSUME_MODE_FULL_I32,
    ROW_CONSUME_MODE_SELECTED_I32,
)

KERNEL_ONLY_CASE_NAMES = (
    "kind0_native_non_tma_page64",
    "kind1_selected_batch_row_non_tma",
    "kind1_selected_batch_head_non_tma",
    "kind1_row_consume_visible_k_non_tma",
    "kind0_native_full4096_non_tma",
    "kind4_rowptr_batch_row_non_tma",
    "kind4_rowptr_batch_head_non_tma",
    "kind4_rowptr_batch_head_samepages_non_tma",
    "kind4_rowptr_irregular_non_tma",
    "kind4_rowptr_irregular_stride1_non_tma",
    "kind4_direct_table_batch_row_non_tma",
    "kind4_direct_table_batch_head_non_tma",
    "kind4_affine_const_non_tma",
    "kind4_affine_tensor_batch_row_non_tma",
    "kind4_affine_tensor_batch_head_non_tma",
    "kind4_affine_tensor_batch_head_samepages_non_tma",
    "kind4_direct_physical_non_tma",
    "kind4_two_segment_non_tma",
    "kind4_visible_k_negative_non_tma",
    "kind4_page_mutation_negative_non_tma",
    "kind0_native_tma_page128",
    "kind1_selected_batch_row_tma_page128",
    "kind1_selected_batch_head_tma_page128",
    "kind4_rowptr_batch_row_tma_page128",
    "kind4_rowptr_batch_head_tma_page128",
    "kind4_direct_table_batch_row_tma_page128",
    "kind4_direct_table_batch_head_tma_page128",
    "kind4_tma_invalid_page_shape_rejected",
    "kind4_tma_leftpad_rejected",
    "kind4_tma_local_rejected",
    "kind4_tma_missing_page_table_rejected",
    "kind4_affine_const_tma_page128",
    "kind4_affine_tensor_batch_row_tma_page128",
    "kind4_affine_tensor_batch_head_tma_page128",
    "kind4_direct_physical_tma_page128",
    "kind4_two_segment_tma_page128",
    "kind4_two_segment_invalid_descriptor_rejected",
    "kind4_affine_tma_unaligned_mapping_rejected",
)
TMA_C1_POSITIVE_CASE_NAMES = (
    "kind0_native_tma_page128",
    "kind1_selected_batch_row_tma_page128",
    "kind1_selected_batch_head_tma_page128",
    "kind4_rowptr_batch_row_tma_page128",
    "kind4_rowptr_batch_head_tma_page128",
    "kind4_direct_table_batch_row_tma_page128",
    "kind4_direct_table_batch_head_tma_page128",
)
TMA_C1_PREDICATE_REJECT_CASE_NAMES = (
    "kind4_tma_invalid_page_shape_rejected",
    "kind4_tma_leftpad_rejected",
    "kind4_tma_local_rejected",
    "kind4_tma_missing_page_table_rejected",
)
TMA_C1_CASE_NAMES = (
    "kind0_native_tma_page128",
    "kind1_selected_batch_row_tma_page128",
    "kind1_selected_batch_head_tma_page128",
    "kind4_rowptr_batch_row_tma_page128",
    "kind4_rowptr_batch_head_tma_page128",
    "kind4_direct_table_batch_row_tma_page128",
    "kind4_direct_table_batch_head_tma_page128",
    "kind4_tma_invalid_page_shape_rejected",
    "kind4_tma_leftpad_rejected",
    "kind4_tma_local_rejected",
    "kind4_tma_missing_page_table_rejected",
)
TMA_C2_POSITIVE_CASE_NAMES = (
    "kind4_affine_const_tma_page128",
    "kind4_affine_tensor_batch_row_tma_page128",
    "kind4_affine_tensor_batch_head_tma_page128",
    "kind4_direct_physical_tma_page128",
    "kind4_two_segment_tma_page128",
)
TMA_C2_PREDICATE_REJECT_CASE_NAMES = (
    "kind4_two_segment_invalid_descriptor_rejected",
    "kind4_affine_tma_unaligned_mapping_rejected",
)
TMA_C2_CASE_NAMES = (
    "kind4_affine_const_tma_page128",
    "kind4_affine_tensor_batch_row_tma_page128",
    "kind4_affine_tensor_batch_head_tma_page128",
    "kind4_direct_physical_tma_page128",
    "kind4_two_segment_tma_page128",
    "kind4_two_segment_invalid_descriptor_rejected",
    "kind4_affine_tma_unaligned_mapping_rejected",
)
TMA_POSITIVE_CASE_NAMES = (
    "kind0_native_tma_page128",
    "kind1_selected_batch_row_tma_page128",
    "kind1_selected_batch_head_tma_page128",
    "kind4_rowptr_batch_row_tma_page128",
    "kind4_rowptr_batch_head_tma_page128",
    "kind4_direct_table_batch_row_tma_page128",
    "kind4_direct_table_batch_head_tma_page128",
    "kind4_affine_const_tma_page128",
    "kind4_affine_tensor_batch_row_tma_page128",
    "kind4_affine_tensor_batch_head_tma_page128",
    "kind4_direct_physical_tma_page128",
    "kind4_two_segment_tma_page128",
)
TMA_PREDICATE_REJECT_CASE_NAMES = (
    "kind4_tma_invalid_page_shape_rejected",
    "kind4_tma_leftpad_rejected",
    "kind4_tma_local_rejected",
    "kind4_tma_missing_page_table_rejected",
    "kind4_two_segment_invalid_descriptor_rejected",
    "kind4_affine_tma_unaligned_mapping_rejected",
)
TMA_CASE_NAMES = (
    "kind0_native_tma_page128",
    "kind1_selected_batch_row_tma_page128",
    "kind1_selected_batch_head_tma_page128",
    "kind4_rowptr_batch_row_tma_page128",
    "kind4_rowptr_batch_head_tma_page128",
    "kind4_direct_table_batch_row_tma_page128",
    "kind4_direct_table_batch_head_tma_page128",
    "kind4_tma_invalid_page_shape_rejected",
    "kind4_tma_leftpad_rejected",
    "kind4_tma_local_rejected",
    "kind4_tma_missing_page_table_rejected",
    "kind4_affine_const_tma_page128",
    "kind4_affine_tensor_batch_row_tma_page128",
    "kind4_affine_tensor_batch_head_tma_page128",
    "kind4_direct_physical_tma_page128",
    "kind4_two_segment_tma_page128",
    "kind4_two_segment_invalid_descriptor_rejected",
    "kind4_affine_tma_unaligned_mapping_rejected",
)
PAGE_RESOLVER_SUBKIND_ROWPTR = 0
PAGE_RESOLVER_SUBKIND_DIRECT_TABLE = 1
PAGE_RESOLVER_SUBKIND_AFFINE_CONST = 2
PAGE_RESOLVER_SUBKIND_AFFINE_TENSOR = 3
PAGE_RESOLVER_SUBKIND_AFFINE_CONST_DIRECT = 4
DEFAULT_NON_TMA_TILE_N = 64
DEFAULT_TMA_TILE_N = 128
KERNEL_ONLY_NON_TMA_PAGE_SIZE = 64
KERNEL_ONLY_TMA_PAGE_SIZE = 128
KERNEL_ONLY_TMA_Q_LEN = 64


@dataclass(frozen=True)
class ResolverKernelOnlyRecord:
    benchmark: str
    case_name: str
    page_resolver_kind: int
    resolver_kind: int
    resolver_subkind: str
    tma_or_non_tma: str
    page_size: int
    tile_n: int
    effective_visible_k: int
    effective_visible_k_by_row: list[int]
    selected_or_resolved_carrier_shape: list[int]
    expected_physical_pages: list[list[int]]
    actual_physical_pages: list[list[int]]
    layout_class: str
    row_consume_mode: list[int]
    capture_enabled: bool
    output_allclose: bool
    capture_allclose_when_enabled: bool | None
    route_counter: str
    extension_build_id_or_so_path: str
    source_status: str
    gpu_name: str
    num_q_heads: int
    num_kv_heads: int
    kv_batch_idx_present: bool
    graph_cache_hit_when_applicable: bool | None
    graph_recapture_count_when_applicable: int | None
    negative_control_differs: bool
    sm_arch: str
    tma_mode: str
    batch: int
    heads: int
    page_block_size: int
    q_tokens: int
    requested_kv_len: int
    kv_tokens: int
    selected_pages: int
    warmup: int
    iters: int
    inner_iters: int
    timing_mode: str
    kernel_us: float
    baseline_kernel_us: float
    baseline_name: str
    baseline_overhead_pct: float
    native_resolver_kernel_us: float
    vs_native_resolver_overhead_pct: float
    raw_native_kernel_us: float
    vs_raw_native_overhead_pct: float
    speed_raw_pair_status: str
    speed_raw_pair_error: str
    speed_raw_baseline_name: str
    speed_raw_shadow_name: str
    speed_raw_pinned_num_splits: int
    raw_native_auto_kernel_us: float
    pinned_case_kernel_us: float
    launch_is_mixed_page: bool
    auto_requested_num_splits: int
    auto_resolved_num_splits: int
    auto_use_dynamic_split: bool
    auto_num_splits_dynamic_min: int
    auto_num_splits_dynamic_max: int
    scheduler_metadata_batch_size: int
    raw_native_requested_num_splits: int
    raw_native_resolved_num_splits: int
    raw_native_use_dynamic_split: bool
    raw_native_num_splits_dynamic_min: int
    raw_native_num_splits_dynamic_max: int
    raw_native_scheduler_metadata_batch_size: int
    raw_native_is_mixed_page: bool
    raw_native_is_pagedkv_tma: bool
    raw_native_tile_n: int
    split_reference_case: str
    split_match_status: str
    compute_match_status: str
    fairness_passed: bool
    correctness_passed: bool
    reference_max_abs_diff: float
    oracle_split_status: str
    oracle_reference_num_splits: int
    gate_passed: bool
    hard_fail: bool
    pass_fail: str
    extension_path: str
    extension_sha256: str
    git_sha: str
    git_dirty: bool
    cuda_device_name: str
    gpu_util_start_pct: int
    gpu_memory_used_start_mb: int
    gpu_util_end_pct: int
    gpu_memory_used_end_mb: int
    route_proof: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Kernel-only CUDA-event microbenchmark for original mixed_page "
            "SelectedTable, native resolver, and ResolvedRowPtr production paths."
        )
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output")
    parser.add_argument(
        "--extension-path",
        default="",
        help=(
            "Absolute path to the exact _vllm_fa3_C candidate under test. "
            "It is loaded before the vendored Python bridge and is also used "
            "for artifact path/SHA reporting, so a clean build can be gated "
            "without overwriting the production extension."
        ),
    )
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", "--repeat", dest="iters", type=int, default=1000)
    parser.add_argument("--inner-iters", type=int, default=20)
    parser.add_argument(
        "--force-num-splits",
        type=int,
        default=-1,
        help=(
            "Pin num_splits uniformly across every launch in the harness (cases,"
            " oracles/references, raw dense baseline) for same-split-regime tier"
            " curves. -1 keeps per-case defaults (auto); only 0 or 1 are legal"
            " otherwise (kind4 rejects explicit >1)."
        ),
    )
    parser.add_argument(
        "--timing-mode",
        choices=("launch_event", "cuda_graph"),
        default="cuda_graph",
        help=(
            "launch_event records CUDA events around the Python runner; cuda_graph "
            "captures inner-iters runner calls and measures graph replay to isolate "
            "recurring device work from host wrapper overhead. [2026-07-11 J1] "
            "Default is cuda_graph: production decode is graph replay, and the "
            "0.4%% kernel-tax speed gate is only meaningful there -- in "
            "launch_event mode small shapes measure the mixed-entry HOST tax "
            "(~+10%% at 30us kernels, kind0 shows it too), so the speed gate is "
            "skipped in that mode (reported, not gated)."
        ),
    )
    parser.add_argument("--max-overhead-pct", type=float, default=0.4)
    parser.add_argument("--hard-fail-overhead-pct", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--batch-sizes", default="")
    parser.add_argument("--num-q-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--kv-len", type=int, default=8192)
    parser.add_argument("--kv-lens", default="")
    parser.add_argument("--selected-pages", type=int, default=16)
    parser.add_argument(
        "--case-filter",
        default="",
        help=(
            "Comma-separated case names, globs, or substrings to run. "
            "Native baseline cases are added automatically for filtered runs."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fail-on-gate", action="store_true")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    if args.warmup < 0:
        parser.error("--warmup must be >= 0")
    if args.iters <= 0:
        parser.error("--iters must be > 0")
    if args.inner_iters <= 0:
        parser.error("--inner-iters must be > 0")
    if args.max_overhead_pct < 0.0:
        parser.error("--max-overhead-pct must be >= 0")
    if args.hard_fail_overhead_pct < 0.0:
        parser.error("--hard-fail-overhead-pct must be >= 0")
    if args.batch_size < 2:
        parser.error("--batch-size must be >= 2 for selected/native row-consume proof rows")
    if args.num_q_heads <= 0 or args.num_kv_heads <= 0:
        parser.error("--num-q-heads and --num-kv-heads must be > 0")
    if args.num_q_heads % args.num_kv_heads != 0:
        parser.error("--num-q-heads must be divisible by --num-kv-heads")
    if args.kv_len <= 0 or args.kv_len % KERNEL_ONLY_TMA_PAGE_SIZE != 0:
        parser.error("--kv-len must be a positive multiple of 128")
    if (
        args.selected_pages <= 1
        or args.selected_pages >= args.kv_len // KERNEL_ONLY_NON_TMA_PAGE_SIZE
        or args.selected_pages >= args.kv_len // KERNEL_ONLY_TMA_PAGE_SIZE
    ):
        parser.error("--selected-pages must be greater than 1 and strictly less than native pages")
    return args


def _case_filter_names(case_filter: str) -> tuple[str, ...]:
    if not case_filter:
        return KERNEL_ONLY_CASE_NAMES
    selected: set[str] = set()
    for raw_part in case_filter.split(","):
        pattern = raw_part.strip()
        if not pattern:
            continue
        matches = [
            name
            for name in KERNEL_ONLY_CASE_NAMES
            if name == pattern or fnmatch.fnmatchcase(name, pattern) or pattern in name
        ]
        if not matches:
            raise ValueError(f"--case-filter matched no kernel-only cases: {pattern}")
        selected.update(matches)
    if not selected:
        raise ValueError("--case-filter must select at least one case")
    selected.add("kind0_native_non_tma_page64")
    if any(name in TMA_CASE_NAMES for name in selected):
        selected.add("kind0_native_tma_page128")
    return tuple(name for name in KERNEL_ONLY_CASE_NAMES if name in selected)


def _json_sanitize(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_sanitize(item) for item in value]
    return value


def _load_extension_override(extension_path: str) -> str:
    """Load and identity-pin an isolated FA3 candidate.

    The vendored bridge imports a Python extension module by package name,
    while clean build verification must not install over that package's live
    binary.  Load the candidate's TORCH_LIBRARY registrations directly and
    provide a module marker for the bridge's import probe.  Refuse an already
    loaded binary with a different identity instead of silently benchmarking
    whichever extension won import order.
    """
    path = Path(extension_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"FA3 extension candidate does not exist: {path}")
    if "_vllm_fa3_C" not in path.name or path.suffix != ".so":
        raise ValueError(f"not an _vllm_fa3_C shared object: {path}")

    module_names = (
        "vllm_flash_attn._vllm_fa3_C",
        "vllm.vllm_flash_attn._vllm_fa3_C",
    )
    for module_name in module_names:
        loaded = sys.modules.get(module_name)
        loaded_path = getattr(loaded, "__file__", "") if loaded is not None else ""
        if loaded_path and Path(str(loaded_path)).resolve() != path:
            raise RuntimeError(
                "refusing FA3 extension identity conflict: "
                f"{module_name} already points to {loaded_path}, requested {path}"
            )

    torch.ops.load_library(str(path))
    marker = ModuleType(module_names[0])
    marker.__file__ = str(path)
    marker.__package__ = "vllm_flash_attn"
    sys.modules.setdefault(module_names[0], marker)
    return str(path)


def record_to_jsonable(record: ResolverKernelOnlyRecord) -> dict[str, object]:
    return _json_sanitize(asdict(record))


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


def _git_dirty() -> bool:
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return True
    return bool(result.stdout.strip())


def _file_sha256(path: str) -> str:
    if not path:
        return ""
    file_path = Path(path)
    if not file_path.exists():
        return ""
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gpu_state() -> tuple[int, int]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
                "-i",
                str(torch.cuda.current_device()),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return (-1, -1)
    first = result.stdout.strip().splitlines()[0]
    util, memory = (part.strip() for part in first.split(",", 1))
    return (int(util), int(memory))


def _cuda_device_name() -> str:
    if not torch.cuda.is_available():
        return ""
    return str(torch.cuda.get_device_name(torch.cuda.current_device()))


def _sm_arch() -> str:
    if not torch.cuda.is_available():
        return ""
    major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
    return f"sm{int(major)}{int(minor)}"


def _make_shape(args: argparse.Namespace) -> BenchShape:
    return BenchShape(
        batch_size=int(args.batch_size),
        q_len=1,
        kv_len=int(args.kv_len),
        selected_ratio=float(args.selected_pages) / float(args.kv_len // KERNEL_ONLY_NON_TMA_PAGE_SIZE),
        page_size=KERNEL_ONLY_NON_TMA_PAGE_SIZE,
        num_q_heads=int(args.num_q_heads),
        num_kv_heads=int(args.num_kv_heads),
        head_dim=128,
        dtype=torch.bfloat16,
        device="cuda",
        seed=int(args.seed),
    )


def _make_tma_shape(args: argparse.Namespace) -> BenchShape:
    return BenchShape(
        batch_size=int(args.batch_size),
        q_len=KERNEL_ONLY_TMA_Q_LEN,
        kv_len=int(args.kv_len),
        selected_ratio=float(args.selected_pages) / float(args.kv_len // KERNEL_ONLY_TMA_PAGE_SIZE),
        page_size=KERNEL_ONLY_TMA_PAGE_SIZE,
        num_q_heads=int(args.num_q_heads),
        num_kv_heads=int(args.num_kv_heads),
        head_dim=128,
        dtype=torch.bfloat16,
        device="cuda",
        seed=int(args.seed) + 17,
    )


_LAST_RAW_SAMPLES: dict[str, list[float]] = {}


def _measure_kernel_us(
    runners: list[tuple[str, Callable[[], object]]],
    *,
    warmup: int,
    repeat: int,
    inner_iters: int,
    timing_mode: str = "launch_event",
) -> dict[str, float]:
    if not runners:
        raise ValueError("at least one runner is required")
    if timing_mode not in {"launch_event", "cuda_graph"}:
        raise ValueError(f"unsupported timing_mode: {timing_mode}")

    for index in range(int(warmup)):
        offset = index % len(runners)
        for _, runner in runners[offset:] + runners[:offset]:
            for _ in range(int(inner_iters)):
                runner()
    torch.cuda.synchronize()

    if timing_mode == "cuda_graph":
        graphs: list[tuple[str, torch.cuda.CUDAGraph]] = []
        for name, runner in runners:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(int(inner_iters)):
                    runner()
            graphs.append((name, graph))
        torch.cuda.synchronize()

        samples: dict[str, list[float]] = {name: [] for name, _ in runners}
        for index in range(int(repeat)):
            offset = index % len(graphs)
            for name, graph in graphs[offset:] + graphs[:offset]:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                graph.replay()
                end.record()
                torch.cuda.synchronize()
                elapsed_us = float(start.elapsed_time(end)) * 1000.0 / float(inner_iters)
                samples[name].append(elapsed_us)
        _LAST_RAW_SAMPLES.clear()
        _LAST_RAW_SAMPLES.update(samples)
        return {name: float(median(values)) for name, values in samples.items()}

    samples: dict[str, list[float]] = {name: [] for name, _ in runners}
    for index in range(int(repeat)):
        offset = index % len(runners)
        for name, runner in runners[offset:] + runners[:offset]:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(int(inner_iters)):
                runner()
            end.record()
            torch.cuda.synchronize()
            elapsed_us = float(start.elapsed_time(end)) * 1000.0 / float(inner_iters)
            samples[name].append(elapsed_us)
    _LAST_RAW_SAMPLES.clear()
    _LAST_RAW_SAMPLES.update(samples)
    return {name: float(median(values)) for name, values in samples.items()}


def _retain_native_aux(bridge, out_accum, softmax_lse_accum) -> None:
    retain = getattr(bridge.interface_module, "_retain_fa3_async_aux_tensors", None)
    if retain is not None:
        retain(out_accum, softmax_lse_accum)


# [2026-07-11 K1/J1] --force-num-splits: pins num_splits uniformly across every
# launch in this harness (mixed cases, oracles/references, raw dense baseline) so
# pin-1 vs auto tier curves compare same-split-regime forms fairly (the production
# patch layer pins 1 on saturated tiers, auto elsewhere). Only 0 (auto) and 1 are
# legal: kind4 rejects explicit >1 at the C++ guard.
_FORCE_NUM_SPLITS: int | None = None


def _force_num_splits_or(default: int) -> int:
    return int(default if _FORCE_NUM_SPLITS is None else _FORCE_NUM_SPLITS)


# [2026-07-11 J1] Launch-rejection texts that mark a RETIRED form: on the arch
# that retired it the fail-closed raise is the expected contract (see the
# matching treatment in mixed_page_resolver_logit_capture_check.py).
_RETIRED_FORM_MARKERS = (
    "retired from the SM8x TU",
    "retired at launch",
)


def _run_native_reference(
    bridge,
    case,
    *,
    page_table: torch.Tensor,
    seqused_k: torch.Tensor,
    max_seqlen_k: int,
    num_splits: int = 0,
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
        num_splits=_force_num_splits_or(int(num_splits)),
    )


def _run_native_direct_op(
    bridge,
    case,
    *,
    page_table: torch.Tensor,
    seqused_k: torch.Tensor,
    max_seqlen_k: int,
    scheduler_metadata: torch.Tensor | None = None,
    num_splits: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    # [2026-07-12 J1-SPEEDGATE-SAMESPLIT] scheduler_metadata / num_splits are
    # only supplied by the pinned raw-native timing legs (same-split-regime
    # comparison columns); every other call keeps the stock auto form.
    out, softmax_lse, out_accum, softmax_lse_accum = torch.ops._vllm_fa3_C.fwd(
        case.q,
        case.k_cache,
        case.v_cache,
        None,
        None,
        None,
        None,
        case.cu_seqlens_q_i32,
        None,
        None,
        None,
        seqused_k,
        case.shape.q_len,
        int(max_seqlen_k),
        page_table,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        case.softmax_scale,
        True,
        -1,
        -1,
        0.0,
        True,
        scheduler_metadata,
        _force_num_splits_or(0) if num_splits is None else int(num_splits),
        None,
        0,
        None,
        1,
        0,
        None,
    )
    _retain_native_aux(bridge, out_accum, softmax_lse_accum)
    return out, softmax_lse


def _run_mixed_page_direct_op(
    case,
    *,
    kv_batch_idx: torch.Tensor | None = None,
    selected_page_table_i32: torch.Tensor | None,
    selected_page_table_batch_stride: int = 0,
    selected_page_table_head_stride: int = 0,
    row_consume_mode_i32: torch.Tensor | None,
    selected_seqused_k_by_head_i32: torch.Tensor | None,
    seqused_k: torch.Tensor,
    max_seqlen_k: int,
    page_resolver_kind: int,
    page_resolver_subkind: int = PAGE_RESOLVER_SUBKIND_ROWPTR,
    resolved_page_table_row_ptr_u64: torch.Tensor | None = None,
    resolved_page_table_row_ptr_batch_stride: int = 0,
    resolved_page_table_row_ptr_head_stride: int = 0,
    resolved_page_table_i32: torch.Tensor | None = None,
    resolved_page_table_batch_stride: int = 0,
    resolved_page_table_head_stride: int = 0,
    resolved_page_table_affine_i32: torch.Tensor | None = None,
    resolved_page_table_affine_batch_stride: int = 0,
    resolved_page_table_affine_head_stride: int = 0,
    resolved_page_table_affine_base: int = 0,
    resolved_page_table_affine_stride: int = 1,
    resolved_page_table_affine_segment_pages: int = 0,
    resolved_page_table_affine_second_base: int = 0,
    resolved_page_table_affine_second_stride: int = 1,
    resolved_page_table_affine_direct: bool = False,
    resolved_page_table_affine_cols: int = 0,
    num_splits: int = 0,
    resolved_seqused_k_by_head_i32: torch.Tensor | None = None,
    resolved_seqused_k_batch_stride: int = 0,
    resolved_seqused_k_head_stride: int = 0,
    block_table_i32: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    out, softmax_lse, _, _ = torch.ops._vllm_fa3_C.fwd_mixed_page(
        case.q,
        case.k_cache,
        case.v_cache,
        None,
        case.cu_seqlens_q_i32,
        None,
        case.shape.q_len,
        int(max_seqlen_k),
        case.page_table_i32 if block_table_i32 is None else block_table_i32,
        seqused_k,
        None,
        case.softmax_scale,
        True,
        -1,
        -1,
        0.0,
        None,
        None,
        None,
        None,
        None,
        1,
        0,
        None,
        kv_batch_idx,
        int(page_resolver_kind),
        int(page_resolver_subkind),
        selected_page_table_i32,
        int(selected_page_table_batch_stride),
        int(selected_page_table_head_stride),
        resolved_page_table_row_ptr_u64,
        int(resolved_page_table_row_ptr_batch_stride),
        int(resolved_page_table_row_ptr_head_stride),
        resolved_page_table_i32,
        int(resolved_page_table_batch_stride),
        int(resolved_page_table_head_stride),
        resolved_page_table_affine_i32,
        int(resolved_page_table_affine_batch_stride),
        int(resolved_page_table_affine_head_stride),
        int(resolved_page_table_affine_base),
        int(resolved_page_table_affine_stride),
        int(resolved_page_table_affine_segment_pages),
        int(resolved_page_table_affine_second_base),
        int(resolved_page_table_affine_second_stride),
        bool(resolved_page_table_affine_direct),
        int(resolved_page_table_affine_cols),
        _force_num_splits_or(int(num_splits)),
        resolved_seqused_k_by_head_i32,
        int(resolved_seqused_k_batch_stride),
        int(resolved_seqused_k_head_stride),
        row_consume_mode_i32,
        selected_seqused_k_by_head_i32,
        None,
        None,
        None,
        None,
        False,
        0,
        None,
    )
    return out, softmax_lse


def _headwise_selected_table(case, selected_table: torch.Tensor) -> torch.Tensor:
    batch = int(case.shape.batch_size)
    num_kv_heads = int(case.shape.num_kv_heads)
    headwise_table = torch.empty(
        (batch * num_kv_heads, int(selected_table.shape[1])),
        device=selected_table.device,
        dtype=selected_table.dtype,
    )
    for row in range(batch):
        base = row * num_kv_heads
        for kv_head in range(num_kv_heads):
            headwise_table[base + kv_head].copy_(selected_table[row])
    return headwise_table.contiguous()


def _batch_head_distinct_table(case, selected_pages: int) -> torch.Tensor:
    batch = int(case.shape.batch_size)
    num_kv_heads = int(case.shape.num_kv_heads)
    native_pages = int(case.page_table_i32.shape[1])
    width = int(selected_pages)
    table = torch.empty(
        (batch * num_kv_heads, width),
        device=case.q.device,
        dtype=torch.int32,
    )
    max_offset = max(1, native_pages - width)
    logical = torch.arange(width, device=case.q.device, dtype=torch.int64)
    for row in range(batch):
        base = row * num_kv_heads
        native_row = case.page_table_i32[row]
        for kv_head in range(num_kv_heads):
            offset = kv_head % max_offset
            page_idx = (logical + offset).remainder(native_pages)
            table[base + kv_head].copy_(native_row.index_select(0, page_idx))
    return table.contiguous()


def _irregular_selected_table(case, selected_pages: int, stride: int = 3) -> torch.Tensor:
    # stride=1 is the iso-locality control twin: identical construction path,
    # buffer layout, and rowptr shape as the stride-3 case, differing ONLY in
    # the physical page set (contiguous prefix vs strided scatter). The shadow
    # pair (batch_row vs irregular) is NOT iso-locality by construction, so
    # attributing its delta needs this single-variable twin.
    native_pages = int(case.page_table_i32.shape[1])
    page_idx = torch.arange(
        int(selected_pages),
        device=case.q.device,
        dtype=torch.int64,
    ).mul(int(stride)).remainder(native_pages)
    return case.page_table_i32.index_select(1, page_idx).contiguous()


def _row_ptr_table(table: torch.Tensor) -> torch.Tensor:
    return torch.tensor(
        [int(table[row].data_ptr()) for row in range(int(table.shape[0]))],
        device=table.device,
        dtype=torch.int64,
    )


def _carrier_shape(table: torch.Tensor | None) -> list[int]:
    if table is None:
        return []
    return [int(dim) for dim in table.shape]


def _tensor_rows_to_int_lists(table: torch.Tensor) -> list[list[int]]:
    return [
        [int(value) for value in row]
        for row in table.detach().cpu().tolist()
    ]


def resolve_expected_pages(
    *,
    subkind: str,
    native_pages: list[int],
    selected_pages: list[int] | None = None,
    affine_row: list[int] | None = None,
    row_mode: int = ROW_CONSUME_MODE_SELECTED_I32,
) -> list[int]:
    if row_mode == ROW_CONSUME_MODE_FULL_I32:
        return list(native_pages)
    if subkind in {"selected", "rowptr", "direct_table"}:
        assert selected_pages is not None
        return list(selected_pages)
    assert affine_row is not None
    base, stride = int(affine_row[0]), int(affine_row[1])
    if len(affine_row) >= 5 and int(affine_row[2]) > 0:
        segment_pages = int(affine_row[2])
        second_base = int(affine_row[3])
        second_stride = int(affine_row[4])
        return [
            base + stride * page
            if page < segment_pages
            else second_base + second_stride * (page - segment_pages)
            for page in range(len(native_pages))
        ]
    return [base + stride * page for page in range(len(native_pages))]


def assert_layout_pages_distinguish_heads(record: ResolverKernelOnlyRecord) -> None:
    if record.layout_class == "batch-row":
        assert record.expected_physical_pages[0] == record.expected_physical_pages[1]
    if record.layout_class == "batch-head-row":
        assert record.expected_physical_pages[0] != record.expected_physical_pages[1]


def _negative_control_differs(
    *,
    actual_out: torch.Tensor,
    full_out: torch.Tensor,
    rows: list[int] | None = None,
) -> bool:
    if rows is None:
        return not bool(torch.allclose(actual_out, full_out, atol=0.0, rtol=0.0))
    for row in rows:
        row_slice = slice(int(row), int(row) + 1)
        if not torch.allclose(actual_out[row_slice], full_out[row_slice], atol=0.0, rtol=0.0):
            return True
    return False


def _assert_rows_close_and_diff(
    *,
    actual_out: torch.Tensor,
    expected_selected: torch.Tensor,
    expected_full: torch.Tensor,
    actual_lse: torch.Tensor,
    expected_selected_lse: torch.Tensor,
    expected_full_lse: torch.Tensor,
    selected_rows: set[int],
) -> tuple[bool, float]:
    max_diff = torch.zeros((), device=actual_out.device, dtype=torch.float32)
    out_ok = True
    lse_ok = True
    for row in range(int(actual_out.shape[0])):
        row_slice = slice(row, row + 1)
        expected_out = expected_selected if row in selected_rows else expected_full
        row_diff = (
            actual_out[row_slice].to(torch.float32)
            - expected_out[row_slice].to(torch.float32)
        ).abs().max()
        max_diff = torch.maximum(max_diff, row_diff)
        out_ok = bool(
            out_ok
            and torch.allclose(
                actual_out[row_slice],
                expected_out[row_slice],
                atol=0.0,
                rtol=0.0,
            )
        )

        expected_lse = expected_selected_lse if row in selected_rows else expected_full_lse
        if actual_lse.dim() != 2:
            lse_ok = False
            continue
        if actual_lse.shape[0] == actual_out.shape[1]:
            actual_lse_row = actual_lse[:, row_slice]
            expected_lse_row = expected_lse[:, row_slice]
        elif actual_lse.shape[0] == actual_out.shape[0]:
            actual_lse_row = actual_lse[row_slice]
            expected_lse_row = expected_lse[row_slice]
        else:
            lse_ok = False
            continue
        lse_ok = bool(
            lse_ok
            and torch.allclose(actual_lse_row, expected_lse_row, atol=1.0e-3, rtol=0.0)
        )
    return bool(out_ok and lse_ok), float(max_diff.item())


def _split_signature(state: dict[str, object]) -> tuple[object, ...]:
    return (
        int(state["requested_num_splits"]),
        int(state["resolved_num_splits"]),
        bool(state["use_dynamic_split"]),
        int(state["num_splits_dynamic_min"]),
        int(state["num_splits_dynamic_max"]),
        int(state["scheduler_metadata_batch_size"]),
    )


def _output_and_lse_bitwise_equal(
    *,
    actual_out: torch.Tensor,
    expected_out: torch.Tensor,
    actual_lse: torch.Tensor,
    expected_lse: torch.Tensor,
) -> bool:
    """Exact replay identity, including signed zero and NaN payload bits.

    This is intentionally stricter than the case-vs-reference correctness
    oracle.  Replaying the same launch with the exact same scheduler metadata
    must not change either output tensor by even one bit; otherwise the replay
    cannot certify a shared reduction envelope.
    """

    def tensor_bits_equal(actual: torch.Tensor, expected: torch.Tensor) -> bool:
        if actual.dtype != expected.dtype or actual.shape != expected.shape:
            return False
        actual_bytes = actual.contiguous().view(torch.uint8)
        expected_bytes = expected.contiguous().view(torch.uint8)
        return bool(torch.equal(actual_bytes, expected_bytes))

    return tensor_bits_equal(actual_out, expected_out) and tensor_bits_equal(
        actual_lse,
        expected_lse,
    )


def _split_match_status(
    *,
    reference: dict[str, object],
    candidate: dict[str, object],
) -> str:
    if _split_signature(reference) != _split_signature(candidate):
        return "split_mismatch"
    return "matched_auto_split"


# [2026-07-12 J1-SPEEDGATE-SAMESPLIT] Same-split, same-prepare-regime speed
# comparison.
#
# SPLIT-ROOT made the in-launch prepare kernel of mixed-resolver launches
# (page_resolver_kind != Native) diverge from the dense one in BOTH outputs
# and cost:
#   1. split solution: makespan argmin vs legacy formula.  On tiers where
#      they disagree (e.g. bs8: argmin 5 vs legacy 2) the RRP-vs-raw time
#      ratio compares different split regimes and is meaningless in either
#      direction (measured bs8: RRP "-23%" = split-count artifact).
#   2. prepare kernel cost: the argmin branch (G-reduction + wave-level
#      scan) costs ~+0.67us/launch over the legacy branch (A100 forensic
#      2026-07-12), which on the 25us default tier alone reads as a fake
#      +2.7% "kernel tax" even when both sides solve the SAME split count.
# The 0.4% gate's contract is the RRP FWD-KERNEL structural tax (same-split
# pure-kernel measurement: +0.23%, gate-green).  It is therefore executed on
# skip-prepare pairs: both comparison legs consume pre-built pinned scheduler
# metadata (mha_fwd/mha_fwd_mixed_page skip their in-launch prepare when
# metadata is supplied - exactly the production K6 replay form, where one
# prepare feeds 36 layers), pinned to the case's own auto split solution.
# Pair statuses:
#   matched_auto_split   - case and raw reference run the same prepare branch
#                          and resolved the same split count on their own
#                          (kind0 native vs raw dense); direct comparison;
#   matched_forced_split - --force-num-splits 1 pinned every launch to the
#                          non-split form (N1: SingleTile launches have no
#                          prepare at all -> both sides prepare-free);
#                          direct comparison;
#   matched_pinned_split - mixed case: the gate ratio is taken from a shadow
#                          pair (case runner + injected pinned metadata) vs
#                          (raw dense + the same pinned metadata), both
#                          prepare-free at the case's auto split count via
#                          the S0-a instrument (_pinned_scheduler_metadata);
#                          the case's own reported timing leg stays stock;
#   split_mismatch_skipped - no same-split/same-regime reference could be
#                          built (heterogeneous per-row splits, pin readback
#                          failure, launch_event mode); the stock ratio is
#                          still reported but the case is excluded from the
#                          speed gate (honest skip: not a fake red, never a
#                          fake green);
#   untimed              - predicate-reject / runtime-fail cases that never
#                          reach the timing loop.
_SPEED_PAIR_MATCHED_STATUSES = (
    "matched_auto_split",
    "matched_forced_split",
    "matched_pinned_split",
)
_SPEED_PAIR_SKIPPED = "split_mismatch_skipped"
_SPEED_PAIR_UNTIMED = "untimed"
_SPEED_PAIR_ERROR = "shadow_pair_error"
# S0-a saturation margin: inflates num_sm inside the (untimed) metadata prep
# only, so the legacy heuristic saturates and the explicit num_splits cap
# pins the dynamic split exactly (dyn == min(saturated, cap) == cap).
_PIN_SM_MARGIN_SATURATE = -10000


def _effective_num_splits(state: dict[str, object]) -> int | None:
    """Split count the fwd kernel actually executed with, or None.

    Dynamic-split launches take their per-row split from the scheduler
    metadata (the probe reads it back from the live buffer after the launch);
    only a row-uniform value is a comparable tier (heterogeneous rows have no
    single split regime -> None).  Static launches execute
    resolved_num_splits.
    """
    if bool(state.get("use_dynamic_split", False)):
        dyn_min = int(state.get("num_splits_dynamic_min", -1))
        dyn_max = int(state.get("num_splits_dynamic_max", -1))
        if dyn_min >= 1 and dyn_min == dyn_max:
            return dyn_min
        return None
    resolved = int(state.get("resolved_num_splits", 0))
    return resolved if resolved >= 1 else None


def _speed_pair_split_status(
    *,
    case_state: dict[str, object],
    reference_state: dict[str, object],
    forced_split_mode: bool,
) -> str:
    """Pair status of a case-vs-reference timing comparison BEFORE any
    pinned-leg rescue (the caller upgrades to matched_pinned_split when it
    builds a same-split raw reference)."""
    case_effective = _effective_num_splits(case_state)
    reference_effective = _effective_num_splits(reference_state)
    if case_effective is not None and case_effective == reference_effective:
        return "matched_forced_split" if forced_split_mode else "matched_auto_split"
    return _SPEED_PAIR_SKIPPED


def _pinned_scheduler_metadata(
    bridge,
    case,
    *,
    seqused_k: torch.Tensor,
    max_seqlen_k: int,
    num_splits: int,
) -> torch.Tensor | None:
    """Build scheduler metadata whose dynamic-split slice is provably pinned
    to num_splits, without distorting the timed launch.

    S0-a instrument (SPLIT-ROOT campaign, scratchpad/s0a_split_pin_runner.py):
    get_scheduler_metadata with a hugely negative sm_margin saturates the
    legacy split heuristic inside the untimed metadata prep only; the
    explicit num_splits is the cap, so the prepared dyn slice equals the cap
    exactly.  The timed fwd launch keeps sm_margin=0 (kernel grid
    undistorted) and consumes the buffer through its scheduler_metadata
    argument.  Returns None when the pin cannot be proven by host readback
    (callers must fall back to split_mismatch_skipped, never gate on it).
    """
    if int(num_splits) < 2:
        return None
    metadata = bridge.interface_module.get_scheduler_metadata(
        int(seqused_k.numel()),
        int(case.shape.q_len),
        int(max_seqlen_k),
        int(case.shape.num_q_heads),
        int(case.shape.num_kv_heads),
        int(case.shape.head_dim),
        seqused_k,
        qkv_dtype=case.q.dtype,
        cu_seqlens_q=case.cu_seqlens_q_i32,
        page_size=int(case.shape.page_size),
        causal=True,
        num_splits=int(num_splits),
        sm_margin=_PIN_SM_MARGIN_SATURATE,
    )
    torch.cuda.synchronize()
    batch = int(seqused_k.numel())
    batch_rounded = (batch + 3) // 4 * 4
    if int(metadata.numel()) < batch_rounded + batch:
        return None
    dyn_slice = metadata[batch_rounded : batch_rounded + batch].cpu().tolist()
    if not all(int(value) == int(num_splits) for value in dyn_slice):
        return None
    return metadata


def _pinned_raw_native_runner(
    bridge,
    case,
    *,
    page_table: torch.Tensor,
    seqused_k: torch.Tensor,
    max_seqlen_k: int,
    pinned_metadata: torch.Tensor,
) -> Callable[[], tuple[torch.Tensor, torch.Tensor]]:
    """Raw dense reference leg of the shadow pair: fwd only (the injected
    metadata makes mha_fwd skip its in-launch prepare), pinned to the case's
    split count by the metadata's dyn slice.  num_splits stays auto so the
    accum-buffer sizing follows the same static get_num_splits solution as
    every other leg (the pinned dyn is <= static by construction).  Compared
    against _injected_metadata_case_runner legs, which are prepare-free the
    same way, so the ratio is a pure fwd-kernel comparison at one split
    count."""

    def runner() -> tuple[torch.Tensor, torch.Tensor]:
        return _run_native_direct_op(
            bridge,
            case,
            page_table=page_table,
            seqused_k=seqused_k,
            max_seqlen_k=int(max_seqlen_k),
            scheduler_metadata=pinned_metadata,
        )

    return runner


# Fail closed if the direct-op ABI changes.  A source contract derives the
# expected value from _run_mixed_page_direct_op's call site so this fingerprint
# cannot silently drift again.
_MIXED_PAGE_DIRECT_OP_ARGC = 59
_MIXED_PAGE_SCHEDULER_METADATA_ARG_INDEX = 16


def _injected_metadata_case_runner(
    case_runner: Callable[[], tuple[torch.Tensor, torch.Tensor]],
    *,
    pinned_metadata: torch.Tensor,
) -> Callable[[], tuple[torch.Tensor, torch.Tensor]]:
    """Shadow leg: the UNMODIFIED case runner, with the pinned scheduler
    metadata injected at the fwd_mixed_page launch site via a scoped op shim
    (the 59-arg direct-op ABI and every case closure stay byte-identical).
    With metadata supplied the mixed launch skips its in-launch prepare
    (flash_api.cpp skip_scheduler_metadata_computation), so the timed region
    is the fwd kernel only - the production K6 replay form.  num_splits is
    left untouched (auto): the launch then sizes its accum buffers by the
    same static get_num_splits solution as the stock case leg, the injected
    dyn slice (== the case's own auto solution, <= static by construction)
    drives the split execution, and the C++ guard that rejects explicit
    num_splits>1 on non-RRP resolver kinds (e.g. kind1 SelectedTable) is
    never in play."""

    def runner() -> tuple[torch.Tensor, torch.Tensor]:
        namespace = torch.ops._vllm_fa3_C
        original_op = namespace.fwd_mixed_page

        def shim(*args, **kwargs):
            if kwargs or len(args) != _MIXED_PAGE_DIRECT_OP_ARGC:
                raise RuntimeError(
                    "fwd_mixed_page shim expects the "
                    f"{_MIXED_PAGE_DIRECT_OP_ARGC}-positional-arg direct-op form"
                )
            if args[_MIXED_PAGE_SCHEDULER_METADATA_ARG_INDEX] is not None:
                raise RuntimeError(
                    "fwd_mixed_page shim refuses to overwrite caller scheduler_metadata"
                )
            injected = list(args)
            injected[_MIXED_PAGE_SCHEDULER_METADATA_ARG_INDEX] = pinned_metadata
            return original_op(*injected)

        setattr(namespace, "fwd_mixed_page", shim)
        try:
            return case_runner()
        finally:
            setattr(namespace, "fwd_mixed_page", original_op)

    return runner


def run_kernel_only(args: argparse.Namespace) -> list[ResolverKernelOnlyRecord]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for resolver kernel-only benchmark")

    bridge = load_vendored_flash_attn_bridge()
    gpu_start_util, gpu_start_memory = _gpu_state()
    active_case_names = _case_filter_names(str(getattr(args, "case_filter", "")))
    case = build_benchmark_case(_make_shape(args))
    tma_case = build_benchmark_case(_make_tma_shape(args))
    batch = int(case.shape.batch_size)
    num_kv_heads = int(case.shape.num_kv_heads)
    selected_pages = int(args.selected_pages)
    native_pages = int(case.shape.kv_len) // int(case.shape.page_size)
    tma_native_pages = int(tma_case.shape.kv_len) // int(tma_case.shape.page_size)
    if selected_pages <= 1 or selected_pages >= native_pages or selected_pages >= tma_native_pages:
        raise ValueError(
            "selected_pages must be greater than 1 and strictly less than native pages "
            "for selected/full negative-control and TMA C1 proof"
        )

    selected_k = selected_pages * int(case.shape.page_size)
    selected_table = _selected_table(case, selected_pages)
    head_table = _batch_head_distinct_table(case, selected_pages)
    head_table_samepages = _headwise_selected_table(case, selected_table)  # bench_rowptr_head_samepages
    irregular_table = _irregular_selected_table(case, selected_pages)
    irregular_table_stride1 = _irregular_selected_table(case, selected_pages, stride=1)
    mutated_table = selected_table.add(1).contiguous()
    two_segment = max(1, selected_pages // 2)
    two_segment_second_base = max(two_segment, native_pages - (selected_pages - two_segment))
    two_segment_logical = list(range(two_segment)) + list(
        range(two_segment_second_base, two_segment_second_base + selected_pages - two_segment)
    )
    two_segment_table = case.page_table_i32.index_select(
        1,
        torch.tensor(two_segment_logical, device=case.q.device, dtype=torch.int64),
    ).contiguous()
    selected_seqused = _selected_seqused(case, selected_k)
    visible_selected = torch.full(
        (batch * num_kv_heads,),
        int(selected_k),
        device=case.q.device,
        dtype=torch.int32,
    )
    visible_too_short = torch.full_like(
        visible_selected,
        max(int(case.shape.page_size), int(selected_k) - int(case.shape.page_size)),
    )
    row_consume_modes = torch.full(
        (batch,),
        ROW_CONSUME_MODE_FULL_I32,
        device=case.q.device,
        dtype=torch.int32,
    )
    row_consume_modes[0] = ROW_CONSUME_MODE_SELECTED_I32
    row_consume_values = [int(value) for value in row_consume_modes.detach().cpu().tolist()]
    row_consume_seqused = case.full_seqused_k_i32.clone()
    row_consume_seqused[0] = int(selected_k)
    row_consume_visible = visible_selected.clone()
    row_consume_visible[num_kv_heads:] = 0
    # Uniform visible K is already represented by seqused_k. Keep the
    # per-head resolved visible carrier for cases that actually test a
    # non-uniform/negative visible-K contract; otherwise it forces the SM90
    # RRP kernel down an avoidable per-head seqlen path.
    resolved_visible_row = None

    selected_out, selected_lse = _run_native_reference(
        bridge,
        case,
        page_table=selected_table,
        seqused_k=selected_seqused,
        max_seqlen_k=selected_k,
    )
    irregular_out, irregular_lse = _run_native_reference(
        bridge,
        case,
        page_table=irregular_table,
        seqused_k=selected_seqused,
        max_seqlen_k=selected_k,
    )
    two_segment_out, two_segment_lse = _run_native_reference(
        bridge,
        case,
        page_table=two_segment_table,
        seqused_k=selected_seqused,
        max_seqlen_k=selected_k,
    )
    full_out, full_lse = _run_native_reference(
        bridge,
        case,
        page_table=case.page_table_i32,
        seqused_k=case.full_seqused_k_i32,
        max_seqlen_k=case.shape.kv_len,
    )
    row_consume_native_out, row_consume_native_lse = _run_mixed_page_direct_op(
        case,
        selected_page_table_i32=None,
        row_consume_mode_i32=row_consume_modes,
        selected_seqused_k_by_head_i32=row_consume_visible,
        seqused_k=row_consume_seqused,
        max_seqlen_k=case.shape.kv_len,
        page_resolver_kind=int(PageResolverKind.NATIVE),
        block_table_i32=case.page_table_i32,
    )

    def materialized_reference_for_head_table(
        ref_case,
        page_table: torch.Tensor,
        seqused_k: torch.Tensor,
        max_seqlen_k: int,
        *,
        num_splits: int | None = 0,
        scheduler_metadata: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ref_batch = int(ref_case.shape.batch_size)
        ref_num_kv_heads = int(ref_case.shape.num_kv_heads)
        width = int(page_table.shape[1])
        k_cache = torch.empty(
            (ref_batch * width, int(ref_case.shape.page_size), ref_num_kv_heads, int(ref_case.shape.head_dim)),
            device=ref_case.q.device,
            dtype=ref_case.k_cache.dtype,
        )
        v_cache = torch.empty_like(ref_case.v_cache[: ref_batch * width])
        for row in range(ref_batch):
            for logical_page in range(width):
                output_page = row * width + logical_page
                for kv_head in range(ref_num_kv_heads):
                    source_page = int(page_table[row * ref_num_kv_heads + kv_head, logical_page].item())
                    k_cache[output_page, :, kv_head, :].copy_(ref_case.k_cache[source_page, :, kv_head, :])
                    v_cache[output_page, :, kv_head, :].copy_(ref_case.v_cache[source_page, :, kv_head, :])
        materialized_case = ref_case.__class__(
            **{
                **ref_case.__dict__,
                "k_cache": k_cache,
                "v_cache": v_cache,
            }
        )
        ref_page_table = torch.arange(
            ref_batch * width,
            device=ref_case.q.device,
            dtype=torch.int32,
        ).reshape(ref_batch, width)
        if scheduler_metadata is not None:
            return _run_native_direct_op(
                bridge,
                materialized_case,
                page_table=ref_page_table,
                seqused_k=seqused_k,
                max_seqlen_k=max_seqlen_k,
                scheduler_metadata=scheduler_metadata,
                num_splits=num_splits,
            )
        return _run_native_reference(
            bridge,
            materialized_case,
            page_table=ref_page_table,
            seqused_k=seqused_k,
            max_seqlen_k=max_seqlen_k,
            num_splits=0 if num_splits is None else int(num_splits),
        )

    head_out, head_lse = materialized_reference_for_head_table(
        case,
        head_table,
        selected_seqused,
        selected_k,
    )
    affine_head_table = torch.stack(
        [
            case.page_table_i32[row].index_select(
                0,
                torch.arange(
                    selected_pages,
                    device=case.q.device,
                    dtype=torch.int64,
                ).add(kv_head % max(1, native_pages - selected_pages)).remainder(native_pages),
            )
            for row in range(batch)
            for kv_head in range(num_kv_heads)
        ],
        dim=0,
    ).contiguous()
    affine_head_out, affine_head_lse = materialized_reference_for_head_table(
        case,
        affine_head_table,
        selected_seqused,
        selected_k,
    )

    tma_selected_k = selected_pages * int(tma_case.shape.page_size)
    tma_selected_table = _selected_table(tma_case, selected_pages)
    tma_head_table = _batch_head_distinct_table(tma_case, selected_pages)
    tma_selected_seqused = _selected_seqused(tma_case, tma_selected_k)
    tma_visible_selected = torch.full(
        (batch * num_kv_heads,),
        int(tma_selected_k),
        device=tma_case.q.device,
        dtype=torch.int32,
    )
    uniform_tma_resolved_visible = None
    tma_two_segment = max(1, selected_pages // 2)
    tma_two_segment_second_base = max(tma_two_segment, tma_native_pages - (selected_pages - tma_two_segment))
    tma_two_segment_logical = list(range(tma_two_segment)) + list(
        range(tma_two_segment_second_base, tma_two_segment_second_base + selected_pages - tma_two_segment)
    )
    tma_two_segment_table = tma_case.page_table_i32.index_select(
        1,
        torch.tensor(tma_two_segment_logical, device=tma_case.q.device, dtype=torch.int64),
    ).contiguous()
    tma_affine_head_table = torch.stack(
        [
            tma_case.page_table_i32[row].index_select(
                0,
                torch.arange(
                    selected_pages,
                    device=tma_case.q.device,
                    dtype=torch.int64,
                ).add(kv_head % max(1, tma_native_pages - selected_pages)).remainder(tma_native_pages),
            )
            for row in range(batch)
            for kv_head in range(num_kv_heads)
        ],
        dim=0,
    ).contiguous()
    tma_native_out, tma_native_lse = _run_native_reference(
        bridge,
        tma_case,
        page_table=tma_selected_table,
        seqused_k=tma_selected_seqused,
        max_seqlen_k=tma_selected_k,
        num_splits=0,
    )
    tma_selected_out, tma_selected_lse = _run_native_reference(
        bridge,
        tma_case,
        page_table=tma_selected_table,
        seqused_k=tma_selected_seqused,
        max_seqlen_k=tma_selected_k,
        num_splits=0,
    )
    tma_head_out, tma_head_lse = materialized_reference_for_head_table(
        tma_case,
        tma_head_table,
        tma_selected_seqused,
        tma_selected_k,
        num_splits=0,
    )
    tma_affine_head_out, tma_affine_head_lse = materialized_reference_for_head_table(
        tma_case,
        tma_affine_head_table,
        tma_selected_seqused,
        tma_selected_k,
        num_splits=0,
    )
    tma_two_segment_out, tma_two_segment_lse = _run_native_reference(
        bridge,
        tma_case,
        page_table=tma_two_segment_table,
        seqused_k=tma_selected_seqused,
        max_seqlen_k=tma_selected_k,
        num_splits=0,
    )
    tma_full_out, tma_full_lse = _run_native_reference(
        bridge,
        tma_case,
        page_table=tma_case.page_table_i32,
        seqused_k=tma_case.full_seqused_k_i32,
        max_seqlen_k=tma_case.shape.kv_len,
        num_splits=0,
    )
    del tma_full_lse

    rowptr_selected = _row_ptr_table(selected_table)
    rowptr_head = _row_ptr_table(head_table)
    rowptr_head_samepages = _row_ptr_table(head_table_samepages)  # bench_rowptr_head_samepages
    rowptr_irregular = _row_ptr_table(irregular_table)
    rowptr_irregular_stride1 = _row_ptr_table(irregular_table_stride1)
    tma_rowptr_selected = _row_ptr_table(tma_selected_table)
    tma_rowptr_head = _row_ptr_table(tma_head_table)
    direct_physical_affine = torch.tensor(
        [[row * native_pages, 1] for row in range(batch)],
        device=case.q.device,
        dtype=torch.int32,
    ).contiguous()
    affine_row = torch.tensor(
        [[0, 1] for _ in range(batch)],
        device=case.q.device,
        dtype=torch.int32,
    ).contiguous()
    affine_head = torch.tensor(
        [[kv_head % max(1, native_pages - selected_pages), 1] for _ in range(batch) for kv_head in range(num_kv_heads)],
        device=case.q.device,
        dtype=torch.int32,
    ).contiguous()
    # discriminator: per-head affine rows all [0,1] -> head_stride still 1 (ctor head-strided
    # load happens) but every head indexes the SAME selected page set, isolating ctor-load vs L2.
    affine_head_samepages = torch.tensor(
        [[0, 1] for _ in range(batch) for _ in range(num_kv_heads)],
        device=case.q.device,
        dtype=torch.int32,
    ).contiguous()
    affine_two_segment = torch.tensor(
        [[0, 1, two_segment, two_segment_second_base, 1] for _ in range(batch)],
        device=case.q.device,
        dtype=torch.int32,
    ).contiguous()
    tma_affine_row = torch.tensor(
        [[0, 1] for _ in range(batch)],
        device=tma_case.q.device,
        dtype=torch.int32,
    ).contiguous()
    tma_affine_head = torch.tensor(
        [[kv_head % max(1, tma_native_pages - selected_pages), 1] for _ in range(batch) for kv_head in range(num_kv_heads)],
        device=tma_case.q.device,
        dtype=torch.int32,
    ).contiguous()
    tma_affine_two_segment = torch.tensor(
        [[0, 1, tma_two_segment, tma_two_segment_second_base, 1] for _ in range(batch)],
        device=tma_case.q.device,
        dtype=torch.int32,
    ).contiguous()

    def head_pages(table: torch.Tensor, layout_class: str) -> list[list[int]]:
        rows = _tensor_rows_to_int_lists(table)
        if layout_class == "batch-head-row":
            return rows
        return [list(rows[row]) for row in range(batch) for _ in range(num_kv_heads)]

    def affine_pages(
        affine_rows: torch.Tensor,
        layout_class: str,
        *,
        direct: bool = False,
        ref_case=case,
    ) -> list[list[int]]:
        rows = _tensor_rows_to_int_lists(affine_rows)
        native_rows = _tensor_rows_to_int_lists(ref_case.page_table_i32)
        ref_batch = int(ref_case.shape.batch_size)
        ref_num_kv_heads = int(ref_case.shape.num_kv_heads)
        physical_rows: list[list[int]] = []
        if layout_class == "batch-head-row":
            iterable = enumerate(rows)
        else:
            iterable = (
                (row * ref_num_kv_heads + head, rows[row])
                for row in range(ref_batch)
                for head in range(ref_num_kv_heads)
            )
        for flat_row, affine_descriptor in iterable:
            batch_row = flat_row // ref_num_kv_heads
            logical = resolve_expected_pages(
                subkind="affine_tensor",
                native_pages=list(range(selected_pages)),
                affine_row=affine_descriptor,
            )
            if direct:
                physical_rows.append(logical)
            else:
                native_row = native_rows[batch_row]
                physical_rows.append([native_row[index] for index in logical])
        return physical_rows

    def call_mixed(**kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        return _run_mixed_page_direct_op(case, **kwargs)

    def call_tma_mixed(**kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        return _run_mixed_page_direct_op(tma_case, **kwargs)

    def call_tma_c2_mixed(*, resolver_subkind: str, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        assert resolver_subkind in {"affine_const", "affine_tensor", "direct_physical", "two_segment"}
        assert kwargs.get("resolved_page_table_row_ptr_u64") is None
        return _run_mixed_page_direct_op(tma_case, **kwargs)

    raw_native_non_tma_name = "__raw_native_fa3_non_tma_page64"
    raw_native_tma_name = "__raw_native_fa3_tma_page128"
    raw_native_runners: dict[str, Callable[[], tuple[torch.Tensor, torch.Tensor]]] = {
        raw_native_non_tma_name: lambda: _run_native_direct_op(
            bridge,
            case,
            page_table=selected_table,
            seqused_k=selected_seqused,
            max_seqlen_k=selected_k,
        ),
        raw_native_tma_name: lambda: _run_native_direct_op(
            bridge,
            tma_case,
            page_table=tma_selected_table,
            seqused_k=tma_selected_seqused,
            max_seqlen_k=tma_selected_k,
        ),
    }
    raw_native_launch_states: dict[str, dict[str, object]] = {}

    tma_predicate_reject_reasons = {
        "kind4_tma_invalid_page_shape_rejected": "source_predicate_rejects_page_size_not_aligned_to_kBlockN",
        "kind4_tma_leftpad_rejected": "source_predicate_rejects_leftpad_k",
        "kind4_tma_local_rejected": "source_predicate_rejects_local_attention",
        "kind4_tma_missing_page_table_rejected": "source_predicate_rejects_missing_page_table",
        "kind4_two_segment_invalid_descriptor_rejected": "source_predicate_rejects_affine_cols_must_be_2_or_5",
        "kind4_affine_tma_unaligned_mapping_rejected": "source_predicate_rejects_unaligned_affine_mapping",
    }

    def source_only_tma_predicate_reject(case_name: str) -> tuple[torch.Tensor, torch.Tensor]:
        raise RuntimeError(f"{case_name} is source-only and must be skipped by the runtime loop")

    runners: dict[str, Callable[[], tuple[torch.Tensor, torch.Tensor]]] = {
        "kind0_native_non_tma_page64": lambda: call_mixed(
            selected_page_table_i32=None,
            row_consume_mode_i32=None,
            selected_seqused_k_by_head_i32=None,
            seqused_k=selected_seqused,
            max_seqlen_k=selected_k,
            page_resolver_kind=int(PageResolverKind.NATIVE),
            block_table_i32=selected_table,
        ),
        "kind1_selected_batch_row_non_tma": lambda: call_mixed(
            selected_page_table_i32=selected_table,
            selected_page_table_batch_stride=1,
            selected_page_table_head_stride=0,
            row_consume_mode_i32=None,
            selected_seqused_k_by_head_i32=None,
            seqused_k=selected_seqused,
            max_seqlen_k=selected_k,
            page_resolver_kind=int(PageResolverKind.SELECTED_TABLE),
        ),
        "kind1_selected_batch_head_non_tma": lambda: call_mixed(
            selected_page_table_i32=head_table,
            selected_page_table_batch_stride=num_kv_heads,
            selected_page_table_head_stride=1,
            row_consume_mode_i32=None,
            selected_seqused_k_by_head_i32=None,
            seqused_k=selected_seqused,
            max_seqlen_k=selected_k,
            page_resolver_kind=int(PageResolverKind.SELECTED_TABLE),
            # fa3_sm90_dtable_u64_base: u64 base array -> launch converts to DirectTable u64-load.
            resolved_page_table_row_ptr_u64=rowptr_head,
            resolved_page_table_row_ptr_batch_stride=num_kv_heads,
            resolved_page_table_row_ptr_head_stride=1,
        ),
        "kind1_row_consume_visible_k_non_tma": lambda: call_mixed(
            selected_page_table_i32=selected_table,
            selected_page_table_batch_stride=1,
            selected_page_table_head_stride=0,
            row_consume_mode_i32=row_consume_modes,
            selected_seqused_k_by_head_i32=row_consume_visible,
            seqused_k=row_consume_seqused,
            max_seqlen_k=case.shape.kv_len,
            page_resolver_kind=int(PageResolverKind.SELECTED_TABLE),
        ),
        "kind0_native_full4096_non_tma": lambda: call_mixed(
            selected_page_table_i32=None,
            row_consume_mode_i32=None,
            selected_seqused_k_by_head_i32=None,
            seqused_k=case.full_seqused_k_i32,
            max_seqlen_k=case.shape.kv_len,
            page_resolver_kind=int(PageResolverKind.NATIVE),
            block_table_i32=case.page_table_i32,
        ),
        "kind4_rowptr_batch_row_non_tma": lambda: call_mixed(
            selected_page_table_i32=None,
            row_consume_mode_i32=None,
            selected_seqused_k_by_head_i32=None,
            seqused_k=selected_seqused,
            max_seqlen_k=selected_k,
            page_resolver_kind=int(PageResolverKind.RESOLVED_ROW_PTR),
            resolved_page_table_row_ptr_u64=rowptr_selected,
            resolved_page_table_row_ptr_batch_stride=1,
            resolved_page_table_row_ptr_head_stride=0,
            resolved_seqused_k_by_head_i32=resolved_visible_row,
            resolved_seqused_k_batch_stride=0,
            resolved_seqused_k_head_stride=0,
            block_table_i32=selected_table,
        ),
        "kind4_rowptr_batch_head_non_tma": lambda: call_mixed(
            selected_page_table_i32=None,
            row_consume_mode_i32=None,
            selected_seqused_k_by_head_i32=None,
            seqused_k=selected_seqused,
            max_seqlen_k=selected_k,
            page_resolver_kind=int(PageResolverKind.RESOLVED_ROW_PTR),
            resolved_page_table_row_ptr_u64=rowptr_head,
            resolved_page_table_row_ptr_batch_stride=num_kv_heads,
            resolved_page_table_row_ptr_head_stride=1,
            resolved_seqused_k_by_head_i32=resolved_visible_row,
            resolved_seqused_k_batch_stride=0,
            resolved_seqused_k_head_stride=0,
        ),
        "kind4_rowptr_batch_head_samepages_non_tma": lambda: call_mixed(
            selected_page_table_i32=None,
            row_consume_mode_i32=None,
            selected_seqused_k_by_head_i32=None,
            seqused_k=selected_seqused,
            max_seqlen_k=selected_k,
            page_resolver_kind=int(PageResolverKind.RESOLVED_ROW_PTR),
            resolved_page_table_row_ptr_u64=rowptr_head_samepages,
            resolved_page_table_row_ptr_batch_stride=num_kv_heads,
            resolved_page_table_row_ptr_head_stride=1,
            resolved_seqused_k_by_head_i32=resolved_visible_row,
            resolved_seqused_k_batch_stride=0,
            resolved_seqused_k_head_stride=0,
        ),
        "kind4_rowptr_irregular_non_tma": lambda: call_mixed(
            selected_page_table_i32=None,
            row_consume_mode_i32=None,
            selected_seqused_k_by_head_i32=None,
            seqused_k=selected_seqused,
            max_seqlen_k=selected_k,
            page_resolver_kind=int(PageResolverKind.RESOLVED_ROW_PTR),
            resolved_page_table_row_ptr_u64=rowptr_irregular,
            resolved_page_table_row_ptr_batch_stride=1,
            resolved_page_table_row_ptr_head_stride=0,
            resolved_seqused_k_by_head_i32=resolved_visible_row,
            resolved_seqused_k_batch_stride=0,
            resolved_seqused_k_head_stride=0,
            block_table_i32=irregular_table,
        ),
        # iso-locality control twin for the irregular case: same construction
        # path/rowptr shape, page set = contiguous prefix (stride=1). Its delta
        # vs baseline bounds the resolver's structural tax; (irregular - this)
        # isolates the pure page-locality component of the shadow-pair gap.
        "kind4_rowptr_irregular_stride1_non_tma": lambda: call_mixed(
            selected_page_table_i32=None,
            row_consume_mode_i32=None,
            selected_seqused_k_by_head_i32=None,
            seqused_k=selected_seqused,
            max_seqlen_k=selected_k,
            page_resolver_kind=int(PageResolverKind.RESOLVED_ROW_PTR),
            resolved_page_table_row_ptr_u64=rowptr_irregular_stride1,
            resolved_page_table_row_ptr_batch_stride=1,
            resolved_page_table_row_ptr_head_stride=0,
            resolved_seqused_k_by_head_i32=resolved_visible_row,
            resolved_seqused_k_batch_stride=0,
            resolved_seqused_k_head_stride=0,
            block_table_i32=irregular_table_stride1,
        ),
        "kind4_direct_table_batch_row_non_tma": lambda: call_mixed(
            selected_page_table_i32=None,
            row_consume_mode_i32=None,
            selected_seqused_k_by_head_i32=None,
            seqused_k=selected_seqused,
            max_seqlen_k=selected_k,
            page_resolver_kind=int(PageResolverKind.RESOLVED_ROW_PTR),
            page_resolver_subkind=PAGE_RESOLVER_SUBKIND_DIRECT_TABLE,
            resolved_page_table_i32=selected_table,
            resolved_page_table_batch_stride=1,
            resolved_page_table_head_stride=0,
            resolved_seqused_k_by_head_i32=resolved_visible_row,
            resolved_seqused_k_batch_stride=0,
            resolved_seqused_k_head_stride=0,
            block_table_i32=selected_table,
        ),
        "kind4_direct_table_batch_head_non_tma": lambda: call_mixed(
            selected_page_table_i32=None,
            row_consume_mode_i32=None,
            selected_seqused_k_by_head_i32=None,
            seqused_k=selected_seqused,
            max_seqlen_k=selected_k,
            page_resolver_kind=int(PageResolverKind.RESOLVED_ROW_PTR),
            page_resolver_subkind=PAGE_RESOLVER_SUBKIND_DIRECT_TABLE,
            resolved_page_table_i32=head_table,
            resolved_page_table_batch_stride=num_kv_heads,
            resolved_page_table_head_stride=1,
            # fa3_sm90_dtable_u64_base: RowPtr-style u64 base array -> kernel LOADs the base.
            resolved_page_table_row_ptr_u64=rowptr_head,
            resolved_page_table_row_ptr_batch_stride=num_kv_heads,
            resolved_page_table_row_ptr_head_stride=1,
            resolved_seqused_k_by_head_i32=resolved_visible_row,
            resolved_seqused_k_batch_stride=0,
            resolved_seqused_k_head_stride=0,
        ),
        "kind4_affine_const_non_tma": lambda: call_mixed(
            selected_page_table_i32=None,
            row_consume_mode_i32=None,
            selected_seqused_k_by_head_i32=None,
            seqused_k=selected_seqused,
            max_seqlen_k=selected_k,
            page_resolver_kind=int(PageResolverKind.RESOLVED_ROW_PTR),
            page_resolver_subkind=PAGE_RESOLVER_SUBKIND_AFFINE_CONST,
            resolved_page_table_affine_base=0,
            resolved_page_table_affine_stride=1,
            resolved_seqused_k_by_head_i32=resolved_visible_row,
            resolved_seqused_k_batch_stride=0,
            resolved_seqused_k_head_stride=0,
        ),
        "kind4_affine_tensor_batch_row_non_tma": lambda: call_mixed(
            selected_page_table_i32=None,
            row_consume_mode_i32=None,
            selected_seqused_k_by_head_i32=None,
            seqused_k=selected_seqused,
            max_seqlen_k=selected_k,
            page_resolver_kind=int(PageResolverKind.RESOLVED_ROW_PTR),
            page_resolver_subkind=PAGE_RESOLVER_SUBKIND_AFFINE_TENSOR,
            resolved_page_table_affine_i32=affine_row,
            resolved_page_table_affine_batch_stride=1,
            resolved_page_table_affine_head_stride=0,
            resolved_page_table_affine_cols=2,
            resolved_seqused_k_by_head_i32=resolved_visible_row,
            resolved_seqused_k_batch_stride=0,
            resolved_seqused_k_head_stride=0,
        ),
        "kind4_affine_tensor_batch_head_non_tma": lambda: call_mixed(
            selected_page_table_i32=None,
            row_consume_mode_i32=None,
            selected_seqused_k_by_head_i32=None,
            seqused_k=selected_seqused,
            max_seqlen_k=selected_k,
            page_resolver_kind=int(PageResolverKind.RESOLVED_ROW_PTR),
            page_resolver_subkind=PAGE_RESOLVER_SUBKIND_AFFINE_TENSOR,
            resolved_page_table_affine_i32=affine_head,
            resolved_page_table_affine_batch_stride=num_kv_heads,
            resolved_page_table_affine_head_stride=1,
            resolved_page_table_affine_cols=2,
            resolved_seqused_k_by_head_i32=resolved_visible_row,
            resolved_seqused_k_batch_stride=0,
            resolved_seqused_k_head_stride=0,
        ),
        "kind4_affine_tensor_batch_head_samepages_non_tma": lambda: call_mixed(
            selected_page_table_i32=None,
            row_consume_mode_i32=None,
            selected_seqused_k_by_head_i32=None,
            seqused_k=selected_seqused,
            max_seqlen_k=selected_k,
            page_resolver_kind=int(PageResolverKind.RESOLVED_ROW_PTR),
            page_resolver_subkind=PAGE_RESOLVER_SUBKIND_AFFINE_TENSOR,
            resolved_page_table_affine_i32=affine_head_samepages,
            resolved_page_table_affine_batch_stride=num_kv_heads,
            resolved_page_table_affine_head_stride=1,
            resolved_page_table_affine_cols=2,
            resolved_seqused_k_by_head_i32=resolved_visible_row,
            resolved_seqused_k_batch_stride=0,
            resolved_seqused_k_head_stride=0,
        ),
        "kind4_direct_physical_non_tma": lambda: call_mixed(
            selected_page_table_i32=None,
            row_consume_mode_i32=None,
            selected_seqused_k_by_head_i32=None,
            seqused_k=selected_seqused,
            max_seqlen_k=selected_k,
            page_resolver_kind=int(PageResolverKind.RESOLVED_ROW_PTR),
            page_resolver_subkind=PAGE_RESOLVER_SUBKIND_AFFINE_CONST_DIRECT,
            resolved_page_table_affine_base=0,
            resolved_page_table_affine_stride=1,
            resolved_page_table_affine_batch_stride=native_pages,
            resolved_page_table_affine_head_stride=0,
            resolved_page_table_affine_direct=True,
            resolved_seqused_k_by_head_i32=resolved_visible_row,
            resolved_seqused_k_batch_stride=0,
            resolved_seqused_k_head_stride=0,
            block_table_i32=selected_table,
        ),
        "kind4_two_segment_non_tma": lambda: call_mixed(
            selected_page_table_i32=None,
            row_consume_mode_i32=None,
            selected_seqused_k_by_head_i32=None,
            seqused_k=selected_seqused,
            max_seqlen_k=selected_k,
            page_resolver_kind=int(PageResolverKind.RESOLVED_ROW_PTR),
            page_resolver_subkind=PAGE_RESOLVER_SUBKIND_AFFINE_CONST,
            resolved_page_table_affine_base=0,
            resolved_page_table_affine_stride=1,
            resolved_page_table_affine_segment_pages=two_segment,
            resolved_page_table_affine_second_base=two_segment_second_base,
            resolved_page_table_affine_second_stride=1,
            resolved_page_table_affine_batch_stride=0,
            resolved_page_table_affine_head_stride=0,
            resolved_seqused_k_by_head_i32=resolved_visible_row,
            resolved_seqused_k_batch_stride=0,
            resolved_seqused_k_head_stride=0,
        ),
        "kind4_visible_k_negative_non_tma": lambda: call_mixed(
            selected_page_table_i32=None,
            row_consume_mode_i32=None,
            selected_seqused_k_by_head_i32=None,
            seqused_k=selected_seqused,
            max_seqlen_k=selected_k,
            page_resolver_kind=int(PageResolverKind.RESOLVED_ROW_PTR),
            resolved_page_table_row_ptr_u64=rowptr_selected,
            resolved_page_table_row_ptr_batch_stride=1,
            resolved_page_table_row_ptr_head_stride=0,
            resolved_seqused_k_by_head_i32=visible_too_short,
            resolved_seqused_k_batch_stride=num_kv_heads,
            resolved_seqused_k_head_stride=1,
            block_table_i32=selected_table,
        ),
        "kind4_page_mutation_negative_non_tma": lambda: call_mixed(
            selected_page_table_i32=None,
            row_consume_mode_i32=None,
            selected_seqused_k_by_head_i32=None,
            seqused_k=selected_seqused,
            max_seqlen_k=selected_k,
            page_resolver_kind=int(PageResolverKind.RESOLVED_ROW_PTR),
            page_resolver_subkind=PAGE_RESOLVER_SUBKIND_DIRECT_TABLE,
            resolved_page_table_i32=mutated_table,
            resolved_page_table_batch_stride=1,
            resolved_page_table_head_stride=0,
            resolved_seqused_k_by_head_i32=resolved_visible_row,
            resolved_seqused_k_batch_stride=0,
            resolved_seqused_k_head_stride=0,
            block_table_i32=selected_table,
        ),
        "kind0_native_tma_page128": lambda: call_tma_mixed(
            selected_page_table_i32=None,
            row_consume_mode_i32=None,
            selected_seqused_k_by_head_i32=None,
            seqused_k=tma_selected_seqused,
            max_seqlen_k=tma_selected_k,
            page_resolver_kind=int(PageResolverKind.NATIVE),
            block_table_i32=tma_selected_table,
        ),
        "kind1_selected_batch_row_tma_page128": lambda: call_tma_mixed(
            selected_page_table_i32=tma_selected_table,
            selected_page_table_batch_stride=1,
            selected_page_table_head_stride=0,
            row_consume_mode_i32=None,
            selected_seqused_k_by_head_i32=None,
            seqused_k=tma_selected_seqused,
            max_seqlen_k=tma_selected_k,
            page_resolver_kind=int(PageResolverKind.SELECTED_TABLE),
            num_splits=0,
        ),
        "kind1_selected_batch_head_tma_page128": lambda: call_tma_mixed(
            selected_page_table_i32=tma_head_table,
            selected_page_table_batch_stride=num_kv_heads,
            selected_page_table_head_stride=1,
            row_consume_mode_i32=None,
            selected_seqused_k_by_head_i32=None,
            seqused_k=tma_selected_seqused,
            max_seqlen_k=tma_selected_k,
            page_resolver_kind=int(PageResolverKind.SELECTED_TABLE),
            num_splits=0,
        ),
        "kind4_rowptr_batch_row_tma_page128": lambda: call_tma_mixed(
            selected_page_table_i32=None,
            row_consume_mode_i32=None,
            selected_seqused_k_by_head_i32=None,
            seqused_k=tma_selected_seqused,
            max_seqlen_k=tma_selected_k,
            page_resolver_kind=int(PageResolverKind.RESOLVED_ROW_PTR),
            resolved_page_table_row_ptr_u64=tma_rowptr_selected,
            resolved_page_table_row_ptr_batch_stride=1,
            resolved_page_table_row_ptr_head_stride=0,
            resolved_seqused_k_by_head_i32=uniform_tma_resolved_visible,
            resolved_seqused_k_batch_stride=0,
            resolved_seqused_k_head_stride=0,
            block_table_i32=tma_selected_table,
            num_splits=0,
        ),
        "kind4_rowptr_batch_head_tma_page128": lambda: call_tma_mixed(
            selected_page_table_i32=None,
            row_consume_mode_i32=None,
            selected_seqused_k_by_head_i32=None,
            seqused_k=tma_selected_seqused,
            max_seqlen_k=tma_selected_k,
            page_resolver_kind=int(PageResolverKind.RESOLVED_ROW_PTR),
            resolved_page_table_row_ptr_u64=tma_rowptr_head,
            resolved_page_table_row_ptr_batch_stride=num_kv_heads,
            resolved_page_table_row_ptr_head_stride=1,
            resolved_seqused_k_by_head_i32=uniform_tma_resolved_visible,
            resolved_seqused_k_batch_stride=0,
            resolved_seqused_k_head_stride=0,
            num_splits=0,
        ),
        "kind4_direct_table_batch_row_tma_page128": lambda: call_tma_mixed(
            selected_page_table_i32=None,
            row_consume_mode_i32=None,
            selected_seqused_k_by_head_i32=None,
            seqused_k=tma_selected_seqused,
            max_seqlen_k=tma_selected_k,
            page_resolver_kind=int(PageResolverKind.RESOLVED_ROW_PTR),
            page_resolver_subkind=PAGE_RESOLVER_SUBKIND_DIRECT_TABLE,
            resolved_page_table_i32=tma_selected_table,
            resolved_page_table_batch_stride=1,
            resolved_page_table_head_stride=0,
            resolved_seqused_k_by_head_i32=uniform_tma_resolved_visible,
            resolved_seqused_k_batch_stride=0,
            resolved_seqused_k_head_stride=0,
            block_table_i32=tma_selected_table,
            num_splits=0,
        ),
        "kind4_direct_table_batch_head_tma_page128": lambda: call_tma_mixed(
            selected_page_table_i32=None,
            row_consume_mode_i32=None,
            selected_seqused_k_by_head_i32=None,
            seqused_k=tma_selected_seqused,
            max_seqlen_k=tma_selected_k,
            page_resolver_kind=int(PageResolverKind.RESOLVED_ROW_PTR),
            page_resolver_subkind=PAGE_RESOLVER_SUBKIND_DIRECT_TABLE,
            resolved_page_table_i32=tma_head_table,
            resolved_page_table_batch_stride=num_kv_heads,
            resolved_page_table_head_stride=1,
            resolved_seqused_k_by_head_i32=uniform_tma_resolved_visible,
            resolved_seqused_k_batch_stride=0,
            resolved_seqused_k_head_stride=0,
            num_splits=0,
        ),
        "kind4_tma_invalid_page_shape_rejected": lambda: source_only_tma_predicate_reject("kind4_tma_invalid_page_shape_rejected"),
        "kind4_tma_leftpad_rejected": lambda: source_only_tma_predicate_reject("kind4_tma_leftpad_rejected"),
        "kind4_tma_local_rejected": lambda: source_only_tma_predicate_reject("kind4_tma_local_rejected"),
        "kind4_tma_missing_page_table_rejected": lambda: source_only_tma_predicate_reject("kind4_tma_missing_page_table_rejected"),
        "kind4_affine_const_tma_page128": lambda: call_tma_c2_mixed(
            resolver_subkind="affine_const",
            selected_page_table_i32=None,
            row_consume_mode_i32=None,
            selected_seqused_k_by_head_i32=None,
            seqused_k=tma_selected_seqused,
            max_seqlen_k=tma_selected_k,
            page_resolver_kind=int(PageResolverKind.RESOLVED_ROW_PTR),
            page_resolver_subkind=PAGE_RESOLVER_SUBKIND_AFFINE_CONST,
            resolved_page_table_affine_base=0,
            resolved_page_table_affine_stride=1,
            resolved_seqused_k_by_head_i32=uniform_tma_resolved_visible,
            resolved_seqused_k_batch_stride=0,
            resolved_seqused_k_head_stride=0,
            num_splits=0,
        ),
        "kind4_affine_tensor_batch_row_tma_page128": lambda: call_tma_c2_mixed(
            resolver_subkind="affine_tensor",
            selected_page_table_i32=None,
            row_consume_mode_i32=None,
            selected_seqused_k_by_head_i32=None,
            seqused_k=tma_selected_seqused,
            max_seqlen_k=tma_selected_k,
            page_resolver_kind=int(PageResolverKind.RESOLVED_ROW_PTR),
            page_resolver_subkind=PAGE_RESOLVER_SUBKIND_AFFINE_TENSOR,
            resolved_page_table_affine_i32=tma_affine_row,
            resolved_page_table_affine_batch_stride=1,
            resolved_page_table_affine_head_stride=0,
            resolved_page_table_affine_cols=2,
            resolved_seqused_k_by_head_i32=uniform_tma_resolved_visible,
            resolved_seqused_k_batch_stride=0,
            resolved_seqused_k_head_stride=0,
            num_splits=0,
        ),
        "kind4_affine_tensor_batch_head_tma_page128": lambda: call_tma_c2_mixed(
            resolver_subkind="affine_tensor",
            selected_page_table_i32=None,
            row_consume_mode_i32=None,
            selected_seqused_k_by_head_i32=None,
            seqused_k=tma_selected_seqused,
            max_seqlen_k=tma_selected_k,
            page_resolver_kind=int(PageResolverKind.RESOLVED_ROW_PTR),
            page_resolver_subkind=PAGE_RESOLVER_SUBKIND_AFFINE_TENSOR,
            resolved_page_table_affine_i32=tma_affine_head,
            resolved_page_table_affine_batch_stride=num_kv_heads,
            resolved_page_table_affine_head_stride=1,
            resolved_page_table_affine_cols=2,
            resolved_seqused_k_by_head_i32=uniform_tma_resolved_visible,
            resolved_seqused_k_batch_stride=0,
            resolved_seqused_k_head_stride=0,
            num_splits=0,
        ),
        "kind4_direct_physical_tma_page128": lambda: call_tma_c2_mixed(
            resolver_subkind="direct_physical",
            selected_page_table_i32=None,
            row_consume_mode_i32=None,
            selected_seqused_k_by_head_i32=None,
            seqused_k=tma_selected_seqused,
            max_seqlen_k=tma_selected_k,
            page_resolver_kind=int(PageResolverKind.RESOLVED_ROW_PTR),
            page_resolver_subkind=PAGE_RESOLVER_SUBKIND_AFFINE_CONST_DIRECT,
            resolved_page_table_affine_base=0,
            resolved_page_table_affine_stride=1,
            resolved_page_table_affine_batch_stride=tma_native_pages,
            resolved_page_table_affine_direct=True,
            resolved_seqused_k_by_head_i32=uniform_tma_resolved_visible,
            resolved_seqused_k_batch_stride=0,
            resolved_seqused_k_head_stride=0,
            block_table_i32=tma_selected_table,
            num_splits=0,
        ),
        "kind4_two_segment_tma_page128": lambda: call_tma_c2_mixed(
            resolver_subkind="two_segment",
            selected_page_table_i32=None,
            row_consume_mode_i32=None,
            selected_seqused_k_by_head_i32=None,
            seqused_k=tma_selected_seqused,
            max_seqlen_k=tma_selected_k,
            page_resolver_kind=int(PageResolverKind.RESOLVED_ROW_PTR),
            page_resolver_subkind=PAGE_RESOLVER_SUBKIND_AFFINE_CONST,
            resolved_page_table_affine_base=0,
            resolved_page_table_affine_stride=1,
            resolved_page_table_affine_segment_pages=tma_two_segment,
            resolved_page_table_affine_second_base=tma_two_segment_second_base,
            resolved_page_table_affine_second_stride=1,
            resolved_page_table_affine_batch_stride=0,
            resolved_page_table_affine_head_stride=0,
            resolved_seqused_k_by_head_i32=uniform_tma_resolved_visible,
            resolved_seqused_k_batch_stride=0,
            resolved_seqused_k_head_stride=0,
            num_splits=0,
        ),
        "kind4_two_segment_invalid_descriptor_rejected": lambda: source_only_tma_predicate_reject("kind4_two_segment_invalid_descriptor_rejected"),
        "kind4_affine_tma_unaligned_mapping_rejected": lambda: source_only_tma_predicate_reject("kind4_affine_tma_unaligned_mapping_rejected"),
    }

    metadata = {
        "kind0_native_non_tma_page64": ("native", int(PageResolverKind.NATIVE), PAGE_RESOLVER_SUBKIND_ROWPTR, "batch-row", [], head_pages(selected_table, "batch-row"), head_pages(selected_table, "batch-row"), selected_k),
        "kind1_selected_batch_row_non_tma": ("selected", int(PageResolverKind.SELECTED_TABLE), PAGE_RESOLVER_SUBKIND_ROWPTR, "batch-row", _carrier_shape(selected_table), head_pages(selected_table, "batch-row"), head_pages(selected_table, "batch-row"), selected_k),
        "kind1_selected_batch_head_non_tma": ("selected", int(PageResolverKind.SELECTED_TABLE), PAGE_RESOLVER_SUBKIND_ROWPTR, "batch-head-row", _carrier_shape(head_table), head_pages(head_table, "batch-head-row"), head_pages(head_table, "batch-head-row"), selected_k),
        "kind1_row_consume_visible_k_non_tma": ("selected", int(PageResolverKind.SELECTED_TABLE), PAGE_RESOLVER_SUBKIND_ROWPTR, "batch-row", _carrier_shape(selected_table), [
            list(row)
            for batch_row in range(batch)
            for row in (
                [head_pages(selected_table, "batch-row")[batch_row * num_kv_heads]]
                if batch_row == 0
                else [head_pages(case.page_table_i32, "batch-row")[batch_row * num_kv_heads]]
            )
            for _ in range(num_kv_heads)
        ], [
            list(row)
            for batch_row in range(batch)
            for row in (
                [head_pages(selected_table, "batch-row")[batch_row * num_kv_heads]]
                if batch_row == 0
                else [head_pages(case.page_table_i32, "batch-row")[batch_row * num_kv_heads]]
            )
            for _ in range(num_kv_heads)
        ], case.shape.kv_len),
        "kind0_native_full4096_non_tma": ("native", int(PageResolverKind.NATIVE), PAGE_RESOLVER_SUBKIND_ROWPTR, "batch-row", [], head_pages(selected_table, "batch-row"), head_pages(selected_table, "batch-row"), selected_k),
        "kind4_rowptr_batch_row_non_tma": ("rowptr", int(PageResolverKind.RESOLVED_ROW_PTR), PAGE_RESOLVER_SUBKIND_ROWPTR, "batch-row", _carrier_shape(rowptr_selected), head_pages(selected_table, "batch-row"), head_pages(selected_table, "batch-row"), selected_k),
        "kind4_rowptr_batch_head_non_tma": ("rowptr", int(PageResolverKind.RESOLVED_ROW_PTR), PAGE_RESOLVER_SUBKIND_ROWPTR, "batch-head-row", _carrier_shape(rowptr_head), head_pages(head_table, "batch-head-row"), head_pages(head_table, "batch-head-row"), selected_k),
        "kind4_rowptr_batch_head_samepages_non_tma": ("rowptr", int(PageResolverKind.RESOLVED_ROW_PTR), PAGE_RESOLVER_SUBKIND_ROWPTR, "batch-head-row", _carrier_shape(rowptr_head_samepages), head_pages(head_table_samepages, "batch-head-row"), head_pages(head_table_samepages, "batch-head-row"), selected_k),
        "kind4_rowptr_irregular_non_tma": ("rowptr", int(PageResolverKind.RESOLVED_ROW_PTR), PAGE_RESOLVER_SUBKIND_ROWPTR, "batch-row", _carrier_shape(rowptr_irregular), head_pages(irregular_table, "batch-row"), head_pages(irregular_table, "batch-row"), selected_k),
        # stride=1 twin reads the contiguous-prefix page set (== selected_table),
        # so the default (selected_out, selected_lse) reference applies bitwise.
        "kind4_rowptr_irregular_stride1_non_tma": ("rowptr", int(PageResolverKind.RESOLVED_ROW_PTR), PAGE_RESOLVER_SUBKIND_ROWPTR, "batch-row", _carrier_shape(rowptr_irregular_stride1), head_pages(irregular_table_stride1, "batch-row"), head_pages(irregular_table_stride1, "batch-row"), selected_k),
        "kind4_direct_table_batch_row_non_tma": ("direct_table", int(PageResolverKind.RESOLVED_ROW_PTR), PAGE_RESOLVER_SUBKIND_DIRECT_TABLE, "batch-row", _carrier_shape(selected_table), head_pages(selected_table, "batch-row"), head_pages(selected_table, "batch-row"), selected_k),
        "kind4_direct_table_batch_head_non_tma": ("direct_table", int(PageResolverKind.RESOLVED_ROW_PTR), PAGE_RESOLVER_SUBKIND_DIRECT_TABLE, "batch-head-row", _carrier_shape(head_table), head_pages(head_table, "batch-head-row"), head_pages(head_table, "batch-head-row"), selected_k),
        "kind4_affine_const_non_tma": ("affine_const", int(PageResolverKind.RESOLVED_ROW_PTR), PAGE_RESOLVER_SUBKIND_AFFINE_CONST, "batch-row", [], head_pages(selected_table, "batch-row"), head_pages(selected_table, "batch-row"), selected_k),
        "kind4_affine_tensor_batch_row_non_tma": ("affine_tensor", int(PageResolverKind.RESOLVED_ROW_PTR), PAGE_RESOLVER_SUBKIND_AFFINE_TENSOR, "batch-row", _carrier_shape(affine_row), affine_pages(affine_row, "batch-row"), affine_pages(affine_row, "batch-row"), selected_k),
        "kind4_affine_tensor_batch_head_non_tma": ("affine_tensor", int(PageResolverKind.RESOLVED_ROW_PTR), PAGE_RESOLVER_SUBKIND_AFFINE_TENSOR, "batch-head-row", _carrier_shape(affine_head), affine_pages(affine_head, "batch-head-row"), affine_pages(affine_head, "batch-head-row"), selected_k),
        "kind4_affine_tensor_batch_head_samepages_non_tma": ("affine_tensor", int(PageResolverKind.RESOLVED_ROW_PTR), PAGE_RESOLVER_SUBKIND_AFFINE_TENSOR, "batch-head-row", _carrier_shape(affine_head_samepages), affine_pages(affine_head_samepages, "batch-head-row"), affine_pages(affine_head_samepages, "batch-head-row"), selected_k),
        "kind4_direct_physical_non_tma": ("affine_const_direct", int(PageResolverKind.RESOLVED_ROW_PTR), PAGE_RESOLVER_SUBKIND_AFFINE_CONST_DIRECT, "batch-row", [], head_pages(selected_table, "batch-row"), head_pages(selected_table, "batch-row"), selected_k),
        "kind4_two_segment_non_tma": ("affine_const_2seg", int(PageResolverKind.RESOLVED_ROW_PTR), PAGE_RESOLVER_SUBKIND_AFFINE_CONST, "batch-row", [], head_pages(two_segment_table, "batch-row"), head_pages(two_segment_table, "batch-row"), selected_k),
        "kind4_visible_k_negative_non_tma": ("rowptr", int(PageResolverKind.RESOLVED_ROW_PTR), PAGE_RESOLVER_SUBKIND_ROWPTR, "batch-row", _carrier_shape(rowptr_selected), head_pages(selected_table, "batch-row"), head_pages(selected_table, "batch-row"), selected_k),
        "kind4_page_mutation_negative_non_tma": ("direct_table", int(PageResolverKind.RESOLVED_ROW_PTR), PAGE_RESOLVER_SUBKIND_DIRECT_TABLE, "batch-row", _carrier_shape(mutated_table), head_pages(selected_table, "batch-row"), head_pages(mutated_table, "batch-row"), selected_k),
        "kind0_native_tma_page128": ("native", int(PageResolverKind.NATIVE), PAGE_RESOLVER_SUBKIND_ROWPTR, "batch-row", [], head_pages(tma_selected_table, "batch-row"), head_pages(tma_selected_table, "batch-row"), tma_selected_k),
        "kind1_selected_batch_row_tma_page128": ("selected", int(PageResolverKind.SELECTED_TABLE), PAGE_RESOLVER_SUBKIND_ROWPTR, "batch-row", _carrier_shape(tma_selected_table), head_pages(tma_selected_table, "batch-row"), head_pages(tma_selected_table, "batch-row"), tma_selected_k),
        "kind1_selected_batch_head_tma_page128": ("selected", int(PageResolverKind.SELECTED_TABLE), PAGE_RESOLVER_SUBKIND_ROWPTR, "batch-head-row", _carrier_shape(tma_head_table), head_pages(tma_head_table, "batch-head-row"), head_pages(tma_head_table, "batch-head-row"), tma_selected_k),
        "kind4_rowptr_batch_row_tma_page128": ("rowptr", int(PageResolverKind.RESOLVED_ROW_PTR), PAGE_RESOLVER_SUBKIND_ROWPTR, "batch-row", _carrier_shape(tma_rowptr_selected), head_pages(tma_selected_table, "batch-row"), head_pages(tma_selected_table, "batch-row"), tma_selected_k),
        "kind4_rowptr_batch_head_tma_page128": ("rowptr", int(PageResolverKind.RESOLVED_ROW_PTR), PAGE_RESOLVER_SUBKIND_ROWPTR, "batch-head-row", _carrier_shape(tma_rowptr_head), head_pages(tma_head_table, "batch-head-row"), head_pages(tma_head_table, "batch-head-row"), tma_selected_k),
        "kind4_direct_table_batch_row_tma_page128": ("direct_table", int(PageResolverKind.RESOLVED_ROW_PTR), PAGE_RESOLVER_SUBKIND_DIRECT_TABLE, "batch-row", _carrier_shape(tma_selected_table), head_pages(tma_selected_table, "batch-row"), head_pages(tma_selected_table, "batch-row"), tma_selected_k),
        "kind4_direct_table_batch_head_tma_page128": ("direct_table", int(PageResolverKind.RESOLVED_ROW_PTR), PAGE_RESOLVER_SUBKIND_DIRECT_TABLE, "batch-head-row", _carrier_shape(tma_head_table), head_pages(tma_head_table, "batch-head-row"), head_pages(tma_head_table, "batch-head-row"), tma_selected_k),
        "kind4_tma_invalid_page_shape_rejected": ("expected_reject", int(PageResolverKind.RESOLVED_ROW_PTR), PAGE_RESOLVER_SUBKIND_ROWPTR, "batch-row", _carrier_shape(tma_rowptr_selected), head_pages(tma_selected_table, "batch-row"), head_pages(tma_selected_table, "batch-row"), tma_selected_k),
        "kind4_tma_leftpad_rejected": ("expected_reject", int(PageResolverKind.RESOLVED_ROW_PTR), PAGE_RESOLVER_SUBKIND_ROWPTR, "batch-row", _carrier_shape(tma_rowptr_selected), head_pages(tma_selected_table, "batch-row"), head_pages(tma_selected_table, "batch-row"), tma_selected_k),
        "kind4_tma_local_rejected": ("expected_reject", int(PageResolverKind.RESOLVED_ROW_PTR), PAGE_RESOLVER_SUBKIND_ROWPTR, "batch-row", _carrier_shape(tma_rowptr_selected), head_pages(tma_selected_table, "batch-row"), head_pages(tma_selected_table, "batch-row"), tma_selected_k),
        "kind4_tma_missing_page_table_rejected": ("expected_reject", int(PageResolverKind.RESOLVED_ROW_PTR), PAGE_RESOLVER_SUBKIND_ROWPTR, "batch-row", _carrier_shape(tma_rowptr_selected), head_pages(tma_selected_table, "batch-row"), head_pages(tma_selected_table, "batch-row"), tma_selected_k),
        "kind4_affine_const_tma_page128": ("affine_const", int(PageResolverKind.RESOLVED_ROW_PTR), PAGE_RESOLVER_SUBKIND_AFFINE_CONST, "batch-row", [], head_pages(tma_selected_table, "batch-row"), head_pages(tma_selected_table, "batch-row"), tma_selected_k),
        "kind4_affine_tensor_batch_row_tma_page128": ("affine_tensor", int(PageResolverKind.RESOLVED_ROW_PTR), PAGE_RESOLVER_SUBKIND_AFFINE_TENSOR, "batch-row", _carrier_shape(tma_affine_row), affine_pages(tma_affine_row, "batch-row", ref_case=tma_case), affine_pages(tma_affine_row, "batch-row", ref_case=tma_case), tma_selected_k),
        "kind4_affine_tensor_batch_head_tma_page128": ("affine_tensor", int(PageResolverKind.RESOLVED_ROW_PTR), PAGE_RESOLVER_SUBKIND_AFFINE_TENSOR, "batch-head-row", _carrier_shape(tma_affine_head), affine_pages(tma_affine_head, "batch-head-row", ref_case=tma_case), affine_pages(tma_affine_head, "batch-head-row", ref_case=tma_case), tma_selected_k),
        "kind4_direct_physical_tma_page128": ("direct_physical_const", int(PageResolverKind.RESOLVED_ROW_PTR), PAGE_RESOLVER_SUBKIND_AFFINE_CONST_DIRECT, "batch-row", [], head_pages(tma_selected_table, "batch-row"), head_pages(tma_selected_table, "batch-row"), tma_selected_k),
        "kind4_two_segment_tma_page128": ("two_segment_const", int(PageResolverKind.RESOLVED_ROW_PTR), PAGE_RESOLVER_SUBKIND_AFFINE_CONST, "batch-row", [], head_pages(tma_two_segment_table, "batch-row"), head_pages(tma_two_segment_table, "batch-row"), tma_selected_k),
        "kind4_two_segment_invalid_descriptor_rejected": ("expected_reject", int(PageResolverKind.RESOLVED_ROW_PTR), PAGE_RESOLVER_SUBKIND_AFFINE_TENSOR, "batch-row", _carrier_shape(tma_affine_two_segment), head_pages(tma_two_segment_table, "batch-row"), affine_pages(tma_affine_two_segment, "batch-row", ref_case=tma_case), tma_selected_k),
        "kind4_affine_tma_unaligned_mapping_rejected": ("expected_reject", int(PageResolverKind.RESOLVED_ROW_PTR), PAGE_RESOLVER_SUBKIND_AFFINE_TENSOR, "batch-row", _carrier_shape(tma_affine_row), affine_pages(tma_affine_row, "batch-row", ref_case=tma_case), affine_pages(tma_affine_row, "batch-row", ref_case=tma_case), tma_selected_k),
    }

    reference_by_case: dict[str, tuple[torch.Tensor, torch.Tensor]] = {
        name: (selected_out, selected_lse) for name in KERNEL_ONLY_CASE_NAMES
    }
    for name in (
        "kind1_selected_batch_head_non_tma",
        "kind4_rowptr_batch_head_non_tma",
        "kind4_direct_table_batch_head_non_tma",
    ):
        reference_by_case[name] = (head_out, head_lse)
    # row_consume baseline: kind0_native_full4096 is a full-KV native (both rows full) -> full_out.
    reference_by_case["kind0_native_full4096_non_tma"] = (full_out, full_lse)
    reference_by_case["kind4_affine_tensor_batch_head_non_tma"] = (
        affine_head_out,
        affine_head_lse,
    )
    reference_by_case["kind4_rowptr_irregular_non_tma"] = (irregular_out, irregular_lse)
    reference_by_case["kind4_two_segment_non_tma"] = (two_segment_out, two_segment_lse)
    reference_by_case["kind0_native_tma_page128"] = (tma_native_out, tma_native_lse)
    for name in (
        "kind1_selected_batch_row_tma_page128",
        "kind4_rowptr_batch_row_tma_page128",
        "kind4_direct_table_batch_row_tma_page128",
    ):
        reference_by_case[name] = (tma_selected_out, tma_selected_lse)
    for name in (
        "kind1_selected_batch_head_tma_page128",
        "kind4_rowptr_batch_head_tma_page128",
        "kind4_direct_table_batch_head_tma_page128",
    ):
        reference_by_case[name] = (tma_head_out, tma_head_lse)
    for name in (
        "kind4_affine_const_tma_page128",
        "kind4_affine_tensor_batch_row_tma_page128",
        "kind4_direct_physical_tma_page128",
    ):
        reference_by_case[name] = (tma_selected_out, tma_selected_lse)
    reference_by_case["kind4_affine_tensor_batch_head_tma_page128"] = (
        tma_affine_head_out,
        tma_affine_head_lse,
    )
    reference_by_case["kind4_two_segment_tma_page128"] = (
        tma_two_segment_out,
        tma_two_segment_lse,
    )
    # [ORACLE-SAME-ENVELOPE 2026-07-13] A dynamic split count alone is not the
    # scheduler contract.  In particular, auto static=9/dynamic=5 and explicit
    # static=5/dynamic=5 can produce a different reduction envelope and differ
    # by one BF16 ulp.  Build pinned metadata out of band, then replay both the
    # case and its native reference with that exact tensor while leaving the
    # forward launches in auto mode.  The readback must match the complete
    # split signature (requested/static/dynamic/batch), and the stock case must
    # be bitwise-equal to its metadata replay.  An unproven pin stays a loud
    # fallback; no tolerance is widened.
    #
    # Keying by cached tensor identity keeps reference_by_case as the only
    # case-to-reference source of truth.
    def _native_ref_runner(ref_case, ref_page_table, ref_seqused_k, ref_max_seqlen_k):
        def _run(
            num_splits: int,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] | None:
            if int(num_splits) < 2:
                ref_out, ref_lse = _run_native_reference(
                    bridge,
                    ref_case,
                    page_table=ref_page_table,
                    seqused_k=ref_seqused_k,
                    max_seqlen_k=ref_max_seqlen_k,
                    num_splits=int(num_splits),
                )
                return ref_out, ref_lse, None
            pinned_metadata = _pinned_scheduler_metadata(
                bridge,
                ref_case,
                seqused_k=ref_seqused_k,
                max_seqlen_k=ref_max_seqlen_k,
                num_splits=int(num_splits),
            )
            if pinned_metadata is None:
                return None
            ref_out, ref_lse = _run_native_direct_op(
                bridge,
                ref_case,
                page_table=ref_page_table,
                seqused_k=ref_seqused_k,
                max_seqlen_k=ref_max_seqlen_k,
                scheduler_metadata=pinned_metadata,
                num_splits=None,
            )
            return ref_out, ref_lse, pinned_metadata

        return _run

    def _materialized_ref_runner(ref_case, ref_page_table, ref_seqused_k, ref_max_seqlen_k):
        def _run(
            num_splits: int,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] | None:
            if int(num_splits) < 2:
                ref_out, ref_lse = materialized_reference_for_head_table(
                    ref_case,
                    ref_page_table,
                    ref_seqused_k,
                    ref_max_seqlen_k,
                    num_splits=int(num_splits),
                )
                return ref_out, ref_lse, None
            pinned_metadata = _pinned_scheduler_metadata(
                bridge,
                ref_case,
                seqused_k=ref_seqused_k,
                max_seqlen_k=ref_max_seqlen_k,
                num_splits=int(num_splits),
            )
            if pinned_metadata is None:
                return None
            ref_out, ref_lse = materialized_reference_for_head_table(
                ref_case,
                ref_page_table,
                ref_seqused_k,
                ref_max_seqlen_k,
                num_splits=None,
                scheduler_metadata=pinned_metadata,
            )
            return ref_out, ref_lse, pinned_metadata

        return _run

    reference_runner_by_ref_id = {
        id(selected_out): _native_ref_runner(case, selected_table, selected_seqused, selected_k),
        id(irregular_out): _native_ref_runner(case, irregular_table, selected_seqused, selected_k),
        id(two_segment_out): _native_ref_runner(case, two_segment_table, selected_seqused, selected_k),
        id(full_out): _native_ref_runner(case, case.page_table_i32, case.full_seqused_k_i32, case.shape.kv_len),
        id(head_out): _materialized_ref_runner(case, head_table, selected_seqused, selected_k),
        id(affine_head_out): _materialized_ref_runner(case, affine_head_table, selected_seqused, selected_k),
        id(tma_native_out): _native_ref_runner(tma_case, tma_selected_table, tma_selected_seqused, tma_selected_k),
        id(tma_selected_out): _native_ref_runner(tma_case, tma_selected_table, tma_selected_seqused, tma_selected_k),
        id(tma_head_out): _materialized_ref_runner(tma_case, tma_head_table, tma_selected_seqused, tma_selected_k),
        id(tma_affine_head_out): _materialized_ref_runner(tma_case, tma_affine_head_table, tma_selected_seqused, tma_selected_k),
        id(tma_two_segment_out): _native_ref_runner(tma_case, tma_two_segment_table, tma_selected_seqused, tma_selected_k),
    }

    def _relaunch_reference_same_split(runner, pin_num_splits: int):
        result = runner(int(pin_num_splits))
        if result is None:
            return None
        ref_out, ref_lse, pinned_metadata = result
        torch.cuda.synchronize()
        state = bridge.interface_module.fwd_last_launch_debug_state(ref_out)
        return ref_out, ref_lse, state, pinned_metadata

    split_reference_case = "kind0_native_non_tma_page64"
    correctness: dict[str, tuple[bool, float]] = {}
    oracle_split_status: dict[str, str] = {}
    oracle_reference_num_splits: dict[str, int] = {}
    negative_control_differs: dict[str, bool] = {}
    runtime_errors: dict[str, str] = {}
    launch_states: dict[str, dict[str, object]] = {}
    if any(name not in TMA_CASE_NAMES and name not in TMA_PREDICATE_REJECT_CASE_NAMES for name in active_case_names):
        raw_native_runners[raw_native_non_tma_name]()
        torch.cuda.synchronize()
        raw_native_launch_states[raw_native_non_tma_name] = (
            bridge.interface_module.fwd_last_launch_debug_state(case.q)
        )
    if any(name in TMA_CASE_NAMES and name not in TMA_PREDICATE_REJECT_CASE_NAMES for name in active_case_names):
        raw_native_runners[raw_native_tma_name]()
        torch.cuda.synchronize()
        raw_native_launch_states[raw_native_tma_name] = (
            bridge.interface_module.fwd_last_launch_debug_state(tma_case.q)
        )
    for name in active_case_names:
        if name in TMA_PREDICATE_REJECT_CASE_NAMES:
            correctness[name] = (False, 0.0)
            negative_control_differs[name] = False
            launch_states[name] = {
                "requested_num_splits": 0,
                "resolved_num_splits": 0,
                "use_dynamic_split": False,
                "num_splits_dynamic_min": -1,
                "num_splits_dynamic_max": -1,
                "scheduler_metadata_batch_size": 0,
                "is_pagedkv_tma": False,
                "tile_n": DEFAULT_TMA_TILE_N,
            }
            continue
        try:
            out, lse = runners[name]()
            torch.cuda.synchronize()
            launch_states[name] = bridge.interface_module.fwd_last_launch_debug_state(case.q)
            if name == "kind1_row_consume_visible_k_non_tma":
                correctness[name] = _assert_close_and_diff(
                    actual_out=out,
                    expected_out=row_consume_native_out,
                    actual_lse=lse,
                    expected_lse=row_consume_native_lse,
                )
            else:
                ref_out, ref_lse = reference_by_case[name]
                oracle_split_status[name] = "auto_reference"
                oracle_reference_num_splits[name] = -1
                case_split = _effective_num_splits(launch_states[name])
                same_split_runner = reference_runner_by_ref_id.get(id(ref_out))
                if same_split_runner is not None and case_split is not None:
                    relaunch = _relaunch_reference_same_split(
                        same_split_runner, case_split
                    )
                    if relaunch is None:
                        oracle_split_status[name] = "split_mismatch_fallback"
                    else:
                        pinned_out, pinned_lse, pinned_state, pinned_metadata = relaunch
                        same_envelope = (
                            _split_signature(pinned_state)
                            == _split_signature(launch_states[name])
                        )
                        metadata_replay_ok = True
                        if pinned_metadata is not None:
                            replay_out, replay_lse = _injected_metadata_case_runner(
                                runners[name],
                                pinned_metadata=pinned_metadata,
                            )()
                            torch.cuda.synchronize()
                            replay_state = bridge.interface_module.fwd_last_launch_debug_state(
                                replay_out
                            )
                            metadata_replay_ok = bool(
                                _split_signature(replay_state)
                                == _split_signature(launch_states[name])
                                and _output_and_lse_bitwise_equal(
                                    actual_out=out,
                                    expected_out=replay_out,
                                    actual_lse=lse,
                                    expected_lse=replay_lse,
                                )
                            )
                        if same_envelope and metadata_replay_ok:
                            ref_out, ref_lse = pinned_out, pinned_lse
                            oracle_split_status[name] = "same_split"
                            oracle_reference_num_splits[name] = int(case_split)
                        elif not metadata_replay_ok:
                            oracle_split_status[name] = "metadata_replay_mismatch_fallback"
                        else:
                            oracle_split_status[name] = "split_mismatch_fallback"
                correctness[name] = _assert_close_and_diff(
                    actual_out=out,
                    expected_out=ref_out,
                    actual_lse=lse,
                    expected_lse=ref_lse,
                )
            negative_control_differs[name] = _negative_control_differs(
                actual_out=out,
                full_out=tma_full_out if name in TMA_CASE_NAMES else full_out,
            )
        except Exception as exc:  # pragma: no cover - exercised only on H100 runtime failures
            runtime_errors[name] = repr(exc)
            correctness[name] = (False, math.inf)
            negative_control_differs[name] = False
            launch_states[name] = {
                "requested_num_splits": 0,
                "resolved_num_splits": 0,
                "use_dynamic_split": False,
                "num_splits_dynamic_min": -1,
                "num_splits_dynamic_max": -1,
                "scheduler_metadata_batch_size": 0,
                "is_pagedkv_tma": False,
                "tile_n": 0,
            }

    # [2026-07-12 J1-SPEEDGATE-SAMESPLIT] Plan the speed-gate comparison pair
    # for every timed case (pair-status contract documented at
    # _SPEED_PAIR_MATCHED_STATUSES).  Case closures and their stock timing
    # legs are untouched; mixed-resolver gate ratios move to prepare-free
    # shadow pairs pinned to each case's own auto split solution.
    forced_split_mode = _FORCE_NUM_SPLITS is not None and int(_FORCE_NUM_SPLITS) >= 1
    negative_case_names = {
        "kind4_visible_k_negative_non_tma",
        "kind4_page_mutation_negative_non_tma",
    }
    speed_pair_status: dict[str, str] = {}
    speed_pair_errors: dict[str, str] = {}
    speed_pair_baseline: dict[str, str] = {}
    speed_pair_shadow: dict[str, str] = {}
    speed_pair_pinned_splits: dict[str, int] = {}
    pinned_metadata_by_key: dict[tuple[str, int], torch.Tensor | None] = {}
    pinned_probe_errors_by_key: dict[tuple[str, int], str] = {}
    pinned_runners: dict[str, Callable[[], tuple[torch.Tensor, torch.Tensor]]] = {}
    pinned_launch_states: dict[str, dict[str, object]] = {}

    def _plan_speed_pair(name: str) -> None:
        is_tma = name in TMA_CASE_NAMES
        raw_name = raw_native_tma_name if is_tma else raw_native_non_tma_name
        speed_pair_baseline[name] = raw_name
        speed_pair_shadow[name] = name
        speed_pair_pinned_splits[name] = -1
        if name in TMA_PREDICATE_REJECT_CASE_NAMES or name in runtime_errors:
            speed_pair_status[name] = _SPEED_PAIR_UNTIMED
            return
        case_state = launch_states[name]
        if name == "kind1_row_consume_visible_k_non_tma":
            # Its dense reference is the full-KV kind0 case (mixed entry on
            # both sides = same prepare branch); no pin rescue is possible
            # because the reference is a case, not an injectable leg.
            speed_pair_baseline[name] = "kind0_native_full4096_non_tma"
            speed_pair_status[name] = _speed_pair_split_status(
                case_state=case_state,
                reference_state=launch_states.get("kind0_native_full4096_non_tma", {}),
                forced_split_mode=forced_split_mode,
            )
            return
        direct_status = _speed_pair_split_status(
            case_state=case_state,
            reference_state=raw_native_launch_states.get(raw_name, {}),
            forced_split_mode=forced_split_mode,
        )
        if forced_split_mode:
            # --force-num-splits 1: SingleTile launches carry no prepare at
            # all (N1 dead-launch elimination) on either side, so a split
            # match is already a pure-kernel direct comparison.
            speed_pair_status[name] = direct_status
            return
        resolver_kind_for_case = int(metadata[name][1])
        if resolver_kind_for_case == int(PageResolverKind.NATIVE):
            # Dense prepare branch on both sides: direct comparison.
            speed_pair_status[name] = direct_status
            return
        if name in negative_case_names:
            # Negative controls never enter the speed gate; report the direct
            # pair status without building shadow legs for them.
            speed_pair_status[name] = direct_status
            return
        # Mixed-resolver case: even a same-split direct comparison carries
        # the mixed-vs-dense prepare branch delta (~+0.67us/launch), so the
        # gate ratio must come from a prepare-free shadow pair.
        case_effective = _effective_num_splits(case_state)
        if case_effective is None or str(args.timing_mode) != "cuda_graph":
            speed_pair_status[name] = _SPEED_PAIR_SKIPPED
            return
        ref_case = tma_case if is_tma else case
        ref_table = tma_selected_table if is_tma else selected_table
        ref_seqused = tma_selected_seqused if is_tma else selected_seqused
        ref_seqlen = tma_selected_k if is_tma else selected_k
        meta_key = ("tma" if is_tma else "non_tma", int(case_effective))
        if meta_key not in pinned_metadata_by_key:
            pinned_metadata_by_key[meta_key] = _pinned_scheduler_metadata(
                bridge,
                ref_case,
                seqused_k=ref_seqused,
                max_seqlen_k=ref_seqlen,
                num_splits=int(case_effective),
            )
        if meta_key in pinned_probe_errors_by_key:
            speed_pair_status[name] = _SPEED_PAIR_ERROR
            speed_pair_errors[name] = pinned_probe_errors_by_key[meta_key]
            return
        pinned_meta = pinned_metadata_by_key[meta_key]
        if pinned_meta is None:
            speed_pair_status[name] = _SPEED_PAIR_SKIPPED
            return
        raw_pin_name = f"{raw_name}_pin{int(case_effective)}"
        if raw_pin_name not in pinned_runners:
            pin_runner = _pinned_raw_native_runner(
                bridge,
                ref_case,
                page_table=ref_table,
                seqused_k=ref_seqused,
                max_seqlen_k=ref_seqlen,
                pinned_metadata=pinned_meta,
            )
            try:
                pin_runner()
                torch.cuda.synchronize()
                pin_state = bridge.interface_module.fwd_last_launch_debug_state(ref_case.q)
            except Exception as exc:
                reason = f"pinned_raw_probe: {type(exc).__name__}: {exc}"
                pinned_probe_errors_by_key[meta_key] = reason
                pinned_metadata_by_key[meta_key] = None
                speed_pair_status[name] = _SPEED_PAIR_ERROR
                speed_pair_errors[name] = reason
                return
            if _effective_num_splits(pin_state) != int(case_effective):
                # Pin not proven at launch: poison this tier so every case on
                # it reports an honest skip instead of a fake-matched gate.
                pinned_metadata_by_key[meta_key] = None
                speed_pair_status[name] = _SPEED_PAIR_SKIPPED
                return
            pinned_runners[raw_pin_name] = pin_runner
            pinned_launch_states[raw_pin_name] = pin_state
        shadow_name = f"__pinned{int(case_effective)}__{name}"
        if shadow_name not in pinned_runners:
            shadow_runner = _injected_metadata_case_runner(
                runners[name],
                pinned_metadata=pinned_meta,
            )
            try:
                shadow_runner()
                torch.cuda.synchronize()
                shadow_state = bridge.interface_module.fwd_last_launch_debug_state(ref_case.q)
            except Exception as exc:
                reason = f"injected_shadow_probe: {type(exc).__name__}: {exc}"
                speed_pair_status[name] = _SPEED_PAIR_ERROR
                speed_pair_errors[name] = reason
                return
            if _effective_num_splits(shadow_state) != int(case_effective):
                speed_pair_status[name] = _SPEED_PAIR_SKIPPED
                return
            pinned_runners[shadow_name] = shadow_runner
            pinned_launch_states[shadow_name] = shadow_state
        speed_pair_baseline[name] = raw_pin_name
        speed_pair_shadow[name] = shadow_name
        speed_pair_pinned_splits[name] = int(case_effective)
        speed_pair_status[name] = "matched_pinned_split"

    for name in active_case_names:
        _plan_speed_pair(name)

    timings: dict[str, float] = {
        name: math.inf
        for name in active_case_names
        if name in TMA_PREDICATE_REJECT_CASE_NAMES or name in runtime_errors
    }
    timing_runners: list[tuple[str, Callable[[], tuple[torch.Tensor, torch.Tensor]]]] = []
    needs_non_tma_raw_native = any(
        name not in TMA_CASE_NAMES and name not in timings
        for name in active_case_names
    )
    needs_tma_raw_native = any(
        name in TMA_CASE_NAMES and name not in timings
        for name in active_case_names
    )
    if needs_non_tma_raw_native:
        timing_runners.append((raw_native_non_tma_name, raw_native_runners[raw_native_non_tma_name]))
    if needs_tma_raw_native:
        timing_runners.append((raw_native_tma_name, raw_native_runners[raw_native_tma_name]))
    timing_runners.extend(
        (name, runners[name])
        for name in active_case_names
        if name not in timings
    )
    # Shadow-pair legs (prepare-free pinned case/raw forms) rotate inside the
    # same timing group so their pair ratios stay drift-immune.
    timing_runners.extend(sorted(pinned_runners.items()))
    try:
        timings.update(
            _measure_kernel_us(
                timing_runners,
                warmup=int(args.warmup),
                repeat=int(args.iters),
                inner_iters=int(args.inner_iters),
                timing_mode=str(args.timing_mode),
            )
        )
    except Exception:  # pragma: no cover - keeps one unexpected timing failure scoped
        for name, runner in timing_runners:
            try:
                timings[name] = _measure_kernel_us(
                    [(name, runner)],
                    warmup=int(args.warmup),
                    repeat=int(args.iters),
                    inner_iters=int(args.inner_iters),
                    timing_mode=str(args.timing_mode),
                )[name]
            except Exception:  # pragma: no cover - exercised only on H100 runtime failures
                timings[name] = math.inf

    extension_override = str(getattr(args, "extension_path", "") or "")
    extension_path = extension_override or _extension_path()
    gpu_end_util, gpu_end_memory = _gpu_state()
    sm_arch = _sm_arch()
    split_reference = launch_states[split_reference_case]
    non_tma_native_resolver_us = timings.get("kind0_native_non_tma_page64", math.inf)
    tma_native_resolver_us = timings.get("kind0_native_tma_page128", math.inf)
    raw_non_tma_native_us = timings.get(raw_native_non_tma_name, math.inf)
    raw_tma_native_us = timings.get(raw_native_tma_name, math.inf)
    _raw_samples_snapshot = dict(_LAST_RAW_SAMPLES)

    def _paired_vs_raw_native_pct(case_name: str, native_runner_name: str):
        # median_i(carrier_i / native_i) - 1 over per-index samples (drift-immune).
        cs = _raw_samples_snapshot.get(case_name)
        ns = _raw_samples_snapshot.get(native_runner_name)
        if not cs or not ns:
            return None
        count = min(len(cs), len(ns))
        ratios = [cs[i] / ns[i] for i in range(count) if ns[i] > 0.0]
        if not ratios:
            return None
        return (float(median(ratios)) - 1.0) * 100.0

    records: list[ResolverKernelOnlyRecord] = []
    for name in active_case_names:
        (
            resolver_subkind,
            resolver_kind,
            _subkind_value,
            layout_class,
            carrier_shape,
            expected_pages,
            actual_pages,
            effective_visible_k,
        ) = metadata[name]
        kernel_us = timings[name]
        baseline_name = (
            "kind1_selected_batch_row_tma_page128"
            if name in TMA_CASE_NAMES
            else "kind1_selected_batch_row_non_tma"
        )
        baseline_us = timings.get(baseline_name, math.inf)
        native_resolver_us = (
            tma_native_resolver_us
            if name in TMA_CASE_NAMES
            else non_tma_native_resolver_us
        )
        raw_native_auto_us = (
            raw_tma_native_us
            if name in TMA_CASE_NAMES
            # row_consume_visible_k deliberately attends FULL 4096-KV on its mode=0 row, so its
            # matched dense baseline is the full-KV native, not the shared sparse-1024 raw_native.
            else timings.get("kind0_native_full4096_non_tma", raw_non_tma_native_us)
            if name == "kind1_row_consume_visible_k_non_tma"
            else raw_non_tma_native_us
        )
        # [2026-07-12 J1-SPEEDGATE-SAMESPLIT] Gate ratio legs: shadow name is
        # the case itself and baseline the stock reference unless the pair
        # planner built a prepare-free pinned shadow pair.
        pair_status = speed_pair_status.get(name, _SPEED_PAIR_UNTIMED)
        pair_baseline_name = speed_pair_baseline.get(
            name,
            raw_native_tma_name if name in TMA_CASE_NAMES else raw_native_non_tma_name,
        )
        pair_shadow_name = speed_pair_shadow.get(name, name)
        pair_pinned_splits = int(speed_pair_pinned_splits.get(name, -1))
        raw_native_us = timings.get(pair_baseline_name, raw_native_auto_us)
        shadow_kernel_us = timings.get(pair_shadow_name, kernel_us)
        raw_native_state = raw_native_launch_states.get(
            raw_native_tma_name if name in TMA_CASE_NAMES else raw_native_non_tma_name,
            {},
        )
        if pair_baseline_name in pinned_launch_states:
            # matched_pinned_split: report the pinned baseline's probe state
            # (dyn slice read back from the injected buffer = in-file proof).
            raw_native_state = pinned_launch_states[pair_baseline_name]
        overhead_pct = (
            (float(kernel_us) / float(baseline_us) - 1.0) * 100.0
            if math.isfinite(kernel_us) and baseline_us > 0.0 and math.isfinite(baseline_us)
            else math.inf
        )
        vs_native_resolver_pct = (
            (float(kernel_us) / float(native_resolver_us) - 1.0) * 100.0
            if math.isfinite(kernel_us) and native_resolver_us > 0.0 and math.isfinite(native_resolver_us)
            else math.inf
        )
        _paired_vs_raw_native = _paired_vs_raw_native_pct(
            pair_shadow_name,
            pair_baseline_name,
        )
        vs_raw_native_pct = (
            float(_paired_vs_raw_native)
            if _paired_vs_raw_native is not None
            else (
                (float(shadow_kernel_us) / float(raw_native_us) - 1.0) * 100.0
                if math.isfinite(shadow_kernel_us) and raw_native_us > 0.0 and math.isfinite(raw_native_us)
                else math.inf
            )
        )
        split_match_status = _split_match_status(
            reference=split_reference,
            candidate=launch_states[name],
        )
        is_negative = name in {
            "kind4_visible_k_negative_non_tma",
            "kind4_page_mutation_negative_non_tma",
        }
        is_tma_positive = name in TMA_POSITIVE_CASE_NAMES
        is_tma_predicate_reject = name in TMA_PREDICATE_REJECT_CASE_NAMES
        speed_gated_kind = (
            resolver_kind == int(PageResolverKind.NATIVE)
            or resolver_kind == int(PageResolverKind.RESOLVED_ROW_PTR)
            or resolver_kind == int(PageResolverKind.SELECTED_TABLE)
        )
        speed_gated = (
            speed_gated_kind
            and pair_status in _SPEED_PAIR_MATCHED_STATUSES
            and not is_negative
            and not is_tma_predicate_reject
            and name != "kind0_native_full4096_non_tma"
        )
        speed_hard_fail = bool(
            speed_gated
            and (
                not math.isfinite(vs_raw_native_pct)
                or vs_raw_native_pct > float(args.hard_fail_overhead_pct)
            )
        )
        actual_is_tma = bool(launch_states[name].get("is_pagedkv_tma", False))
        actual_tile_n = int(launch_states[name].get("tile_n", 0))
        tma_route_pass = bool(
            not is_tma_positive
            or (actual_is_tma and actual_tile_n == DEFAULT_TMA_TILE_N)
        )
        output_allclose = bool(correctness[name][0])
        negative_pass = bool(is_negative and not output_allclose and negative_control_differs[name])
        predicate_reject_source_checked = bool(is_tma_predicate_reject)
        correctness_passed = bool(
            (output_allclose and tma_route_pass)
            or negative_pass
            or predicate_reject_source_checked
        )
        # [2026-07-11 J1] Retired kind/subkind forms (SelectedTable kind at launch;
        # DirectTable/AffineConst/AffineTensor subkinds on the SM8x TU after the
        # instance diet) MUST fail-closed on this arch: the rejection IS the case
        # contract, mirroring the capcheck retired-subkind encoding. SM90 builds
        # still execute these cases numerically.
        retired_fail_closed = bool(
            name in runtime_errors
            and any(
                marker in str(runtime_errors[name])
                for marker in _RETIRED_FORM_MARKERS
            )
        )
        pass_fail = (
            "predicate_reject_source_checked"
            if predicate_reject_source_checked
            else "retired_fail_closed"
            if retired_fail_closed
            else "runtime_fail"
            if name in runtime_errors
            else "speed_pair_error"
            if name in speed_pair_errors
            else "tma_route_fail"
            if is_tma_positive and not tma_route_pass
            else "negative_pass"
            if negative_pass
            else "pass"
            if output_allclose
            else "negative_control_fail"
            if is_negative
            else "fail"
        )
        gate_passed = pass_fail in {
            "pass",
            "negative_pass",
            "predicate_reject_source_checked",
            "retired_fail_closed",
        }
        row_modes = row_consume_values if name == "kind1_row_consume_visible_k_non_tma" else [
            ROW_CONSUME_MODE_SELECTED_I32
        ] * batch
        effective_visible_by_row = [
            int(effective_visible_k)
            for _ in range(batch * num_kv_heads)
        ]
        if name == "kind1_row_consume_visible_k_non_tma":
            effective_visible_by_row = [
                int(selected_k) if row == 0 else int(case.shape.kv_len)
                for row in range(batch)
                for _ in range(num_kv_heads)
            ]
        records.append(
            ResolverKernelOnlyRecord(
                benchmark="mixed_page_resolver_kernel_only",
                case_name=name,
                page_resolver_kind=int(resolver_kind),
                resolver_kind=int(resolver_kind),
                resolver_subkind=str(resolver_subkind),
                tma_or_non_tma="tma" if actual_is_tma else "non_tma",
                page_size=KERNEL_ONLY_TMA_PAGE_SIZE if name in TMA_CASE_NAMES else int(case.shape.page_size),
                tile_n=actual_tile_n or (DEFAULT_TMA_TILE_N if name in TMA_CASE_NAMES else DEFAULT_NON_TMA_TILE_N),
                effective_visible_k=int(effective_visible_k),
                effective_visible_k_by_row=[int(value) for value in effective_visible_by_row],
                selected_or_resolved_carrier_shape=[int(value) for value in carrier_shape],
                expected_physical_pages=[[int(value) for value in row] for row in expected_pages],
                actual_physical_pages=[[int(value) for value in row] for row in actual_pages],
                layout_class=str(layout_class),
                row_consume_mode=[int(value) for value in row_modes],
                capture_enabled=False,
                output_allclose=output_allclose,
                capture_allclose_when_enabled=None,
                route_counter=f"page_resolver_kind{int(resolver_kind)}_count",
                extension_build_id_or_so_path=extension_path,
                source_status=(
                    tma_predicate_reject_reasons[name]
                    if is_tma_predicate_reject
                    else speed_pair_errors[name]
                    if name in speed_pair_errors
                    else "runtime"
                    if name not in runtime_errors
                    else runtime_errors[name]
                ),
                gpu_name=_cuda_device_name(),
                num_q_heads=int(case.shape.num_q_heads),
                num_kv_heads=num_kv_heads,
                kv_batch_idx_present=False,
                graph_cache_hit_when_applicable=None,
                graph_recapture_count_when_applicable=None,
                negative_control_differs=bool(negative_control_differs[name]),
                sm_arch=sm_arch,
                tma_mode="tma" if actual_is_tma else "non_tma",
                batch=batch,
                heads=int(case.shape.num_q_heads),
                page_block_size=KERNEL_ONLY_TMA_PAGE_SIZE if name in TMA_CASE_NAMES else int(case.shape.page_size),
                q_tokens=int(tma_case.shape.batch_size * tma_case.shape.q_len) if name in TMA_CASE_NAMES else int(case.shape.batch_size * case.shape.q_len),
                requested_kv_len=int(case.shape.kv_len),
                kv_tokens=int(effective_visible_k),
                selected_pages=selected_pages,
                warmup=int(args.warmup),
                iters=int(args.iters),
                inner_iters=int(args.inner_iters),
                timing_mode=str(args.timing_mode),
                kernel_us=float(kernel_us),
                baseline_kernel_us=float(baseline_us),
                baseline_name=baseline_name,
                baseline_overhead_pct=float(overhead_pct),
                native_resolver_kernel_us=float(native_resolver_us),
                vs_native_resolver_overhead_pct=float(vs_native_resolver_pct),
                raw_native_kernel_us=float(raw_native_us),
                vs_raw_native_overhead_pct=float(vs_raw_native_pct),
                speed_raw_pair_status=str(pair_status),
                speed_raw_pair_error=str(speed_pair_errors.get(name, "")),
                speed_raw_baseline_name=str(pair_baseline_name),
                speed_raw_shadow_name=str(pair_shadow_name),
                speed_raw_pinned_num_splits=int(pair_pinned_splits),
                raw_native_auto_kernel_us=float(raw_native_auto_us),
                pinned_case_kernel_us=(
                    float(shadow_kernel_us) if pair_shadow_name != name else -1.0
                ),
                launch_is_mixed_page=bool(launch_states[name].get("is_mixed_page", False)),
                auto_requested_num_splits=int(launch_states[name]["requested_num_splits"]),
                auto_resolved_num_splits=int(launch_states[name]["resolved_num_splits"]),
                auto_use_dynamic_split=bool(launch_states[name]["use_dynamic_split"]),
                auto_num_splits_dynamic_min=int(launch_states[name]["num_splits_dynamic_min"]),
                auto_num_splits_dynamic_max=int(launch_states[name]["num_splits_dynamic_max"]),
                scheduler_metadata_batch_size=int(launch_states[name]["scheduler_metadata_batch_size"]),
                raw_native_requested_num_splits=int(raw_native_state.get("requested_num_splits", 0)),
                raw_native_resolved_num_splits=int(raw_native_state.get("resolved_num_splits", 0)),
                raw_native_use_dynamic_split=bool(raw_native_state.get("use_dynamic_split", False)),
                raw_native_num_splits_dynamic_min=int(raw_native_state.get("num_splits_dynamic_min", -1)),
                raw_native_num_splits_dynamic_max=int(raw_native_state.get("num_splits_dynamic_max", -1)),
                raw_native_scheduler_metadata_batch_size=int(raw_native_state.get("scheduler_metadata_batch_size", 0)),
                raw_native_is_mixed_page=bool(raw_native_state.get("is_mixed_page", False)),
                raw_native_is_pagedkv_tma=bool(raw_native_state.get("is_pagedkv_tma", False)),
                raw_native_tile_n=int(raw_native_state.get("tile_n", 0)),
                split_reference_case=split_reference_case,
                split_match_status=split_match_status,
                compute_match_status="matched_q_k_pages",
                fairness_passed=True,
                correctness_passed=correctness_passed,
                reference_max_abs_diff=float(correctness[name][1]),
                oracle_split_status=oracle_split_status.get(name, "not_applicable"),
                oracle_reference_num_splits=int(oracle_reference_num_splits.get(name, -1)),
                gate_passed=gate_passed,
                hard_fail=speed_hard_fail,
                pass_fail=pass_fail,
                extension_path=extension_path,
                extension_sha256=_file_sha256(extension_path),
                git_sha=_git_sha(),
                git_dirty=_git_dirty(),
                cuda_device_name=_cuda_device_name(),
                gpu_util_start_pct=int(gpu_start_util),
                gpu_memory_used_start_mb=int(gpu_start_memory),
                gpu_util_end_pct=int(gpu_end_util),
                gpu_memory_used_end_mb=int(gpu_end_memory),
                route_proof=(
                    "CUDA-event rotated direct-op kernel-only non-TMA matrix: kind0, "
                    "kind1 selected, and kind4 RowPtr/DirectTable/Affine resolver "
                    "forms with batch-row and batch-head-row physical-page oracle rows"
                ),
            )
        )
    return records


def write_records(output: Path, records: list[ResolverKernelOnlyRecord]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record_to_jsonable(record), allow_nan=False, sort_keys=True))
            handle.write("\n")


def _int_list(value: str, fallback: int) -> list[int]:
    if not value:
        return [int(fallback)]
    return [int(part) for part in value.split(",") if part.strip()]


def _summary(
    records: list[ResolverKernelOnlyRecord],
    *,
    max_overhead_pct: float = 0.5,
    hard_fail_overhead_pct: float = 2.0,
    expected_case_names: tuple[str, ...] = KERNEL_ONLY_CASE_NAMES,
    expected_cells: tuple[tuple[int, int], ...] = (),
) -> dict[str, object]:
    expected_cases = set(expected_case_names)
    actual_cells = {
        (int(record.batch), int(record.requested_kv_len))
        for record in records
    }
    sweep_cells = tuple(sorted(set(expected_cells) or actual_cells))
    multi_cell = len(sweep_cells) > 1

    def _cell_key(record: ResolverKernelOnlyRecord) -> tuple[int, int]:
        return (int(record.batch), int(record.requested_kv_len))

    def _case_label(case_name: str, cell: tuple[int, int]) -> str:
        if not multi_cell:
            return case_name
        batch_size, kv_len = cell
        return f"{case_name}@bs{batch_size}/kv{kv_len}"

    def _record_label(record: ResolverKernelOnlyRecord) -> str:
        return _case_label(record.case_name, _cell_key(record))

    actual_case_cells = [(_cell_key(record), record.case_name) for record in records]
    actual_case_cell_counts = Counter(actual_case_cells)
    if sweep_cells:
        expected_case_cells = {
            (cell, case_name)
            for cell in sweep_cells
            for case_name in expected_cases
        }
        missing_cases = sorted(
            _case_label(case_name, cell)
            for cell, case_name in expected_case_cells.difference(actual_case_cell_counts)
        )
        duplicate_cases = sorted(
            _case_label(case_name, cell)
            for (cell, case_name), count in actual_case_cell_counts.items()
            if count > 1
        )
    else:
        # Preserve the empty-input contract for direct callers that do not
        # provide an expected sweep: every requested case is missing.
        missing_cases = sorted(expected_cases)
        duplicate_cases = []
    negative_cases = {
        "kind4_visible_k_negative_non_tma",
        "kind4_page_mutation_negative_non_tma",
    }
    expected_reject_cases = set(TMA_PREDICATE_REJECT_CASE_NAMES)
    retired_fail_closed_cases = [
        _record_label(record)
        for record in records
        if record.pass_fail == "retired_fail_closed"
    ]
    negative_control_mismatches = [
        _record_label(record)
        for record in records
        if record.case_name in negative_cases
        and record.pass_fail not in ("negative_pass", "retired_fail_closed")
    ]
    expected_reject_failures = [
        _record_label(record)
        for record in records
        if record.case_name in expected_reject_cases
        and record.pass_fail != "predicate_reject_source_checked"
    ]
    tma_route_failures = [
        _record_label(record)
        for record in records
        if record.case_name in TMA_POSITIVE_CASE_NAMES
        and (record.tma_mode != "tma" or record.tile_n != DEFAULT_TMA_TILE_N)
    ]
    # [2026-07-11 J1] The 0.4% kernel-tax gate only holds in cuda_graph timing
    # (host wrapper stripped, production replay form). launch_event timings on
    # small shapes are dominated by the mixed-entry host tax (kind0 pays it too:
    # ~+10% at ~30us kernels), so there the comparison is reported but not gated.
    # [2026-07-12 J1-SPEEDGATE-SAMESPLIT] Additionally, the gate only fires on
    # same-split/same-prepare-regime pairs (speed_raw_pair_status matched_*):
    # mixed-resolver ratios come from prepare-free pinned shadow pairs, and
    # pairs that could not be aligned are exposed in speed_pair_skipped
    # instead of failing red (or passing green) on a meaningless ratio.
    def _speed_gate_scoped(record: ResolverKernelOnlyRecord) -> bool:
        return (
            record.timing_mode == "cuda_graph"
            and (
                record.resolver_kind == int(PageResolverKind.NATIVE)
                or record.resolver_kind == int(PageResolverKind.RESOLVED_ROW_PTR)
                or record.resolver_kind == int(PageResolverKind.SELECTED_TABLE)
            )
            and record.case_name not in negative_cases
            and record.case_name not in expected_reject_cases
            and record.pass_fail != "retired_fail_closed"
            and record.case_name != "kind0_native_full4096_non_tma"
        )

    speed_gate_failures = [
        _record_label(record)
        for record in records
        if _speed_gate_scoped(record)
        and record.speed_raw_pair_status in _SPEED_PAIR_MATCHED_STATUSES
        and (
            not math.isfinite(record.vs_raw_native_overhead_pct)
            or record.vs_raw_native_overhead_pct > float(max_overhead_pct)
        )
    ]
    speed_pair_skipped = [
        _record_label(record)
        for record in records
        if _speed_gate_scoped(record)
        and record.speed_raw_pair_status not in _SPEED_PAIR_MATCHED_STATUSES
    ]
    speed_pair_errors = {
        _record_label(record): record.speed_raw_pair_error
        for record in records
        if record.speed_raw_pair_error
    }
    # A filtered correctness run may legitimately contain no production RRP
    # case.  Once an RRP case is in scope, however, an all-skipped shadow-pair
    # population is an inconclusive speed verdict, not a green gate.  Keep the
    # individual split-mismatch records honest and fail closed at the aggregate
    # evidence boundary; callers can pin a common split regime.
    resolved_speed_evidence_required = any(
        _speed_gate_scoped(record)
        and record.resolver_kind == int(PageResolverKind.RESOLVED_ROW_PTR)
        for record in records
    )
    resolved_speed_evidence_present = any(
        _speed_gate_scoped(record)
        and record.resolver_kind == int(PageResolverKind.RESOLVED_ROW_PTR)
        and record.speed_raw_pair_status in _SPEED_PAIR_MATCHED_STATUSES
        for record in records
    )
    # Completeness is per in-scope record, not a global any(). Otherwise one
    # matched cell could mask a split-skipped cell in --batch-sizes/--kv-lens
    # sweeps. Include the shape in each reason so a blocked sweep is directly
    # actionable without reopening the JSONL records.
    resolved_speed_evidence_missing = [
        (
            f"{record.case_name}@bs{record.batch}"
            f"/kv{record.requested_kv_len}"
        )
        for record in records
        if _speed_gate_scoped(record)
        and record.resolver_kind == int(PageResolverKind.RESOLVED_ROW_PTR)
        and record.speed_raw_pair_status not in _SPEED_PAIR_MATCHED_STATUSES
    ]
    resolved_speed_evidence_complete = not resolved_speed_evidence_missing
    speed_gate_scope = (
        "cuda_graph"
        if any(record.timing_mode == "cuda_graph" for record in records)
        else "launch_event_reported_not_gated"
    )
    runtime_failures = [
        _record_label(record)
        for record in records
        if record.pass_fail == "runtime_fail"
    ]
    positive_failures = [
        _record_label(record)
        for record in records
        if record.case_name not in negative_cases
        and record.case_name not in expected_reject_cases
        and record.pass_fail != "retired_fail_closed"
        and not record.output_allclose
    ]
    raw_native_launch_states: dict[str, dict[str, object]] = {}
    for record in records:
        state_key = record.tma_or_non_tma
        if multi_cell:
            batch_size, kv_len = _cell_key(record)
            state_key = f"{state_key}@bs{batch_size}/kv{kv_len}"
        if state_key in raw_native_launch_states:
            continue
        raw_native_launch_states[state_key] = {
            "requested_num_splits": record.raw_native_requested_num_splits,
            "resolved_num_splits": record.raw_native_resolved_num_splits,
            "use_dynamic_split": record.raw_native_use_dynamic_split,
            "num_splits_dynamic_min": record.raw_native_num_splits_dynamic_min,
            "num_splits_dynamic_max": record.raw_native_num_splits_dynamic_max,
            "scheduler_metadata_batch_size": record.raw_native_scheduler_metadata_batch_size,
            "is_mixed_page": record.raw_native_is_mixed_page,
            "is_pagedkv_tma": record.raw_native_is_pagedkv_tma,
            "tile_n": record.raw_native_tile_n,
        }
    kind0_launch_states = {
        _record_label(record): {
            "requested_num_splits": record.auto_requested_num_splits,
            "resolved_num_splits": record.auto_resolved_num_splits,
            "use_dynamic_split": record.auto_use_dynamic_split,
            "num_splits_dynamic_min": record.auto_num_splits_dynamic_min,
            "num_splits_dynamic_max": record.auto_num_splits_dynamic_max,
            "scheduler_metadata_batch_size": record.scheduler_metadata_batch_size,
            "is_mixed_page": record.launch_is_mixed_page,
            "is_pagedkv_tma": record.tma_mode == "tma",
            "tile_n": record.tile_n,
        }
        for record in records
        if record.resolver_kind == int(PageResolverKind.NATIVE)
    }
    return {
        "benchmark": "mixed_page_resolver_kernel_only",
        "records": len(records),
        "gate_family": "mixed_page_tma_non_tma_completeness",
        "force_num_splits": _FORCE_NUM_SPLITS,
        "gate_passed": bool(
            not missing_cases
            and not duplicate_cases
            and not runtime_failures
            and not positive_failures
            and not negative_control_mismatches
            and not expected_reject_failures
            and not tma_route_failures
            and not speed_gate_failures
            and not speed_pair_errors
            and resolved_speed_evidence_complete
        ),
        "fairness_passed": all(record.fairness_passed for record in records),
        "negative_control_passed": not negative_control_mismatches,
        "negative_control_mismatches": negative_control_mismatches,
        "expected_reject_failures": expected_reject_failures,
        "tma_route_failures": tma_route_failures,
        "speed_gate_failures": speed_gate_failures,
        "speed_pair_skipped": speed_pair_skipped,
        "speed_pair_errors": speed_pair_errors,
        "resolved_speed_evidence_required": resolved_speed_evidence_required,
        "resolved_speed_evidence_present": resolved_speed_evidence_present,
        "resolved_speed_evidence_missing": resolved_speed_evidence_missing,
        "resolved_speed_evidence_complete": resolved_speed_evidence_complete,
        "speed_pair_statuses": {
            _record_label(record): {
                "status": record.speed_raw_pair_status,
                "error": record.speed_raw_pair_error,
                "baseline": record.speed_raw_baseline_name,
                "shadow": record.speed_raw_shadow_name,
                "pinned_num_splits": record.speed_raw_pinned_num_splits,
            }
            for record in records
        },
        "runtime_failures": runtime_failures,
        "positive_failures": positive_failures,
        "retired_fail_closed": retired_fail_closed_cases,
        "missing_cases": missing_cases,
        "duplicate_cases": duplicate_cases,
        "hard_fail": any(
            record.hard_fail
            for record in records
            if record.pass_fail != "retired_fail_closed"
        ),
        "resolved_row_ptr_missing": not any(
            record.resolver_kind == int(PageResolverKind.RESOLVED_ROW_PTR)
            for record in records
        ),
        "split_mismatches": [
            _record_label(record)
            for record in records
            if record.case_name not in expected_reject_cases
            and record.pass_fail != "retired_fail_closed"
            and record.split_match_status != "matched_auto_split"
        ],
        "resolved_row_ptr_baseline_name": "kind0_native_non_tma_page64",
        "speed_gate_scope": speed_gate_scope,
        "max_resolved_vs_raw_native_overhead_pct": max(
            (
                record.vs_raw_native_overhead_pct
                for record in records
                if record.resolver_kind == int(PageResolverKind.RESOLVED_ROW_PTR)
                # [2026-07-12 J1-SPEEDGATE-SAMESPLIT] headline max = the gate
                # population only (same scope as speed_gate_failures): pair-
                # matched ratios, no negative controls (their stock direct
                # ratio still carries the prepare-branch delta), no skipped
                # cross-regime ratios.
                and _speed_gate_scoped(record)
                and record.speed_raw_pair_status in _SPEED_PAIR_MATCHED_STATUSES
                and math.isfinite(record.vs_raw_native_overhead_pct)
            ),
            default=math.inf,
        ),
        "resolved_row_ptr_overhead_gate_pct": float(max_overhead_pct),
        "resolved_row_ptr_hard_fail_pct": float(hard_fail_overhead_pct),
        "raw_native_launch_states": raw_native_launch_states,
        "kind0_launch_states": kind0_launch_states,
        "retired_resolver_kinds": (2, 3),
        "expected_cases": list(expected_case_names),
        "sweep_cells": [
            {"batch_size": batch_size, "kv_len": kv_len}
            for batch_size, kv_len in sweep_cells
        ],
        "cases": [_record_label(record) for record in records],
    }


def write_summary(output: Path, summary: dict[str, object]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(_json_sanitize(summary), allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if str(args.extension_path):
        args.extension_path = _load_extension_override(str(args.extension_path))
    force_num_splits = int(getattr(args, "force_num_splits", -1))
    if force_num_splits not in (-1, 0, 1):
        raise SystemExit(
            "--force-num-splits must be -1, 0, or 1 (kind4 rejects explicit >1)"
        )
    global _FORCE_NUM_SPLITS
    _FORCE_NUM_SPLITS = None if force_num_splits < 0 else force_num_splits
    records: list[ResolverKernelOnlyRecord] = []
    sweep_cells = tuple(
        (batch_size, kv_len)
        for batch_size in _int_list(str(args.batch_sizes), int(args.batch_size))
        for kv_len in _int_list(str(args.kv_lens), int(args.kv_len))
    )
    for batch_size, kv_len in sweep_cells:
        case_args = argparse.Namespace(**vars(args))
        case_args.batch_size = int(batch_size)
        case_args.kv_len = int(kv_len)
        case_args.selected_pages = int(args.selected_pages)
        records.extend(run_kernel_only(case_args))
    write_records(Path(str(args.output)), records)
    summary = _summary(
        records,
        max_overhead_pct=float(args.max_overhead_pct),
        hard_fail_overhead_pct=float(args.hard_fail_overhead_pct),
        expected_case_names=_case_filter_names(str(args.case_filter)),
        expected_cells=sweep_cells,
    )
    if args.summary_output:
        write_summary(Path(str(args.summary_output)), summary)
    print(json.dumps(_json_sanitize(summary), allow_nan=False, sort_keys=True))
    return 2 if bool(args.fail_on_gate) and not bool(summary["gate_passed"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
