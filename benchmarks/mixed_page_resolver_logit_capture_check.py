from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from itertools import accumulate
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_FA3_ROOT = REPO_ROOT / "third_party_upstreams" / "vllm-project-flash-attention"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(UPSTREAM_FA3_ROOT) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_FA3_ROOT))

from patches.fa3_native.mixed_page_graph_descriptor import (  # noqa: E402
    PageResolverKind,
    ResolverGraphDescriptor,
)
from patches.fa3_native.row_consume_modes import (  # noqa: E402
    ROW_CONSUME_MODE_FULL_I32,
    ROW_CONSUME_MODE_SELECTED_I32,
)


GATE_FAILURE_EXIT_CODE = 2
DEFAULT_ATOL = 2e-2
DEFAULT_RTOL = 2e-2
SENTINEL_VALUE = -1234.0
FA3_EXTENSION_GLOBS = (
    "_vllm_fa3_C*.so",
    "_vllm_fa3_C*.pyd",
)


@dataclass(frozen=True)
class LogitCaptureCheckRecord:
    case: str
    resolver_kind: int
    row_modes: list[int]
    max_abs_diff: float
    allclose_passed: bool
    capture_rows: int
    graph_replay: bool
    capture_allclose_when_enabled: bool | None = None
    capture_dtype: str = "torch.float16"
    tma_or_non_tma: str = "non_tma"
    stale_sentinel_unchanged: bool = True
    # [2026-07-11 CAP-4] Only set by split-forcing cases: capture buffer of the
    # forced-split run must be BITWISE identical to the single-split run.
    split_invariant_bitwise: bool | None = None


@dataclass(frozen=True)
class CaseSpec:
    name: str
    resolver_kind: int
    row_modes: tuple[int, ...]
    capture_rows: int
    tma_or_non_tma: str = "non_tma"
    graph_replay: bool = False


CASE_SPECS = (
    CaseSpec(
        "kind0_native_capture_non_tma",
        int(PageResolverKind.NATIVE),
        (ROW_CONSUME_MODE_FULL_I32,),
        1,
    ),
    CaseSpec(
        "kind4_rrp_capture_non_tma",
        int(PageResolverKind.RESOLVED_ROW_PTR),
        (ROW_CONSUME_MODE_SELECTED_I32,),
        1,
    ),
    CaseSpec(
        "kind4_direct_table_capture_non_tma",
        int(PageResolverKind.RESOLVED_ROW_PTR),
        (ROW_CONSUME_MODE_SELECTED_I32,),
        1,
    ),
    CaseSpec(
        "kind4_affine_tensor_capture_non_tma",
        int(PageResolverKind.RESOLVED_ROW_PTR),
        (ROW_CONSUME_MODE_SELECTED_I32,),
        1,
    ),
    CaseSpec(
        "kind0_native_capture_tma",
        int(PageResolverKind.NATIVE),
        (ROW_CONSUME_MODE_FULL_I32,),
        1,
        "tma",
    ),
    CaseSpec(
        "kind4_rrp_capture_tma",
        int(PageResolverKind.RESOLVED_ROW_PTR),
        (ROW_CONSUME_MODE_SELECTED_I32,),
        1,
        "tma",
    ),
    CaseSpec(
        "kind4_direct_table_capture_tma",
        int(PageResolverKind.RESOLVED_ROW_PTR),
        (ROW_CONSUME_MODE_SELECTED_I32,),
        1,
        "tma",
    ),
    CaseSpec(
        "kind4_affine_tensor_capture_tma",
        int(PageResolverKind.RESOLVED_ROW_PTR),
        (ROW_CONSUME_MODE_SELECTED_I32,),
        1,
        "tma",
    ),
    CaseSpec(
        "kind4_mixed_capture_non_tma",
        int(PageResolverKind.RESOLVED_ROW_PTR),
        (ROW_CONSUME_MODE_SELECTED_I32, ROW_CONSUME_MODE_FULL_I32),
        2,
    ),
    CaseSpec(
        "kind4_cp_capture_fail_closed",
        int(PageResolverKind.RESOLVED_ROW_PTR),
        (ROW_CONSUME_MODE_SELECTED_I32,),
        1,
    ),
    # [2026-07-11 CAP-4] Split x capture pin. The capture-only dispatch site passes
    # num_splits through (auto), so NATIVE trigger-step captures already run under
    # dynamic split in production; this was an untested surface before opening the
    # kind4 steady-state split gate (K1). Explicit num_splits is only legal for
    # NATIVE (flash_api.cpp rejects >1 for kind4), which matches the production
    # exposure.
    CaseSpec(
        "kind0_native_split4_capture_non_tma",
        int(PageResolverKind.NATIVE),
        (ROW_CONSUME_MODE_FULL_I32, ROW_CONSUME_MODE_FULL_I32),
        2,
    ),
)
CASE_BY_NAME = {case.name: case for case in CASE_SPECS}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check mixed_page resolver logit capture against materialized references."
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    parser.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    if args.warmup < 0:
        parser.error("--warmup must be >= 0")
    if args.iters <= 0:
        parser.error("--iters must be > 0")
    if args.atol < 0.0 or args.rtol < 0.0:
        parser.error("--atol and --rtol must be >= 0")
    return args


def resolver_descriptor_for_contract(
    *,
    resolver_kind: PageResolverKind,
    batch: int,
    num_heads: int,
) -> ResolverGraphDescriptor:
    return ResolverGraphDescriptor(
        resolver_kind=resolver_kind,
        batch=int(batch),
        num_heads=int(num_heads),
        page_block_size=16,
        max_pages_per_row=8,
        max_selected_pages_per_row=8,
        max_capture_rows=int(batch),
        q_layout_key="decode:last-token",
        kv_cache_addr=0,
    )


def record_to_jsonable(record: LogitCaptureCheckRecord) -> dict[str, object]:
    return asdict(record)


def write_records(output: Path, records: list[LogitCaptureCheckRecord]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record_to_jsonable(record), allow_nan=False, sort_keys=True))
            handle.write("\n")


def make_capture_buffer(
    rows: int,
    q_heads: int,
    last_n: int,
    logical_k: int,
    *,
    sentinel_value: float = SENTINEL_VALUE,
) -> torch.Tensor:
    return torch.full(
        (int(rows), int(q_heads), int(last_n), int(logical_k)),
        float(sentinel_value),
        device="cuda",
        dtype=torch.float16,
    )


def capture_written_mask(capture: torch.Tensor, valid_k_by_row: list[int]) -> torch.Tensor:
    mask = torch.zeros(capture.shape, device=capture.device, dtype=torch.bool)
    for row, valid_k in enumerate(valid_k_by_row):
        if row >= int(capture.shape[0]):
            break
        valid_k_i = min(int(valid_k), int(capture.shape[-1]))
        if valid_k_i > 0:
            mask[row : row + 1, :, :, :valid_k_i] = True
    return mask


def stale_sentinel_unchanged(capture: torch.Tensor, written_mask: torch.Tensor) -> bool:
    stale = capture[~written_mask]
    if stale.numel() == 0:
        return True
    sentinel = torch.tensor(SENTINEL_VALUE, device=capture.device, dtype=capture.dtype)
    # Scores are captured after the attention mask is applied. Mixed selected/full
    # rows can legitimately touch masked lanes with -inf; finite non-sentinel
    # values are still stale-tail pollution.
    return bool(torch.all((stale == sentinel) | torch.isneginf(stale)).item())


def expected_case_names() -> set[str]:
    return {case.name for case in CASE_SPECS}


def exit_code_for_records(records: list[LogitCaptureCheckRecord]) -> int:
    actual_cases = {record.case for record in records}
    if actual_cases != expected_case_names():
        return GATE_FAILURE_EXIT_CODE
    if len(records) != len(actual_cases):
        return GATE_FAILURE_EXIT_CODE
    return 0 if all(
        record.allclose_passed
        and record.stale_sentinel_unchanged
        and record.split_invariant_bitwise is not False
        for record in records
    ) else GATE_FAILURE_EXIT_CODE


def _compiled_extension_available() -> bool:
    package_dir = UPSTREAM_FA3_ROOT / "vllm_flash_attn"
    return any(any(package_dir.glob(pattern)) for pattern in FA3_EXTENSION_GLOBS)


def _load_worktree_flash_attn_interface():
    import vllm_flash_attn.flash_attn_interface as interface

    return interface


def _unavailable_records(reason: str) -> list[LogitCaptureCheckRecord]:
    del reason
    records: list[LogitCaptureCheckRecord] = []
    for case in CASE_SPECS:
        records.append(
            LogitCaptureCheckRecord(
                case=case.name,
                resolver_kind=case.resolver_kind,
                row_modes=[int(mode) for mode in case.row_modes],
                max_abs_diff=1.0,
                allclose_passed=False,
                capture_rows=case.capture_rows,
                graph_replay=False,
                capture_allclose_when_enabled=False,
                capture_dtype="torch.float16",
                tma_or_non_tma=case.tma_or_non_tma,
                stale_sentinel_unchanged=False,
            )
        )
    return records


def _dense_capture_reference(
    *,
    q: torch.Tensor,
    key_cache: torch.Tensor,
    block_table: torch.Tensor,
    seqused_k: torch.Tensor,
    capacity: int,
    softmax_scale: float,
) -> torch.Tensor:
    reference = torch.full(
        (int(seqused_k.numel()), q.shape[1], 1, int(capacity)),
        float("-inf"),
        device=q.device,
        dtype=torch.float32,
    )
    num_seqs = int(seqused_k.numel())
    num_query_heads = int(q.shape[1])
    num_kv_heads = int(key_cache.shape[2])
    block_size = int(key_cache.shape[1])
    group = num_query_heads // num_kv_heads if num_kv_heads > 0 else 1
    for row in range(num_seqs):
        valid_k = min(int(seqused_k[row].item()), int(capacity))
        if valid_k <= 0:
            continue
        num_blocks = (valid_k + block_size - 1) // block_size
        k_pieces = []
        for block in range(num_blocks):
            blk = int(block_table[row, block].item())
            start = block * block_size
            take = min(block_size, valid_k - start)
            k_pieces.append(key_cache[blk, :take])
        k_row = torch.cat(k_pieces, dim=0).to(torch.float32)
        kv_for_q = k_row.repeat_interleave(group, dim=1)
        q_row = q[row].to(torch.float32)
        logits = torch.einsum("hd,vhd->hv", q_row, kv_for_q) * float(softmax_scale)
        reference[row, :, 0, :valid_k].copy_(logits)
    return reference


def _resolved_row_ptr_table(page_rows: list[torch.Tensor], *, num_kv_heads: int) -> torch.Tensor:
    values: list[int] = []
    for row_tensor in page_rows:
        for _ in range(int(num_kv_heads)):
            values.append(int(row_tensor.data_ptr()))
    return torch.tensor(values, device=page_rows[0].device, dtype=torch.int64)


def _make_paged_cache(
    k_full: torch.Tensor,
    v_full: torch.Tensor,
    kv_lengths: list[int],
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, num_heads, max_k_len, head_size = k_full.shape
    num_blocks = (max_k_len + block_size - 1) // block_size if max_k_len > 0 else 0
    total_blocks = batch * num_blocks
    device = k_full.device

    k_cache = torch.zeros(
        (total_blocks, block_size, num_heads, head_size),
        dtype=k_full.dtype,
        device=device,
    )
    v_cache = torch.zeros_like(k_cache)
    block_table = torch.full(
        (batch, num_blocks),
        -1,
        dtype=torch.int32,
        device=device,
    )

    for row in range(batch):
        for block in range(num_blocks):
            block_idx = row * num_blocks + block
            block_table[row, block] = block_idx
            start = block * block_size
            end = min(start + block_size, max_k_len)
            length = max(0, min(end, int(kv_lengths[row])) - start)
            if length <= 0:
                continue
            k_slice = k_full[row, :, start : start + length, :].permute(1, 0, 2)
            v_slice = v_full[row, :, start : start + length, :].permute(1, 0, 2)
            k_cache[block_idx, :length].copy_(k_slice)
            v_cache[block_idx, :length].copy_(v_slice)
    return k_cache, v_cache, block_table


def _prepare_cuda_inputs(
    seed: int,
    *,
    q_lens: list[int] | None = None,
    kv_lens: list[int] | None = None,
    block_size: int = 16,
):
    q_lens = [1] if q_lens is None else list(q_lens)
    kv_lens = [192] if kv_lens is None else list(kv_lens)
    if len(q_lens) != len(kv_lens):
        raise ValueError("q_lens and kv_lens must have the same length")

    torch.manual_seed(int(seed))
    batch = len(q_lens)
    num_heads = 32
    head_size = 128
    dtype = torch.bfloat16
    device = "cuda"
    max_q_len = max(q_lens) if q_lens else 0
    max_k_len = max(kv_lens) if kv_lens else 0

    q_pad = torch.zeros(
        (batch, num_heads, max_q_len, head_size),
        dtype=dtype,
        device=device,
    )
    k_full = torch.zeros(
        (batch, num_heads, max_k_len, head_size),
        dtype=dtype,
        device=device,
    )
    v_full = torch.zeros_like(k_full)

    for row in range(batch):
        q_len = int(q_lens[row])
        if q_len > 0:
            q_pad[row, :, :q_len, :] = torch.randn(
                num_heads,
                q_len,
                head_size,
                dtype=dtype,
                device=device,
            )
        kv_len = int(kv_lens[row])
        if kv_len > 0:
            k_full[row, :, :kv_len, :] = torch.randn(
                num_heads,
                kv_len,
                head_size,
                dtype=dtype,
                device=device,
            )
            v_full[row, :, :kv_len, :] = torch.randn(
                num_heads,
                kv_len,
                head_size,
                dtype=dtype,
                device=device,
            )

    k_cache, v_cache, block_table = _make_paged_cache(
        k_full,
        v_full,
        kv_lens,
        block_size,
    )
    q_bshd = q_pad.permute(0, 2, 1, 3).contiguous()
    q = q_bshd.reshape(-1, q_bshd.shape[2], q_bshd.shape[3]).contiguous()
    seqused_k = torch.tensor(kv_lens, dtype=torch.int32, device=device)
    cu_seqlens_actual = torch.tensor(
        [0, *accumulate(q_lens)],
        dtype=torch.int32,
        device=device,
    )
    inputs = {
        "q_pad": q_pad,
        "k_full": k_full,
        "v_full": v_full,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "block_table": block_table,
        "cu_seqlens_actual": cu_seqlens_actual,
        "seqused_k": seqused_k,
        "max_seqlen_k": block_table.shape[1] * block_size,
    }
    kv_len = int(seqused_k.max().item())
    softmax_scale = 1.0 / math.sqrt(float(head_size))
    capture_scores = make_capture_buffer(batch, num_heads, 1, kv_len)

    return inputs, q_bshd, q, kv_len, softmax_scale, capture_scores


def _max_abs_diff(actual: torch.Tensor, expected: torch.Tensor, valid_k: int) -> float:
    diff = actual[:, :, 0, :valid_k].to(torch.float32) - expected[:, :, 0, :valid_k].to(torch.float32)
    return float(diff.abs().max().item())


def _max_abs_diff_by_row(
    actual: torch.Tensor,
    expected: torch.Tensor,
    valid_k_by_row: list[int],
) -> float:
    max_diff = torch.zeros((), device=actual.device, dtype=torch.float32)
    for row, valid_k in enumerate(valid_k_by_row):
        valid_k_i = int(valid_k)
        if valid_k_i <= 0:
            continue
        row_diff = (
            actual[row : row + 1, :, :, :valid_k_i].to(torch.float32)
            - expected[row : row + 1, :, :, :valid_k_i].to(torch.float32)
        )
        max_diff = torch.maximum(max_diff, row_diff.abs().max())
    return float(max_diff.item())


def _allclose_by_row(
    actual: torch.Tensor,
    expected: torch.Tensor,
    valid_k_by_row: list[int],
    *,
    atol: float,
    rtol: float,
) -> bool:
    for row, valid_k in enumerate(valid_k_by_row):
        valid_k_i = int(valid_k)
        if valid_k_i <= 0:
            continue
        if not torch.allclose(
            actual[row : row + 1, :, :, :valid_k_i],
            expected[row : row + 1, :, :, :valid_k_i].to(actual.dtype),
            atol=float(atol),
            rtol=float(rtol),
        ):
            return False
    return True


# [2026-07-11] The SM8x TU retired the DirectTable subkind with the instance diet;
# on that arch the fail-closed rejection IS the case contract (mirrors the
# kind4_cp_capture_fail_closed encoding). capture_allclose_when_enabled=None marks
# honestly that no numeric compare ran on this arch; SM90 builds still take the
# numeric path.
_RETIRED_SUBKIND_MARKER = "retired from the SM8x TU"


def _retired_subkind_record(case: CaseSpec) -> LogitCaptureCheckRecord:
    return LogitCaptureCheckRecord(
        case=case.name,
        resolver_kind=case.resolver_kind,
        row_modes=[int(mode) for mode in case.row_modes],
        max_abs_diff=0.0,
        allclose_passed=True,
        capture_rows=case.capture_rows,
        graph_replay=case.graph_replay,
        capture_allclose_when_enabled=None,
        capture_dtype="torch.float16",
        tma_or_non_tma=case.tma_or_non_tma,
        stale_sentinel_unchanged=True,
    )


def _record_from_capture(
    *,
    case: CaseSpec,
    actual: torch.Tensor,
    expected: torch.Tensor,
    valid_k: int,
    atol: float,
    rtol: float,
) -> LogitCaptureCheckRecord:
    max_abs_diff = _max_abs_diff(actual, expected, valid_k)
    allclose_passed = bool(
        torch.allclose(
            actual[:, :, 0, :valid_k],
            expected[:, :, 0, :valid_k].to(actual.dtype),
            atol=float(atol),
            rtol=float(rtol),
        )
    )
    return LogitCaptureCheckRecord(
        case=case.name,
        resolver_kind=case.resolver_kind,
        row_modes=[int(mode) for mode in case.row_modes],
        max_abs_diff=max_abs_diff,
        allclose_passed=allclose_passed,
        capture_rows=case.capture_rows,
        graph_replay=case.graph_replay,
        capture_allclose_when_enabled=allclose_passed,
        capture_dtype=str(actual.dtype),
        tma_or_non_tma=case.tma_or_non_tma,
        stale_sentinel_unchanged=stale_sentinel_unchanged(
            actual,
            capture_written_mask(actual, [int(valid_k) for _ in range(int(actual.shape[0]))]),
        ),
    )


def _record_from_capture_by_row(
    *,
    case: CaseSpec,
    actual: torch.Tensor,
    expected: torch.Tensor,
    valid_k_by_row: list[int],
    atol: float,
    rtol: float,
) -> LogitCaptureCheckRecord:
    allclose_passed = _allclose_by_row(
        actual,
        expected,
        valid_k_by_row,
        atol=atol,
        rtol=rtol,
    )
    return LogitCaptureCheckRecord(
        case=case.name,
        resolver_kind=case.resolver_kind,
        row_modes=[int(mode) for mode in case.row_modes],
        max_abs_diff=_max_abs_diff_by_row(actual, expected, valid_k_by_row),
        allclose_passed=allclose_passed,
        capture_rows=case.capture_rows,
        graph_replay=case.graph_replay,
        capture_allclose_when_enabled=allclose_passed,
        capture_dtype=str(actual.dtype),
        tma_or_non_tma=case.tma_or_non_tma,
        stale_sentinel_unchanged=stale_sentinel_unchanged(
            actual,
            capture_written_mask(actual, valid_k_by_row),
        ),
    )


def run_checks(*, warmup: int, iters: int, atol: float, rtol: float) -> list[LogitCaptureCheckRecord]:
    del warmup, iters
    if not torch.cuda.is_available():
        return _unavailable_records("CUDA is not available")
    if not _compiled_extension_available():
        return _unavailable_records("compiled native FA3 extension is not available")

    try:
        mod = _load_worktree_flash_attn_interface()
    except ImportError as exc:
        return _unavailable_records(f"native FA3 import failed: {exc}")
    records: list[LogitCaptureCheckRecord] = []
    softmax_scale = 1.0 / math.sqrt(128.0)

    try:
        inputs, q_bshd, q, _, _, _ = _prepare_cuda_inputs(seed=0)
    except ImportError as exc:
        return _unavailable_records(f"reference helper import failed: {exc}")
    selected_page_table = inputs["block_table"][:, :4].contiguous()
    selected_kv_len = int(selected_page_table.shape[1]) * 16
    selected_seqused = torch.tensor([selected_kv_len], device="cuda", dtype=torch.int32)
    selected_seqused_by_head = selected_seqused.repeat_interleave(
        int(inputs["k_cache"].shape[2])
    )
    selected_row_ptr = _resolved_row_ptr_table(
        [selected_page_table[0]],
        num_kv_heads=int(inputs["k_cache"].shape[2]),
    )
    selected_ref = _dense_capture_reference(
        q=q_bshd[:, 0].contiguous(),
        key_cache=inputs["k_cache"],
        block_table=selected_page_table,
        seqused_k=selected_seqused,
        capacity=selected_kv_len,
        softmax_scale=softmax_scale,
    )

    dense_capacity = int(inputs["seqused_k"].max().item())
    dense_ref = _dense_capture_reference(
        q=q_bshd[:, 0].contiguous(),
        key_cache=inputs["k_cache"],
        block_table=inputs["block_table"],
        seqused_k=inputs["seqused_k"],
        capacity=dense_capacity,
        softmax_scale=softmax_scale,
    )

    native_capture = make_capture_buffer(1, 32, 1, dense_capacity)
    mod.mixed_page_attn_varlen_func(
        q=q,
        k=inputs["k_cache"],
        v=inputs["v_cache"],
        max_seqlen_q=1,
        cu_seqlens_q=inputs["cu_seqlens_actual"],
        max_seqlen_k=int(inputs["max_seqlen_k"]),
        seqused_k=inputs["seqused_k"],
        block_table=inputs["block_table"],
        softmax_scale=softmax_scale,
        causal=True,
        page_resolver_kind=int(PageResolverKind.NATIVE),
        capture_scores=native_capture,
        capture_row_index_i32=torch.tensor([0], device="cuda", dtype=torch.int32),
        row_capture_last_n_i32=torch.tensor([1], device="cuda", dtype=torch.int32),
    )
    records.append(
        _record_from_capture(
            case=CASE_BY_NAME["kind0_native_capture_non_tma"],
            actual=native_capture,
            expected=dense_ref,
            valid_k=dense_capacity,
            atol=atol,
            rtol=rtol,
        )
    )

    # [2026-07-11 CAP-4] bs2 x forced num_splits=4 vs num_splits=1: capture
    # writes are per-column raw scaled logits, so split slicing must reassemble a
    # BITWISE-identical buffer, and empty split slots (row1 kv384 -> 3 n-blocks < 4
    # splits) must never touch the buffer (sentinel door). kv 640/384 with
    # kBlockN=128 gives real multi-block split ranges on both rows.
    split_inputs, split_q_bshd, split_q, _, _, _ = _prepare_cuda_inputs(
        seed=0, q_lens=[1, 1], kv_lens=[640, 384]
    )
    split_valid_k_by_row = [640, 384]
    split_capacity = int(split_inputs["seqused_k"].max().item())
    split_ref = _dense_capture_reference(
        q=split_q_bshd[:, 0].contiguous(),
        key_cache=split_inputs["k_cache"],
        block_table=split_inputs["block_table"],
        seqused_k=split_inputs["seqused_k"],
        capacity=split_capacity,
        softmax_scale=softmax_scale,
    )
    split_captures: dict[int, torch.Tensor] = {}
    for forced_splits in (1, 4):
        split_capture_buf = make_capture_buffer(2, 32, 1, split_capacity)
        mod.mixed_page_attn_varlen_func(
            q=split_q,
            k=split_inputs["k_cache"],
            v=split_inputs["v_cache"],
            max_seqlen_q=1,
            cu_seqlens_q=split_inputs["cu_seqlens_actual"],
            max_seqlen_k=int(split_inputs["max_seqlen_k"]),
            seqused_k=split_inputs["seqused_k"],
            block_table=split_inputs["block_table"],
            softmax_scale=softmax_scale,
            causal=True,
            page_resolver_kind=int(PageResolverKind.NATIVE),
            num_splits=forced_splits,
            capture_scores=split_capture_buf,
            capture_row_index_i32=torch.tensor(
                [0, 1], device="cuda", dtype=torch.int32
            ),
            row_capture_last_n_i32=torch.tensor(
                [1, 1], device="cuda", dtype=torch.int32
            ),
        )
        torch.cuda.synchronize()
        split_captures[forced_splits] = split_capture_buf
    split_case_record = _record_from_capture_by_row(
        case=CASE_BY_NAME["kind0_native_split4_capture_non_tma"],
        actual=split_captures[4],
        expected=split_ref,
        valid_k_by_row=split_valid_k_by_row,
        atol=atol,
        rtol=rtol,
    )
    # Bitwise contract holds on the VALID region only: masked stale lanes may
    # legitimately hold -inf vs sentinel depending on split slicing (the sentinel
    # door above already polices them), measured 2026-07-11: 0 valid-region diffs,
    # -inf-vs-sentinel diffs confined to row1's masked tail.
    split_written_mask = capture_written_mask(split_captures[4], split_valid_k_by_row)
    records.append(
        LogitCaptureCheckRecord(
            **{
                **asdict(split_case_record),
                "split_invariant_bitwise": bool(
                    torch.equal(
                        split_captures[4][split_written_mask],
                        split_captures[1][split_written_mask],
                    )
                ),
            }
        )
    )

    # kind1 (SelectedTable) capture cases removed 2026-07-11: the device-side
    # SelectedTable resolver kind is retired at launch (fa3_t1_selectedtable_delete);
    # there is no production path and the kernel rejects it with TORCH_CHECK.

    selected_capture = make_capture_buffer(1, 32, 1, selected_kv_len)
    mod.mixed_page_attn_varlen_func(
        q=q,
        k=inputs["k_cache"],
        v=inputs["v_cache"],
        max_seqlen_q=1,
        cu_seqlens_q=inputs["cu_seqlens_actual"],
        max_seqlen_k=selected_kv_len,
        seqused_k=selected_seqused,
        block_table=inputs["block_table"],
        softmax_scale=softmax_scale,
        causal=True,
        page_resolver_kind=int(PageResolverKind.RESOLVED_ROW_PTR),
        resolved_page_table_row_ptr_u64=selected_row_ptr,
        resolved_seqused_k_by_head_i32=selected_seqused_by_head,
        capture_scores=selected_capture,
        capture_row_index_i32=torch.tensor([0], device="cuda", dtype=torch.int32),
        row_capture_last_n_i32=torch.tensor([1], device="cuda", dtype=torch.int32),
    )
    records.append(
        _record_from_capture(
            case=CASE_BY_NAME["kind4_rrp_capture_non_tma"],
            actual=selected_capture,
            expected=selected_ref,
            valid_k=selected_kv_len,
            atol=atol,
            rtol=rtol,
        )
    )

    direct_table_capture = make_capture_buffer(1, 32, 1, selected_kv_len)
    direct_table_retired = False
    try:
        mod.mixed_page_attn_varlen_func(
            q=q,
            k=inputs["k_cache"],
            v=inputs["v_cache"],
            max_seqlen_q=1,
            cu_seqlens_q=inputs["cu_seqlens_actual"],
            max_seqlen_k=selected_kv_len,
            seqused_k=selected_seqused,
            block_table=inputs["block_table"],
            softmax_scale=softmax_scale,
            causal=True,
            page_resolver_kind=int(PageResolverKind.RESOLVED_ROW_PTR),
            page_resolver_subkind=1,
            resolved_page_table_i32=selected_page_table,
            resolved_page_table_batch_stride=1,
            resolved_page_table_head_stride=0,
            resolved_seqused_k_by_head_i32=selected_seqused_by_head,
            capture_scores=direct_table_capture,
            capture_row_index_i32=torch.tensor([0], device="cuda", dtype=torch.int32),
            row_capture_last_n_i32=torch.tensor([1], device="cuda", dtype=torch.int32),
        )
    except RuntimeError as exc:
        if _RETIRED_SUBKIND_MARKER not in str(exc):
            raise
        direct_table_retired = True
    records.append(
        _retired_subkind_record(CASE_BY_NAME["kind4_direct_table_capture_non_tma"])
        if direct_table_retired
        else _record_from_capture(
            case=CASE_BY_NAME["kind4_direct_table_capture_non_tma"],
            actual=direct_table_capture,
            expected=selected_ref,
            valid_k=selected_kv_len,
            atol=atol,
            rtol=rtol,
        )
    )

    affine_capture = make_capture_buffer(1, 32, 1, selected_kv_len)
    affine_row = torch.tensor([[0, 1]], device="cuda", dtype=torch.int32)
    affine_retired = False
    try:
        mod.mixed_page_attn_varlen_func(
            q=q,
            k=inputs["k_cache"],
            v=inputs["v_cache"],
            max_seqlen_q=1,
            cu_seqlens_q=inputs["cu_seqlens_actual"],
            max_seqlen_k=selected_kv_len,
            seqused_k=selected_seqused,
            block_table=inputs["block_table"],
            softmax_scale=softmax_scale,
            causal=True,
            page_resolver_kind=int(PageResolverKind.RESOLVED_ROW_PTR),
            page_resolver_subkind=3,
            resolved_page_table_affine_i32=affine_row,
            resolved_page_table_affine_batch_stride=1,
            resolved_page_table_affine_head_stride=0,
            resolved_page_table_affine_cols=2,
            resolved_seqused_k_by_head_i32=selected_seqused_by_head,
            capture_scores=affine_capture,
            capture_row_index_i32=torch.tensor([0], device="cuda", dtype=torch.int32),
            row_capture_last_n_i32=torch.tensor([1], device="cuda", dtype=torch.int32),
        )
    except RuntimeError as exc:
        if _RETIRED_SUBKIND_MARKER not in str(exc):
            raise
        affine_retired = True
    records.append(
        _retired_subkind_record(CASE_BY_NAME["kind4_affine_tensor_capture_non_tma"])
        if affine_retired
        else _record_from_capture(
            case=CASE_BY_NAME["kind4_affine_tensor_capture_non_tma"],
            actual=affine_capture,
            expected=selected_ref,
            valid_k=selected_kv_len,
            atol=atol,
            rtol=rtol,
        )
    )

    cp_case = CASE_BY_NAME["kind4_cp_capture_fail_closed"]
    cp_fail_closed = False
    try:
        mod.mixed_page_attn_varlen_func(
            q=q,
            k=inputs["k_cache"],
            v=inputs["v_cache"],
            max_seqlen_q=1,
            cu_seqlens_q=inputs["cu_seqlens_actual"],
            max_seqlen_k=selected_kv_len,
            seqused_k=selected_seqused,
            block_table=inputs["block_table"],
            softmax_scale=softmax_scale,
            causal=True,
            page_resolver_kind=int(PageResolverKind.RESOLVED_ROW_PTR),
            resolved_page_table_row_ptr_u64=selected_row_ptr,
            resolved_seqused_k_by_head_i32=selected_seqused_by_head,
            cp_world_size=2,
            capture_scores=make_capture_buffer(1, 32, 1, selected_kv_len),
            capture_row_index_i32=torch.tensor([0], device="cuda", dtype=torch.int32),
            row_capture_last_n_i32=torch.tensor([1], device="cuda", dtype=torch.int32),
        )
    except RuntimeError as exc:
        cp_fail_closed = "cp_world_size" in str(exc) or "CP capture" in str(exc)
    records.append(
        LogitCaptureCheckRecord(
            case=cp_case.name,
            resolver_kind=cp_case.resolver_kind,
            row_modes=[int(mode) for mode in cp_case.row_modes],
            max_abs_diff=0.0 if cp_fail_closed else 1.0,
            allclose_passed=cp_fail_closed,
            capture_rows=cp_case.capture_rows,
            graph_replay=cp_case.graph_replay,
            capture_allclose_when_enabled=cp_fail_closed,
            capture_dtype="torch.float16",
            tma_or_non_tma=cp_case.tma_or_non_tma,
            stale_sentinel_unchanged=True,
        )
    )

    tma_inputs, tma_q_bshd, tma_q, _, _, _ = _prepare_cuda_inputs(
        seed=2,
        q_lens=[64],
        kv_lens=[2048],
        block_size=128,
    )
    tma_selected_page_table = tma_inputs["block_table"][:, :4].contiguous()
    tma_selected_kv_len = int(tma_selected_page_table.shape[1]) * 128
    tma_selected_seqused = torch.tensor([tma_selected_kv_len], device="cuda", dtype=torch.int32)
    tma_selected_seqused_by_head = tma_selected_seqused.repeat_interleave(
        int(tma_inputs["k_cache"].shape[2])
    )
    tma_selected_row_ptr = _resolved_row_ptr_table(
        [tma_selected_page_table[0]],
        num_kv_heads=int(tma_inputs["k_cache"].shape[2]),
    )
    tma_selected_ref = _dense_capture_reference(
        q=tma_q_bshd[:, -1].contiguous(),
        key_cache=tma_inputs["k_cache"],
        block_table=tma_selected_page_table,
        seqused_k=tma_selected_seqused,
        capacity=tma_selected_kv_len,
        softmax_scale=softmax_scale,
    )

    tma_native_capture = make_capture_buffer(1, 32, 1, tma_selected_kv_len)
    mod.mixed_page_attn_varlen_func(
        q=tma_q,
        k=tma_inputs["k_cache"],
        v=tma_inputs["v_cache"],
        max_seqlen_q=int(tma_q_bshd.shape[1]),
        cu_seqlens_q=tma_inputs["cu_seqlens_actual"],
        max_seqlen_k=tma_selected_kv_len,
        seqused_k=tma_selected_seqused,
        block_table=tma_selected_page_table,
        softmax_scale=softmax_scale,
        causal=True,
        page_resolver_kind=int(PageResolverKind.NATIVE),
        capture_scores=tma_native_capture,
        capture_row_index_i32=torch.tensor([0], device="cuda", dtype=torch.int32),
        row_capture_last_n_i32=torch.tensor([1], device="cuda", dtype=torch.int32),
    )
    records.append(
        _record_from_capture(
            case=CASE_BY_NAME["kind0_native_capture_tma"],
            actual=tma_native_capture,
            expected=tma_selected_ref,
            valid_k=tma_selected_kv_len,
            atol=atol,
            rtol=rtol,
        )
    )

    # kind1 (SelectedTable) tma capture case removed 2026-07-11 (same retirement).

    tma_rrp_capture = make_capture_buffer(1, 32, 1, tma_selected_kv_len)
    mod.mixed_page_attn_varlen_func(
        q=tma_q,
        k=tma_inputs["k_cache"],
        v=tma_inputs["v_cache"],
        max_seqlen_q=int(tma_q_bshd.shape[1]),
        cu_seqlens_q=tma_inputs["cu_seqlens_actual"],
        max_seqlen_k=tma_selected_kv_len,
        seqused_k=tma_selected_seqused,
        block_table=tma_selected_page_table,
        softmax_scale=softmax_scale,
        causal=True,
        page_resolver_kind=int(PageResolverKind.RESOLVED_ROW_PTR),
        resolved_page_table_row_ptr_u64=tma_selected_row_ptr,
        resolved_seqused_k_by_head_i32=tma_selected_seqused_by_head,
        capture_scores=tma_rrp_capture,
        capture_row_index_i32=torch.tensor([0], device="cuda", dtype=torch.int32),
        row_capture_last_n_i32=torch.tensor([1], device="cuda", dtype=torch.int32),
    )
    records.append(
        _record_from_capture(
            case=CASE_BY_NAME["kind4_rrp_capture_tma"],
            actual=tma_rrp_capture,
            expected=tma_selected_ref,
            valid_k=tma_selected_kv_len,
            atol=atol,
            rtol=rtol,
        )
    )

    tma_direct_table_capture = make_capture_buffer(1, 32, 1, tma_selected_kv_len)
    tma_direct_table_retired = False
    try:
        mod.mixed_page_attn_varlen_func(
            q=tma_q,
            k=tma_inputs["k_cache"],
            v=tma_inputs["v_cache"],
            max_seqlen_q=int(tma_q_bshd.shape[1]),
            cu_seqlens_q=tma_inputs["cu_seqlens_actual"],
            max_seqlen_k=tma_selected_kv_len,
            seqused_k=tma_selected_seqused,
            block_table=tma_selected_page_table,
            softmax_scale=softmax_scale,
            causal=True,
            page_resolver_kind=int(PageResolverKind.RESOLVED_ROW_PTR),
            page_resolver_subkind=1,
            resolved_page_table_i32=tma_selected_page_table,
            resolved_page_table_batch_stride=1,
            resolved_page_table_head_stride=0,
            resolved_seqused_k_by_head_i32=tma_selected_seqused_by_head,
            capture_scores=tma_direct_table_capture,
            capture_row_index_i32=torch.tensor([0], device="cuda", dtype=torch.int32),
            row_capture_last_n_i32=torch.tensor([1], device="cuda", dtype=torch.int32),
        )
    except RuntimeError as exc:
        if _RETIRED_SUBKIND_MARKER not in str(exc):
            raise
        tma_direct_table_retired = True
    records.append(
        _retired_subkind_record(CASE_BY_NAME["kind4_direct_table_capture_tma"])
        if tma_direct_table_retired
        else _record_from_capture(
            case=CASE_BY_NAME["kind4_direct_table_capture_tma"],
            actual=tma_direct_table_capture,
            expected=tma_selected_ref,
            valid_k=tma_selected_kv_len,
            atol=atol,
            rtol=rtol,
        )
    )

    tma_affine_capture = make_capture_buffer(1, 32, 1, tma_selected_kv_len)
    tma_affine_row = torch.tensor([[0, 1]], device="cuda", dtype=torch.int32)
    tma_affine_retired = False
    try:
        mod.mixed_page_attn_varlen_func(
            q=tma_q,
            k=tma_inputs["k_cache"],
            v=tma_inputs["v_cache"],
            max_seqlen_q=int(tma_q_bshd.shape[1]),
            cu_seqlens_q=tma_inputs["cu_seqlens_actual"],
            max_seqlen_k=tma_selected_kv_len,
            seqused_k=tma_selected_seqused,
            block_table=tma_inputs["block_table"],
            softmax_scale=softmax_scale,
            causal=True,
            page_resolver_kind=int(PageResolverKind.RESOLVED_ROW_PTR),
            page_resolver_subkind=3,
            resolved_page_table_affine_i32=tma_affine_row,
            resolved_page_table_affine_batch_stride=1,
            resolved_page_table_affine_head_stride=0,
            resolved_page_table_affine_cols=2,
            resolved_seqused_k_by_head_i32=tma_selected_seqused_by_head,
            capture_scores=tma_affine_capture,
            capture_row_index_i32=torch.tensor([0], device="cuda", dtype=torch.int32),
            row_capture_last_n_i32=torch.tensor([1], device="cuda", dtype=torch.int32),
        )
    except RuntimeError as exc:
        if _RETIRED_SUBKIND_MARKER not in str(exc):
            raise
        tma_affine_retired = True
    records.append(
        _retired_subkind_record(CASE_BY_NAME["kind4_affine_tensor_capture_tma"])
        if tma_affine_retired
        else _record_from_capture(
            case=CASE_BY_NAME["kind4_affine_tensor_capture_tma"],
            actual=tma_affine_capture,
            expected=tma_selected_ref,
            valid_k=tma_selected_kv_len,
            atol=atol,
            rtol=rtol,
        )
    )

    mixed_inputs, mixed_q_bshd, mixed_q, _, _, _ = _prepare_cuda_inputs(
        seed=1,
        q_lens=[1, 1],
        kv_lens=[192, 160],
    )
    mixed_selected_rows = torch.tensor(
        [[0, 3, 8, 11], [0, 1, 2, 3]],
        device="cuda",
        dtype=torch.long,
    )
    mixed_selected_page_table = torch.stack(
        [
            mixed_inputs["block_table"][row].index_select(0, mixed_selected_rows[row])
            for row in range(2)
        ],
        dim=0,
    ).contiguous()
    mixed_selected_kv_len = int(mixed_selected_page_table.shape[1]) * 16
    mixed_full_kv_len = int(mixed_inputs["seqused_k"][1].item())
    mixed_max_k = max(mixed_selected_kv_len, mixed_full_kv_len)
    mixed_dense_capacity = int(mixed_inputs["seqused_k"].max().item())
    mixed_seqused = torch.tensor(
        [mixed_selected_kv_len, mixed_full_kv_len],
        device="cuda",
        dtype=torch.int32,
    )
    mixed_seqused_by_head = mixed_seqused.repeat_interleave(
        int(mixed_inputs["k_cache"].shape[2])
    )
    mixed_row_ptr = _resolved_row_ptr_table(
        [
            mixed_selected_page_table[0],
            mixed_inputs["block_table"][1],
        ],
        num_kv_heads=int(mixed_inputs["k_cache"].shape[2]),
    )
    mixed_selected_ref = _dense_capture_reference(
        q=mixed_q_bshd[:, 0].contiguous(),
        key_cache=mixed_inputs["k_cache"],
        block_table=mixed_selected_page_table,
        seqused_k=torch.tensor(
            [mixed_selected_kv_len, mixed_selected_kv_len],
            device="cuda",
            dtype=torch.int32,
        ),
        capacity=mixed_max_k,
        softmax_scale=softmax_scale,
    )
    mixed_dense_ref = _dense_capture_reference(
        q=mixed_q_bshd[:, 0].contiguous(),
        key_cache=mixed_inputs["k_cache"],
        block_table=mixed_inputs["block_table"],
        seqused_k=mixed_inputs["seqused_k"],
        capacity=mixed_dense_capacity,
        softmax_scale=softmax_scale,
    )
    mixed_ref = torch.full(
        (2, 32, 1, mixed_max_k),
        float("-inf"),
        device="cuda",
        dtype=torch.float32,
    )
    mixed_ref[0, :, :, :mixed_selected_kv_len].copy_(mixed_selected_ref[0, :, :, :mixed_selected_kv_len])
    mixed_ref[1, :, :, :mixed_full_kv_len].copy_(mixed_dense_ref[1, :, :, :mixed_full_kv_len])
    mixed_capture = make_capture_buffer(2, 32, 1, mixed_max_k)
    mod.mixed_page_attn_varlen_func(
        q=mixed_q,
        k=mixed_inputs["k_cache"],
        v=mixed_inputs["v_cache"],
        max_seqlen_q=1,
        cu_seqlens_q=mixed_inputs["cu_seqlens_actual"],
        max_seqlen_k=mixed_max_k,
        seqused_k=mixed_seqused,
        block_table=mixed_inputs["block_table"],
        softmax_scale=softmax_scale,
        causal=True,
        page_resolver_kind=int(PageResolverKind.RESOLVED_ROW_PTR),
        resolved_page_table_row_ptr_u64=mixed_row_ptr,
        resolved_seqused_k_by_head_i32=mixed_seqused_by_head,
        capture_scores=mixed_capture,
        capture_row_index_i32=torch.tensor([0, 1], device="cuda", dtype=torch.int32),
        row_capture_last_n_i32=torch.tensor([1, 1], device="cuda", dtype=torch.int32),
    )
    records.append(
        _record_from_capture_by_row(
            case=CASE_BY_NAME["kind4_mixed_capture_non_tma"],
            actual=mixed_capture,
            expected=mixed_ref,
            valid_k_by_row=[mixed_selected_kv_len, mixed_full_kv_len],
            atol=atol,
            rtol=rtol,
        )
    )

    return records


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    records = run_checks(
        warmup=int(args.warmup),
        iters=int(args.iters),
        atol=float(args.atol),
        rtol=float(args.rtol),
    )
    write_records(Path(str(args.output)), records)
    return exit_code_for_records(records)


if __name__ == "__main__":
    raise SystemExit(main())
