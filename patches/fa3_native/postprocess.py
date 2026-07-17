from __future__ import annotations

import hashlib
import math
import os
import struct
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import torch
from utils.selector_log_s_identity import (
    SELECTOR_LOG_F_COUNTER_SEMANTICS,
    SELECTOR_LOG_F_FAST_ROUTE,
    SELECTOR_LOG_F_GENERIC_ROUTE,
    SELECTOR_LOG_F_RETIRED_TP8_EXACT_ENV,
    SELECTOR_LOG_F_ROUTES,
    SELECTOR_LOG_F_TILED_ROUTE,
    SELECTOR_LOG_S_EXTENSION_NAME,
    SELECTOR_LOG_S_REQUIRED_TILED_SYMBOL,
    SELECTOR_LOG_S_RUNTIME_PROOF_SCHEMA,
    SELECTOR_LOG_S_SEMANTIC_IDENTITY,
    SELECTOR_LOG_S_SEMANTIC_IDENTITY_SYMBOL,
    selector_log_s_module_signature,
)


_LOG_F_R2_RESIDENT_BUCKETS = (12288, 16384, 24576, 32768)
_LOG_F_R2_RESIDENT_ROUTE = SELECTOR_LOG_F_FAST_ROUTE
_LOG_F_R2_TILED_ROUTE = SELECTOR_LOG_F_TILED_ROUTE
_LOG_F_GENERIC_ROUTE = SELECTOR_LOG_F_GENERIC_ROUTE
_LOG_F_RESIDENT_REASON = "resident_r2_bucket"
_LOG_F_TILED_REASON = "tiled_r2_large_k"
_LOG_F_GENERIC_ARCH_REASON = "generic_arch"
_LOG_F_GENERIC_DTYPE_REASON = "generic_dtype"
_LOG_F_GENERIC_LAYOUT_REASON = "generic_layout"
_LOG_F_GENERIC_ALPHA_REASON = "generic_alpha"
_LOG_F_GENERIC_ACCUM_REASON = "generic_accumulation"
_LOG_F_GENERIC_ROW_COUNT_REASON = "generic_row_count"
_LOG_F_GENERIC_K_RANGE_REASON = "generic_k_range"
_LOG_F_GENERIC_RESOURCE_REASON = "generic_resource_capacity"
_LOG_F_ADMISSION_CHAIN_ZERO = "0" * 64
_LOG_F_DIRECT_EVENT_SENTINEL = -1
_LOG_F_R2_TILED_MIN_K_EXCLUSIVE = 32_768
_LOG_F_META_I32_MIN = -(1 << 31)
_LOG_F_META_I32_MAX = (1 << 31) - 1
_LOG_F_WORKSPACE_I64_MAX_BYTES = (1 << 63) - 1
# CUDA devices supported by this release expose the standard one-dimensional
# grid-x limit.  The tiled owner uses only portable 256-thread CUDA kernels;
# unlike the resident owner, it has no exact-architecture instruction contract.
_LOG_F_PORTABLE_GRID_X_MAX = (1 << 31) - 1
_PROCESS_LOG_F_ROUTE_COUNTS = {route: 0 for route in SELECTOR_LOG_F_ROUTES}
_PROCESS_LOG_F_LAST_ROUTE = "none"
_PROCESS_LOG_F_LAST_DISPATCH_REASON = "none"
_PROCESS_LOG_F_LAST_ADMISSION_IDENTITY = "none"
_PROCESS_LOG_F_ADMISSION_EVENT_COUNT = 0
_PROCESS_LOG_F_ADMISSION_CHAIN_SHA256 = _LOG_F_ADMISSION_CHAIN_ZERO
_PROCESS_TILED_COHORT_COUNT = 0
_PROCESS_TILED_JOB_COUNT = 0
_PROCESS_TILED_DIRECT_COUNT = 0
_PROCESS_TILED_KERNEL_LAUNCH_COUNT = 0
_PROCESS_TILED_ADMISSION_FAILURE_COUNT = 0
_PROCESS_TILED_RESOURCE_CACHE_LOCK = threading.Lock()


class _ProcessMetaStagingOwner:
    pass


_PROCESS_META_STAGING_OWNER = _ProcessMetaStagingOwner()
_PROCESS_META_STAGING_LOCK = threading.Lock()


@dataclass(frozen=True, slots=True)
class _LogFTensorContract:
    """Pointer-free tensor facts expressible by the log_f kernel ABI."""

    device_type: str
    dtype: torch.dtype
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    is_contiguous: bool


@dataclass(frozen=True, slots=True)
class _LogFReduceAdmission:
    """Complete CPU-authored input to the pure route resolver."""

    meta_i32_rows: tuple[tuple[int, ...], ...]
    scratch_row_indices: tuple[int, ...]
    output_row_indices: tuple[int, ...]
    scratch: _LogFTensorContract
    output: _LogFTensorContract
    denom: _LogFTensorContract
    alpha: float
    capability: tuple[int, ...]
    cpu_authority_validated: bool
    local_same_device: bool
    tiled_resource_available: bool


@dataclass(frozen=True, slots=True)
class _LogFReduceDispatch:
    """One valid production owner; invalid contracts never produce this value."""

    route: str
    reason: str
    resident_k_bucket: Optional[int]
    admission_identity: str


@dataclass(frozen=True, slots=True)
class _TiledJobPlan:
    """Immutable CPU-authority view of one deferred large-K job."""

    job: Any
    meta_i32_rows: tuple[tuple[int, ...], ...]
    scratch_row_indices: tuple[int, ...]
    output_row_indices: tuple[int, ...]
    scratch: torch.Tensor
    output: torch.Tensor
    denom: torch.Tensor
    dispatch: _LogFReduceDispatch
    job_key: tuple[int, int, int]
    event_epoch: int
    num_query_heads: int
    logical_k_max: int
    capability: tuple[int, int]
    tiled_resource_available: bool


@dataclass(frozen=True, slots=True)
class _TouchedByteInterval:
    """One exact contiguous region touched by the tiled kernel contract."""

    device: torch.device
    begin: int
    end: int
    label: str


@dataclass(frozen=True, slots=True)
class TiledCapturePostprocessResourceReport:
    """Profile-time geometry and exact structural GPU resource report.

    ``slot_count`` is the proven owner-concurrency bound.  Every workspace and
    device-metadata buffer is resident before profile work and sealed live
    acquisition never grows that GPU footprint.  Pinned H2D source carriers
    have a separate lightweight lifetime pool and are not GPU resource slots.
    """

    num_rows_capacity: int
    num_query_heads_capacity: int
    logical_k_capacity: int
    slot_count: int
    workspace_bytes_per_slot: int
    device_meta_bytes_per_slot: int
    host_meta_bytes_per_slot: int
    device_bytes_per_slot: int
    total_workspace_bytes: int
    total_device_meta_bytes: int
    total_host_meta_bytes: int
    total_device_bytes: int

    @property
    def structural_slot_count(self) -> int:
        """Explicit name for the exact owner-concurrency ``slot_count``."""

        return int(self.slot_count)


@dataclass(frozen=True, slots=True)
class _PreparedTiledResourceSnapshot:
    """Immutable call-boundary proof for one prepared device/stream pool."""

    device: torch.device
    stream: torch.cuda.Stream
    key: Optional[tuple[int, int, int, int, int]]
    report: Optional[TiledCapturePostprocessResourceReport]


def _tensor_contract(tensor: torch.Tensor) -> _LogFTensorContract:
    return _LogFTensorContract(
        device_type=str(tensor.device.type),
        dtype=tensor.dtype,
        shape=tuple(int(value) for value in tensor.shape),
        stride=tuple(int(value) for value in tensor.stride()),
        is_contiguous=bool(tensor.is_contiguous()),
    )


def _selector_log_f_contract_error(code: str, detail: str) -> None:
    raise RuntimeError(f"E_SELECTOR_LOG_F_{code}: {detail}")


def _digest_text(digest: Any, value: str) -> None:
    encoded = str(value).encode("utf-8")
    digest.update(struct.pack("<I", len(encoded)))
    digest.update(encoded)


def _digest_i64_sequence(digest: Any, values: Sequence[int]) -> None:
    digest.update(struct.pack("<I", len(values)))
    for value in values:
        digest.update(struct.pack("<q", int(value)))


def _rectangular_rows_overlap(
    *,
    row_indices: Sequence[int],
    seq_stride: int,
    head_stride: int,
    row_stride: int,
    head_count: int,
    row_counts: Sequence[int],
    token_counts: Sequence[int],
) -> bool:
    """Exact overlap test for ABI rectangles with unit token stride."""

    for left in range(len(row_indices)):
        for right in range(left + 1, len(row_indices)):
            if int(row_indices[left]) == int(row_indices[right]):
                return True
            seq_delta = int(row_indices[right]) - int(row_indices[left])
            base = seq_delta * int(seq_stride)
            token_lower = -(int(token_counts[right]) - 1)
            token_upper = int(token_counts[left]) - 1
            row_delta_min = -(int(row_counts[left]) - 1)
            row_delta_max = int(row_counts[right]) - 1
            for head_delta in range(-(int(head_count) - 1), int(head_count)):
                head_base = base + head_delta * int(head_stride)
                first_row_delta = max(
                    row_delta_min,
                    -((-(token_lower - head_base)) // int(row_stride)),
                )
                last_row_delta = min(
                    row_delta_max,
                    (token_upper - head_base) // int(row_stride),
                )
                if first_row_delta <= last_row_delta:
                    return True
    return False


def _log_f_admission_identity(
    admission: _LogFReduceAdmission,
    *,
    route: str,
    reason: str,
    resident_k_bucket: Optional[int],
) -> str:
    """Hash the small canonical contract without pointers or tensor reprs."""

    digest = hashlib.sha256()
    digest.update(b"sfi.selector_log_f.admission.v3\0")
    _digest_text(digest, SELECTOR_LOG_S_EXTENSION_NAME)
    _digest_text(digest, route)
    _digest_text(digest, reason)
    digest.update(
        struct.pack(
            "<qd???",
            -1 if resident_k_bucket is None else int(resident_k_bucket),
            float(admission.alpha),
            bool(admission.cpu_authority_validated),
            bool(admission.local_same_device),
            bool(admission.tiled_resource_available),
        )
    )
    _digest_i64_sequence(digest, admission.capability)
    _digest_i64_sequence(digest, admission.scratch_row_indices)
    _digest_i64_sequence(digest, admission.output_row_indices)
    digest.update(struct.pack("<I", len(admission.meta_i32_rows)))
    for row in admission.meta_i32_rows:
        _digest_i64_sequence(digest, row)
    for name, layout in (
        ("scratch", admission.scratch),
        ("output", admission.output),
        ("denom", admission.denom),
    ):
        _digest_text(digest, name)
        _digest_text(digest, layout.device_type)
        _digest_text(digest, str(layout.dtype))
        _digest_i64_sequence(digest, layout.shape)
        _digest_i64_sequence(digest, layout.stride)
        digest.update(b"\x01" if layout.is_contiguous else b"\x00")
    return digest.hexdigest()


def _log_f_r2_resident_bucket_for_meta(
    meta_i32_rows: Sequence[Sequence[int]],
) -> Optional[int]:
    """Choose a loop ceiling without treating bucket padding as logical K."""
    if not meta_i32_rows:
        return None
    logical_k_max = 0
    for raw_row in meta_i32_rows:
        if len(raw_row) < 10:
            return None
        row = tuple(int(value) for value in raw_row)
        logical_k = row[0]
        scratch_head_stride = row[1]
        out_head_stride = row[6]
        scratch_row_stride = row[7]
        if not (
            # The resident two-pass owner stores two fp32 row LSE values in
            # the first four half slots until pass two overwrites them.
            logical_k >= 4
            and row[2] == 2
            and row[3] == 0
            and row[4] == logical_k
            and row[5] == 8
            and scratch_row_stride >= logical_k
            and scratch_head_stride == 2 * scratch_row_stride
            and out_head_stride >= logical_k
            and row[8] == 0
            and row[9] == 0
        ):
            return None
        logical_k_max = max(logical_k_max, logical_k)
    return next(
        (
            bucket
            for bucket in _LOG_F_R2_RESIDENT_BUCKETS
            if logical_k_max <= bucket
        ),
        None,
    )


def _log_f_r2_tiled_resource_reasons(
    *,
    meta_i32_rows: Sequence[Sequence[int]],
    num_query_heads: int,
) -> tuple[str, ...]:
    """Return structural capacity failures for the portable tiled owner."""

    if not meta_i32_rows:
        return ("row_capacity",)
    num_rows = len(meta_i32_rows)
    num_heads = int(num_query_heads)
    logical_k_max = max(int(row[0]) for row in meta_i32_rows)
    n_capacity = _tiled_n_capacity(num_rows)
    k_capacity = _tiled_k_capacity(logical_k_max)

    from utils import selector_log_s_ext

    tile_k = int(selector_log_s_ext.LOG_F_R2_TILED_TILE_K)
    tile_capacity = k_capacity // tile_k
    reasons: list[str] = []
    if n_capacity > 65_535:
        reasons.append("row_capacity")
    if num_heads <= 0 or num_heads > 65_535:
        reasons.append("head_capacity")
    if k_capacity > _LOG_F_META_I32_MAX or tile_capacity > 65_535:
        reasons.append("k_capacity")

    workspace_words = n_capacity * num_heads * (7 * tile_capacity + 4)
    if workspace_words > _LOG_F_WORKSPACE_I64_MAX_BYTES // 4:
        reasons.append("workspace_capacity")

    max_blocks = num_rows * num_heads * 2 * tile_capacity
    if max_blocks > _LOG_F_PORTABLE_GRID_X_MAX:
        reasons.append("grid_x_capacity")
    return tuple(reasons)


def _resolve_log_f_reduce_dispatch(
    admission: _LogFReduceAdmission,
) -> _LogFReduceDispatch:
    """Resolve one total valid route; malformed production contracts fail closed."""

    if not bool(admission.cpu_authority_validated):
        _selector_log_f_contract_error(
            "CPU_AUTHORITY",
            "validated replicated CPU row truth is required",
        )
    rows = admission.meta_i32_rows
    if not rows:
        _selector_log_f_contract_error("META", "gt1 metadata must be non-empty")
    if not math.isfinite(float(admission.alpha)):
        _selector_log_f_contract_error("ALPHA", "alpha must be finite")
    if len(admission.capability) != 2 or any(
        isinstance(value, bool) or int(value) < 0 for value in admission.capability
    ):
        _selector_log_f_contract_error(
            "CAPABILITY",
            f"invalid CUDA capability {admission.capability!r}",
        )

    scratch = admission.scratch
    output = admission.output
    denom = admission.denom
    if not bool(admission.local_same_device):
        _selector_log_f_contract_error(
            "DEVICE",
            "scratch/output/denom must share one local CUDA device",
        )
    if {scratch.device_type, output.device_type, denom.device_type} != {"cuda"}:
        _selector_log_f_contract_error(
            "DEVICE",
            "scratch/output/denom must share a CUDA device type",
        )
    if len(scratch.shape) != 4 or len(scratch.stride) != 4:
        _selector_log_f_contract_error("SCRATCH_LAYOUT", "scratch must be rank-4")
    if len(output.shape) != 4 or len(output.stride) != 4:
        _selector_log_f_contract_error("OUTPUT_LAYOUT", "output must be rank-4")
    if len(denom.shape) != 2 or len(denom.stride) != 2:
        _selector_log_f_contract_error("DENOM_LAYOUT", "denom must be rank-2")
    if scratch.dtype not in {torch.float16, torch.float32}:
        _selector_log_f_contract_error(
            "SCRATCH_DTYPE",
            f"generic ABI supports only float16/float32, got {scratch.dtype}",
        )
    if output.dtype not in {torch.float16, torch.float32}:
        _selector_log_f_contract_error(
            "OUTPUT_DTYPE",
            f"generic ABI supports only float16/float32, got {output.dtype}",
        )
    if denom.dtype != torch.float32:
        _selector_log_f_contract_error(
            "DENOM_DTYPE",
            f"denom pointer ABI requires float32, got {denom.dtype}",
        )
    if scratch.stride[3] != 1 or output.stride[3] != 1 or denom.stride[1] != 1:
        _selector_log_f_contract_error(
            "TOKEN_STRIDE",
            "token and denominator head strides must be one",
        )
    if any(int(scratch.stride[index]) <= 0 for index in (0, 1, 2)):
        _selector_log_f_contract_error(
            "SCRATCH_LAYOUT",
            "scratch sequence, head, and row strides must be positive",
        )
    if any(int(output.stride[index]) <= 0 for index in (0, 1)):
        _selector_log_f_contract_error(
            "OUTPUT_LAYOUT",
            "output sequence and head strides must be positive",
        )
    if int(denom.stride[0]) <= 0:
        _selector_log_f_contract_error(
            "DENOM_LAYOUT",
            "denominator sequence stride must be positive",
        )
    if min(scratch.shape) <= 0 or min(output.shape) <= 0 or min(denom.shape) <= 0:
        _selector_log_f_contract_error("CAPACITY", "tensor shapes must be positive")
    if output.shape[2] != 1:
        _selector_log_f_contract_error("OUTPUT_WINDOW", "output window must equal one")
    if len(admission.scratch_row_indices) != len(rows) or len(
        admission.output_row_indices
    ) != len(rows):
        _selector_log_f_contract_error(
            "CPU_AUTHORITY_SHAPE",
            "scratch and output row indices must align with metadata rows",
        )
    if any(
        index < 0 or index >= scratch.shape[0]
        for index in admission.scratch_row_indices
    ):
        _selector_log_f_contract_error(
            "ROW_CAPACITY",
            "scratch row indices exceed tensor capacity",
        )
    if any(
        index < 0
        or index >= output.shape[0]
        or index >= denom.shape[0]
        for index in admission.output_row_indices
    ):
        _selector_log_f_contract_error(
            "ROW_CAPACITY",
            "output row indices exceed output or denominator capacity",
        )
    if len(set(admission.scratch_row_indices)) != len(
        admission.scratch_row_indices
    ):
        _selector_log_f_contract_error(
            "SCRATCH_SEQ_OVERLAP",
            "metadata rows must own distinct scratch rows",
        )
    if len(set(admission.output_row_indices)) != len(
        admission.output_row_indices
    ):
        _selector_log_f_contract_error(
            "OUTPUT_SEQ_OVERLAP",
            "metadata rows must own distinct output rows",
        )
    if scratch.shape[0] < len(rows) or output.shape[0] < len(rows) or denom.shape[0] < len(rows):
        _selector_log_f_contract_error(
            "ROW_CAPACITY",
            "scratch/output/denom rows must cover metadata rows",
        )
    if output.shape[1] < scratch.shape[1] or denom.shape[1] < scratch.shape[1]:
        _selector_log_f_contract_error(
            "HEAD_CAPACITY",
            "output/denom heads must cover scratch heads",
        )
    for row_index, row in enumerate(rows):
        if len(row) != 10:
            _selector_log_f_contract_error(
                "META_WIDTH",
                f"row {row_index} must have exactly 10 columns",
            )
        if any(
            int(value) < _LOG_F_META_I32_MIN
            or int(value) > _LOG_F_META_I32_MAX
            for value in row
        ):
            _selector_log_f_contract_error(
                "META_INT32",
                f"row {row_index} contains a value outside the int32 ABI",
            )
        logical_k = int(row[0])
        last_n = int(row[2])
        flags = int(row[5])
        if logical_k <= 0:
            _selector_log_f_contract_error(
                "LOGICAL_K",
                f"row {row_index} logical K must be positive",
            )
        if logical_k > scratch.shape[3] or logical_k > output.shape[3]:
            _selector_log_f_contract_error(
                "K_CAPACITY",
                f"row {row_index} logical K exceeds scratch/output capacity",
            )
        if last_n < 1 or last_n > 16 or last_n > scratch.shape[2]:
            _selector_log_f_contract_error(
                "ROW_COUNT",
                f"row {row_index} last_n is outside the kernel ABI",
            )
        if int(row[3]) != 0 or int(row[4]) != logical_k:
            _selector_log_f_contract_error(
                "META_RANGE",
                f"row {row_index} must cover canonical [0, logical_k)",
            )
        if flags not in {8, 24}:
            _selector_log_f_contract_error(
                "FLAGS",
                f"row {row_index} flags must be exactly 8 or 24",
            )
        if int(row[1]) != scratch.stride[1] or int(row[7]) != scratch.stride[2]:
            _selector_log_f_contract_error(
                "SCRATCH_STRIDE",
                f"row {row_index} scratch metadata disagrees with tensor layout",
            )
        if int(row[6]) != output.stride[1]:
            _selector_log_f_contract_error(
                "OUTPUT_STRIDE",
                f"row {row_index} output metadata disagrees with tensor layout",
            )
        scratch_head_extent = (last_n - 1) * scratch.stride[2] + logical_k
        if scratch.stride[2] < logical_k or scratch.stride[1] < scratch_head_extent:
            _selector_log_f_contract_error(
                "SCRATCH_HEAD_OVERLAP",
                f"row {row_index} scratch rows or heads overlap",
            )
        if output.stride[1] < logical_k:
            _selector_log_f_contract_error(
                "OUTPUT_HEAD_OVERLAP",
                f"row {row_index} output heads overlap",
            )
        accum_prev_rows = int(row[8])
        accum_prev_capacity = int(row[9])
        if flags == 8 and (accum_prev_rows != 0 or accum_prev_capacity != 0):
            _selector_log_f_contract_error(
                "ACCUM_META",
                f"row {row_index} non-accumulating fields must be zero",
            )
        if flags == 24 and (
            accum_prev_rows < 0
            or accum_prev_capacity < 0
            or (accum_prev_rows == 0 and accum_prev_capacity != 0)
            or (
                accum_prev_rows > 0
                and not 0 < accum_prev_capacity <= logical_k
            )
            or accum_prev_capacity > output.shape[3]
        ):
            _selector_log_f_contract_error(
                "ACCUM_META",
                f"row {row_index} accumulating fields are inconsistent",
            )

    row_counts = tuple(int(row[2]) for row in rows)
    token_counts = tuple(int(row[0]) for row in rows)
    if _rectangular_rows_overlap(
        row_indices=admission.scratch_row_indices,
        seq_stride=scratch.stride[0],
        head_stride=scratch.stride[1],
        row_stride=scratch.stride[2],
        head_count=scratch.shape[1],
        row_counts=row_counts,
        token_counts=token_counts,
    ):
        _selector_log_f_contract_error(
            "SCRATCH_SEQ_OVERLAP",
            "scratch sequence rectangles overlap",
        )
    unit_rows = (1,) * len(rows)
    if _rectangular_rows_overlap(
        row_indices=admission.output_row_indices,
        seq_stride=output.stride[0],
        head_stride=output.stride[1],
        row_stride=1,
        head_count=scratch.shape[1],
        row_counts=unit_rows,
        token_counts=token_counts,
    ):
        _selector_log_f_contract_error(
            "OUTPUT_SEQ_OVERLAP",
            "output sequence rectangles overlap",
        )
    if _rectangular_rows_overlap(
        row_indices=admission.output_row_indices,
        seq_stride=denom.stride[0],
        head_stride=denom.stride[1],
        row_stride=1,
        head_count=scratch.shape[1],
        row_counts=unit_rows,
        token_counts=unit_rows,
    ):
        _selector_log_f_contract_error(
            "DENOM_SEQ_OVERLAP",
            "denominator sequence rows overlap",
        )

    capability = tuple(int(value) for value in admission.capability)
    resident_k_bucket: Optional[int] = None
    if scratch.dtype != torch.float16 or output.dtype != torch.float16:
        route = _LOG_F_GENERIC_ROUTE
        reason = _LOG_F_GENERIC_DTYPE_REASON
    elif not (
        scratch.is_contiguous and output.is_contiguous and denom.is_contiguous
    ):
        route = _LOG_F_GENERIC_ROUTE
        reason = _LOG_F_GENERIC_LAYOUT_REASON
    elif float(admission.alpha) != 0.5:
        route = _LOG_F_GENERIC_ROUTE
        reason = _LOG_F_GENERIC_ALPHA_REASON
    elif any(
        int(row[2]) != 2
        and not (int(row[5]) == 24 and int(row[2]) == 1)
        for row in rows
    ):
        route = _LOG_F_GENERIC_ROUTE
        reason = _LOG_F_GENERIC_ROW_COUNT_REASON
    else:
        accumulating = any(int(row[5]) == 24 for row in rows)
        resident_candidate = None
        if not accumulating:
            resident_candidate = _log_f_r2_resident_bucket_for_meta(rows)
        all_large_k = all(
            int(row[0]) > _LOG_F_R2_TILED_MIN_K_EXCLUSIVE for row in rows
        )
        if accumulating and not all_large_k:
            route = _LOG_F_GENERIC_ROUTE
            reason = _LOG_F_GENERIC_ACCUM_REASON
        elif all_large_k:
            if type(admission.tiled_resource_available) is not bool:
                _selector_log_f_contract_error(
                    "RESOURCE_AUTHORITY",
                    "tiled_resource_available must be an immutable CPU boolean",
                )
            if not admission.tiled_resource_available:
                route = _LOG_F_GENERIC_ROUTE
                reason = _LOG_F_GENERIC_RESOURCE_REASON
            else:
                tiled_resource_reasons = _log_f_r2_tiled_resource_reasons(
                    meta_i32_rows=rows,
                    num_query_heads=int(scratch.shape[1]),
                )
                if tiled_resource_reasons:
                    route = _LOG_F_GENERIC_ROUTE
                    reason = _LOG_F_GENERIC_RESOURCE_REASON
                else:
                    route = _LOG_F_R2_TILED_ROUTE
                    reason = _LOG_F_TILED_REASON
        elif resident_candidate is not None and capability == (8, 0):
            resident_k_bucket = resident_candidate
            route = _LOG_F_R2_RESIDENT_ROUTE
            reason = _LOG_F_RESIDENT_REASON
        elif resident_candidate is not None:
            route = _LOG_F_GENERIC_ROUTE
            reason = _LOG_F_GENERIC_ARCH_REASON
        else:
            route = _LOG_F_GENERIC_ROUTE
            reason = _LOG_F_GENERIC_K_RANGE_REASON

    identity = _log_f_admission_identity(
        admission,
        route=route,
        reason=reason,
        resident_k_bucket=resident_k_bucket,
    )
    return _LogFReduceDispatch(
        route=route,
        reason=reason,
        resident_k_bucket=resident_k_bucket,
        admission_identity=identity,
    )


def _record_log_f_reduce_route(
    cache_owner: Optional[object],
    route: str,
    *,
    dispatch_reason: Optional[str] = None,
    admission_identity: str = "none",
    event_epoch: int = -1,
    event_layer: int = -1,
    event_handle_id: int = _LOG_F_DIRECT_EVENT_SENTINEL,
    event_handle_generation: int = _LOG_F_DIRECT_EVENT_SENTINEL,
) -> None:
    global _PROCESS_LOG_F_LAST_ROUTE
    global _PROCESS_LOG_F_LAST_DISPATCH_REASON
    global _PROCESS_LOG_F_LAST_ADMISSION_IDENTITY
    global _PROCESS_LOG_F_ADMISSION_EVENT_COUNT
    global _PROCESS_LOG_F_ADMISSION_CHAIN_SHA256

    route = str(route)
    if route not in SELECTOR_LOG_F_ROUTES:
        raise ValueError(f"unknown selector log_f route: {route!r}")
    if dispatch_reason is None:
        raise ValueError("selector log_f dispatch reason is required")
    dispatch_reason = str(dispatch_reason)
    if not dispatch_reason or dispatch_reason == "none":
        raise ValueError("selector log_f dispatch reason must be concrete")
    if admission_identity != "none" and (
        len(admission_identity) != 64
        or any(ch not in "0123456789abcdef" for ch in admission_identity)
    ):
        raise ValueError("selector admission identity must be a lowercase sha256")
    event_handle_id = int(event_handle_id)
    event_handle_generation = int(event_handle_generation)
    direct_event = (
        event_handle_id == _LOG_F_DIRECT_EVENT_SENTINEL
        and event_handle_generation == _LOG_F_DIRECT_EVENT_SENTINEL
    )
    if not direct_event and (
        event_handle_id < 0 or event_handle_generation < 0
    ):
        raise ValueError(
            "selector event handle and generation must both be non-negative, "
            "or both use the direct-event sentinel"
        )

    _PROCESS_LOG_F_ROUTE_COUNTS[route] = int(
        _PROCESS_LOG_F_ROUTE_COUNTS[route]
    ) + 1
    _PROCESS_LOG_F_LAST_ROUTE = route
    _PROCESS_LOG_F_LAST_DISPATCH_REASON = dispatch_reason
    _PROCESS_LOG_F_LAST_ADMISSION_IDENTITY = admission_identity
    event_digest = hashlib.sha256()
    event_digest.update(b"sfi.selector_log_f.admission_chain.v2\0")
    event_digest.update(bytes.fromhex(_PROCESS_LOG_F_ADMISSION_CHAIN_SHA256))
    event_digest.update(
        struct.pack(
            "<qqqq",
            int(event_epoch),
            int(event_layer),
            event_handle_id,
            event_handle_generation,
        )
    )
    _digest_text(event_digest, "direct" if direct_event else "job")
    _digest_text(event_digest, route)
    _digest_text(event_digest, dispatch_reason)
    _digest_text(event_digest, admission_identity)
    _PROCESS_LOG_F_ADMISSION_CHAIN_SHA256 = event_digest.hexdigest()
    _PROCESS_LOG_F_ADMISSION_EVENT_COUNT += 1
    if cache_owner is None:
        return
    counts = getattr(cache_owner, "_selector_log_f_reduce_route_counts", None)
    if not isinstance(counts, dict):
        counts = {}
        setattr(cache_owner, "_selector_log_f_reduce_route_counts", counts)
    counts[route] = int(counts.get(route, 0)) + 1
    setattr(cache_owner, "_selector_log_f_reduce_last_route", route)
    setattr(
        cache_owner,
        "_selector_log_f_reduce_last_dispatch_reason",
        dispatch_reason,
    )
    setattr(
        cache_owner,
        "_selector_log_f_reduce_last_admission_identity",
        admission_identity,
    )


def reset_selector_log_s_runtime_proof_counters() -> None:
    """Reset only process-local selector counters at the measurement barrier."""

    global _PROCESS_LOG_F_LAST_ROUTE
    global _PROCESS_LOG_F_LAST_DISPATCH_REASON
    global _PROCESS_LOG_F_LAST_ADMISSION_IDENTITY
    global _PROCESS_LOG_F_ADMISSION_EVENT_COUNT
    global _PROCESS_LOG_F_ADMISSION_CHAIN_SHA256
    global _PROCESS_TILED_COHORT_COUNT
    global _PROCESS_TILED_JOB_COUNT
    global _PROCESS_TILED_DIRECT_COUNT
    global _PROCESS_TILED_KERNEL_LAUNCH_COUNT
    global _PROCESS_TILED_ADMISSION_FAILURE_COUNT

    for route in SELECTOR_LOG_F_ROUTES:
        _PROCESS_LOG_F_ROUTE_COUNTS[route] = 0
    _PROCESS_LOG_F_LAST_ROUTE = "none"
    _PROCESS_LOG_F_LAST_DISPATCH_REASON = "none"
    _PROCESS_LOG_F_LAST_ADMISSION_IDENTITY = "none"
    _PROCESS_LOG_F_ADMISSION_EVENT_COUNT = 0
    _PROCESS_LOG_F_ADMISSION_CHAIN_SHA256 = _LOG_F_ADMISSION_CHAIN_ZERO
    _PROCESS_TILED_COHORT_COUNT = 0
    _PROCESS_TILED_JOB_COUNT = 0
    _PROCESS_TILED_DIRECT_COUNT = 0
    _PROCESS_TILED_KERNEL_LAUNCH_COUNT = 0
    _PROCESS_TILED_ADMISSION_FAILURE_COUNT = 0


def _reject_retired_selector_log_f_tp8_exact_env() -> None:
    raw = os.environ.get(SELECTOR_LOG_F_RETIRED_TP8_EXACT_ENV, "")
    if raw not in {"", "0"}:
        raise RuntimeError(
            "E_RETIRED_SELECTOR_LOG_F_TP8_EXACT_ENV: "
            f"unset {SELECTOR_LOG_F_RETIRED_TP8_EXACT_ENV}; the dynamic tiled "
            "route is selected from the validated request contract"
        )


def snapshot_selector_log_s_runtime_proof() -> dict[str, object]:
    """Bind capture-route evidence to the already-loaded worker-local .so."""
    _reject_retired_selector_log_f_tp8_exact_env()
    from utils import selector_log_s_ext

    module = selector_log_s_ext._MODULE
    extension_name = SELECTOR_LOG_S_EXTENSION_NAME
    required_symbol = SELECTOR_LOG_S_REQUIRED_TILED_SYMBOL
    if module is None:
        raise RuntimeError(
            "E_SELECTOR_LOG_S_PROOF_MODULE_NOT_LOADED: warmup did not load "
            f"{extension_name}"
        )
    symbol = getattr(module, required_symbol, None)
    if not callable(symbol):
        raise RuntimeError(
            "E_SELECTOR_LOG_S_PROOF_SYMBOL_MISSING: "
            f"{required_symbol}"
        )
    semantic_identity_fn = getattr(
        module, SELECTOR_LOG_S_SEMANTIC_IDENTITY_SYMBOL, None
    )
    if not callable(semantic_identity_fn):
        raise RuntimeError(
            "E_SELECTOR_LOG_S_PROOF_SEMANTIC_SYMBOL_MISSING: "
            f"{SELECTOR_LOG_S_SEMANTIC_IDENTITY_SYMBOL}"
        )
    try:
        semantic_identity = semantic_identity_fn()
    except Exception as exc:
        raise RuntimeError(
            "E_SELECTOR_LOG_S_PROOF_SEMANTIC_QUERY_FAILED"
        ) from exc
    if (
        type(semantic_identity) is not str
        or semantic_identity != SELECTOR_LOG_S_SEMANTIC_IDENTITY
    ):
        raise RuntimeError(
            "E_SELECTOR_LOG_S_PROOF_SEMANTIC_IDENTITY_MISMATCH: "
            f"actual={semantic_identity!r}:"
            f"expected={SELECTOR_LOG_S_SEMANTIC_IDENTITY!r}"
        )
    raw_path = Path(str(getattr(module, "__file__", "") or ""))
    if raw_path.is_symlink():
        raise RuntimeError(
            f"E_SELECTOR_LOG_S_PROOF_SYMLINK: {raw_path}"
        )
    try:
        module_path = raw_path.resolve(strict=True)
        stat_result = module_path.stat()
    except OSError as exc:
        raise RuntimeError(
            f"E_SELECTOR_LOG_S_PROOF_ARTIFACT_MISSING: {raw_path}"
        ) from exc
    if not module_path.is_file() or stat_result.st_size <= 0:
        raise RuntimeError(
            f"E_SELECTOR_LOG_S_PROOF_ARTIFACT_INVALID: {module_path}"
        )
    digest = hashlib.sha256()
    with module_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    module_sha256 = digest.hexdigest()
    module_name = str(getattr(module, "__name__", "") or "")
    module_signature = selector_log_s_module_signature(
        extension_name=extension_name,
        module_name=module_name,
        module_path=str(module_path),
        module_sha256=module_sha256,
        module_size_bytes=int(stat_result.st_size),
        required_symbol=required_symbol,
        semantic_identity_symbol=SELECTOR_LOG_S_SEMANTIC_IDENTITY_SYMBOL,
        semantic_identity=semantic_identity,
    )
    return {
        "schema": SELECTOR_LOG_S_RUNTIME_PROOF_SCHEMA,
        "counter_semantics": SELECTOR_LOG_F_COUNTER_SEMANTICS,
        "extension_name": extension_name,
        "module_name": module_name,
        "module_path": str(module_path),
        "module_sha256": module_sha256,
        "module_size_bytes": int(stat_result.st_size),
        "required_symbol": required_symbol,
        "required_symbol_present": True,
        "semantic_identity_symbol": SELECTOR_LOG_S_SEMANTIC_IDENTITY_SYMBOL,
        "semantic_identity": semantic_identity,
        "module_signature": module_signature,
        "last_route": _PROCESS_LOG_F_LAST_ROUTE,
        "last_dispatch_reason": _PROCESS_LOG_F_LAST_DISPATCH_REASON,
        "last_admission_identity": _PROCESS_LOG_F_LAST_ADMISSION_IDENTITY,
        "admission_event_count": int(_PROCESS_LOG_F_ADMISSION_EVENT_COUNT),
        "admission_chain_sha256": _PROCESS_LOG_F_ADMISSION_CHAIN_SHA256,
        "route_counts": {
            route: int(_PROCESS_LOG_F_ROUTE_COUNTS.get(route, 0))
            for route in SELECTOR_LOG_F_ROUTES
        },
        "tiled_cohort_count": int(_PROCESS_TILED_COHORT_COUNT),
        "tiled_job_count": int(_PROCESS_TILED_JOB_COUNT),
        "tiled_direct_count": int(_PROCESS_TILED_DIRECT_COUNT),
        "tiled_kernel_launch_count": int(_PROCESS_TILED_KERNEL_LAUNCH_COUNT),
        "tiled_admission_failure_count": int(
            _PROCESS_TILED_ADMISSION_FAILURE_COUNT
        ),
    }












def _resolve_phase_output_kv_len_cpu(
    *,
    kv_len: int,
    capture_row: int,
    out_capture_scores: Optional[torch.Tensor],
    out_kv_len_per_capture_row_cpu: Sequence[int] | None,
) -> int:
    effective_kv_len = max(0, int(kv_len))
    if (
        out_kv_len_per_capture_row_cpu is not None
        and 0 <= int(capture_row) < len(out_kv_len_per_capture_row_cpu)
    ):
        effective_kv_len = min(
            effective_kv_len,
            max(0, int(out_kv_len_per_capture_row_cpu[int(capture_row)])),
        )
    if isinstance(out_capture_scores, torch.Tensor) and out_capture_scores.dim() >= 4:
        effective_kv_len = min(effective_kv_len, int(out_capture_scores.shape[-1]))
    return effective_kv_len


def _canonical_cpu_values(
    name: str,
    values: object,
    *,
    as_bool: bool = False,
) -> tuple[int | bool, ...]:
    """Canonicalize replicated host truth without ever reading a CUDA scalar."""

    if values is None:
        _selector_log_f_contract_error(
            "CPU_AUTHORITY",
            f"{name} must be supplied by the CPU authority",
        )
    if isinstance(values, torch.Tensor):
        _selector_log_f_contract_error(
            "CPU_AUTHORITY_TYPE",
            f"{name} must be an immutable host sequence, not a tensor",
        )
    try:
        raw_values = tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise RuntimeError(
            f"E_SELECTOR_LOG_F_CPU_AUTHORITY_TYPE: {name} must be a host sequence"
        ) from exc
    if any(isinstance(value, torch.Tensor) for value in raw_values):
        _selector_log_f_contract_error(
            "CPU_AUTHORITY_TYPE",
            f"{name} must not contain tensor scalars",
        )
    caster = bool if as_bool else int
    try:
        return tuple(caster(value) for value in raw_values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(
            f"E_SELECTOR_LOG_F_CPU_AUTHORITY_VALUE: {name} contains invalid values"
        ) from exc




def _validate_phase_outputs(
    *,
    name: str,
    capture_row_by_batch_row_i32: torch.Tensor,
    out_capture_scores: Optional[torch.Tensor],
    out_log_f_denoms: Optional[torch.Tensor],
    batch_size: int,
    device: torch.device,
) -> None:
    if capture_row_by_batch_row_i32.dim() != 1:
        raise ValueError(f"{name} capture row mapping must be rank-1")
    if int(capture_row_by_batch_row_i32.numel()) != batch_size:
        raise ValueError(f"{name} capture row mapping must align with batch size")
    if capture_row_by_batch_row_i32.device != device:
        raise ValueError(f"{name} capture row mapping must share device")
    if out_capture_scores is None and out_log_f_denoms is None:
        return
    if out_capture_scores is None or out_log_f_denoms is None:
        raise ValueError(f"{name} outputs must be provided together")
    if out_capture_scores.dim() != 4:
        raise ValueError(f"{name} out_capture_scores must be [capture_rows, heads, 1, kv]")
    if out_log_f_denoms.dim() != 2:
        raise ValueError(f"{name} out_log_f_denoms must be [capture_rows, heads]")
    if int(out_capture_scores.shape[2]) != 1:
        raise ValueError(f"{name} out_capture_scores must have window == 1")
    if out_capture_scores.device != device or out_log_f_denoms.device != device:
        raise ValueError(f"{name} outputs must share device with scratch")
    if int(out_capture_scores.shape[0]) != int(out_log_f_denoms.shape[0]):
        raise ValueError(f"{name} output row count must align")
    if int(out_capture_scores.shape[1]) != int(out_log_f_denoms.shape[1]):
        raise ValueError(f"{name} output head count must align")


def _validate_prefill_postprocess_inputs(
    *,
    scratch_capture_scores: torch.Tensor,
    producer_rows_i32: torch.Tensor,
    row_capture_last_n_i32: torch.Tensor,
    row_is_prefill_producer: torch.Tensor,
    seqused_k: torch.Tensor,
    active_capture_row_by_batch_row_i32: torch.Tensor,
    prefill_out_capture_scores: Optional[torch.Tensor],
    prefill_out_log_f_denoms: Optional[torch.Tensor],
    refresh_out_capture_scores: Optional[torch.Tensor],
    refresh_out_log_f_denoms: Optional[torch.Tensor],
) -> None:
    if scratch_capture_scores.dim() != 4:
        raise ValueError("scratch_capture_scores must be [capture_rows, heads, tail_q, kv]")
    if producer_rows_i32.dim() != 1:
        raise ValueError("producer_rows_i32 must be rank-1")
    if row_capture_last_n_i32.dim() != 1:
        raise ValueError("row_capture_last_n_i32 must be rank-1")
    if row_is_prefill_producer.dim() != 1 or seqused_k.dim() != 1:
        raise ValueError("row_is_prefill_producer and seqused_k must be rank-1")
    batch_size = int(row_capture_last_n_i32.numel())
    if int(row_is_prefill_producer.numel()) != batch_size:
        raise ValueError("row_is_prefill_producer must align with row_capture_last_n_i32")
    if int(seqused_k.numel()) != batch_size:
        raise ValueError("seqused_k must align with row_capture_last_n_i32")
    if int(producer_rows_i32.numel()) != int(scratch_capture_scores.shape[0]):
        raise ValueError("producer_rows_i32 must align with scratch capture rows")
    if producer_rows_i32.device != scratch_capture_scores.device:
        raise ValueError("producer_rows_i32 must share device with scratch_capture_scores")

    device = scratch_capture_scores.device
    _validate_phase_outputs(
        name="active",
        capture_row_by_batch_row_i32=active_capture_row_by_batch_row_i32,
        out_capture_scores=None,
        out_log_f_denoms=None,
        batch_size=batch_size,
        device=device,
    )
    _validate_phase_outputs(
        name="prefill",
        capture_row_by_batch_row_i32=active_capture_row_by_batch_row_i32,
        out_capture_scores=prefill_out_capture_scores,
        out_log_f_denoms=prefill_out_log_f_denoms,
        batch_size=batch_size,
        device=device,
    )
    _validate_phase_outputs(
        name="refresh",
        capture_row_by_batch_row_i32=active_capture_row_by_batch_row_i32,
        out_capture_scores=refresh_out_capture_scores,
        out_log_f_denoms=refresh_out_log_f_denoms,
        batch_size=batch_size,
        device=device,
    )


def _stage_meta_rows(
    rows: Sequence[Sequence[int]],
    *,
    dtype: torch.dtype,
    device: torch.device,
    cache_owner: Optional[object],
    cache_name: str,
) -> torch.Tensor:
    if cache_owner is None:
        # The explicit-owner production path remains lock-free.  The public
        # ownerless path shares one process cache, so serialize that exceptional
        # API boundary to preserve carrier reservation across Torch calls that
        # may release the GIL.
        with _PROCESS_META_STAGING_LOCK:
            return _stage_meta_rows(
                rows,
                dtype=dtype,
                device=device,
                cache_owner=_PROCESS_META_STAGING_OWNER,
                cache_name=cache_name,
            )
    row_count = len(rows)
    if row_count <= 0:
        return torch.empty((0, 0), dtype=dtype, device=device)
    col_count = len(rows[0])
    if col_count <= 0:
        raise ValueError("capture postprocess metadata rows must be non-empty")
    for row in rows:
        if len(row) != col_count:
            raise ValueError("capture postprocess metadata rows must have fixed width")

    device_index = -1 if device.index is None else int(device.index)
    current_stream = None
    stream_identity = -1
    if device.type == "cuda":
        current_stream = torch.cuda.current_stream(device=device)
        if device_index < 0:
            stream_device = getattr(current_stream, "device", None)
            stream_device_index = getattr(stream_device, "index", None)
            if stream_device_index is None:
                stream_device_index = torch.cuda.current_device()
            device_index = int(stream_device_index)
        stream_identity = int(
            getattr(current_stream, "cuda_stream", id(current_stream))
        )
    key = (
        str(device.type),
        device_index,
        stream_identity,
        dtype,
        row_count,
        col_count,
    )
    # A pinned source cannot be rewritten until its non-blocking H2D has
    # completed.  Fixed-depth rings turned that lifetime rule into a host
    # synchronize whenever CPU submission outran an empirical slot count.
    # Keep independent pools for every stream/shape contract.  A key is owned
    # by one CUDA stream, so submission order gives the slots a FIFO: the
    # cursor is always the oldest carrier.  One event query therefore proves
    # whether that carrier is reusable; a busy carrier grows the pool at the
    # temporal tail without a scan, arbitrary cap, or host-blocking fallback.
    # Stream identity also protects each slot's GPU metadata consumer through
    # same-stream ordering.
    pools_attr = f"_fa3_capture_postprocess_{cache_name}_pools"
    pools = getattr(cache_owner, pools_attr, None)
    if pools is None:
        pools = {}
        setattr(cache_owner, pools_attr, pools)
    elif not isinstance(pools, dict):
        raise RuntimeError("capture postprocess metadata staging pool is invalid")
    state = pools.get(key)
    if state is None:
        state = {"slots": [], "next": 0}
        pools[key] = state
    elif not isinstance(state, dict) or not isinstance(state.get("slots"), list):
        raise RuntimeError("capture postprocess metadata staging state is invalid")

    slots = state["slots"]
    slot = None
    next_oldest_index = 0
    if slots:
        oldest_index = int(state.get("next", 0)) % len(slots)
        candidate = slots[oldest_index]
        if not isinstance(candidate, dict):
            raise RuntimeError("capture postprocess metadata staging slot is invalid")
        candidate_event = candidate.get("evt")
        if candidate_event is None or candidate_event.query():
            slot = candidate
            next_oldest_index = (oldest_index + 1) % len(slots)

    if slot is None:
        # CUDA non-blocking H2D requires a pinned source for the latency and
        # lifetime contract used below.  A pageable fallback can silently turn
        # this call into a host-blocking transfer, so allocation failure is a
        # deployment error and propagates unchanged.
        cpu_stage = torch.empty(
            (row_count, col_count),
            dtype=dtype,
            device="cpu",
            pin_memory=device.type == "cuda",
        )
        gpu_stage = torch.empty((row_count, col_count), dtype=dtype, device=device)
        slot = {"cpu": cpu_stage, "gpu": gpu_stage, "evt": None}
        if slots:
            # Insert the newest carrier immediately before the oldest physical
            # index.  The old oldest shifts right and remains the cursor, which
            # preserves circular temporal order even when the cursor is nonzero.
            oldest_index = int(state.get("next", 0)) % len(slots)
            slots.insert(oldest_index, slot)
            next_oldest_index = oldest_index + 1
        else:
            slots.append(slot)
            next_oldest_index = 0
    else:
        cpu_stage = slot["cpu"]
        gpu_stage = slot["gpu"]
    state["next"] = next_oldest_index

    # [META-STAGE-BULK 2026-07-06] 逐元素 Python setitem（rows×cols×36 层/步）
    # 换 C 层一次构造+整块拷贝：值/位置逐位等价，纯 host 构造提速 ~10×。
    cpu_stage.copy_(torch.tensor(rows, dtype=dtype))
    gpu_stage.copy_(cpu_stage, non_blocking=True)
    if current_stream is not None:
        evt = slot.get("evt")
        if evt is None:
            evt = torch.cuda.Event(enable_timing=False)
            slot["evt"] = evt
        evt.record(current_stream)
    return gpu_stage


def _tiled_k_capacity(logical_k: int) -> int:
    from utils import selector_log_s_ext

    logical_k = int(logical_k)
    if logical_k <= 0:
        raise ValueError("tiled logical K capacity must be positive")
    tile_k = int(selector_log_s_ext.LOG_F_R2_TILED_TILE_K)
    return ((logical_k + tile_k - 1) // tile_k) * tile_k


def _tiled_n_capacity(num_rows: int) -> int:
    num_rows = int(num_rows)
    if num_rows <= 0:
        raise ValueError("tiled row capacity must be positive")
    return 1 << (num_rows - 1).bit_length()


def plan_tiled_capture_postprocess_resources(
    *,
    slot_count: int,
    num_rows_capacity: int,
    num_query_heads: int,
    logical_k_capacity: int,
) -> TiledCapturePostprocessResourceReport:
    """Return canonical geometry for an exact structural GPU slot count."""

    from utils import selector_log_s_ext

    slots = int(slot_count)
    if slots <= 0:
        raise ValueError("tiled structural slot count must be positive")
    n_capacity = _tiled_n_capacity(num_rows_capacity)
    h_capacity = int(num_query_heads)
    if h_capacity <= 0:
        raise ValueError("tiled head capacity must be positive")
    k_capacity = _tiled_k_capacity(logical_k_capacity)
    workspace_bytes = int(
        selector_log_s_ext.log_f_r2_tiled_workspace_nbytes(
            num_seqs_capacity=n_capacity,
            num_query_heads_capacity=h_capacity,
            logical_k_capacity=k_capacity,
        )
    )
    # Each slot owns [N,10] int32 + [N,4] int64 on host and device.
    meta_bytes = n_capacity * (10 * 4 + 4 * 8)
    device_bytes = workspace_bytes + meta_bytes
    return TiledCapturePostprocessResourceReport(
        num_rows_capacity=n_capacity,
        num_query_heads_capacity=h_capacity,
        logical_k_capacity=k_capacity,
        slot_count=slots,
        workspace_bytes_per_slot=workspace_bytes,
        device_meta_bytes_per_slot=meta_bytes,
        host_meta_bytes_per_slot=meta_bytes,
        device_bytes_per_slot=device_bytes,
        total_workspace_bytes=workspace_bytes * slots,
        total_device_meta_bytes=meta_bytes * slots,
        total_host_meta_bytes=meta_bytes * slots,
        total_device_bytes=device_bytes * slots,
    )


def _tiled_resource_key(
    *,
    device: torch.device,
    stream: torch.cuda.Stream,
    n_capacity: int,
    h_capacity: int,
    k_capacity: int,
) -> tuple[int, int, int, int, int]:
    return (
        -1 if device.index is None else int(device.index),
        int(stream.cuda_stream),
        int(n_capacity),
        int(h_capacity),
        int(k_capacity),
    )


def _is_tiled_tensor_contract_candidate(
    *,
    scratch: torch.Tensor,
    output: torch.Tensor,
    denom: torch.Tensor,
    alpha: float,
) -> bool:
    """Check non-row tiled semantics after the existing loop proves large K."""

    return (
        scratch.dtype == torch.float16
        and output.dtype == torch.float16
        and bool(scratch.is_contiguous())
        and bool(output.is_contiguous())
        and bool(denom.is_contiguous())
        and float(alpha) == 0.5
    )


def _snapshot_prepared_tiled_resource(
    *,
    cache_owner: Optional[object],
    device: torch.device,
) -> _PreparedTiledResourceSnapshot:
    """Freeze one current-stream seal proof without locks or allocation."""

    device = torch.device(device)
    stream = torch.cuda.current_stream(device=device)
    unavailable = _PreparedTiledResourceSnapshot(
        device=device,
        stream=stream,
        key=None,
        report=None,
    )
    if cache_owner is None:
        return unavailable
    seal = getattr(cache_owner, "_selector_log_f_r2_tiled_resource_seal", None)
    if not isinstance(seal, dict):
        return unavailable
    report = seal.get("report")
    key = seal.get("key")
    if not isinstance(report, TiledCapturePostprocessResourceReport):
        return unavailable
    if not isinstance(key, tuple) or len(key) != 5:
        return unavailable
    expected_key = _tiled_resource_key(
        device=device,
        stream=stream,
        n_capacity=report.num_rows_capacity,
        h_capacity=report.num_query_heads_capacity,
        k_capacity=report.logical_k_capacity,
    )
    if key != expected_key:
        return unavailable
    return _PreparedTiledResourceSnapshot(
        device=device,
        stream=stream,
        key=key,
        report=report,
    )


def _prepared_tiled_resource_snapshot_for_device(
    *,
    snapshots: dict[torch.device, _PreparedTiledResourceSnapshot],
    cache_owner: Optional[object],
    device: torch.device,
) -> _PreparedTiledResourceSnapshot:
    """Resolve at most one immutable resource snapshot per device and call."""

    device = torch.device(device)
    snapshot = snapshots.get(device)
    if snapshot is None:
        snapshot = _snapshot_prepared_tiled_resource(
            cache_owner=cache_owner,
            device=device,
        )
        snapshots[device] = snapshot
    return snapshot


def _prepared_tiled_resource_available(
    *,
    snapshot: _PreparedTiledResourceSnapshot,
    device: torch.device,
    num_rows: int,
    num_query_heads: int,
    logical_k_max: int,
) -> bool:
    """Prove the frozen prepared resource covers this live tiled geometry."""

    report = snapshot.report
    key = snapshot.key
    return bool(
        isinstance(report, TiledCapturePostprocessResourceReport)
        and isinstance(key, tuple)
        and snapshot.device == torch.device(device)
        and 0 < int(num_rows) <= report.num_rows_capacity
        and 0 < int(num_query_heads) <= report.num_query_heads_capacity
        and 0 < int(logical_k_max) <= report.logical_k_capacity
    )


def _tiled_resource_cache_for_owner_locked(
    cache_owner: Optional[object],
) -> dict[tuple[int, int, int, int, int], dict[str, Any]]:
    if cache_owner is None:
        raise RuntimeError("E_SELECTOR_LOG_F_TILED_RESOURCE_OWNER_REQUIRED")
    cache = getattr(cache_owner, "_selector_log_f_r2_tiled_resources", None)
    if cache is None:
        cache = {}
        setattr(cache_owner, "_selector_log_f_r2_tiled_resources", cache)
    if not isinstance(cache, dict):
        raise RuntimeError("E_SELECTOR_LOG_F_TILED_RESOURCE_CACHE_DRIFT")
    return cache


def _new_tiled_resource_state() -> dict[str, Any]:
    return {
        "slots": [],
        "free_slots": [],
        "reserved_count": 0,
        "next_reservation_id": 1,
    }


def _tiled_resource_free_slots_locked(
    state: dict[str, Any],
) -> list[dict[str, Any]]:
    slots = state.get("slots")
    free_slots = state.get("free_slots")
    reserved_count = state.get("reserved_count")
    if (
        not isinstance(slots, list)
        or not isinstance(free_slots, list)
        or isinstance(reserved_count, bool)
        or not isinstance(reserved_count, int)
        or reserved_count < 0
        or len(free_slots) + reserved_count != len(slots)
    ):
        raise RuntimeError("E_SELECTOR_LOG_F_TILED_RESOURCE_STATE_DRIFT")
    return free_slots


def _bind_tiled_resource_slot_locked(
    state: dict[str, Any],
    slot: dict[str, Any],
) -> None:
    if not isinstance(slot, dict) or slot.get("_pool_state") is not None:
        raise RuntimeError("E_SELECTOR_LOG_F_TILED_RESOURCE_SLOT_DRIFT")
    slot["_pool_state"] = state
    slot["reservation_id"] = None


def _allocate_tiled_resource_slot(
    *,
    device: torch.device,
    n_capacity: int,
    h_capacity: int,
    k_capacity: int,
) -> dict[str, Any]:
    from utils import selector_log_s_ext

    return {
        "gpu_i32": torch.empty(
            (n_capacity, 10), dtype=torch.int32, device=device
        ),
        "gpu_i64": torch.empty(
            (n_capacity, 4), dtype=torch.int64, device=device
        ),
        "workspace": selector_log_s_ext.allocate_log_f_r2_tiled_workspace(
            device,
            num_seqs_capacity=n_capacity,
            num_query_heads_capacity=h_capacity,
            logical_k_capacity=k_capacity,
        ),
        # GPU storage is structurally bounded and same-stream reusable as soon
        # as its launch sequence is enqueued.  The pinned H2D source has a
        # different lifetime: it cannot be rewritten until both copies finish.
        # Keep that small carrier in an independent pressure-observed FIFO so a
        # busy host source never duplicates this slot's large workspace.
        "meta_carriers": [_allocate_tiled_meta_carrier(n_capacity=n_capacity)],
        "meta_next": 0,
        "reserved": False,
        "reservation_stream": None,
    }


def _allocate_tiled_meta_carrier(*, n_capacity: int) -> dict[str, Any]:
    return {
        "cpu_i32": torch.empty(
            (int(n_capacity), 10),
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        ),
        "cpu_i64": torch.empty(
            (int(n_capacity), 4),
            dtype=torch.int64,
            device="cpu",
            pin_memory=True,
        ),
        "copy_event": torch.cuda.Event(enable_timing=False),
        "copy_recorded": False,
        "reserved": False,
    }


def prepare_tiled_capture_postprocess_resources(
    meta_cache_owner: object,
    device: torch.device,
    stream: torch.cuda.Stream,
    slot_count: int,
    num_rows_capacity: int,
    num_query_heads: int,
    logical_k_capacity: int,
) -> TiledCapturePostprocessResourceReport:
    """Preallocate and seal one exact stream/geometry GPU resource pool."""

    from utils import selector_log_s_ext

    if meta_cache_owner is None:
        raise ValueError("sealed tiled resources require an explicit cache owner")
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("sealed tiled resources require a CUDA device")
    report = plan_tiled_capture_postprocess_resources(
        slot_count=slot_count,
        num_rows_capacity=num_rows_capacity,
        num_query_heads=num_query_heads,
        logical_k_capacity=logical_k_capacity,
    )
    key = _tiled_resource_key(
        device=device,
        stream=stream,
        n_capacity=report.num_rows_capacity,
        h_capacity=report.num_query_heads_capacity,
        k_capacity=report.logical_k_capacity,
    )
    selector_log_s_ext._require_ext(force=True)

    with _PROCESS_TILED_RESOURCE_CACHE_LOCK:
        cache = _tiled_resource_cache_for_owner_locked(meta_cache_owner)
        seal = getattr(
            meta_cache_owner, "_selector_log_f_r2_tiled_resource_seal", None
        )
        if seal is not None:
            if not isinstance(seal, dict) or seal.get("key") != key:
                raise RuntimeError(
                    "E_SELECTOR_LOG_F_TILED_RESOURCE_SEALED_KEY_MISMATCH"
                )
            state = cache.get(key)
            if (
                seal.get("report") != report
                or not isinstance(state, dict)
                or int(state.get("structural_slot_count", -1))
                != report.structural_slot_count
                or len(state.get("slots", ())) != report.structural_slot_count
            ):
                raise RuntimeError("E_SELECTOR_LOG_F_TILED_RESOURCE_SEAL_DRIFT")
            free_slots = _tiled_resource_free_slots_locked(state)
            if (
                int(state["reserved_count"]) != 0
                or len(free_slots) != len(state["slots"])
                or any(
                    not isinstance(slot, dict)
                    or slot.get("_pool_state") is not state
                    or bool(slot.get("reserved", False))
                    or slot.get("reservation_id") is not None
                    for slot in state["slots"]
                )
            ):
                raise RuntimeError("E_SELECTOR_LOG_F_TILED_RESOURCE_SEAL_DRIFT")
            return report
        if any(existing_key != key for existing_key in cache):
            raise RuntimeError("E_SELECTOR_LOG_F_TILED_RESOURCE_PREBUILD_KEY_DRIFT")
        state = cache.get(key)
        if state is None:
            state = _new_tiled_resource_state()
            cache[key] = state
        if not isinstance(state, dict):
            raise RuntimeError("E_SELECTOR_LOG_F_TILED_RESOURCE_STATE_DRIFT")
        free_slots = _tiled_resource_free_slots_locked(state)
        if bool(state.get("prebuilding", False)):
            raise RuntimeError("E_SELECTOR_LOG_F_TILED_RESOURCE_PREBUILD_REENTRY")
        slots = state["slots"]
        if (
            state.get("structural_slot_count") is not None
            or any(
                not isinstance(slot, dict)
                or slot.get("_pool_state") is not state
                or bool(slot.get("reserved", False))
                or slot.get("reservation_id") is not None
                for slot in slots
            )
            or int(state["reserved_count"]) != 0
            or len(free_slots) != len(slots)
        ):
            raise RuntimeError("E_SELECTOR_LOG_F_TILED_RESOURCE_PREBUILD_STATE")
        if len(slots) > report.structural_slot_count:
            raise RuntimeError(
                "E_SELECTOR_LOG_F_TILED_RESOURCE_PREBUILD_CAPACITY_DRIFT"
            )
        state["prebuilding"] = True
        setattr(
            meta_cache_owner,
            "_selector_log_f_r2_tiled_resource_prebuild_key",
            key,
        )
        slot_count_before = len(slots)
        missing = report.structural_slot_count - slot_count_before

    try:
        new_slots = tuple(
            _allocate_tiled_resource_slot(
                device=device,
                n_capacity=report.num_rows_capacity,
                h_capacity=report.num_query_heads_capacity,
                k_capacity=report.logical_k_capacity,
            )
            for _ in range(missing)
        )
    except BaseException:
        with _PROCESS_TILED_RESOURCE_CACHE_LOCK:
            state["prebuilding"] = False
            setattr(
                meta_cache_owner,
                "_selector_log_f_r2_tiled_resource_prebuild_key",
                None,
            )
        raise

    with _PROCESS_TILED_RESOURCE_CACHE_LOCK:
        if cache.get(key) is not state or len(state["slots"]) != slot_count_before:
            state["prebuilding"] = False
            setattr(
                meta_cache_owner,
                "_selector_log_f_r2_tiled_resource_prebuild_key",
                None,
            )
            raise RuntimeError("E_SELECTOR_LOG_F_TILED_RESOURCE_PREBUILD_RACE")
        for slot in new_slots:
            _bind_tiled_resource_slot_locked(state, slot)
        state["slots"].extend(new_slots)
        state["free_slots"].extend(new_slots)
        if len(state["slots"]) != report.structural_slot_count:
            state["prebuilding"] = False
            setattr(
                meta_cache_owner,
                "_selector_log_f_r2_tiled_resource_prebuild_key",
                None,
            )
            raise RuntimeError("E_SELECTOR_LOG_F_TILED_RESOURCE_PREBUILD_RACE")
        state["structural_slot_count"] = report.structural_slot_count
        state["prebuilding"] = False
        setattr(
            meta_cache_owner,
            "_selector_log_f_r2_tiled_resource_seal",
            {"key": key, "report": report},
        )
        setattr(
            meta_cache_owner,
            "_selector_log_f_r2_tiled_resource_prebuild_key",
            None,
        )
    return report


def _acquire_tiled_resource_slot(
    *,
    cache_owner: Optional[object],
    device: torch.device,
    stream: torch.cuda.Stream,
    num_rows: int,
    num_query_heads: int,
    logical_k_max: int,
) -> tuple[dict[str, Any], int, int, int, int]:
    """Load and reserve reusable tiled storage before any job is claimed."""

    from utils import selector_log_s_ext

    if cache_owner is None:
        raise RuntimeError("E_SELECTOR_LOG_F_TILED_RESOURCE_OWNER_REQUIRED")
    # Resolve the immutable extension before touching resource ownership.  The
    # production sealed path performs only O(1) free-stack bookkeeping under
    # the lock and cannot allocate.  An unavailable optimized owner is a
    # deployment error, never a reason to mutate a job and then fall back.
    selector_log_s_ext._require_ext(force=True)
    actual_rows = int(num_rows)
    actual_heads = int(num_query_heads)
    actual_logical_k = int(logical_k_max)
    if actual_rows <= 0:
        raise ValueError("tiled row count must be positive")
    if actual_heads <= 0:
        raise ValueError("tiled head capacity must be positive")
    if actual_logical_k <= 0:
        raise ValueError("tiled logical K must be positive")
    requested_prefix = (
        -1 if device.index is None else int(device.index),
        int(stream.cuda_stream),
    )
    with _PROCESS_TILED_RESOURCE_CACHE_LOCK:
        cache = _tiled_resource_cache_for_owner_locked(cache_owner)
        seal = getattr(
            cache_owner, "_selector_log_f_r2_tiled_resource_seal", None
        )
        if seal is None:
            if getattr(
                cache_owner,
                "_selector_log_f_r2_tiled_resource_prebuild_key",
                None,
            ) is not None:
                raise RuntimeError(
                    "E_SELECTOR_LOG_F_TILED_RESOURCE_PREBUILD_IN_PROGRESS"
                )
            raise RuntimeError("E_SELECTOR_LOG_F_TILED_RESOURCE_SEAL_REQUIRED")
        if not isinstance(seal, dict):
            raise RuntimeError("E_SELECTOR_LOG_F_TILED_RESOURCE_SEAL_DRIFT")
        report = seal.get("report")
        key = seal.get("key")
        if (
            not isinstance(report, TiledCapturePostprocessResourceReport)
            or not isinstance(key, tuple)
            or len(key) != 5
        ):
            raise RuntimeError("E_SELECTOR_LOG_F_TILED_RESOURCE_SEAL_DRIFT")
        n_capacity = report.num_rows_capacity
        h_capacity = report.num_query_heads_capacity
        k_capacity = report.logical_k_capacity
        if tuple(key[:2]) != requested_prefix:
            raise RuntimeError(
                "E_SELECTOR_LOG_F_TILED_RESOURCE_SEALED_KEY_MISMATCH"
            )
        if (
            actual_rows > n_capacity
            or actual_heads > h_capacity
            or actual_logical_k > k_capacity
        ):
            raise RuntimeError(
                "E_SELECTOR_LOG_F_TILED_RESOURCE_SEALED_CAPACITY_EXCEEDED"
            )
        state = cache.get(key)
        if state is None:
            raise RuntimeError("E_SELECTOR_LOG_F_TILED_RESOURCE_SLOT_MISSING")
        if not isinstance(state, dict):
            raise RuntimeError("E_SELECTOR_LOG_F_TILED_RESOURCE_STATE_DRIFT")
        free_slots = _tiled_resource_free_slots_locked(state)
        if bool(state.get("prebuilding", False)):
            raise RuntimeError("E_SELECTOR_LOG_F_TILED_RESOURCE_PREBUILD_IN_PROGRESS")
        structural_slot_count = state.get("structural_slot_count")
        if (
            int(structural_slot_count or -1) != report.structural_slot_count
            or len(state["slots"]) != report.structural_slot_count
        ):
            raise RuntimeError("E_SELECTOR_LOG_F_TILED_RESOURCE_SLOT_MISSING")
        slot = free_slots.pop() if free_slots else None
        if slot is None:
            # Every sealed slot represents one structurally budgeted owner
            # lane.  Exhaustion means ownership escaped that proof; never hide
            # it by allocating an unprofiled giant workspace.
            raise RuntimeError(
                "E_SELECTOR_LOG_F_TILED_RESOURCE_OWNERSHIP_CAPACITY_DRIFT"
            )
        if (
            not isinstance(slot, dict)
            or slot.get("_pool_state") is not state
            or bool(slot.get("reserved", False))
            or slot.get("reservation_id") is not None
        ):
            raise RuntimeError("E_SELECTOR_LOG_F_TILED_RESOURCE_SLOT_DRIFT")
        reservation_id = int(state.get("next_reservation_id", 0))
        if reservation_id <= 0:
            raise RuntimeError("E_SELECTOR_LOG_F_TILED_RESOURCE_STATE_DRIFT")
        state["next_reservation_id"] = reservation_id + 1
        state["reserved_count"] = int(state["reserved_count"]) + 1
        slot["reserved"] = True
        slot["reservation_id"] = reservation_id
        slot["reservation_stream"] = stream
    return slot, reservation_id, n_capacity, h_capacity, k_capacity


def _acquire_tiled_meta_carrier(slot: dict[str, Any]) -> dict[str, Any]:
    """Reserve one pinned H2D source with one oldest-event query at steady state."""

    carriers = slot.get("meta_carriers")
    if not isinstance(carriers, list):
        raise RuntimeError("E_SELECTOR_LOG_F_TILED_META_CARRIER_STATE_DRIFT")
    carrier: dict[str, Any] | None = None
    next_oldest_index = 0
    oldest_index = 0
    if carriers:
        oldest_index = int(slot.get("meta_next", 0)) % len(carriers)
        candidate = carriers[oldest_index]
        if not isinstance(candidate, dict):
            raise RuntimeError("E_SELECTOR_LOG_F_TILED_META_CARRIER_DRIFT")
        copy_event = candidate.get("copy_event")
        if copy_event is None:
            raise RuntimeError("E_SELECTOR_LOG_F_TILED_META_EVENT_MISSING")
        candidate_ready = not bool(candidate.get("reserved", False))
        if candidate_ready and bool(candidate.get("copy_recorded", False)):
            # Same-stream submissions preserve carrier age.  The oldest event
            # being busy proves every later recorded carrier is newer, so one
            # query is the complete steady-state readiness check.
            candidate_ready = bool(copy_event.query())
        if candidate_ready:
            carrier = candidate
            next_oldest_index = (oldest_index + 1) % len(carriers)

    if carrier is None:
        gpu_i32 = slot.get("gpu_i32")
        if not isinstance(gpu_i32, torch.Tensor) or gpu_i32.dim() != 2:
            raise RuntimeError("E_SELECTOR_LOG_F_TILED_META_CAPACITY_DRIFT")
        carrier = _allocate_tiled_meta_carrier(n_capacity=int(gpu_i32.shape[0]))
        if not isinstance(carrier, dict) or carrier.get("copy_event") is None:
            raise RuntimeError("E_SELECTOR_LOG_F_TILED_META_EVENT_MISSING")
        if carriers:
            # Append temporally newest before the physical oldest; the shifted
            # old oldest remains the cursor even when it was nonzero.
            carriers.insert(oldest_index, carrier)
            next_oldest_index = oldest_index + 1
        else:
            carriers.append(carrier)
            next_oldest_index = 0
    slot["meta_next"] = next_oldest_index
    carrier["reserved"] = True
    return carrier


def _stage_tiled_meta_rows(
    slot: dict[str, Any],
    *,
    meta_i32_rows: Sequence[Sequence[int]],
    meta_i64_rows: Sequence[Sequence[int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bulk-stage one combined cohort into its exclusively reserved slot."""

    row_count = len(meta_i32_rows)
    if row_count <= 0 or len(meta_i64_rows) != row_count:
        raise ValueError("tiled metadata rows must be non-empty and aligned")
    gpu_i32 = slot["gpu_i32"]
    gpu_i64 = slot["gpu_i64"]
    reservation_stream = slot.get("reservation_stream")
    if reservation_stream is None:
        raise RuntimeError("E_SELECTOR_LOG_F_TILED_RESERVATION_STREAM_MISSING")
    carrier = _acquire_tiled_meta_carrier(slot)
    cpu_i32 = carrier["cpu_i32"]
    cpu_i64 = carrier["cpu_i64"]
    copy_event = carrier.get("copy_event")
    if copy_event is None:
        raise RuntimeError("E_SELECTOR_LOG_F_TILED_META_EVENT_MISSING")
    try:
        if row_count > int(cpu_i32.shape[0]) or row_count > int(cpu_i64.shape[0]):
            raise RuntimeError("E_SELECTOR_LOG_F_TILED_META_CAPACITY_DRIFT")
        req_meta_i32 = gpu_i32[:row_count]
        req_meta_i64 = gpu_i64[:row_count]
        cpu_i32[:row_count].copy_(torch.tensor(meta_i32_rows, dtype=torch.int32))
        cpu_i64[:row_count].copy_(torch.tensor(meta_i64_rows, dtype=torch.int64))
        req_meta_i32.copy_(cpu_i32[:row_count], non_blocking=True)
        req_meta_i64.copy_(cpu_i64[:row_count], non_blocking=True)
    finally:
        # This record must follow both H2D enqueues.  Recording even on an
        # exceptional partial-copy path fences whatever reached the stream
        # before the pinned source can become reusable.
        copy_event.record(reservation_stream)
        carrier["copy_recorded"] = True
        carrier["reserved"] = False
    return req_meta_i32, req_meta_i64


def _release_tiled_resource_slot(
    slot: dict[str, Any],
    *,
    stream: torch.cuda.Stream,
    reservation_id: int,
) -> None:
    """Release same-stream storage without creating a publication fence."""

    with _PROCESS_TILED_RESOURCE_CACHE_LOCK:
        state = slot.get("_pool_state")
        if not isinstance(state, dict):
            raise RuntimeError("E_SELECTOR_LOG_F_TILED_RESOURCE_SLOT_DRIFT")
        free_slots = _tiled_resource_free_slots_locked(state)
        if not bool(slot.get("reserved", False)):
            raise RuntimeError("E_SELECTOR_LOG_F_TILED_SLOT_NOT_RESERVED")
        if int(slot.get("reservation_id") or -1) != int(reservation_id):
            raise RuntimeError("E_SELECTOR_LOG_F_TILED_RESERVATION_ID_DRIFT")
        reservation_stream = slot.get("reservation_stream")
        if reservation_stream is None or int(reservation_stream.cuda_stream) != int(
            stream.cuda_stream
        ):
            raise RuntimeError("E_SELECTOR_LOG_F_TILED_RESERVATION_STREAM_DRIFT")
        reserved_count = int(state["reserved_count"])
        if reserved_count <= 0:
            raise RuntimeError("E_SELECTOR_LOG_F_TILED_RESOURCE_STATE_DRIFT")
        slot["reserved"] = False
        slot["reservation_id"] = None
        slot["reservation_stream"] = None
        state["reserved_count"] = reserved_count - 1
        free_slots.append(slot)
        if len(free_slots) + int(state["reserved_count"]) != len(state["slots"]):
            raise RuntimeError("E_SELECTOR_LOG_F_TILED_RESOURCE_STATE_DRIFT")


def _publish_tiled_resource_completion(
    slot: dict[str, Any],
    *,
    stream: torch.cuda.Stream,
    reservation_id: int,
) -> torch.cuda.Event:
    """Publish one immutable job fence, then release its same-stream slot.

    A completion event escapes through deferred jobs and may be waited long
    after this storage slot becomes reusable.  Re-recording a slot-owned event
    would make such a delayed wait observe a later publication.  Publication
    events are therefore one-shot identities; only each private pinned
    metadata carrier's copy-lifetime event is persistent.
    """

    event = torch.cuda.Event(enable_timing=False)
    event.record(stream)
    _release_tiled_resource_slot(
        slot,
        stream=stream,
        reservation_id=reservation_id,
    )
    return event


def postprocess_prefill_capture_scores(
    *,
    scratch_capture_scores: torch.Tensor,
    producer_rows_i32: torch.Tensor,
    row_capture_last_n_i32: torch.Tensor,
    row_is_prefill_producer: torch.Tensor,
    seqused_k: torch.Tensor,
    active_capture_row_by_batch_row_i32: torch.Tensor,
    prefill_out_capture_scores: Optional[torch.Tensor],
    prefill_out_log_f_denoms: Optional[torch.Tensor],
    refresh_out_capture_scores: Optional[torch.Tensor],
    refresh_out_log_f_denoms: Optional[torch.Tensor],
    prefill_out_kv_len_per_capture_row_i32: Optional[torch.Tensor] = None,
    refresh_out_kv_len_per_capture_row_i32: Optional[torch.Tensor] = None,
    producer_rows_cpu: Optional[Sequence[int]] = None,
    row_capture_last_n_cpu: Optional[Sequence[int]] = None,
    row_is_prefill_producer_cpu: Optional[Sequence[bool]] = None,
    seqused_k_cpu: Optional[Sequence[int]] = None,
    active_capture_row_by_batch_row_cpu: Optional[Sequence[int]] = None,
    prefill_out_kv_len_per_capture_row_cpu: Optional[Sequence[int]] = None,
    refresh_out_kv_len_per_capture_row_cpu: Optional[Sequence[int]] = None,
    skip_postprocess_rows_cpu: Optional[Sequence[int]] = None,
    row_capture_accum_prev_rows_cpu: Optional[Sequence[int]] = None,
    row_capture_accum_prev_capacity_cpu: Optional[Sequence[int]] = None,
    meta_cache_owner: Optional[object] = None,
    prepared_tiled_resource_snapshot: Optional[
        _PreparedTiledResourceSnapshot
    ] = None,
    alpha: float,
    debug_epoch: Optional[int] = None,
    debug_layer_index: Optional[int] = None,
    proof_job_key: Optional[Sequence[int]] = None,
) -> None:
    _validate_prefill_postprocess_inputs(
        scratch_capture_scores=scratch_capture_scores,
        producer_rows_i32=producer_rows_i32,
        row_capture_last_n_i32=row_capture_last_n_i32,
        row_is_prefill_producer=row_is_prefill_producer,
        seqused_k=seqused_k,
        active_capture_row_by_batch_row_i32=active_capture_row_by_batch_row_i32,
        prefill_out_capture_scores=prefill_out_capture_scores,
        prefill_out_log_f_denoms=prefill_out_log_f_denoms,
        refresh_out_capture_scores=refresh_out_capture_scores,
        refresh_out_log_f_denoms=refresh_out_log_f_denoms,
    )
    if scratch_capture_scores.device.type != "cuda":
        raise ValueError("postprocess_prefill_capture_scores requires CUDA tensors")

    from utils import selector_log_s_ext

    device = scratch_capture_scores.device
    def _profiled_postprocess_call(
        *,
        label: str,
        metadata: dict[str, object],
        call: Any,
    ) -> Any:
        return call()

    batch_size = int(row_capture_last_n_i32.numel())
    producer_rows_i32 = producer_rows_i32.to(device=device, dtype=torch.int32).reshape(-1)
    row_capture_last_n_i32 = row_capture_last_n_i32.to(device=device, dtype=torch.int32).reshape(batch_size)
    row_is_prefill_producer = row_is_prefill_producer.to(device=device, dtype=torch.bool).reshape(batch_size)
    seqused_k = seqused_k.to(device=device, dtype=torch.int32).reshape(batch_size)
    active_capture_row_by_batch_row_i32 = active_capture_row_by_batch_row_i32.reshape(batch_size)

    decode_gt1_rows = []
    lastn1_rows: list[tuple[int, int, int, torch.Tensor, torch.Tensor]] = []
    gt1_prefill_rows: list[tuple[int, int, int, int]] = []
    prefill_capture_rows_seen: set[int] = set()
    refresh_capture_rows_seen: set[int] = set()

    cpu_truth = (
        producer_rows_cpu,
        row_capture_last_n_cpu,
        row_is_prefill_producer_cpu,
        seqused_k_cpu,
        active_capture_row_by_batch_row_cpu,
    )
    if any(values is None for values in cpu_truth):
        _selector_log_f_contract_error(
            "CPU_AUTHORITY",
            "producer rows, row counts, phase flags, K values, and capture rows "
            "must be supplied by the CPU authority",
        )
    producer_rows_tuple = tuple(
        int(v) for v in _canonical_cpu_values("producer_rows_cpu", producer_rows_cpu)
    )
    row_capture_last_n_tuple = tuple(
        int(v)
        for v in _canonical_cpu_values(
            "row_capture_last_n_cpu", row_capture_last_n_cpu
        )
    )
    row_is_prefill_tuple = tuple(
        bool(v)
        for v in _canonical_cpu_values(
            "row_is_prefill_producer_cpu",
            row_is_prefill_producer_cpu,
            as_bool=True,
        )
    )
    seqused_k_tuple = tuple(
        int(v) for v in _canonical_cpu_values("seqused_k_cpu", seqused_k_cpu)
    )
    active_capture_row_tuple = tuple(
        int(v)
        for v in _canonical_cpu_values(
            "active_capture_row_by_batch_row_cpu",
            active_capture_row_by_batch_row_cpu,
        )
    )
    if prefill_out_kv_len_per_capture_row_cpu is not None:
        prefill_out_kv_len_per_capture_row_cpu = tuple(
            int(v)
            for v in _canonical_cpu_values(
                "prefill_out_kv_len_per_capture_row_cpu",
                prefill_out_kv_len_per_capture_row_cpu,
            )
        )
    if refresh_out_kv_len_per_capture_row_cpu is not None:
        refresh_out_kv_len_per_capture_row_cpu = tuple(
            int(v)
            for v in _canonical_cpu_values(
                "refresh_out_kv_len_per_capture_row_cpu",
                refresh_out_kv_len_per_capture_row_cpu,
            )
        )
    if len(producer_rows_tuple) != int(producer_rows_i32.numel()):
        _selector_log_f_contract_error(
            "CPU_AUTHORITY_SHAPE",
            "producer_rows_cpu must align exactly with producer_rows_i32",
        )
    for name, values in (
        ("row_capture_last_n_cpu", row_capture_last_n_tuple),
        ("row_is_prefill_producer_cpu", row_is_prefill_tuple),
        ("seqused_k_cpu", seqused_k_tuple),
        ("active_capture_row_by_batch_row_cpu", active_capture_row_tuple),
    ):
        if len(values) != batch_size:
            _selector_log_f_contract_error(
                "CPU_AUTHORITY_SHAPE",
                f"{name} must align exactly with batch size",
            )
    if len(set(producer_rows_tuple)) != len(producer_rows_tuple):
        _selector_log_f_contract_error(
            "CPU_AUTHORITY_VALUE",
            "producer_rows_cpu must not contain duplicate rows",
        )
    if any(value < 0 for value in row_capture_last_n_tuple):
        _selector_log_f_contract_error(
            "CPU_AUTHORITY_VALUE",
            "row_capture_last_n_cpu must be non-negative",
        )
    if any(value < 0 for value in seqused_k_tuple):
        _selector_log_f_contract_error(
            "CPU_AUTHORITY_VALUE",
            "seqused_k_cpu must be non-negative",
        )
    skip_postprocess_rows = (
        frozenset(
            int(v)
            for v in _canonical_cpu_values(
                "skip_postprocess_rows_cpu", skip_postprocess_rows_cpu
            )
        )
        if skip_postprocess_rows_cpu is not None
        else frozenset()
    )
    # [CHUNKED-CAPTURE-ACCUMULATE 2026-07-06] per-row 跨片累计元数据
    # （-1/缺省=非跨片行走原路径）。跨片行必须走 gt1 reduce（accumulate
    # 位路由），含 last_n==1 的尾片——lastn1 裸拷贝的常数平移破坏合并。
    if (row_capture_accum_prev_rows_cpu is None) != (
        row_capture_accum_prev_capacity_cpu is None
    ):
        _selector_log_f_contract_error(
            "ACCUM_AUTHORITY",
            "accumulation row and capacity truth must be supplied together",
        )
    accum_prev_rows_tuple = (
        tuple(
            int(v)
            for v in _canonical_cpu_values(
                "row_capture_accum_prev_rows_cpu",
                row_capture_accum_prev_rows_cpu,
            )
        )
        if row_capture_accum_prev_rows_cpu is not None
        else (-1,) * batch_size
    )
    accum_prev_capacity_tuple = (
        tuple(
            int(v)
            for v in _canonical_cpu_values(
                "row_capture_accum_prev_capacity_cpu",
                row_capture_accum_prev_capacity_cpu,
            )
        )
        if row_capture_accum_prev_capacity_cpu is not None
        else (0,) * batch_size
    )
    if len(accum_prev_rows_tuple) != batch_size or len(
        accum_prev_capacity_tuple
    ) != batch_size:
        _selector_log_f_contract_error(
            "ACCUM_AUTHORITY",
            "accumulation CPU truth must align exactly with batch size",
        )

    def _accum_prev_rows_for(batch_row: int) -> int:
        return int(accum_prev_rows_tuple[batch_row])

    def _accum_prev_capacity_for(batch_row: int) -> int:
        return int(accum_prev_capacity_tuple[batch_row])

    for scratch_row, batch_row_i32 in enumerate(producer_rows_tuple):
        batch_row = int(batch_row_i32)
        if batch_row < 0 or batch_row >= batch_size:
            _selector_log_f_contract_error(
                "CPU_AUTHORITY_VALUE",
                "producer_rows_cpu contains an out-of-range batch row",
            )
        if batch_row in skip_postprocess_rows:
            continue
        last_n = int(row_capture_last_n_tuple[batch_row])
        kv_len = int(seqused_k_tuple[batch_row])
        is_prefill_producer = bool(row_is_prefill_tuple[batch_row])
        capture_row = int(active_capture_row_tuple[batch_row])
        if is_prefill_producer:
            if prefill_out_capture_scores is None or prefill_out_log_f_denoms is None:
                _selector_log_f_contract_error(
                    "OUTPUT",
                    "prefill producer rows require prefill output tensors",
                )
            out_capture_scores = prefill_out_capture_scores
            out_log_f_denoms = prefill_out_log_f_denoms
            out_kv_len_per_capture_row_cpu = prefill_out_kv_len_per_capture_row_cpu
        else:
            if refresh_out_capture_scores is None or refresh_out_log_f_denoms is None:
                _selector_log_f_contract_error(
                    "OUTPUT",
                    "decode producer rows require refresh output tensors",
                )
            out_capture_scores = refresh_out_capture_scores
            out_log_f_denoms = refresh_out_log_f_denoms
            out_kv_len_per_capture_row_cpu = refresh_out_kv_len_per_capture_row_cpu
        if capture_row < 0:
            _selector_log_f_contract_error(
                "CPU_AUTHORITY_VALUE",
                "producer row is missing its active capture row",
            )
        if out_kv_len_per_capture_row_cpu is None:
            _selector_log_f_contract_error(
                "CPU_AUTHORITY",
                "phase output K values must be supplied by the CPU authority",
            )
        if capture_row >= len(out_kv_len_per_capture_row_cpu):
            _selector_log_f_contract_error(
                "CPU_AUTHORITY_SHAPE",
                "phase output K truth does not cover the active capture row",
            )
        if capture_row >= int(out_capture_scores.shape[0]) or capture_row >= int(
            out_log_f_denoms.shape[0]
        ):
            _selector_log_f_contract_error(
                "OUTPUT_CAPACITY",
                "active capture row exceeds output tensor capacity",
            )
        if int(out_kv_len_per_capture_row_cpu[capture_row]) < 0:
            _selector_log_f_contract_error(
                "CPU_AUTHORITY_VALUE",
                "phase output K values must be non-negative",
            )
        if kv_len <= 0 or last_n <= 0:
            continue
        effective_kv_len = _resolve_phase_output_kv_len_cpu(
            kv_len=kv_len,
            capture_row=capture_row,
            out_capture_scores=out_capture_scores,
            out_kv_len_per_capture_row_cpu=out_kv_len_per_capture_row_cpu,
        )
        if effective_kv_len <= 0:
            continue
        capture_rows_seen = (
            prefill_capture_rows_seen
            if is_prefill_producer
            else refresh_capture_rows_seen
        )
        if capture_row in capture_rows_seen:
            _selector_log_f_contract_error(
                "OUTPUT_SEQ_OVERLAP",
                "one phase cannot schedule multiple writers for one capture row",
            )
        capture_rows_seen.add(capture_row)
        row_is_accum = bool(is_prefill_producer) and _accum_prev_rows_for(batch_row) >= 0
        if last_n == 1 and not row_is_accum:
            lastn1_rows.append(
                (scratch_row, capture_row, effective_kv_len, out_capture_scores, out_log_f_denoms)
            )
            continue
        if not is_prefill_producer:
            decode_gt1_rows.append(batch_row)
            continue
        gt1_prefill_rows.append((scratch_row, batch_row, capture_row, effective_kv_len))

    if decode_gt1_rows:
        raise ValueError(
            "decode producer rows must have last_n == 1 in capture postprocess"
        )
    if lastn1_rows:
        meta_i32_rows = []
        meta_i64_rows = []
        scratch_head_stride = int(scratch_capture_scores.stride(1))
        for scratch_row, capture_row, effective_kv_len, out_capture_scores, out_log_f_denoms in lastn1_rows:
            meta_i32_rows.append(
                [
                    int(effective_kv_len),
                    scratch_head_stride,
                    1,
                    0,
                    int(effective_kv_len),
                    8,
                    int(out_capture_scores.stride(1)),
                ]
            )
            meta_i64_rows.append(
                [
                    0,
                    int(scratch_capture_scores[int(scratch_row)].data_ptr()),
                    int(out_capture_scores[int(capture_row)].data_ptr()),
                    int(out_log_f_denoms[int(capture_row)].data_ptr()),
                ]
            )

        def _stage_lastn1_meta() -> tuple[torch.Tensor, torch.Tensor]:
            return (
                _stage_meta_rows(
                    meta_i32_rows,
                    dtype=torch.int32,
                    device=device,
                    cache_owner=meta_cache_owner,
                    cache_name="lastn1_i32",
                ),
                _stage_meta_rows(
                    meta_i64_rows,
                    dtype=torch.int64,
                    device=device,
                    cache_owner=meta_cache_owner,
                    cache_name="lastn1_i64",
                ),
            )

        lastn1_metadata = {
            "epoch": -1 if debug_epoch is None else int(debug_epoch),
            "layer": -1 if debug_layer_index is None else int(debug_layer_index),
            "row_count": len(meta_i32_rows),
            "num_query_heads": int(scratch_capture_scores.shape[1]),
            "effective_kv_len_max": max(
                (int(row[2]) for row in lastn1_rows),
                default=0,
            ),
            "log_f_out_fp32": bool(lastn1_rows[0][3].dtype == torch.float32),
        }
        req_meta_i32, req_meta_i64 = _profiled_postprocess_call(
            label="capture_postprocess_meta_lastn1",
            metadata=lastn1_metadata,
            call=_stage_lastn1_meta,
        )

        def _copy_lastn1() -> None:
            selector_log_s_ext.copy_log_f_lastn1_scratch_cuda(
                req_meta_i32=req_meta_i32,
                req_meta_i64=req_meta_i64,
                num_seqs=len(meta_i32_rows),
                num_query_heads=int(scratch_capture_scores.shape[1]),
                scratch_in_fp16=scratch_capture_scores.dtype == torch.float16,
                log_f_out_fp32=lastn1_rows[0][3].dtype == torch.float32,
            )

        _profiled_postprocess_call(
            label="capture_postprocess_copy_lastn1",
            metadata=lastn1_metadata,
            call=_copy_lastn1,
        )
    if not gt1_prefill_rows:
        return
    if prefill_out_capture_scores is None or prefill_out_log_f_denoms is None:
        raise ValueError("prefill gt1 rows require prefill output tensors")

    scratch_head_stride = int(scratch_capture_scores.stride(1))
    scratch_row_stride = int(scratch_capture_scores.stride(2))
    meta_i32_rows: list[list[int]] = []
    meta_i64_rows: list[list[int]] = []
    all_large_k = True
    tiled_row_shape_candidate = True
    logical_k_max = 0
    for scratch_row, batch_row, capture_row, effective_kv_len in gt1_prefill_rows:
        last_n = int(row_capture_last_n_tuple[batch_row])
        # The capture store clamps to the same CPU-authored effective K, so the
        # reducer never reads beyond graph-frozen scratch storage.
        accum_prev = _accum_prev_rows_for(int(batch_row))
        flags = 8 | (16 if accum_prev >= 0 else 0)
        effective_k = int(effective_kv_len)
        all_large_k = bool(
            all_large_k and effective_k > _LOG_F_R2_TILED_MIN_K_EXCLUSIVE
        )
        tiled_row_shape_candidate = bool(
            tiled_row_shape_candidate
            and (last_n == 2 or (flags == 24 and last_n == 1))
        )
        logical_k_max = max(logical_k_max, effective_k)
        meta_i32_rows.append(
            [
                effective_k,
                scratch_head_stride,
                last_n,
                0,
                effective_k,
                int(flags),
                int(prefill_out_capture_scores.stride(1)),
                scratch_row_stride,
                max(0, int(accum_prev)),
                _accum_prev_capacity_for(int(batch_row)) if accum_prev >= 0 else 0,
            ]
        )
        meta_i64_rows.append(
            [
                0,
                int(scratch_capture_scores[int(scratch_row)].data_ptr()),
                int(prefill_out_capture_scores[int(capture_row)].data_ptr()),
                int(prefill_out_log_f_denoms[int(capture_row)].data_ptr()),
            ]
        )

    tiled_resource_snapshot = prepared_tiled_resource_snapshot
    tiled_resource_available = True
    if (
        all_large_k
        and tiled_row_shape_candidate
        and _is_tiled_tensor_contract_candidate(
            scratch=scratch_capture_scores,
            output=prefill_out_capture_scores,
            denom=prefill_out_log_f_denoms,
            alpha=float(alpha),
        )
    ):
        if tiled_resource_snapshot is None:
            tiled_resource_snapshot = _snapshot_prepared_tiled_resource(
                cache_owner=meta_cache_owner,
                device=device,
            )
        tiled_resource_available = _prepared_tiled_resource_available(
            snapshot=tiled_resource_snapshot,
            device=device,
            num_rows=len(meta_i32_rows),
            num_query_heads=int(scratch_capture_scores.shape[1]),
            logical_k_max=logical_k_max,
        )

    admission = _LogFReduceAdmission(
        meta_i32_rows=tuple(tuple(row) for row in meta_i32_rows),
        scratch_row_indices=tuple(
            int(scratch_row) for scratch_row, _, _, _ in gt1_prefill_rows
        ),
        output_row_indices=tuple(
            int(capture_row) for _, _, capture_row, _ in gt1_prefill_rows
        ),
        scratch=_tensor_contract(scratch_capture_scores),
        output=_tensor_contract(prefill_out_capture_scores),
        denom=_tensor_contract(prefill_out_log_f_denoms),
        alpha=float(alpha),
        capability=tuple(int(v) for v in torch.cuda.get_device_capability(device)),
        cpu_authority_validated=True,
        local_same_device=bool(
            scratch_capture_scores.device
            == prefill_out_capture_scores.device
            == prefill_out_log_f_denoms.device
        ),
        tiled_resource_available=tiled_resource_available,
    )
    dispatch = _resolve_log_f_reduce_dispatch(admission)
    reduce_route = dispatch.route
    resident_k_bucket = dispatch.resident_k_bucket

    event_epoch = -1 if debug_epoch is None else int(debug_epoch)
    event_layer = -1 if debug_layer_index is None else int(debug_layer_index)
    event_handle_id = _LOG_F_DIRECT_EVENT_SENTINEL
    event_handle_generation = _LOG_F_DIRECT_EVENT_SENTINEL
    if proof_job_key is not None:
        if len(proof_job_key) != 3:
            _selector_log_f_contract_error(
                "JOB_KEY",
                "deferred proof job key must be (handle, generation, layer)",
            )
        event_handle_id, event_handle_generation, event_layer = (
            int(value) for value in proof_job_key
        )
        if event_handle_id < 0 or event_handle_generation < 0:
            _selector_log_f_contract_error(
                "JOB_KEY",
                "deferred proof handle and generation must be non-negative",
            )
    def _stage_gt1_meta() -> tuple[torch.Tensor, torch.Tensor]:
        return (
            _stage_meta_rows(
                meta_i32_rows,
                dtype=torch.int32,
                device=device,
                cache_owner=meta_cache_owner,
                cache_name="gt1_i32",
            ),
            _stage_meta_rows(
                meta_i64_rows,
                dtype=torch.int64,
                device=device,
                cache_owner=meta_cache_owner,
                cache_name="gt1_i64",
            ),
        )

    gt1_metadata = {
        "epoch": event_epoch,
        "layer": event_layer,
        "row_count": len(meta_i32_rows),
        "num_query_heads": int(scratch_capture_scores.shape[1]),
        "effective_kv_len_max": logical_k_max,
        "last_n_max": max(int(row[2]) for row in meta_i32_rows),
        "log_f_out_fp32": bool(prefill_out_capture_scores.dtype == torch.float32),
        "alpha": float(alpha),
        "reduce_route": reduce_route,
        "dispatch_reason": dispatch.reason,
        "resident_k_bucket": resident_k_bucket,
        "admission_identity": dispatch.admission_identity,
    }
    tiled_slot: Optional[dict[str, Any]] = None
    tiled_reservation_id: Optional[int] = None
    tiled_capacities: Optional[tuple[int, int, int]] = None
    tiled_stream: Optional[torch.cuda.Stream] = None
    if reduce_route == _LOG_F_R2_TILED_ROUTE:
        if tiled_resource_snapshot is None:
            _selector_log_f_contract_error(
                "RESOURCE_AUTHORITY",
                "tiled dispatch is missing its prepared stream proof",
            )
        tiled_stream = tiled_resource_snapshot.stream
        tiled_slot, tiled_reservation_id, n_capacity, h_capacity, k_capacity = (
            _acquire_tiled_resource_slot(
                cache_owner=meta_cache_owner,
                device=device,
                stream=tiled_stream,
                num_rows=len(meta_i32_rows),
                num_query_heads=int(scratch_capture_scores.shape[1]),
                logical_k_max=logical_k_max,
            )
        )
        assert tiled_reservation_id is not None
        tiled_capacities = (n_capacity, h_capacity, k_capacity)
        direct_plan = _TiledJobPlan(
            job=None,
            meta_i32_rows=tuple(tuple(row) for row in meta_i32_rows),
            scratch_row_indices=admission.scratch_row_indices,
            output_row_indices=admission.output_row_indices,
            scratch=scratch_capture_scores,
            output=prefill_out_capture_scores,
            denom=prefill_out_log_f_denoms,
            dispatch=dispatch,
            job_key=(-1, -1, -1),
            event_epoch=event_epoch,
            num_query_heads=int(scratch_capture_scores.shape[1]),
            logical_k_max=logical_k_max,
            capability=tuple(int(value) for value in admission.capability),
            tiled_resource_available=bool(admission.tiled_resource_available),
        )
        try:
            _validate_tiled_cohort_aliases(
                (direct_plan,), workspace=tiled_slot["workspace"]
            )
        except BaseException:
            _release_tiled_resource_slot(
                tiled_slot,
                stream=tiled_stream,
                reservation_id=tiled_reservation_id,
            )
            raise

        def _stage_gt1_meta() -> tuple[torch.Tensor, torch.Tensor]:
            assert tiled_slot is not None
            try:
                return _stage_tiled_meta_rows(
                    tiled_slot,
                    meta_i32_rows=meta_i32_rows,
                    meta_i64_rows=meta_i64_rows,
                )
            except BaseException:
                assert tiled_stream is not None
                _release_tiled_resource_slot(
                    tiled_slot,
                    stream=tiled_stream,
                    reservation_id=tiled_reservation_id,
                )
                raise

    req_meta_i32, req_meta_i64 = _profiled_postprocess_call(
        label="capture_postprocess_meta_gt1",
        metadata=gt1_metadata,
        call=_stage_gt1_meta,
    )

    def _reduce_gt1() -> None:
        if resident_k_bucket is not None:
            selector_log_s_ext.reduce_log_f_pre_scratch_r2_alpha0p5_fp16_resident_cuda(
                req_meta_i32=req_meta_i32,
                req_meta_i64=req_meta_i64,
                num_seqs=len(meta_i32_rows),
                num_query_heads=int(scratch_capture_scores.shape[1]),
                logical_k_bucket=resident_k_bucket,
            )
            return
        if reduce_route == _LOG_F_R2_TILED_ROUTE:
            assert (
                tiled_slot is not None
                and tiled_capacities is not None
                and tiled_stream is not None
                and tiled_reservation_id is not None
            )
            n_capacity, h_capacity, k_capacity = tiled_capacities
            workspace = tiled_slot["workspace"]
            try:
                selector_log_s_ext.reduce_log_f_pre_scratch_r2_alpha0p5_fp16_tiled_cuda(
                    req_meta_i32=req_meta_i32,
                    req_meta_i64=req_meta_i64,
                    workspace=workspace,
                    num_seqs=len(meta_i32_rows),
                    num_query_heads=int(scratch_capture_scores.shape[1]),
                    num_seqs_capacity=n_capacity,
                    num_query_heads_capacity=h_capacity,
                    logical_k_capacity=k_capacity,
                )
            except BaseException:
                _release_tiled_resource_slot(
                    tiled_slot,
                    stream=tiled_stream,
                    reservation_id=tiled_reservation_id,
                )
                raise
            _release_tiled_resource_slot(
                tiled_slot,
                stream=tiled_stream,
                reservation_id=tiled_reservation_id,
            )
            return
        selector_log_s_ext.reduce_log_f_pre_scratch_cuda(
            req_meta_i32=req_meta_i32,
            req_meta_i64=req_meta_i64,
            num_seqs=len(meta_i32_rows),
            num_query_heads=int(scratch_capture_scores.shape[1]),
            scratch_in_fp16=scratch_capture_scores.dtype == torch.float16,
            log_f_out_fp32=prefill_out_capture_scores.dtype == torch.float32,
            alpha=float(alpha),
        )

    _profiled_postprocess_call(
        label="capture_postprocess_reduce_gt1",
        metadata=gt1_metadata,
        call=_reduce_gt1,
    )
    if reduce_route == _LOG_F_R2_TILED_ROUTE:
        global _PROCESS_TILED_DIRECT_COUNT
        global _PROCESS_TILED_KERNEL_LAUNCH_COUNT

        _PROCESS_TILED_DIRECT_COUNT += 1
        _PROCESS_TILED_KERNEL_LAUNCH_COUNT += 4
    _record_log_f_reduce_route(
        meta_cache_owner,
        reduce_route,
        dispatch_reason=dispatch.reason,
        admission_identity=dispatch.admission_identity,
        event_epoch=event_epoch,
        event_layer=event_layer,
        event_handle_id=event_handle_id,
        event_handle_generation=event_handle_generation,
    )


def run_prefill_capture_postprocess_if_needed(
    *,
    direct_capture_phase: str = "",
    **kwargs: object,
) -> bool:
    """Run scratch postprocess only when direct capture did not fill the tape."""
    producer_rows_cpu = kwargs.get("producer_rows_cpu")
    row_capture_last_n_cpu = kwargs.get("row_capture_last_n_cpu")
    producer_rows = (
        tuple(int(v) for v in producer_rows_cpu)  # type: ignore[union-attr]
        if producer_rows_cpu is not None
        else tuple()
    )
    last_n_by_row = (
        tuple(max(0, int(v)) for v in row_capture_last_n_cpu)  # type: ignore[union-attr]
        if row_capture_last_n_cpu is not None
        else tuple()
    )
    has_cpu_truth = bool(producer_rows) and len(last_n_by_row) > max(producer_rows, default=-1)
    has_gt1 = bool(has_cpu_truth and any(int(last_n_by_row[row]) > 1 for row in producer_rows))
    kwargs_mut = dict(kwargs)
    # [RING-LASTN1-DRAIN 2026-07-03] under the per-G scratch RING the chunk-tail
    # flush must not read raw ring scratch (the slot is rewritten G*in_flight
    # layers later, depth << chunk). When the caller sets drain_lastn1_rows,
    # last_n==1 rows are NOT skipped: the dedicated lastn1 copy arm drains them
    # into the phase out tensors here (under the WAR fence in early mode / by
    # same-stream order inline) and the payload carries no raw scratch view.
    # Direct-capture prefill phases never wrote scratch, so draining there would
    # copy garbage -- the exclusion below keeps them on the tape-direct contract.
    drain_lastn1_rows = bool(kwargs_mut.pop("drain_lastn1_rows", False)) and (
        str(direct_capture_phase) != "prefill"
    )
    has_lastn1 = bool(
        has_cpu_truth and any(int(last_n_by_row[row]) == 1 for row in producer_rows)
    )
    # [CHUNKED-CAPTURE-ACCUMULATE 2026-07-06] 跨片行（accum_prev_rows>=0）必须
    # 进 gt1 reduce（含 last_n==1 尾片）：不能被 lastn1 skip 剔除，且单独构成
    # scratch_needed（如尾片 q_len==1 且批内无其它 gt1 行的步）。
    accum_rows_cpu = kwargs.get("row_capture_accum_prev_rows_cpu")
    accum_by_row = (
        tuple(int(v) for v in accum_rows_cpu)  # type: ignore[union-attr]
        if accum_rows_cpu is not None
        else tuple()
    )

    def _row_is_accum(row: int) -> bool:
        return row < len(accum_by_row) and int(accum_by_row[row]) >= 0

    has_accum = bool(
        has_cpu_truth and any(_row_is_accum(int(row)) for row in producer_rows)
    )
    if has_cpu_truth and not drain_lastn1_rows:
        kwargs_mut["skip_postprocess_rows_cpu"] = tuple(
            int(row)
            for row in producer_rows
            if int(last_n_by_row[row]) == 1 and not _row_is_accum(int(row))
        )
    scratch_needed = bool(
        has_gt1
        or has_accum
        or (drain_lastn1_rows and has_lastn1)
        or not has_cpu_truth
    )
    if not scratch_needed:
        return False
    postprocess_prefill_capture_scores(**kwargs_mut)  # type: ignore[arg-type]
    return True


def _record_capture_postprocess_tensor_on_current_stream(tensor: object) -> None:
    if not isinstance(tensor, torch.Tensor):
        return
    try:
        tensor.record_stream(torch.cuda.current_stream(device=tensor.device))
    except Exception:
        # record_stream is a lifetime guard; unsupported fake tensors in unit
        # tests should not affect the semantic job contract.
        if tensor.device.type == "cuda":
            raise


def _capture_postprocess_consumer_stream_identity(
    stream: object,
    *,
    scratch: object,
) -> tuple[str, int, int]:
    """Return a stable local target for one stream-ordered completion wait."""

    device_index = -1
    if isinstance(scratch, torch.Tensor) and scratch.device.type == "cuda":
        device_index = -1 if scratch.device.index is None else int(scratch.device.index)
    raw_stream = getattr(stream, "cuda_stream", None)
    if isinstance(raw_stream, int) and not isinstance(raw_stream, bool):
        return ("cuda_stream", device_index, int(raw_stream))
    return ("stream_object", device_index, id(stream))


def _wait_capture_postprocess_completion_event_if_needed(
    job: Any,
    *,
    waited_event_targets: Optional[dict[int, tuple[str, int, int]]] = None,
) -> bool:
    event = getattr(job, "completion_event", None)
    if event is None:
        if bool(getattr(job, "completed", False)):
            _selector_log_f_contract_error(
                "COMPLETION_EVENT",
                "completed deferred job is missing its stream completion event",
            )
        return False
    scratch = getattr(job, "scratch_capture_scores", None)
    if isinstance(scratch, torch.Tensor) and scratch.device.type == "cuda":
        stream = torch.cuda.current_stream(device=scratch.device)
    else:
        stream = torch.cuda.current_stream()
    target = _capture_postprocess_consumer_stream_identity(
        stream,
        scratch=scratch,
    )
    if waited_event_targets is not None:
        event_identity = id(event)
        previous_target = waited_event_targets.get(event_identity)
        if previous_target is not None:
            if previous_target != target:
                _selector_log_f_contract_error(
                    "COMPLETION_TARGET",
                    "one shared completion event cannot be deduplicated across "
                    f"different consumer streams: first={previous_target!r}, "
                    f"current={target!r}",
                )
            return False
    # Event waits belong to the consuming stream, not to the job.  The optional
    # registry coalesces only one sequence invocation on one proven target;
    # another selector/flush stream receives a fresh registry and its own RAW
    # edge without host synchronization.
    stream.wait_event(event)
    if waited_event_targets is not None:
        waited_event_targets[id(event)] = target
    return True


def _record_capture_postprocess_job_tensors(
    job: Any,
    *,
    scratch: torch.Tensor,
    prefill_out_capture: object,
    prefill_out_denoms: object,
) -> None:
    for tensor in (
        scratch,
        getattr(job, "producer_rows_i32", None),
        getattr(job, "row_capture_last_n_i32", None),
        getattr(job, "row_is_prefill_producer", None),
        getattr(job, "seqused_k", None),
        getattr(job, "active_capture_row_by_batch_row_i32", None),
        prefill_out_capture,
        prefill_out_denoms,
        getattr(job, "refresh_out_capture_scores", None),
        getattr(job, "refresh_out_log_f_denoms", None),
        getattr(job, "prefill_out_kv_len_per_capture_row_i32", None),
        getattr(job, "refresh_out_kv_len_per_capture_row_i32", None),
    ):
        _record_capture_postprocess_tensor_on_current_stream(tensor)


def _normalized_deferred_job_key(job: Any) -> tuple[int, int, int]:
    raw_key = getattr(job, "job_key", None)
    if not isinstance(raw_key, tuple) or len(raw_key) != 3 or any(
        isinstance(value, (bool, torch.Tensor)) for value in raw_key
    ):
        _selector_log_f_contract_error(
            "JOB_KEY",
            "deferred job key must be an immutable (handle, generation, layer) tuple",
        )
    try:
        key = tuple(int(value) for value in raw_key)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(
            "E_SELECTOR_LOG_F_JOB_KEY: deferred job key contains invalid values"
        ) from exc
    if any(value < 0 for value in key):
        _selector_log_f_contract_error(
            "JOB_KEY",
            "deferred handle, generation, and global layer must be non-negative",
        )
    return key  # type: ignore[return-value]


def _validate_payload_job_identity(payload: object, job: Any) -> tuple[int, int, int]:
    """Bind payload ownership to the immutable deferred job generation."""

    job_key = _normalized_deferred_job_key(job)
    payload_handle = getattr(payload, "capture_handle_id", None)
    payload_generation = getattr(payload, "capture_handle_generation", None)
    payload_epoch = getattr(payload, "capture_epoch", None)
    if any(
        isinstance(value, (bool, torch.Tensor))
        for value in (payload_handle, payload_generation, payload_epoch)
    ):
        _selector_log_f_contract_error(
            "PAYLOAD_IDENTITY",
            "payload handle, generation, and epoch must be host integers",
        )
    try:
        handle = int(payload_handle)
        generation = int(payload_generation)
        epoch = int(payload_epoch)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(
            "E_SELECTOR_LOG_F_PAYLOAD_IDENTITY: invalid payload identity values"
        ) from exc
    state = getattr(payload, "state", None)
    global_layer = getattr(state, "layer_index", None)
    if isinstance(global_layer, (bool, torch.Tensor)):
        _selector_log_f_contract_error(
            "PAYLOAD_IDENTITY",
            "payload state global layer must be a host integer",
        )
    try:
        global_layer = int(global_layer)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(
            "E_SELECTOR_LOG_F_PAYLOAD_IDENTITY: invalid payload state global layer"
        ) from exc
    expected_epoch = int(getattr(job, "debug_epoch", -1))
    actual = (handle, generation, global_layer)
    if actual != job_key or epoch != expected_epoch:
        _selector_log_f_contract_error(
            "PAYLOAD_IDENTITY",
            "payload ownership does not match deferred job identity: "
            f"payload={actual!r},epoch={epoch};job={job_key!r},epoch={expected_epoch}",
        )
    return job_key


def _build_tiled_job_plan(
    job: Any,
    *,
    meta_cache_owner: Optional[object],
    prepared_resource_snapshots: dict[
        torch.device, _PreparedTiledResourceSnapshot
    ],
) -> Optional[_TiledJobPlan]:
    """Return a tiled plan only when the complete valid job has no other work."""

    if job is None or bool(getattr(job, "completed", False)):
        return None
    if bool(getattr(job, "launched", False)):
        _selector_log_f_contract_error(
            "JOB_LIFECYCLE",
            "deferred job is already launched but not completed",
        )
    scratch = getattr(job, "scratch_capture_scores", None)
    output = getattr(job, "prefill_out_capture_scores", None)
    denom = getattr(job, "prefill_out_log_f_denoms", None)
    if not isinstance(scratch, torch.Tensor):
        _selector_log_f_contract_error("SCRATCH", "deferred job is missing scratch")
    if not isinstance(output, torch.Tensor) or not isinstance(denom, torch.Tensor):
        # This may be a valid refresh-only/last-n=1 job.  The normal owner below
        # performs its complete semantic validation; it is not a tiled candidate.
        return None
    _validate_prefill_postprocess_inputs(
        scratch_capture_scores=scratch,
        producer_rows_i32=getattr(job, "producer_rows_i32"),
        row_capture_last_n_i32=getattr(job, "row_capture_last_n_i32"),
        row_is_prefill_producer=getattr(job, "row_is_prefill_producer"),
        seqused_k=getattr(job, "seqused_k"),
        active_capture_row_by_batch_row_i32=getattr(
            job, "active_capture_row_by_batch_row_i32"
        ),
        prefill_out_capture_scores=output,
        prefill_out_log_f_denoms=denom,
        refresh_out_capture_scores=getattr(job, "refresh_out_capture_scores", None),
        refresh_out_log_f_denoms=getattr(job, "refresh_out_log_f_denoms", None),
    )
    batch_size = int(getattr(job, "row_capture_last_n_i32").numel())
    producer_rows = tuple(
        int(value)
        for value in _canonical_cpu_values(
            "producer_rows_cpu", getattr(job, "producer_rows_cpu", None)
        )
    )
    last_n_by_row = tuple(
        int(value)
        for value in _canonical_cpu_values(
            "row_capture_last_n_cpu",
            getattr(job, "row_capture_last_n_cpu", None),
        )
    )
    prefill_by_row = tuple(
        bool(value)
        for value in _canonical_cpu_values(
            "row_is_prefill_producer_cpu",
            getattr(job, "row_is_prefill_producer_cpu", None),
            as_bool=True,
        )
    )
    seqused_k_by_row = tuple(
        int(value)
        for value in _canonical_cpu_values(
            "seqused_k_cpu", getattr(job, "seqused_k_cpu", None)
        )
    )
    capture_row_by_row = tuple(
        int(value)
        for value in _canonical_cpu_values(
            "active_capture_row_by_batch_row_cpu",
            getattr(job, "active_capture_row_by_batch_row_cpu", None),
        )
    )
    if len(producer_rows) != int(getattr(job, "producer_rows_i32").numel()):
        _selector_log_f_contract_error(
            "CPU_AUTHORITY_SHAPE",
            "producer row truth must align with scratch metadata",
        )
    if len(set(producer_rows)) != len(producer_rows):
        _selector_log_f_contract_error(
            "CPU_AUTHORITY_VALUE", "producer rows must be unique"
        )
    for name, values in (
        ("row_capture_last_n_cpu", last_n_by_row),
        ("row_is_prefill_producer_cpu", prefill_by_row),
        ("seqused_k_cpu", seqused_k_by_row),
        ("active_capture_row_by_batch_row_cpu", capture_row_by_row),
    ):
        if len(values) != batch_size:
            _selector_log_f_contract_error(
                "CPU_AUTHORITY_SHAPE", f"{name} must align with batch size"
            )
    if any(value < 0 for value in last_n_by_row) or any(
        value < 0 for value in seqused_k_by_row
    ):
        _selector_log_f_contract_error(
            "CPU_AUTHORITY_VALUE", "row counts and K values must be non-negative"
        )
    skipped = frozenset(
        int(value)
        for value in _canonical_cpu_values(
            "skip_postprocess_rows_cpu",
            getattr(job, "skip_postprocess_rows_cpu", tuple()),
        )
    )
    raw_accum_rows = getattr(job, "row_capture_accum_prev_rows_cpu", None)
    raw_accum_capacity = getattr(job, "row_capture_accum_prev_capacity_cpu", None)
    if (raw_accum_rows is None) != (raw_accum_capacity is None):
        _selector_log_f_contract_error(
            "ACCUM_AUTHORITY", "accumulation truth must be supplied together"
        )
    accum_rows = (
        tuple(
            int(value)
            for value in _canonical_cpu_values(
                "row_capture_accum_prev_rows_cpu", raw_accum_rows
            )
        )
        if raw_accum_rows is not None
        else (-1,) * batch_size
    )
    accum_capacity = (
        tuple(
            int(value)
            for value in _canonical_cpu_values(
                "row_capture_accum_prev_capacity_cpu", raw_accum_capacity
            )
        )
        if raw_accum_capacity is not None
        else (0,) * batch_size
    )
    if len(accum_rows) != batch_size or len(accum_capacity) != batch_size:
        _selector_log_f_contract_error(
            "ACCUM_AUTHORITY", "accumulation truth must align with batch size"
        )
    output_k = tuple(
        int(value)
        for value in _canonical_cpu_values(
            "prefill_out_kv_len_per_capture_row_cpu",
            getattr(job, "prefill_out_kv_len_per_capture_row_cpu", None),
        )
    )

    meta_i32_rows: list[tuple[int, ...]] = []
    scratch_rows: list[int] = []
    output_rows: list[int] = []
    has_non_tiled_work = str(getattr(job, "direct_capture_phase", "")) == "prefill"
    all_large_k = True
    tiled_row_shape_candidate = True
    logical_k_max = 0
    for scratch_row, batch_row in enumerate(producer_rows):
        if batch_row < 0 or batch_row >= batch_size:
            _selector_log_f_contract_error(
                "CPU_AUTHORITY_VALUE", "producer row is outside batch capacity"
            )
        if batch_row in skipped:
            continue
        last_n = int(last_n_by_row[batch_row])
        logical_k = int(seqused_k_by_row[batch_row])
        if last_n <= 0 or logical_k <= 0:
            continue
        if not bool(prefill_by_row[batch_row]):
            if last_n > 1:
                _selector_log_f_contract_error(
                    "ROW_COUNT", "decode producer rows must have last_n == 1"
                )
            has_non_tiled_work = True
            continue
        capture_row = int(capture_row_by_row[batch_row])
        if capture_row < 0 or capture_row >= len(output_k):
            _selector_log_f_contract_error(
                "CPU_AUTHORITY_SHAPE",
                "prefill output K truth does not cover capture row",
            )
        if int(output_k[capture_row]) < 0:
            _selector_log_f_contract_error(
                "CPU_AUTHORITY_VALUE", "prefill output K must be non-negative"
            )
        effective_k = _resolve_phase_output_kv_len_cpu(
            kv_len=logical_k,
            capture_row=capture_row,
            out_capture_scores=output,
            out_kv_len_per_capture_row_cpu=output_k,
        )
        if effective_k <= 0:
            continue
        accum_prev = int(accum_rows[batch_row])
        flags = 8 | (16 if accum_prev >= 0 else 0)
        all_large_k = bool(
            all_large_k and effective_k > _LOG_F_R2_TILED_MIN_K_EXCLUSIVE
        )
        tiled_row_shape_candidate = bool(
            tiled_row_shape_candidate
            and (last_n == 2 or (flags == 24 and last_n == 1))
        )
        logical_k_max = max(logical_k_max, effective_k)
        meta_i32_rows.append(
            (
                effective_k,
                int(scratch.stride(1)),
                last_n,
                0,
                effective_k,
                flags,
                int(output.stride(1)),
                int(scratch.stride(2)),
                max(0, accum_prev),
                int(accum_capacity[batch_row]) if accum_prev >= 0 else 0,
            )
        )
        scratch_rows.append(scratch_row)
        output_rows.append(capture_row)
        if last_n != 2 and not (flags == 24 and last_n == 1):
            has_non_tiled_work = True
    if not meta_i32_rows:
        return None
    alpha = float(getattr(job, "alpha", 0.0) or 0.0)
    tiled_resource_available = True
    if (
        all_large_k
        and tiled_row_shape_candidate
        and _is_tiled_tensor_contract_candidate(
            scratch=scratch,
            output=output,
            denom=denom,
            alpha=alpha,
        )
    ):
        resource_snapshot = _prepared_tiled_resource_snapshot_for_device(
            snapshots=prepared_resource_snapshots,
            cache_owner=meta_cache_owner,
            device=scratch.device,
        )
        tiled_resource_available = _prepared_tiled_resource_available(
            snapshot=resource_snapshot,
            device=scratch.device,
            num_rows=len(meta_i32_rows),
            num_query_heads=int(scratch.shape[1]),
            logical_k_max=logical_k_max,
        )
    admission = _LogFReduceAdmission(
        meta_i32_rows=tuple(meta_i32_rows),
        scratch_row_indices=tuple(scratch_rows),
        output_row_indices=tuple(output_rows),
        scratch=_tensor_contract(scratch),
        output=_tensor_contract(output),
        denom=_tensor_contract(denom),
        alpha=alpha,
        capability=tuple(
            int(value) for value in torch.cuda.get_device_capability(scratch.device)
        ),
        cpu_authority_validated=True,
        local_same_device=bool(scratch.device == output.device == denom.device),
        tiled_resource_available=tiled_resource_available,
    )
    dispatch = _resolve_log_f_reduce_dispatch(admission)
    if has_non_tiled_work or dispatch.route != _LOG_F_R2_TILED_ROUTE:
        return None
    job_key = _normalized_deferred_job_key(job)
    return _TiledJobPlan(
        job=job,
        meta_i32_rows=tuple(meta_i32_rows),
        scratch_row_indices=tuple(scratch_rows),
        output_row_indices=tuple(output_rows),
        scratch=scratch,
        output=output,
        denom=denom,
        dispatch=dispatch,
        job_key=job_key,
        event_epoch=int(getattr(job, "debug_epoch", -1)),
        num_query_heads=int(scratch.shape[1]),
        logical_k_max=logical_k_max,
        capability=tuple(int(value) for value in admission.capability),
        tiled_resource_available=bool(admission.tiled_resource_available),
    )


def _revalidate_tiled_job_plan_locked(plan: _TiledJobPlan) -> _TiledJobPlan:
    """Recheck only lifecycle and retargetable state while its job lock is held."""

    job = plan.job
    if bool(getattr(job, "completed", False)) or bool(
        getattr(job, "launched", False)
    ):
        _selector_log_f_contract_error(
            "TILED_REVALIDATION", "job lifecycle changed before cohort claim"
        )
    if _normalized_deferred_job_key(job) != plan.job_key:
        _selector_log_f_contract_error(
            "TILED_REVALIDATION", "job identity changed before cohort claim"
        )
    scratch = getattr(job, "scratch_capture_scores", None)
    if scratch is not plan.scratch:
        _selector_log_f_contract_error(
            "TILED_REVALIDATION", "scratch ownership changed before cohort claim"
        )
    output = getattr(job, "prefill_out_capture_scores", None)
    denom = getattr(job, "prefill_out_log_f_denoms", None)
    if not isinstance(output, torch.Tensor) or not isinstance(denom, torch.Tensor):
        _selector_log_f_contract_error(
            "TILED_REVALIDATION", "retargeted outputs are missing"
        )
    if output is plan.output and denom is plan.denom:
        return plan

    retargeted_meta = tuple(
        (*row[:6], int(output.stride(1)), *row[7:])
        for row in plan.meta_i32_rows
    )
    admission = _LogFReduceAdmission(
        meta_i32_rows=retargeted_meta,
        scratch_row_indices=plan.scratch_row_indices,
        output_row_indices=plan.output_row_indices,
        scratch=_tensor_contract(plan.scratch),
        output=_tensor_contract(output),
        denom=_tensor_contract(denom),
        alpha=float(getattr(job, "alpha", 0.0) or 0.0),
        capability=plan.capability,
        cpu_authority_validated=True,
        local_same_device=bool(
            plan.scratch.device == output.device == denom.device
        ),
        tiled_resource_available=plan.tiled_resource_available,
    )
    dispatch = _resolve_log_f_reduce_dispatch(admission)
    if dispatch.route != _LOG_F_R2_TILED_ROUTE:
        _selector_log_f_contract_error(
            "TILED_REVALIDATION",
            f"retargeted outputs changed dispatch to {dispatch.route}",
        )
    return _TiledJobPlan(
        job=job,
        meta_i32_rows=retargeted_meta,
        scratch_row_indices=plan.scratch_row_indices,
        output_row_indices=plan.output_row_indices,
        scratch=plan.scratch,
        output=output,
        denom=denom,
        dispatch=dispatch,
        job_key=plan.job_key,
        event_epoch=plan.event_epoch,
        num_query_heads=plan.num_query_heads,
        logical_k_max=plan.logical_k_max,
        capability=plan.capability,
        tiled_resource_available=plan.tiled_resource_available,
    )


def _tiled_plan_pointer_rows(plan: _TiledJobPlan) -> tuple[tuple[int, ...], ...]:
    return tuple(
        (
            0,
            int(plan.scratch[scratch_row].data_ptr()),
            int(plan.output[output_row].data_ptr()),
            int(plan.denom[output_row].data_ptr()),
        )
        for scratch_row, output_row in zip(
            plan.scratch_row_indices,
            plan.output_row_indices,
            strict=True,
        )
    )


def _touched_interval(
    *,
    tensor: torch.Tensor,
    base_ptr: int,
    max_element_offset: int,
    label: str,
) -> _TouchedByteInterval:
    max_element_offset = int(max_element_offset)
    if max_element_offset < 0:
        raise ValueError("touched interval extent must be non-negative")
    begin = int(base_ptr)
    end = begin + (max_element_offset + 1) * int(tensor.element_size())
    return _TouchedByteInterval(
        device=tensor.device,
        begin=begin,
        end=end,
        label=label,
    )


def _tiled_plan_touched_intervals(
    plan: _TiledJobPlan,
    *,
    job_index: int,
) -> tuple[tuple[_TouchedByteInterval, ...], tuple[_TouchedByteInterval, ...]]:
    reads: list[_TouchedByteInterval] = []
    writes: list[_TouchedByteInterval] = []
    heads = int(plan.num_query_heads)
    for row_index, (meta, scratch_row, output_row) in enumerate(
        zip(
            plan.meta_i32_rows,
            plan.scratch_row_indices,
            plan.output_row_indices,
            strict=True,
        )
    ):
        logical_k = int(meta[0])
        last_n = int(meta[2])
        reads.append(
            _touched_interval(
                tensor=plan.scratch,
                base_ptr=int(plan.scratch[scratch_row].data_ptr()),
                max_element_offset=(heads - 1) * int(meta[1])
                + (last_n - 1) * int(meta[7])
                + logical_k
                - 1,
                label=f"job{job_index}:row{row_index}:scratch",
            )
        )
        writes.append(
            _touched_interval(
                tensor=plan.output,
                base_ptr=int(plan.output[output_row].data_ptr()),
                max_element_offset=(heads - 1) * int(meta[6]) + logical_k - 1,
                label=f"job{job_index}:row{row_index}:output",
            )
        )
        writes.append(
            _touched_interval(
                tensor=plan.denom,
                base_ptr=int(plan.denom[output_row].data_ptr()),
                max_element_offset=heads - 1,
                label=f"job{job_index}:row{row_index}:denom",
            )
        )
    return tuple(reads), tuple(writes)


def _byte_intervals_overlap(
    left: _TouchedByteInterval,
    right: _TouchedByteInterval,
) -> bool:
    return bool(
        left.device == right.device
        and left.begin < right.end
        and right.begin < left.end
    )


def _interval_device_key(interval: _TouchedByteInterval) -> tuple[str, int]:
    return (
        str(interval.device.type),
        -1 if interval.device.index is None else int(interval.device.index),
    )


def _first_touched_interval_overlap(
    left_intervals: Sequence[_TouchedByteInterval],
    right_intervals: Optional[Sequence[_TouchedByteInterval]] = None,
) -> Optional[tuple[_TouchedByteInterval, _TouchedByteInterval]]:
    """Address-sorted overlap proof in O(M log M), without quadratic pairs."""

    if right_intervals is None:
        ordered = sorted(
            left_intervals,
            key=lambda interval: (
                _interval_device_key(interval),
                interval.begin,
                interval.end,
            ),
        )
        active: Optional[_TouchedByteInterval] = None
        for current in ordered:
            if active is None or _interval_device_key(active) != _interval_device_key(
                current
            ):
                active = current
                continue
            if _byte_intervals_overlap(active, current):
                return active, current
            if current.end > active.end:
                active = current
        return None

    left = sorted(
        left_intervals,
        key=lambda interval: (
            _interval_device_key(interval),
            interval.begin,
            interval.end,
        ),
    )
    right = sorted(
        right_intervals,
        key=lambda interval: (
            _interval_device_key(interval),
            interval.begin,
            interval.end,
        ),
    )
    left_index = 0
    right_index = 0
    while left_index < len(left) and right_index < len(right):
        left_interval = left[left_index]
        right_interval = right[right_index]
        left_device = _interval_device_key(left_interval)
        right_device = _interval_device_key(right_interval)
        if left_device < right_device:
            left_index += 1
            continue
        if right_device < left_device:
            right_index += 1
            continue
        if _byte_intervals_overlap(left_interval, right_interval):
            return left_interval, right_interval
        if left_interval.end <= right_interval.begin:
            left_index += 1
        else:
            right_index += 1
    return None


def _validate_tiled_cohort_aliases(
    plans: Sequence[_TiledJobPlan],
    *,
    workspace: torch.Tensor,
) -> None:
    all_reads: list[_TouchedByteInterval] = []
    all_writes: list[_TouchedByteInterval] = []
    for job_index, plan in enumerate(plans):
        reads, writes = _tiled_plan_touched_intervals(plan, job_index=job_index)
        all_reads.extend(reads)
        all_writes.extend(writes)
    write_alias = _first_touched_interval_overlap(all_writes)
    if write_alias is not None:
        left, right = write_alias
        _selector_log_f_contract_error(
            "TILED_WRITE_ALIAS", f"{left.label} overlaps {right.label}"
        )
    read_write_alias = _first_touched_interval_overlap(all_writes, all_reads)
    if read_write_alias is not None:
        write, read = read_write_alias
        _selector_log_f_contract_error(
            "TILED_READ_WRITE_ALIAS", f"{write.label} overlaps {read.label}"
        )
    workspace_interval = _touched_interval(
        tensor=workspace,
        base_ptr=int(workspace.data_ptr()),
        max_element_offset=int(workspace.numel()) - 1,
        label="tiled_workspace",
    )
    workspace_alias = _first_touched_interval_overlap(
        (*all_reads, *all_writes), (workspace_interval,)
    )
    if workspace_alias is not None:
        interval, _workspace = workspace_alias
        _selector_log_f_contract_error(
            "TILED_WORKSPACE_ALIAS",
            f"{interval.label} overlaps tiled workspace",
        )


def _reject_tiled_cohort(code: str, detail: str) -> None:
    global _PROCESS_TILED_ADMISSION_FAILURE_COUNT

    _PROCESS_TILED_ADMISSION_FAILURE_COUNT += 1
    raise RuntimeError(f"E_SELECTOR_LOG_F_TILED_{code}: {detail}")


def _run_tiled_capture_postprocess_job_cohort(
    jobs: Sequence[Any],
    *,
    meta_cache_owner: Optional[object],
    initial_plans: Optional[Sequence[_TiledJobPlan]] = None,
    prepared_resource_snapshot: Optional[
        _PreparedTiledResourceSnapshot
    ] = None,
) -> int:
    """Claim one compatible job sequence and submit one four-kernel cohort."""

    global _PROCESS_TILED_COHORT_COUNT
    global _PROCESS_TILED_JOB_COUNT
    global _PROCESS_TILED_KERNEL_LAUNCH_COUNT

    jobs = tuple(jobs)
    if not jobs:
        return 0
    prepared_resource_snapshots: dict[
        torch.device, _PreparedTiledResourceSnapshot
    ] = {}
    raw_initial_plans: tuple[Optional[_TiledJobPlan], ...]
    if initial_plans is None:
        raw_initial_plans = tuple(
            _build_tiled_job_plan(
                job,
                meta_cache_owner=meta_cache_owner,
                prepared_resource_snapshots=prepared_resource_snapshots,
            )
            for job in jobs
        )
    else:
        supplied_plans = tuple(initial_plans)
        if len(supplied_plans) != len(jobs) or any(
            plan.job is not job
            for plan, job in zip(supplied_plans, jobs, strict=True)
        ):
            _reject_tiled_cohort(
                "PLAN_OWNER", "preplanned tiled jobs do not match the cohort"
            )
        raw_initial_plans = supplied_plans
    if any(plan is None for plan in raw_initial_plans):
        _reject_tiled_cohort(
            "INELIGIBLE", "cohort entrypoint requires only live tiled jobs"
        )
    plans = tuple(plan for plan in raw_initial_plans if plan is not None)
    devices = {plan.scratch.device for plan in plans}
    head_counts = {plan.num_query_heads for plan in plans}
    if len(devices) != 1 or len(head_counts) != 1:
        _reject_tiled_cohort(
            "INCOMPATIBLE", "jobs must share one device and local head count"
        )
    device = plans[0].scratch.device
    if prepared_resource_snapshot is None:
        prepared_resource_snapshot = prepared_resource_snapshots.get(device)
    if prepared_resource_snapshot is None:
        prepared_resource_snapshot = _snapshot_prepared_tiled_resource(
            cache_owner=meta_cache_owner,
            device=device,
        )
    stream = prepared_resource_snapshot.stream
    total_rows = sum(len(plan.meta_i32_rows) for plan in plans)
    logical_k_max = max(plan.logical_k_max for plan in plans)
    if not _prepared_tiled_resource_available(
        snapshot=prepared_resource_snapshot,
        device=device,
        num_rows=total_rows,
        num_query_heads=plans[0].num_query_heads,
        logical_k_max=logical_k_max,
    ):
        _reject_tiled_cohort(
            "RESOURCE",
            "prepared stream geometry does not cover the merged cohort",
        )
    slot, reservation_id, n_capacity, h_capacity, k_capacity = (
        _acquire_tiled_resource_slot(
            cache_owner=meta_cache_owner,
            device=device,
            stream=stream,
            num_rows=total_rows,
            num_query_heads=plans[0].num_query_heads,
            logical_k_max=logical_k_max,
        )
    )
    resource_published = False
    try:
        for job in jobs:
            ready_event = getattr(job, "ready_event", None)
            if ready_event is None:
                _reject_tiled_cohort("READY_EVENT", "eligible job is missing ready event")
            stream.wait_event(ready_event)
            setattr(job, "waited_ready_event", True)

        locks: list[Any] = []
        for job in jobs:
            lock = getattr(job, "lifecycle_lock", None)
            if lock is None or not callable(getattr(lock, "acquire", None)):
                _reject_tiled_cohort(
                    "LIFECYCLE_LOCK", "eligible job is missing lifecycle lock"
                )
            if all(lock is not existing for existing in locks):
                locks.append(lock)
        locks.sort(key=id)
        for lock in locks:
            lock.acquire()
        try:
            locked_plans = tuple(
                _revalidate_tiled_job_plan_locked(plan) for plan in plans
            )
            if any(
                plan.job_key != initial.job_key
                or plan.num_query_heads != plans[0].num_query_heads
                or plan.scratch.device != device
                for plan, initial in zip(locked_plans, plans, strict=True)
            ):
                _reject_tiled_cohort(
                    "REVALIDATION", "job identity, device, or head contract changed"
                )
            _validate_tiled_cohort_aliases(
                locked_plans, workspace=slot["workspace"]
            )
            locked_meta_i64_rows = tuple(
                row
                for plan in locked_plans
                for row in _tiled_plan_pointer_rows(plan)
            )
            tape_events = tuple(
                getattr(job, "tape_stack_evt", None) for job in jobs
            )
            for job in jobs:
                setattr(job, "launched", True)
        finally:
            for lock in reversed(locks):
                lock.release()

        meta_i32_rows = tuple(
            row for plan in locked_plans for row in plan.meta_i32_rows
        )
        req_meta_i32, req_meta_i64 = _stage_tiled_meta_rows(
            slot,
            meta_i32_rows=meta_i32_rows,
            meta_i64_rows=locked_meta_i64_rows,
        )
        waited_tape_events: set[int] = set()
        for job, tape_event in zip(jobs, tape_events, strict=True):
            if tape_event is not None and id(tape_event) not in waited_tape_events:
                stream.wait_event(tape_event)
                waited_tape_events.add(id(tape_event))
            setattr(job, "tape_stack_evt", None)
        for plan in locked_plans:
            _record_capture_postprocess_job_tensors(
                plan.job,
                scratch=plan.scratch,
                prefill_out_capture=plan.output,
                prefill_out_denoms=plan.denom,
            )
        workspace = slot["workspace"]

        from utils import selector_log_s_ext

        reasons = selector_log_s_ext.log_f_r2_tiled_contract_reasons(
            req_meta_i32=req_meta_i32,
            req_meta_i64=req_meta_i64,
            workspace=workspace,
            num_seqs=total_rows,
            num_query_heads=plans[0].num_query_heads,
            num_seqs_capacity=n_capacity,
            num_query_heads_capacity=h_capacity,
            logical_k_capacity=k_capacity,
        )
        if reasons:
            _reject_tiled_cohort("ABI", ",".join(reasons))
        selector_log_s_ext.reduce_log_f_pre_scratch_r2_alpha0p5_fp16_tiled_cuda(
            req_meta_i32=req_meta_i32,
            req_meta_i64=req_meta_i64,
            workspace=workspace,
            num_seqs=total_rows,
            num_query_heads=plans[0].num_query_heads,
            num_seqs_capacity=n_capacity,
            num_query_heads_capacity=h_capacity,
            logical_k_capacity=k_capacity,
        )
        completion_event = _publish_tiled_resource_completion(
            slot,
            stream=stream,
            reservation_id=reservation_id,
        )
        resource_published = True
        _PROCESS_TILED_COHORT_COUNT += 1
        _PROCESS_TILED_JOB_COUNT += len(jobs)
        _PROCESS_TILED_KERNEL_LAUNCH_COUNT += 4
        for plan in locked_plans:
            _record_log_f_reduce_route(
                meta_cache_owner,
                _LOG_F_R2_TILED_ROUTE,
                dispatch_reason=plan.dispatch.reason,
                admission_identity=plan.dispatch.admission_identity,
                event_epoch=plan.event_epoch,
                event_layer=plan.job_key[2],
                event_handle_id=plan.job_key[0],
                event_handle_generation=plan.job_key[1],
            )
        for lock in locks:
            lock.acquire()
        try:
            for job in jobs:
                setattr(job, "completion_event", completion_event)
                setattr(job, "ran_postprocess", True)
                setattr(job, "completed", True)
        finally:
            for lock in reversed(locks):
                lock.release()
        return len(jobs)
    finally:
        if not resource_published:
            # Partial GPU work remains ordered before reuse on this same stream;
            # metadata staging independently fenced its pinned source event.
            _release_tiled_resource_slot(
                slot,
                stream=stream,
                reservation_id=reservation_id,
            )


def run_capture_postprocess_job_if_needed(
    job: Any,
    *,
    meta_cache_owner: Optional[object],
    prepared_tiled_resource_snapshot: Optional[
        _PreparedTiledResourceSnapshot
    ] = None,
    completion_wait_targets: Optional[
        dict[int, tuple[str, int, int]]
    ] = None,
) -> bool:
    """Run one deferred capture postprocess job on the current CUDA stream."""
    if job is None:
        return False
    if bool(getattr(job, "completed", False)):
        _wait_capture_postprocess_completion_event_if_needed(
            job,
            waited_event_targets=completion_wait_targets,
        )
        return False
    scratch = getattr(job, "scratch_capture_scores", None)
    if not isinstance(scratch, torch.Tensor):
        raise ValueError("capture postprocess job requires scratch_capture_scores")
    ready_event = getattr(job, "ready_event", None)
    if ready_event is not None:
        torch.cuda.current_stream(device=scratch.device).wait_event(ready_event)
        setattr(job, "waited_ready_event", True)
    # [DETERMINISTIC-TAPE-WAW 2026-07-03] 与 flush 的 retarget 检查互斥
    # (TOCTOU):锁内置 launched 并快照输出目标——flush 若先进锁完成 retarget,
    # 此处读到 tape 目标+stack 完成事件;flush 若后进锁,读到 launched=True 走
    # 已发射分支(流级屏障保证其 stack 排在本 job kernels 之后)。
    _lifecycle_lock = getattr(job, "lifecycle_lock", None)
    if _lifecycle_lock is not None:
        _lifecycle_lock.acquire()
    try:
        if bool(getattr(job, "completed", False)):
            completed_while_waiting = True
            tape_stack_evt = None
            prefill_out_capture_local = None
            prefill_out_denoms_local = None
        else:
            completed_while_waiting = False
            if bool(getattr(job, "launched", False)):
                _selector_log_f_contract_error(
                    "JOB_LIFECYCLE",
                    "deferred job is already launched but has no completion event",
                )
            setattr(job, "launched", True)
            tape_stack_evt = getattr(job, "tape_stack_evt", None)
            prefill_out_capture_local = getattr(job, "prefill_out_capture_scores", None)
            prefill_out_denoms_local = getattr(job, "prefill_out_log_f_denoms", None)
    finally:
        if _lifecycle_lock is not None:
            _lifecycle_lock.release()
    if completed_while_waiting:
        _wait_capture_postprocess_completion_event_if_needed(
            job,
            waited_event_targets=completion_wait_targets,
        )
        return False
    # job 输出已被 retarget 到 flush 的私有 tape 时,必须排在 tape 的 baseline
    # stack(提交流)之后写,否则会被 stack 的 arena 旧值覆盖(WAW)。GPU 侧
    # no-op 若 stack 已完成,零热路径开销。
    if tape_stack_evt is not None:
        torch.cuda.current_stream(device=scratch.device).wait_event(tape_stack_evt)
        setattr(job, "tape_stack_evt", None)
    _record_capture_postprocess_job_tensors(
        job,
        scratch=scratch,
        prefill_out_capture=prefill_out_capture_local,
        prefill_out_denoms=prefill_out_denoms_local,
    )
    ran = run_prefill_capture_postprocess_if_needed(
        direct_capture_phase=str(getattr(job, "direct_capture_phase", "")),
        scratch_capture_scores=scratch,
        producer_rows_i32=getattr(job, "producer_rows_i32"),
        row_capture_last_n_i32=getattr(job, "row_capture_last_n_i32"),
        row_is_prefill_producer=getattr(job, "row_is_prefill_producer"),
        seqused_k=getattr(job, "seqused_k"),
        active_capture_row_by_batch_row_i32=getattr(
            job, "active_capture_row_by_batch_row_i32"
        ),
        prefill_out_capture_scores=prefill_out_capture_local,
        prefill_out_log_f_denoms=prefill_out_denoms_local,
        refresh_out_capture_scores=getattr(job, "refresh_out_capture_scores", None),
        refresh_out_log_f_denoms=getattr(job, "refresh_out_log_f_denoms", None),
        prefill_out_kv_len_per_capture_row_i32=getattr(
            job, "prefill_out_kv_len_per_capture_row_i32", None
        ),
        refresh_out_kv_len_per_capture_row_i32=getattr(
            job, "refresh_out_kv_len_per_capture_row_i32", None
        ),
        producer_rows_cpu=getattr(job, "producer_rows_cpu", None),
        row_capture_last_n_cpu=getattr(job, "row_capture_last_n_cpu", None),
        row_is_prefill_producer_cpu=getattr(
            job, "row_is_prefill_producer_cpu", None
        ),
        seqused_k_cpu=getattr(job, "seqused_k_cpu", None),
        active_capture_row_by_batch_row_cpu=getattr(
            job, "active_capture_row_by_batch_row_cpu", None
        ),
        prefill_out_kv_len_per_capture_row_cpu=getattr(
            job, "prefill_out_kv_len_per_capture_row_cpu", None
        ),
        refresh_out_kv_len_per_capture_row_cpu=getattr(
            job, "refresh_out_kv_len_per_capture_row_cpu", None
        ),
        skip_postprocess_rows_cpu=getattr(
            job, "skip_postprocess_rows_cpu", tuple()
        ),
        row_capture_accum_prev_rows_cpu=getattr(
            job, "row_capture_accum_prev_rows_cpu", None
        ),
        row_capture_accum_prev_capacity_cpu=getattr(
            job, "row_capture_accum_prev_capacity_cpu", None
        ),
        drain_lastn1_rows=bool(getattr(job, "drain_lastn1_rows", False)),
        meta_cache_owner=meta_cache_owner,
        prepared_tiled_resource_snapshot=prepared_tiled_resource_snapshot,
        alpha=float(getattr(job, "alpha", 0.0) or 0.0),
        debug_epoch=int(getattr(job, "debug_epoch", -1)),
        debug_layer_index=int(getattr(job, "debug_layer_index", -1)),
        proof_job_key=getattr(job, "job_key", None),
    )
    completion_event = torch.cuda.Event(enable_timing=False)
    completion_event.record(torch.cuda.current_stream(device=scratch.device))
    if _lifecycle_lock is not None:
        _lifecycle_lock.acquire()
    try:
        setattr(job, "completion_event", completion_event)
        setattr(job, "ran_postprocess", bool(ran))
        setattr(job, "completed", True)
    finally:
        if _lifecycle_lock is not None:
            _lifecycle_lock.release()
    return bool(ran)


def run_tiled_capture_postprocess_job_cohort(
    jobs: Sequence[Any],
    *,
    meta_cache_owner: Optional[object],
) -> int:
    """Strict bounded-owner API: one all-tiled sequence, exactly four kernels."""

    _reject_retired_selector_log_f_tp8_exact_env()
    jobs = tuple(jobs or tuple())
    if not jobs:
        return 0
    if any(job is None for job in jobs):
        _selector_log_f_contract_error(
            "TILED_COHORT", "strict cohort must not contain missing jobs"
        )
    object_ids = tuple(id(job) for job in jobs)
    if len(set(object_ids)) != len(object_ids):
        _selector_log_f_contract_error(
            "TILED_COHORT", "strict cohort jobs must be distinct objects"
        )
    keys = tuple(_normalized_deferred_job_key(job) for job in jobs)
    if len(set(keys)) != len(keys):
        _selector_log_f_contract_error(
            "JOB_KEY_ALIAS", "strict cohort jobs must have distinct identities"
        )
    return _run_tiled_capture_postprocess_job_cohort(
        jobs,
        meta_cache_owner=meta_cache_owner,
    )


def run_capture_postprocess_job_sequence(
    jobs: Sequence[Any],
    *,
    meta_cache_owner: Optional[object],
) -> int:
    """Run an ordered job sequence, coalescing maximal compatible tiled runs."""

    _reject_retired_selector_log_f_tp8_exact_env()
    unique_jobs: list[Any] = []
    seen_objects: set[int] = set()
    owner_by_key: dict[tuple[int, int, int], Any] = {}
    for job in tuple(jobs or tuple()):
        if job is None:
            continue
        object_id = id(job)
        if object_id in seen_objects:
            continue
        seen_objects.add(object_id)
        key = _normalized_deferred_job_key(job)
        previous = owner_by_key.get(key)
        if previous is not None and previous is not job:
            _selector_log_f_contract_error(
                "JOB_KEY_ALIAS",
                f"distinct deferred jobs claim the same identity {key!r}",
            )
        owner_by_key[key] = job
        unique_jobs.append(job)

    prepared_resource_snapshots: dict[
        torch.device, _PreparedTiledResourceSnapshot
    ] = {}
    plans = tuple(
        _build_tiled_job_plan(
            job,
            meta_cache_owner=meta_cache_owner,
            prepared_resource_snapshots=prepared_resource_snapshots,
        )
        for job in unique_jobs
    )
    completion_wait_targets: dict[int, tuple[str, int, int]] = {}
    ran_count = 0
    index = 0
    while index < len(unique_jobs):
        plan = plans[index]
        if plan is None:
            scratch = getattr(unique_jobs[index], "scratch_capture_scores", None)
            prepared_resource_snapshot = (
                prepared_resource_snapshots.get(scratch.device)
                if isinstance(scratch, torch.Tensor)
                else None
            )
            if run_capture_postprocess_job_if_needed(
                unique_jobs[index],
                meta_cache_owner=meta_cache_owner,
                prepared_tiled_resource_snapshot=prepared_resource_snapshot,
                completion_wait_targets=completion_wait_targets,
            ):
                ran_count += 1
            index += 1
            continue
        prepared_resource_snapshot = prepared_resource_snapshots.get(
            plan.scratch.device
        )
        if prepared_resource_snapshot is None:
            _selector_log_f_contract_error(
                "RESOURCE_AUTHORITY",
                "tiled plan is missing its call-boundary resource snapshot",
            )
        total_rows = len(plan.meta_i32_rows)
        logical_k_max = plan.logical_k_max
        end = index + 1
        while end < len(unique_jobs):
            candidate = plans[end]
            if candidate is None or (
                candidate.scratch.device != plan.scratch.device
                or candidate.num_query_heads != plan.num_query_heads
            ):
                break
            proposed_rows = total_rows + len(candidate.meta_i32_rows)
            proposed_logical_k_max = max(
                logical_k_max, candidate.logical_k_max
            )
            if not _prepared_tiled_resource_available(
                snapshot=prepared_resource_snapshot,
                device=plan.scratch.device,
                num_rows=proposed_rows,
                num_query_heads=plan.num_query_heads,
                logical_k_max=proposed_logical_k_max,
            ):
                # The general sequence owner forms maximal capacity-bounded
                # prefixes.  The strict cohort API instead rejects an
                # over-capacity group as one ownership-contract violation.
                break
            total_rows = proposed_rows
            logical_k_max = proposed_logical_k_max
            end += 1
        ran_count += _run_tiled_capture_postprocess_job_cohort(
            unique_jobs[index:end],
            meta_cache_owner=meta_cache_owner,
            initial_plans=plans[index:end],
            prepared_resource_snapshot=prepared_resource_snapshot,
        )
        index = end
    return int(ran_count)


def run_capture_postprocess_job_sequence_for_cohort(
    jobs: Sequence[Any],
    *,
    meta_cache_owner: Optional[object],
) -> tuple[int, Any]:
    """Run one ordered cohort and return its dominating terminal event.

    Cohort ownership is a stream/lifecycle contract, not a kernel-route
    contract.  A real mixed chunk may legitimately contain tiled, resident,
    generic, direct-prefill, and last-n=1 jobs.  The general sequence owner
    already preserves their order on the current stream while coalescing only
    compatible tiled runs; keep that single route implementation authoritative
    and validate the stricter cohort boundary around it.
    """

    _reject_retired_selector_log_f_tp8_exact_env()
    jobs = tuple(jobs or tuple())
    if not jobs or any(job is None for job in jobs):
        _selector_log_f_contract_error(
            "COHORT_SEQUENCE",
            "strict capture cohort requires non-empty live jobs",
        )
    object_ids = tuple(id(job) for job in jobs)
    job_keys = tuple(_normalized_deferred_job_key(job) for job in jobs)
    if len(set(object_ids)) != len(jobs) or len(set(job_keys)) != len(jobs):
        _selector_log_f_contract_error(
            "COHORT_SEQUENCE",
            "strict capture cohort jobs must have distinct owners",
        )
    if any(
        bool(getattr(job, "completed", False))
        or bool(getattr(job, "launched", False))
        or bool(getattr(job, "ran_postprocess", False))
        or getattr(job, "completion_event", None) is not None
        for job in jobs
    ):
        _selector_log_f_contract_error(
            "COHORT_SEQUENCE",
            "strict capture cohort requires fresh unlaunched jobs",
        )
    if any(
        getattr(job, "ready_event", None) is None
        or not callable(
            getattr(getattr(job, "lifecycle_lock", None), "acquire", None)
        )
        or not callable(
            getattr(getattr(job, "lifecycle_lock", None), "release", None)
        )
        for job in jobs
    ):
        _selector_log_f_contract_error(
            "COHORT_SEQUENCE",
            "strict capture cohort jobs require ready events and lifecycle locks",
        )
    first_scratch = getattr(jobs[0], "scratch_capture_scores", None)
    if not isinstance(first_scratch, torch.Tensor) or any(
        not isinstance(getattr(job, "scratch_capture_scores", None), torch.Tensor)
        or getattr(job, "scratch_capture_scores").device != first_scratch.device
        for job in jobs[1:]
    ):
        _selector_log_f_contract_error(
            "COHORT_SEQUENCE",
            "strict capture cohort must use one scratch device",
        )
    owner_stream_id = int(
        torch.cuda.current_stream(device=first_scratch.device).cuda_stream
    )
    ran_count = run_capture_postprocess_job_sequence(
        jobs,
        meta_cache_owner=meta_cache_owner,
    )
    terminal_stream_id = int(
        torch.cuda.current_stream(device=first_scratch.device).cuda_stream
    )
    if terminal_stream_id != owner_stream_id:
        _selector_log_f_contract_error(
            "COHORT_SEQUENCE",
            "strict capture cohort changed its current stream owner",
        )
    if ran_count != len(jobs):
        _selector_log_f_contract_error(
            "COHORT_SEQUENCE",
            "strict capture cohort did not launch every job",
        )
    if any(
        not bool(getattr(job, "completed", False))
        or not bool(getattr(job, "ran_postprocess", False))
        or getattr(job, "completion_event", None) is None
        for job in jobs
    ):
        _selector_log_f_contract_error(
            "COHORT_SEQUENCE",
            "strict capture cohort did not complete every ordered job",
        )
    terminal_event = getattr(jobs[-1], "completion_event", None)
    if terminal_event is None:
        _selector_log_f_contract_error(
            "COHORT_SEQUENCE",
            "strict capture cohort is missing its terminal event",
        )
    # The authoritative sequence owner preserves job order and submits every
    # route on the current stream (tiled plans freeze that same stream in their
    # prepared snapshot).  The final job's immutable event therefore dominates
    # every earlier mixed-route subrun.  Returning that existing token adds no
    # CUDA event, sync, kernel, or lock protocol.
    return int(ran_count), terminal_event


def run_capture_postprocess_jobs_for_payloads(
    payloads: Sequence[object],
    *,
    meta_cache_owner: Optional[object],
) -> int:
    """Validate payload ownership, deduplicate jobs, then run their sequence."""

    _reject_retired_selector_log_f_tp8_exact_env()
    jobs: list[Any] = []
    seen_objects: set[int] = set()
    owner_by_key: dict[tuple[int, int, int], Any] = {}
    for payload in tuple(payloads or tuple()):
        job = getattr(payload, "capture_postprocess_job", None)
        if job is None:
            continue
        key = _validate_payload_job_identity(payload, job)
        previous = owner_by_key.get(key)
        if previous is not None and previous is not job:
            _selector_log_f_contract_error(
                "JOB_KEY_ALIAS",
                f"distinct deferred jobs claim the same identity {key!r}",
            )
        owner_by_key[key] = job
        object_id = id(job)
        if object_id in seen_objects:
            continue
        seen_objects.add(object_id)
        jobs.append(job)
    return run_capture_postprocess_job_sequence(
        jobs,
        meta_cache_owner=meta_cache_owner,
    )
