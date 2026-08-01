from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from numbers import Real
from typing import Any, Literal, Mapping


CAPTURE_OWNERSHIP_POLICY_SCHEMA = "sfi.fa3_capture_ownership_policy.v8"

RING_EARLY = "ring_early"
CHUNK_COHORT = "chunk_cohort"

_TILED_LAST_N = 2
_TILED_ELEMENT_BYTES = 2
_TILED_ALPHA = 0.5
_TILED_K_MIN_EXCLUSIVE = 32768
_TILED_TILE_K = 2048
_TILED_DIM_CAPACITY_MAX = 65535
_TILED_LOGICAL_K_MAX = 2147483647
_TILED_WORKSPACE_WORDS_MAX = (1 << 63) // 4 - 1
# CUDA devices supported by this release provide at least this grid-x extent.
# The CUDA entrypoint still validates the live cudaDeviceProp before launch.
_TILED_PORTABLE_GRID_X_LIMIT = 2147483647
_DEVICE_BUDGET_DIVISOR = 20

CaptureOwnershipMode = Literal["ring_early", "chunk_cohort"]


@dataclass(frozen=True, slots=True)
class CaptureOwnershipPlan:
    schema: str
    mode: CaptureOwnershipMode
    reason: str
    eligible: bool
    baseline_tiled_kernel_eligible: bool
    baseline_tiled_kernel_reason: str
    tiled_kernel_eligible: bool
    tiled_kernel_reason: str
    one_shot: bool
    async_owner_available: bool
    selector_fixed_k: bool
    last_n: int
    dtype_is_fp16: bool
    element_bytes: int
    alpha: float
    capability: tuple[int, int]
    aligned_k: int
    rows_cap: int
    heads_per_rank: int
    chunk: int
    in_flight: int
    layer_count: int
    tape_slot_count: int
    baseline_reduce_group: int
    cohort_size: int
    elements_per_slot: int
    bytes_per_slot: int
    baseline_depth: int
    target_depth: int
    selected_depth: int
    baseline_capacity_elements: int
    target_capacity_elements: int
    selected_capacity_elements: int
    baseline_bytes: int
    target_bytes: int
    selected_bytes: int
    baseline_meta_staging_device_bytes: int
    target_meta_staging_device_bytes: int
    selected_meta_staging_device_bytes: int
    baseline_tiled_postprocess_device_bytes: int
    target_tiled_postprocess_device_bytes: int
    selected_tiled_postprocess_device_bytes: int
    baseline_postprocess_device_bytes: int
    target_postprocess_device_bytes: int
    selected_postprocess_device_bytes: int
    baseline_tape_device_bytes: int
    target_tape_device_bytes: int
    selected_tape_device_bytes: int
    baseline_total_device_bytes: int
    target_total_device_bytes: int
    selected_total_device_bytes: int
    scratch_incremental_bytes: int
    incremental_bytes: int
    configured_device_bytes: int
    budget_limit_bytes: int
    signature_sha256: str

    def to_dict(self) -> dict[str, Any]:
        """Return the versioned plan in its stable public field order."""
        return {
            "schema": self.schema,
            "mode": self.mode,
            "reason": self.reason,
            "eligible": self.eligible,
            "baseline_tiled_kernel_eligible": (
                self.baseline_tiled_kernel_eligible
            ),
            "baseline_tiled_kernel_reason": self.baseline_tiled_kernel_reason,
            "tiled_kernel_eligible": self.tiled_kernel_eligible,
            "tiled_kernel_reason": self.tiled_kernel_reason,
            "one_shot": self.one_shot,
            "async_owner_available": self.async_owner_available,
            "selector_fixed_k": self.selector_fixed_k,
            "last_n": self.last_n,
            "dtype_is_fp16": self.dtype_is_fp16,
            "element_bytes": self.element_bytes,
            "alpha": self.alpha,
            "capability": list(self.capability),
            "aligned_k": self.aligned_k,
            "rows_cap": self.rows_cap,
            "heads_per_rank": self.heads_per_rank,
            "chunk": self.chunk,
            "in_flight": self.in_flight,
            "layer_count": self.layer_count,
            "tape_slot_count": self.tape_slot_count,
            "baseline_reduce_group": self.baseline_reduce_group,
            "cohort_size": self.cohort_size,
            "elements_per_slot": self.elements_per_slot,
            "bytes_per_slot": self.bytes_per_slot,
            "baseline_depth": self.baseline_depth,
            "target_depth": self.target_depth,
            "selected_depth": self.selected_depth,
            "baseline_capacity_elements": self.baseline_capacity_elements,
            "target_capacity_elements": self.target_capacity_elements,
            "selected_capacity_elements": self.selected_capacity_elements,
            "baseline_bytes": self.baseline_bytes,
            "target_bytes": self.target_bytes,
            "selected_bytes": self.selected_bytes,
            "baseline_meta_staging_device_bytes": (
                self.baseline_meta_staging_device_bytes
            ),
            "target_meta_staging_device_bytes": (
                self.target_meta_staging_device_bytes
            ),
            "selected_meta_staging_device_bytes": (
                self.selected_meta_staging_device_bytes
            ),
            "baseline_tiled_postprocess_device_bytes": (
                self.baseline_tiled_postprocess_device_bytes
            ),
            "target_tiled_postprocess_device_bytes": (
                self.target_tiled_postprocess_device_bytes
            ),
            "selected_tiled_postprocess_device_bytes": (
                self.selected_tiled_postprocess_device_bytes
            ),
            "baseline_postprocess_device_bytes": (
                self.baseline_postprocess_device_bytes
            ),
            "target_postprocess_device_bytes": self.target_postprocess_device_bytes,
            "selected_postprocess_device_bytes": (
                self.selected_postprocess_device_bytes
            ),
            "baseline_tape_device_bytes": self.baseline_tape_device_bytes,
            "target_tape_device_bytes": self.target_tape_device_bytes,
            "selected_tape_device_bytes": self.selected_tape_device_bytes,
            "baseline_total_device_bytes": self.baseline_total_device_bytes,
            "target_total_device_bytes": self.target_total_device_bytes,
            "selected_total_device_bytes": self.selected_total_device_bytes,
            "scratch_incremental_bytes": self.scratch_incremental_bytes,
            "incremental_bytes": self.incremental_bytes,
            "configured_device_bytes": self.configured_device_bytes,
            "budget_limit_bytes": self.budget_limit_bytes,
            "signature_sha256": self.signature_sha256,
        }


def _require_bool(name: str, value: object) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be bool")
    return value


def _require_non_negative_int(name: str, value: object) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be int")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _require_positive_int(name: str, value: object) -> int:
    value_int = _require_non_negative_int(name, value)
    if value_int == 0:
        raise ValueError(f"{name} must be positive")
    return value_int


def _require_alpha(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("alpha must be a real number")
    alpha = float(value)
    if not math.isfinite(alpha):
        raise ValueError("alpha must be finite")
    return alpha


def _require_capability(value: object) -> tuple[int, int]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError("capability must be a two-integer tuple")
    major, minor = value
    if type(major) is not int or type(minor) is not int:
        raise TypeError("capability must be a two-integer tuple")
    if major < 0 or minor < 0:
        raise ValueError("capability components must be non-negative")
    return major, minor


def _stable_signature(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tiled_kernel_ineligibility_reason(
    *,
    last_n: int,
    dtype_is_fp16: bool,
    element_bytes: int,
    alpha: float,
    aligned_k: int,
    num_rows: int,
    num_query_heads: int,
) -> str:
    """Mirror the portable four-stage tiled CUDA launch contract.

    The resident K-bucket kernel is an SM80 specialization; this dynamic tiled
    kernel is not.  Its admission is therefore determined by semantic shape,
    workspace arithmetic and grid extent, while the live CUDA entrypoint keeps
    the final cudaDeviceProp check.
    """

    if last_n != _TILED_LAST_N:
        return "last_n_not_two"
    if not dtype_is_fp16:
        return "dtype_not_fp16"
    if element_bytes != _TILED_ELEMENT_BYTES:
        return "element_bytes_not_two"
    if alpha != _TILED_ALPHA:
        return "alpha_not_half"
    if aligned_k <= _TILED_K_MIN_EXCLUSIVE:
        return "k_not_large"
    if num_rows <= 0 or num_query_heads <= 0:
        return "non_positive_shape"
    num_rows_capacity = 1 << (num_rows - 1).bit_length()
    tile_count = (aligned_k + _TILED_TILE_K - 1) // _TILED_TILE_K
    logical_k_capacity = tile_count * _TILED_TILE_K
    if (
        num_rows_capacity > _TILED_DIM_CAPACITY_MAX
        or num_query_heads > _TILED_DIM_CAPACITY_MAX
        or logical_k_capacity > _TILED_LOGICAL_K_MAX
    ):
        return "capacity_exceeds_kernel_abi"
    if tile_count > _TILED_DIM_CAPACITY_MAX:
        return "tile_capacity_exceeds_kernel_abi"
    workspace_words = (
        num_rows_capacity * num_query_heads * (7 * tile_count + 4)
    )
    if workspace_words > _TILED_WORKSPACE_WORDS_MAX:
        return "workspace_size_overflow"
    max_grid_x = num_rows * num_query_heads * _TILED_LAST_N * tile_count
    if max_grid_x > _TILED_PORTABLE_GRID_X_LIMIT:
        return "grid_x_exceeds_portable_limit"
    return ""


def plan_capture_ownership(
    *,
    one_shot: bool,
    async_owner_available: bool,
    selector_fixed_k: bool,
    last_n: int,
    dtype_is_fp16: bool,
    element_bytes: int,
    alpha: float,
    capability: tuple[int, int],
    aligned_k: int,
    rows_cap: int,
    heads_per_rank: int,
    chunk: int,
    in_flight: int,
    layer_count: int,
    tape_slot_count: int,
    baseline_reduce_group: int,
    baseline_meta_staging_device_bytes: int,
    target_meta_staging_device_bytes: int,
    baseline_tiled_postprocess_device_bytes: int,
    target_tiled_postprocess_device_bytes: int,
    baseline_tape_device_bytes: int,
    target_tape_device_bytes: int,
    configured_device_bytes: int,
) -> CaptureOwnershipPlan:
    """Choose one immutable ownership plan from geometry and memory facts only."""
    one_shot = _require_bool("one_shot", one_shot)
    async_owner_available = _require_bool(
        "async_owner_available", async_owner_available
    )
    selector_fixed_k = _require_bool("selector_fixed_k", selector_fixed_k)
    dtype_is_fp16 = _require_bool("dtype_is_fp16", dtype_is_fp16)
    last_n = _require_non_negative_int("last_n", last_n)
    element_bytes = _require_positive_int("element_bytes", element_bytes)
    alpha = _require_alpha(alpha)
    capability = _require_capability(capability)
    aligned_k = _require_non_negative_int("aligned_k", aligned_k)
    rows_cap = _require_non_negative_int("rows_cap", rows_cap)
    heads_per_rank = _require_non_negative_int("heads_per_rank", heads_per_rank)
    chunk = _require_non_negative_int("chunk", chunk)
    in_flight = _require_non_negative_int("in_flight", in_flight)
    layer_count = _require_non_negative_int("layer_count", layer_count)
    tape_slot_count = _require_non_negative_int(
        "tape_slot_count", tape_slot_count
    )
    baseline_reduce_group = _require_positive_int(
        "baseline_reduce_group", baseline_reduce_group
    )
    baseline_meta_staging_device_bytes = _require_non_negative_int(
        "baseline_meta_staging_device_bytes",
        baseline_meta_staging_device_bytes,
    )
    target_meta_staging_device_bytes = _require_non_negative_int(
        "target_meta_staging_device_bytes",
        target_meta_staging_device_bytes,
    )
    baseline_tiled_postprocess_device_bytes = _require_non_negative_int(
        "baseline_tiled_postprocess_device_bytes",
        baseline_tiled_postprocess_device_bytes,
    )
    target_tiled_postprocess_device_bytes = _require_non_negative_int(
        "target_tiled_postprocess_device_bytes",
        target_tiled_postprocess_device_bytes,
    )
    baseline_tape_device_bytes = _require_non_negative_int(
        "baseline_tape_device_bytes", baseline_tape_device_bytes
    )
    target_tape_device_bytes = _require_non_negative_int(
        "target_tape_device_bytes", target_tape_device_bytes
    )
    configured_device_bytes = _require_positive_int(
        "configured_device_bytes", configured_device_bytes
    )

    elements_per_slot = rows_cap * heads_per_rank * last_n * aligned_k
    bytes_per_slot = elements_per_slot * element_bytes
    # in_flight sizes short-lived capture scratch only.  Deferred tape lifetime
    # is request-owned and independently bounded by tape_slot_count.
    target_depth = chunk * in_flight
    baseline_depth = baseline_reduce_group * in_flight
    baseline_capacity_elements = baseline_depth * elements_per_slot
    target_capacity_elements = target_depth * elements_per_slot
    baseline_bytes = baseline_depth * bytes_per_slot
    target_bytes = target_depth * bytes_per_slot
    baseline_tiled_kernel_reason = _tiled_kernel_ineligibility_reason(
        last_n=last_n,
        dtype_is_fp16=dtype_is_fp16,
        element_bytes=element_bytes,
        alpha=alpha,
        aligned_k=aligned_k,
        num_rows=rows_cap,
        num_query_heads=heads_per_rank,
    )
    # A cohort launch coalesces every layer-row pair into the tiled N dimension.
    # Validate that actual target geometry, not the ring's smaller N, against the
    # same CUDA ABI and grid arithmetic used by the four-stage kernel.
    tiled_kernel_reason = _tiled_kernel_ineligibility_reason(
        last_n=last_n,
        dtype_is_fp16=dtype_is_fp16,
        element_bytes=element_bytes,
        alpha=alpha,
        aligned_k=aligned_k,
        num_rows=rows_cap * chunk,
        num_query_heads=heads_per_rank,
    )
    baseline_tiled_kernel_eligible = not baseline_tiled_kernel_reason
    tiled_kernel_eligible = not tiled_kernel_reason
    if not baseline_tiled_kernel_eligible:
        baseline_tiled_postprocess_device_bytes = 0
    if not tiled_kernel_eligible:
        target_tiled_postprocess_device_bytes = 0
    baseline_postprocess_device_bytes = (
        baseline_meta_staging_device_bytes
        + baseline_tiled_postprocess_device_bytes
    )
    target_postprocess_device_bytes = (
        target_meta_staging_device_bytes
        + target_tiled_postprocess_device_bytes
    )

    scratch_incremental_bytes = max(0, target_bytes - baseline_bytes)
    baseline_total_device_bytes = (
        baseline_bytes
        + baseline_postprocess_device_bytes
        + baseline_tape_device_bytes
    )
    target_total_device_bytes = (
        target_bytes + target_postprocess_device_bytes + target_tape_device_bytes
    )
    incremental_bytes = max(
        0, target_total_device_bytes - baseline_total_device_bytes
    )
    budget_limit_bytes = configured_device_bytes // _DEVICE_BUDGET_DIVISOR

    ineligibility_reason = ""
    if not one_shot:
        ineligibility_reason = "not_one_shot"
    elif not async_owner_available:
        ineligibility_reason = "async_owner_unavailable"
    elif not selector_fixed_k:
        ineligibility_reason = "selector_fixed_k_disabled"
    elif tiled_kernel_reason:
        ineligibility_reason = tiled_kernel_reason
    elif (
        chunk <= 0
        or in_flight <= 0
        or layer_count <= 0
        or tape_slot_count <= 0
        or target_tape_device_bytes <= 0
    ):
        ineligibility_reason = "non_positive_shape"

    eligible = not ineligibility_reason
    if not eligible:
        mode: CaptureOwnershipMode = RING_EARLY
        reason = ineligibility_reason
    elif incremental_bytes > budget_limit_bytes:
        mode = RING_EARLY
        reason = "incremental_bytes_exceed_budget"
    else:
        mode = CHUNK_COHORT
        reason = "eligible_within_budget"

    selected_depth = target_depth if mode == CHUNK_COHORT else baseline_depth
    selected_capacity_elements = selected_depth * elements_per_slot
    selected_bytes = selected_depth * bytes_per_slot
    selected_postprocess_device_bytes = (
        target_postprocess_device_bytes
        if mode == CHUNK_COHORT
        else baseline_postprocess_device_bytes
    )
    selected_meta_staging_device_bytes = (
        target_meta_staging_device_bytes
        if mode == CHUNK_COHORT
        else baseline_meta_staging_device_bytes
    )
    selected_tiled_postprocess_device_bytes = (
        target_tiled_postprocess_device_bytes
        if mode == CHUNK_COHORT
        else baseline_tiled_postprocess_device_bytes
    )
    selected_tape_device_bytes = (
        target_tape_device_bytes
        if mode == CHUNK_COHORT
        else baseline_tape_device_bytes
    )
    selected_total_device_bytes = (
        selected_bytes
        + selected_postprocess_device_bytes
        + selected_tape_device_bytes
    )

    signature_payload = {
        "schema": CAPTURE_OWNERSHIP_POLICY_SCHEMA,
        "mode": mode,
        "reason": reason,
        "eligible": eligible,
        "baseline_tiled_kernel_eligible": baseline_tiled_kernel_eligible,
        "baseline_tiled_kernel_reason": baseline_tiled_kernel_reason,
        "tiled_kernel_eligible": tiled_kernel_eligible,
        "tiled_kernel_reason": tiled_kernel_reason,
        "one_shot": one_shot,
        "async_owner_available": async_owner_available,
        "selector_fixed_k": selector_fixed_k,
        "last_n": last_n,
        "dtype_is_fp16": dtype_is_fp16,
        "element_bytes": element_bytes,
        "alpha": alpha,
        "capability": capability,
        "aligned_k": aligned_k,
        "rows_cap": rows_cap,
        "heads_per_rank": heads_per_rank,
        "chunk": chunk,
        "in_flight": in_flight,
        "layer_count": layer_count,
        "tape_slot_count": tape_slot_count,
        "baseline_reduce_group": baseline_reduce_group,
        "cohort_size": chunk,
        "elements_per_slot": elements_per_slot,
        "bytes_per_slot": bytes_per_slot,
        "baseline_depth": baseline_depth,
        "target_depth": target_depth,
        "selected_depth": selected_depth,
        "baseline_capacity_elements": baseline_capacity_elements,
        "target_capacity_elements": target_capacity_elements,
        "selected_capacity_elements": selected_capacity_elements,
        "baseline_bytes": baseline_bytes,
        "target_bytes": target_bytes,
        "selected_bytes": selected_bytes,
        "baseline_meta_staging_device_bytes": baseline_meta_staging_device_bytes,
        "target_meta_staging_device_bytes": target_meta_staging_device_bytes,
        "selected_meta_staging_device_bytes": selected_meta_staging_device_bytes,
        "baseline_tiled_postprocess_device_bytes": (
            baseline_tiled_postprocess_device_bytes
        ),
        "target_tiled_postprocess_device_bytes": (
            target_tiled_postprocess_device_bytes
        ),
        "selected_tiled_postprocess_device_bytes": (
            selected_tiled_postprocess_device_bytes
        ),
        "baseline_postprocess_device_bytes": baseline_postprocess_device_bytes,
        "target_postprocess_device_bytes": target_postprocess_device_bytes,
        "selected_postprocess_device_bytes": selected_postprocess_device_bytes,
        "baseline_tape_device_bytes": baseline_tape_device_bytes,
        "target_tape_device_bytes": target_tape_device_bytes,
        "selected_tape_device_bytes": selected_tape_device_bytes,
        "baseline_total_device_bytes": baseline_total_device_bytes,
        "target_total_device_bytes": target_total_device_bytes,
        "selected_total_device_bytes": selected_total_device_bytes,
        "scratch_incremental_bytes": scratch_incremental_bytes,
        "incremental_bytes": incremental_bytes,
        "configured_device_bytes": configured_device_bytes,
        "budget_limit_bytes": budget_limit_bytes,
    }
    signature_sha256 = _stable_signature(signature_payload)

    return CaptureOwnershipPlan(
        **signature_payload,
        signature_sha256=signature_sha256,
    )
