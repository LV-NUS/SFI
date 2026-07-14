from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

KPI_SCOPE_SELECTED_NO_CAPTURE = "selected_no_capture_kpi"
EVIDENCE_SCOPE_SELECTED_NO_CAPTURE = "selected_no_capture_kpi"
EVIDENCE_SCOPE_NON_KPI_CAPTURE = "non_kpi_capture"
EVIDENCE_SCOPE_DIAGNOSTIC_ONLY = "diagnostic_only"

FORMAL_POLICY_DISPOSITIONS = (
    "supported",
    "route_out",
    "diagnostic_only",
    "unsupported_fail",
)

FORMAL_ROUTE_OUT_REASONS = (
    "unsupported_geometry_class",
    "unsupported_native_regime",
    "unsupported_runtime_contract",
)

FORMAL_ROUTE_OUT_TARGETS = (
    "native_dense_kv",
    "native_paged_kv",
)

DEGENERATE_GEOMETRY_POLICY_EXPECTATIONS = {
    "compact_kv_max_zero": "supported",
    "recent_visible_kv_max_zero": "supported",
    "both_zero": "supported",
}

FORMAL_DISPATCH_RULE_IDS = (
    "unsupported_fail",
    "diagnostic_only",
    "route_out",
    "supported_single_tile",
    "supported_single_persistent",
    "supported_split",
)

FORMAL_DISPATCH_V1_RULES = (
    ("unsupported_fail", "unsupported_fail"),
    ("diagnostic_only", "diagnostic_only"),
    ("route_out", "route_out"),
    ("supported_single_tile", "supported"),
    ("supported_single_persistent", "supported"),
    ("supported_split", "supported"),
)

FORMAL_TOPOLOGY_FAMILIES = (
    "compact_only",
    "recent_only",
    "dual_source",
)

FORMAL_SPLIT_REGIMES = (
    "single",
    "split",
)

FORMAL_SCHEDULER_KINDS = (
    "single_tile",
    "single_persistent",
)

FORMAL_TILE_PROFILES = (
    "sm80_single_tile",
    "sm80_single_persistent",
    "sm80_split",
)

SMALL_BATCH_TILE_COUNT_MAX = 16
SMALL_REQUEST_TILE_SPAN_MAX = 4
SPLIT_RECENT_TILES_MIN = 8
SPLIT_RECENT_RATIO_SCALE = 1024
SPLIT_RECENT_RATIO_MIN = 256
SPLIT_BALANCED_COMPACT_RECENT_TILES_MIN = 2
SPLIT_BALANCED_COMPACT_RECENT_RATIO_MIN = 256
FORMAL_CONTROL_SKELETON_FAMILY = "dual_source_native_mainloop_v1"
FRAGMENTED_ACTIVATION_SETUP_BUDGET_PCT = 15

WITNESS_GROUP_COMPACT_ONLY = "compact_only"
WITNESS_GROUP_COMPACT_RECENT = "compact_recent"
WITNESS_GROUP_RECENT_HEAVY = "recent_heavy"
WITNESS_GROUP_BALANCED_COMPACT_RECENT = "balanced_compact_recent"
WITNESS_GROUP_FRAGMENTED_RUN_PLAN = "fragmented_run_plan"
WITNESS_GROUP_DUAL_SOURCE_WORKLOAD = "dual_source_workload"
WITNESS_GROUP_HOLE_AWARE_FRAGMENTED_RUN_PLAN = "fragmented_run_plan"

FORMAL_WITNESS_TAXONOMY_FIELDS = (
    WITNESS_GROUP_COMPACT_ONLY,
    WITNESS_GROUP_COMPACT_RECENT,
    WITNESS_GROUP_RECENT_HEAVY,
    WITNESS_GROUP_DUAL_SOURCE_WORKLOAD,
    WITNESS_GROUP_HOLE_AWARE_FRAGMENTED_RUN_PLAN,
)

FORMAL_WITNESS_TAXONOMY_BALANCED_FIELDS = (
    WITNESS_GROUP_BALANCED_COMPACT_RECENT,
)

BALANCED_COMPACT_RECENT_CASE_IDS = (
    "A6_large_balanced_compact_recent",
    "B6_large_balanced_compact_recent_route",
    "P4_large_balanced_compact_recent",
)

BALANCED_COMPACT_RECENT_ARTIFACTS = (
    "artifacts/direct_large_balanced_compact_recent.json",
    "artifacts/patched_route_large_balanced_compact_recent.json",
    "artifacts/profiler_large_balanced_compact_recent.json",
)

CASE_TO_EXPECTED_WITNESS_GROUP = {
    "A1_small_compact_only": WITNESS_GROUP_COMPACT_ONLY,
    "A2_small_compact_recent": WITNESS_GROUP_COMPACT_RECENT,
    "A3_large_compact_only": WITNESS_GROUP_COMPACT_ONLY,
    "A4_large_compact_recent": WITNESS_GROUP_COMPACT_RECENT,
    "A5_large_recent_heavy": WITNESS_GROUP_RECENT_HEAVY,
    "A6_large_balanced_compact_recent": WITNESS_GROUP_BALANCED_COMPACT_RECENT,
    "A7_large_fragmented_run_plan": WITNESS_GROUP_FRAGMENTED_RUN_PLAN,
    "B1_small_compact_only_route": WITNESS_GROUP_COMPACT_ONLY,
    "B2_small_compact_recent_route": WITNESS_GROUP_COMPACT_RECENT,
    "B3_large_compact_only_route": WITNESS_GROUP_COMPACT_ONLY,
    "B4_large_compact_recent_route": WITNESS_GROUP_COMPACT_RECENT,
    "B5_large_recent_heavy_route": WITNESS_GROUP_RECENT_HEAVY,
    "B6_large_balanced_compact_recent_route": WITNESS_GROUP_BALANCED_COMPACT_RECENT,
    "B7_large_fragmented_run_plan_route": WITNESS_GROUP_FRAGMENTED_RUN_PLAN,
    "P1_small_compact_only": WITNESS_GROUP_COMPACT_ONLY,
    "P2_large_compact_recent": WITNESS_GROUP_COMPACT_RECENT,
    "P3_large_recent_heavy": WITNESS_GROUP_RECENT_HEAVY,
    "P4_large_balanced_compact_recent": WITNESS_GROUP_BALANCED_COMPACT_RECENT,
    "P5_large_fragmented_run_plan": WITNESS_GROUP_FRAGMENTED_RUN_PLAN,
}

CASE_TO_EXPECTED_CONTROL_SKELETON_FAMILY = {
    case_id: FORMAL_CONTROL_SKELETON_FAMILY for case_id in CASE_TO_EXPECTED_WITNESS_GROUP
}


@dataclass(frozen=True)
class CompactRecentLaunchConfig:
    max_seqlen_q: int
    is_causal: bool
    window_size_left: int
    window_size_right: int
    softcap: float
    cp_world_size: int
    num_splits: int
    q_v: torch.Tensor | None
    s_aux: torch.Tensor | None
    q_descale: torch.Tensor | None
    k_descale: torch.Tensor | None
    v_descale: torch.Tensor | None


@dataclass(frozen=True)
class CompactRecentWitnessCase:
    case_id: str
    case_mode: str
    route_source_mode: str
    batch_size: int
    total_k: int
    compact_kv_len: int
    recent_page_count: int
    artifact_relpath: str
    witness_group: str = ""
    control_skeleton_family: str = FORMAL_CONTROL_SKELETON_FAMILY
    fragmented_activation_setup_budget_pct: int | None = None


@dataclass(frozen=True)
class CompactRecentMatchedPagedCase:
    case_id: str
    artifact_relpath: str
    matched_baseline_kind: str
    expected_topology_family: str
    expected_split_regime: str
    expected_single_scheduler_kind: str | None
    expected_tile_profile: str


@dataclass(frozen=True)
class CompactRecentMatchedPagedRecord:
    case_id: str
    benchmark: str
    evidence_scope: str
    artifact_relpath: str
    policy_disposition: str
    normalized_launch_mode: str
    topology_family: str
    split_regime: str
    single_scheduler_kind: str | None
    tile_profile: str
    matched_baseline_kind: str
    speedup_scope: str
    speedup_vs_native_paged: float
    repeat_count: int = 0
    aggregation_kind: str = ""
    measurement_bundle_id: str = ""
    implementation_revision: str = ""
    benchmark_config_family: str = ""
    environment_toolchain_class: str = ""
    acceptance_run_id: str = ""


@dataclass(frozen=True)
class CompactRecentKpiRecord:
    case_id: str
    kpi_scope: str
    route_source_mode: str
    evidence_scope: str
    artifact_relpath: str
    policy_disposition: str
    normalized_launch_mode: str
    topology_family: str
    split_regime: str
    single_scheduler_kind: str | None
    tile_profile: str
    speedup: float
    compact_loader_mode: str | None = None
    route_out_reason: str | None = None
    route_out_target: str | None = None
    repeat_count: int = 0
    aggregation_kind: str = ""
    measurement_bundle_id: str = ""
    implementation_revision: str = ""
    benchmark_config_family: str = ""
    environment_toolchain_class: str = ""
    acceptance_run_id: str = ""
    formal_shape_family: str = ""
    formal_shape_detail: str = ""
    control_skeleton_family: str = ""
    witness_group: str = ""
    fragmented_activation_setup_budget_pct: int | None = None


@dataclass(frozen=True)
class CompactRecentProfilerWitnessCase:
    case_id: str
    expected_topology_family: str
    expected_split_regime: str
    expected_single_scheduler_kind: str | None
    expected_tile_profile: str
    route_source_mode: str
    artifact_relpath: str
    witness_group: str = ""
    control_skeleton_family: str = FORMAL_CONTROL_SKELETON_FAMILY
    fragmented_activation_setup_budget_pct: int | None = None


@dataclass(frozen=True)
class CompactRecentProfilerRecord:
    case_id: str
    kpi_scope: str
    route_source_mode: str
    evidence_scope: str
    artifact_relpath: str
    policy_disposition: str
    normalized_launch_mode: str
    topology_family: str
    split_regime: str
    single_scheduler_kind: str | None
    tile_profile: str
    compact_loader_mode: str
    resolved_num_splits: int
    combine_kernel_seen: bool = False
    compact_kv_max: int = 0
    recent_visible_kv_max: int = 0
    compact_contiguous_loader_seen: bool = False
    recent_paged_loader_seen: bool = False
    recent_prepare_events: int | None = 0
    repeated_recent_prepare_detected: bool = False
    route_out_reason: str | None = None
    route_out_target: str | None = None
    compact_phase_paged_address_path_seen: bool = False
    formal_shape_family: str = ""
    formal_shape_detail: str = ""
    control_skeleton_family: str = ""
    witness_group: str = ""
    fragmented_activation_setup_budget_pct: int | None = None


@dataclass(frozen=True)
class ActualCompactRecentLaunchInputs:
    launch_compact_base_block_i32: torch.Tensor
    launch_compact_kv_len_i32: torch.Tensor
    launch_recent_first_logical_page_i32: torch.Tensor
    launch_recent_page_count_i32: torch.Tensor
    launch_effective_k_len_i32: torch.Tensor


def build_compact_recent_host_plan_i32(
    *,
    compact_base_block: Sequence[int],
    compact_valid_tokens: Sequence[int],
    request_recent_first_logical_page: Sequence[int],
    request_recent_page_count: Sequence[int],
    effective_k_len: Sequence[int],
) -> torch.Tensor:
    row_count = len(effective_k_len)
    if not (
        len(compact_base_block)
        == len(compact_valid_tokens)
        == len(request_recent_first_logical_page)
        == len(request_recent_page_count)
        == row_count
    ):
        raise ValueError("compact_recent host plan fields must share row count")
    if row_count == 0:
        return torch.empty((0, 5), dtype=torch.int32, device="cpu")
    rows = [
        (
            int(compact_base_block[row]),
            max(0, int(compact_valid_tokens[row])),
            max(0, int(request_recent_first_logical_page[row])),
            max(0, int(request_recent_page_count[row])),
            max(0, int(effective_k_len[row])),
        )
        for row in range(row_count)
    ]
    return torch.tensor(rows, dtype=torch.int32, device="cpu").contiguous()


CASE_TO_EXPECTED_SPLIT_REGIME = {
    "A1_small_compact_only": "single",
    "A2_small_compact_recent": "single",
    "A3_large_compact_only": "split",
    "A4_large_compact_recent": "split",
    "A5_large_recent_heavy": "split",
    "A6_large_balanced_compact_recent": "split",
    "A7_large_fragmented_run_plan": "split",
    "B1_small_compact_only_route": "single",
    "B2_small_compact_recent_route": "single",
    "B3_large_compact_only_route": "split",
    "B4_large_compact_recent_route": "split",
    "B5_large_recent_heavy_route": "split",
    "B6_large_balanced_compact_recent_route": "split",
    "B7_large_fragmented_run_plan_route": "split",
}

CASE_TO_EXPECTED_SINGLE_SCHEDULER_KIND = {
    "A1_small_compact_only": "single_tile",
    "A2_small_compact_recent": "single_tile",
    "A3_large_compact_only": None,
    "A4_large_compact_recent": None,
    "A5_large_recent_heavy": None,
    "A6_large_balanced_compact_recent": None,
    "A7_large_fragmented_run_plan": None,
    "B1_small_compact_only_route": "single_tile",
    "B2_small_compact_recent_route": "single_tile",
    "B3_large_compact_only_route": None,
    "B4_large_compact_recent_route": None,
    "B5_large_recent_heavy_route": None,
    "B6_large_balanced_compact_recent_route": None,
    "B7_large_fragmented_run_plan_route": None,
}

CASE_TO_EXPECTED_SCHEDULER_KIND = CASE_TO_EXPECTED_SINGLE_SCHEDULER_KIND

CASE_TO_EXPECTED_TILE_PROFILE = {
    "A1_small_compact_only": "sm80_single_tile",
    "A2_small_compact_recent": "sm80_single_tile",
    "A3_large_compact_only": "sm80_split",
    "A4_large_compact_recent": "sm80_split",
    "A5_large_recent_heavy": "sm80_split",
    "A6_large_balanced_compact_recent": "sm80_split",
    "A7_large_fragmented_run_plan": "sm80_split",
    "B1_small_compact_only_route": "sm80_single_tile",
    "B2_small_compact_recent_route": "sm80_single_tile",
    "B3_large_compact_only_route": "sm80_split",
    "B4_large_compact_recent_route": "sm80_split",
    "B5_large_recent_heavy_route": "sm80_split",
    "B6_large_balanced_compact_recent_route": "sm80_split",
    "B7_large_fragmented_run_plan_route": "sm80_split",
}

CASE_TO_EXPECTED_TOPOLOGY_FAMILY = {
    "A1_small_compact_only": "compact_only",
    "A2_small_compact_recent": "dual_source",
    "A3_large_compact_only": "compact_only",
    "A4_large_compact_recent": "dual_source",
    "A5_large_recent_heavy": "dual_source",
    "A6_large_balanced_compact_recent": "dual_source",
    "A7_large_fragmented_run_plan": "dual_source",
    "B1_small_compact_only_route": "compact_only",
    "B2_small_compact_recent_route": "dual_source",
    "B3_large_compact_only_route": "compact_only",
    "B4_large_compact_recent_route": "dual_source",
    "B5_large_recent_heavy_route": "dual_source",
    "B6_large_balanced_compact_recent_route": "dual_source",
    "B7_large_fragmented_run_plan_route": "dual_source",
}

PROFILER_WITNESS_CASES = (
    CompactRecentProfilerWitnessCase(
        "P1_small_compact_only",
        "compact_only",
        "single",
        "single_persistent",
        "sm80_single_persistent",
        "selected/no-capture",
        "artifacts/profiler_small_compact_only.json",
        WITNESS_GROUP_COMPACT_ONLY,
        FORMAL_CONTROL_SKELETON_FAMILY,
    ),
    CompactRecentProfilerWitnessCase(
        "P2_large_compact_recent",
        "dual_source",
        "split",
        None,
        "sm80_split",
        "selected/no-capture",
        "artifacts/profiler_large_compact_recent.json",
        WITNESS_GROUP_COMPACT_RECENT,
        FORMAL_CONTROL_SKELETON_FAMILY,
    ),
    CompactRecentProfilerWitnessCase(
        "P3_large_recent_heavy",
        "dual_source",
        "split",
        None,
        "sm80_split",
        "selected/no-capture",
        "artifacts/profiler_large_recent_heavy.json",
        WITNESS_GROUP_RECENT_HEAVY,
        FORMAL_CONTROL_SKELETON_FAMILY,
    ),
    CompactRecentProfilerWitnessCase(
        "P4_large_balanced_compact_recent",
        "dual_source",
        "split",
        None,
        "sm80_split",
        "selected/no-capture",
        "artifacts/profiler_large_balanced_compact_recent.json",
        WITNESS_GROUP_BALANCED_COMPACT_RECENT,
        FORMAL_CONTROL_SKELETON_FAMILY,
    ),
    CompactRecentProfilerWitnessCase(
        "P5_large_fragmented_run_plan",
        "dual_source",
        "split",
        None,
        "sm80_split",
        "selected/no-capture",
        "artifacts/profiler_large_fragmented_run_plan.json",
        WITNESS_GROUP_FRAGMENTED_RUN_PLAN,
        FORMAL_CONTROL_SKELETON_FAMILY,
        FRAGMENTED_ACTIVATION_SETUP_BUDGET_PCT,
    ),
)

PROFILER_CASE_TO_EXPECTED_TOPOLOGY_FAMILY = {
    case.case_id: case.expected_topology_family for case in PROFILER_WITNESS_CASES
}
PROFILER_CASE_TO_EXPECTED_SPLIT_REGIME = {
    case.case_id: case.expected_split_regime for case in PROFILER_WITNESS_CASES
}
PROFILER_CASE_TO_EXPECTED_SINGLE_SCHEDULER_KIND = {
    case.case_id: case.expected_single_scheduler_kind for case in PROFILER_WITNESS_CASES
}
PROFILER_CASE_TO_EXPECTED_TILE_PROFILE = {
    case.case_id: case.expected_tile_profile for case in PROFILER_WITNESS_CASES
}

LAYER_A_WITNESS_CASES = (
    CompactRecentWitnessCase(
        "A1_small_compact_only",
        "compact_only",
        "direct",
        4,
        128,
        64,
        0,
        "artifacts/direct_small_compact_only.json",
        WITNESS_GROUP_COMPACT_ONLY,
        FORMAL_CONTROL_SKELETON_FAMILY,
    ),
    CompactRecentWitnessCase(
        "A2_small_compact_recent",
        "compact_recent",
        "direct",
        4,
        128,
        64,
        2,
        "artifacts/direct_small_compact_recent.json",
        WITNESS_GROUP_COMPACT_RECENT,
        FORMAL_CONTROL_SKELETON_FAMILY,
    ),
    CompactRecentWitnessCase(
        "A3_large_compact_only",
        "compact_only",
        "direct",
        4,
        2048,
        1568,
        0,
        "artifacts/direct_large_compact_only.json",
        WITNESS_GROUP_COMPACT_ONLY,
        FORMAL_CONTROL_SKELETON_FAMILY,
    ),
    CompactRecentWitnessCase(
        "A4_large_compact_recent",
        "compact_recent",
        "direct",
        4,
        2048,
        1568,
        32,
        "artifacts/direct_large_compact_recent.json",
        WITNESS_GROUP_COMPACT_RECENT,
        FORMAL_CONTROL_SKELETON_FAMILY,
    ),
    CompactRecentWitnessCase(
        "A5_large_recent_heavy",
        "compact_recent",
        "direct",
        4,
        2048,
        448,
        120,
        "artifacts/direct_large_recent_heavy.json",
        WITNESS_GROUP_RECENT_HEAVY,
        FORMAL_CONTROL_SKELETON_FAMILY,
    ),
    CompactRecentWitnessCase(
        "A6_large_balanced_compact_recent",
        "compact_recent",
        "direct",
        8,
        2048,
        1008,
        80,
        "artifacts/direct_large_balanced_compact_recent.json",
        WITNESS_GROUP_BALANCED_COMPACT_RECENT,
        FORMAL_CONTROL_SKELETON_FAMILY,
    ),
    CompactRecentWitnessCase(
        "A7_large_fragmented_run_plan",
        "compact_recent",
        "direct",
        16,
        2048,
        1008,
        80,
        "artifacts/direct_large_fragmented_run_plan.json",
        WITNESS_GROUP_FRAGMENTED_RUN_PLAN,
        FORMAL_CONTROL_SKELETON_FAMILY,
        FRAGMENTED_ACTIVATION_SETUP_BUDGET_PCT,
    ),
)

LAYER_B_WITNESS_CASES = (
    CompactRecentWitnessCase(
        "B1_small_compact_only_route",
        "compact_only",
        "selected/no-capture",
        4,
        128,
        64,
        0,
        "artifacts/patched_route_small_compact_only.json",
        WITNESS_GROUP_COMPACT_ONLY,
        FORMAL_CONTROL_SKELETON_FAMILY,
    ),
    CompactRecentWitnessCase(
        "B2_small_compact_recent_route",
        "compact_recent",
        "selected/no-capture",
        4,
        128,
        64,
        2,
        "artifacts/patched_route_small_compact_recent.json",
        WITNESS_GROUP_COMPACT_RECENT,
        FORMAL_CONTROL_SKELETON_FAMILY,
    ),
    CompactRecentWitnessCase(
        "B3_large_compact_only_route",
        "compact_only",
        "selected/no-capture",
        4,
        2048,
        1568,
        0,
        "artifacts/patched_route_large_compact_only.json",
        WITNESS_GROUP_COMPACT_ONLY,
        FORMAL_CONTROL_SKELETON_FAMILY,
    ),
    CompactRecentWitnessCase(
        "B4_large_compact_recent_route",
        "compact_recent",
        "selected/no-capture",
        4,
        2048,
        1568,
        32,
        "artifacts/patched_route_large_compact_recent.json",
        WITNESS_GROUP_COMPACT_RECENT,
        FORMAL_CONTROL_SKELETON_FAMILY,
    ),
    CompactRecentWitnessCase(
        "B5_large_recent_heavy_route",
        "compact_recent",
        "selected/no-capture",
        4,
        2048,
        448,
        120,
        "artifacts/patched_route_large_recent_heavy.json",
        WITNESS_GROUP_RECENT_HEAVY,
        FORMAL_CONTROL_SKELETON_FAMILY,
    ),
    CompactRecentWitnessCase(
        "B6_large_balanced_compact_recent_route",
        "compact_recent",
        "selected/no-capture",
        8,
        2048,
        1008,
        80,
        "artifacts/patched_route_large_balanced_compact_recent.json",
        WITNESS_GROUP_BALANCED_COMPACT_RECENT,
        FORMAL_CONTROL_SKELETON_FAMILY,
    ),
    CompactRecentWitnessCase(
        "B7_large_fragmented_run_plan_route",
        "compact_recent",
        "selected/no-capture",
        16,
        2048,
        1008,
        80,
        "artifacts/patched_route_large_fragmented_run_plan.json",
        WITNESS_GROUP_FRAGMENTED_RUN_PLAN,
        FORMAL_CONTROL_SKELETON_FAMILY,
        FRAGMENTED_ACTIVATION_SETUP_BUDGET_PCT,
    ),
)

MATCHED_NATIVE_PAGED_CASES = (
    CompactRecentMatchedPagedCase(
        "M1_large_compact_only_paged_match",
        "artifacts/diagnostic_only/matched_native_paged_large_compact_only.json",
        "native_paged_fa3_auto",
        "compact_only",
        "split",
        None,
        "sm80_split",
    ),
    CompactRecentMatchedPagedCase(
        "M2_large_compact_recent_paged_match",
        "artifacts/diagnostic_only/matched_native_paged_large_compact_recent.json",
        "native_paged_fa3_auto",
        "dual_source",
        "split",
        None,
        "sm80_split",
    ),
    CompactRecentMatchedPagedCase(
        "M3_large_recent_full_dominant_paged_match",
        "artifacts/diagnostic_only/matched_native_paged_large_recent_full_dominant.json",
        "native_paged_fa3_auto",
        "dual_source",
        "split",
        None,
        "sm80_split",
    ),
    CompactRecentMatchedPagedCase(
        "M4_recent_tail_partial_fill_paged_match",
        "artifacts/diagnostic_only/matched_native_paged_recent_tail_partial_fill.json",
        "native_paged_fa3_auto",
        "dual_source",
        "split",
        None,
        "sm80_split",
    ),
)

MATCHED_NATIVE_PAGED_CASE_TO_ARTIFACT = {
    case.case_id: case.artifact_relpath for case in MATCHED_NATIVE_PAGED_CASES
}




def validate_compact_recent_support_matrix(cfg: CompactRecentLaunchConfig) -> None:
    if int(cfg.max_seqlen_q) != 1:
        raise ValueError("compact_recent v1 requires max_seqlen_q == 1")
    if not bool(cfg.is_causal):
        raise ValueError("compact_recent v1 requires causal=True")
    if int(cfg.window_size_left) != -1 or int(cfg.window_size_right) != -1:
        raise ValueError("compact_recent v1 requires full causal window")
    if float(cfg.softcap) != 0.0:
        raise ValueError("compact_recent v1 requires softcap == 0.0")
    if int(cfg.cp_world_size) != 1:
        raise ValueError("compact_recent v1 requires cp_world_size == 1")
    if int(cfg.num_splits) < 0:
        raise ValueError("compact_recent v1 requires num_splits >= 0")
    if int(cfg.num_splits) > 255:
        raise ValueError("compact_recent v1 requires num_splits <= 255")
    if cfg.q_v is not None or cfg.s_aux is not None:
        raise ValueError("compact_recent v1 does not support q_v or s_aux")
    if any(x is not None for x in (cfg.q_descale, cfg.k_descale, cfg.v_descale)):
        raise ValueError("compact_recent v1 does not support descale tensors")


def compute_recent_visible_kv_len_i32(
    *,
    seqused_k: torch.Tensor,
    request_recent_first_logical_page_i32: torch.Tensor,
    request_recent_page_count_i32: torch.Tensor,
    page_size: int,
) -> torch.Tensor:
    if int(page_size) <= 0:
        raise ValueError("page_size must be positive")
    recent_first_token_i32 = request_recent_first_logical_page_i32.to(dtype=torch.int32) * int(page_size)
    visible = torch.clamp(
        seqused_k.to(dtype=torch.int32) - recent_first_token_i32,
        min=0,
    )
    max_visible = request_recent_page_count_i32.to(dtype=torch.int32) * int(page_size)
    if bool(torch.any(visible > max_visible).item()):
        raise ValueError("recent_visible_kv_len_i32 exceeds recent page span")
    return visible


def normalize_actual_compact_recent_launch_inputs(
    *,
    row_is_selected: torch.Tensor,
    compact_base_block_i32: torch.Tensor,
    compact_kv_len_i32: torch.Tensor,
    request_recent_first_logical_page_i32: torch.Tensor,
    request_recent_page_count_i32: torch.Tensor,
    canonical_real_kv_len_i32: torch.Tensor,
    page_size: int,
) -> ActualCompactRecentLaunchInputs:
    if int(page_size) <= 0:
        raise ValueError("page_size must be positive")

    selected = row_is_selected.to(dtype=torch.bool).reshape(-1)
    compact_base = compact_base_block_i32.to(dtype=torch.int32).reshape(-1)
    compact_len = compact_kv_len_i32.to(dtype=torch.int32).reshape(-1)
    recent_first = request_recent_first_logical_page_i32.to(dtype=torch.int32).reshape(-1)
    recent_count = request_recent_page_count_i32.to(dtype=torch.int32).reshape(-1)
    canonical_real = canonical_real_kv_len_i32.to(dtype=torch.int32).reshape(-1)

    if not (
        compact_base.numel()
        == compact_len.numel()
        == recent_first.numel()
        == recent_count.numel()
        == canonical_real.numel()
        == selected.numel()
    ):
        raise ValueError("actual compact_recent launch inputs must share the same row count")

    full_recent_page_count = torch.div(
        canonical_real.to(dtype=torch.int64) + int(page_size) - 1,
        int(page_size),
        rounding_mode="floor",
    ).to(dtype=torch.int32)
    launch_compact_base = torch.where(selected, compact_base, torch.zeros_like(compact_base))
    launch_compact_len = torch.where(selected, compact_len, torch.zeros_like(compact_len))
    launch_recent_first = torch.where(selected, recent_first, torch.zeros_like(recent_first))
    launch_recent_count = torch.where(selected, recent_count, full_recent_page_count)
    launch_effective = launch_compact_len + compute_recent_visible_kv_len_i32(
        seqused_k=canonical_real,
        request_recent_first_logical_page_i32=launch_recent_first,
        request_recent_page_count_i32=launch_recent_count,
        page_size=int(page_size),
    )
    return ActualCompactRecentLaunchInputs(
        launch_compact_base_block_i32=launch_compact_base,
        launch_compact_kv_len_i32=launch_compact_len,
        launch_recent_first_logical_page_i32=launch_recent_first,
        launch_recent_page_count_i32=launch_recent_count,
        launch_effective_k_len_i32=launch_effective,
    )


def validate_actual_compact_recent_launch_inputs(
    *,
    seqused_k: "torch.Tensor | None" = None,
    compact_kv_len_i32: "torch.Tensor | None" = None,
    request_recent_first_logical_page_i32: "torch.Tensor | None" = None,
    request_recent_page_count_i32: "torch.Tensor | None" = None,
    canonical_real_kv_len_i32: "torch.Tensor | None" = None,
    page_size: "int | None" = None,
    kBlockN: "int | None" = None,
) -> None:
    # Original effective-K seqused_k contract (runs when full launch tensors are provided).
    if (
        seqused_k is not None
        and compact_kv_len_i32 is not None
        and request_recent_first_logical_page_i32 is not None
        and request_recent_page_count_i32 is not None
        and canonical_real_kv_len_i32 is not None
        and page_size is not None
    ):
        expected = compact_kv_len_i32.to(dtype=torch.int32) + compute_recent_visible_kv_len_i32(
            seqused_k=canonical_real_kv_len_i32,
            request_recent_first_logical_page_i32=request_recent_first_logical_page_i32,
            request_recent_page_count_i32=request_recent_page_count_i32,
            page_size=page_size,
        )
        actual = seqused_k.to(dtype=torch.int32)
        if not torch.equal(actual, expected):
            raise ValueError("compact_recent actual launch requires effective-K seqused_k")

    # NEW: Alignment contract enforcement (host-time early fail layer;
    # the kernel-entry TORCH_CHECK is the authoritative gate for CUDA
    # graph capture/replay — see spec §4.2).
    if compact_kv_len_i32 is not None and kBlockN is not None and int(kBlockN) > 0:
        if int(compact_kv_len_i32.max().item()) > 0:
            _remainder = compact_kv_len_i32 % int(kBlockN)
            if bool((_remainder != 0).any().item()):
                bad_rows = torch.nonzero(_remainder != 0).flatten().tolist()
                bad_values = [int(compact_kv_len_i32[r].item()) for r in bad_rows[:8]]
                raise ValueError(
                    f"compact_kv_len must be a multiple of kBlockN={int(kBlockN)}; "
                    f"violating rows (up to 8 shown): {bad_rows[:8]} "
                    f"with values {bad_values}. "
                    f"Check alpha_fair.k_head or selector output; see "
                    f"docs/superpowers/specs/2026-04-19-compact-recent-"
                    f"dual-source-unified-paged-design.md §4."
                )


def validate_compact_recent_row_descriptors(
    *,
    row_consume_mode_i32: torch.Tensor,
    compact_base_block_i32: torch.Tensor,
    compact_kv_len_i32: torch.Tensor,
    request_recent_first_logical_page_i32: torch.Tensor,
    request_recent_page_count_i32: torch.Tensor,
    seqused_k: torch.Tensor,
    page_size: int,
    page_table_width: int,
    compact_logical_token_idx_i32: torch.Tensor | None = None,
) -> None:
    del seqused_k
    if int(page_size) <= 0:
        raise ValueError("page_size must be positive")
    if int(page_table_width) < 0:
        raise ValueError("page_table_width must be non-negative")

    row_mode = row_consume_mode_i32.to(dtype=torch.int32)
    full_rows = row_mode == 0
    compact_rows = row_mode == 1

    if bool(torch.any(compact_base_block_i32[full_rows] != -1).item()):
        raise ValueError("full rows require compact_base_block_i32 == -1")
    if bool(torch.any(compact_kv_len_i32[full_rows] != 0).item()):
        raise ValueError("full rows require compact_kv_len_i32 == 0")
    if bool(torch.any(request_recent_first_logical_page_i32[full_rows] != -1).item()):
        raise ValueError("full rows require request_recent_first_logical_page_i32 == -1")
    if bool(torch.any(request_recent_page_count_i32[full_rows] != 0).item()):
        raise ValueError("full rows require request_recent_page_count_i32 == 0")

    if bool(torch.any(compact_base_block_i32[compact_rows] < 0).item()):
        raise ValueError("compact rows require non-negative compact base block")
    if bool(torch.any(compact_kv_len_i32[compact_rows] < 0).item()):
        raise ValueError("compact rows require non-negative compact_kv_len_i32")
    if bool(torch.any(request_recent_first_logical_page_i32[compact_rows] < 0).item()):
        raise ValueError("compact rows require non-negative recent first page")
    if bool(torch.any(request_recent_page_count_i32[compact_rows] <= 0).item()):
        raise ValueError("compact rows require positive recent page count")
    if bool(
        torch.any(
            request_recent_first_logical_page_i32[compact_rows]
            + request_recent_page_count_i32[compact_rows]
            > int(page_table_width)
        ).item()
    ):
        raise ValueError("recent descriptor exceeds page table width")

    if compact_logical_token_idx_i32 is None or not bool(torch.any(compact_rows).item()):
        return

    tail_start = request_recent_first_logical_page_i32[compact_rows].to(dtype=torch.int32) * int(page_size)
    compact_tokens = compact_logical_token_idx_i32[compact_rows]
    overlap = compact_tokens.ge(tail_start.unsqueeze(1)) & compact_tokens.ge(0)
    if bool(torch.any(overlap).item()):
        raise ValueError("compact/recent overlap detected")


def _check_runtime_shape(
    *,
    split_regime: str,
    single_scheduler_kind: str | None,
    tile_profile: str,
    topology_family: str,
    case_id: str,
    expected_split: str | None = None,
    expected_scheduler: str | None = None,
    expected_tile_profile: str | None = None,
    expected_topology_family: str | None = None,
) -> None:
    if split_regime not in FORMAL_SPLIT_REGIMES:
        raise ValueError(f"unsupported split_regime: {split_regime}")
    if topology_family not in FORMAL_TOPOLOGY_FAMILIES:
        raise ValueError(f"unsupported topology_family: {topology_family}")
    if split_regime == "single":
        if single_scheduler_kind not in FORMAL_SCHEDULER_KINDS:
            raise ValueError(
                "single regime requires single_scheduler_kind in {'single_tile', 'single_persistent'}"
            )
        if tile_profile not in ("sm80_single_tile", "sm80_single_persistent"):
            raise ValueError("single regime requires a single tile_profile")
    else:
        if single_scheduler_kind is not None:
            raise ValueError("split regime requires single_scheduler_kind == None")
        if tile_profile != "sm80_split":
            raise ValueError("split regime requires tile_profile == sm80_split")
    if expected_split is not None and split_regime != expected_split:
        raise ValueError(f"case_id {case_id} expected split_regime {expected_split}")
    if expected_scheduler is not None and single_scheduler_kind != expected_scheduler:
        raise ValueError(
            f"case_id {case_id} expected single_scheduler_kind {expected_scheduler}"
        )
    if expected_tile_profile is not None and tile_profile != expected_tile_profile:
        raise ValueError(f"case_id {case_id} expected tile_profile {expected_tile_profile}")
    if expected_topology_family is not None and topology_family != expected_topology_family:
        raise ValueError(
            f"case_id {case_id} expected topology_family {expected_topology_family}"
        )


def _validate_formal_shape_fields(
    *,
    case_id: str,
    topology_family: str,
    split_regime: str,
    tile_profile: str,
    formal_shape_family: str,
    formal_shape_detail: str,
    control_skeleton_family: str,
    witness_group: str,
    fragmented_activation_setup_budget_pct: int | None,
) -> None:
    expected_witness_group = CASE_TO_EXPECTED_WITNESS_GROUP.get(case_id)
    if expected_witness_group is None:
        raise ValueError(f"unsupported case_id for witness registration: {case_id}")
    expected_control_skeleton_family = CASE_TO_EXPECTED_CONTROL_SKELETON_FAMILY.get(case_id)
    if expected_control_skeleton_family is None:
        raise ValueError(f"unsupported case_id for control skeleton registration: {case_id}")
    if formal_shape_family and formal_shape_family != f"{topology_family}/{split_regime}":
        raise ValueError(
            "formal_shape_family must mirror topology_family/split_regime"
        )
    if formal_shape_detail and formal_shape_detail != tile_profile:
        raise ValueError("formal_shape_detail must mirror tile_profile")
    if control_skeleton_family and control_skeleton_family != expected_control_skeleton_family:
        raise ValueError(
            f"case_id {case_id} expected control_skeleton_family {expected_control_skeleton_family}"
        )
    if witness_group and witness_group != expected_witness_group:
        raise ValueError(f"case_id {case_id} expected witness_group {expected_witness_group}")
    if fragmented_activation_setup_budget_pct is not None:
        if expected_witness_group == WITNESS_GROUP_FRAGMENTED_RUN_PLAN:
            if fragmented_activation_setup_budget_pct != FRAGMENTED_ACTIVATION_SETUP_BUDGET_PCT:
                raise ValueError(
                    "fragmented run-plan requires an explicit activation/setup budget"
                )
        else:
            raise ValueError("non-fragmented cases must not advertise an activation/setup budget")


def validate_selected_no_capture_kpi_record(record: CompactRecentKpiRecord) -> None:
    if record.kpi_scope != KPI_SCOPE_SELECTED_NO_CAPTURE:
        raise ValueError("KPI record must use selected_no_capture_kpi")
    if record.policy_disposition not in FORMAL_POLICY_DISPOSITIONS:
        raise ValueError(f"unsupported policy_disposition: {record.policy_disposition}")
    if record.policy_disposition != "supported":
        raise ValueError("KPI record must use supported policy_disposition")
    if record.evidence_scope != EVIDENCE_SCOPE_SELECTED_NO_CAPTURE:
        raise ValueError("KPI record must use selected_no_capture_kpi evidence")
    if record.normalized_launch_mode != "selected/no-capture":
        raise ValueError("KPI record must use selected/no-capture normalized_launch_mode")
    if record.artifact_relpath.startswith(
        (
            "artifacts/non_kpi_capture/",
            "artifacts/diagnostic_only/",
            "artifacts/route_out/",
        )
    ):
        raise ValueError("KPI record must reject non-supported artifact prefix")
    if record.route_out_reason is not None or record.route_out_target is not None:
        raise ValueError("KPI record must reject route_out metadata on supported surface")
    if record.route_source_mode != "selected/no-capture":
        raise ValueError("KPI record must use selected/no-capture route")
    if record.compact_loader_mode == "paged_diagnostic":
        raise ValueError("KPI record must reject paged_diagnostic compact_loader_mode")
    _validate_formal_shape_fields(
        case_id=record.case_id,
        topology_family=record.topology_family,
        split_regime=record.split_regime,
        tile_profile=record.tile_profile,
        formal_shape_family=record.formal_shape_family,
        formal_shape_detail=record.formal_shape_detail,
        control_skeleton_family=record.control_skeleton_family,
        witness_group=record.witness_group,
        fragmented_activation_setup_budget_pct=record.fragmented_activation_setup_budget_pct,
    )
    _check_runtime_shape(
        split_regime=record.split_regime,
        single_scheduler_kind=record.single_scheduler_kind,
        tile_profile=record.tile_profile,
        topology_family=record.topology_family,
        case_id=record.case_id,
        expected_split=CASE_TO_EXPECTED_SPLIT_REGIME.get(record.case_id),
        expected_scheduler=CASE_TO_EXPECTED_SINGLE_SCHEDULER_KIND.get(record.case_id),
        expected_tile_profile=CASE_TO_EXPECTED_TILE_PROFILE.get(record.case_id),
        expected_topology_family=CASE_TO_EXPECTED_TOPOLOGY_FAMILY.get(record.case_id),
    )
    if int(record.repeat_count) < 5:
        raise ValueError("repeat_count >= 5 is required")
    if record.aggregation_kind != "median":
        raise ValueError('aggregation_kind = "median" is required')
    if not record.measurement_bundle_id:
        raise ValueError("measurement_bundle_id is required")
    if not record.implementation_revision:
        raise ValueError("implementation_revision is required")
    if not record.benchmark_config_family:
        raise ValueError("benchmark_config_family is required")
    if not record.environment_toolchain_class:
        raise ValueError("environment_toolchain_class is required")
    if not record.acceptance_run_id:
        raise ValueError("acceptance_run_id is required")


def validate_matched_native_paged_record(record: CompactRecentMatchedPagedRecord) -> None:
    if record.benchmark != "compact_recent_matched_native_paged_diagnostic":
        raise ValueError(
            "matched native paged record must use compact_recent_matched_native_paged_diagnostic"
        )
    if record.evidence_scope != EVIDENCE_SCOPE_DIAGNOSTIC_ONLY:
        raise ValueError("matched native paged record must use diagnostic_only evidence")
    if record.policy_disposition != "supported":
        raise ValueError("matched native paged record must use supported policy_disposition")
    if record.normalized_launch_mode != "selected/no-capture":
        raise ValueError(
            "matched native paged record must use selected/no-capture normalized_launch_mode"
        )
    if not record.artifact_relpath.startswith("artifacts/diagnostic_only/"):
        raise ValueError("matched native paged record must use diagnostic_only artifact prefix")
    if record.artifact_relpath.startswith(
        ("artifacts/non_kpi_capture/", "artifacts/route_out/")
    ):
        raise ValueError("matched native paged record must reject non-supported artifact prefix")
    if record.matched_baseline_kind not in {
        "native_paged_fa3_auto",
        "native_paged_fa3_matched_split",
    }:
        raise ValueError(
            "matched native paged record must use native_paged_fa3_auto "
            "or native_paged_fa3_matched_split baseline"
        )
    if record.speedup_scope != "cuda_event_ms":
        raise ValueError("matched native paged record must use cuda_event_ms speedup_scope")
    _check_runtime_shape(
        split_regime=record.split_regime,
        single_scheduler_kind=record.single_scheduler_kind,
        tile_profile=record.tile_profile,
        topology_family=record.topology_family,
        case_id=record.case_id,
        expected_split=CASE_TO_EXPECTED_SPLIT_REGIME.get(
            record.case_id,
            record.split_regime,
        ),
        expected_scheduler=CASE_TO_EXPECTED_SINGLE_SCHEDULER_KIND.get(record.case_id),
        expected_tile_profile=CASE_TO_EXPECTED_TILE_PROFILE.get(record.case_id),
        expected_topology_family=CASE_TO_EXPECTED_TOPOLOGY_FAMILY.get(record.case_id),
    )
    expected_artifact = MATCHED_NATIVE_PAGED_CASE_TO_ARTIFACT.get(record.case_id)
    if expected_artifact is None:
        raise ValueError(f"unsupported matched native paged case_id: {record.case_id}")
    if record.artifact_relpath != expected_artifact:
        raise ValueError(
            f"case_id {record.case_id} expected artifact_relpath {expected_artifact}"
        )
    if int(record.repeat_count) < 5:
        raise ValueError("repeat_count >= 5 is required")
    if record.aggregation_kind != "median":
        raise ValueError('aggregation_kind = "median" is required')
    if not record.measurement_bundle_id:
        raise ValueError("measurement_bundle_id is required")
    if not record.implementation_revision:
        raise ValueError("implementation_revision is required")
    if not record.benchmark_config_family:
        raise ValueError("benchmark_config_family is required")
    if not record.environment_toolchain_class:
        raise ValueError("environment_toolchain_class is required")
    if not record.acceptance_run_id:
        raise ValueError("acceptance_run_id is required")


def validate_selected_no_capture_profiler_record(
    record: CompactRecentProfilerRecord,
) -> None:
    if record.kpi_scope != KPI_SCOPE_SELECTED_NO_CAPTURE:
        raise ValueError("profiler record must use selected_no_capture_kpi")
    if record.policy_disposition not in FORMAL_POLICY_DISPOSITIONS:
        raise ValueError(f"unsupported policy_disposition: {record.policy_disposition}")
    if record.policy_disposition != "supported":
        raise ValueError("profiler record must use supported policy_disposition")
    if record.evidence_scope != EVIDENCE_SCOPE_SELECTED_NO_CAPTURE:
        raise ValueError("profiler record must use selected_no_capture_kpi evidence")
    if record.route_source_mode != "selected/no-capture":
        raise ValueError("profiler record must use selected/no-capture route")
    if record.compact_loader_mode == "paged_diagnostic":
        raise ValueError("profiler record must reject paged_diagnostic compact_loader_mode")
    if record.compact_phase_paged_address_path_seen:
        raise ValueError("profiler record must reject compact_phase_paged_address_path_seen")
    if record.recent_prepare_events is None:
        raise ValueError("recent_prepare_events is required")
    if not 0 <= int(record.recent_prepare_events) <= 1:
        raise ValueError("recent_prepare_events must be within [0, 1] on the KPI hotpath")
    if record.repeated_recent_prepare_detected:
        raise ValueError("repeated recent preparation is forbidden on the KPI hotpath")
    if record.route_out_reason is not None or record.route_out_target is not None:
        raise ValueError("profiler record must reject route_out metadata on supported surface")
    _validate_formal_shape_fields(
        case_id=record.case_id,
        topology_family=record.topology_family,
        split_regime=record.split_regime,
        tile_profile=record.tile_profile,
        formal_shape_family=record.formal_shape_family,
        formal_shape_detail=record.formal_shape_detail,
        control_skeleton_family=record.control_skeleton_family,
        witness_group=record.witness_group,
        fragmented_activation_setup_budget_pct=record.fragmented_activation_setup_budget_pct,
    )
    _check_runtime_shape(
        split_regime=record.split_regime,
        single_scheduler_kind=record.single_scheduler_kind,
        tile_profile=record.tile_profile,
        topology_family=record.topology_family,
        case_id=record.case_id,
        expected_split=PROFILER_CASE_TO_EXPECTED_SPLIT_REGIME.get(record.case_id),
        expected_scheduler=PROFILER_CASE_TO_EXPECTED_SINGLE_SCHEDULER_KIND.get(record.case_id),
        expected_tile_profile=PROFILER_CASE_TO_EXPECTED_TILE_PROFILE.get(record.case_id),
        expected_topology_family=PROFILER_CASE_TO_EXPECTED_TOPOLOGY_FAMILY.get(record.case_id),
    )
    if int(record.resolved_num_splits) < 1:
        raise ValueError("resolved_num_splits must be at least 1")
    if record.split_regime == "split" and int(record.resolved_num_splits) <= 1:
        raise ValueError("split regime requires resolved_num_splits > 1")
    if record.split_regime == "single" and int(record.resolved_num_splits) != 1:
        raise ValueError("single regime requires resolved_num_splits == 1")
    if int(record.compact_kv_max) > 0 and not record.compact_contiguous_loader_seen:
        raise ValueError("compact_kv_max > 0 requires compact_contiguous_loader_seen")
    if int(record.recent_visible_kv_max) > 0 and not record.recent_paged_loader_seen:
        raise ValueError("recent_visible_kv_max > 0 requires recent_paged_loader_seen")
    if not record.normalized_launch_mode:
        raise ValueError("normalized_launch_mode is required")
