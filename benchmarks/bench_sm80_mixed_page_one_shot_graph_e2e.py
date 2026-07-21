from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.bench_sm80_mixed_page_full_cudagraph_phase1 import (
    DEFAULT_FA3_UPSTREAM_ROOT,
    DEFAULT_MODEL,
    DEFAULT_OUT_ATOL,
    DEFAULT_OUT_RTOL,
    DEFAULT_PROMPT,
    DEFAULT_STAGE_A_RECENT,
    DEFAULT_PYTHON,
    DEFAULT_TIMEOUT_S,
    DEFAULT_LSE_ATOL,
    DEFAULT_LSE_RTOL,
    GATE_FAILURE_EXIT_CODE,
    Phase1CommandResult,
    _build_dense_reference_command,
    _build_dense_reference_env,
    _build_env as _build_phase1_env,
    _build_smoke_command,
    _classify_failure,
    _default_decode_metrics_path,
    _default_dense_outputs_path,
    _default_route_trace_path,
    _file_sha256,
    _as_float,
    _as_int,
    _read_json,
    _read_trace_events,
    _resolve_fa3_so_path,
    _RouteTraceEvents,
    _route_summary,
    _route_proof_payload,
    _run_pair_config,
    _run_command,
    _semantic_output_diffs,
    _sparse_controller_payload,
    _tail,
)
from benchmarks.scheduler_contract import (
    CHUNKED_PREFILL_MODES,
    DEFAULT_MAX_NUM_BATCHED_TOKENS,
    DEFAULT_MAX_SEQ_LEN_TO_CAPTURE,
    requested_scheduler_graph_contract,
    resolve_benchmark_max_num_seqs,
    scheduler_graph_runtime_contract_from_metrics,
    validate_scheduler_graph_args,
)
from benchmarks.decode_throughput_window import (
    cudagraph_runtime_observer_proof_reasons,
)
from utils.selector_cache_identity import selector_cache_abi_key_for_python
from utils.model_kv_contract import MODEL_KV_CONTRACT_SCHEMA
from patches.sparse_types import ASYNC_PRODUCER_GPU_PROFILE_STAGES
from benchmarks.sm80_run_pair import (
    ALLOWED_TRACE_ENV_KEYS,
    FULL_CUDAGRAPH_HOOK_PROFILE_ENV_KEY,
    NATIVE_CANONICAL_PAGES_KEY,
    ROUTE_TRACE_ENV_KEY,
    RouteProofResult,
    SPEED_CHILD_PAIRING_IDENTITY_ENV_KEYS,
    STAGE_A_SOURCE_COUNTER_SCHEMA_VERSION,
    _stage_a_source_counter_missing_fields,
    build_pairing_digest,
    build_config_digest,
    classify_speed_child_env_key,
    validate_shared_route_proof,
)
from benchmarks.sm80_refresh_workload_plan import (
    WORKLOAD_PLAN_REPLAY_ENV,
    build_workload_plan_from_route_events,
    load_workload_plan,
    workload_plan_digest,
)
from scripts.check_tp8_arm_teardown import (
    ARM_TOKEN_ENV,
    arm_token_sha256,
    capture_baseline as capture_tp8_process_baseline,
    capture_teardown as capture_tp8_arm_teardown,
    derive_arm_token,
    write_evidence as write_tp8_process_evidence,
)


DEFAULT_COMPACT_BLOCKS_PER_SLOT = 128
DEFAULT_MAX_LIVE_SPARSE_SLOTS = 8
DEFERRED_BRIDGE_ENV_KEYS = (
    "VLLM_SPARSE_DEFER_BOOTSTRAP_PRODUCER",
    "VLLM_SPARSE_BOOTSTRAP_BRIDGE_MAX_TOKENS",
    "VLLM_SPARSE_BOOTSTRAP_BRIDGE_GRAPH_POLICY",
    "VLLM_SPARSE_DEFERRED_PRODUCER_GROUPS_PER_STEP",
)
SELECTOR_PIPELINE_ARTIFACT_PREFIX = "SFI_SELECTOR_PIPELINE_ARTIFACT="

SELECTOR_GPU_PROFILE_STAGES = tuple(
    stage
    for stage in ASYNC_PRODUCER_GPU_PROFILE_STAGES
    if stage not in {"body", "writer"}
)


SELECTOR_PIPELINE_CPU_PROFILE_ENV_KEYS = (
    "VLLM_SPARSE_PIPELINE_CPU_PROFILE",
    "VLLM_SPARSE_PIPELINE_CPU_PROFILE_LOG",
    "VLLM_SPARSE_PIPELINE_CPU_PROFILE_OUTLIER_US",
    "VLLM_SPARSE_DEFERRED_SELECTOR_PROFILE_DETAIL",
    "VLLM_SPARSE_REPLAY_REFRESH_ENQUEUE_PROFILE_DETAIL",
    "VLLM_SPARSE_ASYNC_PRODUCER_GPU_PROFILE",
)
DEFERRED_BRIDGE_GRAPH_POLICIES = ("evict_recapture_once",)
GT1_SELECTOR_TORCH_EXTENSIONS_ROOT = _REPO_ROOT / "tmp" / "torch_extensions"
DEFAULT_GT1_SELECTOR_PREWARM_TIMEOUT_S = 600
BS2_LONG_CAP128_PRESET = "bs2long-cap128"
BS2_LONG_CAP128_PROMPT = "benchmarks/needle_prompt_two_parts.txt"
BS2_LONG_CAP128_BATCH_SIZE = 2
BS2_LONG_CAP128_MAX_NEW_TOKENS = 128
BS2_LONG_CAP128_GPU_MEM_UTIL = 0.4
BS2_LONG_CAP128_FA4_SM100_GPU_MEM_UTIL = 0.95
BS2_LONG_CAP128_REFRESH_INTERVAL = 96
BACKEND_FA3 = "fa3"
BACKEND_FA4_SM100 = "fa4-sm100"
BACKEND_CHOICES = (BACKEND_FA3, BACKEND_FA4_SM100)
HUGE_REFRESH_INTERVAL = 1_000_000_000
PRODUCER_MODE_LEGACY = "legacy"
PRODUCER_MODE_BASELINE = "baseline-lastn1"
PRODUCER_MODE_REFRESH = "refresh-on"
PRODUCER_MODE_TRIGGER = "refresh-trigger-on"
PRODUCER_MODE_FULL_OPEN = "full-open-gt1"
PRODUCER_MODES = (
    PRODUCER_MODE_LEGACY,
    PRODUCER_MODE_BASELINE,
    PRODUCER_MODE_REFRESH,
    PRODUCER_MODE_TRIGGER,
    PRODUCER_MODE_FULL_OPEN,
)
DECODE_TAIL_CORRELATE_ACTION = (
    "correlate_decode_outliers_with_pending_rebuild_deadlines"
)
DECODE_TAIL_PENDING_DRAIN_ACTION = "optimize_pending_rebuild_drain_around_consumer_step"
DECODE_TAIL_NON_REFRESH_ACTION = (
    "split_non_refresh_decode_outliers_before_scheduler_change"
)
DECODE_TAIL_SPLIT_POST_TAIL_ACTION = "split_decode_post_tail_from_in_loop_jitter"
DECODE_TAIL_ENABLE_ACTION = "enable_decode_step_duration_metrics"
DECODE_TAIL_OUTLIER_MIN_EXTRA_US = 500.0
DECODE_TAIL_OUTLIER_MIN_RATIO = 1.25
DECODE_TAIL_PRODUCER_STEP_NEAR_DELTA = 2
DECODE_TAIL_TOP_OUTLIER_LIMIT = 8


@dataclass(frozen=True)
class Phase2OneShotGraphRecord:
    case: str
    one_shot_bootstrap_only: bool
    continuous_producer_enabled: bool | None
    producer_mode: str
    selector_runs: int
    rebuild_runs: int
    continuous_refresh_reqs: int
    refresh_payloads: int
    sentence_trigger_intents: int
    row_mode_distribution: dict[str, int]
    row_source_distribution: dict[str, int]
    one_shot_rebuild_us: float
    prefill_to_first_decode_us: float
    steady_full_graph_replay_us: float
    carrier_update_us: float
    reference_max_abs_diff: float
    reference_semantic_match: bool | None
    reference_quality_reasons: list[str]
    out_atol: float
    out_rtol: float
    lse_atol: float
    lse_rtol: float
    gate_passed: bool
    run_pair_id: str = ""
    config_digest: str = ""
    route_proof_passed: bool = False
    route_proof_reasons: list[str] = field(default_factory=list)
    refresh_reason_counts: dict[str, int] = field(default_factory=dict)
    refresh_trigger_intents: int = -1
    interval_trigger_intents: int = -1
    expected_interval_trigger_intents: int = -1
    interval_trigger_requirement_ok: bool = True
    sentence_trigger_observation_required: bool = False
    gt1_gate_scope: str = ""
    prefill_last_n_gt1_requested: bool = True
    first_decode_wait_us: float = -1.0
    blocked_by_unready_request_us: float = -1.0
    prefill_slowdown_us: float = -1.0
    aux_stream_enabled: bool | None = None
    producer_total_us: float = -1.0
    visible_wait_us: float = -1.0
    bridge_token_count: int = -1
    bridge_decode_p50_us: float = -1.0
    bridge_decode_p95_us: float = -1.0
    producer_launch_step: int = -1
    producer_ready_step: int = -1
    bridge_added_decode_cost_us: float = -1.0
    saved_visible_wait_us: float = -1.0
    captured_route_family: str = ""
    current_route_family: str = ""
    route_family_mismatch: bool | None = None
    bridge_phase_route_summary: dict[str, object] = field(default_factory=dict)
    post_switch_phase_route_summary: dict[str, object] = field(default_factory=dict)
    bridge_token_positions_by_request: dict[str, object] = field(default_factory=dict)
    bridge_token_positions_exact_once: bool | None = None
    compact_middle_excludes_bridge_positions: bool | None = None
    eos_before_ready_seen: bool | None = None
    bootstrap_full_kv_handoff: bool = False
    commit_publish_us: float = -1.0
    producer_overlap_us: float = -1.0
    producer_deadline_wait_us: float = -1.0
    capture_tap_visible_ms_or_unavailable_reason: str = "unavailable:not_measured"
    capture_postprocess_or_reduce_ms: float = -1.0
    lastn1_direct_count: int = -1
    gt1_reduce_count: int = -1
    gt1_scalar_fallback_count: int = -1
    selector_gpu_ms: float = -1.0
    selector_launch_count: int = -1
    pack_rebuild_gpu_ms: float = -1.0
    prefill_selector_gpu_ms: float = -1.0
    prefill_rebuild_gpu_ms: float = -1.0
    prefill_publish_cpu_us: float = -1.0
    page_resolver_kind0_count: int = -1
    page_resolver_kind1_count: int = -1
    page_resolver_kind4_count: int = -1
    kind2_dispatch_count: int = -1
    kind3_dispatch_count: int = -1
    selected_table_publish_count: int = -1
    vector_fallback_rows: int = -1
    full_fallback_rows: int = -1
    diagnostic_flags_enabled: bool | None = None
    generated_token_count: int = -1
    actual_decode_steps: int = -1
    finish_reason: str = ""
    eos_seen: bool | None = None
    semantic_output_health: str = ""
    writer_launch_count: int = -1
    writer_pointer_rebuild_count: int = -1
    writer_pointer_lookup_count: int = -1
    writer_cached_pointer_hit_rate: float = -1.0
    writer_cached_pointer_op_count: int = -1
    writer_vector_fallback_count: int = -1
    writer_kernel_variant: str = ""
    writer_actual_tokens: int = -1
    writer_sink_tokens: int = -1
    writer_persist_tokens: int = -1
    writer_sink_io_bytes: int = -1
    writer_persist_io_bytes: int = -1
    writer_token_tiles_estimated: int = -1
    writer_active_token_tiles_estimated: int = -1
    writer_cta_count_estimated: int = -1
    writer_active_cta_count_estimated: int = -1
    writer_tokens_per_cta: int = -1
    writer_k_read_bytes: int = -1
    writer_v_read_bytes: int = -1
    writer_k_write_bytes: int = -1
    writer_v_write_bytes: int = -1
    writer_pos_write_bytes: int = -1
    writer_total_io_bytes: int = -1
    writer_effective_io_gbps: float = -1.0
    selected_indices_materialized_bytes: int = -1
    selected_indices_io_bytes: int = -1
    selector_writer_current_path_count: int = -1
    selector_writer_boundary_cpu_us: float = -1.0
    deadline_v2_attribution: dict[str, object] = field(default_factory=dict)
    selected_boundary_lower_bound_ms_per_group: float = -1.0
    predicted_front_early_step_improvement_ms: float = -1.0
    residual_fixed_capture_control_ms: float = -1.0
    resolver_kind: str = ""
    route_mode: str = ""
    native_compact_residency_required: bool = True
    native_compact_residency_missing_layers: list[int] = field(default_factory=list)
    expected_group_mask: int = -1
    submitted_group_mask: int = -1
    rrp_publish_us: float = -1.0
    producer_final_event_present: bool | None = None
    producer_final_event_recorded: bool | None = None
    graph_wait_event_used: bool | None = None
    event_query_used_for_publish: bool | None = None
    producer_final_event_gpu_wait_us: float = -1.0
    source_ready_event_generation: int = -1
    production_gate_passed: bool | None = None
    native_compact_copy_bytes: int = -1
    compact_to_compact_copy_bytes: int = -1
    compact_kv_storage_owner: str = ""
    compact_kv_native_residency_bind_count: int = -1
    compact_kv_storage_mismatch_count: int = -1
    compact_kv_reserved_span_mismatch_count: int = -1
    selected_middle_tokens: int = -1
    selected_middle_pages: int = -1
    compact_reserved_pages: int = -1
    recent_canonical_pages: int = -1
    middle_native_canonical_pages: int = -1
    compact_full_native_fallback_rows: int = -1
    source_counter_schema_version: int = -1
    source_counter_missing_fields: list[str] = field(default_factory=list)
    dense_front_ms: float = -1.0
    sparse_front_ms: float = -1.0
    front_ms_source: str = "elapsed_ms - decode_elapsed_ms"
    arena_reserved_bytes: int = -1
    arena_peak_bytes: int = -1
    arena_bucket_bytes: int = -1
    arena_bucket_count: int = -1
    arena_largest_bucket_bytes: int = -1
    arena_expansion_bytes: int = -1
    arena_budget_exceeded: bool | None = None
    arena_prepare_miss_count: int = -1
    arena_bind_status: str = ""
    arena_reservation_status: str = ""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    argv_list = list(sys.argv[1:] if argv is None else argv)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", default="")
    parser.add_argument("--run-nonce", default="")
    parser.add_argument(
        "--mode",
        choices=("legacy", "dense", "sparse"),
        default="legacy",
        help="Gate D single-mode run. Default keeps the legacy one-shot e2e behavior.",
    )
    parser.add_argument(
        "--preset",
        choices=(BS2_LONG_CAP128_PRESET,),
        default="",
        help=(
            "Named comparable gate settings. bs2long-cap128 reproduces the "
            "previous effective long-context observation: two split context "
            "prompts, max_new_tokens=128 per request, no EOS stop, FULL graph."
        ),
    )
    parser.add_argument(
        "--backend",
        choices=BACKEND_CHOICES,
        default=BACKEND_FA3,
        help=(
            "Attention backend expected by Gate D. fa3 preserves the SM80 path; "
            "fa4-sm100 drives the same workload through SM100/FA4 dispatch."
        ),
    )
    parser.add_argument(
        "--full-cuda-graph",
        action="store_true",
        help="Require and request FULL CUDA graph for Gate D single-mode runs.",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--iters",
        type=int,
        default=20,
        help=(
            "Deprecated legacy alias for output length. Prefer "
            "--max-new-tokens so benchmark commands state generation length "
            "explicitly."
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help=(
            "Preferred output token cap passed to dense/sparse child runners. "
            "If omitted, legacy --iters is used as the output length."
        ),
    )
    parser.add_argument(
        "--request-max-new-tokens",
        default="",
        help=(
            "Comma-separated per-request output caps. The vector must contain "
            "exactly --batch-size positive integers; its maximum is the scalar "
            "--max-new-tokens engine cap."
        ),
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--fa3-upstream-root", default=DEFAULT_FA3_UPSTREAM_ROOT)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--request-context-tokens",
        default="",
        help=(
            "Comma-separated exact pre-chat-template context lengths, one "
            "positive integer per request."
        ),
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=0,
        help=(
            "vLLM scheduler concurrency cap (0 = --batch-size). Values below "
            "the request batch are rejected because they change benchmark "
            "scheduling instead of only constraining engine capacity."
        ),
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=DEFAULT_MAX_NUM_BATCHED_TOKENS,
        help="Shared scheduler token budget forwarded to every child.",
    )
    parser.add_argument(
        "--kv-cache-memory-bytes",
        type=int,
        default=0,
        help="Shared per-rank KV cache allocation forwarded to every child.",
    )
    parser.add_argument(
        "--chunked-prefill",
        choices=CHUNKED_PREFILL_MODES,
        default="enabled",
        help="Shared chunked-prefill policy forwarded to every child.",
    )
    parser.add_argument(
        "--max-seq-len-to-capture",
        type=int,
        default=DEFAULT_MAX_SEQ_LEN_TO_CAPTURE,
        help=(
            "Cross-version graph capture length identity. Newer vLLM V1 "
            "releases report this as unsupported rather than silently using "
            "a sparse/dense-specific default."
        ),
    )
    parser.add_argument(
        "--scheduling-mode",
        choices=("auto", "async", "sync"),
        default="auto",
        help=(
            "Engine scheduling policy forwarded identically to dense and "
            "sparse speed children."
        ),
    )
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
        "--skip-dense-reference",
        action="store_true",
        help=(
            "Diagnostic sparse-only mode: keep sparse decoded-text outputs but "
            "skip the dense reference child. Not a final correctness gate."
        ),
    )
    parser.add_argument(
        "--verdict-only",
        action="store_true",
        help=(
            "[VERDICT-ONLY 2026-07-11] Skip the diagnostic child; route/"
            "producer proofs re-source from the speed child route trace "
            "(+hook profile). Verdict-grade correctness screening; tps is "
            "directional only (trace observer tax uncalibrated)."
        ),
    )
    parser.add_argument(
        "--chat-template",
        action="store_true",
        help="Render prompts through the model chat template in sparse/dense runners.",
    )
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument(
        "--gpu-mem-util",
        type=float,
        default=0.9,
        help=(
            "Engine GPU-memory utilization. Named presets supply their "
            "validated default unless this capacity-only value is explicit."
        ),
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=0,
        help=(
            "Optional engine max_model_len override forwarded to sparse/dense "
            "runners (0 = model default). Needed for models whose "
            "max_position_embeddings exceeds GPU KV-cache capacity."
        ),
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help=(
            "vLLM tensor_parallel_size forwarded to sparse/dense runners; "
            "pass a matching multi-GPU --cuda-visible-devices list."
        ),
    )
    parser.add_argument("--timeout-s", type=int, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--route-trace-output", default="")
    parser.add_argument("--decode-metrics-output", default="")
    parser.add_argument("--refresh-profile-output", default="")
    parser.add_argument(
        "--full-cudagraph-hook-profile-output",
        default="",
        help=(
            "Diagnostic-only JSONL path for FULL CUDA graph wrapper hook timing. "
            "This is applied only to the diagnostic child run so speed gate timing "
            "stays unprofiled."
        ),
    )
    parser.add_argument(
        "--selector-pipeline-cpu-profile-output",
        default="",
        help=(
            "Diagnostic-only selector pipeline CPU profile path. When set, only "
            "the diagnostic child run receives VLLM_SPARSE_PIPELINE_CPU_PROFILE=1."
        ),
    )
    parser.add_argument("--one-shot-timeline-output", default="")
    parser.add_argument("--case", default="sm80_one_shot_graph_e2e")
    parser.add_argument(
        "--producer-mode",
        choices=PRODUCER_MODES,
        default=PRODUCER_MODE_LEGACY,
        help=(
            "Explicit SM80 producer gate mode. legacy preserves the historical "
            "one-shot wrapper; baseline-lastn1 freezes the last_n=1 refresh-off "
            "control; refresh-on, refresh-trigger-on, and full-open-gt1 enable "
            "the staged full-open producer gates."
        ),
    )
    parser.add_argument(
        "--refresh-interval",
        type=int,
        default=128,
        help="Decode refresh interval for refresh-on producer modes.",
    )
    parser.add_argument(
        "--trigger-min-gap",
        type=int,
        default=16,
        help="Minimum decode-token gap between sentence-trigger refresh intents.",
    )
    parser.add_argument(
        "--sentence-cooldown",
        type=int,
        default=2,
        help="Sentence-trigger cooldown after a boundary intent fires.",
    )
    parser.add_argument(
        "--refresh-coalesce-window",
        type=int,
        default=0,
        help="Decode-step window used to coalesce adjacent pending refresh tickets.",
    )
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
        default=2,
        help=(
            "Number of prefill query rows captured for bootstrap selection. "
            "The one-shot GT1 gate defaults to the smallest >1 capture window "
            "to keep the first segment light. Use 1 to remove gt1 reduce from "
            "the one-shot hide ablation."
        ),
    )
    bootstrap_producer_group = parser.add_mutually_exclusive_group()
    bootstrap_producer_group.add_argument(
        "--defer-bootstrap-producer",
        dest="defer_bootstrap_producer",
        action="store_true",
        default=True,
        help=(
            "Enable the default bootstrap full-KV handoff protocol: launch the "
            "bootstrap producer after prefill and let not-ready decode rows use "
            "mixed-page/RRP full-KV handoff."
        ),
    )
    bootstrap_producer_group.add_argument(
        "--no-defer-bootstrap-producer",
        dest="defer_bootstrap_producer",
        action="store_false",
        help="Disable bootstrap full-KV handoff and run the no-defer control.",
    )
    parser.add_argument("--bootstrap-bridge-max-tokens", type=int, default=3)
    parser.add_argument(
        "--deferred-producer-groups-per-step",
        type=int,
        default=-1,
        help=(
            "Diagnostic only: when deferred bootstrap producer is enabled, "
            "submit at most this many producer payload groups per step. "
            "-1 lets the runtime adapt to bridge slack and remaining producer "
            "groups, 0 forces submit-all behavior."
        ),
    )
    parser.add_argument(
        "--bootstrap-bridge-graph-policy",
        choices=DEFERRED_BRIDGE_GRAPH_POLICIES,
        default="evict_recapture_once",
    )
    parser.add_argument("--workload-plan-output", default="")
    parser.add_argument("--workload-plan-replay", default="")
    parser.add_argument(
        "--workload-plan-mode",
        choices=("record", "replay", "natural"),
        default="natural",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write a fail-closed Phase 2 record without launching vLLM.",
    )
    args = parser.parse_args(argv_list)
    if not math.isfinite(float(args.gpu_mem_util)) or not (
        0.0 < float(args.gpu_mem_util) <= 1.0
    ):
        parser.error("--gpu-mem-util must be finite and in (0, 1]")
    if args.warmup < 0:
        parser.error("--warmup must be >= 0")
    if args.iters <= 0:
        parser.error("--iters must be > 0 when used as legacy output length")
    max_new_tokens_was_explicit = _argv_has_option(argv_list, "--max-new-tokens")
    iters_was_explicit = _argv_has_option(argv_list, "--iters")
    if args.max_new_tokens is not None and int(args.max_new_tokens) <= 0:
        parser.error("--max-new-tokens must be > 0")
    if (
        max_new_tokens_was_explicit
        and iters_was_explicit
        and int(args.max_new_tokens) != int(args.iters)
    ):
        parser.error(
            "--iters is a deprecated output-length alias; do not pass a "
            "different --max-new-tokens value in the same command"
        )
    _apply_named_preset(
        parser,
        args,
        argv_list=argv_list,
        max_new_tokens_was_explicit=max_new_tokens_was_explicit,
        iters_was_explicit=iters_was_explicit,
    )
    legacy_iters = int(args.iters)
    effective_max_new_tokens = (
        int(args.max_new_tokens)
        if args.max_new_tokens is not None
        else legacy_iters
    )
    args.legacy_iters = legacy_iters
    args.iters = effective_max_new_tokens
    args.output_len = effective_max_new_tokens
    args.max_new_tokens = effective_max_new_tokens
    args.max_new_tokens_effective = effective_max_new_tokens
    args.max_new_tokens_was_explicit = bool(max_new_tokens_was_explicit)
    args.iters_was_explicit = bool(iters_was_explicit)
    args.iters_semantics = "legacy_alias_for_max_new_tokens"
    args.repeat_semantics = "legacy_alias_for_max_new_tokens"
    if args.batch_size <= 0:
        parser.error("--batch-size must be > 0")
    request_context_raw = str(args.request_context_tokens or "").strip()
    if not request_context_raw:
        request_context_raw = str(
            os.environ.get("REQUEST_CONTEXT_TOKENS", "")
            or os.environ.get("SFI_RUNNER_REQUEST_CONTEXT_TOKENS", "")
        ).strip()
    if not request_context_raw:
        scalar_context_raw = str(
            os.environ.get("SFI_RUNNER_CONTEXT_TOKENS", "") or ""
        ).strip()
        if scalar_context_raw:
            request_context_raw = ",".join(
                [scalar_context_raw] * int(args.batch_size)
            )
    request_max_new_raw = str(args.request_max_new_tokens or "").strip()
    if not request_max_new_raw:
        request_max_new_raw = str(
            os.environ.get("REQUEST_MAX_NEW_TOKENS", "")
            or os.environ.get("SFI_RUNNER_REQUEST_MAX_NEW_TOKENS", "")
        ).strip()
    request_context_tokens = (
        _parse_positive_request_vector(
            parser,
            request_context_raw,
            option="--request-context-tokens",
            expected_count=int(args.batch_size),
        )
        if request_context_raw
        else []
    )
    request_max_new_tokens = (
        _parse_positive_request_vector(
            parser,
            request_max_new_raw,
            option="--request-max-new-tokens",
            expected_count=int(args.batch_size),
        )
        if request_max_new_raw
        else [effective_max_new_tokens] * int(args.batch_size)
    )
    request_maximum = max(request_max_new_tokens)
    if request_maximum != effective_max_new_tokens:
        if max_new_tokens_was_explicit or iters_was_explicit:
            parser.error(
                "max(--request-max-new-tokens) must equal the effective "
                f"--max-new-tokens ({effective_max_new_tokens}), got "
                f"{request_maximum}"
            )
        effective_max_new_tokens = request_maximum
        args.iters = request_maximum
        args.output_len = request_maximum
        args.max_new_tokens = request_maximum
        args.max_new_tokens_effective = request_maximum
    scalar_context_raw = str(
        os.environ.get("SFI_RUNNER_CONTEXT_TOKENS", "") or ""
    ).strip()
    if request_context_tokens and scalar_context_raw:
        try:
            scalar_context = int(scalar_context_raw)
        except ValueError:
            parser.error("SFI_RUNNER_CONTEXT_TOKENS must be a positive integer")
        if scalar_context <= 0 or max(request_context_tokens) != scalar_context:
            parser.error(
                "max(--request-context-tokens) must equal "
                "SFI_RUNNER_CONTEXT_TOKENS"
            )
    args.request_context_tokens_vector = request_context_tokens
    args.request_max_new_tokens_vector = request_max_new_tokens
    args.request_context_tokens = _request_vector_csv(request_context_tokens)
    args.request_max_new_tokens = _request_vector_csv(request_max_new_tokens)
    try:
        validate_scheduler_graph_args(args)
    except ValueError as exc:
        parser.error(str(exc))
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
    if args.refresh_interval <= 0:
        parser.error("--refresh-interval must be > 0")
    if args.trigger_min_gap < 0:
        parser.error("--trigger-min-gap must be >= 0")
    if args.sentence_cooldown < 0:
        parser.error("--sentence-cooldown must be >= 0")
    if args.refresh_coalesce_window < 0:
        parser.error("--refresh-coalesce-window must be >= 0")
    if args.producer_mode in {
        PRODUCER_MODE_BASELINE,
        PRODUCER_MODE_REFRESH,
        PRODUCER_MODE_TRIGGER,
    } and int(args.prefill_last_n) != 1:
        parser.error(
            f"--producer-mode {args.producer_mode} requires --prefill-last-n 1"
        )
    if (
        args.producer_mode == PRODUCER_MODE_FULL_OPEN
        and int(args.prefill_last_n) <= 1
    ):
        parser.error("--producer-mode full-open-gt1 requires --prefill-last-n > 1")
    if (
        args.mode == "legacy"
        and args.preset == BS2_LONG_CAP128_PRESET
        and args.producer_mode != PRODUCER_MODE_LEGACY
    ):
        parser.error(
            "--preset bs2long-cap128 with active --producer-mode requires "
            "--mode sparse; legacy mode does not exercise the Gate-D producer path"
        )
    if args.bootstrap_bridge_max_tokens <= 0:
        parser.error("--bootstrap-bridge-max-tokens must be > 0")
    if args.deferred_producer_groups_per_step < -1:
        parser.error("--deferred-producer-groups-per-step must be >= -1")
    if args.mode == "legacy" and not args.summary_output:
        parser.error("--summary-output is required in legacy mode")
    if args.mode != "legacy" and not bool(args.full_cuda_graph):
        parser.error("--full-cuda-graph is required for Gate D mode")
    if args.workload_plan_output and args.workload_plan_replay:
        parser.error("--workload-plan-output and --workload-plan-replay are mutually exclusive")
    if args.workload_plan_output and args.workload_plan_mode == "natural":
        args.workload_plan_mode = "record"
    if args.workload_plan_replay and args.workload_plan_mode == "natural":
        args.workload_plan_mode = "replay"
    if args.workload_plan_mode == "record" and not args.workload_plan_output:
        parser.error("--workload-plan-mode record requires --workload-plan-output")
    if args.workload_plan_mode == "replay" and not args.workload_plan_replay:
        parser.error("--workload-plan-mode replay requires --workload-plan-replay")
    if args.workload_plan_mode in {"record", "replay"} and args.mode != "sparse":
        parser.error("workload plan record/replay is only supported in --mode sparse")
    if bool(args.verdict_only) and args.mode != "sparse":
        parser.error("--verdict-only is only supported in --mode sparse")
    if bool(args.verdict_only) and args.workload_plan_mode != "natural":
        parser.error(
            "--verdict-only cannot be combined with workload plan "
            "record/replay: the plan event source would move to the "
            "speed-child trace, which is uncalibrated; run the full "
            "two-child form for record/replay"
        )
    return args


def _bs2long_cap128_gpu_mem_util(args: argparse.Namespace) -> float:
    if str(getattr(args, "backend", "") or "") == BACKEND_FA4_SM100:
        return BS2_LONG_CAP128_FA4_SM100_GPU_MEM_UTIL
    return BS2_LONG_CAP128_GPU_MEM_UTIL


def _apply_named_preset(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    *,
    argv_list: list[str],
    max_new_tokens_was_explicit: bool,
    iters_was_explicit: bool,
) -> None:
    if str(getattr(args, "preset", "") or "") != BS2_LONG_CAP128_PRESET:
        return

    _reject_conflicting_preset_value(
        parser,
        argv_list,
        "--prompt",
        str(args.prompt),
        BS2_LONG_CAP128_PROMPT,
    )
    _reject_conflicting_preset_value(
        parser,
        argv_list,
        "--batch-size",
        int(args.batch_size),
        BS2_LONG_CAP128_BATCH_SIZE,
    )
    expected_gpu_mem_util = _bs2long_cap128_gpu_mem_util(args)
    _reject_conflicting_preset_value(
        parser,
        argv_list,
        "--refresh-interval",
        int(args.refresh_interval),
        BS2_LONG_CAP128_REFRESH_INTERVAL,
    )
    if bool(args.respect_eos):
        parser.error(f"{BS2_LONG_CAP128_PRESET} preset requires no-EOS generation")
    if max_new_tokens_was_explicit and int(args.max_new_tokens) != BS2_LONG_CAP128_MAX_NEW_TOKENS:
        parser.error(
            f"{BS2_LONG_CAP128_PRESET} preset requires "
            f"--max-new-tokens {BS2_LONG_CAP128_MAX_NEW_TOKENS}"
        )
    if iters_was_explicit and int(args.iters) != BS2_LONG_CAP128_MAX_NEW_TOKENS:
        parser.error(
            f"{BS2_LONG_CAP128_PRESET} preset treats --iters as a legacy "
            f"output-length alias and requires {BS2_LONG_CAP128_MAX_NEW_TOKENS}"
        )

    args.prompt = BS2_LONG_CAP128_PROMPT
    args.batch_size = BS2_LONG_CAP128_BATCH_SIZE
    args.split_context_prompts = True
    args.max_new_tokens = BS2_LONG_CAP128_MAX_NEW_TOKENS
    if not _argv_has_option(argv_list, "--gpu-mem-util"):
        args.gpu_mem_util = expected_gpu_mem_util
    args.refresh_interval = BS2_LONG_CAP128_REFRESH_INTERVAL
    args.full_cuda_graph = True


def _reject_conflicting_preset_value(
    parser: argparse.ArgumentParser,
    argv: list[str],
    option: str,
    value: object,
    expected: object,
) -> None:
    if not _argv_has_option(argv, option):
        return
    if value != expected:
        parser.error(
            f"{BS2_LONG_CAP128_PRESET} preset requires {option} {expected!r}; "
            f"got {value!r}"
        )


def _argv_has_option(argv: list[str], option: str) -> bool:
    return any(arg == option or arg.startswith(f"{option}=") for arg in argv)


def _parse_positive_request_vector(
    parser: argparse.ArgumentParser,
    raw: str,
    *,
    option: str,
    expected_count: int,
) -> list[int]:
    """Parse one exact request vector; reject ambiguous CSV spellings."""
    if not re.fullmatch(r"[1-9][0-9]*(?:,[1-9][0-9]*)*", raw):
        parser.error(
            f"{option} must be a comma-separated list of positive integers"
        )
    values = [int(value) for value in raw.split(",")]
    if len(values) != int(expected_count):
        parser.error(
            f"{option} must contain exactly --batch-size={expected_count} "
            f"values, got {len(values)}"
        )
    return values


def _request_vector_csv(values: list[int]) -> str:
    return ",".join(str(int(value)) for value in values)


def _request_context_tokens(args: argparse.Namespace) -> list[int]:
    values = getattr(args, "request_context_tokens_vector", None)
    if isinstance(values, list):
        return [int(value) for value in values]
    return []


def _bind_runtime_request_context_tokens(
    args: argparse.Namespace,
    speed_metrics: dict[str, Any],
) -> bool:
    """Bind an undeclared context vector from the completed speed artifact."""
    if _request_context_tokens(args):
        return False
    run_config_raw = speed_metrics.get("run_config")
    run_config = run_config_raw if isinstance(run_config_raw, dict) else {}
    values_raw = run_config.get("request_context_tokens")
    if not isinstance(values_raw, list):
        return False
    if len(values_raw) != int(args.batch_size) or any(
        type(value) is not int or value <= 0 for value in values_raw
    ):
        return False
    values = list(values_raw)
    args.request_context_tokens_vector = values
    args.request_context_tokens = _request_vector_csv(values)
    return True


def _request_max_new_tokens(args: argparse.Namespace) -> list[int]:
    values = getattr(args, "request_max_new_tokens_vector", None)
    if isinstance(values, list) and values:
        return [int(value) for value in values]
    return [_effective_max_new_tokens(args)] * int(args.batch_size)


def _effective_max_new_tokens(args: argparse.Namespace) -> int:
    value = getattr(args, "max_new_tokens_effective", None)
    if value is None:
        value = getattr(args, "iters", 1)
    return int(value)


def _effective_max_num_seqs(args: argparse.Namespace) -> int:
    return resolve_benchmark_max_num_seqs(
        batch_size=int(args.batch_size),
        configured=int(getattr(args, "max_num_seqs", 0) or 0),
    )


def _producer_mode(args: argparse.Namespace) -> str:
    mode = str(getattr(args, "producer_mode", PRODUCER_MODE_LEGACY) or PRODUCER_MODE_LEGACY)
    return mode if mode in PRODUCER_MODES else PRODUCER_MODE_LEGACY


def _producer_mode_requires_refresh(args_or_mode: argparse.Namespace | str) -> bool:
    mode = (
        _producer_mode(args_or_mode)
        if isinstance(args_or_mode, argparse.Namespace)
        else str(args_or_mode)
    )
    return mode in {
        PRODUCER_MODE_REFRESH,
        PRODUCER_MODE_TRIGGER,
        PRODUCER_MODE_FULL_OPEN,
    }


def _producer_mode_requires_sentence_trigger(args_or_mode: argparse.Namespace | str) -> bool:
    mode = (
        _producer_mode(args_or_mode)
        if isinstance(args_or_mode, argparse.Namespace)
        else str(args_or_mode)
    )
    return mode in {PRODUCER_MODE_TRIGGER, PRODUCER_MODE_FULL_OPEN}


def _producer_mode_requires_observed_sentence_trigger(
    args_or_mode: argparse.Namespace | str,
) -> bool:
    mode = (
        _producer_mode(args_or_mode)
        if isinstance(args_or_mode, argparse.Namespace)
        else str(args_or_mode)
    )
    return mode == PRODUCER_MODE_TRIGGER


def _producer_mode_requires_gt1(args_or_mode: argparse.Namespace | str) -> bool:
    mode = (
        _producer_mode(args_or_mode)
        if isinstance(args_or_mode, argparse.Namespace)
        else str(args_or_mode)
    )
    return mode == PRODUCER_MODE_FULL_OPEN


def _phase2_continuous_producer_enabled(args: argparse.Namespace) -> bool:
    return _producer_mode_requires_refresh(args)


def record_to_jsonable(record: Phase2OneShotGraphRecord) -> dict[str, object]:
    return asdict(record)


def write_records(path: Path, records: list[Phase2OneShotGraphRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(record_to_jsonable(record), sort_keys=True, allow_nan=False)
            )
            handle.write("\n")


def _route_proof_failure_reason(route_proof_result: RouteProofResult) -> str:
    return "route_proof_failed:" + ",".join(route_proof_result.reasons)


def _apply_route_proof_gate(
    record: Phase2OneShotGraphRecord,
    failure_reason: str,
    route_proof_result: RouteProofResult,
) -> tuple[Phase2OneShotGraphRecord, str]:
    if route_proof_result.passed:
        return record, failure_reason
    route_failure = _route_proof_failure_reason(route_proof_result)
    merged_reason = (
        f"{failure_reason};{route_failure}" if failure_reason else route_failure
    )
    return replace(record, gate_passed=False), merged_reason


def _default_refresh_profile_path(output: Path) -> Path:
    return output.with_name(output.stem + "_refresh_profile.log")


def _default_selector_pipeline_cpu_profile_path(output: Path) -> Path:
    return output.with_name(output.stem + "_selector_pipeline_cpu_profile.log")


def _default_full_cudagraph_hook_profile_path(output: Path) -> Path:
    return output.with_name(output.stem + "_full_cudagraph_hook_profile.jsonl")


def _selector_pipeline_cpu_profile_path(
    env: dict[str, str],
    output: Path,
) -> Path | None:
    if str(env.get("VLLM_SPARSE_PIPELINE_CPU_PROFILE", "") or "") != "1":
        return None
    configured = str(env.get("VLLM_SPARSE_PIPELINE_CPU_PROFILE_LOG", "") or "")
    if configured:
        return Path(configured)
    path = _default_selector_pipeline_cpu_profile_path(output)
    env["VLLM_SPARSE_PIPELINE_CPU_PROFILE_LOG"] = str(path)
    return path


def _default_one_shot_timeline_path(output: Path) -> Path:
    return output.with_name(output.stem + "_one_shot_timeline.jsonl")


def _default_sparse_outputs_path(output: Path) -> Path:
    return output.with_name(output.stem + "_outputs.json")


def _phase2_controller_payload(args: argparse.Namespace) -> dict[str, object]:
    payload = _sparse_controller_payload(args)
    producer_mode = _producer_mode(args)
    payload["one_shot_bootstrap_only"] = True
    payload["continuous_producer_enabled"] = _phase2_continuous_producer_enabled(args)
    payload["refresh_coalesce_window"] = int(args.refresh_coalesce_window)
    payload["refresh_interval"] = (
        int(args.refresh_interval)
        if _producer_mode_requires_refresh(producer_mode)
        else HUGE_REFRESH_INTERVAL
    )
    trigger = payload.setdefault("trigger", {})
    if isinstance(trigger, dict):
        trigger["refresh_interval"] = (
            int(args.refresh_interval)
            if _producer_mode_requires_refresh(producer_mode)
            else HUGE_REFRESH_INTERVAL
        )
        trigger["enable_sentence_triggers"] = _producer_mode_requires_sentence_trigger(
            producer_mode
        )
        trigger["min_refresh_gap"] = int(args.trigger_min_gap)
        trigger["sentence_cooldown"] = int(args.sentence_cooldown)
    return payload


def _workload_plan_config_from_args(args: argparse.Namespace) -> dict[str, object]:
    return {
        "batch_size": int(args.batch_size),
        "request_context_tokens": _request_context_tokens(args),
        "request_max_new_tokens": _request_max_new_tokens(args),
        "max_new_tokens": int(_effective_max_new_tokens(args)),
        "prefill_last_n_query": int(args.prefill_last_n),
        "refresh_interval": int(args.refresh_interval),
        "trigger_min_gap": int(args.trigger_min_gap),
        "sentence_cooldown": int(args.sentence_cooldown),
        "refresh_coalesce_window": int(args.refresh_coalesce_window),
    }


def _workload_plan_expected_counts(plan: dict[str, object]) -> dict[str, object]:
    reason_counts: dict[str, int] = {}
    refresh_payloads = 0
    for event in plan.get("events", []) or []:
        if not isinstance(event, dict):
            continue
        reason = str(event.get("reason", "") or "")
        if reason:
            reason_counts[reason] = int(reason_counts.get(reason, 0)) + 1
        refresh_payloads += max(0, _as_int(event.get("payload_count"), 0))
    return {
        "workload_plan_expected_refresh_payloads": int(refresh_payloads),
        "workload_plan_expected_refresh_reason_counts": dict(reason_counts),
        "workload_plan_expected_sentence_trigger_intents": int(
            reason_counts.get("sentence", 0)
        ),
        "workload_plan_expected_interval_trigger_intents": int(
            reason_counts.get("interval", 0)
        ),
    }


def _workload_plan_payload(
    args: argparse.Namespace,
    *,
    route_trace_path: Path,
    measurement_trace_events: list[dict[str, Any]],
) -> dict[str, object]:
    mode = str(getattr(args, "workload_plan_mode", "natural") or "natural")
    if mode == "record":
        output = Path(str(args.workload_plan_output))
        output.parent.mkdir(parents=True, exist_ok=True)
        plan = build_workload_plan_from_route_events(
            measurement_trace_events,
            source_summary=str(getattr(args, "summary_output", "") or ""),
            source_route_trace=str(route_trace_path),
            config=_workload_plan_config_from_args(args),
        )
        output.write_text(
            json.dumps(plan, ensure_ascii=True, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        return {
            "workload_plan_mode": "record",
            "workload_plan_output": str(output),
            "workload_plan_digest": str(plan.get("workload_plan_digest", "")),
            "workload_plan_event_count": len(plan.get("events", []) or []),
            **_workload_plan_expected_counts(plan),
        }
    if mode == "replay":
        plan_path = Path(str(args.workload_plan_replay))
        plan = load_workload_plan(plan_path)
        return {
            "workload_plan_mode": "replay",
            "workload_plan_replay": str(plan_path),
            "workload_plan_digest": workload_plan_digest(plan),
            "workload_plan_event_count": len(plan.get("events", []) or []),
            **_workload_plan_expected_counts(plan),
        }
    return {
        "workload_plan_mode": "natural",
        "workload_plan_digest": "",
        "workload_plan_event_count": 0,
        "workload_plan_expected_refresh_payloads": 0,
        "workload_plan_expected_refresh_reason_counts": {},
        "workload_plan_expected_sentence_trigger_intents": 0,
        "workload_plan_expected_interval_trigger_intents": 0,
    }


def _workload_plan_replay_count_mismatches(payload: dict[str, Any]) -> list[str]:
    if str(payload.get("workload_plan_mode", "") or "") != "replay":
        return []
    mismatches: list[str] = []

    def _check_int(actual_key: str, expected_key: str) -> None:
        actual = _as_int(payload.get(actual_key), -1)
        expected = _as_int(payload.get(expected_key), -1)
        if actual != expected:
            mismatches.append(f"{actual_key}:actual={actual}:expected={expected}")

    _check_int("refresh_payloads", "workload_plan_expected_refresh_payloads")
    _check_int(
        "sentence_trigger_intents",
        "workload_plan_expected_sentence_trigger_intents",
    )
    _check_int(
        "interval_trigger_intents",
        "workload_plan_expected_interval_trigger_intents",
    )
    actual_reasons = payload.get("refresh_reason_counts", {}) or {}
    expected_reasons = (
        payload.get("workload_plan_expected_refresh_reason_counts", {}) or {}
    )
    if not isinstance(actual_reasons, dict):
        actual_reasons = {}
    if not isinstance(expected_reasons, dict):
        expected_reasons = {}
    actual_normalized = {
        str(k): int(_as_int(v, 0))
        for k, v in actual_reasons.items()
        if int(_as_int(v, 0)) != 0
    }
    expected_normalized = {
        str(k): int(_as_int(v, 0))
        for k, v in expected_reasons.items()
        if int(_as_int(v, 0)) != 0
    }
    if actual_normalized != expected_normalized:
        mismatches.append(
            "refresh_reason_counts:"
            f"actual={actual_normalized}:expected={expected_normalized}"
        )
    return mismatches


def _apply_deferred_bridge_env(env: dict[str, str], args: argparse.Namespace) -> None:
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
    groups_per_step = _effective_deferred_producer_groups_per_step(args)
    if groups_per_step != 0:
        env["VLLM_SPARSE_DEFERRED_PRODUCER_GROUPS_PER_STEP"] = str(groups_per_step)


def _effective_deferred_producer_groups_per_step(args: argparse.Namespace) -> int:
    raw = int(getattr(args, "deferred_producer_groups_per_step", -1))
    if raw >= 0:
        return raw
    return -1 if bool(getattr(args, "defer_bootstrap_producer", False)) else 0


def _apply_gt1_full_cudagraph_refresh_env(
    env: dict[str, str],
    args: argparse.Namespace,
) -> None:
    if not _producer_mode_requires_gt1(args):
        return
    if not str(env.get("TORCH_EXTENSIONS_DIR", "") or "").strip():
        abi_key = selector_cache_abi_key_for_python(str(args.python))
        env["TORCH_EXTENSIONS_DIR"] = str(
            GT1_SELECTOR_TORCH_EXTENSIONS_ROOT / f"sm80_gt1_{abi_key}"
        )
    env.setdefault("VLLM_SPARSE_FULL_CUDAGRAPH_REPLAY_REFRESH_BATCHED_FLUSH", "1")
    env.setdefault("VLLM_SPARSE_FULL_CUDAGRAPH_REPLAY_REFRESH_DEFER_TO_DEADLINE", "1")
    env["VLLM_SPARSE_REFRESH_ENQUEUE_STAGGER"] = "1"
    env["VLLM_SPARSE_ACTIVE_BENCH_RUNNER"] = (
        "bench_sm80_mixed_page_one_shot_graph_e2e.py"
    )


def _selector_extension_prewarm_timeout_s(_args: argparse.Namespace) -> int:
    raw = os.environ.get("VLLM_SPARSE_GT1_SELECTOR_PREWARM_TIMEOUT_S", "").strip()
    if raw:
        try:
            parsed = int(raw)
        except ValueError:
            parsed = 0
        if parsed > 0:
            return parsed
    # Extension compilation has a separate failure domain from the vLLM run.
    # Reusing the one-hour benchmark timeout made a stale Torch FileBaton lock
    # look like a silent startup hang.  Ten minutes still leaves ample room for
    # a cold build while bounding races that appear after the lock preflight.
    return DEFAULT_GT1_SELECTOR_PREWARM_TIMEOUT_S


def _selector_extension_cache_locks(torch_extensions_dir: str) -> tuple[Path, ...]:
    """Return Torch extension FileBaton locks without mutating shared state."""
    root = Path(torch_extensions_dir)
    if not root.is_dir():
        return ()
    locks = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        lock = child / "lock"
        if lock.exists() or lock.is_symlink():
            locks.append(lock)
    return tuple(sorted(locks, key=lambda path: str(path)))


def _prewarm_gt1_selector_extensions(
    args: argparse.Namespace,
    *,
    env: dict[str, str],
) -> Phase1CommandResult | None:
    if not _producer_mode_requires_gt1(args):
        return None
    torch_extensions_dir = env.get("TORCH_EXTENSIONS_DIR", "").strip()
    if torch_extensions_dir:
        Path(torch_extensions_dir).mkdir(parents=True, exist_ok=True)
    command = [
        str(args.python),
        "-c",
        (
            "import hashlib, json, sys; from pathlib import Path; "
            "from utils.bounds_kernel_ext import _require_ext as _bounds; "
            "from utils.bounds_prefill_kernel_ext import _require_ext as _bounds_prefill; "
            "from utils.fa_sparse_runtime_ext import _load_ext as _fa_sparse; "
            "from utils.selector_batch_ext import _load_ext as _selector_batch; "
            "from utils.selector_log_s_ext import _require_ext as _log_s; "
            "from utils.selector_pipeline_identity import ("
            "SELECTOR_PIPELINE_SEMANTIC_VERSION as _pipeline_expected_semantic_version); "
            "from utils.selector_pipeline_ext import _require_ext as _pipeline; "
            "_bounds(); _bounds_prefill(); _fa_sparse(force=True); _selector_batch(); "
            "_log_s(force=True); _pipeline_mod = _pipeline(force=True); "
            "_pipeline_semantic_version = int(_pipeline_mod.selector_pipeline_semantic_version()); "
            "(_pipeline_semantic_version == int(_pipeline_expected_semantic_version)) or "
            "sys.exit(f'selector pipeline semantic mismatch: actual={_pipeline_semantic_version} '"
            "+ f'expected={_pipeline_expected_semantic_version}'); "
            "_pipeline_path = Path(_pipeline_mod.__file__).resolve(); "
            "_pipeline_sha256 = hashlib.sha256(_pipeline_path.read_bytes()).hexdigest(); "
            f"print({SELECTOR_PIPELINE_ARTIFACT_PREFIX!r} + json.dumps({{"
            "'path': str(_pipeline_path), "
            "'semantic_version': _pipeline_semantic_version, "
            "'expected_semantic_version': int(_pipeline_expected_semantic_version), "
            "'sha256': _pipeline_sha256"
            "}, sort_keys=True)); "
            "print('gt1_selector_extensions_ready')"
        ),
    ]
    prewarm_env = dict(env)
    for key in (
        "VLLM_ATTENTION_BACKEND",
        "VLLM_FLASH_ATTN_VERSION",
        "VLLM_SPARSE_CONTROLLER_JSON",
        ROUTE_TRACE_ENV_KEY,
    ):
        prewarm_env.pop(key, None)
    cache_locks = _selector_extension_cache_locks(torch_extensions_dir)
    if cache_locks:
        lock_list = ", ".join(str(path) for path in cache_locks)
        return Phase1CommandResult(
            command=command,
            returncode=73,
            stdout="",
            stderr=(
                "E_SFI_SELECTOR_CACHE_LOCK_PRESENT: refusing to wait on Torch "
                f"extension cache lock(s): {lock_list}. A lock may be active or "
                "stale; coordinate the compiler owner or choose a fresh absolute "
                "TORCH_EXTENSIONS_DIR. The runner never deletes shared cache locks."
            ),
            timed_out=False,
        )
    return _run_command(
        command,
        env=prewarm_env,
        timeout_s=_selector_extension_prewarm_timeout_s(args),
    )


def _selector_pipeline_artifact_from_prewarm_stdout(
    stdout: str,
) -> dict[str, object] | None:
    for line in reversed(str(stdout or "").splitlines()):
        if not line.startswith(SELECTOR_PIPELINE_ARTIFACT_PREFIX):
            continue
        try:
            raw = json.loads(line[len(SELECTOR_PIPELINE_ARTIFACT_PREFIX) :])
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        if not isinstance(raw, dict):
            return None
        path = str(raw.get("path", "") or "")
        sha256 = str(raw.get("sha256", "") or "").lower()
        try:
            semantic_version = int(raw.get("semantic_version", -1))
            expected_semantic_version = int(
                raw.get("expected_semantic_version", -1)
            )
        except (TypeError, ValueError):
            return None
        if (
            not path
            or semantic_version <= 0
            or expected_semantic_version <= 0
            or re.fullmatch(r"[0-9a-f]{64}", sha256) is None
        ):
            return None
        return {
            "path": path,
            "semantic_version": semantic_version,
            "expected_semantic_version": expected_semantic_version,
            "sha256": sha256,
        }
    return None


def _selector_extension_prewarm_payload(
    result: Phase1CommandResult | None,
) -> dict[str, object]:
    if result is None:
        return {
            "selector_extension_prewarm_enabled": False,
        }
    payload: dict[str, object] = {
        "selector_extension_prewarm_enabled": True,
        "selector_extension_prewarm_command": list(result.command),
        "selector_extension_prewarm_returncode": int(result.returncode),
        "selector_extension_prewarm_timed_out": bool(result.timed_out),
        "selector_extension_prewarm_stdout_tail": _tail(result.stdout),
        "selector_extension_prewarm_stderr_tail": _tail(result.stderr),
    }
    artifact = _selector_pipeline_artifact_from_prewarm_stdout(result.stdout)
    if artifact is not None:
        payload["selector_pipeline_artifact"] = artifact
    return payload


def _selector_pipeline_artifact_postflight(
    prewarm_artifact: dict[str, object],
) -> dict[str, object]:
    artifact_path = Path(str(prewarm_artifact.get("path", "") or ""))
    payload: dict[str, object] = {"path": str(artifact_path)}
    try:
        digest = _file_sha256(artifact_path)
    except OSError as exc:
        payload.update({"sha256": "", "read_error": f"{type(exc).__name__}: {exc}"})
    else:
        payload["sha256"] = digest
    return payload


def _apply_selector_extension_prewarm_payload(
    payload: dict[str, Any],
    result: Phase1CommandResult | None,
) -> None:
    prewarm_payload = _selector_extension_prewarm_payload(result)
    payload.update(prewarm_payload)
    artifact = prewarm_payload.get("selector_pipeline_artifact")
    run_provenance = payload.get("run_provenance")
    if not isinstance(artifact, dict):
        # Non-GT1 producer modes do not prewarm selector extensions by design.
        # Fail closed only when a prewarm was actually attempted but failed to
        # produce a well-formed artifact record.
        if result is None:
            return
        artifact_reasons = ["selector_pipeline_artifact_missing_or_invalid"]
        payload["selector_pipeline_artifact_reasons"] = artifact_reasons
        payload["gate_passed"] = False
        payload["production_gate_passed"] = False
        if isinstance(run_provenance, dict):
            run_provenance["selector_pipeline_artifact_reasons"] = list(
                artifact_reasons
            )
        return

    prewarm_artifact = dict(artifact)
    postflight_artifact = _selector_pipeline_artifact_postflight(prewarm_artifact)
    mutated = bool(
        postflight_artifact.get("path") != prewarm_artifact.get("path")
        or postflight_artifact.get("sha256") != prewarm_artifact.get("sha256")
    )
    artifact_reasons: list[str] = []
    if int(prewarm_artifact.get("semantic_version", -1)) != int(
        prewarm_artifact.get("expected_semantic_version", -1)
    ):
        artifact_reasons.append("selector_pipeline_semantic_version_mismatch")
    if mutated:
        artifact_reasons.append("selector_artifact_mutated_after_prewarm")
    payload.update(
        {
            "selector_pipeline_artifact_prewarm": prewarm_artifact,
            "selector_pipeline_artifact_postflight": postflight_artifact,
            "selector_pipeline_artifact_mutated_after_prewarm": mutated,
            "selector_pipeline_artifact_reasons": artifact_reasons,
        }
    )
    if isinstance(run_provenance, dict):
        run_provenance.update(
            {
                "selector_pipeline_artifact": prewarm_artifact,
                "selector_pipeline_artifact_prewarm": prewarm_artifact,
                "selector_pipeline_artifact_postflight": postflight_artifact,
                "selector_pipeline_artifact_mutated_after_prewarm": mutated,
                "selector_pipeline_artifact_reasons": list(artifact_reasons),
            }
        )
    if artifact_reasons:
        payload["gate_passed"] = False
        payload["production_gate_passed"] = False


def _build_phase2_env(
    args: argparse.Namespace,
    *,
    route_trace_path: Path,
    refresh_profile_path: Path,
    one_shot_timeline_path: Path | None = None,
) -> dict[str, str]:
    env = _build_phase1_env(args, route_trace_path=route_trace_path)
    _apply_gate_d_backend_env(args, env)
    if _gate_d_backend(args) == BACKEND_FA4_SM100:
        env["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN_VLLM_V1"
    env["VLLM_SPARSE_ASYNC_REFRESH"] = "1"
    env["VLLM_SPARSE_ONE_SHOT_ASYNC_BOOTSTRAP"] = "1"
    env["VLLM_SPARSE_CONTROLLER_JSON"] = json.dumps(
        _phase2_controller_payload(args),
        sort_keys=True,
    )
    _apply_gt1_full_cudagraph_refresh_env(env, args)
    env["VLLM_SPARSE_REFRESH_PROFILE"] = "1"
    env["VLLM_SPARSE_REFRESH_PROFILE_DETAIL"] = "1"
    env["VLLM_SPARSE_REFRESH_PROFILE_CALL_MIN"] = "0"
    env["VLLM_SPARSE_REFRESH_PROFILE_EVERY"] = "1"
    env["VLLM_SPARSE_REFRESH_PROFILE_LOG"] = str(refresh_profile_path)
    if one_shot_timeline_path is not None:
        env["VLLM_SPARSE_ONE_SHOT_TIMELINE_LOG"] = str(one_shot_timeline_path)
    _apply_deferred_bridge_env(env, args)
    return env


def _build_phase2_command(
    args: argparse.Namespace,
    *,
    metrics_path: Path,
    refresh_profile_path: Path,
    outputs_path: Path,
) -> list[str]:
    command = _build_smoke_command(
        args,
        metrics_path=metrics_path,
        outputs_path=outputs_path,
    )
    _append_scheduler_graph_child_args(command, args)
    _append_request_vector_child_args(command, args)
    command.extend(["--scheduling-mode", str(args.scheduling_mode)])
    command.append("--collect-cudagraph-runtime-proof")
    command.append("--collect-route-counter-proof")
    _append_deferred_bridge_child_args(command, args)
    command.extend(
        [
            "--refresh-profile",
            "--refresh-profile-detail",
            "--refresh-profile-log",
            str(refresh_profile_path),
        ]
    )
    return command


def _append_scheduler_graph_child_args(
    command: list[str],
    args: argparse.Namespace,
) -> None:
    contract = requested_scheduler_graph_contract(args)
    max_num_batched_tokens = contract["max_num_batched_tokens"]
    if max_num_batched_tokens is not None:
        command.extend(
            ["--max-num-batched-tokens", str(max_num_batched_tokens)]
        )
    kv_cache_memory_bytes = contract["kv_cache_memory_bytes"]
    if kv_cache_memory_bytes is not None:
        command.extend(
            ["--kv-cache-memory-bytes", str(kv_cache_memory_bytes)]
        )
    command.extend(
        [
            "--chunked-prefill",
            str(contract["chunked_prefill"]),
            "--max-seq-len-to-capture",
            str(contract["max_seq_len_to_capture"]),
        ]
    )


def _append_request_vector_child_args(
    command: list[str],
    args: argparse.Namespace,
) -> None:
    context_tokens = _request_context_tokens(args)
    if context_tokens:
        command.extend(
            ["--request-context-tokens", _request_vector_csv(context_tokens)]
        )
    command.extend(
        [
            "--request-max-new-tokens",
            _request_vector_csv(_request_max_new_tokens(args)),
        ]
    )


def _append_deferred_bridge_child_args(
    command: list[str],
    args: argparse.Namespace,
) -> None:
    if not bool(getattr(args, "defer_bootstrap_producer", False)):
        command.append("--no-defer-bootstrap-producer")
        return
    command.extend(
        [
            "--defer-bootstrap-producer",
            "--bootstrap-bridge-max-tokens",
            str(int(args.bootstrap_bridge_max_tokens)),
            "--bootstrap-bridge-graph-policy",
            str(args.bootstrap_bridge_graph_policy),
        ]
    )
    groups_per_step = _effective_deferred_producer_groups_per_step(args)
    if groups_per_step != 0:
        command.extend(
            [
                "--deferred-producer-groups-per-step",
                str(groups_per_step),
            ]
        )

def _build_sparse_speed_command(
    args: argparse.Namespace,
    *,
    metrics_path: Path,
    outputs_path: Path,
) -> list[str]:
    command = _build_smoke_command(
        args,
        metrics_path=metrics_path,
        outputs_path=outputs_path,
    )
    _append_scheduler_graph_child_args(command, args)
    _append_request_vector_child_args(command, args)
    command.extend(["--scheduling-mode", str(args.scheduling_mode)])
    _append_deferred_bridge_child_args(command, args)
    return command


def _build_no_eos_diagnostic_command(args: argparse.Namespace) -> list[str]:
    command = [
        str(args.python),
        str(Path(__file__).resolve()),
        "--mode",
        str(args.mode),
        "--backend",
        str(args.backend),
        "--full-cuda-graph",
        "--warmup",
        str(args.warmup),
        "--iters",
        str(_effective_max_new_tokens(args)),
        "--max-new-tokens",
        str(_effective_max_new_tokens(args)),
        "--model",
        str(args.model),
        "--prompt",
        str(args.prompt),
        "--cuda-visible-devices",
        str(args.cuda_visible_devices),
        "--batch-size",
        str(args.batch_size),
        "--request-max-new-tokens",
        _request_vector_csv(_request_max_new_tokens(args)),
        "--max-num-seqs",
        str(_effective_max_num_seqs(args)),
        "--max-num-batched-tokens",
        str(int(args.max_num_batched_tokens)),
        "--kv-cache-memory-bytes",
        str(int(args.kv_cache_memory_bytes)),
        "--chunked-prefill",
        str(args.chunked_prefill),
        "--max-seq-len-to-capture",
        str(int(args.max_seq_len_to_capture)),
        "--scheduling-mode",
        str(args.scheduling_mode),
        "--gpu-mem-util",
        str(args.gpu_mem_util),
        "--timeout-s",
        str(args.timeout_s),
        "--output",
        "logs/no_eos_diagnostic.jsonl",
        "--summary-output",
        "logs/no_eos_diagnostic_summary.json",
    ]
    context_tokens = _request_context_tokens(args)
    if context_tokens:
        command.extend(
            ["--request-context-tokens", _request_vector_csv(context_tokens)]
        )
    if bool(args.split_context_prompts):
        command.append("--split-context-prompts")
    if bool(args.outputs_include_text):
        command.append("--outputs-include-text")
    if bool(args.chat_template):
        command.append("--chat-template")
    if bool(args.enable_thinking):
        command.append("--enable-thinking")
    _append_deferred_bridge_child_args(command, args)
    return command


def _default_gate_d_metrics_path(output: Path, *, suffix: str = "") -> Path:
    stem = output.stem + (suffix if suffix else "") + "_decode_metrics.json"
    return output.with_name(stem)


def _default_gate_d_outputs_path(output: Path, *, suffix: str = "") -> Path:
    stem = output.stem + (suffix if suffix else "") + "_outputs.json"
    return output.with_name(stem)


def _default_gate_d_route_path(output: Path) -> Path:
    return output.with_name(output.stem + "_route.jsonl")


def _default_gate_d_speed_route_path(output: Path) -> Path:
    return output.with_name(output.stem + "_speed_route.jsonl")


def _default_gate_d_dense_route_path(output: Path) -> Path:
    return output.with_name(output.stem + "_dense_route.jsonl")


def _default_gate_d_dense_reference_metrics_path(output: Path) -> Path:
    return output.with_name(output.stem + "_dense_reference_decode_metrics.json")


def _default_gate_d_dense_reference_outputs_path(output: Path) -> Path:
    return output.with_name(output.stem + "_dense_reference_outputs.json")


_TP8_EXACT_ARM_ORDER = (
    "sparse_speed",
    "dense_reference",
    "sparse_diagnostic",
)
_TP8_EXACT_SETUP_ARM_ORDER = ("selector_prewarm",)


def _default_tp8_process_baseline_path(output: Path) -> Path:
    return output.with_name(output.stem + "_tp8_process_baseline.json")


def _default_tp8_arm_teardown_path(output: Path, arm_tag: str) -> Path:
    return output.with_name(output.stem + f"_tp8_{arm_tag}_teardown.json")


def _tp8_exact_pair_required(args: argparse.Namespace) -> bool:
    return (
        _sparse_dense_pair_contract_kind() == "exact_speedup_verdict"
        and str(args.mode) == "sparse"
    )


def _tp8_selected_gpu_indices(args: argparse.Namespace) -> list[int]:
    fields = str(args.cuda_visible_devices).split(",")
    if len(fields) != 8 or any(not field.isdigit() for field in fields):
        raise ValueError(
            "exact TP8 lifecycle requires exactly eight integer GPU IDs"
        )
    gpu_ids = [int(field) for field in fields]
    if len(set(gpu_ids)) != 8:
        raise ValueError("exact TP8 lifecycle GPU IDs must be unique")
    return gpu_ids


def _start_tp8_exact_lifecycle(
    args: argparse.Namespace,
    *,
    output_path: Path,
) -> dict[str, Any]:
    gpu_ids = _tp8_selected_gpu_indices(args)
    run_nonce = str(getattr(args, "run_nonce", "") or "")
    if not run_nonce:
        raise ValueError("exact TP8 lifecycle requires a non-empty run nonce")
    baseline_path = _default_tp8_process_baseline_path(output_path)
    evidence, baseline_reasons = capture_tp8_process_baseline(gpu_ids)
    write_tp8_process_evidence(baseline_path, evidence)
    return {
        "gpu_ids": gpu_ids,
        "baseline_path": baseline_path,
        "baseline_sha256": _file_sha256(baseline_path),
        "baseline_evidence": evidence,
        "baseline_reasons": list(baseline_reasons),
        "run_nonce": run_nonce,
        "setup_records": [],
        "arm_records": [],
        "gate_reasons": [
            f"baseline:{reason}" for reason in baseline_reasons
        ],
    }


def _tp8_exact_arm_env(
    env: dict[str, str],
    state: dict[str, Any],
    arm_tag: str,
) -> tuple[dict[str, str], str]:
    arm_token = derive_arm_token(str(state["run_nonce"]), arm_tag)
    arm_env = dict(env)
    arm_env[ARM_TOKEN_ENV] = arm_token
    return arm_env, arm_token


def _redact_tp8_arm_token_result(
    result: Phase1CommandResult,
    arm_token: str,
) -> Phase1CommandResult:
    """Keep arm ownership secrets out of persisted stdout/stderr tails."""
    if not arm_token:
        return result
    return replace(
        result,
        stdout=result.stdout.replace(arm_token, "[tp8-arm-token-redacted]"),
        stderr=result.stderr.replace(arm_token, "[tp8-arm-token-redacted]"),
    )


def _finish_tp8_exact_arm(
    state: dict[str, Any],
    *,
    output_path: Path,
    arm_tag: str,
    result: Phase1CommandResult,
    arm_token: str,
    setup: bool = False,
) -> bool:
    records_key = "setup_records" if setup else "arm_records"
    expected_order = _TP8_EXACT_SETUP_ARM_ORDER if setup else _TP8_EXACT_ARM_ORDER
    arm_records = state[records_key]
    if len(arm_records) >= len(expected_order):
        raise ValueError(f"exact TP8 {records_key} already complete")
    expected_arm = expected_order[len(arm_records)]
    if arm_tag != expected_arm:
        raise ValueError(
            f"exact TP8 arm order mismatch: expected {expected_arm}, got {arm_tag}"
        )
    if arm_token != derive_arm_token(str(state["run_nonce"]), arm_tag):
        raise ValueError(f"exact TP8 arm token mismatch for {arm_tag}")
    health_reasons: list[str] = []
    if result.returncode != 0:
        health_reasons.append(f"returncode_nonzero:{result.returncode}")
    if result.timed_out:
        health_reasons.append("timed_out")
    fatal_error_detected = _command_output_has_fatal_error(result)
    if fatal_error_detected:
        health_reasons.append("fatal_child_output")
    child_session_id = result.child_session_id
    if (
        child_session_id is not None
        and (
            isinstance(child_session_id, bool)
            or not isinstance(child_session_id, int)
            or child_session_id <= 0
        )
    ):
        health_reasons.append("child_session_id_invalid")
        child_session_id = None
    elif child_session_id is None and result.returncode == 0:
        health_reasons.append("child_session_id_missing")
    teardown_path = _default_tp8_arm_teardown_path(output_path, arm_tag)
    evidence, teardown_reasons = capture_tp8_arm_teardown(
        state["gpu_ids"],
        baseline_path=state["baseline_path"],
        arm_tag=arm_tag,
        child_session_id=child_session_id,
        arm_token=arm_token,
    )
    write_tp8_process_evidence(teardown_path, evidence)
    passed = not health_reasons and not teardown_reasons
    record = {
        "arm_tag": arm_tag,
        "result_returncode": int(result.returncode),
        "result_timed_out": bool(result.timed_out),
        "fatal_error_detected": fatal_error_detected,
        "health_reasons": health_reasons,
        "child_session_id": child_session_id,
        "arm_token_sha256": arm_token_sha256(arm_token),
        "teardown_path": str(teardown_path.resolve(strict=False)),
        "teardown_sha256": _file_sha256(teardown_path),
        "teardown_evidence": evidence,
        "teardown_reasons": list(teardown_reasons),
        "passed": passed,
    }
    arm_records.append(record)
    state["gate_reasons"].extend(
        f"{arm_tag}:health:{reason}" for reason in health_reasons
    )
    state["gate_reasons"].extend(
        f"{arm_tag}:teardown:{reason}" for reason in teardown_reasons
    )
    return passed


def _apply_tp8_exact_lifecycle_payload(
    payload: dict[str, Any],
    state: dict[str, Any] | None,
) -> None:
    if state is None:
        return
    setup_execution_order = [
        record["arm_tag"] for record in state["setup_records"]
    ]
    setup_complete = setup_execution_order == list(_TP8_EXACT_SETUP_ARM_ORDER)
    execution_order = [record["arm_tag"] for record in state["arm_records"]]
    complete = execution_order == list(_TP8_EXACT_ARM_ORDER)
    reasons = list(state["gate_reasons"])
    if not setup_complete:
        reasons.append("setup_arm_sequence_incomplete")
    if not complete:
        reasons.append("arm_sequence_incomplete")
    passed = setup_complete and complete and not reasons
    payload.update(
        {
            "tp8_arm_lifecycle_required": True,
            "tp8_arm_lifecycle_expected_order": list(_TP8_EXACT_ARM_ORDER),
            "tp8_arm_lifecycle_execution_order": execution_order,
            "tp8_arm_lifecycle_setup_expected_order": list(
                _TP8_EXACT_SETUP_ARM_ORDER
            ),
            "tp8_arm_lifecycle_setup_execution_order": setup_execution_order,
            "tp8_arm_lifecycle_baseline": {
                "path": str(state["baseline_path"].resolve(strict=False)),
                "sha256": state["baseline_sha256"],
                "evidence": state["baseline_evidence"],
            },
            "tp8_arm_lifecycle_setup_records": state["setup_records"],
            "tp8_arm_lifecycle_records": state["arm_records"],
            "tp8_arm_lifecycle_gate_passed": passed,
            "tp8_arm_lifecycle_gate_reasons": reasons,
        }
    )
    if not passed:
        payload["gate_passed"] = False
        payload["production_gate_passed"] = False


def _is_gate_d_trace_profile_key(key: str) -> bool:
    # Historical helper/artifact name retained for result-schema stability;
    # the shared classifier now covers every timed-child observer and
    # experimental ablation, not only trace/profile keys.
    return classify_speed_child_env_key(key) is not None


def _clear_gate_d_trace_profile_env(env: dict[str, str]) -> None:
    for key in list(env):
        if _is_gate_d_trace_profile_key(key):
            env.pop(key, None)
    for key in ALLOWED_TRACE_ENV_KEYS:
        env.pop(key, None)


def _copy_mixed_page_kernel_profile_env_for_diagnostic(env: dict[str, str]) -> None:
    value = os.environ.get("VLLM_SPARSE_MIXED_PAGE_KERNEL_PROFILE_LOG", "").strip()
    if value:
        env["VLLM_SPARSE_MIXED_PAGE_KERNEL_PROFILE_LOG"] = value


def _copy_sparse_metadata_profile_env_for_diagnostic(env: dict[str, str]) -> None:
    for key in (
        "VLLM_SPARSE_MB_PROFILE_LOG",
        "VLLM_SPARSE_RRP_PREP_PROFILE_LOG",
        # [T1-FORENSIC 2026-07-09] refresh flush host cProfile 采样目录
        # (名字含 PROFILE 命中 gate-D 通配清洗,须显式回填诊断 child)。
        "VLLM_SPARSE_REFRESH_CPROFILE_DIR",
        # [T2-FORENSIC 2026-07-10] 世代 commit 步 metadata builder host 分相
        # cProfile 采样目录(同上须回填)。
        "VLLM_SPARSE_MB_CPROFILE_DIR",
        # [S7-FORENSIC 2026-07-10] 兑现窗 off-loop enqueue/body 细分计时
        # (pending_group_enqueue_* 键随 stage_profile 进 hook_profile;名字含
        # PROFILE 命中 gate-D 通配清洗,须显式回填诊断 child)。
        "VLLM_SPARSE_REPLAY_REFRESH_ENQUEUE_PROFILE_DETAIL",
        # [S7-FORENSIC 2026-07-10] off-loop selector 组装 12 分量 host 分相
        # (deadline_deferred_selector_* 计数族;同上须回填,env 关=count 恒 0)。
        "VLLM_SPARSE_DEFERRED_SELECTOR_PROFILE_DETAIL",
        # [S7-FORENSIC 2026-07-10] off-loop body 内 GPU 事件段时长
        # (async_producer_*_gpu_ms direct event-pair schema;同上须回填)。
        "VLLM_SPARSE_ASYNC_PRODUCER_GPU_PROFILE",
    ):
        value = os.environ.get(key, "").strip()
        if value:
            env[key] = value


def _copy_torch_profiler_env_for_diagnostic(env: dict[str, str]) -> None:
    # VLLM_TORCH_PROFILER_DIR 命中 _is_gate_d_trace_profile_key 的通配清洗;
    # 仅诊断 child 回填(speed child 保持无 profiler,不扰测速)。
    value = os.environ.get("VLLM_TORCH_PROFILER_DIR", "").strip()
    if value:
        env["VLLM_TORCH_PROFILER_DIR"] = value


def _copy_refresh_micro_profile_env_for_diagnostic(env: dict[str, str]) -> None:
    if os.environ.get("VLLM_SPARSE_REFRESH_MICRO_PROFILE", "") == "1":
        env["VLLM_SPARSE_REFRESH_MICRO_PROFILE"] = "1"
        every = os.environ.get("VLLM_SPARSE_REFRESH_MICRO_PROFILE_EVERY", "")
        if every:
            env["VLLM_SPARSE_REFRESH_MICRO_PROFILE_EVERY"] = str(every)


def _copy_step_profile_env_for_diagnostic(env: dict[str, str]) -> None:
    # VLLM_SPARSE_STEP_PROFILE 本体命中 _is_gate_d_trace_profile_key 的通配
    # 清洗(_EVERY/_LOG/_DETAIL 尾缀不命中而幸存)——与 torch profiler 同款
    # 回填:仅当调用方显式导出时透传,默认零行为。
    for key in (
        "VLLM_SPARSE_STEP_PROFILE",
        "VLLM_SPARSE_STEP_PROFILE_DETAIL",
        "VLLM_SPARSE_STEP_PROFILE_EVERY",
        "VLLM_SPARSE_STEP_PROFILE_LOG",
    ):
        value = os.environ.get(key, "").strip()
        if value:
            env[key] = value


def _copy_selector_profile_env_for_diagnostic(env: dict[str, str]) -> None:
    for key in SELECTOR_PIPELINE_CPU_PROFILE_ENV_KEYS:
        value = os.environ.get(key, "")
        if value:
            env[key] = str(value)


def _gate_d_trace_profile_env(env: dict[str, str]) -> dict[str, str]:
    return {
        key: str(value or "")
        for key, value in sorted(env.items())
        if _is_gate_d_trace_profile_key(key) and str(value or "")
    }


def _gate_d_trace_profile_env_empty(env: dict[str, str]) -> bool:
    return all(not value for value in _gate_d_trace_profile_env(env).values())


def _run_text_command(command: list[str], *, timeout_s: float = 10.0) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            command,
            cwd=str(_REPO_ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=float(timeout_s),
            check=False,
        )
    except Exception as exc:
        return {
            "command": command,
            "returncode": -1,
            "stdout": "",
            "stderr": repr(exc),
        }
    return {
        "command": command,
        "returncode": int(proc.returncode),
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def _first_stdout_line(result: dict[str, Any]) -> str:
    if int(result.get("returncode", -1)) != 0:
        return ""
    stdout = str(result.get("stdout", "") or "")
    return stdout.splitlines()[0] if stdout else ""


def _gpu_snapshot(label: str, *, cuda_visible_devices: str) -> dict[str, Any]:
    return {
        "label": label,
        "cuda_visible_devices": str(cuda_visible_devices),
        "query": _run_text_command(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            timeout_s=10.0,
        ),
        "pmon": _run_text_command(["nvidia-smi", "pmon", "-c", "1"], timeout_s=10.0),
    }


def _gate_d_backend(args: argparse.Namespace) -> str:
    return str(getattr(args, "backend", BACKEND_FA3) or BACKEND_FA3)


def _gate_d_flash_attn_version(args: argparse.Namespace) -> str:
    return "4" if _gate_d_backend(args) == BACKEND_FA4_SM100 else "3"


def _resolve_gate_d_backend_artifact(args: argparse.Namespace) -> Path | None:
    if _gate_d_backend(args) == BACKEND_FA4_SM100:
        return (
            Path(str(args.fa3_upstream_root)).expanduser()
            / "flash_attn"
            / "cute"
            / "interface.py"
        ).resolve()
    return _resolve_fa3_so_path(args)


def _apply_gate_d_backend_env(args: argparse.Namespace, env: dict[str, str]) -> None:
    env["VLLM_SPARSE_FA3_UPSTREAM_ROOT"] = str(args.fa3_upstream_root)
    env["VLLM_FLASH_ATTN_VERSION"] = _gate_d_flash_attn_version(args)


def _file_provenance(path: Path | None, *, sha256: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "path": str(path) if path is not None else "",
        "sha256": str(sha256),
        "exists": bool(path is not None and path.exists()),
    }
    if path is not None and path.exists():
        stat = path.stat()
        payload.update({"mtime_ns": int(stat.st_mtime_ns), "size_bytes": int(stat.st_size)})
    return payload


def _run_provenance_payload(
    args: argparse.Namespace,
    *,
    env: dict[str, str],
    fa3_so_sha256: str,
    gpu_before: dict[str, Any],
    gpu_after: dict[str, Any],
) -> dict[str, Any]:
    git_head = _run_text_command(["git", "rev-parse", "HEAD"], timeout_s=5.0)
    git_status = _run_text_command(["git", "status", "--short"], timeout_s=5.0)
    git_tracked_status = _run_text_command(
        ["git", "status", "--short", "--untracked-files=no", "--", "."],
        timeout_s=5.0,
    )
    build_identity_raw = str(
        env.get("SFI_RUNNER_ATTENTION_BUILD_IDENTITY_JSON", "") or ""
    )
    try:
        build_identity = json.loads(build_identity_raw) if build_identity_raw else {}
    except (TypeError, ValueError, json.JSONDecodeError):
        build_identity = {"invalid_json": build_identity_raw}

    def _env_int(name: str) -> int | None:
        raw = str(env.get(name, "") or "")
        return int(raw) if raw.isdigit() else None

    model_kv_contract = {
        "schema": str(
            env.get("SFI_RUNNER_MODEL_KV_CONTRACT_SCHEMA", "") or ""
        ),
        "model_config_path": str(
            env.get("SFI_RUNNER_MODEL_CONFIG_PATH", "") or ""
        ),
        "model_config_sha256": str(
            env.get("SFI_RUNNER_MODEL_CONFIG_SHA256", "") or ""
        ),
        "num_hidden_layers": _env_int("SFI_RUNNER_MODEL_NUM_HIDDEN_LAYERS"),
        "num_key_value_heads": _env_int(
            "SFI_RUNNER_MODEL_NUM_KEY_VALUE_HEADS"
        ),
        "head_dim": _env_int("SFI_RUNNER_MODEL_HEAD_DIM"),
        "dtype": str(env.get("SFI_RUNNER_MODEL_KV_DTYPE", "") or ""),
        "dtype_bytes": _env_int("SFI_RUNNER_MODEL_KV_DTYPE_BYTES"),
        "total_bytes_per_token": _env_int(
            "SFI_RUNNER_MODEL_KV_TOTAL_BYTES_PER_TOKEN"
        ),
        "per_rank_bytes_per_token_requested": _env_int(
            "SFI_RUNNER_KV_TOKEN_BYTES_PER_RANK_REQUESTED"
        ),
        "per_rank_bytes_per_token_effective": _env_int(
            "SFI_RUNNER_KV_TOKEN_BYTES_PER_RANK_EFFECTIVE"
        ),
        "override_present": str(
            env.get("SFI_RUNNER_KV_TOKEN_BYTES_OVERRIDE_PRESENT", "") or ""
        ),
    }
    # Keep provenance on the same source of truth as the runtime. A hard-coded
    # fallback here mislabeled the promoted default-18 execution as chunk 14,
    # which in turn selected the wrong output anchor in postflight tooling.
    from patches.sparse_constants import (
        resolve_capture_chunk,
        resolve_writer_token_tile,
    )

    capture_chunk = resolve_capture_chunk(env)
    writer_token_tile = resolve_writer_token_tile(env)
    refresh_stream_priority = int(
        env.get("VLLM_SPARSE_REFRESH_STREAM_PRIORITY", "0") or "0"
    )
    refresh_rebuild_max_delay_steps = int(
        env.get("VLLM_SPARSE_REFRESH_REBUILD_MAX_DELAY_STEPS", "0") or "0"
    )
    refresh_rebuild_max_delay_steps_env = int(refresh_rebuild_max_delay_steps)
    clean_metadata_effective = (
        str(env.get("VLLM_SPARSE_CLEAN_METADATA", "1") or "1") == "1"
    )
    return {
        "run_nonce": str(getattr(args, "run_nonce", "") or ""),
        "git_head": _first_stdout_line(git_head),
        "git_status_short": str(git_status.get("stdout", "") or ""),
        "git_head_command": git_head,
        "git_status_command": git_status,
        "git_tracked_status_short": str(
            git_tracked_status.get("stdout", "") or ""
        ),
        "git_tracked_status_command": git_tracked_status,
        "python": str(args.python),
        "preset": str(getattr(args, "preset", "") or ""),
        "model": str(args.model),
        "prompt": str(args.prompt),
        "prompt_artifact": _file_provenance(
            Path(str(args.prompt)).resolve(),
            sha256=str(env.get("SFI_RUNNER_CORPUS_SHA256", "") or ""),
        ),
        "mode": str(args.mode),
        "batch_size": int(args.batch_size),
        "request_context_tokens": _request_context_tokens(args),
        "request_max_new_tokens": _request_max_new_tokens(args),
        "tensor_parallel_size": int(
            getattr(args, "tensor_parallel_size", 1) or 1
        ),
        "max_model_len": int(args.max_model_len),
        "max_num_seqs": _effective_max_num_seqs(args),
        "max_num_batched_tokens": int(args.max_num_batched_tokens),
        "chunked_prefill": str(args.chunked_prefill),
        "max_seq_len_to_capture": int(args.max_seq_len_to_capture),
        "scheduler_graph_contract_requested": requested_scheduler_graph_contract(
            args
        ),
        "scheduling_mode_requested": str(
            getattr(args, "scheduling_mode", "auto") or "auto"
        ),
        "warmup": int(args.warmup),
        "output_len": _effective_max_new_tokens(args),
        "max_new_tokens": _effective_max_new_tokens(args),
        "max_new_tokens_effective": _effective_max_new_tokens(args),
        "max_new_tokens_was_explicit": bool(
            getattr(args, "max_new_tokens_was_explicit", False)
        ),
        "iters": int(args.iters),
        "legacy_iters": int(getattr(args, "legacy_iters", args.iters)),
        "iters_was_explicit": bool(getattr(args, "iters_was_explicit", False)),
        "iters_semantics": "legacy_alias_for_max_new_tokens",
        "repeat": int(args.iters),
        "repeat_semantics": "legacy_alias_for_max_new_tokens",
        "full_cuda_graph": bool(args.full_cuda_graph),
        "backend": _gate_d_backend(args),
        "flash_attn_version_expected": int(_gate_d_flash_attn_version(args)),
        "backend_artifact": _file_provenance(
            _resolve_gate_d_backend_artifact(args),
            sha256=fa3_so_sha256,
        ),
        "split_context_prompts": bool(args.split_context_prompts),
        "respect_eos": bool(args.respect_eos),
        "outputs_include_text": bool(args.outputs_include_text),
        "chat_template": bool(args.chat_template),
        "enable_thinking": bool(args.enable_thinking),
        "gpu_mem_util": float(args.gpu_mem_util),
        "producer_mode": _producer_mode(args),
        "refresh_interval": int(args.refresh_interval),
        "trigger_min_gap": int(args.trigger_min_gap),
        "sentence_cooldown": int(args.sentence_cooldown),
        "refresh_coalesce_window": int(args.refresh_coalesce_window),
        "fa3_upstream_root": str(args.fa3_upstream_root),
        "compact_blocks_per_slot": int(args.compact_blocks_per_slot),
        "max_live_sparse_slots": int(args.max_live_sparse_slots),
        "recent": int(args.recent),
        "prefill_last_n_query": max(0, int(getattr(args, "prefill_last_n", 16))),
        "capture_chunk_effective": int(capture_chunk),
        "writer_token_tile_effective": int(writer_token_tile),
        "clean_metadata_effective": bool(clean_metadata_effective),
        "refresh_stream_priority_effective": int(refresh_stream_priority),
        "refresh_rebuild_max_delay_steps_env": int(
            refresh_rebuild_max_delay_steps_env
        ),
        "refresh_rebuild_max_delay_steps_effective": int(
            refresh_rebuild_max_delay_steps
        ),
        "full_cudagraph_replay_refresh_batched_flush": str(
            env.get("VLLM_SPARSE_FULL_CUDAGRAPH_REPLAY_REFRESH_BATCHED_FLUSH", "") or ""
        ),
        "full_cudagraph_replay_refresh_defer_to_deadline": str(
            env.get("VLLM_SPARSE_FULL_CUDAGRAPH_REPLAY_REFRESH_DEFER_TO_DEADLINE", "") or ""
        ),
        "full_cudagraph_replay_refresh_progressive_consume": str(
            env.get("VLLM_SPARSE_REPLAY_REFRESH_PROGRESSIVE_CONSUME", "") or ""
        ),
        "refresh_enqueue_stagger": str(
            env.get("VLLM_SPARSE_REFRESH_ENQUEUE_STAGGER", "") or ""
        ),
        "refresh_split_selector_writer_release": str(
            env.get("VLLM_SPARSE_REFRESH_SPLIT_SELECTOR_WRITER_RELEASE", "") or ""
        ),
        "refresh_split_selector_writer_release_effective": str(
            env.get("VLLM_SPARSE_REFRESH_SPLIT_SELECTOR_WRITER_RELEASE", "auto")
            or "auto"
        ),
        "refresh_split_selector_writer_release_max_per_handle": str(
            env.get(
                "VLLM_SPARSE_REFRESH_SPLIT_SELECTOR_WRITER_RELEASE_MAX_PER_HANDLE",
                "1",
            )
            or "1"
        ),
        "refresh_split_selector_writer_release_min_layer_start": str(
            env.get(
                "VLLM_SPARSE_REFRESH_SPLIT_SELECTOR_WRITER_RELEASE_MIN_LAYER_START",
                "0",
            )
            or "0"
        ),
        "one_shot_group_ready": str(env.get("VLLM_SPARSE_ONE_SHOT_GROUP_READY", "") or ""),
        "one_shot_ready_chunk": str(env.get("VLLM_SPARSE_ONE_SHOT_READY_CHUNK", "") or ""),
        "cuda_visible_devices_arg": str(args.cuda_visible_devices),
        "cuda_visible_devices_env": str(env.get("CUDA_VISIBLE_DEVICES", "") or ""),
        "runner_kv_preflight_status": str(
            env.get("SFI_RUNNER_KV_PREFLIGHT_STATUS", "") or ""
        ),
        "runner_kv_required_token_blocks": _env_int(
            "SFI_RUNNER_KV_REQUIRED_TOKEN_BLOCKS"
        ),
        "runner_kv_required_tokens_padded": _env_int(
            "SFI_RUNNER_KV_REQUIRED_TOKENS_PADDED"
        ),
        "runner_kv_required_bytes": _env_int(
            "SFI_RUNNER_KV_REQUIRED_BYTES"
        ),
        "runner_kv_compact_lease_bytes": _env_int(
            "SFI_RUNNER_KV_COMPACT_LEASE_BYTES"
        ),
        "runner_max_request_sequence_tokens": _env_int(
            "SFI_RUNNER_MAX_REQUEST_SEQUENCE_TOKENS"
        ),
        "runner_gpu_lock_mode": str(
            env.get("SFI_RUNNER_GPU_LOCK_MODE", "") or ""
        ),
        "runner_gpu_lock_scope": str(
            env.get("SFI_RUNNER_GPU_LOCK_SCOPE", "") or ""
        ),
        "runner_fa3_preflight_status": str(
            env.get("SFI_RUNNER_FA3_PREFLIGHT_STATUS", "") or ""
        ),
        "runner_attention_preflight_status": str(
            env.get("SFI_RUNNER_ATTENTION_PREFLIGHT_STATUS", "") or ""
        ),
        "runner_attention_arch": str(
            env.get("SFI_RUNNER_ATTENTION_ARCH", "") or ""
        ),
        "runner_attention_kernel": str(
            env.get("SFI_RUNNER_ATTENTION_KERNEL", "") or ""
        ),
        "runner_attention_backend": str(
            env.get("SFI_RUNNER_ATTENTION_BACKEND", "") or ""
        ),
        "runner_flash_attn_version": str(
            env.get("SFI_RUNNER_FLASH_ATTN_VERSION", "") or ""
        ),
        "runner_attention_build_identity": build_identity,
        "runner_expected_git_commit": str(
            env.get("SFI_RUNNER_EXPECTED_GIT_COMMIT", "") or ""
        ),
        "runner_expected_model_config_sha256": str(
            env.get("SFI_RUNNER_EXPECTED_MODEL_CONFIG_SHA256", "") or ""
        ),
        "runner_git_head": str(env.get("SFI_RUNNER_GIT_HEAD", "") or ""),
        "runner_git_tracked_clean": str(
            env.get("SFI_RUNNER_GIT_TRACKED_CLEAN", "") or ""
        ),
        "runner_code_scope": str(
            env.get("SFI_RUNNER_CODE_SCOPE", "") or ""
        ),
        "runner_code_scope_untracked_status": str(
            env.get("SFI_RUNNER_CODE_SCOPE_UNTRACKED_STATUS", "") or ""
        ),
        "runner_code_scope_untracked_clean": str(
            env.get("SFI_RUNNER_CODE_SCOPE_UNTRACKED_CLEAN", "") or ""
        ),
        "runner_candidate_full_clean_at_pair_start": str(
            env.get("SFI_RUNNER_CANDIDATE_FULL_CLEAN_AT_PAIR_START", "") or ""
        ),
        "runner_cuda_capabilities": str(
            env.get("SFI_RUNNER_CUDA_CAPABILITIES", "") or ""
        ),
        "runner_gpu_total_memory_bytes": str(
            env.get("SFI_RUNNER_GPU_TOTAL_MEMORY_BYTES", "") or ""
        ),
        "runner_gpu_free_memory_bytes": str(
            env.get("SFI_RUNNER_GPU_FREE_MEMORY_BYTES", "") or ""
        ),
        "runner_gpu_physical_capacity_status": str(
            env.get("SFI_RUNNER_GPU_PHYSICAL_CAPACITY_STATUS", "") or ""
        ),
        "runner_selector_cache_root": str(
            env.get("SFI_RUNNER_SELECTOR_CACHE_ROOT", "") or ""
        ),
        "runner_model_kv_contract": model_kv_contract,
        "runner_kv_token_bytes_per_rank_requested": model_kv_contract[
            "per_rank_bytes_per_token_requested"
        ],
        "runner_kv_token_bytes_per_rank_effective": model_kv_contract[
            "per_rank_bytes_per_token_effective"
        ],
        "runner_corpus_token_status": str(
            env.get("SFI_RUNNER_CORPUS_TOKEN_STATUS", "") or ""
        ),
        "runner_corpus_path": str(
            env.get("SFI_RUNNER_CORPUS_PATH", "") or ""
        ),
        "runner_corpus_sha256": str(
            env.get("SFI_RUNNER_CORPUS_SHA256", "") or ""
        ),
        "runner_corpus_manifest_path": str(
            env.get("SFI_RUNNER_CORPUS_MANIFEST_PATH", "") or ""
        ),
        "runner_corpus_manifest_sha256": str(
            env.get("SFI_RUNNER_CORPUS_MANIFEST_SHA256", "") or ""
        ),
        "runner_corpus_manifest_schema": str(
            env.get("SFI_RUNNER_CORPUS_MANIFEST_SCHEMA", "") or ""
        ),
        "runner_corpus_layout_mode": str(
            env.get("SFI_RUNNER_CORPUS_LAYOUT_MODE", "") or ""
        ),
        "runner_corpus_layout_validation_contract": str(
            env.get(
                "SFI_RUNNER_CORPUS_LAYOUT_VALIDATION_CONTRACT",
                "",
            )
            or ""
        ),
        "runner_corpus_layout_verified_count": _as_int(
            env.get("SFI_RUNNER_CORPUS_LAYOUT_VERIFIED_COUNT", "0"),
            0,
        ),
        "runner_chat_template_reserve_tokens": _as_int(
            env.get("SFI_RUNNER_CHAT_TEMPLATE_RESERVE_TOKENS", "0"),
            0,
        ),
        "runner_chat_template_overhead_min": _as_int(
            env.get("SFI_RUNNER_CHAT_TEMPLATE_OVERHEAD_MIN", "-1"),
            -1,
        ),
        "runner_chat_template_overhead_max": _as_int(
            env.get("SFI_RUNNER_CHAT_TEMPLATE_OVERHEAD_MAX", "-1"),
            -1,
        ),
        "runner_chat_template_verified_count": _as_int(
            env.get("SFI_RUNNER_CHAT_TEMPLATE_VERIFIED_COUNT", "0"),
            0,
        ),
        "runner_tier": str(env.get("SFI_RUNNER_TIER", "") or ""),
        "runner_mode": str(env.get("SFI_RUNNER_MODE", "") or ""),
        "runner_tensor_parallel_size": str(
            env.get("SFI_RUNNER_TENSOR_PARALLEL_SIZE", "") or ""
        ),
        "runner_batch_size": str(
            env.get("SFI_RUNNER_BATCH_SIZE", "") or ""
        ),
        "runner_context_tokens": str(
            env.get("SFI_RUNNER_CONTEXT_TOKENS", "") or ""
        ),
        "runner_request_context_tokens": str(
            env.get("SFI_RUNNER_REQUEST_CONTEXT_TOKENS", "")
            or env.get("REQUEST_CONTEXT_TOKENS", "")
            or ""
        ),
        "runner_kv_cache_memory_bytes": str(
            env.get("SFI_RUNNER_KV_CACHE_MEMORY_BYTES", "") or ""
        ),
        "runner_max_model_len": str(
            env.get("SFI_RUNNER_MAX_MODEL_LEN", "") or ""
        ),
        "runner_max_new_tokens": str(
            env.get("SFI_RUNNER_MAX_NEW_TOKENS", "") or ""
        ),
        "runner_request_max_new_tokens": str(
            env.get("SFI_RUNNER_REQUEST_MAX_NEW_TOKENS", "")
            or env.get("REQUEST_MAX_NEW_TOKENS", "")
            or ""
        ),
        "runner_max_num_seqs": str(
            env.get("SFI_RUNNER_MAX_NUM_SEQS", "") or ""
        ),
        "runner_max_num_batched_tokens": str(
            env.get("SFI_RUNNER_MAX_NUM_BATCHED_TOKENS", "") or ""
        ),
        "runner_chunked_prefill": str(
            env.get("SFI_RUNNER_CHUNKED_PREFILL", "") or ""
        ),
        "runner_max_seq_len_to_capture": str(
            env.get("SFI_RUNNER_MAX_SEQ_LEN_TO_CAPTURE", "") or ""
        ),
        "runner_refresh_interval": str(
            env.get("SFI_RUNNER_REFRESH_INTERVAL", "") or ""
        ),
        "runner_compact_blocks_per_slot": str(
            env.get("SFI_RUNNER_COMPACT_BLOCKS_PER_SLOT", "") or ""
        ),
        "runner_compact_dual_gen": str(
            env.get("SFI_RUNNER_COMPACT_DUAL_GEN", "") or ""
        ),
        "runner_pytorch_alloc_conf": str(
            env.get("SFI_RUNNER_PYTORCH_ALLOC_CONF", "") or ""
        ),
        "runner_pytorch_cuda_alloc_conf": str(
            env.get("SFI_RUNNER_PYTORCH_CUDA_ALLOC_CONF", "") or ""
        ),
        "runner_custom_ar_disabled": str(
            env.get("SFI_RUNNER_CUSTOM_AR_DISABLED", "") or ""
        ),
        "fa3_so": _file_provenance(
            _resolve_fa3_so_path(args),
            sha256=fa3_so_sha256,
        ),
        "gpu_before": gpu_before,
        "gpu_after": gpu_after,
    }


def _build_gate_d_dense_command(
    args: argparse.Namespace,
    *,
    metrics_path: Path,
    outputs_path: Path,
    collect_cudagraph_runtime_proof: bool = False,
) -> list[str]:
    repo_root = _REPO_ROOT
    command = [
        str(args.python),
        str(repo_root / "benchmarks" / "run_dense_only.py"),
        "--model",
        str(args.model),
        "--prompt",
        str(args.prompt),
        "--batch-size",
        str(int(args.batch_size)),
        "--max-num-seqs",
        str(_effective_max_num_seqs(args)),
        "--max-new-tokens",
        str(_effective_max_new_tokens(args)),
        "--scheduling-mode",
        str(args.scheduling_mode),
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
    ]
    _append_scheduler_graph_child_args(command, args)
    _append_request_vector_child_args(command, args)
    if int(getattr(args, "max_model_len", 0) or 0) > 0:
        command.extend(["--max-model-len", str(int(args.max_model_len))])
    if int(getattr(args, "tensor_parallel_size", 1) or 1) > 1:
        command.extend(
            ["--tensor-parallel-size", str(int(args.tensor_parallel_size))]
        )
    if bool(args.full_cuda_graph):
        command.extend(
            [
                "--full-cuda-graph",
                "--cudagraph-capture-sizes",
                str(int(args.batch_size)),
            ]
        )
    if collect_cudagraph_runtime_proof:
        command.append("--collect-cudagraph-runtime-proof")
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


def _build_gate_d_dense_env(
    args: argparse.Namespace,
    *,
    route_trace_path: Path | None = None,
) -> dict[str, str]:
    env = _build_dense_reference_env(args)
    _clear_gate_d_trace_profile_env(env)
    for key in DEFERRED_BRIDGE_ENV_KEYS:
        env.pop(key, None)
    _apply_gate_d_backend_env(args, env)
    env["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN_VLLM_V1"
    if route_trace_path is not None:
        env[ROUTE_TRACE_ENV_KEY] = str(route_trace_path)
    return env


def _build_gate_d_sparse_env(
    args: argparse.Namespace,
    *,
    route_trace_path: Path | None,
    one_shot_timeline_path: Path | None = None,
) -> dict[str, str]:
    env = _build_phase1_env(
        args,
        route_trace_path=route_trace_path or Path(""),
    )
    _clear_gate_d_trace_profile_env(env)
    # [GATE-D-SPEED-ENV-PURITY 2026-07-10] _copy_*_for_diagnostic 系列从本共用
    # 构建函数移出,只在 diagnostic_env 调用点施加:此前 speed child 也被回填
    # profile env,与 _copy_torch_profiler_env_for_diagnostic 自身注释("仅诊断
    # child 回填,speed child 保持无 profiler,不扰测速")矛盾;测速纯净性由
    # 上面的 _clear_gate_d_trace_profile_env 单点保证。
    _apply_gate_d_backend_env(args, env)
    if _gate_d_backend(args) == BACKEND_FA4_SM100:
        env["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN_VLLM_V1"
    env["VLLM_SPARSE_ASYNC_REFRESH"] = "1"
    env["VLLM_SPARSE_ONE_SHOT_ASYNC_BOOTSTRAP"] = "1"
    env["VLLM_SPARSE_CONTROLLER_JSON"] = json.dumps(
        _phase2_controller_payload(args),
        sort_keys=True,
    )
    _apply_gt1_full_cudagraph_refresh_env(env, args)
    _apply_deferred_bridge_env(env, args)
    workload_plan_replay = str(getattr(args, "workload_plan_replay", "") or "").strip()
    if workload_plan_replay:
        env[WORKLOAD_PLAN_REPLAY_ENV] = workload_plan_replay
    if route_trace_path is not None:
        env[ROUTE_TRACE_ENV_KEY] = str(route_trace_path)
    if one_shot_timeline_path is not None:
        env["VLLM_SPARSE_ONE_SHOT_TIMELINE_LOG"] = str(one_shot_timeline_path)
    return env


def _decode_metric(metrics: dict[str, Any], key: str) -> float:
    return _as_float(metrics.get(key), -1.0)


def _env_flag_enabled(env: dict[str, str] | None, key: str) -> bool:
    if not env:
        return False
    raw = str(env.get(key, "") or "").strip().lower()
    return raw not in {"", "0", "false", "no", "off"}


_FATAL_CHILD_OUTPUT_MARKERS = (
    "EngineCore encountered a fatal error",
    "CUDA error: an illegal memory access",
    "illegal memory access",
)


def _command_output_has_fatal_error(result: Phase1CommandResult | None) -> bool:
    if result is None:
        return False
    combined = f"{result.stdout}\n{result.stderr}"
    return any(marker in combined for marker in _FATAL_CHILD_OUTPUT_MARKERS)


def _diagnostic_route_counter_proof(
    metrics: dict[str, Any],
    *,
    require_resolved_row_ptr: bool = False,
    route_summary: dict[str, Any] | None = None,
    producer_route_summary: dict[str, Any] | None = None,
) -> dict[str, object]:
    reasons: list[str] = []
    route_summary = route_summary or {}
    producer_route_summary = producer_route_summary or {}
    available = bool(metrics.get("route_counter_available", False))
    actual = _as_int(metrics.get("route_counter_actual_fwd_mixed_page_count"), -1)
    resolved = _as_int(
        metrics.get("route_counter_resolved_row_ptr_fwd_mixed_page_count"),
        -1,
    )
    has_resolved = _as_int(
        metrics.get("route_counter_has_resolved_row_ptr_count"),
        -1,
    )
    kind_counts = tuple(
        _as_int(metrics.get(f"route_counter_page_resolver_kind{kind}_count"), -1)
        for kind in range(5)
    )
    kind0, kind1, kind2, kind3, kind4 = kind_counts
    compact_steps = _as_int(metrics.get("route_counter_compact_row_steps"), -1)
    compact_rows = _as_int(metrics.get("route_counter_compact_rows"), -1)
    route_family = "unknown"

    def _resolved_graph_family(summary: dict[str, Any]) -> bool:
        graph_route_family = summary.get("graph_route_family", {}) or {}
        return bool(
            isinstance(graph_route_family, dict)
            and str(graph_route_family.get("captured_route_family", ""))
            == "resolved_row_ptr"
            and str(graph_route_family.get("current_route_family", ""))
            == "resolved_row_ptr"
            and not bool(graph_route_family.get("route_family_mismatch", True))
        )

    def _resolved_graph_replay(summary: dict[str, Any]) -> bool:
        return bool(
            summary.get("full_graph_replay_refresh_seen", False)
            and _resolved_graph_family(summary)
            and _as_int(
                summary.get("resolved_row_ptr_fwd_mixed_page_count", 0),
                0,
            )
            > 0
            and _as_int(summary.get("page_resolver_kind4_count", 0), 0) > 0
        )

    if not available:
        reasons.append("diagnostic_route_counters_missing")
    if actual <= 0:
        reasons.append("diagnostic_actual_fwd_mixed_page_count_missing")
    if available and actual > 0:
        if any(count < 0 for count in kind_counts):
            reasons.append("diagnostic_page_resolver_kind_partition_missing")
        elif sum(kind_counts) != actual:
            reasons.append("diagnostic_page_resolver_kind_partition_mismatch")
        if resolved < 0 or kind4 < 0 or resolved != kind4:
            reasons.append("diagnostic_resolved_kind4_count_mismatch")
        if any(count > 0 for count in (kind1, kind2, kind3)):
            reasons.append("diagnostic_page_resolver_route_polluted")
        if not 0 <= resolved <= has_resolved <= actual:
            reasons.append("diagnostic_resolved_row_ptr_bounds_invalid")
        if compact_steps <= 0 or compact_rows < compact_steps:
            reasons.append("diagnostic_compact_replay_liveness_missing")

        graph_replay_rrp = any(
            _resolved_graph_replay(summary)
            for summary in (producer_route_summary, route_summary)
        )
        if require_resolved_row_ptr:
            if resolved <= 0:
                reasons.append("diagnostic_resolved_row_ptr_count_missing")
            if kind4 <= 0:
                reasons.append("diagnostic_page_resolver_kind4_count_missing")
            if not graph_replay_rrp:
                reasons.append("diagnostic_resolved_graph_replay_missing")

        if graph_replay_rrp:
            # Python route counters observe eager/capture calls; CUDA graph
            # replay does not re-enter the callable.  kind0 may therefore come
            # from prefill or capture and is not itself a replay fallback.  The
            # graph-family trace plus compact step liveness owns that verdict.
            route_family = (
                "resolved_row_ptr_graph_replay_with_native_python_calls"
                if kind0 > 0
                else "resolved_row_ptr_graph_replay"
            )
        elif resolved > 0 and kind4 > 0:
            route_family = (
                "resolved_row_ptr_kind4_with_native_python_calls"
                if kind0 > 0
                else "resolved_row_ptr_kind4"
            )
        elif kind0 == actual and resolved == 0 and kind4 == 0:
            route_family = "dense_native_kind0"
    return {
        "passed": not reasons,
        "reasons": reasons,
        "scope": "diagnostic_child_python_calls_and_step_liveness",
        "route_family": route_family,
        "actual_fwd_mixed_page_count": int(actual),
        "resolved_row_ptr_fwd_mixed_page_count": int(resolved),
        "has_resolved_row_ptr_count": int(has_resolved),
        "page_resolver_kind0_count": int(kind0),
        "page_resolver_kind4_count": int(kind4),
        "compact_row_steps": int(compact_steps),
        "compact_rows": int(compact_rows),
    }


def _reference_text_gate_reasons(
    *,
    reference_result: Phase1CommandResult | None,
    reference_semantic_match: bool | None,
) -> list[str]:
    reasons: list[str] = []
    if reference_result is None:
        return ["reference_run_missing"]
    if int(reference_result.returncode) != 0:
        reasons.append("reference_returncode_nonzero")
    if bool(reference_result.timed_out):
        reasons.append("reference_timed_out")
    if reference_semantic_match is None:
        reasons.append("reference_semantic_match_missing")
    return reasons


def _reference_text_informational_reasons(
    reference_semantic_match: bool | None,
) -> list[str]:
    """Report cross-arm text differences without redefining runtime health."""

    return (
        ["reference_semantic_mismatch"]
        if reference_semantic_match is False
        else []
    )


def _full_generated_text_from_output_record(record: dict[str, Any]) -> str:
    for key in ("text", "generated_text", "output_text"):
        value = record.get(key)
        if isinstance(value, str):
            return value
    return ""


def _semantic_output_text_and_proof_reason(
    record: dict[str, Any],
) -> tuple[str, str]:
    """Return stop-bounded semantic text or a fail-closed proof reason."""

    full_text = _full_generated_text_from_output_record(record)
    proof_fields = (
        "semantic_text",
        "semantic_stop_seen",
        "semantic_first_stop_token_index",
        "semantic_stop_token_id",
        "semantic_stop_token_ids",
        "semantic_token_count",
    )
    if not any(field in record for field in proof_fields):
        return full_text, "semantic_stop_proof_missing"
    if any(field not in record for field in proof_fields):
        return full_text, "semantic_stop_proof_incomplete"

    token_ids = record.get("token_ids")
    semantic_text = record.get("semantic_text")
    stop_seen = record.get("semantic_stop_seen")
    first_stop_index = record.get("semantic_first_stop_token_index")
    stop_token_id = record.get("semantic_stop_token_id")
    stop_token_ids = record.get("semantic_stop_token_ids")
    semantic_token_count = record.get("semantic_token_count")
    if (
        not isinstance(token_ids, list)
        or any(type(token_id) is not int for token_id in token_ids)
        or not isinstance(semantic_text, str)
        or type(stop_seen) is not bool
        or type(first_stop_index) is not int
        or type(stop_token_id) is not int
        or not isinstance(stop_token_ids, list)
        or not stop_token_ids
        or any(type(token_id) is not int or token_id < 0 for token_id in stop_token_ids)
        or stop_token_ids != sorted(set(stop_token_ids))
        or type(semantic_token_count) is not int
    ):
        return full_text, "semantic_stop_proof_invalid"

    stop_token_set = set(stop_token_ids)
    if stop_seen:
        if (
            first_stop_index < 0
            or first_stop_index >= len(token_ids)
            or stop_token_id not in stop_token_set
            or token_ids[first_stop_index] != stop_token_id
            or any(
                token_id in stop_token_set
                for token_id in token_ids[:first_stop_index]
            )
            or semantic_token_count != first_stop_index + 1
            or not semantic_text.strip()
            or not full_text.startswith(semantic_text)
        ):
            return full_text, "semantic_stop_proof_invalid"
        return semantic_text, ""

    if (
        first_stop_index != -1
        or stop_token_id != -1
        or semantic_token_count != len(token_ids)
        or any(token_id in stop_token_set for token_id in token_ids)
        or semantic_text != full_text
    ):
        return full_text, "semantic_stop_proof_invalid"
    return full_text, ""


def _generated_text_from_output_record(record: dict[str, Any]) -> str:
    semantic_text, proof_reason = _semantic_output_text_and_proof_reason(record)
    return (
        semantic_text
        if not proof_reason
        else _full_generated_text_from_output_record(record)
    )


def _text_quality_stats(text: str) -> dict[str, float]:
    text = str(text or "")
    words = re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", text)
    non_ws_chars = sum(1 for ch in text if not ch.isspace())
    lexical_chars = sum(1 for ch in text if ch.isalnum())
    return {
        "chars": float(len(text)),
        "words": float(len(words)),
        "code_fences": float(text.count("```")),
        "lexical_ratio": (
            float(lexical_chars) / float(non_ws_chars)
            if non_ws_chars > 0
            else 0.0
        ),
    }


def _text_has_toxic_marker(text: str) -> bool:
    markers = (
        "fuck",
        "fucking",
        "shit",
        "bullshit",
        "bitch",
        "asshole",
        "cunt",
        "slut",
        "whore",
        "idiot",
        "moron",
        "stupid",
    )
    lower_text = str(text or "").lower()
    for marker in markers:
        pattern = r"(?<![a-z0-9_])" + re.escape(marker) + r"(?![a-z0-9_])"
        if re.search(pattern, lower_text):
            return True
    return False


def _reference_output_quality_reasons(
    sparse_records: list[dict[str, Any]],
    dense_records: list[dict[str, Any]],
) -> list[str]:
    if len(sparse_records) != len(dense_records):
        return ["reference_output_row_count_mismatch"]

    reasons: list[str] = []
    sparse_health = _sparse_output_content_health(sparse_records)
    dense_health = _sparse_output_content_health(dense_records)
    if sparse_health != "ok":
        reasons.append(f"reference_sparse_output_health_not_ok:{sparse_health}")
    if dense_health != "ok":
        reasons.append(f"reference_dense_output_health_not_ok:{dense_health}")
    for index, (sparse_record, dense_record) in enumerate(
        zip(sparse_records, dense_records)
    ):
        sparse_text = _generated_text_from_output_record(sparse_record)
        dense_text = _generated_text_from_output_record(dense_record)
        sparse_stats = _text_quality_stats(sparse_text)
        dense_stats = _text_quality_stats(dense_text)
        sparse_words = int(sparse_stats["words"])
        dense_words = int(dense_stats["words"])
        sparse_fences = int(sparse_stats["code_fences"])
        dense_fences = int(dense_stats["code_fences"])

        if dense_text and not sparse_text:
            reasons.append(f"reference_sparse_text_missing:{index}")
        if dense_words >= 24 and sparse_words < max(8, int(dense_words * 0.25)):
            reasons.append(f"reference_sparse_text_collapse:{index}")
        if (
            dense_words >= 24
            and sparse_fences >= 10
            and sparse_fences > dense_fences + 4
            and sparse_words < max(12, int(dense_words * 0.35))
        ):
            reasons.append(f"reference_sparse_format_loop:{index}")
        if (
            dense_stats["lexical_ratio"] >= 0.35
            and sparse_stats["chars"] >= 80.0
            and sparse_stats["lexical_ratio"] < 0.20
        ):
            reasons.append(f"reference_sparse_low_lexical_ratio:{index}")
        if _text_has_toxic_marker(sparse_text) and not _text_has_toxic_marker(
            dense_text
        ):
            reasons.append(f"reference_sparse_toxic_marker:{index}")
    return reasons


def _dense_fa3_route_proof(
    events: list[dict[str, Any]],
    *,
    env: dict[str, str],
    fa3_so_sha256: str,
) -> dict[str, Any]:
    symbol_route_counts: dict[str, int] = {}
    symbol_modules: dict[str, int] = {}
    fa_versions_seen: set[int] = set()
    symbol_event_count = 0
    for event in events:
        if event.get("event") != "flash_attn_varlen_func_call":
            continue
        symbol_event_count += 1
        route = str(event.get("route", "unknown"))
        symbol_route_counts[route] = symbol_route_counts.get(route, 0) + 1
        module = str(event.get("symbol_module", "unknown"))
        symbol_modules[module] = symbol_modules.get(module, 0) + 1
        fa_version = event.get("fa_version")
        if isinstance(fa_version, int):
            fa_versions_seen.add(int(fa_version))
    reasons: list[str] = []
    expected_version_raw = str(env.get("VLLM_FLASH_ATTN_VERSION", "3") or "3")
    try:
        expected_version = int(expected_version_raw)
    except ValueError:
        expected_version = 3
    backend_label = "fa4_sm100" if expected_version == 4 else "fa3"
    if env.get("VLLM_ATTENTION_BACKEND") != "FLASH_ATTN_VLLM_V1":
        reasons.append("attn_backend_not_flash_attn_vllm_v1")
    if env.get("VLLM_FLASH_ATTN_VERSION") != str(expected_version):
        reasons.append(f"flash_attn_version_not_{expected_version}")
    if not fa3_so_sha256:
        reasons.append(f"{backend_label}_artifact_sha256_missing")
    if symbol_event_count <= 0:
        reasons.append("flash_attn_varlen_func_call_missing")
    if int(symbol_route_counts.get("flash_attn_varlen_func", 0)) <= 0:
        reasons.append(f"native_{backend_label}_symbol_missing")
    if expected_version not in fa_versions_seen:
        reasons.append(f"fa_version_{expected_version}_missing")
    return {
        "passed": not reasons,
        "reasons": reasons,
        "symbol_event_count": int(symbol_event_count),
        "symbol_route_counts": symbol_route_counts,
        "symbol_modules": symbol_modules,
        "fa_versions_seen": sorted(fa_versions_seen),
        "requested_attn_backend": env.get("VLLM_ATTENTION_BACKEND", ""),
        "requested_flash_attn_version": env.get("VLLM_FLASH_ATTN_VERSION", ""),
        "fa3_so_sha256": fa3_so_sha256,
        "backend": backend_label,
        "backend_artifact_sha256": fa3_so_sha256,
    }


def _gate_d_config(
    args: argparse.Namespace,
    *,
    mode: str,
    env: dict[str, str],
    fa3_so_sha256: str,
) -> dict[str, Any]:
    return {
        "mode": mode,
        "preset": str(getattr(args, "preset", "") or ""),
        "model": str(args.model),
        "prompt": str(args.prompt),
        "output_len": _effective_max_new_tokens(args),
        "max_new_tokens": _effective_max_new_tokens(args),
        "max_new_tokens_effective": _effective_max_new_tokens(args),
        "iters": int(args.iters),
        "legacy_iters": int(getattr(args, "legacy_iters", args.iters)),
        "iters_semantics": "legacy_alias_for_max_new_tokens",
        "batch_size": int(args.batch_size),
        "request_context_tokens": _request_context_tokens(args),
        "request_max_new_tokens": _request_max_new_tokens(args),
        "max_num_seqs": _effective_max_num_seqs(args),
        "scheduler_graph_contract_requested": requested_scheduler_graph_contract(
            args
        ),
        "scheduling_mode_requested": str(
            getattr(args, "scheduling_mode", "auto") or "auto"
        ),
        "warmup": int(args.warmup),
        "full_cuda_graph": bool(args.full_cuda_graph),
        "backend": _gate_d_backend(args),
        "flash_attn_version_expected": int(_gate_d_flash_attn_version(args)),
        "cuda_visible_devices": str(args.cuda_visible_devices),
        "fa3_so_sha256": fa3_so_sha256,
        "controller_json": env.get("VLLM_SPARSE_CONTROLLER_JSON", ""),
        "env": {
            "VLLM_ATTENTION_BACKEND": env.get("VLLM_ATTENTION_BACKEND", ""),
            "VLLM_FLASH_ATTN_VERSION": env.get("VLLM_FLASH_ATTN_VERSION", ""),
            "VLLM_SPARSE_FA3_UPSTREAM_ROOT": env.get(
                "VLLM_SPARSE_FA3_UPSTREAM_ROOT",
                "",
            ),
            "VLLM_SPARSE_ONE_SHOT_GROUP_READY": env.get(
                "VLLM_SPARSE_ONE_SHOT_GROUP_READY",
                "",
            ),
            "VLLM_SPARSE_ONE_SHOT_READY_CHUNK": env.get(
                "VLLM_SPARSE_ONE_SHOT_READY_CHUNK",
                "",
            ),
            "SFI_RUNNER_REQUEST_CONTEXT_TOKENS": env.get(
                "SFI_RUNNER_REQUEST_CONTEXT_TOKENS", ""
            ),
            "SFI_RUNNER_REQUEST_MAX_NEW_TOKENS": env.get(
                "SFI_RUNNER_REQUEST_MAX_NEW_TOKENS", ""
            ),
            **{
                key: str(env.get(key, "") or "")
                for key in SPEED_CHILD_PAIRING_IDENTITY_ENV_KEYS
            },
            **{
                key: str(env.get(key, "") or "")
                for key in ALLOWED_TRACE_ENV_KEYS
            },
            **_gate_d_trace_profile_env(env),
        },
    }


def _refresh_reason_counts_from_route_summary(
    route_summary: dict[str, Any] | dict[str, object] | None,
) -> dict[str, int]:
    raw = (route_summary or {}).get("refresh_reason_counts", {})
    if not isinstance(raw, dict):
        return {}
    counts: dict[str, int] = {}
    for key, value in raw.items():
        count = _as_int(value, 0)
        if count > 0:
            counts[str(key)] = count
    return counts


def _continuous_refresh_payloads_from_sources(
    refresh_profile: list[dict[str, Any]],
    producer_route_summary: dict[str, Any] | None,
) -> int:
    """Return measurement-window refresh payload evidence.

    The refresh profile is intentionally fail-closed when markers are missing,
    and some producer paths report complete measurement-window payload counts
    only through the route/hook enqueue trace.  Treat both as equivalent
    producer evidence and use the larger count so a partial profile does not
    under-report real full-open refresh work.
    """
    profile_payloads = _sum_int(refresh_profile, "refresh_payloads")
    route_payloads = _as_int(
        (producer_route_summary or {}).get(
            "replay_refresh_payload_enqueue_payloads_total",
            0,
        ),
        0,
    )
    return max(int(profile_payloads), int(route_payloads))


def _async_producer_writer_count_from_sources(
    refresh_profile: list[dict[str, Any]],
    hook_profile_summary: dict[str, Any] | None,
    producer_route_summary: dict[str, Any] | None = None,
) -> tuple[int, str]:
    """Return writer-completion evidence without treating enqueue as writer work."""
    profile_writer_count = _max_int_from_records(
        refresh_profile,
        "deadline_async_producer_writer_count",
    )
    hook_summary = hook_profile_summary or {}
    hook_deferred_count = _as_int(
        hook_summary.get("refresh_stage_deferred_pending_rebuild_count_total"),
        0,
    )
    hook_drained_count = _as_int(
        hook_summary.get("pre_consume_pending_rebuild_drained_total"),
        0,
    )
    hook_writer_count = hook_drained_count if hook_deferred_count > 0 else 0
    route_writer_count = _as_int(
        (producer_route_summary or {}).get(
            "async_producer_writer_complete_count",
            0,
        ),
        0,
    )
    if profile_writer_count > 0:
        return int(profile_writer_count), "refresh_profile"
    if route_writer_count > 0:
        return int(route_writer_count), "route_writer_complete"
    if hook_writer_count > 0:
        return int(hook_writer_count), "full_cudagraph_hook_pre_consume_drain"
    return 0, "missing"


def _expected_interval_trigger_intents(
    args: argparse.Namespace,
    *,
    producer_mode: str,
) -> int:
    if not _producer_mode_requires_refresh(producer_mode):
        return 0
    if bool(getattr(args, "respect_eos", False)):
        return 0
    refresh_interval = int(getattr(args, "refresh_interval", 0) or 0)
    if refresh_interval <= 0:
        return 0
    return sum(
        max(0, int(max_new_tokens)) // int(refresh_interval)
        for max_new_tokens in _request_max_new_tokens(args)
    )


def _interval_trigger_requirement_satisfied(
    *,
    expected_interval_trigger_intents: int,
    refresh_trigger_intents: int,
) -> bool:
    # [INTERVAL-GATE-DYNAMIC 2026-07-07 口径变更] interval 是兜底节拍：任何
    # reason 的世代完成都会推进 last_decode_refresh（selector_compute 世代
    # 终局归零 per-req 计时，跨 reason 成立——用户设计合同），sentence 密集时
    # interval intent 趋零是健康形态而非缺陷。判据改判"触发线活着"：全 reason
    # 意图总数 ≥ bs×(max_new//interval) 节拍下界。refresh-on（无 sentence）下
    # 总数≈interval 数，与旧主判据等价；trigger/full-open 下取代旧 fallback
    # （旧逻辑最终也落到同一比较）。旧静态主判据（interval 单项≥期望）在
    # sentence 重置语义下系统性误报红（4B 实测），故退休。
    if int(expected_interval_trigger_intents) <= 0:
        return True
    return int(refresh_trigger_intents) >= int(expected_interval_trigger_intents)


def _boundary_diagnostics_payload(metrics: dict[str, Any]) -> dict[str, Any]:
    boundary_diagnostics = metrics.get("boundary_diagnostics")
    return {
        "boundary_diagnostics": (
            dict(boundary_diagnostics)
            if isinstance(boundary_diagnostics, dict)
            else {}
        )
    }


def _request_ordered_output_records(path: Path) -> list[dict[str, Any]]:
    """Return outputs only when request keys prove one contiguous row binding."""
    payload = _read_json(path)
    if not isinstance(payload, dict):
        return []
    indexed_keys: list[tuple[int, str]] = []
    for key in payload:
        match = re.fullmatch(r"bench-([0-9]+)-.+", str(key))
        if match is None:
            return []
        indexed_keys.append((int(match.group(1)), str(key)))
    indices = [index for index, _ in indexed_keys]
    if sorted(indices) != list(range(len(indexed_keys))):
        return []
    rows: list[dict[str, Any]] = []
    for request_index, key in sorted(indexed_keys):
        value = payload[key]
        if isinstance(value, list):
            rows.append(
                {
                    "request_index": request_index,
                    "token_ids": [int(token_id) for token_id in value],
                }
            )
            continue
        if not isinstance(value, dict) or not isinstance(
            value.get("token_ids"), list
        ):
            continue
        row: dict[str, Any] = {
            "request_index": request_index,
            "token_ids": [int(token_id) for token_id in value["token_ids"]],
            "text": str(value.get("text", "")),
        }
        for field_name in (
            "semantic_text",
            "semantic_stop_seen",
            "semantic_first_stop_token_index",
            "semantic_stop_token_id",
            "semantic_stop_token_ids",
            "semantic_token_count",
        ):
            if field_name in value:
                row[field_name] = value[field_name]
        rows.append(row)
    return rows


def _output_completion_gate(
    records: list[dict[str, Any]],
    *,
    expected_request_count: int,
    expected_output_tokens: int | None = None,
    expected_output_tokens_by_request: list[int] | None = None,
    require_exact_length: bool = True,
) -> dict[str, Any]:
    """Validate request cardinality and fixed-length decode completion."""
    expected_vector = (
        [int(value) for value in expected_output_tokens_by_request]
        if expected_output_tokens_by_request is not None
        else [int(expected_output_tokens or 0)] * int(expected_request_count)
    )
    expected_maximum = max(expected_vector, default=int(expected_output_tokens or 0))
    token_lengths = [
        len(record.get("token_ids", []))
        if isinstance(record.get("token_ids"), list)
        else -1
        for record in records
    ]
    request_indices = [record.get("request_index") for record in records]
    reasons: list[str] = []
    if len(records) != int(expected_request_count):
        reasons.append(
            "output_record_count_mismatch:"
            f"actual={len(records)}:expected={int(expected_request_count)}"
        )
    expected_indices = list(range(int(expected_request_count)))
    if request_indices != expected_indices:
        reasons.append(
            "output_request_index_binding_mismatch:"
            f"actual={request_indices!r}:expected={expected_indices!r}"
        )
    if len(expected_vector) != int(expected_request_count):
        reasons.append(
            "expected_output_vector_count_mismatch:"
            f"actual={len(expected_vector)}:expected={int(expected_request_count)}"
        )
    if require_exact_length and token_lengths != expected_vector:
        mismatched_indices = [
            index
            for index, (actual, expected) in enumerate(
                zip(token_lengths, expected_vector)
            )
            if actual != expected
        ]
        reasons.append(
            "output_token_count_mismatch:"
            f"min={min(token_lengths, default=-1)}:"
            f"max={max(token_lengths, default=-1)}:"
            f"expected_by_request={_request_vector_csv(expected_vector)}:"
            f"mismatched_indices={mismatched_indices}"
        )
    return {
        "required": True,
        "exact_length_required": bool(require_exact_length),
        "passed": not reasons,
        "reasons": reasons,
        "actual_request_count": len(records),
        "expected_request_count": int(expected_request_count),
        "request_indices": request_indices,
        "token_lengths": token_lengths,
        "expected_output_tokens_by_request": expected_vector,
        "min_output_tokens": min(token_lengths, default=-1),
        "max_output_tokens": max(token_lengths, default=-1),
        "expected_output_tokens": int(expected_maximum),
    }


def _mixed_chunk_postflight_gate(
    args: argparse.Namespace,
    trace_events: list[dict[str, Any]],
) -> dict[str, Any]:
    """Prove mixed chunk-prefill/decode only from the cold route artifact."""
    batch_size = int(args.batch_size)
    context_tokens = _request_context_tokens(args)
    max_new_tokens = _request_max_new_tokens(args)
    context_heterogeneous = bool(
        len(context_tokens) == batch_size and len(set(context_tokens)) > 1
    )
    max_new_heterogeneous = len(set(max_new_tokens)) > 1
    if batch_size == 1:
        return {
            "schema": "sfi.mixed_chunk_postflight.v1",
            "required": False,
            "status": "not_applicable",
            "passed": True,
            "reasons": [],
            "batch_size": batch_size,
            "request_context_tokens": context_tokens,
            "request_max_new_tokens": max_new_tokens,
            "context_heterogeneous": False,
            "max_new_tokens_heterogeneous": False,
            "prepare_step_lengths_count": 0,
            "malformed_prepare_step_lengths_count": 0,
            "mixed_chunk_event_count": 0,
            "mixed_chunk_event_sample": {},
        }
    if not context_heterogeneous:
        return {
            "schema": "sfi.mixed_chunk_postflight.v1",
            "required": False,
            "status": "not_required",
            "passed": True,
            "reasons": [],
            "batch_size": batch_size,
            "request_context_tokens": context_tokens,
            "request_max_new_tokens": max_new_tokens,
            "context_heterogeneous": False,
            "max_new_tokens_heterogeneous": max_new_heterogeneous,
            "prepare_step_lengths_count": 0,
            "malformed_prepare_step_lengths_count": 0,
            "mixed_chunk_event_count": 0,
            "mixed_chunk_event_sample": {},
        }

    prepare_count = 0
    malformed_count = 0
    mixed_samples: list[dict[str, Any]] = []
    for event in trace_events:
        if event.get("event") != "prepare_step_lengths":
            continue
        prepare_count += 1
        req_ids = event.get("req_ids")
        prompt_lengths = event.get("prompt_lengths")
        computed_tokens = event.get("num_computed_tokens")
        scheduled_tokens = event.get("num_scheduled_tokens")
        if not all(
            isinstance(values, list)
            for values in (
                req_ids,
                prompt_lengths,
                computed_tokens,
                scheduled_tokens,
            )
        ):
            malformed_count += 1
            continue
        row_count = len(req_ids)
        if row_count == 0 or any(
            len(values) != row_count
            for values in (
                prompt_lengths,
                computed_tokens,
                scheduled_tokens,
            )
        ):
            malformed_count += 1
            continue
        if (
            any(type(value) is not str or not value for value in req_ids)
            or any(
                type(value) is not int or value < 0
                for values in (
                    prompt_lengths,
                    computed_tokens,
                    scheduled_tokens,
                )
                for value in values
            )
            or any(value <= 0 for value in prompt_lengths)
        ):
            malformed_count += 1
            continue
        # A heterogeneous scheduler legitimately has one active request before
        # another row is admitted, and again after a short row finishes.  Such
        # a record is well formed but cannot prove a simultaneous mixed step.
        if row_count < 2:
            continue
        prefill_indices = [
            index
            for index, (prompt, computed, scheduled) in enumerate(
                zip(prompt_lengths, computed_tokens, scheduled_tokens)
            )
            if computed < prompt and scheduled > 1
        ]
        decode_indices = [
            index
            for index, (prompt, computed, scheduled) in enumerate(
                zip(prompt_lengths, computed_tokens, scheduled_tokens)
            )
            if computed >= prompt and scheduled == 1
        ]
        if prefill_indices and decode_indices:
            mixed_samples.append(
                {
                    "epoch": _as_int(event.get("epoch"), -1),
                    "req_ids": list(req_ids),
                    "prompt_lengths": list(prompt_lengths),
                    "num_computed_tokens": list(computed_tokens),
                    "num_scheduled_tokens": list(scheduled_tokens),
                    "prefill_indices": prefill_indices,
                    "decode_indices": decode_indices,
                }
            )
    reasons: list[str] = []
    if malformed_count:
        reasons.append("mixed_chunk_route_evidence_malformed")
    if not mixed_samples:
        reasons.append("mixed_chunk_route_evidence_missing")
    return {
        "schema": "sfi.mixed_chunk_postflight.v1",
        "required": True,
        "status": "passed" if not reasons else "failed",
        "passed": not reasons,
        "reasons": reasons,
        "batch_size": batch_size,
        "request_context_tokens": context_tokens,
        "request_max_new_tokens": max_new_tokens,
        "context_heterogeneous": True,
        "max_new_tokens_heterogeneous": max_new_heterogeneous,
        "prepare_step_lengths_count": prepare_count,
        "malformed_prepare_step_lengths_count": malformed_count,
        "mixed_chunk_event_count": len(mixed_samples),
        "mixed_chunk_event_sample": mixed_samples[0] if mixed_samples else {},
    }


def _speed_child_custom_all_reduce_provenance(
    metrics: dict[str, Any],
    *,
    expected_batch_size: int | None = None,
) -> dict[str, Any]:
    """Extract the effective child decision and reject contradictory copies."""
    run_config = metrics.get("run_config")
    nested = run_config if isinstance(run_config, dict) else {}
    policy_field_names = (
        "custom_all_reduce_requested",
        "custom_all_reduce_effective",
        "custom_all_reduce_effective_reason",
    )
    runtime_field_names = (
        "custom_all_reduce_runtime_proof_required",
        "custom_all_reduce_runtime_proof_passed",
        "custom_all_reduce_runtime_configured_effective",
        "custom_all_reduce_runtime_tensor_parallel_size",
        "custom_all_reduce_runtime_rank_count",
        "custom_all_reduce_runtime_active_rank_count",
        "custom_all_reduce_runtime_inactive_rank_count",
        "custom_all_reduce_runtime_all_ranks_active",
        "custom_all_reduce_runtime_rank_consistent",
        "custom_all_reduce_runtime_preemptor_flashinfer_enabled",
        "custom_all_reduce_runtime_preemptor_torch_symm_mem_enabled",
        "custom_all_reduce_runtime_preemptor_nccl_symm_mem_enabled",
        "custom_all_reduce_runtime_required_num_tokens",
        "custom_all_reduce_runtime_model_hidden_size",
        "custom_all_reduce_runtime_model_dtype",
        "custom_all_reduce_runtime_model_dtype_bytes",
        "custom_all_reduce_runtime_required_payload_bytes",
        "custom_all_reduce_runtime_all_ranks_payload_eligible",
        "custom_all_reduce_runtime_records",
    )
    values: dict[str, Any] = {}
    mismatches: list[str] = []
    missing: list[str] = []
    for field_name in (*policy_field_names, *runtime_field_names):
        top_present = field_name in metrics
        nested_present = field_name in nested
        top_value = metrics.get(field_name)
        nested_value = nested.get(field_name)
        if top_present and nested_present and top_value != nested_value:
            mismatches.append(field_name)
        if top_present:
            values[field_name] = top_value
        elif nested_present:
            values[field_name] = nested_value
        else:
            missing.append(field_name)
    policy_mismatches = [
        field_name
        for field_name in mismatches
        if field_name in policy_field_names
    ]
    if policy_mismatches:
        return {
            "custom_all_reduce_requested": "invalid",
            "custom_all_reduce_effective": "invalid",
            "custom_all_reduce_effective_reason": (
                "metrics_run_config_mismatch:" + ",".join(policy_mismatches)
            ),
            "custom_all_reduce_runtime_proof_required": False,
            "custom_all_reduce_runtime_proof_passed": False,
            "custom_all_reduce_runtime_configured_effective": "invalid",
            "custom_all_reduce_runtime_tensor_parallel_size": -1,
            "custom_all_reduce_runtime_rank_count": -1,
            "custom_all_reduce_runtime_active_rank_count": -1,
            "custom_all_reduce_runtime_inactive_rank_count": -1,
            "custom_all_reduce_runtime_all_ranks_active": False,
            "custom_all_reduce_runtime_rank_consistent": False,
            "custom_all_reduce_runtime_preemptor_flashinfer_enabled": False,
            "custom_all_reduce_runtime_preemptor_torch_symm_mem_enabled": False,
            "custom_all_reduce_runtime_preemptor_nccl_symm_mem_enabled": False,
            "custom_all_reduce_runtime_required_num_tokens": -1,
            "custom_all_reduce_runtime_model_hidden_size": -1,
            "custom_all_reduce_runtime_model_dtype": "",
            "custom_all_reduce_runtime_model_dtype_bytes": -1,
            "custom_all_reduce_runtime_required_payload_bytes": -1,
            "custom_all_reduce_runtime_all_ranks_payload_eligible": False,
            "custom_all_reduce_runtime_records": [],
            "custom_all_reduce_runtime_proof_error": (
                "metrics_run_config_mismatch:" + ",".join(mismatches)
            ),
        }
    requested = str(values.get("custom_all_reduce_requested", "") or "")
    effective = str(values.get("custom_all_reduce_effective", "") or "")
    effective_reason = str(
        values.get("custom_all_reduce_effective_reason", "") or ""
    )
    if effective not in {"enabled", "disabled", "not_applicable"}:
        requested = requested or "unknown"
        effective = "unknown"
        effective_reason = (
            effective_reason or "child_artifact_missing_effective_decision"
        )

    runtime_errors = [
        f"metrics_run_config_mismatch:{field_name}"
        for field_name in mismatches
        if field_name in runtime_field_names
    ]
    runtime_errors.extend(
        f"missing:{field_name}"
        for field_name in missing
        if field_name in runtime_field_names
    )

    def _strict_bool(field_name: str) -> bool:
        value = values.get(field_name)
        if not isinstance(value, bool):
            runtime_errors.append(f"invalid_bool:{field_name}={value!r}")
            return False
        return value

    def _strict_int(field_name: str) -> int:
        value = values.get(field_name)
        if isinstance(value, bool) or not isinstance(value, int):
            runtime_errors.append(f"invalid_int:{field_name}={value!r}")
            return -1
        return int(value)

    def _strict_str(field_name: str) -> str:
        value = values.get(field_name)
        if not isinstance(value, str) or not value:
            runtime_errors.append(f"invalid_str:{field_name}={value!r}")
            return ""
        return value

    runtime_required = _strict_bool(
        "custom_all_reduce_runtime_proof_required"
    )
    runtime_passed_raw = _strict_bool(
        "custom_all_reduce_runtime_proof_passed"
    )
    runtime_configured_effective = str(
        values.get("custom_all_reduce_runtime_configured_effective", "") or ""
    )
    runtime_tp_size = _strict_int(
        "custom_all_reduce_runtime_tensor_parallel_size"
    )
    runtime_rank_count = _strict_int(
        "custom_all_reduce_runtime_rank_count"
    )
    runtime_active_count = _strict_int(
        "custom_all_reduce_runtime_active_rank_count"
    )
    runtime_inactive_count = _strict_int(
        "custom_all_reduce_runtime_inactive_rank_count"
    )
    runtime_all_active = _strict_bool(
        "custom_all_reduce_runtime_all_ranks_active"
    )
    runtime_rank_consistent = _strict_bool(
        "custom_all_reduce_runtime_rank_consistent"
    )
    runtime_flashinfer_preemptor = _strict_bool(
        "custom_all_reduce_runtime_preemptor_flashinfer_enabled"
    )
    runtime_torch_symm_mem_preemptor = _strict_bool(
        "custom_all_reduce_runtime_preemptor_torch_symm_mem_enabled"
    )
    runtime_nccl_symm_mem_preemptor = _strict_bool(
        "custom_all_reduce_runtime_preemptor_nccl_symm_mem_enabled"
    )
    runtime_required_num_tokens = _strict_int(
        "custom_all_reduce_runtime_required_num_tokens"
    )
    runtime_model_hidden_size = _strict_int(
        "custom_all_reduce_runtime_model_hidden_size"
    )
    runtime_model_dtype = _strict_str(
        "custom_all_reduce_runtime_model_dtype"
    )
    runtime_model_dtype_bytes = _strict_int(
        "custom_all_reduce_runtime_model_dtype_bytes"
    )
    runtime_required_payload_bytes = _strict_int(
        "custom_all_reduce_runtime_required_payload_bytes"
    )
    runtime_all_payload_eligible = _strict_bool(
        "custom_all_reduce_runtime_all_ranks_payload_eligible"
    )
    raw_records = values.get("custom_all_reduce_runtime_records")
    runtime_records = raw_records if isinstance(raw_records, list) else []
    if not isinstance(raw_records, list):
        runtime_errors.append(
            "invalid_list:custom_all_reduce_runtime_records="
            f"{type(raw_records).__name__}"
        )

    if runtime_configured_effective != effective:
        runtime_errors.append(
            "configured_effective_mismatch:"
            f"runtime={runtime_configured_effective!r}:policy={effective!r}"
        )
    if runtime_tp_size <= 0:
        runtime_errors.append(f"invalid_tp_size:{runtime_tp_size}")
    if runtime_required is not (runtime_tp_size > 1):
        runtime_errors.append(
            f"proof_required={runtime_required!r}:tp_size={runtime_tp_size}"
        )
    if runtime_rank_count != runtime_tp_size:
        runtime_errors.append(
            f"rank_count={runtime_rank_count}:tp_size={runtime_tp_size}"
        )
    if len(runtime_records) != runtime_tp_size:
        runtime_errors.append(
            f"record_count={len(runtime_records)}:tp_size={runtime_tp_size}"
        )
    if runtime_active_count + runtime_inactive_count != runtime_tp_size:
        runtime_errors.append(
            "active_inactive_count_mismatch:"
            f"active={runtime_active_count}:inactive={runtime_inactive_count}:"
            f"tp_size={runtime_tp_size}"
        )
    top_batch_size = metrics.get("batch_size")
    child_batch_size = nested.get("batch_size")
    if (
        isinstance(top_batch_size, bool)
        or not isinstance(top_batch_size, int)
        or top_batch_size <= 0
    ):
        runtime_errors.append(
            f"invalid_metrics_batch_size:{top_batch_size!r}"
        )
    if (
        isinstance(child_batch_size, bool)
        or not isinstance(child_batch_size, int)
        or child_batch_size <= 0
    ):
        runtime_errors.append(f"invalid_run_config_batch_size:{child_batch_size!r}")
    elif top_batch_size != child_batch_size:
        runtime_errors.append(
            "metrics_run_config_batch_size_mismatch:"
            f"metrics={top_batch_size!r}:run_config={child_batch_size}"
        )
    if (
        isinstance(child_batch_size, int)
        and not isinstance(child_batch_size, bool)
        and runtime_required_num_tokens != child_batch_size
    ):
        runtime_errors.append(
            "required_num_tokens_batch_size_mismatch:"
            f"required={runtime_required_num_tokens}:batch_size={child_batch_size}"
        )
    if (
        expected_batch_size is not None
        and runtime_required_num_tokens != expected_batch_size
    ):
        runtime_errors.append(
            "required_num_tokens_parent_batch_size_mismatch:"
            f"required={runtime_required_num_tokens}:"
            f"parent_batch_size={expected_batch_size}"
        )

    dtype_bytes_by_name = {"bfloat16": 2, "float16": 2}
    child_dtype = nested.get("dtype")
    if not isinstance(child_dtype, str) or child_dtype not in dtype_bytes_by_name:
        runtime_errors.append(f"invalid_run_config_dtype:{child_dtype!r}")
        expected_runtime_dtype = ""
        expected_dtype_bytes = -1
    else:
        expected_runtime_dtype = f"torch.{child_dtype}"
        expected_dtype_bytes = dtype_bytes_by_name[child_dtype]
    if runtime_model_dtype != expected_runtime_dtype:
        runtime_errors.append(
            "model_dtype_run_config_mismatch:"
            f"runtime={runtime_model_dtype!r}:run_config={child_dtype!r}"
        )
    if runtime_model_dtype_bytes != expected_dtype_bytes:
        runtime_errors.append(
            "model_dtype_bytes_run_config_mismatch:"
            f"runtime={runtime_model_dtype_bytes}:expected={expected_dtype_bytes}"
        )
    expected_payload_bytes = (
        runtime_required_num_tokens
        * runtime_model_hidden_size
        * runtime_model_dtype_bytes
    )
    if (
        runtime_required_num_tokens <= 0
        or runtime_model_hidden_size <= 0
        or runtime_model_dtype_bytes <= 0
        or runtime_required_payload_bytes != expected_payload_bytes
    ):
        runtime_errors.append(
            "required_payload_spec_mismatch:"
            f"tokens={runtime_required_num_tokens}:"
            f"hidden={runtime_model_hidden_size}:"
            f"dtype={runtime_model_dtype!r}:"
            f"dtype_bytes={runtime_model_dtype_bytes}:"
            f"payload_bytes={runtime_required_payload_bytes}:"
            f"expected={expected_payload_bytes}"
        )

    record_ranks: list[int] = []
    record_active_count = 0
    record_flashinfer_preemptor = False
    record_torch_symm_mem_preemptor = False
    record_nccl_symm_mem_preemptor = False
    for index, record in enumerate(runtime_records):
        if not isinstance(record, dict):
            runtime_errors.append(
                f"record[{index}]={type(record).__name__}:expected=dict"
            )
            continue
        rank = record.get("tp_rank")
        world_size = record.get("tp_world_size")
        active = record.get("ca_comm_active")
        ca_present = record.get("ca_comm_present")
        ca_disabled = record.get("ca_comm_disabled")
        ca_fully_connected = record.get("ca_comm_fully_connected")
        ca_max_size = record.get("ca_comm_max_size")
        record_required_num_tokens = record.get("required_num_tokens")
        record_model_hidden_size = record.get("model_hidden_size")
        record_model_dtype = record.get("model_dtype")
        record_model_dtype_bytes = record.get("model_dtype_bytes")
        record_payload_numel = record.get("required_payload_numel")
        record_payload_bytes = record.get("required_payload_bytes")
        record_payload_spec_error = record.get("required_payload_spec_error")
        ca_capacity_eligible = record.get(
            "ca_comm_capacity_covers_required_payload"
        )
        ca_dispatch_size_eligible = record.get(
            "ca_comm_dispatch_size_eligible"
        )
        ca_should_custom_ar_callable = record.get(
            "ca_comm_should_custom_ar_callable"
        )
        ca_synthetic_eligible = record.get(
            "ca_comm_synthetic_should_custom_ar"
        )
        ca_synthetic_error = record.get(
            "ca_comm_synthetic_should_custom_ar_error"
        )
        tri_state_fields = (
            "ca_comm_capacity_covers_required_payload",
            "ca_comm_dispatch_size_eligible",
            "ca_comm_synthetic_should_custom_ar",
        )
        for field_name in tri_state_fields:
            if field_name not in record:
                runtime_errors.append(f"record[{index}] missing={field_name}")
        config_disabled = record.get(
            "parallel_config_disable_custom_all_reduce"
        )
        communicator_enabled = record.get(
            "device_communicator_use_custom_allreduce"
        )
        flashinfer_configured = record.get(
            "device_communicator_use_flashinfer_allreduce"
        )
        flashinfer_active = record.get("fi_ar_comm_active")
        flashinfer_env_enabled = record.get("vllm_allreduce_use_flashinfer")
        torch_symm_mem_configured = record.get(
            "device_communicator_use_torch_symm_mem"
        )
        torch_symm_mem_active = record.get("symm_mem_comm_active")
        torch_symm_mem_env_enabled = record.get(
            "vllm_allreduce_use_symm_mem"
        )
        nccl_symm_mem_enabled = record.get("vllm_use_nccl_symm_mem")
        if isinstance(rank, bool) or not isinstance(rank, int):
            runtime_errors.append(f"record[{index}].tp_rank={rank!r}")
            continue
        record_ranks.append(int(rank))
        if world_size != runtime_tp_size:
            runtime_errors.append(
                f"rank{rank}.tp_world_size={world_size!r}:"
                f"expected={runtime_tp_size}"
            )
        if not isinstance(active, bool):
            runtime_errors.append(f"rank{rank}.ca_comm_active={active!r}")
            continue
        if not all(
            isinstance(value, bool)
            for value in (
                flashinfer_configured,
                flashinfer_active,
                flashinfer_env_enabled,
            )
        ):
            runtime_errors.append(
                f"rank{rank}.flashinfer_state="
                f"{flashinfer_configured!r},{flashinfer_active!r},"
                f"{flashinfer_env_enabled!r}"
            )
        else:
            record_flashinfer_preemptor = bool(
                record_flashinfer_preemptor
                or flashinfer_configured
                or flashinfer_active
                or flashinfer_env_enabled
            )
        if not all(
            isinstance(value, bool)
            for value in (
                torch_symm_mem_configured,
                torch_symm_mem_active,
                torch_symm_mem_env_enabled,
            )
        ):
            runtime_errors.append(
                f"rank{rank}.torch_symm_mem_state="
                f"{torch_symm_mem_configured!r},{torch_symm_mem_active!r},"
                f"{torch_symm_mem_env_enabled!r}"
            )
        else:
            record_torch_symm_mem_preemptor = bool(
                record_torch_symm_mem_preemptor
                or torch_symm_mem_configured
                or torch_symm_mem_active
                or torch_symm_mem_env_enabled
            )
        if not isinstance(nccl_symm_mem_enabled, bool):
            runtime_errors.append(
                f"rank{rank}.vllm_use_nccl_symm_mem="
                f"{nccl_symm_mem_enabled!r}"
            )
        else:
            record_nccl_symm_mem_preemptor = bool(
                record_nccl_symm_mem_preemptor or nccl_symm_mem_enabled
            )
        record_active_count += int(active)
        expected_record_numel = (
            runtime_required_num_tokens * runtime_model_hidden_size
        )
        record_spec = {
            "required_num_tokens": (
                record_required_num_tokens,
                runtime_required_num_tokens,
            ),
            "model_hidden_size": (
                record_model_hidden_size,
                runtime_model_hidden_size,
            ),
            "model_dtype": (record_model_dtype, runtime_model_dtype),
            "model_dtype_bytes": (
                record_model_dtype_bytes,
                runtime_model_dtype_bytes,
            ),
            "required_payload_numel": (
                record_payload_numel,
                expected_record_numel,
            ),
            "required_payload_bytes": (
                record_payload_bytes,
                runtime_required_payload_bytes,
            ),
        }
        for field_name, (actual, expected) in record_spec.items():
            if type(actual) is not type(expected) or actual != expected:
                runtime_errors.append(
                    f"rank{rank}.{field_name}={actual!r}:expected={expected!r}"
                )
        if record_payload_spec_error != "":
            runtime_errors.append(
                f"rank{rank}.required_payload_spec_error="
                f"{record_payload_spec_error!r}"
            )
        if not isinstance(ca_should_custom_ar_callable, bool):
            runtime_errors.append(
                f"rank{rank}.ca_comm_should_custom_ar_callable="
                f"{ca_should_custom_ar_callable!r}"
            )
        for field_name, value in (
            (
                "ca_comm_capacity_covers_required_payload",
                ca_capacity_eligible,
            ),
            ("ca_comm_dispatch_size_eligible", ca_dispatch_size_eligible),
            ("ca_comm_synthetic_should_custom_ar", ca_synthetic_eligible),
        ):
            if value is not None and not isinstance(value, bool):
                runtime_errors.append(
                    f"rank{rank}.{field_name}={value!r}:expected=bool|None"
                )
        if ca_synthetic_error != "":
            runtime_errors.append(
                f"rank{rank}.ca_comm_synthetic_should_custom_ar_error="
                f"{ca_synthetic_error!r}"
            )
        if active is False and ca_synthetic_eligible is not None:
            runtime_errors.append(
                f"rank{rank}.ca_comm_synthetic_should_custom_ar="
                f"{ca_synthetic_eligible!r}:expected=None_when_inactive"
            )
        if runtime_tp_size > 1 and effective in {"enabled", "disabled"}:
            expected_active = effective == "enabled"
            if active is not expected_active:
                runtime_errors.append(
                    f"rank{rank}.ca_comm_active={active!r}:"
                    f"expected={expected_active!r}"
                )
            if config_disabled is not (not expected_active):
                runtime_errors.append(
                    f"rank{rank}.config_disabled={config_disabled!r}:"
                    f"expected={not expected_active!r}"
                )
            if communicator_enabled is not expected_active:
                runtime_errors.append(
                    f"rank{rank}.communicator_enabled="
                    f"{communicator_enabled!r}:expected={expected_active!r}"
                )
            if expected_active:
                if ca_present is not True:
                    runtime_errors.append(
                        f"rank{rank}.ca_comm_present={ca_present!r}"
                    )
                if ca_disabled is not False:
                    runtime_errors.append(
                        f"rank{rank}.ca_comm_disabled={ca_disabled!r}"
                    )
                if not isinstance(ca_fully_connected, bool):
                    runtime_errors.append(
                        f"rank{rank}.ca_comm_fully_connected="
                        f"{ca_fully_connected!r}"
                    )
                elif runtime_tp_size > 2 and not ca_fully_connected:
                    runtime_errors.append(
                        f"rank{rank}.ca_comm_fully_connected=false"
                    )
                if (
                    isinstance(ca_max_size, bool)
                    or not isinstance(ca_max_size, int)
                    or ca_max_size <= 0
                ):
                    runtime_errors.append(
                        f"rank{rank}.ca_comm_max_size={ca_max_size!r}"
                    )
                elif ca_max_size <= runtime_required_payload_bytes:
                    runtime_errors.append(
                        f"rank{rank}.ca_comm_max_size={ca_max_size}:"
                        "must_exceed_required_payload_bytes="
                        f"{runtime_required_payload_bytes}"
                    )
                if ca_capacity_eligible is not True:
                    runtime_errors.append(
                        f"rank{rank}.ca_comm_capacity_covers_required_payload="
                        f"{ca_capacity_eligible!r}"
                    )
                if ca_dispatch_size_eligible is not True:
                    runtime_errors.append(
                        f"rank{rank}.ca_comm_dispatch_size_eligible="
                        f"{ca_dispatch_size_eligible!r}"
                    )
                if ca_should_custom_ar_callable is not True:
                    runtime_errors.append(
                        f"rank{rank}.ca_comm_should_custom_ar_callable="
                        f"{ca_should_custom_ar_callable!r}"
                    )
                if ca_synthetic_eligible is not True:
                    runtime_errors.append(
                        f"rank{rank}.ca_comm_synthetic_should_custom_ar="
                        f"{ca_synthetic_eligible!r}"
                    )
    if sorted(record_ranks) != list(range(max(0, runtime_tp_size))):
        runtime_errors.append(
            f"tp_ranks={sorted(record_ranks)!r}:"
            f"expected={list(range(max(0, runtime_tp_size)))!r}"
        )
    if record_active_count != runtime_active_count:
        runtime_errors.append(
            f"record_active_count={record_active_count}:"
            f"declared={runtime_active_count}"
        )
    if record_flashinfer_preemptor is not runtime_flashinfer_preemptor:
        runtime_errors.append(
            "flashinfer_preemptor_mismatch:"
            f"records={record_flashinfer_preemptor!r}:"
            f"declared={runtime_flashinfer_preemptor!r}"
        )
    if record_torch_symm_mem_preemptor is not runtime_torch_symm_mem_preemptor:
        runtime_errors.append(
            "torch_symm_mem_preemptor_mismatch:"
            f"records={record_torch_symm_mem_preemptor!r}:"
            f"declared={runtime_torch_symm_mem_preemptor!r}"
        )
    if record_nccl_symm_mem_preemptor is not runtime_nccl_symm_mem_preemptor:
        runtime_errors.append(
            "nccl_symm_mem_preemptor_mismatch:"
            f"records={record_nccl_symm_mem_preemptor!r}:"
            f"declared={runtime_nccl_symm_mem_preemptor!r}"
        )
    if effective == "enabled" and runtime_flashinfer_preemptor:
        runtime_errors.append("flashinfer_preempts_custom_all_reduce")
    if effective == "enabled" and runtime_torch_symm_mem_preemptor:
        runtime_errors.append("torch_symm_mem_preempts_custom_all_reduce")
    if effective == "enabled" and runtime_nccl_symm_mem_preemptor:
        runtime_errors.append("nccl_symm_mem_preempts_custom_all_reduce")
    if runtime_all_active is not (
        runtime_active_count == runtime_tp_size
    ):
        runtime_errors.append(
            f"all_ranks_active={runtime_all_active!r}:"
            f"active={runtime_active_count}:tp_size={runtime_tp_size}"
        )
    if not runtime_rank_consistent:
        runtime_errors.append("rank_consistent=false")
    if not runtime_passed_raw:
        runtime_errors.append("child_runtime_proof_passed=false")
    record_payload_eligible = bool(
        runtime_records
        and all(
            isinstance(record, dict)
            and record.get("ca_comm_synthetic_should_custom_ar") is True
            for record in runtime_records
        )
    )
    expected_all_payload_eligible = bool(
        effective == "enabled" and record_payload_eligible
    )
    if runtime_all_payload_eligible is not expected_all_payload_eligible:
        runtime_errors.append(
            "all_ranks_payload_eligible_mismatch:"
            f"declared={runtime_all_payload_eligible!r}:"
            f"records={record_payload_eligible!r}:effective={effective!r}"
        )

    return {
        "custom_all_reduce_requested": requested,
        "custom_all_reduce_effective": effective,
        "custom_all_reduce_effective_reason": effective_reason,
        "custom_all_reduce_runtime_proof_required": runtime_required,
        "custom_all_reduce_runtime_proof_passed": bool(
            runtime_passed_raw and not runtime_errors
        ),
        "custom_all_reduce_runtime_configured_effective": (
            runtime_configured_effective
        ),
        "custom_all_reduce_runtime_tensor_parallel_size": runtime_tp_size,
        "custom_all_reduce_runtime_rank_count": runtime_rank_count,
        "custom_all_reduce_runtime_active_rank_count": runtime_active_count,
        "custom_all_reduce_runtime_inactive_rank_count": runtime_inactive_count,
        "custom_all_reduce_runtime_all_ranks_active": runtime_all_active,
        "custom_all_reduce_runtime_rank_consistent": bool(
            runtime_rank_consistent and not runtime_errors
        ),
        "custom_all_reduce_runtime_preemptor_flashinfer_enabled": (
            runtime_flashinfer_preemptor
        ),
        "custom_all_reduce_runtime_preemptor_torch_symm_mem_enabled": (
            runtime_torch_symm_mem_preemptor
        ),
        "custom_all_reduce_runtime_preemptor_nccl_symm_mem_enabled": (
            runtime_nccl_symm_mem_preemptor
        ),
        "custom_all_reduce_runtime_required_num_tokens": (
            runtime_required_num_tokens
        ),
        "custom_all_reduce_runtime_model_hidden_size": runtime_model_hidden_size,
        "custom_all_reduce_runtime_model_dtype": runtime_model_dtype,
        "custom_all_reduce_runtime_model_dtype_bytes": runtime_model_dtype_bytes,
        "custom_all_reduce_runtime_required_payload_bytes": (
            runtime_required_payload_bytes
        ),
        "custom_all_reduce_runtime_all_ranks_payload_eligible": (
            runtime_all_payload_eligible
        ),
        "custom_all_reduce_runtime_records": runtime_records,
        "custom_all_reduce_runtime_proof_error": ";".join(runtime_errors),
    }


def _speed_child_engine_scheduling_provenance(
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """Lift the child-instantiated scheduler state into the parent artifact."""
    field_names = (
        "engine_scheduling_mode_requested",
        "engine_async_scheduling_configured",
        "engine_async_scheduling_effective",
    )
    nested_raw = metrics.get("run_config")
    nested = nested_raw if isinstance(nested_raw, dict) else {}
    return {field_name: nested.get(field_name) for field_name in field_names}


def _scheduler_graph_runtime_contract_reasons(
    contract: dict[str, object],
    *,
    expected: dict[str, object],
    child: str,
) -> list[str]:
    reasons: list[str] = []
    exact_fields = {
        "scheduler_graph_contract_schema": expected["schema"],
        "engine_max_num_seqs_requested": expected["max_num_seqs"],
        "engine_max_num_seqs_effective": expected["max_num_seqs"],
        "engine_max_num_batched_tokens_requested": expected[
            "max_num_batched_tokens"
        ],
        "engine_max_num_batched_tokens_effective": expected[
            "max_num_batched_tokens"
        ],
        "engine_kv_cache_memory_bytes_requested": expected[
            "kv_cache_memory_bytes"
        ],
        "engine_kv_cache_memory_bytes_effective": expected[
            "kv_cache_memory_bytes"
        ],
        "engine_chunked_prefill_requested": expected["chunked_prefill"],
        "engine_chunked_prefill_configured": (
            None
            if expected["chunked_prefill"] == "auto"
            else expected["chunked_prefill"] == "enabled"
        ),
        "engine_chunked_prefill_effective": (
            expected["chunked_prefill"] == "enabled"
            if expected["chunked_prefill"] != "auto"
            else contract.get("engine_chunked_prefill_effective")
        ),
        "engine_cudagraph_capture_sizes_requested": expected[
            "cudagraph_capture_sizes"
        ],
        "engine_cudagraph_capture_sizes_effective": expected[
            "cudagraph_capture_sizes"
        ],
        "engine_max_cudagraph_capture_size_effective": max(
            expected["cudagraph_capture_sizes"], default=0
        ),
        "engine_decode_batch_cudagraph_covered": bool(
            expected["full_cuda_graph"]
        ),
        "engine_full_cuda_graph_requested": expected["full_cuda_graph"],
        "engine_full_cuda_graph_effective": expected["full_cuda_graph"],
        "engine_max_seq_len_to_capture_requested": expected[
            "max_seq_len_to_capture"
        ],
    }
    for field, expected_value in exact_fields.items():
        actual = contract.get(field)
        if type(actual) is not type(expected_value) or actual != expected_value:
            reasons.append(
                f"{child}_scheduler_graph_contract_mismatch:"
                f"{field}:actual={actual!r}:expected={expected_value!r}"
            )
    supported = contract.get("engine_max_seq_len_to_capture_supported")
    effective = contract.get("engine_max_seq_len_to_capture_effective")
    expected_sequence_control = (
        "legacy_engine_arg" if supported is True else "not_applicable_v1"
    )
    if contract.get("engine_sequence_length_graph_control") != expected_sequence_control:
        reasons.append(
            f"{child}_scheduler_graph_contract_mismatch:"
            "engine_sequence_length_graph_control:"
            f"actual={contract.get('engine_sequence_length_graph_control')!r}:"
            f"expected={expected_sequence_control!r}"
        )
    if type(supported) is not bool:
        reasons.append(
            f"{child}_scheduler_graph_contract_mismatch:"
            "engine_max_seq_len_to_capture_supported_invalid"
        )
    elif supported:
        expected_value = expected["max_seq_len_to_capture"]
        if type(effective) is not int or effective != expected_value:
            reasons.append(
                f"{child}_scheduler_graph_contract_mismatch:"
                "engine_max_seq_len_to_capture_effective:"
                f"actual={effective!r}:expected={expected_value!r}"
            )
    elif effective is not None:
        reasons.append(
            f"{child}_scheduler_graph_contract_mismatch:"
            "unsupported_max_seq_len_to_capture_claims_effective_value"
        )
    return reasons


def _engine_runtime_pair_geometry(proof: dict[str, Any]) -> dict[str, Any]:
    """Return the explicit arm-common physical state, never admission policy."""
    geometry = proof.get("engine_runtime_physical_geometry")
    return dict(geometry) if isinstance(geometry, dict) else {}


def _apply_scheduler_graph_contract_payload(
    payload: dict[str, Any],
    args: argparse.Namespace,
    *,
    speed_metrics: dict[str, Any],
    diagnostic_metrics: dict[str, Any] | None,
    dense_reference_metrics: dict[str, Any] | None = None,
) -> None:
    expected = requested_scheduler_graph_contract(args)
    children: dict[str, dict[str, object]] = {
        "speed_child": scheduler_graph_runtime_contract_from_metrics(speed_metrics)
    }
    if diagnostic_metrics is not None:
        children["diagnostic_child"] = (
            scheduler_graph_runtime_contract_from_metrics(diagnostic_metrics)
        )
    if dense_reference_metrics is not None:
        children["dense_reference_child"] = (
            scheduler_graph_runtime_contract_from_metrics(dense_reference_metrics)
        )

    runtime_metrics: dict[str, dict[str, Any]] = {
        "speed_child": speed_metrics,
    }
    if diagnostic_metrics is not None:
        runtime_metrics["diagnostic_child"] = diagnostic_metrics
    if dense_reference_metrics is not None:
        runtime_metrics["dense_reference_child"] = dense_reference_metrics

    runtime_proofs: dict[str, dict[str, Any]] = {}
    for child, metrics in runtime_metrics.items():
        nested_raw = metrics.get("run_config")
        nested = nested_raw if isinstance(nested_raw, dict) else {}
        proof = {
            str(key): value
            for source in (nested, metrics)
            for key, value in source.items()
            if str(key).startswith(
                ("engine_runtime_", "engine_core_block_pool_")
            )
        }
        runtime_proofs[child] = proof
        payload[f"{child}_engine_runtime_contract_proof"] = proof

    diagnostic_graph_proof: dict[str, Any] = {}
    if diagnostic_metrics is not None:
        boundary_raw = diagnostic_metrics.get("boundary_diagnostics")
        boundary = boundary_raw if isinstance(boundary_raw, dict) else {}
        diagnostic_graph_proof = {
            str(key): value
            for key, value in boundary.items()
            if str(key).startswith("cudagraph_runtime_observer_")
        }
    payload["diagnostic_child_cudagraph_runtime_proof"] = diagnostic_graph_proof

    reasons: list[str] = []
    for child, contract in children.items():
        reasons.extend(
            _scheduler_graph_runtime_contract_reasons(
                contract,
                expected=expected,
                child=child,
            )
        )
    contracts = list(children.values())
    if len(contracts) > 1 and any(
        contract != contracts[0] for contract in contracts[1:]
    ):
        reasons.append("scheduler_graph_child_contracts_diverged")

    expected_request_vectors = {
        "request_context_tokens": _request_context_tokens(args),
        "request_max_new_tokens": _request_max_new_tokens(args),
    }
    observed_request_vectors: dict[str, dict[str, object]] = {}
    request_vector_reasons: list[str] = []
    for child, metrics in runtime_metrics.items():
        run_config_raw = metrics.get("run_config")
        run_config = run_config_raw if isinstance(run_config_raw, dict) else {}
        observed = {
            field: run_config.get(field)
            for field in expected_request_vectors
        }
        observed_request_vectors[child] = observed
        for field, expected_value in expected_request_vectors.items():
            if observed.get(field) != expected_value:
                request_vector_reasons.append(
                    f"{child}_{field}_mismatch"
                )
    payload["request_vector_contract_expected"] = expected_request_vectors
    payload["request_vector_contract_observed"] = observed_request_vectors
    payload["request_vector_contract_match"] = not request_vector_reasons
    payload["request_vector_contract_reasons"] = request_vector_reasons
    reasons.extend(request_vector_reasons)

    expected_decode_step_contract = {
        "decode_step_contract_schema": "sfi.single_token_decode_step.v1",
        "decode_step_contract_proof_passed": True,
        "decode_step_speculative_config_present": False,
        "decode_step_stream_interval": 1,
        "decode_step_max_tokens_per_request": 1,
    }
    decode_step_contract_reasons: list[str] = []
    for child, metrics in runtime_metrics.items():
        run_config_raw = metrics.get("run_config")
        run_config = run_config_raw if isinstance(run_config_raw, dict) else {}
        proof = {
            field: run_config.get(field, metrics.get(field))
            for field in expected_decode_step_contract
        }
        payload[f"{child}_decode_step_contract_proof"] = proof
        if proof != expected_decode_step_contract:
            decode_step_contract_reasons.append(
                f"{child}_decode_step_contract_mismatch"
            )
    payload["decode_step_contract_expected"] = expected_decode_step_contract
    payload["decode_step_contract_match"] = not decode_step_contract_reasons
    payload["decode_step_contract_reasons"] = decode_step_contract_reasons
    reasons.extend(decode_step_contract_reasons)

    runtime_contract_required = (
        os.environ.get("SFI_RUNNER_MODEL_KV_CONTRACT_SCHEMA")
        == MODEL_KV_CONTRACT_SCHEMA
    )
    runtime_reasons: list[str] = []
    if runtime_contract_required:
        for child, proof in runtime_proofs.items():
            if proof.get("engine_runtime_contract_proof_required") is not True:
                runtime_reasons.append(f"{child}_runtime_proof_not_required")
            if proof.get("engine_runtime_contract_proof_passed") is not True:
                runtime_reasons.append(f"{child}_runtime_proof_not_green")
            if proof.get("engine_runtime_graph_mode") != "FULL":
                runtime_reasons.append(f"{child}_runtime_graph_not_full")
            if proof.get("engine_runtime_graph_capture_sizes") != [int(args.batch_size)]:
                runtime_reasons.append(f"{child}_runtime_capture_sizes_mismatch")
            if proof.get(
                "engine_runtime_graph_required_decode_dispatch_passed"
            ) is not True:
                runtime_reasons.append(
                    f"{child}_runtime_required_decode_dispatch_failed"
                )
            if proof.get("engine_runtime_kv_capacity_covers_required_total") is not True:
                runtime_reasons.append(f"{child}_runtime_kv_capacity_not_green")
            if proof.get("engine_core_block_pool_proof_required") is not True:
                runtime_reasons.append(f"{child}_block_pool_proof_not_required")
            if proof.get("engine_core_block_pool_proof_passed") is not True:
                runtime_reasons.append(f"{child}_block_pool_proof_not_green")

        if diagnostic_metrics is not None:
            runtime_reasons.extend(
                "diagnostic_cudagraph_runtime_proof_invalid:" + reason
                for reason in cudagraph_runtime_observer_proof_reasons(
                    diagnostic_graph_proof,
                    expected_batch_size=int(args.batch_size),
                )
            )

        geometries = {
            child: _engine_runtime_pair_geometry(proof)
            for child, proof in runtime_proofs.items()
        }
        reference_geometry = geometries.get("speed_child")
        for child, geometry in geometries.items():
            if geometry != reference_geometry:
                runtime_reasons.append(
                    f"{child}_worker_runtime_geometry_diverged"
                )
        payload["engine_runtime_geometry"] = reference_geometry
        payload["engine_runtime_geometry_digest"] = build_config_digest(
            reference_geometry or {}
        )
    payload["engine_runtime_contract_match"] = not runtime_reasons
    payload["engine_runtime_contract_reasons"] = runtime_reasons
    if runtime_reasons:
        reasons.extend(runtime_reasons)

    payload["scheduler_graph_contract_expected"] = expected
    payload.update(
        {
            f"{child}_scheduler_graph_contract": contract
            for child, contract in children.items()
        }
    )
    payload["scheduler_graph_contract_match"] = not reasons
    payload["scheduler_graph_contract_reasons"] = reasons
    provenance = payload.get("run_provenance")
    if isinstance(provenance, dict):
        provenance["scheduler_graph_contract_expected"] = expected
        provenance["speed_child_scheduler_graph_contract"] = children[
            "speed_child"
        ]
    if reasons:
        payload["gate_passed"] = False
        payload["production_gate_passed"] = False


def _sparse_dense_pair_contract_kind() -> str:
    """Resolve the explicitly requested sparse/dense comparison contract."""
    contract = os.environ.get("SFI_RUNNER_PAIR_CONTRACT", "none").strip()
    if contract not in {
        "none",
        "explicit_local_comparison",
        "exact_speedup_verdict",
    }:
        raise RuntimeError(
            "SFI_RUNNER_PAIR_CONTRACT must be none or "
            "explicit_local_comparison or exact_speedup_verdict, "
            f"got {contract!r}"
        )
    return contract


def _apply_sparse_dense_pair_speedup_payload(
    payload: dict[str, Any],
    args: argparse.Namespace,
    *,
    sparse_metrics: dict[str, Any],
    dense_metrics: dict[str, Any] | None,
    sparse_env: dict[str, str],
    dense_env: dict[str, str] | None,
) -> None:
    """Promote the semantic dense reference into an adjacent timed pair arm."""
    contract_kind = _sparse_dense_pair_contract_kind()
    exact_required = contract_kind == "exact_speedup_verdict"
    local_comparison_required = contract_kind == "explicit_local_comparison"
    pair_required = exact_required or local_comparison_required
    request_context_vector = _request_context_tokens(args)
    request_max_new_vector = _request_max_new_tokens(args)
    all_decode_window_alignment_required = bool(
        len(set(request_context_vector)) <= 1
        and len(set(request_max_new_vector)) <= 1
    )
    reasons: list[str] = []
    provenance_raw = payload.get("run_provenance")
    provenance = provenance_raw if isinstance(provenance_raw, dict) else {}
    dense_metrics_readable = isinstance(dense_metrics, dict) and bool(dense_metrics)
    dense_metrics = dense_metrics if isinstance(dense_metrics, dict) else {}
    dense_env = dense_env if isinstance(dense_env, dict) else {}
    if pair_required and not dense_metrics_readable:
        reasons.append("sparse_dense_pair_dense_metrics_unreadable")
    arm_runner_contract = {
        "sparse": "run_sparse_only.py",
        "dense": "run_dense_only.py",
    }

    def _runner_observed(metrics: dict[str, Any]) -> dict[str, Any]:
        run_config = metrics.get("run_config")
        nested = run_config if isinstance(run_config, dict) else {}
        return {
            "top_level": metrics.get("runner"),
            "run_config": nested.get("runner"),
        }

    arm_runner_observed = {
        "sparse": _runner_observed(sparse_metrics),
        "dense": _runner_observed(dense_metrics),
    }
    arm_runner_contract_passed = all(
        observed.get(location) == arm_runner_contract[arm]
        for arm, observed in arm_runner_observed.items()
        for location in ("top_level", "run_config")
    )
    if pair_required:
        for arm, observed in arm_runner_observed.items():
            for location in ("top_level", "run_config"):
                if observed.get(location) != arm_runner_contract[arm]:
                    reasons.append(
                        f"sparse_dense_pair_{arm}_runner_mismatch:{location}"
                    )

    def _runtime_proof(metrics: dict[str, Any]) -> dict[str, Any]:
        nested_raw = metrics.get("run_config")
        nested = nested_raw if isinstance(nested_raw, dict) else {}
        return {
            str(key): value
            for source in (nested, metrics)
            for key, value in source.items()
            if str(key).startswith("engine_runtime_")
        }

    def _custom_ar_geometry(metrics: dict[str, Any]) -> dict[str, Any]:
        return _speed_child_custom_all_reduce_provenance(
            metrics,
            expected_batch_size=int(args.batch_size),
        )

    def _child_identity(metrics: dict[str, Any]) -> dict[str, Any]:
        run_config = metrics.get("run_config")
        if not isinstance(run_config, dict):
            return {}
        identity = run_config.get("benchmark_child_identity")
        return dict(identity) if isinstance(identity, dict) else {}

    model_kv_raw = provenance.get("runner_model_kv_contract")
    model_kv = model_kv_raw if isinstance(model_kv_raw, dict) else {}
    common_geometry = {
        "schema": "sfi.sparse_dense_pair_geometry.v1",
        "python": provenance.get("python"),
        "git_head": provenance.get("git_head"),
        "expected_git_commit": provenance.get("runner_expected_git_commit"),
        "model": provenance.get("model"),
        "model_config_sha256": model_kv.get("model_config_sha256"),
        "corpus_sha256": provenance.get("runner_corpus_sha256"),
        "tensor_parallel_size": provenance.get("tensor_parallel_size"),
        "cuda_visible_devices": provenance.get("cuda_visible_devices_env"),
        "cuda_capabilities": provenance.get("runner_cuda_capabilities"),
        "batch_size": int(args.batch_size),
        "context_tokens": provenance.get("runner_context_tokens"),
        "request_context_tokens": provenance.get("request_context_tokens"),
        "max_new_tokens": _effective_max_new_tokens(args),
        "request_max_new_tokens": provenance.get("request_max_new_tokens"),
        "max_model_len": int(args.max_model_len),
        "attention_backend_artifact": provenance.get("backend_artifact"),
        "attention_build_identity": provenance.get(
            "runner_attention_build_identity"
        ),
        "gpu_lock_mode": provenance.get("runner_gpu_lock_mode"),
        "gpu_lock_scope": provenance.get("runner_gpu_lock_scope"),
    }
    required_common_fields = {
        "model_config_sha256",
        "corpus_sha256",
        "tensor_parallel_size",
        "cuda_visible_devices",
        "cuda_capabilities",
        "batch_size",
        "context_tokens",
        "request_context_tokens",
        "max_new_tokens",
        "request_max_new_tokens",
        "max_model_len",
        "attention_backend_artifact",
        "attention_build_identity",
        "gpu_lock_mode",
        "gpu_lock_scope",
    }
    if pair_required:
        for field in sorted(required_common_fields):
            value = common_geometry.get(field)
            if value in (None, "", [], {}):
                reasons.append(f"sparse_dense_pair_geometry_missing:{field}")
    expected_child_identity = {
        "schema": "sfi.benchmark_child_identity.v1",
        "model_path": provenance.get("model"),
        "model_config_path": model_kv.get("model_config_path"),
        "model_config_sha256": model_kv.get("model_config_sha256"),
        "parent_model_config_sha256": model_kv.get("model_config_sha256"),
        "expected_model_config_sha256": provenance.get(
            "runner_expected_model_config_sha256"
        ),
        "prompt_path": provenance.get("prompt"),
        "corpus_sha256": provenance.get("runner_corpus_sha256"),
        "expected_corpus_sha256": provenance.get("runner_corpus_sha256"),
        "tensor_parallel_size": provenance.get("tensor_parallel_size"),
        "cuda_visible_devices": provenance.get("cuda_visible_devices_env"),
        "cuda_capabilities": provenance.get("runner_cuda_capabilities"),
        "batch_size": int(args.batch_size),
        "context_tokens": int(provenance.get("runner_context_tokens") or 0),
        "request_context_tokens": _request_context_tokens(args),
        "max_new_tokens": _effective_max_new_tokens(args),
        "request_max_new_tokens": _request_max_new_tokens(args),
        "max_model_len": int(args.max_model_len),
        "runner_tier": provenance.get("runner_tier"),
        "expected_git_commit": provenance.get("runner_expected_git_commit"),
        "gpu_lock_mode": provenance.get("runner_gpu_lock_mode"),
        "gpu_lock_scope": provenance.get("runner_gpu_lock_scope"),
    }
    sparse_child_identity = _child_identity(sparse_metrics)
    dense_child_identity = _child_identity(dense_metrics)
    # The backend name is an arm input, not physical pair geometry: sparse is
    # intentionally routed through FLASH_ATTN while the native dense reference
    # is intentionally routed through FLASH_ATTN_VLLM_V1.  Compare the shared
    # child identity projection and gate each backend independently; retaining
    # the two full records below keeps the audit lossless.
    arm_invariant_child_identity_fields = (
        "schema",
        "model_path",
        "model_config_path",
        "model_config_sha256",
        "parent_model_config_sha256",
        "expected_model_config_sha256",
        "prompt_path",
        "corpus_sha256",
        "expected_corpus_sha256",
        "tensor_parallel_size",
        "cuda_visible_devices",
        "cuda_capabilities",
        "batch_size",
        "context_tokens",
        "request_context_tokens",
        "max_new_tokens",
        "request_max_new_tokens",
        "max_model_len",
        "runner_tier",
        "expected_git_commit",
        "gpu_lock_mode",
        "gpu_lock_scope",
        "flash_attn_version",
        "attention_build_identity_json",
    )

    def _arm_invariant_child_identity(
        identity: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            field: identity.get(field)
            for field in arm_invariant_child_identity_fields
        }

    sparse_child_identity_invariant = _arm_invariant_child_identity(
        sparse_child_identity
    )
    dense_child_identity_invariant = _arm_invariant_child_identity(
        dense_child_identity
    )
    arm_backend_contract = {
        "sparse": "FLASH_ATTN",
        "dense": "FLASH_ATTN_VLLM_V1",
    }
    arm_backend_contract_passed = all(
        identity.get("attention_backend") == arm_backend_contract[arm]
        for arm, identity in (
            ("sparse", sparse_child_identity),
            ("dense", dense_child_identity),
        )
    )
    if pair_required:
        for arm, identity in (
            ("sparse", sparse_child_identity),
            ("dense", dense_child_identity),
        ):
            for field, expected in expected_child_identity.items():
                if identity.get(field) != expected:
                    reasons.append(
                        f"sparse_dense_pair_{arm}_child_identity_mismatch:"
                        f"{field}"
                    )
            expected_backend = arm_backend_contract[arm]
            if identity.get("attention_backend") != expected_backend:
                reasons.append(
                    f"sparse_dense_pair_{arm}_attention_backend_mismatch"
                )
            if identity.get("flash_attn_version") != provenance.get(
                "runner_flash_attn_version"
            ):
                reasons.append(
                    f"sparse_dense_pair_{arm}_flash_attn_version_mismatch"
                )
            build_identity_raw = identity.get("attention_build_identity_json")
            try:
                build_identity = json.loads(build_identity_raw)
            except (TypeError, ValueError, json.JSONDecodeError):
                build_identity = None
            if build_identity != provenance.get("runner_attention_build_identity"):
                reasons.append(
                    f"sparse_dense_pair_{arm}_attention_build_identity_mismatch"
                )
        if provenance.get("runner_gpu_lock_mode") != "exclusive":
            reasons.append("sparse_dense_pair_gpu_lock_not_exclusive")
        if provenance.get("runner_gpu_lock_scope") != "pair":
            reasons.append("sparse_dense_pair_gpu_lock_scope_not_pair")
    sparse_worker_geometry = _engine_runtime_pair_geometry(
        _runtime_proof(sparse_metrics)
    )
    dense_worker_geometry = _engine_runtime_pair_geometry(
        _runtime_proof(dense_metrics)
    )
    sparse_custom_ar_geometry = _custom_ar_geometry(sparse_metrics)
    dense_custom_ar_geometry = _custom_ar_geometry(dense_metrics)
    if exact_required:
        for arm, custom_ar in (
            ("sparse", sparse_custom_ar_geometry),
            ("dense", dense_custom_ar_geometry),
        ):
            if custom_ar.get("custom_all_reduce_runtime_proof_required") is not True:
                reasons.append(f"sparse_dense_pair_{arm}_custom_ar_not_required")
            if custom_ar.get("custom_all_reduce_runtime_proof_passed") is not True:
                reasons.append(f"sparse_dense_pair_{arm}_custom_ar_not_green")
            if custom_ar.get("custom_all_reduce_runtime_proof_error") != "":
                reasons.append(f"sparse_dense_pair_{arm}_custom_ar_error")
        expected_capabilities = str(
            provenance.get("runner_cuda_capabilities", "") or ""
        ).split(",")
        for arm, worker_geometry in (
            ("sparse", sparse_worker_geometry),
            ("dense", dense_worker_geometry),
        ):
            rank_records = worker_geometry.get("rank_records")
            if not isinstance(rank_records, list) or len(rank_records) != len(
                expected_capabilities
            ):
                reasons.append(
                    f"sparse_dense_pair_{arm}_cuda_rank_records_mismatch"
                )
                continue
            for rank, record in enumerate(rank_records):
                device = (
                    record.get("cuda_device_runtime")
                    if isinstance(record, dict)
                    else None
                )
                if (
                    not isinstance(device, dict)
                    or device.get("current_device") != rank
                    or device.get("capability") != expected_capabilities[rank]
                ):
                    reasons.append(
                        f"sparse_dense_pair_{arm}_cuda_runtime_mismatch:"
                        f"rank={rank}"
                    )
    sparse_geometry = {
        **common_geometry,
        "child_identity": sparse_child_identity_invariant,
        "scheduler": payload.get("speed_child_scheduler_graph_contract"),
        "worker_runtime": sparse_worker_geometry,
        "custom_all_reduce": sparse_custom_ar_geometry,
    }
    dense_geometry = {
        **common_geometry,
        "child_identity": dense_child_identity_invariant,
        "scheduler": payload.get("dense_reference_child_scheduler_graph_contract"),
        "worker_runtime": dense_worker_geometry,
        "custom_all_reduce": dense_custom_ar_geometry,
    }
    sparse_digest = build_config_digest(sparse_geometry)
    dense_digest = build_config_digest(dense_geometry)
    geometry_match = sparse_geometry == dense_geometry
    if pair_required and not geometry_match:
        reasons.append("sparse_dense_pair_geometry_mismatch")

    sparse_observer_env = _gate_d_trace_profile_env(sparse_env)
    dense_observer_env = _gate_d_trace_profile_env(dense_env)
    sparse_boundary_raw = sparse_metrics.get("boundary_diagnostics")
    sparse_boundary = (
        sparse_boundary_raw if isinstance(sparse_boundary_raw, dict) else {}
    )
    dense_boundary_raw = dense_metrics.get("boundary_diagnostics")
    dense_boundary = (
        dense_boundary_raw if isinstance(dense_boundary_raw, dict) else {}
    )
    observer_free = bool(
        not sparse_observer_env
        and not dense_observer_env
        and sparse_boundary.get("cudagraph_runtime_observer_enabled") is not True
        and dense_boundary.get("cudagraph_runtime_observer_enabled") is not True
    )
    if pair_required and not observer_free:
        reasons.append("sparse_dense_timed_pair_not_observer_free")

    pair_boundaries: dict[str, dict[str, Any]] = {}
    for arm, arm_metrics in (
        ("sparse", sparse_metrics),
        ("dense", dense_metrics),
    ):
        boundary_raw = arm_metrics.get("boundary_diagnostics")
        boundary = boundary_raw if isinstance(boundary_raw, dict) else {}
        if pair_required:
            pair_boundaries[arm] = boundary
            if boundary.get("all_decode_entered") is not True:
                reasons.append(
                    f"sparse_dense_pair_{arm}_all_decode_entered_mismatch"
                )
            zero_steps = boundary.get("all_decode_zero_token_steps")
            if type(zero_steps) is not int or zero_steps != 0:
                reasons.append(
                    f"sparse_dense_pair_{arm}_all_decode_zero_token_steps_mismatch"
                )
            full_steps = boundary.get("all_decode_full_batch_steps")
            if type(full_steps) is not int or full_steps <= 0:
                reasons.append(
                    f"sparse_dense_pair_{arm}_all_decode_full_steps_missing"
                )
            partial_steps = boundary.get("all_decode_partial_batch_steps")
            if type(partial_steps) is not int or partial_steps < 0:
                reasons.append(
                    f"sparse_dense_pair_{arm}_all_decode_partial_steps_invalid"
                )
            all_decode_steps = boundary.get("all_decode_steps")
            if type(all_decode_steps) is not int or all_decode_steps <= 0:
                reasons.append(
                    f"sparse_dense_pair_{arm}_all_decode_steps_invalid"
                )
            elif (
                type(full_steps) is int
                and type(partial_steps) is int
                and type(zero_steps) is int
                and all_decode_steps
                != full_steps + partial_steps + zero_steps
            ):
                reasons.append(
                    f"sparse_dense_pair_{arm}_all_decode_step_accounting_mismatch"
                )

    if (
        pair_required
        and all_decode_window_alignment_required
        and set(pair_boundaries) == {"sparse", "dense"}
    ):
        for field in (
            "all_decode_entered",
            "all_decode_steps",
            "all_decode_full_batch_steps",
            "all_decode_partial_batch_steps",
            "all_decode_zero_token_steps",
        ):
            if pair_boundaries["sparse"].get(field) != pair_boundaries[
                "dense"
            ].get(field):
                reasons.append(f"sparse_dense_pair_{field}_mismatch")

    def _positive_metric(metrics: dict[str, Any], key: str) -> float | None:
        value = _as_float(metrics.get(key), float("nan"))
        return value if math.isfinite(value) and value > 0.0 else None

    def _ratio(numerator: float | None, denominator: float | None) -> float | None:
        if numerator is None or denominator is None:
            return None
        return numerator / denominator

    sparse_elapsed = _positive_metric(sparse_metrics, "elapsed_s")
    dense_elapsed = _positive_metric(dense_metrics, "elapsed_s")
    sparse_total_tps = _positive_metric(sparse_metrics, "tok_per_s")
    dense_total_tps = _positive_metric(dense_metrics, "tok_per_s")
    sparse_decode_tps = _positive_metric(sparse_metrics, "decode_tok_per_s")
    dense_decode_tps = _positive_metric(dense_metrics, "decode_tok_per_s")
    sparse_all_decode_tps = _positive_metric(
        sparse_metrics, "all_decode_tok_per_s"
    )
    dense_all_decode_tps = _positive_metric(dense_metrics, "all_decode_tok_per_s")

    total_wall_speedup = _ratio(dense_elapsed, sparse_elapsed)
    decode_speedup = _ratio(sparse_decode_tps, dense_decode_tps)
    total_token_speedup = _ratio(sparse_total_tps, dense_total_tps)
    all_decode_speedup = _ratio(sparse_all_decode_tps, dense_all_decode_tps)
    ratios = {
        "total_wall_speedup": total_wall_speedup,
        "total_token_speedup": total_token_speedup,
        "decode_speedup": decode_speedup,
        "all_decode_speedup": all_decode_speedup,
    }
    if local_comparison_required and (
        all_decode_speedup is None
        or not math.isfinite(all_decode_speedup)
        or all_decode_speedup <= 0.0
    ):
        reasons.append("all_decode_speedup_not_finite_positive")
    elif exact_required and (
        all_decode_speedup is None
        or not math.isfinite(all_decode_speedup)
        or all_decode_speedup <= 1.0
    ):
        reasons.append("all_decode_speedup_not_above_one")

    compared_token_fields = ["out_tokens", "decode_tokens"]
    if all_decode_window_alignment_required:
        compared_token_fields.append("all_decode_tokens")
    for field in compared_token_fields:
        if pair_required and sparse_metrics.get(field) != dense_metrics.get(field):
            reasons.append(f"sparse_dense_{field}_mismatch")

    speedup_observed = bool(
        all_decode_speedup is not None
        and math.isfinite(all_decode_speedup)
        and all_decode_speedup > 1.0
    )

    payload.update(
        {
            "sparse_dense_pair_contract_kind": contract_kind,
            "sparse_dense_pair_required": pair_required,
            "sparse_dense_pair_all_decode_window_alignment_required": (
                all_decode_window_alignment_required
            ),
            "sparse_dense_pair_dense_metrics_readable": dense_metrics_readable,
            "sparse_dense_pair_execution_order": (
                "sparse_speed,dense_reference,sparse_diagnostic"
            ),
            "sparse_dense_pair_scope": (
                "same_parent_same_gpu_lock_adjacent_observer_free_engine_loop"
            ),
            "sparse_dense_pair_total_wall_scope": (
                "measurement_engine_loop_prefill_plus_decode_excludes_engine_init"
            ),
            "sparse_dense_pair_sparse_observer_env": sparse_observer_env,
            "sparse_dense_pair_dense_observer_env": dense_observer_env,
            "sparse_dense_pair_observer_free": observer_free,
            "sparse_dense_pair_child_identity_scope": (
                "arm_invariant_projection"
            ),
            "sparse_dense_pair_arm_backend_contract": arm_backend_contract,
            "sparse_dense_pair_arm_backend_contract_passed": (
                arm_backend_contract_passed
            ),
            "sparse_dense_pair_arm_runner_contract": arm_runner_contract,
            "sparse_dense_pair_arm_runner_observed": arm_runner_observed,
            "sparse_dense_pair_arm_runner_contract_passed": (
                arm_runner_contract_passed
            ),
            "sparse_dense_pair_sparse_child_identity": sparse_child_identity,
            "sparse_dense_pair_dense_child_identity": dense_child_identity,
            "sparse_dense_pair_sparse_geometry_digest": sparse_digest,
            "sparse_dense_pair_dense_geometry_digest": dense_digest,
            "sparse_dense_pair_geometry_match": geometry_match,
            "sparse_dense_pair_sparse_geometry": sparse_geometry,
            "sparse_dense_pair_dense_geometry": dense_geometry,
            "sparse_dense_pair_sparse_elapsed_s": sparse_elapsed,
            "sparse_dense_pair_dense_elapsed_s": dense_elapsed,
            "sparse_dense_pair_sparse_total_tps": sparse_total_tps,
            "sparse_dense_pair_dense_total_tps": dense_total_tps,
            "sparse_dense_pair_sparse_decode_tps": sparse_decode_tps,
            "sparse_dense_pair_dense_decode_tps": dense_decode_tps,
            "sparse_dense_pair_sparse_all_decode_tps": sparse_all_decode_tps,
            "sparse_dense_pair_dense_all_decode_tps": dense_all_decode_tps,
            "sparse_dense_pair_verdict_metric": "all_decode_speedup",
            "sparse_dense_pair_diagnostic_metrics": [
                "total_wall_speedup",
                "total_token_speedup",
                "decode_speedup",
            ],
            **ratios,
            # Local comparison validity and an exact speedup verdict are
            # deliberately separate contracts.  A finite, slower local pair
            # remains useful comparison data, but must never serialize a
            # misleading green speedup gate.
            "sparse_dense_pair_comparison_gate_passed": (
                not reasons if local_comparison_required else None
            ),
            "sparse_dense_pair_comparison_gate_reasons": (
                reasons if local_comparison_required else []
            ),
            "sparse_dense_pair_speedup_observed": speedup_observed,
            "sparse_dense_pair_speedup_gate_passed": (
                not reasons if exact_required else None
            ),
            "sparse_dense_pair_speedup_gate_reasons": (
                reasons if exact_required else []
            ),
            "sparse_dense_pair_arm_modes": {
                "sparse": "sparse",
                "dense": "dense",
            },
            "sparse_dense_pair_controller_identities": {
                "sparse": {
                    "mode": "sparse",
                    "controller_json": sparse_env.get(
                        "VLLM_SPARSE_CONTROLLER_JSON", ""
                    ),
                },
                "dense": {
                    "mode": "dense",
                    "controller_json": dense_env.get(
                        "VLLM_SPARSE_CONTROLLER_JSON", ""
                    ),
                },
            },
            "sparse_dense_pair_claim": (
                "sparse_all_decode_steady_speedup"
                if exact_required and not reasons
                else (
                    "paired_engine_loop_comparison"
                    if local_comparison_required and not reasons
                    else "arm_health_only"
                )
            ),
        }
    )
    if pair_required and reasons:
        payload["gate_passed"] = False
        payload["production_gate_passed"] = False


def _gate_d_payload(
    args: argparse.Namespace,
    *,
    mode: str,
    result: Phase1CommandResult,
    metrics_path: Path,
    outputs_path: Path,
    metrics: dict[str, Any],
    route_counter_metrics: dict[str, Any] | None = None,
    route_trace_path: Path | None,
    route_summary: dict[str, Any] | None = None,
    producer_route_summary: dict[str, Any] | None = None,
    route_proof: dict[str, object] | None = None,
    dense_fa3_proof: dict[str, Any] | None = None,
    speed_env: dict[str, str] | None = None,
    diagnostic_env: dict[str, str] | None = None,
    diag_result: Phase1CommandResult | None = None,
    diag_metrics_path: Path | None = None,
    diag_outputs_path: Path | None = None,
    refresh_profile_path: Path | None = None,
    refresh_profile: list[dict[str, Any]] | None = None,
    hook_profile_summary: dict[str, Any] | None = None,
    selector_pipeline_cpu_profile_path: Path | None = None,
    selector_pipeline_cpu_profile: list[dict[str, Any]] | None = None,
    output_records: list[dict[str, Any]] | None = None,
    fa3_so_sha256: str = "",
    run_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    route_summary = route_summary or {}
    producer_route_summary = producer_route_summary or route_summary
    custom_all_reduce_provenance = _speed_child_custom_all_reduce_provenance(
        metrics,
        expected_batch_size=int(getattr(args, "batch_size", 1) or 1),
    )
    engine_scheduling_provenance = _speed_child_engine_scheduling_provenance(
        metrics
    )
    speed_child_engine_scheduling = {
        f"speed_child_{field_name}": value
        for field_name, value in engine_scheduling_provenance.items()
    }
    speed_child_custom_all_reduce_runtime = {
        f"speed_child_{field_name}": value
        for field_name, value in custom_all_reduce_provenance.items()
        if field_name.startswith("custom_all_reduce_runtime_")
    }

    def _route_summary_proof_value(key: str, default: Any = None) -> Any:
        if key in producer_route_summary:
            return producer_route_summary.get(key)
        return route_summary.get(key, default)

    def _route_summary_proof_scope(*keys: str) -> str:
        if any(key in producer_route_summary for key in keys):
            return "measurement_window"
        if any(key in route_summary for key in keys):
            return "full_diagnostic_run"
        return ""

    row_source_distribution = dict(route_summary.get("row_source_distribution", {}) or {})
    dense_native_fallback_count = int(
        route_summary.get("dense_fallback_call_count", 0) or 0
    ) + int(route_summary.get("native_full_block_table_rows", 0) or 0) + int(
        row_source_distribution.get("compact_full_native_fallback_rows", 0) or 0
    )
    vllm_triton_attention_call_count = int(
        route_summary.get("triton_attention_call_count", 0) or 0
    )
    actual_fwd_count = int(route_summary.get("actual_fwd_mixed_page_count", 0) or 0)
    resolved_fwd_count = int(
        route_summary.get("resolved_row_ptr_fwd_mixed_page_count", 0) or 0
    )
    producer_mode = _producer_mode(args)
    if mode == "dense":
        dense_proof_passed = bool((dense_fa3_proof or {}).get("passed", False))
        backend = "FA3" if dense_proof_passed else "unknown"
        route = "native_dense" if dense_proof_passed else "missing"
        full_cudagraph_enabled = bool(args.full_cuda_graph)
    else:
        route_proof_passed = bool((route_proof or {}).get("passed", False))
        backend = "FA3" if route_proof_passed else "unknown"
        route = "ResolvedRowPtr" if resolved_fwd_count > 0 else "missing"
        full_cudagraph_enabled = bool(
            route_summary.get("full_graph_replay_refresh_seen", False)
        )
    route_proof_passed = bool(
        (route_proof or {}).get("passed", bool(mode == "dense"))
    )
    diagnostic_child_route_counter_proof = (
        {"passed": True, "reasons": [], "scope": "not_required_for_dense"}
        if mode == "dense"
        else _diagnostic_route_counter_proof(
            route_counter_metrics or {},
            require_resolved_row_ptr=_producer_mode_requires_refresh(producer_mode),
            route_summary=route_summary,
            producer_route_summary=producer_route_summary,
        )
    )
    diagnostic_child_route_proof_passed = bool(
        diagnostic_child_route_counter_proof.get("passed", False)
    )
    speed_child_fatal_error = _command_output_has_fatal_error(result)
    diagnostic_child_fatal_error = _command_output_has_fatal_error(diag_result)
    diagnostic_ok = bool(
        diag_result is None
        or (
            diag_result.returncode == 0
            and not bool(diag_result.timed_out)
            and not diagnostic_child_fatal_error
        )
    )
    if output_records is None:
        output_records = _request_ordered_output_records(outputs_path)
    output_length_gate = _output_completion_gate(
        output_records,
        expected_request_count=int(getattr(args, "batch_size", 1) or 1),
        expected_output_tokens=_effective_max_new_tokens(args),
        expected_output_tokens_by_request=_request_max_new_tokens(args),
        require_exact_length=not bool(getattr(args, "respect_eos", False)),
    )
    refresh_profile = refresh_profile or []
    hook_profile_summary = hook_profile_summary or {}
    selector_pipeline_cpu_profile = selector_pipeline_cpu_profile or []
    refresh_reason_counts = _refresh_reason_counts_from_route_summary(
        producer_route_summary
    )
    continuous_refresh_reqs = _continuous_refresh_payloads_from_sources(
        refresh_profile,
        producer_route_summary,
    )
    # [INTENTS-SEMANTICS 口径 2026-07-10] 名为 intents 实为 interval-reason 的
    # 世代 enqueue 计数(refresh_reason_counts["interval"],按 payload enqueue
    # 事件×req_count 累加),非 token-time 意图数——判读膨胀比值时以"世代数"
    # 口径解读(远端 5.66× 案即此语义,勿再误读)。改名会破坏远端对照口径,保名注释。
    interval_trigger_intents = int(refresh_reason_counts.get("interval", 0))
    expected_interval_trigger_intents = _expected_interval_trigger_intents(
        args,
        producer_mode=producer_mode,
    )
    refresh_trigger_intents = sum(refresh_reason_counts.values())
    interval_trigger_requirement_ok = _interval_trigger_requirement_satisfied(
        expected_interval_trigger_intents=expected_interval_trigger_intents,
        refresh_trigger_intents=refresh_trigger_intents,
    )
    sentence_trigger_intents = _counter_int_from_all_sources(
        metrics,
        refresh_profile,
        {},
        producer_route_summary,
        "sentence_trigger_intents",
    )
    gt1_reduce_count = _counter_int_from_all_sources(
        metrics,
        refresh_profile,
        {},
        {},
        "gt1_reduce_count",
    )
    gt1_scalar_fallback_count = _counter_int_from_all_sources(
        metrics,
        refresh_profile,
        {},
        {},
        "gt1_scalar_fallback_count",
    )
    refresh_rebuild_enqueued_count = _sum_int(
        refresh_profile,
        "refresh_rebuild_enqueued_count",
    )
    refresh_rebuild_inline_count = _sum_int(
        refresh_profile,
        "refresh_rebuild_inline_count",
    )
    refresh_rebuild_coalesced_count = _sum_int(
        refresh_profile,
        "refresh_rebuild_coalesced_count",
    )
    deadline_rebuild_drop_finished_count = _max_int_from_records(
        refresh_profile,
        "deadline_rebuild_drop_finished_count",
    )
    deadline_rebuild_drain_finish_count = _max_int_from_records(
        refresh_profile,
        "deadline_rebuild_drain_finish_count",
    )
    deadline_rebuild_partial_finish_count = _max_int_from_records(
        refresh_profile,
        "deadline_rebuild_partial_finish_count",
    )
    deadline_rebuild_drain_submit_count = _max_int_from_records(
        refresh_profile,
        "deadline_rebuild_drain_submit_count",
    )
    deadline_rebuild_drain_submit_decode_step_min = _min_nonnegative_int_from_records(
        refresh_profile,
        "deadline_rebuild_drain_submit_decode_step_min",
    )
    deadline_rebuild_drain_submit_decode_step_max = _max_int_from_records(
        refresh_profile,
        "deadline_rebuild_drain_submit_decode_step_max",
        default=-1,
    )
    deadline_rebuild_drain_submit_decode_steps = _int_list_values_from_records(
        refresh_profile,
        "deadline_rebuild_drain_submit_decode_steps",
    )
    refresh_rebuild_delay_max = _max_int_from_records(
        refresh_profile,
        "refresh_rebuild_delay_max",
    )
    async_refresh_enabled_for_gate = _env_flag_enabled(
        speed_env,
        "VLLM_SPARSE_ASYNC_REFRESH",
    )
    (
        async_producer_writer_count_for_gate,
        async_producer_writer_count_source,
    ) = _async_producer_writer_count_from_sources(
        refresh_profile,
        hook_profile_summary,
        producer_route_summary,
    )
    producer_gate_reasons: list[str] = []
    if mode != "dense":
        producer_requires_refresh = _producer_mode_requires_refresh(producer_mode)
        if producer_requires_refresh:
            if continuous_refresh_reqs <= 0:
                producer_gate_reasons.append("continuous_refresh_reqs_missing")
            if not interval_trigger_requirement_ok:
                producer_gate_reasons.append(
                    "interval_trigger_intents_below_expected"
                )
            if (
                continuous_refresh_reqs > 0
                and async_refresh_enabled_for_gate
                and async_producer_writer_count_for_gate <= 0
            ):
                producer_gate_reasons.append("async_producer_writer_missing")
        elif continuous_refresh_reqs != 0:
            producer_gate_reasons.append("continuous_refresh_reqs_nonzero")
        if (
            _producer_mode_requires_observed_sentence_trigger(producer_mode)
            and sentence_trigger_intents <= 0
        ):
            producer_gate_reasons.append("sentence_trigger_intents_missing")
        if _producer_mode_requires_gt1(producer_mode):
            if max(0, int(getattr(args, "prefill_last_n", 16))) <= 1:
                producer_gate_reasons.append("prefill_last_n_not_gt1")
            if gt1_scalar_fallback_count > 0:
                producer_gate_reasons.append("gt1_scalar_fallback_count_nonzero")
    producer_gate_passed = not producer_gate_reasons
    semantic_output_health = str(
        _sparse_output_content_health(output_records)
        if bool(getattr(args, "outputs_include_text", False))
        else metrics.get(
            "semantic_output_health",
            _sparse_output_content_health(output_records),
        )
    )
    semantic_gate_reasons = _semantic_gate_reasons(mode, semantic_output_health)
    sparse_native_lifecycle_required = False
    sparse_native_lifecycle_gate_reasons: list[str] = []
    steady_records = _steady_prefill_profile_records(refresh_profile)
    tensor_parallel_size = max(
        1,
        int(getattr(args, "tensor_parallel_size", 1) or 1),
    )
    custom_all_reduce_runtime_gate_passed = bool(
        tensor_parallel_size <= 1
        or (
            custom_all_reduce_provenance.get(
                "custom_all_reduce_runtime_proof_passed"
            )
            is True
            and int(
                custom_all_reduce_provenance.get(
                    "custom_all_reduce_runtime_tensor_parallel_size",
                    -1,
                )
            )
            == tensor_parallel_size
        )
    )
    requested_scheduling_mode = str(
        getattr(args, "scheduling_mode", "auto") or "auto"
    )
    engine_scheduling_matches = bool(
        requested_scheduling_mode == "auto"
        or engine_scheduling_provenance
        == {
            "engine_scheduling_mode_requested": requested_scheduling_mode,
            "engine_async_scheduling_configured": (
                requested_scheduling_mode == "async"
            ),
            "engine_async_scheduling_effective": (
                requested_scheduling_mode == "async"
            ),
        }
    )
    production_gate_passed = bool(
        result.returncode == 0
        and not bool(result.timed_out)
        and not speed_child_fatal_error
        and diagnostic_ok
        and route_proof_passed
        and diagnostic_child_route_proof_passed
        and producer_gate_passed
        and bool(output_length_gate["passed"])
        and not semantic_gate_reasons
        and not sparse_native_lifecycle_gate_reasons
        and custom_all_reduce_runtime_gate_passed
        and engine_scheduling_matches
    )

    def _metric_or_route(key: str, default: Any = None) -> Any:
        if key in metrics:
            return metrics.get(key)
        return route_summary.get(key, default)

    def _metric_or_producer_route(key: str, default: Any = None) -> Any:
        if key in metrics:
            return metrics.get(key)
        if key in producer_route_summary:
            return producer_route_summary.get(key)
        return route_summary.get(key, default)

    arena_prepare_miss_count = _as_int(
        _metric_or_route("arena_prepare_miss_count", -1),
        -1,
    )
    bridge_token_count = _as_int(
        producer_route_summary.get(
            "bridge_token_count",
            route_summary.get("bridge_token_count", -1),
        ),
        -1,
    )
    producer_launch_step = _as_int(
        _metric_or_producer_route("producer_launch_step", -1),
        -1,
    )
    producer_ready_step = _as_int(
        _metric_or_producer_route("producer_ready_step", -1),
        -1,
    )
    graph_route_family = (
        producer_route_summary.get(
            "graph_route_family",
            route_summary.get("graph_route_family", {}),
        )
        or {}
    )
    if not isinstance(graph_route_family, dict):
        graph_route_family = {}
    route_family_mismatch: bool | None
    if "route_family_mismatch" in graph_route_family:
        route_family_mismatch = bool(graph_route_family["route_family_mismatch"])
    elif "route_family_mismatch" in producer_route_summary:
        route_family_mismatch = bool(producer_route_summary["route_family_mismatch"])
    elif "route_family_mismatch" in route_summary:
        route_family_mismatch = bool(route_summary["route_family_mismatch"])
    else:
        route_family_mismatch = None
    arena_budget_exceeded_raw = _metric_or_route("arena_budget_exceeded")
    arena_bind_status = str(_metric_or_route("arena_bind_status", "") or "")
    arena_reservation_status = str(
        _metric_or_route("arena_reservation_status", "") or ""
    )

    deadline_v2_attribution = _deadline_v2_attribution_summary(
        metrics=metrics,
        refresh_profile=refresh_profile,
        timeline_summary={},
        route_summary=route_summary,
        selector_pipeline_cpu_profile=selector_pipeline_cpu_profile,
    )

    payload: dict[str, Any] = {
        "schema": "sm80_thin_builder_gate_d_run_v1",
        "mode": mode,
        "producer_mode": producer_mode,
        "continuous_producer_enabled": _phase2_continuous_producer_enabled(args),
        "continuous_refresh_reqs": int(continuous_refresh_reqs),
        "refresh_payloads": int(continuous_refresh_reqs),
        "refresh_reason_counts": dict(refresh_reason_counts),
        "refresh_trigger_intents": int(refresh_trigger_intents),
        "interval_trigger_intents": int(interval_trigger_intents),
        "expected_interval_trigger_intents": int(expected_interval_trigger_intents),
        "interval_trigger_requirement_ok": bool(interval_trigger_requirement_ok),
        "sentence_trigger_intents": int(sentence_trigger_intents),
        "sentence_trigger_enabled": bool(
            _producer_mode_requires_sentence_trigger(producer_mode)
        ),
        "sentence_trigger_observation_required": bool(
            _producer_mode_requires_observed_sentence_trigger(producer_mode)
        ),
        "gt1_reduce_count": int(gt1_reduce_count),
        "gt1_scalar_fallback_count": int(gt1_scalar_fallback_count),
        "gt1_gate_scope": "prefill_last_n_bootstrap",
        "prefill_last_n_gt1_requested": bool(
            max(0, int(getattr(args, "prefill_last_n", 16))) > 1
        ),
        "refresh_rebuild_enqueued_count": int(refresh_rebuild_enqueued_count),
        "refresh_rebuild_inline_count": int(refresh_rebuild_inline_count),
        "refresh_rebuild_coalesced_count": int(refresh_rebuild_coalesced_count),
        "deadline_rebuild_drop_finished_count": int(
            deadline_rebuild_drop_finished_count
        ),
        "deadline_rebuild_drain_finish_count": int(
            deadline_rebuild_drain_finish_count
        ),
        "deadline_rebuild_partial_finish_count": int(
            deadline_rebuild_partial_finish_count
        ),
        "deadline_rebuild_drain_submit_count": int(
            deadline_rebuild_drain_submit_count
        ),
        "deadline_rebuild_drain_submit_decode_step_min": int(
            deadline_rebuild_drain_submit_decode_step_min
        ),
        "deadline_rebuild_drain_submit_decode_step_max": int(
            deadline_rebuild_drain_submit_decode_step_max
        ),
        "deadline_rebuild_drain_submit_decode_steps": list(
            deadline_rebuild_drain_submit_decode_steps
        ),
        "refresh_rebuild_delay_max": int(refresh_rebuild_delay_max),
        "async_producer_writer_count_for_gate": int(
            async_producer_writer_count_for_gate
        ),
        "async_producer_writer_count_source": str(
            async_producer_writer_count_source
        ),
        "hook_pre_consume_pending_rebuild_drained_total": int(
            _as_int(
                hook_profile_summary.get(
                    "pre_consume_pending_rebuild_drained_total"
                ),
                0,
            )
        ),
        "producer_gate_passed": bool(producer_gate_passed),
        "producer_gate_reasons": list(producer_gate_reasons),
        "semantic_gate_reasons": list(semantic_gate_reasons),
        "sparse_native_lifecycle_required": bool(
            sparse_native_lifecycle_required
        ),
        "sparse_native_lifecycle_gate_reasons": list(
            sparse_native_lifecycle_gate_reasons
        ),
        "backend": backend,
        "route": route,
        "full_cudagraph_enabled": bool(full_cudagraph_enabled),
        "dense_native_fallback_count": int(dense_native_fallback_count),
        "vllm_triton_attention_call_count": int(vllm_triton_attention_call_count),
        "actual_fwd_mixed_page_count": int(actual_fwd_count),
        "resolved_row_ptr_fwd_mixed_page_count": int(resolved_fwd_count),
        "decode_p50_us": _decode_metric(metrics, "decode_p50_us"),
        "decode_p95_us": _decode_metric(metrics, "decode_p95_us"),
        "decode_tps": _decode_metric(metrics, "decode_tokens_per_s"),
        # 稳态窗（全部请求完成 chunked prefill 后）：长 ctx 大 bs 档 decode_tps
        # 被 prefill 交错段稀释（8×32k 实测占墙钟 81%），此指标反映真实 decode 吞吐。
        "all_decode_tps": _decode_metric(metrics, "all_decode_tok_per_s"),
        "all_decode_elapsed_s": _decode_metric(metrics, "all_decode_elapsed_s"),
        "all_decode_entered": bool(
            (metrics.get("boundary_diagnostics") or {}).get(
                "all_decode_entered", False
            )
            if isinstance(metrics.get("boundary_diagnostics"), dict)
            else False
        ),
        "all_decode_steps": _as_int(
            (metrics.get("boundary_diagnostics") or {}).get(
                "all_decode_steps", -1
            )
            if isinstance(metrics.get("boundary_diagnostics"), dict)
            else -1,
            -1,
        ),
        "all_decode_full_batch_steps": _as_int(
            (metrics.get("boundary_diagnostics") or {}).get(
                "all_decode_full_batch_steps", -1
            )
            if isinstance(metrics.get("boundary_diagnostics"), dict)
            else -1,
            -1,
        ),
        "all_decode_partial_batch_steps": _as_int(
            (metrics.get("boundary_diagnostics") or {}).get(
                "all_decode_partial_batch_steps", -1
            )
            if isinstance(metrics.get("boundary_diagnostics"), dict)
            else -1,
            -1,
        ),
        "all_decode_zero_token_steps": _as_int(
            (metrics.get("boundary_diagnostics") or {}).get(
                "all_decode_zero_token_steps", -1
            )
            if isinstance(metrics.get("boundary_diagnostics"), dict)
            else -1,
            -1,
        ),
        "vllm_reported_tps": _decode_metric(metrics, "tok_per_s"),
        "elapsed_s": _decode_metric(metrics, "elapsed_s"),
        "decode_elapsed_s": _decode_metric(metrics, "decode_elapsed_s"),
        "front_overhead_s": _decode_metric(metrics, "front_overhead_s"),
        "first_emit_delay_s": _decode_metric(metrics, "first_emit_delay_s"),
        "decode_tokens": _as_int(metrics.get("decode_tokens"), -1),
        "decode_metrics_path": str(metrics_path),
        "outputs_path": str(outputs_path),
        "refresh_profile_path": str(refresh_profile_path) if refresh_profile_path else "",
        "selector_pipeline_cpu_profile_path": (
            str(selector_pipeline_cpu_profile_path)
            if selector_pipeline_cpu_profile_path
            else ""
        ),
        "route_trace_path": str(route_trace_path) if route_trace_path else "",
        "route_summary": route_summary,
        "producer_route_summary": producer_route_summary,
        "route_summary_scope": "full_diagnostic_run",
        "producer_route_summary_scope": "measurement_window",
        "rrp_visible_source_kind": str(
            _route_summary_proof_value("rrp_visible_source_kind", "") or ""
        ),
        "rrp_sparse_dynamic_state_covers_rows": bool(
            _route_summary_proof_value(
                "rrp_sparse_dynamic_state_covers_rows", False
            )
        ),
        "rrp_sparse_dynamic_state_failure_reason": str(
            _route_summary_proof_value(
                "rrp_sparse_dynamic_state_failure_reason", ""
            )
            or ""
        ),
        "rrp_visible_data_ptr": _as_int(
            _route_summary_proof_value("rrp_visible_data_ptr", 0),
            0,
        ),
        "rrp_sparse_dynamic_state_data_ptr": _as_int(
            _route_summary_proof_value("rrp_sparse_dynamic_state_data_ptr", 0),
            0,
        ),
        "rrp_visible_shape": list(
            _route_summary_proof_value("rrp_visible_shape", []) or []
        ),
        "rrp_visible_is_arena_seqused": bool(
            _route_summary_proof_value("rrp_visible_is_arena_seqused", False)
        ),
        "rrp_visible_is_arena_batch_seqused": bool(
            _route_summary_proof_value("rrp_visible_is_arena_batch_seqused", False)
        ),
        "rrp_visible_is_launch_effective": bool(
            _route_summary_proof_value("rrp_visible_is_launch_effective", False)
        ),
        "rrp_visible_is_dense_seqused": bool(
            _route_summary_proof_value("rrp_visible_is_dense_seqused", False)
        ),
        "rrp_launch_effective_covers_rows": bool(
            _route_summary_proof_value("rrp_launch_effective_covers_rows", False)
        ),
        "rrp_launch_effective_k_len_cpu": list(
            _route_summary_proof_value("rrp_launch_effective_k_len_cpu", []) or []
        ),
        "rrp_row_effective_k_by_row": list(
            _route_summary_proof_value("rrp_row_effective_k_by_row", []) or []
        ),
        "rrp_visible_source_scope": _route_summary_proof_scope(
            "rrp_visible_source_kind",
            "rrp_sparse_dynamic_state_covers_rows",
            "rrp_sparse_dynamic_state_failure_reason",
        ),
        "bridge_token_count": int(bridge_token_count),
        "producer_launch_step": int(producer_launch_step),
        "producer_ready_step": int(producer_ready_step),
        "route_family_mismatch": route_family_mismatch,
        "graph_route_family": graph_route_family,
        "bridge_token_positions_exact_once": (
            bool(producer_route_summary["bridge_token_positions_exact_once"])
            if "bridge_token_positions_exact_once" in producer_route_summary
            else None
        ),
        "compact_middle_excludes_bridge_positions": (
            bool(producer_route_summary["compact_middle_excludes_bridge_positions"])
            if "compact_middle_excludes_bridge_positions" in producer_route_summary
            else None
        ),
        "bootstrap_full_kv_handoff": bool(
            producer_route_summary.get("bootstrap_full_kv_handoff", False)
        ),
        "route_proof": route_proof or {"passed": mode == "dense", "reasons": []},
        "route_proof_passed": route_proof_passed,
        "diagnostic_child_route_counter_proof": diagnostic_child_route_counter_proof,
        "diagnostic_child_route_proof_passed": diagnostic_child_route_proof_passed,
        "diagnostic_child_route_proof_reasons": list(
            diagnostic_child_route_counter_proof.get("reasons", [])
        ),
        **speed_child_engine_scheduling,
        "speed_child_custom_all_reduce_requested": (
            custom_all_reduce_provenance["custom_all_reduce_requested"]
        ),
        "speed_child_custom_all_reduce_effective": (
            custom_all_reduce_provenance["custom_all_reduce_effective"]
        ),
        "speed_child_custom_all_reduce_effective_reason": (
            custom_all_reduce_provenance[
                "custom_all_reduce_effective_reason"
            ]
        ),
        **speed_child_custom_all_reduce_runtime,
        "speed_child_custom_all_reduce_runtime_gate_passed": (
            custom_all_reduce_runtime_gate_passed
        ),
        "gate_passed": production_gate_passed,
        "production_gate_passed": production_gate_passed,
        "output_length_gate": output_length_gate,
        "output_length_gate_passed": bool(output_length_gate["passed"]),
        "semantic_output_health": semantic_output_health,
        "prefill_steady_profile_count": int(len(steady_records)),
        "dense_fa3_proof": dense_fa3_proof or {},
        "fa3_so_sha256": fa3_so_sha256,
        "command": result.command,
        "returncode": int(result.returncode),
        "timed_out": bool(result.timed_out),
        "stdout_tail": _tail(result.stdout),
        "stderr_tail": _tail(result.stderr),
        "arena_reserved_bytes": _as_int(
            _metric_or_route("arena_reserved_bytes", -1),
            -1,
        ),
        "arena_peak_bytes": _as_int(_metric_or_route("arena_peak_bytes", -1), -1),
        "arena_bucket_bytes": _as_int(
            _metric_or_route("arena_bucket_bytes", -1),
            -1,
        ),
        "arena_bucket_count": _as_int(
            _metric_or_route("arena_bucket_count", -1),
            -1,
        ),
        "arena_largest_bucket_bytes": _as_int(
            _metric_or_route("arena_largest_bucket_bytes", -1),
            -1,
        ),
        "arena_expansion_bytes": _as_int(
            _metric_or_route("arena_expansion_bytes", -1),
            -1,
        ),
        "arena_budget_exceeded": (
            bool(arena_budget_exceeded_raw)
            if arena_budget_exceeded_raw is not None
            else None
        ),
        "arena_prepare_miss_count": arena_prepare_miss_count,
        "arena_bind_status": arena_bind_status,
        "arena_reservation_status": arena_reservation_status,
        "capture_layout_hot_path_alloc_count": _as_int(
            _metric_or_route("capture_layout_hot_path_alloc_count", -1),
            -1,
        ),
        "capture_layout_new_count": _as_int(
            _metric_or_route("capture_layout_new_count", -1),
            -1,
        ),
        "hot_path_d2h_count": _as_int(
            _metric_or_route("hot_path_d2h_count", -1),
            -1,
        ),
        "hot_path_cuda_sync_count": _as_int(
            _metric_or_route("hot_path_cuda_sync_count", -1),
            -1,
        ),
        "tail_path_item_cpu_count": _as_int(
            _metric_or_route("tail_path_item_cpu_count", -1),
            -1,
        ),
        "arena_ready_before_tail": (
            bool(_metric_or_route("arena_ready_before_tail"))
            if _metric_or_route("arena_ready_before_tail") is not None
            else None
        ),
        "arena_prepare_wait_us": _as_float(
            _metric_or_route("arena_prepare_wait_us", -1.0),
            -1.0,
        ),
        "prefill_global_meta_build_us": _as_float(
            _metric_or_route("prefill_global_meta_build_us", -1.0),
            -1.0,
        ),
        "prefill_global_meta_capture_layout_us": _as_float(
            _metric_or_route("prefill_global_meta_capture_layout_us", -1.0),
            -1.0,
        ),
        "prefill_global_meta_buffer_prepare_us": _as_float(
            _metric_or_route("prefill_global_meta_buffer_prepare_us", -1.0),
            -1.0,
        ),
    }
    payload.update(_boundary_diagnostics_payload(metrics))
    payload.update(_falsification_steady_summary(refresh_profile))
    payload["deadline_v2_attribution"] = deadline_v2_attribution
    if run_provenance is not None:
        run_provenance.update(
            {
                "speed_child_custom_all_reduce_requested": (
                    custom_all_reduce_provenance[
                        "custom_all_reduce_requested"
                    ]
                ),
                "speed_child_custom_all_reduce_effective": (
                    custom_all_reduce_provenance[
                        "custom_all_reduce_effective"
                    ]
                ),
                "speed_child_custom_all_reduce_effective_reason": (
                    custom_all_reduce_provenance[
                        "custom_all_reduce_effective_reason"
                    ]
                ),
                **speed_child_custom_all_reduce_runtime,
                "speed_child_custom_all_reduce_runtime_gate_passed": (
                    custom_all_reduce_runtime_gate_passed
                ),
                **speed_child_engine_scheduling,
            }
        )
        payload["run_provenance"] = run_provenance
    if speed_env is not None:
        speed_config = _gate_d_config(
            args,
            mode=mode,
            env=speed_env,
            fa3_so_sha256=fa3_so_sha256,
        )
        payload["speed_trace_profile_env"] = _gate_d_trace_profile_env(speed_env)
        payload["speed_trace_profile_env_empty"] = _gate_d_trace_profile_env_empty(
            speed_env
        )
        payload["speed_config_digest"] = build_config_digest(speed_config)
        payload["speed_pairing_digest"] = build_pairing_digest(speed_config)
    if diagnostic_env is not None:
        diagnostic_config = _gate_d_config(
            args,
            mode=mode,
            env=diagnostic_env,
            fa3_so_sha256=fa3_so_sha256,
        )
        payload["diagnostic_trace_profile_env"] = _gate_d_trace_profile_env(
            diagnostic_env
        )
        payload["diagnostic_config_digest"] = build_config_digest(diagnostic_config)
        payload["diagnostic_pairing_digest"] = build_pairing_digest(diagnostic_config)
        if "speed_pairing_digest" in payload:
            payload["speed_diagnostic_pairing_match"] = bool(
                payload["speed_pairing_digest"] == payload["diagnostic_pairing_digest"]
            )
    payload["speed_diagnostic_pairing_required"] = diagnostic_env is not None
    if (
        diagnostic_env is not None
        and payload.get("speed_diagnostic_pairing_match") is not True
    ):
        payload["gate_passed"] = False
        payload["production_gate_passed"] = False
    if diag_result is not None:
        payload["diagnostic_command"] = diag_result.command
        payload["diagnostic_returncode"] = int(diag_result.returncode)
        payload["diagnostic_timed_out"] = bool(diag_result.timed_out)
        payload["diagnostic_decode_metrics_path"] = (
            str(diag_metrics_path) if diag_metrics_path else ""
        )
        payload["diagnostic_outputs_path"] = (
            str(diag_outputs_path) if diag_outputs_path else ""
        )
        payload["diagnostic_stdout_tail"] = _tail(diag_result.stdout)
        payload["diagnostic_stderr_tail"] = _tail(diag_result.stderr)
    payload["speed_child_fatal_error_detected"] = bool(speed_child_fatal_error)
    payload["diagnostic_child_fatal_error_detected"] = bool(
        diagnostic_child_fatal_error
    )
    return payload


def _command_result_failure_reasons(
    result: Phase1CommandResult,
    *,
    lifecycle_green: bool = True,
) -> list[str]:
    reasons: list[str] = []
    if result.returncode != 0:
        reasons.append(f"returncode_nonzero:{result.returncode}")
    if result.timed_out:
        reasons.append("timed_out")
    if _command_output_has_fatal_error(result):
        reasons.append("fatal_child_output")
    if not lifecycle_green:
        reasons.append("arm_lifecycle_not_green")
    return reasons


def _write_gate_d_child_failure_artifacts(
    *,
    output_path: Path,
    summary_output: str,
    payload: dict[str, Any],
    result: Phase1CommandResult,
    failure_stage: str,
    failure_reasons: list[str],
    downstream_arms_skipped: list[str],
) -> dict[str, Any]:
    """Persist the failed child before any downstream artifact postflight."""
    stdout_path = output_path.with_name(
        f"{output_path.stem}_{failure_stage}_stdout.log"
    )
    stderr_path = output_path.with_name(
        f"{output_path.stem}_{failure_stage}_stderr.log"
    )
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")
    payload.update(
        {
            "gate_passed": False,
            "production_gate_passed": False,
            "failure_stage": failure_stage,
            "failure_reasons": list(failure_reasons),
            "downstream_arms_skipped": list(downstream_arms_skipped),
            "failure_child_session_id": result.child_session_id,
            "failure_child_stdout_path": str(stdout_path.resolve(strict=False)),
            "failure_child_stderr_path": str(stderr_path.resolve(strict=False)),
            "failure_child_stdout_bytes": len(result.stdout.encode("utf-8")),
            "failure_child_stderr_bytes": len(result.stderr.encode("utf-8")),
            "failure_child_streams_persisted": True,
        }
    )
    serialized = json.dumps(payload, ensure_ascii=True, indent=2, allow_nan=False)
    output_path.write_text(serialized, encoding="utf-8")
    if summary_output:
        summary_path = Path(summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(serialized, encoding="utf-8")
    return payload


def _run_gate_d_mode(args: argparse.Namespace) -> int:
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = _default_gate_d_metrics_path(output_path)
    outputs_path = _default_gate_d_outputs_path(output_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    outputs_path.parent.mkdir(parents=True, exist_ok=True)
    # Each arm owns these generated artifacts.  Removing an earlier run's copy
    # before launch prevents a failed child from being diagnosed with stale
    # metrics or outputs under a reused tag.
    metrics_path.unlink(missing_ok=True)
    outputs_path.unlink(missing_ok=True)
    fa3_so_sha256 = _file_sha256(_resolve_gate_d_backend_artifact(args))

    if args.mode == "dense":
        command = _build_gate_d_dense_command(
            args,
            metrics_path=metrics_path,
            outputs_path=outputs_path,
        )
        speed_env = _build_gate_d_dense_env(args)
        gpu_before = _gpu_snapshot("pre", cuda_visible_devices=str(args.cuda_visible_devices))
        result = _run_command(
            command,
            env=speed_env,
            timeout_s=int(args.timeout_s),
        )
        dense_route_trace_path = _default_gate_d_dense_route_path(output_path)
        dense_route_trace_path.write_text("", encoding="utf-8")
        diag_metrics_path = _default_gate_d_metrics_path(output_path, suffix="_diag")
        diag_outputs_path = _default_gate_d_outputs_path(output_path, suffix="_diag")
        diag_command = _build_gate_d_dense_command(
            args,
            metrics_path=diag_metrics_path,
            outputs_path=diag_outputs_path,
            collect_cudagraph_runtime_proof=True,
        )
        diagnostic_env = _build_gate_d_dense_env(
            args,
            route_trace_path=dense_route_trace_path,
        )
        diag_result = _run_command(
            diag_command,
            env=diagnostic_env,
            timeout_s=int(args.timeout_s),
        )
        gpu_after = _gpu_snapshot("post", cuda_visible_devices=str(args.cuda_visible_devices))
        dense_fa3_proof = _dense_fa3_route_proof(
            _read_trace_events(dense_route_trace_path),
            env=diagnostic_env,
            fa3_so_sha256=fa3_so_sha256,
        )
        metrics = _read_json(metrics_path)
        payload = _gate_d_payload(
            args,
            mode="dense",
            result=result,
            metrics_path=metrics_path,
            outputs_path=outputs_path,
            metrics=metrics,
            route_trace_path=dense_route_trace_path,
            route_proof={
                "passed": bool(dense_fa3_proof.get("passed", False)),
                "reasons": list(dense_fa3_proof.get("reasons", [])),
            },
            dense_fa3_proof=dense_fa3_proof,
            speed_env=speed_env,
            diagnostic_env=diagnostic_env,
            diag_result=diag_result,
            diag_metrics_path=diag_metrics_path,
            diag_outputs_path=diag_outputs_path,
            fa3_so_sha256=fa3_so_sha256,
            run_provenance=_run_provenance_payload(
                args,
                env=speed_env,
                fa3_so_sha256=fa3_so_sha256,
                gpu_before=gpu_before,
                gpu_after=gpu_after,
            ),
        )
        _apply_scheduler_graph_contract_payload(
            payload,
            args,
            speed_metrics=metrics,
            diagnostic_metrics=_read_json(diag_metrics_path),
        )
        output_path.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        if args.summary_output:
            summary_path = Path(args.summary_output)
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(
                json.dumps(payload, ensure_ascii=True, indent=2, allow_nan=False),
                encoding="utf-8",
            )
        ok = (
            result.returncode == 0
            and not result.timed_out
            and diag_result.returncode == 0
            and not diag_result.timed_out
            and not _command_output_has_fatal_error(result)
            and not _command_output_has_fatal_error(diag_result)
            and bool(dense_fa3_proof.get("passed", False))
            and bool(payload.get("production_gate_passed", False))
        )
        return 0 if ok else GATE_FAILURE_EXIT_CODE

    speed_command = _build_sparse_speed_command(
        args,
        metrics_path=metrics_path,
        outputs_path=outputs_path,
    )
    one_shot_timeline_path = (
        Path(args.one_shot_timeline_output)
        if args.one_shot_timeline_output
        else None
    )
    if one_shot_timeline_path is not None:
        one_shot_timeline_path.parent.mkdir(parents=True, exist_ok=True)
        one_shot_timeline_path.write_text("", encoding="utf-8")
    # Timed children are observer-free. Full-form route proof is collected only
    # by the later diagnostic child. The legacy verdict-only mode keeps its
    # JSONL trace for diagnostics but cannot satisfy the independent counter
    # proof and therefore cannot produce a production-green speed verdict.
    speed_route_trace_path = (
        _default_gate_d_speed_route_path(output_path)
        if bool(args.verdict_only)
        else None
    )
    if speed_route_trace_path is not None:
        speed_route_trace_path.parent.mkdir(parents=True, exist_ok=True)
        speed_route_trace_path.write_text("", encoding="utf-8")
    full_cudagraph_hook_profile_path = (
        Path(args.full_cudagraph_hook_profile_output)
        if args.full_cudagraph_hook_profile_output
        else _default_full_cudagraph_hook_profile_path(output_path)
    )
    speed_env = _build_gate_d_sparse_env(
        args,
        route_trace_path=speed_route_trace_path,
        one_shot_timeline_path=one_shot_timeline_path,
    )
    if args.verdict_only:
        # [VERDICT-ONLY 2026-07-11] The diagnostic child is skipped, so the
        # speed child carries the hook profile (second source of
        # async_producer_writer_count for the producer gate). Truncate
        # before spawn so a stale file cannot leak into the readers.
        full_cudagraph_hook_profile_path.parent.mkdir(parents=True, exist_ok=True)
        full_cudagraph_hook_profile_path.write_text("", encoding="utf-8")
        speed_env[FULL_CUDAGRAPH_HOOK_PROFILE_ENV_KEY] = str(
            full_cudagraph_hook_profile_path
        )
    gpu_before = _gpu_snapshot("pre", cuda_visible_devices=str(args.cuda_visible_devices))
    tp8_lifecycle_state: dict[str, Any] | None = None
    selector_prewarm_result: Phase1CommandResult | None = None
    selector_prewarm_green = True
    if _tp8_exact_pair_required(args):
        # Establish a stable idle owner boundary before the setup process can
        # initialize CUDA or spawn extension/compiler workers.
        tp8_lifecycle_state = _start_tp8_exact_lifecycle(
            args,
            output_path=output_path,
        )
        selector_prewarm_green = not tp8_lifecycle_state["baseline_reasons"]
        if selector_prewarm_green:
            prewarm_env, prewarm_token = _tp8_exact_arm_env(
                speed_env,
                tp8_lifecycle_state,
                "selector_prewarm",
            )
            selector_prewarm_result = _prewarm_gt1_selector_extensions(
                args,
                env=prewarm_env,
            )
            if selector_prewarm_result is None:
                selector_prewarm_green = False
                tp8_lifecycle_state["gate_reasons"].append(
                    "selector_prewarm:health:not_launched"
                )
            else:
                # Teardown is mandatory even when prewarm itself fails.
                selector_prewarm_green = _finish_tp8_exact_arm(
                    tp8_lifecycle_state,
                    output_path=output_path,
                    arm_tag="selector_prewarm",
                    result=selector_prewarm_result,
                    arm_token=prewarm_token,
                    setup=True,
                )
                selector_prewarm_result = _redact_tp8_arm_token_result(
                    selector_prewarm_result,
                    prewarm_token,
                )
    else:
        selector_prewarm_result = _prewarm_gt1_selector_extensions(
            args,
            env=speed_env,
        )

    selector_prewarm_failed = bool(
        selector_prewarm_result is not None
        and (
            selector_prewarm_result.returncode != 0
            or bool(selector_prewarm_result.timed_out)
        )
    )
    if selector_prewarm_failed or not selector_prewarm_green:
        failure_result = selector_prewarm_result or Phase1CommandResult(
            command=[],
            returncode=GATE_FAILURE_EXIT_CODE,
            stdout="",
            stderr="exact TP8 idle baseline rejected before selector prewarm",
            timed_out=False,
        )
        gpu_after = _gpu_snapshot(
            "post",
            cuda_visible_devices=str(args.cuda_visible_devices),
        )
        payload = _gate_d_payload(
            args,
            mode="sparse",
            result=failure_result,
            metrics_path=metrics_path,
            outputs_path=outputs_path,
            metrics={},
            route_trace_path=None,
            route_summary={},
            producer_route_summary={},
            route_proof={
                "passed": False,
                "reasons": ["selector_extension_prewarm_failed"],
            },
            speed_env=speed_env,
            fa3_so_sha256=fa3_so_sha256,
            output_records=[],
            run_provenance=_run_provenance_payload(
                args,
                env=speed_env,
                fa3_so_sha256=fa3_so_sha256,
                gpu_before=gpu_before,
                gpu_after=gpu_after,
            ),
        )
        _apply_selector_extension_prewarm_payload(payload, selector_prewarm_result)
        _apply_tp8_exact_lifecycle_payload(payload, tp8_lifecycle_state)
        output_path.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        if args.summary_output:
            summary_path = Path(args.summary_output)
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(
                json.dumps(payload, ensure_ascii=True, indent=2, allow_nan=False),
                encoding="utf-8",
            )
        return GATE_FAILURE_EXIT_CODE
    speed_arm_env = speed_env
    speed_arm_token = ""
    if tp8_lifecycle_state is not None:
        speed_arm_env, speed_arm_token = _tp8_exact_arm_env(
            speed_env,
            tp8_lifecycle_state,
            "sparse_speed",
        )
    speed_result = _run_command(
        speed_command,
        env=speed_arm_env,
        timeout_s=int(args.timeout_s),
    )
    speed_arm_green = True
    if tp8_lifecycle_state is not None:
        speed_arm_green = _finish_tp8_exact_arm(
            tp8_lifecycle_state,
            output_path=output_path,
            arm_tag="sparse_speed",
            result=speed_result,
            arm_token=speed_arm_token,
        )
        speed_result = _redact_tp8_arm_token_result(
            speed_result,
            speed_arm_token,
        )
    speed_child_failure_reasons = _command_result_failure_reasons(
        speed_result,
        lifecycle_green=speed_arm_green,
    )
    if speed_child_failure_reasons:
        # A failed speed arm is a hard lifecycle boundary.  Persist its exact
        # streams and a fail-closed summary before launching dense/diagnostic
        # children or reading route/metrics artifacts that the child may never
        # have produced.
        gpu_after = _gpu_snapshot(
            "post",
            cuda_visible_devices=str(args.cuda_visible_devices),
        )
        payload = _gate_d_payload(
            args,
            mode="sparse",
            result=speed_result,
            metrics_path=metrics_path,
            outputs_path=outputs_path,
            metrics={},
            route_trace_path=None,
            route_summary={},
            producer_route_summary={},
            route_proof={
                "passed": False,
                "reasons": ["speed_child_failed_before_route_postflight"],
            },
            speed_env=speed_env,
            fa3_so_sha256=fa3_so_sha256,
            output_records=[],
            run_provenance=_run_provenance_payload(
                args,
                env=speed_env,
                fa3_so_sha256=fa3_so_sha256,
                gpu_before=gpu_before,
                gpu_after=gpu_after,
            ),
        )
        payload["diagnostic_skipped_due_to_upstream_failure"] = True
        payload["reference_scope"] = "not_run_after_sparse_speed_child_failure"
        if bool(args.outputs_include_text):
            payload["reference_gate_passed"] = False
            payload["reference_gate_reasons"] = [
                "reference_not_run_after_sparse_speed_child_failure"
            ]
        _apply_selector_extension_prewarm_payload(payload, selector_prewarm_result)
        _apply_tp8_exact_lifecycle_payload(payload, tp8_lifecycle_state)
        _apply_scheduler_graph_contract_payload(
            payload,
            args,
            speed_metrics={},
            diagnostic_metrics=None,
        )
        _write_gate_d_child_failure_artifacts(
            output_path=output_path,
            summary_output=args.summary_output,
            payload=payload,
            result=speed_result,
            failure_stage="sparse_speed_child",
            failure_reasons=speed_child_failure_reasons,
            downstream_arms_skipped=[
                "dense_reference",
                "sparse_diagnostic",
                "route_metrics_postflight",
            ],
        )
        return GATE_FAILURE_EXIT_CODE
    # Keep the two observer-free timing arms adjacent under the same parent and
    # GPU lock.  Diagnostic tracing runs only after both throughput samples so
    # it cannot heat, allocate, or mutate state between sparse and dense.
    reference_result: Phase1CommandResult | None = None
    reference_semantic_diffs: list[dict[str, Any]] = []
    reference_semantic_match: bool | None = None
    reference_informational_reasons: list[str] = []
    reference_quality_reasons: list[str] = []
    dense_reference_metrics_path: Path | None = None
    dense_reference_outputs_path: Path | None = None
    sparse_reference_outputs_path: Path | None = None
    dense_reference_env: dict[str, str] | None = None
    dense_arm_green = tp8_lifecycle_state is None
    sparse_output_records = _request_ordered_output_records(outputs_path)
    if (
        bool(args.outputs_include_text)
        and not bool(args.skip_dense_reference)
        and speed_arm_green
    ):
        dense_reference_metrics_path = _default_gate_d_dense_reference_metrics_path(
            output_path
        )
        dense_reference_outputs_path = _default_gate_d_dense_reference_outputs_path(
            output_path
        )
        dense_reference_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        dense_reference_outputs_path.parent.mkdir(parents=True, exist_ok=True)
        dense_reference_env = _build_gate_d_dense_env(args)
        dense_arm_env = dense_reference_env
        dense_arm_token = ""
        if tp8_lifecycle_state is not None:
            dense_arm_env, dense_arm_token = _tp8_exact_arm_env(
                dense_reference_env,
                tp8_lifecycle_state,
                "dense_reference",
            )
        reference_result = _run_command(
            _build_gate_d_dense_command(
                args,
                metrics_path=dense_reference_metrics_path,
                outputs_path=dense_reference_outputs_path,
            ),
            env=dense_arm_env,
            timeout_s=int(args.timeout_s),
        )
        if tp8_lifecycle_state is not None:
            dense_arm_green = _finish_tp8_exact_arm(
                tp8_lifecycle_state,
                output_path=output_path,
                arm_tag="dense_reference",
                result=reference_result,
                arm_token=dense_arm_token,
            )
            reference_result = _redact_tp8_arm_token_result(
                reference_result,
                dense_arm_token,
            )
    route_trace_path = _default_gate_d_route_path(output_path)
    refresh_profile_path = (
        Path(args.refresh_profile_output)
        if args.refresh_profile_output
        else _default_refresh_profile_path(output_path)
    )
    diag_metrics_path = _default_gate_d_metrics_path(output_path, suffix="_diag")
    diag_outputs_path = _default_gate_d_outputs_path(output_path, suffix="_diag")
    route_trace_path.write_text("", encoding="utf-8")
    refresh_profile_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostic_env: dict[str, str] | None = None
    diag_result: Phase1CommandResult | None = None
    selector_pipeline_cpu_profile_path: Path | None = None
    if args.verdict_only:
        # [VERDICT-ONLY 2026-07-11] Skip the diagnostic child. The route
        # trace was already truncated above; additionally truncate the
        # refresh profile this run will NOT regenerate so a stale file from
        # a previous full-form run under the same TAG cannot leak into
        # _read_refresh_profile or the manual counts readout. Manual counts
        # readout file in this mode is ``${TAG}_speed_route.jsonl``.
        refresh_profile_path.write_text("", encoding="utf-8")
    elif tp8_lifecycle_state is None or dense_arm_green:
        full_cudagraph_hook_profile_path.parent.mkdir(parents=True, exist_ok=True)
        full_cudagraph_hook_profile_path.write_text("", encoding="utf-8")
        diag_command = _build_phase2_command(
            args,
            metrics_path=diag_metrics_path,
            refresh_profile_path=refresh_profile_path,
            outputs_path=diag_outputs_path,
        )
        diagnostic_env = _build_gate_d_sparse_env(
            args,
            route_trace_path=route_trace_path,
            one_shot_timeline_path=one_shot_timeline_path,
        )
        diagnostic_arm_env = diagnostic_env
        diagnostic_arm_token = ""
        diagnostic_env[FULL_CUDAGRAPH_HOOK_PROFILE_ENV_KEY] = str(
            full_cudagraph_hook_profile_path
        )
        _copy_refresh_micro_profile_env_for_diagnostic(diagnostic_env)
        _copy_selector_profile_env_for_diagnostic(diagnostic_env)
        # [GATE-D-SPEED-ENV-PURITY 2026-07-10] 自 _build_gate_d_sparse_env 移入的
        # diagnostic-only 回填(speed child 不再泄入 profile env)。
        _copy_mixed_page_kernel_profile_env_for_diagnostic(diagnostic_env)
        _copy_sparse_metadata_profile_env_for_diagnostic(diagnostic_env)
        _copy_torch_profiler_env_for_diagnostic(diagnostic_env)
        _copy_step_profile_env_for_diagnostic(diagnostic_env)
        if args.selector_pipeline_cpu_profile_output:
            diagnostic_env["VLLM_SPARSE_PIPELINE_CPU_PROFILE"] = "1"
            diagnostic_env["VLLM_SPARSE_PIPELINE_CPU_PROFILE_LOG"] = str(
                Path(args.selector_pipeline_cpu_profile_output)
            )
        selector_pipeline_cpu_profile_path = _selector_pipeline_cpu_profile_path(
            diagnostic_env,
            output_path,
        )
        if selector_pipeline_cpu_profile_path is not None:
            selector_pipeline_cpu_profile_path.parent.mkdir(
                parents=True, exist_ok=True
            )
            selector_pipeline_cpu_profile_path.write_text("", encoding="utf-8")
        if tp8_lifecycle_state is not None:
            diagnostic_arm_env, diagnostic_arm_token = _tp8_exact_arm_env(
                diagnostic_env,
                tp8_lifecycle_state,
                "sparse_diagnostic",
            )
        diag_result = _run_command(
            diag_command,
            env=diagnostic_arm_env,
            timeout_s=int(args.timeout_s),
        )
        if tp8_lifecycle_state is not None:
            _finish_tp8_exact_arm(
                tp8_lifecycle_state,
                output_path=output_path,
                arm_tag="sparse_diagnostic",
                result=diag_result,
                arm_token=diagnostic_arm_token,
            )
            diag_result = _redact_tp8_arm_token_result(
                diag_result,
                diagnostic_arm_token,
            )
    if (
        bool(args.outputs_include_text)
        and not bool(args.skip_dense_reference)
        and reference_result is not None
    ):
        assert dense_reference_metrics_path is not None
        assert dense_reference_outputs_path is not None
        sparse_reference_outputs_path = outputs_path
        sparse_output_records = _request_ordered_output_records(
            sparse_reference_outputs_path
        )
        dense_output_records = _request_ordered_output_records(
            dense_reference_outputs_path
        )
        reference_semantic_diffs = _semantic_output_diffs(
            sparse_output_records,
            dense_output_records,
        )
        reference_quality_reasons = _reference_output_quality_reasons(
            sparse_output_records,
            dense_output_records,
        )
        if reference_semantic_diffs:
            reference_semantic_match = all(
                bool(diff.get("semantic_match", False))
                for diff in reference_semantic_diffs
            )
        reference_informational_reasons = _reference_text_informational_reasons(
            reference_semantic_match
        )
    dense_reference_skipped = bool(args.outputs_include_text) and bool(
        args.skip_dense_reference
    )
    gpu_after = _gpu_snapshot("post", cuda_visible_devices=str(args.cuda_visible_devices))
    # [VERDICT-ONLY 2026-07-11] Single-variable proof re-sourcing: in
    # verdict-only mode the route/producer proofs read the speed-child trace
    # (the diagnostic child did not run); default mode keeps the diagnostic
    # trace, byte-for-byte.
    effective_route_trace_path = (
        speed_route_trace_path
        if (args.verdict_only and speed_route_trace_path is not None)
        else route_trace_path
    )
    route_summary = _route_summary(_read_trace_events(effective_route_trace_path))
    speed_route_summary = (
        _route_summary(_read_trace_events(speed_route_trace_path))
        if speed_route_trace_path is not None
        else {}
    )
    measurement_trace_events = _read_measurement_trace_events(
        effective_route_trace_path
    )
    producer_route_summary = _route_summary(measurement_trace_events)
    route_proof_result = validate_shared_route_proof(
        _route_proof_payload(
            route_summary=route_summary,
            env=(diagnostic_env if diagnostic_env is not None else speed_env),
            fa3_so_sha256=fa3_so_sha256,
        )
    )
    route_proof = {
        "passed": bool(route_proof_result.passed),
        "reasons": list(route_proof_result.reasons),
    }
    refresh_profile = _read_refresh_profile(refresh_profile_path)
    full_cudagraph_hook_profile = _read_jsonl_records(full_cudagraph_hook_profile_path)
    (
        measurement_hook_profile,
        hook_profile_measurement_markers_present,
    ) = _measurement_window_records(full_cudagraph_hook_profile)
    timeline_records = (
        _read_one_shot_timeline(one_shot_timeline_path)
        if one_shot_timeline_path is not None
        else []
    )
    timeline_summary = _timeline_budget_summary(timeline_records)
    selector_pipeline_cpu_profile = (
        _read_selector_pipeline_cpu_profile(selector_pipeline_cpu_profile_path)
        if selector_pipeline_cpu_profile_path is not None
        else []
    )
    metrics = _read_json(metrics_path)
    _bind_runtime_request_context_tokens(args, metrics)
    diagnostic_metrics = (
        _read_json(diag_metrics_path) if diag_result is not None else None
    )
    if bool(args.outputs_include_text):
        # Runtime health belongs to the sparse output itself.  A healthy sparse
        # answer may reasonably differ from the independently sampled dense arm.
        metrics["semantic_output_health"] = _sparse_output_content_health(
            sparse_output_records
        )
    if reference_semantic_match is not None:
        metrics["dense_reference_semantic_match"] = bool(reference_semantic_match)
    if reference_quality_reasons:
        metrics["dense_reference_quality_ok"] = False
        metrics["dense_reference_quality_reasons"] = list(reference_quality_reasons)
    measurement_hook_summary = _full_cudagraph_hook_profile_summary(
        measurement_hook_profile
    )
    payload = _gate_d_payload(
        args,
        mode="sparse",
        result=speed_result,
        metrics_path=metrics_path,
        outputs_path=outputs_path,
        metrics=metrics,
        route_counter_metrics=diagnostic_metrics,
        route_trace_path=route_trace_path,
        route_summary=route_summary,
        producer_route_summary=producer_route_summary,
        route_proof=route_proof,
        speed_env=speed_env,
        diagnostic_env=diagnostic_env,
        diag_result=diag_result,
        diag_metrics_path=diag_metrics_path,
        diag_outputs_path=diag_outputs_path,
        refresh_profile_path=refresh_profile_path,
        refresh_profile=refresh_profile,
        hook_profile_summary=measurement_hook_summary,
        selector_pipeline_cpu_profile_path=selector_pipeline_cpu_profile_path,
        selector_pipeline_cpu_profile=selector_pipeline_cpu_profile,
        output_records=_request_ordered_output_records(outputs_path),
        fa3_so_sha256=fa3_so_sha256,
        run_provenance=_run_provenance_payload(
            args,
            env=speed_env,
            fa3_so_sha256=fa3_so_sha256,
            gpu_before=gpu_before,
            gpu_after=gpu_after,
        ),
    )
    payload["full_cudagraph_hook_profile_path"] = str(full_cudagraph_hook_profile_path)
    payload["speed_child_route_trace_path"] = (
        str(speed_route_trace_path) if speed_route_trace_path is not None else ""
    )
    payload["speed_child_route_summary"] = speed_route_summary
    payload["verdict_only"] = bool(args.verdict_only)
    payload["route_summary_source"] = (
        "speed_child_route_trace"
        if args.verdict_only
        else "diagnostic_child_route_trace"
    )
    mixed_chunk_postflight_gate = _mixed_chunk_postflight_gate(
        args,
        measurement_trace_events,
    )
    payload["mixed_chunk_postflight_gate"] = mixed_chunk_postflight_gate
    payload["mixed_chunk_postflight_gate_passed"] = bool(
        mixed_chunk_postflight_gate.get("passed", False)
    )
    if not payload["mixed_chunk_postflight_gate_passed"]:
        payload["gate_passed"] = False
        payload["production_gate_passed"] = False
    payload.update(
        _workload_plan_payload(
            args,
            route_trace_path=effective_route_trace_path,
            measurement_trace_events=measurement_trace_events,
        )
    )
    workload_plan_count_mismatches = _workload_plan_replay_count_mismatches(payload)
    payload["workload_plan_replay_counts_match"] = not workload_plan_count_mismatches
    payload["workload_plan_replay_count_mismatches"] = list(
        workload_plan_count_mismatches
    )
    if workload_plan_count_mismatches:
        payload["gate_passed"] = False
        payload["production_gate_passed"] = False
    _apply_selector_extension_prewarm_payload(payload, selector_prewarm_result)
    payload["full_cudagraph_hook_profile_summary_scope"] = "measurement_window"
    payload["full_cudagraph_hook_profile_measurement_markers_present"] = bool(
        hook_profile_measurement_markers_present
    )
    payload["full_cudagraph_hook_profile_summary"] = measurement_hook_summary
    if one_shot_timeline_path is not None:
        payload["one_shot_timeline_path"] = str(one_shot_timeline_path)
        payload["timeline_summary"] = timeline_summary
    run_provenance_payload = payload.get("run_provenance")
    if isinstance(run_provenance_payload, dict):
        observed_slack = int(
            measurement_hook_summary.get(
                "refresh_stage_producer_work_deadline_slack_steps_min",
                -1,
            )
            or -1
        )
        if (
            observed_slack > 0
            and int(
                run_provenance_payload.get(
                    "refresh_rebuild_max_delay_steps_effective",
                    0,
                )
                or 0
            )
            <= 0
        ):
            run_provenance_payload[
                "refresh_rebuild_max_delay_steps_effective"
            ] = int(observed_slack)
            run_provenance_payload[
                "refresh_rebuild_max_delay_steps_effective_source"
            ] = "hook_profile_observed_slack"
    payload["deadline_v2_attribution"] = _deadline_v2_attribution_summary(
        metrics=metrics,
        refresh_profile=refresh_profile,
        timeline_summary=timeline_summary,
        route_summary=route_summary,
        selector_pipeline_cpu_profile=selector_pipeline_cpu_profile,
        hook_profile_summary=measurement_hook_summary,
    )
    payload["full_cudagraph_hook_profile_full_diagnostic_summary_scope"] = (
        "full_diagnostic_run"
    )
    payload["full_cudagraph_hook_profile_full_diagnostic_summary"] = (
        _full_cudagraph_hook_profile_summary(full_cudagraph_hook_profile)
    )
    if reference_result is not None:
        payload["reference_command"] = reference_result.command
        payload["reference_returncode"] = int(reference_result.returncode)
        payload["reference_timed_out"] = bool(reference_result.timed_out)
        payload["reference_stdout_tail"] = _tail(reference_result.stdout)
        payload["reference_stderr_tail"] = _tail(reference_result.stderr)
        payload["reference_dense_metrics_path"] = (
            str(dense_reference_metrics_path) if dense_reference_metrics_path else ""
        )
        payload["reference_dense_outputs_path"] = (
            str(dense_reference_outputs_path) if dense_reference_outputs_path else ""
        )
        payload["reference_sparse_outputs_path"] = (
            str(sparse_reference_outputs_path) if sparse_reference_outputs_path else ""
        )
        payload["reference_scope"] = (
            "output_health_semantic_diff_informational_and_paired_throughput"
            if _sparse_dense_pair_contract_kind()
            in {"exact_speedup_verdict", "explicit_local_comparison"}
            else "output_health_with_semantic_diff_informational"
        )
        payload["reference_semantic_diffs"] = reference_semantic_diffs
        payload["reference_informational_reasons"] = list(
            reference_informational_reasons
        )
        payload["reference_semantic_comparison_informational"] = True
        payload["reference_quality_reasons"] = reference_quality_reasons
    elif dense_reference_skipped:
        payload["reference_scope"] = "debug_skipped_dense_reference"
        payload["reference_semantic_diffs"] = []
        payload["reference_informational_reasons"] = []
        payload["reference_semantic_comparison_informational"] = True
        payload["reference_quality_reasons"] = []
        payload["skip_dense_reference"] = True
    if bool(args.outputs_include_text):
        reference_gate_reasons = _reference_text_gate_reasons(
            reference_result=reference_result,
            reference_semantic_match=reference_semantic_match,
        )
        reference_gate_reasons.extend(reference_quality_reasons)
        if dense_reference_skipped:
            reference_gate_reasons = ["dense_reference_skipped_for_debug"]
            payload["reference_gate_passed"] = None
            payload["reference_gate_reasons"] = reference_gate_reasons
        else:
            payload["reference_gate_passed"] = not reference_gate_reasons
            payload["reference_gate_reasons"] = reference_gate_reasons
        if reference_gate_reasons and not dense_reference_skipped:
            payload["gate_passed"] = False
            payload["production_gate_passed"] = False
    dense_reference_metrics = (
        _read_json(dense_reference_metrics_path)
        if dense_reference_metrics_path is not None
        else None
    )
    _apply_scheduler_graph_contract_payload(
        payload,
        args,
        speed_metrics=metrics,
        diagnostic_metrics=diagnostic_metrics,
        dense_reference_metrics=dense_reference_metrics,
    )
    _apply_sparse_dense_pair_speedup_payload(
        payload,
        args,
        sparse_metrics=metrics,
        dense_metrics=dense_reference_metrics,
        sparse_env=speed_env,
        dense_env=dense_reference_env,
    )
    _apply_tp8_exact_lifecycle_payload(payload, tp8_lifecycle_state)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    if args.summary_output:
        summary_path = Path(args.summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2, allow_nan=False),
            encoding="utf-8",
        )
    ok = (
        speed_result.returncode == 0
        and not speed_result.timed_out
        # [VERDICT-ONLY 2026-07-11] diag_result is None only in verdict-only
        # mode (diagnostic child skipped); the default form keeps the exact
        # original two-condition check via the or-branch.
        and (
            diag_result is None
            or (diag_result.returncode == 0 and not diag_result.timed_out)
        )
        and route_proof_result.passed
        and bool(payload.get("diagnostic_child_route_proof_passed", False))
        and bool(payload.get("production_gate_passed", False))
        and bool(payload.get("workload_plan_replay_counts_match", True))
    )
    return 0 if ok else GATE_FAILURE_EXIT_CODE


def _record_ts_ns(record: dict[str, Any]) -> int | None:
    try:
        return int(record.get("ts_ns", -1))
    except (TypeError, ValueError):
        return None


def _read_refresh_profile(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    measure_begin_ts_ns: int | None = None
    measure_end_ts_ns: int | None = None
    records: list[dict[str, Any]] = []
    baseline: dict[str, Any] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t", 2)
        if len(parts) != 3 or parts[1] not in {"refresh.flush", "refresh.marker"}:
            continue
        try:
            payload = json.loads(parts[2])
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        if parts[1] == "refresh.marker":
            phase = str(payload.get("phase", "") or "")
            if phase == "measure_begin":
                ts_ns = _record_ts_ns(payload)
                if ts_ns is not None:
                    measure_begin_ts_ns = ts_ns
            elif phase == "measure_end":
                ts_ns = _record_ts_ns(payload)
                if ts_ns is not None:
                    measure_end_ts_ns = ts_ns
            continue
        if parts[1] == "refresh.flush":
            records.append(payload)
    if measure_begin_ts_ns is None or measure_end_ts_ns is None:
        return []
    window_records = [
        record
        for record in records
        if (ts_ns := _record_ts_ns(record)) is not None
        and measure_begin_ts_ns < ts_ns <= measure_end_ts_ns
    ]
    for record in records:
        ts_ns = _record_ts_ns(record)
        if ts_ns is None or ts_ns > measure_begin_ts_ns:
            continue
        if baseline is None or int(ts_ns) > int(_record_ts_ns(baseline) or -1):
            baseline = record
    return _normalize_refresh_profile_window_counters(
        window_records,
        baseline=baseline,
    )


def _is_cumulative_refresh_profile_counter(key: str) -> bool:
    if key.startswith("deadline_") and (
        key.endswith("_count")
        or key.endswith("_cpu_us_total")
        or key.endswith("_gpu_ms_total")
        or key.endswith("_total")
    ):
        return True
    return key in {
        "sentence_trigger_admission_coalesced_total",
        "sentence_trigger_admission_coalesced_interval_pending_total",
        "sentence_trigger_admission_coalesced_inflight_total",
        "sentence_trigger_admission_coalesced_pending_rebuild_total",
        "sentence_trigger_admission_dropped_finished_total",
        "refresh_coalesce_skipped_existing_pending_total",
    }


def _refresh_profile_count_key_for_max_counter(key: str) -> str | None:
    if key.startswith("deadline_") and key.endswith("_cpu_us_max"):
        return key[: -len("_cpu_us_max")] + "_count"
    if key.startswith("deadline_") and key.endswith("_gpu_ms_max"):
        return key[: -len("_gpu_ms_max")] + "_gpu_count"
    return None


def _normalize_refresh_profile_window_counters(
    records: list[dict[str, Any]],
    *,
    baseline: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not records or baseline is None:
        return records
    normalized: list[dict[str, Any]] = []
    for record in records:
        adjusted = dict(record)
        for key, value in record.items():
            if not _is_cumulative_refresh_profile_counter(str(key)):
                continue
            if key not in baseline:
                continue
            base_value = baseline.get(key)
            if isinstance(value, int) and isinstance(base_value, int):
                adjusted[key] = max(0, int(value) - int(base_value))
            else:
                numeric = _as_float(value, -1.0)
                base_numeric = _as_float(base_value, -1.0)
                if numeric >= 0.0 and base_numeric >= 0.0:
                    adjusted[key] = max(0.0, numeric - base_numeric)
        for key in tuple(adjusted.keys()):
            count_key = _refresh_profile_count_key_for_max_counter(str(key))
            if count_key is None:
                continue
            if int(adjusted.get(count_key, -1) or 0) == 0:
                adjusted[key] = 0.0
        normalized.append(adjusted)
    return normalized


def _read_measurement_trace_events(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    in_measurement = False
    saw_begin = False
    saw_end = False
    source_events = _read_trace_events(path)
    for event in source_events:
        if event.get("event") == "bench_measure_marker":
            phase = str(event.get("phase", "") or "")
            if phase == "measure_begin":
                records = []
                in_measurement = True
                saw_begin = True
                saw_end = False
            elif phase == "measure_end" and in_measurement:
                in_measurement = False
                saw_end = True
                break
            continue
        if in_measurement:
            records.append(event)
    # [ROUTE-TRACE-PID-NORM] 保留坏行数载体:测量窗切片内的撕裂丢行数
    # ≤ 全文件坏行数,作为 _route_summary pid 归一的容忍上界(TP1 单
    # pid 直通,行为不变)。
    bad_line_count = int(getattr(source_events, "bad_line_count", 0) or 0)
    if not (saw_begin and saw_end):
        records = []
    return _RouteTraceEvents(
        records,
        bad_line_count=bad_line_count,
        source_path=f"{path}#measurement_window",
    )


def _read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _measurement_window_records(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], bool]:
    window_records: list[dict[str, Any]] = []
    in_measurement = False
    saw_begin = False
    saw_end = False
    for record in records:
        if record.get("event") == "bench_measure_marker":
            phase = str(record.get("phase", "") or "")
            if phase == "measure_begin":
                window_records = []
                in_measurement = True
                saw_begin = True
                saw_end = False
            elif phase == "measure_end" and in_measurement:
                in_measurement = False
                saw_end = True
                break
            continue
        if in_measurement:
            window_records.append(record)
    if not (saw_begin and saw_end):
        return [], False
    return window_records, True


def _sum_int(records: list[dict[str, Any]], key: str) -> int:
    total = 0
    for record in records:
        try:
            total += int(record.get(key, 0) or 0)
        except (TypeError, ValueError):
            continue
    return total


def _max_int_from_records(
    records: list[dict[str, Any]],
    key: str,
    *,
    default: int = 0,
) -> int:
    values: list[int] = []
    for record in records:
        if key not in record:
            continue
        try:
            values.append(int(record.get(key, default) or 0))
        except (TypeError, ValueError):
            continue
    return max(values) if values else int(default)


def _nonnegative_int_values_from_records(
    records: list[dict[str, Any]],
    key: str,
) -> list[int]:
    values: list[int] = []
    for record in records:
        if key not in record:
            continue
        try:
            value = int(record.get(key, -1))
        except (TypeError, ValueError):
            continue
        if value >= 0:
            values.append(value)
    return values


def _int_list_values_from_records(
    records: list[dict[str, Any]],
    key: str,
) -> list[int]:
    values: list[int] = []
    for record in records:
        raw = record.get(key)
        if not isinstance(raw, list):
            continue
        for item in raw:
            value = _as_int(item, -1)
            if value >= 0:
                values.append(value)
    return sorted(set(values))


def _min_nonnegative_int_from_records(
    records: list[dict[str, Any]],
    key: str,
    *,
    default: int = -1,
) -> int:
    values = _nonnegative_int_values_from_records(records, key)
    return min(values) if values else int(default)


def _producer_decode_steps_from_records(records: list[dict[str, Any]]) -> list[int]:
    values: list[int] = []
    for record in records:
        step_min = _as_int(record.get("producer_work_decode_step_min"), -1)
        step_max = _as_int(record.get("producer_work_decode_step_max"), -1)
        if step_min >= 0:
            values.append(step_min)
        if step_max >= 0:
            values.append(step_max)
    return sorted(set(values))


def _count_string_from_records(
    records: list[dict[str, Any]],
    key: str,
    value: str,
) -> int:
    expected = str(value)
    return sum(1 for record in records if str(record.get(key, "") or "") == expected)


def _max_ms(records: list[dict[str, Any]], key: str, *, only_prefill: bool) -> float:
    values: list[float] = []
    for record in records:
        try:
            prefill_payloads = int(record.get("prefill_payloads", 0) or 0)
        except (TypeError, ValueError):
            prefill_payloads = 0
        if only_prefill and prefill_payloads <= 0:
            continue
        try:
            value = float(record.get(key, -1.0))
        except (TypeError, ValueError):
            value = -1.0
        if value >= 0.0:
            values.append(value)
    return max(values) if values else -1.0


def _max_us_from_records(records: list[dict[str, Any]], key: str) -> float:
    values = [_as_float(record.get(key), -1.0) for record in records]
    values = [value for value in values if value >= 0.0]
    return max(values) if values else -1.0


def _profile_float_values(records: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for record in records:
        value = record.get(key)
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if numeric >= 0.0:
            values.append(numeric)
    return values


def _prefill_profile_records(
    refresh_profile: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        record
        for record in refresh_profile
        if int(record.get("prefill_rebuild_runs") or 0) > 0
        or record.get("prefill_rebuild_gpu_ms") is not None
        or record.get("prefill_gpu_ms") is not None
    ]


def _steady_prefill_profile_records(
    refresh_profile: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    records = _prefill_profile_records(refresh_profile)
    return records[1:] if len(records) > 1 else []


def _refresh_rebuild_enqueued_records(
    refresh_profile: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        record
        for record in refresh_profile
        if int(record.get("refresh_rebuild_enqueued_count") or 0) > 0
    ]


def _refresh_work_item_records(
    refresh_profile: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    enqueued_records = _refresh_rebuild_enqueued_records(refresh_profile)
    if enqueued_records:
        return enqueued_records
    producer_work_records = [
        record
        for record in refresh_profile
        if _as_int(record.get("producer_work_target_layer_start"), -1) >= 0
        or _as_int(record.get("producer_work_ready_epoch"), -1) >= 0
    ]
    if producer_work_records:
        return producer_work_records
    return [
        record
        for record in refresh_profile
        if int(record.get("refresh_payloads") or 0) > 0
    ]


def _avg(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else -1.0


def _full_cudagraph_hook_profile_summary(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    wrapper_records = [
        record
        for record in records
        if str(record.get("event", "") or "")
        == "mixed_page_full_cudagraph_wrapper_call"
    ]
    model_forward_refresh_records = [
        record
        for record in records
        if str(record.get("event", "") or "")
        == "mixed_page_full_cudagraph_model_forward_refresh"
    ]
    # New runs have one generation-wide producer record at model-forward
    # completion.  Keep old wrapper-owned profiles readable, but never mix the
    # two ownership models in one summary: once owner records are present they
    # are the sole source of producer counts and stage timings.
    producer_records = (
        model_forward_refresh_records
        if model_forward_refresh_records
        else wrapper_records
    )
    post_replay_records = [
        record
        for record in producer_records
        if _as_int(record.get("post_replay_refresh_payloads"), 0) > 0
    ]
    reason_counts: dict[str, int] = {}
    hook_counts: dict[str, int] = {}
    for record in wrapper_records:
        reason = str(record.get("reason", "") or "")
        hook = str(record.get("hook", "") or "")
        if reason:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        if hook:
            hook_counts[hook] = hook_counts.get(hook, 0) + 1
    def _stage_value(record: dict[str, Any], key: str) -> float:
        stage = record.get("refresh_stage_profile")
        if not isinstance(stage, dict):
            return -1.0
        return _as_float(stage.get(key), -1.0)

    def _stage_values(key: str) -> list[float]:
        return [
            _stage_value(record, key)
            for record in post_replay_records
            if _stage_value(record, key) >= 0.0
        ]

    refresh_us_values = [
        _as_float(record.get("refresh_us"), -1.0)
        for record in post_replay_records
        if _as_float(record.get("refresh_us"), -1.0) >= 0.0
    ]
    post_call_us_values = _profile_float_values(wrapper_records, "post_call_us")
    pre_consume_drain_us_values = [
        _as_float(record.get("pre_consume_pending_rebuild_drain_us"), -1.0)
        for record in wrapper_records
        if _as_float(record.get("pre_consume_pending_rebuild_drain_us"), -1.0)
        >= 0.0
    ]
    pre_original_us_values = _profile_float_values(wrapper_records, "pre_original_us")
    pre_forward_context_us_values = _profile_float_values(
        wrapper_records, "pre_forward_context_us"
    )
    pre_graph_lookup_us_values = _profile_float_values(
        wrapper_records, "pre_graph_lookup_us"
    )
    pre_prebound_mark_us_values = _profile_float_values(
        wrapper_records, "pre_prebound_mark_us"
    )
    pre_prebound_state_lookup_us_values = _profile_float_values(
        wrapper_records, "pre_prebound_state_lookup_us"
    )
    pre_ready_event_wait_us_values = _profile_float_values(
        wrapper_records, "pre_ready_event_wait_us"
    )
    pre_prebound_stats_us_values = _profile_float_values(
        wrapper_records, "pre_prebound_stats_us"
    )
    pre_other_us_values = _profile_float_values(wrapper_records, "pre_other_us")
    stage_total_us_values = _stage_values("total_us")
    stage_flush_us_values = _stage_values("flush_us")
    stage_deferred_enqueue_us_values = _stage_values(
        "deferred_pending_rebuild_enqueue_us"
    )
    stage_pending_group_total_us_values = _stage_values("pending_group_total_us")
    stage_pending_group_clear_us_values = _stage_values("pending_group_clear_us")
    stage_pending_group_carrier_prepare_us_values = _stage_values(
        "pending_group_carrier_prepare_us"
    )
    stage_pending_group_bootstrap_slots_list_us_values = _stage_values(
        "pending_group_bootstrap_slots_list_us"
    )
    stage_pending_group_enqueue_pending_us_values = _stage_values(
        "pending_group_enqueue_pending_us"
    )
    stage_pending_group_enqueue_prepare_us_values = _stage_values(
        "pending_group_enqueue_prepare_us"
    )
    stage_pending_group_enqueue_work_item_us_values = _stage_values(
        "pending_group_enqueue_work_item_us"
    )
    stage_pending_group_enqueue_register_us_values = _stage_values(
        "pending_group_enqueue_register_us"
    )
    stage_pending_group_enqueue_construct_us_values = _stage_values(
        "pending_group_enqueue_construct_us"
    )
    stage_pending_group_enqueue_coalesce_us_values = _stage_values(
        "pending_group_enqueue_coalesce_us"
    )
    stage_pending_group_enqueue_record_stream_us_values = _stage_values(
        "pending_group_enqueue_record_stream_us"
    )
    stage_pending_group_enqueue_mark_us_values = _stage_values(
        "pending_group_enqueue_mark_us"
    )
    stage_pending_group_enqueue_record_async_work_us_values = _stage_values(
        "pending_group_enqueue_record_async_work_us"
    )
    stage_pending_group_enqueue_compact_us_values = _stage_values(
        "pending_group_enqueue_compact_us"
    )
    stage_pending_group_enqueue_queue_insert_us_values = _stage_values(
        "pending_group_enqueue_queue_insert_us"
    )
    stage_progressive_selector_precompute_us_values = _stage_values(
        "progressive_selector_precompute_us"
    )
    stage_flush_group_total_us_values = _stage_values("flush_group_total_us")
    stage_flush_group_setup_us_values = _stage_values("flush_group_setup_us")
    stage_flush_group_mark_submitted_us_values = _stage_values(
        "flush_group_mark_submitted_us"
    )
    stage_flush_group_record_stream_us_values = _stage_values(
        "flush_group_record_stream_us"
    )
    stage_flush_group_selector_us_values = _stage_values("flush_group_selector_us")
    stage_flush_group_writer_us_values = _stage_values("flush_group_writer_us")
    stage_flush_group_gather_event_us_values = _stage_values(
        "flush_group_gather_event_us"
    )
    stage_flush_group_async_event_us_values = _stage_values(
        "flush_group_async_event_us"
    )

    def _stage_sum_int(key: str) -> int:
        total = 0
        for record in post_replay_records:
            stage = record.get("refresh_stage_profile")
            if isinstance(stage, dict):
                total += max(0, _as_int(stage.get(key), 0))
        return int(total)

    def _stage_int_values(key: str) -> list[int]:
        values: list[int] = []
        for record in post_replay_records:
            stage = record.get("refresh_stage_profile")
            if not isinstance(stage, dict):
                continue
            value = _as_int(stage.get(key), -1)
            if value >= 0:
                values.append(value)
        return values

    def _deferred_drain_profile_value(record: dict[str, Any], key: str) -> float:
        profile = record.get("deferred_replay_refresh_drain_profile")
        if not isinstance(profile, dict):
            return -1.0
        return _as_float(profile.get(key), -1.0)

    def _deferred_drain_profile_values(key: str) -> list[float]:
        return [
            _deferred_drain_profile_value(record, key)
            for record in wrapper_records
            if _deferred_drain_profile_value(record, key) >= 0.0
        ]

    def _deferred_drain_profile_sum_int(key: str) -> int:
        total = 0
        for record in wrapper_records:
            profile = record.get("deferred_replay_refresh_drain_profile")
            if isinstance(profile, dict):
                total += max(0, _as_int(profile.get(key), 0))
        return int(total)

    deferred_drain_total_us_values = _deferred_drain_profile_values("total_us")
    deferred_drain_pending_group_total_us_values = _deferred_drain_profile_values(
        "pending_group_total_us"
    )
    deferred_drain_pending_group_clear_us_values = _deferred_drain_profile_values(
        "pending_group_clear_us"
    )
    deferred_drain_pending_group_carrier_prepare_us_values = (
        _deferred_drain_profile_values("pending_group_carrier_prepare_us")
    )
    deferred_drain_pending_group_bootstrap_slots_list_us_values = (
        _deferred_drain_profile_values("pending_group_bootstrap_slots_list_us")
    )
    deferred_drain_pending_group_enqueue_pending_us_values = (
        _deferred_drain_profile_values("pending_group_enqueue_pending_us")
    )

    producer_work_deadline_slack_steps = _stage_int_values(
        "producer_work_deadline_slack_steps"
    )
    producer_work_decode_step_min = _stage_int_values("producer_work_decode_step_min")
    producer_work_decode_step_max = _stage_int_values("producer_work_decode_step_max")
    producer_work_decode_steps_observed = sorted(
        {
            int(step)
            for step_pair in zip(
                producer_work_decode_step_min,
                producer_work_decode_step_max,
            )
            for step in range(
                int(step_pair[0]),
                int(step_pair[1]) + 1,
            )
            if int(step_pair[0]) >= 0 and int(step_pair[1]) >= int(step_pair[0])
        }
    )

    return {
        "record_count": int(len(records)),
        "wrapper_call_count": int(len(wrapper_records)),
        "model_forward_refresh_call_count": int(
            len(model_forward_refresh_records)
        ),
        "post_replay_refresh_call_count": int(len(post_replay_records)),
        "post_replay_refresh_payloads_total": int(
            sum(
                max(0, _as_int(record.get("post_replay_refresh_payloads"), 0))
                for record in producer_records
            )
        ),
        "refresh_called_count": int(
            sum(1 for record in producer_records if bool(record.get("refresh_called")))
        ),
        "refresh_us_avg": _avg(refresh_us_values),
        "refresh_us_max": max(refresh_us_values, default=-1.0),
        "refresh_us_total": float(sum(refresh_us_values)),
        "refresh_stage_total_us_avg": _avg(stage_total_us_values),
        "refresh_stage_total_us_max": max(stage_total_us_values, default=-1.0),
        "refresh_stage_total_us_total": float(sum(stage_total_us_values)),
        "refresh_stage_direct_accounted_us_avg": _avg(
            _stage_values("direct_accounted_us")
        ),
        "refresh_stage_residual_us_avg": _avg(_stage_values("residual_us")),
        "refresh_stage_prepare_logits_us_avg": _avg(
            _stage_values("prepare_logits_us")
        ),
        "refresh_stage_payload_build_us_avg": _avg(
            _stage_values("payload_build_us")
        ),
        "refresh_stage_payload_build_us_max": max(
            _stage_values("payload_build_us"),
            default=-1.0,
        ),
        "refresh_stage_slot_req_ids_us_avg": _avg(
            _stage_values("slot_req_ids_us")
        ),
        "refresh_stage_payload_enqueue_us_avg": _avg(
            _stage_values("payload_enqueue_us")
        ),
        "refresh_stage_deferred_pending_rebuild_enqueue_us_avg": _avg(
            stage_deferred_enqueue_us_values
        ),
        "refresh_stage_deferred_pending_rebuild_enqueue_us_total": float(
            sum(stage_deferred_enqueue_us_values)
        ),
        "refresh_stage_pending_group_total_us_avg": _avg(
            stage_pending_group_total_us_values
        ),
        "refresh_stage_pending_group_total_us_total": float(
            sum(stage_pending_group_total_us_values)
        ),
        "refresh_stage_pending_group_clear_us_avg": _avg(
            stage_pending_group_clear_us_values
        ),
        "refresh_stage_pending_group_carrier_prepare_us_avg": _avg(
            stage_pending_group_carrier_prepare_us_values
        ),
        "refresh_stage_pending_group_bootstrap_slots_list_us_avg": _avg(
            stage_pending_group_bootstrap_slots_list_us_values
        ),
        "refresh_stage_pending_group_enqueue_pending_us_avg": _avg(
            stage_pending_group_enqueue_pending_us_values
        ),
        "refresh_stage_pending_group_enqueue_pending_us_total": float(
            sum(stage_pending_group_enqueue_pending_us_values)
        ),
        "refresh_stage_pending_group_enqueue_prepare_us_avg": _avg(
            stage_pending_group_enqueue_prepare_us_values
        ),
        "refresh_stage_pending_group_enqueue_prepare_us_total": float(
            sum(stage_pending_group_enqueue_prepare_us_values)
        ),
        "refresh_stage_pending_group_enqueue_work_item_us_avg": _avg(
            stage_pending_group_enqueue_work_item_us_values
        ),
        "refresh_stage_pending_group_enqueue_work_item_us_total": float(
            sum(stage_pending_group_enqueue_work_item_us_values)
        ),
        "refresh_stage_pending_group_enqueue_register_us_avg": _avg(
            stage_pending_group_enqueue_register_us_values
        ),
        "refresh_stage_pending_group_enqueue_register_us_total": float(
            sum(stage_pending_group_enqueue_register_us_values)
        ),
        "refresh_stage_pending_group_enqueue_construct_us_avg": _avg(
            stage_pending_group_enqueue_construct_us_values
        ),
        "refresh_stage_pending_group_enqueue_construct_us_total": float(
            sum(stage_pending_group_enqueue_construct_us_values)
        ),
        "refresh_stage_pending_group_enqueue_coalesce_us_avg": _avg(
            stage_pending_group_enqueue_coalesce_us_values
        ),
        "refresh_stage_pending_group_enqueue_coalesce_us_total": float(
            sum(stage_pending_group_enqueue_coalesce_us_values)
        ),
        "refresh_stage_pending_group_enqueue_record_stream_us_avg": _avg(
            stage_pending_group_enqueue_record_stream_us_values
        ),
        "refresh_stage_pending_group_enqueue_record_stream_us_total": float(
            sum(stage_pending_group_enqueue_record_stream_us_values)
        ),
        "refresh_stage_pending_group_enqueue_mark_us_avg": _avg(
            stage_pending_group_enqueue_mark_us_values
        ),
        "refresh_stage_pending_group_enqueue_mark_us_total": float(
            sum(stage_pending_group_enqueue_mark_us_values)
        ),
        "refresh_stage_pending_group_enqueue_record_async_work_us_avg": _avg(
            stage_pending_group_enqueue_record_async_work_us_values
        ),
        "refresh_stage_pending_group_enqueue_record_async_work_us_total": float(
            sum(stage_pending_group_enqueue_record_async_work_us_values)
        ),
        "refresh_stage_pending_group_enqueue_compact_us_avg": _avg(
            stage_pending_group_enqueue_compact_us_values
        ),
        "refresh_stage_pending_group_enqueue_compact_us_total": float(
            sum(stage_pending_group_enqueue_compact_us_values)
        ),
        "refresh_stage_pending_group_enqueue_queue_insert_us_avg": _avg(
            stage_pending_group_enqueue_queue_insert_us_values
        ),
        "refresh_stage_pending_group_enqueue_queue_insert_us_total": float(
            sum(stage_pending_group_enqueue_queue_insert_us_values)
        ),
        "refresh_stage_progressive_selector_precompute_us_avg": _avg(
            stage_progressive_selector_precompute_us_values
        ),
        "refresh_stage_progressive_selector_precompute_us_total": float(
            sum(stage_progressive_selector_precompute_us_values)
        ),
        "refresh_stage_progressive_selector_precompute_count_total": _stage_sum_int(
            "progressive_selector_precompute_count"
        ),
        "refresh_stage_progressive_selector_precomputed_group_count_total": (
            _stage_sum_int("progressive_selector_precomputed_group_count")
        ),
        "refresh_stage_flush_us_avg": _avg(stage_flush_us_values),
        "refresh_stage_flush_us_max": max(stage_flush_us_values, default=-1.0),
        "refresh_stage_flush_us_total": float(sum(stage_flush_us_values)),
        "refresh_stage_flush_group_total_us_avg": _avg(
            stage_flush_group_total_us_values
        ),
        "refresh_stage_flush_group_total_us_total": float(
            sum(stage_flush_group_total_us_values)
        ),
        "refresh_stage_flush_group_setup_us_avg": _avg(
            stage_flush_group_setup_us_values
        ),
        "refresh_stage_flush_group_setup_us_total": float(
            sum(stage_flush_group_setup_us_values)
        ),
        "refresh_stage_flush_group_mark_submitted_us_avg": _avg(
            stage_flush_group_mark_submitted_us_values
        ),
        "refresh_stage_flush_group_mark_submitted_us_total": float(
            sum(stage_flush_group_mark_submitted_us_values)
        ),
        "refresh_stage_flush_group_record_stream_us_avg": _avg(
            stage_flush_group_record_stream_us_values
        ),
        "refresh_stage_flush_group_record_stream_us_total": float(
            sum(stage_flush_group_record_stream_us_values)
        ),
        "refresh_stage_flush_group_selector_us_avg": _avg(
            stage_flush_group_selector_us_values
        ),
        "refresh_stage_flush_group_selector_us_total": float(
            sum(stage_flush_group_selector_us_values)
        ),
        "refresh_stage_flush_group_writer_us_avg": _avg(
            stage_flush_group_writer_us_values
        ),
        "refresh_stage_flush_group_writer_us_total": float(
            sum(stage_flush_group_writer_us_values)
        ),
        "refresh_stage_flush_group_gather_event_us_avg": _avg(
            stage_flush_group_gather_event_us_values
        ),
        "refresh_stage_flush_group_gather_event_us_total": float(
            sum(stage_flush_group_gather_event_us_values)
        ),
        "refresh_stage_flush_group_async_event_us_avg": _avg(
            stage_flush_group_async_event_us_values
        ),
        "refresh_stage_flush_group_async_event_us_total": float(
            sum(stage_flush_group_async_event_us_values)
        ),
        "refresh_stage_flush_group_count_total": _stage_sum_int(
            "flush_group_count"
        ),
        "refresh_stage_layer_count_total": _stage_sum_int("layer_count"),
        "refresh_stage_payload_count_total": _stage_sum_int("payload_count"),
        "refresh_stage_flush_call_count_total": _stage_sum_int("flush_call_count"),
        "refresh_stage_deferred_pending_rebuild_count_total": _stage_sum_int(
            "deferred_pending_rebuild_count"
        ),
        "refresh_stage_pending_group_count_total": _stage_sum_int(
            "pending_group_count"
        ),
        "refresh_stage_pending_group_payload_count_total": _stage_sum_int(
            "pending_group_payload_count"
        ),
        "pre_original_us_avg": _avg(pre_original_us_values),
        "pre_original_us_max": max(pre_original_us_values, default=-1.0),
        "pre_original_us_total": float(sum(pre_original_us_values)),
        "pre_forward_context_us_avg": _avg(pre_forward_context_us_values),
        "pre_forward_context_us_max": max(
            pre_forward_context_us_values,
            default=-1.0,
        ),
        "pre_forward_context_us_total": float(sum(pre_forward_context_us_values)),
        "pre_graph_lookup_us_avg": _avg(pre_graph_lookup_us_values),
        "pre_graph_lookup_us_max": max(pre_graph_lookup_us_values, default=-1.0),
        "pre_graph_lookup_us_total": float(sum(pre_graph_lookup_us_values)),
        "pre_prebound_mark_us_avg": _avg(pre_prebound_mark_us_values),
        "pre_prebound_mark_us_max": max(
            pre_prebound_mark_us_values,
            default=-1.0,
        ),
        "pre_prebound_mark_us_total": float(sum(pre_prebound_mark_us_values)),
        "pre_prebound_state_lookup_us_avg": _avg(
            pre_prebound_state_lookup_us_values
        ),
        "pre_prebound_state_lookup_us_max": max(
            pre_prebound_state_lookup_us_values,
            default=-1.0,
        ),
        "pre_prebound_state_lookup_us_total": float(
            sum(pre_prebound_state_lookup_us_values)
        ),
        "pre_ready_event_wait_us_avg": _avg(pre_ready_event_wait_us_values),
        "pre_ready_event_wait_us_max": max(
            pre_ready_event_wait_us_values,
            default=-1.0,
        ),
        "pre_ready_event_wait_us_total": float(sum(pre_ready_event_wait_us_values)),
        "pre_prebound_stats_us_avg": _avg(pre_prebound_stats_us_values),
        "pre_prebound_stats_us_max": max(
            pre_prebound_stats_us_values,
            default=-1.0,
        ),
        "pre_prebound_stats_us_total": float(sum(pre_prebound_stats_us_values)),
        "pre_other_us_avg": _avg(pre_other_us_values),
        "pre_other_us_max": max(pre_other_us_values, default=-1.0),
        "pre_other_us_total": float(sum(pre_other_us_values)),
        "pre_ready_event_wait_count_total": int(
            sum(
                max(0, _as_int(record.get("pre_ready_event_wait_count"), 0))
                for record in wrapper_records
            )
        ),
        "refresh_stage_producer_work_items_with_deadline_total": _stage_sum_int(
            "producer_work_items_with_deadline"
        ),
        "refresh_stage_producer_work_deadline_slack_steps_min": (
            min(producer_work_deadline_slack_steps)
            if producer_work_deadline_slack_steps
            else -1
        ),
        "refresh_stage_producer_work_decode_step_min": (
            min(producer_work_decode_step_min) if producer_work_decode_step_min else -1
        ),
        "refresh_stage_producer_work_decode_step_max": (
            max(producer_work_decode_step_max) if producer_work_decode_step_max else -1
        ),
        "refresh_stage_producer_work_decode_steps_observed": (
            producer_work_decode_steps_observed
        ),
        "post_call_us_avg": _avg(post_call_us_values),
        "post_call_us_max": max(post_call_us_values, default=-1.0),
        "post_call_us_total": float(sum(post_call_us_values)),
        "pre_consume_pending_rebuild_drain_us_avg": _avg(
            pre_consume_drain_us_values
        ),
        "pre_consume_pending_rebuild_drain_us_total": float(
            sum(pre_consume_drain_us_values)
        ),
        "pre_consume_pending_rebuild_drained_total": int(
            sum(
                max(0, _as_int(record.get("pre_consume_pending_rebuild_drained"), 0))
                for record in wrapper_records
            )
        ),
        "pre_consume_pending_rebuild_dropped_total": int(
            sum(
                max(0, _as_int(record.get("pre_consume_pending_rebuild_dropped"), 0))
                for record in wrapper_records
            )
        ),
        "deferred_replay_refresh_drain_profile_count": len(
            deferred_drain_total_us_values
        ),
        "deferred_replay_refresh_drain_total_us_avg": _avg(
            deferred_drain_total_us_values
        ),
        "deferred_replay_refresh_drain_total_us_total": float(
            sum(deferred_drain_total_us_values)
        ),
        "deferred_replay_refresh_drain_drained_total": (
            _deferred_drain_profile_sum_int("drained")
        ),
        "deferred_replay_refresh_drain_pending_group_total_us_avg": _avg(
            deferred_drain_pending_group_total_us_values
        ),
        "deferred_replay_refresh_drain_pending_group_total_us_total": float(
            sum(deferred_drain_pending_group_total_us_values)
        ),
        "deferred_replay_refresh_drain_pending_group_clear_us_avg": _avg(
            deferred_drain_pending_group_clear_us_values
        ),
        "deferred_replay_refresh_drain_pending_group_carrier_prepare_us_avg": _avg(
            deferred_drain_pending_group_carrier_prepare_us_values
        ),
        "deferred_replay_refresh_drain_pending_group_bootstrap_slots_list_us_avg": (
            _avg(deferred_drain_pending_group_bootstrap_slots_list_us_values)
        ),
        "deferred_replay_refresh_drain_pending_group_enqueue_pending_us_avg": _avg(
            deferred_drain_pending_group_enqueue_pending_us_values
        ),
        "deferred_replay_refresh_drain_pending_group_count_total": (
            _deferred_drain_profile_sum_int("pending_group_count")
        ),
        "deferred_replay_refresh_drain_pending_group_payload_count_total": (
            _deferred_drain_profile_sum_int("pending_group_payload_count")
        ),
        "reason_counts": dict(sorted(reason_counts.items())),
        "hook_counts": dict(sorted(hook_counts.items())),
    }


def _records_avg(records: list[dict[str, Any]], key: str) -> float:
    return _avg(_profile_float_values(records, key))


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return -1.0
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * percentile))
    index = min(max(index, 0), len(ordered) - 1)
    return float(ordered[index])


def _nearest_step_delta(step_index: int, steps: list[int]) -> int:
    if step_index < 0 or not steps:
        return -1
    return min(abs(int(step_index) - int(step)) for step in steps)


def _range_step_delta(step_index: int, step_min: int, step_max: int) -> int:
    if step_index < 0 or step_min < 0 or step_max < step_min:
        return -1
    if step_index < step_min:
        return int(step_min - step_index)
    if step_index > step_max:
        return int(step_index - step_max)
    return 0


def _nearest_outlier_delta_to_steps(
    outliers: list[dict[str, object]],
    steps: list[int],
) -> int:
    if not steps or not outliers:
        return -1
    return min(
        _nearest_step_delta(_as_int(row.get("step_index"), -1), steps)
        for row in outliers
    )


def _nearest_outlier_delta_to_range(
    outliers: list[dict[str, object]],
    step_min: int,
    step_max: int,
) -> int:
    if not outliers:
        return -1
    deltas = [
        _range_step_delta(_as_int(row.get("step_index"), -1), step_min, step_max)
        for row in outliers
    ]
    deltas = [delta for delta in deltas if delta >= 0]
    return min(deltas) if deltas else -1


def _decode_tail_max_outlier_attribution(
    *,
    max_step_index: int,
    drain_submit_steps: list[int],
    producer_steps: list[int],
    producer_step_min: int,
    producer_step_max: int,
) -> dict[str, object]:
    max_to_drain = _nearest_step_delta(max_step_index, drain_submit_steps)
    max_to_producer = _nearest_step_delta(max_step_index, producer_steps)
    max_to_range = _range_step_delta(
        max_step_index,
        producer_step_min,
        producer_step_max,
    )
    if 0 <= max_to_drain <= DECODE_TAIL_PRODUCER_STEP_NEAR_DELTA:
        attribution = "drain_submit"
        max_to_attribution = max_to_drain
    elif 0 <= max_to_producer <= DECODE_TAIL_PRODUCER_STEP_NEAR_DELTA:
        attribution = "producer_work"
        max_to_attribution = max_to_producer
    elif (
        not producer_steps
        and 0 <= max_to_range <= DECODE_TAIL_PRODUCER_STEP_NEAR_DELTA
    ):
        attribution = "producer_range"
        max_to_attribution = max_to_range
    elif max_step_index >= 0:
        attribution = "non_refresh_decode_tail"
        deltas = [
            delta
            for delta in (max_to_drain, max_to_producer, max_to_range)
            if delta >= 0
        ]
        max_to_attribution = min(deltas) if deltas else -1
    else:
        attribution = "none"
        max_to_attribution = -1
    return {
        "max_outlier_to_drain_submit_step_delta": int(max_to_drain),
        "max_outlier_to_producer_work_step_delta": int(max_to_producer),
        "max_outlier_to_producer_range_delta": int(max_to_range),
        "max_outlier_to_attribution_step_delta": int(max_to_attribution),
        "max_outlier_attribution": attribution,
    }


def _decode_tail_near_attribution_count(
    *,
    outliers: list[dict[str, object]],
    drain_submit_steps: list[int],
    producer_steps: list[int],
    producer_step_min: int,
    producer_step_max: int,
) -> int:
    count = 0
    for row in outliers:
        step_index = _as_int(row.get("step_index"), -1)
        deltas = [
            _nearest_step_delta(step_index, drain_submit_steps),
            _nearest_step_delta(step_index, producer_steps),
        ]
        if not producer_steps:
            deltas.append(
                _range_step_delta(step_index, producer_step_min, producer_step_max)
            )
        if any(
            0 <= delta <= DECODE_TAIL_PRODUCER_STEP_NEAR_DELTA for delta in deltas
        ):
            count += 1
    return count


def _steady_avg_excluding_first(
    refresh_profile: list[dict[str, Any]],
    key: str,
) -> float:
    return _avg(
        _profile_float_values(_steady_prefill_profile_records(refresh_profile), key)
    )


def _refresh_enqueued_avg(
    refresh_profile: list[dict[str, Any]],
    key: str,
) -> float:
    return _avg(
        _profile_float_values(_refresh_rebuild_enqueued_records(refresh_profile), key)
    )


def _refresh_work_item_avg(
    refresh_profile: list[dict[str, Any]],
    key: str,
) -> float:
    return _avg(
        _profile_float_values(_refresh_work_item_records(refresh_profile), key)
    )


def _refresh_work_item_max(
    refresh_profile: list[dict[str, Any]],
    key: str,
) -> float:
    values = _profile_float_values(_refresh_work_item_records(refresh_profile), key)
    return max(values) if values else -1.0


def _deadline_async_producer_counter_avg(
    refresh_profile: list[dict[str, Any]],
    total_key: str,
    *,
    count_key: str = "deadline_async_producer_key_norms_delta_count",
) -> float:
    count = _max_int_from_records(refresh_profile, count_key)
    total = _max_us_from_records(refresh_profile, total_key)
    return _avg_from_total_count(total, count)


def _refresh_selector_wrapper_gap_enqueued_avg(
    refresh_profile: list[dict[str, Any]],
) -> float:
    return _selector_wrapper_gap_avg(_refresh_rebuild_enqueued_records(refresh_profile))


def _selector_wrapper_gap_avg(records: list[dict[str, Any]]) -> float:
    values: list[float] = []
    for record in records:
        apply_us = _as_float(record.get("refresh_selector_apply_cpu_us"), -1.0)
        if apply_us < 0.0:
            continue
        compute_us = max(
            0.0,
            _as_float(record.get("refresh_selector_compute_cpu_us"), 0.0),
        )
        post_us = max(
            0.0,
            _as_float(record.get("refresh_selector_post_cpu_us"), 0.0),
        )
        stack_us = max(
            0.0,
            _as_float(record.get("refresh_selector_stack_cpu_us"), 0.0),
        )
        values.append(max(0.0, apply_us - compute_us - post_us - stack_us))
    return _avg(values)


def _refresh_selector_wrapper_gap_work_item_avg(
    refresh_profile: list[dict[str, Any]],
) -> float:
    return _selector_wrapper_gap_avg(_refresh_work_item_records(refresh_profile))


def _steady_p50_excluding_first(
    refresh_profile: list[dict[str, Any]],
    key: str,
) -> float:
    return _percentile(
        _profile_float_values(_steady_prefill_profile_records(refresh_profile), key),
        0.50,
    )


def _steady_p95_excluding_first(
    refresh_profile: list[dict[str, Any]],
    key: str,
) -> float:
    return _percentile(
        _profile_float_values(_steady_prefill_profile_records(refresh_profile), key),
        0.95,
    )


def _steady_avg_delta_excluding_first(
    refresh_profile: list[dict[str, Any]],
    *,
    start_key: str,
    end_key: str,
) -> float:
    values: list[float] = []
    for record in _steady_prefill_profile_records(refresh_profile):
        start_value = record.get(start_key)
        end_value = record.get(end_key)
        if start_value is None or end_value is None:
            continue
        try:
            start_numeric = float(start_value)
            end_numeric = float(end_value)
        except (TypeError, ValueError):
            continue
        if start_numeric >= 0.0 and end_numeric >= 0.0:
            values.append(max(0.0, end_numeric - start_numeric))
    return _avg(values)


def _falsification_steady_summary(
    refresh_profile: list[dict[str, Any]],
) -> dict[str, float]:
    group_count = _steady_avg_excluding_first(refresh_profile, "prefill_group_count")
    group_values = [
        _steady_avg_excluding_first(refresh_profile, f"prefill_group{group_id}_gpu_ms")
        for group_id in range(4)
    ]
    present_group_values = [value for value in group_values if value >= 0.0]
    group_sum_ms = sum(present_group_values) if present_group_values else -1.0
    selector_ms = _steady_avg_excluding_first(
        refresh_profile,
        "prefill_selector_gpu_ms",
    )
    rebuild_ms = _steady_avg_excluding_first(
        refresh_profile,
        "prefill_rebuild_gpu_ms",
    )
    prefill_ms = _steady_avg_excluding_first(refresh_profile, "prefill_gpu_ms")

    boundary_total_ms = -1.0
    boundary_per_group = -1.0
    if (
        group_count > 0.0
        and group_sum_ms >= 0.0
        and selector_ms >= 0.0
        and rebuild_ms >= 0.0
    ):
        boundary_total_ms = max(0.0, group_sum_ms - selector_ms - rebuild_ms)
        boundary_per_group = boundary_total_ms / group_count

    if prefill_ms >= 0.0 and group_sum_ms >= 0.0:
        residual_ms = max(0.0, prefill_ms - group_sum_ms)
    elif boundary_total_ms >= 0.0:
        residual_ms = 0.0
    else:
        residual_ms = -1.0

    return {
        "selected_indices_materialized_bytes_steady_avg_excluding_first": (
            _steady_avg_excluding_first(
                refresh_profile,
                "selected_indices_materialized_bytes",
            )
        ),
        "selected_indices_io_bytes_steady_avg_excluding_first": (
            _steady_avg_excluding_first(refresh_profile, "selected_indices_io_bytes")
        ),
        "selector_writer_current_path_count_steady_avg_excluding_first": (
            _steady_avg_excluding_first(
                refresh_profile,
                "selector_writer_current_path_count",
            )
        ),
        "selector_writer_boundary_cpu_us_steady_avg_excluding_first": (
            _steady_avg_excluding_first(
                refresh_profile,
                "selector_writer_boundary_cpu_us",
            )
        ),
        "refresh_selector_apply_cpu_us_enqueued_avg": (
            _refresh_enqueued_avg(
                refresh_profile,
                "refresh_selector_apply_cpu_us",
            )
        ),
        "refresh_selector_compute_cpu_us_enqueued_avg": (
            _refresh_enqueued_avg(
                refresh_profile,
                "refresh_selector_compute_cpu_us",
            )
        ),
        "refresh_selector_post_cpu_us_enqueued_avg": (
            _refresh_enqueued_avg(
                refresh_profile,
                "refresh_selector_post_cpu_us",
            )
        ),
        "refresh_selector_stack_cpu_us_enqueued_avg": (
            _refresh_enqueued_avg(
                refresh_profile,
                "refresh_selector_stack_cpu_us",
            )
        ),
        "refresh_selector_key_norms_cpu_us_enqueued_avg": (
            _refresh_enqueued_avg(
                refresh_profile,
                "refresh_selector_key_norms_cpu_us",
            )
        ),
        "refresh_selector_key_norms_direct_prepare_cpu_us_enqueued_avg": (
            _refresh_enqueued_avg(
                refresh_profile,
                "refresh_selector_key_norms_direct_prepare_cpu_us",
            )
        ),
        "refresh_selector_key_norms_direct_launch_cpu_us_enqueued_avg": (
            _refresh_enqueued_avg(
                refresh_profile,
                "refresh_selector_key_norms_direct_launch_cpu_us",
            )
        ),
        "refresh_selector_key_norms_pack_cpu_us_enqueued_avg": (
            _refresh_enqueued_avg(
                refresh_profile,
                "refresh_selector_key_norms_pack_cpu_us",
            )
        ),
        "refresh_selector_wrapper_gap_cpu_us_enqueued_avg": (
            _refresh_selector_wrapper_gap_enqueued_avg(refresh_profile)
        ),
        "selector_writer_boundary_cpu_us_enqueued_avg": (
            _refresh_enqueued_avg(
                refresh_profile,
                "selector_writer_boundary_cpu_us",
            )
        ),
        "refresh_total_cpu_us_enqueued_avg": (
            _refresh_enqueued_avg(
                refresh_profile,
                "refresh_total_cpu_us",
            )
        ),
        "refresh_selector_cpu_us_enqueued_avg": (
            _refresh_enqueued_avg(
                refresh_profile,
                "refresh_selector_cpu_us",
            )
        ),
        "refresh_rebuild_enqueue_cpu_us_enqueued_avg": (
            _refresh_enqueued_avg(
                refresh_profile,
                "refresh_rebuild_enqueue_cpu_us",
            )
        ),
        "refresh_rebuild_compact_cpu_us_enqueued_avg": (
            _refresh_enqueued_avg(
                refresh_profile,
                "refresh_rebuild_compact_cpu_us",
            )
        ),
        "selected_boundary_lower_bound_ms_per_group": boundary_per_group,
        "predicted_front_early_step_improvement_ms": boundary_total_ms,
        "residual_fixed_capture_control_ms": residual_ms,
    }


def _ms_from_us(value_us: float) -> float:
    return float(value_us) / 1000.0 if value_us >= 0.0 else -1.0


def _avg_from_total_count(total_us: float, count: int) -> float:
    return float(total_us) / float(count) if count > 0 and total_us >= 0.0 else -1.0


def _parse_selector_pipeline_cpu_profile_lines(
    lines: list[str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    current_phase = ""
    int_keys = {
        "pid",
        "L",
        "B",
        "H",
        "G",
        "W",
        "K",
        "k_head",
        "slice_start",
        "slice_end",
        "k_eff",
    }
    us_keys = {"us_log_s", "us_nms", "us_cross", "us_topk", "us_total"}
    for line in lines:
        if line.startswith("pipeline_phase "):
            for part in line.split()[1:]:
                if "=" not in part:
                    continue
                key, value = part.split("=", 1)
                if key == "phase":
                    current_phase = value
                    break
            continue
        if not line.startswith("pipeline_cpu "):
            continue
        record: dict[str, Any] = {}
        if current_phase:
            record["profile_phase"] = current_phase
        for part in line.split()[1:]:
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            if key in int_keys:
                record[key] = _as_int(value, -1)
            elif key in us_keys:
                record[key] = _as_float(value, -1.0)
            else:
                record[key] = value
        if record:
            records.append(record)
    return records


def _read_selector_pipeline_cpu_profile(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return _parse_selector_pipeline_cpu_profile_lines(
        path.read_text(encoding="utf-8").splitlines()
    )


def _deadline_v2_primary_action(primary_bottleneck: str) -> str:
    return {
        "wait": "tighten_deadline_wait_and_ready_epoch_admission",
        "writer_publish": "optimize_writer_publish_pipeline",
        "selector_prepare": "staticize_selector_prepare_carrier",
        "selector_compute": "move_selector_reduce_earlier_without_changing_topk",
        "deferred_selector_compute": "drain_deferred_selector_before_decode_deadline",
        "async_producer_selector": "fuse_or_graph_async_producer_selector_stage",
        "async_producer_writer": "graph_or_staticize_async_producer_writer_stage",
        "async_producer_body_gap": "remove_async_producer_envelope_tax",
        "async_producer_graph_capture": "move_producer_graph_capture_off_decode_path",
        "async_producer_graph_replay": "keep_graph_replay_if_capture_cost_is_amortized",
        "wrapper_jitter": "staticize_wrapper_and_bucket_carriers",
        "decode_jitter": "rerun_case_and_split_decode_tail_jitter",
        "none": "no_dominant_deadline_v2_exposure",
        "unmeasured": "enable_refresh_profile_and_one_shot_timeline",
    }.get(primary_bottleneck, "inspect_deadline_v2_attribution")


def _deadline_v2_selector_stage_action(component: str) -> str:
    return {
        "key_norms": "prefetch_or_cache_key_norm_inputs_before_decode_deadline",
        "pure_preproc": "staticize_preproc_bounds_and_mask_carrier",
        "bounds": "staticize_bounds_carrier_or_move_bounds_to_deadline_producer",
        "pipeline": "split_pipeline_topk_and_prior_before_semantic_changes",
        "seq_full": "prebind_seq_full_bucket_carrier",
        "topk": "schedule_existing_topk_earlier_without_semantic_change",
        "log_s": "prestage_log_f_shell_without_legacy_fallback",
        "gather": "keep_writer_gather_off_decode_deadline",
        "rebuild": "move_rebuild_publish_by_consumer_layer_deadline",
        "selector_total": "split_selector_gpu_profile_before_optimizing",
        "unmeasured": "enable_refresh_profile_detail",
    }.get(component, "inspect_selector_compute_breakdown")


def _deadline_v2_key_norms_stage_action(component: str) -> str:
    return {
        "key_norms_h2d": "cache_key_norms_delta_bounds_on_device",
        "key_norms_delta": "move_key_norms_delta_compute_to_prefetch_deadline",
        "key_norms_pack": "remove_sidecar_pack_or_make_selector_consume_sidecar",
        "key_norms_preproc": "sink_key_norms_preproc_into_static_carrier",
        "key_norms_envelope_residual": "split_key_norms_event_envelope_before_optimizing",
        "unmeasured": "enable_refresh_key_norms_detail_profile",
    }.get(component, "inspect_key_norms_breakdown")


def _deadline_v2_key_norms_envelope_gap_action(component: str) -> str:
    return {
        "pre_h2d": "prebind_key_norms_delta_bounds_and_arena_before_decode_deadline",
        "h2d_to_delta": "chain_key_norms_h2d_to_delta_without_host_gap",
        "delta_to_pack": "chain_key_norms_delta_to_pack_or_consume_sidecar",
        "post_pack": "publish_key_norms_ready_after_pack_without_host_gap",
        "unmeasured": "split_key_norms_event_envelope_before_optimizing",
    }.get(component, "inspect_key_norms_envelope_gap")


def _deadline_v2_deferred_selector_cpu_stage_action(component: str) -> str:
    return {
        "key_norms": "fuse_or_prefetch_async_selector_key_norms_stage",
        "select": "fuse_or_batch_async_selector_select_stage",
        "validate": "trim_async_selector_validation_hotpath",
        "stack": "staticize_async_selector_stack_inputs",
        "post": "trim_async_selector_post_tracking",
        "wrapper_gap": "remove_async_selector_python_wrapper_gap",
        "selector_total": "split_async_selector_cpu_profile_before_optimizing",
        "unmeasured": "enable_deferred_selector_profile_detail",
    }.get(component, "inspect_deferred_selector_cpu_breakdown")


def _deadline_v2_decode_step_durations(metrics: dict[str, Any]) -> list[float]:
    values = metrics.get("decode_step_durations_us")
    if not isinstance(values, list):
        return []
    durations: list[float] = []
    for value in values:
        duration = _as_float(value, -1.0)
        if duration >= 0.0:
            durations.append(duration)
    return durations


def _deadline_v2_decode_tail_jitter_breakdown(
    metrics: dict[str, Any],
    *,
    queue_depth_max: int,
    enqueued_count: int,
    deferred_selector_count: int,
    producer_step_min: int,
    producer_step_max: int,
    producer_steps: list[int],
    drain_submit_steps: list[int],
) -> dict[str, object]:
    steps_us = _deadline_v2_decode_step_durations(metrics)
    if len(steps_us) < 2:
        return {
            "schema": "decode_tail_jitter_breakdown_v1",
            "sample_count": len(steps_us),
            "steady_sample_count": 0,
            "next_action": DECODE_TAIL_ENABLE_ACTION,
        }
    steady_steps = steps_us[1:]
    steady_p50_us = _percentile(steady_steps, 0.50)
    steady_p95_us = _percentile(steady_steps, 0.95)
    steady_jitter_us = max(0.0, steady_p95_us - steady_p50_us)
    threshold_us = steady_p50_us + max(
        DECODE_TAIL_OUTLIER_MIN_EXTRA_US,
        steady_p50_us * (DECODE_TAIL_OUTLIER_MIN_RATIO - 1.0),
    )
    outliers = [
        {
            "step_index": index,
            "duration_us": float(duration),
            "excess_over_steady_p50_us": float(duration - steady_p50_us),
        }
        for index, duration in enumerate(steps_us)
        if index > 0 and duration >= threshold_us
    ]
    outliers.sort(key=lambda row: float(row["duration_us"]), reverse=True)
    producer_step_known = (
        producer_step_min >= 0 and producer_step_max >= producer_step_min
    )
    producer_steps_unique = sorted(set(int(step) for step in producer_steps if step >= 0))
    drain_submit_steps_unique = sorted(
        set(int(step) for step in drain_submit_steps if step >= 0)
    )
    producer_outlier_delta = _nearest_outlier_delta_to_steps(
        outliers,
        producer_steps_unique,
    )
    drain_submit_outlier_delta = _nearest_outlier_delta_to_steps(
        outliers,
        drain_submit_steps_unique,
    )
    attribution_steps = drain_submit_steps_unique or producer_steps_unique
    attribution_anchor_known = bool(attribution_steps or producer_step_known)
    attribution_step_source = (
        "drain_submit"
        if drain_submit_steps_unique
        else ("producer_work" if producer_steps_unique else "range")
    )
    nearest_outlier_delta = -1
    if drain_submit_steps_unique:
        nearest_outlier_delta = drain_submit_outlier_delta
    elif producer_steps_unique:
        nearest_outlier_delta = producer_outlier_delta
    elif producer_step_known and outliers:
        nearest_outlier_delta = _nearest_outlier_delta_to_range(
            outliers,
            producer_step_min,
            producer_step_max,
        )
    tail_start = max(1, len(steps_us) - 2)
    tail_end_outlier_count = sum(
        1 for row in outliers if int(row["step_index"]) >= tail_start
    )
    max_row = outliers[0] if outliers else {}
    max_step_index = _as_int(max_row.get("step_index"), -1)
    max_attr = _decode_tail_max_outlier_attribution(
        max_step_index=max_step_index,
        drain_submit_steps=drain_submit_steps_unique,
        producer_steps=producer_steps_unique,
        producer_step_min=producer_step_min,
        producer_step_max=producer_step_max,
    )
    attributed_outlier_count = _decode_tail_near_attribution_count(
        outliers=outliers,
        drain_submit_steps=drain_submit_steps_unique,
        producer_steps=producer_steps_unique,
        producer_step_min=producer_step_min,
        producer_step_max=producer_step_max,
    )
    if int(deferred_selector_count) > 0:
        next_action = "drain_deferred_selector_before_decode_deadline"
    elif (
        attribution_anchor_known
        and outliers
        and max_attr.get("max_outlier_attribution") == "non_refresh_decode_tail"
    ):
        next_action = DECODE_TAIL_NON_REFRESH_ACTION
    elif attribution_anchor_known and outliers:
        if 0 <= nearest_outlier_delta <= DECODE_TAIL_PRODUCER_STEP_NEAR_DELTA:
            next_action = DECODE_TAIL_PENDING_DRAIN_ACTION
        else:
            next_action = DECODE_TAIL_NON_REFRESH_ACTION
    elif int(queue_depth_max) > 0 or int(enqueued_count) > 0:
        next_action = DECODE_TAIL_CORRELATE_ACTION
    elif int(tail_end_outlier_count) > 0:
        next_action = DECODE_TAIL_SPLIT_POST_TAIL_ACTION
    elif outliers:
        next_action = "inspect_decode_step_outlier_bursts"
    else:
        next_action = "continue_selector_compute_attribution"
    return {
        "schema": "decode_tail_jitter_breakdown_v1",
        "sample_count": len(steps_us),
        "steady_sample_count": len(steady_steps),
        "first_step_us": float(steps_us[0]),
        "steady_p50_us": float(steady_p50_us),
        "steady_p95_us": float(steady_p95_us),
        "steady_jitter_us": float(steady_jitter_us),
        "outlier_threshold_us": float(threshold_us),
        "outlier_count": len(outliers),
        "outlier_fraction": float(len(outliers)) / float(len(steady_steps)),
        "tail_end_outlier_count": int(tail_end_outlier_count),
        "max_outlier_step_index": int(max_row.get("step_index", -1) or -1),
        "max_outlier_duration_us": float(max_row.get("duration_us", -1.0) or -1.0),
        "top_outliers": outliers[:DECODE_TAIL_TOP_OUTLIER_LIMIT],
        "queue_depth_max": int(queue_depth_max),
        "producer_work_decode_step_min": int(producer_step_min),
        "producer_work_decode_step_max": int(producer_step_max),
        "producer_work_decode_steps_observed": producer_steps_unique,
        "pending_rebuild_drain_submit_decode_steps_observed": (
            drain_submit_steps_unique
        ),
        "nearest_outlier_to_producer_decode_step_delta": int(nearest_outlier_delta),
        "nearest_outlier_to_producer_work_step_delta": int(producer_outlier_delta),
        "nearest_outlier_to_drain_submit_step_delta": int(drain_submit_outlier_delta),
        **max_attr,
        "attributed_outlier_count": int(attributed_outlier_count),
        "non_attributed_outlier_count": int(
            max(0, len(outliers) - attributed_outlier_count)
        ),
        "attribution_step_source": str(attribution_step_source),
        "producer_work_enqueued_admission_count": int(enqueued_count),
        "producer_work_deferred_selector_admission_count": int(deferred_selector_count),
        "next_action": next_action,
    }


def _deadline_v2_pipeline_cpu_stage_action(component: str) -> str:
    return {
        "log_s": "prestage_log_f_shell_without_semantic_change",
        "nms": "audit_soft_nms_cost_without_sparse_semantic_change",
        "cross": "move_cross_head_reduce_earlier_without_semantic_change",
        "topk": "schedule_existing_topk_earlier_without_semantic_change",
        "unmeasured": "enable_selector_pipeline_cpu_profile",
    }.get(component, "inspect_selector_pipeline_cpu_breakdown")


def _pipeline_cpu_bucket_key(record: dict[str, Any]) -> tuple[object, ...]:
    return (
        str(record.get("kind", "") or ""),
        _as_int(record.get("L"), -1),
        _as_int(record.get("B"), -1),
        _as_int(record.get("H"), -1),
        _as_int(record.get("G"), -1),
        _as_int(record.get("W"), -1),
        _as_int(record.get("K"), -1),
        _as_int(record.get("k_head"), -1),
        _as_int(record.get("slice_start"), -1),
        _as_int(record.get("slice_end"), -1),
        _as_int(record.get("k_eff"), -1),
    )


def _pipeline_cpu_bucket_label(key: tuple[object, ...]) -> str:
    (
        kind,
        layers,
        batch,
        heads,
        groups,
        window,
        kv_len,
        k_head,
        slice_start,
        slice_end,
        k_eff,
    ) = key
    label = (
        f"kind={kind} L={layers} B={batch} H={heads} "
        f"G={groups} W={window} K={kv_len}"
    )
    if int(k_head) >= 0:
        label += (
            f" k_head={k_head} slice_start={slice_start} "
            f"slice_end={slice_end} k_eff={k_eff}"
        )
    return label


def _pipeline_cpu_bucket_breakdown(
    records: list[dict[str, Any]],
) -> dict[str, object]:
    buckets: dict[tuple[object, ...], list[dict[str, Any]]] = {}
    for record in records:
        buckets.setdefault(_pipeline_cpu_bucket_key(record), []).append(record)

    steady_records: list[dict[str, Any]] = []
    cold_records: list[tuple[tuple[object, ...], dict[str, Any]]] = []
    bucket_topk_rows: list[tuple[tuple[object, ...], float, float, int]] = []
    for key, bucket_records in buckets.items():
        if not bucket_records:
            continue
        cold_records.append((key, bucket_records[0]))
        bucket_steady = bucket_records[1:]
        if not bucket_steady:
            continue
        steady_records.extend(bucket_steady)
        bucket_topk_rows.append(
            (
                key,
                _records_avg(bucket_steady, "us_topk"),
                _records_avg(bucket_steady, "us_total"),
                len(bucket_steady),
            )
        )

    steady_log_s_us = _records_avg(steady_records, "us_log_s")
    steady_nms_us = _records_avg(steady_records, "us_nms")
    steady_cross_us = _records_avg(steady_records, "us_cross")
    steady_topk_us = _records_avg(steady_records, "us_topk")
    steady_total_us = _records_avg(steady_records, "us_total")
    steady_components = {
        "log_s": steady_log_s_us,
        "nms": steady_nms_us,
        "cross": steady_cross_us,
        "topk": steady_topk_us,
    }
    measured_steady_components = {
        key: value for key, value in steady_components.items() if value >= 0.0
    }
    if measured_steady_components:
        largest_component, largest_us = max(
            measured_steady_components.items(),
            key=lambda item: item[1],
        )
    else:
        largest_component, largest_us = "unmeasured", -1.0

    if bucket_topk_rows:
        top_bucket, top_bucket_topk_us, top_bucket_total_us, top_bucket_samples = max(
            bucket_topk_rows,
            key=lambda item: item[1],
        )
        top_bucket_label = _pipeline_cpu_bucket_label(top_bucket)
    else:
        top_bucket_label = ""
        top_bucket_topk_us = -1.0
        top_bucket_total_us = -1.0
        top_bucket_samples = 0

    cold_topk_bucket = ""
    cold_topk_us = -1.0
    for key, record in cold_records:
        value = _as_float(record.get("us_topk"), -1.0)
        if value > cold_topk_us:
            cold_topk_us = value
            cold_topk_bucket = _pipeline_cpu_bucket_label(key)

    return {
        "selector_pipeline_cpu_bucket_count": int(len(buckets)),
        "selector_pipeline_cpu_bucket_steady_sample_count": int(
            len(steady_records)
        ),
        "selector_pipeline_cpu_log_s_us_bucket_steady_avg_excluding_first": float(
            steady_log_s_us
        ),
        "selector_pipeline_cpu_nms_us_bucket_steady_avg_excluding_first": float(
            steady_nms_us
        ),
        "selector_pipeline_cpu_cross_us_bucket_steady_avg_excluding_first": float(
            steady_cross_us
        ),
        "selector_pipeline_cpu_topk_us_bucket_steady_avg_excluding_first": float(
            steady_topk_us
        ),
        "selector_pipeline_cpu_total_us_bucket_steady_avg_excluding_first": float(
            steady_total_us
        ),
        "largest_bucket_steady_pipeline_cpu_component": largest_component,
        "largest_bucket_steady_pipeline_cpu_component_us": float(largest_us),
        "largest_steady_pipeline_cpu_bucket": top_bucket_label,
        "largest_steady_pipeline_cpu_bucket_topk_us": float(top_bucket_topk_us),
        "largest_steady_pipeline_cpu_bucket_total_us": float(top_bucket_total_us),
        "largest_steady_pipeline_cpu_bucket_sample_count": int(top_bucket_samples),
        "selector_pipeline_cpu_cold_topk_us_max": float(cold_topk_us),
        "selector_pipeline_cpu_cold_topk_bucket": cold_topk_bucket,
        "bucket_steady_pipeline_cpu_next_action": (
            _deadline_v2_pipeline_cpu_stage_action(largest_component)
        ),
    }


def _deadline_v2_pipeline_cpu_breakdown(
    selector_pipeline_cpu_profile: list[dict[str, Any]] | None,
) -> dict[str, object]:
    all_records = selector_pipeline_cpu_profile or []
    measurement_records = [
        record
        for record in all_records
        if record.get("profile_phase") == "measure_begin"
    ]
    warmup_records = [
        record
        for record in all_records
        if record.get("profile_phase") == "warmup_begin"
    ]
    records = measurement_records or all_records
    steady_records = records[1:] if len(records) > 1 else []
    sample_count = len(records)
    log_s_us = _records_avg(records, "us_log_s")
    nms_us = _records_avg(records, "us_nms")
    cross_us = _records_avg(records, "us_cross")
    topk_us = _records_avg(records, "us_topk")
    total_us = _records_avg(records, "us_total")
    steady_log_s_us = _records_avg(steady_records, "us_log_s")
    steady_nms_us = _records_avg(steady_records, "us_nms")
    steady_cross_us = _records_avg(steady_records, "us_cross")
    steady_topk_us = _records_avg(steady_records, "us_topk")
    steady_total_us = _records_avg(steady_records, "us_total")
    components = {
        "log_s": log_s_us,
        "nms": nms_us,
        "cross": cross_us,
        "topk": topk_us,
    }
    measured_components = {
        key: value for key, value in components.items() if value >= 0.0
    }
    if measured_components:
        largest_component, largest_us = max(
            measured_components.items(),
            key=lambda item: item[1],
        )
    else:
        largest_component, largest_us = "unmeasured", -1.0
    steady_components = {
        "log_s": steady_log_s_us,
        "nms": steady_nms_us,
        "cross": steady_cross_us,
        "topk": steady_topk_us,
    }
    measured_steady_components = {
        key: value for key, value in steady_components.items() if value >= 0.0
    }
    if measured_steady_components:
        largest_steady_component, largest_steady_us = max(
            measured_steady_components.items(),
            key=lambda item: item[1],
        )
    else:
        largest_steady_component, largest_steady_us = "unmeasured", -1.0
    warmup_topk_values = [
        _as_float(record.get("us_topk"), -1.0) for record in warmup_records
    ]
    warmup_topk_values = [value for value in warmup_topk_values if value >= 0.0]
    warmup_cold_topk_us = max(warmup_topk_values) if warmup_topk_values else -1.0
    return {
        "selector_pipeline_cpu_profile_total_sample_count": int(len(all_records)),
        "selector_pipeline_cpu_profile_measurement_sample_count": int(
            len(measurement_records)
        ),
        "selector_pipeline_cpu_profile_warmup_sample_count": int(
            len(warmup_records)
        ),
        "selector_pipeline_cpu_profile_phase_filter": (
            "measurement" if measurement_records else "all"
        ),
        "selector_pipeline_cpu_profile_sample_count": int(sample_count),
        "selector_pipeline_cpu_log_s_us_avg": float(log_s_us),
        "selector_pipeline_cpu_nms_us_avg": float(nms_us),
        "selector_pipeline_cpu_cross_us_avg": float(cross_us),
        "selector_pipeline_cpu_topk_us_avg": float(topk_us),
        "selector_pipeline_cpu_total_us_avg": float(total_us),
        "selector_pipeline_cpu_steady_sample_count": int(len(steady_records)),
        "selector_pipeline_cpu_log_s_us_steady_avg_excluding_first": float(
            steady_log_s_us
        ),
        "selector_pipeline_cpu_nms_us_steady_avg_excluding_first": float(
            steady_nms_us
        ),
        "selector_pipeline_cpu_cross_us_steady_avg_excluding_first": float(
            steady_cross_us
        ),
        "selector_pipeline_cpu_topk_us_steady_avg_excluding_first": float(
            steady_topk_us
        ),
        "selector_pipeline_cpu_total_us_steady_avg_excluding_first": float(
            steady_total_us
        ),
        "largest_pipeline_cpu_component": largest_component,
        "largest_pipeline_cpu_component_us": float(largest_us),
        "largest_steady_pipeline_cpu_component": largest_steady_component,
        "largest_steady_pipeline_cpu_component_us": float(largest_steady_us),
        "pipeline_cpu_next_action": _deadline_v2_pipeline_cpu_stage_action(
            largest_component
        ),
        "steady_pipeline_cpu_next_action": _deadline_v2_pipeline_cpu_stage_action(
            largest_steady_component
        ),
        "selector_pipeline_cpu_warmup_cold_topk_us_max": float(
            warmup_cold_topk_us
        ),
        **_pipeline_cpu_bucket_breakdown(records),
    }


def _deadline_v2_resolve_selector_next_action(
    *,
    largest_component: str,
    largest_pure_preproc_component: str,
    key_norms_next_action: str,
    largest_deferred_selector_cpu_component: str = "unmeasured",
    deferred_selector_cpu_next_action: str = "enable_deferred_selector_profile_detail",
) -> str:
    if largest_component == "key_norms":
        return key_norms_next_action
    if largest_component == "pure_preproc":
        if largest_pure_preproc_component in {"bounds", "pipeline"}:
            return _deadline_v2_selector_stage_action(largest_pure_preproc_component)
    if (
        largest_component == "unmeasured"
        and largest_deferred_selector_cpu_component != "unmeasured"
    ):
        return deferred_selector_cpu_next_action
    return _deadline_v2_selector_stage_action(largest_component)


def _deadline_v2_selector_compute_breakdown(
    refresh_profile: list[dict[str, Any]],
    selector_pipeline_cpu_profile: list[dict[str, Any]] | None = None,
) -> dict[str, object]:
    stage_gpu_ms: dict[str, float] = {}
    for stage in SELECTOR_GPU_PROFILE_STAGES:
        value = _refresh_work_item_avg(
            refresh_profile,
            f"refresh_{stage}_gpu_ms",
        )
        if value < 0.0:
            value = _refresh_work_item_avg(
                refresh_profile,
                f"async_producer_{stage}_gpu_ms",
            )
        stage_gpu_ms[stage] = value

    selector_gpu_ms = stage_gpu_ms["selector"]
    key_norms_gpu_ms = stage_gpu_ms["key_norms"]
    key_norms_preproc_gpu_ms = stage_gpu_ms["key_norms_preproc"]
    key_norms_h2d_gpu_ms = stage_gpu_ms["key_norms_h2d"]
    key_norms_delta_gpu_ms = stage_gpu_ms["key_norms_delta"]
    key_norms_pack_gpu_ms = stage_gpu_ms["key_norms_pack"]
    key_norms_delta_total_tokens_avg = _deadline_async_producer_counter_avg(
        refresh_profile,
        "deadline_async_producer_key_norms_delta_total_tokens_total",
    )
    if key_norms_delta_total_tokens_avg < 0.0:
        key_norms_delta_total_tokens_avg = _refresh_work_item_avg(
            refresh_profile, "refresh_key_norms_delta_total_tokens"
        )
    key_norms_delta_max_tokens_max = _refresh_work_item_max(
        refresh_profile, "refresh_key_norms_delta_max_tokens"
    )
    async_key_norms_delta_max_tokens_max = _max_us_from_records(
        refresh_profile,
        "deadline_async_producer_key_norms_delta_max_tokens_max",
    )
    if async_key_norms_delta_max_tokens_max >= 0.0:
        key_norms_delta_max_tokens_max = async_key_norms_delta_max_tokens_max
    key_norms_delta_layers_avg = _deadline_async_producer_counter_avg(
        refresh_profile,
        "deadline_async_producer_key_norms_delta_layers_total",
    )
    if key_norms_delta_layers_avg < 0.0:
        key_norms_delta_layers_avg = _refresh_work_item_avg(
            refresh_profile, "refresh_key_norms_delta_layers"
        )
    key_norms_cpu_ms = _ms_from_us(
        _refresh_work_item_avg(refresh_profile, "refresh_selector_key_norms_cpu_us")
    )
    key_norms_cpu_arena_ms = _ms_from_us(
        _refresh_work_item_avg(
            refresh_profile,
            "refresh_selector_key_norms_arena_cpu_us",
        )
    )
    key_norms_cpu_direct_ms = _ms_from_us(
        _refresh_work_item_avg(
            refresh_profile,
            "refresh_selector_key_norms_direct_cpu_us",
        )
    )
    key_norms_cpu_direct_prepare_ms = _ms_from_us(
        _refresh_work_item_avg(
            refresh_profile,
            "refresh_selector_key_norms_direct_prepare_cpu_us",
        )
    )
    key_norms_cpu_direct_launch_ms = _ms_from_us(
        _refresh_work_item_avg(
            refresh_profile,
            "refresh_selector_key_norms_direct_launch_cpu_us",
        )
    )
    key_norms_cpu_pack_ms = _ms_from_us(
        _refresh_work_item_avg(
            refresh_profile,
            "refresh_selector_key_norms_pack_cpu_us",
        )
    )
    key_norms_pre_h2d_gap_gpu_ms = _refresh_work_item_avg(
        refresh_profile,
        "refresh_key_norms_pre_h2d_gap_gpu_ms",
    )
    key_norms_h2d_to_delta_gap_gpu_ms = _refresh_work_item_avg(
        refresh_profile,
        "refresh_key_norms_h2d_to_delta_gap_gpu_ms",
    )
    key_norms_delta_to_pack_gap_gpu_ms = _refresh_work_item_avg(
        refresh_profile,
        "refresh_key_norms_delta_to_pack_gap_gpu_ms",
    )
    key_norms_post_pack_gap_gpu_ms = _refresh_work_item_avg(
        refresh_profile,
        "refresh_key_norms_post_pack_gap_gpu_ms",
    )
    key_norms_subcomponents = [
        key_norms_preproc_gpu_ms,
        key_norms_h2d_gpu_ms,
        key_norms_delta_gpu_ms,
        key_norms_pack_gpu_ms,
    ]
    key_norms_subcomponent_sum = sum(
        value for value in key_norms_subcomponents if value >= 0.0
    )
    key_norms_envelope_residual_ms = (
        max(0.0, key_norms_gpu_ms - key_norms_subcomponent_sum)
        if key_norms_gpu_ms >= 0.0
        else -1.0
    )
    key_norms_envelope_unexplained_after_cpu_ms = (
        max(0.0, key_norms_envelope_residual_ms - key_norms_cpu_ms)
        if key_norms_envelope_residual_ms >= 0.0 and key_norms_cpu_ms >= 0.0
        else -1.0
    )
    pure_preproc_gpu_ms = stage_gpu_ms["pure_preproc"]
    selector_bounds_gpu_ms = stage_gpu_ms["selector_bounds"]
    selector_pipeline_gpu_ms = stage_gpu_ms["selector_pipeline"]
    seq_full_gpu_ms = stage_gpu_ms["seq_full"]
    log_s_gpu_ms = stage_gpu_ms["log_s"]
    topk_gpu_ms = stage_gpu_ms["topk"]
    gather_gpu_ms = _refresh_work_item_avg(refresh_profile, "refresh_gather_gpu_ms")
    rebuild_gpu_ms = _refresh_work_item_avg(refresh_profile, "refresh_rebuild_gpu_ms")
    components = {
        "key_norms": key_norms_gpu_ms,
        "key_norms_preproc": key_norms_preproc_gpu_ms,
        "pure_preproc": pure_preproc_gpu_ms,
        "bounds": selector_bounds_gpu_ms,
        "pipeline": selector_pipeline_gpu_ms,
        "seq_full": seq_full_gpu_ms,
        "topk": topk_gpu_ms,
        "log_s": log_s_gpu_ms,
    }
    measured_components = {
        key: value for key, value in components.items() if value >= 0.0
    }
    if measured_components:
        largest_component, largest_ms = max(
            measured_components.items(),
            key=lambda item: item[1],
        )
    elif selector_gpu_ms >= 0.0:
        largest_component, largest_ms = "selector_total", selector_gpu_ms
    else:
        largest_component, largest_ms = "unmeasured", -1.0
    key_norms_components = {
        "key_norms_h2d": key_norms_h2d_gpu_ms,
        "key_norms_delta": key_norms_delta_gpu_ms,
        "key_norms_pack": key_norms_pack_gpu_ms,
        "key_norms_preproc": key_norms_preproc_gpu_ms,
        "key_norms_envelope_residual": key_norms_envelope_residual_ms,
    }
    measured_key_norms_components = {
        key: value for key, value in key_norms_components.items() if value >= 0.0
    }
    if measured_key_norms_components:
        largest_key_norms_component, largest_key_norms_ms = max(
            measured_key_norms_components.items(),
            key=lambda item: item[1],
        )
    else:
        largest_key_norms_component, largest_key_norms_ms = "unmeasured", -1.0
    key_norms_envelope_gap_components = {
        "pre_h2d": key_norms_pre_h2d_gap_gpu_ms,
        "h2d_to_delta": key_norms_h2d_to_delta_gap_gpu_ms,
        "delta_to_pack": key_norms_delta_to_pack_gap_gpu_ms,
        "post_pack": key_norms_post_pack_gap_gpu_ms,
    }
    measured_key_norms_envelope_gap_components = {
        key: value
        for key, value in key_norms_envelope_gap_components.items()
        if value >= 0.0
    }
    if measured_key_norms_envelope_gap_components:
        largest_key_norms_envelope_gap, largest_key_norms_envelope_gap_ms = max(
            measured_key_norms_envelope_gap_components.items(),
            key=lambda item: item[1],
        )
    else:
        largest_key_norms_envelope_gap = "unmeasured"
        largest_key_norms_envelope_gap_ms = -1.0
    key_norms_next_action = _deadline_v2_key_norms_stage_action(
        largest_key_norms_component
    )
    if (
        largest_key_norms_component == "key_norms_envelope_residual"
        and largest_key_norms_envelope_gap != "unmeasured"
    ):
        key_norms_next_action = _deadline_v2_key_norms_envelope_gap_action(
            largest_key_norms_envelope_gap
        )
    pure_preproc_components = {
        "bounds": selector_bounds_gpu_ms,
        "pipeline": selector_pipeline_gpu_ms,
    }
    measured_pure_preproc_components = {
        key: value for key, value in pure_preproc_components.items() if value >= 0.0
    }
    if measured_pure_preproc_components:
        largest_pure_preproc_component, largest_pure_preproc_ms = max(
            measured_pure_preproc_components.items(),
            key=lambda item: item[1],
        )
    else:
        largest_pure_preproc_component, largest_pure_preproc_ms = "unmeasured", -1.0

    deferred_selector_count = _max_int_from_records(
        refresh_profile,
        "deadline_deferred_selector_compute_count",
    )

    def _deferred_selector_avg_ms(field: str) -> float:
        total_us = _max_us_from_records(
            refresh_profile,
            f"deadline_deferred_selector_{field}_cpu_us_total",
        )
        return _ms_from_us(_avg_from_total_count(total_us, deferred_selector_count))

    deferred_selector_cpu_total_ms = _deferred_selector_avg_ms("compute")
    deferred_selector_cpu_inner_ms = _deferred_selector_avg_ms("inner_compute")
    deferred_selector_cpu_key_norms_ms = _deferred_selector_avg_ms("key_norms")
    deferred_selector_cpu_select_ms = _deferred_selector_avg_ms("select")
    deferred_selector_cpu_validate_ms = _deferred_selector_avg_ms("validate")
    deferred_selector_cpu_stack_ms = _deferred_selector_avg_ms("stack")
    deferred_selector_cpu_post_ms = _deferred_selector_avg_ms("post")
    deferred_selector_cpu_wrapper_gap_ms = _deferred_selector_avg_ms("wrapper_gap")
    deferred_selector_cpu_components = {
        "key_norms": deferred_selector_cpu_key_norms_ms,
        "select": deferred_selector_cpu_select_ms,
        "validate": deferred_selector_cpu_validate_ms,
        "stack": deferred_selector_cpu_stack_ms,
        "post": deferred_selector_cpu_post_ms,
        "wrapper_gap": deferred_selector_cpu_wrapper_gap_ms,
    }
    measured_deferred_selector_cpu_components = {
        key: value
        for key, value in deferred_selector_cpu_components.items()
        if value >= 0.0
    }
    if measured_deferred_selector_cpu_components:
        (
            largest_deferred_selector_cpu_component,
            largest_deferred_selector_cpu_ms,
        ) = max(
            measured_deferred_selector_cpu_components.items(),
            key=lambda item: item[1],
        )
    elif deferred_selector_cpu_total_ms >= 0.0:
        largest_deferred_selector_cpu_component = "selector_total"
        largest_deferred_selector_cpu_ms = deferred_selector_cpu_total_ms
    else:
        largest_deferred_selector_cpu_component = "unmeasured"
        largest_deferred_selector_cpu_ms = -1.0
    deferred_selector_cpu_next_action = (
        _deadline_v2_deferred_selector_cpu_stage_action(
            largest_deferred_selector_cpu_component
        )
    )
    pipeline_cpu_breakdown = _deadline_v2_pipeline_cpu_breakdown(
        selector_pipeline_cpu_profile
    )
    return {
        "schema": "deadline_v2_selector_compute_breakdown_v1",
        "selector_gpu_ms": float(selector_gpu_ms),
        "key_norms_gpu_ms": float(key_norms_gpu_ms),
        "key_norms_preproc_gpu_ms": float(key_norms_preproc_gpu_ms),
        "key_norms_h2d_gpu_ms": float(key_norms_h2d_gpu_ms),
        "key_norms_delta_gpu_ms": float(key_norms_delta_gpu_ms),
        "key_norms_pack_gpu_ms": float(key_norms_pack_gpu_ms),
        "key_norms_delta_total_tokens_avg": float(key_norms_delta_total_tokens_avg),
        "key_norms_delta_max_tokens_max": float(key_norms_delta_max_tokens_max),
        "key_norms_delta_layers_avg": float(key_norms_delta_layers_avg),
        "key_norms_envelope_residual_gpu_ms": float(
            key_norms_envelope_residual_ms
        ),
        "key_norms_cpu_ms": float(key_norms_cpu_ms),
        "key_norms_cpu_arena_ms": float(key_norms_cpu_arena_ms),
        "key_norms_cpu_direct_ms": float(key_norms_cpu_direct_ms),
        "key_norms_cpu_direct_prepare_ms": float(
            key_norms_cpu_direct_prepare_ms
        ),
        "key_norms_cpu_direct_launch_ms": float(key_norms_cpu_direct_launch_ms),
        "key_norms_cpu_pack_ms": float(key_norms_cpu_pack_ms),
        "key_norms_envelope_unexplained_after_cpu_ms": float(
            key_norms_envelope_unexplained_after_cpu_ms
        ),
        "key_norms_pre_h2d_gap_gpu_ms": float(key_norms_pre_h2d_gap_gpu_ms),
        "key_norms_h2d_to_delta_gap_gpu_ms": float(
            key_norms_h2d_to_delta_gap_gpu_ms
        ),
        "key_norms_delta_to_pack_gap_gpu_ms": float(
            key_norms_delta_to_pack_gap_gpu_ms
        ),
        "key_norms_post_pack_gap_gpu_ms": float(key_norms_post_pack_gap_gpu_ms),
        "largest_key_norms_envelope_gap": largest_key_norms_envelope_gap,
        "largest_key_norms_envelope_gap_ms": float(
            largest_key_norms_envelope_gap_ms
        ),
        "key_norms_envelope_gap_next_action": (
            _deadline_v2_key_norms_envelope_gap_action(
                largest_key_norms_envelope_gap
            )
        ),
        "largest_key_norms_component": largest_key_norms_component,
        "largest_key_norms_component_ms": float(largest_key_norms_ms),
        "key_norms_next_action": key_norms_next_action,
        "pure_preproc_gpu_ms": float(pure_preproc_gpu_ms),
        "selector_bounds_gpu_ms": float(selector_bounds_gpu_ms),
        "selector_pipeline_gpu_ms": float(selector_pipeline_gpu_ms),
        "largest_pure_preproc_component": largest_pure_preproc_component,
        "largest_pure_preproc_component_ms": float(largest_pure_preproc_ms),
        "seq_full_gpu_ms": float(seq_full_gpu_ms),
        "log_s_gpu_ms": float(log_s_gpu_ms),
        "topk_gpu_ms": float(topk_gpu_ms),
        "gather_gpu_ms": float(gather_gpu_ms),
        "rebuild_gpu_ms": float(rebuild_gpu_ms),
        "largest_gpu_component": largest_component,
        "largest_gpu_component_ms": float(largest_ms),
        "deferred_selector_cpu_count": int(deferred_selector_count),
        "deferred_selector_cpu_total_ms": float(deferred_selector_cpu_total_ms),
        "deferred_selector_cpu_inner_compute_ms": float(
            deferred_selector_cpu_inner_ms
        ),
        "deferred_selector_cpu_key_norms_ms": float(
            deferred_selector_cpu_key_norms_ms
        ),
        "deferred_selector_cpu_select_ms": float(deferred_selector_cpu_select_ms),
        "deferred_selector_cpu_validate_ms": float(
            deferred_selector_cpu_validate_ms
        ),
        "deferred_selector_cpu_stack_ms": float(deferred_selector_cpu_stack_ms),
        "deferred_selector_cpu_post_ms": float(deferred_selector_cpu_post_ms),
        "deferred_selector_cpu_wrapper_gap_ms": float(
            deferred_selector_cpu_wrapper_gap_ms
        ),
        "largest_deferred_selector_cpu_component": (
            largest_deferred_selector_cpu_component
        ),
        "largest_deferred_selector_cpu_component_ms": float(
            largest_deferred_selector_cpu_ms
        ),
        "deferred_selector_cpu_next_action": deferred_selector_cpu_next_action,
        "next_action": _deadline_v2_resolve_selector_next_action(
            largest_component=largest_component,
            largest_pure_preproc_component=largest_pure_preproc_component,
            key_norms_next_action=key_norms_next_action,
            largest_deferred_selector_cpu_component=(
                largest_deferred_selector_cpu_component
            ),
            deferred_selector_cpu_next_action=deferred_selector_cpu_next_action,
        ),
        **pipeline_cpu_breakdown,
    }


def _deadline_v2_attribution_summary(
    *,
    metrics: dict[str, Any],
    refresh_profile: list[dict[str, Any]],
    timeline_summary: dict[str, Any] | None = None,
    route_summary: dict[str, object] | None = None,
    selector_pipeline_cpu_profile: list[dict[str, Any]] | None = None,
    hook_profile_summary: dict[str, object] | None = None,
) -> dict[str, object]:
    """Classify full-open overhead without changing selector/topk semantics."""
    timeline_summary = timeline_summary or {}
    route_summary = route_summary or {}
    hook_profile_summary = hook_profile_summary or {}
    wait_us_values = [
        _budget_us_from_sources(
            metrics, refresh_profile, timeline_summary, "first_decode_wait_us"
        ),
        _budget_us_from_sources(
            metrics,
            refresh_profile,
            timeline_summary,
            "blocked_by_unready_request_us",
        ),
        _counter_float_from_sources(
            metrics,
            refresh_profile,
            timeline_summary,
            route_summary,
            "visible_wait_us",
        ),
        _counter_float_from_sources(
            metrics,
            refresh_profile,
            timeline_summary,
            route_summary,
            "producer_deadline_wait_us",
        ),
    ]
    exposed_wait_us = max((value for value in wait_us_values if value >= 0.0), default=-1.0)

    work_item_records = _refresh_work_item_records(refresh_profile)
    steady_work_item_records = (
        work_item_records[1:] if len(work_item_records) > 1 else []
    )
    first_work_item = work_item_records[0] if work_item_records else {}
    selector_compute_us = _refresh_work_item_avg(
        refresh_profile, "refresh_selector_compute_cpu_us"
    )
    selector_post_us = _refresh_work_item_avg(
        refresh_profile, "refresh_selector_post_cpu_us"
    )
    selector_stack_us = _refresh_work_item_avg(
        refresh_profile, "refresh_selector_stack_cpu_us"
    )
    wrapper_gap_us = _refresh_selector_wrapper_gap_work_item_avg(refresh_profile)
    prepare_parts = [
        value
        for value in (selector_post_us, selector_stack_us, wrapper_gap_us)
        if value >= 0.0
    ]
    selector_prepare_us = sum(prepare_parts) if prepare_parts else -1.0
    writer_publish_us = _refresh_work_item_avg(
        refresh_profile, "selector_writer_boundary_cpu_us"
    )
    writer_enqueue_us = _refresh_work_item_avg(
        refresh_profile, "refresh_rebuild_enqueue_cpu_us"
    )
    writer_compact_us = _refresh_work_item_avg(
        refresh_profile, "refresh_rebuild_compact_cpu_us"
    )
    deferred_selector_compute_count = _max_int_from_records(
        refresh_profile,
        "deadline_deferred_selector_compute_count",
    )
    deferred_selector_compute_total_us = _max_us_from_records(
        refresh_profile,
        "deadline_deferred_selector_compute_cpu_us_total",
    )
    deferred_selector_compute_max_us = _max_us_from_records(
        refresh_profile,
        "deadline_deferred_selector_compute_cpu_us_max",
    )
    deferred_selector_compute_avg_us = (
        _avg_from_total_count(
            deferred_selector_compute_total_us,
            deferred_selector_compute_count,
        )
    )
    deferred_selector_inner_compute_total_us = _max_us_from_records(
        refresh_profile,
        "deadline_deferred_selector_inner_compute_cpu_us_total",
    )
    deferred_selector_inner_compute_max_us = _max_us_from_records(
        refresh_profile,
        "deadline_deferred_selector_inner_compute_cpu_us_max",
    )
    deferred_selector_inner_compute_avg_us = _avg_from_total_count(
        deferred_selector_inner_compute_total_us,
        deferred_selector_compute_count,
    )
    deferred_selector_stack_total_us = _max_us_from_records(
        refresh_profile,
        "deadline_deferred_selector_stack_cpu_us_total",
    )
    deferred_selector_stack_avg_us = _avg_from_total_count(
        deferred_selector_stack_total_us,
        deferred_selector_compute_count,
    )
    deferred_selector_validate_total_us = _max_us_from_records(
        refresh_profile,
        "deadline_deferred_selector_validate_cpu_us_total",
    )
    deferred_selector_validate_avg_us = _avg_from_total_count(
        deferred_selector_validate_total_us,
        deferred_selector_compute_count,
    )
    deferred_selector_select_total_us = _max_us_from_records(
        refresh_profile,
        "deadline_deferred_selector_select_cpu_us_total",
    )
    deferred_selector_select_avg_us = _avg_from_total_count(
        deferred_selector_select_total_us,
        deferred_selector_compute_count,
    )
    deferred_selector_post_total_us = _max_us_from_records(
        refresh_profile,
        "deadline_deferred_selector_post_cpu_us_total",
    )
    deferred_selector_post_avg_us = _avg_from_total_count(
        deferred_selector_post_total_us,
        deferred_selector_compute_count,
    )
    deferred_selector_wrapper_gap_total_us = _max_us_from_records(
        refresh_profile,
        "deadline_deferred_selector_wrapper_gap_cpu_us_total",
    )
    deferred_selector_wrapper_gap_max_us = _max_us_from_records(
        refresh_profile,
        "deadline_deferred_selector_wrapper_gap_cpu_us_max",
    )
    deferred_selector_wrapper_gap_avg_us = _avg_from_total_count(
        deferred_selector_wrapper_gap_total_us,
        deferred_selector_compute_count,
    )
    deferred_selector_key_norms_total_us = _max_us_from_records(
        refresh_profile,
        "deadline_deferred_selector_key_norms_cpu_us_total",
    )
    deferred_selector_key_norms_max_us = _max_us_from_records(
        refresh_profile,
        "deadline_deferred_selector_key_norms_cpu_us_max",
    )
    deferred_selector_key_norms_avg_us = (
        _avg_from_total_count(
            deferred_selector_key_norms_total_us,
            deferred_selector_compute_count,
        )
    )
    deferred_selector_key_norms_arena_avg_us = _avg_from_total_count(
        _max_us_from_records(
            refresh_profile,
            "deadline_deferred_selector_key_norms_arena_cpu_us_total",
        ),
        deferred_selector_compute_count,
    )
    deferred_selector_key_norms_direct_avg_us = _avg_from_total_count(
        _max_us_from_records(
            refresh_profile,
            "deadline_deferred_selector_key_norms_direct_cpu_us_total",
        ),
        deferred_selector_compute_count,
    )
    deferred_selector_key_norms_direct_prepare_avg_us = _avg_from_total_count(
        _max_us_from_records(
            refresh_profile,
            "deadline_deferred_selector_key_norms_direct_prepare_cpu_us_total",
        ),
        deferred_selector_compute_count,
    )
    deferred_selector_key_norms_direct_launch_avg_us = _avg_from_total_count(
        _max_us_from_records(
            refresh_profile,
            "deadline_deferred_selector_key_norms_direct_launch_cpu_us_total",
        ),
        deferred_selector_compute_count,
    )
    deferred_selector_key_norms_pack_avg_us = _avg_from_total_count(
        _max_us_from_records(
            refresh_profile,
            "deadline_deferred_selector_key_norms_pack_cpu_us_total",
        ),
        deferred_selector_compute_count,
    )
    async_producer_body_count = _max_int_from_records(
        refresh_profile,
        "deadline_async_producer_body_count",
    )
    async_producer_body_total_us = _max_us_from_records(
        refresh_profile,
        "deadline_async_producer_body_cpu_us_total",
    )
    async_producer_body_max_us = _max_us_from_records(
        refresh_profile,
        "deadline_async_producer_body_cpu_us_max",
    )
    async_producer_body_avg_us = _avg_from_total_count(
        async_producer_body_total_us,
        async_producer_body_count,
    )
    async_producer_selector_count = _max_int_from_records(
        refresh_profile,
        "deadline_async_producer_selector_count",
    )
    async_producer_selector_total_us = _max_us_from_records(
        refresh_profile,
        "deadline_async_producer_selector_cpu_us_total",
    )
    async_producer_selector_max_us = _max_us_from_records(
        refresh_profile,
        "deadline_async_producer_selector_cpu_us_max",
    )
    async_producer_selector_avg_us = _avg_from_total_count(
        async_producer_selector_total_us,
        async_producer_selector_count,
    )
    async_producer_writer_count = _max_int_from_records(
        refresh_profile,
        "deadline_async_producer_writer_count",
    )
    async_producer_writer_total_us = _max_us_from_records(
        refresh_profile,
        "deadline_async_producer_writer_cpu_us_total",
    )
    async_producer_writer_max_us = _max_us_from_records(
        refresh_profile,
        "deadline_async_producer_writer_cpu_us_max",
    )
    async_producer_writer_avg_us = _avg_from_total_count(
        async_producer_writer_total_us,
        async_producer_writer_count,
    )
    async_producer_graph_replay_count = _max_int_from_records(
        refresh_profile,
        "deadline_async_producer_graph_replay_count",
    )
    async_producer_graph_replay_total_us = _max_us_from_records(
        refresh_profile,
        "deadline_async_producer_graph_replay_cpu_us_total",
    )
    async_producer_graph_replay_max_us = _max_us_from_records(
        refresh_profile,
        "deadline_async_producer_graph_replay_cpu_us_max",
    )
    async_producer_graph_replay_avg_us = _avg_from_total_count(
        async_producer_graph_replay_total_us,
        async_producer_graph_replay_count,
    )
    if async_producer_graph_replay_count <= 0:
        async_producer_graph_replay_total_us = 0.0
        async_producer_graph_replay_max_us = -1.0
        async_producer_graph_replay_avg_us = -1.0
    graph_replay_stage_stats: dict[str, tuple[int, float, float, float]] = {}
    for stage_name in (
        "stage_selector_inputs",
        "prepare_writer",
        "stage_lens",
        "prepare_events",
        "graph",
    ):
        count = _max_int_from_records(
            refresh_profile,
            f"deadline_async_producer_graph_replay_{stage_name}_count",
        )
        total_us = _max_us_from_records(
            refresh_profile,
            f"deadline_async_producer_graph_replay_{stage_name}_cpu_us_total",
        )
        max_us = _max_us_from_records(
            refresh_profile,
            f"deadline_async_producer_graph_replay_{stage_name}_cpu_us_max",
        )
        avg_us = _avg_from_total_count(total_us, count)
        graph_replay_stage_stats[stage_name] = (
            int(count),
            float(total_us),
            float(max_us),
            float(avg_us),
        )
    if async_producer_graph_replay_count <= 0:
        graph_replay_stage_stats = {
            stage_name: (0, 0.0, -1.0, -1.0)
            for stage_name in graph_replay_stage_stats
        }
    async_producer_graph_capture_count = _max_int_from_records(
        refresh_profile,
        "deadline_async_producer_graph_capture_count",
    )
    async_producer_graph_capture_total_us = _max_us_from_records(
        refresh_profile,
        "deadline_async_producer_graph_capture_cpu_us_total",
    )
    async_producer_graph_capture_max_us = _max_us_from_records(
        refresh_profile,
        "deadline_async_producer_graph_capture_cpu_us_max",
    )
    async_producer_graph_capture_avg_us = _avg_from_total_count(
        async_producer_graph_capture_total_us,
        async_producer_graph_capture_count,
    )
    if async_producer_graph_capture_count <= 0:
        async_producer_graph_capture_total_us = 0.0
        async_producer_graph_capture_max_us = -1.0
        async_producer_graph_capture_avg_us = -1.0
    async_producer_result_precomputed_count = _max_int_from_records(
        refresh_profile,
        "deadline_async_producer_result_precomputed_count",
    )
    async_producer_body_gap_total_us = -1.0
    async_producer_body_gap_avg_us = -1.0
    if async_producer_body_total_us >= 0.0:
        accounted_us = sum(
            value
            for value in (
                async_producer_selector_total_us,
                async_producer_writer_total_us,
                async_producer_graph_replay_total_us,
                async_producer_graph_capture_total_us,
            )
            if value >= 0.0
        )
        async_producer_body_gap_total_us = max(
            0.0,
            async_producer_body_total_us - accounted_us,
        )
        async_producer_body_gap_avg_us = _avg_from_total_count(
            async_producer_body_gap_total_us,
            async_producer_body_count,
        )
    steady_selector_compute_us = _records_avg(
        steady_work_item_records, "refresh_selector_compute_cpu_us"
    )
    steady_selector_post_us = _records_avg(
        steady_work_item_records, "refresh_selector_post_cpu_us"
    )
    steady_selector_stack_us = _records_avg(
        steady_work_item_records, "refresh_selector_stack_cpu_us"
    )
    steady_wrapper_gap_us = _selector_wrapper_gap_avg(steady_work_item_records)
    steady_prepare_parts = [
        value
        for value in (
            steady_selector_post_us,
            steady_selector_stack_us,
            steady_wrapper_gap_us,
        )
        if value >= 0.0
    ]
    steady_selector_prepare_us = (
        sum(steady_prepare_parts) if steady_prepare_parts else -1.0
    )
    steady_writer_publish_us = _records_avg(
        steady_work_item_records, "selector_writer_boundary_cpu_us"
    )
    selector_compute_breakdown = _deadline_v2_selector_compute_breakdown(
        refresh_profile,
        selector_pipeline_cpu_profile=selector_pipeline_cpu_profile,
    )
    async_producer_body_gpu_ms = _refresh_work_item_avg(
        refresh_profile, "async_producer_body_gpu_ms"
    )
    async_producer_selector_gpu_ms = _refresh_work_item_avg(
        refresh_profile, "async_producer_selector_gpu_ms"
    )
    async_producer_writer_gpu_ms = _refresh_work_item_avg(
        refresh_profile, "async_producer_writer_gpu_ms"
    )
    decode_p50_us = _decode_metric(metrics, "decode_p50_us")
    decode_p95_us = _decode_metric(metrics, "decode_p95_us")
    decode_jitter_us = (
        max(0.0, decode_p95_us - decode_p50_us)
        if decode_p50_us >= 0.0 and decode_p95_us >= 0.0
        else -1.0
    )

    candidates_ms = {
        "wait": _ms_from_us(exposed_wait_us),
        "writer_publish": _ms_from_us(writer_publish_us),
        "selector_prepare": _ms_from_us(selector_prepare_us),
        "selector_compute": _ms_from_us(selector_compute_us),
        "deferred_selector_compute": _ms_from_us(
            deferred_selector_compute_avg_us
        ),
        "async_producer_selector": _ms_from_us(async_producer_selector_avg_us),
        "async_producer_writer": _ms_from_us(async_producer_writer_avg_us),
        "async_producer_body_gap": _ms_from_us(async_producer_body_gap_avg_us),
        "async_producer_graph_capture": _ms_from_us(
            async_producer_graph_capture_avg_us
        ),
        "async_producer_graph_replay": _ms_from_us(
            async_producer_graph_replay_avg_us
        ),
        "wrapper_jitter": _ms_from_us(wrapper_gap_us),
        "decode_jitter": _ms_from_us(decode_jitter_us),
    }
    measured_candidates = {
        key: value for key, value in candidates_ms.items() if value >= 0.0
    }
    steady_candidates_ms = (
        {
            "writer_publish": _ms_from_us(steady_writer_publish_us),
            "selector_prepare": _ms_from_us(steady_selector_prepare_us),
            "selector_compute": _ms_from_us(steady_selector_compute_us),
            "deferred_selector_compute": _ms_from_us(
                deferred_selector_compute_avg_us
            ),
            "async_producer_selector": _ms_from_us(async_producer_selector_avg_us),
            "async_producer_writer": _ms_from_us(async_producer_writer_avg_us),
            "async_producer_body_gap": _ms_from_us(async_producer_body_gap_avg_us),
            "async_producer_graph_capture": _ms_from_us(
                async_producer_graph_capture_avg_us
            ),
            "async_producer_graph_replay": _ms_from_us(
                async_producer_graph_replay_avg_us
            ),
            "wrapper_jitter": _ms_from_us(steady_wrapper_gap_us),
            "decode_jitter": _ms_from_us(decode_jitter_us),
        }
        if steady_work_item_records
        else {
            "writer_publish": -1.0,
            "selector_prepare": -1.0,
            "selector_compute": -1.0,
            "deferred_selector_compute": -1.0,
            "async_producer_selector": -1.0,
            "async_producer_writer": -1.0,
            "async_producer_body_gap": -1.0,
            "async_producer_graph_capture": -1.0,
            "async_producer_graph_replay": -1.0,
            "wrapper_jitter": -1.0,
            "decode_jitter": -1.0,
        }
    )
    steady_measured_candidates = {
        key: value for key, value in steady_candidates_ms.items() if value >= 0.0
    }
    if not measured_candidates:
        primary_bottleneck = "unmeasured"
        primary_ms = -1.0
    else:
        primary_bottleneck, primary_ms = max(
            measured_candidates.items(), key=lambda item: item[1]
        )
        if primary_ms < 0.05:
            primary_bottleneck = "none"
    if not steady_measured_candidates:
        steady_primary_bottleneck = "unmeasured"
        steady_primary_ms = -1.0
    else:
        steady_primary_bottleneck, steady_primary_ms = max(
            steady_measured_candidates.items(), key=lambda item: item[1]
        )
        if steady_primary_ms < 0.05:
            steady_primary_bottleneck = "none"

    enqueued_records = _refresh_rebuild_enqueued_records(refresh_profile)
    has_wait_signal = exposed_wait_us >= 0.0
    if enqueued_records and has_wait_signal:
        confidence = "measured"
    elif enqueued_records:
        confidence = "profile_only"
    elif work_item_records:
        confidence = "profile_only"
    elif has_wait_signal:
        confidence = "timeline_only"
    elif measured_candidates:
        confidence = "partial"
    else:
        confidence = "unmeasured"

    drain_finish = _max_int_from_records(
        refresh_profile, "deadline_rebuild_drain_finish_count"
    )
    partial_finish = _max_int_from_records(
        refresh_profile, "deadline_rebuild_partial_finish_count"
    )
    producer_work_records = [
        record
        for record in work_item_records
        if _as_int(record.get("producer_work_target_layer_start"), -1) >= 0
        or _as_int(record.get("producer_work_ready_epoch"), -1) >= 0
    ]
    sentence_trigger_admission_coalesced_count = _max_int_from_records(
        refresh_profile,
        "sentence_trigger_admission_coalesced_total",
        default=0,
    )
    sentence_trigger_admission_coalesced_interval_pending_count = (
        _max_int_from_records(
            refresh_profile,
            "sentence_trigger_admission_coalesced_interval_pending_total",
            default=0,
        )
    )
    sentence_trigger_admission_coalesced_inflight_count = _max_int_from_records(
        refresh_profile,
        "sentence_trigger_admission_coalesced_inflight_total",
        default=0,
    )
    sentence_trigger_admission_coalesced_pending_rebuild_count = (
        _max_int_from_records(
            refresh_profile,
            "sentence_trigger_admission_coalesced_pending_rebuild_total",
            default=0,
        )
    )
    sentence_trigger_admission_dropped_finished_count = _max_int_from_records(
        refresh_profile,
        "sentence_trigger_admission_dropped_finished_total",
        default=0,
    )
    refresh_coalesce_skipped_existing_pending_count = _max_int_from_records(
        refresh_profile,
        "refresh_coalesce_skipped_existing_pending_total",
        default=0,
    )
    refresh_rebuild_coalesced_count = _sum_int(
        refresh_profile,
        "refresh_rebuild_coalesced_count",
    )
    queue_depth_max = _max_int_from_records(
        refresh_profile, "refresh_rebuild_pending_queue_size"
    )
    producer_work_inline_admission_count = _count_string_from_records(
        producer_work_records,
        "producer_work_admission_reason",
        "inline_refresh",
    )
    producer_work_enqueued_admission_count = 0
    producer_work_deferred_selector_admission_count = 0
    producer_work_decode_step_min = _min_nonnegative_int_from_records(
        producer_work_records,
        "producer_work_decode_step_min",
    )
    producer_work_decode_step_max = _max_int_from_records(
        producer_work_records,
        "producer_work_decode_step_max",
        default=-1,
    )
    producer_work_decode_steps_observed = _producer_decode_steps_from_records(
        producer_work_records
    )
    hook_producer_work_items = _as_int(
        hook_profile_summary.get(
            "refresh_stage_producer_work_items_with_deadline_total"
        ),
        0,
    )
    hook_producer_step_min = _as_int(
        hook_profile_summary.get("refresh_stage_producer_work_decode_step_min"),
        -1,
    )
    hook_producer_step_max = _as_int(
        hook_profile_summary.get("refresh_stage_producer_work_decode_step_max"),
        -1,
    )
    hook_producer_steps = [
        int(value)
        for value in (
            hook_profile_summary.get(
                "refresh_stage_producer_work_decode_steps_observed",
                [],
            )
            or []
        )
        if _as_int(value, -1) >= 0
    ]
    hook_producer_slack = _as_int(
        hook_profile_summary.get(
            "refresh_stage_producer_work_deadline_slack_steps_min"
        ),
        -1,
    )
    if producer_work_decode_step_min < 0 and hook_producer_step_min >= 0:
        producer_work_decode_step_min = int(hook_producer_step_min)
    if producer_work_decode_step_max < 0 and hook_producer_step_max >= 0:
        producer_work_decode_step_max = int(hook_producer_step_max)
    if hook_producer_steps:
        producer_work_decode_steps_observed = sorted(
            set(producer_work_decode_steps_observed).union(hook_producer_steps)
        )
    if (
        not producer_work_decode_steps_observed
        and producer_work_decode_step_min >= 0
        and producer_work_decode_step_max >= producer_work_decode_step_min
    ):
        producer_work_decode_steps_observed = list(
            range(
                int(producer_work_decode_step_min),
                int(producer_work_decode_step_max) + 1,
            )
        )
    producer_work_deadline_slack_steps_min = _min_nonnegative_int_from_records(
        producer_work_records, "producer_work_deadline_slack_steps"
    )
    if producer_work_deadline_slack_steps_min < 0 and hook_producer_slack >= 0:
        producer_work_deadline_slack_steps_min = int(hook_producer_slack)
    pending_rebuild_drain_submit_count = _max_int_from_records(
        refresh_profile,
        "deadline_rebuild_drain_submit_count",
    )
    pending_rebuild_drain_submit_decode_steps_observed = (
        _int_list_values_from_records(
            refresh_profile,
            "deadline_rebuild_drain_submit_decode_steps",
        )
    )
    decode_tail_jitter_breakdown = _deadline_v2_decode_tail_jitter_breakdown(
        metrics,
        queue_depth_max=int(queue_depth_max),
        enqueued_count=int(producer_work_enqueued_admission_count),
        deferred_selector_count=int(producer_work_deferred_selector_admission_count),
        producer_step_min=int(producer_work_decode_step_min),
        producer_step_max=int(producer_work_decode_step_max),
        producer_steps=producer_work_decode_steps_observed,
        drain_submit_steps=pending_rebuild_drain_submit_decode_steps_observed,
    )
    return {
        "schema": "deadline_v2_attribution_v1",
        "primary_bottleneck": primary_bottleneck,
        "primary_ms": float(primary_ms),
        "next_action": _deadline_v2_primary_action(primary_bottleneck),
        "steady_primary_bottleneck": steady_primary_bottleneck,
        "steady_primary_ms": float(steady_primary_ms),
        "steady_next_action": _deadline_v2_primary_action(
            steady_primary_bottleneck
        ),
        "confidence": confidence,
        "exposed_wait_ms": candidates_ms["wait"],
        "selector_prepare_ms": candidates_ms["selector_prepare"],
        "selector_compute_ms": candidates_ms["selector_compute"],
        "deferred_selector_compute_avg_ms": candidates_ms[
            "deferred_selector_compute"
        ],
        "deferred_selector_compute_max_ms": _ms_from_us(
            deferred_selector_compute_max_us
        ),
        "deferred_selector_compute_count": int(deferred_selector_compute_count),
        "deferred_selector_inner_compute_avg_ms": _ms_from_us(
            deferred_selector_inner_compute_avg_us
        ),
        "deferred_selector_inner_compute_max_ms": _ms_from_us(
            deferred_selector_inner_compute_max_us
        ),
        "deferred_selector_stack_avg_ms": _ms_from_us(
            deferred_selector_stack_avg_us
        ),
        "deferred_selector_validate_avg_ms": _ms_from_us(
            deferred_selector_validate_avg_us
        ),
        "deferred_selector_key_norms_avg_ms": _ms_from_us(
            deferred_selector_key_norms_avg_us
        ),
        "deferred_selector_key_norms_max_ms": _ms_from_us(
            deferred_selector_key_norms_max_us
        ),
        "deferred_selector_key_norms_arena_avg_ms": _ms_from_us(
            deferred_selector_key_norms_arena_avg_us
        ),
        "deferred_selector_key_norms_direct_avg_ms": _ms_from_us(
            deferred_selector_key_norms_direct_avg_us
        ),
        "deferred_selector_key_norms_direct_prepare_avg_ms": _ms_from_us(
            deferred_selector_key_norms_direct_prepare_avg_us
        ),
        "deferred_selector_key_norms_direct_launch_avg_ms": _ms_from_us(
            deferred_selector_key_norms_direct_launch_avg_us
        ),
        "deferred_selector_key_norms_pack_avg_ms": _ms_from_us(
            deferred_selector_key_norms_pack_avg_us
        ),
        "deferred_selector_select_avg_ms": _ms_from_us(
            deferred_selector_select_avg_us
        ),
        "deferred_selector_post_avg_ms": _ms_from_us(deferred_selector_post_avg_us),
        "deferred_selector_wrapper_gap_avg_ms": _ms_from_us(
            deferred_selector_wrapper_gap_avg_us
        ),
        "deferred_selector_wrapper_gap_max_ms": _ms_from_us(
            deferred_selector_wrapper_gap_max_us
        ),
        "async_producer_body_count": int(async_producer_body_count),
        "async_producer_body_avg_ms": _ms_from_us(async_producer_body_avg_us),
        "async_producer_body_max_ms": _ms_from_us(async_producer_body_max_us),
        "async_producer_body_gap_avg_ms": _ms_from_us(
            async_producer_body_gap_avg_us
        ),
        "async_producer_selector_count": int(async_producer_selector_count),
        "async_producer_selector_avg_ms": _ms_from_us(
            async_producer_selector_avg_us
        ),
        "async_producer_selector_max_ms": _ms_from_us(
            async_producer_selector_max_us
        ),
        "async_producer_writer_count": int(async_producer_writer_count),
        "async_producer_writer_avg_ms": _ms_from_us(async_producer_writer_avg_us),
        "async_producer_writer_max_ms": _ms_from_us(async_producer_writer_max_us),
        "async_producer_body_gpu_ms": float(async_producer_body_gpu_ms),
        "async_producer_selector_gpu_ms": float(async_producer_selector_gpu_ms),
        "async_producer_writer_gpu_ms": float(async_producer_writer_gpu_ms),
        "async_producer_graph_replay_count": int(
            async_producer_graph_replay_count
        ),
        "async_producer_graph_replay_avg_ms": _ms_from_us(
            async_producer_graph_replay_avg_us
        ),
        "async_producer_graph_replay_max_ms": _ms_from_us(
            async_producer_graph_replay_max_us
        ),
        "async_producer_graph_replay_stage_selector_inputs_count": int(
            graph_replay_stage_stats["stage_selector_inputs"][0]
        ),
        "async_producer_graph_replay_stage_selector_inputs_avg_ms": _ms_from_us(
            graph_replay_stage_stats["stage_selector_inputs"][3]
        ),
        "async_producer_graph_replay_stage_selector_inputs_max_ms": _ms_from_us(
            graph_replay_stage_stats["stage_selector_inputs"][2]
        ),
        "async_producer_graph_replay_prepare_writer_count": int(
            graph_replay_stage_stats["prepare_writer"][0]
        ),
        "async_producer_graph_replay_prepare_writer_avg_ms": _ms_from_us(
            graph_replay_stage_stats["prepare_writer"][3]
        ),
        "async_producer_graph_replay_prepare_writer_max_ms": _ms_from_us(
            graph_replay_stage_stats["prepare_writer"][2]
        ),
        "async_producer_graph_replay_stage_lens_count": int(
            graph_replay_stage_stats["stage_lens"][0]
        ),
        "async_producer_graph_replay_stage_lens_avg_ms": _ms_from_us(
            graph_replay_stage_stats["stage_lens"][3]
        ),
        "async_producer_graph_replay_stage_lens_max_ms": _ms_from_us(
            graph_replay_stage_stats["stage_lens"][2]
        ),
        "async_producer_graph_replay_prepare_events_count": int(
            graph_replay_stage_stats["prepare_events"][0]
        ),
        "async_producer_graph_replay_prepare_events_avg_ms": _ms_from_us(
            graph_replay_stage_stats["prepare_events"][3]
        ),
        "async_producer_graph_replay_prepare_events_max_ms": _ms_from_us(
            graph_replay_stage_stats["prepare_events"][2]
        ),
        "async_producer_graph_replay_graph_count": int(
            graph_replay_stage_stats["graph"][0]
        ),
        "async_producer_graph_replay_graph_avg_ms": _ms_from_us(
            graph_replay_stage_stats["graph"][3]
        ),
        "async_producer_graph_replay_graph_max_ms": _ms_from_us(
            graph_replay_stage_stats["graph"][2]
        ),
        "async_producer_graph_capture_count": int(
            async_producer_graph_capture_count
        ),
        "async_producer_graph_capture_avg_ms": _ms_from_us(
            async_producer_graph_capture_avg_us
        ),
        "async_producer_graph_capture_max_ms": _ms_from_us(
            async_producer_graph_capture_max_us
        ),
        "async_producer_result_precomputed_count": int(
            async_producer_result_precomputed_count
        ),
        "writer_publish_ms": candidates_ms["writer_publish"],
        "steady_selector_prepare_ms": steady_candidates_ms["selector_prepare"],
        "steady_selector_compute_ms": steady_candidates_ms["selector_compute"],
        "steady_writer_publish_ms": steady_candidates_ms["writer_publish"],
        "steady_async_producer_selector_ms": steady_candidates_ms[
            "async_producer_selector"
        ],
        "steady_async_producer_writer_ms": steady_candidates_ms[
            "async_producer_writer"
        ],
        "steady_async_producer_body_gap_ms": steady_candidates_ms[
            "async_producer_body_gap"
        ],
        "steady_async_producer_graph_capture_ms": steady_candidates_ms[
            "async_producer_graph_capture"
        ],
        "steady_async_producer_graph_replay_ms": steady_candidates_ms[
            "async_producer_graph_replay"
        ],
        "steady_wrapper_jitter_ms": steady_candidates_ms["wrapper_jitter"],
        "first_work_item_refresh_total_ms": _ms_from_us(
            _as_float(first_work_item.get("refresh_total_cpu_us"), -1.0)
        ),
        "first_work_item_selector_compute_ms": _ms_from_us(
            _as_float(first_work_item.get("refresh_selector_compute_cpu_us"), -1.0)
        ),
        "first_work_item_writer_publish_ms": _ms_from_us(
            _as_float(first_work_item.get("selector_writer_boundary_cpu_us"), -1.0)
        ),
        "first_work_item_key_norms_delta_gpu_ms": _as_float(
            first_work_item.get("refresh_key_norms_delta_gpu_ms"), -1.0
        ),
        "writer_enqueue_ms": _ms_from_us(writer_enqueue_us),
        "writer_compact_ms": _ms_from_us(writer_compact_us),
        "wrapper_jitter_ms": candidates_ms["wrapper_jitter"],
        "decode_jitter_ms": candidates_ms["decode_jitter"],
        "decode_tail_jitter_breakdown": decode_tail_jitter_breakdown,
        "selector_compute_breakdown": selector_compute_breakdown,
        "pending_drop_finished_count": int(
            _max_int_from_records(refresh_profile, "deadline_rebuild_drop_finished_count")
        ),
        "pending_drain_finish_count": int(drain_finish),
        "pending_partial_finish_count": int(partial_finish),
        "pending_drain_total_count": int(drain_finish + partial_finish),
        "pending_rebuild_drain_submit_count": int(
            pending_rebuild_drain_submit_count
        ),
        "pending_rebuild_drain_submit_decode_steps_observed": (
            pending_rebuild_drain_submit_decode_steps_observed
        ),
        "refresh_rebuild_coalesced_count": int(refresh_rebuild_coalesced_count),
        "queue_depth_max": int(queue_depth_max),
        "refresh_work_items_sampled": int(len(work_item_records)),
        "producer_work_items_with_deadline": int(
            max(len(producer_work_records), hook_producer_work_items)
        ),
        "producer_work_items_with_deadline_from_hook": int(hook_producer_work_items),
        "producer_work_target_layer_start_min": int(
            _min_nonnegative_int_from_records(
                producer_work_records, "producer_work_target_layer_start"
            )
        ),
        "producer_work_target_layer_end_max": int(
            _max_int_from_records(
                producer_work_records,
                "producer_work_target_layer_end",
                default=-1,
            )
        ),
        "producer_work_decode_step_min": int(
            producer_work_decode_step_min
        ),
        "producer_work_decode_step_max": int(
            producer_work_decode_step_max
        ),
        "producer_work_decode_steps_observed": producer_work_decode_steps_observed,
        "producer_work_deadline_slack_steps_min": int(
            producer_work_deadline_slack_steps_min
        ),
        "producer_work_inline_admission_count": int(
            producer_work_inline_admission_count
        ),
        "producer_work_enqueued_admission_count": int(
            producer_work_enqueued_admission_count
        ),
        "producer_work_deferred_selector_admission_count": int(
            producer_work_deferred_selector_admission_count
        ),
        "sentence_trigger_admission_coalesced_count": int(
            sentence_trigger_admission_coalesced_count
        ),
        "sentence_trigger_admission_coalesced_interval_pending_count": int(
            sentence_trigger_admission_coalesced_interval_pending_count
        ),
        "sentence_trigger_admission_coalesced_inflight_count": int(
            sentence_trigger_admission_coalesced_inflight_count
        ),
        "sentence_trigger_admission_coalesced_pending_rebuild_count": int(
            sentence_trigger_admission_coalesced_pending_rebuild_count
        ),
        "sentence_trigger_admission_dropped_finished_count": int(
            sentence_trigger_admission_dropped_finished_count
        ),
        "refresh_coalesce_skipped_existing_pending_count": int(
            refresh_coalesce_skipped_existing_pending_count
        ),
    }


def _read_one_shot_timeline(path: Path) -> list[dict[str, Any]]:
    return _read_trace_events(path)


def _timeline_budget_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    finalize_ts: list[int] = []
    wait_start_ts: list[int] = []
    wait_end_us: list[float] = []
    ready_publish_ts: list[int] = []
    aux_values: list[bool] = []
    expected_group_mask = -1
    submitted_group_mask = -1
    source_ready_event_generation = -1
    event_query_used: bool | None = None
    graph_wait_event_used_from_summary = False
    producer_final_event_recorded: bool | None = None
    capture_tap_visible_ms_or_unavailable_reason = "unavailable:not_measured"
    int_max_fields = {
        "lastn1_direct_count": -1,
        "gt1_reduce_count": -1,
        "gt1_scalar_fallback_count": -1,
        "bridge_token_count": -1,
        "producer_launch_step": -1,
        "producer_ready_step": -1,
    }
    int_sum_fields = {
        "prefill_selector_runs": 0,
        "prefill_rebuild_runs": 0,
        "writer_pointer_rebuild_count": 0,
        "writer_pointer_lookup_count": 0,
        "writer_cached_pointer_op_count": 0,
        "writer_vector_fallback_count": 0,
        "source_ready_recorded_after_pointer_publish_count": 0,
    }
    int_sum_seen = {key: False for key in int_sum_fields}
    float_max_fields = {
        "capture_postprocess_or_reduce_ms": -1.0,
        "rrp_publish_us": -1.0,
        "producer_final_event_gpu_wait_us": -1.0,
        "bridge_decode_p50_us": -1.0,
        "bridge_decode_p95_us": -1.0,
        "bridge_added_decode_cost_us": -1.0,
        "saved_visible_wait_us": -1.0,
    }
    bootstrap_full_kv_handoff = False
    timeline_seen = False
    for record in records:
        if record.get("event") != "one_shot_timeline":
            continue
        timeline_seen = True
        phase = str(record.get("phase", ""))
        timestamp_ns = _as_int(record.get("timestamp_ns"), -1)
        if (
            phase
            in (
                "finalize_event_record",
                "finalize_event_deferred_to_chunk_done",
                "deferred_producer_final_event_record",
            )
            and timestamp_ns >= 0
        ):
            finalize_ts.append(timestamp_ns)
        elif phase in ("bootstrap_wait_start", "producer_final_event_wait_start") and timestamp_ns >= 0:
            wait_start_ts.append(timestamp_ns)
        elif phase in ("bootstrap_wait_end", "producer_final_event_wait_end"):
            duration_us = _as_float(record.get("duration_us"), -1.0)
            if duration_us >= 0.0:
                wait_end_us.append(duration_us)
        elif phase == "producer_group_ready_publish" and timestamp_ns >= 0:
            ready_publish_ts.append(timestamp_ns)
        if "aux_stream_enabled" in record:
            aux_values.append(bool(record.get("aux_stream_enabled", False)))
        if "expected_group_mask" in record:
            mask = _as_int(record.get("expected_group_mask"), -1)
            if mask >= 0:
                expected_group_mask = mask if expected_group_mask < 0 else expected_group_mask | mask
        if "submitted_group_mask" in record:
            mask = _as_int(record.get("submitted_group_mask"), -1)
            if mask >= 0:
                submitted_group_mask = mask if submitted_group_mask < 0 else submitted_group_mask | mask
        if "source_ready_event_generation" in record:
            generation = _as_int(record.get("source_ready_event_generation"), -1)
            if generation >= 0:
                source_ready_event_generation = max(source_ready_event_generation, generation)
        if "event_query_used_for_publish" in record:
            if bool(record.get("event_query_used_for_publish", False)):
                event_query_used = True
            elif event_query_used is None:
                event_query_used = False
        if bool(record.get("graph_wait_event_used", False)):
            graph_wait_event_used_from_summary = True
        if "producer_final_event_recorded" in record:
            producer_final_event_recorded = bool(record.get("producer_final_event_recorded"))
        if "bootstrap_full_kv_handoff" in record:
            bootstrap_full_kv_handoff = bool(
                record.get("bootstrap_full_kv_handoff")
            )
        if "capture_tap_visible_ms_or_unavailable_reason" in record:
            capture_tap_visible_ms_or_unavailable_reason = str(
                record.get("capture_tap_visible_ms_or_unavailable_reason")
            )
        for key in int_max_fields:
            if key in record:
                int_max_fields[key] = max(
                    int(int_max_fields[key]),
                    _as_int(record.get(key), -1),
                )
        for key in int_sum_fields:
            if key in record:
                int_sum_seen[key] = True
                int_sum_fields[key] += max(0, _as_int(record.get(key), 0))
        for key in float_max_fields:
            if key in record:
                float_max_fields[key] = max(
                    float(float_max_fields[key]),
                    _as_float(record.get(key), -1.0),
                )
    prefill_to_first_decode_us = -1.0
    if finalize_ts and wait_start_ts:
        delta_us = float(min(wait_start_ts) - min(finalize_ts)) / 1000.0
        prefill_to_first_decode_us = max(0.0, delta_us)
    elif graph_wait_event_used_from_summary and ready_publish_ts:
        prefill_to_first_decode_us = 0.0
    if event_query_used is None and wait_start_ts and wait_end_us:
        event_query_used = False
    if graph_wait_event_used_from_summary and not wait_end_us:
        wait_end_us.append(0.0)
    writer_lookup_count = int_sum_fields["writer_pointer_lookup_count"]
    writer_rebuild_count = int_sum_fields["writer_pointer_rebuild_count"]
    if (
        int_sum_seen["writer_pointer_lookup_count"]
        and int_sum_seen["writer_pointer_rebuild_count"]
        and writer_lookup_count > 0
    ):
        writer_cached_pointer_hit_rate = float(
            max(0, writer_lookup_count - writer_rebuild_count)
        ) / float(writer_lookup_count)
    else:
        writer_cached_pointer_hit_rate = -1.0
    return {
        "first_decode_wait_us": max(wait_end_us) if wait_end_us else -1.0,
        "blocked_by_unready_request_us": sum(wait_end_us) if wait_end_us else -1.0,
        "prefill_slowdown_us": prefill_to_first_decode_us,
        "aux_stream_enabled": any(aux_values) if aux_values else None,
        "producer_final_event_present": bool(finalize_ts) if timeline_seen else None,
        "graph_wait_event_used": (
            bool(wait_start_ts and wait_end_us) or graph_wait_event_used_from_summary
            if timeline_seen
            else None
        ),
        "event_query_used_for_publish": event_query_used,
        "expected_group_mask": expected_group_mask,
        "submitted_group_mask": submitted_group_mask,
        "source_ready_event_generation": source_ready_event_generation,
        "producer_final_event_recorded": producer_final_event_recorded,
        "capture_tap_visible_ms_or_unavailable_reason": (
            capture_tap_visible_ms_or_unavailable_reason
        ),
        "bootstrap_full_kv_handoff": bootstrap_full_kv_handoff,
        "writer_cached_pointer_hit_rate": writer_cached_pointer_hit_rate,
        **int_max_fields,
        **{
            key: int(value) if int_sum_seen[key] else -1
            for key, value in int_sum_fields.items()
        },
        **float_max_fields,
    }


def _phase_summary(route_summary: dict[str, object], phase: str) -> dict[str, object]:
    phases = route_summary.get("phase_summaries", {})
    if isinstance(phases, dict):
        value = phases.get(str(phase), {})
        if isinstance(value, dict):
            return dict(value)
    return {}


def _optional_bool_from_route_summary(
    route_summary: dict[str, object],
    key: str,
) -> bool | None:
    if key not in route_summary:
        return None
    return bool(route_summary.get(key))


def _summary_distribution(
    summary: dict[str, object],
    key: str,
) -> dict[str, object]:
    value = summary.get(key, {})
    return dict(value) if isinstance(value, dict) else {}


def _summary_counter(
    summary: dict[str, object],
    distribution_key: str,
    key: str,
    *,
    default: int = 0,
) -> int:
    distribution = _summary_distribution(summary, distribution_key)
    if key in distribution:
        return _as_int(distribution.get(key), default)
    if key in summary:
        return _as_int(summary.get(key), default)
    return default


def _post_switch_phase_failure_reason(
    post_switch_phase_summary: dict[str, object],
) -> str:
    if not post_switch_phase_summary:
        return ""
    if (
        _summary_counter(
            post_switch_phase_summary,
            "row_mode_distribution",
            "native",
        )
        != 0
    ):
        return "post_switch_native_rows_present"
    if (
        _summary_counter(
            post_switch_phase_summary,
            "row_source_distribution",
            "native_rows",
        )
        != 0
    ):
        return "post_switch_native_rows_present"
    if (
        _summary_counter(
            post_switch_phase_summary,
            "row_source_distribution",
            "middle_native_canonical_pages",
        )
        != 0
    ):
        return "post_switch_middle_native_pages_present"
    if (
        _summary_counter(
            post_switch_phase_summary,
            "row_source_distribution",
            "compact_full_native_fallback_rows",
        )
        != 0
    ):
        return "post_switch_full_native_fallback_present"
    if (
        _summary_counter(
            post_switch_phase_summary,
            "row_source_distribution",
            "recent_canonical_pages",
        )
        <= 0
    ):
        return "post_switch_recent_canonical_missing"
    return ""


def _first_non_negative_int(
    timeline_summary: dict[str, Any],
    route_summary: dict[str, object],
    key: str,
    default: int = -1,
) -> int:
    for source in (timeline_summary, route_summary):
        value = _as_int(source.get(key), default)
        if value >= 0:
            return value
    return default


def _first_non_negative_float(
    timeline_summary: dict[str, Any],
    route_summary: dict[str, object],
    key: str,
    default: float = -1.0,
) -> float:
    for source in (timeline_summary, route_summary):
        value = _as_float(source.get(key), default)
        if value >= 0.0:
            return value
    return default


def _optional_bool_from_metrics_or_records(
    metrics: dict[str, Any],
    records: list[dict[str, Any]],
    timeline_summary: dict[str, Any],
) -> bool | None:
    if "aux_stream_enabled" in metrics:
        return bool(metrics.get("aux_stream_enabled", False))
    values = [
        bool(record.get("aux_stream_enabled", False))
        for record in records
        if "aux_stream_enabled" in record
    ]
    if values:
        return any(values)
    timeline_value = timeline_summary.get("aux_stream_enabled")
    if timeline_value is None:
        return None
    return bool(timeline_value)


def _optional_bool_from_sources(
    metrics: dict[str, Any],
    refresh_profile: list[dict[str, Any]],
    timeline_summary: dict[str, Any],
    route_summary: dict[str, object],
    key: str,
    *,
    default: bool | None = None,
) -> bool | None:
    for source in (metrics, timeline_summary, route_summary):
        if key in source:
            return bool(source.get(key))
    for record in refresh_profile:
        if key in record:
            return bool(record.get(key))
    return default


def _generated_token_count(records: list[dict[str, Any]]) -> int:
    total = 0
    seen = False
    for record in records:
        for key in ("generated_token_ids", "output_token_ids", "token_ids"):
            value = record.get(key)
            if isinstance(value, list):
                total += len(value)
                seen = True
                break
    return int(total) if seen else -1


def _finish_reason(records: list[dict[str, Any]]) -> str:
    reasons = [
        str(record.get("finish_reason"))
        for record in records
        if record.get("finish_reason") is not None
    ]
    return ",".join(reasons) if reasons else ""


def _eos_seen(records: list[dict[str, Any]]) -> bool | None:
    seen_any = False
    for record in records:
        if "eos_seen" not in record:
            continue
        seen_any = True
        if bool(record.get("eos_seen")):
            return True
    return False if seen_any else None


def _semantic_output_health(records: list[dict[str, Any]]) -> str:
    """Return single-arm output health; cross-arm text is separate evidence."""
    return _sparse_output_content_health(records)


def _semantic_gate_reasons(mode: str, semantic_output_health: str) -> list[str]:
    if mode == "dense":
        return []
    health = str(semantic_output_health or "").strip() or "missing"
    if health == "ok":
        return []
    return [f"semantic_output_health_not_ok:{health}"]


def _sparse_output_content_health(records: list[dict[str, Any]]) -> str:
    texts: list[str] = []
    for record in records:
        text, proof_reason = _semantic_output_text_and_proof_reason(record)
        if proof_reason:
            return proof_reason
        texts.append(text)
    if not texts or any(not text.strip() for text in texts):
        return "missing_text"
    for text in texts:
        if "\ufffd" in text:
            return "decode_replacement_char"
        if any(ord(character) < 32 and character not in "\n\r\t" for character in text):
            return "decode_control_char"
        stats = _text_quality_stats(text)
        if stats["chars"] >= 32.0 and stats["lexical_ratio"] < 0.20:
            return "garbled_low_lexical_ratio"
        words = re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", text.casefold())
        if len(words) >= 16:
            trigrams = list(zip(words, words[1:], words[2:]))
            if len(set(trigrams)) <= max(1, len(trigrams) // 4):
                return "repetitive_ngram"
    return "ok"


def _budget_us_from_sources(
    metrics: dict[str, Any],
    refresh_profile: list[dict[str, Any]],
    timeline_summary: dict[str, Any],
    key: str,
) -> float:
    if key in metrics:
        return _as_float(metrics.get(key), -1.0)
    timeline_value = _as_float(timeline_summary.get(key), -1.0)
    if timeline_value >= 0.0:
        return timeline_value
    return _max_us_from_records(refresh_profile, key)


def _front_ms_from_metrics(metrics: dict[str, Any]) -> float:
    elapsed_s = _as_float(metrics.get("elapsed_s"), -1.0)
    decode_elapsed_s = _as_float(metrics.get("decode_elapsed_s"), -1.0)
    if elapsed_s < 0.0 or decode_elapsed_s < 0.0:
        return -1.0
    return max(0.0, float(elapsed_s - decode_elapsed_s) * 1000.0)


def _arena_field_from_sources(
    metrics: dict[str, Any],
    refresh_profile: list[dict[str, Any]],
    timeline_summary: dict[str, Any],
    key: str,
    default: object,
) -> object:
    if key in timeline_summary:
        return timeline_summary[key]
    if key in metrics:
        return metrics[key]
    for record in reversed(refresh_profile):
        if key in record:
            return record[key]
    return default


def _counter_int_from_sources(
    metrics: dict[str, Any],
    refresh_profile: list[dict[str, Any]],
    route_summary: dict[str, object],
    key: str,
) -> int:
    if key in metrics:
        return _as_int(metrics.get(key), -1)
    if key in route_summary:
        return _as_int(route_summary.get(key), -1)
    total = _sum_int(refresh_profile, key)
    return total if total > 0 else -1


def _counter_int_from_all_sources(
    metrics: dict[str, Any],
    refresh_profile: list[dict[str, Any]],
    timeline_summary: dict[str, Any],
    route_summary: dict[str, object],
    key: str,
) -> int:
    if key in metrics:
        return _as_int(metrics.get(key), -1)
    if key in timeline_summary:
        value = _as_int(timeline_summary.get(key), -1)
        if value >= 0:
            return value
    if key in route_summary:
        value = _as_int(route_summary.get(key), -1)
        if value >= 0:
            return value
    found = False
    total = 0
    for record in refresh_profile:
        if key not in record:
            continue
        found = True
        total += _as_int(record.get(key), 0)
    return int(total) if found else -1


_WRITER_POINTER_REFRESH_COUNTER_KEYS = (
    "writer_pointer_lookup_count",
    "writer_pointer_rebuild_count",
    "writer_cached_pointer_op_count",
    "writer_vector_fallback_count",
)


def _refresh_profile_writer_pointer_summary(
    refresh_profile: list[dict[str, Any]],
) -> dict[str, int | float]:
    found = {key: False for key in _WRITER_POINTER_REFRESH_COUNTER_KEYS}
    totals = {key: 0 for key in _WRITER_POINTER_REFRESH_COUNTER_KEYS}
    for record in refresh_profile:
        for key in _WRITER_POINTER_REFRESH_COUNTER_KEYS:
            if key not in record:
                continue
            found[key] = True
            totals[key] += _as_int(record.get(key), 0)
    summary: dict[str, int | float] = {
        key: int(totals[key]) if found[key] else -1
        for key in _WRITER_POINTER_REFRESH_COUNTER_KEYS
    }
    lookups = int(summary["writer_pointer_lookup_count"])
    rebuilds = int(summary["writer_pointer_rebuild_count"])
    if lookups > 0 and rebuilds >= 0:
        hits = max(0, lookups - rebuilds)
        summary["writer_cached_pointer_hit_rate"] = float(hits) / float(lookups)
    else:
        summary["writer_cached_pointer_hit_rate"] = -1.0
    return summary


def _writer_pointer_int_from_sources(
    metrics: dict[str, Any],
    timeline_summary: dict[str, Any],
    route_summary: dict[str, object],
    refresh_summary: dict[str, int | float],
    key: str,
) -> int:
    if key in metrics:
        return _as_int(metrics.get(key), -1)
    if key in timeline_summary:
        value = _as_int(timeline_summary.get(key), -1)
        if value >= 0:
            return value
    if key in route_summary:
        return _as_int(route_summary.get(key), -1)
    return int(refresh_summary.get(key, -1))


def _writer_pointer_hit_rate_from_sources(
    metrics: dict[str, Any],
    timeline_summary: dict[str, Any],
    route_summary: dict[str, object],
    refresh_summary: dict[str, int | float],
) -> float:
    key = "writer_cached_pointer_hit_rate"
    if key in metrics:
        return _as_float(metrics.get(key), -1.0)
    if key in timeline_summary:
        value = _as_float(timeline_summary.get(key), -1.0)
        if value >= 0.0:
            return value
    if key in route_summary:
        return _as_float(route_summary.get(key), -1.0)
    return float(refresh_summary.get(key, -1.0))


def _first_non_empty_refresh_profile_value(
    records: list[dict[str, Any]],
    key: str,
) -> str:
    for record in records:
        value = str(record.get(key) or "")
        if value:
            return value
    return ""


def _counter_float_from_sources(
    metrics: dict[str, Any],
    refresh_profile: list[dict[str, Any]],
    timeline_summary: dict[str, Any],
    route_summary: dict[str, object],
    key: str,
) -> float:
    if key in metrics:
        return _as_float(metrics.get(key), -1.0)
    if key in timeline_summary:
        return _as_float(timeline_summary.get(key), -1.0)
    if key in route_summary:
        return _as_float(route_summary.get(key), -1.0)
    return _max_us_from_records(refresh_profile, key)


def _row_source_counter(
    row_source_distribution: dict[str, Any],
    key: str,
) -> int:
    if key in row_source_distribution:
        return _as_int(row_source_distribution.get(key), -1)
    if key == "inactive_rows":
        return 0
    return -1


def _source_proof_ready(
    *,
    source_counter_schema_version: int,
    source_counter_missing_fields: list[str],
    row_mode_distribution: dict[str, int],
    compact_rows: int,
    inactive_rows: int,
    compact_rows_with_reserved_pages: int,
    compact_rows_with_recent_pages: int,
    expected_rows: int,
    num_kv_heads: int,
    middle_native_canonical_pages: int,
    native_rows: int,
    compact_full_native_fallback_rows: int,
) -> bool:
    return bool(
        source_counter_schema_version == STAGE_A_SOURCE_COUNTER_SCHEMA_VERSION
        and not source_counter_missing_fields
        and "native" in row_mode_distribution
        and compact_rows > 0
        and expected_rows >= 0
        and num_kv_heads >= 0
        and compact_rows + inactive_rows == expected_rows * num_kv_heads
        and compact_rows_with_reserved_pages == compact_rows
        and compact_rows_with_recent_pages == compact_rows
        and middle_native_canonical_pages == 0
        and native_rows == 0
        and compact_full_native_fallback_rows == 0
    )


def _record_from_result(
    args: argparse.Namespace,
    *,
    result: Phase1CommandResult,
    route_summary: dict[str, object],
    producer_route_summary: dict[str, object] | None = None,
    metrics: dict[str, Any],
    refresh_profile: list[dict[str, Any]],
    timeline_records: list[dict[str, Any]] | None = None,
    output_records: list[dict[str, Any]] | None = None,
    reference_max_abs_diff: float = -1.0,
    reference_semantic_match: bool | None = None,
    reference_quality_reasons: list[str] | None = None,
    require_budget_fields: bool = False,
    run_pair_id: str = "",
    config_digest: str = "",
    route_proof_result: RouteProofResult | None = None,
    selector_pipeline_cpu_profile: list[dict[str, Any]] | None = None,
) -> Phase2OneShotGraphRecord:
    producer_route_summary = producer_route_summary or route_summary
    timeline_records = timeline_records or []
    output_records = output_records or []
    reference_quality_reasons = reference_quality_reasons or []
    producer_mode = _producer_mode(args)
    continuous_producer_enabled = _phase2_continuous_producer_enabled(args)
    timeline_summary = _timeline_budget_summary(timeline_records)
    row_mode_distribution = dict(route_summary.get("row_mode_distribution", {}) or {})
    row_source_distribution = dict(route_summary.get("row_source_distribution", {}) or {})
    compact_rows = int(row_mode_distribution.get("compact", 0) or 0)
    source_compact_rows = _row_source_counter(row_source_distribution, "compact_rows")
    compact_rows_with_reserved_pages = _row_source_counter(
        row_source_distribution,
        "compact_rows_with_reserved_pages",
    )
    compact_rows_with_recent_pages = _row_source_counter(
        row_source_distribution,
        "compact_rows_with_recent_pages",
    )
    compact_reserved_pages = _row_source_counter(
        row_source_distribution,
        "compact_reserved_pages",
    )
    recent_canonical_pages = _row_source_counter(
        row_source_distribution,
        "recent_canonical_pages",
    )
    middle_native_canonical_pages = _row_source_counter(
        row_source_distribution,
        "middle_native_canonical_pages",
    )
    native_canonical_pages = _row_source_counter(
        row_source_distribution,
        NATIVE_CANONICAL_PAGES_KEY,
    )
    inactive_rows = _row_source_counter(row_source_distribution, "inactive_rows")
    native_rows = _row_source_counter(row_source_distribution, "native_rows")
    compact_full_native_fallback_rows = _row_source_counter(
        row_source_distribution,
        "compact_full_native_fallback_rows",
    )
    source_counter_schema_version = _as_int(
        route_summary.get("source_counter_schema_version"),
        -1,
    )
    source_counter_missing_fields = list(
        _stage_a_source_counter_missing_fields(
            route_summary,
            row_source_distribution,
        )
    )
    expected_rows = _as_int(route_summary.get("expected_rows"), -1)
    num_kv_heads = _as_int(route_summary.get("num_kv_heads"), -1)
    source_proof_ready = bool(
        native_canonical_pages == 0
        and _source_proof_ready(
            source_counter_schema_version=source_counter_schema_version,
            source_counter_missing_fields=source_counter_missing_fields,
            row_mode_distribution=row_mode_distribution,
            compact_rows=source_compact_rows,
            inactive_rows=inactive_rows,
            compact_rows_with_reserved_pages=compact_rows_with_reserved_pages,
            compact_rows_with_recent_pages=compact_rows_with_recent_pages,
            expected_rows=expected_rows,
            num_kv_heads=num_kv_heads,
            middle_native_canonical_pages=middle_native_canonical_pages,
            native_rows=native_rows,
            compact_full_native_fallback_rows=compact_full_native_fallback_rows,
        )
    )
    selector_runs = _sum_int(refresh_profile, "prefill_selector_runs")
    timeline_selector_runs = _as_int(
        timeline_summary.get("prefill_selector_runs"),
        -1,
    )
    if timeline_selector_runs > 0:
        selector_runs += int(timeline_selector_runs)
    rebuild_runs = _sum_int(refresh_profile, "prefill_rebuild_runs")
    timeline_rebuild_runs = _as_int(
        timeline_summary.get("prefill_rebuild_runs"),
        -1,
    )
    if timeline_rebuild_runs > 0:
        rebuild_runs += int(timeline_rebuild_runs)
    refresh_reason_counts = _refresh_reason_counts_from_route_summary(
        producer_route_summary
    )
    continuous_refresh_reqs = _continuous_refresh_payloads_from_sources(
        refresh_profile,
        producer_route_summary,
    )
    # [INTENTS-SEMANTICS 口径 2026-07-10] 名为 intents 实为 interval-reason 的
    # 世代 enqueue 计数(refresh_reason_counts["interval"],按 payload enqueue
    # 事件×req_count 累加),非 token-time 意图数——判读膨胀比值时以"世代数"
    # 口径解读(远端 5.66× 案即此语义,勿再误读)。改名会破坏远端对照口径,保名注释。
    interval_trigger_intents = int(refresh_reason_counts.get("interval", 0))
    expected_interval_trigger_intents = _expected_interval_trigger_intents(
        args,
        producer_mode=producer_mode,
    )
    refresh_trigger_intents = sum(refresh_reason_counts.values())
    sentence_trigger_intents = _counter_int_from_all_sources(
        metrics,
        refresh_profile,
        timeline_summary,
        producer_route_summary,
        "sentence_trigger_intents",
    )
    one_shot_rebuild_ms = _max_ms(
        refresh_profile,
        "prefill_gpu_ms",
        only_prefill=True,
    )
    steady_full_graph_replay_us = float(metrics.get("decode_p50_us", -1.0) or -1.0)
    first_decode_wait_us = _budget_us_from_sources(
        metrics,
        refresh_profile,
        timeline_summary,
        "first_decode_wait_us",
    )
    blocked_by_unready_request_us = _budget_us_from_sources(
        metrics,
        refresh_profile,
        timeline_summary,
        "blocked_by_unready_request_us",
    )
    prefill_slowdown_us = _budget_us_from_sources(
        metrics,
        refresh_profile,
        timeline_summary,
        "prefill_slowdown_us",
    )
    aux_stream_enabled = _optional_bool_from_metrics_or_records(
        metrics,
        refresh_profile,
        timeline_summary,
    )
    producer_total_us = _counter_float_from_sources(
        metrics,
        refresh_profile,
        timeline_summary,
        route_summary,
        "producer_total_us",
    )
    visible_wait_us = _counter_float_from_sources(
        metrics,
        refresh_profile,
        timeline_summary,
        route_summary,
        "visible_wait_us",
    )
    commit_publish_us = _counter_float_from_sources(
        metrics,
        refresh_profile,
        timeline_summary,
        route_summary,
        "commit_publish_us",
    )
    producer_overlap_us = _counter_float_from_sources(
        metrics,
        refresh_profile,
        timeline_summary,
        route_summary,
        "producer_overlap_us",
    )
    producer_deadline_wait_us = _counter_float_from_sources(
        metrics,
        refresh_profile,
        timeline_summary,
        route_summary,
        "producer_deadline_wait_us",
    )
    bridge_summary_source = producer_route_summary or route_summary
    bridge_phase_summary = _phase_summary(bridge_summary_source, "bridge_phase")
    post_switch_phase_summary = _phase_summary(
        bridge_summary_source,
        "post_switch_phase",
    )
    bridge_token_positions_by_request_raw = bridge_summary_source.get(
        "bridge_token_positions_by_request",
        {},
    )
    bridge_token_positions_by_request = (
        dict(bridge_token_positions_by_request_raw)
        if isinstance(bridge_token_positions_by_request_raw, dict)
        else {}
    )
    bridge_token_positions_exact_once = _optional_bool_from_route_summary(
        bridge_summary_source,
        "bridge_token_positions_exact_once",
    )
    compact_middle_excludes_bridge_positions = _optional_bool_from_route_summary(
        bridge_summary_source,
        "compact_middle_excludes_bridge_positions",
    )
    eos_before_ready_seen = _optional_bool_from_route_summary(
        bridge_summary_source,
        "eos_before_ready_seen",
    )
    bridge_token_count = _first_non_negative_int(
        timeline_summary,
        bridge_summary_source,
        "bridge_token_count",
    )
    bridge_decode_p50_us = _first_non_negative_float(
        timeline_summary,
        bridge_summary_source,
        "bridge_decode_p50_us",
    )
    bridge_decode_p95_us = _first_non_negative_float(
        timeline_summary,
        bridge_summary_source,
        "bridge_decode_p95_us",
    )
    producer_launch_step = _first_non_negative_int(
        timeline_summary,
        bridge_summary_source,
        "producer_launch_step",
    )
    producer_ready_step = _first_non_negative_int(
        timeline_summary,
        bridge_summary_source,
        "producer_ready_step",
    )
    bridge_added_decode_cost_us = _first_non_negative_float(
        timeline_summary,
        bridge_summary_source,
        "bridge_added_decode_cost_us",
    )
    saved_visible_wait_us = _first_non_negative_float(
        timeline_summary,
        bridge_summary_source,
        "saved_visible_wait_us",
    )
    graph_route_family = bridge_summary_source.get("graph_route_family", {}) or {}
    if not isinstance(graph_route_family, dict):
        graph_route_family = {}
    capture_postprocess_or_reduce_ms = _counter_float_from_sources(
        metrics,
        refresh_profile,
        timeline_summary,
        route_summary,
        "capture_postprocess_or_reduce_ms",
    )
    prefill_selector_gpu_ms = _counter_float_from_sources(
        metrics,
        refresh_profile,
        timeline_summary,
        route_summary,
        "prefill_selector_gpu_ms",
    )
    prefill_rebuild_gpu_ms = _counter_float_from_sources(
        metrics,
        refresh_profile,
        timeline_summary,
        route_summary,
        "prefill_rebuild_gpu_ms",
    )
    prefill_publish_cpu_us = _counter_float_from_sources(
        metrics,
        refresh_profile,
        timeline_summary,
        route_summary,
        "prefill_publish_cpu_us",
    )
    refresh_selector_gpu_ms = _counter_float_from_sources(
        metrics,
        refresh_profile,
        timeline_summary,
        route_summary,
        "refresh_selector_gpu_ms",
    )
    refresh_rebuild_gpu_ms = _counter_float_from_sources(
        metrics,
        refresh_profile,
        timeline_summary,
        route_summary,
        "refresh_rebuild_gpu_ms",
    )
    selector_gpu_ms = (
        prefill_selector_gpu_ms
        if prefill_selector_gpu_ms >= 0.0
        else refresh_selector_gpu_ms
    )
    pack_rebuild_gpu_ms = (
        prefill_rebuild_gpu_ms
        if prefill_rebuild_gpu_ms >= 0.0
        else refresh_rebuild_gpu_ms
    )
    rrp_publish_us = _counter_float_from_sources(
        metrics,
        refresh_profile,
        timeline_summary,
        route_summary,
        "rrp_publish_us",
    )
    lastn1_direct_count = _counter_int_from_all_sources(
        metrics,
        refresh_profile,
        timeline_summary,
        route_summary,
        "lastn1_direct_count",
    )
    gt1_reduce_count = _counter_int_from_all_sources(
        metrics,
        refresh_profile,
        timeline_summary,
        {},
        "gt1_reduce_count",
    )
    gt1_scalar_fallback_count = _counter_int_from_all_sources(
        metrics,
        refresh_profile,
        timeline_summary,
        {},
        "gt1_scalar_fallback_count",
    )
    page_resolver_kind0_count = _counter_int_from_all_sources(
        metrics,
        refresh_profile,
        timeline_summary,
        route_summary,
        "page_resolver_kind0_count",
    )
    page_resolver_kind1_count = _counter_int_from_all_sources(
        metrics,
        refresh_profile,
        timeline_summary,
        route_summary,
        "page_resolver_kind1_count",
    )
    page_resolver_kind4_count = _counter_int_from_all_sources(
        metrics,
        refresh_profile,
        timeline_summary,
        route_summary,
        "page_resolver_kind4_count",
    )
    kind2_dispatch_count = _counter_int_from_all_sources(
        metrics,
        refresh_profile,
        timeline_summary,
        route_summary,
        "kind2_dispatch_count",
    )
    kind3_dispatch_count = _counter_int_from_all_sources(
        metrics,
        refresh_profile,
        timeline_summary,
        route_summary,
        "kind3_dispatch_count",
    )
    selected_table_publish_count = _counter_int_from_all_sources(
        metrics,
        refresh_profile,
        timeline_summary,
        route_summary,
        "selected_table_publish_count",
    )
    vector_fallback_rows = _counter_int_from_all_sources(
        metrics,
        refresh_profile,
        timeline_summary,
        route_summary,
        "vector_fallback_rows",
    )
    full_fallback_rows = _counter_int_from_all_sources(
        metrics,
        refresh_profile,
        timeline_summary,
        route_summary,
        "full_fallback_rows",
    )
    diagnostic_flags_enabled = _optional_bool_from_sources(
        metrics,
        refresh_profile,
        timeline_summary,
        route_summary,
        "diagnostic_flags_enabled",
        default=False,
    )
    generated_token_count = _as_int(
        metrics.get("generated_token_count", _generated_token_count(output_records)),
        -1,
    )
    actual_decode_steps = _as_int(
        metrics.get("actual_decode_steps", generated_token_count),
        -1,
    )
    finish_reason = str(metrics.get("finish_reason", _finish_reason(output_records)) or "")
    eos_seen = (
        bool(metrics.get("eos_seen"))
        if "eos_seen" in metrics
        else _eos_seen(output_records)
    )
    semantic_output_health = str(
        metrics.get(
            "semantic_output_health",
            _semantic_output_health(output_records),
        )
        or ""
    )
    capture_tap_visible_ms_or_unavailable_reason = str(
        metrics.get(
            "capture_tap_visible_ms_or_unavailable_reason",
            timeline_summary.get(
                "capture_tap_visible_ms_or_unavailable_reason",
                "unavailable:not_measured",
            ),
        )
    )
    selector_launch_count = _counter_int_from_sources(
        metrics,
        refresh_profile,
        route_summary,
        "selector_launch_count",
    )
    writer_launch_count = _counter_int_from_sources(
        metrics,
        refresh_profile,
        route_summary,
        "writer_launch_count",
    )
    writer_pointer_summary = _refresh_profile_writer_pointer_summary(refresh_profile)
    writer_pointer_rebuild_count = _writer_pointer_int_from_sources(
        metrics,
        timeline_summary,
        route_summary,
        writer_pointer_summary,
        "writer_pointer_rebuild_count",
    )
    writer_pointer_lookup_count = _writer_pointer_int_from_sources(
        metrics,
        timeline_summary,
        route_summary,
        writer_pointer_summary,
        "writer_pointer_lookup_count",
    )
    writer_cached_pointer_hit_rate = _writer_pointer_hit_rate_from_sources(
        metrics,
        timeline_summary,
        route_summary,
        writer_pointer_summary,
    )
    writer_cached_pointer_op_count = _writer_pointer_int_from_sources(
        metrics,
        timeline_summary,
        route_summary,
        writer_pointer_summary,
        "writer_cached_pointer_op_count",
    )
    writer_vector_fallback_count = _writer_pointer_int_from_sources(
        metrics,
        timeline_summary,
        route_summary,
        writer_pointer_summary,
        "writer_vector_fallback_count",
    )
    writer_kernel_variant = _first_non_empty_refresh_profile_value(
        refresh_profile,
        "writer_kernel_variant",
    )
    writer_actual_tokens = _counter_int_from_sources(
        metrics, refresh_profile, route_summary, "writer_actual_tokens"
    )
    writer_sink_tokens = _counter_int_from_sources(
        metrics, refresh_profile, route_summary, "writer_sink_tokens"
    )
    writer_persist_tokens = _counter_int_from_sources(
        metrics, refresh_profile, route_summary, "writer_persist_tokens"
    )
    writer_sink_io_bytes = _counter_int_from_sources(
        metrics, refresh_profile, route_summary, "writer_sink_io_bytes"
    )
    writer_persist_io_bytes = _counter_int_from_sources(
        metrics, refresh_profile, route_summary, "writer_persist_io_bytes"
    )
    writer_token_tiles_estimated = _counter_int_from_sources(
        metrics, refresh_profile, route_summary, "writer_token_tiles_estimated"
    )
    writer_active_token_tiles_estimated = _counter_int_from_sources(
        metrics, refresh_profile, route_summary, "writer_active_token_tiles_estimated"
    )
    writer_cta_count_estimated = _counter_int_from_sources(
        metrics, refresh_profile, route_summary, "writer_cta_count_estimated"
    )
    writer_active_cta_count_estimated = _counter_int_from_sources(
        metrics, refresh_profile, route_summary, "writer_active_cta_count_estimated"
    )
    writer_tokens_per_cta = _counter_int_from_sources(
        metrics, refresh_profile, route_summary, "writer_tokens_per_cta"
    )
    writer_k_read_bytes = _counter_int_from_sources(
        metrics, refresh_profile, route_summary, "writer_k_read_bytes"
    )
    writer_v_read_bytes = _counter_int_from_sources(
        metrics, refresh_profile, route_summary, "writer_v_read_bytes"
    )
    writer_k_write_bytes = _counter_int_from_sources(
        metrics, refresh_profile, route_summary, "writer_k_write_bytes"
    )
    writer_v_write_bytes = _counter_int_from_sources(
        metrics, refresh_profile, route_summary, "writer_v_write_bytes"
    )
    writer_pos_write_bytes = _counter_int_from_sources(
        metrics, refresh_profile, route_summary, "writer_pos_write_bytes"
    )
    writer_total_io_bytes = _counter_int_from_sources(
        metrics, refresh_profile, route_summary, "writer_total_io_bytes"
    )
    writer_effective_io_gbps = (
        float(writer_total_io_bytes) / float(pack_rebuild_gpu_ms) / 1_000_000.0
        if writer_total_io_bytes > 0 and pack_rebuild_gpu_ms > 0.0
        else -1.0
    )
    selected_indices_materialized_bytes = _counter_int_from_sources(
        metrics,
        refresh_profile,
        route_summary,
        "selected_indices_materialized_bytes",
    )
    selected_indices_io_bytes = _counter_int_from_sources(
        metrics,
        refresh_profile,
        route_summary,
        "selected_indices_io_bytes",
    )
    selector_writer_current_path_count = _counter_int_from_sources(
        metrics,
        refresh_profile,
        route_summary,
        "selector_writer_current_path_count",
    )
    selector_writer_boundary_cpu_us = _max_us_from_records(
        refresh_profile,
        "selector_writer_boundary_cpu_us",
    )
    falsification_steady_summary = _falsification_steady_summary(refresh_profile)
    selected_boundary_lower_bound_ms_per_group = float(
        falsification_steady_summary["selected_boundary_lower_bound_ms_per_group"]
    )
    predicted_front_early_step_improvement_ms = float(
        falsification_steady_summary["predicted_front_early_step_improvement_ms"]
    )
    residual_fixed_capture_control_ms = float(
        falsification_steady_summary["residual_fixed_capture_control_ms"]
    )
    deadline_v2_attribution = _deadline_v2_attribution_summary(
        metrics=metrics,
        refresh_profile=refresh_profile,
        timeline_summary=timeline_summary,
        route_summary=route_summary,
        selector_pipeline_cpu_profile=selector_pipeline_cpu_profile,
    )
    native_compact_copy_bytes = _counter_int_from_sources(
        metrics,
        refresh_profile,
        route_summary,
        "native_compact_copy_bytes",
    )
    compact_to_compact_copy_bytes = _counter_int_from_sources(
        metrics,
        refresh_profile,
        route_summary,
        "compact_to_compact_copy_bytes",
    )
    compact_kv_storage_owner = str(
        route_summary.get("compact_kv_storage_owner", "") or ""
    )
    compact_kv_native_residency_bind_count = _counter_int_from_sources(
        metrics,
        refresh_profile,
        route_summary,
        "compact_kv_native_residency_bind_count",
    )
    compact_kv_storage_mismatch_count = _counter_int_from_sources(
        metrics,
        refresh_profile,
        route_summary,
        "compact_kv_storage_mismatch_count",
    )
    compact_kv_reserved_span_mismatch_count = _counter_int_from_sources(
        metrics,
        refresh_profile,
        route_summary,
        "compact_kv_reserved_span_mismatch_count",
    )
    selected_middle_tokens = _counter_int_from_sources(
        metrics,
        refresh_profile,
        route_summary,
        "selected_middle_tokens",
    )
    selected_middle_pages = _counter_int_from_sources(
        metrics,
        refresh_profile,
        route_summary,
        "selected_middle_pages",
    )
    native_residency_ready = bool(
        compact_kv_storage_owner == "native_vllm_page_kv"
        and compact_kv_native_residency_bind_count > 0
        and compact_kv_storage_mismatch_count == 0
        and compact_kv_reserved_span_mismatch_count == 0
        and compact_to_compact_copy_bytes == 0
    )
    resolver_kind = str(
        route_summary.get("resolver_kind")
        or ("resolved_row_ptr" if int(route_summary.get("resolved_row_ptr_fwd_mixed_page_count", 0) or 0) > 0 else "")
    )
    route_mode = str(
        route_summary.get("route_mode")
        or ("resolved_row_ptr" if resolver_kind == "resolved_row_ptr" else "")
    )
    native_compact_residency_missing_layers = [
        int(layer)
        for layer in (route_summary.get("native_compact_residency_missing_layers", []) or [])
    ]
    expected_group_mask = _as_int(
        metrics.get(
            "expected_group_mask",
            timeline_summary.get(
                "expected_group_mask",
                route_summary.get("expected_group_mask"),
            ),
        ),
        -1,
    )
    submitted_group_mask = _as_int(
        metrics.get(
            "submitted_group_mask",
            timeline_summary.get(
                "submitted_group_mask",
                route_summary.get("submitted_group_mask"),
            ),
        ),
        -1,
    )
    producer_final_event_present = timeline_summary.get("producer_final_event_present")
    if producer_final_event_present is None and "producer_final_event_present" in metrics:
        producer_final_event_present = bool(metrics.get("producer_final_event_present"))
    producer_final_event_recorded = timeline_summary.get("producer_final_event_recorded")
    if producer_final_event_recorded is None:
        producer_final_event_recorded = (
            bool(producer_final_event_present)
            if producer_final_event_present is not None
            else None
        )
    graph_wait_event_used = timeline_summary.get("graph_wait_event_used")
    if graph_wait_event_used is None and "graph_wait_event_used" in metrics:
        graph_wait_event_used = bool(metrics.get("graph_wait_event_used"))
    event_query_used_for_publish = timeline_summary.get("event_query_used_for_publish")
    if event_query_used_for_publish is None and "event_query_used_for_publish" in metrics:
        event_query_used_for_publish = bool(metrics.get("event_query_used_for_publish"))
    producer_final_event_gpu_wait_us = _as_float(
        metrics.get(
            "producer_final_event_gpu_wait_us",
            timeline_summary.get("producer_final_event_gpu_wait_us", -1.0),
        ),
        -1.0,
    )
    source_ready_event_generation = _as_int(
        metrics.get(
            "source_ready_event_generation",
            timeline_summary.get(
                "source_ready_event_generation",
                route_summary.get("source_ready_event_generation"),
            ),
        ),
        -1,
    )
    sparse_front_ms = _front_ms_from_metrics(metrics)
    dense_front_ms = _as_float(metrics.get("dense_front_ms"), -1.0)
    arena_prepare_miss_count = _as_int(
        _arena_field_from_sources(
            metrics,
            refresh_profile,
            timeline_summary,
            "arena_prepare_miss_count",
            -1,
        ),
        -1,
    )
    arena_budget_exceeded = bool(
        _arena_field_from_sources(
            metrics,
            refresh_profile,
            timeline_summary,
            "arena_budget_exceeded",
            False,
        )
    )
    event_gate_ready = bool(
        (producer_final_event_present is True or producer_final_event_present is None)
        and (graph_wait_event_used is True or graph_wait_event_used is None)
        and (event_query_used_for_publish is False or event_query_used_for_publish is None)
        and (
            expected_group_mask < 0
            or submitted_group_mask < 0
            or expected_group_mask == submitted_group_mask
        )
    )
    budget_fields_ready = bool(
        first_decode_wait_us >= 0.0
        and blocked_by_unready_request_us >= 0.0
        and prefill_slowdown_us >= 0.0
        and aux_stream_enabled is not None
    )
    reference_quality_ok = bool(
        reference_max_abs_diff >= 0.0
        and not reference_quality_reasons
    )
    refresh_requirement_ok = (
        continuous_refresh_reqs > 0
        if _producer_mode_requires_refresh(producer_mode)
        else continuous_refresh_reqs == 0
    )
    interval_trigger_requirement_ok = _interval_trigger_requirement_satisfied(
        expected_interval_trigger_intents=expected_interval_trigger_intents,
        refresh_trigger_intents=refresh_trigger_intents,
    )
    sentence_requirement_ok = (
        sentence_trigger_intents > 0
        if _producer_mode_requires_observed_sentence_trigger(producer_mode)
        else True
    )
    gt1_requirement_ok = (
        max(0, int(getattr(args, "prefill_last_n", 16))) > 1
        and gt1_scalar_fallback_count <= 0
        if _producer_mode_requires_gt1(producer_mode)
        else True
    )
    gate_passed = bool(
        result.returncode == 0
        and not result.timed_out
        and selector_runs > 0
        and rebuild_runs == selector_runs
        and refresh_requirement_ok
        and interval_trigger_requirement_ok
        and sentence_requirement_ok
        and gt1_requirement_ok
        and compact_rows > 0
        and compact_reserved_pages > 0
        and recent_canonical_pages > 0
        and source_proof_ready
        and native_residency_ready
        and resolver_kind == "resolved_row_ptr"
        and route_mode == "resolved_row_ptr"
        and not native_compact_residency_missing_layers
        and page_resolver_kind4_count > 0
        and kind2_dispatch_count == 0
        and kind3_dispatch_count == 0
        and selected_table_publish_count == 0
        and vector_fallback_rows == 0
        and full_fallback_rows == 0
        and diagnostic_flags_enabled is False
        and event_gate_ready
        and writer_cached_pointer_op_count > 0
        and writer_vector_fallback_count == 0
        and reference_quality_ok
        and (
            not bool(getattr(args, "outputs_include_text", False))
            or semantic_output_health == "ok"
        )
        and int(route_summary.get("resolved_row_ptr_fwd_mixed_page_count", 0) or 0) > 0
        and (budget_fields_ready or not require_budget_fields)
    )
    route_proof_passed = bool(route_proof_result.passed) if route_proof_result else False
    route_proof_reasons = list(route_proof_result.reasons) if route_proof_result else []
    return Phase2OneShotGraphRecord(
        case=str(args.case),
        one_shot_bootstrap_only=True,
        continuous_producer_enabled=continuous_producer_enabled,
        producer_mode=producer_mode,
        selector_runs=int(selector_runs),
        rebuild_runs=int(rebuild_runs),
        continuous_refresh_reqs=int(continuous_refresh_reqs),
        refresh_payloads=int(continuous_refresh_reqs),
        sentence_trigger_intents=int(sentence_trigger_intents),
        row_mode_distribution=row_mode_distribution,
        row_source_distribution=row_source_distribution,
        one_shot_rebuild_us=(
            float(one_shot_rebuild_ms) * 1000.0 if one_shot_rebuild_ms >= 0 else -1.0
        ),
        prefill_to_first_decode_us=-1.0,
        steady_full_graph_replay_us=steady_full_graph_replay_us,
        carrier_update_us=-1.0,
        reference_max_abs_diff=float(reference_max_abs_diff),
        reference_semantic_match=reference_semantic_match,
        reference_quality_reasons=list(reference_quality_reasons),
        out_atol=DEFAULT_OUT_ATOL,
        out_rtol=DEFAULT_OUT_RTOL,
        lse_atol=DEFAULT_LSE_ATOL,
        lse_rtol=DEFAULT_LSE_RTOL,
        gate_passed=gate_passed,
        run_pair_id=run_pair_id,
        config_digest=config_digest,
        route_proof_passed=route_proof_passed,
        route_proof_reasons=route_proof_reasons,
        refresh_reason_counts=refresh_reason_counts,
        refresh_trigger_intents=refresh_trigger_intents,
        interval_trigger_intents=interval_trigger_intents,
        expected_interval_trigger_intents=int(expected_interval_trigger_intents),
        interval_trigger_requirement_ok=bool(interval_trigger_requirement_ok),
        sentence_trigger_observation_required=(
            _producer_mode_requires_observed_sentence_trigger(producer_mode)
        ),
        gt1_gate_scope=(
            "prefill_last_n_bootstrap"
            if _producer_mode_requires_gt1(producer_mode)
            else ""
        ),
        prefill_last_n_gt1_requested=bool(
            max(0, int(getattr(args, "prefill_last_n", 16))) > 1
        ),
        first_decode_wait_us=first_decode_wait_us,
        blocked_by_unready_request_us=blocked_by_unready_request_us,
        prefill_slowdown_us=prefill_slowdown_us,
        aux_stream_enabled=aux_stream_enabled,
        producer_total_us=producer_total_us,
        visible_wait_us=visible_wait_us,
        bridge_token_count=bridge_token_count,
        bridge_decode_p50_us=bridge_decode_p50_us,
        bridge_decode_p95_us=bridge_decode_p95_us,
        producer_launch_step=producer_launch_step,
        producer_ready_step=producer_ready_step,
        bridge_added_decode_cost_us=bridge_added_decode_cost_us,
        saved_visible_wait_us=saved_visible_wait_us,
        captured_route_family=str(graph_route_family.get("captured_route_family", "")),
        current_route_family=str(graph_route_family.get("current_route_family", "")),
        route_family_mismatch=(
            bool(graph_route_family["route_family_mismatch"])
            if "route_family_mismatch" in graph_route_family
            else None
        ),
        bridge_phase_route_summary=bridge_phase_summary,
        post_switch_phase_route_summary=post_switch_phase_summary,
        bridge_token_positions_by_request=bridge_token_positions_by_request,
        bridge_token_positions_exact_once=bridge_token_positions_exact_once,
        compact_middle_excludes_bridge_positions=(
            compact_middle_excludes_bridge_positions
        ),
        eos_before_ready_seen=eos_before_ready_seen,
        bootstrap_full_kv_handoff=bool(
            timeline_summary.get("bootstrap_full_kv_handoff", False)
            or bridge_summary_source.get("bootstrap_full_kv_handoff", False)
        ),
        commit_publish_us=commit_publish_us,
        producer_overlap_us=producer_overlap_us,
        producer_deadline_wait_us=producer_deadline_wait_us,
        capture_tap_visible_ms_or_unavailable_reason=(
            capture_tap_visible_ms_or_unavailable_reason
        ),
        capture_postprocess_or_reduce_ms=capture_postprocess_or_reduce_ms,
        lastn1_direct_count=lastn1_direct_count,
        gt1_reduce_count=gt1_reduce_count,
        gt1_scalar_fallback_count=gt1_scalar_fallback_count,
        selector_gpu_ms=selector_gpu_ms,
        selector_launch_count=selector_launch_count,
        pack_rebuild_gpu_ms=pack_rebuild_gpu_ms,
        prefill_selector_gpu_ms=prefill_selector_gpu_ms,
        prefill_rebuild_gpu_ms=prefill_rebuild_gpu_ms,
        prefill_publish_cpu_us=prefill_publish_cpu_us,
        page_resolver_kind0_count=page_resolver_kind0_count,
        page_resolver_kind1_count=page_resolver_kind1_count,
        page_resolver_kind4_count=page_resolver_kind4_count,
        kind2_dispatch_count=kind2_dispatch_count,
        kind3_dispatch_count=kind3_dispatch_count,
        selected_table_publish_count=selected_table_publish_count,
        vector_fallback_rows=vector_fallback_rows,
        full_fallback_rows=full_fallback_rows,
        diagnostic_flags_enabled=diagnostic_flags_enabled,
        generated_token_count=generated_token_count,
        actual_decode_steps=actual_decode_steps,
        finish_reason=finish_reason,
        eos_seen=eos_seen,
        semantic_output_health=semantic_output_health,
        writer_launch_count=writer_launch_count,
        writer_pointer_rebuild_count=writer_pointer_rebuild_count,
        writer_pointer_lookup_count=writer_pointer_lookup_count,
        writer_cached_pointer_hit_rate=writer_cached_pointer_hit_rate,
        writer_cached_pointer_op_count=writer_cached_pointer_op_count,
        writer_vector_fallback_count=writer_vector_fallback_count,
        writer_kernel_variant=writer_kernel_variant,
        writer_actual_tokens=writer_actual_tokens,
        writer_sink_tokens=writer_sink_tokens,
        writer_persist_tokens=writer_persist_tokens,
        writer_sink_io_bytes=writer_sink_io_bytes,
        writer_persist_io_bytes=writer_persist_io_bytes,
        writer_token_tiles_estimated=writer_token_tiles_estimated,
        writer_active_token_tiles_estimated=writer_active_token_tiles_estimated,
        writer_cta_count_estimated=writer_cta_count_estimated,
        writer_active_cta_count_estimated=writer_active_cta_count_estimated,
        writer_tokens_per_cta=writer_tokens_per_cta,
        writer_k_read_bytes=writer_k_read_bytes,
        writer_v_read_bytes=writer_v_read_bytes,
        writer_k_write_bytes=writer_k_write_bytes,
        writer_v_write_bytes=writer_v_write_bytes,
        writer_pos_write_bytes=writer_pos_write_bytes,
        writer_total_io_bytes=writer_total_io_bytes,
        writer_effective_io_gbps=writer_effective_io_gbps,
        selected_indices_materialized_bytes=selected_indices_materialized_bytes,
        selected_indices_io_bytes=selected_indices_io_bytes,
        selector_writer_current_path_count=selector_writer_current_path_count,
        selector_writer_boundary_cpu_us=selector_writer_boundary_cpu_us,
        deadline_v2_attribution=deadline_v2_attribution,
        selected_boundary_lower_bound_ms_per_group=(
            selected_boundary_lower_bound_ms_per_group
        ),
        predicted_front_early_step_improvement_ms=(
            predicted_front_early_step_improvement_ms
        ),
        residual_fixed_capture_control_ms=residual_fixed_capture_control_ms,
        resolver_kind=resolver_kind,
        route_mode=route_mode,
        native_compact_residency_required=True,
        native_compact_residency_missing_layers=native_compact_residency_missing_layers,
        expected_group_mask=expected_group_mask,
        submitted_group_mask=submitted_group_mask,
        rrp_publish_us=rrp_publish_us,
        producer_final_event_present=producer_final_event_present,
        producer_final_event_recorded=producer_final_event_recorded,
        graph_wait_event_used=graph_wait_event_used,
        event_query_used_for_publish=event_query_used_for_publish,
        producer_final_event_gpu_wait_us=producer_final_event_gpu_wait_us,
        source_ready_event_generation=source_ready_event_generation,
        production_gate_passed=gate_passed,
        native_compact_copy_bytes=native_compact_copy_bytes,
        compact_to_compact_copy_bytes=compact_to_compact_copy_bytes,
        compact_kv_storage_owner=compact_kv_storage_owner,
        compact_kv_native_residency_bind_count=compact_kv_native_residency_bind_count,
        compact_kv_storage_mismatch_count=compact_kv_storage_mismatch_count,
        compact_kv_reserved_span_mismatch_count=compact_kv_reserved_span_mismatch_count,
        selected_middle_tokens=selected_middle_tokens,
        selected_middle_pages=selected_middle_pages,
        compact_reserved_pages=compact_reserved_pages,
        recent_canonical_pages=recent_canonical_pages,
        middle_native_canonical_pages=middle_native_canonical_pages,
        compact_full_native_fallback_rows=compact_full_native_fallback_rows,
        source_counter_schema_version=source_counter_schema_version,
        source_counter_missing_fields=source_counter_missing_fields,
        dense_front_ms=dense_front_ms,
        sparse_front_ms=sparse_front_ms,
        front_ms_source="elapsed_ms - decode_elapsed_ms",
        arena_reserved_bytes=_as_int(
            _arena_field_from_sources(
                metrics, refresh_profile, timeline_summary, "arena_reserved_bytes", -1
            ),
            -1,
        ),
        arena_peak_bytes=_as_int(
            _arena_field_from_sources(
                metrics, refresh_profile, timeline_summary, "arena_peak_bytes", -1
            ),
            -1,
        ),
        arena_bucket_bytes=_as_int(
            _arena_field_from_sources(
                metrics, refresh_profile, timeline_summary, "arena_bucket_bytes", -1
            ),
            -1,
        ),
        arena_bucket_count=_as_int(
            _arena_field_from_sources(
                metrics, refresh_profile, timeline_summary, "arena_bucket_count", -1
            ),
            -1,
        ),
        arena_largest_bucket_bytes=_as_int(
            _arena_field_from_sources(
                metrics,
                refresh_profile,
                timeline_summary,
                "arena_largest_bucket_bytes",
                -1,
            ),
            -1,
        ),
        arena_expansion_bytes=_as_int(
            _arena_field_from_sources(
                metrics, refresh_profile, timeline_summary, "arena_expansion_bytes", -1
            ),
            -1,
        ),
        arena_budget_exceeded=arena_budget_exceeded,
        arena_prepare_miss_count=arena_prepare_miss_count,
        arena_bind_status=str(
            _arena_field_from_sources(
                metrics, refresh_profile, timeline_summary, "arena_bind_status", ""
            )
        ),
        arena_reservation_status=str(
            _arena_field_from_sources(
                metrics,
                refresh_profile,
                timeline_summary,
                "arena_reservation_status",
                "",
            )
        ),
    )


def _classify_phase2_failure(
    record: Phase2OneShotGraphRecord,
    *,
    result: Phase1CommandResult,
    route_summary: dict[str, object],
) -> str:
    process_failure = _classify_failure(result)
    if process_failure:
        return process_failure
    if record.selector_runs <= 0:
        return "one_shot_selector_missing"
    if record.rebuild_runs != record.selector_runs:
        return "one_shot_rebuild_selector_count_mismatch"
    if _producer_mode_requires_refresh(record.producer_mode):
        if record.continuous_refresh_reqs <= 0:
            return "continuous_refresh_reqs_missing"
    elif record.continuous_refresh_reqs != 0:
        return "continuous_refresh_reqs_nonzero"
    if (
        _producer_mode_requires_observed_sentence_trigger(record.producer_mode)
        and record.sentence_trigger_intents <= 0
    ):
        return "sentence_trigger_intents_missing"
    if _producer_mode_requires_gt1(record.producer_mode):
        if not record.prefill_last_n_gt1_requested:
            return "prefill_last_n_not_gt1"
        if record.gt1_gate_scope != "prefill_last_n_bootstrap":
            return "gt1_gate_scope_unset"
        if record.gt1_scalar_fallback_count > 0:
            return "gt1_scalar_fallback_count_nonzero"
    if int(record.row_mode_distribution.get("compact", 0) or 0) <= 0:
        return "compact_row_distribution_missing"
    if record.source_counter_schema_version < 0 or record.source_counter_missing_fields:
        return "source_provenance_missing"
    source_compact_rows = int(record.row_source_distribution.get("compact_rows", 0) or 0)
    if source_compact_rows <= 0:
        return "compact_rows_missing"
    if "native" not in record.row_mode_distribution:
        return "row_mode_native_missing"
    if (
        int(record.row_source_distribution.get("compact_rows_with_reserved_pages", 0) or 0)
        != source_compact_rows
    ):
        return "compact_rows_with_reserved_pages_mismatch"
    if (
        int(record.row_source_distribution.get("compact_rows_with_recent_pages", 0) or 0)
        != source_compact_rows
    ):
        return "compact_rows_with_recent_pages_mismatch"
    expected_rows = _as_int(route_summary.get("expected_rows"), -1)
    num_kv_heads = _as_int(route_summary.get("num_kv_heads"), -1)
    if expected_rows < 0 or num_kv_heads < 0:
        return "source_provenance_missing"
    inactive_rows = int(record.row_source_distribution.get("inactive_rows", 0) or 0)
    if source_compact_rows + inactive_rows != expected_rows * num_kv_heads:
        return "compact_rows_mismatch_expected_rows_heads"
    if record.compact_reserved_pages <= 0:
        return "compact_reserved_page_source_missing"
    post_switch_failure = _post_switch_phase_failure_reason(
        record.post_switch_phase_route_summary,
    )
    if post_switch_failure:
        return post_switch_failure
    if record.recent_canonical_pages <= 0:
        return "recent_page_source_missing"
    if record.middle_native_canonical_pages != 0:
        return "middle_native_canonical_pages_nonzero"
    if int(record.row_source_distribution.get(NATIVE_CANONICAL_PAGES_KEY, 0) or 0) != 0:
        return "native_canonical_pages_nonzero"
    if int(record.row_source_distribution.get("native_rows", 0) or 0) != 0:
        return "native_rows_nonzero"
    if record.compact_full_native_fallback_rows != 0:
        return "compact_row_fell_back_to_native_full_table"
    if record.compact_kv_storage_owner != "native_vllm_page_kv":
        return "compact_kv_storage_owner_not_native_vllm_page_kv"
    if record.compact_kv_native_residency_bind_count <= 0:
        return "compact_kv_native_residency_bind_missing"
    if record.compact_kv_storage_mismatch_count != 0:
        return "compact_kv_storage_mismatch_nonzero"
    if record.compact_kv_reserved_span_mismatch_count != 0:
        return "compact_kv_reserved_span_mismatch_nonzero"
    if record.compact_to_compact_copy_bytes != 0:
        return "compact_to_compact_copy_bytes_nonzero"
    if record.resolver_kind != "resolved_row_ptr":
        return "resolver_kind_not_resolved_row_ptr"
    if record.route_mode != "resolved_row_ptr":
        return "route_mode_not_resolved_row_ptr"
    if record.page_resolver_kind4_count <= 0:
        return "page_resolver_kind4_count_missing"
    if record.kind2_dispatch_count != 0:
        return "kind2_dispatch_count_nonzero"
    if record.kind3_dispatch_count != 0:
        return "kind3_dispatch_count_nonzero"
    if record.selected_table_publish_count != 0:
        return "selected_table_publish_count_nonzero"
    if record.vector_fallback_rows != 0:
        return "vector_fallback_rows_nonzero"
    if record.full_fallback_rows != 0:
        return "full_fallback_rows_nonzero"
    if record.diagnostic_flags_enabled is not False:
        return "diagnostic_flags_enabled_mismatch"
    if record.native_compact_residency_missing_layers:
        return "native_compact_residency_missing_layers_nonempty"
    if (
        record.expected_group_mask >= 0
        and record.submitted_group_mask >= 0
        and record.expected_group_mask != record.submitted_group_mask
    ):
        return "producer_group_mask_mismatch"
    if record.producer_final_event_present is False:
        return "producer_final_event_missing"
    if record.graph_wait_event_used is False:
        return "graph_wait_event_missing"
    if record.event_query_used_for_publish is True:
        return "event_query_used_for_publish"
    if record.writer_cached_pointer_op_count <= 0:
        return "writer_cached_pointer_op_missing"
    if record.writer_vector_fallback_count != 0:
        return "writer_vector_fallback_nonzero"
    if record.reference_max_abs_diff < 0.0:
        return "dense_reference_not_checked"
    if record.reference_quality_reasons:
        return str(record.reference_quality_reasons[0])
    if int(route_summary.get("resolved_row_ptr_fwd_mixed_page_count", 0) or 0) <= 0:
        return "resolved_row_ptr_fwd_mixed_page_missing"
    if record.first_decode_wait_us < 0.0:
        return "first_decode_wait_us_missing"
    if record.blocked_by_unready_request_us < 0.0:
        return "blocked_by_unready_request_us_missing"
    if record.prefill_slowdown_us < 0.0:
        return "prefill_slowdown_us_missing"
    if record.aux_stream_enabled is None:
        return "aux_stream_enabled_missing"
    return ""


def _write_summary(
    path: Path,
    *,
    record: Phase2OneShotGraphRecord,
    result: Phase1CommandResult,
    route_trace_path: Path,
    metrics_path: Path,
    refresh_profile_path: Path,
    one_shot_timeline_path: Path,
    selector_pipeline_cpu_profile_path: Path | None = None,
    route_summary: dict[str, object],
    metrics: dict[str, Any],
    refresh_profile: list[dict[str, Any]],
    failure_reason: str,
    run_pair_id: str,
    config_digest: str,
    route_proof: dict[str, object],
    producer_route_summary: dict[str, object] | None = None,
    reference_result: Phase1CommandResult | None = None,
    reference_sparse_outputs_path: Path | None = None,
    reference_dense_outputs_path: Path | None = None,
    reference_semantic_diffs: list[dict[str, Any]] | None = None,
    run_provenance: dict[str, Any] | None = None,
    fixed_token_no_eos_diagnostic_required: bool = False,
    fixed_token_no_eos_command: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record_payload = record_to_jsonable(record)
    payload = {
        "gate_passed": bool(record.gate_passed),
        "failure_reason": failure_reason,
        "run_pair_id": run_pair_id,
        "config_digest": config_digest,
        "route_proof": route_proof,
        "record": record_payload,
        "route_summary": route_summary,
        "producer_route_summary": producer_route_summary or route_summary,
        "route_summary_scope": "full_diagnostic_run",
        "producer_route_summary_scope": "measurement_window",
        "metrics": metrics,
        "route_trace_path": str(route_trace_path),
        "decode_metrics_path": str(metrics_path),
        "refresh_profile_path": str(refresh_profile_path),
        "one_shot_timeline_path": str(one_shot_timeline_path),
        "selector_pipeline_cpu_profile_path": (
            str(selector_pipeline_cpu_profile_path)
            if selector_pipeline_cpu_profile_path
            else ""
        ),
        "command": result.command,
        "returncode": int(result.returncode),
        "timed_out": bool(result.timed_out),
        "stdout_tail": _tail(result.stdout),
        "stderr_tail": _tail(result.stderr),
        "fixed_token_no_eos_diagnostic_required": bool(
            fixed_token_no_eos_diagnostic_required
        ),
        "fixed_token_no_eos_command": fixed_token_no_eos_command or [],
    }
    if run_provenance is not None:
        payload["run_provenance"] = run_provenance
    payload["no_fa4_cute_dispatch"] = bool(
        isinstance(run_provenance, dict)
        and str(run_provenance.get("backend", "") or "") == BACKEND_FA3
        and int(run_provenance.get("flash_attn_version_expected", -1)) == 3
    )
    payload.update(record_payload)
    payload["total_tok_per_s"] = _decode_metric(metrics, "tok_per_s")
    payload["decode_p50_ms"] = _decode_metric(metrics, "decode_p50_us") / 1000.0
    payload["decode_p95_ms"] = _decode_metric(metrics, "decode_p95_us") / 1000.0
    payload.update(_boundary_diagnostics_payload(metrics))
    steady_records = _steady_prefill_profile_records(refresh_profile)
    payload["writer_kernel_variant"] = _first_non_empty_refresh_profile_value(
        refresh_profile,
        "writer_kernel_variant",
    )
    payload["prefill_steady_profile_count"] = int(len(steady_records))
    payload["prefill_cold_profile_excluded"] = bool(
        len(_prefill_profile_records(refresh_profile)) > len(steady_records)
    )
    payload["prefill_gpu_ms_steady_avg_excluding_first"] = (
        _steady_avg_excluding_first(refresh_profile, "prefill_gpu_ms")
    )
    payload["prefill_selector_gpu_ms_steady_avg_excluding_first"] = (
        _steady_avg_excluding_first(refresh_profile, "prefill_selector_gpu_ms")
    )
    payload["prefill_gather_gpu_ms_steady_avg_excluding_first"] = (
        _steady_avg_excluding_first(refresh_profile, "prefill_gather_gpu_ms")
    )
    payload["prefill_key_norms_preproc_gpu_ms_steady_avg_excluding_first"] = (
        _steady_avg_excluding_first(refresh_profile, "prefill_key_norms_preproc_gpu_ms")
    )
    payload["prefill_key_norms_gpu_ms_steady_avg_excluding_first"] = (
        _steady_avg_excluding_first(refresh_profile, "prefill_key_norms_gpu_ms")
    )
    payload["prefill_key_norms_h2d_gpu_ms_steady_avg_excluding_first"] = (
        _steady_avg_excluding_first(refresh_profile, "prefill_key_norms_h2d_gpu_ms")
    )
    payload["prefill_key_norms_delta_gpu_ms_steady_avg_excluding_first"] = (
        _steady_avg_excluding_first(refresh_profile, "prefill_key_norms_delta_gpu_ms")
    )
    payload["prefill_key_norms_pack_gpu_ms_steady_avg_excluding_first"] = (
        _steady_avg_excluding_first(refresh_profile, "prefill_key_norms_pack_gpu_ms")
    )
    payload["prefill_key_norms_delta_total_tokens_steady_avg_excluding_first"] = (
        _steady_avg_excluding_first(
            refresh_profile, "prefill_key_norms_delta_total_tokens"
        )
    )
    payload["prefill_key_norms_delta_max_tokens_steady_avg_excluding_first"] = (
        _steady_avg_excluding_first(refresh_profile, "prefill_key_norms_delta_max_tokens")
    )
    payload["prefill_key_norms_delta_layers_steady_avg_excluding_first"] = (
        _steady_avg_excluding_first(refresh_profile, "prefill_key_norms_delta_layers")
    )
    payload["prefill_log_s_gpu_ms_steady_avg_excluding_first"] = (
        _steady_avg_excluding_first(refresh_profile, "prefill_log_s_gpu_ms")
    )
    payload["prefill_log_s_triton_gpu_ms_steady_avg_excluding_first"] = (
        _steady_avg_excluding_first(refresh_profile, "prefill_log_s_triton_gpu_ms")
    )
    payload["prefill_log_s_mask_gpu_ms_steady_avg_excluding_first"] = (
        _steady_avg_excluding_first(refresh_profile, "prefill_log_s_mask_gpu_ms")
    )
    payload["prefill_log_s_cross_gpu_ms_steady_avg_excluding_first"] = (
        _steady_avg_excluding_first(refresh_profile, "prefill_log_s_cross_gpu_ms")
    )
    payload["prefill_topk_gpu_ms_steady_avg_excluding_first"] = (
        _steady_avg_excluding_first(refresh_profile, "prefill_topk_gpu_ms")
    )
    payload["prefill_preproc_gpu_ms_steady_avg_excluding_first"] = (
        _steady_avg_excluding_first(refresh_profile, "prefill_preproc_gpu_ms")
    )
    payload["prefill_seq_full_gpu_ms_steady_avg_excluding_first"] = (
        _steady_avg_excluding_first(refresh_profile, "prefill_seq_full_gpu_ms")
    )
    payload["prefill_pure_preproc_gpu_ms_steady_avg_excluding_first"] = (
        _steady_avg_excluding_first(refresh_profile, "prefill_pure_preproc_gpu_ms")
    )
    payload["prefill_rebuild_gpu_ms_steady_avg_excluding_first"] = (
        _steady_avg_excluding_first(refresh_profile, "prefill_rebuild_gpu_ms")
    )
    payload["prefill_rebuild_gpu_ms_steady_p50_excluding_first"] = (
        _steady_p50_excluding_first(refresh_profile, "prefill_rebuild_gpu_ms")
    )
    payload["prefill_rebuild_gpu_ms_steady_p95_excluding_first"] = (
        _steady_p95_excluding_first(refresh_profile, "prefill_rebuild_gpu_ms")
    )
    payload["prefill_group_count_steady_avg_excluding_first"] = (
        _steady_avg_excluding_first(refresh_profile, "prefill_group_count")
    )
    for group_id in range(4):
        payload[f"prefill_group{group_id}_gpu_ms_steady_avg_excluding_first"] = (
            _steady_avg_excluding_first(
                refresh_profile,
                f"prefill_group{group_id}_gpu_ms",
            )
        )
        payload[
            f"prefill_group{group_id}_done_since_prefill_start_ms_steady_avg_excluding_first"
        ] = _steady_avg_excluding_first(
            refresh_profile,
            f"prefill_group{group_id}_done_since_prefill_start_ms",
        )
    payload["prefill_group1_after_group0_done_ms_steady_avg_excluding_first"] = (
        _steady_avg_delta_excluding_first(
            refresh_profile,
            start_key="prefill_group0_done_since_prefill_start_ms",
            end_key="prefill_group1_done_since_prefill_start_ms",
        )
    )
    for key in (
        "writer_actual_tokens",
        "writer_sink_tokens",
        "writer_persist_tokens",
        "writer_sink_io_bytes",
        "writer_persist_io_bytes",
        "writer_token_tiles_estimated",
        "writer_active_token_tiles_estimated",
        "writer_cta_count_estimated",
        "writer_active_cta_count_estimated",
        "writer_tokens_per_cta",
        "writer_k_read_bytes",
        "writer_v_read_bytes",
        "writer_k_write_bytes",
        "writer_v_write_bytes",
        "writer_pos_write_bytes",
        "writer_total_io_bytes",
    ):
        payload[f"{key}_steady_avg_excluding_first"] = _steady_avg_excluding_first(
            refresh_profile,
            key,
        )
    writer_total_io_bytes_avg = float(
        payload["writer_total_io_bytes_steady_avg_excluding_first"]
    )
    rebuild_ms_avg = float(payload["prefill_rebuild_gpu_ms_steady_avg_excluding_first"])
    payload["writer_effective_io_gbps_steady_avg_excluding_first"] = (
        writer_total_io_bytes_avg / rebuild_ms_avg / 1_000_000.0
        if writer_total_io_bytes_avg > 0.0 and rebuild_ms_avg > 0.0
        else -1.0
    )
    payload.update(_falsification_steady_summary(refresh_profile))
    payload["deadline_v2_attribution"] = dict(record.deadline_v2_attribution)
    payload["prefill_publish_cpu_us_steady_avg_excluding_first"] = (
        _steady_avg_excluding_first(refresh_profile, "prefill_publish_cpu_us")
    )
    payload[
        "source_ready_recorded_after_pointer_publish_count_steady_sum_excluding_first"
    ] = int(
        sum(
            int(
                record.get("source_ready_recorded_after_pointer_publish_count") or 0
            )
            for record in steady_records
        )
    )
    payload["native_canonical_pages"] = int(
        record.row_source_distribution.get(NATIVE_CANONICAL_PAGES_KEY, 0) or 0
    )
    if reference_result is not None:
        payload["reference_command"] = reference_result.command
        payload["reference_returncode"] = int(reference_result.returncode)
        payload["reference_timed_out"] = bool(reference_result.timed_out)
        payload["reference_stdout_tail"] = _tail(reference_result.stdout)
        payload["reference_stderr_tail"] = _tail(reference_result.stderr)
        payload["reference_scope"] = "token_ids_or_semantic_text_plus_output_quality"
    else:
        payload["reference_scope"] = "not_checked"
    if reference_sparse_outputs_path is not None:
        payload["reference_sparse_outputs_path"] = str(reference_sparse_outputs_path)
    if reference_dense_outputs_path is not None:
        payload["reference_dense_outputs_path"] = str(reference_dense_outputs_path)
    payload["reference_semantic_diffs"] = reference_semantic_diffs or []
    payload["reference_quality_reasons"] = list(record.reference_quality_reasons)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode != "legacy":
        return _run_gate_d_mode(args)

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
    refresh_profile_path = (
        Path(args.refresh_profile_output)
        if args.refresh_profile_output
        else _default_refresh_profile_path(output_path)
    )
    one_shot_timeline_path = (
        Path(args.one_shot_timeline_output)
        if args.one_shot_timeline_output
        else _default_one_shot_timeline_path(output_path)
    )
    sparse_outputs_path = _default_sparse_outputs_path(output_path)
    dense_outputs_path = _default_dense_outputs_path(output_path)
    reference_result: Phase1CommandResult | None = None
    reference_max_abs_diff = -1.0
    reference_semantic_match: bool | None = None
    reference_semantic_diffs: list[dict[str, Any]] = []
    reference_quality_reasons: list[str] = []
    sparse_records: list[dict[str, Any]] = []
    dense_records: list[dict[str, Any]] = []
    sparse_env = _build_phase2_env(
        args,
        route_trace_path=route_trace_path,
        refresh_profile_path=refresh_profile_path,
        one_shot_timeline_path=one_shot_timeline_path,
    )
    selector_pipeline_cpu_profile_path = _selector_pipeline_cpu_profile_path(
        sparse_env,
        output_path,
    )
    gpu_before = _gpu_snapshot("pre", cuda_visible_devices=str(args.cuda_visible_devices))

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
        refresh_profile_path.parent.mkdir(parents=True, exist_ok=True)
        one_shot_timeline_path.parent.mkdir(parents=True, exist_ok=True)
        sparse_outputs_path.parent.mkdir(parents=True, exist_ok=True)
        route_trace_path.write_text("", encoding="utf-8")
        refresh_profile_path.write_text("", encoding="utf-8")
        one_shot_timeline_path.write_text("", encoding="utf-8")
        if (
            selector_pipeline_cpu_profile_path
            == _default_selector_pipeline_cpu_profile_path(output_path)
        ):
            selector_pipeline_cpu_profile_path.parent.mkdir(parents=True, exist_ok=True)
            selector_pipeline_cpu_profile_path.write_text("", encoding="utf-8")
        command = _build_phase2_command(
            args,
            metrics_path=metrics_path,
            refresh_profile_path=refresh_profile_path,
            outputs_path=sparse_outputs_path,
        )
        result = _run_command(
            command,
            env=sparse_env,
            timeout_s=int(args.timeout_s),
        )
        if (
            result.returncode == 0
            and not result.timed_out
            and not bool(args.skip_dense_reference)
        ):
            dense_outputs_path.parent.mkdir(parents=True, exist_ok=True)
            reference_command = _build_dense_reference_command(
                args,
                outputs_path=dense_outputs_path,
            )
            reference_result = _run_command(
                reference_command,
                env=_build_gate_d_dense_env(args),
                timeout_s=int(args.timeout_s),
            )
            if reference_result.returncode == 0 and not reference_result.timed_out:
                sparse_records = _request_ordered_output_records(
                    sparse_outputs_path
                )
                dense_records = _request_ordered_output_records(
                    dense_outputs_path
                )
                sparse_outputs = [
                    list(record["token_ids"]) for record in sparse_records
                ]
                dense_outputs = [
                    list(record["token_ids"]) for record in dense_records
                ]
                reference_max_abs_diff = 0.0 if sparse_outputs and sparse_outputs == dense_outputs else 1.0
                reference_semantic_diffs = _semantic_output_diffs(
                    sparse_records,
                    dense_records,
                )
                reference_quality_reasons = _reference_output_quality_reasons(
                    sparse_records,
                    dense_records,
                )
                if reference_semantic_diffs:
                    reference_semantic_match = all(
                        bool(diff.get("semantic_match", False))
                        for diff in reference_semantic_diffs
                    )

    gpu_after = _gpu_snapshot("post", cuda_visible_devices=str(args.cuda_visible_devices))
    route_summary = _route_summary(_read_trace_events(route_trace_path))
    producer_route_summary = _route_summary(
        _read_measurement_trace_events(route_trace_path)
    )
    metrics = _read_json(metrics_path)
    refresh_profile = _read_refresh_profile(refresh_profile_path)
    selector_pipeline_cpu_profile = (
        _read_selector_pipeline_cpu_profile(selector_pipeline_cpu_profile_path)
        if selector_pipeline_cpu_profile_path is not None
        else []
    )
    timeline_records = _read_one_shot_timeline(one_shot_timeline_path)
    fa3_so_sha256 = _file_sha256(_resolve_gate_d_backend_artifact(args))
    run_provenance = _run_provenance_payload(
        args,
        env=sparse_env,
        fa3_so_sha256=fa3_so_sha256,
        gpu_before=gpu_before,
        gpu_after=gpu_after,
    )
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
    record = _record_from_result(
        args,
        result=result,
        route_summary=route_summary,
        producer_route_summary=producer_route_summary,
        metrics=metrics,
        refresh_profile=refresh_profile,
        timeline_records=timeline_records,
        output_records=sparse_records,
        reference_max_abs_diff=reference_max_abs_diff,
        reference_semantic_match=reference_semantic_match,
        reference_quality_reasons=reference_quality_reasons,
        require_budget_fields=not bool(args.dry_run),
        run_pair_id=run_pair_id,
        config_digest=config_digest,
        route_proof_result=route_proof_result,
        selector_pipeline_cpu_profile=selector_pipeline_cpu_profile,
    )
    failure_reason = _classify_phase2_failure(
        record,
        result=result,
        route_summary=route_summary,
    )
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
        refresh_profile_path=refresh_profile_path,
        one_shot_timeline_path=one_shot_timeline_path,
        selector_pipeline_cpu_profile_path=selector_pipeline_cpu_profile_path,
        reference_result=reference_result,
        reference_sparse_outputs_path=sparse_outputs_path,
        reference_dense_outputs_path=dense_outputs_path,
        reference_semantic_diffs=reference_semantic_diffs,
        run_provenance=run_provenance,
        route_summary=route_summary,
        producer_route_summary=producer_route_summary,
        metrics=metrics,
        refresh_profile=refresh_profile,
        failure_reason=failure_reason,
        run_pair_id=run_pair_id,
        config_digest=config_digest,
        route_proof=route_proof,
        fixed_token_no_eos_diagnostic_required=bool(args.respect_eos),
        fixed_token_no_eos_command=_build_no_eos_diagnostic_command(args),
    )
    return 0 if record.gate_passed else GATE_FAILURE_EXIT_CODE


if __name__ == "__main__":
    raise SystemExit(main())
