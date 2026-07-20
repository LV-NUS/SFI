from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping


ROUTE_TRACE_ENV_KEY = "VLLM_SPARSE_FA3_ROUTE_TRACE_LOG"
FULL_CUDAGRAPH_HOOK_PROFILE_ENV_KEY = (
    "VLLM_SPARSE_FULL_CUDAGRAPH_HOOK_PROFILE_LOG"
)
VERDICT_ONLY_SPEED_PROOF_ENV_BINDINGS = (
    (ROUTE_TRACE_ENV_KEY, "speed_child_route_trace_path"),
    (
        FULL_CUDAGRAPH_HOOK_PROFILE_ENV_KEY,
        "full_cudagraph_hook_profile_path",
    ),
)

ALLOWED_TRACE_ENV_KEYS = (
    ROUTE_TRACE_ENV_KEY,
    "VLLM_SPARSE_FA3_STEP_TRACE_LOG",
    FULL_CUDAGRAPH_HOOK_PROFILE_ENV_KEY,
    "VLLM_SPARSE_MB_PROFILE_LOG",
    "VLLM_SPARSE_RRP_PREP_PROFILE_LOG",
    "VLLM_SPARSE_ONE_SHOT_TIMELINE_LOG",
    "VLLM_SPARSE_REFRESH_PROFILE",
    "VLLM_SPARSE_REFRESH_PROFILE_DETAIL",
    "VLLM_SPARSE_REFRESH_PROFILE_CALL_MIN",
    "VLLM_SPARSE_REFRESH_PROFILE_EVERY",
    "VLLM_SPARSE_REFRESH_PROFILE_LOG",
)

# A speed child must not inherit observation or experimental-ablation state
# from the caller shell.  Keep this classification in the pairing-contract
# module so environment cleanup, artifact proof, and pair normalization use
# one source of truth instead of independent name lists.
SPEED_CHILD_OBSERVATION_ENV_MARKERS = (
    "TRACE",
    "PROFILE",
    "PROFILER",
    "TIMELINE",
    "TIMING",
    "DEBUG",
    "DUMP",
    "LEDGER",
    "FORENSIC",
    "PROBE",
    "COMPARE",
    "FAULTHANDLER",
    "TEE",
)
SPEED_CHILD_FORBIDDEN_EXPERIMENT_ENV_MARKERS = (
    "ASSERT",
    "ABLATE",
    "FASTHIT",
)
SPEED_CHILD_OBSERVATION_ENV_EXACT_KEYS = frozenset(
    {
        "CUDA_LAUNCH_BLOCKING",
        "NCCL_DEBUG",
        "NCCL_DEBUG_SUBSYS",
        "TORCH_COMPILE_DEBUG",
        "TORCH_DISTRIBUTED_DEBUG",
        "TORCH_LOGS",
        "TORCH_TRACE",
        # vLLM treats DEBUG logging as a runtime validation mode: its CUDA
        # piecewise graph replay rebuilds and compares input-address lists on
        # every call.  A custom logging config can likewise install hot-path
        # handlers, so final timing uses the built-in default configuration.
        "VLLM_LOGGING_CONFIG_PATH",
        "VLLM_LOGGING_LEVEL",
        # Project-local diagnostics whose names intentionally do not use a
        # broad TRACE/PROFILE marker.  Keep these exact: CHECK/VALIDATE are
        # also reasonable names for production policy and must not become a
        # catch-all speed-child deletion rule.
        "VLLM_SPARSE_EVT_BISECT",
        "VLLM_SPARSE_FA3_ROUTE_COUNTER_ENABLED",
        "VLLM_SPARSE_FA3_ROUTE_COUNTER_SLOTS",
        "VLLM_SPARSE_REFRESH_REBUILD_CHECK",
        "VLLM_SPARSE_VALIDATE_LAYER_SLOT_MAP",
        "VLLM_SPARSE_VALIDATE_LSR_CACHE_KEY",
        "VLLM_SPARSE_VALIDATE_META_CONTRACT",
        "VLLM_SPARSE_WRITER_INPUT_BTABLE_CHECK",
    }
)
SPEED_CHILD_FORBIDDEN_EXPERIMENT_ENV_EXACT_KEYS = frozenset(
    {
        # Attribution modes deliberately change admission/producer work.
        "VLLM_SPARSE_ATTRIB_COMPACT_CONSUME_DELAY_STEPS",
        "VLLM_SPARSE_ATTRIB_PREFILL_PRODUCER",
        # Test-only live env reads add hot-path lookups throughout the runtime.
        "VLLM_SPARSE_DYNAMIC_ENV",
        # Unpromoted replay cuts must not leak into a final timing child.
        "VLLM_SPARSE_FULL_CUDAGRAPH_REPLAY_REFRESH_DEFER_COMMIT",
        "VLLM_SPARSE_FULL_CUDAGRAPH_REPLAY_REFRESH_SKIP_SUBMIT_SUMMARY",
        # Promoted production paths: the env surface is rollback/diagnostic,
        # not an ambient final-speed tuning contract.
        "VLLM_SPARSE_CLEAN_METADATA",
        "VLLM_SPARSE_PSC_BSKIP",
        "VLLM_SPARSE_PSC_BSKIP_LOGF",
        "VLLM_SPARSE_RRP_DISABLE_DIRECT_AFFINE",
    }
)
# Selector knobs are part of pair identity, but this module does not own their
# values. The retired value mapping looked like a policy manifest without ever
# applying or validating it; keep only the honest identity role.
SELECTOR_POLICY_IDENTITY_ENV_KEYS = (
    "VLLM_SPARSE_SELECTOR_CUDA_PIPELINE",
    "VLLM_SPARSE_SELECTOR_PIPELINE_WORKSPACE",
    "VLLM_SPARSE_SELECTOR_TRUSTED_SHAPES",
    "VLLM_SPARSE_SELECTOR_KEY_NORMS_CACHE_CAP",
    "VLLM_SPARSE_SELECTOR_FAST_SIG",
    "VLLM_SPARSE_SELECTOR_CPP_PREPROC",
    "VLLM_SPARSE_SELECTOR_CPP_STACK",
    "VLLM_SPARSE_SELECTOR_PIPELINE_UNIFIED",
    "VLLM_SPARSE_SELECTOR_LOGS_CACHE_R",
    "VLLM_SPARSE_SELECTOR_LOGS_CUDA",
    "VLLM_SPARSE_SELECTOR_LOGS_FAST_MATH",
    "VLLM_SPARSE_SELECTOR_FIXED_K",
    "VLLM_SPARSE_SELECTOR_SELECTED_INDICES_OUT",
    "VLLM_SPARSE_SELECTOR_PIPELINE_WORKSPACE_RUNTIME",
    "VLLM_SPARSE_SELECTOR_FIXED_SHAPE_TOPK",
    "VLLM_SPARSE_SELECTOR_KBUCKET",
    "VLLM_SPARSE_SELECTOR_TOPK_GRAPH",
    "VLLM_SPARSE_SELECTOR_FUSE_NMS_CROSS",
)
REMOTE_SELECTOR_ADAPTIVE_OVERRIDE_ENVS = (
    "VLLM_SPARSE_SELECTOR_CROSS_HEAD_BLOCK_K",
    "VLLM_SPARSE_SELECTOR_LOGS_THREADS",
)
# Production tuning is allowed in matched experiments, but it must remain
# visible to the speed/diagnostic pairing digest.  This list is deliberately
# exact so adding an unrelated VLLM_* variable cannot silently redefine run
# identity.
SPEED_CHILD_PAIRING_IDENTITY_ENV_KEYS = (
    "VLLM_SOURCE_ROOT",
    "VLLM_ALLREDUCE_USE_FLASHINFER",
    "VLLM_ALLREDUCE_USE_SYMM_MEM",
    "VLLM_USE_NCCL_SYMM_MEM",
    "VLLM_SPARSE_FULL_CUDAGRAPH_REPLAY_REFRESH_BATCHED_FLUSH",
    "VLLM_SPARSE_FULL_CUDAGRAPH_REPLAY_REFRESH_DEFER_TO_DEADLINE",
    "VLLM_SPARSE_REFRESH_REBUILD_MAX_DELAY_STEPS",
    "VLLM_SPARSE_REFRESH_SPLIT_SELECTOR_WRITER_RELEASE",
    "VLLM_SPARSE_REFRESH_SPLIT_SELECTOR_WRITER_RELEASE_MAX_PER_HANDLE",
    "VLLM_SPARSE_REFRESH_SPLIT_SELECTOR_WRITER_RELEASE_MIN_LAYER_START",
    "VLLM_SPARSE_REFRESH_STREAM_PRIORITY",
    "VLLM_SPARSE_REPLAY_REFRESH_PROGRESSIVE_CONSUME",
    "VLLM_SPARSE_WRITER_TOKEN_TILE",
    *SELECTOR_POLICY_IDENTITY_ENV_KEYS,
    *REMOTE_SELECTOR_ADAPTIVE_OVERRIDE_ENVS,
)
NATIVE_ROW_SOURCE_KEYS = (
    "native_rows",
    "native_pages",
    "native_full_block_table_rows",
)
STAGE_A_SOURCE_SUMMARY_KEYS = (
    "source_counter_schema_version",
    "expected_rows",
    "num_kv_heads",
)
STAGE_A_SOURCE_COUNTER_SCHEMA_VERSION = 1
STAGE_A_ROW_SOURCE_KEYS = (
    "compact_rows",
    "native_canonical_pages",
    "compact_rows_with_reserved_pages",
    "compact_rows_with_recent_pages",
    "compact_reserved_pages",
    "recent_canonical_pages",
    "middle_native_canonical_pages",
    "native_rows",
    "compact_full_native_fallback_rows",
)
NATIVE_CANONICAL_PAGES_KEY = "native_canonical_pages"


@dataclass(frozen=True)
class RouteProofResult:
    passed: bool
    reasons: tuple[str, ...]


def build_config_digest(config: Mapping[str, Any]) -> str:
    payload = _canonical_json(config)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_pairing_digest(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        _canonical_json(_normalize_pairing_config(config)).encode("utf-8")
    ).hexdigest()


def diff_config_inputs(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> dict[str, tuple[Any, Any]]:
    diffs: dict[str, tuple[Any, Any]] = {}
    _collect_diffs("", left, right, diffs)
    return diffs


def only_allowed_trace_diffs(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> bool:
    return all(_is_allowed_trace_diff(key) for key in diff_config_inputs(left, right))


def classify_speed_child_env_key(key: str) -> str | None:
    """Classify caller state that is forbidden in a timed child.

    ``observation`` keys may be re-enabled only in the separate diagnostic
    child and are ignored by the speed/diagnostic pairing digest.  ``experiment``
    keys alter execution or add shadow work; both children clear them and the
    pairing digest deliberately does not hide a future mismatch.
    """
    upper = str(key).upper()
    if upper in SPEED_CHILD_OBSERVATION_ENV_EXACT_KEYS:
        return "observation"
    if upper in SPEED_CHILD_FORBIDDEN_EXPERIMENT_ENV_EXACT_KEYS:
        return "experiment"
    if not upper.startswith("VLLM_"):
        return None
    if any(
        marker in upper
        for marker in SPEED_CHILD_FORBIDDEN_EXPERIMENT_ENV_MARKERS
    ):
        return "experiment"
    if (
        any(marker in upper for marker in SPEED_CHILD_OBSERVATION_ENV_MARKERS)
        or upper.endswith("_LOG")
        or "_LOG_" in upper
    ):
        return "observation"
    return None


def validate_shared_route_proof(summary: Mapping[str, Any]) -> RouteProofResult:
    reasons: list[str] = []
    row_modes = _mapping(summary.get("row_mode_distribution"))
    row_sources = _mapping(summary.get("row_source_distribution"))
    missing_source_fields = _stage_a_source_counter_missing_fields(
        summary,
        row_sources,
    )

    if not bool(summary.get("fa3_backend", False)):
        reasons.append("fa3_backend_missing")
    if not bool(summary.get("full_cuda_graph", False)):
        reasons.append("full_cuda_graph_missing")
    if _as_int(summary.get("resolved_row_ptr_fwd_mixed_page_count")) <= 0:
        reasons.append("resolved_row_ptr_fwd_mixed_page_missing")
    wrapper_count = _mixed_wrapper_count(summary)
    if wrapper_count is None:
        reasons.append("mixed_wrapper_event_count_missing")
    elif wrapper_count <= 0:
        reasons.append("mixed_wrapper_event_count_zero")
    if _actual_mixed_page_count(summary) <= 0:
        reasons.append("actual_fwd_mixed_page_missing")
    if _as_int(row_modes.get("compact")) <= 0 and _as_int(row_sources.get("compact_rows")) <= 0:
        reasons.append("compact_rows_missing")
    if "native" not in row_modes:
        reasons.append("row_mode_native_missing")
    elif _as_int(row_modes.get("native")) != 0:
        reasons.append("native_rows_nonzero")
    if missing_source_fields:
        reasons.append("source_provenance_missing")
        reasons.extend(f"{field}_missing" for field in missing_source_fields)
    compact_rows = _as_int(row_sources.get("compact_rows"))
    inactive_rows = (
        _as_int(row_sources.get("inactive_rows"))
        if "inactive_rows" in row_sources
        else 0
    )
    if _as_int(row_sources.get("compact_rows_with_reserved_pages")) != compact_rows:
        reasons.append("compact_rows_with_reserved_pages_mismatch")
    if _as_int(row_sources.get("compact_rows_with_recent_pages")) != compact_rows:
        reasons.append("compact_rows_with_recent_pages_mismatch")
    expected_rows = _as_int(summary.get("expected_rows"))
    num_kv_heads = _as_int(summary.get("num_kv_heads"))
    if (
        "expected_rows" in summary
        and "num_kv_heads" in summary
        and compact_rows + inactive_rows != expected_rows * num_kv_heads
    ):
        reasons.append("compact_rows_mismatch_expected_rows_heads")
    if (
        "compact_reserved_pages" in row_sources
        and _as_int(row_sources.get("compact_reserved_pages")) <= 0
    ):
        reasons.append("compact_reserved_pages_nonpositive")
    if (
        "recent_canonical_pages" in row_sources
        and _as_int(row_sources.get("recent_canonical_pages")) <= 0
    ):
        reasons.append("recent_canonical_pages_nonpositive")
    if _as_int(row_sources.get("compact_full_native_fallback_rows")) != 0:
        reasons.append("compact_full_native_fallback_rows_nonzero")
    if _as_int(row_sources.get(NATIVE_CANONICAL_PAGES_KEY)) != 0:
        reasons.append("native_canonical_pages_nonzero")
    if _as_int(row_sources.get("middle_native_canonical_pages")) != 0:
        reasons.append("middle_native_canonical_pages_nonzero")
    for key in NATIVE_ROW_SOURCE_KEYS:
        if _as_int(row_sources.get(key)) != 0:
            reasons.append(f"{key}_nonzero")
    _require_zero_counter(summary, "triton_attention_call_count", reasons)
    _require_zero_counter(summary, "dense_fallback_call_count", reasons)
    _require_zero_counter(summary, "native_full_block_table_rows", reasons)
    if _as_int(summary.get("page_resolver_kind4_count")) <= 0:
        reasons.append("page_resolver_kind4_count_missing")
    for key in (
        "kind2_dispatch_count",
        "kind3_dispatch_count",
        "selected_table_publish_count",
        "vector_fallback_rows",
        "full_fallback_rows",
    ):
        _require_zero_counter(summary, key, reasons)
    if bool(summary.get("diagnostic_flags_enabled", False)):
        reasons.append("diagnostic_flags_enabled_true")
    if not str(summary.get("controller_json_sha256", "") or ""):
        reasons.append("controller_json_sha256_missing")
    if not str(summary.get("fa3_so_sha256", "") or ""):
        reasons.append("fa3_so_sha256_missing")
    if _as_int(summary.get("graph_route_family_mismatch_count")) != 0:
        reasons.append("graph_route_family_mismatch_nonzero")
    _require_native_compact_residency(summary, reasons)

    return RouteProofResult(passed=not reasons, reasons=tuple(dict.fromkeys(reasons)))


def _collect_diffs(
    prefix: str,
    left: Any,
    right: Any,
    diffs: dict[str, tuple[Any, Any]],
) -> None:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        for key in sorted(set(left) | set(right), key=str):
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            _collect_diffs(child_prefix, left.get(key), right.get(key), diffs)
        return
    if left != right:
        diffs[prefix] = (left, right)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    )


def _normalize_pairing_config(value: Any) -> Any:
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, child in value.items():
            if key == "env" and isinstance(child, Mapping):
                normalized[str(key)] = {
                    str(env_key): _normalize_pairing_config(env_value)
                    for env_key in child
                    for env_value in (child[env_key],)
                    if classify_speed_child_env_key(str(env_key)) != "observation"
                }
            else:
                normalized[str(key)] = _normalize_pairing_config(child)
        return normalized
    if isinstance(value, list):
        return [_normalize_pairing_config(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_normalize_pairing_config(item) for item in value)
    return value


def _is_allowed_trace_diff(key: str) -> bool:
    if not key.startswith("env."):
        return False
    return (
        classify_speed_child_env_key(key.removeprefix("env.")) == "observation"
    )


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _stage_a_source_counter_missing_fields(
    summary: Mapping[str, Any],
    row_sources: Mapping[str, Any],
) -> tuple[str, ...]:
    missing: list[str] = []
    for key in STAGE_A_SOURCE_SUMMARY_KEYS:
        if key not in summary:
            missing.append(key)
    for key in ("expected_rows", "num_kv_heads"):
        if key in summary and _as_int(summary.get(key)) <= 0:
            missing.append(f"{key}_nonpositive")
    if (
        "source_counter_schema_version" in summary
        and _as_int(summary.get("source_counter_schema_version"))
        != STAGE_A_SOURCE_COUNTER_SCHEMA_VERSION
    ):
        missing.append("source_counter_schema_version_unsupported")
    for key in STAGE_A_ROW_SOURCE_KEYS:
        if key not in row_sources:
            missing.append(key)
    for key in _string_list(summary.get("source_counter_missing_fields")):
        if key not in missing:
            missing.append(key)
    for key in _string_list(row_sources.get("source_counter_missing_fields")):
        if key not in missing:
            missing.append(key)
    return tuple(missing)


def _string_list(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,) if value else ()
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(item) for item in value if str(item))


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _require_zero_counter(
    summary: Mapping[str, Any],
    key: str,
    reasons: list[str],
) -> None:
    if key not in summary:
        reasons.append(f"{key}_missing")
    elif _as_int(summary.get(key)) != 0:
        reasons.append(f"{key}_nonzero")


def _require_native_compact_residency(
    summary: Mapping[str, Any],
    reasons: list[str],
) -> None:
    owner_key = "compact_kv_storage_owner"
    if owner_key not in summary:
        reasons.append(f"{owner_key}_missing")
    elif str(summary.get(owner_key, "") or "") != "native_vllm_page_kv":
        reasons.append("compact_kv_storage_owner_not_native_vllm_page_kv")
    missing_owner_key = "compact_kv_storage_owner_missing_count"
    if missing_owner_key not in summary:
        reasons.append(f"{missing_owner_key}_missing")
    elif _as_int(summary.get(missing_owner_key)) != 0:
        reasons.append(f"{missing_owner_key}_nonzero")

    bind_key = "compact_kv_native_residency_bind_count"
    if bind_key not in summary:
        reasons.append(f"{bind_key}_missing")
    elif _as_int(summary.get(bind_key)) <= 0:
        reasons.append("compact_kv_native_residency_bind_count_zero")

    for key in (
        "compact_kv_storage_mismatch_count",
        "compact_kv_reserved_span_mismatch_count",
    ):
        if key not in summary:
            reasons.append(f"{key}_missing")
        elif _as_int(summary.get(key)) != 0:
            reasons.append(f"{key}_nonzero")

    copy_key = "compact_to_compact_copy_bytes"
    if copy_key not in summary:
        reasons.append(f"{copy_key}_missing")
    elif _as_int(summary.get(copy_key)) != 0:
        reasons.append(f"{copy_key}_nonzero")


def _actual_mixed_page_count(summary: Mapping[str, Any]) -> int:
    for key in ("actual_fwd_mixed_page_count", "fwd_mixed_page_call", "mixed_page_call_count"):
        count = _as_int(summary.get(key))
        if count > 0:
            return count
    return 0


def _mixed_wrapper_count(summary: Mapping[str, Any]) -> int | None:
    for key in ("mixed_wrapper_event_count", "fwd_mixed_page_call"):
        if key in summary:
            return _as_int(summary.get(key))
    return None
