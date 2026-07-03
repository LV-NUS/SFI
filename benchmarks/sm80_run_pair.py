from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping


ALLOWED_TRACE_ENV_KEYS = (
    "VLLM_SPARSE_FA3_ROUTE_TRACE_LOG",
    "VLLM_SPARSE_FA3_STEP_TRACE_LOG",
    "VLLM_SPARSE_FULL_CUDAGRAPH_HOOK_PROFILE_LOG",
    "VLLM_SPARSE_MB_PROFILE_LOG",
    "VLLM_SPARSE_RRP_PREP_PROFILE_LOG",
    "VLLM_SPARSE_ONE_SHOT_TIMELINE_LOG",
    "VLLM_SPARSE_REFRESH_PROFILE",
    "VLLM_SPARSE_REFRESH_PROFILE_DETAIL",
    "VLLM_SPARSE_REFRESH_PROFILE_CALL_MIN",
    "VLLM_SPARSE_REFRESH_PROFILE_EVERY",
    "VLLM_SPARSE_REFRESH_PROFILE_LOG",
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
    "compact_rows_with_reserved_pages",
    "compact_rows_with_recent_pages",
    "compact_reserved_pages",
    "recent_canonical_pages",
    "middle_native_canonical_pages",
    "native_rows",
    "compact_full_native_fallback_rows",
)
LEGACY_MIDDLE_NATIVE_CANONICAL_KEY = "native_canonical_pages"


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
    if (
        LEGACY_MIDDLE_NATIVE_CANONICAL_KEY in row_sources
        and _as_int(row_sources.get(LEGACY_MIDDLE_NATIVE_CANONICAL_KEY)) != 0
    ):
        reasons.append("native_canonical_pages_nonzero")
    if _middle_native_canonical_pages(row_sources) != 0:
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
                    str(env_key): ""
                    if str(env_key) in ALLOWED_TRACE_ENV_KEYS
                    else _normalize_pairing_config(env_value)
                    for env_key in child
                    for env_value in (child[env_key],)
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
    return key.removeprefix("env.") in ALLOWED_TRACE_ENV_KEYS


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


def _middle_native_canonical_pages(row_sources: Mapping[str, Any]) -> int:
    if "middle_native_canonical_pages" in row_sources:
        return _as_int(row_sources.get("middle_native_canonical_pages"))
    return _as_int(row_sources.get(LEGACY_MIDDLE_NATIVE_CANONICAL_KEY))


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
