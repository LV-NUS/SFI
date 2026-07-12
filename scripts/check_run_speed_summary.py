#!/usr/bin/env python3
"""Fail-closed postflight for scripts/run_speed.sh."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
try:
    sys.path.remove(str(_REPO_ROOT))
except ValueError:
    pass
sys.path.insert(0, str(_REPO_ROOT))

from utils.selector_pipeline_identity import (
    SELECTOR_PIPELINE_EXTENSION_NAME,
    SELECTOR_PIPELINE_SEMANTIC_VERSION,
)


EXPECTED_GATE_NOISE = {
    "semantic_output_health_not_ok:unknown_without_reference",
    "interval_trigger_intents_below_expected",
}
EXPECTED_HARNESS_RETURNCODES = {0, 2}


def _failure_returncode(harness_returncode: int) -> int:
    return harness_returncode if harness_returncode != 0 else 1


def _is_zero_int(value: object) -> bool:
    return type(value) is int and value == 0


def _is_exact_positive_int(value: object, expected: object) -> bool:
    return (
        type(value) is int
        and type(expected) is int
        and expected > 0
        and value == expected
    )


def _is_finite_positive_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def _is_expected_gate_noise(reason: object) -> bool:
    return str(reason) in EXPECTED_GATE_NOISE


def _reject_nonstandard_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON numeric constant: {value}")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    with os.fdopen(fd, "rb") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise OSError(f"not a regular file: {path}")
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selector_artifact_reasons(
    summary: dict[str, Any],
    *,
    selector_cache_root: Path | None,
) -> list[str]:
    prewarm_artifact = summary.get("selector_pipeline_artifact_prewarm")
    postflight_artifact = summary.get("selector_pipeline_artifact_postflight")
    provenance = summary.get("run_provenance")
    if not isinstance(prewarm_artifact, dict):
        return ["selector_pipeline_artifact_prewarm_missing"]
    if not isinstance(postflight_artifact, dict):
        return ["selector_pipeline_artifact_postflight_missing"]
    if not isinstance(provenance, dict):
        return ["selector_pipeline_artifact_provenance_missing"]
    if (
        prewarm_artifact != provenance.get("selector_pipeline_artifact_prewarm")
        or postflight_artifact
        != provenance.get("selector_pipeline_artifact_postflight")
        or bool(summary.get("selector_pipeline_artifact_mutated_after_prewarm"))
        != bool(
            provenance.get("selector_pipeline_artifact_mutated_after_prewarm")
        )
    ):
        return ["selector_pipeline_artifact_provenance_mismatch"]

    artifact_path = Path(str(prewarm_artifact.get("path", "") or ""))
    prewarm_sha256 = str(prewarm_artifact.get("sha256", "") or "").lower()
    postflight_path = str(postflight_artifact.get("path", "") or "")
    postflight_sha256 = str(postflight_artifact.get("sha256", "") or "").lower()
    try:
        semantic_version = int(prewarm_artifact.get("semantic_version", -1))
        expected_semantic_version = int(
            prewarm_artifact.get("expected_semantic_version", -1)
        )
    except (TypeError, ValueError):
        semantic_version = -1
        expected_semantic_version = -1
    reasons: list[str] = []
    if semantic_version <= 0:
        reasons.append("selector_pipeline_semantic_version_invalid")
    if expected_semantic_version <= 0:
        reasons.append("selector_pipeline_expected_semantic_version_invalid")
    elif semantic_version != expected_semantic_version:
        reasons.append(
            "selector_pipeline_semantic_version_mismatch:"
            f"actual={semantic_version}:expected={expected_semantic_version}"
        )
    if semantic_version != SELECTOR_PIPELINE_SEMANTIC_VERSION:
        reasons.append(
            "selector_pipeline_semantic_version_source_mismatch:"
            f"actual={semantic_version}:source={SELECTOR_PIPELINE_SEMANTIC_VERSION}"
        )
    if expected_semantic_version != SELECTOR_PIPELINE_SEMANTIC_VERSION:
        reasons.append(
            "selector_pipeline_expected_semantic_version_source_mismatch:"
            f"summary={expected_semantic_version}:source={SELECTOR_PIPELINE_SEMANTIC_VERSION}"
        )
    expected_artifact_name = f"{SELECTOR_PIPELINE_EXTENSION_NAME}.so"
    if artifact_path.name != expected_artifact_name:
        reasons.append(
            "selector_pipeline_artifact_name_mismatch:"
            f"actual={artifact_path.name}:expected={expected_artifact_name}"
        )
    if selector_cache_root is None:
        reasons.append("selector_pipeline_trusted_cache_root_missing")
    else:
        expected_artifact_path = (
            Path(selector_cache_root)
            / SELECTOR_PIPELINE_EXTENSION_NAME
            / expected_artifact_name
        ).resolve(strict=False)
        if artifact_path.resolve(strict=False) != expected_artifact_path:
            reasons.append(
                "selector_pipeline_artifact_outside_trusted_cache_root:"
                f"actual={artifact_path}:expected={expected_artifact_path}"
            )
    if re.fullmatch(r"[0-9a-f]{64}", prewarm_sha256) is None:
        reasons.append("selector_pipeline_artifact_prewarm_sha256_invalid")
    if re.fullmatch(r"[0-9a-f]{64}", postflight_sha256) is None:
        reasons.append("selector_pipeline_artifact_postflight_sha256_invalid")
    mutated_after_prewarm = bool(
        str(artifact_path) != postflight_path
        or prewarm_sha256 != postflight_sha256
    )
    if mutated_after_prewarm:
        reasons.append("selector_artifact_mutated_after_prewarm")
    recorded_mutated = bool(
        summary.get("selector_pipeline_artifact_mutated_after_prewarm", False)
    )
    recorded_reasons = list(summary.get("selector_pipeline_artifact_reasons") or [])
    if recorded_mutated != mutated_after_prewarm:
        reasons.append("selector_pipeline_artifact_mutation_flag_mismatch")
    if mutated_after_prewarm and "selector_artifact_mutated_after_prewarm" not in recorded_reasons:
        reasons.append("selector_pipeline_artifact_mutation_reason_missing")
    if not artifact_path.is_file():
        reasons.append("selector_pipeline_artifact_file_missing")
    elif not mutated_after_prewarm:
        try:
            actual_sha256 = _file_sha256(artifact_path)
        except OSError:
            reasons.append("selector_pipeline_artifact_file_unreadable")
        else:
            if actual_sha256 != postflight_sha256:
                reasons.append("selector_artifact_mutated_after_postflight")
    return reasons


def _reference_gate_integrity_reasons(
    summary: dict[str, Any],
    *,
    mode: str,
) -> list[str]:
    if mode == "dense":
        return []
    provenance = summary.get("run_provenance")
    outputs_include_text = bool(
        isinstance(provenance, dict)
        and provenance.get("outputs_include_text") is True
    )
    if not outputs_include_text:
        return []

    reference_reasons = summary.get("reference_gate_reasons")
    if not isinstance(reference_reasons, list):
        return ["reference_gate_reasons_missing_or_invalid"]
    reference_passed = summary.get("reference_gate_passed")
    if summary.get("skip_dense_reference") is True:
        if reference_passed is not None:
            return ["debug_dense_reference_gate_must_be_none"]
        if reference_reasons != ["dense_reference_skipped_for_debug"]:
            return ["debug_dense_reference_reason_mismatch"]
        return []
    if reference_passed is not True:
        return ["dense_reference_gate_not_green"]
    if reference_reasons:
        return ["dense_reference_gate_reasons_nonempty"]
    return []


def _harness_condition_reasons(
    summary: dict[str, Any],
    *,
    mode: str,
    semantic_gate_reasons: list[object],
    producer_gate_reasons: list[object],
    run_nonce: str,
) -> list[str]:
    reasons: list[str] = []
    provenance = summary.get("run_provenance")
    if not run_nonce:
        reasons.append("run_nonce_check_missing")
    elif not isinstance(provenance, dict) or provenance.get("run_nonce") != run_nonce:
        reasons.append("run_nonce_mismatch_or_missing")
    if isinstance(provenance, dict):
        expected_kv_status = "passed" if mode == "sparse" else "not_applicable"
        if provenance.get("runner_kv_preflight_status") != expected_kv_status:
            reasons.append("runner_kv_preflight_not_verdict_grade")
        if provenance.get("runner_gpu_lock_mode") != "exclusive":
            reasons.append("runner_gpu_lock_not_exclusive")
        if provenance.get("runner_fa3_preflight_status") != "passed":
            reasons.append("runner_fa3_preflight_not_green")
        if provenance.get("runner_corpus_token_status") not in {
            "cache_v2_exact",
            "generated_exact",
            "validated_exact",
        }:
            reasons.append("runner_corpus_token_status_not_exact")
    output_length_gate = summary.get("output_length_gate")
    if not isinstance(output_length_gate, dict):
        reasons.append("output_length_gate_missing_or_invalid")
    else:
        if output_length_gate.get("required") is not True:
            reasons.append("output_length_gate_not_required")
        if output_length_gate.get("exact_length_required") is not True:
            reasons.append("output_length_exact_length_not_required")
        if output_length_gate.get("passed") is not True:
            reasons.append("output_length_gate_not_green")
        if output_length_gate.get("reasons") != []:
            reasons.append("output_length_gate_reasons_nonempty")
        if summary.get("output_length_gate_passed") is not True:
            reasons.append("output_length_gate_passed_not_green")
        if isinstance(provenance, dict):
            expected_requests = provenance.get("batch_size")
            expected_tokens = provenance.get("max_new_tokens_effective")
            if (
                not _is_exact_positive_int(
                    output_length_gate.get("expected_request_count"),
                    expected_requests,
                )
                or not _is_exact_positive_int(
                    output_length_gate.get("actual_request_count"),
                    expected_requests,
                )
            ):
                reasons.append("output_length_request_count_mismatch")
            if (
                not _is_exact_positive_int(
                    output_length_gate.get("expected_output_tokens"),
                    expected_tokens,
                )
                or not _is_exact_positive_int(
                    output_length_gate.get("min_output_tokens"),
                    expected_tokens,
                )
                or not _is_exact_positive_int(
                    output_length_gate.get("max_output_tokens"),
                    expected_tokens,
                )
            ):
                reasons.append("output_length_token_count_mismatch")
            token_lengths = output_length_gate.get("token_lengths")
            if (
                not isinstance(token_lengths, list)
                or len(token_lengths) != expected_requests
                or any(
                    not _is_exact_positive_int(length, expected_tokens)
                    for length in token_lengths
                )
            ):
                reasons.append("output_length_token_lengths_mismatch")
    if not _is_zero_int(summary.get("returncode")):
        reasons.append(f"speed_child_returncode_nonzero:{summary.get('returncode')}")
    if summary.get("timed_out") is not False:
        reasons.append("speed_child_timed_out_or_missing")
    if summary.get("speed_child_fatal_error_detected") is not False:
        reasons.append("speed_child_fatal_error_detected")

    verdict_only = summary.get("verdict_only") if mode == "sparse" else False
    if mode == "sparse" and verdict_only is not True and verdict_only is not False:
        reasons.append("verdict_only_state_missing_or_invalid")
    diagnostic_fields_present = bool(
        "diagnostic_returncode" in summary or "diagnostic_timed_out" in summary
    )
    if verdict_only is not True or diagnostic_fields_present:
        if not _is_zero_int(summary.get("diagnostic_returncode")):
            reasons.append(
                "diagnostic_child_returncode_nonzero_or_missing:"
                f"{summary.get('diagnostic_returncode')}"
            )
        if summary.get("diagnostic_timed_out") is not False:
            reasons.append("diagnostic_child_timed_out_or_missing")
    if summary.get("diagnostic_child_fatal_error_detected") is not False:
        reasons.append("diagnostic_child_fatal_error_detected")

    route_proof = summary.get("route_proof")
    if not isinstance(route_proof, dict) or route_proof.get("passed") is not True:
        reasons.append("route_proof_not_green")
    if summary.get("route_proof_passed") is not True:
        reasons.append("route_proof_passed_not_green")
    if summary.get("speed_child_route_proof_passed") is not True:
        reasons.append("speed_child_route_proof_not_green")
    if (
        mode == "sparse"
        and summary.get("workload_plan_replay_counts_match") is not True
    ):
        reasons.append("workload_plan_replay_counts_mismatch_or_missing")

    lifecycle_reasons = summary.get("sparse_native_lifecycle_gate_reasons")
    if not isinstance(lifecycle_reasons, list):
        reasons.append("sparse_native_lifecycle_gate_reasons_missing_or_invalid")
    elif lifecycle_reasons:
        reasons.append("sparse_native_lifecycle_gate_failed")

    if mode == "sparse":
        if summary.get("selector_extension_prewarm_enabled") is not True:
            reasons.append("selector_extension_prewarm_not_enabled")
        if not _is_zero_int(summary.get("selector_extension_prewarm_returncode")):
            reasons.append("selector_extension_prewarm_returncode_nonzero_or_missing")
        if summary.get("selector_extension_prewarm_timed_out") is not False:
            reasons.append("selector_extension_prewarm_timed_out_or_missing")

    if producer_gate_reasons:
        if summary.get("producer_gate_passed") is not False:
            reasons.append("producer_gate_state_inconsistent_with_reasons")
    elif summary.get("producer_gate_passed") is not True:
        reasons.append("producer_gate_not_green_without_reason")

    all_gate_reasons = semantic_gate_reasons + producer_gate_reasons
    only_allowed_gate_noise = mode == "sparse" and bool(all_gate_reasons) and all(
        _is_expected_gate_noise(reason) for reason in all_gate_reasons
    )
    production_gate_passed = summary.get("production_gate_passed")
    if only_allowed_gate_noise:
        if production_gate_passed is not False:
            reasons.append("production_gate_state_inconsistent_with_allowed_noise")
    elif production_gate_passed is not True:
        reasons.append("production_gate_not_green_without_only_allowed_noise")

    reasons.extend(_reference_gate_integrity_reasons(summary, mode=mode))
    return reasons


def check_run_speed_summary(
    summary_path: Path,
    *,
    mode: str,
    harness_returncode: int,
    run_started_ns: int,
    run_nonce: str,
    selector_cache_root: Path | None,
) -> tuple[int, list[str]]:
    messages: list[str] = []
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        summary_fd = os.open(summary_path, flags)
    except OSError as exc:
        messages.append(f"FAIL: no summary produced ({exc}); check the matching .log")
        return _failure_returncode(harness_returncode), messages
    try:
        with os.fdopen(summary_fd, "r", encoding="utf-8") as handle:
            summary_stat = os.fstat(handle.fileno())
            if not stat.S_ISREG(summary_stat.st_mode):
                raise OSError("summary is not a regular file")
            if int(summary_stat.st_mtime_ns) < int(run_started_ns):
                messages.append(
                    "FAIL: summary predates this run; refusing to reuse a stale artifact"
                )
                return _failure_returncode(harness_returncode), messages
            summary = json.load(
                handle,
                parse_constant=_reject_nonstandard_json_constant,
            )
    except (OSError, ValueError) as exc:
        messages.append(f"FAIL: invalid summary ({exc}); check the matching .log")
        return _failure_returncode(harness_returncode), messages
    if not isinstance(summary, dict):
        messages.append("FAIL: summary root must be a JSON object")
        return _failure_returncode(harness_returncode), messages

    tps = summary.get("decode_tps")
    messages.append(f"decode_tps={tps}")
    tps_valid = _is_finite_positive_number(tps)
    if not tps_valid:
        messages.append("  decode_tps must be a finite positive JSON number")
    all_decode_tps = summary.get("all_decode_tps")
    all_decode_tps_valid = _is_finite_positive_number(all_decode_tps)
    if all_decode_tps is not None:
        messages.append(f"all_decode_tps={all_decode_tps}")
    if not all_decode_tps_valid:
        messages.append("  all_decode_tps must be a finite positive JSON number")
    for key in sorted(summary):
        value = summary[key]
        if (
            "refresh" in key
            and "count" in key
            and isinstance(value, (int, float))
            and value
        ):
            messages.append(f"  {key}={value}")

    fallbacks = summary.get("dense_native_fallback_count", 0) or 0
    route_proof = summary.get("route_proof")
    route_proof_reasons = list(
        route_proof.get("reasons") or [] if isinstance(route_proof, dict) else []
    )
    route_proof_reasons += list(summary.get("speed_child_route_proof_reasons") or [])
    semantic_gate_reasons_raw = summary.get("semantic_gate_reasons")
    producer_gate_reasons_raw = summary.get("producer_gate_reasons")
    gate_reason_shape_reasons: list[str] = []
    if not isinstance(semantic_gate_reasons_raw, list):
        gate_reason_shape_reasons.append("semantic_gate_reasons_missing_or_invalid")
        semantic_gate_reasons: list[object] = []
    else:
        semantic_gate_reasons = list(semantic_gate_reasons_raw)
    if not isinstance(producer_gate_reasons_raw, list):
        gate_reason_shape_reasons.append("producer_gate_reasons_missing_or_invalid")
        producer_gate_reasons: list[object] = []
    else:
        producer_gate_reasons = list(producer_gate_reasons_raw)
    gate_reasons = semantic_gate_reasons + producer_gate_reasons
    unexpected_gate_reasons = [
        reason
        for reason in gate_reasons
        if mode != "sparse" or not _is_expected_gate_noise(reason)
    ]
    documented_gate_noise_present = bool(
        mode == "sparse"
        and any(_is_expected_gate_noise(reason) for reason in gate_reasons)
    )
    selector_reasons = (
        _selector_artifact_reasons(
            summary,
            selector_cache_root=selector_cache_root,
        )
        if mode == "sparse"
        else []
    )
    harness_condition_reasons = gate_reason_shape_reasons + _harness_condition_reasons(
        summary,
        mode=mode,
        semantic_gate_reasons=semantic_gate_reasons,
        producer_gate_reasons=producer_gate_reasons,
        run_nonce=run_nonce,
    )

    if fallbacks:
        messages.append(f"  dense_native_fallback_count={fallbacks} (must be 0)")
    if unexpected_gate_reasons:
        messages.append(f"  unexpected gate reasons: {unexpected_gate_reasons}")
    if route_proof_reasons:
        messages.append(
            "  sparse route proof missing (sparse never engaged): "
            f"{route_proof_reasons[:6]}"
        )
    if selector_reasons:
        messages.append(f"  selector artifact check failed: {selector_reasons}")
    elif mode == "sparse":
        prewarm_artifact = summary["selector_pipeline_artifact_prewarm"]
        postflight_artifact = summary["selector_pipeline_artifact_postflight"]
        messages.append(
            "selector_pipeline_semantic_version="
            f"{prewarm_artifact['semantic_version']} "
            "expected_semantic_version="
            f"{prewarm_artifact['expected_semantic_version']} "
            f"prewarm_sha256={prewarm_artifact['sha256']} "
            f"postflight_sha256={postflight_artifact['sha256']} "
            f"path={prewarm_artifact['path']}"
        )
    if harness_condition_reasons:
        messages.append(
            f"  harness condition check failed: {harness_condition_reasons}"
        )

    summary_ok = bool(
        tps_valid
        and all_decode_tps_valid
        and not fallbacks
        and not unexpected_gate_reasons
        and not route_proof_reasons
        and not selector_reasons
        and not harness_condition_reasons
    )
    if harness_returncode not in EXPECTED_HARNESS_RETURNCODES:
        messages.append(f"HARNESS FAILED: returncode={harness_returncode}")
        messages.append("SPEED RUN CHECK FAILED")
        return harness_returncode, messages
    if harness_returncode == 2 and not documented_gate_noise_present:
        messages.append(
            "HARNESS returncode=2 rejected: current summary has no documented "
            "no-reference gate reason"
        )
        summary_ok = False
    if (
        harness_returncode == 0
        and summary.get("production_gate_passed") is not True
    ):
        messages.append(
            "HARNESS returncode=0 rejected: production gate is not explicitly green"
        )
        summary_ok = False
    if not summary_ok:
        messages.append("SPEED RUN CHECK FAILED")
        return (2 if harness_returncode == 2 else 1), messages

    if harness_returncode == 2:
        messages.append(
            "HARNESS returncode=2 accepted: current summary contains only "
            "the documented no-reference gate noise"
        )
    messages.append("SPEED RUN OK")
    return 0, messages


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--mode", choices=("sparse", "dense"), required=True)
    parser.add_argument("--harness-returncode", type=int, required=True)
    parser.add_argument("--run-started-ns", type=int, required=True)
    parser.add_argument("--run-nonce", required=True)
    parser.add_argument("--selector-cache-root", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    returncode, messages = check_run_speed_summary(
        args.summary,
        mode=args.mode,
        harness_returncode=args.harness_returncode,
        run_started_ns=args.run_started_ns,
        run_nonce=args.run_nonce,
        selector_cache_root=args.selector_cache_root,
    )
    for message in messages:
        print(message)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
