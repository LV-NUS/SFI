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
import subprocess
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
from utils.selector_log_s_identity import (
    SELECTOR_LOG_F_AMORTIZED_TILED_ALLOWED_ROUTES,
    SELECTOR_LOG_F_FAST_ROUTE,
    SELECTOR_LOG_F_GENERIC_ROUTE,
    SELECTOR_LOG_F_TILED_ROUTE,
    SELECTOR_LOG_S_EXTENSION_NAME,
    selector_log_s_artifact_identity,
    selector_log_s_runtime_proof_reasons,
)
from benchmarks.scheduler_contract import (
    SCHEDULER_GRAPH_CONTRACT_SCHEMA,
    SCHEDULER_GRAPH_RUNTIME_FIELDS,
)
from benchmarks.sm80_run_pair import build_config_digest
from scripts.check_tp8_arm_teardown import (
    ATTRIBUTION_SCOPE as TP8_ATTRIBUTION_SCOPE,
    arm_token_sha256,
    derive_arm_token,
)
from utils.model_kv_contract import (
    MODEL_KV_CONTRACT_SCHEMA,
    derive_model_kv_contract,
)
EXPECTED_GATE_NOISE = {
    "semantic_output_health_not_ok:unknown_without_reference",
    "interval_trigger_intents_below_expected",
}
EXPECTED_HARNESS_RETURNCODES = {0, 2}
RUN_SPEED_TIERS = (
    "bs8x12k",
    "bs8x16k",
    "bs4x24k",
    "bs2x30k",
    "tp8x64k",
)
RUNNER_CODE_SCOPE = "benchmarks,patches,scripts,utils,sitecustomize.py"
_RUNNER_CODE_SUFFIXES = (
    ".py",
    ".sh",
    ".patch",
    ".cu",
    ".cuh",
    ".cc",
    ".cpp",
    ".h",
    ".hpp",
    ".toml",
)


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


def _is_allowed_gate_noise(reason: object, *, exact_tier: bool) -> bool:
    """Keep directional no-reference noise out of the exact TP8 contract."""
    if exact_tier and str(reason) == (
        "semantic_output_health_not_ok:unknown_without_reference"
    ):
        return False
    return _is_expected_gate_noise(reason)


def _reject_nonstandard_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON numeric constant: {value}")


def _load_regular_json_object(path: Path) -> dict[str, Any]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    with os.fdopen(fd, "r", encoding="utf-8") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise OSError(f"not a regular file: {path}")
        payload = json.load(
            handle,
            parse_constant=_reject_nonstandard_json_constant,
        )
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return payload


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


def _live_code_scope_untracked_files() -> list[str]:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(_REPO_ROOT),
            "ls-files",
            "--others",
            "--exclude-standard",
            "--",
            "benchmarks",
            "patches",
            "scripts",
            "utils",
            "sitecustomize.py",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10.0,
    )
    return sorted(
        path
        for path in result.stdout.splitlines()
        if path.endswith(_RUNNER_CODE_SUFFIXES)
    )


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
    exact_tier: bool = False,
    allow_local_pair_comparison_only: bool = False,
) -> list[str]:
    if mode == "dense":
        return []
    provenance = summary.get("run_provenance")
    outputs_include_text = bool(
        isinstance(provenance, dict)
        and provenance.get("outputs_include_text") is True
    )
    if not outputs_include_text:
        if exact_tier:
            return ["exact_tier_dense_reference_text_contract_missing"]
        return []

    reference_reasons = summary.get("reference_gate_reasons")
    if not isinstance(reference_reasons, list):
        return ["reference_gate_reasons_missing_or_invalid"]
    reference_passed = summary.get("reference_gate_passed")
    if summary.get("skip_dense_reference") is True:
        if exact_tier:
            return ["exact_tier_dense_reference_skipped"]
        if reference_passed is not None:
            return ["debug_dense_reference_gate_must_be_none"]
        if reference_reasons != ["dense_reference_skipped_for_debug"]:
            return ["debug_dense_reference_reason_mismatch"]
        return []
    if allow_local_pair_comparison_only:
        if reference_passed is not False:
            return ["local_pair_comparison_reference_gate_must_be_false"]
        if reference_reasons != ["reference_semantic_mismatch"]:
            return ["local_pair_comparison_reference_reason_mismatch"]
        return []
    if reference_passed is not True:
        return ["dense_reference_gate_not_green"]
    if reference_reasons:
        return ["dense_reference_gate_reasons_nonempty"]
    return []


def _attention_build_identity_reasons(
    provenance: dict[str, Any],
    *,
    expected_cuda_arch: str,
    expected_attention_kernel: str,
) -> list[str]:
    identity = provenance.get("runner_attention_build_identity")
    if not isinstance(identity, dict):
        return ["runner_attention_build_identity_missing_or_invalid"]
    reasons: list[str] = []
    expected_backend = "fa4_cute" if expected_cuda_arch == "sm100" else "fa3"
    expected_fields = {
        "schema": "sfi.flash_attention_build_identity.v1",
        "architecture": expected_cuda_arch,
        "backend": expected_backend,
        "target": provenance.get("fa3_upstream_root"),
    }
    for field, expected in expected_fields.items():
        if identity.get(field) != expected:
            reasons.append(
                f"runner_attention_build_identity_mismatch:{field}:"
                f"actual={identity.get(field)!r}:expected={expected!r}"
            )
    for field, pattern in (
        ("provenance_sha256", r"[0-9a-f]{64}"),
        ("patch_sha256", r"[0-9a-f]{64}"),
        ("expected_patched_tree", r"[0-9a-f]{40}"),
    ):
        if re.fullmatch(pattern, str(identity.get(field, "") or "")) is None:
            reasons.append(f"runner_attention_build_identity_{field}_invalid")
    provenance_path = Path(str(identity.get("provenance_path", "") or ""))
    if not provenance_path.is_file():
        reasons.append("runner_attention_build_provenance_file_missing")
    else:
        try:
            current_provenance_sha = _file_sha256(provenance_path)
        except OSError:
            reasons.append("runner_attention_build_provenance_file_unreadable")
        else:
            if current_provenance_sha != identity.get("provenance_sha256"):
                reasons.append("runner_attention_build_provenance_mutated_after_preflight")
    if expected_attention_kernel == "fa3-native":
        shared_object = Path(str(identity.get("shared_object", "") or ""))
        shared_sha = str(identity.get("shared_object_sha256", "") or "")
        fa3_so = provenance.get("fa3_so")
        backend_artifact = provenance.get("backend_artifact")
        if (
            re.fullmatch(r"[0-9a-f]{64}", shared_sha) is None
            or not shared_object.is_file()
            or not isinstance(fa3_so, dict)
            or not isinstance(backend_artifact, dict)
            or fa3_so.get("path") != str(shared_object)
            or fa3_so.get("sha256") != shared_sha
            or backend_artifact.get("path") != str(shared_object)
            or backend_artifact.get("sha256") != shared_sha
        ):
            reasons.append("runner_attention_shared_object_identity_mismatch")
        else:
            try:
                current_shared_sha = _file_sha256(shared_object)
            except OSError:
                reasons.append("runner_attention_shared_object_unreadable")
            else:
                if current_shared_sha != shared_sha:
                    reasons.append("runner_attention_shared_object_mutated_after_run")
    elif identity.get("shared_object") or identity.get("shared_object_sha256"):
        reasons.append("runner_fa4_source_identity_claims_fa3_shared_object")
    return reasons


def _tp8_selector_log_s_runtime_reasons(
    metrics: dict[str, Any],
    *,
    tensor_parallel_size: int,
    selector_cache_root: Path,
) -> list[str]:
    reasons: list[str] = []
    expected_route = SELECTOR_LOG_F_TILED_ROUTE
    expected_extension = SELECTOR_LOG_S_EXTENSION_NAME
    phase_proofs: dict[str, list[dict[str, Any]]] = {}
    for phase, field in (
        ("reset", "speed_child_route_counter_reset_records"),
        ("snapshot", "speed_child_route_counter_snapshot_records"),
    ):
        records = metrics.get(field)
        if not isinstance(records, list) or len(records) != tensor_parallel_size:
            reasons.append(
                f"tp8_selector_log_s_{phase}_record_shape_mismatch"
            )
            continue
        proofs: list[dict[str, Any]] = []
        artifact_identities: list[tuple[object, ...]] = []
        route_signatures: list[tuple[object, ...]] = []
        for rank, record in enumerate(records):
            proof = (
                record.get("selector_log_s_runtime_proof")
                if isinstance(record, dict)
                else None
            )
            proof_reasons = selector_log_s_runtime_proof_reasons(
                proof,
                expected_route=expected_route,
                expected_phase=phase,
                allowed_routes=SELECTOR_LOG_F_AMORTIZED_TILED_ALLOWED_ROUTES,
            )
            reasons.extend(
                f"tp8_selector_log_s_{phase}_proof_invalid:rank={rank}:{reason}"
                for reason in proof_reasons
            )
            if not isinstance(proof, dict) or proof_reasons:
                continue
            proofs.append(proof)
            artifact_identities.append(selector_log_s_artifact_identity(proof))
            counts = proof["route_counts"]
            assert isinstance(counts, dict)
            route_signatures.append(
                (
                    proof["last_route"],
                    proof["last_dispatch_reason"],
                    proof["last_admission_identity"],
                    proof["admission_event_count"],
                    proof["admission_chain_sha256"],
                    counts[SELECTOR_LOG_F_FAST_ROUTE],
                    counts[SELECTOR_LOG_F_GENERIC_ROUTE],
                    counts[SELECTOR_LOG_F_TILED_ROUTE],
                    proof["tiled_cohort_count"],
                    proof["tiled_job_count"],
                    proof["tiled_direct_count"],
                    proof["tiled_kernel_launch_count"],
                    proof["tiled_admission_failure_count"],
                )
            )
            artifact_path = Path(str(proof["module_path"]))
            expected_path = (
                selector_cache_root
                / expected_extension
                / f"{expected_extension}.so"
            ).resolve(strict=False)
            if artifact_path.resolve(strict=False) != expected_path:
                reasons.append(
                    "tp8_selector_log_s_artifact_outside_trusted_cache:"
                    f"rank={rank}:actual={artifact_path}:expected={expected_path}"
                )
        if len(artifact_identities) != tensor_parallel_size or len(
            set(artifact_identities)
        ) != 1:
            reasons.append(
                f"tp8_selector_log_s_{phase}_artifact_rank_mismatch"
            )
        if len(route_signatures) != tensor_parallel_size or len(
            set(route_signatures)
        ) != 1:
            reasons.append(
                f"tp8_selector_log_s_{phase}_route_rank_mismatch"
            )
        phase_proofs[phase] = proofs

    reset_proofs = phase_proofs.get("reset", [])
    snapshot_proofs = phase_proofs.get("snapshot", [])
    if (
        len(reset_proofs) == tensor_parallel_size
        and len(snapshot_proofs) == tensor_parallel_size
    ):
        for rank, (reset_proof, snapshot_proof) in enumerate(
            zip(reset_proofs, snapshot_proofs, strict=True)
        ):
            if selector_log_s_artifact_identity(
                reset_proof
            ) != selector_log_s_artifact_identity(snapshot_proof):
                reasons.append(
                    f"tp8_selector_log_s_artifact_changed:rank={rank}"
                )
            reset_counts = reset_proof["route_counts"]
            snapshot_counts = snapshot_proof["route_counts"]
            assert isinstance(reset_counts, dict)
            assert isinstance(snapshot_counts, dict)
            for route in (
                SELECTOR_LOG_F_FAST_ROUTE,
                SELECTOR_LOG_F_GENERIC_ROUTE,
                SELECTOR_LOG_F_TILED_ROUTE,
            ):
                if int(snapshot_counts[route]) < int(reset_counts[route]):
                    reasons.append(
                        "tp8_selector_log_s_route_count_regressed:"
                        f"rank={rank}:route={route}"
                    )
            tiled_cohorts = int(snapshot_proof["tiled_cohort_count"])
            tiled_jobs = int(snapshot_proof["tiled_job_count"])
            tiled_direct = int(snapshot_proof["tiled_direct_count"])
            tiled_kernels = int(snapshot_proof["tiled_kernel_launch_count"])
            tiled_failures = int(snapshot_proof["tiled_admission_failure_count"])
            if tiled_cohorts <= 0:
                reasons.append(
                    f"tp8_selector_log_s_tiled_cohort_missing:rank={rank}"
                )
            if tiled_jobs <= tiled_cohorts:
                reasons.append(
                    "tp8_selector_log_s_tiled_not_amortized:"
                    f"rank={rank}:cohorts={tiled_cohorts}:jobs={tiled_jobs}"
                )
            if tiled_direct != 0:
                reasons.append(
                    "tp8_selector_log_s_direct_tiled_observed:"
                    f"rank={rank}:count={tiled_direct}"
                )
            if tiled_kernels != tiled_cohorts * 4:
                reasons.append(
                    "tp8_selector_log_s_tiled_kernel_count_mismatch:"
                    f"rank={rank}:cohorts={tiled_cohorts}:kernels={tiled_kernels}"
                )
            if tiled_failures != 0:
                reasons.append(
                    "tp8_selector_log_s_tiled_admission_failure:"
                    f"rank={rank}:count={tiled_failures}"
                )
        artifact_path = Path(str(snapshot_proofs[0]["module_path"]))
        try:
            actual_size = artifact_path.stat().st_size
            actual_sha256 = _file_sha256(artifact_path)
        except OSError:
            reasons.append("tp8_selector_log_s_artifact_missing_or_unreadable")
        else:
            if actual_size != snapshot_proofs[0]["module_size_bytes"]:
                reasons.append("tp8_selector_log_s_artifact_size_changed")
            if actual_sha256 != snapshot_proofs[0]["module_sha256"]:
                reasons.append("tp8_selector_log_s_artifact_sha256_changed")
    return reasons


def _model_kv_contract_reasons(
    summary: dict[str, Any],
    *,
    mode: str,
    expected_model_config_sha256: str | None = None,
) -> list[str]:
    provenance = summary.get("run_provenance")
    if not isinstance(provenance, dict):
        return ["runner_model_kv_provenance_missing"]
    recorded = provenance.get("runner_model_kv_contract")
    if not isinstance(recorded, dict):
        return ["runner_model_kv_contract_missing_or_invalid"]
    reasons: list[str] = []
    tensor_parallel_size = provenance.get("tensor_parallel_size")
    if type(tensor_parallel_size) is not int or tensor_parallel_size <= 0:
        return ["runner_model_kv_tensor_parallel_size_invalid"]
    try:
        derived = derive_model_kv_contract(
            Path(str(provenance.get("model", "") or "")),
            tensor_parallel_size=tensor_parallel_size,
        )
    except (OSError, ValueError) as exc:
        return [f"runner_model_kv_contract_unverifiable:{exc}"]
    exact_fields = {
        "schema": MODEL_KV_CONTRACT_SCHEMA,
        "model_config_path": derived["model_config_path"],
        "model_config_sha256": derived["model_config_sha256"],
        "num_hidden_layers": derived["num_hidden_layers"],
        "num_key_value_heads": derived["num_key_value_heads"],
        "head_dim": derived["head_dim"],
        "dtype": derived["dtype"],
        "dtype_bytes": derived["dtype_bytes"],
        "total_bytes_per_token": derived["total_bytes_per_token"],
        "per_rank_bytes_per_token_requested": derived[
            "per_rank_bytes_per_token"
        ],
        "per_rank_bytes_per_token_effective": derived[
            "per_rank_bytes_per_token"
        ],
    }
    for field, expected in exact_fields.items():
        actual = recorded.get(field)
        if type(actual) is not type(expected) or actual != expected:
            reasons.append(
                "runner_model_kv_contract_mismatch:"
                f"{field}:actual={actual!r}:expected={expected!r}"
            )
    if expected_model_config_sha256 is not None:
        if derived["model_config_sha256"] != expected_model_config_sha256:
            reasons.append(
                "runner_model_config_external_identity_mismatch:"
                f"derived={derived['model_config_sha256']!r}:"
                f"expected={expected_model_config_sha256!r}"
            )
        if (
            provenance.get("runner_expected_model_config_sha256")
            != expected_model_config_sha256
        ):
            reasons.append(
                "runner_expected_model_config_sha256_mismatch_or_missing"
            )
    for field in (
        "runner_kv_token_bytes_per_rank_requested",
        "runner_kv_token_bytes_per_rank_effective",
    ):
        if provenance.get(field) != derived["per_rank_bytes_per_token"]:
            reasons.append(f"{field}_mismatch")
    if recorded.get("override_present") not in {"0", "1"}:
        reasons.append("runner_model_kv_override_state_invalid")

    if mode == "sparse":
        try:
            kv_pool_bytes = int(provenance.get("runner_kv_cache_memory_bytes", 0))
            batch_size = int(provenance.get("batch_size", 0))
            compact_blocks = int(provenance.get("compact_blocks_per_slot", 0))
            context_tokens = int(provenance.get("runner_context_tokens", 0))
            max_new_tokens = int(provenance.get("max_new_tokens_effective", 0))
        except (TypeError, ValueError):
            reasons.append("runner_model_kv_capacity_inputs_invalid")
        else:
            dual_gen = provenance.get("runner_compact_dual_gen")
            if dual_gen not in {"0", "1"}:
                reasons.append("runner_model_kv_compact_dual_gen_invalid")
                return reasons
            generation_count = 2 if dual_gen == "1" else 1
            per_rank = int(derived["per_rank_bytes_per_token"])
            lease_bytes = (
                batch_size * compact_blocks * 16 * per_rank * generation_count
            )
            denominator = per_rank * batch_size
            capacity = (
                (kv_pool_bytes - lease_bytes) // denominator
                if denominator > 0
                else -1
            )
            needed = context_tokens + max_new_tokens
            preflight_status = provenance.get("runner_kv_preflight_status")
            if capacity < needed and preflight_status == "passed":
                reasons.append(
                    "runner_model_kv_capacity_insufficient:"
                    f"capacity={capacity}:needed={needed}"
                )
            elif capacity >= needed and preflight_status == "undersized_override":
                reasons.append("runner_model_kv_undersized_override_not_needed")
    return reasons


def _runner_config_identity_reasons(
    summary: dict[str, Any],
    *,
    mode: str,
    expected_tier: str | None,
    selector_cache_root: Path | None,
    expected_model_config_sha256: str | None = None,
) -> list[str]:
    provenance = summary.get("run_provenance")
    if not isinstance(provenance, dict):
        return ["run_provenance_missing_or_invalid"]
    reasons: list[str] = []
    if expected_tier is not None and provenance.get("runner_tier") != expected_tier:
        reasons.append(
            "runner_tier_identity_mismatch:"
            f"actual={provenance.get('runner_tier')!r}:expected={expected_tier!r}"
        )
    if mode == "sparse" and expected_tier is not None:
        recorded_selector_cache = str(
            provenance.get("runner_selector_cache_root", "") or ""
        )
        if selector_cache_root is None:
            reasons.append("runner_selector_trusted_cache_root_missing")
        elif Path(recorded_selector_cache).resolve(strict=False) != Path(
            selector_cache_root
        ).resolve(strict=False):
            reasons.append(
                "runner_selector_cache_root_identity_mismatch:"
                f"recorded={recorded_selector_cache!r}:"
                f"trusted={str(selector_cache_root)!r}"
            )
    exact_fields = {
        "runner_mode": mode,
        "runner_batch_size": str(provenance.get("batch_size")),
        "runner_tensor_parallel_size": str(provenance.get("tensor_parallel_size")),
        "runner_max_model_len": str(provenance.get("max_model_len")),
        "runner_max_new_tokens": str(provenance.get("max_new_tokens_effective")),
        "runner_compact_blocks_per_slot": str(
            provenance.get("compact_blocks_per_slot")
        ),
        "runner_corpus_path": provenance.get("prompt"),
    }
    for field, expected in exact_fields.items():
        if provenance.get(field) != expected:
            reasons.append(
                f"runner_config_identity_mismatch:{field}:"
                f"actual={provenance.get(field)!r}:expected={expected!r}"
            )
    for field in ("runner_context_tokens", "runner_kv_cache_memory_bytes"):
        raw = str(provenance.get(field, "") or "")
        if not raw.isdigit() or int(raw) <= 0:
            reasons.append(f"runner_config_identity_{field}_invalid")
    corpus_sha = str(provenance.get("runner_corpus_sha256", "") or "")
    prompt_artifact = provenance.get("prompt_artifact")
    prompt_path = Path(str(provenance.get("prompt", "") or ""))
    if (
        re.fullmatch(r"[0-9a-f]{64}", corpus_sha) is None
        or not isinstance(prompt_artifact, dict)
        or prompt_artifact.get("path") != str(prompt_path.resolve(strict=False))
        or prompt_artifact.get("sha256") != corpus_sha
    ):
        reasons.append("runner_corpus_identity_missing_or_mismatched")
    elif not prompt_path.is_file():
        reasons.append("runner_corpus_file_missing")
    else:
        try:
            current_corpus_sha = _file_sha256(prompt_path)
        except OSError:
            reasons.append("runner_corpus_file_unreadable")
        else:
            if current_corpus_sha != corpus_sha:
                reasons.append("runner_corpus_mutated_after_run")
    try:
        tensor_parallel_size = int(provenance.get("tensor_parallel_size", 0))
    except (TypeError, ValueError):
        tensor_parallel_size = 0
    capabilities = str(provenance.get("runner_cuda_capabilities", "") or "")
    capability_items = capabilities.split(",") if capabilities else []
    if tensor_parallel_size <= 0 or len(capability_items) != tensor_parallel_size:
        reasons.append("runner_cuda_capability_rank_count_mismatch")
    custom_ar_disabled = str(provenance.get("runner_custom_ar_disabled", "") or "")
    if custom_ar_disabled not in {"0", "1"}:
        reasons.append("runner_custom_ar_policy_missing_or_invalid")
    if tensor_parallel_size > 1 and custom_ar_disabled == "0":
        for field in (
            "runner_pytorch_alloc_conf",
            "runner_pytorch_cuda_alloc_conf",
        ):
            normalized = re.sub(
                r"\s+", "", str(provenance.get(field, "") or "").lower()
            )
            if re.search(r"(?:^|,)expandable_segments:(?:true|1)(?:,|$)", normalized):
                reasons.append(f"runner_tp_custom_ar_expandable_allocator:{field}")
    custom_ar_fields = (
        "speed_child_custom_all_reduce_requested",
        "speed_child_custom_all_reduce_effective",
        "speed_child_custom_all_reduce_effective_reason",
        "speed_child_custom_all_reduce_runtime_proof_required",
        "speed_child_custom_all_reduce_runtime_proof_passed",
        "speed_child_custom_all_reduce_runtime_configured_effective",
        "speed_child_custom_all_reduce_runtime_tensor_parallel_size",
        "speed_child_custom_all_reduce_runtime_rank_count",
        "speed_child_custom_all_reduce_runtime_active_rank_count",
        "speed_child_custom_all_reduce_runtime_inactive_rank_count",
        "speed_child_custom_all_reduce_runtime_all_ranks_active",
        "speed_child_custom_all_reduce_runtime_rank_consistent",
        "speed_child_custom_all_reduce_runtime_preemptor_flashinfer_enabled",
        "speed_child_custom_all_reduce_runtime_preemptor_nccl_symm_mem_enabled",
        "speed_child_custom_all_reduce_runtime_required_num_tokens",
        "speed_child_custom_all_reduce_runtime_model_hidden_size",
        "speed_child_custom_all_reduce_runtime_model_dtype",
        "speed_child_custom_all_reduce_runtime_model_dtype_bytes",
        "speed_child_custom_all_reduce_runtime_required_payload_bytes",
        "speed_child_custom_all_reduce_runtime_all_ranks_payload_eligible",
        "speed_child_custom_all_reduce_runtime_records",
        "speed_child_custom_all_reduce_runtime_proof_error",
        "speed_child_custom_all_reduce_runtime_gate_passed",
    )
    for field in custom_ar_fields:
        if summary.get(field) != provenance.get(field):
            reasons.append(
                "runner_custom_ar_summary_provenance_mismatch:"
                f"{field}:summary={summary.get(field)!r}:"
                f"provenance={provenance.get(field)!r}"
            )
    scheduling_fields = (
        "speed_child_engine_scheduling_mode_requested",
        "speed_child_engine_async_scheduling_configured",
        "speed_child_engine_async_scheduling_effective",
    )
    for field in scheduling_fields:
        if summary.get(field) != provenance.get(field):
            reasons.append(
                "runner_scheduling_summary_provenance_mismatch:"
                f"{field}:summary={summary.get(field)!r}:"
                f"provenance={provenance.get(field)!r}"
            )
    scheduling_expected = {
        "scheduling_mode_requested": "async",
        "speed_child_engine_scheduling_mode_requested": "async",
        "speed_child_engine_async_scheduling_configured": True,
        "speed_child_engine_async_scheduling_effective": True,
    }
    for field, expected in scheduling_expected.items():
        actual = provenance.get(field)
        if type(actual) is not type(expected) or actual != expected:
            reasons.append(
                "runner_explicit_async_scheduling_proof_mismatch:"
                f"{field}:actual={actual!r}:expected={expected!r}"
            )
    if expected_tier is not None:
        reasons.extend(
            _model_kv_contract_reasons(
                summary,
                mode=mode,
                expected_model_config_sha256=expected_model_config_sha256,
            )
        )
    if expected_tier == "tp8x64k":
        tp8_exact_fields: dict[str, object] = {
            "tensor_parallel_size": 8,
            "batch_size": 32,
            "max_model_len": 66_560,
            "max_num_seqs": 32,
            "max_num_batched_tokens": 8_192,
            "chunked_prefill": "enabled",
            "max_seq_len_to_capture": 66_560,
            "max_new_tokens_effective": 2_048,
            "compact_blocks_per_slot": 112,
            "runner_tensor_parallel_size": "8",
            "runner_batch_size": "32",
            "runner_context_tokens": "64000",
            "runner_kv_cache_memory_bytes": "42949672960",
            "runner_max_model_len": "66560",
            "runner_max_new_tokens": "2048",
            "runner_max_num_seqs": "32",
            "runner_max_num_batched_tokens": "8192",
            "runner_chunked_prefill": "enabled",
            "runner_max_seq_len_to_capture": "66560",
            "runner_refresh_interval": "96",
            "runner_compact_blocks_per_slot": "112",
            "runner_cuda_capabilities": ",".join(("8.0",) * 8),
            "runner_code_scope": RUNNER_CODE_SCOPE,
            "runner_code_scope_untracked_status": "",
            "runner_code_scope_untracked_clean": "1",
        }
        if mode == "sparse":
            tp8_exact_fields["runner_compact_dual_gen"] = "1"
        for field, expected in tp8_exact_fields.items():
            actual = provenance.get(field)
            if type(actual) is not type(expected) or actual != expected:
                reasons.append(
                    "runner_tp8_exact_workload_mismatch:"
                    f"{field}:actual={actual!r}:expected={expected!r}"
                )
        if provenance.get("runner_gpu_physical_capacity_status") != "passed":
            reasons.append("runner_tp8_gpu_physical_capacity_not_proven")
        for field in (
            "runner_gpu_total_memory_bytes",
            "runner_gpu_free_memory_bytes",
        ):
            raw_values = str(provenance.get(field, "") or "").split(",")
            if len(raw_values) != 8 or any(
                not value.isdigit() or int(value) <= 42_949_672_960
                for value in raw_values
            ):
                reasons.append(f"runner_tp8_{field}_insufficient_or_invalid")
        try:
            live_untracked = _live_code_scope_untracked_files()
        except (OSError, subprocess.SubprocessError) as exc:
            reasons.append(f"runner_tp8_code_scope_untracked_check_failed:{exc}")
        else:
            if live_untracked:
                reasons.append(
                    "runner_tp8_code_scope_untracked_not_clean:"
                    f"{live_untracked!r}"
                )
        expected_commit = str(
            provenance.get("runner_expected_git_commit", "") or ""
        )
        if re.fullmatch(r"[0-9a-f]{40}", expected_commit) is None:
            reasons.append("runner_tp8_expected_release_sha_invalid")
        else:
            release_fields = {
                "git_head": expected_commit,
                "runner_git_head": expected_commit,
                "runner_git_tracked_clean": "1",
                "git_tracked_status_short": "",
            }
            for field, expected in release_fields.items():
                if provenance.get(field) != expected:
                    reasons.append(
                        "runner_tp8_release_identity_mismatch:"
                        f"{field}:actual={provenance.get(field)!r}:"
                        f"expected={expected!r}"
                    )
        if provenance.get("speed_child_custom_all_reduce_requested") != "auto":
            reasons.append("runner_tp8_custom_ar_requested_not_auto")
        if provenance.get("speed_child_custom_all_reduce_effective") != "enabled":
            reasons.append("runner_tp8_custom_ar_effective_not_enabled")
        effective_reason = str(
            provenance.get("speed_child_custom_all_reduce_effective_reason", "") or ""
        )
        if effective_reason != "nvlink_probe:active NVLink on all visible GPUs":
            reasons.append("runner_tp8_custom_ar_nvlink_proof_missing")
        runtime_expected = {
            "speed_child_custom_all_reduce_runtime_proof_required": True,
            "speed_child_custom_all_reduce_runtime_proof_passed": True,
            "speed_child_custom_all_reduce_runtime_configured_effective": "enabled",
            "speed_child_custom_all_reduce_runtime_tensor_parallel_size": 8,
            "speed_child_custom_all_reduce_runtime_rank_count": 8,
            "speed_child_custom_all_reduce_runtime_active_rank_count": 8,
            "speed_child_custom_all_reduce_runtime_inactive_rank_count": 0,
            "speed_child_custom_all_reduce_runtime_all_ranks_active": True,
            "speed_child_custom_all_reduce_runtime_rank_consistent": True,
            "speed_child_custom_all_reduce_runtime_preemptor_flashinfer_enabled": False,
            "speed_child_custom_all_reduce_runtime_preemptor_nccl_symm_mem_enabled": False,
            "speed_child_custom_all_reduce_runtime_required_num_tokens": 32,
            "speed_child_custom_all_reduce_runtime_model_hidden_size": 2560,
            "speed_child_custom_all_reduce_runtime_model_dtype": "torch.bfloat16",
            "speed_child_custom_all_reduce_runtime_model_dtype_bytes": 2,
            "speed_child_custom_all_reduce_runtime_required_payload_bytes": 163840,
            "speed_child_custom_all_reduce_runtime_all_ranks_payload_eligible": True,
            "speed_child_custom_all_reduce_runtime_proof_error": "",
            "speed_child_custom_all_reduce_runtime_gate_passed": True,
        }
        for field, expected in runtime_expected.items():
            actual = provenance.get(field)
            if type(actual) is not type(expected) or actual != expected:
                reasons.append(
                    "runner_tp8_custom_ar_runtime_proof_mismatch:"
                    f"{field}:actual={actual!r}:expected={expected!r}"
                )
        model_hidden_size = provenance.get(
            "speed_child_custom_all_reduce_runtime_model_hidden_size"
        )
        model_dtype = provenance.get(
            "speed_child_custom_all_reduce_runtime_model_dtype"
        )
        model_dtype_bytes = provenance.get(
            "speed_child_custom_all_reduce_runtime_model_dtype_bytes"
        )
        required_payload_bytes = provenance.get(
            "speed_child_custom_all_reduce_runtime_required_payload_bytes"
        )
        if type(model_hidden_size) is not int or model_hidden_size <= 0:
            reasons.append("runner_tp8_custom_ar_model_hidden_size_invalid")
        if not isinstance(model_dtype, str) or not model_dtype:
            reasons.append("runner_tp8_custom_ar_model_dtype_invalid")
        if type(model_dtype_bytes) is not int or model_dtype_bytes <= 0:
            reasons.append("runner_tp8_custom_ar_model_dtype_bytes_invalid")
        expected_payload_bytes = (
            32 * model_hidden_size * model_dtype_bytes
            if type(model_hidden_size) is int
            and model_hidden_size > 0
            and type(model_dtype_bytes) is int
            and model_dtype_bytes > 0
            else -1
        )
        if (
            type(required_payload_bytes) is not int
            or required_payload_bytes <= 0
            or required_payload_bytes != expected_payload_bytes
        ):
            reasons.append(
                "runner_tp8_custom_ar_required_payload_invalid:"
                f"actual={required_payload_bytes!r}:"
                f"expected={expected_payload_bytes}"
            )
        records = provenance.get("speed_child_custom_all_reduce_runtime_records")
        if not isinstance(records, list) or len(records) != 8:
            reasons.append("runner_tp8_custom_ar_runtime_records_shape_mismatch")
        else:
            for expected_rank, record in enumerate(records):
                if not isinstance(record, dict):
                    reasons.append(
                        "runner_tp8_custom_ar_runtime_record_invalid:"
                        f"rank={expected_rank}:record_type={type(record).__name__}"
                    )
                    continue
                exact_fields = {
                    "tp_rank": expected_rank,
                    "tp_world_size": 8,
                    "parallel_config_disable_custom_all_reduce": False,
                    "device_communicator_use_custom_allreduce": True,
                    "device_communicator_use_flashinfer_allreduce": False,
                    "fi_ar_comm_present": False,
                    "fi_ar_comm_active": False,
                    "device_communicator_use_torch_symm_mem": False,
                    "symm_mem_comm_present": False,
                    "symm_mem_comm_active": False,
                    "vllm_allreduce_use_flashinfer": False,
                    "vllm_allreduce_use_symm_mem": False,
                    "vllm_use_nccl_symm_mem": False,
                    "ca_comm_present": True,
                    "ca_comm_disabled": False,
                    "ca_comm_active": True,
                    "required_num_tokens": 32,
                    "model_hidden_size": model_hidden_size,
                    "model_dtype": model_dtype,
                    "model_dtype_bytes": model_dtype_bytes,
                    "required_payload_numel": (
                        32 * model_hidden_size
                        if type(model_hidden_size) is int
                        and model_hidden_size > 0
                        else -1
                    ),
                    "required_payload_bytes": required_payload_bytes,
                    "required_payload_spec_error": "",
                    "ca_comm_capacity_covers_required_payload": True,
                    "ca_comm_dispatch_size_eligible": True,
                    "ca_comm_should_custom_ar_callable": True,
                    "ca_comm_synthetic_should_custom_ar": True,
                    "ca_comm_synthetic_should_custom_ar_error": "",
                    "reason": "active",
                }
                for field, expected in exact_fields.items():
                    actual = record.get(field)
                    if type(actual) is not type(expected) or actual != expected:
                        reasons.append(
                            "runner_tp8_custom_ar_runtime_record_mismatch:"
                            f"rank={expected_rank}:{field}:"
                            f"actual={actual!r}:expected={expected!r}"
                        )
                for field in (
                    "device",
                    "worker_class",
                    "device_communicator_class",
                    "ca_comm_class",
                ):
                    if not isinstance(record.get(field), str) or not record.get(field):
                        reasons.append(
                            "runner_tp8_custom_ar_runtime_record_identity_missing:"
                            f"rank={expected_rank}:{field}"
                        )
                if record.get("ca_comm_fully_connected") is not True:
                    reasons.append(
                        "runner_tp8_custom_ar_runtime_record_not_fully_connected:"
                        f"rank={expected_rank}"
                    )
                max_size = record.get("ca_comm_max_size")
                if (
                    type(max_size) is not int
                    or max_size <= 0
                    or type(required_payload_bytes) is not int
                    or max_size <= required_payload_bytes
                ):
                    reasons.append(
                        "runner_tp8_custom_ar_runtime_record_max_size_invalid:"
                        f"rank={expected_rank}:value={max_size!r}:"
                        f"required_payload_bytes={required_payload_bytes!r}"
                    )
    return reasons


def _scheduler_graph_contract_reasons(
    summary: dict[str, Any],
    *,
    mode: str,
    exact_tier: bool,
) -> list[str]:
    """Verify the instantiated scheduler/graph identity of every paired arm."""
    reasons: list[str] = []
    expected = summary.get("scheduler_graph_contract_expected")
    provenance = summary.get("run_provenance")
    if not isinstance(expected, dict):
        return ["scheduler_graph_contract_expected_missing_or_invalid"]
    if not isinstance(provenance, dict):
        return ["scheduler_graph_contract_provenance_missing_or_invalid"]
    if provenance.get("scheduler_graph_contract_expected") != expected:
        reasons.append("scheduler_graph_contract_summary_provenance_mismatch")
    if provenance.get("scheduler_graph_contract_requested") != expected:
        reasons.append("scheduler_graph_contract_requested_provenance_mismatch")

    expected_shape: dict[str, object] = {
        "schema": SCHEDULER_GRAPH_CONTRACT_SCHEMA,
        "full_cuda_graph": True,
    }
    if exact_tier:
        expected_shape.update(
            {
                "max_num_seqs": 32,
                "max_num_batched_tokens": 8_192,
                "kv_cache_memory_bytes": 42_949_672_960,
                "chunked_prefill": "enabled",
                "cudagraph_capture_sizes": [32],
                "max_seq_len_to_capture": 66_560,
            }
        )
    for field, expected_value in expected_shape.items():
        actual = expected.get(field)
        if type(actual) is not type(expected_value) or actual != expected_value:
            reasons.append(
                "scheduler_graph_expected_mismatch:"
                f"{field}:actual={actual!r}:expected={expected_value!r}"
            )
    for field in (
        "max_num_seqs",
        "max_num_batched_tokens",
        "kv_cache_memory_bytes",
        "max_seq_len_to_capture",
    ):
        value = expected.get(field)
        if type(value) is not int or value <= 0:
            reasons.append(f"scheduler_graph_expected_{field}_invalid")
    chunked_prefill = expected.get("chunked_prefill")
    if chunked_prefill not in {"enabled", "disabled"}:
        reasons.append("scheduler_graph_expected_chunked_prefill_not_explicit")
    capture_sizes = expected.get("cudagraph_capture_sizes")
    if (
        not isinstance(capture_sizes, list)
        or not capture_sizes
        or any(type(value) is not int or value <= 0 for value in capture_sizes)
    ):
        reasons.append("scheduler_graph_expected_capture_sizes_invalid")

    child_names = ["speed_child"]
    verdict_only = mode == "sparse" and summary.get("verdict_only") is True
    if not verdict_only:
        child_names.append("diagnostic_child")
    if exact_tier and mode == "sparse":
        child_names.append("dense_reference_child")

    child_contracts: list[dict[str, Any]] = []
    for child in child_names:
        field_name = f"{child}_scheduler_graph_contract"
        contract = summary.get(field_name)
        if not isinstance(contract, dict):
            reasons.append(f"{field_name}_missing_or_invalid")
            continue
        child_contracts.append(contract)
        exact_runtime_fields: dict[str, object] = {
            "scheduler_graph_contract_schema": expected.get("schema"),
            "engine_max_num_seqs_requested": expected.get("max_num_seqs"),
            "engine_max_num_seqs_effective": expected.get("max_num_seqs"),
            "engine_max_num_batched_tokens_requested": expected.get(
                "max_num_batched_tokens"
            ),
            "engine_max_num_batched_tokens_effective": expected.get(
                "max_num_batched_tokens"
            ),
            "engine_kv_cache_memory_bytes_requested": expected.get(
                "kv_cache_memory_bytes"
            ),
            "engine_kv_cache_memory_bytes_effective": expected.get(
                "kv_cache_memory_bytes"
            ),
            "engine_chunked_prefill_requested": chunked_prefill,
            "engine_chunked_prefill_configured": (
                chunked_prefill == "enabled"
            ),
            "engine_chunked_prefill_effective": (
                chunked_prefill == "enabled"
            ),
            "engine_cudagraph_capture_sizes_requested": capture_sizes,
            "engine_cudagraph_capture_sizes_effective": capture_sizes,
            "engine_max_cudagraph_capture_size_effective": (
                max(capture_sizes)
                if isinstance(capture_sizes, list) and capture_sizes
                else None
            ),
            "engine_decode_batch_cudagraph_covered": True,
            "engine_max_seq_len_to_capture_requested": expected.get(
                "max_seq_len_to_capture"
            ),
            "engine_full_cuda_graph_requested": True,
            "engine_full_cuda_graph_effective": True,
        }
        for runtime_field, expected_value in exact_runtime_fields.items():
            actual = contract.get(runtime_field)
            if type(actual) is not type(expected_value) or actual != expected_value:
                reasons.append(
                    f"{child}_scheduler_graph_runtime_mismatch:"
                    f"{runtime_field}:actual={actual!r}:"
                    f"expected={expected_value!r}"
                )
        supported = contract.get("engine_max_seq_len_to_capture_supported")
        effective = contract.get("engine_max_seq_len_to_capture_effective")
        sequence_control = contract.get("engine_sequence_length_graph_control")
        if type(supported) is not bool:
            reasons.append(
                f"{child}_scheduler_graph_max_seq_support_not_instantiated"
            )
        elif supported:
            expected_max_seq = expected.get("max_seq_len_to_capture")
            if type(effective) is not int or effective != expected_max_seq:
                reasons.append(
                    f"{child}_scheduler_graph_max_seq_effective_mismatch:"
                    f"actual={effective!r}:expected={expected_max_seq!r}"
                )
        elif effective is not None:
            reasons.append(
                f"{child}_scheduler_graph_unsupported_max_seq_has_value"
            )
        expected_sequence_control = (
            "legacy_engine_arg" if supported is True else "not_applicable_v1"
        )
        if sequence_control != expected_sequence_control:
            reasons.append(
                f"{child}_scheduler_graph_sequence_control_mismatch:"
                f"actual={sequence_control!r}:expected={expected_sequence_control!r}"
            )
        missing_fields = [
            field for field in SCHEDULER_GRAPH_RUNTIME_FIELDS if field not in contract
        ]
        if missing_fields:
            reasons.append(
                f"{child}_scheduler_graph_runtime_fields_missing:{missing_fields!r}"
            )

    if child_contracts and any(
        contract != child_contracts[0] for contract in child_contracts[1:]
    ):
        reasons.append("scheduler_graph_child_contracts_diverged")
    if provenance.get("speed_child_scheduler_graph_contract") != summary.get(
        "speed_child_scheduler_graph_contract"
    ):
        reasons.append("scheduler_graph_speed_summary_provenance_mismatch")
    if summary.get("scheduler_graph_contract_match") is not True:
        reasons.append("scheduler_graph_contract_match_not_green")
    if summary.get("scheduler_graph_contract_reasons") != []:
        reasons.append("scheduler_graph_contract_reasons_nonempty")
    return reasons


def _exact_runtime_proof_reasons(
    proof: object,
    *,
    child: str,
    arm_mode: str,
    expected_kv_bytes_per_token: int,
    expected_layer_count: int,
) -> list[str]:
    """Validate resolved FULL-graph and physical KV state on every TP rank."""
    prefix = f"tp8_exact_{child}"
    if not isinstance(proof, dict):
        return [f"{prefix}_engine_runtime_proof_missing_or_invalid"]
    reasons: list[str] = []
    expected_top = {
        "engine_runtime_contract_proof_required": True,
        "engine_runtime_contract_proof_passed": True,
        "engine_runtime_contract_tensor_parallel_size": 8,
        "engine_runtime_contract_rank_count": 8,
        "engine_runtime_contract_rank_consistent": True,
        "engine_runtime_graph_mode": "FULL",
        "engine_runtime_graph_capture_sizes": [32],
        "engine_runtime_graph_full_decode_key_present": True,
        "engine_runtime_kv_block_size": 16,
        "engine_runtime_kv_actual_bytes_per_token": (
            expected_kv_bytes_per_token
        ),
        "engine_runtime_kv_page_size_identity_passed": True,
        "engine_runtime_kv_null_block_count": 1,
        "engine_runtime_kv_required_blocks_per_request": 4_128,
        "engine_runtime_kv_required_workload_blocks": 132_096,
        "engine_runtime_kv_capacity_covers_required_total": True,
    }
    for field, expected in expected_top.items():
        if proof.get(field) != expected:
            reasons.append(
                f"{prefix}_runtime_field_mismatch:{field}:"
                f"actual={proof.get(field)!r}:expected={expected!r}"
            )

    physical = proof.get("engine_runtime_physical_geometry")
    admission = proof.get("engine_runtime_arm_capacity_admission")
    if not isinstance(physical, dict):
        reasons.append(f"{prefix}_physical_geometry_missing_or_invalid")
        physical_records: list[object] = []
    else:
        if physical.get("tensor_parallel_size") != 8:
            reasons.append(f"{prefix}_physical_geometry_tp_mismatch")
        raw = physical.get("rank_records")
        physical_records = raw if isinstance(raw, list) else []
    if not isinstance(admission, dict):
        reasons.append(f"{prefix}_capacity_admission_missing_or_invalid")
        admission_records: list[object] = []
    else:
        if admission.get("tensor_parallel_size") != 8:
            reasons.append(f"{prefix}_capacity_admission_tp_mismatch")
        raw = admission.get("rank_records")
        admission_records = raw if isinstance(raw, list) else []
    if len(physical_records) != 8:
        reasons.append(f"{prefix}_physical_rank_count={len(physical_records)}")
    if len(admission_records) != 8:
        reasons.append(f"{prefix}_admission_rank_count={len(admission_records)}")

    expected_full_key = {
        "num_tokens": 32,
        "num_reqs": 32,
        "uniform": True,
        "has_lora": False,
        "num_active_loras": 0,
    }
    compact_blocks_per_slot = 112 if arm_mode == "sparse" else 0
    compact_generation_count = 2 if arm_mode == "sparse" else 0
    compact_lease_blocks = 32 * compact_blocks_per_slot * compact_generation_count
    if proof.get("engine_core_block_pool_proof_required") is not True:
        reasons.append(f"{prefix}_core_block_pool_proof_not_required")
    if proof.get("engine_core_block_pool_proof_passed") is not True:
        reasons.append(f"{prefix}_core_block_pool_proof_not_green")
    if proof.get("engine_core_block_pool_expected_reserved_count") != (
        compact_lease_blocks
    ):
        reasons.append(f"{prefix}_core_block_pool_expected_count_mismatch")
    core_state = proof.get("engine_core_block_pool_state")
    if not isinstance(core_state, dict):
        reasons.append(f"{prefix}_core_block_pool_state_invalid")
    else:
        expected_num_blocks = proof.get("engine_runtime_kv_num_blocks")
        expected_free_blocks = (
            expected_num_blocks - 1 - compact_lease_blocks
            if type(expected_num_blocks) is int
            else -1
        )
        expected_reserved_ids = (
            list(
                range(
                    expected_num_blocks - compact_lease_blocks,
                    expected_num_blocks,
                )
            )
            if type(expected_num_blocks) is int
            else []
        )
        core_expected = {
            "schema": "sfi.engine_core_block_pool_state.v1",
            "num_gpu_blocks": expected_num_blocks,
            "num_free_blocks": expected_free_blocks,
            "free_queue_reported_count": expected_free_blocks,
            "null_block_id": 0,
            "null_block_is_null": True,
            "null_block_in_free_queue": False,
            "reserved_block_ids": expected_reserved_ids,
            "reserved_block_count": compact_lease_blocks,
            "reserved_ids_tail_contiguous": True,
            "reserved_ids_in_free_queue": [],
        }
        for field, expected in core_expected.items():
            if core_state.get(field) != expected:
                reasons.append(f"{prefix}_core_block_pool_mismatch:{field}")
        reserved_states = core_state.get("reserved_blocks")
        if not isinstance(reserved_states, list) or len(reserved_states) != (
            compact_lease_blocks
        ):
            reasons.append(f"{prefix}_core_reserved_states_shape_mismatch")
        elif any(
            not isinstance(record, dict)
            or record.get("block_id") != expected_reserved_ids[index]
            or record.get("ref_cnt") != 1
            or record.get("is_null") is not False
            or record.get("in_free_queue") is not False
            for index, record in enumerate(reserved_states)
        ):
            reasons.append(f"{prefix}_core_reserved_state_mismatch")
        if arm_mode == "sparse":
            manager_lease = core_state.get("manager_lease")
            pool_lease = core_state.get("pool_lease")
            if manager_lease != pool_lease or not isinstance(manager_lease, dict):
                reasons.append(f"{prefix}_core_lease_identity_mismatch")
            elif manager_lease.get("reserved_manager_block_ids") != (
                expected_reserved_ids
            ):
                reasons.append(f"{prefix}_core_lease_reserved_ids_mismatch")
            if core_state.get("manager_pool_lease_same_object") is not True:
                reasons.append(f"{prefix}_core_lease_object_identity_mismatch")
        elif (
            core_state.get("manager_lease") is not None
            or core_state.get("pool_lease") is not None
        ):
            reasons.append(f"{prefix}_dense_core_lease_present")
        if isinstance(admission, dict) and admission.get(
            "engine_core_block_pool"
        ) != core_state:
            reasons.append(f"{prefix}_core_state_not_bound_to_admission")
    for rank in range(min(len(physical_records), len(admission_records), 8)):
        physical_record = physical_records[rank]
        admission_record = admission_records[rank]
        if not isinstance(physical_record, dict):
            reasons.append(f"{prefix}_physical_rank{rank}_invalid")
            continue
        if not isinstance(admission_record, dict):
            reasons.append(f"{prefix}_admission_rank{rank}_invalid")
            continue
        for record_name, record in (
            ("physical", physical_record),
            ("admission", admission_record),
        ):
            if record.get("tp_rank") != rank:
                reasons.append(f"{prefix}_{record_name}_rank_order_mismatch:{rank}")
        graph_expected = {
            "tp_world_size": 8,
            "compilation_cudagraph_mode": "FULL",
            "compilation_cudagraph_capture_sizes": [32],
            "compilation_max_cudagraph_capture_size": 32,
            "dispatcher_cudagraph_mode": "FULL",
            "dispatcher_keys_initialized": True,
        }
        for field, expected in graph_expected.items():
            if physical_record.get(field) != expected:
                reasons.append(
                    f"{prefix}_rank{rank}_graph_mismatch:{field}"
                )
        device = physical_record.get("cuda_device_runtime")
        if not isinstance(device, dict):
            reasons.append(f"{prefix}_rank{rank}_cuda_runtime_invalid")
        else:
            if device.get("current_device") != rank:
                reasons.append(f"{prefix}_rank{rank}_cuda_device_mismatch")
            if device.get("capability") != "8.0":
                reasons.append(f"{prefix}_rank{rank}_cuda_capability_mismatch")
            if not isinstance(device.get("name"), str) or not device.get("name"):
                reasons.append(f"{prefix}_rank{rank}_cuda_name_missing")
            if type(device.get("total_memory_bytes")) is not int or device.get(
                "total_memory_bytes"
            ) <= 42_949_672_960:
                reasons.append(f"{prefix}_rank{rank}_cuda_memory_invalid")
            if not isinstance(device.get("uuid"), str):
                reasons.append(f"{prefix}_rank{rank}_cuda_uuid_invalid")
        graph_keys = physical_record.get("dispatcher_graph_keys")
        if not isinstance(graph_keys, dict):
            reasons.append(f"{prefix}_rank{rank}_graph_keys_invalid")
        else:
            full_keys = graph_keys.get("FULL")
            if not isinstance(full_keys, list) or expected_full_key not in full_keys:
                reasons.append(f"{prefix}_rank{rank}_full_key_missing")
            if graph_keys.get("PIECEWISE", []) not in ([], None):
                reasons.append(f"{prefix}_rank{rank}_piecewise_key_present")

        kv = physical_record.get("kv_cache")
        if not isinstance(kv, dict):
            reasons.append(f"{prefix}_rank{rank}_kv_geometry_invalid")
            continue
        num_blocks = kv.get("num_blocks")
        composite_page = kv.get("composite_page_size_bytes_total")
        leaf_page = kv.get("leaf_page_size_bytes_total")
        allocated_bytes = kv.get("allocated_bytes")
        if type(num_blocks) is not int or num_blocks <= 0:
            reasons.append(f"{prefix}_rank{rank}_kv_num_blocks_invalid")
        if (
            type(composite_page) is not int
            or composite_page <= 0
            or composite_page != leaf_page
            or kv.get("composite_page_sizes_match_leaf_sums") is not True
        ):
            reasons.append(f"{prefix}_rank{rank}_kv_composite_leaf_identity")
        if (
            type(num_blocks) is int
            and type(composite_page) is int
            and allocated_bytes != num_blocks * composite_page
        ):
            reasons.append(f"{prefix}_rank{rank}_kv_allocation_identity")
        if kv.get("actual_bytes_per_token") != expected_kv_bytes_per_token:
            reasons.append(f"{prefix}_rank{rank}_kv_bytes_per_token_mismatch")
        if kv.get("uniform_block_size") != 16:
            reasons.append(f"{prefix}_rank{rank}_kv_block_size_mismatch")
        if type(kv.get("runner_tensor_count")) is not int or kv.get(
            "runner_tensor_count"
        ) <= 0:
            reasons.append(f"{prefix}_rank{rank}_runner_kv_tensor_missing")
        groups = kv.get("groups")
        if not isinstance(groups, list) or len(groups) != 1:
            reasons.append(f"{prefix}_rank{rank}_kv_group_shape_mismatch")
        else:
            group = groups[0]
            if not isinstance(group, dict):
                reasons.append(f"{prefix}_rank{rank}_kv_group_invalid")
            else:
                signatures = group.get("leaf_spec_signatures")
                leaf_count = (
                    sum(
                        int(item.get("count", 0))
                        for item in signatures
                        if isinstance(item, dict)
                    )
                    if isinstance(signatures, list)
                    else -1
                )
                if group.get("layer_count") != expected_layer_count:
                    reasons.append(f"{prefix}_rank{rank}_layer_count_mismatch")
                if leaf_count != expected_layer_count:
                    reasons.append(f"{prefix}_rank{rank}_leaf_count_mismatch")

        admission_expected = {
            "null_block_count": 1,
            "compact_blocks_per_slot": compact_blocks_per_slot,
            "compact_generation_count": compact_generation_count,
            "compact_lease_blocks": compact_lease_blocks,
            "required_batch_size": 32,
            "required_tokens_per_request": 66_048,
            "required_workload_tokens": 2_113_536,
            "required_blocks_per_request": 4_128,
            "required_workload_blocks": 132_096,
            "expected_bytes_per_token": expected_kv_bytes_per_token,
            "capacity_covers_required_total": True,
        }
        for field, expected in admission_expected.items():
            if admission_record.get(field) != expected:
                reasons.append(
                    f"{prefix}_rank{rank}_capacity_mismatch:{field}"
                )
        if type(num_blocks) is int:
            expected_ordinary = num_blocks - 1 - compact_lease_blocks
            expected_total = 1 + compact_lease_blocks + 132_096
            if admission_record.get("ordinary_blocks") != expected_ordinary:
                reasons.append(f"{prefix}_rank{rank}_ordinary_blocks_mismatch")
            if admission_record.get("required_total_blocks") != expected_total:
                reasons.append(f"{prefix}_rank{rank}_required_blocks_mismatch")
            if admission_record.get("schedulable_tokens") != expected_ordinary * 16:
                reasons.append(f"{prefix}_rank{rank}_schedulable_tokens_mismatch")
            if admission_record.get("required_total_tokens") != expected_total * 16:
                reasons.append(f"{prefix}_rank{rank}_required_tokens_mismatch")
    device_uuids = [
        str(device.get("uuid", "") or "")
        for record in physical_records
        if isinstance(record, dict)
        for device in [record.get("cuda_device_runtime")]
        if isinstance(device, dict)
    ]
    nonempty_uuids = [value for value in device_uuids if value]
    if nonempty_uuids and (
        len(nonempty_uuids) != 8 or len(set(nonempty_uuids)) != 8
    ):
        reasons.append(f"{prefix}_cuda_uuid_order_identity_invalid")
    return reasons


_PAIR_BACKEND_CONTRACT = {
    "sparse": "FLASH_ATTN",
    "dense": "FLASH_ATTN_VLLM_V1",
}
_PAIR_RUNNER_CONTRACT = {
    "sparse": "run_sparse_only.py",
    "dense": "run_dense_only.py",
}


def _pair_summary_integrity_reasons(
    summary: dict[str, Any],
    *,
    prefix: str,
) -> list[str]:
    """Validate pair evidence shared by local comparison and exact TP8."""
    reasons: list[str] = []
    expected_fields = {
        "sparse_dense_pair_required": True,
        "sparse_dense_pair_execution_order": (
            "sparse_speed,dense_reference,sparse_diagnostic"
        ),
        "sparse_dense_pair_scope": (
            "same_parent_same_gpu_lock_adjacent_observer_free_engine_loop"
        ),
        "sparse_dense_pair_total_wall_scope": (
            "measurement_engine_loop_prefill_plus_decode_excludes_engine_init"
        ),
        "sparse_dense_pair_observer_free": True,
        "sparse_dense_pair_child_identity_scope": "arm_invariant_projection",
        "sparse_dense_pair_arm_backend_contract": _PAIR_BACKEND_CONTRACT,
        "sparse_dense_pair_arm_backend_contract_passed": True,
        "sparse_dense_pair_arm_runner_contract": _PAIR_RUNNER_CONTRACT,
        "sparse_dense_pair_arm_runner_contract_passed": True,
        "sparse_dense_pair_arm_runner_observed": {
            arm: {
                "top_level": runner,
                "run_config": runner,
            }
            for arm, runner in _PAIR_RUNNER_CONTRACT.items()
        },
        "sparse_dense_pair_geometry_match": True,
    }
    for field, expected in expected_fields.items():
        if summary.get(field) != expected:
            reasons.append(f"{prefix}_field_mismatch:{field}")

    geometries: dict[str, dict[str, Any]] = {}
    recorded_digests: dict[str, str] = {}
    for arm in ("sparse", "dense"):
        geometry = summary.get(f"sparse_dense_pair_{arm}_geometry")
        if not isinstance(geometry, dict) or not geometry:
            reasons.append(f"{prefix}_{arm}_geometry_invalid")
            continue
        geometries[arm] = geometry
        digest_field = f"sparse_dense_pair_{arm}_geometry_digest"
        recorded_digest = str(summary.get(digest_field, "") or "")
        if re.fullmatch(r"[0-9a-f]{64}", recorded_digest) is None:
            reasons.append(f"{prefix}_{arm}_geometry_digest_invalid")
            continue
        recorded_digests[arm] = recorded_digest
        if recorded_digest != build_config_digest(geometry):
            reasons.append(f"{prefix}_{arm}_geometry_digest_not_raw")

    if geometries.get("sparse") != geometries.get("dense"):
        reasons.append(f"{prefix}_actual_geometry_mismatch")
    if recorded_digests.get("sparse") != recorded_digests.get("dense"):
        reasons.append(f"{prefix}_geometry_digest_mismatch")

    for arm, expected_backend in _PAIR_BACKEND_CONTRACT.items():
        identity = summary.get(f"sparse_dense_pair_{arm}_child_identity")
        if not isinstance(identity, dict):
            reasons.append(f"{prefix}_{arm}_child_identity_invalid")
        elif identity.get("attention_backend") != expected_backend:
            reasons.append(f"{prefix}_{arm}_attention_backend_mismatch")
    return reasons


def _explicit_local_pair_reasons(summary: dict[str, Any]) -> list[str]:
    """Validate a local timed pair without upgrading it to a speedup verdict."""
    reasons = _pair_summary_integrity_reasons(summary, prefix="local_pair")
    expected_fields = {
        "sparse_dense_pair_contract_kind": "explicit_local_comparison",
        "sparse_dense_pair_dense_metrics_readable": True,
        "sparse_dense_pair_comparison_gate_passed": True,
        "sparse_dense_pair_comparison_gate_reasons": [],
        "sparse_dense_pair_speedup_gate_passed": None,
        "sparse_dense_pair_speedup_gate_reasons": [],
        "sparse_dense_pair_claim": "paired_engine_loop_comparison",
    }
    for field, expected in expected_fields.items():
        if summary.get(field) != expected:
            reasons.append(f"local_pair_field_mismatch:{field}")
    for field in (
        "total_wall_speedup",
        "total_token_speedup",
        "decode_speedup",
        "all_decode_speedup",
    ):
        if not _is_finite_positive_number(summary.get(field)):
            reasons.append(f"local_pair_ratio_invalid:{field}")
    expected_speedup_observed = all(
        _is_finite_positive_number(summary.get(field))
        and float(summary[field]) > 1.0
        for field in (
            "total_wall_speedup",
            "total_token_speedup",
            "decode_speedup",
            "all_decode_speedup",
        )
    )
    if summary.get("sparse_dense_pair_speedup_observed") is not (
        expected_speedup_observed
    ):
        reasons.append("local_pair_speedup_observed_mismatch")
    return reasons


def _is_documented_local_pair_comparison_only(
    summary: dict[str, Any],
    *,
    mode: str,
    exact_tier: bool,
    expect_local_paired_comparison: bool,
    semantic_gate_reasons: list[object],
    producer_gate_reasons: list[object],
) -> bool:
    """Recognize the sole non-production exit allowed for a valid local pair."""
    return bool(
        expect_local_paired_comparison
        and mode == "sparse"
        and not exact_tier
        and semantic_gate_reasons
        == ["semantic_output_health_not_ok:semantic_mismatch"]
        and producer_gate_reasons == []
        and summary.get("reference_gate_passed") is False
        and summary.get("reference_gate_reasons")
        == ["reference_semantic_mismatch"]
        and summary.get("reference_returncode") == 0
        and summary.get("reference_timed_out") is False
        and summary.get("skip_dense_reference") is not True
        and summary.get("gate_passed") is False
        and summary.get("production_gate_passed") is False
        and not _explicit_local_pair_reasons(summary)
    )


def _exact_engine_runtime_and_pair_reasons(
    summary: dict[str, Any],
    *,
    mode: str,
) -> list[str]:
    provenance = summary.get("run_provenance")
    if not isinstance(provenance, dict):
        return ["tp8_exact_runtime_provenance_missing"]
    model_kv = provenance.get("runner_model_kv_contract")
    if not isinstance(model_kv, dict):
        return ["tp8_exact_runtime_model_kv_missing"]
    expected_kv_bytes = model_kv.get("per_rank_bytes_per_token_effective")
    expected_layers = model_kv.get("num_hidden_layers")
    if type(expected_kv_bytes) is not int or expected_kv_bytes <= 0:
        return ["tp8_exact_runtime_kv_bytes_invalid"]
    if type(expected_layers) is not int or expected_layers <= 0:
        return ["tp8_exact_runtime_layer_count_invalid"]

    reasons: list[str] = []
    child_modes = {
        "speed_child": mode,
        "diagnostic_child": mode,
    }
    if mode == "sparse":
        child_modes["dense_reference_child"] = "dense"
    for child, arm_mode in child_modes.items():
        reasons.extend(
            _exact_runtime_proof_reasons(
                summary.get(f"{child}_engine_runtime_contract_proof"),
                child=child,
                arm_mode=arm_mode,
                expected_kv_bytes_per_token=expected_kv_bytes,
                expected_layer_count=expected_layers,
            )
        )
    if summary.get("engine_runtime_contract_match") is not True:
        reasons.append("tp8_exact_engine_runtime_contract_not_matched")
    if summary.get("engine_runtime_contract_reasons") != []:
        reasons.append("tp8_exact_engine_runtime_contract_reasons_nonempty")

    observer = summary.get("diagnostic_child_cudagraph_runtime_proof")
    if not isinstance(observer, dict):
        reasons.append("tp8_exact_diagnostic_graph_observer_missing")
    else:
        all_steps = observer.get("cudagraph_runtime_observer_all_decode_step_count")
        expected_observer = {
            "cudagraph_runtime_observer_scope": (
                "diagnostic_measurement_all_decode"
            ),
            "cudagraph_runtime_observer_enabled": True,
            "cudagraph_runtime_observer_missing_step_count": 0,
            "cudagraph_runtime_observer_all_decode_exact_full": True,
        }
        for field, expected in expected_observer.items():
            if observer.get(field) != expected:
                reasons.append(f"tp8_exact_diagnostic_graph_mismatch:{field}")
        if type(all_steps) is not int or all_steps <= 0:
            reasons.append("tp8_exact_diagnostic_graph_step_count_invalid")
        if observer.get("cudagraph_runtime_observer_exact_full_step_count") != all_steps:
            reasons.append("tp8_exact_diagnostic_graph_exact_step_count_mismatch")
        distribution = observer.get("cudagraph_runtime_observer_distribution")
        expected_distribution = {
            "num_unpadded_tokens": 32,
            "num_padded_tokens": 32,
            "num_paddings": 0,
            "runtime_mode": "FULL",
            "count": all_steps,
        }
        if distribution != [expected_distribution]:
            reasons.append("tp8_exact_diagnostic_graph_distribution_mismatch")

    if mode != "sparse":
        return reasons

    reasons.extend(
        _pair_summary_integrity_reasons(
            summary,
            prefix="tp8_exact_pair",
        )
    )
    exact_pair_fields = {
        "sparse_dense_pair_contract_kind": "exact_speedup_verdict",
        "sparse_dense_pair_dense_metrics_readable": True,
        "sparse_dense_pair_comparison_gate_passed": None,
        "sparse_dense_pair_comparison_gate_reasons": [],
        "sparse_dense_pair_speedup_observed": True,
        "sparse_dense_pair_speedup_gate_passed": True,
        "sparse_dense_pair_speedup_gate_reasons": [],
        "sparse_dense_pair_claim": (
            "sparse_engine_loop_e2e_and_decode_speedup"
        ),
    }
    for field, expected in exact_pair_fields.items():
        if summary.get(field) != expected:
            reasons.append(f"tp8_exact_pair_field_mismatch:{field}")
    for field in (
        "total_wall_speedup",
        "total_token_speedup",
        "decode_speedup",
        "all_decode_speedup",
    ):
        value = summary.get(field)
        if not _is_finite_positive_number(value) or float(value) <= 1.0:
            reasons.append(f"tp8_exact_pair_ratio_not_above_one:{field}")
    if summary.get("reference_returncode") != 0:
        reasons.append("tp8_exact_pair_dense_returncode_nonzero")
    if summary.get("reference_timed_out") is not False:
        reasons.append("tp8_exact_pair_dense_timed_out")

    sparse_metrics_path = str(summary.get("decode_metrics_path", "") or "")
    dense_metrics_path = str(summary.get("reference_dense_metrics_path", "") or "")
    try:
        sparse_metrics = _load_regular_json_object(Path(sparse_metrics_path))
    except (OSError, ValueError) as exc:
        reasons.append(f"tp8_exact_pair_sparse_metrics_unreadable:{exc}")
        return reasons
    try:
        dense_metrics = _load_regular_json_object(Path(dense_metrics_path))
    except (OSError, ValueError) as exc:
        reasons.append(f"tp8_exact_pair_dense_metrics_unreadable:{exc}")
        return reasons
    raw_runner_observed: dict[str, dict[str, Any]] = {}
    for arm, metrics in (("sparse", sparse_metrics), ("dense", dense_metrics)):
        run_config = metrics.get("run_config")
        nested = run_config if isinstance(run_config, dict) else {}
        observed = {
            "top_level": metrics.get("runner"),
            "run_config": nested.get("runner"),
        }
        raw_runner_observed[arm] = observed
        for location in ("top_level", "run_config"):
            if observed.get(location) != _PAIR_RUNNER_CONTRACT[arm]:
                reasons.append(
                    f"tp8_exact_pair_{arm}_raw_runner_mismatch:{location}"
                )
    if summary.get("sparse_dense_pair_arm_runner_observed") != raw_runner_observed:
        reasons.append("tp8_exact_pair_runner_observed_not_raw")
    sparse_boundary = sparse_metrics.get("boundary_diagnostics")
    if isinstance(sparse_boundary, dict) and sparse_boundary.get(
        "cudagraph_runtime_observer_enabled"
    ) is True:
        reasons.append("tp8_exact_pair_sparse_timed_observer_present")
    raw_ratio_inputs = {
        "total_wall_speedup": (
            dense_metrics.get("elapsed_s"),
            sparse_metrics.get("elapsed_s"),
        ),
        "total_token_speedup": (
            sparse_metrics.get("tok_per_s"),
            dense_metrics.get("tok_per_s"),
        ),
        "decode_speedup": (
            sparse_metrics.get("decode_tok_per_s"),
            dense_metrics.get("decode_tok_per_s"),
        ),
        "all_decode_speedup": (
            sparse_metrics.get("all_decode_tok_per_s"),
            dense_metrics.get("all_decode_tok_per_s"),
        ),
    }
    for field, (numerator, denominator) in raw_ratio_inputs.items():
        if not _is_finite_positive_number(numerator) or not _is_finite_positive_number(
            denominator
        ):
            reasons.append(f"tp8_exact_pair_ratio_input_invalid:{field}")
            continue
        expected_ratio = float(numerator) / float(denominator)
        recorded_ratio = summary.get(field)
        if not isinstance(recorded_ratio, (int, float)) or isinstance(
            recorded_ratio, bool
        ) or not math.isclose(
            float(recorded_ratio),
            expected_ratio,
            rel_tol=1e-12,
            abs_tol=0.0,
        ):
            reasons.append(f"tp8_exact_pair_ratio_recompute_mismatch:{field}")
    for field in ("out_tokens", "decode_tokens", "all_decode_tokens"):
        if sparse_metrics.get(field) != dense_metrics.get(field):
            reasons.append(f"tp8_exact_pair_token_count_mismatch:{field}")
    boundary = dense_metrics.get("boundary_diagnostics")
    if not isinstance(boundary, dict):
        reasons.append("tp8_exact_pair_dense_boundary_missing")
    else:
        exact_boundary = {
            "all_decode_entered": True,
            "all_decode_partial_batch_steps": 0,
            "all_decode_zero_token_steps": 0,
            "all_decode_fallback_used": False,
        }
        for field, expected in exact_boundary.items():
            if boundary.get(field) != expected:
                reasons.append(f"tp8_exact_pair_dense_boundary_mismatch:{field}")
        if type(boundary.get("all_decode_full_batch_steps")) is not int or boundary.get(
            "all_decode_full_batch_steps"
        ) <= 0:
            reasons.append("tp8_exact_pair_dense_full_decode_steps_missing")
        if boundary.get("cudagraph_runtime_observer_enabled") is True:
            reasons.append("tp8_exact_pair_dense_timed_observer_present")

    sparse_geometry = summary.get("sparse_dense_pair_sparse_geometry")
    dense_geometry = summary.get("sparse_dense_pair_dense_geometry")
    arm_invariant_fields = (
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
        "max_new_tokens",
        "max_model_len",
        "runner_tier",
        "expected_git_commit",
        "gpu_lock_mode",
        "gpu_lock_scope",
        "flash_attn_version",
        "attention_build_identity_json",
    )
    expected_identity = {
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
        "tensor_parallel_size": 8,
        "cuda_visible_devices": provenance.get("cuda_visible_devices_env"),
        "cuda_capabilities": provenance.get("runner_cuda_capabilities"),
        "batch_size": 32,
        "context_tokens": 64_000,
        "max_new_tokens": 2_048,
        "max_model_len": 66_560,
        "runner_tier": "tp8x64k",
        "expected_git_commit": provenance.get("runner_expected_git_commit"),
        "gpu_lock_mode": "exclusive",
        "gpu_lock_scope": "pair",
        "flash_attn_version": "3",
    }

    def _raw_child_identity(metrics: dict[str, Any]) -> dict[str, Any]:
        run_config = metrics.get("run_config")
        if not isinstance(run_config, dict):
            return {}
        identity = run_config.get("benchmark_child_identity")
        return dict(identity) if isinstance(identity, dict) else {}

    raw_identities = {
        "sparse": _raw_child_identity(sparse_metrics),
        "dense": _raw_child_identity(dense_metrics),
    }
    summary_identities = {
        "sparse": summary.get("sparse_dense_pair_sparse_child_identity"),
        "dense": summary.get("sparse_dense_pair_dense_child_identity"),
    }
    geometries = {"sparse": sparse_geometry, "dense": dense_geometry}
    invariant_projections: dict[str, dict[str, Any]] = {}
    for arm in ("sparse", "dense"):
        geometry = geometries[arm]
        if not isinstance(geometry, dict):
            reasons.append(f"tp8_exact_pair_{arm}_geometry_invalid")
            continue
        raw_identity = raw_identities[arm]
        if not raw_identity:
            reasons.append(f"tp8_exact_pair_{arm}_raw_child_identity_invalid")
            continue
        for field, expected in expected_identity.items():
            if raw_identity.get(field) != expected:
                reasons.append(
                    f"tp8_exact_pair_{arm}_child_identity_mismatch:{field}"
                )
        if raw_identity.get("attention_backend") != _PAIR_BACKEND_CONTRACT[arm]:
            reasons.append(
                f"tp8_exact_pair_{arm}_attention_backend_mismatch"
            )
        build_identity_raw = raw_identity.get("attention_build_identity_json")
        try:
            build_identity = json.loads(build_identity_raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            build_identity = None
        if build_identity != provenance.get("runner_attention_build_identity"):
            reasons.append(
                f"tp8_exact_pair_{arm}_attention_build_identity_mismatch"
            )
        if summary_identities[arm] != raw_identity:
            reasons.append(
                f"tp8_exact_pair_{arm}_full_child_identity_not_raw"
            )
        projection = {
            field: raw_identity.get(field) for field in arm_invariant_fields
        }
        invariant_projections[arm] = projection
        if geometry.get("child_identity") != projection:
            reasons.append(
                f"tp8_exact_pair_{arm}_child_identity_projection_mismatch"
            )
    if invariant_projections.get("sparse") != invariant_projections.get("dense"):
        reasons.append("tp8_exact_pair_child_identity_invariant_mismatch")
    return reasons


def _tp8_exact_arm_lifecycle_reasons(
    summary: dict[str, Any],
    *,
    summary_path: Path,
    mode: str,
) -> list[str]:
    if mode != "sparse":
        return []
    reasons: list[str] = []
    expected_setup_order = ["selector_prewarm"]
    expected_order = [
        "sparse_speed",
        "dense_reference",
        "sparse_diagnostic",
    ]
    exact_fields = {
        "tp8_arm_lifecycle_required": True,
        "tp8_arm_lifecycle_expected_order": expected_order,
        "tp8_arm_lifecycle_execution_order": expected_order,
        "tp8_arm_lifecycle_setup_expected_order": expected_setup_order,
        "tp8_arm_lifecycle_setup_execution_order": expected_setup_order,
        "tp8_arm_lifecycle_gate_passed": True,
        "tp8_arm_lifecycle_gate_reasons": [],
    }
    for field, expected in exact_fields.items():
        if summary.get(field) != expected:
            reasons.append(f"tp8_exact_lifecycle_field_mismatch:{field}")

    if not summary_path.name.endswith("_summary.json"):
        reasons.append("tp8_exact_lifecycle_summary_name_invalid")
        return reasons
    artifact_prefix = summary_path.name.removesuffix("_summary.json")
    expected_baseline_path = summary_path.with_name(
        artifact_prefix + "_tp8_process_baseline.json"
    ).resolve(strict=False)
    provenance = summary.get("run_provenance")
    run_nonce = (
        str(provenance.get("run_nonce", "") or "")
        if isinstance(provenance, dict)
        else ""
    )
    if not run_nonce:
        reasons.append("tp8_exact_lifecycle_run_nonce_missing")
    else:
        serialized_summary = json.dumps(
            summary,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            default=str,
        )
        for arm_tag in (*expected_setup_order, *expected_order):
            if derive_arm_token(run_nonce, arm_tag) in serialized_summary:
                reasons.append(
                    f"tp8_exact_lifecycle_{arm_tag}_raw_arm_token_exposed"
                )
    cuda_visible_devices = (
        str(provenance.get("cuda_visible_devices_env", "") or "")
        if isinstance(provenance, dict)
        else ""
    )
    gpu_fields = cuda_visible_devices.split(",")
    expected_gpu_ids = (
        [int(field) for field in gpu_fields]
        if len(gpu_fields) == 8 and all(field.isdigit() for field in gpu_fields)
        else []
    )
    if len(expected_gpu_ids) != 8 or len(set(expected_gpu_ids)) != 8:
        reasons.append("tp8_exact_lifecycle_gpu_identity_invalid")

    baseline_record = summary.get("tp8_arm_lifecycle_baseline")
    baseline_evidence: dict[str, Any] = {}
    baseline_sha256 = ""
    baseline_gpu_uuids: dict[str, Any] = {}
    previous_monotonic_ns = -1
    if not isinstance(baseline_record, dict):
        reasons.append("tp8_exact_lifecycle_baseline_record_invalid")
    else:
        baseline_path_raw = str(baseline_record.get("path", "") or "")
        baseline_path = Path(baseline_path_raw)
        if baseline_path.resolve(strict=False) != expected_baseline_path:
            reasons.append("tp8_exact_lifecycle_baseline_path_mismatch")
        try:
            baseline_evidence = _load_regular_json_object(baseline_path)
            baseline_sha256 = _file_sha256(baseline_path)
        except (OSError, ValueError) as exc:
            reasons.append(f"tp8_exact_lifecycle_baseline_unreadable:{exc}")
        else:
            if baseline_record.get("evidence") != baseline_evidence:
                reasons.append("tp8_exact_lifecycle_baseline_not_raw")
            if baseline_record.get("sha256") != baseline_sha256:
                reasons.append("tp8_exact_lifecycle_baseline_sha256_mismatch")
            expected_baseline_fields = {
                "schema": "sfi.tp8_process_baseline.v1",
                "status": "pass",
                "selected_gpu_indices": expected_gpu_ids,
                "attribution_scope": "selected_gpu_idle",
                "timeout_s": 30.0,
                "stable_clean_polls_required": 2,
                "selected_gpu_compute_processes": [],
                "attributed_runtime_processes": [],
                "inspection_errors": [],
                "reasons": [],
            }
            for field, expected in expected_baseline_fields.items():
                if baseline_evidence.get(field) != expected:
                    reasons.append(
                        f"tp8_exact_lifecycle_baseline_field_mismatch:{field}"
                    )
            baseline_gpu_uuids_raw = baseline_evidence.get("selected_gpu_uuids")
            if isinstance(baseline_gpu_uuids_raw, dict):
                baseline_gpu_uuids = baseline_gpu_uuids_raw
            expected_uuid_keys = {str(gpu_id) for gpu_id in expected_gpu_ids}
            uuid_values = list(baseline_gpu_uuids.values())
            uuid_values_valid = all(
                isinstance(value, str) and bool(value) for value in uuid_values
            )
            if (
                set(baseline_gpu_uuids) != expected_uuid_keys
                or len(uuid_values) != 8
                or not uuid_values_valid
                or (uuid_values_valid and len(set(uuid_values)) != 8)
            ):
                reasons.append("tp8_exact_lifecycle_baseline_gpu_uuids_invalid")
            baseline_monotonic_ns = baseline_evidence.get(
                "captured_monotonic_ns"
            )
            if type(baseline_monotonic_ns) is not int or baseline_monotonic_ns <= 0:
                reasons.append("tp8_exact_lifecycle_baseline_time_invalid")
            else:
                previous_monotonic_ns = baseline_monotonic_ns
            baseline_stable_observed = baseline_evidence.get(
                "stable_clean_polls_observed"
            )
            baseline_poll_count = baseline_evidence.get("poll_count")
            if (
                type(baseline_stable_observed) is not int
                or baseline_stable_observed < 2
                or type(baseline_poll_count) is not int
                or baseline_poll_count < baseline_stable_observed
            ):
                reasons.append("tp8_exact_lifecycle_baseline_stable_clean_invalid")

    setup_records = summary.get("tp8_arm_lifecycle_setup_records")
    records = summary.get("tp8_arm_lifecycle_records")
    record_groups = (
        ("setup", expected_setup_order, setup_records),
        ("arm", expected_order, records),
    )
    invalid_records = False
    for label, expected_tags, group_records in record_groups:
        if not isinstance(group_records, list) or len(group_records) != len(
            expected_tags
        ):
            reasons.append(f"tp8_exact_lifecycle_{label}_records_invalid")
            invalid_records = True
    if invalid_records:
        return reasons

    for _, expected_tags, group_records in record_groups:
        assert isinstance(group_records, list)
        for arm_tag, record in zip(expected_tags, group_records, strict=True):
            if not isinstance(record, dict):
                reasons.append(f"tp8_exact_lifecycle_{arm_tag}_record_invalid")
                continue
            child_session_id = record.get("child_session_id")
            expected_record_fields = {
                "arm_tag": arm_tag,
                "result_returncode": 0,
                "result_timed_out": False,
                "fatal_error_detected": False,
                "health_reasons": [],
                "teardown_reasons": [],
                "passed": True,
            }
            for field, expected in expected_record_fields.items():
                if record.get(field) != expected:
                    reasons.append(
                        f"tp8_exact_lifecycle_{arm_tag}_record_mismatch:{field}"
                    )
            if type(child_session_id) is not int or child_session_id <= 0:
                reasons.append(
                    f"tp8_exact_lifecycle_{arm_tag}_child_session_invalid"
                )
            try:
                expected_token_sha256 = arm_token_sha256(
                    derive_arm_token(run_nonce, arm_tag)
                )
            except ValueError:
                expected_token_sha256 = ""
            if record.get("arm_token_sha256") != expected_token_sha256:
                reasons.append(
                    f"tp8_exact_lifecycle_{arm_tag}_arm_token_hash_mismatch"
                )
            if "arm_token" in record:
                reasons.append(
                    f"tp8_exact_lifecycle_{arm_tag}_raw_arm_token_exposed"
                )
            expected_teardown_path = summary_path.with_name(
                artifact_prefix + f"_tp8_{arm_tag}_teardown.json"
            ).resolve(strict=False)
            teardown_path = Path(str(record.get("teardown_path", "") or ""))
            if teardown_path.resolve(strict=False) != expected_teardown_path:
                reasons.append(f"tp8_exact_lifecycle_{arm_tag}_path_mismatch")
            try:
                evidence = _load_regular_json_object(teardown_path)
                teardown_sha256 = _file_sha256(teardown_path)
            except (OSError, ValueError) as exc:
                reasons.append(
                    f"tp8_exact_lifecycle_{arm_tag}_evidence_unreadable:{exc}"
                )
                continue
            if record.get("teardown_evidence") != evidence:
                reasons.append(f"tp8_exact_lifecycle_{arm_tag}_evidence_not_raw")
            if record.get("teardown_sha256") != teardown_sha256:
                reasons.append(f"tp8_exact_lifecycle_{arm_tag}_sha256_mismatch")
            expected_teardown_fields = {
                "schema": "sfi.tp8_arm_teardown.v1",
                "status": "pass",
                "arm_tag": arm_tag,
                "selected_gpu_indices": expected_gpu_ids,
                "selected_gpu_uuids": baseline_gpu_uuids,
                "attribution_scope": TP8_ATTRIBUTION_SCOPE,
                "child_session_id": child_session_id,
                "arm_token_sha256": expected_token_sha256,
                "baseline_path": str(expected_baseline_path),
                "baseline_sha256": baseline_sha256,
                "timeout_s": 30.0,
                "stable_clean_polls_required": 2,
                "selected_gpu_compute_processes": [],
                "attributed_runtime_processes": [],
                "inspection_errors": [],
                "reasons": [],
            }
            for field, expected in expected_teardown_fields.items():
                if evidence.get(field) != expected:
                    reasons.append(
                        f"tp8_exact_lifecycle_{arm_tag}_field_mismatch:{field}"
                    )
            if "arm_token" in evidence:
                reasons.append(
                    f"tp8_exact_lifecycle_{arm_tag}_raw_arm_token_exposed"
                )
            captured_monotonic_ns = evidence.get("captured_monotonic_ns")
            if (
                type(captured_monotonic_ns) is not int
                or captured_monotonic_ns <= previous_monotonic_ns
            ):
                reasons.append(f"tp8_exact_lifecycle_{arm_tag}_order_invalid")
            else:
                previous_monotonic_ns = captured_monotonic_ns
            stable_required = evidence.get("stable_clean_polls_required")
            stable_observed = evidence.get("stable_clean_polls_observed")
            poll_count = evidence.get("poll_count")
            if (
                stable_required != 2
                or type(stable_observed) is not int
                or stable_observed < stable_required
                or type(poll_count) is not int
                or poll_count < stable_observed
            ):
                reasons.append(
                    f"tp8_exact_lifecycle_{arm_tag}_stable_clean_invalid"
                )
    return reasons


def _tp_sparse_rank_reasons(
    summary: dict[str, Any],
    *,
    summary_path: Path,
    expected_tier: str | None,
    selector_cache_root: Path | None,
) -> list[str]:
    provenance = summary.get("run_provenance")
    if not isinstance(provenance, dict):
        return ["run_provenance_missing_or_invalid"]
    try:
        tensor_parallel_size = int(provenance.get("tensor_parallel_size", 0))
    except (TypeError, ValueError):
        tensor_parallel_size = 0
    if tensor_parallel_size <= 1:
        return []
    reasons: list[str] = []
    # Timed proof is intentionally observer-free: rank-local RPC/mmap counters
    # are the sole speed-child evidence. JSONL route tracing belongs only to the
    # paired diagnostic child and must never perturb the measured arm.
    metrics_path_text = str(summary.get("decode_metrics_path", "") or "")
    if not metrics_path_text:
        reasons.append("tp_sparse_route_counter_metrics_path_missing")
        return reasons
    metrics_path = Path(metrics_path_text)
    if not summary_path.name.endswith("_summary.json"):
        reasons.append("tp_sparse_summary_name_not_canonical")
    else:
        expected_metrics_path = summary_path.with_name(
            summary_path.name.removesuffix("_summary.json") + "_decode_metrics.json"
        ).resolve(strict=False)
        if metrics_path.resolve(strict=False) != expected_metrics_path:
            reasons.append(
                "tp_sparse_route_counter_metrics_path_mismatch:"
                f"actual={metrics_path}:expected={expected_metrics_path}"
            )
    try:
        metrics = _load_regular_json_object(metrics_path)
    except (OSError, ValueError) as exc:
        reasons.append(f"tp_sparse_route_counter_metrics_invalid:{exc}")
        return reasons

    if metrics.get("speed_child_route_counter_scope") != "measurement_window":
        reasons.append("tp_sparse_route_counter_scope_mismatch")
    if metrics.get("speed_child_route_counter_available") is not True:
        reasons.append("tp_sparse_route_counter_unavailable")
    slots = metrics.get("speed_child_route_counter_rank_slots")
    if type(slots) is not int or slots != tensor_parallel_size:
        reasons.append(
            "tp_sparse_route_counter_slot_count_mismatch:"
            f"actual={slots!r}:expected={tensor_parallel_size}"
        )
    if metrics.get("speed_child_route_counter_rank_consistent") is not True:
        reasons.append("tp_sparse_route_counter_rank_consistency_not_green")

    def _rpc_record_evidence(
        field: str,
        *,
        require_zero: bool,
    ) -> tuple[list[tuple[int, ...]] | None, list[int] | None]:
        evidence_name = (
            "reset" if field == "speed_child_route_counter_reset_records" else "snapshot"
        )
        raw_records = metrics.get(field)
        if (
            not isinstance(raw_records, list)
            or len(raw_records) != tensor_parallel_size
        ):
            reasons.append(
                f"tp_sparse_route_counter_{evidence_name}_records_shape_mismatch:"
                f"actual={len(raw_records) if isinstance(raw_records, list) else type(raw_records).__name__}:"
                f"expected={tensor_parallel_size}"
            )
            return None, None

        values_by_rank: list[tuple[int, ...]] = []
        pids: list[int] = []
        valid = True
        expected_mmap_size = tensor_parallel_size * 10 * 8
        for expected_rank, record in enumerate(raw_records):
            if not isinstance(record, dict):
                reasons.append(
                    f"tp_sparse_route_counter_{evidence_name}_record_invalid:"
                    f"rank={expected_rank}:record_type={type(record).__name__}"
                )
                valid = False
                continue
            rank = record.get("rank")
            slot_count = record.get("slot_count")
            field_count = record.get("field_count")
            values = record.get("values")
            mmap_size = record.get("mmap_size_bytes")
            pid = record.get("pid")
            if type(rank) is not int or rank != expected_rank:
                reasons.append(
                    f"tp_sparse_route_counter_{evidence_name}_rank_mismatch:"
                    f"index={expected_rank}:actual={rank!r}"
                )
                valid = False
            if type(slot_count) is not int or slot_count != tensor_parallel_size:
                reasons.append(
                    f"tp_sparse_route_counter_{evidence_name}_slot_count_mismatch:"
                    f"rank={expected_rank}:actual={slot_count!r}:"
                    f"expected={tensor_parallel_size}"
                )
                valid = False
            if type(field_count) is not int or field_count != 10:
                reasons.append(
                    f"tp_sparse_route_counter_{evidence_name}_field_count_mismatch:"
                    f"rank={expected_rank}:actual={field_count!r}:expected=10"
                )
                valid = False
            values_valid = bool(
                isinstance(values, list)
                and len(values) == 10
                and all(type(value) is int and value >= 0 for value in values)
            )
            if not values_valid:
                reasons.append(
                    f"tp_sparse_route_counter_{evidence_name}_values_invalid:"
                    f"rank={expected_rank}:actual={values!r}"
                )
                valid = False
            elif require_zero and any(value != 0 for value in values):
                reasons.append(
                    f"tp_sparse_route_counter_{evidence_name}_values_not_zero:"
                    f"rank={expected_rank}:actual={values!r}"
                )
                valid = False
            if type(mmap_size) is not int or mmap_size != expected_mmap_size:
                reasons.append(
                    f"tp_sparse_route_counter_{evidence_name}_mmap_size_mismatch:"
                    f"rank={expected_rank}:actual={mmap_size!r}:"
                    f"expected={expected_mmap_size}"
                )
                valid = False
            if type(pid) is not int or pid <= 0:
                reasons.append(
                    f"tp_sparse_route_counter_{evidence_name}_pid_invalid:"
                    f"rank={expected_rank}:actual={pid!r}"
                )
                valid = False
            if values_valid:
                values_by_rank.append(tuple(int(value) for value in values))
            if type(pid) is int and pid > 0:
                pids.append(pid)

        if len(pids) != tensor_parallel_size or len(set(pids)) != tensor_parallel_size:
            reasons.append(
                f"tp_sparse_route_counter_{evidence_name}_pids_not_unique:"
                f"actual={pids!r}"
            )
            valid = False
        if not valid or len(values_by_rank) != tensor_parallel_size:
            return None, None
        return values_by_rank, pids

    _reset_values, reset_pids = _rpc_record_evidence(
        "speed_child_route_counter_reset_records",
        require_zero=True,
    )
    snapshot_values, snapshot_pids = _rpc_record_evidence(
        "speed_child_route_counter_snapshot_records",
        require_zero=False,
    )
    if (
        reset_pids is not None
        and snapshot_pids is not None
        and reset_pids != snapshot_pids
    ):
        reasons.append(
            "tp_sparse_route_counter_reset_snapshot_pid_mismatch:"
            f"reset={reset_pids!r}:snapshot={snapshot_pids!r}"
        )
    if expected_tier == "tp8x64k":
        if selector_cache_root is None:
            reasons.append("tp8_selector_log_s_trusted_cache_root_missing")
        else:
            reasons.extend(
                _tp8_selector_log_s_runtime_reasons(
                    metrics,
                    tensor_parallel_size=tensor_parallel_size,
                    selector_cache_root=Path(selector_cache_root),
                )
            )

    per_rank = metrics.get("speed_child_route_counter_per_rank")
    if not isinstance(per_rank, list) or len(per_rank) != tensor_parallel_size:
        reasons.append(
            "tp_sparse_route_counter_per_rank_shape_mismatch:"
            f"actual={len(per_rank) if isinstance(per_rank, list) else type(per_rank).__name__}:"
            f"expected={tensor_parallel_size}"
        )
        return reasons

    signature_fields = (
        "actual_fwd_mixed_page_count",
        "resolved_row_ptr_fwd_mixed_page_count",
        "has_resolved_row_ptr_count",
        "compact_row_steps",
        "compact_rows",
    )
    signatures: list[tuple[int, ...]] = []
    expected_snapshot_values: list[tuple[int, ...]] = []
    valid_records = True
    aggregate_values = {field: 0 for field in signature_fields[:3]}
    aggregate_kinds = {kind: 0 for kind in range(5)}
    for expected_rank, record in enumerate(per_rank):
        rank_value = record.get("rank") if isinstance(record, dict) else None
        if (
            not isinstance(record, dict)
            or type(rank_value) is not int
            or rank_value != expected_rank
        ):
            reasons.append(
                "tp_sparse_route_counter_rank_record_invalid:"
                f"index={expected_rank}:rank="
                f"{rank_value!r}"
            )
            valid_records = False
            continue
        values = [record.get(field) for field in signature_fields]
        kind_counts = record.get("page_resolver_kind_counts")
        kinds = [
            kind_counts.get(str(kind)) if isinstance(kind_counts, dict) else None
            for kind in range(5)
        ]
        if any(type(value) is not int or value < 0 for value in (*values, *kinds)):
            reasons.append(
                f"tp_sparse_route_counter_rank_values_invalid:rank={expected_rank}"
            )
            valid_records = False
            continue
        actual, resolved, has_resolved, compact_steps, compact_rows = (
            int(value) for value in values
        )
        if sum(int(value) for value in kinds) != actual:
            reasons.append(
                "tp_sparse_route_counter_kind_partition_mismatch:"
                f"rank={expected_rank}"
            )
            valid_records = False
        if resolved != int(kinds[4]):
            reasons.append(
                "tp_sparse_route_counter_resolved_kind4_mismatch:"
                f"rank={expected_rank}"
            )
            valid_records = False
        if resolved <= 0 or any(int(value) != 0 for value in kinds[1:4]):
            reasons.append(
                "tp_sparse_route_counter_resolved_route_missing_or_polluted:"
                f"rank={expected_rank}"
            )
            valid_records = False
        if not 0 <= resolved <= has_resolved <= actual:
            reasons.append(
                "tp_sparse_route_counter_resolved_bounds_invalid:"
                f"rank={expected_rank}"
            )
            valid_records = False
        if compact_steps <= 0 or compact_rows < compact_steps:
            reasons.append(
                "tp_sparse_route_counter_compact_liveness_invalid:"
                f"rank={expected_rank}"
            )
            valid_records = False
        if values[0] <= 0:
            reasons.append(
                f"tp_sparse_route_counter_rank_inactive:rank={expected_rank}"
            )
            valid_records = False
        signature = tuple(int(value) for value in (*values, *kinds))
        signatures.append(signature)
        expected_snapshot_values.append(
            (
                actual,
                resolved,
                has_resolved,
                *(int(value) for value in kinds),
                compact_steps,
                compact_rows,
            )
        )
        for field, value in zip(signature_fields[:3], values[:3]):
            aggregate_values[field] += int(value)
        for kind, value in enumerate(kinds):
            aggregate_kinds[kind] += int(value)

    if valid_records and signatures and any(
        signature != signatures[0] for signature in signatures[1:]
    ):
        reasons.append("tp_sparse_route_counter_rank_values_diverged")
    if (
        valid_records
        and snapshot_values is not None
        and snapshot_values != expected_snapshot_values
    ):
        reasons.append(
            "tp_sparse_route_counter_snapshot_per_rank_mismatch:"
            f"actual={snapshot_values!r}:expected={expected_snapshot_values!r}"
        )
    if valid_records:
        aggregate_fields = {
            "speed_child_actual_fwd_mixed_page_count": aggregate_values[
                "actual_fwd_mixed_page_count"
            ],
            "speed_child_resolved_row_ptr_fwd_mixed_page_count": aggregate_values[
                "resolved_row_ptr_fwd_mixed_page_count"
            ],
            "speed_child_has_resolved_row_ptr_count": aggregate_values[
                "has_resolved_row_ptr_count"
            ],
            **{
                f"speed_child_page_resolver_kind{kind}_count": aggregate_kinds[kind]
                for kind in range(5)
            },
        }
        for field, expected in aggregate_fields.items():
            actual = metrics.get(field)
            if type(actual) is not int or actual != expected:
                reasons.append(
                    "tp_sparse_route_counter_aggregate_mismatch:"
                    f"field={field}:actual={actual!r}:expected={expected}"
                )
    return reasons


def _harness_condition_reasons(
    summary: dict[str, Any],
    *,
    summary_path: Path,
    mode: str,
    semantic_gate_reasons: list[object],
    producer_gate_reasons: list[object],
    run_nonce: str,
    expected_tier: str | None,
    selector_cache_root: Path | None,
    expected_cuda_arch: str | None,
    expected_attention_kernel: str | None,
    expected_backend: str | None,
    expected_flash_attn_version: int | None,
    expected_git_commit: str | None,
    expected_model_config_sha256: str | None,
    expect_local_paired_comparison: bool = False,
    allow_local_pair_comparison_only: bool = False,
) -> list[str]:
    reasons: list[str] = []
    exact_tier = expected_tier == "tp8x64k"
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
        if expected_backend is None:
            # Backward-compatible library use for historical SM80 summaries.
            if provenance.get("runner_fa3_preflight_status") != "passed":
                reasons.append("runner_fa3_preflight_not_green")
        else:
            if provenance.get("runner_attention_preflight_status") != "passed":
                reasons.append("runner_attention_preflight_not_green")
            expected_fa3_status = (
                "passed" if expected_flash_attn_version == 3 else "not_applicable"
            )
            if provenance.get("runner_fa3_preflight_status") != expected_fa3_status:
                reasons.append("runner_fa3_preflight_status_mismatch")
            expected_fields = {
                "runner_attention_arch": expected_cuda_arch,
                "runner_attention_kernel": expected_attention_kernel,
                "runner_attention_backend": expected_backend,
                "runner_flash_attn_version": str(expected_flash_attn_version),
                "backend": expected_backend,
                "flash_attn_version_expected": expected_flash_attn_version,
            }
            for field, expected in expected_fields.items():
                if provenance.get(field) != expected:
                    reasons.append(
                        f"runner_attention_identity_mismatch:{field}:"
                        f"actual={provenance.get(field)!r}:expected={expected!r}"
                    )
            reasons.extend(
                _attention_build_identity_reasons(
                    provenance,
                    expected_cuda_arch=str(expected_cuda_arch),
                    expected_attention_kernel=str(expected_attention_kernel),
                )
            )
        if expected_tier is not None or expected_backend is not None:
            reasons.extend(
                _runner_config_identity_reasons(
                    summary,
                    mode=mode,
                    expected_tier=expected_tier,
                    selector_cache_root=selector_cache_root,
                    expected_model_config_sha256=(
                        expected_model_config_sha256
                    ),
                )
            )
        if expected_git_commit is not None:
            expected_git_fields = {
                "git_head": expected_git_commit,
                "runner_expected_git_commit": expected_git_commit,
                "runner_git_head": expected_git_commit,
                "runner_git_tracked_clean": "1",
                "git_tracked_status_short": "",
            }
            for field, expected in expected_git_fields.items():
                if provenance.get(field) != expected:
                    reasons.append(
                        f"runner_release_identity_mismatch:{field}:"
                        f"actual={provenance.get(field)!r}:expected={expected!r}"
                    )
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
    speed_observer_env = summary.get("speed_trace_profile_env")
    if not isinstance(speed_observer_env, dict):
        reasons.append("speed_observer_env_missing_or_invalid")
    elif speed_observer_env:
        reasons.append("speed_observer_env_nonempty")
    if summary.get("speed_trace_profile_env_empty") is not True:
        reasons.append("speed_observer_env_not_proven_empty")

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
    if exact_tier and mode == "sparse" and verdict_only is not False:
        reasons.append("tp8_exact_verdict_only_forbidden")
    full_form = mode == "dense" or verdict_only is False
    if expected_tier is not None and full_form:
        if summary.get("speed_diagnostic_pairing_required") is not True:
            reasons.append("speed_diagnostic_pairing_not_required")
        if summary.get("speed_diagnostic_pairing_match") is not True:
            reasons.append("speed_diagnostic_pairing_mismatch_or_missing")
    if exact_tier and mode == "sparse":
        if str(summary.get("speed_child_route_trace_path", "") or ""):
            reasons.append("tp8_exact_timed_child_jsonl_observer_present")
        if summary.get("route_summary_source") != "diagnostic_child_route_trace":
            reasons.append("tp8_exact_route_summary_not_diagnostic")
    if exact_tier:
        exact_decode_window = {
            "all_decode_entered": True,
            "all_decode_partial_batch_steps": 0,
            "all_decode_zero_token_steps": 0,
            "all_decode_fallback_used": False,
        }
        for field, expected in exact_decode_window.items():
            actual = summary.get(field)
            if type(actual) is not type(expected) or actual != expected:
                reasons.append(
                    "tp8_exact_decode_window_mismatch:"
                    f"{field}:actual={actual!r}:expected={expected!r}"
                )
        full_batch_steps = summary.get("all_decode_full_batch_steps")
        if type(full_batch_steps) is not int or full_batch_steps <= 0:
            reasons.append("tp8_exact_full_batch_decode_steps_missing")

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
        reasons.extend(
            _tp_sparse_rank_reasons(
                summary,
                summary_path=summary_path,
                expected_tier=expected_tier,
                selector_cache_root=selector_cache_root,
            )
        )

    if expected_tier is not None:
        reasons.extend(
            _scheduler_graph_contract_reasons(
                summary,
                mode=mode,
                exact_tier=exact_tier,
            )
        )
    if exact_tier:
        reasons.extend(
            _exact_engine_runtime_and_pair_reasons(summary, mode=mode)
        )
        reasons.extend(
            _tp8_exact_arm_lifecycle_reasons(
                summary,
                summary_path=summary_path,
                mode=mode,
            )
        )
    if expect_local_paired_comparison:
        if mode != "sparse":
            reasons.append("local_pair_requires_sparse_mode")
        elif exact_tier:
            reasons.append("local_pair_cannot_replace_exact_tp8_verdict")
        else:
            reasons.extend(_explicit_local_pair_reasons(summary))

    if producer_gate_reasons:
        if summary.get("producer_gate_passed") is not False:
            reasons.append("producer_gate_state_inconsistent_with_reasons")
    elif summary.get("producer_gate_passed") is not True:
        reasons.append("producer_gate_not_green_without_reason")

    all_gate_reasons = semantic_gate_reasons + producer_gate_reasons
    only_allowed_gate_noise = mode == "sparse" and bool(all_gate_reasons) and all(
        _is_allowed_gate_noise(reason, exact_tier=exact_tier)
        for reason in all_gate_reasons
    )
    production_gate_passed = summary.get("production_gate_passed")
    if allow_local_pair_comparison_only:
        if production_gate_passed is not False:
            reasons.append(
                "production_gate_state_inconsistent_with_local_comparison_only"
            )
    elif only_allowed_gate_noise:
        if production_gate_passed is not False:
            reasons.append("production_gate_state_inconsistent_with_allowed_noise")
    elif production_gate_passed is not True:
        reasons.append("production_gate_not_green_without_only_allowed_noise")

    reasons.extend(
        _reference_gate_integrity_reasons(
            summary,
            mode=mode,
            exact_tier=exact_tier,
            allow_local_pair_comparison_only=allow_local_pair_comparison_only,
        )
    )
    return reasons


def check_run_speed_summary(
    summary_path: Path,
    *,
    mode: str,
    harness_returncode: int,
    run_started_ns: int,
    run_nonce: str,
    selector_cache_root: Path | None,
    expected_tier: str | None = None,
    expected_cuda_arch: str | None = None,
    expected_attention_kernel: str | None = None,
    expected_backend: str | None = None,
    expected_flash_attn_version: int | None = None,
    expected_git_commit: str | None = None,
    expected_model_config_sha256: str | None = None,
    expect_local_paired_comparison: bool = False,
) -> tuple[int, list[str]]:
    messages: list[str] = []
    if expected_tier is not None and expected_tier not in RUN_SPEED_TIERS:
        return _failure_returncode(harness_returncode), [
            f"FAIL: unsupported expected run tier: {expected_tier!r}"
        ]
    exact_tier = expected_tier == "tp8x64k"
    expected_identity = (
        expected_cuda_arch,
        expected_attention_kernel,
        expected_backend,
        expected_flash_attn_version,
    )
    if any(value is not None for value in expected_identity):
        if not all(value is not None for value in expected_identity):
            return _failure_returncode(harness_returncode), [
                "FAIL: expected attention identity must be provided as one complete tuple"
            ]
        valid_identities = {
            ("sm80", "fa3-native", "fa3", 3),
            ("sm90", "fa3-native", "fa3", 3),
            ("sm100", "fa4-cute", "fa4-sm100", 4),
        }
        if expected_identity not in valid_identities:
            return _failure_returncode(harness_returncode), [
                f"FAIL: unsupported expected attention identity: {expected_identity!r}"
            ]
    if expected_git_commit is not None and re.fullmatch(
        r"[0-9a-f]{40}", expected_git_commit
    ) is None:
        return _failure_returncode(harness_returncode), [
            "FAIL: expected Git commit must be one exact lowercase 40-character SHA"
        ]
    if exact_tier and expected_model_config_sha256 is None:
        return _failure_returncode(harness_returncode), [
            "FAIL: exact TP8 requires an external expected model config SHA256"
        ]
    if expected_model_config_sha256 is not None and re.fullmatch(
        r"[0-9a-f]{64}", expected_model_config_sha256
    ) is None:
        return _failure_returncode(harness_returncode), [
            "FAIL: expected model config SHA256 must be exactly 64 lowercase hex characters"
        ]
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
    local_pair_comparison_only = _is_documented_local_pair_comparison_only(
        summary,
        mode=mode,
        exact_tier=exact_tier,
        expect_local_paired_comparison=expect_local_paired_comparison,
        semantic_gate_reasons=semantic_gate_reasons,
        producer_gate_reasons=producer_gate_reasons,
    )
    unexpected_gate_reasons = (
        []
        if local_pair_comparison_only
        else [
            reason
            for reason in gate_reasons
            if mode != "sparse"
            or not _is_allowed_gate_noise(reason, exact_tier=exact_tier)
        ]
    )
    documented_gate_noise_present = bool(
        mode == "sparse"
        and any(
            _is_allowed_gate_noise(reason, exact_tier=exact_tier)
            for reason in gate_reasons
        )
    )
    selector_reasons = (
        _selector_artifact_reasons(
            summary,
            selector_cache_root=selector_cache_root,
        )
        if mode == "sparse"
        else []
    )
    harness_condition_reasons = (
        gate_reason_shape_reasons
        + _harness_condition_reasons(
            summary,
            summary_path=summary_path,
            mode=mode,
            semantic_gate_reasons=semantic_gate_reasons,
            producer_gate_reasons=producer_gate_reasons,
            run_nonce=run_nonce,
            expected_tier=expected_tier,
            selector_cache_root=selector_cache_root,
            expected_cuda_arch=expected_cuda_arch,
            expected_attention_kernel=expected_attention_kernel,
            expected_backend=expected_backend,
            expected_flash_attn_version=expected_flash_attn_version,
            expected_git_commit=expected_git_commit,
            expected_model_config_sha256=expected_model_config_sha256,
            expect_local_paired_comparison=expect_local_paired_comparison,
            allow_local_pair_comparison_only=local_pair_comparison_only,
        )
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
    if (
        harness_returncode == 2
        and not documented_gate_noise_present
        and not local_pair_comparison_only
    ):
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
        if local_pair_comparison_only:
            messages.append(
                "HARNESS returncode=2 accepted: local pair is valid; "
                "production correctness remains failed on the documented "
                "reference semantic mismatch"
            )
        else:
            messages.append(
                "HARNESS returncode=2 accepted: current summary contains only "
                "the documented no-reference gate noise"
            )
    if exact_tier and mode == "sparse":
        messages.append("SPARSE/DENSE PAIRED ENGINE-LOOP SPEEDUP OK")
    elif local_pair_comparison_only:
        messages.append(
            "SPARSE/DENSE LOCAL PAIRED COMPARISON ONLY "
            "(production correctness failed: reference_semantic_mismatch)"
        )
    elif expect_local_paired_comparison and mode == "sparse":
        messages.append("SPARSE/DENSE LOCAL PAIRED COMPARISON OK")
    else:
        messages.append("SPEED ARM HEALTHY (single arm; no speedup claim)")
    return 0, messages


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--mode", choices=("sparse", "dense"), required=True)
    parser.add_argument("--harness-returncode", type=int, required=True)
    parser.add_argument("--run-started-ns", type=int, required=True)
    parser.add_argument("--run-nonce", required=True)
    parser.add_argument("--selector-cache-root", type=Path, required=True)
    parser.add_argument("--expected-tier", choices=RUN_SPEED_TIERS, required=True)
    parser.add_argument("--expected-cuda-arch", choices=("sm80", "sm90", "sm100"))
    parser.add_argument(
        "--expected-attention-kernel",
        choices=("fa3-native", "fa4-cute"),
    )
    parser.add_argument("--expected-backend", choices=("fa3", "fa4-sm100"))
    parser.add_argument("--expected-flash-attn-version", type=int, choices=(3, 4))
    parser.add_argument("--expected-git-commit")
    parser.add_argument("--expected-model-config-sha256")
    parser.add_argument(
        "--expect-local-paired-comparison",
        action="store_true",
        help=(
            "Require an explicit local sparse/dense timed comparison. "
            "Ratios must be finite and positive, but may be at or below one."
        ),
    )
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
        expected_tier=args.expected_tier,
        expected_cuda_arch=args.expected_cuda_arch,
        expected_attention_kernel=args.expected_attention_kernel,
        expected_backend=args.expected_backend,
        expected_flash_attn_version=args.expected_flash_attn_version,
        expected_git_commit=args.expected_git_commit,
        expected_model_config_sha256=args.expected_model_config_sha256,
        expect_local_paired_comparison=args.expect_local_paired_comparison,
    )
    for message in messages:
        print(message)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
