"""Runtime identity contract for selector log_f production routes."""

from __future__ import annotations

import hashlib
import re
from enum import Enum
from typing import Any


_SELECTOR_LOG_S_KERNEL_NAMESPACE = "r2tiled_v12"
SELECTOR_LOG_S_EXTENSION_NAME = (
    f"selector_log_s_ext_{_SELECTOR_LOG_S_KERNEL_NAMESPACE}"
)
SELECTOR_LOG_S_SEMANTIC_IDENTITY = (
    f"selector_log_s.{_SELECTOR_LOG_S_KERNEL_NAMESPACE}.kernel_abi_v1"
)
SELECTOR_LOG_S_SEMANTIC_IDENTITY_SYMBOL = "selector_log_s_semantic_identity"
SELECTOR_LOG_S_RUNTIME_PROOF_SCHEMA = "sfi.selector_log_s_runtime_proof.v6"
SELECTOR_LOG_S_REQUIRED_TILED_SYMBOL = (
    "reduce_log_f_pre_scratch_r2_alpha0p5_fp16_tiled"
)


class SelectorLogFRoute(str, Enum):
    R2_RESIDENT = "r2_alpha0p5_fp16_resident_bucket_two_pass"
    R2_TILED = "r2_alpha0p5_fp16_tiled_four_stage"
    GENERIC = "generic_reduce"


SELECTOR_LOG_F_FAST_ROUTE = SelectorLogFRoute.R2_RESIDENT.value
SELECTOR_LOG_F_TILED_ROUTE = SelectorLogFRoute.R2_TILED.value
SELECTOR_LOG_F_GENERIC_ROUTE = SelectorLogFRoute.GENERIC.value
SELECTOR_LOG_F_ROUTES = frozenset(route.value for route in SelectorLogFRoute)
SELECTOR_LOG_F_AMORTIZED_TILED_ALLOWED_ROUTES = frozenset(
    (SELECTOR_LOG_F_FAST_ROUTE, SELECTOR_LOG_F_TILED_ROUTE)
)
SELECTOR_LOG_F_COUNTER_SEMANTICS = (
    "python_capture_route_ordered_admission_chain_phase_resettable"
)


def selector_log_s_module_signature(
    *,
    extension_name: object,
    module_name: object,
    module_path: object,
    module_sha256: object,
    module_size_bytes: object,
    required_symbol: object,
    semantic_identity_symbol: object,
    semantic_identity: object,
) -> str:
    """Hash the complete worker-local binary identity into one stable field."""

    canonical = "\n".join(
        (
            str(extension_name),
            str(module_name),
            str(module_path),
            str(module_sha256),
            str(module_size_bytes),
            str(required_symbol),
            str(semantic_identity_symbol),
            str(semantic_identity),
        )
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _expected_module_identity(expected_route: str | None) -> tuple[str, str] | None:
    if expected_route in SELECTOR_LOG_F_ROUTES:
        return (
            SELECTOR_LOG_S_EXTENSION_NAME,
            SELECTOR_LOG_S_REQUIRED_TILED_SYMBOL,
        )
    return None


def selector_log_s_runtime_proof_reasons(
    proof: object,
    *,
    expected_route: str | None,
    expected_phase: str = "snapshot",
    allowed_routes: frozenset[str] | None = None,
) -> list[str]:
    """Validate a reset or measured selector route proof without side effects."""

    if expected_route is not None and expected_route not in SELECTOR_LOG_F_ROUTES:
        raise ValueError(f"unsupported selector log_f route: {expected_route!r}")
    if allowed_routes is not None and (
        expected_route is None
        or expected_route not in allowed_routes
        or not allowed_routes
        or not allowed_routes.issubset(SELECTOR_LOG_F_ROUTES)
    ):
        raise ValueError(
            "allowed selector routes must be a non-empty route subset "
            "containing expected_route"
        )
    if expected_phase not in {"reset", "snapshot"}:
        raise ValueError(f"unsupported selector log_f proof phase: {expected_phase!r}")
    if not isinstance(proof, dict):
        return ["proof_missing_or_invalid"]

    exact: dict[str, object] = {
        "schema": SELECTOR_LOG_S_RUNTIME_PROOF_SCHEMA,
        "counter_semantics": SELECTOR_LOG_F_COUNTER_SEMANTICS,
        "required_symbol_present": True,
        "semantic_identity_symbol": SELECTOR_LOG_S_SEMANTIC_IDENTITY_SYMBOL,
        "semantic_identity": SELECTOR_LOG_S_SEMANTIC_IDENTITY,
    }
    expected_module = _expected_module_identity(expected_route)
    if expected_module is not None:
        expected_extension, expected_symbol = expected_module
        exact.update(
            {
                "extension_name": expected_extension,
                "module_name": expected_extension,
                "required_symbol": expected_symbol,
            }
        )
    reasons = [
        f"field_mismatch:{name}:actual={proof.get(name)!r}:expected={expected!r}"
        for name, expected in exact.items()
        if type(proof.get(name)) is not type(expected)
        or proof.get(name) != expected
    ]

    module_path = proof.get("module_path")
    if not isinstance(module_path, str) or not module_path.startswith("/"):
        reasons.append(f"module_path_invalid:{module_path!r}")
    module_sha256 = proof.get("module_sha256")
    if not isinstance(module_sha256, str) or re.fullmatch(
        r"[0-9a-f]{64}", module_sha256
    ) is None:
        reasons.append(f"module_sha256_invalid:{module_sha256!r}")
    module_size = proof.get("module_size_bytes")
    if type(module_size) is not int or module_size <= 0:
        reasons.append(f"module_size_invalid:{module_size!r}")
    module_signature = proof.get("module_signature")
    if not isinstance(module_signature, str) or re.fullmatch(
        r"[0-9a-f]{64}", module_signature
    ) is None:
        reasons.append(f"module_signature_invalid:{module_signature!r}")
    else:
        expected_signature = selector_log_s_module_signature(
            extension_name=proof.get("extension_name"),
            module_name=proof.get("module_name"),
            module_path=module_path,
            module_sha256=module_sha256,
            module_size_bytes=module_size,
            required_symbol=proof.get("required_symbol"),
            semantic_identity_symbol=proof.get("semantic_identity_symbol"),
            semantic_identity=proof.get("semantic_identity"),
        )
        if module_signature != expected_signature:
            reasons.append("module_signature_mismatch")

    admission_identity = proof.get("last_admission_identity")
    admission_identity_valid = bool(
        admission_identity == "none"
        or (
            isinstance(admission_identity, str)
            and re.fullmatch(r"[0-9a-f]{64}", admission_identity) is not None
        )
    )
    if not admission_identity_valid:
        reasons.append(
            f"last_admission_identity_invalid:{admission_identity!r}"
        )
    dispatch_reason = proof.get("last_dispatch_reason")
    if not isinstance(dispatch_reason, str) or (
        dispatch_reason != "none"
        and re.fullmatch(r"[a-z0-9_]+", dispatch_reason) is None
    ):
        reasons.append(f"last_dispatch_reason_invalid:{dispatch_reason!r}")
    event_count = proof.get("admission_event_count")
    if type(event_count) is not int or event_count < 0:
        reasons.append(f"admission_event_count_invalid:{event_count!r}")
    admission_chain = proof.get("admission_chain_sha256")
    if not isinstance(admission_chain, str) or re.fullmatch(
        r"[0-9a-f]{64}", admission_chain
    ) is None:
        reasons.append(f"admission_chain_sha256_invalid:{admission_chain!r}")

    counts = proof.get("route_counts")
    if not isinstance(counts, dict) or set(counts) != SELECTOR_LOG_F_ROUTES:
        reasons.append(f"route_counts_invalid:{counts!r}")
        return reasons
    if any(type(value) is not int or value < 0 for value in counts.values()):
        reasons.append(f"route_count_values_invalid:{counts!r}")
        return reasons
    if type(event_count) is int and event_count != sum(counts.values()):
        reasons.append(
            "admission_event_count_mismatch:"
            f"events={event_count}:routes={sum(counts.values())}"
        )

    tiled_cohort_count = proof.get("tiled_cohort_count")
    tiled_job_count = proof.get("tiled_job_count")
    tiled_direct_count = proof.get("tiled_direct_count")
    tiled_kernel_launch_count = proof.get("tiled_kernel_launch_count")
    admission_failures = proof.get("tiled_admission_failure_count")
    for name, value in (
        ("tiled_cohort_count", tiled_cohort_count),
        ("tiled_job_count", tiled_job_count),
        ("tiled_direct_count", tiled_direct_count),
        ("tiled_kernel_launch_count", tiled_kernel_launch_count),
        ("tiled_admission_failure_count", admission_failures),
    ):
        if type(value) is not int or value < 0:
            reasons.append(f"{name}_invalid:{value!r}")
    if reasons:
        return reasons

    if expected_phase == "reset":
        if any(counts.values()):
            reasons.append(f"reset_route_counts_nonzero:{counts!r}")
        if proof.get("last_route") != "none":
            reasons.append(f"reset_last_route_invalid:{proof.get('last_route')!r}")
        if dispatch_reason != "none":
            reasons.append(
                f"reset_last_dispatch_reason_invalid:{dispatch_reason!r}"
            )
        if admission_identity != "none":
            reasons.append(
                "reset_last_admission_identity_invalid:"
                f"{admission_identity!r}"
            )
        if event_count != 0:
            reasons.append(f"reset_admission_event_count_nonzero:{event_count!r}")
        if admission_chain != "0" * 64:
            reasons.append(
                f"reset_admission_chain_invalid:{admission_chain!r}"
            )
        if (
            tiled_cohort_count != 0
            or tiled_job_count != 0
            or tiled_direct_count != 0
            or tiled_kernel_launch_count != 0
        ):
            reasons.append(
                "reset_tiled_counts_nonzero:"
                f"cohorts={tiled_cohort_count}:jobs={tiled_job_count}:"
                f"direct={tiled_direct_count}:"
                f"kernels={tiled_kernel_launch_count}"
            )
        if admission_failures != 0:
            reasons.append(f"reset_admission_failures_nonzero:{admission_failures}")
        return reasons

    if admission_failures != 0:
        reasons.append(f"tiled_admission_failures_observed:{admission_failures}")
    if expected_route is None:
        return reasons

    accepted_routes = (
        frozenset((expected_route,))
        if allowed_routes is None
        else allowed_routes
    )
    for route, count in counts.items():
        if route == expected_route:
            if count <= 0:
                reasons.append(f"expected_route_not_captured:{route}")
        elif route not in accepted_routes and count != 0:
            reasons.append(f"unexpected_route_observed:{route}:{count}")
    last_route = proof.get("last_route")
    if last_route not in accepted_routes:
        reasons.append(
            "last_route_mismatch:"
            f"actual={last_route!r}:allowed={sorted(accepted_routes)!r}"
        )
    if dispatch_reason == "none":
        reasons.append("selector_dispatch_reason_not_captured")
    if event_count <= 0:
        reasons.append("selector_admission_event_not_captured")
    if admission_chain == "0" * 64:
        reasons.append("selector_admission_chain_not_captured")

    if expected_route in SELECTOR_LOG_F_ROUTES and admission_identity == "none":
        reasons.append("selector_admission_identity_not_captured")

    if expected_route == SELECTOR_LOG_F_TILED_ROUTE:
        if tiled_cohort_count + tiled_direct_count <= 0:
            reasons.append("tiled_launch_not_captured")
        if tiled_kernel_launch_count != (tiled_cohort_count + tiled_direct_count) * 4:
            reasons.append(
                "tiled_kernel_count_mismatch:"
                f"cohorts={tiled_cohort_count}:direct={tiled_direct_count}:"
                f"kernels={tiled_kernel_launch_count}"
            )
        if counts[expected_route] != tiled_job_count + tiled_direct_count:
            reasons.append(
                "tiled_route_job_count_mismatch:"
                f"route={counts[expected_route]}:jobs={tiled_job_count}:"
                f"direct={tiled_direct_count}"
            )
    elif (
        tiled_cohort_count != 0
        or tiled_job_count != 0
        or tiled_direct_count != 0
        or tiled_kernel_launch_count != 0
    ):
        reasons.append(
            "unexpected_tiled_counts:"
            f"cohorts={tiled_cohort_count}:jobs={tiled_job_count}:"
            f"direct={tiled_direct_count}:"
            f"kernels={tiled_kernel_launch_count}"
        )
    return reasons


def selector_log_s_artifact_identity(proof: dict[str, Any]) -> tuple[object, ...]:
    return (
        proof.get("extension_name"),
        proof.get("module_name"),
        proof.get("module_path"),
        proof.get("module_sha256"),
        proof.get("module_size_bytes"),
        proof.get("required_symbol"),
        proof.get("semantic_identity_symbol"),
        proof.get("semantic_identity"),
        proof.get("module_signature"),
    )


__all__ = [
    "SELECTOR_LOG_F_AMORTIZED_TILED_ALLOWED_ROUTES",
    "SELECTOR_LOG_F_COUNTER_SEMANTICS",
    "SELECTOR_LOG_F_FAST_ROUTE",
    "SELECTOR_LOG_F_GENERIC_ROUTE",
    "SELECTOR_LOG_F_ROUTES",
    "SELECTOR_LOG_F_TILED_ROUTE",
    "SELECTOR_LOG_S_EXTENSION_NAME",
    "SELECTOR_LOG_S_REQUIRED_TILED_SYMBOL",
    "SELECTOR_LOG_S_RUNTIME_PROOF_SCHEMA",
    "SELECTOR_LOG_S_SEMANTIC_IDENTITY",
    "SELECTOR_LOG_S_SEMANTIC_IDENTITY_SYMBOL",
    "SelectorLogFRoute",
    "selector_log_s_artifact_identity",
    "selector_log_s_module_signature",
    "selector_log_s_runtime_proof_reasons",
]
