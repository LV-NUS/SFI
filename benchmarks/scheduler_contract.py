"""Shared scheduler/graph contract for benchmark parents and children."""

from __future__ import annotations

from argparse import Namespace
from collections.abc import Callable
import os
from typing import Any


SCHEDULER_GRAPH_CONTRACT_SCHEMA = "sfi.scheduler_graph_contract.v1"
DEFAULT_MAX_NUM_BATCHED_TOKENS = 8192
DEFAULT_MAX_SEQ_LEN_TO_CAPTURE = 32768
CHUNKED_PREFILL_MODES = ("auto", "enabled", "disabled")

SCHEDULER_GRAPH_RUNTIME_FIELDS = (
    "scheduler_graph_contract_schema",
    "engine_max_num_seqs_requested",
    "engine_max_num_seqs_effective",
    "engine_max_num_batched_tokens_requested",
    "engine_max_num_batched_tokens_effective",
    "engine_kv_cache_memory_bytes_requested",
    "engine_kv_cache_memory_bytes_effective",
    "engine_chunked_prefill_requested",
    "engine_chunked_prefill_configured",
    "engine_chunked_prefill_effective",
    "engine_cudagraph_capture_sizes_requested",
    "engine_cudagraph_capture_sizes_effective",
    "engine_max_cudagraph_capture_size_effective",
    "engine_decode_batch_cudagraph_covered",
    "engine_max_seq_len_to_capture_requested",
    "engine_max_seq_len_to_capture_supported",
    "engine_max_seq_len_to_capture_effective",
    "engine_sequence_length_graph_control",
    "engine_full_cuda_graph_requested",
    "engine_full_cuda_graph_effective",
)


def resolve_benchmark_max_num_seqs(
    *,
    batch_size: int,
    configured: int = 0,
) -> int:
    """Resolve 0 to batch size and reject caps that serialize the workload."""
    batch = int(batch_size)
    requested = int(configured)
    if batch <= 0:
        raise ValueError("--batch-size must be > 0")
    if requested < 0:
        raise ValueError("--max-num-seqs must be >= 0")
    if 0 < requested < batch:
        raise ValueError("--max-num-seqs must be 0 or >= --batch-size")
    return requested if requested > 0 else batch


def _requested_kv_cache_memory_bytes(args: Namespace) -> int | None:
    configured = int(getattr(args, "kv_cache_memory_bytes", 0) or 0)
    if configured < 0:
        raise ValueError("--kv-cache-memory-bytes must be >= 0")
    env_raw = str(os.environ.get("VLLM_KV_CACHE_MEMORY_BYTES", "") or "")
    if env_raw and (not env_raw.isdigit() or int(env_raw) <= 0):
        raise ValueError("VLLM_KV_CACHE_MEMORY_BYTES must be a positive integer")
    env_value = int(env_raw) if env_raw else 0
    if configured and env_value and configured != env_value:
        raise ValueError(
            "--kv-cache-memory-bytes must match VLLM_KV_CACHE_MEMORY_BYTES"
        )
    value = configured or env_value
    return value if value > 0 else None


def validate_scheduler_graph_args(args: Namespace) -> None:
    """Validate the explicit parent/child scheduler contract."""
    resolve_benchmark_max_num_seqs(
        batch_size=int(args.batch_size),
        configured=int(getattr(args, "max_num_seqs", 0) or 0),
    )
    max_num_batched_tokens = int(
        getattr(args, "max_num_batched_tokens", 0) or 0
    )
    if max_num_batched_tokens < 0:
        raise ValueError("--max-num-batched-tokens must be >= 0")
    if 0 < max_num_batched_tokens < int(args.batch_size):
        raise ValueError(
            "--max-num-batched-tokens must be 0 or >= --batch-size"
        )
    max_seq_len_to_capture = int(
        getattr(
            args,
            "max_seq_len_to_capture",
            DEFAULT_MAX_SEQ_LEN_TO_CAPTURE,
        )
        or 0
    )
    if max_seq_len_to_capture <= 0:
        raise ValueError("--max-seq-len-to-capture must be > 0")
    chunked_prefill = str(
        getattr(args, "chunked_prefill", "auto") or "auto"
    )
    if chunked_prefill not in CHUNKED_PREFILL_MODES:
        raise ValueError(
            "--chunked-prefill must be auto, enabled, or disabled"
        )
    _requested_kv_cache_memory_bytes(args)


def requested_scheduler_graph_contract(args: Namespace) -> dict[str, object]:
    """Return the single contract forwarded to every child in a paired run."""
    validate_scheduler_graph_args(args)
    max_num_batched_tokens = int(
        getattr(args, "max_num_batched_tokens", 0) or 0
    )
    capture_sizes_raw = str(
        getattr(args, "cudagraph_capture_sizes", "") or ""
    )
    if capture_sizes_raw:
        capture_sizes = sorted(
            {
                int(value.strip())
                for value in capture_sizes_raw.split(",")
                if value.strip()
            }
        )
    elif bool(getattr(args, "full_cuda_graph", False)):
        capture_sizes = [int(args.batch_size)]
    else:
        capture_sizes = []
    if any(value <= 0 for value in capture_sizes):
        raise ValueError("--cudagraph-capture-sizes must contain positive integers")
    return {
        "schema": SCHEDULER_GRAPH_CONTRACT_SCHEMA,
        "max_num_seqs": resolve_benchmark_max_num_seqs(
            batch_size=int(args.batch_size),
            configured=int(getattr(args, "max_num_seqs", 0) or 0),
        ),
        "max_num_batched_tokens": (
            max_num_batched_tokens if max_num_batched_tokens > 0 else None
        ),
        "kv_cache_memory_bytes": _requested_kv_cache_memory_bytes(args),
        "chunked_prefill": str(
            getattr(args, "chunked_prefill", "auto") or "auto"
        ),
        "cudagraph_capture_sizes": capture_sizes,
        "max_seq_len_to_capture": int(
            getattr(
                args,
                "max_seq_len_to_capture",
                DEFAULT_MAX_SEQ_LEN_TO_CAPTURE,
            )
        ),
        "full_cuda_graph": bool(getattr(args, "full_cuda_graph", False)),
    }


def scheduler_graph_engine_kwargs(
    args: Namespace,
    *,
    engine_args_accepts: Callable[[str], bool],
) -> dict[str, object]:
    """Translate the shared contract to supported vLLM EngineArgs.

    ``max_seq_len_to_capture`` was removed from newer vLLM V1 releases.  It is
    still part of the cross-version benchmark identity, but is forwarded only
    when EngineArgs owns it; the runtime proof records that support state.
    """
    contract = requested_scheduler_graph_contract(args)
    kwargs: dict[str, object] = {"max_num_seqs": contract["max_num_seqs"]}
    max_num_batched_tokens = contract["max_num_batched_tokens"]
    if max_num_batched_tokens is not None:
        if not engine_args_accepts("max_num_batched_tokens"):
            raise RuntimeError(
                "E_SCHEDULER_GRAPH_UNSUPPORTED: explicit "
                "max_num_batched_tokens requires EngineArgs support"
            )
        kwargs["max_num_batched_tokens"] = max_num_batched_tokens
    kv_cache_memory_bytes = contract["kv_cache_memory_bytes"]
    if kv_cache_memory_bytes is not None:
        if not engine_args_accepts("kv_cache_memory_bytes"):
            raise RuntimeError(
                "E_SCHEDULER_GRAPH_UNSUPPORTED: explicit "
                "kv_cache_memory_bytes requires EngineArgs support"
            )
        kwargs["kv_cache_memory_bytes"] = kv_cache_memory_bytes
    chunked_prefill = str(contract["chunked_prefill"])
    if chunked_prefill != "auto":
        if not engine_args_accepts("enable_chunked_prefill"):
            raise RuntimeError(
                "E_SCHEDULER_GRAPH_UNSUPPORTED: explicit chunked prefill "
                "requires EngineArgs support"
            )
        kwargs["enable_chunked_prefill"] = chunked_prefill == "enabled"
    if (
        hasattr(args, "max_seq_len_to_capture")
        and engine_args_accepts("max_seq_len_to_capture")
    ):
        kwargs["max_seq_len_to_capture"] = contract["max_seq_len_to_capture"]
    return kwargs


def _full_cudagraph_effective(compilation_config: object) -> bool:
    mode = getattr(compilation_config, "cudagraph_mode", None)
    has_mode = getattr(mode, "has_full_cudagraphs", None)
    if callable(has_mode):
        try:
            return bool(has_mode())
        except Exception as exc:
            raise RuntimeError(
                "E_SCHEDULER_GRAPH_PROOF_UNAVAILABLE: "
                "cudagraph_mode.has_full_cudagraphs"
            ) from exc
    if mode is not None:
        return str(mode).strip().upper() in {
            "FULL",
            "FULL_AND_PIECEWISE",
            "FULL_DECODE_ONLY",
        }
    return bool(getattr(compilation_config, "full_cuda_graph", False))


def scheduler_graph_runtime_proof(
    engine: object,
    args: Namespace,
    *,
    engine_args_accepts: Callable[[str], bool],
) -> dict[str, object]:
    """Read back instantiated scheduler/graph state and reject drift."""
    contract = requested_scheduler_graph_contract(args)
    vllm_config = getattr(getattr(engine, "llm_engine", None), "vllm_config", None)
    scheduler_config = getattr(vllm_config, "scheduler_config", None)
    if scheduler_config is None:
        raise RuntimeError("E_SCHEDULER_GRAPH_PROOF_UNAVAILABLE: scheduler_config")

    effective_max_num_seqs = getattr(scheduler_config, "max_num_seqs", None)
    effective_max_num_batched_tokens = getattr(
        scheduler_config, "max_num_batched_tokens", None
    )
    effective_chunked_prefill = getattr(
        scheduler_config, "enable_chunked_prefill", None
    )
    configured_chunked_prefill = (
        None
        if contract["chunked_prefill"] == "auto"
        else contract["chunked_prefill"] == "enabled"
    )
    expected_max_num_batched_tokens = contract["max_num_batched_tokens"]
    expected_kv_cache_memory_bytes = contract["kv_cache_memory_bytes"]
    cache_config = getattr(vllm_config, "cache_config", None)
    effective_kv_cache_memory_bytes = getattr(
        cache_config, "kv_cache_memory_bytes", None
    )

    if (
        type(effective_max_num_seqs) is not int
        or effective_max_num_seqs != contract["max_num_seqs"]
    ):
        raise RuntimeError(
            "E_SCHEDULER_GRAPH_EFFECTIVE_MISMATCH: "
            f"max_num_seqs={effective_max_num_seqs!r}:"
            f"expected={contract['max_num_seqs']!r}"
        )
    if (
        expected_max_num_batched_tokens is not None
        and (
            type(effective_max_num_batched_tokens) is not int
            or effective_max_num_batched_tokens
            != expected_max_num_batched_tokens
        )
    ):
        raise RuntimeError(
            "E_SCHEDULER_GRAPH_EFFECTIVE_MISMATCH: "
            f"max_num_batched_tokens={effective_max_num_batched_tokens!r}:"
            f"expected={expected_max_num_batched_tokens!r}"
        )
    if (
        expected_kv_cache_memory_bytes is not None
        and (
            type(effective_kv_cache_memory_bytes) is not int
            or effective_kv_cache_memory_bytes
            != expected_kv_cache_memory_bytes
        )
    ):
        raise RuntimeError(
            "E_SCHEDULER_GRAPH_EFFECTIVE_MISMATCH: "
            f"kv_cache_memory_bytes={effective_kv_cache_memory_bytes!r}:"
            f"expected={expected_kv_cache_memory_bytes!r}"
        )
    if type(effective_chunked_prefill) is not bool:
        raise RuntimeError(
            "E_SCHEDULER_GRAPH_PROOF_UNAVAILABLE: "
            f"enable_chunked_prefill={effective_chunked_prefill!r}"
        )
    if (
        configured_chunked_prefill is not None
        and effective_chunked_prefill is not configured_chunked_prefill
    ):
        raise RuntimeError(
            "E_SCHEDULER_GRAPH_EFFECTIVE_MISMATCH: "
            f"enable_chunked_prefill={effective_chunked_prefill!r}:"
            f"expected={configured_chunked_prefill!r}"
        )

    max_seq_supported = bool(engine_args_accepts("max_seq_len_to_capture"))
    max_seq_effective: int | None = None
    if max_seq_supported:
        model_config = getattr(vllm_config, "model_config", None)
        raw_max_seq = getattr(model_config, "max_seq_len_to_capture", None)
        if type(raw_max_seq) is not int:
            raise RuntimeError(
                "E_SCHEDULER_GRAPH_PROOF_UNAVAILABLE: "
                f"max_seq_len_to_capture={raw_max_seq!r}"
            )
        max_seq_effective = raw_max_seq
        if max_seq_effective != contract["max_seq_len_to_capture"]:
            raise RuntimeError(
                "E_SCHEDULER_GRAPH_EFFECTIVE_MISMATCH: "
                f"max_seq_len_to_capture={max_seq_effective!r}:"
                f"expected={contract['max_seq_len_to_capture']!r}"
            )

    full_cuda_graph_effective = _full_cudagraph_effective(
        getattr(vllm_config, "compilation_config", None)
    )
    if full_cuda_graph_effective is not bool(contract["full_cuda_graph"]):
        raise RuntimeError(
            "E_SCHEDULER_GRAPH_EFFECTIVE_MISMATCH: "
            f"full_cuda_graph={full_cuda_graph_effective!r}:"
            f"expected={contract['full_cuda_graph']!r}"
        )

    compilation_config = getattr(vllm_config, "compilation_config", None)
    raw_capture_sizes = getattr(
        compilation_config, "cudagraph_capture_sizes", None
    )
    if raw_capture_sizes is None and not contract["full_cuda_graph"]:
        raw_capture_sizes = []
    if not isinstance(raw_capture_sizes, (list, tuple)) or any(
        type(value) is not int or value <= 0 for value in raw_capture_sizes
    ):
        raise RuntimeError(
            "E_SCHEDULER_GRAPH_PROOF_UNAVAILABLE: "
            f"cudagraph_capture_sizes={raw_capture_sizes!r}"
        )
    effective_capture_sizes = sorted(set(raw_capture_sizes))
    requested_capture_sizes = list(contract["cudagraph_capture_sizes"])
    if (
        contract["full_cuda_graph"]
        and effective_capture_sizes != requested_capture_sizes
    ):
        raise RuntimeError(
            "E_SCHEDULER_GRAPH_EFFECTIVE_MISMATCH: "
            f"cudagraph_capture_sizes={effective_capture_sizes!r}:"
            f"expected={requested_capture_sizes!r}"
        )
    effective_max_capture_size = getattr(
        compilation_config, "max_cudagraph_capture_size", None
    )
    expected_max_capture_size = (
        max(requested_capture_sizes) if requested_capture_sizes else 0
    )
    if effective_max_capture_size is None and not contract["full_cuda_graph"]:
        effective_max_capture_size = 0
    if (
        type(effective_max_capture_size) is not int
        or (
            contract["full_cuda_graph"]
            and effective_max_capture_size != expected_max_capture_size
        )
    ):
        raise RuntimeError(
            "E_SCHEDULER_GRAPH_EFFECTIVE_MISMATCH: "
            f"max_cudagraph_capture_size={effective_max_capture_size!r}:"
            f"expected={expected_max_capture_size!r}"
        )
    decode_batch_covered = bool(
        contract["max_num_seqs"] in effective_capture_sizes
        and effective_max_capture_size >= contract["max_num_seqs"]
    )
    if contract["full_cuda_graph"] and not decode_batch_covered:
        raise RuntimeError(
            "E_SCHEDULER_GRAPH_EFFECTIVE_MISMATCH: decode batch is not "
            "covered by cudagraph capture sizes"
        )

    return {
        "scheduler_graph_contract_schema": SCHEDULER_GRAPH_CONTRACT_SCHEMA,
        "engine_max_num_seqs_requested": contract["max_num_seqs"],
        "engine_max_num_seqs_effective": effective_max_num_seqs,
        "engine_max_num_batched_tokens_requested": (
            expected_max_num_batched_tokens
        ),
        "engine_max_num_batched_tokens_effective": (
            effective_max_num_batched_tokens
        ),
        "engine_kv_cache_memory_bytes_requested": (
            expected_kv_cache_memory_bytes
        ),
        "engine_kv_cache_memory_bytes_effective": (
            effective_kv_cache_memory_bytes
        ),
        "engine_chunked_prefill_requested": contract["chunked_prefill"],
        "engine_chunked_prefill_configured": configured_chunked_prefill,
        "engine_chunked_prefill_effective": effective_chunked_prefill,
        "engine_cudagraph_capture_sizes_requested": requested_capture_sizes,
        "engine_cudagraph_capture_sizes_effective": effective_capture_sizes,
        "engine_max_cudagraph_capture_size_effective": (
            effective_max_capture_size
        ),
        "engine_decode_batch_cudagraph_covered": decode_batch_covered,
        "engine_max_seq_len_to_capture_requested": contract[
            "max_seq_len_to_capture"
        ],
        "engine_max_seq_len_to_capture_supported": max_seq_supported,
        "engine_max_seq_len_to_capture_effective": max_seq_effective,
        "engine_sequence_length_graph_control": (
            "legacy_engine_arg"
            if max_seq_supported
            else "not_applicable_v1"
        ),
        "engine_full_cuda_graph_requested": contract["full_cuda_graph"],
        "engine_full_cuda_graph_effective": full_cuda_graph_effective,
    }


def scheduler_graph_runtime_contract_from_metrics(
    metrics: dict[str, Any],
) -> dict[str, object]:
    """Extract only the stable instantiated contract from a child artifact."""
    run_config = metrics.get("run_config")
    source = run_config if isinstance(run_config, dict) else {}
    return {field: source.get(field) for field in SCHEDULER_GRAPH_RUNTIME_FIELDS}
