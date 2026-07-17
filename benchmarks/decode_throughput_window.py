from __future__ import annotations

import atexit
import hashlib
import json
import os
import sys
import time
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Tuple


CUSTOM_ALL_REDUCE_RUNTIME_RPC_METHOD = "sfi_custom_all_reduce_runtime_state"
ENGINE_RUNTIME_CONTRACT_RPC_METHOD = "sfi_engine_runtime_contract_state"
SPARSE_ROUTE_COUNTER_RESET_RPC_METHOD = (
    "sfi_reset_sparse_route_counters_for_measurement"
)
SPARSE_ROUTE_COUNTER_SNAPSHOT_RPC_METHOD = (
    "sfi_snapshot_sparse_route_counters_after_measurement"
)
CUSTOM_ALL_REDUCE_RUNTIME_WORKER_EXTENSION = (
    "benchmarks.decode_throughput_window.CustomAllReduceRuntimeWorkerExtension"
)


@dataclass(frozen=True)
class CustomAllReduceDecision:
    """The effective vLLM custom-all-reduce policy for one benchmark arm."""

    requested: str
    effective: str
    reason: str

    def as_dict(self) -> dict[str, str]:
        return {
            "custom_all_reduce_requested": self.requested,
            "custom_all_reduce_effective": self.effective,
            "custom_all_reduce_effective_reason": self.reason,
        }


def _cuda_device_runtime_state() -> dict[str, object]:
    """Return the worker-local CUDA identity used by cold runtime proofs."""
    import torch

    device_index = int(torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(device_index)
    capability = torch.cuda.get_device_capability(device_index)
    return {
        "current_device": device_index,
        "capability": f"{int(capability[0])}.{int(capability[1])}",
        "name": str(getattr(properties, "name", "")),
        "total_memory_bytes": int(getattr(properties, "total_memory", 0)),
        "uuid": str(getattr(properties, "uuid", "") or ""),
    }


def _worker_custom_all_reduce_runtime_state(
    worker: object,
    required_num_tokens: int,
) -> dict[str, object]:
    """Return one JSON-serializable CustomAllreduce fact record per worker.

    The worker extension below calls this helper locally.  The controller uses
    a named RPC method, so vLLM's secure transport never has to serialize a
    Python function or enable its pickle fallback.
    """
    import vllm.envs as vllm_envs
    from vllm.distributed.parallel_state import get_tp_group

    if (
        isinstance(required_num_tokens, bool)
        or not isinstance(required_num_tokens, int)
        or required_num_tokens <= 0
    ):
        raise RuntimeError(
            "E_CUSTOM_ALL_REDUCE_REQUIRED_NUM_TOKENS: "
            f"required_num_tokens={required_num_tokens!r}"
        )

    tp_group = get_tp_group()
    device_communicator = getattr(tp_group, "device_communicator", None)
    ca_comm = getattr(device_communicator, "ca_comm", None)
    fi_ar_comm = getattr(device_communicator, "fi_ar_comm", None)
    symm_mem_comm = getattr(device_communicator, "symm_mem_comm", None)
    parallel_config = getattr(
        getattr(worker, "vllm_config", None),
        "parallel_config",
        None,
    )
    config_disabled_raw = getattr(
        parallel_config,
        "disable_custom_all_reduce",
        None,
    )
    communicator_enabled_raw = getattr(
        device_communicator,
        "use_custom_allreduce",
        None,
    )
    ca_disabled_raw = getattr(ca_comm, "disabled", None)

    config_disabled = (
        bool(config_disabled_raw) if config_disabled_raw is not None else None
    )
    communicator_enabled = (
        bool(communicator_enabled_raw)
        if communicator_enabled_raw is not None
        else None
    )
    ca_disabled = bool(ca_disabled_raw) if ca_disabled_raw is not None else None
    ca_active = bool(ca_comm is not None and ca_disabled is False)
    fi_ar_disabled_raw = getattr(fi_ar_comm, "disabled", None)
    fi_ar_disabled = (
        bool(fi_ar_disabled_raw) if fi_ar_disabled_raw is not None else None
    )
    fi_ar_active = bool(fi_ar_comm is not None and fi_ar_disabled is False)
    symm_mem_disabled_raw = getattr(symm_mem_comm, "disabled", None)
    symm_mem_disabled = (
        bool(symm_mem_disabled_raw)
        if symm_mem_disabled_raw is not None
        else None
    )
    symm_mem_active = bool(
        symm_mem_comm is not None and symm_mem_disabled is not True
    )

    if ca_active:
        reason = "active"
    elif config_disabled is True:
        reason = "parallel_config_disabled"
    elif device_communicator is None:
        reason = "device_communicator_missing"
    elif communicator_enabled is False:
        reason = "device_communicator_policy_disabled"
    elif ca_comm is None:
        reason = "custom_all_reduce_not_instantiated"
    else:
        reason = "custom_all_reduce_runtime_disabled"

    fully_connected_raw = getattr(ca_comm, "fully_connected", None)
    max_size_raw = getattr(ca_comm, "max_size", None)
    ca_max_size = int(max_size_raw) if max_size_raw is not None else None

    model_runner = getattr(worker, "model_runner", None)
    model_config = getattr(model_runner, "model_config", None)
    if model_config is None:
        model_config = getattr(
            getattr(worker, "vllm_config", None),
            "model_config",
            None,
        )

    model_hidden_size: int | None = None
    model_dtype = ""
    model_dtype_bytes: int | None = None
    required_payload_numel: int | None = None
    required_payload_bytes: int | None = None
    payload_spec_error = ""
    torch_dtype: object | None = None
    try:
        import torch

        get_hidden_size = getattr(model_config, "get_hidden_size", None)
        hidden_size_raw = (
            get_hidden_size()
            if callable(get_hidden_size)
            else getattr(model_config, "hidden_size", None)
        )
        if isinstance(hidden_size_raw, bool):
            raise TypeError(f"hidden_size={hidden_size_raw!r}")
        model_hidden_size = int(hidden_size_raw)
        if model_hidden_size <= 0:
            raise ValueError(f"hidden_size={model_hidden_size}")

        torch_dtype = getattr(model_runner, "dtype", None)
        if torch_dtype is None:
            torch_dtype = getattr(model_config, "dtype", None)
        if isinstance(torch_dtype, str):
            torch_dtype = getattr(
                torch,
                torch_dtype.removeprefix("torch."),
                None,
            )
        if not isinstance(torch_dtype, torch.dtype):
            raise TypeError(f"dtype={torch_dtype!r}")
        model_dtype = str(torch_dtype)
        model_dtype_bytes = int(
            torch.empty((), dtype=torch_dtype, device="meta").element_size()
        )
        required_payload_numel = int(required_num_tokens) * model_hidden_size
        required_payload_bytes = required_payload_numel * model_dtype_bytes
    except Exception as exc:
        payload_spec_error = f"{type(exc).__name__}: {exc}"

    capacity_covers_required_payload: bool | None = None
    dispatch_size_eligible: bool | None = None
    if ca_max_size is not None and required_payload_bytes is not None:
        capacity_covers_required_payload = bool(
            ca_max_size >= required_payload_bytes
        )
        # vLLM's CustomAllreduce.should_custom_ar uses a strict '<' check.
        # Equality means the representative payload falls back to PyNCCL.
        dispatch_size_eligible = bool(
            required_payload_bytes < ca_max_size
        )

    should_custom_ar = getattr(ca_comm, "should_custom_ar", None)
    synthetic_should_custom_ar: bool | None = None
    synthetic_should_custom_ar_error = ""
    if (
        ca_active
        and callable(should_custom_ar)
        and torch_dtype is not None
        and model_hidden_size is not None
        and capacity_covers_required_payload is True
    ):
        try:
            import torch

            representative_payload = torch.empty(
                (int(required_num_tokens), model_hidden_size),
                dtype=torch_dtype,
                device=getattr(tp_group, "device"),
            )
            synthetic_should_custom_ar = bool(
                should_custom_ar(representative_payload)
            )
        except Exception as exc:
            synthetic_should_custom_ar_error = (
                f"{type(exc).__name__}: {exc}"
            )

    return {
        "global_rank": int(getattr(tp_group, "rank")),
        "tp_rank": int(getattr(tp_group, "rank_in_group")),
        "tp_world_size": int(getattr(tp_group, "world_size")),
        "local_rank": int(getattr(tp_group, "local_rank")),
        "device": str(getattr(tp_group, "device", "")),
        "cuda_device_runtime": _cuda_device_runtime_state(),
        "worker_class": type(worker).__name__,
        "parallel_config_disable_custom_all_reduce": config_disabled,
        "device_communicator_class": (
            type(device_communicator).__name__
            if device_communicator is not None
            else ""
        ),
        "device_communicator_use_custom_allreduce": communicator_enabled,
        "device_communicator_use_flashinfer_allreduce": bool(
            getattr(device_communicator, "use_flashinfer_allreduce", False)
        ),
        "fi_ar_comm_present": bool(fi_ar_comm is not None),
        "fi_ar_comm_disabled": fi_ar_disabled,
        "fi_ar_comm_active": fi_ar_active,
        "device_communicator_use_torch_symm_mem": bool(
            getattr(device_communicator, "use_torch_symm_mem", False)
        ),
        "symm_mem_comm_present": bool(symm_mem_comm is not None),
        "symm_mem_comm_disabled": symm_mem_disabled,
        "symm_mem_comm_active": symm_mem_active,
        "vllm_allreduce_use_flashinfer": bool(
            getattr(vllm_envs, "VLLM_ALLREDUCE_USE_FLASHINFER", False)
        ),
        "vllm_allreduce_use_symm_mem": bool(
            getattr(vllm_envs, "VLLM_ALLREDUCE_USE_SYMM_MEM", False)
        ),
        "vllm_use_nccl_symm_mem": bool(
            getattr(vllm_envs, "VLLM_USE_NCCL_SYMM_MEM", False)
        ),
        "ca_comm_present": bool(ca_comm is not None),
        "ca_comm_class": type(ca_comm).__name__ if ca_comm is not None else "",
        "ca_comm_disabled": ca_disabled,
        "ca_comm_active": ca_active,
        "ca_comm_fully_connected": (
            bool(fully_connected_raw) if fully_connected_raw is not None else None
        ),
        "ca_comm_max_size": ca_max_size,
        "required_num_tokens": int(required_num_tokens),
        "model_hidden_size": model_hidden_size,
        "model_dtype": model_dtype,
        "model_dtype_bytes": model_dtype_bytes,
        "required_payload_numel": required_payload_numel,
        "required_payload_bytes": required_payload_bytes,
        "required_payload_spec_error": payload_spec_error,
        "ca_comm_capacity_covers_required_payload": (
            capacity_covers_required_payload
        ),
        "ca_comm_dispatch_size_eligible": dispatch_size_eligible,
        "ca_comm_should_custom_ar_callable": callable(should_custom_ar),
        "ca_comm_synthetic_should_custom_ar": synthetic_should_custom_ar,
        "ca_comm_synthetic_should_custom_ar_error": (
            synthetic_should_custom_ar_error
        ),
        "reason": reason,
    }


def _attach_tp8_selector_log_s_runtime_proof(
    record: dict[str, object],
) -> dict[str, object]:
    if os.environ.get("SFI_RUNNER_TIER", "") != "tp8x64k":
        return record
    from patches.fa3_native.postprocess import (
        snapshot_selector_log_s_runtime_proof,
    )

    record["selector_log_s_runtime_proof"] = (
        snapshot_selector_log_s_runtime_proof()
    )
    return record


def _runtime_mode_name(value: object) -> str:
    """Normalize enum/string runtime modes without depending on vLLM internals."""
    name = getattr(value, "name", None)
    if isinstance(name, str) and name:
        return name.upper()
    return str(value).rsplit(".", 1)[-1].strip().upper()


def benchmark_child_identity(args: object) -> dict[str, object]:
    """Emit the identity observed inside each timed worker child.

    Pairing must compare independently emitted child facts. Copying parent
    provenance into both arms cannot detect a drifted dense command or env.
    """

    def _positive_env_int(name: str) -> int:
        raw = str(os.environ.get(name, "") or "")
        return int(raw) if raw.isdigit() and int(raw) > 0 else 0

    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    model_path = Path(str(getattr(args, "model", "") or "")).resolve()
    model_config_path = (model_path / "config.json").resolve()
    prompt_path = Path(str(getattr(args, "prompt", "") or "")).resolve()
    model_config_sha256 = (
        _sha256(model_config_path) if model_config_path.is_file() else ""
    )
    corpus_sha256 = _sha256(prompt_path) if prompt_path.is_file() else ""
    expected_model_sha256 = str(
        os.environ.get("SFI_RUNNER_EXPECTED_MODEL_CONFIG_SHA256", "") or ""
    )
    parent_model_sha256 = str(
        os.environ.get("SFI_RUNNER_MODEL_CONFIG_SHA256", "") or ""
    )
    expected_corpus_sha256 = str(
        os.environ.get("SFI_RUNNER_CORPUS_SHA256", "") or ""
    )
    if os.environ.get("SFI_RUNNER_TIER", "") == "tp8x64k":
        mismatches = []
        if model_config_sha256 != expected_model_sha256:
            mismatches.append(
                "model_config_sha256="
                f"{model_config_sha256!r}:expected={expected_model_sha256!r}"
            )
        if model_config_sha256 != parent_model_sha256:
            mismatches.append(
                "model_config_parent_claim="
                f"{parent_model_sha256!r}:actual={model_config_sha256!r}"
            )
        if corpus_sha256 != expected_corpus_sha256:
            mismatches.append(
                "corpus_sha256="
                f"{corpus_sha256!r}:expected={expected_corpus_sha256!r}"
            )
        if mismatches:
            raise RuntimeError(
                "E_BENCHMARK_CHILD_IDENTITY_MISMATCH: "
                + "; ".join(mismatches)
            )

    return {
        "schema": "sfi.benchmark_child_identity.v1",
        "model_path": str(model_path),
        "model_config_path": str(model_config_path),
        "model_config_sha256": model_config_sha256,
        "parent_model_config_sha256": parent_model_sha256,
        "expected_model_config_sha256": expected_model_sha256,
        "prompt_path": str(prompt_path),
        "corpus_sha256": corpus_sha256,
        "expected_corpus_sha256": expected_corpus_sha256,
        "tensor_parallel_size": int(
            getattr(args, "tensor_parallel_size", 1) or 1
        ),
        "cuda_visible_devices": str(
            os.environ.get("CUDA_VISIBLE_DEVICES", "") or ""
        ),
        "cuda_capabilities": str(
            os.environ.get("SFI_RUNNER_CUDA_CAPABILITIES", "") or ""
        ),
        "batch_size": int(getattr(args, "batch_size", 0) or 0),
        "context_tokens": _positive_env_int("SFI_RUNNER_CONTEXT_TOKENS"),
        "max_new_tokens": int(getattr(args, "max_new_tokens", 0) or 0),
        "max_model_len": int(getattr(args, "max_model_len", 0) or 0),
        "runner_tier": str(os.environ.get("SFI_RUNNER_TIER", "") or ""),
        "expected_git_commit": str(
            os.environ.get("SFI_RUNNER_EXPECTED_GIT_COMMIT", "") or ""
        ),
        "gpu_lock_mode": str(
            os.environ.get("SFI_RUNNER_GPU_LOCK_MODE", "") or ""
        ),
        "gpu_lock_scope": str(
            os.environ.get("SFI_RUNNER_GPU_LOCK_SCOPE", "") or ""
        ),
        "attention_backend": str(
            os.environ.get("VLLM_ATTENTION_BACKEND", "") or ""
        ),
        "flash_attn_version": str(
            os.environ.get("VLLM_FLASH_ATTN_VERSION", "") or ""
        ),
        "attention_build_identity_json": str(
            os.environ.get("SFI_RUNNER_ATTENTION_BUILD_IDENTITY_JSON", "")
            or ""
        ),
    }


def _worker_engine_runtime_contract_state(
    worker: object,
    required_batch_size: int,
    required_tokens_per_request: int,
    compact_blocks_per_slot: int,
    compact_generation_count: int,
    expected_kv_bytes_per_token: int,
) -> dict[str, object]:
    """Read the resolved worker-local graph dispatcher and allocated KV cache.

    Front-end EngineArgs are only intent.  Exact runs need the post-backend
    state after graph-key resolution and KV allocation on every TP rank.
    """
    from collections import Counter

    from vllm.distributed.parallel_state import get_tp_group

    integers = {
        "required_batch_size": required_batch_size,
        "required_tokens_per_request": required_tokens_per_request,
        "compact_blocks_per_slot": compact_blocks_per_slot,
        "compact_generation_count": compact_generation_count,
        "expected_kv_bytes_per_token": expected_kv_bytes_per_token,
    }
    for key, value in integers.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeError(f"E_ENGINE_RUNTIME_CONTRACT_INPUT: {key}={value!r}")
    if required_batch_size <= 0:
        raise RuntimeError(
            "E_ENGINE_RUNTIME_CONTRACT_INPUT: required_batch_size must be > 0"
        )

    tp_group = get_tp_group()
    cuda_device_runtime = _cuda_device_runtime_state()
    model_runner = getattr(worker, "model_runner", None)
    if model_runner is None:
        raise RuntimeError("E_ENGINE_RUNTIME_CONTRACT_UNAVAILABLE: model_runner")
    vllm_config = getattr(worker, "vllm_config", None)
    if vllm_config is None:
        vllm_config = getattr(model_runner, "vllm_config", None)
    if vllm_config is None:
        raise RuntimeError("E_ENGINE_RUNTIME_CONTRACT_UNAVAILABLE: vllm_config")

    compilation_config = getattr(model_runner, "compilation_config", None)
    if compilation_config is None:
        raise RuntimeError(
            "E_ENGINE_RUNTIME_CONTRACT_UNAVAILABLE: compilation_config"
        )
    capture_sizes_raw = getattr(
        compilation_config, "cudagraph_capture_sizes", None
    )
    if not isinstance(capture_sizes_raw, (list, tuple)):
        raise RuntimeError(
            "E_ENGINE_RUNTIME_CONTRACT_UNAVAILABLE: cudagraph_capture_sizes"
        )
    capture_sizes = sorted({int(value) for value in capture_sizes_raw})

    dispatcher = getattr(model_runner, "cudagraph_dispatcher", None)
    if dispatcher is None:
        raise RuntimeError(
            "E_ENGINE_RUNTIME_CONTRACT_UNAVAILABLE: cudagraph_dispatcher"
        )
    raw_keys = getattr(dispatcher, "cudagraph_keys", None)
    if not isinstance(raw_keys, dict):
        raise RuntimeError(
            "E_ENGINE_RUNTIME_CONTRACT_UNAVAILABLE: cudagraph_keys"
        )
    graph_keys: dict[str, list[dict[str, object]]] = {}
    for mode, descriptors in raw_keys.items():
        mode_name = _runtime_mode_name(mode)
        serialized: list[dict[str, object]] = []
        for descriptor in descriptors:
            serialized.append(
                {
                    "num_tokens": int(getattr(descriptor, "num_tokens")),
                    "num_reqs": (
                        None
                        if getattr(descriptor, "num_reqs", None) is None
                        else int(getattr(descriptor, "num_reqs"))
                    ),
                    "uniform": bool(getattr(descriptor, "uniform", False)),
                    "has_lora": bool(getattr(descriptor, "has_lora", False)),
                    "num_active_loras": int(
                        getattr(descriptor, "num_active_loras", 0) or 0
                    ),
                }
            )
        graph_keys[mode_name] = sorted(
            serialized,
            key=lambda item: (
                int(item["num_tokens"]),
                int(item["num_reqs"] or -1),
                bool(item["uniform"]),
                bool(item["has_lora"]),
                int(item["num_active_loras"]),
            ),
        )

    kv_cache_config = getattr(model_runner, "kv_cache_config", None)
    if kv_cache_config is None:
        raise RuntimeError(
            "E_ENGINE_RUNTIME_CONTRACT_UNAVAILABLE: kv_cache_config"
        )
    num_blocks = int(getattr(kv_cache_config, "num_blocks"))
    group_records: list[dict[str, object]] = []
    group_block_sizes: set[int] = set()
    group_page_sizes: list[int] = []
    group_leaf_page_sizes: list[int] = []
    for group in getattr(kv_cache_config, "kv_cache_groups", []):
        spec = getattr(group, "kv_cache_spec", None)
        if spec is None:
            raise RuntimeError(
                "E_ENGINE_RUNTIME_CONTRACT_UNAVAILABLE: kv_cache_group_spec"
            )
        block_size = int(getattr(spec, "block_size"))
        page_size_bytes = int(getattr(spec, "page_size_bytes"))
        group_block_sizes.add(block_size)
        group_page_sizes.append(page_size_bytes)
        nested_specs = getattr(spec, "kv_cache_specs", None)
        leaf_signatures: Counter[tuple[object, ...]] = Counter()
        if isinstance(nested_specs, dict):
            for leaf in nested_specs.values():
                leaf_signatures[
                    (
                        type(leaf).__name__,
                        int(getattr(leaf, "block_size")),
                        int(getattr(leaf, "page_size_bytes")),
                        int(getattr(leaf, "num_kv_heads", 0) or 0),
                        int(getattr(leaf, "head_size", 0) or 0),
                        str(getattr(leaf, "dtype", "")),
                    )
                ] += 1
        else:
            leaf_signatures[
                (
                    type(spec).__name__,
                    block_size,
                    page_size_bytes,
                    int(getattr(spec, "num_kv_heads", 0) or 0),
                    int(getattr(spec, "head_size", 0) or 0),
                    str(getattr(spec, "dtype", "")),
                )
            ] += 1
        leaf_page_size_bytes = sum(
            int(signature[2]) * int(count)
            for signature, count in leaf_signatures.items()
        )
        group_leaf_page_sizes.append(leaf_page_size_bytes)
        group_records.append(
            {
                "spec_class": type(spec).__name__,
                "block_size": block_size,
                "page_size_bytes": page_size_bytes,
                "leaf_page_size_bytes_total": leaf_page_size_bytes,
                "composite_page_size_matches_leaf_sum": (
                    page_size_bytes == leaf_page_size_bytes
                ),
                "layer_count": len(getattr(group, "layer_names", [])),
                "leaf_spec_signatures": [
                    {
                        "spec_class": signature[0],
                        "block_size": signature[1],
                        "page_size_bytes": signature[2],
                        "num_kv_heads": signature[3],
                        "head_size": signature[4],
                        "dtype": signature[5],
                        "count": count,
                    }
                    for signature, count in sorted(leaf_signatures.items())
                ],
            }
        )

    kv_cache_tensors = list(getattr(kv_cache_config, "kv_cache_tensors", []))
    tensor_signatures = Counter(
        (
            int(getattr(tensor, "size")),
            len(getattr(tensor, "shared_by", [])),
        )
        for tensor in kv_cache_tensors
    )
    allocated_bytes = sum(int(getattr(tensor, "size")) for tensor in kv_cache_tensors)
    block_size = next(iter(group_block_sizes)) if len(group_block_sizes) == 1 else 0
    allocated_token_slots = num_blocks * block_size if block_size > 0 else 0
    actual_kv_bytes_per_token = (
        allocated_bytes // allocated_token_slots
        if allocated_token_slots > 0
        and allocated_bytes % allocated_token_slots == 0
        else 0
    )
    required_workload_tokens = required_batch_size * required_tokens_per_request
    required_blocks_per_request = (
        (required_tokens_per_request + block_size - 1) // block_size
        if block_size > 0
        else 0
    )
    required_workload_blocks = required_batch_size * required_blocks_per_request
    compact_lease_blocks = (
        required_batch_size
        * compact_blocks_per_slot
        * compact_generation_count
    )
    compact_lease_tokens = compact_lease_blocks * block_size
    # vLLM reserves physical block zero as the null block.  Capacity must be
    # proven in block units because every request rounds up independently; a
    # token-sum check can otherwise admit up to (batch_size - 1) missing blocks.
    null_block_count = 1
    ordinary_blocks = max(
        num_blocks - null_block_count - compact_lease_blocks,
        0,
    )
    schedulable_tokens = ordinary_blocks * block_size
    required_total_blocks = (
        null_block_count + compact_lease_blocks + required_workload_blocks
    )
    required_total_tokens = required_total_blocks * block_size

    runner_kv_caches = list(getattr(model_runner, "kv_caches", []))
    runner_tensor_signatures = Counter()
    for tensor in runner_kv_caches:
        runner_tensor_signatures[
            (
                tuple(int(value) for value in getattr(tensor, "shape", ())),
                str(getattr(tensor, "dtype", "")),
                int(getattr(tensor, "element_size")()),
                int(getattr(tensor, "numel")()),
            )
        ] += 1
    cross_layers_tensor = getattr(model_runner, "cross_layers_kv_cache", None)
    cross_layers_record: dict[str, object] | None = None
    if cross_layers_tensor is not None:
        cross_layers_record = {
            "shape": [int(value) for value in cross_layers_tensor.shape],
            "dtype": str(cross_layers_tensor.dtype),
            "element_size": int(cross_layers_tensor.element_size()),
            "numel": int(cross_layers_tensor.numel()),
            "nbytes": int(
                cross_layers_tensor.numel() * cross_layers_tensor.element_size()
            ),
        }

    kv_physical_geometry = {
        "num_blocks": num_blocks,
        "group_count": len(group_records),
        "groups": group_records,
        "composite_page_size_bytes_total": sum(group_page_sizes),
        "leaf_page_size_bytes_total": sum(group_leaf_page_sizes),
        "composite_page_sizes_match_leaf_sums": all(
            record["composite_page_size_matches_leaf_sum"]
            for record in group_records
        ),
        "uniform_block_size": block_size,
        "tensor_count": len(kv_cache_tensors),
        "tensor_signatures": [
            {"size": key[0], "shared_by_count": key[1], "count": count}
            for key, count in sorted(tensor_signatures.items())
        ],
        "allocated_bytes": allocated_bytes,
        "allocated_token_slots": allocated_token_slots,
        "runner_tensor_count": len(runner_kv_caches),
        "runner_tensor_signatures": [
            {
                "shape": list(key[0]),
                "dtype": key[1],
                "element_size": key[2],
                "numel": key[3],
                "count": count,
            }
            for key, count in sorted(runner_tensor_signatures.items(), key=repr)
        ],
        "cross_layers_tensor": cross_layers_record,
        "actual_bytes_per_token": actual_kv_bytes_per_token,
    }
    kv_capacity_admission = {
        "null_block_count": null_block_count,
        "compact_blocks_per_slot": compact_blocks_per_slot,
        "compact_generation_count": compact_generation_count,
        "compact_lease_blocks": compact_lease_blocks,
        "compact_lease_tokens": compact_lease_tokens,
        "ordinary_blocks": ordinary_blocks,
        "schedulable_tokens": schedulable_tokens,
        "required_batch_size": required_batch_size,
        "required_tokens_per_request": required_tokens_per_request,
        "required_workload_tokens": required_workload_tokens,
        "required_blocks_per_request": required_blocks_per_request,
        "required_workload_blocks": required_workload_blocks,
        "required_total_blocks": required_total_blocks,
        "required_total_tokens": required_total_tokens,
        "expected_bytes_per_token": expected_kv_bytes_per_token,
        "capacity_covers_required_total": bool(
            ordinary_blocks >= required_workload_blocks > 0
        ),
    }

    return {
        "global_rank": int(getattr(tp_group, "rank")),
        "tp_rank": int(getattr(tp_group, "rank_in_group")),
        "tp_world_size": int(getattr(tp_group, "world_size")),
        "worker_class": type(worker).__name__,
        "model_runner_class": type(model_runner).__name__,
        "cuda_device_runtime": cuda_device_runtime,
        "compilation_cudagraph_mode": _runtime_mode_name(
            getattr(compilation_config, "cudagraph_mode", None)
        ),
        "compilation_cudagraph_capture_sizes": capture_sizes,
        "compilation_max_cudagraph_capture_size": getattr(
            compilation_config, "max_cudagraph_capture_size", None
        ),
        "dispatcher_cudagraph_mode": _runtime_mode_name(
            getattr(dispatcher, "cudagraph_mode", None)
        ),
        "dispatcher_keys_initialized": bool(
            getattr(dispatcher, "keys_initialized", False)
        ),
        "dispatcher_graph_keys": graph_keys,
        "kv_cache_physical_geometry": kv_physical_geometry,
        "kv_cache_arm_capacity_admission": kv_capacity_admission,
        "kv_cache_config_present": True,
        "kv_cache_num_blocks": num_blocks,
        "kv_cache_group_count": len(group_records),
        "kv_cache_groups": group_records,
        "kv_cache_group_page_size_bytes_total": sum(group_page_sizes),
        "kv_cache_group_leaf_page_size_bytes_total": sum(
            group_leaf_page_sizes
        ),
        "kv_cache_composite_page_sizes_match_leaf_sums": all(
            record["composite_page_size_matches_leaf_sum"]
            for record in group_records
        ),
        "kv_cache_uniform_block_size": block_size,
        "kv_cache_tensor_count": len(kv_cache_tensors),
        "kv_cache_tensor_signatures": [
            {"size": key[0], "shared_by_count": key[1], "count": count}
            for key, count in sorted(tensor_signatures.items())
        ],
        "kv_cache_allocated_bytes": allocated_bytes,
        "kv_cache_runner_tensor_count": len(runner_kv_caches),
        "kv_cache_runner_tensor_signatures": [
            {
                "shape": list(key[0]),
                "dtype": key[1],
                "element_size": key[2],
                "numel": key[3],
                "count": count,
            }
            for key, count in sorted(runner_tensor_signatures.items(), key=repr)
        ],
        "kv_cache_cross_layers_tensor": cross_layers_record,
        "kv_cache_actual_bytes_per_token": actual_kv_bytes_per_token,
        "kv_cache_allocated_token_slots": allocated_token_slots,
        "kv_cache_schedulable_tokens": schedulable_tokens,
        "kv_cache_null_block_count": null_block_count,
        "kv_cache_ordinary_blocks": ordinary_blocks,
        "kv_cache_required_batch_size": required_batch_size,
        "kv_cache_required_tokens_per_request": required_tokens_per_request,
        "kv_cache_required_workload_tokens": required_workload_tokens,
        "kv_cache_required_blocks_per_request": required_blocks_per_request,
        "kv_cache_required_workload_blocks": required_workload_blocks,
        "kv_cache_compact_blocks_per_slot": compact_blocks_per_slot,
        "kv_cache_compact_generation_count": compact_generation_count,
        "kv_cache_compact_lease_blocks": compact_lease_blocks,
        "kv_cache_compact_lease_tokens": compact_lease_tokens,
        "kv_cache_required_total_blocks": required_total_blocks,
        "kv_cache_required_total_tokens": required_total_tokens,
        "kv_cache_expected_bytes_per_token": expected_kv_bytes_per_token,
        "kv_cache_capacity_covers_required_total": bool(
            ordinary_blocks >= required_workload_blocks > 0
        ),
    }


class CustomAllReduceRuntimeWorkerExtension:
    """Named benchmark RPC surface that avoids insecure callable transport."""

    def sfi_custom_all_reduce_runtime_state(
        self,
        required_num_tokens: int,
    ) -> dict[str, object]:
        return _worker_custom_all_reduce_runtime_state(
            self,
            required_num_tokens,
        )

    def sfi_engine_runtime_contract_state(
        self,
        required_batch_size: int,
        required_tokens_per_request: int,
        compact_blocks_per_slot: int,
        compact_generation_count: int,
        expected_kv_bytes_per_token: int,
    ) -> dict[str, object]:
        return _worker_engine_runtime_contract_state(
            self,
            required_batch_size,
            required_tokens_per_request,
            compact_blocks_per_slot,
            compact_generation_count,
            expected_kv_bytes_per_token,
        )

    def sfi_reset_sparse_route_counters_for_measurement(self) -> dict[str, object]:
        from patches.fa3_native.install import (
            reset_rank_local_route_counter_slot_for_measurement,
        )
        from patches.fa3_native.postprocess import (
            reset_selector_log_s_runtime_proof_counters,
        )

        reset_selector_log_s_runtime_proof_counters()
        return _attach_tp8_selector_log_s_runtime_proof(
            reset_rank_local_route_counter_slot_for_measurement()
        )

    def sfi_snapshot_sparse_route_counters_after_measurement(
        self,
    ) -> dict[str, object]:
        from patches.fa3_native.install import (
            snapshot_rank_local_route_counter_slot_after_measurement,
        )

        return _attach_tp8_selector_log_s_runtime_proof(
            snapshot_rank_local_route_counter_slot_after_measurement()
        )


def collect_custom_all_reduce_runtime_proof(
    engine: object,
    *,
    tensor_parallel_size: int,
    decision: CustomAllReduceDecision,
    required_num_tokens: int,
) -> dict[str, object]:
    """Collect and validate the instantiated CustomAllreduce state on every rank.

    The pre-launch policy is not runtime proof: vLLM can still decline to
    instantiate CustomAllreduce (for example after its own P2P checks).  A TP
    benchmark must therefore fail before warmup when worker reality disagrees
    with the configured effective policy.
    """
    tp_size = int(tensor_parallel_size)
    if tp_size <= 0:
        raise RuntimeError(
            "E_CUSTOM_ALL_REDUCE_RUNTIME_TP_SIZE: "
            f"tensor_parallel_size={tp_size}"
        )
    if (
        isinstance(required_num_tokens, bool)
        or not isinstance(required_num_tokens, int)
        or required_num_tokens <= 0
    ):
        raise RuntimeError(
            "E_CUSTOM_ALL_REDUCE_REQUIRED_NUM_TOKENS: "
            f"required_num_tokens={required_num_tokens!r}"
        )
    collective_rpc = getattr(engine, "collective_rpc", None)
    if not callable(collective_rpc):
        raise RuntimeError(
            "E_CUSTOM_ALL_REDUCE_RUNTIME_RPC_UNAVAILABLE: "
            "LLM.collective_rpc is required for worker-runtime proof"
        )
    try:
        raw_records = collective_rpc(
            CUSTOM_ALL_REDUCE_RUNTIME_RPC_METHOD,
            timeout=60.0,
            args=(required_num_tokens,),
        )
    except Exception as exc:
        raise RuntimeError(
            "E_CUSTOM_ALL_REDUCE_RUNTIME_RPC_FAILED: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    errors: list[str] = []
    if not isinstance(raw_records, list):
        raise RuntimeError(
            "E_CUSTOM_ALL_REDUCE_RUNTIME_SCHEMA: "
            f"collective_rpc returned {type(raw_records).__name__}, expected list"
        )
    if len(raw_records) != tp_size:
        errors.append(
            f"record_count={len(raw_records)} expected={tp_size}"
        )

    records: list[dict[str, object]] = []
    required_fields = {
        "tp_rank",
        "tp_world_size",
        "parallel_config_disable_custom_all_reduce",
        "device_communicator_use_custom_allreduce",
        "device_communicator_use_flashinfer_allreduce",
        "fi_ar_comm_active",
        "vllm_use_nccl_symm_mem",
        "ca_comm_present",
        "ca_comm_disabled",
        "ca_comm_active",
        "ca_comm_fully_connected",
        "ca_comm_max_size",
        "required_num_tokens",
        "model_hidden_size",
        "model_dtype",
        "model_dtype_bytes",
        "required_payload_numel",
        "required_payload_bytes",
        "required_payload_spec_error",
        "ca_comm_capacity_covers_required_payload",
        "ca_comm_dispatch_size_eligible",
        "ca_comm_should_custom_ar_callable",
        "ca_comm_synthetic_should_custom_ar",
        "ca_comm_synthetic_should_custom_ar_error",
        "reason",
    }
    for index, raw_record in enumerate(raw_records):
        if not isinstance(raw_record, dict):
            errors.append(
                f"record[{index}]={type(raw_record).__name__} expected=dict"
            )
            continue
        missing = sorted(required_fields.difference(raw_record))
        if missing:
            errors.append(f"record[{index}] missing={','.join(missing)}")
            continue
        record = dict(raw_record)
        rank = record.get("tp_rank")
        world_size = record.get("tp_world_size")
        active = record.get("ca_comm_active")
        if isinstance(rank, bool) or not isinstance(rank, int):
            errors.append(f"record[{index}].tp_rank={rank!r}")
            continue
        if isinstance(world_size, bool) or not isinstance(world_size, int):
            errors.append(f"record[{index}].tp_world_size={world_size!r}")
            continue
        if not isinstance(active, bool):
            errors.append(f"record[{index}].ca_comm_active={active!r}")
            continue
        rank_prefix = f"rank{rank}"
        cuda_device_runtime = record.get("cuda_device_runtime")
        if not isinstance(cuda_device_runtime, dict):
            errors.append(f"{rank_prefix}.cuda_device_runtime_invalid")
        else:
            if cuda_device_runtime.get("current_device") != rank:
                errors.append(
                    f"{rank_prefix}.cuda_current_device="
                    f"{cuda_device_runtime.get('current_device')!r}"
                )
            for field in ("capability", "name"):
                if not isinstance(cuda_device_runtime.get(field), str) or not (
                    cuda_device_runtime.get(field)
                ):
                    errors.append(f"{rank_prefix}.cuda_{field}_missing")
            if (
                type(cuda_device_runtime.get("total_memory_bytes")) is not int
                or cuda_device_runtime.get("total_memory_bytes") <= 0
            ):
                errors.append(f"{rank_prefix}.cuda_total_memory_invalid")
            if not isinstance(cuda_device_runtime.get("uuid"), str):
                errors.append(f"{rank_prefix}.cuda_uuid_invalid")
        record_required_tokens = record.get("required_num_tokens")
        hidden_size = record.get("model_hidden_size")
        dtype_name = record.get("model_dtype")
        dtype_bytes = record.get("model_dtype_bytes")
        payload_numel = record.get("required_payload_numel")
        payload_bytes = record.get("required_payload_bytes")
        if record_required_tokens != required_num_tokens:
            errors.append(
                f"{rank_prefix}.required_num_tokens="
                f"{record_required_tokens!r} expected={required_num_tokens}"
            )
        if (
            isinstance(hidden_size, bool)
            or not isinstance(hidden_size, int)
            or hidden_size <= 0
        ):
            errors.append(f"{rank_prefix}.model_hidden_size={hidden_size!r}")
        if not isinstance(dtype_name, str) or not dtype_name:
            errors.append(f"{rank_prefix}.model_dtype={dtype_name!r}")
        if (
            isinstance(dtype_bytes, bool)
            or not isinstance(dtype_bytes, int)
            or dtype_bytes <= 0
        ):
            errors.append(f"{rank_prefix}.model_dtype_bytes={dtype_bytes!r}")
        expected_numel = (
            required_num_tokens * hidden_size
            if isinstance(hidden_size, int) and not isinstance(hidden_size, bool)
            else None
        )
        if payload_numel != expected_numel:
            errors.append(
                f"{rank_prefix}.required_payload_numel={payload_numel!r} "
                f"expected={expected_numel!r}"
            )
        expected_payload_bytes = (
            expected_numel * dtype_bytes
            if isinstance(expected_numel, int)
            and isinstance(dtype_bytes, int)
            and not isinstance(dtype_bytes, bool)
            else None
        )
        if payload_bytes != expected_payload_bytes:
            errors.append(
                f"{rank_prefix}.required_payload_bytes={payload_bytes!r} "
                f"expected={expected_payload_bytes!r}"
            )
        payload_spec_error = record.get("required_payload_spec_error")
        if payload_spec_error != "":
            errors.append(
                f"{rank_prefix}.required_payload_spec_error="
                f"{payload_spec_error!r}"
            )
        records.append(record)

    records.sort(key=lambda record: int(record["tp_rank"]))
    ranks = [int(record["tp_rank"]) for record in records]
    expected_ranks = list(range(tp_size))
    if ranks != expected_ranks:
        errors.append(f"tp_ranks={ranks!r} expected={expected_ranks!r}")
    bad_world_sizes = [
        f"rank{int(record['tp_rank'])}={int(record['tp_world_size'])}"
        for record in records
        if int(record["tp_world_size"]) != tp_size
    ]
    if bad_world_sizes:
        errors.append("tp_world_size=" + ",".join(bad_world_sizes))
    payload_signatures = {
        (
            record["model_hidden_size"],
            record["model_dtype"],
            record["model_dtype_bytes"],
            record["required_payload_bytes"],
        )
        for record in records
    }
    if len(payload_signatures) > 1:
        errors.append(
            "rank_payload_spec_mismatch="
            f"{sorted(payload_signatures, key=repr)!r}"
        )

    if tp_size > 1:
        if decision.effective not in {"enabled", "disabled"}:
            errors.append(
                f"configured_effective={decision.effective!r} expected enabled|disabled"
            )
        expected_active = decision.effective == "enabled"
        expected_config_disabled = not expected_active
        for record in records:
            rank = int(record["tp_rank"])
            active = record["ca_comm_active"]
            config_disabled = record[
                "parallel_config_disable_custom_all_reduce"
            ]
            communicator_enabled = record[
                "device_communicator_use_custom_allreduce"
            ]
            flashinfer_enabled = bool(
                record["device_communicator_use_flashinfer_allreduce"]
                or record["fi_ar_comm_active"]
            )
            nccl_symm_mem_enabled = bool(
                record["vllm_use_nccl_symm_mem"]
            )
            if active is not expected_active:
                errors.append(
                    f"rank{rank}.ca_comm_active={active!r} "
                    f"expected={expected_active!r} reason={record['reason']!r}"
                )
            if config_disabled is not expected_config_disabled:
                errors.append(
                    f"rank{rank}.parallel_config_disable_custom_all_reduce="
                    f"{config_disabled!r} expected={expected_config_disabled!r}"
                )
            if communicator_enabled is not expected_active:
                errors.append(
                    f"rank{rank}.device_communicator_use_custom_allreduce="
                    f"{communicator_enabled!r} expected={expected_active!r}"
                )
            if expected_active:
                if record["ca_comm_present"] is not True:
                    errors.append(f"rank{rank}.ca_comm_present=false")
                if record["ca_comm_disabled"] is not False:
                    errors.append(
                        f"rank{rank}.ca_comm_disabled="
                        f"{record['ca_comm_disabled']!r}"
                    )
                fully_connected = record["ca_comm_fully_connected"]
                if not isinstance(fully_connected, bool):
                    errors.append(
                        f"rank{rank}.ca_comm_fully_connected="
                        f"{fully_connected!r}"
                    )
                elif tp_size > 2 and not fully_connected:
                    errors.append(
                        f"rank{rank}.ca_comm_fully_connected=false"
                    )
                max_size = record["ca_comm_max_size"]
                if (
                    isinstance(max_size, bool)
                    or not isinstance(max_size, int)
                    or max_size <= 0
                ):
                    errors.append(
                        f"rank{rank}.ca_comm_max_size={max_size!r}"
                    )
                required_payload_bytes = record["required_payload_bytes"]
                if (
                    isinstance(max_size, int)
                    and not isinstance(max_size, bool)
                    and isinstance(required_payload_bytes, int)
                    and not isinstance(required_payload_bytes, bool)
                    and max_size <= required_payload_bytes
                ):
                    errors.append(
                        f"rank{rank}.ca_comm_max_size={max_size} must_exceed_"
                        f"required_payload_bytes={required_payload_bytes}; "
                        "vLLM.should_custom_ar uses strict '<'"
                    )
                if record["ca_comm_capacity_covers_required_payload"] is not True:
                    errors.append(
                        f"rank{rank}.ca_comm_capacity_covers_required_payload="
                        f"{record['ca_comm_capacity_covers_required_payload']!r}"
                    )
                if record["ca_comm_dispatch_size_eligible"] is not True:
                    errors.append(
                        f"rank{rank}.ca_comm_dispatch_size_eligible="
                        f"{record['ca_comm_dispatch_size_eligible']!r}"
                    )
                if record["ca_comm_should_custom_ar_callable"] is not True:
                    errors.append(
                        f"rank{rank}.ca_comm_should_custom_ar_callable="
                        f"{record['ca_comm_should_custom_ar_callable']!r}"
                    )
                if record["ca_comm_synthetic_should_custom_ar"] is not True:
                    errors.append(
                        f"rank{rank}.ca_comm_synthetic_should_custom_ar="
                        f"{record['ca_comm_synthetic_should_custom_ar']!r}"
                    )
                synthetic_error = record[
                    "ca_comm_synthetic_should_custom_ar_error"
                ]
                if synthetic_error != "":
                    errors.append(
                        f"rank{rank}.ca_comm_synthetic_should_custom_ar_error="
                        f"{synthetic_error!r}"
                    )
            if expected_active and flashinfer_enabled:
                errors.append(
                    f"rank{rank}.flashinfer_preempts_custom_all_reduce"
                )
            if expected_active and nccl_symm_mem_enabled:
                errors.append(
                    f"rank{rank}.nccl_symm_mem_preempts_custom_all_reduce"
                )
    elif any(bool(record["ca_comm_active"]) for record in records):
        errors.append("tp1_custom_all_reduce_unexpectedly_active")

    if errors:
        raise RuntimeError(
            "E_CUSTOM_ALL_REDUCE_RUNTIME_MISMATCH: " + "; ".join(errors)
        )

    active_rank_count = sum(
        1 for record in records if bool(record["ca_comm_active"])
    )
    flashinfer_preemptor_enabled = any(
        bool(
            record["device_communicator_use_flashinfer_allreduce"]
            or record["fi_ar_comm_active"]
        )
        for record in records
    )
    nccl_symm_mem_preemptor_enabled = any(
        bool(record["vllm_use_nccl_symm_mem"]) for record in records
    )
    reference_record = records[0] if records else {}
    return {
        "custom_all_reduce_runtime_proof_required": bool(tp_size > 1),
        "custom_all_reduce_runtime_proof_passed": True,
        "custom_all_reduce_runtime_configured_effective": str(
            decision.effective
        ),
        "custom_all_reduce_runtime_tensor_parallel_size": tp_size,
        "custom_all_reduce_runtime_rank_count": len(records),
        "custom_all_reduce_runtime_active_rank_count": active_rank_count,
        "custom_all_reduce_runtime_inactive_rank_count": (
            len(records) - active_rank_count
        ),
        "custom_all_reduce_runtime_all_ranks_active": bool(
            active_rank_count == tp_size
        ),
        "custom_all_reduce_runtime_rank_consistent": True,
        "custom_all_reduce_runtime_preemptor_flashinfer_enabled": (
            flashinfer_preemptor_enabled
        ),
        "custom_all_reduce_runtime_preemptor_nccl_symm_mem_enabled": (
            nccl_symm_mem_preemptor_enabled
        ),
        "custom_all_reduce_runtime_required_num_tokens": int(
            required_num_tokens
        ),
        "custom_all_reduce_runtime_model_hidden_size": reference_record.get(
            "model_hidden_size"
        ),
        "custom_all_reduce_runtime_model_dtype": reference_record.get(
            "model_dtype"
        ),
        "custom_all_reduce_runtime_model_dtype_bytes": reference_record.get(
            "model_dtype_bytes"
        ),
        "custom_all_reduce_runtime_required_payload_bytes": (
            reference_record.get("required_payload_bytes")
        ),
        "custom_all_reduce_runtime_all_ranks_payload_eligible": all(
            record.get("ca_comm_synthetic_should_custom_ar") is True
            for record in records
        ) if decision.effective == "enabled" else False,
        "custom_all_reduce_runtime_records": records,
    }


def collect_engine_runtime_contract_proof(
    engine: object,
    *,
    tensor_parallel_size: int,
    required_batch_size: int,
    required_tokens_per_request: int,
    compact_blocks_per_slot: int,
    compact_generation_count: int,
    expected_kv_bytes_per_token: int,
    required: bool,
) -> dict[str, object]:
    """Validate post-resolution graph keys and allocated KV state on all ranks."""
    if not required:
        return {
            "engine_runtime_contract_proof_required": False,
            "engine_runtime_contract_proof_passed": True,
            "engine_runtime_contract_records": [],
        }
    tp_size = int(tensor_parallel_size)
    if tp_size <= 0:
        raise RuntimeError(
            "E_ENGINE_RUNTIME_CONTRACT_TP_SIZE: "
            f"tensor_parallel_size={tp_size}"
        )
    collective_rpc = getattr(engine, "collective_rpc", None)
    if not callable(collective_rpc):
        raise RuntimeError(
            "E_ENGINE_RUNTIME_CONTRACT_RPC_UNAVAILABLE: "
            "LLM.collective_rpc is required"
        )
    try:
        raw_records = collective_rpc(
            ENGINE_RUNTIME_CONTRACT_RPC_METHOD,
            timeout=60.0,
            args=(
                int(required_batch_size),
                int(required_tokens_per_request),
                int(compact_blocks_per_slot),
                int(compact_generation_count),
                int(expected_kv_bytes_per_token),
            ),
        )
    except Exception as exc:
        raise RuntimeError(
            "E_ENGINE_RUNTIME_CONTRACT_RPC_FAILED: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(raw_records, list):
        raise RuntimeError(
            "E_ENGINE_RUNTIME_CONTRACT_SCHEMA: collective_rpc must return list"
        )

    errors: list[str] = []
    records: list[dict[str, object]] = []
    if len(raw_records) != tp_size:
        errors.append(f"record_count={len(raw_records)} expected={tp_size}")
    for index, raw_record in enumerate(raw_records):
        if not isinstance(raw_record, dict):
            errors.append(f"record[{index}]={type(raw_record).__name__}")
            continue
        record = dict(raw_record)
        rank = record.get("tp_rank")
        world_size = record.get("tp_world_size")
        if isinstance(rank, bool) or not isinstance(rank, int):
            errors.append(f"record[{index}].tp_rank={rank!r}")
            continue
        if world_size != tp_size:
            errors.append(f"rank{rank}.tp_world_size={world_size!r}")
        rank_prefix = f"rank{rank}"
        for key in ("compilation_cudagraph_mode", "dispatcher_cudagraph_mode"):
            if record.get(key) != "FULL":
                errors.append(f"{rank_prefix}.{key}={record.get(key)!r}")
        if record.get("dispatcher_keys_initialized") is not True:
            errors.append(f"{rank_prefix}.dispatcher_keys_initialized=false")
        expected_capture_sizes = [int(required_batch_size)]
        if record.get("compilation_cudagraph_capture_sizes") != expected_capture_sizes:
            errors.append(
                f"{rank_prefix}.capture_sizes="
                f"{record.get('compilation_cudagraph_capture_sizes')!r} "
                f"expected={expected_capture_sizes!r}"
            )
        if record.get("compilation_max_cudagraph_capture_size") != required_batch_size:
            errors.append(
                f"{rank_prefix}.max_capture_size="
                f"{record.get('compilation_max_cudagraph_capture_size')!r}"
            )
        graph_keys = record.get("dispatcher_graph_keys")
        if not isinstance(graph_keys, dict):
            errors.append(f"{rank_prefix}.dispatcher_graph_keys_invalid")
        else:
            full_keys = graph_keys.get("FULL")
            expected_key = {
                "num_tokens": int(required_batch_size),
                "num_reqs": int(required_batch_size),
                "uniform": True,
                "has_lora": False,
                "num_active_loras": 0,
            }
            if not isinstance(full_keys, list) or expected_key not in full_keys:
                errors.append(
                    f"{rank_prefix}.full_decode_key_missing:{expected_key!r}"
                )
            piecewise_keys = graph_keys.get("PIECEWISE", [])
            if piecewise_keys not in ([], None):
                errors.append(f"{rank_prefix}.piecewise_keys_nonempty")

        numeric_expectations = {
            "kv_cache_required_batch_size": required_batch_size,
            "kv_cache_required_tokens_per_request": required_tokens_per_request,
            "kv_cache_compact_blocks_per_slot": compact_blocks_per_slot,
            "kv_cache_compact_generation_count": compact_generation_count,
            "kv_cache_expected_bytes_per_token": expected_kv_bytes_per_token,
        }
        for key, expected in numeric_expectations.items():
            if record.get(key) != expected:
                errors.append(
                    f"{rank_prefix}.{key}={record.get(key)!r}:expected={expected}"
                )
        if record.get("kv_cache_config_present") is not True:
            errors.append(f"{rank_prefix}.kv_cache_config_missing")
        if record.get("kv_cache_group_count") != 1:
            errors.append(
                f"{rank_prefix}.kv_cache_group_count="
                f"{record.get('kv_cache_group_count')!r}:expected=1"
            )
        if record.get("kv_cache_uniform_block_size") != 16:
            errors.append(
                f"{rank_prefix}.kv_cache_uniform_block_size="
                f"{record.get('kv_cache_uniform_block_size')!r}:expected=16"
            )
        for key in (
            "kv_cache_num_blocks",
            "kv_cache_allocated_bytes",
            "kv_cache_allocated_token_slots",
            "kv_cache_schedulable_tokens",
            "kv_cache_tensor_count",
            "kv_cache_runner_tensor_count",
            "kv_cache_required_blocks_per_request",
            "kv_cache_required_workload_blocks",
            "kv_cache_required_total_blocks",
        ):
            value = record.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                errors.append(f"{rank_prefix}.{key}={value!r}")
        actual_bytes_per_token = record.get("kv_cache_actual_bytes_per_token")
        if actual_bytes_per_token != expected_kv_bytes_per_token:
            errors.append(
                f"{rank_prefix}.kv_cache_actual_bytes_per_token="
                f"{actual_bytes_per_token!r}:expected={expected_kv_bytes_per_token}"
            )
        allocated_bytes = record.get("kv_cache_allocated_bytes")
        num_blocks = record.get("kv_cache_num_blocks")
        page_size = record.get("kv_cache_group_page_size_bytes_total")
        leaf_page_size = record.get(
            "kv_cache_group_leaf_page_size_bytes_total"
        )
        if record.get("kv_cache_composite_page_sizes_match_leaf_sums") is not True:
            errors.append(
                f"{rank_prefix}.kv_cache_composite_page_leaf_identity=false"
            )
        if page_size != leaf_page_size:
            errors.append(
                f"{rank_prefix}.kv_cache_page_size_identity="
                f"composite={page_size!r}:leaf_sum={leaf_page_size!r}"
            )
        if (
            isinstance(allocated_bytes, int)
            and isinstance(num_blocks, int)
            and isinstance(page_size, int)
            and allocated_bytes != num_blocks * page_size
        ):
            errors.append(
                f"{rank_prefix}.kv_cache_allocation_identity="
                f"{allocated_bytes}!={num_blocks}*{page_size}"
            )
        null_blocks = record.get("kv_cache_null_block_count")
        compact_blocks = record.get("kv_cache_compact_lease_blocks")
        ordinary_blocks = record.get("kv_cache_ordinary_blocks")
        workload_blocks = record.get("kv_cache_required_workload_blocks")
        required_total_blocks = record.get("kv_cache_required_total_blocks")
        if null_blocks != 1:
            errors.append(
                f"{rank_prefix}.kv_cache_null_block_count={null_blocks!r}:expected=1"
            )
        if all(
            type(value) is int
            for value in (
                num_blocks,
                null_blocks,
                compact_blocks,
                ordinary_blocks,
                workload_blocks,
                required_total_blocks,
            )
        ):
            expected_ordinary = max(
                num_blocks - null_blocks - compact_blocks,
                0,
            )
            if ordinary_blocks != expected_ordinary:
                errors.append(
                    f"{rank_prefix}.kv_cache_ordinary_blocks="
                    f"{ordinary_blocks}:expected={expected_ordinary}"
                )
            expected_total = null_blocks + compact_blocks + workload_blocks
            if required_total_blocks != expected_total:
                errors.append(
                    f"{rank_prefix}.kv_cache_required_total_blocks="
                    f"{required_total_blocks}:expected={expected_total}"
                )
        cross_layers = record.get("kv_cache_cross_layers_tensor")
        if isinstance(cross_layers, dict) and cross_layers.get("nbytes") != allocated_bytes:
            errors.append(
                f"{rank_prefix}.cross_layers_nbytes="
                f"{cross_layers.get('nbytes')!r}:expected={allocated_bytes!r}"
            )
        if record.get("kv_cache_capacity_covers_required_total") is not True:
            errors.append(
                f"{rank_prefix}.kv_cache_capacity_covers_required_total=false:"
                f"capacity={record.get('kv_cache_schedulable_tokens')!r}:"
                f"required={record.get('kv_cache_required_total_tokens')!r}"
            )
        records.append(record)

    records.sort(key=lambda record: int(record["tp_rank"]))
    ranks = [int(record["tp_rank"]) for record in records]
    if ranks != list(range(tp_size)):
        errors.append(f"tp_ranks={ranks!r}:expected={list(range(tp_size))!r}")
    rank_signatures = []
    for record in records:
        rank_signatures.append(
            {
                key: value
                for key, value in record.items()
                if key
                not in {
                    "global_rank",
                    "tp_rank",
                    "worker_class",
                    "cuda_device_runtime",
                }
            }
        )
    if rank_signatures and any(
        signature != rank_signatures[0] for signature in rank_signatures[1:]
    ):
        errors.append("worker_runtime_contract_rank_mismatch")
    if errors:
        raise RuntimeError(
            "E_ENGINE_RUNTIME_CONTRACT_MISMATCH: " + "; ".join(errors)
        )

    reference = records[0] if records else {}
    physical_rank_records = [
        {
            "tp_rank": record.get("tp_rank"),
            "tp_world_size": record.get("tp_world_size"),
            "model_runner_class": record.get("model_runner_class"),
            "cuda_device_runtime": record.get("cuda_device_runtime"),
            "compilation_cudagraph_mode": record.get(
                "compilation_cudagraph_mode"
            ),
            "compilation_cudagraph_capture_sizes": record.get(
                "compilation_cudagraph_capture_sizes"
            ),
            "compilation_max_cudagraph_capture_size": record.get(
                "compilation_max_cudagraph_capture_size"
            ),
            "dispatcher_cudagraph_mode": record.get(
                "dispatcher_cudagraph_mode"
            ),
            "dispatcher_keys_initialized": record.get(
                "dispatcher_keys_initialized"
            ),
            "dispatcher_graph_keys": record.get("dispatcher_graph_keys"),
            "kv_cache": record.get("kv_cache_physical_geometry"),
        }
        for record in records
    ]
    arm_capacity_records = [
        {
            "tp_rank": record.get("tp_rank"),
            **(
                record.get("kv_cache_arm_capacity_admission")
                if isinstance(
                    record.get("kv_cache_arm_capacity_admission"), dict
                )
                else {}
            ),
        }
        for record in records
    ]
    return {
        "engine_runtime_contract_proof_required": True,
        "engine_runtime_contract_proof_passed": True,
        "engine_runtime_contract_tensor_parallel_size": tp_size,
        "engine_runtime_contract_rank_count": len(records),
        "engine_runtime_contract_rank_consistent": True,
        "engine_runtime_graph_mode": reference.get(
            "dispatcher_cudagraph_mode"
        ),
        "engine_runtime_graph_capture_sizes": reference.get(
            "compilation_cudagraph_capture_sizes"
        ),
        "engine_runtime_graph_full_decode_key_present": True,
        "engine_runtime_physical_geometry": {
            "tensor_parallel_size": tp_size,
            "rank_records": physical_rank_records,
        },
        "engine_runtime_arm_capacity_admission": {
            "tensor_parallel_size": tp_size,
            "rank_records": arm_capacity_records,
        },
        "engine_runtime_kv_num_blocks": reference.get("kv_cache_num_blocks"),
        "engine_runtime_kv_block_size": reference.get(
            "kv_cache_uniform_block_size"
        ),
        "engine_runtime_kv_allocated_bytes": reference.get(
            "kv_cache_allocated_bytes"
        ),
        "engine_runtime_kv_composite_page_size_bytes": reference.get(
            "kv_cache_group_page_size_bytes_total"
        ),
        "engine_runtime_kv_leaf_page_size_bytes": reference.get(
            "kv_cache_group_leaf_page_size_bytes_total"
        ),
        "engine_runtime_kv_page_size_identity_passed": True,
        "engine_runtime_kv_actual_bytes_per_token": reference.get(
            "kv_cache_actual_bytes_per_token"
        ),
        "engine_runtime_kv_schedulable_tokens": reference.get(
            "kv_cache_schedulable_tokens"
        ),
        "engine_runtime_kv_null_block_count": reference.get(
            "kv_cache_null_block_count"
        ),
        "engine_runtime_kv_ordinary_blocks": reference.get(
            "kv_cache_ordinary_blocks"
        ),
        "engine_runtime_kv_required_blocks_per_request": reference.get(
            "kv_cache_required_blocks_per_request"
        ),
        "engine_runtime_kv_required_workload_blocks": reference.get(
            "kv_cache_required_workload_blocks"
        ),
        "engine_runtime_kv_compact_lease_tokens": reference.get(
            "kv_cache_compact_lease_tokens"
        ),
        "engine_runtime_kv_required_total_tokens": reference.get(
            "kv_cache_required_total_tokens"
        ),
        "engine_runtime_kv_required_total_blocks": reference.get(
            "kv_cache_required_total_blocks"
        ),
        "engine_runtime_kv_capacity_covers_required_total": True,
        "engine_runtime_contract_records": records,
    }


def collect_engine_core_block_pool_reservation_proof(
    engine: object,
    *,
    engine_runtime_contract_proof: dict[str, object],
    expected_compact_blocks_per_slot: int,
    expected_compact_generation_count: int,
    expected_batch_size: int,
    required: bool,
) -> dict[str, object]:
    """Prove the scheduler BlockPool reservation through named core utility."""
    if not required:
        return {
            "engine_core_block_pool_proof_required": False,
            "engine_core_block_pool_proof_passed": True,
        }
    from patches.page_kv_residency import (
        ENGINE_CORE_BLOCK_POOL_STATE_UTILITY,
    )

    llm_engine = getattr(engine, "llm_engine", None)
    core_client = getattr(llm_engine, "engine_core", None)
    call_utility = getattr(core_client, "call_utility", None)
    if not callable(call_utility):
        raise RuntimeError(
            "E_ENGINE_CORE_BLOCK_POOL_PROOF_UNAVAILABLE: named call_utility"
        )
    try:
        raw_state = call_utility(ENGINE_CORE_BLOCK_POOL_STATE_UTILITY)
    except Exception as exc:
        raise RuntimeError(
            "E_ENGINE_CORE_BLOCK_POOL_PROOF_FAILED: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(raw_state, dict):
        raise RuntimeError(
            "E_ENGINE_CORE_BLOCK_POOL_PROOF_SCHEMA: state must be dict"
        )
    state = dict(raw_state)
    errors: list[str] = []
    expected_reserved_count = (
        int(expected_batch_size)
        * int(expected_compact_blocks_per_slot)
        * int(expected_compact_generation_count)
    )
    for name, value in {
        "expected_batch_size": expected_batch_size,
        "expected_compact_blocks_per_slot": expected_compact_blocks_per_slot,
        "expected_compact_generation_count": expected_compact_generation_count,
    }.items():
        if type(value) is not int or value < 0:
            errors.append(f"{name}={value!r}")
    if expected_batch_size <= 0:
        errors.append("expected_batch_size_must_be_positive")
    if state.get("schema") != "sfi.engine_core_block_pool_state.v1":
        errors.append(f"schema={state.get('schema')!r}")
    num_blocks = state.get("num_gpu_blocks")
    worker_num_blocks = engine_runtime_contract_proof.get(
        "engine_runtime_kv_num_blocks"
    )
    if (
        type(num_blocks) is not int
        or num_blocks <= 0
        or num_blocks != worker_num_blocks
    ):
        errors.append(
            f"num_gpu_blocks={num_blocks!r}:worker={worker_num_blocks!r}"
        )
    expected_reserved_ids = (
        list(range(num_blocks - expected_reserved_count, num_blocks))
        if type(num_blocks) is int and num_blocks >= expected_reserved_count
        else []
    )
    if state.get("reserved_block_ids") != expected_reserved_ids:
        errors.append("reserved_block_ids_not_exact_tail")
    if state.get("reserved_block_count") != expected_reserved_count:
        errors.append(
            f"reserved_block_count={state.get('reserved_block_count')!r}:"
            f"expected={expected_reserved_count}"
        )
    if state.get("reserved_ids_tail_contiguous") is not True:
        errors.append("reserved_ids_not_tail_contiguous")
    if state.get("reserved_ids_in_free_queue") != []:
        errors.append("reserved_ids_present_in_free_queue")
    if state.get("null_block_id") != 0:
        errors.append(f"null_block_id={state.get('null_block_id')!r}")
    if state.get("null_block_is_null") is not True:
        errors.append("null_block_is_null=false")
    if state.get("null_block_in_free_queue") is not False:
        errors.append("null_block_in_free_queue=true")
    if type(num_blocks) is int:
        expected_free = num_blocks - 1 - expected_reserved_count
        if state.get("num_free_blocks") != expected_free:
            errors.append(
                f"num_free_blocks={state.get('num_free_blocks')!r}:"
                f"expected={expected_free}"
            )
        if state.get("free_queue_reported_count") != expected_free:
            errors.append("free_queue_reported_count_mismatch")

    reserved_states = state.get("reserved_blocks")
    if not isinstance(reserved_states, list) or len(reserved_states) != (
        expected_reserved_count
    ):
        errors.append("reserved_block_states_shape_mismatch")
    elif any(
        not isinstance(record, dict)
        or record.get("block_id") != expected_reserved_ids[index]
        or record.get("ref_cnt") != 1
        or record.get("is_null") is not False
        or record.get("in_free_queue") is not False
        for index, record in enumerate(reserved_states)
    ):
        errors.append("reserved_block_state_mismatch")

    manager_lease = state.get("manager_lease")
    pool_lease = state.get("pool_lease")
    if expected_reserved_count > 0:
        if not isinstance(manager_lease, dict) or not isinstance(pool_lease, dict):
            errors.append("compact_lease_missing_from_manager_or_pool")
        else:
            expected_lease = {
                "reserved_manager_block_ids": expected_reserved_ids,
                "manager_block_size": 16,
                "kernel_page_size": 16,
                "compact_blocks_per_slot": expected_compact_blocks_per_slot,
                "max_live_sparse_slots": expected_batch_size,
            }
            for field, expected in expected_lease.items():
                if manager_lease.get(field) != expected:
                    errors.append(f"manager_lease_{field}_mismatch")
                if pool_lease.get(field) != expected:
                    errors.append(f"pool_lease_{field}_mismatch")
        if state.get("manager_pool_lease_same_object") is not True:
            errors.append("manager_pool_lease_identity_mismatch")
    else:
        if manager_lease is not None or pool_lease is not None:
            errors.append("dense_block_pool_has_compact_lease")

    if errors:
        raise RuntimeError(
            "E_ENGINE_CORE_BLOCK_POOL_PROOF_MISMATCH: " + "; ".join(errors)
        )
    admission = engine_runtime_contract_proof.get(
        "engine_runtime_arm_capacity_admission"
    )
    if not isinstance(admission, dict):
        raise RuntimeError(
            "E_ENGINE_CORE_BLOCK_POOL_PROOF_SCHEMA: capacity admission missing"
        )
    admission["engine_core_block_pool"] = state
    return {
        "engine_core_block_pool_proof_required": True,
        "engine_core_block_pool_proof_passed": True,
        "engine_core_block_pool_expected_reserved_count": (
            expected_reserved_count
        ),
        "engine_core_block_pool_state": state,
    }


def resolve_custom_all_reduce_decision(
    *,
    tensor_parallel_size: int,
    force_disabled: bool,
    force_enabled: bool,
    nvlink_probe: Callable[[], tuple[bool, str]],
) -> CustomAllReduceDecision:
    """Resolve the exact policy passed to vLLM without probing unnecessarily."""
    tp_size = int(tensor_parallel_size)
    if tp_size <= 1:
        return CustomAllReduceDecision(
            requested="not_applicable",
            effective="not_applicable",
            reason=f"tensor_parallel_size={tp_size}",
        )
    if force_disabled:
        return CustomAllReduceDecision(
            requested="force_disabled",
            effective="disabled",
            reason="VLLM_SPARSE_FORCE_DISABLE_CUSTOM_AR=1",
        )
    if force_enabled:
        return CustomAllReduceDecision(
            requested="force_enabled",
            effective="enabled",
            reason="--enable-custom-all-reduce",
        )
    nvlink_ok, nvlink_reason = nvlink_probe()
    return CustomAllReduceDecision(
        requested="auto",
        effective="enabled" if nvlink_ok else "disabled",
        reason=f"nvlink_probe:{nvlink_reason}",
    )


class EngineShutdownGuard:
    """Own one vLLM V1 engine-core shutdown with an exception-safe fallback.

    Successful benchmark arms call :meth:`close` after every artifact has been
    flushed, so a shutdown failure makes the arm non-green.  If workload code
    raises first, the registered exit callback still asks vLLM to release its
    workers without replacing the original exception.
    """

    def __init__(self, engine: object) -> None:
        self._engine: object | None = engine
        self._closed = False
        self._exit_callback = self._close_at_exit
        atexit.register(self._exit_callback)

    def close(self) -> None:
        if self._closed:
            return
        engine = self._engine
        llm_engine = getattr(engine, "llm_engine", None)
        engine_core = getattr(llm_engine, "engine_core", None)
        shutdown = getattr(engine_core, "shutdown", None)
        if not callable(shutdown):
            raise RuntimeError(
                "E_VLLM_ENGINE_SHUTDOWN_UNAVAILABLE: benchmark arm cannot "
                "prove EngineCore worker teardown"
            )
        shutdown()
        self._closed = True
        self._engine = None
        atexit.unregister(self._exit_callback)

    def _close_at_exit(self) -> None:
        if self._closed:
            return
        try:
            self.close()
        except BaseException as exc:  # Preserve an already-active workload error.
            print(
                "E_VLLM_ENGINE_SHUTDOWN_AT_EXIT: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )


@dataclass
class DecodeWindowMeter:
    """Measure decode throughput in the window [first_emit_ts, last_emit_ts].

    When *batch_size* is provided, an additional **all-decode window** is
    tracked: it starts from the first step where ``new_tokens >= batch_size``
    (i.e. all requests have entered decode) and ends at the last emit.  This
    avoids counting the slow prefill-interleaved steps that occur when
    different-length prompts finish chunked prefill at different times.
    """

    batch_size: int = 0
    first_emit_ts: float | None = None
    last_emit_ts: float | None = None
    total_tokens: int = 0
    first_emit_tokens: int = 0
    _prev_emit_ts: float | None = None
    decode_step_durations_s: list[float] = field(default_factory=list)
    # all-decode window (from first full-batch decode step)
    _ad_start_ts: float | None = None
    _ad_start_tokens: int = 0
    _ad_steps: int = 0
    _ad_full_batch_steps: int = 0
    _ad_partial_batch_steps: int = 0
    _ad_zero_token_steps: int = 0
    _observed_steps: int = 0
    _ad_start_step_index: int = -1

    def observe(self, ts: float, new_tokens: int) -> None:
        step_index = self._observed_steps
        self._observed_steps += 1
        tokens = int(new_tokens)
        if tokens <= 0:
            if self._ad_start_ts is not None:
                self._ad_zero_token_steps += 1
            return
        timestamp = float(ts)
        self.total_tokens += tokens
        if self.first_emit_ts is None:
            self.first_emit_ts = timestamp
            self.last_emit_ts = timestamp
            self._prev_emit_ts = timestamp
            self.first_emit_tokens = tokens
        else:
            prev_ts = self._prev_emit_ts if self._prev_emit_ts is not None else timestamp
            dt = timestamp - prev_ts
            if dt < 0.0:
                dt = 0.0
            self.decode_step_durations_s.append(float(dt))
            self._prev_emit_ts = timestamp
            self.last_emit_ts = timestamp

        # all-decode window: starts when tok/step reaches batch_size
        if self.batch_size > 0 and self._ad_start_ts is None and tokens == self.batch_size:
            self._ad_start_ts = timestamp
            self._ad_start_tokens = self.total_tokens
            self._ad_start_step_index = int(step_index)
        if self._ad_start_ts is not None:
            self._ad_steps += 1
            if tokens == self.batch_size:
                self._ad_full_batch_steps += 1
            else:
                self._ad_partial_batch_steps += 1

    def finalize(self) -> Tuple[float, int, float, list[float]]:
        window_tokens = max(0, int(self.total_tokens) - int(self.first_emit_tokens))
        if self.first_emit_ts is None or self.last_emit_ts is None:
            return 0.0, int(window_tokens), float("nan"), list(self.decode_step_durations_s)

        decode_elapsed_s = float(self.last_emit_ts - self.first_emit_ts)
        if decode_elapsed_s <= 0.0:
            return 0.0, int(window_tokens), float("nan"), list(self.decode_step_durations_s)

        decode_tps = float(window_tokens) / float(decode_elapsed_s)
        return float(decode_elapsed_s), int(window_tokens), float(decode_tps), list(
            self.decode_step_durations_s
        )

    def boundary_delays(self, total_start_s: float, total_end_s: float) -> Tuple[float, float]:
        if self.first_emit_ts is None or self.last_emit_ts is None:
            return float("nan"), float("nan")
        first_emit_delay_s = max(0.0, float(self.first_emit_ts) - float(total_start_s))
        post_decode_tail_s = max(0.0, float(total_end_s) - float(self.last_emit_ts))
        return first_emit_delay_s, post_decode_tail_s

    def finalize_all_decode(self) -> Tuple[float, int, float, int]:
        """Return (elapsed_s, tokens, tok_per_s, steps) for the all-decode window.

        Falls back to the full window if batch_size was not set or all-decode
        was never entered.
        """
        if self._ad_start_ts is not None and self.last_emit_ts is not None:
            ad_elapsed = float(self.last_emit_ts - self._ad_start_ts)
            ad_tokens = max(0, int(self.total_tokens) - int(self._ad_start_tokens))
            ad_tps = float(ad_tokens) / float(ad_elapsed) if ad_elapsed > 0 else float("nan")
            return ad_elapsed, ad_tokens, ad_tps, int(self._ad_steps)
        # fallback: use the full first-emit window
        elapsed, tokens, tps, _ = self.finalize()
        return elapsed, tokens, tps, len(self.decode_step_durations_s)

    def all_decode_contract(self) -> dict[str, object]:
        """Return strict full-batch measurement-window evidence."""
        entered = self._ad_start_ts is not None
        return {
            "all_decode_entered": entered,
            "all_decode_full_batch_steps": int(self._ad_full_batch_steps),
            "all_decode_partial_batch_steps": int(self._ad_partial_batch_steps),
            "all_decode_zero_token_steps": int(self._ad_zero_token_steps),
            "all_decode_fallback_used": not entered,
        }

    @property
    def all_decode_start_step_index(self) -> int:
        """Internal observer boundary; not part of the public decode contract."""
        return int(self._ad_start_step_index)


def reset_cudagraph_runtime_observer(llm_engine: object) -> None:
    """Start an in-memory diagnostic observer for vLLM's own CUDAGraphStat."""
    setattr(llm_engine, "_sfi_cudagraph_runtime_records", [])
    setattr(llm_engine, "_sfi_cudagraph_runtime_observer_enabled", True)


def stop_cudagraph_runtime_observer(llm_engine: object) -> list[dict[str, object] | None]:
    setattr(llm_engine, "_sfi_cudagraph_runtime_observer_enabled", False)
    records = getattr(llm_engine, "_sfi_cudagraph_runtime_records", None)
    if not isinstance(records, list):
        raise RuntimeError(
            "E_CUDAGRAPH_RUNTIME_OBSERVER_UNAVAILABLE: measurement records"
        )
    return list(records)


def summarize_cudagraph_runtime_observer(
    records: list[dict[str, object] | None],
    *,
    all_decode_start_step_index: int,
    expected_batch_size: int,
    expected_total_engine_steps: int,
) -> dict[str, object]:
    """Aggregate actual graph dispatch only for the strict all-decode window."""
    from collections import Counter

    if len(records) != int(expected_total_engine_steps):
        raise RuntimeError(
            "E_CUDAGRAPH_RUNTIME_OBSERVER_STEP_MISMATCH: "
            f"records={len(records)} expected={expected_total_engine_steps}"
        )
    start = int(all_decode_start_step_index)
    if start < 0 or start >= len(records):
        raise RuntimeError(
            "E_CUDAGRAPH_RUNTIME_OBSERVER_WINDOW: "
            f"all_decode_start_step_index={start} records={len(records)}"
        )
    window = records[start:]
    missing = sum(record is None for record in window)
    distribution: Counter[tuple[object, ...]] = Counter()
    for record in window:
        if record is None:
            continue
        distribution[
            (
                record.get("num_unpadded_tokens"),
                record.get("num_padded_tokens"),
                record.get("num_paddings"),
                record.get("runtime_mode"),
            )
        ] += 1
    expected_signature = (expected_batch_size, expected_batch_size, 0, "FULL")
    exact_steps = int(distribution.get(expected_signature, 0))
    return {
        "cudagraph_runtime_observer_scope": "diagnostic_measurement_all_decode",
        "cudagraph_runtime_observer_enabled": True,
        "cudagraph_runtime_observer_total_step_count": len(records),
        "cudagraph_runtime_observer_all_decode_step_count": len(window),
        "cudagraph_runtime_observer_missing_step_count": missing,
        "cudagraph_runtime_observer_exact_full_step_count": exact_steps,
        "cudagraph_runtime_observer_all_decode_exact_full": bool(
            missing == 0 and exact_steps == len(window) and len(window) > 0
        ),
        "cudagraph_runtime_observer_distribution": [
            {
                "num_unpadded_tokens": signature[0],
                "num_padded_tokens": signature[1],
                "num_paddings": signature[2],
                "runtime_mode": signature[3],
                "count": count,
            }
            for signature, count in sorted(distribution.items(), key=repr)
        ],
    }


def count_new_tokens(request_outputs: Iterable[Any], prev_len: dict[str, int]) -> int:
    step_new_tokens = 0
    for item in request_outputs:
        rid = getattr(item, "request_id", None)
        if rid is None:
            continue
        outputs = getattr(item, "outputs", None)
        if not outputs:
            continue
        token_ids = getattr(outputs[0], "token_ids", None)
        if token_ids is None:
            continue
        curr = int(len(token_ids))
        prev = int(prev_len.get(rid, 0))
        if curr > prev:
            step_new_tokens += curr - prev
            prev_len[rid] = curr
    return int(step_new_tokens)


_INNER_TIMING_FALSE_VALUES = {"", "0", "false", "off", "no"}


def _engine_core_inner_timing_enabled() -> bool:
    value = str(os.environ.get("VLLM_DECODE_ENGINE_CORE_INNER_TIMING", "") or "")
    return value.strip().lower() not in _INNER_TIMING_FALSE_VALUES


def _elapsed_us(start_s: float) -> float:
    return float((time.perf_counter() - start_s) * 1_000_000.0)


def _record_engine_core_timing(engine_core: Any, timing: dict[str, float]) -> None:
    try:
        setattr(engine_core, "_decode_inner_timing_last_us", dict(timing))
    except Exception:
        return


def _maybe_install_model_runner_segment_timing(engine_core: Any) -> dict[str, float]:
    """[MR-SEGMENT-TIMING] in-proc 下再往下包一层 model_runner 关键段。

    ec_execute_model_submit 是黑盒总量（in-proc=同步整个 execute_model）；这里
    对 runner 的 _prepare_inputs / execute_model / sample_tokens 各包一层
    perf_counter，写进共享 dict，由 timed_step 并入 inner（mr_* 键）随
    step_engine_core_timing_all 落盘。诊断专用（INNER_TIMING 门控内），
    monkey-patch 只装一次，失败静默降级（返回空 dict 不影响原计时）。
    """
    seg: dict[str, float] = {}
    try:
        wrapper = engine_core.model_executor.driver_worker
        # UniProcExecutor.driver_worker 是 WorkerWrapperBase，真 Worker 在 .worker
        worker = getattr(wrapper, "worker", None) or wrapper
        runner = worker.model_runner
    except Exception as exc:
        print(f"[mr-seg-timing] install skipped: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return seg
    print(f"[mr-seg-timing] installed on {type(runner).__name__}", file=sys.stderr, flush=True)
    if bool(getattr(runner, "_mr_segment_timing_installed", False)):
        return getattr(runner, "_mr_segment_timing_shared", seg)

    def _wrap(name: str) -> bool:
        orig = getattr(runner, name, None)
        if not callable(orig):
            return False

        def timed(*args: Any, _orig: Any = orig, _key: str = f"mr_{name}_us", **kwargs: Any):
            t0 = time.perf_counter()
            try:
                return _orig(*args, **kwargs)
            finally:
                seg[_key] = seg.get(_key, 0.0) + _elapsed_us(t0)

        setattr(runner, name, timed)
        return True

    for method in ("_prepare_inputs", "sample_tokens"):
        _wrap(method)

    # [MR-GPU-CLOCK] execute_model 额外做 CUDA event 对钟，回答 host/GPU 重叠：
    # mr_gpu_span_us = 主流上本步首尾 event 跨度（含流内空隙）；
    # mr_gpu_done_at_return = host 返回时刻 GPU 是否已完成（1/0）。
    # 判读：span≈host 且 done=0 ⇒ GPU 主导（泡小）；span<<host 且 done=1 ⇒
    # host 尾部纯 CPU 段=可重叠泡。event 对轮转成对（上一步的 elapsed 在
    # 下一步开头收割，完成才读，零阻塞）。诊断门（INNER_TIMING）内。
    def _wrap_execute_with_gpu_clock() -> bool:
        orig = getattr(runner, "execute_model", None)
        if not callable(orig):
            return False
        try:
            import torch as _torch
        except Exception:
            return False
        if not _torch.cuda.is_available():
            return _wrap("execute_model")
        ring = [
            (_torch.cuda.Event(enable_timing=True), _torch.cuda.Event(enable_timing=True))
            for _ in range(2)
        ]
        slot_state = {"next": 0, "pending": None}

        def timed(*args: Any, **kwargs: Any):
            pend = slot_state["pending"]
            if pend is not None:
                p_start, p_end = pend
                if p_end.query():
                    try:
                        seg["mr_gpu_span_prev_us"] = float(
                            p_start.elapsed_time(p_end) * 1000.0
                        )
                    except Exception:
                        pass
                    slot_state["pending"] = None
            s_evt, e_evt = ring[slot_state["next"] % 2]
            slot_state["next"] += 1
            t0 = time.perf_counter()
            s_evt.record()
            try:
                return orig(*args, **kwargs)
            finally:
                e_evt.record()
                seg["mr_execute_model_us"] = (
                    seg.get("mr_execute_model_us", 0.0) + _elapsed_us(t0)
                )
                seg["mr_gpu_done_at_return"] = 1.0 if e_evt.query() else 0.0
                slot_state["pending"] = (s_evt, e_evt)

        setattr(runner, "execute_model", timed)
        return True

    _wrap_execute_with_gpu_clock()
    runner._mr_segment_timing_installed = True
    runner._mr_segment_timing_shared = seg
    return seg


def _maybe_install_engine_core_inner_timing(llm_engine: Any) -> None:
    if not _engine_core_inner_timing_enabled():
        return
    client = getattr(llm_engine, "engine_core", None)
    engine_core = getattr(client, "engine_core", None)
    if engine_core is None or bool(
        getattr(engine_core, "_decode_inner_timing_installed", False)
    ):
        return
    mr_seg = _maybe_install_model_runner_segment_timing(engine_core)

    def timed_step(self: Any):
        inner: dict[str, float] = {}
        total0 = time.perf_counter()
        try:
            has0 = time.perf_counter()
            has_requests = bool(self.scheduler.has_requests())
            inner["ec_has_requests_us"] = _elapsed_us(has0)
            if not has_requests:
                inner["ec_total_us"] = _elapsed_us(total0)
                _record_engine_core_timing(self, inner)
                return {}, False

            t0 = time.perf_counter()
            scheduler_output = self.scheduler.schedule()
            inner["ec_schedule_us"] = _elapsed_us(t0)

            t0 = time.perf_counter()
            future = self.model_executor.execute_model(
                scheduler_output, non_block=True
            )
            inner["ec_execute_model_submit_us"] = _elapsed_us(t0)

            t0 = time.perf_counter()
            grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
            inner["ec_grammar_bitmask_us"] = _elapsed_us(t0)

            with (
                self.log_error_detail(scheduler_output),
                self.log_iteration_details(scheduler_output),
            ):
                t0 = time.perf_counter()
                model_output = future.result()
                inner["ec_future_result_us"] = _elapsed_us(t0)
                if model_output is None:
                    t0 = time.perf_counter()
                    model_output = self.model_executor.sample_tokens(grammar_output)
                    inner["ec_sample_tokens_us"] = _elapsed_us(t0)
                else:
                    inner["ec_sample_tokens_us"] = 0.0

            t0 = time.perf_counter()
            self._process_aborts_queue()
            inner["ec_process_aborts_queue_us"] = _elapsed_us(t0)

            t0 = time.perf_counter()
            engine_core_outputs = self.scheduler.update_from_output(
                scheduler_output, model_output
            )
            inner["ec_update_from_output_us"] = _elapsed_us(t0)
            if mr_seg:
                inner.update(mr_seg)
                mr_seg.clear()
            inner["ec_total_us"] = _elapsed_us(total0)
            _record_engine_core_timing(self, inner)
            return (
                engine_core_outputs,
                scheduler_output.total_num_scheduled_tokens > 0,
            )
        except Exception:
            inner["ec_total_us"] = _elapsed_us(total0)
            _record_engine_core_timing(self, inner)
            raise

    def timed_step_with_batch_queue(self: Any):
        inner: dict[str, float] = {}
        total0 = time.perf_counter()
        try:
            batch_queue = self.batch_queue
            assert batch_queue is not None
            assert len(batch_queue) < self.batch_queue_size

            model_executed = False
            deferred_scheduler_output = None
            t0 = time.perf_counter()
            has_requests = bool(self.scheduler.has_requests())
            inner["ec_has_requests_us"] = _elapsed_us(t0)
            if has_requests:
                t0 = time.perf_counter()
                scheduler_output = self.scheduler.schedule()
                inner["ec_schedule_us"] = _elapsed_us(t0)
                with self.log_error_detail(scheduler_output):
                    t0 = time.perf_counter()
                    exec_future = self.model_executor.execute_model(
                        scheduler_output, non_block=True
                    )
                    inner["ec_execute_model_submit_us"] = _elapsed_us(t0)
                if self.is_ec_consumer:
                    model_executed = scheduler_output.total_num_scheduled_tokens > 0

                if self.is_pooling_model or not model_executed:
                    future = exec_future
                    inner["ec_grammar_bitmask_us"] = 0.0
                    inner["ec_sample_tokens_submit_us"] = 0.0
                else:
                    if not scheduler_output.pending_structured_output_tokens:
                        t0 = time.perf_counter()
                        grammar_output = self.scheduler.get_grammar_bitmask(
                            scheduler_output
                        )
                        inner["ec_grammar_bitmask_us"] = _elapsed_us(t0)
                        t0 = time.perf_counter()
                        future = self.model_executor.sample_tokens(
                            grammar_output, non_block=True
                        )
                        inner["ec_sample_tokens_submit_us"] = _elapsed_us(t0)
                    else:
                        deferred_scheduler_output = scheduler_output

                if not deferred_scheduler_output:
                    batch_queue.appendleft((future, scheduler_output, exec_future))
                    if (
                        model_executed
                        and len(batch_queue) < self.batch_queue_size
                        and not batch_queue[-1][0].done()
                    ):
                        inner["ec_returned_none_us"] = _elapsed_us(total0)
                        inner["ec_total_us"] = _elapsed_us(total0)
                        _record_engine_core_timing(self, inner)
                        return None, True

            elif not batch_queue:
                inner["ec_total_us"] = _elapsed_us(total0)
                _record_engine_core_timing(self, inner)
                return None, False

            t0 = time.perf_counter()
            future, scheduler_output, exec_model_fut = batch_queue.pop()
            inner["ec_batch_queue_pop_us"] = _elapsed_us(t0)
            with (
                self.log_error_detail(scheduler_output),
                self.log_iteration_details(scheduler_output),
            ):
                t0 = time.perf_counter()
                model_output = future.result()
                inner["ec_future_result_us"] = _elapsed_us(t0)
                if model_output is None:
                    t0 = time.perf_counter()
                    exec_model_fut.result()
                    inner["ec_exec_model_future_result_us"] = _elapsed_us(t0)
                    raise RuntimeError("unexpected error")

            t0 = time.perf_counter()
            self._process_aborts_queue()
            inner["ec_process_aborts_queue_us"] = _elapsed_us(t0)

            t0 = time.perf_counter()
            engine_core_outputs = self.scheduler.update_from_output(
                scheduler_output, model_output
            )
            inner["ec_update_from_output_us"] = _elapsed_us(t0)
            if mr_seg:
                inner.update(mr_seg)
                mr_seg.clear()

            if deferred_scheduler_output:
                if self.use_spec_decode:
                    t0 = time.perf_counter()
                    draft_token_ids = self.model_executor.take_draft_token_ids()
                    inner["ec_take_draft_token_ids_us"] = _elapsed_us(t0)
                    assert draft_token_ids is not None
                    t0 = time.perf_counter()
                    self.scheduler.update_draft_token_ids_in_output(
                        draft_token_ids, deferred_scheduler_output
                    )
                    inner["ec_update_draft_token_ids_us"] = _elapsed_us(t0)
                t0 = time.perf_counter()
                grammar_output = self.scheduler.get_grammar_bitmask(
                    deferred_scheduler_output
                )
                inner["ec_deferred_grammar_bitmask_us"] = _elapsed_us(t0)
                t0 = time.perf_counter()
                future = self.model_executor.sample_tokens(grammar_output, non_block=True)
                inner["ec_deferred_sample_tokens_submit_us"] = _elapsed_us(t0)
                batch_queue.appendleft(
                    (future, deferred_scheduler_output, exec_future)
                )

            inner["ec_total_us"] = _elapsed_us(total0)
            _record_engine_core_timing(self, inner)
            return engine_core_outputs, model_executed
        except Exception:
            inner["ec_total_us"] = _elapsed_us(total0)
            _record_engine_core_timing(self, inner)
            raise

    original_post_step = engine_core.post_step

    def timed_post_step(self: Any, model_executed: bool) -> None:
        t0 = time.perf_counter()
        try:
            return original_post_step(model_executed)
        finally:
            inner = dict(getattr(self, "_decode_inner_timing_last_us", {}) or {})
            inner["ec_post_step_us"] = _elapsed_us(t0)
            inner["ec_total_plus_post_us"] = float(
                inner.get("ec_total_us", 0.0) + inner["ec_post_step_us"]
            )
            _record_engine_core_timing(self, inner)

    engine_core.step = types.MethodType(timed_step, engine_core)
    engine_core.step_with_batch_queue = types.MethodType(
        timed_step_with_batch_queue, engine_core
    )
    engine_core.post_step = types.MethodType(timed_post_step, engine_core)
    engine_core.step_fn = (
        engine_core.step
        if getattr(engine_core, "batch_queue", None) is None
        else engine_core.step_with_batch_queue
    )
    setattr(engine_core, "_decode_inner_timing_installed", True)


def pull_step_outputs_with_timing(llm_engine) -> tuple[list[Any], float, dict[str, float]]:
    timing = {
        "dummy_batch_us": 0.0,
        "get_output_us": 0.0,
        "process_outputs_us": 0.0,
        "abort_requests_us": 0.0,
    }
    # Keep parity with LLMEngine.step(): execute dummy batch once if requested.
    if bool(getattr(llm_engine, "should_execute_dummy_batch", False)):
        llm_engine.should_execute_dummy_batch = False
        t0 = time.perf_counter()
        llm_engine.engine_core.execute_dummy_batch()
        t1 = time.perf_counter()
        timing["dummy_batch_us"] = float((t1 - t0) * 1_000_000.0)
        return [], float("nan"), timing

    _maybe_install_engine_core_inner_timing(llm_engine)
    t0 = time.perf_counter()
    outputs = llm_engine.engine_core.get_output()
    t1 = time.perf_counter()
    if bool(
        getattr(llm_engine, "_sfi_cudagraph_runtime_observer_enabled", False)
    ):
        scheduler_stats = getattr(outputs, "scheduler_stats", None)
        cudagraph_stats = getattr(scheduler_stats, "cudagraph_stats", None)
        runtime_record: dict[str, object] | None = None
        if cudagraph_stats is not None:
            runtime_record = {
                "num_unpadded_tokens": int(
                    getattr(cudagraph_stats, "num_unpadded_tokens")
                ),
                "num_padded_tokens": int(
                    getattr(cudagraph_stats, "num_padded_tokens")
                ),
                "num_paddings": int(getattr(cudagraph_stats, "num_paddings")),
                "runtime_mode": _runtime_mode_name(
                    getattr(cudagraph_stats, "runtime_mode")
                ),
            }
        records = getattr(llm_engine, "_sfi_cudagraph_runtime_records", None)
        if not isinstance(records, list):
            raise RuntimeError(
                "E_CUDAGRAPH_RUNTIME_OBSERVER_UNAVAILABLE: record sink"
            )
        records.append(runtime_record)
    engine_core = getattr(llm_engine.engine_core, "engine_core", None)
    inner_timing = getattr(engine_core, "_decode_inner_timing_last_us", None)
    if isinstance(inner_timing, dict):
        for key, value in inner_timing.items():
            if key.startswith("ec_") or key.startswith("mr_"):
                timing[str(key)] = float(value)
    processed_outputs = llm_engine.output_processor.process_outputs(
        outputs.outputs,
        engine_core_timestamp=float(outputs.timestamp),
        iteration_stats=None,
    )
    t2 = time.perf_counter()
    llm_engine.engine_core.abort_requests(processed_outputs.reqs_to_abort)
    t3 = time.perf_counter()
    timing["get_output_us"] = float((t1 - t0) * 1_000_000.0)
    timing["process_outputs_us"] = float((t2 - t1) * 1_000_000.0)
    timing["abort_requests_us"] = float((t3 - t2) * 1_000_000.0)
    return list(processed_outputs.request_outputs), float(outputs.timestamp), timing


def pull_step_outputs_with_timestamp(llm_engine) -> tuple[list[Any], float]:
    request_outputs, timestamp, _timing = pull_step_outputs_with_timing(llm_engine)
    return request_outputs, timestamp
