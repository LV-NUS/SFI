from __future__ import annotations

import atexit
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Tuple

from utils.model_kv_contract import MODEL_KV_CONTRACT_SCHEMA


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
CUDAGRAPH_RUNTIME_OBSERVER_SCOPE = (
    "diagnostic_measurement_post_anchor_partitioned"
)


def collect_single_token_decode_step_proof(engine: object) -> dict[str, object]:
    """Prove the aggregate synchronization anchor is one-token-per-request.

    This is evaluated once after engine construction.  It keeps speculative
    decoding or buffered multi-step output from invalidating the zero-overhead
    ``new_tokens == batch_size`` full-output anchor.
    """
    llm_engine = getattr(engine, "llm_engine", None)
    vllm_config = getattr(llm_engine, "vllm_config", None)
    if vllm_config is None:
        raise RuntimeError(
            "E_DECODE_STEP_CONTRACT_UNAVAILABLE: vllm_config missing"
        )
    speculative_config = getattr(vllm_config, "speculative_config", None)
    scheduler_config = getattr(vllm_config, "scheduler_config", None)
    stream_interval = getattr(scheduler_config, "stream_interval", None)
    if speculative_config is not None:
        raise RuntimeError(
            "E_DECODE_STEP_CONTRACT_SPECULATIVE: speculative decoding is "
            "incompatible with the aggregate full-output anchor"
        )
    if type(stream_interval) is not int or stream_interval != 1:
        raise RuntimeError(
            "E_DECODE_STEP_CONTRACT_STREAM_INTERVAL: expected 1, got "
            f"{stream_interval!r}"
        )
    return {
        "decode_step_contract_schema": "sfi.single_token_decode_step.v1",
        "decode_step_contract_proof_passed": True,
        "decode_step_speculative_config_present": False,
        "decode_step_stream_interval": 1,
        "decode_step_max_tokens_per_request": 1,
    }


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
        "fi_ar_comm_active": fi_ar_active,
        "device_communicator_use_torch_symm_mem": bool(
            getattr(device_communicator, "use_torch_symm_mem", False)
        ),
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


def _attach_selector_log_s_runtime_proof(
    record: dict[str, object],
) -> dict[str, object]:
    from patches.fa3_native.postprocess import (
        snapshot_selector_log_s_runtime_proof,
    )

    record["selector_log_s_runtime_proof"] = (
        snapshot_selector_log_s_runtime_proof()
    )
    return record


_RUNNER_ENGINE_CONTRACT_ENV_FIELDS = (
    "SFI_RUNNER_MODEL_KV_CONTRACT_SCHEMA",
    "SFI_RUNNER_CHAT_TEMPLATE_RESERVE_TOKENS",
    "SFI_RUNNER_KV_TOKEN_BYTES_PER_RANK_EFFECTIVE",
)


def runner_engine_runtime_contract_required() -> bool:
    """Return whether this child belongs to a launcher-managed proof run.

    The model/KV contract is emitted for every ``run_speed.sh`` tier.  Treat
    its presence as the common cold-start proof boundary instead of using a
    workload label to change child behavior.  A partially propagated contract
    is invalid rather than silently disabling the proof.
    """

    values = {
        name: str(os.environ.get(name, "") or "").strip()
        for name in _RUNNER_ENGINE_CONTRACT_ENV_FIELDS
    }
    if not any(values.values()):
        return False
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise RuntimeError(
            "E_ENGINE_RUNTIME_CONTRACT_INPUT: incomplete runner contract: "
            f"missing={missing!r}"
        )
    if values["SFI_RUNNER_MODEL_KV_CONTRACT_SCHEMA"] != (
        MODEL_KV_CONTRACT_SCHEMA
    ):
        raise RuntimeError(
            "E_ENGINE_RUNTIME_CONTRACT_INPUT: unsupported model/KV schema: "
            f"actual={values['SFI_RUNNER_MODEL_KV_CONTRACT_SCHEMA']!r}:"
            f"expected={MODEL_KV_CONTRACT_SCHEMA!r}"
        )
    return True


def _runtime_mode_name(value: object) -> str:
    """Normalize enum/string runtime modes without depending on vLLM internals."""
    name = getattr(value, "name", None)
    if isinstance(name, str) and name:
        return name.upper()
    return str(value).rsplit(".", 1)[-1].strip().upper()


def _required_decode_dispatch_state(
    dispatcher: object,
    required_batch_size: int,
) -> dict[str, object]:
    """Resolve one required decode shape through the dispatcher's owner API."""
    raw_keys = getattr(dispatcher, "cudagraph_keys", None)
    if not isinstance(raw_keys, dict):
        raise RuntimeError(
            "E_ENGINE_RUNTIME_CONTRACT_UNAVAILABLE: cudagraph_keys"
        )
    dispatch = getattr(dispatcher, "dispatch", None)
    if not callable(dispatch):
        raise RuntimeError(
            "E_ENGINE_RUNTIME_CONTRACT_UNAVAILABLE: cudagraph dispatch"
        )
    try:
        resolved_mode, resolved_descriptor = dispatch(
            num_tokens=required_batch_size,
            uniform_decode=True,
            has_lora=False,
            num_active_loras=0,
        )
    except Exception as exc:
        raise RuntimeError(
            "E_ENGINE_RUNTIME_CONTRACT_DISPATCH_FAILED: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    resolved_keys = raw_keys.get(resolved_mode)
    return {
        "num_tokens": required_batch_size,
        "uniform_decode": True,
        "has_lora": False,
        "num_active_loras": 0,
        "runtime_mode": _runtime_mode_name(resolved_mode),
        "batch_descriptor": {
            "num_tokens": int(getattr(resolved_descriptor, "num_tokens")),
            "num_reqs": (
                None
                if getattr(resolved_descriptor, "num_reqs", None) is None
                else int(getattr(resolved_descriptor, "num_reqs"))
            ),
            "uniform": bool(getattr(resolved_descriptor, "uniform", False)),
            "has_lora": bool(getattr(resolved_descriptor, "has_lora", False)),
            "num_active_loras": int(
                getattr(resolved_descriptor, "num_active_loras", 0) or 0
            ),
        },
        "key_registered": bool(
            resolved_keys is not None and resolved_descriptor in resolved_keys
        ),
    }


def _runner_kv_storage_geometry(
    runner_kv_caches: list[object],
) -> dict[str, object]:
    """Describe active KV tensors and deduplicated physical storages."""
    from collections import Counter

    tensor_signatures: Counter[tuple[object, ...]] = Counter()
    storages: dict[tuple[str, int], int] = {}
    for tensor in runner_kv_caches:
        element_size = int(getattr(tensor, "element_size")())
        numel = int(getattr(tensor, "numel")())
        tensor_signatures[
            (
                tuple(int(value) for value in getattr(tensor, "shape", ())),
                str(getattr(tensor, "dtype", "")),
                element_size,
                numel,
            )
        ] += 1
        storage = getattr(tensor, "untyped_storage")()
        storage_key = (
            str(getattr(storage, "device", "")),
            int(getattr(storage, "data_ptr")()),
        )
        storage_nbytes = int(getattr(storage, "nbytes")())
        previous_nbytes = storages.setdefault(storage_key, storage_nbytes)
        if previous_nbytes != storage_nbytes:
            raise RuntimeError(
                "E_ENGINE_RUNTIME_CONTRACT_STORAGE_IDENTITY: "
                f"{previous_nbytes}!={storage_nbytes}"
            )
    storage_signatures = Counter(storages.values())
    return {
        "runner_tensor_count": len(runner_kv_caches),
        "runner_tensor_signatures": [
            {
                "shape": list(key[0]),
                "dtype": key[1],
                "element_size": key[2],
                "numel": key[3],
                "count": count,
            }
            for key, count in sorted(tensor_signatures.items(), key=repr)
        ],
        "runner_storage_count": len(storages),
        "runner_storage_signatures": [
            {"nbytes": nbytes, "count": count}
            for nbytes, count in sorted(storage_signatures.items())
        ],
        "runner_storage_bytes": sum(storages.values()),
    }


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
    runner_contract_required = runner_engine_runtime_contract_required()
    mismatches = []
    if runner_contract_required and not parent_model_sha256:
        mismatches.append("model_config_parent_claim_missing")
    if runner_contract_required and not expected_corpus_sha256:
        mismatches.append("corpus_sha256_parent_claim_missing")
    if (
        expected_model_sha256
        and model_config_sha256 != expected_model_sha256
    ):
        mismatches.append(
            "model_config_sha256="
            f"{model_config_sha256!r}:expected={expected_model_sha256!r}"
        )
    if parent_model_sha256 and model_config_sha256 != parent_model_sha256:
        mismatches.append(
            "model_config_parent_claim="
            f"{parent_model_sha256!r}:actual={model_config_sha256!r}"
        )
    if expected_corpus_sha256 and corpus_sha256 != expected_corpus_sha256:
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
    required_tokens_by_request: list[int] | tuple[int, ...],
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
    if not isinstance(required_tokens_by_request, (list, tuple)):
        raise RuntimeError(
            "E_ENGINE_RUNTIME_CONTRACT_INPUT: "
            "required_tokens_by_request must be a sequence"
        )
    required_tokens = tuple(required_tokens_by_request)
    if len(required_tokens) != required_batch_size or any(
        type(value) is not int or value <= 0 for value in required_tokens
    ):
        raise RuntimeError(
            "E_ENGINE_RUNTIME_CONTRACT_INPUT: required_tokens_by_request="
            f"{required_tokens!r}:required_batch_size={required_batch_size}"
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
    required_decode_dispatch = _required_decode_dispatch_state(
        dispatcher,
        required_batch_size,
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
    tensor_owner_count = sum(
        len(getattr(tensor, "shared_by", [])) for tensor in kv_cache_tensors
    )
    configured_bytes = sum(
        int(getattr(tensor, "size")) for tensor in kv_cache_tensors
    )
    block_size = next(iter(group_block_sizes)) if len(group_block_sizes) == 1 else 0
    allocated_token_slots = num_blocks * block_size if block_size > 0 else 0
    required_workload_tokens = sum(required_tokens)
    required_blocks_by_request = tuple(
        (value + block_size - 1) // block_size
        if block_size > 0
        else 0
        for value in required_tokens
    )
    required_workload_blocks = sum(required_blocks_by_request)
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
    runner_storage = _runner_kv_storage_geometry(runner_kv_caches)
    runner_storage_bytes = int(runner_storage["runner_storage_bytes"])
    actual_kv_bytes_per_token = (
        runner_storage_bytes // allocated_token_slots
        if allocated_token_slots > 0
        and runner_storage_bytes % allocated_token_slots == 0
        else 0
    )
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
        "tensor_owner_count": tensor_owner_count,
        "tensor_signatures": [
            {"size": key[0], "shared_by_count": key[1], "count": count}
            for key, count in sorted(tensor_signatures.items())
        ],
        "configured_bytes": configured_bytes,
        "allocated_token_slots": allocated_token_slots,
        **runner_storage,
        "configured_allocation_matches_runner_storage": bool(
            configured_bytes == runner_storage_bytes > 0
        ),
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
        "required_tokens_by_request": list(required_tokens),
        "required_workload_tokens": required_workload_tokens,
        "required_blocks_by_request": list(required_blocks_by_request),
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
        "dispatcher_required_decode_dispatch": required_decode_dispatch,
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
        "kv_cache_tensor_owner_count": tensor_owner_count,
        "kv_cache_tensor_signatures": [
            {"size": key[0], "shared_by_count": key[1], "count": count}
            for key, count in sorted(tensor_signatures.items())
        ],
        "kv_cache_configured_bytes": configured_bytes,
        "kv_cache_runner_tensor_count": runner_storage[
            "runner_tensor_count"
        ],
        "kv_cache_runner_tensor_signatures": runner_storage[
            "runner_tensor_signatures"
        ],
        "kv_cache_runner_storage_count": runner_storage[
            "runner_storage_count"
        ],
        "kv_cache_runner_storage_signatures": runner_storage[
            "runner_storage_signatures"
        ],
        "kv_cache_runner_storage_bytes": runner_storage_bytes,
        "kv_cache_configured_allocation_matches_runner_storage": bool(
            configured_bytes == runner_storage_bytes > 0
        ),
        "kv_cache_cross_layers_tensor": cross_layers_record,
        "kv_cache_actual_bytes_per_token": actual_kv_bytes_per_token,
        "kv_cache_allocated_token_slots": allocated_token_slots,
        "kv_cache_schedulable_tokens": schedulable_tokens,
        "kv_cache_null_block_count": null_block_count,
        "kv_cache_ordinary_blocks": ordinary_blocks,
        "kv_cache_required_batch_size": required_batch_size,
        "kv_cache_required_tokens_by_request": list(required_tokens),
        "kv_cache_required_workload_tokens": required_workload_tokens,
        "kv_cache_required_blocks_by_request": list(
            required_blocks_by_request
        ),
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
        required_tokens_by_request: list[int],
        compact_blocks_per_slot: int,
        compact_generation_count: int,
        expected_kv_bytes_per_token: int,
    ) -> dict[str, object]:
        return _worker_engine_runtime_contract_state(
            self,
            required_batch_size,
            required_tokens_by_request,
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
        return _attach_selector_log_s_runtime_proof(
            reset_rank_local_route_counter_slot_for_measurement()
        )

    def sfi_snapshot_sparse_route_counters_after_measurement(
        self,
    ) -> dict[str, object]:
        from patches.fa3_native.install import (
            snapshot_rank_local_route_counter_slot_after_measurement,
        )

        return _attach_selector_log_s_runtime_proof(
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
        "device_communicator_use_torch_symm_mem",
        "fi_ar_comm_active",
        "symm_mem_comm_active",
        "vllm_allreduce_use_flashinfer",
        "vllm_allreduce_use_symm_mem",
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
            flashinfer_fields = (
                record["device_communicator_use_flashinfer_allreduce"],
                record["fi_ar_comm_active"],
                record["vllm_allreduce_use_flashinfer"],
            )
            torch_symm_mem_fields = (
                record["device_communicator_use_torch_symm_mem"],
                record["symm_mem_comm_active"],
                record["vllm_allreduce_use_symm_mem"],
            )
            nccl_symm_mem_raw = record["vllm_use_nccl_symm_mem"]
            if not all(isinstance(value, bool) for value in flashinfer_fields):
                errors.append(
                    f"rank{rank}.flashinfer_state={flashinfer_fields!r}"
                )
            if not all(isinstance(value, bool) for value in torch_symm_mem_fields):
                errors.append(
                    f"rank{rank}.torch_symm_mem_state={torch_symm_mem_fields!r}"
                )
            if not isinstance(nccl_symm_mem_raw, bool):
                errors.append(
                    f"rank{rank}.vllm_use_nccl_symm_mem={nccl_symm_mem_raw!r}"
                )
            flashinfer_enabled = bool(
                all(isinstance(value, bool) for value in flashinfer_fields)
                and any(flashinfer_fields)
            )
            torch_symm_mem_enabled = bool(
                all(isinstance(value, bool) for value in torch_symm_mem_fields)
                and any(torch_symm_mem_fields)
            )
            nccl_symm_mem_enabled = bool(
                isinstance(nccl_symm_mem_raw, bool) and nccl_symm_mem_raw
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
            if expected_active and torch_symm_mem_enabled:
                errors.append(
                    f"rank{rank}.torch_symm_mem_preempts_custom_all_reduce"
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
            or record["vllm_allreduce_use_flashinfer"]
        )
        for record in records
    )
    torch_symm_mem_preemptor_enabled = any(
        bool(
            record["device_communicator_use_torch_symm_mem"]
            or record["symm_mem_comm_active"]
            or record["vllm_allreduce_use_symm_mem"]
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
        "custom_all_reduce_runtime_preemptor_torch_symm_mem_enabled": (
            torch_symm_mem_preemptor_enabled
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
    required_tokens_by_request: list[int] | tuple[int, ...],
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
    required_tokens = tuple(required_tokens_by_request)
    if len(required_tokens) != int(required_batch_size) or any(
        type(value) is not int or value <= 0 for value in required_tokens
    ):
        raise RuntimeError(
            "E_ENGINE_RUNTIME_CONTRACT_INPUT: required_tokens_by_request="
            f"{required_tokens!r}:required_batch_size={required_batch_size}"
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
                list(required_tokens),
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
        decode_dispatch = record.get("dispatcher_required_decode_dispatch")
        if not isinstance(decode_dispatch, dict):
            errors.append(f"{rank_prefix}.required_decode_dispatch_invalid")
        else:
            expected_request = {
                "num_tokens": int(required_batch_size),
                "uniform_decode": True,
                "has_lora": False,
                "num_active_loras": 0,
            }
            for key, expected in expected_request.items():
                if decode_dispatch.get(key) != expected:
                    errors.append(
                        f"{rank_prefix}.required_decode_dispatch_{key}="
                        f"{decode_dispatch.get(key)!r}:expected={expected!r}"
                    )
            if decode_dispatch.get("runtime_mode") != "FULL":
                errors.append(
                    f"{rank_prefix}.required_decode_runtime_mode="
                    f"{decode_dispatch.get('runtime_mode')!r}"
                )
            if decode_dispatch.get("key_registered") is not True:
                errors.append(
                    f"{rank_prefix}.required_decode_key_not_registered"
                )
            resolved_descriptor = decode_dispatch.get("batch_descriptor")
            if (
                not isinstance(resolved_descriptor, dict)
                or resolved_descriptor.get("num_tokens") != required_batch_size
            ):
                errors.append(
                    f"{rank_prefix}.required_decode_descriptor="
                    f"{resolved_descriptor!r}"
                )

        numeric_expectations = {
            "kv_cache_required_batch_size": required_batch_size,
            "kv_cache_compact_blocks_per_slot": compact_blocks_per_slot,
            "kv_cache_compact_generation_count": compact_generation_count,
            "kv_cache_expected_bytes_per_token": expected_kv_bytes_per_token,
        }
        for key, expected in numeric_expectations.items():
            if record.get(key) != expected:
                errors.append(
                    f"{rank_prefix}.{key}={record.get(key)!r}:expected={expected}"
                )
        if record.get("kv_cache_required_tokens_by_request") != list(
            required_tokens
        ):
            errors.append(
                f"{rank_prefix}.kv_cache_required_tokens_by_request="
                f"{record.get('kv_cache_required_tokens_by_request')!r}:"
                f"expected={list(required_tokens)!r}"
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
            "kv_cache_configured_bytes",
            "kv_cache_allocated_token_slots",
            "kv_cache_schedulable_tokens",
            "kv_cache_tensor_count",
            "kv_cache_tensor_owner_count",
            "kv_cache_runner_tensor_count",
            "kv_cache_runner_storage_count",
            "kv_cache_runner_storage_bytes",
            "kv_cache_required_workload_blocks",
            "kv_cache_required_total_blocks",
        ):
            value = record.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                errors.append(f"{rank_prefix}.{key}={value!r}")
        required_blocks_by_request = record.get(
            "kv_cache_required_blocks_by_request"
        )
        if (
            not isinstance(required_blocks_by_request, list)
            or len(required_blocks_by_request) != int(required_batch_size)
            or any(
                type(value) is not int or value <= 0
                for value in required_blocks_by_request
            )
        ):
            errors.append(
                f"{rank_prefix}.kv_cache_required_blocks_by_request="
                f"{required_blocks_by_request!r}"
            )
        actual_bytes_per_token = record.get("kv_cache_actual_bytes_per_token")
        if actual_bytes_per_token != expected_kv_bytes_per_token:
            errors.append(
                f"{rank_prefix}.kv_cache_actual_bytes_per_token="
                f"{actual_bytes_per_token!r}:expected={expected_kv_bytes_per_token}"
            )
        configured_bytes = record.get("kv_cache_configured_bytes")
        runner_storage_bytes = record.get("kv_cache_runner_storage_bytes")
        num_blocks = record.get("kv_cache_num_blocks")
        block_size = record.get("kv_cache_uniform_block_size")
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
        if record.get(
            "kv_cache_configured_allocation_matches_runner_storage"
        ) is not True:
            errors.append(
                f"{rank_prefix}.kv_cache_configured_runner_storage_mismatch:"
                f"configured={configured_bytes!r}:"
                f"runner_storage={runner_storage_bytes!r}"
            )
        if configured_bytes != runner_storage_bytes:
            errors.append(
                f"{rank_prefix}.kv_cache_storage_identity="
                f"{configured_bytes!r}!={runner_storage_bytes!r}"
            )
        if all(
            type(value) is int
            for value in (
                runner_storage_bytes,
                num_blocks,
                block_size,
                expected_kv_bytes_per_token,
            )
        ):
            expected_storage_bytes = (
                num_blocks * block_size * expected_kv_bytes_per_token
            )
            if runner_storage_bytes != expected_storage_bytes:
                errors.append(
                    f"{rank_prefix}.kv_cache_model_geometry_identity="
                    f"{runner_storage_bytes}!={num_blocks}*{block_size}*"
                    f"{expected_kv_bytes_per_token}"
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
        if (
            isinstance(cross_layers, dict)
            and cross_layers.get("nbytes") != runner_storage_bytes
        ):
            errors.append(
                f"{rank_prefix}.cross_layers_nbytes="
                f"{cross_layers.get('nbytes')!r}:"
                f"expected={runner_storage_bytes!r}"
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
            "dispatcher_required_decode_dispatch": record.get(
                "dispatcher_required_decode_dispatch"
            ),
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
        "engine_runtime_graph_required_decode_dispatch_passed": True,
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
        "engine_runtime_kv_tensor_owner_count": reference.get(
            "kv_cache_tensor_owner_count"
        ),
        "engine_runtime_kv_configured_bytes": reference.get(
            "kv_cache_configured_bytes"
        ),
        "engine_runtime_kv_runner_storage_bytes": reference.get(
            "kv_cache_runner_storage_bytes"
        ),
        "engine_runtime_kv_allocation_identity_passed": True,
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
        "engine_runtime_kv_required_blocks_by_request": reference.get(
            "kv_cache_required_blocks_by_request"
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
    """Measure decode throughput after first_emit_ts through last_emit_ts.

    When *batch_size* is provided, an additional **all-decode window** is
    tracked.  The first output where ``new_tokens == batch_size`` is the
    synchronization anchor proving that every request has emitted.  Measured
    tokens, intervals, and engine records start *after* that anchor and end at
    the last emit.  This excludes the final mixed-prefill execution that can
    produce one token for every request without being a BS-sized decode step.
    """

    batch_size: int = 0
    first_emit_ts: float | None = None
    last_emit_ts: float | None = None
    total_tokens: int = 0
    first_emit_tokens: int = 0
    decode_step_durations_s: list[float] = field(default_factory=list)
    # All-decode measurement intervals after the first full-output anchor.
    _ad_start_ts: float | None = None
    _ad_start_tokens: int = 0
    _ad_full_batch_steps: int = 0
    _ad_partial_batch_steps: int = 0
    _ad_zero_token_steps: int = 0

    def observe(self, ts: float, new_tokens: int) -> None:
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
            self.first_emit_tokens = tokens
        else:
            dt = timestamp - self.last_emit_ts  # type: ignore[operator]
            if dt < 0.0:
                dt = 0.0
            self.decode_step_durations_s.append(float(dt))
            self.last_emit_ts = timestamp

        # The first full-output event closes the prefill-interleaved prefix.
        # It is the timing anchor, not a measured decode step: its work and
        # tokens precede the interval whose throughput starts at this timestamp.
        if self._ad_start_ts is None:
            if self.batch_size > 0 and tokens == self.batch_size:
                self._ad_start_ts = timestamp
                self._ad_start_tokens = self.total_tokens
            return
        if tokens == self.batch_size:
            self._ad_full_batch_steps += 1
        else:
            self._ad_partial_batch_steps += 1

    def finalize(self) -> Tuple[float, int, list[float]]:
        window_tokens = max(0, int(self.total_tokens) - int(self.first_emit_tokens))
        if self.first_emit_ts is None or self.last_emit_ts is None:
            return 0.0, int(window_tokens), self.decode_step_durations_s

        decode_elapsed_s = float(self.last_emit_ts - self.first_emit_ts)
        if decode_elapsed_s <= 0.0:
            return 0.0, int(window_tokens), self.decode_step_durations_s

        return float(decode_elapsed_s), int(window_tokens), self.decode_step_durations_s

    def boundary_delays(self, total_start_s: float, total_end_s: float) -> Tuple[float, float]:
        if self.first_emit_ts is None or self.last_emit_ts is None:
            return float("nan"), float("nan")
        first_emit_delay_s = max(0.0, float(self.first_emit_ts) - float(total_start_s))
        post_decode_tail_s = max(0.0, float(total_end_s) - float(self.last_emit_ts))
        return first_emit_delay_s, post_decode_tail_s

    def finalize_all_decode(self) -> Tuple[float, int, float, int]:
        """Return measured post-anchor (elapsed_s, tokens, tok_per_s, steps).

        The all-decode window starts after a strict synchronization anchor.
        Returning the ordinary first-emit window here would silently mix
        chunked prefill into the reported throughput, so a run that never
        reaches the anchor is invalid instead of having a fallback value.
        """
        if self._ad_start_ts is None or self.last_emit_ts is None:
            raise RuntimeError(
                "E_ALL_DECODE_WINDOW_NOT_ENTERED: no full-output "
                "synchronization anchor was observed"
            )
        ad_elapsed = float(self.last_emit_ts - self._ad_start_ts)
        ad_tokens = max(0, int(self.total_tokens) - int(self._ad_start_tokens))
        if ad_elapsed <= 0.0 or ad_tokens <= 0:
            raise RuntimeError(
                "E_ALL_DECODE_WINDOW_EMPTY: the full-output anchor was "
                "observed without a measurable later decode interval"
            )
        ad_tps = float(ad_tokens) / float(ad_elapsed)
        measured_steps = self._ad_full_batch_steps + self._ad_partial_batch_steps
        return ad_elapsed, ad_tokens, ad_tps, int(measured_steps)

    def all_decode_contract(self) -> dict[str, object]:
        """Return strict post-anchor measurement-window evidence."""
        entered = self._ad_start_ts is not None
        return {
            "all_decode_entered": entered,
            "all_decode_full_batch_steps": int(self._ad_full_batch_steps),
            "all_decode_partial_batch_steps": int(self._ad_partial_batch_steps),
            "all_decode_zero_token_steps": int(self._ad_zero_token_steps),
        }


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


def cudagraph_runtime_observer_proof_reasons(
    proof: object,
    *,
    expected_batch_size: int,
) -> list[str]:
    """Validate the cold graph-observer artifact from one shared contract."""
    if type(expected_batch_size) is not int or expected_batch_size <= 0:
        raise ValueError(
            f"expected_batch_size must be positive, got {expected_batch_size!r}"
        )
    if not isinstance(proof, dict):
        return ["proof_missing_or_invalid"]

    reasons: list[str] = []
    exact: dict[str, object] = {
        "cudagraph_runtime_observer_scope": CUDAGRAPH_RUNTIME_OBSERVER_SCOPE,
        "cudagraph_runtime_observer_enabled": True,
        "cudagraph_runtime_observer_full_batch_missing_step_count": 0,
        "cudagraph_runtime_observer_partial_batch_missing_step_count": 0,
    }
    reasons.extend(
        f"field_mismatch:{field}"
        for field, expected in exact.items()
        if type(proof.get(field)) is not type(expected)
        or proof.get(field) != expected
    )

    total_steps = proof.get("cudagraph_runtime_observer_total_step_count")
    full_steps = proof.get(
        "cudagraph_runtime_observer_full_batch_decode_step_count"
    )
    partial_steps = proof.get(
        "cudagraph_runtime_observer_partial_batch_decode_step_count"
    )
    if type(total_steps) is not int or total_steps <= 0:
        reasons.append("total_step_count_invalid")
    if type(full_steps) is not int or full_steps <= 0:
        reasons.append("full_batch_step_count_invalid")
    if type(partial_steps) is not int or partial_steps < 0:
        reasons.append("partial_batch_step_count_invalid")
    if (
        type(total_steps) is int
        and type(full_steps) is int
        and type(partial_steps) is int
        and full_steps + partial_steps >= total_steps
    ):
        reasons.append("post_anchor_step_count_not_strict_suffix")
    full_distribution = proof.get(
        "cudagraph_runtime_observer_full_batch_distribution"
    )
    expected_full_distribution = [
        {
            "num_unpadded_tokens": expected_batch_size,
            "num_padded_tokens": expected_batch_size,
            "num_paddings": 0,
            "runtime_mode": "FULL",
            "count": full_steps,
        }
    ]
    if full_distribution != expected_full_distribution:
        reasons.append("full_batch_distribution_mismatch")

    partial_distribution = proof.get(
        "cudagraph_runtime_observer_partial_batch_distribution"
    )
    if not isinstance(partial_distribution, list):
        reasons.append("partial_batch_distribution_invalid")
    else:
        observed_partial_steps = 0
        partial_distribution_valid = True
        for record in partial_distribution:
            if not isinstance(record, dict):
                partial_distribution_valid = False
                continue
            unpadded = record.get("num_unpadded_tokens")
            padded = record.get("num_padded_tokens")
            paddings = record.get("num_paddings")
            count = record.get("count")
            if (
                type(unpadded) is not int
                or not 0 < unpadded < expected_batch_size
                or type(padded) is not int
                or padded != expected_batch_size
                or type(paddings) is not int
                or paddings != expected_batch_size - unpadded
                or record.get("runtime_mode") != "FULL"
                or type(count) is not int
                or count <= 0
            ):
                partial_distribution_valid = False
                continue
            observed_partial_steps += count
        if (
            not partial_distribution_valid
            or observed_partial_steps != partial_steps
        ):
            reasons.append("partial_batch_distribution_mismatch")
    return list(dict.fromkeys(reasons))


def summarize_cudagraph_runtime_observer(
    records: list[dict[str, object] | None],
    *,
    expected_full_batch_steps: int,
    expected_partial_batch_steps: int,
    expected_zero_token_steps: int,
    expected_batch_size: int,
    expected_total_engine_steps: int,
) -> dict[str, object]:
    """Validate measured full-batch and partial-tail dispatch separately."""
    from collections import Counter

    if len(records) != int(expected_total_engine_steps):
        raise RuntimeError(
            "E_CUDAGRAPH_RUNTIME_OBSERVER_STEP_MISMATCH: "
            f"records={len(records)} expected={expected_total_engine_steps}"
        )
    full_batch_steps = int(expected_full_batch_steps)
    if full_batch_steps <= 0:
        raise RuntimeError(
            "E_CUDAGRAPH_RUNTIME_OBSERVER_FULL_BATCH_WINDOW: "
            f"full_batch_steps={full_batch_steps} records={len(records)}"
        )
    partial_batch_steps = int(expected_partial_batch_steps)
    if partial_batch_steps < 0:
        raise RuntimeError(
            "E_CUDAGRAPH_RUNTIME_OBSERVER_PARTIAL_BATCH_WINDOW: "
            f"partial_batch_steps={partial_batch_steps} records={len(records)}"
        )
    zero_token_steps = int(expected_zero_token_steps)
    if zero_token_steps != 0:
        raise RuntimeError(
            "E_CUDAGRAPH_RUNTIME_OBSERVER_ZERO_TOKEN_WINDOW: "
            f"zero_token_steps={zero_token_steps}"
        )
    measured_steps = full_batch_steps + partial_batch_steps
    start = len(records) - measured_steps
    if start <= 0 or start >= len(records):
        raise RuntimeError(
            "E_CUDAGRAPH_RUNTIME_OBSERVER_POST_ANCHOR_WINDOW: "
            f"records={len(records)} measured_steps={measured_steps}"
        )
    end = start + full_batch_steps
    tail_end = end + partial_batch_steps
    # After the synchronization anchor, requests only leave this fixed offline
    # batch.  Full-batch decode is therefore a prefix; the remaining records
    # are a legal partial tail outside the exact BS-sized graph proof.
    full_batch_window = records[start:end]
    partial_batch_window = records[end:tail_end]

    def _distribution(
        window: list[dict[str, object] | None],
    ) -> Counter[tuple[object, ...]]:
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
        return distribution

    full_batch_missing = sum(record is None for record in full_batch_window)
    partial_batch_missing = sum(record is None for record in partial_batch_window)
    full_batch_distribution = _distribution(full_batch_window)
    partial_batch_distribution = _distribution(partial_batch_window)

    def _serialized_distribution(
        distribution: Counter[tuple[object, ...]],
    ) -> list[dict[str, object]]:
        return [
            {
                "num_unpadded_tokens": signature[0],
                "num_padded_tokens": signature[1],
                "num_paddings": signature[2],
                "runtime_mode": signature[3],
                "count": count,
            }
            for signature, count in sorted(distribution.items(), key=repr)
        ]

    return {
        "cudagraph_runtime_observer_scope": CUDAGRAPH_RUNTIME_OBSERVER_SCOPE,
        "cudagraph_runtime_observer_enabled": True,
        "cudagraph_runtime_observer_total_step_count": len(records),
        "cudagraph_runtime_observer_full_batch_decode_step_count": len(
            full_batch_window
        ),
        "cudagraph_runtime_observer_partial_batch_decode_step_count": len(
            partial_batch_window
        ),
        "cudagraph_runtime_observer_full_batch_missing_step_count": (
            full_batch_missing
        ),
        "cudagraph_runtime_observer_partial_batch_missing_step_count": (
            partial_batch_missing
        ),
        "cudagraph_runtime_observer_full_batch_distribution": (
            _serialized_distribution(full_batch_distribution)
        ),
        "cudagraph_runtime_observer_partial_batch_distribution": (
            _serialized_distribution(partial_batch_distribution)
        ),
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


def pull_step_outputs_with_timestamp(llm_engine) -> tuple[list[Any], float]:
    """Mirror ``LLMEngine.step`` without debug timing instrumentation."""
    if getattr(llm_engine, "should_execute_dummy_batch", False):
        llm_engine.should_execute_dummy_batch = False
        llm_engine.engine_core.execute_dummy_batch()
        return [], float("nan")

    outputs = llm_engine.engine_core.get_output()
    if getattr(llm_engine, "_sfi_cudagraph_runtime_observer_enabled", False):
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
    timestamp = float(outputs.timestamp)
    processed_outputs = llm_engine.output_processor.process_outputs(
        outputs.outputs,
        engine_core_timestamp=timestamp,
        iteration_stats=None,
    )
    llm_engine.engine_core.abort_requests(processed_outputs.reqs_to_abort)
    return list(processed_outputs.request_outputs), timestamp
