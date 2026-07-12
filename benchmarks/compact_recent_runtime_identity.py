from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from patches.fa3_native.install import load_vendored_flash_attn_bridge


REPO_ROOT = Path(__file__).resolve().parents[1]
VENDORED_FA3_ROOT = REPO_ROOT / "third_party_upstreams" / "vllm-project-flash-attention"
COMPACT_RECENT_ENV_KEYS = (
    "VLLM_FA3_FORCE_PAGED_COMPACT_LOADER",
    "VLLM_FA3_SKIP_FWD_COMBINE",
    "VLLM_FA3_ENABLE_PROFILER_DIAGNOSTICS",
    "VLLM_FA3_DEBUG_CLEAR_KEEPALIVE",
    "VLLM_FA3_COMPACT_RECENT_SEMAPHORE_GUARD",
    "VLLM_SPARSE_FA3_ROUTE_TRACE_LOG",
    "VLLM_SPARSE_FA3_STEP_TRACE_LOG",
)


@dataclass(frozen=True)
class RuntimeIdentity:
    root_revision: str
    vendored_fa3_revision: str
    fa3_so_path: str
    fa3_so_sha256: str
    python_bridge_path: str
    torch_version: str
    torch_cuda_version: str
    cuda_device_name: str
    nvcc_version: str
    compact_recent_env: dict[str, str]


def _git_revision(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
        text=True,
    ).strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nvcc_version() -> str:
    cuda_home = os.environ.get("CUDA_HOME")
    nvcc = str(Path(cuda_home) / "bin" / "nvcc") if cuda_home else "nvcc"
    result = subprocess.run(
        [nvcc, "--version"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return next((line for line in lines if "release" in line), lines[-1] if lines else "")


def _resolve_loaded_fa3_so_path(bridge: Any) -> Path:
    load_extension_module = getattr(bridge.interface_module, "_load_fa3_extension_module", None)
    if callable(load_extension_module):
        extension_module = load_extension_module()
        module_file = getattr(extension_module, "__file__", None)
        if module_file:
            return Path(module_file).resolve()

    candidates = sorted((bridge.root / "vllm_flash_attn").glob("_vllm_fa3_C*.so"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected exactly one vendored FA3 shared object, found {len(candidates)}"
        )
    return candidates[0].resolve()


def collect_runtime_identity() -> RuntimeIdentity:
    bridge = load_vendored_flash_attn_bridge()
    so_path = _resolve_loaded_fa3_so_path(bridge)
    bridge_path = Path(getattr(bridge.interface_module, "__file__", "")).resolve()
    env = {
        key: str(os.environ[key])
        for key in COMPACT_RECENT_ENV_KEYS
        if key in os.environ
    }
    return RuntimeIdentity(
        root_revision=_git_revision(REPO_ROOT),
        vendored_fa3_revision=_git_revision(VENDORED_FA3_ROOT),
        fa3_so_path=str(so_path),
        fa3_so_sha256=_sha256_file(so_path),
        python_bridge_path=str(bridge_path),
        torch_version=str(torch.__version__),
        torch_cuda_version=str(torch.version.cuda),
        cuda_device_name=(
            str(torch.cuda.get_device_name(0))
            if torch.cuda.is_available()
            else "cuda_unavailable"
        ),
        nvcc_version=_nvcc_version(),
        compact_recent_env=env,
    )


def runtime_identity_to_payload(identity: RuntimeIdentity) -> dict[str, object]:
    payload = {
        "root_revision": identity.root_revision,
        "vendored_fa3_revision": identity.vendored_fa3_revision,
        "fa3_so_path": identity.fa3_so_path,
        "fa3_so_sha256": identity.fa3_so_sha256,
        "python_bridge_path": identity.python_bridge_path,
        "torch_version": identity.torch_version,
        "torch_cuda_version": identity.torch_cuda_version,
        "cuda_device_name": identity.cuda_device_name,
        "nvcc_version": identity.nvcc_version,
        "compact_recent_env": dict(identity.compact_recent_env),
    }
    for key, value in payload.items():
        if key == "compact_recent_env":
            continue
        if value in ("", None):
            raise ValueError(f"runtime identity requires {key}")
    return payload
