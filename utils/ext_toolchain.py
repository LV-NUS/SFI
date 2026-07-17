"""Fail-closed CUDA toolkit selection for project-owned JIT extensions.

The resolver runs only when a JIT build or cache identity is initialized.  It
selects one canonical toolkit, validates it against the Torch CUDA major, then
normalizes every CUDA compiler alias so subsequent extension builds cannot
silently switch toolchains.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


_NVCC_RELEASE_RE = re.compile(r"release\s+(\d+)\.(\d+)")
_TORCH_CUDA_RELEASE_RE = re.compile(r"^(\d+)(?:\.(\d+))?")
_CONVENTIONAL_CUDA_HOME = "/usr/local/cuda"
_COMPILER_ALIASES = ("PYTORCH_NVCC", "CUDACXX")
_HOME_ALIASES = ("CUDA_HOME", "CUDA_PATH")


@dataclass(frozen=True)
class CudaToolchain:
    """Canonical identity of the CUDA toolkit selected for this process."""

    nvcc_path: str
    cuda_home: str
    release: str
    major: int
    minor: int


def _contract_error(owner: str, detail: str) -> RuntimeError:
    return RuntimeError(f"{owner}: invalid CUDA JIT toolchain: {detail}")


def _canonical_executable(path_value: str, *, alias: str, owner: str) -> str:
    candidate = Path(path_value)
    if not candidate.is_absolute():
        raise _contract_error(
            owner,
            f"{alias} must be an absolute nvcc path, got {path_value!r}",
        )
    canonical = Path(os.path.realpath(candidate))
    if not canonical.is_file() or not os.access(canonical, os.X_OK):
        raise _contract_error(
            owner,
            f"{alias} does not name an executable file: {canonical}",
        )
    return str(canonical)


def _canonical_cuda_home(
    path_value: str,
    *,
    alias: str,
    owner: str,
) -> tuple[str, str]:
    candidate = Path(path_value)
    if not candidate.is_absolute():
        raise _contract_error(
            owner,
            f"{alias} must be an absolute CUDA toolkit path, got {path_value!r}",
        )
    canonical_home = Path(os.path.realpath(candidate))
    if not canonical_home.is_dir():
        raise _contract_error(
            owner,
            f"{alias} does not name a directory: {canonical_home}",
        )
    nvcc = _canonical_executable(
        str(canonical_home / "bin" / "nvcc"),
        alias=f"{alias}/bin/nvcc",
        owner=owner,
    )
    return str(canonical_home), nvcc


def _one_canonical_value(
    values: dict[str, str],
    *,
    kind: str,
    owner: str,
) -> str | None:
    if not values:
        return None
    canonical_values = set(values.values())
    if len(canonical_values) != 1:
        rendered = ", ".join(f"{name}={value}" for name, value in values.items())
        raise _contract_error(owner, f"conflicting explicit {kind} aliases: {rendered}")
    return next(iter(canonical_values))


def _explicit_compiler(*, owner: str) -> str | None:
    values = {
        alias: _canonical_executable(raw, alias=alias, owner=owner)
        for alias in _COMPILER_ALIASES
        if (raw := os.environ.get(alias))
    }
    return _one_canonical_value(
        values,
        kind="compiler",
        owner=owner,
    )


def _explicit_cuda_home(*, owner: str) -> tuple[str, str] | None:
    homes: dict[str, str] = {}
    nvccs: dict[str, str] = {}
    for alias in _HOME_ALIASES:
        raw = os.environ.get(alias)
        if not raw:
            continue
        home, nvcc = _canonical_cuda_home(raw, alias=alias, owner=owner)
        homes[alias] = home
        nvccs[alias] = nvcc
    home = _one_canonical_value(homes, kind="CUDA home", owner=owner)
    if home is None:
        return None
    nvcc = _one_canonical_value(nvccs, kind="CUDA home compiler", owner=owner)
    assert nvcc is not None
    return home, nvcc


def _discovered_nvcc(path: Path, *, alias: str, owner: str) -> str | None:
    if not os.path.lexists(path):
        return None
    return _canonical_executable(str(path), alias=alias, owner=owner)


def _cuda_home_from_nvcc(nvcc_path: str, *, owner: str) -> str:
    nvcc = Path(nvcc_path)
    if nvcc.name != "nvcc" or nvcc.parent.name != "bin":
        raise _contract_error(
            owner,
            "canonical nvcc must use a <CUDA_HOME>/bin/nvcc layout, got "
            f"{nvcc}",
        )
    cuda_home = Path(os.path.realpath(nvcc.parent.parent))
    if not cuda_home.is_dir():
        raise _contract_error(owner, f"derived CUDA_HOME is not a directory: {cuda_home}")
    return str(cuda_home)


def _nvcc_release(nvcc_path: str, *, owner: str) -> tuple[str, int, int]:
    try:
        completed = subprocess.run(
            [nvcc_path, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise _contract_error(
            owner,
            f"failed to execute {nvcc_path} --version: {type(exc).__name__}: {exc}",
        ) from exc
    version_text = "\n".join(
        part for part in (completed.stdout, completed.stderr) if part
    )
    if completed.returncode != 0:
        raise _contract_error(
            owner,
            f"{nvcc_path} --version exited {completed.returncode}: "
            f"{version_text.strip() or 'no output'}",
        )
    match = _NVCC_RELEASE_RE.search(version_text)
    if match is None:
        raise _contract_error(
            owner,
            f"cannot parse CUDA release from {nvcc_path} --version output",
        )
    major, minor = (int(match.group(1)), int(match.group(2)))
    return f"{major}.{minor}", major, minor


def _torch_cuda_major(*, owner: str) -> tuple[str, int]:
    try:
        import torch
    except Exception as exc:
        raise _contract_error(owner, f"cannot import Torch: {exc}") from exc
    torch_cuda = getattr(torch.version, "cuda", None)
    if not torch_cuda:
        raise _contract_error(owner, "Torch is not a CUDA build (torch.version.cuda is empty)")
    release = str(torch_cuda)
    match = _TORCH_CUDA_RELEASE_RE.match(release)
    if match is None:
        raise _contract_error(owner, f"cannot parse torch.version.cuda={release!r}")
    return release, int(match.group(1))


def resolve_cuda_toolchain_or_raise(*, owner: str) -> CudaToolchain:
    """Resolve and validate exactly one CUDA toolkit without mutating the env."""

    explicit_compiler = _explicit_compiler(owner=owner)
    explicit_home = _explicit_cuda_home(owner=owner)

    if explicit_compiler is not None:
        if explicit_home is not None and explicit_compiler != explicit_home[1]:
            raise _contract_error(
                owner,
                "explicit compiler aliases disagree with CUDA_HOME/CUDA_PATH: "
                f"compiler={explicit_compiler}, home_nvcc={explicit_home[1]}",
            )
        nvcc_path = explicit_compiler
        cuda_home = (
            explicit_home[0]
            if explicit_home is not None
            else _cuda_home_from_nvcc(nvcc_path, owner=owner)
        )
    elif explicit_home is not None:
        cuda_home, nvcc_path = explicit_home
    else:
        interpreter_nvcc = Path(sys.executable).resolve().parent / "nvcc"
        nvcc_path = _discovered_nvcc(
            interpreter_nvcc,
            alias="interpreter-adjacent nvcc",
            owner=owner,
        )
        if nvcc_path is None:
            conventional_home = Path(_CONVENTIONAL_CUDA_HOME)
            nvcc_path = _discovered_nvcc(
                conventional_home / "bin" / "nvcc",
                alias=f"{_CONVENTIONAL_CUDA_HOME}/bin/nvcc",
                owner=owner,
            )
        if nvcc_path is None:
            path_nvcc = shutil.which("nvcc")
            if path_nvcc:
                nvcc_path = _canonical_executable(
                    path_nvcc,
                    alias="PATH nvcc",
                    owner=owner,
                )
        if nvcc_path is None:
            raise _contract_error(
                owner,
                "no nvcc found; set absolute matching PYTORCH_NVCC/CUDACXX or "
                "CUDA_HOME/CUDA_PATH",
            )
        cuda_home = _cuda_home_from_nvcc(nvcc_path, owner=owner)

    release, major, minor = _nvcc_release(nvcc_path, owner=owner)
    if major < 12:
        raise _contract_error(
            owner,
            f"nvcc {nvcc_path} is CUDA {release}; CUDA >=12 is required",
        )
    torch_release, torch_major = _torch_cuda_major(owner=owner)
    if major != torch_major:
        raise _contract_error(
            owner,
            "nvcc/Torch CUDA major mismatch: "
            f"nvcc={release} ({nvcc_path}), torch={torch_release}",
        )
    return CudaToolchain(
        nvcc_path=nvcc_path,
        cuda_home=cuda_home,
        release=release,
        major=major,
        minor=minor,
    )


def _prepend_cuda_bin(path_value: str, cuda_bin: str) -> str:
    entries = [entry for entry in path_value.split(os.pathsep) if entry]
    entries = [entry for entry in entries if entry != cuda_bin]
    return os.pathsep.join((cuda_bin, *entries))


def configure_cuda_toolchain_or_raise(*, owner: str) -> CudaToolchain:
    """Resolve one toolkit and normalize every process-global CUDA alias."""

    toolchain = resolve_cuda_toolchain_or_raise(owner=owner)
    cuda_bin = str(Path(toolchain.cuda_home) / "bin")
    os.environ["PYTORCH_NVCC"] = toolchain.nvcc_path
    os.environ["CUDACXX"] = toolchain.nvcc_path
    os.environ["CUDA_HOME"] = toolchain.cuda_home
    os.environ["CUDA_PATH"] = toolchain.cuda_home
    os.environ["PATH"] = _prepend_cuda_bin(os.environ.get("PATH", ""), cuda_bin)

    try:
        import torch.utils.cpp_extension as torch_cpp_extension
    except Exception as exc:
        raise _contract_error(owner, f"cannot import torch CUDA extension support: {exc}") from exc
    torch_cpp_extension.CUDA_HOME = toolchain.cuda_home
    return toolchain


def configure_jit_toolchain_or_raise(*, ext_name: str) -> str:
    """Backward-compatible JIT entrypoint returning the canonical nvcc path."""

    return configure_cuda_toolchain_or_raise(owner=ext_name).nvcc_path
