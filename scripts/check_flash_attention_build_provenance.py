#!/usr/bin/env python3
"""Validate the exact FlashAttention source/build consumed by a runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any


IDENTITY_SCHEMA = "sfi.flash_attention_build_identity.v1"
PROVENANCE_NAME = "sfi_flash_attention_build_provenance.json"
_SETUP_ASSIGNMENT_RE = re.compile(r'^([A-Z0-9_]+)="([^"]*)"$', re.MULTILINE)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TREE_RE = re.compile(r"^[0-9a-f]{40}$")
_CUDA_RELEASE_RE = re.compile(r"\brelease\s+([0-9]+(?:\.[0-9]+)?)\b")
_CMAKE_VERSION_RE = re.compile(
    r"^cmake version ([0-9]+(?:\.[0-9]+){1,2})(?:\s|$)"
)
_CMAKE_MINIMUM_RE = re.compile(
    r"^\s*cmake_minimum_required\s*\(\s*VERSION\s+"
    r"([0-9]+(?:\.[0-9]+){1,2})(?:\.\.\.[^\s)]+)?(?:\s+FATAL_ERROR)?\s*\)",
    re.IGNORECASE | re.MULTILINE,
)
_ARCH_CONTRACT = {
    "sm80": {
        "backend": "fa3",
        "capability": "8.0",
        "arch_list": "8.0",
        "build_status": "built",
    },
    "sm90": {
        "backend": "fa3",
        "capability": "9.0",
        "arch_list": "9.0a",
        "build_status": "built",
    },
    "sm100": {
        "backend": "fa4_cute",
        "capability": "10.0",
        "arch_list": "",
        "build_status": "cute_jit_source_ready",
    },
}


class BuildProvenanceError(ValueError):
    """Raised when a build cannot be attributed to the current release."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    with os.fdopen(fd, "rb") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise BuildProvenanceError(f"not a regular file: {path}")
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-standard JSON constant: {value}")
            ),
        )
    except (OSError, ValueError) as exc:
        raise BuildProvenanceError(f"invalid build provenance {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise BuildProvenanceError("build provenance root must be a JSON object")
    return payload


def _setup_assignments(sfi_root: Path) -> dict[str, str]:
    setup = sfi_root / "scripts" / "setup_flash_attention.sh"
    try:
        source = setup.read_text(encoding="utf-8")
    except OSError as exc:
        raise BuildProvenanceError(f"cannot read current setup contract: {exc}") from exc
    assignments = dict(_SETUP_ASSIGNMENT_RE.findall(source))
    required = {
        "BASE_COMMIT",
        "EXPECTED_PATCH_SHA256",
        "EXPECTED_PATCHED_TREE",
        "EXPECTED_FA4_PATCH_SHA256",
        "EXPECTED_FA4_PATCHED_TREE",
        "MINIMUM_CMAKE_VERSION",
    }
    missing = sorted(required - assignments.keys())
    if missing:
        raise BuildProvenanceError(f"setup identity assignments missing: {missing}")
    return assignments


def _git(root: Path, *args: str) -> str:
    git_env = os.environ.copy()
    for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        git_env.pop(name, None)
    # Provenance validation is read-only.  Disabling Git's optional locks keeps
    # concurrent launch checks from refreshing or locking the shared index.
    git_env["GIT_OPTIONAL_LOCKS"] = "0"
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        env=git_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise BuildProvenanceError(
            f"git {' '.join(args)} failed for {root}: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def _require_equal(payload: dict[str, Any], field: str, expected: object) -> None:
    actual = payload.get(field)
    if actual != expected:
        raise BuildProvenanceError(
            f"build provenance {field} mismatch: actual={actual!r} expected={expected!r}"
        )


def _canonical_payload_path(
    payload: dict[str, Any],
    field: str,
    *,
    directory: bool = False,
    executable: bool = False,
) -> Path:
    raw = payload.get(field)
    if not isinstance(raw, str) or not raw or not Path(raw).is_absolute():
        raise BuildProvenanceError(
            f"build provenance {field} must be a non-empty absolute path: {raw!r}"
        )
    try:
        resolved = Path(raw).resolve(strict=True)
    except OSError as exc:
        raise BuildProvenanceError(
            f"build provenance {field} does not exist: {raw!r}"
        ) from exc
    if raw != str(resolved):
        raise BuildProvenanceError(
            f"build provenance {field} is not canonical: {raw!r} != {str(resolved)!r}"
        )
    if directory and not resolved.is_dir():
        raise BuildProvenanceError(f"build provenance {field} is not a directory: {resolved}")
    if not directory and not resolved.is_file():
        raise BuildProvenanceError(f"build provenance {field} is not a file: {resolved}")
    if executable and not os.access(resolved, os.X_OK):
        raise BuildProvenanceError(f"build provenance {field} is not executable: {resolved}")
    return resolved


def _nvcc_identity(cudacxx: Path) -> tuple[str, str, str]:
    result = subprocess.run(
        [str(cudacxx), "--version"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise BuildProvenanceError(
            f"canonical CUDACXX --version failed: {cudacxx}: {result.stderr.strip()}"
        )
    version = result.stdout.strip()
    match = _CUDA_RELEASE_RE.search(version)
    if match is None:
        raise BuildProvenanceError(
            f"cannot parse CUDA compiler release from {cudacxx}: {version!r}"
        )
    release = match.group(1)
    return version, release, release.split(".", 1)[0]


def _numeric_version(version: str) -> tuple[int, int, int]:
    if re.fullmatch(r"[0-9]+(?:\.[0-9]+){1,2}", version) is None:
        raise BuildProvenanceError(f"invalid numeric version: {version!r}")
    components = [int(component) for component in version.split(".")]
    padded = components + [0, 0]
    return padded[0], padded[1], padded[2]


def _cmake_identity(cmake: Path) -> str:
    result = subprocess.run(
        [str(cmake), "--version"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise BuildProvenanceError(
            f"canonical CMake --version failed: {cmake}: {result.stderr.strip()}"
        )
    match = _CMAKE_VERSION_RE.match(result.stdout.strip())
    if match is None:
        raise BuildProvenanceError(
            f"cannot parse CMake version from {cmake}: {result.stdout.strip()!r}"
        )
    return match.group(1)


def _source_cmake_minimum(target: Path) -> str:
    cmake_lists = target / "CMakeLists.txt"
    try:
        source = cmake_lists.read_text(encoding="utf-8", errors="strict")
    except (OSError, UnicodeError) as exc:
        raise BuildProvenanceError(
            f"cannot read FlashAttention CMake requirement: {cmake_lists}: {exc}"
        ) from exc
    match = _CMAKE_MINIMUM_RE.search(source)
    if match is None:
        raise BuildProvenanceError(
            f"cannot parse cmake_minimum_required from {cmake_lists}"
        )
    return match.group(1)


def _cmake_cache_entries(cache: Path) -> dict[str, str]:
    try:
        lines = cache.read_text(encoding="utf-8", errors="strict").splitlines()
    except (OSError, UnicodeError) as exc:
        raise BuildProvenanceError(f"cannot read CMake cache {cache}: {exc}") from exc
    entries: dict[str, str] = {}
    for line in lines:
        if not line or line.startswith(("#", "//")) or "=" not in line:
            continue
        typed_key, value = line.split("=", 1)
        entries[typed_key.split(":", 1)[0]] = value
    return entries


def _patch_assets(sfi_root: Path) -> tuple[Path, Path]:
    fa3_candidates = (
        sfi_root
        / "patches/fa3_native/upstream_recovery/latest-sm80-dual-source-current.patch",
        sfi_root / "kernel_patches/sfi_fa3_sm80_sm90.patch",
    )
    fa4_candidates = (
        sfi_root
        / "patches/fa4_cute_sm100/upstream_recovery/latest-sm100-cute-compact-recent-current.patch",
        sfi_root / "kernel_patches/sfi_fa4_sm100_cute.patch",
    )

    def choose(candidates: tuple[Path, ...], label: str) -> Path:
        present = [path for path in candidates if path.is_file()]
        if not present:
            raise BuildProvenanceError(f"current {label} patch asset is missing")
        hashes = {_sha256(path) for path in present}
        if len(hashes) != 1:
            raise BuildProvenanceError(f"current {label} patch exports disagree")
        return present[0]

    return choose(fa3_candidates, "FA3"), choose(fa4_candidates, "FA4")


def validate_build_provenance(
    *,
    sfi_root: Path,
    target: Path,
    architecture: str,
    provenance_path: Path | None = None,
) -> dict[str, Any]:
    """Return a normalized identity or fail on any stale source/build surface."""
    if architecture not in _ARCH_CONTRACT:
        raise BuildProvenanceError(f"unsupported architecture: {architecture}")
    sfi_root = Path(sfi_root).resolve(strict=True)
    target = Path(target).resolve(strict=True)
    provenance_path = (
        Path(provenance_path).resolve(strict=True)
        if provenance_path is not None
        else target / PROVENANCE_NAME
    )
    contract = _ARCH_CONTRACT[architecture]
    assignments = _setup_assignments(sfi_root)
    fa3_patch, fa4_patch = _patch_assets(sfi_root)
    fa3_sha = _sha256(fa3_patch)
    fa4_sha = _sha256(fa4_patch)
    if fa3_sha != assignments["EXPECTED_PATCH_SHA256"]:
        raise BuildProvenanceError("current setup FA3 SHA does not match packaged patch")
    if fa4_sha != assignments["EXPECTED_FA4_PATCH_SHA256"]:
        raise BuildProvenanceError("current setup FA4 SHA does not match packaged patch")

    expected_tree = (
        assignments["EXPECTED_FA4_PATCHED_TREE"]
        if architecture == "sm100"
        else assignments["EXPECTED_PATCHED_TREE"]
    )
    if _TREE_RE.fullmatch(expected_tree) is None:
        raise BuildProvenanceError(f"invalid expected patched tree: {expected_tree!r}")
    source_cmake_minimum = _source_cmake_minimum(target)
    if source_cmake_minimum != assignments["MINIMUM_CMAKE_VERSION"]:
        raise BuildProvenanceError(
            "setup CMake minimum differs from the pinned FlashAttention source: "
            f"setup={assignments['MINIMUM_CMAKE_VERSION']} "
            f"source={source_cmake_minimum}"
        )

    payload = _strict_json(provenance_path)
    _require_equal(payload, "schema_version", 5)
    _require_equal(payload, "architecture", architecture)
    _require_equal(payload, "backend", contract["backend"])
    _require_equal(payload, "base_commit", assignments["BASE_COMMIT"])
    _require_equal(payload, "expected_patched_tree", expected_tree)
    _require_equal(payload, "patch_sha256", fa3_sha)
    _require_equal(payload, "target", str(target))
    _require_equal(payload, "build_status", contract["build_status"])
    _require_equal(payload, "torch_cuda_arch_list", contract["arch_list"])

    detected_capability = str(payload.get("detected_compute_capability", "") or "")
    if detected_capability not in {"", str(contract["capability"])}:
        raise BuildProvenanceError(
            "build provenance detected_compute_capability mismatch: "
            f"actual={detected_capability!r} expected={contract['capability']!r}"
        )

    provenance_python = _canonical_payload_path(
        payload, "python_executable", executable=True
    )
    benchmark_python = Path(sys.executable).resolve(strict=True)
    if provenance_python != benchmark_python:
        raise BuildProvenanceError(
            "build provenance Python differs from the benchmark interpreter: "
            f"build={provenance_python} benchmark={benchmark_python}"
        )
    _require_equal(payload, "python_version", platform.python_version())
    try:
        import torch
    except Exception as exc:
        raise BuildProvenanceError(f"cannot import torch with benchmark Python: {exc}") from exc
    _require_equal(payload, "torch_version", str(torch.__version__))
    _require_equal(payload, "torch_cuda_version", str(torch.version.cuda or ""))

    cuda_home = _canonical_payload_path(payload, "cuda_home", directory=True)
    cuda_path = _canonical_payload_path(payload, "cuda_path", directory=True)
    cudacxx = _canonical_payload_path(payload, "cudacxx", executable=True)
    if cuda_path != cuda_home:
        raise BuildProvenanceError(
            f"build provenance CUDA_PATH differs from CUDA_HOME: {cuda_path} != {cuda_home}"
        )
    try:
        root_nvcc = (cuda_home / "bin" / "nvcc").resolve(strict=True)
    except OSError as exc:
        raise BuildProvenanceError(
            f"build provenance CUDA_HOME has no bin/nvcc: {cuda_home}"
        ) from exc
    if cudacxx != root_nvcc:
        raise BuildProvenanceError(
            f"build provenance CUDACXX is not owned by CUDA_HOME: {cudacxx} != {root_nvcc}"
        )

    nvcc_version, cuda_release, cuda_major = _nvcc_identity(cudacxx)
    _require_equal(payload, "nvcc_version", nvcc_version)
    _require_equal(payload, "cuda_compiler_release", cuda_release)
    _require_equal(payload, "cuda_compiler_major", cuda_major)
    torch_cuda_version = str(torch.version.cuda or "")
    torch_cuda_major = torch_cuda_version.split(".", 1)[0] if torch_cuda_version else ""
    _require_equal(payload, "torch_cuda_major", torch_cuda_major)
    if not torch_cuda_major or cuda_major != torch_cuda_major:
        raise BuildProvenanceError(
            "CUDA compiler major differs from benchmark torch CUDA major: "
            f"compiler={cuda_release} torch={torch_cuda_version or 'unknown'}"
        )

    cmake_build_temp_raw = payload.get("cmake_build_temp")
    cmake_cache_raw = payload.get("cmake_cache_path")
    cmake_compiler_raw = payload.get("cmake_cuda_compiler")
    cmake_root_raw = payload.get("cmake_cuda_toolkit_root")
    cmake_executable_raw = payload.get("cmake_executable")
    cmake_version_raw = payload.get("cmake_version")
    if architecture in {"sm80", "sm90"}:
        cmake_build_temp = _canonical_payload_path(
            payload, "cmake_build_temp", directory=True
        )
        cmake_cache = _canonical_payload_path(payload, "cmake_cache_path")
        cmake_compiler = _canonical_payload_path(
            payload, "cmake_cuda_compiler", executable=True
        )
        cmake_root = _canonical_payload_path(
            payload, "cmake_cuda_toolkit_root", directory=True
        )
        cmake_executable = _canonical_payload_path(
            payload, "cmake_executable", executable=True
        )
        cmake_version = _cmake_identity(cmake_executable)
        _require_equal(payload, "cmake_version", cmake_version)
        minimum_cmake_version = assignments["MINIMUM_CMAKE_VERSION"]
        if _numeric_version(cmake_version) < _numeric_version(minimum_cmake_version):
            raise BuildProvenanceError(
                "CMake is older than the current build contract: "
                f"actual={cmake_version} minimum={minimum_cmake_version}"
            )
        expected_build_parent = (target / "build").resolve()
        if (
            cmake_build_temp.parent != expected_build_parent
            or not cmake_build_temp.name.startswith(f"sfi-{architecture}.")
        ):
            raise BuildProvenanceError(
                "build provenance cmake_build_temp is not a fresh run-scoped directory: "
                f"{cmake_build_temp}"
            )
        try:
            cmake_cache.relative_to(cmake_build_temp)
        except ValueError as exc:
            raise BuildProvenanceError(
                f"build provenance CMake cache is outside build temp: {cmake_cache}"
            ) from exc
        caches = sorted(cmake_build_temp.rglob("CMakeCache.txt"))
        if [path.resolve(strict=True) for path in caches] != [cmake_cache]:
            raise BuildProvenanceError(
                "build provenance fresh build temp must contain exactly its recorded "
                f"CMake cache: found={caches} recorded={cmake_cache}"
            )
        if cmake_compiler != cudacxx:
            raise BuildProvenanceError(
                f"CMake CUDA compiler differs from CUDACXX: {cmake_compiler} != {cudacxx}"
            )
        if cmake_root != cuda_home:
            raise BuildProvenanceError(
                f"CMake CUDA toolkit root differs from CUDA_HOME: {cmake_root} != {cuda_home}"
            )
        cmake_entries = _cmake_cache_entries(cmake_cache)
        cache_compiler_raw = cmake_entries.get("CMAKE_CUDA_COMPILER", "")
        if not cache_compiler_raw:
            raise BuildProvenanceError("recorded CMake cache has no CMAKE_CUDA_COMPILER")
        try:
            cache_compiler = Path(cache_compiler_raw).resolve(strict=True)
        except OSError as exc:
            raise BuildProvenanceError(
                f"recorded CMake CUDA compiler does not exist: {cache_compiler_raw!r}"
            ) from exc
        if cache_compiler != cmake_compiler:
            raise BuildProvenanceError(
                f"recorded CMake cache compiler drift: {cache_compiler} != {cmake_compiler}"
            )
        root_keys = (
            "CUDAToolkit_ROOT",
            "CUDA_TOOLKIT_ROOT_DIR",
            "CMAKE_CUDA_COMPILER_TOOLKIT_ROOT",
        )
        try:
            cache_roots = {
                Path(cmake_entries[key]).resolve(strict=True)
                for key in root_keys
                if cmake_entries.get(key)
            }
        except OSError as exc:
            raise BuildProvenanceError(
                "recorded CMake CUDA toolkit root does not exist"
            ) from exc
        if cache_roots != {cmake_root}:
            raise BuildProvenanceError(
                f"recorded CMake cache toolkit root drift: {cache_roots} != {{{cmake_root}}}"
            )
        cache_command_raw = cmake_entries.get("CMAKE_COMMAND", "")
        if not cache_command_raw:
            raise BuildProvenanceError("recorded CMake cache has no CMAKE_COMMAND")
        try:
            cache_command = Path(cache_command_raw).resolve(strict=True)
        except OSError as exc:
            raise BuildProvenanceError(
                f"recorded CMake command does not exist: {cache_command_raw!r}"
            ) from exc
        if cache_command != cmake_executable:
            raise BuildProvenanceError(
                "recorded CMake cache command drift: "
                f"{cache_command} != {cmake_executable}"
            )
    else:
        for field, actual in (
            ("cmake_build_temp", cmake_build_temp_raw),
            ("cmake_cache_path", cmake_cache_raw),
            ("cmake_cuda_compiler", cmake_compiler_raw),
            ("cmake_cuda_toolkit_root", cmake_root_raw),
            ("cmake_executable", cmake_executable_raw),
            ("cmake_version", cmake_version_raw),
        ):
            if actual != "":
                raise BuildProvenanceError(
                    f"SM100 source-only provenance must leave {field} empty: {actual!r}"
                )
        cmake_build_temp = None
        cmake_cache = None
        cmake_compiler = None
        cmake_root = None
        cmake_executable = None
        cmake_version = ""

    patches = payload.get("patches")
    expected_patch_kinds = ["fa3_sm80_sm90_shared_base"]
    expected_patch_hashes = [fa3_sha]
    if architecture == "sm100":
        expected_patch_kinds.append("fa4_sm100_cute_overlay")
        expected_patch_hashes.append(fa4_sha)
    if not isinstance(patches, list) or len(patches) != len(expected_patch_kinds):
        raise BuildProvenanceError("build provenance patch stack shape mismatch")
    for record, kind, digest in zip(
        patches, expected_patch_kinds, expected_patch_hashes, strict=True
    ):
        if not isinstance(record, dict):
            raise BuildProvenanceError("build provenance patch record must be an object")
        if record.get("kind") != kind or record.get("sha256") != digest:
            raise BuildProvenanceError(
                f"build provenance patch mismatch for {kind}: {record!r}"
            )

    head_commit = _git(target, "rev-parse", "HEAD")
    head_tree = _git(target, "rev-parse", "HEAD^{tree}")
    unmerged_index = _git(target, "ls-files", "--unmerged")
    if unmerged_index:
        raise BuildProvenanceError(
            "FlashAttention index contains unmerged entries: "
            f"{unmerged_index.splitlines()[:8]}"
        )
    index_diff = _git(
        target,
        "diff",
        "--cached",
        "--name-only",
        expected_tree,
        "--",
    )
    index_matches_expected = not bool(index_diff)
    tracked_diff = _git(target, "diff", "--name-only")
    if tracked_diff:
        raise BuildProvenanceError(
            f"FlashAttention tracked source changed after setup: {tracked_diff.splitlines()[:8]}"
        )
    untracked_runtime = _git(
        target,
        "ls-files",
        "--others",
        "--exclude-standard",
        "--",
        "vllm_flash_attn",
        "flash_attn",
    ).splitlines()
    unexpected_runtime = [
        relative
        for relative in untracked_runtime
        if not (
            architecture in {"sm80", "sm90"}
            and Path(relative).parent == Path("vllm_flash_attn")
            and Path(relative).name.startswith("_vllm_fa3_C")
            and Path(relative).suffix == ".so"
        )
    ]
    if unexpected_runtime:
        raise BuildProvenanceError(
            "FlashAttention has untracked runtime source after setup: "
            f"{unexpected_runtime[:8]}"
        )
    source_mode = str(payload.get("source_mode", "") or "")
    if head_tree == expected_tree and index_matches_expected:
        if source_mode != "verified_patched_tree":
            raise BuildProvenanceError(
                f"committed patched tree has inconsistent source_mode={source_mode!r}"
            )
        source_state = "committed_patched_tree"
    elif head_commit == assignments["BASE_COMMIT"] and index_matches_expected:
        allowed_modes = {
            "verified_applied_patch",
            f"fresh_applied_{architecture}_patch_stack",
        }
        if source_mode not in allowed_modes:
            raise BuildProvenanceError(
                f"applied patch tree has inconsistent source_mode={source_mode!r}"
            )
        source_state = "base_with_indexed_patch_stack"
    else:
        raise BuildProvenanceError(
            "FlashAttention source tree does not match the current setup contract: "
            f"head={head_commit} head_tree={head_tree} "
            f"index_diff={index_diff.splitlines()[:8]} "
            f"expected_tree={expected_tree}"
        )

    shared_object = str(payload.get("shared_object", "") or "")
    shared_object_sha = str(payload.get("shared_object_sha256", "") or "").lower()
    if architecture in {"sm80", "sm90"}:
        so_candidates = sorted((target / "vllm_flash_attn").glob("_vllm_fa3_C*.so"))
        if len(so_candidates) != 1:
            raise BuildProvenanceError(
                f"expected exactly one FA3 shared object, found {len(so_candidates)}"
            )
        so_path = so_candidates[0].resolve(strict=True)
        if Path(shared_object).resolve(strict=True) != so_path:
            raise BuildProvenanceError(
                f"build provenance shared_object mismatch: {shared_object!r} != {so_path}"
            )
        actual_so_sha = _sha256(so_path)
        if _SHA256_RE.fullmatch(shared_object_sha) is None or shared_object_sha != actual_so_sha:
            raise BuildProvenanceError(
                "build provenance shared_object_sha256 does not match the loaded FA3 binary"
            )
    else:
        if shared_object or shared_object_sha:
            raise BuildProvenanceError("SM100 source-only build must not claim an FA3 shared object")
        so_path = None
        actual_so_sha = ""

    return {
        "schema": IDENTITY_SCHEMA,
        "provenance_path": str(provenance_path),
        "provenance_sha256": _sha256(provenance_path),
        "architecture": architecture,
        "backend": str(contract["backend"]),
        "detected_compute_capability": detected_capability,
        "base_commit": assignments["BASE_COMMIT"],
        "expected_patched_tree": expected_tree,
        "patch_sha256": fa3_sha,
        "fa4_patch_sha256": fa4_sha if architecture == "sm100" else "",
        "target": str(target),
        "source_mode": source_mode,
        "source_state": source_state,
        "build_status": str(payload["build_status"]),
        "python_executable": str(Path(sys.executable).resolve()),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda or ""),
        "torch_cuda_major": torch_cuda_major,
        "cuda_home": str(cuda_home),
        "cuda_path": str(cuda_path),
        "cudacxx": str(cudacxx),
        "cuda_compiler_release": cuda_release,
        "cuda_compiler_major": cuda_major,
        "cmake_build_temp": (
            str(cmake_build_temp) if cmake_build_temp is not None else ""
        ),
        "cmake_cache_path": str(cmake_cache) if cmake_cache is not None else "",
        "cmake_cuda_compiler": (
            str(cmake_compiler) if cmake_compiler is not None else ""
        ),
        "cmake_cuda_toolkit_root": str(cmake_root) if cmake_root is not None else "",
        "cmake_executable": (
            str(cmake_executable) if cmake_executable is not None else ""
        ),
        "cmake_version": cmake_version,
        "shared_object": str(so_path) if so_path is not None else "",
        "shared_object_sha256": actual_so_sha,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sfi-root", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--architecture", choices=tuple(_ARCH_CONTRACT), required=True)
    parser.add_argument("--provenance", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        identity = validate_build_provenance(
            sfi_root=args.sfi_root,
            target=args.target,
            architecture=args.architecture,
            provenance_path=args.provenance,
        )
    except (BuildProvenanceError, OSError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 78
    print(json.dumps(identity, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
