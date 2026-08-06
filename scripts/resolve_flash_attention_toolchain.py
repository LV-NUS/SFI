#!/usr/bin/env python3
"""Resolve the launch-time CUDA toolchain from validated build provenance."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


IDENTITY_SCHEMA = "sfi.flash_attention_build_identity.v1"
PROVENANCE_NAME = "sfi_flash_attention_build_provenance.json"
PROVENANCE_SCHEMA_VERSION = 5
TOOLCHAIN_FIELDS = (
    "cuda_home",
    "cuda_path",
    "cudacxx",
    "cuda_compiler_release",
    "cuda_compiler_major",
    "torch_cuda_major",
)
CMAKE_FIELDS = (
    "cmake_build_temp",
    "cmake_cache_path",
    "cmake_cuda_compiler",
    "cmake_cuda_toolkit_root",
    "cmake_executable",
    "cmake_version",
)
_NVCC_RELEASE_RE = re.compile(r"\brelease\s+([0-9]+(?:\.[0-9]+)?)\b")
_NUMERIC_VERSION_RE = re.compile(r"^[0-9]+(?:\.[0-9]+){1,2}$")


class ToolchainPreflightError(ValueError):
    """Raised when launch cannot reproduce the build-time CUDA toolchain."""


def _strict_json_text(text: str, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            text,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-standard JSON constant: {value}")
            ),
        )
    except ValueError as exc:
        raise ToolchainPreflightError(f"invalid {label}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ToolchainPreflightError(f"{label} root must be a JSON object")
    return payload


def _strict_json_file(path: Path, *, label: str) -> dict[str, Any]:
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ToolchainPreflightError(f"cannot read {label} {path}: {exc}") from exc
    return _strict_json_text(source, label=label)


def _canonical_directory(value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value or not Path(value).is_absolute():
        raise ToolchainPreflightError(
            f"build provenance {field} must be a nonempty absolute path"
        )
    try:
        resolved = Path(value).resolve(strict=True)
    except OSError as exc:
        raise ToolchainPreflightError(
            f"build provenance {field} is unavailable: {value}: {exc}"
        ) from exc
    if not resolved.is_dir() or value != str(resolved):
        raise ToolchainPreflightError(
            f"build provenance {field} is not a canonical directory: {value}"
        )
    return resolved


def _canonical_file(value: object, *, field: str, executable: bool = False) -> Path:
    if not isinstance(value, str) or not value or not Path(value).is_absolute():
        raise ToolchainPreflightError(
            f"build provenance {field} must be a nonempty absolute path"
        )
    try:
        resolved = Path(value).resolve(strict=True)
        mode = resolved.stat().st_mode
    except OSError as exc:
        raise ToolchainPreflightError(
            f"build provenance {field} is unavailable: {value}: {exc}"
        ) from exc
    if not stat.S_ISREG(mode) or value != str(resolved):
        raise ToolchainPreflightError(
            f"build provenance {field} is not a canonical regular file: {value}"
        )
    if executable and not os.access(resolved, os.X_OK):
        raise ToolchainPreflightError(
            f"build provenance {field} is not executable: {resolved}"
        )
    return resolved


def _require_identity_matches_provenance(
    identity: Mapping[str, Any],
    provenance: Mapping[str, Any],
    field: str,
) -> str:
    identity_value = identity.get(field)
    provenance_value = provenance.get(field)
    if not isinstance(identity_value, str) or not isinstance(provenance_value, str):
        raise ToolchainPreflightError(
            f"validated build identity is missing schema-v{PROVENANCE_SCHEMA_VERSION} field {field}; "
            "rerun setup_flash_attention.sh with the current release"
        )
    if identity_value != provenance_value:
        raise ToolchainPreflightError(
            f"validated build identity {field} disagrees with raw provenance"
        )
    return identity_value


def _run_checker(
    *,
    python: Path,
    checker: Path,
    sfi_root: Path,
    target: Path,
    architecture: str,
) -> dict[str, Any]:
    result = subprocess.run(
        [
            str(python),
            "-I",
            str(checker),
            "--sfi-root",
            str(sfi_root),
            "--target",
            str(target),
            "--architecture",
            architecture,
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no checker output"
        raise ToolchainPreflightError(
            "FlashAttention build provenance is missing, stale, or invalid; "
            "rerun setup_flash_attention.sh with the current release. "
            f"Checker detail: {detail}"
        )
    if result.stderr:
        sys.stderr.write(result.stderr)
        sys.stderr.flush()
    identity = _strict_json_text(
        result.stdout, label="FlashAttention build identity"
    )
    if identity.get("schema") != IDENTITY_SCHEMA:
        raise ToolchainPreflightError(
            "FlashAttention provenance checker is obsolete or returned the wrong schema"
        )
    return identity


def _validate_caller_path(
    *, name: str, value: str, expected: Path, directory: bool
) -> None:
    if not value or not Path(value).is_absolute():
        raise ToolchainPreflightError(
            f"caller {name} conflicts with build provenance: {value!r} is not an absolute path"
        )
    try:
        actual = Path(value).resolve(strict=True)
    except OSError as exc:
        raise ToolchainPreflightError(
            f"caller {name} conflicts with build provenance: {value}: {exc}"
        ) from exc
    if (directory and not actual.is_dir()) or (not directory and not actual.is_file()):
        kind = "directory" if directory else "file"
        raise ToolchainPreflightError(
            f"caller {name} conflicts with build provenance: {value} is not a {kind}"
        )
    if actual != expected:
        raise ToolchainPreflightError(
            f"caller {name} conflicts with build provenance: "
            f"caller={actual} build={expected}"
        )


def resolve_toolchain(
    *,
    python: Path,
    checker: Path,
    sfi_root: Path,
    target: Path,
    architecture: str,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Validate provenance and return the exact environment for child JITs."""
    if architecture not in {"sm80", "sm90", "sm100"}:
        raise ToolchainPreflightError(f"unsupported architecture: {architecture}")
    try:
        python = Path(python).resolve(strict=True)
        checker = Path(checker).resolve(strict=True)
        sfi_root = Path(sfi_root).resolve(strict=True)
        target = Path(target).resolve(strict=True)
    except OSError as exc:
        raise ToolchainPreflightError(f"preflight input path is unavailable: {exc}") from exc
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ToolchainPreflightError(f"runtime Python is not executable: {python}")

    provenance_path = target / PROVENANCE_NAME
    if not provenance_path.is_file():
        raise ToolchainPreflightError(
            f"FlashAttention build provenance is missing: {provenance_path}; "
            "rerun setup_flash_attention.sh before launching"
        )

    identity = _run_checker(
        python=python,
        checker=checker,
        sfi_root=sfi_root,
        target=target,
        architecture=architecture,
    )
    provenance = _strict_json_file(
        provenance_path, label="FlashAttention build provenance"
    )
    if provenance.get("schema_version") != PROVENANCE_SCHEMA_VERSION:
        raise ToolchainPreflightError(
            "FlashAttention build provenance predates the canonical CUDA/CMake toolchain "
            f"contract (expected schema_version={PROVENANCE_SCHEMA_VERSION}); "
            "rerun setup_flash_attention.sh"
        )
    if identity.get("provenance_path") != str(provenance_path.resolve(strict=True)):
        raise ToolchainPreflightError(
            "validated build identity points at a different provenance file"
        )
    if identity.get("architecture") != architecture:
        raise ToolchainPreflightError(
            "validated build identity architecture differs from the launcher"
        )
    provenance_python = _require_identity_matches_provenance(
        identity, provenance, "python_executable"
    )
    validated_python = _canonical_file(
        provenance_python, field="python_executable", executable=True
    )
    if validated_python != python:
        raise ToolchainPreflightError(
            "build provenance Python differs from the launcher interpreter: "
            f"build={validated_python} launcher={python}"
        )

    fields = {
        field: _require_identity_matches_provenance(identity, provenance, field)
        for field in (*TOOLCHAIN_FIELDS, *CMAKE_FIELDS)
    }
    cuda_home = _canonical_directory(fields["cuda_home"], field="cuda_home")
    cuda_path = _canonical_directory(fields["cuda_path"], field="cuda_path")
    cudacxx = _canonical_file(
        fields["cudacxx"], field="cudacxx", executable=True
    )
    if cuda_path != cuda_home:
        raise ToolchainPreflightError(
            f"build provenance CUDA_PATH differs from CUDA_HOME: {cuda_path} != {cuda_home}"
        )
    expected_nvcc = (cuda_home / "bin" / "nvcc").resolve(strict=True)
    if cudacxx != expected_nvcc:
        raise ToolchainPreflightError(
            f"build provenance CUDACXX is not owned by CUDA_HOME: {cudacxx} != {expected_nvcc}"
        )

    release = fields["cuda_compiler_release"]
    major = fields["cuda_compiler_major"]
    torch_major = fields["torch_cuda_major"]
    if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", release) is None:
        raise ToolchainPreflightError(
            f"invalid build provenance cuda_compiler_release: {release!r}"
        )
    if re.fullmatch(r"[0-9]+", major) is None or release.split(".", 1)[0] != major:
        raise ToolchainPreflightError(
            f"invalid build provenance CUDA compiler major: release={release!r} major={major!r}"
        )
    if int(major) < 12 or torch_major != major:
        raise ToolchainPreflightError(
            "build provenance CUDA compiler is incompatible with Torch: "
            f"compiler={release} torch_cuda_major={torch_major!r}"
        )

    if architecture in {"sm80", "sm90"}:
        cmake_build_temp = _canonical_directory(
            fields["cmake_build_temp"], field="cmake_build_temp"
        )
        cmake_cache = _canonical_file(
            fields["cmake_cache_path"], field="cmake_cache_path"
        )
        cmake_compiler = _canonical_file(
            fields["cmake_cuda_compiler"],
            field="cmake_cuda_compiler",
            executable=True,
        )
        cmake_root = _canonical_directory(
            fields["cmake_cuda_toolkit_root"], field="cmake_cuda_toolkit_root"
        )
        _canonical_file(
            fields["cmake_executable"], field="cmake_executable", executable=True
        )
        cmake_version = fields["cmake_version"]
        if _NUMERIC_VERSION_RE.fullmatch(cmake_version) is None:
            raise ToolchainPreflightError(
                f"invalid build provenance cmake_version: {cmake_version!r}"
            )
        if cmake_cache.parent != cmake_build_temp and cmake_build_temp not in cmake_cache.parents:
            raise ToolchainPreflightError(
                "build provenance CMake cache is outside its fresh build directory"
            )
        if cmake_compiler != cudacxx or cmake_root != cuda_home:
            raise ToolchainPreflightError(
                "build provenance CMake CUDA owner differs from the runtime toolchain"
            )
    elif any(fields[field] for field in CMAKE_FIELDS):
        raise ToolchainPreflightError(
            "SM100 source-only provenance must not claim an FA3 CMake build owner"
        )

    version = subprocess.run(
        [str(cudacxx), "--version"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if version.returncode != 0:
        raise ToolchainPreflightError(
            f"provenance CUDA compiler is no longer runnable: {cudacxx}: {version.stderr.strip()}"
        )
    releases = _NVCC_RELEASE_RE.findall(version.stdout)
    if not releases or releases[-1] != release:
        raise ToolchainPreflightError(
            "provenance CUDA compiler release changed after setup: "
            f"recorded={release!r} actual={(releases[-1] if releases else 'unparseable')!r}"
        )
    nvcc_version = provenance.get("nvcc_version")
    if not isinstance(nvcc_version, str) or nvcc_version.strip() != version.stdout.strip():
        raise ToolchainPreflightError(
            "provenance CUDA compiler version output changed after setup"
        )

    caller = dict(os.environ if environ is None else environ)
    for name, expected, directory in (
        ("CUDA_HOME", cuda_home, True),
        ("CUDA_PATH", cuda_home, True),
        ("CUDACXX", cudacxx, False),
        ("PYTORCH_NVCC", cudacxx, False),
    ):
        if name in caller:
            _validate_caller_path(
                name=name,
                value=caller[name],
                expected=expected,
                directory=directory,
            )

    original_path = caller.get("PATH", "")
    if any(character in original_path for character in ("\0", "\n", "\r")):
        raise ToolchainPreflightError("caller PATH contains a forbidden control character")
    cuda_bin = cuda_home / "bin"
    retained: list[str] = []
    for entry in original_path.split(os.pathsep):
        if not entry:
            continue
        try:
            same_bin = Path(entry).resolve(strict=False) == cuda_bin
        except OSError:
            same_bin = False
        if not same_bin:
            retained.append(entry)
    normalized_path = os.pathsep.join((str(cuda_bin), *retained))
    selected_nvcc = shutil.which("nvcc", path=normalized_path)
    if selected_nvcc is None or Path(selected_nvcc).resolve(strict=True) != cudacxx:
        raise ToolchainPreflightError(
            "normalized PATH does not select the provenance CUDA compiler"
        )

    compact_identity = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return {
        "CUDA_HOME": str(cuda_home),
        "CUDA_PATH": str(cuda_home),
        "CUDACXX": str(cudacxx),
        "PYTORCH_NVCC": str(cudacxx),
        "PATH": normalized_path,
        "SFI_RUNNER_ATTENTION_BUILD_IDENTITY_JSON": compact_identity,
        "SFI_RUNNER_CUDA_TOOLCHAIN_STATUS": "passed",
        "SFI_RUNNER_CUDA_COMPILER_RELEASE": release,
    }


def render_shell_exports(environment: Mapping[str, str]) -> str:
    """Render a fixed-name, shell-quoted environment update for launchers."""
    expected = (
        "CUDA_HOME",
        "CUDA_PATH",
        "CUDACXX",
        "PYTORCH_NVCC",
        "PATH",
        "SFI_RUNNER_ATTENTION_BUILD_IDENTITY_JSON",
        "SFI_RUNNER_CUDA_TOOLCHAIN_STATUS",
        "SFI_RUNNER_CUDA_COMPILER_RELEASE",
    )
    if set(environment) != set(expected):
        raise ToolchainPreflightError("internal toolchain environment shape mismatch")
    return "\n".join(
        f"export {name}={shlex.quote(environment[name])}" for name in expected
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sfi-root", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument(
        "--architecture", choices=("sm80", "sm90", "sm100"), required=True
    )
    parser.add_argument("--checker", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    checker = args.checker or args.sfi_root / "scripts/check_flash_attention_build_provenance.py"
    try:
        environment = resolve_toolchain(
            python=Path(sys.executable),
            checker=checker,
            sfi_root=args.sfi_root,
            target=args.target,
            architecture=args.architecture,
        )
        output = render_shell_exports(environment)
    except (OSError, ToolchainPreflightError) as exc:
        print(f"FAIL: CUDA toolchain preflight: {exc}", file=sys.stderr)
        return 78
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
