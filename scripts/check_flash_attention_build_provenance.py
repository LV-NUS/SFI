#!/usr/bin/env python3
"""Validate the exact FlashAttention source/build consumed by a runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
    }
    missing = sorted(required - assignments.keys())
    if missing:
        raise BuildProvenanceError(f"setup identity assignments missing: {missing}")
    return assignments


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
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

    payload = _strict_json(provenance_path)
    _require_equal(payload, "schema_version", 3)
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

    provenance_python = Path(str(payload.get("python_executable", "") or ""))
    if not provenance_python.is_absolute() or provenance_python.resolve() != Path(
        sys.executable
    ).resolve():
        raise BuildProvenanceError(
            "build provenance Python differs from the benchmark interpreter: "
            f"build={provenance_python} benchmark={Path(sys.executable).resolve()}"
        )
    try:
        import torch
    except Exception as exc:
        raise BuildProvenanceError(f"cannot import torch with benchmark Python: {exc}") from exc
    _require_equal(payload, "torch_version", str(torch.__version__))
    _require_equal(payload, "torch_cuda_version", str(torch.version.cuda or ""))

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
    index_tree = _git(target, "write-tree")
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
    if head_tree == expected_tree and index_tree == expected_tree:
        if source_mode != "verified_patched_tree":
            raise BuildProvenanceError(
                f"committed patched tree has inconsistent source_mode={source_mode!r}"
            )
        source_state = "committed_patched_tree"
    elif head_commit == assignments["BASE_COMMIT"] and index_tree == expected_tree:
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
            f"head={head_commit} head_tree={head_tree} index_tree={index_tree} "
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
