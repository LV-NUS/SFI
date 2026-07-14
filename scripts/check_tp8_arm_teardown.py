#!/usr/bin/env python3
"""Fail closed when a TP8 benchmark arm leaves GPU or runtime workers behind."""

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
import tempfile
import time
from pathlib import Path
from typing import Any


BASELINE_SCHEMA = "sfi.tp8_process_baseline.v1"
TEARDOWN_SCHEMA = "sfi.tp8_arm_teardown.v1"
ARM_TOKEN_ENV = "SFI_TP8_ARM_TOKEN"
ATTRIBUTION_SCOPE = "selected_gpu_or_exact_child_session_or_arm_token"


def derive_arm_token(run_nonce: str, arm_tag: str) -> str:
    """Derive one process-only token from the already verified run nonce."""
    if not run_nonce or not arm_tag:
        raise ValueError("run_nonce and arm_tag are required for TP8 arm identity")
    material = "\0".join(("sfi.tp8.arm.v1", run_nonce, arm_tag))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def arm_token_sha256(arm_token: str) -> str:
    if not arm_token:
        raise ValueError("arm token is required")
    return hashlib.sha256(arm_token.encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def _load_json_object(path: Path) -> dict[str, Any]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    with os.fdopen(fd, "r", encoding="utf-8") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise ValueError(f"not a regular file: {path}")
        payload = json.load(handle, parse_constant=_reject_json_constant)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return payload


def write_evidence(path: Path, payload: dict[str, Any]) -> None:
    requested = Path(path)
    if requested.is_symlink():
        raise ValueError(f"refusing symlink output: {requested}")
    output = requested.resolve(strict=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=str(output.parent), text=True
    )
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def _run_nvidia_smi(arguments: list[str]) -> list[str]:
    result = subprocess.run(
        ["nvidia-smi", *arguments],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no diagnostic"
        raise RuntimeError(f"nvidia-smi rc={result.returncode}: {detail}")
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _process_stat_record(pid: int) -> dict[str, Any] | None:
    proc = Path("/proc") / str(pid)
    try:
        proc_stat = proc.stat()
        stat_text = (proc / "stat").read_text(encoding="utf-8")
        right_paren = stat_text.rfind(")")
        if right_paren < 0:
            raise ValueError("malformed proc stat")
        stat_fields = stat_text[right_paren + 2 :].split()
        if len(stat_fields) <= 19:
            raise ValueError("proc stat is missing starttime")
        return {
            "pid": int(pid),
            "uid": int(proc_stat.st_uid),
            "ppid": int(stat_fields[1]),
            "process_group_id": int(stat_fields[2]),
            "session_id": int(stat_fields[3]),
            "start_time_ticks": int(stat_fields[19]),
        }
    except FileNotFoundError:
        return None


def _process_arm_token_matches(pid: int, expected_arm_token: str) -> bool:
    if not expected_arm_token:
        return False
    proc = Path("/proc") / str(pid)
    try:
        environ = (proc / "environ").read_bytes()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return False
    expected = f"{ARM_TOKEN_ENV}={expected_arm_token}".encode("utf-8")
    return expected in environ.split(b"\0")


def _process_record(
    stat_record: dict[str, Any],
    *,
    arm_token_match: bool,
) -> dict[str, Any] | None:
    pid = int(stat_record["pid"])
    proc = Path("/proc") / str(pid)
    try:
        comm = (proc / "comm").read_text(encoding="utf-8").strip()
        cmdline_bytes = (proc / "cmdline").read_bytes()
    except FileNotFoundError:
        return None
    return {
        **stat_record,
        "comm": comm,
        "cmdline_sha256": hashlib.sha256(cmdline_bytes).hexdigest(),
        "arm_token_match": bool(arm_token_match),
    }


def _attributed_runtime_processes(
    *,
    child_session_id: int | None,
    arm_token: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Return only processes owned by this exact launched arm.

    Selected-GPU ownership is checked separately through nvidia-smi.  The
    process supplement must not turn unrelated users or other GPU jobs into a
    teardown failure, so it uses the child session and inherited arm token
    instead of a machine-wide command-line substring.
    """
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        proc_entries = list(Path("/proc").iterdir())
    except OSError as exc:
        return [], [f"proc_scan_failed:{exc}"]
    for entry in proc_entries:
        if not entry.name.isdigit():
            continue
        try:
            stat_record = _process_stat_record(int(entry.name))
        except (OSError, ValueError) as exc:
            # An unrelated or protected process is outside this arm's owner
            # boundary. Selected-GPU processes remain fail-closed in
            # ``_gpu_state`` regardless of UID.
            continue
        if stat_record is None:
            continue
        session_match = bool(
            child_session_id is not None
            and int(stat_record["session_id"]) == int(child_session_id)
        )
        token_match = False
        if int(stat_record["uid"]) == int(os.geteuid()):
            token_match = _process_arm_token_matches(
                int(stat_record["pid"]), arm_token
            )
        if not (session_match or token_match):
            continue
        try:
            record = _process_record(
                stat_record,
                arm_token_match=token_match,
            )
        except (OSError, ValueError) as exc:
            errors.append(
                f"attributed_process_inspection_failed:{entry.name}:{exc}"
            )
            continue
        if record is not None:
            records.append(record)
    records.sort(key=lambda item: (int(item["pid"]), int(item["start_time_ticks"])))
    return records, errors


def _gpu_state(
    gpu_ids: list[int],
    *,
    arm_token: str = "",
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    gpu_lines = _run_nvidia_smi(
        ["--query-gpu=index,uuid", "--format=csv,noheader,nounits"]
    )
    index_to_uuid: dict[int, str] = {}
    for line in gpu_lines:
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2:
            raise RuntimeError(f"malformed GPU identity row: {line!r}")
        index_to_uuid[int(fields[0])] = fields[1]
    missing = sorted(set(gpu_ids) - set(index_to_uuid))
    if missing:
        raise RuntimeError(f"selected GPU indices unavailable: {missing}")
    selected_uuids = {index_to_uuid[index]: str(index) for index in gpu_ids}

    compute_lines = _run_nvidia_smi(
        ["--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits"]
    )
    selected_processes: list[dict[str, Any]] = []
    for line in compute_lines:
        if line.casefold().startswith("no running processes"):
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2:
            raise RuntimeError(f"malformed compute-process row: {line!r}")
        pid_text, gpu_uuid = fields
        if gpu_uuid not in selected_uuids:
            continue
        try:
            pid = int(pid_text)
        except ValueError as exc:
            raise RuntimeError(f"invalid compute-process PID: {line!r}") from exc
        stat_record = _process_stat_record(pid)
        token_match = bool(
            stat_record is not None
            and int(stat_record["uid"]) == int(os.geteuid())
            and _process_arm_token_matches(pid, arm_token)
        )
        record = (
            _process_record(stat_record, arm_token_match=token_match)
            if stat_record is not None
            else None
        ) or {
            "pid": pid,
            "start_time_ticks": None,
            "session_id": None,
            "comm": "",
            "cmdline_sha256": "",
            "arm_token_match": False,
            "process_vanished_after_gpu_query": True,
        }
        record.update({"gpu_index": int(selected_uuids[gpu_uuid]), "gpu_uuid": gpu_uuid})
        selected_processes.append(record)
    selected_processes.sort(key=lambda item: (int(item["gpu_index"]), int(item["pid"])))
    return selected_processes, {str(index): index_to_uuid[index] for index in gpu_ids}


def _snapshot(
    gpu_ids: list[int],
    *,
    child_session_id: int | None = None,
    arm_token: str = "",
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        compute_processes, gpu_uuids = _gpu_state(
            gpu_ids,
            arm_token=arm_token,
        )
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        compute_processes, gpu_uuids = [], {}
        errors.append(f"gpu_inspection_failed:{exc}")
    attributed_processes: list[dict[str, Any]] = []
    if child_session_id is not None or arm_token:
        attributed_processes, attributed_errors = _attributed_runtime_processes(
            child_session_id=child_session_id,
            arm_token=arm_token,
        )
        errors.extend(attributed_errors)
    return {
        "selected_gpu_compute_processes": compute_processes,
        "selected_gpu_uuids": gpu_uuids,
        "attributed_runtime_processes": attributed_processes,
        "inspection_errors": errors,
    }


def _capture_stable_idle(
    gpu_ids: list[int],
    *,
    child_session_id: int | None,
    arm_token: str,
    timeout_s: float,
    poll_interval_s: float,
    stable_clean_polls_required: int,
) -> tuple[dict[str, Any], int, int]:
    deadline = time.monotonic() + max(0.0, float(timeout_s))
    stable_clean_polls = 0
    poll_count = 0
    final_snapshot: dict[str, Any] = {}
    while True:
        poll_count += 1
        final_snapshot = _snapshot(
            gpu_ids,
            child_session_id=child_session_id,
            arm_token=arm_token,
        )
        clean = not (
            final_snapshot["inspection_errors"]
            or final_snapshot["selected_gpu_compute_processes"]
            or final_snapshot["attributed_runtime_processes"]
        )
        stable_clean_polls = stable_clean_polls + 1 if clean else 0
        if stable_clean_polls >= stable_clean_polls_required:
            break
        if time.monotonic() >= deadline:
            break
        time.sleep(max(0.0, float(poll_interval_s)))
    return final_snapshot, poll_count, stable_clean_polls


def capture_baseline(
    gpu_ids: list[int],
    *,
    timeout_s: float = 30.0,
    poll_interval_s: float = 1.0,
    stable_clean_polls_required: int = 2,
) -> tuple[dict[str, Any], list[str]]:
    snapshot, poll_count, stable_clean_polls = _capture_stable_idle(
        gpu_ids,
        child_session_id=None,
        arm_token="",
        timeout_s=timeout_s,
        poll_interval_s=poll_interval_s,
        stable_clean_polls_required=stable_clean_polls_required,
    )
    reasons = list(snapshot["inspection_errors"])
    if snapshot["selected_gpu_compute_processes"]:
        reasons.append("selected_gpus_not_idle_before_pair")
    if stable_clean_polls < stable_clean_polls_required:
        reasons.append("stable_clean_baseline_not_observed")
    payload = {
        "schema": BASELINE_SCHEMA,
        "status": "pass" if not reasons else "fail",
        "captured_at_ns": time.time_ns(),
        "captured_monotonic_ns": time.monotonic_ns(),
        "selected_gpu_indices": gpu_ids,
        "attribution_scope": "selected_gpu_idle",
        "timeout_s": float(timeout_s),
        "poll_count": int(poll_count),
        "stable_clean_polls_required": int(stable_clean_polls_required),
        "stable_clean_polls_observed": int(stable_clean_polls),
        **snapshot,
        "reasons": reasons,
    }
    return payload, reasons


def capture_teardown(
    gpu_ids: list[int],
    *,
    baseline_path: Path,
    arm_tag: str,
    child_session_id: int | None,
    arm_token: str,
    timeout_s: float = 30.0,
    poll_interval_s: float = 1.0,
    stable_clean_polls_required: int = 2,
) -> tuple[dict[str, Any], list[str]]:
    baseline = _load_json_object(baseline_path)
    baseline_sha256 = _sha256(baseline_path)
    setup_reasons: list[str] = []
    if baseline.get("schema") != BASELINE_SCHEMA:
        setup_reasons.append("baseline_schema_mismatch")
    if baseline.get("status") != "pass" or baseline.get("reasons"):
        setup_reasons.append("baseline_not_passed")
    if baseline.get("selected_gpu_indices") != gpu_ids:
        setup_reasons.append("baseline_gpu_selection_mismatch")
    baseline_uuids = baseline.get("selected_gpu_uuids")
    if not isinstance(baseline_uuids, dict):
        setup_reasons.append("baseline_gpu_uuids_invalid")
    if child_session_id is not None and (
        isinstance(child_session_id, bool)
        or not isinstance(child_session_id, int)
        or child_session_id <= 0
    ):
        setup_reasons.append("child_session_id_invalid")
    if re.fullmatch(r"[0-9a-f]{64}", arm_token) is None:
        setup_reasons.append("arm_token_invalid")

    final_snapshot, poll_count, stable_clean_polls = _capture_stable_idle(
        gpu_ids,
        child_session_id=child_session_id,
        arm_token=arm_token,
        timeout_s=timeout_s,
        poll_interval_s=poll_interval_s,
        stable_clean_polls_required=stable_clean_polls_required,
    )
    if isinstance(baseline_uuids, dict) and (
        final_snapshot.get("selected_gpu_uuids") != baseline_uuids
    ):
        setup_reasons.append("selected_gpu_uuid_identity_changed")

    reasons = list(setup_reasons)
    reasons.extend(final_snapshot.get("inspection_errors", []))
    if final_snapshot.get("selected_gpu_compute_processes"):
        reasons.append("selected_gpu_compute_processes_survived_arm")
    if final_snapshot.get("attributed_runtime_processes"):
        reasons.append("attributed_runtime_processes_survived_arm")
    if stable_clean_polls < stable_clean_polls_required:
        reasons.append("stable_clean_teardown_not_observed")
    payload = {
        "schema": TEARDOWN_SCHEMA,
        "status": "pass" if not reasons else "fail",
        "arm_tag": arm_tag,
        "captured_at_ns": time.time_ns(),
        "captured_monotonic_ns": time.monotonic_ns(),
        "selected_gpu_indices": gpu_ids,
        "selected_gpu_uuids": final_snapshot.get("selected_gpu_uuids", {}),
        "attribution_scope": ATTRIBUTION_SCOPE,
        "child_session_id": child_session_id,
        "arm_token_sha256": arm_token_sha256(arm_token),
        "baseline_path": str(baseline_path.resolve(strict=False)),
        "baseline_sha256": baseline_sha256,
        "timeout_s": float(timeout_s),
        "poll_count": poll_count,
        "stable_clean_polls_required": stable_clean_polls_required,
        "stable_clean_polls_observed": stable_clean_polls,
        "selected_gpu_compute_processes": final_snapshot.get(
            "selected_gpu_compute_processes", []
        ),
        "attributed_runtime_processes": final_snapshot.get(
            "attributed_runtime_processes", []
        ),
        "inspection_errors": final_snapshot.get("inspection_errors", []),
        "reasons": reasons,
    }
    return payload, reasons


def _parse_gpu_list(value: str) -> list[int]:
    fields = value.split(",")
    if len(fields) != 8 or any(not field.isdigit() for field in fields):
        raise argparse.ArgumentTypeError("GPU list must contain exactly eight integer IDs")
    values = [int(field) for field in fields]
    if len(set(values)) != 8:
        raise argparse.ArgumentTypeError("GPU list contains duplicate IDs")
    return values


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-list", required=True, type=_parse_gpu_list)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--arm-tag")
    parser.add_argument("--child-session-id", type=int)
    parser.add_argument("--run-nonce")
    parser.add_argument("--timeout-s", type=float, default=30.0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not math.isfinite(args.timeout_s) or args.timeout_s < 2 or args.timeout_s > 60:
        print("FAIL: --timeout-s must be finite and in [2, 60]", file=sys.stderr)
        return 64
    try:
        if args.baseline is None:
            if (
                args.arm_tag
                or args.child_session_id is not None
                or args.run_nonce is not None
            ):
                print(
                    "FAIL: arm identity arguments are only valid with --baseline",
                    file=sys.stderr,
                )
                return 64
            payload, reasons = capture_baseline(
                args.gpu_list,
                timeout_s=args.timeout_s,
            )
        else:
            if not args.arm_tag:
                print("FAIL: --arm-tag is required with --baseline", file=sys.stderr)
                return 64
            if args.child_session_id is None or args.child_session_id <= 0:
                print(
                    "FAIL: positive --child-session-id is required with --baseline",
                    file=sys.stderr,
                )
                return 64
            run_nonce = str(args.run_nonce or "")
            if not run_nonce:
                print(
                    "FAIL: --run-nonce is required with --baseline",
                    file=sys.stderr,
                )
                return 64
            arm_token = derive_arm_token(run_nonce, args.arm_tag)
            payload, reasons = capture_teardown(
                args.gpu_list,
                baseline_path=args.baseline,
                arm_tag=args.arm_tag,
                child_session_id=args.child_session_id,
                arm_token=arm_token,
                timeout_s=args.timeout_s,
            )
        write_evidence(args.output, payload)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"FAIL: TP8 teardown inspection failed: {exc}", file=sys.stderr)
        return 1
    if reasons:
        print(f"FAIL: TP8 teardown gate rejected: {reasons}", file=sys.stderr)
        print(f"evidence={args.output}")
        return 1
    print(f"PASS: TP8 teardown gate accepted evidence={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
