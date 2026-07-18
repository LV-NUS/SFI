#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import struct
import tempfile
import urllib.error
import urllib.parse
import urllib.request


RESET_METHOD = "sfi_reset_sparse_route_counters_for_measurement"
SNAPSHOT_METHOD = "sfi_snapshot_sparse_route_counters_after_measurement"
FIELD_COUNT = 10
STORAGE = "worker_local"
SCHEMA = "sfi.fa3_route_counter.serve_snapshot.v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze worker-local sparse route counters through vLLM's "
            "control-plane collective RPC."
        )
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--tp-size", type=int, required=True)
    parser.add_argument("--phase", choices=("reset", "snapshot"), required=True)
    parser.add_argument("--json-output", required=True)
    parser.add_argument("--binary-output", required=True)
    parser.add_argument("--reset-json", default="")
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()
    if args.tp_size <= 0:
        parser.error("--tp-size must be positive")
    if not args.api_key or any(char.isspace() for char in args.api_key):
        parser.error("--api-key must be nonempty and contain no whitespace")
    if not (0.0 < args.timeout <= 300.0):
        parser.error("--timeout must be in (0, 300]")
    if args.phase == "snapshot" and not args.reset_json:
        parser.error("snapshot phase requires --reset-json")
    if args.phase == "reset" and args.reset_json:
        parser.error("reset phase does not accept --reset-json")
    output_paths = {
        Path(args.json_output).resolve(),
        Path(args.binary_output).resolve(),
    }
    if len(output_paths) != 2:
        parser.error("--json-output and --binary-output must be different paths")
    if args.reset_json and Path(args.reset_json).resolve() in output_paths:
        parser.error("--reset-json must differ from both output paths")
    return args


def _loopback_collective_rpc_url(base_url: str) -> str:
    parsed = urllib.parse.urlsplit(base_url.rstrip("/"))
    if parsed.scheme != "http" or parsed.username or parsed.password:
        raise RuntimeError(
            "route-counter RPC requires plain HTTP without URL credentials"
        )
    if parsed.query or parsed.fragment or parsed.path not in {"", "/"}:
        raise RuntimeError("route-counter RPC base URL must not contain a path/query")
    host = parsed.hostname
    if host is None:
        raise RuntimeError("route-counter RPC URL has no host")
    normalized_host = "127.0.0.1" if host == "localhost" else host
    try:
        is_loopback = ipaddress.ip_address(normalized_host).is_loopback
    except ValueError as exc:
        raise RuntimeError(
            "route-counter RPC host must be a loopback IP literal or localhost"
        ) from exc
    if not is_loopback:
        raise RuntimeError(
            "route-counter RPC is intentionally restricted to loopback"
        )
    try:
        port = parsed.port
    except ValueError as exc:
        raise RuntimeError("route-counter RPC URL has an invalid port") from exc
    if port is not None and not (0 < port < 65536):
        raise RuntimeError("route-counter RPC URL port is out of range")
    return urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, "/collective_rpc", "", "")
    )


def _collective_rpc(
    *,
    base_url: str,
    api_key: str,
    method: str,
    timeout: float,
) -> list[object]:
    url = _loopback_collective_rpc_url(base_url)
    body = json.dumps(
        {"method": method, "timeout": float(timeout)},
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout + 5.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"route-counter collective RPC HTTP {exc.code}: {detail}"
        ) from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"route-counter collective RPC failed: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise RuntimeError("route-counter collective RPC returned an invalid schema")
    return list(payload["results"])


def _validated_records(
    raw_records: list[object],
    *,
    tp_size: int,
    require_zero: bool,
) -> list[dict[str, object]]:
    if len(raw_records) != tp_size:
        raise RuntimeError(
            f"route-counter worker count mismatch: {len(raw_records)} != {tp_size}"
        )
    records: list[dict[str, object]] = []
    for index, raw_record in enumerate(raw_records):
        if not isinstance(raw_record, dict):
            raise RuntimeError(
                f"route-counter record[{index}] is {type(raw_record).__name__}"
            )
        record = dict(raw_record)
        rank = record.get("rank")
        values = record.get("values")
        if isinstance(rank, bool) or not isinstance(rank, int):
            raise RuntimeError(f"route-counter record[{index}] has invalid rank")
        if record.get("slot_count") != tp_size:
            raise RuntimeError(f"route-counter rank {rank} has invalid slot_count")
        if record.get("field_count") != FIELD_COUNT:
            raise RuntimeError(f"route-counter rank {rank} has invalid field_count")
        if record.get("storage") != STORAGE:
            raise RuntimeError(f"route-counter rank {rank} is not worker-local")
        if (
            not isinstance(values, list)
            or len(values) != FIELD_COUNT
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in values
            )
        ):
            raise RuntimeError(f"route-counter rank {rank} has invalid values")
        if require_zero and any(values):
            raise RuntimeError(f"route-counter rank {rank} reset remained nonzero")
        pid = record.get("pid")
        if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
            raise RuntimeError(f"route-counter rank {rank} has invalid pid")
        records.append(record)
    records.sort(key=lambda record: int(record["rank"]))
    if [record["rank"] for record in records] != list(range(tp_size)):
        raise RuntimeError("route-counter records do not cover each TP rank exactly once")
    if len({int(record["pid"]) for record in records}) != tp_size:
        raise RuntimeError("route-counter worker PIDs are not unique")
    return records


def _assert_worker_identity(
    records: list[dict[str, object]],
    reset_json: str,
    *,
    tp_size: int,
) -> None:
    payload = json.loads(Path(reset_json).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("reset snapshot is not a JSON object")
    raw_reset_records = payload.get("records")
    if not isinstance(raw_reset_records, list):
        raise RuntimeError("reset snapshot has an invalid record schema")
    reset_records = _validated_records(
        raw_reset_records,
        tp_size=tp_size,
        require_zero=True,
    )
    if payload != _snapshot_payload(reset_records, phase="reset"):
        raise RuntimeError("reset snapshot metadata or digest is invalid")
    reset_identity = [
        (record.get("rank"), record.get("pid"))
        for record in reset_records
        if isinstance(record, dict)
    ]
    snapshot_identity = [
        (record["rank"], record["pid"])
        for record in records
    ]
    if snapshot_identity != reset_identity:
        raise RuntimeError(
            "route-counter worker identity changed between reset and snapshot"
        )


def _snapshot_payload(
    records: list[dict[str, object]],
    *,
    phase: str,
) -> dict[str, object]:
    values_by_rank = [tuple(int(value) for value in record["values"]) for record in records]
    aggregate = tuple(
        sum(values[index] for values in values_by_rank)
        for index in range(FIELD_COUNT)
    )
    binary = _records_binary(records)
    payload: dict[str, object] = {
        "schema": SCHEMA,
        "source": "blocking_http_collective_rpc",
        "measurement_boundary": "collective_rpc_return",
        "phase": phase,
        "records": records,
        "aggregate_values": list(aggregate),
        "rank_consistent": all(
            values == values_by_rank[0] for values in values_by_rank[1:]
        ),
        "binary_size_bytes": len(binary),
        "binary_sha256": hashlib.sha256(binary).hexdigest(),
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    payload["sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def _records_binary(records: list[dict[str, object]]) -> bytes:
    return b"".join(
        struct.pack("10q", *(int(value) for value in record["values"]))
        for record in records
    )


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.tmp-",
            delete=False,
        ) as fh:
            tmp = Path(fh.name)
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if tmp is not None:
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass


def main() -> None:
    args = _parse_args()
    method = RESET_METHOD if args.phase == "reset" else SNAPSHOT_METHOD
    records = _validated_records(
        _collective_rpc(
            base_url=args.base_url,
            api_key=args.api_key,
            method=method,
            timeout=args.timeout,
        ),
        tp_size=args.tp_size,
        require_zero=args.phase == "reset",
    )
    if args.reset_json:
        _assert_worker_identity(records, args.reset_json, tp_size=args.tp_size)
    binary = _records_binary(records)
    payload = _snapshot_payload(records, phase=args.phase)
    json_bytes = (
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    # The JSON is the commit marker: publish the bound binary first, then its
    # metadata/digest. A successful process exit therefore exposes a complete
    # pair without pretending two renames form a filesystem transaction.
    _atomic_write(Path(args.binary_output), binary)
    _atomic_write(Path(args.json_output), json_bytes)
    print(
        f"route-counter {args.phase} snapshot: ranks={len(records)} "
        f"sha256={payload['sha256']}"
    )


if __name__ == "__main__":
    main()
