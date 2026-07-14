#!/usr/bin/env python3
"""Bind an official LongBench-v2 run to one immutable cached dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
import re
from typing import Any


DATASET_NAME = "THUDM/LongBench-v2"
DATASET_SPLIT = "train"
EXPECTED_ROWS = 503
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")


def _require_hex(value: str, *, pattern: re.Pattern[str], label: str) -> str:
    normalized = value.strip().lower()
    if pattern.fullmatch(normalized) is None:
        raise ValueError(f"{label} must be an immutable lowercase hexadecimal digest")
    return normalized


def summarize_dataset(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    content_digest = hashlib.sha256()
    ids_digest = hashlib.sha256()
    ids: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"dataset row {index} is not a mapping")
        sample_id = row.get("_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"dataset row {index} has no nonempty string _id")
        encoded = json.dumps(
            dict(row),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        content_digest.update(len(encoded).to_bytes(8, "big"))
        content_digest.update(encoded)
        sample_id_bytes = sample_id.encode("utf-8")
        ids_digest.update(len(sample_id_bytes).to_bytes(8, "big"))
        ids_digest.update(sample_id_bytes)
        ids.append(sample_id)
    if len(ids) != EXPECTED_ROWS:
        raise ValueError(
            f"LongBench-v2 row count is {len(ids)}, expected {EXPECTED_ROWS}"
        )
    if len(ids) != len(set(ids)):
        raise ValueError("LongBench-v2 dataset contains duplicate _id values")
    sorted_ids_digest = hashlib.sha256()
    for sample_id in sorted(ids):
        sample_id_bytes = sample_id.encode("utf-8")
        sorted_ids_digest.update(len(sample_id_bytes).to_bytes(8, "big"))
        sorted_ids_digest.update(sample_id_bytes)
    return {
        "rows": len(ids),
        "unique_ids": len(set(ids)),
        "content_sha256": content_digest.hexdigest(),
        "ordered_ids_sha256": ids_digest.hexdigest(),
        "sorted_ids_sha256": sorted_ids_digest.hexdigest(),
    }


def build_dataset_identity(
    *,
    revision: str,
    expected_sha256: str,
    loader: Callable[..., Any],
) -> dict[str, Any]:
    revision = _require_hex(
        revision, pattern=_HEX40, label="LONGBENCH_DATASET_REVISION"
    )
    expected_sha256 = _require_hex(
        expected_sha256, pattern=_HEX64, label="LONGBENCH_DATASET_SHA256"
    )
    pinned = loader(DATASET_NAME, revision=revision, split=DATASET_SPLIT)
    default = loader(DATASET_NAME, split=DATASET_SPLIT)
    pinned_summary = summarize_dataset(pinned)
    default_summary = summarize_dataset(default)
    if pinned_summary != default_summary:
        raise ValueError(
            "official unpinned load_dataset result differs from the requested revision"
        )
    if pinned_summary["content_sha256"] != expected_sha256:
        raise ValueError(
            "LongBench-v2 content SHA256 mismatch: "
            f"actual={pinned_summary['content_sha256']} expected={expected_sha256}"
        )
    pinned_fingerprint = str(getattr(pinned, "_fingerprint", "") or "")
    default_fingerprint = str(getattr(default, "_fingerprint", "") or "")
    if not pinned_fingerprint or not default_fingerprint:
        raise ValueError("datasets cache fingerprint is unavailable")
    return {
        "schema": 1,
        "dataset": DATASET_NAME,
        "split": DATASET_SPLIT,
        "revision": revision,
        "expected_content_sha256": expected_sha256,
        "verified_content_sha256": pinned_summary["content_sha256"],
        "ordered_ids_sha256": pinned_summary["ordered_ids_sha256"],
        "sorted_ids_sha256": pinned_summary["sorted_ids_sha256"],
        "rows": pinned_summary["rows"],
        "unique_ids": pinned_summary["unique_ids"],
        "pinned_cache_fingerprint": pinned_fingerprint,
        "official_default_cache_fingerprint": default_fingerprint,
        "official_default_matches_pinned": True,
        "offline_cache_only": True,
    }


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    for variable in ("HF_DATASETS_OFFLINE", "HF_HUB_OFFLINE"):
        if os.environ.get(variable) != "1":
            parser.error(f"{variable}=1 is required for immutable cache-only evaluation")

    try:
        from datasets import load_dataset

        payload = build_dataset_identity(
            revision=args.revision,
            expected_sha256=args.expected_sha256,
            loader=load_dataset,
        )
        _write_json_atomic(args.output.resolve(), payload)
    except (ImportError, OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(
        "LongBench-v2 dataset identity passed: "
        f"revision={payload['revision']} sha256={payload['verified_content_sha256']} "
        f"rows={payload['rows']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
