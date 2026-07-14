#!/usr/bin/env python3
"""Validate and independently reproduce an official LongBench v2 score row."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


EXPECTED_HEADER = (
    "Model",
    "Overall",
    "Easy",
    "Hard",
    "Short",
    "Medium",
    "Long",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_predictions(predictions_dir: Path) -> tuple[Path, list[dict[str, Any]]]:
    files = sorted(predictions_dir.glob("*.jsonl"))
    if len(files) != 1:
        raise ValueError(f"expected exactly one prediction JSONL, found {files}")

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        files[0].read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid prediction JSON at line {line_number}: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise ValueError(f"prediction line {line_number} is not an object")
        if not isinstance(row.get("judge"), bool):
            raise ValueError(f"prediction line {line_number} has no boolean judge")
        if row.get("difficulty") not in {"easy", "hard"}:
            raise ValueError(f"prediction line {line_number} has invalid difficulty")
        if row.get("length") not in {"short", "medium", "long"}:
            raise ValueError(f"prediction line {line_number} has invalid length")
        rows.append(row)
    if not rows:
        raise ValueError("prediction JSONL is empty")
    return files[0], rows


def _rounded_accuracy(rows: list[dict[str, Any]]) -> float:
    if not rows:
        raise ValueError("cannot score an empty LongBench subgroup")
    correct = sum(int(row["judge"]) for row in rows)
    return round(100.0 * correct / len(rows), 1)


def _recompute_scores(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, float], dict[str, int]]:
    groups = {
        "Overall": rows,
        "Easy": [row for row in rows if row["difficulty"] == "easy"],
        "Hard": [row for row in rows if row["difficulty"] == "hard"],
        "Short": [row for row in rows if row["length"] == "short"],
        "Medium": [row for row in rows if row["length"] == "medium"],
        "Long": [row for row in rows if row["length"] == "long"],
    }
    empty = [name for name, group_rows in groups.items() if not group_rows]
    if empty:
        raise ValueError(f"LongBench prediction subgroups are empty: {empty}")
    return (
        {name: _rounded_accuracy(group_rows) for name, group_rows in groups.items()},
        {name: len(group_rows) for name, group_rows in groups.items()},
    )


def _parse_official_result(result_path: Path) -> tuple[str, dict[str, float]]:
    lines = [
        line.rstrip("\r\n")
        for line in result_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(lines) != 2:
        raise ValueError(
            "official result must contain exactly one header and one model row: "
            f"nonempty_lines={len(lines)}"
        )
    header = tuple(lines[0].split("\t"))
    if header != EXPECTED_HEADER:
        raise ValueError(
            f"official result header mismatch: actual={header}, expected={EXPECTED_HEADER}"
        )
    values = lines[1].split("\t")
    if len(values) != len(EXPECTED_HEADER):
        raise ValueError(
            f"official result row has {len(values)} columns, expected {len(EXPECTED_HEADER)}"
        )
    model = values[0]
    if not model:
        raise ValueError("official result model name is empty")
    scores: dict[str, float] = {}
    for name, raw in zip(EXPECTED_HEADER[1:], values[1:]):
        try:
            value = float(raw)
        except ValueError as exc:
            raise ValueError(f"official {name} score is not numeric: {raw!r}") from exc
        if not math.isfinite(value) or not 0.0 <= value <= 100.0:
            raise ValueError(f"official {name} score is out of range: {raw!r}")
        scores[name] = value
    return model, scores


def validate_score(
    *, predictions_dir: Path, result_path: Path, output_path: Path
) -> dict[str, Any]:
    prediction_path, rows = _load_predictions(predictions_dir)
    recomputed, counts = _recompute_scores(rows)
    model, official = _parse_official_result(result_path)
    if model != prediction_path.stem:
        raise ValueError(
            "official result model does not match the prediction filename: "
            f"{model!r} != {prediction_path.stem!r}"
        )
    mismatches = {
        name: {"official": official[name], "recomputed": recomputed[name]}
        for name in recomputed
        if not math.isclose(official[name], recomputed[name], abs_tol=1.0e-9)
    }
    if mismatches:
        raise ValueError(f"official LongBench scores do not match predictions: {mismatches}")

    payload: dict[str, Any] = {
        "schema": 1,
        "status": "passed",
        "model": model,
        "rows": len(rows),
        "counts": counts,
        "scores": official,
        "prediction_file": str(prediction_path.resolve()),
        "prediction_sha256": _sha256(prediction_path),
        "official_result_file": str(result_path.resolve()),
        "official_result_sha256": _sha256(result_path),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, output_path)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        payload = validate_score(
            predictions_dir=args.predictions_dir.resolve(strict=True),
            result_path=args.result.resolve(strict=True),
            output_path=args.output.resolve(),
        )
    except (OSError, ValueError) as exc:
        print(f"FAIL: {exc}")
        return 1
    print(
        "LongBench score validation passed: "
        f"rows={payload['rows']} overall={payload['scores']['Overall']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
