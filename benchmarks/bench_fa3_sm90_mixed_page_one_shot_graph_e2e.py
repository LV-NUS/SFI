from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_FA3_ROOT = REPO_ROOT / "third_party_upstreams" / "vllm-project-flash-attention"
REQUIRED_SUMMARY_FIELDS = (
    "gate_passed",
    "no_fa4_cute_dispatch",
    "page_resolver_kind0_count",
    "page_resolver_kind1_count",
    "page_resolver_kind4_count",
    "effective_visible_k_by_row",
    "dense_reference_match",
    "unsupported_fallback_count",
)


def _extension_provenance() -> dict[str, str]:
    package_dir = UPSTREAM_FA3_ROOT / "vllm_flash_attn"
    matches = sorted([*package_dir.glob("_vllm_fa3_C*.so"), *package_dir.glob("_vllm_fa3_C*.pyd")])
    if not matches:
        raise RuntimeError("missing _vllm_fa3_C extension")
    so_path = matches[-1]
    return {
        "extension_path": str(so_path),
        "extension_sha256": hashlib.sha256(so_path.read_bytes()).hexdigest(),
    }


def _parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="H100 FA3 SM90 mixed-page sparse decode e2e gate."
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--mode", default="sparse")
    parser.add_argument("--producer-mode", default="full-open-gt1")
    parser.add_argument("--preset", default="bs2long-cap128")
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--full-cuda-graph", action="store_true", default=True)
    parser.add_argument("--outputs-include-text", action="store_true", default=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--timeout-s", type=int, default=600)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    if args.mode != "sparse":
        parser.error("--mode must be sparse for the FA3 SM90 gate")
    if args.warmup != 0:
        parser.error("--warmup must be 0 for correctness gate")
    if not bool(args.full_cuda_graph):
        parser.error("--full-cuda-graph is required for the FA3 SM90 gate")
    if not bool(args.outputs_include_text):
        parser.error("--outputs-include-text is required for the FA3 SM90 gate")
    return args


def _run_sm80_structural_runner(args: argparse.Namespace) -> None:
    runner = REPO_ROOT / "benchmarks" / "bench_sm80_mixed_page_one_shot_graph_e2e.py"
    forwarded = [
        str(runner),
        "--output",
        str(args.output),
        "--summary-output",
        str(args.summary_output),
        "--mode",
        "sparse",
        "--producer-mode",
        str(args.producer_mode),
        "--backend",
        "fa3",
        "--preset",
        str(args.preset),
        "--warmup",
        "0",
        "--full-cuda-graph",
        "--outputs-include-text",
        "--model",
        str(args.model),
        "--python",
        str(args.python),
        "--timeout-s",
        str(args.timeout_s),
    ]
    old_argv = sys.argv
    try:
        sys.argv = forwarded
        try:
            runpy.run_path(str(runner), run_name="__main__")
        except SystemExit as exc:
            if exc.code not in (0, None):
                raise
    finally:
        sys.argv = old_argv


def _load_summary(summary_output: str) -> dict[str, object]:
    summary_path = Path(summary_output)
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _write_summary(summary_output: str, summary: dict[str, object]) -> None:
    Path(summary_output).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _as_int(value: object, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _first_present(summary: dict[str, object], key: str) -> tuple[bool, object]:
    containers: list[object] = [
        summary,
        summary.get("record"),
        summary.get("route_proof"),
        summary.get("route_summary"),
        summary.get("producer_route_summary"),
        summary.get("metrics"),
    ]
    for container in containers:
        if isinstance(container, dict) and key in container:
            return True, container[key]
    return False, None


def _require_present(summary: dict[str, object], key: str, *aliases: str) -> object:
    for candidate in (key, *aliases):
        found, value = _first_present(summary, candidate)
        if found:
            return value
    searched = ", ".join((key, *aliases))
    raise RuntimeError(f"missing required FA3 SM90 e2e summary field: {searched}")


def _require_counter(summary: dict[str, object], key: str, *aliases: str) -> int:
    value = _require_present(summary, key, *aliases)
    parsed = _as_int(value, -1)
    if parsed < 0:
        raise RuntimeError(f"invalid FA3 SM90 e2e counter {key}: {value!r}")
    return parsed


def _require_bool(summary: dict[str, object], key: str, *aliases: str) -> bool:
    value = _require_present(summary, key, *aliases)
    if isinstance(value, bool):
        return bool(value)
    raise RuntimeError(f"invalid FA3 SM90 e2e bool field {key}: {value!r}")


def _require_fallback_counter(summary: dict[str, object], key: str) -> int:
    found, value = _first_present(summary, key)
    if not found:
        raise RuntimeError(f"missing required FA3 SM90 fallback counter: {key}")
    parsed = _as_int(value, -1)
    if parsed < 0:
        raise RuntimeError(f"invalid FA3 SM90 fallback counter {key}: {value!r}")
    return parsed


def _unsupported_fallback_count(summary: dict[str, object]) -> int:
    keys = (
        "kind2_dispatch_count",
        "kind3_dispatch_count",
        "vector_fallback_rows",
        "full_fallback_rows",
        "writer_vector_fallback_count",
        "gt1_scalar_fallback_count",
    )
    return sum(_require_fallback_counter(summary, key) for key in keys)


def _dense_reference_match(summary: dict[str, object]) -> bool:
    found, explicit = _first_present(summary, "dense_reference_match")
    if found:
        if isinstance(explicit, bool):
            return bool(explicit)
        raise RuntimeError(f"invalid FA3 SM90 e2e bool field dense_reference_match: {explicit!r}")
    found_skip, skip_value = _first_present(summary, "skip_dense_reference")
    if found_skip and bool(skip_value):
        return False
    return _require_bool(summary, "reference_semantic_match")


def _effective_visible_k_by_row(summary: dict[str, object]) -> list[int]:
    for key in ("effective_visible_k_by_row", "rrp_row_effective_k_by_row"):
        found, value = _first_present(summary, key)
        if found:
            if isinstance(value, list):
                parsed = [_as_int(item, -1) for item in value]
                if any(item < 0 for item in parsed):
                    raise RuntimeError(f"invalid FA3 SM90 visible-K field {key}: {value!r}")
                return parsed
            raise RuntimeError(f"invalid FA3 SM90 visible-K field {key}: {value!r}")
    raise RuntimeError(
        "missing required FA3 SM90 e2e summary field: "
        "effective_visible_k_by_row, rrp_row_effective_k_by_row"
    )


def _normalize_route_fields(summary: dict[str, object]) -> dict[str, object]:
    normalized = dict(summary)
    normalized["page_resolver_kind0_count"] = _require_counter(
        summary,
        "page_resolver_kind0_count",
    )
    normalized["page_resolver_kind1_count"] = _require_counter(
        summary,
        "page_resolver_kind1_count",
    )
    normalized["page_resolver_kind4_count"] = _require_counter(
        summary,
        "page_resolver_kind4_count",
    )
    normalized["effective_visible_k_by_row"] = _effective_visible_k_by_row(summary)
    normalized["dense_reference_match"] = _dense_reference_match(summary)
    normalized["unsupported_fallback_count"] = _unsupported_fallback_count(summary)
    normalized["no_fa4_cute_dispatch"] = _require_bool(summary, "no_fa4_cute_dispatch")
    return normalized


def _validate_summary(summary: dict[str, object]) -> None:
    missing = [field for field in REQUIRED_SUMMARY_FIELDS if field not in summary]
    if missing:
        raise RuntimeError(f"missing FA3 SM90 e2e summary fields: {missing}")
    if not bool(summary["gate_passed"]):
        raise RuntimeError("FA3 SM90 mixed_page e2e requires underlying gate_passed")
    if int(summary["page_resolver_kind4_count"]) <= 0:
        raise RuntimeError("FA3 SM90 mixed_page e2e requires page_resolver_kind4_count > 0")
    visible_k = summary["effective_visible_k_by_row"]
    if not isinstance(visible_k, list) or not visible_k or any(_as_int(value, -1) <= 0 for value in visible_k):
        raise RuntimeError("FA3 SM90 mixed_page e2e requires effective_visible_k_by_row")
    if not bool(summary["dense_reference_match"]):
        raise RuntimeError("FA3 SM90 mixed_page e2e requires dense_reference_match")
    if int(summary["unsupported_fallback_count"]) != 0:
        raise RuntimeError("FA3 SM90 mixed_page e2e requires unsupported_fallback_count == 0")
    if not bool(summary["no_fa4_cute_dispatch"]):
        raise RuntimeError("FA3 SM90 mixed_page e2e requires no_fa4_cute_dispatch")


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    if not (torch.cuda.get_device_capability() == (9, 0)):
        raise RuntimeError("FA3 SM90 mixed_page e2e gate requires H100 sm90")

    os.environ["VLLM_FLASH_ATTN_VERSION"] = "3"
    os.environ["FA3_SM90_MIXED_PAGE_E2E_GATE"] = "1"
    provenance = _extension_provenance()
    _run_sm80_structural_runner(args)

    summary = _normalize_route_fields(_load_summary(str(args.summary_output)))
    summary.update(
        {
            "case": "fa3_sm90_mixed_page_one_shot_graph_e2e",
            "backend": "fa3_sm90_mixed_page",
            "full_cuda_graph": True,
            "skip_dense_reference": False,
            "outputs_include_text": True,
            **provenance,
        }
    )
    _validate_summary(summary)
    _write_summary(str(args.summary_output), summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
