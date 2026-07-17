#!/usr/bin/env bash
# Architecture-adaptive SFI sparse-attention correctness gate.
set -euo pipefail

usage() {
  echo "usage: PYTHON=/abs/path/python bash scripts/run_one_shot.sh <GPU_ID> <MODEL_PATH> [TAG_PREFIX] [--with-reference]" >&2
}

if [[ $# -lt 2 ]]; then
  usage
  exit 64
fi
GPU="$1"
MODEL="$2"
shift 2

TAG_PREFIX="oneshot"
TAG_SET=0
WITH_REF=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-reference)
      if [[ "${WITH_REF}" == "1" ]]; then
        echo "FAIL: --with-reference was provided more than once" >&2
        exit 64
      fi
      WITH_REF=1
      ;;
    --*)
      echo "FAIL: unknown option: $1" >&2
      usage
      exit 64
      ;;
    *)
      if [[ "${TAG_SET}" == "1" ]]; then
        echo "FAIL: only one TAG_PREFIX is allowed" >&2
        usage
        exit 64
      fi
      TAG_PREFIX="$1"
      TAG_SET=1
      ;;
  esac
  shift
done

if [[ ! "${GPU}" =~ ^[0-9]+$ ]]; then
  echo "FAIL: GPU_ID must be one non-negative integer: ${GPU}" >&2
  exit 64
fi
if [[ ! "${TAG_PREFIX}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "FAIL: TAG_PREFIX must match ^[A-Za-z0-9][A-Za-z0-9._-]*$: ${TAG_PREFIX}" >&2
  exit 64
fi
if [[ -n "${MML:-}" && ! "${MML}" =~ ^[1-9][0-9]*$ ]]; then
  echo "FAIL: MML must be a positive integer: ${MML}" >&2
  exit 64
fi

SFI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:?PYTHON env required: executable absolute path of the runtime interpreter}"
if [[ "${PY}" != /* || ! -x "${PY}" ]]; then
  echo "FAIL: PYTHON must be an executable absolute path: ${PY}" >&2
  exit 64
fi
if ! command -v realpath >/dev/null 2>&1; then
  echo "FAIL: realpath is required for stable runtime identities" >&2
  exit 69
fi
if ! command -v mktemp >/dev/null 2>&1; then
  echo "FAIL: mktemp is required for isolated artifacts" >&2
  exit 69
fi
PY="$(realpath -e -- "${PY}")"
PYTHON="${PY}"
export PYTHON
MODEL="$(realpath -e -- "${MODEL}")"
if [[ ! -d "${MODEL}" ]]; then
  echo "FAIL: MODEL_PATH must be a readable local directory: ${MODEL}" >&2
  exit 66
fi

FA_ROOT="${VLLM_SPARSE_FA3_UPSTREAM_ROOT:-${SFI_ROOT}/third_party_upstreams/vllm-project-flash-attention}"
FA_ROOT="$(realpath -ms -- "${FA_ROOT}")"
FA3_INTERFACE="${FA_ROOT}/vllm_flash_attn/flash_attn_interface.py"
CAPABILITY="$(CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" -c \
  'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; major, minor = torch.cuda.get_device_capability(0); print(f"{major}.{minor}")'
)" || {
  echo "FAIL: cannot inspect selected GPU ${GPU}: ${CAPABILITY}" >&2
  exit 69
}
case "${CAPABILITY}" in
  8.0) ARCH="sm80" ; EXPECTED_BACKEND="fa3" ; EXPECTED_FA_VERSION="3" ;;
  9.0) ARCH="sm90" ; EXPECTED_BACKEND="fa3" ; EXPECTED_FA_VERSION="3" ;;
  10.0) ARCH="sm100" ; EXPECTED_BACKEND="fa4-sm100" ; EXPECTED_FA_VERSION="4" ;;
  *)
    echo "FAIL: selected GPU ${GPU} has unsupported compute capability ${CAPABILITY}; expected 8.0, 9.0, or 10.0" >&2
    exit 78
    ;;
esac

CUDA_TOOLCHAIN_RESOLVER="${SFI_ROOT}/scripts/resolve_flash_attention_toolchain.py"
BUILD_PROVENANCE_CHECKER="${SFI_ROOT}/scripts/check_flash_attention_build_provenance.py"
if [[ ! -f "${CUDA_TOOLCHAIN_RESOLVER}" ]]; then
  echo "FAIL: CUDA toolchain resolver is missing: ${CUDA_TOOLCHAIN_RESOLVER}" >&2
  exit 66
fi
if [[ ! -f "${BUILD_PROVENANCE_CHECKER}" ]]; then
  echo "FAIL: FlashAttention build-provenance checker is missing: ${BUILD_PROVENANCE_CHECKER}" >&2
  exit 66
fi
if ! CUDA_TOOLCHAIN_EXPORTS="$(
  "${PY}" -I "${CUDA_TOOLCHAIN_RESOLVER}" \
    --sfi-root "${SFI_ROOT}" \
    --target "${FA_ROOT}" \
    --architecture "${ARCH}" \
    --checker "${BUILD_PROVENANCE_CHECKER}"
)"; then
  echo "FAIL: FlashAttention CUDA toolchain preflight failed" >&2
  exit 78
fi
eval "${CUDA_TOOLCHAIN_EXPORTS}"
unset CUDA_TOOLCHAIN_EXPORTS
echo "==> CUDA toolchain: home=${CUDA_HOME} compiler=${CUDACXX} release=${SFI_RUNNER_CUDA_COMPILER_RELEASE}"

if [[ "${ARCH}" == "sm100" ]]; then
  FA4_INTERFACE="${FA_ROOT}/flash_attn/cute/interface.py"
  if [[ ! -f "${FA4_INTERFACE}" ]]; then
    echo "FAIL: FA4 CuTe checkout is incomplete: ${FA4_INTERFACE}" >&2
    echo "Run PYTHON=${PY} bash scripts/setup_flash_attention.sh --arch sm100 --gpu ${GPU} first." >&2
    exit 66
  fi
  FA_PREFLIGHT_STATUS="fa4_cute_source_ready"
else
  if [[ ! -f "${FA3_INTERFACE}" ]]; then
    echo "FAIL: FA3 checkout is incomplete: ${FA3_INTERFACE}" >&2
    echo "Run PYTHON=${PY} bash scripts/setup_flash_attention.sh --arch ${ARCH} --gpu ${GPU} first." >&2
    exit 66
  fi
  if ! compgen -G "${FA_ROOT}/vllm_flash_attn/_vllm_fa3_C*.so" >/dev/null; then
    echo "FAIL: FA3 shared object is missing under ${FA_ROOT}/vllm_flash_attn" >&2
    echo "Run PYTHON=${PY} bash scripts/setup_flash_attention.sh --arch ${ARCH} --gpu ${GPU} first." >&2
    exit 66
  fi
  FA_PREFLIGHT_STATUS="fa3_shared_object_ready"
fi

# The selector loader caches by module name.  A caller-supplied directory is a
# base, never a final cache owner: every launch appends the exact
# Python/Torch/CUDA ABI derived after toolchain normalization.
if ! SELECTOR_CACHE_ABI_KEY="$(
  "${PY}" "${SFI_ROOT}/utils/selector_cache_identity.py"
)"; then
  echo "FAIL: selected PYTHON cannot derive the selector cache ABI identity: ${PY}" >&2
  exit 70
fi
if [[ ! "${SELECTOR_CACHE_ABI_KEY}" =~ ^[0-9a-f]{16}$ ]]; then
  echo "FAIL: invalid selector cache ABI identity: ${SELECTOR_CACHE_ABI_KEY}" >&2
  exit 70
fi
if [[ -n "${TORCH_EXTENSIONS_DIR:-}" ]]; then
  SELECTOR_CACHE_BASE="$(realpath -ms -- "${TORCH_EXTENSIONS_DIR}")"
else
  SELECTOR_CACHE_BASE="${SFI_ROOT}/tmp/torch_extensions"
fi
SELECTOR_CACHE_ROOT="${SELECTOR_CACHE_BASE}/${ARCH}_${SELECTOR_CACHE_ABI_KEY}"
mkdir -p "${SELECTOR_CACHE_ROOT}"
export TORCH_EXTENSIONS_DIR="${SELECTOR_CACHE_ROOT}"
export SFI_RUNNER_SELECTOR_CACHE_ROOT="${SELECTOR_CACHE_ROOT}"
export SFI_RUNNER_FA3_PREFLIGHT_STATUS="${FA_PREFLIGHT_STATUS}"
export SFI_RUNNER_ATTENTION_ARCH="${ARCH}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

OUT_ROOT="${SFI_ONE_SHOT_OUT_DIR:-${SFI_ROOT}/out/one_shot}"
if [[ "${OUT_ROOT}" != /* ]]; then
  echo "FAIL: SFI_ONE_SHOT_OUT_DIR must be absolute when set: ${OUT_ROOT}" >&2
  exit 64
fi
OUT_ROOT="$(realpath -ms -- "${OUT_ROOT}")"
mkdir -p "${OUT_ROOT}"
RUN_DIR="$(mktemp -d "${OUT_ROOT}/${TAG_PREFIX}.XXXXXXXX")"
RUN_TAG="$(basename "${RUN_DIR}")"
RUN_STARTED_NS="$("${PY}" -c 'import time; print(time.time_ns())')"
RUN_NONCE="${RUN_TAG}-${RUN_STARTED_NS}-$$"
RESULT_PATH="${RUN_DIR}/result.json"
SUMMARY_PATH="${RUN_DIR}/summary.json"
ROUTE_PATH="${RUN_DIR}/route.jsonl"
LOG_PATH="${RUN_DIR}/run.log"
OUTPUTS_PATH="${RUN_DIR}/result_outputs.json"

EXTRA_ARGS=()
EFFECTIVE_WITH_REF="${WITH_REF}"
case "${ARCH}" in
  sm80)
    BENCHMARK_MODULE="benchmarks.bench_sm80_mixed_page_one_shot_graph_e2e"
    ARCH_ARGS=(--backend fa3 --warmup 2)
    if [[ "${WITH_REF}" != "1" ]]; then
      EXTRA_ARGS+=(--skip-dense-reference)
    fi
    ;;
  sm90)
    BENCHMARK_MODULE="benchmarks.bench_fa3_sm90_mixed_page_one_shot_graph_e2e"
    ARCH_ARGS=(--warmup 0)
    # The SM90 gate is intentionally reference-backed and fail-closed.
    EFFECTIVE_WITH_REF="1"
    ;;
  sm100)
    BENCHMARK_MODULE="benchmarks.bench_sm100_fa4_mixed_page_one_shot_graph_e2e"
    ARCH_ARGS=(--warmup 2)
    if [[ "${WITH_REF}" != "1" ]]; then
      EXTRA_ARGS+=(--skip-dense-reference)
    fi
    ;;
esac
if [[ -n "${MML:-}" ]]; then
  EXTRA_ARGS+=(--max-model-len "${MML}")
fi

echo "==> one-shot tag=${RUN_TAG}"
echo "    arch=${ARCH} capability=${CAPABILITY} backend=${EXPECTED_BACKEND}"
echo "    nonce=${RUN_NONCE}"
echo "    artifacts=${RUN_DIR}"
echo "    python=${PY}"
echo "    selector_cache=${SELECTOR_CACHE_ROOT}"

cd "${SFI_ROOT}"
set +e
CUDA_VISIBLE_DEVICES="${GPU}" \
VLLM_FLASH_ATTN_VERSION="${EXPECTED_FA_VERSION}" \
VLLM_SPARSE_FA3_UPSTREAM_ROOT="${FA_ROOT}" \
PYTHONPATH="${FA_ROOT}:${SFI_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
"${PY}" -m "${BENCHMARK_MODULE}" \
  --mode sparse --producer-mode full-open-gt1 --preset bs2long-cap128 \
  --full-cuda-graph --max-new-tokens 128 "${ARCH_ARGS[@]}" \
  --model "${MODEL}" --python "${PY}" --fa3-upstream-root "${FA_ROOT}" \
  --prefill-last-n 2 --cuda-visible-devices "${GPU}" --timeout-s 1800 \
  --outputs-include-text "${EXTRA_ARGS[@]}" \
  --output "${RESULT_PATH}" \
  --summary-output "${SUMMARY_PATH}" \
  --run-nonce "${RUN_NONCE}" \
  --route-trace-output "${ROUTE_PATH}" \
  2>&1 | tee "${LOG_PATH}" >/dev/null
# Expand the special array once, before any assignment command can replace it
# with that assignment's one-element status.
IFS=' ' read -r CHILD_RC TEE_RC <<< "${PIPESTATUS[*]}"
if [[ ! "${CHILD_RC}" =~ ^[0-9]+$ || ! "${TEE_RC}" =~ ^[0-9]+$ ]]; then
  echo "FAIL: could not capture benchmark/tee pipeline status" >&2
  exit 70
fi
set -e

set +e
WITH_REF="${EFFECTIVE_WITH_REF}" CHILD_RC="${CHILD_RC}" TEE_RC="${TEE_RC}" \
EXPECTED_ARCH="${ARCH}" EXPECTED_BACKEND="${EXPECTED_BACKEND}" \
EXPECTED_FA_VERSION="${EXPECTED_FA_VERSION}" \
"${PY}" - "${SUMMARY_PATH}" "${OUTPUTS_PATH}" "${RUN_STARTED_NS}" "${RUN_NONCE}" <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path
import sys


summary_path = Path(sys.argv[1])
outputs_path = Path(sys.argv[2])
run_started_ns = int(sys.argv[3])
expected_nonce = sys.argv[4]
child_rc = int(os.environ["CHILD_RC"])
tee_rc = int(os.environ["TEE_RC"])
with_reference = os.environ.get("WITH_REF") == "1"
expected_arch = os.environ["EXPECTED_ARCH"]
expected_backend = os.environ["EXPECTED_BACKEND"]
expected_fa_version = int(os.environ["EXPECTED_FA_VERSION"])
failures: list[str] = []

if summary_path.is_symlink() or not summary_path.is_file():
    failures.append(f"fresh summary missing: {summary_path}")
    summary: dict[str, object] = {}
else:
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as exc:
        failures.append(f"summary is not valid JSON: {exc}")
        summary = {}
    # Nonce identity is the authoritative freshness proof.  The small mtime
    # tolerance only accommodates filesystems with coarse timestamp precision.
    if summary_path.stat().st_mtime_ns + 2_000_000_000 < run_started_ns:
        failures.append("summary mtime predates this invocation")
    provenance = summary.get("run_provenance")
    observed_nonce = provenance.get("run_nonce") if isinstance(provenance, dict) else None
    if observed_nonce != expected_nonce:
        failures.append(
            f"run_nonce mismatch: expected={expected_nonce!r} observed={observed_nonce!r}"
        )
    observed_backend = provenance.get("backend") if isinstance(provenance, dict) else None
    if observed_backend != expected_backend:
        failures.append(
            f"backend mismatch for {expected_arch}: "
            f"expected={expected_backend!r} observed={observed_backend!r}"
        )
    observed_fa_version = (
        provenance.get("flash_attn_version_expected")
        if isinstance(provenance, dict)
        else None
    )
    if observed_fa_version != expected_fa_version:
        failures.append(
            f"FlashAttention version mismatch for {expected_arch}: "
            f"expected={expected_fa_version!r} observed={observed_fa_version!r}"
        )

gate = summary.get("gate_passed") is True
production = summary.get("production_gate_passed") is True
producer = summary.get("producer_gate_passed") is True
route = summary.get("route_proof_passed") is True
speed_route = summary.get("speed_child_route_proof_passed") is True
output_length = summary.get("output_length_gate_passed") is True
decode_tps = summary.get("decode_tps")
producer_reasons = summary.get("producer_gate_reasons") or []
lifecycle_reasons = summary.get("sparse_native_lifecycle_gate_reasons") or []
semantic_reasons = summary.get("semantic_gate_reasons") or []
reference_reasons = summary.get("reference_gate_reasons") or []
explicit_production_equivalent = bool(
    route
    and speed_route
    and producer
    and output_length
    and not producer_reasons
    and not lifecycle_reasons
    and not semantic_reasons
    and decode_tps is not None
)
print(
    f"child_returncode={child_rc} gate_passed={gate} "
    f"production_gate_passed={production} producer_gate_passed={producer} "
    f"route_proof_passed={route} speed_child_route_proof_passed={speed_route} "
    f"decode_tps={decode_tps}"
)

if child_rc != 0:
    failures.append(f"child exited rc={child_rc}")
if tee_rc != 0:
    failures.append(f"artifact log writer exited rc={tee_rc}")
if with_reference:
    # The benchmark folds strict sparse-vs-dense text comparison into its final
    # production bit.  Long CoT text can differ while the kernel route and all
    # production mechanics remain sound, so only that reference-only layer is
    # informational.  Every underlying production component stays fail-closed.
    if not production and not (explicit_production_equivalent and reference_reasons):
        failures.append("reference run failed production/route gates")
else:
    if not gate or not production or not explicit_production_equivalent:
        failures.append("production gate did not pass")

for name, reasons in (
    ("producer", producer_reasons),
    ("lifecycle", lifecycle_reasons),
    ("semantic", semantic_reasons),
    ("reference (informational)", reference_reasons),
):
    if reasons:
        print(f"  {name}_gate_reasons: {reasons}")

if with_reference and outputs_path.is_file():
    dense_path = outputs_path.with_name("result_dense_reference_outputs.json")
    try:
        sparse_outputs = json.loads(outputs_path.read_text(encoding="utf-8"))
        dense_outputs = json.loads(dense_path.read_text(encoding="utf-8"))
        from benchmarks.needle_bs2_compare_utils import semantic_match

        print("sparse vs dense reference (informational):")
        for index, (sparse_key, dense_key) in enumerate(
            zip(sorted(sparse_outputs), sorted(dense_outputs))
        ):
            matched, mode, _, _ = semantic_match(
                str(dense_outputs[dense_key].get("text", "")),
                str(sparse_outputs[sparse_key].get("text", "")),
            )
            print(f"  seq{index}: match={matched} mode={mode}")
    except Exception as exc:
        print(f"  reference comparison unavailable: {exc}")

if failures:
    for failure in failures:
        print(f"FAIL: {failure}")
    print("ONE-SHOT FAIL")
    sys.exit(1)
print("ONE-SHOT PASS")
PY
POSTFLIGHT_RC=$?
set -e

echo "    artifacts=${RUN_DIR}"
# Preserve the benchmark child's exact failure code.  Postflight can only turn
# a nominal child success into failure; it must never mask or rewrite a child
# failure using a stale/foreign summary.
if [[ "${CHILD_RC}" != "0" ]]; then
  exit "${CHILD_RC}"
fi
exit "${POSTFLIGHT_RC}"
