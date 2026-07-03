#!/usr/bin/env bash
# =============================================================================
# SFI one-shot correctness test (offline engine, full CUDA graph, gated).
#
# Runs the sparse production pipeline end-to-end on a built-in 2-sequence
# long-context preset and checks the production gate:
#   child exit ∙ kernel-route proof (the sparse kernel really ran) ∙ producer
#   contract ∙ lifecycle invariants ∙ output health.
#
# Usage:
#   bash scripts/run_one_shot.sh <GPU_ID> <MODEL_PATH> [TAG] [--with-reference]
#
#   GPU_ID      CUDA device index (needs >= ~22 GB free for the default preset)
#   MODEL_PATH  any local HF model dir (a small Qwen3 is fine for the default)
#   TAG         output name tag (default: oneshot)
#
#   --with-reference   additionally run a DENSE reference pass and print an
#       answer-level sparse-vs-dense comparison. NOTE: the comparator is
#       strict (boxed-answer, else normalized full text) — long chain-of-
#       thought outputs rarely match verbatim even when the final answers
#       agree, so this mode reports the comparison for HUMAN/agent judgment
#       and does not fail the run on a semantic diff alone.
#
# Env knobs:
#   MML   set --max-model-len; REQUIRED when the model's native context exceeds
#         what your card's KV budget can hold (e.g. MML=16384 for 4B on 40 GB)
#
# Success criterion (checked automatically, exit code 0 = PASS):
#   default:          summary.gate_passed == true and producer_gate_passed == true
#   --with-reference: producer/lifecycle gates clean; semantic diff printed as info
#
# Outputs (under ./out/):
#   <TAG>_summary.json   gate verdict + throughput + producer metrics
#   <TAG>_outputs.json   generated token ids/text per sequence
#   <TAG>_route.jsonl    per-step attention route trace
# =============================================================================
set -euo pipefail

GPU="${1:?usage: run_one_shot.sh <GPU_ID> <MODEL_PATH> [TAG] [--with-reference]}"
MODEL="${2:?usage: run_one_shot.sh <GPU_ID> <MODEL_PATH> [TAG] [--with-reference]}"
TAG="${3:-oneshot}"
MODE="${4:-}"

SFI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FA_ROOT="${VLLM_SPARSE_FA3_UPSTREAM_ROOT:-${SFI_ROOT}/third_party_upstreams/vllm-project-flash-attention}"
PY="${PYTHON:-python}"
OUT="${SFI_ROOT}/out"
mkdir -p "${OUT}"

export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

EXTRA=()
WITH_REF=0
if [[ "${MODE}" == "--with-reference" ]]; then
  WITH_REF=1
else
  EXTRA+=(--skip-dense-reference)
fi
if [[ -n "${MML:-}" ]]; then
  EXTRA+=(--max-model-len "${MML}")
fi

cd "${SFI_ROOT}"
PYTHONPATH="${SFI_ROOT}" "${PY}" -m benchmarks.bench_sm80_mixed_page_one_shot_graph_e2e \
  --mode sparse --producer-mode full-open-gt1 --preset bs2long-cap128 \
  --backend fa3 --full-cuda-graph --warmup 2 --max-new-tokens 128 \
  --model "${MODEL}" --python "${PY}" --fa3-upstream-root "${FA_ROOT}" \
  --prefill-last-n 2 --cuda-visible-devices "${GPU}" --timeout-s 1800 \
  --outputs-include-text ${EXTRA[@]+"${EXTRA[@]}"} \
  --output "${OUT}/${TAG}.json" \
  --summary-output "${OUT}/${TAG}_summary.json" \
  --route-trace-output "${OUT}/${TAG}_route.jsonl" \
  | tee "${OUT}/${TAG}.log" >/dev/null || true

WITH_REF="${WITH_REF}" PYTHONPATH="${SFI_ROOT}" "${PY}" - "${OUT}/${TAG}_summary.json" "${OUT}/${TAG}_outputs.json" <<'EOF'
import json, os, sys
summary_path, outputs_path = sys.argv[1], sys.argv[2]
with_ref = os.environ.get("WITH_REF") == "1"
try:
    s = json.load(open(summary_path))
except Exception as e:
    print(f"FAIL: no summary produced ({e}); see the matching .log for the failure signature")
    sys.exit(1)

gate = bool(s.get("gate_passed"))
prod = bool(s.get("producer_gate_passed"))
tps = s.get("decode_tps")
prod_reasons = s.get("producer_gate_reasons") or []
life_reasons = s.get("sparse_native_lifecycle_gate_reasons") or []
sem_reasons = s.get("semantic_gate_reasons") or []
print(f"gate_passed={gate} producer_gate_passed={prod} decode_tps={tps}")

if not with_ref:
    ok = gate and prod
    for name, reasons in (("producer", prod_reasons), ("lifecycle", life_reasons), ("semantic", sem_reasons)):
        if reasons:
            print(f"  {name}_gate_reasons: {reasons}")
    print("ONE-SHOT PASS" if ok else "ONE-SHOT FAIL")
    sys.exit(0 if ok else 1)

# --with-reference: semantic diff is informational; fail only on pipeline gates
ref_path = outputs_path.replace("_outputs.json", "_dense_reference_outputs.json")
try:
    sp, dn = json.load(open(outputs_path)), json.load(open(ref_path))
    from benchmarks.needle_bs2_compare_utils import semantic_match
    print("sparse vs dense reference (strict comparator, informational):")
    for i, (a, b) in enumerate(zip(sorted(sp), sorted(dn))):
        ok_i, mode, ref, test = semantic_match(str(dn[b].get("text", "")), str(sp[a].get("text", "")))
        print(f"  seq{i}: match={ok_i} mode={mode}")
        print(f"    dense : {str(ref)[:160]!r}")
        print(f"    sparse: {str(test)[:160]!r}")
except Exception as e:
    print(f"  (reference comparison unavailable: {e})")
ok = prod and not prod_reasons and not life_reasons and tps is not None
print("ONE-SHOT PASS (reference diff above is informational)" if ok else "ONE-SHOT FAIL")
sys.exit(0 if ok else 1)
EOF
