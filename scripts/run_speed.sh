#!/usr/bin/env bash
# =============================================================================
# SFI throughput benchmark (offline engine, full CUDA graph).
#
# Usage:
#   bash scripts/run_speed.sh <GPU_ID> <MODEL_PATH> <TIER> [sparse|dense] [TAG]
#
# TIER is one of the pre-tuned batch x context tiers (validated on A100 40GB):
#
#   tier      batch  ctx/req   KV pool     max-model-len
#   bs8x12k     8     12k      18 GiB      16384        <- start here
#   bs8x16k     8     16k      22 GiB      20480
#   bs4x24k     4     24k      16 GiB      28672
#   bs2x30k     2     30k      16 GiB      36864
#
# On GPUs with a different memory size, scale the KV pool: it must fit
#   model weights + KV pool + ~2-3 GB SFI runtime state  <  total VRAM.
# In sparse mode the KV pool itself must hold the FULL KV of every request
# PLUS the compact-page lease (slots x blocks/slot x 16 x KV-bytes/token) —
# the preflight below checks this and warns before launching.
# Override any knob via env: BS CTX KVB MML MAX_NEW REFRESH_INTERVAL BLOCKS.
#
# Multi-GPU (tensor parallel): TP=2 and pass a GPU list, e.g.
#   TP=2 BS=2 CTX=128000 KVB=20401094656 MML=132096 \
#     bash scripts/run_speed.sh "4,5" <MODEL> bs2x30k sparse tag
# KVB is per GPU. TP>1 keeps vLLM's async scheduling (the SFI runtime repairs
# the async placeholder tokens in place); the runner auto-probes NVLink and
# only disables vLLM's custom all-reduce on PCIe-only topologies.
#
# Run "dense" once at the same tier to get your local speedup baseline.
#
# IMPORTANT: throughput numbers are only meaningful on an EXCLUSIVE, idle GPU.
#
# Success criteria (printed at the end):
#   decode_tps reported; all requests decode to MAX_NEW tokens; zero dense
#   fallbacks. Exit code 2 from the harness with reasons
#   'unknown_without_reference' / 'interval_trigger_intents_below_expected'
#   is EXPECTED in this free-corpus + no-reference mode and is NOT an error.
# =============================================================================
set -euo pipefail

GPU="${1:?usage: run_speed.sh <GPU_ID> <MODEL_PATH> <TIER> [sparse|dense] [TAG]}"
MODEL="${2:?usage: run_speed.sh <GPU_ID> <MODEL_PATH> <TIER> [sparse|dense] [TAG]}"
TIER="${3:?tier: bs8x12k | bs8x16k | bs4x24k | bs2x30k}"
MODE="${4:-sparse}"
TAG="${5:-speed_${TIER}_${MODE}}"

case "${TIER}" in
  bs8x12k) BS="${BS:-8}"; CTX="${CTX:-12000}"; KVB="${KVB:-19327352832}"; MML="${MML:-16384}" ;;
  bs8x16k) BS="${BS:-8}"; CTX="${CTX:-16000}"; KVB="${KVB:-23622320128}"; MML="${MML:-20480}" ;;
  bs4x24k) BS="${BS:-4}"; CTX="${CTX:-24000}"; KVB="${KVB:-17179869184}"; MML="${MML:-28672}" ;;
  bs2x30k) BS="${BS:-2}"; CTX="${CTX:-30000}"; KVB="${KVB:-17179869184}"; MML="${MML:-36864}" ;;
  *) echo "unknown tier: ${TIER}" >&2; exit 2 ;;
esac
MAX_NEW="${MAX_NEW:-256}"
REFRESH_INTERVAL="${REFRESH_INTERVAL:-96}"
BLOCKS="${BLOCKS:-112}"
TP="${TP:-1}"
TP_ARGS=()
if [[ "${TP}" -gt 1 ]]; then
  TP_ARGS=(--tensor-parallel-size "${TP}")
fi

# --- Sparse KV-budget preflight (warns, never blocks) ------------------------
# Sparse mode leases compact pages from vLLM's block pool ON TOP of the full
# KV of every request. If the pool cannot hold both, nothing crashes — the
# vLLM scheduler silently serializes the batch (half of it waits, prefill is
# recomputed mid-run, decode takes ~2x the steps, decode_tps roughly halves).
# Capacity check:
#   cap_tokens_per_req = (KVB - slots*blocks*16*kv_B/token) / (kv_B/token * BS)
#   must exceed CTX + MAX_NEW.
# KV_TOKEN_BYTES defaults to Qwen3-4B full-model KV (147456 B/token), divided
# by TP because KV heads shard across ranks and KVB is per GPU. Override it
# for other models: layers x kv_heads x head_dim x 2 (K,V) x dtype_bytes.
KV_TOKEN_BYTES="${KV_TOKEN_BYTES:-$((147456 / TP))}"
if [[ "${MODE}" == "sparse" ]]; then
  # Dual-generation compact read (VLLM_SPARSE_COMPACT_DUAL_GEN=1) leases a
  # spare half-arena per slot: the runtime reserves slots*blocks*GEN blocks,
  # so the preflight must count the same factor or it under-estimates the
  # lease by 2x and green-lights shapes that silently serialize.
  # Mirror the RUNTIME default (dual-gen PROMOTED default ON 2026-07-08:
  # the TP>1 32k wedge is fixed at the root — see handoff §10; the 32k TP1
  # x2-lease capacity wall is a physical constraint, run =0 there or raise
  # KVB; all four standard tiers hold dual-gen at TP1).
  GEN_COUNT=2
  if [[ "${VLLM_SPARSE_COMPACT_DUAL_GEN:-1}" == "0" ]]; then
    GEN_COUNT=1
  fi
  LEASE_BYTES=$((BS * BLOCKS * 16 * KV_TOKEN_BYTES * GEN_COUNT))
  CAP_TOKENS_PER_REQ=$(((KVB - LEASE_BYTES) / (KV_TOKEN_BYTES * BS)))
  NEED_TOKENS_PER_REQ=$((CTX + MAX_NEW))
  if ((CAP_TOKENS_PER_REQ < NEED_TOKENS_PER_REQ)); then
    KVB_SUGGEST_GIB=$(((NEED_TOKENS_PER_REQ * KV_TOKEN_BYTES * BS + LEASE_BYTES) / 1073741824 + 2))
    cat >&2 <<EOW
==============================================================================
WARNING: sparse KV pool too small for this shape — expect the vLLM scheduler
to serialize the batch (decode steps ~2x, decode_tps ~1/2, mid-run prefill
recompute). The pool must hold full KV + compact-page lease:
  compact lease      = ${BS} slots x ${BLOCKS} blocks x 16 x ${KV_TOKEN_BYTES} B x ${GEN_COUNT} gen = ${LEASE_BYTES} B
  cap_tokens_per_req = (KVB - lease) / (${KV_TOKEN_BYTES} x ${BS}) = ${CAP_TOKENS_PER_REQ}
  needed per request = CTX + MAX_NEW = ${NEED_TOKENS_PER_REQ}
Fix: KVB >= ~${KVB_SUGGEST_GIB} GiB (KVB=$((KVB_SUGGEST_GIB * 1073741824))),
or lower CTX/MAX_NEW/BS/BLOCKS. Continuing anyway.
==============================================================================
EOW
  fi
fi

SFI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FA_ROOT="${VLLM_SPARSE_FA3_UPSTREAM_ROOT:-${SFI_ROOT}/third_party_upstreams/vllm-project-flash-attention}"
# PYTHON is REQUIRED (fail-fast, no bare-`python` fallback): the shared
# tmp/torch_extensions/sm80_gt1 cache keys prebuilt .so by NAME only. A wrong
# interpreter (e.g. conda base py3.12) silently rebuilds the whole ext family
# with an incompatible ABI, and the next correct run then imports those
# poisoned artifacts ("Python version mismatch" / symbol errors). Root fix is
# refusing to launch rather than producing the bad cache.
PY="${PYTHON:?PYTHON env required: absolute path of the benchmark interpreter (e.g. /ssd/.../envs/vllm019-cu126/bin/python); bare 'python' poisons the shared ext cache with a wrong-ABI build}"
OUT="${SFI_ROOT}/out"
mkdir -p "${OUT}"

CORPUS="${CORPUS:-${OUT}/ctx_${BS}x${CTX}.txt}"
if [[ ! -f "${CORPUS}" ]]; then
  echo "==> corpus ${CORPUS} missing; generating (${BS} segments x ${CTX} tokens)"
  "${PY}" "${SFI_ROOT}/scripts/make_context_corpus.py" \
    --model "${MODEL}" --segments "${BS}" --tokens-per-segment "${CTX}" \
    --output "${CORPUS}"
fi

# expandable_segments is a shared-GPU nicety (fragmentation/mem-race), but its
# cuMemMap-backed memory has NO cudaIpcGetMemHandle support. vLLM custom
# all-reduce registers every AR buffer inside the CUDA graph via IPC handles at
# capture end (custom_all_reduce.py register_graph_buffers) -> with expandable
# ON, TP>1 + custom AR + FULL cudagraph crashes with "invalid argument"
# (remote exp4 root cause, locally reproduced 2026-07-06). TP>1 boxes are
# dedicated anyway: default expandable OFF there so custom AR can stay ON.
if [ "${TP}" -gt 1 ]; then
  export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-}"
else
  export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
fi
export VLLM_KV_CACHE_MEMORY_BYTES="${KVB}"   # pin the KV pool: SFI runtime state
                                             # lives OUTSIDE vLLM's util budget

echo "==> ${MODE} ${TIER}: bs=${BS} ctx=${CTX} kv_pool=$((KVB/1024/1024/1024))GiB mml=${MML} max_new=${MAX_NEW}"
cd "${SFI_ROOT}"
PYTHONPATH="${SFI_ROOT}" "${PY}" -m benchmarks.bench_sm80_mixed_page_one_shot_graph_e2e \
  --mode "${MODE}" --producer-mode full-open-gt1 \
  --backend fa3 --full-cuda-graph --warmup 1 \
  --prompt "${CORPUS}" --split-context-prompts \
  --batch-size "${BS}" --max-new-tokens "${MAX_NEW}" --max-model-len "${MML}" \
  --refresh-interval "${REFRESH_INTERVAL}" --prefill-last-n 2 \
  --max-live-sparse-slots "${BS}" --compact-blocks-per-slot "${BLOCKS}" \
  --gpu-mem-util "${UTIL:-0.9}" \
  --model "${MODEL}" --python "${PY}" --fa3-upstream-root "${FA_ROOT}" \
  --cuda-visible-devices "${GPU}" --timeout-s 3600 ${TP_ARGS[@]+"${TP_ARGS[@]}"} \
  --skip-dense-reference --outputs-include-text \
  --output "${OUT}/${TAG}.json" \
  --summary-output "${OUT}/${TAG}_summary.json" \
  --route-trace-output "${OUT}/${TAG}_route.jsonl" \
  > "${OUT}/${TAG}.log" 2>&1 || true

"${PY}" - "${OUT}/${TAG}_summary.json" "${BS}" "${MAX_NEW}" <<'EOF'
import json, sys
path, bs, max_new = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
try:
    s = json.load(open(path))
except Exception as e:
    print(f"FAIL: no summary produced ({e}); check the matching .log")
    sys.exit(1)
tps = s.get("decode_tps")
print(f"decode_tps={tps}")
ad_tps = s.get("all_decode_tps")
if ad_tps is not None:
    # steady window: prefill-interleaved head excluded (fair decode comparison
    # for long-ctx / large-bs tiers where that head dominates wall time)
    print(f"all_decode_tps={ad_tps}")
for key in sorted(s):
    if "refresh" in key and "count" in key and isinstance(s[key], (int, float)) and s[key]:
        print(f"  {key}={s[key]}")
fallbacks = s.get("dense_native_fallback_count", 0) or 0
# [SPARSE-ROUTE-PROOF-GATE 2026-07-10] sparse child crashing at boot (e.g.
# sitecustomize patch failure) leaves fallback_count=0 and only noise-class
# gate reasons while dense keeps decoding -> the old checks alone printed
# "SPEED RUN OK" on a run where sparse never engaged (remote TP8 silent-
# fallback case). Route proof is populated on every healthy sparse run
# (empty reasons list), so any entry here means sparse never ran: fail.
route_proof_reasons = list((s.get("route_proof") or {}).get("reasons") or [])
route_proof_reasons += list(s.get("speed_child_route_proof_reasons") or [])
expected_noise = {
    "unknown_without_reference",           # no dense reference in this mode
    "missing_text",                        # outputs captured without text
    "interval_trigger_intents_below_expected",  # free corpus vs preset trigger plan
}
reasons = [r for r in (s.get("semantic_gate_reasons") or []) + (s.get("producer_gate_reasons") or [])
           if not any(n in str(r) for n in expected_noise)]
ok = tps is not None and not fallbacks and not reasons and not route_proof_reasons
if fallbacks:
    print(f"  dense_native_fallback_count={fallbacks} (must be 0)")
if reasons:
    print(f"  unexpected gate reasons: {reasons}")
if route_proof_reasons:
    print(f"  sparse route proof missing (sparse never engaged): {route_proof_reasons[:6]}")
print("SPEED RUN OK" if ok else "SPEED RUN CHECK FAILED")
sys.exit(0 if ok else 1)
EOF
