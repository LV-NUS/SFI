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
# the preflight below checks this and fails before launching by default.
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
if [[ ! "${TAG}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "FAIL: TAG must match ^[A-Za-z0-9][A-Za-z0-9._-]*$: ${TAG}" >&2
  exit 64
fi
if [[ ! "${GPU}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "FAIL: GPU_ID must be a comma-separated list of non-negative integers: ${GPU}" >&2
  exit 64
fi
IFS=',' read -r -a GPU_IDS <<< "${GPU}"
declare -A GPU_ID_SEEN=()
GPU_IDS_CANONICAL=()
for raw_gpu_id in "${GPU_IDS[@]}"; do
  gpu_id="${raw_gpu_id}"
  while [[ "${#gpu_id}" -gt 1 && "${gpu_id:0:1}" == "0" ]]; do
    gpu_id="${gpu_id:1}"
  done
  if [[ -n "${GPU_ID_SEEN[${gpu_id}]:-}" ]]; then
    echo "FAIL: GPU_ID contains a duplicate device: ${gpu_id}" >&2
    exit 64
  fi
  GPU_ID_SEEN["${gpu_id}"]=1
  GPU_IDS_CANONICAL+=("${gpu_id}")
done
GPU_IDS=("${GPU_IDS_CANONICAL[@]}")
printf -v GPU '%s,' "${GPU_IDS[@]}"
GPU="${GPU%,}"

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
# VERDICT_ONLY=1 skips the diagnostic child (verdict-grade screening: proofs
# re-source from the speed-child trace; counts readout file becomes
# ${TAG}_speed_route.jsonl). Default (unset/0) is the unchanged full form.
VERDICT_ARGS=()
if [[ "${VERDICT_ONLY:-0}" == "1" ]]; then
  VERDICT_ARGS=(--verdict-only)
fi

# --- Sparse KV-budget preflight (fail-closed; explicit override available) ---
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
SFI_RUNNER_KV_PREFLIGHT_STATUS="not_applicable"
if [[ "${MODE}" == "sparse" ]]; then
  if [[ "${SFI_ALLOW_UNDERSIZED_KV:-0}" != "0" && "${SFI_ALLOW_UNDERSIZED_KV:-0}" != "1" ]]; then
    echo "FAIL: SFI_ALLOW_UNDERSIZED_KV must be 0 or 1" >&2
    exit 64
  fi
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
  SFI_RUNNER_KV_PREFLIGHT_STATUS="passed"
  if ((CAP_TOKENS_PER_REQ < NEED_TOKENS_PER_REQ)); then
    KVB_SUGGEST_GIB=$(((NEED_TOKENS_PER_REQ * KV_TOKEN_BYTES * BS + LEASE_BYTES) / 1073741824 + 2))
    if [[ "${SFI_ALLOW_UNDERSIZED_KV:-0}" == "1" ]]; then
      SFI_RUNNER_KV_PREFLIGHT_STATUS="undersized_override"
      KV_PREFLIGHT_VERDICT="WARNING: sparse KV pool is undersized; explicit SFI_ALLOW_UNDERSIZED_KV=1 override accepted."
      KV_PREFLIGHT_ACTION="Continuing due to explicit override."
    else
      SFI_RUNNER_KV_PREFLIGHT_STATUS="failed"
      KV_PREFLIGHT_VERDICT="FAIL: sparse KV pool too small for this shape."
      KV_PREFLIGHT_ACTION="Refusing to launch. Set SFI_ALLOW_UNDERSIZED_KV=1 only for intentional diagnostic serialization."
    fi
    cat >&2 <<EOW
==============================================================================
${KV_PREFLIGHT_VERDICT}
The vLLM scheduler would serialize the batch (decode steps ~2x, decode_tps
~1/2, mid-run prefill recompute). The pool must hold full KV + compact lease:
  compact lease      = ${BS} slots x ${BLOCKS} blocks x 16 x ${KV_TOKEN_BYTES} B x ${GEN_COUNT} gen = ${LEASE_BYTES} B
  cap_tokens_per_req = (KVB - lease) / (${KV_TOKEN_BYTES} x ${BS}) = ${CAP_TOKENS_PER_REQ}
  needed per request = CTX + MAX_NEW = ${NEED_TOKENS_PER_REQ}
Fix: KVB >= ~${KVB_SUGGEST_GIB} GiB (KVB=$((KVB_SUGGEST_GIB * 1073741824))),
or lower CTX/MAX_NEW/BS/BLOCKS. ${KV_PREFLIGHT_ACTION}
==============================================================================
EOW
    if [[ "${SFI_ALLOW_UNDERSIZED_KV:-0}" != "1" ]]; then
      exit 78
    fi
  fi
fi
export SFI_RUNNER_KV_PREFLIGHT_STATUS

SFI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FA_ROOT="${VLLM_SPARSE_FA3_UPSTREAM_ROOT:-${SFI_ROOT}/third_party_upstreams/vllm-project-flash-attention}"
# PYTHON is REQUIRED (fail-fast, no bare-`python` fallback): cache partitioning
# and every child must use the exact interpreter that will load the extension.
# A PATH-dependent alias can drift between shells and invalidate that identity.
PY="${PYTHON:?PYTHON env required: absolute path of the benchmark interpreter (e.g. /ssd/.../envs/vllm019-cu126/bin/python); bare 'python' poisons the shared ext cache with a wrong-ABI build}"
if [[ "${PY}" != /* || ! -x "${PY}" ]]; then
  echo "FAIL: PYTHON must be an executable absolute path: ${PY}" >&2
  exit 64
fi

if ! command -v realpath >/dev/null 2>&1; then
  echo "FAIL: realpath is required for stable benchmark path identities" >&2
  exit 69
fi
# Canonicalize lexically while still in the caller's working directory. The
# process later cd's to SFI_ROOT, so leaving relative MODEL/CORPUS values here
# would make generation and execution refer to different files. -s preserves
# the final component identity so explicit corpus symlinks can be rejected.
MODEL="$(realpath -ms -- "${MODEL}")"
FA_ROOT="$(realpath -ms -- "${FA_ROOT}")"
if [[ -n "${CORPUS:-}" ]]; then
  CORPUS="$(realpath -ms -- "${CORPUS}")"
fi
OUT="${SFI_ROOT}/out"
mkdir -p "${OUT}"

reject_symlink_output() {
  local path="$1"
  if [[ -L "${path}" ]]; then
    echo "FAIL: refusing symlink output path: ${path}" >&2
    exit 73
  fi
}

# TAG is path-safe above. Refuse pre-planted symlink artifacts as an additional
# local safety boundary before any shell/Python writer opens predictable names.
while IFS= read -r existing_symlink; do
  reject_symlink_output "${existing_symlink}"
done < <(find "${OUT}" -maxdepth 1 -type l -name "${TAG}*" -print)

# A TAG is the artifact identity for this runner. Two concurrent invocations
# with the same TAG would otherwise overwrite the same summary/log, allowing a
# failed child to consume the other run's fresh green summary. Hold a non-
# blocking per-TAG lock for the entire invocation; callers must choose unique
# tags when they intentionally run in parallel.
if ! command -v flock >/dev/null 2>&1; then
  echo "FAIL: flock is required for per-TAG artifact isolation" >&2
  exit 69
fi
RUN_LOCK_PATH="${OUT}/${TAG}.lock"
reject_symlink_output "${RUN_LOCK_PATH}"
exec {RUN_LOCK_FD}>"${RUN_LOCK_PATH}"
if ! flock -n "${RUN_LOCK_FD}"; then
  echo "FAIL: TAG '${TAG}' is already running (lock: ${RUN_LOCK_PATH})" >&2
  exit 75
fi

# A TAG lock protects artifacts, not devices. Different tags on an overlapping
# GPU set still corrupt throughput measurements and can OOM one another. Hold
# one same-user, non-blocking lock per physical GPU token for the entire run.
if [[ "${SFI_ALLOW_SHARED_GPU:-0}" != "0" && "${SFI_ALLOW_SHARED_GPU:-0}" != "1" ]]; then
  echo "FAIL: SFI_ALLOW_SHARED_GPU must be 0 or 1" >&2
  exit 64
fi
SFI_RUNNER_GPU_LOCK_MODE="shared_override"
if [[ "${SFI_ALLOW_SHARED_GPU:-0}" != "1" ]]; then
  SFI_RUNNER_GPU_LOCK_MODE="exclusive"
  GPU_LOCK_ROOT="${SFI_GPU_LOCK_ROOT:-${TMPDIR:-/tmp}/sfi-run-speed-gpu-locks-${UID}}"
  mkdir -p "${GPU_LOCK_ROOT}"
  for gpu_id in "${GPU_IDS[@]}"; do
    gpu_lock_path="${GPU_LOCK_ROOT}/gpu_${gpu_id}.lock"
    reject_symlink_output "${gpu_lock_path}"
    exec {gpu_lock_fd}>"${gpu_lock_path}"
    if ! flock -n "${gpu_lock_fd}"; then
      echo "FAIL: GPU '${gpu_id}' is already reserved by another run (lock: ${gpu_lock_path})" >&2
      exit 76
    fi
  done
fi
export SFI_RUNNER_GPU_LOCK_MODE

# The extension loader keys builds by module name, so a fixed shared root can
# import or overwrite an incompatible .so. Partition the default cache by the
# selected interpreter's SOABI plus Torch/CUDA ABI. An explicit cache root is
# still an operator-owned override; it retains precedence and is absolutized.
if [[ -n "${TORCH_EXTENSIONS_DIR:-}" ]]; then
  SELECTOR_CACHE_ROOT="$(realpath -ms -- "${TORCH_EXTENSIONS_DIR}")"
else
  if ! SELECTOR_CACHE_ABI_KEY="$(
    "${PY}" "${SFI_ROOT}/utils/selector_cache_identity.py"
  )"; then
    echo "FAIL: selected PYTHON cannot derive the selector cache ABI identity: ${PY}" >&2
    exit 70
  fi
  if [[ ! "${SELECTOR_CACHE_ABI_KEY}" =~ ^[0-9a-f]{16}$ ]]; then
    echo "FAIL: invalid selector cache ABI identity from ${PY}: ${SELECTOR_CACHE_ABI_KEY}" >&2
    exit 70
  fi
  SELECTOR_CACHE_ROOT="${SFI_ROOT}/tmp/torch_extensions/sm80_gt1_${SELECTOR_CACHE_ABI_KEY}"
fi
export TORCH_EXTENSIONS_DIR="${SELECTOR_CACHE_ROOT}"
export SFI_RUNNER_SELECTOR_CACHE_ROOT="${SELECTOR_CACHE_ROOT}"

# Fail before corpus generation or selector prewarm when the vendored FA3
# checkout/build is absent. Without this boundary the sparse harness can spend
# minutes compiling unrelated selector extensions before sitecustomize dies on
# a missing flash_attn_interface.py.
FA3_INTERFACE="${FA_ROOT}/vllm_flash_attn/flash_attn_interface.py"
if [[ ! -f "${FA3_INTERFACE}" ]]; then
  echo "FAIL: vendored FA3 checkout is incomplete: ${FA3_INTERFACE}" >&2
  echo "Run bash scripts/setup_flash_attention.sh, or set VLLM_SPARSE_FA3_UPSTREAM_ROOT to a prepared checkout." >&2
  exit 66
fi
if ! compgen -G "${FA_ROOT}/vllm_flash_attn/_vllm_fa3_C*.so" >/dev/null; then
  echo "FAIL: vendored FA3 shared object is missing under ${FA_ROOT}/vllm_flash_attn" >&2
  echo "Run bash scripts/setup_flash_attention.sh before benchmarking." >&2
  exit 66
fi
SFI_RUNNER_FA3_PREFLIGHT_STATUS="passed"
export SFI_RUNNER_FA3_PREFLIGHT_STATUS

if [[ -n "${CORPUS:-}" ]]; then
  reject_symlink_output "${CORPUS}"
  if [[ ! -f "${CORPUS}" ]]; then
    echo "==> explicit corpus ${CORPUS} missing; generating"
    "${PY}" "${SFI_ROOT}/scripts/make_context_corpus.py" \
      --model "${MODEL}" --segments "${BS}" --tokens-per-segment "${CTX}" \
      --output "${CORPUS}"
    SFI_RUNNER_CORPUS_TOKEN_STATUS="generated_exact"
  else
    "${PY}" "${SFI_ROOT}/scripts/make_context_corpus.py" \
      --model "${MODEL}" --segments "${BS}" --tokens-per-segment "${CTX}" \
      --validate "${CORPUS}"
    SFI_RUNNER_CORPUS_TOKEN_STATUS="validated_exact"
  fi
else
  CORPUS_RESULT="$("${PY}" "${SFI_ROOT}/scripts/make_context_corpus.py" \
    --model "${MODEL}" --segments "${BS}" --tokens-per-segment "${CTX}" \
    --cache-dir "${OUT}/context_corpus_cache")"
  CORPUS="$(printf '%s\n' "${CORPUS_RESULT}" | sed -n 's/^context_corpus_path=//p' | tail -1)"
  if [[ -z "${CORPUS}" || ! -f "${CORPUS}" ]]; then
    echo "FAIL: context corpus cache returned no readable path" >&2
    exit 66
  fi
  SFI_RUNNER_CORPUS_TOKEN_STATUS="cache_v2_exact"
fi
export SFI_RUNNER_CORPUS_TOKEN_STATUS

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
echo "==> infra: kv_preflight=${SFI_RUNNER_KV_PREFLIGHT_STATUS} gpu_lock=${SFI_RUNNER_GPU_LOCK_MODE} fa3_preflight=${SFI_RUNNER_FA3_PREFLIGHT_STATUS} corpus=${SFI_RUNNER_CORPUS_TOKEN_STATUS} selector_cache=${SELECTOR_CACHE_ROOT}"
cd "${SFI_ROOT}"
RUN_STARTED_NS="$("${PY}" -c 'import time; print(time.time_ns())')"
RUN_NONCE="${TAG}-${RUN_STARTED_NS}-$$"
for output_path in \
  "${OUT}/${TAG}.log" \
  "${OUT}/${TAG}.json" \
  "${OUT}/${TAG}_summary.json" \
  "${OUT}/${TAG}_route.jsonl"; do
  reject_symlink_output "${output_path}"
done
# Preserve the harness return code for the postflight. rc=2 is accepted only
# when this run's fresh summary proves it contains documented no-reference
# noise and no real gate failure; every other non-zero code is propagated.
set +e
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
  ${VERDICT_ARGS[@]+"${VERDICT_ARGS[@]}"} \
  --skip-dense-reference --outputs-include-text \
  --output "${OUT}/${TAG}.json" \
  --summary-output "${OUT}/${TAG}_summary.json" \
  --run-nonce "${RUN_NONCE}" \
  --route-trace-output "${OUT}/${TAG}_route.jsonl" \
  > "${OUT}/${TAG}.log" 2>&1
HARNESS_RC=$?
set -e

"${PY}" "${SFI_ROOT}/scripts/check_run_speed_summary.py" \
  "${OUT}/${TAG}_summary.json" \
  --mode "${MODE}" \
  --harness-returncode "${HARNESS_RC}" \
  --run-started-ns "${RUN_STARTED_NS}" \
  --run-nonce "${RUN_NONCE}" \
  --selector-cache-root "${SELECTOR_CACHE_ROOT}"
