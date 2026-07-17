#!/usr/bin/env bash
# =============================================================================
# SFI throughput benchmark (offline engine, full CUDA graph).
#
# Usage:
#   bash scripts/run_speed.sh <GPU_ID> <MODEL_PATH> <TIER> [sparse|dense] [TAG]
#
# TIER is one of the batch x context tiers tuned and validated on A100 40GB,
# or the self-contained exact TP8 remote workload.
# The same shapes may be run on SM90/SM100 for bring-up, but they are not
# claimed to be tuned for those GPUs; override BS/CTX/KVB/MML as appropriate.
#
#   tier      batch  ctx/req   KV pool     max-model-len
#   bs8x12k     8     12k      18 GiB      16384        <- start here
#   bs8x16k     8     16k      22 GiB      20480
#   bs4x24k     4     24k      16 GiB      28672
#   bs2x30k     2     30k      16 GiB      36864
#   tp8x64k     32    64k      40 GiB      66560         <- exact TP8 gate
#
# On GPUs with a different memory size, scale the KV pool: it must fit
#   model weights + KV pool + ~2-3 GB SFI runtime state  <  total VRAM.
# In sparse mode the KV pool itself must hold the FULL KV of every request
# PLUS the compact-page lease (slots x blocks/slot x 16 x KV-bytes/token) —
# the preflight below checks this and fails before launching by default.
# Override any knob via env: BS CTX KVB MML MAX_NEW REFRESH_INTERVAL BLOCKS
# K_HEAD.
#
# Multi-GPU (tensor parallel): TP=2 and pass a GPU list, e.g.
#   TP=2 BS=2 CTX=128000 KVB=20401094656 MML=132096 \
#     bash scripts/run_speed.sh "4,5" <MODEL> bs2x30k sparse tag
# KVB is per GPU. TP>1 keeps vLLM's async scheduling (the SFI runtime repairs
# the async placeholder tokens in place); the runner auto-probes NVLink and
# only disables vLLM's custom all-reduce on PCIe-only topologies.
#
# Exact sparse runs carry an adjacent observer-free dense arm and gate the
# paired engine-loop speedup. A standalone dense run is health evidence only.
#
# IMPORTANT: throughput numbers are only meaningful on an EXCLUSIVE, idle GPU.
#
# Success criteria (printed at the end):
#   all_decode_tps reported; every request reaches its declared output cap; zero dense
#   fallbacks; every producer, semantic, and dense-reference gate is green.
#   Any non-zero harness return code fails closed.
# =============================================================================
set -euo pipefail

GPU="${1:?usage: run_speed.sh <GPU_ID> <MODEL_PATH> <TIER> [sparse|dense] [TAG]}"
MODEL="${2:?usage: run_speed.sh <GPU_ID> <MODEL_PATH> <TIER> [sparse|dense] [TAG]}"
TIER="${3:?tier: bs8x12k | bs8x16k | bs4x24k | bs2x30k | tp8x64k}"
MODE="${4:-sparse}"
TAG="${5:-speed_${TIER}_${MODE}}"
case "${MODE}" in
  sparse|dense) ;;
  *) echo "FAIL: mode must be sparse or dense: ${MODE}" >&2; exit 64 ;;
esac
WITH_DENSE_REFERENCE="${WITH_DENSE_REFERENCE:-0}"
case "${WITH_DENSE_REFERENCE}" in
  0|1) ;;
  *) echo "FAIL: WITH_DENSE_REFERENCE must be 0 or 1: ${WITH_DENSE_REFERENCE}" >&2; exit 64 ;;
esac
if [[ "${MODE}" == "dense" && "${WITH_DENSE_REFERENCE}" == "1" ]]; then
  echo "FAIL: WITH_DENSE_REFERENCE=1 is only valid in sparse mode" >&2
  exit 64
fi
if [[ "${WITH_DENSE_REFERENCE}" == "1" && "${VERDICT_ONLY:-0}" == "1" ]]; then
  echo "FAIL: WITH_DENSE_REFERENCE=1 forbids VERDICT_ONLY=1; a local timed pair requires the full observer-free sparse+dense contract" >&2
  exit 64
fi
RETIRED_ENV_NAMES=(
  "VLLM_SPARSE_SELECTOR_LOG_F_TP8_64K_GROUP"
  "VLLM_SPARSE_FULL_CUDAGRAPH_REPLAY_WRAPPER_ABLATE_EXTRA"
  "VLLM_SPARSE_FULL_CUDAGRAPH_REPLAY_WRAPPER_ABLATE_HIT_LOG"
)
for retired_env_name in "${RETIRED_ENV_NAMES[@]}"; do
  retired_env_value="${!retired_env_name:-}"
  if [[ -n "${retired_env_value}" && "${retired_env_value}" != "0" ]]; then
    echo "FAIL: ${retired_env_name} is retired; unset it." >&2
    exit 64
  fi
done
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

REQUESTED_CUDA_ARCH="${SFI_CUDA_ARCH:-auto}"
case "${REQUESTED_CUDA_ARCH}" in
  auto|sm80|sm90|sm100) ;;
  *)
    echo "FAIL: SFI_CUDA_ARCH must be auto, sm80, sm90, or sm100: ${REQUESTED_CUDA_ARCH}" >&2
    exit 64
    ;;
esac

case "${TIER}" in
  bs8x12k) BS="${BS:-8}"; CTX="${CTX:-12000}"; KVB="${KVB:-19327352832}"; MML="${MML:-16384}" ;;
  bs8x16k) BS="${BS:-8}"; CTX="${CTX:-16000}"; KVB="${KVB:-23622320128}"; MML="${MML:-20480}" ;;
  bs4x24k) BS="${BS:-4}"; CTX="${CTX:-24000}"; KVB="${KVB:-17179869184}"; MML="${MML:-28672}" ;;
  bs2x30k) BS="${BS:-2}"; CTX="${CTX:-30000}"; KVB="${KVB:-17179869184}"; MML="${MML:-36864}" ;;
  tp8x64k) BS="${BS:-32}"; CTX="${CTX:-64000}"; KVB="${KVB:-42949672960}"; MML="${MML:-66560}" ;;
  *) echo "unknown tier: ${TIER}" >&2; exit 2 ;;
esac
if [[ "${TIER}" == "tp8x64k" ]]; then
  MAX_NEW="${MAX_NEW:-2048}"
else
  MAX_NEW="${MAX_NEW:-256}"
fi
CHAT_TEMPLATE_RESERVE_TOKENS=512
REFRESH_INTERVAL="${REFRESH_INTERVAL:-96}"
BLOCKS="${BLOCKS:-112}"
K_HEAD="${K_HEAD:-1536}"
TP="${TP:-1}"
for workload_integer in BS CTX KVB MML MAX_NEW REFRESH_INTERVAL BLOCKS K_HEAD; do
  workload_value="${!workload_integer}"
  if [[ ! "${workload_value}" =~ ^[0-9]+$ ]] || (( 10#${workload_value} <= 0 )); then
    echo "FAIL: ${workload_integer} must be a positive integer: ${workload_value}" >&2
    exit 64
  fi
  printf -v "${workload_integer}" '%d' "$((10#${workload_value}))"
done

normalize_request_vector() {
  local label="$1"
  local raw="$2"
  local expected_count="$3"
  local default_value="$4"
  local index value normalized="" maximum=0
  if [[ -z "${raw}" ]]; then
    for ((index = 0; index < expected_count; index += 1)); do
      raw+="${raw:+,}${default_value}"
    done
  fi
  if [[ ! "${raw}" =~ ^[1-9][0-9]*(,[1-9][0-9]*)*$ ]]; then
    echo "FAIL: ${label} must be a comma-separated list of positive integers: ${raw}" >&2
    exit 64
  fi
  IFS=',' read -r -a REQUEST_VECTOR_VALUES <<< "${raw}"
  if (( ${#REQUEST_VECTOR_VALUES[@]} != expected_count )); then
    echo "FAIL: ${label} must contain exactly BS=${expected_count} values, got ${#REQUEST_VECTOR_VALUES[@]}" >&2
    exit 64
  fi
  for value in "${REQUEST_VECTOR_VALUES[@]}"; do
    if (( 10#${value} <= 0 )); then
      echo "FAIL: every ${label} value must be > 0: ${raw}" >&2
      exit 64
    fi
    value=$((10#${value}))
    normalized+="${normalized:+,}${value}"
    if (( value > maximum )); then
      maximum=${value}
    fi
  done
  IFS=',' read -r -a REQUEST_VECTOR_VALUES <<< "${normalized}"
  REQUEST_VECTOR_NORMALIZED="${normalized}"
  REQUEST_VECTOR_MAXIMUM="${maximum}"
}

normalize_request_vector \
  "REQUEST_CONTEXT_TOKENS" "${REQUEST_CONTEXT_TOKENS:-}" "${BS}" "${CTX}"
REQUEST_CONTEXT_TOKENS="${REQUEST_VECTOR_NORMALIZED}"
REQUEST_CONTEXT_TOKEN_VALUES=("${REQUEST_VECTOR_VALUES[@]}")
REQUEST_CONTEXT_MAXIMUM="${REQUEST_VECTOR_MAXIMUM}"
normalize_request_vector \
  "REQUEST_MAX_NEW_TOKENS" "${REQUEST_MAX_NEW_TOKENS:-}" "${BS}" "${MAX_NEW}"
REQUEST_MAX_NEW_TOKENS="${REQUEST_VECTOR_NORMALIZED}"
REQUEST_MAX_NEW_TOKEN_VALUES=("${REQUEST_VECTOR_VALUES[@]}")
REQUEST_MAX_NEW_MAXIMUM="${REQUEST_VECTOR_MAXIMUM}"
if (( REQUEST_CONTEXT_MAXIMUM != CTX )); then
  echo "FAIL: max(REQUEST_CONTEXT_TOKENS)=${REQUEST_CONTEXT_MAXIMUM} must equal CTX=${CTX}" >&2
  exit 64
fi
if (( REQUEST_MAX_NEW_MAXIMUM != MAX_NEW )); then
  echo "FAIL: max(REQUEST_MAX_NEW_TOKENS)=${REQUEST_MAX_NEW_MAXIMUM} must equal MAX_NEW=${MAX_NEW}" >&2
  exit 64
fi
MAX_REQUEST_SEQUENCE_TOKENS=0
for ((request_index = 0; request_index < BS; request_index += 1)); do
  request_sequence_tokens=$((
    REQUEST_CONTEXT_TOKEN_VALUES[request_index]
    + REQUEST_MAX_NEW_TOKEN_VALUES[request_index]
    + CHAT_TEMPLATE_RESERVE_TOKENS
  ))
  if (( request_sequence_tokens > MAX_REQUEST_SEQUENCE_TOKENS )); then
    MAX_REQUEST_SEQUENCE_TOKENS=${request_sequence_tokens}
  fi
done
export REQUEST_CONTEXT_TOKENS
export REQUEST_MAX_NEW_TOKENS
export SFI_RUNNER_REQUEST_CONTEXT_TOKENS="${REQUEST_CONTEXT_TOKENS}"
export SFI_RUNNER_REQUEST_MAX_NEW_TOKENS="${REQUEST_MAX_NEW_TOKENS}"
# Throughput arms own one explicit engine scheduling policy.  This is not an
# experimental env override: dense and sparse receive the same EngineArgs
# value, and their instantiated scheduler configs are proved in artifacts.
SCHEDULING_MODE="async"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-${BS}}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
MAX_SEQ_LEN_TO_CAPTURE="${MAX_SEQ_LEN_TO_CAPTURE:-${MML}}"
CHUNKED_PREFILL="${CHUNKED_PREFILL:-enabled}"
for scheduler_integer in MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS MAX_SEQ_LEN_TO_CAPTURE; do
  scheduler_value="${!scheduler_integer}"
  if [[ ! "${scheduler_value}" =~ ^[0-9]+$ ]] || (( 10#${scheduler_value} <= 0 )); then
    echo "FAIL: ${scheduler_integer} must be a positive integer: ${scheduler_value}" >&2
    exit 64
  fi
done
MAX_NUM_SEQS=$((10#${MAX_NUM_SEQS}))
MAX_NUM_BATCHED_TOKENS=$((10#${MAX_NUM_BATCHED_TOKENS}))
MAX_SEQ_LEN_TO_CAPTURE=$((10#${MAX_SEQ_LEN_TO_CAPTURE}))
NEEDED_SEQUENCE_TOKENS="${MAX_REQUEST_SEQUENCE_TOKENS}"
if (( MAX_SEQ_LEN_TO_CAPTURE < NEEDED_SEQUENCE_TOKENS || MAX_SEQ_LEN_TO_CAPTURE > MML )); then
  echo "FAIL: MAX_SEQ_LEN_TO_CAPTURE=${MAX_SEQ_LEN_TO_CAPTURE} must cover max per-request context+output+chat_reserve=${NEEDED_SEQUENCE_TOKENS} without exceeding MML=${MML}" >&2
  exit 64
fi
if (( MAX_NUM_SEQS < BS )); then
  echo "FAIL: MAX_NUM_SEQS=${MAX_NUM_SEQS} must be >= BS=${BS}" >&2
  exit 64
fi
if (( MAX_NUM_BATCHED_TOKENS < BS )); then
  echo "FAIL: MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS} must be >= BS=${BS}" >&2
  exit 64
fi
case "${CHUNKED_PREFILL}" in
  enabled|disabled) ;;
  *) echo "FAIL: CHUNKED_PREFILL must be enabled or disabled: ${CHUNKED_PREFILL}" >&2; exit 64 ;;
esac
if [[ ! "${TP}" =~ ^[0-9]+$ ]]; then
  echo "FAIL: TP must be a positive integer: ${TP}" >&2
  exit 64
fi
TP=$((10#${TP}))
if (( TP <= 0 )); then
  echo "FAIL: TP must be a positive integer: ${TP}" >&2
  exit 64
fi
if (( TP != ${#GPU_IDS[@]} )); then
  echo "FAIL: TP=${TP} must match selected GPU count=${#GPU_IDS[@]}" >&2
  exit 64
fi
if [[ "${TIER}" != "tp8x64k" ]] \
   && (( TP == 8 && BS == 32 && CTX == 64000 && MML == 66560 )); then
  echo "FAIL: the TP8 32x64K workload must use tier=tp8x64k so its selector proof cannot be bypassed" >&2
  exit 64
fi
if [[ "${TIER}" == "tp8x64k" ]]; then
  if (( TP != 8 || ${#GPU_IDS[@]} != 8 )); then
    echo "FAIL: tier=tp8x64k requires TP=8 and exactly eight selected GPUs" >&2
    exit 64
  fi
  declare -A TP8_EXACT_VALUES=(
    [BS]=32
    [CTX]=64000
    [KVB]=42949672960
    [MML]=66560
    [MAX_NEW]=2048
    [REFRESH_INTERVAL]=96
    [BLOCKS]=112
    [MAX_NUM_SEQS]=32
    [MAX_NUM_BATCHED_TOKENS]=8192
    [MAX_SEQ_LEN_TO_CAPTURE]=66560
  )
  for exact_name in BS CTX KVB MML MAX_NEW REFRESH_INTERVAL BLOCKS MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS MAX_SEQ_LEN_TO_CAPTURE; do
    exact_actual="${!exact_name}"
    exact_expected="${TP8_EXACT_VALUES[${exact_name}]}"
    if [[ "${exact_actual}" != "${exact_expected}" ]]; then
      echo "FAIL: tier=tp8x64k requires ${exact_name}=${exact_expected}, got ${exact_actual}; use a non-verdict tier for shape experiments" >&2
      exit 64
    fi
  done
  TP8_CONTEXT_VECTOR=""
  TP8_MAX_NEW_VECTOR=""
  for ((request_index = 0; request_index < BS; request_index += 1)); do
    TP8_CONTEXT_VECTOR+="${TP8_CONTEXT_VECTOR:+,}64000"
    TP8_MAX_NEW_VECTOR+="${TP8_MAX_NEW_VECTOR:+,}2048"
  done
  if [[ "${REQUEST_CONTEXT_TOKENS}" != "${TP8_CONTEXT_VECTOR}" ]]; then
    echo "FAIL: tier=tp8x64k requires every REQUEST_CONTEXT_TOKENS value to equal 64000" >&2
    exit 64
  fi
  if [[ "${REQUEST_MAX_NEW_TOKENS}" != "${TP8_MAX_NEW_VECTOR}" ]]; then
    echo "FAIL: tier=tp8x64k requires every REQUEST_MAX_NEW_TOKENS value to equal 2048" >&2
    exit 64
  fi
  if [[ "${MODE}" == "sparse" && "${VLLM_SPARSE_COMPACT_DUAL_GEN:-1}" != "1" ]]; then
    echo "FAIL: sparse tier=tp8x64k requires VLLM_SPARSE_COMPACT_DUAL_GEN=1" >&2
    exit 64
  fi
  if [[ "${CHUNKED_PREFILL}" != "enabled" ]]; then
    echo "FAIL: tier=tp8x64k requires CHUNKED_PREFILL=enabled" >&2
    exit 64
  fi
  if [[ "${VERDICT_ONLY:-0}" == "1" ]]; then
    echo "FAIL: tier=tp8x64k requires the full speed+diagnostic contract; VERDICT_ONLY=1 is directional debug only" >&2
    exit 64
  fi
  if [[ ! "${SFI_EXPECTED_GIT_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "FAIL: tier=tp8x64k requires SFI_EXPECTED_GIT_COMMIT=<exact 40-character lowercase release SHA>" >&2
    exit 64
  fi
  if [[ ! "${SFI_EXPECTED_MODEL_CONFIG_SHA256:-}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "FAIL: tier=tp8x64k requires SFI_EXPECTED_MODEL_CONFIG_SHA256=<exact lowercase config.json SHA256>" >&2
    exit 64
  fi
  if [[ "${SFI_ALLOW_SHARED_GPU:-0}" == "1" ]]; then
    echo "FAIL: tier=tp8x64k forbids SFI_ALLOW_SHARED_GPU=1" >&2
    exit 64
  fi
fi
if [[ "${VLLM_SPARSE_FORCE_DISABLE_CUSTOM_AR:-0}" != "0" \
   && "${VLLM_SPARSE_FORCE_DISABLE_CUSTOM_AR:-0}" != "1" ]]; then
  echo "FAIL: VLLM_SPARSE_FORCE_DISABLE_CUSTOM_AR must be 0 or 1" >&2
  exit 64
fi
if [[ "${TIER}" == "tp8x64k" \
   && "${VLLM_SPARSE_FORCE_DISABLE_CUSTOM_AR:-0}" != "0" ]]; then
  echo "FAIL: tier=tp8x64k requires vLLM custom all-reduce; VLLM_SPARSE_FORCE_DISABLE_CUSTOM_AR must be 0" >&2
  exit 64
fi
if (( TP > 1 )) && [[ "${VLLM_SPARSE_FORCE_DISABLE_CUSTOM_AR:-0}" != "1" ]]; then
  for allocator_var in PYTORCH_ALLOC_CONF PYTORCH_CUDA_ALLOC_CONF; do
    allocator_value="${!allocator_var:-}"
    allocator_compact="${allocator_value//[[:space:]]/}"
    allocator_compact="${allocator_compact,,}"
    if [[ ",${allocator_compact}," == *",expandable_segments:true,"* \
       || ",${allocator_compact}," == *",expandable_segments:1,"* ]]; then
      echo "FAIL: ${allocator_var}=${allocator_value} enables expandable_segments for TP>1 while custom all-reduce is active; CUDA graph IPC cannot register those buffers. Unset the allocator variable, or explicitly set VLLM_SPARSE_FORCE_DISABLE_CUSTOM_AR=1 for a separate matched experiment." >&2
      exit 78
    fi
  done
fi
TP_ARGS=()
if [[ "${TP}" -gt 1 ]]; then
  TP_ARGS=(--tensor-parallel-size "${TP}")
fi
# VERDICT_ONLY=1 skips the sparse diagnostic child (verdict-grade screening:
# proofs re-source from the speed-child trace).  Dense mode has no equivalent
# one-child contract, so it keeps the full form instead of forwarding an
# argument that the harness correctly rejects.
VERDICT_ARGS=()
if [[ "${VERDICT_ONLY:-0}" == "1" && "${MODE}" == "sparse" ]]; then
  VERDICT_ARGS=(--verdict-only)
fi
REFERENCE_ARGS=(--skip-dense-reference)
SFI_RUNNER_PAIR_CONTRACT="none"
PAIR_POSTFLIGHT_ARGS=()
if [[ "${TIER}" == "tp8x64k" ]]; then
  REFERENCE_ARGS=()
  if [[ "${MODE}" == "sparse" ]]; then
    SFI_RUNNER_PAIR_CONTRACT="exact_speedup_verdict"
  fi
elif [[ "${MODE}" == "sparse" && "${WITH_DENSE_REFERENCE}" == "1" ]]; then
  REFERENCE_ARGS=()
  SFI_RUNNER_PAIR_CONTRACT="explicit_local_comparison"
  PAIR_POSTFLIGHT_ARGS=(--expect-local-paired-comparison)
fi
export SFI_RUNNER_PAIR_CONTRACT

SFI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FA_ROOT="${VLLM_SPARSE_FA3_UPSTREAM_ROOT:-${SFI_ROOT}/third_party_upstreams/vllm-project-flash-attention}"
# PYTHON is REQUIRED (fail-fast, no bare-`python` fallback): cache partitioning
# and every child must use the exact interpreter that will load the extension.
# A PATH-dependent alias can drift between shells and invalidate that identity.
PY="${PYTHON:?PYTHON env required: absolute path of the benchmark interpreter (for example /absolute/path/to/environment/bin/python); bare 'python' can poison the shared extension cache with a wrong-ABI build}"
if [[ "${PY}" != /* || ! -x "${PY}" ]]; then
  echo "FAIL: PYTHON must be an executable absolute path: ${PY}" >&2
  exit 64
fi

if ! command -v realpath >/dev/null 2>&1; then
  echo "FAIL: realpath is required for stable benchmark path identities" >&2
  exit 69
fi
PY="$(realpath -e -- "${PY}")"
PYTHON="${PY}"
export PYTHON
# Canonicalize lexically while still in the caller's working directory. The
# process later cd's to SFI_ROOT, so leaving a relative MODEL value here would
# make corpus generation and execution refer to different files.
MODEL="$(realpath -ms -- "${MODEL}")"
FA_ROOT="$(realpath -ms -- "${FA_ROOT}")"
if [[ -n "${CORPUS:-}" ]]; then
  CORPUS="$(realpath -ms -- "${CORPUS}")"
fi

# Derive one model identity for every tier before launch.  Capacity and pair
# geometry must never depend on a model-family constant or an unverified
# override; TP8 adds an externally pinned config hash to this shared contract.
MODEL_KV_SCHEMA=""
MODEL_CONFIG_PATH=""
MODEL_CONFIG_SHA256=""
MODEL_NUM_HIDDEN_LAYERS=""
MODEL_NUM_KEY_VALUE_HEADS=""
MODEL_HEAD_DIM=""
MODEL_KV_DTYPE=""
MODEL_KV_DTYPE_BYTES=""
MODEL_KV_TOTAL_BYTES_PER_TOKEN=""
MODEL_KV_PER_RANK_BYTES_PER_TOKEN=""
KV_TOKEN_BYTES_OVERRIDE_PRESENT="0"
if ! MODEL_KV_TSV="$(
  "${PY}" "${SFI_ROOT}/utils/model_kv_contract.py" \
    --model "${MODEL}" --tensor-parallel-size "${TP}" --format tsv
)"; then
  echo "FAIL: cannot derive an exact per-rank KV contract from ${MODEL}/config.json" >&2
  exit 78
fi
IFS=$'\t' read -r \
  MODEL_KV_SCHEMA MODEL_CONFIG_PATH MODEL_CONFIG_SHA256 \
  MODEL_NUM_HIDDEN_LAYERS MODEL_NUM_KEY_VALUE_HEADS MODEL_HEAD_DIM \
  MODEL_KV_DTYPE MODEL_KV_DTYPE_BYTES MODEL_KV_TP \
  MODEL_KV_TOTAL_BYTES_PER_TOKEN MODEL_KV_PER_RANK_BYTES_PER_TOKEN \
  <<< "${MODEL_KV_TSV}"
if [[ "${MODEL_KV_SCHEMA}" != "sfi.model_kv_contract.v1" \
   || "${MODEL_KV_TP}" != "${TP}" \
   || ! "${MODEL_KV_TOTAL_BYTES_PER_TOKEN}" =~ ^[0-9]+$ \
   || ! "${MODEL_KV_PER_RANK_BYTES_PER_TOKEN}" =~ ^[0-9]+$ \
   || "${MODEL_KV_PER_RANK_BYTES_PER_TOKEN}" == "0" \
   || ! "${MODEL_CONFIG_SHA256}" =~ ^[0-9a-f]{64}$ \
   || -z "${MODEL_CONFIG_PATH}" ]]; then
  echo "FAIL: invalid model-derived KV contract: ${MODEL_KV_TSV}" >&2
  exit 78
fi
if [[ "${TIER}" == "tp8x64k" ]]; then
  if [[ "${MODEL_CONFIG_SHA256}" != "${SFI_EXPECTED_MODEL_CONFIG_SHA256}" ]]; then
    echo "FAIL: model config identity mismatch: derived=${MODEL_CONFIG_SHA256} expected=${SFI_EXPECTED_MODEL_CONFIG_SHA256}" >&2
    exit 78
  fi
fi
if [[ -n "${KV_TOKEN_BYTES:-}" ]]; then
  KV_TOKEN_BYTES_OVERRIDE_PRESENT="1"
  if [[ ! "${KV_TOKEN_BYTES}" =~ ^[0-9]+$ \
     || "${KV_TOKEN_BYTES}" != "${MODEL_KV_PER_RANK_BYTES_PER_TOKEN}" ]]; then
    echo "FAIL: KV_TOKEN_BYTES override must equal model-derived per-rank value ${MODEL_KV_PER_RANK_BYTES_PER_TOKEN}, got ${KV_TOKEN_BYTES}" >&2
    exit 78
  fi
fi
KV_TOKEN_BYTES="${MODEL_KV_PER_RANK_BYTES_PER_TOKEN}"

export SFI_RUNNER_MODEL_KV_CONTRACT_SCHEMA="${MODEL_KV_SCHEMA}"
export SFI_RUNNER_MODEL_CONFIG_PATH="${MODEL_CONFIG_PATH}"
export SFI_RUNNER_MODEL_CONFIG_SHA256="${MODEL_CONFIG_SHA256}"
export SFI_RUNNER_EXPECTED_MODEL_CONFIG_SHA256="${SFI_EXPECTED_MODEL_CONFIG_SHA256:-}"
export SFI_RUNNER_MODEL_NUM_HIDDEN_LAYERS="${MODEL_NUM_HIDDEN_LAYERS}"
export SFI_RUNNER_MODEL_NUM_KEY_VALUE_HEADS="${MODEL_NUM_KEY_VALUE_HEADS}"
export SFI_RUNNER_MODEL_HEAD_DIM="${MODEL_HEAD_DIM}"
export SFI_RUNNER_MODEL_KV_DTYPE="${MODEL_KV_DTYPE}"
export SFI_RUNNER_MODEL_KV_DTYPE_BYTES="${MODEL_KV_DTYPE_BYTES}"
export SFI_RUNNER_MODEL_KV_TOTAL_BYTES_PER_TOKEN="${MODEL_KV_TOTAL_BYTES_PER_TOKEN}"
export SFI_RUNNER_KV_TOKEN_BYTES_PER_RANK_REQUESTED="${MODEL_KV_PER_RANK_BYTES_PER_TOKEN:-${KV_TOKEN_BYTES}}"
export SFI_RUNNER_KV_TOKEN_BYTES_PER_RANK_EFFECTIVE="${KV_TOKEN_BYTES}"
export SFI_RUNNER_KV_TOKEN_BYTES_OVERRIDE_PRESENT="${KV_TOKEN_BYTES_OVERRIDE_PRESENT}"

# --- Sparse KV-budget preflight (single fail-closed admission path) ---
SFI_RUNNER_KV_PREFLIGHT_STATUS="not_applicable"
SFI_RUNNER_KV_REQUIRED_TOKEN_BLOCKS=0
SFI_RUNNER_KV_REQUIRED_TOKENS_PADDED=0
SFI_RUNNER_KV_REQUIRED_BYTES=0
SFI_RUNNER_KV_COMPACT_LEASE_BYTES=0
if [[ "${MODE}" == "sparse" ]]; then
  GEN_COUNT=2
  if [[ "${VLLM_SPARSE_COMPACT_DUAL_GEN:-1}" == "0" ]]; then
    GEN_COUNT=1
  fi
  LEASE_BYTES=$((BS * BLOCKS * 16 * KV_TOKEN_BYTES * GEN_COUNT))
  FULL_KV_REQUIRED_BLOCKS=0
  for ((request_index = 0; request_index < BS; request_index += 1)); do
    request_required_tokens=$((
      REQUEST_CONTEXT_TOKEN_VALUES[request_index]
      + REQUEST_MAX_NEW_TOKEN_VALUES[request_index]
      + CHAT_TEMPLATE_RESERVE_TOKENS
    ))
    request_required_blocks=$(((request_required_tokens + 15) / 16))
    FULL_KV_REQUIRED_BLOCKS=$((FULL_KV_REQUIRED_BLOCKS + request_required_blocks))
  done
  FULL_KV_REQUIRED_TOKENS_PADDED=$((FULL_KV_REQUIRED_BLOCKS * 16))
  FULL_KV_REQUIRED_BYTES=$((FULL_KV_REQUIRED_TOKENS_PADDED * KV_TOKEN_BYTES))
  TOTAL_KV_REQUIRED_BYTES=$((FULL_KV_REQUIRED_BYTES + LEASE_BYTES))
  SFI_RUNNER_KV_REQUIRED_TOKEN_BLOCKS="${FULL_KV_REQUIRED_BLOCKS}"
  SFI_RUNNER_KV_REQUIRED_TOKENS_PADDED="${FULL_KV_REQUIRED_TOKENS_PADDED}"
  SFI_RUNNER_KV_REQUIRED_BYTES="${TOTAL_KV_REQUIRED_BYTES}"
  SFI_RUNNER_KV_COMPACT_LEASE_BYTES="${LEASE_BYTES}"
  SFI_RUNNER_KV_PREFLIGHT_STATUS="passed"
  if ((KVB < TOTAL_KV_REQUIRED_BYTES)); then
    KVB_SUGGEST_GIB=$(((TOTAL_KV_REQUIRED_BYTES + 1073741824 - 1) / 1073741824))
    SFI_RUNNER_KV_PREFLIGHT_STATUS="failed"
    cat >&2 <<EOW
==============================================================================
FAIL: sparse KV pool too small for this shape.
The pool must hold full KV + compact lease without scheduler serialization:
  compact lease      = ${BS} slots x ${BLOCKS} blocks x 16 x ${KV_TOKEN_BYTES} B x ${GEN_COUNT} gen = ${LEASE_BYTES} B
  full KV blocks     = sum_i ceil((ctx_i + max_new_i + ${CHAT_TEMPLATE_RESERVE_TOKENS}) / 16) = ${FULL_KV_REQUIRED_BLOCKS}
  full KV bytes      = ${FULL_KV_REQUIRED_BLOCKS} blocks x 16 x ${KV_TOKEN_BYTES} B = ${FULL_KV_REQUIRED_BYTES} B
  total required     = full KV + compact lease = ${TOTAL_KV_REQUIRED_BYTES} B
Fix: KVB >= ~${KVB_SUGGEST_GIB} GiB (KVB=$((KVB_SUGGEST_GIB * 1073741824))),
or lower CTX/MAX_NEW/BS/BLOCKS. Refusing to launch.
==============================================================================
EOW
    exit 78
  fi
fi
export SFI_RUNNER_KV_PREFLIGHT_STATUS
export SFI_RUNNER_KV_REQUIRED_TOKEN_BLOCKS
export SFI_RUNNER_KV_REQUIRED_TOKENS_PADDED
export SFI_RUNNER_KV_REQUIRED_BYTES
export SFI_RUNNER_KV_COMPACT_LEASE_BYTES
export SFI_RUNNER_MAX_REQUEST_SEQUENCE_TOKENS="${MAX_REQUEST_SEQUENCE_TOKENS}"

EXPECTED_GIT_COMMIT="${SFI_EXPECTED_GIT_COMMIT:-}"
if [[ -n "${EXPECTED_GIT_COMMIT}" && ! "${EXPECTED_GIT_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "FAIL: SFI_EXPECTED_GIT_COMMIT must be one exact 40-character lowercase commit SHA" >&2
  exit 64
fi
RUNNER_GIT_HEAD="$(git -C "${SFI_ROOT}" rev-parse HEAD 2>/dev/null || true)"
RUNNER_TRACKED_STATUS=""
RUNNER_CODE_SCOPE_UNTRACKED_STATUS=""
RUNNER_CODE_SCOPE="benchmarks,patches,scripts,utils,sitecustomize.py"
if [[ -n "${RUNNER_GIT_HEAD}" ]]; then
  RUNNER_TRACKED_STATUS="$(git -C "${SFI_ROOT}" status --short --untracked-files=no -- .)"
  RUNNER_CODE_SCOPE_UNTRACKED_STATUS="$(
    git -C "${SFI_ROOT}" ls-files --others --exclude-standard -- \
      benchmarks patches scripts utils sitecustomize.py \
      | LC_ALL=C sort \
      | awk '/\.(py|sh|patch|cu|cuh|cc|cpp|h|hpp|toml)$/ { print }'
  )"
fi
RUNNER_GIT_TRACKED_CLEAN="0"
RUNNER_CODE_SCOPE_UNTRACKED_CLEAN="0"
if [[ -z "${RUNNER_TRACKED_STATUS}" ]]; then
  RUNNER_GIT_TRACKED_CLEAN="1"
fi
if [[ -z "${RUNNER_CODE_SCOPE_UNTRACKED_STATUS}" ]]; then
  RUNNER_CODE_SCOPE_UNTRACKED_CLEAN="1"
fi
if [[ -n "${EXPECTED_GIT_COMMIT}" ]]; then
  if [[ -z "${RUNNER_GIT_HEAD}" ]]; then
    echo "FAIL: SFI_EXPECTED_GIT_COMMIT requires a readable Git checkout: ${SFI_ROOT}" >&2
    exit 69
  fi
  if [[ "${RUNNER_GIT_HEAD}" != "${EXPECTED_GIT_COMMIT}" ]]; then
    echo "FAIL: release commit mismatch: HEAD=${RUNNER_GIT_HEAD} expected=${EXPECTED_GIT_COMMIT}" >&2
    exit 78
  fi
  if [[ "${RUNNER_GIT_TRACKED_CLEAN}" != "1" ]]; then
    echo "FAIL: release checkout has tracked changes; refusing attributable run" >&2
    printf '%s\n' "${RUNNER_TRACKED_STATUS}" >&2
    exit 78
  fi
  if [[ "${TIER}" == "tp8x64k" && "${RUNNER_CODE_SCOPE_UNTRACKED_CLEAN}" != "1" ]]; then
    echo "FAIL: tier=tp8x64k code scope contains untracked executable sources; commit or remove them before an attributable run" >&2
    printf '%s\n' "${RUNNER_CODE_SCOPE_UNTRACKED_STATUS}" >&2
    exit 78
  fi
fi
export SFI_RUNNER_EXPECTED_GIT_COMMIT="${EXPECTED_GIT_COMMIT}"
export SFI_RUNNER_GIT_HEAD="${RUNNER_GIT_HEAD}"
export SFI_RUNNER_GIT_TRACKED_CLEAN="${RUNNER_GIT_TRACKED_CLEAN}"
export SFI_RUNNER_CODE_SCOPE="${RUNNER_CODE_SCOPE}"
export SFI_RUNNER_CODE_SCOPE_UNTRACKED_STATUS="${RUNNER_CODE_SCOPE_UNTRACKED_STATUS}"
export SFI_RUNNER_CODE_SCOPE_UNTRACKED_CLEAN="${RUNNER_CODE_SCOPE_UNTRACKED_CLEAN}"
EXPECTED_GIT_ARGS=()
if [[ -n "${EXPECTED_GIT_COMMIT}" ]]; then
  EXPECTED_GIT_ARGS=(--expected-git-commit "${EXPECTED_GIT_COMMIT}")
fi
EXPECTED_MODEL_CONFIG_ARGS=()
if [[ "${TIER}" == "tp8x64k" ]]; then
  EXPECTED_MODEL_CONFIG_ARGS=(
    --expected-model-config-sha256 "${SFI_EXPECTED_MODEL_CONFIG_SHA256}"
  )
fi
OUT="${SFI_ROOT}/out"
if [[ -L "${OUT}" || ( -e "${OUT}" && ! -d "${OUT}" ) ]]; then
  echo "FAIL: out must be an absent or real directory: ${OUT}" >&2
  exit 73
fi
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
SFI_RUNNER_GPU_LOCK_SCOPE="none"
if [[ "${SFI_ALLOW_SHARED_GPU:-0}" != "1" ]]; then
  SFI_RUNNER_GPU_LOCK_MODE="exclusive"
  GPU_LOCK_ROOT="${SFI_GPU_LOCK_ROOT:-${TMPDIR:-/tmp}/sfi-run-speed-gpu-locks-${UID}}"
  mkdir -p "${GPU_LOCK_ROOT}"
  GPU_LOCK_ROOT="$(realpath -ms -- "${GPU_LOCK_ROOT}")"
  if [[ -n "${SFI_PREHELD_GPU_LOCK_FDS:-}" ]]; then
    IFS=',' read -r -a PREHELD_GPU_LOCK_ITEMS <<< "${SFI_PREHELD_GPU_LOCK_FDS}"
    if (( ${#PREHELD_GPU_LOCK_ITEMS[@]} != ${#GPU_IDS[@]} )); then
      echo "FAIL: inherited GPU lock count does not match selected GPU count" >&2
      exit 76
    fi
    for logical_rank in "${!GPU_IDS[@]}"; do
      gpu_id="${GPU_IDS[logical_rank]}"
      lock_item="${PREHELD_GPU_LOCK_ITEMS[logical_rank]}"
      expected_prefix="${gpu_id}="
      if [[ "${lock_item}" != "${expected_prefix}"* ]]; then
        echo "FAIL: inherited GPU lock mapping is not rank-aligned: ${lock_item}" >&2
        exit 76
      fi
      preheld_fd="${lock_item#${expected_prefix}}"
      if [[ ! "${preheld_fd}" =~ ^[0-9]+$ || ! -e "/proc/$$/fd/${preheld_fd}" ]]; then
        echo "FAIL: inherited GPU lock descriptor is unavailable: ${lock_item}" >&2
        exit 76
      fi
      gpu_lock_path="${GPU_LOCK_ROOT}/gpu_${gpu_id}.lock"
      inherited_path="$(readlink -f -- "/proc/$$/fd/${preheld_fd}" || true)"
      if [[ "${inherited_path}" != "${gpu_lock_path}" ]]; then
        echo "FAIL: inherited GPU lock descriptor path mismatch: fd=${preheld_fd} actual=${inherited_path} expected=${gpu_lock_path}" >&2
        exit 76
      fi
      if ! flock -n "${preheld_fd}"; then
        echo "FAIL: inherited GPU lock descriptor is not exclusively held: ${lock_item}" >&2
        exit 76
      fi
    done
    SFI_RUNNER_GPU_LOCK_SCOPE="pair"
  else
    for gpu_id in "${GPU_IDS[@]}"; do
      gpu_lock_path="${GPU_LOCK_ROOT}/gpu_${gpu_id}.lock"
      reject_symlink_output "${gpu_lock_path}"
      exec {gpu_lock_fd}>"${gpu_lock_path}"
      if ! flock -n "${gpu_lock_fd}"; then
        echo "FAIL: GPU '${gpu_id}' is already reserved by another run (lock: ${gpu_lock_path})" >&2
        exit 76
      fi
    done
    SFI_RUNNER_GPU_LOCK_SCOPE="arm"
  fi
fi
if [[ "${SFI_RUNNER_PAIR_CONTRACT}" != "none" \
   && "${SFI_RUNNER_GPU_LOCK_MODE}" == "exclusive" ]]; then
  SFI_RUNNER_GPU_LOCK_SCOPE="pair"
fi
export SFI_RUNNER_GPU_LOCK_MODE
export SFI_RUNNER_GPU_LOCK_SCOPE

# Inspect every selected physical GPU through the exact benchmark interpreter.
# CUDA_VISIBLE_DEVICES remaps the physical IDs to logical TP ranks; requiring
# the exact visible count avoids silently benchmarking a partial device list.
CUDA_CAPABILITIES="$(
  CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" -I -c '
import sys
import torch

expected = int(sys.argv[1])
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
visible = int(torch.cuda.device_count())
if visible != expected:
    raise SystemExit(
        f"selected GPU count mismatch: requested {expected}, visible {visible}"
    )
capabilities = []
for logical_rank in range(expected):
    major, minor = (
        int(value) for value in torch.cuda.get_device_capability(logical_rank)
    )
    capabilities.append(f"{major}.{minor}")
print(",".join(capabilities))
' "${#GPU_IDS[@]}"
)" || {
  echo "FAIL: cannot inspect all selected GPUs ${GPU}: ${CUDA_CAPABILITIES}" >&2
  exit 69
}

IFS=',' read -r -a CUDA_CAPABILITY_LIST <<< "${CUDA_CAPABILITIES}"
if (( ${#CUDA_CAPABILITY_LIST[@]} != ${#GPU_IDS[@]} )); then
  echo "FAIL: CUDA capability detection returned ${#CUDA_CAPABILITY_LIST[@]} ranks for ${#GPU_IDS[@]} selected GPUs: ${CUDA_CAPABILITIES}" >&2
  exit 69
fi
DETECTED_CUDA_ARCH=""
for logical_rank in "${!GPU_IDS[@]}"; do
  capability="${CUDA_CAPABILITY_LIST[logical_rank]}"
  case "${capability}" in
    8.0) rank_arch="sm80" ;;
    9.0) rank_arch="sm90" ;;
    10.0) rank_arch="sm100" ;;
    *)
      echo "FAIL: selected GPU ${GPU_IDS[logical_rank]} (rank ${logical_rank}) has unsupported compute capability ${capability}; expected exact 8.0, 9.0, or 10.0" >&2
      exit 78
      ;;
  esac
  if [[ -z "${DETECTED_CUDA_ARCH}" ]]; then
    DETECTED_CUDA_ARCH="${rank_arch}"
  elif [[ "${rank_arch}" != "${DETECTED_CUDA_ARCH}" ]]; then
    echo "FAIL: selected GPUs must be architecture-homogeneous across TP ranks; capabilities=${CUDA_CAPABILITIES}" >&2
    exit 78
  fi
done
if [[ "${REQUESTED_CUDA_ARCH}" != "auto" && "${REQUESTED_CUDA_ARCH}" != "${DETECTED_CUDA_ARCH}" ]]; then
  echo "FAIL: SFI_CUDA_ARCH=${REQUESTED_CUDA_ARCH} does not match detected selected-GPU architecture ${DETECTED_CUDA_ARCH} (capabilities=${CUDA_CAPABILITIES})" >&2
  exit 78
fi
CUDA_ARCH="${DETECTED_CUDA_ARCH}"
if [[ "${TIER}" == "tp8x64k" && "${CUDA_ARCH}" != "sm80" ]]; then
  echo "FAIL: verdict tier=tp8x64k requires a homogeneous SM80 GPU set; detected ${CUDA_ARCH}" >&2
  exit 64
fi
SFI_RUNNER_GPU_TOTAL_MEMORY_BYTES=""
SFI_RUNNER_GPU_FREE_MEMORY_BYTES=""
SFI_RUNNER_GPU_PHYSICAL_CAPACITY_STATUS="not_required"
if [[ "${TIER}" == "tp8x64k" ]]; then
  GPU_MEMORY_PROOF="$({
    CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" -I -c '
import sys
import torch

expected = int(sys.argv[1])
if int(torch.cuda.device_count()) != expected:
    raise SystemExit("selected GPU count changed during memory proof")
totals = []
frees = []
for logical_rank in range(expected):
    totals.append(str(int(torch.cuda.get_device_properties(logical_rank).total_memory)))
    free_bytes, _total_bytes = torch.cuda.mem_get_info(logical_rank)
    frees.append(str(int(free_bytes)))
print(",".join(totals) + "|" + ",".join(frees))
' "${#GPU_IDS[@]}"
  } 2>&1)" || {
    echo "FAIL: cannot prove per-rank physical GPU capacity: ${GPU_MEMORY_PROOF}" >&2
    exit 69
  }
  IFS='|' read -r SFI_RUNNER_GPU_TOTAL_MEMORY_BYTES SFI_RUNNER_GPU_FREE_MEMORY_BYTES <<< "${GPU_MEMORY_PROOF}"
  IFS=',' read -r -a GPU_TOTAL_MEMORY_LIST <<< "${SFI_RUNNER_GPU_TOTAL_MEMORY_BYTES}"
  IFS=',' read -r -a GPU_FREE_MEMORY_LIST <<< "${SFI_RUNNER_GPU_FREE_MEMORY_BYTES}"
  if (( ${#GPU_TOTAL_MEMORY_LIST[@]} != TP || ${#GPU_FREE_MEMORY_LIST[@]} != TP )); then
    echo "FAIL: physical GPU capacity proof rank count mismatch: ${GPU_MEMORY_PROOF}" >&2
    exit 69
  fi
  for logical_rank in "${!GPU_TOTAL_MEMORY_LIST[@]}"; do
    total_bytes="${GPU_TOTAL_MEMORY_LIST[logical_rank]}"
    free_bytes="${GPU_FREE_MEMORY_LIST[logical_rank]}"
    if [[ ! "${total_bytes}" =~ ^[0-9]+$ || ! "${free_bytes}" =~ ^[0-9]+$ ]]; then
      echo "FAIL: invalid GPU memory proof for rank ${logical_rank}: total=${total_bytes} free=${free_bytes}" >&2
      exit 69
    fi
    if (( KVB >= total_bytes )); then
      echo "FAIL: tier=tp8x64k requests KVB=${KVB} bytes per rank but GPU rank ${logical_rank} has only ${total_bytes} total bytes" >&2
      exit 78
    fi
    if (( KVB >= free_bytes )); then
      echo "FAIL: tier=tp8x64k requests KVB=${KVB} bytes per rank but GPU rank ${logical_rank} has only ${free_bytes} free bytes before model load" >&2
      exit 78
    fi
  done
  SFI_RUNNER_GPU_PHYSICAL_CAPACITY_STATUS="passed"
fi
export SFI_RUNNER_GPU_TOTAL_MEMORY_BYTES
export SFI_RUNNER_GPU_FREE_MEMORY_BYTES
export SFI_RUNNER_GPU_PHYSICAL_CAPACITY_STATUS
case "${CUDA_ARCH}" in
  sm80|sm90)
    ATTENTION_BACKEND="fa3"
    ATTENTION_KERNEL="fa3-native"
    FLASH_ATTN_VERSION="3"
    # This is the generic throughput engine despite its historical SM80 name.
    # The SM90 wrapper is a correctness-only gate and intentionally does not
    # accept the throughput workload-shape arguments used below.
    BENCHMARK_MODULE="benchmarks.bench_sm80_mixed_page_one_shot_graph_e2e"
    ;;
  sm100)
    ATTENTION_BACKEND="fa4-sm100"
    ATTENTION_KERNEL="fa4-cute"
    FLASH_ATTN_VERSION="4"
    BENCHMARK_MODULE="benchmarks.bench_sm100_fa4_mixed_page_one_shot_graph_e2e"
    ;;
esac
export SFI_RUNNER_CUDA_CAPABILITIES="${CUDA_CAPABILITIES}"

# Resolve the exact build-time CUDA toolkit before deriving a selector cache
# identity or entering any FA4/selector JIT path.  The resolver delegates the
# source/binary proof to the canonical checker, then rejects caller overrides
# that select a different compiler.
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
    --architecture "${CUDA_ARCH}" \
    --checker "${BUILD_PROVENANCE_CHECKER}"
)"; then
  echo "FAIL: FlashAttention CUDA toolchain preflight failed" >&2
  exit 78
fi
eval "${CUDA_TOOLCHAIN_EXPORTS}"
unset CUDA_TOOLCHAIN_EXPORTS
echo "==> CUDA toolchain: home=${CUDA_HOME} compiler=${CUDACXX} release=${SFI_RUNNER_CUDA_COMPILER_RELEASE}"

# The extension loader keys builds by module name, so every cache root must be
# owned by the selected Python/Torch/CUDA ABI.  An explicit directory is only
# a caller-owned base; it cannot bypass the keyed child partition.
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
if [[ -n "${TORCH_EXTENSIONS_DIR:-}" ]]; then
  SELECTOR_CACHE_BASE="$(realpath -ms -- "${TORCH_EXTENSIONS_DIR}")"
else
  SELECTOR_CACHE_BASE="${SFI_ROOT}/tmp/torch_extensions"
fi
SELECTOR_CACHE_ROOT="${SELECTOR_CACHE_BASE}/${CUDA_ARCH}_gt1_${SELECTOR_CACHE_ABI_KEY}"
export TORCH_EXTENSIONS_DIR="${SELECTOR_CACHE_ROOT}"
export SFI_RUNNER_SELECTOR_CACHE_ROOT="${SELECTOR_CACHE_ROOT}"

# Fail before corpus generation or selector prewarm when the architecture-
# matched FlashAttention checkout is incomplete.  The SM100 path shares the
# Python bridge sources from the FA3 base patch, but it must not require the
# FA3 shared object: FA4 CuTe is JIT compiled from source.
FA3_INTERFACE="${FA_ROOT}/vllm_flash_attn/flash_attn_interface.py"
if [[ ! -f "${FA3_INTERFACE}" ]]; then
  echo "FAIL: vendored FA3-base FlashAttention bridge is incomplete: ${FA3_INTERFACE}" >&2
  echo "Run PYTHON=${PY} bash scripts/setup_flash_attention.sh --arch ${CUDA_ARCH} --gpu ${GPU_IDS[0]} first." >&2
  exit 66
fi
if [[ "${ATTENTION_KERNEL}" == "fa3-native" ]]; then
  if ! compgen -G "${FA_ROOT}/vllm_flash_attn/_vllm_fa3_C*.so" >/dev/null; then
    echo "FAIL: vendored FA3 shared object is missing under ${FA_ROOT}/vllm_flash_attn" >&2
    echo "Run PYTHON=${PY} bash scripts/setup_flash_attention.sh --arch ${CUDA_ARCH} --gpu ${GPU_IDS[0]} first." >&2
    exit 66
  fi
  SFI_RUNNER_FA3_PREFLIGHT_STATUS="passed"
  ATTENTION_PREFLIGHT_DETAIL="fa3_shared_object_ready"
else
  FA4_CUTE_INTERFACE="${FA_ROOT}/flash_attn/cute/interface.py"
  FA4_JIT_PREFLIGHT="${SFI_ROOT}/scripts/probe_fa4_cute_sm100_compact_recent_fake_compile.py"
  if [[ ! -f "${FA4_CUTE_INTERFACE}" ]]; then
    echo "FAIL: vendored FA4 CuTe interface is missing: ${FA4_CUTE_INTERFACE}" >&2
    echo "Run PYTHON=${PY} bash scripts/setup_flash_attention.sh --arch sm100 --gpu ${GPU_IDS[0]} first." >&2
    exit 66
  fi
  if [[ ! -f "${FA4_JIT_PREFLIGHT}" ]]; then
    echo "FAIL: FA4 CuTe JIT preflight is missing: ${FA4_JIT_PREFLIGHT}" >&2
    exit 66
  fi
  if ! FA4_PREFLIGHT_OUTPUT="$({
    CUDA_VISIBLE_DEVICES="${GPU}" \
    VLLM_SPARSE_FA3_UPSTREAM_ROOT="${FA_ROOT}" \
    FLASH_ATTENTION_ARCH="sm_100a" CUTE_DSL_ARCH="sm_100a" \
      "${PY}" -I "${FA4_JIT_PREFLIGHT}"
  } 2>&1)"; then
    echo "FAIL: FA4 CuTe fake-JIT preflight failed: ${FA4_PREFLIGHT_OUTPUT}" >&2
    exit 70
  fi
  for marker in \
    sm100_cute_mixed_page_no_selected_fake_compile_probe_ok \
    sm100_cute_mixed_page_selected_brow_fake_compile_probe_ok \
    sm100_cute_mixed_page_selected_bhrow_capture_fake_compile_probe_ok \
    sm100_cute_mixed_page_rrp_fake_compile_probe_ok; do
    if [[ "${FA4_PREFLIGHT_OUTPUT}" != *"${marker}"* ]]; then
      echo "FAIL: FA4 CuTe fake-JIT preflight did not emit ${marker}" >&2
      exit 70
    fi
  done
  SFI_RUNNER_FA3_PREFLIGHT_STATUS="not_applicable"
  ATTENTION_PREFLIGHT_DETAIL="fa4_cute_fake_jit_ready"
fi
SFI_RUNNER_ATTENTION_PREFLIGHT_STATUS="passed"
export SFI_RUNNER_FA3_PREFLIGHT_STATUS
export SFI_RUNNER_ATTENTION_PREFLIGHT_STATUS
export SFI_RUNNER_ATTENTION_ARCH="${CUDA_ARCH}"
export SFI_RUNNER_ATTENTION_KERNEL="${ATTENTION_KERNEL}"
export SFI_RUNNER_ATTENTION_BACKEND="${ATTENTION_BACKEND}"
export SFI_RUNNER_FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION}"
export SFI_CUDA_ARCH="${CUDA_ARCH}"
export SFI_ATTENTION_KERNEL="${ATTENTION_KERNEL}"
export VLLM_FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION}"
if [[ "${ATTENTION_KERNEL}" == "fa4-cute" ]]; then
  export FLASH_ATTENTION_ARCH="sm_100a"
  export CUTE_DSL_ARCH="sm_100a"
fi

if [[ -n "${CORPUS:-}" ]]; then
  echo "FAIL: explicit CORPUS is retired; release runs require the content-addressed v6 vector corpus cache" >&2
  exit 64
fi
CORPUS_RESULT="$("${PY}" "${SFI_ROOT}/scripts/make_context_corpus.py" \
  --model "${MODEL}" --segments "${BS}" \
  --tokens-per-segment-by-request "${REQUEST_CONTEXT_TOKENS}" \
  --cache-dir "${OUT}/context_corpus_cache")"
CORPUS="$(printf '%s\n' "${CORPUS_RESULT}" | sed -n 's/^context_corpus_path=//p' | tail -1)"
if [[ -z "${CORPUS}" || ! -f "${CORPUS}" ]]; then
  echo "FAIL: context corpus cache returned no readable path" >&2
  exit 66
fi
SFI_RUNNER_CORPUS_TOKEN_STATUS="cache_exact"
export SFI_RUNNER_CORPUS_TOKEN_STATUS
CHAT_TEMPLATE_TSV="$({
  "${PY}" -I "${SFI_ROOT}/scripts/check_chat_template_overhead.py" \
    --model "${MODEL}" \
    --corpus "${CORPUS}" \
    --batch-size "${BS}" \
    --context-tokens-by-request "${REQUEST_CONTEXT_TOKENS}" \
    --reserve-tokens "${CHAT_TEMPLATE_RESERVE_TOKENS}" \
    --format tsv
})" || {
  echo "FAIL: exact chat-template token preflight failed" >&2
  exit 78
}
IFS=$'\t' read -r \
  SFI_RUNNER_CHAT_TEMPLATE_OVERHEAD_MIN \
  SFI_RUNNER_CHAT_TEMPLATE_OVERHEAD_MAX \
  SFI_RUNNER_CHAT_TEMPLATE_VERIFIED_COUNT \
  <<< "${CHAT_TEMPLATE_TSV}"
export SFI_RUNNER_CHAT_TEMPLATE_RESERVE_TOKENS="${CHAT_TEMPLATE_RESERVE_TOKENS}"
export SFI_RUNNER_CHAT_TEMPLATE_OVERHEAD_MIN
export SFI_RUNNER_CHAT_TEMPLATE_OVERHEAD_MAX
export SFI_RUNNER_CHAT_TEMPLATE_VERIFIED_COUNT
CORPUS_SHA256="$({
  "${PY}" -I - "${CORPUS}" <<'PY'
from hashlib import sha256
from pathlib import Path
import sys

print(sha256(Path(sys.argv[1]).read_bytes()).hexdigest())
PY
})"
CORPUS_MANIFEST_TSV="$({
  "${PY}" -I "${SFI_ROOT}/scripts/make_context_corpus.py" \
    --model "${MODEL}" --segments "${BS}" \
    --tokens-per-segment-by-request "${REQUEST_CONTEXT_TOKENS}" \
    --validate "${CORPUS}" --format manifest-tsv
})" || {
  echo "FAIL: cached corpus manifest is not release-semantic grade" >&2
  exit 78
}
IFS=$'\t' read -r \
  SFI_RUNNER_CORPUS_MANIFEST_PATH \
  SFI_RUNNER_CORPUS_MANIFEST_SHA256 \
  SFI_RUNNER_CORPUS_MANIFEST_SCHEMA \
  SFI_RUNNER_CORPUS_LAYOUT_MODE \
  SFI_RUNNER_CORPUS_LAYOUT_VALIDATION_CONTRACT \
  SFI_RUNNER_CORPUS_LAYOUT_VERIFIED_COUNT \
  <<< "${CORPUS_MANIFEST_TSV}"
export SFI_RUNNER_CORPUS_PATH="${CORPUS}"
export SFI_RUNNER_CORPUS_SHA256="${CORPUS_SHA256}"
export SFI_RUNNER_CORPUS_MANIFEST_PATH
export SFI_RUNNER_CORPUS_MANIFEST_SHA256
export SFI_RUNNER_CORPUS_MANIFEST_SCHEMA
export SFI_RUNNER_CORPUS_LAYOUT_MODE
export SFI_RUNNER_CORPUS_LAYOUT_VALIDATION_CONTRACT
export SFI_RUNNER_CORPUS_LAYOUT_VERIFIED_COUNT

export SFI_RUNNER_TENSOR_PARALLEL_SIZE="${TP}"
export SFI_RUNNER_TIER="${TIER}"
export SFI_RUNNER_MODE="${MODE}"
export SFI_RUNNER_BATCH_SIZE="${BS}"
export SFI_RUNNER_CONTEXT_TOKENS="${CTX}"
export SFI_RUNNER_REQUEST_CONTEXT_TOKENS="${REQUEST_CONTEXT_TOKENS}"
export SFI_RUNNER_KV_CACHE_MEMORY_BYTES="${KVB}"
export SFI_RUNNER_MAX_MODEL_LEN="${MML}"
export SFI_RUNNER_MAX_NEW_TOKENS="${MAX_NEW}"
export SFI_RUNNER_REQUEST_MAX_NEW_TOKENS="${REQUEST_MAX_NEW_TOKENS}"
export SFI_RUNNER_MAX_NUM_SEQS="${MAX_NUM_SEQS}"
export SFI_RUNNER_MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS}"
export SFI_RUNNER_CHUNKED_PREFILL="${CHUNKED_PREFILL}"
export SFI_RUNNER_MAX_SEQ_LEN_TO_CAPTURE="${MAX_SEQ_LEN_TO_CAPTURE}"
export SFI_RUNNER_REFRESH_INTERVAL="${REFRESH_INTERVAL}"
export SFI_RUNNER_COMPACT_BLOCKS_PER_SLOT="${BLOCKS}"
export SFI_RUNNER_COMPACT_DUAL_GEN="${VLLM_SPARSE_COMPACT_DUAL_GEN:-1}"

# expandable_segments is a shared-GPU nicety (fragmentation/mem-race), but its
# cuMemMap-backed memory has NO cudaIpcGetMemHandle support. vLLM custom
# all-reduce registers every AR buffer inside the CUDA graph via IPC handles at
# capture end (custom_all_reduce.py register_graph_buffers) -> with expandable
# ON, TP>1 + custom AR + FULL cudagraph crashes with "invalid argument".
# Default expandable OFF for TP>1 so custom AR can stay ON.
if [ "${TP}" -gt 1 ]; then
  export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-}"
else
  export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
fi
export SFI_RUNNER_PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-}"
export SFI_RUNNER_PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-}"
export SFI_RUNNER_CUSTOM_AR_DISABLED="${VLLM_SPARSE_FORCE_DISABLE_CUSTOM_AR:-0}"
export VLLM_KV_CACHE_MEMORY_BYTES="${KVB}"   # pin the KV pool: SFI runtime state
                                             # lives OUTSIDE vLLM's util budget

echo "==> ${MODE} ${TIER}: bs=${BS} ctx_max=${CTX} request_ctx=${REQUEST_CONTEXT_TOKENS} kv_pool=$((KVB/1024/1024/1024))GiB mml=${MML} max_new=${MAX_NEW} request_max_new=${REQUEST_MAX_NEW_TOKENS}"
echo "==> attention: arch=${CUDA_ARCH} capabilities=${CUDA_CAPABILITIES} kernel=${ATTENTION_KERNEL} backend=${ATTENTION_BACKEND} version=${FLASH_ATTN_VERSION} preflight=${ATTENTION_PREFLIGHT_DETAIL}"
if [[ "${CUDA_ARCH}" != "sm80" ]]; then
  echo "==> note: ${TIER} defaults are A100-derived compatibility shapes, not tuned ${CUDA_ARCH} settings"
fi
echo "==> infra: kv_preflight=${SFI_RUNNER_KV_PREFLIGHT_STATUS} gpu_lock=${SFI_RUNNER_GPU_LOCK_MODE} attention_preflight=${SFI_RUNNER_ATTENTION_PREFLIGHT_STATUS} corpus=${SFI_RUNNER_CORPUS_TOKEN_STATUS} chat_template=on overhead=${SFI_RUNNER_CHAT_TEMPLATE_OVERHEAD_MIN}..${SFI_RUNNER_CHAT_TEMPLATE_OVERHEAD_MAX} reserve=${CHAT_TEMPLATE_RESERVE_TOKENS} selector_cache=${SELECTOR_CACHE_ROOT}"
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
# Preserve the harness return code for postflight. The checker accepts only a
# fully green harness; no diagnostic-tier failure is promoted to success.
set +e
VLLM_FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION}" \
VLLM_SPARSE_FA3_UPSTREAM_ROOT="${FA_ROOT}" \
PYTHONPATH="${FA_ROOT}:${SFI_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PY}" -m "${BENCHMARK_MODULE}" \
  --mode "${MODE}" --producer-mode full-open-gt1 \
  --backend "${ATTENTION_BACKEND}" --full-cuda-graph --warmup 1 \
  --prompt "${CORPUS}" --split-context-prompts \
  --batch-size "${BS}" --max-new-tokens "${MAX_NEW}" --max-model-len "${MML}" \
  --request-context-tokens "${REQUEST_CONTEXT_TOKENS}" \
  --request-max-new-tokens "${REQUEST_MAX_NEW_TOKENS}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --kv-cache-memory-bytes "${KVB}" \
  --chunked-prefill "${CHUNKED_PREFILL}" \
  --max-seq-len-to-capture "${MAX_SEQ_LEN_TO_CAPTURE}" \
  --scheduling-mode "${SCHEDULING_MODE}" \
  --refresh-interval "${REFRESH_INTERVAL}" --prefill-last-n 2 \
  --alpha-k-head "${K_HEAD}" \
  --max-live-sparse-slots "${BS}" --compact-blocks-per-slot "${BLOCKS}" \
  --gpu-mem-util "${UTIL:-0.9}" \
  --model "${MODEL}" --python "${PY}" --fa3-upstream-root "${FA_ROOT}" \
  --cuda-visible-devices "${GPU}" --timeout-s 3600 ${TP_ARGS[@]+"${TP_ARGS[@]}"} \
  ${VERDICT_ARGS[@]+"${VERDICT_ARGS[@]}"} \
  ${REFERENCE_ARGS[@]+"${REFERENCE_ARGS[@]}"} --outputs-include-text \
  --chat-template \
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
  --expected-tier "${TIER}" \
  --harness-returncode "${HARNESS_RC}" \
  --run-started-ns "${RUN_STARTED_NS}" \
  --run-nonce "${RUN_NONCE}" \
  --selector-cache-root "${SELECTOR_CACHE_ROOT}" \
  --expected-cuda-arch "${CUDA_ARCH}" \
  --expected-attention-kernel "${ATTENTION_KERNEL}" \
  --expected-backend "${ATTENTION_BACKEND}" \
  --expected-flash-attn-version "${FLASH_ATTN_VERSION}" \
  ${PAIR_POSTFLIGHT_ARGS[@]+"${PAIR_POSTFLIGHT_ARGS[@]}"} \
  ${EXPECTED_GIT_ARGS[@]+"${EXPECTED_GIT_ARGS[@]}"} \
  ${EXPECTED_MODEL_CONFIG_ARGS[@]+"${EXPECTED_MODEL_CONFIG_ARGS[@]}"}
