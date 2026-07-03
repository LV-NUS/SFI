#!/usr/bin/env bash
# =============================================================================
# Launch an OpenAI-compatible vLLM server with SFI sparse decoding enabled
# (FA3 CUDA kernel path, full CUDA graph). Used for online serving and for
# LongBench evaluation.
#
# Usage:
#   bash scripts/serve_sparse.sh <GPU_ID> <MODEL_PATH> [PORT]
#
# Tunables via env (defaults in brackets):
#   K_HEAD    [4096]  tokens selected per KV head. 4096 = quality/eval setting;
#                     1536-2048 = throughput setting.
#   SLOTS     [8]     max concurrent sparse requests (size to your batch needs)
#   BLOCKS    [auto]  compact blocks per slot (16 tokens each); must satisfy
#                     BLOCKS >= K_HEAD/16, so it defaults to
#                     max(144, K_HEAD/16) and only needs overriding to trade
#                     resident-KV footprint (SLOTS x BLOCKS x 16 tokens)
#   REFRESH_INTERVAL [96]   dense-refresh cadence in decode steps
#   MML       [32768] --max-model-len (prompt+generation budget per request)
#   KVB       []      VLLM_KV_CACHE_MEMORY_BYTES; pin the KV pool on shared
#                     GPUs or big models (SFI state lives outside vLLM's
#                     memory-util budget!). Empty = vLLM auto-profiling.
#   GPU_MEM_UTIL [0.9]
#
# Evidence that the sparse path is live:
#   1. patch install:  VLLM_SPARSE_SITE_LOG=1 writes /tmp/vllm_sparse_site.log
#                      -> "patch installed successfully, controller=True"
#   2. kernel routing: set VLLM_SPARSE_FA3_ROUTE_TRACE_LOG=<file>; look for
#                      route=mixed_page_attn_varlen_func, page_resolver_kind=4
# =============================================================================
set -euo pipefail

GPU="${1:?usage: serve_sparse.sh <GPU_ID> <MODEL_PATH> [PORT]}"
MODEL="${2:?usage: serve_sparse.sh <GPU_ID> <MODEL_PATH> [PORT]}"
PORT="${3:-8000}"

SFI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FA_ROOT="${VLLM_SPARSE_FA3_UPSTREAM_ROOT:-${SFI_ROOT}/third_party_upstreams/vllm-project-flash-attention}"

K_HEAD="${K_HEAD:-4096}"
SLOTS="${SLOTS:-8}"
BLOCKS_MIN=$(( (K_HEAD + 15) / 16 ))
BLOCKS="${BLOCKS:-$(( BLOCKS_MIN > 144 ? BLOCKS_MIN : 144 ))}"
if (( BLOCKS < BLOCKS_MIN )); then
  echo "ERROR: BLOCKS=${BLOCKS} cannot hold K_HEAD=${K_HEAD} selections (need >= ${BLOCKS_MIN})" >&2
  exit 2
fi
REFRESH_INTERVAL="${REFRESH_INTERVAL:-96}"
MML="${MML:-32768}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"

# --- SFI runtime injection (sitecustomize.py picks this up in every worker) --
export PYTHONPATH="${SFI_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export VLLM_SPARSE_FA3_UPSTREAM_ROOT="${FA_ROOT}"
export VLLM_ATTENTION_BACKEND="FLASH_ATTN"
export VLLM_FLASH_ATTN_VERSION="${VLLM_FLASH_ATTN_VERSION:-3}"   # 4 on SM100
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export VLLM_SPARSE_ASYNC_REFRESH=1
export VLLM_SPARSE_ONE_SHOT_ASYNC_BOOTSTRAP=1
export VLLM_SPARSE_FULL_CUDAGRAPH_REPLAY_REFRESH_BATCHED_FLUSH=1
export VLLM_SPARSE_FULL_CUDAGRAPH_REPLAY_REFRESH_DEFER_TO_DEADLINE=1
export VLLM_SPARSE_REFRESH_ENQUEUE_STAGGER=1
export VLLM_SPARSE_SITE_LOG="${VLLM_SPARSE_SITE_LOG:-1}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${SFI_ROOT}/tmp/torch_extensions/serve}"
# NOTE: expandable_segments memory has no CUDA IPC handle support; if you pass
# --tensor-parallel-size >1 with custom all-reduce enabled, FULL cudagraph
# capture crashes at register_graph_buffers ("invalid argument", exp4 root
# cause). For TP>1 serving: PYTORCH_ALLOC_CONF= ./serve_sparse.sh ...
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
if [[ -n "${KVB:-}" ]]; then
  export VLLM_KV_CACHE_MEMORY_BYTES="${KVB}"
fi

export VLLM_SPARSE_CONTROLLER_JSON=$(cat <<JSON
{
  "enabled": true,
  "attn_mode": "compact_recent",
  "compact_page_residency_enabled": true,
  "max_live_sparse_slots": ${SLOTS},
  "compact_blocks_per_slot": ${BLOCKS},
  "k_min": 32,
  "k_max": null,
  "sink": 4,
  "recent": 256,
  "refresh_interval": ${REFRESH_INTERVAL},
  "refresh_coalesce_window": 0,
  "alpha_fair": {"k_head": ${K_HEAD}},
  "prefill_last_n_query": 2,
  "one_shot_bootstrap_only": true,
  "continuous_producer_enabled": true,
  "trigger": {
    "refresh_interval": ${REFRESH_INTERVAL},
    "enable_sentence_triggers": true,
    "min_refresh_gap": 16,
    "sentence_cooldown": 2
  }
}
JSON
)

echo "==> serving ${MODEL} on :${PORT} (GPU ${GPU}, k_head=${K_HEAD}, slots=${SLOTS})"
CUDA_VISIBLE_DEVICES="${GPU}" vllm serve "${MODEL}" \
  --port "${PORT}" \
  --max-model-len "${MML}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --compilation-config '{"cudagraph_mode":"FULL","cudagraph_capture_sizes":[1,2,4]}' \
  --disable-cascade-attn
