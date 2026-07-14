#!/usr/bin/env bash
# Launch the canonical SFI sparse vLLM server with fail-closed provenance.
set -euo pipefail

die() {
  echo "FAIL: $*" >&2
  exit 64
}

require_uint() {
  local name="$1"
  local value="$2"
  [[ "${value}" =~ ^[0-9]+$ ]] || die "${name} must be an integer: ${value}"
}

GPU_DEVICES="${1:?usage: serve_sparse.sh <GPU_IDS> <MODEL_PATH> [PORT]}"
MODEL="${2:?usage: serve_sparse.sh <GPU_IDS> <MODEL_PATH> [PORT]}"
PORT="${3:-8000}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFI_ROOT="$(dirname "${SCRIPT_DIR}")"

PYTHON="${PYTHON:?set PYTHON to the absolute vLLM environment interpreter}"
[[ "${PYTHON}" = /* ]] || die "PYTHON must be an absolute path: ${PYTHON}"
PYTHON="$(realpath -e -- "${PYTHON}")"
[[ "${PYTHON}" = /* && -x "${PYTHON}" ]] || die "PYTHON must be an absolute executable path"

# Bind locally by default.  Hostnames other than localhost are rejected to
# avoid DNS-dependent launch identities; a normalized IPv4 literal is stable
# in the manifest and in the live-command proof.
HOST="${HOST-127.0.0.1}"
HOST_IS_LOOPBACK=0
if [[ "${HOST}" == "localhost" ]]; then
  HOST="127.0.0.1"
  HOST_IS_LOOPBACK=1
elif [[ "${HOST}" =~ ^([0-9]{1,3})\.([0-9]{1,3})\.([0-9]{1,3})\.([0-9]{1,3})$ ]]; then
  HOST_OCTETS=("${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}" "${BASH_REMATCH[3]}" "${BASH_REMATCH[4]}")
  for octet in "${HOST_OCTETS[@]}"; do
    (( 10#${octet} <= 255 )) || die "HOST has an invalid IPv4 octet: ${HOST}"
  done
  HOST_FIRST_OCTET=$((10#${HOST_OCTETS[0]}))
  printf -v HOST '%d.%d.%d.%d' \
    "${HOST_FIRST_OCTET}" "$((10#${HOST_OCTETS[1]}))" \
    "$((10#${HOST_OCTETS[2]}))" "$((10#${HOST_OCTETS[3]}))"
  if (( HOST_FIRST_OCTET == 127 )); then
    HOST_IS_LOOPBACK=1
  fi
else
  die "HOST must be localhost or an IPv4 literal: ${HOST}"
fi

DEFAULT_API_KEY="token-abc123"
if [[ -n "${API_KEY+x}" ]]; then
  [[ -n "${API_KEY}" && "${API_KEY}" =~ ^[^[:space:]]+$ ]] || \
    die "explicit API_KEY must be nonempty and contain no whitespace"
  API_KEY_VALUE="${API_KEY}"
  API_KEY_SOURCE="explicit"
else
  (( HOST_IS_LOOPBACK == 1 )) || \
    die "non-loopback HOST=${HOST} requires an explicit non-default API_KEY"
  API_KEY_VALUE="${DEFAULT_API_KEY}"
  API_KEY_SOURCE="default-loopback"
fi
if (( HOST_IS_LOOPBACK == 0 )) && [[ "${API_KEY_VALUE}" == "${DEFAULT_API_KEY}" ]]; then
  die "non-loopback HOST=${HOST} rejects the default API_KEY"
fi
MODEL="$(realpath -e -- "${MODEL}")" || die "MODEL_PATH does not exist: ${MODEL}"
SERVED_MODEL_ID="${SERVED_MODEL_ID-${MODEL}}"
[[ -n "${SERVED_MODEL_ID}" && "${SERVED_MODEL_ID}" != *$'\n'* && "${SERVED_MODEL_ID}" != *$'\r'* ]] || \
  die "SERVED_MODEL_ID must be nonempty and single-line"

[[ "${GPU_DEVICES}" =~ ^[0-9]+(,[0-9]+)*$ ]] || \
  die "GPU_IDS must be a comma-separated numeric list: ${GPU_DEVICES}"
IFS=',' read -r -a GPU_IDS <<< "${GPU_DEVICES}"
declare -A GPU_ID_SEEN=()
GPU_IDS_CANONICAL=()
for raw_gpu_id in "${GPU_IDS[@]}"; do
  gpu_id="${raw_gpu_id}"
  while [[ "${#gpu_id}" -gt 1 && "${gpu_id:0:1}" == "0" ]]; do
    gpu_id="${gpu_id:1}"
  done
  [[ -z "${GPU_ID_SEEN[${gpu_id}]:-}" ]] || \
    die "GPU_IDS contains a duplicate physical device: ${gpu_id}"
  GPU_ID_SEEN["${gpu_id}"]=1
  GPU_IDS_CANONICAL+=("${gpu_id}")
done
GPU_IDS=("${GPU_IDS_CANONICAL[@]}")
printf -v GPU_DEVICES '%s,' "${GPU_IDS[@]}"
GPU_DEVICES="${GPU_DEVICES%,}"
TP_SIZE="${TP_SIZE:-${#GPU_IDS[@]}}"
require_uint "TP_SIZE" "${TP_SIZE}"
TP_SIZE=$((10#${TP_SIZE}))
(( TP_SIZE > 0 )) || die "TP_SIZE must be positive"
(( TP_SIZE == ${#GPU_IDS[@]} )) || \
  die "TP_SIZE=${TP_SIZE} must match visible GPU count=${#GPU_IDS[@]}"

# Enumerate every user-selected physical device through the exact interpreter
# that will launch vLLM.  CUDA_VISIBLE_DEVICES remaps them to logical ranks;
# the count check prevents a missing/invalid token from silently shrinking TP.
REQUESTED_CUDA_ARCH="${SFI_CUDA_ARCH:-auto}"
case "${REQUESTED_CUDA_ARCH}" in
  auto|sm80|sm90|sm100) ;;
  *)
    die "SFI_CUDA_ARCH must be auto, sm80, sm90 or sm100: ${REQUESTED_CUDA_ARCH}"
    ;;
esac
CUDA_CAPABILITIES="$({
  CUDA_VISIBLE_DEVICES="${GPU_DEVICES}" "${PYTHON}" -I -c '
import sys
import torch

expected = int(sys.argv[1])
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable to the selected PYTHON")
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
} 2>&1)" || die "CUDA capability detection failed: ${CUDA_CAPABILITIES}"

IFS=',' read -r -a CUDA_CAPABILITY_LIST <<< "${CUDA_CAPABILITIES}"
(( ${#CUDA_CAPABILITY_LIST[@]} == ${#GPU_IDS[@]} )) || \
  die "CUDA capability detection returned ${#CUDA_CAPABILITY_LIST[@]} ranks for ${#GPU_IDS[@]} selected GPUs: ${CUDA_CAPABILITIES}"
DETECTED_CUDA_ARCH=""
for logical_rank in "${!GPU_IDS[@]}"; do
  capability="${CUDA_CAPABILITY_LIST[logical_rank]}"
  case "${capability}" in
    8.0) rank_arch="sm80" ;;
    9.0) rank_arch="sm90" ;;
    10.0) rank_arch="sm100" ;;
    *)
      die "selected GPU ${GPU_IDS[logical_rank]} (rank ${logical_rank}) has unsupported compute capability ${capability}; expected exact 8.0, 9.0 or 10.0"
      ;;
  esac
  if [[ -z "${DETECTED_CUDA_ARCH}" ]]; then
    DETECTED_CUDA_ARCH="${rank_arch}"
  elif [[ "${rank_arch}" != "${DETECTED_CUDA_ARCH}" ]]; then
    die "selected GPUs must be architecture-homogeneous across TP ranks; capabilities=${CUDA_CAPABILITIES}"
  fi
done
if [[ "${REQUESTED_CUDA_ARCH}" != "auto" && "${REQUESTED_CUDA_ARCH}" != "${DETECTED_CUDA_ARCH}" ]]; then
  die "SFI_CUDA_ARCH=${REQUESTED_CUDA_ARCH} does not match detected selected-GPU architecture ${DETECTED_CUDA_ARCH} (capabilities=${CUDA_CAPABILITIES})"
fi
CUDA_ARCH="${DETECTED_CUDA_ARCH}"
CUDA_ARCH_SOURCE="detected"
if [[ "${REQUESTED_CUDA_ARCH}" != "auto" ]]; then
  CUDA_ARCH_SOURCE="explicit"
fi

case "${CUDA_ARCH}" in
  sm80|sm90)
    ATTENTION_KERNEL="fa3-native"
    FLASH_ATTN_VERSION=3
    ;;
  sm100)
    ATTENTION_KERNEL="fa4-cute"
    FLASH_ATTN_VERSION=4
    ;;
esac

K_HEAD="${K_HEAD:-4096}"
SLOTS="${SLOTS:-8}"
SINK="${SINK:-4}"
RECENT="${RECENT:-256}"
REFRESH_INTERVAL="${REFRESH_INTERVAL:-96}"
MML="${MML:-32768}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
for pair in \
  "PORT:${PORT}" \
  "K_HEAD:${K_HEAD}" \
  "SLOTS:${SLOTS}" \
  "SINK:${SINK}" \
  "RECENT:${RECENT}" \
  "REFRESH_INTERVAL:${REFRESH_INTERVAL}" \
  "MML:${MML}"; do
  require_uint "${pair%%:*}" "${pair#*:}"
done
(( PORT > 0 && PORT < 65536 )) || die "PORT is out of range: ${PORT}"
(( K_HEAD > 0 && SLOTS > 0 && REFRESH_INTERVAL > 0 && MML > 0 )) || \
  die "K_HEAD, SLOTS, REFRESH_INTERVAL and MML must be positive"
[[ "${GPU_MEM_UTIL}" =~ ^(0\.[0-9]*[1-9][0-9]*|1(\.0+)?)$ ]] || \
  die "GPU_MEM_UTIL must be in (0, 1]: ${GPU_MEM_UTIL}"

BLOCKS_ALIGNED_TOKENS=$(( ((SINK + K_HEAD + 111) / 112) * 112 ))
BLOCKS_MIN=$(( (BLOCKS_ALIGNED_TOKENS + 15) / 16 ))
BLOCKS="${BLOCKS:-$(( BLOCKS_MIN > 112 ? BLOCKS_MIN : 112 ))}"
require_uint "BLOCKS" "${BLOCKS}"
(( BLOCKS >= BLOCKS_MIN )) || \
  die "BLOCKS=${BLOCKS} cannot hold sink=${SINK} + K_HEAD=${K_HEAD}; need >=${BLOCKS_MIN}"

CAPTURE_SIZES=(1)
capture_size=2
while (( capture_size < SLOTS )); do
  CAPTURE_SIZES+=("${capture_size}")
  capture_size=$(( capture_size * 2 ))
done
if (( SLOTS > 1 )); then
  CAPTURE_SIZES+=("${SLOTS}")
fi
CAPTURE_SIZES_CSV="$(IFS=,; echo "${CAPTURE_SIZES[*]}")"
CAPTURE_SIZES_JSON="[${CAPTURE_SIZES_CSV}]"

RUN_NONCE="${SFI_RUN_NONCE:-$(date +%Y%m%dT%H%M%S)-$$}"
[[ "${RUN_NONCE}" =~ ^[A-Za-z0-9._-]+$ ]] || \
  die "SFI_RUN_NONCE may contain only letters, digits, dot, underscore and dash"
RUN_ROOT="$(realpath -ms -- "${SFI_SERVE_RUN_ROOT:-${SFI_ROOT}/tmp/serve_runs}")"
ARTIFACT_DIR="$(realpath -ms -- "${SFI_SERVE_ARTIFACT_DIR:-${RUN_ROOT}/${RUN_NONCE}}")"
[[ ! -e "${ARTIFACT_DIR}" ]] || die "artifact directory already exists: ${ARTIFACT_DIR}"
mkdir -p "${RUN_ROOT}"
mkdir "${ARTIFACT_DIR}"
RUN_SINCE="$(date +%s)"

SITE_LOG="${ARTIFACT_DIR}/site.log"
REFRESH_PROFILE_LOG="${ARTIFACT_DIR}/refresh_profile.log"
ROUTE_TRACE_LOG="${ARTIFACT_DIR}/route.jsonl"
STEP_TRACE_LOG="${ARTIFACT_DIR}/step.jsonl"
ROUTE_COUNTER_MMAP="${ARTIFACT_DIR}/route_counter.bin"
MANIFEST="${ARTIFACT_DIR}/serve_manifest.json"
MANIFEST_POINTER="${RUN_ROOT}/port-${PORT}.manifest"

FA_ROOT="$(realpath -ms -- "${VLLM_SPARSE_FA3_UPSTREAM_ROOT:-${SFI_ROOT}/third_party_upstreams/vllm-project-flash-attention}")"
FA3_INTERFACE="${FA_ROOT}/vllm_flash_attn/flash_attn_interface.py"
[[ -f "${FA3_INTERFACE}" ]] || die "vendored FA3 interface is missing: ${FA3_INTERFACE}"
if [[ "${ATTENTION_KERNEL}" == "fa3-native" ]]; then
  compgen -G "${FA_ROOT}/vllm_flash_attn/_vllm_fa3_C*.so" >/dev/null || \
    die "vendored FA3 shared object is missing under ${FA_ROOT}/vllm_flash_attn"
else
  FA4_CUTE_INTERFACE="${FA_ROOT}/flash_attn/cute/interface.py"
  FA4_JIT_PREFLIGHT="${SFI_ROOT}/scripts/probe_fa4_cute_sm100_compact_recent_fake_compile.py"
  [[ -f "${FA4_CUTE_INTERFACE}" ]] || \
    die "vendored FA4 CuTe interface is missing: ${FA4_CUTE_INTERFACE}"
  [[ -f "${FA4_JIT_PREFLIGHT}" ]] || \
    die "FA4 CuTe JIT preflight is missing: ${FA4_JIT_PREFLIGHT}"
fi

SELECTOR_CACHE_ABI_KEY="$(
  env -u PYTHONPATH "${PYTHON}" -I "${SFI_ROOT}/utils/selector_cache_identity.py"
)"
[[ "${SELECTOR_CACHE_ABI_KEY}" =~ ^[0-9a-f]{16}$ ]] || \
  die "invalid selector cache ABI identity: ${SELECTOR_CACHE_ABI_KEY}"
SELECTOR_SEMANTIC="$(
  env -u PYTHONPATH "${PYTHON}" -I - \
    "${SFI_ROOT}/utils/selector_pipeline_identity.py" <<'PY'
import importlib.util
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("sfi_selector_identity", path)
if spec is None or spec.loader is None:
    raise SystemExit(f"cannot load selector identity: {path}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print(module.SELECTOR_PIPELINE_SEMANTIC_VERSION)
PY
)"
[[ "${SELECTOR_SEMANTIC}" =~ ^[0-9]+$ ]] || \
  die "invalid selector semantic identity: ${SELECTOR_SEMANTIC}"
EXTENSIONS_BASE="$(realpath -ms -- "${SFI_TORCH_EXTENSIONS_BASE:-${SFI_ROOT}/tmp/torch_extensions/serve}")"
export TORCH_EXTENSIONS_DIR="${EXTENSIONS_BASE}/abi-${SELECTOR_CACHE_ABI_KEY}/selector-v${SELECTOR_SEMANTIC}"
mkdir -p "${TORCH_EXTENSIONS_DIR}"

export PYTHONPATH="${SFI_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export VLLM_ATTENTION_BACKEND="FLASH_ATTN_VLLM_V1"
export VLLM_FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION}"
export VLLM_SPARSE_FA3_UPSTREAM_ROOT="${FA_ROOT}"
export SFI_CUDA_ARCH="${CUDA_ARCH}"
export SFI_ATTENTION_KERNEL="${ATTENTION_KERNEL}"
if [[ "${ATTENTION_KERNEL}" == "fa4-cute" ]]; then
  export FLASH_ATTENTION_ARCH="sm_100a"
  export CUTE_DSL_ARCH="sm_100a"
fi
export VLLM_TENSOR_PARALLEL_SIZE="${TP_SIZE}"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export VLLM_SPARSE_ASYNC_REFRESH=1
export VLLM_SPARSE_ONE_SHOT_ASYNC_BOOTSTRAP=1
export VLLM_SPARSE_FULL_CUDAGRAPH_REPLAY_REFRESH_BATCHED_FLUSH=1
export VLLM_SPARSE_FULL_CUDAGRAPH_REPLAY_REFRESH_DEFER_TO_DEADLINE=1
export VLLM_SPARSE_REFRESH_ENQUEUE_STAGGER=1
export VLLM_SPARSE_SITE_LOG=1
export VLLM_SPARSE_SITE_LOG_PATH="${SITE_LOG}"
export VLLM_SPARSE_REFRESH_PROFILE=1
export VLLM_SPARSE_REFRESH_PROFILE_LOG="${REFRESH_PROFILE_LOG}"
export VLLM_SPARSE_FA3_ROUTE_TRACE_LOG="${ROUTE_TRACE_LOG}"
export VLLM_SPARSE_FA3_STEP_TRACE_LOG="${STEP_TRACE_LOG}"
export VLLM_SPARSE_FA3_ROUTE_COUNTER_MMAP="${ROUTE_COUNTER_MMAP}"
export VLLM_SPARSE_FA3_ROUTE_COUNTER_SLOTS="${TP_SIZE}"
export VLLM_NO_USAGE_REPORT=1
export CPUINFO_NO_DMI=1
export SFI_RUN_NONCE="${RUN_NONCE}"
export SFI_SERVE_RUN_DIR="${ARTIFACT_DIR}"
export SFI_SERVE_MANIFEST="${MANIFEST}"
export SFI_SERVE_HOST="${HOST}"
export SFI_SERVE_API_KEY_SOURCE="${API_KEY_SOURCE}"
export SFI_SERVED_MODEL_ID="${SERVED_MODEL_ID}"
if (( TP_SIZE > 1 )); then
  export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-}"
else
  export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
fi
if [[ -n "${KVB:-}" ]]; then
  require_uint "KVB" "${KVB}"
  export VLLM_KV_CACHE_MEMORY_BYTES="${KVB}"
fi

export VLLM_SPARSE_CONTROLLER_JSON="$(cat <<JSON
{
  "enabled": true,
  "attn_mode": "compact_recent",
  "compact_page_residency_enabled": true,
  "max_live_sparse_slots": ${SLOTS},
  "compact_blocks_per_slot": ${BLOCKS},
  "k_min": 32,
  "k_max": null,
  "sink": ${SINK},
  "recent": ${RECENT},
  "refresh_interval": ${REFRESH_INTERVAL},
  "refresh_coalesce_window": 0,
  "alpha_fair": {"k_head": ${K_HEAD}},
  "prefill_last_n_query": 2,
  "one_shot_bootstrap_only": true,
  "continuous_producer_enabled": true,
  "trigger": {
    "refresh_interval": ${REFRESH_INTERVAL},
    "enable_sentence_triggers": true,
    "min_refresh_gap": 24,
    "sentence_cooldown": 2
  }
}
JSON
)"

# Validate the selected FlashAttention family and precompile the selectors once
# in the exact interpreter/cache before spawning TP workers.
CUDA_VISIBLE_DEVICES="${GPU_DEVICES}" \
EXPECTED_SELECTOR_SEMANTIC="${SELECTOR_SEMANTIC}" \
EXPECTED_FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION}" \
EXPECTED_CUDA_ARCH="${CUDA_ARCH}" \
EXPECTED_ATTENTION_KERNEL="${ATTENTION_KERNEL}" "${PYTHON}" - <<'PY'
import importlib.util
import os

if importlib.util.find_spec("vllm.entrypoints.openai.api_server") is None:
    raise SystemExit("vLLM OpenAI server module is unavailable in selected PYTHON")
from utils.bounds_kernel_ext import _require_ext as require_bounds
from utils.selector_pipeline_ext import _require_ext as require_pipeline
from patches.fa3_native.install import (
    load_vendored_flash_attn_bridge,
    resolve_vendored_flash_attn_version,
)

bridge = load_vendored_flash_attn_bridge()
expected_fa = int(os.environ["EXPECTED_FLASH_ATTN_VERSION"])
resolved_fa = resolve_vendored_flash_attn_version(bridge)
if resolved_fa != expected_fa:
    raise SystemExit(
        "FlashAttention preflight failed: "
        f"arch={os.environ['EXPECTED_CUDA_ARCH']}, "
        f"kernel={os.environ['EXPECTED_ATTENTION_KERNEL']}, "
        f"expected_version={expected_fa}, resolved_version={resolved_fa}"
    )

require_bounds()
pipeline = require_pipeline()
observed = int(pipeline.selector_pipeline_semantic_version())
expected = int(os.environ["EXPECTED_SELECTOR_SEMANTIC"])
if observed != expected:
    raise SystemExit(
        f"selector semantic mismatch: expected={expected}, observed={observed}"
    )
print(f"selector preflight passed: semantic={observed}")
print(
    "FlashAttention preflight passed: "
    f"arch={os.environ['EXPECTED_CUDA_ARCH']} "
    f"kernel={os.environ['EXPECTED_ATTENTION_KERNEL']} version={resolved_fa}"
)
PY

if [[ "${ATTENTION_KERNEL}" == "fa4-cute" ]]; then
  FA4_PREFLIGHT_LOG="${ARTIFACT_DIR}/fa4_cute_jit_preflight.log"
  CUDA_VISIBLE_DEVICES="${GPU_DEVICES}" \
  FLASH_ATTENTION_ARCH="sm_100a" CUTE_DSL_ARCH="sm_100a" \
    "${PYTHON}" -I "${FA4_JIT_PREFLIGHT}" 2>&1 | tee "${FA4_PREFLIGHT_LOG}"
  FA4_PREFLIGHT_OUTPUT="$(<"${FA4_PREFLIGHT_LOG}")"
  for marker in \
    sm100_cute_mixed_page_no_selected_fake_compile_probe_ok \
    sm100_cute_mixed_page_selected_brow_fake_compile_probe_ok \
    sm100_cute_mixed_page_selected_bhrow_capture_fake_compile_probe_ok \
    sm100_cute_mixed_page_rrp_fake_compile_probe_ok; do
    [[ "${FA4_PREFLIGHT_OUTPUT}" == *"${marker}"* ]] || \
      die "FA4 CuTe JIT preflight did not emit ${marker}"
  done
fi

# Refuse to kill an unrelated listener; ownership stays with this exec PID.
"${PYTHON}" -I - "${HOST}" "${PORT}" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.bind((host, port))
except OSError as exc:
    raise SystemExit(
        f"address {host}:{port} is unavailable; refusing global cleanup: {exc}"
    )
finally:
    sock.close()
PY

"${PYTHON}" -I - \
  "${MANIFEST}" "${MANIFEST_POINTER}" "$$" "${HOST}" "${PORT}" \
  "${API_KEY_SOURCE}" "${MODEL}" "${SERVED_MODEL_ID}" "${MML}" "${RUN_NONCE}" \
  "${RUN_SINCE}" "${PYTHON}" "${ARTIFACT_DIR}" "${SITE_LOG}" \
  "${REFRESH_PROFILE_LOG}" "${ROUTE_TRACE_LOG}" "${STEP_TRACE_LOG}" \
  "${ROUTE_COUNTER_MMAP}" \
  "${TP_SIZE}" "${SLOTS}" "${CAPTURE_SIZES_JSON}" "${SELECTOR_SEMANTIC}" \
  "${TORCH_EXTENSIONS_DIR}" "${GPU_DEVICES}" "${CUDA_ARCH}" "${CUDA_CAPABILITIES}" \
  "${CUDA_ARCH_SOURCE}" "${ATTENTION_KERNEL}" "${FLASH_ATTN_VERSION}" \
  "${FA_ROOT}" <<'PY'
import json
import os
import pathlib
import sys

(
    manifest,
    pointer,
    server_pid,
    host,
    port,
    api_key_source,
    model_path,
    served_model_id,
    max_model_len,
    nonce,
    started_epoch,
    python,
    run_dir,
    site_log,
    profile_log,
    route_trace,
    step_trace,
    route_mmap,
    tp_size,
    slots,
    capture_sizes,
    selector_semantic,
    extensions_dir,
    gpu_devices,
    cuda_arch,
    cuda_capabilities,
    cuda_arch_source,
    attention_kernel,
    flash_attn_version,
    flash_attn_root,
) = sys.argv[1:]
payload = {
    "schema": 4,
    "server_pid": int(server_pid),
    "host": host,
    "port": int(port),
    "api_key_source": api_key_source,
    "model_path": model_path,
    "served_model_id": served_model_id,
    "max_model_len": int(max_model_len),
    "run_nonce": nonce,
    "started_epoch": int(started_epoch),
    "python": os.path.realpath(python),
    "run_dir": run_dir,
    "site_log": site_log,
    "refresh_profile_log": profile_log,
    "route_trace_log": route_trace,
    "step_trace_log": step_trace,
    "route_counter_mmap": route_mmap,
    "tensor_parallel_size": int(tp_size),
    "slots": int(slots),
    "capture_sizes": json.loads(capture_sizes),
    "selector_semantic": int(selector_semantic),
    "torch_extensions_dir": extensions_dir,
    "gpu_devices": gpu_devices,
    "cuda_arch": cuda_arch,
    "cuda_capabilities": cuda_capabilities.split(","),
    "cuda_arch_source": cuda_arch_source,
    "attention_kernel": attention_kernel,
    "flash_attn_root": flash_attn_root,
    "attention_backend": "FLASH_ATTN_VLLM_V1",
    "flash_attn_version": int(flash_attn_version),
}
manifest_path = pathlib.Path(manifest)
manifest_tmp = manifest_path.with_name(f".{manifest_path.name}.tmp-{os.getpid()}")
manifest_tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
os.replace(manifest_tmp, manifest_path)
pointer_path = pathlib.Path(pointer)
pointer_tmp = pointer_path.with_name(f".{pointer_path.name}.tmp-{os.getpid()}")
pointer_tmp.write_text(str(manifest_path) + "\n", encoding="utf-8")
os.replace(pointer_tmp, pointer_path)
PY

echo "==> SFI sparse server"
echo "    python=${PYTHON}"
echo "    gpu=${GPU_DEVICES} tp=${TP_SIZE} slots=max_num_seqs=${SLOTS}"
echo "    bind=${HOST}:${PORT} auth=${API_KEY_SOURCE} served_model=${SERVED_MODEL_ID}"
echo "    arch=${CUDA_ARCH} source=${CUDA_ARCH_SOURCE} kernel=${ATTENTION_KERNEL} version=${FLASH_ATTN_VERSION}"
echo "    capture_sizes=${CAPTURE_SIZES_JSON} selector_semantic=${SELECTOR_SEMANTIC}"
echo "    artifacts=${ARTIFACT_DIR}"
echo "    manifest=${MANIFEST}"

SERVER_ARGS=(
  --model "${MODEL}"
  --served-model-name "${SERVED_MODEL_ID}"
  --host "${HOST}"
  --port "${PORT}"
  --api-key "${API_KEY_VALUE}"
  --trust-remote-code
  --tensor-parallel-size "${TP_SIZE}"
  --dtype bfloat16
  --max-model-len "${MML}"
  --gpu-memory-utilization "${GPU_MEM_UTIL}"
  --max-num-seqs "${SLOTS}"
  --max-num-partial-prefills 1
  --enable-chunked-prefill
  --disable-cascade-attn
  --compilation-config "{\"cudagraph_mode\":\"FULL\",\"cudagraph_capture_sizes\":${CAPTURE_SIZES_JSON}}"
)
if [[ -n "${MAX_BATCHED_TOKENS:-}" ]]; then
  require_uint "MAX_BATCHED_TOKENS" "${MAX_BATCHED_TOKENS}"
  SERVER_ARGS+=(--max-num-batched-tokens "${MAX_BATCHED_TOKENS}")
fi

# Keep an explicitly supplied secret out of the long-lived server environment;
# vLLM receives only the required CLI value already captured in SERVER_ARGS.
unset API_KEY
exec env CUDA_VISIBLE_DEVICES="${GPU_DEVICES}" \
  "${PYTHON}" -m vllm.entrypoints.openai.api_server "${SERVER_ARGS[@]}"
