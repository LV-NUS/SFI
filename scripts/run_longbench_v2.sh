#!/usr/bin/env bash
# Run LongBench v2 only after proving that the current server is sparse-live.
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

(( $# >= 1 && $# <= 3 )) || \
  die "usage: run_longbench_v2.sh <MODEL_NAME> [N_PROC] [PORT]"
MODEL_NAME="$1"
N_PROC="${2:-4}"
PORT="${3:-8000}"
[[ -n "${MODEL_NAME}" && "${MODEL_NAME}" != *$'\n'* && "${MODEL_NAME}" != *$'\r'* ]] || \
  die "MODEL_NAME must be nonempty and single-line"
require_uint "N_PROC" "${N_PROC}"
require_uint "PORT" "${PORT}"
(( N_PROC > 0 )) || die "N_PROC must be positive"
(( PORT > 0 && PORT < 65536 )) || die "PORT is out of range: ${PORT}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFI_ROOT="$(dirname "${SCRIPT_DIR}")"

PYTHON="${PYTHON:?set PYTHON to the absolute server environment interpreter}"
[[ "${PYTHON}" = /* ]] || die "PYTHON must be an absolute path: ${PYTHON}"
PYTHON="$(realpath -e -- "${PYTHON}")"
[[ "${PYTHON}" = /* && -x "${PYTHON}" ]] || die "PYTHON must be an absolute executable path"

LONGBENCH_ROOT="${LONGBENCH_ROOT:?set LONGBENCH_ROOT to an official THUDM/LongBench checkout}"
[[ "${LONGBENCH_ROOT}" = /* ]] || \
  die "LONGBENCH_ROOT must be an absolute path: ${LONGBENCH_ROOT}"
LONGBENCH_ROOT="$(realpath -e -- "${LONGBENCH_ROOT}")"
[[ -d "${LONGBENCH_ROOT}" ]] || die "LONGBENCH_ROOT is not a directory"

LONGBENCH_PYTHON="${LONGBENCH_PYTHON:-${PYTHON}}"
[[ "${LONGBENCH_PYTHON}" = /* ]] || \
  die "LONGBENCH_PYTHON must be an absolute path: ${LONGBENCH_PYTHON}"
LONGBENCH_PYTHON="$(realpath -e -- "${LONGBENCH_PYTHON}")"
[[ -x "${LONGBENCH_PYTHON}" ]] || die "LONGBENCH_PYTHON is not executable"

LONGBENCH_DATASET_REVISION="${LONGBENCH_DATASET_REVISION:?set LONGBENCH_DATASET_REVISION to the immutable 40-hex Hugging Face commit}"
LONGBENCH_DATASET_SHA256="${LONGBENCH_DATASET_SHA256:?set LONGBENCH_DATASET_SHA256 to the canonical 503-row content SHA256}"
[[ "${LONGBENCH_DATASET_REVISION}" =~ ^[0-9a-f]{40}$ ]] || \
  die "LONGBENCH_DATASET_REVISION must be a lowercase 40-hex commit"
[[ "${LONGBENCH_DATASET_SHA256}" =~ ^[0-9a-f]{64}$ ]] || \
  die "LONGBENCH_DATASET_SHA256 must be a lowercase 64-hex digest"
[[ "${HF_DATASETS_OFFLINE:-}" == "1" && "${HF_HUB_OFFLINE:-}" == "1" ]] || \
  die "HF_DATASETS_OFFLINE=1 and HF_HUB_OFFLINE=1 are required; pre-populate the pinned dataset cache before evaluation"

PRED_SCRIPT="${LONGBENCH_ROOT}/pred.py"
RESULT_SCRIPT="${LONGBENCH_ROOT}/result.py"
MODEL_PATH_CONFIG="${LONGBENCH_ROOT}/config/model2path.json"
MODEL_MAXLEN_CONFIG="${LONGBENCH_ROOT}/config/model2maxlen.json"
OFFICIAL_SOURCE_FILES=(
  pred.py
  result.py
  requirements.txt
  config/model2path.json
  config/model2maxlen.json
  prompts/0shot.txt
  prompts/0shot_cot.txt
  prompts/0shot_cot_ans.txt
  prompts/0shot_no_context.txt
  prompts/0shot_rag.txt
)
for relative_path in "${OFFICIAL_SOURCE_FILES[@]}"; do
  [[ -f "${LONGBENCH_ROOT}/${relative_path}" ]] || \
    die "official LongBench file is missing: ${LONGBENCH_ROOT}/${relative_path}"
done

command -v git >/dev/null 2>&1 || die "git is required to identify the official checkout"
LONGBENCH_GIT_HEAD="$(git -C "${LONGBENCH_ROOT}" rev-parse --verify HEAD 2>/dev/null)" || \
  die "LONGBENCH_ROOT is not a Git checkout"
LONGBENCH_GIT_ORIGIN="$(git -C "${LONGBENCH_ROOT}" config --get remote.origin.url 2>/dev/null)" || \
  die "LongBench checkout has no origin URL"
case "${LONGBENCH_GIT_ORIGIN}" in
  https://github.com/THUDM/LongBench|https://github.com/THUDM/LongBench.git|\
  https://github.com/THUDM/LongBench/|git@github.com:THUDM/LongBench|\
  git@github.com:THUDM/LongBench.git|ssh://git@github.com/THUDM/LongBench|\
  ssh://git@github.com/THUDM/LongBench.git) ;;
  *) die "LONGBENCH_ROOT origin is not official THUDM/LongBench: ${LONGBENCH_GIT_ORIGIN}" ;;
esac
verify_official_checkout_unchanged() {
  local current_head relevant_status relative_path
  current_head="$(git -C "${LONGBENCH_ROOT}" rev-parse --verify HEAD 2>/dev/null)" || \
    die "cannot re-read official LongBench HEAD"
  [[ "${current_head}" == "${LONGBENCH_GIT_HEAD}" ]] || \
    die "official LongBench HEAD changed during evaluation: ${LONGBENCH_GIT_HEAD} -> ${current_head}"
  relevant_status="$(
    git -C "${LONGBENCH_ROOT}" status --porcelain=v1 --untracked-files=all -- \
      "${OFFICIAL_SOURCE_FILES[@]}"
  )" || die "cannot inspect relevant LongBench checkout status"
  [[ -z "${relevant_status}" ]] || \
    die "official LongBench evaluator/config/prompt files are dirty; commit the intentional local configuration or use a clean checkout"
  for relative_path in "${OFFICIAL_SOURCE_FILES[@]}"; do
    if ! git -C "${LONGBENCH_ROOT}" show \
      "${LONGBENCH_GIT_HEAD}:${relative_path}" | \
      cmp -s - "${LONGBENCH_ROOT}/${relative_path}"; then
      die "official LongBench file differs from the frozen HEAD: ${relative_path}"
    fi
  done
}
verify_official_checkout_unchanged

RUN_ROOT="$(realpath -ms -- "${SFI_SERVE_RUN_ROOT:-${SFI_ROOT}/tmp/serve_runs}")"
MANIFEST="${SFI_SERVE_MANIFEST:-}"
if [[ -z "${MANIFEST}" ]]; then
  POINTER="${RUN_ROOT}/port-${PORT}.manifest"
  [[ -f "${POINTER}" ]] || \
    die "server manifest pointer is missing: ${POINTER}; start scripts/serve_sparse.sh"
  IFS= read -r MANIFEST < "${POINTER}"
fi
[[ "${MANIFEST}" = /* && -f "${MANIFEST}" ]] || die "invalid server manifest: ${MANIFEST}"

# Bind the manifest to the live exec PID and its exact command/environment.
mapfile -t SERVER_META < <(
  "${PYTHON}" -I - "${MANIFEST}" "${PORT}" "${PYTHON}" "${SFI_ROOT}" <<'PY'
import json
import ipaddress
import os
import pathlib
import sys

manifest_path = pathlib.Path(sys.argv[1]).resolve()
requested_port = int(sys.argv[2])
selected_python = os.path.realpath(sys.argv[3])
repo_root = os.path.realpath(sys.argv[4])
sys.path.insert(0, repo_root)
from benchmarks.sm80_run_pair import classify_speed_child_env_key

data = json.loads(manifest_path.read_text(encoding="utf-8"))
required = {
    "schema",
    "server_pid",
    "host",
    "port",
    "api_key_source",
    "model_path",
    "served_model_id",
    "max_model_len",
    "run_nonce",
    "started_epoch",
    "python",
    "run_dir",
    "site_log",
    "refresh_profile_log",
    "route_trace_log",
    "step_trace_log",
    "route_counter_snapshot",
    "trace_enabled",
    "runtime_proof_enabled",
    "route_counter_rpc_enabled",
    "hot_path_observer_free",
    "tensor_parallel_size",
    "slots",
    "capture_sizes",
    "selector_semantic",
    "torch_extensions_dir",
    "attention_backend",
    "flash_attn_version",
    "gpu_devices",
    "cuda_arch",
    "cuda_capabilities",
    "cuda_arch_source",
    "attention_kernel",
    "flash_attn_root",
}
missing = sorted(required.difference(data))
if missing:
    raise SystemExit(f"server manifest is incomplete: {missing}")
if data["schema"] != 6 or data["port"] != requested_port:
    raise SystemExit("server manifest schema/port mismatch")
try:
    bind_address = ipaddress.IPv4Address(data["host"])
except (ipaddress.AddressValueError, TypeError) as exc:
    raise SystemExit("server manifest host is not a normalized IPv4 literal") from exc
if data["api_key_source"] not in {"default-loopback", "explicit"}:
    raise SystemExit("server manifest has invalid API-key provenance")
for key in (
    "trace_enabled",
    "runtime_proof_enabled",
    "route_counter_rpc_enabled",
    "hot_path_observer_free",
):
    if type(data[key]) is not bool:
        raise SystemExit(f"server manifest {key} must be a boolean")
proof_mode = bool(
    data["trace_enabled"]
    and data["runtime_proof_enabled"]
    and data["route_counter_rpc_enabled"]
    and not data["hot_path_observer_free"]
)
observer_free_mode = bool(
    not data["trace_enabled"]
    and not data["runtime_proof_enabled"]
    and not data["route_counter_rpc_enabled"]
    and data["hot_path_observer_free"]
)
if not (proof_mode or observer_free_mode):
    raise SystemExit(
        "LongBench requires either diagnostic proof mode or the exact "
        "observer-free runtime mode"
    )
if proof_mode and not bind_address.is_loopback:
    raise SystemExit("LongBench runtime-proof RPC is restricted to loopback")
if not isinstance(data["model_path"], str) or not os.path.isabs(data["model_path"]):
    raise SystemExit("server manifest model_path is not absolute")
if not isinstance(data["served_model_id"], str) or not data["served_model_id"]:
    raise SystemExit("server manifest served_model_id is empty")
if int(data["max_model_len"]) <= 0:
    raise SystemExit("server manifest max_model_len must be positive")
if not isinstance(data["run_dir"], str):
    raise SystemExit("server manifest run_dir must be a string")
run_dir = pathlib.Path(data["run_dir"])
if not run_dir.is_absolute() or not run_dir.is_dir():
    raise SystemExit("server manifest run_dir is not an absolute directory")
expected_snapshot = run_dir / "route_counter_snapshot.bin"
if not isinstance(data["route_counter_snapshot"], str) or not os.path.isabs(
    data["route_counter_snapshot"]
):
    raise SystemExit("server route-counter snapshot path is not absolute")
if os.path.realpath(data["route_counter_snapshot"]) != os.path.realpath(
    expected_snapshot
):
    raise SystemExit("server route-counter snapshot escaped its run directory")
if data["attention_backend"] != "FLASH_ATTN_VLLM_V1":
    raise SystemExit("server manifest does not select FLASH_ATTN_VLLM_V1")
supported_kernels = {
    "sm80": ("fa3-native", 3),
    "sm90": ("fa3-native", 3),
    "sm100": ("fa4-cute", 4),
}
expected_kernel = supported_kernels.get(data["cuda_arch"])
observed_kernel = (data["attention_kernel"], data["flash_attn_version"])
if expected_kernel is None or observed_kernel != expected_kernel:
    raise SystemExit(
        "unsupported server architecture/kernel/version tuple: "
        f"{data['cuda_arch']}/{observed_kernel[0]}/v{observed_kernel[1]}"
    )
if data["cuda_arch_source"] not in {"detected", "explicit"}:
    raise SystemExit("server manifest has invalid CUDA architecture provenance")
expected_capability = {"sm80": "8.0", "sm90": "9.0", "sm100": "10.0"}.get(
    data["cuda_arch"]
)
capabilities = data["cuda_capabilities"]
if (
    not isinstance(capabilities, list)
    or len(capabilities) != int(data["tensor_parallel_size"])
    or any(capability != expected_capability for capability in capabilities)
):
    raise SystemExit("server manifest has invalid or heterogeneous CUDA capabilities")
if data["slots"] not in data["capture_sizes"]:
    raise SystemExit("server capture sizes do not include max sparse slots")
if selected_python != os.path.realpath(data["python"]):
    raise SystemExit(
        f"client/server PYTHON mismatch: {selected_python} != {data['python']}"
    )

pid = int(data["server_pid"])
proc = pathlib.Path("/proc") / str(pid)
if not proc.is_dir():
    raise SystemExit(f"manifest server PID is not live: {pid}")
try:
    environ = {
        entry.split("=", 1)[0]: entry.split("=", 1)[1]
        for entry in (proc / "environ").read_bytes().decode(errors="replace").split("\0")
        if "=" in entry
    }
    cmdline = [
        part.decode(errors="replace")
        for part in (proc / "cmdline").read_bytes().split(b"\0")
        if part
    ]
except OSError as exc:
    raise SystemExit(f"cannot inspect server PID {pid}: {exc}") from exc

expected_env = {
    "SFI_RUN_NONCE": str(data["run_nonce"]),
    "SFI_SERVE_RUN_DIR": str(data["run_dir"]),
    "SFI_SERVE_MANIFEST": str(manifest_path),
    "SFI_SERVE_HOST": str(data["host"]),
    "SFI_SERVE_API_KEY_SOURCE": str(data["api_key_source"]),
    "SFI_SERVED_MODEL_ID": str(data["served_model_id"]),
    "VLLM_ATTENTION_BACKEND": "FLASH_ATTN_VLLM_V1",
    "VLLM_FLASH_ATTN_VERSION": str(data["flash_attn_version"]),
    "VLLM_SPARSE_FA3_UPSTREAM_ROOT": str(data["flash_attn_root"]),
    "SFI_CUDA_ARCH": str(data["cuda_arch"]),
    "SFI_ATTENTION_KERNEL": str(data["attention_kernel"]),
    "CUDA_VISIBLE_DEVICES": str(data["gpu_devices"]),
    "VLLM_TENSOR_PARALLEL_SIZE": str(data["tensor_parallel_size"]),
    "VLLM_SPARSE_SITE_LOG": "1",
    "VLLM_SPARSE_SITE_LOG_PATH": str(data["site_log"]),
}
if proof_mode:
    expected_env.update(
        {
            "VLLM_SPARSE_REFRESH_PROFILE_LOG": str(data["refresh_profile_log"]),
            "VLLM_SPARSE_FA3_ROUTE_TRACE_LOG": str(data["route_trace_log"]),
            "VLLM_SPARSE_FA3_STEP_TRACE_LOG": str(data["step_trace_log"]),
            "VLLM_SPARSE_FA3_ROUTE_COUNTER_ENABLED": "1",
            "VLLM_SPARSE_FA3_ROUTE_COUNTER_SLOTS": str(
                data["tensor_parallel_size"]
            ),
            "VLLM_SERVER_DEV_MODE": "1",
        }
    )
for key, expected in expected_env.items():
    if environ.get(key) != expected:
        raise SystemExit(
            f"live server environment mismatch: {key}={environ.get(key)!r}, "
            f"expected={expected!r}"
        )

forbidden_env = set()
if observer_free_mode:
    cold_startup_provenance = {
        "VLLM_SPARSE_SITE_LOG",
        "VLLM_SPARSE_SITE_LOG_PATH",
    }
    forbidden_env.update(
        key
        for key in environ
        if classify_speed_child_env_key(key) == "observation"
        and key not in cold_startup_provenance
    )
for forbidden_key in sorted(forbidden_env):
    if forbidden_key in environ:
        raise SystemExit(
            f"live server retains forbidden hot-path environment: {forbidden_key}"
        )

controller = json.loads(environ.get("VLLM_SPARSE_CONTROLLER_JSON", "null"))
if not isinstance(controller, dict) or not controller.get("enabled"):
    raise SystemExit("live server has no enabled sparse controller")
if int(controller.get("max_live_sparse_slots", -1)) != int(data["slots"]):
    raise SystemExit("controller slots disagree with server manifest")
if "vllm.entrypoints.openai.api_server" not in cmdline:
    raise SystemExit("manifest PID is not the canonical Python vLLM server")

def cli_value(flag: str) -> str:
    try:
        return cmdline[cmdline.index(flag) + 1]
    except (ValueError, IndexError) as exc:
        raise SystemExit(f"live server command is missing {flag}") from exc

if cli_value("--host") != str(bind_address):
    raise SystemExit("live server command host mismatch")
if int(cli_value("--port")) != requested_port:
    raise SystemExit("live server command port mismatch")
if os.path.realpath(cli_value("--model")) != os.path.realpath(data["model_path"]):
    raise SystemExit("live server command model path mismatch")
if cli_value("--served-model-name") != data["served_model_id"]:
    raise SystemExit("live server served-model-name mismatch")
if int(cli_value("--max-model-len")) != int(data["max_model_len"]):
    raise SystemExit("live server max-model-len mismatch")
if int(cli_value("--max-num-seqs")) != int(data["slots"]):
    raise SystemExit("live server max-num-seqs does not equal sparse slots")
if int(cli_value("--tensor-parallel-size")) != int(data["tensor_parallel_size"]):
    raise SystemExit("live server TP size mismatch")
if proof_mode:
    if cli_value("--worker-extension-cls") != (
        "patches.fa3_native.route_counter_worker_extension."
        "SparseRouteCounterWorkerExtension"
    ):
        raise SystemExit("live server route-counter RPC worker extension mismatch")
elif "--worker-extension-cls" in cmdline:
    raise SystemExit("observer-free server retains a worker extension")

default_api_key = "token-abc123"
client_api_key = os.environ.get("API_KEY")
if data["api_key_source"] == "default-loopback":
    if not bind_address.is_loopback:
        raise SystemExit("default API key is forbidden for a non-loopback server")
    expected_api_key = default_api_key if client_api_key is None else client_api_key
else:
    if not client_api_key:
        raise SystemExit("explicit server authentication requires client API_KEY")
    if not bind_address.is_loopback and client_api_key == default_api_key:
        raise SystemExit("default API key is forbidden for a non-loopback server")
    expected_api_key = client_api_key
if cli_value("--api-key") != expected_api_key:
    raise SystemExit("client API_KEY does not match the live server")

site_log = pathlib.Path(data["site_log"])
if not site_log.is_file():
    raise SystemExit(f"server site log is missing: {site_log}")
site_text = site_log.read_text(encoding="utf-8", errors="replace")
if "patch installed successfully, controller=True" not in site_text:
    raise SystemExit("server site log has no successful sparse patch proof")

for key in (
    "server_pid",
    "run_nonce",
    "started_epoch",
    "run_dir",
    "refresh_profile_log",
    "route_trace_log",
    "step_trace_log",
    "route_counter_snapshot",
    "slots",
    "cuda_arch",
    "attention_kernel",
    "flash_attn_version",
    "host",
    "api_key_source",
    "model_path",
    "served_model_id",
    "max_model_len",
    "trace_enabled",
    "runtime_proof_enabled",
    "hot_path_observer_free",
    "tensor_parallel_size",
):
    print(data[key])
PY
)
(( ${#SERVER_META[@]} == 21 )) || die "failed to parse live server manifest"
SERVER_PID="${SERVER_META[0]}"
SERVER_NONCE="${SERVER_META[1]}"
SERVER_STARTED_EPOCH="${SERVER_META[2]}"
SERVER_RUN_DIR="${SERVER_META[3]}"
REFRESH_PROFILE_LOG="${SERVER_META[4]}"
ROUTE_TRACE_LOG="${SERVER_META[5]}"
STEP_TRACE_LOG="${SERVER_META[6]}"
ROUTE_COUNTER_SNAPSHOT="${SERVER_META[7]}"
SLOTS="${SERVER_META[8]}"
CUDA_ARCH="${SERVER_META[9]}"
ATTENTION_KERNEL="${SERVER_META[10]}"
FLASH_ATTN_VERSION="${SERVER_META[11]}"
SERVER_BIND_HOST="${SERVER_META[12]}"
API_KEY_SOURCE="${SERVER_META[13]}"
SERVER_MODEL_PATH="${SERVER_META[14]}"
SERVED_MODEL_ID="${SERVER_META[15]}"
SERVER_MAX_MODEL_LEN="${SERVER_META[16]}"
TRACE_ENABLED="${SERVER_META[17]}"
RUNTIME_PROOF_ENABLED="${SERVER_META[18]}"
HOT_PATH_OBSERVER_FREE="${SERVER_META[19]}"
TP_SIZE="${SERVER_META[20]}"
if [[ "${RUNTIME_PROOF_ENABLED}" == "True" ]]; then
  [[ "${TRACE_ENABLED}" == "True" && "${HOT_PATH_OBSERVER_FREE}" == "False" ]] || \
    die "runtime-proof LongBench mode has an inconsistent observer contract"
  LIVENESS_ENABLED=1
else
  [[ "${TRACE_ENABLED}" == "False" && "${HOT_PATH_OBSERVER_FREE}" == "True" ]] || \
    die "observer-free LongBench mode has an inconsistent observer contract"
  LIVENESS_ENABLED=0
fi
SERVER_CLIENT_HOST="${SERVER_BIND_HOST}"
if [[ "${SERVER_CLIENT_HOST}" == "0.0.0.0" ]]; then
  SERVER_CLIENT_HOST="127.0.0.1"
fi
if [[ "${API_KEY_SOURCE}" == "default-loopback" ]]; then
  CLIENT_API_KEY="${API_KEY-token-abc123}"
else
  CLIENT_API_KEY="${API_KEY-}"
fi
(( N_PROC <= SLOTS )) || \
  die "N_PROC=${N_PROC} exceeds server slots/max-num-seqs=${SLOTS}"

EXPECTED_ROWS=503
OFFICIAL_OUTPUT_HEADROOM=128
EXPECTED_OFFICIAL_URL="http://${SERVER_CLIENT_HOST}:${PORT}/v1"
OFFICIAL_META_TEXT="$(
  "${PYTHON}" -I - \
    "${PRED_SCRIPT}" "${MODEL_PATH_CONFIG}" "${MODEL_MAXLEN_CONFIG}" \
    "${MODEL_NAME}" "${EXPECTED_OFFICIAL_URL}" "${CLIENT_API_KEY}" \
    "${SERVER_MODEL_PATH}" "${SERVED_MODEL_ID}" "${SERVER_MAX_MODEL_LEN}" \
    "${OFFICIAL_OUTPUT_HEADROOM}" <<'PY'
import ast
import json
import os
import pathlib
import sys

(
    pred_arg,
    model_path_config_arg,
    model_maxlen_config_arg,
    model_name,
    expected_url,
    expected_api_key,
    server_model_path,
    served_model_id,
    server_max_model_len_arg,
    output_headroom_arg,
) = sys.argv[1:]
pred_path = pathlib.Path(pred_arg)
tree = ast.parse(pred_path.read_text(encoding="utf-8"), filename=str(pred_path))


def string_constant(name: str) -> str:
    values = []
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == name for target in targets):
                value = node.value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    values.append(value.value)
                else:
                    raise SystemExit(f"official pred.py {name} must be a string literal")
    if len(values) != 1:
        raise SystemExit(f"official pred.py must define exactly one {name} constant")
    return values[0]


official_url = string_constant("URL")
official_api_key = string_constant("API_KEY")
if official_url.rstrip("/") != expected_url.rstrip("/"):
    raise SystemExit(
        "official pred.py URL does not match the live server: "
        f"{official_url!r} != {expected_url!r}; this runner never edits external code"
    )
if official_api_key != expected_api_key:
    raise SystemExit(
        "official pred.py API_KEY does not match the live server; "
        "this runner never edits external code"
    )

model_map = json.loads(pathlib.Path(model_path_config_arg).read_text(encoding="utf-8"))
maxlen_map = json.loads(pathlib.Path(model_maxlen_config_arg).read_text(encoding="utf-8"))
if not isinstance(model_map, dict) or model_name not in model_map:
    raise SystemExit(f"MODEL_NAME {model_name!r} is absent from config/model2path.json")
mapped_model = model_map[model_name]
if not isinstance(mapped_model, str) or not os.path.isabs(mapped_model):
    raise SystemExit("LongBench model alias must map to an absolute local model path")
if os.path.realpath(mapped_model) != os.path.realpath(server_model_path):
    raise SystemExit(
        "LongBench model alias does not map to the live server model path: "
        f"{mapped_model!r} != {server_model_path!r}"
    )
if mapped_model != served_model_id:
    raise SystemExit(
        "official pred.py couples tokenizer path and API model ID: "
        "config/model2path.json must equal both the live model path and "
        f"served_model_id; mapped={mapped_model!r}, served={served_model_id!r}"
    )
configured_maxlen = maxlen_map.get(model_name) if isinstance(maxlen_map, dict) else None
if isinstance(configured_maxlen, bool) or not isinstance(configured_maxlen, int):
    raise SystemExit(f"MODEL_NAME {model_name!r} has no integer model2maxlen entry")
server_max_model_len = int(server_max_model_len_arg)
output_headroom = int(output_headroom_arg)
if configured_maxlen <= 0:
    raise SystemExit("LongBench configured model max length must be positive")
if configured_maxlen + output_headroom > server_max_model_len:
    raise SystemExit(
        "LongBench configured input length plus official output headroom exceeds "
        f"the live max_model_len: {configured_maxlen} + {output_headroom} > "
        f"{server_max_model_len}"
    )
print(official_url)
print(mapped_model)
print(configured_maxlen)
PY
)" || die "official LongBench source/config validation failed"
mapfile -t OFFICIAL_META <<< "${OFFICIAL_META_TEXT}"
(( ${#OFFICIAL_META[@]} == 3 )) || die "failed to parse official LongBench metadata"
OFFICIAL_URL="${OFFICIAL_META[0]}"
OFFICIAL_MODEL_PATH="${OFFICIAL_META[1]}"
OFFICIAL_MODEL_MAXLEN="${OFFICIAL_META[2]}"

SMOKE_WORDS="${SFI_SMOKE_WORDS:-6000}"
SMOKE_MAX_TOKENS="${SFI_SMOKE_MAX_TOKENS:-192}"
SMOKE_MIN_TOKENS="${SFI_SMOKE_MIN_TOKENS:-128}"
for pair in \
  "SFI_SMOKE_WORDS:${SMOKE_WORDS}" \
  "SFI_SMOKE_MAX_TOKENS:${SMOKE_MAX_TOKENS}" \
  "SFI_SMOKE_MIN_TOKENS:${SMOKE_MIN_TOKENS}"; do
  require_uint "${pair%%:*}" "${pair#*:}"
done
(( SMOKE_WORDS > 0 && SMOKE_MIN_TOKENS > 0 )) || \
  die "LongBench smoke sizes must be positive"
(( SMOKE_MIN_TOKENS <= SMOKE_MAX_TOKENS )) || \
  die "SFI_SMOKE_MIN_TOKENS cannot exceed SFI_SMOKE_MAX_TOKENS"

EVAL_NONCE="${SFI_LONGBENCH_RUN_NONCE:-$(date +%Y%m%dT%H%M%S)-$$}"
[[ "${EVAL_NONCE}" =~ ^[A-Za-z0-9._-]+$ ]] || die "invalid LongBench run nonce"
EVAL_ROOT="$(realpath -ms -- "${SFI_LONGBENCH_RUN_ROOT:-${SFI_ROOT}/tmp/longbench_v2_runs}")"
EVAL_DIR="${EVAL_ROOT}/${EVAL_NONCE}"
[[ ! -e "${EVAL_DIR}" ]] || die "LongBench artifact directory already exists: ${EVAL_DIR}"
mkdir -p "${EVAL_ROOT}"
mkdir "${EVAL_DIR}"
PREDICTIONS_DIR="${EVAL_DIR}/predictions"
SCORING_DIR="${EVAL_DIR}/official_scoring"
mkdir "${PREDICTIONS_DIR}"
mkdir "${SCORING_DIR}"
ln -s ../predictions "${SCORING_DIR}/results"
SMOKE_OUTPUT="${EVAL_DIR}/sparse_smoke.json"
PRED_LOG="${EVAL_DIR}/official_pred.log"
PRED_HELP_LOG="${EVAL_DIR}/official_pred_help.log"
CLIENT_DEPS_LOG="${EVAL_DIR}/official_client_deps.log"
SCORE_LOG="${EVAL_DIR}/official_result.log"
PROVENANCE="${EVAL_DIR}/provenance.json"
SCORE_SUMMARY="${EVAL_DIR}/score_summary.json"
DATASET_IDENTITY="${EVAL_DIR}/dataset_identity.json"

"${LONGBENCH_PYTHON}" -I \
  "${SFI_ROOT}/scripts/check_longbench_v2_dataset.py" \
  --revision "${LONGBENCH_DATASET_REVISION}" \
  --expected-sha256 "${LONGBENCH_DATASET_SHA256}" \
  --output "${DATASET_IDENTITY}" || \
  die "LongBench-v2 pinned offline dataset identity validation failed"

"${PYTHON}" -I - \
  "${LONGBENCH_ROOT}" "${PROVENANCE}" "${MANIFEST}" "${MODEL_NAME}" \
  "${N_PROC}" "${EXPECTED_ROWS}" "${LONGBENCH_PYTHON}" \
  "${LONGBENCH_GIT_HEAD}" "${LONGBENCH_GIT_ORIGIN}" "${OFFICIAL_URL}" \
  "${OFFICIAL_MODEL_PATH}" "${OFFICIAL_MODEL_MAXLEN}" \
  "${OFFICIAL_OUTPUT_HEADROOM}" "${SERVED_MODEL_ID}" "${SERVER_MODEL_PATH}" \
  "${SERVER_MAX_MODEL_LEN}" "${SERVER_NONCE}" "${DATASET_IDENTITY}" \
  "${LONGBENCH_DATASET_REVISION}" "${LONGBENCH_DATASET_SHA256}" \
  "${OFFICIAL_SOURCE_FILES[@]}" <<'PY'
import hashlib
import json
import os
import pathlib
import subprocess
import sys

(
    root_arg,
    provenance_arg,
    manifest_arg,
    model_name,
    n_proc_arg,
    expected_rows_arg,
    longbench_python,
    git_head,
    git_origin,
    official_url,
    mapped_model,
    configured_maxlen_arg,
    output_headroom_arg,
    served_model_id,
    server_model_path,
    server_max_model_len_arg,
    server_nonce,
    dataset_identity_arg,
    dataset_revision,
    dataset_sha256,
    *source_files,
) = sys.argv[1:]
root = pathlib.Path(root_arg).resolve()


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


source_hashes = {
    relative: sha256(root / relative)
    for relative in source_files
}
status = subprocess.run(
    ["git", "-C", str(root), "status", "--porcelain", "--", *source_files],
    check=True,
    capture_output=True,
    text=True,
).stdout.splitlines()
if status:
    raise SystemExit(
        "official LongBench evaluator/config/prompt files became dirty: "
        + "; ".join(status)
    )
for relative, worktree_sha256 in source_hashes.items():
    committed = subprocess.run(
        ["git", "-C", str(root), "show", f"{git_head}:{relative}"],
        check=False,
        capture_output=True,
    )
    if committed.returncode != 0:
        raise SystemExit(f"official LongBench file is not tracked at HEAD: {relative}")
    committed_sha256 = hashlib.sha256(committed.stdout).hexdigest()
    if committed_sha256 != worktree_sha256:
        raise SystemExit(
            "official LongBench worktree differs from HEAD for relevant file: "
            f"{relative}"
        )
manifest = json.loads(pathlib.Path(manifest_arg).read_text(encoding="utf-8"))
dataset_identity = json.loads(
    pathlib.Path(dataset_identity_arg).read_text(encoding="utf-8")
)
if (
    dataset_identity.get("revision") != dataset_revision
    or dataset_identity.get("verified_content_sha256") != dataset_sha256
    or dataset_identity.get("rows") != 503
    or dataset_identity.get("official_default_matches_pinned") is not True
    or dataset_identity.get("offline_cache_only") is not True
):
    raise SystemExit("LongBench dataset identity artifact is inconsistent")
payload = {
    "schema": 1,
    "official_checkout": {
        "root": str(root),
        "git_head": git_head,
        "git_origin": git_origin,
        "relevant_status": status,
        "source_sha256": source_hashes,
    },
    "dataset_identity": dataset_identity,
    "official_configuration": {
        "model_name": model_name,
        "model2path": mapped_model,
        "model2maxlen": int(configured_maxlen_arg),
        "pred_url": official_url,
        "pred_api_key_matches_live": True,
        "output_token_headroom": int(output_headroom_arg),
        "compatibility_boundary": (
            "Official pred.py uses one model2path value for both tokenizer loading "
            "and the OpenAI model ID; it must equal live model_path and served_model_id."
        ),
        "length_boundary": (
            "model2maxlen limits the raw official prompt; 128 output tokens are "
            "reserved here, while model-specific server chat-template overhead must "
            "already be accounted for by the configured model2maxlen."
        ),
    },
    "run_settings": {
        "longbench_python": os.path.realpath(longbench_python),
        "n_proc": int(n_proc_arg),
        "expected_rows": int(expected_rows_arg),
        "resume": False,
        "hf_datasets_offline": True,
        "hf_hub_offline": True,
        "scoring": "official result.py with isolated results symlink",
        "runtime_liveness": (
            "same_run_trace_and_frozen_rpc"
            if manifest["runtime_proof_enabled"]
            else "not_run_hot_path_observer_free"
        ),
    },
    "live_server": {
        "manifest": str(pathlib.Path(manifest_arg).resolve()),
        "run_nonce": server_nonce,
        "model_path": server_model_path,
        "served_model_id": served_model_id,
        "max_model_len": int(server_max_model_len_arg),
        "cuda_arch": manifest["cuda_arch"],
        "cuda_capabilities": manifest["cuda_capabilities"],
        "attention_kernel": manifest["attention_kernel"],
        "flash_attn_version": manifest["flash_attn_version"],
        "api_key_source": manifest["api_key_source"],
        "trace_enabled": manifest["trace_enabled"],
        "runtime_proof_enabled": manifest["runtime_proof_enabled"],
        "hot_path_observer_free": manifest["hot_path_observer_free"],
    },
}
path = pathlib.Path(provenance_arg)
temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, path)
PY

set +e
"${LONGBENCH_PYTHON}" -I - >"${CLIENT_DEPS_LOG}" 2>&1 <<'PY'
import importlib

required = ("tqdm", "datasets", "openai", "transformers", "tiktoken", "torch")
failures = {}
for module_name in required:
    try:
        importlib.import_module(module_name)
    except Exception as exc:
        failures[module_name] = f"{type(exc).__name__}: {exc}"
if failures:
    raise SystemExit(
        "LongBench client dependency import failures: "
        + "; ".join(f"{name}={detail}" for name, detail in failures.items())
    )
print("LongBench client dependencies imported: " + ",".join(required))
PY
CLIENT_DEPS_EXIT_CODE=$?
set -e
(( CLIENT_DEPS_EXIT_CODE == 0 )) || \
  die "LONGBENCH_PYTHON cannot import the official client dependencies tqdm,datasets,openai,transformers,tiktoken,torch; keep them in a separate client environment and see ${CLIENT_DEPS_LOG}"

set +e
(
  cd "${LONGBENCH_ROOT}"
  "${LONGBENCH_PYTHON}" "${PRED_SCRIPT}" --help
) >"${PRED_HELP_LOG}" 2>&1
PRED_HELP_EXIT_CODE=$?
set -e
(( PRED_HELP_EXIT_CODE == 0 )) || \
  die "official pred.py CLI preflight failed under the separate LONGBENCH_PYTHON; see ${PRED_HELP_LOG}"
PRED_HELP="$(<"${PRED_HELP_LOG}")"
for required_flag in --save_dir --model --n_proc; do
  [[ "${PRED_HELP}" == *"${required_flag}"* ]] || \
    die "official pred.py lacks required CLI flag: ${required_flag}"
done

snapshot_evidence() {
  "${PYTHON}" -I - \
    "${REFRESH_PROFILE_LOG}" "${ROUTE_COUNTER_SNAPSHOT}" "${STEP_TRACE_LOG}" \
    "${ROUTE_TRACE_LOG}" <<'PY'
import os
import struct
import sys

profile, counter_snapshot, step_trace, route_trace = sys.argv[1:]
print(os.path.getsize(profile) if os.path.isfile(profile) else 0)
compact_steps = 0
if os.path.isfile(counter_snapshot):
    with open(counter_snapshot, "rb") as fh:
        raw = fh.read()
    slot_bytes = 10 * 8
    if len(raw) >= slot_bytes and len(raw) % slot_bytes == 0:
        compact_steps = sum(
            struct.unpack_from("10q", raw, offset)[8]
            for offset in range(0, len(raw), slot_bytes)
        )
print(compact_steps)
print(os.path.getsize(step_trace) if os.path.isfile(step_trace) else 0)
print(os.path.getsize(route_trace) if os.path.isfile(route_trace) else 0)
PY
}

route_counter_rpc() {
  local phase="$1"
  local json_output="$2"
  local reset_json="${3:-}"
  local -a rpc_args=(
    --base-url "http://${SERVER_CLIENT_HOST}:${PORT}"
    --api-key "${CLIENT_API_KEY}"
    --tp-size "${TP_SIZE}"
    --phase "${phase}"
    --json-output "${json_output}"
    --binary-output "${ROUTE_COUNTER_SNAPSHOT}"
  )
  if [[ "${phase}" == "snapshot" ]]; then
    [[ -n "${reset_json}" ]] || die "snapshot RPC requires a reset artifact"
    rpc_args+=(--reset-json "${reset_json}")
  elif [[ "${phase}" != "reset" || -n "${reset_json}" ]]; then
    die "invalid route-counter RPC arguments: phase=${phase}"
  fi
  "${PYTHON}" -I "${SFI_ROOT}/scripts/snapshot_sparse_route_counters.py" \
    "${rpc_args[@]}"
}

run_liveness_delta() {
  local phase="$1"
  local profile_offset="$2"
  local compact_baseline="$3"
  local step_trace_offset="$4"
  local route_trace_offset="$5"
  local request_policy="$6"
  local -a liveness_args=(
    --refresh-profile-log "${REFRESH_PROFILE_LOG}" \
    --route-counter-snapshot "${ROUTE_COUNTER_SNAPSHOT}" \
    --run-since "${SERVER_STARTED_EPOCH}" \
    --min-world-publish 1 \
    --refresh-profile-offset "${profile_offset}" \
    --baseline-compact-row-steps "${compact_baseline}" \
    --min-compact-row-step-delta 1 \
    --expected-route-counter-ranks "${TP_SIZE}" \
    --phase "${phase}"
  )
  if [[ "${request_policy}" != "strict-long" && "${request_policy}" != "sticky" ]]; then
    die "unknown liveness request policy: ${request_policy}"
  fi
  [[ "${TRACE_ENABLED}" == "True" ]] || \
    die "same-run liveness requires the diagnostic trace/proof server mode"
  liveness_args+=(
    --route-trace "${ROUTE_TRACE_LOG}"
    --route-trace-offset "${route_trace_offset}"
    --step-trace "${STEP_TRACE_LOG}"
    --step-trace-offset "${step_trace_offset}"
    --reject-request-fallback
  )
  if [[ "${request_policy}" == "strict-long" ]]; then
    liveness_args+=(--require-mature-decode-compact)
  fi
  "${PYTHON}" -I "${SFI_ROOT}/scripts/check_sparse_liveness.py" \
    "${liveness_args[@]}" \
    2>&1 | tee "${EVAL_DIR}/${phase}_liveness.log"
}

SMOKE_RESET_JSON="${EVAL_DIR}/smoke_route_counter_reset.json"
SMOKE_SNAPSHOT_JSON="${EVAL_DIR}/smoke_route_counter_snapshot.json"
if (( LIVENESS_ENABLED == 1 )); then
  route_counter_rpc "reset" "${SMOKE_RESET_JSON}"
  mapfile -t SMOKE_BASELINE < <(snapshot_evidence)
  (( ${#SMOKE_BASELINE[@]} == 4 )) || die "cannot snapshot sparse evidence"
  (( SMOKE_BASELINE[1] == 0 )) || die "smoke route-counter reset is not zero"
fi

# One long request must create fresh producer and replay-aware read evidence.
"${PYTHON}" -I - \
  "${SERVER_CLIENT_HOST}" "${PORT}" "${SERVED_MODEL_ID}" \
  "${CLIENT_API_KEY}" "${SMOKE_WORDS}" "${SMOKE_MAX_TOKENS}" \
  "${SMOKE_MIN_TOKENS}" "${SMOKE_OUTPUT}" <<'PY'
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request

(
    host,
    port,
    served_model_id,
    api_key,
    smoke_words,
    max_tokens,
    min_tokens,
    output_path,
) = sys.argv[1:]
base_url = f"http://{host}:{int(port)}"
headers = {"Authorization": f"Bearer {api_key}"}
deadline = time.monotonic() + 30
while True:
    try:
        with urllib.request.urlopen(f"{base_url}/health", timeout=5) as response:
            if 200 <= response.status < 300:
                break
    except Exception:
        if time.monotonic() >= deadline:
            raise SystemExit(f"server health check timed out: {base_url}")
        time.sleep(1)

request = urllib.request.Request(f"{base_url}/v1/models", headers=headers)
with urllib.request.urlopen(request, timeout=10) as response:
    model_payload = json.load(response)
served_ids = [entry.get("id") for entry in model_payload.get("data", [])]
if served_model_id not in served_ids:
    raise SystemExit(
        f"manifest model {served_model_id!r} is absent from server models {served_ids!r}"
    )

prompt = (
    "Read the complete synthetic record before answering. "
    + ("evidence " * int(smoke_words))
    + "\nReturn the token sequence A B repeatedly until the output limit."
)
payload = {
    "model": served_model_id,
    "messages": [{"role": "user", "content": prompt}],
    "temperature": 0,
    "max_tokens": int(max_tokens),
    "min_tokens": int(min_tokens),
    "ignore_eos": True,
}
body = json.dumps(payload).encode("utf-8")
headers.update({"Content-Type": "application/json"})
request = urllib.request.Request(
    f"{base_url}/v1/chat/completions", data=body, headers=headers, method="POST"
)
try:
    with urllib.request.urlopen(request, timeout=600) as response:
        result = json.load(response)
except urllib.error.HTTPError as exc:
    detail = exc.read().decode(errors="replace")
    raise SystemExit(f"sparse smoke HTTP {exc.code}: {detail}") from exc
pathlib.Path(output_path).write_text(
    json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
choices = result.get("choices") or []
content = choices[0].get("message", {}).get("content", "") if choices else ""
completion_tokens = int(result.get("usage", {}).get("completion_tokens", 0) or 0)
if not str(content).strip():
    raise SystemExit("sparse smoke returned empty content")
if completion_tokens < int(min_tokens):
    raise SystemExit(
        f"sparse smoke completion too short: {completion_tokens} < {min_tokens}"
    )
print(
    f"sparse smoke response passed: model={served_model_id} "
    f"completion_tokens={completion_tokens}"
)
PY
if (( LIVENESS_ENABLED == 1 )); then
  route_counter_rpc "snapshot" "${SMOKE_SNAPSHOT_JSON}" "${SMOKE_RESET_JSON}"
  run_liveness_delta "smoke" \
    "${SMOKE_BASELINE[0]}" "${SMOKE_BASELINE[1]}" \
    "${SMOKE_BASELINE[2]}" "${SMOKE_BASELINE[3]}" "strict-long"
else
  echo "INFO: smoke R0-R6 NOT_RUN; server is the hot-path observer-free specialization"
fi

EVAL_RESET_JSON="${EVAL_DIR}/eval_route_counter_reset.json"
EVAL_SNAPSHOT_JSON="${EVAL_DIR}/eval_route_counter_snapshot.json"
if (( LIVENESS_ENABLED == 1 )); then
  route_counter_rpc "reset" "${EVAL_RESET_JSON}"
  mapfile -t EVAL_BASELINE < <(snapshot_evidence)
  (( ${#EVAL_BASELINE[@]} == 4 )) || die "cannot snapshot pre-eval sparse evidence"
  (( EVAL_BASELINE[1] == 0 )) || die "eval route-counter reset is not zero"
fi

echo "==> LongBench v2: alias=${MODEL_NAME} served_model=${SERVED_MODEL_ID} n_proc=${N_PROC} server_pid=${SERVER_PID}"
echo "    arch=${CUDA_ARCH} kernel=${ATTENTION_KERNEL} version=${FLASH_ATTN_VERSION}"
echo "    server=http://${SERVER_CLIENT_HOST}:${PORT} served_model=${SERVED_MODEL_ID} auth=${API_KEY_SOURCE}"
echo "    official_checkout=${LONGBENCH_ROOT} head=${LONGBENCH_GIT_HEAD}"
echo "    dataset_revision=${LONGBENCH_DATASET_REVISION} dataset_sha256=${LONGBENCH_DATASET_SHA256} offline_cache=required"
echo "    official_model2maxlen=${OFFICIAL_MODEL_MAXLEN} output_headroom=${OFFICIAL_OUTPUT_HEADROOM} live_max_model_len=${SERVER_MAX_MODEL_LEN}"
echo "    server_run=${SERVER_NONCE} eval_artifacts=${EVAL_DIR}"
set +e
(
  cd "${LONGBENCH_ROOT}"
  HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
    "${LONGBENCH_PYTHON}" "${PRED_SCRIPT}" \
    --model "${MODEL_NAME}" \
    --n_proc "${N_PROC}" \
    --save_dir "${PREDICTIONS_DIR}"
) 2>&1 | tee "${PRED_LOG}"
PRED_EXIT_CODE=${PIPESTATUS[0]}
set -e
(( PRED_EXIT_CODE == 0 )) || \
  die "official LongBench pred.py failed with exit code ${PRED_EXIT_CODE}"

if (( LIVENESS_ENABLED == 1 )); then
  route_counter_rpc "snapshot" "${EVAL_SNAPSHOT_JSON}" "${EVAL_RESET_JSON}"
  run_liveness_delta "posteval" \
    "${EVAL_BASELINE[0]}" "${EVAL_BASELINE[1]}" \
    "${EVAL_BASELINE[2]}" "${EVAL_BASELINE[3]}" "sticky"
else
  echo "INFO: posteval R0-R6 NOT_RUN; quality/scoring ran without hot-path observers"
fi

"${PYTHON}" -I - \
  "${PREDICTIONS_DIR}" "${EXPECTED_ROWS}" "${DATASET_IDENTITY}" \
  "${EVAL_DIR}/output_check.json" <<'PY'
import hashlib
import json
import pathlib
import sys

predictions_dir = pathlib.Path(sys.argv[1]).resolve()
expected_rows = int(sys.argv[2])
dataset_identity_path = pathlib.Path(sys.argv[3]).resolve()
summary_path = pathlib.Path(sys.argv[4])
files = sorted(predictions_dir.glob("*.jsonl"))
if len(files) != 1:
    raise SystemExit(f"expected exactly one LongBench output JSONL, found {files}")
output = files[0]

rows = []
for line_number, line in enumerate(output.read_text(encoding="utf-8").splitlines(), 1):
    if not line.strip():
        continue
    try:
        row = json.loads(line)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid JSON at {files[0]}:{line_number}: {exc}") from exc
    sample_id = row.get("_id")
    response = row.get("response")
    if not sample_id or not isinstance(response, str) or not response.strip():
        raise SystemExit(
            f"incomplete LongBench row at line {line_number}: "
            f"id={sample_id!r}, response_present={bool(response)}"
        )
    if row.get("stub") or row.get("error"):
        raise SystemExit(f"stub/error LongBench row at line {line_number}")
    if not isinstance(row.get("judge"), bool):
        raise SystemExit(f"official LongBench row lacks boolean judge at line {line_number}")
    if row.get("difficulty") not in {"easy", "hard"}:
        raise SystemExit(f"invalid LongBench difficulty at line {line_number}")
    if row.get("length") not in {"short", "medium", "long"}:
        raise SystemExit(f"invalid LongBench length at line {line_number}")
    rows.append(row)

ids = [str(row["_id"]) for row in rows]
if len(ids) != len(set(ids)):
    raise SystemExit("LongBench output contains duplicate sample IDs")
if len(ids) != expected_rows:
    raise SystemExit(
        f"LongBench output is incomplete: rows={len(ids)}, expected={expected_rows}"
    )
sorted_ids_digest = hashlib.sha256()
for sample_id in sorted(ids):
    encoded_id = sample_id.encode("utf-8")
    sorted_ids_digest.update(len(encoded_id).to_bytes(8, "big"))
    sorted_ids_digest.update(encoded_id)
sorted_ids_sha256 = sorted_ids_digest.hexdigest()
dataset_identity = json.loads(dataset_identity_path.read_text(encoding="utf-8"))
if sorted_ids_sha256 != dataset_identity.get("sorted_ids_sha256"):
    raise SystemExit(
        "LongBench output IDs do not match the pinned 503-row dataset identity"
    )
summary = {
    "status": "passed",
    "output": str(output),
    "rows": len(ids),
    "unique_ids": len(set(ids)),
    "nonempty_responses": len(rows),
    "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    "sorted_ids_sha256": sorted_ids_sha256,
}
summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(f"LongBench output completeness passed: {len(ids)}/{expected_rows}")
PY

set +e
(
  cd "${SCORING_DIR}"
  "${LONGBENCH_PYTHON}" "${RESULT_SCRIPT}"
) 2>&1 | tee "${SCORE_LOG}"
SCORE_EXIT_CODE=${PIPESTATUS[0]}
set -e
(( SCORE_EXIT_CODE == 0 )) || \
  die "official LongBench result.py failed with exit code ${SCORE_EXIT_CODE}"
[[ -s "${SCORING_DIR}/result.txt" ]] || \
  die "official LongBench result.py produced no result.txt"
"${PYTHON}" -I "${SFI_ROOT}/scripts/check_longbench_v2_score.py" \
  --predictions-dir "${PREDICTIONS_DIR}" \
  --result "${SCORING_DIR}/result.txt" \
  --output "${SCORE_SUMMARY}" || \
  die "official LongBench score validation failed"
verify_official_checkout_unchanged

if (( LIVENESS_ENABLED == 1 )); then
  echo "PASS: official LongBench v2 sparse liveness, completeness and scoring"
else
  echo "PASS: official LongBench v2 observer-free completeness and scoring; R0-R6 NOT_RUN"
fi
echo "artifacts=${EVAL_DIR}"
echo "score=${SCORING_DIR}/result.txt"
echo "score_summary=${SCORE_SUMMARY}"
