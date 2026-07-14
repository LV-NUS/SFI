#!/usr/bin/env bash
# Reproducibly prepare the architecture-matched SFI FlashAttention source tree.
set -euo pipefail

SFI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_URL="https://github.com/vllm-project/flash-attention.git"
BASE_COMMIT="f5bc33cfc02c744d24a2e9d50e6db656de40611c"
EXPECTED_PATCH_SHA256="919c11b88317c3af9e2eb5d3021cad01029a774d95dce1bdf0e09dd5ef7f9b3a"
EXPECTED_PATCHED_TREE="65d3695a0e88611616b0d6a51ea0fc931f206d65"
# Filled from the checked-in rebased FA4 patch.  FA4 is applied after FA3 so
# SM100 shares the same current wrapper/dispatch base as SM80 and SM90.
EXPECTED_FA4_PATCH_SHA256="d5668e7ed8beb63acc698eadc5ffea01247bf48b154ea042e85b23741698a0c4"
EXPECTED_FA4_PATCHED_TREE="e33196e5f697b809fac88e67d5c9d3c92b233c5e"

TARGET="${SFI_ROOT}/third_party_upstreams/vllm-project-flash-attention"
PROVENANCE_OUTPUT=""
SKIP_BUILD=0
ARCH="auto"
ARCH_WAS_SET=0
GPU=""
WITH_FA4_ALIAS=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-fa4)
      WITH_FA4_ALIAS=1
      shift
      ;;
    --arch)
      if [[ $# -lt 2 ]]; then
        echo "FAIL: --arch requires auto, sm80, sm90, or sm100" >&2
        exit 64
      fi
      ARCH="$2"
      ARCH_WAS_SET=1
      shift 2
      ;;
    --gpu)
      if [[ $# -lt 2 ]]; then
        echo "FAIL: --gpu requires one non-negative device index" >&2
        exit 64
      fi
      GPU="$2"
      shift 2
      ;;
    --skip-build)
      SKIP_BUILD=1
      shift
      ;;
    --target)
      if [[ $# -lt 2 ]]; then
        echo "FAIL: --target requires an absolute directory" >&2
        exit 64
      fi
      TARGET="$2"
      shift 2
      ;;
    --provenance-output)
      if [[ $# -lt 2 ]]; then
        echo "FAIL: --provenance-output requires an absolute file path" >&2
        exit 64
      fi
      PROVENANCE_OUTPUT="$2"
      shift 2
      ;;
    *)
      echo "FAIL: unknown argument: $1" >&2
      exit 64
      ;;
  esac
done

case "${ARCH}" in
  auto|sm80|sm90|sm100) ;;
  *)
    echo "FAIL: --arch requires auto, sm80, sm90, or sm100: ${ARCH}" >&2
    exit 64
    ;;
esac
if [[ -n "${GPU}" && ! "${GPU}" =~ ^[0-9]+$ ]]; then
  echo "FAIL: --gpu requires one non-negative device index: ${GPU}" >&2
  exit 64
fi
if [[ "${WITH_FA4_ALIAS}" == "1" ]]; then
  if [[ "${ARCH_WAS_SET}" == "1" && "${ARCH}" != "auto" && "${ARCH}" != "sm100" ]]; then
    echo "FAIL: --with-fa4 is an SM100 alias and conflicts with --arch ${ARCH}" >&2
    exit 64
  fi
  ARCH="sm100"
fi

PY="${PYTHON:?PYTHON env required: executable absolute path of the build interpreter}"
if [[ "${PY}" != /* || ! -x "${PY}" ]]; then
  echo "FAIL: PYTHON must be an executable absolute path: ${PY}" >&2
  exit 64
fi
if ! command -v realpath >/dev/null 2>&1; then
  echo "FAIL: realpath is required for reproducible target identities" >&2
  exit 69
fi
PY="$(realpath -e -- "${PY}")"

detect_capability() {
  local output
  if [[ -n "${GPU}" ]]; then
    output="$({ CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" -c \
      'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; major, minor = torch.cuda.get_device_capability(0); print(f"{major}.{minor}")'
    } 2>&1)" || {
      echo "FAIL: cannot inspect selected GPU ${GPU}: ${output}" >&2
      exit 69
    }
  else
    output="$("${PY}" -c \
      'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; major, minor = torch.cuda.get_device_capability(0); print(f"{major}.{minor}")' \
      2>&1)" || {
      echo "FAIL: cannot inspect the first CUDA-visible GPU: ${output}" >&2
      exit 69
    }
  fi
  printf '%s\n' "${output}"
}

capability_to_arch() {
  case "$1" in
    8.0) printf '%s\n' "sm80" ;;
    9.0) printf '%s\n' "sm90" ;;
    10.0) printf '%s\n' "sm100" ;;
    *)
      echo "FAIL: unsupported compute capability $1; expected 8.0, 9.0, or 10.0" >&2
      exit 78
      ;;
  esac
}

DETECTED_CAPABILITY=""
if [[ "${ARCH}" == "auto" ]]; then
  DETECTED_CAPABILITY="$(detect_capability)"
  ARCH="$(capability_to_arch "${DETECTED_CAPABILITY}")"
elif [[ -n "${GPU}" ]]; then
  DETECTED_CAPABILITY="$(detect_capability)"
  DETECTED_ARCH="$(capability_to_arch "${DETECTED_CAPABILITY}")"
  if [[ "${DETECTED_ARCH}" != "${ARCH}" ]]; then
    echo "FAIL: --arch ${ARCH} does not match selected GPU ${GPU} (${DETECTED_CAPABILITY}/${DETECTED_ARCH})" >&2
    exit 78
  fi
fi

case "${ARCH}" in
  sm80) DEFAULT_ARCH_LIST="8.0" ; BACKEND="fa3" ;;
  sm90) DEFAULT_ARCH_LIST="9.0a" ; BACKEND="fa3" ;;
  sm100) DEFAULT_ARCH_LIST="" ; BACKEND="fa4_cute" ;;
esac

resolve_fa3_cuda_toolkit() {
  if [[ "${SKIP_BUILD}" == "1" || "${BACKEND}" != "fa3" ]]; then
    return
  fi

  local nvcc_path=""
  if [[ -n "${CUDACXX:-}" ]]; then
    if [[ "${CUDACXX}" != /* || ! -x "${CUDACXX}" ]]; then
      echo "FAIL: CUDACXX must be an executable absolute path: ${CUDACXX}" >&2
      exit 64
    fi
    nvcc_path="$(realpath -e -- "${CUDACXX}")"
  elif [[ -n "${CUDA_HOME:-}" ]]; then
    if [[ "${CUDA_HOME}" != /* || ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
      echo "FAIL: CUDA_HOME must be absolute and contain bin/nvcc: ${CUDA_HOME}" >&2
      exit 64
    fi
    nvcc_path="$(realpath -e -- "${CUDA_HOME}/bin/nvcc")"
  elif [[ -x "/usr/local/cuda/bin/nvcc" ]]; then
    # Prefer the conventional toolkit symlink over PATH: multi-toolkit hosts
    # commonly leave an obsolete distro nvcc under /usr/bin.
    nvcc_path="$(realpath -e -- "/usr/local/cuda/bin/nvcc")"
  elif command -v nvcc >/dev/null 2>&1; then
    nvcc_path="$(realpath -e -- "$(command -v nvcc)")"
  else
    echo "FAIL: no CUDA compiler found; set CUDA_HOME or CUDACXX explicitly" >&2
    exit 69
  fi

  local resolved_home
  local nvcc_release
  local nvcc_major
  local torch_cuda_major
  resolved_home="$(dirname "$(dirname "${nvcc_path}")")"
  if [[ -n "${CUDA_HOME:-}" ]]; then
    CUDA_HOME="$(realpath -e -- "${CUDA_HOME}")"
    if [[ "$(realpath -e -- "${CUDA_HOME}/bin/nvcc")" != "${nvcc_path}" ]]; then
      echo "FAIL: CUDA_HOME and CUDACXX select different CUDA toolkits" >&2
      exit 64
    fi
  else
    CUDA_HOME="${resolved_home}"
  fi
  CUDACXX="${nvcc_path}"

  nvcc_release="$({ "${CUDACXX}" --version; } | sed -n 's/.*release \([^,]*\).*/\1/p' | tail -1)"
  if [[ ! "${nvcc_release}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "FAIL: cannot parse CUDA compiler release from ${CUDACXX}" >&2
    exit 69
  fi
  nvcc_major="${nvcc_release%%.*}"
  if (( 10#${nvcc_major} < 12 )); then
    echo "FAIL: FA3 build requires CUDA 12.0 or newer; selected ${CUDACXX} reports ${nvcc_release}" >&2
    exit 69
  fi
  torch_cuda_major="$("${PY}" -c 'import torch; value = str(torch.version.cuda or ""); print(value.split(".", 1)[0])')"
  if [[ ! "${torch_cuda_major}" =~ ^[0-9]+$ || "${torch_cuda_major}" != "${nvcc_major}" ]]; then
    echo "FAIL: CUDA compiler ${nvcc_release} is incompatible with torch CUDA ${torch_cuda_major:-unknown}.x" >&2
    echo "Set CUDA_HOME or CUDACXX to a toolkit with the same major version." >&2
    exit 69
  fi
  # Caffe2's legacy FindCUDA pass consults PATH independently of CUDACXX.
  # Keep compiler, headers and libraries on one toolkit instead of accepting
  # a CUDACXX=12.x / PATH nvcc=10.x split configuration.
  PATH="${CUDA_HOME}/bin:${PATH}"
  export CUDA_HOME CUDACXX PATH
  echo "==> CUDA toolkit: home=${CUDA_HOME} nvcc=${CUDACXX} release=${nvcc_release}"
}

resolve_fa3_cuda_toolkit

if [[ "${TARGET}" != /* ]]; then
  echo "FAIL: --target must be absolute: ${TARGET}" >&2
  exit 64
fi
TARGET="$(realpath -ms -- "${TARGET}")"
if [[ -z "${PROVENANCE_OUTPUT}" ]]; then
  PROVENANCE_OUTPUT="${TARGET}/sfi_flash_attention_build_provenance.json"
elif [[ "${PROVENANCE_OUTPUT}" != /* ]]; then
  echo "FAIL: --provenance-output must be absolute: ${PROVENANCE_OUTPUT}" >&2
  exit 64
else
  PROVENANCE_OUTPUT="$(realpath -ms -- "${PROVENANCE_OUTPUT}")"
fi

# Keep this script byte-identical between the canonical and release exports.
# The two trees intentionally package the same patch under different folders.
CANONICAL_PATCH="${SFI_ROOT}/patches/fa3_native/upstream_recovery/latest-sm80-dual-source-current.patch"
RELEASE_PATCH="${SFI_ROOT}/kernel_patches/sfi_fa3_sm80_sm90.patch"
CANONICAL_FA4_PATCH="${SFI_ROOT}/patches/fa4_cute_sm100/upstream_recovery/latest-sm100-cute-compact-recent-current.patch"
RELEASE_FA4_PATCH="${SFI_ROOT}/kernel_patches/sfi_fa4_sm100_cute.patch"
patch_sha256() {
  "${PY}" - "$1" <<'PY'
from hashlib import sha256
from pathlib import Path
import sys

path = Path(sys.argv[1])
print(sha256(path.read_bytes()).hexdigest())
PY
}

FA3_PATCH=""
for patch_candidate in "${CANONICAL_PATCH}" "${RELEASE_PATCH}"; do
  if [[ ! -f "${patch_candidate}" ]]; then
    continue
  fi
  candidate_sha256="$(patch_sha256 "${patch_candidate}")"
  if [[ "${candidate_sha256}" != "${EXPECTED_PATCH_SHA256}" ]]; then
    echo "FAIL: FA3 patch freshness mismatch" >&2
    echo "  expected: ${EXPECTED_PATCH_SHA256}" >&2
    echo "  actual:   ${candidate_sha256}" >&2
    echo "  patch:    ${patch_candidate}" >&2
    echo "Regenerate the patch from ${BASE_COMMIT} to the intended nested working tree and update this gate." >&2
    exit 65
  fi
  # Canonical source wins when both export layouts are present.  Every present
  # candidate was hash-checked above, so a stale duplicate cannot stay hidden.
  if [[ -z "${FA3_PATCH}" ]]; then
    FA3_PATCH="${patch_candidate}"
  fi
done
if [[ -z "${FA3_PATCH}" ]]; then
  echo "FAIL: canonical/release FA3 patch is missing" >&2
  exit 66
fi
PATCH_SHA256="${EXPECTED_PATCH_SHA256}"

FA4_PATCH=""
if [[ "${ARCH}" == "sm100" ]]; then
  if [[ "${EXPECTED_FA4_PATCHED_TREE}" == __SFI_* ]]; then
    echo "FAIL: setup script is missing the pinned FA3+FA4 combined tree identity" >&2
    exit 65
  fi
  for patch_candidate in "${CANONICAL_FA4_PATCH}" "${RELEASE_FA4_PATCH}"; do
    if [[ ! -f "${patch_candidate}" ]]; then
      continue
    fi
    candidate_sha256="$(patch_sha256 "${patch_candidate}")"
    if [[ "${candidate_sha256}" != "${EXPECTED_FA4_PATCH_SHA256}" ]]; then
      echo "FAIL: FA4 patch freshness mismatch" >&2
      echo "  expected: ${EXPECTED_FA4_PATCH_SHA256}" >&2
      echo "  actual:   ${candidate_sha256}" >&2
      echo "  patch:    ${patch_candidate}" >&2
      exit 65
    fi
    if [[ -z "${FA4_PATCH}" ]]; then
      FA4_PATCH="${patch_candidate}"
    fi
  done
  if [[ -z "${FA4_PATCH}" ]]; then
    echo "FAIL: canonical/release FA4 SM100 overlay patch is missing" >&2
    exit 66
  fi
fi

echo "==> FlashAttention source: arch=${ARCH} backend=${BACKEND} base=${BASE_COMMIT}"
echo "    fa3_patch_sha256=${PATCH_SHA256}"
if [[ -n "${FA4_PATCH}" ]]; then
  echo "    fa4_patch_sha256=${EXPECTED_FA4_PATCH_SHA256}"
fi
if [[ -e "${TARGET}" && ! -d "${TARGET}/.git" ]]; then
  echo "FAIL: target exists but is not a git checkout: ${TARGET}" >&2
  exit 73
fi
if [[ ! -d "${TARGET}/.git" ]]; then
  mkdir -p "$(dirname "${TARGET}")"
  git clone "${UPSTREAM_URL}" "${TARGET}"
fi

# Ignore only this script's prior provenance file.  Any other untracked or
# tracked source can change the build and therefore invalidates reproduction.
TARGET_STATUS="$(
  git -C "${TARGET}" status --porcelain --untracked-files=all -- \
    . ':(exclude)sfi_fa3_build_provenance.json' \
    ':(exclude)sfi_flash_attention_build_provenance.json'
)"
UNTRACKED_STATUS="$(printf '%s\n' "${TARGET_STATUS}" | sed -n '/^?? /p')"
HEAD_COMMIT="$(git -C "${TARGET}" rev-parse HEAD)"
HEAD_TREE="$(git -C "${TARGET}" rev-parse 'HEAD^{tree}')"
INDEX_TREE="$(git -C "${TARGET}" write-tree)"
SOURCE_MODE=""
EXPECTED_FINAL_TREE="${EXPECTED_PATCHED_TREE}"
if [[ "${ARCH}" == "sm100" ]]; then
  EXPECTED_FINAL_TREE="${EXPECTED_FA4_PATCHED_TREE}"
fi

if [[ -z "${TARGET_STATUS}" \
   && "${HEAD_TREE}" == "${EXPECTED_FINAL_TREE}" ]]; then
  # Source bytes, including gitlinks, are the build identity.  Do not pin a
  # historical commit after the current recovery patch has moved past it.
  SOURCE_MODE="verified_patched_tree"
elif [[ "${HEAD_COMMIT}" == "${BASE_COMMIT}" \
     && "${INDEX_TREE}" == "${EXPECTED_FINAL_TREE}" \
     && -z "${UNTRACKED_STATUS}" \
     && -z "$(git -C "${TARGET}" diff --name-only)" ]]; then
  # Idempotent re-entry after a previous --skip-build or build.
  SOURCE_MODE="verified_applied_patch"
elif [[ -z "${TARGET_STATUS}" ]]; then
  if ! git -C "${TARGET}" cat-file -e "${BASE_COMMIT}^{commit}" 2>/dev/null; then
    git -C "${TARGET}" fetch origin "${BASE_COMMIT}"
  fi
  git -C "${TARGET}" checkout --detach "${BASE_COMMIT}"
  git -C "${TARGET}" apply --check "${FA3_PATCH}"
  git -C "${TARGET}" apply --index "${FA3_PATCH}"
  if [[ "${ARCH}" == "sm100" ]]; then
    git -C "${TARGET}" apply --check "${FA4_PATCH}"
    git -C "${TARGET}" apply --index "${FA4_PATCH}"
  fi
  INDEX_TREE="$(git -C "${TARGET}" write-tree)"
  if [[ "${INDEX_TREE}" != "${EXPECTED_FINAL_TREE}" ]]; then
    echo "FAIL: applied ${ARCH} patch stack produced unexpected tree: ${INDEX_TREE}" >&2
    exit 65
  fi
  SOURCE_MODE="fresh_applied_${ARCH}_patch_stack"
else
  echo "FAIL: target has source changes; refusing to overwrite them: ${TARGET}" >&2
  printf '%s\n' "${TARGET_STATUS}" >&2
  exit 73
fi

git -C "${TARGET}" submodule update --init csrc/cutlass

BUILD_STATUS="patch_only"
SO_PATH=""
if [[ "${ARCH}" == "sm100" ]]; then
  for required_source in \
    "${TARGET}/flash_attn/cute/interface.py" \
    "${TARGET}/flash_attn/cute/flash_fwd_sm100.py"; do
    if [[ ! -f "${required_source}" ]]; then
      echo "FAIL: FA4 CuTe JIT source is missing: ${required_source}" >&2
      exit 66
    fi
  done
  BUILD_STATUS="cute_jit_source_ready"
elif [[ "${SKIP_BUILD}" != "1" ]]; then
  if [[ -n "${TORCH_CUDA_ARCH_LIST:-}" && "${TORCH_CUDA_ARCH_LIST}" != "${DEFAULT_ARCH_LIST}" ]]; then
    echo "FAIL: TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} conflicts with --arch ${ARCH}; expected ${DEFAULT_ARCH_LIST}" >&2
    exit 64
  fi
  export TORCH_CUDA_ARCH_LIST="${DEFAULT_ARCH_LIST}"
  echo "==> Build FA3 with PYTHON=${PY} TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
  (cd "${TARGET}" && "${PY}" setup.py build_ext --inplace)
  SO_PATH="$(find "${TARGET}/vllm_flash_attn" -maxdepth 1 -type f -name '_vllm_fa3_C*.so' -print -quit)"
  if [[ -z "${SO_PATH}" ]]; then
    echo "FAIL: build completed without _vllm_fa3_C*.so" >&2
    exit 1
  fi
  BUILD_STATUS="built"
fi

mkdir -p "$(dirname "${PROVENANCE_OUTPUT}")"
SFI_PROV_BASE_COMMIT="${BASE_COMMIT}" \
SFI_PROV_EXPECTED_TREE="${EXPECTED_FINAL_TREE}" \
SFI_PROV_PATCH_PATH="${FA3_PATCH}" \
SFI_PROV_PATCH_SHA256="${PATCH_SHA256}" \
SFI_PROV_FA4_PATCH_PATH="${FA4_PATCH}" \
SFI_PROV_FA4_PATCH_SHA256="${EXPECTED_FA4_PATCH_SHA256}" \
SFI_PROV_TARGET="${TARGET}" \
SFI_PROV_ARCH="${ARCH}" \
SFI_PROV_BACKEND="${BACKEND}" \
SFI_PROV_DETECTED_CAPABILITY="${DETECTED_CAPABILITY}" \
SFI_PROV_SOURCE_MODE="${SOURCE_MODE}" \
SFI_PROV_BUILD_STATUS="${BUILD_STATUS}" \
SFI_PROV_SO_PATH="${SO_PATH}" \
SFI_PROV_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-}" \
SFI_PROV_CUDA_HOME="${CUDA_HOME:-}" \
SFI_PROV_OUTPUT="${PROVENANCE_OUTPUT}" \
"${PY}" - <<'PY'
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def command(*args: str) -> str:
    result = subprocess.run(args, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else ""


so_path = Path(os.environ["SFI_PROV_SO_PATH"]) if os.environ["SFI_PROV_SO_PATH"] else None
try:
    import torch
except Exception:
    torch_version = ""
    torch_cuda = ""
else:
    torch_version = str(torch.__version__)
    torch_cuda = str(torch.version.cuda or "")

patches = [
    {
        "kind": "fa3_sm80_sm90_shared_base",
        "path": os.environ["SFI_PROV_PATCH_PATH"],
        "sha256": os.environ["SFI_PROV_PATCH_SHA256"],
    }
]
if os.environ["SFI_PROV_FA4_PATCH_PATH"]:
    patches.append(
        {
            "kind": "fa4_sm100_cute_overlay",
            "path": os.environ["SFI_PROV_FA4_PATCH_PATH"],
            "sha256": os.environ["SFI_PROV_FA4_PATCH_SHA256"],
        }
    )

payload = {
    "schema_version": 3,
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "architecture": os.environ["SFI_PROV_ARCH"],
    "backend": os.environ["SFI_PROV_BACKEND"],
    "detected_compute_capability": os.environ["SFI_PROV_DETECTED_CAPABILITY"],
    "base_commit": os.environ["SFI_PROV_BASE_COMMIT"],
    "expected_patched_tree": os.environ["SFI_PROV_EXPECTED_TREE"],
    "patch_path": os.environ["SFI_PROV_PATCH_PATH"],
    "patch_sha256": os.environ["SFI_PROV_PATCH_SHA256"],
    "patches": patches,
    "target": os.environ["SFI_PROV_TARGET"],
    "source_mode": os.environ["SFI_PROV_SOURCE_MODE"],
    "build_status": os.environ["SFI_PROV_BUILD_STATUS"],
    "python_executable": sys.executable,
    "python_version": platform.python_version(),
    "torch_version": torch_version,
    "torch_cuda_version": torch_cuda,
    "cuda_home": os.environ["SFI_PROV_CUDA_HOME"],
    "torch_cuda_arch_list": os.environ["SFI_PROV_ARCH_LIST"],
    "nvcc_version": command(os.environ.get("CUDACXX", "nvcc"), "--version"),
    "shared_object": str(so_path) if so_path is not None else "",
    "shared_object_sha256": (
        sha256(so_path.read_bytes()).hexdigest()
        if so_path is not None and so_path.is_file()
        else ""
    ),
}
output = Path(os.environ["SFI_PROV_OUTPUT"])
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

echo "==> ${ARCH}/${BACKEND} ${BUILD_STATUS}: source_mode=${SOURCE_MODE}"
echo "    target=${TARGET}"
echo "    provenance=${PROVENANCE_OUTPUT}"
if [[ -n "${SO_PATH}" ]]; then
  echo "    shared_object=${SO_PATH}"
fi
