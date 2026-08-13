#!/usr/bin/env bash
# Reproducibly prepare the architecture-matched SFI FlashAttention source tree.
set -euo pipefail

SFI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_URL="https://github.com/vllm-project/flash-attention.git"
BASE_COMMIT="f5bc33cfc02c744d24a2e9d50e6db656de40611c"
EXPECTED_PATCH_SHA256="3c86344cd5c4a43a53e0e172163e270edcf7d812bb887be94625ef34746bf504"
EXPECTED_PATCHED_TREE="b46578a758d4030ff08b808f692fc9cec60d6888"
# Filled from the checked-in rebased FA4 patch.  FA4 is applied after FA3 so
# SM100 shares the same current wrapper/dispatch base as SM80 and SM90.
EXPECTED_FA4_PATCH_SHA256="d5668e7ed8beb63acc698eadc5ffea01247bf48b154ea042e85b23741698a0c4"
EXPECTED_FA4_PATCHED_TREE="3a117db15dd72f5f7f1636145ee40a1bf7711d5d"
MINIMUM_CMAKE_VERSION="3.26"

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
PYTHON_BIN_DIR="$(realpath -e -- "$(dirname "${PY}")")"
PY="$(realpath -e -- "${PY}")"

detect_capability() {
  local output
  if [[ -n "${GPU}" ]]; then
    output="$(CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" -c \
      'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; major, minor = torch.cuda.get_device_capability(0); print(f"{major}.{minor}")'
    )" || {
      echo "FAIL: cannot inspect selected GPU ${GPU}: ${output}" >&2
      exit 69
    }
  else
    output="$("${PY}" -c \
      'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; major, minor = torch.cuda.get_device_capability(0); print(f"{major}.{minor}")'
    )" || {
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

resolve_cuda_toolchain() {
  local -a cuda_candidate_labels=()
  local -a cuda_candidate_nvcc=()
  local candidate_index
  local cuda_compiler_name
  local cuda_compiler_value
  local cuda_root_name
  local cuda_root_value
  local nvcc_path=""
  for cuda_compiler_name in CUDACXX PYTORCH_NVCC; do
    cuda_compiler_value="${!cuda_compiler_name:-}"
    if [[ -z "${cuda_compiler_value}" ]]; then
      continue
    fi
    if [[ "${cuda_compiler_value}" != /* || ! -x "${cuda_compiler_value}" ]]; then
      echo "FAIL: ${cuda_compiler_name} must be an executable absolute path: ${cuda_compiler_value}" >&2
      exit 64
    fi
    cuda_candidate_labels+=("${cuda_compiler_name}")
    cuda_candidate_nvcc+=("$(realpath -e -- "${cuda_compiler_value}")")
  done
  for cuda_root_name in CUDA_HOME CUDA_PATH; do
    cuda_root_value="${!cuda_root_name:-}"
    if [[ -z "${cuda_root_value}" ]]; then
      continue
    fi
    if [[ "${cuda_root_value}" != /* || ! -x "${cuda_root_value}/bin/nvcc" ]]; then
      echo "FAIL: ${cuda_root_name} must be absolute and contain bin/nvcc: ${cuda_root_value}" >&2
      exit 64
    fi
    cuda_candidate_labels+=("${cuda_root_name}")
    cuda_candidate_nvcc+=("$(realpath -e -- "${cuda_root_value}/bin/nvcc")")
  done

  if (( ${#cuda_candidate_nvcc[@]} > 0 )); then
    nvcc_path="${cuda_candidate_nvcc[0]}"
    for ((candidate_index = 1; candidate_index < ${#cuda_candidate_nvcc[@]}; candidate_index++)); do
      if [[ "${cuda_candidate_nvcc[candidate_index]}" != "${nvcc_path}" ]]; then
        echo "FAIL: explicit CUDA toolchain conflict" >&2
        echo "  ${cuda_candidate_labels[0]} -> ${nvcc_path}" >&2
        echo "  ${cuda_candidate_labels[candidate_index]} -> ${cuda_candidate_nvcc[candidate_index]}" >&2
        echo "CUDA_HOME, CUDA_PATH, CUDACXX, and PYTORCH_NVCC must select one canonical toolkit." >&2
        exit 64
      fi
    done
  elif [[ -x "/usr/local/cuda/bin/nvcc" ]]; then
    # Prefer the conventional toolkit symlink over PATH: multi-toolkit hosts
    # commonly leave an obsolete distro nvcc under /usr/bin.
    nvcc_path="$(realpath -e -- "/usr/local/cuda/bin/nvcc")"
  elif command -v nvcc >/dev/null 2>&1; then
    nvcc_path="$(realpath -e -- "$(command -v nvcc)")"
  else
    echo "FAIL: no CUDA compiler found; set CUDA_HOME, CUDA_PATH, CUDACXX, or PYTORCH_NVCC explicitly" >&2
    exit 69
  fi

  local resolved_home
  local nvcc_release
  local nvcc_major
  local torch_cuda_major
  resolved_home="$(dirname "$(dirname "${nvcc_path}")")"
  if [[ "$(realpath -e -- "${resolved_home}/bin/nvcc")" != "${nvcc_path}" ]]; then
    echo "FAIL: selected CUDA compiler is not owned by its derived toolkit root: ${nvcc_path}" >&2
    exit 64
  fi
  CUDA_HOME="${resolved_home}"
  CUDA_PATH="${resolved_home}"
  CUDACXX="${nvcc_path}"
  PYTORCH_NVCC="${nvcc_path}"

  nvcc_release="$({ "${CUDACXX}" --version; } | sed -n 's/.*release \([^,]*\).*/\1/p' | tail -1)"
  if [[ ! "${nvcc_release}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "FAIL: cannot parse CUDA compiler release from ${CUDACXX}" >&2
    exit 69
  fi
  nvcc_major="${nvcc_release%%.*}"
  if (( 10#${nvcc_major} < 12 )); then
    echo "FAIL: SFI CUDA runtime requires CUDA 12.0 or newer; selected ${CUDACXX} reports ${nvcc_release}" >&2
    exit 69
  fi
  torch_cuda_major="$("${PY}" -c 'import torch; value = str(torch.version.cuda or ""); print(value.split(".", 1)[0])')"
  if [[ ! "${torch_cuda_major}" =~ ^[0-9]+$ || "${torch_cuda_major}" != "${nvcc_major}" ]]; then
    echo "FAIL: CUDA compiler ${nvcc_release} is incompatible with torch CUDA ${torch_cuda_major:-unknown}.x" >&2
    echo "Set CUDA_HOME, CUDA_PATH, CUDACXX, or PYTORCH_NVCC to a toolkit with the same major version." >&2
    exit 69
  fi
  # Caffe2's legacy FindCUDA pass consults PATH independently of CUDACXX.
  # Keep compiler, headers and libraries on one toolkit instead of accepting
  # a CUDACXX=12.x / PATH nvcc=10.x split configuration.
  PATH="${CUDA_HOME}/bin:${PATH}"
  if [[ "$(realpath -e -- "$(command -v nvcc)")" != "${CUDACXX}" ]]; then
    echo "FAIL: PATH did not resolve to the canonical CUDA compiler: $(command -v nvcc)" >&2
    exit 69
  fi
  CUDA_COMPILER_RELEASE="${nvcc_release}"
  CUDA_COMPILER_MAJOR="${nvcc_major}"
  export CUDA_HOME CUDA_PATH CUDACXX PYTORCH_NVCC PATH
  echo "==> CUDA toolkit: home=${CUDA_HOME} nvcc=${CUDACXX} release=${nvcc_release}"
}

resolve_cmake_toolchain() {
  local interpreter_cmake
  local python_package_cmake=""
  local ambient_cmake=""
  local -a candidate_labels=()
  local -a candidate_paths=()
  local -a rejected_candidates=()
  local -a seen_candidates=()
  local candidate_index
  local candidate_label
  local candidate_path
  local canonical_candidate
  local seen_candidate
  local candidate_was_seen
  local version_output
  local version_line
  local cmake_major
  local cmake_tail
  local cmake_minor
  local cmake_patch
  local minimum_major
  local minimum_tail
  local minimum_minor
  local minimum_patch

  interpreter_cmake="${PYTHON_BIN_DIR}/cmake"
  if [[ -e "${interpreter_cmake}" || -L "${interpreter_cmake}" ]]; then
    candidate_labels+=("python-bin")
    candidate_paths+=("${interpreter_cmake}")
  fi
  if ! python_package_cmake="$(
    "${PY}" -c 'from pathlib import Path
try:
    import cmake
except Exception:
    raise SystemExit(0)
root = getattr(cmake, "CMAKE_BIN_DIR", "")
candidate = Path(str(root)) / "cmake" if root else None
if candidate is not None and candidate.is_file():
    print(candidate.resolve(strict=True))'
  )"; then
    python_package_cmake=""
  fi
  if [[ -n "${python_package_cmake}" ]]; then
    candidate_labels+=("python-package")
    candidate_paths+=("${python_package_cmake}")
  fi
  if command -v cmake >/dev/null 2>&1; then
    ambient_cmake="$(command -v cmake)"
    candidate_labels+=("PATH")
    candidate_paths+=("${ambient_cmake}")
  fi

  minimum_major="${MINIMUM_CMAKE_VERSION%%.*}"
  minimum_tail="${MINIMUM_CMAKE_VERSION#*.}"
  minimum_minor="${minimum_tail%%.*}"
  minimum_patch="0"
  if [[ "${minimum_tail}" == *.* ]]; then
    minimum_patch="${minimum_tail#*.}"
  fi

  CMAKE_LAUNCHER=""
  CMAKE_VERSION=""
  for ((candidate_index = 0; candidate_index < ${#candidate_paths[@]}; candidate_index++)); do
    candidate_label="${candidate_labels[candidate_index]}"
    candidate_path="${candidate_paths[candidate_index]}"
    if [[ "${candidate_path}" != /* || ! -x "${candidate_path}" ]]; then
      rejected_candidates+=("${candidate_label}=${candidate_path:-<empty>} (not an executable absolute path)")
      continue
    fi
    canonical_candidate="$(realpath -e -- "${candidate_path}")"
    candidate_was_seen=0
    for seen_candidate in "${seen_candidates[@]}"; do
      if [[ "${seen_candidate}" == "${canonical_candidate}" ]]; then
        candidate_was_seen=1
        break
      fi
    done
    if [[ "${candidate_was_seen}" == "1" ]]; then
      continue
    fi
    seen_candidates+=("${canonical_candidate}")
    if ! version_output="$("${canonical_candidate}" --version)"; then
      rejected_candidates+=("${candidate_label}=${canonical_candidate} (not runnable)")
      continue
    fi
    version_line="${version_output%%$'\n'*}"
    if [[ ! "${version_line}" =~ ^cmake[[:space:]]+version[[:space:]]+([0-9]+([.][0-9]+){1,2})([[:space:]]|$) ]]; then
      rejected_candidates+=("${candidate_label}=${canonical_candidate} (unparseable version)")
      continue
    fi
    CMAKE_VERSION="${BASH_REMATCH[1]}"
    cmake_major="${CMAKE_VERSION%%.*}"
    cmake_tail="${CMAKE_VERSION#*.}"
    cmake_minor="${cmake_tail%%.*}"
    cmake_patch="0"
    if [[ "${cmake_tail}" == *.* ]]; then
      cmake_patch="${cmake_tail#*.}"
    fi
    if (( 10#${cmake_major} < 10#${minimum_major} \
       || (10#${cmake_major} == 10#${minimum_major} \
           && (10#${cmake_minor} < 10#${minimum_minor} \
               || (10#${cmake_minor} == 10#${minimum_minor} \
                   && 10#${cmake_patch} < 10#${minimum_patch}))) )); then
      rejected_candidates+=("${candidate_label}=${canonical_candidate} (${CMAKE_VERSION} < ${MINIMUM_CMAKE_VERSION})")
      CMAKE_VERSION=""
      continue
    fi
    CMAKE_LAUNCHER="${canonical_candidate}"
    break
  done

  if [[ -z "${CMAKE_LAUNCHER}" ]]; then
    echo "FAIL: no compatible CMake found; FA3 requires >= ${MINIMUM_CMAKE_VERSION}" >&2
    for candidate_path in "${rejected_candidates[@]}"; do
      echo "  rejected: ${candidate_path}" >&2
    done
    echo "Install a compatible CMake in the build Python environment or expose one on PATH." >&2
    exit 69
  fi

  # setup.py invokes the bare name `cmake`.  Pin its owner after CUDA has
  # normalized PATH, while keeping CUDA first so an obsolete distro nvcc
  # beside an ambient CMake cannot retake ownership of the build.
  PATH="${CUDA_HOME}/bin:$(dirname "${CMAKE_LAUNCHER}"):${PATH}"
  if [[ "$(realpath -e -- "$(command -v cmake)")" != "${CMAKE_LAUNCHER}" ]]; then
    echo "FAIL: PATH did not resolve to the canonical CMake: $(command -v cmake)" >&2
    exit 69
  fi
  if [[ "$(realpath -e -- "$(command -v nvcc)")" != "${CUDACXX}" ]]; then
    echo "FAIL: CMake PATH pin displaced the canonical CUDA compiler: $(command -v nvcc)" >&2
    exit 69
  fi
  export PATH
  echo "==> CMake: launcher=${CMAKE_LAUNCHER} version=${CMAKE_VERSION} minimum=${MINIMUM_CMAKE_VERSION}"
}

CUDA_COMPILER_RELEASE=""
CUDA_COMPILER_MAJOR=""
resolve_cuda_toolchain
CMAKE_LAUNCHER=""
CMAKE_EXECUTABLE=""
CMAKE_VERSION=""
if [[ "${ARCH}" != "sm100" && "${SKIP_BUILD}" != "1" ]]; then
  resolve_cmake_toolchain
fi

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
BUILD_TEMP=""
CMAKE_CACHE_PATH=""
CMAKE_CUDA_COMPILER=""
CMAKE_CUDA_TOOLKIT_ROOT=""
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
  mkdir -p "${TARGET}/build"
  BUILD_TEMP="$(mktemp -d "${TARGET}/build/sfi-${ARCH}.XXXXXXXX")"
  echo "==> Build FA3 with PYTHON=${PY} TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
  echo "    build_temp=${BUILD_TEMP}"
  (cd "${TARGET}" && "${PY}" setup.py build_ext --inplace --build-temp "${BUILD_TEMP}")

  if ! CMAKE_IDENTITY_TSV="$(
    "${PY}" - "${BUILD_TEMP}" "${CUDACXX}" "${CUDA_HOME}" "${CMAKE_VERSION}" <<'PY'
from __future__ import annotations

from pathlib import Path
import re
import subprocess
import sys


build_temp = Path(sys.argv[1]).resolve(strict=True)
expected_compiler = Path(sys.argv[2]).resolve(strict=True)
expected_root = Path(sys.argv[3]).resolve(strict=True)
expected_cmake_version = sys.argv[4]
caches = sorted(build_temp.rglob("CMakeCache.txt"))
if len(caches) != 1:
    raise SystemExit(
        f"expected exactly one fresh CMakeCache.txt under {build_temp}, found {len(caches)}"
    )
cache = caches[0].resolve(strict=True)
entries: dict[str, str] = {}
for line in cache.read_text(encoding="utf-8", errors="strict").splitlines():
    if not line or line.startswith(("#", "//")) or "=" not in line:
        continue
    typed_key, value = line.split("=", 1)
    key = typed_key.split(":", 1)[0]
    entries[key] = value

compiler_raw = entries.get("CMAKE_CUDA_COMPILER", "")
if not compiler_raw:
    raise SystemExit(f"fresh CMake cache has no CMAKE_CUDA_COMPILER: {cache}")
compiler = Path(compiler_raw).resolve(strict=True)
if compiler != expected_compiler:
    raise SystemExit(
        f"fresh CMake cache selected wrong CUDA compiler: {compiler} != {expected_compiler}"
    )

root_keys = (
    "CUDAToolkit_ROOT",
    "CUDA_TOOLKIT_ROOT_DIR",
    "CMAKE_CUDA_COMPILER_TOOLKIT_ROOT",
)
roots = {
    Path(entries[key]).resolve(strict=True)
    for key in root_keys
    if entries.get(key)
}
if not roots:
    raise SystemExit(f"fresh CMake cache has no CUDA toolkit root: {cache}")
if roots != {expected_root}:
    rendered = ", ".join(sorted(str(root) for root in roots))
    raise SystemExit(
        f"fresh CMake cache selected wrong CUDA toolkit root: [{rendered}] != {expected_root}"
    )

cmake_command_raw = entries.get("CMAKE_COMMAND", "")
if not cmake_command_raw:
    raise SystemExit(f"fresh CMake cache has no CMAKE_COMMAND: {cache}")
cmake_command = Path(cmake_command_raw).resolve(strict=True)
cmake_version_result = subprocess.run(
    [str(cmake_command), "--version"],
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    check=False,
)
cmake_version_match = re.match(
    r"^cmake version ([0-9]+(?:\.[0-9]+){1,2})(?:\s|$)",
    cmake_version_result.stdout.strip(),
)
actual_cmake_version = (
    cmake_version_match.group(1)
    if cmake_version_result.returncode == 0 and cmake_version_match is not None
    else ""
)
if actual_cmake_version != expected_cmake_version:
    raise SystemExit(
        "fresh CMake cache selected a different CMake implementation: "
        f"{cmake_command} reports {actual_cmake_version or 'unparseable'} "
        f"!= launcher {expected_cmake_version}"
    )

print("\t".join((str(cache), str(compiler), str(expected_root), str(cmake_command))))
PY
  )"; then
    echo "FAIL: cannot prove the CMake/CUDA owners selected by the fresh FA3 configure" >&2
    exit 69
  fi
  IFS=$'\t' read -r \
    CMAKE_CACHE_PATH CMAKE_CUDA_COMPILER CMAKE_CUDA_TOOLKIT_ROOT CMAKE_EXECUTABLE \
    <<< "${CMAKE_IDENTITY_TSV}"
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
SFI_PROV_CUDA_HOME="${CUDA_HOME}" \
SFI_PROV_CUDA_PATH="${CUDA_PATH}" \
SFI_PROV_CUDACXX="${CUDACXX}" \
SFI_PROV_CUDA_COMPILER_RELEASE="${CUDA_COMPILER_RELEASE}" \
SFI_PROV_CUDA_COMPILER_MAJOR="${CUDA_COMPILER_MAJOR}" \
SFI_PROV_BUILD_TEMP="${BUILD_TEMP}" \
SFI_PROV_CMAKE_CACHE_PATH="${CMAKE_CACHE_PATH}" \
SFI_PROV_CMAKE_CUDA_COMPILER="${CMAKE_CUDA_COMPILER}" \
SFI_PROV_CMAKE_CUDA_TOOLKIT_ROOT="${CMAKE_CUDA_TOOLKIT_ROOT}" \
SFI_PROV_CMAKE_EXECUTABLE="${CMAKE_EXECUTABLE}" \
SFI_PROV_CMAKE_VERSION="${CMAKE_VERSION}" \
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
    torch_cuda_major = ""
else:
    torch_version = str(torch.__version__)
    torch_cuda = str(torch.version.cuda or "")
    torch_cuda_major = torch_cuda.split(".", 1)[0] if torch_cuda else ""

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
    "schema_version": 5,
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
    "python_executable": str(Path(sys.executable).resolve(strict=True)),
    "python_version": platform.python_version(),
    "torch_version": torch_version,
    "torch_cuda_version": torch_cuda,
    "torch_cuda_major": torch_cuda_major,
    "cuda_home": os.environ["SFI_PROV_CUDA_HOME"],
    "cuda_path": os.environ["SFI_PROV_CUDA_PATH"],
    "cudacxx": os.environ["SFI_PROV_CUDACXX"],
    "cuda_compiler_release": os.environ["SFI_PROV_CUDA_COMPILER_RELEASE"],
    "cuda_compiler_major": os.environ["SFI_PROV_CUDA_COMPILER_MAJOR"],
    "torch_cuda_arch_list": os.environ["SFI_PROV_ARCH_LIST"],
    "nvcc_version": command(os.environ["SFI_PROV_CUDACXX"], "--version"),
    "cmake_build_temp": os.environ["SFI_PROV_BUILD_TEMP"],
    "cmake_cache_path": os.environ["SFI_PROV_CMAKE_CACHE_PATH"],
    "cmake_cuda_compiler": os.environ["SFI_PROV_CMAKE_CUDA_COMPILER"],
    "cmake_cuda_toolkit_root": os.environ["SFI_PROV_CMAKE_CUDA_TOOLKIT_ROOT"],
    "cmake_executable": os.environ["SFI_PROV_CMAKE_EXECUTABLE"],
    "cmake_version": os.environ["SFI_PROV_CMAKE_VERSION"],
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
