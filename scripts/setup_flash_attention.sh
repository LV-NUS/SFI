#!/usr/bin/env bash
# =============================================================================
# SFI CUDA kernel setup: clone upstream FlashAttention, apply SFI patches, build.
#
# Usage:
#   bash scripts/setup_flash_attention.sh                 # FA3 (SM80/SM90)
#   bash scripts/setup_flash_attention.sh --with-fa4      # FA3 + FA4 CuTe (SM100)
#   bash scripts/setup_flash_attention.sh --target <dir>  # custom clone location
#   bash scripts/setup_flash_attention.sh --skip-build    # clone + patch only
#
# What it does:
#   1. Clones vllm-project/flash-attention and pins the SFI upstream base commit.
#   2. Applies kernel_patches/sfi_fa3_sm80_sm90.patch (always) and
#      kernel_patches/sfi_fa4_sm100_cute.patch (with --with-fa4).
#   3. Builds the FA3 extension in-tree: vllm_flash_attn/_vllm_fa3_C*.so.
#      (FA4/SM100 CuTe kernels are Python and JIT-compile at runtime; no build.)
#
# Machine adaptation (set BEFORE running; see README "Adapting to your machine"):
#   CUDA_HOME             CUDA toolkit >= 12.0 (e.g. /usr/local/cuda-12.4)
#   TORCH_CUDA_ARCH_LIST  auto-detected from GPU 0 if unset:
#                           A100/SM80 -> "8.0"   H100/H800/SM90 -> "9.0a"
#   NVCC_THREADS/MAX_JOBS build parallelism (full build takes ~25-40 min)
#
# Success criterion (printed at the end):
#   FA3 OK: <target>/vllm_flash_attn/_vllm_fa3_C.abi3.so exists
# =============================================================================
set -euo pipefail

SFI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_URL="https://github.com/vllm-project/flash-attention.git"
BASE_COMMIT="f5bc33cfc02c744d24a2e9d50e6db656de40611c"
FA3_PATCH="${SFI_ROOT}/kernel_patches/sfi_fa3_sm80_sm90.patch"
FA4_PATCH="${SFI_ROOT}/kernel_patches/sfi_fa4_sm100_cute.patch"

TARGET="${SFI_ROOT}/third_party_upstreams/vllm-project-flash-attention"
WITH_FA4=0
SKIP_BUILD=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-fa4) WITH_FA4=1; shift ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    --target) TARGET="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

echo "==> [1/3] Clone upstream @ ${BASE_COMMIT:0:12}"
if [[ -d "${TARGET}/.git" ]]; then
  echo "    reuse existing clone: ${TARGET}"
else
  mkdir -p "$(dirname "${TARGET}")"
  git clone "${UPSTREAM_URL}" "${TARGET}"
fi
git -C "${TARGET}" fetch origin "${BASE_COMMIT}" 2>/dev/null || true
if [[ -n "$(git -C "${TARGET}" status --porcelain)" ]]; then
  echo "ERROR: ${TARGET} has local changes; refusing to reset. Clean it or use --target." >&2
  exit 1
fi
git -C "${TARGET}" checkout --detach "${BASE_COMMIT}"
git -C "${TARGET}" submodule update --init csrc/cutlass

echo "==> [2/3] Apply SFI kernel patches"
git -C "${TARGET}" apply --check "${FA3_PATCH}"
git -C "${TARGET}" apply "${FA3_PATCH}"
echo "    applied: $(basename "${FA3_PATCH}")"
if [[ "${WITH_FA4}" == "1" ]]; then
  git -C "${TARGET}" apply --check "${FA4_PATCH}"
  git -C "${TARGET}" apply "${FA4_PATCH}"
  echo "    applied: $(basename "${FA4_PATCH}")"
fi

if [[ "${SKIP_BUILD}" == "1" ]]; then
  echo "==> --skip-build requested; done (no .so built)."
  exit 0
fi

echo "==> [3/3] Build FA3 extension (this is the slow step: ~25-40 min full build)"
if [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
  CAP="$(python -c 'import torch; print("%d.%d" % torch.cuda.get_device_capability(0))')"
  case "${CAP}" in
    8.*) export TORCH_CUDA_ARCH_LIST="8.0" ;;
    9.*) export TORCH_CUDA_ARCH_LIST="9.0a" ;;
    10.*)
      echo "    NOTE: SM100/Blackwell detected. FA3 does not target SM100;"
      echo "    the sparse fast path on this GPU uses FA4 CuTe (JIT, no .so)."
      echo "    Building FA3 host library with default arches for completeness."
      export TORCH_CUDA_ARCH_LIST="9.0a"
      ;;
    *) echo "ERROR: unsupported compute capability ${CAP}" >&2; exit 1 ;;
  esac
  echo "    TORCH_CUDA_ARCH_LIST auto-set to ${TORCH_CUDA_ARCH_LIST} (GPU cap ${CAP})"
fi
( cd "${TARGET}" && python setup.py build_ext --inplace )

SO_PATH="$(ls "${TARGET}"/vllm_flash_attn/_vllm_fa3_C*.so 2>/dev/null | head -1 || true)"
if [[ -n "${SO_PATH}" ]]; then
  echo "==> FA3 OK: ${SO_PATH}"
  echo "    Point SFI at this clone via --fa3-upstream-root (benchmarks) or"
  echo "    VLLM_SPARSE_FA3_UPSTREAM_ROOT (serve). If you used the default"
  echo "    target location, SFI finds it automatically."
else
  echo "ERROR: build finished but no _vllm_fa3_C*.so found under ${TARGET}/vllm_flash_attn" >&2
  exit 1
fi
