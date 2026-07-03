"""CUDA extension for fused log_f + prior (pre-soft-nms) selector stage."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from typing import Optional, Tuple

import torch
from torch.utils.cpp_extension import load_inline
from utils.torch_extension_cache import load_prebuilt_extension

_MODULE: Optional[torch.nn.Module] = None
_LOAD_ERROR: Optional[Exception] = None
_REQUIRED_EXT_SYMBOLS = (
    "fused_log_f_prior_logits",
    "fused_log_f_prior_pre_denom",
    "reduce_log_f_pre_scratch",
    "copy_log_f_lastn1_scratch",
    "copy_log_f_lastn1_scratch_scalar",
    "reduce_log_f_pre_scratch_scalar",
    # 标记符号：缺它的 prebuilt .so 不支持跨片 accumulate meta（flags bit4 +
    # cols 8/9），必须拒载触发 load_inline 重编（同 fixed-shape topk 先例）。
    "reduce_log_f_pre_scratch_accum_supported",
)
_NVCC_RELEASE_RE = re.compile(r"release\s+(\d+)\.(\d+)")


class SelectorLogSExtUnavailable(RuntimeError):
    """Raised when the optional selector CUDA extension is unavailable."""


def _should_enable() -> bool:
    logs_cuda = os.environ.get("VLLM_SPARSE_SELECTOR_LOGS_CUDA", "0") == "1"
    pipeline = os.environ.get("VLLM_SPARSE_SELECTOR_CUDA_PIPELINE", "0") == "1"
    # Pipeline mode implies log_s CUDA stages are expected to be available.
    return logs_cuda or pipeline


def _ensure_torch_cuda_arch_list() -> None:
    if os.environ.get("TORCH_CUDA_ARCH_LIST"):
        return
    if not torch.cuda.is_available():
        return
    try:
        major, minor = torch.cuda.get_device_capability()
    except Exception:
        return
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{int(major)}.{int(minor)}"


def _cuda_std_flag_for_nvcc_version(version_text: str) -> str:
    match = _NVCC_RELEASE_RE.search(version_text)
    if match is None:
        return "-std=c++17"
    major = int(match.group(1))
    if major < 11:
        return "-std=c++14"
    return "-std=c++17"


def _preferred_nvcc_path() -> Optional[str]:
    explicit_nvcc = os.environ.get("PYTORCH_NVCC") or os.environ.get("CUDACXX")
    if explicit_nvcc:
        return explicit_nvcc

    candidates = []
    env_nvcc = os.path.join(os.path.dirname(sys.executable), "nvcc")
    candidates.append(env_nvcc)
    for cuda_home_var in ("CUDA_HOME", "CUDA_PATH"):
        cuda_home = os.environ.get(cuda_home_var)
        if cuda_home:
            candidates.append(os.path.join(cuda_home, "bin", "nvcc"))
    candidates.extend(
        [
            "/usr/local/cuda/bin/nvcc",
            "/usr/local/cuda-12.6/bin/nvcc",
            "/usr/local/cuda-12.5/bin/nvcc",
            "/usr/local/cuda-12.4/bin/nvcc",
        ]
    )
    path_nvcc = shutil.which("nvcc")
    if path_nvcc:
        candidates.append(path_nvcc)

    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        if os.path.exists(candidate):
            return candidate
    return None


def _configure_torch_cuda_toolchain() -> None:
    nvcc = _preferred_nvcc_path()
    if not nvcc:
        return
    cuda_home = os.path.dirname(os.path.dirname(nvcc))
    os.environ["PYTORCH_NVCC"] = nvcc
    os.environ["CUDA_HOME"] = cuda_home
    os.environ["CUDA_PATH"] = cuda_home
    try:
        import torch.utils.cpp_extension as torch_cpp_extension

        torch_cpp_extension.CUDA_HOME = cuda_home
    except Exception:
        return


def _cuda_std_flag_for_current_nvcc() -> str:
    nvcc = _preferred_nvcc_path()
    if not nvcc:
        return "-std=c++17"
    try:
        completed = subprocess.run(
            [nvcc, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return "-std=c++17"
    version_text = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
    return _cuda_std_flag_for_nvcc_version(version_text)


def _load_ext(*, force: bool = False) -> Optional[torch.nn.Module]:
    global _MODULE, _LOAD_ERROR
    if _MODULE is not None:
        if all(hasattr(_MODULE, name) for name in _REQUIRED_EXT_SYMBOLS):
            return _MODULE
        _MODULE = None
        _LOAD_ERROR = None
    if _LOAD_ERROR is not None:
        return None
    if not force and not _should_enable():
        return None
    prebuilt = load_prebuilt_extension("selector_log_s_ext")
    if prebuilt is not None and all(
        hasattr(prebuilt, name) for name in _REQUIRED_EXT_SYMBOLS
    ):
        _MODULE = prebuilt
        return _MODULE

    cpp_source = r"""
#include <torch/extension.h>
#include <vector>
#include <cstdlib>

std::vector<torch::Tensor> fused_log_f_prior_logits_cuda(
    torch::Tensor scores,
    torch::Tensor row_lo,
    torch::Tensor row_hi,
    torch::Tensor key_norms,
    double alpha,
    double eps,
    double gamma,
    double prior_weight_l2,
    double prior_weight_pos,
    double prior_pos_power,
    double prior_pos_eta,
    double beta,
    double lambda_clip_single,
    double lambda_clip_multi,
    double lambda_tail_kappa,
    double lambda_tail_pivot,
    bool lambda_soft);

std::vector<torch::Tensor> fused_log_f_prior_pre_denom_cuda(
    torch::Tensor scores,
    torch::Tensor denom,
    torch::Tensor row_lo,
    torch::Tensor row_hi,
    torch::Tensor key_norms,
    double alpha,
    double eps,
    double gamma,
    double prior_weight_l2,
    double prior_weight_pos,
    double prior_pos_power,
    double prior_pos_eta,
    double beta,
    double lambda_clip_single,
    double lambda_clip_multi,
    double lambda_tail_kappa,
    double lambda_tail_pivot,
    bool lambda_soft);

void reduce_log_f_pre_scratch_cuda(
    torch::Tensor req_meta_i32,
    torch::Tensor req_meta_i64,
    int64_t num_seqs,
    int64_t num_query_heads,
    bool scratch_in_fp16,
    bool log_f_out_fp32,
    double alpha);

void copy_log_f_lastn1_scratch_cuda(
    torch::Tensor req_meta_i32,
    torch::Tensor req_meta_i64,
    int64_t num_seqs,
    int64_t num_query_heads,
    bool scratch_in_fp16,
    bool log_f_out_fp32);

void copy_log_f_lastn1_scratch_scalar_cuda(
    torch::Tensor scratch_capture_scores,
    int64_t scratch_row,
    int64_t effective_kv_len,
    torch::Tensor out_capture_scores,
    int64_t capture_row,
    torch::Tensor out_log_f_denoms,
    bool log_f_out_fp32);

void reduce_log_f_pre_scratch_scalar_cuda(
    torch::Tensor scratch_capture_scores,
    int64_t scratch_row,
    int64_t effective_kv_len,
    int64_t last_n,
    torch::Tensor out_capture_scores,
    int64_t capture_row,
    torch::Tensor out_log_f_denoms,
    bool log_f_out_fp32,
    double alpha);

std::vector<torch::Tensor> fused_log_f_prior_logits(
    torch::Tensor scores,
    torch::Tensor row_lo,
    torch::Tensor row_hi,
    torch::Tensor key_norms,
    double alpha,
    double eps,
    double gamma,
    double prior_weight_l2,
    double prior_weight_pos,
    double prior_pos_power,
    double prior_pos_eta,
    double beta,
    double lambda_clip_single,
    double lambda_clip_multi,
    double lambda_tail_kappa,
    double lambda_tail_pivot,
    bool lambda_soft) {
    return fused_log_f_prior_logits_cuda(
        scores,
        row_lo,
        row_hi,
        key_norms,
        alpha,
        eps,
        gamma,
        prior_weight_l2,
        prior_weight_pos,
        prior_pos_power,
        prior_pos_eta,
        beta,
        lambda_clip_single,
        lambda_clip_multi,
        lambda_tail_kappa,
        lambda_tail_pivot,
        lambda_soft);
}

std::vector<torch::Tensor> fused_log_f_prior_pre_denom(
    torch::Tensor scores,
    torch::Tensor denom,
    torch::Tensor row_lo,
    torch::Tensor row_hi,
    torch::Tensor key_norms,
    double alpha,
    double eps,
    double gamma,
    double prior_weight_l2,
    double prior_weight_pos,
    double prior_pos_power,
    double prior_pos_eta,
    double beta,
    double lambda_clip_single,
    double lambda_clip_multi,
    double lambda_tail_kappa,
    double lambda_tail_pivot,
    bool lambda_soft) {
    return fused_log_f_prior_pre_denom_cuda(
        scores,
        denom,
        row_lo,
        row_hi,
        key_norms,
        alpha,
        eps,
        gamma,
        prior_weight_l2,
        prior_weight_pos,
        prior_pos_power,
        prior_pos_eta,
        beta,
        lambda_clip_single,
        lambda_clip_multi,
        lambda_tail_kappa,
        lambda_tail_pivot,
        lambda_soft);
}

void reduce_log_f_pre_scratch(
    torch::Tensor req_meta_i32,
    torch::Tensor req_meta_i64,
    int64_t num_seqs,
    int64_t num_query_heads,
    bool scratch_in_fp16,
    bool log_f_out_fp32,
    double alpha) {
    reduce_log_f_pre_scratch_cuda(
        req_meta_i32,
        req_meta_i64,
        num_seqs,
        num_query_heads,
        scratch_in_fp16,
        log_f_out_fp32,
        alpha);
}

void copy_log_f_lastn1_scratch(
    torch::Tensor req_meta_i32,
    torch::Tensor req_meta_i64,
    int64_t num_seqs,
    int64_t num_query_heads,
    bool scratch_in_fp16,
    bool log_f_out_fp32) {
    copy_log_f_lastn1_scratch_cuda(
        req_meta_i32,
        req_meta_i64,
        num_seqs,
        num_query_heads,
        scratch_in_fp16,
        log_f_out_fp32);
}

void copy_log_f_lastn1_scratch_scalar(
    torch::Tensor scratch_capture_scores,
    int64_t scratch_row,
    int64_t effective_kv_len,
    torch::Tensor out_capture_scores,
    int64_t capture_row,
    torch::Tensor out_log_f_denoms,
    bool log_f_out_fp32) {
    copy_log_f_lastn1_scratch_scalar_cuda(
        scratch_capture_scores,
        scratch_row,
        effective_kv_len,
        out_capture_scores,
        capture_row,
        out_log_f_denoms,
        log_f_out_fp32);
}

void reduce_log_f_pre_scratch_scalar(
    torch::Tensor scratch_capture_scores,
    int64_t scratch_row,
    int64_t effective_kv_len,
    int64_t last_n,
    torch::Tensor out_capture_scores,
    int64_t capture_row,
    torch::Tensor out_log_f_denoms,
    bool log_f_out_fp32,
    double alpha) {
    reduce_log_f_pre_scratch_scalar_cuda(
        scratch_capture_scores,
        scratch_row,
        effective_kv_len,
        last_n,
        out_capture_scores,
        capture_row,
        out_log_f_denoms,
        log_f_out_fp32,
        alpha);
}

bool reduce_log_f_pre_scratch_accum_supported() {
    return true;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_log_f_prior_logits", &fused_log_f_prior_logits,
          "Fused log_f + prior (logits path)");
    m.def("fused_log_f_prior_pre_denom", &fused_log_f_prior_pre_denom,
          "Fused log_f + prior (log_f_pre + denom path)");
    m.def("reduce_log_f_pre_scratch", &reduce_log_f_pre_scratch,
          "Reduce FA scratch logits into log_f_pre + denom");
    m.def("reduce_log_f_pre_scratch_accum_supported",
          &reduce_log_f_pre_scratch_accum_supported,
          "Marker: reduce supports chunked-prefill accumulate meta "
          "(flags bit4 routes crossing pieces; i32 cols 8/9 = prev rows/capacity)");
    m.def("copy_log_f_lastn1_scratch", &copy_log_f_lastn1_scratch,
          "Copy last_n==1 FA scratch logits into capture output");
    m.def("copy_log_f_lastn1_scratch_scalar", &copy_log_f_lastn1_scratch_scalar,
          "Copy one last_n==1 FA scratch row into capture output");
    m.def("reduce_log_f_pre_scratch_scalar", &reduce_log_f_pre_scratch_scalar,
          "Reduce one FA scratch row into log_f_pre + denom");
}
"""

    cuda_source = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <cmath>

namespace {

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xffffffff, val, offset);
        val = val > other ? val : other;
    }
    return val;
}

__device__ __forceinline__ float block_reduce_sum(float val) {
    static __shared__ float shared[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    val = warp_reduce_sum(val);
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();
    float out = 0.0f;
    if (wid == 0) {
        out = (lane < ((blockDim.x + 31) >> 5)) ? shared[lane] : 0.0f;
        out = warp_reduce_sum(out);
    }
    __syncthreads();
    if (wid == 0 && lane == 0) {
        shared[0] = out;
    }
    __syncthreads();
    return shared[0];
}

__device__ __forceinline__ float block_reduce_max(float val) {
    static __shared__ float shared[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    val = warp_reduce_max(val);
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();
    float out = -INFINITY;
    if (wid == 0) {
        out = (lane < ((blockDim.x + 31) >> 5)) ? shared[lane] : -INFINITY;
        out = warp_reduce_max(out);
    }
    __syncthreads();
    if (wid == 0 && lane == 0) {
        shared[0] = out;
    }
    __syncthreads();
    return shared[0];
}

__device__ __forceinline__ float safe_log(float x, float eps) {
    return logf(fmaxf(x, eps));
}

__device__ __forceinline__ float sigmoid_tanh_clip(float x) {
    float x_abs = fabsf(x);
    float e = expf(-2.0f * x_abs);
    float tanh_abs = (1.0f - e) / (1.0f + e);
    return x >= 0.0f ? tanh_abs : -tanh_abs;
}

} // namespace


template <typename scalar_t>
__global__ void fused_log_f_prior_kernel(
    const scalar_t* __restrict__ scores,
    const float* __restrict__ denom,
    const int32_t* __restrict__ row_lo,
    const int32_t* __restrict__ row_hi,
    const float* __restrict__ key_norms,
    int M,
    int R,
    int K,
    int stride_scores_m,
    int stride_scores_r,
    int stride_scores_k,
    int stride_denom_m,
    int stride_denom_r,
    int stride_lo_m,
    int stride_lo_r,
    int stride_hi_m,
    int stride_hi_r,
    int stride_kn_m,
    int stride_kn_k,
    bool use_denom,
    float alpha,
    float eps,
    float gamma,
    float prior_weight_l2,
    float prior_weight_pos,
    float prior_pos_power,
    float prior_pos_eta,
    float beta,
    float lambda_clip_single,
    float lambda_clip_multi,
    float lambda_tail_kappa,
    float lambda_tail_pivot,
    bool lambda_soft,
    float* __restrict__ log_r_cache,
    float* __restrict__ out) {
    int m = blockIdx.x;
    if (m >= M) {
        return;
    }

    extern __shared__ float shm[]; // dynamic shared
    float* row_lse_s = shm;                    // size R
    int* row_valid_s = (int*)(row_lse_s + R);  // size R
    int* row_lo_s = row_valid_s + R;           // size R
    int* row_hi_s = row_lo_s + R;              // size R
    int* token_bounds_s = row_hi_s + R;        // size 2 (lo, hi)
    float* lambda_s = (float*)(token_bounds_s + 2); // size 2 (log_one_minus, log_lambda)

    int tid = threadIdx.x;
    for (int r = tid; r < R; r += blockDim.x) {
        int32_t lo = row_lo[m * stride_lo_m + r * stride_lo_r];
        int32_t hi = row_hi[m * stride_hi_m + r * stride_hi_r];
        row_lo_s[r] = lo;
        row_hi_s[r] = hi;
    }
    __syncthreads();

    int token_lo = 0;
    int token_hi = 0;
    if (tid == 0) {
        int lo_min = row_lo_s[0];
        int hi_max = row_hi_s[0];
        for (int r = 1; r < R; ++r) {
            int lo = row_lo_s[r];
            int hi = row_hi_s[r];
            if (lo < lo_min) {
                lo_min = lo;
            }
            if (hi > hi_max) {
                hi_max = hi;
            }
        }
        if (lo_min < 0) {
            lo_min = 0;
        }
        if (hi_max > K) {
            hi_max = K;
        }
        token_lo = lo_min;
        token_hi = hi_max;
        token_bounds_s[0] = token_lo;
        token_bounds_s[1] = token_hi;
    }
    __syncthreads();
    token_lo = token_bounds_s[0];
    token_hi = token_bounds_s[1];

    float row_count = 0.0f;
    const float min_val = -INFINITY;
    if (!use_denom) {
        for (int r = 0; r < R; ++r) {
            float local_max = min_val;
            int has_val = 0;
            int lo = row_lo_s[r];
            int hi = row_hi_s[r];
            for (int k = tid; k < K; k += blockDim.x) {
                if (k < lo || k >= hi) {
                    continue;
                }
                float val = static_cast<float>(scores[m * stride_scores_m + r * stride_scores_r + k * stride_scores_k]);
                if (isfinite(val)) {
                    if (val > local_max) {
                        local_max = val;
                    }
                    has_val = 1;
                }
            }
            float max_val = block_reduce_max(local_max);
            int row_has = (max_val > min_val) && (has_val || (tid == 0));
            float sum_exp = 0.0f;
            if (max_val > min_val) {
                for (int k = tid; k < K; k += blockDim.x) {
                    if (k < lo || k >= hi) {
                        continue;
                    }
                    float val = static_cast<float>(scores[m * stride_scores_m + r * stride_scores_r + k * stride_scores_k]);
                    if (!isfinite(val)) {
                        continue;
                    }
                    sum_exp += expf(val - max_val);
                }
            }
            sum_exp = block_reduce_sum(sum_exp);
            if (tid == 0) {
                if (max_val > min_val && sum_exp > 0.0f && row_has) {
                    row_lse_s[r] = max_val + logf(sum_exp);
                    row_valid_s[r] = 1;
                } else {
                    row_lse_s[r] = min_val;
                    row_valid_s[r] = 0;
                }
            }
        }
        __syncthreads();
    } else {
        for (int r = tid; r < R; r += blockDim.x) {
            int lo = row_lo_s[r];
            int hi = row_hi_s[r];
            row_valid_s[r] = (hi > lo) ? 1 : 0;
        }
        __syncthreads();
    }

    float local_rows = 0.0f;
    for (int r = tid; r < R; r += blockDim.x) {
        local_rows += static_cast<float>(row_valid_s[r]);
    }
    row_count = block_reduce_sum(local_rows);
    __shared__ float row_count_s;
    if (tid == 0) {
        row_count_s = row_count;
    }
    __syncthreads();
    row_count = row_count_s;
    if (row_count <= 0.0f) {
        for (int k = tid; k < K; k += blockDim.x) {
            out[m * K + k] = min_val;
        }
        return;
    }

    const bool use_mean = fabsf(alpha) < 1.0e-6f;
    const float prior_pos_power_f = prior_pos_power < 1.0f ? 1.0f : prior_pos_power;
    __shared__ float denom_pos_s;
    __shared__ float inv_row_count_s;
    __shared__ float log_row_count_s;
    if (tid == 0) {
        int denom_pos_i = token_hi - token_lo - 1;
        if (denom_pos_i < 1) {
            denom_pos_i = 1;
        }
        denom_pos_s = static_cast<float>(denom_pos_i);
        inv_row_count_s = 1.0f / row_count;
        log_row_count_s = logf(row_count);
    }
    __syncthreads();
    float denom_pos = denom_pos_s;
    float inv_row_count = inv_row_count_s;
    float log_row_count = log_row_count_s;

    // Pass 1: denom_f (from log_f) and denom_r (from log_r_raw) via streaming logsumexp
    __shared__ float max_f_s;
    __shared__ float sum_f_s;
    __shared__ float max_r_s;
    __shared__ float sum_r_s;
    if (tid == 0) {
        max_f_s = min_val;
        sum_f_s = 0.0f;
        max_r_s = min_val;
        sum_r_s = 0.0f;
    }
    __syncthreads();

    for (int block_start = 0; block_start < K; block_start += blockDim.x) {
        int k = block_start + tid;
        float log_f = min_val;
        float log_r_raw = min_val;
        bool in_bounds = (k >= token_lo) && (k < token_hi) && (k < K);
        if (in_bounds) {
            if (use_mean) {
                float log_sum = 0.0f;
                for (int r = 0; r < R; ++r) {
                    if (!row_valid_s[r]) {
                        continue;
                    }
                    int lo = row_lo_s[r];
                    int hi = row_hi_s[r];
                    if (k < lo || k >= hi) {
                        continue;
                    }
                    float lp;
                    if (use_denom) {
                        float lfp = static_cast<float>(scores[m * stride_scores_m + r * stride_scores_r + k * stride_scores_k]);
                        float den = denom[m * stride_denom_m + r * stride_denom_r];
                        lp = lfp - den;
                    } else {
                        float val = static_cast<float>(scores[m * stride_scores_m + r * stride_scores_r + k * stride_scores_k]);
                        if (!isfinite(val)) {
                            continue;
                        }
                        lp = val - row_lse_s[r];
                    }
                    if (lp > min_val) {
                        log_sum += lp;
                    }
                }
                log_f = log_sum * inv_row_count;
            } else {
                float log_f_max = min_val;
                float log_f_sum = 0.0f;
                for (int r = 0; r < R; ++r) {
                    if (!row_valid_s[r]) {
                        continue;
                    }
                    int lo = row_lo_s[r];
                    int hi = row_hi_s[r];
                    if (k < lo || k >= hi) {
                        continue;
                    }
                    float lp;
                    if (use_denom) {
                        float lfp = static_cast<float>(scores[m * stride_scores_m + r * stride_scores_r + k * stride_scores_k]);
                        float den = denom[m * stride_denom_m + r * stride_denom_r];
                        lp = lfp - den;
                    } else {
                        float val = static_cast<float>(scores[m * stride_scores_m + r * stride_scores_r + k * stride_scores_k]);
                        if (!isfinite(val)) {
                            continue;
                        }
                        lp = val - row_lse_s[r];
                    }
                    if (lp <= min_val) {
                        continue;
                    }
                    float log_f_pre = alpha * lp;
                    if (log_f_pre > log_f_max) {
                        log_f_sum = log_f_sum * expf(log_f_max - log_f_pre) + 1.0f;
                        log_f_max = log_f_pre;
                    } else {
                        log_f_sum += expf(log_f_pre - log_f_max);
                    }
                }
                if (log_f_max > min_val) {
                    float lse = log_f_max + logf(log_f_sum);
                    log_f = (lse - log_row_count) / alpha;
                }
            }

            float kn = key_norms[m * stride_kn_m + k * stride_kn_k];
            kn = fmaxf(kn, eps);
            float log_pi = -gamma * logf(kn);
            float pos_norm = (static_cast<float>(k - token_lo)) / denom_pos;
            pos_norm = fmaxf(0.0f, fminf(1.0f, pos_norm));
            float pos_norm_pos = fmaxf(pos_norm, eps);
            float pos_shaped = expf(prior_pos_power_f * logf(pos_norm_pos));
            if (pos_norm <= 0.0f) {
                pos_shaped = 0.0f;
            }
            float base_delta = -beta * pos_shaped;
            float one_minus = fmaxf(1.0f - pos_norm, eps);
            base_delta = base_delta + prior_pos_eta * logf(one_minus);
            float log_delta = base_delta;
            log_r_raw = prior_weight_l2 * log_pi + prior_weight_pos * log_delta;
        }

        if (k < K) {
            out[m * K + k] = in_bounds ? log_f : min_val;
            if (log_r_cache != nullptr) {
                log_r_cache[m * K + k] = log_r_raw;
            }
        }

        float block_max_f = block_reduce_max(log_f);
        float block_max_r = block_reduce_max(log_r_raw);

        float max_f = max_f_s;
        float sum_f = sum_f_s;
        float max_r = max_r_s;
        float sum_r = sum_r_s;

        float new_max_f = fmaxf(max_f, block_max_f);
        float new_max_f_safe = (new_max_f == min_val) ? 0.0f : new_max_f;
        float max_f_safe = (max_f == min_val) ? 0.0f : max_f;
        float scale_old_f = (max_f == min_val) ? 0.0f : expf(max_f_safe - new_max_f_safe);
        float exp_f = (log_f > min_val && new_max_f > min_val) ? expf(log_f - new_max_f) : 0.0f;
        float block_sum_f = block_reduce_sum(exp_f);

        float new_max_r = fmaxf(max_r, block_max_r);
        float new_max_r_safe = (new_max_r == min_val) ? 0.0f : new_max_r;
        float max_r_safe = (max_r == min_val) ? 0.0f : max_r;
        float scale_old_r = (max_r == min_val) ? 0.0f : expf(max_r_safe - new_max_r_safe);
        float exp_r = (log_r_raw > min_val && new_max_r > min_val) ? expf(log_r_raw - new_max_r) : 0.0f;
        float block_sum_r = block_reduce_sum(exp_r);

        if (tid == 0) {
            sum_f = sum_f * scale_old_f + block_sum_f;
            max_f = new_max_f;
            sum_r = sum_r * scale_old_r + block_sum_r;
            max_r = new_max_r;
            max_f_s = max_f;
            sum_f_s = sum_f;
            max_r_s = max_r;
            sum_r_s = sum_r;
        }
        __syncthreads();
    }

    float max_f = max_f_s;
    float sum_f = sum_f_s;
    float max_r = max_r_s;
    float sum_r = sum_r_s;
    float denom_f = (max_f > min_val) ? (max_f + logf(fmaxf(sum_f, eps))) : min_val;
    float denom_r = (max_r > min_val) ? (max_r + logf(fmaxf(sum_r, eps))) : min_val;

    // Pass 2: ff/rr/fr and tail stats
    float local_ff = 0.0f;
    float local_rr = 0.0f;
    float local_fr = 0.0f;
    float local_prob_sum = 0.0f;
    float local_pos_sum = 0.0f;

    for (int k = tid; k < K; k += blockDim.x) {
        if (k < token_lo || k >= token_hi) {
            continue;
        }
        float log_f = out[m * K + k];
        float lp = min_val;
        if (log_f > min_val && denom_f > min_val) {
            lp = log_f - denom_f;
        }

        float pos_norm = (static_cast<float>(k - token_lo)) / denom_pos;
        pos_norm = fmaxf(0.0f, fminf(1.0f, pos_norm));
        float log_r_raw = min_val;
        if (log_r_cache != nullptr) {
            log_r_raw = log_r_cache[m * K + k];
        } else {
            float kn = key_norms[m * stride_kn_m + k * stride_kn_k];
            kn = fmaxf(kn, eps);
            float log_pi = -gamma * logf(kn);
            float pos_norm_pos = fmaxf(pos_norm, eps);
            float pos_shaped = expf(prior_pos_power_f * logf(pos_norm_pos));
            if (pos_norm <= 0.0f) {
                pos_shaped = 0.0f;
            }
            float base_delta = -beta * pos_shaped;
            float one_minus = fmaxf(1.0f - pos_norm, eps);
            base_delta = base_delta + prior_pos_eta * logf(one_minus);
            float log_delta = base_delta;
            log_r_raw = prior_weight_l2 * log_pi + prior_weight_pos * log_delta;
        }
        float log_r = log_r_raw - denom_r;

        local_ff += expf(2.0f * lp);
        local_rr += expf(2.0f * log_r);
        local_fr += expf(lp + log_r);

        if (lambda_tail_kappa > 0.0f) {
            float p = expf(lp);
            local_prob_sum += p;
            local_pos_sum += p * pos_norm;
        }
    }

    float ff = block_reduce_sum(local_ff);
    float rr = block_reduce_sum(local_rr);
    float fr = block_reduce_sum(local_fr);
    float prob_sum = block_reduce_sum(local_prob_sum);
    float pos_sum = block_reduce_sum(local_pos_sum);

    float lam = 0.0f;
    if (tid == 0) {
        float denom_lam = fmaxf(ff - 2.0f * fr + rr, eps);
        float lam_star = (ff - fr) / denom_lam;
        lam = fmaxf(lam_star, 0.0f);
        if (lambda_tail_kappa > 0.0f) {
            float prob_safe = fmaxf(prob_sum, eps);
            float mean_pos = pos_sum / prob_safe;
            float tail_excess = fmaxf(mean_pos - lambda_tail_pivot, 0.0f);
            lam = lam * (1.0f + lambda_tail_kappa * tail_excess);
        }
        float lam_max = (row_count > 1.0f) ? lambda_clip_multi : lambda_clip_single;
        if (lambda_soft) {
            float clip = fmaxf(lam_max, eps);
            float x = lam / clip;
            lam = clip * sigmoid_tanh_clip(x);
        } else {
            lam = fminf(lam, lam_max);
        }
        lam = fmaxf(lam, 0.0f);
        lam = fminf(lam, 1.0f - eps);
        lambda_s[0] = logf(fmaxf(1.0f - lam, eps));
        lambda_s[1] = logf(lam + eps);
    }
    __syncthreads();
    float log_one_minus = lambda_s[0];
    float log_lambda = lambda_s[1];

    // Pass 3: denom_fused (cache fused_raw in out to avoid recompute)
    float local_max_fused = min_val;
    for (int k = tid; k < K; k += blockDim.x) {
        if (k < token_lo || k >= token_hi) {
            continue;
        }
        float log_f = out[m * K + k];
        float lp = min_val;
        if (log_f > min_val && denom_f > min_val) {
            lp = log_f - denom_f;
        }
        float log_r_raw = min_val;
        if (log_r_cache != nullptr) {
            log_r_raw = log_r_cache[m * K + k];
        } else {
            float kn = key_norms[m * stride_kn_m + k * stride_kn_k];
            kn = fmaxf(kn, eps);
            float log_pi = -gamma * logf(kn);
            float pos_norm = (static_cast<float>(k - token_lo)) / denom_pos;
            pos_norm = fmaxf(0.0f, fminf(1.0f, pos_norm));
            float pos_norm_pos = fmaxf(pos_norm, eps);
            float pos_shaped = expf(prior_pos_power_f * logf(pos_norm_pos));
            if (pos_norm <= 0.0f) {
                pos_shaped = 0.0f;
            }
            float base_delta = -beta * pos_shaped;
            float one_minus = fmaxf(1.0f - pos_norm, eps);
            base_delta = base_delta + prior_pos_eta * logf(one_minus);
            float log_delta = base_delta;
            log_r_raw = prior_weight_l2 * log_pi + prior_weight_pos * log_delta;
        }
        float log_r = log_r_raw - denom_r;
        float a = log_one_minus + lp;
        float b = log_lambda + log_r;
        float mval = fmaxf(a, b);
        float fused_raw = mval + logf(expf(a - mval) + expf(b - mval));
        out[m * K + k] = fused_raw;
        if (fused_raw > local_max_fused) {
            local_max_fused = fused_raw;
        }
    }

    float max_fused = block_reduce_max(local_max_fused);
    float local_sum_fused = 0.0f;
    for (int k = tid; k < K; k += blockDim.x) {
        if (k < token_lo || k >= token_hi) {
            continue;
        }
        float fused_raw = out[m * K + k];
        if (fused_raw > min_val && max_fused > min_val) {
            local_sum_fused += expf(fused_raw - max_fused);
        }
    }

    float sum_fused = block_reduce_sum(local_sum_fused);
    float denom_fused = (max_fused > min_val) ? (max_fused + logf(fmaxf(sum_fused, eps))) : min_val;

    // Pass 4: write fused (normalize cached fused_raw)
    for (int k = tid; k < K; k += blockDim.x) {
        if (k < token_lo || k >= token_hi) {
            out[m * K + k] = min_val;
            continue;
        }
        float fused_raw = out[m * K + k];
        float fused = fused_raw - denom_fused;
        out[m * K + k] = fused;
    }
}

std::vector<torch::Tensor> fused_log_f_prior_impl(
    torch::Tensor scores,
    c10::optional<torch::Tensor> denom,
    torch::Tensor row_lo,
    torch::Tensor row_hi,
    torch::Tensor key_norms,
    double alpha,
    double eps,
    double gamma,
    double prior_weight_l2,
    double prior_weight_pos,
    double prior_pos_power,
    double prior_pos_eta,
    double beta,
    double lambda_clip_single,
    double lambda_clip_multi,
    double lambda_tail_kappa,
    double lambda_tail_pivot,
    bool lambda_soft) {
    TORCH_CHECK(scores.is_cuda(), "scores must be CUDA");
    TORCH_CHECK(row_lo.is_cuda() && row_hi.is_cuda(), "row_lo/row_hi must be CUDA");
    TORCH_CHECK(key_norms.is_cuda(), "key_norms must be CUDA");
    TORCH_CHECK(scores.dim() == 3, "scores must be [M, R, K]");
    TORCH_CHECK(row_lo.dim() == 2 && row_hi.dim() == 2, "row_lo/row_hi must be [M, R]");
    TORCH_CHECK(key_norms.dim() == 2, "key_norms must be [M, K]");

    auto scores_c = scores.contiguous();
    auto row_lo_c = row_lo.contiguous();
    auto row_hi_c = row_hi.contiguous();
    auto key_norms_f = key_norms.to(torch::kFloat32).contiguous();

    int64_t M = scores_c.size(0);
    int64_t R = scores_c.size(1);
    int64_t K = scores_c.size(2);

    TORCH_CHECK(row_lo_c.size(0) == M && row_lo_c.size(1) == R, "row_lo shape mismatch");
    TORCH_CHECK(row_hi_c.size(0) == M && row_hi_c.size(1) == R, "row_hi shape mismatch");
    TORCH_CHECK(key_norms_f.size(0) == M && key_norms_f.size(1) == K, "key_norms shape mismatch");
    TORCH_CHECK(R <= 128, "rows (R) must be <=128");

    auto out = torch::empty({M, K}, scores_c.options().dtype(torch::kFloat32));
    torch::Tensor log_r_cache;
    float* log_r_cache_ptr = nullptr;
    if (const char* env = std::getenv("VLLM_SPARSE_SELECTOR_LOGS_CACHE_R")) {
        if (std::atoi(env) == 1) {
            log_r_cache = torch::empty({M, K}, scores_c.options().dtype(torch::kFloat32));
            log_r_cache_ptr = log_r_cache.data_ptr<float>();
        }
    }

    int threads = 256;
    if (const char* env = std::getenv("VLLM_SPARSE_SELECTOR_LOGS_THREADS")) {
        int parsed = std::atoi(env);
        if (parsed >= 64 && parsed <= 1024) {
            threads = parsed;
        }
    } else {
        if (K >= 8192) {
            threads = (R <= 16) ? 768 : 512;
        }
    }
    const dim3 blocks(M);
    size_t shm_size = sizeof(float) * R + sizeof(int) * R + sizeof(int) * R + sizeof(int) * R
        + sizeof(int) * 2 + sizeof(float) * 2;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (denom.has_value()) {
        auto denom_f = denom.value().to(torch::kFloat32).contiguous();
        TORCH_CHECK(denom_f.dim() == 2, "denom must be [M, R]");
        TORCH_CHECK(denom_f.size(0) == M && denom_f.size(1) == R, "denom shape mismatch");
        AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, scores_c.scalar_type(), "fused_log_f_prior_pre_denom", [&] {
            fused_log_f_prior_kernel<scalar_t><<<blocks, threads, shm_size, stream>>>(
                scores_c.data_ptr<scalar_t>(),
                denom_f.data_ptr<float>(),
                row_lo_c.data_ptr<int32_t>(),
                row_hi_c.data_ptr<int32_t>(),
                key_norms_f.data_ptr<float>(),
                static_cast<int>(M),
                static_cast<int>(R),
                static_cast<int>(K),
                static_cast<int>(scores_c.stride(0)),
                static_cast<int>(scores_c.stride(1)),
                static_cast<int>(scores_c.stride(2)),
                static_cast<int>(denom_f.stride(0)),
                static_cast<int>(denom_f.stride(1)),
                static_cast<int>(row_lo_c.stride(0)),
                static_cast<int>(row_lo_c.stride(1)),
                static_cast<int>(row_hi_c.stride(0)),
                static_cast<int>(row_hi_c.stride(1)),
                static_cast<int>(key_norms_f.stride(0)),
                static_cast<int>(key_norms_f.stride(1)),
                true,
                static_cast<float>(alpha),
                static_cast<float>(eps),
                static_cast<float>(gamma),
                static_cast<float>(prior_weight_l2),
                static_cast<float>(prior_weight_pos),
                static_cast<float>(prior_pos_power),
                static_cast<float>(prior_pos_eta),
                static_cast<float>(beta),
                static_cast<float>(lambda_clip_single),
                static_cast<float>(lambda_clip_multi),
                static_cast<float>(lambda_tail_kappa),
                static_cast<float>(lambda_tail_pivot),
                lambda_soft,
                log_r_cache_ptr,
                out.data_ptr<float>());
        });
    } else {
        AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, scores_c.scalar_type(), "fused_log_f_prior_logits", [&] {
            fused_log_f_prior_kernel<scalar_t><<<blocks, threads, shm_size, stream>>>(
                scores_c.data_ptr<scalar_t>(),
                nullptr,
                row_lo_c.data_ptr<int32_t>(),
                row_hi_c.data_ptr<int32_t>(),
                key_norms_f.data_ptr<float>(),
                static_cast<int>(M),
                static_cast<int>(R),
                static_cast<int>(K),
                static_cast<int>(scores_c.stride(0)),
                static_cast<int>(scores_c.stride(1)),
                static_cast<int>(scores_c.stride(2)),
                0,
                0,
                static_cast<int>(row_lo_c.stride(0)),
                static_cast<int>(row_lo_c.stride(1)),
                static_cast<int>(row_hi_c.stride(0)),
                static_cast<int>(row_hi_c.stride(1)),
                static_cast<int>(key_norms_f.stride(0)),
                static_cast<int>(key_norms_f.stride(1)),
                false,
                static_cast<float>(alpha),
                static_cast<float>(eps),
                static_cast<float>(gamma),
                static_cast<float>(prior_weight_l2),
                static_cast<float>(prior_weight_pos),
                static_cast<float>(prior_pos_power),
                static_cast<float>(prior_pos_eta),
                static_cast<float>(beta),
                static_cast<float>(lambda_clip_single),
                static_cast<float>(lambda_clip_multi),
                static_cast<float>(lambda_tail_kappa),
                static_cast<float>(lambda_tail_pivot),
                lambda_soft,
                log_r_cache_ptr,
                out.data_ptr<float>());
        });
    }

    return {out};
}

std::vector<torch::Tensor> fused_log_f_prior_logits_cuda(
    torch::Tensor scores,
    torch::Tensor row_lo,
    torch::Tensor row_hi,
    torch::Tensor key_norms,
    double alpha,
    double eps,
    double gamma,
    double prior_weight_l2,
    double prior_weight_pos,
    double prior_pos_power,
    double prior_pos_eta,
    double beta,
    double lambda_clip_single,
    double lambda_clip_multi,
    double lambda_tail_kappa,
    double lambda_tail_pivot,
    bool lambda_soft) {
    return fused_log_f_prior_impl(scores, c10::nullopt, row_lo, row_hi, key_norms,
                                  alpha, eps, gamma, prior_weight_l2, prior_weight_pos,
                                  prior_pos_power, prior_pos_eta, beta, lambda_clip_single,
                                  lambda_clip_multi, lambda_tail_kappa, lambda_tail_pivot, lambda_soft);
}

std::vector<torch::Tensor> fused_log_f_prior_pre_denom_cuda(
    torch::Tensor scores,
    torch::Tensor denom,
    torch::Tensor row_lo,
    torch::Tensor row_hi,
    torch::Tensor key_norms,
    double alpha,
    double eps,
    double gamma,
    double prior_weight_l2,
    double prior_weight_pos,
    double prior_pos_power,
    double prior_pos_eta,
    double beta,
    double lambda_clip_single,
    double lambda_clip_multi,
    double lambda_tail_kappa,
    double lambda_tail_pivot,
    bool lambda_soft) {
    return fused_log_f_prior_impl(scores, denom, row_lo, row_hi, key_norms,
                                  alpha, eps, gamma, prior_weight_l2, prior_weight_pos,
                                  prior_pos_power, prior_pos_eta, beta, lambda_clip_single,
                                  lambda_clip_multi, lambda_tail_kappa, lambda_tail_pivot, lambda_soft);
}

constexpr int kLogFPreMaxR = 16;
constexpr float kLogFPreMinVal = -3.402823466e38f;

template <typename out_t>
__device__ __forceinline__ void store_log_f_pre_value(out_t* ptr, float value);

template <>
__device__ __forceinline__ void store_log_f_pre_value<float>(float* ptr, float value) {
    *ptr = value;
}

template <>
__device__ __forceinline__ void store_log_f_pre_value<half>(half* ptr, float value) {
    *ptr = __float2half_rn(value);
}

template <typename scratch_t>
__device__ __forceinline__ float load_scratch_value(const scratch_t* ptr);

template <>
__device__ __forceinline__ float load_scratch_value<float>(const float* ptr) {
    return *ptr;
}

template <>
__device__ __forceinline__ float load_scratch_value<half>(const half* ptr) {
    return __half2float(*ptr);
}

template <typename scratch_t>
__device__ __forceinline__ float compute_log_f_pre_token(
    const scratch_t* scratch_head_ptr,
    int64_t stride_row,
    int token_idx,
    int max_r,
    const float* row_lse,
    const int* row_has,
    float row_count_f,
    float alpha,
    bool use_mean,
    bool* has_token) {
    *has_token = false;
    if (use_mean) {
        float acc = 0.0f;
        for (int r = 0; r < max_r; ++r) {
            if (row_has[r] == 0) {
                continue;
            }
            float v = load_scratch_value(scratch_head_ptr + static_cast<int64_t>(r) * stride_row + token_idx);
            if (v > kLogFPreMinVal) {
                acc += (v - row_lse[r]);
                *has_token = true;
            }
        }
        return *has_token ? (acc / row_count_f) : kLogFPreMinVal;
    }

    float amax = kLogFPreMinVal;
    for (int r = 0; r < max_r; ++r) {
        if (row_has[r] == 0) {
            continue;
        }
            float v = load_scratch_value(scratch_head_ptr + static_cast<int64_t>(r) * stride_row + token_idx);
        if (v > kLogFPreMinVal) {
            float a = alpha * (v - row_lse[r]);
            amax = amax > a ? amax : a;
            *has_token = true;
        }
    }
    if (!*has_token) {
        return kLogFPreMinVal;
    }

    float asum = 0.0f;
    for (int r = 0; r < max_r; ++r) {
        if (row_has[r] == 0) {
            continue;
        }
            float v = load_scratch_value(scratch_head_ptr + static_cast<int64_t>(r) * stride_row + token_idx);
        if (v > kLogFPreMinVal) {
            float a = alpha * (v - row_lse[r]);
            asum += expf(a - amax);
        }
    }
    return (amax + logf(asum + 1.0e-20f) - logf(row_count_f)) / alpha;
}

// 跨片 accumulate 用：返回未归一化的 sum-form。
// use_mean: Σ_r (v_rj − LSE_r)（未除计数）；否则 log Σ_r exp(α(v_rj − LSE_r))
// （未减 log n、未除 α）。行 LSE 只依赖该行自身因果可见键，故 sum-form 跨片可加。
template <typename scratch_t>
__device__ __forceinline__ float compute_log_f_pre_token_unnormalized(
    const scratch_t* scratch_head_ptr,
    int64_t stride_row,
    int token_idx,
    int max_r,
    const float* row_lse,
    const int* row_has,
    float alpha,
    bool use_mean,
    bool* has_token) {
    *has_token = false;
    if (use_mean) {
        float acc = 0.0f;
        for (int r = 0; r < max_r; ++r) {
            if (row_has[r] == 0) {
                continue;
            }
            float v = load_scratch_value(scratch_head_ptr + static_cast<int64_t>(r) * stride_row + token_idx);
            if (v > kLogFPreMinVal) {
                acc += (v - row_lse[r]);
                *has_token = true;
            }
        }
        return *has_token ? acc : kLogFPreMinVal;
    }

    float amax = kLogFPreMinVal;
    for (int r = 0; r < max_r; ++r) {
        if (row_has[r] == 0) {
            continue;
        }
        float v = load_scratch_value(scratch_head_ptr + static_cast<int64_t>(r) * stride_row + token_idx);
        if (v > kLogFPreMinVal) {
            float a = alpha * (v - row_lse[r]);
            amax = amax > a ? amax : a;
            *has_token = true;
        }
    }
    if (!*has_token) {
        return kLogFPreMinVal;
    }

    float asum = 0.0f;
    for (int r = 0; r < max_r; ++r) {
        if (row_has[r] == 0) {
            continue;
        }
        float v = load_scratch_value(scratch_head_ptr + static_cast<int64_t>(r) * stride_row + token_idx);
        if (v > kLogFPreMinVal) {
            float a = alpha * (v - row_lse[r]);
            asum += expf(a - amax);
        }
    }
    return amax + logf(asum + 1.0e-20f);
}

// 合并前片（out 里已存的归一化部分窗口值 prev，来自 n_prev 行）与本片
// 未归一化 sum-form（cur，来自 n_cur 行），返回 n_prev+n_cur 行的归一化值。
// prev 的归一化用的是全局行数 n_prev（非 per-key 贡献数），故 α·prev+log(n_prev)
// 精确还原 per-key 的 log S_prev；合并数学等价于对全部行做单次 reduce。
template <typename out_t>
__device__ __forceinline__ float merge_log_f_pre_accum(
    const out_t* out_ptr,
    int token_idx,
    int prev_capacity,
    float n_prev_f,
    float cur_unnormalized,
    bool has_cur,
    float n_cum_f,
    float alpha,
    bool use_mean,
    bool* has_any) {
    bool has_prev = false;
    float prev = kLogFPreMinVal;
    if (token_idx < prev_capacity) {
        const float p = load_scratch_value(out_ptr + token_idx);
        if (isfinite(p) && p > kLogFPreMinVal) {
            has_prev = true;
            prev = p;
        }
    }
    *has_any = has_prev || has_cur;
    if (!*has_any) {
        return kLogFPreMinVal;
    }
    if (use_mean) {
        const float acc = (has_prev ? prev * n_prev_f : 0.0f)
            + (has_cur ? cur_unnormalized : 0.0f);
        return acc / n_cum_f;
    }
    if (has_prev && has_cur) {
        const float prev_log_s = alpha * prev + logf(n_prev_f);
        const float m = prev_log_s > cur_unnormalized ? prev_log_s : cur_unnormalized;
        const float s = expf(prev_log_s - m) + expf(cur_unnormalized - m);
        return (m + logf(s) - logf(n_cum_f)) / alpha;
    }
    if (has_prev) {
        const float prev_log_s = alpha * prev + logf(n_prev_f);
        return (prev_log_s - logf(n_cum_f)) / alpha;
    }
    return (cur_unnormalized - logf(n_cum_f)) / alpha;
}

template <typename scratch_t, typename out_t>
__global__ void reduce_log_f_pre_scratch_kernel(
    const int32_t* __restrict__ req_meta_i32,
    const int64_t* __restrict__ req_meta_i64,
    int64_t req_meta_i32_stride_row,
    int64_t req_meta_i32_stride_col,
    int64_t req_meta_i64_stride_row,
    int64_t req_meta_i64_stride_col,
    int num_seqs,
    int num_query_heads,
    float alpha,
    bool use_mean) {
    const int pid_seq = blockIdx.x;
    const int pid_h = blockIdx.y;
    if (pid_seq >= num_seqs || pid_h >= num_query_heads) {
        return;
    }

    const int32_t* meta32_row_ptr = req_meta_i32 + static_cast<int64_t>(pid_seq) * req_meta_i32_stride_row;
    const int64_t* meta64_row_ptr = req_meta_i64 + static_cast<int64_t>(pid_seq) * req_meta_i64_stride_row;

    const int scratch_stride_head_i32 = meta32_row_ptr[1 * req_meta_i32_stride_col];
    const int logits_last_n = meta32_row_ptr[2 * req_meta_i32_stride_col];
    const int logits_capacity = meta32_row_ptr[4 * req_meta_i32_stride_col];
    const int flags = meta32_row_ptr[5 * req_meta_i32_stride_col];
    const int out_stride_head_i32 = meta32_row_ptr[6 * req_meta_i32_stride_col];
    const int scratch_stride_row_i32 = meta32_row_ptr[7 * req_meta_i32_stride_col];

    const bool use_compact = (flags & 1) != 0;
    const bool use_log_f = ((flags & 8) != 0) && !use_compact;
    // flags bit4 = 跨片 accumulate 协议行（chunked prefill 捕获窗跨 chunk）：
    // 允许 last_n==1 走本 kernel（尾片 q_len==1 必须 v−LSE 归一而非裸拷贝），
    // 且 meta 携带 cols 8/9 = 前片累计行数/累计 kv 宽。未置位的行（黄金/refresh
    // /非跨片）不读新列、走原路径，逐位不变。
    const bool accum_mode = (flags & 16) != 0;
    if (!use_log_f || logits_capacity <= 0
        || scratch_stride_head_i32 <= 0 || scratch_stride_row_i32 <= 0
        || logits_last_n < 1 || (logits_last_n == 1 && !accum_mode)) {
        return;
    }
    int accum_prev_rows = 0;
    int accum_prev_capacity = 0;
    if (accum_mode) {
        accum_prev_rows = meta32_row_ptr[8 * req_meta_i32_stride_col];
        accum_prev_capacity = meta32_row_ptr[9 * req_meta_i32_stride_col];
    }
    // 首个跨片片 prev_rows==0：无可合并，但仍须本 kernel 的归一化数学。
    const bool accum_do_merge =
        accum_mode && accum_prev_rows > 0 && accum_prev_capacity > 0;

    const int64_t scratch_base_ptr = meta64_row_ptr[1 * req_meta_i64_stride_col];
    const int64_t out_base_ptr = meta64_row_ptr[2 * req_meta_i64_stride_col];
    const int64_t denom_base_ptr = meta64_row_ptr[3 * req_meta_i64_stride_col];
    if (scratch_base_ptr == 0 || out_base_ptr == 0 || denom_base_ptr == 0) {
        return;
    }

    const int64_t stride_row = static_cast<int64_t>(scratch_stride_row_i32);
    const int max_r = logits_last_n < kLogFPreMaxR ? logits_last_n : kLogFPreMaxR;
    const int64_t stride_head_scratch = static_cast<int64_t>(scratch_stride_head_i32);
    const int64_t stride_head_out = out_stride_head_i32 > 0
        ? static_cast<int64_t>(out_stride_head_i32)
        : static_cast<int64_t>(logits_capacity);

    const scratch_t* scratch_head_ptr =
        reinterpret_cast<const scratch_t*>(scratch_base_ptr) + static_cast<int64_t>(pid_h) * stride_head_scratch;
    out_t* out_head_ptr =
        reinterpret_cast<out_t*>(out_base_ptr) + static_cast<int64_t>(pid_h) * stride_head_out;
    float* denom_ptr = reinterpret_cast<float*>(denom_base_ptr);

    __shared__ float row_max[kLogFPreMaxR];
    __shared__ float row_lse[kLogFPreMaxR];
    __shared__ int row_has[kLogFPreMaxR];
    __shared__ float row_count_f_shared;
    __shared__ float accum_n_cum_f_shared;
    __shared__ float token_max_shared;

    for (int r = 0; r < max_r; ++r) {
        float local_max = kLogFPreMinVal;
        for (int j = threadIdx.x; j < logits_capacity; j += blockDim.x) {
            float v = load_scratch_value(scratch_head_ptr + static_cast<int64_t>(r) * stride_row + j);
            if (v > kLogFPreMinVal) {
                local_max = local_max > v ? local_max : v;
            }
        }
        float block_max = block_reduce_max(local_max);
        if (threadIdx.x == 0) {
            row_has[r] = block_max > kLogFPreMinVal ? 1 : 0;
            row_max[r] = row_has[r] ? block_max : 0.0f;
        }
        __syncthreads();
    }

    for (int r = 0; r < max_r; ++r) {
        float local_sum = 0.0f;
        if (row_has[r] != 0) {
            const float max_val = row_max[r];
            for (int j = threadIdx.x; j < logits_capacity; j += blockDim.x) {
                float v = load_scratch_value(scratch_head_ptr + static_cast<int64_t>(r) * stride_row + j);
                if (v > kLogFPreMinVal) {
                    local_sum += expf(v - max_val);
                }
            }
        }
        float block_sum = block_reduce_sum(local_sum);
        if (threadIdx.x == 0) {
            row_lse[r] = row_has[r] ? (row_max[r] + logf(block_sum + 1.0e-20f)) : kLogFPreMinVal;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        int row_count = 0;
        for (int r = 0; r < max_r; ++r) {
            row_count += row_has[r];
        }
        row_count_f_shared = row_count > 0 ? static_cast<float>(row_count) : 1.0f;
        accum_n_cum_f_shared = static_cast<float>(accum_prev_rows + row_count);
    }
    __syncthreads();

    float local_token_max = kLogFPreMinVal;
    if (!accum_do_merge) {
        for (int j = threadIdx.x; j < logits_capacity; j += blockDim.x) {
            bool has_token = false;
            float log_f_pre = compute_log_f_pre_token(
                scratch_head_ptr,
                stride_row,
                j,
                max_r,
                row_lse,
                row_has,
                row_count_f_shared,
                alpha,
                use_mean,
                &has_token);
            store_log_f_pre_value<out_t>(out_head_ptr + j, log_f_pre);
            if (has_token) {
                local_token_max = local_token_max > log_f_pre ? local_token_max : log_f_pre;
            }
        }
    } else {
        // 跨片合并：out 内是前片写回的归一化部分窗口值（同线程 j-条带
        // 先读后写，无跨线程依赖）。合并后写回的仍是归一化值，中间片
        // 对任何读者良构；末片即全窗结果。
        const float n_prev_f = static_cast<float>(accum_prev_rows);
        for (int j = threadIdx.x; j < logits_capacity; j += blockDim.x) {
            bool has_cur = false;
            float cur = compute_log_f_pre_token_unnormalized(
                scratch_head_ptr,
                stride_row,
                j,
                max_r,
                row_lse,
                row_has,
                alpha,
                use_mean,
                &has_cur);
            bool has_any = false;
            float merged = merge_log_f_pre_accum<out_t>(
                out_head_ptr,
                j,
                accum_prev_capacity,
                n_prev_f,
                cur,
                has_cur,
                accum_n_cum_f_shared,
                alpha,
                use_mean,
                &has_any);
            store_log_f_pre_value<out_t>(out_head_ptr + j, merged);
            if (has_any) {
                local_token_max = local_token_max > merged ? local_token_max : merged;
            }
        }
    }

    const float token_max = block_reduce_max(local_token_max);
    if (threadIdx.x == 0) {
        token_max_shared = token_max;
    }
    __syncthreads();

    if (token_max_shared <= kLogFPreMinVal) {
        if (threadIdx.x == 0) {
            denom_ptr[pid_h] = 0.0f;
        }
        return;
    }

    float local_token_sum = 0.0f;
    if (!accum_do_merge) {
        for (int j = threadIdx.x; j < logits_capacity; j += blockDim.x) {
            bool has_token = false;
            float log_f_pre = compute_log_f_pre_token(
                scratch_head_ptr,
                stride_row,
                j,
                max_r,
                row_lse,
                row_has,
                row_count_f_shared,
                alpha,
                use_mean,
                &has_token);
            if (has_token) {
                local_token_sum += expf(log_f_pre - token_max_shared);
            }
        }
    } else {
        // 合并行的 denom 直接从已写回的 out 求 LSE（与存储值自洽，
        // 避免重放合并链；fp16 的 MinVal 存为 -inf，被 > 判定排除）。
        for (int j = threadIdx.x; j < logits_capacity; j += blockDim.x) {
            const float v = load_scratch_value(out_head_ptr + j);
            if (isfinite(v) && v > kLogFPreMinVal) {
                local_token_sum += expf(v - token_max_shared);
            }
        }
    }
    const float token_sum = block_reduce_sum(local_token_sum);
    if (threadIdx.x == 0) {
        denom_ptr[pid_h] = token_max_shared + logf(token_sum + 1.0e-20f);
    }
}

template <typename scratch_t, typename out_t>
__global__ void copy_log_f_lastn1_scratch_kernel(
    const int32_t* __restrict__ req_meta_i32,
    const int64_t* __restrict__ req_meta_i64,
    int64_t req_meta_i32_stride_row,
    int64_t req_meta_i32_stride_col,
    int64_t req_meta_i64_stride_row,
    int64_t req_meta_i64_stride_col,
    int num_seqs,
    int num_query_heads) {
    const int pid_seq = blockIdx.x;
    const int pid_h = blockIdx.y;
    if (pid_seq >= num_seqs || pid_h >= num_query_heads) {
        return;
    }

    const int32_t* meta32_row_ptr = req_meta_i32 + static_cast<int64_t>(pid_seq) * req_meta_i32_stride_row;
    const int64_t* meta64_row_ptr = req_meta_i64 + static_cast<int64_t>(pid_seq) * req_meta_i64_stride_row;

    const int scratch_stride_head_i32 = meta32_row_ptr[1 * req_meta_i32_stride_col];
    const int logits_last_n = meta32_row_ptr[2 * req_meta_i32_stride_col];
    const int logits_capacity = meta32_row_ptr[4 * req_meta_i32_stride_col];
    const int flags = meta32_row_ptr[5 * req_meta_i32_stride_col];
    const int out_stride_head_i32 = meta32_row_ptr[6 * req_meta_i32_stride_col];

    const bool use_compact = (flags & 1) != 0;
    const bool use_log_f = ((flags & 8) != 0) && !use_compact;
    if (!use_log_f || logits_last_n != 1 || logits_capacity <= 0 || scratch_stride_head_i32 <= 0) {
        return;
    }

    const int64_t scratch_base_ptr = meta64_row_ptr[1 * req_meta_i64_stride_col];
    const int64_t out_base_ptr = meta64_row_ptr[2 * req_meta_i64_stride_col];
    const int64_t denom_base_ptr = meta64_row_ptr[3 * req_meta_i64_stride_col];
    if (scratch_base_ptr == 0 || out_base_ptr == 0) {
        return;
    }

    const scratch_t* scratch_head_ptr =
        reinterpret_cast<const scratch_t*>(scratch_base_ptr) + static_cast<int64_t>(pid_h) * scratch_stride_head_i32;
    const int64_t out_stride_pad = out_stride_head_i32 > 0
        ? static_cast<int64_t>(out_stride_head_i32)
        : static_cast<int64_t>(logits_capacity);
    out_t* out_head_ptr =
        reinterpret_cast<out_t*>(out_base_ptr) + static_cast<int64_t>(pid_h) * out_stride_pad;

    float local_max = kLogFPreMinVal;
    for (int j = threadIdx.x; j < logits_capacity; j += blockDim.x) {
        const float v = load_scratch_value(scratch_head_ptr + j);
        store_log_f_pre_value<out_t>(out_head_ptr + j, v);
        if (v > kLogFPreMinVal) {
            local_max = local_max > v ? local_max : v;
        }
    }

    if (denom_base_ptr == 0) {
        return;
    }

    const float token_max = block_reduce_max(local_max);
    __shared__ float token_max_shared;
    if (threadIdx.x == 0) {
        token_max_shared = token_max;
    }
    __syncthreads();

    float local_sum = 0.0f;
    if (token_max_shared > kLogFPreMinVal) {
        for (int j = threadIdx.x; j < logits_capacity; j += blockDim.x) {
            const float v = load_scratch_value(scratch_head_ptr + j);
            if (v > kLogFPreMinVal) {
                local_sum += expf(v - token_max_shared);
            }
        }
    }
    const float token_sum = block_reduce_sum(local_sum);
    if (threadIdx.x == 0) {
        float* denom_ptr = reinterpret_cast<float*>(denom_base_ptr);
        denom_ptr[pid_h] = token_max_shared > kLogFPreMinVal
            ? token_max_shared + logf(token_sum + 1.0e-20f)
            : 0.0f;
    }
}

template <typename out_t>
__global__ void copy_log_f_lastn1_scratch_scalar_kernel(
    const float* __restrict__ scratch_base_ptr,
    int64_t scratch_stride_head,
    int logits_capacity,
    out_t* __restrict__ out_base_ptr,
    int64_t out_stride_head,
    float* __restrict__ denom_base_ptr,
    int64_t denom_stride_head,
    int num_query_heads) {
    const int pid_h = blockIdx.x;
    if (pid_h >= num_query_heads || logits_capacity <= 0) {
        return;
    }

    const float* scratch_head_ptr = scratch_base_ptr + static_cast<int64_t>(pid_h) * scratch_stride_head;
    out_t* out_head_ptr = out_base_ptr + static_cast<int64_t>(pid_h) * out_stride_head;

    float local_max = kLogFPreMinVal;
    for (int j = threadIdx.x; j < logits_capacity; j += blockDim.x) {
        const float v = scratch_head_ptr[j];
        store_log_f_pre_value<out_t>(out_head_ptr + j, v);
        if (v > kLogFPreMinVal) {
            local_max = local_max > v ? local_max : v;
        }
    }

    const float token_max = block_reduce_max(local_max);
    __shared__ float token_max_shared;
    if (threadIdx.x == 0) {
        token_max_shared = token_max;
    }
    __syncthreads();

    float local_sum = 0.0f;
    if (token_max_shared > kLogFPreMinVal) {
        for (int j = threadIdx.x; j < logits_capacity; j += blockDim.x) {
            const float v = scratch_head_ptr[j];
            if (v > kLogFPreMinVal) {
                local_sum += expf(v - token_max_shared);
            }
        }
    }
    const float token_sum = block_reduce_sum(local_sum);
    if (threadIdx.x == 0) {
        denom_base_ptr[static_cast<int64_t>(pid_h) * denom_stride_head] =
            token_max_shared > kLogFPreMinVal
                ? token_max_shared + logf(token_sum + 1.0e-20f)
                : 0.0f;
    }
}

template <typename out_t>
__global__ void reduce_log_f_pre_scratch_scalar_kernel(
    const float* __restrict__ scratch_base_ptr,
    int64_t scratch_stride_head,
    int64_t scratch_stride_row,
    int logits_last_n,
    int logits_capacity,
    out_t* __restrict__ out_base_ptr,
    int64_t out_stride_head,
    float* __restrict__ denom_base_ptr,
    int64_t denom_stride_head,
    int num_query_heads,
    float alpha,
    bool use_mean) {
    const int pid_h = blockIdx.x;
    if (pid_h >= num_query_heads || logits_last_n <= 1 || logits_capacity <= 0) {
        return;
    }

    const int max_r = logits_last_n < kLogFPreMaxR ? logits_last_n : kLogFPreMaxR;
    const float* scratch_head_ptr = scratch_base_ptr + static_cast<int64_t>(pid_h) * scratch_stride_head;
    out_t* out_head_ptr = out_base_ptr + static_cast<int64_t>(pid_h) * out_stride_head;

    __shared__ float row_max[kLogFPreMaxR];
    __shared__ float row_lse[kLogFPreMaxR];
    __shared__ int row_has[kLogFPreMaxR];
    __shared__ float row_count_f_shared;
    __shared__ float token_max_shared;

    for (int r = 0; r < max_r; ++r) {
        float local_max = kLogFPreMinVal;
        for (int j = threadIdx.x; j < logits_capacity; j += blockDim.x) {
            float v = scratch_head_ptr[static_cast<int64_t>(r) * scratch_stride_row + j];
            if (v > kLogFPreMinVal) {
                local_max = local_max > v ? local_max : v;
            }
        }
        float block_max = block_reduce_max(local_max);
        if (threadIdx.x == 0) {
            row_has[r] = block_max > kLogFPreMinVal ? 1 : 0;
            row_max[r] = row_has[r] ? block_max : 0.0f;
        }
        __syncthreads();
    }

    for (int r = 0; r < max_r; ++r) {
        float local_sum = 0.0f;
        if (row_has[r] != 0) {
            const float max_val = row_max[r];
            for (int j = threadIdx.x; j < logits_capacity; j += blockDim.x) {
                float v = scratch_head_ptr[static_cast<int64_t>(r) * scratch_stride_row + j];
                if (v > kLogFPreMinVal) {
                    local_sum += expf(v - max_val);
                }
            }
        }
        float block_sum = block_reduce_sum(local_sum);
        if (threadIdx.x == 0) {
            row_lse[r] = row_has[r] ? (row_max[r] + logf(block_sum + 1.0e-20f)) : kLogFPreMinVal;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        int row_count = 0;
        for (int r = 0; r < max_r; ++r) {
            row_count += row_has[r];
        }
        row_count_f_shared = row_count > 0 ? static_cast<float>(row_count) : 1.0f;
    }
    __syncthreads();

    float local_token_max = kLogFPreMinVal;
    for (int j = threadIdx.x; j < logits_capacity; j += blockDim.x) {
        bool has_token = false;
        float log_f_pre = compute_log_f_pre_token(
            scratch_head_ptr,
            scratch_stride_row,
            j,
            max_r,
            row_lse,
            row_has,
            row_count_f_shared,
            alpha,
            use_mean,
            &has_token);
        store_log_f_pre_value<out_t>(out_head_ptr + j, log_f_pre);
        if (has_token) {
            local_token_max = local_token_max > log_f_pre ? local_token_max : log_f_pre;
        }
    }

    const float token_max = block_reduce_max(local_token_max);
    if (threadIdx.x == 0) {
        token_max_shared = token_max;
    }
    __syncthreads();

    if (token_max_shared <= kLogFPreMinVal) {
        if (threadIdx.x == 0) {
            denom_base_ptr[static_cast<int64_t>(pid_h) * denom_stride_head] = 0.0f;
        }
        return;
    }

    float local_token_sum = 0.0f;
    for (int j = threadIdx.x; j < logits_capacity; j += blockDim.x) {
        bool has_token = false;
        float log_f_pre = compute_log_f_pre_token(
            scratch_head_ptr,
            scratch_stride_row,
            j,
            max_r,
            row_lse,
            row_has,
            row_count_f_shared,
            alpha,
            use_mean,
            &has_token);
        if (has_token) {
            local_token_sum += expf(log_f_pre - token_max_shared);
        }
    }
    const float token_sum = block_reduce_sum(local_token_sum);
    if (threadIdx.x == 0) {
        denom_base_ptr[static_cast<int64_t>(pid_h) * denom_stride_head] =
            token_max_shared + logf(token_sum + 1.0e-20f);
    }
}

void reduce_log_f_pre_scratch_cuda(
    torch::Tensor req_meta_i32,
    torch::Tensor req_meta_i64,
    int64_t num_seqs,
    int64_t num_query_heads,
    bool scratch_in_fp16,
    bool log_f_out_fp32,
    double alpha) {
    TORCH_CHECK(req_meta_i32.is_cuda(), "req_meta_i32 must be CUDA");
    TORCH_CHECK(req_meta_i64.is_cuda(), "req_meta_i64 must be CUDA");
    TORCH_CHECK(req_meta_i32.scalar_type() == torch::kInt32, "req_meta_i32 must be int32");
    TORCH_CHECK(req_meta_i64.scalar_type() == torch::kInt64, "req_meta_i64 must be int64");
    TORCH_CHECK(req_meta_i32.dim() == 2 && req_meta_i32.size(1) >= 8, "req_meta_i32 must be [N,>=8]");
    TORCH_CHECK(req_meta_i64.dim() == 2 && req_meta_i64.size(1) >= 4, "req_meta_i64 must be [N,>=4]");
    TORCH_CHECK(num_seqs >= 0, "num_seqs must be non-negative");
    TORCH_CHECK(num_query_heads >= 0, "num_query_heads must be non-negative");
    TORCH_CHECK(req_meta_i32.size(0) >= num_seqs, "req_meta_i32 rows must cover num_seqs");
    TORCH_CHECK(req_meta_i64.size(0) >= num_seqs, "req_meta_i64 rows must cover num_seqs");

    const dim3 blocks(static_cast<unsigned int>(num_seqs), static_cast<unsigned int>(num_query_heads));
    constexpr int threads = 256;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const float alpha_f = static_cast<float>(alpha);
    const bool use_mean = std::abs(alpha_f) < 1.0e-6f;

    if (scratch_in_fp16) {
        if (log_f_out_fp32) {
            reduce_log_f_pre_scratch_kernel<half, float><<<blocks, threads, 0, stream>>>(
                req_meta_i32.data_ptr<int32_t>(),
                req_meta_i64.data_ptr<int64_t>(),
                req_meta_i32.stride(0),
                req_meta_i32.stride(1),
                req_meta_i64.stride(0),
                req_meta_i64.stride(1),
                static_cast<int>(num_seqs),
                static_cast<int>(num_query_heads),
                alpha_f,
                use_mean);
        } else {
            reduce_log_f_pre_scratch_kernel<half, half><<<blocks, threads, 0, stream>>>(
                req_meta_i32.data_ptr<int32_t>(),
                req_meta_i64.data_ptr<int64_t>(),
                req_meta_i32.stride(0),
                req_meta_i32.stride(1),
                req_meta_i64.stride(0),
                req_meta_i64.stride(1),
                static_cast<int>(num_seqs),
                static_cast<int>(num_query_heads),
                alpha_f,
                use_mean);
        }
    } else {
        if (log_f_out_fp32) {
            reduce_log_f_pre_scratch_kernel<float, float><<<blocks, threads, 0, stream>>>(
                req_meta_i32.data_ptr<int32_t>(),
                req_meta_i64.data_ptr<int64_t>(),
                req_meta_i32.stride(0),
                req_meta_i32.stride(1),
                req_meta_i64.stride(0),
                req_meta_i64.stride(1),
                static_cast<int>(num_seqs),
                static_cast<int>(num_query_heads),
                alpha_f,
                use_mean);
        } else {
            reduce_log_f_pre_scratch_kernel<float, half><<<blocks, threads, 0, stream>>>(
                req_meta_i32.data_ptr<int32_t>(),
                req_meta_i64.data_ptr<int64_t>(),
                req_meta_i32.stride(0),
                req_meta_i32.stride(1),
                req_meta_i64.stride(0),
                req_meta_i64.stride(1),
                static_cast<int>(num_seqs),
                static_cast<int>(num_query_heads),
                alpha_f,
                use_mean);
        }
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void copy_log_f_lastn1_scratch_cuda(
    torch::Tensor req_meta_i32,
    torch::Tensor req_meta_i64,
    int64_t num_seqs,
    int64_t num_query_heads,
    bool scratch_in_fp16,
    bool log_f_out_fp32) {
    TORCH_CHECK(req_meta_i32.is_cuda(), "req_meta_i32 must be CUDA");
    TORCH_CHECK(req_meta_i64.is_cuda(), "req_meta_i64 must be CUDA");
    TORCH_CHECK(req_meta_i32.scalar_type() == torch::kInt32, "req_meta_i32 must be int32");
    TORCH_CHECK(req_meta_i64.scalar_type() == torch::kInt64, "req_meta_i64 must be int64");
    TORCH_CHECK(req_meta_i32.dim() == 2 && req_meta_i32.size(1) >= 6, "req_meta_i32 must be [N,>=6]");
    TORCH_CHECK(req_meta_i64.dim() == 2 && req_meta_i64.size(1) >= 4, "req_meta_i64 must be [N,>=4]");
    TORCH_CHECK(num_seqs >= 0, "num_seqs must be non-negative");
    TORCH_CHECK(num_query_heads >= 0, "num_query_heads must be non-negative");
    TORCH_CHECK(req_meta_i32.size(0) >= num_seqs, "req_meta_i32 rows must cover num_seqs");
    TORCH_CHECK(req_meta_i64.size(0) >= num_seqs, "req_meta_i64 rows must cover num_seqs");

    const dim3 blocks(static_cast<unsigned int>(num_seqs), static_cast<unsigned int>(num_query_heads));
    constexpr int threads = 256;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (scratch_in_fp16) {
        if (log_f_out_fp32) {
            copy_log_f_lastn1_scratch_kernel<half, float><<<blocks, threads, 0, stream>>>(
                req_meta_i32.data_ptr<int32_t>(),
                req_meta_i64.data_ptr<int64_t>(),
                req_meta_i32.stride(0),
                req_meta_i32.stride(1),
                req_meta_i64.stride(0),
                req_meta_i64.stride(1),
                static_cast<int>(num_seqs),
                static_cast<int>(num_query_heads));
        } else {
            copy_log_f_lastn1_scratch_kernel<half, half><<<blocks, threads, 0, stream>>>(
                req_meta_i32.data_ptr<int32_t>(),
                req_meta_i64.data_ptr<int64_t>(),
                req_meta_i32.stride(0),
                req_meta_i32.stride(1),
                req_meta_i64.stride(0),
                req_meta_i64.stride(1),
                static_cast<int>(num_seqs),
                static_cast<int>(num_query_heads));
        }
    } else {
        if (log_f_out_fp32) {
            copy_log_f_lastn1_scratch_kernel<float, float><<<blocks, threads, 0, stream>>>(
                req_meta_i32.data_ptr<int32_t>(),
                req_meta_i64.data_ptr<int64_t>(),
                req_meta_i32.stride(0),
                req_meta_i32.stride(1),
                req_meta_i64.stride(0),
                req_meta_i64.stride(1),
                static_cast<int>(num_seqs),
                static_cast<int>(num_query_heads));
        } else {
            copy_log_f_lastn1_scratch_kernel<float, half><<<blocks, threads, 0, stream>>>(
                req_meta_i32.data_ptr<int32_t>(),
                req_meta_i64.data_ptr<int64_t>(),
                req_meta_i32.stride(0),
                req_meta_i32.stride(1),
                req_meta_i64.stride(0),
                req_meta_i64.stride(1),
                static_cast<int>(num_seqs),
                static_cast<int>(num_query_heads));
        }
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void copy_log_f_lastn1_scratch_scalar_cuda(
    torch::Tensor scratch_capture_scores,
    int64_t scratch_row,
    int64_t effective_kv_len,
    torch::Tensor out_capture_scores,
    int64_t capture_row,
    torch::Tensor out_log_f_denoms,
    bool log_f_out_fp32) {
    TORCH_CHECK(scratch_capture_scores.is_cuda(), "scratch_capture_scores must be CUDA");
    TORCH_CHECK(out_capture_scores.is_cuda(), "out_capture_scores must be CUDA");
    TORCH_CHECK(out_log_f_denoms.is_cuda(), "out_log_f_denoms must be CUDA");
    TORCH_CHECK(scratch_capture_scores.scalar_type() == torch::kFloat32, "scratch_capture_scores must be float32");
    TORCH_CHECK(out_log_f_denoms.scalar_type() == torch::kFloat32, "out_log_f_denoms must be float32");
    TORCH_CHECK(
        log_f_out_fp32
            ? out_capture_scores.scalar_type() == torch::kFloat32
            : out_capture_scores.scalar_type() == torch::kFloat16,
        "out_capture_scores dtype does not match log_f_out_fp32");
    TORCH_CHECK(scratch_capture_scores.dim() == 4, "scratch_capture_scores must be [rows,heads,last_n,kv]");
    TORCH_CHECK(out_capture_scores.dim() == 4, "out_capture_scores must be [rows,heads,1,kv]");
    TORCH_CHECK(out_log_f_denoms.dim() >= 2, "out_log_f_denoms must be at least 2D");
    TORCH_CHECK(scratch_row >= 0 && scratch_row < scratch_capture_scores.size(0), "scratch_row out of range");
    TORCH_CHECK(capture_row >= 0 && capture_row < out_capture_scores.size(0), "capture_row out of range");
    TORCH_CHECK(effective_kv_len >= 0 && effective_kv_len <= scratch_capture_scores.size(3), "effective_kv_len outside scratch kv");
    TORCH_CHECK(effective_kv_len <= out_capture_scores.size(3), "effective_kv_len outside output kv");
    TORCH_CHECK(scratch_capture_scores.size(1) <= out_capture_scores.size(1), "output heads must cover scratch heads");
    TORCH_CHECK(scratch_capture_scores.size(1) <= out_log_f_denoms.size(1), "denom heads must cover scratch heads");
    if (effective_kv_len == 0) {
        return;
    }

    const int num_query_heads = static_cast<int>(scratch_capture_scores.size(1));
    const dim3 blocks(static_cast<unsigned int>(num_query_heads));
    constexpr int threads = 256;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const float* scratch_base_ptr =
        scratch_capture_scores.data_ptr<float>() + scratch_row * scratch_capture_scores.stride(0);
    float* denom_base_ptr =
        out_log_f_denoms.data_ptr<float>() + capture_row * out_log_f_denoms.stride(0);
    if (log_f_out_fp32) {
        float* out_base_ptr =
            out_capture_scores.data_ptr<float>() + capture_row * out_capture_scores.stride(0);
        copy_log_f_lastn1_scratch_scalar_kernel<float><<<blocks, threads, 0, stream>>>(
            scratch_base_ptr,
            scratch_capture_scores.stride(1),
            static_cast<int>(effective_kv_len),
            out_base_ptr,
            out_capture_scores.stride(1),
            denom_base_ptr,
            out_log_f_denoms.stride(1),
            num_query_heads);
    } else {
        half* out_base_ptr =
            reinterpret_cast<half*>(out_capture_scores.data_ptr<at::Half>()) + capture_row * out_capture_scores.stride(0);
        copy_log_f_lastn1_scratch_scalar_kernel<half><<<blocks, threads, 0, stream>>>(
            scratch_base_ptr,
            scratch_capture_scores.stride(1),
            static_cast<int>(effective_kv_len),
            out_base_ptr,
            out_capture_scores.stride(1),
            denom_base_ptr,
            out_log_f_denoms.stride(1),
            num_query_heads);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void reduce_log_f_pre_scratch_scalar_cuda(
    torch::Tensor scratch_capture_scores,
    int64_t scratch_row,
    int64_t effective_kv_len,
    int64_t last_n,
    torch::Tensor out_capture_scores,
    int64_t capture_row,
    torch::Tensor out_log_f_denoms,
    bool log_f_out_fp32,
    double alpha) {
    TORCH_CHECK(scratch_capture_scores.is_cuda(), "scratch_capture_scores must be CUDA");
    TORCH_CHECK(out_capture_scores.is_cuda(), "out_capture_scores must be CUDA");
    TORCH_CHECK(out_log_f_denoms.is_cuda(), "out_log_f_denoms must be CUDA");
    TORCH_CHECK(scratch_capture_scores.scalar_type() == torch::kFloat32, "scratch_capture_scores must be float32");
    TORCH_CHECK(out_log_f_denoms.scalar_type() == torch::kFloat32, "out_log_f_denoms must be float32");
    TORCH_CHECK(
        log_f_out_fp32
            ? out_capture_scores.scalar_type() == torch::kFloat32
            : out_capture_scores.scalar_type() == torch::kFloat16,
        "out_capture_scores dtype does not match log_f_out_fp32");
    TORCH_CHECK(scratch_capture_scores.dim() == 4, "scratch_capture_scores must be [rows,heads,last_n,kv]");
    TORCH_CHECK(out_capture_scores.dim() == 4, "out_capture_scores must be [rows,heads,1,kv]");
    TORCH_CHECK(out_log_f_denoms.dim() >= 2, "out_log_f_denoms must be at least 2D");
    TORCH_CHECK(scratch_row >= 0 && scratch_row < scratch_capture_scores.size(0), "scratch_row out of range");
    TORCH_CHECK(capture_row >= 0 && capture_row < out_capture_scores.size(0), "capture_row out of range");
    TORCH_CHECK(last_n > 1 && last_n <= scratch_capture_scores.size(2), "last_n outside scratch tail_q");
    TORCH_CHECK(effective_kv_len >= 0 && effective_kv_len <= scratch_capture_scores.size(3), "effective_kv_len outside scratch kv");
    TORCH_CHECK(effective_kv_len <= out_capture_scores.size(3), "effective_kv_len outside output kv");
    TORCH_CHECK(scratch_capture_scores.size(1) <= out_capture_scores.size(1), "output heads must cover scratch heads");
    TORCH_CHECK(scratch_capture_scores.size(1) <= out_log_f_denoms.size(1), "denom heads must cover scratch heads");
    if (effective_kv_len == 0) {
        return;
    }

    const int num_query_heads = static_cast<int>(scratch_capture_scores.size(1));
    const dim3 blocks(static_cast<unsigned int>(num_query_heads));
    constexpr int threads = 256;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const float alpha_f = static_cast<float>(alpha);
    const bool use_mean = std::abs(alpha_f) < 1.0e-6f;

    const float* scratch_base_ptr =
        scratch_capture_scores.data_ptr<float>() + scratch_row * scratch_capture_scores.stride(0);
    float* denom_base_ptr =
        out_log_f_denoms.data_ptr<float>() + capture_row * out_log_f_denoms.stride(0);
    if (log_f_out_fp32) {
        float* out_base_ptr =
            out_capture_scores.data_ptr<float>() + capture_row * out_capture_scores.stride(0);
        reduce_log_f_pre_scratch_scalar_kernel<float><<<blocks, threads, 0, stream>>>(
            scratch_base_ptr,
            scratch_capture_scores.stride(1),
            scratch_capture_scores.stride(2),
            static_cast<int>(last_n),
            static_cast<int>(effective_kv_len),
            out_base_ptr,
            out_capture_scores.stride(1),
            denom_base_ptr,
            out_log_f_denoms.stride(1),
            num_query_heads,
            alpha_f,
            use_mean);
    } else {
        half* out_base_ptr =
            reinterpret_cast<half*>(out_capture_scores.data_ptr<at::Half>()) + capture_row * out_capture_scores.stride(0);
        reduce_log_f_pre_scratch_scalar_kernel<half><<<blocks, threads, 0, stream>>>(
            scratch_base_ptr,
            scratch_capture_scores.stride(1),
            scratch_capture_scores.stride(2),
            static_cast<int>(last_n),
            static_cast<int>(effective_kv_len),
            out_base_ptr,
            out_capture_scores.stride(1),
            denom_base_ptr,
            out_log_f_denoms.stride(1),
            num_query_heads,
            alpha_f,
            use_mean);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

"""

    fast_math = os.environ.get("VLLM_SPARSE_SELECTOR_LOGS_FAST_MATH", "0") == "1"
    extra_cuda_cflags = ["-lineinfo", _cuda_std_flag_for_current_nvcc()]
    if fast_math:
        extra_cuda_cflags.append("--use_fast_math")
    _configure_torch_cuda_toolchain()
    _ensure_torch_cuda_arch_list()
    # Torch may emit NVCC depfile flags unsupported by older toolchains (for
    # example CUDA 10.1 on this cluster). Skipping depfile generation keeps the
    # selector log_s CUDA extension buildable without changing math behavior.
    os.environ.setdefault("TORCH_EXTENSION_SKIP_NVCC_GEN_DEPENDENCIES", "1")
    # Torch appends -std=c++17 unless an explicit CUDA std flag is already
    # present. Older nvcc toolchains on this cluster reject c++17 entirely, so
    # choose the lowest compatible standard at the loader boundary.

    try:
        _MODULE = load_inline(
            name="selector_log_s_ext",
            cpp_sources=cpp_source,
            cuda_sources=cuda_source,
            functions=None,
            extra_cuda_cflags=extra_cuda_cflags,
            verbose=False,
        )
    except Exception as exc:
        _LOAD_ERROR = exc
        return None
    return _MODULE


def _require_ext(*, force: bool = False) -> torch.nn.Module:
    mod = _load_ext(force=force)
    if mod is None:
        if _LOAD_ERROR is not None:
            raise SelectorLogSExtUnavailable("selector_log_s_ext unavailable") from _LOAD_ERROR
        raise SelectorLogSExtUnavailable(
            "selector_log_s_ext unavailable; set VLLM_SPARSE_SELECTOR_LOGS_CUDA=1"
        )
    return mod


def fused_log_f_prior_logits_flat(
    scores: torch.Tensor,
    row_lo: torch.Tensor,
    row_hi: torch.Tensor,
    key_norms: torch.Tensor,
    *,
    alpha: float,
    eps: float,
    gamma: float,
    prior_weight_l2: float,
    prior_weight_pos: float,
    prior_pos_power: float,
    prior_pos_eta: float,
    beta: float,
    lambda_clip_single: float,
    lambda_clip_multi: float,
    lambda_tail_kappa: float,
    lambda_tail_pivot: float,
    lambda_soft: bool,
) -> torch.Tensor:
    mod = _require_ext()
    out = mod.fused_log_f_prior_logits(
        scores,
        row_lo,
        row_hi,
        key_norms,
        float(alpha),
        float(eps),
        float(gamma),
        float(prior_weight_l2),
        float(prior_weight_pos),
        float(prior_pos_power),
        float(prior_pos_eta),
        float(beta),
        float(lambda_clip_single),
        float(lambda_clip_multi),
        float(lambda_tail_kappa),
        float(lambda_tail_pivot),
        bool(lambda_soft),
    )
    return out[0]


def fused_log_f_prior_pre_denom_flat(
    scores: torch.Tensor,
    denom: torch.Tensor,
    row_lo: torch.Tensor,
    row_hi: torch.Tensor,
    key_norms: torch.Tensor,
    *,
    alpha: float,
    eps: float,
    gamma: float,
    prior_weight_l2: float,
    prior_weight_pos: float,
    prior_pos_power: float,
    prior_pos_eta: float,
    beta: float,
    lambda_clip_single: float,
    lambda_clip_multi: float,
    lambda_tail_kappa: float,
    lambda_tail_pivot: float,
    lambda_soft: bool,
) -> torch.Tensor:
    mod = _require_ext()
    out = mod.fused_log_f_prior_pre_denom(
        scores,
        denom,
        row_lo,
        row_hi,
        key_norms,
        float(alpha),
        float(eps),
        float(gamma),
        float(prior_weight_l2),
        float(prior_weight_pos),
        float(prior_pos_power),
        float(prior_pos_eta),
        float(beta),
        float(lambda_clip_single),
        float(lambda_clip_multi),
        float(lambda_tail_kappa),
        float(lambda_tail_pivot),
        bool(lambda_soft),
    )
    return out[0]






def reduce_log_f_pre_scratch_cuda(
    *,
    req_meta_i32: torch.Tensor,
    req_meta_i64: torch.Tensor,
    num_seqs: int,
    num_query_heads: int,
    log_f_out_fp32: bool,
    alpha: float,
    scratch_in_fp16: bool = False,
) -> None:
    mod = _require_ext(force=True)
    if req_meta_i32.dim() == 2 and int(req_meta_i32.size(1)) == 7:
        meta_i32_v2 = torch.zeros(
            (int(req_meta_i32.size(0)), 8),
            device=req_meta_i32.device,
            dtype=req_meta_i32.dtype,
        )
        meta_i32_v2[:, :7] = req_meta_i32
        row_stride = torch.maximum(meta_i32_v2[:, 1], meta_i32_v2[:, 4])
        meta_i32_v2[:, 7] = row_stride
        meta_i32_v2[:, 1] = meta_i32_v2[:, 2] * row_stride
        req_meta_i32 = meta_i32_v2
    mod.reduce_log_f_pre_scratch(
        req_meta_i32,
        req_meta_i64,
        int(num_seqs),
        int(num_query_heads),
        bool(scratch_in_fp16),
        bool(log_f_out_fp32),
        float(alpha),
    )


def copy_log_f_lastn1_scratch_cuda(
    *,
    req_meta_i32: torch.Tensor,
    req_meta_i64: torch.Tensor,
    num_seqs: int,
    num_query_heads: int,
    log_f_out_fp32: bool,
    scratch_in_fp16: bool = False,
) -> None:
    mod = _require_ext(force=True)
    mod.copy_log_f_lastn1_scratch(
        req_meta_i32,
        req_meta_i64,
        int(num_seqs),
        int(num_query_heads),
        bool(scratch_in_fp16),
        bool(log_f_out_fp32),
    )


def copy_log_f_lastn1_scratch_scalar_cuda(
    *,
    scratch_capture_scores: torch.Tensor,
    scratch_row: int,
    effective_kv_len: int,
    out_capture_scores: torch.Tensor,
    capture_row: int,
    out_log_f_denoms: torch.Tensor,
    log_f_out_fp32: bool,
) -> None:
    mod = _require_ext(force=True)
    mod.copy_log_f_lastn1_scratch_scalar(
        scratch_capture_scores,
        int(scratch_row),
        int(effective_kv_len),
        out_capture_scores,
        int(capture_row),
        out_log_f_denoms,
        bool(log_f_out_fp32),
    )


def reduce_log_f_pre_scratch_scalar_cuda(
    *,
    scratch_capture_scores: torch.Tensor,
    scratch_row: int,
    effective_kv_len: int,
    last_n: int,
    out_capture_scores: torch.Tensor,
    capture_row: int,
    out_log_f_denoms: torch.Tensor,
    log_f_out_fp32: bool,
    alpha: float,
) -> None:
    mod = _require_ext(force=True)
    mod.reduce_log_f_pre_scratch_scalar(
        scratch_capture_scores,
        int(scratch_row),
        int(effective_kv_len),
        int(last_n),
        out_capture_scores,
        int(capture_row),
        out_log_f_denoms,
        bool(log_f_out_fp32),
        float(alpha),
    )


__all__ = [
    "SelectorLogSExtUnavailable",
    "fused_log_f_prior_logits_flat",
    "fused_log_f_prior_pre_denom_flat",
    "reduce_log_f_pre_scratch_cuda",
    "copy_log_f_lastn1_scratch_cuda",
    "copy_log_f_lastn1_scratch_scalar_cuda",
    "reduce_log_f_pre_scratch_scalar_cuda",
]
