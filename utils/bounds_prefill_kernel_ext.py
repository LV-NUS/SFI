"""CUDA extension: prefill selector bounds for W > 1."""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
from torch.utils.cpp_extension import load_inline
from utils.torch_extension_cache import load_prebuilt_extension
from utils.ext_toolchain import configure_jit_toolchain_or_raise

_MODULE: Optional[torch.nn.Module] = None
_LOAD_ERROR: Optional[Exception] = None


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


def _load_ext() -> Optional[torch.nn.Module]:
    global _MODULE, _LOAD_ERROR
    if _MODULE is not None:
        return _MODULE
    if _LOAD_ERROR is not None:
        return None
    prebuilt = load_prebuilt_extension("bounds_prefill_kernel_ext")
    if prebuilt is not None:
        _MODULE = prebuilt
        return _MODULE

    cpp_source = r"""
#include <torch/extension.h>
#include <vector>

void compute_bounds_prefill_cuda(
    torch::Tensor kv_lengths,
    c10::optional<torch::Tensor> seq_full_opt,
    torch::Tensor kv_len_head,
    torch::Tensor head_sink,
    torch::Tensor recent_start,
    torch::Tensor allowed_lengths,
    torch::Tensor row_lo,
    torch::Tensor row_hi,
    int64_t L,
    int64_t B,
    int64_t H_kv,
    int64_t G,
    int64_t W,
    int64_t kv_len_total,
    int64_t sink_cfg,
    int64_t recent_cfg,
    int64_t block_size,
    int64_t kv_stride_l,
    int64_t kv_stride_b,
    int64_t kv_stride_h,
    int64_t sf_stride_l,
    int64_t sf_stride_b,
    int64_t sf_stride_h);

std::vector<torch::Tensor> compute_bounds_prefill(
    torch::Tensor kv_lengths,
    c10::optional<torch::Tensor> seq_full_opt,
    torch::Tensor kv_len_head,
    torch::Tensor head_sink,
    torch::Tensor recent_start,
    torch::Tensor allowed_lengths,
    torch::Tensor row_lo,
    torch::Tensor row_hi,
    int64_t L,
    int64_t B,
    int64_t H_kv,
    int64_t G,
    int64_t W,
    int64_t kv_len_total,
    int64_t sink_cfg,
    int64_t recent_cfg,
    int64_t block_size,
    int64_t kv_stride_l,
    int64_t kv_stride_b,
    int64_t kv_stride_h,
    int64_t sf_stride_l,
    int64_t sf_stride_b,
    int64_t sf_stride_h) {
    compute_bounds_prefill_cuda(
        kv_lengths,
        seq_full_opt,
        kv_len_head,
        head_sink,
        recent_start,
        allowed_lengths,
        row_lo,
        row_hi,
        L,
        B,
        H_kv,
        G,
        W,
        kv_len_total,
        sink_cfg,
        recent_cfg,
        block_size,
        kv_stride_l,
        kv_stride_b,
        kv_stride_h,
        sf_stride_l,
        sf_stride_b,
        sf_stride_h);
    return {kv_len_head, head_sink, recent_start, allowed_lengths, row_lo, row_hi};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_bounds_prefill", &compute_bounds_prefill,
          "Compute prefill selector bounds for W > 1 (CUDA kernel)");
}
"""

    cuda_source = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename kv_t>
__global__ void compute_bounds_prefill_kernel(
    const kv_t* __restrict__ kv_lengths,
    const int32_t* __restrict__ seq_full,
    int32_t* __restrict__ kv_len_head,
    int32_t* __restrict__ head_sink,
    int32_t* __restrict__ recent_start,
    int32_t* __restrict__ allowed_lengths,
    int32_t* __restrict__ row_lo,
    int32_t* __restrict__ row_hi,
    int L,
    int B,
    int H_kv,
    int G,
    int W,
    int kv_len_total,
    int sink_cfg,
    int recent_cfg,
    int block_size,
    int64_t kv_stride_l,
    int64_t kv_stride_b,
    int64_t kv_stride_h,
    int64_t sf_stride_l,
    int64_t sf_stride_b,
    int64_t sf_stride_h) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int M = L * B * H_kv;
    if (idx >= M) {
        return;
    }

    int h_kv = idx % H_kv;
    int tmp = idx / H_kv;
    int b = tmp % B;
    int l = tmp / B;

    int max_kv_len = 0;
    for (int g = 0; g < G; ++g) {
        int64_t kv_idx =
            static_cast<int64_t>(l) * kv_stride_l +
            static_cast<int64_t>(b) * kv_stride_b +
            static_cast<int64_t>(h_kv * G + g) * kv_stride_h;
        int kv_len = static_cast<int>(kv_lengths[kv_idx]);
        if (kv_len > max_kv_len) {
            max_kv_len = kv_len;
        }
    }
    if (max_kv_len > kv_len_total) {
        max_kv_len = kv_len_total;
    }
    if (max_kv_len < 0) {
        max_kv_len = 0;
    }
    kv_len_head[idx] = max_kv_len;

    int h_sink = max_kv_len < sink_cfg ? max_kv_len : sink_cfg;
    if (h_sink < 0) {
        h_sink = 0;
    }
    head_sink[idx] = h_sink;

    int r_start = 0;
    if (block_size > 0 && recent_cfg > 0) {
        int seq_len = max_kv_len;
        if (seq_full != nullptr) {
            int64_t sf_idx =
                static_cast<int64_t>(l) * sf_stride_l +
                static_cast<int64_t>(b) * sf_stride_b +
                static_cast<int64_t>(h_kv) * sf_stride_h;
            seq_len = static_cast<int>(seq_full[sf_idx]);
        }
        int cap = seq_len < recent_cfg ? seq_len : recent_cfg;
        r_start = ((seq_len - cap) / block_size) * block_size;
        if (r_start < 0) {
            r_start = 0;
        }
        if (r_start > kv_len_total) {
            r_start = kv_len_total;
        }
    }
    recent_start[idx] = r_start;

    int allowed = r_start - h_sink;
    if (allowed < 0) {
        allowed = 0;
    }
    allowed_lengths[idx] = allowed;

    int row_base = idx * G * W;
    for (int g = 0; g < G; ++g) {
        int64_t kv_idx =
            static_cast<int64_t>(l) * kv_stride_l +
            static_cast<int64_t>(b) * kv_stride_b +
            static_cast<int64_t>(h_kv * G + g) * kv_stride_h;
        int kv_len_g = static_cast<int>(kv_lengths[kv_idx]);
        if (kv_len_g > kv_len_total) {
            kv_len_g = kv_len_total;
        }
        if (kv_len_g < 0) {
            kv_len_g = 0;
        }
        for (int w = 0; w < W; ++w) {
            int hi_stair = max_kv_len - (W - 1 - w);
            int hi = kv_len_g < hi_stair ? kv_len_g : hi_stair;
            hi = hi < r_start ? hi : r_start;
            if (hi < 0) {
                hi = 0;
            }
            if (hi > kv_len_total) {
                hi = kv_len_total;
            }
            int out_idx = row_base + g * W + w;
            row_lo[out_idx] = h_sink;
            row_hi[out_idx] = hi;
        }
    }
}

void compute_bounds_prefill_cuda(
    torch::Tensor kv_lengths,
    c10::optional<torch::Tensor> seq_full_opt,
    torch::Tensor kv_len_head,
    torch::Tensor head_sink,
    torch::Tensor recent_start,
    torch::Tensor allowed_lengths,
    torch::Tensor row_lo,
    torch::Tensor row_hi,
    int64_t L,
    int64_t B,
    int64_t H_kv,
    int64_t G,
    int64_t W,
    int64_t kv_len_total,
    int64_t sink_cfg,
    int64_t recent_cfg,
    int64_t block_size,
    int64_t kv_stride_l,
    int64_t kv_stride_b,
    int64_t kv_stride_h,
    int64_t sf_stride_l,
    int64_t sf_stride_b,
    int64_t sf_stride_h) {
    TORCH_CHECK(kv_lengths.is_cuda(), "kv_lengths must be CUDA");
    TORCH_CHECK(kv_lengths.scalar_type() == torch::kInt64 || kv_lengths.scalar_type() == torch::kInt32,
                "kv_lengths must be int64 or int32");
    TORCH_CHECK(row_lo.is_cuda() && row_hi.is_cuda(), "row bounds must be CUDA");
    TORCH_CHECK(row_lo.scalar_type() == torch::kInt32 && row_hi.scalar_type() == torch::kInt32,
                "row bounds must be int32");

    const int M = static_cast<int>(L * B * H_kv);
    const int threads = 256;
    const int blocks = (M + threads - 1) / threads;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int32_t* seq_ptr = nullptr;
    if (seq_full_opt.has_value() && seq_full_opt.value().defined()) {
        auto seq_full = seq_full_opt.value();
        TORCH_CHECK(seq_full.is_cuda(), "seq_full must be CUDA when defined");
        TORCH_CHECK(seq_full.scalar_type() == torch::kInt32, "seq_full must be int32 when defined");
        seq_ptr = seq_full.data_ptr<int32_t>();
    }

    if (kv_lengths.scalar_type() == torch::kInt64) {
        compute_bounds_prefill_kernel<int64_t><<<blocks, threads, 0, stream>>>(
            kv_lengths.data_ptr<int64_t>(),
            seq_ptr,
            kv_len_head.data_ptr<int32_t>(),
            head_sink.data_ptr<int32_t>(),
            recent_start.data_ptr<int32_t>(),
            allowed_lengths.data_ptr<int32_t>(),
            row_lo.data_ptr<int32_t>(),
            row_hi.data_ptr<int32_t>(),
            static_cast<int>(L),
            static_cast<int>(B),
            static_cast<int>(H_kv),
            static_cast<int>(G),
            static_cast<int>(W),
            static_cast<int>(kv_len_total),
            static_cast<int>(sink_cfg),
            static_cast<int>(recent_cfg),
            static_cast<int>(block_size),
            kv_stride_l,
            kv_stride_b,
            kv_stride_h,
            sf_stride_l,
            sf_stride_b,
            sf_stride_h);
    } else {
        compute_bounds_prefill_kernel<int32_t><<<blocks, threads, 0, stream>>>(
            kv_lengths.data_ptr<int32_t>(),
            seq_ptr,
            kv_len_head.data_ptr<int32_t>(),
            head_sink.data_ptr<int32_t>(),
            recent_start.data_ptr<int32_t>(),
            allowed_lengths.data_ptr<int32_t>(),
            row_lo.data_ptr<int32_t>(),
            row_hi.data_ptr<int32_t>(),
            static_cast<int>(L),
            static_cast<int>(B),
            static_cast<int>(H_kv),
            static_cast<int>(G),
            static_cast<int>(W),
            static_cast<int>(kv_len_total),
            static_cast<int>(sink_cfg),
            static_cast<int>(recent_cfg),
            static_cast<int>(block_size),
            kv_stride_l,
            kv_stride_b,
            kv_stride_h,
            sf_stride_l,
            sf_stride_b,
            sf_stride_h);
    }
}
"""

    _ensure_torch_cuda_arch_list()
    try:
        # [EXT-NVCC-GUARD] JIT 回落前 pin nvcc+版本预检(系统 nvcc 10.1 坑根修);
        # 失败信息进 _LOAD_ERROR,由 require/enabled 语义原样呈报。
        configure_jit_toolchain_or_raise(ext_name="bounds_prefill_kernel_ext")
        _MODULE = load_inline(
            name="bounds_prefill_kernel_ext",
            cpp_sources=cpp_source,
            cuda_sources=cuda_source,
            functions=None,
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "-lineinfo"],
            with_cuda=True,
            verbose=False,
        )
    except Exception as exc:
        _LOAD_ERROR = exc
        return None
    return _MODULE


def _require_ext() -> torch.nn.Module:
    mod = _load_ext()
    if mod is None:
        if _LOAD_ERROR is not None:
            raise RuntimeError(
                f"bounds_prefill_kernel_ext unavailable: {_LOAD_ERROR!r}"
            ) from _LOAD_ERROR
        raise RuntimeError("bounds_prefill_kernel_ext unavailable")
    return mod


def compute_bounds_prefill(
    kv_lengths: torch.Tensor,
    seq_full: Optional[torch.Tensor],
    *,
    num_kv_heads: int,
    num_queries_per_kv: int,
    window: int,
    kv_len_total: int,
    sink_cfg: int,
    recent_cfg: int,
    block_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute prefill bounds as [M, G*W] row intervals with one CUDA kernel."""
    mod = _require_ext()
    if kv_lengths.dim() != 3:
        raise ValueError("kv_lengths must be [layers, batch, heads]")
    if kv_lengths.dtype not in (torch.int32, torch.int64):
        kv_lengths = kv_lengths.to(dtype=torch.int32)

    L, B = int(kv_lengths.shape[0]), int(kv_lengths.shape[1])
    H = int(num_kv_heads)
    G = int(num_queries_per_kv)
    W = int(window)
    M = L * B * H
    device = kv_lengths.device

    seq_i32 = None
    sf_stride_l = sf_stride_b = sf_stride_h = 0
    if seq_full is not None:
        seq_i32 = seq_full if seq_full.dtype == torch.int32 else seq_full.to(dtype=torch.int32)
        sf_stride_l, sf_stride_b, sf_stride_h = (int(s) for s in seq_i32.stride())

    kv_len_head = torch.empty((L, B, H), dtype=torch.int32, device=device)
    head_sink = torch.empty_like(kv_len_head)
    recent_start = torch.empty_like(kv_len_head)
    allowed_lengths = torch.empty_like(kv_len_head)
    row_lo = torch.empty((M, G * W), dtype=torch.int32, device=device)
    row_hi = torch.empty_like(row_lo)
    kv_stride_l, kv_stride_b, kv_stride_h = (int(s) for s in kv_lengths.stride())

    mod.compute_bounds_prefill(
        kv_lengths,
        seq_i32,
        kv_len_head,
        head_sink,
        recent_start,
        allowed_lengths,
        row_lo,
        row_hi,
        int(L),
        int(B),
        int(H),
        int(G),
        int(W),
        int(kv_len_total),
        int(sink_cfg),
        int(recent_cfg),
        int(block_size),
        int(kv_stride_l),
        int(kv_stride_b),
        int(kv_stride_h),
        int(sf_stride_l),
        int(sf_stride_b),
        int(sf_stride_h),
    )
    return kv_len_head, head_sink, recent_start, allowed_lengths, row_lo, row_hi


__all__ = ["compute_bounds_prefill"]
