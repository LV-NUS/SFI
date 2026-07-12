"""CUDA extension: compute_bounds_decode kernel for decode-phase selector.

This kernel replaces ATen operations in compute_preproc_bounds with a single CUDA kernel,
eliminating ~2000us of C++ overhead for decode-phase refresh.

Key optimizations:
- No ATen tensor operations (reshape, to, max, clamp, etc.)
- Direct CUDA kernel computation
- Pre-allocated output buffers with dynamic management
"""

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
    prebuilt = load_prebuilt_extension("bounds_kernel_ext")
    if prebuilt is not None:
        _MODULE = prebuilt
        return _MODULE

    cpp_source = r"""
#include <torch/extension.h>
#include <vector>

// Forward declaration of CUDA kernel wrapper (with stride support for non-contiguous inputs)
void compute_bounds_decode_cuda(
    const torch::Tensor& kv_lengths,      // [L, B, H_total] or [L, B, H_kv, G], may be non-contiguous
    const c10::optional<torch::Tensor>& seq_full_opt,  // [L, B, H_kv] or None, may be non-contiguous
    torch::Tensor& kv_len_head_out,       // [L, B, H_kv] pre-allocated
    torch::Tensor& head_sink_out,         // [L, B, H_kv] pre-allocated
    torch::Tensor& recent_start_out,      // [L, B, H_kv] pre-allocated
    torch::Tensor& allowed_lengths_out,   // [L, B, H_kv] pre-allocated
    torch::Tensor& row_lo_out,            // [M, R] pre-allocated, M=L*B*H_kv, R=G
    torch::Tensor& row_hi_out,            // [M, R] pre-allocated
    int64_t L, int64_t B, int64_t H_kv, int64_t G,
    int64_t kv_len_total, int64_t sink_cfg, int64_t recent_cfg, int64_t block_size,
    // Strides for kv_lengths [L, B, H_total]
    int64_t kv_stride_l, int64_t kv_stride_b, int64_t kv_stride_h,
    // Strides for seq_full [L, B, H_kv]
    int64_t sf_stride_l, int64_t sf_stride_b, int64_t sf_stride_h);

// C++ wrapper for Python binding
std::vector<torch::Tensor> compute_bounds_decode(
    torch::Tensor kv_lengths,
    c10::optional<torch::Tensor> seq_full_opt,
    torch::Tensor kv_len_head_out,
    torch::Tensor head_sink_out,
    torch::Tensor recent_start_out,
    torch::Tensor allowed_lengths_out,
    torch::Tensor row_lo_out,
    torch::Tensor row_hi_out,
    int64_t L, int64_t B, int64_t H_kv, int64_t G,
    int64_t kv_len_total, int64_t sink_cfg, int64_t recent_cfg, int64_t block_size,
    // Strides for kv_lengths [L, B, H_total]
    int64_t kv_stride_l, int64_t kv_stride_b, int64_t kv_stride_h,
    // Strides for seq_full [L, B, H_kv]
    int64_t sf_stride_l, int64_t sf_stride_b, int64_t sf_stride_h) {

    compute_bounds_decode_cuda(
        kv_lengths, seq_full_opt,
        kv_len_head_out, head_sink_out, recent_start_out, allowed_lengths_out,
        row_lo_out, row_hi_out,
        L, B, H_kv, G, kv_len_total, sink_cfg, recent_cfg, block_size,
        kv_stride_l, kv_stride_b, kv_stride_h,
        sf_stride_l, sf_stride_b, sf_stride_h);

    // Return views of the output tensors (already modified in-place)
    return {kv_len_head_out, head_sink_out, recent_start_out, allowed_lengths_out, row_lo_out, row_hi_out};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_bounds_decode", &compute_bounds_decode,
          "Compute bounds for decode-phase selector (CUDA kernel)");
}
"""

    cuda_source = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel: compute all bounds for decode phase (with strided input support)
// Each thread handles one (l, b, h_kv) position
// For decode, W=1, so row_lo and row_hi are [M, G] where M = L*B*H_kv
// Supports non-contiguous input tensors via explicit strides
__global__ void compute_bounds_decode_kernel(
    const int64_t* __restrict__ kv_lengths,  // [L, B, H_total] with strides
    const int32_t* __restrict__ seq_full,    // [L, B, H_kv] with strides, or nullptr
    int32_t* __restrict__ kv_len_head,       // [L * B * H_kv] contiguous output
    int32_t* __restrict__ head_sink,         // [L * B * H_kv]
    int32_t* __restrict__ recent_start,      // [L * B * H_kv]
    int32_t* __restrict__ allowed_lengths,   // [L * B * H_kv]
    int32_t* __restrict__ row_lo,            // [M * G]
    int32_t* __restrict__ row_hi,            // [M * G]
    int L, int B, int H_kv, int G, int H_total,
    int kv_len_total, int sink_cfg, int recent_cfg, int block_size,
    // Strides for kv_lengths [L, B, H_total]
    int64_t kv_stride_l, int64_t kv_stride_b, int64_t kv_stride_h,
    // Strides for seq_full [L, B, H_kv] (ignored if seq_full is nullptr)
    int64_t sf_stride_l, int64_t sf_stride_b, int64_t sf_stride_h) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int M = L * B * H_kv;
    if (idx >= M) return;

    // Compute l, b, h_kv from flat output index
    int h_kv = idx % H_kv;
    int tmp = idx / H_kv;
    int b = tmp % B;
    int l = tmp / B;

    // Compute kv_len_head = max over G queries
    // Use strides for correct indexing into potentially non-contiguous kv_lengths
    int max_kv_len = 0;
    for (int g = 0; g < G; ++g) {
        int64_t kv_idx = l * kv_stride_l + b * kv_stride_b + (h_kv * G + g) * kv_stride_h;
        int kv_len = static_cast<int>(kv_lengths[kv_idx]);
        if (kv_len > max_kv_len) {
            max_kv_len = kv_len;
        }
    }
    // Clamp to kv_len_total
    if (max_kv_len > kv_len_total) {
        max_kv_len = kv_len_total;
    }
    kv_len_head[idx] = max_kv_len;

    // head_sink = min(kv_len_head, sink_cfg)
    int h_sink = max_kv_len < sink_cfg ? max_kv_len : sink_cfg;
    if (h_sink < 0) h_sink = 0;
    head_sink[idx] = h_sink;

    // recent_start computation
    int r_start = 0;
    if (block_size > 0 && recent_cfg > 0) {
        int seq_len;
        if (seq_full != nullptr) {
            // Use strides for correct indexing into potentially non-contiguous seq_full
            int64_t sf_idx = l * sf_stride_l + b * sf_stride_b + h_kv * sf_stride_h;
            seq_len = seq_full[sf_idx];
        } else {
            seq_len = max_kv_len;
        }
        int cap = seq_len < recent_cfg ? seq_len : recent_cfg;
        r_start = ((seq_len - cap) / block_size) * block_size;
        if (r_start < 0) r_start = 0;
        if (r_start > kv_len_total) r_start = kv_len_total;
    }
    recent_start[idx] = r_start;

    // allowed_lengths = max(recent_start - head_sink, 0)
    int allowed = r_start - h_sink;
    if (allowed < 0) allowed = 0;
    allowed_lengths[idx] = allowed;

    // row_lo and row_hi for each query g
    // For decode (W=1), row_lo[m, g] = head_sink, row_hi[m, g] = min(kv_len[g], recent_start)
    int row_base = idx * G;
    for (int g = 0; g < G; ++g) {
        int64_t kv_idx = l * kv_stride_l + b * kv_stride_b + (h_kv * G + g) * kv_stride_h;
        int kv_len_g = static_cast<int>(kv_lengths[kv_idx]);
        if (kv_len_g > kv_len_total) kv_len_g = kv_len_total;

        row_lo[row_base + g] = h_sink;

        int hi = kv_len_g < r_start ? kv_len_g : r_start;
        if (hi < 0) hi = 0;
        if (hi > kv_len_total) hi = kv_len_total;
        row_hi[row_base + g] = hi;
    }
}

// Kernel for int32 input (when kv_lengths is already int32)
__global__ void compute_bounds_decode_kernel_i32(
    const int32_t* __restrict__ kv_lengths,  // [L, B, H_total] with strides
    const int32_t* __restrict__ seq_full,    // [L, B, H_kv] with strides, or nullptr
    int32_t* __restrict__ kv_len_head,       // [L * B * H_kv] contiguous output
    int32_t* __restrict__ head_sink,         // [L * B * H_kv]
    int32_t* __restrict__ recent_start,      // [L * B * H_kv]
    int32_t* __restrict__ allowed_lengths,   // [L * B * H_kv]
    int32_t* __restrict__ row_lo,            // [M * G]
    int32_t* __restrict__ row_hi,            // [M * G]
    int L, int B, int H_kv, int G, int H_total,
    int kv_len_total, int sink_cfg, int recent_cfg, int block_size,
    // Strides for kv_lengths [L, B, H_total]
    int64_t kv_stride_l, int64_t kv_stride_b, int64_t kv_stride_h,
    // Strides for seq_full [L, B, H_kv] (ignored if seq_full is nullptr)
    int64_t sf_stride_l, int64_t sf_stride_b, int64_t sf_stride_h) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int M = L * B * H_kv;
    if (idx >= M) return;

    int h_kv = idx % H_kv;
    int tmp = idx / H_kv;
    int b = tmp % B;
    int l = tmp / B;

    // Use strides for correct indexing
    int max_kv_len = 0;
    for (int g = 0; g < G; ++g) {
        int64_t kv_idx = l * kv_stride_l + b * kv_stride_b + (h_kv * G + g) * kv_stride_h;
        int kv_len = kv_lengths[kv_idx];
        if (kv_len > max_kv_len) {
            max_kv_len = kv_len;
        }
    }
    if (max_kv_len > kv_len_total) {
        max_kv_len = kv_len_total;
    }
    kv_len_head[idx] = max_kv_len;

    int h_sink = max_kv_len < sink_cfg ? max_kv_len : sink_cfg;
    if (h_sink < 0) h_sink = 0;
    head_sink[idx] = h_sink;

    int r_start = 0;
    if (block_size > 0 && recent_cfg > 0) {
        int seq_len;
        if (seq_full != nullptr) {
            int64_t sf_idx = l * sf_stride_l + b * sf_stride_b + h_kv * sf_stride_h;
            seq_len = seq_full[sf_idx];
        } else {
            seq_len = max_kv_len;
        }
        int cap = seq_len < recent_cfg ? seq_len : recent_cfg;
        r_start = ((seq_len - cap) / block_size) * block_size;
        if (r_start < 0) r_start = 0;
        if (r_start > kv_len_total) r_start = kv_len_total;
    }
    recent_start[idx] = r_start;

    int allowed = r_start - h_sink;
    if (allowed < 0) allowed = 0;
    allowed_lengths[idx] = allowed;

    int row_base = idx * G;
    for (int g = 0; g < G; ++g) {
        int64_t kv_idx = l * kv_stride_l + b * kv_stride_b + (h_kv * G + g) * kv_stride_h;
        int kv_len_g = kv_lengths[kv_idx];
        if (kv_len_g > kv_len_total) kv_len_g = kv_len_total;

        row_lo[row_base + g] = h_sink;

        int hi = kv_len_g < r_start ? kv_len_g : r_start;
        if (hi < 0) hi = 0;
        if (hi > kv_len_total) hi = kv_len_total;
        row_hi[row_base + g] = hi;
    }
}

void compute_bounds_decode_cuda(
    const torch::Tensor& kv_lengths,
    const c10::optional<torch::Tensor>& seq_full_opt,
    torch::Tensor& kv_len_head_out,
    torch::Tensor& head_sink_out,
    torch::Tensor& recent_start_out,
    torch::Tensor& allowed_lengths_out,
    torch::Tensor& row_lo_out,
    torch::Tensor& row_hi_out,
    int64_t L, int64_t B, int64_t H_kv, int64_t G,
    int64_t kv_len_total, int64_t sink_cfg, int64_t recent_cfg, int64_t block_size,
    // Strides for kv_lengths [L, B, H_total]
    int64_t kv_stride_l, int64_t kv_stride_b, int64_t kv_stride_h,
    // Strides for seq_full [L, B, H_kv]
    int64_t sf_stride_l, int64_t sf_stride_b, int64_t sf_stride_h) {

    // [BOUNDS-EXT-CONTRACT 2026-07-11 EXT审计·仅卫生] C++ 体原零 TORCH_CHECK
    // （校验全在 python wrapper —— 直调 .so 无护栏 = 防守洼地）。补齐入口
    // 合同：dtype 枚举路由（else 臂原为 "Assume int32" 静默放行域）/ CUDA
    // 驻留 / 输出六件 contiguous+int32+numel（kernel 无输出 stride 参数，
    // 线性写 = 死合同）/ 几何为正 + int32 索引域不溢出。
    TORCH_CHECK(kv_lengths.is_cuda(), "kv_lengths must be CUDA");
    TORCH_CHECK(
        kv_lengths.scalar_type() == torch::kInt64 || kv_lengths.scalar_type() == torch::kInt32,
        "kv_lengths must be int64 or int32");
    TORCH_CHECK(L > 0 && B > 0 && H_kv > 0 && G > 0,
                "L/B/H_kv/G must be positive, got ", L, "/", B, "/", H_kv, "/", G);
    const int64_t m64 = L * B * H_kv;
    TORCH_CHECK(m64 * G <= 2147483647LL, "L*B*H_kv*G overflows int32 indexing");
    auto check_out_i32 = [](const torch::Tensor& t, int64_t numel_min, const char* name) {
        TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
        TORCH_CHECK(t.scalar_type() == torch::kInt32, name, " must be int32");
        TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
        TORCH_CHECK(t.numel() >= numel_min, name, " numel ", t.numel(),
                    " smaller than required ", numel_min);
    };
    check_out_i32(kv_len_head_out, m64, "kv_len_head_out");
    check_out_i32(head_sink_out, m64, "head_sink_out");
    check_out_i32(recent_start_out, m64, "recent_start_out");
    check_out_i32(allowed_lengths_out, m64, "allowed_lengths_out");
    check_out_i32(row_lo_out, m64 * G, "row_lo_out");
    check_out_i32(row_hi_out, m64 * G, "row_hi_out");

    int M = L * B * H_kv;
    int H_total = H_kv * G;

    const int threads = 256;
    const int blocks = (M + threads - 1) / threads;

    auto stream = at::cuda::getCurrentCUDAStream();

    const int32_t* seq_full_ptr = nullptr;
    if (seq_full_opt.has_value() && seq_full_opt.value().defined()) {
        TORCH_CHECK(seq_full_opt.value().is_cuda(), "seq_full must be CUDA");
        TORCH_CHECK(seq_full_opt.value().scalar_type() == torch::kInt32,
                    "seq_full must be int32");
        seq_full_ptr = seq_full_opt.value().data_ptr<int32_t>();
    }

    // Dispatch based on kv_lengths dtype
    if (kv_lengths.scalar_type() == torch::kInt64) {
        compute_bounds_decode_kernel<<<blocks, threads, 0, stream>>>(
            kv_lengths.data_ptr<int64_t>(),
            seq_full_ptr,
            kv_len_head_out.data_ptr<int32_t>(),
            head_sink_out.data_ptr<int32_t>(),
            recent_start_out.data_ptr<int32_t>(),
            allowed_lengths_out.data_ptr<int32_t>(),
            row_lo_out.data_ptr<int32_t>(),
            row_hi_out.data_ptr<int32_t>(),
            L, B, H_kv, G, H_total,
            kv_len_total, sink_cfg, recent_cfg, block_size,
            kv_stride_l, kv_stride_b, kv_stride_h,
            sf_stride_l, sf_stride_b, sf_stride_h);
    } else {
        // int32 (dtype domain enforced by the entry CHECK above)
        compute_bounds_decode_kernel_i32<<<blocks, threads, 0, stream>>>(
            kv_lengths.data_ptr<int32_t>(),
            seq_full_ptr,
            kv_len_head_out.data_ptr<int32_t>(),
            head_sink_out.data_ptr<int32_t>(),
            recent_start_out.data_ptr<int32_t>(),
            allowed_lengths_out.data_ptr<int32_t>(),
            row_lo_out.data_ptr<int32_t>(),
            row_hi_out.data_ptr<int32_t>(),
            L, B, H_kv, G, H_total,
            kv_len_total, sink_cfg, recent_cfg, block_size,
            kv_stride_l, kv_stride_b, kv_stride_h,
            sf_stride_l, sf_stride_b, sf_stride_h);
    }
    // (手写 launch check：本文件 include 面未证含 C10_CUDA_KERNEL_LAUNCH_CHECK
    // 所在头，用 cudaGetLastError 等价形态，零新增 include。)
    const cudaError_t launch_err = cudaGetLastError();
    TORCH_CHECK(launch_err == cudaSuccess,
                "compute_bounds_decode launch failed: ", cudaGetErrorString(launch_err));
    }
"""

    _ensure_torch_cuda_arch_list()
    try:
        # [EXT-NVCC-GUARD] JIT 回落前 pin nvcc+版本预检(系统 nvcc 10.1 坑根修);
        # 失败信息进 _LOAD_ERROR,由 require/enabled 语义原样呈报。
        configure_jit_toolchain_or_raise(ext_name="bounds_kernel_ext")
        _MODULE = load_inline(
            name="bounds_kernel_ext",
            cpp_sources=cpp_source,
            cuda_sources=cuda_source,
            functions=None,
            extra_cuda_cflags=["-lineinfo"],
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
        err_msg = f"bounds_kernel_ext unavailable: {_LOAD_ERROR}" if _LOAD_ERROR else "bounds_kernel_ext unavailable"
        raise RuntimeError(err_msg)
    return mod


def compute_bounds_decode(
    kv_lengths: torch.Tensor,
    seq_full: Optional[torch.Tensor],
    *,
    num_kv_heads: int,
    num_queries_per_kv: int,
    kv_len_total: int,
    sink_cfg: int,
    recent_cfg: int,
    block_size: int,
    out: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute bounds for decode-phase selector using CUDA kernel.

    This replaces the ATen-heavy compute_preproc_bounds for decode phase.

    Args:
        kv_lengths: [L, B, H_total] where H_total = H_kv * G
        seq_full: [L, B, H_kv] or None
        num_kv_heads: H_kv
        num_queries_per_kv: G (typically 1 for MQA, >1 for GQA)
        kv_len_total: maximum KV length
        sink_cfg: sink token count
        recent_cfg: recent token count
        block_size: KV cache block size

    Returns:
        (kv_len_head, head_sink, recent_start, allowed_lengths, row_lo, row_hi)
        - kv_len_head: [L, B, H_kv]
        - head_sink: [L, B, H_kv]
        - recent_start: [L, B, H_kv]
        - allowed_lengths: [L, B, H_kv]
        - row_lo: [M, G] where M = L * B * H_kv
        - row_hi: [M, G]
    """
    mod = _require_ext()

    # Extract strides for kv_lengths (supports non-contiguous/broadcast views without memory copy)
    # Strides are in element units, not bytes
    kv_stride = kv_lengths.stride()
    kv_stride_l = kv_stride[0]
    kv_stride_b = kv_stride[1]
    kv_stride_h = kv_stride[2]

    # Extract strides for seq_full if provided (also supports non-contiguous)
    # Default to 0 strides if seq_full is None (ignored by kernel)
    if seq_full is not None:
        sf_stride = seq_full.stride()
        sf_stride_l = sf_stride[0]
        sf_stride_b = sf_stride[1]
        sf_stride_h = sf_stride[2]
    else:
        sf_stride_l = sf_stride_b = sf_stride_h = 0

    L, B = kv_lengths.shape[0], kv_lengths.shape[1]
    H_kv = num_kv_heads
    G = num_queries_per_kv
    M = L * B * H_kv

    device = kv_lengths.device

    shape_3d = (int(L), int(B), int(H_kv))
    shape_2d = (int(M), int(G))
    if out is None:
        kv_len_head = torch.empty(shape_3d, dtype=torch.int32, device=device)
        head_sink = torch.empty_like(kv_len_head)
        recent_start = torch.empty_like(kv_len_head)
        allowed_lengths = torch.empty_like(kv_len_head)
        row_lo = torch.empty(shape_2d, dtype=torch.int32, device=device)
        row_hi = torch.empty_like(row_lo)
    else:
        if len(out) != 6:
            raise ValueError("decode bounds output tuple must contain 6 tensors")
        kv_len_head, head_sink, recent_start, allowed_lengths, row_lo, row_hi = out
        expected = (
            (kv_len_head, shape_3d),
            (head_sink, shape_3d),
            (recent_start, shape_3d),
            (allowed_lengths, shape_3d),
            (row_lo, shape_2d),
            (row_hi, shape_2d),
        )
        for tensor, shape in expected:
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.device != device
                or tensor.dtype != torch.int32
                or tuple(int(v) for v in tensor.shape) != shape
                or not tensor.is_contiguous()
            ):
                raise ValueError(
                    "decode bounds output tensors must be contiguous int32 tensors "
                    f"on {device} with shapes {shape_3d} and {shape_2d}"
                )

    # Ensure seq_full is int32 if provided
    # Note: dtype conversion may change strides, so we recalculate after conversion
    seq_full_i32 = None
    if seq_full is not None:
        if seq_full.dtype != torch.int32:
            seq_full_i32 = seq_full.to(dtype=torch.int32)
        else:
            seq_full_i32 = seq_full
        # Get strides of the actual tensor passed to kernel (after potential dtype conversion)
        sf_stride_i32 = seq_full_i32.stride()
        sf_stride_l = sf_stride_i32[0]
        sf_stride_b = sf_stride_i32[1]
        sf_stride_h = sf_stride_i32[2]

    # Call CUDA kernel with stride parameters
    mod.compute_bounds_decode(
        kv_lengths,
        seq_full_i32,
        kv_len_head,
        head_sink,
        recent_start,
        allowed_lengths,
        row_lo,
        row_hi,
        int(L), int(B), int(H_kv), int(G),
        int(kv_len_total), int(sink_cfg), int(recent_cfg), int(block_size),
        # kv_lengths strides
        int(kv_stride_l), int(kv_stride_b), int(kv_stride_h),
        # seq_full strides (0 if seq_full is None)
        int(sf_stride_l), int(sf_stride_b), int(sf_stride_h),
    )

    return kv_len_head, head_sink, recent_start, allowed_lengths, row_lo, row_hi


__all__ = [
    "compute_bounds_decode",
]
