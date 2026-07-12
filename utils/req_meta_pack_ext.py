"""CUDA extension: req_meta pack family (decode fast / decode fast layers / prefill fast layers).

[#17-P1 PACK-CUDA 2026-07-12] 生产主链 req_meta pack 从 Triton JIT 迁 C++/CUDA
单发射。三个 kernel 是 `triton_kernel/flash_attn_score_dump_fwd.py` 中
`_kernel_pack_req_meta_decode_fast` / `_kernel_pack_req_meta_decode_fast_layers`
/ `_kernel_pack_req_meta_prefill_fast_layers` 的逐语句直译（纯整数运算，
输出与 Triton 版逐位等价；对拍单测=tests/test_req_meta_pack_ext_bitwise.py）。

迁移动机：Triton launcher 的 host 端开销（python 参数打包+constexpr 特化缓存
查找）远高于 C++ ext 单发射（#13 取证时代实证 C++ 窗 ~128µs 量级可行）。

语义合同（与 Triton 版 bug-for-bug 一致）:
- 所有 1D 输入按 element 连续消费（Triton 版 `ptr + pid` 同款），host 侧
  TORCH_CHECK 连续性 fail-fast（生产全 contiguous，收紧仅曝光病态调用）。
- meta 输出列内偏移恒 +1（Triton 版 `base + 0..6` 同款），行/层 stride 显式传入。
- `log_f_mask * layer_logf_enable != 0` 的 i32 wrap 乘语义用 uint32 乘直译。
- sink_shift 由 utils.req_meta_flag_codec.REQ_META_SINK_SHIFT 单一真源传入。

无 fallback：ext 不可用即 raise（pack 本就 GPU-only；Triton 版同样无 CPU 路径）。
"""

from __future__ import annotations

import os
from typing import Optional

import torch
from torch.utils.cpp_extension import load_inline

from utils.ext_toolchain import configure_jit_toolchain_or_raise
from utils.torch_extension_cache import load_prebuilt_extension

_MODULE = None
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


_CPP_SOURCE = r"""
#include <torch/extension.h>
#include <cstdint>

void pack_req_meta_decode_fast_cuda(
    const torch::Tensor& seqused_k,
    const torch::Tensor& is_compact_i32,
    const torch::Tensor& compact_kv_len_i32,
    const torch::Tensor& compact_offset_tokens_i64,
    torch::Tensor& req_meta_i32,
    torch::Tensor& req_meta_i64,
    int64_t meta_i32_stride0,
    int64_t meta_i64_stride0,
    int64_t batch,
    int64_t block_size,
    int64_t recent_cap,
    int64_t sink_tokens,
    int64_t sink_shift);

void pack_req_meta_decode_fast_layers_cuda(
    const torch::Tensor& seqused_k,
    const torch::Tensor& is_compact_i32,
    const torch::Tensor& compact_kv_len_i32,
    const torch::Tensor& compact_offset_tokens_i64,
    const torch::Tensor& log_f_mask_i32,
    const torch::Tensor& log_f_q_lens_i32,
    int64_t log_f_stride_head,
    const torch::Tensor& capture_row_by_batch_row_buf0_i32,
    const torch::Tensor& capture_row_by_batch_row_buf1_i32,
    int64_t scores_base_ptr_buf0,
    int64_t scores_base_ptr_buf1,
    int64_t scores_stride_chunk_bytes_buf0,
    int64_t scores_stride_chunk_bytes_buf1,
    int64_t scores_stride_slot_bytes_buf0,
    int64_t scores_stride_slot_bytes_buf1,
    const torch::Tensor& buf_id_by_layer_i32,
    const torch::Tensor& slot_in_chunk_by_layer_i32,
    const torch::Tensor& layer_logf_enable_i32,
    torch::Tensor& req_meta_i32,
    torch::Tensor& req_meta_i64,
    int64_t is_compact_stride0, int64_t is_compact_stride1,
    int64_t compact_kv_len_stride0, int64_t compact_kv_len_stride1,
    int64_t compact_offset_stride0, int64_t compact_offset_stride1,
    int64_t meta_i32_stride0, int64_t meta_i32_stride1,
    int64_t meta_i64_stride0, int64_t meta_i64_stride1,
    int64_t layers,
    int64_t batch,
    int64_t block_size,
    int64_t recent_cap,
    int64_t sink_tokens,
    int64_t sink_shift);

void pack_req_meta_prefill_fast_layers_cuda(
    const torch::Tensor& seqused_k,
    const torch::Tensor& cu_seqlens_q,
    const torch::Tensor& log_f_last_n_i32,
    const torch::Tensor& log_f_capacity_i32,
    int64_t log_f_stride_head,
    const torch::Tensor& capture_row_by_batch_row_buf0_i32,
    const torch::Tensor& capture_row_by_batch_row_buf1_i32,
    int64_t scores_base_ptr_buf0,
    int64_t scores_base_ptr_buf1,
    int64_t scores_stride_chunk_bytes_buf0,
    int64_t scores_stride_chunk_bytes_buf1,
    int64_t scores_stride_slot_bytes_buf0,
    int64_t scores_stride_slot_bytes_buf1,
    int64_t denoms_base_ptr_buf0,
    int64_t denoms_base_ptr_buf1,
    int64_t denoms_stride_chunk_bytes_buf0,
    int64_t denoms_stride_chunk_bytes_buf1,
    int64_t denoms_stride_slot_bytes_buf0,
    int64_t denoms_stride_slot_bytes_buf1,
    const torch::Tensor& buf_id_by_layer_i32,
    const torch::Tensor& slot_in_chunk_by_layer_i32,
    torch::Tensor& req_meta_i32,
    torch::Tensor& req_meta_i64,
    int64_t meta_i32_stride0, int64_t meta_i32_stride1,
    int64_t meta_i64_stride0, int64_t meta_i64_stride1,
    int64_t layers,
    int64_t batch,
    int64_t recent_cap,
    int64_t sink_tokens,
    int64_t sink_shift);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pack_req_meta_decode_fast", &pack_req_meta_decode_fast_cuda,
          "Pack req_meta for decode fast-path, single layer (CUDA)");
    m.def("pack_req_meta_decode_fast_layers", &pack_req_meta_decode_fast_layers_cuda,
          "Pack req_meta for decode fast-path, all layers (CUDA)");
    m.def("pack_req_meta_prefill_fast_layers", &pack_req_meta_prefill_fast_layers_cuda,
          "Pack req_meta for prefill fast-path, all layers (CUDA)");
}
"""

_CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {

constexpr int32_t kRecentCapFlag = 4;   // RECENT_CAP_FLAG
constexpr int32_t kLogfFlag = 8;        // LOGF_FLAG
constexpr int32_t kFullContextFlag = 16; // FULL_CONTEXT_FLAG

__device__ __forceinline__ int32_t imax_i32(int32_t a, int32_t b) {
    return a > b ? a : b;
}
__device__ __forceinline__ int64_t imax_i64(int64_t a, int64_t b) {
    return a > b ? a : b;
}

// -----------------------------------------------------------------------------
// decode fast (single layer): 1 thread = 1 batch row.
// Bitwise mirror of _kernel_pack_req_meta_decode_fast.
// -----------------------------------------------------------------------------
template <typename TSeq>
__global__ void pack_decode_fast_kernel(
    const TSeq* __restrict__ seqused_k,
    const int32_t* __restrict__ is_compact,
    const int32_t* __restrict__ compact_kv_len,
    const int64_t* __restrict__ compact_offset_tokens,
    int32_t* __restrict__ req_meta_i32,
    int64_t* __restrict__ req_meta_i64,
    int64_t meta_i32_stride0,
    int64_t meta_i64_stride0,
    int32_t batch,
    int32_t block_size,
    int32_t recent_cap,
    int32_t sink_tokens,
    int32_t sink_shift) {
    const int32_t pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= batch) return;

    int32_t kv_len_visible = imax_i32(static_cast<int32_t>(seqused_k[pid]), 0);
    int32_t compact = (is_compact[pid] != 0) ? 1 : 0;
    int32_t ckl = imax_i32(compact_kv_len[pid], 0);
    int64_t off_tokens = imax_i64(compact_offset_tokens[pid], 0);

    const int32_t compact_block_cnt = (ckl + (block_size - 1)) / block_size;
    const int32_t sink_bits = imax_i32(sink_tokens, 0) << sink_shift;

    int32_t* m32 = req_meta_i32 + static_cast<int64_t>(pid) * meta_i32_stride0;
    m32[0] = kv_len_visible;
    m32[1] = compact_block_cnt;
    m32[2] = ckl;  // logits_last_n column reused to carry compact_kv_len
    m32[3] = 0;
    m32[4] = kv_len_visible;
    int32_t flags = (compact != 0) ? (compact | kRecentCapFlag) : kFullContextFlag;
    flags |= sink_bits;
    m32[5] = flags;
    m32[6] = recent_cap;

    int64_t* m64 = req_meta_i64 + static_cast<int64_t>(pid) * meta_i64_stride0;
    m64[0] = static_cast<int64_t>(pid);
    m64[1] = off_tokens / static_cast<int64_t>(block_size);
    m64[2] = 0;
    m64[3] = off_tokens;
}

// -----------------------------------------------------------------------------
// decode fast layers: 1 thread = 1 (layer, batch row).
// Bitwise mirror of _kernel_pack_req_meta_decode_fast_layers.
// -----------------------------------------------------------------------------
template <typename TSeq>
__global__ void pack_decode_fast_layers_kernel(
    const TSeq* __restrict__ seqused_k,
    const int32_t* __restrict__ is_compact,
    const int32_t* __restrict__ compact_kv_len,
    const int64_t* __restrict__ compact_offset_tokens,
    const int32_t* __restrict__ log_f_mask,
    const int32_t* __restrict__ log_f_q_lens,
    int32_t log_f_stride_head_i32,
    const int32_t* __restrict__ capture_row_buf0,
    const int32_t* __restrict__ capture_row_buf1,
    int64_t scores_base_ptr_buf0,
    int64_t scores_base_ptr_buf1,
    int64_t scores_stride_chunk_bytes_buf0,
    int64_t scores_stride_chunk_bytes_buf1,
    int64_t scores_stride_slot_bytes_buf0,
    int64_t scores_stride_slot_bytes_buf1,
    const int32_t* __restrict__ buf_id_by_layer,
    const int32_t* __restrict__ slot_in_chunk_by_layer,
    const int32_t* __restrict__ layer_logf_enable,
    int32_t* __restrict__ req_meta_i32,
    int64_t* __restrict__ req_meta_i64,
    int64_t is_compact_stride0, int64_t is_compact_stride1,
    int64_t compact_kv_len_stride0, int64_t compact_kv_len_stride1,
    int64_t compact_offset_stride0, int64_t compact_offset_stride1,
    int64_t meta_i32_stride0, int64_t meta_i32_stride1,
    int64_t meta_i64_stride0, int64_t meta_i64_stride1,
    int32_t layers,
    int32_t batch,
    int32_t block_size,
    int32_t recent_cap,
    int32_t sink_tokens,
    int32_t sink_shift) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = static_cast<int64_t>(layers) * batch;
    if (idx >= total) return;
    const int32_t layer = static_cast<int32_t>(idx / batch);
    const int32_t pid = static_cast<int32_t>(idx % batch);

    int32_t kv_len_visible = imax_i32(static_cast<int32_t>(seqused_k[pid]), 0);

    int32_t compact =
        (is_compact[layer * is_compact_stride0 + pid * is_compact_stride1] != 0) ? 1 : 0;

    int32_t ckl = imax_i32(
        compact_kv_len[layer * compact_kv_len_stride0 + pid * compact_kv_len_stride1], 0);

    // compact override: gated-out 层的 refresh 行走 compact（Triton 版嵌套 if 直译）
    const int32_t log_f_mask_early = log_f_mask[pid];
    const int32_t layer_logf_en_early = layer_logf_enable[layer];
    if (log_f_mask_early != 0) {
        if (layer_logf_en_early == 0) {
            if (ckl > 0) {
                compact = 1;
            }
        }
    }

    int64_t off_tokens = imax_i64(
        compact_offset_tokens[layer * compact_offset_stride0 + pid * compact_offset_stride1], 0);

    const int32_t compact_block_cnt = (ckl + (block_size - 1)) / block_size;
    const int32_t sink_bits = imax_i32(sink_tokens, 0) << sink_shift;

    int32_t* m32 = req_meta_i32 + layer * meta_i32_stride0 + pid * meta_i32_stride1;
    m32[0] = kv_len_visible;
    m32[1] = compact_block_cnt;
    m32[2] = ckl;
    m32[3] = 0;
    m32[4] = kv_len_visible;
    int32_t flags = (compact != 0) ? (compact | kRecentCapFlag) : kFullContextFlag;
    flags |= sink_bits;
    m32[5] = flags;
    m32[6] = recent_cap;

    int64_t* m64 = req_meta_i64 + layer * meta_i64_stride0 + pid * meta_i64_stride1;
    m64[0] = static_cast<int64_t>(pid);
    m64[1] = off_tokens / static_cast<int64_t>(block_size);
    m64[2] = 0;
    m64[3] = off_tokens;

    // dense log_f meta override（i32 wrap 乘语义用 uint32 直译）
    const uint32_t logf_and_en =
        static_cast<uint32_t>(log_f_mask_early) * static_cast<uint32_t>(layer_logf_en_early);
    if (logf_and_en != 0u) {
        const int32_t buf_id = buf_id_by_layer[layer];
        const int64_t slot_in_chunk = static_cast<int64_t>(slot_in_chunk_by_layer[layer]);
        int32_t q_len = imax_i32(log_f_q_lens[pid], 1);
        const int32_t row_offset = imax_i32(q_len - 1, 0);
        m32[1] = log_f_stride_head_i32;
        m32[2] = 1;  // logits_last_n
        m32[3] = row_offset;
        m32[4] = kv_len_visible;
        m32[5] = (kRecentCapFlag | kLogfFlag) | sink_bits;
        m32[6] = recent_cap;
        m64[1] = 0;
        const int64_t capture_row0 = static_cast<int64_t>(capture_row_buf0[pid]);
        const int64_t capture_row1 = static_cast<int64_t>(capture_row_buf1[pid]);
        const int64_t capture_row = (buf_id == 0) ? capture_row0 : capture_row1;
        const int64_t base0_buf0 =
            scores_base_ptr_buf0 + slot_in_chunk * scores_stride_chunk_bytes_buf0;
        const int64_t base0_buf1 =
            scores_base_ptr_buf1 + slot_in_chunk * scores_stride_chunk_bytes_buf1;
        const int64_t base0 = (buf_id == 0) ? base0_buf0 : base0_buf1;
        const int64_t stride_slot_bytes =
            (buf_id == 0) ? scores_stride_slot_bytes_buf0 : scores_stride_slot_bytes_buf1;
        const int64_t out_ptr = base0 + capture_row * stride_slot_bytes;
        m64[2] = out_ptr;
        m64[3] = 0;
    }
}

// -----------------------------------------------------------------------------
// prefill fast layers: 1 thread = 1 (layer, batch row).
// Bitwise mirror of _kernel_pack_req_meta_prefill_fast_layers.
// -----------------------------------------------------------------------------
template <typename TSeq, typename TCu>
__global__ void pack_prefill_fast_layers_kernel(
    const TSeq* __restrict__ seqused_k,
    const TCu* __restrict__ cu_seqlens_q,
    const int32_t* __restrict__ log_f_last_n,
    const int32_t* __restrict__ log_f_capacity,
    int32_t log_f_stride_head_i32,
    const int32_t* __restrict__ capture_row_buf0,
    const int32_t* __restrict__ capture_row_buf1,
    int64_t scores_base_ptr_buf0,
    int64_t scores_base_ptr_buf1,
    int64_t scores_stride_chunk_bytes_buf0,
    int64_t scores_stride_chunk_bytes_buf1,
    int64_t scores_stride_slot_bytes_buf0,
    int64_t scores_stride_slot_bytes_buf1,
    int64_t denoms_base_ptr_buf0,
    int64_t denoms_base_ptr_buf1,
    int64_t denoms_stride_chunk_bytes_buf0,
    int64_t denoms_stride_chunk_bytes_buf1,
    int64_t denoms_stride_slot_bytes_buf0,
    int64_t denoms_stride_slot_bytes_buf1,
    const int32_t* __restrict__ buf_id_by_layer,
    const int32_t* __restrict__ slot_in_chunk_by_layer,
    int32_t* __restrict__ req_meta_i32,
    int64_t* __restrict__ req_meta_i64,
    int64_t meta_i32_stride0, int64_t meta_i32_stride1,
    int64_t meta_i64_stride0, int64_t meta_i64_stride1,
    int32_t layers,
    int32_t batch,
    int32_t recent_cap,
    int32_t sink_tokens,
    int32_t sink_shift) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = static_cast<int64_t>(layers) * batch;
    if (idx >= total) return;
    const int32_t layer = static_cast<int32_t>(idx / batch);
    const int32_t pid = static_cast<int32_t>(idx % batch);

    const int32_t kv_len_visible = imax_i32(static_cast<int32_t>(seqused_k[pid]), 0);

    const int32_t q0 = static_cast<int32_t>(cu_seqlens_q[pid]);
    const int32_t q1 = static_cast<int32_t>(cu_seqlens_q[pid + 1]);
    const int32_t q_len = imax_i32(q1 - q0, 1);

    const int32_t last_n = imax_i32(log_f_last_n[pid], 0);
    const int32_t cap = imax_i32(log_f_capacity[pid], 0);

    // Default: dense no-log_f
    int32_t* m32 = req_meta_i32 + layer * meta_i32_stride0 + pid * meta_i32_stride1;
    m32[0] = kv_len_visible;
    m32[1] = 0;
    m32[2] = 0;
    m32[3] = 0;
    m32[4] = 0;
    m32[5] = 0;
    m32[6] = 0;

    int64_t* m64 = req_meta_i64 + layer * meta_i64_stride0 + pid * meta_i64_stride1;
    m64[0] = static_cast<int64_t>(pid);
    m64[1] = 0;
    m64[2] = 0;
    m64[3] = 0;

    if (last_n > 0) {
        const int32_t row_offset = imax_i32(q_len - last_n, 0);
        int32_t cap_eff = kv_len_visible < cap ? kv_len_visible : cap;
        cap_eff = imax_i32(cap_eff, 0);

        const int32_t sink_bits = imax_i32(sink_tokens, 0) << sink_shift;
        m32[1] = log_f_stride_head_i32;
        m32[2] = last_n;
        m32[3] = row_offset;
        m32[4] = cap_eff;
        m32[5] = (kRecentCapFlag | kLogfFlag) | sink_bits;
        m32[6] = recent_cap;

        const int32_t buf_id = buf_id_by_layer[layer];
        const int64_t slot_in_chunk = static_cast<int64_t>(slot_in_chunk_by_layer[layer]);
        const int64_t capture_row0 = static_cast<int64_t>(capture_row_buf0[pid]);
        const int64_t capture_row1 = static_cast<int64_t>(capture_row_buf1[pid]);
        const int64_t capture_row = (buf_id == 0) ? capture_row0 : capture_row1;

        const int64_t base0_scores_buf0 =
            scores_base_ptr_buf0 + slot_in_chunk * scores_stride_chunk_bytes_buf0;
        const int64_t base0_scores_buf1 =
            scores_base_ptr_buf1 + slot_in_chunk * scores_stride_chunk_bytes_buf1;
        const int64_t base0_scores = (buf_id == 0) ? base0_scores_buf0 : base0_scores_buf1;
        const int64_t stride_slot_scores =
            (buf_id == 0) ? scores_stride_slot_bytes_buf0 : scores_stride_slot_bytes_buf1;
        const int64_t out_ptr = base0_scores + capture_row * stride_slot_scores;
        m64[2] = out_ptr;

        if (last_n > 1) {
            const int64_t base0_den_buf0 =
                denoms_base_ptr_buf0 + slot_in_chunk * denoms_stride_chunk_bytes_buf0;
            const int64_t base0_den_buf1 =
                denoms_base_ptr_buf1 + slot_in_chunk * denoms_stride_chunk_bytes_buf1;
            const int64_t base0_den = (buf_id == 0) ? base0_den_buf0 : base0_den_buf1;
            const int64_t stride_slot_den =
                (buf_id == 0) ? denoms_stride_slot_bytes_buf0 : denoms_stride_slot_bytes_buf1;
            const int64_t denom_ptr = base0_den + capture_row * stride_slot_den;
            m64[3] = denom_ptr;
        }
    }
}

inline void check_seq_dtype(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.scalar_type() == torch::kInt32 || t.scalar_type() == torch::kInt64,
                name, " must be int32 or int64");
}

inline void check_i32_contig1d(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.scalar_type() == torch::kInt32, name, " must be int32");
    TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(t.numel() <= 1 || t.stride(-1) == 1, name,
                " must be element-contiguous (stride(-1)==1)");
}

inline void check_contig_last(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(t.numel() <= 1 || t.stride(-1) == 1, name,
                " must be element-contiguous along last dim");
}

}  // namespace

void pack_req_meta_decode_fast_cuda(
    const torch::Tensor& seqused_k,
    const torch::Tensor& is_compact_i32,
    const torch::Tensor& compact_kv_len_i32,
    const torch::Tensor& compact_offset_tokens_i64,
    torch::Tensor& req_meta_i32,
    torch::Tensor& req_meta_i64,
    int64_t meta_i32_stride0,
    int64_t meta_i64_stride0,
    int64_t batch,
    int64_t block_size,
    int64_t recent_cap,
    int64_t sink_tokens,
    int64_t sink_shift) {
    check_seq_dtype(seqused_k, "seqused_k");
    check_contig_last(seqused_k, "seqused_k");
    check_i32_contig1d(is_compact_i32, "is_compact_i32");
    check_i32_contig1d(compact_kv_len_i32, "compact_kv_len_i32");
    TORCH_CHECK(compact_offset_tokens_i64.scalar_type() == torch::kInt64,
                "compact_offset_tokens_i64 must be int64");
    check_contig_last(compact_offset_tokens_i64, "compact_offset_tokens_i64");
    TORCH_CHECK(req_meta_i32.is_cuda() && req_meta_i64.is_cuda(),
                "req_meta tensors must be CUDA");

    const int threads = 256;
    const int blocks = static_cast<int>((batch + threads - 1) / threads);
    auto stream = at::cuda::getCurrentCUDAStream();

    if (seqused_k.scalar_type() == torch::kInt64) {
        pack_decode_fast_kernel<int64_t><<<blocks, threads, 0, stream>>>(
            seqused_k.data_ptr<int64_t>(),
            is_compact_i32.data_ptr<int32_t>(),
            compact_kv_len_i32.data_ptr<int32_t>(),
            compact_offset_tokens_i64.data_ptr<int64_t>(),
            req_meta_i32.data_ptr<int32_t>(),
            req_meta_i64.data_ptr<int64_t>(),
            meta_i32_stride0, meta_i64_stride0,
            static_cast<int32_t>(batch), static_cast<int32_t>(block_size),
            static_cast<int32_t>(recent_cap), static_cast<int32_t>(sink_tokens),
            static_cast<int32_t>(sink_shift));
    } else {
        pack_decode_fast_kernel<int32_t><<<blocks, threads, 0, stream>>>(
            seqused_k.data_ptr<int32_t>(),
            is_compact_i32.data_ptr<int32_t>(),
            compact_kv_len_i32.data_ptr<int32_t>(),
            compact_offset_tokens_i64.data_ptr<int64_t>(),
            req_meta_i32.data_ptr<int32_t>(),
            req_meta_i64.data_ptr<int64_t>(),
            meta_i32_stride0, meta_i64_stride0,
            static_cast<int32_t>(batch), static_cast<int32_t>(block_size),
            static_cast<int32_t>(recent_cap), static_cast<int32_t>(sink_tokens),
            static_cast<int32_t>(sink_shift));
    }
    const cudaError_t launch_err = cudaGetLastError();
    TORCH_CHECK(launch_err == cudaSuccess,
                "pack_req_meta_decode_fast launch failed: ", cudaGetErrorString(launch_err));
}

void pack_req_meta_decode_fast_layers_cuda(
    const torch::Tensor& seqused_k,
    const torch::Tensor& is_compact_i32,
    const torch::Tensor& compact_kv_len_i32,
    const torch::Tensor& compact_offset_tokens_i64,
    const torch::Tensor& log_f_mask_i32,
    const torch::Tensor& log_f_q_lens_i32,
    int64_t log_f_stride_head,
    const torch::Tensor& capture_row_by_batch_row_buf0_i32,
    const torch::Tensor& capture_row_by_batch_row_buf1_i32,
    int64_t scores_base_ptr_buf0,
    int64_t scores_base_ptr_buf1,
    int64_t scores_stride_chunk_bytes_buf0,
    int64_t scores_stride_chunk_bytes_buf1,
    int64_t scores_stride_slot_bytes_buf0,
    int64_t scores_stride_slot_bytes_buf1,
    const torch::Tensor& buf_id_by_layer_i32,
    const torch::Tensor& slot_in_chunk_by_layer_i32,
    const torch::Tensor& layer_logf_enable_i32,
    torch::Tensor& req_meta_i32,
    torch::Tensor& req_meta_i64,
    int64_t is_compact_stride0, int64_t is_compact_stride1,
    int64_t compact_kv_len_stride0, int64_t compact_kv_len_stride1,
    int64_t compact_offset_stride0, int64_t compact_offset_stride1,
    int64_t meta_i32_stride0, int64_t meta_i32_stride1,
    int64_t meta_i64_stride0, int64_t meta_i64_stride1,
    int64_t layers,
    int64_t batch,
    int64_t block_size,
    int64_t recent_cap,
    int64_t sink_tokens,
    int64_t sink_shift) {
    check_seq_dtype(seqused_k, "seqused_k");
    check_contig_last(seqused_k, "seqused_k");
    TORCH_CHECK(is_compact_i32.scalar_type() == torch::kInt32, "is_compact_i32 must be int32");
    TORCH_CHECK(compact_kv_len_i32.scalar_type() == torch::kInt32,
                "compact_kv_len_i32 must be int32");
    TORCH_CHECK(compact_offset_tokens_i64.scalar_type() == torch::kInt64,
                "compact_offset_tokens_i64 must be int64");
    check_i32_contig1d(log_f_mask_i32, "log_f_mask_i32");
    check_i32_contig1d(log_f_q_lens_i32, "log_f_q_lens_i32");
    check_i32_contig1d(capture_row_by_batch_row_buf0_i32, "capture_row_by_batch_row_buf0_i32");
    check_i32_contig1d(capture_row_by_batch_row_buf1_i32, "capture_row_by_batch_row_buf1_i32");
    check_i32_contig1d(buf_id_by_layer_i32, "buf_id_by_layer_i32");
    check_i32_contig1d(slot_in_chunk_by_layer_i32, "slot_in_chunk_by_layer_i32");
    check_i32_contig1d(layer_logf_enable_i32, "layer_logf_enable_i32");
    TORCH_CHECK(req_meta_i32.is_cuda() && req_meta_i64.is_cuda(),
                "req_meta tensors must be CUDA");

    const int64_t total = layers * batch;
    const int threads = 256;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    auto stream = at::cuda::getCurrentCUDAStream();

#define SFI_LAUNCH_DECODE_LAYERS(TSEQ)                                                   \
    pack_decode_fast_layers_kernel<TSEQ><<<blocks, threads, 0, stream>>>(                \
        seqused_k.data_ptr<TSEQ>(),                                                      \
        is_compact_i32.data_ptr<int32_t>(),                                              \
        compact_kv_len_i32.data_ptr<int32_t>(),                                          \
        compact_offset_tokens_i64.data_ptr<int64_t>(),                                   \
        log_f_mask_i32.data_ptr<int32_t>(),                                              \
        log_f_q_lens_i32.data_ptr<int32_t>(),                                            \
        static_cast<int32_t>(log_f_stride_head),                                         \
        capture_row_by_batch_row_buf0_i32.data_ptr<int32_t>(),                           \
        capture_row_by_batch_row_buf1_i32.data_ptr<int32_t>(),                           \
        scores_base_ptr_buf0, scores_base_ptr_buf1,                                      \
        scores_stride_chunk_bytes_buf0, scores_stride_chunk_bytes_buf1,                  \
        scores_stride_slot_bytes_buf0, scores_stride_slot_bytes_buf1,                    \
        buf_id_by_layer_i32.data_ptr<int32_t>(),                                         \
        slot_in_chunk_by_layer_i32.data_ptr<int32_t>(),                                  \
        layer_logf_enable_i32.data_ptr<int32_t>(),                                       \
        req_meta_i32.data_ptr<int32_t>(),                                                \
        req_meta_i64.data_ptr<int64_t>(),                                                \
        is_compact_stride0, is_compact_stride1,                                          \
        compact_kv_len_stride0, compact_kv_len_stride1,                                  \
        compact_offset_stride0, compact_offset_stride1,                                  \
        meta_i32_stride0, meta_i32_stride1,                                              \
        meta_i64_stride0, meta_i64_stride1,                                              \
        static_cast<int32_t>(layers), static_cast<int32_t>(batch),                       \
        static_cast<int32_t>(block_size), static_cast<int32_t>(recent_cap),              \
        static_cast<int32_t>(sink_tokens), static_cast<int32_t>(sink_shift))

    if (seqused_k.scalar_type() == torch::kInt64) {
        SFI_LAUNCH_DECODE_LAYERS(int64_t);
    } else {
        SFI_LAUNCH_DECODE_LAYERS(int32_t);
    }
#undef SFI_LAUNCH_DECODE_LAYERS
    const cudaError_t launch_err = cudaGetLastError();
    TORCH_CHECK(launch_err == cudaSuccess,
                "pack_req_meta_decode_fast_layers launch failed: ",
                cudaGetErrorString(launch_err));
}

void pack_req_meta_prefill_fast_layers_cuda(
    const torch::Tensor& seqused_k,
    const torch::Tensor& cu_seqlens_q,
    const torch::Tensor& log_f_last_n_i32,
    const torch::Tensor& log_f_capacity_i32,
    int64_t log_f_stride_head,
    const torch::Tensor& capture_row_by_batch_row_buf0_i32,
    const torch::Tensor& capture_row_by_batch_row_buf1_i32,
    int64_t scores_base_ptr_buf0,
    int64_t scores_base_ptr_buf1,
    int64_t scores_stride_chunk_bytes_buf0,
    int64_t scores_stride_chunk_bytes_buf1,
    int64_t scores_stride_slot_bytes_buf0,
    int64_t scores_stride_slot_bytes_buf1,
    int64_t denoms_base_ptr_buf0,
    int64_t denoms_base_ptr_buf1,
    int64_t denoms_stride_chunk_bytes_buf0,
    int64_t denoms_stride_chunk_bytes_buf1,
    int64_t denoms_stride_slot_bytes_buf0,
    int64_t denoms_stride_slot_bytes_buf1,
    const torch::Tensor& buf_id_by_layer_i32,
    const torch::Tensor& slot_in_chunk_by_layer_i32,
    torch::Tensor& req_meta_i32,
    torch::Tensor& req_meta_i64,
    int64_t meta_i32_stride0, int64_t meta_i32_stride1,
    int64_t meta_i64_stride0, int64_t meta_i64_stride1,
    int64_t layers,
    int64_t batch,
    int64_t recent_cap,
    int64_t sink_tokens,
    int64_t sink_shift) {
    check_seq_dtype(seqused_k, "seqused_k");
    check_contig_last(seqused_k, "seqused_k");
    check_seq_dtype(cu_seqlens_q, "cu_seqlens_q");
    check_contig_last(cu_seqlens_q, "cu_seqlens_q");
    check_i32_contig1d(log_f_last_n_i32, "log_f_last_n_i32");
    check_i32_contig1d(log_f_capacity_i32, "log_f_capacity_i32");
    check_i32_contig1d(capture_row_by_batch_row_buf0_i32, "capture_row_by_batch_row_buf0_i32");
    check_i32_contig1d(capture_row_by_batch_row_buf1_i32, "capture_row_by_batch_row_buf1_i32");
    check_i32_contig1d(buf_id_by_layer_i32, "buf_id_by_layer_i32");
    check_i32_contig1d(slot_in_chunk_by_layer_i32, "slot_in_chunk_by_layer_i32");
    TORCH_CHECK(req_meta_i32.is_cuda() && req_meta_i64.is_cuda(),
                "req_meta tensors must be CUDA");

    const int64_t total = layers * batch;
    const int threads = 256;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    auto stream = at::cuda::getCurrentCUDAStream();

#define SFI_LAUNCH_PREFILL_LAYERS(TSEQ, TCU)                                             \
    pack_prefill_fast_layers_kernel<TSEQ, TCU><<<blocks, threads, 0, stream>>>(          \
        seqused_k.data_ptr<TSEQ>(),                                                      \
        cu_seqlens_q.data_ptr<TCU>(),                                                    \
        log_f_last_n_i32.data_ptr<int32_t>(),                                            \
        log_f_capacity_i32.data_ptr<int32_t>(),                                          \
        static_cast<int32_t>(log_f_stride_head),                                         \
        capture_row_by_batch_row_buf0_i32.data_ptr<int32_t>(),                           \
        capture_row_by_batch_row_buf1_i32.data_ptr<int32_t>(),                           \
        scores_base_ptr_buf0, scores_base_ptr_buf1,                                      \
        scores_stride_chunk_bytes_buf0, scores_stride_chunk_bytes_buf1,                  \
        scores_stride_slot_bytes_buf0, scores_stride_slot_bytes_buf1,                    \
        denoms_base_ptr_buf0, denoms_base_ptr_buf1,                                      \
        denoms_stride_chunk_bytes_buf0, denoms_stride_chunk_bytes_buf1,                  \
        denoms_stride_slot_bytes_buf0, denoms_stride_slot_bytes_buf1,                    \
        buf_id_by_layer_i32.data_ptr<int32_t>(),                                         \
        slot_in_chunk_by_layer_i32.data_ptr<int32_t>(),                                  \
        req_meta_i32.data_ptr<int32_t>(),                                                \
        req_meta_i64.data_ptr<int64_t>(),                                                \
        meta_i32_stride0, meta_i32_stride1,                                              \
        meta_i64_stride0, meta_i64_stride1,                                              \
        static_cast<int32_t>(layers), static_cast<int32_t>(batch),                       \
        static_cast<int32_t>(recent_cap),                                                \
        static_cast<int32_t>(sink_tokens), static_cast<int32_t>(sink_shift))

    const bool seq64 = seqused_k.scalar_type() == torch::kInt64;
    const bool cu64 = cu_seqlens_q.scalar_type() == torch::kInt64;
    if (seq64 && cu64) {
        SFI_LAUNCH_PREFILL_LAYERS(int64_t, int64_t);
    } else if (seq64) {
        SFI_LAUNCH_PREFILL_LAYERS(int64_t, int32_t);
    } else if (cu64) {
        SFI_LAUNCH_PREFILL_LAYERS(int32_t, int64_t);
    } else {
        SFI_LAUNCH_PREFILL_LAYERS(int32_t, int32_t);
    }
#undef SFI_LAUNCH_PREFILL_LAYERS
    const cudaError_t launch_err = cudaGetLastError();
    TORCH_CHECK(launch_err == cudaSuccess,
                "pack_req_meta_prefill_fast_layers launch failed: ",
                cudaGetErrorString(launch_err));
}
"""


def _load_ext():
    global _MODULE, _LOAD_ERROR
    if _MODULE is not None:
        return _MODULE
    if _LOAD_ERROR is not None:
        return None
    prebuilt = load_prebuilt_extension("req_meta_pack_ext")
    if prebuilt is not None:
        _MODULE = prebuilt
        return _MODULE

    _ensure_torch_cuda_arch_list()
    try:
        # [EXT-NVCC-GUARD] JIT 回落前 pin nvcc+版本预检(系统 nvcc 10.1 坑根修)。
        configure_jit_toolchain_or_raise(ext_name="req_meta_pack_ext")
        _MODULE = load_inline(
            name="req_meta_pack_ext",
            cpp_sources=_CPP_SOURCE,
            cuda_sources=_CUDA_SOURCE,
            functions=None,
            extra_cuda_cflags=["-lineinfo"],
            with_cuda=True,
            verbose=False,
        )
    except Exception as exc:
        _LOAD_ERROR = exc
        return None
    return _MODULE


def require_ext():
    """Load the extension or raise (no fallback: pack 是 GPU-only 生产主链)."""
    mod = _load_ext()
    if mod is None:
        err_msg = (
            f"req_meta_pack_ext unavailable: {_LOAD_ERROR}"
            if _LOAD_ERROR
            else "req_meta_pack_ext unavailable"
        )
        raise RuntimeError(err_msg)
    return mod


__all__ = ["require_ext"]
