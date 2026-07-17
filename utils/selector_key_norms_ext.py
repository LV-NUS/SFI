"""CUDA extension for selector key-norm preparation."""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
from torch.utils.cpp_extension import load_inline
from utils.ext_toolchain import configure_jit_toolchain_or_raise
from utils.torch_extension_cache import load_prebuilt_extension

_MODULE: Optional[torch.nn.Module] = None
_MODULE_FORCED: bool = False
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


def _dtype_code(dtype: torch.dtype) -> int:
    if dtype == torch.float16:
        return 0
    if dtype == torch.bfloat16:
        return 1
    if dtype == torch.float32:
        return 2
    raise ValueError(f"unsupported key dtype {dtype}")


def _load_ext(*, force: bool = False) -> Optional[torch.nn.Module]:
    global _MODULE, _MODULE_FORCED, _LOAD_ERROR
    if _MODULE is not None and (not force or _MODULE_FORCED):
        return _MODULE
    if force:
        _LOAD_ERROR = None
    elif _LOAD_ERROR is not None:
        return None
    if not force:
        prebuilt = load_prebuilt_extension("selector_key_norms_ext")
        if prebuilt is not None:
            _MODULE = prebuilt
            _MODULE_FORCED = False
            return _MODULE

    cpp_source = r"""
#include <torch/extension.h>

void selector_key_norms_paged_layers_cuda(
    torch::Tensor key_ptrs,
    torch::Tensor block_table,
    torch::Tensor kv_lens,
    torch::Tensor row_indices,
    torch::Tensor out_norms,
    int64_t kv_dtype_code,
    bool has_row_indices,
    int64_t head_dim,
    int64_t block_size,
    int64_t stride_k0,
    int64_t stride_k1,
    int64_t stride_k2,
    int64_t stride_k3,
    int64_t stride_bt0,
    int64_t stride_bt1,
    int64_t stride_kvlen,
    int64_t stride_row,
    int64_t stride_out_l,
    int64_t stride_out_b,
    int64_t stride_out_h,
    int64_t stride_out_t);

void selector_key_norms_paged_layers_delta_cuda(
    torch::Tensor key_ptrs,
    torch::Tensor out_ptrs,
    torch::Tensor scratch_norms,
    torch::Tensor block_table,
    torch::Tensor start_lens,
    torch::Tensor end_lens,
    torch::Tensor row_indices,
    torch::Tensor slot_indices,
    int64_t kv_dtype_code,
    int64_t out_dtype_code,
    bool has_row_indices,
    bool write_scratch,
    int64_t head_dim,
    int64_t block_size,
    int64_t max_delta,
    int64_t stride_k0,
    int64_t stride_k1,
    int64_t stride_k2,
    int64_t stride_k3,
    int64_t stride_bt0,
    int64_t stride_bt1,
    int64_t stride_start_l,
    int64_t stride_start_b,
    int64_t stride_end_l,
    int64_t stride_end_b,
    int64_t stride_row,
    int64_t stride_slot,
    int64_t stride_out_slot,
    int64_t stride_out_h,
    int64_t stride_out_t,
    int64_t stride_scratch_l,
    int64_t stride_scratch_b,
    int64_t stride_scratch_h,
    int64_t stride_scratch_t);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "selector_key_norms_paged_layers",
        &selector_key_norms_paged_layers_cuda,
        "Compute selector key norms across layers from paged KV cache (CUDA)");
    m.def(
        "selector_key_norms_paged_layers_delta",
        &selector_key_norms_paged_layers_delta_cuda,
        "Compute selector key-norm deltas across layers into per-layer arenas (CUDA)");
}
"""

    cuda_source = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

template <typename scalar_t>
__device__ __forceinline__ float scalar_to_float(scalar_t value) {
    return static_cast<float>(value);
}

template <typename key_t, typename out_t, typename block_t, int TILE_T>
__global__ void selector_key_norms_paged_layers_warp_kernel(
    const int64_t* __restrict__ key_ptrs,
    const block_t* __restrict__ block_table,
    const int32_t* __restrict__ kv_lens,
    const int32_t* __restrict__ row_indices,
    out_t* __restrict__ out_norms,
    bool has_row_indices,
    int layers,
    int batch,
    int num_heads,
    int out_len,
    int block_table_rows,
    int block_table_cols,
    int head_dim,
    int block_size,
    int64_t stride_k0,
    int64_t stride_k1,
    int64_t stride_k2,
    int64_t stride_k3,
    int64_t stride_bt0,
    int64_t stride_bt1,
    int64_t stride_kvlen,
    int64_t stride_row,
    int64_t stride_out_l,
    int64_t stride_out_b,
    int64_t stride_out_h,
    int64_t stride_out_t) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int token = blockIdx.z * TILE_T + warp;
    if (warp >= TILE_T || token >= out_len) {
        return;
    }

    int pid_lb = blockIdx.x;
    int head = blockIdx.y;
    int layer = pid_lb / batch;
    int b = pid_lb - layer * batch;

    unsigned mask = 0xffffffffu;
    int row = 0;
    int kv_len = 0;
    int block_id = -1;
    if (lane == 0) {
        row = b;
        if (has_row_indices) {
            row = static_cast<int>(row_indices[b * stride_row]);
        }
        bool row_ok = row >= 0 && row < block_table_rows;
        kv_len = static_cast<int>(kv_lens[b * stride_kvlen]);
        int block_index = token / block_size;
        bool valid_block = row_ok && block_index < block_table_cols;
        if (valid_block) {
            block_id = static_cast<int>(block_table[row * stride_bt0 + block_index * stride_bt1]);
        }
    }
    row = __shfl_sync(mask, row, 0);
    kv_len = __shfl_sync(mask, kv_len, 0);
    block_id = __shfl_sync(mask, block_id, 0);

    bool valid = row >= 0 && row < block_table_rows && token < kv_len && block_id >= 0;
    float acc = 0.0f;
    if (valid) {
        int offset = token - (token / block_size) * block_size;
        const key_t* key = reinterpret_cast<const key_t*>(key_ptrs[layer]);
        int64_t base =
            static_cast<int64_t>(block_id) * stride_k0 +
            static_cast<int64_t>(offset) * stride_k1 +
            static_cast<int64_t>(head) * stride_k2;
        for (int d = lane; d < head_dim; d += 32) {
            float v = scalar_to_float<key_t>(key[base + static_cast<int64_t>(d) * stride_k3]);
            acc += v * v;
        }
    }

    for (int delta = 16; delta > 0; delta >>= 1) {
        acc += __shfl_down_sync(mask, acc, delta);
    }
    if (lane == 0) {
        float norm = valid ? sqrtf(acc) : 0.0f;
        int64_t out_offset =
            static_cast<int64_t>(layer) * stride_out_l +
            static_cast<int64_t>(b) * stride_out_b +
            static_cast<int64_t>(head) * stride_out_h +
            static_cast<int64_t>(token) * stride_out_t;
        out_norms[out_offset] = static_cast<out_t>(norm);
    }
}

template <typename key_t, typename out_t, typename block_t>
void launch_selector_key_norms(
    torch::Tensor key_ptrs,
    torch::Tensor block_table,
    torch::Tensor kv_lens,
    torch::Tensor row_indices,
    torch::Tensor out_norms,
    bool has_row_indices,
    int64_t head_dim,
    int64_t block_size,
    int64_t stride_k0,
    int64_t stride_k1,
    int64_t stride_k2,
    int64_t stride_k3,
    int64_t stride_bt0,
    int64_t stride_bt1,
    int64_t stride_kvlen,
    int64_t stride_row,
    int64_t stride_out_l,
    int64_t stride_out_b,
    int64_t stride_out_h,
    int64_t stride_out_t) {
    int layers = static_cast<int>(out_norms.size(0));
    int batch = static_cast<int>(out_norms.size(1));
    int num_heads = static_cast<int>(out_norms.size(2));
    int out_len = static_cast<int>(out_norms.size(3));
    if (layers <= 0 || batch <= 0 || num_heads <= 0 || out_len <= 0) {
        return;
    }

    constexpr int tile_t = 32;
    const int threads = 32 * tile_t;
    dim3 blocks(
        static_cast<unsigned int>(layers * batch),
        static_cast<unsigned int>(num_heads),
        static_cast<unsigned int>((out_len + tile_t - 1) / tile_t));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    selector_key_norms_paged_layers_warp_kernel<key_t, out_t, block_t, tile_t>
        <<<blocks, threads, 0, stream>>>(
            key_ptrs.data_ptr<int64_t>(),
            block_table.data_ptr<block_t>(),
            kv_lens.data_ptr<int32_t>(),
            has_row_indices ? row_indices.data_ptr<int32_t>() : nullptr,
            out_norms.data_ptr<out_t>(),
            has_row_indices,
            layers,
            batch,
            num_heads,
            out_len,
            static_cast<int>(block_table.size(0)),
            static_cast<int>(block_table.size(1)),
            static_cast<int>(head_dim),
            static_cast<int>(block_size),
            stride_k0,
            stride_k1,
            stride_k2,
            stride_k3,
            stride_bt0,
            stride_bt1,
            stride_kvlen,
            stride_row,
            stride_out_l,
            stride_out_b,
            stride_out_h,
            stride_out_t);
}

template <typename key_t, typename out_t, typename block_t, bool BROADCAST_LENS, int TILE_T>
__global__ void selector_key_norms_paged_layers_delta_warp_kernel(
    const int64_t* __restrict__ key_ptrs,
    const int64_t* __restrict__ out_ptrs,
    out_t* __restrict__ scratch_norms,
    const block_t* __restrict__ block_table,
    const int32_t* __restrict__ start_lens,
    const int32_t* __restrict__ end_lens,
    const int32_t* __restrict__ row_indices,
    const int32_t* __restrict__ slot_indices,
    bool has_row_indices,
    bool write_scratch,
    int layers,
    int batch,
    int num_heads,
    int max_delta,
    int block_table_rows,
    int block_table_cols,
    int head_dim,
    int block_size,
    int64_t stride_k0,
    int64_t stride_k1,
    int64_t stride_k2,
    int64_t stride_k3,
    int64_t stride_bt0,
    int64_t stride_bt1,
    int64_t stride_start_l,
    int64_t stride_start_b,
    int64_t stride_end_l,
    int64_t stride_end_b,
    int64_t stride_row,
    int64_t stride_slot,
    int64_t stride_out_slot,
    int64_t stride_out_h,
    int64_t stride_out_t,
    int64_t stride_scratch_l,
    int64_t stride_scratch_b,
    int64_t stride_scratch_h,
    int64_t stride_scratch_t) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int delta_offset = blockIdx.z * TILE_T + warp;
    if (warp >= TILE_T || delta_offset >= max_delta) {
        return;
    }

    int pid_lb = blockIdx.x;
    int head = blockIdx.y;
    int layer = pid_lb / batch;
    int b = pid_lb - layer * batch;

    unsigned mask = 0xffffffffu;
    int row = 0;
    int slot = 0;
    int start = 0;
    int end = 0;
    int token = 0;
    int block_id = -1;
    if (lane == 0) {
        row = b;
        if (has_row_indices) {
            row = static_cast<int>(row_indices[b * stride_row]);
        }
        slot = static_cast<int>(slot_indices[b * stride_slot]);
        int lens_layer = BROADCAST_LENS ? 0 : layer;
        start = static_cast<int>(
            start_lens[lens_layer * stride_start_l + b * stride_start_b]);
        end = static_cast<int>(
            end_lens[lens_layer * stride_end_l + b * stride_end_b]);
        token = start + delta_offset;
        bool row_ok = row >= 0 && row < block_table_rows;
        int block_index = token / block_size;
        bool valid_block =
            row_ok && token >= 0 && token < end && block_index < block_table_cols;
        if (valid_block) {
            block_id = static_cast<int>(
                block_table[row * stride_bt0 + block_index * stride_bt1]);
        }
    }
    row = __shfl_sync(mask, row, 0);
    slot = __shfl_sync(mask, slot, 0);
    start = __shfl_sync(mask, start, 0);
    end = __shfl_sync(mask, end, 0);
    token = __shfl_sync(mask, token, 0);
    block_id = __shfl_sync(mask, block_id, 0);

    bool valid =
        slot >= 0 && row >= 0 && row < block_table_rows &&
        token >= start && token < end && block_id >= 0;
    float acc = 0.0f;
    if (valid) {
        int offset = token - (token / block_size) * block_size;
        const key_t* key = reinterpret_cast<const key_t*>(key_ptrs[layer]);
        int64_t base =
            static_cast<int64_t>(block_id) * stride_k0 +
            static_cast<int64_t>(offset) * stride_k1 +
            static_cast<int64_t>(head) * stride_k2;
        for (int d = lane; d < head_dim; d += 32) {
            float v = scalar_to_float<key_t>(
                key[base + static_cast<int64_t>(d) * stride_k3]);
            acc += v * v;
        }
    }

    for (int delta = 16; delta > 0; delta >>= 1) {
        acc += __shfl_down_sync(mask, acc, delta);
    }
    if (lane == 0 && valid) {
        out_t* out = reinterpret_cast<out_t*>(out_ptrs[layer]);
        float norm = sqrtf(acc);
        int64_t out_offset =
            static_cast<int64_t>(slot) * stride_out_slot +
            static_cast<int64_t>(head) * stride_out_h +
            static_cast<int64_t>(token) * stride_out_t;
        out[out_offset] = static_cast<out_t>(norm);
        if (write_scratch) {
            int64_t scratch_offset =
                static_cast<int64_t>(layer) * stride_scratch_l +
                static_cast<int64_t>(b) * stride_scratch_b +
                static_cast<int64_t>(head) * stride_scratch_h +
                static_cast<int64_t>(token) * stride_scratch_t;
            scratch_norms[scratch_offset] = static_cast<out_t>(norm);
        }
    }
}

template <typename key_t, typename out_t, typename block_t, bool BROADCAST_LENS>
void launch_selector_key_norms_delta(
    torch::Tensor key_ptrs,
    torch::Tensor out_ptrs,
    torch::Tensor scratch_norms,
    torch::Tensor block_table,
    torch::Tensor start_lens,
    torch::Tensor end_lens,
    torch::Tensor row_indices,
    torch::Tensor slot_indices,
    bool has_row_indices,
    bool write_scratch,
    int64_t head_dim,
    int64_t block_size,
    int64_t max_delta,
    int64_t stride_k0,
    int64_t stride_k1,
    int64_t stride_k2,
    int64_t stride_k3,
    int64_t stride_bt0,
    int64_t stride_bt1,
    int64_t stride_start_l,
    int64_t stride_start_b,
    int64_t stride_end_l,
    int64_t stride_end_b,
    int64_t stride_row,
    int64_t stride_slot,
    int64_t stride_out_slot,
    int64_t stride_out_h,
    int64_t stride_out_t,
    int64_t stride_scratch_l,
    int64_t stride_scratch_b,
    int64_t stride_scratch_h,
    int64_t stride_scratch_t) {
    int layers = BROADCAST_LENS
        ? static_cast<int>(key_ptrs.numel())
        : static_cast<int>(start_lens.size(0));
    int batch = static_cast<int>(start_lens.size(1));
    int num_heads = static_cast<int>(
        (stride_out_h > 0) ? (stride_out_slot / stride_out_h) : 0);
    if (num_heads <= 0) {
        TORCH_CHECK(false, "cannot infer num_heads from out strides");
    }
    if (layers <= 0 || batch <= 0 || max_delta <= 0) {
        return;
    }
    constexpr int tile_t = 32;
    const int threads = 32 * tile_t;
    dim3 blocks(
        static_cast<unsigned int>(layers * batch),
        static_cast<unsigned int>(num_heads),
        static_cast<unsigned int>((max_delta + tile_t - 1) / tile_t));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    selector_key_norms_paged_layers_delta_warp_kernel<key_t, out_t, block_t, BROADCAST_LENS, tile_t>
        <<<blocks, threads, 0, stream>>>(
            key_ptrs.data_ptr<int64_t>(),
            out_ptrs.data_ptr<int64_t>(),
            write_scratch ? scratch_norms.data_ptr<out_t>() : nullptr,
            block_table.data_ptr<block_t>(),
            start_lens.data_ptr<int32_t>(),
            end_lens.data_ptr<int32_t>(),
            has_row_indices ? row_indices.data_ptr<int32_t>() : nullptr,
            slot_indices.data_ptr<int32_t>(),
            has_row_indices,
            write_scratch,
            layers,
            batch,
            num_heads,
            static_cast<int>(max_delta),
            static_cast<int>(block_table.size(0)),
            static_cast<int>(block_table.size(1)),
            static_cast<int>(head_dim),
            static_cast<int>(block_size),
            stride_k0,
            stride_k1,
            stride_k2,
            stride_k3,
            stride_bt0,
            stride_bt1,
            stride_start_l,
            stride_start_b,
            stride_end_l,
            stride_end_b,
            stride_row,
            stride_slot,
            stride_out_slot,
            stride_out_h,
            stride_out_t,
            stride_scratch_l,
            stride_scratch_b,
            stride_scratch_h,
            stride_scratch_t);
}

template <typename key_t, typename out_t, bool BROADCAST_LENS>
void dispatch_delta_block_dtype(
    torch::Tensor key_ptrs,
    torch::Tensor out_ptrs,
    torch::Tensor scratch_norms,
    torch::Tensor block_table,
    torch::Tensor start_lens,
    torch::Tensor end_lens,
    torch::Tensor row_indices,
    torch::Tensor slot_indices,
    bool has_row_indices,
    bool write_scratch,
    int64_t head_dim,
    int64_t block_size,
    int64_t max_delta,
    int64_t stride_k0,
    int64_t stride_k1,
    int64_t stride_k2,
    int64_t stride_k3,
    int64_t stride_bt0,
    int64_t stride_bt1,
    int64_t stride_start_l,
    int64_t stride_start_b,
    int64_t stride_end_l,
    int64_t stride_end_b,
    int64_t stride_row,
    int64_t stride_slot,
    int64_t stride_out_slot,
    int64_t stride_out_h,
    int64_t stride_out_t,
    int64_t stride_scratch_l,
    int64_t stride_scratch_b,
    int64_t stride_scratch_h,
    int64_t stride_scratch_t) {
    if (block_table.scalar_type() == torch::kInt64) {
        launch_selector_key_norms_delta<key_t, out_t, int64_t, BROADCAST_LENS>(
            key_ptrs, out_ptrs, scratch_norms, block_table, start_lens, end_lens,
            row_indices, slot_indices, has_row_indices, write_scratch, head_dim,
            block_size, max_delta,
            stride_k0, stride_k1, stride_k2, stride_k3, stride_bt0, stride_bt1,
            stride_start_l, stride_start_b, stride_end_l, stride_end_b,
            stride_row, stride_slot, stride_out_slot, stride_out_h, stride_out_t,
            stride_scratch_l, stride_scratch_b, stride_scratch_h, stride_scratch_t);
    } else if (block_table.scalar_type() == torch::kInt32) {
        launch_selector_key_norms_delta<key_t, out_t, int32_t, BROADCAST_LENS>(
            key_ptrs, out_ptrs, scratch_norms, block_table, start_lens, end_lens,
            row_indices, slot_indices, has_row_indices, write_scratch, head_dim,
            block_size, max_delta,
            stride_k0, stride_k1, stride_k2, stride_k3, stride_bt0, stride_bt1,
            stride_start_l, stride_start_b, stride_end_l, stride_end_b,
            stride_row, stride_slot, stride_out_slot, stride_out_h, stride_out_t,
            stride_scratch_l, stride_scratch_b, stride_scratch_h, stride_scratch_t);
    } else {
        TORCH_CHECK(false, "block_table must be int32 or int64");
    }
}

template <typename key_t, bool BROADCAST_LENS>
void dispatch_delta_out_dtype(
    torch::Tensor key_ptrs,
    torch::Tensor out_ptrs,
    torch::Tensor scratch_norms,
    torch::Tensor block_table,
    torch::Tensor start_lens,
    torch::Tensor end_lens,
    torch::Tensor row_indices,
    torch::Tensor slot_indices,
    int64_t out_dtype_code,
    bool has_row_indices,
    bool write_scratch,
    int64_t head_dim,
    int64_t block_size,
    int64_t max_delta,
    int64_t stride_k0,
    int64_t stride_k1,
    int64_t stride_k2,
    int64_t stride_k3,
    int64_t stride_bt0,
    int64_t stride_bt1,
    int64_t stride_start_l,
    int64_t stride_start_b,
    int64_t stride_end_l,
    int64_t stride_end_b,
    int64_t stride_row,
    int64_t stride_slot,
    int64_t stride_out_slot,
    int64_t stride_out_h,
    int64_t stride_out_t,
    int64_t stride_scratch_l,
    int64_t stride_scratch_b,
    int64_t stride_scratch_h,
    int64_t stride_scratch_t) {
    if (out_dtype_code == 0) {
        dispatch_delta_block_dtype<key_t, c10::Half, BROADCAST_LENS>(
            key_ptrs, out_ptrs, scratch_norms, block_table, start_lens, end_lens,
            row_indices, slot_indices, has_row_indices, write_scratch, head_dim,
            block_size, max_delta,
            stride_k0, stride_k1, stride_k2, stride_k3, stride_bt0, stride_bt1,
            stride_start_l, stride_start_b, stride_end_l, stride_end_b,
            stride_row, stride_slot, stride_out_slot, stride_out_h, stride_out_t,
            stride_scratch_l, stride_scratch_b, stride_scratch_h, stride_scratch_t);
    } else if (out_dtype_code == 1) {
        dispatch_delta_block_dtype<key_t, c10::BFloat16, BROADCAST_LENS>(
            key_ptrs, out_ptrs, scratch_norms, block_table, start_lens, end_lens,
            row_indices, slot_indices, has_row_indices, write_scratch, head_dim,
            block_size, max_delta,
            stride_k0, stride_k1, stride_k2, stride_k3, stride_bt0, stride_bt1,
            stride_start_l, stride_start_b, stride_end_l, stride_end_b,
            stride_row, stride_slot, stride_out_slot, stride_out_h, stride_out_t,
            stride_scratch_l, stride_scratch_b, stride_scratch_h, stride_scratch_t);
    } else if (out_dtype_code == 2) {
        dispatch_delta_block_dtype<key_t, float, BROADCAST_LENS>(
            key_ptrs, out_ptrs, scratch_norms, block_table, start_lens, end_lens,
            row_indices, slot_indices, has_row_indices, write_scratch, head_dim,
            block_size, max_delta,
            stride_k0, stride_k1, stride_k2, stride_k3, stride_bt0, stride_bt1,
            stride_start_l, stride_start_b, stride_end_l, stride_end_b,
            stride_row, stride_slot, stride_out_slot, stride_out_h, stride_out_t,
            stride_scratch_l, stride_scratch_b, stride_scratch_h, stride_scratch_t);
    } else {
        TORCH_CHECK(false, "unsupported out dtype code");
    }
}

template <typename key_t, typename out_t>
void dispatch_block_dtype(
    torch::Tensor key_ptrs,
    torch::Tensor block_table,
    torch::Tensor kv_lens,
    torch::Tensor row_indices,
    torch::Tensor out_norms,
    bool has_row_indices,
    int64_t head_dim,
    int64_t block_size,
    int64_t stride_k0,
    int64_t stride_k1,
    int64_t stride_k2,
    int64_t stride_k3,
    int64_t stride_bt0,
    int64_t stride_bt1,
    int64_t stride_kvlen,
    int64_t stride_row,
    int64_t stride_out_l,
    int64_t stride_out_b,
    int64_t stride_out_h,
    int64_t stride_out_t) {
    if (block_table.scalar_type() == torch::kInt64) {
        launch_selector_key_norms<key_t, out_t, int64_t>(
            key_ptrs, block_table, kv_lens, row_indices, out_norms, has_row_indices,
            head_dim, block_size, stride_k0, stride_k1, stride_k2, stride_k3,
            stride_bt0, stride_bt1, stride_kvlen, stride_row,
            stride_out_l, stride_out_b, stride_out_h, stride_out_t);
    } else if (block_table.scalar_type() == torch::kInt32) {
        launch_selector_key_norms<key_t, out_t, int32_t>(
            key_ptrs, block_table, kv_lens, row_indices, out_norms, has_row_indices,
            head_dim, block_size, stride_k0, stride_k1, stride_k2, stride_k3,
            stride_bt0, stride_bt1, stride_kvlen, stride_row,
            stride_out_l, stride_out_b, stride_out_h, stride_out_t);
    } else {
        TORCH_CHECK(false, "block_table must be int32 or int64");
    }
}

template <typename key_t>
void dispatch_out_dtype(
    torch::Tensor key_ptrs,
    torch::Tensor block_table,
    torch::Tensor kv_lens,
    torch::Tensor row_indices,
    torch::Tensor out_norms,
    bool has_row_indices,
    int64_t head_dim,
    int64_t block_size,
    int64_t stride_k0,
    int64_t stride_k1,
    int64_t stride_k2,
    int64_t stride_k3,
    int64_t stride_bt0,
    int64_t stride_bt1,
    int64_t stride_kvlen,
    int64_t stride_row,
    int64_t stride_out_l,
    int64_t stride_out_b,
    int64_t stride_out_h,
    int64_t stride_out_t) {
    if (out_norms.scalar_type() == torch::kFloat32) {
        dispatch_block_dtype<key_t, float>(
            key_ptrs, block_table, kv_lens, row_indices, out_norms, has_row_indices,
            head_dim, block_size, stride_k0, stride_k1, stride_k2, stride_k3,
            stride_bt0, stride_bt1, stride_kvlen, stride_row,
            stride_out_l, stride_out_b, stride_out_h, stride_out_t);
    } else if (out_norms.scalar_type() == torch::kFloat16) {
        dispatch_block_dtype<key_t, c10::Half>(
            key_ptrs, block_table, kv_lens, row_indices, out_norms, has_row_indices,
            head_dim, block_size, stride_k0, stride_k1, stride_k2, stride_k3,
            stride_bt0, stride_bt1, stride_kvlen, stride_row,
            stride_out_l, stride_out_b, stride_out_h, stride_out_t);
    } else if (out_norms.scalar_type() == torch::kBFloat16) {
        dispatch_block_dtype<key_t, c10::BFloat16>(
            key_ptrs, block_table, kv_lens, row_indices, out_norms, has_row_indices,
            head_dim, block_size, stride_k0, stride_k1, stride_k2, stride_k3,
            stride_bt0, stride_bt1, stride_kvlen, stride_row,
            stride_out_l, stride_out_b, stride_out_h, stride_out_t);
    } else {
        TORCH_CHECK(false, "out_norms must be float16, bfloat16, or float32");
    }
}

void selector_key_norms_paged_layers_cuda(
    torch::Tensor key_ptrs,
    torch::Tensor block_table,
    torch::Tensor kv_lens,
    torch::Tensor row_indices,
    torch::Tensor out_norms,
    int64_t kv_dtype_code,
    bool has_row_indices,
    int64_t head_dim,
    int64_t block_size,
    int64_t stride_k0,
    int64_t stride_k1,
    int64_t stride_k2,
    int64_t stride_k3,
    int64_t stride_bt0,
    int64_t stride_bt1,
    int64_t stride_kvlen,
    int64_t stride_row,
    int64_t stride_out_l,
    int64_t stride_out_b,
    int64_t stride_out_h,
    int64_t stride_out_t) {
    TORCH_CHECK(key_ptrs.is_cuda(), "key_ptrs must be CUDA");
    TORCH_CHECK(block_table.is_cuda(), "block_table must be CUDA");
    TORCH_CHECK(kv_lens.is_cuda(), "kv_lens must be CUDA");
    TORCH_CHECK(out_norms.is_cuda(), "out_norms must be CUDA");
    TORCH_CHECK(key_ptrs.scalar_type() == torch::kInt64, "key_ptrs must be int64");
    TORCH_CHECK(kv_lens.scalar_type() == torch::kInt32, "kv_lens must be int32");
    TORCH_CHECK(out_norms.dim() == 4, "out_norms must be [layers, batch, heads, tokens]");
    TORCH_CHECK(block_table.dim() == 2, "block_table must be [rows, blocks]");
    TORCH_CHECK(block_size > 0, "block_size must be positive");
    TORCH_CHECK(head_dim > 0, "head_dim must be positive");
    if (has_row_indices) {
        TORCH_CHECK(row_indices.is_cuda(), "row_indices must be CUDA");
        TORCH_CHECK(row_indices.scalar_type() == torch::kInt32, "row_indices must be int32");
    }

    if (kv_dtype_code == 0) {
        dispatch_out_dtype<c10::Half>(
            key_ptrs, block_table, kv_lens, row_indices, out_norms, has_row_indices,
            head_dim, block_size, stride_k0, stride_k1, stride_k2, stride_k3,
            stride_bt0, stride_bt1, stride_kvlen, stride_row,
            stride_out_l, stride_out_b, stride_out_h, stride_out_t);
    } else if (kv_dtype_code == 1) {
        dispatch_out_dtype<c10::BFloat16>(
            key_ptrs, block_table, kv_lens, row_indices, out_norms, has_row_indices,
            head_dim, block_size, stride_k0, stride_k1, stride_k2, stride_k3,
            stride_bt0, stride_bt1, stride_kvlen, stride_row,
            stride_out_l, stride_out_b, stride_out_h, stride_out_t);
    } else if (kv_dtype_code == 2) {
        dispatch_out_dtype<float>(
            key_ptrs, block_table, kv_lens, row_indices, out_norms, has_row_indices,
            head_dim, block_size, stride_k0, stride_k1, stride_k2, stride_k3,
            stride_bt0, stride_bt1, stride_kvlen, stride_row,
            stride_out_l, stride_out_b, stride_out_h, stride_out_t);
    } else {
        TORCH_CHECK(false, "unsupported kv dtype code");
    }
}

void selector_key_norms_paged_layers_delta_cuda(
    torch::Tensor key_ptrs,
    torch::Tensor out_ptrs,
    torch::Tensor scratch_norms,
    torch::Tensor block_table,
    torch::Tensor start_lens,
    torch::Tensor end_lens,
    torch::Tensor row_indices,
    torch::Tensor slot_indices,
    int64_t kv_dtype_code,
    int64_t out_dtype_code,
    bool has_row_indices,
    bool write_scratch,
    int64_t head_dim,
    int64_t block_size,
    int64_t max_delta,
    int64_t stride_k0,
    int64_t stride_k1,
    int64_t stride_k2,
    int64_t stride_k3,
    int64_t stride_bt0,
    int64_t stride_bt1,
    int64_t stride_start_l,
    int64_t stride_start_b,
    int64_t stride_end_l,
    int64_t stride_end_b,
    int64_t stride_row,
    int64_t stride_slot,
    int64_t stride_out_slot,
    int64_t stride_out_h,
    int64_t stride_out_t,
    int64_t stride_scratch_l,
    int64_t stride_scratch_b,
    int64_t stride_scratch_h,
    int64_t stride_scratch_t) {
    TORCH_CHECK(key_ptrs.is_cuda(), "key_ptrs must be CUDA");
    TORCH_CHECK(out_ptrs.is_cuda(), "out_ptrs must be CUDA");
    TORCH_CHECK(block_table.is_cuda(), "block_table must be CUDA");
    if (write_scratch) {
        TORCH_CHECK(scratch_norms.is_cuda(), "scratch_norms must be CUDA");
    }
    TORCH_CHECK(start_lens.is_cuda(), "start_lens must be CUDA");
    TORCH_CHECK(end_lens.is_cuda(), "end_lens must be CUDA");
    TORCH_CHECK(slot_indices.is_cuda(), "slot_indices must be CUDA");
    TORCH_CHECK(key_ptrs.scalar_type() == torch::kInt64, "key_ptrs must be int64");
    TORCH_CHECK(out_ptrs.scalar_type() == torch::kInt64, "out_ptrs must be int64");
    TORCH_CHECK(start_lens.scalar_type() == torch::kInt32, "start_lens must be int32");
    TORCH_CHECK(end_lens.scalar_type() == torch::kInt32, "end_lens must be int32");
    TORCH_CHECK(slot_indices.scalar_type() == torch::kInt32, "slot_indices must be int32");
    TORCH_CHECK(start_lens.dim() == 2, "start_lens must be [layers, batch]");
    TORCH_CHECK(end_lens.dim() == 2, "end_lens must be [layers, batch]");
    TORCH_CHECK(start_lens.size(0) == end_lens.size(0), "start/end layer mismatch");
    TORCH_CHECK(start_lens.size(1) == end_lens.size(1), "start/end batch mismatch");
    TORCH_CHECK(key_ptrs.numel() >= start_lens.size(0), "key_ptrs too small");
    TORCH_CHECK(out_ptrs.numel() >= start_lens.size(0), "out_ptrs too small");
    TORCH_CHECK(slot_indices.numel() >= start_lens.size(1), "slot_indices too small");
    TORCH_CHECK(block_table.dim() == 2, "block_table must be [rows, blocks]");
    if (write_scratch) {
        TORCH_CHECK(scratch_norms.dim() == 4, "scratch_norms must be [layers, batch, heads, tokens]");
    }
    TORCH_CHECK(block_size > 0, "block_size must be positive");
    TORCH_CHECK(head_dim > 0, "head_dim must be positive");
    if (has_row_indices) {
        TORCH_CHECK(row_indices.is_cuda(), "row_indices must be CUDA");
        TORCH_CHECK(row_indices.scalar_type() == torch::kInt32, "row_indices must be int32");
        TORCH_CHECK(row_indices.numel() >= start_lens.size(1), "row_indices too small");
    }

    if (kv_dtype_code == 0) {
        dispatch_delta_out_dtype<c10::Half, false>(
            key_ptrs, out_ptrs, scratch_norms, block_table, start_lens, end_lens, row_indices,
            slot_indices, out_dtype_code, has_row_indices, write_scratch, head_dim, block_size,
            max_delta, stride_k0, stride_k1, stride_k2, stride_k3, stride_bt0,
            stride_bt1, stride_start_l, stride_start_b, stride_end_l,
            stride_end_b, stride_row, stride_slot, stride_out_slot,
            stride_out_h, stride_out_t, stride_scratch_l, stride_scratch_b,
            stride_scratch_h, stride_scratch_t);
    } else if (kv_dtype_code == 1) {
        dispatch_delta_out_dtype<c10::BFloat16, false>(
            key_ptrs, out_ptrs, scratch_norms, block_table, start_lens, end_lens, row_indices,
            slot_indices, out_dtype_code, has_row_indices, write_scratch, head_dim, block_size,
            max_delta, stride_k0, stride_k1, stride_k2, stride_k3, stride_bt0,
            stride_bt1, stride_start_l, stride_start_b, stride_end_l,
            stride_end_b, stride_row, stride_slot, stride_out_slot,
            stride_out_h, stride_out_t, stride_scratch_l, stride_scratch_b,
            stride_scratch_h, stride_scratch_t);
    } else if (kv_dtype_code == 2) {
        dispatch_delta_out_dtype<float, false>(
            key_ptrs, out_ptrs, scratch_norms, block_table, start_lens, end_lens, row_indices,
            slot_indices, out_dtype_code, has_row_indices, write_scratch, head_dim, block_size,
            max_delta, stride_k0, stride_k1, stride_k2, stride_k3, stride_bt0,
            stride_bt1, stride_start_l, stride_start_b, stride_end_l,
            stride_end_b, stride_row, stride_slot, stride_out_slot,
            stride_out_h, stride_out_t, stride_scratch_l, stride_scratch_b,
            stride_scratch_h, stride_scratch_t);
    } else {
        TORCH_CHECK(false, "unsupported kv dtype code");
    }
}

"""

    _ensure_torch_cuda_arch_list()
    os.environ.setdefault("TORCH_EXTENSION_SKIP_NVCC_GEN_DEPENDENCIES", "1")
    try:
        configure_jit_toolchain_or_raise(ext_name="selector_key_norms_ext")
        _MODULE = load_inline(
            name="selector_key_norms_ext",
            cpp_sources=cpp_source,
            cuda_sources=cuda_source,
            functions=None,
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "-lineinfo", "-std=c++17"],
            with_cuda=True,
            verbose=False,
        )
        _MODULE_FORCED = bool(force)
    except Exception as exc:
        _LOAD_ERROR = exc
        return None
    return _MODULE


def _require_ext(*, force: bool = False) -> torch.nn.Module:
    mod = _load_ext(force=force)
    if mod is None:
        if _LOAD_ERROR is not None:
            raise RuntimeError(
                f"selector_key_norms_ext unavailable: {_LOAD_ERROR!r}"
            ) from _LOAD_ERROR
        raise RuntimeError("selector_key_norms_ext unavailable")
    return mod


def compute_key_norms_paged_batched_layers_cuda(
    *,
    key_ptrs: torch.Tensor,
    block_table: torch.Tensor,
    kv_lens: torch.Tensor,
    out_norms: torch.Tensor,
    kv_dtype: torch.dtype,
    head_dim: int,
    block_size: int,
    key_strides: Tuple[int, int, int, int],
    row_indices: Optional[torch.Tensor] = None,
) -> None:
    """Compute [layer, batch, head, token] key norms with one CUDA launch."""
    if key_ptrs.dtype != torch.int64:
        key_ptrs = key_ptrs.to(dtype=torch.int64)
    if not key_ptrs.is_contiguous():
        key_ptrs = key_ptrs.contiguous()
    if kv_lens.dtype != torch.int32:
        kv_lens = kv_lens.to(dtype=torch.int32)
    if not kv_lens.is_contiguous():
        kv_lens = kv_lens.contiguous()

    has_rows = row_indices is not None
    if row_indices is None:
        row_indices = torch.empty((0,), device=block_table.device, dtype=torch.int32)
    else:
        if row_indices.dtype != torch.int32:
            row_indices = row_indices.to(dtype=torch.int32)
        if row_indices.device != block_table.device:
            row_indices = row_indices.to(device=block_table.device)
        if not row_indices.is_contiguous():
            row_indices = row_indices.contiguous()

    mod = _require_ext()
    mod.selector_key_norms_paged_layers(
        key_ptrs,
        block_table,
        kv_lens,
        row_indices,
        out_norms,
        int(_dtype_code(kv_dtype)),
        bool(has_rows),
        int(head_dim),
        int(block_size),
        int(key_strides[0]),
        int(key_strides[1]),
        int(key_strides[2]),
        int(key_strides[3]),
        int(block_table.stride(0)),
        int(block_table.stride(1)),
        int(kv_lens.stride(0)),
        int(row_indices.stride(0)) if has_rows else 0,
        int(out_norms.stride(0)),
        int(out_norms.stride(1)),
        int(out_norms.stride(2)),
        int(out_norms.stride(3)),
    )


def compute_key_norms_paged_batched_layers_delta_cuda(
    *,
    key_ptrs: torch.Tensor,
    out_ptrs: torch.Tensor,
    block_table: torch.Tensor,
    start_lens: torch.Tensor,
    end_lens: torch.Tensor,
    slot_indices: torch.Tensor,
    kv_dtype: torch.dtype,
    out_dtype: torch.dtype,
    head_dim: int,
    block_size: int,
    key_strides: Tuple[int, int, int, int],
    out_strides: Tuple[int, int, int],
    max_delta: int,
    row_indices: Optional[torch.Tensor] = None,
    scratch_norms: Optional[torch.Tensor] = None,
) -> None:
    """Compute [start,end) key-norm deltas into arena slots and optional scratch."""
    if int(max_delta) <= 0:
        return
    device = block_table.device
    if key_ptrs.dtype != torch.int64:
        key_ptrs = key_ptrs.to(dtype=torch.int64)
    if key_ptrs.device != device:
        key_ptrs = key_ptrs.to(device=device)
    if not key_ptrs.is_contiguous():
        key_ptrs = key_ptrs.contiguous()

    if out_ptrs.dtype != torch.int64:
        out_ptrs = out_ptrs.to(dtype=torch.int64)
    if out_ptrs.device != device:
        out_ptrs = out_ptrs.to(device=device)
    if not out_ptrs.is_contiguous():
        out_ptrs = out_ptrs.contiguous()

    if start_lens.dtype != torch.int32:
        start_lens = start_lens.to(dtype=torch.int32)
    if start_lens.device != device:
        start_lens = start_lens.to(device=device)
    if not start_lens.is_contiguous():
        start_lens = start_lens.contiguous()

    if end_lens.dtype != torch.int32:
        end_lens = end_lens.to(dtype=torch.int32)
    if end_lens.device != device:
        end_lens = end_lens.to(device=device)
    if not end_lens.is_contiguous():
        end_lens = end_lens.contiguous()

    if slot_indices.dtype != torch.int32:
        slot_indices = slot_indices.to(dtype=torch.int32)
    if slot_indices.device != device:
        slot_indices = slot_indices.to(device=device)
    if not slot_indices.is_contiguous():
        slot_indices = slot_indices.contiguous()

    write_scratch = scratch_norms is not None
    if scratch_norms is None:
        scratch_norms = torch.empty((0,), device=device, dtype=out_dtype)
        scratch_strides = (0, 0, 0, 0)
    else:
        if scratch_norms.dtype != out_dtype:
            raise ValueError("scratch_norms dtype must match out_dtype")
        if scratch_norms.device != device:
            scratch_norms = scratch_norms.to(device=device)
        scratch_strides = tuple(int(s) for s in scratch_norms.stride())

    has_rows = row_indices is not None
    if row_indices is None:
        row_indices = torch.empty((0,), device=device, dtype=torch.int32)
    else:
        if row_indices.dtype != torch.int32:
            row_indices = row_indices.to(dtype=torch.int32)
        if row_indices.device != device:
            row_indices = row_indices.to(device=device)
        if not row_indices.is_contiguous():
            row_indices = row_indices.contiguous()

    mod = _require_ext()
    kernel = mod.selector_key_norms_paged_layers_delta
    kernel(
        key_ptrs,
        out_ptrs,
        scratch_norms,
        block_table,
        start_lens,
        end_lens,
        row_indices,
        slot_indices,
        int(_dtype_code(kv_dtype)),
        int(_dtype_code(out_dtype)),
        bool(has_rows),
        bool(write_scratch),
        int(head_dim),
        int(block_size),
        int(max_delta),
        int(key_strides[0]),
        int(key_strides[1]),
        int(key_strides[2]),
        int(key_strides[3]),
        int(block_table.stride(0)),
        int(block_table.stride(1)),
        int(start_lens.stride(0)),
        int(start_lens.stride(1)),
        int(end_lens.stride(0)),
        int(end_lens.stride(1)),
        int(row_indices.stride(0)) if has_rows else 0,
        int(slot_indices.stride(0)),
        int(out_strides[0]),
        int(out_strides[1]),
        int(out_strides[2]),
        int(scratch_strides[0]),
        int(scratch_strides[1]),
        int(scratch_strides[2]),
        int(scratch_strides[3]),
    )

__all__ = [
    "compute_key_norms_paged_batched_layers_cuda",
    "compute_key_norms_paged_batched_layers_delta_cuda",
]
