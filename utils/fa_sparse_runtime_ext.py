from __future__ import annotations

import os
from typing import Optional

import torch
from torch.utils.cpp_extension import load_inline
from utils.torch_extension_cache import load_prebuilt_extension

from patches.fa_sparse_runtime.contracts import (
    PAGE_SPARSE_STATUS_OK,
    PAGE_SPARSE_STATUS_RECOVERABLE_MISS,
)

_MODULE: Optional[torch.nn.Module] = None
_LOAD_ERROR: Optional[Exception] = None


class FASparseRuntimeExtUnavailable(RuntimeError):
    """Raised when the FA sparse runtime CUDA extension is unavailable."""


def _should_enable() -> bool:
    return os.environ.get("VLLM_SPARSE_FA_RUNTIME_EXT_CUDA", "1") == "1"


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


def _require_rank(name: str, tensor: torch.Tensor, rank: int) -> None:
    if tensor.dim() != int(rank):
        raise ValueError(f"{name} must be rank-{rank}")


def _require_output_tensor(name: str, tensor: torch.Tensor) -> None:
    if tensor.dtype != torch.int32:
        raise ValueError(f"{name} must have dtype int32")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous for in-place CUDA materialize")


def _load_ext(*, force: bool = False) -> Optional[torch.nn.Module]:
    global _MODULE, _LOAD_ERROR
    if _MODULE is not None:
        return _MODULE
    if _LOAD_ERROR is not None:
        return None
    if not force and not _should_enable():
        return None
    if not force:
        prebuilt = load_prebuilt_extension("fa_sparse_runtime_ext")
        # ABI guard: only trust a prebuilt module that actually exposes
        # the load-bearing FA-sparse-runtime entrypoints. A stale /
        # name-collided .so (a documented hazard for this box's shared
        # TORCH_EXTENSIONS_DIR) would otherwise be cached and later raise
        # (-> silent Python fallback) or drift; on mismatch fall through
        # to load_inline so the correct source is rebuilt.
        if prebuilt is not None and all(
            hasattr(prebuilt, _name)
            for _name in (
                "refresh_static_materialize",
                "step_recent_patch",
                "gather_compact_kv_into_arena_ptrs_tiled_autolen",
                # [ABI-GUARD-PROD-SYMBOLS] 生产实际消费的两个入口也必须在:
                # 残留"有 autolen 无 skip_unchanged"的历史 .so 会被静默信任,
                # 直到 selection_worker 取符号才远端 RuntimeError。
                "gather_compact_kv_into_arena_ptrs_tiled_autolen_skip_unchanged",
                "gather_compact_kv_into_arena_ptrs_tiled",
            )
        ):
            _MODULE = prebuilt
            return _MODULE

    cpp_source = r"""
#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> refresh_static_materialize_cuda(
    torch::Tensor final_page_table,
    torch::Tensor selected_middle_pages,
    torch::Tensor selected_middle_counts,
    torch::Tensor request_slot_rows,
    torch::Tensor request_block_table,
    int64_t sink_page_slots,
    torch::Tensor request_refresh_generation,
    bool slot_major);

std::vector<torch::Tensor> step_recent_patch_cuda(
    torch::Tensor final_page_table,
    torch::Tensor request_block_table,
    torch::Tensor request_recent_first_logical_page,
    torch::Tensor request_recent_page_count,
    torch::Tensor request_recent_epoch,
    torch::Tensor selected_page_count,
    int64_t sink_page_slots,
    int64_t num_kv_heads);

void gather_compact_kv_into_arena(
    std::vector<torch::Tensor> flat_k,
    std::vector<torch::Tensor> flat_v,
    std::vector<torch::Tensor> compact_k,
    std::vector<torch::Tensor> compact_v,
    std::vector<torch::Tensor> compact_pos,
    std::vector<torch::Tensor> block_tables,
    torch::Tensor row_tensor,
    torch::Tensor slot_tensor,
    torch::Tensor selected_indices,
    torch::Tensor sink_len,
    torch::Tensor persist_len,
    int64_t page_size,
    int64_t stride_tokens);

void gather_compact_kv_into_arena_ptrs(
    torch::Tensor flat_k_ptrs,
    torch::Tensor flat_v_ptrs,
    torch::Tensor compact_k_ptrs,
    torch::Tensor compact_v_ptrs,
    torch::Tensor compact_pos_ptrs,
    torch::Tensor block_table_ptrs,
    torch::Tensor row_tensor,
    torch::Tensor slot_tensor,
    torch::Tensor selected_indices,
    torch::Tensor sink_len,
    torch::Tensor persist_len,
    int64_t page_size,
    int64_t stride_tokens,
    int64_t block_table_cols,
    int64_t head_dim,
    int64_t compact_pos_per_layer_tokens,
    int64_t flat_k_stride0,
    int64_t flat_k_stride1,
    int64_t flat_k_stride2,
    int64_t flat_v_stride0,
    int64_t flat_v_stride1,
    int64_t flat_v_stride2,
    int64_t compact_k_stride0,
    int64_t compact_k_stride1,
    int64_t compact_k_stride2,
    int64_t compact_v_stride0,
    int64_t compact_v_stride1,
    int64_t compact_v_stride2,
    int64_t compact_pos_stride_head,
    int64_t compact_pos_stride_tok,
    int64_t block_table_stride0,
    int64_t block_table_stride1);

void gather_compact_kv_into_arena_ptrs_tiled(
    torch::Tensor flat_k_ptrs,
    torch::Tensor flat_v_ptrs,
    torch::Tensor compact_k_ptrs,
    torch::Tensor compact_v_ptrs,
    torch::Tensor compact_pos_ptrs,
    torch::Tensor block_table_ptrs,
    torch::Tensor row_tensor,
    torch::Tensor slot_tensor,
    torch::Tensor selected_indices,
    torch::Tensor sink_len,
    torch::Tensor persist_len,
    int64_t page_size,
    int64_t stride_tokens,
    int64_t block_table_cols,
    int64_t head_dim,
    int64_t compact_pos_per_layer_tokens,
    int64_t flat_k_stride0,
    int64_t flat_k_stride1,
    int64_t flat_k_stride2,
    int64_t flat_v_stride0,
    int64_t flat_v_stride1,
    int64_t flat_v_stride2,
    int64_t compact_k_stride0,
    int64_t compact_k_stride1,
    int64_t compact_k_stride2,
    int64_t compact_v_stride0,
    int64_t compact_v_stride1,
    int64_t compact_v_stride2,
    int64_t compact_pos_stride_head,
    int64_t compact_pos_stride_tok,
    int64_t block_table_stride0,
    int64_t block_table_stride1,
    int64_t tile_tokens,
    int64_t max_total_tokens);

void gather_compact_kv_into_arena_ptrs_tiled_autolen(
    torch::Tensor flat_k_ptrs,
    torch::Tensor flat_v_ptrs,
    torch::Tensor compact_k_ptrs,
    torch::Tensor compact_v_ptrs,
    torch::Tensor compact_pos_ptrs,
    torch::Tensor block_table_ptrs,
    torch::Tensor row_tensor,
    torch::Tensor slot_tensor,
    torch::Tensor selected_indices,
    torch::Tensor seq_lens,
    int64_t page_size,
    int64_t stride_tokens,
    int64_t block_table_cols,
    int64_t head_dim,
    int64_t compact_pos_per_layer_tokens,
    int64_t flat_k_stride0,
    int64_t flat_k_stride1,
    int64_t flat_k_stride2,
    int64_t flat_v_stride0,
    int64_t flat_v_stride1,
    int64_t flat_v_stride2,
    int64_t compact_k_stride0,
    int64_t compact_k_stride1,
    int64_t compact_k_stride2,
    int64_t compact_v_stride0,
    int64_t compact_v_stride1,
    int64_t compact_v_stride2,
    int64_t compact_pos_stride_head,
    int64_t compact_pos_stride_tok,
    int64_t block_table_stride0,
    int64_t block_table_stride1,
    int64_t tile_tokens,
    int64_t max_total_tokens,
    int64_t sink_cap,
    int64_t recent_cfg,
    int64_t k_head,
    int64_t threshold_tokens);

void gather_compact_kv_into_arena_ptrs_tiled_autolen_skip_unchanged(
    torch::Tensor flat_k_ptrs,
    torch::Tensor flat_v_ptrs,
    torch::Tensor compact_k_ptrs,
    torch::Tensor compact_v_ptrs,
    torch::Tensor compact_pos_ptrs,
    torch::Tensor block_table_ptrs,
    torch::Tensor row_tensor,
    torch::Tensor slot_tensor,
    torch::Tensor selected_indices,
    torch::Tensor seq_lens,
    int64_t page_size,
    int64_t stride_tokens,
    int64_t block_table_cols,
    int64_t head_dim,
    int64_t compact_pos_per_layer_tokens,
    int64_t flat_k_stride0,
    int64_t flat_k_stride1,
    int64_t flat_k_stride2,
    int64_t flat_v_stride0,
    int64_t flat_v_stride1,
    int64_t flat_v_stride2,
    int64_t compact_k_stride0,
    int64_t compact_k_stride1,
    int64_t compact_k_stride2,
    int64_t compact_v_stride0,
    int64_t compact_v_stride1,
    int64_t compact_v_stride2,
    int64_t compact_pos_stride_head,
    int64_t compact_pos_stride_tok,
    int64_t block_table_stride0,
    int64_t block_table_stride1,
    int64_t tile_tokens,
    int64_t max_total_tokens,
    int64_t sink_cap,
    int64_t recent_cfg,
    int64_t k_head,
    int64_t threshold_tokens);

std::vector<torch::Tensor> refresh_static_materialize(
    torch::Tensor final_page_table,
    torch::Tensor selected_middle_pages,
    torch::Tensor selected_middle_counts,
    torch::Tensor request_slot_rows,
    torch::Tensor request_block_table,
    int64_t sink_page_slots,
    torch::Tensor request_refresh_generation,
    bool slot_major) {
    return refresh_static_materialize_cuda(
        final_page_table,
        selected_middle_pages,
        selected_middle_counts,
        request_slot_rows,
        request_block_table,
        sink_page_slots,
        request_refresh_generation,
        slot_major);
}

std::vector<torch::Tensor> step_recent_patch(
    torch::Tensor final_page_table,
    torch::Tensor request_block_table,
    torch::Tensor request_recent_first_logical_page,
    torch::Tensor request_recent_page_count,
    torch::Tensor request_recent_epoch,
    torch::Tensor selected_page_count,
    int64_t sink_page_slots,
    int64_t num_kv_heads) {
    return step_recent_patch_cuda(
        final_page_table,
        request_block_table,
        request_recent_first_logical_page,
        request_recent_page_count,
        request_recent_epoch,
        selected_page_count,
        sink_page_slots,
        num_kv_heads);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("refresh_static_materialize", &refresh_static_materialize, "FA sparse runtime static materialize (CUDA)");
    m.def("step_recent_patch", &step_recent_patch, "FA sparse runtime recent patch (CUDA)");
    m.def("gather_compact_kv_into_arena", &gather_compact_kv_into_arena, "FA sparse runtime compact KV gather (CUDA)");
    m.def("gather_compact_kv_into_arena_ptrs", &gather_compact_kv_into_arena_ptrs, "FA sparse runtime compact KV gather from cached pointer arrays (CUDA)");
    m.def("gather_compact_kv_into_arena_ptrs_tiled", &gather_compact_kv_into_arena_ptrs_tiled, "FA sparse runtime token-tiled compact KV gather from cached pointer arrays (CUDA)");
    m.def("gather_compact_kv_into_arena_ptrs_tiled_autolen", &gather_compact_kv_into_arena_ptrs_tiled_autolen, "FA sparse runtime token-tiled compact KV gather with in-kernel sink/persist lengths (CUDA)");
    m.def("gather_compact_kv_into_arena_ptrs_tiled_autolen_skip_unchanged", &gather_compact_kv_into_arena_ptrs_tiled_autolen_skip_unchanged, "FA sparse runtime token-tiled compact KV gather with unchanged compact_pos copy skip (CUDA)");
}
"""

    cuda_source = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <vector>

namespace {

constexpr int PAGE_SPARSE_STATUS_OK = 0;
constexpr int PAGE_SPARSE_STATUS_RECOVERABLE_MISS = 1;

__global__ void refresh_static_materialize_kernel(
    int32_t* __restrict__ final_page_table,
    const int32_t* __restrict__ selected_middle_pages,
    const int32_t* __restrict__ selected_middle_counts,
    const int32_t* __restrict__ request_slot_rows,
    const int32_t* __restrict__ request_block_table,
    const int32_t* __restrict__ request_refresh_generation,
    int batch_size,
    int selected_middle_rows,
    int num_kv_heads,
    int middle_slots,
    int block_table_cols,
    int max_page_count,
    int sink_page_slots,
    bool slot_major,
    int32_t* __restrict__ materialize_status,
    int32_t* __restrict__ applied_refresh_generation) {
    int req = blockIdx.x;
    if (req >= batch_size) {
        return;
    }

    __shared__ int req_status;
    __shared__ int middle_count;
    __shared__ int source_row;
    if (threadIdx.x == 0) {
        req_status = PAGE_SPARSE_STATUS_OK;
        source_row = slot_major ? request_slot_rows[req] : req;
        middle_count = 0;
        if (source_row < 0 || source_row >= selected_middle_rows) {
            req_status = PAGE_SPARSE_STATUS_RECOVERABLE_MISS;
        } else {
            middle_count = selected_middle_counts[source_row * num_kv_heads];
            if (middle_count <= 0 || middle_count > middle_slots) {
                req_status = PAGE_SPARSE_STATUS_RECOVERABLE_MISS;
            } else {
                for (int head = 1; head < num_kv_heads; ++head) {
                    if (selected_middle_counts[source_row * num_kv_heads + head] != middle_count) {
                        req_status = PAGE_SPARSE_STATUS_RECOVERABLE_MISS;
                        break;
                    }
                }
            }
        }
    }
    __syncthreads();

    if (req_status != PAGE_SPARSE_STATUS_OK) {
        if (threadIdx.x == 0) {
            materialize_status[req] = req_status;
            applied_refresh_generation[req] = 0;
        }
        return;
    }

    int static_cols = sink_page_slots + middle_count;
    int total = num_kv_heads * static_cols;
    for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        int head = idx / static_cols;
        int col = idx % static_cols;
        int out_row = req * num_kv_heads + head;
        int32_t page = 0;
        if (col < sink_page_slots) {
            if (col >= block_table_cols) {
                atomicExch(&req_status, PAGE_SPARSE_STATUS_RECOVERABLE_MISS);
                continue;
            }
            page = request_block_table[req * block_table_cols + col];
        } else {
            int middle_idx = col - sink_page_slots;
            int32_t logical_page =
                selected_middle_pages[(source_row * num_kv_heads + head) * middle_slots + middle_idx];
            if (logical_page < 0 || logical_page >= block_table_cols) {
                atomicExch(&req_status, PAGE_SPARSE_STATUS_RECOVERABLE_MISS);
                continue;
            }
            page = request_block_table[req * block_table_cols + logical_page];
        }
        final_page_table[out_row * max_page_count + col] = page;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        materialize_status[req] = req_status;
        applied_refresh_generation[req] =
            req_status == PAGE_SPARSE_STATUS_OK ? request_refresh_generation[req] : 0;
    }
}


__global__ void step_recent_patch_kernel(
    int32_t* __restrict__ final_page_table,
    const int32_t* __restrict__ request_block_table,
    const int32_t* __restrict__ request_recent_first_logical_page,
    const int32_t* __restrict__ request_recent_page_count,
    const int32_t* __restrict__ request_recent_epoch,
    const int32_t* __restrict__ selected_page_count,
    int batch_size,
    int num_kv_heads,
    int block_table_cols,
    int max_page_count,
    int sink_page_slots,
    int32_t* __restrict__ patch_status,
    int32_t* __restrict__ applied_recent_epoch) {
    int req = blockIdx.x;
    if (req >= batch_size) {
        return;
    }

    __shared__ int req_status;
    __shared__ int recent_count;
    __shared__ int total_page_count;
    __shared__ int middle_count;
    __shared__ int recent_first;
    __shared__ int tail_fill_page;
    if (threadIdx.x == 0) {
        req_status = PAGE_SPARSE_STATUS_OK;
        recent_count = request_recent_page_count[req];
        total_page_count = selected_page_count[req];
        middle_count = total_page_count - sink_page_slots - recent_count;
        recent_first = request_recent_first_logical_page[req];
        tail_fill_page = 0;

        if (recent_count <= 0 || total_page_count <= 0 || middle_count < 0 || total_page_count > max_page_count) {
            req_status = PAGE_SPARSE_STATUS_RECOVERABLE_MISS;
        } else if (recent_first < 0 || recent_first + recent_count > block_table_cols) {
            req_status = PAGE_SPARSE_STATUS_RECOVERABLE_MISS;
        } else {
            tail_fill_page = request_block_table[req * block_table_cols + recent_first + recent_count - 1];
        }
    }
    __syncthreads();

    if (req_status != PAGE_SPARSE_STATUS_OK) {
        if (threadIdx.x == 0) {
            patch_status[req] = req_status;
            applied_recent_epoch[req] = 0;
        }
        return;
    }

    int suffix_start = sink_page_slots + middle_count;
    int recent_total = num_kv_heads * recent_count;
    for (int idx = threadIdx.x; idx < recent_total; idx += blockDim.x) {
        int head = idx / recent_count;
        int col = idx % recent_count;
        int out_row = req * num_kv_heads + head;
        int32_t page = request_block_table[req * block_table_cols + recent_first + col];
        final_page_table[out_row * max_page_count + suffix_start + col] = page;
    }

    int tail_count = max_page_count - total_page_count;
    int tail_total = num_kv_heads * tail_count;
    for (int idx = threadIdx.x; idx < tail_total; idx += blockDim.x) {
        int head = idx / tail_count;
        int col = idx % tail_count;
        int out_row = req * num_kv_heads + head;
        final_page_table[out_row * max_page_count + total_page_count + col] = tail_fill_page;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        patch_status[req] = req_status;
        applied_recent_epoch[req] =
            req_status == PAGE_SPARSE_STATUS_OK ? request_recent_epoch[req] : 0;
    }
}

}  // namespace

std::vector<torch::Tensor> refresh_static_materialize_cuda(
    torch::Tensor final_page_table,
    torch::Tensor selected_middle_pages,
    torch::Tensor selected_middle_counts,
    torch::Tensor request_slot_rows,
    torch::Tensor request_block_table,
    int64_t sink_page_slots,
    torch::Tensor request_refresh_generation,
    bool slot_major) {
    TORCH_CHECK(final_page_table.is_cuda(), "final_page_table must be CUDA");
    TORCH_CHECK(selected_middle_pages.is_cuda(), "selected_middle_pages must be CUDA");
    TORCH_CHECK(selected_middle_counts.is_cuda(), "selected_middle_counts must be CUDA");
    TORCH_CHECK(request_slot_rows.is_cuda(), "request_slot_rows must be CUDA");
    TORCH_CHECK(request_block_table.is_cuda(), "request_block_table must be CUDA");
    TORCH_CHECK(request_refresh_generation.is_cuda(), "request_refresh_generation must be CUDA");
    TORCH_CHECK(final_page_table.scalar_type() == torch::kInt32, "final_page_table must be int32");
    TORCH_CHECK(selected_middle_pages.scalar_type() == torch::kInt32, "selected_middle_pages must be int32");
    TORCH_CHECK(selected_middle_counts.scalar_type() == torch::kInt32, "selected_middle_counts must be int32");
    TORCH_CHECK(request_slot_rows.scalar_type() == torch::kInt32, "request_slot_rows must be int32");
    TORCH_CHECK(request_block_table.scalar_type() == torch::kInt32, "request_block_table must be int32");
    TORCH_CHECK(request_refresh_generation.scalar_type() == torch::kInt32, "request_refresh_generation must be int32");
    TORCH_CHECK(final_page_table.dim() == 2, "final_page_table must be rank-2");
    TORCH_CHECK(selected_middle_pages.dim() == 3, "selected_middle_pages must be rank-3");
    TORCH_CHECK(selected_middle_counts.dim() == 2, "selected_middle_counts must be rank-2");
    TORCH_CHECK(request_slot_rows.dim() == 1, "request_slot_rows must be rank-1");
    TORCH_CHECK(request_block_table.dim() == 2, "request_block_table must be rank-2");
    TORCH_CHECK(request_refresh_generation.dim() == 1, "request_refresh_generation must be rank-1");
    TORCH_CHECK(final_page_table.is_contiguous(), "final_page_table must be contiguous");
    TORCH_CHECK(selected_middle_pages.is_contiguous(), "selected_middle_pages must be contiguous");
    TORCH_CHECK(selected_middle_counts.is_contiguous(), "selected_middle_counts must be contiguous");
    TORCH_CHECK(request_slot_rows.is_contiguous(), "request_slot_rows must be contiguous");
    TORCH_CHECK(request_block_table.is_contiguous(), "request_block_table must be contiguous");
    TORCH_CHECK(request_refresh_generation.is_contiguous(), "request_refresh_generation must be contiguous");

    int selected_middle_rows = static_cast<int>(selected_middle_pages.size(0));
    int num_kv_heads = static_cast<int>(selected_middle_pages.size(1));
    int middle_slots = static_cast<int>(selected_middle_pages.size(2));
    int batch_size = static_cast<int>(request_slot_rows.size(0));
    TORCH_CHECK(selected_middle_counts.size(0) == selected_middle_rows, "selected_middle_counts row mismatch");
    TORCH_CHECK(selected_middle_counts.size(1) == num_kv_heads, "selected_middle_counts num_kv_heads mismatch");
    TORCH_CHECK(request_block_table.size(0) == batch_size, "request_block_table batch mismatch");
    TORCH_CHECK(request_refresh_generation.size(0) == batch_size, "request_refresh_generation batch mismatch");
    TORCH_CHECK(final_page_table.size(0) == batch_size * num_kv_heads, "final_page_table row mismatch");
    // [STATIC-COLS-BOUND] kernel writes static_cols = sink + middle_count
    // columns with only middle_count <= middle_slots checked device-side; a
    // config where sink + middle_slots exceeds the final table width would
    // silently spill across rows (the Python reference shape-errors instead).
    TORCH_CHECK(
        sink_page_slots + static_cast<int64_t>(middle_slots) <= final_page_table.size(1),
        "sink_page_slots + middle_slots exceeds final_page_table columns");

    auto materialize_status = torch::zeros({batch_size}, final_page_table.options());
    auto applied_refresh_generation = torch::zeros({batch_size}, request_refresh_generation.options());

    auto stream = at::cuda::getCurrentCUDAStream();
    constexpr int threads = 256;
    refresh_static_materialize_kernel<<<batch_size, threads, 0, stream>>>(
        final_page_table.data_ptr<int32_t>(),
        selected_middle_pages.data_ptr<int32_t>(),
        selected_middle_counts.data_ptr<int32_t>(),
        request_slot_rows.data_ptr<int32_t>(),
        request_block_table.data_ptr<int32_t>(),
        request_refresh_generation.data_ptr<int32_t>(),
        batch_size,
        selected_middle_rows,
        num_kv_heads,
        middle_slots,
        static_cast<int>(request_block_table.size(1)),
        static_cast<int>(final_page_table.size(1)),
        static_cast<int>(sink_page_slots),
        slot_major,
        materialize_status.data_ptr<int32_t>(),
        applied_refresh_generation.data_ptr<int32_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {materialize_status, applied_refresh_generation};
}

std::vector<torch::Tensor> step_recent_patch_cuda(
    torch::Tensor final_page_table,
    torch::Tensor request_block_table,
    torch::Tensor request_recent_first_logical_page,
    torch::Tensor request_recent_page_count,
    torch::Tensor request_recent_epoch,
    torch::Tensor selected_page_count,
    int64_t sink_page_slots,
    int64_t num_kv_heads) {
    TORCH_CHECK(final_page_table.is_cuda(), "final_page_table must be CUDA");
    TORCH_CHECK(request_block_table.is_cuda(), "request_block_table must be CUDA");
    TORCH_CHECK(request_recent_first_logical_page.is_cuda(), "request_recent_first_logical_page must be CUDA");
    TORCH_CHECK(request_recent_page_count.is_cuda(), "request_recent_page_count must be CUDA");
    TORCH_CHECK(request_recent_epoch.is_cuda(), "request_recent_epoch must be CUDA");
    TORCH_CHECK(selected_page_count.is_cuda(), "selected_page_count must be CUDA");
    TORCH_CHECK(final_page_table.scalar_type() == torch::kInt32, "final_page_table must be int32");
    TORCH_CHECK(request_block_table.scalar_type() == torch::kInt32, "request_block_table must be int32");
    TORCH_CHECK(request_recent_first_logical_page.scalar_type() == torch::kInt32, "request_recent_first_logical_page must be int32");
    TORCH_CHECK(request_recent_page_count.scalar_type() == torch::kInt32, "request_recent_page_count must be int32");
    TORCH_CHECK(request_recent_epoch.scalar_type() == torch::kInt32, "request_recent_epoch must be int32");
    TORCH_CHECK(selected_page_count.scalar_type() == torch::kInt32, "selected_page_count must be int32");
    TORCH_CHECK(final_page_table.dim() == 2, "final_page_table must be rank-2");
    TORCH_CHECK(request_block_table.dim() == 2, "request_block_table must be rank-2");
    TORCH_CHECK(request_recent_first_logical_page.dim() == 1, "request_recent_first_logical_page must be rank-1");
    TORCH_CHECK(request_recent_page_count.dim() == 1, "request_recent_page_count must be rank-1");
    TORCH_CHECK(request_recent_epoch.dim() == 1, "request_recent_epoch must be rank-1");
    TORCH_CHECK(selected_page_count.dim() == 1, "selected_page_count must be rank-1");
    TORCH_CHECK(final_page_table.is_contiguous(), "final_page_table must be contiguous");
    TORCH_CHECK(request_block_table.is_contiguous(), "request_block_table must be contiguous");
    TORCH_CHECK(request_recent_first_logical_page.is_contiguous(), "request_recent_first_logical_page must be contiguous");
    TORCH_CHECK(request_recent_page_count.is_contiguous(), "request_recent_page_count must be contiguous");
    TORCH_CHECK(request_recent_epoch.is_contiguous(), "request_recent_epoch must be contiguous");
    TORCH_CHECK(selected_page_count.is_contiguous(), "selected_page_count must be contiguous");

    int batch_size = static_cast<int>(request_recent_epoch.size(0));
    TORCH_CHECK(request_block_table.size(0) == batch_size, "request_block_table batch mismatch");
    TORCH_CHECK(request_recent_first_logical_page.size(0) == batch_size, "request_recent_first_logical_page batch mismatch");
    TORCH_CHECK(request_recent_page_count.size(0) == batch_size, "request_recent_page_count batch mismatch");
    TORCH_CHECK(selected_page_count.size(0) == batch_size, "selected_page_count batch mismatch");
    TORCH_CHECK(final_page_table.size(0) == batch_size * static_cast<int>(num_kv_heads), "final_page_table row mismatch");

    auto patch_status = torch::zeros({batch_size}, final_page_table.options());
    auto applied_recent_epoch = torch::zeros({batch_size}, request_recent_epoch.options());

    auto stream = at::cuda::getCurrentCUDAStream();
    constexpr int threads = 256;
    step_recent_patch_kernel<<<batch_size, threads, 0, stream>>>(
        final_page_table.data_ptr<int32_t>(),
        request_block_table.data_ptr<int32_t>(),
        request_recent_first_logical_page.data_ptr<int32_t>(),
        request_recent_page_count.data_ptr<int32_t>(),
        request_recent_epoch.data_ptr<int32_t>(),
        selected_page_count.data_ptr<int32_t>(),
        batch_size,
        static_cast<int>(num_kv_heads),
        static_cast<int>(request_block_table.size(1)),
        static_cast<int>(final_page_table.size(1)),
        static_cast<int>(sink_page_slots),
        patch_status.data_ptr<int32_t>(),
        applied_recent_epoch.data_ptr<int32_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {patch_status, applied_recent_epoch};
}

namespace {

__device__ __forceinline__ int i32_min(int a, int b) {
    return a < b ? a : b;
}

__device__ __forceinline__ int i32_max(int a, int b) {
    return a > b ? a : b;
}

__device__ __forceinline__ void compute_compact_autolen(
    int seq_len_raw,
    int sink_cap,
    int recent_cfg,
    int k_head,
    int threshold_tokens,
    int page_size,
    int stride_tokens,
    int selected_k,
    int* sink_out,
    int* persist_out) {
    int seq_len = i32_max(seq_len_raw, 0);
    bool do_rebuild = threshold_tokens > 0
        ? seq_len >= threshold_tokens
        : seq_len > 0;
    if (!do_rebuild) {
        *sink_out = 0;
        *persist_out = 0;
        return;
    }

    int sink_len = i32_min(seq_len, i32_max(sink_cap, 0));
    int recent_start = 0;
    if (recent_cfg > 0 && page_size > 0) {
        int cap = i32_min(seq_len, recent_cfg);
        recent_start = ((seq_len - cap) / page_size) * page_size;
        recent_start = i32_max(recent_start, 0);
    }
    int allowed_len = i32_max(recent_start - sink_len, 0);
    int persist_len = 0;
    if (k_head > 0) {
        persist_len = i32_min(i32_min(allowed_len, k_head), selected_k);
    }
    int total_len = i32_min(i32_max(stride_tokens, 0), sink_len + persist_len);
    persist_len = i32_max(total_len - sink_len, 0);
    *sink_out = sink_len;
    *persist_out = persist_len;
}

// Semantics mirror the authoritative Triton
// _kernel_gather_compact_kv_batched_layers_indexed kernel:
//   - Both sink and persist positions go through a unified block_table lookup
//     (logical pos -> block_idx/offset -> block_table[row][block_idx]).
//   - Destination slot is read from slot_tensor[b] (not b directly).
//   - row_tensor[l, b] provides the block_table row (not b directly).
//   - compact_pos[l] layout is [H_kv, B*stride_tokens] i32, head-major.
__global__ void gather_compact_kv_into_arena_kernel(
    const void* const* __restrict__ flat_k_ptrs,
    const void* const* __restrict__ flat_v_ptrs,
    void* const* __restrict__ compact_k_ptrs,
    void* const* __restrict__ compact_v_ptrs,
    int32_t* const* __restrict__ compact_pos_ptrs,
    const int32_t* const* __restrict__ block_table_ptrs,
    const int32_t* __restrict__ row_tensor,        // [L, B]
    const int32_t* __restrict__ slot_tensor,       // [B]
    const int32_t* __restrict__ selected_indices,  // [L, B, H_kv, k_persist] LOGICAL
    const int32_t* __restrict__ sink_len,          // [L, B]
    const int32_t* __restrict__ persist_len,       // [L, B]
    int num_layers,
    int batch_size,
    int num_kv_heads,
    int head_dim,
    int k_persist,
    int page_size,
    int stride_tokens,
    int block_table_cols,
    int compact_pos_per_layer_tokens,
    int64_t flat_k_stride0,
    int64_t flat_k_stride1,
    int64_t flat_k_stride2,
    int64_t flat_v_stride0,
    int64_t flat_v_stride1,
    int64_t flat_v_stride2,
    int64_t compact_k_stride0,
    int64_t compact_k_stride1,
    int64_t compact_k_stride2,
    int64_t compact_v_stride0,
    int64_t compact_v_stride1,
    int64_t compact_v_stride2,
    int64_t compact_pos_stride_head,
    int64_t compact_pos_stride_tok,
    int64_t block_table_stride0,
    int64_t block_table_stride1) {

    int lb = blockIdx.x;                   // (layer * batch + b)
    int layer = lb / batch_size;
    int b = lb % batch_size;
    int head = blockIdx.y;
    int tid = threadIdx.x;

    if (layer >= num_layers || b >= batch_size || head >= num_kv_heads) return;

    int s_len = sink_len[layer * batch_size + b];
    int p_len = persist_len[layer * batch_size + b];
    int total_tokens = s_len + p_len;
    if (total_tokens > stride_tokens) total_tokens = stride_tokens;
    if (total_tokens <= 0) return;

    const __nv_bfloat16* src_k = reinterpret_cast<const __nv_bfloat16*>(flat_k_ptrs[layer]);
    const __nv_bfloat16* src_v = reinterpret_cast<const __nv_bfloat16*>(flat_v_ptrs[layer]);
    __nv_bfloat16* dst_k = reinterpret_cast<__nv_bfloat16*>(compact_k_ptrs[layer]);
    __nv_bfloat16* dst_v = reinterpret_cast<__nv_bfloat16*>(compact_v_ptrs[layer]);
    int32_t* dst_pos = compact_pos_ptrs[layer];
    const int32_t* btable = block_table_ptrs[layer];

    int row = row_tensor[layer * batch_size + b];
    int slot = slot_tensor[b];
    int64_t slot_offset_tokens = (int64_t)slot * (int64_t)stride_tokens;
    int max_block = block_table_cols > 0 ? block_table_cols - 1 : 0;

    for (int t = 0; t < total_tokens; ++t) {
        int pos;                              // LOGICAL pos; invalid selected entries fall back to middle order.
        bool valid = true;
        if (t < s_len) {
            pos = t;
        } else {
            int sel = selected_indices[
                ((layer * batch_size + b) * num_kv_heads + head) * k_persist + (t - s_len)
            ];
            if (sel < 0) {
                pos = s_len + (t - s_len);
            } else {
                pos = sel;
            }
        }
        int pos_safe = valid ? pos : 0;
        int block_idx = pos_safe / page_size;
        int offset = pos_safe - block_idx * page_size;
        if (block_idx > max_block) block_idx = max_block;
        int block_id = btable[row * block_table_stride0 + block_idx * block_table_stride1];
        int64_t dst_token = slot_offset_tokens + (int64_t)t;
        if (valid) {
            int64_t src_token = (int64_t)block_id * (int64_t)page_size + (int64_t)offset;
            int64_t src_base = src_token * flat_k_stride0 + (int64_t)head * flat_k_stride1;
            int64_t src_base_v = src_token * flat_v_stride0 + (int64_t)head * flat_v_stride1;
            int64_t dst_base = dst_token * compact_k_stride0 + (int64_t)head * compact_k_stride1;
            int64_t dst_base_v = dst_token * compact_v_stride0 + (int64_t)head * compact_v_stride1;
            // [GATHER-VEC8] d-loop 8xbf16 向量化:host 实现体 TORCH_CHECK 钉死
            // head_dim%8==0 && 各 stride2==1 && token/head stride 8 对齐(生产
            // 恒真,违约 fail-fast 不降级);uint4=8xbf16 单指令搬运。src/dst
            // base 为 8 的整数倍元素偏移 + tensor base 256B 对齐 → 16B 对齐。
            {
                const uint4* src_k_vec = reinterpret_cast<const uint4*>(src_k + src_base);
                uint4* dst_k_vec = reinterpret_cast<uint4*>(dst_k + dst_base);
                const uint4* src_v_vec = reinterpret_cast<const uint4*>(src_v + src_base_v);
                uint4* dst_v_vec = reinterpret_cast<uint4*>(dst_v + dst_base_v);
                const int vecs = head_dim >> 3;
                for (int vi = tid; vi < vecs; vi += blockDim.x) {
                    dst_k_vec[vi] = src_k_vec[vi];
                    dst_v_vec[vi] = src_v_vec[vi];
                }
            }
        }
        // Write compact_pos[head, dst_token] = pos after invalid-selection fallback.
        if (tid == 0) {
            int64_t pos_idx =
                (int64_t)head * compact_pos_stride_head + dst_token * compact_pos_stride_tok;
            dst_pos[pos_idx] = pos;
        }
    }
}

__global__ void gather_compact_kv_into_arena_tiled_kernel(
    const void* const* __restrict__ flat_k_ptrs,
    const void* const* __restrict__ flat_v_ptrs,
    void* const* __restrict__ compact_k_ptrs,
    void* const* __restrict__ compact_v_ptrs,
    int32_t* const* __restrict__ compact_pos_ptrs,
    const int32_t* const* __restrict__ block_table_ptrs,
    const int32_t* __restrict__ row_tensor,        // [L, B]
    const int32_t* __restrict__ slot_tensor,       // [B]
    const int32_t* __restrict__ selected_indices,  // [L, B, H_kv, k_persist] LOGICAL
    const int32_t* __restrict__ sink_len,          // [L, B]
    const int32_t* __restrict__ persist_len,       // [L, B]
    int num_layers,
    int batch_size,
    int num_kv_heads,
    int head_dim,
    int k_persist,
    int page_size,
    int stride_tokens,
    int block_table_cols,
    int compact_pos_per_layer_tokens,
    int64_t flat_k_stride0,
    int64_t flat_k_stride1,
    int64_t flat_k_stride2,
    int64_t flat_v_stride0,
    int64_t flat_v_stride1,
    int64_t flat_v_stride2,
    int64_t compact_k_stride0,
    int64_t compact_k_stride1,
    int64_t compact_k_stride2,
    int64_t compact_v_stride0,
    int64_t compact_v_stride1,
    int64_t compact_v_stride2,
    int64_t compact_pos_stride_head,
    int64_t compact_pos_stride_tok,
    int64_t block_table_stride0,
    int64_t block_table_stride1,
    int tile_tokens) {

    int lb = blockIdx.x;                   // (layer * batch + b)
    int layer = lb / batch_size;
    int b = lb % batch_size;
    int head = blockIdx.y;
    int tile = blockIdx.z;
    int tid = threadIdx.x;

    if (layer >= num_layers || b >= batch_size || head >= num_kv_heads) return;

    int s_len = sink_len[layer * batch_size + b];
    int p_len = persist_len[layer * batch_size + b];
    int total_tokens = s_len + p_len;
    if (total_tokens > stride_tokens) total_tokens = stride_tokens;
    if (total_tokens <= 0) return;

    int token_start = tile * tile_tokens;
    if (token_start >= total_tokens) return;
    int token_end = token_start + tile_tokens;
    if (token_end > total_tokens) token_end = total_tokens;

    const __nv_bfloat16* src_k = reinterpret_cast<const __nv_bfloat16*>(flat_k_ptrs[layer]);
    const __nv_bfloat16* src_v = reinterpret_cast<const __nv_bfloat16*>(flat_v_ptrs[layer]);
    __nv_bfloat16* dst_k = reinterpret_cast<__nv_bfloat16*>(compact_k_ptrs[layer]);
    __nv_bfloat16* dst_v = reinterpret_cast<__nv_bfloat16*>(compact_v_ptrs[layer]);
    int32_t* dst_pos = compact_pos_ptrs[layer];
    const int32_t* btable = block_table_ptrs[layer];

    int row = row_tensor[layer * batch_size + b];
    int slot = slot_tensor[b];
    int64_t slot_offset_tokens = (int64_t)slot * (int64_t)stride_tokens;
    int max_block = block_table_cols > 0 ? block_table_cols - 1 : 0;

    for (int t = token_start; t < token_end; ++t) {
        int pos;                              // LOGICAL pos; invalid selected entries fall back to middle order.
        bool valid = true;
        if (t < s_len) {
            pos = t;
        } else {
            int sel = selected_indices[
                ((layer * batch_size + b) * num_kv_heads + head) * k_persist + (t - s_len)
            ];
            if (sel < 0) {
                pos = s_len + (t - s_len);
            } else {
                pos = sel;
            }
        }
        int pos_safe = valid ? pos : 0;
        int block_idx = pos_safe / page_size;
        int offset = pos_safe - block_idx * page_size;
        if (block_idx > max_block) block_idx = max_block;
        int block_id = btable[row * block_table_stride0 + block_idx * block_table_stride1];
        int64_t dst_token = slot_offset_tokens + (int64_t)t;
        if (valid) {
            int64_t src_token = (int64_t)block_id * (int64_t)page_size + (int64_t)offset;
            int64_t src_base = src_token * flat_k_stride0 + (int64_t)head * flat_k_stride1;
            int64_t src_base_v = src_token * flat_v_stride0 + (int64_t)head * flat_v_stride1;
            int64_t dst_base = dst_token * compact_k_stride0 + (int64_t)head * compact_k_stride1;
            int64_t dst_base_v = dst_token * compact_v_stride0 + (int64_t)head * compact_v_stride1;
            // [GATHER-VEC8] d-loop 8xbf16 向量化:host 实现体 TORCH_CHECK 钉死
            // head_dim%8==0 && 各 stride2==1 && token/head stride 8 对齐(生产
            // 恒真,违约 fail-fast 不降级);uint4=8xbf16 单指令搬运。src/dst
            // base 为 8 的整数倍元素偏移 + tensor base 256B 对齐 → 16B 对齐。
            {
                const uint4* src_k_vec = reinterpret_cast<const uint4*>(src_k + src_base);
                uint4* dst_k_vec = reinterpret_cast<uint4*>(dst_k + dst_base);
                const uint4* src_v_vec = reinterpret_cast<const uint4*>(src_v + src_base_v);
                uint4* dst_v_vec = reinterpret_cast<uint4*>(dst_v + dst_base_v);
                const int vecs = head_dim >> 3;
                for (int vi = tid; vi < vecs; vi += blockDim.x) {
                    dst_k_vec[vi] = src_k_vec[vi];
                    dst_v_vec[vi] = src_v_vec[vi];
                }
            }
        }
        if (tid == 0) {
            int64_t pos_idx =
                (int64_t)head * compact_pos_stride_head + dst_token * compact_pos_stride_tok;
            dst_pos[pos_idx] = pos;
        }
    }
}

__global__ void gather_compact_kv_into_arena_tiled_autolen_kernel(
    const void* const* __restrict__ flat_k_ptrs,
    const void* const* __restrict__ flat_v_ptrs,
    void* const* __restrict__ compact_k_ptrs,
    void* const* __restrict__ compact_v_ptrs,
    int32_t* const* __restrict__ compact_pos_ptrs,
    const int32_t* const* __restrict__ block_table_ptrs,
    const int32_t* __restrict__ row_tensor,        // [L, B]
    const int32_t* __restrict__ slot_tensor,       // [B]
    const int32_t* __restrict__ selected_indices,  // [L, B, H_kv, k_persist] LOGICAL
    const int32_t* __restrict__ seq_lens,          // [B], shared across layers
    int num_layers,
    int batch_size,
    int num_kv_heads,
    int head_dim,
    int k_persist,
    int page_size,
    int stride_tokens,
    int block_table_cols,
    int compact_pos_per_layer_tokens,
    int sink_cap,
    int recent_cfg,
    int k_head,
    int threshold_tokens,
    int64_t flat_k_stride0,
    int64_t flat_k_stride1,
    int64_t flat_k_stride2,
    int64_t flat_v_stride0,
    int64_t flat_v_stride1,
    int64_t flat_v_stride2,
    int64_t compact_k_stride0,
    int64_t compact_k_stride1,
    int64_t compact_k_stride2,
    int64_t compact_v_stride0,
    int64_t compact_v_stride1,
    int64_t compact_v_stride2,
    int64_t compact_pos_stride_head,
    int64_t compact_pos_stride_tok,
    int64_t block_table_stride0,
    int64_t block_table_stride1,
    int tile_tokens,
    bool skip_unchanged) {

    int lb = blockIdx.x;
    int layer = lb / batch_size;
    int b = lb % batch_size;
    int head = blockIdx.y;
    int tile = blockIdx.z;
    int tid = threadIdx.x;

    if (layer >= num_layers || b >= batch_size || head >= num_kv_heads) return;

    int s_len = 0;
    int p_len = 0;
    compute_compact_autolen(
        seq_lens[b],
        sink_cap,
        recent_cfg,
        k_head,
        threshold_tokens,
        page_size,
        stride_tokens,
        k_persist,
        &s_len,
        &p_len);
    int total_tokens = s_len + p_len;
    if (total_tokens > stride_tokens) total_tokens = stride_tokens;
    if (total_tokens <= 0) return;

    int token_start = tile * tile_tokens;
    if (token_start >= total_tokens) return;
    int token_end = token_start + tile_tokens;
    if (token_end > total_tokens) token_end = total_tokens;
    int tile_count = token_end - token_start;

    const __nv_bfloat16* src_k = reinterpret_cast<const __nv_bfloat16*>(flat_k_ptrs[layer]);
    const __nv_bfloat16* src_v = reinterpret_cast<const __nv_bfloat16*>(flat_v_ptrs[layer]);
    __nv_bfloat16* dst_k = reinterpret_cast<__nv_bfloat16*>(compact_k_ptrs[layer]);
    __nv_bfloat16* dst_v = reinterpret_cast<__nv_bfloat16*>(compact_v_ptrs[layer]);
    int32_t* dst_pos = compact_pos_ptrs[layer];
    const int32_t* btable = block_table_ptrs[layer];

    int row = row_tensor[layer * batch_size + b];
    int slot = slot_tensor[b];
    int64_t slot_offset_tokens = (int64_t)slot * (int64_t)stride_tokens;
    int max_block = block_table_cols > 0 ? block_table_cols - 1 : 0;
    int seq_len_b = seq_lens[b];
    // [GATHER-OOB-TRAP 2026-07-07] fail-fast at the first scene instead of
    // wild reads/writes into neighbour memory (no clamp, no fallback):
    //  - dst bound: a poisoned slot would make dst_token index compact K/V/pos
    //    out of the per-layer arena (compact_pos_per_layer_tokens was passed
    //    but never checked before).
    //  - coverage: grid.z is sized from the HOST-side kv_lens snapshot; if the
    //    GPU-side seq_lens disagrees upward, tail tokens are silently never
    //    written — turn that two-source drift into a first-scene signal.
    if (tid == 0) {
        if (slot < 0 ||
            slot_offset_tokens + (int64_t)total_tokens >
                (int64_t)compact_pos_per_layer_tokens) {
            printf("SFI_GATHER_OOB dst: layer=%d b=%d slot=%d total=%d cap=%d\n",
                   layer, b, slot, total_tokens, compact_pos_per_layer_tokens);
            __trap();
        }
        if (tile == 0 &&
            (int64_t)total_tokens > (int64_t)gridDim.z * (int64_t)tile_tokens) {
            printf("SFI_GATHER_OOB cover: layer=%d b=%d total=%d tiles=%d tile=%d\n",
                   layer, b, total_tokens, (int)gridDim.z, tile_tokens);
            __trap();
        }
    }
    constexpr int kMaxSharedTileTokens = 128;
    __shared__ int shared_pos[kMaxSharedTileTokens];
    __shared__ int shared_copy[kMaxSharedTileTokens];
    const bool use_shared_tile_decode = tile_count <= kMaxSharedTileTokens;

    if (use_shared_tile_decode) {
        for (int local_t = tid; local_t < tile_count; local_t += blockDim.x) {
            int t = token_start + local_t;
            int pos;
            if (t < s_len) {
                pos = t;
            } else {
                int sel = selected_indices[
                    ((layer * batch_size + b) * num_kv_heads + head) * k_persist + (t - s_len)
                ];
                if (sel < 0) {
                    pos = s_len + (t - s_len);
                } else {
                    pos = sel;
                }
            }
            int64_t dst_token = slot_offset_tokens + (int64_t)t;
            int64_t pos_idx =
                (int64_t)head * compact_pos_stride_head + dst_token * compact_pos_stride_tok;
            bool copy_token = true;
            if (skip_unchanged) {
                int old_pos = dst_pos[pos_idx];
                copy_token = old_pos != pos;
            }
            shared_pos[local_t] = pos;
            shared_copy[local_t] = copy_token ? 1 : 0;
        }
        __syncthreads();
    }

    for (int t = token_start; t < token_end; ++t) {
        int pos;
        bool valid = true;
        int local_t = t - token_start;
        if (use_shared_tile_decode) {
            pos = shared_pos[local_t];
        } else {
            if (t < s_len) {
                pos = t;
            } else {
                int sel = selected_indices[
                    ((layer * batch_size + b) * num_kv_heads + head) * k_persist + (t - s_len)
                ];
                if (sel < 0) {
                    pos = s_len + (t - s_len);
                } else {
                    pos = sel;
                }
            }
        }
        int64_t dst_token = slot_offset_tokens + (int64_t)t;
        int64_t pos_idx =
            (int64_t)head * compact_pos_stride_head + dst_token * compact_pos_stride_tok;
        bool copy_token = valid;
        if (use_shared_tile_decode) {
            copy_token = shared_copy[local_t] != 0;
        } else if (copy_token && skip_unchanged) {
            int old_pos = dst_pos[pos_idx];
            copy_token = old_pos != pos;
        }
        if (copy_token) {
            // [GATHER-OOB-TRAP] pos is a logical token position and MUST point
            // at an existing token of THIS request: sink pos==t<s_len,
            // selected pos in [0, recent_start), ordinal fallback < total.
            // A pos outside [0, seq_len) is a poisoned selected_indices value
            // (e.g. a leaked masked pick); the old block_idx clamp would
            // silently read the block-table tail (unallocated garbage block
            // ids -> wild reads). Fail fast with a fingerprint instead.
            if (pos < 0 || pos >= seq_len_b) {
                printf(
                    "SFI_GATHER_OOB pos: layer=%d b=%d head=%d t=%d pos=%d seq=%d\n",
                    layer, b, head, t, pos, seq_len_b);
                __trap();
            }
            int pos_safe = valid ? pos : 0;
            int block_idx = pos_safe / page_size;
            int offset = pos_safe - block_idx * page_size;
            if (block_idx > max_block) block_idx = max_block;
            int block_id = btable[row * block_table_stride0 + block_idx * block_table_stride1];
            // [GATHER-OOB-TRAP] sanitizer 终审:replay 内本 kernel 的 src 读越
            // 下界 = btable 该列值为负(-1 负寻址)。毒列坐标 fail-fast 落盘:
            // row/block_idx/pos/seq 直接指认写侧(republish/materialize/初始态)。
            if (block_id < 0) {
                printf(
                    "SFI_GATHER_OOB blkid: layer=%d b=%d head=%d t=%d pos=%d "
                    "seq=%d row=%d bidx=%d blkid=%d\n",
                    layer, b, head, t, pos, seq_len_b, row, block_idx, block_id);
                __trap();
            }
            int64_t src_token = (int64_t)block_id * (int64_t)page_size + (int64_t)offset;
            int64_t src_base = src_token * flat_k_stride0 + (int64_t)head * flat_k_stride1;
            int64_t src_base_v = src_token * flat_v_stride0 + (int64_t)head * flat_v_stride1;
            int64_t dst_base = dst_token * compact_k_stride0 + (int64_t)head * compact_k_stride1;
            int64_t dst_base_v = dst_token * compact_v_stride0 + (int64_t)head * compact_v_stride1;
            // [GATHER-VEC8] d-loop 8xbf16 向量化:host 实现体 TORCH_CHECK 钉死
            // head_dim%8==0 && 各 stride2==1 && token/head stride 8 对齐(生产
            // 恒真,违约 fail-fast 不降级);uint4=8xbf16 单指令搬运。src/dst
            // base 为 8 的整数倍元素偏移 + tensor base 256B 对齐 → 16B 对齐。
            {
                const uint4* src_k_vec = reinterpret_cast<const uint4*>(src_k + src_base);
                uint4* dst_k_vec = reinterpret_cast<uint4*>(dst_k + dst_base);
                const uint4* src_v_vec = reinterpret_cast<const uint4*>(src_v + src_base_v);
                uint4* dst_v_vec = reinterpret_cast<uint4*>(dst_v + dst_base_v);
                const int vecs = head_dim >> 3;
                for (int vi = tid; vi < vecs; vi += blockDim.x) {
                    dst_k_vec[vi] = src_k_vec[vi];
                    dst_v_vec[vi] = src_v_vec[vi];
                }
            }
        }
        if (copy_token && tid == 0) {
            dst_pos[pos_idx] = pos;
        }
    }
}

}  // namespace


void gather_compact_kv_into_arena_ptrs(
    torch::Tensor flat_k_ptrs,
    torch::Tensor flat_v_ptrs,
    torch::Tensor compact_k_ptrs,
    torch::Tensor compact_v_ptrs,
    torch::Tensor compact_pos_ptrs,
    torch::Tensor block_table_ptrs,
    torch::Tensor row_tensor,
    torch::Tensor slot_tensor,
    torch::Tensor selected_indices,
    torch::Tensor sink_len,
    torch::Tensor persist_len,
    int64_t page_size,
    int64_t stride_tokens,
    int64_t block_table_cols,
    int64_t head_dim,
    int64_t compact_pos_per_layer_tokens,
    int64_t flat_k_stride0,
    int64_t flat_k_stride1,
    int64_t flat_k_stride2,
    int64_t flat_v_stride0,
    int64_t flat_v_stride1,
    int64_t flat_v_stride2,
    int64_t compact_k_stride0,
    int64_t compact_k_stride1,
    int64_t compact_k_stride2,
    int64_t compact_v_stride0,
    int64_t compact_v_stride1,
    int64_t compact_v_stride2,
    int64_t compact_pos_stride_head,
    int64_t compact_pos_stride_tok,
    int64_t block_table_stride0,
    int64_t block_table_stride1) {
    TORCH_CHECK(flat_k_ptrs.dim() == 1, "flat_k_ptrs must be [L]");
    int L = static_cast<int>(flat_k_ptrs.size(0));
    TORCH_CHECK(L > 0, "flat_k_ptrs must be non-empty");
    TORCH_CHECK(flat_v_ptrs.dim() == 1 && compact_k_ptrs.dim() == 1
                && compact_v_ptrs.dim() == 1 && compact_pos_ptrs.dim() == 1
                && block_table_ptrs.dim() == 1,
                "pointer tensors must be rank-1");
    TORCH_CHECK(flat_v_ptrs.size(0) == L && compact_k_ptrs.size(0) == L
                && compact_v_ptrs.size(0) == L && compact_pos_ptrs.size(0) == L
                && block_table_ptrs.size(0) == L,
                "all pointer tensors must have same length");
    TORCH_CHECK(flat_k_ptrs.dtype() == torch::kInt64 && flat_v_ptrs.dtype() == torch::kInt64
                && compact_k_ptrs.dtype() == torch::kInt64 && compact_v_ptrs.dtype() == torch::kInt64
                && compact_pos_ptrs.dtype() == torch::kInt64 && block_table_ptrs.dtype() == torch::kInt64,
                "pointer tensors must be int64");
    TORCH_CHECK(flat_k_ptrs.is_cuda() && flat_v_ptrs.is_cuda() && compact_k_ptrs.is_cuda()
                && compact_v_ptrs.is_cuda() && compact_pos_ptrs.is_cuda() && block_table_ptrs.is_cuda(),
                "pointer tensors must be CUDA tensors");
    TORCH_CHECK(flat_k_ptrs.is_contiguous() && flat_v_ptrs.is_contiguous()
                && compact_k_ptrs.is_contiguous() && compact_v_ptrs.is_contiguous()
                && compact_pos_ptrs.is_contiguous() && block_table_ptrs.is_contiguous(),
                "pointer tensors must be contiguous");
    TORCH_CHECK(selected_indices.dim() == 4, "selected_indices must be [L, B, H_kv, k]");
    TORCH_CHECK(selected_indices.size(0) == L, "selected_indices[0] must equal number of layers");
    TORCH_CHECK(selected_indices.dtype() == torch::kInt32, "selected_indices must be int32");
    TORCH_CHECK(selected_indices.is_cuda() && selected_indices.is_contiguous(),
                "selected_indices must be a contiguous CUDA tensor");
    TORCH_CHECK(sink_len.dim() == 2 && persist_len.dim() == 2, "sink/persist_len must be [L, B]");
    TORCH_CHECK(sink_len.dtype() == torch::kInt32 && persist_len.dtype() == torch::kInt32,
                "sink/persist_len must be int32");
    TORCH_CHECK(sink_len.is_cuda() && persist_len.is_cuda()
                && sink_len.is_contiguous() && persist_len.is_contiguous(),
                "sink/persist_len must be contiguous CUDA tensors");
    TORCH_CHECK(row_tensor.dim() == 2 && row_tensor.size(0) == L,
                "row_tensor must be [L, B]");
    TORCH_CHECK(row_tensor.dtype() == torch::kInt32 && row_tensor.is_cuda()
                && row_tensor.is_contiguous(),
                "row_tensor must be a contiguous CUDA int32 tensor");
    TORCH_CHECK(slot_tensor.dim() == 1, "slot_tensor must be [B]");
    TORCH_CHECK(slot_tensor.dtype() == torch::kInt32 && slot_tensor.is_cuda()
                && slot_tensor.is_contiguous(),
                "slot_tensor must be a contiguous CUDA int32 tensor");

    int B = static_cast<int>(selected_indices.size(1));
    int H_kv = static_cast<int>(selected_indices.size(2));
    int k_persist = static_cast<int>(selected_indices.size(3));
    TORCH_CHECK(row_tensor.size(1) == B, "row_tensor must be [L, B]");
    TORCH_CHECK(slot_tensor.size(0) == B, "slot_tensor must be [B]");
    TORCH_CHECK(sink_len.size(0) == L && sink_len.size(1) == B,
                "sink_len must be [L, B]");
    TORCH_CHECK(persist_len.size(0) == L && persist_len.size(1) == B,
                "persist_len must be [L, B]");

    // [GATHER-VEC8] 向量化合同(违约 fail-fast,不降级):
    TORCH_CHECK(head_dim % 8 == 0, "gather vec8: head_dim must be a multiple of 8");
    TORCH_CHECK(flat_k_stride2 == 1 && flat_v_stride2 == 1 &&
                compact_k_stride2 == 1 && compact_v_stride2 == 1,
                "gather vec8: last-dim strides must be 1 (contiguous head_dim)");
    TORCH_CHECK(flat_k_stride0 % 8 == 0 && flat_k_stride1 % 8 == 0 &&
                flat_v_stride0 % 8 == 0 && flat_v_stride1 % 8 == 0 &&
                compact_k_stride0 % 8 == 0 && compact_k_stride1 % 8 == 0 &&
                compact_v_stride0 % 8 == 0 && compact_v_stride1 % 8 == 0,
                "gather vec8: token/head strides must be 8-element aligned");
    dim3 grid(L * B, H_kv);
    dim3 block(128);
    auto stream = at::cuda::getCurrentCUDAStream();

    gather_compact_kv_into_arena_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const void* const*>(flat_k_ptrs.data_ptr<int64_t>()),
        reinterpret_cast<const void* const*>(flat_v_ptrs.data_ptr<int64_t>()),
        reinterpret_cast<void* const*>(compact_k_ptrs.data_ptr<int64_t>()),
        reinterpret_cast<void* const*>(compact_v_ptrs.data_ptr<int64_t>()),
        reinterpret_cast<int32_t* const*>(compact_pos_ptrs.data_ptr<int64_t>()),
        reinterpret_cast<const int32_t* const*>(block_table_ptrs.data_ptr<int64_t>()),
        row_tensor.data_ptr<int32_t>(),
        slot_tensor.data_ptr<int32_t>(),
        selected_indices.data_ptr<int32_t>(),
        sink_len.data_ptr<int32_t>(),
        persist_len.data_ptr<int32_t>(),
        L, B, H_kv, static_cast<int>(head_dim), k_persist,
        static_cast<int>(page_size),
        static_cast<int>(stride_tokens),
        static_cast<int>(block_table_cols),
        static_cast<int>(compact_pos_per_layer_tokens),
        flat_k_stride0, flat_k_stride1, flat_k_stride2,
        flat_v_stride0, flat_v_stride1, flat_v_stride2,
        compact_k_stride0, compact_k_stride1, compact_k_stride2,
        compact_v_stride0, compact_v_stride1, compact_v_stride2,
        compact_pos_stride_head, compact_pos_stride_tok,
        block_table_stride0, block_table_stride1);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


void gather_compact_kv_into_arena_ptrs_tiled(
    torch::Tensor flat_k_ptrs,
    torch::Tensor flat_v_ptrs,
    torch::Tensor compact_k_ptrs,
    torch::Tensor compact_v_ptrs,
    torch::Tensor compact_pos_ptrs,
    torch::Tensor block_table_ptrs,
    torch::Tensor row_tensor,
    torch::Tensor slot_tensor,
    torch::Tensor selected_indices,
    torch::Tensor sink_len,
    torch::Tensor persist_len,
    int64_t page_size,
    int64_t stride_tokens,
    int64_t block_table_cols,
    int64_t head_dim,
    int64_t compact_pos_per_layer_tokens,
    int64_t flat_k_stride0,
    int64_t flat_k_stride1,
    int64_t flat_k_stride2,
    int64_t flat_v_stride0,
    int64_t flat_v_stride1,
    int64_t flat_v_stride2,
    int64_t compact_k_stride0,
    int64_t compact_k_stride1,
    int64_t compact_k_stride2,
    int64_t compact_v_stride0,
    int64_t compact_v_stride1,
    int64_t compact_v_stride2,
    int64_t compact_pos_stride_head,
    int64_t compact_pos_stride_tok,
    int64_t block_table_stride0,
    int64_t block_table_stride1,
    int64_t tile_tokens,
    int64_t max_total_tokens) {
    TORCH_CHECK(flat_k_ptrs.dim() == 1, "flat_k_ptrs must be [L]");
    int L = static_cast<int>(flat_k_ptrs.size(0));
    TORCH_CHECK(L > 0, "flat_k_ptrs must be non-empty");
    TORCH_CHECK(flat_v_ptrs.dim() == 1 && compact_k_ptrs.dim() == 1
                && compact_v_ptrs.dim() == 1 && compact_pos_ptrs.dim() == 1
                && block_table_ptrs.dim() == 1,
                "pointer tensors must be rank-1");
    TORCH_CHECK(flat_v_ptrs.size(0) == L && compact_k_ptrs.size(0) == L
                && compact_v_ptrs.size(0) == L && compact_pos_ptrs.size(0) == L
                && block_table_ptrs.size(0) == L,
                "all pointer tensors must have same length");
    TORCH_CHECK(flat_k_ptrs.dtype() == torch::kInt64 && flat_v_ptrs.dtype() == torch::kInt64
                && compact_k_ptrs.dtype() == torch::kInt64 && compact_v_ptrs.dtype() == torch::kInt64
                && compact_pos_ptrs.dtype() == torch::kInt64 && block_table_ptrs.dtype() == torch::kInt64,
                "pointer tensors must be int64");
    TORCH_CHECK(flat_k_ptrs.is_cuda() && flat_v_ptrs.is_cuda() && compact_k_ptrs.is_cuda()
                && compact_v_ptrs.is_cuda() && compact_pos_ptrs.is_cuda() && block_table_ptrs.is_cuda(),
                "pointer tensors must be CUDA tensors");
    TORCH_CHECK(flat_k_ptrs.is_contiguous() && flat_v_ptrs.is_contiguous()
                && compact_k_ptrs.is_contiguous() && compact_v_ptrs.is_contiguous()
                && compact_pos_ptrs.is_contiguous() && block_table_ptrs.is_contiguous(),
                "pointer tensors must be contiguous");
    TORCH_CHECK(selected_indices.dim() == 4, "selected_indices must be [L, B, H_kv, k]");
    TORCH_CHECK(selected_indices.size(0) == L, "selected_indices[0] must equal number of layers");
    TORCH_CHECK(selected_indices.dtype() == torch::kInt32, "selected_indices must be int32");
    TORCH_CHECK(selected_indices.is_cuda() && selected_indices.is_contiguous(),
                "selected_indices must be a contiguous CUDA tensor");
    TORCH_CHECK(sink_len.dim() == 2 && persist_len.dim() == 2, "sink/persist_len must be [L, B]");
    TORCH_CHECK(sink_len.dtype() == torch::kInt32 && persist_len.dtype() == torch::kInt32,
                "sink/persist_len must be int32");
    TORCH_CHECK(sink_len.is_cuda() && persist_len.is_cuda()
                && sink_len.is_contiguous() && persist_len.is_contiguous(),
                "sink/persist_len must be contiguous CUDA tensors");
    TORCH_CHECK(row_tensor.dim() == 2 && row_tensor.size(0) == L,
                "row_tensor must be [L, B]");
    TORCH_CHECK(row_tensor.dtype() == torch::kInt32 && row_tensor.is_cuda()
                && row_tensor.is_contiguous(),
                "row_tensor must be a contiguous CUDA int32 tensor");
    TORCH_CHECK(slot_tensor.dim() == 1, "slot_tensor must be [B]");
    TORCH_CHECK(slot_tensor.dtype() == torch::kInt32 && slot_tensor.is_cuda()
                && slot_tensor.is_contiguous(),
                "slot_tensor must be a contiguous CUDA int32 tensor");

    int B = static_cast<int>(selected_indices.size(1));
    int H_kv = static_cast<int>(selected_indices.size(2));
    int k_persist = static_cast<int>(selected_indices.size(3));
    TORCH_CHECK(row_tensor.size(1) == B, "row_tensor must be [L, B]");
    TORCH_CHECK(slot_tensor.size(0) == B, "slot_tensor must be [B]");
    TORCH_CHECK(sink_len.size(0) == L && sink_len.size(1) == B,
                "sink_len must be [L, B]");
    TORCH_CHECK(persist_len.size(0) == L && persist_len.size(1) == B,
                "persist_len must be [L, B]");

    const int requested_tile_tokens = std::max<int>(1, static_cast<int>(tile_tokens));
    const int stride_tokens_i32 = std::max<int>(1, static_cast<int>(stride_tokens));
    const int active_tokens_i32 = std::min<int>(
        stride_tokens_i32,
        std::max<int>(1, static_cast<int>(max_total_tokens)));
    const int max_grid_z = 65535;
    const int min_tile_for_grid_z = (active_tokens_i32 + max_grid_z - 1) / max_grid_z;
    const int safe_tile_tokens = std::max<int>(requested_tile_tokens, min_tile_for_grid_z);
    const int token_tiles = std::max<int>(
        1,
        (active_tokens_i32 + safe_tile_tokens - 1) / safe_tile_tokens);
    // [GATHER-VEC8] 向量化合同(违约 fail-fast,不降级):
    TORCH_CHECK(head_dim % 8 == 0, "gather vec8: head_dim must be a multiple of 8");
    TORCH_CHECK(flat_k_stride2 == 1 && flat_v_stride2 == 1 &&
                compact_k_stride2 == 1 && compact_v_stride2 == 1,
                "gather vec8: last-dim strides must be 1 (contiguous head_dim)");
    TORCH_CHECK(flat_k_stride0 % 8 == 0 && flat_k_stride1 % 8 == 0 &&
                flat_v_stride0 % 8 == 0 && flat_v_stride1 % 8 == 0 &&
                compact_k_stride0 % 8 == 0 && compact_k_stride1 % 8 == 0 &&
                compact_v_stride0 % 8 == 0 && compact_v_stride1 % 8 == 0,
                "gather vec8: token/head strides must be 8-element aligned");
    dim3 grid(L * B, H_kv, token_tiles);
    dim3 block(128);
    auto stream = at::cuda::getCurrentCUDAStream();

    gather_compact_kv_into_arena_tiled_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const void* const*>(flat_k_ptrs.data_ptr<int64_t>()),
        reinterpret_cast<const void* const*>(flat_v_ptrs.data_ptr<int64_t>()),
        reinterpret_cast<void* const*>(compact_k_ptrs.data_ptr<int64_t>()),
        reinterpret_cast<void* const*>(compact_v_ptrs.data_ptr<int64_t>()),
        reinterpret_cast<int32_t* const*>(compact_pos_ptrs.data_ptr<int64_t>()),
        reinterpret_cast<const int32_t* const*>(block_table_ptrs.data_ptr<int64_t>()),
        row_tensor.data_ptr<int32_t>(),
        slot_tensor.data_ptr<int32_t>(),
        selected_indices.data_ptr<int32_t>(),
        sink_len.data_ptr<int32_t>(),
        persist_len.data_ptr<int32_t>(),
        L, B, H_kv, static_cast<int>(head_dim), k_persist,
        static_cast<int>(page_size),
        static_cast<int>(stride_tokens),
        static_cast<int>(block_table_cols),
        static_cast<int>(compact_pos_per_layer_tokens),
        flat_k_stride0, flat_k_stride1, flat_k_stride2,
        flat_v_stride0, flat_v_stride1, flat_v_stride2,
        compact_k_stride0, compact_k_stride1, compact_k_stride2,
        compact_v_stride0, compact_v_stride1, compact_v_stride2,
        compact_pos_stride_head, compact_pos_stride_tok,
        block_table_stride0, block_table_stride1,
        safe_tile_tokens);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gather_compact_kv_into_arena_ptrs_tiled_autolen_impl(
    torch::Tensor flat_k_ptrs,
    torch::Tensor flat_v_ptrs,
    torch::Tensor compact_k_ptrs,
    torch::Tensor compact_v_ptrs,
    torch::Tensor compact_pos_ptrs,
    torch::Tensor block_table_ptrs,
    torch::Tensor row_tensor,
    torch::Tensor slot_tensor,
    torch::Tensor selected_indices,
    torch::Tensor seq_lens,
    int64_t page_size,
    int64_t stride_tokens,
    int64_t block_table_cols,
    int64_t head_dim,
    int64_t compact_pos_per_layer_tokens,
    int64_t flat_k_stride0,
    int64_t flat_k_stride1,
    int64_t flat_k_stride2,
    int64_t flat_v_stride0,
    int64_t flat_v_stride1,
    int64_t flat_v_stride2,
    int64_t compact_k_stride0,
    int64_t compact_k_stride1,
    int64_t compact_k_stride2,
    int64_t compact_v_stride0,
    int64_t compact_v_stride1,
    int64_t compact_v_stride2,
    int64_t compact_pos_stride_head,
    int64_t compact_pos_stride_tok,
    int64_t block_table_stride0,
    int64_t block_table_stride1,
    int64_t tile_tokens,
    int64_t max_total_tokens,
    int64_t sink_cap,
    int64_t recent_cfg,
    int64_t k_head,
    int64_t threshold_tokens,
    bool skip_unchanged) {
    TORCH_CHECK(flat_k_ptrs.dim() == 1, "flat_k_ptrs must be [L]");
    int L = static_cast<int>(flat_k_ptrs.size(0));
    TORCH_CHECK(L > 0, "flat_k_ptrs must be non-empty");
    TORCH_CHECK(flat_v_ptrs.dim() == 1 && compact_k_ptrs.dim() == 1
                && compact_v_ptrs.dim() == 1 && compact_pos_ptrs.dim() == 1
                && block_table_ptrs.dim() == 1,
                "pointer tensors must be rank-1");
    TORCH_CHECK(flat_v_ptrs.size(0) == L && compact_k_ptrs.size(0) == L
                && compact_v_ptrs.size(0) == L && compact_pos_ptrs.size(0) == L
                && block_table_ptrs.size(0) == L,
                "all pointer tensors must have same length");
    TORCH_CHECK(flat_k_ptrs.dtype() == torch::kInt64 && flat_v_ptrs.dtype() == torch::kInt64
                && compact_k_ptrs.dtype() == torch::kInt64 && compact_v_ptrs.dtype() == torch::kInt64
                && compact_pos_ptrs.dtype() == torch::kInt64 && block_table_ptrs.dtype() == torch::kInt64,
                "pointer tensors must be int64");
    TORCH_CHECK(flat_k_ptrs.is_cuda() && flat_v_ptrs.is_cuda() && compact_k_ptrs.is_cuda()
                && compact_v_ptrs.is_cuda() && compact_pos_ptrs.is_cuda() && block_table_ptrs.is_cuda(),
                "pointer tensors must be CUDA tensors");
    TORCH_CHECK(flat_k_ptrs.is_contiguous() && flat_v_ptrs.is_contiguous()
                && compact_k_ptrs.is_contiguous() && compact_v_ptrs.is_contiguous()
                && compact_pos_ptrs.is_contiguous() && block_table_ptrs.is_contiguous(),
                "pointer tensors must be contiguous");
    TORCH_CHECK(selected_indices.dim() == 4, "selected_indices must be [L, B, H_kv, k]");
    TORCH_CHECK(selected_indices.size(0) == L, "selected_indices[0] must equal number of layers");
    TORCH_CHECK(selected_indices.dtype() == torch::kInt32, "selected_indices must be int32");
    TORCH_CHECK(selected_indices.is_cuda() && selected_indices.is_contiguous(),
                "selected_indices must be a contiguous CUDA tensor");
    TORCH_CHECK(row_tensor.dim() == 2 && row_tensor.size(0) == L,
                "row_tensor must be [L, B]");
    TORCH_CHECK(row_tensor.dtype() == torch::kInt32 && row_tensor.is_cuda()
                && row_tensor.is_contiguous(),
                "row_tensor must be a contiguous CUDA int32 tensor");
    TORCH_CHECK(slot_tensor.dim() == 1, "slot_tensor must be [B]");
    TORCH_CHECK(slot_tensor.dtype() == torch::kInt32 && slot_tensor.is_cuda()
                && slot_tensor.is_contiguous(),
                "slot_tensor must be a contiguous CUDA int32 tensor");
    TORCH_CHECK(seq_lens.dim() == 1, "seq_lens must be [B]");
    TORCH_CHECK(seq_lens.dtype() == torch::kInt32 && seq_lens.is_cuda()
                && seq_lens.is_contiguous(),
                "seq_lens must be a contiguous CUDA int32 tensor");

    int B = static_cast<int>(selected_indices.size(1));
    int H_kv = static_cast<int>(selected_indices.size(2));
    int k_persist = static_cast<int>(selected_indices.size(3));
    TORCH_CHECK(row_tensor.size(1) == B, "row_tensor must be [L, B]");
    TORCH_CHECK(slot_tensor.size(0) == B, "slot_tensor must be [B]");
    TORCH_CHECK(seq_lens.size(0) == B, "seq_lens must be [B]");

    const int requested_tile_tokens = std::max<int>(1, static_cast<int>(tile_tokens));
    const int stride_tokens_i32 = std::max<int>(1, static_cast<int>(stride_tokens));
    const int active_tokens_i32 = std::min<int>(
        stride_tokens_i32,
        std::max<int>(1, static_cast<int>(max_total_tokens)));
    const int max_grid_z = 65535;
    const int min_tile_for_grid_z = (active_tokens_i32 + max_grid_z - 1) / max_grid_z;
    const int safe_tile_tokens = std::max<int>(requested_tile_tokens, min_tile_for_grid_z);
    const int token_tiles = std::max<int>(
        1,
        (active_tokens_i32 + safe_tile_tokens - 1) / safe_tile_tokens);
    // [GATHER-VEC8] 向量化合同(违约 fail-fast,不降级):
    TORCH_CHECK(head_dim % 8 == 0, "gather vec8: head_dim must be a multiple of 8");
    TORCH_CHECK(flat_k_stride2 == 1 && flat_v_stride2 == 1 &&
                compact_k_stride2 == 1 && compact_v_stride2 == 1,
                "gather vec8: last-dim strides must be 1 (contiguous head_dim)");
    TORCH_CHECK(flat_k_stride0 % 8 == 0 && flat_k_stride1 % 8 == 0 &&
                flat_v_stride0 % 8 == 0 && flat_v_stride1 % 8 == 0 &&
                compact_k_stride0 % 8 == 0 && compact_k_stride1 % 8 == 0 &&
                compact_v_stride0 % 8 == 0 && compact_v_stride1 % 8 == 0,
                "gather vec8: token/head strides must be 8-element aligned");
    dim3 grid(L * B, H_kv, token_tiles);
    dim3 block(128);
    auto stream = at::cuda::getCurrentCUDAStream();

    gather_compact_kv_into_arena_tiled_autolen_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const void* const*>(flat_k_ptrs.data_ptr<int64_t>()),
        reinterpret_cast<const void* const*>(flat_v_ptrs.data_ptr<int64_t>()),
        reinterpret_cast<void* const*>(compact_k_ptrs.data_ptr<int64_t>()),
        reinterpret_cast<void* const*>(compact_v_ptrs.data_ptr<int64_t>()),
        reinterpret_cast<int32_t* const*>(compact_pos_ptrs.data_ptr<int64_t>()),
        reinterpret_cast<const int32_t* const*>(block_table_ptrs.data_ptr<int64_t>()),
        row_tensor.data_ptr<int32_t>(),
        slot_tensor.data_ptr<int32_t>(),
        selected_indices.data_ptr<int32_t>(),
        seq_lens.data_ptr<int32_t>(),
        L, B, H_kv, static_cast<int>(head_dim), k_persist,
        static_cast<int>(page_size),
        static_cast<int>(stride_tokens),
        static_cast<int>(block_table_cols),
        static_cast<int>(compact_pos_per_layer_tokens),
        static_cast<int>(sink_cap),
        static_cast<int>(recent_cfg),
        static_cast<int>(k_head),
        static_cast<int>(threshold_tokens),
        flat_k_stride0, flat_k_stride1, flat_k_stride2,
        flat_v_stride0, flat_v_stride1, flat_v_stride2,
        compact_k_stride0, compact_k_stride1, compact_k_stride2,
        compact_v_stride0, compact_v_stride1, compact_v_stride2,
        compact_pos_stride_head, compact_pos_stride_tok,
        block_table_stride0, block_table_stride1,
        safe_tile_tokens,
        skip_unchanged);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gather_compact_kv_into_arena_ptrs_tiled_autolen(
    torch::Tensor flat_k_ptrs,
    torch::Tensor flat_v_ptrs,
    torch::Tensor compact_k_ptrs,
    torch::Tensor compact_v_ptrs,
    torch::Tensor compact_pos_ptrs,
    torch::Tensor block_table_ptrs,
    torch::Tensor row_tensor,
    torch::Tensor slot_tensor,
    torch::Tensor selected_indices,
    torch::Tensor seq_lens,
    int64_t page_size,
    int64_t stride_tokens,
    int64_t block_table_cols,
    int64_t head_dim,
    int64_t compact_pos_per_layer_tokens,
    int64_t flat_k_stride0,
    int64_t flat_k_stride1,
    int64_t flat_k_stride2,
    int64_t flat_v_stride0,
    int64_t flat_v_stride1,
    int64_t flat_v_stride2,
    int64_t compact_k_stride0,
    int64_t compact_k_stride1,
    int64_t compact_k_stride2,
    int64_t compact_v_stride0,
    int64_t compact_v_stride1,
    int64_t compact_v_stride2,
    int64_t compact_pos_stride_head,
    int64_t compact_pos_stride_tok,
    int64_t block_table_stride0,
    int64_t block_table_stride1,
    int64_t tile_tokens,
    int64_t max_total_tokens,
    int64_t sink_cap,
    int64_t recent_cfg,
    int64_t k_head,
    int64_t threshold_tokens) {
    gather_compact_kv_into_arena_ptrs_tiled_autolen_impl(
        flat_k_ptrs,
        flat_v_ptrs,
        compact_k_ptrs,
        compact_v_ptrs,
        compact_pos_ptrs,
        block_table_ptrs,
        row_tensor,
        slot_tensor,
        selected_indices,
        seq_lens,
        page_size,
        stride_tokens,
        block_table_cols,
        head_dim,
        compact_pos_per_layer_tokens,
        flat_k_stride0,
        flat_k_stride1,
        flat_k_stride2,
        flat_v_stride0,
        flat_v_stride1,
        flat_v_stride2,
        compact_k_stride0,
        compact_k_stride1,
        compact_k_stride2,
        compact_v_stride0,
        compact_v_stride1,
        compact_v_stride2,
        compact_pos_stride_head,
        compact_pos_stride_tok,
        block_table_stride0,
        block_table_stride1,
        tile_tokens,
        max_total_tokens,
        sink_cap,
        recent_cfg,
        k_head,
        threshold_tokens,
        false);
}

void gather_compact_kv_into_arena_ptrs_tiled_autolen_skip_unchanged(
    torch::Tensor flat_k_ptrs,
    torch::Tensor flat_v_ptrs,
    torch::Tensor compact_k_ptrs,
    torch::Tensor compact_v_ptrs,
    torch::Tensor compact_pos_ptrs,
    torch::Tensor block_table_ptrs,
    torch::Tensor row_tensor,
    torch::Tensor slot_tensor,
    torch::Tensor selected_indices,
    torch::Tensor seq_lens,
    int64_t page_size,
    int64_t stride_tokens,
    int64_t block_table_cols,
    int64_t head_dim,
    int64_t compact_pos_per_layer_tokens,
    int64_t flat_k_stride0,
    int64_t flat_k_stride1,
    int64_t flat_k_stride2,
    int64_t flat_v_stride0,
    int64_t flat_v_stride1,
    int64_t flat_v_stride2,
    int64_t compact_k_stride0,
    int64_t compact_k_stride1,
    int64_t compact_k_stride2,
    int64_t compact_v_stride0,
    int64_t compact_v_stride1,
    int64_t compact_v_stride2,
    int64_t compact_pos_stride_head,
    int64_t compact_pos_stride_tok,
    int64_t block_table_stride0,
    int64_t block_table_stride1,
    int64_t tile_tokens,
    int64_t max_total_tokens,
    int64_t sink_cap,
    int64_t recent_cfg,
    int64_t k_head,
    int64_t threshold_tokens) {
    gather_compact_kv_into_arena_ptrs_tiled_autolen_impl(
        flat_k_ptrs,
        flat_v_ptrs,
        compact_k_ptrs,
        compact_v_ptrs,
        compact_pos_ptrs,
        block_table_ptrs,
        row_tensor,
        slot_tensor,
        selected_indices,
        seq_lens,
        page_size,
        stride_tokens,
        block_table_cols,
        head_dim,
        compact_pos_per_layer_tokens,
        flat_k_stride0,
        flat_k_stride1,
        flat_k_stride2,
        flat_v_stride0,
        flat_v_stride1,
        flat_v_stride2,
        compact_k_stride0,
        compact_k_stride1,
        compact_k_stride2,
        compact_v_stride0,
        compact_v_stride1,
        compact_v_stride2,
        compact_pos_stride_head,
        compact_pos_stride_tok,
        block_table_stride0,
        block_table_stride1,
        tile_tokens,
        max_total_tokens,
        sink_cap,
        recent_cfg,
        k_head,
        threshold_tokens,
        true);
}


void gather_compact_kv_into_arena(
    std::vector<torch::Tensor> flat_k,
    std::vector<torch::Tensor> flat_v,
    std::vector<torch::Tensor> compact_k,
    std::vector<torch::Tensor> compact_v,
    std::vector<torch::Tensor> compact_pos,
    std::vector<torch::Tensor> block_tables,
    torch::Tensor row_tensor,
    torch::Tensor slot_tensor,
    torch::Tensor selected_indices,
    torch::Tensor sink_len,
    torch::Tensor persist_len,
    int64_t page_size,
    int64_t stride_tokens) {
    TORCH_CHECK(!flat_k.empty(), "flat_k must be non-empty");
    int L = static_cast<int>(flat_k.size());
    TORCH_CHECK((int)flat_v.size() == L && (int)compact_k.size() == L
                && (int)compact_v.size() == L && (int)compact_pos.size() == L
                && (int)block_tables.size() == L,
                "all per-layer vectors must have same length");
    TORCH_CHECK(selected_indices.dim() == 4, "selected_indices must be [L, B, H_kv, k]");
    TORCH_CHECK(selected_indices.size(0) == L, "selected_indices[0] must equal number of layers");
    TORCH_CHECK(selected_indices.dtype() == torch::kInt32, "selected_indices must be int32");
    TORCH_CHECK(sink_len.dim() == 2 && persist_len.dim() == 2, "sink/persist_len must be [L, B]");
    TORCH_CHECK(sink_len.dtype() == torch::kInt32 && persist_len.dtype() == torch::kInt32,
                "sink/persist_len must be int32");
    TORCH_CHECK(row_tensor.dim() == 2 && row_tensor.size(0) == L,
                "row_tensor must be [L, B]");
    TORCH_CHECK(slot_tensor.dim() == 1, "slot_tensor must be [B]");

    int B = static_cast<int>(selected_indices.size(1));
    int H_kv = static_cast<int>(selected_indices.size(2));
    int k_persist = static_cast<int>(selected_indices.size(3));
    int head_dim = static_cast<int>(flat_k[0].size(-1));
    int block_table_cols = static_cast<int>(block_tables[0].size(1));

    TORCH_CHECK(row_tensor.size(1) == B, "row_tensor must be [L, B]");
    TORCH_CHECK(slot_tensor.size(0) == B, "slot_tensor must be [B]");
    TORCH_CHECK(sink_len.size(0) == L && sink_len.size(1) == B,
                "sink_len must be [L, B]");
    TORCH_CHECK(persist_len.size(0) == L && persist_len.size(1) == B,
                "persist_len must be [L, B]");

    // Cast row/slot to int32 contiguous views (Python caller uses torch.long for slot_tensor).
    auto row_i32 = row_tensor.to(torch::kInt32).contiguous();
    auto slot_i32 = slot_tensor.to(torch::kInt32).contiguous();
    auto sink_c = sink_len.contiguous();
    auto persist_c = persist_len.contiguous();
    auto selected_c = selected_indices.contiguous();

    // Strides (shared across layers — Triton relies on same assumption).
    auto flat_k0 = flat_k[0];
    auto flat_v0 = flat_v[0];
    auto compact_k0 = compact_k[0];
    auto compact_v0 = compact_v[0];
    auto compact_pos0 = compact_pos[0];
    auto block_table0 = block_tables[0];

    int64_t flat_k_stride0 = flat_k0.stride(0);
    int64_t flat_k_stride1 = flat_k0.stride(1);
    int64_t flat_k_stride2 = flat_k0.stride(2);
    int64_t flat_v_stride0 = flat_v0.stride(0);
    int64_t flat_v_stride1 = flat_v0.stride(1);
    int64_t flat_v_stride2 = flat_v0.stride(2);
    int64_t compact_k_stride0 = compact_k0.stride(0);
    int64_t compact_k_stride1 = compact_k0.stride(1);
    int64_t compact_k_stride2 = compact_k0.stride(2);
    int64_t compact_v_stride0 = compact_v0.stride(0);
    int64_t compact_v_stride1 = compact_v0.stride(1);
    int64_t compact_v_stride2 = compact_v0.stride(2);
    int64_t compact_pos_stride_head = compact_pos0.stride(0);
    int64_t compact_pos_stride_tok = compact_pos0.stride(1);
    int64_t block_table_stride0 = block_table0.stride(0);
    int64_t block_table_stride1 = block_table0.stride(1);
    int compact_pos_per_layer_tokens = static_cast<int>(compact_pos0.size(1));

    // Pack per-layer pointers into a CPU int64 buffer, move to GPU.
    std::vector<int64_t> h_flat_k(L), h_flat_v(L), h_compact_k(L), h_compact_v(L),
        h_compact_pos(L), h_btable(L);
    for (int i = 0; i < L; ++i) {
        h_flat_k[i] = reinterpret_cast<int64_t>(flat_k[i].data_ptr());
        h_flat_v[i] = reinterpret_cast<int64_t>(flat_v[i].data_ptr());
        h_compact_k[i] = reinterpret_cast<int64_t>(compact_k[i].data_ptr());
        h_compact_v[i] = reinterpret_cast<int64_t>(compact_v[i].data_ptr());
        h_compact_pos[i] = reinterpret_cast<int64_t>(compact_pos[i].data_ptr<int32_t>());
        h_btable[i] = reinterpret_cast<int64_t>(block_tables[i].data_ptr<int32_t>());
    }
    auto opts = torch::TensorOptions().device(selected_c.device()).dtype(torch::kInt64);
    auto to_gpu_ptr_tensor = [&](const std::vector<int64_t>& h) {
        auto cpu_t = torch::from_blob((void*)h.data(), {(int64_t)L},
            torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
        return cpu_t.to(opts);
    };
    auto d_flat_k = to_gpu_ptr_tensor(h_flat_k);
    auto d_flat_v = to_gpu_ptr_tensor(h_flat_v);
    auto d_compact_k = to_gpu_ptr_tensor(h_compact_k);
    auto d_compact_v = to_gpu_ptr_tensor(h_compact_v);
    auto d_compact_pos = to_gpu_ptr_tensor(h_compact_pos);
    auto d_btable = to_gpu_ptr_tensor(h_btable);

    gather_compact_kv_into_arena_ptrs(
        d_flat_k,
        d_flat_v,
        d_compact_k,
        d_compact_v,
        d_compact_pos,
        d_btable,
        row_i32,
        slot_i32,
        selected_c,
        sink_c,
        persist_c,
        page_size,
        stride_tokens,
        block_table_cols,
        head_dim,
        compact_pos_per_layer_tokens,
        flat_k_stride0, flat_k_stride1, flat_k_stride2,
        flat_v_stride0, flat_v_stride1, flat_v_stride2,
        compact_k_stride0, compact_k_stride1, compact_k_stride2,
        compact_v_stride0, compact_v_stride1, compact_v_stride2,
        compact_pos_stride_head, compact_pos_stride_tok,
        block_table_stride0, block_table_stride1);
}
"""

    _ensure_torch_cuda_arch_list()
    try:
        _MODULE = load_inline(
            name="fa_sparse_runtime_ext",
            cpp_sources=cpp_source,
            cuda_sources=cuda_source,
            functions=None,
            extra_cuda_cflags=["-lineinfo"],
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
            raise FASparseRuntimeExtUnavailable("fa_sparse_runtime_ext unavailable") from _LOAD_ERROR
        raise FASparseRuntimeExtUnavailable(
            "fa_sparse_runtime_ext unavailable; set VLLM_SPARSE_FA_RUNTIME_EXT_CUDA=1"
        )
    return mod


def _refresh_static_materialize_python(
    *,
    final_page_table_i32: torch.Tensor,
    selected_middle_pages_by_row_i32: torch.Tensor | None = None,
    selected_middle_counts_by_row_i32: torch.Tensor | None = None,
    selected_middle_pages_by_slot_i32: torch.Tensor | None = None,
    selected_middle_counts_by_slot_i32: torch.Tensor | None = None,
    request_slot_rows_i32: torch.Tensor | None = None,
    request_block_table_i32: torch.Tensor,
    sink_page_slots: int,
    request_refresh_generation_i32: torch.Tensor,
) -> dict[str, torch.Tensor]:
    _require_rank("final_page_table_i32", final_page_table_i32, 2)
    _require_rank("request_block_table_i32", request_block_table_i32, 2)
    _require_rank("request_refresh_generation_i32", request_refresh_generation_i32, 1)
    use_slot_major = selected_middle_pages_by_slot_i32 is not None
    if use_slot_major:
        if (
            selected_middle_counts_by_slot_i32 is None
            or request_slot_rows_i32 is None
            or selected_middle_pages_by_row_i32 is not None
            or selected_middle_counts_by_row_i32 is not None
        ):
            raise ValueError("slot-major static materialize requires slot cache tensors only")
        _require_rank("selected_middle_pages_by_slot_i32", selected_middle_pages_by_slot_i32, 3)
        _require_rank("selected_middle_counts_by_slot_i32", selected_middle_counts_by_slot_i32, 2)
        _require_rank("request_slot_rows_i32", request_slot_rows_i32, 1)
        selected_middle_pages = selected_middle_pages_by_slot_i32
        selected_middle_counts = selected_middle_counts_by_slot_i32
        source_rows = request_slot_rows_i32.to(dtype=torch.long, device=final_page_table_i32.device)
        batch_size = int(request_slot_rows_i32.shape[0])
    else:
        if (
            selected_middle_pages_by_row_i32 is None
            or selected_middle_counts_by_row_i32 is None
            or request_slot_rows_i32 is not None
            or selected_middle_pages_by_slot_i32 is not None
            or selected_middle_counts_by_slot_i32 is not None
        ):
            raise ValueError("row-major static materialize requires row tensors only")
        _require_rank("selected_middle_pages_by_row_i32", selected_middle_pages_by_row_i32, 3)
        _require_rank("selected_middle_counts_by_row_i32", selected_middle_counts_by_row_i32, 2)
        selected_middle_pages = selected_middle_pages_by_row_i32
        selected_middle_counts = selected_middle_counts_by_row_i32
        batch_size = int(selected_middle_pages_by_row_i32.shape[0])
        source_rows = torch.arange(batch_size, dtype=torch.long, device=final_page_table_i32.device)

    _, num_kv_heads, _ = selected_middle_pages.shape
    materialize_status_i32 = torch.zeros(
        (int(batch_size),),
        device=final_page_table_i32.device,
        dtype=torch.int32,
    )
    applied_refresh_generation_i32 = torch.zeros_like(
        request_refresh_generation_i32,
        device=final_page_table_i32.device,
        dtype=torch.int32,
    )
    sink_slots = int(sink_page_slots)

    for row in range(int(batch_size)):
        source_row = int(source_rows[row].item())
        if source_row < 0 or source_row >= int(selected_middle_pages.shape[0]):
            materialize_status_i32[row] = int(PAGE_SPARSE_STATUS_RECOVERABLE_MISS)
            continue
        middle_counts_row = selected_middle_counts[source_row].reshape(-1)
        if middle_counts_row.numel() != int(num_kv_heads):
            raise ValueError("selected_middle_counts_by_row_i32 must match num_kv_heads")
        middle_count = int(middle_counts_row[0].item()) if middle_counts_row.numel() > 0 else 0
        if middle_count <= 0:
            materialize_status_i32[row] = int(PAGE_SPARSE_STATUS_RECOVERABLE_MISS)
            continue
        block_row = request_block_table_i32[row]
        row_slice = slice(int(row) * int(num_kv_heads), (int(row) + 1) * int(num_kv_heads))

        if sink_slots > 0:
            sink_pages = block_row[:sink_slots].reshape(1, sink_slots).expand(int(num_kv_heads), sink_slots)
        else:
            sink_pages = torch.empty(
                (int(num_kv_heads), 0),
                device=final_page_table_i32.device,
                dtype=torch.int32,
            )
        middle_idx = selected_middle_pages[source_row, :, :middle_count].to(dtype=torch.long)
        middle_pages = torch.gather(
            block_row.reshape(1, -1).expand(int(num_kv_heads), -1),
            1,
            middle_idx,
        )
        static_pages = torch.cat((sink_pages, middle_pages.to(dtype=torch.int32)), dim=1)
        final_page_table_i32[row_slice, : static_pages.shape[1]].copy_(static_pages)
        materialize_status_i32[row] = int(PAGE_SPARSE_STATUS_OK)
        applied_refresh_generation_i32[row] = int(request_refresh_generation_i32[row].item())

    return {
        "materialize_status_i32": materialize_status_i32,
        "applied_refresh_generation_i32": applied_refresh_generation_i32,
    }


def _step_recent_patch_python(
    *,
    final_page_table_i32: torch.Tensor,
    request_block_table_i32: torch.Tensor,
    request_recent_first_logical_page_i32: torch.Tensor,
    request_recent_page_count_i32: torch.Tensor,
    request_recent_epoch_i32: torch.Tensor,
    selected_page_count_i32: torch.Tensor,
    sink_page_slots: int,
    num_kv_heads: int,
) -> dict[str, torch.Tensor]:
    _require_rank("final_page_table_i32", final_page_table_i32, 2)
    _require_rank("request_block_table_i32", request_block_table_i32, 2)
    _require_rank("request_recent_first_logical_page_i32", request_recent_first_logical_page_i32, 1)
    _require_rank("request_recent_page_count_i32", request_recent_page_count_i32, 1)
    _require_rank("request_recent_epoch_i32", request_recent_epoch_i32, 1)
    _require_rank("selected_page_count_i32", selected_page_count_i32, 1)

    batch_size = int(request_recent_epoch_i32.shape[0])
    patch_status_i32 = torch.zeros(
        (batch_size,),
        device=final_page_table_i32.device,
        dtype=torch.int32,
    )
    applied_recent_epoch_i32 = torch.zeros_like(
        request_recent_epoch_i32,
        device=final_page_table_i32.device,
        dtype=torch.int32,
    )
    sink_slots = int(sink_page_slots)
    max_page_count = int(final_page_table_i32.shape[1])

    for row in range(batch_size):
        recent_count = int(request_recent_page_count_i32[row].item())
        total_page_count = int(selected_page_count_i32[row].item())
        if recent_count <= 0 or total_page_count <= 0:
            patch_status_i32[row] = int(PAGE_SPARSE_STATUS_RECOVERABLE_MISS)
            continue
        middle_count = int(total_page_count) - sink_slots - int(recent_count)
        if middle_count < 0:
            raise ValueError("selected_page_count_i32 must cover sink + middle + recent")
        recent_first = int(request_recent_first_logical_page_i32[row].item())
        block_row = request_block_table_i32[row]
        recent_pages = block_row[
            int(recent_first): int(recent_first) + int(recent_count)
        ].reshape(1, int(recent_count)).expand(int(num_kv_heads), int(recent_count))
        row_slice = slice(int(row) * int(num_kv_heads), (int(row) + 1) * int(num_kv_heads))
        suffix_start = sink_slots + middle_count
        final_page_table_i32[row_slice, suffix_start : suffix_start + int(recent_count)].copy_(
            recent_pages.to(dtype=torch.int32)
        )
        if total_page_count < max_page_count:
            final_page_table_i32[row_slice, total_page_count:].fill_(
                int(recent_pages[0, -1].item())
            )
        patch_status_i32[row] = int(PAGE_SPARSE_STATUS_OK)
        applied_recent_epoch_i32[row] = int(request_recent_epoch_i32[row].item())

    return {
        "patch_status_i32": patch_status_i32,
        "applied_recent_epoch_i32": applied_recent_epoch_i32,
    }


def refresh_static_materialize_cuda(
    *,
    final_page_table_i32: torch.Tensor,
    selected_middle_pages_by_row_i32: torch.Tensor | None = None,
    selected_middle_counts_by_row_i32: torch.Tensor | None = None,
    selected_middle_pages_by_slot_i32: torch.Tensor | None = None,
    selected_middle_counts_by_slot_i32: torch.Tensor | None = None,
    request_slot_rows_i32: torch.Tensor | None = None,
    request_block_table_i32: torch.Tensor,
    sink_page_slots: int,
    request_refresh_generation_i32: torch.Tensor,
    _force_python: bool = False,
) -> dict[str, torch.Tensor]:
    _require_output_tensor("final_page_table_i32", final_page_table_i32)
    if _force_python or not final_page_table_i32.is_cuda:
        return _refresh_static_materialize_python(
            final_page_table_i32=final_page_table_i32,
            selected_middle_pages_by_row_i32=selected_middle_pages_by_row_i32,
            selected_middle_counts_by_row_i32=selected_middle_counts_by_row_i32,
            selected_middle_pages_by_slot_i32=selected_middle_pages_by_slot_i32,
            selected_middle_counts_by_slot_i32=selected_middle_counts_by_slot_i32,
            request_slot_rows_i32=request_slot_rows_i32,
            request_block_table_i32=request_block_table_i32,
            sink_page_slots=sink_page_slots,
            request_refresh_generation_i32=request_refresh_generation_i32,
        )

    mod = _require_ext(force=True)
    use_slot_major = selected_middle_pages_by_slot_i32 is not None
    if use_slot_major:
        if (
            selected_middle_counts_by_slot_i32 is None
            or request_slot_rows_i32 is None
            or selected_middle_pages_by_row_i32 is not None
            or selected_middle_counts_by_row_i32 is not None
        ):
            raise ValueError("slot-major static materialize requires slot cache tensors only")
        selected_middle_pages_i32 = selected_middle_pages_by_slot_i32.to(
            device=final_page_table_i32.device,
            dtype=torch.int32,
        ).contiguous()
        selected_middle_counts_i32 = selected_middle_counts_by_slot_i32.to(
            device=final_page_table_i32.device,
            dtype=torch.int32,
        ).contiguous()
        request_slot_rows_i32 = request_slot_rows_i32.to(
            device=final_page_table_i32.device,
            dtype=torch.int32,
        ).contiguous()
    else:
        if (
            selected_middle_pages_by_row_i32 is None
            or selected_middle_counts_by_row_i32 is None
            or request_slot_rows_i32 is not None
            or selected_middle_pages_by_slot_i32 is not None
            or selected_middle_counts_by_slot_i32 is not None
        ):
            raise ValueError("row-major static materialize requires row tensors only")
        selected_middle_pages_i32 = selected_middle_pages_by_row_i32.to(
            device=final_page_table_i32.device,
            dtype=torch.int32,
        ).contiguous()
        selected_middle_counts_i32 = selected_middle_counts_by_row_i32.to(
            device=final_page_table_i32.device,
            dtype=torch.int32,
        ).contiguous()
        request_slot_rows_i32 = torch.arange(
            int(selected_middle_pages_i32.shape[0]),
            device=final_page_table_i32.device,
            dtype=torch.int32,
        )
    request_block_table_i32 = request_block_table_i32.to(
        device=final_page_table_i32.device,
        dtype=torch.int32,
    ).contiguous()
    request_refresh_generation_i32 = request_refresh_generation_i32.to(
        device=final_page_table_i32.device,
        dtype=torch.int32,
    ).contiguous()

    materialize_status_i32, applied_refresh_generation_i32 = mod.refresh_static_materialize(
        final_page_table_i32,
        selected_middle_pages_i32,
        selected_middle_counts_i32,
        request_slot_rows_i32,
        request_block_table_i32,
        int(sink_page_slots),
        request_refresh_generation_i32,
        bool(use_slot_major),
    )
    return {
        "materialize_status_i32": materialize_status_i32,
        "applied_refresh_generation_i32": applied_refresh_generation_i32,
    }


def step_recent_patch_cuda(
    *,
    final_page_table_i32: torch.Tensor,
    request_block_table_i32: torch.Tensor,
    request_recent_first_logical_page_i32: torch.Tensor,
    request_recent_page_count_i32: torch.Tensor,
    request_recent_epoch_i32: torch.Tensor,
    selected_page_count_i32: torch.Tensor,
    sink_page_slots: int,
    num_kv_heads: int,
    _force_python: bool = False,
) -> dict[str, torch.Tensor]:
    _require_output_tensor("final_page_table_i32", final_page_table_i32)
    if _force_python or not final_page_table_i32.is_cuda:
        return _step_recent_patch_python(
            final_page_table_i32=final_page_table_i32,
            request_block_table_i32=request_block_table_i32,
            request_recent_first_logical_page_i32=request_recent_first_logical_page_i32,
            request_recent_page_count_i32=request_recent_page_count_i32,
            request_recent_epoch_i32=request_recent_epoch_i32,
            selected_page_count_i32=selected_page_count_i32,
            sink_page_slots=sink_page_slots,
            num_kv_heads=num_kv_heads,
        )

    mod = _require_ext(force=True)
    request_block_table_i32 = request_block_table_i32.to(
        device=final_page_table_i32.device,
        dtype=torch.int32,
    ).contiguous()
    request_recent_first_logical_page_i32 = request_recent_first_logical_page_i32.to(
        device=final_page_table_i32.device,
        dtype=torch.int32,
    ).contiguous()
    request_recent_page_count_i32 = request_recent_page_count_i32.to(
        device=final_page_table_i32.device,
        dtype=torch.int32,
    ).contiguous()
    request_recent_epoch_i32 = request_recent_epoch_i32.to(
        device=final_page_table_i32.device,
        dtype=torch.int32,
    ).contiguous()
    selected_page_count_i32 = selected_page_count_i32.to(
        device=final_page_table_i32.device,
        dtype=torch.int32,
    ).contiguous()

    patch_status_i32, applied_recent_epoch_i32 = mod.step_recent_patch(
        final_page_table_i32,
        request_block_table_i32,
        request_recent_first_logical_page_i32,
        request_recent_page_count_i32,
        request_recent_epoch_i32,
        selected_page_count_i32,
        int(sink_page_slots),
        int(num_kv_heads),
    )
    return {
        "patch_status_i32": patch_status_i32,
        "applied_recent_epoch_i32": applied_recent_epoch_i32,
    }
