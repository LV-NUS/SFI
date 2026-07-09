"""Lightweight C++ extension for selector batch preparation (optional)."""

from __future__ import annotations

import os
from typing import Iterable, Optional, Tuple

import torch
from torch.utils.cpp_extension import load_inline
from utils.torch_extension_cache import load_prebuilt_extension
from utils.ext_toolchain import configure_jit_toolchain_or_raise

_MODULE: Optional[torch.nn.Module] = None
_LOAD_ERROR: Optional[Exception] = None


def _should_enable() -> bool:
    return (
        os.environ.get("VLLM_SPARSE_SELECTOR_CPP_STACK", "1") == "1"
        or os.environ.get("VLLM_SPARSE_SELECTOR_CPP_PREPROC", "1") == "1"
    )


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
    if not _should_enable():
        return None
    prebuilt = load_prebuilt_extension("selector_batch_ext")
    if prebuilt is not None:
        _MODULE = prebuilt
        return _MODULE
    source = r"""
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <algorithm>
#include <vector>

static bool storage_matches(const torch::Tensor& a, const torch::Tensor& b) {
    if (!a.defined() || !b.defined()) {
        return false;
    }
    return a.storage().unsafeGetStorageImpl() == b.storage().unsafeGetStorageImpl();
}

static bool check_capture_view(const torch::Tensor& t, const torch::Tensor& base, int64_t batch, int64_t kv_len) {
    if (t.dim() != 4 || base.dim() != 5) {
        return false;
    }
    if (t.device() != base.device() || t.dtype() != base.dtype()) {
        return false;
    }
    if (t.size(0) != batch || t.size(3) != kv_len) {
        return false;
    }
    if (base.size(1) < batch || base.size(4) < kv_len) {
        return false;
    }
    if (t.size(1) != base.size(2) || t.size(2) != base.size(3)) {
        return false;
    }
    if (t.stride(0) != base.stride(1) ||
        t.stride(1) != base.stride(2) ||
        t.stride(2) != base.stride(3) ||
        t.stride(3) != base.stride(4)) {
        return false;
    }
    return storage_matches(t, base);
}

static bool check_denoms_view(const torch::Tensor& t, const torch::Tensor& base, int64_t batch) {
    if (t.dim() != 2 || base.dim() != 3) {
        return false;
    }
    if (t.device() != base.device() || t.dtype() != base.dtype()) {
        return false;
    }
    if (t.size(0) != batch || base.size(1) < batch) {
        return false;
    }
    if (t.size(1) != base.size(2)) {
        return false;
    }
    if (t.stride(0) != base.stride(1) || t.stride(1) != base.stride(2)) {
        return false;
    }
    return storage_matches(t, base);
}

static bool check_kv_lengths_view(const torch::Tensor& t, const torch::Tensor& base) {
    if (t.dim() != 2 || base.dim() != 2) {
        return false;
    }
    if (t.device() != base.device() || t.dtype() != base.dtype()) {
        return false;
    }
    if (t.sizes() != base.sizes()) {
        return false;
    }
    if (t.strides() != base.strides()) {
        return false;
    }
    return storage_matches(t, base);
}

std::tuple<torch::Tensor, bool> stack_or_view_capture(
    const std::vector<torch::Tensor>& tensors,
    const c10::optional<torch::Tensor>& base_opt,
    const c10::optional<torch::Tensor>& layer_indices_opt,
    int64_t batch_size,
    int64_t kv_len_total,
    const c10::optional<torch::Tensor>& out_opt) {
    if (tensors.empty()) {
        return std::make_tuple(torch::Tensor(), false);
    }
    bool can_use_base = base_opt.has_value() && layer_indices_opt.has_value();
    if (can_use_base) {
        auto base = base_opt.value();
        for (const auto& t : tensors) {
            if (!check_capture_view(t, base, batch_size, kv_len_total)) {
                can_use_base = false;
                break;
            }
        }
    }
    if (can_use_base) {
        auto base = base_opt.value();
        auto layer_indices = layer_indices_opt.value();
        auto out = base.index_select(0, layer_indices);
        if (batch_size < out.size(1)) {
            out = out.narrow(1, 0, batch_size);
        }
        if (kv_len_total < out.size(4)) {
            out = out.narrow(4, 0, kv_len_total);
        }
        return std::make_tuple(out, true);
    }
    if (out_opt.has_value()) {
        auto out = out_opt.value();
        at::stack_out(out, tensors, 0);
        return std::make_tuple(out, false);
    }
    return std::make_tuple(torch::stack(tensors, 0), false);
}

std::tuple<torch::Tensor, bool> stack_or_view_denoms(
    const std::vector<torch::Tensor>& tensors,
    const c10::optional<torch::Tensor>& base_opt,
    const c10::optional<torch::Tensor>& layer_indices_opt,
    int64_t batch_size,
    const c10::optional<torch::Tensor>& out_opt) {
    if (tensors.empty()) {
        return std::make_tuple(torch::Tensor(), false);
    }
    bool can_use_base = base_opt.has_value() && layer_indices_opt.has_value();
    if (can_use_base) {
        auto base = base_opt.value();
        for (const auto& t : tensors) {
            if (!check_denoms_view(t, base, batch_size)) {
                can_use_base = false;
                break;
            }
        }
    }
    if (can_use_base) {
        auto base = base_opt.value();
        auto layer_indices = layer_indices_opt.value();
        auto out = base.index_select(0, layer_indices);
        if (batch_size < out.size(1)) {
            out = out.narrow(1, 0, batch_size);
        }
        return std::make_tuple(out, true);
    }
    if (out_opt.has_value()) {
        auto out = out_opt.value();
        at::stack_out(out, tensors, 0);
        return std::make_tuple(out, false);
    }
    return std::make_tuple(torch::stack(tensors, 0), false);
}

std::tuple<torch::Tensor, bool> stack_or_expand_kv_lengths(
    const std::vector<torch::Tensor>& tensors,
    const c10::optional<torch::Tensor>& out_opt) {
    if (tensors.empty()) {
        return std::make_tuple(torch::Tensor(), false);
    }
    auto base = tensors[0];
    bool can_expand = true;
    for (const auto& t : tensors) {
        if (!check_kv_lengths_view(t, base)) {
            can_expand = false;
            break;
        }
    }
    if (can_expand) {
        auto out = base.unsqueeze(0).expand({static_cast<long>(tensors.size()), base.size(0), base.size(1)});
        return std::make_tuple(out, true);
    }
    if (out_opt.has_value()) {
        auto out = out_opt.value();
        at::stack_out(out, tensors, 0);
        return std::make_tuple(out, false);
    }
    return std::make_tuple(torch::stack(tensors, 0), false);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
preproc_bounds(
    const torch::Tensor& kv_lengths,
    int64_t num_kv_heads,
    int64_t num_queries_per_kv,
    int64_t kv_len_total,
    int64_t sink_cfg,
    int64_t recent_cfg,
    int64_t block_size,
    int64_t window,
    const c10::optional<torch::Tensor>& seq_full_opt) {
    TORCH_CHECK(kv_lengths.defined(), "kv_lengths must be defined");
    TORCH_CHECK(kv_lengths.dim() == 3, "kv_lengths must have shape [layers, batch, heads]");
    auto sizes = kv_lengths.sizes();
    int64_t layers = sizes[0];
    int64_t batch = sizes[1];
    auto kv_lengths_by_kv = kv_lengths.reshape({layers, batch, num_kv_heads, num_queries_per_kv});
    auto kv_lengths_by_kv_i32 = kv_lengths_by_kv.to(at::kInt);
    auto kv_len_head = std::get<0>(kv_lengths_by_kv_i32.max(3));
    kv_len_head = at::clamp_max(kv_len_head, kv_len_total);

    auto head_sink = at::clamp_max(kv_len_head, std::max<int64_t>(0, sink_cfg));
    torch::Tensor recent_start;
    if (block_size > 0 && recent_cfg > 0) {
        if (seq_full_opt.has_value() && seq_full_opt.value().defined()) {
            auto seq_full = seq_full_opt.value();
            auto cap_full = at::clamp_max(seq_full, recent_cfg);
            auto rs_full = (seq_full - cap_full).div(block_size, "trunc") * block_size;
            recent_start = at::clamp(rs_full, 0, kv_len_total);
        } else {
            auto cap = at::clamp_max(kv_len_head, recent_cfg);
            auto rs = (kv_len_head - cap).div(block_size, "trunc") * block_size;
            recent_start = at::clamp(rs, 0, kv_len_total);
        }
    } else {
        recent_start = torch::zeros_like(kv_len_head);
    }

    auto allowed_lengths = at::clamp_min(recent_start - head_sink, 0);
    auto row_lo = head_sink.unsqueeze(-1).unsqueeze(-1).expand(
        {layers, batch, num_kv_heads, num_queries_per_kv, window});

    auto kv_len_q = kv_lengths_by_kv_i32.unsqueeze(-1).expand(
        {layers, batch, num_kv_heads, num_queries_per_kv, window});
    auto row_hi = kv_len_q;
    if (window > 0) {
        auto options = kv_lengths_by_kv_i32.options();
        auto window_idx = at::arange(window, options);
        auto window_m1 = at::scalar_tensor(window - 1, options);
        auto tail_offsets = (window_m1 - window_idx).view({1, 1, 1, 1, window});
        auto hi_stair = kv_len_head.unsqueeze(-1).unsqueeze(-1) - tail_offsets;
        hi_stair = hi_stair.expand({layers, batch, num_kv_heads, num_queries_per_kv, window});
        row_hi = at::minimum(row_hi, hi_stair);
    }
    auto recent_exp = recent_start.unsqueeze(-1).unsqueeze(-1).expand(
        {layers, batch, num_kv_heads, num_queries_per_kv, window});
    row_hi = at::minimum(row_hi, recent_exp);
    row_hi = at::clamp(row_hi, 0, kv_len_total);

    return std::make_tuple(
        kv_lengths_by_kv_i32,
        kv_len_head,
        head_sink,
        recent_start,
        allowed_lengths,
        row_lo,
        row_hi);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("stack_or_view_capture", &stack_or_view_capture, "Stack or view capture_scores");
    m.def("stack_or_view_denoms", &stack_or_view_denoms, "Stack or view log_f_denoms");
    m.def("stack_or_expand_kv_lengths", &stack_or_expand_kv_lengths, "Stack or expand kv_lengths");
    m.def("preproc_bounds", &preproc_bounds, "Precompute bounds for selector");
}
"""
    _ensure_torch_cuda_arch_list()
    try:
        # [EXT-NVCC-GUARD] JIT 回落前 pin nvcc+版本预检(系统 nvcc 10.1 坑根修);
        # 失败信息进 _LOAD_ERROR,由 require/enabled 语义原样呈报。
        configure_jit_toolchain_or_raise(ext_name="vllm_sparse_selector_ext")
        _MODULE = load_inline(
            name="vllm_sparse_selector_ext",
            cpp_sources=source,
            functions=None,
            extra_cflags=["-O3"],
            with_cuda=False,
            verbose=False,
        )
        return _MODULE
    except Exception as exc:
        _LOAD_ERROR = exc
        return None


def stack_or_view_capture(
    tensors: Iterable[torch.Tensor],
    base: Optional[torch.Tensor],
    layer_indices: Optional[torch.Tensor],
    batch_size: int,
    kv_len_total: int,
    out: Optional[torch.Tensor],
) -> Optional[Tuple[torch.Tensor, bool]]:
    mod = _load_ext()
    if mod is None:
        return None
    try:
        return mod.stack_or_view_capture(
            list(tensors),
            base,
            layer_indices,
            int(batch_size),
            int(kv_len_total),
            out,
        )
    except Exception:
        return None


def stack_or_view_denoms(
    tensors: Iterable[torch.Tensor],
    base: Optional[torch.Tensor],
    layer_indices: Optional[torch.Tensor],
    batch_size: int,
    out: Optional[torch.Tensor],
) -> Optional[Tuple[torch.Tensor, bool]]:
    mod = _load_ext()
    if mod is None:
        return None
    try:
        return mod.stack_or_view_denoms(
            list(tensors),
            base,
            layer_indices,
            int(batch_size),
            out,
        )
    except Exception:
        return None


def stack_or_expand_kv_lengths(
    tensors: Iterable[torch.Tensor],
    out: Optional[torch.Tensor],
) -> Optional[Tuple[torch.Tensor, bool]]:
    mod = _load_ext()
    if mod is None:
        return None
    try:
        return mod.stack_or_expand_kv_lengths(list(tensors), out)
    except Exception:
        return None


def preproc_bounds(
    kv_lengths: torch.Tensor,
    num_kv_heads: int,
    num_queries_per_kv: int,
    kv_len_total: int,
    sink_cfg: int,
    recent_cfg: int,
    block_size: int,
    window: int,
    seq_full: Optional[torch.Tensor],
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    if os.environ.get("VLLM_SPARSE_SELECTOR_CPP_PREPROC", "1") != "1":
        return None
    mod = _load_ext()
    if mod is None:
        return None
    try:
        return mod.preproc_bounds(
            kv_lengths,
            int(num_kv_heads),
            int(num_queries_per_kv),
            int(kv_len_total),
            int(sink_cfg),
            int(recent_cfg),
            int(block_size),
            int(window),
            seq_full,
        )
    except Exception:
        return None
