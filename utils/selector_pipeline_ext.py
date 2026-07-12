"""CUDA extension: single-entry selector pipeline (log_s -> soft-nms -> cross-head -> topk)."""
#
# fa4_selector_fixed_shape_topk: VLLM_SPARSE_SELECTOR_FIXED_SHAPE_TOPK (default OFF) selects a
# fixed-shape topk in post_topk: k is always == k_head and empty slots are
# mapped to -1 by a value-sentinel test (picked value <= -3.0e38f, the
# finite min_val sentinel) instead of a data-dependent k_eff column cutoff.
# The C++/CUDA ext source lives in the r-strings below; editing them changes
# the load_inline source hash -> JIT recompile. With the env ON the prebuilt
# .so is REJECTED (see _fixed_shape_topk_required / _load_ext) because a
# stale prebuilt was compiled from source WITHOUT this kernel and would
# silently run the OFF path (same discipline as VLLM_SPARSE_SELECTOR_FUSE_NMS_CROSS).

from __future__ import annotations

import os
from typing import Optional

import torch
from torch.utils.cpp_extension import load_inline
from utils.torch_extension_cache import load_prebuilt_extension
from utils.ext_toolchain import configure_jit_toolchain_or_raise

_MODULE: Optional[torch.nn.Module] = None
_LOAD_ERROR: Optional[Exception] = None


def _should_enable() -> bool:
    return os.environ.get("VLLM_SPARSE_SELECTOR_CUDA_PIPELINE", "1") == "1"


def _workspace_required() -> bool:
    return os.environ.get("VLLM_SPARSE_SELECTOR_PIPELINE_WORKSPACE", "0") == "1"


def _selected_indices_out_required() -> bool:
    return os.environ.get("VLLM_SPARSE_SELECTOR_SELECTED_INDICES_OUT", "1") == "1"


def _fused_nms_cross_required() -> bool:
    return os.environ.get("VLLM_SPARSE_SELECTOR_FUSE_NMS_CROSS", "1") == "1"


# Durable, profile-INDEPENDENT firing signal for the default-ON FUSE_NMS_CROSS
# path. The fused/unfused split lives entirely in C++ (selector_fuse_nms_cross_
# enabled() reads getenv live), so it cannot be branched on in Python; but the
# import-time coherence shim above forces the env present by dispatch time, so a
# Python predicate that mirrors the C++ gate byte-for-byte (env present AND
# value == "1") provably matches the branch taken. Gated by
# VLLM_SPARSE_FUSE_FIRE_TRACE (default-OFF): when unset this is a single dict
# lookup that returns -- no file I/O, no extra env read, no hot-path cost and no
# effect on CUDA-graph replay output. When set, it appends one byte to the trace
# file iff the fused branch is taken, so an e2e gate can confirm firing without
# the CPU profile. All errors are swallowed so telemetry can never perturb prod.
def _fuse_fire_trace() -> None:
    trace = os.environ.get("VLLM_SPARSE_FUSE_FIRE_TRACE")
    if not trace:
        return
    # Mirror the C++ gate exactly: present AND == "1".
    if os.environ.get("VLLM_SPARSE_SELECTOR_FUSE_NMS_CROSS") != "1":
        return
    try:
        path = trace if (os.sep in trace or trace.endswith(".cnt")) else "/tmp/fuse_fired.cnt"
        with open(path, "ab") as _fh:
            _fh.write(b"1")
    except Exception:
        pass


# Coherence shim (default-ON, FUSE_NMS_CROSS): the prebuilt .so C++ gate reads getenv() live
# at dispatch, so a Python-only default flip would not reach it. Export the env when the
# operator has not set it explicitly so the .so sees the intended default; an explicit
# "0"/"1" is always respected.
if os.environ.get("VLLM_SPARSE_SELECTOR_FUSE_NMS_CROSS") is None:
    os.environ["VLLM_SPARSE_SELECTOR_FUSE_NMS_CROSS"] = "1"


def _fixed_shape_topk_required() -> bool:
    # Live env read; coherent with patches.sparse_constants
    # _SELECTOR_FIXED_SHAPE_TOPK_CACHED via that module's import-time F1 shim
    # (it exports the resolved default when the operator left the env unset),
    # so a future constants-side default flip cannot silently accept a stale
    # prebuilt compiled without the fixed-shape kernel.
    return os.environ.get("VLLM_SPARSE_SELECTOR_FIXED_SHAPE_TOPK", "0") == "1"


def _module_satisfies_runtime_requirements(module: torch.nn.Module) -> bool:
    if _workspace_required() and not (
        hasattr(module, "selector_pipeline_logits_topk_with_bounds_workspace")
        and hasattr(module, "selector_pipeline_pre_denom_topk_with_bounds_workspace")
    ):
        return False
    if _selected_indices_out_required() and not (
        hasattr(module, "selector_pipeline_logits_topk_with_bounds_out")
        and hasattr(module, "selector_pipeline_pre_denom_topk_with_bounds_out")
        and hasattr(module, "selector_pipeline_logits_topk_with_bounds_workspace_out")
        and hasattr(module, "selector_pipeline_pre_denom_topk_with_bounds_workspace_out")
    ):
        return False
    return True


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


def _load_ext(*, force: bool = False) -> Optional[torch.nn.Module]:
    global _MODULE, _LOAD_ERROR
    if _MODULE is not None:
        if _module_satisfies_runtime_requirements(_MODULE):
            return _MODULE
        _MODULE = None
    if _LOAD_ERROR is not None:
        return None
    if not force and not _should_enable():
        return None
    if not force:
        prebuilt = load_prebuilt_extension("selector_pipeline_ext")
        if prebuilt is not None:
            if (
                (
                    not _workspace_required()
                    or (
                        hasattr(prebuilt, "selector_pipeline_logits_topk_with_bounds_workspace")
                        and hasattr(prebuilt, "selector_pipeline_pre_denom_topk_with_bounds_workspace")
                    )
                )
                and (
                    not _selected_indices_out_required()
                    or (
                        hasattr(prebuilt, "selector_pipeline_logits_topk_with_bounds_out")
                        and hasattr(prebuilt, "selector_pipeline_pre_denom_topk_with_bounds_out")
                        and hasattr(prebuilt, "selector_pipeline_logits_topk_with_bounds_workspace_out")
                        and hasattr(prebuilt, "selector_pipeline_pre_denom_topk_with_bounds_workspace_out")
                    )
                )
            ):
                if (
                    not _fixed_shape_topk_required()
                ):
                    _MODULE = prebuilt
                    return _MODULE

    cpp_source = r"""
#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> selector_pipeline_logits_topk_cuda(
    torch::Tensor scores,
    torch::Tensor row_lo,
    torch::Tensor row_hi,
    torch::Tensor key_norms,
    torch::Tensor token_lo,
    torch::Tensor token_hi,
    int64_t k_head,
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
    bool lambda_soft,
    int64_t nms_window,
    double soft_alpha,
    double alpha_cross,
    double temperature,
    double cross_eps,
    int64_t slice_start,
    int64_t slice_end,
    c10::optional<torch::Tensor> log_r_cache_opt,
    c10::optional<torch::Tensor> selected_indices_out_opt,
    c10::optional<torch::Tensor> workspace_a_opt,
    c10::optional<torch::Tensor> workspace_b_opt);

std::vector<torch::Tensor> selector_pipeline_pre_denom_topk_cuda(
    torch::Tensor scores,
    torch::Tensor denom,
    torch::Tensor row_lo,
    torch::Tensor row_hi,
    torch::Tensor key_norms,
    torch::Tensor token_lo,
    torch::Tensor token_hi,
    int64_t k_head,
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
    bool lambda_soft,
    int64_t nms_window,
    double soft_alpha,
    double alpha_cross,
    double temperature,
    double cross_eps,
    int64_t slice_start,
    int64_t slice_end,
    c10::optional<torch::Tensor> log_r_cache_opt,
    c10::optional<torch::Tensor> selected_indices_out_opt,
    c10::optional<torch::Tensor> workspace_a_opt,
    c10::optional<torch::Tensor> workspace_b_opt);

std::vector<torch::Tensor> selector_pipeline_logits_topk(
    torch::Tensor scores,
    torch::Tensor row_lo,
    torch::Tensor row_hi,
    torch::Tensor key_norms,
    torch::Tensor token_lo,
    torch::Tensor token_hi,
    int64_t k_head,
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
    bool lambda_soft,
    int64_t nms_window,
    double soft_alpha,
    double alpha_cross,
    double temperature,
    double cross_eps,
    int64_t slice_start,
    int64_t slice_end) {
    return selector_pipeline_logits_topk_cuda(
        scores,
        row_lo,
        row_hi,
        key_norms,
        token_lo,
        token_hi,
        k_head,
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
        lambda_soft,
        nms_window,
        soft_alpha,
        alpha_cross,
        temperature,
        cross_eps,
        slice_start,
        slice_end,
        c10::nullopt,
        c10::nullopt,
        c10::nullopt,
        c10::nullopt);
}

std::vector<torch::Tensor> selector_pipeline_pre_denom_topk(
    torch::Tensor scores,
    torch::Tensor denom,
    torch::Tensor row_lo,
    torch::Tensor row_hi,
    torch::Tensor key_norms,
    torch::Tensor token_lo,
    torch::Tensor token_hi,
    int64_t k_head,
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
    bool lambda_soft,
    int64_t nms_window,
    double soft_alpha,
    double alpha_cross,
    double temperature,
    double cross_eps,
    int64_t slice_start,
    int64_t slice_end) {
    return selector_pipeline_pre_denom_topk_cuda(
        scores,
        denom,
        row_lo,
        row_hi,
        key_norms,
        token_lo,
        token_hi,
        k_head,
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
        lambda_soft,
        nms_window,
        soft_alpha,
        alpha_cross,
        temperature,
        cross_eps,
        slice_start,
        slice_end,
        c10::nullopt,
        c10::nullopt,
        c10::nullopt,
        c10::nullopt);
}

// ============================================================================
// Fused preproc_bounds + pipeline (single Python->C++ call)
// ============================================================================

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
compute_preproc_bounds(
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

    // Reshape and convert to int32
    auto kv_lengths_by_kv = kv_lengths.reshape({layers, batch, num_kv_heads, num_queries_per_kv});
    auto kv_lengths_by_kv_i32 = kv_lengths_by_kv.to(at::kInt);

    // kv_len_head = max over queries per kv group
    auto kv_len_head = std::get<0>(kv_lengths_by_kv_i32.max(3));
    kv_len_head = at::clamp_max(kv_len_head, kv_len_total);

    // head_sink = min(kv_len_head, sink_cfg)
    auto head_sink = at::clamp_max(kv_len_head, std::max<int64_t>(0, sink_cfg));

    // recent_start computation
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

    // allowed_lengths = max(recent_start - head_sink, 0)
    auto allowed_lengths = at::clamp_min(recent_start - head_sink, 0);

    // row_lo: expand head_sink to [L, B, H, G, W]
    auto row_lo = head_sink.unsqueeze(-1).unsqueeze(-1).expand(
        {layers, batch, num_kv_heads, num_queries_per_kv, window});

    // row_hi computation
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

// Fused: preproc_bounds + selector_pipeline_logits_topk (single call)
std::vector<torch::Tensor> selector_pipeline_logits_topk_fused(
    torch::Tensor scores,
    torch::Tensor kv_lengths,
    torch::Tensor key_norms,
    c10::optional<torch::Tensor> seq_full_opt,
    int64_t num_kv_heads,
    int64_t num_queries_per_kv,
    int64_t kv_len_total,
    int64_t sink_cfg,
    int64_t recent_cfg,
    int64_t block_size,
    int64_t k_head,
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
    bool lambda_soft,
    int64_t nms_window,
    double soft_alpha,
    double alpha_cross,
    double temperature,
    double cross_eps,
    int64_t slice_start,
    int64_t slice_end) {

    // Get window from scores shape: [L, B, H, W, K] or [L, B, H*G, W, K]
    auto scores_sizes = scores.sizes();
    int64_t window = scores_sizes[3];

    // 1. Compute preproc_bounds internally (avoids separate Python->C++ call)
    auto [kv_lengths_by_kv_i32, kv_len_head, head_sink, recent_start, allowed_lengths, row_lo, row_hi] =
        compute_preproc_bounds(
            kv_lengths, num_kv_heads, num_queries_per_kv, kv_len_total,
            sink_cfg, recent_cfg, block_size, window, seq_full_opt);

    // 2. Reshape scores for pipeline: [L, B, H, G, W, K]
    int64_t L = scores_sizes[0];
    int64_t B = scores_sizes[1];
    int64_t HG = scores_sizes[2];
    int64_t W = scores_sizes[3];
    int64_t K = scores_sizes[4];
    auto scores_kv = scores.reshape({L, B, num_kv_heads, num_queries_per_kv, W, K});

    // 3. Call pipeline CUDA kernel
    auto idx_result = selector_pipeline_logits_topk_cuda(
        scores_kv, row_lo, row_hi, key_norms, head_sink, recent_start,
        k_head, alpha, eps, gamma,
        prior_weight_l2, prior_weight_pos, prior_pos_power, prior_pos_eta,
        beta, lambda_clip_single, lambda_clip_multi,
        lambda_tail_kappa, lambda_tail_pivot, lambda_soft,
        nms_window, soft_alpha, alpha_cross, temperature, cross_eps,
        slice_start, slice_end, c10::nullopt, c10::nullopt, c10::nullopt, c10::nullopt);

    // 4. Return [selected_indices, head_sink, recent_start, kv_len_head, allowed_lengths]
    return {idx_result[0], head_sink, recent_start, kv_len_head, allowed_lengths};
}

// ============================================================================
// LAZY BOUNDS OPTIMIZATION: avoid 5D expand, compute bounds inline
// ============================================================================

// Lazy bounds version: skip compute_preproc_bounds entirely
// Instead, compute minimal bounds (2D tensors) and let CUDA kernel use stride=0 broadcast
std::vector<torch::Tensor> selector_pipeline_logits_topk_lazy_fused(
    torch::Tensor scores,
    torch::Tensor kv_lengths,
    torch::Tensor key_norms,
    c10::optional<torch::Tensor> seq_full_opt,
    int64_t num_kv_heads,
    int64_t num_queries_per_kv,
    int64_t kv_len_total,
    int64_t sink_cfg,
    int64_t recent_cfg,
    int64_t block_size,
    int64_t k_head,
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
    bool lambda_soft,
    int64_t nms_window,
    double soft_alpha,
    double alpha_cross,
    double temperature,
    double cross_eps,
    int64_t slice_start,
    int64_t slice_end);

// ============================================================================
// WITH_BOUNDS: accept pre-computed bounds from external CUDA kernel
// ============================================================================
// This function accepts bounds computed by compute_bounds_decode (bounds_kernel_ext.py)
// and directly calls the pipeline CUDA kernel, skipping all ATen bounds computation.
std::vector<torch::Tensor> selector_pipeline_logits_topk_with_bounds(
    torch::Tensor scores,           // [L, B, H_total, W, K] or [L, B, H, G, W, K]
    torch::Tensor row_lo,           // [M, R] pre-computed bounds
    torch::Tensor row_hi,           // [M, R] pre-computed bounds
    torch::Tensor key_norms,        // [L, B, H_kv, K]
    torch::Tensor head_sink,        // [L, B, H_kv] or [M] - token_lo
    torch::Tensor recent_start,     // [L, B, H_kv] or [M] - token_hi
    int64_t num_kv_heads,
    int64_t num_queries_per_kv,
    int64_t k_head,
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
    bool lambda_soft,
    int64_t nms_window,
    double soft_alpha,
    double alpha_cross,
    double temperature,
    double cross_eps,
    int64_t slice_start,
    int64_t slice_end,
    c10::optional<torch::Tensor> log_r_cache_opt);

std::vector<torch::Tensor> selector_pipeline_logits_topk_with_bounds_workspace(
    torch::Tensor scores,
    torch::Tensor row_lo,
    torch::Tensor row_hi,
    torch::Tensor key_norms,
    torch::Tensor head_sink,
    torch::Tensor recent_start,
    int64_t num_kv_heads,
    int64_t num_queries_per_kv,
    int64_t k_head,
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
    bool lambda_soft,
    int64_t nms_window,
    double soft_alpha,
    double alpha_cross,
    double temperature,
    double cross_eps,
    int64_t slice_start,
    int64_t slice_end,
    c10::optional<torch::Tensor> log_r_cache_opt,
    torch::Tensor workspace_a,
    torch::Tensor workspace_b);

std::vector<torch::Tensor> selector_pipeline_logits_topk_with_bounds_out(
    torch::Tensor scores,
    torch::Tensor row_lo,
    torch::Tensor row_hi,
    torch::Tensor key_norms,
    torch::Tensor head_sink,
    torch::Tensor recent_start,
    int64_t num_kv_heads,
    int64_t num_queries_per_kv,
    int64_t k_head,
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
    bool lambda_soft,
    int64_t nms_window,
    double soft_alpha,
    double alpha_cross,
    double temperature,
    double cross_eps,
    int64_t slice_start,
    int64_t slice_end,
    c10::optional<torch::Tensor> log_r_cache_opt,
    torch::Tensor selected_indices_out);

std::vector<torch::Tensor> selector_pipeline_logits_topk_with_bounds_workspace_out(
    torch::Tensor scores,
    torch::Tensor row_lo,
    torch::Tensor row_hi,
    torch::Tensor key_norms,
    torch::Tensor head_sink,
    torch::Tensor recent_start,
    int64_t num_kv_heads,
    int64_t num_queries_per_kv,
    int64_t k_head,
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
    bool lambda_soft,
    int64_t nms_window,
    double soft_alpha,
    double alpha_cross,
    double temperature,
    double cross_eps,
    int64_t slice_start,
    int64_t slice_end,
    c10::optional<torch::Tensor> log_r_cache_opt,
    torch::Tensor workspace_a,
    torch::Tensor workspace_b,
    torch::Tensor selected_indices_out);

std::vector<torch::Tensor> selector_pipeline_pre_denom_topk_with_bounds(
    torch::Tensor scores,
    torch::Tensor denom,
    torch::Tensor row_lo,
    torch::Tensor row_hi,
    torch::Tensor key_norms,
    torch::Tensor head_sink,
    torch::Tensor recent_start,
    int64_t num_kv_heads,
    int64_t num_queries_per_kv,
    int64_t k_head,
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
    bool lambda_soft,
    int64_t nms_window,
    double soft_alpha,
    double alpha_cross,
    double temperature,
    double cross_eps,
    int64_t slice_start,
    int64_t slice_end,
    c10::optional<torch::Tensor> log_r_cache_opt);

std::vector<torch::Tensor> selector_pipeline_pre_denom_topk_with_bounds_workspace(
    torch::Tensor scores,
    torch::Tensor denom,
    torch::Tensor row_lo,
    torch::Tensor row_hi,
    torch::Tensor key_norms,
    torch::Tensor head_sink,
    torch::Tensor recent_start,
    int64_t num_kv_heads,
    int64_t num_queries_per_kv,
    int64_t k_head,
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
    bool lambda_soft,
    int64_t nms_window,
    double soft_alpha,
    double alpha_cross,
    double temperature,
    double cross_eps,
    int64_t slice_start,
    int64_t slice_end,
    c10::optional<torch::Tensor> log_r_cache_opt,
    torch::Tensor workspace_a,
    torch::Tensor workspace_b);

std::vector<torch::Tensor> selector_pipeline_pre_denom_topk_with_bounds_out(
    torch::Tensor scores,
    torch::Tensor denom,
    torch::Tensor row_lo,
    torch::Tensor row_hi,
    torch::Tensor key_norms,
    torch::Tensor head_sink,
    torch::Tensor recent_start,
    int64_t num_kv_heads,
    int64_t num_queries_per_kv,
    int64_t k_head,
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
    bool lambda_soft,
    int64_t nms_window,
    double soft_alpha,
    double alpha_cross,
    double temperature,
    double cross_eps,
    int64_t slice_start,
    int64_t slice_end,
    c10::optional<torch::Tensor> log_r_cache_opt,
    torch::Tensor selected_indices_out);

std::vector<torch::Tensor> selector_pipeline_pre_denom_topk_with_bounds_workspace_out(
    torch::Tensor scores,
    torch::Tensor denom,
    torch::Tensor row_lo,
    torch::Tensor row_hi,
    torch::Tensor key_norms,
    torch::Tensor head_sink,
    torch::Tensor recent_start,
    int64_t num_kv_heads,
    int64_t num_queries_per_kv,
    int64_t k_head,
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
    bool lambda_soft,
    int64_t nms_window,
    double soft_alpha,
    double alpha_cross,
    double temperature,
    double cross_eps,
    int64_t slice_start,
    int64_t slice_end,
    c10::optional<torch::Tensor> log_r_cache_opt,
    torch::Tensor workspace_a,
    torch::Tensor workspace_b,
    torch::Tensor selected_indices_out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("selector_pipeline_logits_topk", &selector_pipeline_logits_topk,
          "Selector pipeline (logits path) -> topk (CUDA)");
    m.def("selector_pipeline_pre_denom_topk", &selector_pipeline_pre_denom_topk,
          "Selector pipeline (pre_denom path) -> topk (CUDA)");
    m.def("selector_pipeline_logits_topk_fused", &selector_pipeline_logits_topk_fused,
          "Fused: preproc_bounds + selector pipeline (logits) -> topk (single call)");
    m.def("selector_pipeline_logits_topk_lazy_fused", &selector_pipeline_logits_topk_lazy_fused,
          "Lazy bounds: skip 5D expand, compute bounds inline (logits path)");
    m.def("selector_pipeline_logits_topk_with_bounds", &selector_pipeline_logits_topk_with_bounds,
          "With pre-computed bounds: skip ATen ops, use external CUDA kernel bounds");
    m.def("selector_pipeline_logits_topk_with_bounds_workspace", &selector_pipeline_logits_topk_with_bounds_workspace,
          "With pre-computed bounds and caller-provided selector scratch workspaces");
    m.def("selector_pipeline_logits_topk_with_bounds_out", &selector_pipeline_logits_topk_with_bounds_out,
          "With pre-computed bounds and caller-provided selected_indices output");
    m.def("selector_pipeline_logits_topk_with_bounds_workspace_out", &selector_pipeline_logits_topk_with_bounds_workspace_out,
          "With pre-computed bounds, caller-provided scratch workspaces, and selected_indices output");
    m.def("selector_pipeline_pre_denom_topk_with_bounds", &selector_pipeline_pre_denom_topk_with_bounds,
          "With pre-computed bounds: pre_denom path without fused denom expand");
    m.def("selector_pipeline_pre_denom_topk_with_bounds_workspace", &selector_pipeline_pre_denom_topk_with_bounds_workspace,
          "With pre-computed bounds and caller-provided selector scratch workspaces (pre_denom)");
    m.def("selector_pipeline_pre_denom_topk_with_bounds_out", &selector_pipeline_pre_denom_topk_with_bounds_out,
          "With pre-computed bounds and caller-provided selected_indices output (pre_denom)");
    m.def("selector_pipeline_pre_denom_topk_with_bounds_workspace_out", &selector_pipeline_pre_denom_topk_with_bounds_workspace_out,
          "With pre-computed bounds, caller-provided scratch workspaces, and selected_indices output (pre_denom)");
}
"""

    cuda_source = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <fstream>
#include <sstream>
#include <string>
#include <unistd.h>

static inline torch::Tensor ensure_contig(const torch::Tensor& t) {
    if (!t.defined()) {
        return t;
    }
    return t.is_contiguous() ? t : t.contiguous();
}

static inline torch::Tensor ensure_int32_contig(const torch::Tensor& t) {
    if (!t.defined()) {
        return t;
    }
    if (t.scalar_type() == torch::kInt32) {
        return t.is_contiguous() ? t : t.contiguous();
    }
    return t.to(torch::kInt32).contiguous();
}

static inline torch::Tensor ensure_f32_contig(const torch::Tensor& t) {
    if (!t.defined()) {
        return t;
    }
    if (t.scalar_type() == torch::kFloat32) {
        return t.is_contiguous() ? t : t.contiguous();
    }
    return t.to(torch::kFloat32).contiguous();
}

static inline int64_t now_us() {
    auto tp = std::chrono::steady_clock::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(tp.time_since_epoch());
    return static_cast<int64_t>(us.count());
}

static inline bool pipeline_cpu_profile_enabled() {
    const char* env = std::getenv("VLLM_SPARSE_PIPELINE_CPU_PROFILE");
    return env != nullptr && std::atoi(env) == 1;
}

static inline int64_t pipeline_cpu_profile_outlier_us() {
    const char* env = std::getenv("VLLM_SPARSE_PIPELINE_CPU_PROFILE_OUTLIER_US");
    if (env == nullptr) {
        return 2000;
    }
    return static_cast<int64_t>(std::atoll(env));
}

static inline const char* pipeline_cpu_profile_log_path() {
    const char* env = std::getenv("VLLM_SPARSE_PIPELINE_CPU_PROFILE_LOG");
    return env ? env : "/tmp/vllm_sparse_pipeline_cpu.log";
}

static inline void pipeline_cpu_profile_write(
    const char* kind,
    int64_t L,
    int64_t B,
    int64_t H,
    int64_t G,
    int64_t W,
    int64_t K,
    int64_t k_head,
    int64_t slice_start,
    int64_t slice_end,
    int64_t k_eff,
    int64_t us_log_s,
    int64_t us_nms,
    int64_t us_cross,
    int64_t us_topk,
    int64_t us_total) {
    std::ofstream ofs(pipeline_cpu_profile_log_path(), std::ios::app);
    if (!ofs.is_open()) {
        return;
    }
    ofs << "pipeline_cpu "
        << "pid=" << static_cast<int>(getpid()) << " "
        << "kind=" << kind << " "
        << "L=" << L << " "
        << "B=" << B << " "
        << "H=" << H << " "
        << "G=" << G << " "
        << "W=" << W << " "
        << "K=" << K << " "
        << "k_head=" << k_head << " "
        << "slice_start=" << slice_start << " "
        << "slice_end=" << slice_end << " "
        << "k_eff=" << k_eff << " "
        << "us_log_s=" << us_log_s << " "
        << "us_nms=" << us_nms << " "
        << "us_cross=" << us_cross << " "
        << "us_topk=" << us_topk << " "
        << "us_total=" << us_total
        << "\n";
}

static inline void pipeline_cpu_profile_outlier_write(
    const char* kind,
    int64_t L,
    int64_t B,
    int64_t H,
    int64_t G,
    int64_t W,
    int64_t K,
    int64_t k_head,
    int64_t slice_start,
    int64_t slice_end,
    int64_t k_eff,
    int64_t us_topk,
    int64_t us_total,
    const torch::Tensor& scores) {
    std::ofstream ofs(pipeline_cpu_profile_log_path(), std::ios::app);
    if (!ofs.is_open()) {
        return;
    }
    ofs << "pipeline_topk_outlier "
        << "pid=" << static_cast<int>(getpid()) << " "
        << "kind=" << kind << " "
        << "L=" << L << " "
        << "B=" << B << " "
        << "H=" << H << " "
        << "G=" << G << " "
        << "W=" << W << " "
        << "K=" << K << " "
        << "k_head=" << k_head << " "
        << "slice_start=" << slice_start << " "
        << "slice_end=" << slice_end << " "
        << "k_eff=" << k_eff << " "
        << "us_topk=" << us_topk << " "
        << "us_total=" << us_total << " "
        << "scores_contig=" << (scores.is_contiguous() ? 1 : 0) << " "
        << "scores_dense=" << (scores.is_non_overlapping_and_dense() ? 1 : 0) << " "
        << "scores_stride0=" << scores.stride(0) << " "
        << "scores_stride1=" << scores.stride(1) << " "
        << "scores_stride2=" << scores.stride(2) << " "
        << "scores_stride3=" << scores.stride(3) << " "
        << "scores_dtype=" << static_cast<int>(scores.scalar_type())
        << "\n";
}

namespace {

__device__ __forceinline__ float warp_reduce_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

__device__ __forceinline__ float warp_reduce_max(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xffffffff, v, offset);
        v = v > other ? v : other;
    }
    return v;
}

__device__ __forceinline__ float block_reduce_sum(float v) {
    static __shared__ float shared[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    v = warp_reduce_sum(v);
    if (lane == 0) {
        shared[wid] = v;
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
    float r = shared[0];
    __syncthreads();
    return r;
}

__device__ __forceinline__ float block_reduce_max(float v) {
    static __shared__ float shared[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    v = warp_reduce_max(v);
    if (lane == 0) {
        shared[wid] = v;
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
    float r = shared[0];
    __syncthreads();
    return r;
}

__device__ __forceinline__ float safe_log(float x, float eps) {
    return logf(fmaxf(x, eps));
}

__device__ __forceinline__ float log_add_exp(float a, float b) {
    if (!isfinite(a) && a < 0.0f) {
        return b;
    }
    if (!isfinite(b) && b < 0.0f) {
        return a;
    }
    float m = a > b ? a : b;
    float diff = a > b ? (b - a) : (a - b);
    return m + log1pf(expf(diff));
}

__device__ __forceinline__ float sigmoid_tanh_clip(float x) {
    float x_abs = fabsf(x);
    float e = expf(-2.0f * x_abs);
    float tanh_abs = (1.0f - e) / (1.0f + e);
    return x >= 0.0f ? tanh_abs : -tanh_abs;
}

} // namespace

template <typename scalar_t, typename key_norms_t>
__global__ void fused_log_f_prior_kernel(
    const scalar_t* __restrict__ scores,
    const float* __restrict__ denom,
    const int32_t* __restrict__ row_lo,
    const int32_t* __restrict__ row_hi,
    const int32_t* __restrict__ token_lo_ptr,
    const int32_t* __restrict__ token_hi_ptr,
    const key_norms_t* __restrict__ key_norms,
    int M,
    int R,
    int K,
    int block_k,
    int num_blocks,
    int stride_scores_m,
    int stride_scores_r,
    int stride_scores_k,
    int stride_denom_m,
    int stride_denom_r,
    int stride_lo_m,
    int stride_lo_r,
    int stride_hi_m,
    int stride_hi_r,
    int stride_tlo_m,
    int stride_thi_m,
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

    extern __shared__ float shm[];
    float* row_lse_s = shm;
    int* row_valid_s = (int*)(row_lse_s + R);
    int* row_lo_s = row_valid_s + R;
    int* row_hi_s = row_lo_s + R;
    int* token_bounds_s = row_hi_s + R;
    float* lambda_s = (float*)(token_bounds_s + 2);

    int tid = threadIdx.x;
    for (int r = tid; r < R; r += blockDim.x) {
        int32_t lo = row_lo[(int64_t)m * stride_lo_m + r * stride_lo_r];
        int32_t hi = row_hi[(int64_t)m * stride_hi_m + r * stride_hi_r];
        row_lo_s[r] = lo;
        row_hi_s[r] = hi;
    }
    __syncthreads();

    int token_lo = 0;
    int token_hi = 0;
    if (tid == 0) {
        int lo = token_lo_ptr[(int64_t)m * stride_tlo_m];
        int hi = token_hi_ptr[(int64_t)m * stride_thi_m];
        if (lo < 0) {
            lo = 0;
        }
        if (hi > K) {
            hi = K;
        }
        token_bounds_s[0] = lo;
        token_bounds_s[1] = hi;
    }
    __syncthreads();
    token_lo = token_bounds_s[0];
    token_hi = token_bounds_s[1];

    float row_count = 0.0f;
    const float min_val = -3.402823466e38f;
    if (!use_denom) {
        for (int r = 0; r < R; ++r) {
            int lo = row_lo_s[r];
            int hi = row_hi_s[r];
            float max_val = min_val;
            float sum_exp = 0.0f;
            float count = 0.0f;
            for (int block_start = 0; block_start < num_blocks; ++block_start) {
                int k = block_start * block_k + tid;
                bool mask_k = (k < K);
                bool in_bounds = mask_k && (k >= lo) && (k < hi);
                float val = min_val;
                if (in_bounds) {
                    val = static_cast<float>(scores[(int64_t)m * stride_scores_m + r * stride_scores_r + k * stride_scores_k]);
                }
                bool valid = in_bounds && (val > min_val);
                float v = valid ? val : min_val;
                float block_max = block_reduce_max(v);
                float new_max = fmaxf(max_val, block_max);
                float exp_block = valid ? expf(v - new_max) : 0.0f;
                float block_sum = block_reduce_sum(exp_block);
                sum_exp = sum_exp * expf(max_val - new_max) + block_sum;
                max_val = new_max;
                float block_count = valid ? 1.0f : 0.0f;
                block_count = block_reduce_sum(block_count);
                count += block_count;
            }
            if (tid == 0) {
                if (count > 0.0f) {
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
            out[(int64_t)m * K + k] = min_val;
        }
        return;
    }

    const bool use_mean = fabsf(alpha) < 1.0e-6f;
    const float prior_pos_power_f = prior_pos_power < 1.0f ? 1.0f : prior_pos_power;
    __shared__ float inv_denom_pos_s;
    __shared__ float inv_row_count_s;
    __shared__ float log_row_count_s;
    if (tid == 0) {
        int denom_pos_i = token_hi - token_lo - 1;
        if (denom_pos_i < 1) {
            denom_pos_i = 1;
        }
        inv_denom_pos_s = 1.0f / static_cast<float>(denom_pos_i);
        inv_row_count_s = 1.0f / row_count;
        log_row_count_s = logf(row_count);
    }
    __syncthreads();
    float inv_denom_pos = inv_denom_pos_s;
    float inv_row_count = inv_row_count_s;
    float log_row_count = log_row_count_s;

    // Pass 1: compute log_f/log_r_raw and denom_f/denom_r (two-pass).
    float local_max_f = min_val;
    float local_max_r = min_val;
    for (int k = tid; k < K; k += blockDim.x) {
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
                        float lfp = static_cast<float>(scores[(int64_t)m * stride_scores_m + r * stride_scores_r + k * stride_scores_k]);
                        float den = denom[(int64_t)m * stride_denom_m + r * stride_denom_r];
                        if (!isfinite(lfp) || !isfinite(den)) {
                            continue;
                        }
                        lp = lfp - den;
                    } else {
                        float val = static_cast<float>(scores[(int64_t)m * stride_scores_m + r * stride_scores_r + k * stride_scores_k]);
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
                        float lfp = static_cast<float>(scores[(int64_t)m * stride_scores_m + r * stride_scores_r + k * stride_scores_k]);
                        float den = denom[(int64_t)m * stride_denom_m + r * stride_denom_r];
                        if (!isfinite(lfp) || !isfinite(den)) {
                            continue;
                        }
                        lp = lfp - den;
                    } else {
                        float val = static_cast<float>(scores[(int64_t)m * stride_scores_m + r * stride_scores_r + k * stride_scores_k]);
                        if (!isfinite(val)) {
                            continue;
                        }
                        lp = val - row_lse_s[r];
                    }
                    if (lp <= min_val) {
                        continue;
                    }
                    float log_f_pre = alpha * lp;
                    float new_max = fmaxf(log_f_max, log_f_pre);
                    float update = log_f_sum * expf(log_f_max - new_max) + expf(log_f_pre - new_max);
                    log_f_sum = update;
                    log_f_max = new_max;
                }
                if (log_f_max > min_val) {
                    float lse = log_f_max + logf(log_f_sum);
                    log_f = (lse - log_row_count) / alpha;
                }
            }

            float kn = static_cast<float>(key_norms[(int64_t)m * stride_kn_m + k * stride_kn_k]);
            kn = fmaxf(kn, eps);
            float log_pi = -gamma * logf(kn);
            float pos_norm = (static_cast<float>(k - token_lo)) * inv_denom_pos;
            pos_norm = fmaxf(0.0f, fminf(1.0f, pos_norm));
            float pos_norm_pos = fmaxf(pos_norm, eps);
            float pos_shaped = expf(prior_pos_power_f * logf(pos_norm_pos));
            if (pos_norm <= 0.0f) {
                pos_shaped = 0.0f;
            }
            float one_minus = fmaxf(1.0f - pos_norm, eps);
            float base_delta = -beta * pos_shaped;
            base_delta = base_delta + prior_pos_eta * logf(one_minus);
            float log_r_l2 = log_pi;
            float log_r_delta = base_delta;
            log_r_raw = prior_weight_l2 * log_r_l2 + prior_weight_pos * log_r_delta;
        }

        if (k < K) {
            out[(int64_t)m * K + k] = in_bounds ? log_f : min_val;
            if (log_r_cache != nullptr) {
                log_r_cache[(int64_t)m * K + k] = log_r_raw;
            }
        }
        if (in_bounds && log_f > local_max_f) {
            local_max_f = log_f;
        }
        if (in_bounds && log_r_raw > local_max_r) {
            local_max_r = log_r_raw;
        }
    }

    float max_f = block_reduce_max(local_max_f);
    float max_r = block_reduce_max(local_max_r);
    float local_sum_f = 0.0f;
    float local_sum_r = 0.0f;
    for (int k = tid; k < K; k += blockDim.x) {
        if (k < token_lo || k >= token_hi) {
            continue;
        }
        float log_f = out[(int64_t)m * K + k];
        if (log_f > min_val && max_f > min_val) {
            local_sum_f += expf(log_f - max_f);
        }
        float log_r_raw = min_val;
        if (log_r_cache != nullptr) {
            log_r_raw = log_r_cache[(int64_t)m * K + k];
        } else {
            float kn = static_cast<float>(key_norms[(int64_t)m * stride_kn_m + k * stride_kn_k]);
            kn = fmaxf(kn, eps);
            float log_pi = -gamma * logf(kn);
            float pos_norm = (static_cast<float>(k - token_lo)) * inv_denom_pos;
            pos_norm = fmaxf(0.0f, fminf(1.0f, pos_norm));
            float pos_norm_pos = fmaxf(pos_norm, eps);
            float pos_shaped = expf(prior_pos_power_f * logf(pos_norm_pos));
            if (pos_norm <= 0.0f) {
                pos_shaped = 0.0f;
            }
            float one_minus = fmaxf(1.0f - pos_norm, eps);
            float base_delta = -beta * pos_shaped;
            base_delta = base_delta + prior_pos_eta * logf(one_minus);
            float log_r_delta = base_delta;
            log_r_raw = prior_weight_l2 * log_pi + prior_weight_pos * log_r_delta;
        }
        if (log_r_raw > min_val && max_r > min_val) {
            local_sum_r += expf(log_r_raw - max_r);
        }
    }
    float sum_f = block_reduce_sum(local_sum_f);
    float sum_r = block_reduce_sum(local_sum_r);
    float denom_f = (max_f > min_val) ? (max_f + logf(fmaxf(sum_f, eps))) : min_val;
    float denom_r = (max_r > min_val) ? (max_r + logf(fmaxf(sum_r, eps))) : min_val;

    // Pass 2: ff/rr/fr and tail stats for lambda.
    float local_ff = 0.0f;
    float local_rr = 0.0f;
    float local_fr = 0.0f;
    float local_prob_sum = 0.0f;
    float local_pos_sum = 0.0f;

    for (int k = tid; k < K; k += blockDim.x) {
        if (k < token_lo || k >= token_hi) {
            continue;
        }
        float log_f = out[(int64_t)m * K + k];
        float lp = min_val;
        if (log_f > min_val && denom_f > min_val) {
            lp = log_f - denom_f;
        }
        float log_r_raw = min_val;
        if (log_r_cache != nullptr) {
            log_r_raw = log_r_cache[(int64_t)m * K + k];
        } else {
            float kn = static_cast<float>(key_norms[(int64_t)m * stride_kn_m + k * stride_kn_k]);
            kn = fmaxf(kn, eps);
            float log_pi = -gamma * logf(kn);
            float pos_norm = (static_cast<float>(k - token_lo)) * inv_denom_pos;
            pos_norm = fmaxf(0.0f, fminf(1.0f, pos_norm));
            float pos_norm_pos = fmaxf(pos_norm, eps);
            float pos_shaped = expf(prior_pos_power_f * logf(pos_norm_pos));
            if (pos_norm <= 0.0f) {
                pos_shaped = 0.0f;
            }
            float one_minus = fmaxf(1.0f - pos_norm, eps);
            float base_delta = -beta * pos_shaped;
            base_delta = base_delta + prior_pos_eta * logf(one_minus);
            float log_r_delta = base_delta;
            log_r_raw = prior_weight_l2 * log_pi + prior_weight_pos * log_r_delta;
        }
        float log_r = min_val;
        if (log_r_raw > min_val && denom_r > min_val) {
            log_r = log_r_raw - denom_r;
        }

        if (lp > min_val) {
            local_ff += expf(2.0f * lp);
        }
        if (log_r > min_val) {
            local_rr += expf(2.0f * log_r);
        }
        if (lp > min_val && log_r > min_val) {
            local_fr += expf(lp + log_r);
        }

        if (lambda_tail_kappa > 0.0f && lp > min_val) {
            float p = expf(lp);
            float pos_norm = (static_cast<float>(k - token_lo)) * inv_denom_pos;
            pos_norm = fmaxf(0.0f, fminf(1.0f, pos_norm));
            local_prob_sum += p;
            local_pos_sum += p * pos_norm;
        }
    }

    float ff = block_reduce_sum(local_ff);
    float rr = block_reduce_sum(local_rr);
    float fr = block_reduce_sum(local_fr);
    float prob_sum = block_reduce_sum(local_prob_sum);
    float pos_sum = block_reduce_sum(local_pos_sum);

    if (tid == 0) {
        float denom_lam = fmaxf(ff - 2.0f * fr + rr, eps);
        float lam_star = (ff - fr) / denom_lam;
        float lam = fmaxf(lam_star, 0.0f);
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

    // Pass 3: fused_raw and denom_fused.
    float local_max_fused = min_val;
    for (int k = tid; k < K; k += blockDim.x) {
        if (k < token_lo || k >= token_hi) {
            continue;
        }
        float log_f = out[(int64_t)m * K + k];
        float lp = min_val;
        if (log_f > min_val && denom_f > min_val) {
            lp = log_f - denom_f;
        }
        float log_r_raw = min_val;
        if (log_r_cache != nullptr) {
            log_r_raw = log_r_cache[(int64_t)m * K + k];
        } else {
            float kn = static_cast<float>(key_norms[(int64_t)m * stride_kn_m + k * stride_kn_k]);
            kn = fmaxf(kn, eps);
            float log_pi = -gamma * logf(kn);
            float pos_norm = (static_cast<float>(k - token_lo)) * inv_denom_pos;
            pos_norm = fmaxf(0.0f, fminf(1.0f, pos_norm));
            float pos_norm_pos = fmaxf(pos_norm, eps);
            float pos_shaped = expf(prior_pos_power_f * logf(pos_norm_pos));
            if (pos_norm <= 0.0f) {
                pos_shaped = 0.0f;
            }
            float one_minus = fmaxf(1.0f - pos_norm, eps);
            float base_delta = -beta * pos_shaped;
            base_delta = base_delta + prior_pos_eta * logf(one_minus);
            float log_r_delta = base_delta;
            log_r_raw = prior_weight_l2 * log_pi + prior_weight_pos * log_r_delta;
        }
        float log_r = min_val;
        if (log_r_raw > min_val && denom_r > min_val) {
            log_r = log_r_raw - denom_r;
        }
        float fused_raw = log_add_exp(log_one_minus + lp, log_lambda + log_r);
        out[(int64_t)m * K + k] = fused_raw;
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
        float fused_raw = out[(int64_t)m * K + k];
        if (fused_raw > min_val && max_fused > min_val) {
            local_sum_fused += expf(fused_raw - max_fused);
        }
    }
    float sum_fused = block_reduce_sum(local_sum_fused);
    float denom_fused = (max_fused > min_val) ? (max_fused + logf(fmaxf(sum_fused, eps))) : min_val;

    // Pass 4: normalize fused.
    for (int k = tid; k < K; k += blockDim.x) {
        if (k < token_lo || k >= token_hi) {
            out[(int64_t)m * K + k] = min_val;
            continue;
        }
        float fused_raw = out[(int64_t)m * K + k];
        out[(int64_t)m * K + k] = fused_raw - denom_fused;
    }
}

template <typename scalar_t>
__global__ void soft_nms_bounds_kernel(
    const scalar_t* __restrict__ log_s,
    const int32_t* __restrict__ token_lo,
    const int32_t* __restrict__ token_hi,
    int N,
    int H,
    int K,
    int stride_s_n,
    int stride_s_h,
    int stride_s_k,
    int stride_lo_n,
    int stride_lo_h,
    int stride_hi_n,
    int stride_hi_h,
    int window,
    float alpha,
    scalar_t* __restrict__ out) {
    int nh = blockIdx.x;
    int k = blockIdx.y * blockDim.x + threadIdx.x;
    if (nh >= N * H) {
        return;
    }
    int n = nh / H;
    int h = nh - n * H;

    int lo = token_lo[(int64_t)n * stride_lo_n + h * stride_lo_h];
    int hi = token_hi[(int64_t)n * stride_hi_n + h * stride_hi_h];
    if (lo < 0) {
        lo = 0;
    }
    if (hi > K) {
        hi = K;
    }

    bool valid_k = (k < K);
    float score = 0.0f;
    if (valid_k) {
        score = static_cast<float>(log_s[(int64_t)n * stride_s_n + h * stride_s_h + k * stride_s_k]);
    }
    bool in_bounds = valid_k && (k >= lo) && (k < hi);

    int pad = window / 2;
    int window_len = 2 * pad + 1;
    int base_k = blockIdx.y * blockDim.x;
    int tile_len = blockDim.x + 2 * pad;
    extern __shared__ float shm[];
    for (int i = threadIdx.x; i < tile_len; i += blockDim.x) {
        int idx = base_k + i - pad;
        float val = -INFINITY;
        if (idx >= 0 && idx < K) {
            bool in_bounds_off = (idx >= lo) && (idx < hi);
            if (in_bounds_off) {
                val = static_cast<float>(log_s[(int64_t)n * stride_s_n + h * stride_s_h + idx * stride_s_k]);
            }
        }
        shm[i] = val;
    }
    __syncthreads();

    if (!valid_k) {
        return;
    }
    if (!in_bounds) {
        out[(int64_t)n * stride_s_n + h * stride_s_h + k * stride_s_k] = static_cast<scalar_t>(score);
        return;
    }
    if (!isfinite(score)) {
        out[(int64_t)n * stride_s_n + h * stride_s_h + k * stride_s_k] = static_cast<scalar_t>(score);
        return;
    }

    float pooled = -INFINITY;
    int start = threadIdx.x;
    for (int i = 0; i < window_len; ++i) {
        float val = shm[start + i];
        pooled = pooled > val ? pooled : val;
    }

    float delta = pooled - score;
    if (delta < 0.0f) {
        delta = 0.0f;
    }
    float out_val = score - alpha * delta;
    out[(int64_t)n * stride_s_n + h * stride_s_h + k * stride_s_k] = static_cast<scalar_t>(out_val);
}

template <typename scalar_t, int BLOCK_K>
__global__ void cross_head_mutex_bounds_kernel(
    const scalar_t* __restrict__ log_s,
    const int32_t* __restrict__ token_lo,
    const int32_t* __restrict__ token_hi,
    int N,
    int H,
    int K,
    int stride_s_n,
    int stride_s_h,
    int stride_s_k,
    int stride_lo_n,
    int stride_lo_h,
    int stride_hi_n,
    int stride_hi_h,
    float alpha_cross,
    float temperature,
    float eps,
    scalar_t* __restrict__ out) {
    int n = blockIdx.x;
    int block_k = blockIdx.y;
    int tid = threadIdx.x;
    int k = block_k * BLOCK_K + tid;
    if (n >= N) {
        return;
    }

    extern __shared__ unsigned char shm_raw[];
    int* lo_s = reinterpret_cast<int*>(shm_raw);
    int* hi_s = lo_s + H;
    float* scores_s = reinterpret_cast<float*>(hi_s + H);
    for (int h = tid; h < H; h += blockDim.x) {
        int lo = token_lo[(int64_t)n * stride_lo_n + h * stride_lo_h];
        int hi = token_hi[(int64_t)n * stride_hi_n + h * stride_hi_h];
        if (lo < 0) {
            lo = 0;
        }
        if (hi > K) {
            hi = K;
        }
        lo_s[h] = lo;
        hi_s[h] = hi;
    }
    __syncthreads();

    if (k >= K) {
        return;
    }

    const float min_val = -3.402823466e38f;
    float inv_temp = 1.0f / temperature;

    for (int h = 0; h < H; ++h) {
        float score = static_cast<float>(log_s[(int64_t)n * stride_s_n + h * stride_s_h + k * stride_s_k]);
        scores_s[h * BLOCK_K + tid] = score;
    }

    float max_val = min_val;
    for (int h = 0; h < H; ++h) {
        int lo = lo_s[h];
        int hi = hi_s[h];
        bool valid = (k >= lo) && (k < hi);
        float score = scores_s[h * BLOCK_K + tid];
        bool score_finite = isfinite(score);
        float masked = (valid && score_finite) ? (score * inv_temp) : min_val;
        max_val = max_val > masked ? max_val : masked;
    }

    float sum_val = 0.0f;
    for (int h = 0; h < H; ++h) {
        int lo = lo_s[h];
        int hi = hi_s[h];
        bool valid = (k >= lo) && (k < hi);
        if (!valid) {
            continue;
        }
        float score = scores_s[h * BLOCK_K + tid];
        if (!isfinite(score)) {
            continue;
        }
        float masked = score * inv_temp;
        sum_val += expf(masked - max_val);
    }
    if (sum_val <= 0.0f) {
        sum_val = 1.0f;
    }

    for (int h = 0; h < H; ++h) {
        int lo = lo_s[h];
        int hi = hi_s[h];
        bool valid = (k >= lo) && (k < hi);
        float score = scores_s[h * BLOCK_K + tid];
        if (valid && isfinite(score)) {
            float masked = score * inv_temp;
            float r = expf(masked - max_val) / sum_val;
            float log_r = logf(fmaxf(r, eps));
            score = score + alpha_cross * log_r;
        }
        out[(int64_t)n * stride_s_n + h * stride_s_h + k * stride_s_k] = static_cast<scalar_t>(score);
    }
}

template <typename scalar_t, int BLOCK_K>
__global__ void soft_nms_cross_head_bounds_kernel(
    const scalar_t* __restrict__ log_s,
    const int32_t* __restrict__ token_lo,
    const int32_t* __restrict__ token_hi,
    int N,
    int H,
    int K,
    int stride_s_n,
    int stride_s_h,
    int stride_s_k,
    int stride_lo_n,
    int stride_lo_h,
    int stride_hi_n,
    int stride_hi_h,
    int window,
    float soft_alpha,
    float alpha_cross,
    float temperature,
    float eps,
    scalar_t* __restrict__ out) {
    int n = blockIdx.x;
    int block_k = blockIdx.y;
    int tid = threadIdx.x;
    int k = block_k * BLOCK_K + tid;
    if (n >= N) {
        return;
    }

    int pad = window / 2;
    int window_len = 2 * pad + 1;
    int tile_len = BLOCK_K + 2 * pad;
    int base_k = block_k * BLOCK_K;

    extern __shared__ unsigned char shm_raw[];
    int* lo_s = reinterpret_cast<int*>(shm_raw);
    int* hi_s = lo_s + H;
    float* scores_s = reinterpret_cast<float*>(hi_s + H);
    float* tile_s = scores_s + static_cast<size_t>(H) * static_cast<size_t>(BLOCK_K);

    for (int h = tid; h < H; h += blockDim.x) {
        int lo = token_lo[(int64_t)n * stride_lo_n + h * stride_lo_h];
        int hi = token_hi[(int64_t)n * stride_hi_n + h * stride_hi_h];
        if (lo < 0) {
            lo = 0;
        }
        if (hi > K) {
            hi = K;
        }
        lo_s[h] = lo;
        hi_s[h] = hi;
    }
    for (int linear = tid; linear < H * tile_len; linear += blockDim.x) {
        int h = linear / tile_len;
        int i = linear - h * tile_len;
        int idx = base_k + i - pad;
        float val = -INFINITY;
        if (idx >= 0 && idx < K) {
            int lo = token_lo[(int64_t)n * stride_lo_n + h * stride_lo_h];
            int hi = token_hi[(int64_t)n * stride_hi_n + h * stride_hi_h];
            if (lo < 0) {
                lo = 0;
            }
            if (hi > K) {
                hi = K;
            }
            if (idx >= lo && idx < hi) {
                val = static_cast<float>(log_s[(int64_t)n * stride_s_n + h * stride_s_h + idx * stride_s_k]);
            }
        }
        tile_s[linear] = val;
    }
    __syncthreads();

    if (k >= K) {
        return;
    }

    const float min_val = -3.402823466e38f;
    float inv_temp = 1.0f / temperature;

    for (int h = 0; h < H; ++h) {
        int lo = lo_s[h];
        int hi = hi_s[h];
        bool in_bounds = (k >= lo) && (k < hi);
        float score = static_cast<float>(log_s[(int64_t)n * stride_s_n + h * stride_s_h + k * stride_s_k]);
        if (in_bounds && isfinite(score)) {
            float pooled = -INFINITY;
            int tile_base = h * tile_len + tid;
            for (int i = 0; i < window_len; ++i) {
                float val = tile_s[tile_base + i];
                pooled = pooled > val ? pooled : val;
            }
            float delta = pooled - score;
            if (delta < 0.0f) {
                delta = 0.0f;
            }
            score = score - soft_alpha * delta;
        }
        scores_s[h * BLOCK_K + tid] = score;
    }

    float max_val = min_val;
    for (int h = 0; h < H; ++h) {
        int lo = lo_s[h];
        int hi = hi_s[h];
        bool valid = (k >= lo) && (k < hi);
        float score = scores_s[h * BLOCK_K + tid];
        bool score_finite = isfinite(score);
        float masked = (valid && score_finite) ? (score * inv_temp) : min_val;
        max_val = max_val > masked ? max_val : masked;
    }

    float sum_val = 0.0f;
    for (int h = 0; h < H; ++h) {
        int lo = lo_s[h];
        int hi = hi_s[h];
        bool valid = (k >= lo) && (k < hi);
        if (!valid) {
            continue;
        }
        float score = scores_s[h * BLOCK_K + tid];
        if (!isfinite(score)) {
            continue;
        }
        float masked = score * inv_temp;
        sum_val += expf(masked - max_val);
    }
    if (sum_val <= 0.0f) {
        sum_val = 1.0f;
    }

    for (int h = 0; h < H; ++h) {
        int lo = lo_s[h];
        int hi = hi_s[h];
        bool valid = (k >= lo) && (k < hi);
        float score = scores_s[h * BLOCK_K + tid];
        if (valid && isfinite(score)) {
            float masked = score * inv_temp;
            float r = expf(masked - max_val) / sum_val;
            float log_r = logf(fmaxf(r, eps));
            score = score + alpha_cross * log_r;
        }
        out[(int64_t)n * stride_s_n + h * stride_s_h + k * stride_s_k] = static_cast<scalar_t>(score);
    }
}

torch::Tensor workspace_2d_or_empty(
    const c10::optional<torch::Tensor>& workspace_opt,
    int64_t rows,
    int64_t cols,
    const torch::Tensor& ref,
    at::ScalarType dtype,
    const char* name) {
    if (workspace_opt.has_value() && workspace_opt.value().defined()) {
        auto workspace = workspace_opt.value();
        TORCH_CHECK(workspace.device() == ref.device(), name, " device mismatch");
        TORCH_CHECK(workspace.scalar_type() == dtype, name, " dtype mismatch");
        TORCH_CHECK(workspace.is_contiguous(), name, " must be contiguous");
        TORCH_CHECK(workspace.dim() == 2, name, " must be [rows, cols]");
        TORCH_CHECK(workspace.size(0) == rows && workspace.size(1) == cols,
                    name, " shape mismatch");
        return workspace;
    }
    return torch::empty({rows, cols}, ref.options().dtype(dtype));
}

// [ENV-PARSE-SINGLE-DIALECT 2026-07-11 EXT审计·仅卫生] 这两个 gate 在 python
// 侧各有严格镜像判定（本文件 python 段: os.environ.get(...) == "1"），而旧
// C++ 侧用 atoi==1 接受 "01"/"1x" 等前缀形态 => 同一进程对同一 env 值 C++/
// python 判定分裂（C++ ON / python OFF 的隐性路由撕裂）。统一为严格 "0"/"1"
// 枚举；其它值 fail-fast（无 fallback 纪律：坏值不再产生分裂路由）。生产
// 形态恒为显式 "0"/"1"（python 段 import 期 coherence shim 保证），零行为变化。
bool selector_fuse_nms_cross_enabled() {
    const char* env = std::getenv("VLLM_SPARSE_SELECTOR_FUSE_NMS_CROSS");
    if (env == nullptr) {
        return false;
    }
    const bool is_one = env[0] == '1' && env[1] == '\0';
    const bool is_zero = env[0] == '0' && env[1] == '\0';
    TORCH_CHECK(is_one || is_zero,
                "VLLM_SPARSE_SELECTOR_FUSE_NMS_CROSS must be '0' or '1', got '",
                env, "'");
    return is_one;
}

// fa4_selector_fixed_shape_topk: env gate for fixed-shape (k == k_head) post_topk.
bool selector_fixed_shape_topk_enabled() {
    const char* env = std::getenv("VLLM_SPARSE_SELECTOR_FIXED_SHAPE_TOPK");
    if (env == nullptr) {
        return false;
    }
    const bool is_one = env[0] == '1' && env[1] == '\0';
    const bool is_zero = env[0] == '0' && env[1] == '\0';
    TORCH_CHECK(is_one || is_zero,
                "VLLM_SPARSE_SELECTOR_FIXED_SHAPE_TOPK must be '0' or '1', got '",
                env, "'");
    return is_one;
}

size_t fused_nms_cross_shm_bytes(int64_t H, int block_k, int window) {
    int pad = window / 2;
    int tile_len = block_k + 2 * pad;
    return sizeof(int) * static_cast<size_t>(H) * 2
        + sizeof(float) * static_cast<size_t>(H) * static_cast<size_t>(block_k)
        + sizeof(float) * static_cast<size_t>(H) * static_cast<size_t>(tile_len);
}

torch::Tensor run_soft_nms(
    torch::Tensor log_s,
    torch::Tensor token_lo,
    torch::Tensor token_hi,
    int64_t window,
    double alpha,
    c10::optional<torch::Tensor> workspace_opt) {
    auto log_s_c = log_s.contiguous();
    auto lo_i32 = token_lo.to(torch::kInt32).contiguous();
    auto hi_i32 = token_hi.to(torch::kInt32).contiguous();

    int64_t N = log_s_c.size(0);
    int64_t H = log_s_c.size(1);
    int64_t K = log_s_c.size(2);
    if (N == 0 || H == 0 || K == 0) {
        return log_s_c;
    }

    int win = static_cast<int>(window);
    if (win < 1) {
        win = 1;
    }
    if (win > K) {
        win = static_cast<int>(K);
    }

    auto out_flat = workspace_2d_or_empty(
        workspace_opt,
        N * H,
        K,
        log_s_c,
        log_s_c.scalar_type(),
        "selector_soft_nms_workspace");
    auto out = out_flat.reshape({N, H, K});
	    int threads = 256;
    dim3 blocks(static_cast<unsigned int>(N * H), static_cast<unsigned int>((K + threads - 1) / threads));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int pad = win / 2;
    int tile_len = threads + 2 * pad;
    size_t shm_size = sizeof(float) * static_cast<size_t>(tile_len);
    // [NMS-SHM-FAILFAST 2026-07-11 EXT审计·理论可达] win 只被钳到 K；大 window
    // 使动态 shm 超过 48KB 默认上限时 launch 会静默失败（无 launch check），
    // workspace 旧值被原样返回 = 静默错值。fail-fast：超限直接 raise（不加
    // 自动降级，无 fallback 纪律），launch 后补 C10_CUDA_KERNEL_LAUNCH_CHECK。
    constexpr size_t kMaxDynamicShmemBytes = 48 * 1024;
    TORCH_CHECK(shm_size <= kMaxDynamicShmemBytes,
                "run_soft_nms: nms_window=", win, " needs ", shm_size,
                " bytes dynamic shared memory (>48KB SM default limit); "
                "reduce the window");

    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, log_s_c.scalar_type(), "soft_nms_bounds", [&] {
        soft_nms_bounds_kernel<scalar_t><<<blocks, threads, shm_size, stream>>>(
            log_s_c.data_ptr<scalar_t>(),
            lo_i32.data_ptr<int32_t>(),
            hi_i32.data_ptr<int32_t>(),
            static_cast<int>(N),
            static_cast<int>(H),
            static_cast<int>(K),
            static_cast<int>(log_s_c.stride(0)),
            static_cast<int>(log_s_c.stride(1)),
            static_cast<int>(log_s_c.stride(2)),
            static_cast<int>(lo_i32.stride(0)),
            static_cast<int>(lo_i32.stride(1)),
            static_cast<int>(hi_i32.stride(0)),
            static_cast<int>(hi_i32.stride(1)),
            win,
            static_cast<float>(alpha),
            out.data_ptr<scalar_t>());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor run_cross_head(
    torch::Tensor log_s,
    torch::Tensor token_lo,
    torch::Tensor token_hi,
    double alpha_cross,
    double temperature,
    double eps,
    c10::optional<torch::Tensor> workspace_opt) {
    auto log_s_c = log_s.contiguous();
    auto lo_i32 = token_lo.to(torch::kInt32).contiguous();
    auto hi_i32 = token_hi.to(torch::kInt32).contiguous();

    int64_t N = log_s_c.size(0);
    int64_t H = log_s_c.size(1);
    int64_t K = log_s_c.size(2);
    if (N == 0 || H == 0 || K == 0) {
        return log_s_c;
    }
    if (H <= 2) {
        return log_s_c;
    }
    auto out_flat = workspace_2d_or_empty(
        workspace_opt,
        N * H,
        K,
        log_s_c,
        log_s_c.scalar_type(),
        "selector_cross_head_workspace");
    auto out = out_flat.reshape({N, H, K});
    int threads = 128;
    if (const char* env = std::getenv("VLLM_SPARSE_SELECTOR_CROSS_HEAD_BLOCK_K")) {
        // [CROSS-HEAD-ENV-DOMAIN 2026-07-11 EXT审计·理论可达] 旧逻辑只认 64，
        // 其它值静默保持 128 且跳过 else 臂的 shm 自适应选择 = env 放行域大于
        // 实现路由域（32/打错字全被静默吞）。收窄：显式值必须 ∈ {64,128}，
        // 否则 raise；显式 128 同样受下方统一 shm fail-fast 约束，不再静默
        // 越过 48KB 检查。
        int parsed = std::atoi(env);
        TORCH_CHECK(parsed == 64 || parsed == 128,
                    "VLLM_SPARSE_SELECTOR_CROSS_HEAD_BLOCK_K must be 64 or 128, got '",
                    env, "'");
        threads = parsed;
    } else {
        size_t shm_bytes_128 = sizeof(int) * static_cast<size_t>(H) * 2
            + sizeof(float) * static_cast<size_t>(H) * static_cast<size_t>(128);
        if (shm_bytes_128 > 48 * 1024) {
            threads = 64;
        }
    }
    int block_k = threads;
    dim3 blocks(static_cast<unsigned int>(N), static_cast<unsigned int>((K + block_k - 1) / block_k));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    size_t shm_bytes = sizeof(int) * static_cast<size_t>(H) * 2 + sizeof(float) * static_cast<size_t>(H) * static_cast<size_t>(block_k);
    // [CROSS-HEAD-SHM-FAILFAST 2026-07-11 EXT审计·理论可达] 与 run_soft_nms
    // 同款：超 48KB 的 launch 会静默失败并返回 workspace 旧值。fail-fast。
    constexpr size_t kMaxDynamicShmemBytes = 48 * 1024;
    TORCH_CHECK(shm_bytes <= kMaxDynamicShmemBytes,
                "run_cross_head: H=", H, " block_k=", block_k, " needs ",
                shm_bytes,
                " bytes dynamic shared memory (>48KB SM default limit)");

    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, log_s_c.scalar_type(), "cross_head_mutex_bounds", [&] {
        if (block_k == 128) {
            cross_head_mutex_bounds_kernel<scalar_t, 128><<<blocks, block_k, shm_bytes, stream>>>(
                log_s_c.data_ptr<scalar_t>(),
                lo_i32.data_ptr<int32_t>(),
                hi_i32.data_ptr<int32_t>(),
                static_cast<int>(N),
                static_cast<int>(H),
                static_cast<int>(K),
                static_cast<int>(log_s_c.stride(0)),
                static_cast<int>(log_s_c.stride(1)),
                static_cast<int>(log_s_c.stride(2)),
                static_cast<int>(lo_i32.stride(0)),
                static_cast<int>(lo_i32.stride(1)),
                static_cast<int>(hi_i32.stride(0)),
                static_cast<int>(hi_i32.stride(1)),
                static_cast<float>(alpha_cross),
                static_cast<float>(temperature),
                static_cast<float>(eps),
                out.data_ptr<scalar_t>());
        } else {
            cross_head_mutex_bounds_kernel<scalar_t, 64><<<blocks, block_k, shm_bytes, stream>>>(
                log_s_c.data_ptr<scalar_t>(),
                lo_i32.data_ptr<int32_t>(),
                hi_i32.data_ptr<int32_t>(),
                static_cast<int>(N),
                static_cast<int>(H),
                static_cast<int>(K),
                static_cast<int>(log_s_c.stride(0)),
                static_cast<int>(log_s_c.stride(1)),
                static_cast<int>(log_s_c.stride(2)),
                static_cast<int>(lo_i32.stride(0)),
                static_cast<int>(lo_i32.stride(1)),
                static_cast<int>(hi_i32.stride(0)),
                static_cast<int>(hi_i32.stride(1)),
                static_cast<float>(alpha_cross),
                static_cast<float>(temperature),
                static_cast<float>(eps),
                out.data_ptr<scalar_t>());
        }
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor run_soft_nms_cross_head(
    torch::Tensor log_s,
    torch::Tensor token_lo,
    torch::Tensor token_hi,
    int64_t window,
    double soft_alpha,
    double alpha_cross,
    double temperature,
    double eps,
    c10::optional<torch::Tensor> workspace_opt) {
    auto log_s_c = log_s.contiguous();
    auto lo_i32 = token_lo.to(torch::kInt32).contiguous();
    auto hi_i32 = token_hi.to(torch::kInt32).contiguous();

    int64_t N = log_s_c.size(0);
    int64_t H = log_s_c.size(1);
    int64_t K = log_s_c.size(2);
    if (N == 0 || H == 0 || K == 0) {
        return log_s_c;
    }
    if (H <= 2) {
        return run_soft_nms(log_s_c, lo_i32, hi_i32, window, soft_alpha, workspace_opt);
    }

    int win = static_cast<int>(window);
    if (win < 1) {
        win = 1;
    }
    if (win > K) {
        win = static_cast<int>(K);
    }

    int block_k = 128;
    constexpr size_t kMaxDynamicShmemBytes = 48 * 1024;
    if (fused_nms_cross_shm_bytes(H, block_k, win) > kMaxDynamicShmemBytes) {
        block_k = 64;
    }
    if (fused_nms_cross_shm_bytes(H, block_k, win) > kMaxDynamicShmemBytes) {
        block_k = 32;
    }
    if (fused_nms_cross_shm_bytes(H, block_k, win) > kMaxDynamicShmemBytes) {
        auto nms = run_soft_nms(log_s_c, lo_i32, hi_i32, win, soft_alpha, c10::nullopt);
        return run_cross_head(nms, lo_i32, hi_i32, alpha_cross, temperature, eps, workspace_opt);
    }

    auto out_flat = workspace_2d_or_empty(
        workspace_opt,
        N * H,
        K,
        log_s_c,
        log_s_c.scalar_type(),
        "selector_soft_nms_cross_workspace");
    auto out = out_flat.reshape({N, H, K});
    dim3 blocks(static_cast<unsigned int>(N), static_cast<unsigned int>((K + block_k - 1) / block_k));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    size_t shm_bytes = fused_nms_cross_shm_bytes(H, block_k, win);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, log_s_c.scalar_type(), "soft_nms_cross_head_bounds", [&] {
        if (block_k == 128) {
            soft_nms_cross_head_bounds_kernel<scalar_t, 128><<<blocks, 128, shm_bytes, stream>>>(
                log_s_c.data_ptr<scalar_t>(),
                lo_i32.data_ptr<int32_t>(),
                hi_i32.data_ptr<int32_t>(),
                static_cast<int>(N),
                static_cast<int>(H),
                static_cast<int>(K),
                static_cast<int>(log_s_c.stride(0)),
                static_cast<int>(log_s_c.stride(1)),
                static_cast<int>(log_s_c.stride(2)),
                static_cast<int>(lo_i32.stride(0)),
                static_cast<int>(lo_i32.stride(1)),
                static_cast<int>(hi_i32.stride(0)),
                static_cast<int>(hi_i32.stride(1)),
                win,
                static_cast<float>(soft_alpha),
                static_cast<float>(alpha_cross),
                static_cast<float>(temperature),
                static_cast<float>(eps),
                out.data_ptr<scalar_t>());
        } else if (block_k == 64) {
            soft_nms_cross_head_bounds_kernel<scalar_t, 64><<<blocks, 64, shm_bytes, stream>>>(
                log_s_c.data_ptr<scalar_t>(),
                lo_i32.data_ptr<int32_t>(),
                hi_i32.data_ptr<int32_t>(),
                static_cast<int>(N),
                static_cast<int>(H),
                static_cast<int>(K),
                static_cast<int>(log_s_c.stride(0)),
                static_cast<int>(log_s_c.stride(1)),
                static_cast<int>(log_s_c.stride(2)),
                static_cast<int>(lo_i32.stride(0)),
                static_cast<int>(lo_i32.stride(1)),
                static_cast<int>(hi_i32.stride(0)),
                static_cast<int>(hi_i32.stride(1)),
                win,
                static_cast<float>(soft_alpha),
                static_cast<float>(alpha_cross),
                static_cast<float>(temperature),
                static_cast<float>(eps),
                out.data_ptr<scalar_t>());
        } else {
            soft_nms_cross_head_bounds_kernel<scalar_t, 32><<<blocks, 32, shm_bytes, stream>>>(
                log_s_c.data_ptr<scalar_t>(),
                lo_i32.data_ptr<int32_t>(),
                hi_i32.data_ptr<int32_t>(),
                static_cast<int>(N),
                static_cast<int>(H),
                static_cast<int>(K),
                static_cast<int>(log_s_c.stride(0)),
                static_cast<int>(log_s_c.stride(1)),
                static_cast<int>(log_s_c.stride(2)),
                static_cast<int>(lo_i32.stride(0)),
                static_cast<int>(lo_i32.stride(1)),
                static_cast<int>(hi_i32.stride(0)),
                static_cast<int>(hi_i32.stride(1)),
                win,
                static_cast<float>(soft_alpha),
                static_cast<float>(alpha_cross),
                static_cast<float>(temperature),
                static_cast<float>(eps),
                out.data_ptr<scalar_t>());
        }
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor run_log_s(
    torch::Tensor scores,
    torch::optional<torch::Tensor> denom,
    torch::Tensor row_lo,
    torch::Tensor row_hi,
    torch::Tensor token_lo,
    torch::Tensor token_hi,
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
	    bool lambda_soft,
	    c10::optional<torch::Tensor> log_r_cache_opt,
	    c10::optional<torch::Tensor> workspace_opt) {
    auto scores_c = ensure_contig(scores);
    auto row_lo_c = ensure_int32_contig(row_lo);
    auto row_hi_c = ensure_int32_contig(row_hi);
    auto token_lo_c = ensure_int32_contig(token_lo);
    auto token_hi_c = ensure_int32_contig(token_hi);
    auto key_norms_c = ensure_contig(key_norms);

    int64_t M = scores_c.size(0);
    int64_t R = scores_c.size(1);
    int64_t K = scores_c.size(2);

    int block_k = 128;
    if (K <= 128) {
        block_k = 128;
    } else if (K <= 4096) {
        block_k = 256;
    } else {
        block_k = 512;
    }
    int num_blocks = (static_cast<int>(K) + block_k - 1) / block_k;
    int num_blocks_bucket = num_blocks;

    TORCH_CHECK(row_lo_c.size(0) == M && row_lo_c.size(1) == R, "row_lo shape mismatch");
    TORCH_CHECK(row_hi_c.size(0) == M && row_hi_c.size(1) == R, "row_hi shape mismatch");
    TORCH_CHECK(token_lo_c.numel() == M, "token_lo shape mismatch");
    TORCH_CHECK(token_hi_c.numel() == M, "token_hi shape mismatch");
    TORCH_CHECK(key_norms_c.size(0) == M && key_norms_c.size(1) == K, "key_norms shape mismatch");
    TORCH_CHECK(R <= 128, "rows (R) must be <=128");

    auto out = workspace_2d_or_empty(
        workspace_opt,
        M,
        K,
        scores_c,
        torch::kFloat32,
        "selector_log_s_workspace");
    torch::Tensor log_r_cache;
    float* log_r_cache_ptr = nullptr;
    if (log_r_cache_opt.has_value() && log_r_cache_opt.value().defined()) {
        log_r_cache = log_r_cache_opt.value();
        TORCH_CHECK(log_r_cache.device() == scores_c.device(), "log_r_cache device mismatch");
        TORCH_CHECK(log_r_cache.scalar_type() == torch::kFloat32, "log_r_cache must be float32");
        TORCH_CHECK(log_r_cache.is_contiguous(), "log_r_cache must be contiguous");
        TORCH_CHECK(log_r_cache.dim() == 2 && log_r_cache.size(0) == M && log_r_cache.size(1) == K,
                    "log_r_cache shape mismatch");
        log_r_cache_ptr = log_r_cache.data_ptr<float>();
    } else if (const char* env = std::getenv("VLLM_SPARSE_SELECTOR_LOGS_CACHE_R")) {
        if (std::atoi(env) == 1) {
            log_r_cache = torch::empty({M, K}, scores_c.options().dtype(torch::kFloat32));
            log_r_cache_ptr = log_r_cache.data_ptr<float>();
        }
    }

    int threads = 256;
    if (const char* env = std::getenv("VLLM_SPARSE_SELECTOR_LOGS_THREADS")) {
        int parsed = std::atoi(env);
        // [THREADS-WARP-ALIGN] non-multiple-of-32 blockDim would make the
        // full-mask __shfl_down_sync reductions UB in the last warp.
        parsed &= ~31;
        if (parsed >= 64 && parsed <= 1024) {
            threads = parsed;
        }
    } else {
        if (K >= 8192) {
            threads = (R <= 16) ? 768 : 512;
        }
    }
    if (threads < block_k) {
        threads = block_k;
    }
    // [LSE-SCAN-DOMAIN-FIX] P0-1:threads > block_k 时 tid∈[block_k,threads)
    // 与下一 block 迭代重叠(k = block_start*block_k + tid 越域),半数 token
    // 的 expf/count 双计 → lse 偏大污染 topk。扫描域必须=线程域,threads
    // 恒 == block_k(K≥8192 档原 768>512 踩雷;K≤4096 档本就相等不受影响)。
    if (threads > block_k) {
        threads = block_k;
    }
    const dim3 blocks(M);
    size_t shm_size = sizeof(float) * R + sizeof(int) * R + sizeof(int) * R + sizeof(int) * R
        + sizeof(int) * 2 + sizeof(float) * 2;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (denom.has_value()) {
        auto denom_f = denom.value();
        if (denom_f.scalar_type() != torch::kFloat32) {
            denom_f = denom_f.to(torch::kFloat32);
        }
        TORCH_CHECK(denom_f.dim() == 2, "denom must be [M, R]");
        TORCH_CHECK(denom_f.size(0) == M && denom_f.size(1) == R, "denom shape mismatch");
        AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, scores_c.scalar_type(), "fused_log_f_prior_pre_denom", [&] {
            using score_t = scalar_t;
            AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, key_norms_c.scalar_type(), "fused_log_f_prior_pre_denom_key_norms", [&] {
                using key_norm_t = scalar_t;
                fused_log_f_prior_kernel<score_t, key_norm_t><<<blocks, threads, shm_size, stream>>>(
                    scores_c.data_ptr<score_t>(),
                    denom_f.data_ptr<float>(),
                    row_lo_c.data_ptr<int32_t>(),
                    row_hi_c.data_ptr<int32_t>(),
                    token_lo_c.data_ptr<int32_t>(),
                    token_hi_c.data_ptr<int32_t>(),
                    key_norms_c.data_ptr<key_norm_t>(),
                    static_cast<int>(M),
                    static_cast<int>(R),
                    static_cast<int>(K),
                    block_k,
                    num_blocks_bucket,
                    static_cast<int>(scores_c.stride(0)),
                    static_cast<int>(scores_c.stride(1)),
                    static_cast<int>(scores_c.stride(2)),
                    static_cast<int>(denom_f.stride(0)),
                    static_cast<int>(denom_f.stride(1)),
                    static_cast<int>(row_lo_c.stride(0)),
                    static_cast<int>(row_lo_c.stride(1)),
                    static_cast<int>(row_hi_c.stride(0)),
                    static_cast<int>(row_hi_c.stride(1)),
                    static_cast<int>(token_lo_c.stride(0)),
                    static_cast<int>(token_hi_c.stride(0)),
                    static_cast<int>(key_norms_c.stride(0)),
                    static_cast<int>(key_norms_c.stride(1)),
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
        });
    } else {
        AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, scores_c.scalar_type(), "fused_log_f_prior_logits", [&] {
            using score_t = scalar_t;
            AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, key_norms_c.scalar_type(), "fused_log_f_prior_logits_key_norms", [&] {
                using key_norm_t = scalar_t;
                fused_log_f_prior_kernel<score_t, key_norm_t><<<blocks, threads, shm_size, stream>>>(
                    scores_c.data_ptr<score_t>(),
                    nullptr,
                    row_lo_c.data_ptr<int32_t>(),
                    row_hi_c.data_ptr<int32_t>(),
                    token_lo_c.data_ptr<int32_t>(),
                    token_hi_c.data_ptr<int32_t>(),
                    key_norms_c.data_ptr<key_norm_t>(),
                    static_cast<int>(M),
                    static_cast<int>(R),
                    static_cast<int>(K),
                    block_k,
                    num_blocks_bucket,
                    static_cast<int>(scores_c.stride(0)),
                    static_cast<int>(scores_c.stride(1)),
                    static_cast<int>(scores_c.stride(2)),
                    0,
                    0,
                    static_cast<int>(row_lo_c.stride(0)),
                    static_cast<int>(row_lo_c.stride(1)),
                    static_cast<int>(row_hi_c.stride(0)),
                    static_cast<int>(row_hi_c.stride(1)),
                    static_cast<int>(token_lo_c.stride(0)),
                    static_cast<int>(token_hi_c.stride(0)),
                    static_cast<int>(key_norms_c.stride(0)),
                    static_cast<int>(key_norms_c.stride(1)),
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
        });
    }
    return out;
}

torch::Tensor post_topk(
    torch::Tensor log_s,
    int64_t k_head,
    int64_t slice_start,
    int64_t slice_end);

// [POST-TOPK-VALUE-SENTINEL 2026-07-07] count-cutoff kernel now ALSO applies
// the value-sentinel: a picked column whose topk VALUE is the finite min_val
// mask (v <= -3.0e38) is an in-row "no valid candidate" pick (row has fewer
// finite candidates than k_eff, e.g. capacity==0 rows -> row_hi==1), and its
// index MUST NOT leak into selected_indices — the writer's sel<0 sentinel
// protocol is the only downstream guard. Rows with enough finite candidates
// are bit-identical to the old kernel (every picked v is finite).
__global__ void post_topk_i64_to_i32_out_kernel(
    const int64_t* __restrict__ topk,
    const float* __restrict__ topk_vals,
    int32_t* __restrict__ out,
    int64_t rows,
    int64_t k_eff,
    int64_t k_head,
    int32_t start) {
    const float MIN_VAL_THRESHOLD = -3.0e38f;
    int64_t total = rows * k_head;
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (; idx < total; idx += stride) {
        int64_t row = idx / k_head;
        int64_t col = idx - row * k_head;
        int32_t value = -1;
        if (col < k_eff) {
            float v = topk_vals[row * k_eff + col];
            if (v > MIN_VAL_THRESHOLD) {
                value = static_cast<int32_t>(topk[row * k_eff + col] + start);
            }
        }
        out[idx] = value;
    }
}

// fa4_selector_fixed_shape_topk: fixed-shape value-sentinel i32-out kernel. k == k_head
// for every row; an output slot is -1 iff its picked topk VALUE is the
// finite min_val sentinel (value <= MIN_VAL_THRESHOLD). This reproduces the
// k_eff<k_head -> -1 tail of the count-cutoff kernel WITHOUT a data-dependent
// k_eff column count: a slice_len<k_head row has exactly (k_head - slice_len)
// sentinel-scored columns -> exactly that many trailing -1 slots, same set,
// same order (ATen sorted=false topk is unchanged on the finite region).
__global__ void post_topk_fixed_shape_i64_to_i32_out_kernel(
    const int64_t* __restrict__ topk,
    const float* __restrict__ topk_vals,
    int32_t* __restrict__ out,
    int64_t rows,
    int64_t k_head,
    int32_t start) {
    const float MIN_VAL_THRESHOLD = -3.0e38f;
    int64_t total = rows * k_head;
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (; idx < total; idx += stride) {
        int32_t value = -1;
        float v = topk_vals[idx];
        if (v > MIN_VAL_THRESHOLD) {
            value = static_cast<int32_t>(topk[idx] + start);
        }
        out[idx] = value;
    }
}

torch::Tensor post_topk(
    torch::Tensor log_s,
    int64_t k_head,
    int64_t slice_start,
    int64_t slice_end,
    c10::optional<torch::Tensor> selected_indices_out_opt) {
    TORCH_CHECK(log_s.dim() == 4, "log_s must be [L,B,H,K]");
    auto scores = log_s;
    int64_t K = scores.size(-1);
    int64_t start = slice_start > 0 ? slice_start : 0;
    int64_t end = slice_end > 0 ? slice_end : K;
    if (end > K) {
        end = K;
    }
    if (start < 0) {
        start = 0;
    }
    bool range_valid = end > start;
    bool full_range = range_valid && start == 0 && end == K;
    // fa4_selector_fixed_shape_topk: fixed-shape (k == k_head) value-sentinel branch.
    if (selector_fixed_shape_topk_enabled()) {
        // Same [start,end) window as the OFF path (variable narrow WIDTH may
        // remain for breaker (a); 256-bucketing is breaker (c)). When the
        // window is shorter than k_head, PAD the tail to exactly k_head
        // columns with the finite min_val sentinel so topk(k_head) is valid
        // and the padded columns sentinel-map to -1.
        const float MIN_VAL = -3.402823466e38f;
        auto fs_scores = scores;
        if (range_valid && !full_range) {
            fs_scores = fs_scores.narrow(-1, start, end - start);
        }
        int64_t fs_start = (range_valid && start > 0) ? start : 0;
        int64_t fs_width = fs_scores.size(-1);
        if (fs_width < k_head) {
            int64_t pad = k_head - fs_width;
            auto pad_shape = fs_scores.sizes().vec();
            pad_shape[pad_shape.size() - 1] = pad;
            auto pad_tensor = torch::full(
                pad_shape, MIN_VAL, fs_scores.options());
            fs_scores = torch::cat({fs_scores.contiguous(), pad_tensor}, -1);
        }
        auto fs_res = fs_scores.topk(k_head, -1, true, false);
        auto fs_vals = std::get<0>(fs_res).contiguous();
        auto fs_idx = std::get<1>(fs_res).contiguous();
        int64_t fs_rows = log_s.size(0) * log_s.size(1) * log_s.size(2);
        if (selected_indices_out_opt.has_value()) {
            auto out = selected_indices_out_opt.value();
            TORCH_CHECK(out.defined(), "selected_indices_out must be defined");
            TORCH_CHECK(out.is_cuda(), "selected_indices_out must be CUDA");
            TORCH_CHECK(out.dtype() == torch::kInt32, "selected_indices_out must be int32");
            TORCH_CHECK(out.is_contiguous(), "selected_indices_out must be contiguous");
            TORCH_CHECK(out.dim() == 4, "selected_indices_out must be [L,B,H,k_head]");
            TORCH_CHECK(out.size(0) == log_s.size(0)
                        && out.size(1) == log_s.size(1)
                        && out.size(2) == log_s.size(2)
                        && out.size(3) == k_head,
                        "selected_indices_out shape mismatch");
            int threads = 256;
            int64_t total = fs_rows * k_head;
            int blocks = static_cast<int>((total + threads - 1) / threads);
            if (blocks > 65535) {
                blocks = 65535;
            }
            if (blocks > 0) {
                auto stream = at::cuda::getCurrentCUDAStream();
                post_topk_fixed_shape_i64_to_i32_out_kernel<<<blocks, threads, 0, stream>>>(
                    fs_idx.data_ptr<int64_t>(),
                    fs_vals.data_ptr<float>(),
                    out.data_ptr<int32_t>(),
                    fs_rows,
                    k_head,
                    static_cast<int32_t>(fs_start));
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            }
            return out;
        }
        // Non-out alloc fallback: mirror the value-sentinel mapping on host
        // ops so the two paths stay byte-identical. Use the SAME threshold
        // (-3.0e38f) as the out-path kernel (post_topk_fixed_shape_*): a
        // picked value is an empty slot iff value <= MIN_VAL_THRESHOLD, so
        // the kernel test (v > MIN_VAL_THRESHOLD) and this mask agree on
        // every value, not just the exact min_val sentinel.
        const float FS_MIN_VAL_THRESHOLD = -3.0e38f;
        auto fs_mask = fs_vals.le(FS_MIN_VAL_THRESHOLD);
        auto fs_out = fs_idx.to(torch::kInt32);
        if (fs_start > 0) {
            fs_out.add_(static_cast<int>(fs_start));
        }
        fs_out.masked_fill_(fs_mask, -1);
        return fs_out;
    }
    if (range_valid && !full_range) {
        scores = scores.narrow(-1, start, end - start);
    }
    int64_t slice_len = range_valid ? (end - start) : K;
    int64_t k_eff = k_head < slice_len ? k_head : slice_len;
    auto topk_res = scores.topk(k_eff, -1, true, false);
    auto topk_vals_f32 = std::get<0>(topk_res).contiguous();
    auto topk_i64 = std::get<1>(topk_res).contiguous();
    if (selected_indices_out_opt.has_value()) {
        auto out = selected_indices_out_opt.value();
        TORCH_CHECK(out.defined(), "selected_indices_out must be defined");
        TORCH_CHECK(out.is_cuda(), "selected_indices_out must be CUDA");
        TORCH_CHECK(out.dtype() == torch::kInt32, "selected_indices_out must be int32");
        TORCH_CHECK(out.is_contiguous(), "selected_indices_out must be contiguous");
        TORCH_CHECK(out.dim() == 4, "selected_indices_out must be [L,B,H,k_head]");
        TORCH_CHECK(out.size(0) == log_s.size(0)
                    && out.size(1) == log_s.size(1)
                    && out.size(2) == log_s.size(2)
                    && out.size(3) == k_head,
                    "selected_indices_out shape mismatch");
        int64_t rows = log_s.size(0) * log_s.size(1) * log_s.size(2);
        int threads = 256;
        int64_t total = rows * k_head;
        int blocks = static_cast<int>((total + threads - 1) / threads);
        if (blocks > 65535) {
            blocks = 65535;
        }
        if (blocks > 0) {
            auto stream = at::cuda::getCurrentCUDAStream();
            post_topk_i64_to_i32_out_kernel<<<blocks, threads, 0, stream>>>(
                topk_i64.data_ptr<int64_t>(),
                topk_vals_f32.data_ptr<float>(),
                out.data_ptr<int32_t>(),
                rows,
                k_eff,
                k_head,
                static_cast<int32_t>(range_valid && start > 0 ? start : 0));
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        return out;
    }
    auto topk = topk_i64.to(torch::kInt32);
    if (range_valid && start > 0) {
        topk.add_(static_cast<int>(start));
    }
    // [POST-TOPK-VALUE-SENTINEL] non-out fallback mirrors the out-path kernel:
    // min_val-masked picks map to -1 (same -3.0e38 threshold), keeping the two
    // paths byte-identical.
    topk.masked_fill_(topk_vals_f32.le(-3.0e38f), -1);
    if (k_eff < k_head) {
        auto out = torch::full(
            {topk.size(0), topk.size(1), topk.size(2), k_head},
            -1,
            topk.options());
        if (k_eff > 0) {
            out.narrow(-1, 0, k_eff).copy_(topk);
        }
        return out;
    }
    return topk;
}

torch::Tensor post_topk(
    torch::Tensor log_s,
    int64_t k_head,
    int64_t slice_start,
    int64_t slice_end) {
    return post_topk(log_s, k_head, slice_start, slice_end, c10::nullopt);
}

std::vector<torch::Tensor> selector_pipeline_logits_topk_cuda(
    torch::Tensor scores,
    torch::Tensor row_lo,
    torch::Tensor row_hi,
    torch::Tensor key_norms,
    torch::Tensor token_lo,
    torch::Tensor token_hi,
    int64_t k_head,
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
    bool lambda_soft,
    int64_t nms_window,
    double soft_alpha,
    double alpha_cross,
    double temperature,
    double cross_eps,
    int64_t slice_start,
    int64_t slice_end,
    c10::optional<torch::Tensor> log_r_cache_opt,
    c10::optional<torch::Tensor> selected_indices_out_opt,
    c10::optional<torch::Tensor> workspace_a_opt,
    c10::optional<torch::Tensor> workspace_b_opt) {
    auto scores_c = ensure_contig(scores);
    TORCH_CHECK(scores_c.dim() == 6, "scores must be [L,B,H,G,W,K]");
    int64_t L = scores_c.size(0);
    int64_t B = scores_c.size(1);
    int64_t H = scores_c.size(2);
    int64_t G = scores_c.size(3);
    int64_t W = scores_c.size(4);
    int64_t K = scores_c.size(5);
    int64_t M = L * B * H;
    int64_t R = G * W;

    auto scores_flat = scores_c.reshape({M, R, K});
    auto row_lo_flat = ensure_int32_contig(row_lo).reshape({M, R});
    auto row_hi_flat = ensure_int32_contig(row_hi).reshape({M, R});
    auto key_norms_flat = ensure_contig(key_norms).reshape({M, K});
    auto token_lo_flat = ensure_int32_contig(token_lo).reshape({L * B, H});
    auto token_hi_flat = ensure_int32_contig(token_hi).reshape({L * B, H});
    auto token_lo_flat_full = token_lo_flat.reshape({M});
    auto token_hi_flat_full = token_hi_flat.reshape({M});
    const bool do_profile = pipeline_cpu_profile_enabled();
    int64_t t0 = 0;
    int64_t t1 = 0;
    int64_t t2 = 0;
    int64_t t3 = 0;
    int64_t t4 = 0;
    if (do_profile) {
        t0 = now_us();
    }
    auto log_s_flat = run_log_s(
        scores_flat,
        torch::nullopt,
        row_lo_flat,
        row_hi_flat,
        token_lo_flat_full,
        token_hi_flat_full,
        key_norms_flat,
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
        lambda_soft,
        log_r_cache_opt,
        workspace_a_opt);
    if (do_profile) {
        t1 = now_us();
    }

    auto log_s = log_s_flat.reshape({L * B, H, K});
    auto token_lo_cross = token_lo_flat;
    auto token_hi_cross = token_hi_flat;

    torch::Tensor log_s_cross;
    if (selector_fuse_nms_cross_enabled()) {
        log_s_cross = run_soft_nms_cross_head(
            log_s,
            token_lo_flat,
            token_hi_flat,
            nms_window,
            soft_alpha,
            alpha_cross,
            temperature,
            cross_eps,
            workspace_b_opt);
        if (do_profile) {
            t2 = now_us();
            t3 = t2;
        }
    } else {
        auto log_s_nms = run_soft_nms(log_s, token_lo_flat, token_hi_flat, nms_window, soft_alpha, workspace_b_opt);
        if (do_profile) {
            t2 = now_us();
        }
        log_s_cross = run_cross_head(log_s_nms, token_lo_cross, token_hi_cross, alpha_cross, temperature, cross_eps, workspace_a_opt);
        if (do_profile) {
            t3 = now_us();
        }
    }
    auto log_s_full = log_s_cross.reshape({L, B, H, K});
    auto idx = post_topk(
        log_s_full,
        k_head,
        slice_start,
        slice_end,
        selected_indices_out_opt);
    if (do_profile) {
        t4 = now_us();
        int64_t us_log_s = (t1 - t0);
        int64_t us_nms = (t2 - t1);
        int64_t us_cross = (t3 - t2);
        int64_t us_topk = (t4 - t3);
        int64_t us_total = (t4 - t0);
        int64_t start = slice_start > 0 ? slice_start : 0;
        int64_t end = slice_end > 0 ? slice_end : K;
        if (end > K) {
            end = K;
        }
        if (start < 0) {
            start = 0;
        }
        int64_t slice_len = end > start ? (end - start) : K;
        int64_t k_eff = k_head < slice_len ? k_head : slice_len;
        pipeline_cpu_profile_write(
            "logits",
            L,
            B,
            H,
            G,
            W,
            K,
            k_head,
            start,
            end,
            k_eff,
            us_log_s,
            us_nms,
            us_cross,
            us_topk,
            us_total);
        if (us_topk >= pipeline_cpu_profile_outlier_us()) {
            pipeline_cpu_profile_outlier_write(
                "logits",
                L,
                B,
                H,
                G,
                W,
                K,
                k_head,
                start,
                end,
                k_eff,
                us_topk,
                us_total,
                log_s_full);
        }
    }
    return {idx};
}

std::vector<torch::Tensor> selector_pipeline_pre_denom_topk_cuda(
    torch::Tensor scores,
    torch::Tensor denom,
    torch::Tensor row_lo,
    torch::Tensor row_hi,
    torch::Tensor key_norms,
    torch::Tensor token_lo,
    torch::Tensor token_hi,
    int64_t k_head,
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
    bool lambda_soft,
    int64_t nms_window,
    double soft_alpha,
    double alpha_cross,
    double temperature,
    double cross_eps,
    int64_t slice_start,
    int64_t slice_end,
    c10::optional<torch::Tensor> log_r_cache_opt,
    c10::optional<torch::Tensor> selected_indices_out_opt,
    c10::optional<torch::Tensor> workspace_a_opt,
    c10::optional<torch::Tensor> workspace_b_opt) {
    auto scores_c = ensure_contig(scores);
    auto denom_c = denom;
    TORCH_CHECK(scores_c.dim() == 6, "scores must be [L,B,H,G,W,K]");
    TORCH_CHECK(denom_c.dim() == 5, "denom must be [L,B,H,G,W]");
    int64_t L = scores_c.size(0);
    int64_t B = scores_c.size(1);
    int64_t H = scores_c.size(2);
    int64_t G = scores_c.size(3);
    int64_t W = scores_c.size(4);
    int64_t K = scores_c.size(5);
    int64_t M = L * B * H;
    int64_t R = G * W;

    auto scores_flat = scores_c.reshape({M, R, K});
    auto denom_flat = denom_c.reshape({M, R});
    auto row_lo_flat = ensure_int32_contig(row_lo).reshape({M, R});
    auto row_hi_flat = ensure_int32_contig(row_hi).reshape({M, R});
    auto key_norms_flat = ensure_contig(key_norms).reshape({M, K});
    auto token_lo_flat = ensure_int32_contig(token_lo).reshape({L * B, H});
    auto token_hi_flat = ensure_int32_contig(token_hi).reshape({L * B, H});
    auto token_lo_flat_full = token_lo_flat.reshape({M});
    auto token_hi_flat_full = token_hi_flat.reshape({M});
    const bool do_profile = pipeline_cpu_profile_enabled();
    int64_t t0 = 0;
    int64_t t1 = 0;
    int64_t t2 = 0;
    int64_t t3 = 0;
    int64_t t4 = 0;
    if (do_profile) {
        t0 = now_us();
    }
    auto log_s_flat = run_log_s(
        scores_flat,
        denom_flat,
        row_lo_flat,
        row_hi_flat,
        token_lo_flat_full,
        token_hi_flat_full,
        key_norms_flat,
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
        lambda_soft,
        log_r_cache_opt,
        workspace_a_opt);
    if (do_profile) {
        t1 = now_us();
    }

    auto log_s = log_s_flat.reshape({L * B, H, K});
    auto token_lo_cross = token_lo_flat;
    auto token_hi_cross = token_hi_flat;
    torch::Tensor log_s_cross;
    if (selector_fuse_nms_cross_enabled()) {
        log_s_cross = run_soft_nms_cross_head(
            log_s,
            token_lo_flat,
            token_hi_flat,
            nms_window,
            soft_alpha,
            alpha_cross,
            temperature,
            cross_eps,
            workspace_b_opt);
        if (do_profile) {
            t2 = now_us();
            t3 = t2;
        }
    } else {
        auto log_s_nms = run_soft_nms(log_s, token_lo_flat, token_hi_flat, nms_window, soft_alpha, workspace_b_opt);
        if (do_profile) {
            t2 = now_us();
        }
        log_s_cross = run_cross_head(log_s_nms, token_lo_cross, token_hi_cross, alpha_cross, temperature, cross_eps, workspace_a_opt);
        if (do_profile) {
            t3 = now_us();
        }
    }
    auto log_s_full = log_s_cross.reshape({L, B, H, K});
    auto idx = post_topk(
        log_s_full,
        k_head,
        slice_start,
        slice_end,
        selected_indices_out_opt);
    if (do_profile) {
        t4 = now_us();
        int64_t us_log_s = (t1 - t0);
        int64_t us_nms = (t2 - t1);
        int64_t us_cross = (t3 - t2);
        int64_t us_topk = (t4 - t3);
        int64_t us_total = (t4 - t0);
        int64_t start = slice_start > 0 ? slice_start : 0;
        int64_t end = slice_end > 0 ? slice_end : K;
        if (end > K) {
            end = K;
        }
        if (start < 0) {
            start = 0;
        }
        int64_t slice_len = end > start ? (end - start) : K;
        int64_t k_eff = k_head < slice_len ? k_head : slice_len;
        pipeline_cpu_profile_write(
            "pre_denom",
            L,
            B,
            H,
            G,
            W,
            K,
            k_head,
            start,
            end,
            k_eff,
            us_log_s,
            us_nms,
            us_cross,
            us_topk,
            us_total);
        if (us_topk >= pipeline_cpu_profile_outlier_us()) {
            pipeline_cpu_profile_outlier_write(
                "pre_denom",
                L,
                B,
                H,
                G,
                W,
                K,
                k_head,
                start,
                end,
                k_eff,
                us_topk,
                us_total,
                log_s_full);
        }
    }
    return {idx};
}

// ============================================================================
// LAZY BOUNDS OPTIMIZATION: avoid 5D expand, compute bounds inline
// ============================================================================
// Key insight: row_lo[m, r] = head_sink[m] for ALL r (constant per row)
// Instead of creating 5D row_lo [L,B,H,G,W] and flattening to [M,R],
// we create head_sink [M] and use broadcast (stride=0).
// This eliminates the expensive 5D expand in compute_preproc_bounds.

std::vector<torch::Tensor> selector_pipeline_logits_topk_lazy_fused(
    torch::Tensor scores,
    torch::Tensor kv_lengths,
    torch::Tensor key_norms,
    c10::optional<torch::Tensor> seq_full_opt,
    int64_t num_kv_heads,
    int64_t num_queries_per_kv,
    int64_t kv_len_total,
    int64_t sink_cfg,
    int64_t recent_cfg,
    int64_t block_size,
    int64_t k_head,
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
    bool lambda_soft,
    int64_t nms_window,
    double soft_alpha,
    double alpha_cross,
    double temperature,
    double cross_eps,
    int64_t slice_start,
    int64_t slice_end) {

    TORCH_CHECK(scores.defined(), "scores must be defined");
    TORCH_CHECK(kv_lengths.defined(), "kv_lengths must be defined");

    auto scores_sizes = scores.sizes();
    int64_t L = scores_sizes[0];
    int64_t B = scores_sizes[1];
    int64_t HG = scores_sizes[2];
    int64_t W = scores_sizes[3];
    int64_t K = scores_sizes[4];
    int64_t H = num_kv_heads;
    int64_t G = num_queries_per_kv;
    int64_t M = L * B * H;
    int64_t R = G * W;

    // ========================================
    // LAZY BOUNDS: minimal computation
    // ========================================
    // Instead of compute_preproc_bounds which creates 5D tensors,
    // we compute only what's needed:
    // 1. head_sink [L, B, H] - for row_lo (broadcast with stride=0)
    // 2. recent_start [L, B, H] - for row_hi upper bound
    // 3. kv_lengths reshaped [L, B, H, G] - for row_hi per-query

    // Reshape kv_lengths: [L, B, H_total] -> [L, B, H, G]
    auto kv_lengths_kv = kv_lengths.reshape({L, B, H, G}).to(at::kInt);

    // kv_len_head = max over G dimension: [L, B, H]
    auto kv_len_head = std::get<0>(kv_lengths_kv.max(-1));
    kv_len_head = at::clamp_max(kv_len_head, kv_len_total);

    // head_sink = min(kv_len_head, sink_cfg): [L, B, H]
    auto head_sink = at::clamp_max(kv_len_head, std::max<int64_t>(0, sink_cfg));

    // recent_start computation: [L, B, H]
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

    // allowed_lengths = max(recent_start - head_sink, 0): [L, B, H]
    auto allowed_lengths = at::clamp_min(recent_start - head_sink, 0);

    // ========================================
    // BUILD row_lo and row_hi with broadcast (stride=0)
    // ========================================
    // row_lo[m, r] = head_sink[m] for all r
    // Use expand with stride=0 to avoid memory allocation
    auto head_sink_flat = head_sink.reshape({M, 1});
    auto row_lo_broadcast = head_sink_flat.expand({M, R});  // stride[1] = 0, no alloc

    // row_hi is more complex: row_hi[m, g, w] = min(kv_len[m, g], hi_stair[m, w], recent_start[m])
    // where hi_stair[m, w] = kv_len_head[m] - (W - 1 - w)
    // kv_lengths_kv is [L, B, H, G], reshape to [M, G]
    auto kv_len_flat = kv_lengths_kv.reshape({M, G});

    // Expand kv_len to [M, G, W] then reshape to [M, R]
    auto kv_len_expanded = kv_len_flat.unsqueeze(-1).expand({M, G, W}).reshape({M, R});

    // hi_stair calculation (window stair pattern)
    // hi_stair[m, g, w] = kv_len_head[m] - (W - 1 - w)
    // tail_offsets = [W-1, W-2, ..., 1, 0]
    torch::Tensor row_hi_pre;
    if (W > 0) {
        auto options = kv_lengths_kv.options();
        auto window_idx = at::arange(W, options);  // [0, 1, ..., W-1]
        auto tail_offsets = (W - 1) - window_idx;  // [W-1, W-2, ..., 0]
        tail_offsets = tail_offsets.view({1, 1, W});  // [1, 1, W]

        // kv_len_head [M] -> [M, 1, 1]
        auto kv_len_head_flat = kv_len_head.reshape({M, 1, 1});
        // hi_stair = kv_len_head - tail_offsets -> [M, 1, W]
        auto hi_stair = kv_len_head_flat - tail_offsets;
        // Expand to [M, G, W]
        hi_stair = hi_stair.expand({M, G, W}).reshape({M, R});

        // row_hi = min(kv_len_expanded, hi_stair)
        row_hi_pre = at::minimum(kv_len_expanded, hi_stair);
    } else {
        row_hi_pre = kv_len_expanded;
    }

    // recent_start_flat [M] -> [M, 1] -> broadcast to [M, R]
    auto recent_start_flat = recent_start.reshape({M, 1});
    auto recent_exp_broadcast = recent_start_flat.expand({M, R});  // stride[1] = 0

    // row_hi = min(row_hi_pre, recent_exp_broadcast)
    row_hi_pre = at::minimum(row_hi_pre, recent_exp_broadcast);
    auto row_hi_flat = at::clamp(row_hi_pre, 0, kv_len_total);

    // ========================================
    // CALL PIPELINE (same as before)
    // ========================================
    auto scores_kv = scores.reshape({L, B, H, G, W, K});
    auto scores_flat = scores_kv.reshape({M, R, K});
    auto key_norms_flat = ensure_contig(key_norms).reshape({M, K});

    // token_lo = head_sink, token_hi = recent_start
    auto token_lo_flat = head_sink.reshape({M});
    auto token_hi_flat = recent_start.reshape({M});

    // row_lo needs to be int32 contiguous (broadcast expand is not contiguous)
    // But the kernel uses strides! We can pass the broadcast tensor if we handle strides correctly.
    // Actually, ensure_int32_contig will make it contiguous, defeating the purpose.
    // So we need to pass stride explicitly or make it contiguous.
    // For simplicity, let's just make row_lo contiguous here (it's small: M*R int32)
    auto row_lo_flat = row_lo_broadcast.to(at::kInt).contiguous();

    // row_hi is already computed, just ensure int32 contiguous
    auto row_hi_flat_c = ensure_int32_contig(row_hi_flat);

    // Run log_s kernel
    auto log_s_flat = run_log_s(
        ensure_contig(scores_flat),
        torch::nullopt,
        row_lo_flat,
        row_hi_flat_c,
        ensure_int32_contig(token_lo_flat),
        ensure_int32_contig(token_hi_flat),
        key_norms_flat,
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
        lambda_soft,
        c10::nullopt,
        c10::nullopt);

    // Run soft-nms
    auto log_s = log_s_flat.reshape({L * B, H, K});
    auto token_lo_cross = head_sink.reshape({L * B, H});
    auto token_hi_cross = recent_start.reshape({L * B, H});
    auto log_s_nms = run_soft_nms(log_s, ensure_int32_contig(token_lo_cross), ensure_int32_contig(token_hi_cross), nms_window, soft_alpha, c10::nullopt);

    // Run cross-head
    auto log_s_cross = run_cross_head(log_s_nms, ensure_int32_contig(token_lo_cross), ensure_int32_contig(token_hi_cross), alpha_cross, temperature, cross_eps, c10::nullopt);

    // Post-topk
    auto log_s_full = log_s_cross.reshape({L, B, H, K});
    auto idx = post_topk(
        log_s_full,
        k_head,
        slice_start,
        slice_end);

    // Return [selected_indices, head_sink, recent_start, kv_len_head, allowed_lengths]
    return {idx, head_sink, recent_start, kv_len_head, allowed_lengths};
}

// ============================================================================
// WITH_BOUNDS: accept pre-computed bounds from external CUDA kernel
// ============================================================================
// This function accepts bounds computed by bounds_kernel_ext.py or
// bounds_prefill_kernel_ext.py and directly calls the pipeline CUDA kernel,
// skipping all ATen bounds computation. Decode may pass [M, G]; prefill W>1
// passes [M, G*W].
std::vector<torch::Tensor> selector_pipeline_logits_topk_with_bounds_impl(
    torch::Tensor scores,           // [L, B, H_total, W, K]
    torch::Tensor row_lo,           // [M, G] or [M, G*W] pre-computed bounds
    torch::Tensor row_hi,           // [M, G] or [M, G*W] pre-computed bounds
    torch::Tensor key_norms,        // [L, B, H_kv, K]
    torch::Tensor head_sink,        // [L, B, H_kv] - token_lo
    torch::Tensor recent_start,     // [L, B, H_kv] - token_hi
    int64_t num_kv_heads,
    int64_t num_queries_per_kv,
    int64_t k_head,
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
    bool lambda_soft,
    int64_t nms_window,
    double soft_alpha,
    double alpha_cross,
    double temperature,
    double cross_eps,
    int64_t slice_start,
    int64_t slice_end,
    c10::optional<torch::Tensor> log_r_cache_opt,
    c10::optional<torch::Tensor> selected_indices_out_opt,
    c10::optional<torch::Tensor> workspace_a_opt,
    c10::optional<torch::Tensor> workspace_b_opt) {

    TORCH_CHECK(scores.defined(), "scores must be defined");
    TORCH_CHECK(row_lo.defined(), "row_lo must be defined");
    TORCH_CHECK(row_hi.defined(), "row_hi must be defined");

    auto scores_sizes = scores.sizes();
    int64_t L = scores_sizes[0];
    int64_t B = scores_sizes[1];
    int64_t HG = scores_sizes[2];
    int64_t W = scores_sizes[3];
    int64_t K = scores_sizes[4];
    int64_t H = num_kv_heads;
    int64_t G = num_queries_per_kv;
    int64_t M = L * B * H;
    int64_t R = G * W;

    // Reshape scores: [L, B, H_total, W, K] -> [L, B, H_kv, G, W, K]
    auto scores_kv = scores.reshape({L, B, H, G, W, K});

    auto row_lo_i32 = ensure_int32_contig(row_lo);
    auto row_hi_i32 = ensure_int32_contig(row_hi);
    TORCH_CHECK(row_lo_i32.numel() == M * R || row_lo_i32.numel() == M * G,
                "row_lo shape mismatch with scores");
    TORCH_CHECK(row_hi_i32.numel() == M * R || row_hi_i32.numel() == M * G,
                "row_hi shape mismatch with scores");
    torch::Tensor row_lo_final, row_hi_final;
    if (row_lo_i32.numel() == M * R) {
        row_lo_final = row_lo_i32.reshape({M, R});
    } else {
        row_lo_final = row_lo_i32.reshape({M, G}).unsqueeze(-1).expand({M, G, W}).reshape({M, R}).contiguous();
    }
    if (row_hi_i32.numel() == M * R) {
        row_hi_final = row_hi_i32.reshape({M, R});
    } else {
        row_hi_final = row_hi_i32.reshape({M, G}).unsqueeze(-1).expand({M, G, W}).reshape({M, R}).contiguous();
    }

    // token_lo and token_hi from head_sink and recent_start
    // Reshape: [L, B, H] -> [M]
    auto token_lo = head_sink.reshape({M}).to(at::kInt).contiguous();
    auto token_hi = recent_start.reshape({M}).to(at::kInt).contiguous();

    // Key norms: [L, B, H_kv, K] is already correct shape
    auto key_norms_c = ensure_contig(key_norms);

    // Call the pipeline CUDA kernel
    auto idx_result = selector_pipeline_logits_topk_cuda(
        scores_kv,
        row_lo_final,
        row_hi_final,
        key_norms_c,
        token_lo,
        token_hi,
        k_head,
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
        lambda_soft,
        nms_window,
        soft_alpha,
        alpha_cross,
        temperature,
        cross_eps,
        slice_start,
        slice_end,
        log_r_cache_opt,
        selected_indices_out_opt,
        workspace_a_opt,
        workspace_b_opt);

    // Return selected_indices only (caller already has bounds from compute_bounds_decode)
    return {idx_result[0]};
}

std::vector<torch::Tensor> selector_pipeline_logits_topk_with_bounds(
    torch::Tensor scores,
    torch::Tensor row_lo,
    torch::Tensor row_hi,
    torch::Tensor key_norms,
    torch::Tensor head_sink,
    torch::Tensor recent_start,
    int64_t num_kv_heads,
    int64_t num_queries_per_kv,
    int64_t k_head,
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
    bool lambda_soft,
    int64_t nms_window,
    double soft_alpha,
    double alpha_cross,
    double temperature,
    double cross_eps,
    int64_t slice_start,
    int64_t slice_end,
    c10::optional<torch::Tensor> log_r_cache_opt) {
    return selector_pipeline_logits_topk_with_bounds_impl(
        scores, row_lo, row_hi, key_norms, head_sink, recent_start,
        num_kv_heads, num_queries_per_kv, k_head, alpha, eps, gamma,
        prior_weight_l2, prior_weight_pos, prior_pos_power, prior_pos_eta,
        beta, lambda_clip_single, lambda_clip_multi, lambda_tail_kappa,
        lambda_tail_pivot, lambda_soft, nms_window, soft_alpha, alpha_cross,
        temperature, cross_eps, slice_start, slice_end, log_r_cache_opt,
        c10::nullopt, c10::nullopt, c10::nullopt);
}

std::vector<torch::Tensor> selector_pipeline_logits_topk_with_bounds_workspace(
    torch::Tensor scores,
    torch::Tensor row_lo,
    torch::Tensor row_hi,
    torch::Tensor key_norms,
    torch::Tensor head_sink,
    torch::Tensor recent_start,
    int64_t num_kv_heads,
    int64_t num_queries_per_kv,
    int64_t k_head,
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
    bool lambda_soft,
    int64_t nms_window,
    double soft_alpha,
    double alpha_cross,
    double temperature,
    double cross_eps,
    int64_t slice_start,
    int64_t slice_end,
    c10::optional<torch::Tensor> log_r_cache_opt,
    torch::Tensor workspace_a,
    torch::Tensor workspace_b) {
    return selector_pipeline_logits_topk_with_bounds_impl(
        scores, row_lo, row_hi, key_norms, head_sink, recent_start,
        num_kv_heads, num_queries_per_kv, k_head, alpha, eps, gamma,
        prior_weight_l2, prior_weight_pos, prior_pos_power, prior_pos_eta,
        beta, lambda_clip_single, lambda_clip_multi, lambda_tail_kappa,
        lambda_tail_pivot, lambda_soft, nms_window, soft_alpha, alpha_cross,
        temperature, cross_eps, slice_start, slice_end, log_r_cache_opt,
        c10::nullopt, workspace_a, workspace_b);
}

std::vector<torch::Tensor> selector_pipeline_logits_topk_with_bounds_out(
    torch::Tensor scores,
    torch::Tensor row_lo,
    torch::Tensor row_hi,
    torch::Tensor key_norms,
    torch::Tensor head_sink,
    torch::Tensor recent_start,
    int64_t num_kv_heads,
    int64_t num_queries_per_kv,
    int64_t k_head,
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
    bool lambda_soft,
    int64_t nms_window,
    double soft_alpha,
    double alpha_cross,
    double temperature,
    double cross_eps,
    int64_t slice_start,
    int64_t slice_end,
    c10::optional<torch::Tensor> log_r_cache_opt,
    torch::Tensor selected_indices_out) {
    return selector_pipeline_logits_topk_with_bounds_impl(
        scores, row_lo, row_hi, key_norms, head_sink, recent_start,
        num_kv_heads, num_queries_per_kv, k_head, alpha, eps, gamma,
        prior_weight_l2, prior_weight_pos, prior_pos_power, prior_pos_eta,
        beta, lambda_clip_single, lambda_clip_multi, lambda_tail_kappa,
        lambda_tail_pivot, lambda_soft, nms_window, soft_alpha, alpha_cross,
        temperature, cross_eps, slice_start, slice_end, log_r_cache_opt,
        selected_indices_out, c10::nullopt, c10::nullopt);
}

std::vector<torch::Tensor> selector_pipeline_logits_topk_with_bounds_workspace_out(
    torch::Tensor scores,
    torch::Tensor row_lo,
    torch::Tensor row_hi,
    torch::Tensor key_norms,
    torch::Tensor head_sink,
    torch::Tensor recent_start,
    int64_t num_kv_heads,
    int64_t num_queries_per_kv,
    int64_t k_head,
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
    bool lambda_soft,
    int64_t nms_window,
    double soft_alpha,
    double alpha_cross,
    double temperature,
    double cross_eps,
    int64_t slice_start,
    int64_t slice_end,
    c10::optional<torch::Tensor> log_r_cache_opt,
    torch::Tensor workspace_a,
    torch::Tensor workspace_b,
    torch::Tensor selected_indices_out) {
    return selector_pipeline_logits_topk_with_bounds_impl(
        scores, row_lo, row_hi, key_norms, head_sink, recent_start,
        num_kv_heads, num_queries_per_kv, k_head, alpha, eps, gamma,
        prior_weight_l2, prior_weight_pos, prior_pos_power, prior_pos_eta,
        beta, lambda_clip_single, lambda_clip_multi, lambda_tail_kappa,
        lambda_tail_pivot, lambda_soft, nms_window, soft_alpha, alpha_cross,
        temperature, cross_eps, slice_start, slice_end, log_r_cache_opt,
        selected_indices_out, workspace_a, workspace_b);
}

std::vector<torch::Tensor> selector_pipeline_pre_denom_topk_with_bounds_impl(
    torch::Tensor scores,
    torch::Tensor denom,
    torch::Tensor row_lo,
    torch::Tensor row_hi,
    torch::Tensor key_norms,
    torch::Tensor head_sink,
    torch::Tensor recent_start,
    int64_t num_kv_heads,
    int64_t num_queries_per_kv,
    int64_t k_head,
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
    bool lambda_soft,
    int64_t nms_window,
    double soft_alpha,
    double alpha_cross,
    double temperature,
    double cross_eps,
    int64_t slice_start,
    int64_t slice_end,
    c10::optional<torch::Tensor> log_r_cache_opt,
    c10::optional<torch::Tensor> selected_indices_out_opt,
    c10::optional<torch::Tensor> workspace_a_opt,
    c10::optional<torch::Tensor> workspace_b_opt) {

    TORCH_CHECK(scores.defined(), "scores must be defined");
    TORCH_CHECK(denom.defined(), "denom must be defined");
    TORCH_CHECK(row_lo.defined(), "row_lo must be defined");
    TORCH_CHECK(row_hi.defined(), "row_hi must be defined");

    auto scores_sizes = scores.sizes();
    TORCH_CHECK(scores_sizes.size() == 5, "scores must be [L, B, H_total, W, K]");
    int64_t L = scores_sizes[0];
    int64_t B = scores_sizes[1];
    int64_t HG = scores_sizes[2];
    int64_t W = scores_sizes[3];
    int64_t K = scores_sizes[4];
    int64_t H = num_kv_heads;
    int64_t G = num_queries_per_kv;
    int64_t M = L * B * H;
    int64_t R = G * W;

    TORCH_CHECK(
        W == 1,
        "selector_pipeline_pre_denom_topk_with_bounds only supports W=1 (decode bounds-first path)");
    TORCH_CHECK(HG == H * G, "scores head dimension mismatch with num_kv_heads * num_queries_per_kv");

    auto scores_kv = scores.reshape({L, B, H, G, W, K});

    auto row_lo_i32 = ensure_int32_contig(row_lo);
    auto row_hi_i32 = ensure_int32_contig(row_hi);
    TORCH_CHECK(row_lo_i32.numel() == M * R, "row_lo shape mismatch with scores");
    TORCH_CHECK(row_hi_i32.numel() == M * R, "row_hi shape mismatch with scores");
    auto row_lo_final = row_lo_i32.reshape({M, R});
    auto row_hi_final = row_hi_i32.reshape({M, R});

    torch::Tensor denom_kv;
    if (denom.dim() == 3) {
        TORCH_CHECK(
            denom.size(0) == L && denom.size(1) == B && denom.size(2) == HG,
            "denom must be [L, B, H_total] when dim==3");
        denom_kv = denom.reshape({L, B, H, G, 1});
    } else if (denom.dim() == 4) {
        TORCH_CHECK(
            denom.size(0) == L && denom.size(1) == B && denom.size(2) == H && denom.size(3) == G,
            "denom must be [L, B, H_kv, G] when dim==4");
        denom_kv = denom.unsqueeze(-1);
    } else if (denom.dim() == 5) {
        TORCH_CHECK(
            denom.size(0) == L && denom.size(1) == B && denom.size(2) == H && denom.size(3) == G && denom.size(4) == 1,
            "denom must be [L, B, H_kv, G, 1] when dim==5");
        denom_kv = denom;
    } else {
        TORCH_CHECK(false, "denom must be [L,B,H_total] or [L,B,H_kv,G] or [L,B,H_kv,G,1]");
    }

    auto token_lo = ensure_int32_contig(head_sink).reshape({M});
    auto token_hi = ensure_int32_contig(recent_start).reshape({M});
    auto key_norms_c = ensure_contig(key_norms);

    auto idx_result = selector_pipeline_pre_denom_topk_cuda(
        scores_kv,
        denom_kv,
        row_lo_final,
        row_hi_final,
        key_norms_c,
        token_lo,
        token_hi,
        k_head,
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
        lambda_soft,
        nms_window,
        soft_alpha,
        alpha_cross,
        temperature,
        cross_eps,
        slice_start,
        slice_end,
        log_r_cache_opt,
        selected_indices_out_opt,
        workspace_a_opt,
        workspace_b_opt);

    return {idx_result[0]};
}

std::vector<torch::Tensor> selector_pipeline_pre_denom_topk_with_bounds(
    torch::Tensor scores,
    torch::Tensor denom,
    torch::Tensor row_lo,
    torch::Tensor row_hi,
    torch::Tensor key_norms,
    torch::Tensor head_sink,
    torch::Tensor recent_start,
    int64_t num_kv_heads,
    int64_t num_queries_per_kv,
    int64_t k_head,
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
    bool lambda_soft,
    int64_t nms_window,
    double soft_alpha,
    double alpha_cross,
    double temperature,
    double cross_eps,
    int64_t slice_start,
    int64_t slice_end,
    c10::optional<torch::Tensor> log_r_cache_opt) {
    return selector_pipeline_pre_denom_topk_with_bounds_impl(
        scores, denom, row_lo, row_hi, key_norms, head_sink, recent_start,
        num_kv_heads, num_queries_per_kv, k_head, alpha, eps, gamma,
        prior_weight_l2, prior_weight_pos, prior_pos_power, prior_pos_eta,
        beta, lambda_clip_single, lambda_clip_multi, lambda_tail_kappa,
        lambda_tail_pivot, lambda_soft, nms_window, soft_alpha, alpha_cross,
        temperature, cross_eps, slice_start, slice_end, log_r_cache_opt,
        c10::nullopt, c10::nullopt, c10::nullopt);
}

std::vector<torch::Tensor> selector_pipeline_pre_denom_topk_with_bounds_workspace(
    torch::Tensor scores,
    torch::Tensor denom,
    torch::Tensor row_lo,
    torch::Tensor row_hi,
    torch::Tensor key_norms,
    torch::Tensor head_sink,
    torch::Tensor recent_start,
    int64_t num_kv_heads,
    int64_t num_queries_per_kv,
    int64_t k_head,
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
    bool lambda_soft,
    int64_t nms_window,
    double soft_alpha,
    double alpha_cross,
    double temperature,
    double cross_eps,
    int64_t slice_start,
    int64_t slice_end,
    c10::optional<torch::Tensor> log_r_cache_opt,
    torch::Tensor workspace_a,
    torch::Tensor workspace_b) {
    return selector_pipeline_pre_denom_topk_with_bounds_impl(
        scores, denom, row_lo, row_hi, key_norms, head_sink, recent_start,
        num_kv_heads, num_queries_per_kv, k_head, alpha, eps, gamma,
        prior_weight_l2, prior_weight_pos, prior_pos_power, prior_pos_eta,
        beta, lambda_clip_single, lambda_clip_multi, lambda_tail_kappa,
        lambda_tail_pivot, lambda_soft, nms_window, soft_alpha, alpha_cross,
        temperature, cross_eps, slice_start, slice_end, log_r_cache_opt,
        c10::nullopt, workspace_a, workspace_b);
}

std::vector<torch::Tensor> selector_pipeline_pre_denom_topk_with_bounds_out(
    torch::Tensor scores,
    torch::Tensor denom,
    torch::Tensor row_lo,
    torch::Tensor row_hi,
    torch::Tensor key_norms,
    torch::Tensor head_sink,
    torch::Tensor recent_start,
    int64_t num_kv_heads,
    int64_t num_queries_per_kv,
    int64_t k_head,
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
    bool lambda_soft,
    int64_t nms_window,
    double soft_alpha,
    double alpha_cross,
    double temperature,
    double cross_eps,
    int64_t slice_start,
    int64_t slice_end,
    c10::optional<torch::Tensor> log_r_cache_opt,
    torch::Tensor selected_indices_out) {
    return selector_pipeline_pre_denom_topk_with_bounds_impl(
        scores, denom, row_lo, row_hi, key_norms, head_sink, recent_start,
        num_kv_heads, num_queries_per_kv, k_head, alpha, eps, gamma,
        prior_weight_l2, prior_weight_pos, prior_pos_power, prior_pos_eta,
        beta, lambda_clip_single, lambda_clip_multi, lambda_tail_kappa,
        lambda_tail_pivot, lambda_soft, nms_window, soft_alpha, alpha_cross,
        temperature, cross_eps, slice_start, slice_end, log_r_cache_opt,
        selected_indices_out, c10::nullopt, c10::nullopt);
}

std::vector<torch::Tensor> selector_pipeline_pre_denom_topk_with_bounds_workspace_out(
    torch::Tensor scores,
    torch::Tensor denom,
    torch::Tensor row_lo,
    torch::Tensor row_hi,
    torch::Tensor key_norms,
    torch::Tensor head_sink,
    torch::Tensor recent_start,
    int64_t num_kv_heads,
    int64_t num_queries_per_kv,
    int64_t k_head,
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
    bool lambda_soft,
    int64_t nms_window,
    double soft_alpha,
    double alpha_cross,
    double temperature,
    double cross_eps,
    int64_t slice_start,
    int64_t slice_end,
    c10::optional<torch::Tensor> log_r_cache_opt,
    torch::Tensor workspace_a,
    torch::Tensor workspace_b,
    torch::Tensor selected_indices_out) {
    return selector_pipeline_pre_denom_topk_with_bounds_impl(
        scores, denom, row_lo, row_hi, key_norms, head_sink, recent_start,
        num_kv_heads, num_queries_per_kv, k_head, alpha, eps, gamma,
        prior_weight_l2, prior_weight_pos, prior_pos_power, prior_pos_eta,
        beta, lambda_clip_single, lambda_clip_multi, lambda_tail_kappa,
        lambda_tail_pivot, lambda_soft, nms_window, soft_alpha, alpha_cross,
        temperature, cross_eps, slice_start, slice_end, log_r_cache_opt,
        selected_indices_out, workspace_a, workspace_b);
}

"""

    _ensure_torch_cuda_arch_list()
    try:
        # [EXT-NVCC-GUARD] JIT 回落前 pin nvcc+版本预检(系统 nvcc 10.1 坑根修);
        # 失败信息进 _LOAD_ERROR,由 require/enabled 语义原样呈报。
        configure_jit_toolchain_or_raise(ext_name="selector_pipeline_ext")
        _MODULE = load_inline(
            name="selector_pipeline_ext",
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


def _require_ext(*, force: bool = False) -> torch.nn.Module:
    mod = _load_ext(force=force)
    if mod is None:
        if _LOAD_ERROR is not None:
            root_cause = repr(_LOAD_ERROR)
            raise RuntimeError(
                "selector_pipeline_ext unavailable; set VLLM_SPARSE_SELECTOR_CUDA_PIPELINE=1; "
                f"root_cause={root_cause}"
            ) from _LOAD_ERROR
        root_cause = "disabled_or_not_loaded"
        raise RuntimeError(
            "selector_pipeline_ext unavailable; set VLLM_SPARSE_SELECTOR_CUDA_PIPELINE=1; "
            f"root_cause={root_cause}"
        )
    return mod






def pipeline_logits_topk_fused(
    capture_scores: torch.Tensor,
    kv_lengths: torch.Tensor,
    key_norms_full: torch.Tensor,
    log_f_denoms: Optional[torch.Tensor],
    seq_full: Optional[torch.Tensor],  # Pre-created by caller, or None
    *,
    num_kv_heads: int,
    num_queries_per_kv: int,
    sink_cfg: int,
    recent_cfg: int,
    block_size: int,
    k_head: int,
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
    nms_window: int,
    soft_alpha: float,
    alpha_cross: float,
    temperature: float,
    cross_eps: float,
    slice_start: int,
    slice_end: int,
) -> tuple:
    """Fused selector pipeline: preproc_bounds + pipeline in SINGLE Python->C++ call.

    This is the optimized version that reduces Python->C++ call overhead by:
    - Computing preproc_bounds inside C++ (eliminates separate call)
    - Calling pipeline kernel directly
    - Returning all needed outputs in one call

    Note:
        This fused entry only supports logits path (log_f_denoms=None).
        pre_denom path must use bounds-first entry
        pipeline_pre_denom_topk_with_bounds(...), fail-fast by design.

    Args:
        seq_full: Pre-created seq_full tensor [L, B, H_kv] or None.
                  Caller (vllm_sparse_patch.py) should prepare this to avoid
                  repeated tensor creation overhead.

    Returns:
        (selected_indices, head_sink, recent_start, kv_len_head, allowed_lengths)
    """
    mod = _require_ext()

    # Extract kv_len_total from scores shape for C++ call
    kv_len_total = capture_scores.shape[4]

    # Call C++ fused function (SINGLE Python->C++ call)
    # Combines: preproc_bounds + reshape + pipeline in one call
    if log_f_denoms is None:
        result = mod.selector_pipeline_logits_topk_fused(
                capture_scores,
                kv_lengths,
                key_norms_full,
                seq_full,  # Optional tensor, C++ handles None
                int(num_kv_heads),
                int(num_queries_per_kv),
                int(kv_len_total),
                int(sink_cfg),
                int(recent_cfg),
                int(block_size),
                int(k_head),
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
                int(nms_window),
                float(soft_alpha),
                float(alpha_cross),
                float(temperature),
                float(cross_eps),
                int(slice_start),
                int(slice_end),
            )
    else:
        raise RuntimeError(
            "pipeline_logits_topk_fused no longer supports pre_denom path; "
            "use pipeline_pre_denom_topk_with_bounds(...)"
        )

    # Unpack result: [selected_indices, head_sink, recent_start, kv_len_head, allowed_lengths]
    selected_indices, head_sink, recent_start, kv_len_head, allowed_lengths = result
    return (selected_indices, head_sink, recent_start, kv_len_head, allowed_lengths)




def pipeline_logits_topk_with_bounds(
    capture_scores: torch.Tensor,
    row_lo: torch.Tensor,
    row_hi: torch.Tensor,
    key_norms_full: torch.Tensor,
    head_sink: torch.Tensor,
    recent_start: torch.Tensor,
    *,
    num_kv_heads: int,
    num_queries_per_kv: int,
    k_head: int,
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
    nms_window: int,
    soft_alpha: float,
    alpha_cross: float,
    temperature: float,
    cross_eps: float,
    slice_start: int,
    slice_end: int,
    log_r_cache: Optional[torch.Tensor] = None,
    pipeline_workspaces: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    selected_indices_out: Optional[torch.Tensor] = None,
    fixed_shape_topk: bool = False,
) -> torch.Tensor:
    """Pipeline with pre-computed bounds from CUDA kernel.

    This function accepts bounds computed by decode or prefill CUDA bounds kernels
    and skips all ATen bounds computation for maximum efficiency.

    Args:
        capture_scores: [L, B, H_total, W, K] logits scores
        row_lo: [M, G] or [M, G*W] pre-computed row lower bounds (M = L*B*H_kv)
        row_hi: [M, G] or [M, G*W] pre-computed row upper bounds
        key_norms_full: [L, B, H_kv, K] key norms
        head_sink: [L, B, H_kv] token lower bounds
        recent_start: [L, B, H_kv] token upper bounds
        num_kv_heads: H_kv
        num_queries_per_kv: G
        ... other selector parameters ...

    Returns:
        selected_indices: [L, B, H_kv, k_head] selected token indices
    """
    mod = _require_ext()
    # fa4_selector_fixed_shape_topk: coherence shim — make the C++ env read match the
    # explicit python gate for the duration of this call.
    _fst_prev = os.environ.get("VLLM_SPARSE_SELECTOR_FIXED_SHAPE_TOPK")
    if fixed_shape_topk and _fst_prev != "1":
        os.environ["VLLM_SPARSE_SELECTOR_FIXED_SHAPE_TOPK"] = "1"

    _fuse_fire_trace()
    args = (
        capture_scores,
        row_lo,
        row_hi,
        key_norms_full,
        head_sink,
        recent_start,
        int(num_kv_heads),
        int(num_queries_per_kv),
        int(k_head),
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
        int(nms_window),
        float(soft_alpha),
        float(alpha_cross),
        float(temperature),
        float(cross_eps),
        int(slice_start),
        int(slice_end),
        log_r_cache,
    )
    if selected_indices_out is None and pipeline_workspaces is None:
        result = mod.selector_pipeline_logits_topk_with_bounds(*args)
    elif selected_indices_out is None:
        if not hasattr(mod, "selector_pipeline_logits_topk_with_bounds_workspace"):
            raise RuntimeError(
                "selector_pipeline_ext workspace path requires rebuilt extension entrypoint"
            )
        workspace_a, workspace_b = pipeline_workspaces
        result = mod.selector_pipeline_logits_topk_with_bounds_workspace(
            *args,
            workspace_a,
            workspace_b,
        )
    elif pipeline_workspaces is None:
        if not hasattr(mod, "selector_pipeline_logits_topk_with_bounds_out"):
            raise RuntimeError(
                "selector_pipeline_ext selected_indices_out path requires rebuilt extension entrypoint"
            )
        result = mod.selector_pipeline_logits_topk_with_bounds_out(
            *args,
            selected_indices_out,
        )
    else:
        if not hasattr(mod, "selector_pipeline_logits_topk_with_bounds_workspace_out"):
            raise RuntimeError(
                "selector_pipeline_ext workspace selected_indices_out path requires rebuilt extension entrypoint"
            )
        workspace_a, workspace_b = pipeline_workspaces
        result = mod.selector_pipeline_logits_topk_with_bounds_workspace_out(
            *args,
            workspace_a,
            workspace_b,
            selected_indices_out,
        )

    # Result is [selected_indices] only
    if fixed_shape_topk and _fst_prev != "1":
        if _fst_prev is None:
            os.environ.pop("VLLM_SPARSE_SELECTOR_FIXED_SHAPE_TOPK", None)
        else:
            os.environ["VLLM_SPARSE_SELECTOR_FIXED_SHAPE_TOPK"] = _fst_prev
    return result[0]


def pipeline_pre_denom_topk_with_bounds(
    capture_scores: torch.Tensor,
    log_f_denoms: torch.Tensor,
    row_lo: torch.Tensor,
    row_hi: torch.Tensor,
    key_norms_full: torch.Tensor,
    head_sink: torch.Tensor,
    recent_start: torch.Tensor,
    *,
    num_kv_heads: int,
    num_queries_per_kv: int,
    k_head: int,
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
    nms_window: int,
    soft_alpha: float,
    alpha_cross: float,
    temperature: float,
    cross_eps: float,
    slice_start: int,
    slice_end: int,
    log_r_cache: Optional[torch.Tensor] = None,
    pipeline_workspaces: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    selected_indices_out: Optional[torch.Tensor] = None,
    fixed_shape_topk: bool = False,
) -> torch.Tensor:
    """Bounds-first pre-denom pipeline (decode path, W=1).

    This entry removes fused denom-expand behavior and requires caller-provided bounds.
    Any unsupported layout triggers fail-fast in C++.
    """
    mod = _require_ext()
    # fa4_selector_fixed_shape_topk: coherence shim — make the C++ env read match the
    # explicit python gate for the duration of this call.
    _fst_prev = os.environ.get("VLLM_SPARSE_SELECTOR_FIXED_SHAPE_TOPK")
    if fixed_shape_topk and _fst_prev != "1":
        os.environ["VLLM_SPARSE_SELECTOR_FIXED_SHAPE_TOPK"] = "1"

    _fuse_fire_trace()
    args = (
        capture_scores,
        log_f_denoms,
        row_lo,
        row_hi,
        key_norms_full,
        head_sink,
        recent_start,
        int(num_kv_heads),
        int(num_queries_per_kv),
        int(k_head),
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
        int(nms_window),
        float(soft_alpha),
        float(alpha_cross),
        float(temperature),
        float(cross_eps),
        int(slice_start),
        int(slice_end),
        log_r_cache,
    )
    if selected_indices_out is None and pipeline_workspaces is None:
        result = mod.selector_pipeline_pre_denom_topk_with_bounds(*args)
    elif selected_indices_out is None:
        if not hasattr(mod, "selector_pipeline_pre_denom_topk_with_bounds_workspace"):
            raise RuntimeError(
                "selector_pipeline_ext pre_denom workspace path requires rebuilt extension entrypoint"
            )
        workspace_a, workspace_b = pipeline_workspaces
        result = mod.selector_pipeline_pre_denom_topk_with_bounds_workspace(
            *args,
            workspace_a,
            workspace_b,
        )
    elif pipeline_workspaces is None:
        if not hasattr(mod, "selector_pipeline_pre_denom_topk_with_bounds_out"):
            raise RuntimeError(
                "selector_pipeline_ext pre_denom selected_indices_out path requires rebuilt extension entrypoint"
            )
        result = mod.selector_pipeline_pre_denom_topk_with_bounds_out(
            *args,
            selected_indices_out,
        )
    else:
        if not hasattr(mod, "selector_pipeline_pre_denom_topk_with_bounds_workspace_out"):
            raise RuntimeError(
                "selector_pipeline_ext pre_denom workspace selected_indices_out path requires rebuilt extension entrypoint"
            )
        workspace_a, workspace_b = pipeline_workspaces
        result = mod.selector_pipeline_pre_denom_topk_with_bounds_workspace_out(
            *args,
            workspace_a,
            workspace_b,
            selected_indices_out,
        )

    if fixed_shape_topk and _fst_prev != "1":
        if _fst_prev is None:
            os.environ.pop("VLLM_SPARSE_SELECTOR_FIXED_SHAPE_TOPK", None)
        else:
            os.environ["VLLM_SPARSE_SELECTOR_FIXED_SHAPE_TOPK"] = _fst_prev
    return result[0]


__all__ = [
    "pipeline_logits_topk_fused",
    "pipeline_logits_topk_with_bounds",
    "pipeline_pre_denom_topk_with_bounds",
]
