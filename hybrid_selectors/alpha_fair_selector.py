"""Alpha-fair token selector with vectorised multi-head support."""

from __future__ import annotations

import dataclasses
import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from triton_kernel.alpha_selector_kernel import (
    alpha_fuse_from_log_probs_bounds_rows1_triton,
    cross_head_mutex_triton,
    cross_head_mutex_triton_bounds,
    log_f_mean_probs_from_log_f_pre_denom_bounds_triton,
    log_f_mean_probs_from_logits_bounds_triton,
    soft_nms_triton,
    soft_nms_triton_bounds,
)


# GPU 配置缓存（避免 CPU 张量导致 torch.compile 失效）
_GPU_CONFIG_CACHE = {}
_ALPHA_ROWS1_FUSE_CACHED = os.environ.get("VLLM_SPARSE_ALPHA_ROWS1_FUSE", "1") == "1"


@dataclass(slots=True)
class AlphaFairSelectorConfig:
    """Hyper-parameters controlling the alpha-fair selector."""

    alpha: float = 0.5
    gamma: float = 0.5
    prior_weight_l2: float = 1.0
    prior_weight_pos: float = 0.7
    prior_pos_power: float = 1.8
    prior_pos_eta: float = 0.4
    prior_beta_theta: float = 0.6
    prior_beta_p: float = 0.9
    nms_window: int = 32
    soft_alpha: float = 1.0
    cross_head_alpha: float = 0.85
    cross_head_temperature: float = 2.0
    cross_head_power: float = 0.0
    lambda_tail_kappa: float = 0.0
    lambda_tail_pivot: float = 0.7
    k_head: Optional[int] = 1568
    selection_mode: str = "token_topk"
    lambda_clip_single: float = 0.35
    lambda_clip_multi: float = 0.25
    lambda_soft: bool = False
    query_norm_min_scale: float = 1.0
    query_norm_max_scale: float = 1.0
    eps: float = 1.0e-12

    def beta(self) -> float:
        return -math.log(max(self.prior_beta_theta, self.eps)) / max(self.prior_beta_p, self.eps)


def _with_mask(tensor: torch.Tensor, mask: torch.Tensor, fill: float) -> torch.Tensor:
    return torch.where(mask, tensor, fill)



def _get_gpu_config(cfg: AlphaFairSelectorConfig, device: torch.device) -> dict:
    """将配置参数缓存为 GPU 标量张量，使 torch.compile 能正常使用 CUDA graph"""
    global _GPU_CONFIG_CACHE
    cache_key = (dataclasses.astuple(cfg), str(device))
    if cache_key not in _GPU_CONFIG_CACHE:
        # 将所有 float 参数转为 GPU 标量张量
        _GPU_CONFIG_CACHE[cache_key] = {
            "alpha": torch.tensor(cfg.alpha, device=device, dtype=torch.float32),
            "gamma": torch.tensor(cfg.gamma, device=device, dtype=torch.float32),
            "prior_weight_l2": torch.tensor(cfg.prior_weight_l2, device=device, dtype=torch.float32),
            "prior_weight_pos": torch.tensor(cfg.prior_weight_pos, device=device, dtype=torch.float32),
            "prior_pos_power": torch.tensor(max(cfg.prior_pos_power, 1.0), device=device, dtype=torch.float32),
            "prior_pos_eta": torch.tensor(cfg.prior_pos_eta, device=device, dtype=torch.float32),
            "beta": torch.tensor(cfg.beta(), device=device, dtype=torch.float32),
            "nms_window": cfg.nms_window,  # int 保持不变
            "soft_alpha": torch.tensor(cfg.soft_alpha, device=device, dtype=torch.float32),
            "lambda_clip_single": torch.tensor(cfg.lambda_clip_single, device=device, dtype=torch.float32),
            "lambda_clip_multi": torch.tensor(cfg.lambda_clip_multi, device=device, dtype=torch.float32),
            "lambda_soft": cfg.lambda_soft,  # bool 保持不变
            "lambda_tail_kappa": torch.tensor(cfg.lambda_tail_kappa, device=device, dtype=torch.float32),
            "lambda_tail_pivot": torch.tensor(cfg.lambda_tail_pivot, device=device, dtype=torch.float32),
            "query_norm_min_scale": torch.tensor(max(cfg.query_norm_min_scale, cfg.eps), device=device, dtype=torch.float32),
            "query_norm_max_scale": torch.tensor(max(cfg.query_norm_max_scale, cfg.query_norm_min_scale), device=device, dtype=torch.float32),
            "eps": torch.tensor(cfg.eps, device=device, dtype=torch.float32),
        }
    return _GPU_CONFIG_CACHE[cache_key]


def _prior_fused_compute(
    log_f: torch.Tensor,
    token_mask: torch.Tensor,
    valid_tokens: torch.Tensor,
    key_norms_flat: torch.Tensor,
    pos_token_f: torch.Tensor,
    lam_max: torch.Tensor,
    *,
    flat: int,
    heads: int,
    kv_len: int,
    gamma: float,
    prior_weight_l2: float,
    prior_weight_pos: float,
    prior_pos_power: float,
    prior_pos_eta: float,
    beta: float,
    eps: float,
    lambda_tail_kappa: float,
    lambda_tail_pivot: float,
    lambda_soft: bool,
) -> torch.Tensor:
    """Pure PyTorch prior + lambda + fused score computation (torch.compile friendly).

    All config-dependent branches use scalar kwargs so torch.compile treats them
    as static guards. No dict lookups or Triton kernel calls inside.
    """
    min_val = float("-inf")

    # Normalize log_f
    log_f = _with_mask(log_f, token_mask, min_val)
    denom_f = torch.logsumexp(log_f, dim=-1, keepdim=True)  # log_f already masked above
    log_f = torch.where(valid_tokens.unsqueeze(-1), log_f - denom_f, min_val)
    log_f = _with_mask(log_f, token_mask, min_val)

    # Prior: key L2 norms
    safe_norms = torch.clamp(key_norms_flat, min=eps)
    log_pi = torch.where(token_mask, -gamma * torch.log(safe_norms), min_val)

    # Prior: position
    pos = pos_token_f.view(1, 1, kv_len).expand(flat, heads, kv_len)
    min_pos = torch.where(token_mask, pos, float("inf")).amin(dim=-1, keepdim=True)
    max_pos = torch.where(token_mask, pos, float("-inf")).amax(dim=-1, keepdim=True)
    range_pos = (max_pos - min_pos).clamp_min(1.0)
    pos_norm = torch.where(token_mask, (pos - min_pos) / range_pos, 0.0)

    pos_shaped = torch.pow(pos_norm, prior_pos_power)
    base_delta = -beta * pos_shaped
    if prior_pos_eta > 0.0:
        one_minus = torch.clamp(1.0 - pos_norm, min=eps)
        base_delta = base_delta + prior_pos_eta * torch.log(one_minus)
    log_delta = torch.where(token_mask, base_delta, min_val)

    # Combined prior
    log_r_raw = prior_weight_l2 * log_pi + prior_weight_pos * log_delta
    log_r = _with_mask(log_r_raw, token_mask, min_val)
    denom_r = torch.logsumexp(log_r, dim=-1, keepdim=True)  # log_r already masked above
    log_r = torch.where(valid_tokens.unsqueeze(-1), log_r - denom_r, min_val)
    log_r = _with_mask(log_r, token_mask, min_val)

    # Lambda computation
    log_f_valid = log_f   # already masked at line 125
    log_r_valid = log_r   # already masked at line 150
    ff = torch.exp(torch.logsumexp(2.0 * log_f_valid, dim=-1))
    rr = torch.exp(torch.logsumexp(2.0 * log_r_valid, dim=-1))
    fr = torch.exp(torch.logsumexp(log_f_valid + log_r_valid, dim=-1))

    denom_lam = torch.clamp(ff - 2.0 * fr + rr, min=eps)
    lam_star = (ff - fr) / denom_lam
    lam = torch.clamp_min(lam_star, 0.0)

    if lambda_tail_kappa > 0.0:
        probs = torch.exp(log_f)
        probs = torch.where(token_mask, probs, 0.0)
        prob_sum = probs.sum(dim=-1, keepdim=True).clamp_min(eps)
        mean_pos = (probs * pos_norm).sum(dim=-1, keepdim=True) / prob_sum
        tail_excess = torch.clamp(mean_pos - lambda_tail_pivot, min=0.0).squeeze(-1)
        lam = lam * (1.0 + lambda_tail_kappa * tail_excess)

    lam = torch.clamp_min(lam, 0.0)
    if lambda_soft:
        clip = torch.clamp(lam_max, min=eps)
        lam = clip * torch.tanh(lam / clip)
    else:
        lam = torch.minimum(lam, lam_max)

    # Fused score
    one_minus_eps = 1.0 - eps
    log_one_minus = torch.log1p(-torch.clamp(lam, max=one_minus_eps))
    log_lambda = torch.log(lam + eps)

    fused = torch.logaddexp(log_one_minus.unsqueeze(-1) + log_f, log_lambda.unsqueeze(-1) + log_r)
    fused = _with_mask(fused, token_mask, min_val)
    denom_fused = torch.logsumexp(_with_mask(fused, token_mask, min_val), dim=-1, keepdim=True)
    fused = torch.where(valid_tokens.unsqueeze(-1), fused - denom_fused, min_val)
    fused = _with_mask(fused, token_mask, min_val)

    return fused


_COMPILE_SELECTOR = os.environ.get("VLLM_SPARSE_COMPILE_SELECTOR", "0") == "1"
_prior_fused_compute_fn = (
    torch.compile(_prior_fused_compute, mode="reduce-overhead")
    if _COMPILE_SELECTOR
    else _prior_fused_compute
)


def _apply_prior_fused_soft_nms(
    log_f_mk: torch.Tensor,
    mean_probs_mk: torch.Tensor,
    row_counts_m: torch.Tensor,
    lo_mr: torch.Tensor,
    hi_mr: torch.Tensor,
    key_norms: torch.Tensor,
    positions: torch.Tensor,
    cfg: AlphaFairSelectorConfig,
    *,
    layers: int,
    batch: int,
    flat: int,
    heads: int,
    kv_len: int,
    ref_device: torch.device,
    ref_dtype: torch.dtype,
    return_mean_probs: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Shared prior + fused + soft-NMS pipeline for both logf and pre-denom paths.

    Handles both the bounds-fused fast path (when return_mean_probs=False)
    and the full PyTorch slow path.
    """
    eps = float(cfg.eps)

    # bounds fused 快路：用 Triton fused 计算替代 token_mask 物化与大量 torch op。
    use_bounds_fused = (not return_mean_probs) and _ALPHA_ROWS1_FUSE_CACHED
    if use_bounds_fused:
        token_lo_m = lo_mr.amin(dim=1).to(dtype=torch.int32)
        token_hi_m = hi_mr.amax(dim=1).to(dtype=torch.int32)
        key_norms_flat = key_norms.reshape(flat * heads, kv_len)

        fused_mk = alpha_fuse_from_log_probs_bounds_rows1_triton(
            log_f_mk,
            key_norms_flat,
            token_lo_m,
            token_hi_m,
            row_counts_m,
            gamma=float(cfg.gamma),
            prior_weight_l2=float(cfg.prior_weight_l2),
            prior_weight_pos=float(cfg.prior_weight_pos),
            prior_pos_power=float(max(cfg.prior_pos_power, 1.0)),
            prior_pos_eta=float(cfg.prior_pos_eta),
            beta=float(cfg.beta()),
            lambda_clip_single=float(cfg.lambda_clip_single),
            lambda_clip_multi=float(cfg.lambda_clip_multi),
            lambda_tail_kappa=float(cfg.lambda_tail_kappa),
            lambda_tail_pivot=float(cfg.lambda_tail_pivot),
            lambda_soft=bool(cfg.lambda_soft),
            eps=eps,
        )
        fused = fused_mk.reshape(flat, heads, kv_len)
        log_s_prime = soft_nms_triton_bounds(
            fused,
            token_lo_m.reshape(flat, heads),
            token_hi_m.reshape(flat, heads),
            int(cfg.nms_window),
            float(cfg.soft_alpha),
        )
        log_s_out = log_s_prime.reshape(layers, batch, heads, kv_len).to(dtype=ref_dtype)
        mean_out = torch.empty(0, 0, 0, 0, device=ref_device, dtype=ref_dtype)
        return log_s_out, mean_out

    log_f = log_f_mk.reshape(flat, heads, kv_len)
    mean_probs = mean_probs_mk.reshape(flat, heads, kv_len) if return_mean_probs else None
    row_counts = row_counts_m.reshape(flat, heads)

    # Prepare inputs for _prior_fused_compute (outside compiled boundary).
    token_lo = lo_mr.amin(dim=1).reshape(flat, heads).to(dtype=torch.int32)
    token_hi = hi_mr.amax(dim=1).reshape(flat, heads).to(dtype=torch.int32)
    if positions.dim() == 1:
        pos_token_i32 = positions.to(device=ref_device, dtype=torch.int32)
    else:
        pos_token_i32 = torch.arange(kv_len, device=ref_device, dtype=torch.int32)
    pos_flat_i32 = pos_token_i32.view(1, 1, kv_len).expand(flat, heads, kv_len)
    token_mask = (pos_flat_i32 >= token_lo.unsqueeze(-1)) & (pos_flat_i32 < token_hi.unsqueeze(-1))
    valid_tokens = token_mask.any(dim=-1)

    key_norms_flat = key_norms.reshape(flat, heads, kv_len)
    if key_norms_flat.dtype != torch.float32:
        key_norms_flat = key_norms_flat.to(dtype=torch.float32)
    pos_token_f = pos_token_i32.to(dtype=torch.float32)

    cfg_gpu = _get_gpu_config(cfg, ref_device)
    lam_max = torch.where(row_counts > 1.0, cfg_gpu["lambda_clip_multi"], cfg_gpu["lambda_clip_single"])

    fused = _prior_fused_compute_fn(
        log_f, token_mask, valid_tokens, key_norms_flat, pos_token_f, lam_max,
        flat=flat, heads=heads, kv_len=kv_len,
        gamma=float(cfg.gamma),
        prior_weight_l2=float(cfg.prior_weight_l2),
        prior_weight_pos=float(cfg.prior_weight_pos),
        prior_pos_power=float(max(cfg.prior_pos_power, 1.0)),
        prior_pos_eta=float(cfg.prior_pos_eta),
        beta=float(cfg.beta()),
        eps=eps,
        lambda_tail_kappa=float(cfg.lambda_tail_kappa),
        lambda_tail_pivot=float(cfg.lambda_tail_pivot),
        lambda_soft=bool(cfg.lambda_soft),
    )

    log_s_prime = soft_nms_triton(fused, token_mask, int(cfg.nms_window), float(cfg.soft_alpha))
    log_s_out = log_s_prime.reshape(layers, batch, heads, kv_len).to(dtype=ref_dtype)

    if not return_mean_probs or mean_probs is None:
        mean_out = torch.empty(0, 0, 0, 0, device=ref_device, dtype=ref_dtype)
    else:
        mean_probs = torch.where(token_mask, mean_probs, 0.0)
        mean_out = mean_probs.reshape(layers, batch, heads, kv_len).to(dtype=ref_dtype)
    return log_s_out, mean_out


def compute_alpha_scores_batched_layers_triton_logf_bounds(
    logits: torch.Tensor,
    row_lo: torch.Tensor,
    row_hi: torch.Tensor,
    key_norms: torch.Tensor,
    positions: torch.Tensor,
    cfg: AlphaFairSelectorConfig,
    *,
    return_mean_probs: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Triton log_f 快路（bounds 版本，不物化 6D mask）。

    Args:
      logits:   [L,B,H,G,W,K]
      row_lo/hi:[L,B,H,G,W]，每行允许的 token 区间 [lo, hi)（int32）
      key_norms/positions: [L,B,H,K]
    """
    if logits.dim() != 6:
        raise ValueError("logits must be [L, B, H, G, W, K]")
    if row_lo.shape != logits.shape[:5] or row_hi.shape != logits.shape[:5]:
        raise ValueError("row_lo/row_hi must match logits leading dims [L, B, H, G, W]")
    if key_norms.shape[:2] != logits.shape[:2] or key_norms.shape[-1] != logits.shape[-1]:
        raise ValueError("key_norms must be [L, B, H, K]")
    # positions 仅用于 position prior 与 token_mask 构造；允许传入 [K] 的 1D positions（推荐），
    # 以避免把 expand view 物化成巨大 [L,B,H,K] 张量。
    if positions.dim() not in (1, 4):
        raise ValueError("positions must be 1D [K] or 4D [L,B,H,K]")
    if positions.dim() == 4 and positions.shape != key_norms.shape:
        raise ValueError("positions must match key_norms shape when provided as 4D tensor")
    if not logits.is_cuda:
        raise RuntimeError("bounds path requires CUDA tensors")

    layers, batch, heads, groups, window, kv_len = logits.shape
    flat = layers * batch
    rows = groups * window
    if rows > 128:
        raise RuntimeError("bounds path only supports rows<=128")

    # 性能关键：避免把 [L,B,H,G,W,K] 的 logits 全量 upcast 成 fp32（会额外触发一次大拷贝/额外 kernel）。
    # Triton kernel 内部会把每个元素转成 fp32 做稳定计算，因此这里保持 logits 原 dtype 即可。
    logits_flat = logits.reshape(flat, heads, rows, kv_len)
    lo_flat = row_lo.to(device=logits.device, dtype=torch.int32).reshape(flat, heads, rows)
    hi_flat = row_hi.to(device=logits.device, dtype=torch.int32).reshape(flat, heads, rows)

    alpha = float(cfg.alpha)
    eps = float(cfg.eps)

    logits_mrk = logits_flat.reshape(flat * heads, rows, kv_len)
    lo_mr = lo_flat.reshape(flat * heads, rows)
    hi_mr = hi_flat.reshape(flat * heads, rows)
    log_f_mk, mean_probs_mk, row_counts_m = log_f_mean_probs_from_logits_bounds_triton(
        logits_mrk,
        lo_mr,
        hi_mr,
        alpha,
        eps,
        return_mean_probs=return_mean_probs,
    )

    return _apply_prior_fused_soft_nms(
        log_f_mk, mean_probs_mk, row_counts_m,
        lo_mr, hi_mr, key_norms, positions, cfg,
        layers=layers, batch=batch, flat=flat, heads=heads, kv_len=kv_len,
        ref_device=logits.device, ref_dtype=logits.dtype,
        return_mean_probs=return_mean_probs,
    )


def compute_alpha_scores_batched_layers_triton_logf_pre_denom_bounds(
    log_f_pre: torch.Tensor,
    denom: torch.Tensor,
    row_lo: torch.Tensor,
    row_hi: torch.Tensor,
    key_norms: torch.Tensor,
    positions: torch.Tensor,
    cfg: AlphaFairSelectorConfig,
    *,
    return_mean_probs: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """log_f_pre + denom 快路（bounds 版本，不物化 6D mask）。"""
    if log_f_pre.dim() != 6:
        raise ValueError("log_f_pre must be [L, B, H, G, W, K]")
    if denom.shape != log_f_pre.shape[:5]:
        raise ValueError("denom must match log_f_pre leading dims [L, B, H, G, W]")
    if row_lo.shape != log_f_pre.shape[:5] or row_hi.shape != log_f_pre.shape[:5]:
        raise ValueError("row_lo/row_hi must match log_f_pre leading dims [L, B, H, G, W]")
    if key_norms.shape[:2] != log_f_pre.shape[:2] or key_norms.shape[-1] != log_f_pre.shape[-1]:
        raise ValueError("key_norms must be [L, B, H, K]")
    if positions.dim() not in (1, 4):
        raise ValueError("positions must be 1D [K] or 4D [L,B,H,K]")
    if positions.dim() == 4 and positions.shape != key_norms.shape:
        raise ValueError("positions must match key_norms shape when provided as 4D tensor")
    if not log_f_pre.is_cuda:
        raise RuntimeError("bounds path requires CUDA tensors")

    layers, batch, heads, groups, window, kv_len = log_f_pre.shape
    flat = layers * batch
    rows = groups * window
    if rows > 128:
        raise RuntimeError("bounds path only supports rows<=128")

    log_f_pre_flat = log_f_pre.reshape(flat, heads, rows, kv_len)
    denom_flat = denom.to(dtype=torch.float32).reshape(flat, heads, rows)
    lo_flat = row_lo.to(device=log_f_pre.device, dtype=torch.int32).reshape(flat, heads, rows)
    hi_flat = row_hi.to(device=log_f_pre.device, dtype=torch.int32).reshape(flat, heads, rows)

    alpha = float(cfg.alpha)
    eps = float(cfg.eps)

    lfp_mrk = log_f_pre_flat.reshape(flat * heads, rows, kv_len)
    denom_mr = denom_flat.reshape(flat * heads, rows)
    lo_mr = lo_flat.reshape(flat * heads, rows)
    hi_mr = hi_flat.reshape(flat * heads, rows)

    log_f_mk, mean_probs_mk, row_counts_m = log_f_mean_probs_from_log_f_pre_denom_bounds_triton(
        lfp_mrk,
        denom_mr,
        lo_mr,
        hi_mr,
        alpha,
        eps,
        return_mean_probs=return_mean_probs,
    )
    return _apply_prior_fused_soft_nms(
        log_f_mk, mean_probs_mk, row_counts_m,
        lo_mr, hi_mr, key_norms, positions, cfg,
        layers=layers, batch=batch, flat=flat, heads=heads, kv_len=kv_len,
        ref_device=log_f_pre.device, ref_dtype=log_f_pre.dtype,
        return_mean_probs=return_mean_probs,
    )


def apply_cross_head_mutex(
    log_s_all: torch.Tensor,
    mask: torch.Tensor,
    alpha_cross: float,
    temperature: float,
    window: int,
    power: float = 0.0,
) -> torch.Tensor:
    if not log_s_all.is_cuda or not mask.is_cuda:
        raise RuntimeError("apply_cross_head_mutex requires CUDA tensors")
    return cross_head_mutex_triton(
        log_s_all,
        mask,
        alpha_cross,
        temperature,
        window,
        power,
    )


def apply_cross_head_mutex_bounds(
    log_s_all: torch.Tensor,
    token_lo: torch.Tensor,
    token_hi: torch.Tensor,
    alpha_cross: float,
    temperature: float,
    window: int,
    power: float = 0.0,
) -> torch.Tensor:
    """cross-head mutex（bounds 版本）：不再物化 [N,H,K] 的 bool mask。"""
    if not log_s_all.is_cuda or not token_lo.is_cuda or not token_hi.is_cuda:
        raise RuntimeError("apply_cross_head_mutex_bounds requires CUDA tensors")
    # P0 优化：当 local kv_heads <= 2 时跳过 mutex kernel。
    # 数学原因：1 head 时 softmax=1.0（恒等），2 heads 时互斥效果极弱且
    # score 扰动可能反而降低选择质量。省掉一次完整的 kernel launch。
    num_heads = log_s_all.shape[1] if log_s_all.dim() >= 2 else 0
    if num_heads <= 2:
        return log_s_all
    return cross_head_mutex_triton_bounds(
        log_s_all,
        token_lo,
        token_hi,
        alpha_cross,
        temperature,
        window,
        power,
    )


__all__ = [
    "AlphaFairSelectorConfig",
    "apply_cross_head_mutex",
    "apply_cross_head_mutex_bounds",
    "compute_alpha_scores_batched_layers_triton_logf_bounds",
    "compute_alpha_scores_batched_layers_triton_logf_pre_denom_bounds",
]
