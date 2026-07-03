"""Triton kernels for alpha selector helpers."""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl



def _read_int_env(key: str, default: int = 0) -> int:
    val = os.environ.get(key, str(default))
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


# ========== Triton Kernel 调优配置 ==========
# 通过环境变量控制 chunked kernel 的调优参数
# VLLM_ALPHA_BLOCK_K: block_k 大小 (默认 512)
# VLLM_ALPHA_NUM_WARPS: num_warps 数量 (默认 4)
# VLLM_ALPHA_NUM_STAGES: num_stages 数量 (默认 2)
_ALPHA_BLOCK_K_OVERRIDE = _read_int_env("VLLM_ALPHA_BLOCK_K", 0)
_ALPHA_NUM_WARPS_OVERRIDE = _read_int_env("VLLM_ALPHA_NUM_WARPS", 0)
_ALPHA_NUM_STAGES_OVERRIDE = _read_int_env("VLLM_ALPHA_NUM_STAGES", 0)
_ALPHA_LSE_1PASS_CACHED = os.environ.get("VLLM_SPARSE_ALPHA_LSE_1PASS", "1") == "1"


@triton.jit
def _cross_head_mutex_kernel(
    log_s_ptr,
    mask_ptr,
    out_ptr,
    stride_s_n,
    stride_s_h,
    stride_s_k,
    stride_m_n,
    stride_m_h,
    stride_m_k,
    H,
    K,
    alpha_cross,
    temperature,
    eps,
    BLOCK_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_h = tl.arange(0, BLOCK_H)

    mask_k = offs_k < K
    mask_h = offs_h < H

    ptr_s = log_s_ptr + pid_n * stride_s_n + offs_h[:, None] * stride_s_h + offs_k[None, :] * stride_s_k
    ptr_m = mask_ptr + pid_n * stride_m_n + offs_h[:, None] * stride_m_h + offs_k[None, :] * stride_m_k
    ptr_o = out_ptr + pid_n * stride_s_n + offs_h[:, None] * stride_s_h + offs_k[None, :] * stride_s_k

    in_bounds = mask_h[:, None] & mask_k[None, :]
    scores = tl.load(ptr_s, mask=in_bounds, other=0.0)
    mask_val = tl.load(ptr_m, mask=in_bounds, other=0).to(tl.int1)

    scores_f = scores.to(tl.float32)
    min_val = -3.402823466e38
    masked_logits = tl.where(mask_val, scores_f / temperature, min_val)

    max_h = tl.max(masked_logits, axis=0)
    exp = tl.exp(masked_logits - max_h)
    exp = tl.where(mask_val, exp, 0.0)
    sum_h = tl.sum(exp, axis=0)
    sum_h = tl.where(sum_h > 0.0, sum_h, 1.0)
    r = exp / sum_h

    log_r = tl.log(tl.maximum(r, eps))
    adjusted = scores_f + alpha_cross * log_r
    out = tl.where(mask_val, adjusted, scores_f)

    tl.store(ptr_o, out.to(scores.dtype), mask=in_bounds)


@triton.jit
def _cross_head_mutex_kernel_bounds(
    log_s_ptr,
    lo_ptr,
    hi_ptr,
    out_ptr,
    stride_s_n,
    stride_s_h,
    stride_s_k,
    stride_lo_n,
    stride_lo_h,
    stride_hi_n,
    stride_hi_h,
    H,
    K,
    alpha_cross,
    temperature,
    eps,
    BLOCK_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_h = tl.arange(0, BLOCK_H)

    mask_k = offs_k < K
    mask_h = offs_h < H

    lo = tl.load(lo_ptr + pid_n * stride_lo_n + offs_h * stride_lo_h, mask=mask_h, other=0).to(tl.int32)
    hi = tl.load(hi_ptr + pid_n * stride_hi_n + offs_h * stride_hi_h, mask=mask_h, other=0).to(tl.int32)
    lo = tl.maximum(lo, 0)
    hi = tl.minimum(hi, K)

    in_bounds = mask_h[:, None] & mask_k[None, :]
    ptr_s = log_s_ptr + pid_n * stride_s_n + offs_h[:, None] * stride_s_h + offs_k[None, :] * stride_s_k
    ptr_o = out_ptr + pid_n * stride_s_n + offs_h[:, None] * stride_s_h + offs_k[None, :] * stride_s_k
    scores = tl.load(ptr_s, mask=in_bounds, other=0.0)

    kk = offs_k[None, :].to(tl.int32)
    mask_val = (kk >= lo[:, None]) & (kk < hi[:, None])
    mask_val = mask_val & in_bounds

    scores_f = scores.to(tl.float32)
    min_val = -3.402823466e38
    masked_logits = tl.where(mask_val, scores_f / temperature, min_val)

    max_h = tl.max(masked_logits, axis=0)
    exp = tl.exp(masked_logits - max_h)
    exp = tl.where(mask_val, exp, 0.0)
    sum_h = tl.sum(exp, axis=0)
    sum_h = tl.where(sum_h > 0.0, sum_h, 1.0)
    r = exp / sum_h

    log_r = tl.log(tl.maximum(r, eps))
    adjusted = scores_f + alpha_cross * log_r
    out = tl.where(mask_val, adjusted, scores_f)

    tl.store(ptr_o, out.to(scores.dtype), mask=in_bounds)


@triton.jit
def _cross_head_softmax_kernel(
    log_s_ptr,
    mask_ptr,
    r_ptr,
    stride_s_n,
    stride_s_h,
    stride_s_k,
    stride_m_n,
    stride_m_h,
    stride_m_k,
    stride_r_n,
    stride_r_h,
    stride_r_k,
    H,
    K,
    temperature,
    BLOCK_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_h = tl.arange(0, BLOCK_H)
    mask_k = offs_k < K
    mask_h = offs_h < H

    ptr_s = log_s_ptr + pid_n * stride_s_n + offs_h[:, None] * stride_s_h + offs_k[None, :] * stride_s_k
    ptr_m = mask_ptr + pid_n * stride_m_n + offs_h[:, None] * stride_m_h + offs_k[None, :] * stride_m_k
    ptr_r = r_ptr + pid_n * stride_r_n + offs_h[:, None] * stride_r_h + offs_k[None, :] * stride_r_k

    in_bounds = mask_h[:, None] & mask_k[None, :]
    scores = tl.load(ptr_s, mask=in_bounds, other=0.0)
    mask_val = tl.load(ptr_m, mask=in_bounds, other=0).to(tl.int1)

    scores_f = scores.to(tl.float32)
    min_val = -3.402823466e38
    masked_logits = tl.where(mask_val, scores_f / temperature, min_val)

    max_h = tl.max(masked_logits, axis=0)
    exp = tl.exp(masked_logits - max_h)
    exp = tl.where(mask_val, exp, 0.0)
    sum_h = tl.sum(exp, axis=0)
    sum_h = tl.where(sum_h > 0.0, sum_h, 1.0)
    r = exp / sum_h
    r = tl.where(mask_val, r, 0.0)

    tl.store(ptr_r, r, mask=in_bounds)


@triton.jit
def _cross_head_softmax_kernel_bounds(
    log_s_ptr,
    lo_ptr,
    hi_ptr,
    r_ptr,
    stride_s_n,
    stride_s_h,
    stride_s_k,
    stride_lo_n,
    stride_lo_h,
    stride_hi_n,
    stride_hi_h,
    stride_r_n,
    stride_r_h,
    stride_r_k,
    H,
    K,
    temperature,
    BLOCK_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_h = tl.arange(0, BLOCK_H)
    mask_k = offs_k < K
    mask_h = offs_h < H

    lo = tl.load(lo_ptr + pid_n * stride_lo_n + offs_h * stride_lo_h, mask=mask_h, other=0).to(tl.int32)
    hi = tl.load(hi_ptr + pid_n * stride_hi_n + offs_h * stride_hi_h, mask=mask_h, other=0).to(tl.int32)
    lo = tl.maximum(lo, 0)
    hi = tl.minimum(hi, K)

    in_bounds = mask_h[:, None] & mask_k[None, :]
    ptr_s = log_s_ptr + pid_n * stride_s_n + offs_h[:, None] * stride_s_h + offs_k[None, :] * stride_s_k
    ptr_r = r_ptr + pid_n * stride_r_n + offs_h[:, None] * stride_r_h + offs_k[None, :] * stride_r_k
    scores = tl.load(ptr_s, mask=in_bounds, other=0.0)

    kk = offs_k[None, :].to(tl.int32)
    mask_val = (kk >= lo[:, None]) & (kk < hi[:, None])
    mask_val = mask_val & in_bounds

    scores_f = scores.to(tl.float32)
    min_val = -3.402823466e38
    masked_logits = tl.where(mask_val, scores_f / temperature, min_val)

    max_h = tl.max(masked_logits, axis=0)
    exp = tl.exp(masked_logits - max_h)
    exp = tl.where(mask_val, exp, 0.0)
    sum_h = tl.sum(exp, axis=0)
    sum_h = tl.where(sum_h > 0.0, sum_h, 1.0)
    r = exp / sum_h
    r = tl.where(mask_val, r, 0.0)

    tl.store(ptr_r, r, mask=in_bounds)


@triton.jit
def _cross_head_smooth_adjust_kernel(
    log_s_ptr,
    mask_ptr,
    r_ptr,
    kernel_ptr,
    out_ptr,
    stride_s_n,
    stride_s_h,
    stride_s_k,
    stride_m_n,
    stride_m_h,
    stride_m_k,
    stride_r_n,
    stride_r_h,
    stride_r_k,
    H,
    K,
    alpha_cross,
    beta_local,
    eps,
    BLOCK_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
    WND: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_h = tl.arange(0, BLOCK_H)
    mask_k = offs_k < K
    mask_h = offs_h < H

    ptr_s = log_s_ptr + pid_n * stride_s_n + offs_h[:, None] * stride_s_h + offs_k[None, :] * stride_s_k
    ptr_m = mask_ptr + pid_n * stride_m_n + offs_h[:, None] * stride_m_h + offs_k[None, :] * stride_m_k
    ptr_r = r_ptr + pid_n * stride_r_n + offs_h[:, None] * stride_r_h + offs_k[None, :] * stride_r_k
    ptr_o = out_ptr + pid_n * stride_s_n + offs_h[:, None] * stride_s_h + offs_k[None, :] * stride_s_k

    in_bounds = mask_h[:, None] & mask_k[None, :]
    scores = tl.load(ptr_s, mask=in_bounds, other=0.0)
    mask_val = tl.load(ptr_m, mask=in_bounds, other=0).to(tl.int1)
    r = tl.load(ptr_r, mask=in_bounds, other=0.0)

    scores_f = scores.to(tl.float32)
    r_f = r.to(tl.float32)

    smooth = tl.zeros([BLOCK_H, BLOCK_K], dtype=tl.float32)
    for offset in range(-WND, WND + 1):
        w = tl.load(kernel_ptr + (offset + WND))
        idx = offs_k + offset
        valid_k = (idx >= 0) & (idx < K)
        ptr_r_off = r_ptr + pid_n * stride_r_n + offs_h[:, None] * stride_r_h + idx[None, :] * stride_r_k
        r_off = tl.load(ptr_r_off, mask=mask_h[:, None] & valid_k[None, :], other=0.0)
        smooth += r_off.to(tl.float32) * w

    smooth = tl.where(mask_val, smooth, 0.0)

    log_r = tl.log(tl.maximum(r_f, eps))
    log_smooth = tl.log(tl.maximum(smooth, eps))
    adjusted = scores_f + alpha_cross * log_r + beta_local * log_smooth
    out = tl.where(mask_val, adjusted, scores_f)

    tl.store(ptr_o, out.to(scores.dtype), mask=in_bounds)


@triton.jit
def _cross_head_smooth_adjust_kernel_bounds(
    log_s_ptr,
    lo_ptr,
    hi_ptr,
    r_ptr,
    kernel_ptr,
    out_ptr,
    stride_s_n,
    stride_s_h,
    stride_s_k,
    stride_lo_n,
    stride_lo_h,
    stride_hi_n,
    stride_hi_h,
    stride_r_n,
    stride_r_h,
    stride_r_k,
    H,
    K,
    alpha_cross,
    beta_local,
    eps,
    BLOCK_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
    WND: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_h = tl.arange(0, BLOCK_H)
    mask_k = offs_k < K
    mask_h = offs_h < H

    lo = tl.load(lo_ptr + pid_n * stride_lo_n + offs_h * stride_lo_h, mask=mask_h, other=0).to(tl.int32)
    hi = tl.load(hi_ptr + pid_n * stride_hi_n + offs_h * stride_hi_h, mask=mask_h, other=0).to(tl.int32)
    lo = tl.maximum(lo, 0)
    hi = tl.minimum(hi, K)

    in_bounds = mask_h[:, None] & mask_k[None, :]

    ptr_s = log_s_ptr + pid_n * stride_s_n + offs_h[:, None] * stride_s_h + offs_k[None, :] * stride_s_k
    ptr_r = r_ptr + pid_n * stride_r_n + offs_h[:, None] * stride_r_h + offs_k[None, :] * stride_r_k
    ptr_o = out_ptr + pid_n * stride_s_n + offs_h[:, None] * stride_s_h + offs_k[None, :] * stride_s_k

    scores = tl.load(ptr_s, mask=in_bounds, other=0.0)
    r = tl.load(ptr_r, mask=in_bounds, other=0.0)

    kk = offs_k[None, :].to(tl.int32)
    mask_val = (kk >= lo[:, None]) & (kk < hi[:, None])
    mask_val = mask_val & in_bounds

    scores_f = scores.to(tl.float32)
    r_f = r.to(tl.float32)

    smooth = tl.zeros([BLOCK_H, BLOCK_K], dtype=tl.float32)
    for offset in range(-WND, WND + 1):
        w = tl.load(kernel_ptr + (offset + WND))
        idx = offs_k + offset
        valid_k = (idx >= 0) & (idx < K)
        ptr_r_off = r_ptr + pid_n * stride_r_n + offs_h[:, None] * stride_r_h + idx[None, :] * stride_r_k
        r_off = tl.load(ptr_r_off, mask=mask_h[:, None] & valid_k[None, :], other=0.0)
        smooth += r_off.to(tl.float32) * w

    smooth = tl.where(mask_val, smooth, 0.0)

    log_r = tl.log(tl.maximum(r_f, eps))
    log_smooth = tl.log(tl.maximum(smooth, eps))
    adjusted = scores_f + alpha_cross * log_r + beta_local * log_smooth
    out = tl.where(mask_val, adjusted, scores_f)

    tl.store(ptr_o, out.to(scores.dtype), mask=in_bounds)


@triton.jit
def _soft_nms_kernel(
    log_s_ptr,
    mask_ptr,
    out_ptr,
    stride_s_n,
    stride_s_h,
    stride_s_k,
    stride_m_n,
    stride_m_h,
    stride_m_k,
    stride_o_n,
    stride_o_h,
    stride_o_k,
    H,
    K,
    alpha,
    BLOCK_K: tl.constexpr,
    WND: tl.constexpr,
):
    pid_nh = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = offs_k < K

    n = pid_nh // H
    h = pid_nh - n * H

    ptr_s = log_s_ptr + n * stride_s_n + h * stride_s_h + offs_k * stride_s_k
    ptr_m = mask_ptr + n * stride_m_n + h * stride_m_h + offs_k * stride_m_k
    ptr_o = out_ptr + n * stride_o_n + h * stride_o_h + offs_k * stride_o_k

    scores = tl.load(ptr_s, mask=mask_k, other=-3.402823466e38)
    mask_val = tl.load(ptr_m, mask=mask_k, other=0).to(tl.int1)
    scores_f = scores.to(tl.float32)
    scores_f = tl.where(mask_val, scores_f, -3.402823466e38)

    pad = WND // 2
    # 兼容旧版本 Triton：避免使用 tl.full_like（部分环境缺失该 API）
    pooled = tl.zeros([BLOCK_K], dtype=tl.float32) + (-3.402823466e38)
    for offset in range(-pad, pad + 1):
        idx = offs_k + offset
        in_bounds = (idx >= 0) & (idx < K)
        ptr_off = log_s_ptr + n * stride_s_n + h * stride_s_h + idx * stride_s_k
        ptr_m_off = mask_ptr + n * stride_m_n + h * stride_m_h + idx * stride_m_k
        val = tl.load(ptr_off, mask=in_bounds, other=-3.402823466e38).to(tl.float32)
        m_off = tl.load(ptr_m_off, mask=in_bounds, other=0).to(tl.int1)
        val = tl.where(m_off, val, -3.402823466e38)
        pooled = tl.maximum(pooled, val)

    delta = tl.maximum(pooled - scores_f, 0.0)
    out = scores_f - alpha * delta
    out = tl.where(mask_val, out, scores_f)

    tl.store(ptr_o, out.to(scores.dtype), mask=mask_k)


def _select_block_h(num_heads: int) -> int:
    if num_heads <= 8:
        return 8
    if num_heads <= 16:
        return 16
    if num_heads <= 32:
        return 32
    return 64


def cross_head_mutex_triton(
    log_s_all: torch.Tensor,
    mask: torch.Tensor,
    alpha_cross: float,
    temperature: float,
    window: int,
    power: float,
    eps: float = 1.0e-12,
) -> torch.Tensor:
    if alpha_cross <= 0.0 or log_s_all.numel() == 0:
        return log_s_all
    if log_s_all.shape != mask.shape:
        raise ValueError("mask must match log_s_all")

    orig_shape = log_s_all.shape
    if log_s_all.dim() == 2:
        log_s = log_s_all.unsqueeze(0)
        mask_view = mask.unsqueeze(0)
    elif log_s_all.dim() == 3:
        log_s = log_s_all
        mask_view = mask
    elif log_s_all.dim() == 4:
        flat = log_s_all.shape[0] * log_s_all.shape[1]
        log_s = log_s_all.reshape(flat, log_s_all.shape[2], log_s_all.shape[3])
        mask_view = mask.reshape(flat, mask.shape[2], mask.shape[3])
    else:
        raise ValueError("log_s_all must be [H, K], [B, H, K], or [L, B, H, K]")

    if log_s.dim() != 3:
        raise ValueError("invalid log_s shape after reshape")

    num_batches, num_heads, kv_len = log_s.shape
    block_h = _select_block_h(num_heads)
    if num_heads > block_h:
        raise ValueError(f"num_heads {num_heads} exceeds supported block {block_h}")

    block_k = 128
    grid = (num_batches, triton.cdiv(kv_len, block_k))
    num_warps = 4 if block_h <= 32 else 8

    # Cache strides to avoid repeated Python→C++ round-trips.
    s_s0, s_s1, s_s2 = log_s.stride(0), log_s.stride(1), log_s.stride(2)
    s_m0, s_m1, s_m2 = mask_view.stride(0), mask_view.stride(1), mask_view.stride(2)

    out = torch.empty_like(log_s)

    if power <= 0.0 or window <= 0 or kv_len <= 1:
        _cross_head_mutex_kernel[grid](
            log_s,
            mask_view,
            out,
            s_s0, s_s1, s_s2,
            s_m0, s_m1, s_m2,
            num_heads,
            kv_len,
            float(alpha_cross),
            float(max(temperature, 1.0e-6)),
            float(eps),
            BLOCK_K=block_k,
            BLOCK_H=block_h,
            num_warps=num_warps,
        )
    else:
        r = torch.empty((num_batches, num_heads, kv_len), device=log_s.device, dtype=torch.float32)
        s_r0, s_r1, s_r2 = r.stride(0), r.stride(1), r.stride(2)
        _cross_head_softmax_kernel[grid](
            log_s,
            mask_view,
            r,
            s_s0, s_s1, s_s2,
            s_m0, s_m1, s_m2,
            s_r0, s_r1, s_r2,
            num_heads,
            kv_len,
            float(max(temperature, 1.0e-6)),
            BLOCK_K=block_k,
            BLOCK_H=block_h,
            num_warps=num_warps,
        )

        kernel = _build_gaussian_kernel_tensor(int(window), log_s.device, torch.float32)
        _cross_head_smooth_adjust_kernel[grid](
            log_s,
            mask_view,
            r,
            kernel,
            out,
            s_s0, s_s1, s_s2,
            s_m0, s_m1, s_m2,
            s_r0, s_r1, s_r2,
            num_heads,
            kv_len,
            float(alpha_cross),
            float(max(power, 0.0)),
            float(eps),
            BLOCK_K=block_k,
            BLOCK_H=block_h,
            WND=int(window),
            num_warps=num_warps,
        )

    if log_s_all.dim() == 2:
        return out.squeeze(0)
    if log_s_all.dim() == 4:
        return out.reshape(orig_shape)
    return out


def cross_head_mutex_triton_bounds(
    log_s_all: torch.Tensor,
    token_lo: torch.Tensor,
    token_hi: torch.Tensor,
    alpha_cross: float,
    temperature: float,
    window: int,
    power: float,
    eps: float = 1.0e-12,
) -> torch.Tensor:
    """cross-head mutex（bounds 版本）：不物化 [N,H,K] 的 bool mask。"""
    if alpha_cross <= 0.0 or log_s_all.numel() == 0:
        return log_s_all
    if not log_s_all.is_cuda:
        raise RuntimeError("cross_head_mutex_triton_bounds requires CUDA tensors")

    orig_shape = log_s_all.shape
    if log_s_all.dim() == 2:
        log_s = log_s_all.unsqueeze(0)
        lo_view = token_lo.unsqueeze(0) if token_lo.dim() == 1 else token_lo
        hi_view = token_hi.unsqueeze(0) if token_hi.dim() == 1 else token_hi
    elif log_s_all.dim() == 3:
        log_s = log_s_all
        lo_view = token_lo
        hi_view = token_hi
    elif log_s_all.dim() == 4:
        flat = log_s_all.shape[0] * log_s_all.shape[1]
        log_s = log_s_all.reshape(flat, log_s_all.shape[2], log_s_all.shape[3])
        lo_view = token_lo.reshape(flat, token_lo.shape[-1])
        hi_view = token_hi.reshape(flat, token_hi.shape[-1])
    else:
        raise ValueError("log_s_all must be [H,K], [B,H,K], or [L,B,H,K]")

    if log_s.dim() != 3:
        raise ValueError("invalid log_s shape after reshape")
    if lo_view.dim() != 2 or hi_view.dim() != 2:
        raise ValueError("token_lo/token_hi must be [N,H] (or broadcastable)")
    if lo_view.shape[0] != log_s.shape[0] or hi_view.shape[0] != log_s.shape[0]:
        raise ValueError("token_lo/token_hi batch dimension mismatch with log_s")
    if lo_view.shape[1] != log_s.shape[1] or hi_view.shape[1] != log_s.shape[1]:
        raise ValueError("token_lo/token_hi head dimension mismatch with log_s")

    num_batches, num_heads, kv_len = log_s.shape
    block_h = _select_block_h(num_heads)
    if num_heads > block_h:
        raise ValueError(f"num_heads {num_heads} exceeds supported block {block_h}")

    lo_i32 = (
        lo_view
        if (lo_view.device == log_s.device and lo_view.dtype == torch.int32)
        else lo_view.to(device=log_s.device, dtype=torch.int32)
    )
    hi_i32 = (
        hi_view
        if (hi_view.device == log_s.device and hi_view.dtype == torch.int32)
        else hi_view.to(device=log_s.device, dtype=torch.int32)
    )

    block_k = 128
    grid = (num_batches, triton.cdiv(kv_len, block_k))
    num_warps = 4 if block_h <= 32 else 8

    # Cache strides to avoid repeated Python→C++ round-trips.
    s_s0, s_s1, s_s2 = log_s.stride(0), log_s.stride(1), log_s.stride(2)
    s_lo0, s_lo1 = lo_i32.stride(0), lo_i32.stride(1)
    s_hi0, s_hi1 = hi_i32.stride(0), hi_i32.stride(1)

    out = torch.empty_like(log_s)

    if power <= 0.0 or window <= 0 or kv_len <= 1:
        _cross_head_mutex_kernel_bounds[grid](
            log_s,
            lo_i32,
            hi_i32,
            out,
            s_s0, s_s1, s_s2,
            s_lo0, s_lo1,
            s_hi0, s_hi1,
            num_heads,
            kv_len,
            float(alpha_cross),
            float(max(temperature, 1.0e-6)),
            float(eps),
            BLOCK_K=block_k,
            BLOCK_H=block_h,
            num_warps=num_warps,
        )
    else:
        r = torch.empty((num_batches, num_heads, kv_len), device=log_s.device, dtype=torch.float32)
        s_r0, s_r1, s_r2 = r.stride(0), r.stride(1), r.stride(2)
        _cross_head_softmax_kernel_bounds[grid](
            log_s,
            lo_i32,
            hi_i32,
            r,
            s_s0, s_s1, s_s2,
            s_lo0, s_lo1,
            s_hi0, s_hi1,
            s_r0, s_r1, s_r2,
            num_heads,
            kv_len,
            float(max(temperature, 1.0e-6)),
            BLOCK_K=block_k,
            BLOCK_H=block_h,
            num_warps=num_warps,
        )
        kernel = _build_gaussian_kernel_tensor(int(window), log_s.device, torch.float32)
        _cross_head_smooth_adjust_kernel_bounds[grid](
            log_s,
            lo_i32,
            hi_i32,
            r,
            kernel,
            out,
            s_s0, s_s1, s_s2,
            s_lo0, s_lo1,
            s_hi0, s_hi1,
            s_r0, s_r1, s_r2,
            num_heads,
            kv_len,
            float(alpha_cross),
            float(max(power, 0.0)),
            float(eps),
            BLOCK_K=block_k,
            BLOCK_H=block_h,
            WND=int(window),
            num_warps=num_warps,
        )

    if log_s_all.dim() == 2:
        return out.squeeze(0)
    if log_s_all.dim() == 4:
        return out.reshape(orig_shape)
    return out


def soft_nms_triton(
    log_s: torch.Tensor,
    mask: torch.Tensor,
    window: int,
    alpha: float,
) -> torch.Tensor:
    if log_s.numel() == 0:
        return log_s
    if not log_s.is_cuda or not mask.is_cuda:
        raise RuntimeError("soft_nms_triton requires CUDA tensors")
    if log_s.shape != mask.shape:
        raise ValueError("log_s/mask shape mismatch")
    if log_s.dim() != 3:
        raise ValueError("log_s must be [N, H, K]")

    N, H, K = log_s.shape
    if N == 0 or H == 0 or K == 0:
        return log_s
    win = int(max(1, min(window, K)))
    out = torch.empty_like(log_s)

    # Static grid — BLOCK_K=128 is a compile-time constant.
    grid = (N * H, triton.cdiv(K, 128))

    # Cache strides to avoid repeated Python→C++ round-trips.
    s_s0, s_s1, s_s2 = log_s.stride(0), log_s.stride(1), log_s.stride(2)
    s_m0, s_m1, s_m2 = mask.stride(0), mask.stride(1), mask.stride(2)
    s_o0, s_o1, s_o2 = out.stride(0), out.stride(1), out.stride(2)

    _soft_nms_kernel[grid](
        log_s,
        mask,
        out,
        s_s0, s_s1, s_s2,
        s_m0, s_m1, s_m2,
        s_o0, s_o1, s_o2,
        H,
        K,
        alpha,
        BLOCK_K=128,
        WND=win,
    )
    return out


@triton.jit
def _soft_nms_kernel_bounds(
    log_s_ptr,
    lo_ptr,
    hi_ptr,
    out_ptr,
    stride_s_n,
    stride_s_h,
    stride_s_k,
    stride_lo_n,
    stride_lo_h,
    stride_hi_n,
    stride_hi_h,
    stride_o_n,
    stride_o_h,
    stride_o_k,
    H,
    K,
    alpha,
    BLOCK_K: tl.constexpr,
    WND: tl.constexpr,
):
    pid_nh = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = offs_k < K

    n = pid_nh // H
    h = pid_nh - n * H

    lo = tl.load(lo_ptr + n * stride_lo_n + h * stride_lo_h).to(tl.int32)
    hi = tl.load(hi_ptr + n * stride_hi_n + h * stride_hi_h).to(tl.int32)
    lo = tl.maximum(lo, 0)
    hi = tl.minimum(hi, K)

    ptr_s = log_s_ptr + n * stride_s_n + h * stride_s_h + offs_k * stride_s_k
    ptr_o = out_ptr + n * stride_o_n + h * stride_o_h + offs_k * stride_o_k

    scores = tl.load(ptr_s, mask=mask_k, other=-float("inf"))
    scores_f = scores.to(tl.float32)
    kk = offs_k.to(tl.int32)
    mask_val = (kk >= lo) & (kk < hi) & mask_k
    scores_masked = tl.where(mask_val, scores_f, -float("inf"))

    pad = WND // 2
    pooled = tl.zeros([BLOCK_K], dtype=tl.float32) + (-float("inf"))
    for offset in range(-pad, pad + 1):
        idx = offs_k + offset
        in_bounds = (idx >= 0) & (idx < K)
        ptr_off = log_s_ptr + n * stride_s_n + h * stride_s_h + idx * stride_s_k
        val = tl.load(ptr_off, mask=in_bounds, other=-float("inf")).to(tl.float32)
        kk_off = idx.to(tl.int32)
        m_off = (kk_off >= lo) & (kk_off < hi) & in_bounds
        val = tl.where(m_off, val, -float("inf"))
        pooled = tl.maximum(pooled, val)

    delta = tl.maximum(pooled - scores_masked, 0.0)
    out_valid = scores_f - alpha * delta
    # masked-out positions must remain unchanged by contract (they are already -inf on input in mainline).
    out = tl.where(mask_val, out_valid, scores_f)

    tl.store(ptr_o, out.to(scores.dtype), mask=mask_k)


def soft_nms_triton_bounds(
    log_s_all: torch.Tensor,
    token_lo: torch.Tensor,
    token_hi: torch.Tensor,
    window: int,
    alpha: float,
) -> torch.Tensor:
    """soft-nms（bounds 版本）：不物化 [N,H,K] 的 bool mask。"""
    if log_s_all.numel() == 0:
        return log_s_all
    if not log_s_all.is_cuda:
        raise RuntimeError("soft_nms_triton_bounds requires CUDA tensors")
    if log_s_all.dim() != 3:
        raise ValueError("log_s_all must be [N, H, K]")
    if token_lo.dim() != 2 or token_hi.dim() != 2:
        raise ValueError("token_lo/token_hi must be [N, H]")
    if token_lo.shape != token_hi.shape:
        raise ValueError("token_lo/token_hi shape mismatch")
    if token_lo.shape[0] != log_s_all.shape[0] or token_lo.shape[1] != log_s_all.shape[1]:
        raise ValueError("token_lo/token_hi mismatch with log_s_all shape")

    N, H, K = log_s_all.shape
    if N == 0 or H == 0 or K == 0:
        return log_s_all
    win = int(max(1, min(window, K)))
    out = torch.empty_like(log_s_all)
    lo_i32 = token_lo.to(device=log_s_all.device, dtype=torch.int32)
    hi_i32 = token_hi.to(device=log_s_all.device, dtype=torch.int32)

    # Static grid — BLOCK_K=128 is a compile-time constant.
    grid = (N * H, triton.cdiv(K, 128))

    # Cache strides to avoid repeated Python→C++ round-trips.
    s_s0, s_s1, s_s2 = log_s_all.stride(0), log_s_all.stride(1), log_s_all.stride(2)
    s_lo0, s_lo1 = lo_i32.stride(0), lo_i32.stride(1)
    s_hi0, s_hi1 = hi_i32.stride(0), hi_i32.stride(1)
    s_o0, s_o1, s_o2 = out.stride(0), out.stride(1), out.stride(2)

    _soft_nms_kernel_bounds[grid](
        log_s_all,
        lo_i32,
        hi_i32,
        out,
        s_s0, s_s1, s_s2,
        s_lo0, s_lo1,
        s_hi0, s_hi1,
        s_o0, s_o1, s_o2,
        H,
        K,
        float(alpha),
        BLOCK_K=128,
        WND=win,
    )
    return out


@triton.jit
def _alpha_fuse_from_log_probs_bounds_rows1_kernel(
    log_probs_ptr,  # [M, K] float32
    key_norms_ptr,  # [M, K] float32
    lo_ptr,  # [M] int32
    hi_ptr,  # [M] int32
    row_counts_ptr,  # [M] float32
    out_ptr,  # [M, K] float32
    stride_lp_m: tl.constexpr,
    stride_lp_k: tl.constexpr,
    stride_kn_m: tl.constexpr,
    stride_kn_k: tl.constexpr,
    stride_lo: tl.constexpr,
    stride_hi: tl.constexpr,
    stride_rc: tl.constexpr,
    stride_o_m: tl.constexpr,
    stride_o_k: tl.constexpr,
    K,
    gamma: tl.constexpr,
    prior_weight_l2: tl.constexpr,
    prior_weight_pos: tl.constexpr,
    prior_pos_power: tl.constexpr,
    prior_pos_eta: tl.constexpr,
    beta: tl.constexpr,
    lambda_clip_single: tl.constexpr,
    lambda_clip_multi: tl.constexpr,
    lambda_tail_kappa: tl.constexpr,
    lambda_tail_pivot: tl.constexpr,
    lambda_soft: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs = tl.arange(0, BLOCK_K)

    lo = tl.load(lo_ptr + pid_m * stride_lo).to(tl.int32)
    hi = tl.load(hi_ptr + pid_m * stride_hi).to(tl.int32)
    lo = tl.maximum(lo, 0)
    hi = tl.minimum(hi, K)

    row_count = tl.load(row_counts_ptr + pid_m * stride_rc).to(tl.float32)
    has_rows = row_count > 0.0

    # ------------------------------------------------------------------
    # 0) token-wise normalization：denom_f = logsumexp(log_probs) over [lo, hi)
    # 1) denom_r = logsumexp(log_r_raw) over [lo, hi)
    #
    # 性能/编译期关键：用 streaming logsumexp 取代 (max pass + sum pass) 的两遍扫描，
    # 将每个 NUM_BLOCKS 的静态展开次数从 2x 降到 1x，避免长 prompt 下 Triton 代码膨胀导致超长编译。
    # ------------------------------------------------------------------
    max_lp = -float("inf")
    sum_exp_lp = 0.0
    max_r = -float("inf")
    sum_exp_r = 0.0

    for block_start in tl.static_range(0, NUM_BLOCKS):
        idx = block_start * BLOCK_K + offs
        mask_k = idx < K
        in_bounds = mask_k & (idx >= lo) & (idx < hi) & has_rows

        ptr_lp = log_probs_ptr + pid_m * stride_lp_m + idx * stride_lp_k
        lp = tl.load(ptr_lp, mask=mask_k, other=-float("inf")).to(tl.float32)
        lp = tl.where(in_bounds, lp, -float("inf"))

        block_max_lp = tl.max(lp, axis=0)
        new_max_lp = tl.maximum(max_lp, block_max_lp)
        new_max_lp_safe = tl.where(new_max_lp == -float("inf"), 0.0, new_max_lp)
        max_lp_safe = tl.where(max_lp == -float("inf"), 0.0, max_lp)
        scale_old_lp = tl.where(max_lp == -float("inf"), 0.0, tl.exp(max_lp_safe - new_max_lp_safe))
        lp_shift = lp - new_max_lp_safe
        lp_shift = tl.where(new_max_lp == -float("inf"), -float("inf"), lp_shift)
        sum_exp_lp = sum_exp_lp * scale_old_lp + tl.sum(tl.exp(lp_shift), axis=0)
        max_lp = new_max_lp

        ptr_kn = key_norms_ptr + pid_m * stride_kn_m + idx * stride_kn_k
        kn = tl.load(ptr_kn, mask=mask_k, other=0.0).to(tl.float32)
        kn = tl.maximum(kn, eps)
        log_pi = -gamma * tl.log(kn)

        denom_pos = tl.maximum(tl.cast(hi - lo - 1, tl.float32), 1.0)
        pos_norm = (idx.to(tl.float32) - lo.to(tl.float32)) / denom_pos
        pos_norm = tl.maximum(0.0, tl.minimum(1.0, pos_norm))
        pos_norm_pos = tl.maximum(pos_norm, eps)
        pos_shaped = tl.exp(prior_pos_power * tl.log(pos_norm_pos))
        pos_shaped = tl.where(pos_norm > 0.0, pos_shaped, 0.0)
        base_delta = -beta * pos_shaped
        one_minus = tl.maximum(tl.maximum(tl.cast(hi - 1, tl.float32) - idx.to(tl.float32), 0.0) / denom_pos, eps)
        if prior_pos_eta > 0.0:
            base_delta = base_delta + prior_pos_eta * tl.log(one_minus)
        log_delta = base_delta

        log_r_raw = prior_weight_l2 * log_pi + prior_weight_pos * log_delta
        log_r_raw = tl.where(in_bounds, log_r_raw, -float("inf"))

        block_max_r = tl.max(log_r_raw, axis=0)
        new_max_r = tl.maximum(max_r, block_max_r)
        new_max_r_safe = tl.where(new_max_r == -float("inf"), 0.0, new_max_r)
        max_r_safe = tl.where(max_r == -float("inf"), 0.0, max_r)
        scale_old_r = tl.where(max_r == -float("inf"), 0.0, tl.exp(max_r_safe - new_max_r_safe))
        r_shift = log_r_raw - new_max_r_safe
        r_shift = tl.where(new_max_r == -float("inf"), -float("inf"), r_shift)
        sum_exp_r = sum_exp_r * scale_old_r + tl.sum(tl.exp(r_shift), axis=0)
        max_r = new_max_r

    denom_f = max_lp + tl.log(tl.maximum(sum_exp_lp, eps))
    denom_f = tl.where(has_rows, denom_f, -float("inf"))

    denom_r = max_r + tl.log(tl.maximum(sum_exp_r, eps))
    denom_r = tl.where(has_rows, denom_r, -float("inf"))

    # ------------------------------------------------------------------
    # 2) ff/rr/fr (+ tail mean_pos)
    # ------------------------------------------------------------------
    sum_ff = 0.0
    sum_rr = 0.0
    sum_fr = 0.0
    prob_sum = 0.0
    pos_sum = 0.0

    for block_start in tl.static_range(0, NUM_BLOCKS):
        idx = block_start * BLOCK_K + offs
        mask_k = idx < K
        in_bounds = mask_k & (idx >= lo) & (idx < hi) & has_rows

        ptr_lp = log_probs_ptr + pid_m * stride_lp_m + idx * stride_lp_k
        lp = tl.load(ptr_lp, mask=mask_k, other=-float("inf")).to(tl.float32)
        lp = tl.where(in_bounds, lp, -float("inf")) - denom_f

        ptr_kn = key_norms_ptr + pid_m * stride_kn_m + idx * stride_kn_k
        kn = tl.load(ptr_kn, mask=mask_k, other=0.0).to(tl.float32)
        kn = tl.maximum(kn, eps)
        log_pi = -gamma * tl.log(kn)
        denom_pos = tl.maximum(tl.cast(hi - lo - 1, tl.float32), 1.0)
        pos_norm = (idx.to(tl.float32) - lo.to(tl.float32)) / denom_pos
        pos_norm = tl.maximum(0.0, tl.minimum(1.0, pos_norm))
        pos_norm_pos = tl.maximum(pos_norm, eps)
        pos_shaped = tl.exp(prior_pos_power * tl.log(pos_norm_pos))
        pos_shaped = tl.where(pos_norm > 0.0, pos_shaped, 0.0)
        base_delta = -beta * pos_shaped
        one_minus = tl.maximum(tl.maximum(tl.cast(hi - 1, tl.float32) - idx.to(tl.float32), 0.0) / denom_pos, eps)
        if prior_pos_eta > 0.0:
            base_delta = base_delta + prior_pos_eta * tl.log(one_minus)
        log_delta = base_delta

        log_r_raw = prior_weight_l2 * log_pi + prior_weight_pos * log_delta
        log_r = log_r_raw - denom_r
        log_r = tl.where(in_bounds, log_r, -float("inf"))

        sum_ff += tl.sum(tl.where(in_bounds, tl.exp(2.0 * lp), 0.0), axis=0)
        sum_rr += tl.sum(tl.where(in_bounds, tl.exp(2.0 * log_r), 0.0), axis=0)
        sum_fr += tl.sum(tl.where(in_bounds, tl.exp(lp + log_r), 0.0), axis=0)

        if lambda_tail_kappa > 0.0:
            p = tl.exp(lp)
            p = tl.where(in_bounds, p, 0.0)
            prob_sum += tl.sum(p, axis=0)
            pos_sum += tl.sum(p * pos_norm, axis=0)

    ff = sum_ff
    rr = sum_rr
    fr = sum_fr

    denom_lam = tl.maximum(ff - 2.0 * fr + rr, eps)
    lam_star = (ff - fr) / denom_lam
    lam = tl.maximum(lam_star, 0.0)

    if lambda_tail_kappa > 0.0:
        prob_sum_safe = tl.maximum(prob_sum, eps)
        mean_pos = pos_sum / prob_sum_safe
        tail_excess = tl.maximum(mean_pos - lambda_tail_pivot, 0.0)
        lam = lam * (1.0 + lambda_tail_kappa * tail_excess)

    # 对齐 torch 语义：row_count>1 时使用 lambda_clip_multi
    lam_max = tl.where(row_count > 1.0, lambda_clip_multi, lambda_clip_single)
    if lambda_soft:
        clip = tl.maximum(lam_max, eps)
        x = lam / clip
        x_abs = tl.abs(x)
        e = tl.exp(-2.0 * x_abs)
        tanh_abs = (1.0 - e) / (1.0 + e)
        tanh_x = tl.where(x >= 0.0, tanh_abs, -tanh_abs)
        lam = clip * tanh_x
    else:
        lam = tl.minimum(lam, lam_max)

    lam = tl.maximum(lam, 0.0)
    lam = tl.minimum(lam, 1.0 - eps)

    log_one_minus = tl.log(tl.maximum(1.0 - lam, eps))
    log_lambda = tl.log(lam + eps)

    # ------------------------------------------------------------------
    # 3) denom_fused = logsumexp(fused_raw)  (streaming)
    # ------------------------------------------------------------------
    max_fused = -float("inf")
    sum_exp_fused = 0.0
    for block_start in tl.static_range(0, NUM_BLOCKS):
        idx = block_start * BLOCK_K + offs
        mask_k = idx < K
        in_bounds = mask_k & (idx >= lo) & (idx < hi) & has_rows

        ptr_lp = log_probs_ptr + pid_m * stride_lp_m + idx * stride_lp_k
        lp = tl.load(ptr_lp, mask=mask_k, other=-float("inf")).to(tl.float32)
        lp = tl.where(in_bounds, lp, -float("inf")) - denom_f

        ptr_kn = key_norms_ptr + pid_m * stride_kn_m + idx * stride_kn_k
        kn = tl.load(ptr_kn, mask=mask_k, other=0.0).to(tl.float32)
        kn = tl.maximum(kn, eps)
        log_pi = -gamma * tl.log(kn)
        denom_pos = tl.maximum(tl.cast(hi - lo - 1, tl.float32), 1.0)
        pos_norm = (idx.to(tl.float32) - lo.to(tl.float32)) / denom_pos
        pos_norm = tl.maximum(0.0, tl.minimum(1.0, pos_norm))
        pos_norm_pos = tl.maximum(pos_norm, eps)
        pos_shaped = tl.exp(prior_pos_power * tl.log(pos_norm_pos))
        pos_shaped = tl.where(pos_norm > 0.0, pos_shaped, 0.0)
        base_delta = -beta * pos_shaped
        one_minus = tl.maximum(tl.maximum(tl.cast(hi - 1, tl.float32) - idx.to(tl.float32), 0.0) / denom_pos, eps)
        if prior_pos_eta > 0.0:
            base_delta = base_delta + prior_pos_eta * tl.log(one_minus)
        log_delta = base_delta
        log_r_raw = prior_weight_l2 * log_pi + prior_weight_pos * log_delta
        log_r = tl.where(in_bounds, log_r_raw - denom_r, -float("inf"))

        a = log_one_minus + lp
        b = log_lambda + log_r
        m = tl.maximum(a, b)
        fused_raw = m + tl.log(tl.exp(a - m) + tl.exp(b - m))
        fused_raw = tl.where(in_bounds, fused_raw, -float("inf"))

        block_max_f = tl.max(fused_raw, axis=0)
        new_max_f = tl.maximum(max_fused, block_max_f)
        new_max_f_safe = tl.where(new_max_f == -float("inf"), 0.0, new_max_f)
        max_fused_safe = tl.where(max_fused == -float("inf"), 0.0, max_fused)
        scale_old_f = tl.where(max_fused == -float("inf"), 0.0, tl.exp(max_fused_safe - new_max_f_safe))
        fused_shift = fused_raw - new_max_f_safe
        fused_shift = tl.where(new_max_f == -float("inf"), -float("inf"), fused_shift)
        sum_exp_fused = sum_exp_fused * scale_old_f + tl.sum(tl.exp(fused_shift), axis=0)
        max_fused = new_max_f

    denom_fused = max_fused + tl.log(tl.maximum(sum_exp_fused, eps))
    denom_fused = tl.where(has_rows, denom_fused, -float("inf"))

    # ------------------------------------------------------------------
    # 4) Write fused normalized
    # ------------------------------------------------------------------
    for block_start in tl.static_range(0, NUM_BLOCKS):
        idx = block_start * BLOCK_K + offs
        mask_k = idx < K
        in_bounds = mask_k & (idx >= lo) & (idx < hi) & has_rows

        ptr_lp = log_probs_ptr + pid_m * stride_lp_m + idx * stride_lp_k
        lp = tl.load(ptr_lp, mask=mask_k, other=-float("inf")).to(tl.float32)
        lp = tl.where(in_bounds, lp, -float("inf")) - denom_f

        ptr_kn = key_norms_ptr + pid_m * stride_kn_m + idx * stride_kn_k
        kn = tl.load(ptr_kn, mask=mask_k, other=0.0).to(tl.float32)
        kn = tl.maximum(kn, eps)
        log_pi = -gamma * tl.log(kn)
        denom_pos = tl.maximum(tl.cast(hi - lo - 1, tl.float32), 1.0)
        pos_norm = (idx.to(tl.float32) - lo.to(tl.float32)) / denom_pos
        pos_norm = tl.maximum(0.0, tl.minimum(1.0, pos_norm))
        pos_norm_pos = tl.maximum(pos_norm, eps)
        pos_shaped = tl.exp(prior_pos_power * tl.log(pos_norm_pos))
        pos_shaped = tl.where(pos_norm > 0.0, pos_shaped, 0.0)
        base_delta = -beta * pos_shaped
        one_minus = tl.maximum(tl.maximum(tl.cast(hi - 1, tl.float32) - idx.to(tl.float32), 0.0) / denom_pos, eps)
        if prior_pos_eta > 0.0:
            base_delta = base_delta + prior_pos_eta * tl.log(one_minus)
        log_delta = base_delta
        log_r_raw = prior_weight_l2 * log_pi + prior_weight_pos * log_delta
        log_r = tl.where(in_bounds, log_r_raw - denom_r, -float("inf"))

        a = log_one_minus + lp
        b = log_lambda + log_r
        m = tl.maximum(a, b)
        fused_raw = m + tl.log(tl.exp(a - m) + tl.exp(b - m))
        fused = fused_raw - denom_fused
        fused = tl.where(in_bounds, fused, -float("inf"))

        ptr_o = out_ptr + pid_m * stride_o_m + idx * stride_o_k
        tl.store(ptr_o, fused, mask=mask_k)


def alpha_fuse_from_log_probs_bounds_rows1_triton(
    log_probs: torch.Tensor,
    key_norms: torch.Tensor,
    token_lo: torch.Tensor,
    token_hi: torch.Tensor,
    row_counts: torch.Tensor,
    *,
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
    eps: float,
    block_k: int = 256,
) -> torch.Tensor:
    """rows==1 bounds 下的 fused 计算（不物化 token_mask），输出 fused（soft-nms 前）。"""
    if log_probs.numel() == 0:
        return log_probs
    if not log_probs.is_cuda or not key_norms.is_cuda:
        raise RuntimeError("alpha_fuse_from_log_probs_bounds_rows1_triton requires CUDA tensors")
    if log_probs.dim() != 2:
        raise ValueError("log_probs must be [M, K]")
    if key_norms.shape != log_probs.shape:
        raise ValueError("key_norms shape mismatch with log_probs")
    if token_lo.dim() != 1 or token_hi.dim() != 1:
        raise ValueError("token_lo/token_hi must be [M]")
    if token_lo.shape[0] != log_probs.shape[0] or token_hi.shape[0] != log_probs.shape[0]:
        raise ValueError("token_lo/token_hi length mismatch")
    if row_counts.dim() != 1 or row_counts.shape[0] != log_probs.shape[0]:
        raise ValueError("row_counts must be [M]")
    if log_probs.dtype != torch.float32:
        raise ValueError("log_probs must be float32")
    if not key_norms.dtype.is_floating_point:
        raise ValueError("key_norms must be a float dtype")

    m, k = log_probs.shape

    # 长上下文（例如 needle two_parts 这种 K~2e4）下，旧的"单 kernel + static_range 扫 K"会触发
    # Triton 编译期代码膨胀（CPU 满、GPU 空、看似卡死）。
    # 对 k>4096 直接切换到 block-parallel 的多 kernel + torch(GPU) reduction 实现：编译开销与 K 解耦。
    if int(k) > 4096:
        return _alpha_fuse_from_log_probs_bounds_rows1_chunked_triton(
            log_probs,
            key_norms,
            token_lo,
            token_hi,
            row_counts,
            gamma=gamma,
            prior_weight_l2=prior_weight_l2,
            prior_weight_pos=prior_weight_pos,
            prior_pos_power=prior_pos_power,
            prior_pos_eta=prior_pos_eta,
            beta=beta,
            lambda_clip_single=lambda_clip_single,
            lambda_clip_multi=lambda_clip_multi,
            lambda_tail_kappa=lambda_tail_kappa,
            lambda_tail_pivot=lambda_tail_pivot,
            lambda_soft=lambda_soft,
            eps=eps,
            block_k=block_k,
        )

    out = torch.empty_like(log_probs)
    lo_i32 = token_lo.to(device=log_probs.device, dtype=torch.int32)
    hi_i32 = token_hi.to(device=log_probs.device, dtype=torch.int32)
    rc_f32 = row_counts.to(device=log_probs.device, dtype=torch.float32)

    if k <= 128:
        block_k = 128
    elif k <= 4096:
        block_k = int(block_k)
    else:
        block_k = max(int(block_k), 512)
    num_blocks = triton.cdiv(int(k), int(block_k))
    if int(num_blocks) <= 16:
        num_blocks_bucket = 16
    elif int(num_blocks) <= 32:
        num_blocks_bucket = 32
    else:
        num_blocks_bucket = ((int(num_blocks) + 15) // 16) * 16
    # 避免过大的 NUM_BLOCKS 触发 Triton 代码膨胀（长 prompt 下会显著增加编译时间）。
    num_blocks_bucket = int(min(int(num_blocks_bucket), 128))
    grid = (m,)
    _alpha_fuse_from_log_probs_bounds_rows1_kernel[grid](
        log_probs,
        key_norms,
        lo_i32,
        hi_i32,
        rc_f32,
        out,
        int(log_probs.stride(0)),
        int(log_probs.stride(1)),
        int(key_norms.stride(0)),
        int(key_norms.stride(1)),
        int(lo_i32.stride(0)),
        int(hi_i32.stride(0)),
        int(rc_f32.stride(0)),
        int(out.stride(0)),
        int(out.stride(1)),
        K=int(k),
        gamma=float(gamma),
        prior_weight_l2=float(prior_weight_l2),
        prior_weight_pos=float(prior_weight_pos),
        prior_pos_power=float(max(prior_pos_power, 1.0)),
        prior_pos_eta=float(prior_pos_eta),
        beta=float(beta),
        lambda_clip_single=float(lambda_clip_single),
        lambda_clip_multi=float(lambda_clip_multi),
        lambda_tail_kappa=float(lambda_tail_kappa),
        lambda_tail_pivot=float(lambda_tail_pivot),
        lambda_soft=bool(lambda_soft),
        eps=float(eps),
        BLOCK_K=int(block_k),
        NUM_BLOCKS=int(num_blocks_bucket),
        num_warps=4,
        # G1 优化：num_stages=3 比 2 在 kernel 层面快约 30%
        num_stages=3,
    )
    return out


@triton.jit
def _alpha_rows1_block_stats_lp_r_kernel(
    log_probs_ptr,  # [M, K] float32
    key_norms_ptr,  # [M, K] float32
    lo_ptr,  # [M] int32
    hi_ptr,  # [M] int32
    row_counts_ptr,  # [M] float32
    out_lp_max_ptr,  # [M, NB] float32
    out_lp_sumexp_ptr,  # [M, NB] float32
    out_r_max_ptr,  # [M, NB] float32
    out_r_sumexp_ptr,  # [M, NB] float32
    stride_lp_m: tl.constexpr,
    stride_lp_k: tl.constexpr,
    stride_kn_m: tl.constexpr,
    stride_kn_k: tl.constexpr,
    stride_lo: tl.constexpr,
    stride_hi: tl.constexpr,
    stride_rc: tl.constexpr,
    stride_out_m: tl.constexpr,
    stride_out_b: tl.constexpr,
    K,
    gamma: tl.constexpr,
    prior_weight_l2: tl.constexpr,
    prior_weight_pos: tl.constexpr,
    prior_pos_power: tl.constexpr,
    prior_pos_eta: tl.constexpr,
    beta: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)

    offs = tl.arange(0, BLOCK_K)
    idx = pid_b * BLOCK_K + offs
    mask_k = idx < K

    lo = tl.load(lo_ptr + pid_m * stride_lo).to(tl.int32)
    hi = tl.load(hi_ptr + pid_m * stride_hi).to(tl.int32)
    lo = tl.maximum(lo, 0)
    hi = tl.minimum(hi, K)

    row_count = tl.load(row_counts_ptr + pid_m * stride_rc).to(tl.float32)
    has_rows = row_count > 0.0

    in_bounds = mask_k & (idx >= lo) & (idx < hi) & has_rows

    ptr_lp = log_probs_ptr + pid_m * stride_lp_m + idx * stride_lp_k
    lp = tl.load(ptr_lp, mask=mask_k, other=-float("inf")).to(tl.float32)
    lp = tl.where(in_bounds, lp, -float("inf"))

    ptr_kn = key_norms_ptr + pid_m * stride_kn_m + idx * stride_kn_k
    kn = tl.load(ptr_kn, mask=mask_k, other=0.0).to(tl.float32)
    kn = tl.maximum(kn, eps)
    log_pi = -gamma * tl.log(kn)

    denom_pos = tl.maximum(tl.cast(hi - lo - 1, tl.float32), 1.0)
    pos_norm = (idx.to(tl.float32) - lo.to(tl.float32)) / denom_pos
    pos_norm = tl.maximum(0.0, tl.minimum(1.0, pos_norm))
    pos_norm_pos = tl.maximum(pos_norm, eps)
    pos_shaped = tl.exp(prior_pos_power * tl.log(pos_norm_pos))
    pos_shaped = tl.where(pos_norm > 0.0, pos_shaped, 0.0)
    base_delta = -beta * pos_shaped
    one_minus = tl.maximum(tl.maximum(tl.cast(hi - 1, tl.float32) - idx.to(tl.float32), 0.0) / denom_pos, eps)
    if prior_pos_eta > 0.0:
        base_delta = base_delta + prior_pos_eta * tl.log(one_minus)
    log_delta = base_delta
    log_r_raw = prior_weight_l2 * log_pi + prior_weight_pos * log_delta
    log_r_raw = tl.where(in_bounds, log_r_raw, -float("inf"))

    valid_lp = in_bounds & (lp > -float("inf"))
    count_lp = tl.sum(valid_lp.to(tl.int32), axis=0)
    has_any_lp = count_lp > 0
    count_r = tl.sum(in_bounds.to(tl.int32), axis=0)
    has_any_r = count_r > 0

    lp_max = tl.max(tl.where(valid_lp, lp, -float("inf")), axis=0)
    lp_max_safe = tl.where(has_any_lp, lp_max, 0.0)
    lp_vals = tl.where(valid_lp, lp, lp_max_safe)
    lp_exp = tl.exp(lp_vals - lp_max_safe)
    lp_exp = tl.where(valid_lp, lp_exp, 0.0)
    lp_sumexp = tl.sum(lp_exp, axis=0)
    lp_max = tl.where(has_any_lp, lp_max, -float("inf"))

    r_max = tl.max(log_r_raw, axis=0)
    r_max_safe = tl.where(has_any_r, r_max, 0.0)
    r_vals = tl.where(in_bounds, log_r_raw, r_max_safe)
    r_exp = tl.exp(r_vals - r_max_safe)
    r_exp = tl.where(in_bounds, r_exp, 0.0)
    r_sumexp = tl.sum(r_exp, axis=0)
    r_max = tl.where(has_any_r, r_max, -float("inf"))

    out_off = pid_m * stride_out_m + pid_b * stride_out_b
    tl.store(out_lp_max_ptr + out_off, lp_max)
    tl.store(out_lp_sumexp_ptr + out_off, lp_sumexp)
    tl.store(out_r_max_ptr + out_off, r_max)
    tl.store(out_r_sumexp_ptr + out_off, r_sumexp)


@triton.jit
def _alpha_rows1_block_sums_kernel(
    log_probs_ptr,  # [M, K] float32
    key_norms_ptr,  # [M, K] float32
    lo_ptr,  # [M] int32
    hi_ptr,  # [M] int32
    row_counts_ptr,  # [M] float32
    denom_f_ptr,  # [M] float32
    denom_r_ptr,  # [M] float32
    out_ff_ptr,  # [M, NB] float32
    out_rr_ptr,  # [M, NB] float32
    out_fr_ptr,  # [M, NB] float32
    out_prob_ptr,  # [M, NB] float32
    out_pos_ptr,  # [M, NB] float32
    stride_lp_m: tl.constexpr,
    stride_lp_k: tl.constexpr,
    stride_kn_m: tl.constexpr,
    stride_kn_k: tl.constexpr,
    stride_lo: tl.constexpr,
    stride_hi: tl.constexpr,
    stride_rc: tl.constexpr,
    stride_df: tl.constexpr,
    stride_dr: tl.constexpr,
    stride_out_m: tl.constexpr,
    stride_out_b: tl.constexpr,
    K,
    gamma: tl.constexpr,
    prior_weight_l2: tl.constexpr,
    prior_weight_pos: tl.constexpr,
    prior_pos_power: tl.constexpr,
    prior_pos_eta: tl.constexpr,
    beta: tl.constexpr,
    lambda_tail_kappa: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)

    offs = tl.arange(0, BLOCK_K)
    idx = pid_b * BLOCK_K + offs
    mask_k = idx < K

    lo = tl.load(lo_ptr + pid_m * stride_lo).to(tl.int32)
    hi = tl.load(hi_ptr + pid_m * stride_hi).to(tl.int32)
    lo = tl.maximum(lo, 0)
    hi = tl.minimum(hi, K)

    row_count = tl.load(row_counts_ptr + pid_m * stride_rc).to(tl.float32)
    has_rows = row_count > 0.0

    denom_f = tl.load(denom_f_ptr + pid_m * stride_df).to(tl.float32)
    denom_r = tl.load(denom_r_ptr + pid_m * stride_dr).to(tl.float32)
    has_df = denom_f > -float("inf")
    has_dr = denom_r > -float("inf")

    in_bounds = mask_k & (idx >= lo) & (idx < hi) & has_rows & has_df & has_dr

    ptr_lp = log_probs_ptr + pid_m * stride_lp_m + idx * stride_lp_k
    lp_raw = tl.load(ptr_lp, mask=mask_k, other=-float("inf")).to(tl.float32)
    lp = tl.where(in_bounds, lp_raw - denom_f, -float("inf"))

    ptr_kn = key_norms_ptr + pid_m * stride_kn_m + idx * stride_kn_k
    kn = tl.load(ptr_kn, mask=mask_k, other=0.0).to(tl.float32)
    kn = tl.maximum(kn, eps)
    log_pi = -gamma * tl.log(kn)

    denom_pos = tl.maximum(tl.cast(hi - lo - 1, tl.float32), 1.0)
    pos_norm = (idx.to(tl.float32) - lo.to(tl.float32)) / denom_pos
    pos_norm = tl.maximum(0.0, tl.minimum(1.0, pos_norm))
    pos_norm_pos = tl.maximum(pos_norm, eps)
    pos_shaped = tl.exp(prior_pos_power * tl.log(pos_norm_pos))
    pos_shaped = tl.where(pos_norm > 0.0, pos_shaped, 0.0)
    base_delta = -beta * pos_shaped
    one_minus = tl.maximum(tl.maximum(tl.cast(hi - 1, tl.float32) - idx.to(tl.float32), 0.0) / denom_pos, eps)
    if prior_pos_eta > 0.0:
        base_delta = base_delta + prior_pos_eta * tl.log(one_minus)
    log_delta = base_delta
    log_r_raw = prior_weight_l2 * log_pi + prior_weight_pos * log_delta
    log_r = tl.where(in_bounds, log_r_raw - denom_r, -float("inf"))

    ff = tl.sum(tl.where(in_bounds, tl.exp(2.0 * lp), 0.0), axis=0)
    rr = tl.sum(tl.where(in_bounds, tl.exp(2.0 * log_r), 0.0), axis=0)
    fr = tl.sum(tl.where(in_bounds, tl.exp(lp + log_r), 0.0), axis=0)

    prob_sum = 0.0
    pos_sum = 0.0
    if lambda_tail_kappa > 0.0:
        p = tl.exp(lp)
        p = tl.where(in_bounds, p, 0.0)
        prob_sum = tl.sum(p, axis=0)
        pos_sum = tl.sum(p * pos_norm, axis=0)

    out_off = pid_m * stride_out_m + pid_b * stride_out_b
    tl.store(out_ff_ptr + out_off, ff)
    tl.store(out_rr_ptr + out_off, rr)
    tl.store(out_fr_ptr + out_off, fr)
    tl.store(out_prob_ptr + out_off, prob_sum)
    tl.store(out_pos_ptr + out_off, pos_sum)


@triton.jit
def _alpha_rows1_block_stats_fused_kernel(
    log_probs_ptr,  # [M, K] float32
    key_norms_ptr,  # [M, K] float32
    lo_ptr,  # [M] int32
    hi_ptr,  # [M] int32
    row_counts_ptr,  # [M] float32
    denom_f_ptr,  # [M] float32
    denom_r_ptr,  # [M] float32
    log_one_minus_ptr,  # [M] float32
    log_lambda_ptr,  # [M] float32
    out_f_max_ptr,  # [M, NB] float32
    out_f_sumexp_ptr,  # [M, NB] float32
    stride_lp_m: tl.constexpr,
    stride_lp_k: tl.constexpr,
    stride_kn_m: tl.constexpr,
    stride_kn_k: tl.constexpr,
    stride_lo: tl.constexpr,
    stride_hi: tl.constexpr,
    stride_rc: tl.constexpr,
    stride_df: tl.constexpr,
    stride_dr: tl.constexpr,
    stride_lom: tl.constexpr,
    stride_ll: tl.constexpr,
    stride_out_m: tl.constexpr,
    stride_out_b: tl.constexpr,
    K,
    gamma: tl.constexpr,
    prior_weight_l2: tl.constexpr,
    prior_weight_pos: tl.constexpr,
    prior_pos_power: tl.constexpr,
    prior_pos_eta: tl.constexpr,
    beta: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)

    offs = tl.arange(0, BLOCK_K)
    idx = pid_b * BLOCK_K + offs
    mask_k = idx < K

    lo = tl.load(lo_ptr + pid_m * stride_lo).to(tl.int32)
    hi = tl.load(hi_ptr + pid_m * stride_hi).to(tl.int32)
    lo = tl.maximum(lo, 0)
    hi = tl.minimum(hi, K)

    row_count = tl.load(row_counts_ptr + pid_m * stride_rc).to(tl.float32)
    has_rows = row_count > 0.0

    denom_f = tl.load(denom_f_ptr + pid_m * stride_df).to(tl.float32)
    denom_r = tl.load(denom_r_ptr + pid_m * stride_dr).to(tl.float32)
    log_one_minus = tl.load(log_one_minus_ptr + pid_m * stride_lom).to(tl.float32)
    log_lambda = tl.load(log_lambda_ptr + pid_m * stride_ll).to(tl.float32)

    has_df = denom_f > -float("inf")
    has_dr = denom_r > -float("inf")

    in_bounds = mask_k & (idx >= lo) & (idx < hi) & has_rows & has_df & has_dr

    ptr_lp = log_probs_ptr + pid_m * stride_lp_m + idx * stride_lp_k
    lp_raw = tl.load(ptr_lp, mask=mask_k, other=-float("inf")).to(tl.float32)
    lp = tl.where(in_bounds, lp_raw - denom_f, -float("inf"))

    ptr_kn = key_norms_ptr + pid_m * stride_kn_m + idx * stride_kn_k
    kn = tl.load(ptr_kn, mask=mask_k, other=0.0).to(tl.float32)
    kn = tl.maximum(kn, eps)
    log_pi = -gamma * tl.log(kn)

    denom_pos = tl.maximum(tl.cast(hi - lo - 1, tl.float32), 1.0)
    pos_norm = (idx.to(tl.float32) - lo.to(tl.float32)) / denom_pos
    pos_norm = tl.maximum(0.0, tl.minimum(1.0, pos_norm))
    pos_norm_pos = tl.maximum(pos_norm, eps)
    pos_shaped = tl.exp(prior_pos_power * tl.log(pos_norm_pos))
    pos_shaped = tl.where(pos_norm > 0.0, pos_shaped, 0.0)
    base_delta = -beta * pos_shaped
    one_minus = tl.maximum(tl.maximum(tl.cast(hi - 1, tl.float32) - idx.to(tl.float32), 0.0) / denom_pos, eps)
    if prior_pos_eta > 0.0:
        base_delta = base_delta + prior_pos_eta * tl.log(one_minus)
    log_delta = base_delta
    log_r_raw = prior_weight_l2 * log_pi + prior_weight_pos * log_delta
    log_r = tl.where(in_bounds, log_r_raw - denom_r, -float("inf"))

    a = log_one_minus + lp
    b = log_lambda + log_r
    mm = tl.maximum(a, b)
    fused_raw = mm + tl.log(tl.exp(a - mm) + tl.exp(b - mm))
    fused_raw = tl.where(in_bounds, fused_raw, -float("inf"))

    count = tl.sum(in_bounds.to(tl.int32), axis=0)
    has_any = count > 0

    f_max = tl.max(fused_raw, axis=0)
    f_max_safe = tl.where(has_any, f_max, 0.0)
    f_vals = tl.where(in_bounds, fused_raw, f_max_safe)
    f_exp = tl.exp(f_vals - f_max_safe)
    f_exp = tl.where(in_bounds, f_exp, 0.0)
    f_sumexp = tl.sum(f_exp, axis=0)
    f_max = tl.where(has_any, f_max, -float("inf"))

    out_off = pid_m * stride_out_m + pid_b * stride_out_b
    tl.store(out_f_max_ptr + out_off, f_max)
    tl.store(out_f_sumexp_ptr + out_off, f_sumexp)


@triton.jit
def _alpha_rows1_block_write_fused_kernel(
    log_probs_ptr,  # [M, K] float32
    key_norms_ptr,  # [M, K] float32
    lo_ptr,  # [M] int32
    hi_ptr,  # [M] int32
    row_counts_ptr,  # [M] float32
    denom_f_ptr,  # [M] float32
    denom_r_ptr,  # [M] float32
    log_one_minus_ptr,  # [M] float32
    log_lambda_ptr,  # [M] float32
    denom_fused_ptr,  # [M] float32
    out_ptr,  # [M, K] float32
    stride_lp_m: tl.constexpr,
    stride_lp_k: tl.constexpr,
    stride_kn_m: tl.constexpr,
    stride_kn_k: tl.constexpr,
    stride_lo: tl.constexpr,
    stride_hi: tl.constexpr,
    stride_rc: tl.constexpr,
    stride_df: tl.constexpr,
    stride_dr: tl.constexpr,
    stride_lom: tl.constexpr,
    stride_ll: tl.constexpr,
    stride_dfused: tl.constexpr,
    stride_out_m: tl.constexpr,
    stride_out_k: tl.constexpr,
    K,
    gamma: tl.constexpr,
    prior_weight_l2: tl.constexpr,
    prior_weight_pos: tl.constexpr,
    prior_pos_power: tl.constexpr,
    prior_pos_eta: tl.constexpr,
    beta: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)

    offs = tl.arange(0, BLOCK_K)
    idx = pid_b * BLOCK_K + offs
    mask_k = idx < K

    lo = tl.load(lo_ptr + pid_m * stride_lo).to(tl.int32)
    hi = tl.load(hi_ptr + pid_m * stride_hi).to(tl.int32)
    lo = tl.maximum(lo, 0)
    hi = tl.minimum(hi, K)

    row_count = tl.load(row_counts_ptr + pid_m * stride_rc).to(tl.float32)
    has_rows = row_count > 0.0

    denom_f = tl.load(denom_f_ptr + pid_m * stride_df).to(tl.float32)
    denom_r = tl.load(denom_r_ptr + pid_m * stride_dr).to(tl.float32)
    log_one_minus = tl.load(log_one_minus_ptr + pid_m * stride_lom).to(tl.float32)
    log_lambda = tl.load(log_lambda_ptr + pid_m * stride_ll).to(tl.float32)
    denom_fused = tl.load(denom_fused_ptr + pid_m * stride_dfused).to(tl.float32)

    has_df = denom_f > -float("inf")
    has_dr = denom_r > -float("inf")
    has_dfused = denom_fused > -float("inf")

    in_bounds = mask_k & (idx >= lo) & (idx < hi) & has_rows & has_df & has_dr & has_dfused

    ptr_lp = log_probs_ptr + pid_m * stride_lp_m + idx * stride_lp_k
    lp_raw = tl.load(ptr_lp, mask=mask_k, other=-float("inf")).to(tl.float32)
    lp = tl.where(in_bounds, lp_raw - denom_f, -float("inf"))

    ptr_kn = key_norms_ptr + pid_m * stride_kn_m + idx * stride_kn_k
    kn = tl.load(ptr_kn, mask=mask_k, other=0.0).to(tl.float32)
    kn = tl.maximum(kn, eps)
    log_pi = -gamma * tl.log(kn)

    denom_pos = tl.maximum(tl.cast(hi - lo - 1, tl.float32), 1.0)
    pos_norm = (idx.to(tl.float32) - lo.to(tl.float32)) / denom_pos
    pos_norm = tl.maximum(0.0, tl.minimum(1.0, pos_norm))
    pos_norm_pos = tl.maximum(pos_norm, eps)
    pos_shaped = tl.exp(prior_pos_power * tl.log(pos_norm_pos))
    pos_shaped = tl.where(pos_norm > 0.0, pos_shaped, 0.0)
    base_delta = -beta * pos_shaped
    one_minus = tl.maximum(tl.maximum(tl.cast(hi - 1, tl.float32) - idx.to(tl.float32), 0.0) / denom_pos, eps)
    if prior_pos_eta > 0.0:
        base_delta = base_delta + prior_pos_eta * tl.log(one_minus)
    log_delta = base_delta
    log_r_raw = prior_weight_l2 * log_pi + prior_weight_pos * log_delta
    log_r = tl.where(in_bounds, log_r_raw - denom_r, -float("inf"))

    a = log_one_minus + lp
    b = log_lambda + log_r
    mm = tl.maximum(a, b)
    fused_raw = mm + tl.log(tl.exp(a - mm) + tl.exp(b - mm))
    fused = fused_raw - denom_fused
    fused = tl.where(in_bounds, fused, -float("inf"))

    ptr_o = out_ptr + pid_m * stride_out_m + idx * stride_out_k
    tl.store(ptr_o, fused, mask=mask_k)


def _reduce_block_logsumexp(
    block_max: torch.Tensor,
    block_sumexp: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    if block_max.dim() != 2 or block_sumexp.dim() != 2 or block_max.shape != block_sumexp.shape:
        raise ValueError("block_max/block_sumexp must be same shape [M, NB]")

    max_val = torch.max(block_max, dim=1).values
    has_any = torch.isfinite(max_val)
    max_val_safe = torch.where(has_any, max_val, torch.zeros_like(max_val))

    scale = torch.exp(block_max - max_val_safe[:, None])
    sum_exp = torch.sum(block_sumexp * scale, dim=1)
    out = max_val_safe + torch.log(torch.clamp(sum_exp, min=float(eps)))
    return torch.where(has_any, out, torch.full_like(out, float("-inf")))


def _alpha_fuse_from_log_probs_bounds_rows1_chunked_triton(
    log_probs: torch.Tensor,
    key_norms: torch.Tensor,
    token_lo: torch.Tensor,
    token_hi: torch.Tensor,
    row_counts: torch.Tensor,
    *,
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
    eps: float,
    block_k: int = 256,
) -> torch.Tensor:
    if log_probs.dim() != 2:
        raise ValueError("log_probs must be [M, K]")
    if key_norms.shape != log_probs.shape:
        raise ValueError("key_norms shape mismatch with log_probs")
    if log_probs.dtype != torch.float32:
        raise ValueError("log_probs must be float32")
    if not key_norms.dtype.is_floating_point:
        raise ValueError("key_norms must be a float dtype")
    if not log_probs.is_cuda:
        raise RuntimeError("_alpha_fuse_from_log_probs_bounds_rows1_chunked_triton requires CUDA tensors")

    m, k = log_probs.shape
    if k <= 0 or m <= 0:
        return torch.empty_like(log_probs)

    # 调优参数：优先使用环境变量覆盖
    if _ALPHA_BLOCK_K_OVERRIDE > 0:
        block_k = _ALPHA_BLOCK_K_OVERRIDE
    elif k <= 128:
        block_k = 128
    elif k <= 8192:
        block_k = max(int(block_k), 256)
    else:
        block_k = max(int(block_k), 512)

    num_warps = _ALPHA_NUM_WARPS_OVERRIDE if _ALPHA_NUM_WARPS_OVERRIDE > 0 else 4
    # G1 优化：num_stages=3 比 2 在 kernel 层面快约 30%（cp.async 流水线更深）
    num_stages = _ALPHA_NUM_STAGES_OVERRIDE if _ALPHA_NUM_STAGES_OVERRIDE > 0 else 3

    num_blocks = triton.cdiv(int(k), int(block_k))

    lo_i32 = token_lo.to(device=log_probs.device, dtype=torch.int32)
    hi_i32 = token_hi.to(device=log_probs.device, dtype=torch.int32)
    rc_f32 = row_counts.to(device=log_probs.device, dtype=torch.float32)

    lp_max = torch.empty((m, num_blocks), device=log_probs.device, dtype=torch.float32)
    lp_sumexp = torch.empty_like(lp_max)
    r_max = torch.empty_like(lp_max)
    r_sumexp = torch.empty_like(lp_max)

    grid = (m, num_blocks)
    _alpha_rows1_block_stats_lp_r_kernel[grid](
        log_probs,
        key_norms,
        lo_i32,
        hi_i32,
        rc_f32,
        lp_max,
        lp_sumexp,
        r_max,
        r_sumexp,
        int(log_probs.stride(0)),
        int(log_probs.stride(1)),
        int(key_norms.stride(0)),
        int(key_norms.stride(1)),
        int(lo_i32.stride(0)),
        int(hi_i32.stride(0)),
        int(rc_f32.stride(0)),
        int(lp_max.stride(0)),
        int(lp_max.stride(1)),
        K=int(k),
        gamma=float(gamma),
        prior_weight_l2=float(prior_weight_l2),
        prior_weight_pos=float(prior_weight_pos),
        prior_pos_power=float(max(float(prior_pos_power), 1.0)),
        prior_pos_eta=float(prior_pos_eta),
        beta=float(beta),
        eps=float(eps),
        BLOCK_K=int(block_k),
        num_warps=num_warps,
        num_stages=num_stages,
    )

    denom_f = _reduce_block_logsumexp(lp_max, lp_sumexp, eps=float(eps))
    denom_r = _reduce_block_logsumexp(r_max, r_sumexp, eps=float(eps))

    has_rows = rc_f32 > 0.0
    denom_f = torch.where(has_rows, denom_f, torch.full_like(denom_f, float("-inf")))
    denom_r = torch.where(has_rows, denom_r, torch.full_like(denom_r, float("-inf")))

    ff_blk = torch.empty((m, num_blocks), device=log_probs.device, dtype=torch.float32)
    rr_blk = torch.empty_like(ff_blk)
    fr_blk = torch.empty_like(ff_blk)
    prob_blk = torch.empty_like(ff_blk)
    pos_blk = torch.empty_like(ff_blk)

    _alpha_rows1_block_sums_kernel[grid](
        log_probs,
        key_norms,
        lo_i32,
        hi_i32,
        rc_f32,
        denom_f,
        denom_r,
        ff_blk,
        rr_blk,
        fr_blk,
        prob_blk,
        pos_blk,
        int(log_probs.stride(0)),
        int(log_probs.stride(1)),
        int(key_norms.stride(0)),
        int(key_norms.stride(1)),
        int(lo_i32.stride(0)),
        int(hi_i32.stride(0)),
        int(rc_f32.stride(0)),
        int(denom_f.stride(0)),
        int(denom_r.stride(0)),
        int(ff_blk.stride(0)),
        int(ff_blk.stride(1)),
        K=int(k),
        gamma=float(gamma),
        prior_weight_l2=float(prior_weight_l2),
        prior_weight_pos=float(prior_weight_pos),
        prior_pos_power=float(max(float(prior_pos_power), 1.0)),
        prior_pos_eta=float(prior_pos_eta),
        beta=float(beta),
        lambda_tail_kappa=float(lambda_tail_kappa),
        eps=float(eps),
        BLOCK_K=int(block_k),
        num_warps=num_warps,
        num_stages=num_stages,
    )

    ff = ff_blk.sum(dim=1)
    rr = rr_blk.sum(dim=1)
    fr = fr_blk.sum(dim=1)
    prob_sum = prob_blk.sum(dim=1)
    pos_sum = pos_blk.sum(dim=1)

    denom_lam = torch.clamp(ff - 2.0 * fr + rr, min=float(eps))
    lam_star = (ff - fr) / denom_lam
    lam = torch.clamp(lam_star, min=0.0)

    if float(lambda_tail_kappa) > 0.0:
        prob_sum_safe = torch.clamp(prob_sum, min=float(eps))
        mean_pos = pos_sum / prob_sum_safe
        tail_excess = torch.clamp(mean_pos - float(lambda_tail_pivot), min=0.0)
        lam = lam * (1.0 + float(lambda_tail_kappa) * tail_excess)

    lam_max = torch.where(
        rc_f32 > 1.0,
        float(lambda_clip_multi),
        float(lambda_clip_single),
    )
    if bool(lambda_soft):
        clip = torch.clamp(lam_max, min=float(eps))
        lam = clip * torch.tanh(lam / clip)
    else:
        lam = torch.minimum(lam, lam_max)

    lam = torch.clamp(lam, min=0.0, max=1.0 - float(eps))
    log_one_minus = torch.log(torch.clamp(1.0 - lam, min=float(eps)))
    log_lambda = torch.log(lam + float(eps))

    f_max = torch.empty((m, num_blocks), device=log_probs.device, dtype=torch.float32)
    f_sumexp = torch.empty_like(f_max)
    _alpha_rows1_block_stats_fused_kernel[grid](
        log_probs,
        key_norms,
        lo_i32,
        hi_i32,
        rc_f32,
        denom_f,
        denom_r,
        log_one_minus,
        log_lambda,
        f_max,
        f_sumexp,
        int(log_probs.stride(0)),
        int(log_probs.stride(1)),
        int(key_norms.stride(0)),
        int(key_norms.stride(1)),
        int(lo_i32.stride(0)),
        int(hi_i32.stride(0)),
        int(rc_f32.stride(0)),
        int(denom_f.stride(0)),
        int(denom_r.stride(0)),
        int(log_one_minus.stride(0)),
        int(log_lambda.stride(0)),
        int(f_max.stride(0)),
        int(f_max.stride(1)),
        K=int(k),
        gamma=float(gamma),
        prior_weight_l2=float(prior_weight_l2),
        prior_weight_pos=float(prior_weight_pos),
        prior_pos_power=float(max(float(prior_pos_power), 1.0)),
        prior_pos_eta=float(prior_pos_eta),
        beta=float(beta),
        eps=float(eps),
        BLOCK_K=int(block_k),
        num_warps=num_warps,
        num_stages=num_stages,
    )

    # Pass 4: reduce block logsumexp + write fused output
    denom_fused = _reduce_block_logsumexp(f_max, f_sumexp, eps)
    out = torch.empty_like(log_probs)
    _alpha_rows1_block_write_fused_kernel[grid](
        log_probs,
        key_norms,
        lo_i32,
        hi_i32,
        rc_f32,
        denom_f,
        denom_r,
        log_one_minus,
        log_lambda,
        denom_fused,
        out,
        int(log_probs.stride(0)),
        int(log_probs.stride(1)),
        int(key_norms.stride(0)),
        int(key_norms.stride(1)),
        int(lo_i32.stride(0)),
        int(hi_i32.stride(0)),
        int(rc_f32.stride(0)),
        int(denom_f.stride(0)),
        int(denom_r.stride(0)),
        int(log_one_minus.stride(0)),
        int(log_lambda.stride(0)),
        int(denom_fused.stride(0)),
        int(out.stride(0)),
        int(out.stride(1)),
        K=int(k),
        gamma=float(gamma),
        prior_weight_l2=float(prior_weight_l2),
        prior_weight_pos=float(prior_weight_pos),
        prior_pos_power=float(max(float(prior_pos_power), 1.0)),
        prior_pos_eta=float(prior_pos_eta),
        beta=float(beta),
        eps=float(eps),
        BLOCK_K=int(block_k),
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out

@triton.jit
def _row_lse_from_logits_bounds_kernel(
    logits_ptr,  # [M, R, K]
    lo_ptr,  # [M, R] int32
    hi_ptr,  # [M, R] int32
    row_lse_ptr,  # [M, R] float32
    row_valid_ptr,  # [M, R] int8
    stride_l_m,
    stride_l_r,
    stride_l_k,
    stride_lo_m,
    stride_lo_r,
    stride_hi_m,
    stride_hi_r,
    stride_lse_m,
    stride_lse_r,
    stride_valid_m,
    stride_valid_r,
    K,
    BLOCK_K: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_r = tl.program_id(1)

    offs_k = tl.arange(0, BLOCK_K)
    min_val = -3.402823466e38
    max_val = min_val
    count = 0.0

    lo = tl.load(lo_ptr + pid_m * stride_lo_m + pid_r * stride_lo_r).to(tl.int32)
    hi = tl.load(hi_ptr + pid_m * stride_hi_m + pid_r * stride_hi_r).to(tl.int32)

    for block_start in tl.static_range(0, NUM_BLOCKS):
        idx = block_start * BLOCK_K + offs_k
        mask_k = idx < K
        in_bounds = mask_k & (idx >= lo) & (idx < hi)

        ptr = logits_ptr + pid_m * stride_l_m + pid_r * stride_l_r + idx * stride_l_k
        vals = tl.load(ptr, mask=mask_k, other=min_val).to(tl.float32)
        finite = vals > min_val
        valid = in_bounds & finite

        vals = tl.where(valid, vals, min_val)
        block_max = tl.max(vals, axis=0)
        max_val = tl.maximum(max_val, block_max)
        count += tl.sum(valid.to(tl.float32), axis=0)

    has_vals = count > 0.0
    max_val = tl.where(has_vals, max_val, 0.0)

    sum_exp = 0.0
    for block_start in tl.static_range(0, NUM_BLOCKS):
        idx = block_start * BLOCK_K + offs_k
        mask_k = idx < K
        in_bounds = mask_k & (idx >= lo) & (idx < hi)

        ptr = logits_ptr + pid_m * stride_l_m + pid_r * stride_l_r + idx * stride_l_k
        vals = tl.load(ptr, mask=mask_k, other=min_val).to(tl.float32)
        finite = vals > min_val
        valid = in_bounds & finite

        vals = tl.where(valid, vals, max_val)
        exp = tl.exp(vals - max_val)
        exp = tl.where(valid, exp, 0.0)
        sum_exp += tl.sum(exp, axis=0)

    lse = max_val + tl.log(sum_exp)
    lse = tl.where(has_vals, lse, min_val)

    tl.store(row_lse_ptr + pid_m * stride_lse_m + pid_r * stride_lse_r, lse)
    tl.store(
        row_valid_ptr + pid_m * stride_valid_m + pid_r * stride_valid_r,
        has_vals.to(tl.int8),
    )


@triton.jit
def _row_lse_from_logits_bounds_1pass_kernel(
    logits_ptr,  # [M, R, K]
    lo_ptr,  # [M, R] int32
    hi_ptr,  # [M, R] int32
    row_lse_ptr,  # [M, R] float32
    row_valid_ptr,  # [M, R] int8
    stride_l_m,
    stride_l_r,
    stride_l_k,
    stride_lo_m,
    stride_lo_r,
    stride_hi_m,
    stride_hi_r,
    stride_lse_m,
    stride_lse_r,
    stride_valid_m,
    stride_valid_r,
    K,
    BLOCK_K: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
):
    """Compute per-row logsumexp (LSE) in a single pass over K blocks.

    This replaces the 2-pass (max + sumexp) implementation to reduce memory
    bandwidth. It still uses a numerically-stable running max + sumexp update.
    """
    pid_m = tl.program_id(0)
    pid_r = tl.program_id(1)

    offs_k = tl.arange(0, BLOCK_K)
    min_val = -3.402823466e38

    lo = tl.load(lo_ptr + pid_m * stride_lo_m + pid_r * stride_lo_r).to(tl.int32)
    hi = tl.load(hi_ptr + pid_m * stride_hi_m + pid_r * stride_hi_r).to(tl.int32)

    max_val = min_val
    sum_exp = 0.0
    count = 0.0

    for block_start in tl.static_range(0, NUM_BLOCKS):
        idx = block_start * BLOCK_K + offs_k
        mask_k = idx < K
        in_bounds = mask_k & (idx >= lo) & (idx < hi)

        ptr = logits_ptr + pid_m * stride_l_m + pid_r * stride_l_r + idx * stride_l_k
        vals = tl.load(ptr, mask=mask_k, other=min_val).to(tl.float32)
        finite = vals > min_val
        valid = in_bounds & finite

        vals = tl.where(valid, vals, min_val)
        block_max = tl.max(vals, axis=0)

        new_max = tl.maximum(max_val, block_max)
        # 注意：当 valid 全 False 时，vals==min_val，exp(...) 会被 valid mask 清零，因此 sum_block==0。
        exp_block = tl.exp(vals - new_max)
        exp_block = tl.where(valid, exp_block, 0.0)
        sum_block = tl.sum(exp_block, axis=0)

        sum_exp = sum_exp * tl.exp(max_val - new_max) + sum_block
        max_val = new_max
        count += tl.sum(valid.to(tl.float32), axis=0)

    has_vals = count > 0.0
    lse = max_val + tl.log(sum_exp)
    lse = tl.where(has_vals, lse, min_val)

    tl.store(row_lse_ptr + pid_m * stride_lse_m + pid_r * stride_lse_r, lse)
    tl.store(
        row_valid_ptr + pid_m * stride_valid_m + pid_r * stride_valid_r,
        has_vals.to(tl.int8),
    )


@triton.jit
def _log_f_mean_probs_from_logits_bounds_kernel(
    logits_ptr,  # [M, R, K]
    row_lse_ptr,  # [M, R] float32
    row_valid_ptr,  # [M, R] int8
    lo_ptr,  # [M, R] int32
    hi_ptr,  # [M, R] int32
    row_counts_ptr,  # [M] float32
    out_log_f_ptr,  # [M, K] float32
    out_mean_ptr,  # [M, K] float32 (optional)
    stride_l_m: tl.constexpr,
    stride_l_r: tl.constexpr,
    stride_l_k: tl.constexpr,
    stride_lse_m: tl.constexpr,
    stride_lse_r: tl.constexpr,
    stride_valid_m: tl.constexpr,
    stride_valid_r: tl.constexpr,
    stride_lo_m: tl.constexpr,
    stride_lo_r: tl.constexpr,
    stride_hi_m: tl.constexpr,
    stride_hi_r: tl.constexpr,
    stride_counts: tl.constexpr,
    stride_out_m: tl.constexpr,
    stride_out_k: tl.constexpr,
    stride_mean_m: tl.constexpr,
    stride_mean_k: tl.constexpr,
    K,
    alpha,
    eps,
    R: tl.constexpr,
    USE_MEAN: tl.constexpr,
    STORE_MEAN: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = offs_k < K

    min_val = -3.402823466e38
    row_count = tl.load(row_counts_ptr + pid_m * stride_counts).to(tl.float32)
    has_rows = row_count > 0.0
    row_count = tl.where(has_rows, row_count, 1.0)

    if USE_MEAN:
        log_sum = tl.zeros([BLOCK_K], dtype=tl.float32)
    else:
        log_f_max = tl.full([BLOCK_K], min_val, dtype=tl.float32)
        log_f_sum = tl.zeros([BLOCK_K], dtype=tl.float32)

    if STORE_MEAN:
        prob_sum = tl.zeros([BLOCK_K], dtype=tl.float32)

    for r in tl.static_range(0, R):
        row_has = tl.load(row_valid_ptr + pid_m * stride_valid_m + r * stride_valid_r).to(tl.int1)
        row_lse = tl.load(row_lse_ptr + pid_m * stride_lse_m + r * stride_lse_r).to(tl.float32)
        lo = tl.load(lo_ptr + pid_m * stride_lo_m + r * stride_lo_r).to(tl.int32)
        hi = tl.load(hi_ptr + pid_m * stride_hi_m + r * stride_hi_r).to(tl.int32)

        in_bounds = mask_k & (offs_k >= lo) & (offs_k < hi)

        ptr = logits_ptr + pid_m * stride_l_m + r * stride_l_r + offs_k * stride_l_k
        vals = tl.load(ptr, mask=mask_k, other=min_val).to(tl.float32)
        finite = vals > min_val
        valid = row_has & in_bounds & finite

        log_probs = vals - row_lse
        log_probs = tl.where(valid, log_probs, min_val)

        if USE_MEAN:
            log_sum += tl.where(valid, log_probs, 0.0)
        else:
            log_f_pre = alpha * log_probs
            new_max = tl.where(valid, tl.maximum(log_f_max, log_f_pre), log_f_max)
            update = log_f_sum * tl.exp(log_f_max - new_max) + tl.exp(log_f_pre - new_max)
            log_f_sum = tl.where(valid, update, log_f_sum)
            log_f_max = new_max

        if STORE_MEAN:
            prob = tl.exp(log_probs)
            prob = tl.where(valid, prob, 0.0)
            prob_sum += prob

    if USE_MEAN:
        log_f = log_sum / row_count
    else:
        lse = log_f_max + tl.log(log_f_sum)
        log_f = (lse - tl.log(row_count)) / alpha

    log_f = tl.where(has_rows, log_f, min_val)

    tl.store(out_log_f_ptr + pid_m * stride_out_m + offs_k * stride_out_k, log_f, mask=mask_k)
    if STORE_MEAN:
        mean_probs = prob_sum / row_count
        mean_probs = tl.where(has_rows, mean_probs, 0.0)
        tl.store(out_mean_ptr + pid_m * stride_mean_m + offs_k * stride_mean_k, mean_probs, mask=mask_k)


def log_f_mean_probs_from_logits_bounds_triton(
    logits: torch.Tensor,
    lo: torch.Tensor,
    hi: torch.Tensor,
    alpha: float,
    eps: float,
    *,
    return_mean_probs: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if logits.numel() == 0:
        empty = logits.new_empty((0, 0), dtype=torch.float32)
        counts = logits.new_empty((0,), dtype=torch.float32)
        return empty, empty, counts
    if not logits.is_cuda:
        raise RuntimeError("log_f_mean_probs_from_logits_bounds_triton requires CUDA tensors")
    if logits.dim() != 3:
        raise ValueError("logits must be [M, R, K]")
    if lo.shape != logits.shape[:2] or hi.shape != logits.shape[:2]:
        raise ValueError("lo/hi must be [M, R] and match logits leading dims")

    m, r, k = logits.shape
    if r > 128:
        raise RuntimeError("log_f_mean_probs_from_logits_bounds_triton only supports R<=128")

    if k <= 128:
        block_k = 128
    elif k <= 4096:
        block_k = 256
    else:
        block_k = 512

    lo_i32 = lo.to(device=logits.device, dtype=torch.int32)
    hi_i32 = hi.to(device=logits.device, dtype=torch.int32)

    row_lse = torch.empty((m, r), device=logits.device, dtype=torch.float32)
    row_valid = torch.empty((m, r), device=logits.device, dtype=torch.int8)

    grid_lse = (m, r)
    num_blocks = triton.cdiv(int(k), int(block_k))
    if int(num_blocks) <= 16:
        num_blocks_bucket = 16
    elif int(num_blocks) <= 32:
        num_blocks_bucket = 32
    else:
        num_blocks_bucket = ((int(num_blocks) + 15) // 16) * 16
    # 避免过大的 NUM_BLOCKS 触发 Triton 代码膨胀（长 prompt 下会显著增加编译时间）。
    num_blocks_bucket = int(min(int(num_blocks_bucket), 128))
    use_1pass = _ALPHA_LSE_1PASS_CACHED
    lse_kernel = _row_lse_from_logits_bounds_1pass_kernel if use_1pass else _row_lse_from_logits_bounds_kernel
    lse_kernel[grid_lse](
        logits,
        lo_i32,
        hi_i32,
        row_lse,
        row_valid,
        logits.stride(0),
        logits.stride(1),
        logits.stride(2),
        lo_i32.stride(0),
        lo_i32.stride(1),
        hi_i32.stride(0),
        hi_i32.stride(1),
        row_lse.stride(0),
        row_lse.stride(1),
        row_valid.stride(0),
        row_valid.stride(1),
        K=k,
        BLOCK_K=block_k,
        NUM_BLOCKS=num_blocks_bucket,
    )

    row_counts = row_valid.to(torch.float32).sum(dim=1).contiguous()

    log_f = torch.empty((m, k), device=logits.device, dtype=torch.float32)
    mean_probs: Optional[torch.Tensor]
    if return_mean_probs:
        mean_probs = torch.empty_like(log_f)
        mean_ptr = mean_probs
        stride_mean_m = mean_probs.stride(0)
        stride_mean_k = mean_probs.stride(1)
    else:
        mean_probs = None
        mean_ptr = log_f
        stride_mean_m = log_f.stride(0)
        stride_mean_k = log_f.stride(1)

    grid_log = (m, triton.cdiv(k, block_k))
    use_mean = abs(float(alpha)) < 1.0e-6
    _log_f_mean_probs_from_logits_bounds_kernel[grid_log](
        logits,
        row_lse,
        row_valid,
        lo_i32,
        hi_i32,
        row_counts,
        log_f,
        mean_ptr,
        int(logits.stride(0)),
        int(logits.stride(1)),
        int(logits.stride(2)),
        int(row_lse.stride(0)),
        int(row_lse.stride(1)),
        int(row_valid.stride(0)),
        int(row_valid.stride(1)),
        int(lo_i32.stride(0)),
        int(lo_i32.stride(1)),
        int(hi_i32.stride(0)),
        int(hi_i32.stride(1)),
        int(row_counts.stride(0)),
        int(log_f.stride(0)),
        int(log_f.stride(1)),
        int(stride_mean_m),
        int(stride_mean_k),
        K=k,
        alpha=float(alpha),
        eps=float(eps),
        R=r,
        USE_MEAN=use_mean,
        STORE_MEAN=return_mean_probs,
        BLOCK_K=block_k,
    )

    if mean_probs is None:
        mean_probs_out = log_f.new_empty((0, 0))
    else:
        mean_probs_out = mean_probs
    return log_f, mean_probs_out, row_counts


@triton.jit
def _log_f_mean_probs_from_log_f_pre_denom_bounds_kernel(
    log_f_pre_ptr,  # [M, R, K]
    denom_ptr,  # [M, R] float32
    lo_ptr,  # [M, R] int32
    hi_ptr,  # [M, R] int32
    row_counts_ptr,  # [M] float32
    out_log_f_ptr,  # [M, K] float32
    out_mean_ptr,  # [M, K] float32 (optional)
    stride_lfp_m: tl.constexpr,
    stride_lfp_r: tl.constexpr,
    stride_lfp_k: tl.constexpr,
    stride_d_m: tl.constexpr,
    stride_d_r: tl.constexpr,
    stride_lo_m: tl.constexpr,
    stride_lo_r: tl.constexpr,
    stride_hi_m: tl.constexpr,
    stride_hi_r: tl.constexpr,
    stride_counts_m: tl.constexpr,
    stride_out_m: tl.constexpr,
    stride_out_k: tl.constexpr,
    stride_mean_m: tl.constexpr,
    stride_mean_k: tl.constexpr,
    K,
    alpha: tl.constexpr,
    eps: tl.constexpr,
    R: tl.constexpr,
    USE_MEAN: tl.constexpr,
    STORE_MEAN: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = offs_k < K

    min_val = float("-inf")
    row_count = tl.load(row_counts_ptr + pid_m * stride_counts_m).to(tl.float32)
    has_rows = row_count > 0.0
    row_count_safe = tl.maximum(row_count, 1.0)

    log_f_max = tl.full([BLOCK_K], min_val, dtype=tl.float32)
    log_f_sum = tl.zeros([BLOCK_K], dtype=tl.float32)
    log_sum = tl.zeros([BLOCK_K], dtype=tl.float32)
    if STORE_MEAN:
        prob_sum = tl.zeros([BLOCK_K], dtype=tl.float32)

    for r in range(R):
        lo = tl.load(lo_ptr + pid_m * stride_lo_m + r * stride_lo_r).to(tl.int32)
        hi = tl.load(hi_ptr + pid_m * stride_hi_m + r * stride_hi_r).to(tl.int32)
        row_has = hi > lo
        denom = tl.load(denom_ptr + pid_m * stride_d_m + r * stride_d_r).to(tl.float32)

        in_bounds = mask_k & (offs_k >= lo) & (offs_k < hi)

        ptr_lfp = log_f_pre_ptr + pid_m * stride_lfp_m + r * stride_lfp_r + offs_k * stride_lfp_k
        lfp = tl.load(ptr_lfp, mask=mask_k, other=min_val).to(tl.float32)
        lp = lfp - denom

        valid = row_has & in_bounds
        # 避免 -inf/-inf 导致 NaN：仅在 lp>min_val 时参与归约
        valid_lp = valid & (lp > min_val)
        lp = tl.where(valid_lp, lp, min_val)

        if USE_MEAN:
            log_sum += tl.where(valid_lp, lp, 0.0)
        else:
            log_f_pre = alpha * lp
            new_max = tl.where(valid_lp, tl.maximum(log_f_max, log_f_pre), log_f_max)
            update = log_f_sum * tl.exp(log_f_max - new_max) + tl.exp(log_f_pre - new_max)
            log_f_sum = tl.where(valid_lp, update, log_f_sum)
            log_f_max = new_max

        if STORE_MEAN:
            prob = tl.exp(lp)
            prob = tl.where(valid_lp, prob, 0.0)
            prob_sum += prob

    if USE_MEAN:
        log_f = log_sum / row_count_safe
    else:
        lse = log_f_max + tl.log(log_f_sum + eps)
        log_f = (lse - tl.log(row_count_safe)) / alpha

    log_f = tl.where(has_rows, log_f, min_val)

    tl.store(out_log_f_ptr + pid_m * stride_out_m + offs_k * stride_out_k, log_f, mask=mask_k)
    if STORE_MEAN:
        mean_probs = prob_sum / row_count_safe
        mean_probs = tl.where(has_rows, mean_probs, 0.0)
        tl.store(out_mean_ptr + pid_m * stride_mean_m + offs_k * stride_mean_k, mean_probs, mask=mask_k)


def log_f_mean_probs_from_log_f_pre_denom_bounds_triton(
    log_f_pre: torch.Tensor,
    denom: torch.Tensor,
    lo: torch.Tensor,
    hi: torch.Tensor,
    alpha: float,
    eps: float,
    *,
    return_mean_probs: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if log_f_pre.numel() == 0:
        empty = log_f_pre.new_empty((0, 0), dtype=torch.float32)
        counts = log_f_pre.new_empty((0,), dtype=torch.float32)
        return empty, empty, counts
    if (not log_f_pre.is_cuda) or (not denom.is_cuda):
        raise RuntimeError("log_f_mean_probs_from_log_f_pre_denom_bounds_triton requires CUDA tensors")
    if log_f_pre.dim() != 3:
        raise ValueError("log_f_pre must be [M, R, K]")
    if denom.dim() != 2:
        raise ValueError("denom must be [M, R]")
    if lo.shape != log_f_pre.shape[:2] or hi.shape != log_f_pre.shape[:2]:
        raise ValueError("lo/hi must be [M, R] and match log_f_pre leading dims")
    if denom.shape != log_f_pre.shape[:2]:
        raise ValueError("denom shape mismatch with log_f_pre rows")

    m, r, k = log_f_pre.shape
    if r > 128:
        raise RuntimeError("log_f_mean_probs_from_log_f_pre_denom_bounds_triton only supports R<=128")

    if k <= 128:
        block_k = 128
    else:
        block_k = 256

    lo_i32 = lo.to(device=log_f_pre.device, dtype=torch.int32)
    hi_i32 = hi.to(device=log_f_pre.device, dtype=torch.int32)
    denom_f = denom.to(dtype=torch.float32)

    row_valid = (hi_i32 > lo_i32).to(dtype=torch.float32)
    row_counts = row_valid.sum(dim=1).contiguous()

    log_f = torch.empty((m, k), device=log_f_pre.device, dtype=torch.float32)
    mean_probs: Optional[torch.Tensor]
    if return_mean_probs:
        mean_probs = torch.empty_like(log_f)
        mean_ptr = mean_probs
        stride_mean_m = mean_probs.stride(0)
        stride_mean_k = mean_probs.stride(1)
    else:
        mean_probs = None
        mean_ptr = log_f
        stride_mean_m = log_f.stride(0)
        stride_mean_k = log_f.stride(1)

    grid_log = (m, triton.cdiv(k, block_k))
    use_mean = abs(float(alpha)) < 1.0e-6
    _log_f_mean_probs_from_log_f_pre_denom_bounds_kernel[grid_log](
        log_f_pre,
        denom_f,
        lo_i32,
        hi_i32,
        row_counts,
        log_f,
        mean_ptr,
        int(log_f_pre.stride(0)),
        int(log_f_pre.stride(1)),
        int(log_f_pre.stride(2)),
        int(denom_f.stride(0)),
        int(denom_f.stride(1)),
        int(lo_i32.stride(0)),
        int(lo_i32.stride(1)),
        int(hi_i32.stride(0)),
        int(hi_i32.stride(1)),
        int(row_counts.stride(0)),
        int(log_f.stride(0)),
        int(log_f.stride(1)),
        int(stride_mean_m),
        int(stride_mean_k),
        K=k,
        alpha=float(alpha),
        eps=float(eps),
        R=r,
        USE_MEAN=use_mean,
        STORE_MEAN=return_mean_probs,
        BLOCK_K=block_k,
    )

    if mean_probs is None:
        mean_probs_out = log_f.new_empty((0, 0))
    else:
        mean_probs_out = mean_probs
    return log_f, mean_probs_out, row_counts



_GAUSSIAN_KERNEL_CACHE: dict[Tuple[int, torch.device, torch.dtype], torch.Tensor] = {}


def _build_gaussian_kernel_tensor(
    window: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    length = 2 * window + 1
    key = (length, device, dtype)
    cached = _GAUSSIAN_KERNEL_CACHE.get(key)
    if cached is not None:
        return cached

    radius = (length - 1) // 2
    positions = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    sigma = max(window / 2.0, 1.0)
    kernel = torch.exp(-0.5 * (positions / max(sigma, 1.0e-6)) ** 2)
    kernel[radius] = 0.0
    total = kernel.sum().clamp_min(1.0)
    kernel = kernel / total
    kernel = kernel.contiguous()
    _GAUSSIAN_KERNEL_CACHE[key] = kernel
    return kernel
