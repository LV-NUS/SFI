"""CUDA extension for fused log_f + prior (pre-soft-nms) selector stage."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.cpp_extension import load_inline
from utils.selector_log_s_identity import (
    SELECTOR_LOG_S_EXTENSION_NAME,
    SELECTOR_LOG_S_SEMANTIC_IDENTITY,
    SELECTOR_LOG_S_SEMANTIC_IDENTITY_SYMBOL,
)
from utils.torch_extension_cache import load_prebuilt_extension

_MODULE: Optional[torch.nn.Module] = None
_VALIDATED_MODULE: Optional[object] = None
_LOAD_ERROR: Optional[Exception] = None
_REQUIRED_EXT_SYMBOLS = (
    SELECTOR_LOG_S_SEMANTIC_IDENTITY_SYMBOL,
    "fused_log_f_prior_logits",
    "fused_log_f_prior_pre_denom",
    "reduce_log_f_pre_scratch",
    "reduce_log_f_pre_scratch_r2_alpha0p5_fp16_resident",
    "reduce_log_f_pre_scratch_r2_alpha0p5_fp16_tiled",
    "reduce_log_f_pre_scratch_r2_alpha0p5_fp16_tiled_workspace_nbytes",
    "copy_log_f_lastn1_scratch",
    "copy_log_f_lastn1_scratch_scalar",
    "reduce_log_f_pre_scratch_scalar",
    # 标记符号：缺它的 prebuilt .so 不支持跨片 accumulate meta（flags bit4 +
    # cols 8/9），必须拒载触发 load_inline 重编（同 fixed-shape topk 先例）。
    "reduce_log_f_pre_scratch_accum_supported",
)
_NVCC_RELEASE_RE = re.compile(r"release\s+(\d+)\.(\d+)")

LOG_F_R2_TILED_TILE_K = 2_048
LOG_F_R2_TILED_ROWS = 2


def log_f_r2_tiled_workspace_layout(
    *,
    num_seqs_capacity: int,
    num_query_heads_capacity: int,
    logical_k_capacity: int,
) -> dict[str, tuple[int, int]]:
    """Return the raw-byte workspace ABI for the dynamic large-K owner."""

    n = int(num_seqs_capacity)
    h = int(num_query_heads_capacity)
    k = int(logical_k_capacity)
    if n <= 0 or h <= 0 or k <= 0:
        raise ValueError("tiled workspace capacities must be positive")
    tiles = (k + LOG_F_R2_TILED_TILE_K - 1) // LOG_F_R2_TILED_TILE_K
    row_partial = n * h * LOG_F_R2_TILED_ROWS * tiles
    row_state = n * h * LOG_F_R2_TILED_ROWS
    token_partial = n * h * tiles
    element_counts = (
        ("row_partial_max", row_partial),
        ("row_partial_sum", row_partial),
        ("row_lse", row_state),
        ("row_has", row_state),
        ("token_partial_max", token_partial),
        ("token_partial_sum", token_partial),
        ("token_has", token_partial),
    )
    layout: dict[str, tuple[int, int]] = {}
    cursor = 0
    for name, count in element_counts:
        begin = cursor
        cursor += int(count) * 4
        layout[name] = (begin, cursor)
    return layout


def log_f_r2_tiled_workspace_nbytes(
    *,
    num_seqs_capacity: int,
    num_query_heads_capacity: int,
    logical_k_capacity: int,
) -> int:
    """Return the caller-owned workspace size without loading the extension."""

    layout = log_f_r2_tiled_workspace_layout(
        num_seqs_capacity=num_seqs_capacity,
        num_query_heads_capacity=num_query_heads_capacity,
        logical_k_capacity=logical_k_capacity,
    )
    return next(reversed(layout.values()))[1]


def allocate_log_f_r2_tiled_workspace(
    device: torch.device | str,
    *,
    num_seqs_capacity: int,
    num_query_heads_capacity: int,
    logical_k_capacity: int,
) -> torch.Tensor:
    """Allocate raw workspace; the caller owns its stream and graph lifetime."""

    return torch.empty(
        log_f_r2_tiled_workspace_nbytes(
            num_seqs_capacity=num_seqs_capacity,
            num_query_heads_capacity=num_query_heads_capacity,
            logical_k_capacity=logical_k_capacity,
        ),
        dtype=torch.uint8,
        device=device,
    )


def log_f_r2_tiled_contract_reasons(
    *,
    req_meta_i32: torch.Tensor,
    req_meta_i64: torch.Tensor,
    workspace: torch.Tensor,
    num_seqs: int,
    num_query_heads: int,
    num_seqs_capacity: int,
    num_query_heads_capacity: int,
    logical_k_capacity: int,
) -> tuple[str, ...]:
    """Validate the host-visible ABI without reading device-authored values."""

    n = int(num_seqs)
    h = int(num_query_heads)
    n_cap = int(num_seqs_capacity)
    h_cap = int(num_query_heads_capacity)
    k_cap = int(logical_k_capacity)
    reasons: list[str] = []
    if n <= 0 or h <= 0:
        reasons.append("runtime_shape")
    if n_cap <= 0 or h_cap <= 0 or k_cap <= 0:
        reasons.append("capacity")
    if n > n_cap or h > h_cap:
        reasons.append("runtime_exceeds_capacity")
    if req_meta_i32.dtype != torch.int32:
        reasons.append("meta_i32_dtype")
    if req_meta_i64.dtype != torch.int64:
        reasons.append("meta_i64_dtype")
    if workspace.dtype != torch.uint8:
        reasons.append("workspace_dtype")
    if (
        req_meta_i32.dim() != 2
        or int(req_meta_i32.shape[0]) < max(n, 0)
        or int(req_meta_i32.shape[1]) < 10
        or not req_meta_i32.is_contiguous()
    ):
        reasons.append("meta_i32_shape")
    if (
        req_meta_i64.dim() != 2
        or int(req_meta_i64.shape[0]) < max(n, 0)
        or int(req_meta_i64.shape[1]) < 4
        or not req_meta_i64.is_contiguous()
    ):
        reasons.append("meta_i64_shape")
    if workspace.dim() != 1 or not workspace.is_contiguous():
        reasons.append("workspace_shape")
    elif n_cap > 0 and h_cap > 0 and k_cap > 0:
        required = log_f_r2_tiled_workspace_nbytes(
            num_seqs_capacity=n_cap,
            num_query_heads_capacity=h_cap,
            logical_k_capacity=k_cap,
        )
        if int(workspace.numel()) < required:
            reasons.append("workspace_too_small")
    devices = {
        tensor.device
        for tensor in (req_meta_i32, req_meta_i64, workspace)
    }
    if len(devices) != 1:
        reasons.append("device_mismatch")
    if any(device.type != "cuda" for device in devices):
        reasons.append("device_not_cuda")
    return tuple(reasons)


class SelectorLogSExtUnavailable(RuntimeError):
    """Raised when the optional selector CUDA extension is unavailable."""


def _module_contract_error(module: object) -> str | None:
    missing = tuple(
        name for name in _REQUIRED_EXT_SYMBOLS if not hasattr(module, name)
    )
    if missing:
        return "missing symbols: " + ",".join(missing)
    identity_fn = getattr(module, SELECTOR_LOG_S_SEMANTIC_IDENTITY_SYMBOL, None)
    if not callable(identity_fn):
        return "semantic identity symbol is not callable"
    try:
        identity = identity_fn()
    except Exception as exc:
        return f"semantic identity query failed: {type(exc).__name__}"
    if type(identity) is not str or identity != SELECTOR_LOG_S_SEMANTIC_IDENTITY:
        return (
            "semantic identity mismatch: "
            f"actual={identity!r}:expected={SELECTOR_LOG_S_SEMANTIC_IDENTITY!r}"
        )
    return None


def _immutable_namespace_error(*, origin: str, contract_error: str) -> RuntimeError:
    return RuntimeError(
        f"{origin} {SELECTOR_LOG_S_EXTENSION_NAME} violates the selector log_s "
        f"semantic contract ({contract_error}); imported native extension "
        "namespaces are immutable in-process, so changed source/ABI requires "
        "a fresh extension namespace"
    )


def _prebuilt_extension_artifact() -> Path | None:
    """Return the current namespace artifact, if one already exists."""

    try:
        from torch.utils.cpp_extension import _get_build_directory

        build_dir = Path(
            _get_build_directory(SELECTOR_LOG_S_EXTENSION_NAME, verbose=False)
        )
        artifact = build_dir / f"{SELECTOR_LOG_S_EXTENSION_NAME}.so"
        if artifact.is_file() and int(artifact.stat().st_size) > 0:
            return artifact
    except OSError:
        return None
    return None


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
    global _MODULE, _VALIDATED_MODULE, _LOAD_ERROR
    if _LOAD_ERROR is not None:
        return None
    if _MODULE is not None:
        loaded_module = _MODULE
        if loaded_module is _VALIDATED_MODULE:
            return _MODULE
        contract_error = _module_contract_error(loaded_module)
        if contract_error is None:
            _VALIDATED_MODULE = loaded_module
            return _MODULE
        _MODULE = None
        _VALIDATED_MODULE = None
        _LOAD_ERROR = _immutable_namespace_error(
            origin="already-loaded module",
            contract_error=contract_error,
        )
        return None
    if not force and not _should_enable():
        return None

    registered = sys.modules.get(SELECTOR_LOG_S_EXTENSION_NAME)
    if registered is not None:
        contract_error = _module_contract_error(registered)
        if contract_error is not None:
            _LOAD_ERROR = _immutable_namespace_error(
                origin="already-imported module",
                contract_error=contract_error,
            )
            return None
        _MODULE = registered
        _VALIDATED_MODULE = registered
        return _MODULE

    # A native extension name is a process-lifetime ABI namespace.  A valid
    # v12 prebuilt is reusable; an invalid/unimportable v12 artifact is
    # terminal.  Compilation is allowed only when this fresh namespace has no
    # artifact yet.
    prebuilt_artifact = _prebuilt_extension_artifact()
    prebuilt = load_prebuilt_extension(SELECTOR_LOG_S_EXTENSION_NAME)
    if prebuilt is not None:
        contract_error = _module_contract_error(prebuilt)
        if contract_error is not None:
            _LOAD_ERROR = _immutable_namespace_error(
                origin="prebuilt module",
                contract_error=contract_error,
            )
            return None
        _MODULE = prebuilt
        _VALIDATED_MODULE = prebuilt
        return _MODULE
    if prebuilt_artifact is not None:
        _LOAD_ERROR = RuntimeError(
            f"prebuilt {SELECTOR_LOG_S_EXTENSION_NAME} artifact could not be "
            f"imported: {prebuilt_artifact}; refusing to rebuild an existing "
            "native extension namespace in-process"
        )
        return None

    cpp_source = r"""
#include <torch/extension.h>
#include <cstdlib>
#include <string>
#include <vector>

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

void reduce_log_f_pre_scratch_r2_alpha0p5_fp16_resident_cuda(
    torch::Tensor req_meta_i32,
    torch::Tensor req_meta_i64,
    int64_t num_seqs,
    int64_t num_query_heads,
    int64_t logical_k_bucket);

void reduce_log_f_pre_scratch_r2_alpha0p5_fp16_tiled_cuda(
    torch::Tensor req_meta_i32,
    torch::Tensor req_meta_i64,
    torch::Tensor workspace,
    int64_t num_seqs,
    int64_t num_query_heads,
    int64_t num_seqs_capacity,
    int64_t num_query_heads_capacity,
    int64_t logical_k_capacity);

int64_t reduce_log_f_pre_scratch_r2_alpha0p5_fp16_tiled_workspace_nbytes_cuda(
    int64_t num_seqs_capacity,
    int64_t num_query_heads_capacity,
    int64_t logical_k_capacity);

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

void reduce_log_f_pre_scratch_r2_alpha0p5_fp16_resident(
    torch::Tensor req_meta_i32,
    torch::Tensor req_meta_i64,
    int64_t num_seqs,
    int64_t num_query_heads,
    int64_t logical_k_bucket) {
    reduce_log_f_pre_scratch_r2_alpha0p5_fp16_resident_cuda(
        req_meta_i32,
        req_meta_i64,
        num_seqs,
        num_query_heads,
        logical_k_bucket);
}

void reduce_log_f_pre_scratch_r2_alpha0p5_fp16_tiled(
    torch::Tensor req_meta_i32,
    torch::Tensor req_meta_i64,
    torch::Tensor workspace,
    int64_t num_seqs,
    int64_t num_query_heads,
    int64_t num_seqs_capacity,
    int64_t num_query_heads_capacity,
    int64_t logical_k_capacity) {
    reduce_log_f_pre_scratch_r2_alpha0p5_fp16_tiled_cuda(
        req_meta_i32,
        req_meta_i64,
        workspace,
        num_seqs,
        num_query_heads,
        num_seqs_capacity,
        num_query_heads_capacity,
        logical_k_capacity);
}

int64_t reduce_log_f_pre_scratch_r2_alpha0p5_fp16_tiled_workspace_nbytes(
    int64_t num_seqs_capacity,
    int64_t num_query_heads_capacity,
    int64_t logical_k_capacity) {
    return reduce_log_f_pre_scratch_r2_alpha0p5_fp16_tiled_workspace_nbytes_cuda(
        num_seqs_capacity,
        num_query_heads_capacity,
        logical_k_capacity);
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
    m.def("selector_log_s_semantic_identity", []() {
        return std::string("__SFI_SELECTOR_LOG_S_SEMANTIC_IDENTITY__");
    }, "Fixed selector log_s kernel/ABI semantic identity");
    m.def("fused_log_f_prior_logits", &fused_log_f_prior_logits,
          "Fused log_f + prior (logits path)");
    m.def("fused_log_f_prior_pre_denom", &fused_log_f_prior_pre_denom,
          "Fused log_f + prior (log_f_pre + denom path)");
    m.def("reduce_log_f_pre_scratch", &reduce_log_f_pre_scratch,
          "Reduce FA scratch logits into log_f_pre + denom");
    m.def("reduce_log_f_pre_scratch_r2_alpha0p5_fp16_resident",
          &reduce_log_f_pre_scratch_r2_alpha0p5_fp16_resident,
          "Resident two-pass SM80 reduce for R=2 K buckets, fp16, alpha=0.5");
    m.def("reduce_log_f_pre_scratch_r2_alpha0p5_fp16_tiled",
          &reduce_log_f_pre_scratch_r2_alpha0p5_fp16_tiled,
          "Dynamic tiled four-stage reduce for R=2, fp16, alpha=0.5");
    m.def("reduce_log_f_pre_scratch_r2_alpha0p5_fp16_tiled_workspace_nbytes",
          &reduce_log_f_pre_scratch_r2_alpha0p5_fp16_tiled_workspace_nbytes,
          "Caller-owned byte workspace required by the dynamic tiled reduce");
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

    semantic_token = "__SFI_SELECTOR_LOG_S_SEMANTIC_IDENTITY__"
    if cpp_source.count(semantic_token) != 1:
        raise RuntimeError("selector log_s semantic identity token drift")
    cpp_source = cpp_source.replace(
        semantic_token,
        SELECTOR_LOG_S_SEMANTIC_IDENTITY,
    )

    cuda_source = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
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

// Reproduce the generic reduce kernel's 256-thread reduction order inside a
// larger resident CTA.  All CTA threads call these helpers; only logical lanes
// [0, 256) contribute, so NaN propagation matches the generic owner exactly.
__device__ __forceinline__ float block_reduce_sum_first_256(float val) {
    static __shared__ float shared[8];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wid = tid >> 5;
    if (tid < 256) {
        val = warp_reduce_sum(val);
        if (lane == 0) {
            shared[wid] = val;
        }
    }
    __syncthreads();
    float out = 0.0f;
    if (wid == 0) {
        out = lane < 8 ? shared[lane] : 0.0f;
        out = warp_reduce_sum(out);
    }
    __syncthreads();
    if (tid == 0) {
        shared[0] = out;
    }
    __syncthreads();
    return shared[0];
}

__device__ __forceinline__ float block_reduce_max_first_256(float val) {
    static __shared__ float shared[8];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wid = tid >> 5;
    if (tid < 256) {
        val = warp_reduce_max(val);
        if (lane == 0) {
            shared[wid] = val;
        }
    }
    __syncthreads();
    float out = -INFINITY;
    if (wid == 0) {
        out = lane < 8 ? shared[lane] : -INFINITY;
        out = warp_reduce_max(out);
    }
    __syncthreads();
    if (tid == 0) {
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


// [FLFP-ILP4 2026-07-10] Pass2/3 的 log_r_raw 单点取值 helper:cache 命中读
// cache,miss 走与原两处 inline 重算逐字同序的表达式(编译期内联,数值逐位
// 等价)。仅供保序 ILP 展开消重复文本,不改任何数学。
__device__ __forceinline__ float flfp_log_r_raw_at(
    const float* __restrict__ log_r_cache,
    const float* __restrict__ key_norms,
    int m,
    int k,
    int K,
    int stride_kn_m,
    int stride_kn_k,
    int token_lo,
    float denom_pos,
    float eps,
    float gamma,
    float prior_pos_power_f,
    float beta,
    float prior_pos_eta,
    float prior_weight_l2,
    float prior_weight_pos) {
    if (log_r_cache != nullptr) {
        return log_r_cache[m * K + k];
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
    return prior_weight_l2 * log_pi + prior_weight_pos * log_delta;
}

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
    // [FLFP-ILP4 2026-07-10] 保序四路展开:每迭代并发发射 4 个 k 的 load 链
    // (占用 13%/1 block/SM 档的延迟遮蔽靠线程内 ILP),累加仍是单累加器按
    // 原 k 升序逐位相同的加法序;OOB 贡献显式 0.0f(+0.0f 恒等,累加器非
    // ±0 场景),OOB 支路的 NaN(denom_r=-inf 时)经 select 丢弃不入账。
    // 规约树/blockDim/shm 布局一字未动(λ 数值链锁死面,T1 定谳)。
    float local_ff = 0.0f;
    float local_rr = 0.0f;
    float local_fr = 0.0f;
    float local_prob_sum = 0.0f;
    float local_pos_sum = 0.0f;

    const int flfp_stride = blockDim.x;
    int flfp_k_base = tid;
    for (; flfp_k_base + 3 * flfp_stride < K; flfp_k_base += 4 * flfp_stride) {
        float c_ff[4];
        float c_rr[4];
        float c_fr[4];
        float c_p[4];
        float c_ppos[4];
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            int k = flfp_k_base + u * flfp_stride;
            bool flfp_valid = (k >= token_lo) && (k < token_hi);
            float log_f = out[m * K + k];
            float lp = min_val;
            if (log_f > min_val && denom_f > min_val) {
                lp = log_f - denom_f;
            }
            float pos_norm = (static_cast<float>(k - token_lo)) / denom_pos;
            pos_norm = fmaxf(0.0f, fminf(1.0f, pos_norm));
            float log_r_raw = flfp_log_r_raw_at(
                log_r_cache, key_norms, m, k, K, stride_kn_m, stride_kn_k,
                token_lo, denom_pos, eps, gamma, prior_pos_power_f, beta,
                prior_pos_eta, prior_weight_l2, prior_weight_pos);
            float log_r = log_r_raw - denom_r;
            c_ff[u] = flfp_valid ? expf(2.0f * lp) : 0.0f;
            c_rr[u] = flfp_valid ? expf(2.0f * log_r) : 0.0f;
            c_fr[u] = flfp_valid ? expf(lp + log_r) : 0.0f;
            if (lambda_tail_kappa > 0.0f) {
                float p = flfp_valid ? expf(lp) : 0.0f;
                c_p[u] = p;
                c_ppos[u] = p * pos_norm;
            } else {
                c_p[u] = 0.0f;
                c_ppos[u] = 0.0f;
            }
        }
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            local_ff += c_ff[u];
            local_rr += c_rr[u];
            local_fr += c_fr[u];
            if (lambda_tail_kappa > 0.0f) {
                local_prob_sum += c_p[u];
                local_pos_sum += c_ppos[u];
            }
        }
    }
    for (int k = flfp_k_base; k < K; k += flfp_stride) {
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
    // [FLFP-ILP4] max 为序无关(fmaxf 交换结合,输入无 NaN 入链:OOB 经
    // select 换 min_val);存储按 valid 谓词化(OOB 保持 Pass1 写下的
    // min_val,与原 continue 语义逐位一致)。
    float local_max_fused = min_val;
    flfp_k_base = tid;
    for (; flfp_k_base + 3 * flfp_stride < K; flfp_k_base += 4 * flfp_stride) {
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            int k = flfp_k_base + u * flfp_stride;
            bool flfp_valid = (k >= token_lo) && (k < token_hi);
            float log_f = out[m * K + k];
            float lp = min_val;
            if (log_f > min_val && denom_f > min_val) {
                lp = log_f - denom_f;
            }
            float log_r_raw = flfp_log_r_raw_at(
                log_r_cache, key_norms, m, k, K, stride_kn_m, stride_kn_k,
                token_lo, denom_pos, eps, gamma, prior_pos_power_f, beta,
                prior_pos_eta, prior_weight_l2, prior_weight_pos);
            float log_r = log_r_raw - denom_r;
            float a = log_one_minus + lp;
            float b = log_lambda + log_r;
            float mval = fmaxf(a, b);
            float fused_raw = mval + logf(expf(a - mval) + expf(b - mval));
            if (flfp_valid) {
                out[m * K + k] = fused_raw;
            }
            local_max_fused = fmaxf(
                local_max_fused, flfp_valid ? fused_raw : min_val);
        }
    }
    for (int k = flfp_k_base; k < K; k += flfp_stride) {
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
    // [FLFP-ILP4] sum_fused:同 Pass2 保序单累加器展开;OOB 读值=Pass1 的
    // min_val(Pass3 谓词化未触碰),贡献显式 0。
    float local_sum_fused = 0.0f;
    flfp_k_base = tid;
    for (; flfp_k_base + 3 * flfp_stride < K; flfp_k_base += 4 * flfp_stride) {
        float c_sf[4];
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            int k = flfp_k_base + u * flfp_stride;
            bool flfp_valid = (k >= token_lo) && (k < token_hi);
            float fused_raw = out[m * K + k];
            c_sf[u] = (flfp_valid && fused_raw > min_val && max_fused > min_val)
                ? expf(fused_raw - max_fused)
                : 0.0f;
        }
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            local_sum_fused += c_sf[u];
        }
    }
    for (int k = flfp_k_base; k < K; k += flfp_stride) {
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
    // [FLFP-ILP4] 双臂皆写,纯逐 k 独立,自由展开。
    flfp_k_base = tid;
    for (; flfp_k_base + 3 * flfp_stride < K; flfp_k_base += 4 * flfp_stride) {
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            int k = flfp_k_base + u * flfp_stride;
            bool flfp_valid = (k >= token_lo) && (k < token_hi);
            float fused_raw = out[m * K + k];
            out[m * K + k] = flfp_valid ? (fused_raw - denom_fused) : min_val;
        }
    }
    for (int k = flfp_k_base; k < K; k += flfp_stride) {
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
        // [THREADS-WARP-ALIGN] non-multiple-of-32 blockDim would make the
        // full-mask __shfl_down_sync reductions UB in the last warp.
        // (2026-07-11 EXT审计·姊妹同步: selector_pipeline_ext.py 同名段已带
        // 此对齐，本独立版漏同步——逐字补齐。)
        parsed &= ~31;
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
constexpr int kLogFPreR2Bucket12K = 12288;
constexpr int kLogFPreR2Bucket16K = 16384;
constexpr int kLogFPreR2Bucket24K = 24576;
constexpr int kLogFPreR2Bucket32K = 32768;
constexpr int kLogFPreR2TiledRows = 2;
constexpr int kLogFPreR2TiledTileK = 2048;
constexpr int kLogFPreR2TiledThreads = 256;
constexpr int kLogFPreR2TiledItemsPerThread =
    kLogFPreR2TiledTileK / kLogFPreR2TiledThreads;

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

// Shared resident owner for the real R=2 buckets.  K, N, H and physical
// strides remain metadata-driven; only the loop ceiling is specialized.  The
// ceiling is never an effective length: bucket padding is neither loaded nor
// written.  This keeps one contract for TP1/TP8 and batch>1 instead of cloning
// the former K=11744 single-row path.
template <int kLogicalKBucket>
__device__ __forceinline__ bool log_f_pre_r2_resident_contract(
    const int32_t* meta32,
    const int64_t* meta64,
    int64_t meta32_stride_col,
    int64_t meta64_stride_col) {
    const int logical_k = meta32[0 * meta32_stride_col];
    const int scratch_head_stride = meta32[1 * meta32_stride_col];
    const int out_head_stride = meta32[6 * meta32_stride_col];
    const int scratch_row_stride = meta32[7 * meta32_stride_col];
    const uint64_t scratch_base = static_cast<uint64_t>(
        meta64[1 * meta64_stride_col]);
    const uint64_t out_base = static_cast<uint64_t>(
        meta64[2 * meta64_stride_col]);
    const uint64_t denom_base = static_cast<uint64_t>(
        meta64[3 * meta64_stride_col]);
    // Two fp32 row LSE values occupy out_head[0:4] between the two kernels.
    // K<4 cannot provide that transient workspace without crossing the head.
    return logical_k >= 4
        && logical_k <= kLogicalKBucket
        && meta32[2 * meta32_stride_col] == 2
        && meta32[3 * meta32_stride_col] == 0
        && meta32[4 * meta32_stride_col] == logical_k
        && meta32[5 * meta32_stride_col] == 8
        && scratch_row_stride >= logical_k
        && scratch_head_stride == 2 * scratch_row_stride
        && out_head_stride >= logical_k
        && meta32[8 * meta32_stride_col] == 0
        && meta32[9 * meta32_stride_col] == 0
        && scratch_base != 0
        && out_base != 0
        && denom_base != 0;
}

__device__ __forceinline__ void log_f_pre_contract_trap() {
    asm volatile("trap;");
}

// out_head is a half tensor, so an arbitrary legal physical head stride only
// guarantees two-byte alignment.  Preserve each transient float LSE bit-for-
// bit without imposing a hidden four-byte stride contract: split it into two
// naturally aligned u16 stores, then reconstruct the same bits in pass two.
__device__ __forceinline__ void store_float_bits_half_aligned(
    half* base,
    int index,
    float value) {
    const uint32_t bits = __float_as_uint(value);
    uint16_t* words = reinterpret_cast<uint16_t*>(base) + 2 * index;
    words[0] = static_cast<uint16_t>(bits & 0xffffU);
    words[1] = static_cast<uint16_t>(bits >> 16);
}

__device__ __forceinline__ float load_float_bits_half_aligned(
    const half* base,
    int index) {
    const uint16_t* words = reinterpret_cast<const uint16_t*>(base)
        + 2 * index;
    const uint32_t bits = static_cast<uint32_t>(words[0])
        | (static_cast<uint32_t>(words[1]) << 16);
    return __uint_as_float(bits);
}

template <int kLogicalKBucket, int kThreads, int kItemsPerThread>
__global__ __launch_bounds__(kThreads, 1)
void reduce_log_f_pre_r2_resident_row_lse_kernel(
    const int32_t* __restrict__ req_meta_i32,
    const int64_t* __restrict__ req_meta_i64,
    int64_t req_meta_i32_stride_row,
    int64_t req_meta_i32_stride_col,
    int64_t req_meta_i64_stride_row,
    int64_t req_meta_i64_stride_col) {
    static_assert(kThreads * kItemsPerThread >= kLogicalKBucket,
                  "resident bucket must be fully covered");
    const int seq = blockIdx.z;
    const int head = blockIdx.x;
    const int row = blockIdx.y;
    const int32_t* meta32 = req_meta_i32
        + static_cast<int64_t>(seq) * req_meta_i32_stride_row;
    const int64_t* meta64 = req_meta_i64
        + static_cast<int64_t>(seq) * req_meta_i64_stride_row;
    __shared__ int contract_ok;
    __shared__ int logical_k;
    __shared__ int scratch_head_stride;
    __shared__ int scratch_row_stride;
    __shared__ int64_t scratch_base_raw;
    __shared__ int64_t out_base_raw;
    if (threadIdx.x == 0) {
        contract_ok = log_f_pre_r2_resident_contract<kLogicalKBucket>(
            meta32,
            meta64,
            req_meta_i32_stride_col,
            req_meta_i64_stride_col) ? 1 : 0;
        logical_k = meta32[0 * req_meta_i32_stride_col];
        scratch_head_stride = meta32[1 * req_meta_i32_stride_col];
        scratch_row_stride = meta32[7 * req_meta_i32_stride_col];
        scratch_base_raw = meta64[1 * req_meta_i64_stride_col];
        out_base_raw = meta64[2 * req_meta_i64_stride_col];
    }
    __syncthreads();
    if (contract_ok == 0) {
        if (threadIdx.x == 0) {
            log_f_pre_contract_trap();
        }
        return;
    }

    const half* scratch = reinterpret_cast<const half*>(scratch_base_raw)
        + static_cast<int64_t>(head) * scratch_head_stride
        + static_cast<int64_t>(row) * scratch_row_stride;
    float values[kItemsPerThread];
    float local_max = kLogFPreMinVal;
#pragma unroll
    for (int item = 0; item < kItemsPerThread; ++item) {
        const int token = threadIdx.x + item * kThreads;
        const float value = token < logical_k
            ? __half2float(scratch[token])
            : kLogFPreMinVal;
        values[item] = value;
        if (value > kLogFPreMinVal) {
            local_max = local_max > value ? local_max : value;
        }
    }
    const float row_max = block_reduce_max(local_max);
    float local_sum = 0.0f;
    if (row_max > kLogFPreMinVal) {
#pragma unroll
        for (int item = 0; item < kItemsPerThread; ++item) {
            const float value = values[item];
            if (value > kLogFPreMinVal) {
                local_sum += expf(value - row_max);
            }
        }
    }
    const float row_sum = block_reduce_sum(local_sum);
    if (threadIdx.x == 0) {
        half* out_head = reinterpret_cast<half*>(out_base_raw)
            + static_cast<int64_t>(head)
                * meta32[6 * req_meta_i32_stride_col];
        // The first eight output bytes are a transient two-float workspace.
        // The output tensor is only half-aligned for odd physical strides, so
        // keep the float bits in two u16 words.  The second pass overwrites
        // every logical token on the same stream.
        store_float_bits_half_aligned(
            out_head,
            row,
            row_max > kLogFPreMinVal
                ? row_max + logf(row_sum + 1.0e-20f)
                : kLogFPreMinVal);
    }
}

__device__ __noinline__ void reduce_log_f_pre_r2_nonfinite_generic_256(
    const half* scratch_head,
    half* out_head,
    float* denom,
    int head,
    int logical_k,
    int scratch_row_stride,
    float* generic_row_lse,
    int* generic_row_has,
    float* generic_row_count,
    float* generic_token_max) {
    const bool generic_lane = threadIdx.x < 256;
    for (int row = 0; row < 2; ++row) {
        float local_max = kLogFPreMinVal;
        if (generic_lane) {
            for (int token = threadIdx.x; token < logical_k; token += 256) {
                const float value = __half2float(
                    scratch_head[static_cast<int64_t>(row)
                        * scratch_row_stride + token]);
                if (value > kLogFPreMinVal) {
                    local_max = local_max > value ? local_max : value;
                }
            }
        }
        const float row_max = block_reduce_max_first_256(local_max);
        if (threadIdx.x == 0) {
            generic_row_has[row] = row_max > kLogFPreMinVal ? 1 : 0;
            generic_row_lse[row] = row_max;
        }
        __syncthreads();

        float local_sum = 0.0f;
        if (generic_lane && generic_row_has[row] != 0) {
            for (int token = threadIdx.x; token < logical_k; token += 256) {
                const float value = __half2float(
                    scratch_head[static_cast<int64_t>(row)
                        * scratch_row_stride + token]);
                if (value > kLogFPreMinVal) {
                    local_sum += expf(value - generic_row_lse[row]);
                }
            }
        }
        const float row_sum = block_reduce_sum_first_256(local_sum);
        if (threadIdx.x == 0) {
            generic_row_lse[row] = generic_row_has[row] != 0
                ? generic_row_lse[row] + logf(row_sum + 1.0e-20f)
                : kLogFPreMinVal;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        const int row_count = generic_row_has[0] + generic_row_has[1];
        *generic_row_count = row_count > 0
            ? static_cast<float>(row_count)
            : 1.0f;
    }
    __syncthreads();

    float local_token_max = kLogFPreMinVal;
    bool local_token_poison = false;
    if (generic_lane) {
        for (int token = threadIdx.x; token < logical_k; token += 256) {
            bool has_token = false;
            const float log_f = compute_log_f_pre_token(
                scratch_head,
                static_cast<int64_t>(scratch_row_stride),
                token,
                2,
                generic_row_lse,
                generic_row_has,
                *generic_row_count,
                0.5f,
                false,
                &has_token);
            out_head[token] = __float2half_rn(log_f);
            if (has_token) {
                if (isnan(log_f)) {
                    local_token_poison = true;
                } else {
                    local_token_max = local_token_max > log_f
                        ? local_token_max
                        : log_f;
                }
            }
        }
    }
    const int token_poison = __syncthreads_count(local_token_poison) > 0
        ? 1
        : 0;
    if (token_poison != 0) {
        if (threadIdx.x == 0) {
            denom[head] = NAN;
        }
        return;
    }
    const float token_max = block_reduce_max_first_256(local_token_max);
    if (threadIdx.x == 0) {
        *generic_token_max = token_max;
    }
    __syncthreads();
    if (*generic_token_max <= kLogFPreMinVal) {
        if (threadIdx.x == 0) {
            denom[head] = 0.0f;
        }
        return;
    }

    float local_token_sum = 0.0f;
    if (generic_lane) {
        for (int token = threadIdx.x; token < logical_k; token += 256) {
            bool has_token = false;
            const float log_f = compute_log_f_pre_token(
                scratch_head,
                static_cast<int64_t>(scratch_row_stride),
                token,
                2,
                generic_row_lse,
                generic_row_has,
                *generic_row_count,
                0.5f,
                false,
                &has_token);
            if (has_token) {
                local_token_sum += expf(log_f - *generic_token_max);
            }
        }
    }
    const float token_sum = block_reduce_sum_first_256(local_token_sum);
    if (threadIdx.x == 0) {
        denom[head] = *generic_token_max + logf(token_sum + 1.0e-20f);
    }
}

template <int kLogicalKBucket, int kThreads, int kItemsPerThread>
__global__ __launch_bounds__(kThreads, 1)
void reduce_log_f_pre_r2_resident_log_f_denom_kernel(
    const int32_t* __restrict__ req_meta_i32,
    const int64_t* __restrict__ req_meta_i64,
    int64_t req_meta_i32_stride_row,
    int64_t req_meta_i32_stride_col,
    int64_t req_meta_i64_stride_row,
    int64_t req_meta_i64_stride_col) {
    static_assert(kThreads * kItemsPerThread >= kLogicalKBucket,
                  "resident bucket must be fully covered");
    const int seq = blockIdx.z;
    const int head = blockIdx.x;
    const int32_t* meta32 = req_meta_i32
        + static_cast<int64_t>(seq) * req_meta_i32_stride_row;
    const int64_t* meta64 = req_meta_i64
        + static_cast<int64_t>(seq) * req_meta_i64_stride_row;
    __shared__ int contract_ok;
    __shared__ int logical_k;
    __shared__ int scratch_head_stride;
    __shared__ int scratch_row_stride;
    __shared__ int out_head_stride;
    __shared__ int64_t scratch_base_raw;
    __shared__ int64_t out_base_raw;
    __shared__ int64_t denom_base_raw;
    __shared__ float row_lse0;
    __shared__ float row_lse1;
    __shared__ float log_row_count;
    __shared__ int row_lse_nonfinite;
    __shared__ float generic_row_lse[2];
    __shared__ int generic_row_has[2];
    __shared__ float generic_row_count;
    __shared__ float generic_token_max;
    if (threadIdx.x == 0) {
        contract_ok = log_f_pre_r2_resident_contract<kLogicalKBucket>(
            meta32,
            meta64,
            req_meta_i32_stride_col,
            req_meta_i64_stride_col) ? 1 : 0;
        logical_k = meta32[0 * req_meta_i32_stride_col];
        scratch_head_stride = meta32[1 * req_meta_i32_stride_col];
        scratch_row_stride = meta32[7 * req_meta_i32_stride_col];
        out_head_stride = meta32[6 * req_meta_i32_stride_col];
        scratch_base_raw = meta64[1 * req_meta_i64_stride_col];
        out_base_raw = meta64[2 * req_meta_i64_stride_col];
        denom_base_raw = meta64[3 * req_meta_i64_stride_col];
    }
    __syncthreads();
    if (contract_ok == 0) {
        if (threadIdx.x == 0) {
            log_f_pre_contract_trap();
        }
        return;
    }

    half* out_head = reinterpret_cast<half*>(out_base_raw)
        + static_cast<int64_t>(head) * out_head_stride;
    if (threadIdx.x == 0) {
        row_lse0 = load_float_bits_half_aligned(out_head, 0);
        row_lse1 = load_float_bits_half_aligned(out_head, 1);
        row_lse_nonfinite = (!isfinite(row_lse0) || !isfinite(row_lse1))
            ? 1
            : 0;
        const int row_count = (row_lse0 > kLogFPreMinVal ? 1 : 0)
            + (row_lse1 > kLogFPreMinVal ? 1 : 0);
        log_row_count = row_count > 0 ? logf(static_cast<float>(row_count)) : 0.0f;
    }
    __syncthreads();

    const half* scratch_head = reinterpret_cast<const half*>(scratch_base_raw)
        + static_cast<int64_t>(head) * scratch_head_stride;

    // A +inf scratch value can make a row LSE non-finite.  Generic semantics
    // are token-local (a masked token in that row need not become NaN), and its
    // denom follows the exact 256-thread reduction order.  Recompute only this
    // exceptional data path with the same 256 logical lanes and math; this is
    // not a fallback launch and adds no host decision or collective.
    if (row_lse_nonfinite != 0) {
        reduce_log_f_pre_r2_nonfinite_generic_256(
            scratch_head,
            out_head,
            reinterpret_cast<float*>(denom_base_raw),
            head,
            logical_k,
            scratch_row_stride,
            generic_row_lse,
            generic_row_has,
            &generic_row_count,
            &generic_token_max);
        return;
    }

    float log_f_values[kItemsPerThread];
    float local_token_max = kLogFPreMinVal;
#pragma unroll
    for (int item = 0; item < kItemsPerThread; ++item) {
        const int token = threadIdx.x + item * kThreads;
        float log_f = kLogFPreMinVal;
        if (token < logical_k) {
            const float value0 = __half2float(scratch_head[token]);
            const float value1 = __half2float(
                scratch_head[scratch_row_stride + token]);
            const bool valid0 = row_lse0 > kLogFPreMinVal
                && value0 > kLogFPreMinVal;
            const bool valid1 = row_lse1 > kLogFPreMinVal
                && value1 > kLogFPreMinVal;
            if (valid0 || valid1) {
                float score_max = kLogFPreMinVal;
                float score0 = kLogFPreMinVal;
                float score1 = kLogFPreMinVal;
                if (valid0) {
                    score0 = 0.5f * (value0 - row_lse0);
                    score_max = score0;
                }
                if (valid1) {
                    score1 = 0.5f * (value1 - row_lse1);
                    score_max = score_max > score1 ? score_max : score1;
                }
                float score_sum = 0.0f;
                if (valid0) {
                    score_sum += expf(score0 - score_max);
                }
                if (valid1) {
                    score_sum += expf(score1 - score_max);
                }
                log_f = 2.0f * (
                    score_max + logf(score_sum + 1.0e-20f) - log_row_count);
                local_token_max = local_token_max > log_f
                    ? local_token_max
                    : log_f;
            }
            out_head[token] = __float2half_rn(log_f);
        }
        log_f_values[item] = log_f;
    }

    const float token_max = block_reduce_max(local_token_max);
    if (token_max <= kLogFPreMinVal) {
        if (threadIdx.x == 0) {
            reinterpret_cast<float*>(denom_base_raw)[head] = 0.0f;
        }
        return;
    }
    float local_token_sum = 0.0f;
#pragma unroll
    for (int item = 0; item < kItemsPerThread; ++item) {
        const float log_f = log_f_values[item];
        if (log_f > kLogFPreMinVal) {
            local_token_sum += expf(log_f - token_max);
        }
    }
    const float token_sum = block_reduce_sum(local_token_sum);
    if (threadIdx.x == 0) {
        reinterpret_cast<float*>(denom_base_raw)[head] =
            token_max + logf(token_sum + 1.0e-20f);
    }
}

struct LogFPreR2TiledWorkspaceLayout {
    int64_t tile_capacity;
    int64_t row_partial_max;
    int64_t row_partial_sum;
    int64_t row_lse;
    int64_t row_has;
    int64_t token_partial_max;
    int64_t token_partial_sum;
    int64_t token_has;
    int64_t total_words;
};

inline LogFPreR2TiledWorkspaceLayout log_f_pre_r2_tiled_workspace_layout(
    int64_t num_seqs_capacity,
    int64_t num_query_heads_capacity,
    int64_t logical_k_capacity) {
    const int64_t tile_capacity =
        (logical_k_capacity + kLogFPreR2TiledTileK - 1)
        / kLogFPreR2TiledTileK;
    const int64_t row_partial_elements = num_seqs_capacity
        * num_query_heads_capacity * kLogFPreR2TiledRows * tile_capacity;
    const int64_t row_state_elements = num_seqs_capacity
        * num_query_heads_capacity * kLogFPreR2TiledRows;
    const int64_t token_partial_elements = num_seqs_capacity
        * num_query_heads_capacity * tile_capacity;
    LogFPreR2TiledWorkspaceLayout layout{};
    layout.tile_capacity = tile_capacity;
    layout.row_partial_max = 0;
    layout.row_partial_sum = layout.row_partial_max + row_partial_elements;
    layout.row_lse = layout.row_partial_sum + row_partial_elements;
    layout.row_has = layout.row_lse + row_state_elements;
    layout.token_partial_max = layout.row_has + row_state_elements;
    layout.token_partial_sum =
        layout.token_partial_max + token_partial_elements;
    layout.token_has = layout.token_partial_sum + token_partial_elements;
    layout.total_words = layout.token_has + token_partial_elements;
    return layout;
}

__device__ __forceinline__ bool log_f_pre_r2_tiled_contract(
    const int32_t* meta32,
    const int64_t* meta64,
    int64_t meta32_stride_col,
    int64_t meta64_stride_col,
    int logical_k_capacity) {
    const int logical_k = meta32[0 * meta32_stride_col];
    const int scratch_head_stride = meta32[1 * meta32_stride_col];
    const int last_n = meta32[2 * meta32_stride_col];
    const int flags = meta32[5 * meta32_stride_col];
    const int out_head_stride = meta32[6 * meta32_stride_col];
    const int scratch_row_stride = meta32[7 * meta32_stride_col];
    const int accum_prev_rows = meta32[8 * meta32_stride_col];
    const int accum_prev_capacity = meta32[9 * meta32_stride_col];
    const bool plain = flags == 8
        && last_n == 2
        && accum_prev_rows == 0
        && accum_prev_capacity == 0;
    const bool accumulating = flags == 24
        && (last_n == 1 || last_n == 2)
        && accum_prev_rows >= 0
        && (
            (accum_prev_rows == 0 && accum_prev_capacity == 0)
            || (accum_prev_rows > 0
                && accum_prev_capacity > 0
                && accum_prev_capacity <= logical_k));
    return logical_k > 0
        && logical_k <= logical_k_capacity
        && meta32[3 * meta32_stride_col] == 0
        && meta32[4 * meta32_stride_col] == logical_k
        && (plain || accumulating)
        && scratch_row_stride >= logical_k
        && scratch_head_stride >= last_n * scratch_row_stride
        && out_head_stride >= logical_k
        && meta64[1 * meta64_stride_col] != 0
        && meta64[2 * meta64_stride_col] != 0
        && meta64[3 * meta64_stride_col] != 0;
}

__device__ __forceinline__ int64_t log_f_pre_r2_tiled_row_partial_index(
    int seq,
    int head,
    int row,
    int tile,
    int num_query_heads_capacity,
    int tile_capacity) {
    return (((static_cast<int64_t>(seq) * num_query_heads_capacity + head)
        * kLogFPreR2TiledRows + row) * tile_capacity + tile);
}

__device__ __forceinline__ int64_t log_f_pre_r2_tiled_row_state_index(
    int seq,
    int head,
    int row,
    int num_query_heads_capacity) {
    return ((static_cast<int64_t>(seq) * num_query_heads_capacity + head)
        * kLogFPreR2TiledRows + row);
}

__device__ __forceinline__ int64_t log_f_pre_r2_tiled_token_partial_index(
    int seq,
    int head,
    int tile,
    int num_query_heads_capacity,
    int tile_capacity) {
    return ((static_cast<int64_t>(seq) * num_query_heads_capacity + head)
        * tile_capacity + tile);
}

__global__ __launch_bounds__(kLogFPreR2TiledThreads)
void reduce_log_f_pre_r2_tiled_row_partial_kernel(
    const int32_t* __restrict__ req_meta_i32,
    const int64_t* __restrict__ req_meta_i64,
    int64_t req_meta_i32_stride_row,
    int64_t req_meta_i32_stride_col,
    int64_t req_meta_i64_stride_row,
    int64_t req_meta_i64_stride_col,
    float* __restrict__ partial_max,
    float* __restrict__ partial_sum,
    int num_seqs,
    int num_query_heads,
    int num_query_heads_capacity,
    int tile_capacity,
    int logical_k_capacity) {
    const int64_t linear = static_cast<int64_t>(blockIdx.x);
    int64_t cursor = linear;
    const int tile = static_cast<int>(cursor % tile_capacity);
    cursor /= tile_capacity;
    const int row = static_cast<int>(cursor % kLogFPreR2TiledRows);
    cursor /= kLogFPreR2TiledRows;
    const int head = static_cast<int>(cursor % num_query_heads);
    const int seq = static_cast<int>(cursor / num_query_heads);
    if (seq >= num_seqs) {
        return;
    }
    const int32_t* meta32 = req_meta_i32
        + static_cast<int64_t>(seq) * req_meta_i32_stride_row;
    const int64_t* meta64 = req_meta_i64
        + static_cast<int64_t>(seq) * req_meta_i64_stride_row;
    __shared__ int contract_ok;
    __shared__ int logical_k;
    __shared__ int last_n;
    __shared__ int scratch_head_stride;
    __shared__ int scratch_row_stride;
    __shared__ int64_t scratch_base_raw;
    if (threadIdx.x == 0) {
        contract_ok = log_f_pre_r2_tiled_contract(
            meta32,
            meta64,
            req_meta_i32_stride_col,
            req_meta_i64_stride_col,
            logical_k_capacity) ? 1 : 0;
        logical_k = meta32[0 * req_meta_i32_stride_col];
        last_n = meta32[2 * req_meta_i32_stride_col];
        scratch_head_stride = meta32[1 * req_meta_i32_stride_col];
        scratch_row_stride = meta32[7 * req_meta_i32_stride_col];
        scratch_base_raw = meta64[1 * req_meta_i64_stride_col];
    }
    __syncthreads();
    if (contract_ok == 0) {
        if (threadIdx.x == 0) {
            log_f_pre_contract_trap();
        }
        return;
    }

    if (row >= last_n) {
        if (threadIdx.x == 0) {
            const int64_t index = log_f_pre_r2_tiled_row_partial_index(
                seq,
                head,
                row,
                tile,
                num_query_heads_capacity,
                tile_capacity);
            partial_max[index] = kLogFPreMinVal;
            partial_sum[index] = 0.0f;
        }
        return;
    }

    const half* scratch = reinterpret_cast<const half*>(scratch_base_raw)
        + static_cast<int64_t>(head) * scratch_head_stride
        + static_cast<int64_t>(row) * scratch_row_stride;
    const int tile_base = tile * kLogFPreR2TiledTileK;
    float values[kLogFPreR2TiledItemsPerThread];
    float local_max = kLogFPreMinVal;
#pragma unroll
    for (int item = 0; item < kLogFPreR2TiledItemsPerThread; ++item) {
        const int token = tile_base + threadIdx.x
            + item * kLogFPreR2TiledThreads;
        const float value = token < logical_k
            ? __half2float(scratch[token])
            : kLogFPreMinVal;
        values[item] = value;
        if (value > kLogFPreMinVal) {
            local_max = local_max > value ? local_max : value;
        }
    }
    const float tile_max = block_reduce_max(local_max);
    float local_sum = 0.0f;
    if (tile_max > kLogFPreMinVal) {
#pragma unroll
        for (int item = 0; item < kLogFPreR2TiledItemsPerThread; ++item) {
            const float value = values[item];
            if (value > kLogFPreMinVal) {
                local_sum += expf(value - tile_max);
            }
        }
    }
    const float tile_sum = block_reduce_sum(local_sum);
    if (threadIdx.x == 0) {
        const int64_t index = log_f_pre_r2_tiled_row_partial_index(
            seq,
            head,
            row,
            tile,
            num_query_heads_capacity,
            tile_capacity);
        partial_max[index] = tile_max;
        partial_sum[index] = tile_sum;
    }
}

__global__ __launch_bounds__(kLogFPreR2TiledThreads)
void reduce_log_f_pre_r2_tiled_row_finalize_kernel(
    const int32_t* __restrict__ req_meta_i32,
    int64_t req_meta_i32_stride_row,
    int64_t req_meta_i32_stride_col,
    const float* __restrict__ partial_max,
    const float* __restrict__ partial_sum,
    float* __restrict__ row_lse,
    int* __restrict__ row_has,
    int num_seqs,
    int num_query_heads,
    int num_query_heads_capacity,
    int tile_capacity) {
    const int64_t linear = static_cast<int64_t>(blockIdx.x);
    int64_t cursor = linear;
    const int row = static_cast<int>(cursor % kLogFPreR2TiledRows);
    cursor /= kLogFPreR2TiledRows;
    const int head = static_cast<int>(cursor % num_query_heads);
    const int seq = static_cast<int>(cursor / num_query_heads);
    if (seq >= num_seqs) {
        return;
    }
    const int32_t* meta32 = req_meta_i32
        + static_cast<int64_t>(seq) * req_meta_i32_stride_row;
    const int last_n = meta32[2 * req_meta_i32_stride_col];
    if (last_n != 1 && last_n != 2) {
        if (threadIdx.x == 0) {
            log_f_pre_contract_trap();
        }
        return;
    }
    // R=1 accumulation still uses the fixed two-row workspace geometry.  The
    // second row is structural padding, not an all-masked captured row.  Own
    // that distinction here instead of inferring it from a partial sentinel:
    // active all-masked rows remain visible to the stage3/4 causal invariant,
    // while structural padding is deterministically excluded from row_count.
    if (row >= last_n) {
        if (threadIdx.x == 0) {
            const int64_t index = log_f_pre_r2_tiled_row_state_index(
                seq, head, row, num_query_heads_capacity);
            row_has[index] = 0;
            row_lse[index] = kLogFPreMinVal;
        }
        return;
    }
    float local_max = kLogFPreMinVal;
    for (int tile = threadIdx.x; tile < tile_capacity; tile += blockDim.x) {
        const int64_t index = log_f_pre_r2_tiled_row_partial_index(
            seq,
            head,
            row,
            tile,
            num_query_heads_capacity,
            tile_capacity);
        const float value = partial_max[index];
        if (value > kLogFPreMinVal) {
            local_max = local_max > value ? local_max : value;
        }
    }
    const float global_max = block_reduce_max(local_max);
    __shared__ int has_row;
    if (threadIdx.x == 0) {
        has_row = global_max > kLogFPreMinVal ? 1 : 0;
    }
    __syncthreads();
    float local_sum = 0.0f;
    if (has_row != 0) {
        for (int tile = threadIdx.x; tile < tile_capacity; tile += blockDim.x) {
            const int64_t index = log_f_pre_r2_tiled_row_partial_index(
                seq,
                head,
                row,
                tile,
                num_query_heads_capacity,
                tile_capacity);
            const float tile_max = partial_max[index];
            if (tile_max > kLogFPreMinVal) {
                local_sum += partial_sum[index] * expf(tile_max - global_max);
            }
        }
    }
    const float global_sum = block_reduce_sum(local_sum);
    if (threadIdx.x == 0) {
        const int64_t index = log_f_pre_r2_tiled_row_state_index(
            seq, head, row, num_query_heads_capacity);
        row_has[index] = has_row;
        row_lse[index] = has_row != 0
            ? global_max + logf(global_sum + 1.0e-20f)
            : kLogFPreMinVal;
    }
}

// 合并前片（out 里已存的归一化部分窗口值 prev，来自 n_prev 行）与本片
// 未归一化 sum-form（cur，来自 n_cur 行），返回 n_prev+n_cur 行的归一化值。
// prev 的归一化用的是全局行数 n_prev（非 per-key 贡献数），故 alpha*prev+
// log(n_prev) 精确还原 per-key 的 log S_prev；合并数学等价于对全部行做
// 单次 reduce。定义在 tiled stage3 之前，让 generic/tiled 共用唯一语义 owner。
__device__ __forceinline__ bool log_f_pre_accum_value_present(float value) {
    // fp16 stores kLogFPreMinVal as -inf.  NaN is an explicit poison state,
    // not an absent token: once produced it must survive every later chunk.
    return isnan(value) || value > kLogFPreMinVal;
}

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
        if (log_f_pre_accum_value_present(p)) {
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
        const float m = prev_log_s > cur_unnormalized
            ? prev_log_s
            : cur_unnormalized;
        const float s = expf(prev_log_s - m)
            + expf(cur_unnormalized - m);
        return (m + logf(s) - logf(n_cum_f)) / alpha;
    }
    if (has_prev) {
        const float prev_log_s = alpha * prev + logf(n_prev_f);
        return (prev_log_s - logf(n_cum_f)) / alpha;
    }
    return (cur_unnormalized - logf(n_cum_f)) / alpha;
}

__global__ __launch_bounds__(kLogFPreR2TiledThreads)
void reduce_log_f_pre_r2_tiled_log_f_partial_kernel(
    const int32_t* __restrict__ req_meta_i32,
    const int64_t* __restrict__ req_meta_i64,
    int64_t req_meta_i32_stride_row,
    int64_t req_meta_i32_stride_col,
    int64_t req_meta_i64_stride_row,
    int64_t req_meta_i64_stride_col,
    const float* __restrict__ row_lse,
    const int* __restrict__ row_has,
    float* __restrict__ token_partial_max,
    float* __restrict__ token_partial_sum,
    int* __restrict__ token_has,
    int num_seqs,
    int num_query_heads,
    int num_query_heads_capacity,
    int tile_capacity,
    int logical_k_capacity) {
    const int64_t linear = static_cast<int64_t>(blockIdx.x);
    int64_t cursor = linear;
    const int tile = static_cast<int>(cursor % tile_capacity);
    cursor /= tile_capacity;
    const int head = static_cast<int>(cursor % num_query_heads);
    const int seq = static_cast<int>(cursor / num_query_heads);
    if (seq >= num_seqs) {
        return;
    }
    const int32_t* meta32 = req_meta_i32
        + static_cast<int64_t>(seq) * req_meta_i32_stride_row;
    const int64_t* meta64 = req_meta_i64
        + static_cast<int64_t>(seq) * req_meta_i64_stride_row;
    __shared__ int contract_ok;
    __shared__ int logical_k;
    __shared__ int last_n;
    __shared__ int scratch_head_stride;
    __shared__ int scratch_row_stride;
    __shared__ int out_head_stride;
    __shared__ int64_t scratch_base_raw;
    __shared__ int64_t out_base_raw;
    __shared__ float row_lse0;
    __shared__ float row_lse1;
    __shared__ int row_has0;
    __shared__ int row_has1;
    __shared__ float log_row_count;
    __shared__ float accum_n_prev_f;
    __shared__ float accum_n_cum_f;
    __shared__ int accum_prev_capacity;
    __shared__ int accum_do_merge;
    __shared__ int accum_mode;
    __shared__ int nonfinite_row_lse;
    if (threadIdx.x == 0) {
        contract_ok = log_f_pre_r2_tiled_contract(
            meta32,
            meta64,
            req_meta_i32_stride_col,
            req_meta_i64_stride_col,
            logical_k_capacity) ? 1 : 0;
        logical_k = meta32[0 * req_meta_i32_stride_col];
        last_n = meta32[2 * req_meta_i32_stride_col];
        scratch_head_stride = meta32[1 * req_meta_i32_stride_col];
        scratch_row_stride = meta32[7 * req_meta_i32_stride_col];
        out_head_stride = meta32[6 * req_meta_i32_stride_col];
        scratch_base_raw = meta64[1 * req_meta_i64_stride_col];
        out_base_raw = meta64[2 * req_meta_i64_stride_col];
        const int64_t row0 = log_f_pre_r2_tiled_row_state_index(
            seq, head, 0, num_query_heads_capacity);
        const int64_t row1 = log_f_pre_r2_tiled_row_state_index(
            seq, head, 1, num_query_heads_capacity);
        row_lse0 = row_lse[row0];
        row_lse1 = row_lse[row1];
        row_has0 = row_has[row0];
        row_has1 = row_has[row1];
        const int row_count = row_has0 + row_has1;
        log_row_count = row_count > 0
            ? logf(static_cast<float>(row_count))
            : 0.0f;
        const int flags = meta32[5 * req_meta_i32_stride_col];
        const int prev_rows = meta32[8 * req_meta_i32_stride_col];
        accum_prev_capacity = meta32[9 * req_meta_i32_stride_col];
        accum_mode = flags == 24 ? 1 : 0;
        accum_do_merge = accum_mode != 0 && prev_rows > 0 ? 1 : 0;
        accum_n_prev_f = static_cast<float>(prev_rows);
        accum_n_cum_f = static_cast<float>(prev_rows + row_count);
        nonfinite_row_lse = (
            (row_has0 != 0 && !isfinite(row_lse0))
            || (row_has1 != 0 && !isfinite(row_lse1))) ? 1 : 0;
        if (accum_mode != 0 && row_count != last_n) {
            contract_ok = 0;
        }
    }
    __syncthreads();
    if (contract_ok == 0) {
        if (threadIdx.x == 0) {
            log_f_pre_contract_trap();
        }
        return;
    }

    const half* scratch_head = reinterpret_cast<const half*>(scratch_base_raw)
        + static_cast<int64_t>(head) * scratch_head_stride;
    half* out_head = reinterpret_cast<half*>(out_base_raw)
        + static_cast<int64_t>(head) * out_head_stride;
    const int tile_base = tile * kLogFPreR2TiledTileK;

    // The fourth stage owns the 256-lane generic-order replay when a +inf row
    // made LSE non-finite.  Keep the prior accumulated output intact here so
    // that replay can merge it exactly once.
    if (nonfinite_row_lse != 0) {
        if (threadIdx.x == 0) {
            const int64_t index = log_f_pre_r2_tiled_token_partial_index(
                seq,
                head,
                tile,
                num_query_heads_capacity,
                tile_capacity);
            token_partial_max[index] = kLogFPreMinVal;
            token_partial_sum[index] = 0.0f;
            token_has[index] = 0;
        }
        return;
    }

    float log_f_values[kLogFPreR2TiledItemsPerThread];
    unsigned int has_mask = 0U;
    unsigned int poison_mask = 0U;
    float local_max = kLogFPreMinVal;
#pragma unroll
    for (int item = 0; item < kLogFPreR2TiledItemsPerThread; ++item) {
        const int token = tile_base + threadIdx.x
            + item * kLogFPreR2TiledThreads;
        float log_f = kLogFPreMinVal;
        bool has_token = false;
        if (token < logical_k) {
            const float value0 = __half2float(scratch_head[token]);
            const float value1 = last_n == 2
                ? __half2float(scratch_head[scratch_row_stride + token])
                : kLogFPreMinVal;
            const bool valid0 = row_has0 != 0 && value0 > kLogFPreMinVal;
            const bool valid1 = row_has1 != 0 && value1 > kLogFPreMinVal;
            if (valid0 || valid1) {
                float score_max = kLogFPreMinVal;
                float score0 = kLogFPreMinVal;
                float score1 = kLogFPreMinVal;
                if (valid0) {
                    score0 = 0.5f * (value0 - row_lse0);
                    score_max = score0;
                }
                if (valid1) {
                    score1 = 0.5f * (value1 - row_lse1);
                    score_max = score_max > score1 ? score_max : score1;
                }
                float score_sum = 0.0f;
                if (valid0) {
                    score_sum += expf(score0 - score_max);
                }
                if (valid1) {
                    score_sum += expf(score1 - score_max);
                }
                const float cur_unnormalized = score_max
                    + logf(score_sum + 1.0e-20f);
                if (accum_do_merge != 0) {
                    log_f = merge_log_f_pre_accum<half>(
                        out_head,
                        token,
                        accum_prev_capacity,
                        accum_n_prev_f,
                        cur_unnormalized,
                        true,
                        accum_n_cum_f,
                        0.5f,
                        false,
                        &has_token);
                } else {
                    log_f = 2.0f * (cur_unnormalized - log_row_count);
                    has_token = true;
                }
            } else if (accum_do_merge != 0) {
                log_f = merge_log_f_pre_accum<half>(
                    out_head,
                    token,
                    accum_prev_capacity,
                    accum_n_prev_f,
                    kLogFPreMinVal,
                    false,
                    accum_n_cum_f,
                    0.5f,
                    false,
                    &has_token);
            }
            out_head[token] = __float2half_rn(log_f);
            if (has_token) {
                has_mask |= 1U << item;
                if (isnan(log_f)) {
                    poison_mask |= 1U << item;
                } else {
                    local_max = local_max > log_f ? local_max : log_f;
                }
            }
        }
        log_f_values[item] = accum_do_merge != 0 && token < logical_k
            ? __half2float(out_head[token])
            : log_f;
    }
    const float tile_max = block_reduce_max(local_max);
    const int tile_has = __syncthreads_count(has_mask != 0U) > 0 ? 1 : 0;
    const int tile_poison = __syncthreads_count(poison_mask != 0U) > 0
        ? 1
        : 0;
    float local_sum = 0.0f;
    if (tile_has != 0 && tile_poison == 0) {
#pragma unroll
        for (int item = 0; item < kLogFPreR2TiledItemsPerThread; ++item) {
            if ((has_mask & (1U << item)) != 0U
                && log_f_pre_accum_value_present(log_f_values[item])) {
                local_sum += expf(log_f_values[item] - tile_max);
            }
        }
    }
    const float tile_sum = block_reduce_sum(local_sum);
    if (threadIdx.x == 0) {
        const int64_t index = log_f_pre_r2_tiled_token_partial_index(
            seq,
            head,
            tile,
            num_query_heads_capacity,
            tile_capacity);
        token_partial_max[index] = tile_poison != 0 ? NAN : tile_max;
        token_partial_sum[index] = tile_poison != 0 ? NAN : tile_sum;
        token_has[index] = tile_poison != 0 ? 2 : tile_has;
    }
}

__global__ __launch_bounds__(kLogFPreR2TiledThreads)
void reduce_log_f_pre_r2_tiled_denom_finalize_kernel(
    const int32_t* __restrict__ req_meta_i32,
    const int64_t* __restrict__ req_meta_i64,
    int64_t req_meta_i32_stride_row,
    int64_t req_meta_i32_stride_col,
    int64_t req_meta_i64_stride_row,
    int64_t req_meta_i64_stride_col,
    const float* __restrict__ row_lse,
    const int* __restrict__ row_has,
    const float* __restrict__ token_partial_max,
    const float* __restrict__ token_partial_sum,
    const int* __restrict__ token_has,
    int num_seqs,
    int num_query_heads,
    int num_query_heads_capacity,
    int tile_capacity,
    int logical_k_capacity) {
    const int64_t linear = static_cast<int64_t>(blockIdx.x);
    const int head = static_cast<int>(linear % num_query_heads);
    const int seq = static_cast<int>(linear / num_query_heads);
    if (seq >= num_seqs) {
        return;
    }
    const int32_t* meta32 = req_meta_i32
        + static_cast<int64_t>(seq) * req_meta_i32_stride_row;
    const int64_t* meta64 = req_meta_i64
        + static_cast<int64_t>(seq) * req_meta_i64_stride_row;
    __shared__ int contract_ok;
    __shared__ int logical_k;
    __shared__ int last_n;
    __shared__ int scratch_head_stride;
    __shared__ int scratch_row_stride;
    __shared__ int out_head_stride;
    __shared__ int64_t scratch_base_raw;
    __shared__ int64_t out_base_raw;
    __shared__ int64_t denom_base_raw;
    __shared__ float row_lse_s[2];
    __shared__ int row_has_s[2];
    __shared__ float row_count_f;
    __shared__ float token_max_s;
    __shared__ int nonfinite_row_lse;
    __shared__ float accum_n_prev_f;
    __shared__ float accum_n_cum_f;
    __shared__ int accum_prev_capacity;
    __shared__ int accum_do_merge;
    __shared__ int accum_mode;
    if (threadIdx.x == 0) {
        contract_ok = log_f_pre_r2_tiled_contract(
            meta32,
            meta64,
            req_meta_i32_stride_col,
            req_meta_i64_stride_col,
            logical_k_capacity) ? 1 : 0;
        logical_k = meta32[0 * req_meta_i32_stride_col];
        last_n = meta32[2 * req_meta_i32_stride_col];
        scratch_head_stride = meta32[1 * req_meta_i32_stride_col];
        scratch_row_stride = meta32[7 * req_meta_i32_stride_col];
        out_head_stride = meta32[6 * req_meta_i32_stride_col];
        scratch_base_raw = meta64[1 * req_meta_i64_stride_col];
        out_base_raw = meta64[2 * req_meta_i64_stride_col];
        denom_base_raw = meta64[3 * req_meta_i64_stride_col];
        for (int row = 0; row < 2; ++row) {
            const int64_t index = log_f_pre_r2_tiled_row_state_index(
                seq, head, row, num_query_heads_capacity);
            row_lse_s[row] = row_lse[index];
            row_has_s[row] = row_has[index];
        }
        nonfinite_row_lse = (
            (row_has_s[0] != 0 && !isfinite(row_lse_s[0]))
            || (row_has_s[1] != 0 && !isfinite(row_lse_s[1]))) ? 1 : 0;
        const int row_count = row_has_s[0] + row_has_s[1];
        row_count_f = row_count > 0 ? static_cast<float>(row_count) : 1.0f;
        const int flags = meta32[5 * req_meta_i32_stride_col];
        const int prev_rows = meta32[8 * req_meta_i32_stride_col];
        accum_prev_capacity = meta32[9 * req_meta_i32_stride_col];
        accum_mode = flags == 24 ? 1 : 0;
        accum_do_merge = accum_mode != 0 && prev_rows > 0 ? 1 : 0;
        accum_n_prev_f = static_cast<float>(prev_rows);
        accum_n_cum_f = static_cast<float>(prev_rows + row_count);
        if (accum_mode != 0 && row_count != last_n) {
            contract_ok = 0;
        }
    }
    __syncthreads();
    if (contract_ok == 0) {
        if (threadIdx.x == 0) {
            log_f_pre_contract_trap();
        }
        return;
    }

    const half* scratch_head = reinterpret_cast<const half*>(scratch_base_raw)
        + static_cast<int64_t>(head) * scratch_head_stride;
    half* out_head = reinterpret_cast<half*>(out_base_raw)
        + static_cast<int64_t>(head) * out_head_stride;
    float* denom = reinterpret_cast<float*>(denom_base_raw) + head;

    // Tile combination changes the reduction tree.  If +inf made a row LSE
    // non-finite, replay the generic 256-lane order inside this already-planned
    // fourth stage.  This preserves token-local NaN behavior without a host
    // branch, extra launch, whole-head sanitize, allocation, or fallback.
    if (nonfinite_row_lse != 0) {
        for (int row = 0; row < last_n; ++row) {
            float local_max = kLogFPreMinVal;
            for (int token = threadIdx.x; token < logical_k; token += blockDim.x) {
                const float value = __half2float(
                    scratch_head[static_cast<int64_t>(row)
                        * scratch_row_stride + token]);
                if (value > kLogFPreMinVal) {
                    local_max = local_max > value ? local_max : value;
                }
            }
            const float row_max = block_reduce_max(local_max);
            if (threadIdx.x == 0) {
                row_has_s[row] = row_max > kLogFPreMinVal ? 1 : 0;
                row_lse_s[row] = row_max;
            }
            __syncthreads();
            float local_sum = 0.0f;
            if (row_has_s[row] != 0) {
                for (int token = threadIdx.x; token < logical_k;
                     token += blockDim.x) {
                    const float value = __half2float(
                        scratch_head[static_cast<int64_t>(row)
                            * scratch_row_stride + token]);
                    if (value > kLogFPreMinVal) {
                        local_sum += expf(value - row_lse_s[row]);
                    }
                }
            }
            const float row_sum = block_reduce_sum(local_sum);
            if (threadIdx.x == 0) {
                row_lse_s[row] = row_has_s[row] != 0
                    ? row_lse_s[row] + logf(row_sum + 1.0e-20f)
                    : kLogFPreMinVal;
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            const int row_count = row_has_s[0] + row_has_s[1];
            row_count_f = row_count > 0
                ? static_cast<float>(row_count)
                : 1.0f;
            accum_n_cum_f = accum_n_prev_f
                + static_cast<float>(row_count);
        }
        __syncthreads();

        float local_token_max = kLogFPreMinVal;
        bool local_token_poison = false;
        for (int token = threadIdx.x; token < logical_k; token += blockDim.x) {
            bool has_token = false;
            float log_f = kLogFPreMinVal;
            if (accum_do_merge != 0) {
                bool has_cur = false;
                const float cur = compute_log_f_pre_token_unnormalized(
                    scratch_head,
                    static_cast<int64_t>(scratch_row_stride),
                    token,
                    last_n,
                    row_lse_s,
                    row_has_s,
                    0.5f,
                    false,
                    &has_cur);
                log_f = merge_log_f_pre_accum<half>(
                    out_head,
                    token,
                    accum_prev_capacity,
                    accum_n_prev_f,
                    cur,
                    has_cur,
                    accum_n_cum_f,
                    0.5f,
                    false,
                    &has_token);
            } else {
                log_f = compute_log_f_pre_token(
                    scratch_head,
                    static_cast<int64_t>(scratch_row_stride),
                    token,
                    last_n,
                    row_lse_s,
                    row_has_s,
                    row_count_f,
                    0.5f,
                    false,
                    &has_token);
            }
            out_head[token] = __float2half_rn(log_f);
            if (has_token) {
                if (isnan(log_f)) {
                    local_token_poison = true;
                } else {
                    local_token_max = local_token_max > log_f
                        ? local_token_max
                        : log_f;
                }
            }
        }
        const int token_poison = __syncthreads_count(local_token_poison) > 0
            ? 1
            : 0;
        if (token_poison != 0) {
            if (threadIdx.x == 0) {
                *denom = NAN;
            }
            return;
        }
        const float token_max = block_reduce_max(local_token_max);
        if (threadIdx.x == 0) {
            token_max_s = token_max;
        }
        __syncthreads();
        if (token_max_s <= kLogFPreMinVal) {
            if (threadIdx.x == 0) {
                *denom = 0.0f;
            }
            return;
        }
        float local_token_sum = 0.0f;
        if (accum_do_merge != 0) {
            for (int token = threadIdx.x; token < logical_k;
                 token += blockDim.x) {
                const float value = __half2float(out_head[token]);
                if (log_f_pre_accum_value_present(value)) {
                    local_token_sum += expf(value - token_max_s);
                }
            }
        } else {
            for (int token = threadIdx.x; token < logical_k;
                 token += blockDim.x) {
                bool has_token = false;
                const float log_f = compute_log_f_pre_token(
                    scratch_head,
                    static_cast<int64_t>(scratch_row_stride),
                    token,
                    last_n,
                    row_lse_s,
                    row_has_s,
                    row_count_f,
                    0.5f,
                    false,
                    &has_token);
                if (has_token) {
                    local_token_sum += expf(log_f - token_max_s);
                }
            }
        }
        const float token_sum = block_reduce_sum(local_token_sum);
        if (threadIdx.x == 0) {
            *denom = token_max_s + logf(token_sum + 1.0e-20f);
        }
        return;
    }

    bool local_has = false;
    bool local_poison = false;
    float local_max = kLogFPreMinVal;
    for (int tile = threadIdx.x; tile < tile_capacity; tile += blockDim.x) {
        const int64_t index = log_f_pre_r2_tiled_token_partial_index(
            seq,
            head,
            tile,
            num_query_heads_capacity,
            tile_capacity);
        if (token_has[index] == 2) {
            local_poison = true;
        } else if (token_has[index] == 1) {
            local_has = true;
            const float value = token_partial_max[index];
            local_max = local_max > value ? local_max : value;
        }
    }
    const int any_poison = __syncthreads_count(local_poison) > 0 ? 1 : 0;
    if (any_poison != 0) {
        if (threadIdx.x == 0) {
            *denom = NAN;
        }
        return;
    }
    const int any_token = __syncthreads_count(local_has) > 0 ? 1 : 0;
    const float token_max = block_reduce_max(local_max);
    if (any_token == 0) {
        if (threadIdx.x == 0) {
            *denom = 0.0f;
        }
        return;
    }
    float local_sum = 0.0f;
    for (int tile = threadIdx.x; tile < tile_capacity; tile += blockDim.x) {
        const int64_t index = log_f_pre_r2_tiled_token_partial_index(
            seq,
            head,
            tile,
            num_query_heads_capacity,
            tile_capacity);
        if (token_has[index] == 1) {
            local_sum += token_partial_sum[index]
                * expf(token_partial_max[index] - token_max);
        }
    }
    const float token_sum = block_reduce_sum(local_sum);
    if (threadIdx.x == 0) {
        *denom = token_max + logf(token_sum + 1.0e-20f);
    }
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
    __shared__ int accum_rows_valid;

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
        accum_rows_valid = !accum_mode || row_count == max_r ? 1 : 0;
    }
    __syncthreads();
    if (accum_rows_valid == 0) {
        // The scalar n_prev ABI is exact only when every causal capture row
        // contributes for every head.  Reject violated semantics on device;
        // never silently merge with a per-head row-count mismatch.
        if (threadIdx.x == 0) {
            log_f_pre_contract_trap();
        }
        return;
    }

    float local_token_max = kLogFPreMinVal;
    bool local_token_poison = false;
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
                if (isnan(log_f_pre)) {
                    local_token_poison = true;
                } else {
                    local_token_max = local_token_max > log_f_pre
                        ? local_token_max
                        : log_f_pre;
                }
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
                if (isnan(merged)) {
                    local_token_poison = true;
                } else {
                    local_token_max = local_token_max > merged
                        ? local_token_max
                        : merged;
                }
            }
        }
    }

    const int token_poison = __syncthreads_count(local_token_poison) > 0
        ? 1
        : 0;
    if (token_poison != 0) {
        if (threadIdx.x == 0) {
            denom_ptr[pid_h] = NAN;
        }
        return;
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
            if (log_f_pre_accum_value_present(v)) {
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

template <int kLogicalKBucket, int kThreads, int kItemsPerThread>
void launch_log_f_pre_r2_resident(
    const torch::Tensor& req_meta_i32,
    const torch::Tensor& req_meta_i64,
    int num_seqs,
    int num_query_heads,
    cudaStream_t stream) {
    const dim3 row_blocks(num_query_heads, 2, num_seqs);
    reduce_log_f_pre_r2_resident_row_lse_kernel<
        kLogicalKBucket, kThreads, kItemsPerThread><<<
        row_blocks, kThreads, 0, stream>>>(
        req_meta_i32.data_ptr<int32_t>(),
        req_meta_i64.data_ptr<int64_t>(),
        req_meta_i32.stride(0),
        req_meta_i32.stride(1),
        req_meta_i64.stride(0),
        req_meta_i64.stride(1));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    const dim3 output_blocks(num_query_heads, 1, num_seqs);
    reduce_log_f_pre_r2_resident_log_f_denom_kernel<
        kLogicalKBucket, kThreads, kItemsPerThread><<<
        output_blocks, kThreads, 0, stream>>>(
        req_meta_i32.data_ptr<int32_t>(),
        req_meta_i64.data_ptr<int64_t>(),
        req_meta_i32.stride(0),
        req_meta_i32.stride(1),
        req_meta_i64.stride(0),
        req_meta_i64.stride(1));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void reduce_log_f_pre_scratch_r2_alpha0p5_fp16_resident_cuda(
    torch::Tensor req_meta_i32,
    torch::Tensor req_meta_i64,
    int64_t num_seqs,
    int64_t num_query_heads,
    int64_t logical_k_bucket) {
    TORCH_CHECK(req_meta_i32.is_cuda(), "req_meta_i32 must be CUDA");
    TORCH_CHECK(req_meta_i64.is_cuda(), "req_meta_i64 must be CUDA");
    TORCH_CHECK(
        req_meta_i32.get_device() == req_meta_i64.get_device(),
        "exact R2 metadata tensors must share one CUDA device");
    const c10::cuda::CUDAGuard device_guard(req_meta_i32.device());
    TORCH_CHECK(
        req_meta_i32.scalar_type() == torch::kInt32,
        "req_meta_i32 must be int32");
    TORCH_CHECK(
        req_meta_i64.scalar_type() == torch::kInt64,
        "req_meta_i64 must be int64");
    TORCH_CHECK(
        req_meta_i32.dim() == 2 && req_meta_i32.size(0) >= num_seqs
            && req_meta_i32.size(1) >= 10 && req_meta_i32.is_contiguous(),
        "resident R2 req_meta_i32 must be contiguous [>=N,>=10]");
    TORCH_CHECK(
        req_meta_i64.dim() == 2 && req_meta_i64.size(0) >= num_seqs
            && req_meta_i64.size(1) >= 4 && req_meta_i64.is_contiguous(),
        "resident R2 req_meta_i64 must be contiguous [>=N,>=4]");
    TORCH_CHECK(num_seqs > 0 && num_seqs <= 65535,
                "resident R2 num_seqs must be in [1,65535]");
    TORCH_CHECK(num_query_heads > 0 && num_query_heads <= 65535,
                "resident R2 num_query_heads must be in [1,65535]");
    TORCH_CHECK(
        logical_k_bucket == kLogFPreR2Bucket12K
            || logical_k_bucket == kLogFPreR2Bucket16K
            || logical_k_bucket == kLogFPreR2Bucket24K
            || logical_k_bucket == kLogFPreR2Bucket32K,
        "resident R2 logical_k_bucket must be one of 12288/16384/24576/32768");
    const cudaDeviceProp* props = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(
        props != nullptr && props->major == 8 && props->minor == 0,
        "resident R2 selector reduce requires SM80");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int n = static_cast<int>(num_seqs);
    const int h = static_cast<int>(num_query_heads);
    switch (logical_k_bucket) {
        case kLogFPreR2Bucket12K:
            launch_log_f_pre_r2_resident<12288, 512, 24>(
                req_meta_i32, req_meta_i64, n, h, stream);
            break;
        case kLogFPreR2Bucket16K:
            launch_log_f_pre_r2_resident<16384, 512, 32>(
                req_meta_i32, req_meta_i64, n, h, stream);
            break;
        case kLogFPreR2Bucket24K:
            launch_log_f_pre_r2_resident<24576, 768, 32>(
                req_meta_i32, req_meta_i64, n, h, stream);
            break;
        case kLogFPreR2Bucket32K:
            launch_log_f_pre_r2_resident<32768, 1024, 32>(
                req_meta_i32, req_meta_i64, n, h, stream);
            break;
    }
}

namespace {

void check_log_f_pre_r2_tiled_capacities(
    int64_t num_seqs_capacity,
    int64_t num_query_heads_capacity,
    int64_t logical_k_capacity) {
    TORCH_CHECK(
        num_seqs_capacity > 0 && num_seqs_capacity <= 65535,
        "tiled R2 num_seqs_capacity must be in [1,65535]");
    TORCH_CHECK(
        num_query_heads_capacity > 0 && num_query_heads_capacity <= 65535,
        "tiled R2 num_query_heads_capacity must be in [1,65535]");
    TORCH_CHECK(
        logical_k_capacity > 0 && logical_k_capacity <= 2147483647LL,
        "tiled R2 logical_k_capacity must fit positive int32");
    const int64_t tile_capacity =
        (logical_k_capacity + kLogFPreR2TiledTileK - 1)
        / kLogFPreR2TiledTileK;
    TORCH_CHECK(
        tile_capacity > 0 && tile_capacity <= 65535,
        "tiled R2 tile_capacity must be in [1,65535]");
    const long double words = static_cast<long double>(num_seqs_capacity)
        * static_cast<long double>(num_query_heads_capacity)
        * (7.0L * static_cast<long double>(tile_capacity) + 4.0L);
    TORCH_CHECK(
        words <= static_cast<long double>(INT64_MAX / 4),
        "tiled R2 workspace size overflows int64");
}

}  // namespace

int64_t reduce_log_f_pre_scratch_r2_alpha0p5_fp16_tiled_workspace_nbytes_cuda(
    int64_t num_seqs_capacity,
    int64_t num_query_heads_capacity,
    int64_t logical_k_capacity) {
    check_log_f_pre_r2_tiled_capacities(
        num_seqs_capacity,
        num_query_heads_capacity,
        logical_k_capacity);
    const LogFPreR2TiledWorkspaceLayout layout =
        log_f_pre_r2_tiled_workspace_layout(
            num_seqs_capacity,
            num_query_heads_capacity,
            logical_k_capacity);
    return layout.total_words * 4;
}

void reduce_log_f_pre_scratch_r2_alpha0p5_fp16_tiled_cuda(
    torch::Tensor req_meta_i32,
    torch::Tensor req_meta_i64,
    torch::Tensor workspace,
    int64_t num_seqs,
    int64_t num_query_heads,
    int64_t num_seqs_capacity,
    int64_t num_query_heads_capacity,
    int64_t logical_k_capacity) {
    check_log_f_pre_r2_tiled_capacities(
        num_seqs_capacity,
        num_query_heads_capacity,
        logical_k_capacity);
    TORCH_CHECK(req_meta_i32.is_cuda(), "req_meta_i32 must be CUDA");
    TORCH_CHECK(req_meta_i64.is_cuda(), "req_meta_i64 must be CUDA");
    TORCH_CHECK(workspace.is_cuda(), "tiled R2 workspace must be CUDA");
    TORCH_CHECK(
        req_meta_i32.get_device() == req_meta_i64.get_device()
            && req_meta_i32.get_device() == workspace.get_device(),
        "tiled R2 metadata and workspace must share one CUDA device");
    const c10::cuda::CUDAGuard device_guard(req_meta_i32.device());
    TORCH_CHECK(
        req_meta_i32.scalar_type() == torch::kInt32,
        "req_meta_i32 must be int32");
    TORCH_CHECK(
        req_meta_i64.scalar_type() == torch::kInt64,
        "req_meta_i64 must be int64");
    TORCH_CHECK(
        workspace.scalar_type() == torch::kUInt8,
        "tiled R2 workspace must be uint8");
    TORCH_CHECK(
        req_meta_i32.dim() == 2 && req_meta_i32.size(0) >= num_seqs
            && req_meta_i32.size(1) >= 10 && req_meta_i32.is_contiguous(),
        "tiled R2 req_meta_i32 must be contiguous [>=N,>=10]");
    TORCH_CHECK(
        req_meta_i64.dim() == 2 && req_meta_i64.size(0) >= num_seqs
            && req_meta_i64.size(1) >= 4 && req_meta_i64.is_contiguous(),
        "tiled R2 req_meta_i64 must be contiguous [>=N,>=4]");
    TORCH_CHECK(
        workspace.dim() == 1 && workspace.is_contiguous(),
        "tiled R2 workspace must be contiguous rank-1 uint8");
    TORCH_CHECK(
        num_seqs > 0 && num_seqs <= num_seqs_capacity,
        "tiled R2 num_seqs must be positive and within capacity");
    TORCH_CHECK(
        num_query_heads > 0
            && num_query_heads <= num_query_heads_capacity,
        "tiled R2 num_query_heads must be positive and within capacity");

    const LogFPreR2TiledWorkspaceLayout layout =
        log_f_pre_r2_tiled_workspace_layout(
            num_seqs_capacity,
            num_query_heads_capacity,
            logical_k_capacity);
    TORCH_CHECK(
        workspace.numel() >= layout.total_words * 4,
        "tiled R2 workspace is smaller than its declared capacity");
    TORCH_CHECK(
        reinterpret_cast<uintptr_t>(workspace.data_ptr()) % alignof(float) == 0,
        "tiled R2 workspace must be float-aligned");
    TORCH_CHECK(
        workspace.data_ptr() != req_meta_i32.data_ptr()
            && workspace.data_ptr() != req_meta_i64.data_ptr(),
        "tiled R2 workspace must not alias metadata");

    const cudaDeviceProp* props = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(props != nullptr, "tiled R2 CUDA device properties unavailable");
    const int64_t n = num_seqs;
    const int64_t h = num_query_heads;
    const int64_t tiles = layout.tile_capacity;
    const int64_t row_partial_blocks = n * h * 2 * tiles;
    const int64_t row_finalize_blocks = n * h * 2;
    const int64_t token_partial_blocks = n * h * tiles;
    const int64_t denom_finalize_blocks = n * h;
    TORCH_CHECK(
        row_partial_blocks <= props->maxGridSize[0]
            && row_finalize_blocks <= props->maxGridSize[0]
            && token_partial_blocks <= props->maxGridSize[0]
            && denom_finalize_blocks <= props->maxGridSize[0],
        "tiled R2 launch exceeds device grid-x capacity");

    uint8_t* raw_workspace = workspace.data_ptr<uint8_t>();
    float* row_partial_max = reinterpret_cast<float*>(raw_workspace)
        + layout.row_partial_max;
    float* row_partial_sum = reinterpret_cast<float*>(raw_workspace)
        + layout.row_partial_sum;
    float* row_lse = reinterpret_cast<float*>(raw_workspace) + layout.row_lse;
    int* row_has = reinterpret_cast<int*>(raw_workspace)
        + layout.row_has;
    float* token_partial_max = reinterpret_cast<float*>(raw_workspace)
        + layout.token_partial_max;
    float* token_partial_sum = reinterpret_cast<float*>(raw_workspace)
        + layout.token_partial_sum;
    int* token_has = reinterpret_cast<int*>(raw_workspace)
        + layout.token_has;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    reduce_log_f_pre_r2_tiled_row_partial_kernel<<<
        static_cast<unsigned int>(row_partial_blocks),
        kLogFPreR2TiledThreads,
        0,
        stream>>>(
        req_meta_i32.data_ptr<int32_t>(),
        req_meta_i64.data_ptr<int64_t>(),
        req_meta_i32.stride(0),
        req_meta_i32.stride(1),
        req_meta_i64.stride(0),
        req_meta_i64.stride(1),
        row_partial_max,
        row_partial_sum,
        static_cast<int>(n),
        static_cast<int>(h),
        static_cast<int>(num_query_heads_capacity),
        static_cast<int>(tiles),
        static_cast<int>(logical_k_capacity));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    reduce_log_f_pre_r2_tiled_row_finalize_kernel<<<
        static_cast<unsigned int>(row_finalize_blocks),
        kLogFPreR2TiledThreads,
        0,
        stream>>>(
        req_meta_i32.data_ptr<int32_t>(),
        req_meta_i32.stride(0),
        req_meta_i32.stride(1),
        row_partial_max,
        row_partial_sum,
        row_lse,
        row_has,
        static_cast<int>(n),
        static_cast<int>(h),
        static_cast<int>(num_query_heads_capacity),
        static_cast<int>(tiles));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    reduce_log_f_pre_r2_tiled_log_f_partial_kernel<<<
        static_cast<unsigned int>(token_partial_blocks),
        kLogFPreR2TiledThreads,
        0,
        stream>>>(
        req_meta_i32.data_ptr<int32_t>(),
        req_meta_i64.data_ptr<int64_t>(),
        req_meta_i32.stride(0),
        req_meta_i32.stride(1),
        req_meta_i64.stride(0),
        req_meta_i64.stride(1),
        row_lse,
        row_has,
        token_partial_max,
        token_partial_sum,
        token_has,
        static_cast<int>(n),
        static_cast<int>(h),
        static_cast<int>(num_query_heads_capacity),
        static_cast<int>(tiles),
        static_cast<int>(logical_k_capacity));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    reduce_log_f_pre_r2_tiled_denom_finalize_kernel<<<
        static_cast<unsigned int>(denom_finalize_blocks),
        kLogFPreR2TiledThreads,
        0,
        stream>>>(
        req_meta_i32.data_ptr<int32_t>(),
        req_meta_i64.data_ptr<int64_t>(),
        req_meta_i32.stride(0),
        req_meta_i32.stride(1),
        req_meta_i64.stride(0),
        req_meta_i64.stride(1),
        row_lse,
        row_has,
        token_partial_max,
        token_partial_sum,
        token_has,
        static_cast<int>(n),
        static_cast<int>(h),
        static_cast<int>(num_query_heads_capacity),
        static_cast<int>(tiles),
        static_cast<int>(logical_k_capacity));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
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
    TORCH_CHECK(
        req_meta_i32.get_device() == req_meta_i64.get_device(),
        "generic log_f metadata tensors must share one CUDA device");
    const c10::cuda::CUDAGuard device_guard(req_meta_i32.device());
    TORCH_CHECK(req_meta_i32.scalar_type() == torch::kInt32, "req_meta_i32 must be int32");
    TORCH_CHECK(req_meta_i64.scalar_type() == torch::kInt64, "req_meta_i64 must be int64");
    // [ACCUM-META-WIDTH 2026-07-11 EXT审计高危#2] flags bit4 (accum) rows read
    // meta cols 8/9 (prev_rows/prev_capacity) inside the kernel; a >=8 contract
    // admits an 8-col tensor whose bit4 rows would read past the row end
    // (silent OOB -> wrong merge math). The kernel cannot see the column count,
    // so the host contract must cover the widest field the kernel may touch:
    // require >=10 unconditionally (production gt1 meta is always 10 cols).
    TORCH_CHECK(req_meta_i32.dim() == 2 && req_meta_i32.size(1) >= 10, "req_meta_i32 must be [N,>=10]");
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
    TORCH_CHECK(
        req_meta_i32.get_device() == req_meta_i64.get_device(),
        "lastn1 log_f metadata tensors must share one CUDA device");
    const c10::cuda::CUDAGuard device_guard(req_meta_i32.device());
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
    TORCH_CHECK(
        scratch_capture_scores.get_device() == out_capture_scores.get_device()
            && scratch_capture_scores.get_device() == out_log_f_denoms.get_device(),
        "scalar lastn1 log_f tensors must share one CUDA device");
    const c10::cuda::CUDAGuard device_guard(scratch_capture_scores.device());
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
    TORCH_CHECK(
        scratch_capture_scores.get_device() == out_capture_scores.get_device()
            && scratch_capture_scores.get_device() == out_log_f_denoms.get_device(),
        "scalar reduce log_f tensors must share one CUDA device");
    const c10::cuda::CUDAGuard device_guard(scratch_capture_scores.device());
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
        loaded_module = load_inline(
            name=SELECTOR_LOG_S_EXTENSION_NAME,
            cpp_sources=cpp_source,
            cuda_sources=cuda_source,
            functions=None,
            extra_cuda_cflags=extra_cuda_cflags,
            verbose=False,
        )
        contract_error = _module_contract_error(loaded_module)
        if contract_error is not None:
            raise RuntimeError(
                "compiled selector_log_s_ext violates its semantic contract: "
                + contract_error
            )
        _MODULE = loaded_module
        _VALIDATED_MODULE = loaded_module
    except Exception as exc:
        _VALIDATED_MODULE = None
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
    # [TRITON-RETIRE 2026-07-12] 7 列 legacy 桥（升格 col1←last_n*pad/col7←pad
    # 的 8 列适配）已删：生产 pack 恒 10 列，[N,>=10] 合同下桥产物必被拒；
    # 窄 meta 一律由 C++ TORCH_CHECK fail-fast。
    mod.reduce_log_f_pre_scratch(
        req_meta_i32,
        req_meta_i64,
        int(num_seqs),
        int(num_query_heads),
        bool(scratch_in_fp16),
        bool(log_f_out_fp32),
        float(alpha),
    )


def reduce_log_f_pre_scratch_r2_alpha0p5_fp16_resident_cuda(
    *,
    req_meta_i32: torch.Tensor,
    req_meta_i64: torch.Tensor,
    num_seqs: int,
    num_query_heads: int,
    logical_k_bucket: int,
) -> None:
    """Launch the shared SM80 R=2 resident K-bucket owner.

    The bucket is only a compile-time loop ceiling selected from CPU-authored
    metadata.  Effective K, N, H and physical strides remain in the staged
    contract, which the device revalidates before touching output.
    """
    mod = _require_ext(force=True)
    mod.reduce_log_f_pre_scratch_r2_alpha0p5_fp16_resident(
        req_meta_i32,
        req_meta_i64,
        int(num_seqs),
        int(num_query_heads),
        int(logical_k_bucket),
    )


def reduce_log_f_pre_scratch_r2_alpha0p5_fp16_tiled_cuda(
    *,
    req_meta_i32: torch.Tensor,
    req_meta_i64: torch.Tensor,
    workspace: torch.Tensor,
    num_seqs: int,
    num_query_heads: int,
    num_seqs_capacity: int,
    num_query_heads_capacity: int,
    logical_k_capacity: int,
) -> None:
    """Launch one allocation-free four-kernel cohort on the current stream."""

    reasons = log_f_r2_tiled_contract_reasons(
        req_meta_i32=req_meta_i32,
        req_meta_i64=req_meta_i64,
        workspace=workspace,
        num_seqs=num_seqs,
        num_query_heads=num_query_heads,
        num_seqs_capacity=num_seqs_capacity,
        num_query_heads_capacity=num_query_heads_capacity,
        logical_k_capacity=logical_k_capacity,
    )
    if reasons:
        raise ValueError(
            "dynamic tiled R2 selector contract rejected: " + ",".join(reasons)
        )
    mod = _require_ext(force=True)
    mod.reduce_log_f_pre_scratch_r2_alpha0p5_fp16_tiled(
        req_meta_i32,
        req_meta_i64,
        workspace,
        int(num_seqs),
        int(num_query_heads),
        int(num_seqs_capacity),
        int(num_query_heads_capacity),
        int(logical_k_capacity),
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
    "LOG_F_R2_TILED_ROWS",
    "LOG_F_R2_TILED_TILE_K",
    "SelectorLogSExtUnavailable",
    "allocate_log_f_r2_tiled_workspace",
    "fused_log_f_prior_logits_flat",
    "fused_log_f_prior_pre_denom_flat",
    "reduce_log_f_pre_scratch_cuda",
    "reduce_log_f_pre_scratch_r2_alpha0p5_fp16_resident_cuda",
    "reduce_log_f_pre_scratch_r2_alpha0p5_fp16_tiled_cuda",
    "log_f_r2_tiled_contract_reasons",
    "log_f_r2_tiled_workspace_layout",
    "log_f_r2_tiled_workspace_nbytes",
    "copy_log_f_lastn1_scratch_cuda",
    "copy_log_f_lastn1_scratch_scalar_cuda",
    "reduce_log_f_pre_scratch_scalar_cuda",
]
