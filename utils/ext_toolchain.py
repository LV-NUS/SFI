"""[EXT-NVCC-GUARD 2026-07-09] JIT 工具链统一守护(安装坑根修)。

已知坑(坑录 2026-07-08 §5.1):删 tmp/torch_extensions 构建缓存后,bench
child 现场 JIT 会沿 PATH 摸到系统 /usr/bin/nvcc(10.1)→ 编译秒死,报错被
包进 "root_cause=<CalledProcessError>" 难以定位。此前仅 selector_key_norms_ext
/ selector_log_s_ext 各自内置了 nvcc 定位守护,其余 load_inline 站点裸奔。

本模块把守护统一成一个调用:JIT 触发前 pin 可用 nvcc(优先 PYTORCH_NVCC/
CUDACXX 显式指定 → python 同目录 → CUDA_HOME/CUDA_PATH → /usr/local/cuda*
→ PATH),并做版本预检——nvcc 主版本 <11 或找不到 nvcc 直接抛可操作
RuntimeError(fail-fast,不留给 torch 报难懂错)。只在 JIT 回落路径调用
(prebuilt 命中不经过此门,无 nvcc 的纯 prebuilt 环境不受影响)。
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from typing import Optional


def preferred_nvcc_path() -> Optional[str]:
    explicit_nvcc = os.environ.get("PYTORCH_NVCC") or os.environ.get("CUDACXX")
    if explicit_nvcc:
        return explicit_nvcc

    candidates = [os.path.join(os.path.dirname(sys.executable), "nvcc")]
    for cuda_home_var in ("CUDA_HOME", "CUDA_PATH"):
        cuda_home = os.environ.get(cuda_home_var)
        if cuda_home:
            candidates.append(os.path.join(cuda_home, "bin", "nvcc"))
    # 本机工具链锚=12.4(/usr/local/cuda 软链亦指 12.4;生产脚本恒
    # CUDA_HOME=/usr/local/cuda-12.4)。12.4 排在更高版本之前:CUDA_HOME
    # 缺席时也钉住项目锚版本,不被偶然装上的新 toolkit 抢先。
    candidates.extend(
        [
            "/usr/local/cuda/bin/nvcc",
            "/usr/local/cuda-12.4/bin/nvcc",
            "/usr/local/cuda-12.6/bin/nvcc",
            "/usr/local/cuda-12.5/bin/nvcc",
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


def _nvcc_major_version(nvcc: str) -> Optional[int]:
    try:
        completed = subprocess.run(
            [nvcc, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    text = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
    match = re.search(r"release\s+(\d+)\.(\d+)", text)
    if not match:
        return None
    return int(match.group(1))


def configure_jit_toolchain_or_raise(*, ext_name: str) -> str:
    """JIT 回落路径的工具链门:pin nvcc + 版本预检,失败即抛可操作错误。

    返回选定的 nvcc 路径(诊断用)。副作用:设 PYTORCH_NVCC/CUDA_HOME/
    CUDA_PATH 并同步 torch.utils.cpp_extension.CUDA_HOME,令本进程内所有
    后续 JIT 一致走同一工具链。
    """
    nvcc = preferred_nvcc_path()
    if nvcc is None:
        raise RuntimeError(
            f"{ext_name}: no usable nvcc found for JIT build. Set "
            "CUDA_HOME=/usr/local/cuda-12.x (or PYTORCH_NVCC=<path-to-nvcc>) "
            "before launching, or restore the prebuilt extension cache under "
            "tmp/torch_extensions/."
        )
    major = _nvcc_major_version(nvcc)
    if major is not None and major < 11:
        raise RuntimeError(
            f"{ext_name}: refusing JIT build with ancient nvcc {nvcc} "
            f"(major={major}; the system /usr/bin/nvcc 10.x pit). Set "
            "CUDA_HOME=/usr/local/cuda-12.x or PYTORCH_NVCC to a CUDA>=11 "
            "toolchain."
        )
    cuda_home = os.path.dirname(os.path.dirname(nvcc))
    os.environ["PYTORCH_NVCC"] = nvcc
    os.environ["CUDA_HOME"] = cuda_home
    os.environ["CUDA_PATH"] = cuda_home
    try:
        import torch.utils.cpp_extension as torch_cpp_extension

        torch_cpp_extension.CUDA_HOME = cuda_home
    except Exception:
        pass
    return nvcc
