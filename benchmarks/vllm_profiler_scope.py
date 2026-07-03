from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Callable


def _env_flag(name: str, *, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def build_vllm_torch_profiler_config(engine_args_accepts: Callable[[str], bool]):
    profile_dir = os.environ.get("VLLM_TORCH_PROFILER_DIR", "")
    if not profile_dir or not engine_args_accepts("profiler_config"):
        return None
    try:
        from vllm.config import ProfilerConfig  # pylint: disable=import-error
    except Exception as exc:
        print(f"[profiler] ProfilerConfig unavailable: {exc}", file=sys.stderr, flush=True)
        return None
    # 必须配 schedule(warmup>=1):vLLM worker 每步无条件 profiler.step(),
    # 无 schedule 的 torch profiler 被 step() 推进会每步清空事件
    # ("Profiler clears events at the end of each cycle"),stop 时零事件不落 trace。
    # active 段结束时 tensorboard handler 自动导出,不依赖 stop 时机。
    warmup_iters = int(os.environ.get("VLLM_TORCH_PROFILER_WARMUP_ITERS", "1") or 1)
    active_iters = int(os.environ.get("VLLM_TORCH_PROFILER_ACTIVE_ITERS", "160") or 160)
    return ProfilerConfig(
        profiler="torch",
        torch_profiler_dir=str(profile_dir),
        torch_profiler_with_stack=_env_flag("VLLM_TORCH_PROFILER_WITH_STACK"),
        torch_profiler_use_gzip=False,
        warmup_iterations=max(1, warmup_iters),
        active_iterations=max(1, active_iters),
    )


def maybe_start_vllm_torch_profile(engine) -> bool:
    profile_dir = os.environ.get("VLLM_TORCH_PROFILER_DIR", "")
    if not profile_dir:
        return False
    try:
        Path(profile_dir).mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    try:
        engine.start_profile()
        return True
    except Exception as exc:
        print(f"[profiler] start_profile skipped: {exc}", file=sys.stderr, flush=True)
        return False


def maybe_stop_vllm_torch_profile(engine, active: bool) -> None:
    if not active:
        return
    try:
        engine.stop_profile()
    except Exception as exc:
        print(f"[profiler] stop_profile skipped: {exc}", file=sys.stderr, flush=True)
        return
    time.sleep(5)
