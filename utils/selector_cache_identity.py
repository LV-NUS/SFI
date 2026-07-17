#!/usr/bin/env python3
"""Derive a stable selector extension cache partition from Python/Torch ABI."""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sysconfig
from functools import lru_cache
from pathlib import Path

if __package__:
    from utils.ext_toolchain import configure_cuda_toolchain_or_raise
else:
    from ext_toolchain import configure_cuda_toolchain_or_raise


_ABI_KEY_RE = re.compile(r"^[0-9a-f]{16}$")


def current_selector_cache_abi_key() -> str:
    """Return the ABI key for the interpreter executing this function."""
    import torch

    toolchain = configure_cuda_toolchain_or_raise(
        owner="selector extension cache identity"
    )
    identity = "|".join(
        (
            str(sysconfig.get_config_var("SOABI") or "unknown"),
            str(torch.__version__),
            str(torch.version.cuda or "cpu"),
            str(getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", "unknown")),
            toolchain.nvcc_path,
            toolchain.release,
        )
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]


@lru_cache(maxsize=None)
def selector_cache_abi_key_for_python(
    python_executable: str,
    *,
    timeout_s: int = 30,
) -> str:
    """Query the ABI key from the exact interpreter that will load the .so."""
    python_path = Path(python_executable).expanduser()
    if (
        not python_path.is_absolute()
        or not python_path.is_file()
        or not os.access(python_path, os.X_OK)
    ):
        raise ValueError(
            f"python executable must be an executable absolute path: {python_path}"
        )
    result = subprocess.run(
        [str(python_path), str(Path(__file__).resolve())],
        text=True,
        capture_output=True,
        check=False,
        timeout=max(1, int(timeout_s)),
    )
    key = result.stdout.strip()
    if result.returncode != 0 or _ABI_KEY_RE.fullmatch(key) is None:
        detail = result.stderr.strip() or result.stdout.strip() or "no output"
        raise RuntimeError(
            "failed to derive selector cache ABI identity from "
            f"{python_path}: returncode={result.returncode}: {detail}"
        )
    return key


def main() -> int:
    print(current_selector_cache_abi_key())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
