from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Optional


def load_prebuilt_extension(name: str) -> Optional[ModuleType]:
    """Load an already-built torch extension without invoking ninja.

    ``torch.utils.cpp_extension.load_inline`` may still enter its build lock in
    fresh subprocesses even when a usable .so is present.  Hot benchmark paths
    should consume the cached binary directly and only fall back to load_inline
    when the binary does not exist or cannot be imported.
    """
    try:
        from torch.utils.cpp_extension import _get_build_directory

        build_dir = Path(_get_build_directory(str(name), verbose=False))
    except Exception:
        return None

    so_path = build_dir / f"{name}.so"
    try:
        if not so_path.is_file() or int(so_path.stat().st_size) <= 0:
            return None
    except OSError:
        return None

    existing = sys.modules.get(str(name))
    if isinstance(existing, ModuleType):
        return existing

    try:
        spec = importlib.util.spec_from_file_location(str(name), so_path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        sys.modules[str(name)] = module
        spec.loader.exec_module(module)
        return module
    except Exception:
        sys.modules.pop(str(name), None)
        return None
