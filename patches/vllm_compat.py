"""Compatibility helpers for vLLM internal namespace moves."""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any


TRITON_UNIFIED_ATTENTION_MODULE_NAMES = (
    "vllm.v1.attention.ops.triton_unified_attention",
    "vllm.attention.ops.triton_unified_attention",
)
TRITON_UNIFIED_ATTENTION_MODULE_NAME = TRITON_UNIFIED_ATTENTION_MODULE_NAMES[0]
FA_UTILS_MODULE_NAME = "vllm.v1.attention.backends.fa_utils"
FLASH_ATTN_BACKEND_MODULE_NAME = "vllm.v1.attention.backends.flash_attn"

FLASH_ATTN_PROBE_MODULE_NAMES = (
    FA_UTILS_MODULE_NAME,
    FLASH_ATTN_BACKEND_MODULE_NAME,
)


def _import_first_available(module_names: tuple[str, ...]) -> ModuleType:
    last_error: ModuleNotFoundError | None = None
    for module_name in module_names:
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            last_error = exc
    assert last_error is not None
    raise last_error


def import_triton_unified_attention_module() -> ModuleType:
    return _import_first_available(TRITON_UNIFIED_ATTENTION_MODULE_NAMES)


def import_fa_utils_module() -> ModuleType:
    return importlib.import_module(FA_UTILS_MODULE_NAME)


