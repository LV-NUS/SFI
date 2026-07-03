"""
`python benchmarks/...` sets `sys.path[0]` to the `benchmarks/` directory.

Some newer `transformers` releases already include model-type registrations
that the vendored vLLM version still tries to register again (e.g. "aimv2"),
which raises at import time and prevents running any benchmark scripts.

This lightweight `sitecustomize` is auto-imported by Python on startup
and applies a small compatibility shim for the benchmark entrypoints.
"""

import os

if os.environ.get("VLLM_IGNORE_DUPLICATE_TRANSFORMERS_CONFIGS", "1") == "1":
    try:
        from transformers import AutoConfig  # type: ignore

        _orig_register = AutoConfig.register

        def _safe_register(model_type, config, exist_ok: bool = False):  # type: ignore[no-redef]
            try:
                return _orig_register(model_type, config, exist_ok=True)
            except ValueError:
                return None

        AutoConfig.register = _safe_register  # type: ignore[assignment]
    except Exception:
        pass

