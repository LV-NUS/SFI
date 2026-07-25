"""PHASE-2B de-legacy: real-cudagraph-mode gate for compact sparse decode.

Pure stdlib (CPU-importable, no torch/triton) so the gate logic is unit-testable.
Replaces VLLM_SPARSE_ATTENTION_IN_CUDAGRAPH: compact engages when the real vLLM
cudagraph mode is FULL / FULL_AND_PIECEWISE (decode runs in a FULL cudagraph),
latched run-level onto the controller as `_sparse_attention_in_cudagraph`.
"""
from __future__ import annotations


def dummy_context_cudagraph_is_full(ctx: object | None) -> bool:
    """True iff a dummy-run context reports a FULL-class cudagraph mode."""
    if not isinstance(ctx, dict):
        return False
    runtime_mode = ctx.get("cudagraph_runtime_mode", None)
    return str(getattr(runtime_mode, "name", runtime_mode)) in {
        "FULL",
        "FULL_AND_PIECEWISE",
    }


def attention_in_cudagraph_enabled(controller: object | None) -> bool:
    """De-legacy gate: read the real-cudagraph-mode latch, not the env var."""
    return bool(getattr(controller, "_sparse_attention_in_cudagraph", False))
