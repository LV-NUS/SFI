from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any


def build_observed_prepare_step_context_impl(
    core_impl: Callable[..., Any],
    *,
    route_counter_enabled: bool,
    step_trace_enabled: bool,
) -> Callable[..., Any]:
    """Build the process-lifetime observed specialization.

    The production loader returns ``core_impl`` directly when both observers
    are disabled.  This wrapper therefore exists only in proof/diagnostic
    processes and never leaves a no-op branch in the trace-off hot path.
    """
    if not route_counter_enabled and not step_trace_enabled:
        return core_impl

    @wraps(core_impl)
    def observed_prepare_step_context_impl(self: object, **kwargs: Any) -> Any:
        ctx = core_impl(self, **kwargs)
        if ctx is None:
            return None

        step_authority = getattr(self, "step_authority", None)
        if step_authority is None:
            raise RuntimeError(
                "E_STEP_OBSERVER_AUTHORITY_MISSING: core returned a context "
                "without publishing StepAuthority"
            )

        if route_counter_enabled:
            use_compact_by_row = tuple(
                bool(value)
                for value in getattr(step_authority, "use_compact_by_row", tuple())
            )
            compact_rows = sum(use_compact_by_row)
            if compact_rows > 0:
                from patches.fa3_native.install import (
                    bump_step_compact_row_liveness,
                )

                bump_step_compact_row_liveness(compact_rows)

        if step_trace_enabled:
            from patches.fa3_native.install import (
                append_fa3_step_trace,
                build_fa3_step_trace_event,
            )

            append_fa3_step_trace(
                build_fa3_step_trace_event(
                    step_authority=step_authority,
                    step_context=ctx,
                    source="prepare_step_context",
                )
            )
        return ctx

    return observed_prepare_step_context_impl
