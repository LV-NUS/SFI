from __future__ import annotations

from dataclasses import dataclass

import torch

from patches.sparse_constants import _LOGF_PRODUCER_ATTN


@dataclass(frozen=True, slots=True)
class MixedPageRowPlan:
    request_selected_rows: torch.Tensor
    row_is_capture_producer: torch.Tensor
    row_capture_last_n_i32: torch.Tensor
    row_is_prefill: torch.Tensor
    row_is_prefill_producer: torch.Tensor
    has_selected_consume: bool
    has_capture: bool


def build_mixed_page_row_plan(
    step_authority: object,
    *,
    batch_size: int | None = None,
    device: torch.device | str = "cpu",
) -> MixedPageRowPlan:
    is_prefill_src = tuple(bool(v) for v in getattr(step_authority, "is_prefill_by_row"))
    total_rows = int(batch_size) if batch_size is not None else len(is_prefill_src)
    if total_rows < 0:
        raise ValueError("batch_size must be non-negative")
    if len(is_prefill_src) < total_rows:
        raise ValueError("is_prefill_by_row coverage is insufficient for batch_size")

    use_compact_src = tuple(bool(v) for v in getattr(step_authority, "use_compact_by_row"))
    producer_src = tuple(
        int(v) for v in getattr(step_authority, "dispatch_logf_producer_by_row")
    )
    last_n_src = tuple(int(v) for v in getattr(step_authority, "logits_last_n_by_row"))

    if len(use_compact_src) < total_rows:
        raise ValueError("use_compact_by_row coverage is insufficient for batch_size")
    if len(producer_src) < total_rows:
        raise ValueError("dispatch_logf_producer_by_row coverage is insufficient for batch_size")
    if len(last_n_src) < total_rows:
        raise ValueError("logits_last_n_by_row coverage is insufficient for batch_size")

    request_selected_rows = torch.tensor(
        use_compact_src[:total_rows],
        device=device,
        dtype=torch.bool,
    )
    producer_mask = torch.tensor(
        [int(v) == int(_LOGF_PRODUCER_ATTN) for v in producer_src[:total_rows]],
        device=device,
        dtype=torch.bool,
    )
    row_capture_last_n_i32 = torch.tensor(
        [max(0, int(v)) for v in last_n_src[:total_rows]],
        device=device,
        dtype=torch.int32,
    )
    row_is_prefill = torch.tensor(
        is_prefill_src[:total_rows],
        device=device,
        dtype=torch.bool,
    )

    row_is_capture_producer = producer_mask
    # Rev 2 (2026-04-23): validations + has_* bools computed CPU-side from
    # the source lists we already materialized above, avoiding 3 host syncs
    # per forward per layer (~2700 syncs in a 32-token decode).
    producer_bools_cpu = tuple(
        int(v) == int(_LOGF_PRODUCER_ATTN) for v in producer_src[:total_rows]
    )
    last_n_ints_cpu = tuple(max(0, int(v)) for v in last_n_src[:total_rows])
    for _idx in range(total_rows):
        if producer_bools_cpu[_idx] and last_n_ints_cpu[_idx] <= 0:
            raise ValueError("capture producer rows must have last_n >= 1")
    has_selected_consume_cpu = any(bool(v) for v in use_compact_src[:total_rows])
    has_capture_cpu = any(producer_bools_cpu)

    row_capture_last_n_i32 = torch.where(
        row_is_capture_producer,
        row_capture_last_n_i32,
        torch.zeros_like(row_capture_last_n_i32),
    )
    row_is_prefill_producer = row_is_capture_producer & row_is_prefill

    return MixedPageRowPlan(
        request_selected_rows=request_selected_rows,
        row_is_capture_producer=row_is_capture_producer,
        row_capture_last_n_i32=row_capture_last_n_i32,
        row_is_prefill=row_is_prefill,
        row_is_prefill_producer=row_is_prefill_producer,
        has_selected_consume=has_selected_consume_cpu,
        has_capture=has_capture_cpu,
    )
