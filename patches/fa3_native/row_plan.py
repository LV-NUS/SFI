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
    authority_rows = int(step_authority.batch_size)
    total_rows = int(batch_size) if batch_size is not None else authority_rows
    if total_rows < 0:
        raise ValueError("batch_size must be non-negative")
    if total_rows != authority_rows:
        raise ValueError(
            "mixed-page row plan requires exact StepAuthority batch coverage: "
            f"requested={total_rows} authority={authority_rows}"
        )
    is_prefill_src = step_authority.is_prefill_by_row
    if len(is_prefill_src) != total_rows:
        raise ValueError("is_prefill_by_row must exactly match batch_size")

    use_compact_src = step_authority.use_compact_by_row
    producer_src = step_authority.dispatch_logf_producer_by_row
    last_n_src = step_authority.logits_last_n_by_row

    if len(use_compact_src) != total_rows:
        raise ValueError("use_compact_by_row must exactly match batch_size")
    if len(producer_src) != total_rows:
        raise ValueError(
            "dispatch_logf_producer_by_row must exactly match batch_size"
        )
    if len(last_n_src) != total_rows:
        raise ValueError("logits_last_n_by_row must exactly match batch_size")

    producer_bools_cpu = tuple(
        int(v) == int(_LOGF_PRODUCER_ATTN)
        for v in producer_src[:total_rows]
    )
    last_n_ints_cpu = tuple(
        max(0, int(v)) for v in last_n_src[:total_rows]
    )
    request_selected_rows = torch.tensor(
        use_compact_src[:total_rows], device=device, dtype=torch.bool
    )
    producer_mask = torch.tensor(
        producer_bools_cpu, device=device, dtype=torch.bool
    )
    row_capture_last_n_i32 = torch.tensor(
        last_n_ints_cpu, device=device, dtype=torch.int32
    )
    row_is_prefill = torch.tensor(
        is_prefill_src[:total_rows],
        device=device,
        dtype=torch.bool,
    )

    row_is_capture_producer = producer_mask
    # Validate from the CPU tuples already materialized above. Batch aggregates
    # come from StepAuthority, avoiding repeat scans and all device syncs.
    for _idx in range(total_rows):
        if producer_bools_cpu[_idx] and last_n_ints_cpu[_idx] <= 0:
            raise ValueError("capture producer rows must have last_n >= 1")
    has_selected_consume_cpu = bool(step_authority.has_compact_row)
    has_capture_cpu = bool(step_authority.hint_has_log_f)

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
