from __future__ import annotations

from dataclasses import dataclass

import torch


CURRENT_SELECTED_STATIC_CARRIER_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class SelectedStaticCarrier:
    schema_version: int
    selected_static_pages_i32: torch.Tensor
    selected_static_seqused_k_by_head_i32: torch.Tensor


@dataclass(frozen=True)
class FinalLaunchScratch:
    selected_page_table_i32: torch.Tensor
    selected_seqused_k_by_head_i32: torch.Tensor
    applied_recent_epoch_i32: torch.Tensor | None = None
    applied_refresh_generation_i32: torch.Tensor | None = None
    materialize_status_i32: torch.Tensor | None = None
    patch_status_i32: torch.Tensor | None = None


def _normalize_recent_start_tokens(
    recent_start_tokens: int | torch.Tensor,
    *,
    num_kv_heads: int,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(recent_start_tokens, torch.Tensor):
        normalized = recent_start_tokens.to(device=device, dtype=torch.int32).reshape(-1)
        if normalized.numel() == 1:
            return normalized.expand(int(num_kv_heads))
        if normalized.numel() != int(num_kv_heads):
            raise ValueError("recent_start_tokens tensor must match num_kv_heads")
        return normalized
    return torch.full(
        (int(num_kv_heads),),
        int(recent_start_tokens),
        dtype=torch.int32,
        device=device,
    )


def project_selected_token_indices_to_middle_pages(
    *,
    selected_token_indices: torch.Tensor,
    page_size: int,
    sink_page_slots: int,
    recent_start_tokens: int | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if selected_token_indices.dim() != 2:
        raise ValueError("selected_token_indices must be [num_kv_heads, topk]")
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    if sink_page_slots < 0:
        raise ValueError("sink_page_slots must be non-negative")

    num_kv_heads, topk = selected_token_indices.shape
    projected = torch.full_like(selected_token_indices, -1, dtype=torch.int32)
    counts = torch.zeros((int(num_kv_heads),), device=selected_token_indices.device, dtype=torch.int32)
    if int(topk) == 0:
        return projected, counts

    recent_start_tokens_i32 = _normalize_recent_start_tokens(
        recent_start_tokens,
        num_kv_heads=int(num_kv_heads),
        device=selected_token_indices.device,
    )
    recent_start_pages = torch.div(
        recent_start_tokens_i32.to(dtype=torch.int64),
        int(page_size),
        rounding_mode="floor",
    ).to(dtype=torch.int32).unsqueeze(1)

    sentinel = torch.iinfo(torch.int32).max
    logical_pages = torch.full_like(selected_token_indices, sentinel, dtype=torch.int32)
    valid_tokens = selected_token_indices >= 0
    logical_pages[valid_tokens] = torch.div(
        selected_token_indices[valid_tokens].to(dtype=torch.int64),
        int(page_size),
        rounding_mode="floor",
    ).to(dtype=torch.int32)

    in_middle = valid_tokens & (logical_pages >= int(sink_page_slots)) & (logical_pages < recent_start_pages)
    filtered = torch.full_like(logical_pages, sentinel, dtype=torch.int32)
    filtered[in_middle] = logical_pages[in_middle]

    sorted_pages = torch.sort(filtered, dim=1).values
    keep = sorted_pages != sentinel
    if int(topk) > 1:
        keep[:, 1:] &= sorted_pages[:, 1:] != sorted_pages[:, :-1]

    positions = torch.cumsum(keep.to(dtype=torch.int32), dim=1) - 1
    row_indices = torch.arange(
        int(num_kv_heads),
        device=selected_token_indices.device,
        dtype=torch.int64,
    ).unsqueeze(1).expand_as(sorted_pages)
    projected[row_indices[keep], positions[keep].to(dtype=torch.int64)] = sorted_pages[keep]
    counts.copy_(keep.sum(dim=1, dtype=torch.int32))
    return projected, counts


def _materialize_selected_page_ids_cpu_ref(
    selected_idx: torch.Tensor,
    *,
    page_size: int,
) -> torch.Tensor:
    if selected_idx.shape[1] == 0:
        return torch.empty(
            (selected_idx.shape[0], 0),
            dtype=torch.int32,
            device=selected_idx.device,
        )

    result = torch.full(
        (selected_idx.shape[0], selected_idx.shape[1]),
        -1,
        dtype=torch.int32,
        device=selected_idx.device,
    )
    for row_idx in range(int(selected_idx.shape[0])):
        row = selected_idx[row_idx]
        valid = row[row >= 0]
        if valid.numel() == 0:
            continue
        pages = torch.div(
            valid.to(dtype=torch.int64),
            int(page_size),
            rounding_mode="floor",
        ).to(dtype=torch.int32)
        pages = torch.unique(pages, sorted=True)
        if pages.numel() > 0:
            result[row_idx, : pages.numel()] = pages
    return result


def materialize_selected_page_ids(
    selected_idx: torch.Tensor,
    *,
    page_size: int,
) -> torch.Tensor:
    if selected_idx.dim() != 2:
        raise ValueError("selected_idx must be [batch_row, topk]")
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    if selected_idx.is_cuda:
        return materialize_selected_page_ids_cuda(
            selected_idx,
            page_size=page_size,
        )
    return _materialize_selected_page_ids_cpu_ref(
        selected_idx,
        page_size=page_size,
    )


def materialize_selected_page_ids_cuda(*args, **kwargs):
    if not args:
        raise TypeError("selected_idx must be provided")
    selected_idx = args[0]
    page_size = kwargs.get("page_size")
    if page_size is None:
        raise TypeError("page_size must be provided")
    if selected_idx.dim() != 2:
        raise ValueError("selected_idx must be [batch_row, topk]")
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    if not selected_idx.is_cuda:
        raise ValueError("CUDA page materialize path requires CUDA tensor input")

    sentinel = torch.iinfo(torch.int32).max
    pages = torch.full_like(selected_idx, sentinel, dtype=torch.int32)
    valid = selected_idx >= 0
    pages[valid] = torch.div(
        selected_idx[valid].to(dtype=torch.int64),
        int(page_size),
        rounding_mode="floor",
    ).to(dtype=torch.int32)

    sorted_pages = torch.sort(pages, dim=1).values
    if sorted_pages.shape[1] == 0:
        return torch.empty(
            (selected_idx.shape[0], 0),
            dtype=torch.int32,
            device=selected_idx.device,
        )

    keep = sorted_pages != sentinel
    if sorted_pages.shape[1] > 1:
        prev = torch.roll(sorted_pages, shifts=1, dims=1)
        keep[:, 1:] &= sorted_pages[:, 1:] != prev[:, 1:]

    result = torch.full(
        (selected_idx.shape[0], sorted_pages.shape[1]),
        -1,
        dtype=torch.int32,
        device=selected_idx.device,
    )
    positions = torch.cumsum(keep.to(dtype=torch.int32), dim=1) - 1
    row_indices = torch.arange(
        sorted_pages.shape[0],
        device=selected_idx.device,
        dtype=torch.int64,
    ).unsqueeze(1).expand_as(sorted_pages)
    result[row_indices[keep], positions[keep].to(dtype=torch.int64)] = sorted_pages[keep]
    return result


def is_prefix_no_hole(page_ids: torch.Tensor) -> bool:
    if page_ids.dim() != 2:
        raise ValueError("page_ids must be [batch_row, max_pages]")
    for row_idx in range(int(page_ids.shape[0])):
        row = page_ids[row_idx]
        valid = row[row >= 0]
        if valid.numel() <= 1:
            continue
        deltas = valid[1:] - valid[:-1]
        if not torch.all(deltas == 1):
            return False
    return True


def _write_selected_static_carrier(
    page_ids: torch.Tensor,
    *,
    page_size: int,
) -> SelectedStaticCarrier:
    if page_ids.dim() != 2:
        raise ValueError("selected_static_pages_i32 must be [batch * kv_head, width]")
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    counts = (page_ids >= 0).sum(dim=1, dtype=torch.int32)
    return SelectedStaticCarrier(
        schema_version=CURRENT_SELECTED_STATIC_CARRIER_SCHEMA_VERSION,
        selected_static_pages_i32=page_ids.to(dtype=torch.int32),
        selected_static_seqused_k_by_head_i32=counts * int(page_size),
    )


def static_materialize_selected_pages_runtime(
    *,
    page_size: int | None = None,
    selected_static_pages_by_slot_i32: torch.Tensor | None = None,
    selected_static_page_count_by_slot_i32: torch.Tensor | None = None,
    request_slot_rows_i32: torch.Tensor | None = None,
) -> SelectedStaticCarrier:
    required = {
        "selected_static_pages_by_slot_i32": selected_static_pages_by_slot_i32,
        "selected_static_page_count_by_slot_i32": selected_static_page_count_by_slot_i32,
        "request_slot_rows_i32": request_slot_rows_i32,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise TypeError(
            f"slot-materialize path requires: {', '.join(missing)}"
        )

    pages_by_slot_i32 = selected_static_pages_by_slot_i32.to(dtype=torch.int32)
    counts_by_slot_i32 = selected_static_page_count_by_slot_i32.to(
        device=pages_by_slot_i32.device,
        dtype=torch.int32,
    )
    slot_rows_i64 = request_slot_rows_i32.to(
        device=pages_by_slot_i32.device,
        dtype=torch.long,
    ).reshape(-1)
    if pages_by_slot_i32.dim() != 3:
        raise ValueError("selected_static_pages_by_slot_i32 must be [slot, kv_head, width]")
    if counts_by_slot_i32.dim() != 2:
        raise ValueError("selected_static_page_count_by_slot_i32 must be [slot, kv_head]")
    if int(pages_by_slot_i32.shape[0]) != int(counts_by_slot_i32.shape[0]):
        raise ValueError("slot dimension must match between pages and counts")
    if int(pages_by_slot_i32.shape[1]) != int(counts_by_slot_i32.shape[1]):
        raise ValueError("kv_head dimension must match between pages and counts")
    if page_size is None:
        raise TypeError("page_size must be provided for per-head selected static carrier materialize")
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    gathered_pages_i32 = pages_by_slot_i32.index_select(0, slot_rows_i64).contiguous()
    gathered_counts_i32 = counts_by_slot_i32.index_select(0, slot_rows_i64).contiguous()
    batch_size = int(gathered_pages_i32.shape[0])
    num_kv_heads = int(gathered_pages_i32.shape[1])
    width = int(gathered_pages_i32.shape[2])
    return SelectedStaticCarrier(
        schema_version=CURRENT_SELECTED_STATIC_CARRIER_SCHEMA_VERSION,
        selected_static_pages_i32=gathered_pages_i32.reshape(batch_size * int(num_kv_heads), width),
        selected_static_seqused_k_by_head_i32=(
            gathered_counts_i32 * int(page_size)
        ).reshape(batch_size * int(num_kv_heads)),
    )


def _normalize_selected_static_pages(
    *,
    selected_static_pages_i32: torch.Tensor,
    selected_static_seqused_k_by_head_i32: torch.Tensor,
    page_size: int | None = None,
    num_kv_heads: int | None = None,
    consumer_row_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pages_i32 = selected_static_pages_i32.to(dtype=torch.int32)
    seqused_k_by_head_i32 = selected_static_seqused_k_by_head_i32.to(
        device=pages_i32.device,
        dtype=torch.int32,
    )

    if pages_i32.dim() != 2:
        raise ValueError("selected_static_pages_i32 must be [batch * kv_head, width]")
    if seqused_k_by_head_i32.dim() != 1:
        raise ValueError("selected_static_seqused_k_by_head_i32 must be [batch * kv_head]")
    if int(seqused_k_by_head_i32.shape[0]) != int(pages_i32.shape[0]):
        raise ValueError("selected_static_seqused_k_by_head_i32 row count mismatch")
    if num_kv_heads is None:
        raise ValueError("num_kv_heads must be provided for selected static carrier validation")
    if page_size is None:
        raise ValueError("page_size must be provided for selected static carrier validation")
    if int(page_size) <= 0:
        raise ValueError("page_size must be positive")

    total_rows, width = pages_i32.shape
    carrier_heads = int(num_kv_heads)
    if carrier_heads <= 0:
        raise ValueError("num_kv_heads must be positive")
    if int(total_rows) % int(carrier_heads) != 0:
        raise ValueError("selected static carrier row count must be divisible by num_kv_heads")
    batch_size = int(total_rows) // int(carrier_heads)

    pages_i32 = pages_i32.reshape(batch_size, int(carrier_heads), width)
    seqused_k_by_head_i32 = seqused_k_by_head_i32.reshape(batch_size, int(carrier_heads))
    if consumer_row_mask is None:
        validated_rows = torch.ones(
            (batch_size,),
            dtype=torch.bool,
            device=pages_i32.device,
        )
    else:
        validated_rows = consumer_row_mask.to(
            device=pages_i32.device,
            dtype=torch.bool,
        ).reshape(batch_size)
    validated_heads = validated_rows[:, None]

    if bool(torch.any((seqused_k_by_head_i32 < 0) & validated_heads).item()):
        raise ValueError("selected_static_seqused_k_by_head_i32 must be non-negative")

    valid_mask = pages_i32 >= 0
    hole_prefix = torch.cumsum((~valid_mask).to(dtype=torch.int32), dim=2)
    if bool(torch.any(valid_mask & (hole_prefix > 0) & validated_heads[:, :, None]).item()):
        raise ValueError("selected_static_pages_i32 must use a no-hole prefix per kv head")
    counts_i32 = valid_mask.sum(dim=2, dtype=torch.int32)
    expected_seqused_k_by_head_i32 = counts_i32 * int(page_size)
    if bool(
        torch.any(
            seqused_k_by_head_i32.ne(expected_seqused_k_by_head_i32) & validated_heads
        ).item()
    ):
        raise ValueError("selected_static_seqused_k_by_head_i32 must match selected static page prefix")

    return (
        pages_i32.contiguous(),
        seqused_k_by_head_i32.contiguous(),
        counts_i32.contiguous(),
    )


def build_final_launch_scratch_runtime(
    selected_static_carrier: SelectedStaticCarrier,
    *,
    page_size: int | None = None,
    request_block_table_i32: torch.Tensor | None = None,
    sink_page_slots: int | None = None,
    request_recent_first_logical_page_i32: torch.Tensor | None = None,
    request_recent_page_count_i32: torch.Tensor | None = None,
    request_recent_epoch_i32: torch.Tensor | None = None,
    request_refresh_generation_i32: torch.Tensor | None = None,
    request_real_kv_len_i32: torch.Tensor | None = None,
    request_selected_rows: torch.Tensor | None = None,
    num_kv_heads: int | None = None,
    out_selected_page_table_i32: torch.Tensor | None = None,
) -> FinalLaunchScratch:
    if selected_static_carrier.schema_version != CURRENT_SELECTED_STATIC_CARRIER_SCHEMA_VERSION:
        raise ValueError("unsupported selected static carrier schema_version")

    consumer_row_mask = request_selected_rows if request_block_table_i32 is not None else None
    pages_i32, selected_static_seqused_k_by_head_i32, counts_i32 = _normalize_selected_static_pages(
        selected_static_pages_i32=selected_static_carrier.selected_static_pages_i32,
        selected_static_seqused_k_by_head_i32=selected_static_carrier.selected_static_seqused_k_by_head_i32,
        page_size=page_size,
        num_kv_heads=num_kv_heads,
        consumer_row_mask=consumer_row_mask,
    )

    if request_block_table_i32 is None:
        return FinalLaunchScratch(
            selected_page_table_i32=pages_i32.reshape(-1, pages_i32.shape[-1]).clone(),
            selected_seqused_k_by_head_i32=selected_static_seqused_k_by_head_i32.reshape(-1).clone(),
        )

    if sink_page_slots is None:
        raise TypeError("sink_page_slots must be provided for runtime final launch scratch")
    required = {
        "request_recent_first_logical_page_i32": request_recent_first_logical_page_i32,
        "request_recent_page_count_i32": request_recent_page_count_i32,
        "request_recent_epoch_i32": request_recent_epoch_i32,
        "request_refresh_generation_i32": request_refresh_generation_i32,
        "request_real_kv_len_i32": request_real_kv_len_i32,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise TypeError(
            f"runtime final launch scratch requires: {', '.join(missing)}"
        )

    device = request_block_table_i32.device
    block_table_i32 = request_block_table_i32.to(device=device, dtype=torch.int32)
    pages_i32 = pages_i32.to(device=device, dtype=torch.int32)
    counts_i32 = counts_i32.to(device=device, dtype=torch.int32)
    batch_size, carrier_heads, carrier_width = pages_i32.shape
    if block_table_i32.dim() != 2 or int(block_table_i32.shape[0]) != int(batch_size):
        raise ValueError("request_block_table_i32 must be [batch, logical_block]")

    request_selected_rows_i1 = (
        (counts_i32.max(dim=1).values > 0)
        if request_selected_rows is None
        else request_selected_rows.to(device=device, dtype=torch.bool).reshape(batch_size)
    )
    recent_first_i32 = request_recent_first_logical_page_i32.to(
        device=device,
        dtype=torch.int32,
    ).reshape(batch_size)
    recent_count_i32 = request_recent_page_count_i32.to(
        device=device,
        dtype=torch.int32,
    ).reshape(batch_size)
    applied_recent_epoch_i32 = request_recent_epoch_i32.to(
        device=device,
        dtype=torch.int32,
    ).reshape(batch_size)
    applied_refresh_generation_i32 = request_refresh_generation_i32.to(
        device=device,
        dtype=torch.int32,
    ).reshape(batch_size)
    real_kv_len_i32 = request_real_kv_len_i32.to(
        device=device,
        dtype=torch.int32,
    ).reshape(batch_size)

    selected_page_count_by_head_i32 = counts_i32 + recent_count_i32[:, None] + int(sink_page_slots)
    selected_page_count_summary_i32 = torch.where(
        request_selected_rows_i1,
        selected_page_count_by_head_i32.max(dim=1).values,
        torch.zeros((batch_size,), dtype=torch.int32, device=device),
    )
    if out_selected_page_table_i32 is None:
        width = int(selected_page_count_summary_i32.max().item()) if int(batch_size) > 0 else 0
        final_page_table_i32 = torch.full(
            (int(batch_size) * int(carrier_heads), int(width)),
            -1,
            dtype=torch.int32,
            device=device,
        )
    else:
        final_page_table_i32 = out_selected_page_table_i32.to(device=device, dtype=torch.int32)
        width = int(final_page_table_i32.shape[1])
        final_page_table_i32.fill_(-1)

    page_table_i32 = final_page_table_i32.reshape(int(batch_size), int(carrier_heads), int(width))
    logical_block_count = int(block_table_i32.shape[1])
    if int(sink_page_slots) > 0 and bool(torch.any(request_selected_rows_i1).item()):
        sink_pages_i32 = block_table_i32[:, : int(sink_page_slots)]
        page_table_i32[request_selected_rows_i1, :, : int(sink_page_slots)] = (
            sink_pages_i32[request_selected_rows_i1]
            .unsqueeze(1)
            .expand(int(request_selected_rows_i1.sum().item()), int(carrier_heads), int(sink_page_slots))
        )

    if int(carrier_width) > 0 and int(width) > int(sink_page_slots):
        middle_prefix_mask = request_selected_rows_i1[:, None, None] & (
            torch.arange(
                int(carrier_width),
                device=device,
                dtype=torch.int32,
            ).view(1, 1, int(carrier_width))
            < counts_i32[:, :, None]
        )
        if bool(torch.any((pages_i32 < 0) & middle_prefix_mask).item()):
            raise ValueError("selected static carrier page ids must be non-negative within selected prefix")
        if bool(torch.any((pages_i32 >= logical_block_count) & middle_prefix_mask).item()):
            raise ValueError("selected static carrier page ids must fit request_block_table_i32")
        safe_pages_i64 = torch.where(
            middle_prefix_mask,
            pages_i32,
            torch.zeros_like(pages_i32),
        ).to(dtype=torch.long)
        block_table_by_head_i32 = block_table_i32.unsqueeze(1).expand(
            int(batch_size),
            int(carrier_heads),
            logical_block_count,
        )
        middle_physical_i32 = torch.gather(
            block_table_by_head_i32,
            2,
            safe_pages_i64,
        ).to(dtype=torch.int32)
        middle_width = min(int(carrier_width), int(width) - int(sink_page_slots))
        middle_col_i32 = torch.arange(middle_width, device=device, dtype=torch.int32).view(1, 1, middle_width)
        middle_mask = request_selected_rows_i1[:, None, None] & (
            middle_col_i32 < counts_i32[:, :, None]
        )
        page_table_i32[:, :, int(sink_page_slots) : int(sink_page_slots) + int(middle_width)] = torch.where(
            middle_mask,
            middle_physical_i32[:, :, : int(middle_width)],
            page_table_i32[:, :, int(sink_page_slots) : int(sink_page_slots) + int(middle_width)],
        )

    max_recent_pages = int(recent_count_i32.max().item()) if int(batch_size) > 0 else 0
    if max_recent_pages > 0 and int(width) > 0 and bool(torch.any(request_selected_rows_i1).item()):
        recent_col_i32 = torch.arange(max_recent_pages, device=device, dtype=torch.int32).view(1, max_recent_pages)
        recent_logical_i32 = (
            recent_first_i32[:, None] + recent_col_i32
        ).to(dtype=torch.int32)
        selected_recent_mask_2d = request_selected_rows_i1[:, None] & (
            recent_col_i32 < recent_count_i32[:, None]
        )
        if bool(torch.any((recent_logical_i32 < 0) & selected_recent_mask_2d).item()):
            raise ValueError("selected recent logical pages must be non-negative")
        if bool(torch.any((recent_logical_i32 >= logical_block_count) & selected_recent_mask_2d).item()):
            raise ValueError("selected recent logical pages must fit request_block_table_i32")
        recent_logical_i64 = torch.where(
            selected_recent_mask_2d,
            recent_logical_i32,
            torch.zeros_like(recent_logical_i32),
        ).to(dtype=torch.long)
        recent_physical_i32 = torch.gather(
            block_table_i32,
            1,
            recent_logical_i64,
        ).to(dtype=torch.int32)
        recent_pos_i32 = int(sink_page_slots) + counts_i32[:, :, None] + recent_col_i32[:, None, :]
        recent_mask = request_selected_rows_i1[:, None, None] & (
            recent_col_i32[:, None, :] < recent_count_i32[:, None, None]
        ) & (recent_pos_i32 < int(width))
        batch_row_i64 = torch.arange(int(batch_size), device=device, dtype=torch.long).view(batch_size, 1, 1)
        kv_head_i64 = torch.arange(int(carrier_heads), device=device, dtype=torch.long).view(1, carrier_heads, 1)
        batch_row_i64 = batch_row_i64.expand(int(batch_size), int(carrier_heads), max_recent_pages)
        kv_head_i64 = kv_head_i64.expand(int(batch_size), int(carrier_heads), max_recent_pages)
        recent_values_i32 = recent_physical_i32[:, None, :].expand(
            int(batch_size),
            int(carrier_heads),
            max_recent_pages,
        )
        page_table_i32[
            batch_row_i64[recent_mask],
            kv_head_i64[recent_mask],
            recent_pos_i32.to(dtype=torch.long)[recent_mask],
        ] = recent_values_i32[recent_mask]

    recent_tail_k_i32 = torch.clamp(
        real_kv_len_i32 - recent_first_i32 * int(page_size),
        min=0,
    )
    selected_seqused_k_by_head_i32 = torch.where(
        request_selected_rows_i1[:, None],
        counts_i32 * int(page_size) + int(sink_page_slots) * int(page_size) + recent_tail_k_i32[:, None],
        torch.zeros((batch_size, int(carrier_heads)), dtype=torch.int32, device=device),
    )

    return FinalLaunchScratch(
        selected_page_table_i32=final_page_table_i32,
        selected_seqused_k_by_head_i32=selected_seqused_k_by_head_i32.reshape(-1).clone(),
        applied_recent_epoch_i32=applied_recent_epoch_i32.clone(),
        applied_refresh_generation_i32=applied_refresh_generation_i32.clone(),
        materialize_status_i32=torch.zeros((batch_size,), dtype=torch.int32, device=device),
        patch_status_i32=torch.zeros((batch_size,), dtype=torch.int32, device=device),
    )
