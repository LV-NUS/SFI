from __future__ import annotations

from dataclasses import dataclass

import torch

from patches.fa3_native.page_materialize import (
    project_selected_token_indices_to_middle_pages,
)

from .contracts import RequestRecentDescriptor


@dataclass(frozen=True, slots=True)
class PageAlignedRecentWindow:
    start_token: int
    first_logical_page: int
    page_count: int
    visible_tokens: int


def derive_page_aligned_recent_window(
    *,
    real_kv_len: int,
    page_size: int,
    recent_tokens: int,
    min_start_token: int = 0,
) -> PageAlignedRecentWindow:
    if int(page_size) <= 0:
        raise ValueError("page_size must be positive")
    real_i = max(0, int(real_kv_len))
    recent_i = max(0, int(recent_tokens))
    min_start_i = max(0, int(min_start_token))
    if real_i <= 0 or recent_i <= 0:
        return PageAlignedRecentWindow(
            start_token=real_i,
            first_logical_page=(real_i + int(page_size) - 1) // int(page_size),
            page_count=0,
            visible_tokens=0,
        )
    start_floor = max(min_start_i, real_i - recent_i)
    start_token = (start_floor // int(page_size)) * int(page_size)
    start_token = min(start_token, real_i)
    first_page = start_token // int(page_size)
    end_page = (real_i + int(page_size) - 1) // int(page_size)
    page_count = max(0, end_page - first_page)
    visible_tokens = max(0, real_i - start_token)
    return PageAlignedRecentWindow(
        start_token=int(start_token),
        first_logical_page=int(first_page),
        page_count=int(page_count),
        visible_tokens=int(visible_tokens),
    )


def build_materialized_recent_descriptor(
    *,
    real_kv_len: int,
    page_size: int,
    sink_page_slots: int,
    recent_page_slots: int,
    recent_tokens: int,
    epoch: int = -1,
) -> RequestRecentDescriptor:
    if real_kv_len <= 0:
        raise ValueError("real_kv_len must be positive")
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    if sink_page_slots < 0 or recent_page_slots < 0 or recent_tokens < 0:
        raise ValueError("page slot counts and recent_tokens must be non-negative")
    if recent_page_slots <= 0:
        raise ValueError("recent_page_slots must be positive")

    recent_window = derive_page_aligned_recent_window(
        real_kv_len=int(real_kv_len),
        page_size=int(page_size),
        recent_tokens=int(recent_tokens),
        min_start_token=int(sink_page_slots) * int(page_size),
    )
    active_recent_first_logical_page = int(recent_window.first_logical_page)
    active_recent_last_logical_page = (int(real_kv_len) - 1) // int(page_size)
    # The formal recent patch must keep the leading partial page whenever
    # recent_tokens starts inside a page; otherwise sink + middle + recent
    # would create a logical hole at the recent boundary.
    materialized_first_logical_page = int(active_recent_first_logical_page)
    materialized_page_count = int(active_recent_last_logical_page) - int(materialized_first_logical_page) + 1
    if materialized_page_count <= 0:
        raise ValueError("materialized recent pages must be non-empty")

    return RequestRecentDescriptor(
        real_kv_len=int(real_kv_len),
        page_size=int(page_size),
        sink_page_slots=int(sink_page_slots),
        recent_page_slots=int(recent_page_slots),
        active_recent_first_logical_page=int(active_recent_first_logical_page),
        active_recent_last_logical_page=int(active_recent_last_logical_page),
        materialized_first_logical_page=int(materialized_first_logical_page),
        materialized_page_count=int(materialized_page_count),
        epoch=int(epoch),
    )






def compute_visible_kv_len(
    *,
    page_size: int,
    sink_page_slots: int,
    middle_page_slots: int,
    recent_descriptor: RequestRecentDescriptor,
) -> int:
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    if sink_page_slots < 0 or middle_page_slots < 0:
        raise ValueError("page slots must be non-negative")
    return (
        int(sink_page_slots) * int(page_size)
        + int(middle_page_slots) * int(page_size)
        + max(
            0,
            int(recent_descriptor.real_kv_len)
            - int(recent_descriptor.materialized_first_logical_page) * int(page_size),
        )
    )


def project_selected_token_indices_to_logical_pages(
    *,
    selected_token_indices: torch.Tensor,
    block_size: int,
    sink_page_slots: int,
    recent_start_token: int | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(recent_start_token, torch.Tensor):
        if torch.any(recent_start_token < 0):
            raise ValueError("recent_start_token must be non-negative")
    elif int(recent_start_token) < 0:
        raise ValueError("recent_start_token must be non-negative")

    return project_selected_token_indices_to_middle_pages(
        selected_token_indices=selected_token_indices,
        page_size=int(block_size),
        sink_page_slots=int(sink_page_slots),
        recent_start_tokens=recent_start_token,
    )
