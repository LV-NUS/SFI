"""Pure compact-page mixed-page overlay composition helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch

from patches.fa3_native.row_consume_modes import (
    ROW_CONSUME_MODE_COMPACT_RECENT_I32,
    ROW_CONSUME_MODE_FULL_I32,
)


@dataclass(slots=True)
class CompactMixedPageOverlay:
    selected_page_table_i32: torch.Tensor
    effective_row_slot_i32: torch.Tensor
    row_consume_mode_i32: torch.Tensor
    selected_seqused_k_by_head_i32: torch.Tensor
    overlay_width_pages: int
    compact_capacity_pages: int
    recent_capacity_pages: int
    safe_page_id: int
    block_table_dirty_keys: tuple[tuple[int, bool, int, int, int, int, int, int], ...]


@dataclass(slots=True)
class CompactMixedPageOverlayStats:
    page_table_rows_rewritten: tuple[int, ...]
    length_rows_rewritten: tuple[int, ...]


@dataclass(slots=True)
class CompactMixedPageOverlayResult:
    overlay: CompactMixedPageOverlay
    stats: CompactMixedPageOverlayStats


def assert_cross_layer_slot_signature(
    *,
    expected: Iterable[int],
    actual: Iterable[int],
    layer_index: int,
) -> None:
    _reject_tensor_metadata(
        expected=expected,
        actual=actual,
        layer_index=layer_index,
    )
    expected_metadata = _metadata_tuple("expected", expected)
    actual_metadata = _metadata_tuple("actual", actual)
    expected_tuple = tuple(int(v) for v in expected_metadata)
    actual_tuple = tuple(int(v) for v in actual_metadata)
    if expected_tuple != actual_tuple:
        raise RuntimeError(
            "slot_signature mismatch for mixed-page overlay "
            f"at layer {int(layer_index)}: expected={expected_tuple}, actual={actual_tuple}"
        )


def validate_safe_page_id(
    *,
    safe_page_id: int,
    reserved_manager_block_ids: Sequence[int],
) -> int:
    _reject_tensor_metadata(
        safe_page_id=safe_page_id,
        reserved_manager_block_ids=reserved_manager_block_ids,
    )
    safe_i = int(safe_page_id)
    if safe_i < 0:
        raise ValueError("safe_page_id must be non-negative")
    reserved_metadata = _metadata_tuple(
        "reserved_manager_block_ids",
        reserved_manager_block_ids,
    )
    reserved = tuple(int(v) for v in reserved_metadata)
    if safe_i in reserved:
        raise ValueError("safe_page_id must not overlap reserved_manager_block_ids")
    return safe_i


def resolve_overlay_width_pages(
    *,
    max_seqlen_k_hint: int,
    page_size: int,
    canonical_width_pages: int,
    compact_capacity_pages: int,
    recent_capacity_pages: int,
) -> int:
    _reject_tensor_metadata(
        max_seqlen_k_hint=max_seqlen_k_hint,
        page_size=page_size,
        canonical_width_pages=canonical_width_pages,
        compact_capacity_pages=compact_capacity_pages,
        recent_capacity_pages=recent_capacity_pages,
    )
    page_size_i = int(page_size)
    if page_size_i <= 0:
        raise ValueError("page_size must be positive")
    hint_i = max(0, int(max_seqlen_k_hint))
    hint_pages = (hint_i + page_size_i - 1) // page_size_i
    return max(
        int(compact_capacity_pages) + int(recent_capacity_pages),
        hint_pages,
    )


def compose_compact_mixed_page_overlay(
    *,
    canonical_block_table_i32: torch.Tensor,
    max_seqlen_k_hint: int,
    row_is_compact: Iterable[bool],
    slot_by_row: Iterable[int],
    reserved_manager_block_ids: Sequence[int],
    compact_valid_tokens_by_row: Iterable[int],
    recent_first_page_by_row: Iterable[int],
    recent_page_count_by_row: Iterable[int],
    row_effective_k_by_row: Iterable[int],
    page_size: int,
    num_kv_heads: int,
    compact_capacity_pages: int,
    recent_capacity_pages: int,
    safe_page_id: int,
    block_table_epoch: int,
    block_table_storage_key: int,
    slot_signature: Iterable[int],
    previous: CompactMixedPageOverlay | None,
) -> CompactMixedPageOverlayResult:
    _reject_tensor_metadata(
        max_seqlen_k_hint=max_seqlen_k_hint,
        row_is_compact=row_is_compact,
        slot_by_row=slot_by_row,
        reserved_manager_block_ids=reserved_manager_block_ids,
        compact_valid_tokens_by_row=compact_valid_tokens_by_row,
        recent_first_page_by_row=recent_first_page_by_row,
        recent_page_count_by_row=recent_page_count_by_row,
        row_effective_k_by_row=row_effective_k_by_row,
        page_size=page_size,
        num_kv_heads=num_kv_heads,
        compact_capacity_pages=compact_capacity_pages,
        recent_capacity_pages=recent_capacity_pages,
        safe_page_id=safe_page_id,
        block_table_epoch=block_table_epoch,
        block_table_storage_key=block_table_storage_key,
        slot_signature=slot_signature,
    )
    if canonical_block_table_i32.ndim != 2:
        raise ValueError("canonical_block_table_i32 must be 2D")
    canonical = (
        canonical_block_table_i32
        if canonical_block_table_i32.dtype == torch.int32
        else canonical_block_table_i32.to(dtype=torch.int32)
    )
    batch_size = int(canonical.shape[0])
    canonical_width = int(canonical.shape[1])
    compact_capacity_i = int(compact_capacity_pages)
    recent_capacity_i = int(recent_capacity_pages)
    if compact_capacity_i < 0 or recent_capacity_i < 0:
        raise ValueError("compact/recent capacity pages must be non-negative")
    num_kv_heads_i = int(num_kv_heads)
    if num_kv_heads_i <= 0:
        raise ValueError("num_kv_heads must be positive")
    page_size_i = int(page_size)
    if page_size_i <= 0:
        raise ValueError("page_size must be positive")

    row_is_compact_metadata = _metadata_tuple("row_is_compact", row_is_compact)
    slot_by_row_metadata = _metadata_tuple("slot_by_row", slot_by_row)
    reserved_metadata = _metadata_tuple(
        "reserved_manager_block_ids",
        reserved_manager_block_ids,
    )
    compact_valid_metadata = _metadata_tuple(
        "compact_valid_tokens_by_row",
        compact_valid_tokens_by_row,
    )
    recent_first_metadata = _metadata_tuple(
        "recent_first_page_by_row",
        recent_first_page_by_row,
    )
    recent_count_metadata = _metadata_tuple(
        "recent_page_count_by_row",
        recent_page_count_by_row,
    )
    row_effective_k_metadata = _metadata_tuple(
        "row_effective_k_by_row",
        row_effective_k_by_row,
    )
    slot_signature_metadata = _metadata_tuple("slot_signature", slot_signature)

    row_is_compact_tuple = tuple(bool(v) for v in row_is_compact_metadata)
    slot_by_row_tuple = tuple(int(v) for v in slot_by_row_metadata)
    compact_valid_tuple = tuple(int(v) for v in compact_valid_metadata)
    recent_first_tuple = tuple(int(v) for v in recent_first_metadata)
    recent_count_tuple = tuple(int(v) for v in recent_count_metadata)
    row_effective_k_tuple = tuple(int(v) for v in row_effective_k_metadata)
    slot_signature_tuple = tuple(int(v) for v in slot_signature_metadata)
    _validate_batch_vectors(
        batch_size=batch_size,
        row_is_compact=row_is_compact_tuple,
        slot_by_row=slot_by_row_tuple,
        compact_valid_tokens_by_row=compact_valid_tuple,
        recent_first_page_by_row=recent_first_tuple,
        recent_page_count_by_row=recent_count_tuple,
        row_effective_k_by_row=row_effective_k_tuple,
        slot_signature=slot_signature_tuple,
    )

    safe_i = validate_safe_page_id(
        safe_page_id=safe_page_id,
        reserved_manager_block_ids=reserved_manager_block_ids,
    )
    reserved_tuple = tuple(int(v) for v in reserved_metadata)
    overlay_width = resolve_overlay_width_pages(
        max_seqlen_k_hint=max_seqlen_k_hint,
        page_size=page_size_i,
        canonical_width_pages=canonical_width,
        compact_capacity_pages=compact_capacity_i,
        recent_capacity_pages=recent_capacity_i,
    )
    _validate_effective_k_bounds(
        row_effective_k_by_row=row_effective_k_tuple,
        overlay_width_pages=overlay_width,
        page_size=page_size_i,
    )
    dirty_keys = _dirty_keys(
        row_is_compact=row_is_compact_tuple,
        slot_by_row=slot_by_row_tuple,
        compact_valid_tokens_by_row=compact_valid_tuple,
        recent_first_page_by_row=recent_first_tuple,
        recent_page_count_by_row=recent_count_tuple,
        block_table_epoch=int(block_table_epoch),
        block_table_storage_key=int(block_table_storage_key),
    )

    overlay, can_reuse = _overlay_or_new(
        previous=previous,
        batch_size=batch_size,
        overlay_width_pages=overlay_width,
        num_kv_heads=num_kv_heads_i,
        compact_capacity_pages=compact_capacity_i,
        recent_capacity_pages=recent_capacity_i,
        safe_page_id=safe_i,
        device=canonical.device,
    )

    page_rewritten: list[int] = []
    length_rewritten: list[int] = []
    for row in range(batch_size):
        mode = (
            ROW_CONSUME_MODE_COMPACT_RECENT_I32
            if row_is_compact_tuple[row]
            else ROW_CONSUME_MODE_FULL_I32
        )
        overlay.row_consume_mode_i32[row] = mode
        length_start = row * num_kv_heads_i
        length_end = length_start + num_kv_heads_i
        overlay.effective_row_slot_i32[length_start:length_end] = (
            row if mode != ROW_CONSUME_MODE_FULL_I32 else -1
        )
        overlay.selected_seqused_k_by_head_i32[
            length_start:length_end
        ] = row_effective_k_tuple[row]
        length_rewritten.append(row)

        previous_key = _previous_dirty_key(previous, row) if can_reuse else None
        if previous_key == dirty_keys[row]:
            continue
        if mode != ROW_CONSUME_MODE_FULL_I32:
            _rewrite_compact_row(
                overlay.selected_page_table_i32[row],
                canonical[row],
                reserved_manager_block_ids=reserved_tuple,
                slot=slot_by_row_tuple[row],
                compact_valid_tokens=compact_valid_tuple[row],
                recent_first_page=recent_first_tuple[row],
                recent_page_count=recent_count_tuple[row],
                row_effective_k=row_effective_k_tuple[row],
                page_size=page_size_i,
                compact_capacity_pages=compact_capacity_i,
                recent_capacity_pages=recent_capacity_i,
                safe_page_id=safe_i,
                canonical_width_pages=canonical_width,
            )
            page_rewritten.append(row)

    overlay.block_table_dirty_keys = dirty_keys
    return CompactMixedPageOverlayResult(
        overlay=overlay,
        stats=CompactMixedPageOverlayStats(
            page_table_rows_rewritten=tuple(page_rewritten),
            length_rows_rewritten=tuple(length_rewritten),
        ),
    )


def _reject_tensor_metadata(**metadata: object) -> None:
    for name, value in metadata.items():
        if isinstance(value, torch.Tensor):
            raise TypeError(
                f"{name} must be CPU-owned tuple/list metadata, not torch.Tensor"
            )


def _metadata_tuple(name: str, values: Iterable[object]) -> tuple[object, ...]:
    materialized = tuple(values)
    for value in materialized:
        if isinstance(value, torch.Tensor):
            raise TypeError(
                f"{name} must contain CPU-owned metadata values, not torch.Tensor"
            )
    return materialized


def _validate_batch_vectors(
    *,
    batch_size: int,
    row_is_compact: tuple[bool, ...],
    slot_by_row: tuple[int, ...],
    compact_valid_tokens_by_row: tuple[int, ...],
    recent_first_page_by_row: tuple[int, ...],
    recent_page_count_by_row: tuple[int, ...],
    row_effective_k_by_row: tuple[int, ...],
    slot_signature: tuple[int, ...],
) -> None:
    expected = int(batch_size)
    vectors = (
        row_is_compact,
        slot_by_row,
        compact_valid_tokens_by_row,
        recent_first_page_by_row,
        recent_page_count_by_row,
        row_effective_k_by_row,
        slot_signature,
    )
    if any(len(vector) != expected for vector in vectors):
        raise ValueError("overlay composer row inputs must match batch size")


def _validate_effective_k_bounds(
    *,
    row_effective_k_by_row: tuple[int, ...],
    overlay_width_pages: int,
    page_size: int,
) -> None:
    max_effective_k = int(overlay_width_pages) * int(page_size)
    for row, effective_k in enumerate(row_effective_k_by_row):
        if effective_k < 0 or effective_k > max_effective_k:
            raise ValueError(
                "row_effective_k exceeds overlay width capacity "
                f"for row {row}: effective={effective_k}, "
                f"overlay_width_pages={int(overlay_width_pages)}, page_size={int(page_size)}"
            )


def _previous_dirty_key(
    previous: CompactMixedPageOverlay | None,
    row: int,
) -> tuple[int, bool, int, int, int, int, int, int] | None:
    if previous is None or row >= len(previous.block_table_dirty_keys):
        return None
    return previous.block_table_dirty_keys[row]


def _dirty_keys(
    *,
    row_is_compact: tuple[bool, ...],
    slot_by_row: tuple[int, ...],
    compact_valid_tokens_by_row: tuple[int, ...],
    recent_first_page_by_row: tuple[int, ...],
    recent_page_count_by_row: tuple[int, ...],
    block_table_epoch: int,
    block_table_storage_key: int,
) -> tuple[tuple[int, bool, int, int, int, int, int, int], ...]:
    return tuple(
        (
            row,
            row_is_compact[row],
            slot_by_row[row],
            compact_valid_tokens_by_row[row],
            recent_first_page_by_row[row],
            recent_page_count_by_row[row],
            int(block_table_epoch),
            int(block_table_storage_key),
        )
        for row in range(len(row_is_compact))
    )


def _overlay_or_new(
    *,
    previous: CompactMixedPageOverlay | None,
    batch_size: int,
    overlay_width_pages: int,
    num_kv_heads: int,
    compact_capacity_pages: int,
    recent_capacity_pages: int,
    safe_page_id: int,
    device: torch.device,
) -> tuple[CompactMixedPageOverlay, bool]:
    table_shape = (int(batch_size), int(overlay_width_pages))
    lengths_shape = (int(batch_size) * int(num_kv_heads),)
    if (
        previous is not None
        and tuple(previous.selected_page_table_i32.shape) == table_shape
        and tuple(previous.effective_row_slot_i32.shape) == lengths_shape
        and tuple(previous.row_consume_mode_i32.shape) == (int(batch_size),)
        and tuple(previous.selected_seqused_k_by_head_i32.shape) == lengths_shape
        and previous.selected_page_table_i32.device == device
        and previous.effective_row_slot_i32.device == device
        and previous.row_consume_mode_i32.device == device
        and previous.selected_seqused_k_by_head_i32.device == device
        and previous.compact_capacity_pages == int(compact_capacity_pages)
        and previous.recent_capacity_pages == int(recent_capacity_pages)
        and previous.safe_page_id == int(safe_page_id)
    ):
        return previous, True

    return (
        CompactMixedPageOverlay(
            selected_page_table_i32=torch.empty(
                table_shape,
                dtype=torch.int32,
                device=device,
            ),
            effective_row_slot_i32=torch.empty(
                lengths_shape,
                dtype=torch.int32,
                device=device,
            ),
            row_consume_mode_i32=torch.empty(
                (int(batch_size),),
                dtype=torch.int32,
                device=device,
            ),
            selected_seqused_k_by_head_i32=torch.empty(
                lengths_shape,
                dtype=torch.int32,
                device=device,
            ),
            overlay_width_pages=int(overlay_width_pages),
            compact_capacity_pages=int(compact_capacity_pages),
            recent_capacity_pages=int(recent_capacity_pages),
            safe_page_id=int(safe_page_id),
            block_table_dirty_keys=(),
        ),
        False,
    )


def _rewrite_compact_row(
    output_row: torch.Tensor,
    canonical_row: torch.Tensor,
    *,
    reserved_manager_block_ids: tuple[int, ...],
    slot: int,
    compact_valid_tokens: int,
    recent_first_page: int,
    recent_page_count: int,
    row_effective_k: int,
    page_size: int,
    compact_capacity_pages: int,
    recent_capacity_pages: int,
    safe_page_id: int,
    canonical_width_pages: int,
) -> None:
    if int(compact_valid_tokens) < 0:
        raise ValueError("compact_valid_tokens must be non-negative")
    if int(compact_valid_tokens) % int(page_size) != 0:
        raise ValueError("compact_valid_tokens must be page-aligned")
    compact_visible_pages = int(compact_valid_tokens) // int(page_size)
    if compact_visible_pages > int(compact_capacity_pages):
        raise ValueError("compact visible pages exceed compact_capacity_pages")
    row_effective_k_i = int(row_effective_k)
    if row_effective_k_i < int(compact_valid_tokens):
        raise ValueError("row_effective_k is smaller than compact_valid_tokens")
    recent_count_i = int(recent_page_count)
    recent_first_i = int(recent_first_page)
    if recent_first_i < 0 or recent_count_i < 0:
        raise ValueError("recent page range must be non-negative")
    if recent_count_i > int(recent_capacity_pages):
        raise ValueError("recent_page_count exceeds recent_capacity_pages")
    if recent_first_i + recent_count_i > int(canonical_width_pages):
        raise ValueError("recent page range exceeds canonical block table width")
    visible_recent_tokens = row_effective_k_i - int(compact_valid_tokens)
    if visible_recent_tokens > recent_count_i * int(page_size):
        raise ValueError("row_effective_k exceeds compact plus recent visible pages")
    recent_visible_pages = (
        visible_recent_tokens + int(page_size) - 1
    ) // int(page_size) if visible_recent_tokens > 0 else 0
    visible_pages = compact_visible_pages + recent_visible_pages
    if visible_pages > int(output_row.shape[0]):
        raise ValueError("visible compact/recent pages exceed overlay row width")

    slot_start = int(slot) * int(compact_capacity_pages)
    slot_end = slot_start + int(compact_capacity_pages)
    if int(slot) < 0 or slot_end > len(reserved_manager_block_ids):
        raise ValueError("slot exceeds reserved_manager_block_ids")

    output_row.fill_(int(safe_page_id))
    for offset in range(compact_visible_pages):
        output_row[offset] = reserved_manager_block_ids[slot_start + offset]
    recent_dst = compact_visible_pages
    output_row[recent_dst : recent_dst + recent_visible_pages].copy_(
        canonical_row[recent_first_i : recent_first_i + recent_visible_pages]
    )
