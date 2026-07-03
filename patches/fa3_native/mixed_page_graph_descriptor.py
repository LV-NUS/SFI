from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch


class PageResolverKind(IntEnum):
    NATIVE = 0
    SELECTED_TABLE = 1
    COMPACT_RECENT_NATIVE = 2
    EFFECTIVE_ROW_TABLE = 3
    RESOLVED_ROW_PTR = 4


class PageResolverSubkind(IntEnum):
    ROWPTR = 0
    DIRECT_TABLE = 1
    AFFINE_CONST = 2
    AFFINE_TENSOR = 3
    AFFINE_CONST_DIRECT = 4


GRAPH_STATIC_AFFINE_CONST_FIELDS = (
    "graph_static_affine_const",
    "resolved_page_table_affine_base",
    "resolved_page_table_affine_stride",
    "resolved_page_table_affine_segment_pages",
    "resolved_page_table_affine_second_base",
    "resolved_page_table_affine_second_stride",
    "resolved_page_table_affine_batch_stride",
    "resolved_page_table_affine_head_stride",
    "resolved_page_table_affine_direct",
    "resolved_page_table_affine_cols",
)

GRAPH_MUTABLE_DEVICE_CARRIERS = (
    "graph_mutable_device_carriers",
    "kv_batch_idx_i32",
    "row_consume_mode_i32",
    "resolver_visible_seqused_k_by_head_i32",
    "selected_page_table_i32",
    "resolved_page_table_row_ptr_u64",
    "resolved_page_table_i32",
    "resolved_page_table_affine_i32",
    "resolved_page_table_affine_per_batch_base_i32",
    "capture_row_index_i32",
    "row_capture_last_n_i32",
)


_DESCRIPTOR_FIELD_NAMES = (
    "resolver_kind",
    "batch",
    "num_heads",
    "page_block_size",
    "max_pages_per_row",
    "max_selected_pages_per_row",
    "max_capture_rows",
    "q_layout_key",
    "kv_cache_addr",
    "effective_width_pages",
    "effective_row_capacity",
    "effective_table_stride",
    "native_block_table_stride",
    "native_block_table_storage_key",
    "native_block_table_epoch",
    "compact_lease_epoch",
    "safe_page_id",
    "safe_page_lease_token",
    "capture_buffer_shape_key",
    "cp_shape_key",
    "producer_stream_key",
    "resolver_subkind",
    "max_seqlen_q_bucket",
)

_CARRIER_FIELD_NAMES = (
    "row_consume_mode_i32",
    "resolver_visible_seqused_k_by_head_i32",
    "selected_page_table_i32",
    "effective_row_slot_i32",
    "compact_base_page_i32",
    "compact_page_count_i32",
    "recent_first_logical_page_i32",
    "resolved_page_table_row_ptr_u64",
    "resolved_page_table_i32",
    "resolved_page_table_affine_i32",
    "resolved_page_table_affine_base",
    "resolved_page_table_affine_stride",
    "resolved_page_table_affine_segment_pages",
    "resolved_page_table_affine_second_base",
    "resolved_page_table_affine_second_stride",
    "resolved_page_table_affine_batch_stride",
    "resolved_page_table_affine_head_stride",
    "resolved_page_table_affine_direct",
    "resolved_page_table_affine_cols",
    "resolved_page_table_affine_per_batch_base_i32",
    "capture_row_index_i32",
    "row_capture_last_n_i32",
    "resolver_subkind",
    "kv_batch_idx_i32",
)


@dataclass(frozen=True)
class ResolverGraphDescriptor:
    resolver_kind: PageResolverKind
    batch: int
    num_heads: int
    page_block_size: int
    max_pages_per_row: int
    max_selected_pages_per_row: int
    max_capture_rows: int
    q_layout_key: str
    kv_cache_addr: int
    effective_width_pages: int = 0
    effective_row_capacity: int = 0
    effective_table_stride: int = 0
    native_block_table_stride: int = 0
    native_block_table_storage_key: int = 0
    native_block_table_epoch: int = 0
    compact_lease_epoch: int = 0
    safe_page_id: int = 0
    safe_page_lease_token: int = 0
    capture_buffer_shape_key: str = ""
    cp_shape_key: str = ""
    producer_stream_key: str = ""
    resolver_subkind: PageResolverSubkind = PageResolverSubkind.ROWPTR
    max_seqlen_q_bucket: int = 0
    max_seqlen_k_bucket: int = 0

    def replay_key(self) -> tuple[object, ...]:
        return (
            int(self.resolver_kind),
            int(self.batch),
            int(self.num_heads),
            int(self.page_block_size),
            int(self.max_pages_per_row),
            int(self.max_selected_pages_per_row),
            int(self.max_capture_rows),
            self.q_layout_key,
            int(self.kv_cache_addr),
            int(self.effective_width_pages),
            int(self.effective_row_capacity),
            int(self.effective_table_stride),
            int(self.native_block_table_stride),
            int(self.native_block_table_storage_key),
            int(self.native_block_table_epoch),
            int(self.compact_lease_epoch),
            int(self.safe_page_id),
            int(self.safe_page_lease_token),
            self.capture_buffer_shape_key,
            self.cp_shape_key,
            self.producer_stream_key,
            int(self.resolver_subkind),
            int(self.max_seqlen_q_bucket),
            int(self.max_seqlen_k_bucket),
        )


@dataclass
class MixedPageResolverCarrierSet:
    row_consume_mode_i32: torch.Tensor | None
    resolver_visible_seqused_k_by_head_i32: torch.Tensor | None
    selected_page_table_i32: torch.Tensor | None
    effective_row_slot_i32: torch.Tensor | None
    compact_base_page_i32: torch.Tensor | None
    compact_page_count_i32: torch.Tensor | None
    recent_first_logical_page_i32: torch.Tensor | None
    resolved_page_table_row_ptr_u64: torch.Tensor | None = None
    resolved_page_table_i32: torch.Tensor | None = None
    resolved_page_table_affine_i32: torch.Tensor | None = None
    resolved_page_table_affine_base: int | None = None
    resolved_page_table_affine_stride: int | None = None
    resolved_page_table_affine_segment_pages: int | None = None
    resolved_page_table_affine_second_base: int | None = None
    resolved_page_table_affine_second_stride: int | None = None
    resolved_page_table_affine_batch_stride: int | None = None
    resolved_page_table_affine_head_stride: int = 0
    resolved_page_table_affine_direct: bool = False
    resolved_page_table_affine_cols: int = 0
    resolved_page_table_affine_per_batch_base_i32: torch.Tensor | None = None  # fa3_sm90_perbatch_base
    capture_row_index_i32: torch.Tensor | None = None
    row_capture_last_n_i32: torch.Tensor | None = None
    resolver_subkind: PageResolverSubkind = PageResolverSubkind.ROWPTR
    kv_batch_idx_i32: torch.Tensor | None = None

    def pointer_signature(self) -> tuple[int | None, ...]:
        values = (
            self.row_consume_mode_i32,
            self.resolver_visible_seqused_k_by_head_i32,
            self.selected_page_table_i32,
            self.effective_row_slot_i32,
            self.compact_base_page_i32,
            self.compact_page_count_i32,
            self.recent_first_logical_page_i32,
            self.resolved_page_table_row_ptr_u64,
            self.resolved_page_table_i32,
            self.resolved_page_table_affine_i32,
            self.resolved_page_table_affine_base,
            self.resolved_page_table_affine_stride,
            self.resolved_page_table_affine_segment_pages,
            self.resolved_page_table_affine_second_base,
            self.resolved_page_table_affine_second_stride,
            self.resolved_page_table_affine_batch_stride,
            self.resolved_page_table_affine_head_stride,
            self.resolved_page_table_affine_direct,
            self.resolved_page_table_affine_cols,
            self.resolved_page_table_affine_per_batch_base_i32,
            self.capture_row_index_i32,
            self.row_capture_last_n_i32,
            int(self.resolver_subkind),
            self.kv_batch_idx_i32,
        )
        return tuple(
            None
            if value is None
            else int(value.data_ptr())
            if isinstance(value, torch.Tensor)
            else int(value)
            for value in values
        )


def assert_descriptor_matches_replay(
    captured: ResolverGraphDescriptor,
    replay: ResolverGraphDescriptor,
) -> None:
    captured_key = captured.replay_key()
    replay_key = replay.replay_key()
    if captured_key == replay_key:
        return

    mismatches = [
        f"{name}: captured={captured_value!r}, replay={replay_value!r}"
        for name, captured_value, replay_value in zip(
            _DESCRIPTOR_FIELD_NAMES,
            captured_key,
            replay_key,
        )
        if captured_value != replay_value
    ]
    raise ValueError(
        "resolver graph descriptor replay key mismatch: "
        + ", ".join(mismatches)
    )


def assert_carrier_addresses_stable(
    captured_signature: tuple[int | None, ...],
    replay_signature: tuple[int | None, ...],
) -> None:
    if captured_signature == replay_signature:
        return

    max_len = max(len(captured_signature), len(replay_signature))
    mismatches: list[str] = []
    for index in range(max_len):
        captured_value = (
            captured_signature[index]
            if index < len(captured_signature)
            else "<missing>"
        )
        replay_value = (
            replay_signature[index]
            if index < len(replay_signature)
            else "<missing>"
        )
        if captured_value == replay_value:
            continue
        name = (
            _CARRIER_FIELD_NAMES[index]
            if index < len(_CARRIER_FIELD_NAMES)
            else f"index_{index}"
        )
        mismatches.append(
            f"{name} (index {index}): captured={captured_value!r}, replay={replay_value!r}"
        )

    raise RuntimeError(
        "mixed-page graph carrier pointer replacement detected: "
        + ", ".join(mismatches)
    )
