"""CPU-owned planning helpers for mixed-page resolver carrier slots."""

from __future__ import annotations

import torch

from patches.fa3_native.mixed_page_graph_descriptor import MixedPageResolverCarrierSet

_RESOLVED_ROW_PTR_REPLAY_ROW_PTR_ATTR_PAIR = (
    "mixed_page_resolver_replay_row_ptr_u64",
    "mixed_page_resolver_replay_row_ptr_source_u64",
)
_RESOLVED_ROW_PTR_REPLAY_SEQUSED_ATTR_PAIR = (
    "mixed_page_resolver_replay_seqused_k_i32",
    "mixed_page_resolver_replay_seqused_k_source_i32",
)












def _same_tensor(lhs: torch.Tensor | None, rhs: torch.Tensor | None) -> bool:
    if lhs is None or rhs is None:
        return lhs is rhs
    return (
        lhs is rhs
        or (
            int(lhs.data_ptr()) == int(rhs.data_ptr())
            and lhs.dtype == rhs.dtype
            and lhs.device == rhs.device
            and tuple(lhs.shape) == tuple(rhs.shape)
            and tuple(lhs.stride()) == tuple(rhs.stride())
        )
    )


def _has_resolved_affine(carriers: MixedPageResolverCarrierSet) -> bool:
    return (
        carriers.resolved_page_table_affine_i32 is not None
        or carriers.resolved_page_table_affine_base is not None
        or carriers.resolved_page_table_affine_stride is not None
        or carriers.resolved_page_table_affine_segment_pages is not None
        or carriers.resolved_page_table_affine_second_base is not None
        or carriers.resolved_page_table_affine_second_stride is not None
        or carriers.resolved_page_table_affine_batch_stride is not None
    )


def _same_optional_scalar(lhs: object, rhs: object) -> bool:
    if lhs is None or rhs is None:
        return lhs is rhs
    return int(lhs) == int(rhs)


def _validate_affine_direct_bound(
    dst: MixedPageResolverCarrierSet,
    src: MixedPageResolverCarrierSet,
) -> None:
    if not _same_tensor(
        dst.resolved_page_table_affine_i32,
        src.resolved_page_table_affine_i32,
    ):
        raise ValueError("ResolvedRowPtr replay affine metadata must be direct-bound")
    for name in (
        "resolved_page_table_affine_base",
        "resolved_page_table_affine_stride",
        "resolved_page_table_affine_segment_pages",
        "resolved_page_table_affine_second_base",
        "resolved_page_table_affine_second_stride",
        "resolved_page_table_affine_batch_stride",
    ):
        if not _same_optional_scalar(getattr(dst, name), getattr(src, name)):
            raise ValueError("ResolvedRowPtr replay affine metadata must be direct-bound")


def _carrier_update_pairs(
    attn_metadata: object,
) -> tuple[tuple[MixedPageResolverCarrierSet, MixedPageResolverCarrierSet], ...]:
    updates = getattr(attn_metadata, "mixed_page_resolver_replay_carrier_updates", None)
    if updates is not None:
        return tuple((dst, src) for dst, src in updates)
    dst = getattr(attn_metadata, "mixed_page_resolver_replay_carriers", None)
    src = getattr(attn_metadata, "mixed_page_resolver_replay_carrier_sources", None)
    if dst is None and src is None:
        return ()
    return ((dst, src),)


def _validate_resolved_row_ptr_direct_bound(
    attn_metadata: object,
    dst: MixedPageResolverCarrierSet,
    src: MixedPageResolverCarrierSet,
) -> str:
    uses_resolved_affine = _has_resolved_affine(dst) or _has_resolved_affine(src)
    uses_resolved_row_ptr = (
        dst.resolved_page_table_row_ptr_u64 is not None
        or src.resolved_page_table_row_ptr_u64 is not None
        or uses_resolved_affine
    )
    uses_compact_native = (
        dst.compact_base_page_i32 is not None
        or dst.compact_page_count_i32 is not None
        or dst.recent_first_logical_page_i32 is not None
        or src.compact_base_page_i32 is not None
        or src.compact_page_count_i32 is not None
        or src.recent_first_logical_page_i32 is not None
    )
    if uses_resolved_row_ptr:
        forbidden = (
            dst.selected_page_table_i32,
            dst.effective_row_slot_i32,
            dst.compact_base_page_i32,
            dst.compact_page_count_i32,
            dst.recent_first_logical_page_i32,
            src.selected_page_table_i32,
            src.effective_row_slot_i32,
            src.compact_base_page_i32,
            src.compact_page_count_i32,
            src.recent_first_logical_page_i32,
        )
        if any(tensor is not None for tensor in forbidden):
            raise ValueError(
                "ResolvedRowPtr replay carriers received forbidden legacy selected/compact/effective fields"
            )
        has_row_consume = (
            dst.row_consume_mode_i32 is not None
            or src.row_consume_mode_i32 is not None
        )
        if has_row_consume and not uses_resolved_affine:
            raise ValueError(
                "ResolvedRowPtr replay row_consume_mode_i32 requires affine carrier"
            )
        if (
            dst.resolver_visible_seqused_k_by_head_i32 is None
            or src.resolver_visible_seqused_k_by_head_i32 is None
        ):
            raise ValueError(
                "ResolvedRowPtr replay carriers require visible lengths"
            )
        if dst.resolved_page_table_row_ptr_u64 is not None or src.resolved_page_table_row_ptr_u64 is not None:
            if not _same_tensor(
                dst.resolved_page_table_row_ptr_u64,
                src.resolved_page_table_row_ptr_u64,
            ):
                raise ValueError(
                    "ResolvedRowPtr replay carriers must be direct-bound; "
                    "source/destination copies are retired"
                )
        elif not uses_resolved_affine:
            raise ValueError(
                "ResolvedRowPtr replay carriers require row-pointer or affine metadata"
            )
        if not _same_tensor(
            dst.resolver_visible_seqused_k_by_head_i32,
            src.resolver_visible_seqused_k_by_head_i32,
        ):
            raise ValueError(
                "ResolvedRowPtr replay visible lengths must be direct-bound; "
                "source/destination copies are retired"
            )
        if uses_resolved_affine:
            _validate_affine_direct_bound(dst, src)
        if has_row_consume and not _same_tensor(
            dst.row_consume_mode_i32,
            src.row_consume_mode_i32,
        ):
            raise ValueError(
                "ResolvedRowPtr replay row consume metadata must be direct-bound"
            )
        row_ptr_dst_attr, row_ptr_src_attr = _RESOLVED_ROW_PTR_REPLAY_ROW_PTR_ATTR_PAIR
        row_ptr_dst = getattr(attn_metadata, row_ptr_dst_attr, None)
        row_ptr_src = getattr(attn_metadata, row_ptr_src_attr, None)
        if row_ptr_dst is not None or row_ptr_src is not None:
            if not isinstance(row_ptr_dst, torch.Tensor) or not isinstance(row_ptr_src, torch.Tensor):
                raise ValueError("ResolvedRowPtr replay row-pointer metadata must be tensors")
            if not _same_tensor(row_ptr_dst, row_ptr_src):
                raise ValueError(
                    "ResolvedRowPtr replay row-pointer metadata must be direct-bound"
                )
        seq_dst_attr, seq_src_attr = _RESOLVED_ROW_PTR_REPLAY_SEQUSED_ATTR_PAIR
        seq_dst = getattr(attn_metadata, seq_dst_attr, None)
        seq_src = getattr(attn_metadata, seq_src_attr, None)
        if seq_dst is not None or seq_src is not None:
            if not isinstance(seq_dst, torch.Tensor) or not isinstance(seq_src, torch.Tensor):
                raise ValueError("ResolvedRowPtr replay visible-length metadata must be tensors")
            if not _same_tensor(seq_dst, seq_src):
                raise ValueError(
                    "ResolvedRowPtr replay visible-length metadata must be direct-bound"
                )
        return "resolved_row_ptr"
    if uses_compact_native:
        raise ValueError(
            "CompactRecentNative replay carriers are retired; "
            "use direct-bound ResolvedRowPtr carriers"
        )
    uses_retired_effective_table = (
        dst.effective_row_slot_i32 is not None
        or src.effective_row_slot_i32 is not None
    )
    if uses_retired_effective_table:
        raise ValueError(
            "EffectiveRowTable replay carriers are retired; "
            "use direct-bound ResolvedRowPtr carriers"
        )
    uses_retired_selected_table = (
        dst.selected_page_table_i32 is not None
        or src.selected_page_table_i32 is not None
        or dst.row_consume_mode_i32 is not None
        or src.row_consume_mode_i32 is not None
    )
    if uses_retired_selected_table:
        raise ValueError(
            "SelectedTable replay carriers are retired; "
            "use direct-bound ResolvedRowPtr carriers"
        )
    raise ValueError(
        "unsupported replay carriers; use direct-bound ResolvedRowPtr carriers"
    )


def refresh_mixed_page_resolver_carriers_for_replay(
    *,
    attn_metadata: object,
    stream_key: str,
) -> dict[str, int]:
    rrp_direct_bound_count = 0
    for dst, src in _carrier_update_pairs(attn_metadata):
        if not isinstance(dst, MixedPageResolverCarrierSet):
            raise ValueError("mixed_page_resolver_replay_carriers must be MixedPageResolverCarrierSet")
        if not isinstance(src, MixedPageResolverCarrierSet):
            raise ValueError("mixed_page_resolver_replay_carrier_sources must be MixedPageResolverCarrierSet")
        carrier_kind = _validate_resolved_row_ptr_direct_bound(attn_metadata, dst, src)
        if carrier_kind == "resolved_row_ptr":
            rrp_direct_bound_count += 1

    stats = {
        "carrier_update_rows": 0,
        "carrier_update_bytes": 0,
        "carrier_update_kernel_count": 0,
    }
    if rrp_direct_bound_count:
        stats["carrier_update_resolved_row_ptr"] = int(rrp_direct_bound_count)
        stats["carrier_update_direct_bound"] = int(rrp_direct_bound_count)
    updated = bool(rrp_direct_bound_count)
    setattr(attn_metadata, "mixed_page_resolver_replay_carriers_updated", updated)
    setattr(attn_metadata, "mixed_page_resolver_replay_carrier_update_stream_key", str(stream_key))
    setattr(attn_metadata, "mixed_page_resolver_replay_carrier_update_stats", stats)
    return stats


