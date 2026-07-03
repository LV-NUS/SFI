from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from patches.fa3_native.mixed_page_graph_descriptor import (
    MixedPageResolverCarrierSet,
    PageResolverKind,
    PageResolverSubkind,
    ResolverGraphDescriptor,
    assert_carrier_addresses_stable,
    assert_descriptor_matches_replay,
)
from patches.fa3_native.row_consume_modes import (
    ROW_CONSUME_MODE_FULL_I32,
    ROW_CONSUME_MODE_SELECTED_I32,
)
from patches.fa_sparse_runtime.contracts import PAGE_TABLE_LAYOUT_KV_HEAD_FLAT

PAGE_RESOLVER_KIND_NATIVE = int(PageResolverKind.NATIVE)
PAGE_RESOLVER_KIND_SELECTED_TABLE = int(PageResolverKind.SELECTED_TABLE)
PAGE_RESOLVER_KIND_COMPACT_RECENT_NATIVE = int(PageResolverKind.COMPACT_RECENT_NATIVE)
PAGE_RESOLVER_KIND_EFFECTIVE_ROW_TABLE = int(PageResolverKind.EFFECTIVE_ROW_TABLE)
PAGE_RESOLVER_KIND_RESOLVED_ROW_PTR = int(PageResolverKind.RESOLVED_ROW_PTR)
PAGE_RESOLVER_SUBKIND_ROWPTR = int(PageResolverSubkind.ROWPTR)
PAGE_RESOLVER_SUBKIND_DIRECT_TABLE = int(PageResolverSubkind.DIRECT_TABLE)
PAGE_RESOLVER_SUBKIND_AFFINE_CONST = int(PageResolverSubkind.AFFINE_CONST)
PAGE_RESOLVER_SUBKIND_AFFINE_TENSOR = int(PageResolverSubkind.AFFINE_TENSOR)
PAGE_RESOLVER_SUBKIND_AFFINE_CONST_DIRECT = int(PageResolverSubkind.AFFINE_CONST_DIRECT)
FA3_SM90_MIXED_PAGE_ROUTE_FIELDS = (
    "backend",
    "extension_path",
    "extension_sha256",
    "no_fa4_cute_dispatch",
    "page_resolver_kind0_count",
    "page_resolver_kind1_count",
    "page_resolver_kind4_count",
    "effective_visible_k_by_row",
    "dense_reference_match",
    "unsupported_fallback_count",
)


def require_fa3_sm90_mixed_page_route_ready(
    *,
    device_capability: tuple[int, int],
    fa3_sm90_mixed_page_tier1_passed: bool,
) -> None:
    if device_capability != (9, 0):
        raise RuntimeError("FA3 SM90 mixed_page route requires H100 sm90")
    if not bool(fa3_sm90_mixed_page_tier1_passed):
        raise RuntimeError("FA3 SM90 mixed_page route requires Tier 1 kernel completeness")


def build_fa3_sm90_mixed_page_route_summary(
    *,
    extension_path: str,
    extension_sha256: str,
    page_resolver_kind0_count: int,
    page_resolver_kind1_count: int,
    page_resolver_kind4_count: int,
    effective_visible_k_by_row: list[int],
    dense_reference_match: bool,
    unsupported_fallback_count: int,
) -> dict[str, object]:
    summary: dict[str, object] = {
        "backend": "fa3_sm90_mixed_page",
        "extension_path": extension_path,
        "extension_sha256": extension_sha256,
        "no_fa4_cute_dispatch": True,
        "page_resolver_kind0_count": int(page_resolver_kind0_count),
        "page_resolver_kind1_count": int(page_resolver_kind1_count),
        "page_resolver_kind4_count": int(page_resolver_kind4_count),
        "effective_visible_k_by_row": [int(value) for value in effective_visible_k_by_row],
        "dense_reference_match": bool(dense_reference_match),
        "unsupported_fallback_count": int(unsupported_fallback_count),
    }
    missing = [field for field in FA3_SM90_MIXED_PAGE_ROUTE_FIELDS if field not in summary]
    if missing:
        raise RuntimeError(f"missing FA3 SM90 mixed_page route fields: {missing}")
    return summary

# NOTE: torch._assert_async used in this module fires device-side and trashes
# the CUDA context on failure (requires process restart). These invariants are
# therefore treated as programmer-error contracts, not recoverable validation:
# host-side `bool(torch.any(...).item())` / `.max().item()` checks were removed
# from `_build_selected_block_inputs`, `_build_capture_block_inputs`, and the
# `build_mixed_page_launch_inputs` routing predicate to meet the hot-path
# "0 sync" rule, then re-expressed as `_assert_async` where applicable. Callers
# must pre-validate inputs (e.g. pass an explicit `has_selected_rows` CPU bool
# into `build_mixed_page_launch_inputs`) rather than relying on this module to
# gracefully reject malformed tensors.


def _resolve_max_seqlen_k_host(
    *,
    seqused_k: torch.Tensor,
    override: int | None,
    batch_size: int,
) -> int:
    """Resolve max_seqlen_k for kernel launch configuration.

    Callers MUST supply ``override`` from CPU-side step authority /
    attn_metadata. Missing bounds are a contract violation; silently deriving
    them from ``seqused_k`` would add a D2H sync and hide route bugs.
    """
    if batch_size == 0:
        return 0
    if override is not None:
        return int(override)
    raise ValueError(
        "max_seqlen_k_override is required; callers must provide the CPU-side "
        "bound instead of reading seqused_k on the host"
    )


def _validate_identity_kv_batch_idx(
    kv_batch_idx_i32: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> None:
    expected = torch.arange(batch_size, device=device, dtype=torch.int32)
    torch._assert_async(
        torch.all(kv_batch_idx_i32.reshape(batch_size) == expected),
        "decode-only mixed-page bridge requires identity kv_batch_idx",
    )


def _validate_kv_head_flat_layout(page_table_layout: int) -> None:
    if int(page_table_layout) != int(PAGE_TABLE_LAYOUT_KV_HEAD_FLAT):
        raise ValueError("decode-only mixed-page bridge requires KV_HEAD_FLAT layout")


def _build_capture_block_inputs(
    *,
    seqused_k: torch.Tensor,
    request_selected_rows: torch.Tensor,
    row_is_capture_producer: torch.Tensor,
    row_capture_last_n_i32: torch.Tensor,
    max_capture_k: int | None,
    max_capture_last_n: int | None,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    batch_size = int(seqused_k.shape[0])
    device = seqused_k.device
    producer_mask = row_is_capture_producer.to(device=device, dtype=torch.bool).reshape(batch_size)
    row_capture_last_n_i32 = row_capture_last_n_i32.to(device=device, dtype=torch.int32).reshape(batch_size)

    # Capture ownership is independent of row consume mode. The kernel uses
    # capture_row_index_i32 (-1 => skip) and row_capture_last_n_i32 to decide
    # which rows write side outputs.

    if max_capture_k is None:
        raise ValueError("max_capture_k must be provided for capture rows")
    if max_capture_last_n is None:
        raise ValueError("max_capture_last_n must be provided for capture rows")

    normalized_max_capture_k = int(max_capture_k)
    normalized_max_capture_last_n = int(max_capture_last_n)
    if normalized_max_capture_k <= 0:
        raise ValueError("max_capture_k must be positive for capture rows")
    if normalized_max_capture_last_n <= 0:
        raise ValueError("max_capture_last_n must be positive for capture rows")

    producer_i32 = producer_mask.to(dtype=torch.int32)
    producer_index_i32 = torch.cumsum(producer_i32, dim=0) - 1
    missing_capture_row_i32 = torch.full(
        (batch_size,),
        -1,
        device=device,
        dtype=torch.int32,
    )
    capture_row_index_i32 = torch.where(
        producer_mask,
        producer_index_i32,
        missing_capture_row_i32,
    )
    normalized_last_n = torch.where(
        producer_mask,
        row_capture_last_n_i32,
        torch.zeros_like(row_capture_last_n_i32),
    )
    torch._assert_async(
        torch.logical_not(torch.any(producer_mask & (normalized_last_n < 1))),
        "producer rows must have row_capture_last_n_i32 >= 1",
    )
    return (
        capture_row_index_i32,
        normalized_last_n,
        normalized_max_capture_k,
        normalized_max_capture_last_n,
    )


def build_mixed_page_launch_inputs(
    *,
    dense_block_table_i32: torch.Tensor,
    dense_seqused_k: torch.Tensor,
    dense_cp_tot_seqused_k: torch.Tensor | None = None,
    step_cache_page_table_i32: torch.Tensor | None = None,
    step_cache_selected_seqused_k_by_head_i32: torch.Tensor | None = None,
    step_cache_kv_batch_idx_i32: torch.Tensor | None = None,
    step_cache_page_table_layout: int | None = None,
    num_kv_heads: int | None = None,
    request_selected_rows: torch.Tensor | None = None,
    step_cache_cp_selected_seqused_k_by_head_i32: torch.Tensor | None = None,
    cp_world_size: int | None = None,
    has_selected_rows: bool | None = None,
    row_is_capture_producer: torch.Tensor | None = None,
    row_capture_last_n_i32: torch.Tensor | None = None,
    max_capture_k: int | None = None,
    max_capture_last_n: int | None = None,
    max_seqlen_k_override: int | None = None,
    page_resolver_kind: int = PAGE_RESOLVER_KIND_NATIVE,
    compact_base_page_i32: torch.Tensor | None = None,
    compact_page_count_i32: torch.Tensor | None = None,
    recent_first_logical_page_i32: torch.Tensor | None = None,
    graph_replay_carriers: bool = False,
    resolver_descriptor: ResolverGraphDescriptor | None = None,
    resolver_carriers: MixedPageResolverCarrierSet | None = None,
    captured_resolver_descriptor: ResolverGraphDescriptor | None = None,
    captured_resolver_pointer_signature: tuple[int | None, ...] | None = None,
) -> MixedPageLaunchInputs:
    del dense_block_table_i32

    batch_size = int(dense_seqused_k.shape[0])
    device = dense_seqused_k.device
    page_resolver_kind_i = int(page_resolver_kind)
    if page_resolver_kind_i == PAGE_RESOLVER_KIND_COMPACT_RECENT_NATIVE:
        raise ValueError(
            "CompactRecentNative page resolver is retired; use ResolvedRowPtr row pointers"
        )
    if page_resolver_kind_i == PAGE_RESOLVER_KIND_EFFECTIVE_ROW_TABLE:
        raise ValueError(
            "EffectiveRowTable page resolver is retired; use ResolvedRowPtr row pointers"
        )
    if page_resolver_kind_i not in (
        PAGE_RESOLVER_KIND_NATIVE,
        PAGE_RESOLVER_KIND_RESOLVED_ROW_PTR,
    ):
        raise ValueError("page_resolver_kind must be 0 or 4")
    dense_seqused_k = dense_seqused_k.to(device=device, dtype=torch.int32).reshape(batch_size)
    dense_cp_tot_seqused_k = (
        dense_seqused_k
        if dense_cp_tot_seqused_k is None
        else dense_cp_tot_seqused_k.to(device=device, dtype=torch.int32).reshape(batch_size)
    )
    cp_world_size = int(cp_world_size) if cp_world_size is not None else 1
    request_selected_rows = (
        torch.zeros((batch_size,), device=device, dtype=torch.bool)
        if request_selected_rows is None
        else request_selected_rows.to(device=device, dtype=torch.bool).reshape(batch_size)
    )

    if graph_replay_carriers:
        validate_mixed_page_resolver_replay(
            enabled=True,
            captured_descriptor=captured_resolver_descriptor,
            replay_descriptor=resolver_descriptor,
            captured_pointer_signature=captured_resolver_pointer_signature,
            replay_carriers=resolver_carriers,
        )
        if row_is_capture_producer is not None or row_capture_last_n_i32 is not None:
            raise ValueError(
                "capture metadata generation is not supported during graph replay; "
                "capture carriers must be prebound before replay"
            )

    if resolver_carriers is not None:
        if page_resolver_kind_i == PAGE_RESOLVER_KIND_RESOLVED_ROW_PTR:
            row_ptr = resolver_carriers.resolved_page_table_row_ptr_u64
            direct_i32 = resolver_carriers.resolved_page_table_i32
            visible = resolver_carriers.resolver_visible_seqused_k_by_head_i32
            affine_i32 = resolver_carriers.resolved_page_table_affine_i32
            affine_base = resolver_carriers.resolved_page_table_affine_base
            affine_stride = resolver_carriers.resolved_page_table_affine_stride
            affine_direct = bool(resolver_carriers.resolved_page_table_affine_direct)
            affine_cols = int(resolver_carriers.resolved_page_table_affine_cols)
            affine_head_stride = int(resolver_carriers.resolved_page_table_affine_head_stride)
            has_affine_const = affine_base is not None or affine_stride is not None
            has_affine = affine_i32 is not None or has_affine_const
            has_direct = direct_i32 is not None
            active_resolved_carriers = int(row_ptr is not None) + int(has_direct) + int(has_affine)
            if has_affine_const and (affine_base is None or affine_stride is None):
                raise ValueError("ResolvedRowPtr affine resolver requires base and stride together")
            has_affine_const_direct = bool(
                has_affine_const
                and affine_i32 is None
                and (
                    affine_direct
                    or resolver_carriers.resolved_page_table_affine_batch_stride is not None
                )
                and resolver_carriers.resolved_page_table_affine_segment_pages in (None, 0)
                and resolver_carriers.resolved_page_table_affine_second_base in (None, 0)
                and resolver_carriers.resolved_page_table_affine_second_stride in (None, 1)
                and affine_head_stride == 0
            )
            if affine_direct and not has_affine_const_direct:
                raise ValueError(
                    "ResolvedRowPtr direct affine resolver requires const base/stride "
                    "with direct-compatible segment fields"
                )
            if active_resolved_carriers == 0:
                raise ValueError("ResolvedRowPtr resolver requires row-pointer, direct table, or affine carrier")
            if active_resolved_carriers > 1:
                raise ValueError("ResolvedRowPtr resolver requires exactly one resolved carrier subkind")
            if has_direct:
                resolver_subkind_i = PAGE_RESOLVER_SUBKIND_DIRECT_TABLE
            elif affine_i32 is not None:
                # [AFFINE-TENSOR-RETIRED 2026-07-02] the AFFINE_TENSOR device arm was
                # retired by TU-INSTANCE-DIET (kernel dispatch TORCH_CHECKs it); fail
                # fast on the host instead of crashing inside the extension. Per-row
                # affine capability returns via the Lite manager arm once its
                # production wiring lands.
                raise ValueError(
                    "AFFINE_TENSOR resolver subkind is retired (TU-INSTANCE-DIET); "
                    "production carriers must not set resolved_page_table_affine_i32"
                )
            elif has_affine_const:
                resolver_subkind_i = (
                    PAGE_RESOLVER_SUBKIND_AFFINE_CONST_DIRECT
                    if has_affine_const_direct
                    else PAGE_RESOLVER_SUBKIND_AFFINE_CONST
                )
            else:
                resolver_subkind_i = PAGE_RESOLVER_SUBKIND_ROWPTR
            if resolver_carriers.selected_page_table_i32 is not None:
                raise ValueError("ResolvedRowPtr resolver must not set selected_page_table_i32")
            if resolver_carriers.row_consume_mode_i32 is not None:
                raise ValueError("ResolvedRowPtr resolver does not accept row_consume_mode_i32")
            if (
                resolver_carriers.effective_row_slot_i32 is not None
                or resolver_carriers.compact_base_page_i32 is not None
                or resolver_carriers.compact_page_count_i32 is not None
                or resolver_carriers.recent_first_logical_page_i32 is not None
            ):
                raise ValueError("ResolvedRowPtr resolver must not set legacy compact/effective carriers")
            if cp_world_size > 1:
                raise ValueError("ResolvedRowPtr resolver launch does not support CP in this landing")
            if graph_replay_carriers and max_seqlen_k_override is None:
                raise ValueError("max_seqlen_k_override is required for graph replay resolver launches")
            max_seqlen_k = _resolve_max_seqlen_k_host(
                seqused_k=dense_seqused_k,
                override=max_seqlen_k_override,
                batch_size=batch_size,
            )
            if row_is_capture_producer is None and row_capture_last_n_i32 is None:
                capture_row_index_i32 = None
                normalized_last_n = None
                max_capture_k = 0
                max_capture_last_n = 0
            else:
                if row_is_capture_producer is None or row_capture_last_n_i32 is None:
                    raise ValueError("capture block must be all-or-none")
                capture_row_index_i32, normalized_last_n, max_capture_k, max_capture_last_n = _build_capture_block_inputs(
                    seqused_k=dense_seqused_k,
                    request_selected_rows=request_selected_rows,
                    row_is_capture_producer=row_is_capture_producer,
                    row_capture_last_n_i32=row_capture_last_n_i32,
                    max_capture_k=max_capture_k,
                    max_capture_last_n=max_capture_last_n,
                )
            return MixedPageLaunchInputs(
                selected_page_table_i32=None,
                effective_row_slot_i32=None,
                row_consume_mode_i32=None,
                selected_seqused_k_by_head_i32=None,
                cp_selected_seqused_k_by_head_i32=None,
                seqused_k=dense_seqused_k,
                cp_tot_seqused_k=dense_cp_tot_seqused_k,
                max_seqlen_k=max_seqlen_k,
                capture_row_index_i32=capture_row_index_i32,
                row_capture_last_n_i32=normalized_last_n,
                max_capture_k=max_capture_k,
                max_capture_last_n=max_capture_last_n,
                page_resolver_kind=page_resolver_kind_i,
                page_resolver_subkind=resolver_subkind_i,
                resolved_page_table_row_ptr_u64=row_ptr,
                resolved_page_table_i32=direct_i32,
                resolved_page_table_affine_i32=affine_i32,
                resolved_page_table_affine_base=resolver_carriers.resolved_page_table_affine_base,
                resolved_page_table_affine_stride=resolver_carriers.resolved_page_table_affine_stride,
                resolved_page_table_affine_segment_pages=resolver_carriers.resolved_page_table_affine_segment_pages,
                resolved_page_table_affine_second_base=resolver_carriers.resolved_page_table_affine_second_base,
                resolved_page_table_affine_second_stride=resolver_carriers.resolved_page_table_affine_second_stride,
                resolved_page_table_affine_batch_stride=resolver_carriers.resolved_page_table_affine_batch_stride,
                resolved_page_table_affine_head_stride=affine_head_stride,
                resolved_page_table_affine_direct=has_affine_const_direct,
                resolved_page_table_affine_cols=affine_cols,
                resolved_seqused_k_by_head_i32=visible,
                compact_base_page_i32=None,
                compact_page_count_i32=None,
                recent_first_logical_page_i32=None,
                graph_replay_carriers=bool(graph_replay_carriers),
                resolver_descriptor=resolver_descriptor,
                resolver_carriers=resolver_carriers,
            )
        raise ValueError("resolver_carriers require the ResolvedRowPtr production resolver")

    if page_resolver_kind_i == PAGE_RESOLVER_KIND_RESOLVED_ROW_PTR:
        raise ValueError("ResolvedRowPtr resolver requires resolver_carriers")

    if has_selected_rows is None:
        raise RuntimeError(
            "build_mixed_page_launch_inputs requires explicit has_selected_rows; "
            "caller must compute via CPU-side step_authority, not GPU readback"
        )

    if bool(has_selected_rows):
        raise ValueError(
            "legacy SelectedTable selected-rows path is removed; "
            "production compact+recent must use ResolvedRowPtr carriers"
        )
    # Dense / full-KV path: the only live non-carrier route. The launch
    # bound must come from CPU-side step authority, not a host readback
    # from seqused_k.
    selected_page_table_i32 = None
    row_consume_mode_i32 = None
    selected_seqused_k_by_head_i32 = None
    cp_selected_seqused_k_by_head_i32 = None
    seqused_k = dense_seqused_k
    cp_tot_seqused_k = dense_cp_tot_seqused_k
    max_seqlen_k = _resolve_max_seqlen_k_host(
        seqused_k=seqused_k,
        override=max_seqlen_k_override,
        batch_size=batch_size,
    )

    if row_is_capture_producer is None and row_capture_last_n_i32 is None:
        capture_row_index_i32 = None
        normalized_last_n = None
        max_capture_k = 0
        max_capture_last_n = 0
    else:
        if row_is_capture_producer is None or row_capture_last_n_i32 is None:
            raise ValueError("capture block must be all-or-none")
        capture_row_index_i32, normalized_last_n, max_capture_k, max_capture_last_n = _build_capture_block_inputs(
            seqused_k=seqused_k,
            request_selected_rows=request_selected_rows,
            row_is_capture_producer=row_is_capture_producer,
            row_capture_last_n_i32=row_capture_last_n_i32,
            max_capture_k=max_capture_k,
            max_capture_last_n=max_capture_last_n,
        )

    return MixedPageLaunchInputs(
        selected_page_table_i32=None,
        effective_row_slot_i32=None,
        row_consume_mode_i32=None,
        selected_seqused_k_by_head_i32=None,
        cp_selected_seqused_k_by_head_i32=None,
        seqused_k=seqused_k,
        cp_tot_seqused_k=cp_tot_seqused_k,
        max_seqlen_k=max_seqlen_k,
        capture_row_index_i32=capture_row_index_i32,
        row_capture_last_n_i32=normalized_last_n,
        max_capture_k=max_capture_k,
        max_capture_last_n=max_capture_last_n,
        page_resolver_kind=page_resolver_kind_i,
        page_resolver_subkind=PAGE_RESOLVER_SUBKIND_ROWPTR,
        compact_base_page_i32=compact_base_page_i32,
        compact_page_count_i32=compact_page_count_i32,
        recent_first_logical_page_i32=recent_first_logical_page_i32,
        graph_replay_carriers=bool(graph_replay_carriers),
        resolver_descriptor=resolver_descriptor,
        resolver_carriers=resolver_carriers,
    )


@dataclass(frozen=True, slots=True)
class MixedPageLaunchInputs:
    selected_page_table_i32: Optional[torch.Tensor]
    effective_row_slot_i32: Optional[torch.Tensor]
    row_consume_mode_i32: Optional[torch.Tensor]
    selected_seqused_k_by_head_i32: Optional[torch.Tensor]
    cp_selected_seqused_k_by_head_i32: Optional[torch.Tensor]
    seqused_k: torch.Tensor
    cp_tot_seqused_k: torch.Tensor
    max_seqlen_k: int
    resolved_page_table_row_ptr_u64: Optional[torch.Tensor] = None
    resolved_page_table_affine_i32: Optional[torch.Tensor] = None
    resolved_page_table_affine_base: Optional[int] = None
    resolved_page_table_affine_stride: Optional[int] = None
    resolved_page_table_affine_segment_pages: Optional[int] = None
    resolved_page_table_affine_second_base: Optional[int] = None
    resolved_page_table_affine_second_stride: Optional[int] = None
    resolved_page_table_affine_batch_stride: Optional[int] = None
    resolved_page_table_affine_head_stride: int = 0
    resolved_page_table_affine_direct: bool = False
    resolved_page_table_affine_cols: int = 0
    resolved_seqused_k_by_head_i32: Optional[torch.Tensor] = None
    capture_row_index_i32: Optional[torch.Tensor] = None
    row_capture_last_n_i32: Optional[torch.Tensor] = None
    max_capture_k: int = 0
    max_capture_last_n: int = 0
    page_resolver_kind: int = PAGE_RESOLVER_KIND_NATIVE
    page_resolver_subkind: int = PAGE_RESOLVER_SUBKIND_ROWPTR
    compact_base_page_i32: Optional[torch.Tensor] = None
    compact_page_count_i32: Optional[torch.Tensor] = None
    recent_first_logical_page_i32: Optional[torch.Tensor] = None
    resolved_page_table_i32: Optional[torch.Tensor] = None
    graph_replay_carriers: bool = False
    resolver_descriptor: Optional[ResolverGraphDescriptor] = None
    resolver_carriers: Optional[MixedPageResolverCarrierSet] = None


def validate_mixed_page_resolver_replay(
    *,
    enabled: bool,
    captured_descriptor: ResolverGraphDescriptor | None,
    replay_descriptor: ResolverGraphDescriptor | None,
    captured_pointer_signature: tuple[int | None, ...] | None,
    replay_carriers: MixedPageResolverCarrierSet | None,
) -> None:
    if not bool(enabled):
        return
    if replay_carriers is None:
        raise ValueError("replay resolver carriers are required for replay")
    if captured_descriptor is None:
        raise ValueError("captured resolver graph descriptor is required for replay")
    if replay_descriptor is None:
        raise ValueError("replay resolver graph descriptor is required for replay")
    if captured_pointer_signature is None:
        raise ValueError("captured resolver carrier pointer signature is required for replay")
    assert_descriptor_matches_replay(captured_descriptor, replay_descriptor)
    assert_carrier_addresses_stable(
        captured_pointer_signature,
        replay_carriers.pointer_signature(),
    )
