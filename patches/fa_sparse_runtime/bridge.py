from __future__ import annotations

from typing import Optional, Sequence

import torch

from .contracts import (
    FASparseLaunchDecision,
    PAGE_TABLE_LAYOUT_KV_HEAD_FLAT,
)


def _normalize_layout(page_table_layout: int | str) -> int:
    if page_table_layout in (PAGE_TABLE_LAYOUT_KV_HEAD_FLAT, "KV_HEAD_FLAT"):
        return PAGE_TABLE_LAYOUT_KV_HEAD_FLAT
    raise ValueError("page_table_layout must be KV_HEAD_FLAT")


def build_fa_sparse_launch_inputs(
    *,
    final_page_table_i32: torch.Tensor,
    real_kv_len_i32: torch.Tensor,
    selected_seqused_k_by_head_i32: torch.Tensor,
    batch_size: int,
    num_kv_heads: int,
    page_table_layout: int | str = PAGE_TABLE_LAYOUT_KV_HEAD_FLAT,
    kv_batch_idx_i32: Optional[torch.Tensor] = None,
    request_sparse_eligible: Optional[Sequence[bool]] = None,
    per_head_page_counts_i32: Optional[torch.Tensor] = None,
    contract_mode: bool = False,
    applied_recent_epoch_i32: Optional[torch.Tensor] = None,
    request_recent_epoch_i32: Optional[torch.Tensor] = None,
    applied_refresh_generation_i32: Optional[torch.Tensor] = None,
    request_refresh_generation_i32: Optional[torch.Tensor] = None,
    materialize_status_i32: Optional[torch.Tensor] = None,
    patch_status_i32: Optional[torch.Tensor] = None,
    cached_lengths_ok: Optional[bool] = None,
    cached_kv_batch_idx_identity: Optional[bool] = None,
    cached_status_ok: Optional[bool] = None,
    cached_freshness_ok: Optional[bool] = None,
) -> dict[str, object]:
    layout = _normalize_layout(page_table_layout)
    del per_head_page_counts_i32
    if batch_size <= 0 or num_kv_heads <= 0:
        raise ValueError("batch_size and num_kv_heads must be positive")
    if final_page_table_i32.dim() != 2:
        raise ValueError("final_page_table_i32 must be rank-2")
    expected_rows = int(batch_size) * int(num_kv_heads)
    real_kv_len_i32 = real_kv_len_i32.to(
        device=final_page_table_i32.device,
        dtype=torch.int32,
    ).reshape(int(batch_size))
    selected_seqused_k_by_head_i32 = selected_seqused_k_by_head_i32.to(
        device=final_page_table_i32.device,
        dtype=torch.int32,
    ).reshape(expected_rows)
    if bool((selected_seqused_k_by_head_i32 < 0).any().item()):
        raise ValueError("selected_seqused_k_by_head_i32 must be non-negative")
    selected_seqused_k_by_head_by_row = selected_seqused_k_by_head_i32.reshape(
        int(batch_size),
        int(num_kv_heads),
    )
    selected_lengths_ok = not bool(
        (selected_seqused_k_by_head_by_row > real_kv_len_i32[:, None]).any().item()
    )
    if not selected_lengths_ok:
        raise ValueError("selected_seqused_k_by_head_i32 must not exceed real_kv_len_i32")

    eligible = (
        [True] * int(batch_size)
        if request_sparse_eligible is None
        else [bool(flag) for flag in request_sparse_eligible]
    )
    if len(eligible) != int(batch_size):
        raise ValueError("request_sparse_eligible must match batch_size")
    if not all(eligible):
        raise ValueError("planned selected rows must stay sparse-eligible")

    if kv_batch_idx_i32 is None:
        kv_batch_idx_i32 = torch.arange(
            int(batch_size),
            device=final_page_table_i32.device,
            dtype=torch.int32,
        )
    else:
        kv_batch_idx_i32 = kv_batch_idx_i32.to(
            device=final_page_table_i32.device,
            dtype=torch.int32,
        ).reshape(-1)
    if kv_batch_idx_i32.numel() != int(batch_size):
        raise ValueError("kv_batch_idx_i32 must match batch_size")

    if layout == PAGE_TABLE_LAYOUT_KV_HEAD_FLAT:
        if final_page_table_i32.shape[0] != expected_rows:
            raise ValueError("page_table row count does not match KV_HEAD_FLAT layout")
        expected_kv = torch.arange(
            int(batch_size),
            device=final_page_table_i32.device,
            dtype=torch.int32,
        )
        kv_batch_idx_is_identity = bool((kv_batch_idx_i32 == expected_kv).all().item())
        if not kv_batch_idx_is_identity:
            raise ValueError("selected sparse launch requires identity kv_batch_idx")
    if materialize_status_i32 is not None:
        materialize_status_i32 = materialize_status_i32.to(
            device=final_page_table_i32.device,
            dtype=torch.int32,
        ).reshape(int(batch_size))
    if patch_status_i32 is not None:
        patch_status_i32 = patch_status_i32.to(
            device=final_page_table_i32.device,
            dtype=torch.int32,
        ).reshape(int(batch_size))
    cached_status_is_ok = not (
        (
            materialize_status_i32 is not None
            and bool((materialize_status_i32 != 0).any().item())
        ) or (
            patch_status_i32 is not None
            and bool((patch_status_i32 != 0).any().item())
        )
    )
    if not cached_status_is_ok:
        raise ValueError("cached carrier status reports recoverable sparse miss")

    if (
        applied_recent_epoch_i32 is not None
        and request_recent_epoch_i32 is not None
        and applied_refresh_generation_i32 is not None
        and request_refresh_generation_i32 is not None
    ):
        applied_recent_epoch_i32 = applied_recent_epoch_i32.to(
            device=final_page_table_i32.device,
            dtype=torch.int32,
        ).reshape(int(batch_size))
        request_recent_epoch_i32 = request_recent_epoch_i32.to(
            device=final_page_table_i32.device,
            dtype=torch.int32,
        ).reshape(int(batch_size))
        applied_refresh_generation_i32 = applied_refresh_generation_i32.to(
            device=final_page_table_i32.device,
            dtype=torch.int32,
        ).reshape(int(batch_size))
        request_refresh_generation_i32 = request_refresh_generation_i32.to(
            device=final_page_table_i32.device,
            dtype=torch.int32,
        ).reshape(int(batch_size))
        freshness_matches = bool(
            (applied_recent_epoch_i32 == request_recent_epoch_i32).all().item()
            and (applied_refresh_generation_i32 == request_refresh_generation_i32).all().item()
        )
        if not freshness_matches:
            raise ValueError("cached carrier freshness mismatch")

    return {
        "decision": FASparseLaunchDecision.USE_SPARSE,
        "use_sparse": True,
        "page_table_i32": final_page_table_i32,
        "selected_seqused_k_by_head_i32": selected_seqused_k_by_head_i32,
        "real_kv_len_i32": real_kv_len_i32,
        "page_table_layout": int(layout),
        "kv_batch_idx_i32": kv_batch_idx_i32,
    }
