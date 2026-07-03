from __future__ import annotations

import os
from typing import Mapping, Sequence, Tuple, TYPE_CHECKING

import torch

from patches.cpu_gpu_staging import cached_sequence_to_device

from .bridge import build_fa_sparse_launch_inputs
from .contracts import (
    PAGE_TABLE_LAYOUT_KV_HEAD_FLAT,
)
from .materialize import build_materialized_recent_descriptor, compute_visible_kv_len

if TYPE_CHECKING:
    from patches.sparse_types import StepMeta


def _selected_ready_trace_enabled() -> bool:
    return bool(os.environ.get("VLLM_SPARSE_SELECTED_READY_TRACE_LOG", "").strip())


def _ceil_div(value: int, divisor: int) -> int:
    return (int(value) + int(divisor) - 1) // int(divisor)


def _identity_kv_rows(batch_size: int) -> tuple[int, ...]:
    return tuple(range(int(batch_size)))


def _current_real_kv_lens_cpu(
    *,
    step_meta: "StepMeta",
    batch_size: int,
) -> tuple[int, ...]:
    values = tuple(int(v) for v in step_meta.canonical_real_kv_len_cpu[:batch_size])
    if len(values) != int(batch_size):
        raise RuntimeError(
            "canonical real-kv CPU carrier must cover batch_size; "
            f"truth={len(values)} batch_size={int(batch_size)}"
        )
    return values


def _ensure_or_refresh_i32_buffer(
    buf: torch.Tensor | None,
    values: Sequence[int],
    *,
    device: torch.device,
    stage_cache: dict[str, object] | None = None,
    cache_name: str = "i32",
) -> torch.Tensor:
    return cached_sequence_to_device(
        tuple(int(v) for v in values),
        dtype=torch.int32,
        device=device,
        cache_name=cache_name,
        stage_cache=stage_cache,
        out=buf,
    )




def ensure_step_recent_descriptors(
    *,
    step_meta: "StepMeta",
    page_size: int,
    layer_effective_refresh_by_row: Tuple[bool, ...],
) -> None:
    """Build / refresh step-level recent descriptor.

    Refresh row full-KV degrade (spec 2026-04-24 §2): refresh rows always
    cover full KV by passing `sink_page_slots=0` and
    `recent_tokens=real_kv_len` to the builder. Builder math naturally yields
    `first=0, count=ceil(real_kv_len/page_size)`. No builder API change, no
    new branch inside the builder.
    """
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    batch_size = int(step_meta.batch_size)
    if len(layer_effective_refresh_by_row) < batch_size:
        raise ValueError(
            f"layer_effective_refresh_by_row shorter than batch_size: "
            f"rows={len(layer_effective_refresh_by_row)} batch={batch_size}"
        )
    if (
        int(step_meta.recent_descriptor_block_size) == int(page_size)
        and int(step_meta.request_recent_epoch) == int(step_meta.epoch)
        and len(step_meta.request_recent_first_logical_page) == batch_size
        and len(step_meta.request_recent_page_count) == batch_size
        and len(step_meta.request_kv_rows) == batch_size
    ):
        return

    sink_page_slots = _ceil_div(int(step_meta.sink_tokens), int(page_size))
    recent_page_slots = _ceil_div(int(step_meta.recent_cap), int(page_size))
    real_kv_lens_cpu = _current_real_kv_lens_cpu(
        step_meta=step_meta,
        batch_size=batch_size,
    )

    refresh_by_row = tuple(bool(v) for v in layer_effective_refresh_by_row[:batch_size])

    if recent_page_slots <= 0:
        # Contract rejection: refresh requires a non-zero recent budget because
        # the degrade path targets full-KV coverage through the recent channel.
        if any(refresh_by_row):
            raise RuntimeError(
                "refresh row full-KV degrade requires recent_page_slots > 0; "
                f"got step_meta.recent_cap={step_meta.recent_cap}, "
                f"page_size={page_size}"
            )
        step_meta.recent_descriptor_block_size = int(page_size)
        step_meta.request_recent_first_logical_page = (0,) * batch_size
        step_meta.request_recent_page_count = (0,) * batch_size
        step_meta.request_recent_epoch = int(step_meta.epoch)
        step_meta.request_kv_rows = _identity_kv_rows(batch_size)
        return

    first_pages: list[int] = []
    page_counts: list[int] = []
    for row_idx, real_kv_len in enumerate(real_kv_lens_cpu):
        real_kv_len_i = int(real_kv_len)
        if real_kv_len_i <= 0:
            first_pages.append(0)
            page_counts.append(0)
            continue
        # Per-row gate: refresh row -> substitute sink=0 and
        # recent_tokens=real_kv_len. Builder math produces first=0,
        # count=ceil(real_kv_len/page_size).
        row_degrade = refresh_by_row[row_idx]
        eff_sink_page_slots = 0 if row_degrade else int(sink_page_slots)
        eff_recent_tokens = real_kv_len_i if row_degrade else int(step_meta.recent_cap)

        descriptor = build_materialized_recent_descriptor(
            real_kv_len=real_kv_len_i,
            page_size=int(page_size),
            sink_page_slots=eff_sink_page_slots,
            recent_page_slots=int(recent_page_slots),
            recent_tokens=eff_recent_tokens,
            epoch=int(step_meta.epoch),
        )
        first_pages.append(int(descriptor.materialized_first_logical_page))
        page_counts.append(int(descriptor.materialized_page_count))

    step_meta.recent_descriptor_block_size = int(page_size)
    step_meta.request_recent_first_logical_page = tuple(first_pages)
    step_meta.request_recent_page_count = tuple(page_counts)
    step_meta.request_recent_epoch = int(step_meta.epoch)
    step_meta.request_kv_rows = _identity_kv_rows(batch_size)


def ensure_step_request_sparse_tensors(
    *,
    step_meta: "StepMeta",
    page_size: int,
    device: torch.device,
    layer_effective_refresh_by_row: Tuple[bool, ...],
) -> dict[str, torch.Tensor]:
    ensure_step_recent_descriptors(
        step_meta=step_meta,
        page_size=page_size,
        layer_effective_refresh_by_row=layer_effective_refresh_by_row,
    )

    batch_size = int(step_meta.batch_size)
    stage_cache = getattr(step_meta, "small_tensor_stage", None)

    # Rev 2 (2026-04-23 post-ship): O(1) fast-path keyed on step_meta's own
    # request_recent_epoch_i32_gpu_value marker. The producer side advances
    # step_meta.request_recent_epoch when descriptors change; populate below
    # writes request_recent_epoch_i32_gpu_value = request_recent_epoch after
    # syncing GPU tensors. Equality therefore means "GPU tensors currently
    # reflect this epoch". When step_meta is a fresh instance, both fields
    # default to -1 so the extra tensor-non-None + shape/device guards force
    # a populate. This keys on data-owner invariant, not object identity,
    # following sfi-release's pattern; no external cache, no H2D per layer
    # after the first. StepMeta slots safety: all accessed attributes are
    # declared fields.
    if (
        step_meta.request_recent_epoch_i32_gpu_value == int(step_meta.request_recent_epoch)
        and step_meta.request_recent_first_logical_page_i32_gpu is not None
        and step_meta.request_recent_page_count_i32_gpu is not None
        and step_meta.request_recent_epoch_i32_gpu is not None
        and step_meta.request_kv_rows_i32_gpu is not None
        and step_meta.request_recent_first_logical_page_i32_gpu.shape[0] == batch_size
        and step_meta.request_recent_first_logical_page_i32_gpu.device == device
    ):
        return {
            "request_recent_first_logical_page_i32": step_meta.request_recent_first_logical_page_i32_gpu,
            "request_recent_page_count_i32": step_meta.request_recent_page_count_i32_gpu,
            "request_recent_epoch_i32": step_meta.request_recent_epoch_i32_gpu,
            "request_kv_rows_i32": step_meta.request_kv_rows_i32_gpu,
        }

    step_meta.request_recent_first_logical_page_i32_gpu = _ensure_or_refresh_i32_buffer(
        step_meta.request_recent_first_logical_page_i32_gpu,
        step_meta.request_recent_first_logical_page[:batch_size],
        device=device,
        stage_cache=stage_cache,
        cache_name="request_recent_first",
    )
    step_meta.request_recent_page_count_i32_gpu = _ensure_or_refresh_i32_buffer(
        step_meta.request_recent_page_count_i32_gpu,
        step_meta.request_recent_page_count[:batch_size],
        device=device,
        stage_cache=stage_cache,
        cache_name="request_recent_count",
    )
    if (
        step_meta.request_recent_epoch_i32_gpu is None
        or step_meta.request_recent_epoch_i32_gpu.device != device
        or step_meta.request_recent_epoch_i32_gpu.shape[0] != batch_size
    ):
        step_meta.request_recent_epoch_i32_gpu = torch.empty(
            (batch_size,),
            dtype=torch.int32,
            device=device,
        )
    step_meta.request_recent_epoch_i32_gpu.fill_(int(step_meta.request_recent_epoch))
    step_meta.request_recent_epoch_i32_gpu_value = int(step_meta.request_recent_epoch)
    step_meta.request_kv_rows_i32_gpu = _ensure_or_refresh_i32_buffer(
        step_meta.request_kv_rows_i32_gpu,
        step_meta.request_kv_rows[:batch_size],
        device=device,
        stage_cache=stage_cache,
        cache_name="request_kv_rows",
    )

    return {
        "request_recent_first_logical_page_i32": step_meta.request_recent_first_logical_page_i32_gpu,
        "request_recent_page_count_i32": step_meta.request_recent_page_count_i32_gpu,
        "request_recent_epoch_i32": step_meta.request_recent_epoch_i32_gpu,
        "request_kv_rows_i32": step_meta.request_kv_rows_i32_gpu,
    }


def build_layer_page_sparse_metadata(
    *,
    step_meta: "StepMeta",
    request_id_to_slot: Mapping[str, int],
    selected_middle_pages_by_slot: Sequence[torch.Tensor],
    selected_middle_counts_by_slot: Sequence[torch.Tensor],
    refresh_generation_by_slot: Sequence[int],
    refresh_by_row: Tuple[bool, ...],
    selected_middle_uniform_counts_cpu: Sequence[int] | None = None,
    selected_middle_min_logical_page_cpu: Sequence[int] | None = None,
    selected_middle_max_logical_page_cpu: Sequence[int] | None = None,
    request_selected_rows: Sequence[bool] | None = None,
    num_kv_heads: int,
    page_size: int,
    device: torch.device,
    min_selected_middle_pages: int = 2,
    min_recent_pages: int = 2,
) -> dict[str, object]:
    del selected_middle_uniform_counts_cpu
    del selected_middle_min_logical_page_cpu
    del selected_middle_max_logical_page_cpu

    if num_kv_heads <= 0:
        raise ValueError("num_kv_heads must be positive")

    # Spec 2026-04-24 §3.3.2: legacy page_sparse path does NOT support refresh
    # row full-KV degrade. This path is default-off (_SKIP_PAGE_SPARSE_STATE_CACHED
    # defaults True). If any refresh row reaches here, the K-view contract is
    # broken -- raise rather than silently produce wrong descriptors.
    batch_size = int(step_meta.batch_size)
    if any(refresh_by_row[:batch_size]):
        raise RuntimeError(
            "page_sparse legacy path does not support refresh row full-KV degrade; "
            "refresh is handled by compact_recent mode. "
            "Check VLLM_SPARSE_SKIP_PAGE_SPARSE_STATE env and attn_mode."
        )

    # Refresh rows already raised above; pass all-False to the descriptor helper.
    ensure_step_recent_descriptors(
        step_meta=step_meta,
        page_size=page_size,
        layer_effective_refresh_by_row=(False,) * batch_size,
    )
    real_kv_len_i32 = step_meta.canonical_real_kv_len_i32_gpu
    if (
        not isinstance(real_kv_len_i32, torch.Tensor)
        or real_kv_len_i32.device != device
        or real_kv_len_i32.dtype != torch.int32
        or real_kv_len_i32.dim() != 1
        or int(real_kv_len_i32.numel()) != int(batch_size)
    ):
        raise RuntimeError(
            "runtime_cache requires canonical real-kv GPU carrier; "
            f"batch_size={int(batch_size)}"
        )
    real_kv_lens_cpu = _current_real_kv_lens_cpu(
        step_meta=step_meta,
        batch_size=batch_size,
    )

    selected_summary_k_cpu: list[int] = [0] * batch_size
    selected_page_count_summary_cpu: list[int] = [0] * batch_size
    applied_refresh_generation_cpu: list[int] = [0] * batch_size
    per_head_page_counts_cpu: list[list[int]] = [
        [0] * int(num_kv_heads) for _ in range(batch_size)
    ]
    selected_seqused_k_by_head_cpu: list[list[int]] = [
        [0] * int(num_kv_heads) for _ in range(batch_size)
    ]
    selected_rows = (
        [True] * batch_size
        if request_selected_rows is None
        else [bool(v) for v in request_selected_rows]
    )
    if len(selected_rows) != batch_size:
        raise ValueError("request_selected_rows must match batch_size")

    request_sparse_eligible = [False] * batch_size
    request_sparse_debug_rows: list[dict[str, object]] | None = (
        [{} for _ in range(batch_size)] if _selected_ready_trace_enabled() else None
    )
    max_page_count = 0
    sink_page_slots = _ceil_div(int(step_meta.sink_tokens), int(page_size))
    recent_page_slots = _ceil_div(int(step_meta.recent_cap), int(page_size))
    middle_col_i32 = None

    for row, req_id in enumerate(step_meta.req_ids[:batch_size]):
        debug_row = request_sparse_debug_rows[row] if request_sparse_debug_rows is not None else None
        if debug_row is not None:
            debug_row["selected_row"] = bool(selected_rows[row])
            debug_row["req_id"] = str(req_id)
        if not bool(selected_rows[row]):
            if debug_row is not None:
                debug_row["reason"] = "selected_row_false"
            continue

        slot = int(request_id_to_slot.get(req_id, -1))
        if debug_row is not None:
            debug_row["slot"] = int(slot)
        if (
            slot < 0
            or slot >= len(selected_middle_pages_by_slot)
            or slot >= len(selected_middle_counts_by_slot)
            or slot >= len(refresh_generation_by_slot)
        ):
            if debug_row is not None:
                debug_row["reason"] = "slot_binding_invalid"
            continue

        refresh_generation = int(refresh_generation_by_slot[slot])
        if debug_row is not None:
            debug_row["refresh_generation"] = int(refresh_generation)
        if int(refresh_generation) <= 0:
            if debug_row is not None:
                debug_row["reason"] = "refresh_generation_nonpositive"
            continue

        middle_pages = selected_middle_pages_by_slot[slot]
        middle_counts = selected_middle_counts_by_slot[slot]
        if middle_pages.dim() != 2 or int(middle_pages.shape[0]) != int(num_kv_heads):
            if debug_row is not None:
                debug_row["reason"] = "middle_pages_shape_invalid"
            continue
        middle_counts_i32 = middle_counts.reshape(-1).to(dtype=torch.int32)
        if int(middle_counts_i32.numel()) != int(num_kv_heads):
            if debug_row is not None:
                debug_row["reason"] = "middle_counts_shape_invalid"
            continue
        if bool(torch.any(middle_counts_i32 < 0).item()):
            if debug_row is not None:
                debug_row["reason"] = "middle_counts_negative"
            continue
        middle_width = int(middle_pages.shape[1])
        if bool(torch.any(middle_counts_i32 > int(middle_width)).item()):
            if debug_row is not None:
                debug_row["reason"] = "middle_counts_exceed_width"
            continue

        max_middle_count = int(middle_counts_i32.max().item()) if int(num_kv_heads) > 0 else 0
        if int(max_middle_count) < int(min_selected_middle_pages):
            if debug_row is not None:
                debug_row["reason"] = "middle_page_count_too_small"
            continue

        real_kv_len = int(real_kv_lens_cpu[row])
        if debug_row is not None:
            debug_row["real_kv_len"] = int(real_kv_len)
        if int(real_kv_len) <= 0 or int(page_size) <= 0 or int(recent_page_slots) <= 0:
            if debug_row is not None:
                debug_row["reason"] = "request_geometry_invalid"
            continue

        recent_descriptor = build_materialized_recent_descriptor(
            real_kv_len=int(real_kv_len),
            page_size=int(page_size),
            sink_page_slots=int(sink_page_slots),
            recent_page_slots=int(recent_page_slots),
            recent_tokens=int(step_meta.recent_cap),
        )
        recent_page_count = int(recent_descriptor.materialized_page_count)
        recent_first_logical_page = int(recent_descriptor.materialized_first_logical_page)
        if debug_row is not None:
            debug_row["recent_first_logical_page"] = int(recent_first_logical_page)
            debug_row["recent_page_count"] = int(recent_page_count)
        if int(recent_page_count) < int(min_recent_pages):
            if debug_row is not None:
                debug_row["reason"] = "recent_page_count_too_small"
            continue

        middle_pages_i32 = middle_pages.to(device=device, dtype=torch.int32)
        if middle_col_i32 is None or int(middle_col_i32.numel()) != int(middle_width):
            middle_col_i32 = torch.arange(int(middle_width), device=device, dtype=torch.int32)
        middle_prefix_mask = middle_col_i32.view(1, int(middle_width)) < middle_counts_i32.to(
            device=device,
            dtype=torch.int32,
        ).view(int(num_kv_heads), 1)
        if bool(torch.any((middle_pages_i32 < 0) & middle_prefix_mask).item()):
            if debug_row is not None:
                debug_row["reason"] = "middle_prefix_has_negative_page"
            continue

        valid_middle_pages = middle_pages_i32[middle_prefix_mask]
        if valid_middle_pages.numel() > 0:
            if bool(torch.any(valid_middle_pages < int(sink_page_slots)).item()):
                if debug_row is not None:
                    debug_row["reason"] = "middle_overlaps_sink"
                continue
            if bool(torch.any(valid_middle_pages >= int(recent_first_logical_page)).item()):
                if debug_row is not None:
                    debug_row["reason"] = "middle_overlaps_recent"
                continue
            if debug_row is not None:
                valid_middle_pages_cpu = valid_middle_pages.detach().cpu()
                debug_row["min_logical_page"] = int(valid_middle_pages_cpu.min().item())
                debug_row["max_logical_page"] = int(valid_middle_pages_cpu.max().item())
        elif debug_row is not None:
            debug_row["min_logical_page"] = -1
            debug_row["max_logical_page"] = -1

        counts_cpu = [int(v) for v in middle_counts_i32.detach().cpu().tolist()]
        per_head_page_counts_row = [
            int(sink_page_slots) + int(count) + int(recent_page_count)
            for count in counts_cpu
        ]
        per_head_selected_k_row = [
            int(
                compute_visible_kv_len(
                    page_size=int(page_size),
                    sink_page_slots=int(sink_page_slots),
                    middle_page_slots=int(count),
                    recent_descriptor=recent_descriptor,
                )
            )
            for count in counts_cpu
        ]
        selected_summary_k_cpu[row] = max(per_head_selected_k_row) if per_head_selected_k_row else 0
        selected_page_count_summary_cpu[row] = max(per_head_page_counts_row) if per_head_page_counts_row else 0
        per_head_page_counts_cpu[row] = per_head_page_counts_row
        selected_seqused_k_by_head_cpu[row] = per_head_selected_k_row
        max_page_count = max(int(max_page_count), int(selected_page_count_summary_cpu[row]))
        applied_refresh_generation_cpu[row] = int(refresh_generation)
        request_sparse_eligible[row] = True
        if debug_row is not None:
            debug_row["reason"] = "eligible"
            debug_row["selected_page_count"] = int(selected_page_count_summary_cpu[row])
            debug_row["visible_kv_len"] = int(selected_summary_k_cpu[row])
            debug_row["per_head_page_count"] = [int(v) for v in per_head_page_counts_row]
            debug_row["selected_seqused_k_by_head"] = [int(v) for v in per_head_selected_k_row]

    if request_selected_rows is None:
        use_sparse = (
            all(request_sparse_eligible)
            and tuple(int(v) for v in step_meta.request_kv_rows[:batch_size])
            == _identity_kv_rows(batch_size)
        )
    else:
        use_sparse = (
            any(selected_rows)
            and all(
                (not bool(selected_rows[row])) or bool(request_sparse_eligible[row])
                for row in range(batch_size)
            )
            and tuple(int(v) for v in step_meta.request_kv_rows[:batch_size])
            == _identity_kv_rows(batch_size)
        )

    # Refresh rows already raised at function entry; pass all-False here.
    request_sparse_tensors = ensure_step_request_sparse_tensors(
        step_meta=step_meta,
        page_size=page_size,
        device=device,
        layer_effective_refresh_by_row=(False,) * batch_size,
    )
    kv_batch_idx_i32 = request_sparse_tensors["request_kv_rows_i32"]
    selected_summary_k_i32 = torch.tensor(
        tuple(int(v) for v in selected_summary_k_cpu),
        dtype=torch.int32,
        device=device,
    )
    selected_page_count_summary_i32 = torch.tensor(
        tuple(int(v) for v in selected_page_count_summary_cpu),
        dtype=torch.int32,
        device=device,
    )
    per_head_page_counts_i32 = torch.tensor(
        tuple(tuple(int(v) for v in row) for row in per_head_page_counts_cpu),
        dtype=torch.int32,
        device=device,
    )
    selected_seqused_k_by_head_i32 = torch.tensor(
        tuple(tuple(int(v) for v in row) for row in selected_seqused_k_by_head_cpu),
        dtype=torch.int32,
        device=device,
    )
    applied_refresh_generation_i32 = torch.tensor(
        tuple(int(v) for v in applied_refresh_generation_cpu),
        dtype=torch.int32,
        device=device,
    )
    return {
        "use_sparse": bool(use_sparse),
        "request_sparse_eligible": tuple(request_sparse_eligible),
        "selected_summary_k_i32": selected_summary_k_i32,
        "real_kv_len_i32": real_kv_len_i32,
        "selected_page_count_summary_i32": selected_page_count_summary_i32,
        "per_head_page_counts_i32": per_head_page_counts_i32,
        "selected_seqused_k_by_head_i32": selected_seqused_k_by_head_i32,
        "kv_batch_idx_i32": kv_batch_idx_i32,
        "cached_lengths_ok": all(
            int(selected_summary_k_cpu[row]) <= int(real_kv_lens_cpu[row])
            for row in range(batch_size)
        ),
        "cached_kv_batch_idx_identity": True,
        "page_table_layout": int(PAGE_TABLE_LAYOUT_KV_HEAD_FLAT),
        "applied_refresh_generation_i32": applied_refresh_generation_i32,
        "request_refresh_generation_signature": tuple(
            int(v) for v in applied_refresh_generation_cpu
        ),
        "request_recent_epoch_value": int(step_meta.request_recent_epoch),
        "max_page_count": int(max_page_count),
        "request_sparse_debug_rows": tuple(request_sparse_debug_rows)
        if request_sparse_debug_rows is not None
        else tuple(),
    }


def build_layer_page_sparse_launch(
    *,
    step_meta: "StepMeta",
    request_id_to_slot: Mapping[str, int],
    selected_middle_pages_by_slot: Sequence[torch.Tensor],
    selected_middle_counts_by_slot: Sequence[torch.Tensor],
    refresh_generation_by_slot: Sequence[int],
    refresh_by_row: Tuple[bool, ...],
    block_table: torch.Tensor,
    num_kv_heads: int,
    page_size: int,
    device: torch.device,
    min_selected_middle_pages: int = 2,
    min_recent_pages: int = 2,
    contract_mode: bool = False,
) -> dict[str, object]:
    metadata = build_layer_page_sparse_metadata(
        step_meta=step_meta,
        request_id_to_slot=request_id_to_slot,
        selected_middle_pages_by_slot=selected_middle_pages_by_slot,
        selected_middle_counts_by_slot=selected_middle_counts_by_slot,
        refresh_generation_by_slot=refresh_generation_by_slot,
        refresh_by_row=refresh_by_row,
        num_kv_heads=num_kv_heads,
        page_size=page_size,
        device=device,
        min_selected_middle_pages=min_selected_middle_pages,
        min_recent_pages=min_recent_pages,
    )
    batch_size = int(step_meta.batch_size)
    if not bool(metadata["use_sparse"]):
        raise ValueError("planned selected rows must stay sparse-eligible")

    if block_table.dim() != 2 or block_table.shape[0] < batch_size:
        raise ValueError("block_table must be rank-2 and cover all batch rows")
    block_table_i32 = block_table.to(device=device, dtype=torch.int32)
    sink_page_slots = _ceil_div(int(step_meta.sink_tokens), int(page_size))
    request_sparse_tensors = ensure_step_request_sparse_tensors(
        step_meta=step_meta,
        page_size=page_size,
        device=device,
        # Refresh rows already raised in build_layer_page_sparse_metadata above.
        layer_effective_refresh_by_row=(False,) * batch_size,
    )
    max_page_count = int(metadata["selected_page_count_summary_i32"].max().item())
    final_page_table_i32 = torch.empty(
        (int(batch_size) * int(num_kv_heads), int(max_page_count)),
        dtype=torch.int32,
        device=device,
    )

    per_head_page_counts_i32 = metadata["per_head_page_counts_i32"].to(
        device=device,
        dtype=torch.int32,
    )
    for row, req_id in enumerate(step_meta.req_ids[:batch_size]):
        slot = int(request_id_to_slot.get(req_id, -1))
        row_block_table = block_table_i32[row]
        middle_pages = selected_middle_pages_by_slot[slot].to(dtype=torch.int32, device=device)
        middle_counts = selected_middle_counts_by_slot[slot].reshape(-1).to(
            device=device,
            dtype=torch.int32,
        )
        recent_first_logical_page = int(
            request_sparse_tensors["request_recent_first_logical_page_i32"][row].item()
        )
        recent_page_count = int(
            request_sparse_tensors["request_recent_page_count_i32"][row].item()
        )
        sink_physical_pages = row_block_table[: int(sink_page_slots)]
        recent_physical_pages = row_block_table[
            int(recent_first_logical_page): int(recent_first_logical_page) + int(recent_page_count)
        ]
        for kv_head in range(int(num_kv_heads)):
            middle_count = int(middle_counts[kv_head].item())
            total_page_count = int(per_head_page_counts_i32[row, kv_head].item())
            middle_logical_pages = middle_pages[kv_head, :middle_count].to(dtype=torch.long)
            middle_physical_pages = row_block_table.index_select(0, middle_logical_pages)
            row_pages = torch.cat(
                (
                    sink_physical_pages,
                    middle_physical_pages.to(dtype=torch.int32),
                    recent_physical_pages,
                ),
                dim=0,
            ).contiguous()
            flat_row = int(row) * int(num_kv_heads) + int(kv_head)
            final_page_table_i32[flat_row, :total_page_count].copy_(row_pages)
            if total_page_count < max_page_count and total_page_count > 0:
                final_page_table_i32[flat_row, total_page_count:].copy_(
                    row_pages[-1].expand(max_page_count - total_page_count)
                )

    launch = build_fa_sparse_launch_inputs(
        final_page_table_i32=final_page_table_i32,
        real_kv_len_i32=metadata["real_kv_len_i32"],
        selected_seqused_k_by_head_i32=metadata["selected_seqused_k_by_head_i32"].reshape(
            int(batch_size) * int(num_kv_heads)
        ),
        batch_size=batch_size,
        num_kv_heads=num_kv_heads,
        page_table_layout=metadata["page_table_layout"],
        kv_batch_idx_i32=metadata["kv_batch_idx_i32"],
        request_sparse_eligible=metadata["request_sparse_eligible"],
        contract_mode=contract_mode,
    )
    return {
        **launch,
        **metadata,
        "page_table_i32": launch["page_table_i32"],
        "applied_recent_epoch_i32": torch.full(
            (batch_size,),
            int(step_meta.request_recent_epoch),
            dtype=torch.int32,
            device=device,
        ),
    }
