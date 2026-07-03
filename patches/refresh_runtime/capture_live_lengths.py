from __future__ import annotations

from typing import Sequence, Tuple

import torch

from patches.cpu_gpu_staging import cached_sequence_to_device
from patches.sparse_types import StepCaptureLayout


def _tensor_live_key(tensor: torch.Tensor | None) -> Tuple[object, ...]:
    if not isinstance(tensor, torch.Tensor):
        return ("none",)
    device = tensor.device
    return (
        int(tensor.data_ptr()),
        str(device.type),
        -1 if device.index is None else int(device.index),
        str(tensor.dtype),
        tuple(int(v) for v in tensor.shape),
        tuple(int(v) for v in tensor.stride()),
    )


def _cpu_long_tensor_from_values(
    existing: torch.Tensor | None,
    values: Sequence[int],
) -> torch.Tensor:
    values_tuple = tuple(max(0, int(v)) for v in values)
    if (
        isinstance(existing, torch.Tensor)
        and existing.device.type == "cpu"
        and existing.dtype == torch.long
        and int(existing.numel()) == len(values_tuple)
    ):
        existing.copy_(torch.tensor(values_tuple, dtype=torch.long))
        return existing
    return torch.tensor(values_tuple, dtype=torch.long)


def get_step_plan_cap_tensor(
    *,
    controller: object,
    bound_meta: object,
    plan_cap_by_row: Sequence[int],
    device: torch.device,
) -> torch.Tensor:
    """Return the step-level logits capacity tensor cached by bound-meta signature."""
    dev_idx = int(device.index) if device.index is not None else -1
    dev_key = (str(device.type), dev_idx)
    bound_sig = tuple(getattr(bound_meta, "bound_meta_signature", tuple()))
    cache_epoch = int(getattr(controller, "_capture_plan_cap_tensor_epoch", -1))
    cache_dev = getattr(controller, "_capture_plan_cap_tensor_device", ("", -1))
    cache_sig = tuple(getattr(controller, "_capture_plan_cap_tensor_signature", tuple()))
    cache_tensor = getattr(controller, "_capture_plan_cap_tensor", None)
    if (
        not isinstance(cache_tensor, torch.Tensor)
        or cache_epoch != int(getattr(bound_meta, "epoch", -1))
        or cache_dev != dev_key
        or cache_sig != bound_sig
        or cache_tensor.device != device
        or cache_tensor.dtype != torch.long
        or int(cache_tensor.numel()) < len(plan_cap_by_row)
    ):
        cache_tensor = cached_sequence_to_device(
            tuple(int(v) for v in plan_cap_by_row),
            device=device,
            dtype=torch.long,
            cache_name="capture_plan_cap_i64",
            cache_owner=controller,
            out=cache_tensor if isinstance(cache_tensor, torch.Tensor) else None,
        )
        setattr(controller, "_capture_plan_cap_tensor", cache_tensor)
        setattr(controller, "_capture_plan_cap_tensor_epoch", int(getattr(bound_meta, "epoch", -1)))
        setattr(controller, "_capture_plan_cap_tensor_device", dev_key)
        setattr(controller, "_capture_plan_cap_tensor_signature", bound_sig)
    return cache_tensor


def refresh_capture_layout_live_lengths(
    *,
    layout: StepCaptureLayout,
    row_tensor: torch.Tensor,
    row_list: Sequence[int],
    seqused_k: torch.Tensor,
    cap_tensor: torch.Tensor,
    step_context: object,
    device: torch.device,
    num_heads: int,
    chunk_query_lengths: torch.Tensor | None = None,
    plan_cap_by_row_cpu: Sequence[int] | None = None,
    context_kv_len_by_row_cpu: Sequence[int] | None = None,
) -> bool:
    """Refresh small live length views without reallocating capture buffers."""
    seq_lens_src = getattr(step_context, "seq_lens", tuple())
    q_lens_src = getattr(step_context, "q_lens", tuple())
    context_kv_src = (
        tuple(int(v) for v in context_kv_len_by_row_cpu)
        if context_kv_len_by_row_cpu is not None
        else tuple()
    )
    cap_src = (
        tuple(int(v) for v in plan_cap_by_row_cpu)
        if plan_cap_by_row_cpu is not None
        else tuple()
    )
    row_count = len(row_list)
    row_key = tuple(int(r) for r in row_list)
    use_cpu_context_key = bool(
        context_kv_src
        and cap_src
        and all(0 <= int(row) < len(context_kv_src) and 0 <= int(row) < len(cap_src) for row in row_key)
    )
    use_cpu_chunk_lengths = bool(
        chunk_query_lengths is None
        or (
            use_cpu_context_key
            and q_lens_src
            and all(0 <= int(row) < len(q_lens_src) for row in row_key)
        )
    )
    live_key = (
        int(getattr(step_context, "epoch", -1)),
        int(getattr(step_context, "step_handle_id", -1)),
        int(getattr(step_context, "step_handle_generation", -1)),
        int(row_count),
        row_key,
        seq_lens_src,
        context_kv_src if use_cpu_context_key else tuple(),
        cap_src if use_cpu_context_key else tuple(),
        int(num_heads),
        ("cpu_context_kv",) if use_cpu_context_key else _tensor_live_key(seqused_k),
        ("cpu_plan_cap",) if use_cpu_context_key else _tensor_live_key(cap_tensor),
        ("cpu_q_lens", tuple(int(v) for v in q_lens_src)) if use_cpu_chunk_lengths else _tensor_live_key(chunk_query_lengths),
    )
    if (
        getattr(layout, "live_lengths_key", None) == live_key
        and isinstance(layout.seq_lens_batch, torch.Tensor)
        and isinstance(layout.seq_lens_batch_i32, torch.Tensor)
        and isinstance(layout.kv_lengths, torch.Tensor)
        and isinstance(layout.kv_len_per_row_i32, torch.Tensor)
        and getattr(layout, "kv_len_per_row_cpu", None) is not None
        and int(layout.seq_lens_batch_i32.numel()) >= int(row_count)
        and int(layout.kv_len_per_row_i32.numel()) >= int(row_count)
    ):
        return False

    seq_lens_cpu = tuple(
        int(seq_lens_src[int(r)]) if 0 <= int(r) < len(seq_lens_src) else 0
        for r in row_key
    )

    if use_cpu_context_key:
        context_kv_cpu = tuple(int(context_kv_src[int(row)]) for row in row_key)
        kv_len_per_row_cpu = tuple(
            max(1, min(int(context_kv_cpu[idx]), int(cap_src[int(row)])))
            for idx, row in enumerate(row_key)
        )
        stage_cache = getattr(layout, "small_tensor_stage", None)
        seq_full = cached_sequence_to_device(
            context_kv_cpu,
            dtype=torch.long,
            device=device,
            cache_name="live_seq_full_i64",
            stage_cache=stage_cache,
            out=layout.seq_lens_batch,
        )
        seq_full_i32 = cached_sequence_to_device(
            context_kv_cpu,
            dtype=torch.int32,
            device=device,
            cache_name="live_seq_full_i32",
            stage_cache=stage_cache,
            out=layout.seq_lens_batch_i32,
        )
        kv_len = cached_sequence_to_device(
            kv_len_per_row_cpu,
            dtype=torch.long,
            device=device,
            cache_name="live_kv_len_i64",
            stage_cache=stage_cache,
        )
        layout.seq_lens_batch = seq_full
        layout.seq_lens_batch_i32 = seq_full_i32
        layout.seq_lens_cpu = seq_lens_cpu
        layout.kv_len_per_row_cpu = kv_len_per_row_cpu
        layout.seq_lens_tensor_cpu = _cpu_long_tensor_from_values(
            layout.seq_lens_tensor_cpu,
            seq_lens_cpu,
        )
        layout.kv_len_per_row_i32 = cached_sequence_to_device(
            kv_len_per_row_cpu,
            dtype=torch.int32,
            device=device,
            cache_name="live_kv_len_i32",
            stage_cache=stage_cache,
            out=layout.kv_len_per_row_i32,
        )
        layout.kv_lengths = kv_len.unsqueeze(1).expand(-1, int(num_heads))
        if chunk_query_lengths is not None and use_cpu_chunk_lengths:
            layout.chunk_lengths = cached_sequence_to_device(
                tuple(int(q_lens_src[int(row)]) for row in row_key),
                device=device,
                dtype=torch.long,
                cache_name="live_chunk_i64",
                stage_cache=stage_cache,
                out=layout.chunk_lengths,
            )
        elif chunk_query_lengths is not None:
            layout.chunk_lengths = chunk_query_lengths.index_select(
                0,
                row_tensor.to(chunk_query_lengths.device),
            ).to(device=device, dtype=torch.long)
        layout.live_lengths_key = live_key
        return True

    seqused_live = (
        seqused_k
        if seqused_k.device == device
        else seqused_k.to(device=device)
    )
    row_index = row_tensor.to(device=seqused_live.device)
    seq_full = seqused_live.index_select(0, row_index).to(device=device, dtype=torch.long)
    cap = cap_tensor.index_select(0, row_tensor.to(device=cap_tensor.device)).to(
        device=device,
        dtype=torch.long,
    )
    kv_len = torch.minimum(seq_full, cap)
    kv_len = torch.clamp(kv_len, min=1)

    kv_len_per_row_cpu = None
    if plan_cap_by_row_cpu is not None:
        cap_src = tuple(int(v) for v in plan_cap_by_row_cpu)
        kv_len_per_row_cpu = tuple(
            max(
                1,
                min(
                    int(seq_lens_cpu[idx]),
                    int(cap_src[row]) if 0 <= int(row) < len(cap_src) else 0,
                ),
            )
            for idx, row in enumerate(row_key)
        )
    elif cap_tensor.device.type == "cpu":
        kv_len_per_row_cpu = tuple(
            max(1, min(int(seq_lens_cpu[idx]), int(cap_tensor[int(row)].item())))
            for idx, row in enumerate(row_key)
        )

    layout.seq_lens_batch = seq_full
    layout.seq_lens_batch_i32 = seq_full.to(device=device, dtype=torch.int32)
    layout.seq_lens_cpu = seq_lens_cpu
    layout.kv_len_per_row_cpu = kv_len_per_row_cpu
    layout.seq_lens_tensor_cpu = _cpu_long_tensor_from_values(
        layout.seq_lens_tensor_cpu,
        seq_lens_cpu,
    )
    layout.kv_len_per_row_i32 = kv_len.to(device=device, dtype=torch.int32)
    layout.kv_lengths = kv_len.unsqueeze(1).expand(-1, int(num_heads))
    if chunk_query_lengths is not None:
        layout.chunk_lengths = chunk_query_lengths.index_select(
            0,
            row_tensor.to(chunk_query_lengths.device),
        ).to(device=device, dtype=torch.long)
    layout.live_lengths_key = live_key
    return True
