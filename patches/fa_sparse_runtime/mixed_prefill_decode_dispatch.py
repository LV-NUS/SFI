"""Production mixed prefill/decode owner orchestration."""
from __future__ import annotations

import os
from typing import Optional, Sequence, Tuple

import torch

from patches.fa_sparse_runtime.compact_recent_route_authority import CompactRecentRailMode
from patches.fa_sparse_runtime.mixed_prefill_decode_owner import MixedPrefillDecodeOwnerPlan
from patches.fa3_native.mixed_page_graph_descriptor import PageResolverKind


def _normalize_window_tuple(
    window_size: Tuple[int, int] | Sequence[int] | None,
) -> tuple[int, int]:
    if window_size is None:
        return (-1, -1)
    return (int(window_size[0]), int(window_size[1]))


def _maybe_int_rows(value: object) -> tuple[int, ...] | None:
    if value is None:
        return None
    try:
        return tuple(int(row) for row in value)  # type: ignore[union-attr]
    except (TypeError, ValueError):
        return None


def phase_lastn1_direct_capture_rows(
    *,
    side_outputs: object,
    phase: str,
    rows: Sequence[int],
) -> tuple[int, ...]:
    """Rows that can write logits directly into the phase capture ring.

    The direct path is intentionally narrow: one phase, last_n==1 for every
    producer row, and a real phase output tensor/mapping.  last_n>1 still needs
    the scratch + reducer path because the selector consumes log_f_pre+denom.
    """
    rows_tuple = tuple(int(row) for row in rows)
    if not rows_tuple:
        return tuple()
    if phase == "prefill":
        out_scores = getattr(side_outputs, "prefill_out_capture_scores", None)
        mapping = getattr(side_outputs, "prefill_capture_row_by_batch_row_i32", None)
    elif phase == "refresh":
        out_scores = getattr(side_outputs, "refresh_out_capture_scores", None)
        mapping = getattr(side_outputs, "refresh_capture_row_by_batch_row_i32", None)
    else:
        raise ValueError(f"unsupported capture phase: {phase}")
    if not (
        isinstance(out_scores, torch.Tensor)
        and out_scores.dim() == 4
        and out_scores.dtype == torch.float16
        and int(out_scores.shape[2]) == 1
        and isinstance(mapping, torch.Tensor)
        and mapping.dtype == torch.int32
    ):
        return tuple()
    last_n_by_row = getattr(side_outputs, "row_capture_last_n_cpu", tuple())
    if len(last_n_by_row) <= max(rows_tuple):
        return tuple()
    if any(int(last_n_by_row[row]) != 1 for row in rows_tuple):
        return tuple()
    return rows_tuple


def _single_phase_lastn1_direct_capture_target(
    *,
    side_outputs: object,
    owner_plan: MixedPrefillDecodeOwnerPlan,
) -> tuple[str, torch.Tensor, torch.Tensor] | None:
    """Return the phase output that can replace scratch capture, if any."""
    prefill_rows = _maybe_int_rows(getattr(owner_plan, "prefill_capture_rows", None)) or tuple()
    decode_rows = _maybe_int_rows(getattr(owner_plan, "decode_capture_rows", None)) or tuple()
    if prefill_rows and not decode_rows:
        direct_rows = phase_lastn1_direct_capture_rows(
            side_outputs=side_outputs,
            phase="prefill",
            rows=prefill_rows,
        )
        out_scores = getattr(side_outputs, "prefill_out_capture_scores", None)
        mapping = getattr(side_outputs, "prefill_capture_row_by_batch_row_i32", None)
        if (
            len(direct_rows) == len(prefill_rows)
            and isinstance(out_scores, torch.Tensor)
            and isinstance(mapping, torch.Tensor)
        ):
            return "prefill", out_scores, mapping
    if decode_rows and not prefill_rows:
        direct_rows = phase_lastn1_direct_capture_rows(
            side_outputs=side_outputs,
            phase="refresh",
            rows=decode_rows,
        )
        out_scores = getattr(side_outputs, "refresh_out_capture_scores", None)
        mapping = getattr(side_outputs, "refresh_capture_row_by_batch_row_i32", None)
        if (
            len(direct_rows) == len(decode_rows)
            and isinstance(out_scores, torch.Tensor)
            and isinstance(mapping, torch.Tensor)
        ):
            return "refresh", out_scores, mapping
    return None


def _decode_capture_rows_have_capture_rail(
    *,
    owner_plan: MixedPrefillDecodeOwnerPlan,
    rail_decision: object | None,
) -> bool:
    """Return whether every decode capture row has DECODE_CAPTURE authority."""
    decode_capture_rows = _maybe_int_rows(
        getattr(owner_plan, "decode_capture_rows", None)
    )
    if not decode_capture_rows or rail_decision is None:
        return False

    mode = getattr(rail_decision, "mode", None)
    if mode is not CompactRecentRailMode.DECODE_CAPTURE:
        return False

    authority_capture_rows = _maybe_int_rows(
        getattr(rail_decision, "capture_rows", None)
    )
    if authority_capture_rows is not None:
        return set(decode_capture_rows).issubset(set(authority_capture_rows))

    return True


def dispatch_mixed_prefill_decode_active_workload(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    seqused_k: torch.Tensor,
    softmax_scale: float,
    window_size: Tuple[int, int] | Sequence[int] | None,
    softcap: float,
    block_table: torch.Tensor,
    owner_plan: MixedPrefillDecodeOwnerPlan,
    bridge: object,
    controller: object,
    state: object,
    step_authority: object,
    step_bound_meta: object,
    step_ctx: object,
    q_v: Optional[torch.Tensor] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    s_aux: Optional[torch.Tensor] = None,
    scheduler_metadata: Optional[torch.Tensor] = None,
    num_splits: int = 0,
    cp_world_size: int = 1,
    cp_rank: int = 0,
    cp_tot_seqused_k: Optional[torch.Tensor] = None,
    capture_scores: Optional[torch.Tensor] = None,
    capture_row_index_i32: Optional[torch.Tensor] = None,
    row_capture_last_n_i32: Optional[torch.Tensor] = None,
    rail_decision: object | None = None,
    resolver_descriptor: object | None = None,
    resolver_carriers: object | None = None,
    resolver_seqused_k: Optional[torch.Tensor] = None,
    captured_resolver_descriptor: object | None = None,
    captured_resolver_pointer_signature: tuple[int | None, ...] | None = None,
    graph_replay_carriers: bool = False,
) -> torch.Tensor:
    """Launch mixed prefill/decode through one full-batch mixed-page overlay."""
    if not isinstance(cu_seqlens_q, torch.Tensor):
        raise RuntimeError("mixed production dispatcher requires tensor cu_seqlens_q")
    if cu_seqlens_q.device != q.device or cu_seqlens_q.dtype != torch.int32:
        raise RuntimeError("mixed production dispatcher requires int32 CUDA cu_seqlens_q")
    if int(cu_seqlens_q.numel()) < len(owner_plan.q_lens) + 1:
        raise RuntimeError("mixed production dispatcher requires full-batch cu_seqlens_q")

    if owner_plan.has_capture_rows:
        if not (
            isinstance(capture_scores, torch.Tensor)
            and isinstance(capture_row_index_i32, torch.Tensor)
            and isinstance(row_capture_last_n_i32, torch.Tensor)
        ):
            raise RuntimeError("mixed production dispatcher requires capture tensors")

    window_tuple = _normalize_window_tuple(window_size)

    if k.dim() != 4 or v.dim() != 4:
        raise RuntimeError("mixed production dispatcher requires paged k/v tensors")
    page_size = int(k.shape[1])
    if page_size <= 0 or int(v.shape[1]) != page_size:
        raise RuntimeError(
            "mixed production dispatcher could not infer page_size from key/value cache"
        )
    if int(k.shape[2]) != int(v.shape[2]):
        raise RuntimeError("mixed production dispatcher k/v head mismatch")

    launch_plan = getattr(step_bound_meta, "compact_recent_launch_plan", None)
    use_no_compact_full_k_geometry = False
    if launch_plan is None or not bool(getattr(launch_plan, "valid", False)):
        use_compact_by_row = tuple(
            bool(v) for v in getattr(step_authority, "use_compact_by_row", tuple())
        )
        if not any(use_compact_by_row[: len(owner_plan.q_lens)]):
            use_no_compact_full_k_geometry = True
        else:
            raise RuntimeError(
                "mixed production dispatcher requires a valid compact launch metadata plan"
            )
    if not use_no_compact_full_k_geometry:
        launch_plan_page_size = int(getattr(launch_plan, "page_size", 0) or 0)
        if launch_plan_page_size != page_size:
            raise RuntimeError("mixed production dispatcher launch plan page_size mismatch")
    else:
        launch_plan_page_size = page_size

    if launch_plan_page_size != page_size:
        raise RuntimeError(
            "mixed production dispatcher launch plan page_size mismatch"
        )

    from patches.fa_sparse_runtime.compact_mixed_page_route import (
        resolve_compact_mixed_page_overlay_cpu_geometry,
        run_compact_mixed_page_overlay_route,
    )

    real_kv_len_hint = getattr(
        step_bound_meta,
        "canonical_real_kv_len_cpu",
        getattr(step_authority, "context_kv_len_by_row", None),
    )
    batch_size = len(owner_plan.q_lens)
    if use_no_compact_full_k_geometry:
        real_kv_len_cpu = tuple(int(v) for v in tuple(real_kv_len_hint)[:batch_size])
        if len(real_kv_len_cpu) < batch_size:
            raise RuntimeError(
                "mixed production dispatcher no-compact RRP path requires full real-K coverage"
            )
        overlay_compact_valid_cpu = tuple(0 for _ in range(batch_size))
        overlay_recent_first_cpu = tuple(0 for _ in range(batch_size))
        overlay_recent_count_cpu = tuple(
            (max(0, int(real)) + page_size - 1) // page_size
            for real in real_kv_len_cpu
        )
        overlay_effective_k_len_cpu = tuple(max(0, int(real)) for real in real_kv_len_cpu)
        overlay_compact_offset_cpu = tuple(0 for _ in range(batch_size))
    else:
        (
            overlay_compact_valid_cpu,
            overlay_recent_first_cpu,
            overlay_recent_count_cpu,
            overlay_effective_k_len_cpu,
        ) = resolve_compact_mixed_page_overlay_cpu_geometry(
            launch_plan=launch_plan,
            step_bound_meta=step_bound_meta,
            step_authority=step_authority,
            real_kv_len_hint=real_kv_len_hint,
            page_size=page_size,
            batch_size=batch_size,
        )
        overlay_compact_offset_cpu = tuple(
            int(v)
            for v in tuple(getattr(launch_plan, "compact_offset_tokens_cpu", tuple()))[
                :batch_size
            ]
        )
    launch_max_seqlen_k = max(overlay_effective_k_len_cpu[:batch_size], default=0)
    if launch_max_seqlen_k <= 0:
        raise RuntimeError(
            "mixed production dispatcher requires positive overlay max_seqlen_k"
        )

    launch_resolver_seqused_k = resolver_seqused_k
    if use_no_compact_full_k_geometry and launch_resolver_seqused_k is None:
        launch_resolver_seqused_k = seqused_k

    return run_compact_mixed_page_overlay_route(
        bridge=bridge,
        q=q,
        k=k,
        v=v,
        out=out,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=int(max_seqlen_q),
        seqused_k=seqused_k,
        max_seqlen_k=launch_max_seqlen_k,
        softmax_scale=softmax_scale,
        window_size=window_tuple,
        softcap=float(softcap),
        block_table=block_table,
        controller=controller,
        state=state,
        step_authority=step_authority,
        step_bound_meta=step_bound_meta,
        page_size=page_size,
        num_kv_heads=int(k.shape[2]),
        compact_valid_tokens_by_row=overlay_compact_valid_cpu[:batch_size],
        compact_offset_tokens_by_row=overlay_compact_offset_cpu[:batch_size],
        recent_first_page_by_row=overlay_recent_first_cpu[:batch_size],
        recent_page_count_by_row=overlay_recent_count_cpu[:batch_size],
        row_effective_k_by_row=overlay_effective_k_len_cpu[:batch_size],
        safe_page_id=0,
        num_splits=int(num_splits),
        cp_world_size=int(cp_world_size),
        cp_rank=int(cp_rank),
        cp_tot_seqused_k=cp_tot_seqused_k,
        q_v=q_v,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        s_aux=s_aux,
        capture_scores=capture_scores,
        capture_row_index_i32=capture_row_index_i32,
        row_capture_last_n_i32=row_capture_last_n_i32,
        scheduler_metadata=scheduler_metadata,
        prefill_active_worklist=False,
        prefill_active_count=0,
        resolver_descriptor=resolver_descriptor,
        resolver_carriers=resolver_carriers,
        resolver_seqused_k=launch_resolver_seqused_k,
        captured_resolver_descriptor=captured_resolver_descriptor,
        captured_resolver_pointer_signature=captured_resolver_pointer_signature,
        graph_replay_carriers=bool(graph_replay_carriers),
    )


def dispatch_capture_mixed_owner(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    seqused_k: torch.Tensor,
    softmax_scale: float,
    window_size: Tuple[int, int] | Sequence[int] | None,
    softcap: float,
    block_table: torch.Tensor,
    owner_plan: MixedPrefillDecodeOwnerPlan,
    side_outputs: object,
    bridge: object,
    controller: object,
    state: object,
    step_authority: object,
    step_bound_meta: object,
    step_ctx: object,
    rail_decision: object,
    q_v: Optional[torch.Tensor] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    s_aux: Optional[torch.Tensor] = None,
    scheduler_metadata: Optional[torch.Tensor] = None,
    num_splits: int = 0,
    cp_world_size: int = 1,
    cp_rank: int = 0,
    cp_tot_seqused_k: Optional[torch.Tensor] = None,
    capture_postprocess_state: Optional[dict[str, object]] = None,
    resolver_descriptor: object | None = None,
    resolver_carriers: object | None = None,
    resolver_seqused_k: Optional[torch.Tensor] = None,
    captured_resolver_descriptor: object | None = None,
    captured_resolver_pointer_signature: tuple[int | None, ...] | None = None,
    graph_replay_carriers: bool = False,
) -> torch.Tensor:
    """Capture path owner-based dispatcher (rev 2).

    Rule: capture rows are handled by one full-batch mixed-page launch.  The
    capture step needs the full KV domain, so decode capture no longer routes
    through compact_recent full-recent emulation.

    Capture scratch is shared via side_outputs (from
    prepare_capture_forward_side_outputs). owner_plan guarantees
    prefill_capture_rows and decode_capture_rows are disjoint, so two kernels
    write non-overlapping slots of side_outputs.scratch_capture_scores.
    """
    if not owner_plan.has_capture_rows:
        raise RuntimeError(
            "dispatch_capture_mixed_owner invoked with zero capture rows; "
            "caller must route non-capture batches through "
            "dispatch_mixed_prefill_decode_active_workload or the mixed-page overlay route"
        )

    # Note (rev 2 revision after E2E 2026-04-23):
    # Production batches routinely include non-capture decode rows alongside
    # capture decode rows (and similarly for prefill). Per-row capture is
    # driven by capture_row_index_i32 (-1 => skip capture for that row); the
    # kernel handles the mix correctly. An earlier defense-in-depth assertion
    # that required every row in a capture batch to also be a capture row was
    # too strict and broke real inference — removed.

    window_tuple = _normalize_window_tuple(window_size)

    has_prefill_cap = bool(owner_plan.has_prefill_capture_rows)
    has_decode_cap = bool(owner_plan.has_decode_capture_rows)
    decode_capture_has_authority = bool(
        has_decode_cap
        and _decode_capture_rows_have_capture_rail(
            owner_plan=owner_plan,
            rail_decision=rail_decision,
        )
    )
    try:
        from patches.fa3_native.install import (
            append_fa3_route_trace,
            fa3_route_trace_enabled,
        )

        if fa3_route_trace_enabled():
            append_fa3_route_trace(
                {
                    "event": "capture_mixed_owner_dispatch",
                    "epoch": int(getattr(step_authority, "epoch", -1)),
                    "has_prefill_capture_rows": bool(has_prefill_cap),
                    "has_decode_capture_rows": bool(has_decode_cap),
                    "decode_capture_uses_mixed_page": bool(has_decode_cap),
                    "decode_capture_owned_by_compact": False,
                    "two_rail": False,
                    "rail_mode": getattr(
                        getattr(rail_decision, "mode", None),
                        "name",
                        str(getattr(rail_decision, "mode", "")),
                    ),
                    "prefill_rows": [
                        int(v) for v in tuple(getattr(owner_plan, "prefill_rows", tuple()))
                    ],
                    "decode_rows": [
                        int(v) for v in tuple(getattr(owner_plan, "decode_rows", tuple()))
                    ],
                    "prefill_capture_rows": [
                        int(v) for v in tuple(getattr(owner_plan, "prefill_capture_rows", tuple()))
                    ],
                    "decode_capture_rows": [
                        int(v) for v in tuple(getattr(owner_plan, "decode_capture_rows", tuple()))
                    ],
                }
            )
    except Exception:
        pass

    if has_decode_cap and not decode_capture_has_authority:
        raise RuntimeError(
            "decode capture rows must route through mixed-page DECODE_CAPTURE authority"
        )

    force_scratch_capture = bool(
        isinstance(capture_postprocess_state, dict)
        and capture_postprocess_state.get("force_scratch_capture", False)
    )
    direct_capture_target = (
        None
        if force_scratch_capture
        else _single_phase_lastn1_direct_capture_target(
            side_outputs=side_outputs,
            owner_plan=owner_plan,
        )
    )
    capture_scores_arg = side_outputs.scratch_capture_scores
    capture_row_index_arg = side_outputs.capture_row_index_i32
    direct_capture_phase = ""
    if direct_capture_target is not None:
        direct_capture_phase, capture_scores_arg, capture_row_index_arg = direct_capture_target
    if isinstance(capture_postprocess_state, dict):
        capture_postprocess_state["direct_capture_phase"] = str(direct_capture_phase)

    uses_resolver_overlay = resolver_descriptor is not None or resolver_carriers is not None

    def _launch_full_capture() -> None:
        if uses_resolver_overlay:
            dispatch_mixed_prefill_decode_active_workload(
                q=q,
                k=k,
                v=v,
                out=out,
                cu_seqlens_q=cu_seqlens_q,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                seqused_k=seqused_k,
                softmax_scale=softmax_scale,
                window_size=window_tuple,
                softcap=softcap,
                block_table=block_table,
                owner_plan=owner_plan,
                bridge=bridge,
                controller=controller,
                state=state,
                step_authority=step_authority,
                step_bound_meta=step_bound_meta,
                step_ctx=step_ctx,
                q_v=q_v,
                q_descale=q_descale,
                k_descale=k_descale,
                v_descale=v_descale,
                s_aux=s_aux,
                scheduler_metadata=scheduler_metadata,
                num_splits=num_splits,
                cp_world_size=cp_world_size,
                cp_rank=cp_rank,
                cp_tot_seqused_k=cp_tot_seqused_k,
                capture_scores=capture_scores_arg,
                capture_row_index_i32=capture_row_index_arg,
                row_capture_last_n_i32=side_outputs.row_capture_last_n_i32,
                rail_decision=rail_decision,
                resolver_descriptor=resolver_descriptor,
                resolver_carriers=resolver_carriers,
                resolver_seqused_k=resolver_seqused_k,
                captured_resolver_descriptor=captured_resolver_descriptor,
                captured_resolver_pointer_signature=captured_resolver_pointer_signature,
                graph_replay_carriers=graph_replay_carriers,
            )
            return

        bridge.mixed_page_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            max_seqlen_q=int(max_seqlen_q),
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_k=int(max_seqlen_k),
            seqused_k=seqused_k,
            q_v=q_v,
            softmax_scale=float(softmax_scale),
            causal=True,
            window_size=list(window_tuple),
            softcap=float(softcap),
            block_table=block_table,
            page_resolver_kind=int(PageResolverKind.NATIVE),
            return_softmax_lse=False,
            out=out,
            scheduler_metadata=scheduler_metadata,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            num_splits=int(num_splits),
            s_aux=s_aux,
            cp_world_size=int(cp_world_size),
            cp_rank=int(cp_rank),
            cp_tot_seqused_k=cp_tot_seqused_k,
            capture_scores=capture_scores_arg,
            capture_row_index_i32=capture_row_index_arg,
            row_capture_last_n_i32=side_outputs.row_capture_last_n_i32,
            prefill_active_worklist=False,
            prefill_active_count=0,
        )

    _launch_full_capture()
    return out
