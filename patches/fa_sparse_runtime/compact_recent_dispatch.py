"""Compact-arena wait helper for the live mixed-page / compact-recent route.

OWNS:
  - _maybe_wait_for_async_compact_arena(...): wait on the controller's
    chunk/buf dependency before compact-arena reads. compact_recent reads the
    same arena the mixed-page dispatch consumes, so it must wait on the
    refresh stream when present; the older per-layer gather event remains a
    fallback, not the only guard. Consumed by the live mixed-page route
    (patches.fa_sparse_runtime.compact_mixed_page_route).

The retired direct compact_recent dispatch route (dispatch_compact_recent,
build_row_consume_mode_i32 and their geometry / native-fwd / trace helpers)
was removed; production routes through mixed-page ResolvedRowPtr via the FA3
bridge (see tests/test_attn_mode_gating.py).
"""
from __future__ import annotations

import torch

from patches.sparse_utils import _is_stream_capturing_or_raise


def _int_attr(obj: object, name: str, default: int = 0) -> int:
    value = getattr(obj, name, default)
    if isinstance(value, (int, float)):
        return int(value)
    return int(default)


def _maybe_wait_for_async_compact_arena(
    *,
    controller: object,
    state: object,
    device: torch.device,
) -> None:
    """Wait on the controller's chunk/buf dependency before compact arena reads.

    The mixed-page dispatch already establishes a producer/consumer boundary
    before consuming compact K/V. compact_recent reads the same arena, so it
    must wait on the refresh stream when present; the older per-layer gather
    event remains a fallback, not the only guard.
    """
    if controller is None:
        return
    if device.type != "cuda":
        return
    epoch = _int_attr(controller, "step_context_epoch", -1)
    compute_wait = getattr(controller, "_compute_wait_decision", None)
    wait_done = getattr(controller, "_main_stream_wait_for_chunk_done", None)
    map_layer = getattr(controller, "_map_global_layer_to_capture_slot", None)
    if compute_wait is None or wait_done is None or map_layer is None:
        return
    flags = getattr(controller, "_buf_pending_work_flags", None)
    has_known_pending_flags = isinstance(flags, (list, tuple)) and bool(flags)
    release_epoch = _int_attr(controller, "_prefill_release_pending_epoch", -1)
    need_release = release_epoch >= 0 and epoch >= 0 and release_epoch == epoch
    bootstrap_pending_request_ids = getattr(controller, "_bootstrap_pending_request_ids", None)
    need_bootstrap = bool(bootstrap_pending_request_ids)
    if (
        epoch >= 0
        and _int_attr(controller, "_compact_recent_no_async_wait_epoch", -2) == epoch
        and has_known_pending_flags
        and all(int(flag) == 0 for flag in flags)
        and not need_release
        and not need_bootstrap
    ):
        return
    layer_index = int(getattr(state, "layer_index", -1))
    if layer_index < 0:
        return
    chunk_id = int(getattr(state, "capture_chunk_id", -1))
    buf_id = int(getattr(state, "capture_buf_id", -1))
    layer_epoch = int(getattr(state, "layer_index_epoch", -1))
    controller_epoch = int(getattr(controller, "_layer_index_cache_epoch", -1))
    if layer_epoch != controller_epoch or chunk_id < 0 or buf_id < 0:
        chunk_id, buf_id, slot_in_chunk = map_layer(layer_index)
        try:
            state.capture_chunk_id = int(chunk_id)
            state.capture_buf_id = int(buf_id)
            state.capture_slot_in_chunk = int(slot_in_chunk)
            state.layer_index_epoch = int(controller_epoch)
        except Exception:
            pass
    pending_flags = 0
    if has_known_pending_flags:
        pending_flags = int(flags[int(buf_id) % len(flags)])
    group_ready_wait = getattr(
        controller,
        "_maybe_wait_one_shot_group_ready_for_layer",
        None,
    )
    if callable(group_ready_wait):
        group_ready_waited = bool(
            group_ready_wait(
                layer_index=int(layer_index),
                device=device,
            )
        )
        if (
            group_ready_waited
            and pending_flags == 0
            and not need_release
            and not need_bootstrap
        ):
            return
    if (
        has_known_pending_flags
        and pending_flags == 0
        and not need_release
        and not need_bootstrap
    ):
        if epoch >= 0 and all(int(flag) == 0 for flag in flags):
            try:
                controller._compact_recent_no_async_wait_epoch = epoch
            except Exception:
                pass
        return
    need_wait, _reason = compute_wait(
        buf_id=int(buf_id),
        epoch=epoch,
        path_tag="compact_recent_layer",
        consume_step_token=False,
    )
    if not need_wait:
        return
    refresh_stream = getattr(controller, "refresh_stream", None)
    if refresh_stream is not None:
        if not _is_stream_capturing_or_raise(stage="compact_recent_refresh_stream_wait_probe"):
            wait_done(
                buf_id=int(buf_id),
                device=device,
                chunk_id=int(chunk_id),
                epoch=epoch,
            )
            return
        marker_epoch = int(getattr(controller, "_compact_recent_wait_stream_epoch", -1))
        marker = getattr(controller, "_compact_recent_wait_stream_chunks", set())
        if marker_epoch != epoch:
            marker = set()
            try:
                controller._compact_recent_wait_stream_epoch = epoch
                controller._compact_recent_wait_stream_chunks = marker
            except Exception:
                pass
        wait_key = (int(buf_id), int(chunk_id))
        if wait_key not in marker:
            torch.cuda.current_stream(device=device).wait_stream(refresh_stream)
            clear_buf = getattr(controller, "_pending_work_clear_buf", None)
            if callable(clear_buf):
                try:
                    clear_buf(buf_id=int(buf_id))
                except Exception:
                    pass
            marker.add(wait_key)
        return
    if need_wait:
        wait_done(
            buf_id=int(buf_id),
            device=device,
            chunk_id=int(chunk_id),
            epoch=epoch,
        )
