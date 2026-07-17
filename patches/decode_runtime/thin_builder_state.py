from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class DecodeRuntimeMode(str, Enum):
    STEADY_DELTA = "steady_delta"
    PAGE_BOUNDARY_DELTA = "page_boundary_delta"
    REFRESH_COMMIT = "refresh_commit"
    FULL_RECOMPILE = "full_recompile"
    # [LITE-P0 2026-07-11] signature-return arm: the step after a trigger
    # (capture) step whose q/logf signatures flip BACK to the pre-trigger
    # steady regime while the value plane is a pure token-advance. Admitted
    # outside classify (which compares against the TRIGGER delta and would
    # see q/logf changed); recorded via DecodeRuntimeState.apply_sig_return.
    SIG_RETURN_DELTA = "sig_return_delta"


def _int_value(value: Any) -> int:
    return int(value)


def _str_value(value: Any) -> str:
    return str(value)


def _tuple_value(value: Any) -> tuple[Any, ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def _int_tuple(value: Any) -> tuple[int, ...]:
    return tuple(_int_value(item) for item in _tuple_value(value))


def _str_tuple(value: Any) -> tuple[str, ...]:
    return tuple(_str_value(item) for item in _tuple_value(value))


@dataclass(frozen=True, slots=True)
class DecodeStaticGuard:
    graph_key: str
    batch_size: int
    block_size: int
    num_kv_heads: int
    page_size: int
    max_pages_per_row: int
    slot_signature: tuple[int, ...]
    resolver_kind: str
    row_mode_class_signature: tuple[str, ...]
    compact_capacity_pages: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "graph_key", _str_value(self.graph_key))
        object.__setattr__(self, "batch_size", _int_value(self.batch_size))
        object.__setattr__(self, "block_size", _int_value(self.block_size))
        object.__setattr__(self, "num_kv_heads", _int_value(self.num_kv_heads))
        object.__setattr__(self, "page_size", _int_value(self.page_size))
        object.__setattr__(self, "max_pages_per_row", _int_value(self.max_pages_per_row))
        object.__setattr__(self, "slot_signature", _int_tuple(self.slot_signature))
        object.__setattr__(self, "resolver_kind", _str_value(self.resolver_kind))
        object.__setattr__(
            self,
            "row_mode_class_signature",
            _str_tuple(self.row_mode_class_signature),
        )
        object.__setattr__(
            self,
            "compact_capacity_pages",
            _int_value(self.compact_capacity_pages),
        )


@dataclass(frozen=True, slots=True)
class DecodeDeltaPacket:
    step_id: int
    batch_size: int
    row_effective_k_by_row: tuple[int, ...]
    request_recent_len_by_row: tuple[int, ...]
    launch_effective_k_by_row: tuple[int, ...]
    recent_first_page_by_row: tuple[int, ...]
    recent_page_count_by_row: tuple[int, ...]
    q_lens_by_row: tuple[int, ...]
    logf_mask_generation: tuple[int, ...]
    pending_refresh_state: str = "nil"
    row_dynamic_signature: tuple[object, ...] = tuple()

    def __post_init__(self) -> None:
        object.__setattr__(self, "step_id", _int_value(self.step_id))
        object.__setattr__(self, "batch_size", _int_value(self.batch_size))
        object.__setattr__(self, "pending_refresh_state", _str_value(self.pending_refresh_state))
        object.__setattr__(
            self,
            "row_dynamic_signature",
            tuple(self.row_dynamic_signature),
        )
        for field_name in (
            "row_effective_k_by_row",
            "request_recent_len_by_row",
            "launch_effective_k_by_row",
            "recent_first_page_by_row",
            "recent_page_count_by_row",
            "q_lens_by_row",
            "logf_mask_generation",
        ):
            normalized = _int_tuple(getattr(self, field_name))
            if len(normalized) != self.batch_size:
                raise ValueError(f"{field_name} must cover batch_size={self.batch_size}")
            object.__setattr__(self, field_name, normalized)


@dataclass(slots=True)
class DecodeRuntimeCounters:
    """Classification updates mode counters.

    Hotpath integration phases update builder/cache/sync/allocation counters.
    """

    same_page_step_count: int = 0
    page_boundary_step_count: int = 0
    refresh_commit_step_count: int = 0
    full_recompile_count: int = 0
    slow_builder_call_count: int = 0
    old_cache_probe_count: int = 0
    adapter_invocation_count: int = 0
    steady_delta_d2h_sync_count: int = 0
    implicit_sync_count: int = 0
    per_step_allocation_count: int = 0
    carrier_update_kernel_count: int = 0
    predicted_same_page_step_count: int = 0
    # [LITE-P0] signature-return arm hits / fail-close falls (reason strings
    # carry the lite_fallback: prefix into decode_runtime_reason).
    sig_return_step_count: int = 0
    lite_fallback_count: int = 0


@dataclass(slots=True)
class DecodeRuntimeState:
    active_guard: DecodeStaticGuard | None = None
    last_delta: DecodeDeltaPacket | None = None
    last_mode: DecodeRuntimeMode | None = None
    last_reason: str = ""
    last_applied_step_id: int | None = None
    counters: DecodeRuntimeCounters = field(default_factory=DecodeRuntimeCounters)

    def update_classification(
        self,
        current_guard: DecodeStaticGuard,
        current_delta: DecodeDeltaPacket,
    ) -> tuple[DecodeRuntimeMode, str]:
        mode, reason = classify_decode_runtime_mode(
            self.active_guard,
            self.last_delta,
            current_guard,
            current_delta,
        )
        self.active_guard = current_guard
        self.last_delta = current_delta
        self.last_mode = mode
        self.last_reason = reason
        self.last_applied_step_id = current_delta.step_id
        self._increment_counter(mode)
        return mode, reason

    def _increment_counter(self, mode: DecodeRuntimeMode) -> None:
        if mode is DecodeRuntimeMode.STEADY_DELTA:
            self.counters.same_page_step_count += 1
        elif mode is DecodeRuntimeMode.PAGE_BOUNDARY_DELTA:
            self.counters.page_boundary_step_count += 1
        elif mode is DecodeRuntimeMode.REFRESH_COMMIT:
            self.counters.refresh_commit_step_count += 1
        elif mode is DecodeRuntimeMode.FULL_RECOMPILE:
            self.counters.full_recompile_count += 1
        elif mode is DecodeRuntimeMode.SIG_RETURN_DELTA:
            self.counters.sig_return_step_count += 1

    def apply_sig_return(
        self,
        current_guard: DecodeStaticGuard,
        current_delta: DecodeDeltaPacket,
    ) -> None:
        """[LITE-P0] Record a SIG_RETURN_DELTA step OUTSIDE classify.

        The signature-return arm proves (before calling this) that the step
        is a pure signature flip-back to the pre-trigger steady regime with a
        value plane rebuilt from ground truth. classify itself would compare
        against the TRIGGER delta (q/logf changed) and land FULL_RECOMPILE,
        so the arm advances the state manually. The NEXT step's classify then
        compares against THIS delta and lands STEADY_DELTA naturally.
        """
        self.active_guard = current_guard
        self.last_delta = current_delta
        self.last_mode = DecodeRuntimeMode.SIG_RETURN_DELTA
        self.last_reason = "sig_return_steady_resume"
        self.last_applied_step_id = current_delta.step_id
        self._increment_counter(DecodeRuntimeMode.SIG_RETURN_DELTA)


def classify_decode_runtime_mode(
    active_guard: DecodeStaticGuard | None,
    previous_delta: DecodeDeltaPacket | None,
    current_guard: DecodeStaticGuard,
    current_delta: DecodeDeltaPacket,
) -> tuple[DecodeRuntimeMode, str]:
    if active_guard is None or previous_delta is None:
        return DecodeRuntimeMode.FULL_RECOMPILE, "uninitialized"
    if current_guard != active_guard:
        return DecodeRuntimeMode.FULL_RECOMPILE, "static_guard_changed"
    if current_delta.batch_size != current_guard.batch_size:
        return DecodeRuntimeMode.FULL_RECOMPILE, "guard_delta_batch_mismatch"
    if current_delta.batch_size != previous_delta.batch_size:
        return DecodeRuntimeMode.FULL_RECOMPILE, "batch_size_changed"
    if current_delta.q_lens_by_row != previous_delta.q_lens_by_row:
        return DecodeRuntimeMode.FULL_RECOMPILE, "q_layout_changed"
    if current_delta.logf_mask_generation != previous_delta.logf_mask_generation:
        return DecodeRuntimeMode.FULL_RECOMPILE, "logf_generation_changed"
    if current_delta.pending_refresh_state == "ready":
        return DecodeRuntimeMode.REFRESH_COMMIT, "pending_refresh_ready"
    if current_delta.pending_refresh_state not in ("nil", "empty"):
        return DecodeRuntimeMode.FULL_RECOMPILE, "pending_refresh_not_steady"
    if current_delta.row_dynamic_signature != previous_delta.row_dynamic_signature:
        # The five-field row signature is
        # (slot, row_mode, use_compact, compact_valid, compact_offset).
        # Only row_mode is a true row-table dynamic.  The other four fields
        # select static launch-plan rows 0/1 and the compact offset mirror;
        # apply_launch_template_row_delta updates rows 2..5 only.  Treating a
        # bootstrap handoff (use_compact 0->1) as PAGE_BOUNDARY_DELTA reused the
        # previous full-row layout, so RRP observed e.g. slot=2 with offset=0.
        # Project/compare only on this already-cold mismatch branch: steady
        # classification keeps its existing single tuple comparison.
        previous_static = _row_static_layout_signature(
            previous_delta.row_dynamic_signature
        )
        current_static = _row_static_layout_signature(
            current_delta.row_dynamic_signature
        )
        # PAGE_BOUNDARY_DELTA is safe only when both signatures prove that
        # every static launch field stayed identical.  Missing/malformed
        # signatures are an absence of proof, not evidence of equality.
        if previous_static is None or current_static is None:
            return DecodeRuntimeMode.FULL_RECOMPILE, "row_static_layout_unproven"
        if current_static != previous_static:
            return DecodeRuntimeMode.FULL_RECOMPILE, "row_static_layout_changed"
        return DecodeRuntimeMode.PAGE_BOUNDARY_DELTA, "row_dynamic_layout_changed"
    if (
        current_delta.recent_first_page_by_row != previous_delta.recent_first_page_by_row
        or current_delta.recent_page_count_by_row != previous_delta.recent_page_count_by_row
    ):
        return DecodeRuntimeMode.PAGE_BOUNDARY_DELTA, "recent_window_changed"
    return DecodeRuntimeMode.STEADY_DELTA, "same_page_row_delta"


def _row_static_layout_signature(
    row_dynamic_signature: tuple[object, ...],
) -> tuple[tuple[object, object, object, object], ...] | None:
    """Project fields that cannot be updated by a launch-template row delta."""
    projected = []
    for raw_row in row_dynamic_signature:
        if not isinstance(raw_row, (tuple, list)) or len(raw_row) != 5:
            return None
        projected.append((raw_row[0], raw_row[2], raw_row[3], raw_row[4]))
    return tuple(projected) if projected else None


def _page_aligned_recent_first_visible(
    real_kv_len: int,
    page_size: int,
    recent_tokens: int,
) -> tuple[int, int]:
    """Pure-int replica of derive_page_aligned_recent_window
    (patches/fa_sparse_runtime/materialize.py:23-54) with min_start_token=0,
    the Site B decode-delta default. Returns (first_logical_page,
    visible_tokens). Verified byte-identical to Site B across page sizes
    16..256 and all real/recent_tokens; kept inline so this module stays
    import-free (no torch / no sync tokens).
    """
    page_i = int(page_size)
    real_i = max(0, int(real_kv_len))
    recent_i = max(0, int(recent_tokens))
    if real_i <= 0 or recent_i <= 0:
        first_page = (real_i + page_i - 1) // page_i
        return first_page, 0
    start_floor = max(0, real_i - recent_i)
    start_token = (start_floor // page_i) * page_i
    start_token = min(start_token, real_i)
    first_page = start_token // page_i
    visible_tokens = max(0, real_i - start_token)
    return first_page, visible_tokens


def extrapolate_sig_return_delta(
    *,
    pre_trigger_delta: DecodeDeltaPacket | None,
    trigger_delta: DecodeDeltaPacket | None,
    step_id: int,
    page_size: int,
    q_lens_by_row: Any,
    logf_mask_generation: Any,
    pending_refresh_state: str,
    real_kv_len_by_row: Any,
    row_dynamic_signature: Any = tuple(),
    recent_tokens: int = -1,
) -> tuple[DecodeDeltaPacket | None, str]:
    """[LITE-P0 fail-close #3] Two-step cluster extrapolation for the
    signature-return arm — the dedicated entry predict_same_page_delta cannot
    serve (its nonconsecutive_step guard is correct for its caller and MUST
    NOT be bypassed).

    Reconstructs the return step's value plane from the PRE-TRIGGER steady
    packet using invariants only (no per-step guesses): the true advance is
    ``real_now - real_pre`` where real_pre is recovered from the carried
    packet via real = first_page*page_size + visible (same invariant predict
    uses) and real_now comes from the caller's step_authority truth. The
    recent window is then RE-DERIVED for real_now; if the first logical page
    advanced anywhere inside the cluster the row slid a page boundary and we
    refuse (the full path's PBD handling must run — a carried-forward stale
    first page would hide the slide exactly like the p12 bug this mirrors).

    Refuses (None, reason) on ANY doubt: missing packets, non-2-step cluster,
    batch drift, q/logf not flipped back to the steady signature, refresh
    pending, implausible advance, window reconstruction mismatch, or a
    boundary slide inside the cluster.
    """
    if pre_trigger_delta is None or trigger_delta is None:
        return None, "uninitialized"
    # Strict 2-step cluster shape: steady(t0) -> trigger(t0+1) -> return(t0+2).
    if int(trigger_delta.step_id) != int(pre_trigger_delta.step_id) + 1:
        return None, "nonconsecutive_cluster"
    if int(step_id) != int(trigger_delta.step_id) + 1:
        return None, "nonconsecutive_cluster"
    page_size_i = int(page_size)
    if page_size_i <= 0:
        return None, "invalid_page_size"
    batch_size = int(pre_trigger_delta.batch_size)
    if int(trigger_delta.batch_size) != batch_size:
        return None, "batch_size_changed"
    q_lens = _int_tuple(q_lens_by_row)
    if len(q_lens) != batch_size or q_lens != pre_trigger_delta.q_lens_by_row:
        return None, "q_layout_changed"
    if any(int(q) <= 0 for q in q_lens):
        return None, "q_layout_changed"
    logf_signature = _int_tuple(logf_mask_generation)
    if (
        len(logf_signature) != batch_size
        or logf_signature != pre_trigger_delta.logf_mask_generation
    ):
        return None, "logf_generation_changed"
    refresh_state = _str_value(pending_refresh_state)
    if refresh_state not in ("nil", "empty"):
        return None, "pending_refresh_not_steady"
    real_now_tuple = _int_tuple(real_kv_len_by_row)
    if len(real_now_tuple) != batch_size:
        return None, "real_kv_len_coverage"
    recent_tokens_i = int(recent_tokens)
    if recent_tokens_i <= 0:
        return None, "invalid_recent_window"

    next_request_recent_len: list[int] = []
    next_launch_effective_k: list[int] = []
    next_row_effective_k: list[int] = []
    for row in range(batch_size):
        prev_first = int(pre_trigger_delta.recent_first_page_by_row[row])
        prev_visible = int(pre_trigger_delta.request_recent_len_by_row[row])
        recent_count = int(pre_trigger_delta.recent_page_count_by_row[row])
        if recent_count < 0:
            return None, "invalid_recent_window"
        prev_real = prev_first * page_size_i + prev_visible
        # Rows the steady packet did NOT place on the page-aligned window path
        # (dense/recent_cap branch) cannot be reconstructed — refuse rather
        # than guess (mirrors predict's reconstruction-agreement guard).
        recon_first, recon_visible = _page_aligned_recent_first_visible(
            prev_real, page_size_i, recent_tokens_i
        )
        if recon_first != prev_first or recon_visible != prev_visible:
            return None, "window_reconstruction_mismatch"
        real_now = int(real_now_tuple[row])
        advance = real_now - prev_real
        # Sanity: a 2-step cluster advances each row by exactly the tokens
        # generated across trigger+return (1/step in this runtime's decode;
        # allow a small bound rather than hard-coding 2 so gt1 query layouts
        # stay admissible, but refuse implausible jumps).
        if advance < 1 or advance > 4:
            return None, "implausible_advance"
        new_first, new_visible = _page_aligned_recent_first_visible(
            real_now, page_size_i, recent_tokens_i
        )
        if new_first != prev_first:
            # Boundary slid somewhere inside the cluster: the carried first
            # page is stale — full path (PBD) must handle this step.
            return None, "page_boundary_in_cluster"
        if new_visible > recent_count * page_size_i:
            return None, "page_boundary_countdown_expired"
        next_request_recent_len.append(new_visible)
        next_launch_effective_k.append(
            int(pre_trigger_delta.launch_effective_k_by_row[row])
            + (new_visible - prev_visible)
        )
        next_row_effective_k.append(
            int(pre_trigger_delta.row_effective_k_by_row[row]) + advance
        )

    return (
        DecodeDeltaPacket(
            step_id=int(step_id),
            batch_size=batch_size,
            row_effective_k_by_row=tuple(next_row_effective_k),
            request_recent_len_by_row=tuple(next_request_recent_len),
            launch_effective_k_by_row=tuple(next_launch_effective_k),
            recent_first_page_by_row=pre_trigger_delta.recent_first_page_by_row,
            recent_page_count_by_row=pre_trigger_delta.recent_page_count_by_row,
            q_lens_by_row=q_lens,
            logf_mask_generation=logf_signature,
            pending_refresh_state=refresh_state,
            row_dynamic_signature=tuple(row_dynamic_signature),
        ),
        "sig_return_extrapolated",
    )


def predict_same_page_delta(
    *,
    previous_delta: DecodeDeltaPacket | None,
    step_id: int,
    page_size: int,
    q_lens_by_row: Any,
    logf_mask_generation: Any,
    pending_refresh_state: str,
    row_dynamic_signature: Any = tuple(),
    recent_tokens: int = -1,
) -> tuple[DecodeDeltaPacket | None, str]:
    if previous_delta is None:
        return None, "uninitialized"
    if int(step_id) != int(previous_delta.step_id) + 1:
        return None, "nonconsecutive_step"
    page_size_i = int(page_size)
    if page_size_i <= 0:
        return None, "invalid_page_size"
    batch_size = int(previous_delta.batch_size)
    q_lens = _int_tuple(q_lens_by_row)
    if len(q_lens) != batch_size:
        return None, "q_layout_changed"
    if q_lens != previous_delta.q_lens_by_row:
        return None, "q_layout_changed"
    if any(int(q) <= 0 for q in q_lens):
        return None, "q_layout_changed"
    logf_signature = _int_tuple(logf_mask_generation)
    if len(logf_signature) != batch_size:
        return None, "logf_generation_changed"
    if logf_signature != previous_delta.logf_mask_generation:
        return None, "logf_generation_changed"
    refresh_state = _str_value(pending_refresh_state)
    if refresh_state == "ready":
        return None, "pending_refresh_ready"
    if refresh_state not in ("nil", "empty"):
        return None, "pending_refresh_not_steady"

    next_request_recent_len: list[int] = []
    next_launch_effective_k: list[int] = []
    next_row_effective_k: list[int] = []
    for row, q_len in enumerate(q_lens):
        request_len = int(previous_delta.request_recent_len_by_row[row]) + int(q_len)
        recent_count = int(previous_delta.recent_page_count_by_row[row])
        if recent_count < 0:
            return None, "invalid_recent_window"
        if request_len > recent_count * page_size_i:
            return None, "page_boundary_countdown_expired"
        # [p12 true root fix 20260612] The countdown above only catches the
        # END page growing; it is BLIND to the recent window's FIRST LOGICAL
        # PAGE advancing while the visible length stays inside the window.
        # On such a same-count slide the verbatim copy of
        # recent_first_page_by_row below would carry the STALE first page,
        # hiding the slide from classify_decode_runtime_mode (misclassified
        # steady_delta -> the slide-capable PAGE_BOUNDARY_DELTA path is
        # unreachable -> 353us heavy bind every slide). Recompute Site B's
        # first_logical_page for the NEW request_len and refuse if it would
        # advance, so the caller falls to the fresh Site-B packet whose
        # recent_window change the classifier (correctly) routes to PBD.
        # Mirrors derive_page_aligned_recent_window
        # (patches/fa_sparse_runtime/materialize.py:23-54, min_start_token=0,
        # the Site B delta-path default) byte-for-byte; recent_tokens is the
        # step's recent_cap. real is recovered EXACTLY from the carried
        # packet via the invariant real = first_page*page_size + visible.
        recent_tokens_i = int(recent_tokens)
        if recent_tokens_i > 0:
            prev_first = int(previous_delta.recent_first_page_by_row[row])
            prev_visible = int(previous_delta.request_recent_len_by_row[row])
            prev_real = prev_first * page_size_i + prev_visible
            recon_first, recon_visible = _page_aligned_recent_first_visible(
                prev_real, page_size_i, recent_tokens_i
            )
            # Only trust the slide test for rows Site B actually placed on the
            # page-aligned window path; if reconstruction disagrees the row was
            # on Site B's dense/recent_cap branch (first page pinned at 0, never
            # slides) and we must NOT claim a slide.
            if recon_first == prev_first and recon_visible == prev_visible:
                new_real = prev_real + int(q_len)
                new_first, _new_visible = _page_aligned_recent_first_visible(
                    new_real, page_size_i, recent_tokens_i
                )
                if new_first != prev_first:
                    return None, "page_boundary_window_slide"
        next_request_recent_len.append(request_len)
        next_launch_effective_k.append(
            int(previous_delta.launch_effective_k_by_row[row]) + int(q_len)
        )
        next_row_effective_k.append(
            int(previous_delta.row_effective_k_by_row[row]) + int(q_len)
        )

    return (
        DecodeDeltaPacket(
            step_id=int(step_id),
            batch_size=batch_size,
            row_effective_k_by_row=tuple(next_row_effective_k),
            request_recent_len_by_row=tuple(next_request_recent_len),
            launch_effective_k_by_row=tuple(next_launch_effective_k),
            recent_first_page_by_row=previous_delta.recent_first_page_by_row,
            recent_page_count_by_row=previous_delta.recent_page_count_by_row,
            q_lens_by_row=q_lens,
            logf_mask_generation=logf_signature,
            pending_refresh_state=refresh_state,
            row_dynamic_signature=tuple(row_dynamic_signature),
        ),
        "predicted_same_page_row_delta",
    )
