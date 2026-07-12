from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Tuple

import torch

import os as _os
import atexit as _atexit
import json as _json
_RRP_MISS_PROBE = _os.environ.get("VLLM_SPARSE_RRP_MISS_PROBE") == "1"
_RRP_MISS_ACC = {"vpc": 0, "recent_first": 0, "structural": 0, "samples": []}
def _rrp_miss_probe_log(reason, computed, stored, row_eff, compact_valid):
    if not _RRP_MISS_PROBE:
        return
    _RRP_MISS_ACC[reason] = _RRP_MISS_ACC.get(reason, 0) + 1
    if len(_RRP_MISS_ACC["samples"]) < 30:
        try:
            _RRP_MISS_ACC["samples"].append({
                "reason": reason,
                "computed": list(computed) if computed is not None else None,
                "stored": list(stored) if stored is not None else None,
                "row_eff": [int(v) for v in row_eff] if row_eff is not None else None,
                "compact_valid": list(compact_valid) if compact_valid is not None else None,
            })
        except Exception:
            pass
def _rrp_miss_probe_flush():
    if not _RRP_MISS_PROBE:
        return
    try:
        p = _os.environ.get("VLLM_SPARSE_RRP_MISS_PROBE_LOG", "logs/rrp_miss_probe.json")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(_json.dumps(_RRP_MISS_ACC, indent=2))
    except Exception:
        pass
_atexit.register(_rrp_miss_probe_flush)


class RrpUpdateKind(str, Enum):
    INITIAL_BIND = "initial_bind"
    SIGNATURE_CHANGED = "signature_changed"
    PAGE_BOUNDARY_DELTA = "page_boundary_delta"
    SAME_PAGE_DELTA = "same_page_delta"
    HIT = "hit"


@dataclass(frozen=True, slots=True)
class RrpRowTableInputs:
    batch_size: int
    req_ids_by_row: Tuple[str, ...]
    slot_by_row: Tuple[int, ...]
    row_mode_by_row: Tuple[int, ...]
    compact_ready_by_row: Tuple[bool, ...]
    row_effective_k_by_row: Tuple[int, ...]
    compact_valid_tokens_by_row: Tuple[int, ...]
    compact_offset_tokens_by_row: Tuple[int, ...]
    recent_first_page_by_row: Tuple[int, ...]
    reserved_manager_block_ids: Tuple[int, ...]
    compact_capacity_pages: int
    max_pages_per_row: int
    page_size: int
    row_effective_k_i32_gpu: torch.Tensor | None = None

    def __post_init__(self) -> None:
        batch_size = int(self.batch_size)
        object.__setattr__(self, "batch_size", batch_size)
        if batch_size < 0:
            raise ValueError("batch_size must be non-negative")
        for name in (
            "req_ids_by_row",
            "slot_by_row",
            "row_mode_by_row",
            "compact_ready_by_row",
            "row_effective_k_by_row",
            "compact_valid_tokens_by_row",
            "compact_offset_tokens_by_row",
            "recent_first_page_by_row",
        ):
            values = tuple(getattr(self, name))
            if len(values) < batch_size:
                raise ValueError(f"{name} must cover batch_size")
            object.__setattr__(self, name, values)
        object.__setattr__(
            self,
            "req_ids_by_row",
            tuple(str(v) for v in self.req_ids_by_row),
        )
        object.__setattr__(
            self,
            "compact_ready_by_row",
            tuple(bool(v) for v in self.compact_ready_by_row),
        )
        for name in (
            "slot_by_row",
            "row_mode_by_row",
            "row_effective_k_by_row",
            "compact_valid_tokens_by_row",
            "compact_offset_tokens_by_row",
            "recent_first_page_by_row",
            "reserved_manager_block_ids",
        ):
            values = getattr(self, name)
            if (
                name == "reserved_manager_block_ids"
                and isinstance(values, tuple)
                and (not values or isinstance(values[0], int))
            ):
                # Lease snapshots already normalize this tuple; keep identity for hot-path compares.
                coerced = values
            else:
                coerced = tuple(int(v) for v in values)
            object.__setattr__(self, name, coerced)
        for name in ("compact_capacity_pages", "max_pages_per_row", "page_size"):
            object.__setattr__(self, name, int(getattr(self, name)))


@dataclass(frozen=True, slots=True)
class RrpUpdateResult:
    hit: bool
    miss_reason: str
    delta_rows: Tuple[int, ...]
    full_bind: bool
    kind: RrpUpdateKind = RrpUpdateKind.HIT
    metadata_bind_required: bool = False
    update_kernel_count: int = 0

    @property
    def is_same_page_delta(self) -> bool:
        return self.kind is RrpUpdateKind.SAME_PAGE_DELTA


def should_record_rrp_ready_event(
    update: RrpUpdateResult,
    *,
    metadata_bind_required: bool,
    existing_ready_event_generation: int,
) -> bool:
    if int(existing_ready_event_generation) < 0:
        return True
    if bool(metadata_bind_required):
        return True
    if bool(update.full_bind) or bool(update.delta_rows):
        return True
    return int(update.update_kernel_count) > 0


class RrpRowTableManager:
    def __init__(self) -> None:
        self._signature: tuple[object, ...] | None = None
        self._row_effective_k_by_row: Tuple[int, ...] | None = None
        self._recent_first_page_by_row: Tuple[int, ...] | None = None
        self._visible_page_count_by_row: Tuple[int, ...] | None = None
        # [VPC-SHADOW-METRIC 2026-07-08] same-page probe comparand: the
        # arithmetic (token-ceil) page count evaluated AT WRITE TIME with the
        # same inputs the probe recomputes later. _visible_page_count_by_row
        # stays the writers' physical laid-page count (len semantics) for the
        # publish dirty-diff; comparing the probe's arithmetic recompute
        # against THAT mixes metrics whose recent-window phase differs by ±1
        # (window-span vs token-ceil; floor-vs-ceil compact pages in the
        # page-add leg) and mass-missed genuine same-page STEADY steps. The
        # shadow makes the probe self-consistent: page-boundary crossings
        # still flip token-ceil and miss (full bind re-lays the row), while a
        # pure metric mismatch no longer forces a spurious full bind. NOTE the
        # rejected alternative: admitting computed<=stored (PAGE_GE) changed
        # golden output — it admitted steps whose page tables genuinely
        # needed re-laying. The shadow admits ONLY steps where the detection
        # inputs are bitwise-unchanged.
        self._probe_visible_page_count_by_row: Tuple[int, ...] | None = None
        self._row_table_layout_by_row: Tuple[tuple[object, ...], ...] | None = None
        # Publish-only comparand for publish_from_descriptor_snapshot's dirty diff,
        # kept DISTINCT from _row_table_layout_by_row (which update() writes in a
        # 7-tuple encoding) so the descriptor 5-tuple diff is never a permanent
        # representation mismatch (that made every row spuriously dirty every step).
        self._published_descriptor_layout_by_row: Tuple[object, ...] | None = None
        self._seqused_staging_cpu_i32: torch.Tensor | None = None
        self._seqused_staging_gpu_i32: torch.Tensor | None = None
        self._descriptor_snapshot_epoch: int | None = None
        self._compact_ready_by_row: Tuple[bool, ...] | None = None
        self._compact_valid_tokens_by_row: Tuple[int, ...] | None = None
        self._page_size: int | None = None
        self._max_pages_per_row: int | None = None
        # Set by try_apply_live_page_boundary on a non-apply so the dev can see the
        # specific reason and root-cause it (never a silent fall-back to heavy).
        self.last_failure_reason: str | None = None

    def update(self, arena: object, inputs: RrpRowTableInputs) -> RrpUpdateResult:
        signature = self._signature_for(inputs)
        self._compact_ready_by_row = tuple(
            bool(v) for v in inputs.compact_ready_by_row[: inputs.batch_size]
        )
        self._compact_valid_tokens_by_row = tuple(
            int(v) for v in inputs.compact_valid_tokens_by_row[: inputs.batch_size]
        )
        self._page_size = int(inputs.page_size)
        self._max_pages_per_row = int(inputs.max_pages_per_row)
        row_effective_k = inputs.row_effective_k_by_row[: inputs.batch_size]
        recent_first_page = inputs.recent_first_page_by_row[: inputs.batch_size]
        visible_page_count = self._visible_page_count_for(inputs, row_effective_k)
        row_table_layout = self._row_table_layout_for(inputs, visible_page_count)

        if self._signature is None:
            self._store(
                signature,
                row_effective_k,
                recent_first_page,
                visible_page_count,
                row_table_layout,
            )
            return RrpUpdateResult(
                hit=False,
                miss_reason="initial_bind",
                delta_rows=tuple(range(inputs.batch_size)),
                full_bind=True,
                kind=RrpUpdateKind.INITIAL_BIND,
                metadata_bind_required=True,
            )

        if signature != self._signature:
            self._store(
                signature,
                row_effective_k,
                recent_first_page,
                visible_page_count,
                row_table_layout,
            )
            return RrpUpdateResult(
                hit=False,
                miss_reason="signature_changed",
                delta_rows=tuple(range(inputs.batch_size)),
                full_bind=True,
                kind=RrpUpdateKind.SIGNATURE_CHANGED,
                metadata_bind_required=True,
            )

        previous_recent = self._recent_first_page_by_row or ()
        previous_visible = self._visible_page_count_by_row or ()
        previous_layout = self._row_table_layout_by_row or ()
        delta_rows = tuple(
            row
            for row in range(inputs.batch_size)
            if (
                row >= len(previous_recent)
                or int(previous_recent[row]) != recent_first_page[row]
                or row >= len(previous_visible)
                or int(previous_visible[row]) != visible_page_count[row]
                or row >= len(previous_layout)
                or previous_layout[row] != row_table_layout[row]
            )
        )
        if delta_rows:
            self._store(
                signature,
                row_effective_k,
                recent_first_page,
                visible_page_count,
                row_table_layout,
            )
            return RrpUpdateResult(
                hit=False,
                miss_reason="page_boundary_delta",
                delta_rows=delta_rows,
                full_bind=False,
                kind=RrpUpdateKind.PAGE_BOUNDARY_DELTA,
            )

        if row_effective_k != self._row_effective_k_by_row:
            update_kernel_count = self._seqused_update_kernel_count(arena)
            if not self._try_increment_seqused_k(arena, row_effective_k):
                self._write_seqused_k(
                    arena,
                    row_effective_k,
                    row_effective_k_i32_gpu=inputs.row_effective_k_i32_gpu,
                )
            self._store(
                signature,
                row_effective_k,
                recent_first_page,
                visible_page_count,
                row_table_layout,
            )
            return RrpUpdateResult(
                hit=False,
                miss_reason="same_page_delta",
                delta_rows=(),
                full_bind=False,
                kind=RrpUpdateKind.SAME_PAGE_DELTA,
                update_kernel_count=update_kernel_count,
            )

        return RrpUpdateResult(
            hit=True,
            miss_reason="hit",
            delta_rows=(),
            full_bind=False,
            kind=RrpUpdateKind.HIT,
        )

    def try_apply_same_page_delta(
        self,
        arena: object,
        row_effective_k_by_row: Iterable[int],
        *,
        recent_first_page_by_row: Iterable[int] | None = None,
    ) -> RrpUpdateResult | None:
        # Decode-delta fast path: stay seqused-only ONLY when the step is a pure
        # length change with no page-mapping movement. The same-page predicate
        # mirrors update()'s page/recent delta detection; otherwise miss to the
        # full path, which re-derives pages from the live block table on the same
        # captured graph. See
        # docs/SM100_SPARSE_DECODE_PAGE_COHERENCE_FIX_SPEC_2026-05-31.md.
        row_effective_k = tuple(int(v) for v in row_effective_k_by_row)
        if not self._is_pure_seqused_delta(row_effective_k, recent_first_page_by_row):
            return None
        return self._apply_seqused_delta(arena, row_effective_k)

    def can_apply_same_page_delta(
        self,
        row_effective_k_by_row: Iterable[int],
        *,
        recent_first_page_by_row: Iterable[int] | None = None,
    ) -> bool:
        """Read-only admission precheck for ``try_apply_same_page_delta``.

        Makes the SAME ``_is_pure_seqused_delta`` call with the SAME argument
        shapes as ``try_apply_same_page_delta`` (normalized int tuple +
        recent-first iterable passed through raw) and never touches arena
        tensors or manager state. When the predicate holds the apply path
        cannot return None: the predicate already proved the stored
        signature/baseline exist and the row count matches -- everything
        ``_apply_seqused_delta`` re-checks -- so True here <=> the apply call
        would return a non-None result. Used by the metadata-builder steady
        fast-path admission gate (#12 ultra) to fail fast BEFORE the heavy
        per-step work instead of discovering the terminal rrp miss after it.
        """
        return self._is_pure_seqused_delta(
            tuple(int(v) for v in row_effective_k_by_row),
            recent_first_page_by_row,
        )

    def _is_pure_seqused_delta(
        self,
        row_effective_k: Tuple[int, ...],
        recent_first_page_by_row: Iterable[int] | None,
    ) -> bool:
        # [VPC-SHADOW-METRIC] compare token-ceil(new effk) against
        # token-ceil(write-epoch effk) — the shadow — instead of the writers'
        # physical laid-page count. Same-metric comparison: a page-boundary
        # crossing since the last write still flips token-ceil and misses; a
        # pure metric phase difference no longer forces a spurious full bind.
        stored_visible = self._probe_visible_page_count_by_row
        if (
            self._signature is None
            or self._row_effective_k_by_row is None
            or stored_visible is None
            or self._compact_ready_by_row is None
            or self._compact_valid_tokens_by_row is None
            or self._page_size is None
            or self._max_pages_per_row is None
            or len(self._row_effective_k_by_row) != len(row_effective_k)
            or len(stored_visible) != len(row_effective_k)
        ):
            _rrp_miss_probe_log("structural", None, None, row_effective_k, None)
            return False
        _rrp_probe_vpc = self._visible_page_count_from_fields(
            batch_size=len(row_effective_k),
            row_effective_k=row_effective_k,
            compact_ready_by_row=self._compact_ready_by_row,
            compact_valid_tokens_by_row=self._compact_valid_tokens_by_row,
            page_size=self._page_size,
            max_pages_per_row=self._max_pages_per_row,
        )
        if _rrp_probe_vpc != stored_visible:
            _rrp_miss_probe_log("vpc", _rrp_probe_vpc, stored_visible, row_effective_k, self._compact_valid_tokens_by_row)
            # [RRP-PAGE-GE-HIT-DELETED 2026-07-08] the "computed <= stored
            # admits" arm (env VLLM_SPARSE_RRP_PAGE_GE_HIT) is DELETED, not
            # just default-off: enabling it flipped golden output
            # deterministically ({0d5f663c,636fb032} -> {894b571a,8b35f6c6}).
            # Subset-read is OOB-safe but NOT content-fresh — a page-count
            # change is exactly the signal that the row's table needs
            # re-laying. Any future fast-path here must re-lay pages (see
            # PAGE_ADD_INCREMENTAL), never skip the re-lay.
            return False
        if recent_first_page_by_row is not None and self._recent_first_page_by_row is not None:
            recent = tuple(int(v) for v in recent_first_page_by_row)
            if (
                len(recent) == len(self._recent_first_page_by_row)
                and recent != self._recent_first_page_by_row
            ):
                _rrp_miss_probe_log("recent_first", recent, self._recent_first_page_by_row, row_effective_k, self._compact_valid_tokens_by_row)
                return False
        return True

    def _apply_seqused_delta(
        self,
        arena: object,
        row_effective_k: Tuple[int, ...],
    ) -> RrpUpdateResult | None:
        previous = self._row_effective_k_by_row
        if self._signature is None or previous is None:
            return None
        if len(row_effective_k) != len(previous):
            return None
        if row_effective_k == previous:
            return RrpUpdateResult(
                hit=True,
                miss_reason="hit",
                delta_rows=(),
                full_bind=False,
                kind=RrpUpdateKind.HIT,
            )
        batch_target = self._arena_batch_seqused_write_target(arena)
        update_kernel_count = self._seqused_update_kernel_count(
            arena, batch_target
        )
        if not self._try_increment_seqused_k(
            arena, row_effective_k, batch_target
        ):
            self._write_seqused_k(arena, row_effective_k)
        self._row_effective_k_by_row = row_effective_k
        return RrpUpdateResult(
            hit=False,
            miss_reason="same_page_delta",
            delta_rows=(),
            full_bind=False,
            kind=RrpUpdateKind.SAME_PAGE_DELTA,
            update_kernel_count=update_kernel_count,
        )

    def update_from_descriptor_snapshot(
        self,
        arena: object,
        snapshot: object,
        *,
        recent_first_page_by_row: Iterable[int] | None = None,
    ) -> RrpUpdateResult | None:
        row_effective_k = self.row_effective_from_descriptor_snapshot(snapshot)
        if row_effective_k is None:
            return None
        # Native snapshot page coherence: gate the seqused-only write with the
        # SAME same-page predicate as the decode-delta fast path. The predicate
        # recomputes the mode-aware page count from the STORED compact fields (set
        # by the last full bind off the live block table), so a recent-window
        # page-boundary cross is detected even though the snapshot's descriptor
        # pages are carried forward from step_bound_meta (stale-vs-stale).
        # Callers that have decode-delta state also pass recent_first_page_by_row
        # so same-count recent-window slides miss instead of reusing stale pages.
        # On a miss we return None and the caller falls through to the full
        # rebuild that re-derives pages from the live block table on the same
        # captured graph (no rebind, addresses stable, dirty rows only).
        # See docs/SM100_SPARSE_DECODE_PAGE_COHERENCE_FIX_SPEC_2026-05-31.md.
        row_effective_k = tuple(int(v) for v in row_effective_k)
        recent_first = self._recent_first_from_descriptor_snapshot(
            snapshot,
            recent_first_page_by_row,
            batch_size=len(row_effective_k),
        )
        if not self._is_pure_seqused_delta(row_effective_k, recent_first):
            return None
        update = self._apply_seqused_delta(arena, row_effective_k)
        if update is None:
            return None
        self._descriptor_snapshot_epoch = int(getattr(snapshot, "epoch"))
        return update

    def _recent_first_from_descriptor_snapshot(
        self,
        snapshot: object,
        recent_first_page_by_row: Iterable[int] | None,
        *,
        batch_size: int,
    ) -> Tuple[int, ...] | None:
        if recent_first_page_by_row is None:
            return None
        try:
            recent_by_live_row = tuple(int(v) for v in recent_first_page_by_row)
        except Exception:
            return tuple(-1 for _ in range(int(batch_size)))
        if len(recent_by_live_row) == int(batch_size):
            return recent_by_live_row
        stored = self._recent_first_page_by_row
        if isinstance(stored, tuple) and len(stored) == int(batch_size):
            projected = [int(v) for v in stored]
        else:
            projected = [0 for _ in range(int(batch_size))]
        try:
            rows_by_graph_row = tuple(getattr(snapshot, "rows_by_graph_row"))
        except Exception:
            return tuple(-1 for _ in range(int(batch_size)))
        for graph_row, row in enumerate(rows_by_graph_row[: int(batch_size)]):
            if row is None:
                continue
            try:
                live_row = int(getattr(row, "live_row_index"))
            except Exception:
                return tuple(-1 for _ in range(int(batch_size)))
            if live_row < 0 or live_row >= len(recent_by_live_row):
                return tuple(-1 for _ in range(int(batch_size)))
            projected[int(graph_row)] = int(recent_by_live_row[live_row])
        return tuple(projected)

    def publish_from_descriptor_snapshot(
        self,
        arena: object,
        snapshot: object,
    ) -> RrpUpdateResult | None:
        row_effective_k = self.row_effective_from_descriptor_snapshot(snapshot)
        if row_effective_k is None:
            return None
        previous = self._row_effective_k_by_row
        previous_visible = self._visible_page_count_by_row
        # Diff against the PUBLISH-only descriptor layout (5-tuple form), NOT
        # _row_table_layout_by_row: update() overwrites the latter with a 7-tuple
        # (_row_table_layout_for) on every non-HIT step, so diffing it here would be
        # a permanent representation mismatch -> every row spuriously dirty every
        # step. A missing/short published comparand just means "no baseline yet" ->
        # publish once to establish it, then same-page steps diff equal (fast path).
        previous_published = self._published_descriptor_layout_by_row
        if (
            self._signature is None
            or previous is None
            or previous_visible is None
            or len(row_effective_k) != len(previous)
            or len(previous_visible) != len(previous)
        ):
            return None
        published_ok = (
            isinstance(previous_published, tuple)
            and len(previous_published) == len(previous)
        )
        try:
            snapshot_epoch = int(getattr(snapshot, "epoch"))
            rows_by_graph_row = tuple(getattr(snapshot, "rows_by_graph_row"))
        except (AttributeError, TypeError, ValueError):
            return None
        if len(rows_by_graph_row) < len(previous):
            return None

        visible_page_count = list(int(v) for v in previous_visible)
        published_layout = (
            list(previous_published) if published_ok else [None] * len(previous)
        )
        dirty_rows: list[int] = []
        for graph_row in range(len(previous)):
            descriptor = rows_by_graph_row[graph_row]
            if descriptor is None:
                continue
            if not self._descriptor_row_epoch_current(
                descriptor,
                graph_row=graph_row,
                snapshot_epoch=snapshot_epoch,
            ):
                return None
            layout = self._descriptor_row_table_layout(descriptor)
            if layout is None:
                return None
            page_count = len(layout[2])
            if (
                not published_ok
                or int(previous_visible[graph_row]) != int(page_count)
                or published_layout[graph_row] != layout
            ):
                dirty_rows.append(graph_row)
            visible_page_count[graph_row] = int(page_count)
            published_layout[graph_row] = layout

        if dirty_rows:
            publish = getattr(arena, "publish_descriptor_rows", None)
            if not callable(publish):
                return None
            if not bool(publish(snapshot=snapshot, dirty_rows=tuple(dirty_rows))):
                return None
            # Advance shared length/visible state and the publish-only comparand,
            # but do NOT touch _row_table_layout_by_row (update()'s 7-tuple field)
            # so update()'s own delta detection stays consistent across phases.
            self._row_effective_k_by_row = row_effective_k
            self._visible_page_count_by_row = tuple(visible_page_count)
            # [VPC-SHADOW-METRIC] visible_page_count above is the descriptor
            # layout's laid-page count; the probe compares token-ceil, so
            # refresh the shadow at the same write epoch.
            self._probe_visible_page_count_by_row = self._shadow_visible_pages(
                row_effective_k
            )
            self._published_descriptor_layout_by_row = tuple(published_layout)
            self._descriptor_snapshot_epoch = snapshot_epoch
            return RrpUpdateResult(
                hit=False,
                miss_reason="page_boundary_delta",
                delta_rows=tuple(dirty_rows),
                full_bind=False,
                kind=RrpUpdateKind.PAGE_BOUNDARY_DELTA,
            )

        update = self.try_apply_same_page_delta(arena, row_effective_k)
        if update is None:
            return None
        self._descriptor_snapshot_epoch = snapshot_epoch
        return update

    def try_apply_live_page_boundary(
        self,
        arena: object,
        *,
        pages_by_row: dict[int, tuple[int, ...]],
        recent_first_page_by_row: Iterable[int] | None = None,
    ) -> RrpUpdateResult | None:
        """Leg-B cheap in-place boundary writer.

        Detects which BATCH rows' live page tuples actually changed (dirty),
        publishes ONLY those via ``arena.publish_live_rows`` (in place, graph
        stable), advances the manager's published baseline + visible page counts,
        applies the seqused delta so visible length stays correct, and returns a
        PAGE_BOUNDARY_DELTA hit (``full_bind=False``).

        The live layout key mirrors ``_descriptor_row_table_layout`` exactly: the
        Leg-B fallback rows are row-table fallback (segment_pages == -1,
        affine_key == ("row_table_fallback",)), so a row whose live pages match the
        stored baseline diffs EQUAL and is left clean (zero-write fast path).

        This method must HANDLE the normal recent-slide boundary; it never silently
        returns None to force the heavy rebuild. Any state it cannot apply sets a
        specific ``self.last_failure_reason`` (``"live_page_boundary:<why>"``) and
        returns None so the dev can root-cause it.
        """
        self.last_failure_reason = None
        previous = self._row_effective_k_by_row
        previous_visible = self._visible_page_count_by_row
        previous_published = self._published_descriptor_layout_by_row
        if (
            self._signature is None
            or previous is None
            or previous_visible is None
            or not isinstance(previous_published, tuple)
            or len(previous_published) != len(previous)
            or len(previous_visible) != len(previous)
        ):
            self.last_failure_reason = "live_page_boundary:no_baseline"
            return None

        batch_size = len(previous)
        visible_page_count = list(int(v) for v in previous_visible)
        published_layout = list(previous_published)
        dirty_rows: list[int] = []
        for batch_row in range(batch_size):
            if batch_row not in pages_by_row:
                # No live pages offered for this row -> nothing to diff/write; the
                # baseline row stays as published.
                continue
            stored = published_layout[batch_row]
            if not (isinstance(stored, tuple) and len(stored) == 5):
                self.last_failure_reason = "live_page_boundary:nonfallback_baseline_row"
                return None
            # Mirror _descriptor_row_table_layout: reuse the stored row_mode so equal
            # rows compare equal; Leg-B rows are row-table fallback.
            pages = tuple(int(p) for p in pages_by_row[batch_row])
            live_layout = (
                "descriptor",
                stored[1],
                pages,
                -1,
                ("row_table_fallback",),
            )
            if (
                int(previous_visible[batch_row]) != len(pages)
                or published_layout[batch_row] != live_layout
            ):
                dirty_rows.append(batch_row)
            visible_page_count[batch_row] = len(pages)
            published_layout[batch_row] = live_layout

        publish = getattr(arena, "publish_live_rows", None)
        if not callable(publish):
            self.last_failure_reason = "live_page_boundary:arena_missing_publish_live_rows"
            return None
        publish(pages_by_row=pages_by_row, dirty_rows=tuple(dirty_rows))

        # Advance the published baseline + visible page counts for the dirty rows
        # (mirrors publish_from_descriptor_snapshot) so a repeat with the same pages
        # diffs equal (zero dirty rows, zero writes).
        self._visible_page_count_by_row = tuple(visible_page_count)
        # [VPC-SHADOW-METRIC] live rows carry window-span page counts; refresh
        # the probe's token-ceil shadow at the same write epoch (row_effective
        # is unchanged on this leg — `previous` is the stored value).
        self._probe_visible_page_count_by_row = self._shadow_visible_pages(
            tuple(int(v) for v in previous)
        )
        self._published_descriptor_layout_by_row = tuple(published_layout)

        # leg-b surface 4 recent-first: advance the stored recent-first baseline so
        # the binding-currency page-carriers gate (_is_pure_seqused_delta recent-first
        # equality) diffs EQUAL after a same-count recent slide. The new recent pages
        # were just written into row_table_i32; this makes the stored recent-first
        # reflect that (legitimately-stale bookkeeping, NOT a stamp). Compact stable.
        if recent_first_page_by_row is not None:
            _legb_recent_first = tuple(int(v) for v in recent_first_page_by_row)
            if len(_legb_recent_first) == batch_size:
                self._recent_first_page_by_row = _legb_recent_first

        # Keep visible length correct via the same seqused primitive the publish/
        # delta paths use. row_effective stays the stored value (a recent-window
        # slide that keeps the page COUNT does not change visible length); pass it
        # through so the seqused carrier stays coherent.
        self._apply_seqused_delta(arena, tuple(int(v) for v in previous))

        return RrpUpdateResult(
            hit=False,
            miss_reason="page_boundary_delta",
            delta_rows=tuple(dirty_rows),
            full_bind=False,
            kind=RrpUpdateKind.PAGE_BOUNDARY_DELTA,
        )


    def row_effective_from_descriptor_snapshot(
        self,
        snapshot: object,
    ) -> Tuple[int, ...] | None:
        if not bool(getattr(snapshot, "valid", False)):
            return None
        try:
            snapshot_epoch = int(getattr(snapshot, "epoch"))
        except (AttributeError, TypeError, ValueError):
            return None
        if snapshot_epoch < 0:
            return None
        previous_epoch = self._descriptor_snapshot_epoch
        if previous_epoch is not None and snapshot_epoch < int(previous_epoch):
            return None

        rows_by_graph_row = getattr(snapshot, "rows_by_graph_row", None)
        if rows_by_graph_row is None:
            return None

        previous = self._row_effective_k_by_row
        if previous is None:
            return None
        row_effective_by_graph_row: list[int | None] = []
        try:
            for graph_row, row in enumerate(rows_by_graph_row):
                if row is None:
                    row_effective_by_graph_row.append(None)
                    continue
                if int(getattr(row, "epoch")) != snapshot_epoch:
                    return None
                if int(getattr(row, "visible_epoch")) != snapshot_epoch:
                    return None
                if int(getattr(row, "row_mode_epoch")) != snapshot_epoch:
                    return None
                if int(getattr(row, "graph_row_index")) != int(graph_row):
                    return None
                row_effective = int(getattr(row, "row_effective"))
                if row_effective < 0:
                    return None
                row_effective_by_graph_row.append(row_effective)
        except (AttributeError, TypeError, ValueError):
            return None
        if not any(value is not None for value in row_effective_by_graph_row):
            return None

        if len(row_effective_by_graph_row) == len(previous):
            row_effective_k = [
                int(previous[row]) if value is None else int(value)
                for row, value in enumerate(row_effective_by_graph_row)
            ]
        else:
            row_effective_k = [
                int(value)
                for value in row_effective_by_graph_row
                if value is not None
            ]
        return tuple(row_effective_k)

    @staticmethod
    def _descriptor_row_epoch_current(
        descriptor: object,
        *,
        graph_row: int,
        snapshot_epoch: int,
    ) -> bool:
        try:
            return (
                int(getattr(descriptor, "epoch")) == int(snapshot_epoch)
                and int(getattr(descriptor, "visible_epoch")) == int(snapshot_epoch)
                and int(getattr(descriptor, "row_mode_epoch")) == int(snapshot_epoch)
                and int(getattr(descriptor, "graph_row_index")) == int(graph_row)
            )
        except (AttributeError, TypeError, ValueError):
            return False

    @staticmethod
    def _descriptor_row_table_layout(descriptor: object) -> tuple[object, ...] | None:
        try:
            row_mode = str(getattr(descriptor, "row_mode"))
            pages = tuple(int(v) for v in getattr(descriptor, "row_table_pages"))
            segment_pages = int(getattr(descriptor, "segment_pages"))
            affine_descriptor = getattr(descriptor, "affine_descriptor")
        except (AttributeError, TypeError, ValueError):
            return None
        try:
            affine_key: tuple[object, ...]
            if affine_descriptor == "row_table_fallback":
                affine_key = ("row_table_fallback",)
            else:
                affine_key = tuple(int(v) for v in affine_descriptor)
        except (TypeError, ValueError):
            return None
        return (
            "descriptor",
            row_mode,
            pages,
            int(segment_pages),
            affine_key,
        )

    def _shadow_visible_pages(
        self,
        row_effective_k: Tuple[int, ...],
    ) -> Tuple[int, ...] | None:
        """[VPC-SHADOW-METRIC] token-ceil page count for the given effk under
        the CURRENT stored compact fields — the probe's own metric, evaluated
        at write time. None when the compact baseline is incomplete (probe
        then takes its structural miss -> full bind, the correct slow path).
        """
        if (
            self._compact_ready_by_row is None
            or self._compact_valid_tokens_by_row is None
            or self._page_size is None
            or self._max_pages_per_row is None
            or len(self._compact_ready_by_row) != len(row_effective_k)
            or len(self._compact_valid_tokens_by_row) != len(row_effective_k)
        ):
            return None
        return self._visible_page_count_from_fields(
            batch_size=len(row_effective_k),
            row_effective_k=tuple(int(v) for v in row_effective_k),
            compact_ready_by_row=self._compact_ready_by_row,
            compact_valid_tokens_by_row=self._compact_valid_tokens_by_row,
            page_size=self._page_size,
            max_pages_per_row=self._max_pages_per_row,
        )

    def _store(
        self,
        signature: tuple[object, ...],
        row_effective_k: Tuple[int, ...],
        recent_first_page: Tuple[int, ...],
        visible_page_count: Tuple[int, ...],
        row_table_layout: Tuple[tuple[object, ...], ...],
    ) -> None:
        self._signature = signature
        self._row_effective_k_by_row = row_effective_k
        self._recent_first_page_by_row = recent_first_page
        self._visible_page_count_by_row = visible_page_count
        # [VPC-SHADOW-METRIC] recompute (not copy) so the shadow is the
        # probe's metric regardless of which metric the caller stored.
        self._probe_visible_page_count_by_row = self._shadow_visible_pages(
            row_effective_k
        )
        self._row_table_layout_by_row = row_table_layout

    @staticmethod
    def _seqused_update_kernel_count(
        arena: object,
        batch_target: torch.Tensor | None = None,
    ) -> int:
        if batch_target is None:
            batch_target = RrpRowTableManager._arena_batch_seqused_write_target(arena)
        if isinstance(batch_target, torch.Tensor):
            return 1 if batch_target.device.type == "cuda" else 0
        if RrpRowTableManager._uses_external_batch_seqused_source(arena):
            return 0
        seqused_k_i32 = getattr(arena, "seqused_k_i32")
        return 1 if seqused_k_i32.device.type == "cuda" else 0

    @staticmethod
    def _arena_batch_seqused_write_target(arena: object) -> torch.Tensor | None:
        if str(getattr(arena, "_resolved_seqused_source_kind", "")) != "arena_batch_seqused":
            return None
        carriers = getattr(arena, "carriers", None)
        visible = getattr(carriers, "resolver_visible_seqused_k_by_head_i32", None)
        batch_seqused = getattr(arena, "batch_seqused_k_i32", None)
        if not isinstance(visible, torch.Tensor) or not isinstance(
            batch_seqused,
            torch.Tensor,
        ):
            return None
        if int(visible.data_ptr()) != int(batch_seqused.data_ptr()):
            return None
        if int(visible.numel()) != int(getattr(arena, "batch_size")):
            return None
        return batch_seqused

    @staticmethod
    def _uses_external_batch_seqused_source(arena: object) -> bool:
        if str(getattr(arena, "_resolved_seqused_source_kind", "")) not in {
            "dense_seqused",
            "launch_effective",
            "sparse_dynamic_state",
        }:
            return False
        carriers = getattr(arena, "carriers", None)
        visible = getattr(carriers, "resolver_visible_seqused_k_by_head_i32", None)
        seqused_k_i32 = getattr(arena, "seqused_k_i32", None)
        if not isinstance(visible, torch.Tensor) or not isinstance(
            seqused_k_i32,
            torch.Tensor,
        ):
            return False
        if int(visible.data_ptr()) == int(seqused_k_i32.data_ptr()):
            return False
        return int(visible.numel()) == int(getattr(arena, "batch_size"))

    @staticmethod
    def _visible_page_count_for(
        inputs: RrpRowTableInputs,
        row_effective_k: Tuple[int, ...],
    ) -> Tuple[int, ...]:
        return RrpRowTableManager._visible_page_count_from_fields(
            batch_size=inputs.batch_size,
            row_effective_k=row_effective_k,
            compact_ready_by_row=inputs.compact_ready_by_row,
            compact_valid_tokens_by_row=inputs.compact_valid_tokens_by_row,
            page_size=inputs.page_size,
            max_pages_per_row=inputs.max_pages_per_row,
        )

    @staticmethod
    def _visible_page_count_from_fields(
        *,
        batch_size: int,
        row_effective_k: Tuple[int, ...],
        compact_ready_by_row: Tuple[bool, ...],
        compact_valid_tokens_by_row: Tuple[int, ...],
        page_size: int,
        max_pages_per_row: int,
    ) -> Tuple[int, ...]:
        page_size = max(1, int(page_size))
        max_pages = max(0, int(max_pages_per_row))
        counts: list[int] = []
        for row in range(int(batch_size)):
            effective_k = max(0, int(row_effective_k[row]))
            if bool(compact_ready_by_row[row]):
                compact_tokens = max(0, int(compact_valid_tokens_by_row[row]))
                compact_pages = (compact_tokens + page_size - 1) // page_size
                recent_tokens = max(0, effective_k - compact_tokens)
                recent_pages = (recent_tokens + page_size - 1) // page_size
                visible_pages = compact_pages + recent_pages
            else:
                visible_pages = (effective_k + page_size - 1) // page_size
            counts.append(min(visible_pages, max_pages))
        return tuple(counts)

    @staticmethod
    def _signature_for(inputs: RrpRowTableInputs) -> tuple[object, ...]:
        batch = inputs.batch_size
        return (
            batch,
            tuple(inputs.req_ids_by_row[:batch]),
            int(inputs.compact_capacity_pages),
            int(inputs.max_pages_per_row),
            int(inputs.page_size),
        )

    @staticmethod
    def _row_table_layout_for(
        inputs: RrpRowTableInputs,
        visible_page_count: Tuple[int, ...],
    ) -> Tuple[tuple[object, ...], ...]:
        batch = int(inputs.batch_size)
        reserved_signature = inputs.reserved_manager_block_ids
        rows: list[tuple[object, ...]] = []
        for row in range(batch):
            is_compact = bool(inputs.compact_ready_by_row[row])
            if is_compact:
                rows.append(
                    (
                        int(inputs.slot_by_row[row]),
                        1,
                        int(inputs.compact_valid_tokens_by_row[row]),
                        int(inputs.compact_offset_tokens_by_row[row]),
                        int(inputs.recent_first_page_by_row[row]),
                        int(visible_page_count[row]),
                        reserved_signature,
                    )
                )
            else:
                rows.append((0, 0, 0, 0, 0, int(visible_page_count[row]), ()))
        return tuple(rows)

    def _try_increment_seqused_k(
        self,
        arena: object,
        row_effective_k: Tuple[int, ...],
        batch_target: torch.Tensor | None = None,
    ) -> bool:
        previous = self._row_effective_k_by_row
        if previous is None or len(previous) != len(row_effective_k):
            return False
        if batch_target is None:
            batch_target = self._arena_batch_seqused_write_target(arena)
        if isinstance(batch_target, torch.Tensor):
            deltas = tuple(
                int(row_effective_k[row]) - int(previous[row])
                for row in range(len(row_effective_k))
            )
            if not deltas:
                return False
            changed_rows = tuple(
                index for index, delta in enumerate(deltas) if int(delta) != 0
            )
            if not changed_rows:
                return False
            first_delta = int(deltas[changed_rows[0]])
            if first_delta <= 0 or any(
                int(deltas[index]) != first_delta for index in changed_rows
            ):
                return False
            start = int(changed_rows[0])
            stop = int(changed_rows[-1]) + 1
            if changed_rows != tuple(range(start, stop)):
                return False
            delta = int(first_delta)
            target = batch_target[start:stop]
            if int(target.numel()) <= 0:
                return False
            target.add_(delta)
            self._seqused_fast_increment_count = int(
                getattr(self, "_seqused_fast_increment_count", 0)
            ) + 1
            return True
        if self._uses_external_batch_seqused_source(arena):
            self._seqused_fast_increment_count = int(
                getattr(self, "_seqused_fast_increment_count", 0)
            ) + 1
            return True
        deltas = tuple(
            int(row_effective_k[row]) - int(previous[row])
            for row in range(len(row_effective_k))
        )
        if not deltas or any(delta != deltas[0] for delta in deltas):
            return False
        delta = int(deltas[0])
        if delta <= 0:
            return False
        seqused_k_i32 = getattr(arena, "seqused_k_i32")
        seqused_k_i32.add_(delta)
        self._seqused_fast_increment_count = int(
            getattr(self, "_seqused_fast_increment_count", 0)
        ) + 1
        return True

    def _write_seqused_k(
        self,
        arena: object,
        row_effective_k: Iterable[int],
        *,
        row_effective_k_i32_gpu: torch.Tensor | None = None,
    ) -> None:
        batch_target = self._arena_batch_seqused_write_target(arena)
        if isinstance(batch_target, torch.Tensor):
            row_values = tuple(int(v) for v in row_effective_k)
            batch_size = len(row_values)
            if batch_size <= 0:
                return
            if (
                batch_target.device.type == "cuda"
                and isinstance(row_effective_k_i32_gpu, torch.Tensor)
                and row_effective_k_i32_gpu.device == batch_target.device
                and row_effective_k_i32_gpu.dtype == torch.int32
                and row_effective_k_i32_gpu.dim() == 1
                and int(row_effective_k_i32_gpu.numel()) >= batch_size
            ):
                batch_target[:batch_size].copy_(
                    row_effective_k_i32_gpu[:batch_size],
                    non_blocking=True,
                )
                return
            if batch_target.device.type == "cuda":
                # [S1-KC-PIN-STAGING 2026-07-12] 原 torch.tensor(tuple,
                # device=cuda)=每次一个 pageable H2D memcpy_and_sync(cudaMemcpy
                # Async+cudaStreamSynchronize),等当前流全排空=host 领先被排干
                # (S1 探针定谳:372 次/轮,单个 5-14ms 级,mb.misc 洞前沿主料)。
                # 改 SEQUSED-STAGING-INDEPENDENT 同款(本函数下方 seqused_k_i32
                # 分支先例):每次独立 pinned 小分配(bs×4B,CachingHostAllocator
                # 缓存命中 µs 级)+non_blocking H2D。WAR 护栏=allocator 事件跟踪
                # (pinned 块 free 后须等 copy_ 记录的 stream 事件完成才复用,
                # 07-02 pinned staging 铁律满足;零共享单例=零 WAR 窗)。
                # batch_target[:bs] 连续,pinned→GPU 单 memcpy;值逐位同源。
                try:
                    staging_cpu = torch.empty(
                        (batch_size,),
                        dtype=torch.int32,
                        device="cpu",
                        pin_memory=True,
                    )
                except RuntimeError:
                    staging_cpu = torch.empty(
                        (batch_size,),
                        dtype=torch.int32,
                        device="cpu",
                    )
                staging_cpu.copy_(
                    torch.as_tensor(
                        row_values,
                        dtype=torch.int32,
                        device="cpu",
                    )
                )
                batch_target[:batch_size].copy_(staging_cpu, non_blocking=True)
                return
            values = torch.tensor(
                row_values,
                dtype=torch.int32,
                device=batch_target.device,
            )
            batch_target[:batch_size].copy_(values, non_blocking=True)
            return
        if self._uses_external_batch_seqused_source(arena):
            return
        seqused_k_i32 = getattr(arena, "seqused_k_i32")
        num_kv_heads = int(getattr(arena, "num_kv_heads"))
        row_values = tuple(int(v) for v in row_effective_k)
        batch_size = len(row_values)
        if batch_size <= 0:
            return
        target = seqused_k_i32.view(batch_size, num_kv_heads)
        if (
            seqused_k_i32.device.type == "cuda"
            and isinstance(row_effective_k_i32_gpu, torch.Tensor)
            and row_effective_k_i32_gpu.device == seqused_k_i32.device
            and row_effective_k_i32_gpu.dtype == torch.int32
            and row_effective_k_i32_gpu.dim() == 1
            and int(row_effective_k_i32_gpu.numel()) >= batch_size
        ):
            target.copy_(
                row_effective_k_i32_gpu[:batch_size].reshape(
                    batch_size,
                    1,
                ).expand(batch_size, num_kv_heads),
                non_blocking=True,
            )
            return
        if seqused_k_i32.device.type == "cuda":
            # [SEQUSED-STAGING-INDEPENDENT 2026-07-07] 原 manager 级单例
            # pinned/GPU staging 对与 seq_lens/slot 共享 staging 同族第六处:
            # pinned 单例每步被 host copy_ 覆写,而上一步的 non_blocking H2D
            # 可能未决(WAR 撕裂,批2 pinned staging 根修先例同型);GPU 侧
            # 容量换代直接替换引用零 record_stream(UAF 臂)。修同款:每次
            # 独立小分配(bs×4B,allocator 缓存命中 µs 级),局部引用持有到
            # copy 入队,零共享零 WAR 窗;值语义不变。
            try:
                staging_cpu = torch.empty(
                    (batch_size,),
                    dtype=torch.int32,
                    device="cpu",
                    pin_memory=True,
                )
            except RuntimeError:
                staging_cpu = torch.empty(
                    (batch_size,),
                    dtype=torch.int32,
                    device="cpu",
                )
            staging_cpu.copy_(
                torch.as_tensor(
                    row_values[:batch_size],
                    dtype=torch.int32,
                    device="cpu",
                )
            )
            staging_gpu = torch.empty(
                (batch_size,),
                dtype=torch.int32,
                device=seqused_k_i32.device,
            )
            staging_gpu.copy_(
                staging_cpu,
                non_blocking=True,
            )
            target.copy_(
                staging_gpu.reshape(batch_size, 1).expand(
                    batch_size,
                    num_kv_heads,
                ),
                non_blocking=True,
            )
            return

        values = torch.tensor(
            row_values,
            dtype=torch.int32,
            device=seqused_k_i32.device,
        )
        target.copy_(
            values.reshape(batch_size, 1).expand(batch_size, num_kv_heads)
        )
