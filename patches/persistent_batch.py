"""
patches/persistent_batch.py — Persistent batch state for incremental step updates.

Instead of rebuilding every list/tuple from scratch each decode step,
PersistentStepBatch maintains pre-allocated NumPy arrays that are updated
incrementally via diff (added/removed requests).

OWNS:
  - PersistentStepBatch: the persistent batch container
  - apply_diff(): incremental update logic

DEPENDS_ON:
  - stdlib (dataclasses, typing)
  - numpy

ENTRY_POINTS:
  - Used by step_context_worker.py when VLLM_SPARSE_PERSISTENT_BATCH=1
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np


@dataclass(slots=True)
class PersistentStepBatch:
    """Persistent batch state that survives across decode steps.

    Pre-allocates fixed-size NumPy arrays up to ``max_reqs`` and maintains
    a compact, contiguous view of the first ``num_reqs`` rows.  Each step,
    ``apply_diff`` computes the minimal set of changes (added/removed requests)
    and updates only those rows.
    """

    max_reqs: int

    # --- pre-allocated arrays (shape = max_reqs) ---
    req_ids: np.ndarray            # dtype=object
    seq_lens: np.ndarray           # dtype=int64
    q_lens: np.ndarray             # dtype=int64
    slot_by_row: np.ndarray        # dtype=int32
    row_mode: np.ndarray           # dtype=int32
    bootstrap_done: np.ndarray     # dtype=bool
    is_prefill: np.ndarray         # dtype=bool
    pending_refresh: np.ndarray    # dtype=bool
    decode_step: np.ndarray        # dtype=int64
    short_dense: np.ndarray        # dtype=bool

    # --- scalar state ---
    num_reqs: int = 0
    has_prefill_row: bool = False
    has_decode_row: bool = False
    epoch: int = 0

    # --- per-step diff tracking ---
    added_indices: List[int] = field(default_factory=list)
    removed_req_ids: Set[str] = field(default_factory=set)
    changed_mask: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))

    # --- internal lookup ---
    _rid_to_row: Dict[str, int] = field(default_factory=dict)

    # --- export caches (invalidated on change, reused in steady-state) ---
    _cached_is_prefill_tuple: Optional[Tuple[bool, ...]] = None
    _cached_bootstrap_done_tuple: Optional[Tuple[bool, ...]] = None
    _cached_prefill_rows_tuple: Optional[Tuple[int, ...]] = None
    _cached_short_dense_tuple: Optional[Tuple[bool, ...]] = None

    # ------------------------------------------------------------------ #
    #  Factory
    # ------------------------------------------------------------------ #

    @staticmethod
    def create(max_reqs: int) -> "PersistentStepBatch":
        return PersistentStepBatch(
            max_reqs=max_reqs,
            req_ids=np.empty(max_reqs, dtype=object),
            seq_lens=np.zeros(max_reqs, dtype=np.int64),
            q_lens=np.zeros(max_reqs, dtype=np.int64),
            slot_by_row=np.full(max_reqs, -1, dtype=np.int32),
            row_mode=np.zeros(max_reqs, dtype=np.int32),
            bootstrap_done=np.zeros(max_reqs, dtype=bool),
            is_prefill=np.zeros(max_reqs, dtype=bool),
            pending_refresh=np.zeros(max_reqs, dtype=bool),
            decode_step=np.full(max_reqs, -1, dtype=np.int64),
            short_dense=np.zeros(max_reqs, dtype=bool),
            changed_mask=np.zeros(max_reqs, dtype=bool),
        )

    # ------------------------------------------------------------------ #
    #  Incremental diff
    # ------------------------------------------------------------------ #

    def apply_diff(
        self,
        new_req_ids: Sequence[str],
        new_seq_lens: Sequence[int],
        new_q_lens: Sequence[int],
    ) -> None:
        """Compute diff between current batch and new batch, then update arrays.

        After this call:
          - ``added_indices`` contains row indices of newly added requests
          - ``removed_req_ids`` contains IDs of removed requests
          - ``changed_mask[:num_reqs]`` is True for rows that changed
          - ``num_reqs`` reflects the new batch size
          - Arrays are compacted (no holes)
        """
        new_set = set(new_req_ids)
        old_set = set(self._rid_to_row.keys())

        removed = old_set - new_set
        added_set = new_set - old_set
        self.removed_req_ids = removed
        self.added_indices = []

        # Invalidate export caches when composition changes
        if removed or added_set:
            self._cached_is_prefill_tuple = None
            self._cached_bootstrap_done_tuple = None
            self._cached_prefill_rows_tuple = None
            self._cached_short_dense_tuple = None

        n = self.num_reqs

        # --- Step 1: swap-remove exiting requests ---
        if removed:
            # Collect indices to remove (sorted descending for stable swap)
            remove_rows = sorted(
                (self._rid_to_row[rid] for rid in removed if rid in self._rid_to_row),
                reverse=True,
            )
            for row in remove_rows:
                if row >= n:
                    continue
                last = n - 1
                if row != last:
                    # Swap with last occupied row
                    self._swap_rows(row, last)
                    moved_rid = self.req_ids[row]
                    if moved_rid is not None:
                        self._rid_to_row[moved_rid] = row
                n -= 1
            # Clean up rid_to_row for removed
            for rid in removed:
                self._rid_to_row.pop(rid, None)
            self.num_reqs = n

        # --- Step 2: build new_req_id → desired position mapping ---
        new_rid_to_pos = {rid: i for i, rid in enumerate(new_req_ids)}

        # --- Step 3: reorder existing rows to match new order ---
        # Build current mapping
        new_n = len(new_req_ids)
        if new_n > self.max_reqs:
            self._grow(new_n * 2)

        # Place existing requests into their correct positions
        # First pass: figure out where each existing request should go
        existing_placement: Dict[str, int] = {}
        for rid in new_req_ids:
            if rid in self._rid_to_row:
                existing_placement[rid] = new_rid_to_pos[rid]

        # Check if reorder is needed (positions mismatch)
        needs_reorder = False
        for rid, target_pos in existing_placement.items():
            if self._rid_to_row[rid] != target_pos:
                needs_reorder = True
                break

        if needs_reorder:
            # Rebuild from scratch into a temporary buffer approach:
            # Copy existing data to temp, then place back
            old_n = self.num_reqs
            tmp_req_ids = self.req_ids[:old_n].copy()
            tmp_seq_lens = self.seq_lens[:old_n].copy()
            tmp_q_lens = self.q_lens[:old_n].copy()
            tmp_slot = self.slot_by_row[:old_n].copy()
            tmp_row_mode = self.row_mode[:old_n].copy()
            tmp_boot = self.bootstrap_done[:old_n].copy()
            tmp_prefill = self.is_prefill[:old_n].copy()
            tmp_pending = self.pending_refresh[:old_n].copy()
            tmp_decode = self.decode_step[:old_n].copy()
            tmp_short = self.short_dense[:old_n].copy()

            old_rid_to_old_row = dict(self._rid_to_row)
            self._rid_to_row.clear()

            for rid, target in existing_placement.items():
                old_row = old_rid_to_old_row[rid]
                self.req_ids[target] = tmp_req_ids[old_row]
                self.seq_lens[target] = tmp_seq_lens[old_row]
                self.q_lens[target] = tmp_q_lens[old_row]
                self.slot_by_row[target] = tmp_slot[old_row]
                self.row_mode[target] = tmp_row_mode[old_row]
                self.bootstrap_done[target] = tmp_boot[old_row]
                self.is_prefill[target] = tmp_prefill[old_row]
                self.pending_refresh[target] = tmp_pending[old_row]
                self.decode_step[target] = tmp_decode[old_row]
                self.short_dense[target] = tmp_short[old_row]
                self._rid_to_row[rid] = target
        # else: positions already match, no reorder needed

        # --- Step 4: append new requests at their target positions ---
        for rid in new_req_ids:
            if rid in added_set:
                pos = new_rid_to_pos[rid]
                self.req_ids[pos] = rid
                self.seq_lens[pos] = 0
                self.q_lens[pos] = 0
                self.slot_by_row[pos] = -1
                self.row_mode[pos] = 0
                self.bootstrap_done[pos] = False
                self.is_prefill[pos] = True
                self.pending_refresh[pos] = False
                self.decode_step[pos] = -1
                self.short_dense[pos] = False
                self._rid_to_row[rid] = pos
                self.added_indices.append(pos)

        self.num_reqs = new_n

        # --- Step 5: update per-step changing fields ---
        n = self.num_reqs
        for i in range(n):
            self.seq_lens[i] = new_seq_lens[i]
            self.q_lens[i] = new_q_lens[i]

        # --- Step 6: build changed_mask ---
        if len(self.changed_mask) < self.max_reqs:
            self.changed_mask = np.zeros(self.max_reqs, dtype=bool)
        self.changed_mask[:n] = False
        for idx in self.added_indices:
            if idx < n:
                self.changed_mask[idx] = True
        # Also mark rows whose seq_lens changed significantly
        # (for now, mark all rows as "may need derived update" only for added)

        # --- Step 7: update scalar state ---
        self.has_prefill_row = bool(np.any(self.is_prefill[:n]))
        self.has_decode_row = bool(np.any(~self.is_prefill[:n]))
        self.epoch += 1

    # ------------------------------------------------------------------ #
    #  Derived array sync (called by step_context_worker after loops)
    # ------------------------------------------------------------------ #

    def sync_derived_arrays(
        self,
        is_prefill_list: Sequence[bool],
        bootstrap_done_list: Sequence[bool],
        has_prefill_row: bool,
        has_decode_row: bool,
    ) -> None:
        """Sync derived state computed by tracking loops into arrays.

        Compares with existing values and only invalidates export caches
        when data actually changed.  In steady-state decode this is a no-op
        on the cache, giving O(1) subsequent exports.
        """
        n = self.num_reqs
        is_prefill_changed = False
        bootstrap_changed = False
        for i in range(n):
            new_pf = is_prefill_list[i]
            if bool(self.is_prefill[i]) != new_pf:
                self.is_prefill[i] = new_pf
                is_prefill_changed = True
            new_bd = bootstrap_done_list[i]
            if bool(self.bootstrap_done[i]) != new_bd:
                self.bootstrap_done[i] = new_bd
                bootstrap_changed = True
        if is_prefill_changed:
            self._cached_is_prefill_tuple = None
            self._cached_prefill_rows_tuple = None
        if bootstrap_changed:
            self._cached_bootstrap_done_tuple = None
        self.has_prefill_row = has_prefill_row
        self.has_decode_row = has_decode_row

    def sync_short_dense(
        self,
        short_dense_list: Sequence[bool],
    ) -> None:
        """Sync short_dense flags; invalidates cache only on change."""
        n = self.num_reqs
        changed = False
        for i in range(n):
            new_val = short_dense_list[i]
            if bool(self.short_dense[i]) != new_val:
                self.short_dense[i] = new_val
                changed = True
        if changed:
            self._cached_short_dense_tuple = None

    # ------------------------------------------------------------------ #
    #  Export to tuples (for StepContext / StepMeta construction)
    # ------------------------------------------------------------------ #




    def export_bootstrap_done(self) -> Tuple[bool, ...]:
        c = self._cached_bootstrap_done_tuple
        if c is not None and len(c) == self.num_reqs:
            return c
        n = self.num_reqs
        result = tuple(bool(x) for x in self.bootstrap_done[:n])
        self._cached_bootstrap_done_tuple = result
        return result

    def export_is_prefill(self) -> Tuple[bool, ...]:
        c = self._cached_is_prefill_tuple
        if c is not None and len(c) == self.num_reqs:
            return c
        n = self.num_reqs
        result = tuple(bool(x) for x in self.is_prefill[:n])
        self._cached_is_prefill_tuple = result
        return result

    def export_short_dense(self) -> Tuple[bool, ...]:
        c = self._cached_short_dense_tuple
        if c is not None and len(c) == self.num_reqs:
            return c
        n = self.num_reqs
        result = tuple(bool(x) for x in self.short_dense[:n])
        self._cached_short_dense_tuple = result
        return result

    def export_slot_by_row(self) -> Tuple[int, ...]:
        n = self.num_reqs
        return tuple(int(x) for x in self.slot_by_row[:n])

    def export_row_mode(self) -> Tuple[int, ...]:
        n = self.num_reqs
        return tuple(int(x) for x in self.row_mode[:n])

    def export_req_id_to_index(self) -> Dict[str, int]:
        return dict(self._rid_to_row)

    def get_prefill_rows(self) -> Tuple[int, ...]:
        c = self._cached_prefill_rows_tuple
        if c is not None:
            # Validate: cached length consistency
            if self.num_reqs == 0 or not self.has_prefill_row:
                if len(c) == 0:
                    return c
            else:
                # Cache was built when is_prefill was last synced
                return c
        n = self.num_reqs
        result = tuple(i for i in range(n) if self.is_prefill[i])
        self._cached_prefill_rows_tuple = result
        return result



    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    def _swap_rows(self, a: int, b: int) -> None:
        """Swap data at row a and row b across all arrays."""
        for arr in (
            self.req_ids, self.seq_lens, self.q_lens, self.slot_by_row,
            self.row_mode, self.bootstrap_done, self.is_prefill,
            self.pending_refresh, self.decode_step, self.short_dense,
        ):
            arr[a], arr[b] = arr[b], arr[a]

    def _grow(self, new_max: int) -> None:
        """Grow all arrays to new_max capacity."""
        old_max = self.max_reqs
        if new_max <= old_max:
            return
        self.max_reqs = new_max

        def _resize(arr: np.ndarray, fill: object = None) -> np.ndarray:
            new = np.empty(new_max, dtype=arr.dtype)
            new[:old_max] = arr
            if fill is not None and arr.dtype != object:
                new[old_max:] = fill
            return new

        self.req_ids = _resize(self.req_ids)
        self.seq_lens = _resize(self.seq_lens, 0)
        self.q_lens = _resize(self.q_lens, 0)
        self.slot_by_row = _resize(self.slot_by_row, -1)
        self.row_mode = _resize(self.row_mode, 0)
        self.bootstrap_done = _resize(self.bootstrap_done, False)
        self.is_prefill = _resize(self.is_prefill, False)
        self.pending_refresh = _resize(self.pending_refresh, False)
        self.decode_step = _resize(self.decode_step, -1)
        self.short_dense = _resize(self.short_dense, False)
        self.changed_mask = np.zeros(new_max, dtype=bool)

    def reset(self) -> None:
        """Clear all state, returning to empty batch."""
        self.num_reqs = 0
        self.has_prefill_row = False
        self.has_decode_row = False
        self.epoch = 0
        self.added_indices.clear()
        self.removed_req_ids.clear()
        self._rid_to_row.clear()
        self.changed_mask[:] = False
        self._cached_is_prefill_tuple = None
        self._cached_bootstrap_done_tuple = None
        self._cached_prefill_rows_tuple = None
        self._cached_short_dense_tuple = None
