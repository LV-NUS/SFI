"""[JUDGE-REPLAY-AWARE 2026-07-09] Causal poison probe for compact residency.

Purpose
-------
Prove (or disprove) that decode attention in the FULL-graph *serve* form
physically reads the compact residency arena bytes. Python route counters only
observe capture/eager steps; under FULL cudagraph replay the router never runs.
This probe writes finite garbage into the *physical* KV pool bytes that the
compact arena is a view of (``page_kv_residency`` narrows the pool at the
reserved block range), on the default CUDA stream, *before* the model forward
is submitted. Same-stream ordering means a replay that reads those addresses
reads the poisoned bytes -> the decode output visibly changes. If the output is
unaffected, decode did not read the poisoned bytes.

Two modes
---------
* ``compact`` -- poison the target compact row's reserved-block residency
  (across all compact generations for that slot). A dense/compact row that
  reads its compact arena will corrupt.
* ``native``  -- control. Poison ~8 mid-context canonical (full-KV) pages of the
  same row that are NOT compact-reserved, NOT in the recent window, and not the
  sink page. A sparse (compact) row should skip these; a dense row would not.

Env protocol
------------
``VLLM_SPARSE_DEBUG_COMPACT_POISON="<mode>:<fire_step>[:<log_path>]"``

* ``mode``      -- ``compact`` | ``native``
* ``fire_step`` -- 1-based count of probe invocations. The probe maintains its
  own per-process counter incremented on every call (i.e. every compact-row
  decode step, since the call site is gated on ``_has_compact_row``); it fires
  exactly once when the counter reaches ``fire_step``.
* ``log_path``  -- optional file to append the one-line JSON evidence record to
  (default: stderr).

Cost model
----------
When the env var is unset the module is never imported (both call sites gate on
a module-level boolean read once at their own import). If it *is* imported, the
spec is parsed exactly once here; ``maybe_register_layer_kv`` /
``maybe_fire_compact_poison_probe`` return after a single boolean check until
the fire step. This is a diagnostic: on any inconsistency it RAISES (fail-fast,
never silently skips).
"""

from __future__ import annotations

import json
import math
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple


_POISON_ENV = "VLLM_SPARSE_DEBUG_COMPACT_POISON"
_POISON_SPEC = os.environ.get(_POISON_ENV)

# Finite garbage; never NaN/inf. Representable in bf16/fp16/fp8 (e4m3 max ~448).
_POISON_VALUE = 30.0


def _parse_spec(spec: str) -> Tuple[str, int, Optional[str]]:
    """Parse ``<mode>:<fire_step>[:<log_path>]``; raise on any malformation."""
    parts = spec.split(":")
    if len(parts) < 2:
        raise RuntimeError(
            f"{_POISON_ENV} must be '<mode>:<fire_step>[:<log_path>]', got {spec!r}"
        )
    mode = parts[0].strip().lower()
    if mode not in ("compact", "native"):
        raise RuntimeError(
            f"{_POISON_ENV} mode must be 'compact' or 'native', got {mode!r}"
        )
    try:
        fire_step = int(parts[1].strip())
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"{_POISON_ENV} fire_step must be an int, got {parts[1]!r}"
        ) from exc
    if fire_step < 1:
        raise RuntimeError(f"{_POISON_ENV} fire_step must be >= 1, got {fire_step}")
    log_path: Optional[str] = None
    if len(parts) >= 3:
        # Re-join the remainder so an (unusual) path with a colon survives.
        tail = ":".join(parts[2:]).strip()
        log_path = tail or None
    return mode, fire_step, log_path


if _POISON_SPEC is not None:
    _PROBE_MODE, _PROBE_FIRE_STEP, _PROBE_LOG_PATH = _parse_spec(_POISON_SPEC)
    _PROBE_ARMED = True
else:
    _PROBE_MODE, _PROBE_FIRE_STEP, _PROBE_LOG_PATH = None, 0, None
    _PROBE_ARMED = False

# Public alias so a caller can gate on the parsed result if it prefers.
PROBE_ARMED = _PROBE_ARMED

# Per-process mutable state.
_STEP_COUNTER = 0
_FIRED = False
# layer_index -> (key_cache, value_cache). Pool tensors are process-stable, so
# direct refs are safe; first-seen registration is enough (idempotent overwrite).
_LAYER_KV_REGISTRY: Dict[int, Tuple[object, object]] = {}


def maybe_register_layer_kv(
    layer_index: int,
    key_cache: object,
    value_cache: object,
) -> None:
    """Register a layer's pool KV tensors (env-gated; unset -> no-op).

    Called from the selector rebuild loop where ``payload.key_cache`` /
    ``payload.value_cache`` are the vLLM paged-KV pool tensors of shape
    ``[num_blocks, block_size, num_kv_heads, head_dim]``. The compact residency
    is a ``narrow``+``view`` of exactly these tensors at the reserved blocks, so
    poisoning ``key_cache[block_id]`` poisons the physical residency bytes.
    """
    if not _PROBE_ARMED:
        return
    if key_cache is None or value_cache is None:
        return
    _LAYER_KV_REGISTRY[int(layer_index)] = (key_cache, value_cache)


def maybe_fire_compact_poison_probe(
    controller: object,
    use_compact: Sequence[bool],
    slot_by_row: Sequence[int],
    seq_lens_by_row: Sequence[int],
) -> None:
    """Advance the step counter; fire exactly once at ``fire_step``.

    All arguments are host-deterministic step values taken straight from the
    R3c bump point in ``prepare_step_context_impl``: ``use_compact`` and
    ``slot_by_row`` are the exact tuples that feed ``StepAuthority`` this step,
    and ``seq_lens_by_row`` is the per-row context KV length.
    """
    global _STEP_COUNTER, _FIRED
    if not _PROBE_ARMED or _FIRED:
        return
    _STEP_COUNTER += 1
    if _STEP_COUNTER < _PROBE_FIRE_STEP:
        return
    # Latch before doing work: fire exactly once even if the body raises.
    _FIRED = True
    _fire(controller, use_compact, slot_by_row, seq_lens_by_row)


def _fire(
    controller: object,
    use_compact: Sequence[bool],
    slot_by_row: Sequence[int],
    seq_lens_by_row: Sequence[int],
) -> None:
    from patches.sparse_constants import compact_gen_count

    # 1. Target row = first compact row this step.
    target_row: Optional[int] = None
    for i, use in enumerate(use_compact):
        if use:
            target_row = i
            break
    if target_row is None:
        raise RuntimeError(
            "compact_poison_probe: no _use_compact row is True at fire step "
            f"{_PROBE_FIRE_STEP} (call site should be gated on _has_compact_row)"
        )
    if target_row >= len(slot_by_row):
        raise RuntimeError(
            "compact_poison_probe: target row "
            f"{target_row} outside slot_by_row (len={len(slot_by_row)})"
        )
    slot = int(slot_by_row[target_row])
    if slot < 0:
        raise RuntimeError(
            f"compact_poison_probe: invalid slot {slot} for compact row {target_row}"
        )

    # 2. Layer KV registry (populated by the selector rebuild hook).
    if not _LAYER_KV_REGISTRY:
        raise RuntimeError(
            "compact_poison_probe: no layer KV registered; a refresh/rebuild must "
            "run before poison can fire (selection_worker register hook)"
        )
    layers: List[Tuple[int, Tuple[object, object]]] = sorted(
        _LAYER_KV_REGISTRY.items()
    )
    n_layers = len(layers)
    sample_key_cache = layers[0][1][0]
    if sample_key_cache.dim() < 2:
        raise RuntimeError(
            "compact_poison_probe: registered key_cache rank < 2; expected "
            "[num_blocks, block_size, num_kv_heads, head_dim]"
        )
    num_blocks = int(sample_key_cache.shape[0])
    block_size = int(sample_key_cache.shape[1])
    if block_size <= 0:
        raise RuntimeError("compact_poison_probe: non-positive KV block_size")

    # 3. Compact page lease (reserved pool + per-slot slice).
    lease = _lease_from_controller(controller)
    if lease is None:
        raise RuntimeError(
            "compact_poison_probe: no compact page lease reachable from controller "
            "(layer_states[*].compact_page_residency.lease)"
        )

    if _PROBE_MODE == "compact":
        block_ids = _compact_slot_block_ids(lease, slot, int(compact_gen_count()))
    else:
        block_ids = _native_control_block_ids(
            controller=controller,
            row=target_row,
            seq_lens_by_row=seq_lens_by_row,
            block_size=block_size,
            lease=lease,
        )
    if not block_ids:
        raise RuntimeError(
            "compact_poison_probe: empty poison block set "
            f"(mode={_PROBE_MODE}, row={target_row}, slot={slot})"
        )
    for block_id in block_ids:
        if block_id < 0 or block_id >= num_blocks:
            raise RuntimeError(
                f"compact_poison_probe: block id {block_id} out of pool range "
                f"num_blocks={num_blocks}"
            )

    # 4. Poison every registered layer's key+value at the chosen blocks. Writes
    #    go on the current (default) CUDA stream -> ordered before the forward,
    #    which a FULL-graph replay reads on the same stream. No new stream/event.
    for _layer_index, (key_cache, value_cache) in layers:
        for j, block_id in enumerate(block_ids):
            val = _POISON_VALUE if (j % 2 == 0) else -_POISON_VALUE
            key_cache[block_id].fill_(val)
            value_cache[block_id].fill_(-val)

    _emit_evidence(row=target_row, slot=slot, block_ids=block_ids, n_layers=n_layers)


def _lease_from_controller(controller: object) -> Optional[object]:
    """First compact page lease found on any layer state (all layers share it)."""
    layer_states = getattr(controller, "layer_states", None)
    if not layer_states:
        return None
    for state in layer_states.values():
        residency = getattr(state, "compact_page_residency", None)
        lease = getattr(residency, "lease", None)
        if lease is not None:
            return lease
    return None


def _compact_slot_block_ids(lease: object, slot: int, gens: int) -> List[int]:
    """Reserved compact block ids for ``slot`` across every generation.

    ``slot_manager_block_ids`` indexes by sub-slot ``gen*max_live + slot``. Dual
    generation (gens==2) gives a slot two half-regions; poison both so whichever
    generation the current read targets is covered. gens==1 -> raw slot only.
    """
    max_live = int(getattr(lease, "max_live_sparse_slots"))
    ids: List[int] = []
    seen: set[int] = set()
    for gen in range(int(gens)):
        sub_slot = gen * max_live + int(slot)
        for block_id in lease.slot_manager_block_ids(sub_slot):
            block_id_i = int(block_id)
            if block_id_i not in seen:
                seen.add(block_id_i)
                ids.append(block_id_i)
    return ids


def _native_control_block_ids(
    *,
    controller: object,
    row: int,
    seq_lens_by_row: Sequence[int],
    block_size: int,
    lease: object,
) -> List[int]:
    """~8 mid-context canonical (full-KV) blocks the sparse row should skip.

    Excludes: logical page 0 (sink), any physical block id 0 (sink) or padding,
    the last ``ceil((recent+block_size-1)/block_size)+2`` logical pages (recent
    window), and any block already in the compact reserved pool.
    """
    if row >= len(seq_lens_by_row):
        raise RuntimeError(
            f"compact_poison_probe(native): row {row} outside seq_lens "
            f"(len={len(seq_lens_by_row)})"
        )
    seq_len = int(seq_lens_by_row[row])
    if seq_len <= 0:
        raise RuntimeError(
            f"compact_poison_probe(native): non-positive seq_len {seq_len} for row {row}"
        )
    logical_pages = (seq_len + block_size - 1) // block_size

    config = getattr(controller, "config", None)
    if config is None or not hasattr(config, "recent"):
        raise RuntimeError(
            "compact_poison_probe(native): controller.config.recent unavailable"
        )
    recent = int(getattr(config, "recent"))
    recent_window_pages = math.ceil((recent + block_size - 1) / block_size) + 2

    row_blocks = _canonical_row_block_ids(controller, row)
    reserved_set = {
        int(block_id)
        for block_id in getattr(lease, "reserved_manager_block_ids", ())
    }

    # Exclusive upper bound on the logical page index we may poison.
    hi = logical_pages - recent_window_pages
    upper = min(hi, len(row_blocks))
    candidates: List[int] = []
    for logical_page in range(1, upper):  # start at 1 -> exclude sink page 0
        block_id = int(row_blocks[logical_page])
        if block_id <= 0:  # physical block 0 (sink) or padding sentinel
            continue
        if block_id in reserved_set:  # compact-reserved -> not a native control
            continue
        candidates.append(block_id)
    if not candidates:
        raise RuntimeError(
            "compact_poison_probe(native): no eligible mid-context blocks "
            f"(seq_len={seq_len}, logical_pages={logical_pages}, "
            f"recent_window_pages={recent_window_pages})"
        )
    if len(candidates) <= 8:
        return candidates
    mid = len(candidates) // 2
    return candidates[max(0, mid - 4): mid + 4]


def _canonical_row_block_ids(controller: object, row: int) -> List[int]:
    """Physical block ids of the row's canonical (full-KV) page table.

    Prefers the CPU table (host-deterministic, no sync). Falls back to the GPU
    table's single row (``.tolist()`` triggers one sync, acceptable one-shot).
    """
    table_cpu = getattr(controller, "_worker_block_table_cpu", None)
    if table_cpu is not None:
        return _row_to_int_list(table_cpu, row)
    table_gpu = getattr(controller, "_worker_block_table", None)
    if table_gpu is not None:
        return _row_to_int_list(table_gpu, row)
    raise RuntimeError(
        "compact_poison_probe(native): no canonical block table on controller "
        "(_worker_block_table_cpu / _worker_block_table both None)"
    )


def _row_to_int_list(table: object, row: int) -> List[int]:
    row_obj = table[row]
    tolist = getattr(row_obj, "tolist", None)
    if callable(tolist):
        return [int(x) for x in tolist()]
    return [int(x) for x in row_obj]


def _emit_evidence(
    *,
    row: int,
    slot: int,
    block_ids: Sequence[int],
    n_layers: int,
) -> None:
    record = {
        "probe": "compact_poison",
        "mode": _PROBE_MODE,
        "fire_step": _PROBE_FIRE_STEP,
        "row": int(row),
        "slot": int(slot),
        "n_blocks_poisoned": len(block_ids),
        "block_ids": [int(b) for b in block_ids[:8]],
        "n_layers": int(n_layers),
        "pid": os.getpid(),
    }
    line = json.dumps(record)
    if _PROBE_LOG_PATH:
        try:
            with open(_PROBE_LOG_PATH, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
            return
        except OSError as exc:
            # Redirect (not swallow) so the evidence is never lost.
            sys.stderr.write(
                f"compact_poison_probe: log path {_PROBE_LOG_PATH!r} unwritable "
                f"({exc}); emitting to stderr\n"
            )
    sys.stderr.write(line + "\n")
    sys.stderr.flush()
