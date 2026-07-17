"""Generation handshake between RRP graph readers and descriptor writers."""

from __future__ import annotations

from collections.abc import Callable

from patches.fa_sparse_runtime.mixed_page_cudagraph_replay import (
    validate_resolved_row_ptr_ready_state,
)


def _require_exact_int(value: object, *, context: str) -> int:
    if type(value) is not int:
        raise RuntimeError(context)
    return value


def rrp_storage_key(
    *,
    arena_key: object,
) -> tuple[object, ...]:
    """Build the stable physical-owner identity shared by readers and writers."""

    if type(arena_key) is not tuple or not arena_key:
        raise RuntimeError(
            "RRP replay-consumed state has malformed storage identity"
        )
    # ``pointer_signature`` also contains mutable affine scalars and resolver
    # mode values, so it is a descriptor identity, not a storage identity.
    # The controller arena key is the owner/cache key for all graph-live tensors
    # in one RRP arena; conservatively joining those tensors is both stable and
    # bounded across layout changes.
    storage_key = arena_key
    try:
        hash(storage_key)
    except TypeError as exc:
        raise RuntimeError(
            "RRP replay-consumed state has unhashable storage identity"
        ) from exc
    return storage_key


def _rrp_storage_key(state: dict[str, object]) -> tuple[object, ...]:
    if state.get("route_family") != "resolved_row_ptr":
        raise RuntimeError("RRP replay-consumed state has a non-RRP route family")
    pointer_signature = state.get("pointer_signature")
    if type(pointer_signature) is not tuple or not pointer_signature:
        raise RuntimeError(
            "RRP replay-consumed state has malformed descriptor identity"
        )
    return rrp_storage_key(arena_key=state.get("arena_key"))


def _rrp_consumed_ledger(
    controller: object,
    *,
    create: bool,
) -> dict[tuple[object, ...], tuple[object, ...]]:
    ledger = getattr(controller, "_rrp_replay_consumed_by_storage", None)
    if ledger is None:
        if not create:
            return {}
        ledger = {}
        setattr(controller, "_rrp_replay_consumed_by_storage", ledger)
    if type(ledger) is not dict:
        raise RuntimeError("RRP replay-consumed storage ledger is malformed")
    return ledger


def _validated_replay_generation_proof(
    replay_proof: object,
    *,
    state: dict[str, object],
    missing_context: str,
    generation_context: str,
) -> int:
    if getattr(replay_proof, "replay_generation_state", None) is not state:
        raise RuntimeError(missing_context)
    return _require_exact_int(
        getattr(replay_proof, "replay_generation", None),
        context=generation_context,
    )


def _validated_consumed_entry(
    entry: object,
    *,
    storage_key: tuple[object, ...],
    context: str,
) -> tuple[dict[str, object], int, int, object, int]:
    if not isinstance(entry, tuple) or len(entry) != 5:
        raise RuntimeError(context)
    replay_state, generation_raw, sequence_raw, event, stream_raw = entry
    generation = _require_exact_int(generation_raw, context=context)
    sequence = _require_exact_int(sequence_raw, context=context)
    replay_stream_identity = _require_exact_int(stream_raw, context=context)
    try:
        replay_storage_key = (
            _rrp_storage_key(replay_state)
            if isinstance(replay_state, dict)
            else None
        )
    except RuntimeError as exc:
        raise RuntimeError(context) from exc
    if (
        replay_storage_key != storage_key
        or generation < 0
        or sequence < 0
        or event is None
        or replay_stream_identity < 0
    ):
        raise RuntimeError(context)
    return (
        replay_state,
        generation,
        sequence,
        event,
        replay_stream_identity,
    )


def snapshot_rrp_replay_consumed_generation(
    controller: object,
    *,
    state: dict[str, object],
) -> object:
    """Return the immutable consumed entry for exactly one RRP storage."""

    ledger = _rrp_consumed_ledger(controller, create=False)
    return ledger.get(_rrp_storage_key(state))


def record_rrp_replay_consumed_generation(
    controller: object,
    *,
    state: dict[str, object],
    replay_proof: object,
    stream: object,
    stream_identity: int,
    event_factory: Callable[[], object],
) -> int:
    """Record the completion of the exact RRP generation selected for replay."""

    generation = _validated_replay_generation_proof(
        replay_proof,
        state=state,
        missing_context=(
            "RRP graph replay completed without an exact prebound generation proof"
        ),
        generation_context="RRP graph replay has malformed consumed generation",
    )
    replay_stream_identity = _require_exact_int(
        stream_identity,
        context="RRP graph replay has malformed stream identity",
    )
    if replay_stream_identity < 0:
        raise RuntimeError("RRP graph replay has malformed stream identity")
    ready_state = validate_resolved_row_ptr_ready_state(
        event=state.get("ready_event"),
        generation=state.get("ready_event_generation", -1),
        ready_stream=state.get("ready_event_stream", -1),
        same_stream_ordered=state.get("same_stream_ordered", False),
        require_published=True,
        context="RRP graph replay consumed state",
    )
    if ready_state.generation != generation:
        raise RuntimeError(
            "RRP graph replay ready generation changed between prebind and completion"
        )

    storage_key = _rrp_storage_key(state)
    ledger = _rrp_consumed_ledger(controller, create=True)
    previous = ledger.get(storage_key)
    if previous is None:
        event = event_factory()
        sequence = 0
    else:
        (
            _previous_state,
            _previous_generation,
            previous_sequence,
            event,
            previous_stream_identity,
        ) = _validated_consumed_entry(
            previous,
            storage_key=storage_key,
            context="RRP replay-consumed storage ledger entry is malformed",
        )
        sequence = previous_sequence + 1
        # A single latest event is sufficient only if it dominates every prior
        # reader of this storage.  When replay moves streams, append the prior
        # event wait *after* the current graph replay and before re-recording the
        # event, so the new record joins both reader streams without serializing
        # unrelated arenas.
        if previous_stream_identity != replay_stream_identity:
            wait_event = getattr(stream, "wait_event", None)
            if not callable(wait_event):
                raise RuntimeError(
                    "RRP replay stream does not support cross-stream event chaining"
                )
            wait_event(event)
    record = getattr(event, "record", None)
    if not callable(record):
        raise RuntimeError("RRP replay-consumed event does not support record")
    record(stream)
    consumed_state = (
        state,
        generation,
        sequence,
        event,
        replay_stream_identity,
    )
    ledger[storage_key] = consumed_state
    return sequence


def replay_consumed_generation_was_recorded_since(
    controller: object,
    *,
    state: dict[str, object],
    replay_proof: object,
    previous_consumed_state: object,
) -> bool:
    """Detect an exact nested replay record without admitting a double record."""

    storage_key = _rrp_storage_key(state)
    consumed_state = _rrp_consumed_ledger(controller, create=False).get(storage_key)
    if consumed_state is previous_consumed_state:
        return False
    replay_state, generation, sequence, event, _stream = _validated_consumed_entry(
        consumed_state,
        storage_key=storage_key,
        context="nested RRP replay published malformed consumed state",
    )
    expected_generation = _validated_replay_generation_proof(
        replay_proof,
        state=state,
        missing_context="nested RRP replay replaced the prebound generation proof",
        generation_context="nested RRP replay has malformed prebound generation",
    )
    if (
        replay_state is not state
        or generation != expected_generation
        or sequence < 0
        or event is None
    ):
        raise RuntimeError("nested RRP replay published conflicting consumed state")
    return True


def wait_for_prior_rrp_replay_before_mutation(
    controller: object,
    *,
    stream: object,
    stream_identity: int,
    target_storage_key: tuple[object, ...] | None,
) -> int:
    """Order an actual RRP mutation after the latest replay of that storage."""

    ledger = _rrp_consumed_ledger(controller, create=False)
    if not ledger:
        return 0
    writer_stream_identity = _require_exact_int(
        stream_identity,
        context="RRP replay-consumed generation state is malformed",
    )
    if writer_stream_identity < 0:
        raise RuntimeError("RRP replay-consumed generation state is malformed")

    if target_storage_key is None:
        raise RuntimeError(
            "RRP replay-consumed storage exists without a target writer binding"
        )
    if type(target_storage_key) is not tuple or not target_storage_key:
        raise RuntimeError("RRP target writer storage identity is malformed")
    # Revalidate rather than trusting a caller-created tuple.  This also
    # rejects unhashable nested identities before a dictionary lookup.
    target_storage_key = rrp_storage_key(
        arena_key=target_storage_key,
    )
    consumed_state = ledger.get(target_storage_key)
    if consumed_state is None:
        return 0
    (
        _replay_state,
        generation,
        sequence,
        event,
        replay_stream_identity,
    ) = _validated_consumed_entry(
        consumed_state,
        storage_key=target_storage_key,
        context="RRP replay-consumed generation state is malformed",
    )

    waited_by_storage = getattr(
        controller,
        "_rrp_writer_waited_replay_by_storage",
        None,
    )
    if waited_by_storage is None:
        waited_by_storage = {}
        setattr(
            controller,
            "_rrp_writer_waited_replay_by_storage",
            waited_by_storage,
        )
    if type(waited_by_storage) is not dict:
        raise RuntimeError("RRP writer wait ledger is malformed")
    wait_key = (generation, sequence, writer_stream_identity)
    if waited_by_storage.get(target_storage_key) == wait_key:
        return 0

    wait_count = 0
    if (
        replay_stream_identity < 0
        or writer_stream_identity < 0
        or replay_stream_identity != writer_stream_identity
    ):
        wait_event = getattr(stream, "wait_event", None)
        if not callable(wait_event):
            raise RuntimeError("RRP mutation stream does not support wait_event")
        wait_event(event)
        wait_count = 1
    waited_by_storage[target_storage_key] = wait_key
    return wait_count
