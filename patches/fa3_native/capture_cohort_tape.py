from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import Any, Hashable, Optional, Sequence

import torch


CAPTURE_COHORT_TAPE_SCHEMA = "sfi.fa3_capture_cohort_tape.v4"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")

CaptureCohortTapeOwnerKey = tuple[str, int, int, int, int]
CaptureCohortTapeGenerationKey = tuple[str, int, int, int]


def _positive_int(name: str, value: object) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact int")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _non_negative_int(name: str, value: object) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact int")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _plan_signature(value: object) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError("capture cohort tape requires a lowercase sha256 plan signature")
    return value


def _unique_hashable_tokens(
    name: str,
    values: Sequence[Hashable],
    *,
    allow_empty: bool,
) -> tuple[Hashable, ...]:
    tokens = tuple(values)
    if not allow_empty and not tokens:
        raise RuntimeError(f"E_SFI_CAPTURE_COHORT_TAPE_{name}_EMPTY")
    try:
        token_set = set(tokens)
    except TypeError as exc:
        raise TypeError(f"capture cohort tape {name.lower()} must be hashable") from exc
    if len(token_set) != len(tokens):
        raise RuntimeError(f"E_SFI_CAPTURE_COHORT_TAPE_DUPLICATE_{name}")
    return tokens


def capture_cohort_stream_identity(stream: object) -> int:
    identity = getattr(stream, "cuda_stream", None)
    if type(identity) is not int or identity < 0:
        raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_STREAM_IDENTITY")
    return identity


@dataclass(frozen=True, slots=True)
class CaptureCohortTapeReport:
    schema: str
    slot_capacity: int
    layer_capacity: int
    group_count: int
    num_query_heads: int
    logical_k_capacity: int
    cohort_size: int
    scores_bytes: int
    denoms_bytes: int
    total_device_bytes: int


@dataclass(frozen=True, slots=True)
class CaptureCohortTapeState:
    report: CaptureCohortTapeReport
    plan_signature: str
    scores: torch.Tensor
    denoms: torch.Tensor


@dataclass(frozen=True, slots=True)
class CaptureCohortTapeGroup:
    request_slot: int
    owner_key: CaptureCohortTapeOwnerKey
    layer_slots: tuple[int, ...]
    scores_base: torch.Tensor
    denoms_base: torch.Tensor
    num_query_heads: int
    logical_k: int
    has_denoms: bool

    @property
    def scores(self) -> torch.Tensor:
        first = self.layer_slots[0]
        return self.scores_base[
            first : first + len(self.layer_slots),
            self.request_slot : self.request_slot + 1,
            : self.num_query_heads,
            :1,
            : self.logical_k,
        ]

    @property
    def denoms(self) -> Optional[torch.Tensor]:
        if not self.has_denoms:
            return None
        first = self.layer_slots[0]
        return self.denoms_base[
            first : first + len(self.layer_slots),
            self.request_slot : self.request_slot + 1,
            : self.num_query_heads,
        ]

@dataclass(frozen=True, slots=True)
class CaptureCohortSnapshot:
    layer_slots: tuple[int, ...]
    copy_completion_event: Any


def plan_capture_cohort_tape(
    *,
    slot_capacity: int,
    layer_capacity: int,
    num_query_heads: int,
    logical_k_capacity: int,
    cohort_size: int,
) -> CaptureCohortTapeReport:
    """Return the exact persistent footprint without depending on a policy stamp.

    A zero slot capacity represents an unavailable tape during cold-side policy
    planning.  It can select the ring owner, but can never be prepared or used.
    """

    slots = _non_negative_int("slot_capacity", slot_capacity)
    layers = _positive_int("layer_capacity", layer_capacity)
    heads = _positive_int("num_query_heads", num_query_heads)
    logical_k = _positive_int("logical_k_capacity", logical_k_capacity)
    cohort = _positive_int("cohort_size", cohort_size)
    group_count = (layers + cohort - 1) // cohort
    score_elements = slots * layers * heads * logical_k
    denom_elements = slots * layers * heads
    scores_bytes = score_elements * torch.tensor([], dtype=torch.float16).element_size()
    denoms_bytes = denom_elements * torch.tensor([], dtype=torch.float32).element_size()
    return CaptureCohortTapeReport(
        schema=CAPTURE_COHORT_TAPE_SCHEMA,
        slot_capacity=slots,
        layer_capacity=layers,
        group_count=group_count,
        num_query_heads=heads,
        logical_k_capacity=logical_k,
        cohort_size=cohort,
        scores_bytes=scores_bytes,
        denoms_bytes=denoms_bytes,
        total_device_bytes=scores_bytes + denoms_bytes,
    )


@dataclass(slots=True)
class _GroupLease:
    owner_key: CaptureCohortTapeOwnerKey
    reservation_serial: int
    consumer_token: Hashable
    submission_started: bool = False
    sealed: bool = False
    released: bool = False


@dataclass(slots=True)
class _SlotLease:
    generation_key: CaptureCohortTapeGenerationKey
    request_id: str
    groups: dict[int, _GroupLease]
    retired: bool = False


@dataclass(frozen=True, slots=True)
class CaptureCohortTapeSnapshotGrant:
    owner_key: CaptureCohortTapeOwnerKey
    request_slots: tuple[int, ...]
    reservation_serial: int


def _validate_owner_key(value: object) -> CaptureCohortTapeOwnerKey:
    if not isinstance(value, tuple) or len(value) != 5:
        raise ValueError("capture cohort tape owner key must have five fields")
    signature = _plan_signature(value[0])
    numeric = tuple(_non_negative_int("owner_key", item) for item in value[1:])
    return (signature, numeric[0], numeric[1], numeric[2], numeric[3])


def _generation_key(
    owner_key: CaptureCohortTapeOwnerKey,
) -> CaptureCohortTapeGenerationKey:
    return owner_key[:4]


def _consumer_token_fields(
    value: object,
    *,
    expected_owner: Optional[CaptureCohortTapeOwnerKey] = None,
) -> tuple[CaptureCohortTapeOwnerKey, int, str]:
    if not isinstance(value, tuple) or len(value) != 3:
        raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_CONSUMER_TOKEN")
    owner = _validate_owner_key(value[0])
    if expected_owner is not None and owner != expected_owner:
        raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_CONSUMER_OWNER_DRIFT")
    slot = _non_negative_int("consumer_slot", value[1])
    request_id = value[2]
    if not isinstance(request_id, str) or not request_id:
        raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_CONSUMER_REQUEST")
    return owner, slot, request_id


class CaptureCohortTapeLeaseTracker:
    """Own fixed request slots across snapshot writes and deferred reads.

    The global request-slot allocator is the concurrency authority.  A tape row
    therefore follows that stable slot instead of consuming one full bank per
    capture generation.  Every write/read/retire submission is bound to one
    consumer stream, so its FIFO is the reuse order; the runtime never host-
    waits, scans for a spare bank, or allocates a fallback.
    """

    __slots__ = (
        "_lock",
        "_report",
        "_consumer_stream_identity",
        "_slots",
        "_next_reservation_serial",
        "_prior_slots_by_reservation",
        "_poison_reason",
    )

    def __init__(
        self,
        report: CaptureCohortTapeReport,
        *,
        consumer_stream_identity: int,
    ) -> None:
        if not isinstance(report, CaptureCohortTapeReport):
            raise TypeError("capture cohort tape tracker requires its immutable report")
        if report.slot_capacity <= 0:
            raise ValueError("capture cohort tape tracker requires positive slot capacity")
        self._lock = threading.Lock()
        self._report = report
        self._consumer_stream_identity = _non_negative_int(
            "consumer_stream_identity", consumer_stream_identity
        )
        self._slots: list[Optional[_SlotLease]] = [None] * report.slot_capacity
        self._next_reservation_serial = 0
        self._prior_slots_by_reservation: dict[
            int, tuple[tuple[int, Optional[_SlotLease]], ...]
        ] = {}
        self._poison_reason = ""

    @property
    def report(self) -> CaptureCohortTapeReport:
        return self._report

    @property
    def consumer_stream_identity(self) -> int:
        return self._consumer_stream_identity

    def begin_snapshot(
        self,
        *,
        owner_key: CaptureCohortTapeOwnerKey,
        layer_slots: Sequence[int],
        expected_group_size: int,
        consumer_tokens: Sequence[Hashable],
    ) -> CaptureCohortTapeSnapshotGrant:
        owner = _validate_owner_key(owner_key)
        tokens = _unique_hashable_tokens(
            "CONSUMER", consumer_tokens, allow_empty=False
        )
        token_fields = tuple(
            _consumer_token_fields(token, expected_owner=owner) for token in tokens
        )
        request_slots = tuple(fields[1] for fields in token_fields)
        if len(set(request_slots)) != len(request_slots):
            raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_DUPLICATE_REQUEST_SLOT")
        slots = tuple(_non_negative_int("layer_slot", value) for value in layer_slots)
        group_size = _positive_int("expected_group_size", expected_group_size)
        report = self._report
        if not slots or len(slots) != group_size:
            raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_GROUP_SIZE")
        if slots != tuple(range(slots[0], slots[0] + len(slots))):
            raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_NONCONTIGUOUS_GROUP")
        group_id = owner[-1]
        expected_first = group_id * report.cohort_size
        expected_size = min(report.cohort_size, report.layer_capacity - expected_first)
        if (
            expected_first < 0
            or expected_first >= report.layer_capacity
            or slots[0] != expected_first
            or group_size != expected_size
            or slots[-1] >= report.layer_capacity
        ):
            raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_LAYER_GROUP_DRIFT")
        if any(slot >= report.slot_capacity for slot in request_slots):
            raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_REQUEST_SLOT_CAPACITY")

        generation = _generation_key(owner)
        with self._lock:
            self._raise_if_poisoned_locked()
            replacements: list[tuple[int, Optional[_SlotLease]]] = []
            for token, (_, request_slot, request_id) in zip(
                tokens, token_fields, strict=True
            ):
                lease = self._slots[request_slot]
                same_generation = bool(
                    lease is not None and lease.generation_key == generation
                )
                if same_generation:
                    assert lease is not None
                    if lease.request_id != request_id or lease.retired:
                        raise RuntimeError(
                            "E_SFI_CAPTURE_COHORT_TAPE_SLOT_OWNER_DRIFT"
                        )
                    if lease.groups and not lease.groups[len(lease.groups) - 1].sealed:
                        raise RuntimeError(
                            "E_SFI_CAPTURE_COHORT_TAPE_PENDING_SNAPSHOT"
                        )
                    if group_id in lease.groups:
                        raise RuntimeError(
                            "E_SFI_CAPTURE_COHORT_TAPE_DUPLICATE_SNAPSHOT"
                        )
                    if group_id != len(lease.groups):
                        raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_GROUP_ORDER")
                    continue
                if group_id != 0:
                    raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_GROUP_ORDER")
                if lease is not None:
                    if not self._slot_complete_locked(lease):
                        raise RuntimeError(
                            "E_SFI_CAPTURE_COHORT_TAPE_SLOT_LEASE_CONFLICT: "
                            f"slot={request_slot} current_request={lease.request_id!r} "
                            f"next_request={request_id!r}"
                        )
                replacements.append((request_slot, lease))

            self._next_reservation_serial += 1
            reservation_serial = self._next_reservation_serial
            replacement_slots = {slot for slot, _ in replacements}
            for token, (_, request_slot, request_id) in zip(
                tokens, token_fields, strict=True
            ):
                if request_slot in replacement_slots:
                    lease = _SlotLease(
                        generation_key=generation,
                        request_id=request_id,
                        groups={},
                    )
                    self._slots[request_slot] = lease
                else:
                    lease = self._slots[request_slot]
                if lease is None or lease.generation_key != generation:
                    raise RuntimeError(
                        "E_SFI_CAPTURE_COHORT_TAPE_GENERATION_DRIFT"
                    )
                lease.groups[group_id] = _GroupLease(
                    owner_key=owner,
                    reservation_serial=reservation_serial,
                    consumer_token=token,
                )
            if replacements:
                self._prior_slots_by_reservation[reservation_serial] = tuple(
                    replacements
                )
            return CaptureCohortTapeSnapshotGrant(
                owner_key=owner,
                request_slots=request_slots,
                reservation_serial=reservation_serial,
            )

    def mark_snapshot_submission_started(
        self,
        *,
        grant: CaptureCohortTapeSnapshotGrant,
    ) -> None:
        with self._lock:
            self._raise_if_poisoned_locked()
            groups = self._require_grant_locked(grant)
            if any(group.sealed for group in groups):
                raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_SUBMIT_AFTER_COMMIT")
            if any(group.submission_started for group in groups):
                raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_DUPLICATE_SUBMISSION")
            for group in groups:
                group.submission_started = True

    def commit_snapshot(
        self,
        *,
        grant: CaptureCohortTapeSnapshotGrant,
        consumer_tokens: Sequence[Hashable],
    ) -> None:
        tokens = _unique_hashable_tokens(
            "CONSUMER", consumer_tokens, allow_empty=False
        )
        with self._lock:
            self._raise_if_poisoned_locked()
            groups = self._require_grant_locked(grant)
            if set(tokens) != {group.consumer_token for group in groups}:
                raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_CONSUMER_DRIFT")
            if any(group.sealed for group in groups):
                raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_DUPLICATE_COMMIT")
            if any(not group.submission_started for group in groups):
                raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_COMMIT_BEFORE_SUBMIT")
            for group in groups:
                group.sealed = True
            self._prior_slots_by_reservation.pop(
                grant.reservation_serial, None
            )

    def abort_snapshot(
        self,
        *,
        grant: CaptureCohortTapeSnapshotGrant,
    ) -> None:
        """Rollback one untouched tail reservation without hiding submitted work."""

        with self._lock:
            self._raise_if_poisoned_locked()
            groups = self._require_grant_locked(grant)
            if any(group.sealed for group in groups):
                raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_ABORT_AFTER_COMMIT")
            if any(group.submission_started for group in groups):
                raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_ABORT_AFTER_SUBMIT")
            group_id = grant.owner_key[-1]
            prior_slots = dict(
                self._prior_slots_by_reservation.pop(
                    grant.reservation_serial, tuple()
                )
            )
            for request_slot in grant.request_slots:
                lease = self._slots[request_slot]
                if lease is None or group_id != len(lease.groups) - 1:
                    raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_ABORT_NONTAIL")
                if request_slot in prior_slots:
                    self._slots[request_slot] = prior_slots[request_slot]
                else:
                    del lease.groups[group_id]

    def poison_snapshot(
        self,
        *,
        grant: CaptureCohortTapeSnapshotGrant,
        reason: str,
    ) -> None:
        """Make ambiguous/post-submit ownership terminal instead of reusing it."""

        reason_s = str(reason).strip()
        if not reason_s:
            reason_s = "snapshot lifecycle failed"
        with self._lock:
            if self._poison_reason:
                return
            self._require_grant_locked(grant)
            self._poison_reason = reason_s

    def release_consumers(
        self,
        *,
        owner_key: CaptureCohortTapeOwnerKey,
        consumer_tokens: Sequence[Hashable],
        consumer_stream_identity: int,
    ) -> bool:
        owner = _validate_owner_key(owner_key)
        tokens = _unique_hashable_tokens(
            "RELEASE", consumer_tokens, allow_empty=False
        )
        if (
            _non_negative_int("consumer_stream_identity", consumer_stream_identity)
            != self._consumer_stream_identity
        ):
            raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_RELEASE_STREAM_DRIFT")
        with self._lock:
            self._raise_if_poisoned_locked()
            pending_releases: list[tuple[int, _SlotLease, _GroupLease]] = []
            for token in tokens:
                _, request_slot, request_id = _consumer_token_fields(
                    token, expected_owner=owner
                )
                lease, group = self._require_group_locked(owner, request_slot)
                if lease.request_id != request_id or group.consumer_token != token:
                    raise RuntimeError(
                        "E_SFI_CAPTURE_COHORT_TAPE_UNEXPECTED_RELEASE"
                    )
                if not group.sealed:
                    raise RuntimeError(
                        "E_SFI_CAPTURE_COHORT_TAPE_RELEASE_BEFORE_SEAL"
                    )
                if group.released:
                    raise RuntimeError(
                        "E_SFI_CAPTURE_COHORT_TAPE_DUPLICATE_RELEASE"
                    )
                pending_releases.append((request_slot, lease, group))
            if any(
                self._slots[request_slot] is not lease
                for request_slot, lease, _ in pending_releases
            ):
                raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_SLOT_OWNER_DRIFT")

            # Validation above is deliberately transactional.  A malformed
            # token batch must not release an earlier slot and leave ownership
            # half-committed before the caller fails closed.
            for _, _, group in pending_releases:
                group.released = True
            all_complete = True
            for _, lease, _ in pending_releases:
                if not self._slot_complete_locked(lease):
                    all_complete = False
            return all_complete

    def request_requires_retirement(self, *, slot: int, request_id: str) -> bool:
        request_slot = _non_negative_int("slot", slot)
        if request_slot >= self._report.slot_capacity:
            raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_REQUEST_SLOT_CAPACITY")
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("capture cohort tape request_id must be non-empty")
        with self._lock:
            self._raise_if_poisoned_locked()
            lease = self._slots[request_slot]
            if lease is None:
                return False
            if lease.request_id != request_id:
                # A request may acquire and release a global slot without ever
                # publishing a tape snapshot.  In that case the slot can still
                # contain its previous, fully released tape owner.  It is stale
                # metadata, not an ownership conflict; an incomplete mismatch is
                # the allocator/tape lifecycle violation that must fail closed.
                if self._slot_complete_locked(lease):
                    return False
                raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_SLOT_OWNER_DRIFT")
            return not self._slot_complete_locked(lease)

    def retire_request(
        self,
        *,
        slot: int,
        request_id: str,
        consumer_stream_identity: int,
    ) -> bool:
        request_slot = _non_negative_int("slot", slot)
        if request_slot >= self._report.slot_capacity:
            raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_REQUEST_SLOT_CAPACITY")
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("capture cohort tape request_id must be non-empty")
        if (
            _non_negative_int("consumer_stream_identity", consumer_stream_identity)
            != self._consumer_stream_identity
        ):
            raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_RETIRE_STREAM_DRIFT")
        with self._lock:
            self._raise_if_poisoned_locked()
            lease = self._slots[request_slot]
            if lease is None:
                return False
            if lease.request_id != request_id:
                raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_SLOT_OWNER_DRIFT")
            if self._slot_complete_locked(lease):
                return False
            if any(not group.sealed for group in lease.groups.values()):
                # Retirement is allowed only after snapshot publication became
                # immutable. Otherwise retirement could become visible before a
                # concurrent copy is submitted and let later FIFO reuse overtake it.
                raise RuntimeError(
                    "E_SFI_CAPTURE_COHORT_TAPE_RETIRE_DURING_SNAPSHOT"
                )
            lease.retired = True
            return True

    def _require_group_locked(
        self,
        owner_key: CaptureCohortTapeOwnerKey,
        request_slot: int,
    ) -> tuple[_SlotLease, _GroupLease]:
        if request_slot >= self._report.slot_capacity:
            raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_OWNER_DRIFT")
        lease = self._slots[request_slot]
        if lease is None or lease.generation_key != _generation_key(owner_key):
            raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_OWNER_DRIFT")
        group = lease.groups.get(owner_key[-1])
        if group is None or group.owner_key != owner_key:
            raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_OWNER_DRIFT")
        return lease, group

    def _require_grant_locked(
        self,
        grant: CaptureCohortTapeSnapshotGrant,
    ) -> tuple[_GroupLease, ...]:
        if not isinstance(grant, CaptureCohortTapeSnapshotGrant):
            raise TypeError("capture cohort tape requires a snapshot grant")
        owner = _validate_owner_key(grant.owner_key)
        reservation_serial = _positive_int(
            "reservation_serial", grant.reservation_serial
        )
        groups: list[_GroupLease] = []
        for request_slot in grant.request_slots:
            _, group = self._require_group_locked(owner, request_slot)
            if group.reservation_serial != reservation_serial:
                raise RuntimeError(
                    "E_SFI_CAPTURE_COHORT_TAPE_RESERVATION_DRIFT"
                )
            groups.append(group)
        if not groups:
            raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_RESERVATION_EMPTY")
        return tuple(groups)

    def _raise_if_poisoned_locked(self) -> None:
        if self._poison_reason:
            raise RuntimeError(
                "E_SFI_CAPTURE_COHORT_TAPE_TRACKER_POISONED: "
                f"{self._poison_reason}"
            )

    def _slot_complete_locked(self, lease: Optional[_SlotLease]) -> bool:
        if lease is None:
            return True
        if lease.retired:
            return True
        if len(lease.groups) != self._report.group_count:
            return False
        return all(group.sealed and group.released for group in lease.groups.values())


def prepare_capture_cohort_tape(
    *,
    controller: object,
    device: torch.device,
    report: CaptureCohortTapeReport,
    plan_signature: str,
    consumer_stream: torch.cuda.Stream,
) -> CaptureCohortTapeState:
    if controller is None:
        raise ValueError("capture cohort tape requires an explicit controller owner")
    if not isinstance(report, CaptureCohortTapeReport):
        raise TypeError("capture cohort tape requires its immutable report")
    if report.slot_capacity <= 0:
        raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_PREBUILD_CAPACITY")
    signature = _plan_signature(plan_signature)
    stream_identity = capture_cohort_stream_identity(consumer_stream)
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("capture cohort tape prebuild requires CUDA")
    prior = getattr(controller, "_capture_cohort_tape_state", None)
    prior_tracker = getattr(controller, "_capture_cohort_tape_lease_tracker", None)
    if prior is not None or prior_tracker is not None:
        if (
            not isinstance(prior, CaptureCohortTapeState)
            or not isinstance(prior_tracker, CaptureCohortTapeLeaseTracker)
            or prior.report != report
            or prior.plan_signature != signature
            or prior_tracker.report != report
            or prior_tracker.consumer_stream_identity != stream_identity
        ):
            raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_PREBUILD_DRIFT")
        return prior
    # Physical request-major storage makes any one-request, contiguous-layer
    # consumer view dense without another D2D materialization.  The logical
    # view is [layer, global_request_slot, head, window, k].
    scores = torch.empty(
        (
            report.slot_capacity,
            report.layer_capacity,
            report.num_query_heads,
            1,
            report.logical_k_capacity,
        ),
        dtype=torch.float16,
        device=device,
    ).permute(1, 0, 2, 3, 4)
    denoms = torch.empty(
        (
            report.slot_capacity,
            report.layer_capacity,
            report.num_query_heads,
        ),
        dtype=torch.float32,
        device=device,
    ).permute(1, 0, 2)
    state = CaptureCohortTapeState(
        report=report,
        plan_signature=signature,
        scores=scores,
        denoms=denoms,
    )
    setattr(controller, "_capture_cohort_tape_state", state)
    setattr(
        controller,
        "_capture_cohort_tape_lease_tracker",
        CaptureCohortTapeLeaseTracker(
            report,
            consumer_stream_identity=stream_identity,
        ),
    )
    return state


def quiesce_and_reset_capture_cohort_tape_leases(
    *,
    controller: object,
    consumer_stream: object,
) -> bool:
    """Reset request-slot leases at an explicit idle lifecycle boundary.

    The persistent tape allocation survives idle release, but its request
    ownership cannot survive a wholesale reset of the global slot allocator.
    Quiesce the sole consumer stream before replacing the tracker so a new
    request can reuse the same physical row without inheriting stale leases.
    """

    state = getattr(controller, "_capture_cohort_tape_state", None)
    tracker = getattr(controller, "_capture_cohort_tape_lease_tracker", None)
    if state is None and tracker is None:
        return False
    if not isinstance(state, CaptureCohortTapeState) or not isinstance(
        tracker, CaptureCohortTapeLeaseTracker
    ):
        raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_IDLE_OWNER_DRIFT")
    stream_identity = capture_cohort_stream_identity(consumer_stream)
    if (
        tracker.report != state.report
        or stream_identity != tracker.consumer_stream_identity
    ):
        raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_IDLE_STREAM_DRIFT")
    synchronize = getattr(consumer_stream, "synchronize", None)
    if not callable(synchronize):
        raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_IDLE_SYNC_MISSING")
    synchronize()
    setattr(
        controller,
        "_capture_cohort_tape_lease_tracker",
        CaptureCohortTapeLeaseTracker(
            state.report,
            consumer_stream_identity=tracker.consumer_stream_identity,
        ),
    )
    return True


def capture_cohort_tape_owner_key(
    state: CaptureCohortTapeState,
    payloads: Sequence[object],
) -> CaptureCohortTapeOwnerKey:
    payloads_t = tuple(payloads)
    if not payloads_t:
        raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_EMPTY_GROUP")
    layers = tuple(
        int(getattr(getattr(payload, "state", None), "layer_index", -1))
        for payload in payloads_t
    )
    if any(layer < 0 for layer in layers):
        raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_GLOBAL_LAYER_MISSING")
    group_id = layers[0] // state.report.cohort_size
    identities = tuple(
        (
            int(getattr(payload, "capture_handle_id", -1)),
            int(getattr(payload, "capture_handle_generation", -1)),
            int(getattr(payload, "capture_epoch", -1)),
        )
        for payload in payloads_t
    )
    if len(set(identities)) != 1 or any(value < 0 for value in identities[0]):
        raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_GENERATION_IDENTITY")
    handle_id, generation, epoch = identities[0]
    return _validate_owner_key(
        (state.plan_signature, handle_id, generation, epoch, group_id)
    )


def capture_cohort_consumer_token(
    owner_key: CaptureCohortTapeOwnerKey,
    *,
    slot: int,
    request_id: str,
) -> Hashable:
    owner = _validate_owner_key(owner_key)
    slot_i = _non_negative_int("slot", slot)
    request_id_s = str(request_id)
    if not request_id_s:
        raise ValueError("capture cohort tape consumer request_id must be non-empty")
    return (owner, slot_i, request_id_s)


def require_capture_cohort_consumer_coverage(
    *,
    registered_slots: Sequence[int],
    producer_slots: Sequence[int],
) -> tuple[int, ...]:
    """Prove every registered tape lease has exactly one deferred consumer."""

    registered = tuple(
        _non_negative_int("registered_slot", value) for value in registered_slots
    )
    producers = tuple(
        _non_negative_int("producer_slot", value) for value in producer_slots
    )
    if not registered:
        raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_CONSUMER_COVERAGE_EMPTY")
    if len(set(registered)) != len(registered):
        raise RuntimeError(
            "E_SFI_CAPTURE_COHORT_TAPE_DUPLICATE_REGISTERED_CONSUMER_SLOT"
        )
    if len(set(producers)) != len(producers):
        raise RuntimeError(
            "E_SFI_CAPTURE_COHORT_TAPE_DUPLICATE_PRODUCER_CONSUMER_SLOT"
        )
    registered_set = set(registered)
    producer_set = set(producers)
    missing = tuple(sorted(registered_set - producer_set))
    outside = tuple(sorted(producer_set - registered_set))
    if missing or outside:
        raise RuntimeError(
            "E_SFI_CAPTURE_COHORT_TAPE_CONSUMER_COVERAGE: "
            f"missing={missing} outside={outside}"
        )
    return tuple(sorted(registered))


def _require_affine_cohort_view(
    name: str,
    sources: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Recover the arena's layer dimension without materializing a stack.

    Capture layouts allocate one ``[cohort, row_capacity, ...]`` tensor and
    publish layer-local row-prefix views.  The row prefix may be smaller than
    the arena capacity, so adjacent layers need not be packed by ``numel``;
    their storage offsets must nevertheless form one exact affine sequence.
    """

    sources_t = tuple(sources)
    if not sources_t:
        raise RuntimeError(f"E_SFI_CAPTURE_COHORT_TAPE_{name}_EMPTY")
    first = sources_t[0]
    if (
        not isinstance(first, torch.Tensor)
        or first.layout != torch.strided
        or first.numel() <= 0
        or not first.is_contiguous()
    ):
        raise RuntimeError(f"E_SFI_CAPTURE_COHORT_TAPE_{name}_AFFINE_SOURCE")
    first_storage = first.untyped_storage()
    storage_ptr = int(first_storage.data_ptr())
    storage_nbytes = int(first_storage.nbytes())
    first_offset = int(first.storage_offset())
    element_size = int(first.element_size())
    source_numel = int(first.numel())
    for source in sources_t[1:]:
        if (
            not isinstance(source, torch.Tensor)
            or source.layout != torch.strided
            or source.dtype != first.dtype
            or source.device != first.device
            or tuple(source.shape) != tuple(first.shape)
            or tuple(source.stride()) != tuple(first.stride())
            or not source.is_contiguous()
            or int(source.untyped_storage().data_ptr()) != storage_ptr
            or int(source.untyped_storage().nbytes()) != storage_nbytes
        ):
            raise RuntimeError(
                f"E_SFI_CAPTURE_COHORT_TAPE_{name}_AFFINE_SOURCE"
            )
    if len(sources_t) == 1:
        layer_stride = source_numel
    else:
        layer_stride = int(sources_t[1].storage_offset()) - first_offset
        if layer_stride < source_numel or any(
            int(source.storage_offset()) != first_offset + index * layer_stride
            for index, source in enumerate(sources_t)
        ):
            raise RuntimeError(
                f"E_SFI_CAPTURE_COHORT_TAPE_{name}_AFFINE_SOURCE"
            )
    storage_elements = storage_nbytes // element_size
    if (
        first_offset < 0
        or storage_nbytes % element_size != 0
        or first_offset + (len(sources_t) - 1) * layer_stride + source_numel
        > storage_elements
    ):
        raise RuntimeError(f"E_SFI_CAPTURE_COHORT_TAPE_{name}_AFFINE_BOUNDS")
    return torch.as_strided(
        first,
        size=(len(sources_t), *tuple(first.shape)),
        stride=(layer_stride, *tuple(first.stride())),
        storage_offset=first_offset,
    )


def _same_tensor_view(lhs: torch.Tensor, rhs: torch.Tensor) -> bool:
    return bool(
        lhs.layout == rhs.layout
        and lhs.dtype == rhs.dtype
        and lhs.device == rhs.device
        and tuple(lhs.shape) == tuple(rhs.shape)
        and tuple(lhs.stride()) == tuple(rhs.stride())
        and int(lhs.storage_offset()) == int(rhs.storage_offset())
        and int(lhs.untyped_storage().data_ptr())
        == int(rhs.untyped_storage().data_ptr())
        and int(lhs.untyped_storage().nbytes())
        == int(rhs.untyped_storage().nbytes())
    )


def _require_source_outside_tape_storage(
    name: str,
    source: Optional[torch.Tensor],
    tape: torch.Tensor,
) -> None:
    if source is None:
        return
    if (
        source.device == tape.device
        and int(source.untyped_storage().data_ptr())
        == int(tape.untyped_storage().data_ptr())
    ):
        raise RuntimeError(
            f"E_SFI_CAPTURE_COHORT_TAPE_{name}_SOURCE_ALIASES_TAPE"
        )


def _coalesced_row_slot_runs(
    *,
    source_rows: Sequence[int],
    destination_slots: Sequence[int],
) -> tuple[tuple[int, int, int], ...]:
    source_rows_t = tuple(source_rows)
    destination_slots_t = tuple(destination_slots)
    if not source_rows_t or len(source_rows_t) != len(destination_slots_t):
        raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_ROW_SLOT_MAP")
    runs: list[tuple[int, int, int]] = []
    source_start = source_rows_t[0]
    source_stop = source_start + 1
    destination_start = destination_slots_t[0]
    previous_destination = destination_start
    for source_row, destination_slot in zip(
        source_rows_t[1:], destination_slots_t[1:], strict=True
    ):
        if source_row == source_stop and destination_slot == previous_destination + 1:
            source_stop += 1
            previous_destination = destination_slot
            continue
        runs.append((source_start, source_stop, destination_start))
        source_start = source_row
        source_stop = source_row + 1
        destination_start = destination_slot
        previous_destination = destination_slot
    runs.append((source_start, source_stop, destination_start))
    return tuple(runs)


def _copy_affine_cohort_slot_rows(
    *,
    destination_scores: torch.Tensor,
    destination_denoms: torch.Tensor,
    scores_source: torch.Tensor,
    denoms_source: Optional[torch.Tensor],
    lastn1_source: Optional[torch.Tensor],
    source_rows: Sequence[int],
    destination_slots: Sequence[int],
    lastn1_row_positions: Sequence[int],
) -> None:
    row_slot_runs = _coalesced_row_slot_runs(
        source_rows=source_rows,
        destination_slots=destination_slots,
    )
    for source_start, source_stop, destination_start in row_slot_runs:
        width = source_stop - source_start
        destination_stop = destination_start + width
        destination_scores[:, destination_start:destination_stop].copy_(
            scores_source[:, source_start:source_stop]
        )
        if denoms_source is not None:
            destination_denoms[:, destination_start:destination_stop].copy_(
                denoms_source[:, source_start:source_stop]
            )
    if lastn1_source is not None:
        lastn1_positions = set(lastn1_row_positions)
        lastn1_pairs = tuple(
            (source_row, destination_slot)
            for source_row, destination_slot in zip(
                source_rows, destination_slots, strict=True
            )
            if source_row in lastn1_positions
        )
        if lastn1_pairs:
            for source_start, source_stop, destination_start in (
                _coalesced_row_slot_runs(
                    source_rows=tuple(pair[0] for pair in lastn1_pairs),
                    destination_slots=tuple(pair[1] for pair in lastn1_pairs),
                )
            ):
                width = source_stop - source_start
                destination_scores[
                    :, destination_start : destination_start + width
                ].copy_(lastn1_source[:, source_start:source_stop])


def snapshot_capture_cohort_payload_group(
    *,
    state: CaptureCohortTapeState,
    tracker: CaptureCohortTapeLeaseTracker,
    payloads: Sequence[object],
    consumer_tokens: Sequence[Hashable],
    source_slots: Sequence[int],
    lastn1_row_positions: Sequence[int] = tuple(),
    stream: Optional[torch.cuda.Stream] = None,
    copy_completion_event: Any = None,
) -> CaptureCohortSnapshot:
    """Copy finalized request rows into their fixed global-slot tape rows."""

    if not isinstance(state, CaptureCohortTapeState):
        raise TypeError("capture cohort snapshot requires prebuilt state")
    if not isinstance(tracker, CaptureCohortTapeLeaseTracker):
        raise TypeError("capture cohort snapshot requires its lease tracker")
    if tracker.report != state.report:
        raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_TRACKER_DRIFT")
    payloads_t = tuple(payloads)
    owner = capture_cohort_tape_owner_key(state, payloads_t)
    layers = tuple(
        int(getattr(getattr(payload, "state", None), "layer_index", -1))
        for payload in payloads_t
    )
    tokens = _unique_hashable_tokens(
        "CONSUMER", consumer_tokens, allow_empty=False
    )
    token_fields = tuple(
        _consumer_token_fields(token, expected_owner=owner) for token in tokens
    )
    request_slots = tuple(fields[1] for fields in token_fields)
    if (
        len(set(request_slots)) != len(request_slots)
        or any(slot >= state.report.slot_capacity for slot in request_slots)
    ):
        raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_REQUEST_SLOT_CAPACITY")
    source_slots_t = tuple(
        _non_negative_int("source_slot", value) for value in source_slots
    )
    if len(set(source_slots_t)) != len(source_slots_t):
        raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_DUPLICATE_SOURCE_SLOT")
    if any(slot >= state.report.slot_capacity for slot in source_slots_t):
        raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_SOURCE_SLOT_CAPACITY")

    scores_sources: list[torch.Tensor] = []
    denoms_sources: list[torch.Tensor] = []
    lastn1_sources: list[torch.Tensor] = []
    completion_events: list[Any] = []
    shape: Optional[tuple[int, int, int, int]] = None
    for payload in payloads_t:
        scores = getattr(payload, "capture_scores", None)
        denoms = getattr(payload, "log_f_denoms", None)
        if (
            not isinstance(scores, torch.Tensor)
            or scores.dim() != 4
            or scores.dtype != torch.float16
        ):
            raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_SCORE_SOURCE")
        rows, heads, window, logical_k = tuple(int(v) for v in scores.shape)
        current_shape = (rows, heads, window, logical_k)
        if (
            window != 1
            or scores.device != state.scores.device
            or (shape is not None and current_shape != shape)
        ):
            raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_SOURCE_SHAPE")
        if denoms is not None:
            if (
                not isinstance(denoms, torch.Tensor)
                or denoms.dim() != 2
                or denoms.dtype != torch.float32
                or tuple(denoms.shape) != (rows, heads)
                or denoms.device != state.denoms.device
            ):
                raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_DENOM_SOURCE")
            denoms_sources.append(denoms)
        lastn1_source = getattr(payload, "lastn1_capture_scores", None)
        if lastn1_source is None:
            lastn1_source = scores
        if (
            not isinstance(lastn1_source, torch.Tensor)
            or lastn1_source.dtype != torch.float16
            or tuple(lastn1_source.shape) != current_shape
            or lastn1_source.device != state.scores.device
        ):
            raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_LASTN1_SOURCE")
        job = getattr(payload, "capture_postprocess_job", None)
        if job is not None:
            completion_event = getattr(job, "completion_event", None)
            if completion_event is None:
                raise RuntimeError(
                    "E_SFI_CAPTURE_COHORT_TAPE_POSTPROCESS_COMPLETION_MISSING"
                )
            if not bool(getattr(job, "completed", False)):
                raise RuntimeError(
                    "E_SFI_CAPTURE_COHORT_TAPE_POSTPROCESS_NOT_FINALIZED"
                )
            if all(id(event) != id(completion_event) for event in completion_events):
                completion_events.append(completion_event)
        shape = current_shape
        scores_sources.append(scores)
        lastn1_sources.append(lastn1_source)
    if shape is None:
        raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_EMPTY_GROUP")
    rows, heads, _, logical_k = shape
    report = state.report
    if (
        rows <= 0
        or heads != report.num_query_heads
        or logical_k != report.logical_k_capacity
    ):
        raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_FIXED_K_DRIFT")
    if denoms_sources and len(denoms_sources) != len(payloads_t):
        raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_MIXED_DENOM_GROUP")
    if len(source_slots_t) != rows:
        raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_SOURCE_SLOT_COVERAGE")
    source_position_by_slot = {
        source_slot: position
        for position, source_slot in enumerate(source_slots_t)
    }
    missing_source_slots = tuple(
        slot for slot in request_slots if slot not in source_position_by_slot
    )
    if missing_source_slots:
        raise RuntimeError(
            "E_SFI_CAPTURE_COHORT_TAPE_SOURCE_SLOT_COVERAGE: "
            f"missing={missing_source_slots}"
        )
    source_rows = tuple(source_position_by_slot[slot] for slot in request_slots)
    lastn1_positions = tuple(
        _non_negative_int("lastn1_row_position", value)
        for value in lastn1_row_positions
    )
    if (
        len(set(lastn1_positions)) != len(lastn1_positions)
        or tuple(sorted(lastn1_positions)) != lastn1_positions
        or any(position >= rows for position in lastn1_positions)
    ):
        raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_LASTN1_ROWS")

    # The arena is the source-of-truth owner.  Recover its affine layer view
    # and issue one strided copy instead of C layer-local stack kernels.  A
    # stamped cohort whose payloads no longer describe that arena is invalid;
    # there is deliberately no allocation or stack fallback.
    scores_source = _require_affine_cohort_view("SCORE", scores_sources)
    denoms_source = (
        _require_affine_cohort_view("DENOM", denoms_sources)
        if denoms_sources
        else None
    )
    lastn1_source: Optional[torch.Tensor] = None
    if lastn1_positions and not all(
        _same_tensor_view(lastn1, scores)
        for lastn1, scores in zip(lastn1_sources, scores_sources, strict=True)
    ):
        lastn1_source = _require_affine_cohort_view("LASTN1", lastn1_sources)
    # Validate ownership against the full tape allocations before acquiring a
    # slot ownership.  An invalid repeated publication therefore cannot leave
    # a tracker lease behind even though no device write was submitted.
    for name, source in (
        ("SCORE", scores_source),
        ("DENOM", denoms_source),
        ("LASTN1", lastn1_source),
    ):
        for tape in (state.scores, state.denoms):
            _require_source_outside_tape_storage(name, source, tape)

    # Stream identity and event allocation are deterministic preflight.  They
    # must not create tracker ownership that a rejected publication cannot
    # retire.  The event itself submits no device work until record().
    if state.scores.device.type == "cuda":
        if stream is None:
            raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_SNAPSHOT_STREAM")
        if capture_cohort_stream_identity(stream) != tracker.consumer_stream_identity:
            raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_SNAPSHOT_STREAM_DRIFT")
        copy_completion_event = torch.cuda.Event(enable_timing=False)
    elif copy_completion_event is None:
        copy_completion_event = object()

    grant = tracker.begin_snapshot(
        owner_key=owner,
        layer_slots=layers,
        expected_group_size=len(layers),
        consumer_tokens=tokens,
    )
    submission_started = False
    try:
        tracker.mark_snapshot_submission_started(grant=grant)
        submission_started = True
        if state.scores.device.type == "cuda":
            assert stream is not None
            for completion_event in completion_events:
                stream.wait_event(completion_event)
            with torch.cuda.stream(stream):
                destination_scores = state.scores[
                    layers[0] : layers[0] + len(layers),
                    :,
                    :heads,
                    :1,
                    :,
                ]
                destination_denoms = state.denoms[
                    layers[0] : layers[0] + len(layers),
                    :,
                    :heads,
                ]
                _copy_affine_cohort_slot_rows(
                    destination_scores=destination_scores,
                    destination_denoms=destination_denoms,
                    scores_source=scores_source,
                    denoms_source=denoms_source,
                    lastn1_source=lastn1_source,
                    source_rows=source_rows,
                    destination_slots=request_slots,
                    lastn1_row_positions=lastn1_positions,
                )
                copy_completion_event.record(stream)
        else:
            destination_scores = state.scores[
                layers[0] : layers[0] + len(layers),
                :,
                :heads,
                :1,
                :,
            ]
            destination_denoms = state.denoms[
                layers[0] : layers[0] + len(layers),
                :,
                :heads,
            ]
            _copy_affine_cohort_slot_rows(
                destination_scores=destination_scores,
                destination_denoms=destination_denoms,
                scores_source=scores_source,
                denoms_source=denoms_source,
                lastn1_source=lastn1_source,
                source_rows=source_rows,
                destination_slots=request_slots,
                lastn1_row_positions=lastn1_positions,
            )
        for payload in payloads_t:
            setattr(payload, "lastn1_capture_scores", None)
            # The copy event is now the sole source boundary.  Keep immutable
            # handle/layer provenance on the payload, but retire the executable job
            # owner so the cross-step producer cannot repeat lifecycle validation or
            # wait on an event already subsumed by its source-ready event.
            setattr(payload, "capture_postprocess_job", None)
            setattr(payload, "cohort_tape_cohort_size", report.cohort_size)
            setattr(payload, "cohort_tape_owner_key", owner)
            setattr(payload, "cohort_tape_scores_base", state.scores)
            setattr(payload, "cohort_tape_denoms_base", state.denoms)
            setattr(payload, "cohort_tape_consumer_tokens", tokens)
        # Commit is the final publication step.  Until every per-request clone
        # releases its token, none of these stable rows can be reused.
        tracker.commit_snapshot(
            grant=grant,
            consumer_tokens=tokens,
        )
        return CaptureCohortSnapshot(
            layer_slots=layers,
            copy_completion_event=copy_completion_event,
        )
    except BaseException as exc:
        if submission_started:
            tracker.poison_snapshot(
                grant=grant,
                reason=f"{type(exc).__name__}: {exc}",
            )
        else:
            tracker.abort_snapshot(grant=grant)
        raise


def resolve_capture_cohort_tape_group(
    payloads: Sequence[object],
) -> Optional[CaptureCohortTapeGroup]:
    payloads_t = tuple(payloads)
    if not payloads_t:
        return None
    owner_keys = tuple(
        getattr(payload, "cohort_tape_owner_key", None) for payload in payloads_t
    )
    has_owner = tuple(key is not None for key in owner_keys)
    if not any(has_owner):
        # owner_key is the only presence marker.  Any other tape field without
        # it is a partial publication and must not masquerade as the ring path.
        if any(
            int(getattr(payload, "cohort_tape_cohort_size", 0) or 0) != 0
            or getattr(payload, "cohort_tape_scores_base", None) is not None
            or getattr(payload, "cohort_tape_denoms_base", None) is not None
            or bool(
                getattr(payload, "cohort_tape_consumer_tokens", tuple())
                or tuple()
            )
            for payload in payloads_t
        ):
            raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_PARTIAL_PUBLICATION")
        return None
    if not all(has_owner):
        raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_MIXED_GROUP")
    owner = _validate_owner_key(owner_keys[0])
    if any(_validate_owner_key(key) != owner for key in owner_keys[1:]):
        raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_OWNER_DRIFT")
    layer_slots = tuple(
        int(getattr(getattr(payload, "state", None), "layer_index", -1))
        for payload in payloads_t
    )
    cohort_sizes = tuple(
        int(getattr(payload, "cohort_tape_cohort_size", -1)) for payload in payloads_t
    )
    if len(set(cohort_sizes)) != 1:
        raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_GROUP_DRIFT")
    group_id = owner[-1]
    cohort_size = _positive_int("cohort_tape_cohort_size", cohort_sizes[0])
    if any(layer_slot < 0 for layer_slot in layer_slots):
        raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_GLOBAL_LAYER_MISSING")
    if layer_slots != tuple(
        range(layer_slots[0], layer_slots[0] + len(layer_slots))
    ):
        raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_NONCONTIGUOUS_GROUP")
    if (
        len(payloads_t) > cohort_size
        or layer_slots[0] < group_id * cohort_size
        or layer_slots[-1] // cohort_size != group_id
    ):
        raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_CROSS_GROUP")

    consumer_tokens = payload_group_consumer_tokens(payloads_t)
    if len(consumer_tokens) != 1:
        raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_CONSUMER_GROUP_DRIFT")
    _, request_slot, _ = _consumer_token_fields(
        consumer_tokens[0], expected_owner=owner
    )
    scores_bases = tuple(
        getattr(payload, "cohort_tape_scores_base", None) for payload in payloads_t
    )
    denoms_bases = tuple(
        getattr(payload, "cohort_tape_denoms_base", None) for payload in payloads_t
    )
    scores_base = scores_bases[0]
    denoms_base = denoms_bases[0]
    if (
        not isinstance(scores_base, torch.Tensor)
        or not isinstance(denoms_base, torch.Tensor)
        or any(base is not scores_base for base in scores_bases[1:])
        or any(base is not denoms_base for base in denoms_bases[1:])
        or scores_base.dim() != 5
        or denoms_base.dim() != 3
        or scores_base.dtype != torch.float16
        or denoms_base.dtype != torch.float32
        or scores_base.device != denoms_base.device
        or layer_slots[-1] >= int(scores_base.shape[0])
        or layer_slots[-1] >= int(denoms_base.shape[0])
        or request_slot >= int(scores_base.shape[1])
        or request_slot >= int(denoms_base.shape[1])
        or int(scores_base.shape[2]) != int(denoms_base.shape[2])
        or int(scores_base.shape[3]) != 1
    ):
        raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_BASE_DRIFT")
    first_scores = getattr(payloads_t[0], "capture_scores", None)
    first_denoms = getattr(payloads_t[0], "log_f_denoms", None)
    if (
        not isinstance(first_scores, torch.Tensor)
        or first_scores.dim() != 4
    ):
        raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_VIEW_MISSING")
    rows, heads, window, logical_k = tuple(
        int(value) for value in first_scores.shape
    )
    has_denoms = first_denoms is not None
    if window != 1 or (
        has_denoms
        and (
            not isinstance(first_denoms, torch.Tensor)
            or first_denoms.dim() != 2
            or tuple(first_denoms.shape) != (rows, heads)
        )
    ):
        raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_VIEW_SHAPE")
    if rows != 1:
        raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_REQUEST_VIEW_ROWS")
    for payload, layer_slot in zip(payloads_t, layer_slots, strict=True):
        scores = getattr(payload, "capture_scores", None)
        denoms = getattr(payload, "log_f_denoms", None)
        expected_scores = scores_base[
            layer_slot,
            request_slot : request_slot + 1,
            :heads,
            :1,
            :logical_k,
        ]
        expected_denoms = denoms_base[
            layer_slot,
            request_slot : request_slot + 1,
            :heads,
        ]
        if (
            not isinstance(scores, torch.Tensor)
            or tuple(scores.shape) != (rows, heads, 1, logical_k)
            or scores.data_ptr() != expected_scores.data_ptr()
            or tuple(scores.stride()) != tuple(expected_scores.stride())
        ):
            raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_VIEW_DRIFT")
        if has_denoms:
            if (
                not isinstance(denoms, torch.Tensor)
                or tuple(denoms.shape) != (rows, heads)
                or denoms.data_ptr() != expected_denoms.data_ptr()
                or tuple(denoms.stride()) != tuple(expected_denoms.stride())
            ):
                raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_DENOM_VIEW_DRIFT")
        elif denoms is not None:
            raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_MIXED_DENOM_GROUP")
    return CaptureCohortTapeGroup(
        request_slot=request_slot,
        owner_key=owner,
        layer_slots=layer_slots,
        scores_base=scores_base,
        denoms_base=denoms_base,
        num_query_heads=heads,
        logical_k=logical_k,
        has_denoms=has_denoms,
    )


def payload_group_consumer_tokens(payloads: Sequence[object]) -> tuple[Hashable, ...]:
    payloads_t = tuple(payloads)
    if not payloads_t:
        return tuple()
    token_groups = tuple(
        tuple(getattr(payload, "cohort_tape_consumer_tokens", tuple()) or tuple())
        for payload in payloads_t
    )
    expected = token_groups[0]
    if any(tokens != expected for tokens in token_groups[1:]):
        raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_CONSUMER_GROUP_DRIFT")
    return _unique_hashable_tokens("CONSUMER", expected, allow_empty=True)


def require_capture_cohort_selector_view(
    payloads: Sequence[object],
) -> Optional[CaptureCohortTapeGroup]:
    """Prove that the CUDA selector will not materialize an implicit copy."""

    group = resolve_capture_cohort_tape_group(payloads)
    if group is None:
        return None
    scores = group.scores
    denoms = group.denoms
    if (
        int(scores.shape[-1]) != int(group.scores_base.shape[-1])
        or not scores.is_contiguous()
        or (denoms is not None and not denoms.is_contiguous())
    ):
        raise RuntimeError(
            "E_SFI_CAPTURE_COHORT_SELECTOR_VIEW: selector input must be one "
            "request, full fixed-K, and contiguous"
        )
    return group


def release_capture_cohort_payload_group(
    *,
    controller: object,
    payloads: Sequence[object],
    consumer_stream: torch.cuda.Stream,
) -> bool:
    group = resolve_capture_cohort_tape_group(payloads)
    if group is None:
        return False
    tracker = getattr(controller, "_capture_cohort_tape_lease_tracker", None)
    if not isinstance(tracker, CaptureCohortTapeLeaseTracker):
        raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_TRACKER_MISSING")
    tokens = payload_group_consumer_tokens(payloads)
    if not tokens:
        raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_CONSUMER_MISSING")
    tracker.release_consumers(
        owner_key=group.owner_key,
        consumer_tokens=tokens,
        consumer_stream_identity=capture_cohort_stream_identity(consumer_stream),
    )
    return True
