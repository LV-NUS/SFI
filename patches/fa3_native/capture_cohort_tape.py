from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import Any, Hashable, Optional, Sequence

import torch


CAPTURE_COHORT_TAPE_SCHEMA = "sfi.fa3_capture_cohort_tape.v3"
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
    bank_capacity: int
    layer_capacity: int
    group_count: int
    rows_capacity: int
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
    consumer_stream_identity: int
    scores: torch.Tensor
    denoms: torch.Tensor


@dataclass(frozen=True, slots=True)
class CaptureCohortTapeGroup:
    plan_signature: str
    bank_index: int
    group_id: int
    cohort_size: int
    expected_group_size: int
    owner_key: CaptureCohortTapeOwnerKey
    layer_slots: tuple[int, ...]
    scores_base: torch.Tensor
    denoms_base: torch.Tensor
    row_start: int
    rows: int
    num_query_heads: int
    logical_k: int
    has_denoms: bool

    @property
    def scores(self) -> torch.Tensor:
        first = self.layer_slots[0]
        return self.scores_base[
            self.bank_index,
            first : first + len(self.layer_slots),
            self.row_start : self.row_start + self.rows,
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
            self.bank_index,
            first : first + len(self.layer_slots),
            self.row_start : self.row_start + self.rows,
            : self.num_query_heads,
        ]


@dataclass(frozen=True, slots=True)
class CaptureCohortSnapshot:
    group: CaptureCohortTapeGroup
    copy_completion_event: Any
    consumer_tokens: tuple[Hashable, ...]


def plan_capture_cohort_tape(
    *,
    bank_capacity: int,
    layer_capacity: int,
    rows_capacity: int,
    num_query_heads: int,
    logical_k_capacity: int,
    cohort_size: int,
) -> CaptureCohortTapeReport:
    """Return the exact persistent footprint without depending on a policy stamp."""

    banks = _positive_int("bank_capacity", bank_capacity)
    layers = _positive_int("layer_capacity", layer_capacity)
    rows = _positive_int("rows_capacity", rows_capacity)
    heads = _positive_int("num_query_heads", num_query_heads)
    logical_k = _positive_int("logical_k_capacity", logical_k_capacity)
    cohort = _positive_int("cohort_size", cohort_size)
    group_count = (layers + cohort - 1) // cohort
    score_elements = banks * layers * rows * heads * logical_k
    denom_elements = banks * layers * rows * heads
    scores_bytes = score_elements * torch.tensor([], dtype=torch.float16).element_size()
    denoms_bytes = denom_elements * torch.tensor([], dtype=torch.float32).element_size()
    return CaptureCohortTapeReport(
        schema=CAPTURE_COHORT_TAPE_SCHEMA,
        bank_capacity=banks,
        layer_capacity=layers,
        group_count=group_count,
        rows_capacity=rows,
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
    layer_slots: tuple[int, ...]
    reservation_serial: int
    expected_consumers: set[Hashable]
    pending_consumers: set[Hashable]
    released_consumers: set[Hashable]
    submission_started: bool = False
    sealed: bool = False


@dataclass(slots=True)
class _BankLease:
    generation_key: CaptureCohortTapeGenerationKey
    bank_index: int
    groups: dict[int, _GroupLease]
    release_event: Any = None


@dataclass(frozen=True, slots=True)
class CaptureCohortTapeSnapshotGrant:
    owner_key: CaptureCohortTapeOwnerKey
    bank_index: int
    reservation_serial: int
    prior_release_event: Any


def _validate_owner_key(value: object) -> CaptureCohortTapeOwnerKey:
    if not isinstance(value, tuple) or len(value) != 5:
        raise ValueError("capture cohort tape owner key must have five fields")
    signature = _plan_signature(value[0])
    numeric = tuple(_non_negative_int("owner_key", item) for item in value[1:])
    return (signature, *numeric)


def _generation_key(
    owner_key: CaptureCohortTapeOwnerKey,
) -> CaptureCohortTapeGenerationKey:
    return owner_key[:4]


class CaptureCohortTapeLeaseTracker:
    """Bounded generation-bank owner for snapshot writes and deferred reads.

    A generation owns exactly one bank across all model cohorts.  A bank is
    reusable only after every model cohort has been published and every
    request-local consumer has released its lease.  Exhaustion is terminal;
    the runtime never host-waits for a future consumer or allocates a fallback.
    """

    __slots__ = (
        "_lock",
        "_report",
        "_consumer_stream_identity",
        "_banks",
        "_bank_by_generation",
        "_next_reservation_serial",
        "_prior_bank_by_reservation",
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
        self._lock = threading.Lock()
        self._report = report
        self._consumer_stream_identity = _non_negative_int(
            "consumer_stream_identity", consumer_stream_identity
        )
        self._banks: list[Optional[_BankLease]] = [None] * report.bank_capacity
        self._bank_by_generation: dict[CaptureCohortTapeGenerationKey, int] = {}
        self._next_reservation_serial = 0
        self._prior_bank_by_reservation: dict[int, Optional[_BankLease]] = {}
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
    ) -> CaptureCohortTapeSnapshotGrant:
        owner = _validate_owner_key(owner_key)
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

        generation = _generation_key(owner)
        with self._lock:
            self._raise_if_poisoned_locked()
            bank_index = self._bank_by_generation.get(generation)
            prior_event = None
            prior_bank: Optional[_BankLease] = None
            opened_generation = bank_index is None
            if bank_index is None:
                if group_id != 0:
                    raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_GROUP_ORDER")
                bank_index, prior_event, prior_bank = self._acquire_bank_locked(
                    generation
                )
            bank = self._banks[bank_index]
            if bank is None or bank.generation_key != generation:
                raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_GENERATION_DRIFT")
            if bank.groups and not bank.groups[len(bank.groups) - 1].sealed:
                raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_PENDING_SNAPSHOT")
            if group_id in bank.groups:
                raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_DUPLICATE_SNAPSHOT")
            if group_id != len(bank.groups):
                raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_GROUP_ORDER")
            self._next_reservation_serial += 1
            reservation_serial = self._next_reservation_serial
            bank.groups[group_id] = _GroupLease(
                owner_key=owner,
                layer_slots=slots,
                reservation_serial=reservation_serial,
                expected_consumers=set(),
                pending_consumers=set(),
                released_consumers=set(),
            )
            if opened_generation:
                self._prior_bank_by_reservation[reservation_serial] = prior_bank
            return CaptureCohortTapeSnapshotGrant(
                owner_key=owner,
                bank_index=bank_index,
                reservation_serial=reservation_serial,
                prior_release_event=prior_event,
            )

    def mark_snapshot_submission_started(
        self,
        *,
        grant: CaptureCohortTapeSnapshotGrant,
    ) -> None:
        with self._lock:
            self._raise_if_poisoned_locked()
            _, group = self._require_grant_locked(grant)
            if group.sealed:
                raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_SUBMIT_AFTER_COMMIT")
            if group.submission_started:
                raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_DUPLICATE_SUBMISSION")
            group.submission_started = True

    def commit_snapshot(
        self,
        *,
        grant: CaptureCohortTapeSnapshotGrant,
        consumer_tokens: Sequence[Hashable],
        snapshot_completion_event: Any,
    ) -> Any:
        tokens = _unique_hashable_tokens(
            "CONSUMER", consumer_tokens, allow_empty=True
        )
        if len(tokens) > self._report.rows_capacity:
            raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_CONSUMER_CAPACITY")
        if snapshot_completion_event is None:
            raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_SNAPSHOT_EVENT")
        with self._lock:
            self._raise_if_poisoned_locked()
            bank, group = self._require_grant_locked(grant)
            if group.sealed:
                raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_DUPLICATE_COMMIT")
            if not group.submission_started:
                raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_COMMIT_BEFORE_SUBMIT")
            if (
                group.expected_consumers
                or group.pending_consumers
                or group.released_consumers
            ):
                raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_COMMIT_STATE_DRIFT")
            token_set = set(tokens)
            group.expected_consumers.update(token_set)
            group.pending_consumers.update(token_set)
            group.sealed = True
            if not group.pending_consumers:
                if self._bank_complete_locked(bank):
                    bank.release_event = snapshot_completion_event
            self._prior_bank_by_reservation.pop(group.reservation_serial, None)
            return bank.release_event

    def abort_snapshot(
        self,
        *,
        grant: CaptureCohortTapeSnapshotGrant,
    ) -> None:
        """Rollback one untouched tail reservation without hiding submitted work."""

        with self._lock:
            self._raise_if_poisoned_locked()
            bank, group = self._require_grant_locked(grant)
            if group.sealed:
                raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_ABORT_AFTER_COMMIT")
            if group.submission_started:
                raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_ABORT_AFTER_SUBMIT")
            group_id = grant.owner_key[-1]
            if group_id != len(bank.groups) - 1:
                raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_ABORT_NONTAIL")
            del bank.groups[group_id]
            prior_bank_present = (
                grant.reservation_serial in self._prior_bank_by_reservation
            )
            prior_bank = self._prior_bank_by_reservation.pop(
                grant.reservation_serial, None
            )
            if bank.groups:
                if prior_bank_present:
                    raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_ABORT_STATE_DRIFT")
                return
            generation = bank.generation_key
            self._bank_by_generation.pop(generation, None)
            if prior_bank_present and prior_bank is not None:
                self._banks[bank.bank_index] = prior_bank
                self._bank_by_generation[prior_bank.generation_key] = bank.bank_index
            else:
                self._banks[bank.bank_index] = None

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
        completion_event: Any,
        consumer_stream_identity: int,
    ) -> bool:
        owner = _validate_owner_key(owner_key)
        tokens = _unique_hashable_tokens(
            "RELEASE", consumer_tokens, allow_empty=False
        )
        if completion_event is None:
            raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_RELEASE_EVENT")
        if (
            _non_negative_int("consumer_stream_identity", consumer_stream_identity)
            != self._consumer_stream_identity
        ):
            raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_RELEASE_STREAM_DRIFT")
        with self._lock:
            self._raise_if_poisoned_locked()
            bank, group = self._require_group_locked(owner)
            if not group.sealed:
                raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_RELEASE_BEFORE_SEAL")
            token_set = set(tokens)
            if not token_set.issubset(group.expected_consumers):
                raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_UNEXPECTED_RELEASE")
            if group.released_consumers.intersection(token_set):
                raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_DUPLICATE_RELEASE")
            if not token_set.issubset(group.pending_consumers):
                raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_MISSING_RELEASE")
            group.pending_consumers.difference_update(token_set)
            group.released_consumers.update(token_set)
            if not self._bank_complete_locked(bank):
                return False
            bank.release_event = completion_event
            return True

    def bank_index_for_owner(
        self, owner_key: CaptureCohortTapeOwnerKey
    ) -> int:
        owner = _validate_owner_key(owner_key)
        with self._lock:
            self._raise_if_poisoned_locked()
            bank, _ = self._require_group_locked(owner)
            return bank.bank_index

    def _acquire_bank_locked(
        self,
        generation: CaptureCohortTapeGenerationKey,
    ) -> tuple[int, Any, Optional[_BankLease]]:
        for bank_index, bank in enumerate(self._banks):
            if bank is None:
                self._banks[bank_index] = _BankLease(
                    generation_key=generation,
                    bank_index=bank_index,
                    groups={},
                )
                self._bank_by_generation[generation] = bank_index
                return bank_index, None, None
            if self._bank_complete_locked(bank):
                if bank.release_event is None:
                    raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_RELEASE_MISSING")
                prior_event = bank.release_event
                self._bank_by_generation.pop(bank.generation_key, None)
                self._banks[bank_index] = _BankLease(
                    generation_key=generation,
                    bank_index=bank_index,
                    groups={},
                )
                self._bank_by_generation[generation] = bank_index
                return bank_index, prior_event, bank
        raise RuntimeError(
            "E_SFI_CAPTURE_COHORT_TAPE_BANKS_EXHAUSTED: every bounded generation "
            "bank still has an unreleased deferred consumer"
        )

    def _require_group_locked(
        self,
        owner_key: CaptureCohortTapeOwnerKey,
    ) -> tuple[_BankLease, _GroupLease]:
        bank_index = self._bank_by_generation.get(_generation_key(owner_key))
        if bank_index is None:
            raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_OWNER_DRIFT")
        bank = self._banks[bank_index]
        if bank is None or bank.generation_key != _generation_key(owner_key):
            raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_OWNER_DRIFT")
        group = bank.groups.get(owner_key[-1])
        if group is None or group.owner_key != owner_key:
            raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_OWNER_DRIFT")
        return bank, group

    def _require_grant_locked(
        self,
        grant: CaptureCohortTapeSnapshotGrant,
    ) -> tuple[_BankLease, _GroupLease]:
        if not isinstance(grant, CaptureCohortTapeSnapshotGrant):
            raise TypeError("capture cohort tape requires a snapshot grant")
        owner = _validate_owner_key(grant.owner_key)
        bank, group = self._require_group_locked(owner)
        if (
            bank.bank_index != _non_negative_int("bank_index", grant.bank_index)
            or group.reservation_serial
            != _positive_int("reservation_serial", grant.reservation_serial)
        ):
            raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_RESERVATION_DRIFT")
        return bank, group

    def _raise_if_poisoned_locked(self) -> None:
        if self._poison_reason:
            raise RuntimeError(
                "E_SFI_CAPTURE_COHORT_TAPE_TRACKER_POISONED: "
                f"{self._poison_reason}"
            )

    def _bank_complete_locked(self, bank: _BankLease) -> bool:
        if len(bank.groups) != self._report.group_count:
            return False
        return all(group.sealed and not group.pending_consumers for group in bank.groups.values())


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
    signature = _plan_signature(plan_signature)
    stream_identity = capture_cohort_stream_identity(consumer_stream)
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("capture cohort tape prebuild requires CUDA")
    prior = getattr(controller, "_capture_cohort_tape_state", None)
    if prior is not None:
        if (
            not isinstance(prior, CaptureCohortTapeState)
            or prior.report != report
            or prior.plan_signature != signature
            or prior.consumer_stream_identity != stream_identity
        ):
            raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_PREBUILD_DRIFT")
        return prior
    # Physical request-major storage makes any one-request, contiguous-layer
    # consumer view dense without another D2D materialization.  The published
    # logical view remains [bank, layer, row, head, window, k].
    scores = torch.empty(
        (
            report.bank_capacity,
            report.rows_capacity,
            report.layer_capacity,
            report.num_query_heads,
            1,
            report.logical_k_capacity,
        ),
        dtype=torch.float16,
        device=device,
    ).permute(0, 2, 1, 3, 4, 5)
    denoms = torch.empty(
        (
            report.bank_capacity,
            report.rows_capacity,
            report.layer_capacity,
            report.num_query_heads,
        ),
        dtype=torch.float32,
        device=device,
    ).permute(0, 2, 1, 3)
    state = CaptureCohortTapeState(
        report=report,
        plan_signature=signature,
        consumer_stream_identity=stream_identity,
        scores=scores,
        denoms=denoms,
    )
    setattr(controller, "_capture_cohort_tape_state", state)
    setattr(controller, "_capture_cohort_tape_report", report)
    setattr(
        controller,
        "_capture_cohort_tape_lease_tracker",
        CaptureCohortTapeLeaseTracker(
            report,
            consumer_stream_identity=stream_identity,
        ),
    )
    return state


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


def _contiguous_row_runs(positions: Sequence[int]) -> tuple[tuple[int, int], ...]:
    positions_t = tuple(positions)
    if not positions_t:
        return tuple()
    runs: list[tuple[int, int]] = []
    start = positions_t[0]
    stop = start + 1
    for position in positions_t[1:]:
        if position == stop:
            stop += 1
            continue
        runs.append((start, stop))
        start = position
        stop = position + 1
    runs.append((start, stop))
    return tuple(runs)


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


def _copy_affine_cohort_views(
    *,
    destination_scores: torch.Tensor,
    destination_denoms: torch.Tensor,
    scores_source: torch.Tensor,
    denoms_source: Optional[torch.Tensor],
    lastn1_source: Optional[torch.Tensor],
    lastn1_row_runs: Sequence[tuple[int, int]],
) -> None:
    # A source may never be another view of the generation tape.  Reject the
    # whole publication before its first write so a repeated/incorrect owner
    # cannot turn copy_ into an overlapping operation or partially update the
    # destination before a later plane discovers the drift.
    sources = (
        ("SCORE", scores_source),
        ("DENOM", denoms_source),
        ("LASTN1", lastn1_source),
    )
    destinations = (destination_scores, destination_denoms)
    for name, source in sources:
        for destination in destinations:
            _require_source_outside_tape_storage(name, source, destination)
    destination_scores.copy_(scores_source)
    if denoms_source is not None:
        destination_denoms.copy_(denoms_source)
    if lastn1_source is not None:
        for row_start, row_stop in lastn1_row_runs:
            destination_scores[:, row_start:row_stop].copy_(
                lastn1_source[:, row_start:row_stop]
            )


def snapshot_capture_cohort_payload_group(
    *,
    state: CaptureCohortTapeState,
    tracker: CaptureCohortTapeLeaseTracker,
    payloads: Sequence[object],
    consumer_tokens: Sequence[Hashable],
    lastn1_row_positions: Sequence[int] = tuple(),
    stream: Optional[torch.cuda.Stream] = None,
    copy_completion_event: Any = None,
) -> CaptureCohortSnapshot:
    """Copy one finalized arena cohort into a fixed generation-bank view."""

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
        "CONSUMER", consumer_tokens, allow_empty=True
    )
    if len(tokens) > state.report.rows_capacity:
        raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_CONSUMER_CAPACITY")

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
        or rows > report.rows_capacity
        or heads != report.num_query_heads
        or logical_k != report.logical_k_capacity
    ):
        raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_FIXED_K_DRIFT")
    if denoms_sources and len(denoms_sources) != len(payloads_t):
        raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_MIXED_DENOM_GROUP")
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
    lastn1_row_runs = _contiguous_row_runs(lastn1_positions)

    # Validate ownership against the full tape allocations before acquiring a
    # generation bank.  An invalid repeated publication therefore cannot leave
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
    if tracker.consumer_stream_identity != state.consumer_stream_identity:
        raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_TRACKER_STREAM_DRIFT")
    if state.scores.device.type == "cuda":
        if stream is None:
            raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_SNAPSHOT_STREAM")
        if capture_cohort_stream_identity(stream) != state.consumer_stream_identity:
            raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_SNAPSHOT_STREAM_DRIFT")
        copy_completion_event = torch.cuda.Event(enable_timing=False)
    elif copy_completion_event is None:
        copy_completion_event = object()

    grant = tracker.begin_snapshot(
        owner_key=owner,
        layer_slots=layers,
        expected_group_size=len(layers),
    )
    submission_started = False
    try:
        if state.scores.device.type != "cuda" and grant.prior_release_event is not None:
            raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_CPU_REUSE_EVENT")
        tracker.mark_snapshot_submission_started(grant=grant)
        submission_started = True
        if state.scores.device.type == "cuda":
            assert stream is not None
            if grant.prior_release_event is not None:
                stream.wait_event(grant.prior_release_event)
            for completion_event in completion_events:
                stream.wait_event(completion_event)
            with torch.cuda.stream(stream):
                destination_scores = state.scores[
                    grant.bank_index,
                    layers[0] : layers[0] + len(layers),
                    :rows,
                    :heads,
                    :1,
                    :,
                ]
                destination_denoms = state.denoms[
                    grant.bank_index,
                    layers[0] : layers[0] + len(layers),
                    :rows,
                    :heads,
                ]
                _copy_affine_cohort_views(
                    destination_scores=destination_scores,
                    destination_denoms=destination_denoms,
                    scores_source=scores_source,
                    denoms_source=denoms_source,
                    lastn1_source=lastn1_source,
                    lastn1_row_runs=lastn1_row_runs,
                )
                copy_completion_event.record(stream)
        else:
            destination_scores = state.scores[
                grant.bank_index,
                layers[0] : layers[0] + len(layers),
                :rows,
                :heads,
                :1,
                :,
            ]
            destination_denoms = state.denoms[
                grant.bank_index,
                layers[0] : layers[0] + len(layers),
                :rows,
                :heads,
            ]
            _copy_affine_cohort_views(
                destination_scores=destination_scores,
                destination_denoms=destination_denoms,
                scores_source=scores_source,
                denoms_source=denoms_source,
                lastn1_source=lastn1_source,
                lastn1_row_runs=lastn1_row_runs,
            )
        group_id = owner[-1]
        expected_group_size = len(layers)
        for layer_offset, payload in enumerate(payloads_t):
            layer = layers[layer_offset]
            setattr(payload, "capture_scores", destination_scores[layer_offset])
            setattr(
                payload,
                "log_f_denoms",
                destination_denoms[layer_offset] if denoms_sources else None,
            )
            setattr(payload, "lastn1_capture_scores", None)
            # The copy event is now the sole source boundary.  Keep immutable
            # handle/layer provenance on the payload, but retire the executable job
            # owner so the cross-step producer cannot repeat lifecycle validation or
            # wait on an event already subsumed by its source-ready event.
            setattr(payload, "capture_postprocess_job", None)
            setattr(payload, "cohort_tape_plan_signature", state.plan_signature)
            setattr(payload, "cohort_tape_bank", grant.bank_index)
            setattr(payload, "cohort_tape_slot", layer)
            setattr(payload, "cohort_tape_lane", group_id)
            setattr(payload, "cohort_tape_cohort_size", report.cohort_size)
            setattr(payload, "cohort_tape_expected_group_size", expected_group_size)
            setattr(payload, "cohort_tape_row_start", 0)
            setattr(payload, "cohort_tape_owner_key", owner)
            setattr(payload, "cohort_tape_scores_base", state.scores)
            setattr(payload, "cohort_tape_denoms_base", state.denoms)
            setattr(payload, "cohort_tape_consumer_tokens", tokens)
        group = resolve_capture_cohort_tape_group(payloads_t)
        if group is None:
            raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_PUBLICATION_MISSING")
        # Commit is the final publication step: zero-consumer generations must
        # not become reusable while their payload views are still being stamped.
        tracker.commit_snapshot(
            grant=grant,
            consumer_tokens=tokens,
            snapshot_completion_event=copy_completion_event,
        )
        return CaptureCohortSnapshot(
            group=group,
            copy_completion_event=copy_completion_event,
            consumer_tokens=tokens,
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
    signatures = tuple(
        str(getattr(payload, "cohort_tape_plan_signature", "") or "")
        for payload in payloads_t
    )
    if not any(signatures):
        return None
    if any(not signature for signature in signatures) or len(set(signatures)) != 1:
        raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_MIXED_GROUP")
    signature = _plan_signature(signatures[0])
    banks = tuple(int(getattr(payload, "cohort_tape_bank", -1)) for payload in payloads_t)
    slots = tuple(int(getattr(payload, "cohort_tape_slot", -1)) for payload in payloads_t)
    group_ids = tuple(int(getattr(payload, "cohort_tape_lane", -1)) for payload in payloads_t)
    cohort_sizes = tuple(
        int(getattr(payload, "cohort_tape_cohort_size", -1)) for payload in payloads_t
    )
    expected_sizes = tuple(
        int(getattr(payload, "cohort_tape_expected_group_size", -1))
        for payload in payloads_t
    )
    row_starts = tuple(
        int(getattr(payload, "cohort_tape_row_start", -1))
        for payload in payloads_t
    )
    if any(
        len(set(values)) != 1
        for values in (banks, group_ids, cohort_sizes, expected_sizes, row_starts)
    ):
        raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_GROUP_DRIFT")
    bank = _non_negative_int("cohort_tape_bank", banks[0])
    group_id = _non_negative_int("cohort_tape_lane", group_ids[0])
    cohort_size = _positive_int("cohort_tape_cohort_size", cohort_sizes[0])
    expected_size = _positive_int(
        "cohort_tape_expected_group_size", expected_sizes[0]
    )
    row_start = _non_negative_int("cohort_tape_row_start", row_starts[0])
    if slots != tuple(range(slots[0], slots[0] + len(slots))):
        raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_NONCONTIGUOUS_GROUP")
    if (
        len(payloads_t) > expected_size
        or slots[0] < group_id * cohort_size
        or slots[-1] // cohort_size != group_id
    ):
        raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_CROSS_GROUP")

    owner_keys = tuple(
        getattr(payload, "cohort_tape_owner_key", None) for payload in payloads_t
    )
    owner = _validate_owner_key(owner_keys[0])
    if owner[-1] != group_id or any(
        _validate_owner_key(key) != owner for key in owner_keys[1:]
    ):
        raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_OWNER_DRIFT")
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
        or scores_base.dim() != 6
        or denoms_base.dim() != 4
        or scores_base.dtype != torch.float16
        or denoms_base.dtype != torch.float32
        or scores_base.device != denoms_base.device
        or bank >= int(scores_base.shape[0])
        or bank >= int(denoms_base.shape[0])
        or row_start >= int(scores_base.shape[2])
    ):
        raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_BASE_DRIFT")
    first_scores = getattr(payloads_t[0], "capture_scores", None)
    first_denoms = getattr(payloads_t[0], "log_f_denoms", None)
    if (
        not isinstance(first_scores, torch.Tensor)
        or first_scores.dim() != 4
    ):
        raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_VIEW_MISSING")
    rows, heads, window, logical_k = tuple(int(value) for value in first_scores.shape)
    if row_start + rows > int(scores_base.shape[2]):
        raise RuntimeError("E_SFI_CAPTURE_COHORT_TAPE_ROW_CAPACITY")
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
    for payload, slot in zip(payloads_t, slots, strict=True):
        scores = getattr(payload, "capture_scores", None)
        denoms = getattr(payload, "log_f_denoms", None)
        expected_scores = scores_base[
            bank,
            slot,
            row_start : row_start + rows,
            :heads,
            :1,
            :logical_k,
        ]
        expected_denoms = denoms_base[
            bank,
            slot,
            row_start : row_start + rows,
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
        plan_signature=signature,
        bank_index=bank,
        group_id=group_id,
        cohort_size=cohort_size,
        expected_group_size=expected_size,
        owner_key=owner,
        layer_slots=slots,
        scores_base=scores_base,
        denoms_base=denoms_base,
        row_start=row_start,
        rows=rows,
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
        group.rows != 1
        or int(scores.shape[-1]) != int(group.scores_base.shape[-1])
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
    completion_event: Any,
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
        completion_event=completion_event,
        consumer_stream_identity=capture_cohort_stream_identity(consumer_stream),
    )
    return True
