from __future__ import annotations

from dataclasses import dataclass, field, fields as dataclass_fields
from typing import Any, Dict, Mapping, MutableSequence, Sequence, Tuple


@dataclass(slots=True)
class RefreshProducerCarrier:
    """Reusable references needed to publish one refresh producer work item."""

    payloads: Sequence[Any] = ()
    first_payload: Any | None = None
    bootstrap_slots_by_layer: list[Any] = field(default_factory=list)
    layer_indices: list[int] = field(default_factory=list)
    capture_handle_id: int = -1
    capture_handle_generation: int = -1
    capture_epoch: int = -1
    target_selected_scope_key: Any = None
    chunk_id: int = -1
    buf_id: int = -1
    pending_buf_ids: Tuple[int, ...] = tuple()
    req_ids: Tuple[str, ...] = tuple()
    target_layer_start: int = -1
    target_layer_end: int = -1
    deadline_slack_steps: int = -1


@dataclass(frozen=True)
class RefreshProducerWorkItem:
    target_layer_start: int
    target_layer_end: int
    decode_step_min: int
    decode_step_max: int
    ready_epoch: int
    deadline_epoch: int
    deadline_handle_id: int
    deadline_slack_steps: int
    can_drop: bool
    can_coalesce: bool
    admission_reason: str

    def apply_to_profile(self, profile: Any) -> None:
        profile.producer_work_target_layer_start = int(self.target_layer_start)
        profile.producer_work_target_layer_end = int(self.target_layer_end)
        profile.producer_work_decode_step_min = int(self.decode_step_min)
        profile.producer_work_decode_step_max = int(self.decode_step_max)
        profile.producer_work_ready_epoch = int(self.ready_epoch)
        profile.producer_work_deadline_epoch = int(self.deadline_epoch)
        profile.producer_work_deadline_handle_id = int(self.deadline_handle_id)
        profile.producer_work_deadline_slack_steps = int(self.deadline_slack_steps)
        profile.producer_work_can_drop = 1 if bool(self.can_drop) else 0
        profile.producer_work_can_coalesce = 1 if bool(self.can_coalesce) else 0
        profile.producer_work_admission_reason = str(self.admission_reason or "")


class RefreshProducerWorkspace:
    """Reusable per-controller scratch for refresh producer orchestration."""

    def __init__(self) -> None:
        self._prefill_payloads: list[Any] = []
        self._refresh_payloads: list[Any] = []
        self._replay_refresh_payloads: list[Any] = []
        self._refresh_bootstrap_slots_by_layer: list[Any] = []
        self._refresh_layer_indices: list[int] = []
        self._refresh_carrier = RefreshProducerCarrier(
            payloads=self._refresh_payloads,
            bootstrap_slots_by_layer=self._refresh_bootstrap_slots_by_layer,
            layer_indices=self._refresh_layer_indices,
        )

    def drain_payload_bucket(
        self,
        *,
        kind: str,
        bucket: MutableSequence[Any],
        mask: int,
        size: int,
    ) -> tuple[list[Any], int]:
        """Move the active prefix of a chunk bucket into a reusable payload list."""
        if kind == "prefill":
            payloads = self._prefill_payloads
        elif kind == "refresh":
            payloads = self._refresh_payloads
        else:
            raise ValueError(f"unknown producer payload kind: {kind}")

        payloads.clear()
        limit = max(0, min(int(size), len(bucket)))
        next_mask = int(mask)
        for slot_in_chunk in range(limit):
            payload = bucket[slot_in_chunk]
            bucket[slot_in_chunk] = None
            if payload is None:
                continue
            next_mask &= ~(1 << slot_in_chunk)
            payloads.append(payload)
        return payloads, next_mask

    def drain_payload_buckets(
        self,
        *,
        kind: str,
        buckets: Sequence[MutableSequence[Any]],
        masks: Sequence[int],
        size: int,
    ) -> tuple[list[Any], list[int]]:
        """Move active prefixes from multiple chunk buckets into one reusable list."""
        if kind == "prefill":
            payloads = self._prefill_payloads
        elif kind == "refresh":
            payloads = self._refresh_payloads
        else:
            raise ValueError(f"unknown producer payload kind: {kind}")

        payloads.clear()
        next_masks: list[int] = []
        limit_per_bucket = max(0, int(size))
        for bucket, mask in zip(buckets, masks):
            limit = min(limit_per_bucket, len(bucket))
            next_mask = int(mask)
            for slot_in_chunk in range(limit):
                payload = bucket[slot_in_chunk]
                bucket[slot_in_chunk] = None
                if payload is None:
                    continue
                next_mask &= ~(1 << slot_in_chunk)
                payloads.append(payload)
            next_masks.append(next_mask)
        return payloads, next_masks

    def begin_replay_refresh_payloads(self) -> list[Any]:
        """Return a reusable all-layer payload list that does not occupy chunk rings."""
        self._replay_refresh_payloads.clear()
        return self._replay_refresh_payloads

    def prepare_refresh_carrier(
        self,
        payloads: Sequence[Any],
        *,
        default_buf_id: int = -1,
        map_layer: Any = None,
        normalize_req_ids: Any = None,
        deadline_slack_steps: int = -1,
    ) -> RefreshProducerCarrier:
        """Fill the reusable refresh carrier with payload-owned references."""
        bootstrap_slots_by_layer = self._refresh_bootstrap_slots_by_layer
        layer_indices = self._refresh_layer_indices
        bootstrap_slots_by_layer.clear()
        layer_indices.clear()
        pending_buf_ids_list: list[int] = []
        req_ids: Tuple[str, ...] | None = None
        capture_handle_id = -1
        capture_handle_generation = -1
        capture_epoch = -1
        target_selected_scope_key = None
        chunk_id = -1
        buf_id = -1
        validate_pending = callable(map_layer)
        for payload_index, payload in enumerate(payloads):
            bootstrap_slots_by_layer.append(getattr(payload, "bootstrap_slots", None))
            layer_index = _payload_layer_index(payload, payload_index)
            layer_indices.append(layer_index)
            if payload_index == 0:
                capture_handle_id = int(getattr(payload, "capture_handle_id", -1))
                capture_handle_generation = int(
                    getattr(payload, "capture_handle_generation", -1)
                )
                capture_epoch = int(getattr(payload, "capture_epoch", -1))
                target_selected_scope_key = getattr(
                    payload,
                    "target_selected_scope_key",
                    None,
                )
            elif validate_pending:
                if (
                    int(getattr(payload, "capture_handle_id", -1)) != capture_handle_id
                    or int(getattr(payload, "capture_handle_generation", -1))
                    != capture_handle_generation
                ):
                    raise RuntimeError(
                        "pending refresh rebuild payload handle mismatch across layers"
                    )
                if (
                    getattr(payload, "target_selected_scope_key", None)
                    != target_selected_scope_key
                ):
                    raise RuntimeError(
                        "pending refresh rebuild target scope mismatch across payloads"
                    )
            if validate_pending:
                if capture_handle_id <= 0 or capture_handle_generation <= 0:
                    raise RuntimeError(
                        "pending refresh rebuild missing capture handle identity: "
                        f"handle_id={capture_handle_id} "
                        f"generation={capture_handle_generation}"
                    )
                if layer_index < 0:
                    raise RuntimeError(
                        "pending refresh rebuild missing layer index"
                    )
                mapped_chunk, mapped_buf, _ = map_layer(int(layer_index))
                if chunk_id < 0:
                    chunk_id = int(mapped_chunk)
                    buf_id = int(mapped_buf)
                pending_buf = int(mapped_buf)
                if pending_buf not in pending_buf_ids_list:
                    pending_buf_ids_list.append(pending_buf)
                cur_req_ids = _payload_pending_request_ids(
                    payload,
                    normalize_req_ids=normalize_req_ids,
                )
                if not cur_req_ids:
                    raise RuntimeError("pending refresh rebuild missing req_ids")
                if req_ids is None:
                    req_ids = cur_req_ids
                elif cur_req_ids != req_ids:
                    raise RuntimeError(
                        "pending refresh rebuild req_ids mismatch across payloads"
                    )
        if not pending_buf_ids_list and int(default_buf_id) >= 0:
            pending_buf_ids_list.append(int(default_buf_id))
        target_layer_start = min(layer_indices) if layer_indices else -1
        target_layer_end = max(layer_indices) if layer_indices else -1

        carrier = self._refresh_carrier
        carrier.payloads = payloads
        carrier.first_payload = payloads[0] if len(payloads) > 0 else None
        carrier.capture_handle_id = int(capture_handle_id)
        carrier.capture_handle_generation = int(capture_handle_generation)
        carrier.capture_epoch = int(capture_epoch)
        carrier.target_selected_scope_key = target_selected_scope_key
        carrier.chunk_id = int(chunk_id)
        carrier.buf_id = int(buf_id)
        carrier.pending_buf_ids = tuple(int(v) for v in pending_buf_ids_list)
        carrier.req_ids = tuple(req_ids or tuple())
        carrier.target_layer_start = int(target_layer_start)
        carrier.target_layer_end = int(target_layer_end)
        carrier.deadline_slack_steps = int(deadline_slack_steps)
        return carrier


def replay_refresh_payload_groups_have_adjacent_selector_fusion(
    payload_groups: Sequence[Sequence[Any]],
) -> bool:
    """Return whether adjacent writer groups can share one selector call."""
    previous_key: tuple[Any, ...] | None = None
    for group in payload_groups:
        if not group:
            previous_key = None
            continue
        first_payload = group[0]
        key = _tensor_storage_compat_key_for_replay_group(
            getattr(first_payload, "capture_scores", None)
        )
        if key is not None and previous_key is not None and key == previous_key:
            return True
        previous_key = key
    return False


def fuse_replay_refresh_payload_groups_for_adjacent_selector_source(
    payload_groups: Sequence[Sequence[Any]],
    *,
    max_payloads_per_group: int,
) -> list[list[Any]]:
    """Fuse adjacent replay-refresh groups that share selector source storage.

    The caller chooses ``max_payloads_per_group`` to preserve the desired
    readiness boundary while avoiding unsafe cross-source fusion.
    """
    limit = max(1, int(max_payloads_per_group))
    fused: list[list[Any]] = []
    current: list[Any] = []
    current_key: tuple[Any, ...] | None = None

    def _group_key(group: Sequence[Any]) -> tuple[Any, ...] | None:
        if not group:
            return None
        return _tensor_storage_compat_key_for_replay_group(
            getattr(group[0], "capture_scores", None)
        )

    for raw_group in payload_groups:
        group = list(raw_group)
        if not group:
            continue
        key = _group_key(group)
        can_fuse = (
            current
            and key is not None
            and key == current_key
            and len(current) + len(group) <= limit
        )
        if not can_fuse:
            if current:
                fused.append(current)
            current = group
            current_key = key
            continue
        current.extend(group)
    if current:
        fused.append(current)
    return fused


def get_refresh_producer_workspace(owner: Any) -> RefreshProducerWorkspace:
    workspace = getattr(owner, "_refresh_producer_workspace", None)
    if not isinstance(workspace, RefreshProducerWorkspace):
        workspace = RefreshProducerWorkspace()
        setattr(owner, "_refresh_producer_workspace", workspace)
    return workspace


def _tensor_storage_key_for_replay_group(tensor: Any) -> tuple[Any, ...] | None:
    import torch

    if not isinstance(tensor, torch.Tensor):
        return None
    try:
        storage_ptr = int(tensor.untyped_storage().data_ptr())
    except Exception:
        storage_ptr = int(tensor.data_ptr())
    if storage_ptr == 0:
        return None
    return (
        str(tensor.device),
        str(tensor.dtype),
        int(storage_ptr),
        int(tensor.storage_offset()) if hasattr(tensor, "storage_offset") else -1,
        tuple(int(v) for v in tensor.shape),
        tuple(int(v) for v in tensor.stride()),
    )


def _tensor_storage_compat_key_for_replay_group(tensor: Any) -> tuple[Any, ...] | None:
    key = _tensor_storage_key_for_replay_group(tensor)
    if key is None:
        return None
    device, dtype, storage_ptr, _storage_offset, shape, stride = key
    return (device, dtype, storage_ptr, shape, stride)




def _can_append_replay_group_offset(offsets: Sequence[int], next_offset: int) -> bool:
    if not offsets:
        return True
    first_offset = int(offsets[0])
    next_offset = int(next_offset)
    if len(offsets) == 1:
        return next_offset > first_offset
    layer_stride = int(offsets[1]) - first_offset
    return layer_stride > 0 and next_offset - first_offset == len(offsets) * layer_stride


def partition_replay_refresh_payloads_for_direct_submit(
    payloads: Sequence[Any],
) -> list[tuple[Any, ...]]:
    """Group replay refresh payloads without stacking or copying capture tensors."""
    groups: list[tuple[Any, ...]] = []
    current: list[Any] = []
    current_storage_key: tuple[Any, ...] | None = None
    current_offsets: list[int] = []
    current_layer_indices: set[int] = set()
    for payload in payloads:
        capture_scores = getattr(payload, "capture_scores", None)
        storage_key = _tensor_storage_compat_key_for_replay_group(capture_scores)
        storage_offset = (
            int(capture_scores.storage_offset())
            if hasattr(capture_scores, "storage_offset")
            else -1
        )
        layer_index = int(getattr(payload, "layer_index", -1))
        if (
            current
            and (
                storage_key is None
                or storage_key != current_storage_key
                or layer_index < 0
                or layer_index in current_layer_indices
                or not _can_append_replay_group_offset(current_offsets, storage_offset)
            )
        ):
            groups.append(tuple(current))
            current = []
            current_offsets = []
            current_layer_indices = set()
        current.append(payload)
        current_storage_key = storage_key
        if storage_offset >= 0:
            current_offsets.append(storage_offset)
        if layer_index >= 0:
            current_layer_indices.add(layer_index)
    if current:
        groups.append(tuple(current))
    return groups


def _payload_layer_index(payload: Any, payload_index: int) -> int:
    try:
        stagger_layer_index = int(getattr(payload, "stagger_layer_index", -1))
    except (TypeError, ValueError):
        stagger_layer_index = -1
    if stagger_layer_index >= 0:
        return stagger_layer_index
    layer_index = int(getattr(payload, "layer_index", -1))
    if layer_index < 0:
        layer_index = int(getattr(getattr(payload, "state", None), "layer_index", -1))
    return layer_index if layer_index >= 0 else int(payload_index)


def _payload_request_ids(payload: Any) -> Tuple[str, ...]:
    req_ids: list[str] = []

    def _append(raw: Any) -> None:
        req_id = str(raw or "")
        if req_id and req_id not in req_ids:
            req_ids.append(req_id)

    for req_id in getattr(payload, "refresh_intent_req_ids", ()) or ():
        _append(req_id)
    if req_ids:
        return tuple(req_ids)

    # [DETERMINISTIC-REQIDS-SNAPSHOT 2026-07-03] slot_req_ids 恒于入队口提交步
    # 物化;live 的 state.batch_request_ids 合并兜底已退休(deferred 晚读脏值)。
    for req_id in getattr(payload, "slot_req_ids", ()) or ():
        _append(req_id)
    return tuple(req_ids)


def _normalize_request_ids(raw_req_ids: Any) -> Tuple[str, ...]:
    req_ids: list[str] = []
    for req_id in raw_req_ids or ():
        rid = str(req_id or "")
        if rid and rid not in req_ids:
            req_ids.append(rid)
    return tuple(req_ids)


def _payload_pending_request_ids(
    payload: Any,
    *,
    normalize_req_ids: Any = None,
) -> Tuple[str, ...]:
    # [DETERMINISTIC-REQIDS-SNAPSHOT 2026-07-03] slot_req_ids 已在
    # _enqueue_refresh_capture 提交步物化;此处为 deferred 晚读点,禁止 fallback
    # 读 live 的 state.batch_request_ids(晚读时 batch 成员可能已变,与
    # payload.slot_list 不再自洽)。缺失即 fail-fast。
    cur_req_ids = getattr(payload, "slot_req_ids", None)
    if cur_req_ids is None:
        raise RuntimeError(
            "pending refresh payload missing submission-step slot_req_ids "
            "(live batch_request_ids fallback retired)"
        )
    if callable(normalize_req_ids):
        return tuple(normalize_req_ids(cur_req_ids))
    return _normalize_request_ids(cur_req_ids)


def build_refresh_producer_work_item(
    *,
    payloads: Sequence[Any],
    request_states: Mapping[str, Any] | None,
    req_ids: Sequence[str] | None = None,
    layer_indices: Sequence[int] | None = None,
    max_delay_steps: int,
    current_epoch: int,
    current_handle_id: int = -1,
    admission_reason: str,
    can_drop: bool,
    can_coalesce: bool,
) -> RefreshProducerWorkItem:
    layer_values = (
        tuple(
            _payload_layer_index(payload, payload_index)
            for payload_index, payload in enumerate(payloads)
        )
        if layer_indices is None
        else layer_indices
    )
    target_layer_start = min(layer_values) if layer_values else -1
    target_layer_end = max(layer_values) if layer_values else -1

    decode_steps: list[int] = []
    if request_states:
        req_ids_seen: set[str] = set()
        for req_id in req_ids or ():
            if req_id is None:
                continue
            rid = str(req_id)
            if rid:
                req_ids_seen.add(rid)
        if not req_ids_seen:
            for payload in payloads:
                req_ids_seen.update(_payload_request_ids(payload))
        for req_id in req_ids_seen:
            tracking = request_states.get(req_id)
            decode_step = int(getattr(tracking, "decode_step", -1))
            if decode_step >= 0:
                decode_steps.append(decode_step)

    first_payload = payloads[0] if payloads else None
    capture_epoch = int(getattr(first_payload, "capture_epoch", -1))
    current_epoch = int(current_epoch)
    if current_epoch >= 0:
        ready_epoch = max(capture_epoch, current_epoch)
    else:
        ready_epoch = capture_epoch
    capture_handle_id = int(getattr(first_payload, "capture_handle_id", -1))
    current_handle_id = int(current_handle_id)
    if current_handle_id > 0:
        ready_handle_id = max(capture_handle_id, current_handle_id)
    else:
        ready_handle_id = capture_handle_id
    delay = int(max_delay_steps)
    use_capture_deadline = (
        str(admission_reason or "") == "pending_refresh_rebuild"
        and capture_epoch >= 0
    )
    deadline_base_epoch = capture_epoch if use_capture_deadline else ready_epoch
    deadline_base_handle_id = (
        capture_handle_id
        if use_capture_deadline and capture_handle_id > 0
        else ready_handle_id
    )
    deadline_epoch = (
        int(deadline_base_epoch + delay) if deadline_base_epoch >= 0 else -1
    )
    deadline_handle_id = (
        int(deadline_base_handle_id + delay)
        if deadline_base_handle_id > 0
        else -1
    )
    deadline_slack = (
        int(deadline_epoch - int(current_epoch))
        if deadline_epoch >= 0 and int(current_epoch) >= 0
        else -1
    )

    return RefreshProducerWorkItem(
        target_layer_start=int(target_layer_start),
        target_layer_end=int(target_layer_end),
        decode_step_min=int(min(decode_steps)) if decode_steps else -1,
        decode_step_max=int(max(decode_steps)) if decode_steps else -1,
        ready_epoch=int(ready_epoch),
        deadline_epoch=int(deadline_epoch),
        deadline_handle_id=int(deadline_handle_id),
        deadline_slack_steps=int(deadline_slack),
        can_drop=bool(can_drop),
        can_coalesce=bool(can_coalesce),
        admission_reason=str(admission_reason or ""),
    )


__all__ = [
    "RefreshProducerCarrier",
    "RefreshProducerWorkspace",
    "RefreshProducerWorkItem",
    "build_refresh_producer_work_item",
    "get_refresh_producer_workspace",
    "partition_replay_refresh_payloads_for_direct_submit",
]
