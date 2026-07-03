from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence, TypeAlias

import os
import torch


@dataclass(frozen=True)
class SparseDecodeStructuralKey:
    graph_key: object
    batch_capacity: int
    max_pages_per_row: int
    block_size: int
    page_size: int
    num_kv_heads: int
    device_type: str
    device_index: int


ROW_TABLE_FALLBACK: Literal["row_table_fallback"] = "row_table_fallback"
SparseDecodeAffineDescriptor: TypeAlias = Literal["row_table_fallback"] | tuple[int, ...]


@dataclass(frozen=True)
class SparseDecodeRowDescriptor:
    epoch: int
    visible_epoch: int
    row_mode_epoch: int
    live_row_index: int
    graph_row_index: int
    req_id: str
    visible_k: int
    row_effective: int
    launch_effective: int
    row_mode: str
    affine_descriptor: SparseDecodeAffineDescriptor
    row_table_pages: tuple[int, ...]
    segment_pages: int

    @property
    def uses_row_table_fallback(self) -> bool:
        return self.affine_descriptor == ROW_TABLE_FALLBACK or int(self.segment_pages) < 0


@dataclass(frozen=True)
class SparseDecodeRowDescriptorSnapshot:
    valid: bool
    epoch: int
    rows_by_graph_row: tuple[SparseDecodeRowDescriptor | None, ...]
    failure_reason: str | None = None


class SparseDecodeBatchState:
    def __init__(self, *, device: torch.device | None = None) -> None:
        self._device = torch.device("cpu") if device is None else torch.device(device)
        self._visible_effective_k_len_i32: torch.Tensor | None = None
        self._visible_effective_k_len_cpu_buffer: torch.Tensor | None = None
        self._visible_effective_k_len_cpu: tuple[int, ...] = tuple()
        self._row_effective_k_len_cpu: tuple[int, ...] = tuple()
        self._launch_effective_k_len_cpu: tuple[int, ...] = tuple()
        self._row_affine_descriptor_by_graph_row: tuple[
            SparseDecodeAffineDescriptor, ...
        ] = tuple()
        self._row_table_pages_by_graph_row: tuple[tuple[int, ...], ...] = tuple()
        self._segment_pages_by_graph_row: tuple[int, ...] = tuple()
        self._visible_step_id: int = -1
        self._structural_key: SparseDecodeStructuralKey | None = None
        self._step_id: int = -1
        self._req_ids: tuple[str, ...] = tuple()
        self._active_row_indices: tuple[int, ...] = tuple()
        self._slot_mapping_signature: tuple[int, ...] = tuple()
        self._q_lens_by_row: tuple[int, ...] = tuple()
        self._row_mode_class: tuple[str, ...] = tuple()
        self._capacity_key: object = None
        self._async_spec_active: bool = False
        self._native_gpu_lengths_authoritative: bool = False
        self._last_update_source: str = "uninitialized"
        self.last_failure_reason: str | None = "uninitialized"

    def reset_for_graph_capacity(
        self,
        *,
        graph_key: object,
        batch_capacity: int,
        max_pages_per_row: int,
        block_size: int,
        page_size: int,
        num_kv_heads: int,
        device: torch.device,
    ) -> None:
        device = torch.device(device)
        if int(batch_capacity) < 0:
            raise ValueError("batch_capacity must be non-negative")
        device_index = int(device.index) if device.index is not None else -1
        key = SparseDecodeStructuralKey(
            graph_key=graph_key,
            batch_capacity=int(batch_capacity),
            max_pages_per_row=int(max_pages_per_row),
            block_size=int(block_size),
            page_size=int(page_size),
            num_kv_heads=int(num_kv_heads),
            device_type=str(device.type),
            device_index=device_index,
        )
        needs_alloc = (
            self._visible_effective_k_len_i32 is None
            or self._structural_key != key
            or self._visible_effective_k_len_i32.device != device
            or int(self._visible_effective_k_len_i32.numel()) != int(batch_capacity)
        )
        self._device = device
        self._structural_key = key
        if needs_alloc:
            self._visible_effective_k_len_i32 = torch.zeros(
                (int(batch_capacity),), dtype=torch.int32, device=device
            )
            self._visible_effective_k_len_cpu = tuple(0 for _ in range(int(batch_capacity)))
            self._row_effective_k_len_cpu = tuple(0 for _ in range(int(batch_capacity)))
            self._launch_effective_k_len_cpu = tuple(0 for _ in range(int(batch_capacity)))
            self._row_affine_descriptor_by_graph_row = tuple(
                ROW_TABLE_FALLBACK for _ in range(int(batch_capacity))
            )
            self._row_table_pages_by_graph_row = tuple(
                tuple() for _ in range(int(batch_capacity))
            )
            self._segment_pages_by_graph_row = tuple(0 for _ in range(int(batch_capacity)))
        if (
            self._visible_effective_k_len_cpu_buffer is None
            or int(self._visible_effective_k_len_cpu_buffer.numel()) != int(batch_capacity)
        ):
            self._visible_effective_k_len_cpu_buffer = torch.zeros(
                (int(batch_capacity),), dtype=torch.int32, device="cpu"
            )
        self.last_failure_reason = None

    def update_runner_identity_from_prepare_inputs(
        self,
        *,
        step_id: int,
        req_ids: Sequence[object],
        active_row_indices: Sequence[int],
        slot_mapping_signature: Sequence[int],
        q_lens_by_row: Sequence[int],
        row_mode_class: Sequence[object],
        capacity_key: object,
        async_spec_active: bool,
        native_gpu_lengths_authoritative: bool,
    ) -> None:
        self._step_id = int(step_id)
        self._req_ids = tuple(str(v) for v in req_ids)
        self._active_row_indices = tuple(int(v) for v in active_row_indices)
        self._slot_mapping_signature = tuple(int(v) for v in slot_mapping_signature)
        self._q_lens_by_row = tuple(int(v) for v in q_lens_by_row)
        self._row_mode_class = tuple(str(v) for v in row_mode_class)
        self._capacity_key = capacity_key
        self._async_spec_active = bool(async_spec_active)
        self._native_gpu_lengths_authoritative = bool(native_gpu_lengths_authoritative)
        self._visible_step_id = -1
        self._visible_effective_k_len_cpu = tuple()
        self._row_effective_k_len_cpu = tuple()
        self._launch_effective_k_len_cpu = tuple()
        self._row_affine_descriptor_by_graph_row = tuple()
        self._row_table_pages_by_graph_row = tuple()
        self._segment_pages_by_graph_row = tuple()
        self._last_update_source = "prepare_inputs"
        self.last_failure_reason = None

    def initialize_visible_k_for_graph_capture(
        self,
        *,
        visible_k: int,
        update_source: str = "graph_capture",
    ) -> None:
        target = self._require_tensor()
        value = int(visible_k)
        if value < 0:
            self.last_failure_reason = "visible_k_negative"
            raise RuntimeError("graph capture visible K must be non-negative")
        target.fill_(value)
        values = tuple(value for _ in range(int(target.numel())))
        self._visible_effective_k_len_cpu = values
        self._row_effective_k_len_cpu = values
        self._launch_effective_k_len_cpu = values
        self._row_affine_descriptor_by_graph_row = tuple(
            ROW_TABLE_FALLBACK for _ in range(int(target.numel()))
        )
        self._row_table_pages_by_graph_row = tuple(tuple() for _ in range(int(target.numel())))
        self._segment_pages_by_graph_row = tuple(0 for _ in range(int(target.numel())))
        self._visible_step_id = int(self._step_id)
        self._last_update_source = str(update_source)
        self.last_failure_reason = None

    def update_visible_k_from_step_bound_meta(
        self,
        *,
        step_id: int,
        step_bound_meta: object,
        launch_plan: object,
        row_effective_k_by_row: Sequence[int],
        launch_effective_k_by_row: Sequence[int],
    ) -> None:
        meta_step_id = self._step_bound_meta_step_id(step_bound_meta)
        meta_has_descriptor_payload = self._step_bound_meta_has_descriptor_payload(
            step_bound_meta
        )
        if meta_has_descriptor_payload and meta_step_id is None:
            self._invalidate_visible_descriptor_state(
                reason="step_bound_meta_epoch_missing"
            )
            raise RuntimeError("step_bound_meta epoch missing")
        if meta_step_id is not None:
            if int(meta_step_id) != int(step_id) or int(meta_step_id) != int(self._step_id):
                self._invalidate_visible_descriptor_state(
                    reason="step_bound_meta_epoch_mismatch"
                )
                raise RuntimeError("step_bound_meta epoch mismatch")
        tensor = getattr(launch_plan, "launch_effective_k_len_i32", None)
        if not isinstance(tensor, torch.Tensor):
            self.last_failure_reason = "missing_launch_effective_tensor"
            raise RuntimeError("launch_plan missing launch_effective_k_len_i32")
        if tensor.dtype != torch.int32:
            self.last_failure_reason = "launch_effective_dtype"
            raise RuntimeError("launch_effective_k_len_i32 must be torch.int32")
        if tensor.device != self._device:
            self.last_failure_reason = "launch_effective_device"
            raise RuntimeError("launch_effective_k_len_i32 device mismatch")
        if tensor.dim() != 1 or not tensor.is_contiguous():
            self.last_failure_reason = "launch_effective_shape"
            raise RuntimeError("launch_effective_k_len_i32 must be contiguous rank-1")

        launch_values = tuple(int(v) for v in launch_effective_k_by_row)
        launch_cpu = getattr(launch_plan, "launch_effective_k_len_cpu", None)
        if launch_cpu is not None:
            try:
                launch_cpu_values = tuple(int(v) for v in tuple(launch_cpu))
            except Exception:
                launch_cpu_values = tuple()
            if len(launch_cpu_values) >= len(launch_values):
                if launch_cpu_values[: len(launch_values)] != launch_values:
                    self.last_failure_reason = "launch_effective_cpu_mismatch"
                    raise RuntimeError(
                        "launch_effective_k_len_cpu does not match framework launch values"
                    )
        # SparseDecodeBatchState is the per-step writer for sparse-visible
        # seqused. The GPU tensor holds the sparse effective K used by RRP;
        # CPU mirrors keep row/launch coverage checks aligned with the delta.
        self.update_visible_k_from_values(
            step_id=step_id,
            visible_k_tensor=None,
            row_effective_k_by_row=row_effective_k_by_row,
            launch_effective_k_by_row=launch_values,
            affine_descriptor_by_row=(
                getattr(step_bound_meta, "affine_descriptor_by_row", None)
                if meta_has_descriptor_payload
                else None
            ),
            row_table_pages_by_row=(
                getattr(step_bound_meta, "row_table_pages_by_row", None)
                if meta_has_descriptor_payload
                else None
            ),
            segment_pages_by_row=(
                getattr(step_bound_meta, "segment_pages_by_row", None)
                if meta_has_descriptor_payload
                else None
            ),
            descriptor_epoch=meta_step_id,
        )

    def update_visible_k_from_values(
        self,
        *,
        step_id: int,
        visible_k_tensor: torch.Tensor | None,
        row_effective_k_by_row: Sequence[int],
        launch_effective_k_by_row: Sequence[int],
        affine_descriptor_by_row: Sequence[object] | None = None,
        row_table_pages_by_row: Sequence[Sequence[int] | None] | None = None,
        segment_pages_by_row: Sequence[int] | None = None,
        descriptor_epoch: int | None = None,
        update_source: str = "step_bound_meta",
    ) -> None:
        target = self._require_tensor()
        if int(step_id) != int(self._step_id):
            self.last_failure_reason = "step_id_mismatch"
            raise RuntimeError("SparseDecodeBatchState visible K update step_id mismatch")
        has_descriptor_payload = self._descriptor_payload_supplied(
            affine_descriptor_by_row=affine_descriptor_by_row,
            row_table_pages_by_row=row_table_pages_by_row,
            segment_pages_by_row=segment_pages_by_row,
        )
        if has_descriptor_payload:
            if descriptor_epoch is None:
                self._invalidate_visible_descriptor_state(reason="descriptor_epoch_missing")
                raise RuntimeError("descriptor epoch missing")
            if int(descriptor_epoch) != int(step_id) or int(descriptor_epoch) != int(self._step_id):
                self._invalidate_visible_descriptor_state(reason="descriptor_epoch_mismatch")
                raise RuntimeError("descriptor epoch mismatch")
            if visible_k_tensor is not None:
                self._invalidate_visible_descriptor_state(
                    reason="descriptor_visible_tensor_unsupported"
                )
                raise RuntimeError("descriptor payload cannot use visible_k_tensor")
        previous_visible_step_id = int(self._visible_step_id)
        previous_visible_values = tuple(self._visible_effective_k_len_cpu)
        previous_row_values = tuple(self._row_effective_k_len_cpu)
        previous_launch_values = tuple(self._launch_effective_k_len_cpu)
        previous_affine_values = tuple(self._row_affine_descriptor_by_graph_row)
        previous_row_table_values = tuple(self._row_table_pages_by_graph_row)
        previous_segment_pages = tuple(self._segment_pages_by_graph_row)
        self._visible_step_id = -1
        self._visible_effective_k_len_cpu = tuple()
        self._row_effective_k_len_cpu = tuple()
        self._launch_effective_k_len_cpu = tuple()
        self._row_affine_descriptor_by_graph_row = tuple()
        self._row_table_pages_by_graph_row = tuple()
        self._segment_pages_by_graph_row = tuple()

        live_count = len(self._req_ids)
        identity_lengths = (
            ("active_row_indices", len(self._active_row_indices)),
            ("slot_mapping_signature", len(self._slot_mapping_signature)),
            ("q_lens_by_row", len(self._q_lens_by_row)),
            ("row_mode_class", len(self._row_mode_class)),
        )
        for name, length in identity_lengths:
            if int(length) != live_count:
                self.last_failure_reason = f"{name}_len_mismatch"
                raise RuntimeError(
                    f"{name} length {length} does not match live row count {live_count}"
                )

        row_values = tuple(int(v) for v in row_effective_k_by_row)
        launch_values = tuple(int(v) for v in launch_effective_k_by_row)
        if len(row_values) != live_count:
            self.last_failure_reason = "row_effective_k_len_mismatch"
            raise RuntimeError(
                "row_effective_k_by_row length "
                f"{len(row_values)} does not match live row count {live_count}"
            )
        if len(launch_values) != live_count:
            self.last_failure_reason = "launch_effective_k_len_mismatch"
            raise RuntimeError(
                "launch_effective_k_by_row length "
                f"{len(launch_values)} does not match live row count {live_count}"
            )
        if int(target.numel()) < live_count:
            self.last_failure_reason = "visible_k_coverage"
            raise RuntimeError("visible K tensor does not cover live rows")

        for row, value in enumerate(row_values):
            if int(value) < 0:
                self.last_failure_reason = "row_effective_k_negative"
                raise RuntimeError(f"row_effective_k_by_row contains negative K at row {row}")
        for row, value in enumerate(launch_values):
            if int(value) < 0:
                self.last_failure_reason = "launch_effective_k_negative"
                raise RuntimeError(f"launch_effective_k_by_row contains negative K at row {row}")

        capacity = int(target.numel())
        active_rows = tuple(int(v) for v in self._active_row_indices)
        if len(active_rows) != live_count:
            self.last_failure_reason = "active_row_indices_len_mismatch"
            raise RuntimeError("active row index coverage does not match live row count")
        if any(row < 0 or row >= capacity for row in active_rows):
            self.last_failure_reason = "active_row_index_out_of_bounds"
            raise RuntimeError("active row index is outside visible K tensor capacity")
        if len(set(active_rows)) != len(active_rows):
            self.last_failure_reason = "active_row_index_duplicate"
            raise RuntimeError("active row indices must be unique")

        row_graph_values = [0 for _ in range(capacity)]
        launch_graph_values = [0 for _ in range(capacity)]
        for live_row, arena_row in enumerate(active_rows):
            row_graph_values[int(arena_row)] = int(row_values[live_row])
            launch_graph_values[int(arena_row)] = int(launch_values[live_row])
        row_graph_values_tuple = tuple(int(v) for v in row_graph_values)
        launch_graph_values_tuple = tuple(int(v) for v in launch_graph_values)
        if has_descriptor_payload:
            _dn_on = os.environ.get("VLLM_SPARSE_CACHE_DESC_NORMALIZE") == "1"
            _dn_key = (
                (id(affine_descriptor_by_row), id(row_table_pages_by_row), id(segment_pages_by_row), active_rows, capacity)
                if _dn_on else None
            )
            _dn_c = getattr(self, "_desc_normalize_cache", None) if _dn_on else None
            if _dn_c is not None and _dn_c[0] == _dn_key:
                (
                    affine_graph_values_tuple,
                    row_table_graph_values_tuple,
                    segment_pages_graph_values_tuple,
                ) = _dn_c[1]
            else:
                affine_values = self._normalize_affine_descriptors(
                    affine_descriptor_by_row=affine_descriptor_by_row,
                    live_count=live_count,
                )
                row_table_values = self._normalize_row_table_pages(
                    row_table_pages_by_row=row_table_pages_by_row,
                    live_count=live_count,
                )
                segment_page_values = self._normalize_segment_pages(
                    segment_pages_by_row=segment_pages_by_row,
                    live_count=live_count,
                )
                affine_graph_values = [ROW_TABLE_FALLBACK for _ in range(capacity)]
                row_table_graph_values = [tuple() for _ in range(capacity)]
                segment_pages_graph_values = [0 for _ in range(capacity)]
                for live_row, arena_row in enumerate(active_rows):
                    affine_graph_values[int(arena_row)] = affine_values[live_row]
                    row_table_graph_values[int(arena_row)] = row_table_values[live_row]
                    segment_pages_graph_values[int(arena_row)] = int(segment_page_values[live_row])
                affine_graph_values_tuple = tuple(affine_graph_values)
                row_table_graph_values_tuple = tuple(row_table_graph_values)
                segment_pages_graph_values_tuple = tuple(
                    int(v) for v in segment_pages_graph_values
                )
                if _dn_on:
                    self._desc_normalize_cache = (
                        _dn_key,
                        (affine_graph_values_tuple, row_table_graph_values_tuple, segment_pages_graph_values_tuple),
                    )
        elif os.environ.get("VLLM_FIX_CARRY_DESCRIPTOR") == "1":
            affine_graph_values_tuple = previous_affine_values
            row_table_graph_values_tuple = previous_row_table_values
            segment_pages_graph_values_tuple = previous_segment_pages
        else:
            affine_graph_values_tuple = tuple()
            row_table_graph_values_tuple = tuple()
            segment_pages_graph_values_tuple = tuple()

        if (
            previous_visible_step_id == int(step_id)
            and previous_visible_values == row_graph_values_tuple
            and previous_row_values == row_graph_values_tuple
            and previous_launch_values == launch_graph_values_tuple
            and (
                not has_descriptor_payload
                or (
                    previous_affine_values == affine_graph_values_tuple
                    and previous_row_table_values == row_table_graph_values_tuple
                    and previous_segment_pages == segment_pages_graph_values_tuple
                )
            )
        ):
            self._visible_effective_k_len_cpu = row_graph_values_tuple
            self._row_effective_k_len_cpu = row_graph_values_tuple
            self._launch_effective_k_len_cpu = launch_graph_values_tuple
            self._row_affine_descriptor_by_graph_row = affine_graph_values_tuple
            self._row_table_pages_by_graph_row = row_table_graph_values_tuple
            self._segment_pages_by_graph_row = segment_pages_graph_values_tuple
            self._visible_step_id = int(step_id)
            self._last_update_source = str(update_source)
            self.last_failure_reason = None
            return

        # VLLM_SPARSE_VISIBLE_FAST_ADD: collapse the per-step CPU staging loop +
        # copy_ to a constant-time target[start:stop].add_(delta) on uniform monotone
        # growth (mirrors rrp_row_table_manager._try_increment_seqused_k). Default OFF.
        if (
            visible_k_tensor is None
            and not has_descriptor_payload
            and os.environ.get("VLLM_SPARSE_VISIBLE_FAST_ADD") == "1"
        ):
            _fa_delta = self._visible_fast_add_uniform_delta(
                previous_visible_values=previous_visible_values,
                previous_row_values=previous_row_values,
                previous_launch_values=previous_launch_values,
                previous_visible_step_id=previous_visible_step_id,
                row_graph_values_tuple=row_graph_values_tuple,
                launch_graph_values_tuple=launch_graph_values_tuple,
                capacity=capacity,
                target=target,
            )
            if _fa_delta is not None:
                _fa_start, _fa_stop, _fa_step = _fa_delta
                _fa_assert = (
                    os.environ.get("VLLM_SPARSE_VISIBLE_FAST_ADD_ASSERT") == "1"
                )
                if _fa_assert:
                    # Shadow path: run fast-add on a SCRATCH clone, run the full
                    # staging path on the real target, then assert equality and
                    # keep the FULL result. Randomness-immune (validates tensor).
                    _fa_scratch = target.clone()
                    _fa_scratch[_fa_start:_fa_stop].add_(int(_fa_step))
                    cpu_buffer = self._visible_effective_k_len_cpu_buffer
                    if (
                        not isinstance(cpu_buffer, torch.Tensor)
                        or int(cpu_buffer.numel()) < capacity
                    ):
                        self.last_failure_reason = "visible_k_cpu_buffer_coverage"
                        raise RuntimeError(
                            "visible K CPU staging buffer does not cover graph rows"
                        )
                    for row, value in enumerate(row_graph_values):
                        cpu_buffer[row] = int(value)
                    target[:capacity].copy_(
                        cpu_buffer[:capacity],
                        non_blocking=target.device.type == "cuda",
                    )
                    if not torch.equal(_fa_scratch[:capacity], target[:capacity]):
                        self.last_failure_reason = "visible_fast_add_tensor_mismatch"
                        raise RuntimeError(
                            "VLLM_SPARSE_VISIBLE_FAST_ADD shadow-assert: fast-add tensor "
                            "!= full restage+copy result"
                        )
                else:
                    target[_fa_start:_fa_stop].add_(int(_fa_step))
                # Cheap CPU-tuple maintenance (length-only update): for a uniform
                # positive delta the new graph tuples ARE row_graph_values_tuple /
                # launch_graph_values_tuple already computed above; descriptor tuples
                # are unchanged (no descriptor payload). Mirror the full-path tail.
                self._visible_effective_k_len_cpu = row_graph_values_tuple
                self._row_effective_k_len_cpu = row_graph_values_tuple
                self._launch_effective_k_len_cpu = launch_graph_values_tuple
                self._row_affine_descriptor_by_graph_row = affine_graph_values_tuple
                self._row_table_pages_by_graph_row = row_table_graph_values_tuple
                self._segment_pages_by_graph_row = segment_pages_graph_values_tuple
                self._visible_step_id = int(step_id)
                self._last_update_source = str(update_source)
                self.last_failure_reason = None
                self._visible_fast_add_hits = (
                    int(getattr(self, "_visible_fast_add_hits", 0)) + 1
                )
                return

        if visible_k_tensor is not None:
            if visible_k_tensor.dtype != torch.int32:
                self.last_failure_reason = "visible_k_source_dtype"
                raise RuntimeError("visible K source tensor must be torch.int32")
            if visible_k_tensor.device != target.device:
                self.last_failure_reason = "visible_k_source_device"
                raise RuntimeError("visible K source tensor device mismatch")
            if visible_k_tensor.dim() != 1 or not visible_k_tensor.is_contiguous():
                self.last_failure_reason = "visible_k_source_shape"
                raise RuntimeError("visible K source tensor must be contiguous rank-1")
            if int(visible_k_tensor.numel()) < live_count:
                self.last_failure_reason = "visible_k_source_coverage"
                raise RuntimeError("visible K source tensor does not cover live rows")
            if int(visible_k_tensor.data_ptr()) == int(target.data_ptr()):
                source_values = row_values
            elif (
                int(visible_k_tensor.numel()) >= capacity
                and active_rows == tuple(range(live_count))
            ):
                target[:capacity].copy_(
                    visible_k_tensor[:capacity],
                    non_blocking=target.device.type == "cuda",
                )
                source_values = None
            elif visible_k_tensor.device.type == "cuda" and target.device.type == "cuda":
                target.zero_()
                active_index = torch.tensor(
                    active_rows,
                    dtype=torch.long,
                    device=target.device,
                )
                target.index_copy_(0, active_index, visible_k_tensor[:live_count])
                source_values = None
            else:
                source_values = tuple(
                    int(v) for v in visible_k_tensor[:live_count].detach().cpu().tolist()
                )
            if source_values is not None:
                for live_row, arena_row in enumerate(active_rows):
                    row_graph_values[int(arena_row)] = int(source_values[live_row])
                cpu_buffer = self._visible_effective_k_len_cpu_buffer
                if (
                    not isinstance(cpu_buffer, torch.Tensor)
                    or int(cpu_buffer.numel()) < capacity
                ):
                    self.last_failure_reason = "visible_k_cpu_buffer_coverage"
                    raise RuntimeError("visible K CPU staging buffer does not cover graph rows")
                cpu_buffer[:capacity].copy_(
                    torch.as_tensor(
                        row_graph_values[:capacity],
                        dtype=torch.int32,
                        device="cpu",
                    )
                )
                target[:capacity].copy_(
                    cpu_buffer[:capacity],
                    non_blocking=target.device.type == "cuda",
                )
        else:
            cpu_buffer = self._visible_effective_k_len_cpu_buffer
            if not isinstance(cpu_buffer, torch.Tensor) or int(cpu_buffer.numel()) < capacity:
                self.last_failure_reason = "visible_k_cpu_buffer_coverage"
                raise RuntimeError("visible K CPU staging buffer does not cover graph rows")
            cpu_buffer[:capacity].copy_(
                torch.as_tensor(
                    row_graph_values[:capacity],
                    dtype=torch.int32,
                    device="cpu",
                )
            )
            target[:capacity].copy_(
                cpu_buffer[:capacity],
                non_blocking=target.device.type == "cuda",
            )
        self._visible_effective_k_len_cpu = row_graph_values_tuple
        self._row_effective_k_len_cpu = row_graph_values_tuple
        self._launch_effective_k_len_cpu = launch_graph_values_tuple
        self._row_affine_descriptor_by_graph_row = affine_graph_values_tuple
        self._row_table_pages_by_graph_row = row_table_graph_values_tuple
        self._segment_pages_by_graph_row = segment_pages_graph_values_tuple
        self._visible_step_id = int(step_id)
        self._last_update_source = str(update_source)
        self.last_failure_reason = None

    def current_step_row_descriptors(self) -> SparseDecodeRowDescriptorSnapshot:
        if int(self._visible_step_id) != int(self._step_id):
            self.last_failure_reason = "visible_k_stale"
            raise RuntimeError("SparseDecodeBatchState visible K values are stale")
        capacity = len(self._visible_effective_k_len_cpu)
        if capacity <= 0:
            self.last_failure_reason = "visible_k_coverage"
            raise RuntimeError("SparseDecodeBatchState visible K values are unavailable")
        graph_fields = (
            ("row_effective_k", len(self._row_effective_k_len_cpu)),
            ("launch_effective_k", len(self._launch_effective_k_len_cpu)),
        )
        for name, length in graph_fields:
            if int(length) < capacity:
                self.last_failure_reason = f"{name}_coverage"
                raise RuntimeError(f"SparseDecodeBatchState {name} does not cover graph rows")
        if len(self._row_mode_class) != len(self._req_ids):
            self.last_failure_reason = "row_mode_class_len_mismatch"
            raise RuntimeError("SparseDecodeBatchState row modes do not cover live rows")
        if len(self._active_row_indices) != len(self._req_ids):
            self.last_failure_reason = "active_row_indices_len_mismatch"
            raise RuntimeError("SparseDecodeBatchState active rows do not cover live rows")

        rows: list[SparseDecodeRowDescriptor | None] = [None for _ in range(capacity)]
        for live_row, graph_row in enumerate(self._active_row_indices):
            graph_row = int(graph_row)
            if graph_row < 0 or graph_row >= capacity:
                self.last_failure_reason = "active_row_index_out_of_bounds"
                raise RuntimeError("SparseDecodeBatchState active row is outside graph capacity")
            if rows[graph_row] is not None:
                self.last_failure_reason = "active_row_index_duplicate"
                raise RuntimeError("SparseDecodeBatchState active rows must be unique")
            affine_descriptor = (
                self._row_affine_descriptor_by_graph_row[graph_row]
                if len(self._row_affine_descriptor_by_graph_row) > graph_row
                else ROW_TABLE_FALLBACK
            )
            row_table_pages = (
                self._row_table_pages_by_graph_row[graph_row]
                if len(self._row_table_pages_by_graph_row) > graph_row
                else tuple()
            )
            segment_pages = (
                self._segment_pages_by_graph_row[graph_row]
                if len(self._segment_pages_by_graph_row) > graph_row
                else 0
            )
            rows[graph_row] = SparseDecodeRowDescriptor(
                epoch=int(self._step_id),
                visible_epoch=int(self._visible_step_id),
                row_mode_epoch=int(self._step_id),
                live_row_index=int(live_row),
                graph_row_index=int(graph_row),
                req_id=str(self._req_ids[live_row]),
                visible_k=int(self._visible_effective_k_len_cpu[graph_row]),
                row_effective=int(self._row_effective_k_len_cpu[graph_row]),
                launch_effective=int(self._launch_effective_k_len_cpu[graph_row]),
                row_mode=str(self._row_mode_class[live_row]),
                affine_descriptor=affine_descriptor,
                row_table_pages=row_table_pages,
                segment_pages=int(segment_pages),
            )
        self.last_failure_reason = None
        return SparseDecodeRowDescriptorSnapshot(
            valid=True,
            epoch=int(self._step_id),
            rows_by_graph_row=tuple(rows),
            failure_reason=None,
        )

    def descriptor_payload_covers_active_rows(self, *, require_pages: bool = True) -> bool:
        if int(self._visible_step_id) != int(self._step_id):
            self.last_failure_reason = "visible_k_stale"
            return False
        capacity = len(self._visible_effective_k_len_cpu)
        if capacity <= 0:
            self.last_failure_reason = "visible_k_coverage"
            return False
        descriptor_lengths = (
            ("affine_descriptor", len(self._row_affine_descriptor_by_graph_row)),
            ("row_table_pages", len(self._row_table_pages_by_graph_row)),
            ("segment_pages", len(self._segment_pages_by_graph_row)),
        )
        for name, length in descriptor_lengths:
            if int(length) < capacity:
                self.last_failure_reason = f"{name}_coverage"
                return False
        if len(self._row_effective_k_len_cpu) < capacity:
            self.last_failure_reason = "row_effective_k_coverage"
            return False
        _covers_key = None
        if os.environ.get("VLLM_SPARSE_CACHE_COVERS_VALIDATION") == "1":
            _covers_key = (
                self._row_table_pages_by_graph_row,
                self._row_affine_descriptor_by_graph_row,
                self._segment_pages_by_graph_row,
                self._active_row_indices,
                bool(require_pages),
            )
            if getattr(self, "_covers_true_cache_key", None) == _covers_key:
                self.last_failure_reason = None
                return True
        max_pages = (
            int(self._structural_key.max_pages_per_row)
            if self._structural_key is not None
            else 0
        )
        for graph_row in self._active_row_indices:
            graph_row_i = int(graph_row)
            if graph_row_i < 0 or graph_row_i >= capacity:
                self.last_failure_reason = "active_row_index_out_of_bounds"
                return False
            row_effective = int(self._row_effective_k_len_cpu[graph_row_i])
            pages = tuple(int(v) for v in self._row_table_pages_by_graph_row[graph_row_i])
            if max_pages > 0 and len(pages) > max_pages:
                self.last_failure_reason = "row_table_pages_exceed_capacity"
                return False
            if bool(require_pages) and row_effective > 0 and not pages:
                self.last_failure_reason = "row_table_pages_missing"
                return False
            segment_pages = int(self._segment_pages_by_graph_row[graph_row_i])
            affine_descriptor = self._row_affine_descriptor_by_graph_row[graph_row_i]
            if affine_descriptor == ROW_TABLE_FALLBACK or segment_pages < 0:
                continue
            try:
                affine_tuple = tuple(int(v) for v in affine_descriptor)
            except (TypeError, ValueError):
                self.last_failure_reason = "affine_descriptor_unsupported"
                return False
            if len(affine_tuple) != 5:
                self.last_failure_reason = "affine_descriptor_len_mismatch"
                return False
            if not self._affine_descriptor_matches_pages(
                affine=affine_tuple,
                pages=pages,
            ):
                self.last_failure_reason = "descriptor_affine_mismatch"
                return False
        self.last_failure_reason = None
        if _covers_key is not None:
            self._covers_true_cache_key = _covers_key
        return True

    def covers_step(
        self,
        *,
        step_id: int,
        req_ids: Sequence[object],
        active_row_indices: Sequence[int],
        slot_mapping_signature: Sequence[int],
        q_lens_by_row: Sequence[int],
        row_mode_class: Sequence[object],
        capacity_key: object,
        row_effective_k_by_row: Sequence[int],
        launch_effective_k_by_row: Sequence[int],
        native_gpu_lengths_authoritative: bool,
    ) -> bool:
        if not self.covers_step_identity(
            step_id=step_id,
            req_ids=req_ids,
            active_row_indices=active_row_indices,
            slot_mapping_signature=slot_mapping_signature,
            q_lens_by_row=q_lens_by_row,
            row_mode_class=row_mode_class,
            capacity_key=capacity_key,
            native_gpu_lengths_authoritative=native_gpu_lengths_authoritative,
        ):
            return False

        if int(self._visible_step_id) != int(self._step_id):
            self.last_failure_reason = "visible_k_stale"
            return False

        live_count = len(self._req_ids)
        active_values = tuple(int(v) for v in active_row_indices)
        live_values = tuple(int(v) for v in row_effective_k_by_row)
        if int(len(live_values)) != live_count:
            self.last_failure_reason = "row_effective_k_len_mismatch"
            return False

        capacity = len(self._visible_effective_k_len_cpu)
        if capacity <= 0:
            self.last_failure_reason = "visible_k_coverage"
            return False
        if len(self._row_effective_k_len_cpu) < capacity:
            self.last_failure_reason = "row_effective_k_coverage"
            return False
        if any(row < 0 or row >= capacity for row in active_values):
            self.last_failure_reason = "active_row_index_out_of_bounds"
            return False
        if len(set(active_values)) != len(active_values):
            self.last_failure_reason = "active_row_index_duplicate"
            return False
        for row, value in enumerate(live_values):
            if int(value) < 0:
                self.last_failure_reason = "row_effective_k_negative"
                return False
            arena_row = int(active_values[row])
            if int(self._row_effective_k_len_cpu[arena_row]) != int(value):
                self.last_failure_reason = "row_effective_k_mismatch"
                return False
        self.last_failure_reason = None
        return True

    def covers_step_identity(
        self,
        *,
        step_id: int,
        req_ids: Sequence[object],
        active_row_indices: Sequence[int],
        slot_mapping_signature: Sequence[int],
        q_lens_by_row: Sequence[int],
        row_mode_class: Sequence[object],
        capacity_key: object,
        native_gpu_lengths_authoritative: bool,
    ) -> bool:
        requested_step_id = int(step_id)
        if requested_step_id != int(self._step_id):
            self.last_failure_reason = "step_id_mismatch"
            return False
        if bool(native_gpu_lengths_authoritative) or self._native_gpu_lengths_authoritative:
            self.last_failure_reason = "async_spec_gpu_lengths_authoritative"
            return False

        live_count = len(self._req_ids)
        req_values = tuple(str(v) for v in req_ids)
        active_values = tuple(int(v) for v in active_row_indices)
        slot_values = tuple(int(v) for v in slot_mapping_signature)
        q_len_values = tuple(int(v) for v in q_lens_by_row)
        row_mode_values = tuple(str(v) for v in row_mode_class)
        input_lengths = (
            ("req_ids", len(req_values)),
            ("active_row_indices", len(active_values)),
            ("slot_mapping_signature", len(slot_values)),
            ("q_lens_by_row", len(q_len_values)),
            ("row_mode_class", len(row_mode_values)),
        )
        for name, length in input_lengths:
            if int(length) != live_count:
                self.last_failure_reason = f"{name}_len_mismatch"
                return False

        if req_values != self._req_ids:
            self.last_failure_reason = "req_ids_mismatch"
            return False
        if active_values != self._active_row_indices:
            self.last_failure_reason = "active_row_indices_mismatch"
            return False
        if slot_values != self._slot_mapping_signature:
            self.last_failure_reason = "slot_mapping_signature_mismatch"
            return False
        if q_len_values != self._q_lens_by_row:
            self.last_failure_reason = "q_lens_by_row_mismatch"
            return False
        if row_mode_values != self._row_mode_class:
            self.last_failure_reason = "row_mode_class_mismatch"
            return False
        if capacity_key != self._capacity_key:
            self.last_failure_reason = "capacity_key_mismatch"
            return False
        capacity = (
            int(self._visible_effective_k_len_i32.numel())
            if isinstance(self._visible_effective_k_len_i32, torch.Tensor)
            else 0
        )
        if capacity <= 0:
            self.last_failure_reason = "visible_k_coverage"
            return False
        if any(row < 0 or row >= capacity for row in active_values):
            self.last_failure_reason = "active_row_index_out_of_bounds"
            return False
        if len(set(active_values)) != len(active_values):
            self.last_failure_reason = "active_row_index_duplicate"
            return False
        self.last_failure_reason = None
        return True

    def invalidate_for_step_failure(self, *, step_id: int, reason: str) -> None:
        self._step_id = int(step_id)
        self._visible_step_id = -1
        self._visible_effective_k_len_cpu = tuple()
        self._row_effective_k_len_cpu = tuple()
        self._launch_effective_k_len_cpu = tuple()
        self._row_affine_descriptor_by_graph_row = tuple()
        self._row_table_pages_by_graph_row = tuple()
        self._segment_pages_by_graph_row = tuple()
        self._last_update_source = "invalidated"
        self.last_failure_reason = str(reason)

    def visible_effective_k_tensor(self) -> torch.Tensor:
        return self._require_tensor()

    def visible_effective_k_cpu(self) -> tuple[int, ...]:
        return self._visible_effective_k_len_cpu

    def row_effective_k_cpu(self) -> tuple[int, ...]:
        return self._row_effective_k_len_cpu

    def launch_effective_k_cpu(self) -> tuple[int, ...]:
        return self._launch_effective_k_len_cpu

    def structural_key(self) -> SparseDecodeStructuralKey:
        if self._structural_key is None:
            raise RuntimeError("SparseDecodeBatchState structural key is not initialized")
        return self._structural_key

    @property
    def last_update_source(self) -> str:
        return self._last_update_source

    @property
    def step_id(self) -> int:
        return int(self._step_id)

    @property
    def native_gpu_lengths_authoritative(self) -> bool:
        return bool(self._native_gpu_lengths_authoritative)

    def _visible_fast_add_uniform_delta(
        self,
        *,
        previous_visible_values: tuple,
        previous_row_values: tuple,
        previous_launch_values: tuple,
        previous_visible_step_id: int,
        row_graph_values_tuple: tuple,
        launch_graph_values_tuple: tuple,
        capacity: int,
        target: torch.Tensor,
    ):
        """Return (start, stop, delta) iff the visible/row/launch graph values grew by a
        SINGLE uniform positive delta over a CONTIGUOUS changed-row range vs the previous
        step, the previous step's values are present + aligned, and `target` is this
        state's own visible tensor (so it currently holds previous_visible_values).
        Mirrors rrp_row_table_manager._try_increment_seqused_k. Else None (full path).
        """
        # (a) prior values must exist and align: the previous full write established
        #     target[:capacity] == previous_visible_values. A wiped/first step has
        #     previous_visible_step_id == -1 or empty tuples -> bail.
        if int(previous_visible_step_id) < 0:
            return None
        prev = previous_visible_values
        cur = row_graph_values_tuple
        if (
            len(prev) != capacity
            or len(cur) != capacity
            or len(previous_row_values) != capacity
            or len(previous_launch_values) != capacity
            or len(launch_graph_values_tuple) != capacity
        ):
            return None
        # visible/row tracks must agree both before and after (the full path keeps
        # _visible_effective_k_len_cpu == _row_effective_k_len_cpu).
        if previous_row_values != prev or cur != row_graph_values_tuple:
            return None
        # (b) uniform positive contiguous delta over the row track.
        deltas = tuple(int(cur[r]) - int(prev[r]) for r in range(capacity))
        if not deltas:
            return None
        changed_rows = tuple(i for i, d in enumerate(deltas) if int(d) != 0)
        if not changed_rows:
            # No change in the row track -> the early-return above should have caught
            # this; nothing to add. Defer to the full path (cheap, correct).
            return None
        first_delta = int(deltas[changed_rows[0]])
        if first_delta <= 0 or any(
            int(deltas[i]) != first_delta for i in changed_rows
        ):
            return None
        start = int(changed_rows[0])
        stop = int(changed_rows[-1]) + 1
        if changed_rows != tuple(range(start, stop)):
            return None
        # The launch track must grow by the IDENTICAL per-row delta over the SAME rows
        # (launch_effective tracks visible for length-only steps); otherwise the full
        # path's distinct launch tuple semantics could differ -> bail.
        launch_deltas = tuple(
            int(launch_graph_values_tuple[r]) - int(previous_launch_values[r])
            for r in range(capacity)
        )
        if launch_deltas != deltas:
            return None
        # (d) target must be this state's own visible tensor (no foreign writer moved it).
        own = self._visible_effective_k_len_i32
        if not isinstance(own, torch.Tensor) or own is not target:
            return None
        if int(target.numel()) < stop:
            return None
        return (start, stop, first_delta)

    def _require_tensor(self) -> torch.Tensor:
        if self._visible_effective_k_len_i32 is None:
            raise RuntimeError("SparseDecodeBatchState visible tensor is not initialized")
        return self._visible_effective_k_len_i32

    def _step_bound_meta_step_id(self, step_bound_meta: object) -> int | None:
        for name in ("rrp_descriptor_epoch", "descriptor_epoch", "descriptor_step_id"):
            value = getattr(step_bound_meta, name, None)
            if value is None:
                continue
            value_i = int(value)
            if value_i >= 0:
                return value_i
        for name in ("step_id", "epoch"):
            value = getattr(step_bound_meta, name, None)
            if value is not None:
                return int(value)
        return None

    def _step_bound_meta_has_descriptor_payload(self, step_bound_meta: object) -> bool:
        return self._descriptor_payload_supplied(
            affine_descriptor_by_row=getattr(
                step_bound_meta, "affine_descriptor_by_row", None
            ),
            row_table_pages_by_row=getattr(
                step_bound_meta, "row_table_pages_by_row", None
            ),
            segment_pages_by_row=getattr(
                step_bound_meta, "segment_pages_by_row", None
            ),
        )

    def _descriptor_payload_supplied(
        self,
        *,
        affine_descriptor_by_row: Sequence[object] | None,
        row_table_pages_by_row: Sequence[Sequence[int] | None] | None,
        segment_pages_by_row: Sequence[int] | None,
    ) -> bool:
        for value in (
            affine_descriptor_by_row,
            row_table_pages_by_row,
            segment_pages_by_row,
        ):
            if value is None:
                continue
            try:
                if len(value) <= 0:  # type: ignore[arg-type]
                    continue
            except TypeError:
                return True
            return True
        return False

    def _invalidate_visible_descriptor_state(self, *, reason: str) -> None:
        self._visible_step_id = -1
        self._visible_effective_k_len_cpu = tuple()
        self._row_effective_k_len_cpu = tuple()
        self._launch_effective_k_len_cpu = tuple()
        self._row_affine_descriptor_by_graph_row = tuple()
        self._row_table_pages_by_graph_row = tuple()
        self._segment_pages_by_graph_row = tuple()
        self.last_failure_reason = str(reason)

    def _normalize_affine_descriptors(
        self,
        *,
        affine_descriptor_by_row: Sequence[object] | None,
        live_count: int,
    ) -> tuple[SparseDecodeAffineDescriptor, ...]:
        if affine_descriptor_by_row is None:
            return tuple(ROW_TABLE_FALLBACK for _ in range(int(live_count)))
        values = tuple(affine_descriptor_by_row)
        if len(values) != int(live_count):
            self.last_failure_reason = "affine_descriptor_len_mismatch"
            raise RuntimeError(
                "affine_descriptor_by_row length "
                f"{len(values)} does not match live row count {live_count}"
            )
        return tuple(self._normalize_affine_descriptor(value) for value in values)

    def _normalize_affine_descriptor(self, value: object) -> SparseDecodeAffineDescriptor:
        if value == ROW_TABLE_FALLBACK:
            return ROW_TABLE_FALLBACK
        if isinstance(value, str):
            self.last_failure_reason = "affine_descriptor_unsupported"
            raise RuntimeError("affine descriptor string must be row_table_fallback")
        if isinstance(value, Sequence):
            try:
                return tuple(int(v) for v in value)
            except Exception as exc:
                self.last_failure_reason = "affine_descriptor_unsupported"
                raise RuntimeError("affine descriptor must contain integer fields") from exc
        self.last_failure_reason = "affine_descriptor_unsupported"
        raise RuntimeError("affine descriptor must be row_table_fallback or tuple[int, ...]")

    def _normalize_row_table_pages(
        self,
        *,
        row_table_pages_by_row: Sequence[Sequence[int] | None] | None,
        live_count: int,
    ) -> tuple[tuple[int, ...], ...]:
        if row_table_pages_by_row is None:
            return tuple(tuple() for _ in range(int(live_count)))
        values = tuple(row_table_pages_by_row)
        if len(values) != int(live_count):
            self.last_failure_reason = "row_table_pages_len_mismatch"
            raise RuntimeError(
                "row_table_pages_by_row length "
                f"{len(values)} does not match live row count {live_count}"
            )
        return tuple(tuple() if value is None else tuple(int(v) for v in value) for value in values)

    def _normalize_segment_pages(
        self,
        *,
        segment_pages_by_row: Sequence[int] | None,
        live_count: int,
    ) -> tuple[int, ...]:
        if segment_pages_by_row is None:
            return tuple(0 for _ in range(int(live_count)))
        values = tuple(int(v) for v in segment_pages_by_row)
        if len(values) != int(live_count):
            self.last_failure_reason = "segment_pages_len_mismatch"
            raise RuntimeError(
                "segment_pages_by_row length "
                f"{len(values)} does not match live row count {live_count}"
            )
        return values

    @staticmethod
    def _affine_descriptor_matches_pages(
        *,
        affine: tuple[int, int, int, int, int],
        pages: tuple[int, ...],
    ) -> bool:
        base, stride, segment_pages, second_base, second_stride = (
            int(v) for v in affine
        )
        if segment_pages < 0 or segment_pages > len(pages):
            return False
        rebuilt: list[int] = []
        for page_idx in range(len(pages)):
            if page_idx < segment_pages:
                rebuilt.append(base + stride * page_idx)
            else:
                rebuilt.append(
                    second_base + second_stride * (page_idx - segment_pages)
                )
        return tuple(rebuilt) == tuple(int(v) for v in pages)
