"""
patches/controller_mixins/capture_ring_mixin.py — Capture layout, ring-buffer lease, and pointer-array management.

OWNS:
  - _init_capture_ring_state(): capture ring state initialization
  - _ensure_capture_row_by_batch_row(): cached capture-row tensor lookup
  - _ensure_capture_layout_cpu_tensors(): CPU-side slot/seq tensor materialization
  - _capture_scores_ptrs_for_rows / _log_f_denoms_ptrs_for_rows(): pointer-array builders
  - _map_global_layer_to_capture_slot(): global-layer to ring-slot mapping
  - _ensure_capture_ring_active_lease / _retire_capture_layout(): lease lifecycle
  - _reclaim_capture_ring_leases / _reclaim_retired_buffers(): GPU event-driven reclaim

DEPENDS_ON:
  - patches.buffer_lease_protocol (BufferLease, BufferLeaseRegistry, LeaseKind)
  - patches.sparse_constants (_CAPTURE_CHUNK, _CAPTURE_IN_FLIGHT)
  - step_context_epoch, refresh_stream, chunk_done_evt (cross-mixin state)

ENTRY_POINTS:
  - _init_capture_ring_state(): called from VLLMSparseController.__init__
"""
from __future__ import annotations

import logging
import os
import time
from collections import deque
from typing import TYPE_CHECKING, Deque, Dict, List, Optional, Set, Tuple

import torch

from patches.buffer_lease_protocol import BufferLease, BufferLeaseRegistry, LeaseKind
from patches.sparse_constants import _CAPTURE_CHUNK, _CAPTURE_IN_FLIGHT
from patches.sparse_utils import _is_stream_capturing_or_raise

_log = logging.getLogger(__name__)
_REBUILD_PTRS_CPU_FREE_MAX_PER_NAME = max(2, 2 * int(_CAPTURE_IN_FLIGHT))

if TYPE_CHECKING:
    from patches.sparse_types import StepCaptureLayout, StepContext


class CaptureRingMixin:
    """Capture layout buffer management and ring-buffer lease lifecycle.

    All capture-ring state is initialised via ``_init_capture_ring_state()``
    which must be called from the controller's ``__init__``.
    """

    _MIXIN_REQUIRES: tuple = ("SelectorComputeMixin",)

    # ------------------------------------------------------------------
    # State initialisation
    # ------------------------------------------------------------------

    def _init_capture_ring_state(self) -> None:
        # capture_row_by_batch_row cache
        self._capture_rows_cache: Dict[Tuple[object, ...], torch.Tensor] = {}
        # refresh CPU-side cache (per step)
        self._step_refresh_cpu_cache_epoch: int = -1
        self._step_refresh_cpu_cache_handle_id: int = -1
        self._step_refresh_cpu_cache_handle_generation: int = -1
        self._step_refresh_slot_tensor_cpu: Dict[Tuple[int, ...], torch.Tensor] = {}
        self._step_refresh_seq_tensor_cpu: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], torch.Tensor] = {}
        # rebuild gather pointer arrays
        self._rebuild_ptrs_cpu: Dict[str, torch.Tensor] = {}
        self._rebuild_ptrs_cpu_free: Dict[str, List[torch.Tensor]] = {}
        self._rebuild_ptrs_cpu_pending: Dict[str, Deque[Tuple[torch.Tensor, torch.cuda.Event]]] = {}
        self._rebuild_ptrs_gpu: Dict[str, torch.Tensor] = {}
        self._rebuild_ptrs_signature: Dict[str, Tuple[Tuple[object, ...], int, int, str]] = {}
        self._rebuild_ptrs_ready_events: Dict[str, torch.cuda.Event] = {}
        self._rebuild_ptrs_capture_wait_satisfied_names: Set[str] = set()
        self._writer_pointer_lookup_count_total: int = 0
        self._writer_pointer_rebuild_count_total: int = 0
        self._writer_cached_pointer_op_count_total: int = 0
        self._writer_vector_fallback_count_total: int = 0
        self._source_ready_recorded_after_pointer_publish_total: int = 0
        # ring-buffer lease management
        self._capture_ring_lease_registry: BufferLeaseRegistry = BufferLeaseRegistry(
            num_slots=int(_CAPTURE_IN_FLIGHT)
        )
        self._capture_ring_active_lease_by_buf: List[Optional[BufferLease]] = [
            None for _ in range(int(_CAPTURE_IN_FLIGHT))
        ]
        self._capture_ring_retired_events: Deque[Tuple[str, torch.cuda.Event]] = deque()
        self._lease_stats: Dict[str, int] = {
            "capture_ring_retired": 0,
            "capture_ring_reclaimed": 0,
            "pending": 0,
        }
        # grow-to-fit arena: superseded prefill-capture buffers awaiting async-safe free
        # (held ref + record_stream; freed when the buf_id chunk_done_evt fires).
        self._arena_retired_buckets: Deque[Tuple[object, object]] = deque()

    # ------------------------------------------------------------------
    # Pointer-buffer helpers
    # ------------------------------------------------------------------

    def _get_rebuild_ptr_buffers(
        self, *, name: str, device: torch.device, size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (cpu_pinned, gpu) int64 buffers for pointer arrays."""
        if size <= 0:
            empty_cpu = torch.empty((0,), device="cpu", dtype=torch.int64)
            empty_gpu = torch.empty((0,), device=device, dtype=torch.int64)
            return empty_cpu, empty_gpu
        free_pool = self._rebuild_ptrs_cpu_free.setdefault(name, [])
        pending = self._rebuild_ptrs_cpu_pending.get(name)
        if pending is not None:
            still_pending: Deque[Tuple[torch.Tensor, torch.cuda.Event]] = deque()
            while pending:
                pending_cpu, pending_event = pending.popleft()
                try:
                    ready = bool(pending_event.query())
                except Exception:
                    # [GUARD-NO-SWALLOW] 假不就绪=pinned 池永不回收+事件失效被掩盖。
                    _log.warning("rebuild pointer staging event query failed for name=%s", name, exc_info=True)
                    raise
                if ready:
                    if pending_cpu.is_pinned():
                        self._release_rebuild_ptr_cpu_buffer(
                            name=name,
                            cpu=pending_cpu,
                        )
                else:
                    still_pending.append((pending_cpu, pending_event))
            if still_pending:
                self._rebuild_ptrs_cpu_pending[name] = still_pending
            else:
                self._rebuild_ptrs_cpu_pending.pop(name, None)

        cpu: Optional[torch.Tensor] = None
        for idx, candidate in enumerate(free_pool):
            if candidate.numel() == size and candidate.is_pinned():
                cpu = free_pool.pop(idx)
                break
        if cpu is None:
            cpu = torch.empty((size,), device="cpu", dtype=torch.int64, pin_memory=True)
        self._rebuild_ptrs_cpu[name] = cpu
        gpu = self._rebuild_ptrs_gpu.get(name)
        if gpu is None or gpu.numel() != size or gpu.device != device:
            if gpu is not None and gpu.is_cuda:
                # [REBUILD-PTRS-GPU-REALLOC-UAF-FIX] P1:指针数组换代弃旧,
                # writer gather(deferred replay 另一流)可持旧数组在飞。
                # size=layers 稳态恒定=近死臂,族纯度收口。冷事件。
                self._uaf_guard_record_streams_before_discard(gpu)
                # [PTR-REPUBLISH-REPLAY-SAFE 2026-07-09] captured writer graphs
                # bake THIS buffer's address into their launch_args; a realloc
                # kills every baked address. Invalidate the whole writer-graph
                # cache here (the ONLY event that stales baked args — content
                # republish keeps the buffer and stays replay-safe). Cold
                # event: size=layers is steady-state constant.
                self._writer_graph_state = None
            gpu = torch.empty((size,), device=device, dtype=torch.int64)
            self._rebuild_ptrs_gpu[name] = gpu
        return cpu, gpu

    def _release_rebuild_ptr_cpu_buffer(self, *, name: str, cpu: torch.Tensor) -> None:
        if cpu.numel() > 0 and cpu.is_pinned():
            free_pool = self._rebuild_ptrs_cpu_free.setdefault(name, [])
            if len(free_pool) < int(_REBUILD_PTRS_CPU_FREE_MAX_PER_NAME):
                free_pool.append(cpu)

    def _get_rebuild_ptr_gpu_if_signature_ready(
        self,
        *,
        name: str,
        device: torch.device,
        size: int,
        signature: Tuple[object, ...],
    ) -> Optional[torch.Tensor]:
        """Return a ready GPU pointer buffer without acquiring CPU staging."""
        if size <= 0:
            return None
        gpu = self._rebuild_ptrs_gpu.get(name)
        if gpu is None or gpu.numel() != size or gpu.device != device:
            return None
        sig = (
            tuple(signature),
            int(gpu.data_ptr()) if gpu.numel() > 0 else 0,
            int(gpu.numel()),
            str(gpu.device),
        )
        ready_event = self._rebuild_ptrs_ready_events.get(name)
        if self._rebuild_ptrs_signature.get(name) != sig or ready_event is None:
            return None
        if name in self._rebuild_ptrs_capture_wait_satisfied_names:
            try:
                if torch.cuda.is_current_stream_capturing():
                    return gpu
            except Exception:
                _log.warning("failed to query capture state for rebuild ptr hit", exc_info=True)
                raise
        torch.cuda.current_stream(device=gpu.device).wait_event(ready_event)
        return gpu

    def _prepare_rebuild_ptr_ready_events_for_capture(
        self,
        *,
        device: torch.device,
    ) -> Set[str]:
        """Wait pointer-buffer ready events before entering CUDA graph capture.

        CUDA graph capture cannot create a dependency on work from an
        uncaptured stream. The rebuild writer normally calls wait_event on
        every cached pointer-buffer hit, so graph capture must move those waits
        outside the capture region and then skip the duplicate wait inside.
        """
        waited: Set[str] = set()
        if not self._rebuild_ptrs_ready_events:
            self._rebuild_ptrs_capture_wait_satisfied_names = waited
            return waited
        stream = torch.cuda.current_stream(device=device)
        for name, ready_event in tuple(self._rebuild_ptrs_ready_events.items()):
            gpu = self._rebuild_ptrs_gpu.get(name)
            if gpu is None or gpu.device != device:
                continue
            stream.wait_event(ready_event)
            waited.add(str(name))
        self._rebuild_ptrs_capture_wait_satisfied_names = waited
        return waited

    def _clear_rebuild_ptr_capture_wait_satisfied(self) -> None:
        self._rebuild_ptrs_capture_wait_satisfied_names = set()

    @staticmethod
    def _rebuild_ptr_sig_diff_reason(old: object, new: tuple) -> str:
        """Name the FIRST differing component between two publish signatures.

        Forensics-only (writer-graph miss attribution): the dispatcher's
        pointer_rebuild_miss is the OR of six per-name publishes; naming which
        layer/field flipped is the difference between "every generation
        republishes by design" and "a semantic key leaked into the signature"
        (the selector-track key_norms verdict shape).
        """
        try:
            if old is None:
                return "cold"
            if not isinstance(old, tuple) or len(old) != len(new):
                return "sig_shape"
            if old[1] != new[1]:
                return "gpu_realloc"
            if old[2:] != new[2:]:
                return "numel_or_device"
            o_layers, n_layers = old[0], new[0]
            if len(o_layers) != len(n_layers):
                return f"layer_count:{len(o_layers)}->{len(n_layers)}"
            for i, (o, n) in enumerate(zip(o_layers, n_layers)):
                if o == n:
                    continue
                if o[0] != n[0]:
                    return f"L{i}:ptr"
                o_sig, n_sig = o[1], n[1]
                field_names = (
                    "cache_key",
                    "payload_layer_idx",
                    "state_layer_idx",
                    "compact_gen",
                    "residency_gen",
                    "residency_sig",
                )
                for j, fname in enumerate(field_names):
                    if (
                        j < len(o_sig)
                        and j < len(n_sig)
                        and o_sig[j] != n_sig[j]
                    ):
                        return (
                            f"L{i}:{fname}:{o_sig[j]!r}->{n_sig[j]!r}"[:200]
                        )
                return f"L{i}:layer_sig_len"
            return "eq_but_missed"
        except Exception:
            return "diff_error"

    def _publish_rebuild_ptr_buffer_if_needed(
        self,
        *,
        name: str,
        cpu: torch.Tensor,
        gpu: torch.Tensor,
        signature: Tuple[object, ...],
    ) -> bool:
        """Publish a CPU-built pointer signature to GPU only when storage changed."""
        sig = (
            tuple(signature),
            int(gpu.data_ptr()) if gpu.numel() > 0 else 0,
            int(gpu.numel()),
            str(gpu.device),
        )
        ready_event = self._rebuild_ptrs_ready_events.get(name)
        if self._rebuild_ptrs_signature.get(name) == sig and ready_event is not None:
            torch.cuda.current_stream(device=gpu.device).wait_event(ready_event)
            self._release_rebuild_ptr_cpu_buffer(name=name, cpu=cpu)
            return False
        _wg_dbg = os.environ.get("VLLM_SPARSE_SELECTOR_TOPK_GRAPH_DEBUG_LOG", "")
        if _wg_dbg:
            # miss 归因取证(诊断档默认关;与 writer dispatch 落盘同文件)。
            try:
                with open(_wg_dbg, "a") as _fh:
                    _fh.write(
                        f"{os.getpid()}\trepublish\t{name}\t"
                        f"{self._rebuild_ptr_sig_diff_reason(self._rebuild_ptrs_signature.get(name), sig)}\n"
                    )
            except OSError:
                pass
        stream = torch.cuda.current_stream(device=gpu.device)
        # [REBUILD-PTRS-OVERWRITE-WAR-FIX] P1:签名变化原位覆写持久 GPU 指针
        # 数组前,query-first 等 writer dispatch-done(R3 反向序同型)——
        # deferred replay 在另一流可能仍按旧指针数组 gather。稳态零成本。
        # [直调] 方法由 SelectorComputeMixin 恒定组合提供;软绑定缺失=守卫
        # 静默不跑,组合错误必须炸。
        self._wait_writer_dispatch_done_before_stable_overwrite()
        gpu.copy_(cpu, non_blocking=True)
        gpu.record_stream(stream)
        ready_event = torch.cuda.Event(enable_timing=False)
        ready_event.record(stream)
        self._source_ready_recorded_after_pointer_publish_total += 1
        self._rebuild_ptrs_signature[name] = sig
        self._rebuild_ptrs_ready_events[name] = ready_event
        self._rebuild_ptrs_cpu_pending.setdefault(name, deque()).append((cpu, ready_event))
        return True

    def _record_writer_pointer_lookup(
        self,
        *,
        pointer_rebuild: bool,
        cached_pointer_op: bool,
        vector_fallback: bool,
    ) -> None:
        self._writer_pointer_lookup_count_total += 1
        if pointer_rebuild:
            self._writer_pointer_rebuild_count_total += 1
        if cached_pointer_op:
            self._writer_cached_pointer_op_count_total += 1
        if vector_fallback:
            self._writer_vector_fallback_count_total += 1

    def _writer_pointer_telemetry_snapshot(self) -> Tuple[int, int, int, int, int]:
        return (
            int(self._writer_pointer_lookup_count_total),
            int(self._writer_pointer_rebuild_count_total),
            int(self._writer_cached_pointer_op_count_total),
            int(self._writer_vector_fallback_count_total),
            int(self._source_ready_recorded_after_pointer_publish_total),
        )

    def _writer_pointer_telemetry_delta(
        self,
        snapshot: Optional[Tuple[int, int, int, int, int]],
    ) -> Tuple[int, int, float, int, int, int]:
        if snapshot is None:
            return 0, 0, -1.0, 0, 0, 0
        lookup0, rebuild0, cached_op0, vector_fallback0, source_ready0 = snapshot
        lookups = max(0, int(self._writer_pointer_lookup_count_total) - int(lookup0))
        rebuilds = max(0, int(self._writer_pointer_rebuild_count_total) - int(rebuild0))
        cached_ops = max(
            0,
            int(self._writer_cached_pointer_op_count_total) - int(cached_op0),
        )
        vector_fallbacks = max(
            0,
            int(self._writer_vector_fallback_count_total) - int(vector_fallback0),
        )
        source_ready_records = max(
            0,
            int(self._source_ready_recorded_after_pointer_publish_total) - int(source_ready0),
        )
        if lookups <= 0:
            return rebuilds, lookups, -1.0, cached_ops, vector_fallbacks, source_ready_records
        hits = max(0, int(lookups) - int(rebuilds))
        return (
            rebuilds,
            lookups,
            float(hits) / float(lookups),
            cached_ops,
            vector_fallbacks,
            source_ready_records,
        )

    # ------------------------------------------------------------------
    # Capture layout tensor construction
    # ------------------------------------------------------------------

    @staticmethod
    def _rows_cache_key_rows_tuple(
        *,
        rows_tuple: Optional[Tuple[int, ...]],
        row_index: torch.Tensor,
    ) -> Tuple[object, ...]:
        """Build cache-key rows signature without forcing GPU row_index D2H tolist."""
        if rows_tuple is not None:
            return tuple(int(x) for x in rows_tuple)
        if row_index.device.type == "cpu":
            return tuple(int(x) for x in row_index.tolist())
        dev = row_index.device
        dev_index = int(dev.index) if dev.index is not None else -1
        return (
            "__row_index_tensor__",
            str(dev.type),
            dev_index,
            int(row_index.numel()),
            int(row_index.data_ptr()) if row_index.numel() > 0 else 0,
            int(getattr(row_index, "_version", 0)),
            int(id(row_index)),
        )

    def _ensure_capture_row_by_batch_row(
        self,
        *,
        layout: "StepCaptureLayout",
        step_context: "StepContext",
        device: torch.device,
    ) -> torch.Tensor:
        """构建/复用 batch_row->capture_row 的反向映射（GPU int32）。"""
        size = int(step_context.num_reqs)
        if size <= 0:
            active = torch.empty((0,), device=device, dtype=torch.int32)
            layout.active_capture_row_by_batch_row_i32 = active
            return active
        inv = layout.capture_row_by_batch_row_i32
        if inv is None or inv.device != device or inv.dtype != torch.int32 or inv.dim() != 1 or inv.numel() < size:
            inv = torch.full((size,), -1, device=device, dtype=torch.int32)
        else:
            # Reused buffers may outlive a larger previous batch. Clear the
            # whole tensor so stale tail rows cannot leak into later
            # layout/mapping validation when the current batch shrinks.
            inv.fill_(-1)
        row_tensor = layout.row_tensor.to(device=device, dtype=torch.long)
        pos = self._get_positions_i32(kv_len=int(row_tensor.numel()), device=device)
        inv.scatter_(0, row_tensor, pos)
        layout.capture_row_by_batch_row_i32 = inv
        layout.active_capture_row_by_batch_row_i32 = inv[:size]
        return layout.active_capture_row_by_batch_row_i32

    def _ensure_capture_layout_cpu_tensors(self, *, layout: "StepCaptureLayout", phase: str) -> None:
        """构建/复用 StepCaptureLayout 的 CPU 侧张量缓存。"""
        slots = layout.slot_list
        if phase == "refresh":
            epoch = int(getattr(self, "step_context_epoch", -1))
            handle_id = int(getattr(self, "_current_step_handle_id", -1))
            handle_generation = int(
                getattr(self, "_current_step_handle_generation", -1)
            )
            if (
                int(self._step_refresh_cpu_cache_epoch) != int(epoch)
                or int(getattr(self, "_step_refresh_cpu_cache_handle_id", -1))
                != handle_id
                or int(
                    getattr(self, "_step_refresh_cpu_cache_handle_generation", -1)
                )
                != handle_generation
            ):
                self._step_refresh_cpu_cache_epoch = int(epoch)
                self._step_refresh_cpu_cache_handle_id = handle_id
                self._step_refresh_cpu_cache_handle_generation = handle_generation
                self._step_refresh_slot_tensor_cpu.clear()
                self._step_refresh_seq_tensor_cpu.clear()
            slot_key = tuple(int(s) for s in slots)
            slot_tensor_cpu = self._step_refresh_slot_tensor_cpu.get(slot_key)
            if slot_tensor_cpu is None or slot_tensor_cpu.numel() != len(slot_key):
                slot_tensor_cpu = torch.tensor(slot_key, dtype=torch.long)
                self._step_refresh_slot_tensor_cpu[slot_key] = slot_tensor_cpu
            layout.slot_tensor_cpu = slot_tensor_cpu
            seq_cpu = layout.seq_lens_cpu
            if seq_cpu is not None:
                seq_key = (slot_key, tuple(int(s) for s in seq_cpu))
                seq_tensor_cpu = self._step_refresh_seq_tensor_cpu.get(seq_key)
                if seq_tensor_cpu is None or seq_tensor_cpu.numel() != len(seq_key[1]):
                    seq_tensor_cpu = torch.tensor(
                        [max(0, int(s)) for s in seq_key[1]], dtype=torch.long
                    )
                    self._step_refresh_seq_tensor_cpu[seq_key] = seq_tensor_cpu
                layout.seq_lens_tensor_cpu = seq_tensor_cpu
            return

        if layout.slot_tensor_cpu is None or layout.slot_tensor_cpu.numel() != len(slots):
            layout.slot_tensor_cpu = torch.tensor(slots, dtype=torch.long)
        seq_cpu = layout.seq_lens_cpu
        if seq_cpu is not None:
            if (
                layout.seq_lens_tensor_cpu is None
                or layout.seq_lens_tensor_cpu.numel() != len(seq_cpu)
            ):
                layout.seq_lens_tensor_cpu = torch.tensor(
                    [max(0, int(s)) for s in seq_cpu], dtype=torch.long
                )

    def _capture_scores_ptrs_for_rows(
        self,
        *,
        layout: "StepCaptureLayout",
        slot_in_chunk: int,
        row_index: torch.Tensor,
        rows_tuple: Optional[Tuple[int, ...]] = None,
        device: torch.device,
    ) -> torch.Tensor:
        """根据 StepCaptureLayout 生成 capture_scores 的 base_ptrs（int64 [N]）。"""
        capture_rows = self._capture_rows_i64_for_layout_rows(
            layout=layout, row_index=row_index, rows_tuple=rows_tuple, device=device
        )
        scores = layout.capture_scores
        base_ptr = int(scores.data_ptr())
        stride_chunk = int(scores.stride(0))
        stride_slot = int(scores.stride(1))
        elem_size = int(scores.element_size())
        base0 = int(base_ptr + int(slot_in_chunk) * stride_chunk * elem_size)
        stride_slot_bytes = int(stride_slot * elem_size)
        return capture_rows * int(stride_slot_bytes) + int(base0)

    def _capture_rows_i64_for_layout_rows(
        self,
        *,
        layout: "StepCaptureLayout",
        row_index: torch.Tensor,
        rows_tuple: Optional[Tuple[int, ...]] = None,
        device: torch.device,
    ) -> torch.Tensor:
        """按 (layout,row_index,rows_tuple,device) 复用/构建 capture_rows_i64。"""
        inv = layout.capture_row_by_batch_row_i32
        if inv is None or inv.device != device or inv.dtype != torch.int32 or inv.dim() != 1:
            raise RuntimeError("capture_row_by_batch_row_i32 missing for layout")
        rows_key = self._rows_cache_key_rows_tuple(rows_tuple=rows_tuple, row_index=row_index)
        dev_index = int(device.index) if device.index is not None else -1
        inv_ptr = int(inv.data_ptr()) if inv.numel() > 0 else 0
        # P0: cache key 绑定 step identity，避免同 epoch 重入时复用旧 capture_row 映射。
        cache_key = (
            int(layout.epoch),
            int(getattr(layout, "step_handle_id", -1)),
            int(getattr(layout, "step_handle_generation", -1)),
            int(layout.buf_id),
            int(inv_ptr),
            str(device.type),
            dev_index,
            rows_key,
        )
        capture_rows = self._capture_rows_cache.get(cache_key)
        cache_hit = (
            capture_rows is not None
            and capture_rows.device == device
            and capture_rows.dtype == torch.int64
        )
        if not cache_hit:
            capture_rows_i32 = inv.index_select(0, row_index.to(device=device))
            capture_rows = capture_rows_i32.to(dtype=torch.int64)
            if len(self._capture_rows_cache) > 128:
                # [CAPTURE-ROWS-CLEAR-UAF-GUARD 2026-07-07] 与 ROW-CACHE-CLEAR
                # 同族(P2-9):弃引用前三流守卫,消费者=writer ptrs 乘加(双上
                # 下文);冷事件零热开销。
                for _stale_t in self._capture_rows_cache.values():
                    self._uaf_guard_record_streams_before_discard(_stale_t)
                self._capture_rows_cache.clear()
            self._capture_rows_cache[cache_key] = capture_rows
        return capture_rows

    def _log_f_denoms_ptrs_for_rows(
        self,
        *,
        layout: "StepCaptureLayout",
        slot_in_chunk: int,
        row_index: torch.Tensor,
        rows_tuple: Optional[Tuple[int, ...]] = None,
        device: torch.device,
    ) -> torch.Tensor:
        """根据 StepCaptureLayout 生成 log_f_denoms 的 base_ptrs（int64 [N]）。"""
        capture_rows = self._capture_rows_i64_for_layout_rows(
            layout=layout, row_index=row_index, rows_tuple=rows_tuple, device=device
        )
        denoms = layout.log_f_denoms
        base_ptr = int(denoms.data_ptr())
        stride_chunk = int(denoms.stride(0))
        stride_slot = int(denoms.stride(1))
        elem_size = int(denoms.element_size())
        base0 = int(base_ptr + int(slot_in_chunk) * stride_chunk * elem_size)
        stride_slot_bytes = int(stride_slot * elem_size)
        return capture_rows * int(stride_slot_bytes) + int(base0)

    # ------------------------------------------------------------------
    # Layer -> capture slot mapping
    # ------------------------------------------------------------------

    @staticmethod
    def _map_global_layer_to_capture_slot(global_layer_index: int) -> Tuple[int, int, int]:
        """Map global layer index -> (chunk_id, buf_id, slot_in_chunk)."""
        if global_layer_index < 0:
            return 0, 0, 0
        chunk_id = global_layer_index // _CAPTURE_CHUNK
        buf_id = chunk_id % _CAPTURE_IN_FLIGHT
        slot_in_chunk = global_layer_index % _CAPTURE_CHUNK
        return chunk_id, buf_id, slot_in_chunk

    # ------------------------------------------------------------------
    # Ring-buffer lease lifecycle
    # ------------------------------------------------------------------

    def _reclaim_capture_ring_leases(self) -> int:
        if not self._capture_ring_retired_events:
            return 0
        ready_ids: Set[str] = set()
        pending: Deque[Tuple[str, torch.cuda.Event]] = deque()
        while self._capture_ring_retired_events:
            retire_id, done_evt = self._capture_ring_retired_events.popleft()
            try:
                done = bool(done_evt.query())
            except Exception:
                _log.warning("done_evt.query() failed for retire_id=%s", retire_id, exc_info=True)
                raise
            if done:
                ready_ids.add(str(retire_id))
            else:
                pending.append((str(retire_id), done_evt))
        self._capture_ring_retired_events = pending
        reclaimed = 0
        if ready_ids:
            reclaimed = int(self._capture_ring_lease_registry.reclaim(ready_event_ids=ready_ids))
        return reclaimed

    def _reclaim_retired_buffers(self) -> None:
        reclaimed = self._reclaim_capture_ring_leases()
        if reclaimed > 0:
            self._lease_stats["capture_ring_reclaimed"] += int(reclaimed)
        self._lease_stats["pending"] = int(self._capture_ring_lease_registry.pending_retired())

    def _ensure_capture_ring_active_lease(
        self,
        *,
        buf_id: int,
        epoch: int,
        min_capacity: int,
    ) -> BufferLease:
        buf = int(buf_id) % int(_CAPTURE_IN_FLIGHT)
        cur = self._capture_ring_active_lease_by_buf[buf]
        if cur is not None:
            return cur
        lease = self._capture_ring_lease_registry.acquire(
            kind=LeaseKind.CAPTURE_RING,
            slot=buf,
            min_capacity=max(1, int(min_capacity)),
            epoch=int(epoch),
        )
        self._capture_ring_active_lease_by_buf[buf] = lease
        return lease

    def _record_streams_for_release(
        self,
        *,
        capture_scores: "torch.Tensor",
        log_f_denoms: "torch.Tensor",
        device: torch.device,
    ) -> None:
        """Bind a to-be-dropped capture buffer's storage lifetime to the compute
        AND refresh (async selector-writer) streams so the caching allocator defers
        the free past both readers. Same async-UAF contract as _retire_capture_layout."""
        if capture_scores.device.type != "cuda":
            return
        cur = torch.cuda.current_stream(device=device)
        capture_scores.record_stream(cur)
        log_f_denoms.record_stream(cur)
        rs = getattr(self, "refresh_stream", None)
        if rs is not None:
            capture_scores.record_stream(rs)
            log_f_denoms.record_stream(rs)

    def _reclaim_retired_arena_buckets(self) -> int:
        """Drop held refs to pruned capture buffers whose buf_id chunk_done_evt has
        fired (async producer done) -> storage actually freed. Mirrors the ring's
        _reclaim_retired_buffers deferral; runs at step-prep (off hot path)."""
        q = getattr(self, "_arena_retired_buckets", None)
        if not q:
            return 0
        pending: Deque[Tuple[object, object]] = deque()
        reclaimed = 0
        while q:
            evt, layout = q.popleft()
            ready = True
            if evt is not None:
                try:
                    ready = bool(evt.query())
                except Exception:
                    # [GUARD-NO-SWALLOW] 假不就绪=退休桶永不释放（静默泄漏）。
                    _log.warning("arena retired-bucket chunk_done_evt.query() failed", exc_info=True)
                    raise
            if ready:
                reclaimed += 1  # drop ref -> free (record_stream already gated allocator reuse)
            else:
                pending.append((evt, layout))
        while pending:
            q.append(pending.popleft())
        return reclaimed

    def _prune_superseded_arena_buckets(self) -> int:
        """Keep only the high-water (max kv_max_bucket) prefill-capture buffer per
        (buf_id, slots_cap, num_heads, dtype, device, phase, intent); smaller buffers
        can never be selected by bind() again (it picks the smallest fit >= need, and
        the high-water one always fits), so release them async-safely. Off hot path;
        never runs during graph capture. Keeps the arena bounded to the actual
        high-water (no accumulation)."""
        # [GUARD-NO-SWALLOW] capture 态查询失败若吞掉则会在 capture 内 prune/free。
        if torch.cuda.is_available() and _is_stream_capturing_or_raise(
            stage="prune_superseded_arena_buckets"
        ):
            return 0
        arena = getattr(self, "prefill_capture_meta_arena", None)
        if arena is None:
            return 0
        from patches.sparse_types import StepCaptureLayout
        best_by_group: Dict[Tuple[object, ...], object] = {}
        for key, layout in arena.layouts_by_key.items():
            if not isinstance(layout, StepCaptureLayout):
                continue
            group = (
                int(key.buf_id), int(key.slots_cap_bucket), int(key.num_heads),
                str(key.dtype), str(key.device), key.phase, key.intent,
            )
            cur = best_by_group.get(group)
            if cur is None or int(key.kv_max_bucket) > int(cur.kv_max_bucket):
                best_by_group[group] = key
        keep = set(best_by_group.values())
        drop = [
            k for k, lay in list(arena.layouts_by_key.items())
            if isinstance(lay, StepCaptureLayout) and k not in keep
        ]
        if not drop:
            self._reclaim_retired_arena_buckets()
            return 0
        q = getattr(self, "_arena_retired_buckets", None)
        if q is None:
            q = deque()
            self._arena_retired_buckets = q
        cde = getattr(self, "chunk_done_evt", None)
        pruned = 0
        for key in drop:
            layout = arena.layouts_by_key.pop(key, None)
            if layout is None:
                continue
            br = arena.bucket_bytes_by_key.pop(key, None)
            if br is not None:
                arena.metrics.arena_reserved_bytes = max(
                    0, int(arena.metrics.arena_reserved_bytes) - int(br.total_bytes)
                )
                arena.metrics.arena_expansion_bytes = max(
                    0, int(arena.metrics.arena_expansion_bytes) - int(br.total_bytes)
                )
            self._record_streams_for_release(
                capture_scores=layout.capture_scores,
                log_f_denoms=layout.log_f_denoms,
                device=layout.capture_scores.device,
            )
            evt = None
            bid = int(key.buf_id)
            if cde is not None and 0 <= bid < len(cde):
                evt = cde[bid]
            q.append((evt, layout))
            pruned += 1
        arena.metrics.arena_bucket_count = int(
            sum(1 for lay in arena.layouts_by_key.values() if isinstance(lay, StepCaptureLayout))
        )
        arena.metrics.arena_budget_exceeded = (
            int(arena.metrics.arena_reserved_bytes) > int(arena.budget_bytes)
        )
        self._reclaim_retired_arena_buckets()
        return pruned

    def _retire_capture_layout(
        self,
        *,
        buf_id: int,
        layout: "StepCaptureLayout",
        device: torch.device,
    ) -> None:
        self._reclaim_retired_buffers()
        buf = int(buf_id) % int(_CAPTURE_IN_FLIGHT)
        # 在替换 ring layout 前绑定旧 storage 生命周期，避免异步路径 UAF。
        if layout.capture_scores.device.type == "cuda":
            try:
                cur = torch.cuda.current_stream(device=device)
                layout.capture_scores.record_stream(cur)
                layout.log_f_denoms.record_stream(cur)
            except Exception:
                _log.warning("record_stream(current_stream) failed for buf_id=%s", buf_id, exc_info=True)
                raise
            rs = self.refresh_stream
            if rs is not None:
                try:
                    layout.capture_scores.record_stream(rs)
                    layout.log_f_denoms.record_stream(rs)
                except Exception:
                    _log.warning("record_stream(refresh_stream) failed for buf_id=%s", buf_id, exc_info=True)
                    raise

        active = self._capture_ring_active_lease_by_buf[buf]
        if active is None:
            return
        retire_id = f"capture-ring-retire-{buf}-{active.generation}-{time.time_ns()}"
        try:
            self._capture_ring_lease_registry.retire(lease=active, event_id=retire_id)
        except Exception:
            _log.warning("lease_registry.retire() failed for buf=%d retire_id=%s", buf, retire_id, exc_info=True)
            raise
        self._lease_stats["capture_ring_retired"] += 1
        self._capture_ring_active_lease_by_buf[buf] = None

        if device.type != "cuda":
            self._capture_ring_lease_registry.reclaim(ready_event_ids={retire_id})
            return

        if 0 <= buf < len(self.chunk_done_evt):
            done_evt = self.chunk_done_evt[buf]
            if done_evt is not None:
                self._capture_ring_retired_events.append((retire_id, done_evt))
                return
        self._capture_ring_lease_registry.reclaim(ready_event_ids={retire_id})
