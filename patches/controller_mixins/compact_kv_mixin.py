"""
patches/controller_mixins/compact_kv_mixin.py — Compact KV buffer allocation, capacity management, and copy operations.

OWNS:
  - _init_compact_kv_state(): compact KV state initialization
  - _compact_threshold_tokens / _compact_stride_tokens(): capacity math helpers
  - _reset_compact_slot(): per-slot compact buffer teardown
  - _zero_compact_pad_if_needed / _check_pad_cleanup_needed / _execute_pad_cleanup(): padding management
  - _ensure_compact_capacity(): grow-or-allocate compact KV storage

DEPENDS_ON:
  - patches.buffer_lease_protocol (LeaseKind)
  - config, _allocator_backend, refresh_stream, step_context_epoch (cross-mixin state)

ENTRY_POINTS:
  - _init_compact_kv_state(): called from VLLMSparseController.__init__
"""
from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING, Optional, Tuple

import torch

_log = logging.getLogger(__name__)
_COMPACT_GROW_DEBUG = os.environ.get("VLLM_SPARSE_COMPACT_GROW_DEBUG", "0") == "1"

from patches.buffer_lease_protocol import LeaseKind
from patches.fa_sparse_runtime.compact_recent_alignment import (
    compact_recent_effective_k_head,
    compact_slot_offset_tokens,
)
from patches.sparse_constants import compact_gen_count

if TYPE_CHECKING:
    from patches.layer_state import LayerState


class CompactKVMixin:
    """Compact KV buffer operations for VLLMSparseController.

    All compact-KV state is initialised via ``_init_compact_kv_state()``
    which must be called from the controller's ``__init__``.
    """

    _MIXIN_REQUIRES: tuple = ()

    # ------------------------------------------------------------------
    # State initialisation
    # ------------------------------------------------------------------

    def _init_compact_kv_state(self) -> None:
        return None



    # ------------------------------------------------------------------
    # Compact threshold / stride
    # ------------------------------------------------------------------

    def _compact_threshold_tokens(self) -> int:
        """返回启动 compact 的最小上下文长度（sink+persist_cap+recent_cap）。"""
        cfg = self.config
        if cfg is None:
            return 0
        semantic_snapshot = self._get_step_semantic_snapshot()
        sink_cap = max(0, int(semantic_snapshot.sink_tokens))
        recent_cap = max(0, int(semantic_snapshot.recent_tokens))
        persist_cap = 0
        k_head = cfg.alpha_fair.k_head
        if k_head is not None:
            persist_cap = max(persist_cap, int(k_head))
        if cfg.k_max is not None:
            persist_cap = max(persist_cap, int(cfg.k_max))
        if cfg.k_min is not None:
            persist_cap = max(persist_cap, int(cfg.k_min))
        persist_cap = compact_recent_effective_k_head(
            k_head=persist_cap,
            sink_tokens=sink_cap,
            attn_mode=str(getattr(cfg, "attn_mode", "compact_recent")),
        )
        return sink_cap + persist_cap + recent_cap

    def _compact_stride_tokens(self, block_size: int) -> int:
        """固定 stride 的 compact 容量（仅 sink+persist，recent 不入 compact）。"""
        cfg = self.config
        if cfg is None:
            return 0
        semantic_snapshot = self._get_step_semantic_snapshot()
        sink_cap = max(0, int(semantic_snapshot.sink_tokens))
        persist_cap = 0
        k_head = cfg.alpha_fair.k_head
        if k_head is not None:
            persist_cap = max(persist_cap, int(k_head))
        if cfg.k_max is not None:
            persist_cap = max(persist_cap, int(cfg.k_max))
        if cfg.k_min is not None:
            persist_cap = max(persist_cap, int(cfg.k_min))
        persist_cap = compact_recent_effective_k_head(
            k_head=persist_cap,
            sink_tokens=sink_cap,
            attn_mode=str(getattr(cfg, "attn_mode", "compact_recent")),
        )
        tokens = sink_cap + persist_cap
        if block_size > 0 and tokens > 0:
            rem = tokens % block_size
            if rem != 0:
                tokens += block_size - rem
        return tokens

    # ------------------------------------------------------------------
    # Compact KV helpers
    # ------------------------------------------------------------------

    def _reset_compact_slot(self, state: "LayerState", slot: int) -> None:
        if slot < 0 or slot >= state.batch_size:
            return
        empty_k = torch.empty(0, device=state.device)
        empty_v = torch.empty(0, device=state.device)
        empty_p = torch.empty(0, device=state.device, dtype=torch.int32)
        if slot < len(state.compact_k):
            state.compact_k[slot] = empty_k
            state.compact_v[slot] = empty_v
            state.compact_pos[slot] = empty_p
            state.compact_capacity[slot] = 0
            state.compact_offset_tokens[slot] = 0
            state.compact_sink_len[slot] = 0
            state.compact_persist_len[slot] = 0
            state.compact_kv_len[slot] = 0
            if slot < len(state.compact_pad_zeroed_len):
                state.compact_pad_zeroed_len[slot] = -1
            # 该 slot 的 arena view 被重置为空，前缀绑定标记需要降级。
            state.compact_views_bound_slots = min(
                int(getattr(state, "compact_views_bound_slots", 0)),
                int(slot),
            )
            state.bump_compact_meta_epoch()

    def _check_pad_cleanup_needed(
        self,
        state: "LayerState",
        slot: int,
        kv_len: int,
        stride_tokens: int,
    ) -> Optional[Tuple[int, int]]:
        """检查是否需要 pad 清理，返回 (pad_start, pad_end) 或 None。

        注意：此函数会更新 compact_pad_zeroed_len，但不执行 GPU 操作。
        """
        if slot < 0 or slot >= len(state.compact_k):
            return None
        if stride_tokens <= 0:
            return None
        if slot >= len(state.compact_pad_zeroed_len):
            return None
        if kv_len >= stride_tokens:
            state.compact_pad_zeroed_len[slot] = stride_tokens
            return None
        prev = state.compact_pad_zeroed_len[slot]
        if prev == kv_len:
            return None
        if prev < 0:
            pad_start = kv_len
            pad_end = stride_tokens
        elif kv_len < prev:
            pad_start = kv_len
            pad_end = prev
        else:
            state.compact_pad_zeroed_len[slot] = kv_len
            return None
        if pad_end <= pad_start:
            state.compact_pad_zeroed_len[slot] = kv_len
            return None
        state.compact_pad_zeroed_len[slot] = kv_len
        return (pad_start, pad_end)

    def _execute_pad_cleanup(
        self,
        state: "LayerState",
        slot: int,
        pad_start: int,
        pad_end: int,
        *,
        write_offset_tokens: "int | None" = None,
    ) -> None:
        """执行单个 slot 的 pad 清理 GPU 操作。

        [DUAL-GEN-L2a] write_offset_tokens 非 None 时按 writer 半区绝对偏移
        直址 arena(读侧视图仍指旧代,不可用于写代清理);None=历史视图路径。
        """
        if write_offset_tokens is None:
            state.compact_k[slot][pad_start:pad_end].zero_()
            state.compact_v[slot][pad_start:pad_end].zero_()
            state.compact_pos[slot][:, pad_start:pad_end].fill_(-1)
            return
        wo = int(write_offset_tokens)
        state.compact_arena_k[wo + pad_start : wo + pad_end].zero_()
        state.compact_arena_v[wo + pad_start : wo + pad_end].zero_()
        state.compact_arena_pos[:, wo + pad_start : wo + pad_end].fill_(-1)

    def _ensure_compact_capacity(
        self,
        state: "LayerState",
        slot: int,
        capacity_tokens: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        block_size: int,
    ) -> None:
        """固定 stride 的 compact arena 分配/扩容（按 slot 切片）。"""
        if slot < 0:
            return
        if block_size <= 0:
            raise RuntimeError("block_size must be positive for compact stride")
        if state.compact_page_residency is not None:
            CompactKVMixin._ensure_residency_compact_capacity(
                state=state,
                slot=slot,
                capacity_tokens=capacity_tokens,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                dtype=dtype,
                block_size=block_size,
            )
            return

        stride_tokens = state.compact_stride_tokens
        if stride_tokens <= 0:
            stride_tokens = self._compact_stride_tokens(block_size)
            if stride_tokens <= 0:
                raise RuntimeError("compact stride tokens must be positive")
            state.compact_stride_tokens = stride_tokens
            state.compact_stride_blocks = stride_tokens // block_size
            state.compact_stride_block_size = block_size
        elif state.compact_stride_block_size != block_size:
            raise RuntimeError("compact stride block_size mismatch")

        if capacity_tokens > stride_tokens:
            raise RuntimeError(
                f"compact capacity {capacity_tokens} exceeds stride {stride_tokens}"
            )

        # 确保列表长度充足
        while len(state.compact_k) <= slot:
            state.compact_k.append(torch.empty(0, device=state.device))
            state.compact_v.append(torch.empty(0, device=state.device))
            state.compact_pos.append(torch.empty(0, device=state.device, dtype=torch.int32))
            state.compact_capacity.append(0)
            state.compact_offset_tokens.append(0)
            state.compact_sink_len.append(0)
            state.compact_persist_len.append(0)
            state.compact_kv_len.append(0)
            state.compact_pad_zeroed_len.append(-1)

        required_slots = max(state.batch_size, slot + 1)
        total_tokens = required_slots * stride_tokens
        arena_reallocated = False
        if state.compact_arena_capacity_tokens < total_tokens or state.compact_arena_k.numel() == 0:
            if _COMPACT_GROW_DEBUG:
                _log.warning(
                    "compact arena grow: layer=%d slot=%d state_batch_size=%d required_slots=%d "
                    "stride_tokens=%d old_capacity_tokens=%d new_capacity_tokens=%d "
                    "num_kv_heads=%d head_dim=%d dtype=%s",
                    int(getattr(state, "layer_index", -1)),
                    int(slot),
                    int(state.batch_size),
                    int(required_slots),
                    int(stride_tokens),
                    int(state.compact_arena_capacity_tokens),
                    int(total_tokens),
                    int(num_kv_heads),
                    int(head_dim),
                    str(dtype),
                )
            new_arena_k = self._allocator_backend.alloc_empty(
                (total_tokens, num_kv_heads, head_dim),
                device=state.device,
                dtype=dtype,
            )
            new_arena_v = self._allocator_backend.alloc_empty(
                (total_tokens, num_kv_heads, head_dim),
                device=state.device,
                dtype=dtype,
            )
            new_arena_pos = self._allocator_backend.alloc_full(
                (num_kv_heads, total_tokens),
                fill_value=-1,
                device=state.device,
                dtype=torch.int32,
            )
            if state.compact_arena_k.numel() > 0:
                # F6: 旧 arena 可能仍被 refresh_stream 访问，record_stream 防止 UAF
                _rs = self.refresh_stream
                try:
                    _cur = torch.cuda.current_stream(device=state.device)
                except Exception:
                    # CPU device has no CUDA stream — expected in test/non-CUDA paths
                    _cur = None
                if _cur is not None:
                    state.compact_arena_k.record_stream(_cur)
                    state.compact_arena_v.record_stream(_cur)
                    state.compact_arena_pos.record_stream(_cur)
                if _rs is not None:
                    state.compact_arena_k.record_stream(_rs)
                    state.compact_arena_v.record_stream(_rs)
                    state.compact_arena_pos.record_stream(_rs)
                copy_tokens = min(state.compact_arena_k.shape[0], total_tokens)
                new_arena_k[:copy_tokens].copy_(state.compact_arena_k[:copy_tokens])
                new_arena_v[:copy_tokens].copy_(state.compact_arena_v[:copy_tokens])
                new_arena_pos[:, :copy_tokens].copy_(state.compact_arena_pos[:, :copy_tokens])
            # 先 retire 旧 lease（失败时 arena 不变，状态完全一致）
            old_lease = state._compact_active_lease
            if old_lease is not None:
                retire_id = f"compact-retire-{old_lease.generation}-{time.time_ns()}"
                state._compact_lease_registry.retire(lease=old_lease, event_id=retire_id)
            # retire 成功后替换 arena
            state.compact_arena_k = new_arena_k
            state.compact_arena_v = new_arena_v
            state.compact_arena_pos = new_arena_pos
            state.compact_arena_capacity_tokens = total_tokens
            new_lease = state._compact_lease_registry.acquire(
                kind=LeaseKind.COMPACT,
                slot=0,
                min_capacity=int(total_tokens * max(1, num_kv_heads) * max(1, head_dim)),
                epoch=int(getattr(self, "step_context_epoch", -1)),
            )
            state._compact_active_lease = new_lease
            state.compact_generation = int(new_lease.generation)
            state.bump_compact_meta_epoch()
            arena_reallocated = True

        # 热路径快路径：arena 未变更且视图前缀已绑定时，跳过全量 rebind。
        if (
            not arena_reallocated
            and int(getattr(state, "compact_views_bound_slots", 0)) >= int(required_slots)
            and len(state.compact_k) >= required_slots
            and len(state.compact_v) >= required_slots
            and len(state.compact_pos) >= required_slots
            and len(state.compact_capacity) >= required_slots
            and len(state.compact_offset_tokens) >= required_slots
            and int(state.compact_arena_capacity_tokens) >= int(total_tokens)
        ):
            return

        # 固定 stride：slot -> offset / view([DUAL-GEN-L1] 经 read_gen 公式,
        # gen 恒 0 时与历史 idx*stride 逐位同址;gen_stride 待 L2 扩容启用)
        for idx in range(required_slots):
            if idx >= len(state.compact_read_gen):
                state.compact_read_gen.append(0)
            off = compact_slot_offset_tokens(
                slot=idx,
                stride_tokens=stride_tokens,
                read_gen=int(state.compact_read_gen[idx]),
                gen_stride_tokens=0,
            )
            state.compact_capacity[idx] = stride_tokens
            state.compact_offset_tokens[idx] = off
            state.compact_k[idx] = state.compact_arena_k.narrow(0, off, stride_tokens)
            state.compact_v[idx] = state.compact_arena_v.narrow(0, off, stride_tokens)
            state.compact_pos[idx] = state.compact_arena_pos.narrow(1, off, stride_tokens)
            if idx >= len(state.compact_pad_zeroed_len):
                state.compact_pad_zeroed_len.append(-1)
        state.compact_views_bound_slots = max(
            int(getattr(state, "compact_views_bound_slots", 0)),
            int(required_slots),
        )

    @staticmethod
    def _ensure_residency_compact_capacity(
        *,
        state: "LayerState",
        slot: int,
        capacity_tokens: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        block_size: int,
    ) -> None:
        residency = state.compact_page_residency
        if residency is None:
            raise RuntimeError("compact page residency is not bound")
        lease = residency.lease
        capacity_slots = int(lease.max_live_sparse_slots)
        if slot < 0 or slot >= capacity_slots:
            raise RuntimeError(
                f"compact page slot {slot} is out of range for capacity {capacity_slots}"
            )
        if (
            block_size != int(residency.page_size)
            or block_size != int(lease.manager_block_size)
        ):
            raise RuntimeError(
                "compact page block_size mismatch: "
                f"requested={block_size}, residency={residency.page_size}, "
                f"lease={lease.manager_block_size}"
            )
        if (
            num_kv_heads != int(residency.num_heads)
            or num_kv_heads != int(state.num_kv_heads)
        ):
            raise RuntimeError(
                "compact page num_kv_heads mismatch: "
                f"requested={num_kv_heads}, residency={residency.num_heads}, "
                f"state={state.num_kv_heads}"
            )
        state_head_dim = getattr(state, "head_dim", None)
        if head_dim != int(residency.head_dim) or (
            state_head_dim is not None and head_dim != int(state_head_dim)
        ):
            raise RuntimeError(
                "compact page head_dim mismatch: "
                f"requested={head_dim}, residency={residency.head_dim}, "
                f"state={state_head_dim}"
            )

        stride_tokens = int(lease.compact_blocks_per_slot) * block_size
        if int(state.compact_stride_tokens) != stride_tokens:
            raise RuntimeError(
                "compact page stride_tokens mismatch: "
                f"state={state.compact_stride_tokens}, expected={stride_tokens}"
            )
        if int(state.compact_stride_blocks) != int(lease.compact_blocks_per_slot):
            raise RuntimeError(
                "compact page stride_blocks mismatch: "
                f"state={state.compact_stride_blocks}, expected={lease.compact_blocks_per_slot}"
            )
        if int(state.compact_stride_block_size) != block_size:
            raise RuntimeError(
                "compact page stride block_size mismatch: "
                f"state={state.compact_stride_block_size}, requested={block_size}"
            )
        if capacity_tokens > stride_tokens:
            raise RuntimeError(
                f"compact capacity {capacity_tokens} exceeds page-backed stride {stride_tokens}"
            )

        required_slots = max(int(state.batch_size), slot + 1)
        if required_slots > capacity_slots:
            raise RuntimeError(
                f"compact page required slots {required_slots} exceed capacity {capacity_slots}"
            )
        # [DUAL-GEN-L2a] arena 物理域含双代半区(lease reserve 已 ×factor,
        # L2a-1);gen_stride=单半区跨度(slots×stride),开关关时 factor=1、
        # gen_stride 不参与 gen0 偏移,全部逐位同历史。
        gen_stride_tokens = capacity_slots * stride_tokens
        total_tokens = gen_stride_tokens * compact_gen_count()
        required_tokens = required_slots * stride_tokens
        arena_k = state.compact_arena_k
        arena_v = state.compact_arena_v
        if not isinstance(arena_k, torch.Tensor) or not isinstance(arena_v, torch.Tensor):
            raise RuntimeError("compact page arena K/V views must be tensors")
        expected_shape = (total_tokens, num_kv_heads, head_dim)
        for name, tensor in (("K", arena_k), ("V", arena_v)):
            if tuple(tensor.shape) != expected_shape:
                raise RuntimeError(
                    f"compact page arena {name} shape mismatch: "
                    f"actual={tuple(tensor.shape)}, expected={expected_shape}"
                )
            if tensor.dtype != dtype:
                raise RuntimeError(
                    f"compact page arena {name} dtype mismatch: "
                    f"actual={tensor.dtype}, expected={dtype}"
                )
            if tensor.device != state.device:
                raise RuntimeError(
                    f"compact page arena {name} device mismatch: "
                    f"actual={tensor.device}, expected={state.device}"
                )
            if tensor.stride() != (num_kv_heads * head_dim, head_dim, 1):
                raise RuntimeError(
                    f"compact page arena {name} must be token-major viewable without copy; "
                    f"stride={tuple(tensor.stride())}"
                )
        state_dtype = getattr(state, "kv_cache_dtype", None)
        if state_dtype is not None and state_dtype != dtype:
            raise RuntimeError(
                f"compact page state kv_cache_dtype mismatch: state={state_dtype}, requested={dtype}"
            )

        metadata = state.compact_metadata_buffers
        expected_storage_ptr = int(getattr(metadata, "kv_storage_data_ptr", 0) or 0)
        if expected_storage_ptr <= 0:
            raise RuntimeError("compact page metadata is missing KV storage identity")
        for name, tensor in (("K", arena_k), ("V", arena_v)):
            if CompactKVMixin._storage_data_ptr(tensor) != expected_storage_ptr:
                raise RuntimeError(
                    f"compact page arena {name} is not backed by reserved page KV storage"
                )
        expected_k_data_ptr = int(getattr(metadata, "compact_k_data_ptr", 0) or 0)
        expected_v_data_ptr = int(getattr(metadata, "compact_v_data_ptr", 0) or 0)
        if expected_k_data_ptr <= 0 or expected_v_data_ptr <= 0:
            raise RuntimeError("compact page metadata is missing reserved K/V view identity")
        if int(arena_k.data_ptr()) != expected_k_data_ptr:
            raise RuntimeError("compact page arena K is not the reserved K view")
        if int(arena_v.data_ptr()) != expected_v_data_ptr:
            raise RuntimeError("compact page arena V is not the reserved V view")
        if int(state.compact_arena_capacity_tokens) < total_tokens:
            raise RuntimeError(
                "compact page arena capacity is smaller than reserved capacity: "
                f"actual={state.compact_arena_capacity_tokens}, expected={total_tokens}"
            )
        if int(state.compact_arena_capacity_tokens) < required_tokens:
            raise RuntimeError(
                "compact page arena capacity is smaller than required slots: "
                f"actual={state.compact_arena_capacity_tokens}, required={required_tokens}"
            )

        expected_pos_shape = (num_kv_heads, total_tokens)
        pos = state.compact_arena_pos
        pos_reallocated = False
        if (
            not isinstance(pos, torch.Tensor)
            or tuple(pos.shape) != expected_pos_shape
            or pos.dtype != torch.int32
            or pos.device != state.device
        ):
            pos = torch.full(
                expected_pos_shape,
                -1,
                device=state.device,
                dtype=torch.int32,
            )
            state.compact_arena_pos = pos
            if metadata is not None:
                metadata.compact_arena_pos = pos
                metadata.capacity_slots = capacity_slots
                metadata.compact_blocks_per_slot = int(lease.compact_blocks_per_slot)
                metadata.page_size = block_size
            state.compact_layout_token_positions_view = None
            state._compact_layout_token_pos_view_key = None
            state.compact_views_bound_slots = 0
            pos_reallocated = True
            state.bump_compact_meta_epoch()

        while len(state.compact_k) < required_slots:
            state.compact_k.append(torch.empty(0, device=state.device))
            state.compact_v.append(torch.empty(0, device=state.device))
            state.compact_pos.append(torch.empty(0, device=state.device, dtype=torch.int32))
            state.compact_capacity.append(0)
            state.compact_offset_tokens.append(0)
            state.compact_sink_len.append(0)
            state.compact_persist_len.append(0)
            state.compact_kv_len.append(0)
            state.compact_pad_zeroed_len.append(-1)

        if (
            not pos_reallocated
            and int(getattr(state, "compact_views_bound_slots", 0)) >= required_slots
            and len(state.compact_capacity) >= required_slots
            and all(
                int(state.compact_capacity[idx]) == stride_tokens
                for idx in range(required_slots)
            )
        ):
            return

        for idx in range(required_slots):
            if idx >= len(state.compact_read_gen):
                state.compact_read_gen.append(0)
            off = compact_slot_offset_tokens(
                slot=idx,
                stride_tokens=stride_tokens,
                read_gen=int(state.compact_read_gen[idx]),
                gen_stride_tokens=gen_stride_tokens,
            )
            state.compact_capacity[idx] = stride_tokens
            state.compact_offset_tokens[idx] = off
            state.compact_k[idx] = arena_k.narrow(0, off, stride_tokens)
            state.compact_v[idx] = arena_v.narrow(0, off, stride_tokens)
            state.compact_pos[idx] = state.compact_arena_pos.narrow(1, off, stride_tokens)
            if idx >= len(state.compact_pad_zeroed_len):
                state.compact_pad_zeroed_len.append(-1)
        state.compact_views_bound_slots = max(
            int(getattr(state, "compact_views_bound_slots", 0)),
            int(required_slots),
        )

    @staticmethod
    def _storage_data_ptr(tensor: torch.Tensor) -> int:
        if hasattr(tensor, "untyped_storage"):
            return int(tensor.untyped_storage().data_ptr())
        return int(tensor.storage().data_ptr())



