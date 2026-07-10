"""
LayerState: per-layer mutable state for the sparse attention engine.

Extracted from vllm_sparse_patch.py to reduce main-file complexity.
No circular dependency with vllm_sparse_patch.py.
"""
from __future__ import annotations

import heapq
import json
import logging
import os
import time
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch

from patches.buffer_allocator_backends import AllocatorBackend, resolve_allocator_backend
from patches.buffer_lease_protocol import BufferLease, BufferLeaseRegistry, LeaseKind
from patches.page_kv_residency import CompactMetadataBuffers, CompactPageResidency
from patches.sparse_cache import _get_cached_empty_tensor
from patches.sparse_constants import _DYNAMIC_ENV
from patches.sparse_constants import _FREE_SLOT_ID, _is_free_slot_id
from patches.selector_runtime.entry import run_selector_step

_log = logging.getLogger(__name__)
_SLOT_GROW_DEBUG = os.environ.get("VLLM_SPARSE_SLOT_GROW_DEBUG", "0") == "1"
# FIX(bs8 key_norms cross-stream realloc race): floor the key_norms arena stride so the arena
# is allocated once at full size and NEVER reallocs on the async prefill hot path (the realloc
# frees the slot-keyed arena while a refresh kernel still uses it -> stale base ptr -> illegal
# address). Default covers up to 262k context; set to 0 to restore old grow-on-demand behavior.
_KEY_NORMS_STRIDE_FLOOR = int(os.environ.get("VLLM_SPARSE_KEY_NORMS_STRIDE_FLOOR", "262144") or "262144")

_U64_MASK = (1 << 64) - 1
_FNV64_OFFSET_BASIS = 1469598103934665603
_FNV64_PRIME = 1099511628211


# [F2] trace path 进程内不变：import-time 缓存；pytest/显式动态档 live 读。
_SELECTED_READY_TRACE_PATH_CACHED = os.environ.get(
    "VLLM_SPARSE_SELECTED_READY_TRACE_LOG", ""
).strip()


def _selected_ready_trace_path() -> str:
    if _DYNAMIC_ENV:
        return os.environ.get("VLLM_SPARSE_SELECTED_READY_TRACE_LOG", "").strip()
    return _SELECTED_READY_TRACE_PATH_CACHED


def _append_selected_ready_trace(event: dict[str, object]) -> None:
    path = _selected_ready_trace_path()
    if not path:
        return
    record = dict(event)
    record.setdefault("pid", os.getpid())
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _mix_u64(sig: int, value: int) -> int:
    sig_u64 = int(sig) & _U64_MASK
    sig_u64 ^= int(value) & _U64_MASK
    sig_u64 = (sig_u64 * _FNV64_PRIME) & _U64_MASK
    sig_u64 ^= (sig_u64 >> 32)
    return sig_u64 & _U64_MASK


# rid→hash64 纯函数 memo：签名重算占 align slow-path 的 ~85%（逐字节 FNV），
# rid 字符串不可变故结果恒同；容量上限防 serve 长跑无界增长（清空仅触发重算）。
_STR_HASH64_MEMO: Dict[str, int] = {}
_STR_HASH64_MEMO_CAP = 65536


def _stable_str_hash64(value: str) -> int:
    cached = _STR_HASH64_MEMO.get(value)
    if cached is not None:
        return cached
    h = _FNV64_OFFSET_BASIS
    for byte in value.encode("utf-8"):
        h ^= int(byte)
        h = (h * _FNV64_PRIME) & _U64_MASK
    h &= _U64_MASK
    if len(_STR_HASH64_MEMO) >= _STR_HASH64_MEMO_CAP:
        _STR_HASH64_MEMO.clear()
    _STR_HASH64_MEMO[value] = h
    return h


def _stable_slot_signature64(
    request_id_to_slot: Dict[str, int],
    active_request_ids: Optional[Tuple[str, ...]],
) -> int:
    active_ids = active_request_ids or tuple()
    sig = _mix_u64(_FNV64_OFFSET_BASIS, len(active_ids))
    for idx, rid in enumerate(active_ids):
        slot = int(request_id_to_slot.get(rid, -1))
        sig = _mix_u64(sig, idx + 1)
        sig = _mix_u64(sig, _stable_str_hash64(str(rid)))
        sig = _mix_u64(sig, slot + 2)
    return sig & _U64_MASK


class LayerState:
    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        device: torch.device,
    ) -> None:
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.device = device
        self._allocator_backend: AllocatorBackend = resolve_allocator_backend()
        # vLLM 的真实层序号（0..L-1），由 controller 在注册层时填充；用于 refresh layer-group gating。
        self.layer_index: int = -1
        # layer_index 缓存 epoch：与 controller._layer_index_cache_epoch 对齐
        self.layer_index_epoch: int = -1
        # per-layer capture slot 缓存（减少 per-layer 映射开销）
        self.capture_chunk_id: int = -1
        self.capture_buf_id: int = -1
        self.capture_slot_in_chunk: int = -1
        # 由首次进入 attention 时设置（用于 step_decode_data 的正确 shape/dtype）
        self.head_dim: Optional[int] = None
        self.kv_cache_dtype: Optional[torch.dtype] = None
        # batch_size 表示当前 LayerState 中已分配的 slot 数（即最大 slot+1），
        # 而不是“本次 kernel 调用的 batch 大小”。slot 与 request 绑定，生命周期等于 request。
        self.batch_size: int = 0
        # Compact KV 缓冲区（按 slot 存储），避免 decode 阶段每步重建。
        # 每个 slot 一条记录，token 轴共享所有 kv 头。
        self.compact_k: List[torch.Tensor] = []           # [tokens, num_kv_heads, head_dim]
        self.compact_v: List[torch.Tensor] = []
        self.compact_pos: List[torch.Tensor] = []         # [num_kv_heads, tokens]
        self.compact_capacity: List[int] = []
        self.compact_offset_tokens: List[int] = []        # token offset into arena (block 对齐)
        # [DUAL-GEN-L1] per-slot 当前读代(0/1,分半布局:gen1 半区基址=gen_stride)。
        # L1 垫层恒 0(offset 公式与历史逐位同址);L2 切代时由 meta commit 翻转。
        self.compact_read_gen: List[int] = []
        self.compact_sink_len: List[int] = []
        self.compact_persist_len: List[int] = []
        self.compact_kv_len: List[int] = []
        # 记录每个 slot 已清理的 pad 起点，避免重复清理
        self.compact_pad_zeroed_len: List[int] = []

        # 历史名保留为 compact_arena_*；Stage A 生产路径必须把它绑定为
        # vLLM native page KV 内的 reserved compact page view，而不是额外
        # compact storage。slot 切片仍复用这些 view，避免每步 cat。
        self.compact_arena_k: torch.Tensor = torch.empty(0, device=device)
        self.compact_arena_v: torch.Tensor = torch.empty(0, device=device)
        self.compact_arena_pos: torch.Tensor = torch.empty(0, device=device, dtype=torch.int32)
        self.compact_arena_capacity_tokens: int = 0
        self.compact_generation: int = 0
        self._compact_lease_registry: BufferLeaseRegistry = BufferLeaseRegistry(num_slots=1)
        self._compact_active_lease: Optional[BufferLease] = None
        # 固定 stride（按 slot 切片）布局：token/block 数一旦确定即保持稳定
        self.compact_stride_tokens: int = 0
        self.compact_stride_blocks: int = 0
        self.compact_stride_block_size: int = 0
        self.compact_page_residency: Optional[CompactPageResidency] = None
        self.compact_metadata_buffers: Optional[CompactMetadataBuffers] = None
        self.compact_page_residency_signature: Optional[Tuple[object, ...]] = None
        self.compact_page_residency_generation: int = 0
        # compact arena 视图（slot -> narrow）已完成绑定的前缀长度。
        # 用于跳过热路径中重复的全量 view 重绑定。
        self.compact_views_bound_slots: int = 0
        self.batch_request_ids: List[str] = []
        self.free_slots: List[int] = []
        # request_id -> slot 索引缓存，避免在热路径反复 list.index()
        self.request_id_to_slot: Dict[str, int] = {}
        # 复用的 launch 缓冲（避免每步重建 meta）
        self.launch_meta_i32: Optional[torch.Tensor] = None  # [batch, 7]
        self.launch_meta_i64: Optional[torch.Tensor] = None  # [batch, 4]
        # compact 元数据变更 epoch：用于 compact layout cache 失效（避免陈旧 offset/len）
        self.compact_meta_epoch: int = 0
        # compact layout 缓冲（避免每层每步创建小 CUDA tensor）
        self.compact_layout_is_compact_i32: Optional[torch.Tensor] = None
        self.compact_layout_kv_len_i32: Optional[torch.Tensor] = None
        self.compact_layout_offset_tokens_i64: Optional[torch.Tensor] = None
        # compact layout cache（尽量让后续 step 可被 CUDA Graph 捕获）
        self.compact_layout_cache_key: Optional[Tuple[object, ...]] = None
        # 复用 arena view，避免每层每步创建新 Tensor view 导致 launcher stride cache 失效
        self.compact_layout_key_view: Optional[torch.Tensor] = None
        self.compact_layout_value_view: Optional[torch.Tensor] = None
        self.compact_layout_token_positions_view: Optional[torch.Tensor] = None
        self._compact_layout_view_key: Optional[Tuple[int, int, int, int, int]] = None
        self._compact_layout_token_pos_view_key: Optional[Tuple[int, int, int, int]] = None
        # 当前 layer 上一次看到的 active request_ids（用于跳过无意义的 align_slots 调用）
        self.last_active_request_ids: Optional[Tuple[str, ...]] = None
        # 跨层一致性签名：用于 fused rebuild 前 O(L) hard-fail 校验。
        self.slot_epoch: int = -1
        self.slot_signature64: int = _stable_slot_signature64(self.request_id_to_slot, tuple())
        # 避免同一步内重复 align_slots（由 controller 传入 step epoch）
        self._align_epoch_seen: int = -1
        # batch_last_seq_len removed - use RequestTracking.last_seq_len per request instead
        self.bootstrap_done: bool = False
        self.last_refresh_step: int = -1
        # per-slot refresh step（request-wise 触发计时）— GPU 版已废弃，仅保留 CPU 版
        self.last_refresh_step_per_slot_cpu: Optional[List[int]] = None
        self.last_reason: str = "init"
        self.last_coverage: Optional[torch.Tensor] = None
        self.last_capped: Optional[torch.Tensor] = None
        self.k_max_current: Optional[int] = None
        # Prefill accumulation buffers (for chunked prefill)
        self.prefill_kv_lengths: Optional[torch.Tensor] = None
        self.prefill_kv_len_per_row_i32: Optional[torch.Tensor] = None
        self.prefill_active_mask: Optional[torch.Tensor] = None
        # CPU-side counter to avoid GPU->CPU sync in hot paths (do NOT derive from prefill_active_mask.any()).
        self.prefill_active_count: int = 0
        self.prefill_done_mask: Optional[torch.Tensor] = None
        # CPU-side FIFO counts (GPU version removed — only CPU version used).
        self.prefill_fifo_counts_cpu: Optional[List[int]] = None
        self.prefill_total_chunks: Optional[torch.Tensor] = None
        self.prefill_chunks_seen: Optional[torch.Tensor] = None
        # slot->row 仅供 Python 控制逻辑读取；保持 CPU-only，避免 graph 热路径 GPU 标量写。
        self.slot_batch_rows: Optional[torch.Tensor] = None
        self.slot_batch_rows_cpu: Optional[List[int]] = None
        # slot->row 映射缓存（按 step_context_epoch 去重，避免每层重复清空/填充）
        self._slot_row_map_epoch: int = -1
        self._slot_row_map_key: Optional[Tuple[int, ...]] = None
        # decode 刷新计数（基于 decode_step）
        self.last_refresh_decode_per_slot: Optional[torch.Tensor] = None
        self.last_refresh_decode_per_slot_cpu: Optional[List[int]] = None
        # key_norms 缓存：避免在 refresh/selector 阶段从 paged KV 反复 gather 全量 key 再求范数。
        # - 每个 slot 维护 [num_kv_heads, capacity_tokens] 的 L2 norms（float32）
        # - 仅在 context_kv_len 增长时增量计算新增 token 的范数
        #
        # P0.1（增量 key_norms）主线实现：
        # - 使用连续 arena：key_norms_arena[slot, kv_head, token]
        # - key_norms_len[slot] 表示已计算到的 token 数（单调递增）
        # - key_norms_stride_tokens 为固定 stride（按 256 对齐，减少重分配频率）
        # 方案 1+2 优化：key_norms_len 改为 CPU tensor，消除热路径 Python 循环
        # [KEY-NORMS-LEN-I32 2026-07-06] dtype 统一 int32：全体真实下游（CUDA ext
        # 中转 delta 缓冲/staging）均 int32，旧 int64 载体使热路径每层付读入/回写
        # cast+对齐守卫；值域=token 计数 ≤ ctx << 2^31，从不作 torch 索引张量。
        self.key_norms_len: torch.Tensor = torch.zeros(0, dtype=torch.int32)  # CPU tensor [S]
        self.key_norms_capacity: List[int] = []
        self.key_norms_arena: torch.Tensor = torch.empty(0, device=device, dtype=torch.float16)  # [S,H,T]
        self.key_norms_stride_tokens: int = 0
        self.key_norms_generation: int = 0
        self._key_norms_lease_registry: BufferLeaseRegistry = BufferLeaseRegistry(num_slots=1)
        self._key_norms_active_lease: Optional[BufferLease] = None

        # ============ FA sparse runtime: per-slot selected middle logical pages ============
        self.sparse_selected_middle_pages: List[torch.Tensor] = []
        self.sparse_selected_middle_counts: List[torch.Tensor] = []
        self.sparse_request_refresh_generation: List[int] = []
        self.sparse_selected_middle_uniform_count_cpu: List[int] = []
        self.sparse_selected_middle_min_logical_page_cpu: List[int] = []
        self.sparse_selected_middle_max_logical_page_cpu: List[int] = []
        self.sparse_selected_middle_pages_dense_i32: Optional[torch.Tensor] = None
        self.sparse_selected_middle_counts_dense_i32: Optional[torch.Tensor] = None

        # ============ 两阶段缓存：per-layer step cache ============
        # 用于消除 per-layer 的 build_plan 循环和 CPU→GPU copy。
        # 通过 step_cache_epoch 与 StepMeta.epoch 保持同步。
        self.step_cache_epoch: int = -1
        self.step_cache_key: Optional[Tuple[object, ...]] = None
        self.step_cache_plan_version: int = -1
        # M5 Part C (2026-04-24): 4 个 per-layer compact 载体已删除
        # (step_cache_compact_kv_len_gpu / step_cache_compact_offset_gpu /
        # step_cache_compact_kv_len_cpu_i32 / step_cache_compact_offset_cpu_i64)。
        # Compact 布局的单源是 step_bound_meta.compact_recent_launch_plan，
        # 由 build_compact_recent_launch_plan 在 step 级别一次构建 + 一次 H2D，
        # 被所有层共享（消除了 28 次冗余的 per-layer H2D）。
        self.step_cache_req_meta_i32: Optional[torch.Tensor] = None         # [batch, 7]
        self.step_cache_req_meta_i64: Optional[torch.Tensor] = None         # [batch, 4]
        self.step_cache_meta_packed: bool = False  # req_meta 是否已完成打包
        self.step_cache_has_compact: bool = False
        self.step_cache_all_compact: bool = False
        self.step_cache_selected_static_pages_i32: Optional[torch.Tensor] = None
        self.step_cache_selected_static_seqused_k_by_head_i32: Optional[torch.Tensor] = None
        self.step_cache_selected_static_schema_version: int = 0
        self.step_cache_page_table_i32: Optional[torch.Tensor] = None
        self.step_cache_selected_seqused_k_by_head_i32: Optional[torch.Tensor] = None
        self.step_cache_cp_selected_seqused_k_by_head_i32: Optional[torch.Tensor] = None
        self.step_cache_real_kv_len_i32: Optional[torch.Tensor] = None
        self.step_cache_kv_batch_idx_i32: Optional[torch.Tensor] = None
        self.step_cache_page_table_layout: int = -1
        self.step_cache_has_page_sparse: bool = False
        self.step_cache_all_page_sparse: bool = False
        self.step_cache_applied_recent_epoch_i32: Optional[torch.Tensor] = None
        self.step_cache_applied_refresh_generation_i32: Optional[torch.Tensor] = None
        self.step_cache_materialize_status_i32: Optional[torch.Tensor] = None
        self.step_cache_patch_status_i32: Optional[torch.Tensor] = None
        self.step_cache_applied_recent_epoch_value: int = -1
        self.step_cache_requested_refresh_generation_signature: Optional[Tuple[int, ...]] = None
        self.step_cache_cached_lengths_ok: bool = False
        self.step_cache_cached_kv_batch_idx_identity: bool = False
        self.step_cache_cached_status_ok: bool = False
        self.step_cache_cached_freshness_ok: bool = False
        self.step_cache_cached_launch_ready: bool = False


    def ensure_batch(self, batch_size: int) -> None:
        # 只允许扩容，不再收缩；slot 生命周期绑定 request，不能随单次 batch 重置。
        if batch_size <= 0 or batch_size <= self.batch_size:
            return
        if _SLOT_GROW_DEBUG:
            _log.warning(
                "layer ensure_batch grow: layer=%d old_batch_size=%d new_batch_size=%d",
                int(getattr(self, "layer_index", -1)),
                int(self.batch_size),
                int(batch_size),
            )
        extra = batch_size - self.batch_size
        for _ in range(extra):
            # compact 缓冲按 slot 维度存储
            self.compact_k.append(_get_cached_empty_tensor(device=self.device, dtype=torch.float32, shape=(0,)))
            self.compact_v.append(_get_cached_empty_tensor(device=self.device, dtype=torch.float32, shape=(0,)))
            self.compact_pos.append(_get_cached_empty_tensor(device=self.device, dtype=torch.int32, shape=(0,)))
            self.compact_capacity.append(0)
            self.compact_offset_tokens.append(0)
            self.compact_read_gen.append(0)
            self.compact_sink_len.append(0)
            self.compact_persist_len.append(0)
            self.compact_kv_len.append(0)
            self.compact_pad_zeroed_len.append(-1)
            self.key_norms_capacity.append(0)
            self.sparse_selected_middle_pages.append(
                _get_cached_empty_tensor(device=self.device, dtype=torch.int32, shape=(0,))
            )
            self.sparse_selected_middle_counts.append(
                _get_cached_empty_tensor(device=self.device, dtype=torch.int32, shape=(0,))
            )
            self.sparse_request_refresh_generation.append(0)
            self.sparse_selected_middle_uniform_count_cpu.append(0)
            self.sparse_selected_middle_min_logical_page_cpu.append(-1)
            self.sparse_selected_middle_max_logical_page_cpu.append(-1)
        if self.sparse_selected_middle_pages_dense_i32 is not None:
            dense_pages = self.sparse_selected_middle_pages_dense_i32
            if dense_pages.shape[0] < batch_size:
                new_dense_pages = torch.full(
                    (batch_size, dense_pages.shape[1], dense_pages.shape[2]),
                    -1,
                    dtype=dense_pages.dtype,
                    device=dense_pages.device,
                )
                if dense_pages.numel() > 0:
                    new_dense_pages[: dense_pages.shape[0]].copy_(dense_pages)
                self.sparse_selected_middle_pages_dense_i32 = new_dense_pages
        if self.sparse_selected_middle_counts_dense_i32 is not None:
            dense_counts = self.sparse_selected_middle_counts_dense_i32
            if dense_counts.shape[0] < batch_size:
                new_dense_counts = torch.zeros(
                    (batch_size, dense_counts.shape[1]),
                    dtype=dense_counts.dtype,
                    device=dense_counts.device,
                )
                if dense_counts.numel() > 0:
                    new_dense_counts[: dense_counts.shape[0]].copy_(dense_counts)
                self.sparse_selected_middle_counts_dense_i32 = new_dense_counts
        # 方案 1+2 优化：key_norms_len tensor 扩展
        if self.key_norms_len.size(0) < batch_size:
            new_len = torch.zeros(batch_size, dtype=torch.int32)
            if self.key_norms_len.numel() > 0:
                new_len[: self.key_norms_len.size(0)] = self.key_norms_len
            self.key_norms_len = new_len
        self.batch_size = batch_size

    def _ensure_sparse_selected_middle_dense_capacity(self, *, middle_width: int) -> None:
        width = max(0, int(middle_width))
        if width <= 0:
            return
        slots = max(0, int(self.batch_size))
        heads = int(self.num_kv_heads)
        dense_pages = self.sparse_selected_middle_pages_dense_i32
        if (
            dense_pages is None
            or dense_pages.device != self.device
            or dense_pages.dtype != torch.int32
            or dense_pages.dim() != 3
            or dense_pages.shape[0] < slots
            or dense_pages.shape[1] != heads
            or dense_pages.shape[2] < width
        ):
            old_pages = dense_pages
            new_width = width
            if old_pages is not None and old_pages.dim() == 3 and old_pages.shape[2] > new_width:
                new_width = int(old_pages.shape[2])
            new_pages = torch.full(
                (slots, heads, new_width),
                -1,
                dtype=torch.int32,
                device=self.device,
            )
            if old_pages is not None and old_pages.numel() > 0:
                copy_slots = min(int(old_pages.shape[0]), slots)
                copy_heads = min(int(old_pages.shape[1]), heads)
                copy_width = min(int(old_pages.shape[2]), new_width)
                new_pages[:copy_slots, :copy_heads, :copy_width].copy_(
                    old_pages[:copy_slots, :copy_heads, :copy_width]
                )
            self.sparse_selected_middle_pages_dense_i32 = new_pages

        dense_counts = self.sparse_selected_middle_counts_dense_i32
        if (
            dense_counts is None
            or dense_counts.device != self.device
            or dense_counts.dtype != torch.int32
            or dense_counts.dim() != 2
            or dense_counts.shape[0] < slots
            or dense_counts.shape[1] != heads
        ):
            old_counts = dense_counts
            new_counts = torch.zeros(
                (slots, heads),
                dtype=torch.int32,
                device=self.device,
            )
            if old_counts is not None and old_counts.numel() > 0:
                copy_slots = min(int(old_counts.shape[0]), slots)
                copy_heads = min(int(old_counts.shape[1]), heads)
                new_counts[:copy_slots, :copy_heads].copy_(
                    old_counts[:copy_slots, :copy_heads]
                )
            self.sparse_selected_middle_counts_dense_i32 = new_counts

    def _sync_sparse_selected_middle_dense_for_slots(self, slots: Sequence[int]) -> None:
        normalized_slots = sorted({int(slot) for slot in slots if int(slot) >= 0})
        if not normalized_slots:
            return
        max_width = 0
        for slot in normalized_slots:
            if slot >= len(self.sparse_selected_middle_pages):
                continue
            pages = self.sparse_selected_middle_pages[slot]
            if isinstance(pages, torch.Tensor) and pages.dim() == 2:
                max_width = max(max_width, int(pages.shape[1]))
        if max_width <= 0:
            return
        self._ensure_sparse_selected_middle_dense_capacity(middle_width=int(max_width))
        dense_pages = self.sparse_selected_middle_pages_dense_i32
        dense_counts = self.sparse_selected_middle_counts_dense_i32
        if dense_pages is None or dense_counts is None:
            return
        for slot in normalized_slots:
            if slot >= len(self.sparse_selected_middle_pages):
                continue
            pages = self.sparse_selected_middle_pages[slot]
            counts = self.sparse_selected_middle_counts[slot]
            if not isinstance(pages, torch.Tensor) or pages.dim() != 2:
                dense_pages[slot].fill_(-1)
                dense_counts[slot].zero_()
                continue
            width = min(int(dense_pages.shape[2]), int(pages.shape[1]))
            dense_pages[slot].fill_(-1)
            dense_pages[slot, :, :width].copy_(
                pages[:, :width].to(device=self.device, dtype=torch.int32)
            )
            dense_counts[slot].zero_()
            dense_counts[slot].copy_(
                counts.reshape(-1).to(device=self.device, dtype=torch.int32)
            )

    def record_sparse_selected_middle(
        self,
        *,
        slot: int,
        selected_middle_pages: torch.Tensor,
        selected_middle_counts: torch.Tensor,
        refresh_generation: int,
    ) -> None:
        slot_i = int(slot)
        if slot_i < 0:
            raise ValueError("slot must be non-negative")
        self.ensure_batch(slot_i + 1)
        pages_i32 = selected_middle_pages.to(device=self.device, dtype=torch.int32).contiguous()
        counts_i32 = selected_middle_counts.reshape(-1).to(
            device=self.device,
            dtype=torch.int32,
        )
        if pages_i32.dim() != 2 or pages_i32.shape[0] != int(self.num_kv_heads):
            raise ValueError("selected_middle_pages must be [num_kv_heads, middle_slots]")
        if counts_i32.numel() != int(self.num_kv_heads):
            raise ValueError("selected_middle_counts must match num_kv_heads")

        self.sparse_selected_middle_pages[slot_i] = pages_i32
        self.sparse_selected_middle_counts[slot_i] = counts_i32
        self.sparse_request_refresh_generation[slot_i] = int(refresh_generation)
        # [F2] 调用点先查门控：trace 关闭（生产默认）时省掉 dict 构造，尤其是
        # counts 的 GPU→CPU 同步拷贝（每世代每层一次的流阻塞税）。
        if _selected_ready_trace_path():
            _append_selected_ready_trace(
                {
                    "event": "record_sparse_selected_middle",
                    "slot": slot_i,
                    "refresh_generation": int(refresh_generation),
                    "layer_index": int(getattr(self, "layer_index", -1)),
                    "counts": [int(v) for v in counts_i32.detach().to("cpu").tolist()],
                    "width": int(pages_i32.shape[1]),
                }
            )

        counts_cpu = [int(v) for v in counts_i32.detach().cpu().tolist()]
        first_count = counts_cpu[0] if counts_cpu else 0
        uniform_count = int(first_count) if counts_cpu and all(v == first_count for v in counts_cpu) else -1
        self.sparse_selected_middle_uniform_count_cpu[slot_i] = int(uniform_count)

        valid_pages = pages_i32[pages_i32 >= 0]
        if valid_pages.numel() > 0:
            valid_pages_cpu = valid_pages.detach().cpu()
            self.sparse_selected_middle_min_logical_page_cpu[slot_i] = int(valid_pages_cpu.min().item())
            self.sparse_selected_middle_max_logical_page_cpu[slot_i] = int(valid_pages_cpu.max().item())
        else:
            self.sparse_selected_middle_min_logical_page_cpu[slot_i] = -1
            self.sparse_selected_middle_max_logical_page_cpu[slot_i] = -1

        self._ensure_sparse_selected_middle_dense_capacity(middle_width=int(pages_i32.shape[1]))
        dense_pages = self.sparse_selected_middle_pages_dense_i32
        dense_counts = self.sparse_selected_middle_counts_dense_i32
        if dense_pages is None or dense_counts is None:
            raise RuntimeError("dense sparse-selected-middle cache must be allocated")
        dense_pages[slot_i].fill_(-1)
        dense_pages[slot_i, :, : int(pages_i32.shape[1])].copy_(pages_i32)
        dense_counts[slot_i].zero_()
        dense_counts[slot_i].copy_(counts_i32)


    def reset_idle_buffers(self) -> None:
        """释放与 request 生命周期相关的缓存，用于 idle 期显存回落。"""
        layer_index = int(self.layer_index)
        head_dim = int(self.head_dim) if self.head_dim is not None else None
        kv_cache_dtype = self.kv_cache_dtype
        num_heads = int(self.num_heads)
        num_kv_heads = int(self.num_kv_heads)
        device = self.device
        page_backed_arena_k = self.compact_arena_k
        page_backed_arena_v = self.compact_arena_v
        page_backed_arena_pos = self.compact_arena_pos
        page_backed_capacity_tokens = self.compact_arena_capacity_tokens
        page_backed_stride_tokens = self.compact_stride_tokens
        page_backed_stride_blocks = self.compact_stride_blocks
        page_backed_stride_block_size = self.compact_stride_block_size
        page_backed_residency = self.compact_page_residency
        page_backed_metadata = self.compact_metadata_buffers
        page_backed_signature = self.compact_page_residency_signature
        page_backed_generation = self.compact_page_residency_generation
        # 重新初始化以清空所有 slot/arena/caches（保持层规格与 device）
        self.__init__(num_heads=num_heads, num_kv_heads=num_kv_heads, device=device)
        self.layer_index = layer_index
        self.head_dim = head_dim
        self.kv_cache_dtype = kv_cache_dtype
        if page_backed_residency is not None:
            self.compact_arena_k = page_backed_arena_k
            self.compact_arena_v = page_backed_arena_v
            self.compact_arena_pos = page_backed_arena_pos
            self.compact_arena_capacity_tokens = page_backed_capacity_tokens
            self.compact_stride_tokens = page_backed_stride_tokens
            self.compact_stride_blocks = page_backed_stride_blocks
            self.compact_stride_block_size = page_backed_stride_block_size
            self.compact_page_residency = page_backed_residency
            self.compact_metadata_buffers = page_backed_metadata
            self.compact_page_residency_signature = page_backed_signature
            self.compact_page_residency_generation = page_backed_generation
        self._resize_prefill_counters()
        self._resize_refresh_steps()

    def ensure_key_norms_arena(
        self,
        *,
        stride_tokens: int,
        required_slots: int,
        num_kv_heads: int,
        refresh_stream: Optional[torch.cuda.Stream] = None,
        stride_floor: Optional[int] = None,
    ) -> None:
        """确保 key_norms_arena 可覆盖 required_slots 与 stride_tokens（按需扩容，保留旧数据）。"""
        stride = max(0, int(stride_tokens))
        # [KEY-NORMS-MML-CAP 2026-07-07] floor 语义=一次分配到顶、热路径永不
        # realloc(防跨流 UAF race)。"顶"由调用方按 max_model_len 对齐值传入
        # (prebuild 时 authoritative,同 capture arena 的 kv_max_cap 模式)——
        # ctx 永不超 MML,race 防护语义不变;env 262144 仅兜 MML 未知窗(如
        # 直调测试),避免 4B 12k 形态 21×/64k 形态 4× 的纯浪费(GB 级)。
        floor_eff = (
            int(stride_floor)
            if stride_floor is not None and int(stride_floor) > 0
            else int(_KEY_NORMS_STRIDE_FLOOR)
        )
        if floor_eff > 0 and stride > 0:
            stride = max(stride, floor_eff)
        slots = max(0, int(required_slots))
        if stride <= 0 or slots <= 0 or num_kv_heads <= 0:
            return

        if self.key_norms_arena.numel() > 0:
            if (
                self.key_norms_arena.device == self.device
                and self.key_norms_arena.dtype == torch.float16
                and self.key_norms_arena.dim() == 3
                and self.key_norms_arena.shape[0] >= slots
                and self.key_norms_arena.shape[1] == int(num_kv_heads)
                and self.key_norms_arena.shape[2] >= stride
                and int(self.key_norms_stride_tokens) >= stride
            ):
                return

        old = self.key_norms_arena
        old_slots = int(old.shape[0]) if old.dim() == 3 else 0
        old_heads = int(old.shape[1]) if old.dim() == 3 else 0
        old_stride = int(old.shape[2]) if old.dim() == 3 else 0

        # 关键约束：扩容时不能通过全局同步硬挡。改为 record_stream + generation lease：
        # 旧 arena 的 storage 生命周期绑定到当前流与 refresh_stream，避免跨流 UAF。
        if old.numel() > 0 and old.device.type == "cuda":
            old.record_stream(torch.cuda.current_stream(device=self.device))
            if refresh_stream is not None:
                old.record_stream(refresh_stream)

        new_slots = max(slots, old_slots)
        new_stride = max(stride, old_stride)
        new_arena = self._allocator_backend.alloc_empty(
            (new_slots, int(num_kv_heads), new_stride),
            device=self.device,
            dtype=torch.float16,
        )
        # 不强制清零：key_norms_len 控制有效范围；仅在增量写入区间覆盖。
        if old.numel() > 0 and old_heads == int(num_kv_heads) and old_stride > 0 and old_slots > 0:
            copy_slots = min(old_slots, new_slots)
            copy_stride = min(old_stride, new_stride)
            new_arena[:copy_slots, :, :copy_stride].copy_(old[:copy_slots, :, :copy_stride])
        self.key_norms_arena = new_arena
        self.key_norms_stride_tokens = int(new_stride)
        old_lease = self._key_norms_active_lease
        if old_lease is not None:
            retire_id = f"key_norms-retire-{old_lease.generation}-{time.time_ns()}"
            self._key_norms_lease_registry.retire(lease=old_lease, event_id=retire_id)
        new_lease = self._key_norms_lease_registry.acquire(
            kind=LeaseKind.KEY_NORMS,
            slot=0,
            min_capacity=int(new_slots * int(num_kv_heads) * new_stride),
            epoch=int(time.time_ns() & 0x7FFFFFFF),
        )
        self._key_norms_active_lease = new_lease
        self.key_norms_generation = int(new_lease.generation)
        # 同步更新 per-slot capacity（避免旧逻辑误判容量不足走慢路径）
        if len(self.key_norms_capacity) < new_slots:
            self.key_norms_capacity.extend([0 for _ in range(new_slots - len(self.key_norms_capacity))])
        for s in range(new_slots):
            self.key_norms_capacity[s] = int(new_stride)

        # cat buffer 已移除

    def ensure_compact_layout_buffers(self, batch_size: int) -> None:
        """为 compact layout 预分配小张量缓冲，避免热路径 cuda malloc/free。"""
        if batch_size <= 0:
            self.compact_layout_is_compact_i32 = None
            self.compact_layout_kv_len_i32 = None
            self.compact_layout_offset_tokens_i64 = None
            self.compact_layout_cache_key = None
            self.compact_layout_key_view = None
            self.compact_layout_value_view = None
            self.compact_layout_token_positions_view = None
            self._compact_layout_view_key = None
            self._compact_layout_token_pos_view_key = None
            return

        device = self.device
        selector_enabled = getattr(getattr(self, "config", None), "enabled", True)

        t_i32 = self.compact_layout_is_compact_i32
        if (
            t_i32 is None
            or t_i32.device != device
            or t_i32.dtype != torch.int32
            or t_i32.dim() != 1
            or run_selector_step(
                enabled=selector_enabled,
                force_dense=False,
                capacity=(t_i32.numel() if t_i32 is not None else 0),
                batch_size=batch_size,
                shape_signature=("compact_layout_is_compact_i32",),
            ).resize_needed
        ):
            self.compact_layout_is_compact_i32 = torch.empty((batch_size,), device=device, dtype=torch.int32)

        t_kv_i32 = self.compact_layout_kv_len_i32
        if (
            t_kv_i32 is None
            or t_kv_i32.device != device
            or t_kv_i32.dtype != torch.int32
            or t_kv_i32.dim() != 1
            or run_selector_step(
                enabled=selector_enabled,
                force_dense=False,
                capacity=(t_kv_i32.numel() if t_kv_i32 is not None else 0),
                batch_size=batch_size,
                shape_signature=("compact_layout_kv_len_i32",),
            ).resize_needed
        ):
            self.compact_layout_kv_len_i32 = torch.empty((batch_size,), device=device, dtype=torch.int32)

        t_off_i64 = self.compact_layout_offset_tokens_i64
        if (
            t_off_i64 is None
            or t_off_i64.device != device
            or t_off_i64.dtype != torch.int64
            or t_off_i64.dim() != 1
            or run_selector_step(
                enabled=selector_enabled,
                force_dense=False,
                capacity=(t_off_i64.numel() if t_off_i64 is not None else 0),
                batch_size=batch_size,
                shape_signature=("compact_layout_offset_tokens_i64",),
            ).resize_needed
        ):
            self.compact_layout_offset_tokens_i64 = torch.empty((batch_size,), device=device, dtype=torch.int64)

    def align_slots_from_snapshot(
        self,
        request_ids: Sequence[str],
        slot_by_row: Sequence[int],
        *,
        epoch: Optional[int] = None,
    ) -> None:
        # Allocation-free steady-state fast path: when request_ids is
        # already a plain tuple (the per-layer hot path passes
        # step_ctx.req_ids, a frozen-dataclass Tuple), the steady-state
        # epoch guard can be evaluated WITHOUT the list()/tuple() allocs.
        # Returns early in exactly the cases the original guard does; any
        # other case falls through to the unchanged block below.
        if (
            epoch is not None
            and type(request_ids) is tuple
            and len(slot_by_row) >= len(request_ids)
        ):
            epoch_i = int(epoch)
            if (
                epoch_i >= 0
                and self._align_epoch_seen == epoch_i
                and self.last_active_request_ids == request_ids
            ):
                return
        req_ids = list(request_ids)
        if len(slot_by_row) < len(req_ids):
            raise RuntimeError(
                "compiled slot snapshot length mismatch: "
                f"slot_by_row={len(slot_by_row)}, req_ids={len(req_ids)}"
            )
        if epoch is not None:
            epoch_i = int(epoch)
            if (
                epoch_i >= 0
                and self._align_epoch_seen == epoch_i
                and self.last_active_request_ids == tuple(req_ids)
            ):
                return

        if not req_ids:
            self.align_slots([], epoch=epoch)
            return

        slot_by_req: Dict[str, int] = {}
        snapshot_owner_by_slot: Dict[int, str] = {}
        for row, rid in enumerate(req_ids):
            slot = slot_by_row[row]
            if isinstance(slot, bool) or not isinstance(slot, int):
                raise RuntimeError(
                    f"compiled slot must be int, got {type(slot).__name__} for request {rid}"
                )
            slot_i = int(slot)
            if slot_i < 0:
                raise RuntimeError(
                    f"compiled slot must be >= 0, got {slot_i} for request {rid}"
                )
            prev_slot = slot_by_req.get(rid)
            if prev_slot is not None and int(prev_slot) != slot_i:
                raise RuntimeError(
                    f"compiled slot snapshot mismatch for request {rid}: "
                    f"prev_slot={int(prev_slot)}, new_slot={slot_i}"
                )
            slot_by_req[rid] = slot_i
            owner = snapshot_owner_by_slot.get(slot_i)
            if owner is not None and owner != rid:
                raise RuntimeError(
                    f"compiled slot collision at slot={slot_i}: owner={owner}, contender={rid}"
                )
            snapshot_owner_by_slot[slot_i] = rid

            existing = self.request_id_to_slot.get(rid)
            if existing is not None and int(existing) != slot_i:
                raise RuntimeError(
                    f"compiled slot mismatch for request {rid}: "
                    f"layer_slot={int(existing)}, compiled_slot={slot_i}"
                )

        new_rids: List[str] = []
        seen: Set[str] = set()
        for rid in req_ids:
            if rid in self.request_id_to_slot or rid in seen:
                continue
            seen.add(rid)
            new_rids.append(rid)

        if not new_rids:
            self.last_active_request_ids = tuple(req_ids)
            self._refresh_slot_signature(
                active_request_ids=self.last_active_request_ids,
                epoch=epoch,
            )
            if epoch is not None and int(epoch) >= 0:
                self._align_epoch_seen = int(epoch)
            return

        max_slot = max(int(slot_by_req[rid]) for rid in req_ids)
        target = int(max_slot + 1)
        if target > self.batch_size:
            self.ensure_batch(target)
        old_len = len(self.batch_request_ids)
        free_slot_set: Set[int] = {int(slot) for slot in self.free_slots}
        if target > old_len:
            for idx in range(old_len, target):
                self.batch_request_ids.append(_FREE_SLOT_ID)
                if idx not in free_slot_set:
                    heapq.heappush(self.free_slots, int(idx))
                    free_slot_set.add(int(idx))

        assigned_new_slots: Set[int] = set()
        for rid in new_rids:
            new_slot = int(slot_by_req[rid])
            if new_slot < 0 or new_slot >= len(self.batch_request_ids):
                raise RuntimeError(
                    f"compiled slot {new_slot} is out of range for request {rid} "
                    f"(batch_request_ids={len(self.batch_request_ids)})"
                )
            owner = self.batch_request_ids[new_slot]
            if (not _is_free_slot_id(owner)) and owner != rid:
                raise RuntimeError(
                    f"compiled slot collision at slot={new_slot}: owner={owner}, contender={rid}"
                )
            assigned_new_slots.add(int(new_slot))
            self.batch_request_ids[new_slot] = rid
            self.request_id_to_slot[rid] = int(new_slot)
        if assigned_new_slots and self.free_slots:
            self.free_slots = [
                int(slot)
                for slot in self.free_slots
                if int(slot) not in assigned_new_slots
            ]
            heapq.heapify(self.free_slots)

        self.bump_compact_meta_epoch()
        self.bootstrap_done = False
        self._resize_prefill_counters()
        self._resize_refresh_steps()
        self.last_active_request_ids = tuple(req_ids)
        self._refresh_slot_signature(
            active_request_ids=self.last_active_request_ids,
            epoch=epoch,
        )
        if epoch is not None and int(epoch) >= 0:
            self._align_epoch_seen = int(epoch)

    def bump_compact_meta_epoch(self) -> None:
        """标记 compact 元数据已变更，失效 compact layout cache。"""
        self.compact_meta_epoch += 1
        self.compact_layout_cache_key = None
        # arena 发生变化时，相关 view 也必须重建，否则 stride cache 会错配/失效
        self.compact_layout_key_view = None
        self.compact_layout_value_view = None
        self.compact_layout_token_positions_view = None
        self._compact_layout_view_key = None
        self._compact_layout_token_pos_view_key = None
    def _refresh_slot_signature(
        self,
        *,
        active_request_ids: Optional[Tuple[str, ...]],
        epoch: Optional[int],
    ) -> None:
        epoch_i = int(epoch) if epoch is not None else -1
        if epoch_i >= 0:
            self.slot_epoch = int(epoch_i)
        else:
            self.slot_epoch = int(self.slot_epoch) + 1
        active_ids = tuple(active_request_ids or tuple())
        self.slot_signature64 = _stable_slot_signature64(
            self.request_id_to_slot,
            active_ids,
        )

    def get_compact_kv_views(
        self,
        *,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回 compact arena 的 K/V 4D 视图，尽量复用同一个 view 对象以稳定 id()/stride cache。"""
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        key = (id(self.compact_arena_k), id(self.compact_arena_v), block_size, num_kv_heads, head_dim)
        if self.compact_layout_key_view is None or self.compact_layout_value_view is None or self._compact_layout_view_key != key:
            if self.compact_arena_k.numel() <= 0 or self.compact_arena_v.numel() <= 0:
                raise RuntimeError("compact arena is empty")
            self.compact_layout_key_view = self.compact_arena_k.view(-1, block_size, num_kv_heads, head_dim)
            self.compact_layout_value_view = self.compact_arena_v.view(-1, block_size, num_kv_heads, head_dim)
            self._compact_layout_view_key = key
        return self.compact_layout_key_view, self.compact_layout_value_view

    def get_compact_token_positions_view(
        self,
        *,
        block_size: int,
        num_kv_heads: int,
        device: torch.device,
        rows: int,
    ) -> torch.Tensor:
        """返回 token_positions 的 arena 视图（row stride=0，可配合 token_row_base 区分行）。"""
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        rows = max(1, int(rows))
        if self.compact_arena_pos.numel() > 0:
            total_blocks = max(1, int(self.compact_arena_pos.shape[1] // block_size))
            key = (id(self.compact_arena_pos), block_size, num_kv_heads, total_blocks, rows)
            if self.compact_layout_token_positions_view is None or self._compact_layout_token_pos_view_key != key:
                self.compact_layout_token_positions_view = self.compact_arena_pos.as_strided(
                    size=(rows, num_kv_heads, total_blocks, block_size),
                    stride=(
                        0,  # row stride 0，依赖 token_row_base 区分
                        self.compact_arena_pos.stride(0),
                        block_size,
                        1,
                    ),
                )
                self._compact_layout_token_pos_view_key = key
            return self.compact_layout_token_positions_view

        key = (0, block_size, num_kv_heads, 1, rows)
        if self.compact_layout_token_positions_view is None or self._compact_layout_token_pos_view_key != key:
            self.compact_layout_token_positions_view = torch.full(
                (rows, num_kv_heads, 1, block_size),
                -1,
                dtype=torch.int32,
                device=device,
            )
            self._compact_layout_token_pos_view_key = key
        return self.compact_layout_token_positions_view

    def _compact_page_slot_capacity(self) -> Optional[int]:
        metadata = self.compact_metadata_buffers
        if metadata is not None and int(metadata.capacity_slots) > 0:
            return int(metadata.capacity_slots)
        residency = self.compact_page_residency
        if residency is not None:
            capacity = int(getattr(residency.lease, "max_live_sparse_slots", 0))
            if capacity > 0:
                return capacity
        return None

    def align_slots(
        self,
        request_ids: Sequence[str],
        *,
        epoch: Optional[int] = None,
        slot_by_request: Optional[Dict[str, int]] = None,
    ) -> None:
        if epoch is not None:
            epoch_i = int(epoch)
            if (
                epoch_i >= 0
                and self._align_epoch_seen == epoch_i
                and self.last_active_request_ids == tuple(request_ids)
                and slot_by_request is None
            ):
                return
        if not request_ids:
            page_backed_arena_k = self.compact_arena_k
            page_backed_arena_v = self.compact_arena_v
            page_backed_arena_pos = self.compact_arena_pos
            page_backed_capacity_tokens = self.compact_arena_capacity_tokens
            page_backed_stride_tokens = self.compact_stride_tokens
            page_backed_stride_blocks = self.compact_stride_blocks
            page_backed_stride_block_size = self.compact_stride_block_size
            page_backed_residency = self.compact_page_residency
            page_backed_metadata = self.compact_metadata_buffers
            page_backed_signature = self.compact_page_residency_signature
            page_backed_generation = self.compact_page_residency_generation
            # 当前 layer 已无活跃 request，安全清空全部 slot 状态。
            self.compact_k = []
            self.compact_v = []
            self.compact_pos = []
            self.compact_capacity = []
            self.compact_offset_tokens = []
            self.compact_read_gen = []
            self.compact_sink_len = []
            self.compact_persist_len = []
            self.compact_kv_len = []
            self.compact_pad_zeroed_len = []
            self.key_norms_len = torch.zeros(0, dtype=torch.int32)  # 方案 1+2：CPU tensor
            self.key_norms_capacity = []
            self.key_norms_arena = torch.empty(0, device=self.device, dtype=torch.float16)
            self.key_norms_stride_tokens = 0
            self.compact_arena_k = torch.empty(0, device=self.device)
            self.compact_arena_v = torch.empty(0, device=self.device)
            self.compact_arena_pos = torch.empty(0, device=self.device, dtype=torch.int32)
            self.compact_arena_capacity_tokens = 0
            self.compact_stride_tokens = 0
            self.compact_stride_blocks = 0
            self.compact_stride_block_size = 0
            self.compact_views_bound_slots = 0
            if page_backed_residency is not None:
                self.compact_arena_k = page_backed_arena_k
                self.compact_arena_v = page_backed_arena_v
                self.compact_arena_pos = page_backed_arena_pos
                self.compact_arena_capacity_tokens = page_backed_capacity_tokens
                self.compact_stride_tokens = page_backed_stride_tokens
                self.compact_stride_blocks = page_backed_stride_blocks
                self.compact_stride_block_size = page_backed_stride_block_size
                self.compact_page_residency = page_backed_residency
                self.compact_metadata_buffers = page_backed_metadata
                self.compact_page_residency_signature = page_backed_signature
                self.compact_page_residency_generation = page_backed_generation
            self.batch_request_ids = []
            self.free_slots = []
            self.request_id_to_slot = {}
            self.sparse_selected_middle_pages = []
            self.sparse_selected_middle_counts = []
            self.sparse_request_refresh_generation = []
            self.sparse_selected_middle_uniform_count_cpu = []
            self.sparse_selected_middle_min_logical_page_cpu = []
            self.sparse_selected_middle_max_logical_page_cpu = []
            self.sparse_selected_middle_pages_dense_i32 = None
            self.sparse_selected_middle_counts_dense_i32 = None
            self.launch_meta_i32 = None
            self.launch_meta_i64 = None
            self.batch_size = 0
            self.prefill_total_chunks = None
            self.prefill_chunks_seen = None
            # 清理残留的 prefill 缓存，避免占用无意义的张量
            self.prefill_kv_lengths = None
            self.prefill_active_mask = None
            self.prefill_done_mask = None
            self.prefill_fifo_counts_cpu = None
            self.slot_batch_rows = None
            self.slot_batch_rows_cpu = None
            # slot->row 映射缓存（按 step_context_epoch 去重，避免每层重复清空/填充）
            self._slot_row_map_epoch = -1
            self._slot_row_map_key = None
            self.last_refresh_decode_per_slot = None
            self.last_refresh_step_per_slot_cpu = None
            self.last_refresh_decode_per_slot_cpu = None
            self.compact_layout_is_compact_i32 = None
            self.compact_layout_kv_len_i32 = None
            self.compact_layout_offset_tokens_i64 = None
            self.compact_layout_cache_key = None
            self.compact_layout_key_view = None
            self.compact_layout_value_view = None
            self.compact_layout_token_positions_view = None
            self._compact_layout_view_key = None
            self._compact_layout_token_pos_view_key = None
            self.step_cache_epoch = -1
            self.step_cache_key = None
            self.step_cache_plan_version = -1
            self.step_cache_meta_packed = False
            self.step_cache_selected_static_pages_i32 = None
            self.step_cache_selected_static_seqused_k_by_head_i32 = None
            self.step_cache_selected_static_schema_version = 0
            self.step_cache_page_table_i32 = None
            self.step_cache_selected_seqused_k_by_head_i32 = None
            self.step_cache_cp_selected_seqused_k_by_head_i32 = None
            self.step_cache_real_kv_len_i32 = None
            self.step_cache_kv_batch_idx_i32 = None
            self.step_cache_page_table_layout = -1
            self.step_cache_has_page_sparse = False
            self.step_cache_all_page_sparse = False
            self.step_cache_applied_recent_epoch_i32 = None
            self.step_cache_applied_refresh_generation_i32 = None
            self.step_cache_materialize_status_i32 = None
            self.step_cache_patch_status_i32 = None
            self.step_cache_applied_recent_epoch_value = -1
            self.step_cache_requested_refresh_generation_signature = None
            self.step_cache_cached_lengths_ok = False
            self.step_cache_cached_kv_batch_idx_identity = False
            self.step_cache_cached_status_ok = False
            self.step_cache_cached_freshness_ok = False
            self.step_cache_cached_launch_ready = False
            self.last_active_request_ids = None
            self._refresh_slot_signature(
                active_request_ids=tuple(),
                epoch=epoch,
            )
            if epoch is not None and int(epoch) >= 0:
                self._align_epoch_seen = int(epoch)
            return

        # 多 request 管理原则：slot 与 request 生命周期绑定，只在 request 第一次出现该 layer 时分配。
        # 这一帧 block_table 中没出现的 request，只是“本帧未调度”，不能把其 slot 状态清空。
        new_rids: List[str] = []
        seen: Set[str] = set()
        for rid in request_ids:
            if rid in self.request_id_to_slot or rid in seen:
                continue
            seen.add(rid)
            new_rids.append(rid)

        if slot_by_request is not None:
            for rid in request_ids:
                slot = slot_by_request.get(rid)
                if slot is None:
                    raise RuntimeError(f"missing global slot binding for request {rid}")
                if isinstance(slot, bool) or not isinstance(slot, int):
                    raise RuntimeError(f"global slot must be int, got {type(slot).__name__} for request {rid}")
                slot_i = int(slot)
                if slot_i < 0:
                    raise RuntimeError(f"global slot must be >= 0, got {slot_i} for request {rid}")
                compact_page_capacity = self._compact_page_slot_capacity()
                if compact_page_capacity is not None and slot_i >= compact_page_capacity:
                    raise RuntimeError(
                        "global slot capacity exceeded for compact page residency: "
                        f"slot={slot_i}, capacity={compact_page_capacity}, request={rid}"
                    )
                existing = self.request_id_to_slot.get(rid)
                if existing is not None and int(existing) != slot_i:
                    raise RuntimeError(
                        f"global slot mismatch for request {rid}: "
                        f"layer_slot={int(existing)}, global_slot={slot_i}"
                    )

        if not new_rids:
            # active requests 未变化且无新 slot：无需重复 resize/重置状态。
            self.last_active_request_ids = tuple(request_ids)
            self._refresh_slot_signature(
                active_request_ids=self.last_active_request_ids,
                epoch=epoch,
            )
            if epoch is not None and int(epoch) >= 0:
                self._align_epoch_seen = int(epoch)
            return

        if slot_by_request is not None:
            max_slot = max(int(slot_by_request[rid]) for rid in request_ids)
            target = int(max_slot + 1)
            if target > self.batch_size:
                self.ensure_batch(target)
            old_len = len(self.batch_request_ids)
            free_slot_set: Set[int] = {int(slot) for slot in self.free_slots}
            if target > old_len:
                for idx in range(old_len, target):
                    self.batch_request_ids.append(_FREE_SLOT_ID)
                    if idx not in free_slot_set:
                        heapq.heappush(self.free_slots, int(idx))
                        free_slot_set.add(int(idx))

            assigned_new_slots: Set[int] = set()
            for rid in new_rids:
                new_slot = int(slot_by_request[rid])
                if new_slot < 0 or new_slot >= len(self.batch_request_ids):
                    raise RuntimeError(
                        f"global slot {new_slot} is out of range for request {rid} "
                        f"(batch_request_ids={len(self.batch_request_ids)})"
                    )
                owner = self.batch_request_ids[new_slot]
                if (not _is_free_slot_id(owner)) and owner != rid:
                    raise RuntimeError(
                        f"global slot collision at slot={new_slot}: owner={owner}, contender={rid}"
                    )
                assigned_new_slots.add(int(new_slot))
                self.batch_request_ids[new_slot] = rid
                self.request_id_to_slot[rid] = int(new_slot)
            if assigned_new_slots and self.free_slots:
                self.free_slots = [
                    int(slot)
                    for slot in self.free_slots
                    if int(slot) not in assigned_new_slots
                ]
                heapq.heapify(self.free_slots)
        else:
            # 批量扩容：避免在多 request 下每个新 rid 都触发一次 ensure_batch/bump_epoch。
            need_extra = max(0, int(len(new_rids) - len(self.free_slots)))
            if need_extra > 0:
                target = int(len(self.batch_request_ids) + need_extra)
                self.ensure_batch(target)

            for rid in new_rids:
                # 为新 request 分配一个新的 slot，并扩容各个 per-head store。
                # slot 索引必须与 batch_request_ids 的下标一致，否则会导致：
                # - prefill_capture_plan 以 list.index(req_id) 产生的 slot_idx 无法命中；
                # - request_id_to_slot/get(slot_idx) 产生的 slot 发生偏移，进而让 prefill 不捕获 logits，
                #   decode 永远走 dense（看不到任何稀疏加速）。
                if self.free_slots:
                    # 关键：多层必须对同一批新 request 做出一致的 slot 分配。
                    # free_slots 的历史顺序在不同 layer 上可能不同（尤其是同一步内“结束 + 新来”的复用场景），
                    # 使用最小堆拿最小 slot，避免线性扫描在高 churn 场景放大控制面开销。
                    # 不在热路径重复 heapify；free_slots 由 cleanup 的 heappush 维护堆不变式。
                    new_slot = int(heapq.heappop(self.free_slots))
                    if new_slot >= len(self.batch_request_ids):
                        raise RuntimeError(
                            f"free_slot {new_slot} >= batch_request_ids length "
                            f"{len(self.batch_request_ids)}; slot state corrupted"
                        )
                    self.batch_request_ids[new_slot] = rid
                else:
                    new_slot = int(len(self.batch_request_ids))
                    self.batch_request_ids.append(rid)
                self.request_id_to_slot[rid] = int(new_slot)

        self.bump_compact_meta_epoch()

        self.bootstrap_done = False
        self._resize_prefill_counters()
        self._resize_refresh_steps()
        self.last_active_request_ids = tuple(request_ids)
        self._refresh_slot_signature(
            active_request_ids=self.last_active_request_ids,
            epoch=epoch,
        )
        if epoch is not None and int(epoch) >= 0:
            self._align_epoch_seen = int(epoch)
        # batch_last_seq_len removed - use RequestTracking.last_seq_len instead

    def _resize_prefill_counters(self) -> None:
        """Resize per-slot prefill counters to match current slot capacity."""

        size = self.batch_size
        if size <= 0:
            self.prefill_total_chunks = None
            self.prefill_chunks_seen = None
            self.prefill_fifo_counts_cpu = None
            return

        def _resize(attr: str) -> None:
            tensor = getattr(self, attr)
            if tensor is None:
                tensor = torch.zeros((size,), dtype=torch.long, device=self.device)
            elif tensor.shape[0] != size:
                new_tensor = torch.zeros((size,), dtype=tensor.dtype, device=self.device)
                copy = min(size, tensor.shape[0])
                if copy > 0:
                    new_tensor[:copy] = tensor[:copy]
                tensor = new_tensor
            setattr(self, attr, tensor)

        _resize("prefill_total_chunks")
        _resize("prefill_chunks_seen")
        # FIFO counts are only used in Python control logic; keep them on CPU to avoid any `.item()` syncs.
        if self.prefill_fifo_counts_cpu is None:
            self.prefill_fifo_counts_cpu = [0 for _ in range(size)]
        elif len(self.prefill_fifo_counts_cpu) != size:
            if len(self.prefill_fifo_counts_cpu) < size:
                self.prefill_fifo_counts_cpu.extend([0 for _ in range(size - len(self.prefill_fifo_counts_cpu))])
            else:
                self.prefill_fifo_counts_cpu = self.prefill_fifo_counts_cpu[:size]
        self.slot_batch_rows = None

    def _resize_refresh_steps(self) -> None:
        """Resize per-slot refresh step tracker to current slot capacity."""

        size = self.batch_size
        if size <= 0:
            self.last_refresh_decode_per_slot = None
            self.last_refresh_step_per_slot_cpu = None
            self.last_refresh_decode_per_slot_cpu = None
            return

        # 这些值仅在 Python 控制逻辑中使用；使用 CPU list 避免 GPU sync。
        steps = self.last_refresh_step_per_slot_cpu
        if steps is None:
            steps = [-1 for _ in range(size)]
        elif len(steps) != size:
            if len(steps) < size:
                steps.extend([-1 for _ in range(size - len(steps))])
            else:
                steps = steps[:size]
        self.last_refresh_step_per_slot_cpu = steps

        steps_decode = self.last_refresh_decode_per_slot_cpu
        if steps_decode is None:
            steps_decode = [-1 for _ in range(size)]
        elif len(steps_decode) != size:
            if len(steps_decode) < size:
                steps_decode.extend([-1 for _ in range(size - len(steps_decode))])
            else:
                steps_decode = steps_decode[:size]
        self.last_refresh_decode_per_slot_cpu = steps_decode

        # 保守：清空 GPU tensor 版本，避免后续路径误写入导致性能回退。
        self.last_refresh_decode_per_slot = None

    def update_slot_rows(self, slot_to_row: Dict[int, int]) -> None:
        if self.batch_size <= 0:
            self.slot_batch_rows = None
            self.slot_batch_rows_cpu = None
            return
        rows_cpu = [-1 for _ in range(self.batch_size)]
        for slot, row in slot_to_row.items():
            if slot < 0 or slot >= self.batch_size:
                continue
            row_idx = max(-1, int(row))
            rows_cpu[slot] = row_idx
        self.slot_batch_rows = None
        self.slot_batch_rows_cpu = rows_cpu
