from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from patches.sparse_constants import _CAPTURE_CHUNK, _CAPTURE_IN_FLIGHT
from patches.sparse_types import LayerDecodeData, StepDecodeData, StepMeta
from patches.fa_sparse_runtime.compact_recent_alignment import (
    compact_recent_effective_k_head,
)

def build_step_decode_data_impl(
    self,
    step_meta: StepMeta,
    batch_size: int,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
    kv_cache_dtype: torch.dtype,
    cache_key: Tuple[object, ...],
    decode_plan_version: int = -1,
) -> None:
    """构建 StepDecodeData，包含所有层的预构建 decode 数据。

    在 Triton metadata builder 阶段调用。
    目标：使 dispatcher 入口只需"读取 LayerDecodeData + 调用 kernel"。
    """
    if not self.layer_states or not self.layer_cache_keys:
        self.step_decode_data = None
        self.step_decode_cache_key = None
        self.step_dispatch_plan = None
        return

    # 设备信息从任意 layer state 获取
    first_state = next(iter(self.layer_states.values()))
    device = first_state.device
    layer_dtype = kv_cache_dtype
    num_query_heads = int(first_state.num_heads)
    num_seqs = batch_size
    num_queries_per_kv = max(1, num_query_heads // num_kv_heads)
    block_m = 16
    block_q = max(1, block_m // num_queries_per_kv)
    total_num_q_blocks = (num_seqs // block_q) + num_seqs
    if total_num_q_blocks <= 16:
        grid_bin = 0
    elif total_num_q_blocks <= 64:
        grid_bin = 1
    elif total_num_q_blocks <= 256:
        grid_bin = 2
    else:
        grid_bin = 3
    if num_seqs <= 4:
        num_seqs_bin = 0
    elif num_seqs <= 8:
        num_seqs_bin = 1
    elif num_seqs <= 16:
        num_seqs_bin = 2
    else:
        num_seqs_bin = 3
    head_size_padded = 1 << (head_dim - 1).bit_length() if head_dim > 0 else 1

    # 获取 persist 上界配置用于 compact_kv_len_max hint
    persist_cap = 0
    sink_cap = 0
    semantic_snapshot = self._get_step_semantic_snapshot()
    sink_cap = max(0, int(semantic_snapshot.sink_tokens))
    if self.config is not None:
        if self.config.alpha_fair is not None and self.config.alpha_fair.k_head is not None:
            persist_cap = max(persist_cap, int(self.config.alpha_fair.k_head))
        if self.config.k_max is not None:
            persist_cap = max(persist_cap, int(self.config.k_max))
        if self.config.k_min is not None:
            persist_cap = max(persist_cap, int(self.config.k_min))
        persist_cap = compact_recent_effective_k_head(
            k_head=persist_cap,
            sink_tokens=sink_cap,
            attn_mode=str(getattr(self.config, "attn_mode", "compact_recent")),
        )

    layer_count = len(self.layer_cache_keys)
    layer_data_list: List[Optional[LayerDecodeData]] = [None] * layer_count

    # 缓存空 tensor（dense/compact 空路径复用），避免 per-layer 重复分配
    _empty_kv = getattr(self, "_cached_empty_kv", None)
    if (
        _empty_kv is None
        or _empty_kv.device != device
        or _empty_kv.dtype != layer_dtype
        or _empty_kv.shape[1] != block_size
        or _empty_kv.shape[2] != num_kv_heads
        or _empty_kv.shape[3] != head_dim
    ):
        _empty_kv = torch.empty((0, block_size, num_kv_heads, head_dim), device=device, dtype=layer_dtype)
        _empty_tp = torch.empty((0, num_kv_heads, 0, block_size), dtype=torch.int32, device=device)
        self._cached_empty_kv = _empty_kv
        self._cached_empty_tp = _empty_tp
    else:
        _empty_tp = self._cached_empty_tp

    # [T2-HOST-DIET 2026-07-10] per-layer 静态量((layer_key, layer_index,
    # chunk_id, buf_id, 跨层 req_meta 视图对) memo。anchor=层表/索引表/两
    # 跨层 buffer 的对象身份,任一变整表重建(SLOT-VIEW-MEMO 同款 fail-close);
    # state 存活性与 per-step 字段仍逐步活读。视图对为 None = 该层回退
    # state 自有 buffer(anchor 时刻跨层 buffer 缺失或越界,语义与原逐步判定一致)。
    _sdd_anchor = (
        id(self.layer_cache_keys),
        id(self.layer_index_by_cache_key),
        id(self.step_decode_req_meta_i32_all),
        id(self.step_decode_req_meta_i64_all),
        layer_count,
    )
    _sdd_cached = getattr(self, "_sdd_layer_static_cache", None)
    if _sdd_cached is None or _sdd_cached[0] != _sdd_anchor:
        _capture_chunk = int(_CAPTURE_CHUNK)
        _capture_in_flight = int(_CAPTURE_IN_FLIGHT)
        _i32_all = self.step_decode_req_meta_i32_all
        _i64_all = self.step_decode_req_meta_i64_all
        _static_rows = []
        for layer_key in self.layer_cache_keys:
            layer_index = self.layer_index_by_cache_key.get(layer_key, -1)
            if layer_index < 0 or layer_index >= layer_count:
                continue
            _i32_view = None
            _i64_view = None
            if (
                _i32_all is not None
                and _i64_all is not None
                and layer_index < _i32_all.shape[0]
                and layer_index < _i64_all.shape[0]
            ):
                _i32_view = _i32_all[layer_index]
                _i64_view = _i64_all[layer_index]
            _static_rows.append(
                (
                    layer_key,
                    int(layer_index),
                    int(layer_index) // _capture_chunk,
                    (int(layer_index) // _capture_chunk) % _capture_in_flight,
                    _i32_view,
                    _i64_view,
                )
            )
        _sdd_cached = (_sdd_anchor, tuple(_static_rows))
        self._sdd_layer_static_cache = _sdd_cached

    for layer_key, layer_index, chunk_id, buf_id, req_meta_i32_use, req_meta_i64_use in _sdd_cached[1]:
        state = self.layer_states.get(layer_key)
        if state is None:
            continue
        # 确保 step_cache 已构建
        if state.step_cache_req_meta_i32 is None or state.step_cache_req_meta_i64 is None:
            continue

        # 获取 has_compact（已在 _build_layer_step_cache 中预计算）
        has_compact = getattr(state, "step_cache_has_compact", False)

        if req_meta_i32_use is None or req_meta_i64_use is None:
            req_meta_i32_use = state.step_cache_req_meta_i32
            req_meta_i64_use = state.step_cache_req_meta_i64

        # StepBoundMeta owns the actual per-layer compact K/V binding and
        # refreshes it from state_ref below. Keep StepDecodeData light here:
        # ordered-reuse only needs readiness/hints, not a duplicate view build.
        layer_data = LayerDecodeData(
            req_meta_i32=req_meta_i32_use,
            req_meta_i64=req_meta_i64_use,
            k_compact=_empty_kv,
            v_compact=_empty_kv,
            token_positions=_empty_tp,
            layer_index=layer_index,
            chunk_id=chunk_id,
            buf_id=buf_id,
            has_compact=getattr(state, "step_cache_all_compact", False),
            compact_kv_len_max=(sink_cap + persist_cap) if has_compact else 0,
            page_sparse_enabled=bool(getattr(state, "step_cache_has_page_sparse", False)),
            state_ref=state,
        )
        layer_data_list[layer_index] = layer_data

    # 创建 StepDecodeData
    self.step_decode_data = StepDecodeData(
        cache_key=cache_key,
        layer_data=layer_data_list,
        seqused_k=step_meta.seqused_k_gpu if step_meta.seqused_k_gpu is not None else torch.empty(0, device=device),
        batch_size=batch_size,
        num_query_heads=num_query_heads,
        num_queries_per_kv=num_queries_per_kv,
        head_size_padded=head_size_padded,
        block_q=block_q,
        total_num_q_blocks=total_num_q_blocks,
        grid_bin=grid_bin,
        num_seqs_bin=num_seqs_bin,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        launch_large=(int(total_num_q_blocks) * int(num_kv_heads)) > 128,
        grid_2d=(int(total_num_q_blocks), int(num_kv_heads)) if (int(total_num_q_blocks) * int(num_kv_heads)) > 128 else None,
        epoch=step_meta.epoch,
        decode_plan_version=int(decode_plan_version),
        # [T2-HOST-DIET 2026-07-10] chunk_layer_indices 为写而不读的死字段
        # (全仓零运行时消费者,唯一同名处是 profile 基准自建副本),每 commit
        # 步的 dict 推导+tuple 物化纯税 → 停建置 None;字段声明保留在
        # sparse_types 合同面。
        chunk_layer_indices=None,
    )
