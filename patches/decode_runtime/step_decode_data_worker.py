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

    layer_data_list: List[Optional[LayerDecodeData]] = [None] * len(self.layer_cache_keys)
    chunk_layer_indices: dict[int, list[int]] = {}

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

    for layer_key in self.layer_cache_keys:
        state = self.layer_states.get(layer_key)
        if state is None:
            continue
        layer_index = self.layer_index_by_cache_key.get(layer_key, -1)
        if layer_index < 0:
            continue
        # 确保 step_cache 已构建
        if state.step_cache_req_meta_i32 is None or state.step_cache_req_meta_i64 is None:
            continue

        # 获取 has_compact（已在 _build_layer_step_cache 中预计算）
        has_compact = getattr(state, "step_cache_has_compact", False)

        # StepBoundMeta owns the actual per-layer compact K/V binding and
        # refreshes it from state_ref below. Keep StepDecodeData light here:
        # ordered-reuse only needs readiness/hints, not a duplicate view build.
        k_compact = _empty_kv
        v_compact = _empty_kv
        token_positions = _empty_tp

        req_meta_i32_use = state.step_cache_req_meta_i32
        req_meta_i64_use = state.step_cache_req_meta_i64
        if (
            self.step_decode_req_meta_i32_all is not None
            and self.step_decode_req_meta_i64_all is not None
            and layer_index < self.step_decode_req_meta_i32_all.shape[0]
            and layer_index < self.step_decode_req_meta_i64_all.shape[0]
        ):
            req_meta_i32_use = self.step_decode_req_meta_i32_all[layer_index]
            req_meta_i64_use = self.step_decode_req_meta_i64_all[layer_index]

        layer_data = LayerDecodeData(
            req_meta_i32=req_meta_i32_use,
            req_meta_i64=req_meta_i64_use,
            k_compact=k_compact,
            v_compact=v_compact,
            token_positions=token_positions,
            layer_index=layer_index,
            chunk_id=(int(layer_index) // int(_CAPTURE_CHUNK)),
            buf_id=((int(layer_index) // int(_CAPTURE_CHUNK)) % int(_CAPTURE_IN_FLIGHT)),
            has_compact=getattr(state, "step_cache_all_compact", False),
            compact_kv_len_max=(sink_cap + persist_cap) if has_compact else 0,
            page_sparse_enabled=bool(getattr(state, "step_cache_has_page_sparse", False)),
            state_ref=state,
        )
        if 0 <= layer_index < len(layer_data_list):
            layer_data_list[layer_index] = layer_data
            chunk_layer_indices.setdefault(int(layer_data.chunk_id), []).append(int(layer_index))

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
        chunk_layer_indices={
            int(chunk_id): tuple(int(v) for v in layer_indices)
            for chunk_id, layer_indices in chunk_layer_indices.items()
            if int(chunk_id) >= 0 and layer_indices
        } or None,
    )
