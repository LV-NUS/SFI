from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

_log = logging.getLogger(__name__)

import torch
from patches.cpu_gpu_staging import cached_cpu_tensor_to_device, cached_sequence_to_device
from patches.sparse_types import SelectorResult
from utils.selector_key_norms_ext import (
    compute_key_norms_paged_batched_layers_delta_cuda,
)




def _stack_with_cpp_or_fail(
    *,
    tensors: List[torch.Tensor],
    out_buffer: torch.Tensor,
    cpp_call: Callable,
    use_cpp_stack: bool,
    label: str,
) -> torch.Tensor:
    """Use C++ stack extension when enabled; fail-fast if unavailable.

    Args:
        cpp_call: callable that invokes the C++ extension and returns its result.
    Returns the stacked tensor (either from C++ extension or torch.stack).
    """
    stacked = None
    if use_cpp_stack:
        try:
            stacked = cpp_call()
        except Exception:
            _log.warning("C++ %s stack failed", label, exc_info=True)
            raise
    if stacked is not None and isinstance(stacked, tuple):
        result, _ = stacked
        return result
    if use_cpp_stack:
        raise RuntimeError(
            f"selector cpp {label} stack is enabled but unavailable"
        )
    torch.stack(tensors, dim=0, out=out_buffer)
    return out_buffer


def _unpack_detail_events(
    detail_events: Optional[Dict[str, Any]],
) -> Dict[str, Tuple[Any, Any]]:
    """Unpack profiling event pairs from detail_events dict.

    Returns a dict mapping event name to (evt0, evt1) pairs.
    """
    result: Dict[str, Tuple[Any, Any]] = {}
    if detail_events is None or not isinstance(detail_events, dict):
        return result
    for key in ("preproc", "seq_full", "pure_preproc", "bounds", "pipeline", "log_s",
                "log_s_triton", "log_s_mask", "log_s_cross", "topk"):
        pair = detail_events.get(key)
        if isinstance(pair, tuple) and len(pair) == 2:
            result[key] = pair
    return result


def _key_norms_arena_ready(
    arena: torch.Tensor,
    *,
    device: torch.device,
    stride_tokens: int,
    required_slots: int,
    num_kv_heads: int,
) -> bool:
    return (
        isinstance(arena, torch.Tensor)
        and arena.numel() > 0
        and arena.device == device
        and arena.dtype == torch.float16
        and arena.dim() == 3
        and int(arena.shape[0]) >= int(required_slots)
        and int(arena.shape[1]) == int(num_kv_heads)
        and int(arena.shape[2]) >= int(stride_tokens)
    )


def _view_batch_start_from_layer_base(
    view: torch.Tensor,
    base: torch.Tensor,
    *,
    layer_index: int,
    expected_shape: Tuple[int, ...],
) -> Optional[int]:
    """Return the batch offset of a layer-local view into a [L, B, ...] base."""
    if not isinstance(view, torch.Tensor) or not isinstance(base, torch.Tensor):
        return None
    if view.dim() != len(expected_shape) or base.dim() != len(expected_shape) + 1:
        return None
    if tuple(int(v) for v in view.shape) != tuple(int(v) for v in expected_shape):
        return None
    layer_i = int(layer_index)
    if layer_i < 0 or layer_i >= int(base.shape[0]):
        return None
    if any(int(base.shape[i + 1]) < int(expected_shape[i]) for i in range(len(expected_shape))):
        return None
    expected_stride = tuple(int(v) for v in base.stride()[1:])
    if tuple(int(v) for v in view.stride()) != expected_stride:
        return None
    batch_stride = int(base.stride(1))
    if batch_stride <= 0:
        return None
    offset = (
        int(view.storage_offset())
        - int(base.storage_offset())
        - layer_i * int(base.stride(0))
    )
    if offset < 0 or offset % batch_stride != 0:
        return None
    batch_start = offset // batch_stride
    if batch_start < 0 or batch_start + int(expected_shape[0]) > int(base.shape[1]):
        return None
    return int(batch_start)


def _stack_layer_storage_views_if_strided(
    tensors: Sequence[torch.Tensor],
    *,
    expected_shape: Tuple[int, ...],
) -> Optional[torch.Tensor]:
    """Return a zero-copy layer stack when views share strided storage.

    ``torch.inference_mode`` can drop ``Tensor._base`` even for ordinary
    capture-ring views.  Production should still consume the direct tape
    without copying the large logits tensor, so prove the same contract from
    storage pointer + offsets + strides and materialize the stack with
    ``as_strided``.
    """
    if not tensors:
        return None
    first = tensors[0]
    if not isinstance(first, torch.Tensor):
        return None
    if first.dim() != len(expected_shape):
        return None
    if tuple(int(v) for v in first.shape) != tuple(int(v) for v in expected_shape):
        return None
    if len(tensors) == 1:
        return first.unsqueeze(0)

    try:
        storage = first.untyped_storage()
        storage_key = int(storage.data_ptr())
        storage_elems = int(storage.nbytes()) // max(1, int(first.element_size()))
    except Exception:
        return None
    first_offset = int(first.storage_offset())
    view_stride = tuple(int(v) for v in first.stride())
    offsets: List[int] = []
    for tensor in tensors:
        if not isinstance(tensor, torch.Tensor):
            return None
        if tensor.device != first.device or tensor.dtype != first.dtype:
            return None
        if tuple(int(v) for v in tensor.shape) != tuple(int(v) for v in expected_shape):
            return None
        if tuple(int(v) for v in tensor.stride()) != view_stride:
            return None
        try:
            if int(tensor.untyped_storage().data_ptr()) != storage_key:
                return None
        except Exception:
            return None
        offsets.append(int(tensor.storage_offset()))
    layer_stride = int(offsets[1] - offsets[0])
    if layer_stride <= 0:
        return None
    for idx, offset in enumerate(offsets):
        if int(offset - first_offset) != int(idx * layer_stride):
            return None
    max_inner_offset = 0
    for size, stride in zip(expected_shape, view_stride):
        dim = int(size)
        if dim <= 0:
            return None
        max_inner_offset += (dim - 1) * int(stride)
    last_required_offset = first_offset + (len(tensors) - 1) * layer_stride + max_inner_offset
    if last_required_offset < 0 or last_required_offset >= storage_elems:
        return None
    try:
        return first.as_strided(
            (len(tensors), *tuple(int(v) for v in expected_shape)),
            (layer_stride, *view_stride),
            storage_offset=first_offset,
        )
    except Exception:
        return None


def _fill_key_norms_visible_lens_cpu_i32(
    out: torch.Tensor,
    seq_lens_cpu: Sequence[int],
    *,
    batch_size: int,
    phase: str,
    one_shot_bootstrap_only: bool,
    block_size: int,
    recent_tokens: int,
    cap_key_norms: int,
) -> torch.Tensor:
    if out.device.type != "cpu" or out.dtype != torch.int32 or out.numel() < batch_size:
        raise RuntimeError("key_norms target buffer must be a CPU int32 tensor with batch capacity")
    cap = max(0, int(cap_key_norms))
    trim_recent = (
        bool(one_shot_bootstrap_only)
        and str(phase) in ("prefill", "decode")
        and int(block_size) > 0
        and int(recent_tokens) > 0
    )
    block = max(1, int(block_size))
    recent = max(0, int(recent_tokens))
    for i in range(int(batch_size)):
        seq_len = max(0, int(seq_lens_cpu[i]))
        if trim_recent:
            cap_recent = min(seq_len, recent)
            target = ((seq_len - cap_recent) // block) * block
        else:
            target = seq_len
        if target < 0:
            target = 0
        elif target > cap:
            target = cap
        out[i] = int(target)
    return out[:batch_size]


def _bucket_topk_slice_end(slice_start: int, slice_end: int, kv_len_total: int) -> int:
    """#13(c) K-bucket: round the topk slice width UP to a 256 multiple.

    Pure CPU helper (no triton/torch) so the bucket math is unit-testable.
    Returns the bucketed end ``slice_start + align_up(W, 256)`` when it still
    fits inside the buffer (``<= kv_len_total``); otherwise CLAMP-FALLBACK to
    the original un-bucketed ``slice_end`` (never emit a non-256-aligned
    widened width).
    """
    start = int(slice_start)
    end = int(slice_end)
    cap = int(kv_len_total)
    width = end - start
    width256 = ((width + 255) // 256) * 256
    bucketed_end = start + width256
    if bucketed_end <= cap:
        return bucketed_end
    # clamp-fallback: bucketed end overflows the buffer -> keep original end.
    return end


def compute_alpha_selection_batched_impl(
    self: Any,
    payloads: Any,
    *,
    require_base: bool = False,
    phase: str = "decode",
    align_up_int_fn: Callable[[int, int], int],
    selector_cpp_stack_cached: bool,
    get_selector_batch_ext_fn: Callable[[], object],
    rebuild_physical_block_sort_cached: bool,
    selector_kbucket_cached: bool = False,
):
    _align_up_int = align_up_int_fn
    _SELECTOR_CPP_STACK_CACHED = bool(selector_cpp_stack_cached)
    _get_selector_batch_ext = get_selector_batch_ext_fn
    _REBUILD_PHYSICAL_BLOCK_SORT_CACHED = bool(rebuild_physical_block_sort_cached)
    if not payloads:
        return None
    if self.config is None or self.config.alpha_fair is None:
        raise RuntimeError("alpha selector requires config.alpha_fair")

    first = payloads[0]
    batch_size, num_heads, window, kv_len_total = first.capture_scores.shape
    if batch_size <= 0:
        return None
    slot_list_ref = first.slot_list if isinstance(first.slot_list, list) else list(first.slot_list)
    if len(slot_list_ref) != batch_size:
        raise ValueError("slot_list length must match capture batch size")
    num_kv_heads = first.state.num_kv_heads
    if num_kv_heads <= 0:
        raise ValueError("num_kv_heads must be > 0 for alpha selector.")
    if first.state.num_heads != num_heads:
        raise ValueError("capture_scores num_heads mismatch")
    num_queries_per_kv = num_heads // num_kv_heads
    if num_queries_per_kv <= 0:
        raise ValueError("num_queries_per_kv must be > 0")

    device = first.capture_scores.device
    key_cache_dtype = first.key_cache.dtype

    max_slot = max(slot_list_ref) if slot_list_ref else -1

    layer_indices: Optional[List[int]] = None
    layer_indices_expected = True
    capture_scores_all: List[torch.Tensor] = []
    log_f_denoms_all: Optional[List[torch.Tensor]] = None
    kv_lengths_all: Optional[List[torch.Tensor]] = None

    # 优化：批量调用 ensure_batch（避免 32 次函数调用开销）
    if max_slot >= 0:
        batch_target = max_slot + 1
        for p in payloads:
            p.state.ensure_batch(batch_target)
    layer_index_by_cache_key = getattr(self, "layer_index_by_cache_key", {})
    key_norms_layer_indices = tuple(
        int(
            payload.layer_index
            if int(getattr(payload, "layer_index", -1)) >= 0
            else layer_index_by_cache_key.get(payload.cache_key, payload_index)
        )
        for payload_index, payload in enumerate(payloads)
    )
    key_norms_kv_bucket = _align_up_int(int(kv_len_total) + 1, 4096)
    key_norms_cache_key = (
        str(phase),
        key_norms_layer_indices,
        tuple(int(slot) for slot in slot_list_ref),
        int(batch_size),
        int(num_kv_heads),
        int(key_norms_kv_bucket),
        str(device),
        "float16",
    )
    block_size_ref: Optional[int] = None
    head_dim_ref: Optional[int] = None
    row_list_ref: Optional[List[int]] = None
    key_norms_all = self._ensure_selector_key_norms_buffer(
        layers=len(payloads),
        batch=batch_size,
        num_kv_heads=num_kv_heads,
        kv_len=kv_len_total,
        device=device,
        dtype=torch.float16,
        cache_key=key_norms_cache_key,
    )
    key_norms_active_valid = bool(self._selector_key_norms_active_buffer_valid())

    capture_base = getattr(first.capture_scores, "_base", None)
    use_capture_base = (
        capture_base is not None
        and isinstance(capture_base, torch.Tensor)
        and capture_base.dim() == 5
        and capture_base.shape[1] >= batch_size
        and capture_base.shape[2] == num_heads
        and capture_base.shape[3] == window
        and capture_base.shape[4] >= kv_len_total
    )
    denoms_ref = first.log_f_denoms
    denoms_base = getattr(denoms_ref, "_base", None) if denoms_ref is not None else None
    use_denoms_base = (
        denoms_ref is not None
        and denoms_base is not None
        and isinstance(denoms_base, torch.Tensor)
        and denoms_base.dim() == 3
        and denoms_ref.device == device
        and denoms_ref.dtype == torch.float32
        and denoms_base.shape[1] >= int(denoms_ref.shape[0])
        and denoms_base.shape[2] == int(denoms_ref.shape[1])
    )
    kv_lengths_ref = first.kv_lengths
    kv_lengths_direct_ref = kv_lengths_ref
    use_kv_lengths_direct_ref = (
        kv_lengths_direct_ref is not None
        and kv_lengths_direct_ref.device == device
        and kv_lengths_direct_ref.dtype == torch.long
        and kv_lengths_direct_ref.shape == (batch_size, num_heads)
    )

    block_table_ref: Optional[torch.Tensor] = None
    # 方案 I 优化：使用辅助方法判断 profile 并批量创建 events
    profile_detail = bool(getattr(self, "_refresh_profile_active", False)) and bool(
        self._refresh_profile_detail_enabled()
    )
    profile_cpu_validate_us: Optional[float] = None
    profile_cpu_key_norms_us: Optional[float] = None
    profile_cpu_key_norms_arena_us: Optional[float] = None
    profile_cpu_key_norms_direct_us: Optional[float] = None
    profile_cpu_key_norms_direct_prepare_us: Optional[float] = None
    profile_cpu_key_norms_direct_launch_us: Optional[float] = None
    profile_cpu_key_norms_pack_us: Optional[float] = None
    profile_cpu_select_us: Optional[float] = None
    t_validate0_ns: Optional[int] = time.perf_counter_ns() if profile_detail else None
    _main_evts = self._create_profile_events_batch(12, device, profile_detail)
    (
        profile_gather_evt0,
        profile_gather_evt1,
        profile_key_norms_preproc_evt0,
        profile_key_norms_preproc_evt1,
        profile_key_norms_evt0,
        profile_key_norms_evt1,
        profile_key_norms_h2d_evt0,
        profile_key_norms_h2d_evt1,
        profile_key_norms_delta_evt0,
        profile_key_norms_delta_evt1,
        profile_key_norms_pack_evt0,
        profile_key_norms_pack_evt1,
    ) = _main_evts

    # ========== 优化：利用 payloads 同质性减少检查 ==========
    # 首个 payload 做完整验证，后续 payload 只验证关键变化属性并收集数据
    expected_cs_shape = (batch_size, num_heads, window, kv_len_total)
    expected_kv_shape = (batch_size, num_heads)
    expected_denoms_shape = (batch_size, num_heads) if denoms_ref is not None else None
    capture_base_batch_start: Optional[int] = None
    denoms_base_batch_start: Optional[int] = None
    trusted_shapes = self._selector_trusted_shapes_enabled(phase=phase)
    fast_sig_ok = False
    if self._selector_fast_sig_enabled(phase=phase):
        try:
            sig = getattr(first, "fast_signature", None)
            if sig is not None:
                fast_sig_ok = all(getattr(p, "fast_signature", None) == sig for p in payloads)
        except Exception:
            _log.warning("fast_signature check failed, falling back to full validation", exc_info=True)
            fast_sig_ok = False
    if fast_sig_ok:
        # fast signature 只允许跳过重 shape 检查；共享 view/base 仍必须逐层证明。
        trusted_shapes = True
    # (2026-07-03 晚) 旧降级门"row_tensor/row_tensor_i32/kv_len_per_row_i32 非
    # None 才 trusted"已删:三字段的值消费者已全部退休(DETERMINISTIC-SLOT-SOURCE
    # 同源化,恒从 host list 构造),presence 检查是死维护且诱导 live 视图回潮。

    # [BASE-PROOF-GEN-CACHE 2026-07-05] base/direct-ref 证明的世代缓存：证明结论
    # 只取决于 (base 对象身份, 每层 view 的 ptr/shape/stride[已由 fast_signature
    # 涵盖], layer_index 序, batch, kv bucket)——同 (chunk,buf) 世代间逐位相同，
    # 旧实现每层×每 chunk 重跑 _view_batch_start 的 tuple churn（~110-160µs/chunk）。
    # 键含 fast_sig 值比较（payload 每 flush 新建，id 不稳）；值持 base 强引用
    # （id 在持有期恒有效+命中时 is 双验）。capture ring realloc/K-bucket 跨界/
    # 批组成变化均经由键自身失效——zero-stale-by-construction。仅 fast_sig_ok
    # （全层签名可用）时启用；命中后 slot/row 逐层等值活护栏仍保留（循环内）。
    _proof_cache_key = None
    _proof_hit = None
    if fast_sig_ok and (use_capture_base or use_denoms_base or use_kv_lengths_direct_ref):
        _proof_cache_key = (
            id(capture_base),
            id(denoms_base),
            int(kv_lengths_direct_ref.data_ptr()) if kv_lengths_direct_ref is not None else -1,
            tuple(getattr(p, "fast_signature", None) for p in payloads),
            int(batch_size),
            int(kv_len_total),
        )
        _proof_cache = getattr(self, "_selector_base_proof_cache", None)
        if _proof_cache is not None:
            _cand = _proof_cache.get(_proof_cache_key)
            if (
                _cand is not None
                and _cand[7] is capture_base
                and _cand[8] is denoms_base
            ):
                _proof_hit = _cand
    if _proof_hit is not None:
        use_capture_base = bool(_proof_hit[0])
        capture_base_batch_start = _proof_hit[1]
        use_denoms_base = bool(_proof_hit[2])
        denoms_base_batch_start = _proof_hit[3]
        use_kv_lengths_direct_ref = bool(_proof_hit[4])
        layer_indices = list(_proof_hit[5]) if _proof_hit[5] is not None else None
        layer_indices_expected = bool(_proof_hit[6])

    for layer_idx, payload in enumerate(payloads):
        # 首个 payload：完整检查
        if layer_idx == 0:
            if not trusted_shapes:
                if payload.capture_scores.shape != expected_cs_shape:
                    raise ValueError("capture_scores shape mismatch across layers")
                if payload.kv_lengths.shape != expected_kv_shape:
                    raise ValueError("kv_lengths shape mismatch across layers")
                if denoms_ref is None:
                    if payload.log_f_denoms is not None and payload.log_f_denoms.numel() > 0:
                        raise ValueError("log_f_denoms mismatch across layers (expected None)")
                else:
                    if payload.log_f_denoms is None or payload.log_f_denoms.shape != expected_denoms_shape:
                        raise ValueError("log_f_denoms must have shape [batch, num_heads] when provided")
                if payload.state.num_kv_heads != num_kv_heads or payload.state.num_heads != num_heads:
                    raise ValueError("layer state head configuration mismatch")
                if len(payload.row_list) != batch_size:
                    raise RuntimeError("row_list length mismatch")
                row_list_ref = payload.row_list if isinstance(payload.row_list, list) else list(payload.row_list)
                if payload.key_cache.dtype != key_cache_dtype:
                    raise RuntimeError("key_cache dtype mismatch across layers")
                if payload.capture_scores.device != device:
                    raise RuntimeError("capture_scores device mismatch across layers")
                if payload.block_table.device != payload.key_cache.device:
                    raise RuntimeError("block_table device mismatch with key_cache")
                if payload.key_cache.dim() < 4:
                    raise RuntimeError("key_cache must be 4D for key_norms")
                block_size_ref = int(payload.key_cache.shape[1]) if payload.key_cache.dim() >= 2 else 0
                head_dim_ref = int(payload.key_cache.shape[-1])
                if head_dim_ref <= 0:
                    raise RuntimeError("head_dim must be > 0 for key_norms")
                block_table_ref = payload.block_table
            else:
                row_list_ref = payload.row_list if isinstance(payload.row_list, list) else list(payload.row_list)
                block_table_ref = payload.block_table
                if payload.key_cache.dim() < 4:
                    raise RuntimeError("key_cache must be 4D for key_norms")
                block_size_ref = int(payload.key_cache.shape[1]) if payload.key_cache.dim() >= 2 else 0
                head_dim_ref = int(payload.key_cache.shape[-1])
                if head_dim_ref <= 0:
                    raise RuntimeError("head_dim must be > 0 for key_norms")
        else:
            slot_list_cur = payload.slot_list if isinstance(payload.slot_list, list) else list(payload.slot_list)
            if len(slot_list_cur) != batch_size:
                raise RuntimeError("slot_list length mismatch across layers")
            if slot_list_cur != slot_list_ref:
                raise RuntimeError(
                    f"cross-layer slot_list mismatch at layer_idx={layer_idx}: "
                    f"ref={slot_list_ref} got={slot_list_cur}"
                )

            row_list_cur = payload.row_list if isinstance(payload.row_list, list) else list(payload.row_list)
            if len(row_list_cur) != batch_size:
                raise RuntimeError("row_list length mismatch")
            if row_list_ref is None:
                row_list_ref = row_list_cur
            elif row_list_cur != row_list_ref:
                raise RuntimeError(
                    f"cross-layer row_list mismatch at layer_idx={layer_idx}: "
                    f"ref={row_list_ref} got={row_list_cur}"
                )

            if payload.key_cache.dim() < 4:
                raise RuntimeError("key_cache must be 4D for key_norms")
            head_dim_cur = int(payload.key_cache.shape[-1])
            if head_dim_cur <= 0:
                raise RuntimeError("head_dim must be > 0 for key_norms")
            if payload.block_table.device != payload.key_cache.device:
                raise RuntimeError("block_table device mismatch with key_cache")

            if not trusted_shapes:
                if payload.capture_scores.shape != expected_cs_shape:
                    raise ValueError("capture_scores shape mismatch across layers")
                if payload.kv_lengths.shape != expected_kv_shape:
                    raise ValueError("kv_lengths shape mismatch across layers")
                if denoms_ref is None:
                    if payload.log_f_denoms is not None and payload.log_f_denoms.numel() > 0:
                        raise ValueError("log_f_denoms mismatch across layers (expected None)")
                else:
                    if payload.log_f_denoms is None or payload.log_f_denoms.shape != expected_denoms_shape:
                        raise ValueError("log_f_denoms must have shape [batch, num_heads] when provided")
                if payload.state.num_kv_heads != num_kv_heads or payload.state.num_heads != num_heads:
                    raise ValueError("layer state head configuration mismatch")
                if payload.key_cache.dtype != key_cache_dtype:
                    raise RuntimeError("key_cache dtype mismatch across layers")
                if payload.capture_scores.device != device:
                    raise RuntimeError("capture_scores device mismatch across layers")
                block_size_cur = int(payload.key_cache.shape[1]) if payload.key_cache.dim() >= 2 else 0
                if block_size_ref is not None and block_size_cur != block_size_ref:
                    raise RuntimeError("key_cache block_size mismatch across layers")
                if head_dim_ref is not None and head_dim_cur != head_dim_ref:
                    raise RuntimeError("key_cache head_dim mismatch across layers")

        # 所有 payload：收集数据
        # 注意：ensure_batch 已在循环前批量调用

        # [BASE-PROOF-GEN-CACHE] 缓存命中时 layer_indices 与 proof 结论均取自
        # 缓存（键含逐层 fast_signature，输入逐位相同⇒结论逐位相同）。
        if _proof_hit is not None:
            continue

        layer_index = payload.layer_index if payload.layer_index >= 0 else self.layer_index_by_cache_key.get(payload.cache_key, -1)
        if layer_indices_expected:
            if int(layer_index) != int(layer_idx):
                layer_indices_expected = False
                layer_indices = list(range(layer_idx))
                layer_indices.append(int(layer_index))
        else:
            if layer_indices is None:
                layer_indices = []
            layer_indices.append(int(layer_index))

        # base/direct-ref proof is correctness-critical. Decode fast path may
        # skip heavy shape checks, but it must never skip shared storage proof.
        if use_capture_base:
            base_candidate = getattr(payload.capture_scores, "_base", None)
            view_batch_start = (
                _view_batch_start_from_layer_base(
                    payload.capture_scores,
                    capture_base,
                    layer_index=int(layer_index),
                    expected_shape=expected_cs_shape,
                )
                if base_candidate is capture_base and isinstance(capture_base, torch.Tensor)
                else None
            )
            if (
                base_candidate is not capture_base
                or payload.capture_scores.device != device
                or payload.capture_scores.shape != expected_cs_shape
                or view_batch_start is None
            ):
                use_capture_base = False
            elif capture_base_batch_start is None:
                capture_base_batch_start = int(view_batch_start)
            elif int(capture_base_batch_start) != int(view_batch_start):
                use_capture_base = False
        if use_denoms_base and denoms_ref is not None:
            base_candidate = (
                getattr(payload.log_f_denoms, "_base", None)
                if payload.log_f_denoms is not None
                else None
            )
            view_batch_start = (
                _view_batch_start_from_layer_base(
                    payload.log_f_denoms,
                    denoms_base,
                    layer_index=int(layer_index),
                    expected_shape=expected_denoms_shape,
                )
                if (
                    base_candidate is denoms_base
                    and isinstance(denoms_base, torch.Tensor)
                    and expected_denoms_shape is not None
                    and payload.log_f_denoms is not None
                )
                else None
            )
            if (
                base_candidate is not denoms_base
                or payload.log_f_denoms is None
                or payload.log_f_denoms.device != device
                or payload.log_f_denoms.dtype != torch.float32
                or payload.log_f_denoms.shape != expected_denoms_shape
                or view_batch_start is None
            ):
                use_denoms_base = False
            elif denoms_base_batch_start is None:
                denoms_base_batch_start = int(view_batch_start)
            elif int(denoms_base_batch_start) != int(view_batch_start):
                use_denoms_base = False
        if use_kv_lengths_direct_ref:
            base_candidate = payload.kv_lengths
            if (
                base_candidate.device != device
                or base_candidate.dtype != torch.long
                or base_candidate.shape != kv_lengths_direct_ref.shape
                or base_candidate.data_ptr() != kv_lengths_direct_ref.data_ptr()
            ):
                use_kv_lengths_direct_ref = False
        # 注意：log_f_denoms_all 和 kv_lengths_all 改为惰性构建，避免热路径无效分配。

    # [BASE-PROOF-GEN-CACHE] miss 时把本世代证明结论写缓存（含 base 强引用）。
    if _proof_cache_key is not None and _proof_hit is None:
        _proof_cache = getattr(self, "_selector_base_proof_cache", None)
        if _proof_cache is None:
            _proof_cache = {}
            self._selector_base_proof_cache = _proof_cache
        elif len(_proof_cache) > 16:
            # 层段×buf×批形态的合法组合为个位数；超界=键高频漂（防泄漏上界）。
            _proof_cache.clear()
        _proof_cache[_proof_cache_key] = (
            bool(use_capture_base),
            capture_base_batch_start,
            bool(use_denoms_base),
            denoms_base_batch_start,
            bool(use_kv_lengths_direct_ref),
            tuple(layer_indices) if layer_indices is not None else None,
            bool(layer_indices_expected),
            capture_base,
            denoms_base,
        )

    if layer_indices is None:
        selector_layer_start = 0
        selector_layer_end = len(payloads) - 1
    else:
        selector_layer_start = min(layer_indices) if layer_indices else 0
        selector_layer_end = max(layer_indices) if layer_indices else len(payloads) - 1
    selector_chunk_id = int(getattr(getattr(first, "state", None), "capture_chunk_id", -1) or -1)
    selector_buf_id = int(getattr(getattr(first, "state", None), "capture_buf_id", -1) or -1)
    selector_layer_span_cache_suffix = (
        f"_l{selector_layer_start}_{selector_layer_end}"
        f"_c{selector_chunk_id}_b{selector_buf_id}"
    )

    if profile_detail and t_validate0_ns is not None:
        profile_cpu_validate_us = (time.perf_counter_ns() - t_validate0_ns) / 1000.0
    t_key_norms0_ns: Optional[int] = time.perf_counter_ns() if profile_detail else None

    # key_norms：P0.1 增量更新（delta 到 target_len），避免 refresh 每次扫全量 K
    self._record_event_safe(profile_key_norms_preproc_evt0, device)

    # --------------------------------------------------------------
    # P0.1：key_norms 增量更新（delta 到 target_len），避免 refresh 每次扫全量 K
    # --------------------------------------------------------------
    # 方案 I 优化：使用辅助方法记录 event
    self._record_event_safe(profile_key_norms_preproc_evt1, device)
    self._record_event_safe(profile_key_norms_evt0, device)

    # [DETERMINISTIC-SLOT-SOURCE 2026-07-03] 恒从 slot_list(host 权威值快照)
    # 构造,勿改回 payload 的 slot_tensor(复用 buffer 的 live 视图,错峰
    # bootstrap 下一行 flush 会原位覆写):它喂非连续 slots 分支的
    # arena.index_select 打包,脏 slot=打包错行 norms 的静默错选。
    slot_tensor_ref = cached_sequence_to_device(
        slot_list_ref,
        device=device,
        dtype=torch.long,
        cache_name="selector_slot_tensor_i64",
        cache_owner=self,
        reuse_unchanged=True,
    )

    # key_norms arena stride：优先使用 capture_base 的 K 维（已按 256 对齐，可跨 step 稳定复用）
    kv_stride_tokens = 0
    if use_capture_base and isinstance(capture_base, torch.Tensor) and capture_base.dim() == 5:
        kv_stride_tokens = int(capture_base.shape[4])
    if kv_stride_tokens <= 0:
        kv_stride_tokens = _align_up_int(int(kv_len_total), 256)

    # 方案 A 优化：将重复的 tensor 创建移到循环外（节省 ~670us）
    # slot_indices_cpu 和 seq_lens_tensor 在所有层中相同，只需创建一次
    # [DETERMINISTIC-SLOT-SOURCE 2026-07-03] 恒从 slot_list(host 权威值快照)构造,
    # 勿改回 payload 的 slot_tensor_cpu(CPU 复用 buffer 的 live 视图,错峰
    # bootstrap 下一行 flush 会原位覆写):它喂 delta 的 start_lens 读取(index_select)
    # 与 key_norms_len 簿记发布(index_copy_),读错/写错 slot 都是静默错数。
    slot_tensor_cpu_hit = False
    slot_indices_cpu = torch.tensor(slot_list_ref, dtype=torch.long)
    first_seq_lens_cpu = first.seq_lens_cpu
    if first_seq_lens_cpu is None or len(first_seq_lens_cpu) < batch_size:
        raise RuntimeError("key_norms delta requires seq_lens_cpu")
    # [注 2026-07-03 晚更新] seq_lens_tensor_cpu 自确定性快照族起已是 flush 入
    # 口的提交步快照([DETERMINISTIC-LEN-SNAPSHOT],flush_worker by-ptr clone),
    # 此处读到的即提交步值;黄金判据已重锚定(3ff97b89/0d5f663c)。旧"保持实时
    # 语义/t14 实证"表述随旧黄金 5b2f4444 一并作废。
    seq_lens_tensor_cpu_ref = getattr(first, "seq_lens_tensor_cpu", None)
    seq_lens_tensor_cpu_hit = False
    if (
        isinstance(seq_lens_tensor_cpu_ref, torch.Tensor)
        and seq_lens_tensor_cpu_ref.device.type == "cpu"
        and seq_lens_tensor_cpu_ref.numel() >= batch_size
    ):
        if seq_lens_tensor_cpu_ref.dtype == torch.long:
            seq_lens_tensor = seq_lens_tensor_cpu_ref[:batch_size]
        else:
            seq_lens_tensor = seq_lens_tensor_cpu_ref[:batch_size].to(dtype=torch.long)
        seq_lens_tensor_cpu_hit = True
    else:
        seq_lens_tensor = torch.tensor(
            [max(0, int(s)) for s in first_seq_lens_cpu[:batch_size]],
            dtype=torch.long,
        )
    cap_key_norms = min(int(kv_len_total), int(kv_stride_tokens))
    semantic_snapshot = self._get_step_semantic_snapshot() if self.config is not None else None

    any_delta = False
    profile_key_norms_delta_total_tokens = 0
    profile_key_norms_delta_max_tokens = -1
    profile_key_norms_delta_layers = 0
    key_norms_cur_nonzero_count = 0
    max_delta_direct_value = 0
    # 方案 F+G 合并优化：单次遍历同时完成 ensure_batch + index_select + ensure_arena
    recent_cfg_direct = (
        int(semantic_snapshot.recent_tokens)
        if semantic_snapshot is not None
        else 0
    )
    tgt_lens_global_i32 = self._ensure_selector_key_norms_target_cpu_buffer(
        batch=batch_size,
    )
    _fill_key_norms_visible_lens_cpu_i32(
        tgt_lens_global_i32,
        first_seq_lens_cpu,
        batch_size=batch_size,
        phase=phase,
        one_shot_bootstrap_only=bool(
            getattr(getattr(self, "config", None), "one_shot_bootstrap_only", False)
        ),
        block_size=int(block_size_ref or 0),
        recent_tokens=recent_cfg_direct,
        cap_key_norms=cap_key_norms,
    )
    t_key_norms_arena0_ns: Optional[int] = (
        time.perf_counter_ns() if profile_detail else None
    )
    _arenas: List[torch.Tensor] = []
    _kv_stride_int = int(kv_stride_tokens)
    _required_slots = max_slot + 1
    _num_kv_heads_int = int(num_kv_heads)
    _slot0 = int(slot_list_ref[0]) if slot_list_ref else 0
    _slotN = int(slot_list_ref[-1]) if slot_list_ref else -1
    _slots_contiguous = (_slotN - _slot0 + 1 == batch_size)
    _single_slot_delta_fast_path = bool(_slots_contiguous and int(batch_size) == 1)
    (
        _stacked_start_cpu,
        _stacked_end_cpu,
        _stacked_start_gpu_buf,
        _stacked_end_gpu_buf,
    ) = self._ensure_selector_key_norms_delta_buffers(
        layers=len(payloads),
        batch=batch_size,
        device=device,
    )
    if _single_slot_delta_fast_path:
        target_len_single = int(tgt_lens_global_i32[0])
        delta_total_tokens = 0
        delta_layers = 0
        for layer_idx, payload in enumerate(payloads):
            st = payload.state
            start_len = int(st.key_norms_len[_slot0])
            end_len = start_len if start_len >= target_len_single else target_len_single
            _stacked_start_cpu[layer_idx, 0] = int(start_len)
            _stacked_end_cpu[layer_idx, 0] = int(end_len)
            delta = int(end_len) - int(start_len)
            if start_len != 0:
                key_norms_cur_nonzero_count += 1
            if delta > 0:
                any_delta = True
                delta_total_tokens += delta
                delta_layers += 1
            if delta > max_delta_direct_value:
                max_delta_direct_value = delta
            if not _key_norms_arena_ready(
                st.key_norms_arena,
                device=device,
                stride_tokens=_kv_stride_int,
                required_slots=_required_slots,
                num_kv_heads=_num_kv_heads_int,
            ):
                st.ensure_key_norms_arena(
                    stride_tokens=_kv_stride_int,
                    required_slots=_required_slots,
                    num_kv_heads=_num_kv_heads_int,
                    refresh_stream=self.refresh_stream,
                    stride_floor=int(
                        getattr(self, "_key_norms_stride_floor_mml", 0) or 0
                    ),
                )
            arena = st.key_norms_arena
            if arena.numel() == 0:
                raise RuntimeError("key_norms_arena missing after ensure_key_norms_arena")
            _arenas.append(arena)
        if profile_detail:
            profile_key_norms_delta_total_tokens = int(delta_total_tokens)
            if any_delta:
                profile_key_norms_delta_max_tokens = int(max_delta_direct_value)
                profile_key_norms_delta_layers = int(delta_layers)
    else:
        for layer_idx, payload in enumerate(payloads):
            st = payload.state
            if _slots_contiguous:
                _stacked_start_cpu[layer_idx].copy_(
                    st.key_norms_len.narrow(0, _slot0, batch_size)
                )
            else:
                _stacked_start_cpu[layer_idx].copy_(
                    st.key_norms_len.index_select(0, slot_indices_cpu)
                )
            if not _key_norms_arena_ready(
                st.key_norms_arena,
                device=device,
                stride_tokens=_kv_stride_int,
                required_slots=_required_slots,
                num_kv_heads=_num_kv_heads_int,
            ):
                st.ensure_key_norms_arena(
                    stride_tokens=_kv_stride_int,
                    required_slots=_required_slots,
                    num_kv_heads=_num_kv_heads_int,
                    refresh_stream=self.refresh_stream,
                    stride_floor=int(
                        getattr(self, "_key_norms_stride_floor_mml", 0) or 0
                    ),
                )
            arena = st.key_norms_arena
            if arena.numel() == 0:
                raise RuntimeError("key_norms_arena missing after ensure_key_norms_arena")
            _arenas.append(arena)
        torch.maximum(
            _stacked_start_cpu,
            tgt_lens_global_i32.unsqueeze(0),
            out=_stacked_end_cpu,
        )
        key_norms_delta_matrix = _stacked_end_cpu - _stacked_start_cpu
        if key_norms_delta_matrix.numel() > 0:
            max_delta_direct_value = int(key_norms_delta_matrix.max().item())
            any_delta = max_delta_direct_value > 0
            key_norms_cur_nonzero_count = int(torch.count_nonzero(_stacked_start_cpu).item())
            if profile_detail:
                profile_key_norms_delta_total_tokens = int(
                    key_norms_delta_matrix.sum().item()
                )
            if any_delta:
                profile_key_norms_delta_max_tokens = max_delta_direct_value
                profile_key_norms_delta_layers = int(
                    torch.count_nonzero(
                        torch.amax(key_norms_delta_matrix, dim=1) > 0
                    ).item()
                )
    max_delta_override = getattr(self, "_selector_key_norms_max_delta_override", None)
    if max_delta_override is not None:
        try:
            max_delta_override_int = int(max_delta_override)
        except (TypeError, ValueError):
            max_delta_override_int = 0
        if max_delta_override_int > max_delta_direct_value:
            max_delta_direct_value = int(max_delta_override_int)
            any_delta = True
    # [DETERMINISTIC-SLOT-SOURCE 2026-07-03] 原地曾有 slot_tensor_i32_ref/
    # row_indices_i32_ref 的 live 视图捕获与 dtype 转换(deferred 实读),delta
    # kernel 改用 list 同源张量后成为死存储,整块删除防回潮。

    # Phase 2: 预计算循环不变量
    _use_direct_layers_key_norms = False
    _direct_key_norms_written = False
    _direct_key_norms_scratch_needs_pack = False
    if (
        any_delta
        and phase in ("prefill", "decode")
        and bool(getattr(getattr(self, "config", None), "one_shot_bootstrap_only", False))
        and block_table_ref is not None
        and block_size_ref is not None
        and head_dim_ref is not None
    ):
        cur_is_empty = key_norms_cur_nonzero_count == 0
        key_strides_ref = tuple(int(s) for s in first.key_cache.stride())
        same_key_strides = all(
            tuple(int(s) for s in payload.key_cache.stride()) == key_strides_ref
            for payload in payloads
        )
        if cur_is_empty and not same_key_strides:
            raise RuntimeError(
                "one-shot direct key_norms requires identical key_cache strides"
            )
        _use_direct_layers_key_norms = bool(same_key_strides)
        _direct_key_norms_scratch_needs_pack = bool(
            getattr(self, "_selector_key_norms_all_reallocated", False)
            or not key_norms_active_valid
        ) and not cur_is_empty
    if profile_detail and t_key_norms_arena0_ns is not None:
        profile_cpu_key_norms_arena_us = (
            time.perf_counter_ns() - t_key_norms_arena0_ns
        ) / 1000.0

    t_key_norms_direct0_ns: Optional[int] = (
        time.perf_counter_ns() if profile_detail else None
    )

    # Phase 3: 如果有 delta，将 stacked_cur / stacked_tgt 批量转换+拷贝到 GPU。
    _stacked_start_gpu: Optional[torch.Tensor] = None
    _stacked_end_gpu: Optional[torch.Tensor] = None
    _idx_gpu: Optional[torch.Tensor] = None
    key_ptrs_direct: Optional[torch.Tensor] = None
    key_norms_arena_ptrs: Optional[torch.Tensor] = None
    row_indices_direct: Optional[torch.Tensor] = None
    _key_norm_views: Optional[List[torch.Tensor]] = [] if _slots_contiguous else None

    def _pack_key_norms_from_sidecar() -> None:
        nonlocal profile_cpu_key_norms_pack_us
        # pack key_norms sidecar even when no delta was needed: prefill can
        # warm key_norms_arena before selector, but selector kernels still
        # consume the cross-layer key_norms_all buffer today.
        t_pack0_ns: Optional[int] = time.perf_counter_ns() if profile_detail else None
        self._record_event_safe(profile_key_norms_pack_evt0, device)
        for layer_idx, arena in enumerate(_arenas):
            if _slots_contiguous:
                view = arena.narrow(0, _slot0, batch_size)[:, :, :kv_len_total]
                if _key_norm_views is not None:
                    _key_norm_views.append(view)
            else:
                view = arena.index_select(0, slot_tensor_ref)
                key_norms_all[layer_idx].copy_(view[:, :, :kv_len_total])
        if _key_norm_views is not None:
            torch.stack(_key_norm_views, dim=0, out=key_norms_all)
        self._record_event_safe(profile_key_norms_pack_evt1, device)
        if t_pack0_ns is not None:
            profile_cpu_key_norms_pack_us = (
                time.perf_counter_ns() - t_pack0_ns
            ) / 1000.0

    def _publish_key_norms_lens_from_stacked_end() -> None:
        if not any_delta:
            return
        for layer_idx, payload in enumerate(payloads):
            st = payload.state
            layer_end_cpu_i32 = _stacked_end_cpu[layer_idx]
            if _slots_contiguous:
                st.key_norms_len[_slot0 : _slot0 + batch_size] = layer_end_cpu_i32
            else:
                # [KEY-NORMS-LEN-I32 2026-07-06] 载体统一 int32 后与 delta 源同
                # dtype，index_copy_ 直写（旧 int64 载体时代的对齐 cast 已退休；
                # gpu mirror 死载体同批删除）。
                st.key_norms_len.index_copy_(0, slot_indices_cpu, layer_end_cpu_i32)

    if any_delta:
        t_key_norms_direct_prepare0_ns: Optional[int] = (
            time.perf_counter_ns()
            if profile_detail and _use_direct_layers_key_norms
            else None
        )
        self._record_event_safe(profile_key_norms_h2d_evt0, device)
        _stacked_start_gpu = _stacked_start_gpu_buf
        _stacked_end_gpu = _stacked_end_gpu_buf
        _stacked_start_gpu.copy_(_stacked_start_cpu, non_blocking=True)
        _stacked_end_gpu.copy_(_stacked_end_cpu, non_blocking=True)
        if not _slots_contiguous:
            _idx_gpu = cached_cpu_tensor_to_device(
                slot_indices_cpu,
                device=device,
                dtype=slot_indices_cpu.dtype,
                cache_name=f"selector_slot_indices{selector_layer_span_cache_suffix}",
                cache_owner=self,
            )
        if _use_direct_layers_key_norms:
            key_ptrs_direct = cached_sequence_to_device(
                [int(payload.key_cache.data_ptr()) for payload in payloads],
                device=device,
                dtype=torch.int64,
                cache_name=f"selector_key_ptrs_i64{selector_layer_span_cache_suffix}",
                cache_owner=self,
                reuse_unchanged=True,
            )
            if key_norms_arena_ptrs is None:
                key_norms_arena_ptrs = cached_sequence_to_device(
                    [int(arena.data_ptr()) for arena in _arenas],
                    device=device,
                    dtype=torch.int64,
                    cache_name=f"selector_key_norms_arena_ptrs_i64{selector_layer_span_cache_suffix}",
                    cache_owner=self,
                    reuse_unchanged=True,
                )
            # [DETERMINISTIC-SLOT-SOURCE 2026-07-03] payload 的 row_tensor_i32 与
            # slot_tensor_i32 同为 per-batch 复用 buffer 的 live 视图:错峰
            # bootstrap 下一行的 flush 会原位覆写,deferred selector 的后续层组
            # 晚读时拿到覆写值(slot 失配已被 Qwen3-4B bs2 探针实证)。delta
            # kernel 的 row/slot 输入恒从 row_list/slot_list(host 值快照,与
            # ensure/簿记同源)构造;黄金工况 list 恒定,reuse_unchanged 缓存
            # 命中,数值路径逐字不变。
            if row_list_ref is None:
                raise RuntimeError("direct key_norms layers path requires row_list")
            row_indices_direct = cached_sequence_to_device(
                row_list_ref,
                device=device,
                dtype=torch.int32,
                cache_name=f"selector_row_indices_direct_i32{selector_layer_span_cache_suffix}",
                cache_owner=self,
                reuse_unchanged=True,
            )
            slot_indices_from_list_i32 = cached_sequence_to_device(
                [int(s) for s in slot_list_ref],
                device=device,
                dtype=torch.int32,
                cache_name=f"selector_key_norms_slot_list_i32{selector_layer_span_cache_suffix}",
                cache_owner=self,
                reuse_unchanged=True,
            )
        self._record_event_safe(profile_key_norms_h2d_evt1, device)
        if profile_detail and t_key_norms_direct_prepare0_ns is not None:
            profile_cpu_key_norms_direct_prepare_us = (
                time.perf_counter_ns() - t_key_norms_direct_prepare0_ns
            ) / 1000.0

    if _use_direct_layers_key_norms and any_delta and _stacked_start_gpu is not None:
        if (
            key_ptrs_direct is None
            or key_norms_arena_ptrs is None
            or row_indices_direct is None
            or _stacked_end_gpu is None
            or max_delta_direct_value <= 0
        ):
            raise RuntimeError("direct key_norms delta path missing prepared tensors")
        max_delta_direct = int(max_delta_direct_value)
        t_key_norms_direct_launch0_ns: Optional[int] = (
            time.perf_counter_ns() if profile_detail else None
        )
        self._record_event_safe(profile_key_norms_delta_evt0, device)
        compute_key_norms_paged_batched_layers_delta_cuda(
            key_ptrs=key_ptrs_direct,
            out_ptrs=key_norms_arena_ptrs,
            block_table=block_table_ref,
            start_lens=_stacked_start_gpu,
            end_lens=_stacked_end_gpu,
            # [DETERMINISTIC-SLOT-SOURCE 2026-07-03] 恒用 slot_list 同源张量,
            # 勿改回 payload 的 slot_tensor_i32(live 视图,错峰下被覆写)。
            slot_indices=slot_indices_from_list_i32,
            kv_dtype=key_cache_dtype,
            out_dtype=_arenas[0].dtype,
            head_dim=int(head_dim_ref),
            block_size=int(block_size_ref),
            key_strides=tuple(int(s) for s in first.key_cache.stride()),
            out_strides=tuple(int(s) for s in _arenas[0].stride()),
            max_delta=max_delta_direct,
            row_indices=row_indices_direct,
            scratch_norms=key_norms_all,
        )
        self._record_event_safe(profile_key_norms_delta_evt1, device)
        if profile_detail and t_key_norms_direct_launch0_ns is not None:
            profile_cpu_key_norms_direct_launch_us = (
                time.perf_counter_ns() - t_key_norms_direct_launch0_ns
            ) / 1000.0
        _publish_key_norms_lens_from_stacked_end()
        if _direct_key_norms_scratch_needs_pack:
            _pack_key_norms_from_sidecar()
        else:
            self._record_event_safe(profile_key_norms_pack_evt0, device)
            self._record_event_safe(profile_key_norms_pack_evt1, device)
            if profile_detail:
                profile_cpu_key_norms_pack_us = 0.0
        self._mark_selector_key_norms_active_buffer_valid()
        _direct_key_norms_written = True
    if profile_detail and t_key_norms_direct0_ns is not None:
        profile_cpu_key_norms_direct_us = (
            time.perf_counter_ns() - t_key_norms_direct0_ns
        ) / 1000.0

    if not _direct_key_norms_written:
        if any_delta:
            raise RuntimeError(
                "selector key_norms delta requires CUDA layers delta path; "
                "legacy Triton key_norms fallback is retired"
            )
        self._record_event_safe(profile_key_norms_delta_evt0, device)
        _publish_key_norms_lens_from_stacked_end()
        self._record_event_safe(profile_key_norms_delta_evt1, device)

        if key_norms_active_valid:
            self._record_event_safe(profile_key_norms_pack_evt0, device)
            self._record_event_safe(profile_key_norms_pack_evt1, device)
            if profile_detail:
                profile_cpu_key_norms_pack_us = 0.0
        else:
            _pack_key_norms_from_sidecar()
            self._mark_selector_key_norms_active_buffer_valid()

    # 方案 I 优化：使用辅助方法记录 event
    self._record_event_safe(profile_key_norms_evt1, device)
    if profile_detail and t_key_norms0_ns is not None:
        profile_cpu_key_norms_us = (time.perf_counter_ns() - t_key_norms0_ns) / 1000.0

    if layer_indices_expected:
        layer_indices_valid = True
        max_layer_index = len(payloads) - 1
    else:
        if layer_indices is None:
            layer_indices = []
        layer_indices_valid = len(layer_indices) == len(payloads) and all(idx >= 0 for idx in layer_indices)
        max_layer_index = max(layer_indices) if layer_indices else -1
    if use_capture_base and capture_base is not None and max_layer_index >= 0:
        base_layers = int(capture_base.shape[0]) if capture_base.dim() >= 1 else 0
        if max_layer_index >= base_layers:
            raise RuntimeError(
                "selector payload layer_index must be slot_in_chunk; "
                f"max_layer_index={max_layer_index} base_layers={base_layers}"
            )
    can_use_capture_base = (
        use_capture_base
        and layer_indices_valid
        and capture_base is not None
        and capture_base.shape[0] > max_layer_index
        and capture_base_batch_start is not None
    )
    can_use_denoms_base = (
        use_denoms_base
        and denoms_base is not None
        and layer_indices_valid
        and denoms_base.shape[0] > max_layer_index
        and denoms_base_batch_start is not None
    )
    capture_storage_stack: Optional[torch.Tensor] = None
    if not can_use_capture_base:
        capture_scores_all = [p.capture_scores for p in payloads]
        capture_storage_stack = _stack_layer_storage_views_if_strided(
            capture_scores_all,
            expected_shape=expected_cs_shape,
        )
    denoms_storage_stack: Optional[torch.Tensor] = None
    if denoms_ref is not None and not can_use_denoms_base:
        log_f_denoms_all = []
        for payload in payloads:
            if payload.log_f_denoms is None:
                raise RuntimeError("log_f_denoms missing while denoms_ref is set")
            log_f_denoms_all.append(payload.log_f_denoms)
        denoms_storage_stack = _stack_layer_storage_views_if_strided(
            log_f_denoms_all,
            expected_shape=expected_denoms_shape,
        )
    has_direct_capture_tape = bool(can_use_capture_base or capture_storage_stack is not None)
    if require_base and not has_direct_capture_tape:
        _first_cs = first.capture_scores
        _first_base = getattr(_first_cs, "_base", None)
        raise RuntimeError(
            "selector batched path requires step capture arena base; "
            f"capture_base={bool(can_use_capture_base)} "
            f"capture_storage_stack={bool(capture_storage_stack is not None)} "
            f"denoms_base={bool(can_use_denoms_base)} "
            f"denoms_storage_stack={bool(denoms_storage_stack is not None)} "
            f"kv_lengths_direct={bool(use_kv_lengths_direct_ref)} "
            f"layer_indices={'expected' if layer_indices_expected else layer_indices} "
            f"phase={phase} layers={len(payloads)} "
            f"first_cs_shape={tuple(_first_cs.shape)} "
            f"first_base_shape={tuple(_first_base.shape) if isinstance(_first_base, torch.Tensor) else None} "
            f"expected_cs_shape={tuple(expected_cs_shape)} "
            f"batch_size={batch_size} num_heads={num_heads} window={window} "
            f"kv_len_total={kv_len_total} "
            f"use_capture_base_initial={use_capture_base} "
            f"capture_base_batch_start={capture_base_batch_start} "
            f"layer_storage_fps={[(int(p.capture_scores.untyped_storage().data_ptr()), int(p.capture_scores.storage_offset()), tuple(p.capture_scores.shape), tuple(p.capture_scores.stride())) for p in payloads[:4]]} "
            f"payload_ids={[id(p) for p in payloads]} "
            f"cs_ids={[id(p.capture_scores) for p in payloads[:6]]} "
            f"slot_lists={[tuple(p.slot_list) if p.slot_list is not None else None for p in payloads[:4]]} "
            f"cache_keys={[int(getattr(p, 'cache_key', -1)) for p in payloads[:6]]}"
        )

    # 方案 I 优化：使用辅助方法记录 event
    self._record_event_safe(profile_gather_evt0, device)

    profile_cpu_stack_us: Optional[float] = None
    t_stack0_ns: Optional[int] = time.perf_counter_ns() if profile_detail else None

    use_cpp_stack = _SELECTOR_CPP_STACK_CACHED

    if capture_storage_stack is not None:
        capture_scores_stack = capture_storage_stack
    elif can_use_capture_base:
        batch_start = int(capture_base_batch_start)
        batch_end = int(batch_start) + int(batch_size)
        if layer_indices_expected:
            capture_scores_stack = capture_base[
                : len(payloads),
                batch_start:batch_end,
                :,
                :,
                :kv_len_total,
            ]
        else:
            if layer_indices is None:
                raise RuntimeError("layer_indices missing for non-contiguous capture base")
            layer_index_tensor = self._get_selector_layer_index_tensor(layer_indices, capture_base.device)
            capture_scores_stack = capture_base.index_select(0, layer_index_tensor)[
                :,
                batch_start:batch_end,
                :,
                :,
                :kv_len_total,
            ]
    else:
        if not capture_scores_all:
            capture_scores_all = [p.capture_scores for p in payloads]
        capture_scores_stack = self._ensure_selector_capture_scores_buffer(
            layers=len(payloads),
            batch=batch_size,
            num_heads=num_heads,
            window=window,
            kv_len=kv_len_total,
            device=device,
            dtype=first.capture_scores.dtype,
        )
        _ext_cap = _get_selector_batch_ext()
        capture_scores_stack = _stack_with_cpp_or_fail(
            tensors=capture_scores_all,
            out_buffer=capture_scores_stack,
            cpp_call=lambda: _ext_cap.stack_or_view_capture(
                capture_scores_all, None, None, batch_size, kv_len_total, capture_scores_stack,
            ) if _ext_cap is not None else None,
            use_cpp_stack=use_cpp_stack,
            label="capture",
        )
    log_f_denoms_stack: Optional[torch.Tensor] = None
    if denoms_ref is not None:
        if denoms_storage_stack is not None:
            log_f_denoms_stack = denoms_storage_stack
        elif can_use_denoms_base:
            denom_batch_start = int(denoms_base_batch_start)
            denom_batch_end = int(denom_batch_start) + int(batch_size)
            if layer_indices_expected:
                log_f_denoms_stack = denoms_base[
                    : len(payloads),
                    denom_batch_start:denom_batch_end,
                ]
            else:
                if layer_indices is None:
                    raise RuntimeError("layer_indices missing for non-contiguous denoms base")
                if "layer_index_tensor" in locals() and layer_index_tensor.device == denoms_base.device:
                    layer_index_tensor_denoms = layer_index_tensor
                else:
                    layer_index_tensor_denoms = self._get_selector_layer_index_tensor(layer_indices, denoms_base.device)
                log_f_denoms_stack = denoms_base.index_select(0, layer_index_tensor_denoms)[
                    :,
                    denom_batch_start:denom_batch_end,
                ]
        else:
            denoms_buf = self._ensure_selector_log_f_denoms_buffer(
                layers=len(payloads),
                batch=batch_size,
                num_heads=num_heads,
                device=device,
            )
            if log_f_denoms_all is None:
                raise RuntimeError("log_f_denoms list not prepared")
            _ext_den = _get_selector_batch_ext()
            log_f_denoms_stack = _stack_with_cpp_or_fail(
                tensors=log_f_denoms_all,
                out_buffer=denoms_buf,
                cpp_call=lambda: _ext_den.stack_or_view_denoms(
                    log_f_denoms_all, None, None, batch_size, denoms_buf,
                ) if _ext_den is not None else None,
                use_cpp_stack=use_cpp_stack,
                label="denoms",
            )

    if use_kv_lengths_direct_ref:
        kv_lengths_stack = kv_lengths_direct_ref.unsqueeze(0).expand(len(payloads), -1, -1)
    else:
        if kv_lengths_all is None:
            kv_lengths_all = [p.kv_lengths for p in payloads]
        kv_lengths_stack = self._ensure_selector_kv_lengths_buffer(
            layers=len(payloads),
            batch=batch_size,
            num_heads=num_heads,
            device=device,
        )
        kv_lengths_inputs = [
            k if (k.device == device and k.dtype == torch.long) else k.to(device=device, dtype=torch.long)
            for k in kv_lengths_all
        ]
        _ext_kv = _get_selector_batch_ext()
        kv_lengths_stack = _stack_with_cpp_or_fail(
            tensors=kv_lengths_inputs,
            out_buffer=kv_lengths_stack,
            cpp_call=lambda: _ext_kv.stack_or_expand_kv_lengths(kv_lengths_inputs, kv_lengths_stack),
            use_cpp_stack=use_cpp_stack,
            label="kv_lengths",
        )

    if profile_detail and t_stack0_ns is not None:
        profile_cpu_stack_us = (time.perf_counter_ns() - t_stack0_ns) / 1000.0

    # 方案 I 优化：使用辅助方法记录 event
    self._record_event_safe(profile_gather_evt1, device)

    # P1.4：topk “统一 slice” 快路径（避免 full-K 无效区间扫描）
    # - 仅当本 batch 的 slice_end（≈recent_start，且受 kv_len_total cap）一致时启用；
    # - 判定完全基于 CPU 的 seq_lens_cpu（无 DtoH 同步）。
    topk_slice_start = int(semantic_snapshot.sink_tokens) if semantic_snapshot is not None else 0
    topk_slice_end: Optional[int] = None
    if self.config is not None:
        blk = int(block_size_ref or 0)
        recent_cfg = int(semantic_snapshot.recent_tokens) if semantic_snapshot is not None else 0
        seq_lens_cpu = first.seq_lens_cpu
        if blk > 0 and recent_cfg > 0 and seq_lens_cpu is not None and len(seq_lens_cpu) >= batch_size:
            # 单调化简：candidate_end(seq) 对 seq=max(0,seq_lens_cpu[pos]) 单调非减
            # （clamp/affine recent/floor-to-blk/min-cap 均保序），故
            #   max_pos candidate_end(seq) == candidate_end(max_pos seq)
            # 循环内只做 1 次 int 比较的 max 归约，循环外对 max_seq 算一次 f。
            # 同源 seq_lens_cpu，不引入 tensor/DtoH；下游语义逐字保留。
            max_seq = 0
            for pos in range(batch_size):
                _s = max(0, int(seq_lens_cpu[pos]))
                if _s > max_seq:
                    max_seq = _s
            cap_recent = min(max_seq, recent_cfg)
            if max_seq > cap_recent:
                rs = ((max_seq - cap_recent) // blk) * blk
            else:
                rs = 0
            rs = max(0, int(rs))
            max_topk_slice_end = min(int(rs), int(kv_len_total))
            # 语义安全：取 batch 内最大 end，保证不会截断任何样本的有效 token 区间。
            # 相比 full kv_len_total，可在变长 batch 中减少 topk 扫描长度。
            if max_topk_slice_end > topk_slice_start:
                topk_slice_end = max_topk_slice_end
    # #13(c) K-bucket: round the topk slice width UP to 256 so the topk-scan
    # domain is shape-stable across refreshes (capture prerequisite). Gate is
    # coupled to fixed-shape topk upstream; clamp-fallback keeps the original
    # end when the bucketed end would overflow the buffer.
    if selector_kbucket_cached and topk_slice_end is not None:
        topk_slice_end = _bucket_topk_slice_end(
            topk_slice_start, topk_slice_end, int(kv_len_total)
        )

    t_select0_ns: Optional[int] = time.perf_counter_ns() if profile_detail else None
    # [DETERMINISTIC-SELECTOR-BOUNDS 2026-07-02] Same race family as the gather
    # [DETERMINISTIC-AUTOLEN] fix: `first.seq_lens_batch_i32` is a device VIEW of a
    # reused length buffer that refresh_capture_layout_live_lengths rewrites IN
    # PLACE (out=) as decode steps advance. The selector runs on refresh_stream at
    # a floating time relative to the decode main stream, so reading the live view
    # here made recent_start/row_lo/row_hi (the topk selection domain) -- and hence
    # selected_indices -- depend on WHEN the producer happened to execute
    # (run-to-run output drift in the bootstrap/refresh switchover window). The
    # selection plan must be computed against the submission-step snapshot, which
    # is exactly what seq_lens_cpu carries. One batch-sized H2D per selector run.
    seq_lens_full_for_bounds = None
    _det_seq_cpu = getattr(first, "seq_lens_cpu", None)
    if _det_seq_cpu is not None and len(_det_seq_cpu) >= int(batch_size):
        # cached_sequence_to_device: pinned staging + reuse_unchanged,替代裸
        # torch.tensor 的 pageable 同步 H2D(值=提交步快照 tuple,天然可作缓存键)。
        seq_lens_full_for_bounds = cached_sequence_to_device(
            tuple(max(0, int(_det_seq_cpu[i])) for i in range(int(batch_size))),
            device=device,
            dtype=torch.int32,
            cache_name="selector_bounds_seq_lens_i32",
            cache_owner=self,
            reuse_unchanged=True,
        )
    if (
        not isinstance(seq_lens_full_for_bounds, torch.Tensor)
        or seq_lens_full_for_bounds.device != device
        or seq_lens_full_for_bounds.dtype != torch.int32
        or int(seq_lens_full_for_bounds.numel()) < int(batch_size)
    ):
        seq_lens_full_for_bounds = first.seq_lens_batch

    (
        selected_indices_all,
        head_sink_all,
        recent_start_all,
        kv_len_head_all,
        allowed_lengths_all,
        selected_middle_pages_all,
        selected_middle_counts_all,
        selected_token_scores_all,
        detail_events,
    ) = self._compute_alpha_selection_batched_layers(
        capture_scores=capture_scores_stack,
        log_f_denoms=log_f_denoms_stack,
        kv_lengths=kv_lengths_stack,
        key_norms_full=key_norms_all,
        num_kv_heads=num_kv_heads,
        num_queries_per_kv=num_queries_per_kv,
        block_size=block_size_ref or 0,
        phase=phase,
        topk_slice_start=int(topk_slice_start),
        topk_slice_end=int(topk_slice_end) if topk_slice_end is not None else None,
        seq_lens_full=seq_lens_full_for_bounds,
        seq_lens_cpu=first.seq_lens_cpu,
        seq_lens_tensor_cpu=seq_lens_tensor,  # 方案 A 优化：复用预创建的 tensor
        profile_detail=bool(profile_detail),
        return_selected_token_scores=True,
    )
    if profile_detail and t_select0_ns is not None:
        profile_cpu_select_us = (time.perf_counter_ns() - t_select0_ns) / 1000.0

    # 可选：按 physical block-major 重排 selected_indices（不改变 token 集合，仅更换顺序）。
    # 该优化仅用于性能实验：可能带来轻微数值漂移（attention 归约顺序变化）。
    physical_sort = bool(_REBUILD_PHYSICAL_BLOCK_SORT_CACHED)
    if physical_sort and phase == "decode":
        try:
            if block_table_ref is not None and block_size_ref is not None and int(block_size_ref) > 0:
                # [DETERMINISTIC-SLOT-SOURCE 2026-07-03] row 恒从 row_list 构造
                # (payload 的 row_tensor(_i32) 是复用 buffer live 视图,错峰下
                # 被覆写;此处喂 block_table 行查表做重排键)。
                rt_i32 = None
                if row_list_ref is not None and len(row_list_ref) == batch_size:
                    rt_i32 = cached_sequence_to_device(
                        row_list_ref,
                        device=device,
                        dtype=torch.int32,
                        cache_name=f"selector_physical_row_i32{selector_layer_span_cache_suffix}",
                        cache_owner=self,
                        reuse_unchanged=True,
                    )
                if rt_i32 is not None:
                    reorder_order = self._compute_selected_indices_physical_block_major_order(
                        selected_indices=selected_indices_all,
                        block_table=block_table_ref,
                        row_tensor_i32=rt_i32,
                        block_size=int(block_size_ref),
                    )
                    if reorder_order is not None:
                        if (
                            selected_token_scores_all is not None
                            and selected_token_scores_all.shape != selected_indices_all.shape
                        ):
                            raise ValueError(
                                "selected_token_scores shape must match selected_indices for physical reorder"
                            )
                        selected_indices_all = selected_indices_all.gather(-1, reorder_order)
                        if selected_token_scores_all is not None:
                            selected_token_scores_all = selected_token_scores_all.gather(-1, reorder_order)
        except Exception:
            _log.warning("physical_block_major reorder failed", exc_info=True)
            raise
    _evts = _unpack_detail_events(detail_events)
    _none2 = (None, None)

    return SelectorResult(
        selected_indices=selected_indices_all,
        head_sink=head_sink_all,
        recent_start=recent_start_all,
        kv_len_head=kv_len_head_all,
        allowed_lengths=allowed_lengths_all,
        slot_list=slot_list_ref,
        selected_middle_pages=selected_middle_pages_all,
        selected_middle_counts=selected_middle_counts_all,
        selected_token_scores=selected_token_scores_all,
        profile_cpu_stack_us=profile_cpu_stack_us,
        profile_cpu_validate_us=profile_cpu_validate_us,
        profile_cpu_key_norms_us=profile_cpu_key_norms_us,
        profile_cpu_key_norms_arena_us=profile_cpu_key_norms_arena_us,
        profile_cpu_key_norms_direct_us=profile_cpu_key_norms_direct_us,
        profile_cpu_key_norms_direct_prepare_us=profile_cpu_key_norms_direct_prepare_us,
        profile_cpu_key_norms_direct_launch_us=profile_cpu_key_norms_direct_launch_us,
        profile_cpu_key_norms_pack_us=profile_cpu_key_norms_pack_us,
        profile_cpu_select_us=profile_cpu_select_us,
        profile_gather_evt0=profile_gather_evt0,
        profile_gather_evt1=profile_gather_evt1,
        profile_key_norms_preproc_evt0=profile_key_norms_preproc_evt0,
        profile_key_norms_preproc_evt1=profile_key_norms_preproc_evt1,
        profile_key_norms_evt0=profile_key_norms_evt0,
        profile_key_norms_evt1=profile_key_norms_evt1,
        profile_key_norms_h2d_evt0=profile_key_norms_h2d_evt0,
        profile_key_norms_h2d_evt1=profile_key_norms_h2d_evt1,
        profile_key_norms_delta_evt0=profile_key_norms_delta_evt0,
        profile_key_norms_delta_evt1=profile_key_norms_delta_evt1,
        profile_key_norms_pack_evt0=profile_key_norms_pack_evt0,
        profile_key_norms_pack_evt1=profile_key_norms_pack_evt1,
        profile_key_norms_delta_total_tokens=int(profile_key_norms_delta_total_tokens),
        profile_key_norms_delta_max_tokens=int(profile_key_norms_delta_max_tokens),
        profile_key_norms_delta_layers=int(profile_key_norms_delta_layers),
        profile_log_s_evt0=_evts.get("log_s", _none2)[0],
        profile_log_s_evt1=_evts.get("log_s", _none2)[1],
        profile_log_s_triton_evt0=_evts.get("log_s_triton", _none2)[0],
        profile_log_s_triton_evt1=_evts.get("log_s_triton", _none2)[1],
        profile_log_s_mask_evt0=_evts.get("log_s_mask", _none2)[0],
        profile_log_s_mask_evt1=_evts.get("log_s_mask", _none2)[1],
        profile_log_s_cross_evt0=_evts.get("log_s_cross", _none2)[0],
        profile_log_s_cross_evt1=_evts.get("log_s_cross", _none2)[1],
        profile_topk_evt0=_evts.get("topk", _none2)[0],
        profile_topk_evt1=_evts.get("topk", _none2)[1],
        profile_preproc_evt0=_evts.get("preproc", _none2)[0],
        profile_preproc_evt1=_evts.get("preproc", _none2)[1],
        profile_seq_full_evt0=_evts.get("seq_full", _none2)[0],
        profile_seq_full_evt1=_evts.get("seq_full", _none2)[1],
        profile_pure_preproc_evt0=_evts.get("pure_preproc", _none2)[0],
        profile_pure_preproc_evt1=_evts.get("pure_preproc", _none2)[1],
        profile_selector_bounds_evt0=_evts.get("bounds", _none2)[0],
        profile_selector_bounds_evt1=_evts.get("bounds", _none2)[1],
        profile_selector_pipeline_evt0=_evts.get("pipeline", _none2)[0],
        profile_selector_pipeline_evt1=_evts.get("pipeline", _none2)[1],
    )
