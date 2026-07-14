from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch

from patches.runtime_deps import require_runtime_dep
from patches.sparse_utils import _selector_fixed_k_enabled

_normalize_capture_layout_views = require_runtime_dep("_normalize_capture_layout_views")

def _refresh_payload_views_key(
    *,
    step_context: object,
    bound_meta: object,
    layout: "StepCaptureLayout",
    slot_list: Sequence[int],
) -> Tuple[object, ...]:
    row_key = getattr(layout, "slot_row_map_key", None)
    if row_key is None:
        row_list_cpu = getattr(layout, "row_list_cpu", None)
        row_key = tuple(int(row) for row in row_list_cpu) if row_list_cpu is not None else tuple()
    return (
        int(getattr(step_context, "epoch", -1)),
        int(getattr(step_context, "step_handle_id", -1)),
        int(getattr(step_context, "step_handle_generation", -1)),
        tuple(int(slot) for slot in slot_list),
        tuple(int(row) for row in row_key),
        tuple(getattr(bound_meta, "bound_meta_signature", tuple())),
        id(bound_meta),
    )


def _refresh_payload_views_minimal_ready(layout: "StepCaptureLayout") -> bool:
    return (
        isinstance(getattr(layout, "kv_lengths", None), torch.Tensor)
        and isinstance(getattr(layout, "seq_lens_batch", None), torch.Tensor)
        and isinstance(getattr(layout, "seq_lens_batch_i32", None), torch.Tensor)
        and isinstance(getattr(layout, "kv_len_per_row_i32", None), torch.Tensor)
        and isinstance(getattr(layout, "slot_tensor_i32", None), torch.Tensor)
        and isinstance(getattr(layout, "row_tensor_i32", None), torch.Tensor)
        and isinstance(getattr(layout, "slot_tensor_cpu", None), torch.Tensor)
        and isinstance(getattr(layout, "seq_lens_tensor_cpu", None), torch.Tensor)
        and getattr(layout, "row_list_cpu", None) is not None
        and getattr(layout, "seq_lens_cpu", None) is not None
    )


def _refresh_payload_views_ready(
    *,
    layout: "StepCaptureLayout",
    slot_list: Sequence[int],
    views_key: Tuple[object, ...],
) -> bool:
    live_lengths_key = getattr(layout, "live_lengths_key", None)
    if live_lengths_key is None:
        return False
    layout_slot_list = getattr(layout, "slot_list", tuple())
    slot_tuple = (
        tuple(int(v) for v in layout_slot_list)
        if layout_slot_list is not slot_list
        else tuple(int(v) for v in layout_slot_list)
    )
    requested_slot_tuple = (
        slot_tuple if layout_slot_list is slot_list else tuple(int(v) for v in slot_list)
    )
    if slot_tuple != requested_slot_tuple:
        return False
    if not _refresh_payload_views_minimal_ready(layout):
        return False
    if getattr(layout, "refresh_payload_views_key", None) == views_key:
        return True
    if not isinstance(live_lengths_key, tuple) or len(live_lengths_key) < 5:
        return False
    live_identity = tuple(int(v) for v in live_lengths_key[:3])
    view_identity = tuple(int(v) for v in views_key[:3])
    live_row_key = tuple(int(v) for v in live_lengths_key[4])
    view_row_key = tuple(int(v) for v in views_key[4])
    return live_identity == view_identity and live_row_key == view_row_key


def _refresh_payload_views_from_layout(
    layout: "StepCaptureLayout",
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    List[int],
    Tuple[int, ...],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    return (
        layout.kv_lengths,
        layout.seq_lens_batch,
        layout.kv_len_per_row_i32,
        layout.row_list_cpu,
        layout.seq_lens_cpu,
        layout.slot_tensor_i32,
        layout.slot_tensor_cpu,
        layout.seq_lens_tensor_cpu,
    )


def _prefill_payload_layout_matches(
    *,
    layout: object,
    step_context: object,
    slot_list: Sequence[int],
    num_heads: int,
    device: torch.device,
    kv_needed: int,
    expected_buf_id: int,
    chunk_query_lengths: Optional[torch.Tensor],
) -> bool:
    """Validate a caller-resolved prefill layout before skipping lookup."""

    if layout is None:
        return False
    step_identity = (
        int(getattr(step_context, "epoch", -1)),
        int(getattr(step_context, "step_handle_id", -1)),
        int(getattr(step_context, "step_handle_generation", -1)),
    )
    if (
        int(getattr(layout, "epoch", -1)) != step_identity[0]
        or int(getattr(layout, "step_handle_id", -1)) != step_identity[1]
        or int(getattr(layout, "step_handle_generation", -1)) != step_identity[2]
        or int(getattr(layout, "buf_id", -1)) != int(expected_buf_id)
        or int(getattr(layout, "num_heads", -1)) != int(num_heads)
        or int(getattr(layout, "kv_max", -1)) < int(kv_needed)
        or tuple(int(v) for v in getattr(layout, "slot_list", tuple()))
        != tuple(int(v) for v in slot_list)
    ):
        return False
    live_lengths_key = getattr(layout, "live_lengths_key", None)
    if not isinstance(live_lengths_key, tuple) or len(live_lengths_key) < 5:
        return False
    if tuple(int(v) for v in live_lengths_key[:3]) != step_identity:
        return False
    row_key = getattr(layout, "slot_row_map_key", None)
    if row_key is None or tuple(int(v) for v in live_lengths_key[4]) != tuple(
        int(v) for v in row_key
    ):
        return False
    required_tensors = (
        getattr(layout, "capture_scores", None),
        getattr(layout, "log_f_denoms", None),
        getattr(layout, "slot_tensor", None),
        getattr(layout, "slot_tensor_i32", None),
        getattr(layout, "row_tensor", None),
        getattr(layout, "row_tensor_i32", None),
        getattr(layout, "capture_row_by_batch_row_i32", None),
        getattr(layout, "seq_lens_batch", None),
        getattr(layout, "kv_lengths", None),
        getattr(layout, "kv_len_per_row_i32", None),
    )
    if any(
        not isinstance(tensor, torch.Tensor) or tensor.device != device
        for tensor in required_tensors
    ):
        return False
    capture_scores = required_tensors[0]
    if (
        capture_scores.dim() != 5
        or int(capture_scores.shape[1]) < len(slot_list)
        or int(capture_scores.shape[2]) != int(num_heads)
        or int(capture_scores.shape[-1]) < int(kv_needed)
    ):
        return False
    if chunk_query_lengths is not None and not isinstance(
        getattr(layout, "chunk_lengths", None), torch.Tensor
    ):
        return False
    return True


def prepare_prefill_capture_payload_impl(
    *,
    controller: "VLLMSparseController",
    cache_key: int,
    state: LayerState,
    step_context: StepContext,
    q: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqused_k: torch.Tensor,
    chunk_query_lengths: torch.Tensor,
    num_heads: int,
    device: torch.device,
    capture_plan: Optional[Dict[int, int]] = None,
    layout: Optional["StepCaptureLayout"] = None,
) -> Optional[
    Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        torch.Tensor,
        List[int],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        List[int],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Tuple[int, ...],
        Optional[torch.Tensor],
        int,
    ]
]:
    bound_meta = controller._require_step_bound_meta(
        step_context=step_context,
        stage="prefill payload",
    )
    plan_last_n_by_row = bound_meta.logits_last_n_by_row
    plan_caps_by_row = bound_meta.logits_capacity_by_row
    plan_max_last_n = max((int(v) for v in plan_last_n_by_row), default=0)
    plan_max_kv = max((int(v) for v in plan_caps_by_row), default=0)
    if capture_plan is None:
        return None
    slot_list = sorted(int(slot) for slot in capture_plan.keys())
    if not slot_list:
        return None
    if plan_max_last_n <= 0 or plan_max_kv <= 0:
        raise RuntimeError("prefill capture plan set but logits buffers not prepared")
    global_layer_index = controller.layer_index_by_cache_key.get(cache_key, -1)
    if global_layer_index < 0:
        return None
    _, expected_buf_id, slot_in_chunk = controller._map_global_layer_to_capture_slot(
        global_layer_index
    )
    if not _prefill_payload_layout_matches(
        layout=layout,
        step_context=step_context,
        slot_list=slot_list,
        num_heads=int(num_heads),
        device=device,
        kv_needed=int(plan_max_kv),
        expected_buf_id=int(expected_buf_id),
        chunk_query_lengths=chunk_query_lengths,
    ):
        layout = controller._get_step_capture_layout(
            phase="prefill",
            state=state,
            step_context=step_context,
            global_layer_index=global_layer_index,
            slot_list=slot_list,
            seqused_k=seqused_k,
            num_heads=num_heads,
            device=device,
            chunk_query_lengths=chunk_query_lengths,
            prepared_only=bool(
                getattr(controller, "_prefill_capture_meta_arena_enabled", False)
            ),
        )
    if layout is None:
        raise RuntimeError("prefill capture plan set but capture layout missing")
    kv_needed = int(plan_max_kv)
    # 性能关键：CUDA selector 的固定 K 让 graph/workspace 形态稳定，并使
    # request-major cohort tape 的 request-local layer view 保持 contiguous。
    # layout.kv_max 之外的 padding 只作为容量存在，row/token bounds 保证不可见。
    use_fixed_k = _selector_fixed_k_enabled()
    kv_slice = int(layout.kv_max) if use_fixed_k else int(kv_needed)
    capture_scores = layout.capture_scores[slot_in_chunk, : len(slot_list), :, :, :kv_slice]

    # prefill capture：last_n 可能 >1，此时 kernel 输出的是 (log_f_pre, denom_f)。
    # - 若 last_n==1：log_f == log_probs，无需 denom。
    # - 若 last_n>1：selector 需要拿到 denom 才能恢复 log_probs（用于 cross-head mutex 等语义）。
    #
    # 注意：batch 内混合 last_n==1 与 last_n>1 时，kernel 对 last_n==1 行不保证写 denom；
    # 必须显式把这些行的 denom 清零，避免复用 layout 时的 stale 值污染。
    log_f_denoms: Optional[torch.Tensor] = None
    # last_n==1：按约定必须传 None（selector 侧走 logits-only 的 log_softmax）
    has_gt1 = any(int(v) > 1 for v in capture_plan.values()) if capture_plan else False
    if has_gt1:
        log_f_denoms = layout.log_f_denoms[slot_in_chunk, : len(slot_list)]
    rows = layout.row_tensor
    if rows.numel() == 0:
        return None

    (
        kv_lengths_tensor,
        seq_lens_batch,
        kv_len_per_row_i32,
        row_list_cpu,
        seq_lens_cpu,
        slot_tensor_i32,
        self_slot_tensor_cpu,
        self_seq_lens_tensor_cpu,
    ) = _normalize_capture_layout_views(
        layout=layout,
        slot_list=slot_list,
        seqused_k=seqused_k,
        device=device,
        step_context=step_context,
        controller=controller,
        state=state,
    )
    seq_lens_batch_i32 = getattr(layout, "seq_lens_batch_i32", None)
    row_tensor_i32 = layout.row_tensor_i32
    if row_tensor_i32 is None:
        raise RuntimeError("prefill capture layout missing normalized row_tensor_i32")

    return (
        capture_scores,
        log_f_denoms,
        kv_lengths_tensor,
        seq_lens_batch,
        seq_lens_batch_i32,
        layout.chunk_lengths if layout.chunk_lengths is not None else torch.zeros((len(slot_list),), device=device, dtype=torch.long),
        layout.slot_list,
        layout.slot_tensor,
        slot_tensor_i32,
        self_slot_tensor_cpu,
        row_list_cpu if row_list_cpu is not None else [],
        layout.row_tensor,
        row_tensor_i32,
        kv_len_per_row_i32,
        seq_lens_cpu,
        self_seq_lens_tensor_cpu,
        int(slot_in_chunk),
    )

def prepare_refresh_capture_payload_impl(
    *,
    controller: "VLLMSparseController",
    cache_key: int,
    state: LayerState,
    step_context: StepContext,
    q: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqused_k: torch.Tensor,
    slots_filter: Optional[Sequence[int]] = None,
    slots_filter_sorted: bool = False,
    layout: Optional["StepCaptureLayout"] = None,
) -> Optional[
    Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        List[int],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        List[int],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Tuple[int, ...],
        Optional[torch.Tensor],
        int,
    ]
]:
    bound_meta = controller._require_step_bound_meta(
        step_context=step_context,
        stage="refresh payload",
    )
    # [PAYLOAD-VIEWS-FAST-IDENT 2026-07-09] 同 step 同 layout 的逐层重建短路
    # （取证 payload_build 537µs/世代主项=每层重建 views_key 三 tuple+ready 双
    # tuple+preamble max()）。首层全路径校验通过后置 ident（引用 `is` 比较，
    # 无 id 复用风险；epoch/handle 换步天然失效，slot_row_map_key/bound_meta/
    # slots_filter 对象替换即失效）。命中=直接产出与全路径 ready-hit 分支逐位
    # 相同的返回元组，值语义零变化。
    if layout is not None:
        _fast_pack = layout.refresh_payload_views_fast_ident
        if (
            _fast_pack is not None
            and _fast_pack[0] == int(step_context.epoch)
            and _fast_pack[1] == int(step_context.step_handle_id)
            and _fast_pack[2] == int(step_context.step_handle_generation)
            and _fast_pack[3] is bound_meta
            and _fast_pack[4] is layout.slot_row_map_key
            and _fast_pack[5] is slots_filter
            and _fast_pack[6] == bool(slots_filter_sorted)
        ):
            slot_list_fast = layout.slot_list
            kv_needed_fast = int(_fast_pack[7])
            if (
                int(getattr(state, "layer_index_epoch", -2))
                == int(getattr(controller, "_layer_index_cache_epoch", -1))
                and int(getattr(state, "capture_slot_in_chunk", -1)) >= 0
            ):
                slot_in_chunk = int(state.capture_slot_in_chunk)
            else:
                global_layer_index = controller.layer_index_by_cache_key.get(cache_key, -1)
                if global_layer_index < 0:
                    raise RuntimeError(
                        "refresh payload missing global layer index; "
                        f"cache_key={int(cache_key)} epoch={int(getattr(step_context, 'epoch', -1))}"
                    )
                _, _, slot_in_chunk = controller._map_global_layer_to_capture_slot(
                    global_layer_index
                )
            kv_slice_fast = (
                int(layout.kv_max) if _selector_fixed_k_enabled() else int(kv_needed_fast)
            )
            _scores_src = layout.capture_scores
            _sv_key = (int(slot_in_chunk), int(len(slot_list_fast)), int(kv_slice_fast))
            _sv = layout.refresh_scores_subviews.get(_sv_key)
            if _sv is not None and _sv[0] is _scores_src:
                capture_scores = _sv[1]
            else:
                capture_scores = _scores_src[
                    slot_in_chunk, : len(slot_list_fast), :, :, :kv_slice_fast
                ]
                layout.refresh_scores_subviews[_sv_key] = (_scores_src, capture_scores)
            (
                kv_lengths_tensor,
                seq_lens_batch,
                kv_len_per_row_i32,
                row_list_cpu,
                seq_lens_cpu,
                slot_tensor_i32,
                self_slot_tensor_cpu,
                self_seq_lens_tensor_cpu,
            ) = _refresh_payload_views_from_layout(layout)
            row_tensor_i32 = layout.row_tensor_i32
            if row_tensor_i32 is None:
                raise RuntimeError(
                    "refresh capture layout missing normalized row_tensor_i32"
                )
            return (
                capture_scores,
                None,
                kv_lengths_tensor,
                seq_lens_batch,
                getattr(layout, "seq_lens_batch_i32", None),
                layout.slot_list,
                layout.slot_tensor,
                slot_tensor_i32,
                self_slot_tensor_cpu,
                row_list_cpu if row_list_cpu is not None else [],
                layout.row_tensor,
                row_tensor_i32,
                kv_len_per_row_i32,
                seq_lens_cpu,
                self_seq_lens_tensor_cpu,
                int(slot_in_chunk),
            )
    plan_last_n_by_row = bound_meta.logits_last_n_by_row
    plan_caps_by_row = bound_meta.logits_capacity_by_row
    plan_max_last_n = max((int(v) for v in plan_last_n_by_row), default=0)
    plan_max_kv = max((int(v) for v in plan_caps_by_row), default=0)
    if plan_max_last_n <= 0 or plan_max_kv <= 0:
        raise RuntimeError(
            "refresh payload requires prepared logits buffers; "
            f"cache_key={int(cache_key)} epoch={int(getattr(step_context, 'epoch', -1))}"
        )
    step_authority = getattr(step_context, "step_authority", None)
    if step_authority is None or step_authority.epoch != step_context.epoch:
        raise RuntimeError(
            "refresh payload requires step_authority single-source slot list; "
            f"cache_key={int(cache_key)} epoch={int(getattr(step_context, 'epoch', -1))}"
        )
    authority_slot_list = step_authority.refresh_capture_slot_list
    if not isinstance(authority_slot_list, tuple):
        raise RuntimeError(
            "refresh payload requires tuple refresh_capture_slot_list from step_authority"
        )
    if any(slot < 0 for slot in authority_slot_list):
        raise RuntimeError(
            "refresh payload requires non-negative refresh_capture_slot_list"
        )
    if not authority_slot_list:
        raise RuntimeError(
            "refresh payload missing refresh_capture_slot_list from StepAuthority; "
            f"cache_key={int(cache_key)} epoch={int(getattr(step_context, 'epoch', -1))}"
        )
    slot_list: Sequence[int]
    if slots_filter is None:
        slot_list = authority_slot_list
    elif slots_filter_sorted and slots_filter is authority_slot_list:
        slot_list = authority_slot_list
    else:
        requested_slots = (
            [int(slot) for slot in slots_filter]
            if slots_filter_sorted
            else sorted(int(slot) for slot in slots_filter)
        )
        requested_slot_set = set(requested_slots)
        missing_slots = sorted(
            int(slot) for slot in requested_slot_set if int(slot) not in authority_slot_list
        )
        if missing_slots:
            raise RuntimeError(
                "refresh payload slots_filter must be a subset of "
                "step_authority.refresh_capture_slot_list"
            )
        slot_list = [
            int(slot) for slot in authority_slot_list if int(slot) in requested_slot_set
        ]
    device = q.device
    num_heads = q.shape[1]
    global_layer_index = controller.layer_index_by_cache_key.get(cache_key, -1)
    if global_layer_index < 0:
        raise RuntimeError(
            "refresh payload missing global layer index; "
            f"cache_key={int(cache_key)} epoch={int(getattr(step_context, 'epoch', -1))}"
        )
    if layout is not None:
        same_step_identity = (
            int(layout.epoch) == int(step_context.epoch)
            and int(getattr(layout, "step_handle_id", -1)) == int(step_context.step_handle_id)
            and int(getattr(layout, "step_handle_generation", -1))
            == int(step_context.step_handle_generation)
        )
        if same_step_identity and tuple(layout.slot_list) == tuple(slot_list):
            # 复用同一步同子集 layout，避免每层重复 list(...) 物化。
            slot_list = layout.slot_list
        else:
            layout = None
    else:
        slot_list = list(slot_list)
    if not slot_list:
        raise RuntimeError(
            "refresh payload slot_list is empty after normalization; "
            f"cache_key={int(cache_key)} epoch={int(getattr(step_context, 'epoch', -1))}"
        )
    if layout is None:
        layout = controller._get_step_capture_layout(
            phase="refresh",
            state=state,
            step_context=step_context,
            global_layer_index=global_layer_index,
            slot_list=slot_list,
            seqused_k=seqused_k,
            num_heads=num_heads,
            device=device,
            chunk_query_lengths=None,
        )
    if layout is None:
        raise RuntimeError(
            "refresh payload capture layout unavailable; "
            f"cache_key={int(cache_key)} epoch={int(getattr(step_context, 'epoch', -1))} "
            f"slot_count={int(len(slot_list))}"
        )
    _, _, slot_in_chunk = controller._map_global_layer_to_capture_slot(global_layer_index)
    kv_needed = int(plan_max_kv)
    use_fixed_k = _selector_fixed_k_enabled()
    kv_slice = int(layout.kv_max) if use_fixed_k else int(kv_needed)
    # [PAYLOAD-VIEWS-FAST-IDENT] 全路径切片同走子视图复用（src `is` 校验防替换）。
    _scores_src_full = layout.capture_scores
    _sv_key_full = (int(slot_in_chunk), int(len(slot_list)), int(kv_slice))
    _sv_full = layout.refresh_scores_subviews.get(_sv_key_full)
    if _sv_full is not None and _sv_full[0] is _scores_src_full:
        capture_scores = _sv_full[1]
    else:
        capture_scores = _scores_src_full[
            slot_in_chunk, : len(slot_list), :, :, :kv_slice
        ]
        layout.refresh_scores_subviews[_sv_key_full] = (_scores_src_full, capture_scores)

    # refresh(decode) 默认 last_n==1（见 prepare_step_logits_buffers）。
    # B1：kernel 仅写 logits（fp16/fp32），selector 内部完成 log_softmax；
    # 因此这里必须传 denom=None，避免任何额外广播减法/大张量物化。
    log_f_denoms: Optional[torch.Tensor] = None
    rows = layout.row_tensor
    if rows.numel() == 0:
        raise RuntimeError(
            "refresh payload row_tensor is empty; "
            f"cache_key={int(cache_key)} epoch={int(getattr(step_context, 'epoch', -1))} "
            f"slot_count={int(len(slot_list))}"
        )
    if not step_context.seq_lens:
        raise RuntimeError(
            "capture payload missing seq_lens in strict path; "
            f"(epoch={int(step_context.epoch)}, slots={len(slot_list)})"
        )

    refresh_views_key = _refresh_payload_views_key(
        step_context=step_context,
        bound_meta=bound_meta,
        layout=layout,
        slot_list=slot_list,
    )
    _views_ready_for_fast = False
    if _refresh_payload_views_ready(
        layout=layout,
        slot_list=slot_list,
        views_key=refresh_views_key,
    ):
        layout.refresh_payload_views_key = refresh_views_key
        _views_ready_for_fast = True
        (
            kv_lengths_tensor,
            seq_lens_batch,
            kv_len_per_row_i32,
            row_list_cpu,
            seq_lens_cpu,
            slot_tensor_i32,
            self_slot_tensor_cpu,
            self_seq_lens_tensor_cpu,
        ) = _refresh_payload_views_from_layout(layout)
    else:
        (
            kv_lengths_tensor,
            seq_lens_batch,
            kv_len_per_row_i32,
            row_list_cpu,
            seq_lens_cpu,
            slot_tensor_i32,
            self_slot_tensor_cpu,
            self_seq_lens_tensor_cpu,
        ) = _normalize_capture_layout_views(
            layout=layout,
            slot_list=slot_list,
            seqused_k=seqused_k,
            device=device,
            step_context=step_context,
            controller=controller,
            state=state,
        )
        if _refresh_payload_views_minimal_ready(layout):
            layout.refresh_payload_views_key = _refresh_payload_views_key(
                step_context=step_context,
                bound_meta=bound_meta,
                layout=layout,
                slot_list=slot_list,
            )
            _views_ready_for_fast = True
    if _views_ready_for_fast:
        # [PAYLOAD-VIEWS-FAST-IDENT] 首层全路径校验通过：置同 step 逐层短路
        # 身份，并清跨代 src 已替换的残留子视图（防旧 capture_scores 被条目
        # 强引用滞留）。
        _sv_map = layout.refresh_scores_subviews
        if _sv_map:
            for _sv_entry in _sv_map.values():
                if _sv_entry[0] is not _scores_src_full:
                    _sv_map.clear()
                    break
        layout.refresh_payload_views_fast_ident = (
            int(step_context.epoch),
            int(step_context.step_handle_id),
            int(step_context.step_handle_generation),
            bound_meta,
            layout.slot_row_map_key,
            slots_filter,
            bool(slots_filter_sorted),
            int(kv_needed),
        )
    seq_lens_batch_i32 = getattr(layout, "seq_lens_batch_i32", None)
    row_tensor_i32 = layout.row_tensor_i32
    if row_tensor_i32 is None:
        raise RuntimeError("refresh capture layout missing normalized row_tensor_i32")
    return (
        capture_scores,
        log_f_denoms,
        kv_lengths_tensor,
        seq_lens_batch,
        seq_lens_batch_i32,
        layout.slot_list,
        layout.slot_tensor,
        slot_tensor_i32,
        self_slot_tensor_cpu,
        row_list_cpu if row_list_cpu is not None else [],
        layout.row_tensor,
        row_tensor_i32,
        kv_len_per_row_i32,
        seq_lens_cpu,
        self_seq_lens_tensor_cpu,
        int(slot_in_chunk),
    )
