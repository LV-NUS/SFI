from __future__ import annotations

import json
import os
import logging
import time
import atexit
from dataclasses import dataclass
from typing import List, Optional

import torch

_log = logging.getLogger(__name__)

from patches.cpu_gpu_staging import cached_sequence_to_device
from patches.sparse_constants import _CAPTURE_CHUNK, _CAPTURE_KV_BUCKET_CACHED
from patches.refresh_runtime.capture_live_lengths import (
    get_step_plan_cap_tensor,
    refresh_capture_layout_live_lengths,
)
from patches.sparse_types import StepCaptureLayout
from patches.sparse_utils import _align_up_int, _is_stream_capturing_or_raise



def _capture_layout_profile_log_path() -> str:
    return os.getenv("VLLM_SPARSE_MB_PROFILE_LOG", "")


def _append_capture_layout_profile(payload: dict[str, object]) -> None:
    path = _capture_layout_profile_log_path()
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    except Exception:
        _log.debug("failed to append capture layout profile", exc_info=True)


# ---- VLLM_SPARSE_CAPTURE_LAYOUT_PROBE (#6 diagnostic, in-memory, atexit flush; env-OFF) ----
_CAPTURE_LAYOUT_PROBE_ENABLED = os.getenv("VLLM_SPARSE_CAPTURE_LAYOUT_PROBE", "0") == "1"
# [T6-LL-SKIP-LAND 2026-07-10] VLLM_SPARSE_DECODE_OUTPTR_SKIP_LIVE_LENGTHS 旋钮
# 删除,skip_live_lengths 形参转正默认生效(无旋钮原则)。byte-safe 定谳
# (SM100 旧档设计原文):decode out_ptr prebuild 只读 capture_row_by_batch_row
# +capture_scores 指针/stride,不读 live-lengths;同 step refresh 相首层调用
# (skip=False)在任何 live-lengths 读者之前按 live_lengths_key 重建。
# 必须 per-call 粒度,不能 layout 级(meta_pack/payload_worker 也读 live-lengths)。
_COLDBUF_PREDRAIN_PROBE = os.getenv("VLLM_SPARSE_COLDBUF_PREDRAIN_PROBE", "0") == "1"
_CAPTURE_LAYOUT_PROBE_ACC: dict = {}


def _capture_layout_probe_accumulate(payload: dict) -> None:
    phase = str(payload.get("phase", "?"))
    status = str(payload.get("status", "?"))
    buf = payload.get("buf_id", -1)
    key = (phase, status, int(buf) if isinstance(buf, int) else -1)
    rec = _CAPTURE_LAYOUT_PROBE_ACC.get(key)
    if rec is None:
        rec = {"n": 0, "total_us": 0.0, "phase_us": {}}
        _CAPTURE_LAYOUT_PROBE_ACC[key] = rec
    rec["n"] += 1
    try:
        rec["total_us"] += float(payload.get("total_us", 0.0))
    except Exception:
        pass
    pu = payload.get("phase_us") or {}
    if isinstance(pu, dict):
        for k, v in pu.items():
            try:
                rec["phase_us"][k] = rec["phase_us"].get(k, 0.0) + float(v)
            except Exception:
                pass


def _capture_layout_probe_flush() -> None:
    if not _CAPTURE_LAYOUT_PROBE_ACC:
        print("[CAPTURE_LAYOUT_PROBE] (empty)", flush=True)
        return
    out = ["[CAPTURE_LAYOUT_PROBE] per (phase,status,buf): mean_us over n calls"]
    for key in sorted(_CAPTURE_LAYOUT_PROBE_ACC.keys()):
        rec = _CAPTURE_LAYOUT_PROBE_ACC[key]
        n = max(1, int(rec["n"]))
        mean_total = float(rec["total_us"]) / n
        pu = rec["phase_us"]
        top = sorted(pu.items(), key=lambda kv: -kv[1])[:8]
        top_str = " ".join("%s=%.1f" % (k, float(v) / n) for k, v in top)
        out.append(
            "  phase=%s status=%s buf=%d n=%d mean_total_us=%.1f | %s"
            % (key[0], key[1], key[2], int(rec["n"]), mean_total, top_str)
        )
    print("\n".join(out), flush=True)


if _CAPTURE_LAYOUT_PROBE_ENABLED:
    atexit.register(_capture_layout_probe_flush)
# ---- end CAPTURE_LAYOUT_PROBE ----


def _prefill_slot_by_row_source(step_context: object) -> Optional[tuple[int, ...]]:
    """Return the immutable step-level slot map used by layout row resolution.

    LayerState is per-layer, so its fallback ``slot_batch_rows_cpu`` cannot prove
    that a layout is reusable across layers.  Memoization is therefore enabled
    only when the step publishes an immutable slot map.
    """

    step_envelope = getattr(step_context, "step_envelope_v2", None)
    step_authority = getattr(step_context, "step_authority", None)
    for owner in (step_envelope, step_authority):
        slot_by_row = getattr(owner, "slot_by_row", None)
        if isinstance(slot_by_row, tuple) and slot_by_row:
            return slot_by_row
    return None


def _prefill_step_layout_memo_keys(
    *,
    layout: StepCaptureLayout,
    step_context: object,
    chunk_id: int,
    buf_id: int,
    slot_list: List[int],
    num_heads: int,
    device: torch.device,
    kv_needed: int,
    bound_meta: object,
    slot_by_row: Optional[tuple[int, ...]],
    arena: object,
    arena_intent: object,
    prepared_only: bool,
    skip_live_lengths: bool,
) -> tuple[tuple[object, ...], tuple[object, ...]]:
    """Build scalar/value and strong-reference halves of the reuse proof."""

    value_key: tuple[object, ...] = (
        "prefill_prepared_v1",
        int(getattr(step_context, "epoch", -1)),
        int(getattr(step_context, "step_handle_id", -1)),
        int(getattr(step_context, "step_handle_generation", -1)),
        int(chunk_id),
        int(buf_id),
        tuple(int(v) for v in slot_list),
        int(num_heads),
        str(device),
        int(kv_needed),
        bool(prepared_only),
        bool(skip_live_lengths),
        id(layout),
        int(getattr(layout, "epoch", -1)),
        int(getattr(layout, "step_handle_id", -1)),
        int(getattr(layout, "step_handle_generation", -1)),
        int(getattr(layout, "kv_max", -1)),
        int(getattr(layout, "lease_generation", -1)),
        tuple(int(v) for v in getattr(layout, "slot_list", tuple())),
        tuple(
            int(v) for v in (getattr(layout, "row_list_cpu", None) or tuple())
        ),
    )
    reference_key = (
        bound_meta,
        getattr(bound_meta, "bound_meta_signature", None),
        getattr(bound_meta, "logits_last_n_by_row", None),
        getattr(bound_meta, "logits_capacity_by_row", None),
        getattr(bound_meta, "context_kv_len_by_row", None),
        getattr(bound_meta, "q_lens_by_row", None),
        slot_by_row,
        arena,
        arena_intent,
        getattr(layout, "slot_list", None),
        getattr(layout, "slot_to_capture_row", None),
        getattr(layout, "slot_row_map_key", None),
        getattr(layout, "live_lengths_key", None),
        getattr(layout, "slot_tensor", None),
        getattr(layout, "slot_tensor_i32", None),
        getattr(layout, "row_tensor", None),
        getattr(layout, "row_tensor_i32", None),
        getattr(layout, "capture_row_by_batch_row_i32", None),
        getattr(layout, "active_capture_row_by_batch_row_i32", None),
        getattr(layout, "seq_lens_batch", None),
        getattr(layout, "seq_lens_batch_i32", None),
        getattr(layout, "kv_lengths", None),
        getattr(layout, "kv_len_per_row_i32", None),
        getattr(layout, "chunk_lengths", None),
        getattr(layout, "capture_scores", None),
        getattr(layout, "log_f_denoms", None),
        getattr(layout, "row_list_cpu", None),
        getattr(layout, "seq_lens_cpu", None),
        getattr(layout, "kv_len_per_row_cpu", None),
    )
    return value_key, reference_key


@dataclass(frozen=True, slots=True)
class _PrefillStepLayoutMemo:
    """Fail-closed proof that a prepared prefill layout is layer-invariant."""

    value_key: tuple[object, ...]
    reference_key: tuple[object, ...]

    @classmethod
    def create(
        cls,
        *,
        layout: StepCaptureLayout,
        step_context: object,
        chunk_id: int,
        buf_id: int,
        slot_list: List[int],
        num_heads: int,
        device: torch.device,
        kv_needed: int,
        bound_meta: object,
        slot_by_row: tuple[int, ...],
        arena: object,
        arena_intent: object,
    ) -> "_PrefillStepLayoutMemo":
        value_key, reference_key = _prefill_step_layout_memo_keys(
            layout=layout,
            step_context=step_context,
            chunk_id=chunk_id,
            buf_id=buf_id,
            slot_list=slot_list,
            num_heads=num_heads,
            device=device,
            kv_needed=kv_needed,
            bound_meta=bound_meta,
            slot_by_row=slot_by_row,
            arena=arena,
            arena_intent=arena_intent,
            prepared_only=True,
            skip_live_lengths=False,
        )
        return cls(value_key=value_key, reference_key=reference_key)

    def matches(
        self,
        *,
        layout: StepCaptureLayout,
        step_context: object,
        chunk_id: int,
        buf_id: int,
        slot_list: List[int],
        num_heads: int,
        device: torch.device,
        kv_needed: int,
        bound_meta: object,
        slot_by_row: Optional[tuple[int, ...]],
        arena: object,
        arena_intent: object,
        prepared_only: bool,
        skip_live_lengths: bool,
    ) -> bool:
        if not prepared_only or skip_live_lengths or slot_by_row is None:
            return False
        value_key, reference_key = _prefill_step_layout_memo_keys(
            layout=layout,
            step_context=step_context,
            chunk_id=chunk_id,
            buf_id=buf_id,
            slot_list=slot_list,
            num_heads=num_heads,
            device=device,
            kv_needed=kv_needed,
            bound_meta=bound_meta,
            slot_by_row=slot_by_row,
            arena=arena,
            arena_intent=arena_intent,
            prepared_only=prepared_only,
            skip_live_lengths=skip_live_lengths,
        )
        return (
            self.value_key == value_key
            and len(self.reference_key) == len(reference_key)
            and all(
                expected is actual
                for expected, actual in zip(self.reference_key, reference_key)
            )
        )


def get_step_capture_layout_impl(
    self,
    *,
    phase: str,
    state: LayerState,
    step_context: StepContext,
    global_layer_index: int,
    slot_list: List[int],
    seqused_k: torch.Tensor,
    num_heads: int,
    device: torch.device,
    chunk_query_lengths: Optional[torch.Tensor] = None,
    prepared_only: bool = False,
    skip_live_lengths: bool = False,
) -> Optional[StepCaptureLayout]:
    profile_enabled = bool(_capture_layout_profile_log_path())
    probe_enabled = _CAPTURE_LAYOUT_PROBE_ENABLED
    _timing_on = profile_enabled or probe_enabled
    profile_start_ns = time.perf_counter_ns() if _timing_on else 0
    profile_phase_start_ns = profile_start_ns
    profile_phase_us: dict[str, float] = {}

    def _mark_phase(name: str) -> None:
        nonlocal profile_phase_start_ns
        if not _timing_on:
            return
        now_ns = time.perf_counter_ns()
        profile_phase_us[name] = float(now_ns - profile_phase_start_ns) / 1000.0
        profile_phase_start_ns = now_ns

    def _emit(status: str, **extra: object) -> None:
        if not _timing_on:
            return
        payload: dict[str, object] = {
            "event": "capture_layout_worker_detail",
            "status": str(status),
            "phase": str(phase),
            "epoch": int(getattr(step_context, "epoch", -1)),
            "global_layer_index": int(global_layer_index),
            "slot_count": int(len(slot_list)),
            "phase_us": dict(profile_phase_us),
            "total_us": float(time.perf_counter_ns() - profile_start_ns) / 1000.0,
        }
        payload.update(extra)
        if probe_enabled:
            _capture_layout_probe_accumulate(payload)
        if profile_enabled:
            _append_capture_layout_profile(payload)

    if not slot_list:
        _emit("empty_slot_list")
        return None
    if len(slot_list) != len(set(slot_list)):
        raise RuntimeError("capture layout slot_list contains duplicates")
    bound_meta = self._require_step_bound_meta(
        step_context=step_context,
        stage=f"capture layout[{phase}]",
    )
    plan_last_n_by_row = bound_meta.logits_last_n_by_row
    plan_cap_by_row = bound_meta.logits_capacity_by_row
    if (
        len(plan_last_n_by_row) < int(step_context.num_reqs)
        or len(plan_cap_by_row) < int(step_context.num_reqs)
    ):
        raise RuntimeError(
            "capture layout bound_meta row coverage mismatch; "
            f"plan_last_n={len(plan_last_n_by_row)} plan_cap={len(plan_cap_by_row)} "
            f"num_reqs={int(step_context.num_reqs)} phase={phase}"
        )
    _mark_phase("require_bound_meta")

    def _get_step_plan_cap_tensor() -> torch.Tensor:
        """StepBoundMeta logits capacity 的 step 级 GPU 缓存（单真源，不回写 StepContext）。"""
        return get_step_plan_cap_tensor(
            controller=self,
            bound_meta=bound_meta,
            plan_cap_by_row=plan_cap_by_row,
            device=device,
        )

    def _prepared_live_metadata_miss_reason(
        *,
        layout: StepCaptureLayout,
        row_key: tuple[int, ...],
        slot_list_for_rows: List[int],
    ) -> str:
        if getattr(layout, "live_lengths_key", None) is None:
            return "missing_live_lengths_key"
        if tuple(int(v) for v in (layout.row_list_cpu or [])) != row_key:
            return "row_key_mismatch"
        if tuple(int(v) for v in layout.slot_list) != tuple(int(v) for v in slot_list_for_rows):
            return "slot_list_mismatch"
        if getattr(layout, "slot_row_map_key", None) != row_key:
            return "slot_row_map_key_mismatch"
        row_count = int(len(row_key))
        if (
            not isinstance(layout.row_tensor, torch.Tensor)
            or not isinstance(layout.row_tensor_i32, torch.Tensor)
            or not isinstance(layout.capture_row_by_batch_row_i32, torch.Tensor)
            or not isinstance(layout.seq_lens_batch, torch.Tensor)
            or not isinstance(layout.kv_lengths, torch.Tensor)
            or not isinstance(layout.kv_len_per_row_i32, torch.Tensor)
            or int(layout.row_tensor.numel()) < row_count
            or int(layout.row_tensor_i32.numel()) < row_count
            or int(layout.seq_lens_batch.numel()) < row_count
            or int(layout.kv_len_per_row_i32.numel()) < row_count
            or layout.kv_lengths.dim() != 2
            or int(layout.kv_lengths.shape[0]) < row_count
            or int(layout.kv_lengths.shape[1]) < int(num_heads)
        ):
            return "tensor_coverage_mismatch"
        if chunk_query_lengths is not None and not isinstance(layout.chunk_lengths, torch.Tensor):
            return "missing_chunk_lengths"
        return ""

    def _prepared_live_metadata_ready(
        *,
        layout: StepCaptureLayout,
        row_key: tuple[int, ...],
        slot_list_for_rows: List[int],
    ) -> bool:
        return not _prepared_live_metadata_miss_reason(
            layout=layout,
            row_key=row_key,
            slot_list_for_rows=slot_list_for_rows,
        )

    def _prepared_live_metadata_profile_fields(
        *,
        layout: StepCaptureLayout,
        row_key: tuple[int, ...],
        slot_list_for_rows: List[int],
        miss_reason: str,
    ) -> dict[str, object]:
        slot_row_map_key = getattr(layout, "slot_row_map_key", None)
        return {
            "prepared_fast_miss_reason": str(miss_reason),
            "prepared_layout_epoch": int(getattr(layout, "epoch", -1)),
            "prepared_layout_step_handle_id": int(getattr(layout, "step_handle_id", -1)),
            "prepared_layout_step_handle_generation": int(getattr(layout, "step_handle_generation", -1)),
            "prepared_layout_slot_list": tuple(int(v) for v in getattr(layout, "slot_list", []) or []),
            "requested_slot_list": tuple(int(v) for v in slot_list_for_rows),
            "prepared_layout_row_list_cpu": tuple(int(v) for v in (getattr(layout, "row_list_cpu", None) or tuple())),
            "requested_row_key": tuple(int(v) for v in row_key),
            "prepared_layout_slot_row_map_key": (
                tuple(int(v) for v in slot_row_map_key)
                if slot_row_map_key is not None
                else None
            ),
            "live_lengths_key_present": getattr(layout, "live_lengths_key", None) is not None,
            "has_row_tensor": isinstance(getattr(layout, "row_tensor", None), torch.Tensor),
            "has_row_tensor_i32": isinstance(getattr(layout, "row_tensor_i32", None), torch.Tensor),
            "has_capture_row_by_batch_row_i32": isinstance(
                getattr(layout, "capture_row_by_batch_row_i32", None),
                torch.Tensor,
            ),
            "has_seq_lens_batch": isinstance(getattr(layout, "seq_lens_batch", None), torch.Tensor),
            "has_kv_lengths": isinstance(getattr(layout, "kv_lengths", None), torch.Tensor),
            "has_kv_len_per_row_i32": isinstance(getattr(layout, "kv_len_per_row_i32", None), torch.Tensor),
            "has_chunk_lengths": isinstance(getattr(layout, "chunk_lengths", None), torch.Tensor),
        }

    # log_f capture：kernel 输出为 [Hq, 1, K]（window 固定为 1），避免跨层/跨 slot 累积 logits 窗口。
    # logits_last_n/capacity 的语义真源为 StepPlan；
    # capacity tensor 使用 controller 本地 step 级缓存，避免 StepContext 双源字段。
    window = 1
    slots_needed = int(len(slot_list))
    if slots_needed <= 0:
        _emit("bad_slots_needed", slots_needed=int(slots_needed))
        return None
    kv_needed = max((int(v) for v in plan_cap_by_row), default=0)
    # refresh/prefill 高频时，kv_len（因此 kv_needed）会随 step 增长而频繁变化。
    # 如果按 kv_needed 精确分配，会导致每次 refresh 都触发一次大 buffer 重新分配，开销巨大。
    # 这里对 K 维做按 256 对齐的超额分配，仅在跨过对齐边界时扩容。
    kv_bucket = int(_CAPTURE_KV_BUCKET_CACHED)
    if kv_bucket <= 0:
        kv_bucket = 256
    kv_bucket = max(256, int(kv_bucket))
    kv_max = _align_up_int(kv_needed, kv_bucket)
    if window <= 0 or kv_max <= 0:
        _mark_phase("capacity")
        _emit(
            "bad_kv_max",
            slots_needed=int(slots_needed),
            kv_needed=int(kv_needed),
            kv_max=int(kv_max),
            plan_cap_by_row=tuple(int(v) for v in plan_cap_by_row[: int(step_context.num_reqs)]),
            plan_last_n_by_row=tuple(int(v) for v in plan_last_n_by_row[: int(step_context.num_reqs)]),
        )
        return None
    _mark_phase("capacity")
    chunk_id, buf_id, _ = self._map_global_layer_to_capture_slot(global_layer_index)
    ring = self.step_prefill_capture_layout_ring if phase == "prefill" else self.step_refresh_capture_layout_ring
    layout = ring[buf_id]
    self._reclaim_retired_buffers()
    _mark_phase("reclaim_retired")
    # [LAYOUT-STEP-MEMO 2026-07-06] refresh 布局在同 step 同 chunk 内逐层重建
    # 是纯冗余（rows/cap tensor/live lengths/cpu tensors/lease 全同值重做，
    # 实测 X 账单单点 ~100-250µs/层）：chunk 首层走完整 reuse_same_step 路径
    # 后置 memo，同 chunk 2..N 层直接返回。
    layout_step_memo_token = None
    if phase == "refresh" and layout is not None:
        layout_step_memo_token = (
            int(step_context.epoch),
            int(getattr(step_context, "step_handle_id", -1)),
            int(getattr(step_context, "step_handle_generation", -1)),
            int(chunk_id),
        )
        if layout.step_memo_token == layout_step_memo_token:
            _mark_phase("reuse_step_memo")
            return layout
    arena = getattr(self, "prefill_capture_meta_arena", None)
    arena_intent = getattr(self, "_prefill_capture_arena_intent", None)
    prefill_slot_by_row = _prefill_slot_by_row_source(step_context)
    if phase == "prefill" and layout is not None:
        prefill_memo = getattr(layout, "prefill_step_layout_memo", None)
        if isinstance(prefill_memo, _PrefillStepLayoutMemo) and prefill_memo.matches(
            layout=layout,
            step_context=step_context,
            chunk_id=int(chunk_id),
            buf_id=int(buf_id),
            slot_list=slot_list,
            num_heads=int(num_heads),
            device=device,
            kv_needed=int(kv_needed),
            bound_meta=bound_meta,
            slot_by_row=prefill_slot_by_row,
            arena=arena,
            arena_intent=arena_intent,
            prepared_only=bool(prepared_only),
            skip_live_lengths=bool(skip_live_lengths),
        ):
            _mark_phase("reuse_prefill_step_memo")
            _emit(
                "reuse_prefill_step_memo",
                buf_id=int(buf_id),
                kv_needed=int(kv_needed),
                kv_max=int(kv_max),
                layout_kv_max=int(layout.kv_max),
                row_count=int(len(getattr(layout, "row_list_cpu", None) or tuple())),
                live_lengths_rebuilt=False,
                arena_bind_status="memo_hit",
            )
            return layout
        # Drop all strong references held by a stale proof before the fallback
        # path rebinds and validates the layout.
        layout.prefill_step_layout_memo = None
    prepared_bound = False
    if prepared_only:
        if arena is None:
            _emit(
                "sync_expansion_miss",
                buf_id=int(buf_id),
                kv_needed=int(kv_needed),
                kv_max=int(kv_max),
                arena_bind_status="sync_expansion_miss",
                arena_reservation_status="missing",
            )
            return None
        reservation = arena.lookup_reservation(
            step_epoch=int(getattr(step_context, "epoch", -1)),
            step_handle_id=int(getattr(step_context, "step_handle_id", -1)),
            step_handle_generation=int(
                getattr(step_context, "step_handle_generation", -1)
            ),
            intent=getattr(self, "_prefill_capture_arena_intent"),
        )
        prepared_layout = arena.bind_prepared_prefill_capture_layout(
            reservation=reservation,
            phase=phase,
            buf_id=int(buf_id),
            step_epoch=int(getattr(step_context, "epoch", -1)),
            step_handle_id=int(getattr(step_context, "step_handle_id", -1)),
            step_handle_generation=int(
                getattr(step_context, "step_handle_generation", -1)
            ),
            slot_list=tuple(int(v) for v in slot_list),
            num_heads=int(num_heads),
            kv_needed=int(kv_needed),
            device=device,
        )
        if prepared_layout is None:
            _emit(
                "sync_expansion_miss",
                buf_id=int(buf_id),
                kv_needed=int(kv_needed),
                kv_max=int(kv_max),
                arena_reservation_status=str(reservation.status.value),
                arena_bind_status="sync_expansion_miss",
            )
            return None
        layout = prepared_layout
        ring[int(buf_id)] = layout
        prepared_bound = True
        _mark_phase("prepared_bind")
        prepared_row_key = tuple(
            int(row) for row in (getattr(prepared_layout, "row_list_cpu", None) or tuple())
        )
        if (
            getattr(prepared_layout, "slot_row_map_key", None) is None
            and prepared_row_key
            and getattr(prepared_layout, "live_lengths_key", None) is not None
        ):
            prepared_layout.slot_row_map_key = prepared_row_key
        prepared_slot_list = [int(slot) for slot in slot_list]
        prepared_miss_reason = _prepared_live_metadata_miss_reason(
            layout=prepared_layout,
            row_key=prepared_row_key,
            slot_list_for_rows=prepared_slot_list,
        )
        if not prepared_miss_reason:
            lease = self._ensure_capture_ring_active_lease(
                buf_id=buf_id,
                epoch=int(step_context.epoch),
                min_capacity=int(prepared_layout.capture_scores.numel()),
            )
            prepared_layout.lease_generation = int(lease.generation)
            _mark_phase("prepared_fast_ready")
            if not skip_live_lengths and prefill_slot_by_row is not None:
                prepared_layout.prefill_step_layout_memo = (
                    _PrefillStepLayoutMemo.create(
                        layout=prepared_layout,
                        step_context=step_context,
                        chunk_id=int(chunk_id),
                        buf_id=int(buf_id),
                        slot_list=prepared_slot_list,
                        num_heads=int(num_heads),
                        device=device,
                        kv_needed=int(kv_needed),
                        bound_meta=bound_meta,
                        slot_by_row=prefill_slot_by_row,
                        arena=arena,
                        arena_intent=arena_intent,
                    )
                )
            _emit(
                "prepared_bind",
                buf_id=int(buf_id),
                kv_needed=int(kv_needed),
                kv_max=int(kv_max),
                layout_kv_max=int(prepared_layout.kv_max),
                row_count=int(len(prepared_row_key)),
                live_lengths_rebuilt=False,
                prepared_fast_ready=True,
                arena_bind_status="prepared_bind",
                arena_reservation_status=str(reservation.status.value),
            )
            return prepared_layout
        if profile_enabled:
            _emit(
                "prepared_fast_miss",
                buf_id=int(buf_id),
                kv_needed=int(kv_needed),
                kv_max=int(kv_max),
                layout_kv_max=int(prepared_layout.kv_max),
                row_count=int(len(prepared_row_key)),
                arena_bind_status="prepared_bind",
                arena_reservation_status=str(reservation.status.value),
                **_prepared_live_metadata_profile_fields(
                    layout=prepared_layout,
                    row_key=prepared_row_key,
                    slot_list_for_rows=prepared_slot_list,
                    miss_reason=prepared_miss_reason,
                ),
            )

    # 允许跨 step 复用：只要 slots/head/window 不变，且 slots/K 的 capacity 足够大。
    # 关键目标：避免 slot_list 频繁变化（多 request / 结尾 capture）的情况下反复分配大张量。
    can_reuse = (
        layout is not None
        and layout.num_heads == num_heads
        and layout.window == window
        and layout.kv_max >= kv_max
        and isinstance(layout.capture_scores, torch.Tensor)
        and layout.capture_scores.dim() == 5
        and int(layout.capture_scores.shape[1]) >= slots_needed
        and isinstance(layout.log_f_denoms, torch.Tensor)
        and layout.log_f_denoms.dim() == 3
        and int(layout.log_f_denoms.shape[1]) >= slots_needed
    )
    _mark_phase("reuse_eval")
    if can_reuse:
        # 同一 step 内必须保持 slot_list 不扩张；若仅收缩（new ⊆ old）允许继续执行。
        slot_list_for_rows = slot_list
        step_handle_id = int(getattr(step_context, "step_handle_id", -1))
        step_handle_generation = int(getattr(step_context, "step_handle_generation", -1))
        same_step_identity = (
            int(layout.epoch) == int(step_context.epoch)
            and int(getattr(layout, "step_handle_id", -1)) == step_handle_id
            and int(getattr(layout, "step_handle_generation", -1)) == step_handle_generation
        )
        if same_step_identity and layout.slot_list != slot_list:
            if not self._sorted_list_is_subset(layout.slot_list, slot_list):
                raise RuntimeError(
                    f"capture layout slot_list changed within step. phase={phase} buf_id={buf_id} "
                    f"prev={layout.slot_list} new={slot_list}"
                )

        if layout.slot_tensor is not None and layout.slot_tensor.numel() == 0:
            _emit(
                "reuse_empty_slot_tensor",
                buf_id=int(buf_id),
                kv_needed=int(kv_needed),
                kv_max=int(kv_max),
                layout_kv_max=int(layout.kv_max),
                layout_slot_count=int(len(layout.slot_list)),
                layout_slots_cap=int(layout.capture_scores.shape[1]),
            )
            return None

        row_list = self._slots_to_rows_for_step_context(
            step_context=step_context,
            state=state,
            slot_list=slot_list_for_rows,
        )
        if any(row < 0 for row in row_list):
            raise RuntimeError("capture layout missing batch row index for slots")
        row_limit = int(step_context.num_reqs)
        if any(row >= row_limit for row in row_list):
            raise RuntimeError("capture layout: row index out of range for step_context")
        row_key = tuple(int(r) for r in row_list)
        row_key_changed = layout.slot_row_map_key != row_key
        if row_key_changed and getattr(self, "_capture_rows_cache", None) is not None:
            # row 映射变化时，清空 ptr cache，避免复用旧的 capture_row 索引
            # [CAPTURE-ROWS-CLEAR-UAF-GUARD 2026-07-07] P2-9 高频臂(错峰下
            # row_key 常变):弃引用前三流守卫,防另一上下文在飞读者。
            _guard = getattr(
                self, "_uaf_guard_record_streams_before_discard", None
            )
            if callable(_guard):
                for _stale_t in self._capture_rows_cache.values():
                    _guard(_stale_t)
            self._capture_rows_cache.clear()
        _mark_phase("reuse_rows")

        cap_tensor = _get_step_plan_cap_tensor()
        _mark_phase("reuse_cap_tensor")
        # 同一 step 内：若 row 映射未变化，仅补齐 cached 派生张量
        if same_step_identity and not row_key_changed and layout.slot_list == slot_list_for_rows:
            if layout.row_tensor_i32 is None:
                layout.row_tensor_i32 = cached_sequence_to_device(
                    layout.row_list_cpu or row_list,
                    dtype=torch.int32,
                    device=device,
                    cache_name="row_i32",
                    stage_cache=layout.small_tensor_stage,
                    out=layout.row_tensor_i32,
                )
            if (
                layout.slot_tensor_i32 is None
                or layout.slot_tensor_i32.device != device
                or layout.slot_tensor_i32.dtype != torch.int32
                or layout.slot_tensor_i32.numel() < len(layout.slot_list)
            ):
                layout.slot_tensor_i32 = cached_sequence_to_device(
                    layout.slot_list,
                    dtype=torch.int32,
                    device=device,
                    cache_name="slot_i32",
                    stage_cache=layout.small_tensor_stage,
                    out=layout.slot_tensor_i32,
                )
            if layout.capture_row_by_batch_row_i32 is None:
                self._ensure_capture_row_by_batch_row(layout=layout, step_context=step_context, device=device)
            elif (
                layout.active_capture_row_by_batch_row_i32 is None
                or layout.active_capture_row_by_batch_row_i32.device != device
                or layout.active_capture_row_by_batch_row_i32.dtype != torch.int32
                or int(layout.active_capture_row_by_batch_row_i32.numel()) != int(step_context.num_reqs)
            ):
                layout.active_capture_row_by_batch_row_i32 = layout.capture_row_by_batch_row_i32[
                    : int(step_context.num_reqs)
                ]
            if not layout.slot_to_capture_row:
                layout.slot_to_capture_row = {slot: idx for idx, slot in enumerate(layout.slot_list)}
            # row_list_cpu 只描述 row 映射；seq/kv live lengths 优先由 StepBoundMeta CPU 真源刷新。
            if layout.row_list_cpu is None:
                layout.row_list_cpu = list(row_list)
            live_lengths_rebuilt = False
            if not skip_live_lengths and not (
                prepared_bound
                and _prepared_live_metadata_ready(
                    layout=layout,
                    row_key=row_key,
                    slot_list_for_rows=slot_list_for_rows,
                )
            ):
                live_lengths_rebuilt = refresh_capture_layout_live_lengths(
                    layout=layout,
                    row_tensor=layout.row_tensor,
                    row_list=layout.row_list_cpu,
                    seqused_k=seqused_k,
                    cap_tensor=cap_tensor,
                    step_context=step_context,
                    device=device,
                    num_heads=num_heads,
                    chunk_query_lengths=chunk_query_lengths,
                    plan_cap_by_row_cpu=plan_cap_by_row,
                    context_kv_len_by_row_cpu=bound_meta.context_kv_len_by_row,
                )
            _mark_phase("reuse_same_step_live_lengths")
            if layout.slot_row_map_key is None:
                layout.slot_row_map_key = row_key
            self._ensure_capture_layout_cpu_tensors(layout=layout, phase=phase)
            _mark_phase("reuse_same_step_cpu_tensors")
            lease = self._ensure_capture_ring_active_lease(
                buf_id=buf_id,
                epoch=int(step_context.epoch),
                min_capacity=int(layout.capture_scores.numel()),
            )
            layout.lease_generation = int(lease.generation)
            _mark_phase("reuse_same_step_lease")
            _emit(
                "prepared_bind" if prepared_bound else "reuse_same_step",
                buf_id=int(buf_id),
                kv_needed=int(kv_needed),
                kv_max=int(kv_max),
                layout_kv_max=int(layout.kv_max),
                row_count=int(len(row_list)),
                live_lengths_rebuilt=bool(live_lengths_rebuilt),
                arena_bind_status="prepared_bind" if prepared_bound else "",
            )
            # [LAYOUT-STEP-MEMO 2026-07-06] reuse_same_step 全路径完成=layout
            # 对本 (step, chunk) 自洽，置 memo 供同 chunk 后续层零成本复用。
            # 跨 step rebuild 路径不置位（epoch 变化使旧 token 天然失配）。
            # [T6-LL-SKIP-LAND 2026-07-10] skip_live_lengths 调用不置位:memo
            # 只能由全保真通道(live lengths 已按本 step 重建)建立,防 skip
            # 调用把陈旧 live lengths 定格给同 chunk 后续层(今日调用序不可达,
            # 结构性闩死)。
            if layout_step_memo_token is not None and not skip_live_lengths:
                layout.step_memo_token = layout_step_memo_token
            return layout

        # 跨 step 或 row 映射变化：更新动态元数据（row_tensor/kv_lengths/chunk_lengths）
        # [STEP-PREP-CHUNK-DONE-GATE] R6:组成变化覆写共享 layout 载体(slot/row
        # 张量 out= 原位覆写 + live-lengths 重建)前,前置本 buf 的 chunk_done 设备
        # 侧等待——上一世代 chunk 的 grouped producer 在 refresh_stream 上可能仍在
        # 读旧载体(录制→执行毫秒窗;P0-4 coverage 双源同窗)。复用 wait_decider
        # 现成机制:flags==0 早退 + (epoch,buf,chunk) 去重 + wait_event 纯设备侧,
        # 稳态零开销、每 (buf,step) 至多一次;后续 dispatch 同键调用被去重跳过。
        # [R6-CAPTURE-BOUNDARY-ASSERT] 组成变化分支在 graph capture 内构造不可达
        # (组成重建只发生在 step-prep eager 段);wait 帮手在 capture 态会静默
        # return(服务其他合法 capture 调用方),故此处若真在 capture 内走到,
        # 共享 layout 覆写将失去 chunk_done 门保护——可执行断言替代论证。
        if device.type == "cuda" and _is_stream_capturing_or_raise(
            stage="step_prep_composition_change_gate"
        ):
            raise RuntimeError(
                "step-prep composition-change branch entered during CUDA graph "
                "capture; chunk_done gate cannot protect the shared layout here "
                "(unreachable by construction — investigate the capture path)"
            )
        # [直调] 方法由 WaitDeciderMixin 恒定组合提供;软绑定缺失=R6 门静默
        # 不跑(录制→执行毫秒窗回归),组合错误必须炸。
        self._main_stream_wait_for_chunk_done(
            buf_id=int(buf_id),
            device=device,
            chunk_id=int(chunk_id),
            epoch=int(step_context.epoch),
        )
        layout.epoch = step_context.epoch
        layout.step_handle_id = step_handle_id
        layout.step_handle_generation = step_handle_generation
        if _COLDBUF_PREDRAIN_PROBE:
            torch.cuda.synchronize()
            _mark_phase("coldbuf_predrain_sync")
        # slot_list 可能变化，但 capture_scores/log_f_denoms 的 capacity 可复用；这里只更新轻量元数据。
        if layout.slot_list != slot_list_for_rows:
            layout.slot_list = list(slot_list_for_rows)
            layout.slot_tensor = cached_sequence_to_device(
                layout.slot_list,
                dtype=torch.long,
                device=device,
                cache_name="slot_i64",
                stage_cache=layout.small_tensor_stage,
                out=layout.slot_tensor,
            )
            layout.slot_tensor_i32 = cached_sequence_to_device(
                layout.slot_list,
                dtype=torch.int32,
                device=device,
                cache_name="slot_i32",
                stage_cache=layout.small_tensor_stage,
                out=layout.slot_tensor_i32,
            )
            layout.slot_tensor_cpu = None
            layout.slot_to_capture_row = {slot: idx for idx, slot in enumerate(layout.slot_list)}
        elif not layout.slot_to_capture_row:
            layout.slot_to_capture_row = {slot: idx for idx, slot in enumerate(layout.slot_list)}
        if (
            layout.slot_tensor_i32 is None
            or layout.slot_tensor_i32.device != device
            or layout.slot_tensor_i32.dtype != torch.int32
            or layout.slot_tensor_i32.numel() < len(layout.slot_list)
        ):
            layout.slot_tensor_i32 = cached_sequence_to_device(
                layout.slot_list,
                dtype=torch.int32,
                device=device,
                cache_name="slot_i32",
                stage_cache=layout.small_tensor_stage,
                out=layout.slot_tensor_i32,
            )

        # [T6-LAYOUT-GEN-MEMO 2026-07-10] 跨世代结构 memo(旧档 §10.23 设计,
        # 键=slot_list/req 序):row 结构与上次上传值等价(锚=row_list_cpu 逐值
        # 相等,该字段只在成功上传后写入)且载体有效时,row 双精度 H2D 重传与
        # capture_row 反向映射的 fill/scatter 重建均为同值重做 → 跳过;任一
        # 锚/载体不满足=全量重建(fail-close),不做部分复用。
        _row_structure_unchanged = (
            not row_key_changed
            and isinstance(layout.row_tensor, torch.Tensor)
            and layout.row_tensor.device == device
            and layout.row_tensor.dtype == torch.long
            and int(layout.row_tensor.numel()) == len(row_list)
            and isinstance(layout.row_tensor_i32, torch.Tensor)
            and layout.row_tensor_i32.device == device
            and layout.row_tensor_i32.dtype == torch.int32
            and int(layout.row_tensor_i32.numel()) == len(row_list)
            and layout.row_list_cpu == list(row_list)
        )
        if _row_structure_unchanged:
            row_tensor = layout.row_tensor
        else:
            row_tensor = cached_sequence_to_device(
                row_list,
                dtype=torch.long,
                device=device,
                cache_name="row_i64",
                stage_cache=layout.small_tensor_stage,
                out=layout.row_tensor,
            )
            layout.row_tensor = row_tensor
            layout.row_tensor_i32 = cached_sequence_to_device(
                row_list,
                dtype=torch.int32,
                device=device,
                cache_name="row_i32",
                stage_cache=layout.small_tensor_stage,
                out=layout.row_tensor_i32,
            )
            layout.row_list_cpu = list(row_list)
        layout.slot_row_map_key = row_key
        if not (
            _row_structure_unchanged
            and isinstance(layout.capture_row_by_batch_row_i32, torch.Tensor)
            and layout.capture_row_by_batch_row_i32.device == device
            and layout.capture_row_by_batch_row_i32.dtype == torch.int32
            and isinstance(layout.active_capture_row_by_batch_row_i32, torch.Tensor)
            and int(layout.active_capture_row_by_batch_row_i32.numel())
            == int(step_context.num_reqs)
        ):
            # 反向映射内容=f(row_tensor 内容, num_reqs);两者均未变时为同值重做。
            self._ensure_capture_row_by_batch_row(layout=layout, step_context=step_context, device=device)

        if not step_context.seq_lens:
            raise RuntimeError(
                "capture layout missing seq_lens in strict path; "
                f"(phase={phase}, epoch={int(step_context.epoch)}, "
                f"layer={int(global_layer_index)}, slots={len(row_list)})"
            )
        cap_tensor = _get_step_plan_cap_tensor()
        live_lengths_rebuilt = False
        if not skip_live_lengths and not (
            prepared_bound
            and _prepared_live_metadata_ready(
                layout=layout,
                row_key=row_key,
                slot_list_for_rows=slot_list_for_rows,
            )
        ):
            live_lengths_rebuilt = refresh_capture_layout_live_lengths(
                layout=layout,
                row_tensor=row_tensor,
                row_list=row_list,
                seqused_k=seqused_k,
                cap_tensor=cap_tensor,
                step_context=step_context,
                device=device,
                num_heads=num_heads,
                chunk_query_lengths=chunk_query_lengths,
                plan_cap_by_row_cpu=plan_cap_by_row,
                context_kv_len_by_row_cpu=bound_meta.context_kv_len_by_row,
            )
        _mark_phase("reuse_live_lengths")
        if chunk_query_lengths is None:
            layout.chunk_lengths = None
        self._ensure_capture_layout_cpu_tensors(layout=layout, phase=phase)
        _mark_phase("reuse_cpu_tensors")
        lease = self._ensure_capture_ring_active_lease(
            buf_id=buf_id,
            epoch=int(step_context.epoch),
            min_capacity=int(layout.capture_scores.numel()),
        )
        layout.lease_generation = int(lease.generation)
        _mark_phase("reuse_lease")
        _emit(
            "prepared_bind" if prepared_bound else "reuse_cross_step",
            buf_id=int(buf_id),
            kv_needed=int(kv_needed),
            kv_max=int(kv_max),
            layout_kv_max=int(layout.kv_max),
            row_count=int(len(row_list)),
            live_lengths_rebuilt=bool(live_lengths_rebuilt),
            arena_bind_status="prepared_bind" if prepared_bound else "",
        )
        return layout

    # 新建 layout（带 slots capacity）：避免 slot_list 频繁变化导致反复分配大张量。
    # slot capacity 取按 8 对齐的超额分配，减少 allocator 抖动；后续复用只更新 slot_list/row_tensor/kv_lengths。
    if prepared_only:
        arena = getattr(self, "prefill_capture_meta_arena", None)
        if arena is not None:
            arena.metrics.arena_prepare_miss_count += 1
            arena.metrics.arena_ready_before_tail = False
            arena.metrics.arena_bind_status = "sync_expansion_miss"
        _emit(
            "sync_expansion_miss",
            buf_id=int(buf_id),
            kv_needed=int(kv_needed),
            kv_max=int(kv_max),
            arena_bind_status="sync_expansion_miss",
        )
        return None
    if layout is not None:
        self._retire_capture_layout(buf_id=buf_id, layout=layout, device=device)
        _mark_phase("retire_old_layout")
    slots_cap = _align_up_int(slots_needed, 8)
    small_tensor_stage: dict[str, object] = {}
    slot_tensor = cached_sequence_to_device(
        slot_list,
        dtype=torch.long,
        device=device,
        cache_name="slot_i64",
        stage_cache=small_tensor_stage,
    )
    slot_tensor_i32 = cached_sequence_to_device(
        slot_list,
        dtype=torch.int32,
        device=device,
        cache_name="slot_i32",
        stage_cache=small_tensor_stage,
    )
    slot_to_capture_row = {slot: idx for idx, slot in enumerate(slot_list)}
    if slot_tensor.numel() == 0:
        _emit(
            "new_empty_slot_tensor",
            buf_id=int(buf_id),
            kv_needed=int(kv_needed),
            kv_max=int(kv_max),
            slots_cap=int(slots_cap),
        )
        return None
    _mark_phase("new_slot_tensors")
    row_list = self._slots_to_rows_for_step_context(
        step_context=step_context,
        state=state,
        slot_list=slot_list,
    )
    row_list_cpu = row_list
    if any(row < 0 for row in row_list):
        raise RuntimeError("capture layout missing batch row index for slots")
    row_limit = int(step_context.num_reqs)
    if any(row >= row_limit for row in row_list):
        raise RuntimeError("capture layout: row index out of range for step_context")
    row_tensor = cached_sequence_to_device(
        row_list,
        dtype=torch.long,
        device=device,
        cache_name="row_i64",
        stage_cache=small_tensor_stage,
    )
    row_tensor_i32 = cached_sequence_to_device(
        row_list,
        dtype=torch.int32,
        device=device,
        cache_name="row_i32",
        stage_cache=small_tensor_stage,
    )
    _mark_phase("new_rows")

    seq_lens_cpu = tuple(
        int(step_context.seq_lens[int(r)]) if (0 <= int(r) < len(step_context.seq_lens)) else 0
        for r in row_list_cpu
    )
    _mark_phase("new_live_lengths_cpu_values")
    context_kv_by_row = tuple(int(v) for v in bound_meta.context_kv_len_by_row)
    if not all(
        0 <= int(row) < len(context_kv_by_row) and 0 <= int(row) < len(plan_cap_by_row)
        for row in row_list_cpu
    ):
        raise RuntimeError(
            "capture layout bound_meta row coverage mismatch for live lengths; "
            f"context_kv={len(context_kv_by_row)} plan_cap={len(plan_cap_by_row)} "
            f"rows={tuple(int(r) for r in row_list_cpu)} phase={phase}"
        )
    context_kv_cpu = tuple(int(context_kv_by_row[int(row)]) for row in row_list_cpu)
    kv_len_per_row_cpu = tuple(
        max(
            1,
            min(
                int(context_kv_cpu[idx]),
                int(plan_cap_by_row[int(row)]),
            ),
        )
        for idx, row in enumerate(row_list_cpu)
    )
    seq_full = cached_sequence_to_device(
        context_kv_cpu,
        dtype=torch.long,
        device=device,
        cache_name="live_seq_full_i64",
        stage_cache=small_tensor_stage,
    )
    _mark_phase("new_live_lengths_seq_tensor")
    seq_full_i32 = cached_sequence_to_device(
        context_kv_cpu,
        dtype=torch.int32,
        device=device,
        cache_name="live_seq_full_i32",
        stage_cache=small_tensor_stage,
    )
    _mark_phase("new_live_lengths_seq_i32")
    kv_len = cached_sequence_to_device(
        kv_len_per_row_cpu,
        dtype=torch.long,
        device=device,
        cache_name="live_kv_len_i64",
        stage_cache=small_tensor_stage,
    )
    _mark_phase("new_live_lengths_kv_tensor")
    kv_len_per_row_i32 = cached_sequence_to_device(
        kv_len_per_row_cpu,
        dtype=torch.int32,
        device=device,
        cache_name="live_kv_len_i32",
        stage_cache=small_tensor_stage,
    )
    _mark_phase("new_live_lengths_kv_i32")
    kv_lengths = kv_len.unsqueeze(1).expand(-1, num_heads)
    _mark_phase("new_live_lengths")

    chunk_lengths = None
    if chunk_query_lengths is not None:
        q_lens_src = tuple(int(v) for v in getattr(step_context, "q_lens", tuple()))
        if q_lens_src and all(0 <= int(row) < len(q_lens_src) for row in row_list_cpu):
            chunk_lengths = cached_sequence_to_device(
                tuple(int(q_lens_src[int(row)]) for row in row_list_cpu),
                dtype=torch.long,
                device=device,
                cache_name="live_chunk_i64",
                stage_cache=small_tensor_stage,
            )
        else:
            chunk_lengths = chunk_query_lengths.index_select(
                0,
                row_tensor.to(chunk_query_lengths.device),
            ).to(device=device, dtype=torch.long)

    # 新建 layout
    # FA4 CuTe mixed-page 要求 capture_scores 为 fp32；旧 FA3/SM80 路径保留 fp16。
    # dtype 在 layout 分配时一次确定，避免 kernel 调用前做 cast/copy。
    # 备注：
    # - last_n==1：B1 logits-only：kernel 只写 fp16 logits，log_softmax 在 selector 内完成
    # - last_n>1：kernel 后处理输出 (log_f_pre, denom_f)；log_f_pre dtype 为 fp16
    # - window 固定为 1，因此相比旧 logits 窗口缓存（window=last_n）显存仍显著更小。
    # Unified fp16 capture: FA3/SM80 already store fp16; the FA4 CuTe store is
    # element_type()-generic and the selector consumes the capture dtype
    # generically, so FA4 also stores fp16. Raw logits are O(10), far inside
    # fp16 range; only top-k ranking precision matters. Halves the capture
    # buffer + store/load bandwidth and the arena footprint.
    capture_dtype = torch.float16
    # [DETERMINISTIC-CAPTURE-ZERO-INIT 2026-07-03] 必须 zeros 不能 empty:kernel
    # 的 rolling 写只覆盖有效 kv 域,而 kbucket 把 topk 扫描域向上取整到 256,
    # padding 区若为 cudaMalloc 垃圾页(共享卡上被其它进程污染,每 run 不同)会
    # 参与 topk 比较→选择漂移(指纹实证:refresh 相 capture 整缓冲 hash 在污染
    # 卡上每 run 一值,denoms 因 zeros 分配从未漂,与 empty/zeros 一一对应)。
    # 零初始化后 buffer 任意时刻内容=确定写序列的函数。仅非复用路径,低频。
    capture_scores = torch.zeros(
        (int(_CAPTURE_CHUNK), int(slots_cap), num_heads, window, kv_max),
        device=device,
        dtype=capture_dtype,
    )
    log_f_denoms = torch.zeros(
        (int(_CAPTURE_CHUNK), int(slots_cap), num_heads),
        device=device,
        dtype=torch.float32,
    )
    _mark_phase("new_big_tensors")

    capture_row_by_batch_row_cpu = [-1] * int(step_context.num_reqs)
    for capture_row, batch_row in enumerate(row_list_cpu):
        row_i = int(batch_row)
        if 0 <= row_i < len(capture_row_by_batch_row_cpu):
            capture_row_by_batch_row_cpu[row_i] = int(capture_row)
    capture_row_by_batch_row_i32 = cached_sequence_to_device(
        capture_row_by_batch_row_cpu,
        dtype=torch.int32,
        device=device,
        cache_name="capture_row_by_batch_i32",
        stage_cache=small_tensor_stage,
    )
    _mark_phase("new_capture_row")
    lease = self._ensure_capture_ring_active_lease(
        buf_id=buf_id,
        epoch=int(step_context.epoch),
        min_capacity=int(_CAPTURE_CHUNK * slots_cap * num_heads * window * kv_max),
    )
    _mark_phase("new_lease")

    layout = StepCaptureLayout(
        epoch=step_context.epoch,
        step_handle_id=int(getattr(step_context, "step_handle_id", -1)),
        step_handle_generation=int(getattr(step_context, "step_handle_generation", -1)),
        slot_list=slot_list,
        slot_tensor=slot_tensor,
        slot_tensor_i32=slot_tensor_i32,
        slot_to_capture_row=slot_to_capture_row,
        row_tensor=row_tensor,
        row_tensor_i32=row_tensor_i32,
        capture_row_by_batch_row_i32=capture_row_by_batch_row_i32,
        kv_lengths=kv_lengths,
        kv_len_per_row_i32=kv_len_per_row_i32,
        chunk_lengths=chunk_lengths,
        num_heads=num_heads,
        window=window,
        kv_max=kv_max,
        capture_scores=capture_scores,
        log_f_denoms=log_f_denoms,
        row_list_cpu=row_list_cpu,
        seq_lens_batch=seq_full,
        seq_lens_batch_i32=seq_full_i32,
        seq_lens_cpu=seq_lens_cpu,
        kv_len_per_row_cpu=kv_len_per_row_cpu,
        active_capture_row_by_batch_row_i32=capture_row_by_batch_row_i32[: int(step_context.num_reqs)],
        slot_row_map_key=tuple(int(r) for r in row_list_cpu),
        buf_id=buf_id,  # P0-2 FIX: 包含 buf_id 用于缓存区分
        lease_generation=int(lease.generation),
        small_tensor_stage=small_tensor_stage,
    )

    ring[buf_id] = layout
    self._ensure_capture_layout_cpu_tensors(layout=layout, phase=phase)
    _mark_phase("new_cpu_tensors")
    _emit(
        "new_layout",
        buf_id=int(buf_id),
        kv_needed=int(kv_needed),
        kv_max=int(kv_max),
        slots_cap=int(slots_cap),
        row_count=int(len(row_list)),
        capture_numel=int(capture_scores.numel()),
        capture_dtype=str(capture_scores.dtype),
    )
    return layout
