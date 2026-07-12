from __future__ import annotations

import atexit
from contextlib import nullcontext
import logging
import json
import os
import time
from typing import ContextManager, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.profiler import record_function

_log = logging.getLogger(__name__)

_MB_PROFILE_LOG_ENV = "VLLM_SPARSE_MB_PROFILE_LOG"
_METADATA_TIMING_LOG_ENV = "VLLM_SPARSE_METADATA_TIMING_LOG"
# Read once at import: the RRP debug-fields gate consults this on every metadata
# build. Cached per the _DYNAMIC_ENV / _<NAME>_CACHED steady-hot-path convention
# (see patches/sparse_constants.py); dynamic callers keep the live read below.
_FA3_ROUTE_TRACE_LOG_CACHED = os.environ.get("VLLM_SPARSE_FA3_ROUTE_TRACE_LOG", "")
# Steady-hot-path env-cache (project _DYNAMIC_ENV / _<NAME>_CACHED
# convention). Both log paths default OFF (); _mb_profile_log_path /
# _metadata_timing_log_path are reached from the per-decode-step builder
# body, so the os.environ read is hoisted to import time. Under
# _DYNAMIC_ENV (pytest / VLLM_SPARSE_DYNAMIC_ENV) the helpers still read
# live, so tests observe updated values.
_MB_PROFILE_LOG_CACHED = os.environ.get(_MB_PROFILE_LOG_ENV, "")
_METADATA_TIMING_LOG_CACHED = os.environ.get(_METADATA_TIMING_LOG_ENV, "")
# [T2-FORENSIC 2026-07-10] 世代 commit 步 host 分相取证目录(默认空=零税;
# cProfile 扭曲相位时长,取证发与判速发分离,同 flush_worker REFRESH_CPROFILE_DIR)。
_MB_CPROFILE_DIR_ENV = "VLLM_SPARSE_MB_CPROFILE_DIR"
_MB_CPROFILE_DIR_CACHED = os.environ.get(_MB_CPROFILE_DIR_ENV, "").strip()
# [U14-ZERO-ALLOC-2-8 2026-07-12] Z-RRP diagnostic probe gate 在每个 steady
# delta 步的 rrp_step_state 解析入口读一次(默认 OFF)。按既定 _DYNAMIC_ENV /
# _<NAME>_CACHED 约定 hoist 到 import 时;pytest / VLLM_SPARSE_DYNAMIC_ENV 下
# 消费点仍走活读。默认 OFF → 缓存布尔与活读逐位同,仅省每步一次 os.environ.get。
_Z_RRP_PROBE_CACHED = os.environ.get("VLLM_SPARSE_Z_RRP_PROBE") == "1"
_METADATA_TIMING_FLUSH_EVERY_ENV = "VLLM_SPARSE_METADATA_TIMING_FLUSH_EVERY"
_RRP_PREP_PROFILE_LOG_ENV = "VLLM_SPARSE_RRP_PREP_PROFILE_LOG"
_RRP_READY_EVENT_ATTR = "mixed_page_resolver_replay_ready_event"
_RRP_READY_EVENT_GENERATION_ATTR = "mixed_page_resolver_replay_ready_event_generation"
_RRP_READY_EVENT_STREAM_ATTR = "mixed_page_resolver_replay_ready_event_stream"

from patches.runtime_deps import require_runtime_dep
from patches.layer_state import _stable_slot_signature64
from patches.refresh_runtime.post_kernel_worker import (
    build_step_cache_invariants,
    publish_selected_scope_launch_ready_if_needed,
    refresh_selected_launch_view_for_current_step,
)
from patches.sparse_constants import (
    _CAPTURE_CHUNK,
    _ONE_SHOT_BLOCKED_DENSE_FALLBACK_CACHED,
    _CAPTURE_IN_FLIGHT,
    _CAPTURE_KV_BUCKET_CACHED,
    _DYNAMIC_ENV,
    _RRP_SAME_PAGE_SKIP_REVALIDATION_CACHED,
    _CLEAN_METADATA_CACHED,
    _PAGE_ADD_INCREMENTAL_CACHED,
    _SAME_PAGE_MINIMAL_UPDATE_CACHED,
    _SAME_PAGE_READY_EVENT_ONLY_CACHED,
    _SAME_PAGE_MINIMAL_ASSERT_CACHED,
    _FORCE_COMPACT_OFF_CACHED,
    _FORCE_DENSE_CACHED,
    _ROW_MODE_DENSE,
    _ROW_MODE_COMPACT,
    _VALIDATE_LAYER_SLOT_MAP_CACHED,
    _VALIDATE_META_CONTRACT_CACHED,
    should_skip_page_sparse_state,
)
from patches.sparse_types import BoundLayerMeta, StepBoundMeta
from patches.sparse_utils import (
    _align_up_int,
    _make_step_decode_cache_key,
    normalize_layer_effective_refresh_signature,
    ALL_FALSE_SIGNATURE,
)
from patches.fa_sparse_runtime.resolved_row_ptr_arena import (
    PlannedCompactRowLayout,
    ResolvedRowPtrDescriptorPayload,
    ResolvedRowPtrArena,
    ResolvedRowPtrReplayMetadataBinding,
    attach_resolved_row_ptr_replay_metadata,
    bind_resolved_row_ptr_replay_metadata,
    build_resolved_row_ptr_descriptor_payload,
    build_source_counter_fields,
    validate_block_table_row_pointer_source,
)
from patches.decode_runtime.rrp_row_table_manager import (
    RrpRowTableInputs,
    RrpRowTableManager,
    RrpUpdateKind,
    RrpUpdateResult,
    should_record_rrp_ready_event,
)
from patches.decode_runtime.launch_template import (
    LaunchTemplate,
    apply_launch_template_row_delta,
    compile_launch_template,
)
from patches.decode_runtime.row_policy import classify_one_shot_bootstrap_decode_guard
from patches.decode_runtime.thin_builder_state import (
    DecodeDeltaPacket,
    DecodeRuntimeMode,
    DecodeRuntimeState,
    DecodeStaticGuard,
    classify_decode_runtime_mode,
    predict_same_page_delta,
)

from patches.step_decode_pipeline import apply_reuse_ordered_plan_state
from triton_kernel.flash_attn_score_dump_fwd import pack_req_meta_decode_fast_layers
from patches.fa_sparse_runtime.materialize import derive_page_aligned_recent_window
from triton_kernel.flash_attn_score_dump_fwd import pack_req_meta_prefill_fast_layers

_build_layer_step_cache = require_runtime_dep("_build_layer_step_cache")


# Steady-decode identity row tuple cache. ``batch_size`` is constant across a
# steady CUDA-graph replay window, and tuple(range(n)) is an immutable value
# consumed only by value (==/!=/len/slice; never mutated or identity-checked).
# Memoizing it shares one equal immutable instance instead of rebuilding it on
# every steady hit; behaviour is byte-identical.
_IDENTITY_ROWS_CACHE: Dict[int, Tuple[int, ...]] = {}


def _identity_rows(batch_size: int) -> Tuple[int, ...]:
    n = int(batch_size)
    cached = _IDENTITY_ROWS_CACHE.get(n)
    if cached is None:
        cached = tuple(range(n))
        _IDENTITY_ROWS_CACHE[n] = cached
    return cached


def _torch_profiler_enabled() -> bool:
    profiler_enabled = getattr(torch.autograd, "_profiler_enabled", None)
    if not callable(profiler_enabled):
        return False
    try:
        return bool(profiler_enabled())
    except Exception:
        return False


def _mb_record_function(name: str) -> ContextManager[None]:
    if _torch_profiler_enabled():
        return record_function(name)
    return nullcontext()


def _mb_profile_log_path() -> str:
    return os.environ.get(_MB_PROFILE_LOG_ENV, "") if _DYNAMIC_ENV else _MB_PROFILE_LOG_CACHED


def _append_mb_profile(event: dict[str, object]) -> None:
    path = _mb_profile_log_path()
    if not path:
        return
    payload = dict(event)
    payload.setdefault("event", "metadata_builder_step")
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


def _mb_cprofile_dir() -> str:
    return (
        os.environ.get(_MB_CPROFILE_DIR_ENV, "").strip()
        if _DYNAMIC_ENV
        else _MB_CPROFILE_DIR_CACHED
    )


# [T2-FORENSIC 2026-07-10] 单 profiler 跨 commit 步累积;每次窗口关闭后 dump
# 覆盖同一文件(child 被 kill 也保留已积累样本)。目录空=永不构造。
_MB_COMMIT_CPROFILE_STATE: dict = {}


def _mb_commit_cprofile() -> object | None:
    cprofile_dir = _mb_cprofile_dir()
    if not cprofile_dir:
        return None
    prof = _MB_COMMIT_CPROFILE_STATE.get("prof")
    if prof is None:
        import cProfile

        prof = cProfile.Profile()
        _MB_COMMIT_CPROFILE_STATE["prof"] = prof
        _MB_COMMIT_CPROFILE_STATE["path"] = os.path.join(
            cprofile_dir,
            f"mb_commit_{os.getpid()}_{time.time_ns()}.pstats",
        )
    return prof


def _mb_commit_cprofile_dump() -> None:
    prof = _MB_COMMIT_CPROFILE_STATE.get("prof")
    path = _MB_COMMIT_CPROFILE_STATE.get("path")
    if prof is None or not path:
        return
    prof.dump_stats(str(path))


_metadata_timing_rows: List[dict[str, object]] = []
_metadata_timing_atexit_registered = False


def _metadata_timing_log_path() -> str:
    return os.environ.get(_METADATA_TIMING_LOG_ENV, "") if _DYNAMIC_ENV else _METADATA_TIMING_LOG_CACHED


def _flush_metadata_timing() -> None:
    path = _metadata_timing_log_path()
    if not path or not _metadata_timing_rows:
        return
    rows = list(_metadata_timing_rows)
    _metadata_timing_rows.clear()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def _append_metadata_timing(event: dict[str, object]) -> None:
    global _metadata_timing_atexit_registered
    if not _metadata_timing_log_path():
        return
    if not _metadata_timing_atexit_registered:
        atexit.register(_flush_metadata_timing)
        _metadata_timing_atexit_registered = True
    payload = dict(event)
    payload.setdefault("event", "metadata_builder_timing")
    _metadata_timing_rows.append(payload)
    try:
        flush_every = int(os.environ.get(_METADATA_TIMING_FLUSH_EVERY_ENV, "0") or "0")
    except ValueError:
        flush_every = 0
    if flush_every > 0 and len(_metadata_timing_rows) >= flush_every:
        _flush_metadata_timing()


def _rrp_prep_profile_log_path() -> str:
    return os.environ.get(_RRP_PREP_PROFILE_LOG_ENV, "")


def _append_rrp_prep_profile(event: dict[str, object]) -> None:
    path = _rrp_prep_profile_log_path()
    if not path:
        return
    payload = dict(event)
    payload.setdefault("event", "rrp_replay_metadata_bind")
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")




# [T4-STREAM-WRAPPER-MEMO 2026-07-10] torch.cuda.current_stream 每调都新构
# Stream wrapper(re_record 相位实测 19.9µs/步);wrapper 只是 (device,
# cuda_stream) 句柄的不可变包装,同句柄复用同对象语义等价。键=raw 句柄
# (torch._C._cuda_getCurrentRawStream,纯 C 调用):capture/replay 切流时
# 句柄变则自动 miss 走全构造;memo 持有 wrapper 引用,其句柄不会被 torch
# 流池回收复用,无同址重生歧义。decode 主线程单写者,无锁。
_CURRENT_STREAM_WRAPPER_MEMO: dict[tuple[int, int], object] = {}


def _current_stream_cached(device: torch.device) -> object:
    idx = device.index
    if idx is None:
        idx = torch.cuda.current_device()
    idx = int(idx)
    raw = int(torch._C._cuda_getCurrentRawStream(idx))
    key = (idx, raw)
    stream = _CURRENT_STREAM_WRAPPER_MEMO.get(key)
    if stream is None:
        stream = torch.cuda.current_stream(device)
        _CURRENT_STREAM_WRAPPER_MEMO[key] = stream
    return stream


def _cuda_stream_identity(stream: object | None) -> int:
    if stream is None:
        return -1
    raw = getattr(stream, "cuda_stream", None)
    if raw is not None:
        try:
            return int(raw)
        except (TypeError, ValueError):
            pass
    return int(id(stream))


def _resolved_row_ptr_ready_event_holders(
    attn_metadata: object,
) -> tuple[object, ...]:
    return (
        attn_metadata,
        getattr(attn_metadata, "mixed_page_resolver_replay_arena", None),
        getattr(attn_metadata, "mixed_page_resolver_source_arena", None),
    )


def _resolved_row_ptr_ready_event_state(
    attn_metadata: object,
    *,
    holders: tuple[object, ...] | None = None,
) -> tuple[int, object | None, int]:
    best_generation = -1
    best_event: object | None = None
    best_stream = -1
    if holders is None:
        holders = _resolved_row_ptr_ready_event_holders(attn_metadata)
    for holder in holders:
        if holder is None:
            continue
        event = getattr(holder, _RRP_READY_EVENT_ATTR, None)
        if event is None:
            continue
        try:
            generation = int(getattr(holder, _RRP_READY_EVENT_GENERATION_ATTR, -1))
            stream_raw = getattr(holder, _RRP_READY_EVENT_STREAM_ATTR, -1)
            stream = -1 if stream_raw is None else int(stream_raw)
        except Exception:
            continue
        if generation >= best_generation:
            best_generation = generation
            best_event = event
            best_stream = stream
    return best_generation, best_event, best_stream


def _attach_resolved_row_ptr_ready_event_state(
    attn_metadata: object,
    *,
    ready_event_generation: int,
    ready_event: object | None,
    ready_event_stream: int,
    holders: tuple[object, ...] | None = None,
) -> None:
    if ready_event is None or int(ready_event_generation) < 0:
        return
    if holders is None:
        holders = _resolved_row_ptr_ready_event_holders(attn_metadata)
    for holder in holders:
        if holder is None:
            continue
        setattr(holder, _RRP_READY_EVENT_ATTR, ready_event)
        setattr(holder, _RRP_READY_EVENT_GENERATION_ATTR, int(ready_event_generation))
        setattr(holder, _RRP_READY_EVENT_STREAM_ATTR, int(ready_event_stream))


def _record_resolved_row_ptr_replay_ready_event(
    attn_metadata: object,
    *,
    device: torch.device,
    holders: tuple[object, ...] | None = None,
    profile_phase_us: dict[str, float] | None = None,
) -> bool:
    if torch.device(device).type != "cuda":
        return False
    # [T4-FORENSIC 2026-07-10] re_record 44.6µs 内层拆分(调用方传 dict 才计时)。
    _rec_t0 = time.perf_counter_ns() if profile_phase_us is not None else 0

    def _mark_rec_phase(name: str) -> None:
        nonlocal _rec_t0
        if profile_phase_us is None:
            return
        now_ns = time.perf_counter_ns()
        profile_phase_us[name] = float(now_ns - _rec_t0) / 1000.0
        _rec_t0 = now_ns

    try:
        if holders is None:
            holders = _resolved_row_ptr_ready_event_holders(attn_metadata)
        _generation, event, _stream = _resolved_row_ptr_ready_event_state(
            attn_metadata, holders=holders
        )
        if event is None:
            event = torch.cuda.Event(blocking=False, enable_timing=False)
            setattr(attn_metadata, _RRP_READY_EVENT_ATTR, event)
        _mark_rec_phase("rec_state_scan")
        stream = _current_stream_cached(device)
        _mark_rec_phase("rec_current_stream")
        event.record(stream)
        _mark_rec_phase("rec_event_record")
        stream_identity = _cuda_stream_identity(stream)
        generation = (
            max(
                int(
                    getattr(
                        holder,
                        _RRP_READY_EVENT_GENERATION_ATTR,
                        -1,
                    )
                )
                for holder in holders
                if holder is not None
            )
            + 1
        )
        for holder in holders:
            if holder is None:
                continue
            setattr(holder, _RRP_READY_EVENT_ATTR, event)
            setattr(holder, _RRP_READY_EVENT_GENERATION_ATTR, generation)
            setattr(holder, _RRP_READY_EVENT_STREAM_ATTR, stream_identity)
        _mark_rec_phase("rec_identity_publish")
        return True
    except Exception as exc:
        raise RuntimeError(
            "resolved-row-ptr replay metadata could not record ready event"
        ) from exc


def _record_resolved_row_ptr_owner_update_ready_event(
    controller: object,
    *,
    attn_metadata: object,
    device: torch.device,
    update_kernel_count: int,
    profile_phase_us: dict[str, float] | None = None,
) -> bool:
    if int(update_kernel_count) <= 0:
        return False
    if torch.device(device).type != "cuda":
        return False
    # [T4-FORENSIC 2026-07-10] 仿 _mark_rrp_attach_phase:调用方传 dict 才计时,
    # 默认 None 零税。
    _re_phase_start_ns = (
        time.perf_counter_ns() if profile_phase_us is not None else 0
    )

    def _mark_re_phase(name: str) -> None:
        nonlocal _re_phase_start_ns
        if profile_phase_us is None:
            return
        now_ns = time.perf_counter_ns()
        profile_phase_us[name] = float(now_ns - _re_phase_start_ns) / 1000.0
        _re_phase_start_ns = now_ns

    binding = getattr(controller, "_resolved_row_ptr_replay_metadata_binding", None)
    if not isinstance(binding, ResolvedRowPtrReplayMetadataBinding):
        raise RuntimeError(
            "resolved-row-ptr owner update requires an active replay metadata binding"
        )
    arena_key = getattr(controller, "_resolved_row_ptr_arena_key", None)
    if arena_key is None:
        raise RuntimeError(
            "resolved-row-ptr owner update requires an active arena key"
        )
    _ready_event_holders = _resolved_row_ptr_ready_event_holders(attn_metadata)
    _mark_re_phase("re_guards")
    ready_event_recorded = _record_resolved_row_ptr_replay_ready_event(
        attn_metadata,
        device=device,
        holders=_ready_event_holders,
        profile_phase_us=profile_phase_us,
    )
    _mark_re_phase("re_record")
    (
        ready_event_generation,
        ready_event,
        ready_event_stream,
    ) = _resolved_row_ptr_ready_event_state(
        attn_metadata, holders=_ready_event_holders
    )
    setattr(
        controller,
        "_resolved_row_ptr_ready_event_recorded",
        bool(ready_event_recorded),
    )
    _mark_re_phase("re_state")
    if _same_page_ready_event_only_publish_enabled() and (
        _update_resolved_row_ptr_graph_binding_ready_event_only(
            controller,
            arena_key=tuple(arena_key),
            ready_event_generation=int(ready_event_generation),
            ready_event=ready_event,
            ready_event_stream=int(ready_event_stream),
        )
    ):
        _mark_re_phase("re_publish_fast")
        return bool(ready_event_recorded)
    if not _refresh_resolved_row_ptr_graph_binding_ready_state(
        controller,
        binding=binding,
        arena_key=tuple(arena_key),
        ready_event_generation=int(ready_event_generation),
        ready_event=ready_event,
        ready_event_stream=int(ready_event_stream),
    ):
        _publish_resolved_row_ptr_graph_binding_state(
            controller,
            binding=binding,
            arena_key=tuple(arena_key),
            ready_event_generation=int(ready_event_generation),
            ready_event=ready_event,
            ready_event_stream=int(ready_event_stream),
        )
    _mark_re_phase("re_publish_slow")
    return bool(ready_event_recorded)


def _publish_resolved_row_ptr_graph_binding_state(
    controller: object,
    *,
    binding: ResolvedRowPtrReplayMetadataBinding,
    arena_key: tuple[object, ...],
    ready_event_generation: int,
    ready_event: object | None = None,
    ready_event_stream: int = -1,
    metadata_count: int = 1,
) -> None:
    def _batch_row_modes() -> tuple[str, ...]:
        arena = binding.source_arena
        row_modes = tuple(
            str(value) for value in getattr(arena, "row_mode_by_row_head", ())
        )
        num_kv_heads = max(1, int(binding.num_kv_heads))
        batch_size = int(binding.descriptor.batch)
        out: list[str] = []
        for row in range(batch_size):
            start = row * num_kv_heads
            stop = start + num_kv_heads
            chunk = row_modes[start:stop]
            if not chunk:
                out.append("unset")
                continue
            first = str(chunk[0])
            out.append(
                first if all(str(value) == first for value in chunk) else "mixed"
            )
        return tuple(out)

    setattr(
        controller,
        "_resolved_row_ptr_graph_binding_state",
        {
            "route_family": "resolved_row_ptr",
            "arena_key": tuple(arena_key),
            "q_layout_key": str(binding.descriptor.q_layout_key),
            "pointer_signature": tuple(binding.pointer_signature),
            "metadata_count": int(metadata_count),
            "row_mode_by_row": _batch_row_modes(),
            "row_mode_distribution": dict(binding.row_mode_distribution),
            "row_source_distribution": dict(binding.row_source_distribution),
            "source_counter_schema_version": int(
                binding.source_counter_schema_version
            ),
            "expected_rows": int(binding.expected_rows),
            "num_kv_heads": int(binding.num_kv_heads),
            "source_counter_missing_fields": tuple(
                str(field)
                for field in binding.source_counter_missing_fields
                if str(field)
            ),
            "ready_event_generation": int(ready_event_generation),
            "ready_event": ready_event,
            "ready_event_stream": int(ready_event_stream),
        },
    )


def _refresh_resolved_row_ptr_graph_binding_ready_state(
    controller: object,
    *,
    binding: ResolvedRowPtrReplayMetadataBinding,
    arena_key: tuple[object, ...],
    ready_event_generation: int,
    ready_event: object | None = None,
    ready_event_stream: int = -1,
) -> bool:
    state = getattr(controller, "_resolved_row_ptr_graph_binding_state", None)
    if not isinstance(state, dict):
        return False
    if state.get("route_family") != "resolved_row_ptr":
        return False
    try:
        if tuple(state.get("arena_key", ())) != tuple(arena_key):
            return False
        descriptor = binding.descriptor
        if str(state.get("q_layout_key", "")) != str(descriptor.q_layout_key):
            return False
        if tuple(state.get("pointer_signature", ())) != tuple(binding.pointer_signature):
            return False
        if int(state.get("metadata_count", 0)) <= 0:
            return False
    except Exception:
        return False
    state["ready_event_generation"] = int(ready_event_generation)
    state["ready_event"] = ready_event
    state["ready_event_stream"] = int(ready_event_stream)
    return True


def _resolved_row_ptr_step_bind_identity(
    *,
    step_authority: object,
    live_batch_size: int,
) -> tuple[object, ...] | None:
    live_batch_size_i = int(live_batch_size)
    if live_batch_size_i <= 0:
        return None

    def _slice_tuple(name: str, count: int) -> tuple[object, ...] | None:
        values = getattr(step_authority, name, None)
        if values is None or isinstance(values, torch.Tensor):
            return None
        try:
            materialized = tuple(values)
        except Exception:
            return None
        if len(materialized) < int(count):
            return None
        return materialized[: int(count)]

    req_ids = _slice_tuple("req_ids", live_batch_size_i)
    q_lens = _slice_tuple("q_lens_by_row", live_batch_size_i)
    q_start = _slice_tuple("q_start_loc", live_batch_size_i + 1)
    row_modes = _slice_tuple("row_mode_by_row", live_batch_size_i)
    slot_by_row = _slice_tuple("slot_by_row", live_batch_size_i)
    if (
        req_ids is None
        or q_lens is None
        or q_start is None
        or row_modes is None
        or slot_by_row is None
    ):
        return None
    return (
        int(getattr(step_authority, "epoch", -1)),
        int(getattr(step_authority, "step_handle_id", -1)),
        int(getattr(step_authority, "step_handle_generation", -1)),
        tuple(str(value) for value in req_ids),
        tuple(int(value) for value in q_lens),
        tuple(int(value) for value in q_start),
        tuple(int(value) for value in row_modes),
        tuple(int(value) for value in slot_by_row),
    )


def _stamp_resolved_row_ptr_graph_binding_step(
    *,
    controller: object,
    step_authority: object,
    live_batch_size: int,
    effective_batch_size: int,
) -> None:
    state = getattr(controller, "_resolved_row_ptr_graph_binding_state", None)
    if not isinstance(state, dict):
        return
    step_fast_identity = getattr(step_authority, "step_fast_identity", None)
    if isinstance(step_fast_identity, (list, tuple)) and len(step_fast_identity) >= 6:
        try:
            state["bind_step_fast_identity"] = (
                int(step_fast_identity[0]),
                int(step_fast_identity[1]),
                int(step_fast_identity[2]),
                int(step_fast_identity[3]),
                int(step_fast_identity[4]),
                int(step_fast_identity[5]),
                int(live_batch_size),
                int(effective_batch_size),
            )
            state["bind_live_batch_size"] = int(live_batch_size)
            state["bind_effective_batch_size"] = int(effective_batch_size)
            return
        except Exception:
            pass
    try:
        state["bind_step_fast_identity"] = (
            int(getattr(step_authority, "epoch", -1)),
            int(getattr(step_authority, "step_handle_id", -1)),
            int(getattr(step_authority, "step_handle_generation", -1)),
            int(getattr(step_authority, "step_identity_token")),
            int(getattr(step_authority, "req_set_hash")),
            int(getattr(step_authority, "row_phase_hash")),
            int(live_batch_size),
            int(effective_batch_size),
        )
        state["bind_live_batch_size"] = int(live_batch_size)
        state["bind_effective_batch_size"] = int(effective_batch_size)
        return
    except Exception:
        pass
    bind_identity = _resolved_row_ptr_step_bind_identity(
        step_authority=step_authority,
        live_batch_size=int(live_batch_size),
    )
    if bind_identity is None:
        return
    # Each path writes exactly one identity key (fast getattr path -> bind_step_fast_identity;
    # slow path -> bind_step_identity). Benign: the canonical StepAuthority always populates
    # step_fast_identity as a 6-tuple (step_authority.py __post_init__), so the writer always
    # takes the fast path and this slow bind_step_identity branch is effectively dead in
    # production; a stale fast key from a reused state dict therefore cannot mix with a
    # slow-path write. Do NOT "fix" this by writing both keys: that changes the binding-state
    # dict contents and is NOT byte-safe.
    state["bind_step_identity"] = bind_identity
    state["bind_live_batch_size"] = int(live_batch_size)
    state["bind_effective_batch_size"] = int(effective_batch_size)


def _stamp_resolved_row_ptr_step_state(
    *,
    controller: object,
    step_authority: object,
    live_batch_size: int,
    effective_batch_size: int,
) -> None:
    step_fast_identity = getattr(step_authority, "step_fast_identity", None)
    if isinstance(step_fast_identity, (list, tuple)) and len(step_fast_identity) >= 6:
        try:
            identity = (
                int(step_fast_identity[0]),
                int(step_fast_identity[1]),
                int(step_fast_identity[2]),
                int(step_fast_identity[3]),
                int(step_fast_identity[4]),
                int(step_fast_identity[5]),
                int(live_batch_size),
                int(effective_batch_size),
            )
        except Exception:
            identity = None
    else:
        try:
            identity = (
                int(getattr(step_authority, "epoch", -1)),
                int(getattr(step_authority, "step_handle_id", -1)),
                int(getattr(step_authority, "step_handle_generation", -1)),
                int(getattr(step_authority, "step_identity_token")),
                int(getattr(step_authority, "req_set_hash")),
                int(getattr(step_authority, "row_phase_hash")),
                int(live_batch_size),
                int(effective_batch_size),
            )
        except Exception:
            identity = None
    if identity is not None:
        setattr(controller, "_resolved_row_ptr_step_fast_identity", identity)
    setattr(controller, "_resolved_row_ptr_step_live_batch_size", int(live_batch_size))
    setattr(
        controller,
        "_resolved_row_ptr_step_effective_batch_size",
        int(effective_batch_size),
    )


def _rrp_reserved_source_overlap_profile(
    *,
    worker_block_table_i32: torch.Tensor,
    reserved_manager_block_ids: Tuple[int, ...],
    compact_ready_by_batch_row: Tuple[bool, ...],
    row_effective_k_by_row: Tuple[int, ...],
    compact_valid_tokens_by_row: Tuple[int, ...],
    recent_first_page_by_row: Tuple[int, ...],
    canonical_row_index_by_batch_row: Tuple[int, ...] = (),
    batch_size: int,
    page_size: int,
) -> dict[str, object]:
    reserved_ids = {int(block_id) for block_id in reserved_manager_block_ids}
    if not reserved_ids:
        return {
            "rrp_reserved_source_overlap_count": 0,
            "rrp_reserved_source_overlap_row_count": 0,
            "rrp_reserved_source_overlap_rows": [],
        }

    canonical_row_by_batch = (
        tuple(int(v) for v in canonical_row_index_by_batch_row[: int(batch_size)])
        if canonical_row_index_by_batch_row
        else tuple(range(int(batch_size)))
    )
    source_row_count = max(
        (row for row in canonical_row_by_batch if row >= 0),
        default=0,
    ) + 1
    table_cpu = (
        worker_block_table_i32[:source_row_count]
        .detach()
        .to(device="cpu", dtype=torch.int32)
    )
    width = int(table_cpu.shape[1]) if table_cpu.dim() == 2 else 0
    rows: list[dict[str, object]] = []
    total_hits = 0
    page_size_i = max(1, int(page_size))
    for row in range(int(batch_size)):
        source_row = int(canonical_row_by_batch[row])
        if source_row < 0:
            continue
        effective_k = max(0, int(row_effective_k_by_row[row]))
        if bool(compact_ready_by_batch_row[row]):
            compact_tokens = max(0, int(compact_valid_tokens_by_row[row]))
            visible_tokens = max(0, effective_k - compact_tokens)
            start_page = max(0, int(recent_first_page_by_row[row]))
            source_kind = "recent"
        else:
            visible_tokens = effective_k
            start_page = 0
            source_kind = "native"
        page_count = (
            (visible_tokens + page_size_i - 1) // page_size_i
            if visible_tokens > 0
            else 0
        )
        if width <= 0 or page_count <= 0:
            continue
        start_page = min(start_page, width)
        page_count = min(page_count, max(0, width - start_page))
        if page_count <= 0:
            continue
        source_pages = [
            int(v)
            for v in table_cpu[
                source_row,
                start_page : start_page + page_count,
            ].tolist()
        ]
        hits = [page for page in source_pages if page in reserved_ids]
        if not hits:
            continue
        total_hits += len(hits)
        rows.append(
            {
                "row": int(row),
                "source_kind": source_kind,
                "logical_start_page": int(start_page),
                "logical_page_count": int(page_count),
                "reserved_hit_count": int(len(hits)),
                "reserved_hit_ids": sorted(set(hits))[:16],
            }
        )
    return {
        "rrp_reserved_source_overlap_count": int(total_hits),
        "rrp_reserved_source_overlap_row_count": int(len(rows)),
        "rrp_reserved_source_overlap_rows": rows,
    }


def _should_bind_resolved_row_ptr_replay_metadata_for_update(
    mode: object,
    rrp_update: object,
    metadata_ready: bool,
) -> bool:
    if not bool(metadata_ready):
        return True
    if mode in (DecodeRuntimeMode.STEADY_DELTA, DecodeRuntimeMode.PAGE_BOUNDARY_DELTA):
        return bool(getattr(rrp_update, "metadata_bind_required", True))
    return True


def _should_update_launch_template_gpu_for_decode_delta(
    mode: DecodeRuntimeMode,
) -> bool:
    del mode  # [LIFECYCLE-OFF-ONLY] 原 ON 下 STEADY_DELTA 不更 GPU 的分支已删。
    return True


def _should_build_compact_recent_launch_plan(
    mode: DecodeRuntimeMode,
    launch_template_ready: bool,
) -> bool:
    if not bool(launch_template_ready):
        return True
    return mode not in (
        DecodeRuntimeMode.STEADY_DELTA,
        DecodeRuntimeMode.PAGE_BOUNDARY_DELTA,
    )


def _steady_fast_path_terminal_rrp_miss(
    controller: object,
    *,
    mode: DecodeRuntimeMode,
    delta: object,
    batch_size: int,
) -> str | None:
    """#12 ultra fail-fast admission gate for the steady decode fast path.

    Read-only, host-only prediction of the two TERMINAL miss reasons of
    ``_try_update_same_page_resolved_row_ptr_step_state`` -- the reasons that
    are fully determined by (mode, env, manager state, delta) BEFORE any of
    the heavy per-step steady work (step_bound_meta refresh, launch-template
    GPU delta, cache-key publish) runs. Returns None to ADMIT the step (the
    rrp gate keeps sole authority over every other, non-terminal miss
    reason), or the byte-identical reason string the gate itself would
    record:

      * ``mode_not_steady:page_boundary_delta`` -- PAGE_BOUNDARY_DELTA
        without a live boundary writer to admit into: CLEAN_METADATA off, or
        (lifecycle-OFF) PAGE_ADD_INCREMENTAL off. With CLEAN_METADATA on the
        gate routes the step to the lifecycle-ON Leg-B writer or the
        lifecycle-OFF steady-arm page-add (the gate's
        ``f"mode_not_steady:{mode.value}"`` exit).
      * ``row_table_same_page_delta_update_miss`` -- STEADY_DELTA whose
        same-page predicate provably fails with no rescue available
        (VLLM_SPARSE_PAGE_ADD_INCREMENTAL off; the gate only consults the
        page-add rescue when the native lifecycle is off, which the
        lifecycle admit above already proved).

    Every condition mirrors the gate's own; any uncertainty (Leg-B,
    lifecycle path, missing manager, malformed delta, possible page-add
    rescue) ADMITS the step so behavior is unchanged.
    """
    if mode is DecodeRuntimeMode.PAGE_BOUNDARY_DELTA:
        if (
            os.environ.get("VLLM_SPARSE_CLEAN_METADATA", "1") == "1"
            if _DYNAMIC_ENV
            else _CLEAN_METADATA_CACHED
        ):
            # Leg-B live page-boundary writer may legitimately apply: admit.
            return None
        return f"mode_not_steady:{mode.value}"
    if mode is not DecodeRuntimeMode.STEADY_DELTA:
        # Defensive admit: other modes are filtered out before the steady
        # fast path runs; let the gate own their miss reasons.
        return None
    manager = getattr(controller, "_rrp_row_table_manager", None)
    if not isinstance(manager, RrpRowTableManager):
        # The gate misses with its own granular reason here: admit.
        return None
    batch_size_i = int(batch_size)
    if (
        not isinstance(delta, DecodeDeltaPacket)
        or int(delta.batch_size) != batch_size_i
    ):
        return None
    row_effective_k_by_row = tuple(
        int(v) for v in tuple(delta.row_effective_k_by_row)[:batch_size_i]
    )
    if len(row_effective_k_by_row) != batch_size_i:
        return None
    if manager.can_apply_same_page_delta(
        row_effective_k_by_row,
        recent_first_page_by_row=tuple(
            int(v) for v in tuple(delta.recent_first_page_by_row)[:batch_size_i]
        ),
    ):
        return None
    if (
        os.environ.get("VLLM_SPARSE_PAGE_ADD_INCREMENTAL", "1") == "1"
        if _DYNAMIC_ENV
        else _PAGE_ADD_INCREMENTAL_CACHED
    ):
        # The page-add incremental rescue may still turn this into a hit:
        # admit and let the gate decide.
        return None
    return "row_table_same_page_delta_update_miss"


def _should_pack_dynamic_req_meta_for_step(
    *,
    has_decode_row: bool,
    mode: DecodeRuntimeMode,
    rrp_replay_state_ready: bool,
    q_layout_changed: bool,
    logf_generation_changed: bool,
    req_meta_buffers_ready: bool = True,
    previous_pack_ready: bool = True,
    runtime_classification_current: bool = True,
    has_logf_rows: bool = False,
    has_refresh_rows: bool = False,
    refresh_layer_group_active: bool = False,
) -> bool:
    if not bool(has_decode_row):
        return False
    if bool(has_logf_rows) or bool(has_refresh_rows) or bool(refresh_layer_group_active):
        return True
    if mode not in (DecodeRuntimeMode.STEADY_DELTA, DecodeRuntimeMode.PAGE_BOUNDARY_DELTA):
        return True
    if not bool(runtime_classification_current):
        return True
    if not bool(rrp_replay_state_ready):
        return True
    if not bool(req_meta_buffers_ready) or not bool(previous_pack_ready):
        return True
    if bool(q_layout_changed) or bool(logf_generation_changed):
        return True
    return False


def _decode_logf_signature_for_step(
    *,
    step_authority: object,
    batch_size: int,
) -> tuple[int, ...]:
    mask_by_row = getattr(step_authority, "logf_mask_by_row", None)
    if mask_by_row is None or len(mask_by_row) < int(batch_size):
        return (-1,) * int(batch_size)
    return tuple(int(v) for v in mask_by_row[: int(batch_size)])


def _decode_pending_refresh_state_for_step(step_authority: object) -> str:
    explicit = getattr(step_authority, "pending_refresh_state", None)
    if explicit not in (None, "", "nil", "empty"):
        return str(explicit)
    if bool(getattr(step_authority, "has_refresh_row", False)):
        return "ready"
    if int(getattr(step_authority, "refresh_decode_count", 0)) > 0:
        return "ready"
    if int(getattr(step_authority, "refresh_prefill_count", 0)) > 0:
        return "ready"
    layer_refresh = getattr(step_authority, "layer_effective_refresh_by_row", tuple())
    if any(bool(v) for v in tuple(layer_refresh)[: int(getattr(step_authority, "batch_size", 0))]):
        return "ready"
    return "empty" if explicit in (None, "") else str(explicit)


def _decode_runtime_classification_is_current(
    controller: object,
    step_authority: object,
) -> bool:
    delta = getattr(controller, "_decode_runtime_delta", None)
    return bool(
        isinstance(delta, DecodeDeltaPacket)
        and int(delta.step_id) == int(step_authority.epoch)
    )


def _decode_row_dynamic_signature_for_step(
    *,
    step_authority: object,
    launch_plan: object | None,
    batch_size: int,
) -> tuple[tuple[int, int, int, int, int], ...]:
    batch_size_i = int(batch_size)

    def _int_tuple_or_none(value: object) -> tuple[int, ...] | None:
        if value is None or isinstance(value, torch.Tensor):
            return None
        if isinstance(value, tuple):
            raw = value
        elif isinstance(value, list):
            raw = tuple(value)
        else:
            raw = (value,)
        if len(raw) < batch_size_i:
            return None
        return tuple(int(raw[row]) for row in range(batch_size_i))

    slot_by_row = _int_tuple_or_none(getattr(step_authority, "slot_by_row", None))
    if slot_by_row is None:
        slot_by_row = tuple(range(batch_size_i))
    row_mode_by_row = _int_tuple_or_none(getattr(step_authority, "row_mode_by_row", None))
    use_compact_by_row = _int_tuple_or_none(
        getattr(step_authority, "use_compact_by_row", None)
    )
    if row_mode_by_row is None and use_compact_by_row is None:
        return tuple()
    if row_mode_by_row is None:
        row_mode_by_row = tuple(
            int(_ROW_MODE_COMPACT) if bool(use_compact_by_row[row]) else int(_ROW_MODE_DENSE)
            for row in range(batch_size_i)
        )
    if use_compact_by_row is None:
        use_compact_by_row = tuple(
            1 if int(row_mode_by_row[row]) == int(_ROW_MODE_COMPACT) else 0
            for row in range(batch_size_i)
        )
    compact_valid = _int_tuple_or_none(
        getattr(launch_plan, "compact_valid_tokens_cpu", None)
        if launch_plan is not None
        else None
    )
    if compact_valid is None:
        compact_valid = (0,) * batch_size_i
    compact_offset = _int_tuple_or_none(
        getattr(launch_plan, "compact_offset_tokens_cpu", None)
        if launch_plan is not None
        else None
    )
    if compact_offset is None:
        compact_offset = (0,) * batch_size_i
    return tuple(
        (
            int(slot_by_row[row]),
            int(row_mode_by_row[row]),
            int(bool(use_compact_by_row[row])),
            int(compact_valid[row]),
            int(compact_offset[row]),
        )
        for row in range(batch_size_i)
    )


def _decode_graph_slot_signature(batch_size: int) -> tuple[int, ...]:
    return tuple(range(int(batch_size)))


def _decode_graph_row_mode_class_signature(batch_size: int) -> tuple[str, ...]:
    return ("resolved_row_ptr_row_dynamic",) * int(batch_size)


def _decode_graph_compact_capacity_pages(
    *,
    controller: object | None,
    launch_plan: object | None,
) -> int:
    plan_capacity = getattr(launch_plan, "compact_capacity_pages", None)
    if plan_capacity is not None:
        return int(plan_capacity)
    layer_states = getattr(controller, "layer_states", None) if controller is not None else None
    if layer_states:
        for state in layer_states.values():
            residency = getattr(state, "compact_page_residency", None)
            lease = getattr(residency, "lease", None)
            if lease is None:
                continue
            capacity = int(getattr(lease, "compact_blocks_per_slot", 0) or 0)
            if capacity > 0:
                return capacity
    return 0


def _decode_runtime_mode_for_controller(controller: object) -> DecodeRuntimeMode:
    mode = getattr(controller, "_decode_runtime_mode", None)
    state = getattr(controller, "_decode_runtime_state", None)
    if mode is None and isinstance(state, DecodeRuntimeState):
        mode = state.last_mode
    if isinstance(mode, DecodeRuntimeMode):
        return mode
    return DecodeRuntimeMode.FULL_RECOMPILE


def _can_skip_layer_state_refresh_for_step(
    controller: object,
    *,
    step_authority: object,
    step_decode_cache_key: object,
) -> bool:
    mode = _decode_runtime_mode_for_controller(controller)
    if mode not in (DecodeRuntimeMode.STEADY_DELTA, DecodeRuntimeMode.PAGE_BOUNDARY_DELTA):
        return False
    if not _decode_runtime_classification_is_current(controller, step_authority):
        return False
    step_decode_data = getattr(controller, "step_decode_data", None)
    step_dispatch_plan = getattr(controller, "step_dispatch_plan", None)
    if step_decode_data is None or step_dispatch_plan is None:
        return False
    if (
        getattr(step_decode_data, "cache_key", None) == step_decode_cache_key
        and getattr(step_dispatch_plan, "cache_key", None) == step_decode_cache_key
    ):
        return True
    batch_size = int(getattr(step_authority, "batch_size", 0))
    if batch_size <= 0:
        return False
    if int(getattr(step_decode_data, "batch_size", -1)) != batch_size:
        return False
    if int(getattr(step_dispatch_plan, "batch_size", -1)) != batch_size:
        return False
    if getattr(step_dispatch_plan, "step_decode_data", step_decode_data) is not step_decode_data:
        return False
    layer_data = getattr(step_decode_data, "layer_data", None)
    ordered = getattr(step_dispatch_plan, "layer_data_list_ordered", None)
    if not layer_data or not ordered:
        return False
    return True


def _build_decode_static_guard_for_launch_template(
    *,
    controller: object,
    attn_metadata: object,
    step_authority: object,
    existing_plan: object,
    block_size: int,
    num_kv_heads: int,
) -> DecodeStaticGuard:
    batch_size = int(step_authority.batch_size)
    compact_capacity_pages = _decode_graph_compact_capacity_pages(
        controller=controller,
        launch_plan=existing_plan,
    )
    return DecodeStaticGuard(
        graph_key=str(getattr(attn_metadata, "graph_key", "default")),
        batch_size=batch_size,
        block_size=int(block_size),
        num_kv_heads=int(num_kv_heads),
        page_size=int(block_size),
        max_pages_per_row=int(getattr(existing_plan, "max_pages_per_row", 0)),
        slot_signature=_decode_graph_slot_signature(batch_size),
        resolver_kind="ResolvedRowPtr",
        row_mode_class_signature=_decode_graph_row_mode_class_signature(batch_size),
        compact_capacity_pages=int(compact_capacity_pages),
    )


def _build_decode_static_guard_from_launch_template(
    *,
    controller: object | None = None,
    attn_metadata: object,
    step_authority: object,
    launch_template: LaunchTemplate,
    block_size: int,
    num_kv_heads: int,
) -> DecodeStaticGuard:
    batch_size = int(step_authority.batch_size)
    plan = launch_template.plan
    compact_capacity_pages = _decode_graph_compact_capacity_pages(
        controller=controller,
        launch_plan=plan,
    )
    return DecodeStaticGuard(
        graph_key=str(getattr(attn_metadata, "graph_key", "default")),
        batch_size=batch_size,
        block_size=int(block_size),
        num_kv_heads=int(num_kv_heads),
        page_size=int(block_size),
        max_pages_per_row=int(getattr(plan, "max_pages_per_row", 0)),
        slot_signature=_decode_graph_slot_signature(batch_size),
        resolver_kind="ResolvedRowPtr",
        row_mode_class_signature=_decode_graph_row_mode_class_signature(batch_size),
        compact_capacity_pages=int(compact_capacity_pages),
    )


def _reset_decode_runtime_full_recompile(controller: object, reason: str) -> None:
    controller._decode_runtime_mode = DecodeRuntimeMode.FULL_RECOMPILE
    controller._decode_runtime_delta = None
    controller._decode_runtime_reason = str(reason)
    state = getattr(controller, "_decode_runtime_state", None)
    if isinstance(state, DecodeRuntimeState):
        state.active_guard = None
        state.last_delta = None
        state.last_mode = DecodeRuntimeMode.FULL_RECOMPILE
        state.last_reason = str(reason)
        state.last_applied_step_id = None
    # [LITE-P0 fail-close #2 2026-07-11] runtime 强制全量 = 批组成/形态可能
    # 已变(抢占/驱逐/容量),触发前稳态快照随之作废——不清则成陈旧别名源
    # (审计档 §D-2:同 bs 同 q_lens 的批重组下快照会通过签名比对误 admit)。
    if getattr(controller, "_lite_pre_trigger_steady_delta", None) is not None:
        controller._lite_pre_trigger_steady_delta = None


def _seed_decode_runtime_state_after_full_recompile(
    controller: object,
    *,
    attn_metadata: object,
    step_authority: object,
    launch_template: LaunchTemplate,
    block_size: int,
    num_kv_heads: int,
    reason: str,
) -> bool:
    plan = getattr(launch_template, "plan", None)
    if plan is None or not bool(getattr(plan, "valid", False)):
        return False
    state = getattr(controller, "_decode_runtime_state", None)
    if not isinstance(state, DecodeRuntimeState):
        state = DecodeRuntimeState()
        controller._decode_runtime_state = state
    try:
        guard = _build_decode_static_guard_for_launch_template(
            controller=controller,
            attn_metadata=attn_metadata,
            step_authority=step_authority,
            existing_plan=plan,
            block_size=int(block_size),
            num_kv_heads=int(num_kv_heads),
        )
        # [RESEED-TO-DELTA 转正 2026-07-10] 原 VLLM_SPARSE_REFRESH_RESEED_TO_DELTA
        # 旋钮(默认开)已跑全判据,循无旋钮纪律删除转无条件:先盖 CURRENT recent
        # 描述符(request_recent_epoch / recent first+count,与 template-delta 分支
        # 同手法),使 _collect_decode_delta_packet 的 epoch/block-size 守卫不抛 →
        # seed 落地 → 下一步重分类 STEADY/PAGE_BOUNDARY(无空 reason 的
        # FULL_RECOMPILE 级联)。下方共享 except=真实采集失败的 FAILED-seed 慢路
        # (维持 FULL_RECOMPILE 级联,功能正确),非旋钮臂,保留。
        _ensure_current_recent_descriptors_for_launch_template(
            step_bound_meta=getattr(controller, "step_bound_meta", None),
            step_authority=step_authority,
            page_size=int(getattr(plan, "page_size", 0) or int(block_size)),
        )
        delta = _collect_decode_delta_packet(
            step_authority=step_authority,
            step_bound_meta=getattr(controller, "step_bound_meta", None),
            launch_template=launch_template,
        )
    except Exception as exc:
        controller._decode_runtime_full_recompile_seed_reason = (
            f"failed:{type(exc).__name__}:{exc}"
        )
        return False

    state.active_guard = guard
    state.last_delta = delta
    state.last_mode = DecodeRuntimeMode.FULL_RECOMPILE
    state.last_reason = str(reason)
    state.last_applied_step_id = int(delta.step_id)
    controller._decode_runtime_mode = DecodeRuntimeMode.FULL_RECOMPILE
    controller._decode_runtime_delta = delta
    controller._decode_runtime_reason = str(reason)
    controller._decode_runtime_full_recompile_seed_reason = "seeded"
    controller._decode_runtime_full_recompile_seed_count = (
        int(getattr(controller, "_decode_runtime_full_recompile_seed_count", 0)) + 1
    )
    return True


def _try_get_decode_runtime_cached_kv_specs(
    controller: object,
    *,
    prefer_cache: bool,
) -> tuple[int, int, int, object] | None:
    if not bool(prefer_cache):
        return None
    cached = getattr(controller, "_decode_runtime_kv_specs", None)
    if not isinstance(cached, tuple) or len(cached) != 4:
        return None
    block_size, num_kv_heads, head_dim, kv_dtype = cached
    block_size_i = int(block_size)
    num_kv_heads_i = int(num_kv_heads)
    head_dim_i = int(head_dim)
    if block_size_i <= 0 or num_kv_heads_i <= 0 or head_dim_i <= 0:
        return None
    controller._decode_runtime_kv_spec_cache_hit_count = (
        int(getattr(controller, "_decode_runtime_kv_spec_cache_hit_count", 0)) + 1
    )
    return (block_size_i, num_kv_heads_i, head_dim_i, kv_dtype)


def _try_bind_decode_seq_lens_source_no_copy(
    *,
    attn_metadata: object,
    step_meta: object,
    batch_size: int,
) -> bool:
    batch_size_i = int(batch_size)
    if batch_size_i < 0:
        return False
    seq_lens = getattr(attn_metadata, "seq_lens", None)
    if not isinstance(seq_lens, torch.Tensor):
        return False
    if seq_lens.dtype != torch.int32 or seq_lens.dim() != 1:
        return False
    if int(seq_lens.numel()) < batch_size_i:
        return False
    seq_lens = seq_lens[:batch_size_i]
    if not seq_lens.is_contiguous():
        return False
    context_kv_len = getattr(step_meta, "context_kv_len", None)
    if context_kv_len is None or len(context_kv_len) < batch_size_i:
        return False
    step_meta.seqused_k_gpu = seq_lens
    step_meta.canonical_real_kv_len_cpu = tuple(
        int(v) for v in context_kv_len[:batch_size_i]
    )
    step_meta.canonical_real_kv_len_i32_gpu = seq_lens
    return True


def _ensure_current_recent_descriptors_for_launch_template(
    *,
    step_bound_meta: object,
    step_authority: object,
    page_size: int,
) -> None:
    batch_size = int(step_authority.batch_size)
    layer_effective_refresh_by_row = tuple(
        bool(v)
        for v in tuple(
            getattr(step_authority, "layer_effective_refresh_by_row", tuple())
        )[:batch_size]
    )
    if len(layer_effective_refresh_by_row) < batch_size:
        raise RuntimeError(
            "launch template delta requires layer_effective_refresh_by_row coverage; "
            f"batch_size={batch_size} coverage={len(layer_effective_refresh_by_row)}"
        )

    from patches.fa_sparse_runtime.runtime_cache import ensure_step_recent_descriptors

    ensure_step_recent_descriptors(
        step_meta=step_bound_meta,
        page_size=int(page_size),
        layer_effective_refresh_by_row=layer_effective_refresh_by_row,
    )




def _collect_decode_delta_packet(
    *,
    step_authority: object,
    step_bound_meta: object,
    launch_template: LaunchTemplate,
) -> DecodeDeltaPacket:
    batch_size = int(step_authority.batch_size)
    launch_plan = launch_template.plan
    page_size = int(getattr(launch_plan, "page_size", 0))
    if page_size <= 0:
        raise RuntimeError("decode delta packet requires positive launch plan page_size")

    def _cpu_tuple(value: object, field_name: str) -> tuple[int, ...]:
        if isinstance(value, torch.Tensor):
            raise RuntimeError(
                f"decode delta packet requires CPU {field_name} tuple/list/scalar mirror"
            )
        if value is None:
            return tuple()
        if isinstance(value, tuple):
            raw = value
        elif isinstance(value, list):
            raw = tuple(value)
        else:
            raw = (value,)
        return tuple(int(v) for v in raw)

    context_kv_len_by_row = _cpu_tuple(
        getattr(step_authority, "context_kv_len_by_row", None),
        "context_kv_len_by_row",
    )
    canonical_real_kv_len_cpu = _cpu_tuple(
        getattr(step_bound_meta, "canonical_real_kv_len_cpu", None),
        "canonical_real_kv_len_cpu",
    )
    real_kv_len_by_row = (
        canonical_real_kv_len_cpu
        if len(canonical_real_kv_len_cpu) >= batch_size
        else context_kv_len_by_row
    )[:batch_size]
    if len(real_kv_len_by_row) < batch_size:
        raise RuntimeError(
            "decode delta packet requires real kv length CPU coverage; "
            f"batch_size={batch_size} coverage={len(real_kv_len_by_row)}"
        )

    use_compact_by_row = _cpu_tuple(
        getattr(step_authority, "use_compact_by_row", None),
        "use_compact_by_row",
    )[:batch_size]
    compact_valid_tokens_by_row = _cpu_tuple(
        getattr(launch_plan, "compact_valid_tokens_cpu", None),
        "compact_valid_tokens_cpu",
    )[:batch_size]
    current_recent_first_by_row = _cpu_tuple(
        getattr(step_bound_meta, "request_recent_first_logical_page", None),
        "request_recent_first_logical_page",
    )[:batch_size]
    current_recent_count_by_row = _cpu_tuple(
        getattr(step_bound_meta, "request_recent_page_count", None),
        "request_recent_page_count",
    )[:batch_size]
    recent_descriptor_block_size = int(
        getattr(step_bound_meta, "recent_descriptor_block_size", 0)
    )
    request_recent_epoch = int(
        getattr(step_bound_meta, "request_recent_epoch", -1)
    )
    request_kv_rows = _cpu_tuple(
        getattr(step_bound_meta, "request_kv_rows", None),
        "request_kv_rows",
    )[:batch_size]
    if (
        len(use_compact_by_row) < batch_size
        or len(compact_valid_tokens_by_row) < batch_size
        or len(current_recent_first_by_row) < batch_size
        or len(current_recent_count_by_row) < batch_size
        or len(request_kv_rows) < batch_size
    ):
        raise RuntimeError(
            "decode delta packet requires current compact/recent CPU coverage; "
            f"batch_size={batch_size} compact={len(use_compact_by_row)} "
            f"valid={len(compact_valid_tokens_by_row)} "
            f"first={len(current_recent_first_by_row)} "
            f"count={len(current_recent_count_by_row)} "
            f"kv_rows={len(request_kv_rows)}"
        )
    if recent_descriptor_block_size != page_size:
        raise RuntimeError(
            "decode delta packet requires current recent descriptor block size; "
            f"page_size={page_size} descriptor_block_size={recent_descriptor_block_size}"
        )
    if request_recent_epoch != int(step_authority.epoch):
        raise RuntimeError(
            "decode delta packet requires current recent descriptor epoch; "
            f"step_epoch={int(step_authority.epoch)} "
            f"request_recent_epoch={request_recent_epoch}"
        )
    if request_kv_rows != tuple(range(batch_size)):
        raise RuntimeError(
            "decode delta packet requires identity request_kv_rows coverage"
        )

    request_recent_len_list: list[int] = []
    launch_effective_k_list: list[int] = []
    recent_first_list: list[int] = []
    recent_count_list: list[int] = []
    for row in range(batch_size):
        real = max(0, int(real_kv_len_by_row[row]))
        compact_valid = max(0, int(compact_valid_tokens_by_row[row]))
        if bool(use_compact_by_row[row]) and compact_valid > 0:
            first = max(0, int(current_recent_first_by_row[row]))
            count = max(0, int(current_recent_count_by_row[row]))
            visible_recent = min(
                max(0, real - first * page_size),
                count * page_size,
            )
            request_recent_len = visible_recent
            launch_effective_k = compact_valid + visible_recent
        else:
            first = 0
            count = (real + page_size - 1) // page_size
            request_recent_len = real
            launch_effective_k = real
        recent_first_list.append(first)
        recent_count_list.append(count)
        request_recent_len_list.append(request_recent_len)
        launch_effective_k_list.append(launch_effective_k)

    q_lens_by_row = _cpu_tuple(
        getattr(step_authority, "q_lens_by_row", None),
        "q_lens_by_row",
    )[:batch_size]
    if len(q_lens_by_row) < batch_size:
        raise RuntimeError(
            "decode delta packet requires q_lens_by_row CPU coverage; "
            f"batch_size={batch_size} coverage={len(q_lens_by_row)}"
        )

    return DecodeDeltaPacket(
        step_id=int(step_authority.epoch),
        batch_size=batch_size,
        row_effective_k_by_row=tuple(launch_effective_k_list),
        request_recent_len_by_row=tuple(request_recent_len_list),
        launch_effective_k_by_row=tuple(launch_effective_k_list),
        recent_first_page_by_row=tuple(recent_first_list),
        recent_page_count_by_row=tuple(recent_count_list),
        q_lens_by_row=q_lens_by_row,
        logf_mask_generation=_decode_logf_signature_for_step(
            step_authority=step_authority,
            batch_size=batch_size,
        ),
        pending_refresh_state=_decode_pending_refresh_state_for_step(step_authority),
        row_dynamic_signature=_decode_row_dynamic_signature_for_step(
            step_authority=step_authority,
            launch_plan=launch_plan,
            batch_size=batch_size,
        ),
    )


def _collect_decode_delta_packet_from_launch_plan(
    *,
    step_authority: object,
    launch_plan: object,
    page_size: int,
) -> DecodeDeltaPacket:
    batch_size = int(step_authority.batch_size)
    if int(page_size) <= 0:
        raise RuntimeError("decode delta fast path requires positive page_size")
    context_kv_len_by_row = getattr(step_authority, "context_kv_len_by_row", None)
    use_compact_by_row = getattr(step_authority, "use_compact_by_row", None)
    q_lens_by_row = getattr(step_authority, "q_lens_by_row", None)
    if (
        context_kv_len_by_row is None
        or use_compact_by_row is None
        or q_lens_by_row is None
        or len(context_kv_len_by_row) < batch_size
        or len(use_compact_by_row) < batch_size
        or len(q_lens_by_row) < batch_size
    ):
        raise RuntimeError("decode delta fast path requires CPU row coverage")
    compact_valid_tokens_by_row = tuple(
        int(v)
        for v in tuple(getattr(launch_plan, "compact_valid_tokens_cpu", tuple()))[
            :batch_size
        ]
    )
    if len(compact_valid_tokens_by_row) < batch_size:
        raise RuntimeError("decode delta fast path requires compact_valid CPU mirror")
    recent_cap = int(getattr(step_authority, "recent_cap", 0))
    row_effective_k_list: list[int] = []
    request_recent_len_list: list[int] = []
    launch_effective_k_list: list[int] = []
    recent_first_list: list[int] = []
    recent_count_list: list[int] = []
    for row in range(batch_size):
        real = max(0, int(context_kv_len_by_row[row]))
        compact_valid = max(0, int(compact_valid_tokens_by_row[row]))
        if bool(use_compact_by_row[row]) and compact_valid > 0 and recent_cap > 0:
            window = derive_page_aligned_recent_window(
                real_kv_len=real,
                page_size=int(page_size),
                recent_tokens=recent_cap,
            )
            request_recent_len = int(window.visible_tokens)
            recent_first = int(window.first_logical_page)
            recent_count = int(window.page_count)
            launch_effective_k = compact_valid + request_recent_len
        else:
            request_recent_len = real
            recent_first = 0
            recent_count = (real + int(page_size) - 1) // int(page_size)
            launch_effective_k = real
        row_effective_k_list.append(launch_effective_k)
        request_recent_len_list.append(request_recent_len)
        launch_effective_k_list.append(launch_effective_k)
        recent_first_list.append(recent_first)
        recent_count_list.append(recent_count)

    q_lens = tuple(int(v) for v in q_lens_by_row[:batch_size])
    return DecodeDeltaPacket(
        step_id=int(step_authority.epoch),
        batch_size=batch_size,
        row_effective_k_by_row=tuple(row_effective_k_list),
        request_recent_len_by_row=tuple(request_recent_len_list),
        launch_effective_k_by_row=tuple(launch_effective_k_list),
        recent_first_page_by_row=tuple(recent_first_list),
        recent_page_count_by_row=tuple(recent_count_list),
        q_lens_by_row=q_lens,
        logf_mask_generation=_decode_logf_signature_for_step(
            step_authority=step_authority,
            batch_size=batch_size,
        ),
        pending_refresh_state=_decode_pending_refresh_state_for_step(step_authority),
        row_dynamic_signature=_decode_row_dynamic_signature_for_step(
            step_authority=step_authority,
            launch_plan=launch_plan,
            batch_size=batch_size,
        ),
    )


def _try_predict_same_page_delta_packet(
    *,
    runtime_state: DecodeRuntimeState,
    step_authority: object,
    page_size: int,
    launch_plan: object | None = None,
) -> tuple[DecodeDeltaPacket | None, str]:
    batch_size = int(getattr(step_authority, "batch_size", 0))
    q_lens_by_row = getattr(step_authority, "q_lens_by_row", None)
    if q_lens_by_row is None or len(q_lens_by_row) < batch_size:
        return None, "q_layout_changed"
    previous_delta = runtime_state.last_delta
    previous_signature = (
        tuple(previous_delta.row_dynamic_signature)
        if isinstance(previous_delta, DecodeDeltaPacket)
        else tuple()
    )
    current_signature = _decode_row_dynamic_signature_for_step(
        step_authority=step_authority,
        launch_plan=launch_plan,
        batch_size=batch_size,
    )
    if current_signature != previous_signature:
        return None, "row_dynamic_layout_changed"
    return predict_same_page_delta(
        previous_delta=previous_delta,
        step_id=int(getattr(step_authority, "epoch", -1)),
        page_size=int(page_size),
        q_lens_by_row=tuple(int(v) for v in q_lens_by_row[:batch_size]),
        logf_mask_generation=_decode_logf_signature_for_step(
            step_authority=step_authority,
            batch_size=batch_size,
        ),
        pending_refresh_state=_decode_pending_refresh_state_for_step(step_authority),
        row_dynamic_signature=previous_signature,
        # [p12 true root fix 20260612] recent_cap is the Site B recent_tokens;
        # it is not on the carried packet, so plumb it to enable the exact
        # first-logical-page slide check inside the predictor.
        recent_tokens=int(getattr(step_authority, "recent_cap", 0)),
    )




def _refresh_step_bound_meta_for_steady_delta(
    *,
    step_bound_meta: StepBoundMeta,
    step_meta: object,
    step_authority: object,
    delta: DecodeDeltaPacket,
    launch_plan: object,
    q_start_loc: Tuple[int, ...],
    prefill_rows: Tuple[int, ...],
) -> None:
    batch_size = int(step_authority.batch_size)
    step_bound_meta.step_handle_id = int(step_authority.step_handle_id)
    step_bound_meta.step_handle_generation = int(
        step_authority.step_handle_generation
    )
    step_bound_meta.epoch = int(step_authority.epoch)
    step_bound_meta.batch_size = batch_size
    _q_start_loc_t = tuple(int(v) for v in q_start_loc)
    _logits_last_n_t = tuple(
        int(v) for v in step_authority.logits_last_n_by_row[:batch_size]
    )
    _logits_capacity_t = tuple(
        int(v) for v in step_authority.logits_capacity_by_row[:batch_size]
    )
    _logf_mask_t = tuple(
        int(v) for v in step_authority.logf_mask_by_row[:batch_size]
    )
    _plan_signature_t = tuple(step_authority.plan_signature)
    step_bound_meta.q_start_loc = _q_start_loc_t
    step_bound_meta.q_lens_by_row = tuple(delta.q_lens_by_row)
    step_bound_meta.context_kv_len_by_row = tuple(
        int(v) for v in step_authority.context_kv_len_by_row[:batch_size]
    )
    step_bound_meta.logits_last_n_by_row = _logits_last_n_t
    step_bound_meta.logits_capacity_by_row = _logits_capacity_t
    step_bound_meta.logf_mask_by_row = _logf_mask_t
    step_bound_meta.logf_attn_rows = tuple(
        int(row)
        for row in step_authority.logf_attn_rows
        if 0 <= int(row) < batch_size
    )
    step_bound_meta.prefill_rows = tuple(prefill_rows)
    step_bound_meta.recent_cap = int(step_authority.recent_cap)
    step_bound_meta.sink_tokens = int(step_authority.sink_tokens)
    step_bound_meta.canonical_real_kv_len_cpu = step_bound_meta.context_kv_len_by_row
    step_bound_meta.canonical_real_kv_len_i32_gpu = getattr(
        step_meta,
        "canonical_real_kv_len_i32_gpu",
        None,
    )
    step_bound_meta.recent_descriptor_block_size = int(
        getattr(launch_plan, "page_size", 0)
    )
    step_bound_meta.request_recent_first_logical_page = tuple(
        delta.recent_first_page_by_row
    )
    step_bound_meta.request_recent_page_count = tuple(delta.recent_page_count_by_row)
    step_bound_meta.request_recent_epoch = int(step_authority.epoch)
    step_bound_meta.request_kv_rows = _identity_rows(batch_size)
    step_bound_meta.req_set_hash = int(step_authority.req_set_hash)
    step_bound_meta.row_phase_hash = int(step_authority.row_phase_hash)
    step_bound_meta.step_identity_token = int(step_authority.step_identity_token)
    step_bound_meta.plan_signature = _plan_signature_t
    step_bound_meta.bound_meta_signature = (
        int(step_authority.epoch),
        int(step_authority.step_handle_id),
        int(step_authority.step_handle_generation),
        _plan_signature_t,
        _logits_last_n_t,
        _logits_capacity_t,
        _logf_mask_t,
        _q_start_loc_t,
        tuple(int(v) for v in prefill_rows),
        tuple(int(v) for v in step_authority.row_mode_by_row[:batch_size]),
        _logf_mask_t,
    )
    step_bound_meta.compact_recent_launch_plan = launch_plan
    step_bound_meta.prologue_done_for_identity_token = -1
    step_bound_meta.prologue_owner_plan = None
    step_bound_meta.prologue_rail_decision = None
    step_bound_meta.prologue_selected_row_plan = None
    step_bound_meta.prologue_full_kv_handoff = False
    step_bound_meta.compact_mixed_page_overlay_by_layer.clear()
    step_bound_meta.resolved_row_ptr_arena_by_layer.clear()
    step_bound_meta.resolved_row_ptr_arena_key_by_layer.clear()
    step_bound_meta.compact_mixed_page_overlay_trace = tuple()
    descriptor_payload = _rrp_descriptor_payload_from_step_bound_meta(
        step_bound_meta,
        batch_size=batch_size,
    )
    if descriptor_payload is None:
        step_bound_meta.rrp_descriptor_epoch = -1
        step_bound_meta.affine_descriptor_by_row = tuple()
        step_bound_meta.row_table_pages_by_row = tuple()
        step_bound_meta.segment_pages_by_row = tuple()
    else:
        step_bound_meta.rrp_descriptor_epoch = int(step_authority.epoch)
        step_bound_meta.affine_descriptor_by_row = descriptor_payload.affine_descriptor_by_row
        step_bound_meta.row_table_pages_by_row = descriptor_payload.row_table_pages_by_row
        step_bound_meta.segment_pages_by_row = descriptor_payload.segment_pages_by_row


def _attn_metadata_has_resolved_row_ptr_replay_binding(attn_metadata: object) -> bool:
    return isinstance(
        getattr(attn_metadata, "mixed_page_resolver_replay_metadata_binding", None),
        ResolvedRowPtrReplayMetadataBinding,
    )


def _resolved_row_ptr_q_layout_key(
    *,
    step_authority: object,
    batch_size: int,
    live_batch_size: int | None = None,
    active_arena_row_indices: tuple[int, ...] | None = None,
    inactive_q_len: int = 0,
) -> str | None:
    batch_size_i = int(batch_size)
    live_batch_size_i = (
        batch_size_i if live_batch_size is None else int(live_batch_size)
    )
    q_lens_by_row = getattr(step_authority, "q_lens_by_row", None)
    q_start_loc_by_row = getattr(step_authority, "q_start_loc", None)
    if isinstance(q_lens_by_row, torch.Tensor) or isinstance(
        q_start_loc_by_row,
        torch.Tensor,
    ):
        return None
    if q_lens_by_row is None or len(q_lens_by_row) < live_batch_size_i:
        return None
    if q_start_loc_by_row is None or len(q_start_loc_by_row) < live_batch_size_i + 1:
        return None
    if live_batch_size_i != batch_size_i:
        if active_arena_row_indices is None:
            active_arena_row_indices = tuple(range(live_batch_size_i))
        if len(active_arena_row_indices) < live_batch_size_i:
            return None
        try:
            q_lens, q_start_loc = _project_rrp_q_layout(
                q_lens_by_row=q_lens_by_row,
                live_batch_size=live_batch_size_i,
                arena_batch_size=batch_size_i,
                active_arena_row_indices=tuple(
                    int(v) for v in active_arena_row_indices[:live_batch_size_i]
                ),
                inactive_q_len=max(0, int(inactive_q_len)),
            )
        except RuntimeError:
            return None
    else:
        q_lens = tuple(int(v) for v in q_lens_by_row[:batch_size_i])
        q_start_loc = tuple(int(v) for v in q_start_loc_by_row[: batch_size_i + 1])
    return f"q_lens={q_lens};q_start={q_start_loc}"


def _infer_rrp_graph_batch_size(
    *,
    attn_metadata: object,
    controller: object | None = None,
    live_batch_size: int,
) -> int:
    candidates: list[object] = [
        getattr(attn_metadata, "_resolved_row_ptr_graph_batch_size_override", None),
        getattr(
            controller,
            "_resolved_row_ptr_full_cudagraph_graph_batch_size",
            None,
        )
        if controller is not None
        else None,
        getattr(attn_metadata, "num_reqs", None),
        getattr(getattr(attn_metadata, "common_attn_metadata", None), "num_reqs", None),
    ]
    # [LIFECYCLE-OFF-ONLY 2026-07-10] 原 ON 专属 native capacity/structural
    # 候选块已删。
    try:
        from vllm.forward_context import (  # type: ignore[import]
            get_forward_context,
            is_forward_context_available,
        )
    except Exception:
        pass
    else:
        try:
            if bool(is_forward_context_available()):
                forward_context = get_forward_context()
                batch_descriptor = getattr(forward_context, "batch_descriptor", None)
                candidates.append(getattr(batch_descriptor, "num_reqs", None))
        except Exception:
            pass

    out = int(live_batch_size)
    for value in candidates:
        if value is None:
            continue
        try:
            parsed = int(value)
        except Exception:
            continue
        if parsed > out:
            out = parsed
    return out


def _project_rrp_row_tuple(
    values: tuple[object, ...],
    *,
    default: object,
    arena_batch_size: int,
    active_arena_row_indices: tuple[int, ...],
) -> tuple[object, ...]:
    projected = [default for _ in range(int(arena_batch_size))]
    for live_row, arena_row in enumerate(active_arena_row_indices):
        projected[int(arena_row)] = values[live_row]
    return tuple(projected)


def _project_rrp_q_layout(
    *,
    q_lens_by_row: object,
    live_batch_size: int,
    arena_batch_size: int,
    active_arena_row_indices: tuple[int, ...],
    inactive_q_len: int = 0,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if isinstance(q_lens_by_row, torch.Tensor):
        raise RuntimeError("RRP q projection requires CPU q_lens_by_row")
    if q_lens_by_row is None or len(q_lens_by_row) < int(live_batch_size):
        raise RuntimeError(
            "resolved-row-ptr replay metadata requires q_lens_by_row coverage; "
            f"live_batch_size={int(live_batch_size)}"
        )
    active_q_lens = tuple(int(v) for v in tuple(q_lens_by_row)[: int(live_batch_size)])
    q_lens = tuple(
        int(v)
        for v in _project_rrp_row_tuple(
            active_q_lens,
            default=max(0, int(inactive_q_len)),
            arena_batch_size=int(arena_batch_size),
            active_arena_row_indices=active_arena_row_indices,
        )
    )
    q_start_list = [0]
    for q_len in q_lens:
        q_start_list.append(q_start_list[-1] + max(0, int(q_len)))
    return q_lens, tuple(q_start_list)


def _rrp_descriptor_payload_from_step_bound_meta(
    step_bound_meta: object | None,
    *,
    batch_size: int,
) -> ResolvedRowPtrDescriptorPayload | None:
    if step_bound_meta is None:
        return None
    affine_raw = getattr(step_bound_meta, "affine_descriptor_by_row", None)
    row_table_raw = getattr(step_bound_meta, "row_table_pages_by_row", None)
    segment_raw = getattr(step_bound_meta, "segment_pages_by_row", None)
    if affine_raw is None or row_table_raw is None or segment_raw is None:
        return None
    batch_size_i = int(batch_size)
    # [T4-SBM-PAYLOAD-IDCACHE 2026-07-10] 单槽身份 memo(接替 07-02 实验遗留
    # 的 VLLM_SPARSE_CACHE_DESC_PAYLOAD 值比较旋钮:值比较逐 int 遍历与全量
    # 构造同量级=伪省,旋钮从未启用,循无旋钮纪律删除转正)。payload 是三
    # 源字段的纯函数;键存 payload 自身字段对象(1769 调用点会把 payload
    # 字段原样写回 sbm,下步 getattr 即同对象,is 比较 O(1) 首个 same-page
    # 步即命中);memo 持有引用无同址重生歧义;任何身份变化(commit 步重建
    # 描述符)=miss 走全量重建,fail-close 不做部分复用。
    cache = getattr(step_bound_meta, "_rrp_payload_cache", None)
    if (
        cache is not None
        and cache[0] is affine_raw
        and cache[1] is row_table_raw
        and cache[2] is segment_raw
        and cache[3] == batch_size_i
    ):
        return cache[4]
    try:
        affine_values = tuple(affine_raw)
        row_table_values = tuple(row_table_raw)
        segment_values = tuple(int(v) for v in segment_raw)
    except (TypeError, ValueError):
        return None
    if (
        batch_size_i <= 0
        or len(affine_values) < batch_size_i
        or len(row_table_values) < batch_size_i
        or len(segment_values) < batch_size_i
    ):
        return None
    _payload = ResolvedRowPtrDescriptorPayload(
        affine_descriptor_by_row=tuple(affine_values[:batch_size_i]),
        row_table_pages_by_row=tuple(
            tuple(int(page) for page in row_table_values[row])
            for row in range(batch_size_i)
        ),
        segment_pages_by_row=tuple(int(v) for v in segment_values[:batch_size_i]),
    )
    try:
        step_bound_meta._rrp_payload_cache = (
            _payload.affine_descriptor_by_row,
            _payload.row_table_pages_by_row,
            _payload.segment_pages_by_row,
            batch_size_i,
            _payload,
        )
    except Exception:
        pass
    return _payload


def _try_attach_same_page_resolved_row_ptr_replay_metadata(
    controller: object,
    *,
    attn_metadata: object,
    step_authority: object,
    batch_size: int,
    block_size: int,
    num_kv_heads: int,
    device: torch.device,
    launch_plan: object | None = None,
    profile_phase_us: dict[str, float] | None = None,
    live_batch_size: int | None = None,
    active_arena_row_indices: tuple[int, ...] | None = None,
) -> object | None:
    _mark_rrp_attach_phase = None
    if profile_phase_us is not None:
        attach_phase_start_ns = time.perf_counter_ns()

        def _mark_rrp_attach_phase(name: str) -> None:
            nonlocal attach_phase_start_ns
            now_ns = time.perf_counter_ns()
            profile_phase_us[name] = float(now_ns - attach_phase_start_ns) / 1000.0
            attach_phase_start_ns = now_ns

    mode = _decode_runtime_mode_for_controller(controller)
    if mode not in (
        DecodeRuntimeMode.STEADY_DELTA,
        DecodeRuntimeMode.PAGE_BOUNDARY_DELTA,
    ):
        return None
    if not _decode_runtime_classification_is_current(controller, step_authority):
        return None
    if not bool(getattr(controller, "_resolved_row_ptr_metadata_ready", False)):
        return None
    batch_size_i = int(batch_size)
    live_batch_size_i = (
        batch_size_i if live_batch_size is None else int(live_batch_size)
    )
    if live_batch_size_i <= 0 or live_batch_size_i > batch_size_i:
        return None
    if active_arena_row_indices is None:
        active_arena_row_indices = tuple(range(live_batch_size_i))
    active_arena_row_indices = tuple(
        int(v) for v in tuple(active_arena_row_indices)[:live_batch_size_i]
    )
    if len(active_arena_row_indices) != live_batch_size_i:
        return None
    if any(row < 0 or row >= batch_size_i for row in active_arena_row_indices):
        return None
    projected_graph_capacity = live_batch_size_i != batch_size_i
    inactive_dummy_k = 0

    def _project_live_values(values: object, *, default: object) -> tuple[object, ...] | None:
        if isinstance(values, torch.Tensor):
            return None
        try:
            live_values = tuple(values)[:live_batch_size_i]
        except Exception:
            return None
        if len(live_values) < live_batch_size_i:
            return None
        if not projected_graph_capacity:
            return tuple(live_values)
        return _project_rrp_row_tuple(
            tuple(live_values),
            default=default,
            arena_batch_size=batch_size_i,
            active_arena_row_indices=active_arena_row_indices,
        )

    if projected_graph_capacity:
        canonical_rows = [-1] * batch_size_i
        for live_row, arena_row in enumerate(active_arena_row_indices):
            canonical_rows[int(arena_row)] = int(live_row)
        canonical_row_index_by_batch_row = tuple(canonical_rows)
    else:
        canonical_row_index_by_batch_row = tuple(range(batch_size_i))

    delta = getattr(controller, "_decode_runtime_delta", None)
    if not isinstance(delta, DecodeDeltaPacket) or int(delta.batch_size) != live_batch_size_i:
        return None
    replay_arena = getattr(controller, "_resolved_row_ptr_replay_arena", None)
    if not isinstance(replay_arena, ResolvedRowPtrArena):
        return None
    if (
        int(replay_arena.batch_size) != batch_size_i
        or int(replay_arena.num_kv_heads) != int(num_kv_heads)
    ):
        return None
    current_arena_key = getattr(controller, "_resolved_row_ptr_arena_key", None)
    expected_arena_key = (
        batch_size_i,
        int(num_kv_heads),
        int(block_size),
        int(replay_arena.max_pages_per_row),
        _device_identity(device),
    )
    if current_arena_key != expected_arena_key:
        return None
    if not _resolved_row_ptr_current_visible_source_matches_binding(
        controller=controller,
        attn_metadata=attn_metadata,
        replay_arena=replay_arena,
        batch_size=batch_size_i,
        launch_plan=launch_plan,
    ):
        return None
    if _mark_rrp_attach_phase is not None:
        _mark_rrp_attach_phase("rrp_attach_guards")
    binding = getattr(controller, "_resolved_row_ptr_replay_metadata_binding", None)
    if not isinstance(binding, ResolvedRowPtrReplayMetadataBinding):
        return None
    if binding.replay_arena is not replay_arena:
        return None
    q_layout_key = _resolved_row_ptr_q_layout_key(
        step_authority=step_authority,
        batch_size=batch_size_i,
        live_batch_size=live_batch_size_i,
        active_arena_row_indices=active_arena_row_indices,
        inactive_q_len=inactive_dummy_k,
    )
    if q_layout_key is None or str(binding.descriptor.q_layout_key) != q_layout_key:
        return None
    manager = getattr(controller, "_rrp_row_table_manager", None)
    if not isinstance(manager, RrpRowTableManager):
        return None
    if not _same_page_rrp_signature_matches_step_authority(
        manager=manager,
        step_authority=step_authority,
        batch_size=batch_size_i,
        live_batch_size=live_batch_size_i,
        active_arena_row_indices=active_arena_row_indices,
    ):
        return None
    if _mark_rrp_attach_phase is not None:
        _mark_rrp_attach_phase("rrp_attach_binding_checks")
    if mode is DecodeRuntimeMode.STEADY_DELTA:
        row_effective_k_by_row = tuple(
            int(v) for v in tuple(delta.row_effective_k_by_row)[:live_batch_size_i]
        )
        if len(row_effective_k_by_row) != live_batch_size_i:
            return None
        if projected_graph_capacity:
            previous_row_effective = getattr(manager, "_row_effective_k_by_row", None)
            if (
                not isinstance(previous_row_effective, tuple)
                or len(previous_row_effective) != batch_size_i
            ):
                return None
            row_effective_list = [
                int(v) for v in previous_row_effective[:batch_size_i]
            ]
            for live_row, arena_row in enumerate(active_arena_row_indices):
                row_effective_list[int(arena_row)] = int(row_effective_k_by_row[live_row])
            row_effective_k_by_row = tuple(row_effective_list)
        update = manager.try_apply_same_page_delta(
            replay_arena,
            row_effective_k_by_row,
        )
        if update is None:
            return None
    else:
        if launch_plan is None:
            launch_plan = getattr(
                getattr(controller, "step_bound_meta", None),
                "compact_recent_launch_plan",
                None,
            )
        if launch_plan is None or not bool(getattr(launch_plan, "valid", False)):
            return None
        worker_block_table = getattr(controller, "_worker_block_table", None)
        worker_block_table_cpu = getattr(controller, "_worker_block_table_cpu", None)
        try:
            worker_block_table_i32 = _validate_block_table_for_resolved_row_ptr_replay(
                block_table=worker_block_table,
                batch_size=live_batch_size_i,
                max_pages_per_row=int(replay_arena.max_pages_per_row),
                device=device,
            )
        except Exception:
            return None
        req_ids_source = getattr(step_authority, "req_ids", None)
        slot_source = getattr(step_authority, "slot_by_row", None)
        row_mode_source = getattr(step_authority, "row_mode_by_row", None)
        if (
            req_ids_source is None
            or slot_source is None
            or row_mode_source is None
            or isinstance(req_ids_source, torch.Tensor)
            or isinstance(slot_source, torch.Tensor)
            or isinstance(row_mode_source, torch.Tensor)
            or len(req_ids_source) < live_batch_size_i
            or len(slot_source) < live_batch_size_i
            or len(row_mode_source) < live_batch_size_i
        ):
            return None
        req_ids_projected = _project_live_values(
            req_ids_source,
            default="__inactive_rrp_row__",
        )
        slot_projected = _project_live_values(slot_source, default=-1)
        row_mode_projected = _project_live_values(
            row_mode_source,
            default=int(_ROW_MODE_DENSE),
        )
        if (
            req_ids_projected is None
            or slot_projected is None
            or row_mode_projected is None
        ):
            return None
        req_ids_by_row = tuple(str(v) for v in req_ids_projected[:batch_size_i])
        slot_by_row = tuple(int(v) for v in slot_projected[:batch_size_i])
        row_mode_by_row = tuple(int(v) for v in row_mode_projected[:batch_size_i])
        compact_ready_by_batch_row = tuple(
            int(v) == int(_ROW_MODE_COMPACT) for v in row_mode_by_row
        )
        if not any(compact_ready_by_batch_row):
            return None
        compact_valid_source = getattr(
            launch_plan,
            "compact_valid_tokens_cpu",
            tuple(),
        )
        compact_offset_source = getattr(
            launch_plan,
            "compact_offset_tokens_cpu",
            tuple(),
        )
        if (
            isinstance(compact_valid_source, torch.Tensor)
            or isinstance(compact_offset_source, torch.Tensor)
            or len(compact_valid_source) < live_batch_size_i
            or len(compact_offset_source) < live_batch_size_i
        ):
            return None
        compact_valid_projected = _project_live_values(
            compact_valid_source,
            default=0,
        )
        compact_offset_projected = _project_live_values(
            compact_offset_source,
            default=0,
        )
        if compact_valid_projected is None or compact_offset_projected is None:
            return None
        compact_valid_tokens_by_row = tuple(
            int(v) for v in compact_valid_projected[:batch_size_i]
        )
        compact_offset_tokens_by_row = tuple(
            int(v) for v in compact_offset_projected[:batch_size_i]
        )
        descriptor_snapshot = None
        if len(delta.row_effective_k_by_row) < live_batch_size_i or len(
            delta.recent_first_page_by_row
        ) < live_batch_size_i:
            return None
        row_effective_projected = _project_live_values(
            delta.row_effective_k_by_row,
            default=inactive_dummy_k,
        )
        recent_first_projected = _project_live_values(
            delta.recent_first_page_by_row,
            default=0,
        )
        if row_effective_projected is None or recent_first_projected is None:
            return None
        row_effective_k_by_row = tuple(
            int(v) for v in row_effective_projected[:batch_size_i]
        )
        recent_first_page_by_row = tuple(
            int(v) for v in recent_first_projected[:batch_size_i]
        )
        lease = _compact_page_lease_from_controller(controller)
        (
            reserved_manager_block_ids,
            reserved_manager_block_ids_arg,
            compact_capacity_pages,
        ) = _resolved_row_ptr_lease_snapshot(controller, lease=lease, device=device)
        if int(compact_capacity_pages) <= 0:
            return None
        row_effective_k_i32_gpu = _resolved_row_ptr_launch_effective_k_len_source(
            launch_plan=launch_plan,
            batch_size=batch_size_i,
            device=device,
        )
        if projected_graph_capacity:
            row_effective_k_i32_gpu = None
        manager_inputs = RrpRowTableInputs(
            batch_size=batch_size_i,
            req_ids_by_row=req_ids_by_row,
            slot_by_row=slot_by_row,
            row_mode_by_row=row_mode_by_row,
            compact_ready_by_row=compact_ready_by_batch_row,
            row_effective_k_by_row=row_effective_k_by_row,
            compact_valid_tokens_by_row=compact_valid_tokens_by_row,
            compact_offset_tokens_by_row=compact_offset_tokens_by_row,
            recent_first_page_by_row=recent_first_page_by_row,
            reserved_manager_block_ids=reserved_manager_block_ids,
            compact_capacity_pages=int(compact_capacity_pages),
            max_pages_per_row=int(replay_arena.max_pages_per_row),
            page_size=int(block_size),
            row_effective_k_i32_gpu=row_effective_k_i32_gpu,
        )
        if _mark_rrp_attach_phase is not None:
            _mark_rrp_attach_phase("rrp_attach_inputs_ready")
        if descriptor_snapshot is not None:
            update = manager.publish_from_descriptor_snapshot(
                replay_arena,
                descriptor_snapshot,
            )
        else:
            update = manager.update(replay_arena, manager_inputs)
        if _mark_rrp_attach_phase is not None:
            _mark_rrp_attach_phase("rrp_attach_manager_update")
        if update is None:
            return None
        if bool(getattr(update, "metadata_bind_required", False)) or bool(
            getattr(update, "full_bind", False)
        ):
            return None
        if descriptor_snapshot is None and bool(getattr(update, "delta_rows", tuple())):
            direct_planned_layout = PlannedCompactRowLayout(
                compact_ready_by_batch_row=manager_inputs.compact_ready_by_row,
                slot_by_row=manager_inputs.slot_by_row,
                compact_valid_tokens_by_row=manager_inputs.compact_valid_tokens_by_row,
                compact_offset_tokens_by_row=manager_inputs.compact_offset_tokens_by_row,
                recent_first_page_by_row=manager_inputs.recent_first_page_by_row,
                recent_page_count_by_row=(0,) * int(manager_inputs.batch_size),
                row_effective_k_by_row=manager_inputs.row_effective_k_by_row,
            )
            direct_affine_updated = replay_arena.try_bind_production_direct_affine(
                canonical_block_table=worker_block_table_i32,
                compact_ready_by_batch_row=compact_ready_by_batch_row,
                row_effective_k_by_row=row_effective_k_by_row,
                page_size=int(block_size),
                reserved_manager_block_ids=reserved_manager_block_ids_arg,
                slot_by_row=slot_by_row,
                compact_valid_tokens_by_row=compact_valid_tokens_by_row,
                compact_offset_tokens_by_row=compact_offset_tokens_by_row,
                recent_first_page_by_row=recent_first_page_by_row,
                canonical_row_index_by_batch_row=canonical_row_index_by_batch_row,
                compact_capacity_pages=int(compact_capacity_pages),
                profile_phase_us=profile_phase_us,
                planned_layout=direct_planned_layout,
                reserved_manager_block_ids_cpu=reserved_manager_block_ids,
                canonical_block_table_cpu=worker_block_table_cpu,
            )
            if direct_affine_updated:
                if _mark_rrp_attach_phase is not None:
                    _mark_rrp_attach_phase("rrp_attach_direct_affine_update")
            else:
                replay_arena.bind_production_row_table(
                    canonical_block_table=worker_block_table_i32,
                    compact_ready_by_batch_row=compact_ready_by_batch_row,
                    row_effective_k_by_row=row_effective_k_by_row,
                    page_size=int(block_size),
                    reserved_manager_block_ids=reserved_manager_block_ids_arg,
                    slot_by_row=slot_by_row,
                    compact_valid_tokens_by_row=compact_valid_tokens_by_row,
                    compact_offset_tokens_by_row=compact_offset_tokens_by_row,
                    recent_first_page_by_row=recent_first_page_by_row,
                    canonical_row_index_by_batch_row=canonical_row_index_by_batch_row,
                    compact_capacity_pages=int(compact_capacity_pages),
                    profile_phase_us=profile_phase_us,
                    reserved_manager_block_ids_cpu=reserved_manager_block_ids,
                    canonical_block_table_cpu=worker_block_table_cpu,
                )
                if _mark_rrp_attach_phase is not None:
                    _mark_rrp_attach_phase("rrp_attach_production_row_table")
    if _mark_rrp_attach_phase is not None:
        _mark_rrp_attach_phase("rrp_attach_manager_delta")
    attach_resolved_row_ptr_replay_metadata(
        attn_metadata=attn_metadata,
        binding=binding,
    )
    if _mark_rrp_attach_phase is not None:
        _mark_rrp_attach_phase("rrp_attach_metadata_publish")
    if launch_plan is None:
        launch_plan = getattr(
            getattr(controller, "step_bound_meta", None),
            "compact_recent_launch_plan",
            None,
        )
    _publish_resolved_row_ptr_profile_max_seqlen_k(
        attn_metadata=attn_metadata,
        launch_plan=launch_plan,
        row_effective_k_by_row=row_effective_k_by_row,
        batch_size=batch_size_i,
        block_size=int(block_size),
        max_pages_per_row=int(replay_arena.max_pages_per_row),
    )
    if _mark_rrp_attach_phase is not None:
        _mark_rrp_attach_phase("rrp_attach_profile_max_publish")
    controller._decode_runtime_rrp_metadata_light_attach_count = (
        int(getattr(controller, "_decode_runtime_rrp_metadata_light_attach_count", 0))
        + 1
    )
    controller._resolved_row_ptr_metadata_ready = True
    _ready_event_holders = _resolved_row_ptr_ready_event_holders(attn_metadata)
    (
        existing_ready_event_generation,
        existing_ready_event,
        existing_ready_event_stream,
    ) = _resolved_row_ptr_ready_event_state(
        attn_metadata, holders=_ready_event_holders
    )
    _attach_resolved_row_ptr_ready_event_state(
        attn_metadata,
        ready_event_generation=existing_ready_event_generation,
        ready_event=existing_ready_event,
        ready_event_stream=existing_ready_event_stream,
        holders=_ready_event_holders,
    )
    ready_event_recorded = False
    if should_record_rrp_ready_event(
        update,
        metadata_bind_required=False,
        existing_ready_event_generation=existing_ready_event_generation,
    ):
        ready_event_recorded = _record_resolved_row_ptr_replay_ready_event(
            attn_metadata,
            device=device,
            holders=_ready_event_holders,
        )
    (
        ready_event_generation,
        ready_event,
        ready_event_stream,
    ) = _resolved_row_ptr_ready_event_state(
        attn_metadata, holders=_ready_event_holders
    )
    if _mark_rrp_attach_phase is not None:
        _mark_rrp_attach_phase("rrp_attach_ready_state")
    setattr(
        controller,
        "_resolved_row_ptr_ready_event_recorded",
        bool(ready_event_recorded),
    )
    if not _refresh_resolved_row_ptr_graph_binding_ready_state(
        controller,
        binding=binding,
        arena_key=expected_arena_key,
        ready_event_generation=int(ready_event_generation),
        ready_event=ready_event,
        ready_event_stream=int(ready_event_stream),
    ):
        _publish_resolved_row_ptr_graph_binding_state(
            controller,
            binding=binding,
            arena_key=expected_arena_key,
            ready_event_generation=int(ready_event_generation),
            ready_event=ready_event,
            ready_event_stream=int(ready_event_stream),
        )
    _stamp_resolved_row_ptr_graph_binding_step(
        controller=controller,
        step_authority=step_authority,
        live_batch_size=live_batch_size_i,
        effective_batch_size=batch_size_i,
    )
    if _mark_rrp_attach_phase is not None:
        _mark_rrp_attach_phase("rrp_attach_graph_binding_publish")
    return update


def _try_page_add_incremental(
    controller,
    *,
    step_authority,
    delta,
    replay_arena,
    manager,
    batch_size,
    block_size,
    launch_plan,
    device,
):
    # VLLM_SPARSE_PAGE_ADD_INCREMENTAL (lifecycle-OFF): a genuine recent-page
    # GROWTH miss (try_apply_same_page_delta -> None because visible_page_count
    # grew) ADDS the grown page(s) in place via publish_live_rows instead of the
    # ~641us full rebind. Pages are derived to MATCH the heavy bind byte-for-byte
    # (compact from the lease reserved ids; recent via the heavy
    # ceil((effk-compact)/ps) formula, NOT delta.recent_page_count which can
    # diverge). Manager currency (_row_effective via _apply_seqused_delta,
    # _visible_page_count = recompute formula, _recent_first) is advanced so the
    # NEXT step _is_pure_seqused_delta is self-consistent (hit until real growth).
    # Returns a PAGE_BOUNDARY_DELTA RrpUpdateResult, or None to fall through to
    # the full rebind (graceful). BYTE-IDENTITY is the gate (kernel reads the
    # fallback row_table_i32 directly; Leg-B precedent).
    try:
        ps = int(block_size)
        if ps <= 0 or launch_plan is None:
            return None
        compact_valid_cpu = getattr(launch_plan, "compact_valid_tokens_cpu", None)
        compact_offset_cpu = getattr(launch_plan, "compact_offset_tokens_cpu", None)
        block_table_cpu = getattr(controller, "_worker_block_table_cpu", None)
        publish = getattr(replay_arena, "publish_live_rows", None)
        if (compact_valid_cpu is None or compact_offset_cpu is None
                or block_table_cpu is None or not callable(publish)):
            return None
        bs = int(batch_size)
        if (len(compact_valid_cpu) < bs or len(compact_offset_cpu) < bs
                or len(delta.row_effective_k_by_row) < bs
                or len(delta.recent_first_page_by_row) < bs):
            return None
        reserved = _resolved_row_ptr_lease_snapshot(
            controller,
            lease=_compact_page_lease_from_controller(controller),
            device=device,
        )[0]
        from patches.fa_sparse_runtime.live_page_table import derive_live_page_rows
        # [T5-PAGE-ADD-COMPACT-MEMO 2026-07-10] compact 段输入(valid/offset)
        # 在 STEADY 模式内恒定(变则重分类,到不了本臂;见下 DIRTY-ONLY 注),
        # 而 compact_pages_by_row 曾每页界步全量重物化 8×~百元素 tuple(int)。
        # memo 键=旧档 §10.23 设计:(reserved 元组身份+长度, valid, offset,
        # page_size, bs)——reserved 经 lease snapshot memo 命中时跨步同对象,
        # id 锚成立;任一键分量变(含 lease 重建)整表重建,越界防卫随建随验,
        # miss=全量重建非跳过。消费方(derive 只读 .get/publish del 形参)均只读。
        _cpm_valid_t = tuple(int(compact_valid_cpu[r]) for r in range(bs))
        _cpm_off_t = tuple(int(compact_offset_cpu[r]) for r in range(bs))
        _cpm_key = (id(reserved), len(reserved), _cpm_valid_t, _cpm_off_t, ps, bs)
        _cpm_cached = getattr(controller, "_page_add_compact_pages_memo", None)
        if (
            isinstance(_cpm_cached, tuple)
            and len(_cpm_cached) == 3
            and _cpm_cached[0] == _cpm_key
        ):
            compact_pages_by_row = _cpm_cached[1]
            compact_pages_count_by_row = _cpm_cached[2]
        else:
            compact_pages_by_row = {}
            compact_pages_count_by_row = {}
            for r in range(bs):
                compact_tokens = max(0, _cpm_valid_t[r])
                compact_pages = compact_tokens // ps
                compact_off_pages = _cpm_off_t[r] // ps
                if compact_off_pages < 0 or compact_off_pages + compact_pages > len(reserved):
                    return None
                compact_pages_by_row[r] = tuple(
                    int(v) for v in reserved[compact_off_pages:compact_off_pages + compact_pages]
                )
                compact_pages_count_by_row[r] = compact_pages
            controller._page_add_compact_pages_memo = (
                _cpm_key,
                compact_pages_by_row,
                compact_pages_count_by_row,
            )
        recent_first_by_row = {}
        recent_count_by_row = {}
        vpc = []
        for r in range(bs):
            compact_tokens = max(0, _cpm_valid_t[r])
            row_effk = max(0, int(delta.row_effective_k_by_row[r]))
            rc = (max(0, row_effk - compact_tokens) + ps - 1) // ps
            recent_count_by_row[r] = rc
            recent_first_by_row[r] = int(delta.recent_first_page_by_row[r])
            vpc.append(compact_pages_count_by_row[r] + rc)
        # [PAGE-ADD-DIRTY-ONLY 2026-07-08] re-lay ONLY the rows whose page set
        # actually changed instead of the whole batch every step. A row is
        # dirty when its token-ceil page count moved vs the write-epoch shadow
        # (same metric both sides) or its recent-first page slid. Unchanged
        # rows keep their previously-laid pages — identical reuse semantics to
        # a try_apply HIT; they only entered this leg because ANOTHER row in
        # the batch missed (batch-global probe). Compact segment inputs
        # (valid/offset) are constant inside STEADY mode: any change flips
        # row_dynamic_signature and classifies as PAGE_BOUNDARY/REFRESH_COMMIT,
        # which never reaches this leg. Incomplete shadow state -> conservative
        # full re-lay.
        _effk_now = tuple(
            int(v) for v in tuple(delta.row_effective_k_by_row)[:bs]
        )
        _shadow_now = manager._shadow_visible_pages(_effk_now)
        _shadow_stored = manager._probe_visible_page_count_by_row
        _rf_stored = manager._recent_first_page_by_row
        if (
            _shadow_now is not None
            and isinstance(_shadow_stored, tuple)
            and isinstance(_rf_stored, tuple)
            and len(_shadow_stored) == bs
            and len(_rf_stored) == bs
        ):
            dirty = tuple(
                r
                for r in range(bs)
                if int(_shadow_now[r]) != int(_shadow_stored[r])
                or int(recent_first_by_row[r]) != int(_rf_stored[r])
            )
        else:
            dirty = tuple(range(bs))
        if dirty:
            pages = derive_live_page_rows(
                active_rows=dirty,
                block_table_cpu=block_table_cpu,
                compact_pages_by_row=compact_pages_by_row,
                recent_first_page_by_row=recent_first_by_row,
                recent_page_count_by_row=recent_count_by_row,
            )
            publish(pages_by_row=pages, dirty_rows=dirty,
                    compact_pages_by_row=compact_pages_by_row)
        manager._apply_seqused_delta(
            replay_arena,
            tuple(int(v) for v in tuple(delta.row_effective_k_by_row)[:bs]),
        )
        manager._visible_page_count_by_row = tuple(vpc)
        # [VPC-SHADOW-METRIC] this leg's vpc uses floor compact pages
        # (compact_tokens // ps) vs the probe's ceil — recompute the probe's
        # own token-ceil shadow instead of copying, so a non-page-aligned
        # compact_valid can never phase-split the comparison.
        manager._probe_visible_page_count_by_row = manager._shadow_visible_pages(
            tuple(int(v) for v in tuple(delta.row_effective_k_by_row)[:bs])
        )
        manager._recent_first_page_by_row = tuple(
            recent_first_by_row[r] for r in range(bs)
        )
        return RrpUpdateResult(
            hit=False,
            miss_reason="page_add_incremental",
            delta_rows=tuple(range(bs)),
            full_bind=False,
            kind=RrpUpdateKind.PAGE_BOUNDARY_DELTA,
        )
    except Exception:
        return None


def _try_update_same_page_resolved_row_ptr_step_state(
    controller: object,
    *,
    attn_metadata: object,
    step_authority: object,
    batch_size: int,
    block_size: int,
    num_kv_heads: int,
    device: torch.device,
    launch_plan: object | None = None,
    same_page_proven: bool = False,
) -> object | None:
    setattr(controller, "_decode_runtime_rrp_step_state_miss_reason", "")
    # [Z-RRP-PROBE] diagnostic-only (env, default off) — 7.9ms plateau hunt.
    # [U14-ZERO-ALLOC-2-8 2026-07-12] env-cache(生产期免每步 os.environ.get)。
    _z_probe = (
        (os.environ.get("VLLM_SPARSE_Z_RRP_PROBE") == "1")
        if _DYNAMIC_ENV
        else _Z_RRP_PROBE_CACHED
    )
    _z_t0 = _z_t1 = 0
    _z_hit_before_page_add = False

    def _miss(reason: str) -> object | None:
        setattr(controller, "_decode_runtime_rrp_step_state_miss_reason", str(reason))
        return None

    mode = _decode_runtime_mode_for_controller(controller)
    if mode is not DecodeRuntimeMode.STEADY_DELTA:
        _page_boundary_live_route = (
            os.environ.get("VLLM_SPARSE_CLEAN_METADATA", "1") == "1"
            if _DYNAMIC_ENV
            else _CLEAN_METADATA_CACHED
        ) and mode is DecodeRuntimeMode.PAGE_BOUNDARY_DELTA
        if not (
            _page_boundary_live_route
            and (
                os.environ.get("VLLM_SPARSE_PAGE_ADD_INCREMENTAL", "1") == "1"
                if _DYNAMIC_ENV
                else _PAGE_ADD_INCREMENTAL_CACHED
            )
        ):
            return _miss(f"mode_not_steady:{mode.value}")
        # [PAGE-BOUNDARY-FIRST-PASS 2026-07-09] lifecycle-OFF production route:
        # fall through into the steady verification arm below. A boundary step
        # runs the SAME sequence the slow path's rescue ran on its second
        # attempt (full replay-safety preconditions -> try_apply_same_page_delta
        # says None on a true boundary -> _try_page_add_incremental lays the
        # slid/grown rows from the lease + launch_plan, byte-identical to the
        # heavy bind) — so the FIRST attempt now succeeds where it previously
        # burned ~400us on the dead Leg-B arm and forced a full second
        # metadata pass (592/1068 steps double-ran per 12k run).
    if not _decode_runtime_classification_is_current(controller, step_authority):
        return _miss("classification_not_current")
    if not bool(getattr(controller, "_resolved_row_ptr_metadata_ready", False)):
        return _miss("rrp_metadata_not_ready")
    batch_size_i = int(batch_size)
    delta = getattr(controller, "_decode_runtime_delta", None)
    if not isinstance(delta, DecodeDeltaPacket) or int(delta.batch_size) != batch_size_i:
        return _miss("delta_missing_or_batch_mismatch")
    if (
        bool(same_page_proven)
        and (
            os.environ.get("VLLM_SPARSE_RRP_SAME_PAGE_SKIP_REVALIDATION", "1") == "1"
            if _DYNAMIC_ENV
            else _RRP_SAME_PAGE_SKIP_REVALIDATION_CACHED
        )
    ):
        # Cut #3 structured gated skip: the minimal super-fast-path already proved
        # THIS step is a pure same-page STEADY_DELTA step (recent_first/recent_count/
        # q_lens/logf_generation unchanged vs the prior applied delta, classification
        # re-confirmed STEADY_DELTA, capacity re-checked by apply_launch_template_row_
        # delta). So arena identity, the carrier->seqused-source binding, the q-layout
        # key and the req-id row signature are bind-time invariants that cannot have
        # changed -> skip the by-value re-proof and run only the truly-variant work:
        # the seqused/effk GPU write (whose own _is_pure_seqused_delta is the last-line
        # page-move catch) + the step-fast-identity stamp. NOT a memoized proof.
        replay_arena = getattr(controller, "_resolved_row_ptr_replay_arena", None)
        if not isinstance(replay_arena, ResolvedRowPtrArena):
            return _miss("replay_arena_missing")
        binding = getattr(
            controller, "_resolved_row_ptr_replay_metadata_binding", None
        )
        if not isinstance(binding, ResolvedRowPtrReplayMetadataBinding):
            return _miss("binding_missing")
        manager = getattr(controller, "_rrp_row_table_manager", None)
        if not isinstance(manager, RrpRowTableManager):
            return _miss("row_table_manager_missing")
        visible = replay_arena.carriers.resolver_visible_seqused_k_by_head_i32
        if not isinstance(visible, torch.Tensor):
            return _miss("visible_tensor_missing")
        row_effective_k_by_row = tuple(
            int(v) for v in tuple(delta.row_effective_k_by_row)[:batch_size_i]
        )
        if len(row_effective_k_by_row) != batch_size_i:
            return _miss("row_effective_len_mismatch")
        update = manager.try_apply_same_page_delta(
            replay_arena,
            row_effective_k_by_row,
            recent_first_page_by_row=tuple(
                int(v) for v in tuple(delta.recent_first_page_by_row)[:batch_size_i]
            ),
        )
        if (
            update is None
            and (
                os.environ.get("VLLM_SPARSE_PAGE_ADD_INCREMENTAL", "1") == "1"
                if _DYNAMIC_ENV
                else _PAGE_ADD_INCREMENTAL_CACHED
            )
        ):
            update = _try_page_add_incremental(
                controller,
                step_authority=step_authority,
                delta=delta,
                replay_arena=replay_arena,
                manager=manager,
                batch_size=batch_size_i,
                block_size=int(block_size),
                launch_plan=launch_plan,
                device=device,
            )
        if update is None:
            return _miss("row_table_same_page_delta_update_miss")
        if (
            getattr(attn_metadata, "mixed_page_resolver_replay_metadata_binding", None)
            is not binding
        ):
            setattr(
                controller,
                "_decode_runtime_rrp_step_state_missing_metadata_view_count",
                int(
                    getattr(
                        controller,
                        "_decode_runtime_rrp_step_state_missing_metadata_view_count",
                        0,
                    )
                )
                + 1,
            )
        _stamp_resolved_row_ptr_step_state(
            controller=controller,
            step_authority=step_authority,
            live_batch_size=batch_size_i,
            effective_batch_size=batch_size_i,
        )
        setattr(controller, "_decode_runtime_rrp_step_state_miss_reason", "hit")
        return update
    replay_arena = getattr(controller, "_resolved_row_ptr_replay_arena", None)
    if not isinstance(replay_arena, ResolvedRowPtrArena):
        return _miss("replay_arena_missing")
    if (
        int(replay_arena.batch_size) != batch_size_i
        or int(replay_arena.num_kv_heads) != int(num_kv_heads)
    ):
        return _miss("replay_arena_shape_mismatch")
    expected_arena_key = (
        batch_size_i,
        int(num_kv_heads),
        int(block_size),
        int(replay_arena.max_pages_per_row),
        _device_identity(device),
    )
    if getattr(controller, "_resolved_row_ptr_arena_key", None) != expected_arena_key:
        return _miss("arena_key_mismatch")
    binding = getattr(controller, "_resolved_row_ptr_replay_metadata_binding", None)
    if not isinstance(binding, ResolvedRowPtrReplayMetadataBinding):
        return _miss("binding_missing")
    if binding.replay_arena is not replay_arena:
        return _miss("binding_arena_mismatch")
    q_layout_key = _resolved_row_ptr_q_layout_key(
        step_authority=step_authority,
        batch_size=batch_size_i,
    )
    if q_layout_key is None or str(binding.descriptor.q_layout_key) != q_layout_key:
        return _miss("q_layout_key_mismatch")
    manager = getattr(controller, "_rrp_row_table_manager", None)
    if not isinstance(manager, RrpRowTableManager):
        return _miss("row_table_manager_missing")
    if not _same_page_rrp_signature_matches_step_authority(
        manager=manager,
        step_authority=step_authority,
        batch_size=batch_size_i,
    ):
        return _miss("row_table_signature_mismatch")
    visible = replay_arena.carriers.resolver_visible_seqused_k_by_head_i32
    if not isinstance(visible, torch.Tensor):
        return _miss("visible_tensor_missing")
    if not _resolved_row_ptr_current_visible_source_matches_binding(
        controller=controller,
        attn_metadata=attn_metadata,
        replay_arena=replay_arena,
        batch_size=batch_size_i,
        launch_plan=launch_plan,
    ):
        return _miss("visible_source_mismatch")

    if _z_probe:
        _z_t0 = time.perf_counter_ns()
    row_effective_k_by_row = tuple(
        int(v) for v in tuple(delta.row_effective_k_by_row)[:batch_size_i]
    )
    if len(row_effective_k_by_row) != batch_size_i:
        return _miss("row_effective_len_mismatch")
    update = manager.try_apply_same_page_delta(
        replay_arena,
        row_effective_k_by_row,
        recent_first_page_by_row=tuple(
            int(v) for v in tuple(delta.recent_first_page_by_row)[:batch_size_i]
        ),
    )
    if _z_probe:
        _z_t1 = time.perf_counter_ns()
        _z_hit_before_page_add = update is not None
    if (
        update is None
        and (
            os.environ.get("VLLM_SPARSE_PAGE_ADD_INCREMENTAL", "1") == "1"
            if _DYNAMIC_ENV
            else _PAGE_ADD_INCREMENTAL_CACHED
        )
    ):
        update = _try_page_add_incremental(
            controller,
            step_authority=step_authority,
            delta=delta,
            replay_arena=replay_arena,
            manager=manager,
            batch_size=batch_size_i,
            block_size=int(block_size),
            launch_plan=launch_plan,
            device=device,
        )
    if _z_probe and _z_t1:
        _z_t2 = time.perf_counter_ns()
        _append_metadata_timing(
            {
                "event": "z_rrp_probe",
                "mgr_apply_us": (_z_t1 - _z_t0) / 1000.0,
                "page_add_us": (_z_t2 - _z_t1) / 1000.0,
                "hit_before_page_add": _z_hit_before_page_add,
                "hit_after_page_add": update is not None,
            }
        )
    if update is None:
        return _miss("row_table_same_page_delta_update_miss")
    if (
        getattr(attn_metadata, "mixed_page_resolver_replay_metadata_binding", None)
        is not binding
    ):
        setattr(
            controller,
            "_decode_runtime_rrp_step_state_missing_metadata_view_count",
            int(
                getattr(
                    controller,
                    "_decode_runtime_rrp_step_state_missing_metadata_view_count",
                    0,
                )
            )
            + 1,
        )
    _stamp_resolved_row_ptr_step_state(
        controller=controller,
        step_authority=step_authority,
        live_batch_size=batch_size_i,
        effective_batch_size=batch_size_i,
    )
    setattr(controller, "_decode_runtime_rrp_step_state_miss_reason", "hit")
    return update


def _same_page_rrp_signature_matches_step_authority(
    *,
    manager: RrpRowTableManager,
    step_authority: object,
    batch_size: int,
    live_batch_size: int | None = None,
    active_arena_row_indices: tuple[int, ...] | None = None,
) -> bool:
    signature = getattr(manager, "_signature", None)
    if not isinstance(signature, tuple) or len(signature) < 2:
        return False
    batch_size_i = int(batch_size)
    live_batch_size_i = (
        batch_size_i if live_batch_size is None else int(live_batch_size)
    )
    if int(signature[0]) != batch_size_i:
        return False

    req_ids = getattr(step_authority, "req_ids", None)
    if req_ids is not None:
        if isinstance(req_ids, torch.Tensor) or len(req_ids) < live_batch_size_i:
            return False
        signature_req_ids = tuple(signature[1])
        if live_batch_size_i == batch_size_i:
            expected_req_ids = tuple(str(v) for v in tuple(req_ids)[:batch_size_i])
            actual_req_ids = tuple(
                str(v) for v in signature_req_ids[:batch_size_i]
            )
        else:
            if active_arena_row_indices is None:
                active_arena_row_indices = tuple(range(live_batch_size_i))
            if len(active_arena_row_indices) < live_batch_size_i:
                return False
            if len(signature_req_ids) < batch_size_i:
                return False
            expected_req_ids = tuple(
                str(v) for v in tuple(req_ids)[:live_batch_size_i]
            )
            try:
                actual_req_ids = tuple(
                    str(signature_req_ids[int(active_arena_row_indices[live_row])])
                    for live_row in range(live_batch_size_i)
                )
            except Exception:
                return False
        if expected_req_ids != actual_req_ids:
            return False
    return True


def _resolved_row_ptr_profile_max_seqlen_k(
    *,
    launch_plan: object | None,
    row_effective_k_by_row: object,
    batch_size: int,
    block_size: int,
    max_pages_per_row: int,
) -> int:
    capacity_tokens = max(1, int(block_size) * int(max_pages_per_row))
    plan_max = (
        getattr(launch_plan, "max_seqlen_k", None)
        if launch_plan is not None
        else None
    )
    if isinstance(plan_max, torch.Tensor):
        raise RuntimeError(
            "resolved-row-ptr profile max_seqlen_k requires CPU launch-plan metadata"
        )
    if isinstance(row_effective_k_by_row, torch.Tensor):
        raise RuntimeError(
            "resolved-row-ptr profile max_seqlen_k requires CPU row-effective metadata"
        )
    row_values = tuple(int(v) for v in tuple(row_effective_k_by_row)[: int(batch_size)])
    if len(row_values) < int(batch_size):
        raise RuntimeError(
            "resolved-row-ptr profile max_seqlen_k requires row-effective coverage"
        )
    row_max = max((max(0, int(v)) for v in row_values), default=1)
    profile_max = int(row_max)
    if profile_max > capacity_tokens:
        raise RuntimeError(
            "resolved-row-ptr profile max_seqlen_k exceeds carrier capacity; "
            f"profile_max={profile_max} capacity={capacity_tokens}"
        )
    # PHASE3: size the captured SplitKV scratch for the bucket CEILING so a replay
    # whose KV extent is larger (e.g. a demoted-dense row) but in the same bucket
    # can never under-size the frozen out_partial/num_splits. Capped at capacity.
    # PHASE3b: vLLM captures the FULL decode graph ONCE; its out_partial/num_splits/grid
    # are frozen at capture-time max_seqlen_k. A dynamically demoted-dense row at replay
    # can exceed any sparse/dummy bound, so size the captured SplitKV plan for the carrier
    # capacity (worst-case max_seqlen_k). num_splits caps at 128 so this is bounded; every
    # replay then fits the captured scratch. (Not a guard.)
    return max(1, int(capacity_tokens))


def _publish_resolved_row_ptr_profile_max_seqlen_k(
    *,
    attn_metadata: object,
    launch_plan: object | None,
    row_effective_k_by_row: object,
    batch_size: int,
    block_size: int,
    max_pages_per_row: int,
) -> int:
    profile_max = _resolved_row_ptr_profile_max_seqlen_k(
        launch_plan=launch_plan,
        row_effective_k_by_row=row_effective_k_by_row,
        batch_size=int(batch_size),
        block_size=int(block_size),
        max_pages_per_row=int(max_pages_per_row),
    )
    setattr(attn_metadata, "mixed_page_profile_max_seqlen_k", int(profile_max))
    return int(profile_max)


def _resolved_row_ptr_launch_effective_k_len_source(
    *,
    launch_plan: object | None,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor | None:
    source = (
        getattr(launch_plan, "launch_effective_k_len_i32", None)
        if launch_plan is not None
        else None
    )
    if not isinstance(source, torch.Tensor):
        return None
    if source.dtype != torch.int32:
        return None
    if source.device != device:
        return None
    if source.dim() != 1:
        return None
    if int(source.numel()) != int(batch_size):
        return None
    if not source.is_contiguous():
        return None
    return source


class _ResolvedRowPtrLaunchEffectiveView:
    __slots__ = ("valid", "launch_effective_k_len_i32", "launch_effective_k_len_cpu")

    def __init__(
        self,
        *,
        launch_effective_k_len_i32: torch.Tensor,
        launch_effective_k_len_cpu: tuple[int, ...],
    ) -> None:
        self.valid = True
        self.launch_effective_k_len_i32 = launch_effective_k_len_i32
        self.launch_effective_k_len_cpu = launch_effective_k_len_cpu


def _resolved_row_ptr_graph_launch_effective_view(
    *,
    controller: object,
    launch_plan: object | None,
    launch_effective_k_by_row: tuple[int, ...],
    batch_size: int,
    device: torch.device,
) -> object | None:
    source = _resolved_row_ptr_launch_effective_k_len_source(
        launch_plan=launch_plan,
        batch_size=int(batch_size),
        device=device,
    )
    if source is not None:
        return launch_plan
    if len(launch_effective_k_by_row) != int(batch_size):
        return launch_plan
    try:
        from patches.decode_runtime.compact_recent_launch_plan_builder import (
            ensure_cached_launch_effective_k_len_i32,
        )
    except Exception:
        return launch_plan
    tensor = ensure_cached_launch_effective_k_len_i32(
        controller,
        batch_size=int(batch_size),
        device=device,
    )
    if not (
        isinstance(tensor, torch.Tensor)
        and tensor.dtype == torch.int32
        and tensor.device == device
        and tensor.dim() == 1
        and int(tensor.numel()) == int(batch_size)
        and tensor.is_contiguous()
    ):
        return launch_plan
    cpu_buffer = getattr(controller, "_rrp_launch_effective_cpu_staging_i32", None)
    if not isinstance(cpu_buffer, torch.Tensor) or int(cpu_buffer.numel()) < int(batch_size):
        cpu_buffer = torch.empty((int(batch_size),), dtype=torch.int32, device="cpu")
        setattr(controller, "_rrp_launch_effective_cpu_staging_i32", cpu_buffer)
    cpu_buffer[: int(batch_size)].copy_(
        torch.as_tensor(
            launch_effective_k_by_row[: int(batch_size)],
            dtype=torch.int32,
            device="cpu",
        )
    )
    tensor.copy_(cpu_buffer[: int(batch_size)], non_blocking=device.type == "cuda")
    values = tuple(int(v) for v in launch_effective_k_by_row[: int(batch_size)])
    view = getattr(controller, "_rrp_graph_launch_effective_view", None)
    if not isinstance(view, _ResolvedRowPtrLaunchEffectiveView):
        view = _ResolvedRowPtrLaunchEffectiveView(
            launch_effective_k_len_i32=tensor,
            launch_effective_k_len_cpu=values,
        )
        setattr(controller, "_rrp_graph_launch_effective_view", view)
    else:
        view.launch_effective_k_len_i32 = tensor
        view.launch_effective_k_len_cpu = values
    return view


def _resolved_row_ptr_launch_effective_covers_rows(
    *,
    launch_plan: object | None,
    row_effective_k_by_row: tuple[int, ...],
    batch_size: int,
) -> bool:
    values = (
        getattr(launch_plan, "launch_effective_k_len_cpu", None)
        if launch_plan is not None
        else None
    )
    if values is None:
        return False
    if isinstance(values, torch.Tensor):
        if values.device.type != "cpu":
            return False
        try:
            values_tuple = tuple(int(v) for v in values[: int(batch_size)].tolist())
        except Exception:
            return False
    else:
        try:
            values_tuple = tuple(int(v) for v in tuple(values)[: int(batch_size)])
        except Exception:
            return False
    if len(values_tuple) < int(batch_size):
        return False
    return all(
        int(values_tuple[row]) >= int(row_effective_k_by_row[row])
        for row in range(int(batch_size))
    )


def _resolved_row_ptr_dense_seqused_k_source(
    *,
    step_meta: object,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor | None:
    source = getattr(step_meta, "seqused_k_gpu", None)
    if not isinstance(source, torch.Tensor):
        source = getattr(step_meta, "canonical_real_kv_len_i32_gpu", None)
    if not isinstance(source, torch.Tensor):
        return None
    if source.dtype != torch.int32:
        return None
    if source.device != device:
        return None
    if source.dim() != 1:
        return None
    if int(source.numel()) < int(batch_size):
        return None
    visible = source[: int(batch_size)]
    if not visible.is_contiguous():
        return None
    return visible


def _resolved_row_ptr_current_visible_source_matches_binding(
    *,
    controller: object,
    attn_metadata: object,
    replay_arena: ResolvedRowPtrArena,
    batch_size: int,
    launch_plan: object | None = None,
) -> bool:
    visible = replay_arena.carriers.resolver_visible_seqused_k_by_head_i32
    if not isinstance(visible, torch.Tensor):
        return False
    if int(visible.data_ptr()) == int(replay_arena.seqused_k_i32.data_ptr()):
        return True
    arena_batch = getattr(replay_arena, "batch_seqused_k_i32", None)
    if isinstance(arena_batch, torch.Tensor) and int(visible.data_ptr()) == int(
        arena_batch.data_ptr()
    ):
        return True
    step_meta = getattr(controller, "step_meta", None)
    if step_meta is not None:
        dense = _resolved_row_ptr_dense_seqused_k_source(
            step_meta=step_meta,
            batch_size=int(batch_size),
            device=visible.device,
        )
        if dense is not None and int(visible.data_ptr()) == int(dense.data_ptr()):
            return True
    del attn_metadata, launch_plan
    return False


def _rrp_visible_source_debug_fields(
    *,
    controller: object,
    replay_arena: ResolvedRowPtrArena,
    launch_plan: object | None,
    row_effective_k_by_row: tuple[int, ...],
    batch_size: int,
    device: torch.device,
    sparse_dynamic_state: object | None = None,
) -> dict[str, object]:
    def _data_ptr(tensor: object) -> int:
        return int(tensor.data_ptr()) if isinstance(tensor, torch.Tensor) else 0

    def _shape(tensor: object) -> list[int]:
        if not isinstance(tensor, torch.Tensor):
            return []
        return [int(v) for v in tuple(tensor.shape)]

    batch_size_i = int(batch_size)
    visible = replay_arena.carriers.resolver_visible_seqused_k_by_head_i32
    arena_seqused = replay_arena.seqused_k_i32
    arena_batch = getattr(replay_arena, "batch_seqused_k_i32", None)
    visible_device = visible.device if isinstance(visible, torch.Tensor) else device
    launch_effective = _resolved_row_ptr_launch_effective_k_len_source(
        launch_plan=launch_plan,
        batch_size=batch_size_i,
        device=visible_device,
    )
    sparse_dynamic_tensor = None
    if sparse_dynamic_state is not None:
        getter = getattr(sparse_dynamic_state, "visible_effective_k_tensor", None)
        if callable(getter):
            try:
                candidate = getter()
            except RuntimeError:
                candidate = None
            if isinstance(candidate, torch.Tensor):
                sparse_dynamic_tensor = candidate

    dense = None
    step_meta = getattr(controller, "step_meta", None)
    if step_meta is not None and isinstance(visible_device, torch.device):
        dense = _resolved_row_ptr_dense_seqused_k_source(
            step_meta=step_meta,
            batch_size=batch_size_i,
            device=visible_device,
        )

    visible_ptr = _data_ptr(visible)
    arena_ptr = _data_ptr(arena_seqused)
    arena_batch_ptr = _data_ptr(arena_batch)
    launch_ptr = _data_ptr(launch_effective)
    sparse_dynamic_ptr = _data_ptr(sparse_dynamic_tensor)
    dense_ptr = _data_ptr(dense)
    sparse_dynamic_failure_reason = str(
        getattr(sparse_dynamic_state, "last_failure_reason", "") or ""
    )
    bound_source_kind = str(
        getattr(replay_arena, "_resolved_seqused_source_kind", "") or ""
    )
    visible_is_sparse_dynamic = bool(
        visible_ptr != 0 and visible_ptr == sparse_dynamic_ptr
    )
    visible_is_arena = bool(visible_ptr != 0 and visible_ptr == arena_ptr)
    visible_is_arena_batch = bool(visible_ptr != 0 and visible_ptr == arena_batch_ptr)
    visible_is_launch = bool(visible_ptr != 0 and visible_ptr == launch_ptr)
    visible_is_dense = bool(visible_ptr != 0 and visible_ptr == dense_ptr)
    if not isinstance(visible, torch.Tensor):
        source_kind = "missing"
    elif visible_is_sparse_dynamic:
        source_kind = "sparse_dynamic_state"
    elif visible_is_arena:
        source_kind = "arena_seqused"
    elif visible_is_arena_batch:
        source_kind = "arena_batch_seqused"
    elif visible_is_launch:
        source_kind = "launch_effective"
    elif visible_is_dense:
        source_kind = "dense_seqused"
    else:
        source_kind = "other"

    launch_cpu_raw = (
        getattr(launch_plan, "launch_effective_k_len_cpu", None)
        if launch_plan is not None
        else None
    )
    try:
        launch_cpu = tuple(int(v) for v in tuple(launch_cpu_raw)[:batch_size_i])
    except Exception:
        launch_cpu = tuple()
    row_effective = tuple(int(v) for v in tuple(row_effective_k_by_row)[:batch_size_i])
    if len(row_effective) < batch_size_i:
        row_effective = row_effective + (0,) * (batch_size_i - len(row_effective))

    return {
        "rrp_visible_source_kind": source_kind,
        "rrp_visible_shape": _shape(visible),
        "rrp_visible_data_ptr": int(visible_ptr),
        "rrp_visible_is_arena_seqused": bool(visible_is_arena),
        "rrp_visible_is_arena_batch_seqused": bool(visible_is_arena_batch),
        "rrp_visible_is_launch_effective": bool(visible_is_launch),
        "rrp_visible_is_dense_seqused": bool(visible_is_dense),
        "rrp_arena_seqused_shape": _shape(arena_seqused),
        "rrp_arena_batch_seqused_shape": _shape(arena_batch),
        "rrp_launch_effective_shape": _shape(launch_effective),
        "rrp_launch_effective_data_ptr": int(launch_ptr),
        "rrp_sparse_dynamic_state_shape": _shape(sparse_dynamic_tensor),
        "rrp_sparse_dynamic_state_data_ptr": int(sparse_dynamic_ptr),
        "rrp_sparse_dynamic_state_covers_rows": bool(
            _resolved_row_ptr_visible_source_covers_arena(
                sparse_dynamic_tensor,
                replay_arena,
            )
        ),
        "rrp_sparse_dynamic_state_failure_reason": sparse_dynamic_failure_reason,
        "rrp_launch_effective_covers_rows": bool(
            _resolved_row_ptr_launch_effective_covers_rows(
                launch_plan=launch_plan,
                row_effective_k_by_row=row_effective,
                batch_size=batch_size_i,
            )
        ),
        "rrp_launch_effective_k_len_cpu": [int(v) for v in launch_cpu],
        "rrp_row_effective_k_by_row": [int(v) for v in row_effective],
    }



_RRP_VISIBLE_SOURCE_METADATA_ATTRS = {
    "rrp_visible_source_kind": "_sparse_rrp_visible_source_kind",
    "rrp_sparse_dynamic_state_covers_rows": "_sparse_dynamic_state_covers_rows",
    "rrp_sparse_dynamic_state_failure_reason": "_sparse_dynamic_state_failure_reason",
    "rrp_visible_data_ptr": "_sparse_rrp_visible_data_ptr",
    "rrp_sparse_dynamic_state_data_ptr": "_sparse_dynamic_state_data_ptr",
    "rrp_visible_shape": "_sparse_rrp_visible_shape",
    "rrp_visible_is_arena_seqused": "_sparse_rrp_visible_is_arena_seqused",
    "rrp_visible_is_arena_batch_seqused": "_sparse_rrp_visible_is_arena_batch_seqused",
    "rrp_visible_is_launch_effective": "_sparse_rrp_visible_is_launch_effective",
    "rrp_visible_is_dense_seqused": "_sparse_rrp_visible_is_dense_seqused",
    "rrp_launch_effective_covers_rows": "_sparse_rrp_launch_effective_covers_rows",
    "rrp_launch_effective_k_len_cpu": "_sparse_rrp_launch_effective_k_len_cpu",
    "rrp_row_effective_k_by_row": "_sparse_rrp_row_effective_k_by_row",
}


def _publish_rrp_visible_source_debug_attrs(
    attn_metadata: object,
    fields: dict[str, object],
) -> None:
    for field_name, attr_name in _RRP_VISIBLE_SOURCE_METADATA_ATTRS.items():
        if field_name in fields:
            setattr(attn_metadata, attr_name, fields[field_name])

def _resolved_row_ptr_visible_source_covers_arena(
    visible: object,
    arena: ResolvedRowPtrArena,
) -> bool:
    if not isinstance(visible, torch.Tensor):
        return False
    if visible.dtype != torch.int32:
        return False
    if visible.device != arena.seqused_k_i32.device:
        return False
    if visible.dim() != 1 or not visible.is_contiguous():
        return False
    visible_count = int(visible.numel())
    return visible_count in (int(arena.batch_size), int(arena.total_rows))


def _bind_resolved_row_ptr_visible_seqused_source(
    *,
    replay_arena: ResolvedRowPtrArena,
    source_arena: ResolvedRowPtrArena,
    launch_plan: object | None,
    step_meta: object,
    batch_size: int,
    device: torch.device,
    sparse_dynamic_state: object | None = None,
    row_effective_k_by_row: tuple[int, ...] | None = None,
    launch_effective_k_by_row: object | None = None,
    step_authority: object | None = None,
    capacity_key: object | None = None,
    active_row_indices: tuple[int, ...] | None = None,
    slot_mapping_signature: tuple[int, ...] | None = None,
) -> bool:
    batch_size_i = int(batch_size)

    def _bind_arena_batch_source() -> bool:
        replay_batch = getattr(replay_arena, "batch_seqused_k_i32", None)
        source_batch = getattr(source_arena, "batch_seqused_k_i32", None)
        if not (
            isinstance(replay_batch, torch.Tensor)
            and isinstance(source_batch, torch.Tensor)
            and _resolved_row_ptr_visible_source_covers_arena(
                replay_batch,
                replay_arena,
            )
            and _resolved_row_ptr_visible_source_covers_arena(
                source_batch,
                source_arena,
            )
        ):
            return False
        replay_arena.bind_resolved_seqused_source(
            replay_batch,
            source_kind="arena_batch_seqused",
        )
        source_arena.bind_resolved_seqused_source(
            source_batch,
            source_kind="arena_batch_seqused",
        )
        return True

    del launch_plan, step_meta, device
    return bool(_bind_arena_batch_source())









def _same_page_minimal_update_enabled() -> bool:
    if _DYNAMIC_ENV:
        return os.environ.get("VLLM_SPARSE_SAME_PAGE_MINIMAL_UPDATE", "1") == "1"
    return _SAME_PAGE_MINIMAL_UPDATE_CACHED


def _same_page_ready_event_only_publish_enabled() -> bool:
    if _DYNAMIC_ENV:
        return os.environ.get("VLLM_SPARSE_SAME_PAGE_READY_EVENT_ONLY", "1") == "1"
    return _SAME_PAGE_READY_EVENT_ONLY_CACHED


def _update_resolved_row_ptr_graph_binding_ready_event_only(
    controller: object,
    *,
    arena_key: tuple[object, ...],
    ready_event_generation: int,
    ready_event: object | None,
    ready_event_stream: int,
) -> bool:
    """Same-page steady step: the binding topology is invariant (static cache
    key proved unchanged upstream), so only the ready-event handle/generation
    advanced. Update the already-published dict's 3 event fields in place and
    skip the full _publish rebuild. Stricter than the cheap branch of
    _refresh_resolved_row_ptr_graph_binding_ready_state: a missing/foreign/
    uncovered state dict returns False so the caller keeps the original
    _refresh/_publish path."""
    state = getattr(controller, "_resolved_row_ptr_graph_binding_state", None)
    if not isinstance(state, dict):
        return False
    if state.get("route_family") != "resolved_row_ptr":
        return False
    if tuple(state.get("arena_key", ())) != tuple(arena_key):
        return False
    if int(state.get("metadata_count", 0)) <= 0:
        return False
    state["ready_event_generation"] = int(ready_event_generation)
    state["ready_event"] = ready_event
    state["ready_event_stream"] = int(ready_event_stream)
    return True


def _same_page_minimal_assert_enabled() -> bool:
    if _DYNAMIC_ENV:
        return os.environ.get("VLLM_SPARSE_SAME_PAGE_MINIMAL_ASSERT") == "1"
    return _SAME_PAGE_MINIMAL_ASSERT_CACHED


def _same_page_minimal_reject_reason(
    *,
    runtime_state: DecodeRuntimeState,
    predicted_delta: DecodeDeltaPacket | None,
    previous_step_bound_meta: StepBoundMeta | None,
    step_authority: object,
    template: LaunchTemplate,
    batch_size: int,
    prefill_rows: Tuple[int, ...],
    layer_effective_refresh_by_row: Tuple[bool, ...],
) -> str:
    if predicted_delta is None:
        return "predicted_delta_missing"
    if not isinstance(runtime_state.active_guard, DecodeStaticGuard):
        return "active_guard_missing"
    previous_delta = runtime_state.last_delta
    if not isinstance(previous_delta, DecodeDeltaPacket):
        return "previous_delta_missing"
    if previous_step_bound_meta is None:
        return "previous_step_bound_meta_missing"
    if prefill_rows:
        return "prefill_rows_present"
    if any(bool(v) for v in layer_effective_refresh_by_row[: int(batch_size)]):
        return "layer_effective_refresh_present"
    if any(int(v) != 0 for v in step_authority.logf_mask_by_row[: int(batch_size)]):
        return "logf_mask_present"
    if predicted_delta.pending_refresh_state not in ("nil", "empty"):
        return "pending_refresh_not_steady"
    if predicted_delta.recent_first_page_by_row != previous_delta.recent_first_page_by_row:
        return "recent_first_changed"
    if predicted_delta.recent_page_count_by_row != previous_delta.recent_page_count_by_row:
        return "recent_count_changed"
    if predicted_delta.q_lens_by_row != previous_delta.q_lens_by_row:
        return "q_layout_changed"
    if predicted_delta.logf_mask_generation != previous_delta.logf_mask_generation:
        return "logf_generation_changed"
    if max(predicted_delta.launch_effective_k_by_row, default=0) > int(
        template.max_seqlen_k_capacity
    ):
        return "max_seqlen_k_capacity_exceeded"
    return ""


def _assert_same_page_minimal_matches_full(
    *,
    runtime_state: DecodeRuntimeState,
    predicted_delta: DecodeDeltaPacket | None,
    previous_step_bound_meta: StepBoundMeta | None,
    step_authority: object,
    template: LaunchTemplate,
    batch_size: int,
    prefill_rows: Tuple[int, ...],
    layer_effective_refresh_by_row: Tuple[bool, ...],
) -> None:
    reason = _same_page_minimal_reject_reason(
        runtime_state=runtime_state,
        predicted_delta=predicted_delta,
        previous_step_bound_meta=previous_step_bound_meta,
        step_authority=step_authority,
        template=template,
        batch_size=batch_size,
        prefill_rows=prefill_rows,
        layer_effective_refresh_by_row=layer_effective_refresh_by_row,
    )
    if reason:
        return
    assert predicted_delta is not None
    previous_delta = runtime_state.last_delta
    assert isinstance(previous_delta, DecodeDeltaPacket)
    assert predicted_delta.recent_first_page_by_row == previous_delta.recent_first_page_by_row
    assert predicted_delta.recent_page_count_by_row == previous_delta.recent_page_count_by_row
    assert predicted_delta.request_recent_len_by_row != previous_delta.request_recent_len_by_row
    assert predicted_delta.launch_effective_k_by_row != previous_delta.launch_effective_k_by_row


def _try_apply_same_page_minimal_metadata_update(
    controller: object,
    *,
    attn_metadata: object,
    step_meta: object,
    step_authority: object,
    previous_step_bound_meta: StepBoundMeta | None,
    block_size: int,
    num_kv_heads: int,
    q_start_loc: Tuple[int, ...],
    layer_effective_refresh_by_row: Tuple[bool, ...],
    template: LaunchTemplate,
    launch_plan: object,
    runtime_state: DecodeRuntimeState,
    predicted_delta: DecodeDeltaPacket | None,
    predicted_reason: str,
    batch_size: int,
    prefill_rows: Tuple[int, ...],
    mb_profile_enabled: bool,
    steady_phase_enabled: bool,
    mb_total_start_ns: int,
    mb_phase_us: dict[str, float],
) -> bool:
    if not _same_page_minimal_update_enabled():
        return False
    reject_reason = _same_page_minimal_reject_reason(
        runtime_state=runtime_state,
        predicted_delta=predicted_delta,
        previous_step_bound_meta=previous_step_bound_meta,
        step_authority=step_authority,
        template=template,
        batch_size=batch_size,
        prefill_rows=prefill_rows,
        layer_effective_refresh_by_row=layer_effective_refresh_by_row,
    )
    if reject_reason:
        controller._decode_runtime_same_page_minimal_miss_reason = reject_reason
        return False
    assert predicted_delta is not None
    assert previous_step_bound_meta is not None
    guard = runtime_state.active_guard
    if not isinstance(guard, DecodeStaticGuard):
        controller._decode_runtime_same_page_minimal_miss_reason = "active_guard_missing"
        return False
    mode, reason = classify_decode_runtime_mode(
        guard,
        runtime_state.last_delta,
        guard,
        predicted_delta,
    )
    if mode is not DecodeRuntimeMode.STEADY_DELTA:
        controller._decode_runtime_same_page_minimal_miss_reason = (
            "classification_not_steady:" + str(reason)
        )
        return False

    phase_start_ns = time.perf_counter_ns()

    def _mark(name: str) -> None:
        nonlocal phase_start_ns
        if not (mb_profile_enabled or steady_phase_enabled):
            return
        now_ns = time.perf_counter_ns()
        mb_phase_us[name] = float(now_ns - phase_start_ns) / 1000.0
        phase_start_ns = now_ns

    replay_bind_device = (
        step_meta.seqused_k_gpu.device
        if isinstance(getattr(step_meta, "seqused_k_gpu", None), torch.Tensor)
        else next(iter(controller.layer_states.values())).device
    )
    replay_bind_device_t = torch.device(replay_bind_device)
    replay_arena_for_visible = getattr(controller, "_resolved_row_ptr_replay_arena", None)
    launch_effective_source = _resolved_row_ptr_launch_effective_k_len_source(
        launch_plan=launch_plan,
        batch_size=batch_size,
        device=replay_bind_device_t,
    )
    visible_source = (
        replay_arena_for_visible.carriers.resolver_visible_seqused_k_by_head_i32
        if isinstance(replay_arena_for_visible, ResolvedRowPtrArena)
        else None
    )
    visible_uses_launch_effective = (
        isinstance(visible_source, torch.Tensor)
        and isinstance(launch_effective_source, torch.Tensor)
        and int(visible_source.data_ptr()) == int(launch_effective_source.data_ptr())
    )
    _mark("same_page_minimal_device_source")

    old_mode = getattr(controller, "_decode_runtime_mode", None)
    old_delta = getattr(controller, "_decode_runtime_delta", None)
    old_reason = getattr(controller, "_decode_runtime_reason", None)
    controller._decode_runtime_mode = mode
    controller._decode_runtime_delta = predicted_delta
    controller._decode_runtime_reason = reason
    controller._decode_runtime_same_page_minimal_miss_reason = ""
    try:
        update = apply_launch_template_row_delta(
            template,
            request_recent_len_by_row=predicted_delta.request_recent_len_by_row,
            launch_effective_k_by_row=predicted_delta.launch_effective_k_by_row,
            recent_first_page_by_row=predicted_delta.recent_first_page_by_row,
            recent_page_count_by_row=predicted_delta.recent_page_count_by_row,
            update_gpu=bool(visible_uses_launch_effective),
            force_gpu_refresh=bool(visible_uses_launch_effective),
            # [ARM-WAR-R1-PINNED-INDEPENDENT 2026-07-12] fresh 镜像双引用替换。
            controller=controller,
        )
        if bool(update.requires_recompile):
            controller._decode_runtime_same_page_minimal_miss_reason = (
                "launch_template_delta_requires_recompile:" + str(update.reason)
            )
            return False
        _mark("same_page_minimal_template_apply")
        rrp_update = _try_update_same_page_resolved_row_ptr_step_state(
            controller,
            attn_metadata=attn_metadata,
            step_authority=step_authority,
            batch_size=batch_size,
            block_size=int(block_size),
            num_kv_heads=int(num_kv_heads),
            device=replay_bind_device_t,
            launch_plan=launch_plan,
            same_page_proven=True,
        )
        if rrp_update is None:
            controller._decode_runtime_same_page_minimal_miss_reason = (
                "delta_rrp_step_state_update_miss"
            )
            return False
        _mark("same_page_minimal_rrp_call")
    finally:
        if getattr(controller, "_decode_runtime_same_page_minimal_miss_reason", ""):
            controller._decode_runtime_mode = old_mode
            controller._decode_runtime_delta = old_delta
            controller._decode_runtime_reason = old_reason

    runtime_state.active_guard = guard
    runtime_state.last_delta = predicted_delta
    runtime_state.last_mode = mode
    runtime_state.last_reason = reason
    runtime_state.last_applied_step_id = predicted_delta.step_id
    runtime_state.counters.same_page_step_count += 1
    runtime_state.counters.predicted_same_page_step_count += 1
    controller._decode_runtime_predicted_same_page_delta_count = (
        int(getattr(controller, "_decode_runtime_predicted_same_page_delta_count", 0))
        + 1
    )
    controller._decode_runtime_predicted_same_page_reason = str(predicted_reason)
    controller._decode_runtime_template_update_reason = update.reason
    controller._decode_runtime_same_page_minimal_miss_reason = "hit"

    _refresh_step_bound_meta_for_steady_delta(
        step_bound_meta=previous_step_bound_meta,
        step_meta=step_meta,
        step_authority=step_authority,
        delta=predicted_delta,
        launch_plan=launch_plan,
        q_start_loc=q_start_loc,
        prefill_rows=prefill_rows,
    )
    controller.step_bound_meta = previous_step_bound_meta
    # [T4-FORENSIC 2026-07-10] 拆开旧 ready_event 相位:SBM 刷新与 event 机器分账。
    _mark("same_page_minimal_sbm_refresh")

    same_page_update_kernel_count = int(update.carrier_update_kernel_count) + int(
        getattr(rrp_update, "update_kernel_count", 0)
    )
    runtime_state.counters.carrier_update_kernel_count += int(
        same_page_update_kernel_count
    )
    ready_event_recorded = _record_resolved_row_ptr_owner_update_ready_event(
        controller,
        attn_metadata=attn_metadata,
        device=replay_bind_device_t,
        update_kernel_count=int(same_page_update_kernel_count),
        profile_phase_us=(
            mb_phase_us if (mb_profile_enabled or steady_phase_enabled) else None
        ),
    )
    _mark("same_page_minimal_ready_event")

    if controller.step_decode_data is not None:
        if step_meta.seqused_k_gpu is not None:
            controller.step_decode_data.seqused_k = step_meta.seqused_k_gpu
        controller.step_decode_data.epoch = int(step_authority.epoch)
        controller.step_decode_data.decode_plan_version = int(
            getattr(step_authority, "decode_plan_version", -1)
        )
    controller._resolved_row_ptr_same_page_metadata_bind_skipped_count = (
        int(getattr(controller, "_resolved_row_ptr_same_page_metadata_bind_skipped_count", 0))
        + 1
    )
    controller._decode_runtime_launch_template_update_count = (
        int(getattr(controller, "_decode_runtime_launch_template_update_count", 0))
        + 1
    )
    setattr(controller, "_decode_runtime_steady_fast_path_miss_reason", "hit")
    _mark("same_page_minimal_publish")

    if mb_profile_enabled:
        _append_mb_profile(
            {
                "event": "metadata_builder_step",
                "epoch": int(getattr(step_authority, "epoch", -1)),
                "batch_size": batch_size,
                "num_layers": int(len(getattr(controller, "layer_cache_keys", ()))),
                "decode_only": True,
                "has_decode_row": True,
                "reuse_decode_data": True,
                "full_decode_reuse_hit": True,
                "same_page_minimal_update": True,
                "decode_runtime_mode": mode.value,
                "decode_runtime_reason": str(reason),
                "runtime_classification_current": True,
                "same_page_step_count": 1,
                "page_boundary_step_count": 0,
                "refresh_commit_step_count": 0,
                "carrier_update_kernel_count": int(same_page_update_kernel_count),
                "rrp_ready_event_recorded": bool(ready_event_recorded),
                "predicted_same_page_step_count": 1,
                "phase_us": dict(mb_phase_us),
                "rrp_update_kind": str(getattr(rrp_update, "kind", "")),
                "total_us": float(time.perf_counter_ns() - mb_total_start_ns)
                / 1000.0,
            }
        )
    return True

def _try_run_steady_decode_metadata_fast_path_inner(
    controller: object,
    *,
    attn_metadata: object,
    step_meta: object,
    step_authority: object,
    previous_step_bound_meta: StepBoundMeta | None,
    kv_cache_spec: Optional[object],
    block_size: int,
    num_kv_heads: int,
    q_start_loc: Tuple[int, ...],
    layer_effective_refresh_by_row: Tuple[bool, ...],
    mb_profile_enabled: bool,
    steady_phase_enabled: bool,
    mb_total_start_ns: int,
    mb_phase_us: dict[str, float],
) -> bool:
    del kv_cache_spec
    setattr(controller, "_decode_runtime_steady_fast_path_miss_reason", "")

    def _miss(reason: str) -> bool:
        setattr(controller, "_decode_runtime_steady_fast_path_miss_reason", str(reason))
        return False

    if previous_step_bound_meta is None:
        return _miss("previous_step_bound_meta_missing")
    if not bool(getattr(step_authority, "is_decode_only", False)):
        return _miss("not_decode_only")
    batch_size = int(step_authority.batch_size)
    prefill_rows = tuple(
        int(row)
        for row in getattr(step_authority, "prefill_rows", tuple())
        if 0 <= int(row) < batch_size
    )
    if prefill_rows:
        return _miss("prefill_rows_present")
    if any(bool(v) for v in layer_effective_refresh_by_row[:batch_size]):
        # [LITE-P0 快照 2026-07-11] 触发步在此早退,不进 classify——runtime
        # state 的 last_delta 此刻仍是"触发前最后一个稳态 delta"(下一站慢路径
        # :6242 update_classification 会用触发步 delta 覆盖它)。这是全程唯一
        # 天然截存窗口:返程步的 SIG_RETURN admit 以此快照为签名基准(q/logf
        # 逐位翻回比对)。纯引用赋值零成本零行为;快照随每个触发步刷新,批组成
        # 变化(抢占/驱逐)时 last_delta 的 batch_size 自带甄别。
        _lite_state = getattr(controller, "_decode_runtime_state", None)
        if isinstance(_lite_state, DecodeRuntimeState):
            _lite_last = _lite_state.last_delta
            if _lite_last is not None and _lite_state.last_mode in (
                DecodeRuntimeMode.STEADY_DELTA,
                DecodeRuntimeMode.PAGE_BOUNDARY_DELTA,
                DecodeRuntimeMode.SIG_RETURN_DELTA,
            ):
                controller._lite_pre_trigger_steady_delta = _lite_last
        return _miss("layer_effective_refresh_present")
    if any(int(v) != 0 for v in step_authority.logf_mask_by_row[:batch_size]):
        return _miss("logf_mask_present")
    if not any(
        int(value) == int(_ROW_MODE_COMPACT)
        for value in step_authority.row_mode_by_row[:batch_size]
    ):
        return _miss("no_compact_row")
    template = getattr(controller, "_compact_recent_launch_template", None)
    if not isinstance(template, LaunchTemplate):
        return _miss("launch_template_missing")
    launch_plan = template.plan
    if launch_plan is None or not bool(getattr(launch_plan, "valid", False)):
        return _miss("launch_template_plan_invalid")
    steady_phase_start_ns = (
        time.perf_counter_ns()
        if (mb_profile_enabled or steady_phase_enabled)
        else 0
    )

    def _mark_steady_phase(name: str) -> None:
        nonlocal steady_phase_start_ns
        if not (mb_profile_enabled or steady_phase_enabled):
            return
        now_ns = time.perf_counter_ns()
        mb_phase_us[name] = float(now_ns - steady_phase_start_ns) / 1000.0
        steady_phase_start_ns = now_ns
    runtime_state = getattr(controller, "_decode_runtime_state", None)
    if not isinstance(runtime_state, DecodeRuntimeState):
        return _miss("decode_runtime_state_missing")
    try:
        predicted_delta, predicted_reason = _try_predict_same_page_delta_packet(
            runtime_state=runtime_state,
            step_authority=step_authority,
            page_size=int(block_size),
            launch_plan=launch_plan,
        )
        if _same_page_minimal_assert_enabled():
            _assert_same_page_minimal_matches_full(
                runtime_state=runtime_state,
                predicted_delta=predicted_delta,
                previous_step_bound_meta=previous_step_bound_meta,
                step_authority=step_authority,
                template=template,
                batch_size=batch_size,
                prefill_rows=prefill_rows,
                layer_effective_refresh_by_row=layer_effective_refresh_by_row,
            )
        if _try_apply_same_page_minimal_metadata_update(
            controller,
            attn_metadata=attn_metadata,
            step_meta=step_meta,
            step_authority=step_authority,
            previous_step_bound_meta=previous_step_bound_meta,
            block_size=int(block_size),
            num_kv_heads=int(num_kv_heads),
            q_start_loc=q_start_loc,
            layer_effective_refresh_by_row=layer_effective_refresh_by_row,
            template=template,
            launch_plan=launch_plan,
            runtime_state=runtime_state,
            predicted_delta=predicted_delta,
            predicted_reason=predicted_reason,
            batch_size=batch_size,
            prefill_rows=prefill_rows,
            mb_profile_enabled=mb_profile_enabled,
            steady_phase_enabled=steady_phase_enabled,
            mb_total_start_ns=mb_total_start_ns,
            mb_phase_us=mb_phase_us,
        ):
            return True
        guard = _build_decode_static_guard_from_launch_template(
            controller=controller,
            attn_metadata=attn_metadata,
            step_authority=step_authority,
            launch_template=template,
            block_size=int(block_size),
            num_kv_heads=int(num_kv_heads),
        )
        if predicted_delta is not None:
            delta = predicted_delta
        else:
            delta = _collect_decode_delta_packet_from_launch_plan(
                step_authority=step_authority,
                launch_plan=launch_plan,
                page_size=int(block_size),
            )
    except Exception:
        return _miss("guard_or_delta_build_exception")
    _mark_steady_phase("steady_guard_and_delta")

    mode, reason = classify_decode_runtime_mode(
        runtime_state.active_guard,
        runtime_state.last_delta,
        guard,
        delta,
    )
    if _should_build_compact_recent_launch_plan(mode, launch_template_ready=True):
        return _miss(f"classification_requires_plan_build:{mode.value}")
    if max(delta.launch_effective_k_by_row, default=0) > int(
        template.max_seqlen_k_capacity
    ):
        _reset_decode_runtime_full_recompile(
            controller,
            "launch_template_delta_requires_recompile:max_seqlen_k_capacity_exceeded",
        )
        return _miss("launch_template_delta_requires_recompile:max_seqlen_k_capacity_exceeded")
    mode, reason = runtime_state.update_classification(guard, delta)
    _mark_steady_phase("steady_classification")
    predicted_same_page = predicted_delta is not None and mode is DecodeRuntimeMode.STEADY_DELTA
    if predicted_same_page:
        runtime_state.counters.predicted_same_page_step_count += 1
        controller._decode_runtime_predicted_same_page_delta_count = (
            int(getattr(controller, "_decode_runtime_predicted_same_page_delta_count", 0))
            + 1
        )
        controller._decode_runtime_predicted_same_page_reason = str(predicted_reason)
    controller._decode_runtime_mode = mode
    controller._decode_runtime_delta = delta
    controller._decode_runtime_reason = reason
    # #12 ultra fail-fast admission gate: the two TERMINAL rrp-gate miss
    # reasons are fully decidable right here (classification counters above
    # already advanced; mode/delta/reason already published for the slow
    # path, exactly as on the late-miss path). Predicting one skips the
    # doomed heavy steady work below and falls to the slow path immediately
    # with the SAME outer miss reason and the SAME granular rrp reason the
    # late gate would have recorded. None => admitted; the rrp gate below
    # stays the sole authority for every non-terminal miss.
    early_rrp_miss = _steady_fast_path_terminal_rrp_miss(
        controller,
        mode=mode,
        delta=delta,
        batch_size=batch_size,
    )
    if early_rrp_miss is not None:
        setattr(
            controller,
            "_decode_runtime_rrp_step_state_miss_reason",
            str(early_rrp_miss),
        )
        controller._decode_runtime_speculative_delta_miss_reason = (
            "delta_rrp_step_state_update_miss"
        )
        if os.environ.get("VLLM_ABLATE_FORCE_FASTPATH") != "1":
            return _miss("delta_rrp_step_state_update_miss")
    _refresh_step_bound_meta_for_steady_delta(
        step_bound_meta=previous_step_bound_meta,
        step_meta=step_meta,
        step_authority=step_authority,
        delta=delta,
        launch_plan=launch_plan,
        q_start_loc=q_start_loc,
        prefill_rows=prefill_rows,
    )
    controller.step_bound_meta = previous_step_bound_meta

    compact_layout_generation = int(getattr(launch_plan, "compact_meta_epoch", -1))
    step_decode_cache_key = _make_step_decode_cache_key(
        step_authority,
        layer_cache_keys=controller.layer_cache_keys,
        layer_effective_refresh_by_row=layer_effective_refresh_by_row,
        compact_layout_generation=compact_layout_generation,
        refresh_signature_override=ALL_FALSE_SIGNATURE,
    )
    controller.step_decode_cache_key = step_decode_cache_key
    controller.step_decode_plan_version = int(
        getattr(step_authority, "decode_plan_version", -1)
    )
    if not _can_skip_layer_state_refresh_for_step(
        controller,
        step_authority=step_authority,
        step_decode_cache_key=step_decode_cache_key,
    ):
        return _miss("layer_state_refresh_guard_failed")
    _mark_steady_phase("steady_layer_refresh_guard")

    replay_bind_device = (
        step_meta.seqused_k_gpu.device
        if isinstance(getattr(step_meta, "seqused_k_gpu", None), torch.Tensor)
        else next(iter(controller.layer_states.values())).device
    )
    replay_bind_device_t = torch.device(replay_bind_device)
    replay_arena_for_visible = getattr(controller, "_resolved_row_ptr_replay_arena", None)
    launch_effective_source = _resolved_row_ptr_launch_effective_k_len_source(
        launch_plan=launch_plan,
        batch_size=batch_size,
        device=replay_bind_device_t,
    )
    visible_source = (
        replay_arena_for_visible.carriers.resolver_visible_seqused_k_by_head_i32
        if isinstance(replay_arena_for_visible, ResolvedRowPtrArena)
        else None
    )
    visible_uses_launch_effective = (
        isinstance(visible_source, torch.Tensor)
        and isinstance(launch_effective_source, torch.Tensor)
        and int(visible_source.data_ptr()) == int(launch_effective_source.data_ptr())
    )
    _mark_steady_phase("z_device_source")
    update = None
    update_template_before_rrp = (
        visible_uses_launch_effective or mode is DecodeRuntimeMode.PAGE_BOUNDARY_DELTA
    )
    if update_template_before_rrp:
        update = apply_launch_template_row_delta(
            template,
            request_recent_len_by_row=delta.request_recent_len_by_row,
            launch_effective_k_by_row=delta.launch_effective_k_by_row,
            recent_first_page_by_row=delta.recent_first_page_by_row,
            recent_page_count_by_row=delta.recent_page_count_by_row,
            update_gpu=True,
            force_gpu_refresh=bool(visible_uses_launch_effective),
            # [ARM-WAR-R1-PINNED-INDEPENDENT 2026-07-12] fresh 镜像双引用替换。
            controller=controller,
        )
        if bool(update.requires_recompile):
            _reset_decode_runtime_full_recompile(
                controller,
                "launch_template_delta_requires_recompile:" f"{update.reason}",
            )
            return _miss("launch_template_delta_requires_recompile:" f"{update.reason}")
    _mark_steady_phase("z_template_apply")
    rrp_update = _try_update_same_page_resolved_row_ptr_step_state(
        controller,
        attn_metadata=attn_metadata,
        step_authority=step_authority,
        batch_size=batch_size,
        block_size=int(block_size),
        num_kv_heads=int(num_kv_heads),
        device=replay_bind_device_t,
        launch_plan=launch_plan,
    )
    if rrp_update is None:
        controller._decode_runtime_speculative_delta_miss_reason = (
            "delta_rrp_step_state_update_miss"
        )
        if os.environ.get("VLLM_ABLATE_FORCE_FASTPATH") != "1":
            return _miss("delta_rrp_step_state_update_miss")
    _mark_steady_phase("z_rrp_call")
    _mark_steady_phase("steady_rrp_attach")
    if update is None:
        update = apply_launch_template_row_delta(
            template,
            request_recent_len_by_row=delta.request_recent_len_by_row,
            launch_effective_k_by_row=delta.launch_effective_k_by_row,
            recent_first_page_by_row=delta.recent_first_page_by_row,
            recent_page_count_by_row=delta.recent_page_count_by_row,
            update_gpu=False,
            # [ARM-WAR-R1-PINNED-INDEPENDENT 2026-07-12] fresh 镜像双引用替换。
            controller=controller,
        )
    controller._decode_runtime_template_update_reason = update.reason
    same_page_update_kernel_count = int(
        update.carrier_update_kernel_count
    ) + int(getattr(rrp_update, "update_kernel_count", 0))
    runtime_state.counters.carrier_update_kernel_count += int(
        same_page_update_kernel_count
    )
    _mark_steady_phase("z_pre_record")
    ready_event_recorded = _record_resolved_row_ptr_owner_update_ready_event(
        controller,
        attn_metadata=attn_metadata,
        device=replay_bind_device_t,
        update_kernel_count=int(same_page_update_kernel_count),
    )
    _mark_steady_phase("steady_launch_template_update")

    _mark_steady_phase("steady_step_bound_meta_refresh")
    if controller.step_decode_data is not None:
        if step_meta.seqused_k_gpu is not None:
            controller.step_decode_data.seqused_k = step_meta.seqused_k_gpu
        controller.step_decode_data.epoch = int(step_authority.epoch)
        controller.step_decode_data.decode_plan_version = int(
            getattr(step_authority, "decode_plan_version", -1)
        )
    _mark_steady_phase("steady_step_decode_publish")
    controller._resolved_row_ptr_same_page_metadata_bind_skipped_count = (
        int(
            getattr(
                controller,
                "_resolved_row_ptr_same_page_metadata_bind_skipped_count",
                0,
            )
        )
        + 1
    )
    controller._decode_runtime_launch_template_update_count = (
        int(getattr(controller, "_decode_runtime_launch_template_update_count", 0))
        + 1
    )
    setattr(controller, "_decode_runtime_steady_fast_path_miss_reason", "hit")
    rrp_visible_debug_fields: dict[str, object] = {}
    replay_arena_for_debug = getattr(
        controller,
        "_resolved_row_ptr_replay_arena",
        None,
    )
    if (
        isinstance(replay_arena_for_debug, ResolvedRowPtrArena)
        and (
            mb_profile_enabled
            or bool(os.environ.get("VLLM_SPARSE_FA3_ROUTE_TRACE_LOG", "") if _DYNAMIC_ENV else _FA3_ROUTE_TRACE_LOG_CACHED)
        )
    ):
        rrp_visible_debug_fields = _rrp_visible_source_debug_fields(
            controller=controller,
            replay_arena=replay_arena_for_debug,
            launch_plan=launch_plan,
            row_effective_k_by_row=tuple(delta.row_effective_k_by_row),
            batch_size=batch_size,
            device=torch.device(replay_bind_device),
            sparse_dynamic_state=None,
        )
    _publish_rrp_visible_source_debug_attrs(attn_metadata, rrp_visible_debug_fields)
    if mb_profile_enabled:
        mb_phase_us.setdefault("steady_fast_path", 0.0)
        _append_mb_profile(
            {
                "event": "metadata_builder_step",
                "epoch": int(getattr(step_authority, "epoch", -1)),
                "batch_size": batch_size,
                "num_layers": int(len(getattr(controller, "layer_cache_keys", ()))),
                "decode_only": True,
                "has_decode_row": True,
                "reuse_decode_data": True,
                "full_decode_reuse_hit": True,
                "decode_reuse_miss_reason": "hit",
                "decode_reuse_miss_reasons": [],
                "step_dispatch_plan_reused": True,
                "ordered_layer_data_reused": True,
                "ordered_step_decode_data_reused": True,
                "buffers_ready": True,
                "need_fill_compact_layout": False,
                "need_pack_dynamic_req_meta": False,
                "skip_layer_state_refresh": True,
                "need_decode_out_ptr": False,
                "decode_logf_attn_rows": 0,
                "compact_rows": int(
                    sum(
                        1
                        for value in step_authority.row_mode_by_row[:batch_size]
                        if int(value) == int(_ROW_MODE_COMPACT)
                    )
                ),
                "decode_runtime_mode": mode.value,
                "decode_runtime_reason": str(reason),
                "runtime_classification_current": True,
                "same_page_step_count": int(mode is DecodeRuntimeMode.STEADY_DELTA),
                "page_boundary_step_count": int(
                    mode is DecodeRuntimeMode.PAGE_BOUNDARY_DELTA
                ),
                "refresh_commit_step_count": 0,
                "old_cache_probe_count": 0,
                "adapter_invocation_count": 0,
                "steady_delta_d2h_sync_count": 0,
                "implicit_sync_count": 0,
                "per_step_allocation_count": 0,
                "carrier_update_kernel_count": int(same_page_update_kernel_count),
                "rrp_ready_event_recorded": bool(ready_event_recorded),
                "rrp_ready_event_generation": int(
                    getattr(
                        attn_metadata,
                        "mixed_page_resolver_replay_ready_event_generation",
                        -1,
                    )
                ),
                "predicted_same_page_step_count": int(bool(predicted_same_page)),
                "build_compact_recent_launch_plan_call_count": 0,
                "bind_resolved_row_ptr_replay_metadata_call_count": 0,
                "pack_req_meta_decode_fast_layers_call_count": 0,
                "layer_states_traversal_count": 0,
                "decode_runtime_update_step_us": float(
                    time.perf_counter_ns() - mb_total_start_ns
                )
                / 1000.0,
                "phase_us": dict(mb_phase_us),
                "rrp_update_kind": str(getattr(rrp_update, "kind", "")),
                "ultra_first_miss_reason": str(
                    getattr(controller, "_forensic_ultra_first_miss_reason", "")
                ),
                "ultra_first_rrp_miss_reason": str(
                    getattr(controller, "_forensic_ultra_first_rrp_miss_reason", "")
                ),
                **rrp_visible_debug_fields,
                "total_us": float(time.perf_counter_ns() - mb_total_start_ns)
                / 1000.0,
            }
        )
    return True


_try_run_steady_decode_metadata_fast_path = _try_run_steady_decode_metadata_fast_path_inner


def _try_run_ultra_steady_decode_metadata_fast_path(
    controller: object,
    *,
    attn_metadata: object,
    step_meta: object,
    step_authority: object,
    previous_step_bound_meta: StepBoundMeta | None,
    kv_cache_spec: Optional[object],
    mb_profile_enabled: bool,
    steady_phase_enabled: bool,
    mb_total_start_ns: int,
    mb_phase_us: dict[str, float],
) -> bool:
    del kv_cache_spec

    def _wrapper_miss(reason: str) -> bool:
        # Wrapper-gate misses record onto the same miss-reason channel the
        # inner fast path owns, so per-step diagnostics always reflect THIS
        # attempt (the inner path resets the attribute on entry; wrapper
        # exits previously left a stale value behind).
        setattr(
            controller,
            "_decode_runtime_steady_fast_path_miss_reason",
            str(reason),
        )
        return False

    if previous_step_bound_meta is None:
        return _wrapper_miss("ultra_gate:previous_step_bound_meta_missing")
    if not bool(getattr(step_authority, "is_decode_only", False)):
        return _wrapper_miss("ultra_gate:not_decode_only")
    cached_kv_specs = _try_get_decode_runtime_cached_kv_specs(
        controller,
        prefer_cache=True,
    )
    if cached_kv_specs is None:
        return _wrapper_miss("ultra_gate:kv_spec_cache_missing")
    block_size_i, num_kv_heads_i, _head_dim_i, _kv_dtype_value = cached_kv_specs
    batch_size = int(getattr(step_authority, "batch_size", 0))
    if not _try_bind_decode_seq_lens_source_no_copy(
        attn_metadata=attn_metadata,
        step_meta=step_meta,
        batch_size=batch_size,
    ):
        return _wrapper_miss("ultra_gate:seq_lens_source_bind_failed")
    q_start_loc = getattr(step_authority, "q_start_loc", None)
    if q_start_loc is None or len(q_start_loc) < batch_size + 1:
        return _wrapper_miss("ultra_gate:q_start_loc_coverage")
    layer_effective_refresh_by_row = getattr(
        step_authority,
        "layer_effective_refresh_by_row",
        None,
    )
    if (
        layer_effective_refresh_by_row is None
        or len(layer_effective_refresh_by_row) < batch_size
    ):
        return _wrapper_miss("ultra_gate:layer_effective_refresh_coverage")
    # [LITE-P0 刀F 显式化 2026-07-11] 懒兑现的隐式依赖显式化——**只计数不
    # miss**:翻代 commit 推进 canonical layer 的 compact_meta_epoch,ultra 命
    # 中臂完全不看 live offset,过期 plan 被继续用=读旧半区(内容完好)的懒兑现
    # 语义=黄金锚锚定行为。强制 miss 会在"commit 落在稳态区间"的形态(U2 未验)
    # 下提前 pickup=改变读半区时机=数值面变(翻锚风险),故 P0 仅把该隐式依赖
    # 变成可观测计数:drift 窗口内的 ultra 命中步数落 `_lite_epoch_drift_hit_
    # count`(遥测,与懒兑现窗宽对账;若实测恒 0=commit 恰在簇内,P1 升级增量
    # pickup 臂时强制语义才安全)。零行为改变。
    _lite_template = getattr(controller, "_compact_recent_launch_template", None)
    if _lite_template is not None:
        _lite_keys = getattr(controller, "layer_cache_keys", None)
        _lite_states = getattr(controller, "layer_states", None)
        if _lite_keys and isinstance(_lite_states, dict):
            _lite_canonical = _lite_states.get(_lite_keys[0])
            if _lite_canonical is not None and int(
                getattr(_lite_canonical, "compact_meta_epoch", -1)
            ) != int(getattr(_lite_template, "compact_meta_epoch", -1)):
                controller._lite_epoch_drift_hit_count = (
                    int(getattr(controller, "_lite_epoch_drift_hit_count", 0)) + 1
                )
    return _try_run_steady_decode_metadata_fast_path(
        controller,
        attn_metadata=attn_metadata,
        step_meta=step_meta,
        step_authority=step_authority,
        previous_step_bound_meta=previous_step_bound_meta,
        kv_cache_spec=None,
        block_size=int(block_size_i),
        num_kv_heads=int(num_kv_heads_i),
        q_start_loc=tuple(int(v) for v in q_start_loc[: batch_size + 1]),
        layer_effective_refresh_by_row=tuple(
            bool(v) for v in layer_effective_refresh_by_row[:batch_size]
        ),
        mb_profile_enabled=mb_profile_enabled,
        steady_phase_enabled=steady_phase_enabled,
        mb_total_start_ns=mb_total_start_ns,
        mb_phase_us=mb_phase_us,
    )


def _validate_layer_slot_map_enabled() -> bool:
    if _DYNAMIC_ENV:
        return os.environ.get("VLLM_SPARSE_VALIDATE_LAYER_SLOT_MAP", "0") == "1"
    return bool(_VALIDATE_LAYER_SLOT_MAP_CACHED)


def _validate_bound_meta_contract_enabled() -> bool:
    if _DYNAMIC_ENV:
        return os.environ.get("VLLM_SPARSE_VALIDATE_META_CONTRACT", "0") == "1"
    return bool(_VALIDATE_META_CONTRACT_CACHED)


def _validate_layer_slot_signature_consistency(
    *,
    layer_cache_keys: List[object],
    layer_states: Dict[object, object],
    step_epoch: int,
    stage: str,
) -> None:
    if not layer_cache_keys:
        return
    first_key = layer_cache_keys[0]
    first_state = layer_states.get(first_key)
    if first_state is None:
        return
    base_epoch = int(getattr(first_state, "slot_epoch", -1))
    base_sig = int(getattr(first_state, "slot_signature64", -1))
    for layer_idx, layer_key in enumerate(layer_cache_keys[1:], start=1):
        state = layer_states.get(layer_key)
        if state is None:
            continue
        cur_epoch = int(getattr(state, "slot_epoch", -1))
        cur_sig = int(getattr(state, "slot_signature64", -1))
        if cur_epoch != base_epoch or cur_sig != base_sig:
            raise RuntimeError(
                f"slot signature mismatch before {stage} meta pack; "
                f"step_epoch={int(step_epoch)} base_layer=0 base_epoch={base_epoch} "
                f"base_sig={base_sig} layer_index={layer_idx} "
                f"layer_key={layer_key!r} layer_epoch={cur_epoch} layer_sig={cur_sig}"
            )


def _validate_bound_meta_compact_contract(
    *,
    step_bound_meta: StepBoundMeta,
    step_row_mode_by_row: Tuple[int, ...],
) -> None:
    batch_size = int(step_bound_meta.batch_size)
    if batch_size <= 0:
        return
    compact_rows = tuple(
        row
        for row in range(batch_size)
        if row < len(step_row_mode_by_row) and int(step_row_mode_by_row[row]) == int(_ROW_MODE_COMPACT)
    )
    if not compact_rows:
        return
    for layer_index, layer_bound in enumerate(step_bound_meta.layer_bound):
        if layer_bound is None:
            continue
        req_meta_i32 = layer_bound.req_meta_i32
        for row in compact_rows:
            compact_kv_len = int(req_meta_i32[row, 2].item())
            compact_block_cnt = int(req_meta_i32[row, 1].item())
            if compact_kv_len <= 0 or compact_block_cnt <= 0:
                raise RuntimeError(
                    "bound-meta compact contract violated: "
                    f"layer={layer_index} row={row} "
                    f"kv_len={compact_kv_len} block_cnt={compact_block_cnt}"
                )


def _ensure_decode_logf_no_rows_buffers(
    owner: object,
    *,
    device: torch.device,
    batch_size: int,
    max_batch: int,
) -> None:
    mask_storage_rebuilt = False
    if (
        getattr(owner, "_decode_log_f_mask_i32", None) is None
        or owner._decode_log_f_mask_i32.device != device
        or owner._decode_log_f_mask_i32.numel() < max_batch
    ):
        owner._decode_log_f_mask_i32 = torch.zeros(
            (max_batch,),
            device=device,
            dtype=torch.int32,
        )
        mask_storage_rebuilt = True
    q_lens_storage_rebuilt = False
    if (
        getattr(owner, "_decode_q_lens_i32", None) is None
        or owner._decode_q_lens_i32.device != device
        or owner._decode_q_lens_i32.numel() < max_batch
    ):
        owner._decode_q_lens_i32 = torch.ones(
            (max_batch,),
            device=device,
            dtype=torch.int32,
        )
        q_lens_storage_rebuilt = True
    if (
        batch_size > 0
        and not mask_storage_rebuilt
        and bool(getattr(owner, "_decode_log_f_mask_may_be_nonzero", False))
    ):
        owner._decode_log_f_mask_i32[:batch_size].zero_()
    if (
        batch_size > 0
        and not q_lens_storage_rebuilt
        and bool(getattr(owner, "_decode_q_lens_may_be_nonone", False))
    ):
        owner._decode_q_lens_i32[:batch_size].fill_(1)
    owner._decode_log_f_mask_may_be_nonzero = False
    owner._decode_q_lens_may_be_nonone = False


def _bind_canonical_real_kv_len(
    *,
    step_meta: object,
    seq_lens: torch.Tensor | None,
    batch_size: int,
    device: torch.device,
) -> None:
    if batch_size < 0:
        raise RuntimeError("canonical real-kv binding requires non-negative batch_size")
    if batch_size == 0:
        step_meta.canonical_real_kv_len_cpu = tuple()
        step_meta.canonical_real_kv_len_i32_gpu = torch.empty((0,), dtype=torch.int32, device=device)
        return

    cpu_truth = tuple(int(v) for v in step_meta.context_kv_len[:batch_size])
    if len(cpu_truth) != batch_size:
        raise RuntimeError(
            "canonical real-kv CPU truth must cover batch_size; "
            f"truth={len(cpu_truth)} batch_size={batch_size}"
        )

    canonical_gpu: torch.Tensor
    seqused_k_gpu = getattr(step_meta, "seqused_k_gpu", None)
    if (
        isinstance(seqused_k_gpu, torch.Tensor)
        and seqused_k_gpu.device == device
        and seqused_k_gpu.dtype == torch.int32
        and seqused_k_gpu.numel() >= batch_size
    ):
        canonical_gpu = seqused_k_gpu[:batch_size]
    elif isinstance(seq_lens, torch.Tensor) and seq_lens.numel() >= batch_size:
        canonical_gpu = seq_lens[:batch_size].to(device=device, dtype=torch.int32)
    else:
        canonical_gpu = torch.tensor(cpu_truth, dtype=torch.int32, device=device)

    if canonical_gpu.dim() != 1 or int(canonical_gpu.numel()) != int(batch_size):
        raise RuntimeError(
            "canonical real-kv GPU carrier must be activity-sized; "
            f"numel={int(canonical_gpu.numel())} batch_size={int(batch_size)}"
        )

    step_meta.canonical_real_kv_len_cpu = cpu_truth
    step_meta.canonical_real_kv_len_i32_gpu = canonical_gpu


def _device_identity(device: torch.device) -> Tuple[str, int]:
    return (str(device.type), -1 if device.index is None else int(device.index))


def _compact_page_lease_from_controller(controller: object) -> object | None:
    for state in getattr(controller, "layer_states", {}).values():
        residency = getattr(state, "compact_page_residency", None)
        lease = getattr(residency, "lease", None)
        if lease is not None:
            return lease
    return None


def _resolved_row_ptr_lease_snapshot(
    controller: object,
    *,
    lease: object | None,
    device: torch.device,
) -> tuple[tuple[int, ...], object, int]:
    if lease is None:
        return tuple(), tuple(), 0

    reserved_ids = getattr(lease, "reserved_manager_block_ids", tuple())
    compact_capacity_pages = int(getattr(lease, "compact_blocks_per_slot", 0) or 0)
    snapshot_key = (
        id(lease),
        int(getattr(lease, "reserve_epoch", -1)),
        id(reserved_ids),
        len(reserved_ids) if hasattr(reserved_ids, "__len__") else -1,
        compact_capacity_pages,
        _device_identity(device),
    )
    cached = getattr(controller, "_resolved_row_ptr_lease_snapshot_cache", None)
    if (
        isinstance(cached, tuple)
        and len(cached) == 4
        and cached[0] == snapshot_key
    ):
        return cached[1], cached[2], cached[3]

    reserved_manager_block_ids = tuple(int(v) for v in reserved_ids)
    reserved_manager_block_ids_arg: object = reserved_manager_block_ids
    if reserved_manager_block_ids:
        if device.type == "cuda":
            # [S1-KC-PIN-STAGING 2026-07-12] 原 torch.tensor(tuple, device=
            # cuda)=pageable H2D memcpy_and_sync(等当前流排空)。冷路径(lease
            # 换代才走,S1 探针 2 次/跑)但同栈同害,一并收口。独立 pinned 小
            # 分配+non_blocking H2D;结果 tensor 常驻 cache 跨步复用,首个消费
            # 与本 copy 同流=流序保证;WAR 护栏=CachingHostAllocator 事件跟踪。
            _reserved_len = len(reserved_manager_block_ids)
            try:
                _reserved_stage_cpu = torch.empty(
                    (_reserved_len,),
                    dtype=torch.int32,
                    device="cpu",
                    pin_memory=True,
                )
            except RuntimeError:
                _reserved_stage_cpu = torch.empty(
                    (_reserved_len,),
                    dtype=torch.int32,
                    device="cpu",
                )
            _reserved_stage_cpu.copy_(
                torch.as_tensor(
                    reserved_manager_block_ids,
                    dtype=torch.int32,
                    device="cpu",
                )
            )
            reserved_tensor = torch.empty(
                (_reserved_len,),
                dtype=torch.int32,
                device=device,
            )
            reserved_tensor.copy_(_reserved_stage_cpu, non_blocking=True)
        else:
            reserved_tensor = torch.tensor(
                reserved_manager_block_ids,
                dtype=torch.int32,
                device=device,
            )
        reserved_manager_block_ids_arg = reserved_tensor

    setattr(
        controller,
        "_resolved_row_ptr_lease_snapshot_cache",
        (
            snapshot_key,
            reserved_manager_block_ids,
            reserved_manager_block_ids_arg,
            compact_capacity_pages,
        ),
    )
    return (
        reserved_manager_block_ids,
        reserved_manager_block_ids_arg,
        compact_capacity_pages,
    )


def _positive_int_or_none(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        raise RuntimeError("kv block_size source must be a CPU scalar, not a tensor")
    value_i = int(value)
    if value_i <= 0:
        return None
    return value_i


def _resolve_positive_kv_block_size(
    controller: object,
    *,
    kv_cache_spec: Optional[object],
    step_meta: object,
) -> int | None:
    cached = _positive_int_or_none(getattr(controller, "kv_cache_block_size", None))
    if cached is not None:
        spec_value = _positive_int_or_none(getattr(kv_cache_spec, "block_size", None))
        step_value = _positive_int_or_none(getattr(step_meta, "block_size", None))
        if spec_value is not None and spec_value != cached:
            raise RuntimeError(
                "kv block_size sources disagree: "
                f"controller.kv_cache_block_size={cached} "
                f"kv_cache_spec.block_size={spec_value}"
            )
        if step_value is not None and step_value != cached:
            raise RuntimeError(
                "kv block_size sources disagree: "
                f"controller.kv_cache_block_size={cached} "
                f"step_meta.block_size={step_value}"
            )
        step_meta.block_size = int(cached)
        return int(cached)

    resolved: int | None = None
    resolved_label = ""
    for label, value in (
        ("kv_cache_spec.block_size", getattr(kv_cache_spec, "block_size", None)),
        ("step_meta.block_size", getattr(step_meta, "block_size", None)),
        (
            "controller.kv_cache_block_size",
            getattr(controller, "kv_cache_block_size", None),
        ),
    ):
        value_i = _positive_int_or_none(value)
        if value_i is None:
            continue
        if resolved is None:
            resolved = value_i
            resolved_label = label
            continue
        if value_i != resolved:
            raise RuntimeError(
                "kv block_size sources disagree: "
                f"{resolved_label}={resolved} {label}={value_i}"
            )

    if resolved is None:
        lease = _compact_page_lease_from_controller(controller)
        for label, value in (
            (
                "compact_page_lease.manager_block_size",
                getattr(lease, "manager_block_size", None),
            ),
            (
                "compact_page_lease.kernel_page_size",
                getattr(lease, "kernel_page_size", None),
            ),
        ):
            value_i = _positive_int_or_none(value)
            if value_i is None:
                continue
            if resolved is None:
                resolved = value_i
                resolved_label = label
                continue
            if value_i != resolved:
                raise RuntimeError(
                    "kv block_size sources disagree: "
                    f"{resolved_label}={resolved} {label}={value_i}"
                )

    if resolved is not None:
        step_meta.block_size = int(resolved)
        controller.kv_cache_block_size = int(resolved)
    return resolved


def _resolve_positive_kv_int(
    controller: object,
    *,
    kv_cache_spec: Optional[object],
    spec_attr: str,
    controller_attr: str,
    layer_attr: str,
) -> int | None:
    cached = _positive_int_or_none(getattr(controller, controller_attr, None))
    if cached is not None:
        spec_value = _positive_int_or_none(getattr(kv_cache_spec, spec_attr, None))
        if spec_value is not None and spec_value != cached:
            raise RuntimeError(
                "kv cache spec sources disagree: "
                f"controller.{controller_attr}={cached} "
                f"kv_cache_spec.{spec_attr}={spec_value}"
            )
        return int(cached)

    resolved: int | None = None
    resolved_label = ""
    for label, value in (
        (f"kv_cache_spec.{spec_attr}", getattr(kv_cache_spec, spec_attr, None)),
        (f"controller.{controller_attr}", getattr(controller, controller_attr, None)),
    ):
        value_i = _positive_int_or_none(value)
        if value_i is None:
            continue
        if resolved is None:
            resolved = value_i
            resolved_label = label
            continue
        if value_i != resolved:
            raise RuntimeError(
                "kv cache spec sources disagree: "
                f"{resolved_label}={resolved} {label}={value_i}"
            )

    if resolved is None:
        for state in getattr(controller, "layer_states", {}).values():
            value_i = _positive_int_or_none(getattr(state, layer_attr, None))
            if value_i is None:
                continue
            resolved = value_i
            resolved_label = f"layer_state.{layer_attr}"
            break

    if resolved is not None:
        setattr(controller, controller_attr, int(resolved))
    return resolved


def _resolve_kv_cache_dtype(
    controller: object,
    *,
    kv_cache_spec: Optional[object],
) -> torch.dtype | None:
    cached = getattr(controller, "kv_cache_dtype", None)
    if cached is not None:
        spec_value = getattr(kv_cache_spec, "dtype", None)
        if spec_value is not None and spec_value != cached:
            raise RuntimeError(
                "kv cache dtype sources disagree: "
                f"controller.kv_cache_dtype={cached} kv_cache_spec.dtype={spec_value}"
            )
        return cached

    resolved: torch.dtype | None = None
    resolved_label = ""
    for label, value in (
        ("kv_cache_spec.dtype", getattr(kv_cache_spec, "dtype", None)),
        ("controller.kv_cache_dtype", getattr(controller, "kv_cache_dtype", None)),
    ):
        if value is None:
            continue
        if resolved is None:
            resolved = value
            resolved_label = label
            continue
        if value != resolved:
            raise RuntimeError(
                "kv cache dtype sources disagree: "
                f"{resolved_label}={resolved} {label}={value}"
            )

    if resolved is None:
        for state in getattr(controller, "layer_states", {}).values():
            value = getattr(state, "kv_cache_dtype", None)
            if value is None:
                continue
            resolved = value
            break

    if resolved is not None:
        controller.kv_cache_dtype = resolved
    return resolved


def _validate_block_table_for_resolved_row_ptr_replay(
    *,
    block_table: object,
    batch_size: int,
    max_pages_per_row: int,
    device: torch.device | str,
) -> torch.Tensor:
    return validate_block_table_row_pointer_source(
        block_table=block_table,
        batch_size=batch_size,
        max_pages_per_row=max_pages_per_row,
        device=device,
    )


def _resolved_row_ptr_geometry(
    *,
    controller: object,
    step_authority: object,
    step_bound_meta: object | None,
    context_kv_len: Tuple[int, ...],
    page_size: int,
    batch_size: int,
) -> tuple[tuple[bool, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    row_mode_by_row = getattr(step_authority, "row_mode_by_row", None)
    if isinstance(row_mode_by_row, torch.Tensor):
        raise RuntimeError(
            "resolved-row-ptr replay metadata requires CPU row_mode_by_row sequence"
        )
    if row_mode_by_row is None or len(row_mode_by_row) < int(batch_size):
        raise RuntimeError(
            "resolved-row-ptr replay metadata requires step_authority.row_mode_by_row "
            f"coverage; batch_size={int(batch_size)}"
        )

    compact_ready_by_batch_row = tuple(
        int(row_mode_by_row[row]) == int(_ROW_MODE_COMPACT)
        for row in range(int(batch_size))
    )
    slot_by_row_source = getattr(step_authority, "slot_by_row", None)
    if isinstance(slot_by_row_source, torch.Tensor):
        raise RuntimeError(
            "resolved-row-ptr replay metadata requires CPU slot_by_row sequence"
        )
    if slot_by_row_source is not None and len(slot_by_row_source) >= int(batch_size):
        native_slot_by_row = tuple(
            int(v) for v in tuple(slot_by_row_source)[: int(batch_size)]
        )
    else:
        native_slot_by_row = tuple(range(int(batch_size)))
    native_geometry = (
        compact_ready_by_batch_row,
        tuple(int(v) for v in context_kv_len[: int(batch_size)]),
        native_slot_by_row,
        (0,) * int(batch_size),
        (0,) * int(batch_size),
    )
    if not any(compact_ready_by_batch_row):
        return native_geometry

    launch_plan = (
        getattr(step_bound_meta, "compact_recent_launch_plan", None)
        if step_bound_meta is not None
        else None
    )
    lease = _compact_page_lease_from_controller(controller)
    if launch_plan is None or not bool(getattr(launch_plan, "valid", False)) or lease is None:
        raise RuntimeError(
            "resolved-row-ptr compact row replay requires a valid compact_recent_launch_plan "
            "and compact page lease before graph replay"
        )

    from patches.fa_sparse_runtime.compact_mixed_page_route import (
        resolve_compact_mixed_page_overlay_cpu_geometry,
    )

    (
        compact_valid_tokens,
        recent_first_pages,
        _recent_page_count,
        row_effective_k,
    ) = resolve_compact_mixed_page_overlay_cpu_geometry(
        launch_plan=launch_plan,
        step_bound_meta=step_bound_meta,
        step_authority=step_authority,
        real_kv_len_hint=context_kv_len,
        page_size=int(page_size),
        batch_size=int(batch_size),
    )
    slot_by_row = tuple(
        int(v)
        for v in tuple(getattr(step_authority, "slot_by_row", tuple()))[
            : int(batch_size)
        ]
    )
    if len(slot_by_row) < int(batch_size):
        raise RuntimeError(
            "resolved-row-ptr compact rows require step_authority.slot_by_row coverage"
        )
    return (
        compact_ready_by_batch_row,
        tuple(int(v) for v in row_effective_k[: int(batch_size)]),
        slot_by_row,
        tuple(int(v) for v in compact_valid_tokens[: int(batch_size)]),
        tuple(int(v) for v in recent_first_pages[: int(batch_size)]),
    )


def _maybe_bind_resolved_row_ptr_replay_metadata(
    self,
    *,
    attn_metadata: object,
    batch_size: int,
    block_size: int,
    num_kv_heads: int,
    device: torch.device,
) -> None:
    profile_enabled = bool(_rrp_prep_profile_log_path())
    total_start_ns = time.perf_counter_ns() if profile_enabled else 0
    phase_start_ns = total_start_ns
    phase_us: dict[str, float] = {}
    self._decode_runtime_rrp_metadata_bind_call_count = 0

    def _mark_phase(name: str) -> None:
        nonlocal phase_start_ns
        if not profile_enabled:
            return
        now_ns = time.perf_counter_ns()
        phase_us[name] = float(now_ns - phase_start_ns) / 1000.0
        phase_start_ns = now_ns

    if not bool(getattr(self, "_sparse_attention_in_cudagraph", False)):
        return
    if not bool(getattr(self.config, "compact_page_residency_enabled", False)):
        return
    if self.step_decode_data is None:
        return

    step_authority = self.step_authority
    if step_authority is None:
        return
    live_batch_size_i = int(batch_size)
    graph_batch_size_i = _infer_rrp_graph_batch_size(
        attn_metadata=attn_metadata,
        controller=self,
        live_batch_size=live_batch_size_i,
    )
    batch_size_i = max(live_batch_size_i, int(graph_batch_size_i))
    live_arena_rows = tuple(range(live_batch_size_i))
    projected_graph_capacity = batch_size_i != live_batch_size_i
    inactive_dummy_k = 0
    canonical_row_index_by_batch_row = [-1] * batch_size_i
    for live_row in range(live_batch_size_i):
        canonical_row_index_by_batch_row[live_row] = live_row
    num_kv_heads_i = int(num_kv_heads)
    block_size_i = int(block_size)

    live_q_lens_source = getattr(step_authority, "q_lens_by_row", None)
    if isinstance(live_q_lens_source, torch.Tensor):
        raise RuntimeError(
            "resolved-row-ptr replay metadata requires CPU q_lens_by_row sequence"
        )
    if live_q_lens_source is None or len(live_q_lens_source) < live_batch_size_i:
        raise RuntimeError(
            "resolved-row-ptr replay metadata requires q_lens_by_row coverage; "
            f"live_batch_size={live_batch_size_i}"
        )
    live_q_lens = tuple(
        int(v) for v in tuple(live_q_lens_source)[:live_batch_size_i]
    )
    prefill_q_layout = any(int(v) > 1 for v in live_q_lens)
    if (
        prefill_q_layout
        and projected_graph_capacity
        and not bool(getattr(attn_metadata, "mixed_page_resolver_graph_replay_expected", False))
    ):
        graph_batch_size_i = live_batch_size_i
        batch_size_i = live_batch_size_i
        live_arena_rows = tuple(range(live_batch_size_i))
        projected_graph_capacity = False
        inactive_dummy_k = 0
        canonical_row_index_by_batch_row = list(range(live_batch_size_i))
    # Mixed prefill+compact rows under a q_len>1 launch are now representable:
    # the full-bind path below (_project_rrp_q_layout -> per-row affine 5-tuple,
    # incl. the compact branch) bakes the exact per-row q_lens into q_layout_key,
    # and max_seqlen_q is bucketed into the resolver-graph replay key so a replay
    # whose max prefill q_len exceeds the captured grid fails closed rather than
    # silently mis-tiling. The previous has_compact_row raise is therefore removed.

    same_page_fast_update = None
    if not projected_graph_capacity:
        same_page_fast_update = _try_attach_same_page_resolved_row_ptr_replay_metadata(
            self,
            attn_metadata=attn_metadata,
            step_authority=step_authority,
            batch_size=batch_size_i,
            block_size=block_size_i,
            num_kv_heads=num_kv_heads_i,
            device=device,
        )
    if same_page_fast_update is not None:
        self._resolved_row_ptr_same_page_metadata_bind_skipped_count = (
            int(
                getattr(
                    self,
                    "_resolved_row_ptr_same_page_metadata_bind_skipped_count",
                    0,
                )
            )
            + 1
        )
        replay_arena = getattr(self, "_resolved_row_ptr_replay_arena", None)
        row_modes_for_trace = tuple(
            int(v)
            for v in tuple(getattr(step_authority, "row_mode_by_row", tuple()))[
                :live_batch_size_i
            ]
        )
        delta_for_trace = getattr(self, "_decode_runtime_delta", None)
        row_effective_for_trace = tuple(
            int(v)
            for v in tuple(
                getattr(
                    delta_for_trace,
                    "row_effective_k_by_row",
                    getattr(step_authority, "context_kv_len_by_row", tuple()),
                )
            )[:live_batch_size_i]
        )
        if profile_enabled:
            replay_arena = getattr(self, "_resolved_row_ptr_replay_arena", None)
            _mark_phase("same_page_light_attach")
            _append_rrp_prep_profile(
                {
                    "event": "rrp_replay_metadata_bind",
                    "epoch": int(getattr(step_authority, "epoch", -1)),
                    "batch_size": batch_size_i,
                    "num_kv_heads": num_kv_heads_i,
                    "block_size": block_size_i,
                    "rrp_signature_hit": bool(
                        getattr(same_page_fast_update, "hit", False)
                    ),
                    "rrp_update_reason": str(
                        getattr(same_page_fast_update, "miss_reason", "")
                    ),
                    "rrp_update_kind": str(
                        getattr(same_page_fast_update, "kind", "")
                    ),
                    "rrp_delta_rows": [
                        int(row)
                        for row in getattr(same_page_fast_update, "delta_rows", ())
                    ],
                    "rrp_full_bind": bool(
                        getattr(same_page_fast_update, "full_bind", False)
                    ),
                    "rrp_actual_full_bind": False,
                    "rrp_metadata_bind_required": False,
                    "rrp_metadata_bind_skipped": True,
                    "rrp_metadata_light_attach": True,
                    "rrp_update_kernel_count": int(
                        getattr(same_page_fast_update, "update_kernel_count", 0)
                    ),
                    "rrp_ready_event_recorded": bool(
                        getattr(
                            self,
                            "_resolved_row_ptr_ready_event_recorded",
                            False,
                        )
                    ),
                    "rrp_ready_event_generation": int(
                        getattr(
                            attn_metadata,
                            "mixed_page_resolver_replay_ready_event_generation",
                            -1,
                        )
                    ),
                    "arena_generation": int(
                        getattr(
                            replay_arena,
                            "generation",
                            -1,
                        )
                    ),
                    "row_source_distribution": dict(
                        getattr(replay_arena, "row_source_distribution", {})
                    ),
                    **build_source_counter_fields(
                        batch_size=batch_size_i,
                        num_kv_heads=num_kv_heads_i,
                        row_source_distribution=getattr(
                            replay_arena,
                            "row_source_distribution",
                            {},
                        ),
                    ),
                    "phase_us": dict(phase_us),
                    "total_us": float(time.perf_counter_ns() - total_start_ns)
                    / 1000.0,
                }
            )
        return

    context_kv_len_by_row = getattr(step_authority, "context_kv_len_by_row", None)
    if isinstance(context_kv_len_by_row, torch.Tensor):
        raise RuntimeError(
            "resolved-row-ptr replay metadata requires CPU context_kv_len_by_row sequence"
        )
    if context_kv_len_by_row is None or len(context_kv_len_by_row) < live_batch_size_i:
        raise RuntimeError(
            "resolved-row-ptr replay metadata requires context_kv_len_by_row coverage; "
            f"live_batch_size={live_batch_size_i}"
        )
    worker_block_table = getattr(self, "_worker_block_table", None)
    worker_block_table_cpu = getattr(self, "_worker_block_table_cpu", None)
    worker_block_table_i32 = _validate_block_table_for_resolved_row_ptr_replay(
        block_table=worker_block_table,
        batch_size=live_batch_size_i,
        max_pages_per_row=1,
        device=device,
    )
    block_table_capacity_pages = int(worker_block_table_i32.shape[1])
    live_context_kv_len = tuple(
        int(v) for v in tuple(context_kv_len_by_row)[:live_batch_size_i]
    )
    context_kv_len = tuple(
        int(v)
        for v in _project_rrp_row_tuple(
            live_context_kv_len,
            default=inactive_dummy_k,
            arena_batch_size=batch_size_i,
            active_arena_row_indices=live_arena_rows,
        )
    )
    max_seq_len = max(live_context_kv_len, default=0)
    required_pages_per_row = max(
        1,
        (max_seq_len + block_size_i - 1) // block_size_i,
    )
    if required_pages_per_row > block_table_capacity_pages:
        raise RuntimeError(
            "resolved-row-ptr replay metadata requires block_table coverage for "
            f"live pages; required={required_pages_per_row} "
            f"capacity={block_table_capacity_pages}"
        )
    max_pages_per_row = block_table_capacity_pages
    q_lens, q_start_loc = _project_rrp_q_layout(
        q_lens_by_row=getattr(step_authority, "q_lens_by_row", None),
        live_batch_size=live_batch_size_i,
        arena_batch_size=batch_size_i,
        active_arena_row_indices=live_arena_rows,
        inactive_q_len=inactive_dummy_k,
    )
    _mark_phase("validate_and_geometry_inputs")

    arena_key = (
        batch_size_i,
        num_kv_heads_i,
        block_size_i,
        int(max_pages_per_row),
        _device_identity(device),
    )
    graph_replay_enabled = bool(
        getattr(attn_metadata, "mixed_page_resolver_graph_replay_expected", False)
    )
    current_arena_key = getattr(self, "_resolved_row_ptr_arena_key", None)
    arena_by_key = getattr(self, "_resolved_row_ptr_replay_arena_by_key", None)
    if not isinstance(arena_by_key, dict):
        arena_by_key = {}
        self._resolved_row_ptr_replay_arena_by_key = arena_by_key
    manager_by_key = getattr(self, "_rrp_row_table_manager_by_key", None)
    if not isinstance(manager_by_key, dict):
        manager_by_key = {}
        self._rrp_row_table_manager_by_key = manager_by_key
    binding_by_key = getattr(
        self,
        "_resolved_row_ptr_replay_metadata_binding_by_key",
        None,
    )
    if not isinstance(binding_by_key, dict):
        binding_by_key = {}
        self._resolved_row_ptr_replay_metadata_binding_by_key = binding_by_key

    replay_arena = arena_by_key.get(arena_key)
    if replay_arena is None:
        if graph_replay_enabled and current_arena_key != arena_key:
            raise RuntimeError(
                "resolved-row-ptr replay carrier capacity changed during CUDA graph replay; "
                "graph replay cannot resize carrier tensors"
            )
        replay_arena = ResolvedRowPtrArena.allocate(
            batch_size=batch_size_i,
            num_kv_heads=num_kv_heads_i,
            max_pages_per_row=int(max_pages_per_row),
            device=device,
        )
        arena_by_key[arena_key] = replay_arena
    if arena_key not in manager_by_key:
        manager_by_key[arena_key] = RrpRowTableManager()

    self._resolved_row_ptr_replay_arena = replay_arena
    self._resolved_row_ptr_source_arena = replay_arena
    self._resolved_row_ptr_arena_key = arena_key
    self._rrp_row_table_manager = manager_by_key[arena_key]
    self._resolved_row_ptr_replay_metadata_binding = binding_by_key.get(arena_key)
    self._resolved_row_ptr_metadata_ready = (
        self._resolved_row_ptr_replay_metadata_binding is not None
    )
    source_arena = replay_arena
    if replay_arena is None or source_arena is None:
        return
    _mark_phase("arena_lookup")

    launch_plan = (
        getattr(getattr(self, "step_bound_meta", None), "compact_recent_launch_plan", None)
    )
    (
        live_compact_ready_by_batch_row,
        live_row_effective_k_by_row,
        live_slot_by_row,
        live_compact_valid_tokens_by_row,
        live_recent_first_page_by_row,
    ) = _resolved_row_ptr_geometry(
        controller=self,
        step_authority=step_authority,
        step_bound_meta=getattr(self, "step_bound_meta", None),
        context_kv_len=live_context_kv_len,
        page_size=block_size_i,
        batch_size=live_batch_size_i,
    )
    compact_ready_by_batch_row = tuple(
        bool(v)
        for v in _project_rrp_row_tuple(
            tuple(live_compact_ready_by_batch_row),
            default=False,
            arena_batch_size=batch_size_i,
            active_arena_row_indices=live_arena_rows,
        )
    )
    row_effective_k_by_row = tuple(
        int(v)
        for v in _project_rrp_row_tuple(
            tuple(live_row_effective_k_by_row),
            default=inactive_dummy_k,
            arena_batch_size=batch_size_i,
            active_arena_row_indices=live_arena_rows,
        )
    )
    slot_by_row = tuple(
        int(v)
        for v in _project_rrp_row_tuple(
            tuple(live_slot_by_row),
            default=-1,
            arena_batch_size=batch_size_i,
            active_arena_row_indices=live_arena_rows,
        )
    )
    compact_valid_tokens_by_row = tuple(
        int(v)
        for v in _project_rrp_row_tuple(
            tuple(live_compact_valid_tokens_by_row),
            default=0,
            arena_batch_size=batch_size_i,
            active_arena_row_indices=live_arena_rows,
        )
    )
    recent_first_page_by_row = tuple(
        int(v)
        for v in _project_rrp_row_tuple(
            tuple(live_recent_first_page_by_row),
            default=0,
            arena_batch_size=batch_size_i,
            active_arena_row_indices=live_arena_rows,
        )
    )
    attention_in_cudagraph = bool(
        getattr(self, "_sparse_attention_in_cudagraph", False)
    )
    launch_effective_source = _resolved_row_ptr_launch_effective_k_len_source(
        launch_plan=launch_plan,
        batch_size=batch_size_i,
        device=device,
    )
    launch_effective_covers_rows = _resolved_row_ptr_launch_effective_covers_rows(
        launch_plan=launch_plan,
        row_effective_k_by_row=row_effective_k_by_row,
        batch_size=batch_size_i,
    )
    sparse_visible_k_by_live_row = tuple(
        int(row_effective_k_by_row[int(arena_row)])
        for arena_row in tuple(live_arena_rows)[:live_batch_size_i]
    )
    active_live_arena_rows = tuple(live_arena_rows)[:live_batch_size_i]

    def _project_launch_effective_live_rows(
        values: object,
    ) -> tuple[int, ...]:
        if values is None or isinstance(values, torch.Tensor):
            return tuple()
        try:
            value_tuple = tuple(int(x) for x in tuple(values))
        except Exception:
            return tuple()
        if len(value_tuple) >= batch_size_i:
            return tuple(
                int(value_tuple[int(arena_row)])
                for arena_row in active_live_arena_rows
            )
        if len(value_tuple) >= live_batch_size_i:
            return tuple(int(v) for v in value_tuple[:live_batch_size_i])
        return tuple()

    # [LIFECYCLE-OFF-ONLY 2026-07-10] native lifecycle 整臂下线:本区原
    # ON 专属的 live-row 投影/graph launch-effective 视图/native 批状态取值
    # 全部删除,三元式塌缩到 OFF 值;capacity_key 的 "full" 构造提为无条件
    # (_sparse_native_capacity_key 全仓零写点,getattr 恒 None,
    # RESIDENCY-CAPACITY-MEMO 依赖此键)。
    decode_delta_for_launch = getattr(self, "_decode_runtime_delta", None)
    if (
        decode_delta_for_launch is not None
        and int(getattr(decode_delta_for_launch, "step_id", -2))
        == int(getattr(step_authority, "epoch", -1))
    ):
        sparse_launch_effective_k_by_live_row = _project_launch_effective_live_rows(
            getattr(decode_delta_for_launch, "launch_effective_k_by_row", None)
        )
    else:
        sparse_launch_effective_k_by_live_row = tuple()
    if len(sparse_launch_effective_k_by_live_row) != live_batch_size_i:
        sparse_launch_effective_k_by_live_row = _project_launch_effective_live_rows(
            getattr(launch_plan, "launch_effective_k_len_cpu", None)
        )
    launch_plan_for_visible_source = launch_plan
    sparse_launch_effective_k_by_row = getattr(
        getattr(self, "_decode_runtime_delta", None),
        "launch_effective_k_by_row",
        tuple(),
    )
    sparse_native_capacity_key = (
        "full",
        batch_size_i,
        block_size_i,
        num_kv_heads_i,
        int(replay_arena.max_pages_per_row),
    )
    sparse_bind_batch_size = batch_size_i
    sparse_bind_row_effective_k_by_row = tuple(row_effective_k_by_row[:batch_size_i])
    compact_offset_tokens_by_row = (0,) * batch_size_i
    live_compact_offset_tokens_by_row = (0,) * live_batch_size_i
    row_effective_k_i32_gpu = (
        getattr(launch_plan, "launch_effective_k_len_i32", None)
        if launch_plan is not None
        else None
    )
    if not (
        isinstance(row_effective_k_i32_gpu, torch.Tensor)
        and row_effective_k_i32_gpu.device == device
        and row_effective_k_i32_gpu.dtype == torch.int32
        and row_effective_k_i32_gpu.dim() == 1
        and int(row_effective_k_i32_gpu.numel()) >= live_batch_size_i
        and not projected_graph_capacity
    ):
        row_effective_k_i32_gpu = None
    plan_compact_offsets = (
        getattr(launch_plan, "compact_offset_tokens_cpu", tuple())
        if launch_plan is not None
        else tuple()
    )
    if len(plan_compact_offsets) >= live_batch_size_i:
        live_compact_offset_tokens_by_row = tuple(
            int(v) for v in tuple(plan_compact_offsets)[:live_batch_size_i]
        )
        compact_offset_tokens_by_row = tuple(
            int(v)
            for v in _project_rrp_row_tuple(
                live_compact_offset_tokens_by_row,
                default=0,
                arena_batch_size=batch_size_i,
                active_arena_row_indices=live_arena_rows,
            )
        )
    elif any(compact_ready_by_batch_row) and bool(getattr(launch_plan, "valid", False)):
        raise RuntimeError(
            "compact offset mirror must cover compact rows before resolved-row-ptr replay; "
            f"offsets={len(plan_compact_offsets)} live_batch_size={live_batch_size_i}"
        )
    _mark_phase("cpu_geometry")
    lease = _compact_page_lease_from_controller(self)
    (
        reserved_manager_block_ids,
        reserved_manager_block_ids_arg,
        compact_capacity_pages,
    ) = _resolved_row_ptr_lease_snapshot(self, lease=lease, device=device)
    _mark_phase("lease_snapshot")
    # [LIFECYCLE-OFF-ONLY] 原 ON 专属 descriptor_payload 构建/native visible
    # state 更新/绑定失败早退整块删除;native 批状态参数塌缩为 None 字面量。
    visible_source_bound = _bind_resolved_row_ptr_visible_seqused_source(
        replay_arena=replay_arena,
        source_arena=source_arena,
        launch_plan=launch_plan_for_visible_source,
        step_meta=getattr(self, "step_meta", None),
        batch_size=sparse_bind_batch_size,
        device=device,
        sparse_dynamic_state=None,
        row_effective_k_by_row=sparse_bind_row_effective_k_by_row,
        launch_effective_k_by_row=sparse_launch_effective_k_by_row,
        step_authority=step_authority,
        capacity_key=sparse_native_capacity_key,
        active_row_indices=None,
        slot_mapping_signature=None,
    )
    replay_visible = replay_arena.carriers.resolver_visible_seqused_k_by_head_i32
    source_visible = source_arena.carriers.resolver_visible_seqused_k_by_head_i32
    if (
        not _resolved_row_ptr_visible_source_covers_arena(
            replay_visible, replay_arena
        )
        or not _resolved_row_ptr_visible_source_covers_arena(
            source_visible, source_arena
        )
    ):
        replay_count = (
            int(replay_visible.numel())
            if isinstance(replay_visible, torch.Tensor)
            else -1
        )
        source_count = (
            int(source_visible.numel())
            if isinstance(source_visible, torch.Tensor)
            else -1
        )
        raise RuntimeError(
            "resolved-row-ptr replay metadata requires graph-stable visible "
            "seqused storage "
            f"(replay_count={replay_count} "
            f"replay_batch={int(replay_arena.batch_size)} "
            f"replay_total={int(replay_arena.total_rows)} "
            f"source_count={source_count} "
            f"source_batch={int(source_arena.batch_size)} "
            f"source_total={int(source_arena.total_rows)})"
        )
    rrp_manager = getattr(self, "_rrp_row_table_manager", None)
    if not isinstance(rrp_manager, RrpRowTableManager):
        rrp_manager = RrpRowTableManager()
        self._rrp_row_table_manager = rrp_manager
    live_req_ids = tuple(
        str(v)
        for v in tuple(getattr(step_authority, "req_ids", tuple()))[:live_batch_size_i]
    )
    if len(live_req_ids) < live_batch_size_i:
        raise RuntimeError(
            "resolved-row-ptr replay metadata requires step_authority.req_ids coverage; "
            f"live_batch_size={live_batch_size_i} req_ids={len(live_req_ids)}"
        )
    req_ids_by_row = tuple(
        str(v)
        for v in _project_rrp_row_tuple(
            live_req_ids,
            default="__inactive_rrp_row__",
            arena_batch_size=batch_size_i,
            active_arena_row_indices=live_arena_rows,
        )
    )
    live_row_mode_by_row = tuple(
        int(_ROW_MODE_COMPACT) if bool(v) else int(_ROW_MODE_DENSE)
        for v in tuple(live_compact_ready_by_batch_row)
    )
    row_mode_by_row = tuple(
        int(v)
        for v in _project_rrp_row_tuple(
            live_row_mode_by_row,
            default=int(_ROW_MODE_DENSE),
            arena_batch_size=batch_size_i,
            active_arena_row_indices=live_arena_rows,
        )
    )
    rrp_update = rrp_manager.update(
        replay_arena,
        RrpRowTableInputs(
            batch_size=batch_size_i,
            req_ids_by_row=req_ids_by_row,
            slot_by_row=slot_by_row,
            row_mode_by_row=row_mode_by_row,
            compact_ready_by_row=compact_ready_by_batch_row,
            row_effective_k_by_row=row_effective_k_by_row,
            compact_valid_tokens_by_row=compact_valid_tokens_by_row,
            compact_offset_tokens_by_row=compact_offset_tokens_by_row,
            recent_first_page_by_row=recent_first_page_by_row,
            reserved_manager_block_ids=reserved_manager_block_ids,
            compact_capacity_pages=compact_capacity_pages,
            max_pages_per_row=int(max_pages_per_row),
            page_size=block_size_i,
            row_effective_k_i32_gpu=row_effective_k_i32_gpu,
        ),
    )
    _mark_phase("rrp_row_table_manager")
    rrp_actual_full_bind = bool(rrp_update.full_bind or rrp_update.delta_rows)
    if rrp_actual_full_bind:
        canonical_rows = tuple(int(row) for row in canonical_row_index_by_batch_row)
        direct_native_block_table = (
            not bool(attention_in_cudagraph)
            and not bool(graph_replay_enabled)
            and not any(bool(value) for value in compact_ready_by_batch_row)
            and len(canonical_rows) >= batch_size_i
            and canonical_rows[:batch_size_i] == tuple(range(batch_size_i))
        )
        if direct_native_block_table:
            replay_arena.bind_block_table_row_pointers(
                block_table=worker_block_table_i32,
                compact_ready_by_batch_row=(False,) * batch_size_i,
                row_effective_k_by_row=row_effective_k_by_row,
                page_size=block_size_i,
            )
        else:
            replay_arena.bind_production_row_table(
                canonical_block_table=worker_block_table_i32,
                compact_ready_by_batch_row=compact_ready_by_batch_row,
                row_effective_k_by_row=row_effective_k_by_row,
                page_size=block_size_i,
                reserved_manager_block_ids=reserved_manager_block_ids_arg,
                slot_by_row=slot_by_row,
                compact_valid_tokens_by_row=compact_valid_tokens_by_row,
                compact_offset_tokens_by_row=compact_offset_tokens_by_row,
                recent_first_page_by_row=recent_first_page_by_row,
                canonical_row_index_by_batch_row=canonical_rows,
                compact_capacity_pages=compact_capacity_pages,
                profile_phase_us=phase_us if profile_enabled else None,
                reserved_manager_block_ids_cpu=reserved_manager_block_ids,
                canonical_block_table_cpu=worker_block_table_cpu,
            )
    _mark_phase("bind_production_row_table")
    q_layout_key = f"q_lens={q_lens};q_start={q_start_loc}"
    decode_runtime_mode = getattr(self, "_decode_runtime_mode", None)
    runtime_state = getattr(
        self,
        "_decode_runtime_state",
        getattr(self, "decode_runtime_state", None),
    )
    if decode_runtime_mode is None and runtime_state is not None:
        decode_runtime_mode = getattr(runtime_state, "last_mode", None)
    binding = getattr(self, "_resolved_row_ptr_replay_metadata_binding", None)
    binding_visible_matches_current = False
    if isinstance(binding, ResolvedRowPtrReplayMetadataBinding):
        binding_visible = (
            binding.replay_carriers.resolver_visible_seqused_k_by_head_i32
        )
        current_visible = (
            replay_arena.carriers.resolver_visible_seqused_k_by_head_i32
        )
        if isinstance(binding_visible, torch.Tensor) and isinstance(
            current_visible,
            torch.Tensor,
        ):
            binding_visible_matches_current = (
                int(binding_visible.data_ptr()) == int(current_visible.data_ptr())
            )
    cached_binding_ready = bool(
        getattr(self, "_resolved_row_ptr_metadata_ready", False)
        and isinstance(binding, ResolvedRowPtrReplayMetadataBinding)
        and binding.replay_arena is replay_arena
        and str(binding.descriptor.q_layout_key) == q_layout_key
        and binding_visible_matches_current
    )
    metadata_ready = bool(
        getattr(self, "_resolved_row_ptr_metadata_ready", False)
        and (
            _attn_metadata_has_resolved_row_ptr_replay_binding(attn_metadata)
            or cached_binding_ready
        )
    )
    should_bind_metadata = _should_bind_resolved_row_ptr_replay_metadata_for_update(
        decode_runtime_mode,
        rrp_update,
        metadata_ready,
    )
    rrp_metadata_bind_skipped = not bool(should_bind_metadata)
    if should_bind_metadata:
        self._decode_runtime_rrp_metadata_bind_call_count = 1
        binding = bind_resolved_row_ptr_replay_metadata(
            attn_metadata=attn_metadata,
            replay_arena=replay_arena,
            source_arena=replay_arena,
            batch_size=batch_size_i,
            num_kv_heads=num_kv_heads_i,
            page_block_size=block_size_i,
            max_pages_per_row=int(max_pages_per_row),
            q_layout_key=q_layout_key,
            max_seqlen_q=max((int(v) for v in q_lens), default=1),
            max_seqlen_k=_resolved_row_ptr_profile_max_seqlen_k(
                launch_plan=launch_plan,
                row_effective_k_by_row=row_effective_k_by_row,
                batch_size=batch_size_i,
                block_size=block_size_i,
                max_pages_per_row=int(max_pages_per_row),
            ),
        )
        self._resolved_row_ptr_replay_metadata_binding = binding
        binding_by_key[arena_key] = binding
        self._resolved_row_ptr_metadata_ready = True
    else:
        self._resolved_row_ptr_same_page_metadata_bind_skipped_count = (
            int(
                getattr(
                    self,
                    "_resolved_row_ptr_same_page_metadata_bind_skipped_count",
                    0,
                )
            )
            + 1
        )
        if isinstance(binding, ResolvedRowPtrReplayMetadataBinding):
            attach_resolved_row_ptr_replay_metadata(
                attn_metadata=attn_metadata,
                binding=binding,
            )
    rrp_profile_max_seqlen_k = _publish_resolved_row_ptr_profile_max_seqlen_k(
        attn_metadata=attn_metadata,
        launch_plan=launch_plan,
        row_effective_k_by_row=row_effective_k_by_row,
        batch_size=batch_size_i,
        block_size=block_size_i,
        max_pages_per_row=int(max_pages_per_row),
    )
    rrp_visible_debug_fields = _rrp_visible_source_debug_fields(
        controller=self,
        replay_arena=replay_arena,
        launch_plan=launch_plan_for_visible_source,
        row_effective_k_by_row=tuple(row_effective_k_by_row),
        batch_size=batch_size_i,
        device=device,
        sparse_dynamic_state=None,
    )
    _publish_rrp_visible_source_debug_attrs(attn_metadata, rrp_visible_debug_fields)
    _mark_phase("bind_metadata_attrs")
    _ready_event_holders = _resolved_row_ptr_ready_event_holders(attn_metadata)
    (
        existing_ready_event_generation,
        existing_ready_event,
        existing_ready_event_stream,
    ) = _resolved_row_ptr_ready_event_state(
        attn_metadata, holders=_ready_event_holders
    )
    ready_event_recorded = False
    if should_record_rrp_ready_event(
        rrp_update,
        metadata_bind_required=bool(should_bind_metadata),
        existing_ready_event_generation=existing_ready_event_generation,
    ):
        ready_event_recorded = _record_resolved_row_ptr_replay_ready_event(
            attn_metadata,
            device=device,
            holders=_ready_event_holders,
        )
    # Design A (byte-neutral): the prior unconditional re-attach of the values
    # just read above was a value no-op (all holders stay synchronized), and the
    # second state re-read is only needed when we actually recorded a new event;
    # otherwise reuse the existing_* triple read above (bit-identical).
    (
        ready_event_generation,
        ready_event,
        ready_event_stream,
    ) = (
        _resolved_row_ptr_ready_event_state(
            attn_metadata, holders=_ready_event_holders
        )
        if ready_event_recorded
        else (
            existing_ready_event_generation,
            existing_ready_event,
            existing_ready_event_stream,
        )
    )
    self._resolved_row_ptr_ready_event_recorded = bool(ready_event_recorded)
    if isinstance(binding, ResolvedRowPtrReplayMetadataBinding):
        _publish_resolved_row_ptr_graph_binding_state(
            self,
            binding=binding,
            arena_key=arena_key,
            ready_event_generation=int(ready_event_generation),
            ready_event=ready_event,
            ready_event_stream=int(ready_event_stream),
        )
        _stamp_resolved_row_ptr_graph_binding_step(
            controller=self,
            step_authority=step_authority,
            live_batch_size=live_batch_size_i,
            effective_batch_size=batch_size_i,
        )
    _mark_phase("record_ready_event")
    if profile_enabled:
        _append_rrp_prep_profile(
            {
                "event": "rrp_replay_metadata_bind",
                "epoch": int(getattr(step_authority, "epoch", -1)),
                "batch_size": batch_size_i,
                "live_batch_size": live_batch_size_i,
                "graph_batch_size": graph_batch_size_i,
                "active_arena_row_indices": [int(v) for v in live_arena_rows],
                "canonical_row_index_by_batch_row": [
                    int(v) for v in canonical_row_index_by_batch_row
                ],
                "num_kv_heads": num_kv_heads_i,
                "block_size": block_size_i,
                "block_table_capacity_pages": int(block_table_capacity_pages),
                "required_pages_per_row": int(required_pages_per_row),
                "max_seq_len": int(max_seq_len),
                "compact_rows": int(sum(1 for v in compact_ready_by_batch_row if v)),
                "req_ids_by_row": [str(v) for v in req_ids_by_row],
                "slot_by_row": [int(v) for v in slot_by_row],
                "row_mode_by_row": [int(v) for v in row_mode_by_row],
                "q_lens": [int(v) for v in q_lens],
                "q_start_loc": [int(v) for v in q_start_loc],
                "row_effective_k_by_row": [int(v) for v in row_effective_k_by_row],
                "compact_valid_tokens_by_row": [
                    int(v) for v in compact_valid_tokens_by_row
                ],
                "recent_first_page_by_row": [int(v) for v in recent_first_page_by_row],
                "rrp_profile_max_seqlen_k": int(rrp_profile_max_seqlen_k),
                "compact_capacity_pages": int(compact_capacity_pages),
                "reserved_manager_block_ids": int(len(reserved_manager_block_ids)),
                "rrp_signature_hit": bool(rrp_update.hit),
                "rrp_update_reason": str(rrp_update.miss_reason),
                "rrp_update_kind": str(getattr(rrp_update, "kind", "")),
                "rrp_delta_rows": [int(row) for row in rrp_update.delta_rows],
                "rrp_full_bind": bool(rrp_update.full_bind),
                "rrp_actual_full_bind": bool(rrp_actual_full_bind),
                "rrp_metadata_bind_required": bool(
                    getattr(rrp_update, "metadata_bind_required", True)
                ),
                "rrp_metadata_bind_skipped": bool(rrp_metadata_bind_skipped),
                "rrp_update_kernel_count": int(
                    getattr(rrp_update, "update_kernel_count", 0)
                ),
                "rrp_ready_event_recorded": bool(ready_event_recorded),
                "rrp_ready_event_generation": int(
                    getattr(
                        attn_metadata,
                        "mixed_page_resolver_replay_ready_event_generation",
                        -1,
                    )
                ),
                "arena_generation": int(getattr(replay_arena, "generation", -1)),
                "row_source_distribution": dict(
                    getattr(replay_arena, "row_source_distribution", {})
                ),
                **build_source_counter_fields(
                    batch_size=batch_size_i,
                    num_kv_heads=num_kv_heads_i,
                    row_source_distribution=getattr(
                        replay_arena,
                        "row_source_distribution",
                        {},
                    ),
                ),
                **_rrp_reserved_source_overlap_profile(
                    worker_block_table_i32=worker_block_table_i32,
                    reserved_manager_block_ids=reserved_manager_block_ids,
                    compact_ready_by_batch_row=compact_ready_by_batch_row,
                    row_effective_k_by_row=row_effective_k_by_row,
                    compact_valid_tokens_by_row=compact_valid_tokens_by_row,
                    recent_first_page_by_row=recent_first_page_by_row,
                    canonical_row_index_by_batch_row=tuple(
                        canonical_row_index_by_batch_row
                    ),
                    batch_size=batch_size_i,
                    page_size=block_size_i,
                ),
                **rrp_visible_debug_fields,
                "phase_us": dict(phase_us),
                "total_us": float(time.perf_counter_ns() - total_start_ns) / 1000.0,
            }
        )


def maybe_build_step_decode_data_from_metadata_impl(
    self,
    *,
    attn_metadata: object,
    kv_cache_spec: Optional[object],
) -> None:
    """在 Triton metadata builder 阶段构建 step_cache + StepDecodeData。"""
    # GRAPH-WAR FENCE (event, GPU-side, no CPU block): wait for the prior decode FULL
    # cudagraph's producer reads (launch stream S) to finish before this data build
    # overwrites the RRP descriptor on its own (side) stream. One-way cudaStreamWaitEvent
    # -- no host block, no device drain. Shared armed-flag + event via sparse_constants;
    # armed only in the bootstrap window -> zero steady-state cost.
    from patches import sparse_constants as _war_sc
    if _war_sc._RRP_WAR_FENCE_ARMED[0]:
        _war_evt = _war_sc._RRP_WAR_FENCE_EVT[0]
        if _war_evt is not None:
            import torch as _war_torch
            if _war_torch.cuda.is_available():
                _war_torch.cuda.current_stream().wait_event(_war_evt)
    mb_profile_enabled = bool(_mb_profile_log_path())
    metadata_timing_enabled = bool(_metadata_timing_log_path())
    mb_total_start_ns = time.perf_counter_ns() if mb_profile_enabled else 0
    mb_phase_start_ns = mb_total_start_ns
    mb_phase_us: dict[str, float] = {}
    metadata_timing_total_start_ns = (
        time.perf_counter_ns() if metadata_timing_enabled else 0
    )
    metadata_timing_phase_start_ns = metadata_timing_total_start_ns
    metadata_timing_phase_us: dict[str, float] = {}
    metadata_timing_xlayer_phase_us: dict[str, float] = {}

    def _mark_mb_phase(name: str) -> None:
        nonlocal mb_phase_start_ns, metadata_timing_phase_start_ns
        now_ns = 0
        if mb_profile_enabled:
            now_ns = time.perf_counter_ns()
            mb_phase_us[name] = float(now_ns - mb_phase_start_ns) / 1000.0
            mb_phase_start_ns = now_ns
        if metadata_timing_enabled:
            if now_ns == 0:
                now_ns = time.perf_counter_ns()
            metadata_timing_phase_us[name] = (
                float(now_ns - metadata_timing_phase_start_ns) / 1000.0
            )
            metadata_timing_phase_start_ns = now_ns

    def _emit_metadata_timing(
        status: str,
        *,
        extra: Optional[dict[str, object]] = None,
    ) -> None:
        if not metadata_timing_enabled:
            return
        payload: dict[str, object] = {
            "event": "metadata_builder_timing",
            "status": str(status),
            "epoch": int(getattr(step_authority, "epoch", -1)),
            "batch_size": int(getattr(step_authority, "batch_size", -1)),
            "num_layers": int(len(getattr(self, "layer_cache_keys", ()))),
            "decode_only": bool(getattr(step_authority, "is_decode_only", False)),
            "has_decode_row": bool(
                getattr(
                    step_authority,
                    "has_decode_row",
                    getattr(step_authority, "is_decode_only", False),
                )
            ),
            "phase_us": dict(metadata_timing_phase_us),
            "xlayer_phase_us": dict(metadata_timing_xlayer_phase_us),
            "total_us": (
                float(time.perf_counter_ns() - metadata_timing_total_start_ns)
                / 1000.0
            ),
        }
        if extra:
            payload.update(extra)
        if os.environ.get("VLLM_SPARSE_REFRESH_CP_PROBE") == "1":
            try:
                payload["decode_runtime_mode"] = _decode_runtime_mode_for_controller(self).value
                payload["classify_reason"] = str(
                    getattr(getattr(self, "_decode_runtime_state", None), "last_reason", "")
                )
            except Exception:
                pass
        _append_metadata_timing(payload)

    previous_step_bound_meta = self.step_bound_meta
    step_meta = self.step_meta
    step_authority = self.step_authority
    self.step_bound_meta = None
    decode_runtime_launch_builder_call_count = 0
    decode_runtime_req_meta_pack_call_count = 0
    decode_runtime_layer_states_traversal_count = 0
    decode_runtime_carrier_update_kernel_count = 0
    decode_runtime_old_cache_probe_count = 0
    decode_runtime_adapter_invocation_count = 0
    decode_runtime_steady_delta_d2h_sync_count = 0
    decode_runtime_implicit_sync_count = 0
    # [ALLOC-COUNT-TRUTH 2026-07-11] 考古 2-10:此计数此前全仓从未自增=假遥测
    # (Gate C same_page 行"零分配"守门被恒 0 空转成假绿)。现接真值:计数本函数
    # 体内 torch 构造器语句(empty/zeros/ones/full/tensor/as_tensor/arange)的
    # 执行次数,含懒建持久缓冲与每步临时;视图/就地/算子临时(.ne/.to/.gt/
    # index_select)不逐个计数,但每个会分配的支路至少含一个计数点=branch 粒度
    # 完备。steady/ultra 快路径上报(:4158 附近)保持字面 0=结构性真值:快路径
    # 体内与其唯一 helper(apply_launch_template_row_delta,纯 setitem+copy_)
    # 均无构造器语句,已亲证。
    decode_runtime_per_step_allocation_count = 0
    if step_meta is None or step_authority is None:
        self._compact_recent_launch_template = None
        self.step_decode_plan_version = -1
        return
    # 单源约束：执行语义只能由 StepAuthority 提供。
    # metadata 不参与 decode/prefill 语义修正，避免双源漂移。

    # [p12 submarks 20260612] lever-1 §5.1 sub-phase 1/2: mb_preamble =
    # function entry -> just before the ultra-steady fast-path attempt. Splits
    # the legacy "resolve_kv_specs" window (entry -> first mark) without
    # renaming or moving that mark. Reuses the existing _mark_mb_phase closure
    # and its VLLM_SPARSE_MB_PROFILE_LOG / VLLM_SPARSE_METADATA_TIMING_LOG
    # gates; unarmed this is the same no-op path as every existing mark (zero
    # new env). NOTE: on ultra-HIT steps "ultra_steady_metadata_fast_path" now
    # excludes the preamble (it gains its own key); offline diffs against
    # pre-instrumentation runs must group the two keys.
    _mark_mb_phase("mb_preamble")
    if mb_profile_enabled:
        # [ultra-miss forensics] shadow attrs, fresh per step; snapshotted
        # right after an ultra miss below (before the second steady attempt
        # resets/overwrites the product miss-reason channels).
        self._forensic_ultra_first_miss_reason = ""
        self._forensic_ultra_first_rrp_miss_reason = ""

    if _try_run_ultra_steady_decode_metadata_fast_path(
        self,
        attn_metadata=attn_metadata,
        step_meta=step_meta,
        step_authority=step_authority,
        previous_step_bound_meta=previous_step_bound_meta,
        kv_cache_spec=kv_cache_spec,
        mb_profile_enabled=mb_profile_enabled,
        steady_phase_enabled=metadata_timing_enabled,
        mb_total_start_ns=mb_total_start_ns,
        mb_phase_us=mb_phase_us,
    ):
        _mark_mb_phase("ultra_steady_metadata_fast_path")
        if metadata_timing_enabled and mb_phase_us:
            metadata_timing_phase_us.update(mb_phase_us)
        _emit_metadata_timing(
            "ultra_steady_metadata_fast_path",
            extra={
                "reuse_decode_data": True,
                "full_decode_reuse_hit": True,
                "step_dispatch_plan_reused": True,
                "ordered_layer_data_reused": True,
                "ordered_step_decode_data_reused": True,
                "fast_path": "ultra_steady",
            },
        )
        return

    # [p12 submarks 20260612] lever-1 §5.1 sub-phase 2/2: the ultra-steady
    # fast-path attempt above returned False (miss). Mark the failed attempt
    # as its own phase so the "resolve_kv_specs" mark below (kept, unmoved)
    # shrinks to the true residual window: record_function entry + KV-spec
    # cache read. Same _mark_mb_phase machinery / same env gates; unarmed =
    # no-op, zero new env, no execution-semantics change.
    _mark_mb_phase("ultra_miss_probe")
    if mb_profile_enabled:
        # [ultra-miss forensics] capture the FIRST attempt's miss reasons.
        self._forensic_ultra_first_miss_reason = str(
            getattr(self, "_decode_runtime_steady_fast_path_miss_reason", "")
        )
        self._forensic_ultra_first_rrp_miss_reason = str(
            getattr(self, "_decode_runtime_rrp_step_state_miss_reason", "")
        )

    with _mb_record_function("sfi::mb.misc"):
        # 解析 KV cache 真实规格
        cached_kv_specs = _try_get_decode_runtime_cached_kv_specs(
            self,
            prefer_cache=bool(getattr(step_authority, "is_decode_only", False)),
        )
        if cached_kv_specs is not None:
            block_size_i, num_kv_heads_i, head_dim_i, kv_dtype_value = cached_kv_specs
        else:
            block_size_i = _resolve_positive_kv_block_size(
                self,
                kv_cache_spec=kv_cache_spec,
                step_meta=step_meta,
            )
            num_kv_heads_i = _resolve_positive_kv_int(
                self,
                kv_cache_spec=kv_cache_spec,
                spec_attr="num_kv_heads",
                controller_attr="kv_cache_num_kv_heads",
                layer_attr="num_kv_heads",
            )
            head_dim_i = _resolve_positive_kv_int(
                self,
                kv_cache_spec=kv_cache_spec,
                spec_attr="head_size",
                controller_attr="kv_cache_head_dim",
                layer_attr="head_dim",
            )
            kv_dtype_value = _resolve_kv_cache_dtype(
                self,
                kv_cache_spec=kv_cache_spec,
            )
            self._decode_runtime_kv_specs = (
                int(block_size_i),
                int(num_kv_heads_i),
                int(head_dim_i),
                kv_dtype_value,
            )
    _mark_mb_phase("resolve_kv_specs")

    with _mb_record_function("sfi::mb.seqused_write"):
        batch_size = int(step_authority.batch_size)
        bound_seq_lens_no_copy = bool(
            getattr(step_authority, "is_decode_only", False)
        ) and _try_bind_decode_seq_lens_source_no_copy(
            attn_metadata=attn_metadata,
            step_meta=step_meta,
            batch_size=batch_size,
        )
        if not bound_seq_lens_no_copy:
            # 复用 metadata 中的 seq_lens（GPU tensor，避免 per-step torch.tensor(list)）
            seq_lens = getattr(attn_metadata, "seq_lens", None)
            if isinstance(seq_lens, torch.Tensor):
                if seq_lens.dtype != torch.int32:
                    seq_lens = seq_lens.to(dtype=torch.int32)
                if seq_lens.numel() < batch_size and step_authority.is_decode_only:
                    raise RuntimeError(
                        "decode metadata mismatch: seq_lens shorter than batch_size "
                        f"(seq_lens={int(seq_lens.numel())}, batch_size={batch_size})"
                    )
                if seq_lens.numel() > batch_size:
                    seq_lens = seq_lens[:batch_size]
                step_meta.seqused_k_gpu = seq_lens
            bind_device = (
                seq_lens.device
                if isinstance(seq_lens, torch.Tensor)
                else step_meta.seqused_k_gpu.device
                if isinstance(step_meta.seqused_k_gpu, torch.Tensor)
                else next(iter(self.layer_states.values())).device
                if self.layer_states
                else torch.device("cpu")
            )
            _bind_canonical_real_kv_len(
                step_meta=step_meta,
                seq_lens=seq_lens if isinstance(seq_lens, torch.Tensor) else None,
                batch_size=batch_size,
                device=bind_device,
            )
    _mark_mb_phase("seqused_write")

    with _mb_record_function("sfi::mb.misc"):
        if not self.layer_states:
            self.step_decode_cache_key = None
            self.step_decode_plan_version = -1
            self.prefill_global_meta_epoch = -1
            self.prefill_global_meta_handle_id = -1
            self.prefill_global_meta_handle_generation = -1
            return

        # layer_index_by_cache_key 既用于 decode 的全局 meta，也用于 prefill 的跨层 pack；
        # 因此这里在 metadata builder 阶段统一校验/重建一次。
        rebuild_index = False
        if not self.layer_cache_keys:
            rebuild_index = True
        elif len(self.layer_cache_keys) != len(self.layer_states):
            rebuild_index = True
        else:
            for idx, key in enumerate(self.layer_cache_keys):
                if self.layer_index_by_cache_key.get(key, -1) != idx:
                    rebuild_index = True
                    break
            if not rebuild_index:
                for key in self.layer_states.keys():
                    if key not in self.layer_index_by_cache_key:
                        rebuild_index = True
                        break

        if rebuild_index:
            ordered_keys = [key for key in self.layer_cache_keys if key in self.layer_states]
            for key in self.layer_states.keys():
                if key not in ordered_keys:
                    ordered_keys.append(key)
            if not ordered_keys:
                self.step_decode_cache_key = None
                self.step_decode_plan_version = -1
                self.step_dispatch_plan = None
                return
            self.layer_cache_keys = ordered_keys
            self.layer_index_by_cache_key = {key: idx for idx, key in enumerate(ordered_keys)}
            self.step_decode_data = None
            self.step_decode_cache_key = None
            self.step_decode_plan_version = -1
            self._step_decode_spec_key = None
            self.step_dispatch_plan = None
            # layer_index 重建后同步刷新缓存（避免热路径重复 map）
            self._refresh_layer_index_cache()

        step_ctx = self.step_context
        if step_ctx is None or step_ctx.epoch != step_authority.epoch:
            raise RuntimeError(
                "metadata prebuild requires current step_context; "
                f"step_epoch={step_authority.epoch} ctx_epoch="
                f"{-1 if step_ctx is None else step_ctx.epoch}"
            )
        batch_size = int(step_authority.batch_size)
        if (
            bool(getattr(getattr(self, "config", None), "one_shot_bootstrap_only", False))
            and bool(getattr(step_authority, "is_decode_only", False))
        ):
            req_ids = tuple(str(rid) for rid in step_authority.req_ids[:batch_size])
            (
                bridge_not_ready,
                blocked_not_ready,
                decode_phase,
            ) = classify_one_shot_bootstrap_decode_guard(
                req_ids,
                compact_ready_all_layers=self._request_compact_ready_all_layers,
                can_bridge_bootstrap_decode=self._request_can_bridge_bootstrap_decode,
            )
            setattr(
                self,
                "_one_shot_bootstrap_bridge_active_request_ids",
                tuple(bridge_not_ready),
            )
            setattr(self, "_one_shot_bootstrap_decode_phase", decode_phase)
            if blocked_not_ready:
                blocked_details = []
                request_states = getattr(self, "request_states", {})
                for blocked_rid in blocked_not_ready:
                    tracking = (
                        request_states.get(str(blocked_rid))
                        if isinstance(request_states, dict)
                        else None
                    )
                    ready_state = getattr(tracking, "producer_ready_state", None)
                    blocked_details.append(
                        {
                            "rid": str(blocked_rid),
                            "bootstrap_pending": bool(
                                getattr(tracking, "bootstrap_pending", False)
                            ),
                            "bootstrap_done": bool(
                                getattr(tracking, "bootstrap_done", False)
                            ),
                            "bridge_active": bool(
                                getattr(tracking, "bootstrap_bridge_active", False)
                            ),
                            "bridge_token_count": int(
                                getattr(tracking, "bridge_token_count", 0) or 0
                            ),
                            "bridge_max_tokens": int(
                                getattr(tracking, "bridge_max_tokens", 0) or 0
                            ),
                            "deferred_job_present": getattr(
                                tracking,
                                "deferred_producer_job",
                                None,
                            )
                            is not None,
                            "producer_ready_state_present": ready_state is not None,
                            "producer_final_event_present": (
                                ready_state is not None
                                and getattr(ready_state, "final_event", None) is not None
                            ),
                        }
                    )
                if not _ONE_SHOT_BLOCKED_DENSE_FALLBACK_CACHED:
                    raise RuntimeError(
                        "one-shot bootstrap graph decode requires compact_ready before replay: "
                        + ", ".join(blocked_not_ready)
                        + f"; details={blocked_details!r}"
                    )
                # ROBUST MODE: a not-compact-ready row has full native paged-KV; only the
                # sparse compact buffer is absent. resolve_decode_row_policy (row_policy.py:76)
                # already routes bootstrap_done=False rows to _ROW_MODE_DENSE (full-KV) every
                # step, unconditionally; this guard's raise is a pre-emptive tripwire firing
                # BEFORE that correct dense path. Demote (do NOT crash) -> decode dense until
                # compact_ready flips it to sparse. No bridge-accept (would pollute the bridge
                # bookkeeping of a never-bridged request). Scoped to this guard; selector
                # key_norms path + one_shot_bootstrap_only untouched.
                if not getattr(self, "_one_shot_blocked_dense_warned", False):
                    _log.warning(
                        "one-shot decode: %d not-compact-ready row(s) demoted to dense "
                        "full-KV (will switch to sparse when compact ready); details=%r",
                        len(blocked_not_ready), blocked_details,
                    )
                    self._one_shot_blocked_dense_warned = True
            if bridge_not_ready:
                self._mark_bridge_decode_metadata_accepted(
                    req_ids=tuple(bridge_not_ready),
                    epoch=int(step_authority.epoch),
                )
        q_start_loc = step_authority.q_start_loc[: batch_size + 1]
        if len(q_start_loc) < batch_size + 1:
            raise RuntimeError(
                "metadata prebuild q_start_loc coverage mismatch; "
                f"q_start={len(q_start_loc)} batch={batch_size}"
            )
        if (
            len(step_authority.q_lens_by_row) < batch_size
            or len(step_authority.context_kv_len_by_row) < batch_size
            or len(step_authority.logits_last_n_by_row) < batch_size
            or len(step_authority.logits_capacity_by_row) < batch_size
            or len(step_authority.logf_mask_by_row) < batch_size
        ):
            raise RuntimeError(
                "metadata prebuild step_authority row coverage mismatch; "
                f"q_lens={len(step_authority.q_lens_by_row)} "
                f"context={len(step_authority.context_kv_len_by_row)} "
                f"last_n={len(step_authority.logits_last_n_by_row)} "
                f"cap={len(step_authority.logits_capacity_by_row)} "
                f"mask={len(step_authority.logf_mask_by_row)} "
                f"batch={batch_size}"
            )
        layer_effective_refresh_by_row_step = step_authority.layer_effective_refresh_by_row
        if len(layer_effective_refresh_by_row_step) < step_authority.batch_size:
            raise RuntimeError(
                "metadata prebuild requires full layer_effective_refresh_by_row coverage; "
                f"rows={len(layer_effective_refresh_by_row_step)} batch={step_authority.batch_size}"
            )
        _fp_hit_result = _try_run_steady_decode_metadata_fast_path(
            self,
            attn_metadata=attn_metadata,
            step_meta=step_meta,
            step_authority=step_authority,
            previous_step_bound_meta=previous_step_bound_meta,
            kv_cache_spec=kv_cache_spec,
            block_size=int(block_size_i),
            num_kv_heads=int(num_kv_heads_i),
            q_start_loc=q_start_loc,
            layer_effective_refresh_by_row=layer_effective_refresh_by_row_step,
            mb_profile_enabled=mb_profile_enabled,
            steady_phase_enabled=metadata_timing_enabled,
            mb_total_start_ns=mb_total_start_ns,
            mb_phase_us=mb_phase_us,
        )
        if _fp_hit_result:
            _mark_mb_phase("steady_metadata_fast_path")
            if metadata_timing_enabled and mb_phase_us:
                metadata_timing_phase_us.update(mb_phase_us)
            _emit_metadata_timing(
                "steady_metadata_fast_path",
                extra={
                    "reuse_decode_data": True,
                    "full_decode_reuse_hit": True,
                    "step_dispatch_plan_reused": True,
                    "ordered_layer_data_reused": True,
                    "ordered_step_decode_data_reused": True,
                    "fast_path": "steady",
                },
            )
            return
        prefill_rows = tuple(
            int(row) for row in step_authority.prefill_rows if 0 <= int(row) < batch_size
        )
        # 先发布 row-only bound meta，供 prefill-global-meta 阶段的 capture_layout 单源消费；
        # layer_bound 会在后续 decode/prefill req_meta 打包完成后补齐。
        self.step_bound_meta = StepBoundMeta(
            step_handle_id=int(step_authority.step_handle_id),
            step_handle_generation=int(step_authority.step_handle_generation),
            epoch=int(step_authority.epoch),
            batch_size=batch_size,
            q_start_loc=q_start_loc,
            q_lens_by_row=tuple(int(v) for v in step_authority.q_lens_by_row[:batch_size]),
            context_kv_len_by_row=tuple(
                int(v) for v in step_authority.context_kv_len_by_row[:batch_size]
            ),
            logits_last_n_by_row=tuple(
                int(v) for v in step_authority.logits_last_n_by_row[:batch_size]
            ),
            logits_capacity_by_row=tuple(
                int(v) for v in step_authority.logits_capacity_by_row[:batch_size]
            ),
            logf_mask_by_row=tuple(int(v) for v in step_authority.logf_mask_by_row[:batch_size]),
            logf_attn_rows=tuple(
                int(row) for row in step_authority.logf_attn_rows if 0 <= int(row) < batch_size
            ),
            prefill_rows=prefill_rows,
            layer_bound=tuple(),
            recent_cap=int(step_authority.recent_cap),
            sink_tokens=int(step_authority.sink_tokens),
            canonical_real_kv_len_cpu=tuple(
                int(v) for v in step_authority.context_kv_len_by_row[:batch_size]
            ),
            recent_descriptor_block_size=int(
                getattr(step_meta, "recent_descriptor_block_size", 0)
            ),
            request_recent_first_logical_page=tuple(
                int(v)
                for v in tuple(
                    getattr(step_meta, "request_recent_first_logical_page", tuple())
                )[:batch_size]
            ),
            request_recent_page_count=tuple(
                int(v)
                for v in tuple(
                    getattr(step_meta, "request_recent_page_count", tuple())
                )[:batch_size]
            ),
            request_recent_epoch=int(getattr(step_meta, "request_recent_epoch", -1)),
            request_kv_rows=tuple(
                int(v)
                for v in tuple(getattr(step_meta, "request_kv_rows", tuple()))[
                    :batch_size
                ]
            ),
            req_set_hash=int(step_authority.req_set_hash),
            row_phase_hash=int(step_authority.row_phase_hash),
            plan_signature=tuple(step_authority.plan_signature),
            bound_meta_signature=(
                int(step_authority.epoch),
                int(step_authority.step_handle_id),
                int(step_authority.step_handle_generation),
                tuple(step_authority.plan_signature),
                tuple(int(v) for v in step_authority.logits_last_n_by_row[:batch_size]),
                tuple(int(v) for v in step_authority.logits_capacity_by_row[:batch_size]),
                tuple(int(v) for v in step_authority.logf_mask_by_row[:batch_size]),
                tuple(int(v) for v in q_start_loc),
                tuple(int(v) for v in prefill_rows),
            ),
        )
        _mark_mb_phase("step_bound_meta_prepare")

        # 非 decode-only 步也要统一走 step-bound 预构建（单写者），
        # prefill_global_meta 仅作为 pure-prefill 的跨层 req_meta 物料保留。
        if not step_authority.is_decode_only:
            try:
                self._maybe_build_step_prefill_global_meta_from_metadata(
                    attn_metadata=attn_metadata,
                    kv_cache_spec=kv_cache_spec,
                )
            except Exception as exc:
                self.prefill_global_meta_epoch = -1
                self.prefill_global_meta_handle_id = -1
                self.prefill_global_meta_handle_generation = -1
                raise RuntimeError(
                    "prefill global meta build failed: "
                    f"epoch={step_authority.epoch} batch={step_authority.batch_size}"
                ) from exc
        _mark_mb_phase("prefill_global_meta_build")
        # [T3-FORENSIC 2026-07-10] 窗口0:launch_plan_build 段(每主体步)。
        # 盖住分类 collect(_collect_decode_delta_packet)与 FULL_RECOMPILE
        # seed 重算的双算面;累积进窗口1 同一 profiler,dump 复用窗口1 的
        # 落盘点(下一 FULL_RECOMPILE 步),env 未设=永不构造零税。
        _mb_lpb_prof = _mb_commit_cprofile() if _mb_cprofile_dir() else None
        if _mb_lpb_prof is not None:
            _mb_lpb_prof.enable()
        # CompactRecentLaunchPlan is the step-level source for compact layout
        # generation. Build it once before the decode cache key so the key does
        # not walk per-layer compact epochs.
        with _mb_record_function("sfi::mb.launch_plan_build"):
            _plan_bind_device: Optional[torch.device] = None
            if self.layer_states:
                _plan_bind_device = next(iter(self.layer_states.values())).device
            _plan_page_size = int(block_size_i) if block_size_i else 0
            _launch_template_ready = False
            _used_launch_template_delta = False
            if _plan_bind_device is not None and self.step_bound_meta is not None:
                _launch_template = getattr(
                    self, "_compact_recent_launch_template", None
                )
                _existing_plan = (
                    _launch_template.plan
                    if isinstance(_launch_template, LaunchTemplate)
                    else None
                )
                if (
                    isinstance(_launch_template, LaunchTemplate)
                    and _existing_plan is not None
                    and bool(getattr(_existing_plan, "valid", False))
                ):
                    _launch_template_ready = True
                    try:
                        _ensure_current_recent_descriptors_for_launch_template(
                            step_bound_meta=self.step_bound_meta,
                            step_authority=step_authority,
                            page_size=int(_plan_page_size),
                        )
                        _decode_delta = _collect_decode_delta_packet(
                            step_authority=step_authority,
                            step_bound_meta=self.step_bound_meta,
                            launch_template=_launch_template,
                        )
                        _decode_guard = _build_decode_static_guard_for_launch_template(
                            controller=self,
                            attn_metadata=attn_metadata,
                            step_authority=step_authority,
                            existing_plan=_existing_plan,
                            block_size=int(_plan_page_size),
                            num_kv_heads=int(num_kv_heads_i),
                        )
                        _decode_runtime_state = getattr(
                            self, "_decode_runtime_state", None
                        )
                        if not isinstance(
                            _decode_runtime_state, DecodeRuntimeState
                        ):
                            _decode_runtime_state = DecodeRuntimeState()
                            self._decode_runtime_state = _decode_runtime_state
                        _decode_runtime_mode, _decode_runtime_reason = (
                            _decode_runtime_state.update_classification(
                                _decode_guard,
                                _decode_delta,
                            )
                        )
                        self._decode_runtime_mode = _decode_runtime_mode
                        self._decode_runtime_delta = _decode_delta
                        self._decode_runtime_reason = _decode_runtime_reason
                        if not _should_build_compact_recent_launch_plan(
                            _decode_runtime_mode,
                            launch_template_ready=_launch_template_ready,
                        ):
                            _template_update = apply_launch_template_row_delta(
                                _launch_template,
                                request_recent_len_by_row=(
                                    _decode_delta.request_recent_len_by_row
                                ),
                                launch_effective_k_by_row=(
                                    _decode_delta.launch_effective_k_by_row
                                ),
                                recent_first_page_by_row=(
                                    _decode_delta.recent_first_page_by_row
                                ),
                                recent_page_count_by_row=(
                                    _decode_delta.recent_page_count_by_row
                                ),
                                update_gpu=_should_update_launch_template_gpu_for_decode_delta(
                                    _decode_runtime_mode
                                ),
                                # [ARM-WAR-R1-PINNED-INDEPENDENT 2026-07-12]
                                # fresh 镜像双引用替换。
                                controller=self,
                            )
                            self._decode_runtime_template_update_reason = (
                                _template_update.reason
                            )
                            if bool(_template_update.requires_recompile):
                                _reset_decode_runtime_full_recompile(
                                    self,
                                    "launch_template_delta_requires_recompile:"
                                    f"{_template_update.reason}",
                                )
                            else:
                                self.step_bound_meta.compact_recent_launch_plan = (
                                    _launch_template.plan
                                )
                                self._decode_runtime_launch_template_update_count = (
                                    int(
                                        getattr(
                                            self,
                                            "_decode_runtime_launch_template_update_count",
                                            0,
                                        )
                                    )
                                    + 1
                                )
                                _decode_runtime_state.counters.carrier_update_kernel_count += int(
                                    _template_update.carrier_update_kernel_count
                                )
                                decode_runtime_carrier_update_kernel_count += int(
                                    _template_update.carrier_update_kernel_count
                                )
                                _used_launch_template_delta = True
                    except Exception as exc:
                        _reset_decode_runtime_full_recompile(
                            self,
                            "launch_template_delta_failed:"
                            f"{type(exc).__name__}:{exc}",
                        )
                if not _used_launch_template_delta:
                    from patches.decode_runtime.compact_recent_launch_plan_builder import (
                        build_compact_recent_launch_plan,
                    )

                    _decode_runtime_state = getattr(
                        self, "_decode_runtime_state", None
                    )
                    if isinstance(_decode_runtime_state, DecodeRuntimeState):
                        _decode_runtime_state.counters.slow_builder_call_count += 1
                    decode_runtime_launch_builder_call_count += 1
                    self.step_bound_meta.compact_recent_launch_plan = (
                        build_compact_recent_launch_plan(
                            controller=self,
                            step_authority=step_authority,
                            step_bound_meta=self.step_bound_meta,
                            page_size=_plan_page_size,
                            device=_plan_bind_device,
                        )
                    )
                    _launch_plan = self.step_bound_meta.compact_recent_launch_plan
                    _launch_template = None
                    _descriptor_cpu_i32 = getattr(
                        self, "_compact_recent_launch_plan_descriptor_cpu_i32", None
                    )
                    _descriptor_gpu_i32 = getattr(
                        self, "_compact_recent_launch_plan_descriptor_i32", None
                    )
                    if (
                        _launch_plan is not None
                        and bool(getattr(_launch_plan, "valid", False))
                        and isinstance(_descriptor_cpu_i32, torch.Tensor)
                        and isinstance(_descriptor_gpu_i32, torch.Tensor)
                    ):
                        _launch_template = compile_launch_template(
                            _launch_plan,
                            descriptor_cpu_i32=_descriptor_cpu_i32,
                            descriptor_gpu_i32=_descriptor_gpu_i32,
                        )
                    self._compact_recent_launch_template = _launch_template
                    if _launch_template is not None:
                        _seed_decode_runtime_state_after_full_recompile(
                            self,
                            attn_metadata=attn_metadata,
                            step_authority=step_authority,
                            launch_template=_launch_template,
                            block_size=int(_plan_page_size),
                            num_kv_heads=int(num_kv_heads_i),
                            reason=str(
                                getattr(self, "_decode_runtime_reason", "full_recompile")
                            ),
                        )
        # [T3-FORENSIC] 窗口0关闭(与窗口1 无重叠:窗口1 enable 在本函数
        # 更下游的 FULL_RECOMPILE 分支)。
        if _mb_lpb_prof is not None:
            _mb_lpb_prof.disable()
        _mark_mb_phase("launch_plan_build")

        _compact_launch_plan = (
            self.step_bound_meta.compact_recent_launch_plan
            if self.step_bound_meta is not None
            else None
        )
        compact_layout_generation = int(
            getattr(_compact_launch_plan, "compact_meta_epoch", -1)
        )
        step_decode_cache_key = _make_step_decode_cache_key(
            step_authority,
            layer_cache_keys=self.layer_cache_keys,
            layer_effective_refresh_by_row=layer_effective_refresh_by_row_step,
            compact_layout_generation=compact_layout_generation,
        )
        self.step_decode_cache_key = step_decode_cache_key
        self.step_decode_plan_version = int(getattr(step_authority, "decode_plan_version", -1))

        force_dense = _FORCE_DENSE_CACHED
        force_compact_off = _FORCE_COMPACT_OFF_CACHED
        if force_dense:
            return
        if force_compact_off:
            self._build_step_dispatch_plan(step_meta, cache_key=step_decode_cache_key)
            return

        # 规格不完整时直接跳过，避免错误 meta
        if (
            block_size_i is None
            or num_kv_heads_i is None
            or head_dim_i is None
            or kv_dtype_value is None
        ):
            self.step_decode_cache_key = None
            self.step_decode_plan_version = -1
            self.step_dispatch_plan = None
            return

        spec_key = (
            block_size_i,
            num_kv_heads_i,
            head_dim_i,
            kv_dtype_value,
        )
    _mark_mb_phase("step_bound_and_cache_key")

    with _mb_record_function("sfi::mb.logf_plan_staging"):
        # ============ decode step-wise log_f plan（StepAuthority 单源）============
        capacity_by_row = step_authority.logits_capacity_by_row
        q_lens_by_row = step_authority.q_lens_by_row
        mask_by_row = tuple(
            int(v) for v in step_authority.logf_mask_by_row[:batch_size]
        )
        max_batch = step_authority.max_batch_size
        decode_logf_attn_rows = tuple(
            row for row in step_authority.logf_attn_rows if 0 <= row < batch_size
        )
        decode_logf_stride_head = int(step_authority.logf_stride_head)
        decode_logf_max_kv = max((int(v) for v in capacity_by_row[:batch_size]), default=0)
        if decode_logf_max_kv > 0 and decode_logf_stride_head <= 0:
            raise RuntimeError(
                "step_authority decode invalid stride_head for non-zero max_kv: "
                f"stride_head={decode_logf_stride_head} max_kv={decode_logf_max_kv}"
            )
        _sit = int(step_authority.step_identity_token)
        cap_device = (
            step_meta.seqused_k_gpu.device
            if step_meta.seqused_k_gpu is not None
            else next(iter(self.layer_states.values())).device
        )
        cap_device_key = (
            str(cap_device.type),
            -1 if cap_device.index is None else int(cap_device.index),
        )
        decode_logf_stage_signature: Tuple[object, ...] = (
            int(_sit),
            int(batch_size),
            int(max_batch),
            tuple(int(v) for v in capacity_by_row[:batch_size]),
            tuple(int(v) for v in q_lens_by_row[:batch_size]),
            tuple(int(v) for v in mask_by_row),
            tuple(int(v) for v in decode_logf_attn_rows),
            int(decode_logf_stride_head),
            int(decode_logf_max_kv),
            cap_device_key,
        )
        if not decode_logf_attn_rows:
            device = cap_device
            _ensure_decode_logf_no_rows_buffers(
                self,
                device=device,
                batch_size=batch_size,
                max_batch=max_batch,
            )
            self._decode_logf_stage_token = _sit
            self._decode_logf_stage_signature = decode_logf_stage_signature
            self._decode_logf_stage_bound_signature = None
        elif (
            int(getattr(self, "_decode_logf_stage_token", -1)) != _sit
            or getattr(self, "_decode_logf_stage_signature", None)
            != decode_logf_stage_signature
        ):
            def _ensure_decode_cpu_stage(
                name: str,
                dtype: torch.dtype,
                *,
                init_fill: Optional[int] = None,
            ) -> torch.Tensor:
                nonlocal decode_runtime_per_step_allocation_count
                buf = getattr(self, name, None)
                if (
                    buf is None
                    or not isinstance(buf, torch.Tensor)
                    or buf.device.type != "cpu"
                    or buf.dtype != dtype
                    or buf.numel() < max_batch
                ):
                    buf = torch.empty((max_batch,), device="cpu", dtype=dtype, pin_memory=True)
                    decode_runtime_per_step_allocation_count += 1
                    if init_fill is not None:
                        buf.fill_(int(init_fill))
                    setattr(self, name, buf)
                return buf

            cap_cpu_stage = _ensure_decode_cpu_stage("_decode_logits_cap_cpu_i64", torch.long)[:batch_size]
            last_n_cpu_stage = _ensure_decode_cpu_stage("_decode_logits_last_n_cpu_i64", torch.long)[:batch_size]
            mask_cpu_stage = _ensure_decode_cpu_stage("_decode_log_f_mask_cpu_i32", torch.int32)[:batch_size]
            q_lens_cpu_stage = _ensure_decode_cpu_stage("_decode_q_lens_cpu_i32", torch.int32)[:batch_size]
            cap_cpu_stage.copy_(
                torch.as_tensor(capacity_by_row[:batch_size], dtype=torch.long).clamp_min_(0)
            )
            q_lens_cpu_stage.copy_(
                torch.as_tensor(q_lens_by_row[:batch_size], dtype=torch.int32).clamp_min_(0)
            )
            mask_cpu_stage.copy_(
                torch.as_tensor(mask_by_row, dtype=torch.int32).ne(0).to(dtype=torch.int32)
            )
            last_n_cpu_stage.copy_(cap_cpu_stage.gt(0).to(dtype=torch.long))
            # [ALLOC-COUNT-TRUTH] 三个 as_tensor 每步临时(上方 cap/q_lens/mask)。
            decode_runtime_per_step_allocation_count += 3
            cap_storage_rebuilt = False
            if (
                self._decode_logits_cap_i64 is None
                or self._decode_logits_cap_i64.device != cap_device
                or self._decode_logits_cap_i64.numel() < max_batch
            ):
                self._decode_logits_cap_i64 = torch.empty((max_batch,), device=cap_device, dtype=torch.long)
                decode_runtime_per_step_allocation_count += 1
                cap_storage_rebuilt = True
            last_n_storage_rebuilt = False
            if (
                self._decode_logits_last_n_i64 is None
                or self._decode_logits_last_n_i64.device != cap_device
                or self._decode_logits_last_n_i64.numel() < max_batch
            ):
                self._decode_logits_last_n_i64 = torch.empty((max_batch,), device=cap_device, dtype=torch.long)
                decode_runtime_per_step_allocation_count += 1
                last_n_storage_rebuilt = True
            device = cap_device
            mask_storage_rebuilt = False
            if (
                self._decode_log_f_mask_i32 is None
                or self._decode_log_f_mask_i32.device != device
                or self._decode_log_f_mask_i32.numel() < max_batch
            ):
                self._decode_log_f_mask_i32 = torch.empty((max_batch,), device=device, dtype=torch.int32)
                decode_runtime_per_step_allocation_count += 1
                mask_storage_rebuilt = True
            q_lens_storage_rebuilt = False
            if (
                self._decode_q_lens_i32 is None
                or self._decode_q_lens_i32.device != device
                or self._decode_q_lens_i32.numel() < max_batch
            ):
                self._decode_q_lens_i32 = torch.empty((max_batch,), device=device, dtype=torch.int32)
                decode_runtime_per_step_allocation_count += 1
                q_lens_storage_rebuilt = True

            if (
                cap_storage_rebuilt
                or last_n_storage_rebuilt
                or mask_storage_rebuilt
                or q_lens_storage_rebuilt
            ):
                dirty_rows = torch.arange(batch_size, device="cpu", dtype=torch.long)
            else:
                dirty_rows_tuple = tuple(
                    row for row in step_authority.logf_dirty_rows if 0 <= row < batch_size
                )
                if dirty_rows_tuple:
                    dirty_rows = torch.as_tensor(dirty_rows_tuple, dtype=torch.long, device="cpu")
                else:
                    dirty_rows = torch.empty((0,), dtype=torch.long, device="cpu")
            # [ALLOC-COUNT-TRUTH] dirty_rows 每步临时(上方三臂各恰一次构造)。
            decode_runtime_per_step_allocation_count += 1

            if dirty_rows.numel() > 0:
                cap_rows = dirty_rows.to(device=cap_device, dtype=torch.long, non_blocking=True)
                self._decode_logits_cap_i64.index_copy_(
                    0,
                    cap_rows,
                    cap_cpu_stage.index_select(0, dirty_rows).to(
                        device=cap_device, dtype=torch.long, non_blocking=True
                    ),
                )
                self._decode_logits_last_n_i64.index_copy_(
                    0,
                    cap_rows,
                    last_n_cpu_stage.index_select(0, dirty_rows).to(
                        device=cap_device, dtype=torch.long, non_blocking=True
                    ),
                )
                mask_rows = dirty_rows.to(device=device, dtype=torch.long, non_blocking=True)
                self._decode_log_f_mask_i32.index_copy_(
                    0,
                    mask_rows,
                    mask_cpu_stage.index_select(0, dirty_rows).to(
                        device=device, dtype=torch.int32, non_blocking=True
                    ),
                )
                self._decode_q_lens_i32.index_copy_(
                    0,
                    mask_rows,
                    q_lens_cpu_stage.index_select(0, dirty_rows).to(
                        device=device, dtype=torch.int32, non_blocking=True
                    ),
                )

            if batch_size < max_batch:
                self._decode_log_f_mask_i32[batch_size:max_batch].zero_()
                self._decode_q_lens_i32[batch_size:max_batch].fill_(1)
            self._decode_log_f_mask_may_be_nonzero = any(
                int(v) != 0 for v in mask_by_row[:batch_size]
            )
            self._decode_q_lens_may_be_nonone = any(
                int(v) != 1 for v in q_lens_by_row[:batch_size]
            )
            self._decode_logf_stage_token = _sit
            self._decode_logf_stage_signature = decode_logf_stage_signature
            self._decode_logf_stage_bound_signature = None
    _mark_mb_phase("logf_plan_staging")

    # [T2-FORENSIC] commit 步窗口1:layer_state_refresh→dispatch_plan_build。
    # rrp_bind 同步吸收段被窗口排除;env 未设=None,零判速税。
    _mb_commit_prof = None
    if _mb_cprofile_dir() and (
        _decode_runtime_mode_for_controller(self) is DecodeRuntimeMode.FULL_RECOMPILE
    ):
        _mb_commit_prof = _mb_commit_cprofile()
        if _mb_commit_prof is not None:
            _mb_commit_prof.enable()

    with _mb_record_function("sfi::mb.misc"):
        # seqused_k_gpu 应由 build_for_sparse 从 attn_metadata.seq_lens 赋值（line 127）；
        # 回退路径的同步 CPU→GPU copy 已移除——若仍为 None 则 fail-fast 暴露根因。
        if step_meta.seqused_k_gpu is None and self.layer_states:
            raise RuntimeError(
                "seqused_k_gpu unexpectedly None after build_for_sparse; "
                f"epoch={step_authority.epoch} batch={step_authority.batch_size}"
            )

        global_slot_map_step = self.get_step_global_slot_map(step_authority.req_ids)
        global_slot_signature_step = _stable_slot_signature64(
            global_slot_map_step,
            step_authority.req_ids,
        )
        has_refresh_rows = any(layer_effective_refresh_by_row_step)
        layer_group_enabled = bool(self._refresh_layer_group_enabled)
        layer_group_active = self._refresh_layer_group_active
        layer_group_epoch = self._refresh_layer_group_epoch
        layer_group_event_idx = self._refresh_layer_group_event_idx
        decode_plan_version = int(getattr(step_authority, "decode_plan_version", -1))
        # Pre-compute refresh signature before the loop to avoid per-layer normalize calls.
        _batch_size_i = int(step_authority.batch_size)
        _refresh_sig_active = normalize_layer_effective_refresh_signature(
            layer_effective_refresh_by_row_step, batch_size=_batch_size_i,
        )
        # Pre-build the step-constant parts of the cache key prefix.
        _sa_req_ids = step_authority.req_ids
        _sa_slot_by_row = step_authority.slot_by_row
        _sa_row_mode = step_authority.row_mode_by_row
        if len(_sa_slot_by_row) < _batch_size_i:
            raise RuntimeError(
                "metadata prebuild requires full slot_by_row coverage; "
                f"slots={len(_sa_slot_by_row)} batch={_batch_size_i}"
            )
        if len(_sa_row_mode) < _batch_size_i:
            raise RuntimeError(
                "metadata prebuild requires full row_mode_by_row coverage; "
                f"modes={len(_sa_row_mode)} batch={_batch_size_i}"
            )
        _sa_slot_sig = (
            _sa_slot_by_row
            if isinstance(_sa_slot_by_row, tuple) and len(_sa_slot_by_row) == _batch_size_i
            else tuple(int(_sa_slot_by_row[_idx]) for _idx in range(_batch_size_i))
        )
        _sa_row_mode_sig = (
            _sa_row_mode
            if isinstance(_sa_row_mode, tuple) and len(_sa_row_mode) == _batch_size_i
            else tuple(int(_sa_row_mode[_idx]) for _idx in range(_batch_size_i))
        )
        _sa_bootstrap_done = step_authority.bootstrap_done_by_row
        _sa_short_dense = step_authority.short_dense_by_row
        _ck_prefix_active = (
            _sa_req_ids,
            _sa_slot_sig,
            _sa_row_mode_sig,
            _sa_bootstrap_done,
            _sa_short_dense,
            _refresh_sig_active,
        )
        _attn_mode_peripheral = str(
            getattr(getattr(self, "config", None), "attn_mode", "compact_recent")
        )
        _skip_page_sparse_refresh = should_skip_page_sparse_state(_attn_mode_peripheral)

        skip_layer_state_refresh = _can_skip_layer_state_refresh_for_step(
            self,
            step_authority=step_authority,
            step_decode_cache_key=step_decode_cache_key,
        )
        pure_prefill_step = (
            _batch_size_i > 0
            and all(
                bool(v)
                for v in tuple(step_authority.is_prefill_by_row)[:_batch_size_i]
            )
        )
        # [FASTEST-PATH-SWEEP 2026-07-10] VLLM_SPARSE_ABLATE_LAYER_REFRESH
        # ablation 旋钮(每步活 env 读)与 VLLM_SPARSE_LSR_PROBE 探针
        # (每步 env 读+每层 5 站点残税)已删——LSR 取证使命被
        # VLLM_SPARSE_MB_CPROFILE_DIR(函数级 ncalls+分相,严格更强)接替,
        # 破案后拆;ablation 属实验废墟,无旋钮原则。
        # VLLM_SPARSE_ABLATE_FORCE_REUSE(强制 reuse_decode_data 命中的 ablation
        # 旋钮,每步活 env 读)同批删除 2026-07-10,同属实验废墟。
        layer_states_for_refresh = (
            tuple()
            if skip_layer_state_refresh or pure_prefill_step
            else tuple(self.layer_states.values())
        )
        decode_runtime_layer_states_traversal_count += int(
            len(layer_states_for_refresh)
        )
        # P12 lever-2 (2026-06-12): cross-layer slot-state consistency check
        # HOISTED from after the loop (same call, same raise semantics; runs
        # on every non-fast-path step exactly as before, loop empty or not).
        # It must run BEFORE the loop because the loop computes active_slots
        # once from the first post-align layer and threads it into every
        # layer's launch-view refresh, where it gates static-carrier
        # invalidation and is persisted into per-layer fields
        # (_step_cache_selected_static_slots). Divergence must abort before
        # those writes, not after the in-loop align has papered over it.
        _validate_layer_slot_signature_consistency(
            layer_cache_keys=self.layer_cache_keys,
            layer_states=self.layer_states,
            step_epoch=step_authority.epoch,
            stage="decode",
        )
        # P12 lever-2: per-step active_slots hoist target. Lazy: filled by the
        # first layer that takes the launch-view refresh leg (post-align, so
        # the value equals what every layer would compute for itself).
        _p12l2_active_slots = None
        # [T2-HOST-DIET 2026-07-10] 步级不变量包 hoist 目标。Lazy:由第一个
        # cache-miss 层(post-align)构建,commit 步 36 层共享一份;非 commit
        # 步该分支不触发=零成本。论证同 P12 active_slots(跨层 slot 一致性
        # 已在循环前验证)。
        _lsc_step_invariants = None
        for state in layer_states_for_refresh:
            if (
                state.last_active_request_ids == _sa_req_ids
                and state.slot_signature64 == global_slot_signature_step
                and state.slot_epoch != step_authority.epoch
            ):
                state.slot_epoch = int(step_authority.epoch)
                state._align_epoch_seen = int(step_authority.epoch)
            if (
                state.last_active_request_ids != _sa_req_ids
                or state.slot_epoch != step_authority.epoch
                or state.slot_signature64 != global_slot_signature_step
            ):
                state.align_slots(
                    _sa_req_ids,
                    epoch=step_authority.epoch,
                    slot_by_request=global_slot_map_step,
                )
                state.last_active_request_ids = _sa_req_ids
            layer_effective_refresh_by_row = layer_effective_refresh_by_row_step
            # Inline _make_step_cache_key from pre-computed row semantic prefix.
            # force_dense=False, force_compact_off=False are constants in this loop.
            cache_key = _ck_prefix_active + (state.compact_meta_epoch, False, False)
            if state.step_cache_key != cache_key:
                if _lsc_step_invariants is None:
                    _lsc_step_invariants = build_step_cache_invariants(
                        state=state,
                        step_meta=step_meta,
                        step_authority=step_authority,
                        step_bound_meta=self.step_bound_meta,
                        device=state.device,
                        force_dense=False,
                        force_compact_off=False,
                        layer_effective_refresh_by_row=layer_effective_refresh_by_row,
                    )
                _build_layer_step_cache(
                    state=state,
                    step_meta=step_meta,
                    step_authority=step_authority,
                    block_table=getattr(self, "_worker_block_table", None),
                    device=state.device,
                    force_dense=False,
                    force_compact_off=False,
                    skip_meta_pack=True,
                    layer_effective_refresh_by_row=layer_effective_refresh_by_row,
                    # M5 Part B2 (2026-04-24): pass step_bound_meta so the
                    # bail-out / compact_cache_valid gate can read
                    # plan.valid instead of per-layer step_cache_*_gpu.
                    step_bound_meta=self.step_bound_meta,
                    # P12 lever-2 (2026-06-12): cache_key was just compared
                    # against state.step_cache_key to take this leg; thread it
                    # down so build_layer_step_cache_impl skips the duplicate
                    # _make_step_cache_key 9-tuple rebuild.
                    precomputed_cache_key=cache_key,
                    # [T2-HOST-DIET 2026-07-10] commit 步 36 层共享步级不变量包。
                    step_invariants=_lsc_step_invariants,
                )
            elif (
                _skip_page_sparse_refresh
                and not bool(getattr(state, "step_cache_has_page_sparse", False))
                and getattr(state, "step_cache_selected_static_pages_i32", None) is None
                and getattr(state, "step_cache_selected_static_seqused_k_by_head_i32", None) is None
                and getattr(state, "step_cache_page_table_i32", None) is None
                and getattr(state, "step_cache_selected_seqused_k_by_head_i32", None) is None
            ):
                state.step_cache_epoch = int(step_authority.epoch)
                state.step_cache_meta_packed = False
            else:
                # P12 lever-2 (2026-06-12): compute active_slots ONCE per step
                # from the first post-align layer taking this leg (slot maps
                # are cross-layer identical here: mutual consistency is
                # validated before the loop and the align block is
                # deterministic from shared step inputs). req_ids source
                # intentionally stays step_meta.req_ids -- the per-layer
                # code's exact source (review MINOR: not step_authority's).
                if _p12l2_active_slots is None:
                    _p12l2_active_slots = tuple(
                        int(state.request_id_to_slot.get(_p12l2_rid, -1))
                        for _p12l2_rid in step_meta.req_ids[:_batch_size_i]
                    )
                refresh_selected_launch_view_for_current_step(
                    state=state,
                    step_meta=step_meta,
                    step_authority=step_authority,
                    block_table=getattr(self, "_worker_block_table", None),
                    device=state.device,
                    layer_effective_refresh_by_row=layer_effective_refresh_by_row,
                    # M5 Part B2 (2026-04-24): thread plan access through so
                    # the inner build_layer_step_cache_impl bail-out can hit.
                    step_bound_meta=self.step_bound_meta,
                    # P12 lever-2: hoisted per-step values; the callee falls
                    # back to its legacy per-layer computation when None.
                    precomputed_active_slots=_p12l2_active_slots,
                    precomputed_cache_key=cache_key,
                )
            if (
                int(getattr(state, "step_cache_plan_version", -1)) != decode_plan_version
                and bool(getattr(state, "step_cache_cached_launch_ready", False))
            ):
                publish_selected_scope_launch_ready_if_needed(
                    state=state,
                    step_authority=step_authority,
                )
            state.step_cache_plan_version = decode_plan_version

        # P12 lever-2 (2026-06-12): _validate_layer_slot_signature_consistency
        # moved BEFORE the layer_states_for_refresh loop (see above) so slot
        # drift aborts before the hoisted active_slots is persisted into
        # per-layer static-carrier fields. Call/raise semantics unchanged.
        if _validate_layer_slot_map_enabled():
            req_ids = step_authority.req_ids[: step_authority.batch_size]
            if req_ids and self.layer_cache_keys:
                first_key = self.layer_cache_keys[0]
                first_state = self.layer_states.get(first_key)
                if first_state is not None:
                    base_slots = [
                        int(first_state.request_id_to_slot.get(rid, -1))
                        for rid in req_ids
                    ]
                    for layer_idx, layer_key in enumerate(self.layer_cache_keys[1:], start=1):
                        layer_state = self.layer_states.get(layer_key)
                        if layer_state is None:
                            continue
                        for row, rid in enumerate(req_ids):
                            layer_slot = int(layer_state.request_id_to_slot.get(rid, -1))
                            if layer_slot != base_slots[row]:
                                raise RuntimeError(
                                    "layer slot map drift before decode meta pack; "
                                    f"epoch={step_authority.epoch} row={row} req={rid} "
                                    f"base_layer=0 base_slot={base_slots[row]} "
                                    f"layer_index={layer_idx} layer_key={layer_key} "
                                    f"layer_slot={layer_slot}"
                                )
    _mark_mb_phase("layer_state_refresh")

    with _mb_record_function("sfi::mb.req_meta_prealloc_cross_layer"):
        xlayer_detail_enabled = bool(mb_profile_enabled or metadata_timing_enabled)
        xlayer_detail_start_ns = (
            time.perf_counter_ns() if xlayer_detail_enabled else 0
        )
        xlayer_detail_phase_ns = xlayer_detail_start_ns
        xlayer_detail_us: dict[str, float] = {}

        def _mark_xlayer_detail(name: str) -> None:
            nonlocal xlayer_detail_phase_ns
            if not xlayer_detail_enabled:
                return
            now_ns = time.perf_counter_ns()
            xlayer_detail_us[name] = float(now_ns - xlayer_detail_phase_ns) / 1000.0
            xlayer_detail_phase_ns = now_ns

        # ========= 跨层 req_meta 预分配与打包门控 =========
        num_layers = len(self.layer_cache_keys)
        max_batch_size = step_authority.max_batch_size
        batch_size = step_authority.batch_size
        first_state = next(iter(self.layer_states.values()))
        device = first_state.device

        decode_reuse_arena_key = None
        worker_block_table = getattr(self, "_worker_block_table", None)
        if (
            bool(getattr(self, "_sparse_attention_in_cudagraph", False))
            and bool(getattr(self.config, "compact_page_residency_enabled", False))
            and isinstance(worker_block_table, torch.Tensor)
            and worker_block_table.ndim >= 2
        ):
            decode_reuse_arena_key = (
                int(batch_size),
                int(num_kv_heads_i),
                int(block_size_i),
                int(worker_block_table.shape[1]),
                _device_identity(device),
            )
        _mark_xlayer_detail("arena_key")
        # [转正清理 2026-07-11] VLLM_SPARSE_REFRESH_REUSE_DECODE_DATA 残旋钮
        # 下线(默认 OFF 从未转正,无旋钮纪律):logf-only reuse 臂整删,判据
        # 恒为 cache_key+spec_key 双同的 verbatim 形态。
        reuse_decode_data = (
            self.step_decode_data is not None
            and self.step_decode_data.cache_key == step_decode_cache_key
            and self._step_decode_spec_key == spec_key
        )
        reuse_decision = None
        full_decode_reuse_hit = bool(reuse_decode_data)
        _mark_xlayer_detail("reuse_decision")
        step_dispatch_plan_reused = (
            getattr(getattr(self, "step_dispatch_plan", None), "cache_key", None)
            == step_decode_cache_key
        )
        ordered_layer_data_reused = getattr(self, "ordered_layer_data_list", None) is not None
        ordered_step_decode_data_reused = (
            getattr(self, "ordered_step_decode_data", None) is not None
        )
        def _buffer_ready(
            buf: Optional[torch.Tensor],
            shape: Tuple[int, ...],
            dtype: torch.dtype,
        ) -> bool:
            if buf is None:
                return False
            if buf.device != device or buf.dtype != dtype:
                return False
            if len(buf.shape) < len(shape):
                return False
            for dim_idx, dim in enumerate(shape):
                if buf.shape[dim_idx] < dim:
                    return False
            return True

        buffers_ready = (
            _buffer_ready(self.step_decode_is_compact_all, (num_layers, max_batch_size), torch.int32)
            and _buffer_ready(self.step_decode_compact_kv_len_all, (num_layers, max_batch_size), torch.int32)
            and _buffer_ready(self.step_decode_compact_offset_all, (num_layers, max_batch_size), torch.int64)
            and _buffer_ready(self.step_decode_req_meta_i32_all, (num_layers, max_batch_size, 7), torch.int32)
            and _buffer_ready(self.step_decode_req_meta_i64_all, (num_layers, max_batch_size, 4), torch.int64)
        )
        _mark_xlayer_detail("buffer_ready_check")
        row_mode_by_row: Tuple[int, ...] = step_authority.row_mode_by_row
        if len(row_mode_by_row) < batch_size:
            raise RuntimeError(
                "decode metadata missing step_authority.row_mode_by_row; "
                f"batch_size={batch_size} row_mode_len={len(row_mode_by_row)}"
            )
        _xlayer_plan = (
            self.step_bound_meta.compact_recent_launch_plan
            if self.step_bound_meta is not None
            else None
        )
        _plan_slot_signature = tuple(
            int(v)
            for v in tuple(getattr(_xlayer_plan, "slot_signature", tuple()))[:batch_size]
        )
        if len(_plan_slot_signature) < batch_size:
            _plan_slot_signature = tuple(
                int(v) for v in step_authority.slot_by_row[:batch_size]
            )
        _compact_layout_generation = int(
            getattr(_xlayer_plan, "compact_meta_epoch", -1)
        )
        compact_layout_signature = (
            _device_identity(device),
            int(num_layers),
            int(max_batch_size),
            int(batch_size),
            tuple(self.layer_cache_keys),
            tuple(int(v) for v in row_mode_by_row[:batch_size]),
            _plan_slot_signature,
            False,
            _compact_layout_generation,
        )
        if _xlayer_plan is not None and bool(getattr(_xlayer_plan, "valid", False)):
            _plan_compact_tokens_cpu = tuple(
                int(v)
                for v in tuple(
                    getattr(_xlayer_plan, "compact_valid_tokens_cpu", tuple())
                )[:batch_size]
            )
            _plan_compact_offsets_cpu = tuple(
                int(v)
                for v in tuple(
                    getattr(_xlayer_plan, "compact_offset_tokens_cpu", tuple())
                )[:batch_size]
            )
            if (
                len(_plan_compact_tokens_cpu) >= batch_size
                and len(_plan_compact_offsets_cpu) >= batch_size
            ):
                compact_layout_signature = (
                    _device_identity(device),
                    int(num_layers),
                    int(max_batch_size),
                    int(batch_size),
                    tuple(self.layer_cache_keys),
                    tuple(int(v) for v in row_mode_by_row[:batch_size]),
                    _plan_slot_signature,
                    True,
                    _plan_compact_tokens_cpu,
                    _plan_compact_offsets_cpu,
                    _compact_layout_generation,
                )
        _mark_xlayer_detail("signature")
        # 静态 compact 布局只在内容签名变更/缓冲不足时填充。
        need_fill_compact_layout = (
            not buffers_ready
            or compact_layout_signature is None
            or getattr(self, "_step_decode_compact_layout_signature", None)
            != compact_layout_signature
        )
        # 动态 req_meta 打包门控与静态 compact 填充分离：
        # - decode 路径（存在 decode row）每个 step 都必须 pack；
        # - pure-prefill 路径不消费 decode dynamic req_meta。若存在 prefill
        #   capture，prefill_global_meta 单独打包；若没有 capture，零初始化
        #   buffer 只作为 StepBoundMeta 占位，避免 chunked prefill 多次支付
        #   decode-pack 启动税。
        has_decode_row = bool(
            getattr(step_authority, "has_decode_row", step_authority.is_decode_only)
        )
        decode_runtime_mode = _decode_runtime_mode_for_controller(self)
        decode_runtime_reason = str(getattr(self, "_decode_runtime_reason", ""))
        rrp_replay_state_ready = bool(
            getattr(self, "_resolved_row_ptr_metadata_ready", False)
        )
        runtime_classification_current = _decode_runtime_classification_is_current(
            self,
            step_authority,
        )
        q_layout_changed = decode_runtime_reason == "q_layout_changed"
        logf_generation_changed = decode_runtime_reason == "logf_generation_changed"
        need_pack_dynamic_req_meta = _should_pack_dynamic_req_meta_for_step(
            has_decode_row=has_decode_row,
            mode=decode_runtime_mode,
            rrp_replay_state_ready=rrp_replay_state_ready,
            q_layout_changed=q_layout_changed,
            logf_generation_changed=logf_generation_changed,
            req_meta_buffers_ready=bool(
                buffers_ready and not need_fill_compact_layout
            ),
            previous_pack_ready=int(
                getattr(self, "_decode_dynamic_pack_epoch", -1)
            )
            >= 0,
            runtime_classification_current=runtime_classification_current,
            has_logf_rows=bool(decode_logf_attn_rows),
            has_refresh_rows=bool(has_refresh_rows),
            refresh_layer_group_active=bool(has_refresh_rows and layer_group_enabled),
        )
        _mark_xlayer_detail("pack_gate")
        active_layer_count = 0
        if need_fill_compact_layout:
            def _ensure_buffer(
                name: str,
                shape: Tuple[int, ...],
                dtype: torch.dtype,
            ) -> torch.Tensor:
                nonlocal decode_runtime_per_step_allocation_count
                buf = getattr(self, name, None)
                if (
                    buf is None
                    or buf.device != device
                    or buf.dtype != dtype
                    or buf.ndim != len(shape)
                    or any(buf.shape[i] < shape[i] for i in range(len(shape)))
                ):
                    buf = torch.zeros(shape, device=device, dtype=dtype)
                    decode_runtime_per_step_allocation_count += 1
                    setattr(self, name, buf)
                return buf

            is_compact_all = _ensure_buffer(
                "step_decode_is_compact_all",
                (num_layers, max_batch_size),
                torch.int32,
            )
            compact_kv_len_all = _ensure_buffer(
                "step_decode_compact_kv_len_all",
                (num_layers, max_batch_size),
                torch.int32,
            )
            compact_offset_all = _ensure_buffer(
                "step_decode_compact_offset_all",
                (num_layers, max_batch_size),
                torch.int64,
            )
            req_meta_i32_all = _ensure_buffer(
                "step_decode_req_meta_i32_all",
                (num_layers, max_batch_size, 7),
                torch.int32,
            )
            req_meta_i64_all = _ensure_buffer(
                "step_decode_req_meta_i64_all",
                (num_layers, max_batch_size, 4),
                torch.int64,
            )
            _mark_xlayer_detail("ensure_buffers")

            # M5 Part B2 (2026-04-24): cross-layer broadcast now consumes the
            # step-level CompactRecentLaunchPlan. All layers share the same
            # compact descriptors (spec §4.3 canonical layer 0 decision), so
            # we write the plan's per-batch_size slice into every layer row
            # of the triplet of cross-layer buffers. Previously this loop
            # sourced each layer's own step_cache_compact_*_gpu tensors,
            # which Part C deletes.
            active_layer_indices = tuple(
                layer_index
                for layer_index, layer_key in enumerate(self.layer_cache_keys)
                if self.layer_states.get(layer_key) is not None
            )
            plan_valid = bool(
                _xlayer_plan is not None and bool(getattr(_xlayer_plan, "valid", False))
            )
            active_layer_count = len(active_layer_indices)
            _mark_xlayer_detail("active_layers")
            if plan_valid:
                if (
                    self._decode_row_is_compact_i32 is None
                    or self._decode_row_is_compact_i32.device != device
                    or self._decode_row_is_compact_i32.numel() < max_batch_size
                ):
                    self._decode_row_is_compact_i32 = torch.zeros((max_batch_size,), device=device, dtype=torch.int32)
                    decode_runtime_per_step_allocation_count += 1
                _mark_xlayer_detail("row_compact_buffer")
                # 从 row_mode_by_row 就地派生 compact 标记（_ROW_MODE_COMPACT == 1）。
                # 这里在首个 compact row 进入 graph bridge 时处于热路径，必须用
                # pinned staging，否则小 tensor H2D 也可能把 first emit 卡成同步尾巴。
                # [ARM-WAR-R1-PINNED-INDEPENDENT 2026-07-12] E4 根修:staging 由
                # 常驻单例(_decode_compact_staging_cpu,已退休)改为每次 fill 独立
                # fresh pinned 分配——单例的「本步 host 复写 ←→ 前一整建步同缓冲
                # H2D 未决」前向 WAR 窗物理消灭(CachingHostAllocator 事件护栏,
                # 机理同 compact_recent_launch_plan_builder 同名标记)。need_fill
                # 仅整建步真,分配频率=整建步频,尺寸桶稳态命中 µs 级。
                try:
                    _compact_staging = torch.empty(
                        (max_batch_size,),
                        device="cpu",
                        dtype=torch.int32,
                        pin_memory=True,
                    )
                except RuntimeError:
                    _compact_staging = torch.empty((max_batch_size,), device="cpu", dtype=torch.int32)
                # [ALLOC-COUNT-TRUTH][合入注记 07-12] fresh pinned 每 fill +1(整建步频,原单例仅首分配计数)。
                decode_runtime_per_step_allocation_count += 1
                # [U14-ZERO-ALLOC-2-9 2026-07-12] 逐行 torch 标量 setitem
                # (×bs,~4.75µs/次=38µs/步 @bs8) → list-comp + numpy 视图整段
                # 向量化赋值(~sub-µs)。_compact_staging 是 fresh pinned int32
                # 连续 1D 缓冲,.numpy() 共享存储;下方 copy_ 读同一内存=逐位不变。
                _rm_compact = int(_ROW_MODE_COMPACT)
                _compact_staging.numpy()[:batch_size] = [
                    1 if row_mode_by_row[_ci] == _rm_compact else 0
                    for _ci in range(batch_size)
                ]
                self._decode_row_is_compact_i32[:batch_size].copy_(
                    _compact_staging[:batch_size],
                    non_blocking=True,
                )
                row_is_compact_i32 = self._decode_row_is_compact_i32[:batch_size]
                _mark_xlayer_detail("row_compact_h2d")
                _plan_kv_slice = _xlayer_plan.compact_valid_tokens_i32[:batch_size]
                _plan_offset_slice = _xlayer_plan.compact_offset_tokens_i64[:batch_size]
                if len(active_layer_indices) == num_layers:
                    layer_slice = slice(0, num_layers)
                    row_view_i32 = row_is_compact_i32.reshape(1, batch_size)
                    kv_view_i32 = _plan_kv_slice.reshape(1, batch_size)
                    offset_view_i64 = _plan_offset_slice.reshape(1, batch_size)
                    is_compact_all[layer_slice, :batch_size].copy_(
                        row_view_i32.expand(num_layers, batch_size),
                        non_blocking=True,
                    )
                    compact_kv_len_all[layer_slice, :batch_size].copy_(
                        kv_view_i32.expand(num_layers, batch_size),
                        non_blocking=True,
                    )
                    compact_offset_all[layer_slice, :batch_size].copy_(
                        offset_view_i64.expand(num_layers, batch_size),
                        non_blocking=True,
                    )
                else:
                    for layer_index in active_layer_indices:
                        is_compact_all[layer_index, :batch_size].copy_(
                            row_is_compact_i32,
                            non_blocking=True,
                        )
                        compact_kv_len_all[layer_index, :batch_size].copy_(
                            _plan_kv_slice,
                            non_blocking=True,
                        )
                        compact_offset_all[layer_index, :batch_size].copy_(
                            _plan_offset_slice,
                            non_blocking=True,
                        )
            else:
                if len(active_layer_indices) == num_layers:
                    layer_slice = slice(0, num_layers)
                    is_compact_all[layer_slice, :batch_size].zero_()
                    compact_kv_len_all[layer_slice, :batch_size].zero_()
                    compact_offset_all[layer_slice, :batch_size].zero_()
                else:
                    for layer_index in active_layer_indices:
                        is_compact_all[layer_index, :batch_size].zero_()
                        compact_kv_len_all[layer_index, :batch_size].zero_()
                        compact_offset_all[layer_index, :batch_size].zero_()
            self._step_decode_compact_layout_signature = compact_layout_signature
            _mark_xlayer_detail("layout_fill")
        else:
            _mark_xlayer_detail("layout_skip")
        if metadata_timing_enabled:
            metadata_timing_xlayer_phase_us = dict(xlayer_detail_us)
        if mb_profile_enabled:
            _append_mb_profile(
                {
                    "event": "metadata_builder_cross_layer_detail",
                    "epoch": int(getattr(step_authority, "epoch", -1)),
                    "batch_size": int(batch_size),
                    "num_layers": int(num_layers),
                    "buffers_ready": bool(buffers_ready),
                    "need_fill_compact_layout": bool(need_fill_compact_layout),
                    "need_pack_dynamic_req_meta": bool(need_pack_dynamic_req_meta),
                    "has_decode_row": bool(has_decode_row),
                    "decode_only": bool(
                        getattr(step_authority, "is_decode_only", False)
                    ),
                    "plan_valid": bool(
                        _xlayer_plan is not None
                        and bool(getattr(_xlayer_plan, "valid", False))
                    ),
                    "active_layer_count": int(active_layer_count),
                    "phase_us": dict(xlayer_detail_us),
                    "total_us": (
                        float(time.perf_counter_ns() - xlayer_detail_start_ns)
                        / 1000.0
                    ),
                }
            )
        _mark_mb_phase("cross_layer_static_layout")

        if step_meta.seqused_k_gpu is None:
            self.step_decode_cache_key = None
            self.step_decode_plan_version = -1
            self.step_dispatch_plan = None
            if _mb_commit_prof is not None:
                _mb_commit_prof.disable()
            return

        # decode：把 meta64[2] out_ptr 融合进跨层 pack（一次 kernel），避免 dispatcher 每层 patch。
        # 仅在本 step 存在 decode log_f rows 时启用；否则传入 dummy 映射以避免每步分配。
        decode_capture_row_buf0_i32: torch.Tensor
        decode_capture_row_buf1_i32: torch.Tensor
        decode_scores_base_ptr_buf0 = 0
        decode_scores_base_ptr_buf1 = 0
        decode_scores_stride_chunk_bytes_buf0 = 0
        decode_scores_stride_chunk_bytes_buf1 = 0
        decode_scores_stride_slot_bytes_buf0 = 0
        decode_scores_stride_slot_bytes_buf1 = 0

        # dummy capture_row_by_batch_row：长度 batch_size，内容全 0（mask=0 时不会被使用）
        if (
            self._decode_dummy_capture_row_by_batch_row_i32 is None
            or self._decode_dummy_capture_row_device != device
            or self._decode_dummy_capture_row_cap < max_batch_size
        ):
            cap = _align_up_int(max_batch_size, 256)
            self._decode_dummy_capture_row_by_batch_row_i32 = torch.zeros((cap,), device=device, dtype=torch.int32)
            decode_runtime_per_step_allocation_count += 1
            self._decode_dummy_capture_row_device = device
            self._decode_dummy_capture_row_cap = cap
        decode_capture_row_buf0_i32 = self._decode_dummy_capture_row_by_batch_row_i32[:batch_size]
        decode_capture_row_buf1_i32 = self._decode_dummy_capture_row_by_batch_row_i32[:batch_size]

        need_decode_out_ptr = (
            bool(decode_logf_attn_rows)
            and int(decode_logf_stride_head) > 0
            and (step_ctx is not None)
        )
        if need_decode_out_ptr:
            # per-layer buf_id/slot_in_chunk 映射（只依赖 num_layers；可跨 step 复用）
            if (
                self._decode_buf_id_by_layer_i32 is None
                or self._decode_slot_in_chunk_by_layer_i32 is None
                or self._decode_buf_id_by_layer_i32.device != device
                or self._decode_slot_in_chunk_by_layer_i32.device != device
                or self._decode_layer_map_num_layers != num_layers
            ):
                # [S1-KC-PIN-STAGING 2026-07-12] 原 copy_(as_tensor(CPU list))
                # =pageable H2D memcpy_and_sync(等当前流排空;冷路径,S1 探针
                # 各 2 次/跑)。两映射均为 arange 的纯整式,直接 device 上生成
                # =拷贝整体消灭(下方 _logf_pair arange 先例同型);值与原 CPU
                # list comprehension 逐位相同(非负整数 floor_divide/mod)。
                _layer_idx_i32 = torch.arange(
                    num_layers, device=device, dtype=torch.int32
                )
                self._decode_buf_id_by_layer_i32 = (
                    (_layer_idx_i32 // _CAPTURE_CHUNK) % _CAPTURE_IN_FLIGHT
                ).contiguous()
                self._decode_slot_in_chunk_by_layer_i32 = (
                    _layer_idx_i32 % _CAPTURE_CHUNK
                ).contiguous()
                # [ALLOC-COUNT-TRUTH][合入注记 07-12] K-c arange 路径分配账:
                # 1 arange 临时+2 contiguous 持久(原口径 4=2 empty+2 staging)。
                decode_runtime_per_step_allocation_count += 3
                self._decode_layer_map_num_layers = num_layers

            # layer_logf_enable：layer-group gating 下只有 active 组的层允许 log_f override。
            # 每次 refresh_event 活跃组可能切换，因此每步重算。
            if has_refresh_rows and layer_group_enabled:
                active_group = (
                    layer_group_active
                    if layer_group_epoch == step_authority.epoch
                    else (layer_group_event_idx & 1)
                )
                # [OUT-PTR-H2D-DIET 2026-07-09] 原 torch.as_tensor(list)→copy_
                # =每触发步一次 pageable 同步 H2D(等 GPU 队列排空,z_rrp 同族,
                # decode_out_ptr_prep 2-8ms 族根源之一)。enable 向量只有两种
                # 内容(active_group 奇偶),预建 GPU 常驻 even/odd 对,触发步按
                # parity 切换引用=零 H2D 零同步。消费点(pack_req_meta_decode_
                # fast_layers)每步重读本属性且为 eager 发射不烘地址,交替引用
                # 安全;numel 精确等于 num_layers(模型常量),消费合同不变。
                _logf_pair = getattr(self, "_decode_layer_logf_enable_pair", None)
                if (
                    _logf_pair is None
                    or _logf_pair[0].device != device
                    or _logf_pair[0].numel() != num_layers
                ):
                    _li = torch.arange(num_layers, device=device, dtype=torch.int32)
                    _even = ((_li & 1) == 0).to(torch.int32).contiguous()
                    _logf_pair = (_even, (1 - _even).contiguous())
                    # [ALLOC-COUNT-TRUTH] arange 构造(后续 ==/.to/.contiguous 为算子临时)。
                    decode_runtime_per_step_allocation_count += 1
                    self._decode_layer_logf_enable_pair = _logf_pair
                self._decode_layer_logf_enable_i32 = _logf_pair[int(active_group) & 1]
            else:
                # 无 refresh 或无 layer-group gating：所有层统一启用（全 1）。
                # pack kernel 对 log_f_mask==0 的行直接跳过，故全 1 无副作用。
                self._decode_layer_logf_enable_i32 = None

            # refresh capture slot_list 单真源：来自 StepAuthority，禁止在 metadata_builder 侧二次推导。
            slot_list = step_authority.refresh_capture_slot_list
            if not isinstance(slot_list, tuple):
                raise RuntimeError(
                    "decode out_ptr requires tuple refresh_capture_slot_list from step_authority"
                )
            if any(int(slot) < 0 for slot in slot_list):
                raise RuntimeError(
                    "decode out_ptr requires non-negative refresh_capture_slot_list"
                )
            if not slot_list:
                raise RuntimeError(
                    "decode out_ptr requires non-empty refresh_capture_slot_list; "
                    f"epoch={int(step_authority.epoch)} batch={int(batch_size)} "
                    f"logf_rows={int(len(decode_logf_attn_rows))}"
                )

        if need_decode_out_ptr:
            # 关键：maybe_build_step_decode_data_from_metadata 在 layer 前置阶段运行，
            # 此时 slot_batch_rows 可能仍停留在上一 step（多 request / 早结束 / 调度重排会触发）。
            # 在构建 capture layout 前先更新 slot->row 映射，避免 row 越界导致硬崩溃。
            try:
                slot_row_map: Dict[int, int] = {}
                for row, rid in enumerate(step_authority.req_ids[:batch_size]):
                    slot = int(first_state.request_id_to_slot.get(rid, -1))
                    if slot >= 0:
                        slot_row_map[slot] = int(row)
                first_state.update_slot_rows(slot_row_map)
            except Exception as exc:
                raise RuntimeError(
                    "decode out_ptr slot->row refresh failed; refuse silent fallback"
                ) from exc

            # [CU-SEQLENS-DEAD-PARAM-RETIRE 2026-07-09] 原地死段下线:此处曾
            # 每触发步做 numpy cumsum+pageable 同步 H2D 生成 cu_seqlens_q_gpu,
            # 唯一去处是 layout impl 的死形参(全程不读)——纯为喂死参付
            # 2-8ms 族同步代价(z_rrp 同族)。参数链已连根拔除。

            # buf0：global_layer_index=0 必定映射到 buf0（chunk_id=0）
            layout0 = self._get_step_capture_layout(
                phase="refresh",
                state=first_state,
                step_context=step_ctx,  # type: ignore[arg-type]
                global_layer_index=0,
                slot_list=slot_list,
                seqused_k=step_meta.seqused_k_gpu,
                num_heads=int(first_state.num_heads),
                device=device,
                chunk_query_lengths=None,
                skip_live_lengths=True,
            )
            if layout0 is None or layout0.capture_row_by_batch_row_i32 is None:
                raise RuntimeError("decode out_ptr prebuild failed for buf0 layout")
            decode_capture_row_buf0_i32 = layout0.capture_row_by_batch_row_i32[:batch_size]
            scores0 = layout0.capture_scores
            decode_scores_base_ptr_buf0 = int(scores0.data_ptr())
            decode_scores_stride_chunk_bytes_buf0 = int(scores0.stride(0) * scores0.element_size())
            decode_scores_stride_slot_bytes_buf0 = int(scores0.stride(1) * scores0.element_size())

            # buf1：若存在第 2 个 chunk（layer_index>=CAPTURE_CHUNK），则必须预建，否则 buf1 层会拿到 out_ptr=0。
            if num_layers > int(_CAPTURE_CHUNK):
                layout1 = self._get_step_capture_layout(
                    phase="refresh",
                    state=first_state,
                    step_context=step_ctx,  # type: ignore[arg-type]
                    global_layer_index=int(_CAPTURE_CHUNK),
                    slot_list=slot_list,
                    seqused_k=step_meta.seqused_k_gpu,
                    num_heads=int(first_state.num_heads),
                    device=device,
                    chunk_query_lengths=None,
                    skip_live_lengths=True,
                )
                if layout1 is None or layout1.capture_row_by_batch_row_i32 is None:
                    raise RuntimeError("decode out_ptr prebuild failed for buf1 layout")
                decode_capture_row_buf1_i32 = layout1.capture_row_by_batch_row_i32[:batch_size]
                scores1 = layout1.capture_scores
                decode_scores_base_ptr_buf1 = int(scores1.data_ptr())
                decode_scores_stride_chunk_bytes_buf1 = int(scores1.stride(0) * scores1.element_size())
                decode_scores_stride_slot_bytes_buf1 = int(scores1.stride(1) * scores1.element_size())
        _mark_mb_phase("decode_out_ptr_prep")

        if need_pack_dynamic_req_meta:
            decode_runtime_req_meta_pack_call_count += 1
            _effective_recent_cap = int(step_authority.recent_cap)
            pack_req_meta_decode_fast_layers(
                seqused_k=step_meta.seqused_k_gpu,
                is_compact_i32=self.step_decode_is_compact_all,
                compact_kv_len_i32=self.step_decode_compact_kv_len_all,
                compact_offset_tokens_i64=self.step_decode_compact_offset_all,
                log_f_mask_i32=(self._decode_log_f_mask_i32[:batch_size] if self._decode_log_f_mask_i32 is not None else None),
                log_f_q_lens_i32=(self._decode_q_lens_i32[:batch_size] if self._decode_q_lens_i32 is not None else None),
                log_f_stride_head=int(decode_logf_stride_head),
                capture_row_by_batch_row_buf0_i32=decode_capture_row_buf0_i32,
                capture_row_by_batch_row_buf1_i32=decode_capture_row_buf1_i32,
                scores_base_ptr_buf0=decode_scores_base_ptr_buf0,
                scores_base_ptr_buf1=decode_scores_base_ptr_buf1,
                scores_stride_chunk_bytes_buf0=decode_scores_stride_chunk_bytes_buf0,
                scores_stride_chunk_bytes_buf1=decode_scores_stride_chunk_bytes_buf1,
                scores_stride_slot_bytes_buf0=decode_scores_stride_slot_bytes_buf0,
                scores_stride_slot_bytes_buf1=decode_scores_stride_slot_bytes_buf1,
                buf_id_by_layer_i32=self._decode_buf_id_by_layer_i32,
                slot_in_chunk_by_layer_i32=self._decode_slot_in_chunk_by_layer_i32,
                layer_logf_enable_i32=self._decode_layer_logf_enable_i32,
                req_meta_i32=self.step_decode_req_meta_i32_all,
                req_meta_i64=self.step_decode_req_meta_i64_all,
                block_size=block_size_i,
                recent_cap=_effective_recent_cap,
                sink_tokens=step_authority.sink_tokens,
                num_layers=num_layers,
                num_seqs=batch_size,
            )
            # 最小 fail-fast 标记：记录本 step 已完成动态 req_meta 打包。
            self._decode_dynamic_pack_epoch = int(step_authority.epoch)
        else:
            if has_decode_row:
                self._decode_dynamic_pack_skipped_same_page_count = (
                    int(
                        getattr(
                            self,
                            "_decode_dynamic_pack_skipped_same_page_count",
                            0,
                        )
                    )
                    + 1
                )
            else:
                self._decode_dynamic_pack_epoch = -1
        _mark_mb_phase("pack_req_meta_decode_fast_layers")

    with _mb_record_function("sfi::mb.misc"):
        # 标记 per-layer meta 已就绪（来自全局一次性打包）
        # M5 Part B2 (2026-04-24): the per-layer readiness marker is gated
        # on the step-level CompactRecentLaunchPlan existence + validity,
        # not per-layer step_cache_compact_*_gpu which Part C removes. Plan
        # validity implies the cross-layer req_meta pack above already ran.
        _marker_plan = (
            self.step_bound_meta.compact_recent_launch_plan
            if self.step_bound_meta is not None
            else None
        )
        if (
            _marker_plan is not None
            and bool(getattr(_marker_plan, "valid", False))
            and not bool(skip_layer_state_refresh)
            and not bool(pure_prefill_step)
        ):
            decode_runtime_layer_states_traversal_count += int(len(self.layer_states))
            for state in self.layer_states.values():
                state.step_cache_meta_packed = True

        if step_meta.seqused_k_gpu is None:
            self.step_decode_cache_key = None
            self.step_decode_plan_version = -1
            self.step_dispatch_plan = None
            if _mb_commit_prof is not None:
                _mb_commit_prof.disable()
            return

        # 构建 StepDecodeData（list 索引）
        if not reuse_decode_data:
            self._build_step_decode_data(
                step_meta=step_meta,
                batch_size=step_authority.batch_size,
                block_size=block_size_i,
                num_kv_heads=num_kv_heads_i,
                head_dim=head_dim_i,
                kv_cache_dtype=kv_dtype_value,
                cache_key=step_decode_cache_key,
                decode_plan_version=int(getattr(step_authority, "decode_plan_version", -1)),
            )
            self._step_decode_spec_key = spec_key
        elif self.step_decode_data is not None:
            self.step_decode_data.cache_key = step_decode_cache_key
            if step_meta.seqused_k_gpu is not None:
                self.step_decode_data.seqused_k = step_meta.seqused_k_gpu
            self.step_decode_data.epoch = step_authority.epoch
            self.step_decode_data.decode_plan_version = int(
                getattr(step_authority, "decode_plan_version", -1)
            )
        # NOTE: compact readiness is now checked per-layer in
        # _build_step_dispatch_plan (CPU-only, no GPU sync).

        if (
            self.step_dispatch_plan is None
            or self.step_dispatch_plan.cache_key != step_decode_cache_key
        ):
            self._build_step_dispatch_plan(step_meta, cache_key=step_decode_cache_key)
        else:
            self.step_dispatch_plan.epoch = step_authority.epoch
            self.step_dispatch_plan.batch_size = step_authority.batch_size
            self.step_dispatch_plan.decode_plan_version = int(
                getattr(step_authority, "decode_plan_version", -1)
            )
            if (
                self.step_dispatch_plan.step_decode_data is not None
                and self.step_dispatch_plan.layer_data_list_ordered is not None
            ):
                apply_reuse_ordered_plan_state(
                    self,
                    epoch=step_authority.epoch,
                    batch_size=step_authority.batch_size,
                )
        _mark_mb_phase("dispatch_plan_build")
        # [T2-FORENSIC] 窗口1关闭(rrp_bind 同步吸收段不采样)。
        if _mb_commit_prof is not None:
            _mb_commit_prof.disable()

        replay_bind_device = (
            step_meta.seqused_k_gpu.device
            if isinstance(step_meta.seqused_k_gpu, torch.Tensor)
            else next(iter(self.layer_states.values())).device
        )
        # [LIFECYCLE-OFF-ONLY 2026-07-10] native lifecycle 整臂下线(用户拍板:
        # 唯速度,保 OFF 删 ON);此处原 ON 专属 visible-state 逐步维护块已删。
        _maybe_bind_resolved_row_ptr_replay_metadata(
            self,
            attn_metadata=attn_metadata,
            batch_size=int(step_authority.batch_size),
            block_size=block_size_i,
            num_kv_heads=num_kv_heads_i,
            device=torch.device(replay_bind_device),
        )
        _mark_mb_phase("resolved_row_ptr_bind")

        # [T2-FORENSIC] 窗口2:build_step_bound_meta_final。
        if _mb_commit_prof is not None:
            _mb_commit_prof.enable()
        build_step_bound_meta_from_metadata_impl(
            self,
            attn_metadata=attn_metadata,
            kv_cache_spec=kv_cache_spec,
        )
        _mark_mb_phase("build_step_bound_meta_final")
        if _mb_commit_prof is not None:
            _mb_commit_prof.disable()
            _mb_commit_cprofile_dump()
        if mb_profile_enabled:
            _decode_runtime_state_profile = getattr(
                self, "_decode_runtime_state", None
            )
            _decode_runtime_mode_value = (
                decode_runtime_mode.value
                if isinstance(decode_runtime_mode, DecodeRuntimeMode)
                else str(decode_runtime_mode)
            )
            _same_page_step_count = (
                1
                if (
                    decode_runtime_mode is DecodeRuntimeMode.STEADY_DELTA
                    and bool(runtime_classification_current)
                )
                else 0
            )
            _page_boundary_step_count = (
                1
                if (
                    decode_runtime_mode is DecodeRuntimeMode.PAGE_BOUNDARY_DELTA
                    and bool(runtime_classification_current)
                )
                else 0
            )
            _refresh_commit_step_count = (
                1
                if (
                    decode_runtime_mode is DecodeRuntimeMode.REFRESH_COMMIT
                    and bool(runtime_classification_current)
                )
                else 0
            )
            _rrp_visible_debug_fields: dict[str, object] = {}
            _replay_arena_debug = getattr(self, "_resolved_row_ptr_replay_arena", None)
            _launch_plan_debug = getattr(
                getattr(self, "step_bound_meta", None),
                "compact_recent_launch_plan",
                None,
            )
            if _launch_plan_debug is None:
                _launch_plan_debug = getattr(
                    previous_step_bound_meta,
                    "compact_recent_launch_plan",
                    None,
                )
            _delta_debug = getattr(self, "_decode_runtime_delta", None)
            _row_effective_debug = getattr(
                _delta_debug,
                "row_effective_k_by_row",
                getattr(step_authority, "context_kv_len_by_row", tuple()),
            )
            _device_debug = torch.device(replay_bind_device)
            if isinstance(_replay_arena_debug, ResolvedRowPtrArena):
                _rrp_visible_debug_fields = _rrp_visible_source_debug_fields(
                    controller=self,
                    replay_arena=_replay_arena_debug,
                    launch_plan=_launch_plan_debug,
                    row_effective_k_by_row=tuple(_row_effective_debug),
                    batch_size=int(getattr(step_authority, "batch_size", 0)),
                    device=_device_debug,
                    sparse_dynamic_state=None,
                )
            _publish_rrp_visible_source_debug_attrs(attn_metadata, _rrp_visible_debug_fields)
            _append_mb_profile(
                {
                    "event": "metadata_builder_step",
                    "epoch": int(getattr(step_authority, "epoch", -1)),
                    "batch_size": int(getattr(step_authority, "batch_size", -1)),
                    "num_layers": int(len(getattr(self, "layer_cache_keys", ()))),
                    "decode_only": bool(
                        getattr(step_authority, "is_decode_only", False)
                    ),
                    "has_decode_row": bool(
                        getattr(
                            step_authority,
                            "has_decode_row",
                            getattr(step_authority, "is_decode_only", False),
                        )
                    ),
                    "reuse_decode_data": bool(reuse_decode_data),
                    "full_decode_reuse_hit": bool(full_decode_reuse_hit),
                    "decode_reuse_miss_reason": str(
                        getattr(reuse_decision, "miss_reason", "")
                    ),
                    "decode_reuse_miss_reasons": list(
                        getattr(reuse_decision, "miss_reasons", ())
                    ),
                    "step_dispatch_plan_reused": bool(step_dispatch_plan_reused),
                    "ordered_layer_data_reused": bool(ordered_layer_data_reused),
                    "ordered_step_decode_data_reused": bool(ordered_step_decode_data_reused),
                    "buffers_ready": bool(buffers_ready),
                    "need_fill_compact_layout": bool(need_fill_compact_layout),
                    "need_pack_dynamic_req_meta": bool(need_pack_dynamic_req_meta),
                    "skip_layer_state_refresh": bool(skip_layer_state_refresh),
                    "need_decode_out_ptr": bool(need_decode_out_ptr),
                    "decode_logf_attn_rows": int(len(decode_logf_attn_rows)),
                    "compact_rows": int(
                        sum(
                            1
                            for value in step_authority.row_mode_by_row[:batch_size]
                            if int(value) == int(_ROW_MODE_COMPACT)
                        )
                    ),
                    "decode_runtime_mode": _decode_runtime_mode_value,
                    "decode_runtime_reason": str(decode_runtime_reason),
                    "ultra_first_miss_reason": str(
                        getattr(self, "_forensic_ultra_first_miss_reason", "")
                    ),
                    "ultra_first_rrp_miss_reason": str(
                        getattr(self, "_forensic_ultra_first_rrp_miss_reason", "")
                    ),
                    "runtime_classification_current": bool(
                        runtime_classification_current
                    ),
                    "same_page_step_count": int(_same_page_step_count),
                    "page_boundary_step_count": int(_page_boundary_step_count),
                    "refresh_commit_step_count": int(_refresh_commit_step_count),
                    "old_cache_probe_count": int(decode_runtime_old_cache_probe_count),
                    "adapter_invocation_count": int(
                        decode_runtime_adapter_invocation_count
                    ),
                    "steady_delta_d2h_sync_count": int(
                        decode_runtime_steady_delta_d2h_sync_count
                    ),
                    "implicit_sync_count": int(decode_runtime_implicit_sync_count),
                    "per_step_allocation_count": int(
                        decode_runtime_per_step_allocation_count
                    ),
                    "carrier_update_kernel_count": int(
                        decode_runtime_carrier_update_kernel_count
                    ),
                    "build_compact_recent_launch_plan_call_count": int(
                        decode_runtime_launch_builder_call_count
                    ),
                    "bind_resolved_row_ptr_replay_metadata_call_count": int(
                        getattr(
                            self,
                            "_decode_runtime_rrp_metadata_bind_call_count",
                            0,
                        )
                    ),
                    "pack_req_meta_decode_fast_layers_call_count": int(
                        decode_runtime_req_meta_pack_call_count
                    ),
                    "layer_states_traversal_count": int(
                        decode_runtime_layer_states_traversal_count
                    ),
                    "decode_runtime_update_step_us": float(
                        sum(float(v) for v in mb_phase_us.values())
                    ),
                    "phase_us": dict(mb_phase_us),
                    **_rrp_visible_debug_fields,
                    "total_us": float(time.perf_counter_ns() - mb_total_start_ns)
                    / 1000.0,
                }
            )
        _emit_metadata_timing(
            "complete",
            extra={
                "reuse_decode_data": bool(reuse_decode_data),
                "full_decode_reuse_hit": bool(full_decode_reuse_hit),
                "step_dispatch_plan_reused": bool(step_dispatch_plan_reused),
                "ordered_layer_data_reused": bool(ordered_layer_data_reused),
                "ordered_step_decode_data_reused": bool(ordered_step_decode_data_reused),
                "buffers_ready": bool(buffers_ready),
                "need_fill_compact_layout": bool(need_fill_compact_layout),
                "need_pack_dynamic_req_meta": bool(need_pack_dynamic_req_meta),
                "skip_layer_state_refresh": bool(skip_layer_state_refresh),
                "need_decode_out_ptr": bool(need_decode_out_ptr),
                "decode_logf_attn_rows": int(len(decode_logf_attn_rows)),
                "compact_rows": int(
                    sum(
                        1
                        for value in step_authority.row_mode_by_row[:batch_size]
                        if int(value) == int(_ROW_MODE_COMPACT)
                    )
                ),
                "decode_runtime_mode": (
                    decode_runtime_mode.value
                    if isinstance(decode_runtime_mode, DecodeRuntimeMode)
                    else str(decode_runtime_mode)
                ),
                "decode_runtime_reason": str(decode_runtime_reason),
                "runtime_classification_current": bool(runtime_classification_current),
                "steady_fast_path_miss_reason": str(
                    getattr(self, "_decode_runtime_steady_fast_path_miss_reason", "")
                ),
                "speculative_delta_miss_reason": str(
                    getattr(self, "_decode_runtime_speculative_delta_miss_reason", "")
                ),
                "rrp_step_state_miss_reason": str(
                    getattr(self, "_decode_runtime_rrp_step_state_miss_reason", "")
                ),
                "old_cache_probe_count": int(decode_runtime_old_cache_probe_count),
                "adapter_invocation_count": int(decode_runtime_adapter_invocation_count),
                "steady_delta_d2h_sync_count": int(
                    decode_runtime_steady_delta_d2h_sync_count
                ),
                "implicit_sync_count": int(decode_runtime_implicit_sync_count),
                "per_step_allocation_count": int(decode_runtime_per_step_allocation_count),
                "carrier_update_kernel_count": int(
                    decode_runtime_carrier_update_kernel_count
                ),
                "build_compact_recent_launch_plan_call_count": int(
                    decode_runtime_launch_builder_call_count
                ),
                "bind_resolved_row_ptr_replay_metadata_call_count": int(
                    getattr(
                        self,
                        "_decode_runtime_rrp_metadata_bind_call_count",
                        0,
                    )
                ),
                "pack_req_meta_decode_fast_layers_call_count": int(
                    decode_runtime_req_meta_pack_call_count
                ),
                "layer_states_traversal_count": int(
                    decode_runtime_layer_states_traversal_count
                ),
                # fa4_writer_graph_engagement_export: #9 captured-writer-graph engagement
                # counters (cheap getattr ints; controller == self via
                # RefreshRebuildMixin). replay>>capture == high K-bucket hit
                # rate == net host-dispatch win; capture~replay == thrash.
                "writer_graph_replay_count": int(
                    getattr(
                        self, "_deadline_async_producer_graph_replay_count", 0
                    )
                ),
                "writer_graph_capture_count": int(
                    getattr(
                        self, "_deadline_async_producer_graph_capture_count", 0
                    )
                ),
                "writer_graph_recapture_count": int(
                    getattr(self, "_writer_graph_recapture_count", 0)
                ),
                "writer_graph_bypass": bool(
                    (getattr(self, "_writer_graph_state", None) or {}).get(
                        "bypass"
                    )
                    if isinstance(
                        getattr(self, "_writer_graph_state", None), dict
                    )
                    else False
                ),
            },
        )
    # M5 Part B2 (2026-04-24): CompactRecentLaunchPlan producer was hoisted
    # earlier in this function (search for `sfi::mb.launch_plan_build`), so
    # the cross-layer broadcast at `sfi::mb.req_meta_prealloc_cross_layer`
    # can read plan.* directly. The previous call site here (after
    # build_step_bound_meta_from_metadata_impl) has been removed to avoid
    # double-building.


def build_step_bound_meta_from_metadata_impl(
    self,
    *,
    attn_metadata: object,
    kv_cache_spec: Optional[object],
) -> None:
    """单写者入口：在 metadata builder 阶段统一绑定 StepBoundMeta。"""
    del attn_metadata, kv_cache_spec
    step_authority = self.step_authority
    step_ctx = self.step_context
    existing_launch_plan = getattr(
        getattr(self, "step_bound_meta", None),
        "compact_recent_launch_plan",
        None,
    )
    # [T2-HOST-DIET 2026-07-10] VLLM_FIX_CARRY_DESC_BUILD 死实验枝已删
    # (默认恒 OFF,判据基线全部未开;每步 5 getattr + __import__ + env 读纯税)。
    self.step_bound_meta = None
    if step_authority is None or step_ctx is None:
        self._compact_recent_launch_template = None
        return
    if step_ctx.epoch != step_authority.epoch:
        raise RuntimeError(
            "bound-meta build requires step_context epoch match; "
            f"ctx_epoch={step_ctx.epoch} step_epoch={step_authority.epoch}"
        )
    step_handle_id = int(step_authority.step_handle_id)
    step_handle_generation = int(step_authority.step_handle_generation)
    if step_handle_id <= 0 or step_handle_generation <= 0:
        raise RuntimeError(
            "bound-meta build missing step handle identity; "
            f"step_handle_id={step_handle_id} step_handle_generation={step_handle_generation}"
        )
    step_decode_data = self.step_decode_data
    step_decode_cache_key = self.step_decode_cache_key
    if step_decode_data is None or step_decode_cache_key is None:
        raise RuntimeError(
            "bound-meta build requires step_decode_data cache; "
            f"step_epoch={step_authority.epoch}"
        )
    if step_decode_data.cache_key != step_decode_cache_key:
        raise RuntimeError(
            "bound-meta build cache_key mismatch: "
            f"decode_data={step_decode_data.cache_key!r} current={step_decode_cache_key!r}"
        )
    batch_size = int(step_authority.batch_size)
    q_start_loc = tuple(int(v) for v in step_authority.q_start_loc[: batch_size + 1])
    if len(q_start_loc) < batch_size + 1:
        raise RuntimeError(
            "bound-meta build q_start_loc coverage mismatch; "
            f"q_start={len(q_start_loc)} batch={batch_size}"
        )
    if (
        len(step_authority.q_lens_by_row) < batch_size
        or len(step_authority.context_kv_len_by_row) < batch_size
        or len(step_authority.logits_last_n_by_row) < batch_size
        or len(step_authority.logits_capacity_by_row) < batch_size
        or len(step_authority.logf_mask_by_row) < batch_size
    ):
        raise RuntimeError(
            "bound-meta build step_authority row coverage mismatch; "
            f"q_lens={len(step_authority.q_lens_by_row)} "
            f"context={len(step_authority.context_kv_len_by_row)} "
            f"last_n={len(step_authority.logits_last_n_by_row)} "
            f"cap={len(step_authority.logits_capacity_by_row)} "
            f"mask={len(step_authority.logf_mask_by_row)} "
            f"batch={batch_size}"
        )
    # [T2-HOST-DIET 2026-07-10] 逐行 tuple(int) 规范化一次;签名/StepBoundMeta/
    # canonical/logf-prefix 全复用,消除同一 8 元组的三重重物化(值等价:
    # 底层已是 int 序列,int() 规范化不改比较语义,且与 steady 路径
    # _refresh_step_bound_meta_for_steady_delta 的规范化口径一致)。
    q_lens_by_row = tuple(
        int(v) for v in step_authority.q_lens_by_row[:batch_size]
    )
    context_kv_len_by_row = tuple(
        int(v) for v in step_authority.context_kv_len_by_row[:batch_size]
    )
    logits_last_n_by_row = tuple(
        int(v) for v in step_authority.logits_last_n_by_row[:batch_size]
    )
    logits_capacity_by_row = tuple(
        int(v) for v in step_authority.logits_capacity_by_row[:batch_size]
    )
    logf_mask_by_row = tuple(
        int(v) for v in step_authority.logf_mask_by_row[:batch_size]
    )
    row_mode_by_row = tuple(
        int(v) for v in step_authority.row_mode_by_row[:batch_size]
    )
    logf_attn_rows = tuple(
        int(row) for row in step_authority.logf_attn_rows if 0 <= row < batch_size
    )
    prefill_rows = tuple(
        int(row) for row in step_authority.prefill_rows if 0 <= row < batch_size
    )
    bound_meta_signature = (
        int(step_authority.epoch),
        int(step_handle_id),
        int(step_handle_generation),
        step_decode_cache_key,
        logits_last_n_by_row,
        logits_capacity_by_row,
        logf_mask_by_row,
        q_start_loc,
        prefill_rows,
    )

    req_meta_i32_all = None
    req_meta_i64_all = None
    if (
        (not bool(step_authority.has_decode_row))
        and self.prefill_global_meta_epoch == step_authority.epoch
        and int(getattr(self, "prefill_global_meta_handle_id", -1))
        == int(step_authority.step_handle_id)
        and int(getattr(self, "prefill_global_meta_handle_generation", -1))
        == int(step_authority.step_handle_generation)
        and self.step_prefill_req_meta_i32_all is not None
        and self.step_prefill_req_meta_i64_all is not None
    ):
        req_meta_i32_all = self.step_prefill_req_meta_i32_all
        req_meta_i64_all = self.step_prefill_req_meta_i64_all
    else:
        req_meta_i32_all = self.step_decode_req_meta_i32_all
        req_meta_i64_all = self.step_decode_req_meta_i64_all
    if req_meta_i32_all is None or req_meta_i64_all is None:
        raise RuntimeError(
            "bound-meta build requires packed req_meta buffers"
        )
    bound_meta_signature = tuple(bound_meta_signature) + (
        row_mode_by_row,
        logf_mask_by_row,
    )
    # [FINAL-LAYER-BOUND-LAZY 2026-07-10] layer_bound 全树唯一读者是
    # _validate_bound_meta_compact_contract(门控验证器,默认关);默认档
    # 36 层 BoundLayerMeta 构造(含 72 次 view getter + 72 次 req_meta 切片)
    # 是纯税,只在验证门开时构造。req_meta 存在性 raise 留在门外,保住
    # "pack 相位已跑"的活体合同。
    validate_bound_meta_contract = _validate_bound_meta_contract_enabled()
    layer_bound_tuple: Tuple[Optional[BoundLayerMeta], ...] = tuple()
    if validate_bound_meta_contract:
        layer_bound_cache_key = (
            id(step_decode_data),
            step_decode_data.cache_key,
            id(req_meta_i32_all),
            id(req_meta_i64_all),
            int(batch_size),
            bool(step_authority.hint_all_compact),
            bool(step_authority.hint_has_log_f),
            bool(step_authority.hint_log_f_eq1),
            bool(step_authority.hint_log_f_gt1),
        )
        cached_layer_bound = getattr(self, "_step_bound_layer_bound_cache", None)
        if (
            isinstance(cached_layer_bound, tuple)
            and len(cached_layer_bound) == 2
            and cached_layer_bound[0] == layer_bound_cache_key
            and isinstance(cached_layer_bound[1], tuple)
        ):
            layer_bound_tuple = cached_layer_bound[1]
        else:
            layer_count = len(step_decode_data.layer_data)
            layer_bound_list: List[Optional[BoundLayerMeta]] = [None] * layer_count
            # [T2-HOST-DIET 2026-07-10] 36 层循环的步级不变量提环外
            # (int()/bool()/shape 逐层重转 ×36 → 一次)。
            _lb_block_size = int(step_decode_data.block_size)
            _lb_num_kv_heads = int(step_decode_data.num_kv_heads)
            _lb_head_dim = int(step_decode_data.head_dim)
            _lb_i32_layers = int(req_meta_i32_all.shape[0])
            _lb_i64_layers = int(req_meta_i64_all.shape[0])
            _hint_all_compact = bool(step_authority.hint_all_compact)
            _hint_has_log_f = bool(step_authority.hint_has_log_f)
            _hint_log_f_eq1 = bool(step_authority.hint_log_f_eq1)
            _hint_log_f_gt1 = bool(step_authority.hint_log_f_gt1)
            for layer_index, layer_data in enumerate(step_decode_data.layer_data):
                if layer_data is None:
                    continue
                if layer_index >= _lb_i32_layers or layer_index >= _lb_i64_layers:
                    raise RuntimeError(
                        "bound-meta layer index out of req_meta range: "
                        f"layer_index={layer_index} "
                        f"i32_layers={_lb_i32_layers} "
                        f"i64_layers={_lb_i64_layers}"
                    )
                k_compact = layer_data.k_compact
                v_compact = layer_data.v_compact
                token_positions = layer_data.token_positions
                compact_kv_len_max = int(layer_data.compact_kv_len_max)
                all_compact = bool(layer_data.has_compact)
                state_ref = layer_data.state_ref
                has_compact_layout = False
                if state_ref is not None:
                    has_compact_layout = bool(
                        getattr(state_ref, "step_cache_has_compact", False)
                        and int(getattr(state_ref, "compact_stride_blocks", 0)) > 0
                    )
                    if has_compact_layout:
                        k_compact, v_compact = state_ref.get_compact_kv_views(
                            block_size=_lb_block_size,
                            num_kv_heads=_lb_num_kv_heads,
                            head_dim=_lb_head_dim,
                        )
                        token_positions = state_ref.get_compact_token_positions_view(
                            block_size=_lb_block_size,
                            num_kv_heads=_lb_num_kv_heads,
                            device=state_ref.device,
                            rows=batch_size,
                        )
                    all_compact = bool(getattr(state_ref, "step_cache_all_compact", all_compact))
                    if has_compact_layout:
                        persist_cap = max(0, compact_kv_len_max)
                        compact_kv_len_max = persist_cap
                    else:
                        compact_kv_len_max = 0
                layer_bound_list[layer_index] = BoundLayerMeta(
                    req_meta_i32=req_meta_i32_all[layer_index],
                    req_meta_i64=req_meta_i64_all[layer_index],
                    k_compact=k_compact,
                    v_compact=v_compact,
                    token_positions=token_positions,
                    compact_kv_len_max=compact_kv_len_max,
                    all_compact=bool(_hint_all_compact and all_compact),
                    hint_has_log_f=_hint_has_log_f,
                    hint_log_f_eq1=_hint_log_f_eq1,
                    hint_log_f_gt1=_hint_log_f_gt1,
                )
            layer_bound_tuple = tuple(layer_bound_list)
            self._step_bound_layer_bound_cache = (
                layer_bound_cache_key,
                layer_bound_tuple,
            )
    step_bound_meta = StepBoundMeta(
        step_handle_id=step_handle_id,
        step_handle_generation=step_handle_generation,
        epoch=step_authority.epoch,
        batch_size=batch_size,
        q_start_loc=q_start_loc,
        q_lens_by_row=q_lens_by_row,
        context_kv_len_by_row=context_kv_len_by_row,
        logits_last_n_by_row=logits_last_n_by_row,
        logits_capacity_by_row=logits_capacity_by_row,
        logf_mask_by_row=logf_mask_by_row,
        logf_attn_rows=logf_attn_rows,
        prefill_rows=prefill_rows,
        recent_cap=int(step_authority.recent_cap),
        sink_tokens=int(step_authority.sink_tokens),
        canonical_real_kv_len_cpu=context_kv_len_by_row,
        req_set_hash=int(step_authority.req_set_hash),
        row_phase_hash=int(step_authority.row_phase_hash),
        plan_signature=tuple(step_authority.plan_signature),
        bound_meta_signature=bound_meta_signature,
        layer_bound=layer_bound_tuple,
        compact_recent_launch_plan=existing_launch_plan,
    )
    if validate_bound_meta_contract:
        step_envelope = step_ctx.step_envelope_v2
        if step_envelope is None:
            raise RuntimeError("bound-meta compact contract requires step_envelope_v2")
        if step_envelope.epoch != step_authority.epoch:
            raise RuntimeError(
                "bound-meta compact contract epoch mismatch: "
                f"env_epoch={step_envelope.epoch} step_epoch={step_authority.epoch}"
            )
        _validate_bound_meta_compact_contract(
            step_bound_meta=step_bound_meta,
            step_row_mode_by_row=step_envelope.row_mode_by_row,
        )
    self.step_bound_meta = step_bound_meta
    final_launch_plan = step_bound_meta.compact_recent_launch_plan
    descriptor_cpu_i32 = getattr(
        self, "_compact_recent_launch_plan_descriptor_cpu_i32", None
    )
    descriptor_gpu_i32 = getattr(
        self, "_compact_recent_launch_plan_descriptor_i32", None
    )
    if (
        final_launch_plan is not None
        and bool(getattr(final_launch_plan, "valid", False))
        and isinstance(descriptor_cpu_i32, torch.Tensor)
        and isinstance(descriptor_gpu_i32, torch.Tensor)
    ):
        launch_template = getattr(self, "_compact_recent_launch_template", None)
        if (
            not isinstance(launch_template, LaunchTemplate)
            or launch_template.plan is not final_launch_plan
            or launch_template.descriptor_cpu_i32 is not descriptor_cpu_i32
            or launch_template.descriptor_gpu_i32 is not descriptor_gpu_i32
        ):
            launch_template = compile_launch_template(
                final_launch_plan,
                descriptor_cpu_i32=descriptor_cpu_i32,
                descriptor_gpu_i32=descriptor_gpu_i32,
            )
        self._compact_recent_launch_template = launch_template
    else:
        self._compact_recent_launch_template = None
    staged_logf_signature = getattr(self, "_decode_logf_stage_signature", None)
    expected_logf_stage_prefix: Tuple[object, ...] = (
        int(step_authority.step_identity_token),
        int(batch_size),
        int(step_authority.max_batch_size),
        logits_capacity_by_row,
        q_lens_by_row,
        logf_mask_by_row,
        logf_attn_rows,
        int(step_authority.logf_stride_head),
        max(logits_capacity_by_row, default=0),
    )
    if (
        int(getattr(self, "_decode_logf_stage_token", -1))
        == int(step_authority.step_identity_token)
        and isinstance(staged_logf_signature, tuple)
        and tuple(staged_logf_signature[: len(expected_logf_stage_prefix)])
        == expected_logf_stage_prefix
    ):
        self._decode_logf_stage_bound_signature = tuple(
            step_bound_meta.bound_meta_signature
        )
    else:
        self._decode_logf_stage_bound_signature = None

def maybe_build_step_prefill_global_meta_from_metadata_impl(
    self,
    *,
    attn_metadata: object,
    kv_cache_spec: Optional[object],
) -> None:
    """在 Triton metadata builder 阶段构建 prefill 跨层 req_meta（一次 pack）。

    目标：
    - 把 prefill 的 req_meta 填充从 per-layer Python 循环搬到一次 Triton pack；
    - 同时在 pack 内写入 out_ptr(meta64[2]) 与 denom_ptr(meta64[3])（last_n>1），
      避免 dispatcher 侧 torch.tensor(list) 的 ptr patch；
    - 保守启用：仅在"纯 prefill（无 decode row）"的 step 才构建全局 meta，mix chunk 直接禁用。
    """
    def _invalidate_prefill_global_meta() -> None:
        self.prefill_global_meta_epoch = -1
        self.prefill_global_meta_handle_id = -1
        self.prefill_global_meta_handle_generation = -1

    detail_profile_enabled = bool(
        _mb_profile_log_path()
        or (os.environ.get("VLLM_SPARSE_FA3_ROUTE_TRACE_LOG", "") if _DYNAMIC_ENV else _FA3_ROUTE_TRACE_LOG_CACHED)
    )
    detail_total_start_ns = time.perf_counter_ns() if detail_profile_enabled else 0
    detail_phase_start_ns = detail_total_start_ns
    detail_phase_us: dict[str, float] = {}

    def _mark_detail_phase(name: str) -> None:
        nonlocal detail_phase_start_ns
        if not detail_profile_enabled:
            return
        now_ns = time.perf_counter_ns()
        detail_phase_us[name] = float(now_ns - detail_phase_start_ns) / 1000.0
        detail_phase_start_ns = now_ns

    def _append_detail_event(status: str, **extra: object) -> None:
        if not detail_profile_enabled:
            return
        payload: dict[str, object] = {
            "event": "prefill_global_meta_build_detail",
            "status": str(status),
            "epoch": int(getattr(step_authority, "epoch", -1))
            if step_authority is not None
            else -1,
            "batch_size": int(getattr(step_authority, "batch_size", -1))
            if step_authority is not None
            else -1,
            "phase_us": dict(detail_phase_us),
            "total_us": float(time.perf_counter_ns() - detail_total_start_ns)
            / 1000.0,
        }
        payload.update(extra)
        if hasattr(self, "get_prefill_capture_arena_metrics"):
            payload.update(self.get_prefill_capture_arena_metrics())
        _append_mb_profile(payload)
        if (os.environ.get("VLLM_SPARSE_FA3_ROUTE_TRACE_LOG", "") if _DYNAMIC_ENV else _FA3_ROUTE_TRACE_LOG_CACHED):
            try:
                from patches.fa3_native.install import append_fa3_route_trace

                append_fa3_route_trace(payload)
            except Exception:
                _log.debug("failed to append prefill global meta route trace", exc_info=True)

    step_meta = self.step_meta
    step_authority = self.step_authority
    step_ctx = self.step_context
    current_step_handle_id = int(getattr(step_ctx, "step_handle_id", -1)) if step_ctx is not None else -1
    current_step_handle_generation = (
        int(getattr(step_ctx, "step_handle_generation", -1))
        if step_ctx is not None
        else -1
    )
    if step_authority is None or step_meta is None or step_ctx is None:
        _invalidate_prefill_global_meta()
        _append_detail_event("missing_state")
        return
    if step_authority.is_decode_only:
        _invalidate_prefill_global_meta()
        _append_detail_event("decode_only")
        return
    if step_ctx.epoch != step_authority.epoch:
        _invalidate_prefill_global_meta()
        _append_detail_event("stale_step_context")
        return
    _mark_detail_phase("state_guard")

    # 仅在能确认"纯 prefill（无 decode row）"时启用。
    prompt_lens = step_ctx.prompt_lens
    computed = step_ctx.num_computed_tokens
    batch_size = step_authority.batch_size
    if (
        prompt_lens is None
        or computed is None
        or len(prompt_lens) != batch_size
        or len(computed) != batch_size
    ):
        _invalidate_prefill_global_meta()
        _append_detail_event("prompt_lengths_unavailable")
        return
    for idx in range(batch_size):
        p = int(prompt_lens[idx])
        c = int(computed[idx])
        # prompt_len<=0 的场景无法可靠区分 prefill/decode（保守禁用）
        if p <= 0:
            _invalidate_prefill_global_meta()
            _append_detail_event("nonpositive_prompt_len")
            return
        if c >= p:
            # 至少一条已进入 decode → mix chunk，禁用全局 prefill meta
            _invalidate_prefill_global_meta()
            _append_detail_event("prompt_guard_detected_decode_row")
            return
    _mark_detail_phase("prefill_guard")

    if not self.layer_states or not self.layer_cache_keys:
        _invalidate_prefill_global_meta()
        _append_detail_event("missing_layer_states")
        return
    _mark_detail_phase("layer_state_guard")

    # 依赖 TritonAttentionMetadata：query_start_loc (cu_seqlens_q), max_seq_len, seq_lens
    cu_seqlens_q = getattr(attn_metadata, "query_start_loc", None)
    if not isinstance(cu_seqlens_q, torch.Tensor):
        _invalidate_prefill_global_meta()
        _append_detail_event("missing_query_start_loc")
        return
    if cu_seqlens_q.dtype != torch.int32:
        cu_seqlens_q = cu_seqlens_q.to(dtype=torch.int32)
    if cu_seqlens_q.numel() < (batch_size + 1):
        _invalidate_prefill_global_meta()
        _append_detail_event("query_start_loc_too_short")
        return

    if step_meta.seqused_k_gpu is None:
        seq_lens = getattr(attn_metadata, "seq_lens", None)
        if isinstance(seq_lens, torch.Tensor):
            if seq_lens.dtype != torch.int32:
                seq_lens = seq_lens.to(dtype=torch.int32)
            if seq_lens.numel() >= batch_size:
                step_meta.seqused_k_gpu = seq_lens[:batch_size]
    if step_meta.seqused_k_gpu is None or not isinstance(step_meta.seqused_k_gpu, torch.Tensor):
        _invalidate_prefill_global_meta()
        _append_detail_event("missing_seqused_k")
        return
    seqused_k = step_meta.seqused_k_gpu
    if seqused_k.dtype != torch.int32:
        seqused_k = seqused_k.to(dtype=torch.int32)
    if seqused_k.numel() < batch_size:
        _invalidate_prefill_global_meta()
        _append_detail_event("seqused_k_too_short")
        return
    _mark_detail_phase("metadata_tensors")

    # Pure prefill may span several chunked-prefill steps. Only the tail chunk
    # that actually captures logits needs the prefill-global capture metadata;
    # earlier chunks can use the normal dense/mixed launch without paying this
    # setup tax.
    capture_plan_full_by_req, _finalize_req_ids = self.get_step_prefill_plan_by_req(
        step_context=step_ctx
    )
    capture_plan_active_by_req: Dict[str, int] = {}
    if capture_plan_full_by_req:
        for rid, last_n_raw in capture_plan_full_by_req.items():
            last_n = int(last_n_raw or 0)
            if last_n > 0:
                capture_plan_active_by_req[str(rid)] = int(last_n)
    if step_ctx is not None:
        self._ensure_step_prefill_capture_last_n_by_row(
            step_context=step_ctx,
            capture_plan_active_by_req=(
                capture_plan_active_by_req if capture_plan_active_by_req else None
            ),
        )
    if not capture_plan_active_by_req:
        _invalidate_prefill_global_meta()
        _append_detail_event(
            "no_active_capture_plan",
            capture_plan_full_req_count=int(len(capture_plan_full_by_req)),
            capture_plan_active_req_count=0,
        )
        return
    _mark_detail_phase("capture_plan")

    # This prefill-global path is an optimization only. The authoritative
    # capture layout builder consumes StepBoundMeta logits fields, which are
    # populated later by prepare_step_logits_buffers in the normal per-layer
    # path. If they are not ready yet, fail fast instead of updating GPU
    # slot-row tensors and then inevitably returning layout0_missing.
    bound_meta = self.step_bound_meta
    bound_last_n_by_row = (
        tuple(int(v) for v in getattr(bound_meta, "logits_last_n_by_row", tuple()))
        if bound_meta is not None
        else tuple()
    )
    bound_capacity_by_row = (
        tuple(int(v) for v in getattr(bound_meta, "logits_capacity_by_row", tuple()))
        if bound_meta is not None
        else tuple()
    )
    capture_rows_missing_bound_logits: list[int] = []
    for row, rid in enumerate(step_authority.req_ids[:batch_size]):
        if str(rid) not in capture_plan_active_by_req:
            continue
        if (
            row >= len(bound_last_n_by_row)
            or row >= len(bound_capacity_by_row)
            or int(bound_last_n_by_row[row]) <= 0
            or int(bound_capacity_by_row[row]) <= 0
        ):
            capture_rows_missing_bound_logits.append(int(row))
    if capture_rows_missing_bound_logits:
        _invalidate_prefill_global_meta()
        _append_detail_event(
            "bound_meta_logits_not_ready",
            capture_req_count=int(len(capture_plan_active_by_req)),
            capture_rows=tuple(capture_rows_missing_bound_logits),
            bound_last_n_by_row=bound_last_n_by_row[:batch_size],
            bound_capacity_by_row=bound_capacity_by_row[:batch_size],
        )
        return

    # Align slots for all layers once (dispatcher 将复用 last_active_request_ids 避免重复 align)
    active_req_ids = step_authority.req_ids[:batch_size]
    global_slot_map_step = self.get_step_global_slot_map(active_req_ids)
    for state in self.layer_states.values():
        if state.last_active_request_ids != step_authority.req_ids:
            state.align_slots(
                active_req_ids,
                epoch=step_authority.epoch,
                slot_by_request=global_slot_map_step,
            )
            state.last_active_request_ids = step_authority.req_ids

    _validate_layer_slot_signature_consistency(
        layer_cache_keys=self.layer_cache_keys,
        layer_states=self.layer_states,
        step_epoch=step_authority.epoch,
        stage="prefill",
    )
    _mark_detail_phase("align_slots")

    first_state = next(iter(self.layer_states.values()))
    device = first_state.device
    num_layers = len(self.layer_cache_keys)
    max_batch_size = step_authority.max_batch_size
    if cu_seqlens_q.device != device:
        cu_seqlens_q = cu_seqlens_q.to(device=device)
    if seqused_k.device != device:
        seqused_k = seqused_k.to(device=device)

    # layout 构建仍需要 slot_list（selector 以 slot 为主键）；在纯 prefill 场景下可用 first_state 映射。
    capture_plan_active: Dict[int, int] = {}
    if capture_plan_active_by_req:
        for rid, last_n in capture_plan_active_by_req.items():
            slot = int(first_state.request_id_to_slot.get(rid, -1))
            if slot >= 0:
                capture_plan_active[int(slot)] = int(last_n)

    # 构建 per-row last_n/cap（int32，step-epoch 缓存）
    kv_needed_cpu = 0
    if (
        self._prefill_i32_epoch != step_authority.epoch
        or int(getattr(self, "_prefill_i32_handle_id", -1)) != current_step_handle_id
        or int(getattr(self, "_prefill_i32_handle_generation", -1))
        != current_step_handle_generation
        or self._prefill_last_n_i32 is None
        or self._prefill_cap_i32 is None
    ):
        def _ensure_prefill_cpu_stage(name: str) -> torch.Tensor:
            buf = getattr(self, name, None)
            if (
                buf is None
                or not isinstance(buf, torch.Tensor)
                or buf.device.type != "cpu"
                or buf.dtype != torch.int32
                or buf.numel() < max_batch_size
            ):
                buf = torch.empty((max_batch_size,), device="cpu", dtype=torch.int32, pin_memory=True)
                setattr(self, name, buf)
            return buf

        last_n_cpu_stage = _ensure_prefill_cpu_stage("_prefill_last_n_cpu_i32")[:batch_size]
        cap_cpu_stage = _ensure_prefill_cpu_stage("_prefill_cap_cpu_i32")[:batch_size]
        last_n_cpu_stage.zero_()
        cap_cpu_stage.zero_()
        if capture_plan_active_by_req:
            max_seq_len = int(getattr(attn_metadata, "max_seq_len", 0) or 0)
            max_seq_len = max(max_seq_len, 0)
            # row -> req_id
            for row, rid in enumerate(step_authority.req_ids[:batch_size]):
                ln = int(capture_plan_active_by_req.get(str(rid), 0) or 0)
                if ln <= 0:
                    continue
                last_n_cpu_stage[row] = int(ln)
                seq_len = int(step_authority.context_kv_len_by_row[row]) if row < len(step_authority.context_kv_len_by_row) else 0
                seq_len = max(seq_len, 0)
                cap = seq_len
                if max_seq_len > 0:
                    cap = min(cap, max_seq_len)
                if cap <= 0:
                    cap = 1
                cap_cpu_stage[row] = int(cap)
                kv_needed_cpu = max(int(kv_needed_cpu), int(cap))
        # 预分配/复用缓冲，避免每步 torch.tensor(list) 重新分配
        if (
            self._prefill_last_n_i32 is None
            or self._prefill_last_n_i32.device != device
            or self._prefill_last_n_i32.dtype != torch.int32
            or self._prefill_last_n_i32.numel() < max_batch_size
        ):
            self._prefill_last_n_i32 = torch.empty((max_batch_size,), device=device, dtype=torch.int32)
        if (
            self._prefill_cap_i32 is None
            or self._prefill_cap_i32.device != device
            or self._prefill_cap_i32.dtype != torch.int32
            or self._prefill_cap_i32.numel() < max_batch_size
        ):
            self._prefill_cap_i32 = torch.empty((max_batch_size,), device=device, dtype=torch.int32)
        self._prefill_last_n_i32[:batch_size].copy_(last_n_cpu_stage, non_blocking=True)
        self._prefill_cap_i32[:batch_size].copy_(cap_cpu_stage, non_blocking=True)
        if batch_size < max_batch_size:
            self._prefill_last_n_i32[batch_size:max_batch_size].zero_()
            self._prefill_cap_i32[batch_size:max_batch_size].zero_()
        self._prefill_i32_epoch = step_authority.epoch
        self._prefill_i32_handle_id = current_step_handle_id
        self._prefill_i32_handle_generation = current_step_handle_generation
    else:
        # 复用已有 CPU 结果不可得；仅在有 capture 时用 step_ctx 的 CPU tuple 推导 kv_needed。
        # 注意：这里不允许读取 GPU tensor 的 .item()，避免 DtoH 同步。
        if capture_plan_active_by_req:
            for row, rid in enumerate(step_authority.req_ids[:batch_size]):
                ln = int(capture_plan_active_by_req.get(str(rid), 0) or 0)
                if ln <= 0:
                    continue
                seq_len = int(step_authority.context_kv_len_by_row[row]) if row < len(step_authority.context_kv_len_by_row) else 0
                seq_len = max(seq_len, 0)
                kv_needed_cpu = max(int(kv_needed_cpu), int(seq_len))
    _mark_detail_phase("lastn_cap_stage")

    # log_f stride_head：与 capture layout 的 kv_max 对齐。
    # - 有 capture 时：kv_max>0，pack kernel 用它计算 out_ptr/denom_ptr 的 stride。
    # - 无 capture 时：kv_max=0，pack kernel 会把 log_f ptr/stride 置零（纯 dense prefill 仍可用全局 pack）。
    kv_bucket = int(_CAPTURE_KV_BUCKET_CACHED)
    if kv_bucket <= 0:
        kv_bucket = 256
    kv_bucket = max(256, int(kv_bucket))
    kv_max = _align_up_int(int(kv_needed_cpu), kv_bucket) if int(kv_needed_cpu) > 0 else 0
    self._prefill_log_f_stride_head = int(kv_max)
    self._prefill_log_f_stride_epoch = step_authority.epoch
    self._prefill_log_f_stride_handle_id = int(step_authority.step_handle_id)
    self._prefill_log_f_stride_handle_generation = int(step_authority.step_handle_generation)

    # 预分配跨层 req_meta buffer
    def _ensure_prefill_buf(
        name: str,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        buf = getattr(self, name)
        if (
            buf is None
            or buf.device != device
            or buf.dtype != dtype
            or buf.ndim != len(shape)
            or any(buf.shape[i] < shape[i] for i in range(len(shape)))
        ):
            buf = torch.zeros(shape, device=device, dtype=dtype)
            setattr(self, name, buf)
        return buf

    req_meta_i32_all = _ensure_prefill_buf(
        "step_prefill_req_meta_i32_all",
        (num_layers, max_batch_size, 7),
        torch.int32,
    )
    req_meta_i64_all = _ensure_prefill_buf(
        "step_prefill_req_meta_i64_all",
        (num_layers, max_batch_size, 4),
        torch.int64,
    )
    if hasattr(self, "prefill_capture_meta_arena"):
        req_meta_bytes = int(req_meta_i32_all.numel() * req_meta_i32_all.element_size())
        req_meta_bytes += int(req_meta_i64_all.numel() * req_meta_i64_all.element_size())
        self.prefill_capture_meta_arena.metrics.arena_bucket_bytes = max(
            int(self.prefill_capture_meta_arena.metrics.arena_bucket_bytes),
            int(req_meta_bytes),
        )

    # per-layer buf_id/slot_in_chunk 映射（只依赖 num_layers，可跨 step 复用）
    if (
        self._decode_buf_id_by_layer_i32 is None
        or self._decode_slot_in_chunk_by_layer_i32 is None
        or self._decode_buf_id_by_layer_i32.device != device
        or self._decode_slot_in_chunk_by_layer_i32.device != device
        or self._decode_layer_map_num_layers != num_layers
    ):
        # [S1-KC-PIN-STAGING 2026-07-12] decode 侧同名映射的孪生件(共享同一对
        # 属性);同型同修:device arange 直接生成=拷贝整体消灭,值逐位同原
        # CPU list comprehension。机理注释见 decode 侧(need_decode_out_ptr 块)。
        _layer_idx_i32 = torch.arange(num_layers, device=device, dtype=torch.int32)
        self._decode_buf_id_by_layer_i32 = (
            (_layer_idx_i32 // _CAPTURE_CHUNK) % _CAPTURE_IN_FLIGHT
        ).contiguous()
        self._decode_slot_in_chunk_by_layer_i32 = (
            _layer_idx_i32 % _CAPTURE_CHUNK
        ).contiguous()
        self._decode_layer_map_num_layers = num_layers
    _mark_detail_phase("buffer_prepare")

    # 默认使用 dummy layout（无 capture）以避免额外小张量分配
    capture_row_buf0_i32: torch.Tensor
    capture_row_buf1_i32: torch.Tensor
    scores_base_ptr_buf0 = 0
    scores_base_ptr_buf1 = 0
    scores_stride_chunk_bytes_buf0 = 0
    scores_stride_chunk_bytes_buf1 = 0
    scores_stride_slot_bytes_buf0 = 0
    scores_stride_slot_bytes_buf1 = 0
    denoms_base_ptr_buf0 = 0
    denoms_base_ptr_buf1 = 0
    denoms_stride_chunk_bytes_buf0 = 0
    denoms_stride_chunk_bytes_buf1 = 0
    denoms_stride_slot_bytes_buf0 = 0
    denoms_stride_slot_bytes_buf1 = 0

    if capture_plan_active and kv_max > 0:
        # 为 layout 构建 slot->row 映射（依赖当前 step 的 batch row）
        slot_row_map: Dict[int, int] = {}
        for row, rid in enumerate(step_authority.req_ids[:batch_size]):
            slot = int(first_state.request_id_to_slot.get(rid, -1))
            if slot >= 0:
                slot_row_map[slot] = int(row)
        first_state.update_slot_rows(slot_row_map)
        _mark_detail_phase("capture_slot_row_update")

        prefill_slots = sorted(int(s) for s in capture_plan_active.keys())
        _mark_detail_phase("capture_prefill_slots")
        if not prefill_slots:
            _invalidate_prefill_global_meta()
            _mark_detail_phase("capture_layout")
            _append_detail_event(
                "no_prefill_slots",
                capture_req_count=int(len(capture_plan_active_by_req)),
                capture_slot_count=0,
                kv_max=int(kv_max),
            )
            return
        chunk_query_lengths = (cu_seqlens_q[1 : batch_size + 1] - cu_seqlens_q[:batch_size]).to(dtype=torch.long)
        _mark_detail_phase("capture_chunk_query_lengths")

        layout0 = self._get_step_capture_layout(
            phase="prefill",
            state=first_state,
            step_context=step_ctx,
            global_layer_index=0,
            slot_list=prefill_slots,
            seqused_k=seqused_k[:batch_size],
            num_heads=int(first_state.num_heads),
            device=device,
            chunk_query_lengths=chunk_query_lengths,
            prepared_only=bool(getattr(self, "_prefill_capture_meta_arena_enabled", False)),
        )
        _mark_detail_phase("capture_layout0_call")
        if layout0 is None or layout0.capture_row_by_batch_row_i32 is None:
            _invalidate_prefill_global_meta()
            _mark_detail_phase("capture_layout")
            if hasattr(self, "prefill_capture_meta_arena"):
                self.prefill_capture_meta_arena.metrics.arena_ready_before_tail = False
                self.prefill_capture_meta_arena.metrics.arena_bind_status = "sync_expansion_miss"
            _append_detail_event(
                "layout0_missing",
                capture_req_count=int(len(capture_plan_active_by_req)),
                capture_slot_count=int(len(capture_plan_active)),
                kv_max=int(kv_max),
            )
            return
        capture_row_buf0_i32 = layout0.capture_row_by_batch_row_i32
        scores0 = layout0.capture_scores
        scores_base_ptr_buf0 = int(scores0.data_ptr())
        scores_stride_chunk_bytes_buf0 = int(scores0.stride(0) * scores0.element_size())
        scores_stride_slot_bytes_buf0 = int(scores0.stride(1) * scores0.element_size())
        den0 = layout0.log_f_denoms
        denoms_base_ptr_buf0 = int(den0.data_ptr())
        denoms_stride_chunk_bytes_buf0 = int(den0.stride(0) * den0.element_size())
        denoms_stride_slot_bytes_buf0 = int(den0.stride(1) * den0.element_size())

        if num_layers > int(_CAPTURE_CHUNK):
            layout1 = self._get_step_capture_layout(
                phase="prefill",
                state=first_state,
                step_context=step_ctx,
                global_layer_index=int(_CAPTURE_CHUNK),
                slot_list=prefill_slots,
                seqused_k=seqused_k[:batch_size],
                num_heads=int(first_state.num_heads),
                device=device,
                chunk_query_lengths=chunk_query_lengths,
                prepared_only=bool(getattr(self, "_prefill_capture_meta_arena_enabled", False)),
            )
            _mark_detail_phase("capture_layout1_call")
            if layout1 is None or layout1.capture_row_by_batch_row_i32 is None:
                _invalidate_prefill_global_meta()
                _mark_detail_phase("capture_layout")
                if hasattr(self, "prefill_capture_meta_arena"):
                    self.prefill_capture_meta_arena.metrics.arena_ready_before_tail = False
                    self.prefill_capture_meta_arena.metrics.arena_bind_status = "sync_expansion_miss"
                _append_detail_event(
                    "layout1_missing",
                    capture_req_count=int(len(capture_plan_active_by_req)),
                    capture_slot_count=int(len(capture_plan_active)),
                    kv_max=int(kv_max),
                )
                return
            capture_row_buf1_i32 = layout1.capture_row_by_batch_row_i32
            scores1 = layout1.capture_scores
            scores_base_ptr_buf1 = int(scores1.data_ptr())
            scores_stride_chunk_bytes_buf1 = int(scores1.stride(0) * scores1.element_size())
            scores_stride_slot_bytes_buf1 = int(scores1.stride(1) * scores1.element_size())
            den1 = layout1.log_f_denoms
            denoms_base_ptr_buf1 = int(den1.data_ptr())
            denoms_stride_chunk_bytes_buf1 = int(den1.stride(0) * den1.element_size())
            denoms_stride_slot_bytes_buf1 = int(den1.stride(1) * den1.element_size())
        else:
            capture_row_buf1_i32 = capture_row_buf0_i32
            scores_base_ptr_buf1 = scores_base_ptr_buf0
            scores_stride_chunk_bytes_buf1 = scores_stride_chunk_bytes_buf0
            scores_stride_slot_bytes_buf1 = scores_stride_slot_bytes_buf0
            denoms_base_ptr_buf1 = denoms_base_ptr_buf0
            denoms_stride_chunk_bytes_buf1 = denoms_stride_chunk_bytes_buf0
            denoms_stride_slot_bytes_buf1 = denoms_stride_slot_bytes_buf0
    else:
        # no capture：提供稳定的 dummy mapping，避免 pack 内部创建新 tensor
        if (
            self._decode_dummy_capture_row_by_batch_row_i32 is None
            or self._decode_dummy_capture_row_device != device
            or self._decode_dummy_capture_row_cap < max_batch_size
        ):
            cap = _align_up_int(max_batch_size, 256)
            self._decode_dummy_capture_row_by_batch_row_i32 = torch.zeros((cap,), device=device, dtype=torch.int32)
            self._decode_dummy_capture_row_device = device
            self._decode_dummy_capture_row_cap = cap
        capture_row_buf0_i32 = self._decode_dummy_capture_row_by_batch_row_i32
        capture_row_buf1_i32 = self._decode_dummy_capture_row_by_batch_row_i32
    _mark_detail_phase("capture_layout")

    recent_cap = step_authority.recent_cap
    pack_req_meta_prefill_fast_layers(
        seqused_k=seqused_k,
        cu_seqlens_q=cu_seqlens_q,
        log_f_last_n_i32=self._prefill_last_n_i32[:batch_size],
        log_f_capacity_i32=self._prefill_cap_i32[:batch_size],
        log_f_stride_head=int(kv_max),
        capture_row_by_batch_row_buf0_i32=capture_row_buf0_i32,
        capture_row_by_batch_row_buf1_i32=capture_row_buf1_i32,
        scores_base_ptr_buf0=int(scores_base_ptr_buf0),
        scores_base_ptr_buf1=int(scores_base_ptr_buf1),
        scores_stride_chunk_bytes_buf0=int(scores_stride_chunk_bytes_buf0),
        scores_stride_chunk_bytes_buf1=int(scores_stride_chunk_bytes_buf1),
        scores_stride_slot_bytes_buf0=int(scores_stride_slot_bytes_buf0),
        scores_stride_slot_bytes_buf1=int(scores_stride_slot_bytes_buf1),
        denoms_base_ptr_buf0=int(denoms_base_ptr_buf0),
        denoms_base_ptr_buf1=int(denoms_base_ptr_buf1),
        denoms_stride_chunk_bytes_buf0=int(denoms_stride_chunk_bytes_buf0),
        denoms_stride_chunk_bytes_buf1=int(denoms_stride_chunk_bytes_buf1),
        denoms_stride_slot_bytes_buf0=int(denoms_stride_slot_bytes_buf0),
        denoms_stride_slot_bytes_buf1=int(denoms_stride_slot_bytes_buf1),
        buf_id_by_layer_i32=self._decode_buf_id_by_layer_i32,
        slot_in_chunk_by_layer_i32=self._decode_slot_in_chunk_by_layer_i32,
        req_meta_i32=req_meta_i32_all,
        req_meta_i64=req_meta_i64_all,
        recent_cap=recent_cap,
        sink_tokens=step_authority.sink_tokens,
        num_layers=num_layers,
        num_seqs=batch_size,
    )
    _mark_detail_phase("pack_req_meta_prefill_fast_layers")

    self.prefill_global_meta_epoch = step_authority.epoch
    self.prefill_global_meta_handle_id = current_step_handle_id
    self.prefill_global_meta_handle_generation = current_step_handle_generation
    if hasattr(self, "prefill_capture_meta_arena") and capture_plan_active:
        self.prefill_capture_meta_arena.metrics.arena_ready_before_tail = True
        self.prefill_capture_meta_arena.metrics.arena_bind_status = "prepared_bind"
    if detail_profile_enabled:
        _append_detail_event(
            "built",
            num_layers=int(num_layers),
            max_batch_size=int(max_batch_size),
            capture_req_count=int(len(capture_plan_active_by_req)),
            capture_slot_count=int(len(capture_plan_active)),
            kv_max=int(kv_max),
        )
