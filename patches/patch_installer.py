"""
patches/patch_installer.py — vLLM monkey-patch installation and teardown.

OWNS:
  - Patch state variables (_GLOBAL_CONTROLLER, _PATCH_INSTALLED, etc.)
  - _patch_*(): individual monkey-patch functions for vLLM internals
  - _install_patch(): orchestrates all patches
  - apply_vllm_sparse_patch() / ensure_vllm_sparse_patch_from_env(): public API
  - disable_vllm_sparse_patch(): teardown/restore

DEPENDS_ON:
  - patches.vllm_sparse_patch: VLLMSparseController (lazy import)
  - patches.sparse_types: SparseControllerConfig
  - vllm.*: monkey-patched targets

ENTRY_POINTS:
  - apply_vllm_sparse_patch(config): 应用 sparse patch
  - ensure_vllm_sparse_patch_from_env(): 从环境变量自动应用
  - disable_vllm_sparse_patch(): 禁用并恢复原始函数
"""
from __future__ import annotations

import atexit
import json
import math
import os
import sys
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

import torch

from hybrid_selectors.alpha_fair_selector import AlphaFairSelectorConfig
from patches.sparse_constants import (
    _DYNAMIC_ENV,
    _REFRESH_ENQUEUE_STAGGER_CACHED,
    _REFRESH_REBUILD_MAX_DELAY_STEPS_CACHED,
    _REPLAY_REFRESH_NOOP_FAST_SKIP_CACHED,
    _ROW_MODE_DENSE,
)
from patches.sparse_utils import assert_cleanup_ledgers_drained_for_step_build
from utils.sentence_triggers import RefreshTriggerConfig
from patches.sparse_types import SparseControllerConfig, StepContext, StepTicket, _normalize_prefill_capture_config
from patches.step_authority import StepAuthority
from patches.tp_contract import E_TP_INPUT_CONTRACT, ensure_tp_prompt_lengths, validate_tp_input_contract
from patches.controller_mixins.refresh_rebuild_mixin import (
    _mark_pending_selected_scope_terminal,
    _pending_refresh_rebuild_matches_current_target_scope,
)
from patches.fa3_native.compact_recent_contract import (
    CompactRecentLaunchConfig,
    build_compact_recent_host_plan_i32,
    compute_recent_visible_kv_len_i32,
    validate_compact_recent_support_matrix,
)
from patches.vllm_compat import (
    import_fa_utils_module,
)

try:
    from vllm.logger import init_logger
except Exception:
    init_logger = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from patches.vllm_sparse_patch import VLLMSparseController

if init_logger is not None:
    _log = init_logger(__name__)
else:
    import logging
    _log = logging.getLogger(__name__)



def _metadata_rrp_int_list_attr(metadata: object, attr_name: str) -> list[int]:
    raw = getattr(metadata, attr_name, ())
    if raw is None or isinstance(raw, torch.Tensor):
        return []
    try:
        return [int(v) for v in tuple(raw)]
    except Exception:
        return []


def _metadata_rrp_visible_source_fields(metadata: object) -> dict[str, object]:
    return {
        "rrp_visible_source_kind": str(
            getattr(metadata, "_sparse_rrp_visible_source_kind", "") or ""
        ),
        "rrp_sparse_dynamic_state_covers_rows": bool(
            getattr(metadata, "_sparse_dynamic_state_covers_rows", False)
        ),
        "rrp_sparse_dynamic_state_failure_reason": str(
            getattr(metadata, "_sparse_dynamic_state_failure_reason", "") or ""
        ),
        "rrp_visible_data_ptr": int(
            getattr(metadata, "_sparse_rrp_visible_data_ptr", 0) or 0
        ),
        "rrp_sparse_dynamic_state_data_ptr": int(
            getattr(metadata, "_sparse_dynamic_state_data_ptr", 0) or 0
        ),
        "rrp_visible_shape": _metadata_rrp_int_list_attr(
            metadata, "_sparse_rrp_visible_shape"
        ),
        "rrp_visible_is_arena_seqused": bool(
            getattr(metadata, "_sparse_rrp_visible_is_arena_seqused", False)
        ),
        "rrp_visible_is_arena_batch_seqused": bool(
            getattr(metadata, "_sparse_rrp_visible_is_arena_batch_seqused", False)
        ),
        "rrp_visible_is_launch_effective": bool(
            getattr(metadata, "_sparse_rrp_visible_is_launch_effective", False)
        ),
        "rrp_visible_is_dense_seqused": bool(
            getattr(metadata, "_sparse_rrp_visible_is_dense_seqused", False)
        ),
        "rrp_launch_effective_covers_rows": bool(
            getattr(metadata, "_sparse_rrp_launch_effective_covers_rows", False)
        ),
        "rrp_launch_effective_k_len_cpu": _metadata_rrp_int_list_attr(
            metadata, "_sparse_rrp_launch_effective_k_len_cpu"
        ),
        "rrp_row_effective_k_by_row": _metadata_rrp_int_list_attr(
            metadata, "_sparse_rrp_row_effective_k_by_row"
        ),
    }


def _metadata_items_rrp_visible_source_fields(metadata_items: object) -> dict[str, object]:
    fallback: dict[str, object] | None = None
    for metadata in metadata_items or ():
        fields = _metadata_rrp_visible_source_fields(metadata)
        if fallback is None:
            fallback = fields
        if str(fields.get("rrp_visible_source_kind", "")) or int(
            fields.get("rrp_visible_data_ptr", 0) or 0
        ):
            return fields
    if fallback is not None:
        return fallback
    return _metadata_rrp_visible_source_fields(None)


# -----------------------------------------------------------------------------
# Controller helpers and patched unified_attention
# -----------------------------------------------------------------------------

_ORIGINAL_V1_FLASH_ATTN_VARLEN_FUNC = None
_ORIGINAL_V1_FLASH_ATTN_FORWARD = None
_ORIGINAL_V1_FLASH_ATTN_GET_SCHEDULER_METADATA = None
_ORIGINAL_V1_FLASH_ATTN_GET_FLASH_ATTN_VERSION = None
_ORIGINAL_FA_UTILS_FLASH_ATTN_VARLEN_FUNC = None
_ORIGINAL_FA_UTILS_GET_SCHEDULER_METADATA = None
_ORIGINAL_FA_UTILS_GET_FLASH_ATTN_VERSION = None
_ORIGINAL_V1_FLASH_METADATA_FULL_CUDAGRAPH_SUPPORTED = None
_FLASH_ATTN_FORWARD_PATCHED: bool = False
_V1_FLASH_ATTN_MODULES = None
_GLOBAL_CONTROLLER: Optional[VLLMSparseController] = None
_PATCH_INSTALLED: bool = False
_UBATCH_WRAPPER_PATCHED: bool = False
_ORIGINAL_UBATCH_WRAPPER_CALL = None
_CUDAGRAPH_WRAPPER_PATCHED: bool = False
_ORIGINAL_CUDAGRAPH_WRAPPER_CALL = None
_MODEL_FORWARD_REFRESH_OWNER_PATCHED: bool = False
_ORIGINAL_MODEL_FORWARD_FOR_REFRESH_OWNER = None
_MODEL_FORWARD_REFRESH_READY_ATTR = (
    "_mixed_page_model_forward_refresh_generation_ready"
)
_MODEL_FORWARD_REFRESH_ACTIVE_ATTR = "_mixed_page_model_forward_refresh_owner_active"
_SERIALIZED_CONFIG_ENV = "VLLM_SPARSE_CONTROLLER_JSON"
# [EVT-BISECT 2026-07-07] 4B illegal 归因量具:每步两枚事件(drain 后/replay 后)
# 非阻塞 query 夹逼 sticky error 的毒源段。默认关=零开销;开=每步 2 次
# Event.record+若干 query(µs 级),不 sync 不扰时序。frontier 打印:
# last_ok 与 first_err 之间提交的段=嫌疑区间。
_EVT_BISECT_ENABLED = os.environ.get("VLLM_SPARSE_EVT_BISECT", "0") == "1"
_EVT_BISECT_RING: "deque" = deque()
_EVT_BISECT_LAST_OK: list = ["<none>"]
_EVT_BISECT_STEP: list = [0]


def _evt_bisect_mark(label: str, controller=None) -> None:
    if not _EVT_BISECT_ENABLED:
        return
    if label == "post_drain":
        _EVT_BISECT_STEP[0] += 1
    snap = ""
    if controller is not None:
        try:
            pend = len(getattr(controller, "_pending_refresh_rebuilds", ()) or ())
            grouped = getattr(
                controller, "_pending_refresh_grouped_async_records", None
            )
            grp = len(grouped) if grouped else 0
            snap = f" pend={pend} grp={grp}"
        except Exception:
            snap = " snap_err"
    tagged = f"{label}@{_EVT_BISECT_STEP[0]}{snap}"
    while _EVT_BISECT_RING:
        old_label, old_ev = _EVT_BISECT_RING[0]
        try:
            done = old_ev.query()
        except RuntimeError as exc:
            recent = " | ".join(lbl for lbl, _ in list(_EVT_BISECT_RING)[:6])
            print(
                "SFI_EVT_BISECT frontier: last_ok="
                f"{_EVT_BISECT_LAST_OK[0]} first_err={old_label} "
                f"ring_head=[{recent}] err={exc}",
                flush=True,
            )
            raise
        if not done:
            break
        _EVT_BISECT_LAST_OK[0] = old_label
        _EVT_BISECT_RING.popleft()
    ev = torch.cuda.Event()
    ev.record(torch.cuda.current_stream())
    _EVT_BISECT_RING.append((tagged, ev))
_REPLAY_REFRESH_ENQUEUE_PROFILE_DETAIL_CACHED = (
    os.environ.get("VLLM_SPARSE_REPLAY_REFRESH_ENQUEUE_PROFILE_DETAIL", "0") == "1"
)
_FULL_CUDAGRAPH_REPLAY_REFRESH_BATCHED_FLUSH_CACHED = (
    os.environ.get("VLLM_SPARSE_FULL_CUDAGRAPH_REPLAY_REFRESH_BATCHED_FLUSH", "0")
    == "1"
)
_FULL_CUDAGRAPH_REPLAY_REFRESH_DEFER_TO_DEADLINE_CACHED = (
    os.environ.get(
        "VLLM_SPARSE_FULL_CUDAGRAPH_REPLAY_REFRESH_DEFER_TO_DEADLINE",
        "0",
    )
    == "1"
)

_FULL_CUDAGRAPH_HOOK_PROFILE_LOG_CACHED = os.environ.get(
    "VLLM_SPARSE_FULL_CUDAGRAPH_HOOK_PROFILE_LOG",
    "",
)
_FULL_CUDAGRAPH_REPLAY_CUDA_EVENT_LOG_CACHED = os.environ.get(
    "VLLM_SPARSE_FULL_CUDAGRAPH_REPLAY_CUDA_EVENT_LOG",
    "",
)
_FULL_CUDAGRAPH_REPLAY_CUDA_EVENT_ROWS: List[dict] = []
_FULL_CUDAGRAPH_REPLAY_CUDA_EVENT_PATH = ""
_FULL_CUDAGRAPH_REPLAY_CUDA_EVENT_REGISTERED = False
_FA3_ROUTE_TRACE_LOG_CACHED = os.environ.get("VLLM_SPARSE_FA3_ROUTE_TRACE_LOG", "")
# [TPX-D5] prepare 步长取证与 FA3 trace writer 共享唯一规范开关；进程内不变。
_FA3_ROUTE_TRACE_LOG_RAW_CACHED = bool(
    str(os.environ.get("VLLM_SPARSE_FA3_ROUTE_TRACE_LOG", "")).strip()
)
_BOOTSTRAP_BRIDGE_GRAPH_POLICY_CACHED = os.environ.get(
    "VLLM_SPARSE_BOOTSTRAP_BRIDGE_GRAPH_POLICY",
    "",
).strip()

# Cache for scheduler_output's finished request IDs attribute name.
_FINISHED_REQ_IDS_ATTR: Optional[str] = None
_COMPACT_PAGE_RESIDENCY_PATCHED: bool = False
_ORIGINAL_KV_CACHE_MANAGER_INIT = None
_ORIGINAL_BLOCK_POOL_METHODS = None
_COMPACT_PAGE_BLOCK_POOL_CLS = None
_COMPACT_PAGE_KV_CACHE_MANAGER_CLS = None


def _cached_or_dynamic_env(name: str, cached: str, default: str = "") -> str:
    if _DYNAMIC_ENV:
        return os.environ.get(name, default)
    return cached


def _full_cudagraph_hook_profile_log(refresh_enabled: bool) -> str:
    if not refresh_enabled:
        return ""
    return _cached_or_dynamic_env(
        "VLLM_SPARSE_FULL_CUDAGRAPH_HOOK_PROFILE_LOG",
        _FULL_CUDAGRAPH_HOOK_PROFILE_LOG_CACHED,
    )




def _full_cudagraph_replay_cuda_event_log(refresh_enabled: bool) -> str:
    log_all_replays = str(
        os.environ.get("VLLM_FULL_CUDAGRAPH_REPLAY_CUDA_EVENT_LOG_ALL", "0")
        or "0"
    ).strip().lower() in {"1", "true", "yes", "on"}
    if not refresh_enabled and not log_all_replays:
        return ""
    return _cached_or_dynamic_env(
        "VLLM_SPARSE_FULL_CUDAGRAPH_REPLAY_CUDA_EVENT_LOG",
        _FULL_CUDAGRAPH_REPLAY_CUDA_EVENT_LOG_CACHED,
    )


from patches.cudagraph_mode_gate import (  # PHASE-2B de-legacy gate (real cudagraph mode)
    attention_in_cudagraph_enabled as _attention_in_cudagraph_enabled,
    dummy_context_cudagraph_is_full as _dummy_context_cudagraph_is_full,
)


def _fa3_route_trace_log() -> str:
    return _cached_or_dynamic_env(
        "VLLM_SPARSE_FA3_ROUTE_TRACE_LOG",
        _FA3_ROUTE_TRACE_LOG_CACHED,
    )


def _fa3_route_trace_enabled() -> bool:
    return bool(_fa3_route_trace_log())


def _bootstrap_bridge_graph_policy() -> str:
    return _cached_or_dynamic_env(
        "VLLM_SPARSE_BOOTSTRAP_BRIDGE_GRAPH_POLICY",
        _BOOTSTRAP_BRIDGE_GRAPH_POLICY_CACHED,
    ).strip()


def _get_v1_flash_attn_modules():
    global _V1_FLASH_ATTN_MODULES
    if _V1_FLASH_ATTN_MODULES is None:
        from vllm.v1.attention.backends import flash_attn as v1_flash_attn
        fa_utils = import_fa_utils_module()

        _V1_FLASH_ATTN_MODULES = (v1_flash_attn, fa_utils)
    return _V1_FLASH_ATTN_MODULES


def _call_original_v1_flash_attn_forward(
    *,
    self,
    layer,
    query,
    key,
    value,
    kv_cache,
    attn_metadata,
    output,
    output_scale,
    output_block_scale,
):
    if _ORIGINAL_V1_FLASH_ATTN_FORWARD is None:
        raise RuntimeError("original FlashAttentionImpl.forward is not initialized")
    if output_block_scale is None:
        return _ORIGINAL_V1_FLASH_ATTN_FORWARD(
            self,
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
            output_scale,
        )
    return _ORIGINAL_V1_FLASH_ATTN_FORWARD(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output,
        output_scale,
        output_block_scale,
    )




def _infer_paged_kv_page_size(cache_tensor: object) -> int:
    if not isinstance(cache_tensor, torch.Tensor):
        return 0
    if cache_tensor.dim() >= 5:
        return int(cache_tensor.shape[2])
    if cache_tensor.dim() >= 4:
        return int(cache_tensor.shape[1])
    if cache_tensor.dim() >= 3:
        return int(cache_tensor.shape[0])
    return 0


def _first_kv_cache_tensor(kv_cache_obj: object) -> Optional[torch.Tensor]:
    if kv_cache_obj is None:
        return None
    if isinstance(kv_cache_obj, torch.Tensor):
        return kv_cache_obj
    if isinstance(kv_cache_obj, (list, tuple)):
        if len(kv_cache_obj) == 0:
            return None
        first = kv_cache_obj[0]
        return first if isinstance(first, torch.Tensor) else None
    return None


def _get_block_table_device_tensor(table: object, num_reqs: int) -> torch.Tensor:
    if hasattr(table, "get_device_tensor"):
        return table.get_device_tensor(int(num_reqs))
    return table[: int(num_reqs)]


def _get_block_table_cpu_source(table: object, num_reqs: int) -> object | None:
    # [BIND-CPP-CANONICAL-TENSOR 2026-07-09] canonical_cpu 全链合同=torch CPU
    # tensor（rrp bind C++ host 推导快路径的 BUFFER ABI 只收 tensor）。此前
    # 优先 get_numpy_array() 喂出 numpy → 稳态 full-bind 步 canonical 在 GPU
    # 时 100% ValueError 静默落穿、Python 慢 loop 双跑（短轮取证 fired=146 /
    # fallthrough=122，全部同型别雷）。numpy 与 cpu tensor 在 vLLM BlockTable
    # 里共享同一存储，from_numpy 为零拷贝视图，内容语义不变。
    cpu_source = None
    if hasattr(table, "get_cpu_tensor"):
        cpu_source = table.get_cpu_tensor()
    if cpu_source is None and hasattr(table, "get_numpy_array"):
        numpy_source = table.get_numpy_array()
        if numpy_source is not None:
            cpu_source = torch.from_numpy(numpy_source)
    if (
        cpu_source is None
        and isinstance(table, torch.Tensor)
        and table.device.type == "cpu"
    ):
        cpu_source = table
    if cpu_source is None:
        return None
    return cpu_source[: int(num_reqs)]


def _slice_block_table_cpu_rows(
    block_table_cpu: object | None,
    row_indices: Sequence[int],
) -> object | None:
    if block_table_cpu is None:
        return None
    rows = [int(row) for row in row_indices]
    if not rows:
        try:
            return block_table_cpu[:0]  # type: ignore[index]
        except Exception:
            return tuple()
    try:
        return block_table_cpu[rows]  # type: ignore[index]
    except Exception:
        return tuple(block_table_cpu[row] for row in rows)  # type: ignore[index]


def _device_identity(device: torch.device) -> Tuple[str, int]:
    return (str(device.type), -1 if device.index is None else int(device.index))


def _positive_int_value(value: object, *, label: str) -> int:
    if value is None:
        raise RuntimeError(f"{label} is required")
    value_i = int(value)
    if value_i <= 0:
        raise RuntimeError(f"{label} must be positive")
    return value_i


def _profile_mixed_page_block_table(
    *,
    attn_metadata: object,
) -> torch.Tensor:
    metadata_block_table = getattr(attn_metadata, "block_table", None)
    if isinstance(metadata_block_table, torch.Tensor):
        return metadata_block_table
    raise RuntimeError(
            "vLLM profile ResolvedRowPtr metadata requires tensor block_table"
    )


def _validate_profile_block_table(
    *,
    block_table: object,
    batch_size: int,
    max_pages_per_row: int,
    device: torch.device,
) -> torch.Tensor:
    if not isinstance(block_table, torch.Tensor):
        raise TypeError("block_table must be a torch.Tensor")
    if block_table.dtype != torch.int32:
        raise ValueError("block_table must have dtype torch.int32")
    if block_table.dim() != 2:
        raise ValueError("block_table must be rank 2")
    if block_table.device != device:
        raise ValueError("block_table device must match profile resolver device")
    if int(block_table.shape[0]) < int(batch_size):
        raise ValueError("block_table rows must cover batch_size")
    if int(block_table.shape[1]) < int(max_pages_per_row):
        raise ValueError("block_table cols must cover max_pages_per_row")
    if int(block_table.stride(1)) != 1:
        raise ValueError("block_table last-dim stride must be 1")
    return block_table


def _round_up_to_page_tokens(value: int, block_size: int) -> int:
    block = max(1, int(block_size))
    tokens = max(0, int(value))
    return ((tokens + block - 1) // block) * block


def _profile_resolved_row_ptr_sparse_capture_bound(
    *,
    controller: object,
    block_size: int,
) -> int:
    config = getattr(controller, "config", None)
    compact_blocks = getattr(config, "compact_blocks_per_slot", None)
    if compact_blocks is None:
        return 0
    block = max(1, int(block_size))
    compact_tokens = max(0, int(compact_blocks)) * block
    recent_tokens = max(0, int(getattr(config, "recent", 0) or 0))
    sink_tokens = max(0, int(getattr(config, "sink", 0) or 0))
    if compact_tokens <= 0 and recent_tokens <= 0 and sink_tokens <= 0:
        return 0
    recent_sink_bound = _round_up_to_page_tokens(recent_tokens + sink_tokens, block)
    return compact_tokens + recent_sink_bound + block


def _profile_resolved_row_ptr_capture_max_seqlen_k(
    *,
    controller: object,
    block_size: int,
    context_kv_len_by_row: Tuple[int, ...],
    profile_max_seq_len: object | None = None,
) -> int:
    if isinstance(profile_max_seq_len, torch.Tensor):
        raise RuntimeError(
            "vLLM profile ResolvedRowPtr max_seq_len must be CPU-owned metadata"
        )
    context_max = max((max(0, int(v)) for v in context_kv_len_by_row), default=0)
    sparse_bound = _profile_resolved_row_ptr_sparse_capture_bound(
        controller=controller,
        block_size=block_size,
    )
    if sparse_bound > 0:
        return max(1, context_max, int(sparse_bound))

    profile_max = 0 if profile_max_seq_len is None else max(0, int(profile_max_seq_len))
    return max(1, context_max, profile_max)
def _bind_resolved_row_ptr_profile_metadata(
    *,
    attn_metadata: object,
    arena: object,
    batch_size: int,
    num_kv_heads: int,
    page_block_size: int,
    max_pages_per_row: int,
    q_layout_key: str,
) -> None:
    from patches.fa_sparse_runtime.resolved_row_ptr_arena import (
        bind_resolved_row_ptr_replay_metadata,
    )

    bind_resolved_row_ptr_replay_metadata(
        attn_metadata=attn_metadata,
        replay_arena=arena,
        source_arena=arena,
        batch_size=int(batch_size),
        num_kv_heads=int(num_kv_heads),
        page_block_size=int(page_block_size),
        max_pages_per_row=int(max_pages_per_row),
        q_layout_key=str(q_layout_key),
        capture_buffer_shape_key="resolved_row_ptr:profile",
        producer_stream_key="resolved-row-ptr-profile-arena",
    )


def _bind_profile_resolved_row_ptr_visible_source(
    *,
    controller: object,
    arena: object,
    batch_size: int,
    device: torch.device,
    block_size: int,
    num_kv_heads: int,
    max_pages_per_row: int,
    profile_capture_max_seqlen_k: int,
) -> torch.Tensor:
    profile_k = int(profile_capture_max_seqlen_k)
    del controller, batch_size, device, block_size, num_kv_heads, max_pages_per_row
    batch_seqused = getattr(arena, "batch_seqused_k_i32", None)
    if not isinstance(batch_seqused, torch.Tensor):
        raise RuntimeError("vLLM profile ResolvedRowPtr metadata requires arena seqused buffers")
    batch_seqused.fill_(profile_k)
    arena.bind_resolved_seqused_source(
        batch_seqused,
        source_kind="arena_batch_seqused",
    )
    return batch_seqused


def _append_profile_resolved_row_ptr_bind_snapshot(
    *,
    label: str,
    attn_metadata: object,
    arena: object,
    arena_key: tuple[object, ...],
    batch_size: int,
    num_kv_heads: int,
    q_lens_by_row: tuple[int, ...],
    q_start_loc: tuple[int, ...],
) -> None:
    if os.environ.get("VLLM_SPARSE_RRP_BIND_SNAPSHOT", "0") != "1":
        return
    if not _fa3_route_trace_enabled():
        return
    try:
        from patches.fa3_native.install import append_fa3_route_trace
    except Exception:
        return

    def _data_ptr(tensor: object) -> int:
        return int(tensor.data_ptr()) if isinstance(tensor, torch.Tensor) else 0

    def _shape(tensor: object) -> list[int]:
        if not isinstance(tensor, torch.Tensor):
            return []
        return [int(v) for v in tuple(tensor.shape)]

    carriers = getattr(attn_metadata, "mixed_page_resolver_replay_carriers", None)
    if carriers is None:
        carriers = getattr(arena, "carriers", None)
    visible = getattr(carriers, "resolver_visible_seqused_k_by_head_i32", None)
    pointer_signature: list[int] = []
    if carriers is not None and hasattr(carriers, "pointer_signature"):
        pointer_signature = [
            0 if value is None else int(value)
            for value in tuple(carriers.pointer_signature())
        ]
    captured_signature = getattr(
        attn_metadata,
        "mixed_page_resolver_captured_pointer_signature",
        None,
    )
    captured_pointer_signature = (
        [
            0 if value is None else int(value)
            for value in tuple(captured_signature)
        ]
        if isinstance(captured_signature, (list, tuple))
        else []
    )
    arena_seqused = getattr(arena, "seqused_k_i32", None)
    batch_seqused = getattr(arena, "batch_seqused_k_i32", None)
    append_fa3_route_trace(
        {
            "event": "rrp_profile_bind_snapshot",
            "label": str(label),
            "arena_key": [str(value) for value in tuple(arena_key)],
            "batch_size": int(batch_size),
            "num_kv_heads": int(num_kv_heads),
            "q_lens": [int(v) for v in q_lens_by_row[: int(batch_size)]],
            "q_start_loc": [int(v) for v in q_start_loc[: int(batch_size) + 1]],
            "pointer_signature": pointer_signature,
            "captured_pointer_signature": captured_pointer_signature,
            "row_ptr_data_ptr": _data_ptr(getattr(arena, "carrier_u64", None)),
            "row_table_data_ptr": _data_ptr(getattr(arena, "row_table_i32", None)),
            "visible_data_ptr": _data_ptr(visible),
            "visible_shape": _shape(visible),
            "arena_seqused_data_ptr": _data_ptr(arena_seqused),
            "arena_seqused_shape": _shape(arena_seqused),
            "arena_batch_seqused_data_ptr": _data_ptr(batch_seqused),
            "arena_batch_seqused_shape": _shape(batch_seqused),
            "visible_is_arena_seqused": bool(
                _data_ptr(visible) != 0 and _data_ptr(visible) == _data_ptr(arena_seqused)
            ),
        }
    )


def _bind_profile_resolved_row_ptr_metadata(
    *,
    controller: object,
    attn_metadata: object,
    kv_cache_spec: object | None,
) -> bool:
    if not _profile_mixed_page_cudagraph_capture_enabled(controller):
        return False

    profile_ctx = getattr(attn_metadata, "sparse_step_context", None)
    profile_authority = getattr(profile_ctx, "step_authority", None)
    if profile_ctx is None or profile_authority is None:
        raise RuntimeError(
            "vLLM profile ResolvedRowPtr metadata requires sparse profile context"
        )

    batch_size = int(getattr(profile_ctx, "num_reqs", 0) or 0)
    if batch_size <= 0:
        return False
    block_size = _positive_int_value(
        getattr(
            kv_cache_spec,
            "block_size",
            getattr(controller, "kv_cache_block_size", None),
        ),
        label="kv_cache_spec.block_size",
    )
    num_kv_heads = _positive_int_value(
        getattr(
            kv_cache_spec,
            "num_kv_heads",
            getattr(controller, "kv_cache_num_kv_heads", None),
        ),
        label="kv_cache_spec.num_kv_heads",
    )

    block_table = _profile_mixed_page_block_table(
        attn_metadata=attn_metadata,
    )
    device = block_table.device

    block_table_i32 = _validate_profile_block_table(
        block_table=block_table,
        batch_size=batch_size,
        max_pages_per_row=1,
        device=device,
    )
    block_table_capacity_pages = int(block_table_i32.shape[1])

    context_kv_len_by_row = tuple(
        int(v)
        for v in tuple(
            getattr(
                profile_authority,
                "context_kv_len_by_row",
                getattr(profile_ctx, "seq_lens", tuple()),
            )
        )[:batch_size]
    )
    if len(context_kv_len_by_row) < batch_size:
        raise RuntimeError(
            "vLLM profile ResolvedRowPtr metadata requires context length coverage"
        )
    profile_capture_max_seqlen_k = _profile_resolved_row_ptr_capture_max_seqlen_k(
        controller=controller,
        block_size=block_size,
        context_kv_len_by_row=context_kv_len_by_row,
        profile_max_seq_len=getattr(profile_ctx, "max_seq_len", None),
    )
    profile_row_effective_k_by_row = tuple(
        int(profile_capture_max_seqlen_k)
        for _ in range(batch_size)
    )
    required_pages_per_row = max(
        1,
        (int(profile_capture_max_seqlen_k) + block_size - 1) // block_size,
    )
    if required_pages_per_row > block_table_capacity_pages:
        raise RuntimeError(
            "vLLM profile ResolvedRowPtr metadata requires block_table coverage; "
            f"required={required_pages_per_row} capacity={block_table_capacity_pages}"
        )

    row_mode_by_row = tuple(
        int(v)
        for v in tuple(getattr(profile_authority, "row_mode_by_row", tuple()))[
            :batch_size
        ]
    )
    if len(row_mode_by_row) < batch_size:
        raise RuntimeError(
            "vLLM profile ResolvedRowPtr metadata requires row_mode coverage"
        )
    q_lens_by_row = tuple(
        int(v)
        for v in tuple(
            getattr(
                profile_authority,
                "q_lens_by_row",
                getattr(profile_ctx, "q_lens", tuple()),
            )
        )[:batch_size]
    )
    q_start_loc = tuple(
        int(v) for v in tuple(getattr(profile_authority, "q_start_loc", tuple()))[
            : batch_size + 1
        ]
    )
    if len(q_lens_by_row) < batch_size or len(q_start_loc) < batch_size + 1:
        raise RuntimeError(
            "vLLM profile ResolvedRowPtr metadata requires q layout coverage"
        )
    if any(int(q_len) > 1 for q_len in q_lens_by_row):
        return False

    arena_key = (
        batch_size,
        num_kv_heads,
        block_size,
        block_table_capacity_pages,
        _device_identity(device),
    )
    arena_by_key = getattr(controller, "_resolved_row_ptr_replay_arena_by_key", None)
    if not isinstance(arena_by_key, dict):
        arena_by_key = {}
        setattr(controller, "_resolved_row_ptr_replay_arena_by_key", arena_by_key)
    arena = arena_by_key.get(arena_key)
    if arena is None:
        from patches.fa_sparse_runtime.resolved_row_ptr_arena import (
            ResolvedRowPtrArena,
        )

        arena = ResolvedRowPtrArena.allocate(
            batch_size=batch_size,
            num_kv_heads=num_kv_heads,
            max_pages_per_row=block_table_capacity_pages,
            device=device,
        )
        arena_by_key[arena_key] = arena
    setattr(controller, "_resolved_row_ptr_replay_arena", arena)
    setattr(controller, "_resolved_row_ptr_source_arena", arena)
    setattr(controller, "_resolved_row_ptr_arena_key", arena_key)

    del row_mode_by_row
    arena.bind_production_row_table(
        canonical_block_table=block_table_i32,
        compact_ready_by_batch_row=(False,) * batch_size,
        row_effective_k_by_row=profile_row_effective_k_by_row,
        page_size=block_size,
    )
    visible_seqused = _bind_profile_resolved_row_ptr_visible_source(
        controller=controller,
        arena=arena,
        batch_size=batch_size,
        device=device,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        max_pages_per_row=block_table_capacity_pages,
        profile_capture_max_seqlen_k=int(profile_capture_max_seqlen_k),
    )
    setattr(attn_metadata, "seq_lens", visible_seqused)
    setattr(
        attn_metadata,
        "mixed_page_profile_max_seqlen_k",
        int(profile_capture_max_seqlen_k),
    )
    _bind_resolved_row_ptr_profile_metadata(
        attn_metadata=attn_metadata,
        arena=arena,
        batch_size=batch_size,
        num_kv_heads=num_kv_heads,
        page_block_size=block_size,
        max_pages_per_row=block_table_capacity_pages,
        q_layout_key=f"q_lens={q_lens_by_row};q_start={q_start_loc}",
    )
    _append_profile_resolved_row_ptr_bind_snapshot(
        label="profile_capture_bind",
        attn_metadata=attn_metadata,
        arena=arena,
        arena_key=arena_key,
        batch_size=batch_size,
        num_kv_heads=num_kv_heads,
        q_lens_by_row=q_lens_by_row,
        q_start_loc=q_start_loc,
    )
    return True


def _maybe_build_step_decode_data_from_attn_metadata(
    *,
    controller: object | None,
    attn_metadata: object | None,
    metadata_builder: object | None,
) -> bool:
    if controller is None or attn_metadata is None:
        return False
    live_step_context = getattr(controller, "step_context", None)
    metadata_step_context = getattr(attn_metadata, "sparse_step_context", None)
    is_profile_metadata = bool(
        getattr(attn_metadata, "sparse_vllm_profile_step", False)
    )
    if live_step_context is None and metadata_step_context is None:
        return False
    kv_cache_spec = (
        getattr(metadata_builder, "kv_cache_spec", None)
        if metadata_builder is not None
        else None
    )
    if is_profile_metadata:
        return _bind_profile_resolved_row_ptr_metadata(
            controller=controller,
            attn_metadata=attn_metadata,
            kv_cache_spec=kv_cache_spec,
        )
    if live_step_context is not None and not is_profile_metadata:
        controller.maybe_build_step_decode_data_from_metadata(
            attn_metadata=attn_metadata,
            kv_cache_spec=kv_cache_spec,
        )
        return True
    return False


def _mixed_page_full_cudagraph_replay_stats_for_step(
    cached_stats: object,
    *,
    step_id: int,
    graph_key: str,
    updated_metadata_count: int | None = None,
) -> object:
    try:
        return cached_stats.__class__(
            metadata_count=int(getattr(cached_stats, "metadata_count")),
            updated_metadata_count=(
                int(updated_metadata_count)
                if updated_metadata_count is not None
                else int(getattr(cached_stats, "updated_metadata_count"))
            ),
            carrier_update_rows=int(getattr(cached_stats, "carrier_update_rows")),
            carrier_update_bytes=int(getattr(cached_stats, "carrier_update_bytes")),
            carrier_update_kernel_count=int(
                getattr(cached_stats, "carrier_update_kernel_count")
            ),
            row_mode_distribution=dict(
                getattr(cached_stats, "row_mode_distribution", {})
            ),
            row_source_distribution=dict(
                getattr(cached_stats, "row_source_distribution", {})
            ),
            source_counter_schema_version=int(
                getattr(cached_stats, "source_counter_schema_version", -1)
            ),
            expected_rows=int(getattr(cached_stats, "expected_rows", -1)),
            num_kv_heads=int(getattr(cached_stats, "num_kv_heads", -1)),
            source_counter_missing_fields=tuple(
                str(field)
                for field in getattr(
                    cached_stats,
                    "source_counter_missing_fields",
                    (),
                )
                if str(field)
            ),
            step_id=step_id,
            graph_key=graph_key,
        )
    except Exception:
        return cached_stats


def _mixed_page_replay_stats_source_counter_payload(stats: object) -> dict[str, object]:
    from patches.fa_sparse_runtime.resolved_row_ptr_arena import (
        SOURCE_COUNTER_SCHEMA_VERSION,
    )

    def _stats_int(attr_name: str) -> int:
        try:
            return int(getattr(stats, attr_name, -1))
        except (TypeError, ValueError):
            return -1

    fields = getattr(stats, "source_counter_missing_fields", ())
    if isinstance(fields, str):
        missing_fields = {fields} if fields else set()
    elif isinstance(fields, (list, tuple)):
        missing_fields = {str(field) for field in fields if str(field)}
    else:
        missing_fields = set()
    source_counter_schema_version = _stats_int("source_counter_schema_version")
    expected_rows = _stats_int("expected_rows")
    num_kv_heads = _stats_int("num_kv_heads")
    if not hasattr(stats, "source_counter_schema_version") or source_counter_schema_version < 0:
        missing_fields.add("source_counter_schema_version")
    elif source_counter_schema_version != SOURCE_COUNTER_SCHEMA_VERSION:
        missing_fields.add("source_counter_schema_version_unsupported")
    if not hasattr(stats, "expected_rows") or expected_rows < 0:
        missing_fields.add("expected_rows")
    if not hasattr(stats, "num_kv_heads") or num_kv_heads < 0:
        missing_fields.add("num_kv_heads")
    return {
        "source_counter_schema_version": source_counter_schema_version,
        "expected_rows": expected_rows,
        "num_kv_heads": num_kv_heads,
        "source_counter_missing_fields": sorted(missing_fields),
    }


def _refresh_mixed_page_resolver_replay_carriers_from_attn_metadata(attn_metadata: object) -> dict[str, int]:
    from patches.fa_sparse_runtime.effective_page_table_producer import (
        refresh_mixed_page_resolver_carriers_for_replay,
    )

    return refresh_mixed_page_resolver_carriers_for_replay(
        attn_metadata=attn_metadata,
        stream_key="vllm-cudagraph-direct-bound",
    )


def _mixed_page_resolver_replay_carriers_updated(attn_metadata: object) -> bool:
    return bool(
        getattr(attn_metadata, "mixed_page_resolver_replay_carriers_updated", False)
    )


def _mixed_page_resolver_graph_replay_expected(attn_metadata: object) -> bool:
    return bool(
        getattr(attn_metadata, "mixed_page_resolver_graph_replay_expected", False)
    )


def _mixed_page_resolver_kwargs_from_attn_metadata(
    attn_metadata: object,
) -> Dict[str, object]:
    descriptor = getattr(attn_metadata, "mixed_page_resolver_descriptor", None)
    carriers = getattr(attn_metadata, "mixed_page_resolver_replay_carriers", None)
    if descriptor is None and carriers is None:
        return {}

    return {
        "resolver_descriptor": descriptor,
        "resolver_carriers": carriers,
        "resolver_seqused_k": getattr(
            attn_metadata,
            "mixed_page_resolver_replay_seqused_k_i32",
            None,
        ),
        "captured_resolver_descriptor": getattr(
            attn_metadata,
            "mixed_page_resolver_captured_descriptor",
            descriptor,
        ),
        "captured_resolver_pointer_signature": getattr(
            attn_metadata,
            "mixed_page_resolver_captured_pointer_signature",
            None,
        ),
        "graph_replay_carriers": _mixed_page_resolver_graph_replay_expected(
            attn_metadata
        ),
    }


@dataclass(frozen=True)
class _MetadataStepBinding:
    step_ctx: object | None
    step_authority: object | None
    snapshot: object | None
    is_profile_step: bool


def _maybe_cpu_int_tuple(value: object, *, label: str) -> Tuple[int, ...] | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            return None
        return tuple(int(v) for v in value.reshape(-1).tolist())
    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value)
    if hasattr(value, "tolist"):
        raw = value.tolist()
        if isinstance(raw, (int, float)):
            return (int(raw),)
        if isinstance(raw, list):
            return tuple(int(v) for v in raw)
    raise RuntimeError(f"{label} must be a CPU integer sequence")


def _first_cpu_int_tuple(
    *,
    sources: Tuple[object | None, ...],
    names: Tuple[str, ...],
    label: str,
) -> Tuple[int, ...]:
    for source in sources:
        if source is None:
            continue
        for name in names:
            try:
                value = getattr(source, name)
            except Exception:
                continue
            parsed = _maybe_cpu_int_tuple(value, label=f"{label}.{name}")
            if parsed is not None:
                return parsed
    raise RuntimeError(
        f"vLLM profile step requires CPU {label}; refusing implicit GPU sync"
    )


def _build_vllm_profile_step_context(
    *,
    controller: object,
    attn_metadata: object,
    common_attn_metadata: object | None,
) -> StepContext:
    from patches.sparse_constants import (
        _LOGF_PRODUCER_NONE,
        _ROW_MODE_DENSE,
    )

    sources = (common_attn_metadata, attn_metadata)
    q_start_loc = _first_cpu_int_tuple(
        sources=sources,
        names=("query_start_loc_cpu", "query_start_loc"),
        label="query_start_loc",
    )
    if len(q_start_loc) < 1:
        raise RuntimeError("vLLM profile step requires non-empty query_start_loc")

    seq_lens = _first_cpu_int_tuple(
        sources=sources,
        names=("_seq_lens_cpu", "seq_lens"),
        label="seq_lens",
    )
    inferred_batch = len(q_start_loc) - 1
    requested_batch = int(
        getattr(common_attn_metadata, "num_reqs", inferred_batch)
        if common_attn_metadata is not None
        else inferred_batch
    )
    batch_size = min(inferred_batch, requested_batch, len(seq_lens))
    if batch_size < 0:
        raise RuntimeError("vLLM profile step resolved negative batch size")
    if len(q_start_loc) < batch_size + 1 or len(seq_lens) < batch_size:
        raise RuntimeError(
            "vLLM profile step metadata row coverage is incomplete"
        )

    q_start_loc = tuple(int(v) for v in q_start_loc[: batch_size + 1])
    q_lens = tuple(
        max(0, int(q_start_loc[row + 1]) - int(q_start_loc[row]))
        for row in range(batch_size)
    )
    seq_lens = tuple(max(0, int(v)) for v in seq_lens[:batch_size])
    max_query_len = int(
        getattr(common_attn_metadata, "max_query_len", max(q_lens, default=0))
        if common_attn_metadata is not None
        else max(q_lens, default=0)
    )
    max_seq_len = int(
        getattr(common_attn_metadata, "max_seq_len", max(seq_lens, default=0))
        if common_attn_metadata is not None
        else max(seq_lens, default=0)
    )
    num_actual_tokens = int(
        getattr(common_attn_metadata, "num_actual_tokens", q_start_loc[-1])
        if common_attn_metadata is not None
        else q_start_loc[-1]
    )

    req_ids = tuple(f"_vllm_profile_req_{row}" for row in range(batch_size))
    req_id_to_index = {req_id: idx for idx, req_id in enumerate(req_ids)}
    is_prefill_by_row = tuple(int(q_len) > 1 for q_len in q_lens)
    prefill_rows = tuple(
        row for row, is_prefill in enumerate(is_prefill_by_row) if is_prefill
    )
    has_prefill_row = any(is_prefill_by_row)
    has_decode_row = any(not is_prefill for is_prefill in is_prefill_by_row)
    config = getattr(controller, "config", None)
    false_by_row = (False,) * batch_size
    profile_decode_like = all(int(q_len) <= 1 for q_len in q_lens)
    profile_mixed_page_capture = (
        bool(profile_decode_like)
        and _profile_mixed_page_cudagraph_capture_enabled(controller)
    )
    use_compact_by_row = false_by_row
    zero_by_row = (0,) * batch_size
    row_mode_by_row = (int(_ROW_MODE_DENSE),) * batch_size
    none_logf = (int(_LOGF_PRODUCER_NONE),) * batch_size
    slot_by_row = tuple(range(batch_size))
    plan_signature_prefix = (
        "vllm_profile_resolved_row_ptr"
        if profile_mixed_page_capture
        else "vllm_profile_dense"
    )

    authority = StepAuthority(
        epoch=-1,
        step_handle_id=-1,
        step_handle_generation=-1,
        batch_size=batch_size,
        max_batch_size=batch_size,
        req_ids=req_ids,
        req_id_to_index=req_id_to_index,
        q_lens_by_row=q_lens,
        context_kv_len_by_row=seq_lens,
        q_start_loc=q_start_loc,
        is_prefill_by_row=is_prefill_by_row,
        has_prefill_row=has_prefill_row,
        has_decode_row=has_decode_row,
        prefill_rows=prefill_rows,
        is_decode_only=not has_prefill_row,
        has_prefill_by_prompt=has_prefill_row,
        row_policy_ready_by_row=(True,) * batch_size,
        short_dense_by_row=false_by_row,
        slot_by_row=slot_by_row,
        row_mode_by_row=row_mode_by_row,
        refresh_mode_by_row=zero_by_row,
        layer_effective_refresh_by_row=false_by_row,
        logf_producer_by_row=none_logf,
        logf_attn_rows=tuple(),
        logf_mask_by_row=zero_by_row,
        logf_stride_head=0,
        logf_dirty_rows=tuple(),
        logits_last_n_by_row=zero_by_row,
        logits_capacity_by_row=zero_by_row,
        step_identity_token=-1,
        use_compact_by_row=use_compact_by_row,
        recent_cap=int(getattr(config, "recent", 0) or 0),
        sink_tokens=int(getattr(config, "sink", 0) or 0),
        compact_bootstrap_threshold=0,
        plan_signature=(plan_signature_prefix, batch_size, q_lens, seq_lens),
        req_set_hash=int(hash(req_ids) & 0x7FFFFFFFFFFFFFFF),
        row_phase_hash=int(hash(is_prefill_by_row) & 0x7FFFFFFFFFFFFFFF),
        has_request_phase_mix=bool(
            any(is_prefill_by_row) and any(not v for v in is_prefill_by_row)
        ),
        has_request_phase_mix_i32=int(
            any(is_prefill_by_row) and any(not v for v in is_prefill_by_row)
        ),
        needs_logits_by_row=false_by_row,
        dispatch_logf_producer_by_row=none_logf,
    )
    return StepContext(
        req_ids=req_ids,
        num_reqs=batch_size,
        num_actual_tokens=max(0, num_actual_tokens),
        q_start_loc=q_start_loc,
        q_lens=q_lens,
        seq_lens=seq_lens,
        max_query_len=max_query_len,
        max_seq_len=max_seq_len,
        epoch=-1,
        req_id_to_index=req_id_to_index,
        step_handle_id=-1,
        step_handle_generation=-1,
        prompt_lens=seq_lens,
        num_computed_tokens=tuple(
            max(0, int(seq_lens[row]) - int(q_lens[row]))
            for row in range(batch_size)
        ),
        step_authority=authority,
        step_identity_token=-1,
    )


def _is_vllm_dummy_run_active(controller: object | None) -> bool:
    if controller is None:
        return False
    return int(getattr(controller, "_vllm_dummy_run_depth", 0) or 0) > 0


def _mixed_page_cudagraph_config_enabled(controller: object | None) -> bool:
    if controller is None:
        return False
    config = getattr(controller, "config", None)
    return bool(
        getattr(config, "enabled", False)
        and getattr(config, "compact_page_residency_enabled", False)
    )


def _mixed_page_full_cudagraph_replay_refresh_enabled(
    controller: object | None,
) -> bool:
    return bool(
        _mixed_page_cudagraph_config_enabled(controller)
        and _attention_in_cudagraph_enabled(controller)
    )


def _compact_recent_launch_plan_trace_payload(
    *,
    controller: object,
    page_size: int,
) -> dict[str, object]:
    step_bound_meta = getattr(controller, "step_bound_meta", None)
    launch_plan = getattr(step_bound_meta, "compact_recent_launch_plan", None)
    if launch_plan is None or not bool(getattr(launch_plan, "valid", False)):
        return {}
    recent_first_pages = [
        int(v) for v in tuple(getattr(launch_plan, "recent_first_cpu", tuple()))
    ]
    request_recent_lens = [
        int(v)
        for v in tuple(getattr(launch_plan, "request_recent_len_cpu", tuple()))
    ]
    compact_valid_tokens = [
        int(v)
        for v in tuple(getattr(launch_plan, "compact_valid_tokens_cpu", tuple()))
    ]
    launch_effective_k = [
        int(v)
        for v in tuple(getattr(launch_plan, "launch_effective_k_len_cpu", tuple()))
    ]
    trace_page_size = int(getattr(launch_plan, "page_size", page_size) or page_size)
    step_ctx = getattr(controller, "step_context", None)
    req_ids_by_row = [
        str(rid) for rid in tuple(getattr(step_ctx, "req_ids", tuple()))
    ]
    return {
        "req_ids_by_row": req_ids_by_row,
        "compact_valid_tokens_by_row": compact_valid_tokens,
        "recent_first_logical_page_by_row": recent_first_pages,
        "recent_first_tokens_by_row": [
            int(page) * int(trace_page_size) for page in recent_first_pages
        ],
        "request_recent_len_by_row": request_recent_lens,
        "launch_effective_k_len_by_row": launch_effective_k,
        "launch_plan_page_size": int(trace_page_size),
    }


def _profile_mixed_page_cudagraph_capture_enabled(controller: object | None) -> bool:
    if not _mixed_page_cudagraph_config_enabled(controller):
        return False
    if bool(getattr(controller, "_sparse_force_mixed_page_profile_capture", False)):
        return True
    dummy_context = getattr(controller, "_vllm_dummy_run_context", None)
    if not isinstance(dummy_context, dict):
        return False
    if bool(dummy_context.get("is_graph_capturing", False)):
        return True
    return _dummy_context_cudagraph_is_full(dummy_context)


def _dummy_run_arg(
    args: tuple[object, ...],
    kwargs: dict[str, object],
    *,
    index: int,
    name: str,
    default: object,
) -> object:
    return args[index] if len(args) > index else kwargs.get(name, default)


def _dummy_run_context_from_call(
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> dict[str, object]:
    return {
        "num_tokens": int(_dummy_run_arg(args, kwargs, index=0, name="num_tokens", default=0) or 0),
        "force_attention": bool(
            _dummy_run_arg(args, kwargs, index=2, name="force_attention", default=False)
        ),
        "is_profile": bool(
            _dummy_run_arg(args, kwargs, index=6, name="is_profile", default=False)
        ),
        "is_graph_capturing": bool(
            _dummy_run_arg(args, kwargs, index=9, name="is_graph_capturing", default=False)
        ),
        "cudagraph_runtime_mode": _dummy_run_arg(
            args,
            kwargs,
            index=1,
            name="cudagraph_runtime_mode",
            default=None,
        ),
    }


def _resolve_metadata_step_binding(
    *,
    controller: object,
    attn_metadata: object,
    common_attn_metadata: object | None,
) -> _MetadataStepBinding:
    if _is_vllm_dummy_run_active(controller):
        step_ctx = _build_vllm_profile_step_context(
            controller=controller,
            attn_metadata=attn_metadata,
            common_attn_metadata=common_attn_metadata,
        )
        return _MetadataStepBinding(
            step_ctx=step_ctx,
            step_authority=getattr(step_ctx, "step_authority", None),
            snapshot=None,
            is_profile_step=True,
        )

    step_ctx = getattr(controller, "step_context", None)
    return _MetadataStepBinding(
        step_ctx=step_ctx,
        step_authority=(
            getattr(step_ctx, "step_authority", None)
            if step_ctx is not None
            else getattr(controller, "step_authority", None)
        ),
        snapshot=getattr(controller, "_active_step_snapshot", None),
        is_profile_step=False,
    )




def _compact_recent_support_error(
    *,
    max_seqlen_q: int,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    softcap: float,
    cp_world_size: int,
    num_splits: int,
    q_v: torch.Tensor | None,
    s_aux: torch.Tensor | None,
    q_descale: torch.Tensor | None,
    k_descale: torch.Tensor | None,
    v_descale: torch.Tensor | None,
) -> str | None:
    if (
        int(max_seqlen_q) == 1
        and bool(is_causal)
        and int(window_size_left) == -1
        and int(window_size_right) == -1
        and float(softcap) == 0.0
        and int(cp_world_size) == 1
        and 0 <= int(num_splits) <= 255
        and q_v is None
        and s_aux is None
        and q_descale is None
        and k_descale is None
        and v_descale is None
    ):
        return None
    try:
        validate_compact_recent_support_matrix(
            CompactRecentLaunchConfig(
                max_seqlen_q=max_seqlen_q,
                is_causal=is_causal,
                window_size_left=window_size_left,
                window_size_right=window_size_right,
                softcap=softcap,
                cp_world_size=cp_world_size,
                num_splits=num_splits,
                q_v=q_v,
                s_aux=s_aux,
                q_descale=q_descale,
                k_descale=k_descale,
                v_descale=v_descale,
            )
        )
    except ValueError as exc:
        return str(exc)
    return None


def _raise_selected_compact_recent_launch_rejected(reason: str) -> None:
    from patches.fa3_native.install import (
        append_fa3_route_trace,
        fa3_route_trace_enabled,
    )

    if fa3_route_trace_enabled():
        append_fa3_route_trace(
            {
                "event": "compact_recent_launch_rejected",
                "mode": "selected",
                "route_source_mode": "selected/no-capture",
                "normalized_launch_mode": "selected/no-capture",
                "kpi_scope": "selected_no_capture_kpi",
                "evidence_scope": "selected_no_capture_kpi",
                "failure_mode": "fail_fast",
                "failure_reason": reason,
            }
        )
    raise RuntimeError(
        "selected/no-capture compact_recent launch-level gate: "
        f"{reason}"
    )


# Non-quantized KV dtypes: descale is semantically identity (no dequantization
# needed), so vLLM's placeholder ``layer._q_scale.expand(...)`` can be safely
# dropped. FP8 dtypes (torch.float8_e4m3fn / e5m2) signal real quantized KV
# where descale values matter — preserve those so the compact_recent contract
# can fail-fast against a path that does not consume them.
_NON_QUANTIZED_KV_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def _strip_identity_descale_kwargs_inplace(kwargs: Dict[str, object]) -> None:
    """Drop vLLM's placeholder descale tensors when KV dtype is non-quantized.

    vLLM's FA3 backend unconditionally passes
    ``q/k/v_descale = layer._q_scale.expand((b, h_kv))`` even for bf16 models
    where ``_q_scale`` is identity 1.0 (no quantization). compact_recent v1
    kernel does not consume descale, and ``compact_recent_contract`` rejects
    any non-None descale defensively. Rather than peek at tensor values
    (``.item()``, ``.all()``, or GPU→CPU sync on the hot path — incompatible
    with CUDA-graph capture), we use the KV cache dtype as the authoritative
    "does this kernel call use quantization?" signal:

    - bf16 / fp16 / fp32 KV → descale is always semantically identity → drop
    - FP8 KV → real descale → preserve (contract rejects, fail-fast)

    Pure CPU metadata check (``k.dtype``), zero GPU ops, zero sync, safe
    under torch.compile / CUDA-graph capture. Runs per layer per step.
    """
    k = kwargs.get("k")
    if not isinstance(k, torch.Tensor):
        return
    if k.dtype not in _NON_QUANTIZED_KV_DTYPES:
        return
    for name in ("q_descale", "k_descale", "v_descale"):
        if kwargs.get(name) is not None:
            kwargs[name] = None




def _normalize_window_tuple(window_size: Tuple[int, int] | List[int] | None) -> Tuple[int, int]:
    if window_size is None:
        return (-1, -1)
    if len(window_size) != 2:
        raise ValueError("window_size must contain exactly 2 entries")
    return (int(window_size[0]), int(window_size[1]))










































def _reject_legacy_selected_request_level_kwargs(
    *,
    wrapper_name: str,
    kwargs: Dict[str, object],
) -> None:
    legacy_keys = []
    for key in ("selected_page_count_i32", "visible_kv_len_i32"):
        if key in kwargs:
            legacy_keys.append(key)
    if legacy_keys:
        legacy_keys_str = ", ".join(legacy_keys)
        raise RuntimeError(
            f"{wrapper_name} does not accept legacy selected request-level kwargs: {legacy_keys_str}"
        )






def _get_finished_req_ids(scheduler_output: object) -> set:
    """Probe and cache which attribute name holds finished request IDs."""
    global _FINISHED_REQ_IDS_ATTR
    if _FINISHED_REQ_IDS_ATTR is not None:
        val = getattr(scheduler_output, _FINISHED_REQ_IDS_ATTR, None)
        if val is not None:
            return val
    for attr in ("finished_req_ids", "finished_requests_ids", "finished_request_ids"):
        val = getattr(scheduler_output, attr, None)
        if val is not None:
            _FINISHED_REQ_IDS_ATTR = attr
            return val
    return set()


def _build_step_ticket(
    *,
    controller: "VLLMSparseController",
    req_ids: List[str],
    num_scheduled_tokens: Dict[str, int],
    finished_req_ids: set,
    dispatch_token: int,
) -> StepTicket:
    dispatch_token = int(dispatch_token)
    if dispatch_token <= 0:
        raise RuntimeError(
            "E_STEP_DISPATCH_TOKEN: _prepare_inputs requires a positive "
            "worker dispatch token"
        )
    req_ids_tuple = tuple(str(rid) for rid in req_ids)
    scheduled_tuple = tuple(int(num_scheduled_tokens.get(rid, 0)) for rid in req_ids_tuple)
    finished_tuple = tuple(
        sorted(
            {
                rid.strip()
                for rid in finished_req_ids
                if isinstance(rid, str) and rid.strip()
            }
        )
    )
    source_signature = int(
        hash((req_ids_tuple, scheduled_tuple, finished_tuple)) & 0x7FFFFFFFFFFFFFFF
    )
    target_epoch = int(getattr(controller, "step_context_epoch", 0)) + 1
    cached_snapshot = getattr(controller, "_active_step_snapshot", None)
    cached_ticket = getattr(controller, "_active_step_ticket", None)
    cached_source_signature = int(
        getattr(controller, "_active_step_source_signature", -1)
    )
    if cached_snapshot is not None and cached_ticket is not None:
        cached_dispatch_token = int(getattr(cached_ticket, "dispatch_token", -1))
        cached_epoch = int(getattr(cached_snapshot, "step_epoch", -1))
        if (
            cached_epoch > 0
            and cached_dispatch_token == dispatch_token
            and cached_source_signature == source_signature
        ):
            # 同一 execute_model dispatch 内复入：复用已分配
            # target_epoch。对象地址不是稳定身份，不得参与判定。
            target_epoch = int(cached_epoch)
    return StepTicket(
        target_epoch=target_epoch,
        req_ids_signature=int(hash(req_ids_tuple) & 0x7FFFFFFFFFFFFFFF),
        scheduled_signature=int(hash(scheduled_tuple) & 0x7FFFFFFFFFFFFFFF),
        finished_signature=int(hash(finished_tuple) & 0x7FFFFFFFFFFFFFFF),
        source_signature=source_signature,
        dispatch_token=dispatch_token,
    )


_PREPARE_PATCHED: bool = False
_ORIGINAL_PREPARE_INPUTS = None
_BATCH_EXECUTION_ADMISSION_PATCHED: bool = False
_ORIGINAL_DETERMINE_BATCH_EXECUTION = None
_INSTALLED_DETERMINE_BATCH_EXECUTION_WRAPPER = None
_DUMMY_RUN_PATCHED: bool = False
_ORIGINAL_DUMMY_RUN = None
_UPDATE_STATES_PATCHED: bool = False
_ORIGINAL_UPDATE_STATES = None
_INSTALLED_UPDATE_STATES_WRAPPER = None
SPARSE_UPDATE_STATES_PREDECESSOR_ABI = "sfi_sparse_update_states_predecessor/v1"
_FLASH_METADATA_PATCHED: bool = False
_ORIGINAL_FLASH_METADATA_BUILD = None
_REQUEST_PATCHED: bool = False
_ORIGINAL_APPEND_OUTPUT_TOKEN_IDS = None
_KV_INIT_PATCHED: bool = False
_ORIGINAL_INIT_KV_CACHE = None


def load_fa3_native_contracts():
    from patches.fa3_native.contracts import FA3NativeContracts

    return FA3NativeContracts()


def bind_fa3_native_attn_metadata_contracts(
    *,
    attn_metadata: object,
    step_ctx: object | None,
    step_authority: object | None = None,
    snapshot: object | None = None,
) -> object:
    from patches.fa3_native.contracts import TargetSelectedScopeKey
    from patches.fa3_native.snapshot_binding import LaunchLocalSnapshot, SelectedScopeKey
    from patches.sparse_constants import _LOGF_PRODUCER_ATTN, _LOGF_PRODUCER_NONE

    def _build_snapshot_from_step_authority() -> object | None:
        effective_step_authority = getattr(step_ctx, "step_authority", None)
        if effective_step_authority is None:
            effective_step_authority = step_authority
        if (
            snapshot is not None
            and hasattr(snapshot, "selected_scope_wait_handle")
            and effective_step_authority is None
        ):
            return snapshot
        if step_ctx is None or effective_step_authority is None:
            return snapshot
        target_scope_key = getattr(effective_step_authority, "target_selected_scope_key", None)
        scope_wait_handle = getattr(effective_step_authority, "selected_scope_wait_handle", None)
        consume_scope_key = getattr(
            effective_step_authority, "consume_selected_scope_key", None
        )
        consume_scope_wait_handle = getattr(
            effective_step_authority, "consume_selected_scope_wait_handle", None
        )
        step_epoch = int(getattr(step_ctx, "epoch", -1))
        if (
            target_scope_key is None
            and step_epoch > 0
        ):
            step_envelope = getattr(step_ctx, "step_envelope_v2", None)
            layer_group_id = int(
                getattr(step_envelope, "layer_group_active", 0)
                if bool(getattr(step_envelope, "layer_group_enabled", False))
                else 0
            )
            if layer_group_id < 0:
                layer_group_id = 0
            target_scope_key = TargetSelectedScopeKey(
                consumer_step_id=step_epoch,
                layer_group_id=layer_group_id,
            )
        if target_scope_key is None or scope_wait_handle is None:
            return snapshot
        has_selected_consume = bool(effective_step_authority.has_compact_row)
        has_capture = bool(effective_step_authority.hint_has_log_f)
        selected_scope_key = SelectedScopeKey(
            int(target_scope_key.consumer_step_id),
            int(target_scope_key.layer_group_id),
            0,
        )
        if consume_scope_key is None:
            consume_scope_key = selected_scope_key
        if consume_scope_wait_handle is None:
            consume_scope_wait_handle = scope_wait_handle
        return LaunchLocalSnapshot(
            target_selected_scope_key=target_scope_key,
            selected_scope_key=selected_scope_key,
            selected_scope_wait_handle=scope_wait_handle,
            mode_signature=(
                "selected" if has_selected_consume else "full",
                "capture" if has_capture else "no_capture",
            ),
            has_selected_consume=has_selected_consume,
            has_capture=has_capture,
            consume_selected_scope_key=consume_scope_key,
            consume_selected_scope_wait_handle=consume_scope_wait_handle,
        )

    snapshot = _build_snapshot_from_step_authority()
    step_handle_id = int(getattr(step_ctx, "step_handle_id", -1)) if step_ctx is not None else -1
    step_handle_generation = (
        int(getattr(step_ctx, "step_handle_generation", -1))
        if step_ctx is not None
        else -1
    )
    setattr(attn_metadata, "sparse_step_handle_id", step_handle_id)
    setattr(attn_metadata, "sparse_step_handle_generation", step_handle_generation)
    setattr(attn_metadata, "sparse_step_context", step_ctx)

    from patches.fa3_native.route_adapter import bind_launch_route_hints
    from patches.fa3_native.snapshot_binding import bind_snapshot

    bind_launch_route_hints(
        attn_metadata,
        has_selected_consume=bool(
            getattr(snapshot, "has_selected_consume", False)
        ),
        has_capture=bool(getattr(snapshot, "has_capture", False)),
    )
    if snapshot is not None:
        bind_snapshot(attn_metadata, snapshot=snapshot)
    return attn_metadata


def _resolve_launch_local_snapshot_from_step_source(
    *,
    attn_metadata: object,
    step_ctx: object,
    step_authority: object | None,
) -> object | None:
    from patches.fa3_native.snapshot_binding import SelectedScopeKey

    snapshot = getattr(attn_metadata, "fa3_native_snapshot", None)
    current_target_scope_key = getattr(step_authority, "target_selected_scope_key", None)
    current_scope_wait_handle = getattr(step_authority, "selected_scope_wait_handle", None)
    current_consume_scope_key = getattr(step_authority, "consume_selected_scope_key", None)
    current_consume_scope_wait_handle = getattr(
        step_authority, "consume_selected_scope_wait_handle", None
    )
    if current_target_scope_key is None or current_scope_wait_handle is None:
        return snapshot
    if current_consume_scope_key is None:
        current_consume_scope_key = SelectedScopeKey(
            int(current_target_scope_key.consumer_step_id),
            int(current_target_scope_key.layer_group_id),
            0,
        )
    if current_consume_scope_wait_handle is None:
        current_consume_scope_wait_handle = current_scope_wait_handle
    if (
        snapshot is not None
        and getattr(snapshot, "target_selected_scope_key", None) == current_target_scope_key
        and getattr(snapshot, "selected_scope_wait_handle", None) is current_scope_wait_handle
        and getattr(snapshot, "consume_selected_scope_key", None) == current_consume_scope_key
        and getattr(snapshot, "consume_selected_scope_wait_handle", None)
        is current_consume_scope_wait_handle
    ):
        return snapshot
    bind_fa3_native_attn_metadata_contracts(
        attn_metadata=attn_metadata,
        step_ctx=step_ctx,
        step_authority=step_authority,
        snapshot=None,
    )
    return getattr(attn_metadata, "fa3_native_snapshot", None)


def _get_global_controller() -> Optional["VLLMSparseController"]:
    return _GLOBAL_CONTROLLER


def _resolve_mixed_route_step_context(
    *,
    controller: object,
    attn_metadata: object,
    stage: str,
):
    meta_ctx = getattr(attn_metadata, "sparse_step_context", None)
    live_ctx = getattr(controller, "step_context", None)
    if meta_ctx is None and live_ctx is None:
        raise RuntimeError(f"{stage} requires sparse step context")
    if bool(getattr(attn_metadata, "sparse_vllm_profile_step", False)):
        if meta_ctx is None:
            raise RuntimeError(f"{stage} profile metadata missing sparse step context")
        return meta_ctx

    step_authority = getattr(controller, "step_authority", None)
    step_bound_meta = getattr(controller, "step_bound_meta", None)

    expected_epoch = None
    expected_handle_id = None
    expected_handle_generation = None
    if (
        step_bound_meta is not None
        and hasattr(step_bound_meta, "step_handle_id")
        and hasattr(step_bound_meta, "step_handle_generation")
    ):
        expected_epoch = int(getattr(step_bound_meta, "epoch", -1))
        expected_handle_id = int(getattr(step_bound_meta, "step_handle_id", -1))
        expected_handle_generation = int(
            getattr(step_bound_meta, "step_handle_generation", -1)
        )
    elif step_authority is not None:
        expected_epoch = int(getattr(step_authority, "epoch", -1))
        expected_handle_id = int(getattr(step_authority, "step_handle_id", -1))
        expected_handle_generation = int(
            getattr(step_authority, "step_handle_generation", -1)
        )

    def _matches_expected(step_ctx: object | None) -> bool:
        if step_ctx is None:
            return False
        if expected_epoch is None:
            return True
        return (
            int(getattr(step_ctx, "epoch", -1)) == expected_epoch
            and int(getattr(step_ctx, "step_handle_id", -1)) == expected_handle_id
            and int(getattr(step_ctx, "step_handle_generation", -1))
            == expected_handle_generation
        )

    if _matches_expected(meta_ctx):
        return meta_ctx
    if _matches_expected(live_ctx):
        return live_ctx
    if meta_ctx is not None and live_ctx is None:
        return meta_ctx
    if live_ctx is not None and meta_ctx is None:
        return live_ctx
    raise RuntimeError(
        f"{stage} step context identity mismatch "
        f"(meta=({int(getattr(meta_ctx, 'epoch', -1))},"
        f" {int(getattr(meta_ctx, 'step_handle_id', -1))},"
        f" {int(getattr(meta_ctx, 'step_handle_generation', -1))}) "
        f"live=({int(getattr(live_ctx, 'epoch', -1))},"
        f" {int(getattr(live_ctx, 'step_handle_id', -1))},"
        f" {int(getattr(live_ctx, 'step_handle_generation', -1))}) "
        f"expected=({expected_epoch}, {expected_handle_id}, {expected_handle_generation}))"
    )


def _step_context_identity_token(step_ctx: object) -> int:
    token = int(getattr(step_ctx, "step_identity_token", 0))
    if token > 0:
        return token
    epoch = int(getattr(step_ctx, "epoch", -1))
    if epoch < 0:
        return -1
    return (
        epoch * 1_000_000_000
        + int(getattr(step_ctx, "step_handle_id", -1)) * 1_000_000
        + int(getattr(step_ctx, "step_handle_generation", -1))
    )


def _ensure_layer_state_snapshot_alignment(
    *,
    state: object,
    step_ctx: object,
    step_envelope: object,
) -> None:
    req_ids_obj = getattr(step_ctx, "req_ids", None)
    if req_ids_obj is None:
        raise RuntimeError("mixed route requires step context req_ids")
    req_ids = req_ids_obj if isinstance(req_ids_obj, tuple) else tuple(req_ids_obj)
    epoch = int(getattr(step_envelope, "epoch", -1))
    if (
        epoch >= 0
        and int(getattr(state, "_align_epoch_seen", -1)) == epoch
        and getattr(state, "last_active_request_ids", None) == req_ids
    ):
        return
    state.align_slots_from_snapshot(
        request_ids=req_ids,
        slot_by_row=getattr(step_envelope, "slot_by_row"),
        epoch=epoch,
    )


def _resolve_selected_no_capture_row_plan(
    *,
    controller: object,
    step_ctx: object,
    step_authority: object,
    batch_size: int,
    device: torch.device,
):
    from patches.fa3_native.row_plan import build_mixed_page_row_plan

    # The selected/no-capture prologue only needs CPU route facts
    # (has_capture/has_selected_consume). Capture-side GPU carriers are built
    # by prepare_capture_forward_side_outputs from CPU owner-plan truth when
    # actually needed. Avoid tiny CUDA tensor construction here; on short
    # decode it serializes behind prior stream work and dominates latency.
    plan_device = torch.device("cpu")
    cache_key = (
        _step_context_identity_token(step_ctx),
        int(batch_size),
        str(plan_device),
        id(step_authority),
    )
    cached_key = getattr(controller, "_fa3_selected_no_capture_row_plan_cache_key", None)
    cached_plan = getattr(controller, "_fa3_selected_no_capture_row_plan_cache", None)
    if cached_plan is not None and cached_key == cache_key:
        return cached_plan

    row_plan = build_mixed_page_row_plan(
        step_authority,
        batch_size=batch_size,
        device=plan_device,
    )
    setattr(controller, "_fa3_selected_no_capture_row_plan_cache_key", cache_key)
    setattr(controller, "_fa3_selected_no_capture_row_plan_cache", row_plan)
    return row_plan


def _resolve_selected_no_capture_bridge(
    *,
    controller: object,
):
    cached_bridge = getattr(controller, "_fa3_selected_no_capture_bridge", None)
    if cached_bridge is not None:
        return cached_bridge

    from patches.fa3_native.install import load_vendored_flash_attn_bridge

    bridge = load_vendored_flash_attn_bridge()
    setattr(controller, "_fa3_selected_no_capture_bridge", bridge)
    return bridge


def _expand_layer_descale(
    layer: object,
    attr_name: str,
    descale_shape: Tuple[int, int],
) -> torch.Tensor | None:
    scale = getattr(layer, attr_name, None)
    if scale is None:
        return None
    return scale.expand(descale_shape)


_MIXED_PAGE_SM_COUNT_BY_DEVICE: Dict[int, int] = {}


def _mixed_page_device_sm_count(device: torch.device) -> int:
    index = device.index
    if index is None:
        index = int(torch.cuda.current_device())
    count = _MIXED_PAGE_SM_COUNT_BY_DEVICE.get(index)
    if count is None:
        count = int(torch.cuda.get_device_properties(index).multi_processor_count)
        if count <= 0:
            raise RuntimeError(
                "mixed-page split gate requires a positive multi_processor_count"
            )
        _MIXED_PAGE_SM_COUNT_BY_DEVICE[index] = count
    return count


# [2026-07-11 SPLIT-ROOT Phase1] Host-side scan bound for the split argmin.
# Mirrors the max_splits=128 cap flash_api.cpp get_num_splits feeds
# num_splits_heuristic; the host argmin's solution is passed as the launch's
# explicit num_splits (dynamic cap + sizing), so this also bounds the accum
# buffer sizing. If the host argmin ever hits this bound the fa3 route trace
# records clamped_at_scan_cap=True.
_MIXED_PAGE_SPLIT_ARGMIN_SCAN_CAP = 128

# [2026-07-11 SPLIT-ROOT Phase1] n-block unit for the host-side upper-bound
# nb. 64 is the smallest d=128 decode kBlockN across the FA3 tile tables (the
# SM80 non-split arm), NOT a mirrored tile constant: any unit <= the true
# kBlockN yields nb_upper >= nb_true, which biases the host solution (= the
# launch's dynamic cap) upward, so the device prepare kernel -- which
# recomputes the argmin per replay from the template-exact kBlockN and live
# resolved seqused -- is maximally unlikely to be clamped by the cap. The unit
# only shapes the conservative capture-time bound, never the executed split
# count. tests/test_split_makespan_argmin_parity.py pins the cap-dominance
# property over the tile family.
_MIXED_PAGE_SPLIT_ARGMIN_NB_UNIT = 64


def _mixed_page_split_makespan_argmin(
    total_ctas_single_split: int,
    num_sm: int,
    num_n_blocks_single: int,
    num_n_blocks_split: int,
    num_splits_static: int,
) -> int:
    """[2026-07-11 SPLIT-ROOT Phase1] Discrete makespan argmin.

    T(1) = ceil(G/P) * (nb_single + 1);
    T(s>=2) = ceil(G*s/P) * (ceil(nb_split/s) + 1).
    The "+1" is a per-wave fixed cost of one virtual block (launch ramp +
    tail per wave; dimensionless structural constant, not a machine
    constant) -- without it the model over-favoured deep-wave splits on
    mid-G tiers (measured: bs4xsel8k s=10 ran +8.9% vs legacy s=3).
    Multi-wave domain bounded at waves(s) <= max(waves(1), 3): S0-a falsified
    the single-wave cap (multi-wave splits win on saturated tiers) and
    validated the model order up to 3 waves; deeper waves stay outside the
    solution domain until an S0-b experiment extends it (experiment-domain
    bound). s <= num_splits_static, ties to the smallest s. Literal twin of
    split_makespan_argmin() in the vendored hopper/flash_prepare_scheduler.cu
    -- same T(s), same domain, same tie-break.
    tests/test_split_makespan_argmin_parity.py extracts both bodies and pins
    them point-identical over a G x P x nb_single x nb_split x s_ub grid; edit
    both together.
    """
    if total_ctas_single_split <= 0:
        return 1
    # Wave-level enumeration, literally the device kernel's loop shape (the
    # naive per-s scan costs 2 software int-divides per s on GPU and measured
    # +2.7us on the prepare kernel; within a wave level w the makespan
    # w*ceil(nb/s) is non-increasing in s, so only each level's largest
    # admissible s matters and the smallest-s tie resolves to
    # max(s_lo, ceil(nb/q))). Point-identical to the naive scan by
    # tests/test_split_makespan_argmin_parity.py's oracle grid.
    max_validated_waves = 3  # S0-a validated wave depth
    waves_one = (total_ctas_single_split + num_sm - 1) // num_sm
    waves_cap = max(waves_one, max_validated_waves)
    best_s = 1
    best_t = waves_one * (num_n_blocks_single + 1)
    s_scan_max = min(num_splits_static, num_n_blocks_split)
    if s_scan_max >= 2:
        w_first = (2 * total_ctas_single_split + num_sm - 1) // num_sm
        s_lo = 2
        w = w_first
        while w <= waves_cap and s_lo <= s_scan_max:
            s_hi = (w * num_sm) // total_ctas_single_split
            if s_hi > s_scan_max:
                s_hi = s_scan_max
            if s_hi < s_lo:
                w += 1
                continue  # empty level after clamping
            q = (num_n_blocks_split + s_hi - 1) // s_hi
            t = w * (q + 1)
            if t < best_t:
                s_star = (num_n_blocks_split + q - 1) // q if q > 0 else s_lo
                if s_star < s_lo:
                    s_star = s_lo
                best_t = t
                best_s = s_star
            s_lo = s_hi + 1
            w += 1
    return best_s


def _mixed_page_split_domain_cap(
    total_ctas_single_split: int,
    num_sm: int,
    num_n_blocks_upper: int,
) -> int:
    """[2026-07-11 SPLIT-ROOT Phase1] Algebraic upper bound of the argmin's
    solution DOMAIN: waves(s) <= max(waves(1), 3) and s <= nb together give
    s <= floor(max(ceil(G/P), 3) * P / G) and s <= nb_upper. This bound is
    TILE-INDEPENDENT (the wave domain does not involve kBlockN, and every
    real tile's nb is <= the 64-token-unit upper bound), so passing it as the
    launch's explicit num_splits (= dynamic cap + sizing) provably admits the
    device prepare kernel's per-replay argmin under any tile calibre -- the
    cap-dominance the same-formula design needs. A host-side ARGMIN solution
    cannot serve as the cap: multi-wave tie structures make the solution
    non-monotonic in nb (a coarser nb can solve a SMALLER s than a real
    tile's), which would clamp the device solution.
    """
    max_validated_waves = 3  # S0-a validated wave depth (same as the argmin)
    waves_one = (total_ctas_single_split + num_sm - 1) // num_sm
    waves_cap = max(waves_one, max_validated_waves)
    domain_cap = (waves_cap * num_sm) // total_ctas_single_split
    return max(1, min(_MIXED_PAGE_SPLIT_ARGMIN_SCAN_CAP, num_n_blocks_upper, domain_cap))


# [2026-07-12 K6-SCHED-MD-SINGLE-SOURCE] Per-device arch-major memo, sibling of
# _MIXED_PAGE_SM_COUNT_BY_DEVICE (compute capability is a hardware constant per
# device index; no env knob).
_MIXED_PAGE_ARCH_MAJOR_BY_DEVICE: Dict[int, int] = {}


def _mixed_page_device_arch_major(device: torch.device) -> int:
    index = device.index
    if index is None:
        index = int(torch.cuda.current_device())
    major = _MIXED_PAGE_ARCH_MAJOR_BY_DEVICE.get(index)
    if major is None:
        major = int(torch.cuda.get_device_properties(index).major)
        if major <= 0:
            raise RuntimeError(
                "mixed-page scheduler-metadata gate requires a positive device arch major"
            )
        _MIXED_PAGE_ARCH_MAJOR_BY_DEVICE[index] = major
    return major


# [2026-07-12 K6-SCHED-MD-SINGLE-SOURCE] Key under which the shared
# scheduler-metadata entry lives inside the per-dummy-run context dict
# (controller._vllm_dummy_run_context). The dict is created fresh for every
# _dummy_run call and cleared in its finally block, so entry lifetime ==
# one profile/warmup/capture run BY CONSTRUCTION -- cross-run reuse is
# structurally impossible (the "done_for_identity_token" defensive pattern,
# realized as host-object lifetime instead of a new epoch counter).
_MIXED_PAGE_PROFILE_SHARED_SCHED_MD_KEY = "k6_profile_shared_scheduler_metadata"


def _mixed_page_profile_shared_scheduler_metadata(
    *,
    bridge: object,
    device: torch.device,
    batch_size: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    num_heads: int,
    num_heads_k: int,
    headdim: int,
    headdim_v: int,
    qkv_dtype: torch.dtype,
    scheduler_seqused_k: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    page_size: int,
    causal: bool,
    window_size: Tuple[int, int],
    has_softcap: bool,
    num_splits_cap: int,
    page_resolver_kind: int,
) -> torch.Tensor | None:
    """[2026-07-12 K6-SCHED-MD-SINGLE-SOURCE] One prepare launch per profile/
    capture run instead of one per layer (36x -> 1x on the speed-tier form).

    Mechanism: the first attention layer of a dummy run calls the vendored
    get_scheduler_metadata (flash_api.cpp:838-999) -- the structural twin of
    the main mha_fwd path's metadata block (:1392-1458): same
    scheduler_needs_semaphore / use_dynamic_split / vector-count /
    metadata_size arithmetic, and it launches prepare_varlen_num_blocks once
    on the CURRENT stream (captured into the CUDA graph when capturing). The
    remaining 35 layers pass the returned tensor as scheduler_metadata, which
    flips params.skip_scheduler_metadata_computation (flash_api.cpp:1420) so
    the launch template's prepare launch (flash_fwd_launch_template.h:409
    guard: `!skip_scheduler_metadata_computation`) is skipped. At replay the
    single in-graph prepare still re-solves the makespan argmin against the
    canonical graph-live resolved-length carrier every step (<= the domain
    cap). The producer and RRP consumer are required to read the same pointer;
    this is both a launch-count diet and a single-source scheduling contract,
    not a semantics freeze.

    Layer-order safety (audited, R2 material-B): the prepare kernel zeroes the
    tile-count semaphore (flash_prepare_scheduler.cu:224) and every split>1
    layer's combine kernel self-resets it (flash_fwd_combine_kernel.h:427-430,
    args wire params.tile_count_semaphore as semaphore_to_reset); layers run
    serially on one stream, so each layer observes semaphore==0 exactly as if
    it had run its own prepare. The three prepare-produced batch vectors are
    pure functions of (resolved visible length, geometry) shared by all 36
    layers (layer-invariance), and are read-only to the attention/combine
    kernels.

    Shape and policy safety are fail-fast by construction: the launch passes
    the SAME num_splits (the split-domain cap), geometry, and resolver kind to
    both this metadata producer and the mha_fwd consumer. C++
    CHECK_SHAPE(scheduler_metadata, metadata_size) at flash_api.cpp:1424
    rejects any size mismatch (the
    2026-07-01 rejection was two call sites hand-computing DIFFERENT calibres;
    a single producer consumed under the same calibre is immune).

    Scope (Phase1, design verdict): SM8x only -- the caller gates on device
    arch major < 9. On SM90 a splits==1 launch with a passed metadata walks
    the per-call semaphore memset arm (flash_api.cpp:1662-1664) and the
    prepare kernel runs with PDL there; both are unmeasured on this route, so
    SM90 keeps the legacy per-layer prepare (returns None -> unchanged path).
    Returns None (legacy path, one prepare per layer) when no per-dummy-run
    host dict exists; profile-route calls outside a dummy run have no safe
    step-identity carrier, and correctness there is the legacy behaviour.
    """
    controller = _get_global_controller()
    dummy_context = (
        getattr(controller, "_vllm_dummy_run_context", None)
        if controller is not None
        else None
    )
    if not isinstance(dummy_context, dict):
        return None
    # prepare reads these buffers by POINTER inside the captured kernel. The
    # scheduler carrier is deliberately batch-row only: Phase1 metadata owns
    # one dynamic split decision per batch, so silently accepting a per-head
    # carrier would make the producer and consumer disagree about its layout.
    if not isinstance(scheduler_seqused_k, torch.Tensor):
        raise RuntimeError(
            "K6 shared scheduler_metadata requires tensor scheduler_seqused_k"
        )
    if scheduler_seqused_k.dtype != torch.int32:
        raise RuntimeError(
            "K6 shared scheduler_metadata scheduler_seqused_k must have dtype "
            "torch.int32"
        )
    if scheduler_seqused_k.device != device:
        raise RuntimeError(
            "K6 shared scheduler_metadata scheduler_seqused_k device must "
            "match launch device"
        )
    if scheduler_seqused_k.dim() != 1 or int(
        scheduler_seqused_k.numel()
    ) != int(batch_size):
        raise RuntimeError(
            "K6 shared scheduler_metadata scheduler_seqused_k must provide "
            "one int32 entry per batch row"
        )
    if not scheduler_seqused_k.is_contiguous():
        raise RuntimeError(
            "K6 shared scheduler_metadata requires contiguous scheduler_seqused_k"
        )
    if cu_seqlens_q.dtype != torch.int32:
        raise RuntimeError(
            "K6 shared scheduler_metadata cu_seqlens_q must have dtype torch.int32"
        )
    if cu_seqlens_q.device != device:
        raise RuntimeError(
            "K6 shared scheduler_metadata cu_seqlens_q device must match "
            "launch device"
        )
    if cu_seqlens_q.dim() != 1 or int(cu_seqlens_q.numel()) != int(
        batch_size
    ) + 1:
        raise RuntimeError(
            "K6 shared scheduler_metadata cu_seqlens_q must provide batch_size + 1 entries"
        )
    if not cu_seqlens_q.is_contiguous():
        raise RuntimeError(
            "K6 shared scheduler_metadata requires contiguous cu_seqlens_q"
        )
    page_resolver_kind = int(page_resolver_kind)
    if page_resolver_kind not in (0, 4):
        raise RuntimeError(
            "K6 shared scheduler_metadata page_resolver_kind must be "
            f"0 (Native) or 4 (ResolvedRowPtr); got {page_resolver_kind}"
        )
    # Cache identity belongs to the graph-live buffers, not to the ambient
    # CUDA thread state.  Querying current_device() here is both redundant
    # after the exact device checks above and unsafe for CPU contract/fake
    # bridges (and for multi-device workers whose current device can drift).
    # A real CUDA tensor always carries its concrete ordinal; index-less
    # devices such as CPU use -1 while device_type keeps the key unambiguous.
    carrier_device = scheduler_seqused_k.device
    device_index = carrier_device.index
    device_identity = (
        str(carrier_device.type),
        -1 if device_index is None else int(device_index),
    )
    stream_is_capturing = (
        bool(torch.cuda.is_current_stream_capturing())
        if carrier_device.type == "cuda"
        else False
    )
    # The capturing flag separates the warmup (eager) and capture phases that
    # share one dummy-run dict: a warmup-phase tensor lives in the ordinary
    # allocator pool and must NEVER be baked into a graph, and vice versa.
    key = (
        stream_is_capturing,
        device_identity,
        int(batch_size),
        int(max_seqlen_q),
        int(max_seqlen_k),
        int(num_heads),
        int(num_heads_k),
        int(headdim),
        int(headdim_v),
        str(qkv_dtype),
        int(scheduler_seqused_k.data_ptr()),
        int(cu_seqlens_q.data_ptr()),
        int(page_size),
        bool(causal),
        (int(window_size[0]), int(window_size[1])),
        bool(has_softcap),
        int(num_splits_cap),
        page_resolver_kind,
    )
    entry = dummy_context.get(_MIXED_PAGE_PROFILE_SHARED_SCHED_MD_KEY)
    if (
        isinstance(entry, tuple)
        and len(entry) == 2
        and entry[0] == key
        and isinstance(entry[1], torch.Tensor)
    ):
        return entry[1]
    get_scheduler_metadata = getattr(bridge, "get_scheduler_metadata", None)
    if not callable(get_scheduler_metadata):
        raise RuntimeError(
            "K6 shared scheduler_metadata requires bridge.get_scheduler_metadata"
        )
    metadata = get_scheduler_metadata(
        batch_size,
        max_seqlen_q,
        max_seqlen_k,
        num_heads,
        num_heads_k,
        headdim,
        scheduler_seqused_k,
        qkv_dtype=qkv_dtype,
        headdim_v=headdim_v,
        cu_seqlens_q=cu_seqlens_q,
        page_size=page_size,
        causal=causal,
        window_size=(int(window_size[0]), int(window_size[1])),
        has_softcap=bool(has_softcap),
        num_splits=int(num_splits_cap),
        pack_gqa=None,
        sm_margin=0,
        prefill_active_worklist=False,
        page_resolver_kind=page_resolver_kind,
    )
    if not isinstance(metadata, torch.Tensor):
        raise RuntimeError(
            "K6 shared scheduler_metadata: vendored get_scheduler_metadata "
            "did not return a tensor"
        )
    dummy_context[_MIXED_PAGE_PROFILE_SHARED_SCHED_MD_KEY] = (key, metadata)
    return metadata


def _require_shared_graph_live_resolved_lengths(
    *,
    resolver_seqused_k: object,
    carriers: object,
) -> torch.Tensor:
    """Return the single length carrier shared by scheduler and RRP forward.

    Both captured kernels retain a raw device pointer. Shape equality alone is
    insufficient: two equal-valued tensors can diverge on the next replay.
    Refuse any non-aliasing binding instead of scheduling resolved work from a
    stale/full-capacity shadow buffer.
    """
    visible = getattr(carriers, "resolver_visible_seqused_k_by_head_i32", None)
    if not isinstance(resolver_seqused_k, torch.Tensor):
        raise RuntimeError(
            "vLLM profile ResolvedRowPtr mixed forward requires graph-live "
            "resolver_seqused_k"
        )
    if not isinstance(visible, torch.Tensor):
        raise RuntimeError(
            "vLLM profile ResolvedRowPtr mixed forward requires graph-live "
            "visible lengths"
        )
    if (
        tuple(resolver_seqused_k.shape) != tuple(visible.shape)
        or int(resolver_seqused_k.data_ptr()) != int(visible.data_ptr())
    ):
        raise RuntimeError(
            "vLLM profile ResolvedRowPtr scheduler producer and attention consumer "
            "must share the same graph-live visible-length carrier"
        )
    return visible


def _run_profile_resolved_row_ptr_mixed_forward(
    *,
    self,
    layer,
    query,
    key,
    value,
    kv_cache,
    attn_metadata,
    output=None,
    output_scale=None,
    output_block_scale=None,
):
    del key, value
    if not bool(getattr(attn_metadata, "sparse_vllm_profile_step", False)):
        raise RuntimeError(
            "vLLM profile ResolvedRowPtr mixed forward requires profile metadata"
        )
    if output is None:
        raise RuntimeError(
            "vLLM profile ResolvedRowPtr mixed forward requires output tensor"
        )
    if output_scale is not None or output_block_scale is not None:
        raise NotImplementedError(
            "fused output quantization is not supported for profile mixed forward"
        )
    if int(getattr(self, "dcp_world_size", 1) or 1) != 1:
        raise RuntimeError(
            "vLLM profile ResolvedRowPtr mixed forward does not support DCP"
        )

    resolver_kwargs = _mixed_page_resolver_kwargs_from_attn_metadata(attn_metadata)
    descriptor = resolver_kwargs.get("resolver_descriptor")
    carriers = resolver_kwargs.get("resolver_carriers")
    if descriptor is None or carriers is None:
        raise RuntimeError(
            "vLLM profile mixed forward requires resolver metadata"
        )
    from patches.fa3_native.mixed_page_graph_descriptor import (
        PageResolverKind,
        PageResolverSubkind,
    )

    resolver_kind = int(getattr(descriptor, "resolver_kind", 0))
    if resolver_kind != int(PageResolverKind.RESOLVED_ROW_PTR):
        raise RuntimeError(
            "vLLM profile mixed forward requires ResolvedRowPtr resolver"
        )

    key_cache, value_cache = kv_cache.unbind(0)
    kv_cache_dtype = str(getattr(self, "kv_cache_dtype", "") or "")
    if kv_cache_dtype.startswith("fp8"):
        from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend

        fp8_dtype = FlashAttentionBackend.get_fp8_dtype_for_flashattn(
            kv_cache_dtype
        )
        key_cache = key_cache.view(fp8_dtype)
        value_cache = value_cache.view(fp8_dtype)

    cu_seqlens_q = getattr(attn_metadata, "query_start_loc")
    if not isinstance(cu_seqlens_q, torch.Tensor) or cu_seqlens_q.dim() != 1:
        raise RuntimeError(
            "vLLM profile ResolvedRowPtr mixed forward requires query_start_loc"
        )
    batch_size = int(cu_seqlens_q.shape[0]) - 1
    if batch_size <= 0:
        raise RuntimeError(
            "vLLM profile ResolvedRowPtr mixed forward requires non-empty batch"
        )

    resolver_seqused_k = resolver_kwargs.get("resolver_seqused_k")
    visible = _require_shared_graph_live_resolved_lengths(
        resolver_seqused_k=resolver_seqused_k,
        carriers=carriers,
    )
    seqused_k = getattr(attn_metadata, "seq_lens", None)
    if not isinstance(seqused_k, torch.Tensor):
        raise RuntimeError(
            "vLLM profile ResolvedRowPtr mixed forward requires tensor seqused_k"
        )
    max_seqlen_q = max(1, int(getattr(attn_metadata, "max_query_len", 1) or 1))
    max_seqlen_k = max(
        1,
        int(
            getattr(
                attn_metadata,
                "mixed_page_profile_max_seqlen_k",
                getattr(attn_metadata, "max_seq_len", max_seqlen_q),
            )
            or max_seqlen_q
        ),
    )
    block_table = getattr(attn_metadata, "block_table", None)
    if not isinstance(block_table, torch.Tensor):
        raise RuntimeError(
            "vLLM profile ResolvedRowPtr mixed forward requires tensor block_table"
        )

    alibi_slopes = getattr(self, "alibi_slopes", None)
    if alibi_slopes is not None:
        raise RuntimeError(
            "vLLM profile ResolvedRowPtr mixed forward does not support ALiBi"
        )

    num_actual_tokens = int(
        getattr(attn_metadata, "num_actual_tokens", int(query.shape[0]))
        or int(query.shape[0])
    )
    q_arg = query[:num_actual_tokens]
    out_arg = output[:num_actual_tokens]
    num_kv_heads = int(getattr(self, "num_kv_heads", int(key_cache.shape[2])) or 0)
    if num_kv_heads <= 0:
        raise RuntimeError(
            "vLLM profile ResolvedRowPtr mixed forward requires num_kv_heads"
        )
    descale_shape = (batch_size, num_kv_heads)
    sliding_window = getattr(self, "sliding_window", None)
    window_size = list(sliding_window) if sliding_window is not None else None

    # [2026-07-11 SPLIT-ROOT Phase1] Static instance selection by the same
    # discrete makespan argmin the vendored prepare kernel now runs for mixed
    # rows: T(1) = ceil(G/P)*nb_single, T(s>=2) = ceil(G*s/P)*ceil(nb_split/s),
    # full wave domain (S0-a falsified the single-wave cap: bs8xsel4k s=3 runs
    # -15.6% across 2 waves, bs8xsel12k s=5 -27.3% across 3), ties to the
    # smallest s (design verdict SFI_KIND4_NEXT_OPT_DIRECTIONS_2026-07-11 par.1
    # + S0-a addendum; supersedes the [2026-07-11 K1] half-SM gate -- the
    # machine constant is gone, the arithmetic stays).
    # Cap semantics (the launched value is the DOMAIN upper bound, not a host
    # argmin solution -- two designs died to real counterexamples: a host
    # argmin solution as the cap is non-monotonic in nb under multi-wave ties
    # and can clamp the device solution; a host argmin==1 instance gate under
    # the 64-token calibre mis-pins forms whose device tile still profits
    # from splits, e.g. H100 bs16 x h_k8 mid-sel):
    # - cap > 1  -> pass it as the launch's EXPLICIT num_splits: under
    #   use_dynamic_split (b <= 992 && splits > 1) it is a dynamic CAP +
    #   sizing bound, not a pinned count -- the in-graph prepare kernel
    #   solves the same argmin per replay against the live resolved seqused
    #   and the template-exact kBlockN, PROVABLY <= this cap (the bound is
    #   tile-independent, see _mixed_page_split_domain_cap; flash_api.cpp
    #   guard now admits explicit >1 for ResolvedRowPtr; S0-a measured the
    #   cap semantics on the native form). Rows the device argmin resolves to
    #   1 run one split on the split instance (numerically exact; the
    #   epsilon vs the non-split instance only exists on degenerate short-sel
    #   forms).
    # - cap == 1 -> only when the solution domain itself collapses
    #   (max_seqlen_k <= one 64-token block, or G > 1.5*P super-saturation):
    #   the non-split kernel instance, prepare early-outs, bit-identical to
    #   the retired-K1 pin1 arm by construction.
    # G = batch_size*num_kv_heads: this capture-time launch is the decode form
    # (one m-block per row under PackGQA); for profile-step calls with
    # max_seqlen_q > 1 this undercounts G, which only lowers the cap the
    # profile step runs under (never affects capture anchors).
    # nb upper bound: max_seqlen_k (native >= resolved) in units of the
    # smallest decode kBlockN (see _MIXED_PAGE_SPLIT_ARGMIN_NB_UNIT), so every
    # real tile's n-block count is <= it.
    # This function runs only during FULL-cudagraph capture, so the decision
    # freezes per (batch bucket, h_k, device); accum/semaphore buffers allocate
    # inside capture into the capture pool with sizes fixed by the frozen cap.
    # Split-combine reassociates fp32 -> low bits move wherever the executed
    # split count changes (allclose level; e2e speed-tier hash anchors are
    # expected to move and must be re-pinned with a pin1 control leg).
    mixed_page_argmin_nb_ub = (
        int(max_seqlen_k) + _MIXED_PAGE_SPLIT_ARGMIN_NB_UNIT - 1
    ) // _MIXED_PAGE_SPLIT_ARGMIN_NB_UNIT
    _mixed_page_g = batch_size * num_kv_heads
    _mixed_page_p = _mixed_page_device_sm_count(q_arg.device)
    # host argmin: observability only (route trace) -- what the model would
    # solve on the capture-time upper bounds; the executed split count is the
    # device prepare kernel's per-replay solution under the cap below.
    mixed_page_host_split_argmin = _mixed_page_split_makespan_argmin(
        _mixed_page_g,
        _mixed_page_p,
        mixed_page_argmin_nb_ub,
        mixed_page_argmin_nb_ub,
        _MIXED_PAGE_SPLIT_ARGMIN_SCAN_CAP,
    )
    mixed_page_split_cap = _mixed_page_split_domain_cap(
        _mixed_page_g, _mixed_page_p, mixed_page_argmin_nb_ub
    )

    from patches.fa3_native.install import load_vendored_flash_attn_bridge

    bridge = load_vendored_flash_attn_bridge()
    mixed_page_causal = bool(getattr(attn_metadata, "causal", True))
    mixed_page_softcap = float(getattr(self, "logits_soft_cap", 0.0) or 0.0)
    mixed_page_window_tuple = (
        (int(window_size[0]), int(window_size[1]))
        if window_size is not None
        else (-1, -1)
    )
    # [2026-07-12 K6-SCHED-MD-SINGLE-SOURCE] One prepare per run instead of one
    # per layer. The first layer of a dummy run produces the metadata via the
    # vendored get_scheduler_metadata with the explicit RRP resolver policy
    # (single prepare launch, captured into the graph when capturing); the
    # other 35 layers pass the same tensor so the C++
    # side skips its per-call prepare launch (flash_api.cpp:1420 +
    # flash_fwd_launch_template.h:409). The producer reads resolver_seqused_k,
    # the exact graph-live pointer passed to the RRP consumer as visible below;
    # full-capacity attn_metadata.seq_lens remains only the native forward ABI
    # input. Every other argument below is the SAME local the launch's
    # common_kwargs uses -- one calibre, one producer, and the C++
    # CHECK_SHAPE(scheduler_metadata, metadata_size) (flash_api.cpp:1424) hard-
    # rejects any residual size drift (constructive fix for the 2026-07-01
    # two-hand-computed-calibres rejection). SM8x-only Phase1 gate: on SM90 the
    # splits==1 + passed-metadata combination walks a per-call semaphore memset
    # arm (flash_api.cpp:1662-1664, unmeasured there) -> keep legacy per-layer
    # prepare (shared metadata stays None, byte-identical old path).
    mixed_page_shared_scheduler_metadata = None
    if _mixed_page_device_arch_major(q_arg.device) < 9:
        mixed_page_shared_scheduler_metadata = (
            _mixed_page_profile_shared_scheduler_metadata(
                bridge=bridge,
                device=q_arg.device,
                batch_size=batch_size,
                max_seqlen_q=int(max_seqlen_q),
                max_seqlen_k=int(max_seqlen_k),
                num_heads=int(q_arg.shape[1]),
                num_heads_k=int(num_kv_heads),
                headdim=int(q_arg.shape[2]),
                headdim_v=int(value_cache.shape[-1]),
                qkv_dtype=q_arg.dtype,
                scheduler_seqused_k=resolver_seqused_k,
                cu_seqlens_q=cu_seqlens_q,
                page_size=int(key_cache.shape[1]),
                causal=mixed_page_causal,
                window_size=mixed_page_window_tuple,
                has_softcap=mixed_page_softcap > 0.0,
                num_splits_cap=int(mixed_page_split_cap),
                page_resolver_kind=int(PageResolverKind.RESOLVED_ROW_PTR),
            )
        )
    common_kwargs = {
        "q": q_arg,
        "k": key_cache,
        "v": value_cache,
        "out": out_arg,
        "cu_seqlens_q": cu_seqlens_q,
        "max_seqlen_q": max_seqlen_q,
        "seqused_k": seqused_k,
        "max_seqlen_k": max_seqlen_k,
        "softmax_scale": getattr(self, "scale", None),
        "causal": mixed_page_causal,
        "window_size": window_size,
        "block_table": block_table,
        "softcap": mixed_page_softcap,
        # [2026-07-01, kept under 2026-07-11 SPLIT-ROOT; scoped by 2026-07-12
        # K6] None used to be the only correct value here: vLLM's own
        # precomputed scheduler_metadata is sized for max_num_splits=32 and the
        # C++ CHECK_SHAPE (flash_api.cpp:1424) rejects any foreign calibre.
        # K6 does NOT resurrect that path -- the shared tensor above is
        # produced by the vendored twin under the exact launch calibre (same
        # num_splits cap, same geometry, same live seqused buffer), so the
        # in-graph prepare (now launched once per run by the metadata
        # producer instead of once per layer) keeps re-solving per-batch
        # dynamic splits <= cap at every replay. None (SM90 / no dummy-run
        # host) preserves the legacy per-layer prepare byte-for-byte.
        "scheduler_metadata": mixed_page_shared_scheduler_metadata,
        "q_descale": _expand_layer_descale(layer, "_q_scale", descale_shape),
        "k_descale": _expand_layer_descale(layer, "_k_scale", descale_shape),
        "v_descale": _expand_layer_descale(layer, "_v_scale", descale_shape),
        # [2026-07-11 SPLIT-ROOT Phase1] The split-domain cap, passed as the
        # launch's explicit num_splits = dynamic cap + sizing bound (see the
        # argmin comment above; ==1 pins the non-split instance, >1 selects
        # the split instance with in-graph prepare refinement <= cap). The
        # 2026-07-01 "contractually single-split / SingleTileVarlenScheduler"
        # note that used to pin 1 here stays superseded: varlen+split walks
        # VarlenDynamicPersistentTileScheduler on SM80 and SM90, and kind4
        # split is measured green (verdict §1b + Step0 sanitizer + S0-a).
        # Native split path is unaffected (different call site).
        "num_splits": int(mixed_page_split_cap),
        "s_aux": getattr(self, "sinks", None),
        "cp_world_size": int(getattr(attn_metadata, "cp_world_size", 1) or 1),
        "cp_rank": int(getattr(attn_metadata, "cp_rank", 0) or 0),
        "cp_tot_seqused_k": getattr(attn_metadata, "cp_tot_seqused_k", None),
        "q_v": None,
        "return_softmax_lse": False,
        "graph_replay_carriers": bool(
            resolver_kwargs.get("graph_replay_carriers", False)
        ),
    }
    row_ptr = getattr(carriers, "resolved_page_table_row_ptr_u64", None)
    affine_i32 = getattr(carriers, "resolved_page_table_affine_i32", None)
    affine_base = getattr(carriers, "resolved_page_table_affine_base", None)
    affine_stride = getattr(carriers, "resolved_page_table_affine_stride", None)
    affine_direct = bool(getattr(carriers, "resolved_page_table_affine_direct", False))
    affine_cols = int(getattr(carriers, "resolved_page_table_affine_cols", 0))
    has_affine_const = affine_base is not None or affine_stride is not None
    has_affine = affine_i32 is not None or has_affine_const
    if isinstance(affine_i32, torch.Tensor):
        resolver_subkind_i = int(PageResolverSubkind.AFFINE_TENSOR)
    elif has_affine_const:
        resolver_subkind_i = int(
            PageResolverSubkind.AFFINE_CONST_DIRECT
            if affine_direct
            else PageResolverSubkind.AFFINE_CONST
        )
    else:
        resolver_subkind_i = int(PageResolverSubkind.ROWPTR)
    if has_affine_const and (affine_base is None or affine_stride is None):
        raise RuntimeError("vLLM profile ResolvedRowPtr affine mixed forward requires base and stride together")
    if has_affine_const and row_ptr is not None:
        raise RuntimeError("vLLM profile ResolvedRowPtr affine mixed forward must not also set row-pointer carrier")

    if _fa3_route_trace_enabled():
        try:
            from patches.fa3_native.install import append_fa3_route_trace

            resolver_shape = None
            if isinstance(resolver_seqused_k, torch.Tensor):
                resolver_shape = tuple(int(v) for v in resolver_seqused_k.shape)
            append_fa3_route_trace(
                {
                    "event": "profile_resolved_row_ptr_mixed_forward",
                    "num_actual_tokens": int(num_actual_tokens),
                    "query_shape": tuple(int(v) for v in query.shape),
                    "q_arg_shape": tuple(int(v) for v in q_arg.shape),
                    "out_arg_shape": tuple(int(v) for v in out_arg.shape),
                    "block_table_shape": tuple(int(v) for v in block_table.shape),
                    "seq_lens_shape": tuple(int(v) for v in seqused_k.shape),
                    "resolver_seqused_shape": resolver_shape,
                    "max_seqlen_q": int(max_seqlen_q),
                    "max_seqlen_k": int(max_seqlen_k),
                    "num_splits": int(common_kwargs["num_splits"]),
                    # [2026-07-11 SPLIT-ROOT Phase1] host argmin observability:
                    # the capture-time solution, the domain cap actually
                    # launched (num_splits above), inputs, and whether the 128
                    # scan cap clamped the solution (device-side clamp probing
                    # is the FLASH_SPLIT_ARGMIN_CLAMP_DEBUG rebuild of the
                    # prepare TU).
                    "host_split_argmin": int(mixed_page_host_split_argmin),
                    "host_split_cap": int(mixed_page_split_cap),
                    "host_split_argmin_nb_ub": int(mixed_page_argmin_nb_ub),
                    # [2026-07-12 K6] shared scheduler_metadata observability:
                    # non-None => this layer consumes the run-shared tensor
                    # (prepare launched once per run, not per layer).
                    "k6_shared_scheduler_metadata": bool(
                        mixed_page_shared_scheduler_metadata is not None
                    ),
                    "k6_scheduler_metadata_numel": (
                        int(mixed_page_shared_scheduler_metadata.numel())
                        if mixed_page_shared_scheduler_metadata is not None
                        else -1
                    ),
                    "host_split_argmin_clamped_at_scan_cap": bool(
                        mixed_page_host_split_argmin
                        == _MIXED_PAGE_SPLIT_ARGMIN_SCAN_CAP
                    ),
                    "num_kv_heads": int(num_kv_heads),
                    "graph_replay_carriers": bool(
                        common_kwargs["graph_replay_carriers"]
                    ),
                    "has_resolved_affine": bool(has_affine),
                    "has_resolved_affine_direct": affine_direct,
                    "profile_max_seqlen_k_attr": int(
                        getattr(attn_metadata, "mixed_page_profile_max_seqlen_k", -1)
                        or -1
                    ),
                    "attn_metadata_max_seq_len": int(
                        getattr(attn_metadata, "max_seq_len", -1) or -1
                    ),
                }
            )
        except Exception:
            pass
    if row_ptr is None and not has_affine:
        raise RuntimeError(
            "vLLM profile ResolvedRowPtr mixed forward requires row-pointer or affine carrier plus visible lengths"
        )
    _record_mixed_page_actual_route_family(
        _get_global_controller(),
        "resolved_row_ptr",
    )
    bridge.mixed_page_attn_varlen_func(
        **common_kwargs,
        page_resolver_kind=int(PageResolverKind.RESOLVED_ROW_PTR),
        page_resolver_subkind=resolver_subkind_i,
        selected_page_table_i32=None,
        row_consume_mode_i32=getattr(carriers, "row_consume_mode_i32", None) if has_affine else None,
        selected_seqused_k_by_head_i32=None,
        resolved_page_table_row_ptr_u64=row_ptr,
        resolved_page_table_affine_i32=affine_i32,
        resolved_page_table_affine_base=affine_base,
        resolved_page_table_affine_stride=affine_stride,
        resolved_page_table_affine_segment_pages=getattr(carriers, "resolved_page_table_affine_segment_pages", None),
        resolved_page_table_affine_second_base=getattr(carriers, "resolved_page_table_affine_second_base", None),
        resolved_page_table_affine_second_stride=getattr(carriers, "resolved_page_table_affine_second_stride", None),
        resolved_page_table_affine_batch_stride=getattr(carriers, "resolved_page_table_affine_batch_stride", None),
        resolved_page_table_affine_head_stride=getattr(carriers, "resolved_page_table_affine_head_stride", 0),
        resolved_page_table_affine_direct=affine_direct,
        resolved_page_table_affine_cols=affine_cols,
        resolved_seqused_k_by_head_i32=visible,
        compact_base_page_i32=None,
        compact_page_count_i32=None,
        recent_first_logical_page_i32=None,
    )
    return output

def _ensure_selected_no_capture_recent_descriptors(
    *,
    step_meta,
    page_size: int,
    step_authority: object,
) -> None:
    from patches.fa_sparse_runtime.runtime_cache import ensure_step_recent_descriptors

    batch_size = int(step_meta.batch_size)
    layer_effective_refresh_by_row = step_authority.layer_effective_refresh_by_row
    if len(layer_effective_refresh_by_row) != batch_size:
        raise RuntimeError(
            "recent descriptor build requires exact refresh-row coverage: "
            f"rows={len(layer_effective_refresh_by_row)} batch={batch_size}"
        )
    ensure_step_recent_descriptors(
        step_meta=step_meta,
        page_size=page_size,
        layer_effective_refresh_by_row=layer_effective_refresh_by_row,
    )


def _resolve_step_real_seqused_k(
    *,
    step_bound_meta,
    original_seq_lens: object,
    device: torch.device,
    batch_size: int,
    cache_owner: object | None = None,
) -> object:
    canonical = getattr(step_bound_meta, "canonical_real_kv_len_i32_gpu", None)
    if (
        isinstance(canonical, torch.Tensor)
        and canonical.device == device
        and canonical.dtype == torch.int32
        and canonical.dim() == 1
        and int(canonical.numel()) >= int(batch_size)
    ):
        return canonical[:batch_size].reshape(batch_size)
    if isinstance(original_seq_lens, torch.Tensor):
        device_index = device.index
        cache_key = (
            int(getattr(step_bound_meta, "step_identity_token", 0) or 0),
            int(original_seq_lens.data_ptr()),
            int(getattr(original_seq_lens, "_version", 0) or 0),
            str(original_seq_lens.device),
            str(original_seq_lens.dtype),
            int(batch_size),
            device.type,
            -1 if device_index is None else int(device_index),
        )
        owner = cache_owner if cache_owner is not None else step_bound_meta
        cached = getattr(owner, "_sfi_real_seqused_k_cache", None)
        if isinstance(cached, tuple) and len(cached) == 2 and cached[0] == cache_key:
            cached_tensor = cached[1]
            if (
                isinstance(cached_tensor, torch.Tensor)
                and cached_tensor.device == device
                and cached_tensor.dtype == torch.int32
                and cached_tensor.dim() == 1
                and int(cached_tensor.numel()) == int(batch_size)
            ):
                return cached_tensor
        resolved = original_seq_lens.to(device=device, dtype=torch.int32).reshape(batch_size)
        try:
            owner._sfi_real_seqused_k_cache = (cache_key, resolved)
        except Exception:
            pass
        return resolved
    return canonical


# [OVERLAY-GEOM-STEP-CACHE] 单槽步级缓存：几何解析输入全部步内不变（launch_plan
# 由 prologue 每步构造一次、seq_lens/authority 步级快照），旧实现每 layer 重跑
# per-row Python 循环（36 层逐位同结果）。key=身份比较五元组+step_identity_token，
# 任一输入对象换新即 miss——不存在 stale 可能；持有强引用一步（下步覆盖释放）。
_OVERLAY_GEOM_STEP_CACHE: tuple | None = None


def _resolve_selected_overlay_cpu_geometry(
    *,
    launch_plan: object,
    step_bound_meta: object,
    step_authority: object,
    original_seq_lens: object,
    page_size: int,
    batch_size: int,
) -> tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
    global _OVERLAY_GEOM_STEP_CACHE
    token = int(getattr(step_bound_meta, "step_identity_token", 0) or 0)
    cache = _OVERLAY_GEOM_STEP_CACHE
    if (
        cache is not None
        and token > 0
        and cache[0] == token
        and cache[1] is launch_plan
        and cache[2] is step_bound_meta
        and cache[3] is step_authority
        and cache[4] is original_seq_lens
        and cache[5] == page_size
        and cache[6] == batch_size
    ):
        return cache[7]
    from patches.fa_sparse_runtime.compact_mixed_page_route import (
        resolve_compact_mixed_page_overlay_cpu_geometry,
    )

    result = resolve_compact_mixed_page_overlay_cpu_geometry(
        launch_plan=launch_plan,
        step_bound_meta=step_bound_meta,
        step_authority=step_authority,
        real_kv_len_hint=original_seq_lens,
        page_size=page_size,
        batch_size=batch_size,
    )
    if token > 0:
        _OVERLAY_GEOM_STEP_CACHE = (
            token,
            launch_plan,
            step_bound_meta,
            step_authority,
            original_seq_lens,
            page_size,
            batch_size,
            result,
        )
    return result


def _step_uses_full_kv_handoff(
    *,
    controller: object,
    step_authority: object,
    owner_plan: object,
) -> bool:
    """True when not-ready decode rows should stay on mixed-page as full-KV."""

    if not bool(getattr(owner_plan, "has_decode_rows", False)):
        return False
    if bool(getattr(owner_plan, "has_prefill_rows", False)):
        return False

    batch_size = int(getattr(step_authority, "batch_size", 0) or 0)
    req_ids = tuple(str(rid) for rid in tuple(getattr(step_authority, "req_ids", ()))[:batch_size])
    if not req_ids:
        return False

    can_bridge = getattr(controller, "_request_can_bridge_bootstrap_decode", None)
    if not callable(can_bridge):
        return False
    return any(bool(can_bridge(rid)) for rid in req_ids)


def _ensure_step_prologue(
    *,
    controller: object,
    step_ctx: object,
    step_authority: object,
    step_bound_meta,
    batch_size: int,
    kv_cache: torch.Tensor,
    device: torch.device,
    canonical_state: object | None = None,
) -> None:
    """Per-step prologue shared by both rail wrappers.

    Hoists per-step owner/rail/row/launch-plan facts onto
    step_bound_meta.prologue_* fields. Idempotent:
    repeated invocations within the same step return immediately via
    step_identity_token comparison.

    Side effect: compact/full-KV mixed-page paths populate
    step_bound_meta.request_recent_*_i32_gpu once through recent descriptors.

    See docs/superpowers/specs/2026-04-24-per-layer-to-per-step-dispatch-hoist-design.md §4.2.
    """
    if step_bound_meta.prologue_done_for_identity_token == step_bound_meta.step_identity_token:
        return
    from patches.fa_sparse_runtime.compact_mixed_page_route import (
        bind_compact_arena_consumer_stream_for_step,
    )

    bind_compact_arena_consumer_stream_for_step(
        step_bound_meta=step_bound_meta,
        device=device,
    )

    from patches.fa_sparse_runtime.mixed_prefill_decode_owner import (
        resolve_mixed_prefill_decode_owner_plan,
    )
    from patches.fa_sparse_runtime.compact_recent_route_authority import (
        CompactRecentRailMode,
        resolve_compact_recent_rail_mode,
    )

    page_size = _infer_paged_kv_page_size(kv_cache)

    step_bound_meta.prologue_owner_plan = resolve_mixed_prefill_decode_owner_plan(step_authority)
    step_bound_meta.prologue_rail_decision = resolve_compact_recent_rail_mode(step_authority)
    step_bound_meta.prologue_full_kv_handoff = (
        step_bound_meta.prologue_rail_decision.mode is CompactRecentRailMode.NO_COMPACT
        and _step_uses_full_kv_handoff(
            controller=controller,
            step_authority=step_authority,
            owner_plan=step_bound_meta.prologue_owner_plan,
        )
    )
    step_bound_meta.prologue_selected_row_plan = _resolve_selected_no_capture_row_plan(
        controller=controller,
        step_ctx=step_ctx,
        step_authority=step_authority,
        batch_size=batch_size,
        device=device,
    )
    if (
        step_bound_meta.prologue_rail_decision.mode is not CompactRecentRailMode.NO_COMPACT
        or bool(step_bound_meta.prologue_full_kv_handoff)
    ):
        _ensure_selected_no_capture_recent_descriptors(
            step_meta=step_bound_meta,
            page_size=page_size,
            step_authority=step_authority,
        )
        current_plan = getattr(step_bound_meta, "compact_recent_launch_plan", None)
        slot_signature = tuple(
            int(v) for v in tuple(getattr(step_authority, "slot_by_row", tuple()))[:batch_size]
        )
        use_compact_signature = tuple(
            1 if bool(v) else 0
            for v in tuple(getattr(step_authority, "use_compact_by_row", tuple()))[:batch_size]
        )
        plan_needs_rebuild = (
            current_plan is None
            or not bool(getattr(current_plan, "valid", False))
            or int(getattr(current_plan, "step_identity_token", 0) or 0)
            != int(getattr(step_bound_meta, "step_identity_token", 0) or 0)
            or tuple(getattr(current_plan, "slot_signature", tuple())) != slot_signature
            or tuple(getattr(current_plan, "use_compact_signature", tuple()))
            != use_compact_signature
        )
        if plan_needs_rebuild:
            from patches.decode_runtime.compact_recent_launch_plan_builder import (
                build_compact_recent_launch_plan as build_step_compact_recent_launch_plan,
            )

            step_bound_meta.compact_recent_launch_plan = build_step_compact_recent_launch_plan(
                controller=controller,
                step_authority=step_authority,
                step_bound_meta=step_bound_meta,
                page_size=page_size,
                device=device,
                canonical_state=canonical_state,
                allow_full_kv_handoff=bool(
                    step_bound_meta.prologue_full_kv_handoff
                ),
            )
    step_bound_meta.prologue_done_for_identity_token = step_bound_meta.step_identity_token


def _build_prefill_capture_last_n_by_row(
    *,
    controller: object,
    step_ctx: object,
    batch_size: int,
) -> Tuple[int, ...]:
    capture_plan_by_req: Dict[str, int] = {}
    if hasattr(controller, "get_step_prefill_plan_by_req"):
        capture_plan_by_req, _ = controller.get_step_prefill_plan_by_req(step_context=step_ctx)
    if not capture_plan_by_req or batch_size <= 0:
        return tuple(0 for _ in range(max(0, int(batch_size))))

    req_ids = step_ctx.req_ids
    q_lens = step_ctx.q_lens
    step_authority = getattr(step_ctx, "step_authority", None)
    if step_authority is None:
        raise RuntimeError("prefill FA3 route requires StepAuthority")
    is_prefill_by_row = step_authority.is_prefill_by_row
    if (
        int(step_authority.batch_size) != int(batch_size)
        or len(req_ids) != int(batch_size)
        or len(q_lens) != int(batch_size)
        or len(is_prefill_by_row) != int(batch_size)
    ):
        raise RuntimeError(
            "prefill FA3 route requires exact row coverage: "
            f"batch={int(batch_size)} authority={int(step_authority.batch_size)} "
            f"req_ids={len(req_ids)} q_lens={len(q_lens)} "
            f"prefill={len(is_prefill_by_row)}"
        )
    last_n_by_row = [0] * int(batch_size)
    for row in range(int(batch_size)):
        if not bool(is_prefill_by_row[row]):
            continue
        req_id = req_ids[row]
        last_n = int(capture_plan_by_req.get(req_id, 0) or 0)
        if last_n <= 0:
            continue
        q_len = int(q_lens[row])
        if q_len > 0:
            last_n = min(int(last_n), int(q_len))
        last_n_by_row[row] = max(0, int(last_n))
    return tuple(last_n_by_row)


def resolve_live_fa3_launch_route(
    *,
    attn_metadata: object,
) -> str:
    from patches.fa3_native.route_adapter import (
        _COMPACT_RECENT_ROUTE,
        _MIXED_PAGE_ROUTE,
        bind_launch_route_hints,
        get_bound_launch_route_hints,
        route_attention_launch,
    )
    from patches.fa3_native.row_plan import build_mixed_page_row_plan
    from patches.sparse_constants import (
        _LOGF_PRODUCER_ATTN,
        _LOGF_PRODUCER_NONE,
        _ROW_MODE_COMPACT,
        _ROW_MODE_DENSE,
    )

    bound_hints = get_bound_launch_route_hints(attn_metadata)
    bound_route = route_attention_launch(
        has_selected_consume=bool(bound_hints.has_selected_consume),
        has_capture=bool(bound_hints.has_capture),
        has_compact_recent=bool(bound_hints.has_compact_recent),
    )

    controller = _get_global_controller()
    if controller is None or not bool(getattr(getattr(controller, "config", None), "enabled", False)):
        return bound_route
    if (
        getattr(attn_metadata, "sparse_step_context", None) is None
        and getattr(controller, "step_context", None) is None
    ):
        raise RuntimeError("native FA3 launch route requires sparse step context")

    step_ctx = _resolve_mixed_route_step_context(
        controller=controller,
        attn_metadata=attn_metadata,
        stage="native FA3 launch route",
    )
    step_authority = getattr(step_ctx, "step_authority", None)
    if step_authority is None:
        step_authority = getattr(controller, "step_authority", None)
    if step_authority is None:
        raise RuntimeError("native FA3 launch route requires step_authority")
    is_profile_step = bool(getattr(attn_metadata, "sparse_vllm_profile_step", False))
    if is_profile_step:
        if int(getattr(step_ctx, "epoch", 0)) != -1:
            raise RuntimeError(
                "vLLM profile FA3 route requires an isolated profile step context"
            )
        if bool(step_authority.has_compact_row) and not (
            _profile_mixed_page_cudagraph_capture_enabled(controller)
        ):
            raise RuntimeError("vLLM profile FA3 route must remain dense")
        valid_profile_row_modes = {int(_ROW_MODE_DENSE), int(_ROW_MODE_COMPACT)}
        profile_row_modes = step_authority.row_mode_by_row
        if any(v not in valid_profile_row_modes for v in profile_row_modes):
            raise RuntimeError(
                "vLLM profile row_mode invalid: expected legacy dense/compact row modes"
            )
        if any(
            int(v) != int(_LOGF_PRODUCER_NONE)
            for v in step_authority.dispatch_logf_producer_by_row
        ):
            raise RuntimeError("vLLM profile FA3 route must not publish log_f")
        if any(int(v) > 0 for v in step_authority.logits_last_n_by_row):
            raise RuntimeError("vLLM profile FA3 route must not request logits capture")
    step_token = _step_context_identity_token(step_ctx)

    batch_size = None
    query_start_loc = getattr(attn_metadata, "query_start_loc", None)
    if isinstance(query_start_loc, torch.Tensor) and query_start_loc.ndim == 1 and query_start_loc.numel() > 0:
        batch_size = int(query_start_loc.shape[0]) - 1

    cached_token = int(getattr(controller, "_fa3_live_route_token", -1))
    cached_batch_size = int(getattr(controller, "_fa3_live_route_batch_size", -1))
    cached_route = getattr(controller, "_fa3_live_route", None)
    cached_selected = getattr(controller, "_fa3_live_route_has_selected_consume", None)
    cached_capture = getattr(controller, "_fa3_live_route_has_capture", None)
    cached_compact_recent = getattr(controller, "_fa3_live_route_has_compact_recent", None)
    cached_is_legacy_compact = (
        cached_compact_recent is True or str(cached_route) == _COMPACT_RECENT_ROUTE
    )
    if (
        cached_route is not None
        and not is_profile_step
        and not cached_is_legacy_compact
        and cached_token == step_token
        and cached_batch_size == int(batch_size if batch_size is not None else -1)
        and isinstance(cached_selected, bool)
        and isinstance(cached_capture, bool)
        and isinstance(cached_compact_recent, bool)
    ):
        bind_launch_route_hints(
            attn_metadata,
            has_selected_consume=bool(cached_selected),
            has_capture=bool(cached_capture),
            has_compact_recent=bool(cached_compact_recent),
        )
        return str(cached_route)

    effective_step_authority = step_authority
    authority_rows = int(step_authority.batch_size)
    total_rows = int(batch_size) if batch_size is not None else authority_rows
    if total_rows != authority_rows:
        raise RuntimeError(
            "native FA3 launch route requires exact StepAuthority batch coverage: "
            f"metadata={total_rows} authority={authority_rows}"
        )
    if (
        len(step_authority.req_ids) != total_rows
        or len(step_authority.q_lens_by_row) != total_rows
        or len(step_authority.row_policy_ready_by_row) != total_rows
        or len(step_authority.row_mode_by_row) != total_rows
    ):
        raise RuntimeError(
            "native FA3 launch route requires exact authority row vectors: "
            f"batch={total_rows} req_ids={len(step_authority.req_ids)} "
            f"q_lens={len(step_authority.q_lens_by_row)} "
            f"row_ready={len(step_authority.row_policy_ready_by_row)} "
            f"row_mode={len(step_authority.row_mode_by_row)}"
        )
    if hasattr(controller, "get_step_prefill_plan_by_req"):
        prefill_capture_last_n_by_row = _build_prefill_capture_last_n_by_row(
            controller=controller,
            step_ctx=step_ctx,
            batch_size=total_rows,
        )
        if any(int(v) > 0 for v in prefill_capture_last_n_by_row):
            producer_src = step_authority.dispatch_logf_producer_by_row
            last_n_src = step_authority.logits_last_n_by_row
            if (
                len(producer_src) != total_rows
                or len(last_n_src) != total_rows
                or len(prefill_capture_last_n_by_row) != total_rows
            ):
                raise RuntimeError(
                    "prefill FA3 route overlay requires exact row coverage: "
                    f"batch={total_rows} producer={len(producer_src)} "
                    f"last_n={len(last_n_src)} "
                    f"planned={len(prefill_capture_last_n_by_row)}"
                )
            effective_producer = []
            effective_last_n = []
            effective_has_capture = False
            for row in range(total_rows):
                existing_producer = int(producer_src[row])
                existing_last_n = int(last_n_src[row])
                planned_last_n = int(prefill_capture_last_n_by_row[row])
                if planned_last_n > 0:
                    effective_producer.append(int(_LOGF_PRODUCER_ATTN))
                    effective_last_n.append(int(planned_last_n))
                else:
                    effective_producer.append(int(existing_producer))
                    effective_last_n.append(int(existing_last_n))
                if int(effective_producer[-1]) == int(_LOGF_PRODUCER_ATTN):
                    effective_has_capture = True
            effective_step_authority = type("EffectiveStepAuthority", (), {})()
            effective_step_authority.batch_size = int(step_authority.batch_size)
            effective_step_authority.is_prefill_by_row = (
                step_authority.is_prefill_by_row
            )
            effective_step_authority.use_compact_by_row = (
                step_authority.use_compact_by_row
            )
            effective_step_authority.dispatch_logf_producer_by_row = tuple(effective_producer)
            effective_step_authority.logits_last_n_by_row = tuple(effective_last_n)
            effective_step_authority.has_compact_row = bool(
                step_authority.has_compact_row
            )
            effective_step_authority.hint_has_log_f = bool(
                effective_has_capture
            )
    row_plan = build_mixed_page_row_plan(
        effective_step_authority,
        batch_size=batch_size,
        device="cpu",
    )
    live_has_selected = bool(row_plan.has_selected_consume)
    profile_q_lens = step_authority.q_lens_by_row
    profile_decode_like = all(int(v) <= 1 for v in profile_q_lens)
    if (
        is_profile_step
        and bool(profile_decode_like)
        and _profile_mixed_page_cudagraph_capture_enabled(controller)
    ):
        live_has_selected = True
    live_has_capture = bool(row_plan.has_capture)
    is_prefill_by_row = step_authority.is_prefill_by_row
    live_has_compact_recent = False
    live_route = route_attention_launch(
        has_selected_consume=live_has_selected,
        has_capture=live_has_capture,
        has_compact_recent=live_has_compact_recent,
    )
    if (
        not is_profile_step
        and not live_has_selected
        and not live_has_capture
        and not live_has_compact_recent
    ):
        live_route = _MIXED_PAGE_ROUTE
    from patches.fa3_native.install import append_fa3_route_trace, fa3_route_trace_enabled

    if fa3_route_trace_enabled():
        append_fa3_route_trace(
            {
                "event": "fa3_live_route_decision",
                "epoch": int(getattr(step_authority, "epoch", -1)),
                "step_identity_token": int(step_token),
                "batch_size": int(total_rows),
                "route": str(live_route),
                "has_selected_consume": bool(live_has_selected),
                "has_capture": bool(live_has_capture),
                "has_compact_recent": bool(live_has_compact_recent),
                "vllm_profile_step": bool(is_profile_step),
                "is_prefill_by_row": [
                    bool(v)
                    for v in is_prefill_by_row
                ],
                "bootstrap_done_by_row": [
                    bool(v)
                    for v in step_authority.row_policy_ready_by_row
                ],
                "use_compact_by_row": [
                    bool(v)
                    for v in step_authority.use_compact_by_row
                ],
                "row_mode_by_row": [
                    int(v)
                    for v in step_authority.row_mode_by_row
                ],
                "logits_last_n_by_row": [
                    int(v)
                    for v in step_authority.logits_last_n_by_row
                ],
            }
        )
    setattr(controller, "_fa3_live_route_token", step_token)
    setattr(controller, "_fa3_live_route_batch_size", int(batch_size if batch_size is not None else -1))
    setattr(controller, "_fa3_live_route_has_selected_consume", live_has_selected)
    setattr(controller, "_fa3_live_route_has_capture", live_has_capture)
    setattr(controller, "_fa3_live_route_has_compact_recent", live_has_compact_recent)
    setattr(controller, "_fa3_live_route", live_route)
    bind_launch_route_hints(
        attn_metadata,
        has_selected_consume=live_has_selected,
        has_capture=live_has_capture,
        has_compact_recent=live_has_compact_recent,
    )
    return live_route


def _deserialize_config(payload: str) -> Optional[SparseControllerConfig]:
    try:
        raw = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid {_SERIALIZED_CONFIG_ENV}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{_SERIALIZED_CONFIG_ENV} must be a JSON object")
    raw.pop("adaptive", None)
    raw.pop("tau", None)
    alpha_raw = dict(raw.get("alpha_fair", {}) or {})
    alpha_fields = {f.name for f in fields(AlphaFairSelectorConfig)}
    for alpha_field in alpha_fields:
        legacy_key = f"alpha_{alpha_field}"
        if legacy_key in raw and alpha_field not in alpha_raw:
            alpha_raw[alpha_field] = raw[legacy_key]
    # #15: 陈旧 payload(已退役 page_* 等)只丢键不丢命;带语义的退役模式值仍 fail-closed
    unknown_alpha = set(alpha_raw) - alpha_fields
    if unknown_alpha:
        _log.warning(
            "%s: dropping unknown/retired alpha_fair fields: %s",
            _SERIALIZED_CONFIG_ENV, sorted(unknown_alpha),
        )
        alpha_raw = {k: v for k, v in alpha_raw.items() if k in alpha_fields}
    selection_mode = str(alpha_raw.get("selection_mode", "token_topk"))
    if selection_mode != "token_topk":
        raise ValueError(
            f"{_SERIALIZED_CONFIG_ENV}: unsupported alpha selection_mode: {selection_mode!r}"
        )
    alpha_cfg = AlphaFairSelectorConfig(**alpha_raw)
    raw["alpha_fair"] = alpha_cfg
    trigger_raw = raw.get("trigger", {})
    if "single_end_tokens" in trigger_raw:
        trigger_raw["single_end_tokens"] = set(trigger_raw["single_end_tokens"])
    if "pair_end_tokens" in trigger_raw:
        trigger_raw["pair_end_tokens"] = {tuple(pair) for pair in trigger_raw["pair_end_tokens"]}
    if "start_exclude_tokens" in trigger_raw:
        trigger_raw["start_exclude_tokens"] = set(trigger_raw["start_exclude_tokens"])
    trigger_cfg = RefreshTriggerConfig(**trigger_raw)
    raw["trigger"] = trigger_cfg
    # 过滤掉 SparseControllerConfig 不接受的未知字段——但必须响亮:字段名打错
    # (如 residency 三字段拼写错)静默丢弃=配置静默不生效(sparse 活性陷阱),
    # 对齐 alpha_fair 侧 unknown-field warning 的既有做法。
    known_fields = {f.name for f in fields(SparseControllerConfig)}
    unknown_top = set(raw) - known_fields
    if unknown_top:
        _log.warning(
            "%s: dropping unknown/retired top-level config fields "
            "(typo'd fields silently do nothing — check spelling): %s",
            _SERIALIZED_CONFIG_ENV,
            sorted(unknown_top),
        )
    raw = {k: v for k, v in raw.items() if k in known_fields}
    try:
        return SparseControllerConfig(**raw)
    except TypeError:
        _log.error("Failed to deserialize SparseControllerConfig from env", exc_info=True)
        raise

def _patch_prepare_inputs() -> None:
    global _PREPARE_PATCHED, _ORIGINAL_PREPARE_INPUTS
    if _PREPARE_PATCHED:
        return
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner  # type: ignore[import]
    except Exception:
        _log.warning("Cannot import GPUModelRunner for _prepare_inputs patch, skipping")
        return

    _orig_prepare_raw_for_split = GPUModelRunner._prepare_inputs  # type: ignore[attr-defined]

    # [PREP-FWD-SPLIT-RETIRE 2026-07-07] VLLM_SPARSE_PREP_FWD_SPLIT 计时残码整删：
    # 汇聚字典 _PREP_FWD_SPLIT_US 全仓无定义亦无消费者（开 env 即 NameError），
    # 关闭时每步白付 1-4 次 os.environ.get——从未生效的观测探针。
    def original_prepare(self, *a, **k):
        return _orig_prepare_raw_for_split(self, *a, **k)

    def _sparse_prepare_inputs(self, scheduler_output, num_scheduled_tokens=None):
        controller = _GLOBAL_CONTROLLER or _ensure_controller()
        if num_scheduled_tokens is None:
            result = original_prepare(self, scheduler_output)
        else:
            result = original_prepare(self, scheduler_output, num_scheduled_tokens)

        if controller is not None:
            assert_cleanup_ledgers_drained_for_step_build(
                controller,
                stage="prepare_inputs",
            )
            num_reqs = int(getattr(self.input_batch, "num_reqs", 0) or 0)
            input_num_reqs = num_reqs
            req_ids: List[str] = []
            block_tables: List[torch.Tensor] = []
            block_tables_cpu: List[object | None] = []
            if input_num_reqs > 0 and hasattr(self.input_batch, "req_ids"):
                req_ids = list(self.input_batch.req_ids[:num_reqs])
            if input_num_reqs > 0:
                block_table_group = getattr(self.input_batch, "block_table", None)
                if block_table_group is not None:
                    tables = getattr(block_table_group, "block_tables", [])
                    for table in tables:
                        block_tables.append(
                            _get_block_table_device_tensor(table, input_num_reqs)
                        )
                        block_tables_cpu.append(
                            _get_block_table_cpu_source(table, input_num_reqs)
                        )
            # register_block_tables 使用全局 req_ids 集合来维护 request_states，
            # 不感知 per-layer slot 布局，因此与 LayerState.align_slots 的改动解耦。
            try:
                scheduled = scheduler_output.num_scheduled_tokens
            except Exception as exc:
                raise RuntimeError(
                    "prepare_inputs failed to read scheduler_output.num_scheduled_tokens"
                ) from exc

            step_ticket = _build_step_ticket(
                controller=controller,
                req_ids=req_ids,
                num_scheduled_tokens=scheduled,
                finished_req_ids=set(),
                dispatch_token=int(
                    getattr(self, "_sparse_worker_dispatch_token", -1)
                ),
            )
            snapshot = controller.get_or_build_active_step_snapshot(
                req_ids=req_ids,
                num_scheduled_tokens=scheduled,
                step_ticket=step_ticket,
            )
            req_ids = list(snapshot.active_req_ids)
            active_row_indices = tuple(int(i) for i in snapshot.active_row_indices)
            num_scheduled_tokens = [int(v) for v in snapshot.num_scheduled_tokens]
            snapshot_q_start = getattr(snapshot, "q_start_loc", None)
            if snapshot_q_start is None:
                snapshot_q_start = getattr(snapshot, "query_start_loc", None)
            if snapshot_q_start is None:
                raise RuntimeError("active step snapshot missing q_start_loc")
            q_start_loc = [int(v) for v in snapshot_q_start]

            # row-index tensor 只按 snapshot 签名失效，不再重算 active compaction。
            cache_key = tuple(int(v) for v in snapshot.snapshot_signature)
            cache_entry = getattr(self, "_sparse_active_compaction_cache_entry", None)
            if (
                not isinstance(cache_entry, dict)
                or tuple(cache_entry.get("key", tuple())) != cache_key
            ):
                cache_entry = {
                    "key": cache_key,
                    "active_row_indices": tuple(int(i) for i in active_row_indices),
                    "row_index_tensor_by_device": {},
                }
                self._sparse_active_compaction_cache_entry = cache_entry
            elif tuple(cache_entry.get("active_row_indices", tuple())) != active_row_indices:
                cache_entry["active_row_indices"] = active_row_indices
                cache_entry["row_index_tensor_by_device"] = {}

            row_index_tensor_by_device = cache_entry.get("row_index_tensor_by_device", {})
            if not isinstance(row_index_tensor_by_device, dict):
                row_index_tensor_by_device = {}
                cache_entry["row_index_tensor_by_device"] = row_index_tensor_by_device

            if len(active_row_indices) != input_num_reqs:
                compacted_tables: List[torch.Tensor] = []
                compacted_tables_cpu: List[object | None] = []
                for table, table_cpu in zip(block_tables, block_tables_cpu, strict=True):
                    if active_row_indices:
                        device_key = (
                            str(table.device.type),
                            int(table.device.index) if table.device.index is not None else -1,
                        )
                        row_index_tensor = row_index_tensor_by_device.get(device_key)
                        if (
                            row_index_tensor is None
                            or row_index_tensor.device != table.device
                            or row_index_tensor.dtype != torch.long
                            or int(row_index_tensor.numel()) != len(active_row_indices)
                        ):
                            row_index_tensor = torch.tensor(
                                active_row_indices,
                                dtype=torch.long,
                                device=table.device,
                            )
                            row_index_tensor_by_device[device_key] = row_index_tensor
                        compacted_tables.append(table.index_select(0, row_index_tensor))
                        compacted_tables_cpu.append(
                            _slice_block_table_cpu_rows(table_cpu, active_row_indices)
                        )
                    else:
                        compacted_tables.append(table[:0])
                        compacted_tables_cpu.append(
                            _slice_block_table_cpu_rows(table_cpu, tuple())
                        )
                block_tables = compacted_tables
                block_tables_cpu = compacted_tables_cpu
            num_reqs = len(req_ids)

            if num_reqs > 0:
                computed_tokens: List[int]
                num_computed = None  # sentinel; set below if available
                try:
                    num_computed = self.input_batch.num_computed_tokens_cpu
                    computed_tokens = [0] * len(req_ids)
                    seq_lens = [0] * len(req_ids)
                except Exception:
                    _log.warning("Failed to read num_computed_tokens_cpu from input_batch", exc_info=True)
                    raise

                # [TPX-D5] tp_size 是 engine 生命周期常量：首步解析后缓存到 runner，
                # 免每步双层 getattr 链（同 step_context_worker._cached_tp_size 先例）。
                tp_size = getattr(self, "_sfi_cached_tp_size", None)
                if tp_size is None:
                    parallel_config = getattr(getattr(self, "vllm_config", None), "parallel_config", None)
                    tp_size = int(getattr(parallel_config, "tensor_parallel_size", 1) or 1)
                    self._sfi_cached_tp_size = tp_size
                prompt_array = getattr(self.input_batch, "num_prompt_tokens", None)
                index_map = getattr(self.input_batch, "req_id_to_index", None)
                if not isinstance(req_ids, list):
                    raise RuntimeError(f"{E_TP_INPUT_CONTRACT}: req_ids must be list")
                if num_computed is None:
                    raise RuntimeError(f"{E_TP_INPUT_CONTRACT}: num_computed_tokens_cpu missing")
                token_ids_cpu = getattr(self.input_batch, "token_ids_cpu", None)
                token_ids_cpu_rows: Optional[int] = None
                if token_ids_cpu is not None:
                    try:
                        token_ids_cpu_rows = int(len(token_ids_cpu))
                    except Exception as exc:
                        raise RuntimeError(
                            f"{E_TP_INPUT_CONTRACT}: token_ids_cpu row count unavailable"
                        ) from exc
                mapped_indices = validate_tp_input_contract(
                    req_ids=req_ids,
                    req_id_to_index=index_map,
                    num_computed_tokens_len=int(len(num_computed)),
                    num_prompt_tokens_len=(
                        int(len(prompt_array))
                        if prompt_array is not None
                        else None
                    ),
                    token_ids_rows=token_ids_cpu_rows,
                    tp_size=tp_size,
                )
                prompt_lengths: List[int] = [0] * len(req_ids)
                num_tokens_no_spec_cpu = getattr(self.input_batch, "num_tokens_no_spec", None)
                if num_tokens_no_spec_cpu is not None:
                    try:
                        num_tokens_no_spec_len = int(len(num_tokens_no_spec_cpu))
                    except Exception as exc:
                        raise RuntimeError(
                            f"{E_TP_INPUT_CONTRACT}: num_tokens_no_spec row count unavailable"
                        ) from exc
                else:
                    num_tokens_no_spec_len = None
                for i, idx_int in enumerate(mapped_indices):
                    # prompt len（必须按 req_id_to_index 映射，不能假设数组与 req_ids 顺序一致）
                    if prompt_array is not None:
                        prompt_lengths[i] = int(prompt_array[idx_int])

                    # computed tokens（同上：按 req_id_to_index 映射）
                    computed = int(num_computed[idx_int])
                    computed_tokens[i] = computed

                    # seq_len（=computed + scheduled）用于本步 metadata（同 req_ids 顺序）
                    scheduled = int(num_scheduled_tokens[i]) if i < len(num_scheduled_tokens) else 0
                    seq_lens[i] = computed + scheduled
                # [TPX-D5] 取证开关进程内不变：生产走 import-time 缓存，
                # pytest/显式动态档 live 读。
                route_trace_enabled = (
                    bool(
                        str(
                            os.environ.get(
                                "VLLM_SPARSE_FA3_ROUTE_TRACE_LOG",
                                "",
                            )
                        ).strip()
                    )
                    if _DYNAMIC_ENV
                    else _FA3_ROUTE_TRACE_LOG_RAW_CACHED
                )
                if route_trace_enabled:
                    from patches.fa3_native.install import append_fa3_route_trace

                    num_tokens_no_spec_values: List[int] = [0] * len(req_ids)
                    if num_tokens_no_spec_cpu is not None and num_tokens_no_spec_len is not None:
                        for i, idx_int in enumerate(mapped_indices):
                            if idx_int < num_tokens_no_spec_len:
                                num_tokens_no_spec_values[i] = int(num_tokens_no_spec_cpu[idx_int])
                    append_fa3_route_trace(
                        {
                            "event": "prepare_step_lengths",
                            "epoch": int(step_ticket.target_epoch),
                            "req_ids": [str(req_id) for req_id in req_ids],
                            "mapped_indices": [int(v) for v in mapped_indices],
                            "prompt_lengths": [int(v) for v in prompt_lengths],
                            "num_computed_tokens": [int(v) for v in computed_tokens],
                            "num_tokens_no_spec": [int(v) for v in num_tokens_no_spec_values],
                            "num_scheduled_tokens": [
                                int(num_scheduled_tokens[i]) if i < len(num_scheduled_tokens) else 0
                                for i in range(len(req_ids))
                            ],
                            "seq_lens": [int(v) for v in seq_lens],
                        }
                    )
                if tp_size > 1:
                    ensure_tp_prompt_lengths(
                        prompt_lengths=prompt_lengths,
                        num_reqs=num_reqs,
                    )
                controller.record_prompt_tokens(req_ids, prompt_lengths)

                # [TP-ASYNC-HARVEST] async 调度下，绑定 worker token source 之前
                # 先排空搭车收割的采样 token 副本（FIFO、event.query 非阻塞），把
                # 每请求的真 token 写到该行最老的 -1 槽位。-1 占位符本身就是修复
                # 位置队列（vLLM 每个 async 步恰写一个），无需每请求水位账；
                # resume/preempt 重建的全实值行没有 -1，天然跳过。语义与 TP=1
                # enginecore hook 的事后补喂同构（1 步滞后）。
                _stash = getattr(controller, "_async_sampled_stash", None)
                if (
                    _stash
                    and tp_size > 1
                    and token_ids_cpu is not None
                    and index_map
                    and prompt_array is not None
                    and num_tokens_no_spec_cpu is not None
                ):
                    # 待修占位符必然聚在已写区尾部（vLLM 每 async 步恰写一个、
                    # FIFO 修复恰消一个）。扫描宽度直接取本次真实 stash 深度+1，
                    # 不再假设“最多滞后 5 步”；稳态深度=2 时只扫 3 格，复杂调度
                    # 下则覆盖全部真实 backlog，不漏修也不做全输出历史扫描。
                    # [HARVEST-FIXED-LAG 2026-07-08] 消费"除最新一档外"全部档。
                    # 原 [TP-DET-HARVEST-COUNT] 把 query→break 改成必消到空,
                    # rank 无关性正确,但最新档的 D2H 上游=本批 forward+采样:
                    # async 调度下 step N 的 prepare 与 step N-1 的 GPU 并行,
                    # 等最新档=把流水线重叠吃成串行(TP2 12k 实测 prepare 空洞
                    # p50 9.35ms 的主嫌)。改档龄判据:len-1 是 len 的纯函数、
                    # stash 每 async 步两 rank 锁步各进一档→消费数 rank 无关
                    # (决定论与必消到空同强);老档事件在上一步墙钟内已完成=
                    # 稳态零阻塞。语义=修复滞后 1→2 步:evaluate 侧本就按实值
                    # 前缀喂入(vllm_sparse_patch "实值前缀截断"合同:尾部 -1
                    # 容忍,水位只推已喂部分),-1 回填 6 元素尾窗容纳 stash 深
                    # 度 4≥2。query/synchronize 臂保留=正确性护栏(老档理应已
                    # 完成,GPU 极端落后时正确等待,绝不读半成品缓冲)。
                    _repair_tail_span = len(_stash) + 1
                    while len(_stash) > 1:
                        _ids_cpu, _ready_evt, _prev_map = _stash[0]
                        if _ready_evt is not None:
                            # [TP-DET-HARVEST-COUNT 2026-07-08] 本步消档数必须
                            # rank 无关。原 `if not query(): break` 以 per-rank
                            # 的 D2H 完成态决定喂几个 token：两 rank 完成时刻不同
                            # → 喂入 token 数分叉 → sentence 触发票落在不同 decode
                            # 步 → refresh 组成分叉 → TP all_reduce 序列长度分叉 →
                            # 对端 NCCL 永久 spin 楔死(32k×TP2×双代 blocking 态
                            # 实证；与 TP-DET-TRIGGER / BOOTSTRAP-PUBLISH-SUBMIT-
                            # FINAL 同族=决策读了 GPU 完成态)。根修:query-first
                            # 后必消——未就绪则 host 等(稳态上一步早已完成=query
                            # 即过零成本),while 恒消到 stash 空=rank 无关的确定
                            # 终止条件,两 rank 每步喂入 token 前缀逐位相同。
                            # [GUARD-NO-SWALLOW] query/synchronize 失败自然抛(无
                            # try),绝不吞成假就绪(=读半成品缓冲=TP 静默错 token)。
                            if not _ready_evt.query():
                                _ready_evt.synchronize()
                        if (
                            hasattr(_ids_cpu, "shape")
                            and len(getattr(_ids_cpu, "shape", ())) == 2
                            and int(_ids_cpu.shape[-1]) != 1
                        ):
                            raise RuntimeError(
                                "E_TP2_TOKEN_SOURCE_INCOMPLETE: async sampled ids "
                                f"max_gen_len={int(_ids_cpu.shape[-1])} != 1 (spec "
                                "decode unsupported with sparse TP async); rerun "
                                "with --scheduling-mode sync"
                            )
                        # [TPX-D1] 档级一次 numpy 零拷贝视图：torch CPU 标量索引
                        # ~3.3µs/请求 vs 视图索引 ~0.1µs（event 已就绪后才读，
                        # 无同步语义变化）。若声明为 tensor 却不能导出 CPU numpy
                        # 视图，这是 rank-local token archive 损坏；不可回退后静默
                        # 丢 token，否则 sentence-trigger 决策会跨 rank 分叉。
                        if hasattr(_ids_cpu, "numpy"):
                            try:
                                _ids_view = _ids_cpu.numpy()
                            except (RuntimeError, TypeError, ValueError) as exc:
                                raise RuntimeError(
                                    "E_TP2_TOKEN_SOURCE_INCOMPLETE: async sampled "
                                    "ids cannot expose a CPU numpy view; "
                                    f"shape={getattr(_ids_cpu, 'shape', None)!r}, "
                                    f"device={getattr(_ids_cpu, 'device', None)!r}"
                                ) from exc
                        else:
                            _ids_view = _ids_cpu
                        for _rid, _prev_idx in _prev_map.items():
                            _cur = index_map.get(str(_rid))
                            if _cur is None:
                                continue
                            _cur = int(_cur)
                            _tok = _read_async_sampled_token(
                                _ids_view,
                                request_id=_rid,
                                previous_row_index=_prev_idx,
                            )
                            if _tok < 0:
                                # discard 行：vLLM 在该步也没写占位符，两侧同跳。
                                continue
                            _row_end = int(num_tokens_no_spec_cpu[_cur])
                            _p_len = int(prompt_array[_cur])
                            if _row_end <= _p_len:
                                continue
                            _lo = max(_p_len, _row_end - int(_repair_tail_span))
                            # [TPX-D2] 找最老 -1 槽位首个命中即写；窗口由真实
                            # backlog 派生。无 -1（resume 重建全实值行）则不写。
                            for _j in range(_lo, _row_end):
                                if token_ids_cpu[_cur, _j] < 0:
                                    token_ids_cpu[_cur, _j] = _tok
                                    break
                        _stash.popleft()
                # Step-wise bind: always point controller to current input_batch
                # token source to avoid stale references across steps/engines.
                # R.3: 同 step reentry 且 source_signature 不变时，允许复用上一步绑定，
                # 避免 sentence trigger 在 worker token source 暂缺时误报错。
                src_epoch = int(getattr(controller, "_worker_token_source_epoch", -1))
                src_sig = int(getattr(controller, "_worker_token_source_signature", -1))
                reuse_step_source = bool(
                    token_ids_cpu is None
                    and getattr(controller, "_worker_token_ids_cpu", None) is not None
                    and getattr(controller, "_worker_batch_id_to_idx", None) is not None
                    and src_epoch == int(step_ticket.target_epoch)
                    and src_sig == int(step_ticket.source_signature)
                )
                if not reuse_step_source:
                    controller._worker_token_ids_cpu = None
                    controller._worker_batch_id_to_idx = None
                    controller._worker_token_source_rows = -1
                    controller._worker_token_source_epoch = -1
                    controller._worker_token_source_signature = -1
                # Freeze mapping snapshot for this step. input_batch.req_id_to_index may be
                # reused/mutated by scheduler in subsequent steps.
                batch_id_to_idx = {
                    str(req_ids[i]): int(mapped_indices[i])
                    for i in range(len(req_ids))
                }
                if token_ids_cpu is not None:
                    controller._worker_token_ids_cpu = token_ids_cpu
                    controller._worker_batch_id_to_idx = batch_id_to_idx
                    # [DEAD-SENTINEL-RETIRE] version/data_ptr 哨兵已删除：token 源是
                    # vLLM 的 numpy 视图（gpu_input_batch .numpy()），无 _version /
                    # data_ptr 属性，两支检查自设计起恒短路（零保护的虚假合同）。
                    # 活的保护 = rows 哨兵 + evaluate 侧实值前缀/FIFO 序校验。
                    controller._worker_token_source_rows = int(
                        token_ids_cpu_rows
                        if token_ids_cpu_rows is not None
                        else len(token_ids_cpu)
                    )
                    controller._worker_token_source_epoch = int(step_ticket.target_epoch)
                    controller._worker_token_source_signature = int(step_ticket.source_signature)

                controller.tp_size = tp_size

                controller.prepare_step_context(
                    req_ids=req_ids,
                    num_scheduled_tokens=num_scheduled_tokens,
                    q_start_loc=q_start_loc,
                    seq_lens=seq_lens,
                    num_actual_tokens=int(getattr(scheduler_output, "total_num_scheduled_tokens", 0)),
                    prompt_lengths=prompt_lengths,
                    num_computed_tokens=computed_tokens,
                    step_ticket=step_ticket,
                )
            else:
                controller.prepare_step_context(
                    req_ids=[],
                    num_scheduled_tokens=[],
                    q_start_loc=[0],
                    seq_lens=[],
                    num_actual_tokens=0,
                    prompt_lengths=[],
                    num_computed_tokens=[],
                    step_ticket=step_ticket,
                )
            controller.register_block_tables(
                block_tables,
                req_ids,
                finished_req_ids=set(),
                block_tables_cpu=block_tables_cpu,
            )
            if num_reqs:
                attn_metadata_obj = None
                try:
                    if isinstance(result, tuple) and result:
                        attn_mapping = result[0]
                        if isinstance(attn_mapping, dict) and attn_mapping:
                            attn_metadata_obj = next(iter(attn_mapping.values()))
                except Exception:
                    _log.warning("Failed to extract attn_metadata from prepare_inputs result", exc_info=True)
                    raise

                if attn_metadata_obj is not None:
                    builder0 = None
                    if hasattr(self, "attn_metadata_builders") and self.attn_metadata_builders:
                        builder0 = self.attn_metadata_builders[0]
                    _maybe_build_step_decode_data_from_attn_metadata(
                        controller=controller,
                        attn_metadata=attn_metadata_obj,
                        metadata_builder=builder0,
                    )

                    if not _mixed_page_full_cudagraph_replay_refresh_enabled(
                        controller
                    ):
                        _refresh_mixed_page_resolver_replay_carriers_from_attn_metadata(
                            attn_metadata_obj
                        )
            # The graph dispatcher runs immediately after _prepare_inputs.
            # Arm a request-local, step-identity-bound admission ticket only
            # when the CPU prefill plan contains an actual capture producer.
            admission_step_authority = getattr(
                controller,
                "step_authority",
                None,
            )
            if bool(
                getattr(admission_step_authority, "has_prefill_row", False)
            ):
                _arm_prefill_capture_cudagraph_admission(self, controller)
        return result

    GPUModelRunner._prepare_inputs = _sparse_prepare_inputs  # type: ignore[assignment]
    _ORIGINAL_PREPARE_INPUTS = original_prepare
    _PREPARE_PATCHED = True


_PREFILL_CAPTURE_CUDAGRAPH_ADMISSION_ATTR = (
    "_sfi_prefill_capture_cudagraph_admission_identity"
)


def _resolve_active_prefill_capture_admission(
    controller: object,
) -> Optional[Tuple[Tuple[int, int, int, int], Tuple[str, ...]]]:
    """Return current CPU identity and exact finalizing prefill producers.

    FULL graph replay cannot execute the request-local Python attention
    producer.  This predicate is deliberately evaluated after
    ``prepare_step_context``: the prefill plan is then a step-scoped CPU truth,
    and steady decode exits after the ``has_prefill_row`` branch without
    scanning requests or touching CUDA.
    """
    step_context = getattr(controller, "step_context", None)
    step_authority = getattr(controller, "step_authority", None)
    if step_context is None or step_authority is None:
        return None
    if not bool(getattr(step_authority, "has_prefill_row", False)):
        return None
    if getattr(step_context, "step_authority", None) is not step_authority:
        raise RuntimeError(
            "E_PREFILL_CAPTURE_GRAPH_ADMISSION: step authority ownership drift"
        )
    get_plan = getattr(controller, "get_step_prefill_plan_by_req", None)
    if not callable(get_plan):
        raise RuntimeError(
            "E_PREFILL_CAPTURE_GRAPH_ADMISSION: active prefill has no "
            "capture-plan authority"
        )
    capture_plan_by_req, finalize_req_ids = get_plan(step_context=step_context)
    if not capture_plan_by_req:
        return None
    return (
        (
            int(getattr(step_authority, "epoch", -1)),
            int(getattr(step_authority, "step_handle_id", -1)),
            int(getattr(step_authority, "step_handle_generation", -1)),
            int(getattr(step_authority, "step_identity_token", -1)),
        ),
        tuple(str(rid) for rid in finalize_req_ids),
    )


def _active_prefill_capture_admission_identity(
    controller: object,
) -> Optional[Tuple[int, int, int, int]]:
    admission = _resolve_active_prefill_capture_admission(controller)
    return admission[0] if admission is not None else None


def _arm_prefill_capture_cudagraph_admission(
    runner: object,
    controller: object,
) -> None:
    # Steady decode stops here.  Keep the full step-context/plan identity
    # validation on the rare prefill branch rather than paying that helper
    # call and its extra attribute reads on every decode step.
    step_authority = getattr(controller, "step_authority", None)
    if step_authority is None or not bool(
        getattr(step_authority, "has_prefill_row", False)
    ):
        return
    admission = _resolve_active_prefill_capture_admission(controller)
    if admission is not None:
        identity, finalize_req_ids = admission
        if finalize_req_ids:
            arm_submission_boundary = getattr(
                controller,
                "_arm_prefill_submission_boundary",
                None,
            )
            if not callable(arm_submission_boundary):
                raise RuntimeError(
                    "E_TP_BOOTSTRAP_SUBMISSION_LEDGER_UNAVAILABLE"
                )
            arm_submission_boundary(
                request_ids=finalize_req_ids,
                epoch=int(identity[0]),
            )
        setattr(
            runner,
            _PREFILL_CAPTURE_CUDAGRAPH_ADMISSION_ATTR,
            (
                int(getattr(runner, "_sparse_worker_dispatch_token", -1)),
                identity,
            ),
        )


class _FullCudagraphIneligibleDispatcher:
    """A call-scoped dispatcher view that preserves PIECEWISE/NONE fallback."""

    __slots__ = ("_dispatcher", "_full_mode")

    def __init__(self, dispatcher: object, full_mode: object) -> None:
        self._dispatcher = dispatcher
        self._full_mode = full_mode

    def dispatch(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        invalid_modes = kwargs.get("invalid_modes")
        if invalid_modes is None:
            kwargs["invalid_modes"] = frozenset((self._full_mode,))
        elif self._full_mode not in invalid_modes:
            kwargs["invalid_modes"] = frozenset(invalid_modes) | frozenset(
                (self._full_mode,)
            )
        return self._dispatcher.dispatch(*args, **kwargs)

    def __getattr__(self, name: str) -> object:
        return getattr(self._dispatcher, name)


def _preflight_batch_execution_admission_hook_lease() -> None:
    """Validate admission-hook ownership before install reuse or teardown."""

    original_present = _ORIGINAL_DETERMINE_BATCH_EXECUTION is not None
    wrapper_present = _INSTALLED_DETERMINE_BATCH_EXECUTION_WRAPPER is not None
    if not _BATCH_EXECUTION_ADMISSION_PATCHED:
        if original_present or wrapper_present:
            raise RuntimeError(
                "E_PREFILL_CAPTURE_GRAPH_ADMISSION: patch lease state corrupt"
            )
        return
    if not original_present or not wrapper_present:
        raise RuntimeError(
            "E_PREFILL_CAPTURE_GRAPH_ADMISSION: patch lease state corrupt"
        )
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner  # type: ignore[import]
    except Exception as exc:
        raise RuntimeError(
            "E_PREFILL_CAPTURE_GRAPH_ADMISSION: patch lease interface unavailable"
        ) from exc
    if (
        GPUModelRunner._determine_batch_execution_and_padding
        is not _INSTALLED_DETERMINE_BATCH_EXECUTION_WRAPPER
    ):
        raise RuntimeError(
            "E_PREFILL_CAPTURE_GRAPH_ADMISSION: patch lease lost"
        )


def _patch_batch_execution_cudagraph_admission() -> None:
    """Exclude FULL only for a step with a live Python prefill producer.

    The wrapper's common path is one absent instance-attribute read.  The
    proxy allocation and dispatcher swap exist only for the rare capture
    window step, are restored in ``finally``, and never evict or recapture a
    graph.
    """
    global _BATCH_EXECUTION_ADMISSION_PATCHED
    global _ORIGINAL_DETERMINE_BATCH_EXECUTION
    global _INSTALLED_DETERMINE_BATCH_EXECUTION_WRAPPER
    _preflight_batch_execution_admission_hook_lease()
    if _BATCH_EXECUTION_ADMISSION_PATCHED:
        return
    try:
        from vllm.config import CUDAGraphMode  # type: ignore[import]
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner  # type: ignore[import]
    except Exception as exc:
        raise RuntimeError(
            "E_PREFILL_CAPTURE_GRAPH_ADMISSION: required vLLM graph "
            "admission interface is unavailable"
        ) from exc

    original_determine = GPUModelRunner._determine_batch_execution_and_padding

    def _sparse_determine_batch_execution(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        admission_ticket = getattr(
            self,
            _PREFILL_CAPTURE_CUDAGRAPH_ADMISSION_ATTR,
            None,
        )
        if admission_ticket is None:
            return original_determine(self, *args, **kwargs)
        delattr(self, _PREFILL_CAPTURE_CUDAGRAPH_ADMISSION_ATTR)

        if (
            not isinstance(admission_ticket, tuple)
            or len(admission_ticket) != 2
            or not isinstance(admission_ticket[1], tuple)
            or len(admission_ticket[1]) != 4
        ):
            raise RuntimeError(
                "E_PREFILL_CAPTURE_GRAPH_ADMISSION: malformed admission ticket"
            )
        armed_dispatch_token = int(admission_ticket[0])
        identity = tuple(int(v) for v in admission_ticket[1])

        # A ticket belongs to exactly one execute_model dispatch.  If prepare
        # armed it but determine never consumed it, the next regular dispatch
        # has a new worker token; dummy/profile work is explicitly outside the
        # live request policy.  Both cases self-retire the stale ticket without
        # penalizing the absent-ticket steady path with a cleanup lookup.
        current_dispatch_token = int(
            getattr(self, "_sparse_worker_dispatch_token", -1)
        )
        controller = _GLOBAL_CONTROLLER
        if (
            armed_dispatch_token != current_dispatch_token
            or _is_vllm_dummy_run_active(controller)
        ):
            return original_determine(self, *args, **kwargs)

        current_identity = (
            _active_prefill_capture_admission_identity(controller)
            if controller is not None
            else None
        )
        if current_identity is None:
            # Same-dispatch prepare reentry can legitimately retire a plan
            # before graph admission.  Current CPU policy is authoritative.
            return original_determine(self, *args, **kwargs)
        if current_identity != identity:
            raise RuntimeError(
                "E_PREFILL_CAPTURE_GRAPH_ADMISSION: stale step admission; "
                f"armed={identity!r} current={current_identity!r}"
            )

        dispatcher = getattr(self, "cudagraph_dispatcher", None)
        if dispatcher is None:
            raise RuntimeError(
                "E_PREFILL_CAPTURE_GRAPH_ADMISSION: runner dispatcher missing"
            )
        admission_dispatcher = _FullCudagraphIneligibleDispatcher(
            dispatcher,
            CUDAGraphMode.FULL,
        )
        self.cudagraph_dispatcher = admission_dispatcher
        try:
            result = original_determine(self, *args, **kwargs)
            if (
                isinstance(result, tuple)
                and result
                and result[0] == CUDAGraphMode.FULL
            ):
                raise RuntimeError(
                    "E_PREFILL_CAPTURE_GRAPH_ADMISSION: active prefill was "
                    "still admitted to FULL replay"
                )
            return result
        finally:
            # Restore the predecessor even when dispatch itself raises.  The
            # call-scoped capability view must never leak into a later step or
            # obscure the original exception with cleanup diagnostics.
            self.cudagraph_dispatcher = dispatcher

    GPUModelRunner._determine_batch_execution_and_padding = (  # type: ignore[assignment]
        _sparse_determine_batch_execution
    )
    _ORIGINAL_DETERMINE_BATCH_EXECUTION = original_determine
    _INSTALLED_DETERMINE_BATCH_EXECUTION_WRAPPER = (
        _sparse_determine_batch_execution
    )
    _BATCH_EXECUTION_ADMISSION_PATCHED = True


def _capture_profile_device_properties(device: torch.device):
    """Single injectable source for profile-time capability and memory facts."""
    return torch.cuda.get_device_properties(device)


def _capture_configured_device_bytes(runner, device_properties: object) -> int:
    cache_config = getattr(runner, "cache_config", None)
    utilization_raw = getattr(cache_config, "gpu_memory_utilization", None)
    try:
        utilization = float(utilization_raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "E_SFI_CAPTURE_OWNERSHIP_MEMORY: gpu_memory_utilization must be "
            "finite and within (0, 1]"
        ) from exc
    if not math.isfinite(utilization) or not 0.0 < utilization <= 1.0:
        raise RuntimeError(
            "E_SFI_CAPTURE_OWNERSHIP_MEMORY: gpu_memory_utilization must be "
            "finite and within (0, 1]"
        )
    total_memory = int(getattr(device_properties, "total_memory", 0) or 0)
    if total_memory <= 0:
        raise RuntimeError(
            "E_SFI_CAPTURE_OWNERSHIP_MEMORY: device total_memory must be positive"
        )
    configured_device_bytes = math.floor(total_memory * utilization)
    if configured_device_bytes <= 0:
        raise RuntimeError(
            "E_SFI_CAPTURE_OWNERSHIP_MEMORY: configured device bytes must be positive"
        )
    return int(configured_device_bytes)


def _capture_ownership_mode(plan: object) -> str:
    from patches.fa3_native.capture_ownership import (
        CAPTURE_OWNERSHIP_POLICY_SCHEMA,
        CHUNK_COHORT,
        RING_EARLY,
        CaptureOwnershipPlan,
    )

    if not isinstance(plan, CaptureOwnershipPlan):
        raise RuntimeError(
            "E_SFI_CAPTURE_OWNERSHIP_PLAN: stamped plan has an invalid type"
        )
    if plan.schema != CAPTURE_OWNERSHIP_POLICY_SCHEMA:
        raise RuntimeError(
            "E_SFI_CAPTURE_OWNERSHIP_PLAN: stamped plan schema mismatch"
        )
    if plan.mode not in (RING_EARLY, CHUNK_COHORT):
        raise RuntimeError(
            "E_SFI_CAPTURE_OWNERSHIP_PLAN: stamped plan mode is invalid"
        )
    return str(plan.mode)


def _chunk_cohort_runtime_scratch_binding(
    *,
    plan: object,
    global_layer_index: int,
    slot_in_chunk: int,
    aligned_k_bucket: int,
    rows_bucket: int,
    last_n_bucket: int,
    actual_k: int,
    actual_rows: int,
    actual_last_n: int,
    heads_per_rank: int,
    chunk: int,
    in_flight: int,
    baseline_reduce_group: int,
) -> tuple[int, int, int]:
    from patches.fa3_native.capture_ownership import CHUNK_COHORT

    if _capture_ownership_mode(plan) != CHUNK_COHORT:
        raise RuntimeError(
            "E_SFI_CAPTURE_OWNERSHIP_PLAN: runtime binding requires chunk_cohort"
        )

    expected_capacity = (
        int(rows_bucket)
        * int(heads_per_rank)
        * int(last_n_bucket)
        * int(aligned_k_bucket)
    )
    expected_depth = int(chunk) * int(in_flight)
    expected_values = (
        int(aligned_k_bucket),
        int(rows_bucket),
        int(last_n_bucket),
        int(heads_per_rank),
        int(chunk),
        int(in_flight),
        int(baseline_reduce_group),
        int(chunk),
        int(expected_depth),
        int(expected_depth),
        int(expected_capacity),
        int(expected_capacity) * int(expected_depth),
        int(expected_capacity) * int(expected_depth) * 2,
    )
    stamped_values = (
        int(getattr(plan, "aligned_k")),
        int(getattr(plan, "rows_cap")),
        int(getattr(plan, "last_n")),
        int(getattr(plan, "heads_per_rank")),
        int(getattr(plan, "chunk")),
        int(getattr(plan, "in_flight")),
        int(getattr(plan, "baseline_reduce_group")),
        int(getattr(plan, "cohort_size")),
        int(getattr(plan, "target_depth")),
        int(getattr(plan, "selected_depth")),
        int(getattr(plan, "elements_per_slot")),
        int(getattr(plan, "selected_capacity_elements")),
        int(getattr(plan, "selected_bytes")),
    )
    if stamped_values != expected_values:
        raise RuntimeError(
            "E_SFI_CAPTURE_OWNERSHIP_PLAN: stamped capacities do not match "
            "the live prebuilt bucket"
        )
    if not bool(getattr(plan, "dtype_is_fp16")) or int(
        getattr(plan, "element_bytes")
    ) != 2:
        raise RuntimeError(
            "E_SFI_CAPTURE_OWNERSHIP_PLAN: stamped scratch dtype contract mismatch"
        )
    if (
        int(actual_k) <= 0
        or int(actual_rows) <= 0
        or int(actual_last_n) <= 0
        or int(actual_k) > int(aligned_k_bucket)
        or int(actual_rows) > int(rows_bucket)
        or int(actual_last_n) > int(last_n_bucket)
    ):
        raise RuntimeError(
            "E_SFI_CAPTURE_OWNERSHIP_PLAN: live capture shape exceeds the "
            "stamped capacity"
        )
    global_layer_index = int(global_layer_index)
    slot_in_chunk = int(slot_in_chunk)
    cohort_size = int(getattr(plan, "cohort_size"))
    selected_depth = int(getattr(plan, "selected_depth"))
    if global_layer_index < 0 or not 0 <= slot_in_chunk < cohort_size:
        raise RuntimeError(
            "E_SFI_CAPTURE_OWNERSHIP_PLAN: invalid live layer/cohort slot"
        )
    cohort_origin = global_layer_index - slot_in_chunk
    if cohort_origin < 0 or cohort_origin % cohort_size != 0:
        raise RuntimeError(
            "E_SFI_CAPTURE_OWNERSHIP_PLAN: layer index and cohort slot disagree"
        )
    cohort_lane = (cohort_origin // cohort_size) % int(in_flight)
    scratch_slot = cohort_lane * cohort_size + slot_in_chunk
    if not 0 <= scratch_slot < selected_depth:
        raise RuntimeError(
            "E_SFI_CAPTURE_OWNERSHIP_PLAN: resolved scratch slot exceeds capacity"
        )
    return int(scratch_slot), int(cohort_size), int(selected_depth)


def _prebuild_capture_buffers(runner, controller) -> None:
    """Prebuild the bounded capture arena and DEFER scratch before profile work.

    The caller must run this before the profile ``_dummy_run``.  vLLM derives the
    KV budget from the peak allocated bytes observed during that dummy forward;
    allocating these persistent buffers afterwards records ``max(activation,
    SFI)`` even though the live request needs ``activation + SFI``.  Keeping the
    buffers resident while the profile forward runs makes the measured peak
    match their real lifetime overlap.

    This remains profile-only, one-shot and fail-closed: it only fires for the
    gt1 one-shot capture config, and the live bucket stamps are written last.
    """
    import torch

    from patches.sparse_constants import (
        _CAPTURE_CHUNK,
        _CAPTURE_IN_FLIGHT,
        _CAPTURE_REDUCE_GROUP,
        _CAPTURE_KV_BUCKET_CACHED,
    )
    from patches.fa3_native.capture_ownership import (
        CHUNK_COHORT,
        plan_capture_ownership,
    )
    from patches.fa3_native.capture_cohort_tape import (
        plan_capture_cohort_tape,
        prepare_capture_cohort_tape,
    )
    from patches.fa3_native.postprocess import (
        plan_tiled_capture_postprocess_resources,
        prepare_tiled_capture_postprocess_resources,
    )
    from patches.sparse_utils import _align_up_int, _selector_fixed_k_enabled

    cfg = getattr(controller, "config", None)
    if cfg is None or not bool(getattr(cfg, "enabled", False)):
        return
    # The DEFER capture scratch only exists for gt1 prefill capture, reached only by
    # the one_shot_bootstrap_only auto-early path. Gate on both so non-capture /
    # short-ctx runs are byte-identical to HEAD (no bucket stamped -> EDIT-1 no-op).
    last_n = int(getattr(cfg, "prefill_last_n_query", 0) or 0)
    if last_n <= 1:
        return
    if not bool(getattr(cfg, "one_shot_bootstrap_only", False)):
        return
    if not bool(getattr(controller, "_prefill_capture_meta_arena_enabled", False)):
        return
    if not torch.cuda.is_available():
        return

    max_model_len = int(getattr(runner, "max_model_len", 0) or 0)
    num_heads = int(getattr(runner, "num_query_heads", 0) or 0)
    device = getattr(runner, "device", None)
    if max_model_len <= 0 or num_heads <= 0 or device is None:
        raise RuntimeError(
            "E_SFI_CAPTURE_PREBUILD_GEOMETRY: capture prebuild requires "
            "positive max_model_len, positive num_query_heads, and a device"
        )
    dev = torch.device(device)

    kv_min = max(256, int(_CAPTURE_KV_BUCKET_CACHED))
    kv_max_bucket = max(kv_min, _align_up_int(int(max_model_len), kv_min))

    # producer_rows_worst = the live scratch dim-1 (= len(producer_rows_cpu), the number
    # of prefill-capture rows in one forward) that the prebuilt slab must byte-match.
    # The legacy auto-early DEFER path is prefill-only. A stamped chunk cohort
    # activates later at the semantic-work boundary and may own mixed layouts;
    # max_num_seqs therefore remains the real worst-case row concurrency cap.
    # Size against max_num_seqs -- the REAL forward-batch
    # concurrency cap: vLLM v1 0.19 sets max_num_running_reqs = max_num_seqs
    # (sched/scheduler.py:105) and does NOT consult max_num_partial_prefills in the
    # scheduling loop, so up to max_num_seqs prefills can share ONE forward -> live
    # dim-1 can reach max_num_seqs. Reserving that many rows (paired with the live dim-1
    # bucket -- forward_capture.py rounds the keyed dim-1 UP to the stamped
    # _capture_rows_bucket) makes concurrent prefill HIT the prebuilt slab instead of
    # re-OOMing. Cost: only the DEFER per-chunk slabs scale with dim-1 (~3.5 GiB/row *
    # num_chunks); the window=1 arena buffer is FIXED (slots_cap rounds 1 and 2 both to
    # 8), so the total reserve goes ~14.0 -> ~24.5 GiB at max_num_seqs=2 (~1.75x, KV
    # pool shrinks ~10.5 GiB), NOT a clean 2x. A KNOWN-sequential deployment can reclaim it with
    # VLLM_SPARSE_CAPTURE_PREBUILD_ROWS=1 (now COMPLETE: the same int drives BOTH the
    # prebuilt slab dim-1 AND the live dim-1 bucket via _capture_rows_bucket).
    sched = getattr(runner, "scheduler_config", None)
    # Residual-hazard clamp (audit w1hrqiqrk): the env override may only WIDEN the
    # reserve, never SHRINK it below max_num_seqs (the real forward concurrency cap;
    # sched/scheduler.py:105 max_num_running_reqs = max_num_seqs). The earlier
    # `(env or 0) or max_num_seqs` let env=1 short-circuit max_num_seqs -> bucket=1
    # under --max-num-seqs 2 -> a concurrent dim-1=2 forward MISSes the dim-1=1 slab
    # -> hot-path torch.empty re-OOM (the SAME class this fix closed). max(.., max_num_
    # seqs, env) makes the slab dim-1, the stamped _capture_rows_bucket, AND any
    # override all >= max_num_seqs unconditionally, so live concurrency can never
    # exceed the reserve and the scratch_key always HITs. A truly sequential
    # deployment reclaims memory implicitly via max_num_seqs==1, not via a free-form
    # row count that can fall below what the scheduler co-batches.
    _capture_max_num_seqs = int(getattr(sched, "max_num_seqs", 1) or 1)
    producer_rows_worst = max(
        1,
        _capture_max_num_seqs,
        int(os.environ.get("VLLM_SPARSE_CAPTURE_PREBUILD_ROWS", "0") or "0"),
    )

    # num_chunks = ceil(num_capture_layers / _CAPTURE_CHUNK). The DEFER scratch is keyed by
    # chunk_id (sync_fa4_capture_scratch_chunkid_reuse), so the live cache holds one slab per
    # chunk_id; prebuild ALL of them so the 262k capture HITS every chunk_id (a MISS would
    # hot-path torch.empty into a pool already sized for the prebuilt set). get_num_layers(
    # parallel_config) is the per-rank count (== total for PP=1, the launch config); fall back
    # through total_num_hidden_layers / hf_config. Fail-safe: if the layer count is unknown,
    # skip the prebuild (the live path reallocs, still bounded by chunk_id, no per-step churn).
    mc = getattr(runner, "model_config", None)
    pc = getattr(runner, "parallel_config", None)
    num_layers = 0
    if mc is not None:
        try:
            num_layers = int(mc.get_num_layers(pc))
        except Exception:
            num_layers = 0
        if num_layers <= 0:
            try:
                num_layers = int(mc.get_total_num_hidden_layers())
            except Exception:
                num_layers = 0
        if num_layers <= 0:
            hf = getattr(mc, "hf_text_config", None) or getattr(mc, "hf_config", None)
            num_layers = int(getattr(hf, "num_hidden_layers", 0) or 0) if hf is not None else 0
    if num_layers <= 0:
        raise RuntimeError(
            "E_SFI_CAPTURE_PREBUILD_GEOMETRY: capture prebuild could not "
            "prove the per-rank model layer count"
        )
    num_chunks = max(1, (num_layers + int(_CAPTURE_CHUNK) - 1) // int(_CAPTURE_CHUNK))

    device_properties = _capture_profile_device_properties(dev)
    capability = (
        int(getattr(device_properties, "major", -1)),
        int(getattr(device_properties, "minor", -1)),
    )
    configured_device_bytes = _capture_configured_device_bytes(
        runner, device_properties
    )
    alpha_fair = getattr(cfg, "alpha_fair", None)
    alpha = float(getattr(alpha_fair, "alpha", 0.5))
    async_refresh_enabled = getattr(controller, "_async_refresh_enabled", None)
    ensure_refresh_stream = getattr(controller, "_ensure_refresh_stream", None)
    async_owner_requested = bool(
        callable(async_refresh_enabled) and bool(async_refresh_enabled())
    )
    refresh_stream = None
    if async_owner_requested and callable(ensure_refresh_stream):
        ensure_refresh_stream(dev)
        refresh_stream = getattr(controller, "refresh_stream", None)
    async_owner_available = bool(refresh_stream is not None)

    # Both owners may use the dynamic tiled selector kernel.  Budget the exact
    # structural GPU-owner concurrency before profile work: ring_early owns one
    # workspace per scratch lane; chunk_cohort owns one per configured in-flight
    # bank.  The sealed live path never grows this GPU footprint.  Pinned H2D
    # metadata sources are a separate small FIFO and therefore cannot duplicate
    # a giant tiled workspace merely because a prior host source is still busy.
    ring_structural_slots = (
        int(_CAPTURE_REDUCE_GROUP) * int(_CAPTURE_IN_FLIGHT)
        if int(_CAPTURE_REDUCE_GROUP) > 0
        else int(_CAPTURE_CHUNK) * int(_CAPTURE_IN_FLIGHT)
    )
    ring_tiled_resources = plan_tiled_capture_postprocess_resources(
        slot_count=max(1, ring_structural_slots),
        num_rows_capacity=int(producer_rows_worst),
        num_query_heads=int(num_heads),
        logical_k_capacity=int(kv_max_bucket),
    )
    cohort_tiled_resources = plan_tiled_capture_postprocess_resources(
        slot_count=int(_CAPTURE_IN_FLIGHT),
        num_rows_capacity=int(producer_rows_worst) * int(_CAPTURE_CHUNK),
        num_query_heads=int(num_heads),
        logical_k_capacity=int(kv_max_bucket),
    )
    selector_fixed_k = bool(_selector_fixed_k_enabled())
    cohort_tape_report = plan_capture_cohort_tape(
        bank_capacity=int(_CAPTURE_IN_FLIGHT),
        layer_capacity=int(num_layers),
        rows_capacity=int(producer_rows_worst),
        num_query_heads=int(num_heads),
        logical_k_capacity=int(kv_max_bucket),
        cohort_size=int(_CAPTURE_CHUNK),
    )
    ownership_plan = plan_capture_ownership(
        one_shot=bool(getattr(cfg, "one_shot_bootstrap_only", False)),
        async_owner_available=async_owner_available,
        selector_fixed_k=selector_fixed_k,
        last_n=int(last_n),
        dtype_is_fp16=True,
        element_bytes=int(torch.finfo(torch.float16).bits // 8),
        alpha=alpha,
        capability=capability,
        aligned_k=int(kv_max_bucket),
        rows_cap=int(producer_rows_worst),
        heads_per_rank=int(num_heads),
        chunk=int(_CAPTURE_CHUNK),
        in_flight=int(_CAPTURE_IN_FLIGHT),
        layer_count=int(num_layers),
        tape_bank_count=int(cohort_tape_report.bank_capacity),
        baseline_reduce_group=int(_CAPTURE_REDUCE_GROUP),
        baseline_postprocess_device_bytes=int(
            ring_tiled_resources.total_device_bytes
        ),
        target_postprocess_device_bytes=int(
            cohort_tiled_resources.total_device_bytes
        ),
        baseline_tape_device_bytes=0,
        target_tape_device_bytes=int(cohort_tape_report.total_device_bytes),
        configured_device_bytes=int(configured_device_bytes),
    )
    if (
        int(ring_tiled_resources.structural_slot_count)
        != int(ownership_plan.baseline_depth)
        or int(cohort_tiled_resources.structural_slot_count)
        != int(ownership_plan.in_flight)
    ):
        raise RuntimeError(
            "E_SFI_CAPTURE_TILED_OWNERSHIP: structural workspace slots "
            "disagree with the immutable scratch-owner concurrency"
        )

    # (finding-5 fail-safe) The live buckets (kv / last_n / rows) are stamped LAST,
    # only after the arena reserve + every per-chunk scratch slab is resident (see the
    # end of this function). EDIT-1 / EDIT-7 / the live dim-1 bucket force-form the
    # large bucketed shape, which is correct ONLY if the matching prebuilt slab exists;
    # a partial prebuild RAISES before the stamps -> NO bucket stamped -> the live path
    # uses the raw per-forward shape == exact HEAD behaviour (still bounded by
    # chunk_id), instead of force-forming a large shape with no slab to HIT (re-OOM).

    # (2) Seal the selector workspace against the stream and maximum live
    # geometry selected above.  Live acquisition may use any smaller N/K within
    # this capacity but may never create another slot or cache key after profile.
    # This allocation must precede the dummy forward for honest KV budgeting.
    selected_tiled_resources = None
    if int(ownership_plan.selected_postprocess_device_bytes) > 0:
        if ownership_plan.mode == CHUNK_COHORT:
            if refresh_stream is None:
                raise RuntimeError(
                    "E_SFI_CAPTURE_TILED_PREBUILD: chunk cohort has no refresh stream"
                )
            tiled_stream = refresh_stream
            selected_tiled_resources = cohort_tiled_resources
        else:
            tiled_stream = (
                refresh_stream
                if refresh_stream is not None
                else torch.cuda.current_stream(device=dev)
            )
            selected_tiled_resources = ring_tiled_resources
        prepared_tiled_resources = prepare_tiled_capture_postprocess_resources(
            controller,
            dev,
            tiled_stream,
            int(selected_tiled_resources.slot_count),
            int(selected_tiled_resources.num_rows_capacity),
            int(selected_tiled_resources.num_query_heads_capacity),
            int(selected_tiled_resources.logical_k_capacity),
        )
        if prepared_tiled_resources != selected_tiled_resources or int(
            prepared_tiled_resources.total_device_bytes
        ) != int(ownership_plan.selected_postprocess_device_bytes):
            raise RuntimeError(
                "E_SFI_CAPTURE_TILED_PREBUILD: prepared resources disagree "
                "with the immutable ownership budget"
            )

    # (2b) The chunk owner publishes request-major full-layer generation banks.
    # Their complete footprint participated in the policy gate above; allocate
    # exactly that report before the ownership signature becomes visible.
    if ownership_plan.mode == CHUNK_COHORT:
        if refresh_stream is None:
            raise RuntimeError(
                "E_SFI_CAPTURE_COHORT_TAPE_PREBUILD: chunk cohort has no refresh stream"
            )
        if (
            not bool(ownership_plan.selector_fixed_k)
            or int(ownership_plan.layer_count) != int(cohort_tape_report.layer_capacity)
            or int(ownership_plan.tape_bank_count) != int(cohort_tape_report.bank_capacity)
            or int(ownership_plan.selected_tape_device_bytes)
            != int(cohort_tape_report.total_device_bytes)
        ):
            raise RuntimeError(
                "E_SFI_CAPTURE_COHORT_TAPE_PREBUILD: immutable ownership plan "
                "disagrees with the precomputed bank report"
            )
        prepare_capture_cohort_tape(
            controller=controller,
            device=dev,
            report=cohort_tape_report,
            plan_signature=str(ownership_plan.signature_sha256),
            consumer_stream=refresh_stream,
        )

    # (3) Prebuild the arena max-bucket via the existing reserve path (allocates the
    # resident capture_scores/log_f_denoms tensors). Cap first so the window=1 bucket
    # is built at exactly kv_max_bucket (arena head-stride byte-match).
    arena = getattr(controller, "prefill_capture_meta_arena", None)
    # [KEY-NORMS-MML-CAP] key_norms arena 的 stride 顶=MML 对齐值,与 capture
    # arena kv_max_cap 同点 authoritative 注入(此处 max_model_len 已知)。
    try:
        controller._key_norms_stride_floor_mml = int(kv_max_bucket)
    except Exception:
        pass
    if arena is not None:
        arena.kv_max_cap_bucket = int(kv_max_bucket)
        from patches.prefill_capture_meta_arena import (
            ArenaReservationStatus,
            CaptureArenaIntent,
        )

        arena_reservation = arena.reserve_prefill_layouts(
            step_epoch=-1,
            step_handle_id=-1,
            step_handle_generation=-1,
            intent=CaptureArenaIntent.ONE_SHOT_BOOTSTRAP,
            slot_list=tuple(range(producer_rows_worst)),
            row_list=tuple(range(producer_rows_worst)),
            batch_size=int(
                getattr(runner, "max_num_reqs", producer_rows_worst)
                or producer_rows_worst
            ),
            num_heads=int(num_heads),
            kv_needed=int(kv_max_bucket),
            device=dev,
        )
        if getattr(arena_reservation, "status", None) is not ArenaReservationStatus.READY:
            raise RuntimeError(
                "E_SFI_CAPTURE_ARENA_PREBUILD: profile-time capture arena "
                "reservation was not ready; refusing to stamp live cache buckets "
                f"(status={getattr(arena_reservation, 'status', None)!r}, "
                f"reason={getattr(arena_reservation, 'error_reason', '')!r}, "
                f"kv_max_bucket={kv_max_bucket}, rows={producer_rows_worst})"
            )

    # (4) Prebuild the exact DEFER scratch ownership selected above. ring_early keeps
    # the existing G-ring / per-chunk keys byte-for-byte. chunk_cohort owns one
    # C*in_flight-deep slab; its lane mapping and RingWarFence provide bounded overlap
    # without retaining one allocation per model chunk. The published plan is stamped
    # only after this exact key and shape are resident.
    scratch_storage_shape = (
        int(_CAPTURE_CHUNK),
        int(producer_rows_worst),
        int(num_heads),
        int(last_n),
        int(kv_max_bucket),
    )
    scratch_dtype = torch.float16
    cache_map = getattr(controller, "_fa3_capture_scratch_cache_by_key", None)
    if not isinstance(cache_map, dict):
        cache_map = {}
        setattr(controller, "_fa3_capture_scratch_cache_by_key", cache_map)
    _reduce_group = int(_CAPTURE_REDUCE_GROUP)
    _ring_slabs = int(_reduce_group) * int(_CAPTURE_IN_FLIGHT) if _reduce_group > 0 else 0
    if ownership_plan.mode == CHUNK_COHORT:
        scratch_storage_shape = (
            int(ownership_plan.selected_depth),
            int(producer_rows_worst),
            int(num_heads),
            int(last_n),
            int(kv_max_bucket),
        )
    elif _reduce_group > 0:
        # Per-G ring: ONE [G*in_flight]-deep slab (constant key) instead of num_chunks
        # chunk-deep slabs -> raw staging num_layers-deep => ring_slabs-deep (the 8.4x cut).
        # NOT yet safe to ENABLE: the live key/slot + per-G reduce + refresh->main WAR event
        # must also land before G>0 can run without corrupting reused ring slots.
        scratch_storage_shape = (
            int(_ring_slabs),
            int(producer_rows_worst),
            int(num_heads),
            int(last_n),
            int(kv_max_bucket),
        )
    _prebuild_iter = (
        1
        if ownership_plan.mode == CHUNK_COHORT or _reduce_group > 0
        else int(num_chunks)
    )
    for chunk_id in range(int(_prebuild_iter)):
        if ownership_plan.mode == CHUNK_COHORT:
            extra_key = (
                "defer_postprocess_chunk",
                int(ownership_plan.cohort_size),
                int(ownership_plan.selected_depth),
            )
        elif _reduce_group > 0:
            extra_key = ("defer_postprocess_chunk", 0, int(_ring_slabs))
        else:
            extra_key = (
                "defer_postprocess_chunk",
                int(chunk_id),
                int(_CAPTURE_CHUNK),
            )
        scratch_key = (
            str(dev.type),
            -1 if dev.index is None else int(dev.index),
            scratch_dtype,
            scratch_storage_shape,
            extra_key,
        )
        scratch_cache_hit = cache_map.get(scratch_key) is not None
        if not scratch_cache_hit:
            from patches.fa3_native.forward_capture import (
                _store_capture_scratch_cache_entry,
            )

            _store_capture_scratch_cache_entry(
                cache_owner=controller,
                cache_map=cache_map,
                scratch_key=scratch_key,
                scratch_storage=torch.empty(
                    scratch_storage_shape, device=dev, dtype=scratch_dtype
                ),
            )
        from patches.fa3_native.forward_capture import log_capture_scratch_probe

        log_capture_scratch_probe(
            source="prebuild",
            cache_key=scratch_key,
            cache_hit=bool(scratch_cache_hit),
            scratch_storage_shape=scratch_storage_shape,
            scratch_dtype=scratch_dtype,
            element_size_bytes=int(torch.finfo(scratch_dtype).bits // 8),
            actual_rows=None,
            bucket_rows=int(producer_rows_worst),
            heads=int(num_heads),
            last_n=int(last_n),
            capture_k=int(kv_max_bucket),
            reduce_group=int(_reduce_group),
            in_flight=int(_CAPTURE_IN_FLIGHT),
            key_kind=str(extra_key[0]),
        )

    # (5) Stamp the live buckets now that the selector resources, arena reserve,
    # and all num_chunks scratch
    # slabs are resident (finding-5 fail-safe; see (1) above). _capture_rows_bucket is
    # the dim-1 (concurrent-prefill row) bucket the live path rounds the keyed scratch
    # dim-1 UP to (forward_capture.py), closing the dim-1 re-OOM hazard symmetrically
    # with the kv / last_n buckets.
    setattr(controller, "_capture_kv_max_bucket", int(kv_max_bucket))
    setattr(controller, "_capture_last_n_bucket", int(last_n))
    setattr(controller, "_capture_rows_bucket", int(producer_rows_worst))
    setattr(
        controller,
        "_capture_tiled_postprocess_resource_report",
        selected_tiled_resources,
    )
    setattr(controller, "_capture_prebuilt", True)
    # The immutable plan is the publication latch: all capacities, resources,
    # streams and cache entries above must already be resident before it exists.
    setattr(controller, "_capture_ownership_plan", ownership_plan)


def _run_dummy_after_profile_capture_prebuild(
    *,
    runner: object,
    controller: object,
    dummy_context: dict[str, object],
    run_original,
):
    """Run the original dummy forward with persistent SFI buffers already live.

    Non-capture configurations return normally from the prebuilder. Once a
    capture configuration requests resident storage, allocation or contract
    failure is terminal: continuing would move allocation back onto the live
    path after vLLM has already sized its KV budget.
    """
    if (
        bool(dummy_context.get("is_profile"))
        and not bool(dummy_context.get("is_graph_capturing"))
        and not bool(getattr(controller, "_capture_prebuilt", False))
    ):
        _prebuild_capture_buffers(runner, controller)
    return run_original()


_EXP4_PREFLIGHT_LATCHED = False


def _custom_allreduce_is_live(parallel_config: object) -> bool:
    """True when vLLM custom all-reduce will IPC-register graph buffers at capture end."""
    try:
        from vllm.distributed.parallel_state import get_tp_group  # type: ignore[import]

        tp_group = get_tp_group()
        # vLLM 0.19 owns the CUDA communicator under the TP group; the custom
        # communicator is not a direct group attribute.  Reading
        # ``get_tp_group().ca_comm`` therefore returned None on every healthy
        # TP run and silently disabled this capture-safety guard.  Keep the
        # direct form only for older compatible layouts.
        device_communicator = getattr(tp_group, "device_communicator", None)
        if device_communicator is not None:
            ca = getattr(device_communicator, "ca_comm", None)
            return ca is not None and not bool(getattr(ca, "disabled", False))
        ca = getattr(tp_group, "ca_comm", None)
        if ca is not None:
            return not bool(getattr(ca, "disabled", False))
        raise AttributeError("TP group exposes no custom-all-reduce communicator")
    except Exception:
        # parallel_state 内省不可用(版本形态差异)时退到配置旗标——方向更严:
        # 预检宁可多拦(报错给出关 AR 的操作口),不可漏拦撞 capture 收尾的
        # 隐晦崩溃。
        return not bool(getattr(parallel_config, "disable_custom_all_reduce", True))


def _expandable_segments_enabled_from_env() -> bool:
    """Return whether either PyTorch allocator alias explicitly enables it.

    PyTorch allocator configuration is a comma-separated ``key:value`` list.
    Treat key/value case and surrounding whitespace as presentation details so
    a shell spelling such as ``expandable_segments:true`` cannot bypass the
    capture-safety preflight that historically matched only ``True``.
    """
    for name in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"):
        raw = os.environ.get(name, "")
        for item in raw.split(","):
            key, separator, value = item.partition(":")
            if not separator:
                continue
            if key.strip().lower() != "expandable_segments":
                continue
            if value.strip().lower() in {"1", "true"}:
                return True
    return False


def _exp4_capture_alloc_preflight(runner: object) -> None:
    """[EXP4-RUNTIME-PREFLIGHT] expandable_segments × custom-AR 图注册互斥预检。

    expandable_segments(cuMemMap 后端)分配无 cudaIpcGetMemHandle;vLLM custom
    all-reduce 在 capture 收尾 register_graph_buffers 按 IPC handle 注册图内
    AR 缓冲 → TP>1 + custom AR + graph capture + expandable = "invalid
    argument"(远端 exp4 根因,2026-07-06 本地实证)。脚本层(发布仓
    run_speed.sh)已默认 TP>1 关 expandable;本预检是随 runtime 走的启动器
    无关守卫:capture 前 fail-fast 出可操作信息。冷路径:进程内至多完整
    评估一次(条件均为进程静态)。
    """
    global _EXP4_PREFLIGHT_LATCHED
    if _EXP4_PREFLIGHT_LATCHED:
        return
    _EXP4_PREFLIGHT_LATCHED = True
    vllm_config = getattr(runner, "vllm_config", None)
    parallel_config = getattr(vllm_config, "parallel_config", None)
    if parallel_config is None:
        parallel_config = getattr(runner, "parallel_config", None)
    tp_size = int(getattr(parallel_config, "tensor_parallel_size", 1) or 1)
    if tp_size <= 1:
        return
    if not _expandable_segments_enabled_from_env():
        return
    if not _custom_allreduce_is_live(parallel_config):
        return
    raise RuntimeError(
        "E_EXP4_EXPANDABLE_CUSTOM_AR_CAPTURE: TP>1 + custom all-reduce + CUDA "
        "graph capture with PYTORCH_(CUDA_)ALLOC_CONF expandable_segments:True. "
        "cuMemMap-backed allocations expose no cudaIpc handles, so custom-AR "
        "register_graph_buffers fails with 'invalid argument' at capture end "
        "(remote exp4 root cause, locally reproduced 2026-07-06). Fix one of: "
        "unset expandable_segments for TP>1 (release run_speed.sh default), "
        "disable custom AR (--disable-custom-all-reduce or "
        "VLLM_SPARSE_FORCE_DISABLE_CUSTOM_AR=1), or run with enforce_eager."
    )


_SLOTS_CONCURRENCY_PREFLIGHT_LATCHED = False


def _sparse_slots_concurrency_preflight(runner: object, controller: object) -> None:
    """[SERVE-LIVENESS-PREFLIGHT 2026-07-09] slots vs scheduler concurrency.

    With page residency ON the global slot allocator hard-raises the moment a
    step batch carries more live requests than max_live_sparse_slots
    (global_slot_allocator "capacity exceeded") — an engine-killing death mid
    workload, far from its config root cause, killing every in-flight request
    (the serve/LongBench trap: --max-num-seqs defaults to 128+ while the
    sparse JSON pins slots to a small bench batch). vLLM v1 schedules at most
    scheduler_config.max_num_seqs requests into one forward
    (max_num_running_reqs = max_num_seqs, sched/scheduler.py:105 — same
    invariant the capture prebuild sizing relies on), so
    slots >= max_num_seqs proves the allocator can never overflow. Enforce
    that at the profile dummy_run (startup, runner config resolved), where a
    raise aborts serve before it takes traffic.
    """
    global _SLOTS_CONCURRENCY_PREFLIGHT_LATCHED
    if _SLOTS_CONCURRENCY_PREFLIGHT_LATCHED:
        return
    _SLOTS_CONCURRENCY_PREFLIGHT_LATCHED = True
    cfg = getattr(controller, "config", None)
    if cfg is None or not bool(getattr(cfg, "compact_page_residency_enabled", False)):
        return
    slots = int(getattr(cfg, "max_live_sparse_slots", 0) or 0)
    sched = getattr(runner, "scheduler_config", None)
    if sched is None:
        vllm_config = getattr(runner, "vllm_config", None)
        sched = getattr(vllm_config, "scheduler_config", None)
    max_num_seqs = int(getattr(sched, "max_num_seqs", 0) or 0)
    if max_num_seqs <= 0:
        # Config shape unknown (exotic runner) — cannot prove either way;
        # the allocator's own capacity raise remains the backstop.
        _log.warning(
            "slots-concurrency preflight: scheduler max_num_seqs unresolved; "
            "skipping the startup capacity proof"
        )
        return
    if slots < max_num_seqs:
        # LOUD warning, not a raise: slots < max_num_seqs only dies when the
        # SUBMITTED concurrency actually exceeds slots (the bench harness
        # legitimately runs batch_size==slots requests under vLLM's default
        # max_num_seqs=256 — a raise here killed that healthy shape, wg3
        # forensics). The true capacity contract stays at the allocator
        # ("capacity exceeded" raise, whose message names this config root
        # cause); this preflight makes the hazard visible BEFORE traffic.
        _log.warning(
            "W_SPARSE_SLOTS_LT_MAX_NUM_SEQS: max_live_sparse_slots=%d < "
            "scheduler max_num_seqs=%d. If more than %d requests are ever "
            "co-batched (serve/LongBench concurrency!), the sparse global "
            "slot allocator hard-raises and kills the engine mid-workload. "
            "For serve deployments set max_live_sparse_slots >= expected "
            "concurrency (lease grows by slots x blocks x 16 x "
            "KV-bytes/token x gen_count) or pass --max-num-seqs %d.",
            slots,
            max_num_seqs,
            slots,
            slots,
        )


def _patch_dummy_run() -> None:
    global _DUMMY_RUN_PATCHED, _ORIGINAL_DUMMY_RUN
    if _DUMMY_RUN_PATCHED:
        return
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner  # type: ignore[import]
    except Exception:
        _log.warning("Cannot import GPUModelRunner for _dummy_run patch, skipping")
        return

    original_dummy_run = getattr(GPUModelRunner, "_dummy_run", None)
    if original_dummy_run is None:
        return

    def _sparse_dummy_run(self, *args, **kwargs):  # type: ignore[override]
        controller = _GLOBAL_CONTROLLER or _ensure_controller()
        if controller is None:
            return original_dummy_run(self, *args, **kwargs)
        previous_depth = int(getattr(controller, "_vllm_dummy_run_depth", 0) or 0)
        setattr(controller, "_vllm_dummy_run_depth", previous_depth + 1)
        _dummy_ctx = _dummy_run_context_from_call(args, kwargs)
        setattr(controller, "_vllm_dummy_run_context", _dummy_ctx)
        # [EXP4-RUNTIME-PREFLIGHT] capture 型 dummy_run 即将录图:在 capture
        # 开始前拦下 expandable×custom-AR 致命组合(见 helper docstring)。
        if bool(_dummy_ctx.get("is_graph_capturing")):
            _exp4_capture_alloc_preflight(self)
        # [SERVE-LIVENESS-PREFLIGHT] profile 型 dummy_run=启动必经且 runner
        # 配置已解析:residency 槽容量 < 调度并发上限=运行中必炸(allocator
        # capacity raise),启动期 fail-fast(见 helper docstring)。
        if bool(_dummy_ctx.get("is_profile")):
            _sparse_slots_concurrency_preflight(self, controller)
        # PHASE-2B de-legacy: latch the real cudagraph mode so steady-state gates
        # engage compact without VLLM_SPARSE_ATTENTION_IN_CUDAGRAPH. Run-level sticky
        # (only ever set True, never cleared) so a later PIECEWISE prefill dummy-run
        # cannot turn decode compact off.
        try:
            if _dummy_context_cudagraph_is_full(
                getattr(controller, "_vllm_dummy_run_context", None)
            ):
                setattr(controller, "_sparse_attention_in_cudagraph", True)
        except Exception:
            _log.warning("PHASE-2B cudagraph latch set skipped", exc_info=True)
        try:
            # The resident SFI capture arena/scratch must exist BEFORE vLLM's
            # profile forward reaches its activation peak.  Building it after
            # original_dummy_run records max(activation, SFI), while live execution
            # needs their sum and can OOM after startup when the auto-sized KV pool
            # consumes the missing overlap.  The helper keeps the old one-shot gates
            # and failure fallback, but fixes the lifetime ordering.
            return _run_dummy_after_profile_capture_prebuild(
                runner=self,
                controller=controller,
                dummy_context=_dummy_ctx,
                run_original=lambda: original_dummy_run(self, *args, **kwargs),
            )
        finally:
            if previous_depth <= 0:
                setattr(controller, "_vllm_dummy_run_depth", 0)
                setattr(controller, "_vllm_dummy_run_context", None)
            else:
                setattr(controller, "_vllm_dummy_run_depth", previous_depth)

    GPUModelRunner._dummy_run = _sparse_dummy_run  # type: ignore[assignment]
    _ORIGINAL_DUMMY_RUN = original_dummy_run
    _DUMMY_RUN_PATCHED = True

def _patch_update_states() -> None:
    global _UPDATE_STATES_PATCHED, _ORIGINAL_UPDATE_STATES
    global _INSTALLED_UPDATE_STATES_WRAPPER
    _preflight_update_states_hook_lease()
    if _UPDATE_STATES_PATCHED:
        return
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner  # type: ignore[import]
    except Exception:
        _log.warning("Cannot import GPUModelRunner for _update_states patch, skipping")
        return

    original_update = GPUModelRunner._update_states  # type: ignore[attr-defined]
    _ORIGINAL_UPDATE_STATES = original_update

    def _sparse_update_states(self, scheduler_output):
        controller = _GLOBAL_CONTROLLER or _ensure_controller()
        if controller is not None:
            finished_req_ids = _get_finished_req_ids(scheduler_output)
            dispatch_token = int(
                getattr(self, "_sparse_worker_dispatch_token", 0)
            ) + 1
            setattr(self, "_sparse_worker_dispatch_token", dispatch_token)
            controller.consume_finished_at_worker_boundary(
                step_token=dispatch_token,
                finished_req_ids=finished_req_ids,
            )
        return original_update(self, scheduler_output)

    setattr(
        _sparse_update_states,
        "_sfi_sparse_update_states_predecessor_abi",
        SPARSE_UPDATE_STATES_PREDECESSOR_ABI,
    )
    setattr(
        _sparse_update_states,
        "_sfi_sparse_update_states_predecessor",
        original_update,
    )
    GPUModelRunner._update_states = _sparse_update_states  # type: ignore[assignment]
    _INSTALLED_UPDATE_STATES_WRAPPER = _sparse_update_states
    _UPDATE_STATES_PATCHED = True


def is_exact_sparse_update_states_predecessor(candidate: object) -> bool:
    """Return whether ``candidate`` owns the live sparse hook lease."""

    return bool(
        _UPDATE_STATES_PATCHED
        and candidate is _INSTALLED_UPDATE_STATES_WRAPPER
        and getattr(
            candidate,
            "_sfi_sparse_update_states_predecessor_abi",
            None,
        )
        == SPARSE_UPDATE_STATES_PREDECESSOR_ABI
        and getattr(
            candidate,
            "_sfi_sparse_update_states_predecessor",
            None,
        )
        is _ORIGINAL_UPDATE_STATES
    )


def _preflight_update_states_hook_lease() -> None:
    """Reject out-of-order teardown before any sparse state is mutated."""

    original_present = _ORIGINAL_UPDATE_STATES is not None
    wrapper_present = _INSTALLED_UPDATE_STATES_WRAPPER is not None
    if not _UPDATE_STATES_PATCHED:
        if original_present or wrapper_present:
            raise RuntimeError("E_SPARSE_UPDATE_STATES_HOOK_LEASE_STATE")
        return
    if not original_present or not wrapper_present:
        raise RuntimeError("E_SPARSE_UPDATE_STATES_HOOK_LEASE_STATE")
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner  # type: ignore[import]

    if GPUModelRunner._update_states is not _INSTALLED_UPDATE_STATES_WRAPPER:
        raise RuntimeError("E_SPARSE_UPDATE_STATES_HOOK_LEASE_LOST")



def _patch_flash_metadata_builder() -> None:
    global _FLASH_METADATA_PATCHED, _ORIGINAL_FLASH_METADATA_BUILD
    if _FLASH_METADATA_PATCHED:
        return
    # [TRITON-LINE-RETIRED 2026-07-07] FA3-only 后 FlashAttentionMetadataBuilder
    # 是唯一 builder:import 失败=环境坏,静默跳过等于隐性 fallback,升级
    # fail-fast(无兜底铁律)。
    from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadataBuilder  # type: ignore[import]

    original_build = FlashAttentionMetadataBuilder.build

    def _patched_build(self, common_prefix_len, common_attn_metadata):  # type: ignore[override]
        attn_metadata = original_build(self, common_prefix_len, common_attn_metadata)
        controller = _GLOBAL_CONTROLLER or _ensure_controller()
        if controller is not None:
            binding = _resolve_metadata_step_binding(
                controller=controller,
                attn_metadata=attn_metadata,
                common_attn_metadata=common_attn_metadata,
            )
            setattr(attn_metadata, "sparse_vllm_profile_step", binding.is_profile_step)
            bind_fa3_native_attn_metadata_contracts(
                attn_metadata=attn_metadata,
                step_ctx=binding.step_ctx,
                step_authority=binding.step_authority,
                snapshot=binding.snapshot,
            )
            _maybe_build_step_decode_data_from_attn_metadata(
                controller=controller,
                attn_metadata=attn_metadata,
                metadata_builder=self,
            )
        return attn_metadata

    FlashAttentionMetadataBuilder.build = _patched_build  # type: ignore[assignment]
    _ORIGINAL_FLASH_METADATA_BUILD = original_build
    _FLASH_METADATA_PATCHED = True

def _refresh_mixed_page_full_cudagraph_replay_for_forward_context(
    *,
    controller: object,
    forward_context: object,
    graph_key: str,
) -> object:
    from patches.fa_sparse_runtime.mixed_page_cudagraph_replay import (
        iter_attention_metadata,
        refresh_mixed_page_carriers_for_forward_context,
        wait_mixed_page_resolver_ready_events_for_forward_context,
    )

    step_ctx = getattr(controller, "step_context", None)
    step_id_value = getattr(
        controller,
        "step_context_epoch",
        getattr(step_ctx, "epoch", -1),
    )
    step_id = -1 if step_id_value is None else int(step_id_value)
    attn_metadata = getattr(forward_context, "attn_metadata", None)
    replay_guard = (int(step_id), id(attn_metadata))
    profile_path = _full_cudagraph_hook_profile_log(True)
    route_trace_enabled = _fa3_route_trace_enabled()
    metadata_direct_bound_cache_key = (id(attn_metadata), str(graph_key))
    graph_direct_bound_cache_key = ("resolved_row_ptr_direct_bound", str(graph_key))
    prebound_rrp_graph_state = _prebound_rrp_current_graph_state_for_forward_context(
        controller=controller,
        forward_context=forward_context,
    )
    if prebound_rrp_graph_state is not None:
        stats = _mark_prebound_rrp_full_cudagraph_replay(
            controller=controller,
            forward_context=forward_context,
            graph_key=graph_key,
            state=prebound_rrp_graph_state,
            binding_is_current_prevalidated=True,
        )
        setattr(
            controller,
            "_mixed_page_full_cudagraph_last_replay_stats",
            stats,
        )
        return stats
    rebound_for_graph_capacity = _maybe_rebind_rrp_for_full_graph_replay(
        controller=controller,
        forward_context=forward_context,
        metadata_items=iter_attention_metadata(attn_metadata),
    )
    ready_event_wait_count = wait_mixed_page_resolver_ready_events_for_forward_context(
        forward_context
    )
    if (
        step_id >= 0
        and getattr(
            forward_context,
            "mixed_page_resolver_replay_refresh_guard",
            None,
        )
        == replay_guard
    ):
        stats = getattr(
            forward_context,
            "mixed_page_resolver_replay_last_stats",
            None,
        )
        if stats is not None:
            setattr(
                controller,
                "_mixed_page_full_cudagraph_last_replay_stats",
                stats,
            )
            return stats
    def _append_replay_refresh_route_trace(
        stats: object,
        *,
        fast_cached_direct_bound: bool = False,
        cache_scope: str | None = None,
        metadata_mark_skipped: bool = False,
    ) -> None:
        if not route_trace_enabled:
            return
        if fast_cached_direct_bound:
            pointer_signature_count = int(getattr(stats, "metadata_count", -1))
            pointer_signature_stable = True
        else:
            pointer_signature_stable = True
            pointer_signature_count = 0
            for metadata in iter_attention_metadata(
                getattr(forward_context, "attn_metadata", None)
            ):
                carriers = getattr(
                    metadata,
                    "mixed_page_resolver_replay_carriers",
                    None,
                )
                captured_signature = getattr(
                    metadata,
                    "mixed_page_resolver_captured_pointer_signature",
                    None,
                )
                if carriers is None or not hasattr(carriers, "pointer_signature"):
                    continue
                pointer_signature_count += 1
                replay_signature = tuple(carriers.pointer_signature())
                if (
                    captured_signature is not None
                    and tuple(captured_signature) != replay_signature
                ):
                    pointer_signature_stable = False
        from patches.fa3_native.install import append_fa3_route_trace

        launch_plan_trace = _compact_recent_launch_plan_trace_payload(
            controller=controller,
            page_size=int(getattr(stats, "page_size", 16) or 16),
        )
        rrp_visible_source_trace = _metadata_items_rrp_visible_source_fields(
            iter_attention_metadata(getattr(forward_context, "attn_metadata", None))
        )
        payload = {
            "event": "mixed_page_full_cudagraph_replay_refresh",
            "step_id": int(step_id),
            "graph_key": str(getattr(stats, "graph_key", graph_key)),
            "metadata_count": int(getattr(stats, "metadata_count", -1)),
            "updated_metadata_count": int(
                getattr(stats, "updated_metadata_count", -1)
            ),
            "carrier_update_rows": int(getattr(stats, "carrier_update_rows", -1)),
            "carrier_update_bytes": int(getattr(stats, "carrier_update_bytes", -1)),
            "carrier_update_kernel_count": int(
                getattr(stats, "carrier_update_kernel_count", -1)
            ),
            "ready_event_wait_count": int(
                getattr(
                    forward_context,
                    "mixed_page_resolver_replay_ready_event_wait_total",
                    ready_event_wait_count,
                )
            ),
            "carrier_pointer_signature_count": int(pointer_signature_count),
            "carrier_pointer_signature_stable": bool(pointer_signature_stable),
            "row_mode_distribution": dict(
                getattr(stats, "row_mode_distribution", {})
            ),
            "row_source_distribution": dict(
                getattr(stats, "row_source_distribution", {})
            ),
            **launch_plan_trace,
            **rrp_visible_source_trace,
            **_mixed_page_replay_stats_source_counter_payload(stats),
        }
        if fast_cached_direct_bound:
            payload["fast_cached_direct_bound"] = True
        if cache_scope is not None:
            payload["cache_scope"] = str(cache_scope)
        if metadata_mark_skipped:
            payload["metadata_mark_skipped"] = True
        append_fa3_route_trace(payload)

    if not rebound_for_graph_capacity:
        controller_graph_cache = getattr(
            controller,
            "_mixed_page_direct_bound_refresh_stats_by_graph_key",
            None,
        )
        if isinstance(controller_graph_cache, dict):
            cached_stats = controller_graph_cache.get(
                graph_direct_bound_cache_key
            ) or controller_graph_cache.get(metadata_direct_bound_cache_key)
            if cached_stats is not None:
                stats = _mixed_page_full_cudagraph_replay_stats_for_step(
                    cached_stats,
                    step_id=step_id,
                    graph_key=graph_key,
                    updated_metadata_count=0,
                )
                setattr(
                    controller,
                    "_mixed_page_full_cudagraph_last_replay_stats",
                    stats,
                )
                setattr(
                    forward_context,
                    "mixed_page_resolver_replay_last_step_id",
                    step_id,
                )
                setattr(
                    forward_context,
                    "mixed_page_resolver_replay_last_graph_key",
                    graph_key,
                )
                setattr(
                    forward_context,
                    "mixed_page_resolver_replay_last_stats",
                    stats,
                )
                if step_id >= 0:
                    setattr(
                        forward_context,
                        "mixed_page_resolver_replay_refresh_guard",
                        replay_guard,
                    )
                if profile_path:
                    try:
                        with open(profile_path, "a", encoding="utf-8") as fh:
                            fh.write(
                                json.dumps(
                                    {
                                        "event": "mixed_page_full_cudagraph_replay_refresh",
                                        "step_id": int(step_id),
                                        "graph_key": str(graph_key),
                                        "elapsed_us": 0.0,
                                        "fast_cached_direct_bound": True,
                                        "cache_scope": "controller_metadata_graph_key",
                                        "metadata_mark_skipped": True,
                                        "ready_event_wait_count": int(
                                            ready_event_wait_count
                                        ),
                                        "metadata_count": int(
                                            getattr(stats, "metadata_count", -1)
                                        ),
                                        "updated_metadata_count": int(
                                            getattr(
                                                stats,
                                                "updated_metadata_count",
                                                -1,
                                            )
                                        ),
                                        "carrier_update_rows": int(
                                            getattr(stats, "carrier_update_rows", -1)
                                        ),
                                        "carrier_update_bytes": int(
                                            getattr(stats, "carrier_update_bytes", -1)
                                        ),
                                        "carrier_update_kernel_count": int(
                                            getattr(
                                                stats,
                                                "carrier_update_kernel_count",
                                                -1,
                                            )
                                        ),
                                        **_mixed_page_replay_stats_source_counter_payload(
                                            stats
                                        ),
                                    },
                                    sort_keys=True,
                                    separators=(",", ":"),
                                )
                                + "\n"
                            )
                    except Exception:
                        pass
                _append_replay_refresh_route_trace(
                    stats,
                    fast_cached_direct_bound=True,
                    cache_scope="controller_metadata_graph_key",
                    metadata_mark_skipped=True,
                )
                return stats
        cached_key = getattr(
            forward_context,
            "mixed_page_resolver_direct_bound_refresh_cache_key",
            None,
        )
        cached_stats = getattr(
            forward_context,
            "mixed_page_resolver_direct_bound_refresh_stats",
            None,
        )
        if cached_key == metadata_direct_bound_cache_key and cached_stats is not None:
            stats = _mixed_page_full_cudagraph_replay_stats_for_step(
                cached_stats,
                step_id=step_id,
                graph_key=graph_key,
            )
            setattr(
                controller,
                "_mixed_page_full_cudagraph_last_replay_stats",
                stats,
            )
            setattr(
                forward_context,
                "mixed_page_resolver_replay_last_step_id",
                step_id,
            )
            setattr(
                forward_context,
                "mixed_page_resolver_replay_last_graph_key",
                graph_key,
            )
            setattr(
                forward_context,
                "mixed_page_resolver_replay_last_stats",
                stats,
            )
            if step_id >= 0:
                setattr(
                    forward_context,
                    "mixed_page_resolver_replay_refresh_guard",
                    replay_guard,
                )
            if profile_path:
                try:
                    with open(profile_path, "a", encoding="utf-8") as fh:
                        fh.write(
                            json.dumps(
                                {
                                    "event": "mixed_page_full_cudagraph_replay_refresh",
                                    "step_id": int(step_id),
                                    "graph_key": str(graph_key),
                                    "elapsed_us": 0.0,
                                    "fast_cached_direct_bound": True,
                                    "ready_event_wait_count": int(
                                        ready_event_wait_count
                                    ),
                                    "metadata_count": int(
                                        getattr(stats, "metadata_count", -1)
                                    ),
                                    "updated_metadata_count": int(
                                        getattr(stats, "updated_metadata_count", -1)
                                    ),
                                    "carrier_update_rows": int(
                                        getattr(stats, "carrier_update_rows", -1)
                                    ),
                                    "carrier_update_bytes": int(
                                        getattr(stats, "carrier_update_bytes", -1)
                                    ),
                                    "carrier_update_kernel_count": int(
                                        getattr(
                                            stats,
                                            "carrier_update_kernel_count",
                                            -1,
                                        )
                                    ),
                                    **_mixed_page_replay_stats_source_counter_payload(
                                        stats
                                    ),
                                },
                                sort_keys=True,
                                separators=(",", ":"),
                            )
                            + "\n"
                        )
                except Exception:
                    pass
            _append_replay_refresh_route_trace(
                stats,
                fast_cached_direct_bound=True,
                cache_scope="forward_context_metadata_key",
            )
            return stats

    profile_start_ns = time.perf_counter_ns() if profile_path else 0
    stats = refresh_mixed_page_carriers_for_forward_context(
        forward_context,
        step_id=step_id,
        graph_key=graph_key,
    )
    if profile_path:
        try:
            with open(profile_path, "a", encoding="utf-8") as fh:
                fh.write(
                    json.dumps(
                        {
                            "event": "mixed_page_full_cudagraph_replay_refresh",
                            "step_id": int(step_id),
                            "graph_key": str(graph_key),
                            "elapsed_us": float(
                                time.perf_counter_ns() - profile_start_ns
                            )
                            / 1000.0,
                            "metadata_count": int(
                                getattr(stats, "metadata_count", -1)
                            ),
                            "updated_metadata_count": int(
                                getattr(stats, "updated_metadata_count", -1)
                            ),
                            "carrier_update_rows": int(
                                getattr(stats, "carrier_update_rows", -1)
                            ),
                            "carrier_update_bytes": int(
                                getattr(stats, "carrier_update_bytes", -1)
                            ),
                            "carrier_update_kernel_count": int(
                                getattr(stats, "carrier_update_kernel_count", -1)
                            ),
                            "ready_event_wait_count": int(
                                getattr(
                                    forward_context,
                                    "mixed_page_resolver_replay_ready_event_wait_total",
                                    ready_event_wait_count,
                                )
                            ),
                            **_mixed_page_replay_stats_source_counter_payload(stats),
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
        except Exception:
            pass
    if (
        int(getattr(stats, "metadata_count", 0)) > 0
        and int(getattr(stats, "updated_metadata_count", -1))
        == int(getattr(stats, "metadata_count", 0))
        and int(getattr(stats, "carrier_update_rows", -1)) == 0
        and int(getattr(stats, "carrier_update_bytes", -1)) == 0
        and int(getattr(stats, "carrier_update_kernel_count", -1)) == 0
    ):
        setattr(
            forward_context,
            "mixed_page_resolver_direct_bound_refresh_cache_key",
            metadata_direct_bound_cache_key,
        )
        setattr(
            forward_context,
            "mixed_page_resolver_direct_bound_refresh_stats",
            stats,
        )
        controller_graph_cache = getattr(
            controller,
            "_mixed_page_direct_bound_refresh_stats_by_graph_key",
            None,
        )
        if not isinstance(controller_graph_cache, dict):
            controller_graph_cache = {}
            setattr(
                controller,
                "_mixed_page_direct_bound_refresh_stats_by_graph_key",
                controller_graph_cache,
            )
        controller_graph_cache[metadata_direct_bound_cache_key] = stats
        controller_graph_cache[graph_direct_bound_cache_key] = stats
    if step_id >= 0:
        setattr(
            forward_context,
            "mixed_page_resolver_replay_refresh_guard",
            replay_guard,
        )
    setattr(
        controller,
        "_mixed_page_full_cudagraph_last_replay_stats",
        stats,
    )
    _append_replay_refresh_route_trace(stats)
    return stats


def _prebound_rrp_current_row_modes(
    *,
    controller: object,
    live_batch_size: int,
    graph_batch_size: int,
) -> tuple[str, ...] | None:
    step_authority = getattr(controller, "step_authority", None)
    row_mode_by_row = getattr(step_authority, "row_mode_by_row", None)
    if isinstance(row_mode_by_row, torch.Tensor):
        return None
    if row_mode_by_row is None:
        return None
    try:
        live_modes = tuple(
            int(value) for value in tuple(row_mode_by_row)[: int(live_batch_size)]
        )
    except Exception:
        return None
    if len(live_modes) < int(live_batch_size):
        return None

    from patches.sparse_constants import _ROW_MODE_COMPACT

    modes = tuple(
        "compact" if int(value) == int(_ROW_MODE_COMPACT) else "native"
        for value in live_modes
    )
    effective_batch_size = max(int(live_batch_size), int(graph_batch_size))
    if effective_batch_size > int(live_batch_size):
        modes = modes + ("unset",) * (effective_batch_size - int(live_batch_size))
    return modes


def _prebound_rrp_static_stats_values_with_carrier_publish(
    static_values: tuple[object, ...],
    carrier_publish: object | None,
) -> tuple[object, ...]:
    """Reflect an in-place dirty-row carrier republish in the otherwise-static
    prebound replay stats. Field order matches
    ``_PREBOUND_RRP_STATIC_STATS_FIELD_NAMES``: index 1 updated_metadata_count,
    2 carrier_update_rows, 4 carrier_update_kernel_count. Same-page steps publish
    zero dirty rows and leave the static (zero-update) values untouched.
    """
    if carrier_publish is None:
        return static_values
    delta_rows = tuple(getattr(carrier_publish, "delta_rows", ()) or ())
    dirty = len(delta_rows)
    if dirty <= 0:
        return static_values
    values = list(static_values)
    values[1] = int(dirty)
    values[2] = int(dirty)
    try:
        existing_kernels = int(values[4])
    except (TypeError, ValueError):
        existing_kernels = 0
    values[4] = max(existing_kernels, 1)
    return tuple(values)


def _prebound_rrp_row_modes_require_rebind(
    *,
    controller: object,
    live_batch_size: int,
    graph_batch_size: int,
    num_kv_heads: int,
) -> bool:
    state = getattr(controller, "_resolved_row_ptr_graph_binding_state", None)
    if not isinstance(state, dict):
        return False
    current_modes = _prebound_rrp_current_row_modes(
        controller=controller,
        live_batch_size=int(live_batch_size),
        graph_batch_size=int(graph_batch_size),
    )
    if current_modes is None:
        return False

    state_modes = state.get("row_mode_by_row")
    if isinstance(state_modes, (list, tuple)):
        state_tuple = tuple(str(value) for value in state_modes)
        if len(state_tuple) >= len(current_modes):
            if state_tuple[: len(current_modes)] != current_modes:
                return True

    state_distribution = state.get("row_mode_distribution")
    if not isinstance(state_distribution, dict):
        return False
    expected = {
        "compact": 0,
        "safe_full_recent": 0,
        "native": 0,
        "unset": 0,
    }
    heads = max(1, int(num_kv_heads))
    for mode in current_modes:
        if mode == "compact":
            expected["compact"] += heads
        elif mode == "unset":
            expected["unset"] += heads
        else:
            expected["native"] += heads
    return any(
        int(state_distribution.get(key, 0) or 0) != int(value)
        for key, value in expected.items()
    )


def _prebound_rrp_step_fast_identity(
    step_authority: object,
) -> tuple[int, int, int, int, int, int] | None:
    fast_identity = getattr(step_authority, "step_fast_identity", None)
    if isinstance(fast_identity, (list, tuple)) and len(fast_identity) >= 6:
        try:
            return (
                int(fast_identity[0]),
                int(fast_identity[1]),
                int(fast_identity[2]),
                int(fast_identity[3]),
                int(fast_identity[4]),
                int(fast_identity[5]),
            )
        except Exception:
            return None
    try:
        return (
            int(getattr(step_authority, "epoch", -1)),
            int(getattr(step_authority, "step_handle_id", -1)),
            int(getattr(step_authority, "step_handle_generation", -1)),
            int(getattr(step_authority, "step_identity_token")),
            int(getattr(step_authority, "req_set_hash")),
            int(getattr(step_authority, "row_phase_hash")),
        )
    except Exception:
        return None


def _prebound_rrp_graph_binding_is_current_for_step(
    *,
    controller: object,
    step_authority: object,
    live_batch_size: int,
    expected_batch_size: int,
    state: dict[str, object] | None = None,
) -> bool:
    if state is None:
        state = _resolved_row_ptr_graph_binding_state_for_replay(controller)
    if state is None:
        return False
    try:
        if int(state.get("bind_live_batch_size", -1)) != int(live_batch_size):
            return False
        if int(state.get("bind_effective_batch_size", -1)) != int(expected_batch_size):
            return False
    except Exception:
        return False
    private_fast_identity = getattr(
        controller,
        "_resolved_row_ptr_step_fast_identity",
        None,
    )
    if isinstance(private_fast_identity, (list, tuple)) and len(private_fast_identity) >= 8:
        step_fast_identity = _prebound_rrp_step_fast_identity(step_authority)
        if step_fast_identity is not None:
            try:
                private_live_batch_size = int(
                    getattr(
                        controller,
                        "_resolved_row_ptr_step_live_batch_size",
                        -1,
                    )
                )
                private_effective_batch_size = int(
                    getattr(
                        controller,
                        "_resolved_row_ptr_step_effective_batch_size",
                        -1,
                    )
                )
                if (
                    private_live_batch_size == int(live_batch_size)
                    and private_effective_batch_size == int(expected_batch_size)
                    and (
                        int(private_fast_identity[0]),
                        int(private_fast_identity[1]),
                        int(private_fast_identity[2]),
                        int(private_fast_identity[3]),
                        int(private_fast_identity[4]),
                        int(private_fast_identity[5]),
                    )
                    == step_fast_identity
                    and int(private_fast_identity[6]) == int(live_batch_size)
                    and int(private_fast_identity[7]) == int(expected_batch_size)
                ):
                    return True
            except Exception:
                return False
    fast_identity = state.get("bind_step_fast_identity", None)
    if isinstance(fast_identity, (list, tuple)) and len(fast_identity) >= 8:
        step_fast_identity = _prebound_rrp_step_fast_identity(step_authority)
        if step_fast_identity is not None:
            try:
                identity_matches = (
                    (
                        int(fast_identity[0]),
                        int(fast_identity[1]),
                        int(fast_identity[2]),
                        int(fast_identity[3]),
                        int(fast_identity[4]),
                        int(fast_identity[5]),
                    )
                    == step_fast_identity
                    and int(fast_identity[6]) == int(live_batch_size)
                    and int(fast_identity[7]) == int(expected_batch_size)
                )
                if not bool(identity_matches):
                    return False
                return True
            except Exception:
                return False
    identity = state.get("bind_step_identity", ())
    if not isinstance(identity, (list, tuple)) or len(identity) < 8:
        return False
    live_batch_size_i = int(live_batch_size)

    def _values_match(
        stored: object,
        values: object,
        *,
        count: int,
        string_values: bool = False,
    ) -> bool:
        if not isinstance(stored, (list, tuple)):
            return False
        if values is None or isinstance(values, torch.Tensor):
            return False
        try:
            if len(values) < int(count):  # type: ignore[arg-type]
                return False
            for index in range(int(count)):
                if string_values:
                    if str(values[index]) != str(stored[index]):  # type: ignore[index]
                        return False
                elif int(values[index]) != int(stored[index]):  # type: ignore[index]
                    return False
        except Exception:
            return False
        return True

    try:
        if int(identity[0]) != int(getattr(step_authority, "epoch", -1)):
            return False
        if int(identity[1]) != int(getattr(step_authority, "step_handle_id", -1)):
            return False
        if int(identity[2]) != int(
            getattr(step_authority, "step_handle_generation", -1)
        ):
            return False
    except Exception:
        return False
    identity_matches = (
        _values_match(
            identity[3],
            getattr(step_authority, "req_ids", None),
            count=live_batch_size_i,
            string_values=True,
        )
        and _values_match(
            identity[4],
            getattr(step_authority, "q_lens_by_row", None),
            count=live_batch_size_i,
        )
        and _values_match(
            identity[5],
            getattr(step_authority, "q_start_loc", None),
            count=live_batch_size_i + 1,
        )
        and _values_match(
            identity[6],
            getattr(step_authority, "row_mode_by_row", None),
            count=live_batch_size_i,
        )
        and _values_match(
            identity[7],
            getattr(step_authority, "slot_by_row", None),
            count=live_batch_size_i,
        )
    )
    if not bool(identity_matches):
        return False
    return True


def _prebound_rrp_graph_binding_step_markers_match(
    *,
    controller: object,
    step_authority: object,
) -> bool:
    state = getattr(controller, "_resolved_row_ptr_graph_binding_state", None)
    if not isinstance(state, dict):
        return False
    identity = state.get("bind_step_identity", ())
    if not isinstance(identity, (list, tuple)) or len(identity) < 3:
        return False
    try:
        return (
            int(identity[0]) == int(getattr(step_authority, "epoch", -1))
            and int(identity[1]) == int(getattr(step_authority, "step_handle_id", -1))
            and int(identity[2])
            == int(getattr(step_authority, "step_handle_generation", -1))
        )
    except Exception:
        return False


def _prebound_rrp_graph_state_static_cache_key(
    state: dict[str, object],
) -> tuple[object, ...] | None:
    cached = state.get("_prebound_rrp_graph_state_static_cache_key")
    if isinstance(cached, tuple):
        return cached

    pointer_signature = state.get("pointer_signature")
    arena_key = state.get("arena_key")
    q_layout_key = state.get("q_layout_key")
    if pointer_signature is None or arena_key is None or q_layout_key is None:
        return None
    try:
        metadata_count = int(state.get("metadata_count", 0))
    except Exception:
        return None
    if metadata_count <= 0:
        return None
    row_modes = state.get("row_mode_by_row", ())
    state_payload = state if isinstance(state, dict) else {}
    row_mode_distribution = state_payload.get("row_mode_distribution", {})
    row_source_distribution = state_payload.get("row_source_distribution", {})
    key = (
        tuple(arena_key) if isinstance(arena_key, (list, tuple)) else arena_key,
        str(q_layout_key),
        tuple(pointer_signature)
        if isinstance(pointer_signature, (list, tuple))
        else pointer_signature,
        int(metadata_count),
        tuple(row_modes) if isinstance(row_modes, (list, tuple)) else (),
        tuple(sorted(dict(row_mode_distribution).items()))
        if isinstance(row_mode_distribution, dict)
        else (),
        tuple(sorted(dict(row_source_distribution).items()))
        if isinstance(row_source_distribution, dict)
        else (),
    )
    state["_prebound_rrp_graph_state_static_cache_key"] = key
    return key


def _attach_resolved_row_ptr_replay_metadata_if_needed(
    metadata_items: tuple[object, ...],
    *,
    binding: object,
    attach_fn: object,
) -> None:
    if not callable(attach_fn):
        return
    for metadata in metadata_items:
        if (
            getattr(metadata, "mixed_page_resolver_replay_metadata_binding", None)
            is binding
        ):
            continue
        attach_fn(attn_metadata=metadata, binding=binding)


def _maybe_rebind_rrp_for_full_graph_replay(
    *,
    controller: object,
    forward_context: object,
    metadata_items: tuple[object, ...],
) -> bool:
    """Refresh graph-stable RRP row tables on shape or row-mode changes."""

    def _append_rebind_probe(reason: str, **fields: object) -> None:
        if not _fa3_route_trace_enabled():
            return
        try:
            from patches.fa3_native.install import append_fa3_route_trace

            sparse_state = None
            capacity_key = None
            delta = getattr(controller, "_decode_runtime_delta", None)
            step_authority_for_probe = getattr(controller, "step_authority", None)

            def _tuple_head(value: object, limit: int = 4) -> tuple[object, ...]:
                if value is None or isinstance(value, torch.Tensor):
                    return tuple()
                try:
                    return tuple(value)[: int(limit)]
                except Exception:
                    return tuple()

            def _int_value(value: object, default: int = -1) -> int:
                try:
                    return int(value)
                except Exception:
                    return int(default)

            visible_cpu = getattr(
                sparse_state,
                "_visible_effective_k_len_cpu",
                tuple(),
            )
            launch_cpu = getattr(
                sparse_state,
                "_launch_effective_k_len_cpu",
                tuple(),
            )
            payload = {
                "event": "full_graph_rrp_rebind_probe",
                "reason": reason,
                "step_authority_epoch": _int_value(
                    getattr(step_authority_for_probe, "epoch", -1)
                ),
                "controller_capacity_key_present": capacity_key is not None,
                "controller_capacity_key_repr": repr(capacity_key),
                "sparse_state_present": sparse_state is not None,
                "sparse_state_step_id": _int_value(
                    getattr(sparse_state, "_step_id", -1)
                ),
                "sparse_state_visible_step_id": _int_value(
                    getattr(sparse_state, "_visible_step_id", -1)
                ),
                "sparse_state_last_update_source": str(
                    getattr(sparse_state, "_last_update_source", "")
                ),
                "sparse_state_last_failure_reason": str(
                    getattr(sparse_state, "last_failure_reason", "") or ""
                ),
                "sparse_state_req_count": len(
                    getattr(sparse_state, "_req_ids", tuple())
                )
                if sparse_state is not None
                else 0,
                "sparse_state_q_lens_head": _tuple_head(
                    getattr(sparse_state, "_q_lens_by_row", tuple())
                ),
                "sparse_state_row_mode_head": _tuple_head(
                    getattr(sparse_state, "_row_mode_class", tuple())
                ),
                "sparse_state_visible_cpu_head": _tuple_head(visible_cpu),
                "sparse_state_launch_cpu_head": _tuple_head(launch_cpu),
                "delta_step_id": _int_value(getattr(delta, "step_id", -1)),
                "delta_row_effective_head": _tuple_head(
                    getattr(delta, "row_effective_k_by_row", tuple())
                ),
                "delta_launch_effective_head": _tuple_head(
                    getattr(delta, "launch_effective_k_by_row", tuple())
                ),
                "step_authority_q_lens_head": _tuple_head(
                    getattr(step_authority_for_probe, "q_lens_by_row", tuple())
                ),
                "step_authority_row_mode_head": _tuple_head(
                    getattr(step_authority_for_probe, "row_mode_by_row", tuple())
                ),
            }
            payload.update(fields)
            append_fa3_route_trace(payload)
        except Exception:
            pass

    step_authority = getattr(controller, "step_authority", None)
    if step_authority is None:
        _append_rebind_probe("missing_step_authority")
        return False
    try:
        live_batch_size = int(getattr(step_authority, "batch_size", 0))
    except Exception:
        _append_rebind_probe("invalid_live_batch_size")
        return False
    if live_batch_size <= 0:
        _append_rebind_probe("empty_live_batch", live_batch_size=live_batch_size)
        return False
    batch_descriptor = getattr(forward_context, "batch_descriptor", None)
    try:
        graph_batch_size = int(getattr(batch_descriptor, "num_reqs", 0) or 0)
    except Exception:
        graph_batch_size = 0
    if graph_batch_size > 0:
        setattr(
            controller,
            "_resolved_row_ptr_full_cudagraph_graph_batch_size",
            int(graph_batch_size),
        )
    if not metadata_items:
        _append_rebind_probe(
            "missing_metadata_items",
            live_batch_size=live_batch_size,
            graph_batch_size=graph_batch_size,
        )
        return False
    current_key = getattr(controller, "_resolved_row_ptr_arena_key", None)
    if not (isinstance(current_key, tuple) and len(current_key) >= 4):
        _append_rebind_probe(
            "missing_or_invalid_arena_key",
            live_batch_size=live_batch_size,
            graph_batch_size=graph_batch_size,
            has_current_key=current_key is not None,
            current_key_repr=repr(current_key),
            metadata_items_count=len(metadata_items),
        )
        return False
    try:
        num_kv_heads = int(current_key[1])
        block_size = int(current_key[2])
    except Exception:
        _append_rebind_probe(
            "invalid_arena_key_fields",
            live_batch_size=live_batch_size,
            graph_batch_size=graph_batch_size,
            current_key_repr=repr(current_key),
            metadata_items_count=len(metadata_items),
        )
        return False
    replay_arena = getattr(controller, "_resolved_row_ptr_replay_arena", None)
    row_table = getattr(replay_arena, "row_table_i32", None)
    device = getattr(row_table, "device", None)
    if device is None:
        _append_rebind_probe(
            "missing_replay_arena_row_table_device",
            live_batch_size=live_batch_size,
            graph_batch_size=graph_batch_size,
            current_key_repr=repr(current_key),
            has_replay_arena=replay_arena is not None,
            has_row_table=row_table is not None,
            metadata_items_count=len(metadata_items),
        )
        return False
    first_metadata = metadata_items[0]
    expected_batch_size = max(int(live_batch_size), int(graph_batch_size))
    existing_binding = getattr(
        controller,
        "_resolved_row_ptr_replay_metadata_binding",
        None,
    )
    step_markers_match = _prebound_rrp_graph_binding_step_markers_match(
        controller=controller,
        step_authority=step_authority,
    )
    graph_binding_state = _resolved_row_ptr_graph_binding_state_for_replay(controller)
    binding_current_for_step = _prebound_rrp_graph_binding_is_current_for_step(
        controller=controller,
        step_authority=step_authority,
        live_batch_size=live_batch_size,
        expected_batch_size=expected_batch_size,
        state=graph_binding_state,
    )
    if binding_current_for_step:
        from patches.fa_sparse_runtime.resolved_row_ptr_arena import (
            ResolvedRowPtrReplayMetadataBinding,
        )

        if isinstance(existing_binding, ResolvedRowPtrReplayMetadataBinding):
            result = bool(graph_batch_size > live_batch_size)
            _append_rebind_probe(
                "reused_current_binding",
                live_batch_size=live_batch_size,
                graph_batch_size=graph_batch_size,
                expected_batch_size=expected_batch_size,
                metadata_items_count=len(metadata_items),
                step_markers_match=step_markers_match,
                graph_binding_state=graph_binding_state,
                result=result,
            )
            return result

    graph_capacity_or_mode_rebind = graph_batch_size > live_batch_size
    if not graph_capacity_or_mode_rebind:
        graph_capacity_or_mode_rebind = _prebound_rrp_row_modes_require_rebind(
            controller=controller,
            live_batch_size=live_batch_size,
            graph_batch_size=graph_batch_size,
            num_kv_heads=num_kv_heads,
        )

    if existing_binding is not None:
        descriptor = getattr(existing_binding, "descriptor", None)
        try:
            existing_batch = int(getattr(descriptor, "batch", -1))
        except Exception:
            existing_batch = -1
        if existing_batch == int(expected_batch_size):
            from patches.decode_runtime.metadata_builder import (
                _try_attach_same_page_resolved_row_ptr_replay_metadata,
            )
            from patches.fa_sparse_runtime.resolved_row_ptr_arena import (
                ResolvedRowPtrReplayMetadataBinding,
                attach_resolved_row_ptr_replay_metadata,
            )

            if isinstance(existing_binding, ResolvedRowPtrReplayMetadataBinding):
                same_page_update = _try_attach_same_page_resolved_row_ptr_replay_metadata(
                    controller,
                    attn_metadata=first_metadata,
                    step_authority=step_authority,
                    batch_size=expected_batch_size,
                    block_size=block_size,
                    num_kv_heads=num_kv_heads,
                    device=device,
                    live_batch_size=live_batch_size,
                    active_arena_row_indices=tuple(range(int(live_batch_size))),
                )
                if same_page_update is not None:
                    binding = getattr(
                        controller,
                        "_resolved_row_ptr_replay_metadata_binding",
                        existing_binding,
                    )
                    if isinstance(binding, ResolvedRowPtrReplayMetadataBinding):
                        _attach_resolved_row_ptr_replay_metadata_if_needed(
                            metadata_items[1:],
                            binding=binding,
                            attach_fn=attach_resolved_row_ptr_replay_metadata,
                        )
                        _append_rebind_probe(
                            "same_page_binding_refreshed",
                            live_batch_size=live_batch_size,
                            graph_batch_size=graph_batch_size,
                            expected_batch_size=expected_batch_size,
                            metadata_items_count=len(metadata_items),
                            existing_batch=existing_batch,
                            step_markers_match=step_markers_match,
                            graph_capacity_or_mode_rebind=graph_capacity_or_mode_rebind,
                        )
                        return False
                    _attach_resolved_row_ptr_replay_metadata_if_needed(
                        metadata_items[1:],
                        binding=existing_binding,
                        attach_fn=attach_resolved_row_ptr_replay_metadata,
                    )
                    _append_rebind_probe(
                        "same_page_update_returned_non_binding",
                        live_batch_size=live_batch_size,
                        graph_batch_size=graph_batch_size,
                        expected_batch_size=expected_batch_size,
                        metadata_items_count=len(metadata_items),
                        existing_batch=existing_batch,
                        step_markers_match=step_markers_match,
                        graph_capacity_or_mode_rebind=graph_capacity_or_mode_rebind,
                    )
                    return False

    if graph_batch_size > live_batch_size:
        setattr(
            first_metadata,
            "_resolved_row_ptr_graph_batch_size_override",
            int(graph_batch_size),
        )
    from patches.decode_runtime.metadata_builder import (
        _maybe_bind_resolved_row_ptr_replay_metadata,
    )
    from patches.fa_sparse_runtime.resolved_row_ptr_arena import (
        ResolvedRowPtrReplayMetadataBinding,
        attach_resolved_row_ptr_replay_metadata,
    )

    _maybe_bind_resolved_row_ptr_replay_metadata(
        controller,
        attn_metadata=first_metadata,
        batch_size=live_batch_size,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        device=device,
    )
    binding = getattr(
        controller,
        "_resolved_row_ptr_replay_metadata_binding",
        None,
    )
    if not isinstance(binding, ResolvedRowPtrReplayMetadataBinding):
        _append_rebind_probe(
            "metadata_builder_did_not_bind_rrp",
            live_batch_size=live_batch_size,
            graph_batch_size=graph_batch_size,
            expected_batch_size=expected_batch_size,
            metadata_items_count=len(metadata_items),
            step_markers_match=step_markers_match,
            graph_capacity_or_mode_rebind=graph_capacity_or_mode_rebind,
            has_existing_binding=existing_binding is not None,
            sparse_native_capacity_key_present=False,
            **_metadata_rrp_visible_source_fields(first_metadata),
            route_family_after_bind=_mixed_page_metadata_route_family(first_metadata),
        )
        return False
    if int(binding.descriptor.batch) != int(expected_batch_size):
        _append_rebind_probe(
            "metadata_builder_bound_wrong_batch",
            live_batch_size=live_batch_size,
            graph_batch_size=graph_batch_size,
            expected_batch_size=expected_batch_size,
            metadata_items_count=len(metadata_items),
            binding_batch=int(binding.descriptor.batch),
            step_markers_match=step_markers_match,
            graph_capacity_or_mode_rebind=graph_capacity_or_mode_rebind,
            **_metadata_rrp_visible_source_fields(first_metadata),
            route_family_after_bind=_mixed_page_metadata_route_family(first_metadata),
        )
        return False
    _attach_resolved_row_ptr_replay_metadata_if_needed(
        metadata_items[1:],
        binding=binding,
        attach_fn=attach_resolved_row_ptr_replay_metadata,
    )
    result = bool(graph_capacity_or_mode_rebind)
    _append_rebind_probe(
        "metadata_builder_bound_rrp",
        live_batch_size=live_batch_size,
        graph_batch_size=graph_batch_size,
        expected_batch_size=expected_batch_size,
        metadata_items_count=len(metadata_items),
        binding_batch=int(binding.descriptor.batch),
        step_markers_match=step_markers_match,
        graph_capacity_or_mode_rebind=graph_capacity_or_mode_rebind,
        result=result,
        **_metadata_rrp_visible_source_fields(first_metadata),
        route_family_after_bind=_mixed_page_metadata_route_family(first_metadata),
    )
    return result

def _mixed_page_metadata_route_family(metadata: object) -> str:
    from patches.fa3_native.mixed_page_graph_descriptor import PageResolverKind

    descriptor = getattr(metadata, "mixed_page_resolver_descriptor", None)
    resolver_kind = getattr(descriptor, "resolver_kind", None)
    try:
        resolver_kind_i = int(resolver_kind)
    except Exception:
        resolver_kind_i = -1
    carriers = getattr(
        metadata,
        "mixed_page_resolver_replay_carriers",
        None,
    )
    if resolver_kind_i == int(PageResolverKind.RESOLVED_ROW_PTR):
        return "resolved_row_ptr"
    if carriers is not None and any(
        getattr(carriers, name, None) is not None
        for name in (
            "resolved_page_table_row_ptr_u64",
            "resolved_page_table_affine_i32",
            "resolved_page_table_affine_base",
            "resolved_page_table_affine_stride",
            "resolved_page_table_affine_segment_pages",
            "resolved_page_table_affine_second_base",
            "resolved_page_table_affine_second_stride",
            "resolved_page_table_affine_batch_stride",
            "resolved_page_table_affine_head_stride",
            "resolved_page_table_affine_direct",
            "resolved_page_table_affine_cols",
        )
    ):
        return "resolved_row_ptr"
    if resolver_kind_i == int(PageResolverKind.SELECTED_TABLE):
        return "legacy_selected_table"
    if carriers is not None and (
        getattr(carriers, "selected_page_table_i32", None) is not None
        or getattr(carriers, "row_consume_mode_i32", None) is not None
    ):
        return "legacy_selected_table"
    return "native_or_capture"


def _iter_attention_metadata_lazily(attn_metadata: object):
    if attn_metadata is None:
        return
    if isinstance(attn_metadata, dict):
        for value in attn_metadata.values():
            yield from _iter_attention_metadata_lazily(value)
        return
    if isinstance(attn_metadata, (list, tuple)):
        for value in attn_metadata:
            yield from _iter_attention_metadata_lazily(value)
        return
    yield attn_metadata



def _mixed_page_graph_entry_batch_descriptor_repr(
    entry: object | None,
    batch_descriptor: object,
) -> str:
    if entry is not None:
        cached = getattr(entry, "_sfi_mixed_page_batch_descriptor_repr", None)
        if isinstance(cached, str):
            return cached
    value = repr(batch_descriptor)
    if entry is not None:
        try:
            setattr(entry, "_sfi_mixed_page_batch_descriptor_repr", value)
        except Exception:
            pass
    return value


def _mixed_page_full_graph_key_for_entry(
    entry: object | None,
    batch_descriptor: object,
) -> str:
    if entry is not None:
        cached = getattr(entry, "_sfi_mixed_page_full_graph_key", None)
        if isinstance(cached, str) and cached:
            return cached
    descriptor_repr = _mixed_page_graph_entry_batch_descriptor_repr(
        entry,
        batch_descriptor,
    )
    value = f"full:{descriptor_repr}"
    if entry is not None:
        try:
            setattr(entry, "_sfi_mixed_page_full_graph_key", value)
        except Exception:
            pass
    return value

def _mixed_page_full_cudagraph_route_family_and_metadata_items(
    forward_context: object,
    *,
    controller: object | None = None,
) -> tuple[str, tuple[object, ...]]:
    """Classify the FULL-graph route and keep the metadata scan result."""
    owner_state = (
        _prebound_rrp_current_graph_state_for_forward_context(
            controller=controller,
            forward_context=forward_context,
        )
        if controller is not None
        else None
    )
    if owner_state is not None:
        return "resolved_row_ptr", ()
    metadata_items = tuple(
        _iter_attention_metadata_lazily(
            getattr(forward_context, "attn_metadata", None)
        )
    )
    if not metadata_items:
        return "unknown", ()
    route_family = "native_or_capture"
    for metadata in metadata_items:
        metadata_route_family = _mixed_page_metadata_route_family(metadata)
        if metadata_route_family != "native_or_capture":
            route_family = metadata_route_family
            break
    return route_family, metadata_items


def _mixed_page_full_cudagraph_route_family(
    forward_context: object,
    *,
    controller: object | None = None,
) -> str:
    """Classify the attention route family captured by a FULL graph entry.

    vLLM keys FULL graphs by batch descriptor only. During one-shot bootstrap a
    mixed prefill/decode step can have the same descriptor as later decode-only
    replay, so the guard only distinguishes graph-body route families. Compact
    vs mixed row distribution is carrier data and may change safely before replay.
    """
    route_family, _metadata_items = (
        _mixed_page_full_cudagraph_route_family_and_metadata_items(
            forward_context,
            controller=controller,
        )
    )
    return route_family


_MIXED_PAGE_FULL_CUDAGRAPH_ROUTE_FAMILIES = frozenset(
    {
        "resolved_row_ptr",
        "legacy_selected_table",
        "native_or_capture",
    }
)


def _record_mixed_page_actual_route_family(
    controller: object | None,
    route_family: str,
) -> None:
    if controller is None:
        return
    if route_family not in _MIXED_PAGE_FULL_CUDAGRAPH_ROUTE_FAMILIES:
        return
    setattr(
        controller,
        "_sfi_mixed_page_last_actual_route_family",
        {
            "step_id": _mixed_page_full_cudagraph_profile_step_id(controller),
            "route_family": str(route_family),
        },
    )


def _mixed_page_actual_route_family_for_capture_tag(
    *,
    controller: object | None,
    default_route_family: str,
) -> str:
    record = (
        getattr(controller, "_sfi_mixed_page_last_actual_route_family", None)
        if controller is not None
        else None
    )
    if isinstance(record, dict):
        try:
            record_step_id = int(record.get("step_id", -1))
        except Exception:
            record_step_id = -2
        route_family = str(record.get("route_family", ""))
        if (
            record_step_id == _mixed_page_full_cudagraph_profile_step_id(controller)
            and route_family in _MIXED_PAGE_FULL_CUDAGRAPH_ROUTE_FAMILIES
        ):
            return route_family
    if default_route_family in _MIXED_PAGE_FULL_CUDAGRAPH_ROUTE_FAMILIES:
        return str(default_route_family)
    return "native_or_capture"


def _mixed_page_full_cudagraph_route_family_mismatch(
    *,
    captured_route_family: object,
    current_route_family: str,
) -> bool:
    if current_route_family == "unknown":
        return False
    if not isinstance(captured_route_family, str):
        return False
    return captured_route_family != current_route_family


def _resolved_row_ptr_graph_binding_state_for_replay(
    controller: object | None,
) -> dict[str, object] | None:
    if controller is None:
        return None
    state = getattr(controller, "_resolved_row_ptr_graph_binding_state", None)
    if not isinstance(state, dict):
        return None
    if state.get("route_family") != "resolved_row_ptr":
        return None
    if state.get("arena_key") != getattr(controller, "_resolved_row_ptr_arena_key", None):
        return None

    binding = getattr(controller, "_resolved_row_ptr_replay_metadata_binding", None)
    if binding is None:
        return None
    pointer_signature = state.get("pointer_signature")
    if pointer_signature is None:
        return None
    if tuple(pointer_signature) != tuple(getattr(binding, "pointer_signature", ())):
        return None
    q_layout_key = state.get("q_layout_key")
    if q_layout_key is None:
        return None
    descriptor = getattr(binding, "descriptor", None)
    if str(getattr(descriptor, "q_layout_key", "")) != str(q_layout_key):
        return None
    try:
        if int(state.get("metadata_count", 0)) <= 0:
            return None
    except Exception:
        return None
    return state


def _mixed_page_full_cudagraph_prebound_rrp_graph_state(
    *,
    controller: object | None,
    captured_route_family: object | None,
    current_route_family: object | None,
) -> dict[str, object] | None:
    if captured_route_family != "resolved_row_ptr":
        return None
    if current_route_family != "resolved_row_ptr":
        return None
    return _resolved_row_ptr_graph_binding_state_for_replay(controller)




def _raise_missing_prebound_rrp_graph_state(
    *,
    captured_route_family: object | None,
    current_route_family: object | None,
) -> None:
    if current_route_family != "resolved_row_ptr":
        return
    raise RuntimeError(
        "resolved-row-ptr FULL CUDA graph replay requires prebound RRP graph state; "
        f"captured_route_family={captured_route_family!r}"
    )


def _prebound_rrp_replay_cache_key(
    *,
    state: dict[str, object],
    graph_key: str,
) -> tuple[object, ...] | None:
    static_key = _prebound_rrp_graph_state_static_cache_key(state)
    if static_key is None:
        return None
    try:
        ready_event_generation = int(state.get("ready_event_generation", -1))
    except Exception:
        return None
    if ready_event_generation < 0:
        return None
    return (
        str(graph_key),
        static_key,
        int(ready_event_generation),
    )


def _prebound_rrp_static_cache_key(
    *,
    state: dict[str, object],
    graph_key: str,
) -> tuple[object, ...] | None:
    static_key = _prebound_rrp_graph_state_static_cache_key(state)
    if static_key is None:
        return None
    return (
        str(graph_key),
        static_key,
    )


def _prebound_rrp_static_stats_fields(
    state: dict[str, object],
) -> dict[str, object]:
    fields = state.get("source_counter_missing_fields", ())
    if isinstance(fields, str):
        missing_fields = (fields,) if fields else ()
    elif isinstance(fields, (list, tuple)):
        missing_fields = tuple(str(field) for field in fields if str(field))
    else:
        missing_fields = ()
    return {
        "metadata_count": int(state.get("metadata_count", 0)),
        "updated_metadata_count": 0,
        "carrier_update_rows": 0,
        "carrier_update_bytes": 0,
        "carrier_update_kernel_count": 0,
        "row_mode_distribution": dict(state.get("row_mode_distribution", {})),
        "row_source_distribution": dict(state.get("row_source_distribution", {})),
        "source_counter_schema_version": int(
            state.get("source_counter_schema_version", -1)
        ),
        "expected_rows": int(state.get("expected_rows", -1)),
        "num_kv_heads": int(state.get("num_kv_heads", -1)),
        "source_counter_missing_fields": missing_fields,
    }


_PREBOUND_RRP_STATIC_STATS_FIELD_NAMES = (
    "metadata_count",
    "updated_metadata_count",
    "carrier_update_rows",
    "carrier_update_bytes",
    "carrier_update_kernel_count",
    "row_mode_distribution",
    "row_source_distribution",
    "source_counter_schema_version",
    "expected_rows",
    "num_kv_heads",
    "source_counter_missing_fields",
)


def _prebound_rrp_static_stats_values_from_fields(
    fields: dict[str, object],
) -> tuple[object, ...]:
    return tuple(fields[name] for name in _PREBOUND_RRP_STATIC_STATS_FIELD_NAMES)


def _prebound_rrp_static_stats_values_cached(
    *,
    controller: object,
    state: dict[str, object],
    graph_key: str,
) -> tuple[object, ...]:
    cache_key = _prebound_rrp_static_cache_key(
        state=state,
        graph_key=graph_key,
    )
    cache = getattr(controller, "_prebound_rrp_static_stats_cache", None)
    if cache_key is not None and isinstance(cache, dict):
        cached_key = cache.get("key")
        cached_values = cache.get("values")
        if (
            cached_key == cache_key
            and isinstance(cached_values, tuple)
            and len(cached_values) == len(_PREBOUND_RRP_STATIC_STATS_FIELD_NAMES)
        ):
            return cached_values
        cached_fields = cache.get("fields")
        if cached_key == cache_key and isinstance(cached_fields, dict):
            values = _prebound_rrp_static_stats_values_from_fields(cached_fields)
            cache["values"] = values
            return values

    fields = _prebound_rrp_static_stats_fields(state)
    values = _prebound_rrp_static_stats_values_from_fields(fields)
    if cache_key is not None:
        setattr(
            controller,
            "_prebound_rrp_static_stats_cache",
            {
                "key": cache_key,
                "fields": dict(fields),
                "values": values,
            },
        )
    return values




def _prebound_rrp_rebind_already_current(
    *,
    controller: object,
    forward_context: object,
    metadata_items: tuple[object, ...],
    state: dict[str, object] | None = None,
) -> bool:
    if not metadata_items:
        return False
    step_authority = getattr(controller, "step_authority", None)
    if step_authority is None:
        return False
    if state is None:
        state = _resolved_row_ptr_graph_binding_state_for_replay(controller)
    if state is None:
        return False
    try:
        live_batch_size = int(getattr(step_authority, "batch_size", 0))
    except Exception:
        return False
    if live_batch_size <= 0:
        return False
    batch_descriptor = getattr(forward_context, "batch_descriptor", None)
    try:
        graph_batch_size = int(getattr(batch_descriptor, "num_reqs", 0) or 0)
    except Exception:
        graph_batch_size = 0
    expected_batch_size = max(int(live_batch_size), int(graph_batch_size))
    if not _prebound_rrp_graph_binding_is_current_for_step(
        controller=controller,
        step_authority=step_authority,
        live_batch_size=live_batch_size,
        expected_batch_size=expected_batch_size,
        state=state,
    ):
        return False
    binding = getattr(
        controller,
        "_resolved_row_ptr_replay_metadata_binding",
        None,
    )
    if binding is None:
        return False
    return True


def _prebound_rrp_graph_state_is_current_for_forward_context(
    *,
    controller: object,
    forward_context: object,
    state: dict[str, object] | None,
) -> bool:
    if state is None:
        return False
    step_authority = getattr(controller, "step_authority", None)
    if step_authority is None:
        return False
    try:
        live_batch_size = int(getattr(step_authority, "batch_size", 0))
    except Exception:
        return False
    if live_batch_size <= 0:
        return False
    batch_descriptor = getattr(forward_context, "batch_descriptor", None)
    try:
        graph_batch_size = int(getattr(batch_descriptor, "num_reqs", 0) or 0)
    except Exception:
        graph_batch_size = 0
    expected_batch_size = max(int(live_batch_size), int(graph_batch_size))
    return _prebound_rrp_graph_binding_is_current_for_step(
        controller=controller,
        step_authority=step_authority,
        live_batch_size=live_batch_size,
        expected_batch_size=expected_batch_size,
        state=state,
    )


def _prebound_rrp_current_graph_state_for_forward_context(
    *,
    controller: object,
    forward_context: object,
) -> dict[str, object] | None:
    state = _resolved_row_ptr_graph_binding_state_for_replay(controller)
    if not _prebound_rrp_graph_state_is_current_for_forward_context(
        controller=controller,
        forward_context=forward_context,
        state=state,
    ):
        return None
    return state



def _prebound_rrp_cuda_stream_identity(stream: object | None) -> int:
    if stream is None:
        return -1
    raw = getattr(stream, "cuda_stream", None)
    if raw is not None:
        try:
            return int(raw)
        except (TypeError, ValueError):
            pass
    return int(id(stream))




def _prebound_rrp_current_cuda_stream_identity_raw() -> int | None:
    try:
        import torch

        # `_cuda_getCurrentStream` exposes Torch's stream id tuple, not the
        # cudaStream_t persisted by `Stream.cuda_stream` in producer state.
        get_current_raw_stream = getattr(
            getattr(torch, "_C", None),
            "_cuda_getCurrentRawStream",
            None,
        )
        current_device = getattr(torch.cuda, "current_device", None)
        if not callable(get_current_raw_stream) or not callable(current_device):
            return None
        return int(get_current_raw_stream(int(current_device())))
    except Exception:
        return None


def _wait_prebound_rrp_ready_event_from_graph_state(
    *,
    controller: object,
    forward_context: object,
    state: dict[str, object],
    profile_pre_timing: Optional[dict[str, float]] = None,
) -> object:
    from patches.fa_sparse_runtime.mixed_page_cudagraph_replay import (
        validate_resolved_row_ptr_ready_state,
    )

    ready_state = validate_resolved_row_ptr_ready_state(
        event=state.get("ready_event"),
        generation=state.get("ready_event_generation", -1),
        ready_stream=state.get("ready_event_stream", -1),
        same_stream_ordered=state.get("same_stream_ordered", False),
        require_published=True,
        context="mixed-page CUDA graph replay prebound RRP state",
    )
    event = ready_state.event
    same_stream_ordered = ready_state.same_stream_ordered
    generation = ready_state.generation
    ready_stream_id = ready_state.ready_stream_id

    _t_current_stream_ns = _full_cudagraph_pre_timing_start(profile_pre_timing)
    import torch

    wait_stream = None
    wait_stream_id = _prebound_rrp_current_cuda_stream_identity_raw()
    raw_stream_identity_available = wait_stream_id is not None
    if wait_stream_id is None:
        wait_stream = torch.cuda.current_stream()
        wait_stream_id = _prebound_rrp_cuda_stream_identity(wait_stream)
        # Producer state also stores `Stream.cuda_stream`, so this fallback
        # remains a proven raw-handle comparison on older Torch builds.
        wait_stream_raw = getattr(wait_stream, "cuda_stream", None)
        if wait_stream_raw is not None:
            try:
                wait_stream_id = int(wait_stream_raw)
                raw_stream_identity_available = True
            except (TypeError, ValueError):
                pass
    _full_cudagraph_pre_timing_add(
        profile_pre_timing,
        "pre_ready_event_current_stream_us",
        _t_current_stream_ns,
    )
    if same_stream_ordered and event is None:
        if (
            not raw_stream_identity_available
            or ready_stream_id < 0
            or ready_stream_id != wait_stream_id
        ):
            raise RuntimeError(
                "mixed-page CUDA graph replay cannot consume same-stream ordered "
                "RRP metadata without an equal proven raw CUDA stream"
            )
        wait_key_owner = state
        wait_key_ordered = True
    else:
        wait_key_owner = event
        wait_key_ordered = False
    wait_key = (
        wait_key_owner,
        int(generation),
        int(wait_stream_id),
        wait_key_ordered,
    )
    previous_wait_key = getattr(
        controller,
        "_prebound_rrp_ready_event_waited_key",
        None,
    )
    already_waited = bool(
        isinstance(previous_wait_key, tuple)
        and len(previous_wait_key) == 4
        and previous_wait_key[0] is wait_key_owner
        and previous_wait_key[1:] == wait_key[1:]
    )
    wait_count = 0
    if not already_waited:
        if event is not None and (
            not raw_stream_identity_available
            or ready_stream_id < 0
            or ready_stream_id != wait_stream_id
        ):
            _t_wait_call_ns = _full_cudagraph_pre_timing_start(profile_pre_timing)
            if wait_stream is None:
                wait_stream = torch.cuda.current_stream()
            wait_event = getattr(wait_stream, "wait_event", None)
            if not callable(wait_event):
                raise RuntimeError(
                    "mixed-page CUDA graph replay requires a stream with wait_event"
                )
            wait_event(event)
            wait_count = 1
            _full_cudagraph_pre_timing_add(
                profile_pre_timing,
                "pre_ready_event_wait_call_us",
                _t_wait_call_ns,
            )
        setattr(controller, "_prebound_rrp_ready_event_waited_key", wait_key)

    # The next metadata-owner update may elide its event only after this exact
    # state object and generation have reached a FULL replay consumer on a
    # proven stream.  Holding the object is intentional: an integer ``id`` can
    # be reused after state replacement and create an ABA false proof.
    if raw_stream_identity_available:
        setattr(
            controller,
            "_prebound_rrp_observed_ready_state",
            (state, int(generation), int(wait_stream_id)),
        )
    else:
        setattr(controller, "_prebound_rrp_observed_ready_state", None)
    _t_attr_publish_ns = _full_cudagraph_pre_timing_start(profile_pre_timing)
    previous_total = int(
        getattr(
            forward_context,
            "mixed_page_resolver_replay_ready_event_wait_total",
            0,
        )
    )
    setattr(
        forward_context,
        "mixed_page_resolver_replay_ready_event_wait_count",
        int(wait_count),
    )
    setattr(
        forward_context,
        "mixed_page_resolver_replay_ready_event_wait_total",
        previous_total + int(wait_count),
    )
    _full_cudagraph_pre_timing_add(
        profile_pre_timing,
        "pre_ready_event_attr_publish_us",
        _t_attr_publish_ns,
    )
    # Return the already-created validated state.  The caller stores its exact
    # generation in the existing immutable replay-stats object, avoiding a new
    # per-replay proof container on the hot path.
    return ready_state


def _record_prebound_rrp_replay_consumed_generation(
    *,
    controller: object,
    state: dict[str, object] | None,
    replay_proof: object,
) -> int:
    if state is None or not torch.cuda.is_available():
        return 0
    from patches.fa_sparse_runtime.rrp_replay_handshake import (
        record_rrp_replay_consumed_generation,
    )

    stream = torch.cuda.current_stream()
    stream_identity = _prebound_rrp_current_cuda_stream_identity_raw()
    if stream_identity is None:
        stream_identity = _prebound_rrp_cuda_stream_identity(stream)
    return int(
        record_rrp_replay_consumed_generation(
            controller,
            state=state,
            replay_proof=replay_proof,
            stream=stream,
            stream_identity=int(stream_identity),
            event_factory=lambda: torch.cuda.Event(
                blocking=False,
                enable_timing=False,
            ),
        )
    )


def _prebound_rrp_replay_consumed_by_nested_wrapper(
    *,
    controller: object,
    state: dict[str, object] | None,
    replay_proof: object,
    previous_consumed_state: object,
) -> bool:
    if state is None:
        raise RuntimeError("prebound RRP replay is missing its graph state")
    from patches.fa_sparse_runtime.rrp_replay_handshake import (
        replay_consumed_generation_was_recorded_since,
    )

    return bool(
        replay_consumed_generation_was_recorded_since(
            controller,
            state=state,
            replay_proof=replay_proof,
            previous_consumed_state=previous_consumed_state,
        )
    )


def _snapshot_prebound_rrp_replay_consumed_generation(
    *,
    controller: object,
    state: dict[str, object] | None,
) -> object:
    if state is None:
        raise RuntimeError("prebound RRP replay is missing its graph state")
    from patches.fa_sparse_runtime.rrp_replay_handshake import (
        snapshot_rrp_replay_consumed_generation,
    )

    return snapshot_rrp_replay_consumed_generation(
        controller,
        state=state,
    )


def _full_cudagraph_replay_wait_device(controller: object) -> torch.device | None:
    device = getattr(controller, "device", None)
    if isinstance(device, torch.device):
        return device
    if device is not None:
        try:
            parsed = torch.device(device)
            if parsed.type == "cuda":
                return parsed
        except (TypeError, RuntimeError):
            pass
    if not torch.cuda.is_available():
        return None
    return torch.device("cuda", torch.cuda.current_device())


def _release_refresh_producer_after_decode_if_pending(controller: object) -> bool:
    if not controller._refresh_producer_stream_release_pending:
        return False
    controller._release_refresh_producer_after_decode()
    return True


def _wait_one_shot_group_ready_before_prebound_rrp_full_graph_replay(
    *,
    controller: object,
    forward_context: object,
) -> int:
    """Order one-shot compact producer writes before prebound graph replay."""
    wait_group_ready = getattr(
        controller,
        "_wait_one_shot_group_ready_for_full_graph_replay",
        None,
    )
    if not callable(wait_group_ready):
        return 0
    cached_enabled = getattr(
        controller,
        "_sfi_one_shot_group_ready_graph_wait_enabled_cached",
        None,
    )
    if cached_enabled is None:
        enabled = getattr(controller, "_one_shot_group_ready_graph_wait_enabled", None)
        enabled_value = True
        if callable(enabled) and not bool(enabled()):
            enabled_value = False
        try:
            setattr(
                controller,
                "_sfi_one_shot_group_ready_graph_wait_enabled_cached",
                bool(enabled_value),
            )
        except Exception:
            pass
    else:
        enabled_value = bool(cached_enabled)
    if not bool(enabled_value):
        return 0
    device = _full_cudagraph_replay_wait_device(controller)
    if device is None or device.type != "cuda":
        return 0
    waited = int(wait_group_ready(device=device) or 0)
    if waited <= 0:
        return 0

    previous_count = int(
        getattr(
            forward_context,
            "mixed_page_resolver_replay_ready_event_wait_count",
            0,
        )
    )
    previous_total = int(
        getattr(
            forward_context,
            "mixed_page_resolver_replay_ready_event_wait_total",
            0,
        )
    )
    setattr(
        forward_context,
        "mixed_page_resolver_replay_ready_event_wait_count",
        previous_count + waited,
    )
    setattr(
        forward_context,
        "mixed_page_resolver_replay_ready_event_wait_total",
        previous_total + waited,
    )
    return int(waited)


def _full_cudagraph_current_step_authority(controller: object) -> object | None:
    step_ctx = getattr(controller, "step_context", None)
    step_authority = getattr(step_ctx, "step_authority", None)
    if step_authority is not None:
        return step_authority
    return getattr(controller, "step_authority", None)


def _full_cudagraph_selector_writer_submit_summary(
    controller: object,
) -> dict[str, object] | None:
    summary = getattr(
        controller,
        "_mixed_page_full_cudagraph_last_selector_writer_submit_summary",
        None,
    )
    if not isinstance(summary, dict):
        return None
    return dict(summary)


def _current_compact_consume_req_ids_for_full_cudagraph_replay(
    controller: object,
) -> set[str]:
    has_decode_consumer = getattr(controller, "_step_has_decode_consumer", None)
    if callable(has_decode_consumer) and not bool(has_decode_consumer()):
        return set()
    step_ctx = getattr(controller, "step_context", None)
    step_bound_meta = getattr(controller, "step_bound_meta", None)
    launch_plan = getattr(step_bound_meta, "compact_recent_launch_plan", None)
    req_ids_source = getattr(step_ctx, "req_ids", tuple())
    compact_valid_source = getattr(launch_plan, "compact_valid_tokens_cpu", None)
    if (
        launch_plan is not None
        and bool(getattr(launch_plan, "valid", False))
        and compact_valid_source is not None
        and not isinstance(req_ids_source, torch.Tensor)
        and not isinstance(compact_valid_source, torch.Tensor)
    ):
        req_ids = tuple(str(v) for v in tuple(req_ids_source or tuple()))
        compact_valid = tuple(int(v) for v in tuple(compact_valid_source or tuple()))
        if req_ids and compact_valid:
            return {
                str(req_ids[idx])
                for idx, valid_tokens in enumerate(compact_valid)
                if int(valid_tokens) > 0 and idx < len(req_ids)
            }
    step_authority = _full_cudagraph_current_step_authority(controller)
    req_ids_source = getattr(step_authority, "req_ids", tuple())
    use_compact_source = getattr(step_authority, "use_compact_by_row", tuple())
    if isinstance(req_ids_source, torch.Tensor) or isinstance(
        use_compact_source,
        torch.Tensor,
    ):
        return set()
    req_ids = tuple(str(v) for v in tuple(req_ids_source or tuple()))
    use_compact = tuple(bool(v) for v in tuple(use_compact_source or tuple()))
    return {
        str(req_ids[idx])
        for idx, uses_compact in enumerate(use_compact)
        if bool(uses_compact) and idx < len(req_ids)
    }


def _wait_replay_refresh_selectors_before_full_cudagraph_replay(
    *,
    controller: object,
    device: torch.device,
    pending_flags: Optional[Sequence[int]] = None,
    profile_pre_timing: Optional[dict[str, float]] = None,
) -> tuple[int, set[int]]:
    pending = controller._pending_refresh_rebuilds
    if not pending:
        return (0, set())
    current_stream = torch.cuda.current_stream(device=device)
    if pending_flags is None:
        flags_obj = getattr(controller, "_buf_pending_work_flags", None)
        pending_flags = (
            tuple(int(value) for value in flags_obj)
            if isinstance(flags_obj, (list, tuple))
            else ()
        )
    else:
        pending_flags = tuple(int(value) for value in pending_flags)
    waited = 0
    covered_bufs: set[int] = set()
    uncovered_bufs: set[int] = set()
    writer_done_bufs: set[int] = set()
    writer_ready_items: list[object] = []
    pending_buf_ids = controller._pending_refresh_rebuild_buf_ids
    wait_start_ns = _full_cudagraph_pre_timing_start(profile_pre_timing)

    def _current_step_handle_id() -> int:
        step_authority = _full_cudagraph_current_step_authority(controller)
        handle_id = int(getattr(step_authority, "step_handle_id", -1) or -1)
        if handle_id > 0:
            return handle_id
        step_context = getattr(controller, "step_context", None)
        return int(getattr(step_context, "step_handle_id", -1) or -1)

    def _current_epoch() -> int:
        step_authority = getattr(controller, "step_authority", None)
        epoch = int(getattr(step_authority, "epoch", -1) or -1)
        if epoch >= 0:
            return epoch
        return int(getattr(controller, "step_context_epoch", -1) or -1)

    def _buf_has_non_refresh_pending(buf_id: int) -> bool:
        buf = int(buf_id)
        if buf < 0 or buf >= len(pending_flags):
            return False
        # ExecutionBackendLedger uses bit0=prefill and bit1=refresh.  The
        # selector/writer events here cover refresh work only.
        return bool(int(pending_flags[buf]) & ~2)

    def _stream_wait_key(event: object) -> tuple[int, int]:
        stream_id = getattr(current_stream, "cuda_stream", None)
        try:
            stream_key = int(stream_id)
        except (TypeError, ValueError):
            stream_key = id(current_stream)
        return (id(event), int(stream_key))

    def _event_waited_for_current_stream(
        item: object,
        *,
        event: object,
        key_attr: str,
    ) -> bool:
        wait_key = _stream_wait_key(event)
        if getattr(item, key_attr, None) == wait_key:
            return True
        current_stream.wait_event(event)
        setattr(item, key_attr, wait_key)
        return False

    def _pending_writer_required_for_current_replay(item: object) -> bool:
        target_scope = getattr(item, "target_selected_scope_key", None)
        if target_scope is not None:
            return bool(
                _pending_refresh_rebuild_matches_current_target_scope(
                    controller,
                    item,  # type: ignore[arg-type]
                )
            )

        deadline_handle_id = int(getattr(item, "deadline_handle_id", -1) or -1)
        deadline_epoch = int(getattr(item, "deadline_epoch", -1) or -1)
        if deadline_handle_id > 0:
            current_handle_id = _current_step_handle_id()
            return current_handle_id <= 0 or deadline_handle_id <= current_handle_id
        if deadline_epoch >= 0:
            current_epoch = _current_epoch()
            return current_epoch < 0 or deadline_epoch <= current_epoch
        return True

    def _pending_writer_release_not_due(item: object) -> bool:
        release_after = int(getattr(item, "writer_release_after_handle_id", -1) or -1)
        if release_after <= 0:
            return False
        current_handle_id = _current_step_handle_id()
        return current_handle_id <= 0 or release_after > current_handle_id

    compact_consume_req_ids: set[str] | None = None

    def _current_compact_consume_req_ids() -> set[str]:
        nonlocal compact_consume_req_ids
        if compact_consume_req_ids is not None:
            return compact_consume_req_ids
        compact_consume_req_ids = (
            _current_compact_consume_req_ids_for_full_cudagraph_replay(
                controller
            )
        )
        return compact_consume_req_ids

    def _pending_writer_covers_current_compact_consumer(item: object) -> bool:
        compact_req_ids = _current_compact_consume_req_ids()
        if not compact_req_ids:
            return False
        pending_req_ids = tuple(
            str(v) for v in tuple(getattr(item, "req_ids", tuple()) or tuple())
        )
        if not pending_req_ids:
            return False
        return any(str(req_id) in compact_req_ids for req_id in pending_req_ids)

    completed_item_ids: set[int] = set()

    def _drop_non_latest_for_replay(item: object) -> bool:
        is_latest = getattr(controller, "_pending_refresh_rebuild_is_latest", None)
        if not callable(is_latest) or bool(is_latest(item)):
            return False
        drop_pending = getattr(controller, "_drop_pending_refresh_rebuild", None)
        if callable(drop_pending):
            remaining_pending = tuple(
                other
                for other in controller._pending_refresh_rebuilds
                if other is not item
            )
            drop_pending(
                item,
                status="drop_non_latest",
                lease_reason="pending_rebuild_drop_non_latest_full_cudagraph_replay",
                remaining_pending=remaining_pending,
                wait_recorded_work=False,
            )
        else:
            _mark_pending_selected_scope_terminal(
                item,  # type: ignore[arg-type]
                status="drop_non_latest",
            )
            drop_req_ids = tuple(controller._pending_refresh_rebuild_drop_req_ids(item))
            if drop_req_ids:
                controller._resolve_refresh_lease(
                    req_ids=drop_req_ids,
                    reason="pending_rebuild_drop_non_latest_full_cudagraph_replay",
                )
            controller._pending_refresh_rebuild_clear(item)
        completed_item_ids.add(id(item))
        return True

    for item in tuple(pending):
        item_bufs = tuple(
            v for v in (int(raw) for raw in pending_buf_ids(item)) if v >= 0
        )

        if str(getattr(item, "producer_kind", "")) != "full_cudagraph_replay_refresh":
            uncovered_bufs.update(int(v) for v in item_bufs)
            continue

        if _drop_non_latest_for_replay(item):
            covered_bufs.update(int(v) for v in item_bufs)
            continue

        writer_event = getattr(item, "writer_done_event", None)
        if writer_event is not None:
            if _pending_writer_required_for_current_replay(
                item
            ) or _pending_writer_covers_current_compact_consumer(item):
                if not _event_waited_for_current_stream(
                    item,
                    event=writer_event,
                    key_attr="writer_done_event_waited_for_replay_key",
                ):
                    waited += 1
                covered_bufs.update(int(v) for v in item_bufs)
                writer_done_bufs.update(int(v) for v in item_bufs)
                writer_ready_items.append(item)
                continue
        else:
            selector_event = getattr(item, "selector_done_event", None)
            selector_recorded = bool(
                getattr(item, "selector_done_event_recorded", False)
            )
            writer_required = _pending_writer_required_for_current_replay(item)
            writer_covers_compact_consumer = (
                _pending_writer_covers_current_compact_consumer(item)
            )
            if selector_event is not None and selector_recorded and (
                not writer_covers_compact_consumer
                and (not writer_required or _pending_writer_release_not_due(item))
            ):
                if not _event_waited_for_current_stream(
                    item,
                    event=selector_event,
                    key_attr="selector_done_event_waited_key",
                ):
                    waited += 1
                covered_bufs.update(int(v) for v in item_bufs)
                continue
            uncovered_bufs.update(int(v) for v in item_bufs)
            continue

        selector_event = getattr(item, "selector_done_event", None)
        selector_recorded = bool(
            getattr(item, "selector_done_event_recorded", False)
        )
        if selector_event is not None and selector_recorded:
            if not _event_waited_for_current_stream(
                item,
                event=selector_event,
                key_attr="selector_done_event_waited_key",
            ):
                waited += 1
            covered_bufs.update(int(v) for v in item_bufs)
            continue

        uncovered_bufs.update(int(v) for v in item_bufs)
    for buf_id in tuple(covered_bufs):
        if _buf_has_non_refresh_pending(int(buf_id)):
            uncovered_bufs.add(int(buf_id))
    if uncovered_bufs:
        covered_bufs.difference_update(uncovered_bufs)
        writer_done_bufs.difference_update(uncovered_bufs)
    def _run_writer_ready_commit() -> None:
        # Writer-ready CPU commit (publish_tracking / commit_compact_meta /
        # mark_accepted / clear_pending + _pending_refresh_rebuilds rebuild +
        # clearable-buf clears). Mutates bookkeeping consumed by FUTURE steps,
        # NOT by this step's original_call (the graph replay). Closes over the
        # wait-phase locals: writer_ready_items, completed_item_ids, pending,
        # writer_done_bufs, uncovered_bufs.
        if writer_ready_items:
            publish_tracking = controller._publish_pending_refresh_rebuild_selection_tracking
            commit_compact_meta = controller._commit_pending_refresh_rebuild_compact_meta
            mark_accepted = controller._mark_pending_refresh_rebuild_accepted
            clear_pending = controller._pending_refresh_rebuild_clear
            is_latest = controller._pending_refresh_rebuild_is_latest

            def _resolve_writer_ready_drop(item: object, *, reason: str) -> None:
                req_ids = tuple(controller._pending_refresh_rebuild_drop_req_ids(item))
                if not req_ids:
                    return
                controller._resolve_refresh_lease(req_ids=req_ids, reason=str(reason))

            for item in writer_ready_items:
                if not bool(is_latest(item)):
                    _mark_pending_selected_scope_terminal(
                        item,  # type: ignore[arg-type]
                        status="drop_non_latest",
                    )
                    _resolve_writer_ready_drop(
                        item,
                        reason="pending_rebuild_drop_non_latest_full_cudagraph_replay",
                    )
                    clear_pending(item)
                    completed_item_ids.add(id(item))
                    continue
                result = getattr(item, "result", None)
                commit_compact_meta(item)
                if not bool(getattr(item, "tracking_published", False)):
                    if result is None:
                        raise RuntimeError(
                            "full cudagraph replay pending refresh writer_done_event "
                            "has no selection result"
                        )
                    publish_tracking(item, result)
                    setattr(item, "tracking_published", True)
                mark_accepted(item)
                clear_pending(item)
                completed_item_ids.add(id(item))
        if completed_item_ids:
            controller._pending_refresh_rebuilds = deque(
                item for item in pending if id(item) not in completed_item_ids
            )
            clearable_writer_done_bufs: set[int] = set()
            for item in writer_ready_items:
                if id(item) not in completed_item_ids:
                    continue
                for raw_buf in pending_buf_ids(item):
                    buf = int(raw_buf)
                    if buf >= 0 and buf in writer_done_bufs:
                        clearable_writer_done_bufs.add(buf)
            clearable_writer_done_bufs.difference_update(uncovered_bufs)
            if clearable_writer_done_bufs:
                for item in controller._pending_refresh_rebuilds:
                    for raw_buf in pending_buf_ids(item):
                        buf = int(raw_buf)
                        if buf >= 0:
                            clearable_writer_done_bufs.discard(buf)
                for buf_id in sorted(clearable_writer_done_bufs):
                    controller._pending_work_clear_buf(buf_id=int(buf_id))

    _run_writer_ready_commit()
    _full_cudagraph_pre_timing_add(
        profile_pre_timing,
        "pre_replay_refresh_selector_wait_us",
        wait_start_ns,
    )
    return (int(waited), covered_bufs)


def _wait_pending_async_refresh_before_full_cudagraph_replay(
    *,
    controller: object,
    profile_pre_timing: Optional[dict[str, float]] = None,
) -> int:
    if not bool(controller._async_refresh_enabled()):
        return 0
    compute_wait = controller._compute_wait_decision
    wait_done = controller._main_stream_wait_for_chunk_done

    epoch = int(
        getattr(
            controller,
            "step_context_epoch",
            getattr(getattr(controller, "step_context", None), "epoch", -1),
        )
    )
    flags_obj = controller._buf_pending_work_flags
    flags: tuple[int, ...] = ()
    if isinstance(flags_obj, (list, tuple)):
        flags = tuple(int(value) for value in flags_obj)
    blockers = False
    blockers = bool(controller._pending_work_blockers(epoch=epoch))
    pending_queue = controller._pending_refresh_rebuilds
    if (
        flags
        and all(int(value) == 0 for value in flags)
        and not blockers
        and not pending_queue
    ):
        return 0

    device = _full_cudagraph_replay_wait_device(controller)
    if device is None or device.type != "cuda":
        return 0
    if torch.cuda.is_available():
        try:
            from patches.sparse_utils import _is_stream_capturing_or_raise

            if _is_stream_capturing_or_raise(
                stage="full_cudagraph_replay_pending_async_wait"
            ):
                return 0
        except RuntimeError:
            raise
        except Exception:
            pass

    collect_submit_diagnostics = bool(
        profile_pre_timing is not None or _fa3_route_trace_enabled()
    )
    if collect_submit_diagnostics:
        setattr(
            controller,
            "_mixed_page_full_cudagraph_last_selector_writer_submit_summary",
            None,
        )
    if pending_queue and bool(controller._refresh_producer_stream_release_pending):
        step_authority = _full_cudagraph_current_step_authority(controller)
        handle_id = int(getattr(step_authority, "step_handle_id", -1) or -1)
        if handle_id <= 0:
            step_context = getattr(controller, "step_context", None)
            handle_id = int(getattr(step_context, "step_handle_id", -1) or -1)
        if handle_id > 0:
            force_req_ids = tuple(
                sorted(
                    _current_compact_consume_req_ids_for_full_cudagraph_replay(
                        controller
                    )
                )
            )
            submit_debug: list[dict[str, object]] | None = (
                [] if collect_submit_diagnostics else None
            )
            setattr(
                controller,
                "_mixed_page_full_cudagraph_last_selector_writer_submit_debug",
                submit_debug,
            )
            pending_before = (
                int(len(pending_queue)) if collect_submit_diagnostics else 0
            )
            submitted = int(controller._submit_due_selector_prepared_refresh_writers(
                handle_id=handle_id,
                force_req_ids=force_req_ids,
            ))
            pending_queue = controller._pending_refresh_rebuilds
            submit_summary = None
            if collect_submit_diagnostics:
                submit_summary = {
                    "handle_id": int(handle_id),
                    "force_req_ids": list(force_req_ids),
                    "pending_before": int(pending_before),
                    "pending_after": int(len(tuple(pending_queue))),
                    "submitted": int(submitted),
                    "release_pending_before": True,
                    "release_pending_after": bool(
                        getattr(controller, "_refresh_producer_stream_release_pending", False)
                    ),
                    "records": list(submit_debug or []),
                }
                setattr(
                    controller,
                    "_mixed_page_full_cudagraph_last_selector_writer_submit_summary",
                    submit_summary,
                )
            if _fa3_route_trace_enabled() and submitted > 0:
                try:
                    from patches.fa3_native.install import append_fa3_route_trace

                    append_fa3_route_trace(
                        {
                            "event": "mixed_page_full_cudagraph_selector_writer_submit",
                            "step_id": _mixed_page_full_cudagraph_profile_step_id(
                                controller
                            ),
                            **submit_summary,
                        }
                    )
                except Exception:
                    pass
            flags_obj = controller._buf_pending_work_flags
            flags = (
                tuple(int(value) for value in flags_obj)
                if isinstance(flags_obj, (list, tuple))
                else ()
            )
            if (
                flags
                and all(int(value) == 0 for value in flags)
                and not blockers
                and not pending_queue
            ):
                return 0

    from patches.sparse_constants import _CAPTURE_IN_FLIGHT

    selector_waited, selector_covered_bufs = (
        _wait_replay_refresh_selectors_before_full_cudagraph_replay(
            controller=controller,
            device=device,
            pending_flags=flags,
            profile_pre_timing=profile_pre_timing,
        )
    )
    buf_count = len(flags) if flags else int(_CAPTURE_IN_FLIGHT)
    waited = 0
    wait_start_ns = _full_cudagraph_pre_timing_start(profile_pre_timing)
    for buf_id in range(int(buf_count)):
        if int(buf_id) in selector_covered_bufs:
            continue
        if flags and int(flags[buf_id]) == 0 and not blockers:
            continue
        need_wait, _reason = compute_wait(
            buf_id=int(buf_id),
            epoch=epoch,
            path_tag="full_cudagraph_replay",
            consume_step_token=False,
        )
        if not need_wait:
            continue
        wait_done(
            buf_id=int(buf_id),
            device=device,
            epoch=epoch,
        )
        waited += 1
    _full_cudagraph_pre_timing_add(
        profile_pre_timing,
        "pre_async_refresh_wait_us",
        wait_start_ns,
    )
    return int(waited) + int(selector_waited)


def _mark_prebound_rrp_full_cudagraph_replay(
    *,
    controller: object,
    forward_context: object,
    graph_key: str,
    profile_pre_timing: Optional[dict[str, float]] = None,
    state: dict[str, object] | None = None,
    metadata_items: tuple[object, ...] | None = None,
    binding_is_current_prevalidated: bool = False,
) -> object:
    from patches.fa_sparse_runtime.mixed_page_cudagraph_replay import (
        MixedPageForwardContextReplayStats,
        iter_attention_metadata,
    )

    _t_state_ns = _full_cudagraph_pre_timing_start(profile_pre_timing)
    if state is None:
        state = _resolved_row_ptr_graph_binding_state_for_replay(controller)
    _t_current_check_ns = _full_cudagraph_pre_timing_start(profile_pre_timing)
    owner_state_current = bool(
        state is not None
        and (
            binding_is_current_prevalidated
            or _prebound_rrp_graph_state_is_current_for_forward_context(
                controller=controller,
                forward_context=forward_context,
                state=state,
            )
        )
    )
    if metadata_items is None and owner_state_current:
        metadata_items = ()
    elif metadata_items is None:
        _t_metadata_items_ns = _full_cudagraph_pre_timing_start(profile_pre_timing)
        metadata_items = iter_attention_metadata(
            getattr(forward_context, "attn_metadata", None)
        )
        _full_cudagraph_pre_timing_add(
            profile_pre_timing,
            "pre_prebound_metadata_items_us",
            _t_metadata_items_ns,
        )
    else:
        metadata_items = tuple(metadata_items)
    if owner_state_current:
        binding_is_current = True
    else:
        binding_is_current = _prebound_rrp_rebind_already_current(
            controller=controller,
            forward_context=forward_context,
            metadata_items=metadata_items,
            state=state,
        )
    _full_cudagraph_pre_timing_add(
        profile_pre_timing,
        "pre_prebound_current_check_us",
        _t_current_check_ns,
    )
    if not binding_is_current:
        _t_rebind_ns = _full_cudagraph_pre_timing_start(profile_pre_timing)
        _maybe_rebind_rrp_for_full_graph_replay(
            controller=controller,
            forward_context=forward_context,
            metadata_items=metadata_items,
        )
        state = _resolved_row_ptr_graph_binding_state_for_replay(controller)
        _full_cudagraph_pre_timing_add(
            profile_pre_timing,
            "pre_prebound_rebind_us",
            _t_rebind_ns,
        )
    _full_cudagraph_pre_timing_add(
        profile_pre_timing,
        "pre_prebound_state_lookup_us",
        _t_state_ns,
    )
    if state is None:
        raise RuntimeError("resolved-row-ptr graph binding state is not replay-ready")

    step_ctx = getattr(controller, "step_context", None)
    step_id_value = getattr(
        controller,
        "step_context_epoch",
        getattr(step_ctx, "epoch", -1),
    )
    step_id = -1 if step_id_value is None else int(step_id_value)
    carrier_publish = None
    _t_wait_ns = _full_cudagraph_pre_timing_start(profile_pre_timing)
    replay_ready_state = _wait_prebound_rrp_ready_event_from_graph_state(
        controller=controller,
        forward_context=forward_context,
        state=state,
        profile_pre_timing=profile_pre_timing,
    )
    ready_event_wait_count = int(
        getattr(
            forward_context,
            "mixed_page_resolver_replay_ready_event_wait_count",
            0,
        )
    )
    _full_cudagraph_pre_timing_add(
        profile_pre_timing,
        "pre_ready_event_wait_us",
        _t_wait_ns,
    )
    _t_group_ready_wait_ns = _full_cudagraph_pre_timing_start(profile_pre_timing)
    group_ready_wait_count = int(
        _wait_one_shot_group_ready_before_prebound_rrp_full_graph_replay(
            controller=controller,
            forward_context=forward_context,
        )
        or 0
    )
    _full_cudagraph_pre_timing_add(
        profile_pre_timing,
        "pre_group_ready_check_us",
        _t_group_ready_wait_ns,
    )
    ready_event_wait_count += group_ready_wait_count
    if group_ready_wait_count > 0:
        _full_cudagraph_pre_timing_add(
            profile_pre_timing,
            "pre_ready_event_wait_us",
            _t_group_ready_wait_ns,
        )

    _t_stats_ns = _full_cudagraph_pre_timing_start(profile_pre_timing)
    static_values = _prebound_rrp_static_stats_values_cached(
        controller=controller,
        state=state,
        graph_key=graph_key,
    )
    static_values = _prebound_rrp_static_stats_values_with_carrier_publish(
        static_values,
        carrier_publish,
    )
    stats = MixedPageForwardContextReplayStats(
        *static_values,
        step_id,
        graph_key,
        int(ready_event_wait_count),
        state,
        int(replay_ready_state.generation),
    )
    _full_cudagraph_pre_timing_add(
        profile_pre_timing,
        "pre_prebound_stats_us",
        _t_stats_ns,
    )
    _t_publish_ns = _full_cudagraph_pre_timing_start(profile_pre_timing)
    setattr(
        controller,
        "_mixed_page_full_cudagraph_last_replay_stats",
        stats,
    )
    setattr(forward_context, "mixed_page_resolver_replay_last_step_id", step_id)
    setattr(forward_context, "mixed_page_resolver_replay_last_graph_key", graph_key)
    setattr(forward_context, "mixed_page_resolver_replay_last_stats", stats)
    setattr(
        forward_context,
        "mixed_page_resolver_replay_ready_event_wait_count",
        int(ready_event_wait_count),
    )
    if not hasattr(
        forward_context,
        "mixed_page_resolver_replay_ready_event_wait_total",
    ):
        setattr(
            forward_context,
            "mixed_page_resolver_replay_ready_event_wait_total",
            0,
        )
    if step_id >= 0:
        setattr(
            forward_context,
            "mixed_page_resolver_replay_refresh_guard",
            (int(step_id), id(getattr(forward_context, "attn_metadata", None))),
        )
    _full_cudagraph_pre_timing_add(
        profile_pre_timing,
        "pre_stats_publish_us",
        _t_publish_ns,
    )
    return stats


def _append_prebound_rrp_full_cudagraph_replay_trace(
    *,
    controller: object,
    forward_context: object,
    stats: object,
) -> None:
    if not _fa3_route_trace_enabled():
        return
    try:
        from patches.fa3_native.install import append_fa3_route_trace
        from patches.fa_sparse_runtime.mixed_page_cudagraph_replay import (
            iter_attention_metadata,
        )
    except Exception:
        return
    launch_plan_trace = _compact_recent_launch_plan_trace_payload(
        controller=controller,
        page_size=int(getattr(stats, "page_size", 16) or 16),
    )
    rrp_visible_source_trace = _metadata_items_rrp_visible_source_fields(
        iter_attention_metadata(getattr(forward_context, "attn_metadata", None))
    )
    append_fa3_route_trace(
        {
            "event": "mixed_page_full_cudagraph_replay_refresh",
            "step_id": int(getattr(stats, "step_id", -1)),
            "graph_key": str(getattr(stats, "graph_key", "")),
            "metadata_count": int(getattr(stats, "metadata_count", -1)),
            "updated_metadata_count": int(
                getattr(stats, "updated_metadata_count", -1)
            ),
            "carrier_update_rows": int(getattr(stats, "carrier_update_rows", -1)),
            "carrier_update_bytes": int(getattr(stats, "carrier_update_bytes", -1)),
            "carrier_update_kernel_count": int(
                getattr(stats, "carrier_update_kernel_count", -1)
            ),
            "ready_event_wait_count": int(
                getattr(
                    forward_context,
                    "mixed_page_resolver_replay_ready_event_wait_total",
                    0,
                )
            ),
            "carrier_pointer_signature_count": int(
                getattr(stats, "metadata_count", -1)
            ),
            "carrier_pointer_signature_stable": True,
            "row_mode_distribution": dict(
                getattr(stats, "row_mode_distribution", {})
            ),
            "row_source_distribution": dict(
                getattr(stats, "row_source_distribution", {})
            ),
            "fast_cached_direct_bound": True,
            "prebound_rrp_graph_state": True,
            "metadata_mark_skipped": int(getattr(stats, "carrier_update_rows", 0) or 0) <= 0,
            **launch_plan_trace,
            **rrp_visible_source_trace,
            **_mixed_page_replay_stats_source_counter_payload(stats),
        }
    )


def _append_mixed_page_full_cudagraph_hook_trace(
    *,
    hook: str,
    reason: str,
    controller: object | None,
    forward_context_available: bool,
    forward_context: object | None = None,
    wrapper_runtime_mode: object | None = None,
    batch_descriptor: object | None = None,
    entry_present: bool | None = None,
    cudagraph_present: bool | None = None,
    ubatch_num_tokens: int | None = None,
    ubatch_graph_present: bool | None = None,
    captured_route_family: object | None = None,
    current_route_family: object | None = None,
    route_family_mismatch: bool | None = None,
    bridge_graph_policy: str | None = None,
) -> None:
    if not _fa3_route_trace_enabled():
        return
    try:
        from patches.fa3_native.install import append_fa3_route_trace
    except Exception:
        return

    def _mode_name(value: object | None) -> str | None:
        if value is None:
            return None
        return str(getattr(value, "name", value))

    config = getattr(controller, "config", None) if controller is not None else None
    step_ctx = getattr(controller, "step_context", None) if controller is not None else None
    step_id_value = getattr(
        controller,
        "step_context_epoch",
        getattr(step_ctx, "epoch", -1),
    )
    try:
        step_id = int(step_id_value)
    except Exception:
        step_id = -1
    fc_mode = (
        getattr(forward_context, "cudagraph_runtime_mode", None)
        if forward_context is not None
        else None
    )
    append_fa3_route_trace(
        {
            "event": "mixed_page_full_cudagraph_replay_hook_check",
            "hook": str(hook),
            "reason": str(reason),
            "step_id": int(step_id),
            "controller_live": controller is not None,
            "config_enabled": bool(getattr(config, "enabled", False)),
            "compact_page_residency_enabled": bool(
                getattr(config, "compact_page_residency_enabled", False)
            ),
            "attention_in_cudagraph": bool(
                _attention_in_cudagraph_enabled(controller)
            ),
            "refresh_enabled": bool(
                _mixed_page_full_cudagraph_replay_refresh_enabled(controller)
            ),
            "forward_context_available": bool(forward_context_available),
            "cudagraph_runtime_mode": _mode_name(fc_mode),
            "wrapper_runtime_mode": _mode_name(wrapper_runtime_mode),
            "batch_descriptor": None
            if batch_descriptor is None
            else repr(batch_descriptor),
            "entry_present": entry_present,
            "cudagraph_present": cudagraph_present,
            "ubatch_num_tokens": ubatch_num_tokens,
            "ubatch_graph_present": ubatch_graph_present,
            "captured_route_family": captured_route_family,
            "current_route_family": current_route_family,
            "route_family_mismatch": route_family_mismatch,
            "bridge_graph_policy": bridge_graph_policy,
        }
    )


def _append_mixed_page_full_cudagraph_profile_event(
    profile_path: str,
    payload: dict,
) -> None:
    if not profile_path:
        return
    try:
        with open(profile_path, "a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    payload,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
    except Exception:
        pass









def _begin_full_cudagraph_replay_cuda_event(
    *,
    path: str,
    reason: str,
    step_id: int,
    graph_key: str,
    batch_descriptor: object,
    state: dict[str, object] | None,
    step_identity_payload: dict[str, object],
    prebound_identity_payload: dict[str, object],
) -> dict[str, object] | None:
    log_all_replays = str(
        os.environ.get("VLLM_FULL_CUDAGRAPH_REPLAY_CUDA_EVENT_LOG_ALL", "0")
        or "0"
    ).strip().lower() in {"1", "true", "yes", "on"}
    if not path or (
        str(reason) != "prebound_rrp_graph_state" and not log_all_replays
    ):
        return None
    if (state is None and not log_all_replays) or not torch.cuda.is_available():
        return None
    try:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    except Exception:
        return None
    state_payload = state if isinstance(state, dict) else {}
    row_mode_distribution = state_payload.get("row_mode_distribution", {})
    row_source_distribution = state_payload.get("row_source_distribution", {})
    payload: dict[str, object] = {
        "event": "mixed_page_full_cudagraph_replay_cuda_event",
        "pid": int(os.getpid()),
        "host_begin_ns": int(time.time_ns()),
        "step_id": int(step_id),
        "graph_key": str(graph_key),
        "batch_descriptor": "" if batch_descriptor is None else str(batch_descriptor),
        "reason": str(reason),
        "row_mode_distribution": dict(row_mode_distribution)
        if isinstance(row_mode_distribution, dict)
        else {},
        "row_source_distribution": dict(row_source_distribution)
        if isinstance(row_source_distribution, dict)
        else {},
        "expected_rows": int(state_payload.get("expected_rows", -1) or -1),
        "num_kv_heads": int(state_payload.get("num_kv_heads", -1) or -1),
        "path": str(path),
        "_start_event": start_event,
        "_end_event": end_event,
    }
    payload.update(step_identity_payload)
    payload.update(prebound_identity_payload)
    return payload


def _end_full_cudagraph_replay_cuda_event(sample: dict[str, object] | None) -> None:
    if sample is None:
        return
    try:
        end_event = sample.get("_end_event")
        record = getattr(end_event, "record", None)
        if not callable(record):
            return
        record()
    except Exception:
        return
    global _FULL_CUDAGRAPH_REPLAY_CUDA_EVENT_PATH
    global _FULL_CUDAGRAPH_REPLAY_CUDA_EVENT_REGISTERED
    path = str(sample.get("path", "") or "")
    if not path:
        return
    _FULL_CUDAGRAPH_REPLAY_CUDA_EVENT_PATH = path
    if not _FULL_CUDAGRAPH_REPLAY_CUDA_EVENT_REGISTERED:
        atexit.register(_flush_full_cudagraph_replay_cuda_events)
        _FULL_CUDAGRAPH_REPLAY_CUDA_EVENT_REGISTERED = True
    sample["host_end_ns"] = int(time.time_ns())
    _FULL_CUDAGRAPH_REPLAY_CUDA_EVENT_ROWS.append(sample)


def _flush_full_cudagraph_replay_cuda_events(profile_path: str | None = None) -> None:
    global _FULL_CUDAGRAPH_REPLAY_CUDA_EVENT_PATH
    path = str(profile_path or _FULL_CUDAGRAPH_REPLAY_CUDA_EVENT_PATH or "")
    rows = list(_FULL_CUDAGRAPH_REPLAY_CUDA_EVENT_ROWS)
    if not path or not rows:
        return
    parent = os.path.dirname(path)
    if parent:
        try:
            os.makedirs(parent, exist_ok=True)
        except Exception:
            return
    try:
        with open(path, "a", encoding="utf-8") as fh:
            for row in rows:
                start_event = row.get("_start_event")
                end_event = row.get("_end_event")
                payload = {
                    key: value
                    for key, value in row.items()
                    if key != "path" and not str(key).startswith("_")
                }
                try:
                    synchronize = getattr(end_event, "synchronize", None)
                    elapsed_time = getattr(start_event, "elapsed_time", None)
                    if not callable(synchronize) or not callable(elapsed_time):
                        raise RuntimeError("missing CUDA event pair")
                    synchronize()
                    payload["cuda_ms"] = float(elapsed_time(end_event))
                except Exception as exc:
                    payload["cuda_ms"] = -1.0
                    payload["cuda_event_error"] = str(exc)
                fh.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    except Exception:
        return
    _FULL_CUDAGRAPH_REPLAY_CUDA_EVENT_ROWS.clear()
    _FULL_CUDAGRAPH_REPLAY_CUDA_EVENT_PATH = path

def _new_full_cudagraph_pre_timing(profile_enabled: object) -> Optional[dict[str, float]]:
    return {} if profile_enabled else None


def _full_cudagraph_pre_timing_start(timing: Optional[dict[str, float]]) -> int:
    return time.perf_counter_ns() if timing is not None else 0


def _full_cudagraph_pre_timing_add(
    timing: Optional[dict[str, float]],
    key: str,
    start_ns: int,
) -> None:
    if timing is None or start_ns <= 0:
        return
    timing[key] = float(timing.get(key, 0.0)) + (
        time.perf_counter_ns() - start_ns
    ) / 1000.0


def _full_cudagraph_pre_timing_get(
    timing: Optional[dict[str, float]],
    key: str,
) -> float:
    if timing is None:
        return 0.0
    return float(timing.get(key, 0.0) or 0.0)


def _full_cudagraph_pre_other_us(
    timing: Optional[dict[str, float]],
    *,
    pre_original_us: float,
    pre_consume_drain_us: float,
) -> float:
    accounted = (
        _full_cudagraph_pre_timing_get(timing, "pre_step_identity_us")
        + _full_cudagraph_pre_timing_get(timing, "pre_initial_bind_identity_us")
        + _full_cudagraph_pre_timing_get(timing, "pre_prebound_bind_identity_us")
        + _full_cudagraph_pre_timing_get(timing, "pre_async_refresh_probe_us")
        + _full_cudagraph_pre_timing_get(timing, "pre_forward_context_us")
        + _full_cudagraph_pre_timing_get(timing, "pre_graph_lookup_us")
        + _full_cudagraph_pre_timing_get(timing, "pre_graph_key_us")
        + _full_cudagraph_pre_timing_get(timing, "pre_prebound_graph_state_us")
        + _full_cudagraph_pre_timing_get(timing, "pre_prebound_mark_us")
        + _full_cudagraph_pre_timing_get(timing, "pre_async_refresh_wait_us")
        + max(0.0, float(pre_consume_drain_us))
    )
    return max(0.0, float(pre_original_us) - accounted)


def _mixed_page_full_cudagraph_profile_step_id(controller: object | None) -> int:
    step_ctx = getattr(controller, "step_context", None) if controller is not None else None
    step_id_value = getattr(
        controller,
        "step_context_epoch",
        getattr(step_ctx, "epoch", -1),
    )
    try:
        return int(step_id_value)
    except Exception:
        return -1


def _mixed_page_full_cudagraph_int_attr(
    obj: object | None,
    name: str,
    default: int = -1,
) -> int:
    if obj is None:
        return int(default)
    try:
        return int(getattr(obj, name, default))
    except Exception:
        return int(default)


def _mixed_page_full_cudagraph_step_identity_payload(
    controller: object | None,
) -> dict[str, object]:
    step_ctx = getattr(controller, "step_context", None) if controller is not None else None
    step_authority = getattr(step_ctx, "step_authority", None)
    if step_authority is None and controller is not None:
        step_authority = getattr(controller, "step_authority", None)
    epoch = _mixed_page_full_cudagraph_profile_step_id(controller)
    step_handle_id = _mixed_page_full_cudagraph_int_attr(step_authority, "step_handle_id")
    step_handle_generation = _mixed_page_full_cudagraph_int_attr(step_authority, "step_handle_generation")
    step_identity_token = _mixed_page_full_cudagraph_int_attr(step_authority, "step_identity_token")
    req_set_hash = _mixed_page_full_cudagraph_int_attr(step_authority, "req_set_hash")
    row_phase_hash = _mixed_page_full_cudagraph_int_attr(step_authority, "row_phase_hash")
    batch_size = _mixed_page_full_cudagraph_int_attr(step_authority, "batch_size")
    return {
        "step_handle_id": int(step_handle_id),
        "step_handle_generation": int(step_handle_generation),
        "step_identity_token": int(step_identity_token),
        "step_req_set_hash": int(req_set_hash),
        "step_row_phase_hash": int(row_phase_hash),
        "step_batch_size": int(batch_size),
        "step_identity_key": (
            f"{int(epoch)}:{int(step_handle_id)}:{int(step_handle_generation)}:"
            f"{int(step_identity_token)}:{int(req_set_hash)}:{int(row_phase_hash)}:"
            f"{int(batch_size)}"
        ),
    }


def _mixed_page_full_cudagraph_bind_identity_payload(
    state: dict[str, object] | None,
) -> dict[str, object]:
    bind_epoch = -1
    bind_handle_id = -1
    bind_handle_generation = -1
    bind_identity_token = -1
    bind_req_set_hash = -1
    bind_row_phase_hash = -1
    bind_live_batch_size = -1
    bind_effective_batch_size = -1
    if isinstance(state, dict):
        fast_identity = state.get("bind_step_fast_identity", None)
        if isinstance(fast_identity, (list, tuple)) and len(fast_identity) >= 8:
            try:
                bind_epoch = int(fast_identity[0])
                bind_handle_id = int(fast_identity[1])
                bind_handle_generation = int(fast_identity[2])
                bind_identity_token = int(fast_identity[3])
                bind_req_set_hash = int(fast_identity[4])
                bind_row_phase_hash = int(fast_identity[5])
                bind_live_batch_size = int(fast_identity[6])
                bind_effective_batch_size = int(fast_identity[7])
            except Exception:
                bind_epoch = -1
                bind_handle_id = -1
                bind_handle_generation = -1
                bind_identity_token = -1
                bind_req_set_hash = -1
                bind_row_phase_hash = -1
        else:
            identity = state.get("bind_step_identity", None)
            if isinstance(identity, (list, tuple)) and len(identity) >= 3:
                try:
                    bind_epoch = int(identity[0])
                    bind_handle_id = int(identity[1])
                    bind_handle_generation = int(identity[2])
                except Exception:
                    bind_epoch = -1
                    bind_handle_id = -1
                    bind_handle_generation = -1
        try:
            bind_live_batch_size = int(state.get("bind_live_batch_size", bind_live_batch_size))
        except Exception:
            bind_live_batch_size = -1
        try:
            bind_effective_batch_size = int(state.get("bind_effective_batch_size", bind_effective_batch_size))
        except Exception:
            bind_effective_batch_size = -1
    return {
        "bind_step_epoch": int(bind_epoch),
        "bind_step_handle_id": int(bind_handle_id),
        "bind_step_handle_generation": int(bind_handle_generation),
        "bind_step_identity_token": int(bind_identity_token),
        "bind_req_set_hash": int(bind_req_set_hash),
        "bind_row_phase_hash": int(bind_row_phase_hash),
        "bind_live_batch_size": int(bind_live_batch_size),
        "bind_effective_batch_size": int(bind_effective_batch_size),
        "bind_identity_key": (
            f"{int(bind_epoch)}:{int(bind_handle_id)}:{int(bind_handle_generation)}:"
            f"{int(bind_identity_token)}:{int(bind_req_set_hash)}:"
            f"{int(bind_row_phase_hash)}:{int(bind_live_batch_size)}:"
            f"{int(bind_effective_batch_size)}"
        ),
    }


def _cache_full_cudagraph_replay_payload_refs(
    *,
    state: object,
    key_cache: object,
    value_cache: object,
    block_table: object,
    q: object,
    cu_seqlens_q: object,
    seqused_k: object,
    softmax_scale: object,
    softcap: object,
    window_size: object,
    alibi_slopes: object,
    k_descale: object,
) -> None:
    """Cache stable layer inputs needed to build refresh payloads after graph replay."""
    setattr(state, "_sfi_replay_key_cache", key_cache)
    setattr(state, "_sfi_replay_value_cache", value_cache)
    setattr(state, "_sfi_replay_block_table", block_table)
    setattr(state, "_sfi_replay_q", q)
    setattr(state, "_sfi_replay_cu_seqlens_q", cu_seqlens_q)
    setattr(state, "_sfi_replay_seqused_k", seqused_k)
    setattr(state, "_sfi_replay_softmax_scale", float(softmax_scale or 0.0))
    setattr(state, "_sfi_replay_softcap", float(softcap or 0.0))
    setattr(state, "_sfi_replay_window_size", window_size)
    setattr(state, "_sfi_replay_alibi_slopes", alibi_slopes)
    setattr(state, "_sfi_replay_k_descale", k_descale)


def _full_cudagraph_replay_refresh_batched_flush_enabled(controller: object) -> bool:
    return bool(_FULL_CUDAGRAPH_REPLAY_REFRESH_BATCHED_FLUSH_CACHED)


def _full_cudagraph_replay_refresh_defer_to_deadline_enabled(
    controller: object,
) -> bool:
    if not _full_cudagraph_replay_refresh_batched_flush_enabled(controller):
        return False
    return bool(_FULL_CUDAGRAPH_REPLAY_REFRESH_DEFER_TO_DEADLINE_CACHED)


def _drain_full_cudagraph_pending_refresh_before_replay(
    *,
    controller: object | None,
) -> dict[str, int]:
    if controller is None:
        return {"drained": 0, "dropped": 0, "remaining": 0}
    if not _full_cudagraph_replay_refresh_defer_to_deadline_enabled(controller):
        return {"drained": 0, "dropped": 0, "remaining": 0}
    pending = controller._pending_refresh_rebuilds
    if _REPLAY_REFRESH_NOOP_FAST_SKIP_CACHED and not pending:
        return {"drained": 0, "dropped": 0, "remaining": 0}
    drain = controller._pending_refresh_rebuild_pre_consume_drain
    drained, dropped, remaining = drain()
    return {
        "drained": int(drained),
        "dropped": int(dropped),
        "remaining": int(remaining),
    }


def _drain_full_cudagraph_pending_refresh_before_replay_profiled(
    *,
    controller: object | None,
    profile_path: str,
) -> tuple[dict[str, int], float]:
    if not _full_cudagraph_replay_refresh_defer_to_deadline_enabled(controller):
        return ({"drained": 0, "dropped": 0, "remaining": 0}, 0.0)
    start_ns = time.perf_counter_ns() if profile_path else 0
    stats = _drain_full_cudagraph_pending_refresh_before_replay(
        controller=controller,
    )
    elapsed_us = (
        (time.perf_counter_ns() - start_ns) / 1000.0 if profile_path else 0.0
    )
    return (stats, float(elapsed_us))


def _full_cudagraph_replay_step_has_refresh_row(controller: object) -> bool:
    step_ctx = getattr(controller, "step_context", None)
    step_authority = getattr(step_ctx, "step_authority", None)
    if step_authority is None:
        step_authority = getattr(controller, "step_authority", None)
    if step_ctx is None or step_authority is None:
        return False
    if int(getattr(step_ctx, "epoch", -1)) != int(getattr(step_authority, "epoch", -2)):
        raise RuntimeError(
            "full cudagraph replay refresh payload enqueue requires matching step authority"
        )
    return bool(getattr(step_authority, "has_refresh_row", False))


def _flush_full_cudagraph_refresh_payloads_batched_after_replay(
    *,
    controller: object,
    payloads: Sequence[object],
    stage_profile: Optional[dict[str, object]] = None,
) -> int:
    from patches.sparse_constants import _CAPTURE_IN_FLIGHT

    if not payloads:
        return 0
    stage_profile_enabled = isinstance(stage_profile, dict)
    group_total_start_ns = time.perf_counter_ns() if stage_profile_enabled else 0

    def _stage_add(key: str, elapsed_us: float) -> None:
        if stage_profile is None:
            return
        stage_profile[key] = float(stage_profile.get(key, 0.0) or 0.0) + float(elapsed_us)

    def _stage_inc(key: str, amount: int = 1) -> None:
        if stage_profile is None:
            return
        stage_profile[key] = int(stage_profile.get(key, 0) or 0) + int(amount)

    first_payload = payloads[0]
    setup_start_ns = time.perf_counter_ns() if stage_profile_enabled else 0
    device = getattr(getattr(first_payload, "capture_scores", None), "device", None)
    if device is None:
        raise RuntimeError("batched replay refresh requires CUDA payload tensors")
    ensure_refresh_stream = getattr(controller, "_ensure_refresh_stream", None)
    if callable(ensure_refresh_stream):
        ensure_refresh_stream(device)
    refresh_stream = getattr(controller, "refresh_stream", None)
    chunk_ready_evt = getattr(controller, "chunk_ready_evt", None)
    chunk_done_evt = getattr(controller, "chunk_done_evt", None)
    refresh_done_evt = getattr(controller, "refresh_done_evt", None)
    async_enabled = getattr(controller, "_async_refresh_enabled", None)
    do_async = bool(callable(async_enabled) and async_enabled())
    if do_async and (refresh_stream is None or not chunk_ready_evt or not chunk_done_evt):
        do_async = False

    layer_to_buf: dict[int, int] = {}
    map_layer = getattr(controller, "_map_global_layer_to_capture_slot", None)
    for payload in payloads:
        state = getattr(payload, "state", None)
        layer_index = int(getattr(state, "layer_index", -1))
        if layer_index < 0:
            layer_index = int(getattr(payload, "layer_index", -1))
        if layer_index < 0 or not callable(map_layer):
            continue
        _, buf_id, _ = map_layer(layer_index)
        layer_to_buf[int(layer_index)] = int(buf_id) % int(_CAPTURE_IN_FLIGHT)
    buf_ids = tuple(sorted(set(layer_to_buf.values())))
    if stage_profile_enabled:
        _stage_add(
            "flush_group_setup_us",
            (time.perf_counter_ns() - setup_start_ns) / 1000.0,
        )

    mark_submitted = getattr(controller, "_pending_work_mark_submitted", None)
    step_epoch = int(getattr(controller, "step_context_epoch", -1))
    mark_start_ns = time.perf_counter_ns() if stage_profile_enabled else 0
    if callable(mark_submitted):
        for buf_id in buf_ids:
            mark_submitted(
                buf_id=int(buf_id),
                kind="refresh",
                async_mode=bool(do_async),
                epoch=step_epoch,
            )
    if stage_profile_enabled:
        _stage_add(
            "flush_group_mark_submitted_us",
            (time.perf_counter_ns() - mark_start_ns) / 1000.0,
        )

    def _record_stream_for_payloads() -> None:
        if not do_async or refresh_stream is None:
            return
        seen_ptrs: set[int] = set()

        def _record(tensor: object) -> None:
            if not isinstance(tensor, torch.Tensor):
                return
            try:
                ptr = int(tensor.untyped_storage().data_ptr())
            except Exception:
                ptr = int(tensor.data_ptr())
            if ptr == 0 or ptr in seen_ptrs:
                return
            tensor.record_stream(refresh_stream)
            seen_ptrs.add(ptr)

        for payload in payloads:
            for tensor in (
                getattr(getattr(payload, "capture_scores", None), "_base", None),
                getattr(payload, "capture_scores", None),
                getattr(getattr(payload, "lastn1_capture_scores", None), "_base", None),
                getattr(payload, "lastn1_capture_scores", None),
                getattr(getattr(payload, "log_f_denoms", None), "_base", None),
                getattr(payload, "log_f_denoms", None),
                getattr(payload, "seq_lens_batch", None),
                getattr(payload, "seq_lens_batch_i32", None),
                getattr(payload, "kv_lengths", None),
                getattr(payload, "kv_len_per_row_i32", None),
                getattr(payload, "row_tensor_i32", None),
                getattr(payload, "row_tensor", None),
                getattr(payload, "refresh_rows_long", None),
                getattr(payload, "refresh_block_table_sub", None),
                getattr(payload, "refresh_seq_lens_i32", None),
                getattr(payload, "slot_tensor", None),
                getattr(payload, "slot_tensor_i32", None),
                getattr(payload, "cu_seqlens_q", None),
                getattr(payload, "alibi_slopes", None),
                getattr(payload, "k_descale", None),
                getattr(payload, "block_table", None),
            ):
                _record(tensor)

    def _publish_refresh_writer(result: object) -> None:
        rebuild = getattr(
            controller,
            "_rebuild_compact_slots_batched_layers_from_selection",
            None,
        )
        if not callable(rebuild):
            raise RuntimeError("batched replay refresh requires compact rebuild hook")
        writer_start_ns = time.perf_counter_ns() if stage_profile_enabled else 0
        fused_ok = rebuild(
            payloads,
            getattr(result, "selected_indices"),
            phase="refresh",
            bootstrap_slots_by_layer=[
                getattr(payload, "bootstrap_slots", None) for payload in payloads
            ],
        )
        if stage_profile_enabled:
            _stage_add(
                "flush_group_writer_us",
                (time.perf_counter_ns() - writer_start_ns) / 1000.0,
            )
        if not fused_ok:
            raise RuntimeError("batched replay refresh compact rebuild failed")
    def _run_refresh() -> None:
        selector = getattr(controller, "_apply_alpha_selector_batched_fused", None)
        if not callable(selector):
            raise RuntimeError("batched replay refresh requires selector hook")
        selector_start_ns = time.perf_counter_ns() if stage_profile_enabled else 0
        result = selector(
            payloads,
            phase="decode",
            update_tracking=True,
        )
        if stage_profile_enabled:
            _stage_add(
                "flush_group_selector_us",
                (time.perf_counter_ns() - selector_start_ns) / 1000.0,
            )
        if result is None:
            return
        _publish_refresh_writer(result)

    if getattr(controller, "_refresh_layer_group_enabled", False):
        setattr(controller, "_refresh_layer_group_any_refresh", True)

    if do_async:
        main_stream = torch.cuda.current_stream(device=device)
        async_event_start_ns = time.perf_counter_ns() if stage_profile_enabled else 0
        for buf_id in buf_ids:
            chunk_ready_evt[int(buf_id)].record(main_stream)
        if stage_profile_enabled:
            _stage_add(
                "flush_group_async_event_us",
                (time.perf_counter_ns() - async_event_start_ns) / 1000.0,
            )
        with torch.cuda.stream(refresh_stream):
            cur_stream = torch.cuda.current_stream(device=device)
            async_event_start_ns = time.perf_counter_ns() if stage_profile_enabled else 0
            for buf_id in buf_ids:
                cur_stream.wait_event(chunk_ready_evt[int(buf_id)])
            if stage_profile_enabled:
                _stage_add(
                    "flush_group_async_event_us",
                    (time.perf_counter_ns() - async_event_start_ns) / 1000.0,
                )
            record_stream_start_ns = time.perf_counter_ns() if stage_profile_enabled else 0
            _record_stream_for_payloads()
            if stage_profile_enabled:
                _stage_add(
                    "flush_group_record_stream_us",
                    (time.perf_counter_ns() - record_stream_start_ns) / 1000.0,
                )
            _run_refresh()
            async_event_start_ns = time.perf_counter_ns() if stage_profile_enabled else 0
            for buf_id in buf_ids:
                if refresh_done_evt:
                    refresh_done_evt[int(buf_id)].record(cur_stream)
                chunk_done_evt[int(buf_id)].record(cur_stream)
            if stage_profile_enabled:
                _stage_add(
                    "flush_group_async_event_us",
                    (time.perf_counter_ns() - async_event_start_ns) / 1000.0,
                )
    else:
        record_stream_start_ns = time.perf_counter_ns() if stage_profile_enabled else 0
        _record_stream_for_payloads()
        if stage_profile_enabled:
            _stage_add(
                "flush_group_record_stream_us",
                (time.perf_counter_ns() - record_stream_start_ns) / 1000.0,
            )
        _run_refresh()
        if callable(async_enabled) and async_enabled():
            if chunk_done_evt:
                cur_stream = torch.cuda.current_stream(device=device)
                async_event_start_ns = time.perf_counter_ns() if stage_profile_enabled else 0
                for buf_id in buf_ids:
                    cur_stream.record_event(chunk_done_evt[int(buf_id)])
                    if refresh_done_evt:
                        cur_stream.record_event(refresh_done_evt[int(buf_id)])
                if stage_profile_enabled:
                    _stage_add(
                        "flush_group_async_event_us",
                        (time.perf_counter_ns() - async_event_start_ns) / 1000.0,
                    )
            clear_buf = getattr(controller, "_pending_work_clear_buf", None)
            if callable(clear_buf):
                for buf_id in buf_ids:
                    clear_buf(buf_id=int(buf_id))

    if stage_profile_enabled:
        _stage_inc("flush_group_count")
        _stage_add(
            "flush_group_total_us",
            (time.perf_counter_ns() - group_total_start_ns) / 1000.0,
        )
    return int(len(payloads))


def _enqueue_one_replay_refresh_payload_group(
    *,
    controller: object,
    group: list,
    result: object | None = None,
    workspace: object,
    enqueue_pending,
    map_layer,
    deadline_slack_steps: int = -1,
    stage_profile: Optional[dict[str, object]] = None,
    profile_detail: bool = False,
) -> bool:
    """Enqueue a single payload_group via the controller's pending-rebuild path.

    Extracted so the stagger drain hook and the trigger-fire enqueue path
    can share the per-group body.
    """
    if not group:
        return False
    detail_enabled = bool(profile_detail and isinstance(stage_profile, dict))

    def _detail_add_elapsed(key: str, start_ns: int) -> None:
        if not detail_enabled or stage_profile is None:
            return
        elapsed_us = (time.perf_counter_ns() - int(start_ns)) / 1000.0
        stage_profile[key] = (
            float(stage_profile.get(key, 0.0) or 0.0) + float(elapsed_us)
        )

    def _detail_inc(key: str, amount: int = 1) -> None:
        if not detail_enabled or stage_profile is None:
            return
        stage_profile[key] = int(stage_profile.get(key, 0) or 0) + int(amount)

    group_total_start_ns = time.perf_counter_ns() if detail_enabled else 0
    _detail_inc("pending_group_count")
    _detail_inc("pending_group_payload_count", len(group))
    clear_start_ns = time.perf_counter_ns() if detail_enabled else 0
    for payload in group:
        # Direct replay-refresh producer publishes compact KV state, not a
        # selected-scope launch gate. Binding to the step's selected-scope
        # handle would make a valid deadline item look stale on the next step.
        setattr(payload, "target_selected_scope_key", None)
        setattr(payload, "selected_scope_wait_handle", None)
    _detail_add_elapsed("pending_group_clear_us", clear_start_ns)
    carrier_start_ns = time.perf_counter_ns() if detail_enabled else 0
    carrier = workspace.prepare_refresh_carrier(
        group,
        map_layer=map_layer,
        normalize_req_ids=getattr(controller, "_normalize_refresh_req_ids", None),
        deadline_slack_steps=int(deadline_slack_steps),
    )
    _detail_add_elapsed("pending_group_carrier_prepare_us", carrier_start_ns)
    first_layer = (
        int(carrier.layer_indices[0]) if carrier.layer_indices else -1
    )
    if first_layer < 0:
        raise RuntimeError(
            "full cudagraph replay refresh deadline deferral missing layer index"
        )
    chunk_id = int(getattr(carrier, "chunk_id", -1))
    buf_id = int(getattr(carrier, "buf_id", -1))
    if chunk_id < 0 or buf_id < 0:
        raise RuntimeError(
            "full cudagraph replay refresh carrier missing capture slot: "
            f"layer={first_layer} chunk_id={chunk_id} buf_id={buf_id}"
        )
    bootstrap_start_ns = time.perf_counter_ns() if detail_enabled else 0
    bootstrap_slots_by_layer = list(carrier.bootstrap_slots_by_layer)
    _detail_add_elapsed("pending_group_bootstrap_slots_list_us", bootstrap_start_ns)
    enqueue_start_ns = time.perf_counter_ns() if detail_enabled else 0
    enqueue_pending(
        payloads=group,
        result=result,
        selection_phase="decode",
        rebuild_phase="refresh",
        bootstrap_slots_by_layer=bootstrap_slots_by_layer,
        chunk_id=int(chunk_id),
        buf_id=int(buf_id),
        producer_kind="full_cudagraph_replay_refresh",
        producer_carrier=carrier,
        stage_profile_detail=stage_profile if detail_enabled else None,
    )
    _detail_add_elapsed("pending_group_enqueue_pending_us", enqueue_start_ns)
    _detail_add_elapsed("pending_group_total_us", group_total_start_ns)
    return True


def _full_cudagraph_replay_refresh_progressive_consume_enabled() -> bool:
    return os.environ.get("VLLM_SPARSE_REPLAY_REFRESH_PROGRESSIVE_CONSUME", "0") == "1"


def _resolve_replay_refresh_progressive_group_count(
    *,
    total_layers: int,
    capture_chunk: int,
    default_group_count: int,
) -> int:
    from patches.refresh_runtime.producer_ready import (
        resolve_one_shot_ready_chunk,
        validate_one_shot_ready_chunk_alignment,
    )

    capture_chunk_i = max(1, int(capture_chunk))
    ready_chunk = resolve_one_shot_ready_chunk(capture_chunk=capture_chunk_i)
    if (
        not _full_cudagraph_replay_refresh_progressive_consume_enabled()
        and int(ready_chunk) >= capture_chunk_i
    ):
        return int(default_group_count)
    validate_one_shot_ready_chunk_alignment(
        capture_chunk=capture_chunk_i,
        ready_chunk=int(ready_chunk),
    )
    if int(ready_chunk) >= capture_chunk_i:
        return int(default_group_count)
    total_layers_i = max(1, int(total_layers))
    return max(1, (total_layers_i + int(ready_chunk) - 1) // int(ready_chunk))


def _payload_layer_index_for_stagger(payload: object) -> int:
    try:
        stagger_layer_index = int(getattr(payload, "stagger_layer_index", -1))
    except (TypeError, ValueError):
        stagger_layer_index = -1
    if stagger_layer_index >= 0:
        return int(stagger_layer_index)
    try:
        layer_index = int(getattr(payload, "layer_index", -1))
    except (TypeError, ValueError):
        layer_index = -1
    if layer_index < 0:
        try:
            layer_index = int(
                getattr(getattr(payload, "state", None), "layer_index", -1)
            )
        except (TypeError, ValueError):
            layer_index = -1
    return int(layer_index)


def _payload_capture_layer_index_for_stagger(payload: object) -> int:
    try:
        layer_index = int(getattr(payload, "layer_index", -1))
    except (TypeError, ValueError):
        layer_index = -1
    if layer_index < 0:
        try:
            layer_index = int(
                getattr(getattr(payload, "state", None), "layer_index", -1)
            )
        except (TypeError, ValueError):
            layer_index = -1
    return int(layer_index)


def _replay_refresh_group_has_unique_layer_indices(group: Sequence[object]) -> bool:
    seen: set[int] = set()
    prev_layer = -1
    for payload in group:
        layer_index = _payload_layer_index_for_stagger(payload)
        if layer_index < 0:
            return False
        if layer_index in seen:
            return False
        if prev_layer >= 0 and layer_index <= prev_layer:
            return False
        seen.add(layer_index)
        prev_layer = int(layer_index)
    return True


def _replay_refresh_group_has_monotonic_capture_layers(
    group: Sequence[object],
) -> bool:
    prev_layer = -1
    saw_layer = False
    for payload in group:
        layer_index = _payload_capture_layer_index_for_stagger(payload)
        if layer_index < 0:
            continue
        saw_layer = True
        if prev_layer >= 0 and layer_index <= prev_layer:
            return False
        prev_layer = int(layer_index)
    return bool(saw_layer)


def _split_replay_refresh_payloads_on_layer_reuse(
    payloads: Sequence[object],
) -> list[list[object]]:
    groups: list[list[object]] = []
    current: list[object] = []
    seen: set[int] = set()
    prev_layer = -1
    for payload in payloads:
        layer_index = _payload_layer_index_for_stagger(payload)
        wraps_layer_order = prev_layer >= 0 and layer_index <= prev_layer
        if current and (layer_index < 0 or layer_index in seen or wraps_layer_order):
            groups.append(current)
            current = []
            seen = set()
            prev_layer = -1
        current.append(payload)
        if layer_index >= 0:
            seen.add(layer_index)
            prev_layer = int(layer_index)
    if current:
        groups.append(current)
    return groups


def _split_replay_refresh_payloads_on_capture_layer_reuse(
    payloads: Sequence[object],
) -> list[list[object]]:
    groups: list[list[object]] = []
    current: list[object] = []
    prev_layer = -1
    saw_layer = False
    for payload in payloads:
        layer_index = _payload_capture_layer_index_for_stagger(payload)
        wraps_layer_order = (
            current and layer_index >= 0 and prev_layer >= 0 and layer_index <= prev_layer
        )
        if wraps_layer_order:
            groups.append(current)
            current = []
            prev_layer = -1
        current.append(payload)
        if layer_index >= 0:
            saw_layer = True
            prev_layer = int(layer_index)
    if current:
        groups.append(current)
    return groups if saw_layer else []


def _rebucket_replay_refresh_payload_groups_for_stagger(
    payload_groups: Sequence[Sequence[object]],
    *,
    target_group_count: int,
) -> list[list[object]]:
    """Build a small number of layer groups for true decode-step staggering."""
    original = [list(group) for group in payload_groups if group]
    flattened = [payload for group in original for payload in group]
    if not flattened:
        return []
    if any(_payload_layer_index_for_stagger(payload) < 0 for payload in flattened):
        return original
    group_count = max(1, int(target_group_count))
    capture_segments = _split_replay_refresh_payloads_on_capture_layer_reuse(flattened)
    if len(capture_segments) > 1:
        base_groups = max(1, group_count // len(capture_segments))
        extra_groups = max(0, group_count - base_groups * len(capture_segments))
        out: list[list[object]] = []
        for segment_index, segment in enumerate(capture_segments):
            segment_group_count = base_groups + (1 if segment_index < extra_groups else 0)
            group_size = max(1, (len(segment) + segment_group_count - 1) // segment_group_count)
            for start in range(0, len(segment), group_size):
                out.append(segment[start:start + group_size])
        if all(
            _replay_refresh_group_has_unique_layer_indices(group)
            and _replay_refresh_group_has_monotonic_capture_layers(group)
            for group in out
        ):
            return out
        return capture_segments
    group_size = max(1, (len(flattened) + group_count - 1) // group_count)
    out: list[list[object]] = []
    for start in range(0, len(flattened), group_size):
        out.append(flattened[start:start + group_size])
    if not all(
        _replay_refresh_group_has_unique_layer_indices(group)
        and _replay_refresh_group_has_monotonic_capture_layers(group)
        for group in out
    ):
        return _split_replay_refresh_payloads_on_layer_reuse(flattened)
    return out


def _adaptive_replay_refresh_stagger_group_count(
    *,
    total_layers: int,
    capture_chunk: int,
) -> int:
    total_layers = max(0, int(total_layers))
    if total_layers <= 0:
        return 1
    capture_chunk = max(1, int(capture_chunk))
    layer_chunks = max(1, (total_layers + capture_chunk - 1) // capture_chunk)
    # The rebucket helper preserves capture segments when layer ordinals wrap.
    # Using one producer group per capture chunk is the lowest launch-count
    # schedule that keeps those segments independent; smaller groups only add
    # post-call drain launches that the full-KV handoff can no longer hide.
    return min(total_layers, layer_chunks)


def _adaptive_replay_refresh_deferred_deadline_steps(
    *,
    target_group_count: int,
    ready_chunk_subgroups: bool = False,
) -> int:
    """Give deferred replay one post-call slot per producer group."""
    if int(_REFRESH_REBUILD_MAX_DELAY_STEPS_CACHED) > 0:
        return max(1, int(_REFRESH_REBUILD_MAX_DELAY_STEPS_CACHED))
    if bool(ready_chunk_subgroups):
        return 1
    return max(1, int(target_group_count))


def _enqueue_full_cudagraph_refresh_payloads_after_replay(
    *,
    controller: object,
    graph_key: str = "",
) -> int:
    stage_profile_enabled = bool(_full_cudagraph_hook_profile_log(True))
    stage_profile: Optional[dict[str, object]] = None
    stage_total_start_ns = time.perf_counter_ns() if stage_profile_enabled else 0
    if stage_profile_enabled:
        stage_profile = {
            "schema": "full_cudagraph_replay_refresh_stage_profile_v1",
            "graph_key": str(graph_key),
            "layer_count": 0,
            "payload_count": 0,
            "prepare_logits_call_count": 0,
            "flush_call_count": 0,
            "prepare_logits_us": 0.0,
            "payload_build_us": 0.0,
            "payload_build_us_max": 0.0,
            "slot_req_ids_us": 0.0,
            "payload_enqueue_us": 0.0,
            "deferred_pending_rebuild_enqueue_us": 0.0,
            "flush_us": 0.0,
            "flush_us_max": 0.0,
            "direct_accounted_us": 0.0,
            "total_us": 0.0,
            "residual_us": 0.0,
        }
        setattr(
            controller,
            "_mixed_page_full_cudagraph_last_replay_refresh_stage_profile",
            None,
        )

    def _stage_add(key: str, elapsed_us: float) -> None:
        if stage_profile is None:
            return
        stage_profile[key] = float(stage_profile.get(key, 0.0) or 0.0) + float(elapsed_us)

    def _stage_max(key: str, elapsed_us: float) -> None:
        if stage_profile is None:
            return
        stage_profile[key] = max(
            float(stage_profile.get(key, 0.0) or 0.0),
            float(elapsed_us),
        )

    def _stage_inc(key: str, amount: int = 1) -> None:
        if stage_profile is None:
            return
        stage_profile[key] = int(stage_profile.get(key, 0) or 0) + int(amount)

    step_ctx = getattr(controller, "step_context", None)
    step_authority = getattr(step_ctx, "step_authority", None)
    if step_authority is None:
        step_authority = getattr(controller, "step_authority", None)
    if step_ctx is None or step_authority is None:
        return 0
    if not _full_cudagraph_replay_step_has_refresh_row(controller):
        return 0
    refresh_slot_list = getattr(step_authority, "refresh_capture_slot_list", None)
    if not isinstance(refresh_slot_list, tuple) or not refresh_slot_list:
        raise RuntimeError(
            "full cudagraph replay refresh payload enqueue requires non-empty "
            "step_authority.refresh_capture_slot_list"
        )
    if not hasattr(controller, "_enqueue_refresh_capture") or not hasattr(controller, "_flush_prefill_batches"):
        raise RuntimeError("full cudagraph replay refresh payload enqueue requires refresh runtime hooks")

    from patches.sparse_constants import _CAPTURE_CHUNK
    from patches.sparse_types import SelectorBatchPayload
    from patches.vllm_sparse_patch import _prepare_refresh_capture_payload

    step_meta = getattr(controller, "step_meta", None)
    seqused_k = getattr(step_meta, "seqused_k_gpu", None)
    if not isinstance(seqused_k, torch.Tensor):
        seqused_k = getattr(step_meta, "canonical_real_kv_len_i32_gpu", None)
    layer_keys = tuple(getattr(controller, "layer_cache_keys", tuple()) or tuple())
    if not layer_keys:
        raise RuntimeError("full cudagraph replay refresh payload enqueue requires layer_cache_keys")
    layer_states = getattr(controller, "layer_states", None)
    if not isinstance(layer_states, dict):
        raise RuntimeError("full cudagraph replay refresh payload enqueue requires layer_states")

    handle_id = int(getattr(step_ctx, "step_handle_id", -1))
    handle_generation = int(getattr(step_ctx, "step_handle_generation", -1))
    if handle_id <= 0 or handle_generation <= 0:
        raise RuntimeError("full cudagraph replay refresh payload enqueue requires step handle identity")
    claim_generation = getattr(
        controller,
        "_step_refresh_commit_claim_replay_payload_generation",
        None,
    )
    if not callable(claim_generation):
        raise RuntimeError(
            "full cudagraph replay refresh requires a generation claim hook"
        )
    if not bool(
        claim_generation(
            handle_id=handle_id,
            handle_generation=handle_generation,
            step_identity_token=int(
                getattr(step_authority, "step_identity_token", -1)
            ),
            expected_payloads=len(layer_keys),
        )
    ):
        return 0
    step_envelope = getattr(step_ctx, "step_envelope_v2", None)
    refresh_reason = str(
        getattr(
            step_envelope,
            "refresh_reason",
            getattr(controller, "_step_profile_refresh_reason", ""),
        )
    )
    refresh_intent_req_ids = tuple(
        str(req_id)
        for req_id in (getattr(step_envelope, "refresh_reqs", ()) or ())
    )
    cache_epoch = int(getattr(controller, "_step_refresh_slot_req_ids_epoch", -1))
    cache_handle_id = int(getattr(controller, "_step_refresh_slot_req_ids_handle_id", -1))
    cache_handle_generation = int(
        getattr(controller, "_step_refresh_slot_req_ids_handle_generation", -1)
    )
    if (
        cache_epoch != int(getattr(step_ctx, "epoch", -1))
        or cache_handle_id != int(handle_id)
        or cache_handle_generation != int(handle_generation)
    ):
        controller._step_refresh_slot_req_ids_epoch = int(getattr(step_ctx, "epoch", -1))
        controller._step_refresh_slot_req_ids_handle_id = int(handle_id)
        controller._step_refresh_slot_req_ids_handle_generation = int(handle_generation)
        controller._step_refresh_slot_req_ids_cache = {}
    slot_req_ids_cache = getattr(controller, "_step_refresh_slot_req_ids_cache", None)
    if not isinstance(slot_req_ids_cache, dict):
        slot_req_ids_cache = {}
        controller._step_refresh_slot_req_ids_cache = slot_req_ids_cache

    def _slot_req_ids_for_state(state: object, slot_list: Sequence[int]) -> tuple[str, ...]:
        batch_request_ids = getattr(state, "batch_request_ids", ())
        slot_key = tuple(int(slot) for slot in slot_list)
        cache_key = ("full_cudagraph_replay", id(batch_request_ids), slot_key)
        cached = slot_req_ids_cache.get(cache_key)
        if cached is not None:
            return cached
        resolved = tuple(str(batch_request_ids[int(slot)]) for slot in slot_key)
        slot_req_ids_cache[cache_key] = resolved
        return resolved

    route_trace_enabled = _fa3_route_trace_enabled()
    workload_plan_replay_enabled = bool(
        os.environ.get("VLLM_SPARSE_REFRESH_WORKLOAD_PLAN_REPLAY")
    )

    def _route_refresh_intent_debug() -> list[dict[str, object]]:
        if not route_trace_enabled or workload_plan_replay_enabled:
            return []
        from patches.request_intent_ticket import pending_reason_code_to_text

        intent_debug = []
        tickets = getattr(controller, "_request_intent_tickets", {})
        request_states = getattr(controller, "request_states", {})
        for req_id in refresh_intent_req_ids:
            ticket = tickets.get(req_id) if isinstance(tickets, dict) else None
            tracking = request_states.get(req_id) if isinstance(request_states, dict) else None
            pending_reason_code = int(getattr(ticket, "pending_reason_code", 0))
            compact_ready = False
            compact_ready_all_layers = getattr(
                controller,
                "_request_compact_ready_all_layers",
                None,
            )
            if callable(compact_ready_all_layers):
                compact_ready = bool(compact_ready_all_layers(req_id))
            pending_rebuild_ids_for_req = getattr(
                controller,
                "_pending_refresh_rebuild_ids_for_req",
                None,
            )
            if callable(pending_rebuild_ids_for_req):
                pending_rebuild_ids = [
                    int(v) for v in pending_rebuild_ids_for_req(req_id)
                ]
            else:
                pending_rebuild_ids = []
            intent_debug.append(
                {
                    "req_id": str(req_id),
                    "decode_step": int(getattr(tracking, "decode_step", -1)),
                    "last_decode_refresh_step": int(
                        getattr(tracking, "last_decode_refresh_step", -1)
                    ),
                    "scheduled_decode_refresh_step": int(
                        getattr(tracking, "scheduled_decode_refresh_step", -1)
                    ),
                    "scheduled_refresh_ctrl_step": int(
                        getattr(tracking, "scheduled_refresh_ctrl_step", -1)
                    ),
                    "trigger_intent_decode_step": int(
                        getattr(tracking, "trigger_intent_decode_step", -1)
                    ),
                    "trigger_intent_reason": str(
                        getattr(tracking, "trigger_intent_reason", "none")
                    ),
                    "lease_rearm": bool(getattr(tracking, "lease_rearm", False)),
                    "lease_rearm_decode_step": int(
                        getattr(tracking, "lease_rearm_decode_step", -1)
                    ),
                    "pending_refresh": bool(getattr(ticket, "pending_refresh", False)),
                    "pending_reason_code": pending_reason_code,
                    "pending_reason": pending_reason_code_to_text(pending_reason_code),
                    "pending_decode_step": int(
                        getattr(ticket, "pending_decode_step", -1)
                    ),
                    "pending_ctrl_step": int(
                        getattr(ticket, "pending_ctrl_step", -1)
                    ),
                    "pending_policy": int(getattr(ticket, "pending_policy", 0)),
                    "compact_ready_all_layers": bool(compact_ready),
                    "pending_rebuild_ids": pending_rebuild_ids,
                }
            )
        return intent_debug

    if route_trace_enabled:
        try:
            from patches.fa3_native.install import append_fa3_route_trace

            append_fa3_route_trace(
                {
                    "event": "mixed_page_full_cudagraph_replay_refresh_payload_plan",
                    "step_id": int(getattr(step_ctx, "epoch", -1)),
                    "graph_key": str(graph_key),
                    "refresh_reason": str(refresh_reason),
                    "refresh_intent_req_count": int(len(refresh_intent_req_ids)),
                    "refresh_intent_req_ids": list(refresh_intent_req_ids),
                    "refresh_intent_debug": _route_refresh_intent_debug(),
                }
            )
        except Exception:
            pass

    def _refresh_logits_buffers_ready() -> bool:
        bound_meta = getattr(controller, "step_bound_meta", None)
        last_n_by_row = tuple(
            int(v) for v in getattr(bound_meta, "logits_last_n_by_row", tuple())
        )
        capacity_by_row = tuple(
            int(v) for v in getattr(bound_meta, "logits_capacity_by_row", tuple())
        )
        return (
            max(last_n_by_row, default=0) > 0
            and max(capacity_by_row, default=0) > 0
        )

    replay_logits_buffers_ready = _refresh_logits_buffers_ready()
    enqueued = 0
    total_layers = len(layer_keys)
    batched_flush_enabled = _full_cudagraph_replay_refresh_batched_flush_enabled(controller)
    batched_payloads: Optional[list[object]] = None
    batched_note_enqueues = None
    batched_note_inflight = None
    if batched_flush_enabled:
        from patches.refresh_runtime.producer_workspace import get_refresh_producer_workspace

        batched_payloads = get_refresh_producer_workspace(
            controller
        ).begin_replay_refresh_payloads()
        batched_note_enqueues = getattr(
            controller,
            "_step_refresh_commit_note_enqueues",
            None,
        )
        batched_note_inflight = getattr(
            controller,
            "_step_refresh_commit_note_inflight_from_payload",
            None,
        )
        if not callable(batched_note_enqueues) or not callable(batched_note_inflight):
            raise RuntimeError(
                "batched replay refresh requires refresh commit note hooks"
            )
    batched_inflight_commit_noted = False
    first_payload_slot_req_ids: Optional[tuple[str, ...]] = None
    # [ROW-SNAPSHOT-CARRIER 2026-07-07] 票自带提交时刻 block_table 快照(attn
    # kernel tail_fill_page"表满合法页不变量"同款哲学):payload 的 row_list 与
    # 表内容取自同一时刻,vLLM condense 行迁移从此与在飞票彻底无关——gather
    # 收集的 KV 本就是提交时刻的,映射同时刻语义更正确。每世代每源表 clone
    # 一次(~32KB D2D,全层共享,µs 级);is_latest 的行号对账退化为纯防线。
    _btable_snapshot_by_ptr: Dict[int, torch.Tensor] = {}
    # [HOOK-PERLAYER-DIET 2026-07-09] 世代内层不变量上提（取证 residual 662µs/世代主项）：
    # layout ring 同步校验/step 级标量/authority 字段在同一世代内逐层重算为纯冗余。
    # layout 判定按 buf_id 记忆（同 buf 同世代同判定）；capture slot 映射走
    # _register_layer 已维护的 state 缓存（layer_index_epoch 门失效即回退全量调用）。
    _step_epoch_hoisted = int(getattr(step_ctx, "epoch", -1))
    _refresh_slot_tuple_hoisted = tuple(int(v) for v in refresh_slot_list)
    _target_scope_key_hoisted = getattr(step_authority, "target_selected_scope_key", None)
    _scope_wait_handle_hoisted = getattr(step_authority, "selected_scope_wait_handle", None)
    _layer_index_cache_epoch = int(getattr(controller, "_layer_index_cache_epoch", -1))
    _layout_verdict_by_buf: Dict[int, object] = {}
    refresh_ring = getattr(controller, "step_refresh_capture_layout_ring", None)
    if not isinstance(refresh_ring, list):
        raise RuntimeError(
            "full cudagraph replay refresh payload enqueue missing refresh layout ring"
        )
    for ordinal, cache_key_raw in enumerate(layer_keys):
        _stage_inc("layer_count")
        cache_key = int(cache_key_raw)
        state = layer_states.get(cache_key)
        if state is None:
            raise RuntimeError(
                "full cudagraph replay refresh payload enqueue missing layer state"
            )
        layer_index = int(getattr(state, "layer_index", -1))
        if layer_index < 0:
            layer_index = int(getattr(controller, "layer_index_by_cache_key", {}).get(cache_key, ordinal))
        if layer_index < 0:
            raise RuntimeError(
                "full cudagraph replay refresh payload enqueue missing layer index"
            )
        if (
            int(getattr(state, "layer_index_epoch", -2)) == _layer_index_cache_epoch
            and int(getattr(state, "capture_chunk_id", -1)) >= 0
            and int(getattr(state, "layer_index", -1)) == layer_index
        ):
            chunk_id = int(state.capture_chunk_id)
            buf_id = int(state.capture_buf_id)
            slot_in_chunk = int(state.capture_slot_in_chunk)
        else:
            chunk_id, buf_id, slot_in_chunk = controller._map_global_layer_to_capture_slot(layer_index)
        if int(buf_id) >= len(refresh_ring):
            raise RuntimeError(
                "full cudagraph replay refresh payload enqueue missing refresh layout ring"
            )
        # None 判定不记忆：builder 会在 layout=None 时经 _get_step_capture_layout
        # 就地重建 ring 槽，同 chunk 次层必须能看到新建 layout（原行为）。
        layout = _layout_verdict_by_buf.get(int(buf_id))
        if layout is None:
            layout = refresh_ring[int(buf_id)]
            if layout is not None:
                same_step_layout = (
                    int(getattr(layout, "epoch", -1)) == _step_epoch_hoisted
                    and int(getattr(layout, "step_handle_id", -1)) == handle_id
                    and int(getattr(layout, "step_handle_generation", -1)) == handle_generation
                    and tuple(int(v) for v in getattr(layout, "slot_list", tuple()))
                    == _refresh_slot_tuple_hoisted
                )
                if not same_step_layout:
                    layout = None
                else:
                    _layout_verdict_by_buf[int(buf_id)] = layout

        key_cache = getattr(state, "_sfi_replay_key_cache", None)
        value_cache = getattr(state, "_sfi_replay_value_cache", None)
        block_table = getattr(state, "_sfi_replay_block_table", None)
        if not isinstance(block_table, torch.Tensor):
            block_table = getattr(controller, "_worker_block_table", None)
        if isinstance(block_table, torch.Tensor):
            # [ROW-SNAPSHOT-CARRIER] 世代级快照(循环外 dict,per 源表一次)。
            _bt_key = int(block_table.data_ptr())
            _bt_snap = _btable_snapshot_by_ptr.get(_bt_key)
            if _bt_snap is None:
                _bt_snap = block_table.clone()
                _btable_snapshot_by_ptr[_bt_key] = _bt_snap
            block_table = _bt_snap
        q = getattr(state, "_sfi_replay_q", None)
        cu_seqlens_q = getattr(state, "_sfi_replay_cu_seqlens_q", None)
        layer_seqused_k = seqused_k if isinstance(seqused_k, torch.Tensor) else getattr(
            state,
            "_sfi_replay_seqused_k",
            None,
        )
        if (
            not isinstance(key_cache, torch.Tensor)
            or not isinstance(value_cache, torch.Tensor)
            or not isinstance(block_table, torch.Tensor)
            or not isinstance(q, torch.Tensor)
            or not isinstance(cu_seqlens_q, torch.Tensor)
            or not isinstance(layer_seqused_k, torch.Tensor)
        ):
            raise RuntimeError(
                "full cudagraph replay refresh payload enqueue missing cached layer tensors"
            )

        if not replay_logits_buffers_ready:
            prepare_logits_buffers = getattr(controller, "prepare_step_logits_buffers", None)
            if not callable(prepare_logits_buffers):
                raise RuntimeError(
                    "full cudagraph replay refresh payload enqueue requires "
                    "prepare_step_logits_buffers"
                )
            capture_plan_by_req = None
            get_prefill_plan = getattr(controller, "get_step_prefill_plan_by_req", None)
            if callable(get_prefill_plan):
                capture_plan_by_req, _ = get_prefill_plan(step_context=step_ctx)
            seq_lens_cpu = tuple(
                int(v) for v in getattr(step_ctx, "seq_lens", tuple())
            )
            max_seqlen_k = int(getattr(step_ctx, "max_seq_len", 0) or 0)
            if max_seqlen_k <= 0:
                max_seqlen_k = max(seq_lens_cpu, default=0)
            block_size = (
                int(key_cache.shape[1])
                if int(getattr(key_cache, "ndim", 0)) >= 2
                else int(getattr(step_meta, "block_size", 0) or 0)
            )
            _t_prepare0_ns = time.perf_counter_ns() if stage_profile_enabled else 0
            prepare_logits_buffers(
                state=state,
                step_context=step_ctx,
                capture_plan_by_req=(
                    capture_plan_by_req if capture_plan_by_req else None
                ),
                seqused_k=layer_seqused_k,
                max_seqlen_k=int(max_seqlen_k),
                block_size=int(block_size),
                num_heads=int(q.shape[1]),
                device=q.device,
            )
            if stage_profile_enabled:
                _prepare_us = (time.perf_counter_ns() - _t_prepare0_ns) / 1000.0
                _stage_add("prepare_logits_us", _prepare_us)
                _stage_inc("prepare_logits_call_count")
            replay_logits_buffers_ready = _refresh_logits_buffers_ready()
            if not replay_logits_buffers_ready:
                raise RuntimeError(
                    "full cudagraph replay refresh payload enqueue failed to prepare refresh logits buffers"
                )

        _t_payload0_ns = time.perf_counter_ns() if stage_profile_enabled else 0
        payload = _prepare_refresh_capture_payload(
            controller=controller,
            cache_key=cache_key,
            state=state,
            step_context=step_ctx,
            q=q,
            cu_seqlens_q=cu_seqlens_q,
            seqused_k=layer_seqused_k,
            slots_filter=refresh_slot_list,
            slots_filter_sorted=True,
            layout=layout,
        )
        if stage_profile_enabled:
            _payload_build_us = (time.perf_counter_ns() - _t_payload0_ns) / 1000.0
            _stage_add("payload_build_us", _payload_build_us)
            _stage_max("payload_build_us_max", _payload_build_us)
        if payload is None:
            raise RuntimeError(
                "full cudagraph replay refresh payload build returned None"
            )
        (
            capture_scores,
            log_f_denoms,
            kv_lengths_tensor,
            seq_lens_batch,
            seq_lens_batch_i32,
            slot_list,
            slot_tensor,
            slot_tensor_i32,
            slot_tensor_cpu,
            row_list,
            row_tensor,
            row_tensor_i32,
            kv_len_per_row_i32,
            seq_lens_cpu,
            seq_lens_tensor_cpu,
            layer_index_in_chunk,
        ) = payload
        _t_slot_req0_ns = time.perf_counter_ns() if stage_profile_enabled else 0
        slot_req_ids = _slot_req_ids_for_state(state, slot_list)
        if stage_profile_enabled:
            _stage_add(
                "slot_req_ids_us",
                (time.perf_counter_ns() - _t_slot_req0_ns) / 1000.0,
            )
        if first_payload_slot_req_ids is None:
            first_payload_slot_req_ids = slot_req_ids
        _t_enqueue0_ns = time.perf_counter_ns() if stage_profile_enabled else 0
        payload_obj = SelectorBatchPayload(
            cache_key=cache_key,
            state=state,
            capture_scores=capture_scores,
            log_f_denoms=log_f_denoms,
            kv_lengths=kv_lengths_tensor,
            kv_len_per_row_i32=kv_len_per_row_i32,
            seq_lens_batch=seq_lens_batch,
            seq_lens_batch_i32=seq_lens_batch_i32,
            slot_list=slot_list,
            row_list=row_list,
            slot_req_ids=slot_req_ids,
            slot_tensor=slot_tensor,
            slot_tensor_i32=slot_tensor_i32,
            slot_tensor_cpu=slot_tensor_cpu,
            row_tensor=row_tensor,
            row_tensor_i32=row_tensor_i32,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=block_table,
            bootstrap_slots=set(),
            q=q,
            q_is_sub=False,
            cu_seqlens_q=cu_seqlens_q,
            softmax_scale=float(getattr(state, "_sfi_replay_softmax_scale", 0.0) or 0.0),
            softcap=float(getattr(state, "_sfi_replay_softcap", 0.0) or 0.0),
            window_size=getattr(state, "_sfi_replay_window_size", None),
            alibi_slopes=getattr(state, "_sfi_replay_alibi_slopes", None),
            k_descale=getattr(state, "_sfi_replay_k_descale", None),
            layer_index=int(layer_index_in_chunk),
            seq_lens_cpu=seq_lens_cpu if seq_lens_cpu is not None else tuple(),
            seq_lens_tensor_cpu=seq_lens_tensor_cpu,
            target_selected_scope_key=_target_scope_key_hoisted,
            selected_scope_wait_handle=_scope_wait_handle_hoisted,
            # Replay refresh runs with trusted-shape validation by default.
            # Avoid per-layer signature tuple/string construction on the
            # trigger hot path; strict fallback validation remains available.
            fast_signature=None,
            capture_handle_id=handle_id,
            capture_handle_generation=handle_generation,
            capture_epoch=_step_epoch_hoisted,
            refresh_reason=refresh_reason,
            refresh_intent_req_ids=refresh_intent_req_ids,
            stagger_layer_index=int(layer_index),
        )
        if batched_flush_enabled:
            if not batched_inflight_commit_noted:
                batched_note_inflight(payload_obj)
                batched_inflight_commit_noted = True
            if batched_payloads is None:
                raise RuntimeError("batched replay refresh payload workspace missing")
            batched_payloads.append(payload_obj)
        else:
            controller._enqueue_refresh_capture(payload_obj)
        if stage_profile_enabled:
            _stage_add(
                "payload_enqueue_us",
                (time.perf_counter_ns() - _t_enqueue0_ns) / 1000.0,
            )
            _stage_inc("payload_count")
        enqueued += 1
        is_last_layer = bool(layer_index == total_layers - 1 or ordinal == total_layers - 1)
        if (
            not batched_flush_enabled
            and ((int(slot_in_chunk) == int(_CAPTURE_CHUNK) - 1) or is_last_layer)
        ):
            chunk_size = (int(slot_in_chunk) + 1) if is_last_layer else int(_CAPTURE_CHUNK)
            _t_flush0_ns = time.perf_counter_ns() if stage_profile_enabled else 0
            controller._flush_prefill_batches(
                buf_id=int(buf_id),
                chunk_id=int(chunk_id),
                chunk_size=int(chunk_size),
                is_last_layer=bool(is_last_layer),
            )
            if stage_profile_enabled:
                _flush_us = (time.perf_counter_ns() - _t_flush0_ns) / 1000.0
                _stage_add("flush_us", _flush_us)
                _stage_max("flush_us_max", _flush_us)
                _stage_inc("flush_call_count")

    if enqueued <= 0:
        raise RuntimeError(
            "full cudagraph replay refresh payload enqueue found refresh rows but enqueued nothing"
        )
    if batched_flush_enabled:
        if not callable(batched_note_enqueues):
            raise RuntimeError("batched replay refresh enqueue counter hook missing")
        batched_note_enqueues(
            handle_id=handle_id,
            handle_generation=handle_generation,
            count=int(enqueued),
        )
    record_refresh_payloads = getattr(
        controller,
        "_step_profile_record_refresh_payloads",
        None,
    )
    if callable(record_refresh_payloads):
        record_refresh_payloads(
            count=int(enqueued),
            slot_req_ids=first_payload_slot_req_ids,
        )
    else:
        record_refresh_payload = getattr(
            controller,
            "_step_profile_record_refresh_payload",
            None,
        )
        if callable(record_refresh_payload):
            for _ in range(int(enqueued)):
                record_refresh_payload()
    if batched_flush_enabled:
        if batched_payloads is None:
            raise RuntimeError("batched replay refresh payload workspace missing")
        from patches.refresh_runtime.producer_workspace import (
            build_refresh_producer_work_item,
            get_refresh_producer_workspace,
            partition_replay_refresh_payloads_for_direct_submit,
        )

        payload_groups = partition_replay_refresh_payloads_for_direct_submit(
            batched_payloads
        )
        if not payload_groups:
            raise RuntimeError("batched replay refresh found no submit groups")
        submit_group_count_for_profile = len(payload_groups)
        defer_to_deadline = _full_cudagraph_replay_refresh_defer_to_deadline_enabled(
            controller
        )
        _t_flush0_ns = time.perf_counter_ns() if stage_profile_enabled else 0
        if defer_to_deadline:
            enqueue_pending = getattr(controller, "_enqueue_pending_refresh_rebuild", None)
            map_layer = getattr(controller, "_map_global_layer_to_capture_slot", None)
            if not callable(enqueue_pending) or not callable(map_layer):
                raise RuntimeError(
                    "full cudagraph replay refresh deadline deferral requires "
                    "pending rebuild enqueue hooks"
                )
            workspace = get_refresh_producer_workspace(controller)
            stagger_enabled = bool(_REFRESH_ENQUEUE_STAGGER_CACHED)
            default_group_count = _adaptive_replay_refresh_stagger_group_count(
                total_layers=len(layer_keys),
                capture_chunk=_CAPTURE_CHUNK,
            )
            target_group_count = _resolve_replay_refresh_progressive_group_count(
                total_layers=len(layer_keys),
                capture_chunk=_CAPTURE_CHUNK,
                default_group_count=default_group_count,
            )
            new_groups = (
                _rebucket_replay_refresh_payload_groups_for_stagger(
                    payload_groups,
                    target_group_count=target_group_count,
                )
                if stagger_enabled
                else [list(g) for g in payload_groups if g]
            )
            delay_steps = _adaptive_replay_refresh_deferred_deadline_steps(
                target_group_count=target_group_count,
                ready_chunk_subgroups=(
                    int(target_group_count) > int(default_group_count)
                ),
            )
            if delay_steps <= 0:
                delay_steps = max(1, target_group_count)
            submit_group_count_for_profile = len(new_groups)
            pending_count = 0
            queued_count = 0
            grouped_async_started = False
            grouped_async_loop_ok = False
            grouped_async_submitted = 0

            if (
                _full_cudagraph_replay_refresh_progressive_consume_enabled()
                and len(new_groups) > 1
            ):
                from patches.refresh_runtime.producer_workspace import (
                    fuse_replay_refresh_payload_groups_for_adjacent_selector_source,
                    replay_refresh_payload_groups_have_adjacent_selector_fusion,
                )

                if replay_refresh_payload_groups_have_adjacent_selector_fusion(
                    new_groups
                ):
                    fused_groups = (
                        fuse_replay_refresh_payload_groups_for_adjacent_selector_source(
                            new_groups,
                            max_payloads_per_group=_CAPTURE_CHUNK,
                        )
                    )
                    if len(fused_groups) < len(new_groups):
                        if stage_profile_enabled:
                            _stage_inc(
                                "progressive_adjacent_fused_group_saved",
                                len(new_groups) - len(fused_groups),
                            )
                            _stage_inc(
                                "progressive_adjacent_fused_group_count",
                                len(fused_groups),
                            )
                        new_groups = fused_groups
                        submit_group_count_for_profile = len(new_groups)
                    elif len(fused_groups) != len(new_groups):
                        raise RuntimeError(
                            "replay-refresh progressive consume group fusion grew "
                            f"groups: before={len(new_groups)} after={len(fused_groups)}"
                        )

            grouped_async_started = bool(
                controller._begin_pending_refresh_grouped_async_envelope()
            )
            try:
                if stagger_enabled and new_groups:
                    # ===== P3: submit all producer groups in the capture step =====
                    # Payloads point at this step's graph-capture logits buffers.
                    # Queueing raw payload groups across replay steps lets the next
                    # graph overwrite those buffers before selector/writer starts.
                    # Submit every group to the pending-rebuild producer now; the
                    # pending path can still delay publish/accept without rereading
                    # overwritten capture storage.
                    for group in new_groups:
                        pending_count += int(
                            _enqueue_one_replay_refresh_payload_group(
                                controller=controller,
                                group=group,
                                result=None,
                                workspace=workspace,
                                enqueue_pending=enqueue_pending,
                                map_layer=map_layer,
                                deadline_slack_steps=int(delay_steps),
                                stage_profile=stage_profile,
                                profile_detail=_REPLAY_REFRESH_ENQUEUE_PROFILE_DETAIL_CACHED,
                            )
                        )
                else:
                    # Stagger disabled OR no new groups: legacy unconditional loop.
                    for group in new_groups:
                        pending_count += int(
                            _enqueue_one_replay_refresh_payload_group(
                                controller=controller,
                                group=group,
                                result=None,
                                workspace=workspace,
                                enqueue_pending=enqueue_pending,
                                map_layer=map_layer,
                                stage_profile=stage_profile,
                                profile_detail=_REPLAY_REFRESH_ENQUEUE_PROFILE_DETAIL_CACHED,
                            )
                        )
                grouped_async_loop_ok = True
            finally:
                if grouped_async_started:
                    if grouped_async_loop_ok:
                        # [STAGE-SPLIT 2026-07-06] finish 跑的是全部延迟 body
                        # （3×selector+writer launch，~3.1ms/世代实测），旧口径
                        # 混进 deferred_pending_rebuild_enqueue_us 使 "enqueue"
                        # 读数被误读为 bookkeeping。单列，enqueue 减法可得纯
                        # bookkeeping；多源 selector 融合（削减刀 #1）的前后
                        # 对照直接看本字段。
                        _t_body0_ns = time.perf_counter_ns()
                        grouped_async_submitted = int(
                            controller._finish_pending_refresh_grouped_async_envelope()
                        )
                        if stage_profile_enabled:
                            _stage_add(
                                "grouped_async_body_launch_us",
                                (time.perf_counter_ns() - _t_body0_ns) / 1000.0,
                            )
                    else:
                        controller._abort_pending_refresh_grouped_async_envelope()
            if stage_profile_enabled and grouped_async_started:
                _stage_inc("grouped_async_envelope_count", 1)
                _stage_inc(
                    "grouped_async_envelope_submitted_count",
                    int(grouped_async_submitted),
                )
            if pending_count + queued_count <= 0:
                raise RuntimeError(
                    "full cudagraph replay refresh deadline deferral enqueued nothing"
                )
            if stage_profile_enabled:
                _stage_inc("deferred_pending_rebuild_count", pending_count)
                _stage_inc("deferred_replay_refresh_queued_count", queued_count)
                stage_profile["deferred_pending_rebuild"] = True
        else:
            flushed = 0
            for payload_group in payload_groups:
                flushed += _flush_full_cudagraph_refresh_payloads_batched_after_replay(
                    controller=controller,
                    payloads=payload_group,
                    stage_profile=stage_profile,
                )
            if int(flushed) != int(enqueued):
                raise RuntimeError(
                    "full cudagraph batched replay refresh flushed payload count mismatch: "
                    f"enqueued={int(enqueued)} flushed={int(flushed)}"
                )
        if stage_profile_enabled:
            _flush_us = (time.perf_counter_ns() - _t_flush0_ns) / 1000.0
            if defer_to_deadline:
                _stage_add("deferred_pending_rebuild_enqueue_us", _flush_us)
            else:
                _stage_add("flush_us", _flush_us)
                _stage_max("flush_us_max", _flush_us)
                _stage_inc("flush_call_count", amount=len(payload_groups))
            stage_profile["batched_flush"] = True
            # [PROFILE-CARRIER-HINT 2026-07-06] REFRESH_PROFILE 的 per-chunk
            # 载体只挂 eager flush 路径；本 replay-batched 路径下 refresh
            # profile log 恒无 per-chunk 行（只有 marker），事件级分解应看
            # hook_profile 的 refresh_stage_profile。曾三次误导跑批取证——
            # 开着旋钮走到这里就提示一次。
            if os.environ.get("VLLM_SPARSE_REFRESH_PROFILE", "0") == "1" and not getattr(
                controller, "_refresh_profile_batched_hint_emitted", False
            ):
                controller._refresh_profile_batched_hint_emitted = True
                _log.info(
                    "VLLM_SPARSE_REFRESH_PROFILE per-chunk records are inactive on "
                    "the replay-batched flush path; use the hook_profile "
                    "refresh_stage_profile fields instead"
                )
            stage_profile["batched_flush_group_count"] = int(
                submit_group_count_for_profile
            )
            layer_indices = [
                int(getattr(getattr(payload, "state", None), "layer_index", -1))
                for payload in batched_payloads
            ]
            layer_indices = [layer for layer in layer_indices if layer >= 0]
            num_chunks = (len(layer_keys) + _CAPTURE_CHUNK - 1) // _CAPTURE_CHUNK
            refresh_rebuild_max_delay = getattr(
                controller,
                "_refresh_rebuild_max_delay_steps",
                None,
            )
            max_delay_steps = (
                int(delay_steps)
                if defer_to_deadline
                else (
                    int(refresh_rebuild_max_delay(num_chunks))
                    if callable(refresh_rebuild_max_delay)
                    else 0
                )
            )
            request_states = getattr(controller, "request_states", None)
            work_item = build_refresh_producer_work_item(
                payloads=batched_payloads,
                request_states=request_states if isinstance(request_states, dict) else None,
                layer_indices=layer_indices,
                max_delay_steps=max_delay_steps,
                current_epoch=int(getattr(step_ctx, "epoch", -1)),
                current_handle_id=int(getattr(step_ctx, "step_handle_id", -1)),
                admission_reason="direct_replay_refresh",
                can_drop=True,
                can_coalesce=True,
            )
            stage_profile["producer_work_items_with_deadline"] = int(
                len(new_groups) if defer_to_deadline else 1
            )
            stage_profile["producer_work_target_layer_start"] = int(
                work_item.target_layer_start
            )
            stage_profile["producer_work_target_layer_end"] = int(
                work_item.target_layer_end
            )
            stage_profile["producer_work_decode_step_min"] = int(
                work_item.decode_step_min
            )
            stage_profile["producer_work_decode_step_max"] = int(
                work_item.decode_step_max
            )
            stage_profile["producer_work_deadline_slack_steps"] = int(
                work_item.deadline_slack_steps
            )
            stage_profile["producer_work_admission_reason"] = (
                "deferred_replay_refresh" if defer_to_deadline else str(work_item.admission_reason)
            )
        if getattr(controller, "_refresh_layer_group_enabled", False) and getattr(
            controller,
            "_refresh_layer_group_any_refresh",
            False,
        ):
            try:
                controller._refresh_layer_group_event_idx += 1
            except Exception:
                setattr(controller, "_refresh_layer_group_event_idx", 0)
                raise
        setattr(controller, "_refresh_layer_group_any_refresh", False)
    if route_trace_enabled:
        try:
            from patches.fa3_native.install import append_fa3_route_trace

            _sor = getattr(controller, "_selected_out_ring", None)
            append_fa3_route_trace(
                {
                    "event": "mixed_page_full_cudagraph_replay_refresh_payload_enqueue",
                    "step_id": int(getattr(step_ctx, "epoch", -1)),
                    "graph_key": str(graph_key),
                    "payload_count": int(enqueued),
                    "refresh_slot_count": int(len(refresh_slot_list)),
                    "refresh_reason": str(refresh_reason),
                    "refresh_intent_req_count": int(len(refresh_intent_req_ids)),
                    "refresh_intent_req_ids": list(refresh_intent_req_ids),
                    "refresh_intent_debug": _route_refresh_intent_debug(),
                    # [RING-RELEASE-ON-WRITER-SUBMIT 遥测] speed child 视角的
                    # 环健康累计(REFRESH_CPROFILE 只覆盖 diagnostic child 的
                    # eager flush,pending 路径无 pstats 出口——ring 真实
                    # spill 率以本字段为准;稳态合同=spill 恒 0)。
                    "selected_out_ring_acquire_count": (
                        int(getattr(_sor, "acquire_count", -1)) if _sor is not None else -1
                    ),
                    "selected_out_ring_spill_count": (
                        int(getattr(_sor, "spill_run_count", -1)) if _sor is not None else -1
                    ),
                }
            )
        except Exception:
            pass
    if stage_profile is not None:
        direct_accounted_us = (
            float(stage_profile.get("prepare_logits_us", 0.0) or 0.0)
            + float(stage_profile.get("payload_build_us", 0.0) or 0.0)
            + float(stage_profile.get("slot_req_ids_us", 0.0) or 0.0)
            + float(stage_profile.get("payload_enqueue_us", 0.0) or 0.0)
            + float(
                stage_profile.get("deferred_pending_rebuild_enqueue_us", 0.0)
                or 0.0
            )
            + float(stage_profile.get("flush_us", 0.0) or 0.0)
        )
        total_us = (time.perf_counter_ns() - stage_total_start_ns) / 1000.0
        stage_profile["direct_accounted_us"] = float(direct_accounted_us)
        stage_profile["total_us"] = float(total_us)
        stage_profile["residual_us"] = float(max(0.0, total_us - direct_accounted_us))
        setattr(
            controller,
            "_mixed_page_full_cudagraph_last_replay_refresh_stage_profile",
            dict(stage_profile),
        )
    return int(enqueued)


def _current_model_forward_refresh_identity(
    controller: object,
    *,
    stage: str,
) -> tuple[int, int, int, int]:
    step_ctx = getattr(controller, "step_context", None)
    step_authority = getattr(step_ctx, "step_authority", None)
    if step_authority is None:
        step_authority = getattr(controller, "step_authority", None)
    if step_ctx is None or step_authority is None:
        raise RuntimeError(f"{stage} requires step context and step authority")
    context_identity = (
        int(getattr(step_ctx, "epoch", -1)),
        int(getattr(step_ctx, "step_handle_id", -1)),
        int(getattr(step_ctx, "step_handle_generation", -1)),
        int(getattr(step_ctx, "step_identity_token", -1)),
    )
    authority_identity = (
        int(getattr(step_authority, "epoch", -1)),
        int(getattr(step_authority, "step_handle_id", -1)),
        int(getattr(step_authority, "step_handle_generation", -1)),
        int(getattr(step_authority, "step_identity_token", -1)),
    )
    if (
        any(value <= 0 for value in context_identity)
        or context_identity != authority_identity
    ):
        raise RuntimeError(
            f"{stage} has inconsistent step identity: "
            f"context={context_identity!r} authority={authority_identity!r}"
        )
    expected_token = (
        context_identity[0] * 1_000_000_000
        + context_identity[1] * 1_000_000
        + context_identity[2]
    )
    if context_identity[3] != expected_token:
        raise RuntimeError(
            f"{stage} has invalid step identity token: "
            f"identity={context_identity!r} expected_token={expected_token}"
        )
    return context_identity


def _mark_model_forward_refresh_generation_ready(
    *,
    controller: object,
    graph_key: str,
) -> None:
    """Record that one graph segment replayed for the current model forward."""
    if not _full_cudagraph_replay_step_has_refresh_row(controller):
        return
    if not bool(getattr(controller, _MODEL_FORWARD_REFRESH_ACTIVE_ATTR, False)):
        raise RuntimeError(
            "model-forward refresh generation ready mark occurred outside the "
            "model-forward owner"
        )
    identity = _current_model_forward_refresh_identity(
        controller,
        stage="model-forward refresh generation ready mark",
    )
    current = getattr(controller, _MODEL_FORWARD_REFRESH_READY_ATTR, None)
    if current is None:
        setattr(
            controller,
            _MODEL_FORWARD_REFRESH_READY_ATTR,
            {"identity": identity, "graph_key": str(graph_key)},
        )
        return
    if not isinstance(current, dict) or tuple(current.get("identity", ())) != identity:
        raise RuntimeError(
            "model-forward refresh generation ready identity changed within one "
            f"forward: previous={current!r} current={identity!r}"
        )
    if not str(current.get("graph_key", "") or "") and graph_key:
        current["graph_key"] = str(graph_key)


def _patch_model_forward_refresh_owner() -> None:
    """Move generation-wide producer work behind the complete model forward."""
    global _MODEL_FORWARD_REFRESH_OWNER_PATCHED
    global _ORIGINAL_MODEL_FORWARD_FOR_REFRESH_OWNER
    if _MODEL_FORWARD_REFRESH_OWNER_PATCHED:
        return
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner  # type: ignore[import]
        original_model_forward = GPUModelRunner._model_forward  # type: ignore[attr-defined]
    except Exception as exc:
        raise RuntimeError(
            "sparse patch requires GPUModelRunner._model_forward for the "
            "single-owner refresh lifecycle"
        ) from exc
    if not callable(original_model_forward):
        raise RuntimeError(
            "sparse patch requires callable GPUModelRunner._model_forward"
        )
    ready_attr = _MODEL_FORWARD_REFRESH_READY_ATTR
    active_attr = _MODEL_FORWARD_REFRESH_ACTIVE_ATTR

    def _sparse_model_forward(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        controller = _GLOBAL_CONTROLLER
        if controller is None:
            return original_model_forward(self, *args, **kwargs)
        stale_ready = getattr(controller, ready_attr, None)
        if stale_ready is not None:
            raise RuntimeError(
                "model-forward refresh owner found an unconsumed generation at "
                f"forward entry: ready={stale_ready!r}"
            )
        if bool(getattr(controller, active_attr, False)):
            raise RuntimeError(
                "model-forward refresh owner does not support nested entry"
            )
        staged_intents = ()
        if getattr(controller, "_deferred_bootstrap_launch_intents", None):
            take_staged_intents = getattr(
                controller,
                "_take_staged_deferred_bootstrap_producer_jobs",
                None,
            )
            drain_staged_intents = getattr(
                controller,
                "_drain_staged_deferred_bootstrap_producer_jobs",
                None,
            )
            fail_staged_intents = getattr(
                controller,
                "_fail_deferred_bootstrap_launch_intents",
                None,
            )
            if not all(
                callable(method)
                for method in (
                    take_staged_intents,
                    drain_staged_intents,
                    fail_staged_intents,
                )
            ):
                raise RuntimeError(
                    "model-forward owner requires deferred producer ownership methods"
                )
            staged_intents = tuple(take_staged_intents())
        setattr(controller, active_attr, True)
        try:
            result = original_model_forward(self, *args, **kwargs)
        except Exception as exc:
            setattr(controller, ready_attr, None)
            setattr(controller, active_attr, False)
            if staged_intents:
                assert callable(fail_staged_intents)
                fail_staged_intents(
                    staged_intents,
                    reason=f"model forward failed before deferred producer drain: {exc}"
                )
            raise
        setattr(controller, active_attr, False)
        ready = getattr(controller, ready_attr, None)
        setattr(controller, ready_attr, None)
        deferred_launch_count = 0
        if staged_intents:
            assert callable(drain_staged_intents)
            deferred_launch_count = int(
                drain_staged_intents(intents=staged_intents)
            )
        if not isinstance(ready, dict):
            return result

        current_identity = _current_model_forward_refresh_identity(
            controller,
            stage="model-forward refresh owner completion",
        )
        ready_identity = tuple(ready.get("identity", ()))
        if ready_identity != current_identity:
            raise RuntimeError(
                "model-forward refresh owner identity mismatch: "
                f"ready={ready_identity!r} current={current_identity!r}"
            )

        refresh_enabled = _mixed_page_full_cudagraph_replay_refresh_enabled(
            controller
        )
        if not refresh_enabled:
            raise RuntimeError(
                "model-forward refresh generation was marked while refresh is disabled"
            )
        profile_path = _full_cudagraph_hook_profile_log(refresh_enabled)
        start_ns = time.perf_counter_ns() if profile_path else 0
        payload_count = _enqueue_full_cudagraph_refresh_payloads_after_replay(
            controller=controller,
            graph_key=str(ready.get("graph_key", "") or ""),
        )
        refresh_us = (
            (time.perf_counter_ns() - start_ns) / 1000.0 if profile_path else 0.0
        )
        if profile_path:
            stage_profile = getattr(
                controller,
                "_mixed_page_full_cudagraph_last_replay_refresh_stage_profile",
                None,
            )
            _append_mixed_page_full_cudagraph_profile_event(
                profile_path,
                {
                    "event": "mixed_page_full_cudagraph_model_forward_refresh",
                    "hook": "model_forward_completion",
                    "step_id": int(current_identity[0]),
                    "step_handle_id": int(current_identity[1]),
                    "step_handle_generation": int(current_identity[2]),
                    "step_identity_token": int(current_identity[3]),
                    "graph_key": str(ready.get("graph_key", "") or ""),
                    "refresh_called": bool(payload_count > 0),
                    "post_replay_refresh_payloads": int(payload_count),
                    "post_forward_deferred_producer_launches": int(
                        deferred_launch_count
                    ),
                    "refresh_us": float(refresh_us),
                    "refresh_stage_profile": (
                        dict(stage_profile)
                        if isinstance(stage_profile, dict)
                        else None
                    ),
                },
            )
        return result

    _ORIGINAL_MODEL_FORWARD_FOR_REFRESH_OWNER = original_model_forward
    GPUModelRunner._model_forward = _sparse_model_forward  # type: ignore[assignment]
    _MODEL_FORWARD_REFRESH_OWNER_PATCHED = True


def _patch_cuda_graph_wrapper_for_sparse_cudagraph() -> None:
    global _CUDAGRAPH_WRAPPER_PATCHED, _ORIGINAL_CUDAGRAPH_WRAPPER_CALL
    if _CUDAGRAPH_WRAPPER_PATCHED:
        return
    try:
        from vllm.compilation.cuda_graph import CUDAGraphWrapper  # type: ignore[import]
        from vllm.config import CUDAGraphMode  # type: ignore[import]
        from vllm.forward_context import (  # type: ignore[import]
            get_forward_context,
            is_forward_context_available,
        )
    except Exception:
        _log.warning(
            "Cannot import CUDAGraphWrapper for sparse full cudagraph replay hook, skipping"
        )
        return

    original_call = CUDAGraphWrapper.__call__

    def _sparse_cudagraph_call(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        controller = _GLOBAL_CONTROLLER
        refresh_enabled = _mixed_page_full_cudagraph_replay_refresh_enabled(controller)
        release_pending = bool(
            getattr(controller, "_refresh_producer_stream_release_pending", False)
        )
        profile_path = _full_cudagraph_hook_profile_log(refresh_enabled)
        cuda_event_path = _full_cudagraph_replay_cuda_event_log(
            refresh_enabled
        )
        profile_enabled = bool(profile_path)
        route_trace_enabled = _fa3_route_trace_enabled()
        diagnostic_enabled = bool(profile_enabled or cuda_event_path or route_trace_enabled)
        if (
            not refresh_enabled
            and not release_pending
            and not profile_path
            and not cuda_event_path
            and not route_trace_enabled
        ):
            return original_call(self, *args, **kwargs)
        profile_total_start_ns = time.perf_counter_ns() if profile_enabled else 0
        profile_refresh_us = 0.0
        profile_refresh_called = False
        profile_payload_enqueue_count = 0
        profile_refresh_stage_profile = None
        profile_pre_consume_drain_us = 0.0
        profile_pre_consume_drain_stats = None
        profile_pre_timing = _new_full_cudagraph_pre_timing(profile_enabled)
        profile_ready_event_wait_count = 0
        profile_async_refresh_wait_count = 0
        profile_reason = "refresh_disabled" if not refresh_enabled else ""
        profile_graph_key = ""
        profile_entry_present = None
        profile_cudagraph_present = None
        profile_batch_descriptor = None
        profile_forward_runtime_mode = None
        profile_wrapper_runtime_mode = None
        if diagnostic_enabled:
            _t_step_identity_ns = _full_cudagraph_pre_timing_start(profile_pre_timing)
            profile_step_identity_payload = _mixed_page_full_cudagraph_step_identity_payload(controller)
            _full_cudagraph_pre_timing_add(
                profile_pre_timing,
                "pre_step_identity_us",
                _t_step_identity_ns,
            )
            _t_initial_bind_identity_ns = _full_cudagraph_pre_timing_start(
                profile_pre_timing
            )
            profile_prebound_identity_payload = _mixed_page_full_cudagraph_bind_identity_payload(None)
            _full_cudagraph_pre_timing_add(
                profile_pre_timing,
                "pre_initial_bind_identity_us",
                _t_initial_bind_identity_ns,
            )
        else:
            profile_step_identity_payload = {}
            profile_prebound_identity_payload = {}
        profile_prebound_rrp_graph_state = None
        profile_prebound_rrp_replay_proof = None
        _t_forward_context_ns = _full_cudagraph_pre_timing_start(profile_pre_timing)
        forward_context_available = bool(is_forward_context_available())
        pre_call_batch_descriptor = None
        pre_call_had_cudagraph = False
        route_family_mismatch = False
        captured_route_family = None
        current_route_family = None
        bridge_graph_policy = _bootstrap_bridge_graph_policy()
        if refresh_enabled and forward_context_available:
            forward_context = get_forward_context()
            cudagraph_runtime_mode = getattr(
                forward_context, "cudagraph_runtime_mode", None
            )
            wrapper_runtime_mode = getattr(self, "runtime_mode", None)
            if profile_enabled:
                profile_forward_runtime_mode = (
                    getattr(cudagraph_runtime_mode, "name", None)
                    or str(cudagraph_runtime_mode)
                )
                profile_wrapper_runtime_mode = (
                    getattr(wrapper_runtime_mode, "name", None)
                    or str(wrapper_runtime_mode)
                )
            _full_cudagraph_pre_timing_add(
                profile_pre_timing,
                "pre_forward_context_us",
                _t_forward_context_ns,
            )
            if cudagraph_runtime_mode == CUDAGraphMode.FULL and wrapper_runtime_mode == CUDAGraphMode.FULL:
                _t_graph_lookup_ns = _full_cudagraph_pre_timing_start(
                    profile_pre_timing
                )
                _t_graph_entry_lookup_ns = _full_cudagraph_pre_timing_start(
                    profile_pre_timing
                )
                batch_descriptor = getattr(forward_context, "batch_descriptor", None)
                pre_call_batch_descriptor = batch_descriptor
                entries = getattr(self, "concrete_cudagraph_entries", {})
                entry = (
                    entries.get(batch_descriptor)
                    if batch_descriptor is not None and hasattr(entries, "get")
                    else None
                )
                profile_batch_descriptor = (
                    _mixed_page_graph_entry_batch_descriptor_repr(
                        entry,
                        batch_descriptor,
                    )
                    if diagnostic_enabled
                    else None
                )
                cudagraph_present = bool(
                    entry is not None and getattr(entry, "cudagraph", None) is not None
                )
                pre_call_had_cudagraph = cudagraph_present
                profile_entry_present = entry is not None
                profile_cudagraph_present = cudagraph_present
                _full_cudagraph_pre_timing_add(
                    profile_pre_timing,
                    "pre_graph_entry_lookup_us",
                    _t_graph_entry_lookup_ns,
                )
                _t_graph_capture_family_ns = _full_cudagraph_pre_timing_start(
                    profile_pre_timing
                )
                captured_route_family = (
                    getattr(entry, "_sfi_mixed_page_capture_route_family", None)
                    if entry is not None
                    else None
                )
                _full_cudagraph_pre_timing_add(
                    profile_pre_timing,
                    "pre_graph_capture_family_us",
                    _t_graph_capture_family_ns,
                )
                prevalidated_rrp_graph_state = None
                current_route_metadata_items = ()
                if cudagraph_present and captured_route_family == "resolved_row_ptr":
                    prevalidated_rrp_graph_state = (
                        _prebound_rrp_current_graph_state_for_forward_context(
                            controller=controller,
                            forward_context=forward_context,
                        )
                    )
                if prevalidated_rrp_graph_state is not None:
                    current_route_family = "resolved_row_ptr"
                else:
                    _t_graph_route_family_ns = _full_cudagraph_pre_timing_start(
                        profile_pre_timing
                    )
                    (
                        current_route_family,
                        current_route_metadata_items,
                    ) = _mixed_page_full_cudagraph_route_family_and_metadata_items(
                        forward_context,
                        controller=controller,
                    )
                    if current_route_family == "native_or_capture":
                        rebound_current_metadata = _maybe_rebind_rrp_for_full_graph_replay(
                            controller=controller,
                            forward_context=forward_context,
                            metadata_items=current_route_metadata_items,
                        )
                        if rebound_current_metadata:
                            (
                                current_route_family,
                                current_route_metadata_items,
                            ) = _mixed_page_full_cudagraph_route_family_and_metadata_items(
                                forward_context,
                                controller=controller,
                            )
                    _full_cudagraph_pre_timing_add(
                        profile_pre_timing,
                        "pre_graph_route_family_us",
                        _t_graph_route_family_ns,
                    )
                _full_cudagraph_pre_timing_add(
                    profile_pre_timing,
                    "pre_graph_lookup_us",
                    _t_graph_lookup_ns,
                )
                if (
                    cudagraph_present
                    and entry is not None
                    and _mixed_page_full_cudagraph_route_family_mismatch(
                        captured_route_family=captured_route_family,
                        current_route_family=current_route_family,
                    )
                ):
                    route_family_mismatch = True
                    setattr(
                        controller,
                        "_mixed_page_full_cudagraph_graph_route_family_mismatch",
                        {
                            "batch_descriptor": repr(batch_descriptor),
                            "captured_route_family": captured_route_family,
                            "current_route_family": current_route_family,
                        },
                    )
                    if bridge_graph_policy == "evict_recapture_once":
                        descriptor_key = repr(batch_descriptor)
                        recapture_count_by_descriptor = getattr(
                            controller,
                            "_sfi_mixed_page_route_family_recapture_count_by_descriptor",
                            None,
                        )
                        if not isinstance(recapture_count_by_descriptor, dict):
                            recapture_count_by_descriptor = {}
                            setattr(
                                controller,
                                "_sfi_mixed_page_route_family_recapture_count_by_descriptor",
                                recapture_count_by_descriptor,
                            )
                        recapture_count = int(
                            recapture_count_by_descriptor.get(descriptor_key, 0)
                            or 0
                        )
                        if recapture_count > 0:
                            raise RuntimeError(
                                "route-family recapture already used for this request batch"
                            )
                        try:
                            entries.pop(batch_descriptor, None)
                        except Exception as exc:
                            raise RuntimeError(
                                "failed to evict stale route-family graph entry"
                            ) from exc
                        setattr(
                            controller,
                            "_sfi_mixed_page_route_family_recapture_count",
                            int(
                                getattr(
                                    controller,
                                    "_sfi_mixed_page_route_family_recapture_count",
                                    0,
                                )
                                or 0
                            )
                            + 1,
                        )
                        recapture_count_by_descriptor[descriptor_key] = recapture_count + 1
                        route_family_mismatch = False
                        cudagraph_present = False
                        pre_call_had_cudagraph = False
                        profile_entry_present = False
                        profile_cudagraph_present = False
                        profile_reason = "route_family_mismatch_evict_recapture_once"
                    mismatch_trace_reason = (
                        "route_family_mismatch_evict_recapture_once:"
                        f"{captured_route_family}->{current_route_family}"
                        if profile_reason
                        == "route_family_mismatch_evict_recapture_once"
                        else (
                            "graph_route_family_mismatch_skip_refresh:"
                            f"{captured_route_family}->{current_route_family}"
                        )
                    )
                    if route_trace_enabled:
                        _append_mixed_page_full_cudagraph_hook_trace(
                            hook="cuda_graph_wrapper",
                            reason=mismatch_trace_reason,
                            controller=controller,
                            forward_context_available=forward_context_available,
                            forward_context=forward_context,
                            wrapper_runtime_mode=wrapper_runtime_mode,
                            batch_descriptor=batch_descriptor,
                            entry_present=True,
                            cudagraph_present=cudagraph_present,
                            captured_route_family=captured_route_family,
                            current_route_family=current_route_family,
                            route_family_mismatch=route_family_mismatch,
                            bridge_graph_policy=bridge_graph_policy,
                        )
                if profile_reason != "route_family_mismatch_evict_recapture_once":
                    profile_reason = (
                        "refresh"
                        if cudagraph_present and not route_family_mismatch
                        else "graph_route_family_mismatch_skip_refresh"
                        if route_family_mismatch
                        else "waiting_for_cudagraph_capture"
                    )
                if not (cudagraph_present and not route_family_mismatch):
                    if route_trace_enabled:
                        _append_mixed_page_full_cudagraph_hook_trace(
                            hook="cuda_graph_wrapper",
                            reason=profile_reason,
                            controller=controller,
                            forward_context_available=forward_context_available,
                            forward_context=forward_context,
                            wrapper_runtime_mode=wrapper_runtime_mode,
                            batch_descriptor=batch_descriptor,
                            entry_present=profile_entry_present,
                            cudagraph_present=cudagraph_present,
                            captured_route_family=captured_route_family,
                            current_route_family=current_route_family,
                            route_family_mismatch=route_family_mismatch,
                            bridge_graph_policy=bridge_graph_policy,
                        )
                if cudagraph_present and not route_family_mismatch:
                    _t_graph_key_ns = _full_cudagraph_pre_timing_start(
                        profile_pre_timing
                    )
                    profile_graph_key = _mixed_page_full_graph_key_for_entry(
                        entry,
                        batch_descriptor,
                    )
                    _full_cudagraph_pre_timing_add(
                        profile_pre_timing,
                        "pre_graph_key_us",
                        _t_graph_key_ns,
                    )
                    _t_prebound_graph_state_ns = _full_cudagraph_pre_timing_start(
                        profile_pre_timing
                    )
                    prebound_rrp_graph_state = (
                        prevalidated_rrp_graph_state
                        if prevalidated_rrp_graph_state is not None
                        else _mixed_page_full_cudagraph_prebound_rrp_graph_state(
                            controller=controller,
                            captured_route_family=captured_route_family,
                            current_route_family=current_route_family,
                        )
                    )
                    _full_cudagraph_pre_timing_add(
                        profile_pre_timing,
                        "pre_prebound_graph_state_us",
                        _t_prebound_graph_state_ns,
                    )
                    profile_prebound_rrp_graph_state = prebound_rrp_graph_state
                    if diagnostic_enabled:
                        _t_prebound_bind_identity_ns = (
                            _full_cudagraph_pre_timing_start(profile_pre_timing)
                        )
                        profile_prebound_identity_payload = (
                            _mixed_page_full_cudagraph_bind_identity_payload(
                                prebound_rrp_graph_state
                            )
                        )
                        _full_cudagraph_pre_timing_add(
                            profile_pre_timing,
                            "pre_prebound_bind_identity_us",
                            _t_prebound_bind_identity_ns,
                        )
                    if prebound_rrp_graph_state is not None:
                        profile_reason = "prebound_rrp_graph_state"
                        if route_trace_enabled:
                            _append_mixed_page_full_cudagraph_hook_trace(
                                hook="cuda_graph_wrapper",
                                reason=profile_reason,
                                controller=controller,
                                forward_context_available=forward_context_available,
                                forward_context=forward_context,
                                wrapper_runtime_mode=wrapper_runtime_mode,
                                batch_descriptor=batch_descriptor,
                                entry_present=profile_entry_present,
                                cudagraph_present=cudagraph_present,
                                captured_route_family=captured_route_family,
                                current_route_family=current_route_family,
                                route_family_mismatch=route_family_mismatch,
                                bridge_graph_policy=bridge_graph_policy,
                            )
                        _evt_bisect_mark("pre_drain", controller)
                        if _full_cudagraph_replay_refresh_defer_to_deadline_enabled(
                            controller
                        ):
                            (
                                profile_pre_consume_drain_stats,
                                profile_pre_consume_drain_us,
                            ) = _drain_full_cudagraph_pending_refresh_before_replay_profiled(
                                controller=controller,
                                profile_path=profile_path,
                            )
                        _evt_bisect_mark("post_drain", controller)
                        _t_prebound_mark_ns = _full_cudagraph_pre_timing_start(
                            profile_pre_timing
                        )
                        prebound_stats = _mark_prebound_rrp_full_cudagraph_replay(
                            controller=controller,
                            forward_context=forward_context,
                            graph_key=profile_graph_key,
                            profile_pre_timing=profile_pre_timing,
                            state=prebound_rrp_graph_state,
                            metadata_items=current_route_metadata_items,
                            binding_is_current_prevalidated=bool(
                                prevalidated_rrp_graph_state is not None
                            ),
                        )
                        profile_prebound_rrp_graph_state = (
                            prebound_stats.replay_generation_state
                        )
                        profile_prebound_rrp_replay_proof = prebound_stats
                        _evt_bisect_mark("post_prebind", controller)
                        _full_cudagraph_pre_timing_add(
                            profile_pre_timing,
                            "pre_prebound_mark_us",
                            _t_prebound_mark_ns,
                        )
                        profile_ready_event_wait_count = int(
                            getattr(prebound_stats, "ready_event_wait_count", 0)
                            or 0
                        )
                        _t_async_refresh_probe_ns = (
                            _full_cudagraph_pre_timing_start(profile_pre_timing)
                            if profile_pre_timing is not None
                            else 0
                        )
                        profile_async_refresh_wait_count = int(
                            _wait_pending_async_refresh_before_full_cudagraph_replay(
                                controller=controller,
                                profile_pre_timing=profile_pre_timing,
                            )
                        )
                        if profile_pre_timing is not None:
                            _full_cudagraph_pre_timing_add(
                                profile_pre_timing,
                                "pre_async_refresh_probe_us",
                                _t_async_refresh_probe_ns,
                            )
                        if route_trace_enabled:
                            _append_prebound_rrp_full_cudagraph_replay_trace(
                                controller=controller,
                                forward_context=forward_context,
                                stats=prebound_stats,
                            )
                    elif current_route_family == "resolved_row_ptr":
                        _raise_missing_prebound_rrp_graph_state(
                            captured_route_family=captured_route_family,
                            current_route_family=current_route_family,
                        )
                    else:
                        if route_trace_enabled:
                            _append_mixed_page_full_cudagraph_hook_trace(
                                hook="cuda_graph_wrapper",
                                reason=profile_reason,
                                controller=controller,
                                forward_context_available=forward_context_available,
                                forward_context=forward_context,
                                wrapper_runtime_mode=wrapper_runtime_mode,
                                batch_descriptor=batch_descriptor,
                                entry_present=profile_entry_present,
                                cudagraph_present=cudagraph_present,
                                captured_route_family=captured_route_family,
                                current_route_family=current_route_family,
                                route_family_mismatch=route_family_mismatch,
                                bridge_graph_policy=bridge_graph_policy,
                            )
                        profile_refresh_called = True
                        profile_refresh_start_ns = time.perf_counter_ns() if profile_enabled else 0
                        _refresh_mixed_page_full_cudagraph_replay_for_forward_context(
                            controller=controller,
                            forward_context=forward_context,
                            graph_key=profile_graph_key,
                        )
                        if profile_enabled:
                            profile_refresh_us += (
                                time.perf_counter_ns() - profile_refresh_start_ns
                            ) / 1000.0
            else:
                if (
                    not release_pending
                    and not profile_path
                    and not cuda_event_path
                    and not route_trace_enabled
                ):
                    try:
                        return original_call(self, *args, **kwargs)
                    finally:
                        # [2026-07-12 RRP-DONE-EVTS-RETIRED] per-call done-event
                        # record deleted: zero wait/query consumers repo-wide.
                        if refresh_enabled or release_pending:
                            _release_refresh_producer_after_decode_if_pending(
                                controller
                            )
                profile_reason = "runtime_mode_mismatch"
                if route_trace_enabled:
                    _append_mixed_page_full_cudagraph_hook_trace(
                        hook="cuda_graph_wrapper",
                        reason="runtime_mode_mismatch",
                        controller=controller,
                        forward_context_available=forward_context_available,
                        forward_context=forward_context,
                        wrapper_runtime_mode=wrapper_runtime_mode,
                        batch_descriptor=getattr(forward_context, "batch_descriptor", None),
                    )
                if profile_enabled or cuda_event_path:
                    profile_batch_descriptor = repr(
                        getattr(forward_context, "batch_descriptor", None)
                    )
        else:
            profile_reason = (
                "refresh_disabled"
                if not refresh_enabled
                else "forward_context_unavailable"
            )
            if route_trace_enabled:
                _append_mixed_page_full_cudagraph_hook_trace(
                    hook="cuda_graph_wrapper",
                    reason=profile_reason,
                    controller=controller,
                    forward_context_available=forward_context_available,
                )
        profile_entry_for_payload = locals().get("entry")
        profile_original_start_ns = time.perf_counter_ns() if profile_enabled else 0
        profile_cuda_event_sample = (
            _begin_full_cudagraph_replay_cuda_event(
                path=cuda_event_path,
                reason=str(profile_reason),
                step_id=_mixed_page_full_cudagraph_profile_step_id(controller),
                graph_key=profile_graph_key,
                batch_descriptor=profile_batch_descriptor,
                state=profile_prebound_rrp_graph_state,
                step_identity_payload=profile_step_identity_payload,
                prebound_identity_payload=profile_prebound_identity_payload,
            )
            if cuda_event_path
            else None
        )
        try:
            result = original_call(self, *args, **kwargs)
        finally:
            profile_original_end_ns = time.perf_counter_ns() if profile_enabled else 0
            _end_full_cudagraph_replay_cuda_event(profile_cuda_event_sample)
            if refresh_enabled or release_pending:
                _release_refresh_producer_after_decode_if_pending(controller)
        _evt_bisect_mark("post_replay", controller)
        if profile_reason == "prebound_rrp_graph_state":
            _record_prebound_rrp_replay_consumed_generation(
                controller=controller,
                state=profile_prebound_rrp_graph_state,
                replay_proof=profile_prebound_rrp_replay_proof,
            )
        if (
            refresh_enabled
            and pre_call_had_cudagraph
            and profile_reason == "prebound_rrp_graph_state"
        ):
            if (
                not _REPLAY_REFRESH_NOOP_FAST_SKIP_CACHED
                or _full_cudagraph_replay_step_has_refresh_row(controller)
            ):
                _mark_model_forward_refresh_generation_ready(
                    controller=controller,
                    graph_key=profile_graph_key,
                )
            _evt_bisect_mark("post_deferred", controller)
        if refresh_enabled and forward_context_available:
            try:
                forward_context_after = get_forward_context()
                if (
                    getattr(forward_context_after, "cudagraph_runtime_mode", None)
                    == CUDAGraphMode.FULL
                    and getattr(self, "runtime_mode", None) == CUDAGraphMode.FULL
                ):
                    batch_descriptor_after = getattr(
                        forward_context_after,
                        "batch_descriptor",
                        None,
                    )
                    entries_after = getattr(self, "concrete_cudagraph_entries", {})
                    entry_after = (
                        entries_after.get(batch_descriptor_after)
                        if batch_descriptor_after is not None
                        and hasattr(entries_after, "get")
                        else None
                    )
                    if (
                        entry_after is not None
                        and getattr(entry_after, "cudagraph", None) is not None
                        and not pre_call_had_cudagraph
                        and batch_descriptor_after == pre_call_batch_descriptor
                    ):
                        inferred_route_family = _mixed_page_full_cudagraph_route_family(
                            forward_context_after,
                            controller=controller,
                        )
                        captured_actual_route_family = (
                            _mixed_page_actual_route_family_for_capture_tag(
                                controller=controller,
                                default_route_family=inferred_route_family,
                            )
                        )
                        setattr(
                            entry_after,
                            "_sfi_mixed_page_capture_route_family",
                            captured_actual_route_family,
                        )
                        if route_trace_enabled:
                            try:
                                from patches.fa3_native.install import append_fa3_route_trace

                                append_fa3_route_trace(
                                    {
                                        "event": "mixed_page_full_cudagraph_capture_tag",
                                        "step_id": _mixed_page_full_cudagraph_profile_step_id(
                                            controller
                                        ),
                                        "batch_descriptor": repr(batch_descriptor_after),
                                        "captured_route_family": captured_actual_route_family,
                                        "inferred_route_family": inferred_route_family,
                                    }
                                )
                            except Exception:
                                pass
            except Exception:
                pass
        if profile_enabled:
            profile_pre_consume_drain_stats = (
                profile_pre_consume_drain_stats or {}
            )
            profile_total_end_ns = time.perf_counter_ns()
            profile_pre_original_us = float(
                profile_original_start_ns - profile_total_start_ns
            ) / 1000.0
            profile_payload = {
                    "event": "mixed_page_full_cudagraph_wrapper_call",
                    "hook": "cuda_graph_wrapper",
                    "reason": profile_reason,
                    "step_id": _mixed_page_full_cudagraph_profile_step_id(controller),
                    **profile_step_identity_payload,
                    **profile_prebound_identity_payload,
                    "graph_key": profile_graph_key,
                    "batch_descriptor": profile_batch_descriptor,
                    "wrapper_id": int(id(self)),
                    "entry_id": int(id(profile_entry_for_payload)) if profile_entry_for_payload is not None else -1,
                    "cudagraph_id": int(id(getattr(profile_entry_for_payload, "cudagraph", None))) if profile_entry_for_payload is not None else -1,
                    "forward_runtime_mode": profile_forward_runtime_mode,
                    "wrapper_runtime_mode": profile_wrapper_runtime_mode,
                    "refresh_called": bool(profile_refresh_called),
                    "post_replay_refresh_payloads": int(profile_payload_enqueue_count),
                    "refresh_us": float(profile_refresh_us),
                    "refresh_stage_profile": (
                        dict(profile_refresh_stage_profile)
                        if isinstance(profile_refresh_stage_profile, dict)
                        else None
                    ),
                    "pre_consume_pending_rebuild_drain_us": float(
                        profile_pre_consume_drain_us
                    ),
                    "pre_consume_pending_rebuild_drained": int(
                        profile_pre_consume_drain_stats.get("drained", 0)
                    ),
                    "pre_consume_pending_rebuild_dropped": int(
                        profile_pre_consume_drain_stats.get("dropped", 0)
                    ),
                    "pre_consume_pending_rebuild_remaining": int(
                        profile_pre_consume_drain_stats.get("remaining", 0)
                    ),
                    "pre_async_refresh_writer_submit_summary": (
                        _full_cudagraph_selector_writer_submit_summary(controller)
                    ),
                    "pre_original_us": profile_pre_original_us,
                    "pre_step_identity_us": _full_cudagraph_pre_timing_get(
                        profile_pre_timing,
                        "pre_step_identity_us",
                    ),
                    "pre_initial_bind_identity_us": _full_cudagraph_pre_timing_get(
                        profile_pre_timing,
                        "pre_initial_bind_identity_us",
                    ),
                    "pre_prebound_bind_identity_us": _full_cudagraph_pre_timing_get(
                        profile_pre_timing,
                        "pre_prebound_bind_identity_us",
                    ),
                    "pre_async_refresh_probe_us": _full_cudagraph_pre_timing_get(
                        profile_pre_timing,
                        "pre_async_refresh_probe_us",
                    ),
                    "pre_forward_context_us": _full_cudagraph_pre_timing_get(
                        profile_pre_timing,
                        "pre_forward_context_us",
                    ),
                    "pre_graph_lookup_us": _full_cudagraph_pre_timing_get(
                        profile_pre_timing,
                        "pre_graph_lookup_us",
                    ),
                    "pre_graph_entry_lookup_us": _full_cudagraph_pre_timing_get(
                        profile_pre_timing,
                        "pre_graph_entry_lookup_us",
                    ),
                    "pre_graph_route_family_us": _full_cudagraph_pre_timing_get(
                        profile_pre_timing,
                        "pre_graph_route_family_us",
                    ),
                    "pre_graph_capture_family_us": _full_cudagraph_pre_timing_get(
                        profile_pre_timing,
                        "pre_graph_capture_family_us",
                    ),
                    "pre_graph_key_us": _full_cudagraph_pre_timing_get(
                        profile_pre_timing,
                        "pre_graph_key_us",
                    ),
                    "pre_prebound_graph_state_us": _full_cudagraph_pre_timing_get(
                        profile_pre_timing,
                        "pre_prebound_graph_state_us",
                    ),
                    "pre_prebound_mark_us": _full_cudagraph_pre_timing_get(
                        profile_pre_timing,
                        "pre_prebound_mark_us",
                    ),
                    "pre_prebound_state_lookup_us": _full_cudagraph_pre_timing_get(
                        profile_pre_timing,
                        "pre_prebound_state_lookup_us",
                    ),
                    "pre_prebound_metadata_items_us": _full_cudagraph_pre_timing_get(
                        profile_pre_timing,
                        "pre_prebound_metadata_items_us",
                    ),
                    "pre_prebound_current_check_us": _full_cudagraph_pre_timing_get(
                        profile_pre_timing,
                        "pre_prebound_current_check_us",
                    ),
                    "pre_prebound_rebind_us": _full_cudagraph_pre_timing_get(
                        profile_pre_timing,
                        "pre_prebound_rebind_us",
                    ),
                    "pre_replay_cache_key_us": _full_cudagraph_pre_timing_get(
                        profile_pre_timing,
                        "pre_replay_cache_key_us",
                    ),
                    "pre_replay_cache_hit_us": _full_cudagraph_pre_timing_get(
                        profile_pre_timing,
                        "pre_replay_cache_hit_us",
                    ),
                    "pre_group_ready_check_us": _full_cudagraph_pre_timing_get(
                        profile_pre_timing,
                        "pre_group_ready_check_us",
                    ),
                    "pre_ready_event_wait_us": _full_cudagraph_pre_timing_get(
                        profile_pre_timing,
                        "pre_ready_event_wait_us",
                    ),
                    "pre_ready_event_current_stream_us": _full_cudagraph_pre_timing_get(
                        profile_pre_timing,
                        "pre_ready_event_current_stream_us",
                    ),
                    "pre_ready_event_wait_call_us": _full_cudagraph_pre_timing_get(
                        profile_pre_timing,
                        "pre_ready_event_wait_call_us",
                    ),
                    "pre_ready_event_attr_publish_us": _full_cudagraph_pre_timing_get(
                        profile_pre_timing,
                        "pre_ready_event_attr_publish_us",
                    ),
                    "pre_prebound_stats_us": _full_cudagraph_pre_timing_get(
                        profile_pre_timing,
                        "pre_prebound_stats_us",
                    ),
                    "pre_stats_publish_us": _full_cudagraph_pre_timing_get(
                        profile_pre_timing,
                        "pre_stats_publish_us",
                    ),
                    "pre_other_us": _full_cudagraph_pre_other_us(
                        profile_pre_timing,
                        pre_original_us=profile_pre_original_us,
                        pre_consume_drain_us=profile_pre_consume_drain_us,
                    ),
                    "pre_ready_event_wait_count": int(profile_ready_event_wait_count),
                    "pre_async_refresh_wait_us": _full_cudagraph_pre_timing_get(
                        profile_pre_timing,
                        "pre_async_refresh_wait_us",
                    ),
                    "pre_async_refresh_wait_count": int(
                        profile_async_refresh_wait_count
                    ),
                    "original_call_us": float(
                        profile_original_end_ns - profile_original_start_ns
                    )
                    / 1000.0,
                    "post_call_us": float(
                        profile_total_end_ns - profile_original_end_ns
                    )
                    / 1000.0,
                    "total_us": float(profile_total_end_ns - profile_total_start_ns)
                    / 1000.0,
                    "entry_present": profile_entry_present,
                    "cudagraph_present": profile_cudagraph_present,
                    "captured_route_family": captured_route_family,
                    "current_route_family": current_route_family,
                    "route_family_mismatch": bool(route_family_mismatch),
                    "bridge_graph_policy": bridge_graph_policy,
                }
            if profile_path:
                _append_mixed_page_full_cudagraph_profile_event(
                    profile_path,
                    profile_payload,
                )
        return result

    CUDAGraphWrapper.__call__ = _sparse_cudagraph_call  # type: ignore[assignment]
    _ORIGINAL_CUDAGRAPH_WRAPPER_CALL = original_call
    _CUDAGRAPH_WRAPPER_PATCHED = True


def _patch_gpu_ubatch_wrapper_for_sparse_cudagraph() -> None:
    global _UBATCH_WRAPPER_PATCHED, _ORIGINAL_UBATCH_WRAPPER_CALL
    if _UBATCH_WRAPPER_PATCHED:
        return
    try:
        from vllm.config import CUDAGraphMode  # type: ignore[import]
        from vllm.forward_context import get_forward_context  # type: ignore[import]
        from vllm.v1.worker.gpu_ubatch_wrapper import UBatchWrapper  # type: ignore[import]
    except Exception:
        _log.warning(
            "Cannot import UBatchWrapper for sparse full cudagraph replay hook, skipping"
        )
        return

    original_call = UBatchWrapper.__call__

    def _sparse_ubatch_call(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        controller = _GLOBAL_CONTROLLER
        refresh_enabled = _mixed_page_full_cudagraph_replay_refresh_enabled(controller)
        release_pending = bool(
            getattr(controller, "_refresh_producer_stream_release_pending", False)
        )
        profile_path = _full_cudagraph_hook_profile_log(refresh_enabled)
        cuda_event_path = _full_cudagraph_replay_cuda_event_log(refresh_enabled)
        route_trace_enabled = _fa3_route_trace_enabled()
        diagnostic_enabled = bool(profile_path or cuda_event_path or route_trace_enabled)
        if (
            not refresh_enabled
            and not release_pending
            and not profile_path
            and not cuda_event_path
            and not route_trace_enabled
        ):
            return original_call(self, *args, **kwargs)
        profile_total_start_ns = time.perf_counter_ns() if profile_path else 0
        profile_refresh_us = 0.0
        profile_refresh_called = False
        profile_payload_enqueue_count = 0
        profile_refresh_stage_profile = None
        profile_pre_consume_drain_us = 0.0
        profile_pre_consume_drain_stats = None
        profile_pre_timing = _new_full_cudagraph_pre_timing(profile_path)
        profile_ready_event_wait_count = 0
        profile_async_refresh_wait_count = 0
        profile_reason = "refresh_disabled" if not refresh_enabled else ""
        profile_graph_key = ""
        profile_prebound_rrp_graph_state = None
        profile_prebound_rrp_replay_proof = None
        if diagnostic_enabled:
            _t_step_identity_ns = _full_cudagraph_pre_timing_start(profile_pre_timing)
            profile_step_identity_payload = _mixed_page_full_cudagraph_step_identity_payload(controller)
            _full_cudagraph_pre_timing_add(
                profile_pre_timing,
                "pre_step_identity_us",
                _t_step_identity_ns,
            )
            _t_initial_bind_identity_ns = _full_cudagraph_pre_timing_start(
                profile_pre_timing
            )
            profile_prebound_identity_payload = _mixed_page_full_cudagraph_bind_identity_payload(None)
            _full_cudagraph_pre_timing_add(
                profile_pre_timing,
                "pre_initial_bind_identity_us",
                _t_initial_bind_identity_ns,
            )
        else:
            profile_step_identity_payload = {}
            profile_prebound_identity_payload = {}
        profile_ubatch_num_tokens = None
        profile_ubatch_graph_present = None
        if refresh_enabled:
            _t_forward_context_ns = _full_cudagraph_pre_timing_start(
                profile_pre_timing
            )
            forward_context = get_forward_context()
            cudagraph_runtime_mode = getattr(
                forward_context, "cudagraph_runtime_mode", None
            )
            _full_cudagraph_pre_timing_add(
                profile_pre_timing,
                "pre_forward_context_us",
                _t_forward_context_ns,
            )
            if cudagraph_runtime_mode is CUDAGraphMode.FULL:
                ubatch_slices = getattr(forward_context, "ubatch_slices", None)
                if ubatch_slices is not None:
                    _t_graph_lookup_ns = _full_cudagraph_pre_timing_start(
                        profile_pre_timing
                    )
                    num_tokens = sum(
                        int(getattr(ubatch_slice, "num_tokens", 0))
                        for ubatch_slice in ubatch_slices
                    )
                    profile_ubatch_num_tokens = int(num_tokens)
                    if num_tokens in getattr(self, "cudagraphs", {}):
                        profile_graph_key = f"full:num_tokens={int(num_tokens)}"
                        profile_ubatch_graph_present = True
                        current_route_family = _mixed_page_full_cudagraph_route_family(
                            forward_context,
                            controller=controller,
                        )
                        _full_cudagraph_pre_timing_add(
                            profile_pre_timing,
                            "pre_graph_lookup_us",
                            _t_graph_lookup_ns,
                        )
                        if route_trace_enabled:
                            _append_mixed_page_full_cudagraph_hook_trace(
                                hook="ubatch_wrapper",
                                reason=(
                                    "prebound_rrp_graph_state"
                                    if current_route_family == "resolved_row_ptr"
                                    else "refresh"
                                ),
                                controller=controller,
                                forward_context_available=True,
                                forward_context=forward_context,
                                wrapper_runtime_mode=CUDAGraphMode.FULL,
                                batch_descriptor=getattr(
                                    forward_context, "batch_descriptor", None
                                ),
                                ubatch_num_tokens=int(num_tokens),
                                ubatch_graph_present=True,
                            )
                        if current_route_family == "resolved_row_ptr":
                            _t_prebound_graph_state_ns = _full_cudagraph_pre_timing_start(
                                profile_pre_timing
                            )
                            prebound_rrp_graph_state = (
                                _prebound_rrp_current_graph_state_for_forward_context(
                                    controller=controller,
                                    forward_context=forward_context,
                                )
                            )
                            profile_prebound_rrp_graph_state = prebound_rrp_graph_state
                            _full_cudagraph_pre_timing_add(
                                profile_pre_timing,
                                "pre_prebound_graph_state_us",
                                _t_prebound_graph_state_ns,
                            )
                            if diagnostic_enabled:
                                _t_prebound_bind_identity_ns = (
                                    _full_cudagraph_pre_timing_start(profile_pre_timing)
                                )
                                profile_prebound_identity_payload = (
                                    _mixed_page_full_cudagraph_bind_identity_payload(
                                        prebound_rrp_graph_state
                                    )
                                )
                                _full_cudagraph_pre_timing_add(
                                    profile_pre_timing,
                                    "pre_prebound_bind_identity_us",
                                    _t_prebound_bind_identity_ns,
                                )
                            if prebound_rrp_graph_state is None:
                                _raise_missing_prebound_rrp_graph_state(
                                    captured_route_family="ubatch_full_graph",
                                    current_route_family=current_route_family,
                            )
                            profile_reason = "prebound_rrp_graph_state"
                            if _full_cudagraph_replay_refresh_defer_to_deadline_enabled(
                                controller
                            ):
                                (
                                    profile_pre_consume_drain_stats,
                                    profile_pre_consume_drain_us,
                                ) = _drain_full_cudagraph_pending_refresh_before_replay_profiled(
                                    controller=controller,
                                    profile_path=profile_path,
                                )
                            _t_prebound_mark_ns = _full_cudagraph_pre_timing_start(
                                profile_pre_timing
                            )
                            prebound_stats = _mark_prebound_rrp_full_cudagraph_replay(
                                controller=controller,
                                forward_context=forward_context,
                                graph_key=profile_graph_key,
                                profile_pre_timing=profile_pre_timing,
                                state=prebound_rrp_graph_state,
                                metadata_items=(),
                                binding_is_current_prevalidated=True,
                            )
                            profile_prebound_rrp_graph_state = (
                                prebound_stats.replay_generation_state
                            )
                            profile_prebound_rrp_replay_proof = prebound_stats
                            _full_cudagraph_pre_timing_add(
                                profile_pre_timing,
                                "pre_prebound_mark_us",
                                _t_prebound_mark_ns,
                            )
                            profile_ready_event_wait_count = int(
                                getattr(prebound_stats, "ready_event_wait_count", 0)
                                or 0
                            )
                            _t_async_refresh_probe_ns = (
                                _full_cudagraph_pre_timing_start(profile_pre_timing)
                                if profile_pre_timing is not None
                                else 0
                            )
                            profile_async_refresh_wait_count = int(
                                _wait_pending_async_refresh_before_full_cudagraph_replay(
                                    controller=controller,
                                    profile_pre_timing=profile_pre_timing,
                                )
                            )
                            if profile_pre_timing is not None:
                                _full_cudagraph_pre_timing_add(
                                    profile_pre_timing,
                                    "pre_async_refresh_probe_us",
                                    _t_async_refresh_probe_ns,
                                )
                            if route_trace_enabled:
                                _append_prebound_rrp_full_cudagraph_replay_trace(
                                    controller=controller,
                                    forward_context=forward_context,
                                    stats=prebound_stats,
                                )
                        else:
                            profile_reason = "refresh"
                            profile_refresh_called = True
                            profile_refresh_start_ns = time.perf_counter_ns() if profile_path else 0
                            _refresh_mixed_page_full_cudagraph_replay_for_forward_context(
                                controller=controller,
                                forward_context=forward_context,
                                graph_key=profile_graph_key,
                            )
                            if profile_path:
                                profile_refresh_us += (
                                    time.perf_counter_ns() - profile_refresh_start_ns
                                ) / 1000.0
                    else:
                        profile_reason = "ubatch_graph_missing"
                        profile_ubatch_graph_present = False
                        if route_trace_enabled:
                            _append_mixed_page_full_cudagraph_hook_trace(
                                hook="ubatch_wrapper",
                                reason="ubatch_graph_missing",
                                controller=controller,
                                forward_context_available=True,
                                forward_context=forward_context,
                                wrapper_runtime_mode=CUDAGraphMode.FULL,
                                batch_descriptor=getattr(
                                    forward_context, "batch_descriptor", None
                                ),
                                ubatch_num_tokens=int(num_tokens),
                                ubatch_graph_present=False,
                            )
                else:
                    profile_reason = "ubatch_slices_missing"
                    if route_trace_enabled:
                        _append_mixed_page_full_cudagraph_hook_trace(
                            hook="ubatch_wrapper",
                            reason="ubatch_slices_missing",
                            controller=controller,
                            forward_context_available=True,
                            forward_context=forward_context,
                            wrapper_runtime_mode=CUDAGraphMode.FULL,
                            batch_descriptor=getattr(
                                forward_context, "batch_descriptor", None
                            ),
                        )
            else:
                if not release_pending and not profile_path and not route_trace_enabled:
                    try:
                        return original_call(self, *args, **kwargs)
                    finally:
                        # [2026-07-12 RRP-DONE-EVTS-RETIRED] per-call done-event
                        # record deleted: zero wait/query consumers repo-wide.
                        if refresh_enabled or release_pending:
                            _release_refresh_producer_after_decode_if_pending(
                                controller
                            )
                profile_reason = "runtime_mode_mismatch"
                if route_trace_enabled:
                    _append_mixed_page_full_cudagraph_hook_trace(
                        hook="ubatch_wrapper",
                        reason="runtime_mode_mismatch",
                        controller=controller,
                        forward_context_available=True,
                        forward_context=forward_context,
                        wrapper_runtime_mode=CUDAGraphMode.FULL,
                        batch_descriptor=getattr(forward_context, "batch_descriptor", None),
                    )
        else:
            if route_trace_enabled:
                _append_mixed_page_full_cudagraph_hook_trace(
                    hook="ubatch_wrapper",
                    reason="refresh_disabled",
                    controller=controller,
                    forward_context_available=True,
                )
        profile_original_start_ns = time.perf_counter_ns() if profile_path else 0
        profile_cuda_event_sample = (
            _begin_full_cudagraph_replay_cuda_event(
                path=cuda_event_path,
                reason=str(profile_reason),
                step_id=_mixed_page_full_cudagraph_profile_step_id(controller),
                graph_key=profile_graph_key,
                batch_descriptor=(
                    f"ubatch:num_tokens={profile_ubatch_num_tokens}"
                    if profile_ubatch_num_tokens is not None
                    else None
                ),
                state=profile_prebound_rrp_graph_state,
                step_identity_payload=profile_step_identity_payload,
                prebound_identity_payload=profile_prebound_identity_payload,
            )
            if cuda_event_path
            else None
        )
        prebound_consumed_state_before = (
            _snapshot_prebound_rrp_replay_consumed_generation(
                controller=controller,
                state=profile_prebound_rrp_graph_state,
            )
            if profile_reason == "prebound_rrp_graph_state"
            else None
        )
        try:
            result = original_call(self, *args, **kwargs)
        finally:
            profile_original_end_ns = time.perf_counter_ns() if profile_path else 0
            _end_full_cudagraph_replay_cuda_event(profile_cuda_event_sample)
            if refresh_enabled or release_pending:
                _release_refresh_producer_after_decode_if_pending(controller)
        if profile_reason == "prebound_rrp_graph_state" and not (
            _prebound_rrp_replay_consumed_by_nested_wrapper(
                controller=controller,
                state=profile_prebound_rrp_graph_state,
                replay_proof=profile_prebound_rrp_replay_proof,
                previous_consumed_state=prebound_consumed_state_before,
            )
        ):
            _record_prebound_rrp_replay_consumed_generation(
                controller=controller,
                state=profile_prebound_rrp_graph_state,
                replay_proof=profile_prebound_rrp_replay_proof,
            )
        if refresh_enabled and profile_reason == "prebound_rrp_graph_state":
            if (
                not _REPLAY_REFRESH_NOOP_FAST_SKIP_CACHED
                or _full_cudagraph_replay_step_has_refresh_row(controller)
            ):
                _mark_model_forward_refresh_generation_ready(
                    controller=controller,
                    graph_key=profile_graph_key,
                )
        if profile_path:
            profile_pre_consume_drain_stats = (
                profile_pre_consume_drain_stats or {}
            )
            profile_total_end_ns = time.perf_counter_ns()
            profile_pre_original_us = float(
                profile_original_start_ns - profile_total_start_ns
            ) / 1000.0
            _append_mixed_page_full_cudagraph_profile_event(
                profile_path,
                {
                    "event": "mixed_page_full_cudagraph_wrapper_call",
                    "hook": "ubatch_wrapper",
                    "reason": profile_reason,
                    "step_id": _mixed_page_full_cudagraph_profile_step_id(controller),
                    **profile_step_identity_payload,
                    **profile_prebound_identity_payload,
                    "graph_key": profile_graph_key,
                    "refresh_called": bool(profile_refresh_called),
                    "post_replay_refresh_payloads": int(profile_payload_enqueue_count),
                    "refresh_us": float(profile_refresh_us),
                    "refresh_stage_profile": (
                        dict(profile_refresh_stage_profile)
                        if isinstance(profile_refresh_stage_profile, dict)
                        else None
                    ),
                    "pre_consume_pending_rebuild_drain_us": float(
                        profile_pre_consume_drain_us
                    ),
                    "pre_consume_pending_rebuild_drained": int(
                        profile_pre_consume_drain_stats.get("drained", 0)
                    ),
                    "pre_consume_pending_rebuild_dropped": int(
                        profile_pre_consume_drain_stats.get("dropped", 0)
                    ),
                    "pre_consume_pending_rebuild_remaining": int(
                        profile_pre_consume_drain_stats.get("remaining", 0)
                    ),
                    "pre_async_refresh_writer_submit_summary": (
                        _full_cudagraph_selector_writer_submit_summary(controller)
                    ),
                    "pre_original_us": profile_pre_original_us,
                    "pre_step_identity_us": _full_cudagraph_pre_timing_get(
                        profile_pre_timing,
                        "pre_step_identity_us",
                    ),
                    "pre_initial_bind_identity_us": _full_cudagraph_pre_timing_get(
                        profile_pre_timing,
                        "pre_initial_bind_identity_us",
                    ),
                    "pre_prebound_bind_identity_us": _full_cudagraph_pre_timing_get(
                        profile_pre_timing,
                        "pre_prebound_bind_identity_us",
                    ),
                    "pre_async_refresh_probe_us": _full_cudagraph_pre_timing_get(
                        profile_pre_timing,
                        "pre_async_refresh_probe_us",
                    ),
                    "pre_forward_context_us": _full_cudagraph_pre_timing_get(
                        profile_pre_timing,
                        "pre_forward_context_us",
                    ),
                    "pre_graph_lookup_us": _full_cudagraph_pre_timing_get(
                        profile_pre_timing,
                        "pre_graph_lookup_us",
                    ),
                    "pre_graph_entry_lookup_us": _full_cudagraph_pre_timing_get(
                        profile_pre_timing,
                        "pre_graph_entry_lookup_us",
                    ),
                    "pre_graph_route_family_us": _full_cudagraph_pre_timing_get(
                        profile_pre_timing,
                        "pre_graph_route_family_us",
                    ),
                    "pre_graph_capture_family_us": _full_cudagraph_pre_timing_get(
                        profile_pre_timing,
                        "pre_graph_capture_family_us",
                    ),
                    "pre_graph_key_us": _full_cudagraph_pre_timing_get(
                        profile_pre_timing,
                        "pre_graph_key_us",
                    ),
                    "pre_prebound_graph_state_us": _full_cudagraph_pre_timing_get(
                        profile_pre_timing,
                        "pre_prebound_graph_state_us",
                    ),
                    "pre_prebound_mark_us": _full_cudagraph_pre_timing_get(
                        profile_pre_timing,
                        "pre_prebound_mark_us",
                    ),
                    "pre_prebound_state_lookup_us": _full_cudagraph_pre_timing_get(
                        profile_pre_timing,
                        "pre_prebound_state_lookup_us",
                    ),
                    "pre_prebound_metadata_items_us": _full_cudagraph_pre_timing_get(
                        profile_pre_timing,
                        "pre_prebound_metadata_items_us",
                    ),
                    "pre_prebound_current_check_us": _full_cudagraph_pre_timing_get(
                        profile_pre_timing,
                        "pre_prebound_current_check_us",
                    ),
                    "pre_prebound_rebind_us": _full_cudagraph_pre_timing_get(
                        profile_pre_timing,
                        "pre_prebound_rebind_us",
                    ),
                    "pre_replay_cache_key_us": _full_cudagraph_pre_timing_get(
                        profile_pre_timing,
                        "pre_replay_cache_key_us",
                    ),
                    "pre_replay_cache_hit_us": _full_cudagraph_pre_timing_get(
                        profile_pre_timing,
                        "pre_replay_cache_hit_us",
                    ),
                    "pre_group_ready_check_us": _full_cudagraph_pre_timing_get(
                        profile_pre_timing,
                        "pre_group_ready_check_us",
                    ),
                    "pre_ready_event_wait_us": _full_cudagraph_pre_timing_get(
                        profile_pre_timing,
                        "pre_ready_event_wait_us",
                    ),
                    "pre_ready_event_current_stream_us": _full_cudagraph_pre_timing_get(
                        profile_pre_timing,
                        "pre_ready_event_current_stream_us",
                    ),
                    "pre_ready_event_wait_call_us": _full_cudagraph_pre_timing_get(
                        profile_pre_timing,
                        "pre_ready_event_wait_call_us",
                    ),
                    "pre_ready_event_attr_publish_us": _full_cudagraph_pre_timing_get(
                        profile_pre_timing,
                        "pre_ready_event_attr_publish_us",
                    ),
                    "pre_prebound_stats_us": _full_cudagraph_pre_timing_get(
                        profile_pre_timing,
                        "pre_prebound_stats_us",
                    ),
                    "pre_stats_publish_us": _full_cudagraph_pre_timing_get(
                        profile_pre_timing,
                        "pre_stats_publish_us",
                    ),
                    "pre_other_us": _full_cudagraph_pre_other_us(
                        profile_pre_timing,
                        pre_original_us=profile_pre_original_us,
                        pre_consume_drain_us=profile_pre_consume_drain_us,
                    ),
                    "pre_ready_event_wait_count": int(profile_ready_event_wait_count),
                    "pre_async_refresh_wait_us": _full_cudagraph_pre_timing_get(
                        profile_pre_timing,
                        "pre_async_refresh_wait_us",
                    ),
                    "pre_async_refresh_wait_count": int(
                        profile_async_refresh_wait_count
                    ),
                    "original_call_us": float(
                        profile_original_end_ns - profile_original_start_ns
                    )
                    / 1000.0,
                    "post_call_us": float(
                        profile_total_end_ns - profile_original_end_ns
                    )
                    / 1000.0,
                    "total_us": float(profile_total_end_ns - profile_total_start_ns)
                    / 1000.0,
                    "ubatch_num_tokens": profile_ubatch_num_tokens,
                    "ubatch_graph_present": profile_ubatch_graph_present,
                },
            )
        return result

    UBatchWrapper.__call__ = _sparse_ubatch_call  # type: ignore[assignment]
    _ORIGINAL_UBATCH_WRAPPER_CALL = original_call
    _UBATCH_WRAPPER_PATCHED = True


_ORIGINAL_SET_ASYNC_SAMPLED_TOKEN_IDS = None
_INSTALLED_SET_ASYNC_SAMPLED_TOKEN_IDS_WRAPPER = None


def _read_async_sampled_token(
    sampled_ids: object,
    *,
    request_id: object,
    previous_row_index: object,
) -> int:
    """Read one TP async token without allowing a rank-local dropout."""
    shape_raw = getattr(sampled_ids, "shape", None)
    try:
        shape = tuple(int(value) for value in shape_raw)
    except (RuntimeError, TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(
            "E_TP2_TOKEN_SOURCE_INCOMPLETE: async sampled ids have no valid "
            f"rank-2 shape for request={request_id!r}, "
            f"previous_row_index={previous_row_index!r}, shape={shape_raw!r}"
        ) from exc
    if len(shape) != 2 or shape[1] != 1:
        raise RuntimeError(
            "E_TP2_TOKEN_SOURCE_INCOMPLETE: async sampled ids require shape "
            f"[batch, 1] for request={request_id!r}, "
            f"previous_row_index={previous_row_index!r}, shape={shape!r}"
        )
    try:
        row_index = int(previous_row_index)
    except (RuntimeError, TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(
            "E_TP2_TOKEN_SOURCE_INCOMPLETE: async sampled row index is not an "
            f"integer for request={request_id!r}, "
            f"previous_row_index={previous_row_index!r}, shape={shape!r}"
        ) from exc
    if row_index < 0 or row_index >= shape[0]:
        raise RuntimeError(
            "E_TP2_TOKEN_SOURCE_INCOMPLETE: async sampled row index out of "
            f"range for request={request_id!r}, previous_row_index={row_index}, "
            f"shape={shape!r}"
        )
    try:
        raw_token = sampled_ids[row_index, 0]  # type: ignore[index]
    except (RuntimeError, IndexError, KeyError, TypeError) as exc:
        raise RuntimeError(
            "E_TP2_TOKEN_SOURCE_INCOMPLETE: async sampled token access failed "
            f"for request={request_id!r}, previous_row_index={row_index}, "
            f"shape={shape!r}"
        ) from exc
    try:
        return int(raw_token)
    except (RuntimeError, TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(
            "E_TP2_TOKEN_SOURCE_INCOMPLETE: async sampled token is not an "
            f"integer for request={request_id!r}, previous_row_index={row_index}, "
            f"shape={shape!r}, value={raw_token!r}"
        ) from exc


def _install_async_sampled_token_stash() -> None:
    """[TP-ASYNC-HARVEST] 搭车收割 vLLM 自己的 async 采样 token D2H 副本。

    async 调度下 worker 每步把 -1 占位符写进 input_batch.token_ids_cpu 且永不
    回填（真值只存在于 AsyncGPUModelRunnerOutput 的 CPU 副本 + copy-ready
    event）。vLLM 在 execute_model 返回路径上对每个 async 步无条件调用
    InputBatch.set_async_sampled_token_ids(cpu_copy, event)——在这里搭车把
    (cpu_copy, event, prev_req_id_to_index) 三元组压入 controller 的 FIFO
    stash（零新增拷贝/同步），下一步 prepare 阶段由修复通道排空并原位回填
    占位符。仅 TP>1 时收集（TP=1 走 enginecore sentence hook，语义同构）。
    """
    global _ORIGINAL_SET_ASYNC_SAMPLED_TOKEN_IDS
    global _INSTALLED_SET_ASYNC_SAMPLED_TOKEN_IDS_WRAPPER
    _preflight_async_sampled_token_stash_hook_lease()
    if _ORIGINAL_SET_ASYNC_SAMPLED_TOKEN_IDS is not None:
        return
    try:
        from vllm.v1.worker.gpu_input_batch import InputBatch  # type: ignore[import]
    except Exception:
        _log.warning(
            "Cannot import InputBatch for async sampled-token stash patch, skipping"
        )
        return

    original_set = InputBatch.set_async_sampled_token_ids

    def _patched_set_async_sampled_token_ids(
        self, sampled_token_ids_cpu, async_copy_ready_event
    ):  # type: ignore[override]
        result = original_set(self, sampled_token_ids_cpu, async_copy_ready_event)
        controller = _GLOBAL_CONTROLLER or _ensure_controller()
        # TP 判定必须与 evaluate_tp_sentence_token_window 同源：worker 进程里
        # _init_tp_size 来自 env（常缺省=1），controller.tp_size 才是 worker
        # 每步覆写的真值——用前者曾导致 stash 永不收集、async 首窗误判为
        # "synchronous scheduling" fail-fast。
        _tp_size = int(getattr(controller, "tp_size", 0) or 0) or int(
            getattr(controller, "_init_tp_size", 1) or 1
        )
        if controller is not None and _tp_size > 1:
            prev_map = getattr(self, "prev_req_id_to_index", None)
            if prev_map:
                stash = getattr(controller, "_async_sampled_stash", None)
                if stash is None:
                    stash = deque()
                    controller._async_sampled_stash = stash
                stash.append(
                    (sampled_token_ids_cpu, async_copy_ready_event, prev_map)
                )
        return result

    InputBatch.set_async_sampled_token_ids = _patched_set_async_sampled_token_ids  # type: ignore[assignment]
    _ORIGINAL_SET_ASYNC_SAMPLED_TOKEN_IDS = original_set
    _INSTALLED_SET_ASYNC_SAMPLED_TOKEN_IDS_WRAPPER = (
        _patched_set_async_sampled_token_ids
    )


def _preflight_async_sampled_token_stash_hook_lease() -> None:
    """Validate ownership of the async sampled-token hook on cold paths."""

    original_present = _ORIGINAL_SET_ASYNC_SAMPLED_TOKEN_IDS is not None
    wrapper_present = _INSTALLED_SET_ASYNC_SAMPLED_TOKEN_IDS_WRAPPER is not None
    if not original_present and not wrapper_present:
        return
    if not original_present or not wrapper_present:
        raise RuntimeError("E_TP_ASYNC_TOKEN_STASH_HOOK_LEASE_STATE")
    try:
        from vllm.v1.worker.gpu_input_batch import InputBatch  # type: ignore[import]
    except Exception as exc:
        raise RuntimeError(
            "E_TP_ASYNC_TOKEN_STASH_HOOK_LEASE_INTERFACE"
        ) from exc
    if (
        InputBatch.set_async_sampled_token_ids
        is not _INSTALLED_SET_ASYNC_SAMPLED_TOKEN_IDS_WRAPPER
    ):
        raise RuntimeError("E_TP_ASYNC_TOKEN_STASH_HOOK_LEASE_LOST")


def _patch_request_append() -> None:
    global _REQUEST_PATCHED, _ORIGINAL_APPEND_OUTPUT_TOKEN_IDS
    if _REQUEST_PATCHED:
        return
    try:
        from vllm.v1.request import Request  # type: ignore[import]
    except Exception:
        _log.warning("Cannot import Request for append_output_token_ids patch, skipping")
        return

    original_append = Request.append_output_token_ids

    def _patched_append(self, token_ids):  # type: ignore[override]
        output_len_before = int(len(getattr(self, "_output_token_ids", ())))
        result = original_append(self, token_ids)
        controller = _GLOBAL_CONTROLLER or _ensure_controller()
        if controller is not None:
            if getattr(controller, "_init_tp_size", 1) > 1:
                controller._has_enginecore_sentence_hook = False
                return result
            if isinstance(token_ids, int):
                tokens = [int(token_ids)]
            else:
                tokens = [int(t) for t in token_ids]
            worker_source_available = bool(
                getattr(controller, "_worker_token_ids_cpu", None) is not None
                and getattr(controller, "_worker_batch_id_to_idx", None) is not None
            )
            tracking = getattr(controller, "request_states", {}).get(
                str(getattr(self, "request_id", ""))
            )
            current_fed = (
                int(getattr(tracking, "_trig_fed", 0))
                if tracking is not None
                else 0
            )
            output_len_after = int(output_len_before) + len(tokens)
            if worker_source_available and current_fed >= output_len_after:
                controller._has_enginecore_sentence_hook = False
                return result
            # Fallback for TP=1 configurations where worker token source is not
            # available yet or still contains async placeholders.  The worker
            # source path remains authoritative once it has already fed this
            # output range.
            if not getattr(controller, '_has_enginecore_sentence_hook', False):
                controller._has_enginecore_sentence_hook = True
            skip_prefix = max(0, min(len(tokens), current_fed - int(output_len_before)))
            tokens_to_record = tokens[skip_prefix:]
            if tokens_to_record:
                controller.record_generated_tokens(self.request_id, tokens_to_record)
            if tracking is not None:
                tracking._trig_fed = max(
                    int(getattr(tracking, "_trig_fed", 0)),
                    output_len_after,
                )
        return result

    Request.append_output_token_ids = _patched_append  # type: ignore[assignment]
    _ORIGINAL_APPEND_OUTPUT_TOKEN_IDS = original_append
    _REQUEST_PATCHED = True



def _patch_initialize_kv_cache() -> None:
    global _KV_INIT_PATCHED, _ORIGINAL_INIT_KV_CACHE
    if _KV_INIT_PATCHED:
        return
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner  # type: ignore[import]
    except Exception:
        _log.warning("Cannot import GPUModelRunner for initialize_kv_cache patch, skipping")
        return

    original_init = GPUModelRunner.initialize_kv_cache

    def _is_minimal_profile_kv_cache(self, kv_cache_config) -> bool:
        compilation_config = getattr(self, "compilation_config", None)
        max_capture_size = int(
            getattr(compilation_config, "max_cudagraph_capture_size", 0) or 0
        )
        if max_capture_size <= 0:
            return False
        try:
            return int(getattr(kv_cache_config, "num_blocks")) == max_capture_size
        except Exception:
            return False

    def _patched_initialize_kv_cache(self, kv_cache_config):  # type: ignore[override]
        result = original_init(self, kv_cache_config)
        controller = _GLOBAL_CONTROLLER or _ensure_controller()
        if controller is None:
            return result
        # 初始化 KV cache 时，清理可能残留的跨 engine 状态，避免 slot/prefill 污染
        if controller.layer_states or controller.request_states:
            controller.reset_for_new_engine()
        try:
            from vllm.model_executor.models.utils import extract_layer_index  # type: ignore[import]
        except Exception as exc:
            raise RuntimeError(
                "initialize_kv_cache requires extract_layer_index; refuse positional fallback"
            ) from exc
        from patches.page_kv_residency import (
            bind_compact_page_residency_to_layer,
            ensure_compact_page_lease_transport,
            resolve_compact_page_lease,
        )

        try:
            lease = resolve_compact_page_lease(kv_cache_config, controller.config)
        except RuntimeError as exc:
            if (
                getattr(controller.config, "compact_page_residency_enabled", False)
                and _is_minimal_profile_kv_cache(self, kv_cache_config)
            ):
                return result
            if "compact page lease is missing" in str(exc):
                lease = ensure_compact_page_lease_transport(
                    kv_cache_config,
                    controller.config,
                )
            else:
                raise
        try:
            forward_ctx = getattr(self.compilation_config, "static_forward_context", None)
            if not forward_ctx:
                if lease is not None:
                    raise RuntimeError(
                        "compact page residency requires static_forward_context"
                    )
                return result
            cache_key_to_layer_idx: Dict[int, int] = {}
            layer_infos: List[Tuple[int, int, int, torch.device, Optional[int], torch.dtype, int, torch.Tensor]] = []
            for layer_name, attn in forward_ctx.items():
                kv_cache = _first_kv_cache_tensor(getattr(attn, "kv_cache", None))
                if not isinstance(kv_cache, torch.Tensor):
                    continue
                if kv_cache.dim() < 1 or kv_cache.shape[0] < 2:
                    if lease is not None:
                        raise RuntimeError(
                            "compact page residency requires two-plane KV cache tensor"
                        )
                    continue
                key_cache = kv_cache[0]
                cache_key = int(key_cache.data_ptr())
                num_heads = getattr(attn, "num_heads", None)
                num_kv_heads = getattr(attn, "num_kv_heads", None)
                head_dim = getattr(attn, "head_size", None)
                if num_heads is None or num_kv_heads is None:
                    if lease is not None:
                        raise RuntimeError(
                            "compact page residency requires attention head metadata"
                        )
                    continue
                if cache_key not in controller.layer_states:
                    state = controller._register_layer(
                        cache_key,
                        int(num_heads),
                        int(num_kv_heads),
                        key_cache.device,
                        head_dim=int(head_dim) if head_dim is not None else None,
                        kv_cache_dtype=key_cache.dtype,
                    )
                    state.kv_cache_dtype = key_cache.dtype
                else:
                    state = controller.layer_states.get(cache_key)
                    if state is not None:
                        if state.head_dim is None and head_dim is not None:
                            state.head_dim = int(head_dim)
                        if state.kv_cache_dtype is None:
                            state.kv_cache_dtype = key_cache.dtype
                try:
                    layer_idx = int(extract_layer_index(layer_name))
                except Exception as exc:
                    raise RuntimeError(
                        f"extract_layer_index failed for layer {layer_name!r}; "
                        "refuse positional fallback"
                    ) from exc
                prev = cache_key_to_layer_idx.get(cache_key)
                if prev is None or layer_idx < prev:
                    cache_key_to_layer_idx[cache_key] = layer_idx

                layer_infos.append(
                    (
                        cache_key,
                        int(num_heads),
                        int(num_kv_heads),
                        key_cache.device,
                        int(head_dim) if head_dim is not None else None,
                        key_cache.dtype,
                        int(layer_idx),
                        kv_cache,
                    )
                )

            if lease is not None and not cache_key_to_layer_idx:
                raise RuntimeError(
                    "compact page residency found no bindable KV cache layers"
                )
            if cache_key_to_layer_idx:
                ordered = sorted(cache_key_to_layer_idx.items(), key=lambda x: x[1])
                new_keys = [key for key, _ in ordered]
                if controller.layer_cache_keys and set(new_keys) != set(controller.layer_cache_keys):
                    controller.reset_for_new_engine()
                for cache_key, num_heads, num_kv_heads, device, head_dim, dtype, _, kv_cache in layer_infos:
                    if cache_key not in controller.layer_states:
                        state = controller._register_layer(
                            cache_key,
                            num_heads,
                            num_kv_heads,
                            device,
                            head_dim=head_dim,
                            kv_cache_dtype=dtype,
                        )
                        state.kv_cache_dtype = dtype
                    else:
                        state = controller.layer_states.get(cache_key)
                        if state is not None:
                            if state.head_dim is None and head_dim is not None:
                                state.head_dim = int(head_dim)
                            if state.kv_cache_dtype is None:
                                state.kv_cache_dtype = dtype
                    if lease is not None:
                        state = controller.layer_states.get(cache_key)
                        if state is None:
                            raise RuntimeError(
                                "compact page residency layer state missing after registration"
                            )
                        bind_compact_page_residency_to_layer(
                            state,
                            kv_cache,
                            kv_cache_config,
                            controller.config,
                            lease=lease,
                        )
                controller.layer_cache_keys = new_keys
                controller.layer_index_by_cache_key = {key: idx for idx, key in enumerate(new_keys)}
                controller.step_decode_data = None
                controller.step_decode_cache_key = None
                controller._step_decode_spec_key = None
                controller.step_dispatch_plan = None
        except Exception as exc:
            raise RuntimeError(
                "initialize_kv_cache sparse layer discovery failed; "
                f"refuse silent skip: {type(exc).__name__}: {exc}"
            ) from exc
        return result

    GPUModelRunner.initialize_kv_cache = _patched_initialize_kv_cache  # type: ignore[assignment]
    _ORIGINAL_INIT_KV_CACHE = original_init
    _KV_INIT_PATCHED = True


def _patch_compact_page_residency_core() -> None:
    global _COMPACT_PAGE_RESIDENCY_PATCHED, _ORIGINAL_KV_CACHE_MANAGER_INIT
    global _ORIGINAL_BLOCK_POOL_METHODS, _COMPACT_PAGE_BLOCK_POOL_CLS
    global _COMPACT_PAGE_KV_CACHE_MANAGER_CLS
    controller = _GLOBAL_CONTROLLER
    config = getattr(controller, "config", None) if controller is not None else None
    compact_page_enabled = bool(
        getattr(config, "compact_page_residency_enabled", False)
    )
    if not compact_page_enabled:
        if _COMPACT_PAGE_RESIDENCY_PATCHED:
            _restore_compact_page_residency_core_patch()
        return
    if _COMPACT_PAGE_RESIDENCY_PATCHED:
        return
    try:
        from vllm.v1.core.block_pool import BlockPool  # type: ignore[import]
        from vllm.v1.core.kv_cache_manager import KVCacheManager  # type: ignore[import]
    except Exception as exc:
        message = "Cannot import vLLM BlockPool/KVCacheManager for compact page residency"
        raise RuntimeError(message) from exc

    from patches.page_kv_residency import (
        attach_compact_page_lease,
        patch_compact_page_block_pool_methods,
        reserve_compact_page_blocks,
    )

    original_init = KVCacheManager.__init__
    original_methods = patch_compact_page_block_pool_methods(BlockPool)

    def _patched_kv_cache_manager_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        result = original_init(self, *args, **kwargs)
        controller = _GLOBAL_CONTROLLER
        config = getattr(controller, "config", None) if controller is not None else None
        if config is None:
            return result
        lease = reserve_compact_page_blocks(
            getattr(self, "block_pool"),
            getattr(self, "kv_cache_config"),
            config,
        )
        if lease is not None:
            setattr(self, "_sfi_compact_page_lease", lease)
            attach_compact_page_lease(getattr(self, "kv_cache_config"), lease)
        return result

    KVCacheManager.__init__ = _patched_kv_cache_manager_init  # type: ignore[assignment]
    _ORIGINAL_KV_CACHE_MANAGER_INIT = original_init
    _ORIGINAL_BLOCK_POOL_METHODS = original_methods
    _COMPACT_PAGE_BLOCK_POOL_CLS = BlockPool
    _COMPACT_PAGE_KV_CACHE_MANAGER_CLS = KVCacheManager
    _COMPACT_PAGE_RESIDENCY_PATCHED = True


def _restore_compact_page_residency_core_patch() -> None:
    global _COMPACT_PAGE_RESIDENCY_PATCHED, _ORIGINAL_KV_CACHE_MANAGER_INIT
    global _ORIGINAL_BLOCK_POOL_METHODS, _COMPACT_PAGE_BLOCK_POOL_CLS
    global _COMPACT_PAGE_KV_CACHE_MANAGER_CLS
    if _COMPACT_PAGE_RESIDENCY_PATCHED:
        if (
            _COMPACT_PAGE_KV_CACHE_MANAGER_CLS is not None
            and _ORIGINAL_KV_CACHE_MANAGER_INIT is not None
        ):
            _COMPACT_PAGE_KV_CACHE_MANAGER_CLS.__init__ = _ORIGINAL_KV_CACHE_MANAGER_INIT  # type: ignore[assignment]
        if _COMPACT_PAGE_BLOCK_POOL_CLS is not None:
            from patches.page_kv_residency import restore_compact_page_block_pool_methods

            restore_compact_page_block_pool_methods(
                _COMPACT_PAGE_BLOCK_POOL_CLS,
                _ORIGINAL_BLOCK_POOL_METHODS,
            )
    _COMPACT_PAGE_RESIDENCY_PATCHED = False
    _ORIGINAL_KV_CACHE_MANAGER_INIT = None
    _ORIGINAL_BLOCK_POOL_METHODS = None
    _COMPACT_PAGE_BLOCK_POOL_CLS = None
    _COMPACT_PAGE_KV_CACHE_MANAGER_CLS = None
    try:
        from patches.page_kv_residency import clear_compact_page_lease_transport

        clear_compact_page_lease_transport()
    except Exception:
        _log.error("Failed to clear compact page lease transport", exc_info=True)


def _preflight_controller_hook_leases() -> None:
    """Validate every hook with an explicit ownership lease before mutation."""

    patch_installed = bool(_PATCH_INSTALLED)
    fa3_installed = bool(_FLASH_ATTN_FORWARD_PATCHED)
    if patch_installed != fa3_installed:
        raise RuntimeError("E_SFI_FA3_GATEWAY_INSTALL_STATE")
    fa3_predecessors = (
        _ORIGINAL_V1_FLASH_ATTN_VARLEN_FUNC,
        _ORIGINAL_V1_FLASH_ATTN_FORWARD,
        _ORIGINAL_V1_FLASH_ATTN_GET_SCHEDULER_METADATA,
        _ORIGINAL_V1_FLASH_ATTN_GET_FLASH_ATTN_VERSION,
        _ORIGINAL_FA_UTILS_FLASH_ATTN_VARLEN_FUNC,
        _ORIGINAL_FA_UTILS_GET_SCHEDULER_METADATA,
        _ORIGINAL_FA_UTILS_GET_FLASH_ATTN_VERSION,
        _ORIGINAL_V1_FLASH_METADATA_FULL_CUDAGRAPH_SUPPORTED,
    )
    if not fa3_installed:
        if any(value is not None for value in fa3_predecessors):
            raise RuntimeError("E_SFI_FA3_GATEWAY_INSTALL_STATE")
    else:
        if _ORIGINAL_V1_FLASH_ATTN_FORWARD is None:
            raise RuntimeError("E_SFI_FA3_GATEWAY_INSTALL_STATE")
        try:
            from vllm.v1.attention.backends import flash_attn as v1_flash_attn

            if getattr(v1_flash_attn, "FlashAttentionImpl", None) is None:
                raise AttributeError("FlashAttentionImpl is unavailable")
            if any(
                value is not None
                for value in (
                    _ORIGINAL_FA_UTILS_FLASH_ATTN_VARLEN_FUNC,
                    _ORIGINAL_FA_UTILS_GET_SCHEDULER_METADATA,
                    _ORIGINAL_FA_UTILS_GET_FLASH_ATTN_VERSION,
                )
            ):
                import_fa_utils_module()
        except Exception as exc:
            raise RuntimeError("E_SFI_FA3_GATEWAY_LEASE_INTERFACE") from exc
    _preflight_update_states_hook_lease()
    _preflight_batch_execution_admission_hook_lease()
    _preflight_async_sampled_token_stash_hook_lease()


def _set_controller(config: SparseControllerConfig) -> "VLLMSparseController":
    from patches.vllm_sparse_patch import VLLMSparseController  # lazy: avoid circular
    global _GLOBAL_CONTROLLER
    _normalize_prefill_capture_config(config)
    controller = VLLMSparseController(config)
    _GLOBAL_CONTROLLER = controller
    _patch_prepare_inputs()
    _patch_batch_execution_cudagraph_admission()
    _patch_dummy_run()
    _patch_update_states()
    _patch_model_forward_refresh_owner()
    _patch_request_append()
    _install_async_sampled_token_stash()
    _patch_initialize_kv_cache()
    return controller

def _ensure_controller() -> Optional["VLLMSparseController"]:
    global _GLOBAL_CONTROLLER
    if _GLOBAL_CONTROLLER is not None:
        return _GLOBAL_CONTROLLER
    payload = os.environ.get(_SERIALIZED_CONFIG_ENV)
    if not payload:
        return None
    config = _deserialize_config(payload)
    if config is None or not config.enabled:
        return None
    return _install_controller_patch_transaction(config)






def _set_unified_attention_mode(mode: str) -> None:
    import patches.vllm_sparse_patch as _main  # lazy: state lives in main module
    if mode == _main._CURRENT_UNIFIED_ATTENTION_MODE:
        return
    # 仅更新 dispatch mode；不再执行 unified_attention 函数对象交换。
    _main._CURRENT_UNIFIED_ATTENTION_MODE = mode



def _patch_worker_busy_loop_tp_exception_failfast() -> None:
    """[TP-EXC-FAILFAST 2026-07-08] 非 output-rank worker 异常改为大声即死。

    vLLM WorkerProc.worker_busy_loop 对 execute 异常按 output_rank 过滤:
    非 output rank 只 logger.exception(默认流向 stdout,被 bench IPC 管道
    吞)后 ``continue``。TP>1 下该 rank 已放弃本步剩余 collectives,对端
    rank 永久卡在 all_reduce(NCCL kernel 自旋 100%,pynccl 不在 c10d
    watchdog 覆盖面)——32k×TP2×双代楔死案的"楔死机器"(step-ledger 活体
    实证:rank1 flip-parity 断言开火被吞→rank0 卡 o_proj AR)。任何单
    rank 的步内异常在 TP>1 下都不可恢复(集合序列已错位),静默续跑=
    坏值继续产生;唯一正确形态=异常全文写 stderr 后 os._exit,vLLM 的
    worker 死亡监测随即终止全家,失败快而可见。output rank 保留原生
    FAILURE 上报路径(engine 拿得到原始异常,原生即 loud)。
    忠实复刻 vllm019-cu126 的循环体+单臂改动;与 step-ledger 诊断钩共存
    时本替换覆盖其 busy_loop 包装(exec_exc/worker_shutdown 账本行不受
    影响,楔死取证不依赖 busy_loop_exit 行)。
    """
    try:
        from vllm.v1.executor import multiproc_executor as _mpe
    except Exception:
        return
    if getattr(_mpe.WorkerProc.worker_busy_loop, "_sfi_tp_exc_failfast", False):
        return

    import os as _os
    import sys as _sys
    import traceback as _tb
    from functools import partial as _partial

    import cloudpickle as _cloudpickle

    def worker_busy_loop(self):
        assert self.rpc_broadcast_mq is not None
        while True:
            method, args, kwargs, output_rank = self.rpc_broadcast_mq.dequeue(
                indefinite=True
            )
            try:
                if isinstance(method, str):
                    func = getattr(self.worker, method)
                elif isinstance(method, bytes):
                    func = _partial(_cloudpickle.loads(method), self.worker)
                output = func(*args, **kwargs)
            except Exception as e:
                if hasattr(e, "add_note"):
                    e.add_note(_tb.format_exc())
                _mpe.logger.exception("WorkerProc hit an exception.")
                if output_rank is None or self.rank == output_rank:
                    self.handle_output(e)
                    continue
                # [TP-EXC-FAILFAST] vLLM 原生此处 continue=静默吞掉。
                _sys.stderr.write(
                    "[SFI TP-EXC-FAILFAST] non-output-rank worker exception "
                    f"(rank={getattr(self, 'rank', '?')}) would be swallowed "
                    "by vLLM and desync TP collectives; dying loudly instead:\n"
                    + _tb.format_exc()
                )
                _sys.stderr.flush()
                _os._exit(70)

            if output_rank is None or self.rank == output_rank:
                self.handle_output(output)

    worker_busy_loop._sfi_tp_exc_failfast = True
    _mpe.WorkerProc.worker_busy_loop = worker_busy_loop


def _install_patch() -> None:
    # [TRITON-LINE-RETIRED 2026-07-07] TRITON_ATTN sparse 线已在 main 下线,
    # 由专门 triton branch 承载;此处为三条安装路径(apply/env/lazy 重入)
    # 的唯一咽喉,设起即 fail-fast,无回退。env 未设时不拦(vLLM 自动选
    # FLASH_ATTN)。
    _backend = os.environ.get("VLLM_ATTENTION_BACKEND", "")
    if _backend.startswith("TRITON"):
        raise RuntimeError(
            "TRITON_ATTN sparse line retired on main; use the dedicated "
            "triton branch. Main is FA3-only (FLASH_ATTN_VLLM_V1). "
            "No fallback."
        )
    _patch_worker_busy_loop_tp_exception_failfast()
    global _PATCH_INSTALLED
    if _PATCH_INSTALLED:
        if not _FLASH_ATTN_FORWARD_PATCHED:
            raise RuntimeError(
                "E_SFI_FA3_GATEWAY_INSTALL_STATE: patch is marked installed "
                "without the required FA3 forward gateway"
            )
        _patch_compact_page_residency_core()
        return
    _patch_cuda_graph_wrapper_for_sparse_cudagraph()
    _patch_gpu_ubatch_wrapper_for_sparse_cudagraph()
    _patch_compact_page_residency_core()
    try:
        _patch_flash_attention_forward_and_helpers()
        if not _FLASH_ATTN_FORWARD_PATCHED:
            raise RuntimeError(
                "E_SFI_FA3_GATEWAY_INSTALL_STATE: required FA3 forward "
                "gateway did not publish its installed state"
            )
        _PATCH_INSTALLED = True
        import patches.vllm_sparse_patch as _main
        _main._CURRENT_UNIFIED_ATTENTION_MODE = "default"
    except Exception:
        _restore_compact_page_residency_core_patch()
        raise




def _run_selected_no_capture_mixed_forward(
    *,
    self,
    layer,
    query,
    key,
    value,
    kv_cache,
    attn_metadata,
    output=None,
    output_scale=None,
    output_block_scale=None,
):
    v1_flash_attn, fa_utils = _get_v1_flash_attn_modules()

    from patches.fa3_native.install import (
        append_fa3_route_trace,
        fa3_route_trace_enabled,
    )
    from patches.fa3_native.scope_async import consume_selected_scope

    controller = _get_global_controller()
    if controller is None or not bool(getattr(getattr(controller, "config", None), "enabled", False)):
        raise RuntimeError(
            "native FA3 selected mixed route requires a live sparse controller"
        )
    step_ctx = _resolve_mixed_route_step_context(
        controller=controller,
        attn_metadata=attn_metadata,
        stage="native FA3 selected mixed route",
    )
    step_authority = getattr(step_ctx, "step_authority", None)
    if step_authority is None:
        step_authority = getattr(controller, "step_authority", None)
    if step_authority is None:
        raise RuntimeError(
            "native FA3 selected mixed route requires step_authority"
        )
    step_envelope = getattr(step_ctx, "step_envelope_v2", None)
    if step_envelope is None:
        raise RuntimeError(
            "native FA3 selected mixed route requires step_envelope_v2"
        )
    step_bound_meta = getattr(controller, "step_bound_meta", None)
    if step_bound_meta is None:
        raise RuntimeError(
            "native FA3 selected mixed route requires step_bound_meta"
        )

    cu_seqlens_q = getattr(attn_metadata, "query_start_loc")
    seqused_k = getattr(attn_metadata, "seq_lens")
    batch_size = int(cu_seqlens_q.shape[0]) - 1
    row_plan = (
        getattr(step_bound_meta, "prologue_selected_row_plan", None)
        if int(getattr(step_bound_meta, "prologue_done_for_identity_token", -1))
        == int(getattr(step_bound_meta, "step_identity_token", 0))
        else None
    )
    if row_plan is None:
        row_plan = _resolve_selected_no_capture_row_plan(
            controller=controller,
            step_ctx=step_ctx,
            step_authority=step_authority,
            batch_size=batch_size,
            device=query.device,
        )
    if row_plan.has_capture:
        raise RuntimeError(
            "native FA3 selected mixed route requires no capture rows"
        )

    key_cache, _ = kv_cache.unbind(0)
    cache_key = key_cache.data_ptr()
    state = controller.get_state(
        cache_key,
        query.shape[1],
        key_cache.shape[2],
        query.device,
        head_dim=query.shape[2],
        kv_cache_dtype=key_cache.dtype,
    )
    _ensure_layer_state_snapshot_alignment(
        state=state,
        step_ctx=step_ctx,
        step_envelope=step_envelope,
    )
    block_table = getattr(attn_metadata, "block_table", None)
    # Rev 2 (2026-04-23): under default-on sel-page skip, the producer chain
    # (selector commit of selected_scope_wait_handle) is bypassed; consuming
    # an un-committed handle here would deadlock with NotReadyError. Gate the
    # snapshot resolve + consume by the same helper that controls production.
    # The inner _selected_wrapper owner-based dispatcher does not depend on
    # the snapshot/scope handle, so skipping outer sel-page work is safe.
    from patches.sparse_constants import should_skip_page_sparse_state
    _attn_mode_peripheral = str(
        getattr(getattr(controller, "config", None), "attn_mode", "compact_recent")
    )
    if row_plan.has_selected_consume and not should_skip_page_sparse_state(
        _attn_mode_peripheral
    ):
        snapshot = _resolve_launch_local_snapshot_from_step_source(
            attn_metadata=attn_metadata,
            step_ctx=step_ctx,
            step_authority=step_authority,
        )
        if snapshot is None:
            raise RuntimeError(
                "native FA3 selected mixed route requires launch-local snapshot"
            )
        if fa3_route_trace_enabled():
            handle = getattr(snapshot, "selected_scope_wait_handle", None)
            append_fa3_route_trace(
                {
                    "event": "selected_scope_consume_attempt",
                    "mode": "selected",
                    "epoch": int(getattr(step_authority, "epoch", -1)),
                    "handle_id": id(handle),
                    "handle_ready": bool(getattr(handle, "is_ready", False)) if handle is not None else None,
                    "consumer_step_id": int(
                        getattr(getattr(snapshot, "target_selected_scope_key", None), "consumer_step_id", -1)
                    ),
                }
            )
        consume_selected_scope(snapshot)
    default_cp_world_size = int(getattr(attn_metadata, "cp_world_size", 1) or 1)
    bridge = _resolve_selected_no_capture_bridge(controller=controller)
    original_seq_lens = getattr(attn_metadata, "seq_lens")
    original_max_seq_len = getattr(attn_metadata, "max_seq_len", None)
    original_scheduler_metadata = getattr(attn_metadata, "scheduler_metadata", None)
    page_size = _infer_paged_kv_page_size(kv_cache)
    if page_size <= 0:
        raise RuntimeError("native FA3 selected mixed route requires positive page_size")
    real_seqused_k = _resolve_step_real_seqused_k(
        step_bound_meta=step_bound_meta,
        original_seq_lens=original_seq_lens,
        device=query.device,
        batch_size=batch_size,
        cache_owner=controller,
    )
    if (
        not isinstance(real_seqused_k, torch.Tensor)
        or real_seqused_k.device != query.device
        or real_seqused_k.dtype != torch.int32
        or real_seqused_k.dim() != 1
        or int(real_seqused_k.numel()) != int(batch_size)
    ):
        raise RuntimeError(
            "native FA3 selected mixed route requires full-batch real seqused_k truth"
        )
    def _selected_wrapper(*args, **kwargs):
        _ctrl = _get_global_controller()
        if _ctrl is None or getattr(getattr(_ctrl, "config", None), "attn_mode", "compact_recent") != "compact_recent":
            if _ORIGINAL_V1_FLASH_ATTN_VARLEN_FUNC is None:
                raise RuntimeError(
                    "_selected_wrapper bypass: _ORIGINAL_V1_FLASH_ATTN_VARLEN_FUNC not initialized"
                )
            return _ORIGINAL_V1_FLASH_ATTN_VARLEN_FUNC(*args, **kwargs)
        if args:
            raise RuntimeError("selected mixed wrapper expects keyword arguments only")
        # Drop vLLM's placeholder identity descale for non-quantized KV so
        # the compact_recent contract accepts bf16/fp16/fp32. See
        # _strip_identity_descale_kwargs_inplace docstring — pure CPU dtype
        # check, no GPU ops, CUDA-graph safe.
        _strip_identity_descale_kwargs_inplace(kwargs)
        _reject_legacy_selected_request_level_kwargs(
            wrapper_name="selected mixed wrapper",
            kwargs=kwargs,
        )
        actual_cp_world_size = int(kwargs.get("cp_world_size", default_cp_world_size) or 1)
        max_seqlen_q = int(kwargs.get("max_seqlen_q") or 1)
        is_causal = bool(kwargs.get("causal"))
        softcap = float(kwargs.get("softcap", 0.0) or 0.0)
        num_splits = int(kwargs.get("num_splits", 0) or 0)
        dense_input_cp_tot = kwargs.get("cp_tot_seqused_k")
        window_tuple = _normalize_window_tuple(kwargs.get("window_size"))
        compact_decode_launch_support_error = _compact_recent_support_error(
            max_seqlen_q=1,
            is_causal=is_causal,
            window_size_left=int(window_tuple[0]),
            window_size_right=int(window_tuple[1]),
            softcap=softcap,
            cp_world_size=actual_cp_world_size,
            num_splits=num_splits,
            q_v=kwargs.get("q_v"),
            s_aux=kwargs.get("s_aux"),
            q_descale=kwargs.get("q_descale"),
            k_descale=kwargs.get("k_descale"),
            v_descale=kwargs.get("v_descale"),
        )

        kwargs.pop("fa_version", None)
        kwargs.pop("max_seqlen_k", None)
        kwargs.pop("seqused_k", None)
        kwargs.pop("cp_world_size", None)
        kwargs.pop("cp_tot_seqused_k", None)
        kwargs.pop("selected_page_table_i32", None)
        kwargs.pop("row_consume_mode_i32", None)
        kwargs.pop("selected_seqused_k_by_head_i32", None)
        kwargs.pop("cp_selected_seqused_k_by_head_i32", None)
        key_arg = kwargs.get("k")
        value_arg = kwargs.get("v")
        q_arg = kwargs.get("q")
        block_table_arg = kwargs.get("block_table")
        if not isinstance(q_arg, torch.Tensor):
            raise RuntimeError("selected mixed wrapper requires tensor q")
        if not isinstance(key_arg, torch.Tensor) or not isinstance(value_arg, torch.Tensor):
            raise RuntimeError("selected mixed wrapper requires tensor k/v")
        if not isinstance(block_table_arg, torch.Tensor):
            raise RuntimeError("selected mixed wrapper requires tensor block_table")
        if compact_decode_launch_support_error is not None:
            _raise_selected_compact_recent_launch_rejected(
                compact_decode_launch_support_error
            )
        # Prologue runs after tensor validation and compact decode rail support
        # checks. Full-batch max_seqlen_q is intentionally not rejected here:
        # mixed chunk dispatch launches prefill through mixed_page and decode
        # through compact_recent active_worklist with physical max_seqlen_q=1.
        # The prologue resolves 4 per-step plans and populates
        # step_bound_meta.prologue_*.
        _ensure_step_prologue(
            controller=controller,
            step_ctx=step_ctx,
            step_authority=step_authority,
            step_bound_meta=step_bound_meta,
            batch_size=batch_size,
            kv_cache=kv_cache,
            device=query.device,
            canonical_state=state,
        )
        # C.5 wiring: use the CPU route authority to decide whether this step
        # needs compact overlay attention at all. Compact rows are handled by
        # the mixed-page overlay route; full-only rows keep the existing
        # single-launch body below.
        from patches.fa_sparse_runtime.compact_recent_route_authority import (
            CompactRecentRailMode,
        )

        rail_decision = step_bound_meta.prologue_rail_decision
        full_kv_handoff = bool(
            getattr(step_bound_meta, "prologue_full_kv_handoff", False)
        )
        overlay_route_required = (
            rail_decision.mode is not CompactRecentRailMode.NO_COMPACT
        )
        if fa3_route_trace_enabled():
            _owner_plan_trace = step_bound_meta.prologue_owner_plan
            append_fa3_route_trace(
                {
                    "event": "compact_recent_rail_decision",
                    "epoch": int(getattr(step_authority, "epoch", -1)),
                    "mode": getattr(getattr(rail_decision, "mode", None), "name", str(getattr(rail_decision, "mode", ""))),
                    "compact_rows": [
                        int(v) for v in tuple(getattr(rail_decision, "compact_rows", tuple()))
                    ],
                    "inactive_rows": [
                        int(v) for v in tuple(getattr(rail_decision, "inactive_rows", tuple()))
                    ],
                    "has_prefill_rows": bool(getattr(_owner_plan_trace, "has_prefill_rows", False)),
                    "has_decode_rows": bool(getattr(_owner_plan_trace, "has_decode_rows", False)),
                    "prefill_active_count": int(getattr(_owner_plan_trace, "prefill_active_count", 0)),
                    "decode_active_count": int(getattr(_owner_plan_trace, "decode_active_count", 0)),
                    "use_compact_by_row": [
                        bool(v)
                        for v in tuple(getattr(step_authority, "use_compact_by_row", tuple()))[:batch_size]
                    ],
                    "full_kv_handoff": bool(full_kv_handoff),
                }
            )
        if rail_decision.mode is not CompactRecentRailMode.NO_COMPACT:
            owner_plan = step_bound_meta.prologue_owner_plan
            if owner_plan.has_prefill_rows and owner_plan.has_decode_rows:
                from patches.fa_sparse_runtime.mixed_prefill_decode_dispatch import (
                    dispatch_mixed_prefill_decode_active_workload,
                )

                _record_mixed_page_actual_route_family(
                    controller,
                    _mixed_page_metadata_route_family(attn_metadata),
                )
                dispatch_out = kwargs.get("out")
                if dispatch_out is None:
                    dispatch_out = q_arg.new_empty(q_arg.shape)
                mixed_cu_seqlens_q = kwargs.get("cu_seqlens_q")
                if not isinstance(mixed_cu_seqlens_q, torch.Tensor):
                    mixed_cu_seqlens_q = cu_seqlens_q
                return dispatch_mixed_prefill_decode_active_workload(
                    q=q_arg,
                    k=key_arg,
                    v=value_arg,
                    out=dispatch_out,
                    cu_seqlens_q=mixed_cu_seqlens_q,
                    max_seqlen_q=int(max_seqlen_q),
                    max_seqlen_k=int(original_max_seq_len or max_seqlen_q),
                    seqused_k=real_seqused_k,
                    softmax_scale=kwargs.get("softmax_scale"),
                    window_size=window_tuple,
                    softcap=float(softcap),
                    block_table=block_table_arg,
                    owner_plan=owner_plan,
                    bridge=bridge,
                    controller=controller,
                    state=state,
                    step_authority=step_authority,
                    step_bound_meta=step_bound_meta,
                    step_ctx=step_ctx,
                    q_v=kwargs.get("q_v"),
                    q_descale=kwargs.get("q_descale"),
                    k_descale=kwargs.get("k_descale"),
                    v_descale=kwargs.get("v_descale"),
                    s_aux=kwargs.get("s_aux"),
                    num_splits=int(num_splits),
                    cp_world_size=int(actual_cp_world_size),
                    cp_rank=int(kwargs.get("cp_rank", 0) or 0),
                    cp_tot_seqused_k=(
                        dense_input_cp_tot if isinstance(dense_input_cp_tot, torch.Tensor) else None
                    ),
                    capture_scores=kwargs.get("capture_scores"),
                    capture_row_index_i32=kwargs.get("capture_row_index_i32"),
                    row_capture_last_n_i32=kwargs.get("row_capture_last_n_i32"),
                    rail_decision=rail_decision,
                    **_mixed_page_resolver_kwargs_from_attn_metadata(attn_metadata),
                )

        if overlay_route_required:
            from patches.fa_sparse_runtime.compact_mixed_page_route import (
                run_compact_mixed_page_overlay_route,
            )

            if int(max_seqlen_q) != 1:
                compact_full_launch_support_error = _compact_recent_support_error(
                    max_seqlen_q=max_seqlen_q,
                    is_causal=is_causal,
                    window_size_left=int(window_tuple[0]),
                    window_size_right=int(window_tuple[1]),
                    softcap=softcap,
                    cp_world_size=actual_cp_world_size,
                    num_splits=num_splits,
                    q_v=kwargs.get("q_v"),
                    s_aux=kwargs.get("s_aux"),
                    q_descale=kwargs.get("q_descale"),
                    k_descale=kwargs.get("k_descale"),
                    v_descale=kwargs.get("v_descale"),
                )
                if compact_full_launch_support_error is not None:
                    _raise_selected_compact_recent_launch_rejected(
                        compact_full_launch_support_error
                    )

            dispatch_out = kwargs.get("out")
            if dispatch_out is None:
                dispatch_out = q_arg.new_empty(q_arg.shape)
            launch_plan = getattr(step_bound_meta, "compact_recent_launch_plan", None)
            if launch_plan is None or not bool(getattr(launch_plan, "valid", False)):
                if full_kv_handoff:
                    raise RuntimeError(
                        "full-KV handoff mixed-page overlay requires a valid "
                        "RRP full-recent launch plan"
                    )
                raise RuntimeError(
                    "compact mixed-page overlay requires a valid compact launch metadata plan"
                )
            launch_plan_page_size = int(getattr(launch_plan, "page_size", 0) or 0)
            if launch_plan_page_size <= 0 or launch_plan_page_size != int(page_size):
                raise RuntimeError(
                    "compact mixed-page overlay launch plan page_size mismatch"
                )
            if key_arg.dim() != 4 or value_arg.dim() != 4:
                raise RuntimeError(
                    "compact mixed-page overlay requires paged k/v tensors"
                )
            if (
                int(key_arg.shape[1]) != launch_plan_page_size
                or int(value_arg.shape[1]) != launch_plan_page_size
                or int(key_arg.shape[2]) != int(value_arg.shape[2])
            ):
                raise RuntimeError(
                    "compact mixed-page overlay paged k/v metadata mismatch"
                )
            (
                overlay_compact_valid_cpu,
                overlay_recent_first_cpu,
                overlay_recent_count_cpu,
                overlay_effective_k_len_cpu,
            ) = _resolve_selected_overlay_cpu_geometry(
                launch_plan=launch_plan,
                step_bound_meta=step_bound_meta,
                step_authority=step_authority,
                original_seq_lens=original_seq_lens,
                page_size=launch_plan_page_size,
                batch_size=batch_size,
            )
            launch_max_seqlen_k = max(
                overlay_effective_k_len_cpu[:batch_size],
                default=0,
            )
            if launch_max_seqlen_k <= 0:
                raise RuntimeError(
                    "compact mixed-page overlay requires positive launch max_seqlen_k"
                )
            _record_mixed_page_actual_route_family(
                controller,
                "resolved_row_ptr",
            )
            return run_compact_mixed_page_overlay_route(
                bridge=bridge,
                q=q_arg,
                k=key_arg,
                v=value_arg,
                out=dispatch_out,
                cu_seqlens_q=kwargs.get("cu_seqlens_q", cu_seqlens_q),
                max_seqlen_q=int(max_seqlen_q),
                seqused_k=real_seqused_k,
                max_seqlen_k=launch_max_seqlen_k,
                softmax_scale=kwargs.get("softmax_scale"),
                window_size=window_tuple,
                softcap=float(softcap),
                block_table=block_table_arg,
                controller=controller,
                state=state,
                step_authority=step_authority,
                step_bound_meta=step_bound_meta,
                page_size=launch_plan_page_size,
                num_kv_heads=int(key_arg.shape[2]),
                compact_valid_tokens_by_row=overlay_compact_valid_cpu[:batch_size],
                compact_offset_tokens_by_row=tuple(
                    int(v)
                    for v in tuple(
                        getattr(launch_plan, "compact_offset_tokens_cpu", tuple())
                    )[:batch_size]
                ),
                recent_first_page_by_row=overlay_recent_first_cpu[:batch_size],
                recent_page_count_by_row=overlay_recent_count_cpu[:batch_size],
                row_effective_k_by_row=overlay_effective_k_len_cpu[:batch_size],
                safe_page_id=0,
                num_splits=int(num_splits),
                cp_world_size=int(actual_cp_world_size),
                cp_rank=int(kwargs.get("cp_rank", 0) or 0),
                cp_tot_seqused_k=(
                    dense_input_cp_tot if isinstance(dense_input_cp_tot, torch.Tensor) else None
                ),
                q_v=kwargs.get("q_v"),
                q_descale=kwargs.get("q_descale"),
                k_descale=kwargs.get("k_descale"),
                v_descale=kwargs.get("v_descale"),
                s_aux=kwargs.get("s_aux"),
                **_mixed_page_resolver_kwargs_from_attn_metadata(attn_metadata),
            )

        # Remaining NO_COMPACT path: dense/full-K mixed-page resolver through
        # the same RRP overlay used by compact/replay paths.
        dispatch_out = kwargs.get("out")
        if dispatch_out is None:
            dispatch_out = q_arg.new_empty(q_arg.shape)
        no_compact_owner_plan = step_bound_meta.prologue_owner_plan
        dense_no_compact_max_seqlen_q = int(max_seqlen_q)
        if (
            bool(getattr(no_compact_owner_plan, "has_decode_rows", False))
            and not bool(getattr(no_compact_owner_plan, "has_prefill_rows", False))
        ):
            dense_no_compact_max_seqlen_q = 1
        if no_compact_owner_plan.has_prefill_rows and not no_compact_owner_plan.has_decode_rows:
            from patches.fa3_native.mixed_page_graph_descriptor import (
                PageResolverKind,
            )

            _record_mixed_page_actual_route_family(
                controller,
                "native_or_capture",
            )
            return bridge.mixed_page_attn_varlen_func(
                q=q_arg,
                k=key_arg,
                v=value_arg,
                max_seqlen_q=int(max_seqlen_q),
                cu_seqlens_q=kwargs.get("cu_seqlens_q", cu_seqlens_q),
                max_seqlen_k=int(original_max_seq_len or max_seqlen_q),
                seqused_k=real_seqused_k,
                q_v=kwargs.get("q_v"),
                softmax_scale=kwargs.get("softmax_scale"),
                causal=is_causal,
                window_size=list(window_tuple),
                softcap=float(softcap),
                block_table=block_table_arg,
                page_resolver_kind=int(PageResolverKind.NATIVE),
                return_softmax_lse=False,
                out=dispatch_out,
                scheduler_metadata=kwargs.get("scheduler_metadata"),
                q_descale=kwargs.get("q_descale"),
                k_descale=kwargs.get("k_descale"),
                v_descale=kwargs.get("v_descale"),
                num_splits=int(num_splits),
                s_aux=kwargs.get("s_aux"),
                cp_world_size=int(actual_cp_world_size),
                cp_rank=int(kwargs.get("cp_rank", 0) or 0),
                cp_tot_seqused_k=(
                    dense_input_cp_tot if isinstance(dense_input_cp_tot, torch.Tensor) else None
                ),
            )
        from patches.fa_sparse_runtime.mixed_prefill_decode_dispatch import (
            dispatch_mixed_prefill_decode_active_workload,
        )

        _record_mixed_page_actual_route_family(
            controller,
            "resolved_row_ptr",
        )
        return dispatch_mixed_prefill_decode_active_workload(
            q=q_arg,
            k=key_arg,
            v=value_arg,
            out=dispatch_out,
            cu_seqlens_q=kwargs.get("cu_seqlens_q", cu_seqlens_q),
            max_seqlen_q=dense_no_compact_max_seqlen_q,
            max_seqlen_k=int(original_max_seq_len or max_seqlen_q),
            seqused_k=real_seqused_k,
            softmax_scale=kwargs.get("softmax_scale"),
            window_size=window_tuple,
            softcap=float(softcap),
            block_table=block_table_arg,
            owner_plan=no_compact_owner_plan,
            bridge=bridge,
            controller=controller,
            state=state,
            step_authority=step_authority,
            step_bound_meta=step_bound_meta,
            step_ctx=step_ctx,
            q_v=kwargs.get("q_v"),
            q_descale=kwargs.get("q_descale"),
            k_descale=kwargs.get("k_descale"),
            v_descale=kwargs.get("v_descale"),
            s_aux=kwargs.get("s_aux"),
            scheduler_metadata=kwargs.get("scheduler_metadata"),
            num_splits=int(num_splits),
            cp_world_size=int(actual_cp_world_size),
            cp_rank=int(kwargs.get("cp_rank", 0) or 0),
            cp_tot_seqused_k=(
                dense_input_cp_tot if isinstance(dense_input_cp_tot, torch.Tensor) else None
            ),
            rail_decision=rail_decision,
        )

    old_v1_flash = getattr(v1_flash_attn, "flash_attn_varlen_func", None)
    old_fa_utils_flash = getattr(fa_utils, "flash_attn_varlen_func", None)
    setattr(v1_flash_attn, "flash_attn_varlen_func", _selected_wrapper)
    setattr(fa_utils, "flash_attn_varlen_func", _selected_wrapper)
    try:
        return _call_original_v1_flash_attn_forward(
            self=self,
            layer=layer,
            query=query,
            key=key,
            value=value,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            output=output,
            output_scale=output_scale,
            output_block_scale=output_block_scale,
        )
    finally:
        setattr(attn_metadata, "seq_lens", original_seq_lens)
        setattr(attn_metadata, "max_seq_len", original_max_seq_len)
        setattr(attn_metadata, "scheduler_metadata", original_scheduler_metadata)
        if old_v1_flash is not None:
            setattr(v1_flash_attn, "flash_attn_varlen_func", old_v1_flash)
        if old_fa_utils_flash is not None:
            setattr(fa_utils, "flash_attn_varlen_func", old_fa_utils_flash)


def _run_capture_only_mixed_forward(
    *,
    self,
    layer,
    query,
    key,
    value,
    kv_cache,
    attn_metadata,
    output=None,
    output_scale=None,
    output_block_scale=None,
):
    v1_flash_attn, fa_utils = _get_v1_flash_attn_modules()

    from patches.fa3_native.capture_cohort import (
        CaptureCohortCoordinator,
        publish_capture_cohort_completion,
    )
    from patches.fa3_native.capture_ownership import CHUNK_COHORT
    from patches.fa3_native.forward_capture import prepare_capture_forward_side_outputs
    from patches.fa3_native.install import load_vendored_flash_attn_bridge
    from patches.fa3_native.postprocess import (
        run_capture_postprocess_job_sequence_for_cohort,
        run_capture_postprocess_job_if_needed,
        run_prefill_capture_postprocess_if_needed,
    )
    from patches.fa3_native.row_plan import build_mixed_page_row_plan
    from patches.sparse_constants import _CAPTURE_CHUNK, _CAPTURE_IN_FLIGHT, _CAPTURE_REDUCE_GROUP
    from patches.fa3_native.ring_capture import ring_scratch_slot, ring_depth, RingWarFence
    from patches.sparse_types import CapturePostprocessJob, SelectorBatchPayload
    from patches.sparse_utils import (
        _make_selector_fast_signature,
        _selector_fixed_k_enabled,
    )
    from patches.vllm_sparse_patch import (
        _prepare_prefill_capture_payload,
        _prepare_refresh_capture_payload,
    )

    controller = _get_global_controller()
    if controller is None or not bool(getattr(getattr(controller, "config", None), "enabled", False)):
        raise RuntimeError(
            "native FA3 capture mixed route requires a live sparse controller"
        )
    step_ctx = _resolve_mixed_route_step_context(
        controller=controller,
        attn_metadata=attn_metadata,
        stage="native FA3 capture mixed route",
    )
    step_authority = getattr(step_ctx, "step_authority", None)
    if step_authority is None:
        step_authority = getattr(controller, "step_authority", None)
    if step_authority is None:
        raise RuntimeError(
            "native FA3 capture mixed route requires step_authority"
        )
    step_envelope = getattr(step_ctx, "step_envelope_v2", None)
    if step_envelope is None:
        raise RuntimeError(
            "native FA3 capture mixed route requires step_envelope_v2"
        )
    step_bound_meta = getattr(controller, "step_bound_meta", None)
    if step_bound_meta is None:
        raise RuntimeError(
            "native FA3 capture mixed route requires step_bound_meta"
        )

    cu_seqlens_q = getattr(attn_metadata, "query_start_loc")
    seqused_k = getattr(attn_metadata, "seq_lens")
    batch_size = int(cu_seqlens_q.shape[0]) - 1
    key_cache, value_cache = kv_cache.unbind(0)
    cache_key = key_cache.data_ptr()
    state = controller.get_state(
        cache_key,
        query.shape[1],
        key_cache.shape[2],
        query.device,
        head_dim=query.shape[2],
        kv_cache_dtype=key_cache.dtype,
    )
    _ensure_layer_state_snapshot_alignment(
        state=state,
        step_ctx=step_ctx,
        step_envelope=step_envelope,
    )
    if hasattr(controller, "get_step_prefill_plan_by_req") and hasattr(controller, "prepare_step_logits_buffers"):
        capture_plan_by_req, _ = controller.get_step_prefill_plan_by_req(step_context=step_ctx)
        controller.prepare_step_logits_buffers(
            state=state,
            step_context=step_ctx,
            capture_plan_by_req=capture_plan_by_req if capture_plan_by_req else None,
            seqused_k=seqused_k,
            max_seqlen_k=int(getattr(attn_metadata, "max_seq_len", 0) or getattr(step_ctx, "max_seq_len", 0) or 0),
            block_size=_infer_paged_kv_page_size(key_cache),
            num_heads=query.shape[1],
            device=query.device,
        )
        step_authority = getattr(step_ctx, "step_authority", None)
        if step_authority is None:
            step_authority = getattr(controller, "step_authority", None)
        if step_authority is None:
            raise RuntimeError(
                "native FA3 capture mixed route requires logits-synchronised step_authority"
            )
    _ensure_step_prologue(
        controller=controller,
        step_ctx=step_ctx,
        step_authority=step_authority,
        step_bound_meta=step_bound_meta,
        batch_size=batch_size,
        kv_cache=kv_cache,
        device=query.device,
        canonical_state=state,
    )
    owner_plan = getattr(step_bound_meta, "prologue_owner_plan", None)
    rail_decision = getattr(step_bound_meta, "prologue_rail_decision", None)
    row_plan = getattr(step_bound_meta, "prologue_selected_row_plan", None)
    if owner_plan is None or rail_decision is None or row_plan is None:
        from patches.fa_sparse_runtime.mixed_prefill_decode_owner import (
            resolve_mixed_prefill_decode_owner_plan,
        )
        from patches.fa_sparse_runtime.compact_recent_route_authority import (
            resolve_compact_recent_rail_mode,
        )

        owner_plan = resolve_mixed_prefill_decode_owner_plan(step_authority)
        rail_decision = resolve_compact_recent_rail_mode(step_authority)
        row_plan = build_mixed_page_row_plan(
            step_authority,
            batch_size=batch_size,
            device=query.device,
        )
    if not owner_plan.has_capture_rows:
        raise RuntimeError(
            "native FA3 capture-only mixed route resolved zero capture rows"
        )

    block_table = getattr(attn_metadata, "block_table", None)
    default_cp_world_size = int(getattr(attn_metadata, "cp_world_size", 1) or 1)

    slot_by_row = tuple(int(v) for v in getattr(step_envelope, "slot_by_row"))
    # Scratch rows are ordered by phase-local slot order.  That makes prefill
    # and refresh scratch tapes contiguous views, so mixed-phase last_n==1
    # selector inputs do not need index_select/copy.
    prefill_rows_by_slot = tuple(
        sorted(
            (int(row) for row in owner_plan.prefill_capture_rows),
            key=lambda row: int(slot_by_row[row]) if 0 <= int(row) < len(slot_by_row) else -1,
        )
    )
    decode_rows_by_slot = tuple(
        sorted(
            (int(row) for row in owner_plan.decode_capture_rows),
            key=lambda row: int(slot_by_row[row]) if 0 <= int(row) < len(slot_by_row) else -1,
        )
    )
    # Rev 2: producer_rows_cpu from owner_plan (0 GPU readback).
    producer_rows_cpu = tuple(prefill_rows_by_slot + decode_rows_by_slot)
    # Upper bounds from CPU-side step_authority + layout metadata.
    logits_last_n_by_row = tuple(int(v) for v in getattr(step_authority, "logits_last_n_by_row", ()))
    planned_max_capture_last_n = max(
        (logits_last_n_by_row[r] for r in producer_rows_cpu), default=0
    )
    context_lengths = tuple(int(v) for v in getattr(step_authority, "context_kv_len_by_row", ()))
    planned_max_capture_k = max(
        (context_lengths[r] for r in producer_rows_cpu if r < len(context_lengths)),
        default=0,
    )
    _actual_planned_max_capture_k = int(planned_max_capture_k)
    # 262k OOM fix (prebuild byte-match): force the capture-scratch WIDTH to the
    # single max-bucket the profile prebuild reserved, so the deferred capture
    # scratch_key (forward_capture.py:430-436, shape-keyed via
    # scratch_storage_shape[-1] == max_capture_k) BYTE-MATCHES the pre-allocated
    # slab -> live cache HIT (forward_capture.py:444) -> the ~3.14GB lazy
    # torch.empty at forward_capture.py:459 NEVER fires on the 262k hot path.
    # WIDTH only (store writes only valid_k_len; forward_capture.py:358-360), so
    # the extra width past the item's real context is never read. All large (>0)
    # prefills round to the ONE prebuilt slab (a 100k and a 262k prefill share it).
    # _capture_kv_max_bucket is the grid-aligned(max_model_len) int stamped by the
    # profile prebuild (_prebuild_capture_buffers); 0 (prebuild skipped / short-ctx
    # / non-capture run) => no-op => exact HEAD behaviour (fail-safe).
    _capture_kv_max_bucket = int(getattr(controller, "_capture_kv_max_bucket", 0) or 0)
    if _capture_kv_max_bucket > 0 and int(planned_max_capture_k) > 0:
        planned_max_capture_k = int(_capture_kv_max_bucket)
    _capture_ownership_plan = getattr(
        controller, "_capture_ownership_plan", None
    )
    _capture_ownership_mode_value = ""
    _capture_ownership_plan_signature = ""
    if _capture_ownership_plan is not None:
        _capture_ownership_mode_value = _capture_ownership_mode(
            _capture_ownership_plan
        )
        _capture_ownership_plan_signature = str(
            getattr(_capture_ownership_plan, "signature_sha256")
        )
    if _capture_ownership_mode_value == CHUNK_COHORT:
        if not bool(getattr(_capture_ownership_plan, "selector_fixed_k", False)):
            raise RuntimeError(
                "E_SFI_CAPTURE_COHORT_FIXED_K: stamped chunk cohort plan did "
                "not prove fixed-K selector ownership"
            )
        if not bool(_selector_fixed_k_enabled()):
            raise RuntimeError(
                "E_SFI_CAPTURE_COHORT_FIXED_K: live selector fixed-K contract "
                "drifted after ownership planning"
            )

    # Build slot sets from owner_plan (CPU) instead of GPU row_plan iteration.
    prefill_slot_set: set[int] = set()
    refresh_slot_set: set[int] = set()
    for batch_row in producer_rows_cpu:
        if batch_row >= len(slot_by_row):
            raise RuntimeError("step envelope slot_by_row coverage is insufficient for capture rows")
        slot = int(slot_by_row[batch_row])
        if slot < 0:
            raise RuntimeError("capture producer rows require valid slot assignments")
        if batch_row in owner_plan.prefill_capture_rows:
            prefill_slot_set.add(slot)
        else:
            refresh_slot_set.add(slot)

    # chunk_query_lengths from CPU q_lens_by_row (no .any().item()).
    chunk_query_lengths = None
    q_lens_cpu = tuple(int(v) for v in getattr(step_authority, "q_lens_by_row", ()))
    has_prefill_any = any(q_lens_cpu[r] > 1 for r in range(batch_size) if r < len(q_lens_cpu))
    if has_prefill_any:
        chunk_query_lengths = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).to(
            device=query.device,
            dtype=torch.long,
        )

    global_layer_index = int(controller.layer_index_by_cache_key.get(cache_key, -1))
    if global_layer_index < 0:
        raise RuntimeError("native FA3 capture mixed route requires registered layer cache key")
    # Freeze the step owner once and carry the same identity through the
    # postprocess job and every transport payload.  Deferred launch validation
    # must never reconstruct ownership from mutable controller state.
    capture_handle_id = int(getattr(step_ctx, "step_handle_id", -1))
    capture_handle_generation = int(
        getattr(step_ctx, "step_handle_generation", -1)
    )
    capture_epoch = int(getattr(step_authority, "epoch", -1))
    capture_step_identity_token = int(
        getattr(step_ctx, "step_identity_token", 0)
    )
    prefill_layout = None
    prefill_slot_list = sorted(int(slot) for slot in prefill_slot_set)
    if prefill_slot_list:
        prefill_layout = controller._get_step_capture_layout(
            phase="prefill",
            state=state,
            step_context=step_ctx,
            global_layer_index=global_layer_index,
            slot_list=prefill_slot_list,
            seqused_k=seqused_k,
            num_heads=query.shape[1],
            device=query.device,
            chunk_query_lengths=chunk_query_lengths,
            prepared_only=bool(
                getattr(controller, "_prefill_capture_meta_arena_enabled", False)
            ),
        )
        if prefill_layout is None:
            raise RuntimeError("native FA3 capture mixed route failed to build prefill capture layout")

    refresh_layout = None
    active_refresh_slot_list: list[int] = []
    if refresh_slot_set:
        refresh_slot_list = tuple(
            int(slot) for slot in getattr(step_authority, "refresh_capture_slot_list", tuple())
        )
        if refresh_slot_list:
            missing_refresh_slots = sorted(int(slot) for slot in refresh_slot_set if int(slot) not in set(refresh_slot_list))
            if missing_refresh_slots:
                raise RuntimeError(
                    "native FA3 capture mixed route refresh slot_list missing producer slots"
                )
            active_refresh_slot_list = [
                int(slot) for slot in refresh_slot_list if int(slot) in refresh_slot_set
            ]
        else:
            active_refresh_slot_list = sorted(int(slot) for slot in refresh_slot_set)
        if not active_refresh_slot_list:
            raise RuntimeError(
                "native FA3 capture mixed route resolved zero active refresh slots"
            )
        refresh_layout = controller._get_step_capture_layout(
            phase="refresh",
            state=state,
            step_context=step_ctx,
            global_layer_index=global_layer_index,
            slot_list=active_refresh_slot_list,
            seqused_k=seqused_k,
            num_heads=query.shape[1],
            device=query.device,
            chunk_query_lengths=None,
        )
        if refresh_layout is None:
            raise RuntimeError("native FA3 capture mixed route failed to build refresh capture layout")
    if prefill_layout is None and refresh_layout is None:
        raise RuntimeError("native FA3 capture mixed route resolved zero capture layouts")

    capture_chunk_id, _, slot_in_chunk = (
        controller._map_global_layer_to_capture_slot(global_layer_index)
    )
    _rg = int(_CAPTURE_REDUCE_GROUP)
    _ring_slabs = ring_depth(_rg, int(_CAPTURE_IN_FLIGHT))
    _ring_slot = ring_scratch_slot(int(global_layer_index), _rg, int(_CAPTURE_IN_FLIGHT))
    _ek_chunk_id = 0 if _rg > 0 else (int(global_layer_index) // max(1, int(_CAPTURE_CHUNK)))
    _ek_depth = int(_ring_slabs) if _rg > 0 else int(_CAPTURE_CHUNK)
    if _capture_ownership_mode_value == CHUNK_COHORT:
        _ring_slot, _ek_chunk_id, _ek_depth = (
            _chunk_cohort_runtime_scratch_binding(
                plan=_capture_ownership_plan,
                global_layer_index=int(global_layer_index),
                slot_in_chunk=int(slot_in_chunk),
                aligned_k_bucket=int(_capture_kv_max_bucket),
                rows_bucket=int(
                    getattr(controller, "_capture_rows_bucket", 0) or 0
                ),
                last_n_bucket=int(
                    getattr(controller, "_capture_last_n_bucket", 0) or 0
                ),
                actual_k=int(_actual_planned_max_capture_k),
                actual_rows=int(len(producer_rows_cpu)),
                actual_last_n=int(planned_max_capture_last_n),
                heads_per_rank=int(query.shape[1]),
                chunk=int(_CAPTURE_CHUNK),
                in_flight=int(_CAPTURE_IN_FLIGHT),
                baseline_reduce_group=int(_CAPTURE_REDUCE_GROUP),
            )
        )
    _effective_capture_ring_slot = int(_ring_slot)
    _effective_capture_scratch_depth = int(_ek_depth)
    _effective_capture_cohort_size = (
        int(_ek_chunk_id)
        if _capture_ownership_mode_value == CHUNK_COHORT
        else int(_CAPTURE_CHUNK)
    )
    _capture_ring_fence_active = bool(
        _rg > 0 or _capture_ownership_mode_value == CHUNK_COHORT
    )
    if _capture_ring_fence_active:
        _fence = getattr(controller, "_ring_war_fence", None)
        if _fence is None:
            _fence = RingWarFence()
            setattr(controller, "_ring_war_fence", _fence)
        _prev_evt = _fence.war_event_before_capture(
            int(_effective_capture_ring_slot)
        )
        if _prev_evt is not None:
            # [RING-GUARD 2026-07-03] a residual fence event while the current stream
            # is capturing would bake an external-event wait into the graph (capture
            # failure or a stale WAR edge on every replay). Init-time captures always
            # see a drained fence; anything else is a program error - fail fast.
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "ring WAR fence holds a residual event while the current stream is "
                    "capturing; graph capture must start with a drained fence"
                )
            # WAR: order this layer's ring-slot capture AFTER the prior occupant's reduce read.
            torch.cuda.current_stream(device=torch.device(query.device)).wait_event(_prev_evt)
    defer_capture_postprocess_requested = (
        os.environ.get("VLLM_SPARSE_DEFER_CAPTURE_POSTPROCESS", "0") == "1"
    )
    early_capture_postprocess_requested = (
        os.environ.get("VLLM_SPARSE_EARLY_CAPTURE_POSTPROCESS", "0") == "1"
    )
    async_capture_postprocess_requested = (
        os.environ.get("VLLM_SPARSE_ASYNC_CAPTURE_POSTPROCESS", "0") == "1"
    ) and not defer_capture_postprocess_requested

    def _auto_early_capture_postprocess_requested() -> bool:
        if (
            defer_capture_postprocess_requested
            or early_capture_postprocess_requested
            or async_capture_postprocess_requested
        ):
            return False
        if int(planned_max_capture_last_n) <= 1:
            return False
        if prefill_layout is None or refresh_layout is not None:
            return False
        one_shot_bootstrap_only = bool(
            getattr(
                getattr(controller, "config", None),
                "one_shot_bootstrap_only",
                False,
            )
        )
        if not one_shot_bootstrap_only:
            return False
        if not torch.cuda.is_available():
            return False
        if torch.cuda.is_current_stream_capturing():
            return False
        async_refresh_enabled = getattr(controller, "_async_refresh_enabled", None)
        if not callable(async_refresh_enabled) or not bool(async_refresh_enabled()):
            return False
        ensure_refresh_stream = getattr(controller, "_ensure_refresh_stream", None)
        if not callable(ensure_refresh_stream):
            return False
        ensure_refresh_stream(torch.device(query.device))
        return getattr(controller, "refresh_stream", None) is not None

    # CHUNK_COHORT is an immutable structural plan, not a legacy last_n/layout
    # heuristic.  Its execution owner is selected once after semantic work is
    # known; do not let the old prefill-only auto gate infer it independently.
    auto_early_capture_postprocess_requested = (
        False
        if _capture_ownership_mode_value == CHUNK_COHORT
        else _auto_early_capture_postprocess_requested()
    )
    if auto_early_capture_postprocess_requested:
        defer_capture_postprocess_requested = True
        early_capture_postprocess_requested = True
    # 262k OOM fix (last_n bucketing; symmetric with the EDIT-1 kv_max force-form):
    # round the DEFER capture-scratch last_n dim UP to the configured
    # prefill_last_n_query bucket so scratch_storage_shape[-2] is INVARIANT and the
    # deferred scratch_key HITS the profile-prebuilt slab. Without this the FINAL
    # prefill tail chunk (q_len < 16 -> planned_max_capture_last_n = min(16, q_len)
    # < 16; selector_compute_mixin.py:2255-2264) mints a DISTINCT shape[-2] -> cache
    # MISS at forward_capture.py:444 -> the ~3GB lazy torch.empty at :459 re-OOMs the
    # 262k hot path (launch-scale hazard: prompt_len mod chunk in [2,15]). SAFE: the
    # capture store kernel (FA4 CuTe store_capture_scores / FA3 C++) uses shape[-2]
    # only as the row STRIDE and clamps writes to min(per_row_last_n, shape[-2],
    # seqlen_q); postprocess reads the per-row ACTUAL last_n (postprocess.py:354), so
    # rows [actual:bucket) stay UNWRITTEN + UNREAD -> identical numerics (the same
    # mechanism the kv_max width bucket relies on). GUARD > 1: never promote a decode
    # last_n==1 step (the auto-early gate at :12172 already used the RAW planned
    # value); per-row logits_last_n_by_row (the actual write counts) pass UNCHANGED.
    # 0 bucket (prebuild skipped) => no-op. A stamped chunk cohort always uses its
    # prebuilt R capacity, including an actual last_n==1 tail; unused rows remain
    # unwritten and unread, while preserving the single cache key.
    _capture_last_n_bucket = int(getattr(controller, "_capture_last_n_bucket", 0) or 0)
    _planned_max_capture_last_n_alloc = int(planned_max_capture_last_n)
    if _capture_ownership_mode_value == CHUNK_COHORT:
        _planned_max_capture_last_n_alloc = int(
            getattr(_capture_ownership_plan, "last_n")
        )
    elif _capture_last_n_bucket > 1 and _planned_max_capture_last_n_alloc > 1:
        _planned_max_capture_last_n_alloc = max(
            _planned_max_capture_last_n_alloc, int(_capture_last_n_bucket)
        )
    side_outputs = prepare_capture_forward_side_outputs(
        prefill_layout=prefill_layout,
        refresh_layout=refresh_layout,
        row_plan=row_plan,
        slot_in_chunk=int(slot_in_chunk),
        ring_scratch_slot=int(_effective_capture_ring_slot),
        seqused_k=seqused_k,
        device=query.device,
        producer_rows_cpu=producer_rows_cpu,
        max_capture_k_cpu=planned_max_capture_k,
        max_capture_last_n_cpu=_planned_max_capture_last_n_alloc,
        prefill_producer_rows_cpu=tuple(sorted(int(row) for row in owner_plan.prefill_capture_rows)),
        decode_producer_rows_cpu=tuple(sorted(int(row) for row in owner_plan.decode_capture_rows)),
        row_capture_last_n_cpu=logits_last_n_by_row[:batch_size],
        seqused_k_cpu=context_lengths[:batch_size],
        scratch_cache_owner=controller,
        scratch_cache_extra_key=(
            (
                "defer_postprocess_chunk",
                _effective_capture_cohort_size,
                _effective_capture_scratch_depth,
            )
            if _capture_ownership_mode_value == CHUNK_COHORT
            else
            (
                "defer_postprocess_chunk",
                # 262k OOM fix: drop the per-step nonce (step_handle_id/generation) from
                # the deferred capture scratch key, keeping ONLY the chunk_id
                # (global_layer_index // _CAPTURE_CHUNK). The nonce forced a NEW ~3.14GB
                # slab per step on the lazy torch.empty (forward_capture.py:459) -> 262k
                # prefill OOM. chunk_id keying bounds live slabs to num_chunks
                # (= ceil(num_layers / _CAPTURE_CHUNK)), reused across steps, and KEEPS the
                # original per-chunk buffer DISJOINTNESS: distinct chunk_id -> distinct
                # slab, so layers sharing a buf_id (chunk_id % _CAPTURE_IN_FLIGHT) within
                # one forward (e.g. layer 0 chunk0 and layer 28 chunk2) never alias the
                # same scratch slot -> NO intra-forward write-after-read race against the
                # auto-early refresh_stream reduce, with ZERO added sync (a buf_id key
                # WOULD alias them; chunk_id does not). Cross-step reuse of a chunk_id slab
                # is ordered by data dependency (the prior request consumes its reduce +
                # decodes before the next sequential n_proc=1 / max_num_partial_prefills=1
                # prefill). [RING-AUDIT 2026-07-03] that serial assumption is load-bearing:
                # under G=0 (env override; default G=1 ring) + deferred producer, a
                # scheduler that co-runs a NEW prefill before the prior step's flush
                # consumed this chunk_id slab would overwrite unread scratch with no
                # event guard. If G=0 ever becomes a served mode, add a per-(state,
                # chunk_id) flush-completion event (record after the flush tape stack,
                # wait before the next step's first slab write) before enabling it.
                # The profile prebuild (_prebuild_capture_buffers) pre-reserves
                # one slab per chunk_id so the 262k capture HITS (forward_capture.py:444)
                # and :459 never fires on the hot path.
                _ek_chunk_id,
                _ek_depth,
            )
            if defer_capture_postprocess_requested
            else (
                "async_postprocess_chunk",
                _ek_chunk_id,
                _ek_depth,
            )
            if async_capture_postprocess_requested
            else None
        ),
    )
    # [CHUNKED-CAPTURE-ACCUMULATE 2026-07-06] 跨片捕获片（含 q_len==1 尾片）
    # 必须走 scratch+reduce（accumulate 合并）：direct-capture 直写 tape 是纯
    # 覆写，会重现尾片清盖前片。从 controller 读提交步物化（token 对账）。
    _accum_token = int(getattr(controller, "_step_capture_accum_token", -1))
    _step_epoch_now = int(getattr(step_authority, "epoch", -2))
    step_capture_accum_prev_rows_cpu: tuple[int, ...] = tuple()
    step_capture_accum_prev_capacity_cpu: tuple[int, ...] = tuple()
    if _accum_token == _step_epoch_now:
        step_capture_accum_prev_rows_cpu = tuple(
            getattr(controller, "_step_capture_accum_prev_rows_by_row", tuple()) or tuple()
        )
        step_capture_accum_prev_capacity_cpu = tuple(
            getattr(controller, "_step_capture_accum_prev_capacity_by_row", tuple())
            or tuple()
        )
    _step_has_accum_capture_rows = any(
        int(v) >= 0 for v in step_capture_accum_prev_rows_cpu
    )
    capture_postprocess_state: dict[str, object] = {
        "force_scratch_capture": bool(_step_has_accum_capture_rows),
        "direct_capture_phase": "",
    }
    # Production pure last_n==1 capture may write directly to the phase tape.
    # Diagnostic compare keeps the scratch path so before/after postprocess
    # records still mean what they say.
    skip_postprocess_rows = tuple()

    def _ring_lastn1_drain_active() -> bool:
        # [RING-LASTN1-DRAIN 2026-07-03] prefill-only steps under the per-G
        # scratch RING must drain last_n==1 rows at postprocess time (dedicated
        # lastn1 copy arm) instead of carrying raw ring-scratch views to the
        # chunk-tail flush -- the ring slot is rewritten G*in_flight layers
        # later (depth << chunk), so a chunk-tail read returns overwritten data
        # on mixed lastn1/gt1 steps. Direct-capture phases never write scratch
        # (draining would copy garbage) -- excluded via the phase checks below.
        # [RING-LASTN1-DRAIN-MIXED 2026-07-03 晚] 混合步(错峰 bootstrap:同一
        # batched step 里 prefill-capture 行与 refresh decode 行并存)时
        # direct_capture_phase==""、kernel 把 refresh 行的 B1 logits 写进
        # per-step 单一 scratch(每层复写,只有末层数据存活),而 chunk 批量
        # deferred selector 逐层消费——旧门把 refresh 步一律排除是错的:纯
        # refresh 步(direct_capture_phase=="refresh")直写 layout base 无需
        # drain,但混合步必须 drain(decode 行经行循环选 refresh_out 目标拷回
        # layout base),否则 base 无数据、payload 只能携带被复写的 scratch 视
        # 图(4B bs8 错峰实证:14 层同 storage 同 offset,selector storage 证明
        # layer_stride=0 fail-fast)。
        _phase = str(capture_postprocess_state.get("direct_capture_phase", ""))
        return bool(
            _capture_ring_fence_active
            and (refresh_layout is None or _phase == "")
            and _phase != "prefill"
        )

    semantic_snapshot = controller._get_step_semantic_snapshot()
    alpha_log_f = float(getattr(semantic_snapshot, "alpha_log_f"))
    if not math.isfinite(alpha_log_f):
        raise ValueError(f"Invalid alpha_log_f from snapshot: {alpha_log_f}")
    bridge = load_vendored_flash_attn_bridge()
    original_seq_lens = getattr(attn_metadata, "seq_lens")
    original_max_seq_len = getattr(attn_metadata, "max_seq_len", None)
    original_scheduler_metadata = getattr(attn_metadata, "scheduler_metadata", None)
    page_size = _infer_paged_kv_page_size(kv_cache)
    if page_size <= 0:
        raise RuntimeError("native FA3 capture mixed route requires positive page_size")
    real_seqused_k = _resolve_step_real_seqused_k(
        step_bound_meta=step_bound_meta,
        original_seq_lens=original_seq_lens,
        device=query.device,
        batch_size=batch_size,
        cache_owner=controller,
    )
    if (
        not isinstance(real_seqused_k, torch.Tensor)
        or real_seqused_k.device != query.device
        or real_seqused_k.dtype != torch.int32
        or real_seqused_k.dim() != 1
        or int(real_seqused_k.numel()) != int(batch_size)
    ):
        raise RuntimeError(
            "native FA3 capture mixed route requires full-batch real seqused_k truth"
        )
    _cache_full_cudagraph_replay_payload_refs(
        state=state,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table,
        q=query,
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=real_seqused_k,
        softmax_scale=float(getattr(self, "scale", 1.0)),
        softcap=float(getattr(self, "logits_soft_cap", 0.0) or 0.0),
        window_size=getattr(self, "sliding_window", None),
        alibi_slopes=getattr(self, "alibi_slopes", None),
        k_descale=None,
    )
    def _capture_wrapper(*args, **kwargs):
        _ctrl = _get_global_controller()
        if _ctrl is None or getattr(getattr(_ctrl, "config", None), "attn_mode", "compact_recent") != "compact_recent":
            if _ORIGINAL_V1_FLASH_ATTN_VARLEN_FUNC is None:
                raise RuntimeError(
                    "_capture_wrapper bypass: _ORIGINAL_V1_FLASH_ATTN_VARLEN_FUNC not initialized"
                )
            return _ORIGINAL_V1_FLASH_ATTN_VARLEN_FUNC(*args, **kwargs)
        if args:
            raise RuntimeError("capture mixed wrapper expects keyword arguments only")
        _strip_identity_descale_kwargs_inplace(kwargs)
        _reject_legacy_selected_request_level_kwargs(
            wrapper_name="capture mixed wrapper",
            kwargs=kwargs,
        )
        _ensure_step_prologue(
            controller=controller,
            step_ctx=step_ctx,
            step_authority=step_authority,
            step_bound_meta=step_bound_meta,
            batch_size=batch_size,
            kv_cache=kv_cache,
            device=query.device,
            canonical_state=state,
        )
        actual_cp_world_size = int(kwargs.get("cp_world_size", default_cp_world_size) or 1)
        dense_input_cp_tot = kwargs.get("cp_tot_seqused_k")
        alibi_slopes = kwargs.pop("alibi_slopes", None)
        # Rev 2: strip legacy sel-page kwargs (caller may still pass under env=0).
        kwargs.pop("fa_version", None)
        kwargs.pop("max_seqlen_k", None)
        kwargs.pop("seqused_k", None)
        kwargs.pop("cp_world_size", None)
        kwargs.pop("cp_tot_seqused_k", None)
        kwargs.pop("selected_page_table_i32", None)
        kwargs.pop("row_consume_mode_i32", None)
        kwargs.pop("selected_seqused_k_by_head_i32", None)
        kwargs.pop("cp_selected_seqused_k_by_head_i32", None)
        kwargs.pop("scheduler_metadata", None)
        if alibi_slopes is not None:
            raise RuntimeError("native FA3 capture mixed route does not support ALiBi yet")

        from patches.fa_sparse_runtime.mixed_prefill_decode_dispatch import (
            dispatch_capture_mixed_owner,
        )

        out_tensor = kwargs.pop("out", None)
        if out_tensor is None:
            out_tensor = query.new_zeros(query.shape)

        resolver_kwargs = _mixed_page_resolver_kwargs_from_attn_metadata(attn_metadata)
        # Capture still computes full-KV scores, but when full-cudagraph replay
        # has RRP metadata attached the capture graph must use the same
        # mixed-page ABI.  The RRP row table can represent full/native rows
        # during capture and be updated in-place for compact rows on replay.
        _record_mixed_page_actual_route_family(
            controller,
            "resolved_row_ptr" if resolver_kwargs else "native_or_capture",
        )
        return dispatch_capture_mixed_owner(
            q=kwargs.get("q", query),
            k=kwargs.get("k", key),
            v=kwargs.get("v", value),
            out=out_tensor,
            cu_seqlens_q=kwargs.get("cu_seqlens_q", cu_seqlens_q),
            max_seqlen_q=int(kwargs.get("max_seqlen_q") or 1),
            max_seqlen_k=int(original_max_seq_len or kwargs.get("max_seqlen_q") or 1),
            seqused_k=real_seqused_k,
            softmax_scale=kwargs.get("softmax_scale"),
            window_size=(-1, -1),
            softcap=float(kwargs.get("softcap", 0.0) or 0.0),
            block_table=kwargs.get("block_table", block_table),
            owner_plan=owner_plan,
            side_outputs=side_outputs,
            bridge=bridge,
            controller=controller,
            state=state,
            step_authority=step_authority,
            step_bound_meta=step_bound_meta,
            step_ctx=step_ctx,
            rail_decision=rail_decision,
            q_v=kwargs.get("q_v"),
            q_descale=kwargs.get("q_descale"),
            k_descale=kwargs.get("k_descale"),
            v_descale=kwargs.get("v_descale"),
            s_aux=kwargs.get("s_aux"),
            scheduler_metadata=original_scheduler_metadata,
            num_splits=int(kwargs.get("num_splits", 0) or 0),
            cp_world_size=actual_cp_world_size,
            cp_rank=int(kwargs.get("cp_rank", 0) or 0),
            cp_tot_seqused_k=(
                dense_input_cp_tot if isinstance(dense_input_cp_tot, torch.Tensor) else None
            ),
            capture_postprocess_state=capture_postprocess_state,
            **resolver_kwargs,
        )

    old_v1_flash = getattr(v1_flash_attn, "flash_attn_varlen_func", None)
    old_fa_utils_flash = getattr(fa_utils, "flash_attn_varlen_func", None)
    setattr(v1_flash_attn, "flash_attn_varlen_func", _capture_wrapper)
    setattr(fa_utils, "flash_attn_varlen_func", _capture_wrapper)
    try:
        result = _call_original_v1_flash_attn_forward(
            self=self,
            layer=layer,
            query=query,
            key=key,
            value=value,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            output=output,
            output_scale=output_scale,
            output_block_scale=output_block_scale,
        )
    finally:
        setattr(attn_metadata, "seq_lens", original_seq_lens)
        setattr(attn_metadata, "max_seq_len", original_max_seq_len)
        setattr(attn_metadata, "scheduler_metadata", original_scheduler_metadata)
        if old_v1_flash is not None:
            setattr(v1_flash_attn, "flash_attn_varlen_func", old_v1_flash)
        if old_fa_utils_flash is not None:
            setattr(fa_utils, "flash_attn_varlen_func", old_fa_utils_flash)


    capture_postprocess_job: CapturePostprocessJob | None = None

    def _capture_postprocess_required() -> bool:
        producer_rows_cpu = tuple(
            int(v) for v in tuple(getattr(side_outputs, "producer_rows_cpu", tuple()) or tuple())
        )
        row_capture_last_n_cpu = tuple(
            max(0, int(v))
            for v in tuple(getattr(side_outputs, "row_capture_last_n_cpu", tuple()) or tuple())
        )
        if not producer_rows_cpu:
            return True
        if len(row_capture_last_n_cpu) <= max(producer_rows_cpu, default=-1):
            return True
        if any(int(row_capture_last_n_cpu[row]) > 1 for row in producer_rows_cpu):
            return True
        # [RING-LASTN1-DRAIN 2026-07-03] lastn1-only scratch steps still need the
        # postprocess run under the ring: the lastn1 copy arm is the only fenced
        # consumer of their ring slots (the raw-view path is retired for prefill).
        return _ring_lastn1_drain_active() and any(
            int(row_capture_last_n_cpu[row]) == 1 for row in producer_rows_cpu
        )

    capture_postprocess_required = bool(_capture_postprocess_required())
    # A stamped cohort owns postprocess jobs, not every attention call carrying
    # a capture route bit. Chunked prefill legitimately emits metadata-not-ready
    # calls (last_n==0) and direct last-n=1 calls with no postprocess work; those
    # calls must create neither a deferred job nor a completion event. Conversely,
    # a mixed prefill/refresh call with real scratch work must keep the immutable
    # cohort owner even though the legacy prefill-only auto gate is ineligible.
    # Resolve that distinction once from the semantic work predicate. This adds
    # no CUDA work and is outside the steady decode path.
    if (
        _capture_ownership_mode_value == CHUNK_COHORT
        and capture_postprocess_required
        and not defer_capture_postprocess_requested
        and not early_capture_postprocess_requested
        and not async_capture_postprocess_requested
    ):
        defer_capture_postprocess_requested = True
        early_capture_postprocess_requested = True

    def _run_capture_postprocess() -> bool:
        return run_prefill_capture_postprocess_if_needed(
            direct_capture_phase=str(capture_postprocess_state.get("direct_capture_phase", "")),
            scratch_capture_scores=side_outputs.scratch_capture_scores,
            producer_rows_i32=side_outputs.producer_rows_i32,
            row_capture_last_n_i32=side_outputs.row_capture_last_n_i32,
            row_is_prefill_producer=side_outputs.row_is_prefill_producer,
            seqused_k=real_seqused_k,
            active_capture_row_by_batch_row_i32=side_outputs.active_capture_row_by_batch_row_i32,
            prefill_out_capture_scores=side_outputs.prefill_out_capture_scores,
            prefill_out_log_f_denoms=side_outputs.prefill_out_log_f_denoms,
            refresh_out_capture_scores=side_outputs.refresh_out_capture_scores,
            refresh_out_log_f_denoms=side_outputs.refresh_out_log_f_denoms,
            prefill_out_kv_len_per_capture_row_i32=(
                prefill_layout.kv_len_per_row_i32 if prefill_layout is not None else None
            ),
            refresh_out_kv_len_per_capture_row_i32=(
                refresh_layout.kv_len_per_row_i32 if refresh_layout is not None else None
            ),
            producer_rows_cpu=getattr(side_outputs, "producer_rows_cpu", None),
            row_capture_last_n_cpu=getattr(side_outputs, "row_capture_last_n_cpu", None),
            row_is_prefill_producer_cpu=getattr(side_outputs, "row_is_prefill_producer_cpu", None),
            seqused_k_cpu=getattr(side_outputs, "seqused_k_cpu", None),
            active_capture_row_by_batch_row_cpu=getattr(side_outputs, "active_capture_row_by_batch_row_cpu", None),
            prefill_out_kv_len_per_capture_row_cpu=getattr(
                side_outputs,
                "prefill_out_kv_len_per_capture_row_cpu",
                None,
            ),
            refresh_out_kv_len_per_capture_row_cpu=getattr(
                side_outputs,
                "refresh_out_kv_len_per_capture_row_cpu",
                None,
            ),
            skip_postprocess_rows_cpu=skip_postprocess_rows,
            row_capture_accum_prev_rows_cpu=step_capture_accum_prev_rows_cpu,
            row_capture_accum_prev_capacity_cpu=step_capture_accum_prev_capacity_cpu,
            drain_lastn1_rows=_ring_lastn1_drain_active(),
            meta_cache_owner=controller,
            alpha=alpha_log_f,
            debug_epoch=int(getattr(step_authority, "epoch", -1)),
            debug_layer_index=int(getattr(state, "layer_index", -1)),
        )

    postprocess_profile_metadata = {
        "epoch": int(getattr(step_authority, "epoch", -1)),
        "layer": int(getattr(state, "layer_index", -1)),
        "direct_capture_phase": str(capture_postprocess_state.get("direct_capture_phase", "")),
        "producer_rows_cpu": tuple(
            int(v) for v in tuple(getattr(side_outputs, "producer_rows_cpu", tuple()) or tuple())
        ),
        "row_capture_last_n_cpu": tuple(
            int(v) for v in tuple(getattr(side_outputs, "row_capture_last_n_cpu", tuple()) or tuple())
        ),
        "skip_postprocess_rows_cpu": tuple(int(v) for v in skip_postprocess_rows),
    }

    def _profiled_capture_postprocess(*, label: str, async_mode: bool) -> bool:
        return _run_capture_postprocess()

    def _defer_capture_postprocess_available() -> bool:
        if not defer_capture_postprocess_requested:
            return False
        if not capture_postprocess_required:
            return False
        if (
            refresh_layout is not None
            and _capture_ownership_mode_value != CHUNK_COHORT
        ):
            return False
        if not bool(getattr(getattr(controller, "config", None), "one_shot_bootstrap_only", False)):
            return False
        if not torch.cuda.is_available():
            return False
        if torch.cuda.is_current_stream_capturing():
            return False
        return True

    def _async_capture_postprocess_available() -> bool:
        if not async_capture_postprocess_requested:
            return False
        if not capture_postprocess_required:
            return False
        if not torch.cuda.is_available():
            return False
        if torch.cuda.is_current_stream_capturing():
            return False
        async_refresh_enabled = getattr(controller, "_async_refresh_enabled", None)
        if not callable(async_refresh_enabled) or not bool(async_refresh_enabled()):
            return False
        ensure_refresh_stream = getattr(controller, "_ensure_refresh_stream", None)
        if not callable(ensure_refresh_stream):
            return False
        ensure_refresh_stream(torch.device(query.device))
        return getattr(controller, "refresh_stream", None) is not None

    def _early_capture_postprocess_stream_available() -> bool:
        async_refresh_enabled = getattr(controller, "_async_refresh_enabled", None)
        if not callable(async_refresh_enabled) or not bool(async_refresh_enabled()):
            return False
        ensure_refresh_stream = getattr(controller, "_ensure_refresh_stream", None)
        if not callable(ensure_refresh_stream):
            return False
        ensure_refresh_stream(torch.device(query.device))
        return getattr(controller, "refresh_stream", None) is not None

    _POSTPROCESS_OWNER_NONE = "none"
    _POSTPROCESS_OWNER_DEFERRED_EARLY = "deferred_early"
    _POSTPROCESS_OWNER_DEFERRED_LATE = "deferred_late"
    _POSTPROCESS_OWNER_ASYNC = "async"
    _POSTPROCESS_OWNER_INLINE = "inline"
    _POSTPROCESS_OWNER_CHUNK_COHORT_EARLY = "chunk_cohort_early"

    def _resolve_capture_postprocess_execution_owner() -> str:
        """Freeze one postprocess owner; stamped cohorts never fall back."""
        binding = getattr(
            controller,
            "_capture_postprocess_step_owner_binding",
            None,
        )
        binding_identity_valid = bool(
            capture_handle_id >= 0
            and capture_handle_generation >= 0
            and capture_epoch >= 0
        )
        binding_is_valid = bool(
            isinstance(binding, tuple) and len(binding) == 7
        )
        if binding is not None and not binding_is_valid:
            raise RuntimeError(
                "E_SFI_CAPTURE_STEP_OWNER_DRIFT: invalid step owner binding"
            )
        binding_matches_logical_step = bool(
            binding_identity_valid
            and binding_is_valid
            and int(binding[0]) == capture_handle_id
            and int(binding[1]) == capture_handle_generation
            and int(binding[2]) == capture_epoch
            and int(binding[3]) == capture_step_identity_token
        )
        if binding_matches_logical_step:
            if (
                str(binding[4]) != _capture_ownership_mode_value
                or str(binding[5]) != _capture_ownership_plan_signature
            ):
                raise RuntimeError(
                    "E_SFI_CAPTURE_STEP_OWNER_DRIFT: ownership plan changed "
                    "within one logical capture step"
                )
            bound_owner = str(binding[6])
            bound_has_work = bound_owner != _POSTPROCESS_OWNER_NONE
            if bound_has_work != bool(capture_postprocess_required):
                raise RuntimeError(
                    "E_SFI_CAPTURE_STEP_OWNER_DRIFT: postprocess semantic work "
                    "changed across layers in one immutable capture step"
                )
            return bound_owner

        def _bind_owner(resolved_owner: str) -> str:
            if binding_identity_valid:
                setattr(
                    controller,
                    "_capture_postprocess_step_owner_binding",
                    (
                        capture_handle_id,
                        capture_handle_generation,
                        capture_epoch,
                        capture_step_identity_token,
                        _capture_ownership_mode_value,
                        _capture_ownership_plan_signature,
                        resolved_owner,
                    ),
                )
            return resolved_owner

        if not capture_postprocess_required:
            return _bind_owner(_POSTPROCESS_OWNER_NONE)

        if _capture_ownership_mode_value == CHUNK_COHORT:
            if not binding_identity_valid:
                raise RuntimeError(
                    "E_SFI_CAPTURE_COHORT_OWNER_UNAVAILABLE: stamped chunk "
                    "cohort requires a valid logical step identity"
                )
            if (
                not defer_capture_postprocess_requested
                or not early_capture_postprocess_requested
                or async_capture_postprocess_requested
            ):
                raise RuntimeError(
                    "E_SFI_CAPTURE_COHORT_OWNER: stamped chunk cohort requires "
                    "the deferred early owner for every live postprocess job"
                )
            if not _defer_capture_postprocess_available():
                raise RuntimeError(
                    "E_SFI_CAPTURE_COHORT_OWNER_UNAVAILABLE: stamped chunk "
                    "cohort cannot execute its deferred early owner"
                )
            if getattr(controller, "refresh_stream", None) is None:
                raise RuntimeError(
                    "E_SFI_CAPTURE_COHORT_OWNER_UNAVAILABLE: stamped chunk "
                    "cohort requires a pre-established refresh_stream"
                )
            return _bind_owner(_POSTPROCESS_OWNER_CHUNK_COHORT_EARLY)

        # Every reusable scratch owner publishes reduce completion through the
        # RingWarFence. Validate only live scratch work: no-work/direct steps
        # must not be rejected by a structural plan they do not execute.
        if _capture_ring_fence_active and (
            (defer_capture_postprocess_requested and not early_capture_postprocess_requested)
            or async_capture_postprocess_requested
        ):
            raise RuntimeError(
                "capture scratch reuse requires the early-reduce fence: "
                "pure-defer (VLLM_SPARSE_DEFER_CAPTURE_POSTPROCESS=1 without "
                "VLLM_SPARSE_EARLY_CAPTURE_POSTPROCESS=1) and VLLM_SPARSE_ASYNC_CAPTURE_"
                "POSTPROCESS modes read reusable scratch without a WAR fence event; "
                "enable EARLY=1 or use a ring_early ownership plan"
            )
        if _defer_capture_postprocess_available():
            if early_capture_postprocess_requested:
                if not _early_capture_postprocess_stream_available():
                    raise RuntimeError(
                        "E_SFI_CAPTURE_POSTPROCESS_OWNER_UNAVAILABLE: deferred "
                        "early owner requires async refresh_stream"
                    )
                return _bind_owner(_POSTPROCESS_OWNER_DEFERRED_EARLY)
            return _bind_owner(_POSTPROCESS_OWNER_DEFERRED_LATE)
        if _async_capture_postprocess_available():
            return _bind_owner(_POSTPROCESS_OWNER_ASYNC)
        return _bind_owner(_POSTPROCESS_OWNER_INLINE)

    capture_postprocess_execution_owner = (
        _resolve_capture_postprocess_execution_owner()
    )

    if capture_postprocess_execution_owner == _POSTPROCESS_OWNER_NONE:
        postprocess_ran = False
    elif (
        capture_postprocess_execution_owner
        == _POSTPROCESS_OWNER_DEFERRED_EARLY
        or capture_postprocess_execution_owner
        == _POSTPROCESS_OWNER_DEFERRED_LATE
        or capture_postprocess_execution_owner
        == _POSTPROCESS_OWNER_CHUNK_COHORT_EARLY
    ):
        ready_event = torch.cuda.Event(enable_timing=False)
        ready_event.record(torch.cuda.current_stream(device=query.device))
        capture_postprocess_job = CapturePostprocessJob(
            direct_capture_phase=str(capture_postprocess_state.get("direct_capture_phase", "")),
            scratch_capture_scores=side_outputs.scratch_capture_scores,
            producer_rows_i32=side_outputs.producer_rows_i32,
            row_capture_last_n_i32=side_outputs.row_capture_last_n_i32,
            row_is_prefill_producer=side_outputs.row_is_prefill_producer,
            seqused_k=real_seqused_k,
            active_capture_row_by_batch_row_i32=side_outputs.active_capture_row_by_batch_row_i32,
            prefill_out_capture_scores=side_outputs.prefill_out_capture_scores,
            prefill_out_log_f_denoms=side_outputs.prefill_out_log_f_denoms,
            refresh_out_capture_scores=side_outputs.refresh_out_capture_scores,
            refresh_out_log_f_denoms=side_outputs.refresh_out_log_f_denoms,
            prefill_out_kv_len_per_capture_row_i32=(
                prefill_layout.kv_len_per_row_i32 if prefill_layout is not None else None
            ),
            refresh_out_kv_len_per_capture_row_i32=(
                refresh_layout.kv_len_per_row_i32 if refresh_layout is not None else None
            ),
            producer_rows_cpu=tuple(
                int(v) for v in tuple(getattr(side_outputs, "producer_rows_cpu", tuple()) or tuple())
            ),
            row_capture_last_n_cpu=tuple(
                int(v) for v in tuple(getattr(side_outputs, "row_capture_last_n_cpu", tuple()) or tuple())
            ),
            row_is_prefill_producer_cpu=tuple(
                bool(v)
                for v in tuple(getattr(side_outputs, "row_is_prefill_producer_cpu", tuple()) or tuple())
            ),
            seqused_k_cpu=tuple(
                int(v) for v in tuple(getattr(side_outputs, "seqused_k_cpu", tuple()) or tuple())
            ),
            active_capture_row_by_batch_row_cpu=tuple(
                int(v)
                for v in tuple(
                    getattr(side_outputs, "active_capture_row_by_batch_row_cpu", tuple())
                    or tuple()
                )
            ),
            prefill_out_kv_len_per_capture_row_cpu=tuple(
                int(v)
                for v in tuple(
                    getattr(side_outputs, "prefill_out_kv_len_per_capture_row_cpu", tuple())
                    or tuple()
                )
            ),
            refresh_out_kv_len_per_capture_row_cpu=tuple(
                int(v)
                for v in tuple(
                    getattr(side_outputs, "refresh_out_kv_len_per_capture_row_cpu", tuple())
                    or tuple()
                )
            ),
            skip_postprocess_rows_cpu=tuple(int(v) for v in skip_postprocess_rows),
            row_capture_accum_prev_rows_cpu=step_capture_accum_prev_rows_cpu,
            row_capture_accum_prev_capacity_cpu=step_capture_accum_prev_capacity_cpu,
            drain_lastn1_rows=_ring_lastn1_drain_active(),
            alpha=float(alpha_log_f),
            debug_epoch=int(getattr(step_authority, "epoch", -1)),
            debug_layer_index=int(getattr(state, "layer_index", -1)),
            ready_event=ready_event,
            job_key=(
                capture_handle_id,
                capture_handle_generation,
                int(global_layer_index),
            ),
            # [DETERMINISTIC-TAPE-WAW 2026-07-03] flush retarget 与 deferred
            # launch 的 TOCTOU 互斥锁(见 sparse_types 字段注释)。
            lifecycle_lock=threading.Lock(),
        )
        postprocess_ran = False
        if (
            capture_postprocess_execution_owner
            != _POSTPROCESS_OWNER_DEFERRED_LATE
        ):
            refresh_stream = getattr(controller, "refresh_stream", None)
            if refresh_stream is None:
                raise RuntimeError(
                    "E_SFI_CAPTURE_POSTPROCESS_OWNER_UNAVAILABLE: frozen early "
                    "owner requires refresh_stream"
                )

            if (
                capture_postprocess_execution_owner
                == _POSTPROCESS_OWNER_CHUNK_COHORT_EARLY
            ):
                coordinator = getattr(
                    controller, "_capture_cohort_coordinator", None
                )
                if coordinator is None:
                    coordinator = CaptureCohortCoordinator()
                    setattr(controller, "_capture_cohort_coordinator", coordinator)
                elif not isinstance(coordinator, CaptureCohortCoordinator):
                    raise RuntimeError(
                        "E_SFI_CAPTURE_COHORT_OWNER: controller coordinator "
                        "has an invalid type"
                    )
                total_layers = len(
                    tuple(getattr(controller, "layer_cache_keys", tuple()) or tuple())
                )
                ready_cohort = coordinator.submit(
                    job=capture_postprocess_job,
                    handle_id=capture_handle_id,
                    handle_generation=capture_handle_generation,
                    epoch=capture_epoch,
                    plan_signature=_capture_ownership_plan_signature,
                    global_layer_index=int(global_layer_index),
                    total_layers=int(total_layers),
                    chunk_id=int(capture_chunk_id),
                    slot_in_chunk=int(slot_in_chunk),
                    scratch_slot=int(_effective_capture_ring_slot),
                    cohort_size=int(_effective_capture_cohort_size),
                    selected_depth=int(_effective_capture_scratch_depth),
                )
                postprocess_ran = False
                if ready_cohort is not None:
                    with torch.cuda.stream(refresh_stream):
                        (
                            ran_count,
                            cohort_terminal_event,
                        ) = run_capture_postprocess_job_sequence_for_cohort(
                            ready_cohort.jobs,
                            meta_cache_owner=controller,
                        )
                        fence = getattr(controller, "_ring_war_fence", None)
                        publish_capture_cohort_completion(
                            ready_cohort,
                            ran_count=int(ran_count),
                            terminal_event=cohort_terminal_event,
                            fence=fence,
                        )
                    postprocess_ran = True
            else:
                with torch.cuda.stream(refresh_stream):
                    postprocess_ran = run_capture_postprocess_job_if_needed(
                        capture_postprocess_job,
                        meta_cache_owner=controller,
                    )
                    if bool(getattr(capture_postprocess_job, "completed", False)):
                        completion_event = getattr(
                            capture_postprocess_job, "completion_event", None
                        )
                        if completion_event is None:
                            raise RuntimeError(
                                "E_SFI_CAPTURE_POSTPROCESS_EVENT: completed job "
                                "has no completion event"
                            )
                        if _capture_ring_fence_active:
                            fence = getattr(controller, "_ring_war_fence", None)
                            if fence is None:
                                raise RuntimeError(
                                    "E_SFI_CAPTURE_POSTPROCESS_FENCE: reusable "
                                    "scratch has no WAR fence owner"
                                )
                            fence.on_reduce(
                                int(_effective_capture_ring_slot),
                                completion_event,
                                postprocess_ran,
                            )
    elif capture_postprocess_execution_owner == _POSTPROCESS_OWNER_ASYNC:
        refresh_stream = getattr(controller, "refresh_stream", None)
        if refresh_stream is None:
            raise RuntimeError(
                "E_SFI_CAPTURE_ASYNC_OWNER_UNAVAILABLE: frozen async owner "
                "requires refresh_stream"
            )
        else:
            main_stream = torch.cuda.current_stream(device=query.device)
            ready_event = torch.cuda.Event(enable_timing=False)
            ready_event.record(main_stream)

            def _record_async_postprocess_tensor(tensor: Optional[torch.Tensor]) -> None:
                if not isinstance(tensor, torch.Tensor):
                    return
                try:
                    tensor.record_stream(refresh_stream)
                except Exception:
                    _log.warning("async capture postprocess record_stream failed", exc_info=True)
                    raise

            _record_async_postprocess_tensor(side_outputs.scratch_capture_scores)
            _record_async_postprocess_tensor(side_outputs.producer_rows_i32)
            _record_async_postprocess_tensor(side_outputs.row_capture_last_n_i32)
            _record_async_postprocess_tensor(side_outputs.row_is_prefill_producer)
            _record_async_postprocess_tensor(side_outputs.active_capture_row_by_batch_row_i32)
            _record_async_postprocess_tensor(side_outputs.prefill_out_capture_scores)
            _record_async_postprocess_tensor(side_outputs.prefill_out_log_f_denoms)
            _record_async_postprocess_tensor(side_outputs.refresh_out_capture_scores)
            _record_async_postprocess_tensor(side_outputs.refresh_out_log_f_denoms)
            _record_async_postprocess_tensor(real_seqused_k)

            with torch.cuda.stream(refresh_stream):
                torch.cuda.current_stream(device=query.device).wait_event(ready_event)
                postprocess_ran = _profiled_capture_postprocess(
                    label="capture_postprocess_async",
                    async_mode=True,
                )
    elif capture_postprocess_execution_owner == _POSTPROCESS_OWNER_INLINE:
        postprocess_ran = _profiled_capture_postprocess(
            label="capture_postprocess",
            async_mode=False,
        )
    else:
        raise AssertionError(
            "unhandled capture postprocess execution owner: "
            f"{capture_postprocess_execution_owner}"
        )

    def _lastn1_scratch_view_for_rows(
        *,
        phase: str,
        row_list: Sequence[int],
        kv_len_total: int,
    ) -> Optional[torch.Tensor]:
        if str(capture_postprocess_state.get("direct_capture_phase", "")) == str(phase):
            return None
        rows_tuple = tuple(int(row) for row in row_list)
        if not rows_tuple:
            return None
        if not any(
            0 <= int(row) < len(logits_last_n_by_row)
            and int(logits_last_n_by_row[int(row)]) == 1
            for row in rows_tuple
        ):
            return None
        row_to_scratch = {
            int(row): int(idx) for idx, row in enumerate(tuple(side_outputs.producer_rows_cpu))
        }
        scratch_indices = tuple(int(row_to_scratch.get(int(row), -1)) for row in rows_tuple)
        if any(idx < 0 for idx in scratch_indices):
            raise RuntimeError(
                f"{phase} last_n==1 scratch tape missing producer row mapping"
            )
        start = int(scratch_indices[0])
        end = int(scratch_indices[-1])
        if (end - start + 1) != len(scratch_indices):
            raise RuntimeError(
                f"{phase} last_n==1 scratch tape requires phase-local contiguous rows"
            )
        return side_outputs.scratch_capture_scores[
            start : start + len(scratch_indices),
            :,
            :1,
            : int(kv_len_total),
        ]

    if getattr(step_ctx, "step_authority", None) is None:
        try:
            setattr(step_ctx, "step_authority", step_authority)
        except Exception:
            pass

    has_runtime_enqueue = (
        block_table is not None
        and hasattr(controller, "_enqueue_prefill_capture")
        and hasattr(controller, "_flush_prefill_batches")
        and (refresh_layout is None or hasattr(controller, "_enqueue_refresh_capture"))
    )
    if not has_runtime_enqueue:
        return result
    softmax_scale = float(getattr(self, "scale", 1.0))
    softcap = float(getattr(self, "logits_soft_cap", 0.0) or 0.0)
    window_size = getattr(self, "sliding_window", None)
    alibi_slopes = getattr(self, "alibi_slopes", None)
    k_descale = None

    capture_plan_active_by_slot: dict[int, int] = {}
    for batch_row in owner_plan.prefill_capture_rows:
        if batch_row >= len(slot_by_row):
            continue
        slot = int(slot_by_row[batch_row])
        last_n = (
            logits_last_n_by_row[batch_row]
            if batch_row < len(logits_last_n_by_row)
            else 0
        )
        capture_plan_active_by_slot[slot] = int(last_n)

    if capture_plan_active_by_slot:
        payload = _prepare_prefill_capture_payload(
            controller=controller,
            cache_key=cache_key,
            state=state,
            step_context=step_ctx,
            q=query,
            cu_seqlens_q=cu_seqlens_q,
            seqused_k=real_seqused_k,
            chunk_query_lengths=chunk_query_lengths,
            num_heads=query.shape[1],
            device=query.device,
            capture_plan=capture_plan_active_by_slot,
            layout=prefill_layout,
        )
        if payload is None:
            raise RuntimeError("native FA3 capture mixed route failed to build prefill payload")
        (
            capture_scores,
            log_f_denoms,
            kv_lengths,
            seq_lens_batch,
            seq_lens_batch_i32,
            _chunk_lengths,
            slot_list,
            slot_tensor,
            slot_tensor_i32,
            slot_tensor_cpu,
            row_list_cpu,
            row_tensor,
            row_tensor_i32,
            kv_len_per_row_i32,
            seq_lens_cpu,
            seq_lens_tensor_cpu,
            layer_index_in_chunk,
        ) = payload
        # [RING-LASTN1-DRAIN 2026-07-03] under the ring the postprocess run drains
        # last_n==1 rows into the phase out tensors (fenced), so the payload must
        # not carry a raw ring-scratch view -- the flush falls back to the tape
        # (lastn1_capture_scores=None -> p.capture_scores in the !use_denoms group).
        lastn1_capture_scores = (
            None
            if _ring_lastn1_drain_active()
            else _lastn1_scratch_view_for_rows(
                phase="prefill",
                row_list=row_list_cpu,
                kv_len_total=int(capture_scores.shape[-1]),
            )
        )
        controller._enqueue_prefill_capture(
            SelectorBatchPayload(
                cache_key=cache_key,
                state=state,
                capture_scores=capture_scores,
                log_f_denoms=log_f_denoms,
                lastn1_capture_scores=lastn1_capture_scores,
                capture_postprocess_job=capture_postprocess_job,
                kv_lengths=kv_lengths,
                slot_list=slot_list,
                row_list=row_list_cpu,
                key_cache=key_cache,
                value_cache=value_cache,
                block_table=block_table,
                bootstrap_slots=set(slot_list),
                layer_index=int(layer_index_in_chunk),
                seq_lens_batch=seq_lens_batch,
                seq_lens_batch_i32=seq_lens_batch_i32,
                slot_tensor=slot_tensor,
                slot_tensor_i32=slot_tensor_i32,
                slot_tensor_cpu=slot_tensor_cpu,
                row_tensor=row_tensor,
                row_tensor_i32=row_tensor_i32,
                kv_len_per_row_i32=kv_len_per_row_i32,
                seq_lens_cpu=seq_lens_cpu if seq_lens_cpu is not None else tuple(),
                seq_lens_tensor_cpu=seq_lens_tensor_cpu,
                target_selected_scope_key=getattr(step_authority, "target_selected_scope_key", None),
                selected_scope_wait_handle=getattr(step_authority, "selected_scope_wait_handle", None),
                fast_signature=_make_selector_fast_signature(
                    capture_scores=capture_scores,
                    log_f_denoms=log_f_denoms,
                    kv_lengths=kv_lengths,
                    block_table=block_table,
                ),
                capture_handle_id=capture_handle_id,
                capture_handle_generation=capture_handle_generation,
                capture_epoch=capture_epoch,
            ),
        )
        # native FA3 prefill enqueue 只提交 transport payload；
        # request-facing bootstrap pending 统一延后到 flush 成功后的 finalize boundary。

    if refresh_layout is not None and refresh_slot_set:
        step_envelope = getattr(step_ctx, "step_envelope_v2", None)
        refresh_reason = str(getattr(step_envelope, "refresh_reason", ""))
        refresh_intent_req_ids = tuple(
            str(req_id)
            for req_id in (getattr(step_envelope, "refresh_reqs", ()) or ())
        )
        payload = _prepare_refresh_capture_payload(
            controller=controller,
            cache_key=cache_key,
            state=state,
            step_context=step_ctx,
            q=query,
            cu_seqlens_q=cu_seqlens_q,
            seqused_k=real_seqused_k,
            slots_filter=active_refresh_slot_list,
            slots_filter_sorted=True,
            layout=refresh_layout,
        )
        if payload is None:
            raise RuntimeError("native FA3 capture mixed route failed to build refresh payload")
        (
            capture_scores,
            log_f_denoms,
            kv_lengths_tensor,
            seq_lens_batch,
            seq_lens_batch_i32,
            slot_list,
            slot_tensor,
            slot_tensor_i32,
            slot_tensor_cpu,
            row_list,
            row_tensor,
            row_tensor_i32,
            kv_len_per_row_i32,
            seq_lens_cpu,
            seq_lens_tensor_cpu,
            layer_index_in_chunk,
        ) = payload
        # [RING-LASTN1-DRAIN-MIXED 2026-07-03 晚] refresh 相的 scratch 视图替换
        # 已退休:混合步(direct_capture_phase=="")的 scratch 是 per-step 单一缓
        # 冲、每层复写,chunk 批量 deferred 消费必读脏(4B bs8 错峰实证);现由
        # _ring_lastn1_drain_active 在混合步激活 lastn1 drain,把 refresh 行的
        # B1 logits 逐层拷回 layout base,payload 恒携带 base 视图(与纯 refresh
        # 步 direct-capture 直写 base 后的消费形态一致,selector 的 chunk-deep
        # 5 维 tape 证明天然成立)。
        controller._enqueue_refresh_capture(
            SelectorBatchPayload(
                cache_key=int(cache_key),
                state=state,
                capture_scores=capture_scores,
                log_f_denoms=log_f_denoms,
                lastn1_capture_scores=None,
                capture_postprocess_job=capture_postprocess_job,
                kv_lengths=kv_lengths_tensor,
                kv_len_per_row_i32=kv_len_per_row_i32,
                seq_lens_batch=seq_lens_batch,
                seq_lens_batch_i32=seq_lens_batch_i32,
                slot_list=slot_list,
                row_list=row_list,
                slot_tensor=slot_tensor,
                slot_tensor_i32=slot_tensor_i32,
                slot_tensor_cpu=slot_tensor_cpu,
                row_tensor=row_tensor,
                row_tensor_i32=row_tensor_i32,
                key_cache=key_cache,
                value_cache=value_cache,
                block_table=block_table,
                bootstrap_slots=set(),
                q=query,
                q_is_sub=False,
                cu_seqlens_q=cu_seqlens_q,
                softmax_scale=softmax_scale,
                softcap=softcap,
                window_size=window_size,
                alibi_slopes=alibi_slopes,
                k_descale=k_descale,
                layer_index=int(layer_index_in_chunk),
                seq_lens_cpu=seq_lens_cpu if seq_lens_cpu is not None else tuple(),
                seq_lens_tensor_cpu=seq_lens_tensor_cpu,
                target_selected_scope_key=getattr(step_authority, "target_selected_scope_key", None),
                selected_scope_wait_handle=getattr(step_authority, "selected_scope_wait_handle", None),
                fast_signature=_make_selector_fast_signature(
                    capture_scores=capture_scores,
                    log_f_denoms=log_f_denoms,
                    kv_lengths=kv_lengths_tensor,
                    block_table=block_table,
                ),
                capture_handle_id=capture_handle_id,
                capture_handle_generation=capture_handle_generation,
                capture_epoch=capture_epoch,
                refresh_reason=refresh_reason,
                refresh_intent_req_ids=refresh_intent_req_ids,
            )
        )

    layer_index_global = int(getattr(state, "layer_index", -1))
    if layer_index_global < 0:
        layer_index_global = global_layer_index
    total_layers = len(getattr(controller, "layer_cache_keys", tuple()))
    is_last_layer = bool(total_layers > 0 and layer_index_global == total_layers - 1)
    chunk_id, buf_id, slot_in_chunk = controller._map_global_layer_to_capture_slot(layer_index_global)
    if (int(slot_in_chunk) == int(_CAPTURE_CHUNK) - 1) or is_last_layer:
        chunk_size = (int(slot_in_chunk) + 1) if is_last_layer else int(_CAPTURE_CHUNK)
        if _capture_ownership_mode_value == CHUNK_COHORT:
            coordinator = getattr(controller, "_capture_cohort_coordinator", None)
            if coordinator is not None:
                if not isinstance(coordinator, CaptureCohortCoordinator):
                    raise RuntimeError(
                        "E_SFI_CAPTURE_COHORT_OWNER: controller coordinator "
                        "has an invalid type at the flush boundary"
                    )
                coordinator.assert_idle()
        controller._flush_prefill_batches(
            buf_id=int(buf_id),
            chunk_id=int(chunk_id),
            chunk_size=int(chunk_size),
            is_last_layer=bool(is_last_layer),
        )
    return result


def _patched_v1_flash_attn_forward_mixed_impl(
    *,
    self,
    layer,
    query,
    key,
    value,
    kv_cache,
    attn_metadata,
    output=None,
    output_scale=None,
    output_block_scale=None,
):
    from patches.fa3_native.route_adapter import get_bound_launch_route_hints

    if bool(getattr(attn_metadata, "use_cascade", False)):
        raise RuntimeError("native FA3 adapter does not support use_cascade=True yet")
    if getattr(attn_metadata, "local_attn_metadata", None) is not None:
        raise RuntimeError(
            "native FA3 adapter does not support local_attn_metadata yet"
        )
    graph_replay_enabled = _mixed_page_resolver_graph_replay_expected(attn_metadata)
    if graph_replay_enabled and not _mixed_page_resolver_replay_carriers_updated(attn_metadata):
        raise RuntimeError(
            "mixed-page resolver CUDA graph replay requires in-place carrier update before cudagraph.replay()"
        )

    route_hints = get_bound_launch_route_hints(attn_metadata)
    if bool(getattr(attn_metadata, "sparse_vllm_profile_step", False)):
        if route_hints.has_capture:
            raise RuntimeError(
                "vLLM profile mixed route must not request capture rows"
            )
        if not route_hints.has_selected_consume:
            raise RuntimeError(
                "vLLM profile mixed route requires selected consumer rows"
            )
        return _run_profile_resolved_row_ptr_mixed_forward(
            self=self,
            layer=layer,
            query=query,
            key=key,
            value=value,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            output=output,
            output_scale=output_scale,
            output_block_scale=output_block_scale,
        )
    if route_hints.has_capture:
        return _run_capture_only_mixed_forward(
            self=self,
            layer=layer,
            query=query,
            key=key,
            value=value,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            output=output,
            output_scale=output_scale,
            output_block_scale=output_block_scale,
        )

    controller = _get_global_controller()
    if controller is None or not bool(getattr(getattr(controller, "config", None), "enabled", False)):
        raise RuntimeError(
            "native FA3 adapter requires a live sparse controller for mixed route"
        )
    step_authority = getattr(controller, "step_authority", None)
    if step_authority is None:
        raise RuntimeError(
            "native FA3 adapter requires step_authority for mixed route"
        )
    return _run_selected_no_capture_mixed_forward(
        self=self,
        layer=layer,
        query=query,
        key=key,
        value=value,
        kv_cache=kv_cache,
        attn_metadata=attn_metadata,
        output=output,
        output_scale=output_scale,
        output_block_scale=output_block_scale,
    )


class _FlashAttnGatewayFallback(Exception):
    """Soft fallback marker for the installed FA varlen gateway."""


_FA4_DENSE_CUTE_VARLEN_FUNC = None


def _normalize_fa4_window_size(value):
    if value is None:
        return (None, None)
    if tuple(value) == (-1, -1):
        return (None, None)
    left, right = tuple(value)
    return (
        None if left is None or int(left) < 0 else int(left),
        None if right is None or int(right) < 0 else int(right),
    )


def _is_fa4_fp8_tensor(value) -> bool:
    return str(getattr(value, "dtype", "")).startswith("torch.float8")


def _call_fa4_dense_fallback(bridge: object, **kwargs):
    global _FA4_DENSE_CUTE_VARLEN_FUNC
    if _FA4_DENSE_CUTE_VARLEN_FUNC is None:
        import sys

        root = getattr(bridge, "root", None)
        if root is not None and str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from flash_attn.cute.interface import flash_attn_varlen_func as _fa4_cute_varlen

        _FA4_DENSE_CUTE_VARLEN_FUNC = _fa4_cute_varlen

    if float(kwargs.get("dropout_p", 0.0) or 0.0) != 0.0:
        raise ValueError("FA4 dense fallback only supports dropout_p=0")
    if kwargs.get("alibi_slopes") is not None:
        raise ValueError("FA4 dense fallback does not support alibi_slopes")
    if bool(kwargs.get("return_attn_probs", False)):
        raise ValueError("FA4 dense fallback does not support return_attn_probs")
    if kwargs.get("q_v") is not None:
        raise ValueError("FA4 dense fallback does not support q_v")
    if int(kwargs.get("cp_world_size", 1) or 1) != 1:
        raise ValueError("FA4 dense fallback requires cp_world_size=1")
    has_descale = any(
        kwargs.get(name) is not None for name in ("q_descale", "k_descale", "v_descale")
    )
    if has_descale and any(
        _is_fa4_fp8_tensor(kwargs.get(name)) for name in ("q", "k", "v")
    ):
        raise ValueError("FA4 dense fallback does not support FP8 descale tensors")

    out = kwargs.get("out")
    return_lse = bool(kwargs.get("return_softmax_lse", False))
    result = _FA4_DENSE_CUTE_VARLEN_FUNC(
        q=kwargs.get("q"),
        k=kwargs.get("k"),
        v=kwargs.get("v"),
        cu_seqlens_q=kwargs.get("cu_seqlens_q"),
        cu_seqlens_k=kwargs.get("cu_seqlens_k"),
        max_seqlen_q=kwargs.get("max_seqlen_q"),
        max_seqlen_k=kwargs.get("max_seqlen_k"),
        seqused_k=kwargs.get("seqused_k"),
        page_table=kwargs.get("block_table"),
        softmax_scale=kwargs.get("softmax_scale"),
        causal=bool(kwargs.get("causal", False)),
        window_size=_normalize_fa4_window_size(kwargs.get("window_size")),
        learnable_sink=kwargs.get("s_aux"),
        softcap=float(kwargs.get("softcap", 0.0) or 0.0),
        num_splits=int(kwargs.get("num_splits", 1) or 0),
        deterministic=bool(kwargs.get("deterministic", False)),
        return_lse=return_lse,
    )
    if isinstance(result, tuple):
        result_out = result[0]
        lse = result[1] if len(result) > 1 else None
    else:
        result_out = result
        lse = None
    if out is not None:
        out.copy_(result_out)
        return (out, lse) if return_lse else out
    return result


def build_fa4_dense_fallback_varlen_func(bridge: object):
    append_route_trace = None
    if _fa3_route_trace_enabled():
        try:
            from patches.fa3_native.install import append_fa3_route_trace

            append_route_trace = append_fa3_route_trace
        except Exception:
            append_route_trace = None

    def _fa4_dense_fallback_varlen_func(*args, **kwargs):
        if args:
            raise ValueError("FA4 dense fallback expects keyword arguments")
        if append_route_trace is not None:
            append_route_trace(
                {
                    "event": "flash_attn_varlen_func_call",
                    "route": "flash_attn_varlen_func",
                    "symbol_module": "flash_attn.cute.interface",
                    "fa_version": 4,
                    "pid": int(os.getpid()),
                }
            )
        return _call_fa4_dense_fallback(bridge, **kwargs)

    setattr(_fa4_dense_fallback_varlen_func, "_sfi_fa4_dense_gateway", True)
    return _fa4_dense_fallback_varlen_func


def install_fa4_dense_fallback_gateway(bridge: object | None = None) -> dict[str, object]:
    if bridge is None:
        from patches.fa3_native.install import load_vendored_flash_attn_bridge

        bridge = load_vendored_flash_attn_bridge()
    try:
        from vllm.v1.attention.backends import fa_utils
        from vllm.v1.attention.backends import flash_attn as v1_flash_attn
    except Exception as exc:
        return {"applied": False, "reason": f"import_failed:{exc}"}

    current = getattr(v1_flash_attn, "flash_attn_varlen_func", None)
    if bool(getattr(current, "_sfi_fa4_dense_gateway", False)):
        return {"applied": False, "reason": "already_installed"}

    gateway = build_fa4_dense_fallback_varlen_func(bridge)
    if hasattr(v1_flash_attn, "flash_attn_varlen_func"):
        v1_flash_attn.flash_attn_varlen_func = gateway  # type: ignore[assignment]
    if hasattr(fa_utils, "flash_attn_varlen_func"):
        fa_utils.flash_attn_varlen_func = gateway  # type: ignore[assignment]
    return {"applied": True, "reason": "installed"}


def _build_sfi_flash_attn_varlen_gateway(
    bridge: object,
    *,
    dense_fallback: object | None,
):
    fallback = dense_fallback
    if fallback is None:
        fallback = getattr(bridge, "flash_attn_varlen_func")

    compact_recent_route = "test_only_compact_recent_reference_disabled"
    fa4_dense_fallback = build_fa4_dense_fallback_varlen_func(bridge)

    def _call_dense_fallback(*args, **kwargs):
        if not args and int(kwargs.get("fa_version", 0) or 0) == 4:
            return fa4_dense_fallback(**kwargs)
        return fallback(*args, **kwargs)

    def _sfi_flash_attn_varlen_gateway(*args, **kwargs):
        if args:
            return _call_dense_fallback(*args, **kwargs)
        try:
            return _run_compact_recent_varlen_gateway(
                bridge=bridge,
                compact_recent_route=compact_recent_route,
                kwargs=kwargs,
            )
        except _FlashAttnGatewayFallback:
            return _call_dense_fallback(**kwargs)

    setattr(_sfi_flash_attn_varlen_gateway, "_sfi_flash_attn_gateway", True)
    return _sfi_flash_attn_varlen_gateway


def _raise_flash_attn_gateway_fallback() -> None:
    raise _FlashAttnGatewayFallback()


def _run_compact_recent_varlen_gateway(
    *,
    bridge: object,
    compact_recent_route: str,
    kwargs: Dict[str, object],
):
    controller = _get_global_controller()
    del bridge, compact_recent_route, kwargs, controller
    _raise_flash_attn_gateway_fallback()


def _patch_flash_attention_forward_and_helpers() -> None:
    global _FLASH_ATTN_FORWARD_PATCHED
    global _ORIGINAL_V1_FLASH_ATTN_VARLEN_FUNC, _ORIGINAL_V1_FLASH_ATTN_FORWARD
    global _ORIGINAL_V1_FLASH_ATTN_GET_SCHEDULER_METADATA
    global _ORIGINAL_V1_FLASH_ATTN_GET_FLASH_ATTN_VERSION
    global _ORIGINAL_FA_UTILS_FLASH_ATTN_VARLEN_FUNC, _ORIGINAL_FA_UTILS_GET_SCHEDULER_METADATA
    global _ORIGINAL_FA_UTILS_GET_FLASH_ATTN_VERSION
    global _ORIGINAL_V1_FLASH_METADATA_FULL_CUDAGRAPH_SUPPORTED
    if _FLASH_ATTN_FORWARD_PATCHED:
        return
    try:
        from vllm.v1.attention.backends import flash_attn as v1_flash_attn
        fa_utils = import_fa_utils_module()
    except Exception as exc:
        raise RuntimeError(
            "E_SFI_FA3_GATEWAY_INTERFACE: required flash_attn/fa_utils "
            "interface is unavailable"
        ) from exc

    from patches.fa3_native.install import (
        build_vendored_get_flash_attn_version,
        build_patched_flash_attention_forward,
        load_vendored_flash_attn_bridge,
        unwrap_dense_original_flash_attention_forward,
    )

    bridge = load_vendored_flash_attn_bridge()
    native_get_flash_attn_version = build_vendored_get_flash_attn_version(bridge)
    impl_cls = getattr(v1_flash_attn, "FlashAttentionImpl", None)
    if impl_cls is None:
        raise RuntimeError(
            "E_SFI_FA3_GATEWAY_INTERFACE: FlashAttentionImpl is unavailable"
        )

    old_fa_utils_flash = getattr(fa_utils, "flash_attn_varlen_func", None)
    old_fa_utils_scheduler = getattr(fa_utils, "get_scheduler_metadata", None)
    old_fa_utils_get_version = getattr(fa_utils, "get_flash_attn_version", None)
    old_v1_flash = getattr(v1_flash_attn, "flash_attn_varlen_func", None)
    old_v1_scheduler = getattr(v1_flash_attn, "get_scheduler_metadata", None)
    old_v1_get_version = getattr(v1_flash_attn, "get_flash_attn_version", None)
    old_forward = getattr(impl_cls, "forward", None)
    dense_original_forward = None
    if old_forward is not None:
        dense_original_forward = unwrap_dense_original_flash_attention_forward(
            old_forward
        )
    metadata_builder_cls = getattr(v1_flash_attn, "FlashAttentionMetadataBuilder", None)
    old_full_cudagraph_supported = getattr(
        metadata_builder_cls,
        "full_cudagraph_supported",
        None,
    )
    if old_forward is None:
        raise RuntimeError(
            "E_SFI_FA3_GATEWAY_INTERFACE: FlashAttentionImpl.forward is unavailable"
        )
    sfi_flash_gateway = _build_sfi_flash_attn_varlen_gateway(
        bridge,
        dense_fallback=old_v1_flash,
    )

    try:
        if old_v1_flash is not None:
            v1_flash_attn.flash_attn_varlen_func = sfi_flash_gateway  # type: ignore[assignment]
        if old_fa_utils_flash is not None:
            fa_utils.flash_attn_varlen_func = sfi_flash_gateway  # type: ignore[assignment]
        # [SM80-SCHED-NONE-FIX 2026-07-07] SM80 上 FA3 AOT scheduler 不存在:
        # 官方 _vllm_fa3_C 与 vendored so 的 prepare_varlen_num_blocks 均为
        # SM90-only TU(hopper/flash_prepare_scheduler.cu),SM80 调用即
        # cudaErrorNoKernelImageForDevice(209)。FULL cudagraph 下 vLLM builder
        # 在 capture 内首调它 → capture 被毒化 → replay illegal address
        # (4B bs8x12k 实锤,compute-sanitizer 栈钉死;kind4 双补丁当年按形态
        # 局部绕过,存在覆盖盲区)。终极语义化修复(单一路径,无兜底):
        # scheduler_metadata 在 SM80 恒 None → builder 走非 AOT else 分支
        # (原生完备,kind4 黄金已证数值合法)。三个消费名字空间统一替换,含
        # backend 模块文件头 from-import 的局部名(绑定早于 patch install,
        # 仅替换源模块属性够不着——本崩的直接盲区)。
        def _sm80_get_scheduler_metadata_none(*_args, **_kwargs):
            return None

        if old_fa_utils_scheduler is not None:
            fa_utils.get_scheduler_metadata = _sm80_get_scheduler_metadata_none  # type: ignore[assignment]
        if old_v1_scheduler is not None:
            v1_flash_attn.get_scheduler_metadata = _sm80_get_scheduler_metadata_none  # type: ignore[assignment]
        if old_fa_utils_get_version is not None:
            fa_utils.get_flash_attn_version = native_get_flash_attn_version  # type: ignore[assignment]
        if old_v1_get_version is not None:
            v1_flash_attn.get_flash_attn_version = native_get_flash_attn_version  # type: ignore[assignment]
        if metadata_builder_cls is not None and old_full_cudagraph_supported is not None:
            metadata_builder_cls.full_cudagraph_supported = (
                native_get_flash_attn_version() == 3
            )  # type: ignore[assignment]
        impl_cls.forward = build_patched_flash_attention_forward(
            dense_original_forward if dense_original_forward is not None else old_forward,
            mixed_forward_impl=_patched_v1_flash_attn_forward_mixed_impl,
        )  # type: ignore[assignment]
    except Exception:
        if old_v1_flash is not None:
            v1_flash_attn.flash_attn_varlen_func = old_v1_flash  # type: ignore[assignment]
        if old_fa_utils_flash is not None:
            fa_utils.flash_attn_varlen_func = old_fa_utils_flash  # type: ignore[assignment]
        if old_fa_utils_scheduler is not None:
            fa_utils.get_scheduler_metadata = old_fa_utils_scheduler  # type: ignore[assignment]
        if old_fa_utils_get_version is not None:
            fa_utils.get_flash_attn_version = old_fa_utils_get_version  # type: ignore[assignment]
        if old_v1_scheduler is not None:
            v1_flash_attn.get_scheduler_metadata = old_v1_scheduler  # type: ignore[assignment]
        if old_v1_get_version is not None:
            v1_flash_attn.get_flash_attn_version = old_v1_get_version  # type: ignore[assignment]
        if metadata_builder_cls is not None and old_full_cudagraph_supported is not None:
            metadata_builder_cls.full_cudagraph_supported = old_full_cudagraph_supported  # type: ignore[assignment]
        impl_cls.forward = old_forward  # type: ignore[assignment]
        raise

    _ORIGINAL_V1_FLASH_ATTN_VARLEN_FUNC = old_v1_flash
    _ORIGINAL_FA_UTILS_FLASH_ATTN_VARLEN_FUNC = old_fa_utils_flash
    _ORIGINAL_FA_UTILS_GET_SCHEDULER_METADATA = old_fa_utils_scheduler
    _ORIGINAL_FA_UTILS_GET_FLASH_ATTN_VERSION = old_fa_utils_get_version
    _ORIGINAL_V1_FLASH_ATTN_GET_SCHEDULER_METADATA = old_v1_scheduler
    _ORIGINAL_V1_FLASH_ATTN_GET_FLASH_ATTN_VERSION = old_v1_get_version
    _ORIGINAL_V1_FLASH_METADATA_FULL_CUDAGRAPH_SUPPORTED = old_full_cudagraph_supported
    _ORIGINAL_V1_FLASH_ATTN_FORWARD = (
        dense_original_forward if dense_original_forward is not None else old_forward
    )
    _FLASH_ATTN_FORWARD_PATCHED = True

def _serialize_config(config: SparseControllerConfig) -> Dict[str, object]:
    payload = asdict(config)
    trigger = payload.get("trigger")
    if isinstance(trigger, dict):
        if "single_end_tokens" in trigger:
            trigger["single_end_tokens"] = list(trigger["single_end_tokens"])
        if "pair_end_tokens" in trigger:
            trigger["pair_end_tokens"] = [list(pair) for pair in trigger["pair_end_tokens"]]
        if "start_exclude_tokens" in trigger:
            trigger["start_exclude_tokens"] = list(trigger["start_exclude_tokens"])
    return payload


def _snapshot_env_var(name: str) -> Tuple[bool, Optional[str]]:
    return name in os.environ, os.environ.get(name)


def _restore_env_var(name: str, snapshot: Tuple[bool, Optional[str]]) -> None:
    existed, value = snapshot
    if existed:
        os.environ[name] = value or ""
    else:
        os.environ.pop(name, None)


def _restore_sparse_patch_entry_env(
    snapshot: Tuple[Tuple[bool, Optional[str]], Tuple[bool, Optional[str]]],
) -> None:
    serialized_config, pythonpath = snapshot
    _restore_env_var(_SERIALIZED_CONFIG_ENV, serialized_config)
    _restore_env_var("PYTHONPATH", pythonpath)


def _preflight_controller_config_contract(config: SparseControllerConfig) -> None:
    """Install-time fail-fast for config combinations that die later and darker.

    [SERVE-LIVENESS-PREFLIGHT 2026-07-09] this transaction is the single
    convergence point of BOTH install paths (bench ``apply_vllm_sparse_patch``
    and serve ``ensure_vllm_sparse_patch_from_env``), so contracts here cover
    the serve path that previously ran bare:

    1. dual-gen (default ON since 758185e) requires page residency — without
       it the first long-request rebuild raises deep inside
       selection_worker ("compact dual-gen requires page residency"), an
       engine-killing error far from its config root cause. Promote that
       death to install time with an actionable message.
    2. FORCE_DENSE / FORCE_COMPACT_OFF conflict with compact_recent: the env
       silently routes every row dense (row_policy) with zero counters — a
       sparse-liveness trap on serve/LongBench. Previously only the bench
       path checked FORCE_DENSE.
    """
    from patches.sparse_constants import (
        _DYNAMIC_ENV,
        _FORCE_COMPACT_OFF_CACHED,
        _FORCE_DENSE_CACHED,
        compact_gen_count,
    )

    if compact_gen_count() > 1 and not bool(
        getattr(config, "compact_page_residency_enabled", False)
    ):
        raise RuntimeError(
            "sparse config contract: dual-generation compact read is ON "
            "(VLLM_SPARSE_COMPACT_DUAL_GEN default) but the controller config "
            "has compact_page_residency_enabled=false (legacy arena is "
            "unsupported for dual-gen). Fix the serve/bench config JSON: set "
            "compact_page_residency_enabled=true with positive "
            "max_live_sparse_slots and compact_blocks_per_slot, or explicitly "
            "run single-generation via VLLM_SPARSE_COMPACT_DUAL_GEN=0."
        )
    if config.attn_mode == "compact_recent":
        force_dense = (
            (os.environ.get("VLLM_SPARSE_FORCE_DENSE") == "1")
            if _DYNAMIC_ENV
            else _FORCE_DENSE_CACHED
        )
        force_compact_off = (
            (os.environ.get("VLLM_SPARSE_FORCE_COMPACT_OFF") == "1")
            if _DYNAMIC_ENV
            else _FORCE_COMPACT_OFF_CACHED
        )
        if force_dense or force_compact_off:
            offender = (
                "VLLM_SPARSE_FORCE_DENSE"
                if force_dense
                else "VLLM_SPARSE_FORCE_COMPACT_OFF"
            )
            raise RuntimeError(
                f"attn_mode=compact_recent conflicts with {offender}=1: every "
                "decode row would silently route dense (zero sparse activity, "
                "zero counters). Unset it, or run a dense baseline without "
                "installing the sparse controller."
            )


def _install_controller_patch_transaction(
    config: SparseControllerConfig,
    *,
    env_snapshot: Optional[
        Tuple[Tuple[bool, Optional[str]], Tuple[bool, Optional[str]]]
    ] = None,
) -> "VLLMSparseController":
    controller_before = _GLOBAL_CONTROLLER
    mutation_started = False
    try:
        # Phase 1 is read-only.  A rejected reapply must preserve the old live
        # controller and hooks instead of entering teardown with a lost lease.
        _preflight_controller_hook_leases()
        _preflight_controller_config_contract(config)
        controller = _set_controller(config)
        mutation_started = True
        _install_patch()
        _patch_flash_metadata_builder()
        return controller
    except Exception as install_exc:
        rollback_needed = bool(
            mutation_started or _GLOBAL_CONTROLLER is not controller_before
        )
        rollback_exc: Optional[BaseException] = None
        if rollback_needed:
            try:
                disable_vllm_sparse_patch()
            except Exception as exc:
                rollback_exc = exc
                _log.error(
                    "Failed to roll back vLLM sparse patch after install failure",
                    exc_info=True,
                )
        if rollback_needed:
            os.environ.pop(_SERIALIZED_CONFIG_ENV, None)
            if env_snapshot is not None:
                # A post-activation failure tears the runtime down fail-closed;
                # restoring an old enabled JSON would permit a zombie lazy
                # reinstall.  Preserve only the caller's PYTHONPATH contract.
                _restore_env_var("PYTHONPATH", env_snapshot[1])
        elif env_snapshot is not None:
            _restore_sparse_patch_entry_env(env_snapshot)
        if rollback_exc is not None:
            raise RuntimeError(
                "vLLM sparse patch install failed "
                f"({type(install_exc).__name__}: {install_exc}) and rollback "
                "also failed"
            ) from rollback_exc
        raise


def apply_vllm_sparse_patch(config: SparseControllerConfig) -> "VLLMSparseController":
    """Install the sparse wrapper and expose config to child processes."""
    # Config-contract fail-fast (FORCE_DENSE conflict, dual-gen x residency)
    # lives in _install_controller_patch_transaction: the single convergence
    # point shared with the serve path (ensure_vllm_sparse_patch_from_env).
    env_snapshot = (
        _snapshot_env_var(_SERIALIZED_CONFIG_ENV),
        _snapshot_env_var("PYTHONPATH"),
    )
    os.environ[_SERIALIZED_CONFIG_ENV] = json.dumps(_serialize_config(config))
    repo_root = os.path.dirname(os.path.dirname(__file__))
    pythonpath = os.environ.get("PYTHONPATH", "")
    paths = pythonpath.split(os.pathsep) if pythonpath else []
    if repo_root not in paths:
        paths.insert(0, repo_root)
    os.environ["PYTHONPATH"] = os.pathsep.join(paths)
    return _install_controller_patch_transaction(config, env_snapshot=env_snapshot)


def ensure_vllm_sparse_patch_from_env() -> Optional["VLLMSparseController"]:
    """Install the sparse patch when serialized config is explicitly enabled."""
    payload = os.environ.get(_SERIALIZED_CONFIG_ENV)
    if not payload:
        return None
    config = _deserialize_config(payload)
    if config is None or not config.enabled:
        return None
    env_snapshot = (
        _snapshot_env_var(_SERIALIZED_CONFIG_ENV),
        _snapshot_env_var("PYTHONPATH"),
    )
    return _install_controller_patch_transaction(config, env_snapshot=env_snapshot)


def disable_vllm_sparse_patch() -> None:
    global _GLOBAL_CONTROLLER, _PATCH_INSTALLED
    global _PREPARE_PATCHED, _ORIGINAL_PREPARE_INPUTS
    global _BATCH_EXECUTION_ADMISSION_PATCHED
    global _ORIGINAL_DETERMINE_BATCH_EXECUTION
    global _INSTALLED_DETERMINE_BATCH_EXECUTION_WRAPPER
    global _DUMMY_RUN_PATCHED, _ORIGINAL_DUMMY_RUN
    global _FLASH_METADATA_PATCHED, _ORIGINAL_FLASH_METADATA_BUILD
    global _REQUEST_PATCHED, _ORIGINAL_APPEND_OUTPUT_TOKEN_IDS
    global _KV_INIT_PATCHED, _ORIGINAL_INIT_KV_CACHE
    global _UBATCH_WRAPPER_PATCHED, _ORIGINAL_UBATCH_WRAPPER_CALL
    global _CUDAGRAPH_WRAPPER_PATCHED, _ORIGINAL_CUDAGRAPH_WRAPPER_CALL
    global _MODEL_FORWARD_REFRESH_OWNER_PATCHED
    global _ORIGINAL_MODEL_FORWARD_FOR_REFRESH_OWNER
    global _UPDATE_STATES_PATCHED, _ORIGINAL_UPDATE_STATES
    global _INSTALLED_UPDATE_STATES_WRAPPER
    global _ORIGINAL_V1_FLASH_ATTN_VARLEN_FUNC
    global _ORIGINAL_V1_FLASH_ATTN_FORWARD, _ORIGINAL_V1_FLASH_ATTN_GET_SCHEDULER_METADATA
    global _ORIGINAL_V1_FLASH_ATTN_GET_FLASH_ATTN_VERSION
    global _ORIGINAL_FA_UTILS_FLASH_ATTN_VARLEN_FUNC, _ORIGINAL_FA_UTILS_GET_SCHEDULER_METADATA
    global _ORIGINAL_FA_UTILS_GET_FLASH_ATTN_VERSION
    global _ORIGINAL_V1_FLASH_METADATA_FULL_CUDAGRAPH_SUPPORTED
    global _FLASH_ATTN_FORWARD_PATCHED
    global _COMPACT_PAGE_RESIDENCY_PATCHED, _ORIGINAL_KV_CACHE_MANAGER_INIT
    global _ORIGINAL_BLOCK_POOL_METHODS, _COMPACT_PAGE_BLOCK_POOL_CLS
    global _COMPACT_PAGE_KV_CACHE_MANAGER_CLS
    global _ORIGINAL_SET_ASYNC_SAMPLED_TOKEN_IDS
    global _INSTALLED_SET_ASYNC_SAMPLED_TOKEN_IDS_WRAPPER
    _preflight_controller_hook_leases()
    _GLOBAL_CONTROLLER = None
    if _SERIALIZED_CONFIG_ENV in os.environ:
        del os.environ[_SERIALIZED_CONFIG_ENV]
    _PATCH_INSTALLED = False
    if _UBATCH_WRAPPER_PATCHED and _ORIGINAL_UBATCH_WRAPPER_CALL is not None:
        try:
            from vllm.v1.worker.gpu_ubatch_wrapper import UBatchWrapper  # type: ignore[import]
            UBatchWrapper.__call__ = _ORIGINAL_UBATCH_WRAPPER_CALL  # type: ignore[assignment]
        except Exception:
            _log.error("Failed to restore UBatchWrapper.__call__ during patch uninstall")
            raise
    _UBATCH_WRAPPER_PATCHED = False
    _ORIGINAL_UBATCH_WRAPPER_CALL = None
    if (
        _CUDAGRAPH_WRAPPER_PATCHED
        and _ORIGINAL_CUDAGRAPH_WRAPPER_CALL is not None
    ):
        try:
            from vllm.compilation.cuda_graph import CUDAGraphWrapper  # type: ignore[import]
            CUDAGraphWrapper.__call__ = _ORIGINAL_CUDAGRAPH_WRAPPER_CALL  # type: ignore[assignment]
        except Exception:
            _log.error("Failed to restore CUDAGraphWrapper.__call__ during patch uninstall")
            raise
    _CUDAGRAPH_WRAPPER_PATCHED = False
    _ORIGINAL_CUDAGRAPH_WRAPPER_CALL = None
    if (
        _MODEL_FORWARD_REFRESH_OWNER_PATCHED
        and _ORIGINAL_MODEL_FORWARD_FOR_REFRESH_OWNER is not None
    ):
        try:
            from vllm.v1.worker.gpu_model_runner import GPUModelRunner  # type: ignore[import]
            GPUModelRunner._model_forward = _ORIGINAL_MODEL_FORWARD_FOR_REFRESH_OWNER  # type: ignore[assignment]
        except Exception:
            _log.error("Failed to restore _model_forward during patch uninstall")
            raise
    _MODEL_FORWARD_REFRESH_OWNER_PATCHED = False
    _ORIGINAL_MODEL_FORWARD_FOR_REFRESH_OWNER = None
    if _FLASH_ATTN_FORWARD_PATCHED:
        if _ORIGINAL_V1_FLASH_ATTN_FORWARD is None:
            raise RuntimeError(
                "E_SFI_FA3_GATEWAY_INSTALL_STATE: installed FA3 gateway has "
                "no forward predecessor"
        )
        try:
            from vllm.v1.attention.backends import flash_attn as v1_flash_attn
            needs_fa_utils_restore = any(
                value is not None
                for value in (
                    _ORIGINAL_FA_UTILS_FLASH_ATTN_VARLEN_FUNC,
                    _ORIGINAL_FA_UTILS_GET_SCHEDULER_METADATA,
                    _ORIGINAL_FA_UTILS_GET_FLASH_ATTN_VERSION,
                )
            )
            fa_utils = (
                import_fa_utils_module() if needs_fa_utils_restore else None
            )
            if _ORIGINAL_V1_FLASH_ATTN_VARLEN_FUNC is not None:
                v1_flash_attn.flash_attn_varlen_func = _ORIGINAL_V1_FLASH_ATTN_VARLEN_FUNC  # type: ignore[assignment]
            v1_flash_attn.FlashAttentionImpl.forward = _ORIGINAL_V1_FLASH_ATTN_FORWARD  # type: ignore[assignment]
            if _ORIGINAL_V1_FLASH_ATTN_GET_SCHEDULER_METADATA is not None:
                v1_flash_attn.get_scheduler_metadata = _ORIGINAL_V1_FLASH_ATTN_GET_SCHEDULER_METADATA  # type: ignore[assignment]
            if _ORIGINAL_V1_FLASH_ATTN_GET_FLASH_ATTN_VERSION is not None:
                v1_flash_attn.get_flash_attn_version = _ORIGINAL_V1_FLASH_ATTN_GET_FLASH_ATTN_VERSION  # type: ignore[assignment]
            metadata_builder_cls = getattr(v1_flash_attn, "FlashAttentionMetadataBuilder", None)
            if (
                metadata_builder_cls is not None
                and _ORIGINAL_V1_FLASH_METADATA_FULL_CUDAGRAPH_SUPPORTED is not None
            ):
                metadata_builder_cls.full_cudagraph_supported = _ORIGINAL_V1_FLASH_METADATA_FULL_CUDAGRAPH_SUPPORTED  # type: ignore[assignment]
            if fa_utils is not None:
                if _ORIGINAL_FA_UTILS_FLASH_ATTN_VARLEN_FUNC is not None:
                    fa_utils.flash_attn_varlen_func = _ORIGINAL_FA_UTILS_FLASH_ATTN_VARLEN_FUNC  # type: ignore[assignment]
                if _ORIGINAL_FA_UTILS_GET_SCHEDULER_METADATA is not None:
                    fa_utils.get_scheduler_metadata = _ORIGINAL_FA_UTILS_GET_SCHEDULER_METADATA  # type: ignore[assignment]
                if _ORIGINAL_FA_UTILS_GET_FLASH_ATTN_VERSION is not None:
                    fa_utils.get_flash_attn_version = _ORIGINAL_FA_UTILS_GET_FLASH_ATTN_VERSION  # type: ignore[assignment]
        except Exception:
            _log.error("Failed to restore v1 flash_attn forward/helpers during patch uninstall")
            raise
    _ORIGINAL_V1_FLASH_ATTN_VARLEN_FUNC = None
    _ORIGINAL_V1_FLASH_ATTN_FORWARD = None
    _ORIGINAL_V1_FLASH_ATTN_GET_SCHEDULER_METADATA = None
    _ORIGINAL_V1_FLASH_ATTN_GET_FLASH_ATTN_VERSION = None
    _ORIGINAL_FA_UTILS_FLASH_ATTN_VARLEN_FUNC = None
    _ORIGINAL_FA_UTILS_GET_SCHEDULER_METADATA = None
    _ORIGINAL_FA_UTILS_GET_FLASH_ATTN_VERSION = None
    _ORIGINAL_V1_FLASH_METADATA_FULL_CUDAGRAPH_SUPPORTED = None
    _FLASH_ATTN_FORWARD_PATCHED = False
    if _UPDATE_STATES_PATCHED:
        if (
            _ORIGINAL_UPDATE_STATES is None
            or _INSTALLED_UPDATE_STATES_WRAPPER is None
        ):
            raise RuntimeError("E_SPARSE_UPDATE_STATES_HOOK_LEASE_STATE")
        try:
            from vllm.v1.worker.gpu_model_runner import GPUModelRunner  # type: ignore[import]

            if (
                GPUModelRunner._update_states
                is not _INSTALLED_UPDATE_STATES_WRAPPER
            ):
                raise RuntimeError("E_SPARSE_UPDATE_STATES_HOOK_LEASE_LOST")
            GPUModelRunner._update_states = _ORIGINAL_UPDATE_STATES  # type: ignore[assignment]
        except Exception:
            _log.error("Failed to restore _update_states during patch uninstall")
            raise
        _UPDATE_STATES_PATCHED = False
        _ORIGINAL_UPDATE_STATES = None
        _INSTALLED_UPDATE_STATES_WRAPPER = None
    if _ORIGINAL_SET_ASYNC_SAMPLED_TOKEN_IDS is not None:
        try:
            from vllm.v1.worker.gpu_input_batch import InputBatch  # type: ignore[import]

            InputBatch.set_async_sampled_token_ids = (  # type: ignore[assignment]
                _ORIGINAL_SET_ASYNC_SAMPLED_TOKEN_IDS
            )
        except Exception:
            _log.error(
                "Failed to restore async sampled-token stash hook during patch uninstall"
            )
            raise
    _ORIGINAL_SET_ASYNC_SAMPLED_TOKEN_IDS = None
    _INSTALLED_SET_ASYNC_SAMPLED_TOKEN_IDS_WRAPPER = None
    if _PREPARE_PATCHED and _ORIGINAL_PREPARE_INPUTS is not None:
        try:
            from vllm.v1.worker.gpu_model_runner import GPUModelRunner  # type: ignore[import]
            GPUModelRunner._prepare_inputs = _ORIGINAL_PREPARE_INPUTS  # type: ignore[assignment]
        except Exception:
            _log.error("Failed to restore _prepare_inputs during patch uninstall")
            raise
    _PREPARE_PATCHED = False
    _ORIGINAL_PREPARE_INPUTS = None
    if _BATCH_EXECUTION_ADMISSION_PATCHED:
        try:
            from vllm.v1.worker.gpu_model_runner import GPUModelRunner  # type: ignore[import]

            if (
                GPUModelRunner._determine_batch_execution_and_padding
                is not _INSTALLED_DETERMINE_BATCH_EXECUTION_WRAPPER
            ):
                raise RuntimeError(
                    "E_PREFILL_CAPTURE_GRAPH_ADMISSION: patch lease lost"
                )
            GPUModelRunner._determine_batch_execution_and_padding = (  # type: ignore[assignment]
                _ORIGINAL_DETERMINE_BATCH_EXECUTION
            )
        except Exception:
            _log.error(
                "Failed to restore batch execution cudagraph admission patch",
                exc_info=True,
            )
            raise
    _BATCH_EXECUTION_ADMISSION_PATCHED = False
    _ORIGINAL_DETERMINE_BATCH_EXECUTION = None
    _INSTALLED_DETERMINE_BATCH_EXECUTION_WRAPPER = None
    if _DUMMY_RUN_PATCHED and _ORIGINAL_DUMMY_RUN is not None:
        try:
            from vllm.v1.worker.gpu_model_runner import GPUModelRunner  # type: ignore[import]
            GPUModelRunner._dummy_run = _ORIGINAL_DUMMY_RUN  # type: ignore[assignment]
        except Exception:
            _log.error("Failed to restore _dummy_run during patch uninstall")
            raise
    _DUMMY_RUN_PATCHED = False
    _ORIGINAL_DUMMY_RUN = None
    if _FLASH_METADATA_PATCHED and _ORIGINAL_FLASH_METADATA_BUILD is not None:
        try:
            from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadataBuilder  # type: ignore[import]
            FlashAttentionMetadataBuilder.build = _ORIGINAL_FLASH_METADATA_BUILD  # type: ignore[assignment]
        except Exception:
            _log.error("Failed to restore FlashAttentionMetadataBuilder.build during patch uninstall")
            raise
    _FLASH_METADATA_PATCHED = False
    _ORIGINAL_FLASH_METADATA_BUILD = None
    if _REQUEST_PATCHED and _ORIGINAL_APPEND_OUTPUT_TOKEN_IDS is not None:
        try:
            from vllm.v1.request import Request  # type: ignore[import]
            Request.append_output_token_ids = _ORIGINAL_APPEND_OUTPUT_TOKEN_IDS  # type: ignore[assignment]
        except Exception:
            _log.error("Failed to restore Request.append_output_token_ids during patch uninstall")
            raise
    _REQUEST_PATCHED = False
    _ORIGINAL_APPEND_OUTPUT_TOKEN_IDS = None
    if _KV_INIT_PATCHED and _ORIGINAL_INIT_KV_CACHE is not None:
        try:
            from vllm.v1.worker.gpu_model_runner import GPUModelRunner  # type: ignore[import]
            GPUModelRunner.initialize_kv_cache = _ORIGINAL_INIT_KV_CACHE  # type: ignore[assignment]
        except Exception:
            _log.error("Failed to restore initialize_kv_cache during patch uninstall")
            raise
    _KV_INIT_PATCHED = False
    _ORIGINAL_INIT_KV_CACHE = None
    _restore_compact_page_residency_core_patch()
