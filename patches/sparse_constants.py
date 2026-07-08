"""
Immutable constants and cached environment variables for the sparse attention engine.

All values are read once at module import time and never mutated afterwards.
This module has no dependency on vllm_sparse_patch.py, so it can be imported
freely by both the main patch and all runtime worker modules.
"""
from __future__ import annotations

import os
from typing import Optional

from patches.runtime_contracts import validate_capture_inflight

__all__ = [
    "_DYNAMIC_ENV",
    "_FORCE_DENSE_CACHED",
    "_FORCE_COMPACT_OFF_CACHED",
    "_RELEASE_ON_IDLE_CACHED",
    "_REBUILD_PTRS_PINNED_CACHED",
    "_WRITER_TOKEN_TILE_CACHED",
    "_GATHER_QOS_CACHED",
    "_GATHER_QOS_NUM_WARPS_CACHED",
    "_GATHER_QOS_NUM_STAGES_CACHED",
    "_REBUILD_PHYSICAL_BLOCK_SORT_CACHED",
    "_SELECTOR_TRUSTED_SHAPES_CACHED",
    "_SELECTOR_FAST_SIG_CACHED",
    "_SELECTOR_CPP_PREPROC_CACHED",
    "_SELECTOR_CPP_STACK_CACHED",
    "_SELECTOR_PIPELINE_UNIFIED_CACHED",
    "_SELECTOR_LOGS_CACHE_R_CACHED",
    "_ONE_SHOT_ASYNC_BOOTSTRAP_CACHED",
    "_NATIVE_LIFECYCLE_CACHED",
    "_LIFECYCLE_LIVE_STEADY_CACHED",
    "_LIFECYCLE_LIVE_STEADY_ASSERT_CACHED",
    "_LIFECYCLE_ONLY_FOR_SPEC_CACHED",
    "_RRP_SAME_PAGE_SKIP_REVALIDATION_CACHED",
    "_SAME_PAGE_MINIMAL_UPDATE_CACHED",
    "_SAME_PAGE_READY_EVENT_ONLY_CACHED",
    "_PROBE_CACHEKEY_CACHED",
    "_SAME_PAGE_MINIMAL_ASSERT_CACHED",
    "_DECODE_BOUNDS_KERNEL_CACHED",
    "_LOGF_OUT_FP32_CACHED",
    "_CAPTURE_CHUNK",
    "_CAPTURE_KV_BUCKET_CACHED",
    "_CAPTURE_IN_FLIGHT",
    "_CAPTURE_REDUCE_GROUP",
    "_COMPACT_DUAL_GEN_CACHED",
    "compact_gen_count",
    "_REFRESH_STREAM_PRIORITY_CACHED",
    "_ASYNC_REFRESH_CACHED",
    "_REFRESH_PRODUCER_SPLIT_SELECTOR_WRITER_RELEASE_MODE_CACHED",
    "_REFRESH_PRODUCER_SPLIT_SELECTOR_WRITER_RELEASE_CACHED",
    "_REFRESH_PRODUCER_SPLIT_SELECTOR_WRITER_RELEASE_AUTO_CACHED",
    "_REFRESH_PRODUCER_SPLIT_SELECTOR_WRITER_RELEASE_MAX_PER_HANDLE_CACHED",
    "_REFRESH_PRODUCER_SPLIT_SELECTOR_WRITER_RELEASE_MIN_LAYER_START_CACHED",
    "_REFRESH_GROUPED_ASYNC_ENVELOPE_CACHED",
    "_REFRESH_ENQUEUE_STAGGER_CACHED",
    "_REPLAY_REFRESH_NOOP_FAST_SKIP_CACHED",
    "_REFRESH_REBUILD_MAX_DELAY_STEPS_CACHED",
    "_REFRESH_REBUILD_CHECK_CACHED",
    "_PENDING_REBUILD_MAX_QUEUE_CACHED",
    "_PREFILL_RELEASE_GRACE_STEPS",
    "_STEP_PROFILE_CACHED",
    "_STEP_PROFILE_DETAIL_CACHED",
    "_STEP_PROFILE_EVERY_CACHED",
    "_STEP_PROFILE_LOG_CACHED",
    "_VALIDATE_COMPACT_META_CACHED",
    "_SELECTOR_FIXED_K_CACHED",
    "_SELECTOR_SELECTED_INDICES_OUT_CACHED",
    "_SELECTOR_PIPELINE_WORKSPACE_CACHED",
    "_SELECTOR_FIXED_SHAPE_TOPK_CACHED",
    "_SELECTOR_KBUCKET_CACHED",
    "_SELECTOR_KEY_NORMS_CACHE_CAP_CACHED",
    "_SELECTOR_TOPK_GRAPH_CACHED",
    "_SELECTOR_GRAPH_LRU_CACHED",
    "_selector_graph_lru_enabled",
    "_REFRESH_PROFILE_CACHED",
    "_REFRESH_PROFILE_CALL_MIN_CACHED",
    "_REFRESH_PROFILE_EVERY_CACHED",
    "_REFRESH_PROFILE_LOG_CACHED",
    "_REFRESH_PROFILE_DETAIL_CACHED",
    "_ASYNC_PRODUCER_GPU_PROFILE_CACHED",
    "_DEFERRED_SELECTOR_PROFILE_DETAIL_CACHED",
    "_REFRESH_MICRO_PROFILE_CACHED",
    "_VALIDATE_META_CONTRACT_CACHED",
    "_VALIDATE_LAYER_SLOT_MAP_CACHED",
    "_ATTN_MODE_CACHED",
    # Peripheral companion scaffolding (see module-level ⚠️ comment near definition):
    "_SKIP_PAGE_SPARSE_STATE_CACHED",
    "should_skip_page_sparse_state",
    "_ROW_MODE_DENSE",
    "_ROW_MODE_COMPACT",
    "_ROW_MODE_LOG_F_PREFILL",
    "_ROW_MODE_LOG_F_REFRESH",
    "_LOGF_PRODUCER_NONE",
    "_LOGF_PRODUCER_ATTN",
    "_FREE_SLOT_ID",
    "_is_free_slot_id",
]

# ---------------------------------------------------------------------------
# Dynamic environment detection
# ---------------------------------------------------------------------------
_DYNAMIC_ENV = (os.environ.get("PYTEST_CURRENT_TEST") is not None) or (
    os.environ.get("VLLM_SPARSE_DYNAMIC_ENV", "0") == "1"
)

# ---------------------------------------------------------------------------
# Force-mode flags
# ---------------------------------------------------------------------------
_FORCE_DENSE_CACHED = os.environ.get("VLLM_SPARSE_FORCE_DENSE", "0") == "1"
_FORCE_COMPACT_OFF_CACHED = os.environ.get("VLLM_SPARSE_FORCE_COMPACT_OFF", "0") == "1"
_RELEASE_ON_IDLE_CACHED = os.environ.get("VLLM_SPARSE_RELEASE_ON_IDLE", "1") == "1"
# Robust mode: demote not-compact-ready one-shot decode rows to dense full-KV
# instead of raising "compact_ready before replay". Default ON (1).
_ONE_SHOT_BLOCKED_DENSE_FALLBACK_CACHED = (
    os.environ.get("VLLM_SPARSE_ONE_SHOT_BLOCKED_DENSE_FALLBACK", "1") == "1"
)

# ---------------------------------------------------------------------------
# Rebuild / gather hot-path knobs
# ---------------------------------------------------------------------------
_REBUILD_PTRS_PINNED_CACHED = os.environ.get("VLLM_SPARSE_REBUILD_PTRS_PINNED", "1") == "1"
_WRITER_TOKEN_TILE_CACHED: int = int(os.environ.get("VLLM_SPARSE_WRITER_TOKEN_TILE", "16") or "0")
_GATHER_QOS_CACHED = os.environ.get("VLLM_SPARSE_GATHER_QOS", "1") == "1"
_GATHER_QOS_NUM_WARPS_CACHED: int = int(os.environ.get("VLLM_SPARSE_GATHER_QOS_NUM_WARPS", "0") or "0")
_GATHER_QOS_NUM_STAGES_CACHED: int = int(os.environ.get("VLLM_SPARSE_GATHER_QOS_NUM_STAGES", "0") or "0")

# ---------------------------------------------------------------------------
# Selector / rebuild experiment switches
# ---------------------------------------------------------------------------
_REBUILD_PHYSICAL_BLOCK_SORT_CACHED = (
    os.environ.get("VLLM_SPARSE_REBUILD_PHYSICAL_BLOCK_SORT", "0") == "1"
)
_SELECTOR_TRUSTED_SHAPES_CACHED = os.environ.get("VLLM_SPARSE_SELECTOR_TRUSTED_SHAPES", "1") == "1"
# Upper bound on distinct key_norms_all GPU buffers retained at once. Each is
# [layers, batch, kv_heads, align_up(kv_len+1,4096)] fp16 = multi-GiB at long
# contexts; without a cap the per-(bucket, slot-set) dict grows unbounded under a
# variable-length server load and OOMs outside vLLM's util budget. 2 keeps the
# active set + one in-flight transition; raise to disable bounding.
_SELECTOR_KEY_NORMS_CACHE_CAP_CACHED: int = max(
    1, int(os.environ.get("VLLM_SPARSE_SELECTOR_KEY_NORMS_CACHE_CAP", "2") or "2")
)
_SELECTOR_FAST_SIG_CACHED = os.environ.get("VLLM_SPARSE_SELECTOR_FAST_SIG", "1") == "1"
_SELECTOR_CPP_PREPROC_CACHED = os.environ.get("VLLM_SPARSE_SELECTOR_CPP_PREPROC", "1") == "1"
_SELECTOR_CPP_STACK_CACHED = os.environ.get("VLLM_SPARSE_SELECTOR_CPP_STACK", "1") == "1"
_SELECTOR_PIPELINE_UNIFIED_CACHED = (
    os.environ.get("VLLM_SPARSE_SELECTOR_PIPELINE_UNIFIED", "1") == "1"
)
_DECODE_BOUNDS_KERNEL_CACHED = os.environ.get("VLLM_SPARSE_DECODE_BOUNDS_KERNEL", "1") == "1"
_SELECTOR_LOGS_CACHE_R_CACHED = (
    os.environ.get("VLLM_SPARSE_SELECTOR_LOGS_CACHE_R", "").strip() == "1"
)
# --- one-shot async bootstrap gate deployment flags (phase_h FIND #5) ---
_ONE_SHOT_ASYNC_BOOTSTRAP_CACHED = (
    os.environ.get("VLLM_SPARSE_ONE_SHOT_ASYNC_BOOTSTRAP", "0") == "1"
)
# --- STEADY-decode metadata hot-path deployment flags (phase_h FIND #1) ---
# Cached deployment toggles only; _VLLM_SPARSE_SESSION_SPEC stays a LIVE read
# (sticky runtime latch in patch_installer) and is NOT cached here.
_NATIVE_LIFECYCLE_CACHED = (
    os.environ.get("VLLM_SPARSE_NATIVE_LIFECYCLE", "0").strip() == "1"
)
_LIFECYCLE_LIVE_STEADY_CACHED = (
    os.environ.get("VLLM_SPARSE_LIFECYCLE_LIVE_STEADY", "0").strip() == "1"
)
_LIFECYCLE_LIVE_STEADY_ASSERT_CACHED = (
    os.environ.get("VLLM_SPARSE_LIFECYCLE_LIVE_STEADY_ASSERT", "0").strip() == "1"
)
_LIFECYCLE_ONLY_FOR_SPEC_CACHED = (
    os.environ.get("VLLM_SPARSE_LIFECYCLE_ONLY_FOR_SPEC") == "1"
)
_RRP_SAME_PAGE_SKIP_REVALIDATION_CACHED = (
    os.environ.get("VLLM_SPARSE_RRP_SAME_PAGE_SKIP_REVALIDATION", "1") == "1"
)
_SAME_PAGE_MINIMAL_UPDATE_CACHED = (
    os.environ.get("VLLM_SPARSE_SAME_PAGE_MINIMAL_UPDATE", "1") == "1"
)
_SAME_PAGE_READY_EVENT_ONLY_CACHED = (
    os.environ.get("VLLM_SPARSE_SAME_PAGE_READY_EVENT_ONLY", "1") == "1"
)
_PROBE_CACHEKEY_CACHED = (
    os.environ.get("VLLM_PROBE_CACHEKEY") == "1"
)
_SAME_PAGE_MINIMAL_ASSERT_CACHED = (
    os.environ.get("VLLM_SPARSE_SAME_PAGE_MINIMAL_ASSERT") == "1"
)

# ---------------------------------------------------------------------------
# log_f capture output dtype control
# ---------------------------------------------------------------------------
_LOGF_OUT_FP32_CACHED = os.environ.get("VLLM_SPARSE_LOGF_OUT_FP32", "0") == "1"

# ---------------------------------------------------------------------------
# Chunk-batched capture ring (memory + async overlap)
# ---------------------------------------------------------------------------
try:
    _CAPTURE_CHUNK: int = int(os.environ.get("VLLM_SPARSE_CAPTURE_CHUNK", "14") or "14")
except ValueError:
    _CAPTURE_CHUNK = 14
if _CAPTURE_CHUNK <= 0:
    _CAPTURE_CHUNK = 14

try:
    _CAPTURE_KV_BUCKET_CACHED: int = int(os.environ.get("VLLM_SPARSE_CAPTURE_KV_BUCKET", "2048") or "2048")
except ValueError:
    _CAPTURE_KV_BUCKET_CACHED = 2048

try:
    _CAPTURE_IN_FLIGHT: int = int(os.environ.get("VLLM_SPARSE_CAPTURE_IN_FLIGHT", "2") or "2")
except ValueError:
    _CAPTURE_IN_FLIGHT = 2
_CAPTURE_IN_FLIGHT = validate_capture_inflight(_CAPTURE_IN_FLIGHT)

# ---------------------------------------------------------------------------
# Per-G-layer capture reduce group (memory-bounded raw ring; bit-identical).
# G==0 => scheme OFF: legacy per-chunk reduce over a _CAPTURE_CHUNK-deep raw slab.
# G in {1,2,4} => reduce every G layers into a (G*_CAPTURE_IN_FLIGHT)-slot raw RING,
#   draining last_n=16 -> window=1 into the (unchanged) chunk-deep p_f tape. Raw staging
#   becomes chunk-INDEPENDENT (bounded by G*_CAPTURE_IN_FLIGHT slots). Requires
#   _CAPTURE_CHUNK % G == 0; otherwise falls back to 0 (OFF, fail-safe).
# Default 0 until the A/B gate passes; flip to 1 to ship G=1 ON.
# ---------------------------------------------------------------------------
try:
    _CAPTURE_REDUCE_GROUP: int = int(os.environ.get("VLLM_SPARSE_CAPTURE_REDUCE_GROUP", "1") or "1")
except ValueError:
    _CAPTURE_REDUCE_GROUP = 0
if _CAPTURE_REDUCE_GROUP not in (0, 1, 2, 4):
    _CAPTURE_REDUCE_GROUP = 0
if _CAPTURE_REDUCE_GROUP > 0 and (_CAPTURE_CHUNK % _CAPTURE_REDUCE_GROUP) != 0:
    _CAPTURE_REDUCE_GROUP = 0

# ---------------------------------------------------------------------------
# Refresh stream management
# ---------------------------------------------------------------------------
# [DUAL-GEN-L2] compact 双代读:重建写备用半区、旧代照读、writer_done 后原子
# 切代(行不再因 INFLIGHT 退 rail)。开时 lease/arena 容量 ×2(KVB 预算须按
# 2×lease 复核;发布仓 run_speed.sh 预检公式已 ×gen_count)。
# [DUAL-GEN 转正 2026-07-08 用户拍板] 默认开:INFLIGHT dense-rail 兜底轨随之
# 不再被走(4B bs8×12k 同卡单变量:TP1 262.4→299.4 +14.1%/TP2 286.1→331.8
# +16%,中位步 21.3→12.7ms=dense 轨移除直接形态;07-06"12k 净负"为旧触发语义
# 账)。曾挡默认的两案处置:
#   ①双代×TP>1×32k 楔死——**已破案收案**(交接 §10:重绑继承层间错代
#     read_gen×vLLM 吞非 output-rank 异常;[DUAL-GEN-REBIND-NORMALIZE]+
#     [TP-EXC-FAILFAST] 根修后 blocking 32k×TP2 ×8=0 楔死/0 parity);
#   ②双代 ×2 lease 在 32k×TP1 需 KVB≥41GiB=40G 卡容量壁——物理约束非 bug,
#     标准四档 tier 均已复核可容双代;该形态跑 =0 单代档或加大 KVB,
#     run_speed 预检公式(×gen_count)会先警。
# 显式 =0 为单代诊断档。
_COMPACT_DUAL_GEN_CACHED = os.environ.get("VLLM_SPARSE_COMPACT_DUAL_GEN", "1") != "0"


def compact_gen_count() -> int:
    """双代开=2,关=1;lease/arena 容量与 sub-slot 域统一乘此因子。"""
    return 2 if _COMPACT_DUAL_GEN_CACHED else 1


_REFRESH_STREAM_PRIORITY_CACHED: int = int(os.environ.get("VLLM_SPARSE_REFRESH_STREAM_PRIORITY", "0"))
_ASYNC_REFRESH_CACHED = os.environ.get("VLLM_SPARSE_ASYNC_REFRESH", "1") == "1"
_REFRESH_PRODUCER_SPLIT_SELECTOR_WRITER_RELEASE_MODE_CACHED = (
    os.environ.get("VLLM_SPARSE_REFRESH_SPLIT_SELECTOR_WRITER_RELEASE", "auto")
    .strip()
    .lower()
)
if _REFRESH_PRODUCER_SPLIT_SELECTOR_WRITER_RELEASE_MODE_CACHED in (
    "",
    "auto",
    "adaptive",
):
    _REFRESH_PRODUCER_SPLIT_SELECTOR_WRITER_RELEASE_AUTO_CACHED = True
    _REFRESH_PRODUCER_SPLIT_SELECTOR_WRITER_RELEASE_CACHED = True
elif _REFRESH_PRODUCER_SPLIT_SELECTOR_WRITER_RELEASE_MODE_CACHED in (
    "1",
    "true",
    "on",
    "yes",
    "force",
):
    _REFRESH_PRODUCER_SPLIT_SELECTOR_WRITER_RELEASE_AUTO_CACHED = False
    _REFRESH_PRODUCER_SPLIT_SELECTOR_WRITER_RELEASE_CACHED = True
elif _REFRESH_PRODUCER_SPLIT_SELECTOR_WRITER_RELEASE_MODE_CACHED in (
    "0",
    "false",
    "off",
    "no",
):
    _REFRESH_PRODUCER_SPLIT_SELECTOR_WRITER_RELEASE_AUTO_CACHED = False
    _REFRESH_PRODUCER_SPLIT_SELECTOR_WRITER_RELEASE_CACHED = False
else:
    raise ValueError(
        "VLLM_SPARSE_REFRESH_SPLIT_SELECTOR_WRITER_RELEASE must be 0, 1, or auto"
    )
try:
    _REFRESH_PRODUCER_SPLIT_SELECTOR_WRITER_RELEASE_MAX_PER_HANDLE_CACHED: int = int(
        os.environ.get(
            "VLLM_SPARSE_REFRESH_SPLIT_SELECTOR_WRITER_RELEASE_MAX_PER_HANDLE",
            "1",
        )
        or "1"
    )
except ValueError:
    raise ValueError(
        "VLLM_SPARSE_REFRESH_SPLIT_SELECTOR_WRITER_RELEASE_MAX_PER_HANDLE "
        "must be an integer"
    )
if _REFRESH_PRODUCER_SPLIT_SELECTOR_WRITER_RELEASE_MAX_PER_HANDLE_CACHED < 1:
    raise ValueError(
        "VLLM_SPARSE_REFRESH_SPLIT_SELECTOR_WRITER_RELEASE_MAX_PER_HANDLE must be >= 1"
    )
try:
    _REFRESH_PRODUCER_SPLIT_SELECTOR_WRITER_RELEASE_MIN_LAYER_START_CACHED: int = int(
        os.environ.get(
            "VLLM_SPARSE_REFRESH_SPLIT_SELECTOR_WRITER_RELEASE_MIN_LAYER_START",
            "0",
        )
        or "0"
    )
except ValueError:
    raise ValueError(
        "VLLM_SPARSE_REFRESH_SPLIT_SELECTOR_WRITER_RELEASE_MIN_LAYER_START "
        "must be an integer"
    )
if _REFRESH_PRODUCER_SPLIT_SELECTOR_WRITER_RELEASE_MIN_LAYER_START_CACHED < 0:
    raise ValueError(
        "VLLM_SPARSE_REFRESH_SPLIT_SELECTOR_WRITER_RELEASE_MIN_LAYER_START must be >= 0"
    )
_REFRESH_GROUPED_ASYNC_ENVELOPE_CACHED = (
    os.environ.get("VLLM_SPARSE_REFRESH_GROUPED_ASYNC_ENVELOPE", "1") == "1"
)
# P3 (spec 2026-05-10 enqueue stagger): spread replay-refresh producer
# submissions across wrapper post-calls. The current default rebuckets to one
# producer group per capture chunk, which preserves capture-segment ordering
# while avoiding extra drain launches that full-KV handoff cannot hide.
# Earlier v1 spike that drained only inside the enqueue-call body has been
# replaced by the independent drain helper
# `_drain_deferred_replay_refresh_payload_groups` invoked from the wrapper
# post-call envelope (independent of trigger fire).
# Deadline-overflow always raises (no silent flush) per
# `eliminate_not_mask_hotpath`.
_REFRESH_ENQUEUE_STAGGER_CACHED = (
    os.environ.get("VLLM_SPARSE_REFRESH_ENQUEUE_STAGGER", "1") == "1"
)
_REPLAY_REFRESH_NOOP_FAST_SKIP_CACHED = (
    os.environ.get("VLLM_SPARSE_REPLAY_REFRESH_NOOP_FAST_SKIP", "1") == "1"
)
try:
    # Explicit env keeps old/manual scheduling experiments available. When unset
    # the controller derives the delay from producer group pressure: one group
    # is enqueued immediately and the deferred remainder is drained with one
    # handle of urgency before the strict deadline check.
    _REFRESH_REBUILD_MAX_DELAY_STEPS_CACHED: int = int(
        os.environ.get("VLLM_SPARSE_REFRESH_REBUILD_MAX_DELAY_STEPS", "0") or "0"
    )
except ValueError:
    _REFRESH_REBUILD_MAX_DELAY_STEPS_CACHED = 0
if _REFRESH_REBUILD_MAX_DELAY_STEPS_CACHED < 0:
    _REFRESH_REBUILD_MAX_DELAY_STEPS_CACHED = 0
# [WRITER-RELEASE-STAGGER 2026-07-06] opt-in：同一 step 内背靠背入队的多个
_REFRESH_REBUILD_CHECK_CACHED = os.environ.get("VLLM_SPARSE_REFRESH_REBUILD_CHECK", "0") == "1"
try:
    _PENDING_REBUILD_MAX_QUEUE_CACHED: int = int(
        os.environ.get("VLLM_SPARSE_PENDING_REBUILD_MAX_QUEUE", "64") or "64"
    )
except ValueError:
    _PENDING_REBUILD_MAX_QUEUE_CACHED = 64
if _PENDING_REBUILD_MAX_QUEUE_CACHED < 0:
    _PENDING_REBUILD_MAX_QUEUE_CACHED = 0

# Prefill buffer release grace period
_PREFILL_RELEASE_GRACE_STEPS: int = 2

# ---------------------------------------------------------------------------
# Step-level profiling (default off)
# ---------------------------------------------------------------------------
_STEP_PROFILE_CACHED = os.environ.get("VLLM_SPARSE_STEP_PROFILE", "0") == "1"
_STEP_PROFILE_DETAIL_CACHED = os.environ.get("VLLM_SPARSE_STEP_PROFILE_DETAIL", "0") == "1"
_STEP_PROFILE_EVERY_CACHED = int(os.environ.get("VLLM_SPARSE_STEP_PROFILE_EVERY", "50") or "50")
_STEP_PROFILE_LOG_CACHED = os.environ.get(
    "VLLM_SPARSE_STEP_PROFILE_LOG", "/tmp/vllm_sparse_step_profile.log"
)
_VALIDATE_COMPACT_META_CACHED = os.environ.get("VLLM_SPARSE_VALIDATE_COMPACT_META", "0") == "1"
_SELECTOR_FIXED_K_CACHED = os.environ.get("VLLM_SPARSE_SELECTOR_FIXED_K", "1") == "1"
_SELECTOR_SELECTED_INDICES_OUT_CACHED = os.environ.get("VLLM_SPARSE_SELECTOR_SELECTED_INDICES_OUT", "1") == "1"
_SELECTOR_PIPELINE_WORKSPACE_CACHED = os.environ.get("VLLM_SPARSE_SELECTOR_PIPELINE_WORKSPACE_RUNTIME", "1") == "1"
# fa4_selector_fixed_shape_topk: #13 breaker (a) fixed-shape topk gate (default OFF).
_SELECTOR_FIXED_SHAPE_TOPK_CACHED = os.environ.get("VLLM_SPARSE_SELECTOR_FIXED_SHAPE_TOPK", "0") == "1"
# fa4_selector_kbucket: #13 breaker (c) K-bucket gate (default OFF). Rounds the
# per-refresh narrow topk slice width UP to 256 so the topk-scan domain is
# shape-stable across refreshes (capture prerequisite). GATE-COUPLING: effective
# ONLY when the fixed-shape topk gate is also ON, because only the fixed-shape
# value-sentinel post_topk maps the extra (out-of-window) pad picks to -1; the
# OFF path would leak them into selected_indices.
_SELECTOR_KBUCKET_CACHED = (
    os.environ.get("VLLM_SPARSE_SELECTOR_KBUCKET", "0") == "1"
    and _SELECTOR_FIXED_SHAPE_TOPK_CACHED
)
# fa4_selector_topk_graph: #13 STAGE-0 captured-selector-graph consumer
# (#13a fixed-shape-topk + #13c K-bucket). Default OFF. Capture/replay the
# decode-branch selector pipeline (_pipeline_with_bounds) into a CUDA graph on
# the SYNCHRONOUS refresh path where the 4 _ensure_selector_* buffers are
# shape-keyed and data_ptr-stable. GATE-COUPLING: the captured graph can only
# hit when EVERY consumed/produced buffer is stable AND the topk-scan domain is
# shape-stable, so the flag is effective ONLY when fixed-shape topk, the
# K-bucket, the stable selected_indices_out buffer, and the stable pipeline
# workspaces are ALL on. Any of them OFF => a fresh allocation / changing scan
# domain per refresh => the graph key never hits, so we force the flag False
# and the runtime keeps the verbatim eager call (byte-identical to HEAD).
_SELECTOR_TOPK_GRAPH_CACHED = (
    os.environ.get("VLLM_SPARSE_SELECTOR_TOPK_GRAPH", "0") == "1"
    and _SELECTOR_FIXED_SHAPE_TOPK_CACHED
    and _SELECTOR_KBUCKET_CACHED
    and _SELECTOR_SELECTED_INDICES_OUT_CACHED
    and _SELECTOR_PIPELINE_WORKSPACE_CACHED
)
# SELECTOR_GRAPH_LRU (default OFF == HEAD): when ON, the captured-graph eviction
# policy switches from clear-all (drop all 8 graphs on a new key beyond the
# bound) to single-entry LRU (evict only the least-recently-used entry, keep the
# other 7). Default "0" keeps the verbatim clear-all behaviour bit-identical.
_SELECTOR_GRAPH_LRU_CACHED = (
    os.environ.get("VLLM_SPARSE_SELECTOR_GRAPH_LRU", "0") == "1"
)


def _selector_graph_lru_enabled() -> bool:
    """Whether captured-graph eviction uses single-entry LRU (default False ==
    HEAD clear-all). Dynamic/cached convention: live env read under
    ``_DYNAMIC_ENV`` (e.g. pytest), else the import-time cached bool."""
    if _DYNAMIC_ENV:
        return os.environ.get("VLLM_SPARSE_SELECTOR_GRAPH_LRU", "0") == "1"
    return _SELECTOR_GRAPH_LRU_CACHED
# ASYNC_PRODUCER_WRITER_GRAPH (task #9): capture the single fused compact WRITER launch into a
# CUDA graph and replay it on steady refresh steps (host-dispatch cut).
_ASYNC_PRODUCER_WRITER_GRAPH_CACHED = os.environ.get("VLLM_SPARSE_ASYNC_PRODUCER_WRITER_GRAPH", "1") == "1"

# ---------------------------------------------------------------------------
# Refresh profiling (default off)
# ---------------------------------------------------------------------------
_REFRESH_PROFILE_CACHED = os.environ.get("VLLM_SPARSE_REFRESH_PROFILE", "0") == "1"
_REFRESH_PROFILE_CALL_MIN_CACHED = int(os.environ.get("VLLM_SPARSE_REFRESH_PROFILE_CALL_MIN", "0") or "0")
_REFRESH_PROFILE_EVERY_CACHED = int(os.environ.get("VLLM_SPARSE_REFRESH_PROFILE_EVERY", "1") or "1")
_REFRESH_PROFILE_LOG_CACHED = os.environ.get(
    "VLLM_SPARSE_REFRESH_PROFILE_LOG", "/tmp/vllm_sparse_refresh_profile.log"
)
_REFRESH_PROFILE_DETAIL_CACHED = os.environ.get("VLLM_SPARSE_REFRESH_PROFILE_DETAIL", "0") == "1"
_ASYNC_PRODUCER_GPU_PROFILE_CACHED = (
    os.environ.get("VLLM_SPARSE_ASYNC_PRODUCER_GPU_PROFILE", "0") == "1"
)
_DEFERRED_SELECTOR_PROFILE_DETAIL_CACHED = (
    os.environ.get("VLLM_SPARSE_DEFERRED_SELECTOR_PROFILE_DETAIL", "0") == "1"
)
_REFRESH_MICRO_PROFILE_CACHED = os.environ.get("VLLM_SPARSE_REFRESH_MICRO_PROFILE", "0") == "1"

# ---------------------------------------------------------------------------
# Validation / debug tracing (default off)
# ---------------------------------------------------------------------------
_VALIDATE_META_CONTRACT_CACHED = os.environ.get("VLLM_SPARSE_VALIDATE_META_CONTRACT", "0") == "1"
_VALIDATE_LAYER_SLOT_MAP_CACHED = os.environ.get("VLLM_SPARSE_VALIDATE_LAYER_SLOT_MAP", "0") == "1"

# ---------------------------------------------------------------------------
# compact_recent routing mode (global switch)
# ---------------------------------------------------------------------------
_ATTN_MODE_CACHED: str = os.environ.get("VLLM_SPARSE_ATTN_MODE", "compact_recent")
if _ATTN_MODE_CACHED != "compact_recent":
    raise RuntimeError(
        f"VLLM_SPARSE_ATTN_MODE must be 'compact_recent'; "
        f"got {_ATTN_MODE_CACHED!r}"
    )

# ---------------------------------------------------------------------------
# Peripheral companion scaffolding (2026-04-22-compact-recent-peripheral-companion-design)
# Rev 2 (2026-04-23): default flipped. Under compact_recent, sel-page state
# production can be skipped to meet the "hot path 0 D2H/H2D/any/all/item" rule.
#
# Rev 2 rollback (2026-04-23): default flipped back to OPT-IN (env="0").
# Reason: E2E inference exposed sel-page consumers outside the production
# chain (specifically `_run_selected_no_capture_mixed_forward` outer body
# calling `consume_selected_scope(snapshot)` and requiring the wait handle to
# be ready). Turning on skip by default left those consumers waiting on an
# un-committed handle → NotReadyError in production. Opt-in via env=1 keeps
# the gate available for perf testing and contract experiments while preserving
# correctness on the default path.
# ---------------------------------------------------------------------------
_SKIP_PAGE_SPARSE_STATE_CACHED: bool = (
    os.environ.get("VLLM_SPARSE_SKIP_PAGE_SPARSE_STATE", "1") != "0"
)


def should_skip_page_sparse_state(attn_mode: str) -> bool:
    """Return True iff sel-page production chain + lengths_ok check should be skipped.

    Args:
        attn_mode: authoritative source controller.config.attn_mode. Do NOT
            pass _ATTN_MODE_CACHED — that is startup env snapshot and can
            diverge from controller.config.

    Default (rev 2 rollback): env not set or "0" → do NOT skip. env=1 opt-in
    enables skip under compact_recent.

    Testing: module-scope _SKIP_PAGE_SPARSE_STATE_CACHED requires
    importlib.reload(sparse_constants) after monkeypatch.setenv.
    """
    return _SKIP_PAGE_SPARSE_STATE_CACHED and attn_mode == "compact_recent"


# ---------------------------------------------------------------------------
# Row-wise dispatch intent enums (request-wise semantics)
# ---------------------------------------------------------------------------
_ROW_MODE_DENSE = 0
_ROW_MODE_COMPACT = 1
_PP_STEP = [0]  # ping-pong per-step tick (incremented at the post-graph point)
_RRP_GRAPH_DONE_EVT = [None]  # CUDA event: prior decode graph retired (WAR gate)
_RRP_GRAPH_DONE_EVTS = {}  # {stream_id: CUDA event} per-stream WAR gate

# Cross-step WAR fence (decode FULL cudagraph reads vs next-step RRP descriptor
# overwrite). Shared between the post-graph RECORD (patch_installer) and the next
# step's data-build WAIT (metadata_builder) so they need no common object identity.
_RRP_WAR_FENCE_EVT = [None]     # latest CUDA event recorded after a decode FULL graph
_RRP_WAR_FENCE_ARMED = [False]  # armed only in the bootstrap window
_ROW_MODE_LOG_F_PREFILL = 2
_ROW_MODE_LOG_F_REFRESH = 3

# log_f producer enums (single-source contract between decode and metadata pack)
_LOGF_PRODUCER_NONE = 0
_LOGF_PRODUCER_ATTN = 1

# ---------------------------------------------------------------------------
# Sentinel / slot constants
# ---------------------------------------------------------------------------
_FREE_SLOT_ID = "__FREE__"


def _is_free_slot_id(req_id: Optional[str]) -> bool:
    return (req_id is None) or (req_id == _FREE_SLOT_ID)
