"""
Immutable constants and cached environment variables for the sparse attention engine.

All values are read once at module import time and never mutated afterwards.
This module has no dependency on vllm_sparse_patch.py, so it can be imported
freely by both the main patch and all runtime worker modules.
"""
from __future__ import annotations

import os
import sys
from typing import Mapping, Optional

from patches.runtime_contracts import validate_capture_inflight, validate_reduce_group

__all__ = [
    "_DYNAMIC_ENV",
    "_FORCE_DENSE_CACHED",
    "_FORCE_COMPACT_OFF_CACHED",
    "_RELEASE_ON_IDLE_CACHED",
    "_REBUILD_PTRS_PINNED_CACHED",
    "_WRITER_TOKEN_TILE_DEFAULT",
    "_WRITER_TOKEN_TILE_CACHED",
    "resolve_writer_token_tile",
    "_WRITER_INPUT_BTABLE_CHECK_CACHED",
    "_SELECTOR_TRUSTED_SHAPES_CACHED",
    "_SELECTOR_FAST_SIG_CACHED",
    "_SELECTOR_CPP_PREPROC_CACHED",
    "_SELECTOR_CPP_STACK_CACHED",
    "_SELECTOR_PIPELINE_UNIFIED_CACHED",
    "_SELECTOR_LOGS_CACHE_R_CACHED",
    "_ONE_SHOT_ASYNC_BOOTSTRAP_CACHED",
    "_RRP_SAME_PAGE_SKIP_REVALIDATION_CACHED",
    "_CLEAN_METADATA_CACHED",
    "_PAGE_ADD_INCREMENTAL_CACHED",
    "_SAME_PAGE_MINIMAL_UPDATE_CACHED",
    "_SAME_PAGE_READY_EVENT_ONLY_CACHED",
    "_SAME_PAGE_MINIMAL_ASSERT_CACHED",
    "_LITE_SIG_RETURN_ASSERT_CACHED",
    "_DECODE_BOUNDS_KERNEL_CACHED",
    "_CAPTURE_CHUNK",
    "_CAPTURE_CHUNK_DEFAULT",
    "resolve_capture_chunk",
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
    "_PENDING_REBUILD_MAX_QUEUE_OVERRIDE_CACHED",
    "_PREFILL_RELEASE_GRACE_STEPS",
    "_STEP_PROFILE_CACHED",
    "_STEP_PROFILE_DETAIL_CACHED",
    "_STEP_PROFILE_EVERY_CACHED",
    "_STEP_PROFILE_LOG_CACHED",
    "_SELECTOR_FIXED_K_CACHED",
    "_SELECTOR_SELECTED_INDICES_OUT_CACHED",
    "_SELECTOR_PIPELINE_WORKSPACE_CACHED",
    "_SELECTOR_FIXED_SHAPE_TOPK_CACHED",
    "_SELECTOR_KBUCKET_CACHED",
    "_SELECTOR_TOPK_GRAPH_CACHED",
    "_SELECTED_OUT_RING_CACHED",
    "_SELECTED_OUT_RING_SLOTS_CACHED",
    "_selected_out_ring_enabled",
    "_selected_out_ring_slots",
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
# [DYNAMIC-ENV-COLLECT-FIX 2026-07-08] PYTEST_CURRENT_TEST is only set while a
# test RUNS; test modules that import this package at module level do so during
# pytest COLLECTION, where it is absent — _DYNAMIC_ENV froze False and every
# `if _DYNAMIC_ENV else _CACHED` consumer silently ignored monkeypatch.setenv
# (root cause of the standing lifecycle-admission contract red). "pytest" in
# sys.modules is already true at collection import time and never true in
# production, so it is the correct import-time signal.
_DYNAMIC_ENV = (
    (os.environ.get("PYTEST_CURRENT_TEST") is not None)
    or ("pytest" in sys.modules)
    or (os.environ.get("VLLM_SPARSE_DYNAMIC_ENV", "0") == "1")
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
# [WRITER-TILE-128 2026-07-08] gather 相1 每 token 单线程译码:tile=16 时 128
# 线程仅 16 活跃(12.5%)且 block 数 ×8,microbench(12k 档 L36 B8 H8 k1536)
# steady 全 skip 0.80→0.24ms、cold 全拷贝 5.97→2.15ms;128=kMaxSharedTileTokens
# 上限,相1 满活跃。tile 只改并行拆分不改写集合(每 (t,vi) 单写手)=逐位等价。
_WRITER_TOKEN_TILE_DEFAULT = 128


def resolve_writer_token_tile(environ: Mapping[str, str] | None = None) -> int:
    """Resolve the writer tile with the runtime's exact legacy-zero semantics."""
    source = os.environ if environ is None else environ
    return int(
        source.get(
            "VLLM_SPARSE_WRITER_TOKEN_TILE",
            str(_WRITER_TOKEN_TILE_DEFAULT),
        )
        or "0"
    )


_WRITER_TOKEN_TILE_CACHED: int = resolve_writer_token_tile()
# [WRITER-ENQUEUE-DIET 2026-07-12 ext批] 诊断档 btable 前置断言开关(默认关)。
# 旧形态=writer 每次 dispatch 热路径 os.environ.get;循 _DYNAMIC_ENV 惯例迁到
# import 期缓存(pytest 域消费点走 if _DYNAMIC_ENV else CACHED 双臂)。
_WRITER_INPUT_BTABLE_CHECK_CACHED = (
    os.environ.get("VLLM_SPARSE_WRITER_INPUT_BTABLE_CHECK", "0") == "1"
)

# ---------------------------------------------------------------------------
# Selector / rebuild experiment switches
# ---------------------------------------------------------------------------
# NOTE: VLLM_SPARSE_REBUILD_PHYSICAL_BLOCK_SORT(物理块序重排实验旋钮)已删除:
# [SELECTOR-PACK-ORDER-DETERMINISM 2026-07-11] 起 pack 序在
# batched_selection.canonicalize_selected_indices_pack_order 中无条件按逻辑
# token index 升序规范化(无旋钮默认落地,换锚件)。
_SELECTOR_TRUSTED_SHAPES_CACHED = os.environ.get("VLLM_SPARSE_SELECTOR_TRUSTED_SHAPES", "1") == "1"
_SELECTOR_FAST_SIG_CACHED = os.environ.get("VLLM_SPARSE_SELECTOR_FAST_SIG", "1") == "1"
_SELECTOR_CPP_PREPROC_CACHED = os.environ.get("VLLM_SPARSE_SELECTOR_CPP_PREPROC", "1") == "1"
_SELECTOR_CPP_STACK_CACHED = os.environ.get("VLLM_SPARSE_SELECTOR_CPP_STACK", "1") == "1"
_SELECTOR_PIPELINE_UNIFIED_CACHED = (
    os.environ.get("VLLM_SPARSE_SELECTOR_PIPELINE_UNIFIED", "1") == "1"
)
_DECODE_BOUNDS_KERNEL_CACHED = os.environ.get("VLLM_SPARSE_DECODE_BOUNDS_KERNEL", "1") == "1"
# [T3-LOG-R-WS-B 2026-07-08] default ON: the cache now borrows the selector
# pipeline's ws_b (zero extra VRAM), removing the 3x full-K log_r recompute in
# the fused kernel's passes 2/3/4. Bitwise-identical numerics. Escape =0.
_SELECTOR_LOGS_CACHE_R_CACHED = (
    os.environ.get("VLLM_SPARSE_SELECTOR_LOGS_CACHE_R", "1").strip() != "0"
)
# --- one-shot async bootstrap gate deployment flags (phase_h FIND #5) ---
_ONE_SHOT_ASYNC_BOOTSTRAP_CACHED = (
    os.environ.get("VLLM_SPARSE_ONE_SHOT_ASYNC_BOOTSTRAP", "0") == "1"
)
# --- STEADY-decode metadata hot-path deployment flags (phase_h FIND #1) ---
_RRP_SAME_PAGE_SKIP_REVALIDATION_CACHED = (
    os.environ.get("VLLM_SPARSE_RRP_SAME_PAGE_SKIP_REVALIDATION", "1") == "1"
)
# [LEG-B-PROMOTED 2026-07-08] live page-boundary writer default ON (user call:
# "能加速没有bug都可以考虑转正"). Evidence: golden anchors MATCH, 4B 12k TP1
# 8-hash bitwise IDENTICAL vs OFF, counts identical, TP2 same-card pair +1.5%.
# Escape hatch: VLLM_SPARSE_CLEAN_METADATA=0.
_CLEAN_METADATA_CACHED = os.environ.get("VLLM_SPARSE_CLEAN_METADATA", "1") == "1"
# [PAGE-ADD-PROMOTED 2026-07-08] growth page-boundary live re-lay default ON:
# a genuine recent-page growth re-lays the row's pages in place instead of the
# heavy full bind. Evidence: golden x2 MATCH, 4B 8-hash bitwise IDENTICAL,
# full_bind steps 804->288 (-64%) with remaining-bind p50 8.2ms->1.8ms; wall
# clock flat at TP1/TP2 (host-structural win: the metadata host slice shrinks,
# freeing headroom under TP>1 refresh contention). Miss still falls through to
# the full bind (fail-safe direction). Escape: VLLM_SPARSE_PAGE_ADD_INCREMENTAL=0.
_PAGE_ADD_INCREMENTAL_CACHED = (
    os.environ.get("VLLM_SPARSE_PAGE_ADD_INCREMENTAL", "1") == "1"
)
_SAME_PAGE_MINIMAL_UPDATE_CACHED = (
    os.environ.get("VLLM_SPARSE_SAME_PAGE_MINIMAL_UPDATE", "1") == "1"
)
_SAME_PAGE_READY_EVENT_ONLY_CACHED = (
    os.environ.get("VLLM_SPARSE_SAME_PAGE_READY_EVENT_ONLY", "1") == "1"
)
_SAME_PAGE_MINIMAL_ASSERT_CACHED = (
    os.environ.get("VLLM_SPARSE_SAME_PAGE_MINIMAL_ASSERT") == "1"
)
# [LITE-P0] SIG_RETURN 臂影子对拍(诊断仪器非兜底,默认关):臂命中时对拍外推
# delta vs _collect_decode_delta_packet_from_launch_plan 全量真值逐字段,不等
# 即 raise(bring-up 期用;判据绿后关闭,fail-close 由 admit 链承载)。
_LITE_SIG_RETURN_ASSERT_CACHED = (
    os.environ.get("VLLM_SPARSE_LITE_SIG_RETURN_ASSERT") == "1"
)

# ---------------------------------------------------------------------------
# Chunk-batched capture ring (memory + async overlap)
# ---------------------------------------------------------------------------
# [CHUNK18-PROMOTION 2026-07-12] Promoted after a same-GPU six-leg 4B/bs8x12k
# A/B (+1.0156% all-decode TPS, every pair positive), full c14/c18 route and
# artifact proof, and a 0.6B golden run with identical hashes/counts and both
# production gates true. The env remains the explicit rollback/experiment
# surface; invalid/non-positive values fall back to the proven default.
_CAPTURE_CHUNK_DEFAULT = 18


def resolve_capture_chunk(environ: Mapping[str, str] | None = None) -> int:
    """Resolve capture chunk, falling back for malformed/non-positive values."""
    source = os.environ if environ is None else environ
    try:
        value = int(
            source.get(
                "VLLM_SPARSE_CAPTURE_CHUNK",
                str(_CAPTURE_CHUNK_DEFAULT),
            )
            or str(_CAPTURE_CHUNK_DEFAULT)
        )
    except (TypeError, ValueError):
        return _CAPTURE_CHUNK_DEFAULT
    return value if value > 0 else _CAPTURE_CHUNK_DEFAULT


_CAPTURE_CHUNK: int = resolve_capture_chunk()

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
#   _CAPTURE_CHUNK % G == 0.
# [REDUCE-GROUP-DOMAIN 收窄 2026-07-11] 三个静默降 0 口子全部改 raise(域扫描
# A2):打错字/超域/CHUNK 不整除原先静默落 G=0——而 G=0 的 chunk-deep scratch
# 路径在 patch_installer [RING-AUDIT 2026-07-03] 注释里自认"串行调度假设
# load-bearing"(deferred producer 并发 prefill 无事件护栏),静默切入=无人知晓
# 地踩进有前提的路径;且 CHUNK=18 实验(新方向 #11)配 G=4 时 18%4!=0 恰会中招。
# 与同文件 SPLIT_* env 的 raise 风格对齐(fail-fast,坏值不再产生);G=0 仍可
# 显式配置(诊断档)。
# ---------------------------------------------------------------------------
try:
    _CAPTURE_REDUCE_GROUP: int = int(os.environ.get("VLLM_SPARSE_CAPTURE_REDUCE_GROUP", "1") or "1")
except ValueError as _rg_exc:
    raise ValueError(
        "VLLM_SPARSE_CAPTURE_REDUCE_GROUP must be an integer in (0, 1, 2, 4); "
        f"got {os.environ.get('VLLM_SPARSE_CAPTURE_REDUCE_GROUP')!r}"
    ) from _rg_exc
# [REDUCE-GROUP-SINGLE-SOURCE 2026-07-11 EXT审计·随手批] 域判定唯一真源
# = runtime_contracts.validate_reduce_group（与 validate_capture_inflight 同居
# 的纯合同件）。此前 fa3_native/ring_capture.py 携带第二份实现且语义已漂移
# （静默 return 0 vs 此处 raise）= 测试测的不是生产路径。判定逻辑与
# [REDUCE-GROUP-DOMAIN 收窄 2026-07-11] 逐位同判；此处仅补 env 语境后上抛。
try:
    _CAPTURE_REDUCE_GROUP = validate_reduce_group(_CAPTURE_REDUCE_GROUP, _CAPTURE_CHUNK)
except ValueError as _rg_domain_exc:
    raise ValueError(
        f"VLLM_SPARSE_CAPTURE_REDUCE_GROUP={_CAPTURE_REDUCE_GROUP} invalid: "
        f"{_rg_domain_exc}"
    ) from _rg_domain_exc

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
def _parse_pending_rebuild_max_queue_override(
    environ: Mapping[str, str],
) -> Optional[int]:
    """Return an explicit fail-closed queue ceiling, or runtime auto.

    The default queue capacity is derived from the live request/layer-scope
    ownership ledger.  A process-global population guess cannot represent
    different batch sizes, model depths, or producer chunking.  The optional
    override is retained only as a stricter diagnostic ceiling.
    """
    raw = environ.get("VLLM_SPARSE_PENDING_REBUILD_MAX_QUEUE", "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            "VLLM_SPARSE_PENDING_REBUILD_MAX_QUEUE must be a positive integer"
        ) from exc
    if value <= 0:
        raise ValueError(
            "VLLM_SPARSE_PENDING_REBUILD_MAX_QUEUE must be a positive integer"
        )
    return value


_PENDING_REBUILD_MAX_QUEUE_OVERRIDE_CACHED: Optional[int] = (
    _parse_pending_rebuild_max_queue_override(os.environ)
)

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
_SELECTOR_FIXED_K_CACHED = os.environ.get("VLLM_SPARSE_SELECTOR_FIXED_K", "1") == "1"
_SELECTOR_SELECTED_INDICES_OUT_CACHED = os.environ.get("VLLM_SPARSE_SELECTOR_SELECTED_INDICES_OUT", "1") == "1"
_SELECTOR_PIPELINE_WORKSPACE_CACHED = os.environ.get("VLLM_SPARSE_SELECTOR_PIPELINE_WORKSPACE_RUNTIME", "1") == "1"
# fa4_selector_fixed_shape_topk: #13 breaker (a) fixed-shape topk gate.
# [三 env 转正 2026-07-11] default ON (was OFF)。判据链在案:正确性两档全绿
# (黄金锚 MATCH tk3_golden/bprime_golden+判速 8-hash 多轮 MATCH+counts 逐位
# =基线)+性能=判速中性+长跑 512 领先 +0.31%(615.7 vs 613.8,graph 稳态纯赚,
# capture 学费已证为地板=SFI_P3_PREWARM_DESIGN_2026-07-11.md §7-§8)。
# 显式 env=0 仍为完整回退路径；配置只在此处读取一次，随后作为显式参数
# 贯穿 Python/pybind/CUDA，避免并发调用通过进程全局环境变量串扰。旧 C++
# dispatch 对非 0/1 值会 fail-fast；边界迁到 Python 后仍保留该严格语义，
# 不能把拼写错误静默解释成关闭。
_SELECTOR_FIXED_SHAPE_TOPK_RAW = os.environ.get(
    "VLLM_SPARSE_SELECTOR_FIXED_SHAPE_TOPK", "1"
)
if _SELECTOR_FIXED_SHAPE_TOPK_RAW not in ("0", "1"):
    raise ValueError(
        "VLLM_SPARSE_SELECTOR_FIXED_SHAPE_TOPK must be '0' or '1', got "
        f"{_SELECTOR_FIXED_SHAPE_TOPK_RAW!r}"
    )
_SELECTOR_FIXED_SHAPE_TOPK_CACHED = _SELECTOR_FIXED_SHAPE_TOPK_RAW == "1"
# fa4_selector_kbucket: #13 breaker (c) K-bucket gate ([三 env 转正 2026-07-11]
# default ON, was OFF). Rounds the per-refresh narrow topk slice width UP to
# 256 so the topk-scan domain is shape-stable across refreshes (capture
# prerequisite). GATE-COUPLING: effective ONLY when the fixed-shape topk gate
# is also ON, because only the fixed-shape value-sentinel post_topk maps the
# extra (out-of-window) pad picks to -1; the OFF path would leak them into
# selected_indices.
_SELECTOR_KBUCKET_CACHED = (
    os.environ.get("VLLM_SPARSE_SELECTOR_KBUCKET", "1") == "1"
    and _SELECTOR_FIXED_SHAPE_TOPK_CACHED
)
# fa4_selector_topk_graph: #13 STAGE-0 captured-selector-graph consumer
# (#13a fixed-shape-topk + #13c K-bucket). [三 env 转正 2026-07-11] default ON
# (was OFF). Capture/replay the decode-branch selector pipeline
# (_pipeline_with_bounds) into a CUDA graph on the SYNCHRONOUS refresh path
# where the 4 _ensure_selector_* buffers are shape-keyed and data_ptr-stable.
# GATE-COUPLING: the captured graph can only hit when EVERY consumed/produced
# buffer is stable AND the topk-scan domain is shape-stable, so the flag is
# effective ONLY when fixed-shape topk, the K-bucket, the stable
# selected_indices_out buffer, and the stable pipeline workspaces are ALL on.
# Any of them OFF => a fresh allocation / changing scan domain per refresh =>
# the graph key never hits, so we force the flag False and the runtime keeps
# the verbatim eager call (byte-identical to the pre-graph path).
_SELECTOR_TOPK_GRAPH_CACHED = (
    os.environ.get("VLLM_SPARSE_SELECTOR_TOPK_GRAPH", "1") == "1"
    and _SELECTOR_FIXED_SHAPE_TOPK_CACHED
    and _SELECTOR_KBUCKET_CACHED
    and _SELECTOR_SELECTED_INDICES_OUT_CACHED
    and _SELECTOR_PIPELINE_WORKSPACE_CACHED
)
# [SCOPE-OWNED GRAPH 2026-07-16] selector graph 不再使用全局/LRU/经验
# population 上限。物理 ring 槽的 structural-shape scope 只持有当前 exact
# graph，K/slice/storage 变化精确替换该 scope；热命中无 LRU touch。
# 冷 exact key 仅记一个 scope candidate；同 key 再现才 capture，异构长度流
# 因而不会为一次性 geometry 持续占用 CUDA private pool。
# [SELECTED-OUT-RING 2026-07-09] pending-path selected_indices_out stable ring
# (default ON). Replaces [SELECTED-PRIVATE-OUT 2026-07-07]'s per-run fresh
# allocation with a bounded ring of data_ptr-stable buffers guarded by per-slot
# release events (produce/consume ordering) and released at the pending
# terminal funnel (_pending_refresh_rebuild_clear). Liveness semantics are
# unchanged (a slot is never reused before its pending is terminal); what
# changes is pointer stability, which is what lets the #13 STAGE-0
# selector-topk captured graph hit on the production pending path (fresh
# per-run pointers miss the ptr-encoding key forever). Escape:
# VLLM_SPARSE_SELECTED_OUT_RING=0 restores the per-run private dict verbatim.
_SELECTED_OUT_RING_CACHED = (
    os.environ.get("VLLM_SPARSE_SELECTED_OUT_RING", "1") == "1"
)
def _parse_selected_out_ring_slots_override(
    environ: Mapping[str, str],
) -> Optional[int]:
    """Return the explicit ring-size override, or ``None`` for runtime auto."""
    raw = environ.get("VLLM_SPARSE_SELECTED_OUT_RING_SLOTS", "").strip()
    if not raw:
        return None
    try:
        slots = int(raw)
    except ValueError as exc:
        raise ValueError(
            "VLLM_SPARSE_SELECTED_OUT_RING_SLOTS must be a positive integer"
        ) from exc
    if slots <= 0:
        raise ValueError(
            "VLLM_SPARSE_SELECTED_OUT_RING_SLOTS must be a positive integer"
        )
    return slots


# None means auto: the lazy construction site derives one stable owner slot per
# actual layer chunk. This removes the former model-specific default of three
# without adding a per-run calculation. An explicit positive override remains
# available for controlled experiments.
_SELECTED_OUT_RING_SLOTS_CACHED: Optional[int] = (
    _parse_selected_out_ring_slots_override(os.environ)
)


def _selected_out_ring_enabled() -> bool:
    """Pending-path selected-out ring gate. Dynamic/cached convention: live
    env read under ``_DYNAMIC_ENV`` (e.g. pytest), else the import-time bool."""
    if _DYNAMIC_ENV:
        return os.environ.get("VLLM_SPARSE_SELECTED_OUT_RING", "1") == "1"
    return _SELECTED_OUT_RING_CACHED


def _selected_out_ring_slots(*, layer_count: int, capture_chunk: int) -> int:
    """Resolve one stable owner slot per real layer chunk, once at ring init."""
    override = (
        _parse_selected_out_ring_slots_override(os.environ)
        if _DYNAMIC_ENV
        else _SELECTED_OUT_RING_SLOTS_CACHED
    )
    if override is not None:
        return int(override)
    layers = int(layer_count)
    chunk = int(capture_chunk)
    if layers <= 0:
        raise RuntimeError(
            "selected-out ring requires registered model layers before construction"
        )
    if chunk <= 0:
        raise RuntimeError("selected-out ring requires a positive capture chunk")
    return (layers + chunk - 1) // chunk


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
