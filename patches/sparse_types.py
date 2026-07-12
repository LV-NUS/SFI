"""
patches/sparse_types.py — Pure data types for the sparse attention engine.

OWNS:
  - SparseControllerConfig: 稀疏控制器配置
  - StepMeta / StepHandle / StepContext / StepCaptureLayout: 每步共享元数据
  - LayerDecodeData / StepDecodeData / StepDispatchPlan: decode 数据结构
  - PendingRefreshRebuild: 待刷新重建记录
  - RequestTracking / LogitSpec: 请求追踪与 logit 规范
  - SelectorBatchPayload / SelectorResult: 选择器批量负载与结果
  - _RefreshProfilePending / _FlushProfileAccum: 性能分析数据类

DEPENDS_ON:
  - stdlib (dataclasses, typing)
  - torch
  - hybrid_selectors.alpha_fair_selector (AlphaFairSelectorConfig)
  - utils.sentence_triggers (RefreshTrigger, RefreshTriggerConfig)

ENTRY_POINTS: None (pure data definitions, imported by all runtime modules)
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Set, Tuple

import logging

import torch

_log = logging.getLogger(__name__)

from hybrid_selectors.alpha_fair_selector import AlphaFairSelectorConfig
from utils.sentence_triggers import RefreshTrigger, RefreshTriggerConfig

if TYPE_CHECKING:
    from patches.step_authority import StepAuthority

# ---------------------------------------------------------------------------
# Profile data types
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class _RefreshProfilePending:
    epoch: int
    chunk_id: int
    buf_id: int
    device: str
    device_index: int
    prefill_payloads: int
    refresh_payloads: int
    sentence_trigger_intents: int
    prefill_selector_runs: int
    prefill_rebuild_runs: int
    prefill_cpu_us: float
    prefill_selector_compute_cpu_us: float
    prefill_selector_post_cpu_us: float
    prefill_selector_stack_cpu_us: float
    prefill_selector_validate_cpu_us: float
    prefill_selector_key_norms_cpu_us: float
    prefill_selector_key_norms_arena_cpu_us: float
    prefill_selector_key_norms_direct_cpu_us: float
    prefill_selector_key_norms_direct_prepare_cpu_us: float
    prefill_selector_key_norms_direct_launch_cpu_us: float
    prefill_selector_key_norms_pack_cpu_us: float
    prefill_selector_select_cpu_us: float
    prefill_rebuild_cpu_us: float
    refresh_selector_cpu_us: float
    refresh_selector_apply_cpu_us: float
    refresh_selector_compute_cpu_us: float
    refresh_selector_post_cpu_us: float
    refresh_selector_stack_cpu_us: float
    refresh_selector_key_norms_cpu_us: float
    refresh_selector_key_norms_arena_cpu_us: float
    refresh_selector_key_norms_direct_cpu_us: float
    refresh_selector_key_norms_direct_prepare_cpu_us: float
    refresh_selector_key_norms_direct_launch_cpu_us: float
    refresh_selector_key_norms_pack_cpu_us: float
    refresh_rebuild_cpu_us: float
    refresh_total_cpu_us: float
    refresh_rebuild_enqueue_cpu_us: float
    refresh_rebuild_compact_cpu_us: float
    prefill_evt0: Optional[torch.cuda.Event]
    prefill_evt1: Optional[torch.cuda.Event]
    prefill_selector_evt_pairs: Tuple[Tuple[torch.cuda.Event, torch.cuda.Event], ...]
    prefill_key_norms_preproc_evt_pairs: Tuple[
        Tuple[Optional[torch.cuda.Event], Optional[torch.cuda.Event]], ...
    ]
    prefill_gather_evt_pairs: Tuple[
        Tuple[Optional[torch.cuda.Event], Optional[torch.cuda.Event]], ...
    ]
    prefill_key_norms_evt_pairs: Tuple[
        Tuple[Optional[torch.cuda.Event], Optional[torch.cuda.Event]], ...
    ]
    prefill_key_norms_h2d_evt_pairs: Tuple[
        Tuple[Optional[torch.cuda.Event], Optional[torch.cuda.Event]], ...
    ]
    prefill_key_norms_delta_evt_pairs: Tuple[
        Tuple[Optional[torch.cuda.Event], Optional[torch.cuda.Event]], ...
    ]
    prefill_key_norms_pack_evt_pairs: Tuple[
        Tuple[Optional[torch.cuda.Event], Optional[torch.cuda.Event]], ...
    ]
    prefill_log_s_evt_pairs: Tuple[
        Tuple[Optional[torch.cuda.Event], Optional[torch.cuda.Event]], ...
    ]
    prefill_log_s_triton_evt_pairs: Tuple[
        Tuple[Optional[torch.cuda.Event], Optional[torch.cuda.Event]], ...
    ]
    prefill_log_s_mask_evt_pairs: Tuple[
        Tuple[Optional[torch.cuda.Event], Optional[torch.cuda.Event]], ...
    ]
    prefill_log_s_cross_evt_pairs: Tuple[
        Tuple[Optional[torch.cuda.Event], Optional[torch.cuda.Event]], ...
    ]
    prefill_topk_evt_pairs: Tuple[
        Tuple[Optional[torch.cuda.Event], Optional[torch.cuda.Event]], ...
    ]
    prefill_preproc_evt_pairs: Tuple[
        Tuple[Optional[torch.cuda.Event], Optional[torch.cuda.Event]], ...
    ]
    prefill_seq_full_evt_pairs: Tuple[
        Tuple[Optional[torch.cuda.Event], Optional[torch.cuda.Event]], ...
    ]
    prefill_pure_preproc_evt_pairs: Tuple[
        Tuple[Optional[torch.cuda.Event], Optional[torch.cuda.Event]], ...
    ]
    prefill_selector_bounds_evt_pairs: Tuple[
        Tuple[Optional[torch.cuda.Event], Optional[torch.cuda.Event]], ...
    ]
    prefill_selector_pipeline_evt_pairs: Tuple[
        Tuple[Optional[torch.cuda.Event], Optional[torch.cuda.Event]], ...
    ]
    prefill_rebuild_evt_pairs: Tuple[Tuple[torch.cuda.Event, torch.cuda.Event], ...]
    prefill_group_evt_pairs: Tuple[
        Tuple[int, torch.cuda.Event, torch.cuda.Event], ...
    ]
    prefill_publish_cpu_us: float
    prefill_key_norms_delta_total_tokens: int
    prefill_key_norms_delta_max_tokens: int
    prefill_key_norms_delta_layers: int
    refresh_sel_evt0: Optional[torch.cuda.Event]
    refresh_sel_evt1: Optional[torch.cuda.Event]
    refresh_rebuild_evt0: Optional[torch.cuda.Event]
    refresh_rebuild_evt1: Optional[torch.cuda.Event]
    refresh_gather_evt0: Optional[torch.cuda.Event]
    refresh_gather_evt1: Optional[torch.cuda.Event]
    refresh_key_norms_preproc_evt0: Optional[torch.cuda.Event]
    refresh_key_norms_preproc_evt1: Optional[torch.cuda.Event]
    refresh_key_norms_evt0: Optional[torch.cuda.Event]
    refresh_key_norms_evt1: Optional[torch.cuda.Event]
    refresh_key_norms_h2d_evt0: Optional[torch.cuda.Event]
    refresh_key_norms_h2d_evt1: Optional[torch.cuda.Event]
    refresh_key_norms_delta_evt0: Optional[torch.cuda.Event]
    refresh_key_norms_delta_evt1: Optional[torch.cuda.Event]
    refresh_key_norms_pack_evt0: Optional[torch.cuda.Event]
    refresh_key_norms_pack_evt1: Optional[torch.cuda.Event]
    refresh_key_norms_delta_total_tokens: int
    refresh_key_norms_delta_max_tokens: int
    refresh_key_norms_delta_layers: int
    refresh_log_s_evt0: Optional[torch.cuda.Event]
    refresh_log_s_evt1: Optional[torch.cuda.Event]
    refresh_log_s_triton_evt0: Optional[torch.cuda.Event]
    refresh_log_s_triton_evt1: Optional[torch.cuda.Event]
    refresh_log_s_mask_evt0: Optional[torch.cuda.Event]
    refresh_log_s_mask_evt1: Optional[torch.cuda.Event]
    refresh_log_s_cross_evt0: Optional[torch.cuda.Event]
    refresh_log_s_cross_evt1: Optional[torch.cuda.Event]
    refresh_topk_evt0: Optional[torch.cuda.Event]
    refresh_topk_evt1: Optional[torch.cuda.Event]
    refresh_preproc_evt0: Optional[torch.cuda.Event]
    refresh_preproc_evt1: Optional[torch.cuda.Event]
    # 细粒度计时：分离seq_full准备和纯preproc_bounds开销
    refresh_seq_full_evt0: Optional[torch.cuda.Event] = None
    refresh_seq_full_evt1: Optional[torch.cuda.Event] = None
    refresh_pure_preproc_evt0: Optional[torch.cuda.Event] = None
    refresh_pure_preproc_evt1: Optional[torch.cuda.Event] = None
    refresh_selector_bounds_evt0: Optional[torch.cuda.Event] = None
    refresh_selector_bounds_evt1: Optional[torch.cuda.Event] = None
    refresh_selector_pipeline_evt0: Optional[torch.cuda.Event] = None
    refresh_selector_pipeline_evt1: Optional[torch.cuda.Event] = None
    async_producer_body_evt_pairs: Tuple[
        Tuple[Optional[torch.cuda.Event], Optional[torch.cuda.Event]], ...
    ] = ()
    async_producer_selector_evt_pairs: Tuple[
        Tuple[Optional[torch.cuda.Event], Optional[torch.cuda.Event]], ...
    ] = ()
    async_producer_writer_evt_pairs: Tuple[
        Tuple[Optional[torch.cuda.Event], Optional[torch.cuda.Event]], ...
    ] = ()
    async_producer_seq_full_evt_pairs: Tuple[
        Tuple[Optional[torch.cuda.Event], Optional[torch.cuda.Event]], ...
    ] = ()
    async_producer_pure_preproc_evt_pairs: Tuple[
        Tuple[Optional[torch.cuda.Event], Optional[torch.cuda.Event]], ...
    ] = ()
    async_producer_selector_bounds_evt_pairs: Tuple[
        Tuple[Optional[torch.cuda.Event], Optional[torch.cuda.Event]], ...
    ] = ()
    async_producer_selector_pipeline_evt_pairs: Tuple[
        Tuple[Optional[torch.cuda.Event], Optional[torch.cuda.Event]], ...
    ] = ()
    async_producer_key_norms_preproc_evt_pairs: Tuple[
        Tuple[Optional[torch.cuda.Event], Optional[torch.cuda.Event]], ...
    ] = ()
    async_producer_key_norms_evt_pairs: Tuple[
        Tuple[Optional[torch.cuda.Event], Optional[torch.cuda.Event]], ...
    ] = ()
    async_producer_key_norms_h2d_evt_pairs: Tuple[
        Tuple[Optional[torch.cuda.Event], Optional[torch.cuda.Event]], ...
    ] = ()
    async_producer_key_norms_delta_evt_pairs: Tuple[
        Tuple[Optional[torch.cuda.Event], Optional[torch.cuda.Event]], ...
    ] = ()
    async_producer_key_norms_pack_evt_pairs: Tuple[
        Tuple[Optional[torch.cuda.Event], Optional[torch.cuda.Event]], ...
    ] = ()
    async_producer_log_s_evt_pairs: Tuple[
        Tuple[Optional[torch.cuda.Event], Optional[torch.cuda.Event]], ...
    ] = ()
    async_producer_topk_evt_pairs: Tuple[
        Tuple[Optional[torch.cuda.Event], Optional[torch.cuda.Event]], ...
    ] = ()
    # rebuild meta（用于在 benchmark 侧估算带宽/字节数；不引入任何同步点）
    rebuild_head_dim: int = 0
    rebuild_kv_dtype: str = ""
    rebuild_block_size: int = 0
    rebuild_stride_tokens: int = 0
    rebuild_selected_k: int = 0
    rebuild_num_kv_heads: int = 0
    rebuild_batch_slots: int = 0
    capture_kv_len_total: int = 0
    writer_pointer_rebuild_count: int = 0
    writer_pointer_lookup_count: int = 0
    writer_cached_pointer_hit_rate: float = -1.0
    writer_cached_pointer_op_count: int = 0
    writer_vector_fallback_count: int = 0
    writer_kernel_variant: str = ""
    writer_actual_tokens: int = 0
    writer_sink_tokens: int = 0
    writer_persist_tokens: int = 0
    writer_sink_io_bytes: int = 0
    writer_persist_io_bytes: int = 0
    writer_token_tiles_estimated: int = 0
    writer_active_token_tiles_estimated: int = 0
    writer_cta_count_estimated: int = 0
    writer_active_cta_count_estimated: int = 0
    writer_tokens_per_cta: int = 0
    writer_k_read_bytes: int = 0
    writer_v_read_bytes: int = 0
    writer_k_write_bytes: int = 0
    writer_v_write_bytes: int = 0
    writer_pos_write_bytes: int = 0
    writer_total_io_bytes: int = 0
    writer_effective_io_gbps: float = -1.0
    selected_indices_materialized_bytes: int = 0
    selected_indices_io_bytes: int = 0
    selector_writer_current_path_count: int = 0
    selector_writer_boundary_cpu_us: float = 0.0
    selected_boundary_lower_bound_ms_per_group: float = -1.0
    predicted_front_early_step_improvement_ms: float = -1.0
    residual_fixed_capture_control_ms: float = -1.0
    source_ready_recorded_after_pointer_publish_count: int = 0
    lastn1_direct_count: int = 0
    gt1_reduce_count: int = 0
    gt1_scalar_fallback_count: int = 0
    refresh_rebuild_budget_before: int = -1
    refresh_rebuild_budget_after: int = -1
    refresh_rebuild_enqueued_count: int = 0
    refresh_rebuild_inline_count: int = 0
    refresh_rebuild_pending_queue_size: int = 0
    refresh_rebuild_coalesced_count: int = 0
    deadline_rebuild_drop_finished_count: int = 0
    deadline_rebuild_drain_finish_count: int = 0
    deadline_rebuild_partial_finish_count: int = 0
    deadline_rebuild_drain_submit_count: int = 0
    deadline_rebuild_drain_submit_decode_step_min: int = -1
    deadline_rebuild_drain_submit_decode_step_max: int = -1
    deadline_rebuild_drain_submit_decode_steps: Tuple[int, ...] = ()
    deadline_deferred_selector_compute_count: int = 0
    deadline_deferred_selector_compute_cpu_us_total: float = 0.0
    deadline_deferred_selector_compute_cpu_us_max: float = 0.0
    deadline_deferred_selector_inner_compute_cpu_us_total: float = 0.0
    deadline_deferred_selector_inner_compute_cpu_us_max: float = 0.0
    deadline_deferred_selector_stack_cpu_us_total: float = 0.0
    deadline_deferred_selector_stack_cpu_us_max: float = 0.0
    deadline_deferred_selector_validate_cpu_us_total: float = 0.0
    deadline_deferred_selector_validate_cpu_us_max: float = 0.0
    deadline_deferred_selector_key_norms_cpu_us_total: float = 0.0
    deadline_deferred_selector_key_norms_cpu_us_max: float = 0.0
    deadline_deferred_selector_key_norms_arena_cpu_us_total: float = 0.0
    deadline_deferred_selector_key_norms_arena_cpu_us_max: float = 0.0
    deadline_deferred_selector_key_norms_direct_cpu_us_total: float = 0.0
    deadline_deferred_selector_key_norms_direct_cpu_us_max: float = 0.0
    deadline_deferred_selector_key_norms_direct_prepare_cpu_us_total: float = 0.0
    deadline_deferred_selector_key_norms_direct_prepare_cpu_us_max: float = 0.0
    deadline_deferred_selector_key_norms_direct_launch_cpu_us_total: float = 0.0
    deadline_deferred_selector_key_norms_direct_launch_cpu_us_max: float = 0.0
    deadline_deferred_selector_key_norms_pack_cpu_us_total: float = 0.0
    deadline_deferred_selector_key_norms_pack_cpu_us_max: float = 0.0
    deadline_deferred_selector_select_cpu_us_total: float = 0.0
    deadline_deferred_selector_select_cpu_us_max: float = 0.0
    deadline_deferred_selector_post_cpu_us_total: float = 0.0
    deadline_deferred_selector_post_cpu_us_max: float = 0.0
    deadline_deferred_selector_wrapper_gap_cpu_us_total: float = 0.0
    deadline_deferred_selector_wrapper_gap_cpu_us_max: float = 0.0
    # [S7-FORENSIC 2026-07-10] off-loop selector/writer impl 内部 host 分相
    # dict 载体(键=sel_*/wr_* 段名,值=cpu_us 累计;detail 门关=恒空 dict,零税)。
    deadline_deferred_producer_detail_us: Dict[str, float] = field(
        default_factory=dict
    )
    deadline_async_producer_body_count: int = 0
    deadline_async_producer_body_cpu_us_total: float = 0.0
    deadline_async_producer_body_cpu_us_max: float = 0.0
    deadline_async_producer_selector_count: int = 0
    deadline_async_producer_selector_cpu_us_total: float = 0.0
    deadline_async_producer_selector_cpu_us_max: float = 0.0
    deadline_async_producer_key_norms_delta_count: int = 0
    deadline_async_producer_key_norms_delta_total_tokens_total: int = 0
    deadline_async_producer_key_norms_delta_max_tokens_max: int = -1
    deadline_async_producer_key_norms_delta_layers_total: int = 0
    deadline_async_producer_writer_count: int = 0
    deadline_async_producer_writer_cpu_us_total: float = 0.0
    deadline_async_producer_writer_cpu_us_max: float = 0.0
    deadline_async_producer_graph_replay_count: int = 0
    deadline_async_producer_graph_replay_cpu_us_total: float = 0.0
    deadline_async_producer_graph_replay_cpu_us_max: float = 0.0
    deadline_async_producer_graph_replay_stage_selector_inputs_count: int = 0
    deadline_async_producer_graph_replay_stage_selector_inputs_cpu_us_total: float = 0.0
    deadline_async_producer_graph_replay_stage_selector_inputs_cpu_us_max: float = 0.0
    deadline_async_producer_graph_replay_prepare_writer_count: int = 0
    deadline_async_producer_graph_replay_prepare_writer_cpu_us_total: float = 0.0
    deadline_async_producer_graph_replay_prepare_writer_cpu_us_max: float = 0.0
    deadline_async_producer_graph_replay_stage_lens_count: int = 0
    deadline_async_producer_graph_replay_stage_lens_cpu_us_total: float = 0.0
    deadline_async_producer_graph_replay_stage_lens_cpu_us_max: float = 0.0
    deadline_async_producer_graph_replay_prepare_events_count: int = 0
    deadline_async_producer_graph_replay_prepare_events_cpu_us_total: float = 0.0
    deadline_async_producer_graph_replay_prepare_events_cpu_us_max: float = 0.0
    deadline_async_producer_graph_replay_graph_count: int = 0
    deadline_async_producer_graph_replay_graph_cpu_us_total: float = 0.0
    deadline_async_producer_graph_replay_graph_cpu_us_max: float = 0.0
    deadline_async_producer_graph_capture_count: int = 0
    deadline_async_producer_graph_capture_cpu_us_total: float = 0.0
    deadline_async_producer_graph_capture_cpu_us_max: float = 0.0
    deadline_async_producer_result_precomputed_count: int = 0
    refresh_rebuild_delay_max: int = 0
    producer_work_target_layer_start: int = -1
    producer_work_target_layer_end: int = -1
    producer_work_decode_step_min: int = -1
    producer_work_decode_step_max: int = -1
    producer_work_ready_epoch: int = -1
    producer_work_deadline_epoch: int = -1
    producer_work_deadline_handle_id: int = -1
    producer_work_deadline_slack_steps: int = -1
    producer_work_can_drop: int = 0
    producer_work_can_coalesce: int = 0
    producer_work_admission_reason: str = ""
    refresh_overlap_ratio: Optional[float] = None
    refresh_overlap_new_k: Optional[int] = None
    refresh_overlap_old_k: Optional[int] = None


@dataclass(slots=True)
class _FlushProfileAccum:
    """Mutable accumulator for profiling data collected across _run_prefill/_run_refresh closures.

    Replaces 45 nonlocal variable declarations with a single object whose attributes
    can be mutated from within closures without ``nonlocal`` statements.
    Created once per ``_flush_prefill_batches`` call; zero-cost when ``do_profile=False``
    because attribute writes are guarded by the same ``if do_profile`` checks.
    """
    # CPU timings (μs)
    prefill_selector_runs: int = 0
    prefill_rebuild_runs: int = 0
    prefill_cpu_us: float = 0.0
    prefill_selector_compute_cpu_us: float = 0.0
    prefill_selector_post_cpu_us: float = 0.0
    prefill_selector_stack_cpu_us: float = 0.0
    prefill_selector_validate_cpu_us: float = 0.0
    prefill_selector_key_norms_cpu_us: float = 0.0
    prefill_selector_key_norms_arena_cpu_us: float = 0.0
    prefill_selector_key_norms_direct_cpu_us: float = 0.0
    prefill_selector_key_norms_direct_prepare_cpu_us: float = 0.0
    prefill_selector_key_norms_direct_launch_cpu_us: float = 0.0
    prefill_selector_key_norms_pack_cpu_us: float = 0.0
    prefill_selector_select_cpu_us: float = 0.0
    prefill_rebuild_cpu_us: float = 0.0
    refresh_selector_cpu_us: float = 0.0
    refresh_selector_apply_cpu_us: float = 0.0
    refresh_selector_compute_cpu_us: float = 0.0
    refresh_selector_post_cpu_us: float = 0.0
    refresh_selector_stack_cpu_us: float = 0.0
    refresh_selector_key_norms_cpu_us: float = 0.0
    refresh_selector_key_norms_arena_cpu_us: float = 0.0
    refresh_selector_key_norms_direct_cpu_us: float = 0.0
    refresh_selector_key_norms_direct_prepare_cpu_us: float = 0.0
    refresh_selector_key_norms_direct_launch_cpu_us: float = 0.0
    refresh_selector_key_norms_pack_cpu_us: float = 0.0
    refresh_rebuild_cpu_us: float = 0.0
    refresh_total_cpu_us: float = 0.0
    refresh_rebuild_enqueue_cpu_us: float = 0.0
    refresh_rebuild_compact_cpu_us: float = 0.0
    # Outer-scope GPU events (created in _flush_prefill_batches, read by closures)
    prefill_evt0: Optional[torch.cuda.Event] = None
    prefill_evt1: Optional[torch.cuda.Event] = None
    prefill_selector_evt_pairs: List[Tuple[torch.cuda.Event, torch.cuda.Event]] = field(default_factory=list)
    prefill_key_norms_preproc_evt_pairs: List[Tuple[Optional[torch.cuda.Event], Optional[torch.cuda.Event]]] = field(default_factory=list)
    prefill_gather_evt_pairs: List[Tuple[Optional[torch.cuda.Event], Optional[torch.cuda.Event]]] = field(default_factory=list)
    prefill_key_norms_evt_pairs: List[Tuple[Optional[torch.cuda.Event], Optional[torch.cuda.Event]]] = field(default_factory=list)
    prefill_key_norms_h2d_evt_pairs: List[Tuple[Optional[torch.cuda.Event], Optional[torch.cuda.Event]]] = field(default_factory=list)
    prefill_key_norms_delta_evt_pairs: List[Tuple[Optional[torch.cuda.Event], Optional[torch.cuda.Event]]] = field(default_factory=list)
    prefill_key_norms_pack_evt_pairs: List[Tuple[Optional[torch.cuda.Event], Optional[torch.cuda.Event]]] = field(default_factory=list)
    prefill_log_s_evt_pairs: List[Tuple[Optional[torch.cuda.Event], Optional[torch.cuda.Event]]] = field(default_factory=list)
    prefill_log_s_triton_evt_pairs: List[Tuple[Optional[torch.cuda.Event], Optional[torch.cuda.Event]]] = field(default_factory=list)
    prefill_log_s_mask_evt_pairs: List[Tuple[Optional[torch.cuda.Event], Optional[torch.cuda.Event]]] = field(default_factory=list)
    prefill_log_s_cross_evt_pairs: List[Tuple[Optional[torch.cuda.Event], Optional[torch.cuda.Event]]] = field(default_factory=list)
    prefill_topk_evt_pairs: List[Tuple[Optional[torch.cuda.Event], Optional[torch.cuda.Event]]] = field(default_factory=list)
    prefill_preproc_evt_pairs: List[Tuple[Optional[torch.cuda.Event], Optional[torch.cuda.Event]]] = field(default_factory=list)
    prefill_seq_full_evt_pairs: List[Tuple[Optional[torch.cuda.Event], Optional[torch.cuda.Event]]] = field(default_factory=list)
    prefill_pure_preproc_evt_pairs: List[Tuple[Optional[torch.cuda.Event], Optional[torch.cuda.Event]]] = field(default_factory=list)
    prefill_selector_bounds_evt_pairs: List[Tuple[Optional[torch.cuda.Event], Optional[torch.cuda.Event]]] = field(default_factory=list)
    prefill_selector_pipeline_evt_pairs: List[Tuple[Optional[torch.cuda.Event], Optional[torch.cuda.Event]]] = field(default_factory=list)
    prefill_rebuild_evt_pairs: List[Tuple[torch.cuda.Event, torch.cuda.Event]] = field(default_factory=list)
    prefill_group_evt_pairs: List[Tuple[int, torch.cuda.Event, torch.cuda.Event]] = field(default_factory=list)
    prefill_publish_cpu_us: float = 0.0
    prefill_key_norms_delta_total_tokens: int = 0
    prefill_key_norms_delta_max_tokens: int = -1
    prefill_key_norms_delta_layers: int = 0
    refresh_sel_evt0: Optional[torch.cuda.Event] = None
    refresh_sel_evt1: Optional[torch.cuda.Event] = None
    refresh_rebuild_evt0: Optional[torch.cuda.Event] = None
    refresh_rebuild_evt1: Optional[torch.cuda.Event] = None
    # GPU events written by _run_refresh (selector sub-stages)
    refresh_gather_evt0: Optional[torch.cuda.Event] = None
    refresh_gather_evt1: Optional[torch.cuda.Event] = None
    refresh_key_norms_preproc_evt0: Optional[torch.cuda.Event] = None
    refresh_key_norms_preproc_evt1: Optional[torch.cuda.Event] = None
    refresh_key_norms_evt0: Optional[torch.cuda.Event] = None
    refresh_key_norms_evt1: Optional[torch.cuda.Event] = None
    refresh_key_norms_h2d_evt0: Optional[torch.cuda.Event] = None
    refresh_key_norms_h2d_evt1: Optional[torch.cuda.Event] = None
    refresh_key_norms_delta_evt0: Optional[torch.cuda.Event] = None
    refresh_key_norms_delta_evt1: Optional[torch.cuda.Event] = None
    refresh_key_norms_pack_evt0: Optional[torch.cuda.Event] = None
    refresh_key_norms_pack_evt1: Optional[torch.cuda.Event] = None
    refresh_key_norms_delta_total_tokens: int = 0
    refresh_key_norms_delta_max_tokens: int = -1
    refresh_key_norms_delta_layers: int = 0
    refresh_log_s_evt0: Optional[torch.cuda.Event] = None
    refresh_log_s_evt1: Optional[torch.cuda.Event] = None
    refresh_log_s_triton_evt0: Optional[torch.cuda.Event] = None
    refresh_log_s_triton_evt1: Optional[torch.cuda.Event] = None
    refresh_log_s_mask_evt0: Optional[torch.cuda.Event] = None
    refresh_log_s_mask_evt1: Optional[torch.cuda.Event] = None
    refresh_log_s_cross_evt0: Optional[torch.cuda.Event] = None
    refresh_log_s_cross_evt1: Optional[torch.cuda.Event] = None
    refresh_topk_evt0: Optional[torch.cuda.Event] = None
    refresh_topk_evt1: Optional[torch.cuda.Event] = None
    refresh_preproc_evt0: Optional[torch.cuda.Event] = None
    refresh_preproc_evt1: Optional[torch.cuda.Event] = None
    refresh_seq_full_evt0: Optional[torch.cuda.Event] = None
    refresh_seq_full_evt1: Optional[torch.cuda.Event] = None
    refresh_pure_preproc_evt0: Optional[torch.cuda.Event] = None
    refresh_pure_preproc_evt1: Optional[torch.cuda.Event] = None
    refresh_selector_bounds_evt0: Optional[torch.cuda.Event] = None
    refresh_selector_bounds_evt1: Optional[torch.cuda.Event] = None
    refresh_selector_pipeline_evt0: Optional[torch.cuda.Event] = None
    refresh_selector_pipeline_evt1: Optional[torch.cuda.Event] = None
    async_producer_body_evt_pairs: List[
        Tuple[Optional[torch.cuda.Event], Optional[torch.cuda.Event]]
    ] = field(default_factory=list)
    async_producer_selector_evt_pairs: List[
        Tuple[Optional[torch.cuda.Event], Optional[torch.cuda.Event]]
    ] = field(default_factory=list)
    async_producer_writer_evt_pairs: List[
        Tuple[Optional[torch.cuda.Event], Optional[torch.cuda.Event]]
    ] = field(default_factory=list)
    async_producer_seq_full_evt_pairs: List[
        Tuple[Optional[torch.cuda.Event], Optional[torch.cuda.Event]]
    ] = field(default_factory=list)
    async_producer_pure_preproc_evt_pairs: List[
        Tuple[Optional[torch.cuda.Event], Optional[torch.cuda.Event]]
    ] = field(default_factory=list)
    async_producer_selector_bounds_evt_pairs: List[
        Tuple[Optional[torch.cuda.Event], Optional[torch.cuda.Event]]
    ] = field(default_factory=list)
    async_producer_selector_pipeline_evt_pairs: List[
        Tuple[Optional[torch.cuda.Event], Optional[torch.cuda.Event]]
    ] = field(default_factory=list)
    async_producer_key_norms_preproc_evt_pairs: List[
        Tuple[Optional[torch.cuda.Event], Optional[torch.cuda.Event]]
    ] = field(default_factory=list)
    async_producer_key_norms_evt_pairs: List[
        Tuple[Optional[torch.cuda.Event], Optional[torch.cuda.Event]]
    ] = field(default_factory=list)
    async_producer_key_norms_h2d_evt_pairs: List[
        Tuple[Optional[torch.cuda.Event], Optional[torch.cuda.Event]]
    ] = field(default_factory=list)
    async_producer_key_norms_delta_evt_pairs: List[
        Tuple[Optional[torch.cuda.Event], Optional[torch.cuda.Event]]
    ] = field(default_factory=list)
    async_producer_key_norms_pack_evt_pairs: List[
        Tuple[Optional[torch.cuda.Event], Optional[torch.cuda.Event]]
    ] = field(default_factory=list)
    async_producer_log_s_evt_pairs: List[
        Tuple[Optional[torch.cuda.Event], Optional[torch.cuda.Event]]
    ] = field(default_factory=list)
    async_producer_topk_evt_pairs: List[
        Tuple[Optional[torch.cuda.Event], Optional[torch.cuda.Event]]
    ] = field(default_factory=list)
    # Rebuild metadata
    rebuild_head_dim: int = 0
    rebuild_kv_dtype: str = ""
    rebuild_block_size: int = 0
    rebuild_stride_tokens: int = 0
    rebuild_selected_k: int = 0
    rebuild_num_kv_heads: int = 0
    rebuild_batch_slots: int = 0
    capture_kv_len_total: int = 0
    writer_pointer_rebuild_count: int = 0
    writer_pointer_lookup_count: int = 0
    writer_cached_pointer_hit_rate: float = -1.0
    writer_cached_pointer_op_count: int = 0
    writer_vector_fallback_count: int = 0
    writer_kernel_variant: str = ""
    writer_actual_tokens: int = 0
    writer_sink_tokens: int = 0
    writer_persist_tokens: int = 0
    writer_sink_io_bytes: int = 0
    writer_persist_io_bytes: int = 0
    writer_token_tiles_estimated: int = 0
    writer_active_token_tiles_estimated: int = 0
    writer_cta_count_estimated: int = 0
    writer_active_cta_count_estimated: int = 0
    writer_tokens_per_cta: int = 0
    writer_k_read_bytes: int = 0
    writer_v_read_bytes: int = 0
    writer_k_write_bytes: int = 0
    writer_v_write_bytes: int = 0
    writer_pos_write_bytes: int = 0
    writer_total_io_bytes: int = 0
    writer_effective_io_gbps: float = -1.0
    selected_indices_materialized_bytes: int = 0
    selected_indices_io_bytes: int = 0
    selector_writer_current_path_count: int = 0
    selector_writer_boundary_cpu_us: float = 0.0
    selected_boundary_lower_bound_ms_per_group: float = -1.0
    predicted_front_early_step_improvement_ms: float = -1.0
    residual_fixed_capture_control_ms: float = -1.0
    source_ready_recorded_after_pointer_publish_count: int = 0
    lastn1_direct_count: int = 0
    gt1_reduce_count: int = 0
    gt1_scalar_fallback_count: int = 0
    refresh_rebuild_budget_before: int = -1
    refresh_rebuild_budget_after: int = -1
    refresh_rebuild_enqueued_count: int = 0
    refresh_rebuild_inline_count: int = 0
    refresh_rebuild_pending_queue_size: int = 0
    refresh_rebuild_coalesced_count: int = 0
    deadline_rebuild_drop_finished_count: int = 0
    deadline_rebuild_drain_finish_count: int = 0
    deadline_rebuild_partial_finish_count: int = 0
    deadline_rebuild_drain_submit_count: int = 0
    deadline_rebuild_drain_submit_decode_step_min: int = -1
    deadline_rebuild_drain_submit_decode_step_max: int = -1
    deadline_rebuild_drain_submit_decode_steps: Tuple[int, ...] = ()
    refresh_rebuild_delay_max: int = 0
    producer_work_target_layer_start: int = -1
    producer_work_target_layer_end: int = -1
    producer_work_decode_step_min: int = -1
    producer_work_decode_step_max: int = -1
    producer_work_ready_epoch: int = -1
    producer_work_deadline_epoch: int = -1
    producer_work_deadline_handle_id: int = -1
    producer_work_deadline_slack_steps: int = -1
    producer_work_can_drop: int = 0
    producer_work_can_coalesce: int = 0
    producer_work_admission_reason: str = ""
    # Overlap metrics
    refresh_overlap_ratio: Optional[float] = None
    refresh_overlap_new_k: Optional[int] = None
    refresh_overlap_old_k: Optional[int] = None
    # Overlap count as 0-d int32 tensor; .item() deferred to serializer emit
    # boundary (profile_mixin / flush_worker JSON record). None when no overlap
    # sample was produced this cycle. Rev 2 (2026-04-23).
    refresh_overlap_count_tensor: Optional[torch.Tensor] = None
    # Micro-profiling (lightweight online instrumentation, independent of do_profile)
    micro_selector_ns: int = 0
    micro_rebuild_ns: int = 0


# ---------------------------------------------------------------------------
# Controller config
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class SparseControllerConfig:
    k_min: int = 32
    k_max: Optional[int] = None
    sink: int = 2
    recent: int = 512
    refresh_interval: int = 128
    # sentence trigger 合并窗口（默认 0 关闭；>0 时允许将相邻步的 sentence refresh 合并）
    refresh_coalesce_window: int = 0
    # interval 合并策略：
    # - delta1: 允许将“仅差 1 step”请求提前并入 refresh（吞吐优先）
    # - off: 严格按 request 自身 interval 触发，不提前合并
    interval_merge_policy: str = "delta1"
    # refresh 时按 layer 组做 sub-sampling（默认 1=全层；2=交错奇偶层）。
    # 仅在 decode 且全 batch bootstrap_done 时启用（避免破坏 prefill/bootstrap）。
    refresh_layer_groups: int = 1
    enabled: bool = True
    log_prefix: str = "[vllm-sparse]"
    log_interval: int = 0
    alpha_fair: AlphaFairSelectorConfig = field(default_factory=AlphaFairSelectorConfig)
    trigger: RefreshTriggerConfig = field(default_factory=RefreshTriggerConfig)
    prefill_last_n_query: Optional[int] = 16  # number of prefill queries to capture (<=0 disables capture)
    # attention routing mode: current production path is native compact_recent.
    attn_mode: Literal["compact_recent"] = "compact_recent"
    compact_page_residency_enabled: bool = False
    max_live_sparse_slots: int = 0
    compact_blocks_per_slot: int = 0
    # Graph-on bootstrap mode: run prefill selector/rebuild once. The legacy
    # baseline also disables continuous decode refresh after bootstrap; new
    # full-open gates set continuous_producer_enabled=True to keep the bootstrap
    # mechanics without using this flag as a refresh suppressor.
    one_shot_bootstrap_only: bool = False
    continuous_producer_enabled: Optional[bool] = None

    def __post_init__(self) -> None:
        allowed = ("compact_recent",)
        if self.attn_mode not in allowed:
            raise ValueError(f"attn_mode must be one of {allowed}, got {self.attn_mode!r}")
        if self.compact_page_residency_enabled:
            for name in ("max_live_sparse_slots", "compact_blocks_per_slot"):
                value = getattr(self, name)
                if type(value) is not int or value <= 0:
                    raise ValueError(f"{name} must be a positive int when compact page residency is enabled")


def continuous_producer_enabled(config: object) -> bool:
    value = getattr(config, "continuous_producer_enabled", None)
    if value is None:
        return not bool(getattr(config, "one_shot_bootstrap_only", False))
    return bool(value)


# ---------------------------------------------------------------------------
# Step-level shared types
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class StepExecHints:
    """step 级执行提示：将高频判定上提到 step 边界，避免 per-layer 重复计算。"""

    epoch: int
    batch_size: int
    wait_token_by_buf: Tuple[int, ...]


@dataclass(slots=True)
class ActiveStepSnapshot:
    """Step 边界产出的 active request 快照，供热路径只读消费。"""

    step_epoch: int
    finished_generation: int
    active_req_ids: Tuple[str, ...]
    active_row_indices: Tuple[int, ...]
    q_start_loc: Tuple[int, ...]
    num_scheduled_tokens: Tuple[int, ...]
    snapshot_signature: Tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class StepTicket:
    """_prepare_inputs 为每个 step 构建的语义票据（用于同-step 复入判定）。"""

    target_epoch: int
    req_ids_signature: int
    scheduled_signature: int
    finished_signature: int
    source_signature: int
    scheduler_token: int = field(compare=False)


@dataclass(frozen=True, slots=True)
class StepHandle:
    """StepContext 的稳定句柄（单真源主键）。"""

    handle_id: int
    epoch: int
    generation: int
    req_ids_signature: int
    num_actual_tokens: int
    refresh_plan_signature: Tuple[object, ...] = tuple()


@dataclass(frozen=True, slots=True)
class StepEnvelopeV2:
    """step 级执行单源对象（V2 硬切版本）。"""

    epoch: int
    handle_id: int
    handle_generation: int
    req_ids: Tuple[str, ...]
    slot_by_row: Tuple[int, ...]
    row_mode_by_row: Tuple[int, ...]
    refresh_signals: Tuple[int, ...]
    refresh_rows: Tuple[int, ...]
    refresh_reqs: Tuple[str, ...]
    layer_effective_refresh_by_row: Tuple[bool, ...]
    refresh_reason: str = ""
    bootstrap_done: bool = False
    plan_signature: Tuple[object, ...] = tuple()
    layer_group_active: int = -1
    layer_group_enabled: bool = False
    cache_signature: Tuple[object, ...] = tuple()


@dataclass(frozen=True, slots=True)
class StepPlan:
    """step 级不可变执行计划（单真源）。"""

    epoch: int
    step_handle_id: int
    step_handle_generation: int
    req_ids: Tuple[str, ...]
    slot_by_row: Tuple[int, ...]
    row_mode_by_row: Tuple[int, ...]
    refresh_mode_by_row: Tuple[int, ...]
    layer_effective_refresh_by_row: Tuple[bool, ...]
    logf_producer_by_row: Tuple[int, ...]
    logf_attn_rows: Tuple[int, ...]
    logf_mask_by_row: Tuple[int, ...]
    logits_last_n_by_row: Tuple[int, ...]
    logits_capacity_by_row: Tuple[int, ...]
    q_lens_by_row: Tuple[int, ...]
    context_kv_len_by_row: Tuple[int, ...]
    logf_stride_head: int
    logf_dirty_rows: Tuple[int, ...]
    plan_signature: Tuple[object, ...] = tuple()

    def with_logits(
        self,
        *,
        logits_last_n_by_row: Tuple[int, ...],
        logits_capacity_by_row: Tuple[int, ...],
    ) -> "StepPlan":
        return replace(
            self,
            logits_last_n_by_row=tuple(int(v) for v in logits_last_n_by_row),
            logits_capacity_by_row=tuple(int(v) for v in logits_capacity_by_row),
        )


class StepRefreshMode(IntEnum):
    """StepRefreshPlan 的 per-row 模式枚举（单真源）。"""

    NONE = 0
    DEFER = 1
    MUST_NOW = 2
    INFLIGHT = 3


@dataclass(frozen=True, slots=True)
class StepRefreshPlan:
    """step 级 refresh 单真源对象。"""

    epoch: int
    req_ids: Tuple[str, ...]
    mode_by_row: Tuple[int, ...]
    refresh_rows: Tuple[int, ...]
    refresh_reqs: Tuple[str, ...]
    refresh_reason: str
    bootstrap_done: bool
    plan_signature: Tuple[object, ...] = tuple()
    force_dense_while_inflight_by_row: Tuple[bool, ...] = tuple()

    @property
    def has_refresh_reqs(self) -> bool:
        return bool(self.refresh_reqs)


@dataclass(slots=True)
class StepContext:
    req_ids: Tuple[str, ...]
    num_reqs: int
    num_actual_tokens: int
    q_start_loc: Tuple[int, ...]
    q_lens: Tuple[int, ...]
    seq_lens: Tuple[int, ...]
    max_query_len: int
    max_seq_len: int
    epoch: int
    req_id_to_index: Dict[str, int]
    step_handle_id: int = -1
    step_handle_generation: int = -1
    step_handle: Optional["StepHandle"] = None
    # vLLM V1 multi_step_stream_outputs 下，decode 可能出现 q_len>1（单步生成多个 token）。
    # 为避免把 multi-step decode 误判成 prefill，这里保留 prompt/computed 计数用于阶段判定。
    prompt_lens: Optional[Tuple[int, ...]] = None
    num_computed_tokens: Optional[Tuple[int, ...]] = None
    step_envelope_v2: Optional["StepEnvelopeV2"] = None
    step_authority: Optional["StepAuthority"] = None
    step_identity_token: int = 0

    def __post_init__(self) -> None:
        if self.step_identity_token == 0 and self.epoch >= 0:
            self.step_identity_token = (
                int(self.epoch) * 1_000_000_000
                + int(self.step_handle_id) * 1_000_000
                + int(self.step_handle_generation)
            )


@dataclass(slots=True)
class StepCaptureLayout:
    """step 级 capture 打包布局（跨层共享）。"""
    epoch: int
    step_handle_id: int
    step_handle_generation: int
    slot_list: List[int]
    slot_tensor: torch.Tensor
    slot_tensor_i32: Optional[torch.Tensor]
    slot_to_capture_row: Dict[int, int]
    row_tensor: torch.Tensor
    row_tensor_i32: Optional[torch.Tensor]
    # 反向映射：batch_row -> capture_row（int32），用于在 pack_meta/patch 阶段避免 Python list→tensor
    capture_row_by_batch_row_i32: Optional[torch.Tensor]
    kv_lengths: torch.Tensor
    kv_len_per_row_i32: Optional[torch.Tensor]
    chunk_lengths: Optional[torch.Tensor]
    num_heads: int
    window: int
    kv_max: int
    capture_scores: torch.Tensor
    # last_n>1：attention kernel 会写 denom_f（fp32 [Hq]）；last_n==1 时该 buffer 可忽略。
    log_f_denoms: torch.Tensor
    # 纯 CPU 元数据：避免 per-layer 重复构造 list（尤其在 refresh/prefill capture 热路径）。
    row_list_cpu: Optional[List[int]] = None
    # seq_lens_batch：真实上下文长度（seqused_k）按 slot_list 对齐后的 GPU tensor（torch.long [slots]）。
    # 注意：这不是 capture 的 K 上限（kv_len_per_row_i32），避免在 selector 内出现“重复扣 recent”的静默错误。
    seq_lens_batch: Optional[torch.Tensor] = None
    # seq_lens_batch_i32：同一真实上下文长度的 int32 GPU carrier，供 selector bounds
    # 直接消费，避免在 selector deadline 内重复 long->int32 包装。
    seq_lens_batch_i32: Optional[torch.Tensor] = None
    # seq_lens_cpu：与 slot_list 对齐的 CPU 侧序列长度（用于 key_norms delta、topk slice 判定等）。
    seq_lens_cpu: Optional[Tuple[int, ...]] = None
    # kv_len_per_row_cpu：与 slot_list 对齐的 CPU 侧 capture K 长度，避免 capture postprocess 热路径 D2H item().
    kv_len_per_row_cpu: Optional[Tuple[int, ...]] = None
    # CPU 侧轻量缓存：避免 selector 内重复 list->tensor 物化
    slot_tensor_cpu: Optional[torch.Tensor] = None
    seq_lens_tensor_cpu: Optional[torch.Tensor] = None
    # [LAYOUT-STEP-MEMO 2026-07-06] 同 step 同 chunk 内逐层幂等重建的 memo 键
    # (epoch, handle_id, handle_generation, chunk_id)：chunk 首层走完整
    # reuse_same_step 路径后置位，同 chunk 后续层直接复用 layout（rows/cap
    # tensor/live lengths/cpu tensors/lease 全部同值重做为纯冗余）。键含
    # chunk_id——chunk2 复用 buf0 时 lease/live lengths 必须重建，跨 chunk 必失效。
    step_memo_token: Optional[Tuple[int, int, int, int]] = None
    # 正式 active mapping contract：仅覆盖当前 batch 的活动视图，供消费层只读。
    active_capture_row_by_batch_row_i32: Optional[torch.Tensor] = None
    # slot->row 映射快照：用于检测同一 epoch 内的 row 变化并重建 row_tensor/capture_row
    slot_row_map_key: Optional[Tuple[int, ...]] = None
    # live 长度小 tensor 的缓存 key；step identity 相同不代表 seqused_k 不变。
    live_lengths_key: Optional[Tuple[object, ...]] = None
    # refresh payload 对 layout 派生 view 的只读复用签名；用于同 step 同 slot 的逐层 payload build 快路径。
    refresh_payload_views_key: Optional[Tuple[object, ...]] = None
    # [PAYLOAD-VIEWS-FAST-IDENT 2026-07-09] 同 step 逐层 payload build 的 O(1) 身份短路：
    # (epoch, handle_id, handle_gen, bound_meta 引用, slot_row_map_key 引用,
    #  slots_filter 引用, slots_filter_sorted, kv_needed)。引用用 `is` 比较（持强引用，
    # 无 id() 复用风险）；epoch/handle 变化天然失效。首层全路径校验通过后置位。
    refresh_payload_views_fast_ident: Optional[Tuple[object, ...]] = None
    # capture_scores 逐层 5 维子视图复用：key=(slot_in_chunk, batch, kv_slice)，
    # value=(src 引用, view)。src 以 `is` 校验防替换；ident 置位时清理跨代残留。
    refresh_scores_subviews: Dict[Tuple[int, int, int], Tuple[torch.Tensor, torch.Tensor]] = field(
        default_factory=dict
    )
    # P0-2 FIX: buf_id 用于 cache key 区分不同 ring buffer 位置，防止内存复用时的缓存错误
    buf_id: int = -1
    lease_generation: int = 0
    # Per-layout pinned CPU/GPU staging for tiny metadata carriers.
    small_tensor_stage: Dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CaptureForwardSideOutputs:
    scratch_capture_scores: torch.Tensor
    capture_row_index_i32: torch.Tensor
    producer_rows_i32: torch.Tensor
    row_capture_last_n_i32: torch.Tensor
    row_is_prefill_producer: torch.Tensor
    active_capture_row_by_batch_row_i32: torch.Tensor
    prefill_capture_row_by_batch_row_i32: torch.Tensor
    refresh_capture_row_by_batch_row_i32: torch.Tensor
    prefill_out_capture_scores: Optional[torch.Tensor]
    prefill_out_log_f_denoms: Optional[torch.Tensor]
    refresh_out_capture_scores: Optional[torch.Tensor]
    refresh_out_log_f_denoms: Optional[torch.Tensor]
    max_capture_k: int
    max_capture_last_n: int
    producer_rows_cpu: Tuple[int, ...] = tuple()
    row_capture_last_n_cpu: Tuple[int, ...] = tuple()
    row_is_prefill_producer_cpu: Tuple[bool, ...] = tuple()
    seqused_k_cpu: Tuple[int, ...] = tuple()
    active_capture_row_by_batch_row_cpu: Tuple[int, ...] = tuple()
    prefill_out_kv_len_per_capture_row_cpu: Tuple[int, ...] = tuple()
    refresh_out_kv_len_per_capture_row_cpu: Tuple[int, ...] = tuple()


@dataclass
class CompactRecentLaunchPlan:
    """Step-level single source of truth for compact_recent kernel launch args.

    Produced once per decode step in scheduler stage
    (metadata_builder.maybe_build_step_decode_data_from_metadata_impl).
    Consumed by fa_sparse_runtime.compact_recent_dispatch.dispatch_compact_recent,
    read-only, shared across all layers within the same step.

    All 10 descriptor fields are step-level despite compact_kv arena itself
    being per-layer — the structural layout (length / offset / recent window)
    is derived from the global selector's selection counts, which are
    layer-invariant (see selection_worker.py:637 kv_lens_ref broadcast).

    See docs/superpowers/specs/2026-04-24-compact-recent-launch-plan-design.md.
    """

    # --- GPU tensors, [batch_size], int32 ---
    compact_base_block_i32: torch.Tensor
    compact_valid_tokens_i32: torch.Tensor   # kBlockN-aligned
    recent_first_i32: torch.Tensor
    recent_count_i32: torch.Tensor
    request_recent_len_i32: torch.Tensor
    launch_effective_k_len_i32: torch.Tensor

    # --- GPU tensor, [batch_size], int64 (token offsets for downstream kernels) ---
    compact_offset_tokens_i64: torch.Tensor

    # --- CPU tensor, host plan ---
    host_plan_i32: torch.Tensor              # int32, contiguous, shape [batch, N]

    # --- Step-level scalars ---
    page_size: int
    batch_size: int
    max_seqlen_k: int

    # --- Validity flag ---
    # False when rail mode is NO_COMPACT (dispatch short-circuits without
    # reading any tensor). True when plan is populated and ready to launch.
    valid: bool

    # --- Lightweight snapshot guards, CPU-only ---
    step_identity_token: int = 0
    req_set_hash: int = 0
    row_phase_hash: int = 0
    slot_signature: Tuple[int, ...] = tuple()
    use_compact_signature: Tuple[int, ...] = tuple()
    compact_meta_epoch: int = -1
    compact_valid_tokens_cpu: Tuple[int, ...] = tuple()
    compact_offset_tokens_cpu: Tuple[int, ...] = tuple()
    recent_first_cpu: Tuple[int, ...] = tuple()
    recent_count_cpu: Tuple[int, ...] = tuple()
    request_recent_len_cpu: Tuple[int, ...] = tuple()
    launch_effective_k_len_cpu: Tuple[int, ...] = tuple()
    full_recent_only: bool = False
    # True means the launch reads real compact K/V pages and must have a
    # compact arena bound. Decode-capture/full-recent rows set this False:
    # compact_valid_tokens_i32 is all-zero, so the native ABI only needs a
    # shape-compatible backing tensor and must not imply compact readiness.
    requires_compact_kv: bool = True


@dataclass(slots=True)
class StepMeta:
    """每 step 构建一次，所有层共享，存储在 VLLMSparseController 中。

    包含跨层共享的信息（常量 + GPU tensors），用于消除 per-layer 的 Python 循环。
    通过 epoch 机制与 LayerState.step_cache_* 保持同步。

    注意：slot_by_row 和 use_compact_mask 是 per-layer 的（因为 slot 分配是 per-layer），
    存储在 LayerState.step_cache_* 中，而不是这里。
    """
    epoch: int                              # 用于判断缓存是否过期
    batch_size: int
    max_batch_size: int                     # 预分配 buffer 的最大 batch 大小（避免热路径 slice）
    req_ids: Tuple[str, ...]
    req_id_to_index: Dict[str, int]         # req_id → batch 中的 row index

    # 跨层共享的数据（在 prepare_step_context 中构建）
    context_kv_len: Tuple[int, ...]         # seqused_k (CPU tuple)
    # ⚠️ 性能优化：seqused_k_gpu 只构建一次，所有 36 层复用（消除 per-layer torch.tensor()）
    seqused_k_gpu: Optional[torch.Tensor]   # [batch_size], int32, GPU tensor

    # 跨层共享的常量
    recent_cap: int
    sink_tokens: int
    block_size: int
    compact_bootstrap_threshold: int

    # request 级别的状态（跨层共享，因为 bootstrap_done 是 request 级别的）
    bootstrap_done_by_row: Tuple[bool, ...]     # 每个 request 是否已 bootstrap
    q_lens: Tuple[int, ...]                     # 每个 request 的 q_len
    # 阶段判定（跨层共享，避免 per-layer 反复扫描 prompt/computed 或 request_states）
    is_prefill_by_row: Tuple[bool, ...]         # True=prefill, False=decode
    has_prefill_row: bool                       # batch 内是否存在 prefill row
    has_decode_row: bool                        # batch 内是否存在 decode row
    prefill_rows: Tuple[int, ...]               # batch 内 prefill 行号（用于 mixed request-phase 过滤）
    is_decode_only: bool                        # 是否全部是 decode（q_len == 1）
    # 仅当 vLLM 提供 prompt_lens/num_computed_tokens 时有效：用于防止“prefill 尾 chunk q_len==1”被误判为 decode-only
    has_prefill_by_prompt: bool                 # batch 内是否存在仍处于 prefill 的 request
    short_dense_by_row: Tuple[bool, ...]        # 是否短上下文（避免每步重建 use_compact）
    # 运行时消费层唯一正式 real-KV length carrier。边界层一次性规范化，后续只读消费。
    canonical_real_kv_len_cpu: Tuple[int, ...] = tuple()
    canonical_real_kv_len_i32_gpu: Optional[torch.Tensor] = None
    compact_eligible_rows: Tuple[int, ...] = tuple()
    # decode 计划版本（step_handle 派生的 int，供 ordered reuse / step-cache 轻量门控）
    decode_plan_version: int = -1
    # request-bound recent 描述（每 step 只构建一次；block_size 确定后写入）
    recent_descriptor_block_size: int = 0
    request_recent_first_logical_page: Tuple[int, ...] = tuple()
    request_recent_page_count: Tuple[int, ...] = tuple()
    request_recent_epoch: int = -1
    request_kv_rows: Tuple[int, ...] = tuple()
    request_recent_first_logical_page_i32_gpu: Optional[torch.Tensor] = None
    request_recent_page_count_i32_gpu: Optional[torch.Tensor] = None
    request_recent_epoch_i32_gpu: Optional[torch.Tensor] = None
    request_recent_epoch_i32_gpu_value: int = -1
    request_kv_rows_i32_gpu: Optional[torch.Tensor] = None
    rrp_descriptor_epoch: int = -1
    affine_descriptor_by_row: Tuple[object, ...] = tuple()
    row_table_pages_by_row: Tuple[Tuple[int, ...], ...] = tuple()
    segment_pages_by_row: Tuple[int, ...] = tuple()
    # Per-step prologue-hoisted resolver results (2026-04-24 spec v1.3 §4.1).
    # Kept symmetric with StepBoundMeta so that tests (and any code) mocking
    # StepMeta in place of StepBoundMeta continue to satisfy the prologue
    # contract. Runtime stores on StepBoundMeta; fields here are populated by
    # patches/patch_installer._ensure_step_prologue if invoked with a StepMeta.
    step_identity_token: int = 0
    prologue_done_for_identity_token: int = -1
    prologue_owner_plan: Optional[object] = None  # MixedPrefillDecodeOwnerPlan
    prologue_rail_decision: Optional[object] = None  # CompactRecentRailDecision
    prologue_selected_row_plan: Optional[object] = None  # build_mixed_page_row_plan(...)
    # True when a decode-only not-ready step is intentionally encoded as
    # full-KV rows inside the same mixed-page/RRP carrier family.
    prologue_full_kv_handoff: bool = False
    compact_recent_launch_plan: Optional["CompactRecentLaunchPlan"] = None
    compact_mixed_page_overlay_by_layer: Dict[int, object] = field(default_factory=dict)
    resolved_row_ptr_arena_by_layer: Dict[int, object] = field(default_factory=dict)
    resolved_row_ptr_arena_key_by_layer: Dict[int, object] = field(default_factory=dict)
    compact_mixed_page_overlay_trace: Tuple[object, ...] = tuple()
    # Per-step staging for request-bound tiny metadata tensors.
    small_tensor_stage: Dict[str, object] = field(default_factory=dict)

# ---------------------------------------------------------------------------
# Decode data types
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class BoundLayerMeta:
    """Bound layer metadata shared by step-bound decode bridge."""

    req_meta_i32: torch.Tensor
    req_meta_i64: torch.Tensor
    k_compact: torch.Tensor
    v_compact: torch.Tensor
    token_positions: torch.Tensor
    compact_kv_len_max: int
    all_compact: bool = False
    hint_has_log_f: bool = False
    hint_log_f_eq1: bool = False
    hint_log_f_gt1: bool = False

@dataclass(slots=True)
class StepBoundMeta:
    """Step-bound metadata container keyed by step handle identity."""

    step_handle_id: int
    step_handle_generation: int
    epoch: int
    batch_size: int
    q_start_loc: Tuple[int, ...]
    q_lens_by_row: Tuple[int, ...]
    context_kv_len_by_row: Tuple[int, ...]
    logits_last_n_by_row: Tuple[int, ...]
    logits_capacity_by_row: Tuple[int, ...]
    logf_mask_by_row: Tuple[int, ...]
    logf_attn_rows: Tuple[int, ...]
    prefill_rows: Tuple[int, ...]
    layer_bound: Tuple[Optional[BoundLayerMeta], ...]
    # FA3 selected/no-capture compact_recent host-plan carriers.
    recent_cap: int = 0
    sink_tokens: int = 0
    canonical_real_kv_len_cpu: Tuple[int, ...] = tuple()
    canonical_real_kv_len_i32_gpu: Optional[torch.Tensor] = None
    recent_descriptor_block_size: int = 0
    request_recent_first_logical_page: Tuple[int, ...] = tuple()
    request_recent_page_count: Tuple[int, ...] = tuple()
    request_recent_epoch: int = -1
    request_kv_rows: Tuple[int, ...] = tuple()
    request_recent_first_logical_page_i32_gpu: Optional[torch.Tensor] = None
    request_recent_page_count_i32_gpu: Optional[torch.Tensor] = None
    request_recent_epoch_i32_gpu: Optional[torch.Tensor] = None
    request_recent_epoch_i32_gpu_value: int = -1
    request_kv_rows_i32_gpu: Optional[torch.Tensor] = None
    rrp_descriptor_epoch: int = -1
    affine_descriptor_by_row: Tuple[object, ...] = tuple()
    row_table_pages_by_row: Tuple[Tuple[int, ...], ...] = tuple()
    segment_pages_by_row: Tuple[int, ...] = tuple()
    # 轻量探针签名（跨进程/跨通道一致性防御，仅 O(1) 比较）
    req_set_hash: int = 0
    row_phase_hash: int = 0
    step_identity_token: int = 0
    plan_signature: Tuple[object, ...] = tuple()
    bound_meta_signature: Tuple[object, ...] = tuple()
    # ---------------------------------------------------------------
    # Per-step prologue-hoisted resolver results (2026-04-24 v1.2 spec:
    # docs/superpowers/specs/2026-04-24-per-layer-to-per-step-dispatch-hoist-design.md).
    # Populated once per step by patches/patch_installer._ensure_step_prologue;
    # layer hot path reads these fields only. Idempotency via
    # prologue_done_for_identity_token == step_identity_token (int compare, no
    # GPU state involved).
    # ---------------------------------------------------------------
    prologue_done_for_identity_token: int = -1
    prologue_owner_plan: Optional[object] = None  # MixedPrefillDecodeOwnerPlan
    prologue_rail_decision: Optional[object] = None  # CompactRecentRailDecision
    prologue_selected_row_plan: Optional[object] = None  # build_mixed_page_row_plan(...)
    # True when a decode-only not-ready step is intentionally encoded as
    # full-KV rows inside the same mixed-page/RRP carrier family.
    prologue_full_kv_handoff: bool = False
    compact_recent_launch_plan: Optional["CompactRecentLaunchPlan"] = None
    compact_mixed_page_overlay_by_layer: Dict[int, object] = field(default_factory=dict)
    resolved_row_ptr_arena_by_layer: Dict[int, object] = field(default_factory=dict)
    resolved_row_ptr_arena_key_by_layer: Dict[int, object] = field(default_factory=dict)
    compact_mixed_page_overlay_trace: Tuple[object, ...] = tuple()

    def __post_init__(self) -> None:
        self.req_set_hash = int(self.req_set_hash)
        self.row_phase_hash = int(self.row_phase_hash)
        if not isinstance(self.q_start_loc, tuple):
            self.q_start_loc = tuple(int(v) for v in self.q_start_loc)
        if not isinstance(self.q_lens_by_row, tuple):
            self.q_lens_by_row = tuple(int(v) for v in self.q_lens_by_row)
        if not isinstance(self.context_kv_len_by_row, tuple):
            self.context_kv_len_by_row = tuple(
                int(v) for v in self.context_kv_len_by_row
            )
        if not isinstance(self.logits_last_n_by_row, tuple):
            self.logits_last_n_by_row = tuple(
                int(v) for v in self.logits_last_n_by_row
            )
        if not isinstance(self.logits_capacity_by_row, tuple):
            self.logits_capacity_by_row = tuple(
                int(v) for v in self.logits_capacity_by_row
            )
        if not isinstance(self.logf_mask_by_row, tuple):
            self.logf_mask_by_row = tuple(int(v) for v in self.logf_mask_by_row)
        if not isinstance(self.logf_attn_rows, tuple):
            self.logf_attn_rows = tuple(int(v) for v in self.logf_attn_rows)
        if not isinstance(self.prefill_rows, tuple):
            self.prefill_rows = tuple(int(v) for v in self.prefill_rows)
        if not isinstance(self.canonical_real_kv_len_cpu, tuple):
            self.canonical_real_kv_len_cpu = tuple(
                int(v) for v in self.canonical_real_kv_len_cpu
            )
        if not isinstance(self.request_recent_first_logical_page, tuple):
            self.request_recent_first_logical_page = tuple(
                int(v) for v in self.request_recent_first_logical_page
            )
        if not isinstance(self.request_recent_page_count, tuple):
            self.request_recent_page_count = tuple(
                int(v) for v in self.request_recent_page_count
            )
        if not isinstance(self.request_kv_rows, tuple):
            self.request_kv_rows = tuple(int(v) for v in self.request_kv_rows)
        if not isinstance(self.plan_signature, tuple):
            self.plan_signature = tuple(self.plan_signature)
        if not isinstance(self.bound_meta_signature, tuple):
            self.bound_meta_signature = tuple(self.bound_meta_signature)
        if not isinstance(self.layer_bound, tuple):
            self.layer_bound = tuple(self.layer_bound)
        if self.step_identity_token == 0:
            self.step_identity_token = (
                int(self.epoch) * 1_000_000_000
                + int(self.step_handle_id) * 1_000_000
                + int(self.step_handle_generation)
            )


@dataclass(slots=True)
class LayerDecodeData:
    """预构建的 per-layer decode 数据，step 开始时一次性构建。

    目标：每层 dispatcher 入口只需读取此结构 + 调用 kernel，无任何 Python 处理。
    """
    # Meta tensors（已打包，可直接传给 kernel）
    req_meta_i32: torch.Tensor      # [max_batch, 7], int32 - 预分配 buffer
    req_meta_i64: torch.Tensor      # [max_batch, 4], int64 - 预分配 buffer

    # Compact KV views（从 arena 获取的 view，避免每层构造）
    k_compact: torch.Tensor         # [stride_blocks, block_size, num_kv_heads, head_dim]
    v_compact: torch.Tensor         # 同上
    token_positions: torch.Tensor   # [rows, num_kv_heads, max_blocks, block_size]
    # Layer index (for profiling/diagnostics)
    layer_index: int

    # 预计算的 hints（避免 dispatcher 内计算）
    # True 代表本层本步所有行都是 compact（compact-only kernel 可用）
    has_compact: bool
    compact_kv_len_max: int
    # 预计算 capture ring 位置，避免 decode 热路径重复 layer->chunk/buf 映射
    chunk_id: int = -1
    buf_id: int = -1
    page_sparse_enabled: bool = False
    page_sparse_page_table_i32: Optional[torch.Tensor] = None
    page_sparse_real_kv_len_i32: Optional[torch.Tensor] = None
    page_sparse_selected_seqused_k_by_head_i32: Optional[torch.Tensor] = None
    page_sparse_kv_batch_idx_i32: Optional[torch.Tensor] = None
    page_sparse_layout: int = 0
    page_sparse_applied_recent_epoch_i32: Optional[torch.Tensor] = None
    page_sparse_applied_refresh_generation_i32: Optional[torch.Tensor] = None
    page_sparse_materialize_status_i32: Optional[torch.Tensor] = None
    page_sparse_patch_status_i32: Optional[torch.Tensor] = None
    page_sparse_cached_lengths_ok: Optional[bool] = None
    page_sparse_cached_kv_batch_idx_identity: Optional[bool] = None
    page_sparse_cached_status_ok: Optional[bool] = None
    page_sparse_cached_freshness_ok: Optional[bool] = None
    page_sparse_cached_launch_ready: Optional[bool] = None
    # 可选：指向 LayerState，用于跨 step 复用 runtime pack
    state_ref: Optional["LayerState"] = None


@dataclass(slots=True)
class StepDecodeData:
    """每 step 构建一次，包含所有层的 decode 数据。

    存储在 VLLMSparseController.step_decode_data，由 step metadata builder 构建。
    """
    cache_key: Tuple[object, ...]
    # layer_index -> LayerDecodeData（索引由 controller.layer_index_by_cache_key 提供）
    layer_data: List[Optional[LayerDecodeData]]

    # 跨层共享的数据（避免每层重复传递）
    seqused_k: torch.Tensor         # [batch], int32 - 跨层共享
    batch_size: int
    num_query_heads: int
    num_queries_per_kv: int
    head_size_padded: int
    block_q: int
    total_num_q_blocks: int
    grid_bin: int
    num_seqs_bin: int
    block_size: int
    num_kv_heads: int
    head_dim: int

    # step 级预计算（避免 per-layer 重复计算）
    launch_large: bool  # (total_num_q_blocks * num_kv_heads) > 128
    grid_2d: Optional[Tuple[int, int]]  # (total_num_q_blocks, num_kv_heads) if launch_large else None

    # 用于验证的 epoch
    epoch: int
    # 轻量版本门控：与 step_handle 派生版本一致。
    decode_plan_version: int = -1
    # 预计算的 sliding_window（模型常量，避免 per-layer 重复计算）
    resolved_sliding_window: int = 0
    # chunk_id -> 全局 layer_index 列表；用于 chunk 级 live bind
    chunk_layer_indices: Optional[Dict[int, Tuple[int, ...]]] = None


@dataclass(slots=True)
class StepDispatchPlan:
    """每 step 构建一次的调度计划，避免 per-layer 入口做重复判断。"""
    cache_key: Optional[Tuple[object, ...]]
    epoch: int
    batch_size: int
    step_decode_data: Optional[StepDecodeData]
    decode_plan_version: int = -1
    # 按 layer_cache_keys 顺序对齐的 LayerDecodeData 列表；存在时可复用 ordered layer data。
    layer_data_list_ordered: Optional[List[LayerDecodeData]] = None


# ---------------------------------------------------------------------------
# Rebuild types
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class PendingRefreshRebuild:
    payloads: List[SelectorBatchPayload]
    result: Optional[SelectorResult]
    selection_phase: str
    rebuild_phase: str
    bootstrap_slots_by_layer: Optional[List[Set[int]]]
    chunk_id: int
    buf_id: int
    capture_handle_id: int
    capture_handle_generation: int
    buf_ids: Tuple[int, ...] = tuple()
    target_selected_scope_key: object | None = None
    # Debug-only mirror; commit/dispatch consistency must not depend on epoch.
    capture_epoch: int = -1
    pending_id: int = -1
    req_ids: Optional[Tuple[str, ...]] = None
    producer_kind: str = "refresh_rebuild"
    target_layer_start: int = -1
    target_layer_end: int = -1
    ready_epoch: int = -1
    deadline_epoch: int = -1
    deadline_handle_id: int = -1
    can_drop: bool = True
    can_coalesce: bool = True
    admission_reason: str = ""
    # Off-loop pre-publish carriers (see
    # docs/superpowers/specs/2026-05-10-sm80-refresh-producer-off-loop-pre-publish-design.md):
    #  - writer_done_event: producer writer done event recorded on the refresh
    #    stream after the compact writer is actually submitted; drain
    #    query/waits this event on the consumer stream.
    #  - selector_done_event: replay-refresh selector completion event. Full
    #    CUDA graph replay only needs this boundary before reusing capture-logit
    #    buffers; compact writer completion remains tracked by writer_done_event.
    #    The waited keys are per consumer stream; a single boolean would hide
    #    multi-stream replay hazards.
    #  - tracking_published: drain idempotency flag so a re-drain (e.g.
    #    supersede edge case) does not republish selection tracking twice.
    #  - producer_work_item: frozen enqueue-time descriptor for deadline,
    #    layer span, and decode-step attribution. Consumer drain reads this
    #    instead of re-deriving producer metadata from request state.
    producer_work_item: Any = None
    selector_scratch_refs: Tuple[Any, ...] = tuple()
    selector_done_event: Any = None
    selector_done_event_recorded: bool = False
    selector_done_event_waited_key: Tuple[int, int] | None = None
    writer_done_event: Any = None
    writer_done_event_waited_for_replay_key: Tuple[int, int] | None = None
    writer_release_after_handle_id: int = -1
    tracking_published: bool = False
    compact_meta_defer_publish: bool = False
    compact_meta_commit_log: Optional[List[Dict[str, object]]] = None
    # [SELECTED-OUT-RING 2026-07-09] 本 pending 的 selector run 占用的稳定环
    # 槽(spill/环关闭=None)。终局唯一漏斗 _pending_refresh_rebuild_clear
    # 释放(存 writer_done_event 消费序);释放前该槽绝不被后续 run 重用。
    selected_out_ring_slot: Any = None


def _normalize_prefill_capture_config(config: SparseControllerConfig) -> None:
    """Ensure prefill capture knobs have safe, non-negative semantics."""

    value = config.prefill_last_n_query
    if value is None or value <= 0:
        if value is None:
            _log.warning(
                "%s prefill_last_n_query unset; defaulting to 0 (capture disabled)",
                config.log_prefix,
            )
        elif value < 0:
            _log.warning(
                "%s prefill_last_n_query=%s invalid; clamping to 0",
                config.log_prefix,
                value,
            )
        config.prefill_last_n_query = 0
    elif value > 16:
        # [LAST-N-DOMAIN 收窄 2026-07-11] the log_s reduce kernels hard-cap
        # rows at kLogFPreMaxR=16 (selector_log_s_ext.py kLogFPreMaxR): any
        # row beyond 16 is SILENTLY dropped from the reduce — configured
        # last_n > 16 would quietly ignore captured data with no error (same
        # silent-domain-overflow class as the capture depth-3 case). Fail
        # fast at config normalization so the bad value never reaches the
        # kernel; raise (not clamp) per 无 fallback 纪律 — a clamp would
        # silently change requested semantics.
        raise ValueError(
            f"{config.log_prefix} prefill_last_n_query={value} exceeds the "
            "log_s reduce kernel row cap (kLogFPreMaxR=16); rows beyond 16 "
            "would be silently dropped. Lower the config or generalize the "
            "kernel cap first."
        )


# ---------------------------------------------------------------------------
# Request tracking & logit spec
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class RequestTracking:
    """Lightweight per-request bookkeeping used by the sparse controller."""

    last_seq_len: int = 0
    trigger: Optional[RefreshTrigger] = None
    # refresh 计划时 latch 的 decode_step（用于异步 refresh：避免用"执行时刻"的 decode_step 更新 last_decode_refresh_step）
    scheduled_decode_refresh_step: int = -1
    # refresh 计划时 latch 的 controller.step（用于异步 refresh：避免在 refresh 完成前重复计划 interval refresh）
    scheduled_refresh_ctrl_step: int = -1
    total_prompt_tokens: int = 0
    prompt_chunk_size: int = 0
    prefill_chunks_seen: int = 0
    bootstrap_done: bool = False
    prefill_capture_ready: bool = False
    prefill_capture_last_n: int = 0
    # [CHUNKED-CAPTURE-ACCUMULATE 2026-07-06] last_n 捕获窗跨 chunk 边界时的
    # 分片累计状态（方案 B）。rows_accum/prev_capacity 含当前 capture 步的片；
    # *_prev_* 是本步开始前的视图（供 meta 物化与同步重算幂等恢复——转移是
    # 累加/覆写，重算不能二次应用）；accum_sit=本步 step_identity_token 幂等键。
    # finalize 对账：prev_rows+尾片 eff == min(last_n,total)，不齐 fail-fast。
    prefill_capture_rows_accum: int = 0
    prefill_capture_prev_capacity: int = 0
    prefill_capture_accum_prev_rows: int = 0
    prefill_capture_accum_prev_capacity: int = 0
    prefill_capture_accum_sit: int = -1
    # request 级 prefill 完成标记（用于避免 prefill 计划阶段读取 GPU tensor 触发 DtoH 同步）
    prefill_done: bool = False
    # prefill bootstrap（selector+compact rebuild）异步流水线中：已入队但尚未确认完成。
    # 仅用于阶段/门禁判定；bootstrap_done 只在确认完成后置位。
    bootstrap_pending: bool = False
    # bootstrap_pending 对应的 step_context_epoch（用于避免在同一 step 内误置位 bootstrap_done）
    bootstrap_pending_epoch: int = -1
    # request-wise completion token：仅绑定该 request 最后一次 prefill finalize 的异步完成。
    # 不能退化为全局 buf 事件，否则会被无关 request 的后续 prefill 拖慢 selected-ready。
    bootstrap_pending_events: List[object] = field(default_factory=list)
    # one-shot RRP producer readiness: CPU manifest plus one final event.
    # Kept as Any to avoid importing CUDA-facing runtime helpers from the
    # shared type module.
    producer_ready_state: Any = None
    # one-shot deferred bootstrap bridge state. Epochs are scalar generation
    # counters for the single request-scoped bootstrap job; they do not imply a
    # second compact arena or multiple active versions.
    deferred_producer_job: Any = None
    bootstrap_bridge_active: bool = False
    bridge_token_count: int = 0
    bridge_token_positions: List[int] = field(default_factory=list)
    bridge_max_tokens: int = 0
    producer_job_epoch: int = -1
    producer_launch_step: int = -1
    bridge_last_counted_epoch: int = -1
    building_compact_epoch: int = -1
    ready_compact_epoch: int = -1
    active_compact_epoch: int = -1
    bridge_graph_policy: str = ""
    bootstrap_publish_skipped_reason: str = ""
    # one-shot bridge 后的轻量 catch-up refresh；避免 compact cache 对
    # bridge 后生成 token 的更新完全依赖 sentence trigger 是否碰巧命中。
    # [CREDIT-RETIRE 2026-07-07] post_bridge_refresh_done(credit 状态机)
    # 已整机退休:其 done 标志被任何 FORCE_NOW+SENTENCE 世代完成误置,吞掉
    # 其后的 interval 到点;计时校准由 last_decode_refresh_step 推进天然覆盖。
    post_bridge_refresh_due_decode_step: int = -1
    last_refresh_step: int = -1
    # decode token计数（纯decode步，不含prompt）；首个decode token后变为>=0
    decode_step: int = -1
    # 上次decode刷新所处的decode_step
    last_decode_refresh_step: int = -1
    # token-time trigger 的轻量意图（由 planner 在 step 边界统一落票）
    trigger_intent_reason: str = "none"
    trigger_intent_decode_step: int = -1
    trigger_intent_force_now: bool = False
    # refresh lease 异常恢复意图（由 planner 在 step 边界统一落票）
    lease_rearm: bool = False
    lease_rearm_reason: str = "none"
    lease_rearm_decode_step: int = -1
    # 短上下文标记（原为动态 setattr，提升为正式字段以配合 slots=True）
    _was_short_dense: bool = False
    # [TP-DET-TRIGGER 2026-07-07] 在飞世代的读侧镜像:票在 enqueue commit 点
    # 转 consumed(决策面),但读侧闸(dense-consume 防 torn-read/short_dense
    # crossing 保护)需要 reason/policy 存续到 GPU 终局——commit 写入,
    # selector publish final(读侧终局)清除。决策路径禁止消费。
    inflight_reason_code: int = -1
    inflight_policy: int = -1
    # TP>1 sentence trigger: 已 feed 给 trigger 的 decode observed 计数
    _trig_fed: int = 0


@dataclass(slots=True)
class LogitSpec:
    """Specification for where to write attention logits for a single request."""
    base_ptr: int           # tensor.data_ptr() or pooled buffer base address
    stride_head: int        # stride between heads
    stride_token: int       # stride between tokens (kv positions)
    row_offset: int         # query dimension start row; decode=0; prefill=max(0, q_len - last_n)
    capacity: int           # writable token column count (boundary protection)


@dataclass(slots=True)
class CapturePostprocessJob:
    """Deferred FA3 capture postprocess work consumed before selector."""

    direct_capture_phase: str
    scratch_capture_scores: torch.Tensor
    producer_rows_i32: torch.Tensor
    row_capture_last_n_i32: torch.Tensor
    row_is_prefill_producer: torch.Tensor
    seqused_k: torch.Tensor
    active_capture_row_by_batch_row_i32: torch.Tensor
    prefill_out_capture_scores: Optional[torch.Tensor]
    prefill_out_log_f_denoms: Optional[torch.Tensor]
    refresh_out_capture_scores: Optional[torch.Tensor]
    refresh_out_log_f_denoms: Optional[torch.Tensor]
    prefill_out_kv_len_per_capture_row_i32: Optional[torch.Tensor] = None
    refresh_out_kv_len_per_capture_row_i32: Optional[torch.Tensor] = None
    producer_rows_cpu: Optional[Tuple[int, ...]] = None
    row_capture_last_n_cpu: Optional[Tuple[int, ...]] = None
    row_is_prefill_producer_cpu: Optional[Tuple[bool, ...]] = None
    seqused_k_cpu: Optional[Tuple[int, ...]] = None
    active_capture_row_by_batch_row_cpu: Optional[Tuple[int, ...]] = None
    prefill_out_kv_len_per_capture_row_cpu: Optional[Tuple[int, ...]] = None
    refresh_out_kv_len_per_capture_row_cpu: Optional[Tuple[int, ...]] = None
    skip_postprocess_rows_cpu: Tuple[int, ...] = tuple()
    # [CHUNKED-CAPTURE-ACCUMULATE 2026-07-06] per-batch-row 跨片累计元数据
    # （提交步冻结，deferred 执行时步态已换代不可回读 controller）：
    # prev_rows=-1 非跨片行（原路径）；>=0 为该片 reduce 的前片累计行数。
    row_capture_accum_prev_rows_cpu: Optional[Tuple[int, ...]] = None
    row_capture_accum_prev_capacity_cpu: Optional[Tuple[int, ...]] = None
    # [RING-LASTN1-DRAIN 2026-07-03] when True the postprocess run must NOT skip
    # last_n==1 rows: under the per-G scratch RING the chunk-tail flush cannot
    # read raw ring scratch (the slot is rewritten G*in_flight layers later), so
    # the dedicated lastn1 copy arm drains those rows into the phase out tensors
    # at postprocess time instead.
    drain_lastn1_rows: bool = False
    alpha: float = 0.0
    debug_epoch: int = -1
    debug_layer_index: int = -1
    ready_event: object | None = None
    completion_event: object | None = None
    job_key: object | None = None
    launched: bool = False
    completed: bool = False
    ran_postprocess: bool = False
    waited_ready_event: bool = False
    waited_completion_event: bool = False
    # [DETERMINISTIC-TAPE-WAW 2026-07-03] flush 把本 job 输出 retarget 到私有
    # tape 时挂上的"tape baseline stack 完成"事件:job 的写必须排在 stack 之后
    # (否则 stack 的 arena 旧值会覆盖 job 输出)。
    tape_stack_evt: object | None = None
    # [DETERMINISTIC-TAPE-WAW] flush 的 retarget 检查(launched?)与 deferred
    # drain 线程的 launch 并发,检查-后-行动不原子(TOCTOU);双方以此锁互斥,
    # 锁窗口为纯 host 字段操作(µs 级,bootstrap-only)。
    lifecycle_lock: object | None = None


# ---------------------------------------------------------------------------
# Selector batch types
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class SelectorBatchPayload:
    cache_key: int
    state: "LayerState"
    capture_scores: torch.Tensor
    log_f_denoms: Optional[torch.Tensor]
    kv_lengths: torch.Tensor
    seq_lens_batch: torch.Tensor
    slot_list: List[int]
    row_list: List[int]
    key_cache: torch.Tensor
    value_cache: torch.Tensor
    block_table: torch.Tensor
    bootstrap_slots: Set[int]
    # Optional production tape for last_n==1 rows when the capture kernel had
    # to write a mixed-phase batch into scratch.  The selector consumes this
    # view directly; gt1 rows continue to use capture_scores after batched
    # reduce.  This is not a base-pointer mirror.
    lastn1_capture_scores: Optional[torch.Tensor] = None
    slot_req_ids: Optional[Tuple[str, ...]] = None
    q: Optional[torch.Tensor] = None
    cu_seqlens_q: Optional[torch.Tensor] = None
    softmax_scale: float = 0.0
    softcap: float = 0.0
    window_size: Optional[Tuple[int, int]] = None
    alibi_slopes: Optional[torch.Tensor] = None
    k_descale: Optional[torch.Tensor] = None
    slot_tensor: Optional[torch.Tensor] = None
    slot_tensor_i32: Optional[torch.Tensor] = None
    row_tensor: Optional[torch.Tensor] = None
    row_tensor_i32: Optional[torch.Tensor] = None
    refresh_rows_long: Optional[torch.Tensor] = None
    refresh_block_table_sub: Optional[torch.Tensor] = None
    refresh_seq_lens_i32: Optional[torch.Tensor] = None
    kv_len_per_row_i32: Optional[torch.Tensor] = None
    layer_index: int = -1
    seq_lens_cpu: Optional[List[int]] = None
    slot_tensor_cpu: Optional[torch.Tensor] = None
    seq_lens_tensor_cpu: Optional[torch.Tensor] = None
    target_selected_scope_key: object | None = None
    selected_scope_wait_handle: object | None = None
    capture_handle_id: int = -1
    capture_handle_generation: int = -1
    # Debug-only mirror; payload ownership is keyed by capture_handle_*.
    capture_epoch: int = -1
    # True sequence lengths aligned with seq_lens_batch, pre-staged as int32 for
    # selector bounds.  This is not kv_len_per_row_i32, which may be capture-capped.
    seq_lens_batch_i32: Optional[torch.Tensor] = None
    # CPU-only producer provenance for refresh/profile gates. Selector math
    # continues to consume the tensor fields above.
    refresh_reason: str = ""
    refresh_intent_req_ids: Tuple[str, ...] = tuple()
    # Optional global layer identity used only by replay-refresh stagger
    # grouping. Selector math still uses layer_index, which is chunk-local for
    # capture-base views.
    stagger_layer_index: int = -1
    q_is_sub: bool = False
    fast_signature: Optional[Tuple[object, ...]] = None
    capture_postprocess_job: Optional[CapturePostprocessJob] = None


@dataclass(slots=True)
class SelectorResult:
    selected_indices: torch.Tensor
    head_sink: torch.Tensor
    recent_start: torch.Tensor
    kv_len_head: torch.Tensor
    allowed_lengths: torch.Tensor
    slot_list: List[int]
    selected_middle_pages: Optional[torch.Tensor] = None
    selected_middle_counts: Optional[torch.Tensor] = None
    selected_token_scores: Optional[torch.Tensor] = None
    profile_cpu_compute_us: Optional[float] = None
    profile_cpu_post_us: Optional[float] = None
    profile_cpu_stack_us: Optional[float] = None
    profile_cpu_validate_us: Optional[float] = None
    profile_cpu_key_norms_us: Optional[float] = None
    profile_cpu_key_norms_arena_us: Optional[float] = None
    profile_cpu_key_norms_direct_us: Optional[float] = None
    profile_cpu_key_norms_direct_prepare_us: Optional[float] = None
    profile_cpu_key_norms_direct_launch_us: Optional[float] = None
    profile_cpu_key_norms_pack_us: Optional[float] = None
    profile_cpu_select_us: Optional[float] = None
    profile_gather_evt0: Optional[torch.cuda.Event] = None
    profile_gather_evt1: Optional[torch.cuda.Event] = None
    profile_key_norms_preproc_evt0: Optional[torch.cuda.Event] = None
    profile_key_norms_preproc_evt1: Optional[torch.cuda.Event] = None
    profile_key_norms_evt0: Optional[torch.cuda.Event] = None
    profile_key_norms_evt1: Optional[torch.cuda.Event] = None
    profile_key_norms_h2d_evt0: Optional[torch.cuda.Event] = None
    profile_key_norms_h2d_evt1: Optional[torch.cuda.Event] = None
    profile_key_norms_delta_evt0: Optional[torch.cuda.Event] = None
    profile_key_norms_delta_evt1: Optional[torch.cuda.Event] = None
    profile_key_norms_pack_evt0: Optional[torch.cuda.Event] = None
    profile_key_norms_pack_evt1: Optional[torch.cuda.Event] = None
    profile_key_norms_delta_total_tokens: int = 0
    profile_key_norms_delta_max_tokens: int = -1
    profile_key_norms_delta_layers: int = 0
    profile_log_s_evt0: Optional[torch.cuda.Event] = None
    profile_log_s_evt1: Optional[torch.cuda.Event] = None
    profile_log_s_triton_evt0: Optional[torch.cuda.Event] = None
    profile_log_s_triton_evt1: Optional[torch.cuda.Event] = None
    profile_log_s_mask_evt0: Optional[torch.cuda.Event] = None
    profile_log_s_mask_evt1: Optional[torch.cuda.Event] = None
    profile_log_s_cross_evt0: Optional[torch.cuda.Event] = None
    profile_log_s_cross_evt1: Optional[torch.cuda.Event] = None
    profile_topk_evt0: Optional[torch.cuda.Event] = None
    profile_topk_evt1: Optional[torch.cuda.Event] = None
    profile_preproc_evt0: Optional[torch.cuda.Event] = None
    profile_preproc_evt1: Optional[torch.cuda.Event] = None
    profile_seq_full_evt0: Optional[torch.cuda.Event] = None
    profile_seq_full_evt1: Optional[torch.cuda.Event] = None
    profile_pure_preproc_evt0: Optional[torch.cuda.Event] = None
    profile_pure_preproc_evt1: Optional[torch.cuda.Event] = None
    profile_selector_bounds_evt0: Optional[torch.cuda.Event] = None
    profile_selector_bounds_evt1: Optional[torch.cuda.Event] = None
    profile_selector_pipeline_evt0: Optional[torch.cuda.Event] = None
    profile_selector_pipeline_evt1: Optional[torch.cuda.Event] = None
