"""StepAuthority: step 级不可变权威单源。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, Tuple


@dataclass(frozen=False, slots=True)
class StepAuthority:
    """Step 级不可变权威单源。"""

    # ── 身份 ──
    epoch: int
    step_handle_id: int
    step_handle_generation: int

    # ── 批次 ──
    batch_size: int
    max_batch_size: int
    req_ids: Tuple[str, ...]
    req_id_to_index: Dict[str, int]

    # ── 行级 query/context ──
    q_lens_by_row: Tuple[int, ...]
    context_kv_len_by_row: Tuple[int, ...]
    q_start_loc: Tuple[int, ...]

    # ── 行级阶段判定 ──
    is_prefill_by_row: Tuple[bool, ...]
    has_prefill_row: bool
    has_decode_row: bool
    prefill_rows: Tuple[int, ...]
    is_decode_only: bool
    has_prefill_by_prompt: bool

    # ── 行级 bootstrap ──
    bootstrap_done_by_row: Tuple[bool, ...]
    short_dense_by_row: Tuple[bool, ...]

    # ── 行级 slot/mode ──
    slot_by_row: Tuple[int, ...]
    row_mode_by_row: Tuple[int, ...]
    refresh_mode_by_row: Tuple[int, ...]
    layer_effective_refresh_by_row: Tuple[bool, ...]

    # ── 行级 logf ──
    logf_producer_by_row: Tuple[int, ...]
    logf_attn_rows: Tuple[int, ...]
    logf_mask_by_row: Tuple[int, ...]
    logf_stride_head: int
    logf_dirty_rows: Tuple[int, ...]

    # ── 行级 logits（selector 可通过 with_logits 更新）──
    logits_last_n_by_row: Tuple[int, ...]
    logits_capacity_by_row: Tuple[int, ...]
    decode_plan_version: int = -1
    step_identity_token: int = 0
    step_fast_identity: Tuple[int, int, int, int, int, int] = ()
    has_refresh_row: bool = False

    # ── 行级 use_compact（从 row_mode_by_row 派生）──
    use_compact_by_row: Tuple[bool, ...] = ()
    slot_by_row_has_negative: bool = False

    # ── 预计算 kernel hints ──
    hint_has_log_f: bool = False
    hint_all_compact: bool = False
    hint_log_f_eq1: bool = False
    hint_log_f_gt1: bool = False
    has_compact_row: bool = False

    # ── 常量（从 controller config 传入）──
    recent_cap: int = 0
    sink_tokens: int = 0
    compact_bootstrap_threshold: int = 0

    # ── 签名 ──
    plan_signature: tuple = ()
    req_set_hash: int = 0
    row_phase_hash: int = 0
    has_request_phase_mix: bool = False
    has_request_phase_mix_i32: int = 0
    refresh_decode_count: int = 0
    refresh_non_last_n1_count: int = 0
    refresh_prefill_count: int = 0

    # ── refresh/bootstrap slot 预计算 ──
    refresh_slots: Tuple[int, ...] = ()
    bootstrap_slots: Tuple[int, ...] = ()
    refresh_capture_slot_list: Tuple[int, ...] = ()
    refresh_capture_slot_set: FrozenSet[int] = frozenset()
    consume_selected_scope_key: object | None = None
    consume_selected_scope_wait_handle: object | None = None
    target_selected_scope_key: object | None = None
    selected_scope_wait_handle: object | None = None

    # ── needs_logits/logits_rows 聚合 ──
    needs_logits_by_row: Tuple[bool, ...] = ()
    any_needs_logits: bool = False
    prefill_needs_logits: bool = False
    refresh_needs_logits: bool = False
    logits_rows: Tuple[int, ...] = ()
    logits_rows_gt1: Tuple[int, ...] = ()

    # ── dispatch 级 logf producer（pack 约束兼容）──
    dispatch_logf_producer_by_row: Tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.step_identity_token == 0:
            self.step_identity_token = (
                int(self.epoch) * 1_000_000_000
                + int(self.step_handle_id) * 1_000_000
                + int(self.step_handle_generation)
            )
        self.step_fast_identity = (
            int(self.epoch),
            int(self.step_handle_id),
            int(self.step_handle_generation),
            int(self.step_identity_token),
            int(self.req_set_hash),
            int(self.row_phase_hash),
        )

    def with_logits(
        self,
        *,
        logits_last_n_by_row: Tuple[int, ...],
        logits_capacity_by_row: Tuple[int, ...],
    ) -> "StepAuthority":
        """原地更新 canonical logits 视图。"""
        from patches.sparse_constants import _LOGF_PRODUCER_ATTN, _LOGF_PRODUCER_NONE

        new_last_n = tuple(int(v) for v in logits_last_n_by_row)
        new_capacity = tuple(int(v) for v in logits_capacity_by_row)
        use_compact = self.use_compact_by_row
        short_dense = self.short_dense_by_row
        is_prefill = self.is_prefill_by_row
        layer_refresh = self.layer_effective_refresh_by_row
        rows = min(len(new_last_n), len(new_capacity), int(self.batch_size))

        dispatch_list: list[int] = []
        needs_list: list[bool] = []
        hint_has_log_f = False
        hint_log_f_eq1 = False
        hint_log_f_gt1 = False

        for idx in range(rows):
            last_n = int(new_last_n[idx])
            compact = bool(use_compact[idx]) if idx < len(use_compact) else False
            dense_short = bool(short_dense[idx]) if idx < len(short_dense) else False
            if compact:
                producer = int(_LOGF_PRODUCER_NONE)
                needs_logits = False
            elif last_n > 0 and not dense_short:
                producer = int(_LOGF_PRODUCER_ATTN)
                needs_logits = True
            else:
                producer = int(_LOGF_PRODUCER_NONE)
                needs_logits = False
            dispatch_list.append(producer)
            needs_list.append(needs_logits)
            if producer == int(_LOGF_PRODUCER_ATTN):
                if last_n == 1:
                    hint_log_f_eq1 = True
                elif last_n > 1:
                    hint_log_f_gt1 = True
                hint_has_log_f = True

        new_needs = tuple(needs_list)
        new_dispatch = tuple(dispatch_list)
        any_needs_logits = any(new_needs)
        prefill_needs_logits = any(
            new_needs[i] and bool(is_prefill[i]) for i in range(rows)
        )
        refresh_needs_logits = any(
            new_needs[i] and bool(layer_refresh[i]) for i in range(rows)
        )
        logits_rows = tuple(
            i for i in range(rows) if new_needs[i] and int(new_last_n[i]) > 0
        )
        logits_rows_gt1 = tuple(
            i for i in logits_rows if int(new_last_n[i]) > 1
        )

        self.logits_last_n_by_row = new_last_n
        self.logits_capacity_by_row = new_capacity
        self.needs_logits_by_row = new_needs
        self.any_needs_logits = any_needs_logits
        self.prefill_needs_logits = prefill_needs_logits
        self.refresh_needs_logits = refresh_needs_logits
        self.logits_rows = logits_rows
        self.logits_rows_gt1 = logits_rows_gt1
        self.dispatch_logf_producer_by_row = new_dispatch
        self.hint_has_log_f = hint_has_log_f
        self.hint_log_f_eq1 = hint_log_f_eq1
        self.hint_log_f_gt1 = hint_log_f_gt1
        self.hint_all_compact = all(
            bool(use_compact[i]) for i in range(rows)
        ) if rows > 0 else False
        self.has_compact_row = any(
            bool(use_compact[i]) for i in range(rows)
        ) if rows > 0 else False
        return self
