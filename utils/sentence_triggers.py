"""
Refresh trigger helpers.

The default policy combines an interval-based trigger with a very cheap sentence
ending heuristic. 句末触发依赖 token ID 匹配，避免频繁 decode 原文本。

注意: DEFAULT_SINGLE_ENDERS 中的 token ID 针对 **Qwen3** tokenizer。
Qwen3 BPE 会将标点+换行合并为单 token（如 ".\n" → 624），
因此 pair_enders 留空，single_enders 同时包含纯标点和合并形式。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Qwen3 tokenizer 句末 token ID 集合
# Qwen3 BPE 会将标点+换行合并为单 token，因此需要同时包含纯标点和合并形式。
# ---------------------------------------------------------------------------
DEFAULT_SINGLE_ENDERS: Set[int] = {
    # 英文纯标点
    13,     # “.”
    30,     # “?”
    0,      # “!”
    # 中文纯标点
    1773,   # “。”
    11319,  # “？”
    6313,   # “！”
    # BPE 合并: 标点 + \n
    624,    # “.\n”
    5267,   # “?\n”
    4894,   # “!\n”
    8997,   # “。\n”
    94432,  # “？\n”
    # BPE 合并: 标点 + \n\n
    382,    # “.\n\n”
    1939,   # “?\n\n”
    2219,   # “!\n\n”
    3407,   # “。\n\n”
    17701,  # “！\n\n”
    26850,  # “？\n\n”
}

# 当前 bare 标点已在 SINGLE_ENDERS 中，以下 pair 作为未来扩展预留
# （若需精确匹配”标点+空格”而移除 bare 标点时启用）。
# 仅保留高频模式：英文标点+空格、未被 BPE 合并的 “！\n”。
DEFAULT_PAIR_ENDERS: Set[Tuple[int, int]] = {
    (13, 220),      # “. “
    (30, 220),      # “? “
    (0, 220),       # “! “
    (6313, 198),    # “！\n” (未被 BPE 合并)
}


@dataclass
class RefreshTriggerState:
    last_token_id: Optional[int] = None
    prev_token_id: Optional[int] = None
    cooldown: int = 0
    reason: Optional[str] = None
    steps_since_refresh: int = 0
    awaiting_sentence_start: bool = False


# [TP-DET-TRIGGER 2026-07-07] planner 决定论挡板与 token-time trigger 共用的
# 全局 min_gap 默认(用户设计合同:任意两次 refresh ≥ min_refresh_gap,跨
# reason)。trigger 缺席时(refresh-on 纯 interval 形态)planner 侧 fallback
# 到此值,保证挡板不因 sentence 关闭而失效。
# [MIN-GAP-24 2026-07-07 用户拍板] 16→24:句触发过密意义不大,per-req 触发
# 均值控制在 16-32 token/世代(4B 实测 gap=16 时 18-26,提到 24 把均值推入
# 区间上半,直接降 refresh 频率=速度回收的显式口径旋钮,不靠隐式状态机)。
DEFAULT_MIN_REFRESH_GAP = 24


@dataclass
class RefreshTriggerConfig:
    refresh_interval: int = 256
    enable_sentence_triggers: bool = True
    single_end_tokens: Set[int] = field(default_factory=lambda: set(DEFAULT_SINGLE_ENDERS))
    pair_end_tokens: Set[Tuple[int, int]] = field(default_factory=lambda: set(DEFAULT_PAIR_ENDERS))
    start_exclude_tokens: Set[int] = field(default_factory=lambda: {198, 271})  # Qwen3: \n=198, \n\n=271
    sentence_cooldown: int = 2  # 触发后跳过的 decode 步数，防止重复触发
    min_refresh_gap: int = DEFAULT_MIN_REFRESH_GAP  # 相邻刷新之间至少间隔的 decode 步数
    # Trigger policy ablation (Exp #8). "punctuation_tmax" = default (sentence
    # triggers + step-time interval). The others REPLACE the trigger schedule
    # (use refresh_interval as the period N): "fixed_periodic" fires at step%N==0,
    # "random_periodic" fires ~Bernoulli(1/N) with a per-request seeded RNG
    # (deterministic reruns), "disabled" fires only the bootstrap step (t==0).
    policy: str = "punctuation_tmax"
    seed: int = 42  # random_periodic determinism


class RefreshTrigger:
    def __init__(self, config: RefreshTriggerConfig):
        self.config = config
        self.state = RefreshTriggerState()
        self._rng = None  # lazily seeded per-request RNG for policy="random_periodic"

    def _check_sentence_end(self, token_id: Optional[int]) -> Optional[str]:
        if not self.config.enable_sentence_triggers or token_id is None:
            return None

        prev_token = self.state.last_token_id
        cooldown = self.state.cooldown
        triggered = False

        if cooldown == 0:
            if token_id in self.config.single_end_tokens:
                triggered = True
            elif (
                prev_token is not None
                and (prev_token, token_id) in self.config.pair_end_tokens
            ):
                triggered = True

        if triggered:
            self.state.cooldown = self.config.sentence_cooldown
            self.state.awaiting_sentence_start = True
            return "sentence_end"

        if cooldown > 0:
            self.state.cooldown = max(cooldown - 1, 0)
        return None

    def _check_sentence_start(self, token_id: Optional[int]) -> Optional[str]:
        if not self.config.enable_sentence_triggers or token_id is None:
            return None
        state = self.state
        if not state.awaiting_sentence_start:
            return None
        if token_id in self.config.single_end_tokens or token_id in self.config.start_exclude_tokens:
            return None
        state.awaiting_sentence_start = False
        state.cooldown = max(state.cooldown, self.config.sentence_cooldown)
        return "sentence_start"

    def should_refresh(
        self,
        step: int,
        cache_position: Optional[int],
        token_id: Optional[int] = None,
        extra_reasons: Optional[Iterable[str]] = None,
    ) -> Tuple[bool, str]:
        reasons = []
        state = self.state
        state.steps_since_refresh += 1

        # --- Trigger policy ablation (Exp #8) -------------------------------
        # Non-default policies REPLACE the sentence/interval schedule entirely.
        policy = getattr(self.config, "policy", "punctuation_tmax")
        if policy != "punctuation_tmax":
            n = max(1, int(self.config.refresh_interval))
            if policy == "fixed_periodic":
                fire = (int(step) % n == 0)
            elif policy == "random_periodic":
                if self._rng is None:
                    import random as _random
                    self._rng = _random.Random(int(getattr(self.config, "seed", 42)))
                fire = (self._rng.random() < (1.0 / float(n)))
            elif policy == "disabled":
                fire = (int(step) == 0)
            else:
                # [GUARD-NO-SWALLOW] 未知策略静默降级成 bootstrap-only 会让
                # 消融实验带着错拼写跑完并产出错数据;必须炸。
                raise ValueError(
                    f"unknown refresh trigger policy {policy!r}; expected one of "
                    "'punctuation_tmax', 'fixed_periodic', 'random_periodic', "
                    "'disabled'"
                )
            reason = policy if fire else "none"
            if fire:
                state.steps_since_refresh = 0
            state.reason = reason
            return bool(fire), reason
        # -------------------------------------------------------------------

        min_gap = max(0, self.config.min_refresh_gap)
        within_gap = min_gap > 0 and state.steps_since_refresh < min_gap

        # NOTE: interval 不在 token-time 检查。
        # interval 由 step-time 的 plan_refresh_requests() 独立管理
        # （基于 decode_step - last_decode_refresh >= interval）。
        # 之前 token-time 的 interval 检查会设置 cooldown=3 封锁
        # sentence_end，但 INTERVAL intent 又被调用方过滤掉、最终被
        # compact_ready 清票——形成"幽灵触发"：压制了 sentence 却
        # 不产生任何实际 refresh。

        self._check_sentence_end(token_id)

        start_reason = self._check_sentence_start(token_id)
        if start_reason is not None:
            reasons.append(start_reason)

        if extra_reasons:
            reasons.extend(extra_reasons)

        # 更新最近 token 状态
        state.prev_token_id = state.last_token_id
        state.last_token_id = token_id

        should = bool(reasons) and not within_gap
        if should:
            state.steps_since_refresh = 0
        else:
            if within_gap and reasons:
                reason = "cooldown_gap"
            else:
                reason = "none"
            state.reason = reason
            return False, reason
        state.reason = ",".join(reasons)
        return True, state.reason


__all__ = [
    "RefreshTriggerConfig",
    "RefreshTrigger",
    "RefreshTriggerState",
    "DEFAULT_SINGLE_ENDERS",
    "DEFAULT_PAIR_ENDERS",
]
