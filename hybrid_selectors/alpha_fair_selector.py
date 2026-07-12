"""Alpha-fair token selector configuration.

The alpha-fair scoring pipeline itself runs in the CUDA C++ selector exts
(utils/selector_log_s_ext.py, utils/selector_soft_nms_ext.py,
utils/selector_cross_head_ext.py); the Triton reference kernels used by their
contract tests live in triton_kernel/alpha_selector_kernel.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True)
class AlphaFairSelectorConfig:
    """Hyper-parameters controlling the alpha-fair selector."""

    alpha: float = 0.5
    gamma: float = 0.5
    prior_weight_l2: float = 1.0
    prior_weight_pos: float = 0.7
    prior_pos_power: float = 1.8
    prior_pos_eta: float = 0.4
    prior_beta_theta: float = 0.6
    prior_beta_p: float = 0.9
    nms_window: int = 32
    soft_alpha: float = 1.0
    cross_head_alpha: float = 0.85
    cross_head_temperature: float = 2.0
    cross_head_power: float = 0.0
    lambda_tail_kappa: float = 0.0
    lambda_tail_pivot: float = 0.7
    k_head: Optional[int] = 1568
    selection_mode: str = "token_topk"
    lambda_clip_single: float = 0.35
    lambda_clip_multi: float = 0.25
    lambda_soft: bool = False
    query_norm_min_scale: float = 1.0
    query_norm_max_scale: float = 1.0
    eps: float = 1.0e-12

    def beta(self) -> float:
        return -math.log(max(self.prior_beta_theta, self.eps)) / max(self.prior_beta_p, self.eps)


__all__ = [
    "AlphaFairSelectorConfig",
]
