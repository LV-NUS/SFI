from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

PAGE_TABLE_LAYOUT_KV_HEAD_FLAT = 1
PAGE_SPARSE_STATUS_OK = 0
PAGE_SPARSE_STATUS_RECOVERABLE_MISS = 1


class FASparseLaunchDecision(str, Enum):
    USE_SPARSE = "use_sparse"
    FALLBACK_LAUNCH_SLICE = "fallback_launch_slice"


@dataclass(slots=True)
class RequestRecentDescriptor:
    real_kv_len: int
    page_size: int
    sink_page_slots: int
    recent_page_slots: int
    active_recent_first_logical_page: int
    active_recent_last_logical_page: int
    materialized_first_logical_page: int
    materialized_page_count: int
    epoch: int = -1


