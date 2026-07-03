from __future__ import annotations

from enum import IntEnum

import torch


class RowConsumeMode(IntEnum):
    FULL = 0
    SELECTED_TABLE = 1
    COMPACT_RECENT_NATIVE = 2


ROW_CONSUME_MODE_FULL_I32 = int(RowConsumeMode.FULL)
ROW_CONSUME_MODE_SELECTED_I32 = int(RowConsumeMode.SELECTED_TABLE)
ROW_CONSUME_MODE_COMPACT_RECENT_I32 = int(RowConsumeMode.COMPACT_RECENT_NATIVE)


def uses_resolver_visible_length_tensor(row_consume_mode_i32: torch.Tensor) -> torch.Tensor:
    return row_consume_mode_i32 != ROW_CONSUME_MODE_FULL_I32


def uses_selected_table_tensor(row_consume_mode_i32: torch.Tensor) -> torch.Tensor:
    return row_consume_mode_i32 == ROW_CONSUME_MODE_SELECTED_I32
