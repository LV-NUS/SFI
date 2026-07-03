from __future__ import annotations

import torch

REQ_META_FLAG_BASE_MASK = 0xFF
REQ_META_SINK_SHIFT = 8
REQ_META_SINK_MAX = (1 << 23) - 1
REQ_META_SINK_BITS_MASK = REQ_META_SINK_MAX << REQ_META_SINK_SHIFT


def validate_sink_tokens(sink_tokens: int) -> int:
    s = int(sink_tokens)
    if s < 0 or s > REQ_META_SINK_MAX:
        raise ValueError(
            f"sink_tokens out of range: {s}, allowed=[0,{REQ_META_SINK_MAX}]"
        )
    return s


def decode_sink_from_flags(flags: int) -> int:
    return (int(flags) & REQ_META_SINK_BITS_MASK) >> REQ_META_SINK_SHIFT


def encode_sink_into_scalar_flags(*, base_flags: int, sink_tokens: int) -> int:
    s = validate_sink_tokens(sink_tokens)
    f = int(base_flags) & REQ_META_FLAG_BASE_MASK
    return int(f | (s << REQ_META_SINK_SHIFT))


