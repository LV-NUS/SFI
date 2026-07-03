"""Compact_recent geometry helpers shared by selector/rebuild/dispatch."""

from __future__ import annotations


COMPACT_RECENT_K_BLOCK_N = 112


def compact_slot_offset_tokens(
    *,
    slot: int,
    stride_tokens: int,
    read_gen: int = 0,
    gen_stride_tokens: int = 0,
) -> int:
    """[DUAL-GEN] slot -> arena token offset(分半双代布局)。

    gen0 半区 = [0, gen_stride);gen1 半区 = [gen_stride, 2*gen_stride)。
    L1 垫层下 read_gen 恒 0 且 gen_stride_tokens 传 0,结果与历史
    `slot * stride_tokens` 逐位相同;L2 扩容后 gen_stride = slots * stride。
    """
    gen = int(read_gen)
    if gen not in (0, 1):
        raise ValueError(f"compact read_gen must be 0/1, got {read_gen}")
    if gen == 1 and int(gen_stride_tokens) <= 0:
        raise ValueError("gen1 requires positive gen_stride_tokens (dual-gen arena)")
    return gen * int(gen_stride_tokens) + int(slot) * int(stride_tokens)


def compact_write_sub_slot(
    *,
    slot: int,
    read_gen: int,
    gen_count: int,
    max_live_slots: int,
) -> int:
    """[DUAL-GEN] writer 的物理写 sub-slot(备用半区);单代=slot 逐位。

    gather kernel 以 sub_slot*stride 寻址,gen1 半区基址=max_live*stride,
    故 sub_slot = write_gen*max_live + slot(write_gen=1-read_gen)。
    """
    if int(gen_count) <= 1:
        return int(slot)
    gen = int(read_gen)
    if gen not in (0, 1):
        raise ValueError(f"compact read_gen must be 0/1, got {read_gen}")
    if int(max_live_slots) <= 0:
        raise ValueError("dual-gen write sub-slot requires positive max_live_slots")
    return (1 - gen) * int(max_live_slots) + int(slot)


def compact_recent_effective_k_head(
    *,
    k_head: int | None,
    sink_tokens: int,
    attn_mode: str,
    block_n: int = COMPACT_RECENT_K_BLOCK_N,
) -> int:
    """Return the persist budget that makes sink+persist tile-aligned.

    FA3 compact_recent consumes the compact prefix in full kBlockN tiles.  The
    user-facing k_head excludes sink tokens, so the alignment domain is
    sink+k_head rather than k_head alone.
    """
    k_head_i = 0 if k_head is None else max(0, int(k_head))
    if k_head_i <= 0:
        return 0
    if str(attn_mode) != "compact_recent":
        return k_head_i
    sink_i = max(0, int(sink_tokens))
    block_i = max(1, int(block_n))
    total = sink_i + k_head_i
    aligned_total = ((total + block_i - 1) // block_i) * block_i
    return max(0, int(aligned_total) - sink_i)
