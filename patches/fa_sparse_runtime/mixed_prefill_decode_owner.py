"""CPU-only owner matrix for mixed prefill/decode attention rows."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Sequence


class MixedPrefillDecodeOwner(Enum):
    PREFILL_NATIVE = auto()
    PREFILL_CAPTURE_MIXED_PAGE = auto()
    DECODE_COMPACT_RECENT = auto()
    DECODE_CAPTURE_COMPACT_RECENT = auto()


@dataclass(frozen=True)
class MixedPrefillDecodeOwnerPlan:
    owners: tuple[MixedPrefillDecodeOwner, ...]
    q_spans: tuple[tuple[int, int], ...]
    q_lens: tuple[int, ...]
    prefill_rows: tuple[int, ...]
    prefill_capture_rows: tuple[int, ...]
    decode_rows: tuple[int, ...]
    decode_capture_rows: tuple[int, ...]
    has_prefill_rows: bool
    has_decode_rows: bool
    has_capture_rows: bool
    has_prefill_capture_rows: bool
    has_decode_capture_rows: bool
    prefill_active_count: int
    decode_active_count: int


def _prefix_bool(values: Sequence[object], rows: int, field: str) -> tuple[bool, ...]:
    if len(values) < rows:
        raise ValueError(f"{field} coverage insufficient: {len(values)} < {rows}")
    return tuple(bool(values[idx]) for idx in range(rows))


def _prefix_int(values: Sequence[object], rows: int, field: str) -> tuple[int, ...]:
    if len(values) < rows:
        raise ValueError(f"{field} coverage insufficient: {len(values)} < {rows}")
    return tuple(int(values[idx]) for idx in range(rows))


def _resolve_q_spans(
    q_start_loc: Sequence[object],
    rows: int,
) -> tuple[tuple[int, int], ...]:
    expected = rows + 1
    if len(q_start_loc) < expected:
        raise ValueError(
            f"q_start_loc coverage insufficient: {len(q_start_loc)} < {expected}"
        )

    starts = tuple(int(q_start_loc[idx]) for idx in range(expected))
    spans: list[tuple[int, int]] = []
    for row in range(rows):
        start = starts[row]
        end = starts[row + 1]
        if end < start:
            raise ValueError("q_start_loc must be monotonically nondecreasing")
        spans.append((start, end))
    return tuple(spans)


def _resolve_q_spans_from_q_lens(
    q_lens: tuple[int, ...],
) -> tuple[tuple[int, int], ...]:
    spans: list[tuple[int, int]] = []
    start = 0
    for q_len in q_lens:
        if q_len < 0:
            raise ValueError("q_lens_by_row must be non-negative")
        end = start + q_len
        spans.append((start, end))
        start = end
    return tuple(spans)


def _resolve_q_lens(
    step_authority: object,
    rows: int,
) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...]]:
    raw_q_lens = tuple(getattr(step_authority, "q_lens_by_row", ()))
    raw_q_start_loc = tuple(getattr(step_authority, "q_start_loc", ()))
    q_start_loc_covers = _q_start_loc_covers_full_batch(raw_q_start_loc, rows)

    if len(raw_q_lens) >= rows:
        q_lens = _prefix_int(raw_q_lens, rows, "q_lens_by_row")
        if q_start_loc_covers:
            q_spans = _resolve_q_spans(raw_q_start_loc, rows)
            span_lens = tuple(end - start for start, end in q_spans)
            for row, (from_q_lens, from_spans) in enumerate(zip(q_lens, span_lens)):
                if from_q_lens != from_spans:
                    raise ValueError(
                        "q_lens_by_row mismatch with q_start_loc "
                        f"at row {row}: {from_q_lens} != {from_spans}"
                    )
            return q_lens, q_spans
        return q_lens, _resolve_q_spans_from_q_lens(q_lens)

    if q_start_loc_covers:
        q_spans = _resolve_q_spans(raw_q_start_loc, rows)
        q_lens = tuple(end - start for start, end in q_spans)
        return q_lens, q_spans

    raise ValueError(
        "q_lens_by_row coverage insufficient without full q_start_loc coverage"
    )


def _q_start_loc_covers_full_batch(q_start_loc: Sequence[object], rows: int) -> bool:
    return len(q_start_loc) >= rows + 1


def resolve_mixed_prefill_decode_owner_plan(
    step_authority: object,
) -> MixedPrefillDecodeOwnerPlan:
    """Resolve mixed row owners from StepAuthority CPU facts only."""
    raw_is_prefill = tuple(getattr(step_authority, "is_prefill_by_row", ()))
    rows = int(getattr(step_authority, "batch_size", len(raw_is_prefill)))
    if rows < 0:
        raise ValueError("batch_size must be non-negative")

    is_prefill = _prefix_bool(raw_is_prefill, rows, "is_prefill_by_row")
    needs_logits = _prefix_bool(
        tuple(getattr(step_authority, "needs_logits_by_row", ())),
        rows,
        "needs_logits_by_row",
    )
    last_n = _prefix_int(
        tuple(getattr(step_authority, "logits_last_n_by_row", ())),
        rows,
        "logits_last_n_by_row",
    )
    q_lens, q_spans = _resolve_q_lens(step_authority, rows)

    owners: list[MixedPrefillDecodeOwner] = []
    prefill_rows: list[int] = []
    prefill_capture_rows: list[int] = []
    decode_rows: list[int] = []
    decode_capture_rows: list[int] = []

    for row in range(rows):
        q_len = q_lens[row]
        if not needs_logits[row] and last_n[row] > 0:
            raise ValueError(
                f"last_n requires needs_logits at row {row}: last_n={last_n[row]}"
            )
        if needs_logits[row] and last_n[row] <= 0:
            raise ValueError(
                f"capture rows must have last_n >= 1 at row {row}: last_n={last_n[row]}"
            )

        if is_prefill[row]:
            # [CHUNKED-TAIL-QLEN1 2026-07-05] chunked prefill 的尾 chunk 可以恰剩
            # 1 token（prompt_len % chunk == 1）：分类层（step_context_worker 的
            # computed < prompt_len 与 is_decode_only 判定）正确判其为 prefill，
            # 旧断言 q_len <= 1 却要求 prefill 必须 q_len > 1——对账矛盾 fail-fast，
            # 是 chunked prefill 无法开启的唯一死门（22cc154 修的是镜像方向：
            # q_len>1 误判 decode）。放行 q_len==1：下游 last_n<=q_len 门（1>1=False）
            # 与 owner 消费方均无 q_len>=2 假设；非 chunked prefill 恒整段提交
            # q_len>=2，此分支不变——黄金路径构造性零影响。
            if q_len < 1:
                raise ValueError(
                    f"prefill rows must have q_len >= 1 at row {row}: q_len={q_len}"
                )
            prefill_rows.append(row)
            if needs_logits[row]:
                if last_n[row] > q_len:
                    raise ValueError(
                        "prefill capture last_n must be <= q_len "
                        f"at row {row}: last_n={last_n[row]}, q_len={q_len}"
                    )
                owners.append(MixedPrefillDecodeOwner.PREFILL_CAPTURE_MIXED_PAGE)
                prefill_capture_rows.append(row)
            else:
                owners.append(MixedPrefillDecodeOwner.PREFILL_NATIVE)
            continue

        if q_len != 1:
            raise ValueError(
                f"decode rows must have q_len == 1 at row {row}: q_len={q_len}"
            )
        decode_rows.append(row)
        if needs_logits[row]:
            if last_n[row] != 1:
                raise ValueError(
                    f"decode capture requires last_n == 1 at row {row}: last_n={last_n[row]}"
                )
            owners.append(MixedPrefillDecodeOwner.DECODE_CAPTURE_COMPACT_RECENT)
            decode_capture_rows.append(row)
        else:
            owners.append(MixedPrefillDecodeOwner.DECODE_COMPACT_RECENT)

    return MixedPrefillDecodeOwnerPlan(
        owners=tuple(owners),
        q_spans=q_spans,
        q_lens=q_lens,
        prefill_rows=tuple(prefill_rows),
        prefill_capture_rows=tuple(prefill_capture_rows),
        decode_rows=tuple(decode_rows),
        decode_capture_rows=tuple(decode_capture_rows),
        has_prefill_rows=bool(prefill_rows),
        has_decode_rows=bool(decode_rows),
        has_capture_rows=bool(prefill_capture_rows or decode_capture_rows),
        has_prefill_capture_rows=bool(prefill_capture_rows),
        has_decode_capture_rows=bool(decode_capture_rows),
        prefill_active_count=len(prefill_rows),
        decode_active_count=len(decode_rows),
    )
