"""Comparison helpers for bs=2 sparse parity checks.

These helpers intentionally separate:
1. token-level exact matching (strict, format-sensitive)
2. semantic matching (format-insensitive, boxed-answer first)
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple


_BOXED_TEXT_PATTERN = re.compile(r"\\boxed\{\\text\{([^{}]+)\}\}")
_BOXED_PATTERN = re.compile(r"\\boxed\{([^{}]+)\}")
_WORD_PATTERN = re.compile(r"\w+", re.UNICODE)


def _normalize_text(text: str) -> str:
    # Collapse all whitespace and drop markdown emphasis markers.
    return " ".join(str(text).replace("**", "").split()).strip()


def _normalize_answer(text: str) -> str:
    answer = _normalize_text(text).strip().strip(".")
    words = answer.split()
    if len(words) >= 3 and words[0].casefold() in {"the", "a", "an"}:
        answer = " ".join(words[1:])
    return answer


def _answer_key(text: str) -> str:
    return _normalize_answer(text).casefold()


def _answer_keys_equivalent(left: str, right: str) -> bool:
    return _answer_key(left) == _answer_key(right)


def _contains_answer_phrase(text: str, answer: str) -> bool:
    answer_key = _answer_key(answer)
    if not answer_key:
        return False
    # Avoid accepting a broad one-word answer inside a more specific phrase
    # such as "Hungary" in "Royal Hungary".
    if len(_WORD_PATTERN.findall(answer_key)) < 2:
        return _answer_key(text) == answer_key
    text_key = _answer_key(text)
    # A prose answer may precede the final answer with an explanation, but it
    # must not extend the answer after the matched phrase.  This rejects a
    # broad reference such as "University of California" against the more
    # specific "University of California, San Diego" while still accepting
    # "The answer is Royal Hungary.".  Only terminal punctuation/wrappers are
    # ignored; trailing lexical content is never swallowed.
    terminal_text_key = re.sub(
        r'''(?:[.!?\u2026;:]+|["'\u201d\u2019)\]\}]+|\$+|`+)+$''',
        "",
        text_key,
    ).rstrip()
    return re.search(
        rf"(?<!\w){re.escape(answer_key)}$",
        terminal_text_key,
    ) is not None


def extract_boxed_answer(text: str) -> Optional[str]:
    raw_text = str(text)
    matches = _BOXED_TEXT_PATTERN.findall(raw_text)
    if not matches:
        matches = _BOXED_PATTERN.findall(raw_text)
    if not matches:
        return None
    return _normalize_answer(matches[-1])


def semantic_match(ref_text: str, test_text: str) -> Tuple[bool, str, str, str]:
    ref_boxed = extract_boxed_answer(ref_text)
    test_boxed = extract_boxed_answer(test_text)
    if ref_boxed is not None and test_boxed is not None:
        return (
            _answer_keys_equivalent(ref_boxed, test_boxed),
            "boxed_answer",
            ref_boxed,
            test_boxed,
        )
    if ref_boxed is not None and _contains_answer_phrase(test_text, ref_boxed):
        return True, "boxed_answer_in_text", ref_boxed, _normalize_text(test_text)
    if test_boxed is not None and _contains_answer_phrase(ref_text, test_boxed):
        return True, "boxed_answer_in_text", _normalize_text(ref_text), test_boxed

    ref_norm = _normalize_text(ref_text)
    test_norm = _normalize_text(test_text)
    return ref_norm == test_norm, "normalized_text", ref_norm, test_norm


def diff_tokens_with_semantic(ref: Dict[str, object], test: Dict[str, object]) -> Dict[str, object]:
    ref_tokens: List[int] = ref["token_ids"]  # type: ignore[assignment]
    test_tokens: List[int] = test["token_ids"]  # type: ignore[assignment]
    token_match = ref_tokens == test_tokens
    first_diff = -1
    for idx, (a, b) in enumerate(zip(ref_tokens, test_tokens)):
        if a != b:
            first_diff = idx
            break
    if first_diff == -1 and len(ref_tokens) != len(test_tokens):
        first_diff = min(len(ref_tokens), len(test_tokens))

    semantic_ok, semantic_mode, ref_semantic, test_semantic = semantic_match(
        str(ref.get("text", "")),
        str(test.get("text", "")),
    )

    return {
        "match": token_match,
        "len_ref": len(ref_tokens),
        "len_test": len(test_tokens),
        "first_diff": first_diff,
        "semantic_match": bool(semantic_ok),
        "semantic_mode": semantic_mode,
        "ref_semantic": ref_semantic,
        "test_semantic": test_semantic,
        "format_only_mismatch": bool((not token_match) and semantic_ok),
    }


def is_diff_acceptable(diff: Dict[str, object], *, strict_token_match: bool) -> bool:
    if bool(diff.get("match", False)):
        return True
    if strict_token_match:
        return False
    return bool(diff.get("semantic_match", False))
