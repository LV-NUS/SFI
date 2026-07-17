#!/usr/bin/env python3
"""Verify exact chat-template token overhead for a context corpus."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Sequence


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.prompt_batch_io import load_prompt_batch
from scripts.make_context_corpus import (
    normalize_tokens_per_segment_by_request,
    parse_tokens_per_segment_by_request,
)


def chat_template_overheads(
    tokenizer: Any,
    prompts: Sequence[str],
    *,
    context_tokens_by_request: Sequence[int],
) -> tuple[int, ...]:
    expected_tokens = normalize_tokens_per_segment_by_request(
        context_tokens_by_request
    )
    if not prompts:
        raise ValueError("prompt batch is empty")
    if len(prompts) != len(expected_tokens):
        raise ValueError(
            "prompt batch size does not match context_tokens_by_request: "
            f"prompts={len(prompts)} expected={len(expected_tokens)}"
        )

    raw_lengths = tuple(
        len(tokenizer.encode(prompt, add_special_tokens=False))
        for prompt in prompts
    )
    if raw_lengths != expected_tokens:
        raise ValueError(
            "corpus token length drift before chat rendering: "
            f"actual={raw_lengths!r} expected={expected_tokens!r}"
        )

    rendered_lengths: list[int] = []
    for prompt, expected in zip(prompts, expected_tokens):
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        rendered_lengths.append(
            len(tokenizer.encode(rendered, add_special_tokens=False))
        )
    return tuple(
        length - expected
        for length, expected in zip(rendered_lengths, expected_tokens)
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    context_group = parser.add_mutually_exclusive_group(required=True)
    context_group.add_argument("--context-tokens", type=int)
    context_group.add_argument("--context-tokens-by-request")
    parser.add_argument("--reserve-tokens", type=int, required=True)
    parser.add_argument("--format", choices=("tsv",), default="tsv")
    return parser.parse_args()


def _resolve_context_tokens_by_request(
    args: argparse.Namespace,
) -> tuple[int, ...]:
    vector_raw = getattr(args, "context_tokens_by_request", None)
    scalar_raw = getattr(args, "context_tokens", None)
    if vector_raw is not None and scalar_raw is not None:
        raise ValueError(
            "--context-tokens and --context-tokens-by-request are mutually exclusive"
        )
    if vector_raw is not None:
        expected = (
            parse_tokens_per_segment_by_request(
                vector_raw,
                option_name="--context-tokens-by-request",
            )
            if isinstance(vector_raw, str)
            else normalize_tokens_per_segment_by_request(vector_raw)
        )
        if len(expected) != int(args.batch_size):
            raise ValueError(
                "--context-tokens-by-request count must equal --batch-size"
            )
        return expected
    if type(scalar_raw) is not int or scalar_raw <= 0:
        raise ValueError("--context-tokens must be positive")
    return (scalar_raw,) * int(args.batch_size)


def main() -> int:
    args = _parse_args()
    if int(args.batch_size) <= 0:
        raise SystemExit("--batch-size must be positive")
    if int(args.reserve_tokens) <= 0:
        raise SystemExit("--reserve-tokens must be positive")
    try:
        expected_tokens = _resolve_context_tokens_by_request(args)
    except (TypeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    model_path = args.model.expanduser().resolve()
    corpus_path = Path(
        os.path.abspath(os.path.expanduser(os.fspath(args.corpus)))
    )
    if not model_path.is_dir():
        raise SystemExit(f"local tokenizer model not found: {model_path}")
    if corpus_path.is_symlink() or not corpus_path.is_file():
        raise SystemExit(f"regular context corpus not found: {corpus_path}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True,
    )
    prompts = load_prompt_batch(
        corpus_path,
        batch_size=int(args.batch_size),
        split_context_prompts=True,
    )
    overheads = chat_template_overheads(
        tokenizer,
        prompts,
        context_tokens_by_request=expected_tokens,
    )
    if (
        len(overheads) != int(args.batch_size)
        or min(overheads) < 0
        or max(overheads) > int(args.reserve_tokens)
    ):
        raise SystemExit(
            "chat-template overhead exceeds the fail-closed reserve: "
            f"overheads={overheads!r} reserve={int(args.reserve_tokens)}"
        )
    print(f"{min(overheads)}\t{max(overheads)}\t{len(overheads)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
