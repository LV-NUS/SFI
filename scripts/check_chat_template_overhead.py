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


def chat_template_overheads(
    tokenizer: Any,
    prompts: Sequence[str],
    *,
    context_tokens: int,
) -> tuple[int, ...]:
    expected_tokens = int(context_tokens)
    if expected_tokens <= 0:
        raise ValueError("context_tokens must be positive")
    if not prompts:
        raise ValueError("prompt batch is empty")

    raw_lengths = tuple(
        len(tokenizer.encode(prompt, add_special_tokens=False))
        for prompt in prompts
    )
    if raw_lengths != (expected_tokens,) * len(prompts):
        raise ValueError(
            "corpus token length drift before chat rendering: "
            f"actual={raw_lengths!r} expected={expected_tokens}"
        )

    rendered_lengths: list[int] = []
    for prompt in prompts:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        rendered_lengths.append(
            len(tokenizer.encode(rendered, add_special_tokens=False))
        )
    return tuple(length - expected_tokens for length in rendered_lengths)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--context-tokens", type=int, required=True)
    parser.add_argument("--reserve-tokens", type=int, required=True)
    parser.add_argument("--format", choices=("tsv",), default="tsv")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if int(args.batch_size) <= 0:
        raise SystemExit("--batch-size must be positive")
    if int(args.context_tokens) <= 0:
        raise SystemExit("--context-tokens must be positive")
    if int(args.reserve_tokens) <= 0:
        raise SystemExit("--reserve-tokens must be positive")

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
        context_tokens=int(args.context_tokens),
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
