#!/usr/bin/env python
"""Build a 'Context:'-segmented prompt corpus for throughput benchmarks.

The SFI benchmark runners accept a corpus file via ``--prompt <file>
--split-context-prompts``. The file format contract (enforced by
``benchmarks/prompt_batch_io.py``) is:

* the file is split on the literal string ``Context:``;
* the number of non-empty segments must be **>= batch size**;
* each request in the batch receives one segment.

This helper cuts a long source text into ``--segments`` chunks of roughly
``--tokens-per-segment`` tokens each (measured with the target model's
tokenizer, so the resulting per-request context length is what you intend).

Example (8 requests x ~12k tokens each):

    python scripts/make_context_corpus.py \
        --model /path/to/model --segments 8 --tokens-per-segment 12000 \
        --output out/ctx_8x12k.txt

Do NOT truncate a corpus file with ``head -c`` afterwards: byte-level cuts can
destroy the segment structure and break the >= batch-size contract.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True,
                        help="model dir; its tokenizer measures segment length")
    parser.add_argument("--segments", type=int, required=True,
                        help="number of 'Context:' segments (>= benchmark batch size)")
    parser.add_argument("--tokens-per-segment", type=int, required=True,
                        help="approx. prompt tokens per segment")
    parser.add_argument("--source", default=None,
                        help="long source text (default: benchmarks/longbench_prompt_full_1.txt)")
    parser.add_argument("--output", required=True, help="corpus file to write")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    source = Path(args.source) if args.source else root / "benchmarks" / "longbench_prompt_full_1.txt"
    text = source.read_text(encoding="utf-8", errors="ignore").replace("Context:", " ")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    ids = tok(text, add_special_tokens=False).input_ids
    need = args.segments * args.tokens_per_segment
    if len(ids) < need:
        reps = need // len(ids) + 1
        ids = (ids * reps)[:need]
        print(f"note: source shorter than requested ({len(ids)//reps} tokens); repeated {reps}x")

    segments = []
    for i in range(args.segments):
        chunk = ids[i * args.tokens_per_segment:(i + 1) * args.tokens_per_segment]
        segments.append("Context:" + tok.decode(chunk, skip_special_tokens=True))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(segments), encoding="utf-8")

    made = out.read_text(encoding="utf-8").count("Context:")
    print(f"wrote {out} segments={made} target_tokens/segment={args.tokens_per_segment}")
    if made < args.segments:
        raise SystemExit(f"ERROR: expected {args.segments} segments, got {made}")


if __name__ == "__main__":
    main()
