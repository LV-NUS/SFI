from __future__ import annotations

from pathlib import Path
from typing import Any


def load_prompt_batch(
    prompt_path: Path,
    *,
    batch_size: int,
    split_context_prompts: bool,
) -> list[str]:
    raw = prompt_path.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError(f"Prompt file {prompt_path} is empty.")
    count = max(1, int(batch_size))
    if not split_context_prompts:
        return [raw] * count

    segments = [segment for segment in raw.split("Context:") if segment.strip()]
    if len(segments) < count:
        raise ValueError(
            f"Expected at least {count} 'Context:' segments in {prompt_path}, "
            f"found {len(segments)}."
        )
    return ["Context:" + segment.strip() for segment in segments[:count]]


def output_payload_from_generation(item: Any, *, include_text: bool) -> object:
    output = item.outputs[0]
    token_ids = [int(token_id) for token_id in output.token_ids]
    if not include_text:
        return token_ids
    return {
        "token_ids": token_ids,
        "text": str(getattr(output, "text", "")),
    }


def maybe_apply_chat_template(
    engine: Any,
    prompts: list[str],
    *,
    use_chat_template: bool,
    enable_thinking: bool,
) -> list[str]:
    if not use_chat_template:
        return prompts
    tokenizer = engine.get_tokenizer()
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        for prompt in prompts
    ]
