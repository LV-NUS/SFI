from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable


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


def resolved_generation_stop_token_ids(engine: Any, tokenizer: Any) -> tuple[int, ...]:
    """Return the exact stop-token set used by vLLM generation setup.

    vLLM combines the tokenizer EOS with every ``eos_token_id`` declared by
    the model generation config.  Qwen instruct models use more than one such
    token, so treating the tokenizer's primary EOS as the whole contract can
    place the semantic boundary after an earlier valid stop token.  Resolve
    this once outside the timed loop and fail closed on malformed metadata.
    """

    stop_token_ids: set[int] = set()

    def _add(raw: object, *, source: str) -> None:
        values: Iterable[object]
        if raw is None:
            return
        if isinstance(raw, int) and not isinstance(raw, bool):
            values = (raw,)
        elif isinstance(raw, (list, tuple, set)):
            values = raw
        else:
            raise RuntimeError(
                f"E_STOP_TOKEN_CONTRACT: {source} must be an int or sequence"
            )
        for value in values:
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or int(value) < 0
            ):
                raise RuntimeError(
                    f"E_STOP_TOKEN_CONTRACT: invalid {source} token {value!r}"
                )
            stop_token_ids.add(int(value))

    _add(getattr(tokenizer, "eos_token_id", None), source="tokenizer EOS")
    model_config = getattr(engine, "model_config", None)
    getter = getattr(model_config, "try_get_generation_config", None)
    if not callable(getter):
        raise RuntimeError(
            "E_STOP_TOKEN_CONTRACT: engine model generation config unavailable"
        )
    generation_config = getter()
    if not isinstance(generation_config, dict):
        raise RuntimeError(
            "E_STOP_TOKEN_CONTRACT: model generation config is not an object"
        )
    _add(
        generation_config.get("eos_token_id"),
        source="generation-config EOS",
    )
    if not stop_token_ids:
        raise RuntimeError("E_STOP_TOKEN_CONTRACT: no generation stop token")
    return tuple(sorted(stop_token_ids))


def decoded_output_payload(
    token_ids: list[int],
    tokenizer: Any,
    *,
    stop_token_ids: Iterable[int],
) -> dict[str, object]:
    """Decode full timing output and its model-semantic prefix.

    Fixed-length throughput runs intentionally continue after EOS.  Everything
    after the first tokenizer EOS is timing load, not model output semantics.
    Persist the exact token proof so postflight can validate that boundary
    instead of weakening output-health checks for repetitive post-EOS tails.
    This helper runs only after the timed engine loop has ended.
    """

    normalized_token_ids = [int(token_id) for token_id in token_ids]
    decode = getattr(tokenizer, "decode", None)
    text = str(decode(normalized_token_ids)) if callable(decode) else ""
    normalized_stop_token_ids = sorted({int(value) for value in stop_token_ids})
    if not normalized_stop_token_ids or any(
        isinstance(value, bool) or value < 0 for value in normalized_stop_token_ids
    ):
        raise RuntimeError("E_STOP_TOKEN_CONTRACT: invalid decoded stop-token set")
    stop_token_set = set(normalized_stop_token_ids)
    first_stop_index = next(
        (
            index
            for index, token_id in enumerate(normalized_token_ids)
            if token_id in stop_token_set
        ),
        -1,
    )
    boundary_stop_token_id = (
        normalized_token_ids[first_stop_index] if first_stop_index >= 0 else -1
    )
    semantic_token_ids = (
        normalized_token_ids[: first_stop_index + 1]
        if first_stop_index >= 0
        else normalized_token_ids
    )
    semantic_text = (
        str(decode(semantic_token_ids)) if callable(decode) else ""
    )
    return {
        "token_ids": normalized_token_ids,
        "text": text,
        "semantic_text": semantic_text,
        "semantic_stop_seen": first_stop_index >= 0,
        "semantic_first_stop_token_index": int(first_stop_index),
        "semantic_stop_token_id": int(boundary_stop_token_id),
        "semantic_stop_token_ids": normalized_stop_token_ids,
        "semantic_token_count": len(semantic_token_ids),
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
