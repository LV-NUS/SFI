from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Iterable


_POSITIVE_DECIMAL = re.compile(r"[1-9][0-9]*")


def parse_optional_exact_positive_int(
    raw: str | None,
    *,
    option_name: str,
) -> int | None:
    """Parse one optional canonical positive scalar without silent coercion."""
    if raw is None or raw == "":
        return None
    if not isinstance(raw, str) or _POSITIVE_DECIMAL.fullmatch(raw) is None:
        raise ValueError(f"{option_name} must be a canonical positive integer")
    return int(raw)


def parse_exact_positive_int_vector(
    raw: str,
    *,
    option_name: str,
) -> tuple[int, ...]:
    """Parse one canonical comma-separated positive-integer vector.

    Empty elements, signs, zero, and non-canonical decimal spellings are
    rejected instead of being silently normalized.  This keeps command-line
    workload identity stable across the sparse and dense children.
    """

    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{option_name} must be a comma-separated positive vector")
    values: list[int] = []
    for index, part in enumerate(raw.split(",")):
        if _POSITIVE_DECIMAL.fullmatch(part) is None:
            raise ValueError(
                f"{option_name}[{index}] must be a canonical positive integer: "
                f"{part!r}"
            )
        values.append(int(part))
    return tuple(values)


def resolve_exact_positive_int_vector(
    raw: str | None,
    *,
    request_count: int,
    option_name: str,
    homogeneous_value: int | None = None,
) -> tuple[int, ...] | None:
    """Resolve an exact per-request vector once, outside generation loops.

    An explicit vector must have exactly one value per request.  The scalar is
    the maximum envelope when a vector is present, or the homogeneous value
    expanded once when the vector is omitted.  ``None`` is returned only when
    neither representation was supplied; callers such as the context-token
    proof may then derive the vector from the loaded rows.
    """

    count = int(request_count)
    if count <= 0:
        raise ValueError("request_count must be positive")
    if raw is None:
        if homogeneous_value is None:
            return None
        if (
            isinstance(homogeneous_value, bool)
            or not isinstance(homogeneous_value, int)
            or homogeneous_value <= 0
        ):
            raise ValueError(
                f"{option_name} homogeneous value must be a positive integer"
            )
        return (int(homogeneous_value),) * count
    values = parse_exact_positive_int_vector(raw, option_name=option_name)
    if len(values) != count:
        raise ValueError(
            f"{option_name} must contain exactly {count} values, got {len(values)}"
        )
    if homogeneous_value is not None:
        if (
            isinstance(homogeneous_value, bool)
            or not isinstance(homogeneous_value, int)
            or homogeneous_value <= 0
        ):
            raise ValueError(
                f"{option_name} homogeneous value must be a positive integer"
            )
        if max(values) != int(homogeneous_value):
            raise ValueError(
                f"max({option_name}) must equal the scalar envelope "
                f"{int(homogeneous_value)}, got {max(values)}"
            )
    return values


def exact_prompt_token_lengths(
    tokenizer: Any,
    prompts: Iterable[str],
) -> tuple[int, ...]:
    """Return the exact pre-chat-template token count for every prompt row."""

    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        raise RuntimeError("E_REQUEST_CONTEXT_TOKENS: tokenizer.encode unavailable")
    lengths = tuple(
        len(encode(str(prompt), add_special_tokens=False)) for prompt in prompts
    )
    if not lengths or any(length <= 0 for length in lengths):
        raise RuntimeError(
            f"E_REQUEST_CONTEXT_TOKENS: non-positive prompt lengths {lengths!r}"
        )
    return lengths


def resolve_exact_request_context_tokens(
    tokenizer: Any,
    prompts: Iterable[str],
    *,
    declared: tuple[int, ...] | None,
) -> tuple[int, ...]:
    """Prove an optional declared context vector against loaded prompt rows."""

    actual = exact_prompt_token_lengths(tokenizer, prompts)
    if declared is not None and tuple(declared) != actual:
        raise RuntimeError(
            "E_REQUEST_CONTEXT_TOKENS_MISMATCH: "
            f"declared={tuple(declared)!r}:actual={actual!r}"
        )
    return actual


def build_request_sampling_params(
    sampling_params_factory: Callable[..., Any],
    request_max_new_tokens: Iterable[int],
    *,
    respect_eos: bool,
    detokenize: bool,
) -> tuple[Any, ...]:
    """Build one immutable request-parameter object per row before timing."""

    caps = tuple(int(value) for value in request_max_new_tokens)
    if not caps or any(value <= 0 for value in caps):
        raise ValueError("request_max_new_tokens must be a non-empty positive vector")
    return tuple(
        sampling_params_factory(
            temperature=0.0,
            top_p=1.0,
            max_tokens=cap,
            ignore_eos=not bool(respect_eos),
            detokenize=bool(detokenize),
        )
        for cap in caps
    )


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
