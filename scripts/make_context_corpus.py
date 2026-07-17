#!/usr/bin/env python3
"""Build deterministic ``Context:`` segments for throughput benchmarks."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable


CONTEXT_DELIMITER = "Context:"
ESCAPED_CONTEXT_DELIMITER = "Context："
CACHE_SCHEMA = "sfi.context_corpus_cache.v4"
SOURCE_STREAM_MODE = "cyclic_token_stream_v1"
SOURCE_PHASE_SOLVER = "minimal_exact_cyclic_phase_v1"
WIRE_VALIDATION_CONTRACT = "load_prompt_batch_exact_v1"
_MODEL_WEIGHT_SUFFIXES = {
    ".bin",
    ".gguf",
    ".h5",
    ".msgpack",
    ".npy",
    ".npz",
    ".onnx",
    ".pt",
    ".pth",
    ".safetensors",
}
DEFAULT_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "longbench_prompt_full_1.txt"
)
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _build_context_corpus_with_report(
    source_text: str,
    tokenizer: Any,
    *,
    segments: int,
    tokens_per_segment: int,
) -> tuple[str, dict[str, object]]:
    """Token-slice ``source_text`` into the prompt loader's exact wire format."""
    segment_count = int(segments)
    segment_tokens = int(tokens_per_segment)
    if segment_count <= 0:
        raise ValueError("segments must be positive")
    if segment_tokens <= 0:
        raise ValueError("tokens_per_segment must be positive")
    if not source_text:
        raise ValueError("source prompt is empty")

    # ``Context:`` is the on-wire record delimiter consumed by
    # load_prompt_batch.  Long sources can legitimately contain the same text
    # far beyond the short-context smoke range.  Escape it once, before
    # tokenization, so source content can never create extra records while the
    # exact-token solver still measures the bytes the runner will consume.
    escaped_source_text = source_text.replace(
        CONTEXT_DELIMITER,
        ESCAPED_CONTEXT_DELIMITER,
    )
    source_token_ids = list(
        tokenizer.encode(escaped_source_text, add_special_tokens=False)
    )
    if not source_token_ids:
        raise ValueError("source prompt tokenizes to zero tokens")

    # The source is benchmark content, not a capacity limit.  Treat its tokens
    # as one deterministic cyclic stream so every requested batch row can be
    # built exactly, even when aggregate batch tokens exceed the source file.
    # This is an offline corpus-build operation; inference never executes it.
    def _source_slice(start: int, count: int) -> list[int]:
        if count <= 0:
            return []
        source_size = len(source_token_ids)
        offset = int(start) % source_size
        first_count = min(int(count), source_size - offset)
        result = source_token_ids[offset : offset + first_count]
        remaining = int(count) - first_count
        if remaining > 0:
            full_cycles, tail = divmod(remaining, source_size)
            if full_cycles:
                result.extend(source_token_ids * full_cycles)
            if tail:
                result.extend(source_token_ids[:tail])
        return result

    def _try_render_exact_prompt(
        start: int,
        segment_index: int,
    ) -> tuple[tuple[str, int] | None, tuple[int, int]]:
        prefix_tokens = len(
            tokenizer.encode(CONTEXT_DELIMITER, add_special_tokens=False)
        )
        candidate = max(1, segment_tokens - prefix_tokens)
        max_candidate = max(4096, segment_tokens * 4)
        seen: set[int] = set()
        measured: dict[int, tuple[str, int]] = {}

        def _measure(count: int) -> tuple[str, int]:
            cached = measured.get(count)
            if cached is not None:
                return cached
            decoded = str(
                tokenizer.decode(
                    _source_slice(start, count),
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
            )
            if CONTEXT_DELIMITER in decoded:
                raise ValueError(
                    "decoded source chunk contains reserved prompt delimiter: "
                    f"segment={segment_index}, delimiter={CONTEXT_DELIMITER!r}"
                )
            prompt = CONTEXT_DELIMITER + decoded.strip()
            actual = len(tokenizer.encode(prompt, add_special_tokens=False))
            measured[count] = (prompt, actual)
            return prompt, actual

        for _ in range(16):
            if candidate in seen:
                break
            seen.add(candidate)
            prompt, actual = _measure(candidate)
            if actual == segment_tokens:
                return (prompt, candidate), (candidate, actual)
            next_candidate = candidate + (segment_tokens - actual)
            if next_candidate < 1 or next_candidate > max_candidate:
                break
            candidate = next_candidate

        center = candidate
        lower = max(1, center - 64)
        upper = min(max_candidate, center + 64)
        for candidate in sorted(
            range(lower, upper + 1), key=lambda value: abs(value - center)
        ):
            prompt, actual = _measure(candidate)
            if actual == segment_tokens:
                return (prompt, candidate), (candidate, actual)

        closest_count, (_, closest_actual) = min(
            measured.items(),
            key=lambda item: abs(item[1][1] - segment_tokens),
        )
        return None, (closest_count, closest_actual)

    # Decode/strip/re-encode is not length-surjective at every token boundary:
    # a fixed source start can jump from N-1 to N+1 wire tokens.  Select one
    # corpus-wide cyclic phase, then keep every segment contiguous and
    # non-overlapping.  This retires per-segment source switching/skipping and
    # makes the chosen stream deterministic for every batch shape.
    # One complete token period is both necessary and sufficient: every later
    # cyclic phase repeats a start already considered here.  Do not impose an
    # arbitrary retry cap that could reject an existing canonical exact window.
    phase_count = len(source_token_ids)
    best_failure: tuple[int, int, int, int] | None = None
    for source_phase in range(phase_count):
        rendered_segments: list[str] = []
        consumed_by_segment: list[int] = []
        source_cursor = int(source_phase)
        for segment_index in range(segment_count):
            rendered, closest = _try_render_exact_prompt(
                source_cursor,
                segment_index,
            )
            if rendered is None:
                closest_count, closest_actual = closest
                failure = (
                    abs(int(closest_actual) - segment_tokens),
                    int(source_phase),
                    int(segment_index),
                    int(closest_count),
                )
                if best_failure is None or failure < best_failure:
                    best_failure = failure
                break
            prompt, consumed = rendered
            rendered_segments.append(prompt)
            consumed_by_segment.append(int(consumed))
            source_cursor += int(consumed)
        else:
            return "\n".join(rendered_segments), {
                "source_phase": int(source_phase),
                "source_phase_period_tokens": int(len(source_token_ids)),
                "source_tokens_consumed_by_segment": consumed_by_segment,
            }

    _, closest_phase, closest_segment, closest_count = best_failure or (
        -1,
        -1,
        -1,
        -1,
    )
    raise ValueError(
        "unable to render exact wire prompt token length with canonical "
        "cyclic source phase: "
        f"segment={closest_segment}, expected={segment_tokens}, "
        f"closest_source_phase={closest_phase}, "
        f"closest_source_tokens_used={closest_count}, "
        f"phases_considered={phase_count}"
    )


def build_context_corpus(
    source_text: str,
    tokenizer: Any,
    *,
    segments: int,
    tokens_per_segment: int,
) -> str:
    """Build the exact wire corpus from one canonical cyclic source stream."""
    corpus, _ = _build_context_corpus_with_report(
        source_text,
        tokenizer,
        segments=segments,
        tokens_per_segment=tokens_per_segment,
    )
    return corpus


def write_text_atomic(path: Path, text: str) -> None:
    """Publish a complete corpus atomically across parallel benchmark tags."""
    output_path = Path(path)
    if output_path.is_symlink():
        raise ValueError(f"refusing symlink output path: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=str(output_path.parent),
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_path, 0o644)
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


def tokenizer_model_fingerprint(model_path: Path) -> str:
    """Hash every non-weight local model file that may affect tokenization."""
    root = Path(model_path).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"local tokenizer model directory not found: {root}")
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and ".git" not in path.relative_to(root).parts
        and ".cache" not in path.relative_to(root).parts
        and path.suffix.lower() not in _MODEL_WEIGHT_SUFFIXES
    )
    if not files:
        raise ValueError(f"local tokenizer model has no identity files: {root}")
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_file_sha256(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def context_corpus_cache_identity(
    *,
    source_path: Path,
    model_path: Path,
    segments: int,
    tokens_per_segment: int,
) -> dict[str, object]:
    source = Path(source_path).expanduser().resolve()
    model = Path(model_path).expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"source prompt not found: {source}")
    if int(segments) <= 0:
        raise ValueError("segments must be positive")
    if int(tokens_per_segment) <= 0:
        raise ValueError("tokens_per_segment must be positive")
    return {
        "schema": CACHE_SCHEMA,
        "source_sha256": _file_sha256(source),
        "tokenizer_fingerprint": tokenizer_model_fingerprint(model),
        "source_delimiter_escape": ESCAPED_CONTEXT_DELIMITER,
        "source_stream_mode": SOURCE_STREAM_MODE,
        "source_phase_solver": SOURCE_PHASE_SOLVER,
        "segments": int(segments),
        "tokens_per_segment": int(tokens_per_segment),
    }


def _cache_entry_is_valid(
    *,
    corpus_path: Path,
    manifest_path: Path,
    identity: dict[str, object],
) -> bool:
    try:
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
        if not isinstance(manifest, dict):
            return False
        expected_segments = int(identity["segments"])
        expected_tokens = int(identity["tokens_per_segment"])
        wire_token_lengths = manifest.get("wire_token_lengths")
        consumed_by_segment = manifest.get(
            "source_tokens_consumed_by_segment"
        )
        source_phase_period_tokens = manifest.get(
            "source_phase_period_tokens"
        )
        return bool(
            manifest.get("identity") == identity
            and manifest.get("wire_validation_contract")
            == WIRE_VALIDATION_CONTRACT
            and isinstance(wire_token_lengths, list)
            and len(wire_token_lengths) == expected_segments
            and all(
                type(length) is int and length == expected_tokens
                for length in wire_token_lengths
            )
            and type(manifest.get("source_phase")) is int
            and type(source_phase_period_tokens) is int
            and source_phase_period_tokens > 0
            and 0 <= int(manifest["source_phase"]) < source_phase_period_tokens
            and isinstance(consumed_by_segment, list)
            and len(consumed_by_segment) == expected_segments
            and all(
                type(count) is int and count > 0
                for count in consumed_by_segment
            )
            and corpus_path.is_file()
            and int(manifest.get("corpus_size_bytes", -1))
            == int(corpus_path.stat().st_size)
            and manifest.get("corpus_sha256") == _file_sha256(corpus_path)
        )
    except (OSError, TypeError, ValueError):
        return False


def _load_local_tokenizer(model_path: Path) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True,
    )


def ensure_context_corpus_cached(
    *,
    cache_dir: Path,
    source_path: Path,
    model_path: Path,
    segments: int,
    tokens_per_segment: int,
    tokenizer_loader: Callable[[Path], Any] = _load_local_tokenizer,
) -> tuple[Path, bool]:
    """Return a verified content-addressed corpus, building it once per identity."""
    source = Path(source_path).expanduser().resolve()
    model = Path(model_path).expanduser().resolve()
    identity = context_corpus_cache_identity(
        source_path=source,
        model_path=model,
        segments=segments,
        tokens_per_segment=tokens_per_segment,
    )
    identity_json = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    cache_key = hashlib.sha256(identity_json.encode("utf-8")).hexdigest()
    cache_root = Path(cache_dir).expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    corpus_path = cache_root / f"ctx_{int(segments)}x{int(tokens_per_segment)}_{cache_key}.txt"
    manifest_path = corpus_path.with_suffix(".manifest.json")
    lock_path = corpus_path.with_suffix(".lock")

    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        if _cache_entry_is_valid(
            corpus_path=corpus_path,
            manifest_path=manifest_path,
            identity=identity,
        ):
            return corpus_path, True

        source_text = source.read_text(encoding="utf-8")
        tokenizer = tokenizer_loader(model)
        corpus, build_report = _build_context_corpus_with_report(
            source_text,
            tokenizer,
            segments=int(segments),
            tokens_per_segment=int(tokens_per_segment),
        )
        write_text_atomic(corpus_path, corpus)
        wire_token_lengths = validate_context_corpus(
            corpus_path,
            tokenizer,
            segments=int(segments),
            tokens_per_segment=int(tokens_per_segment),
        )
        manifest = {
            "identity": identity,
            "corpus_sha256": _file_sha256(corpus_path),
            "corpus_size_bytes": int(corpus_path.stat().st_size),
            "wire_validation_contract": WIRE_VALIDATION_CONTRACT,
            "wire_token_lengths": list(wire_token_lengths),
            **build_report,
        }
        write_text_atomic(
            manifest_path,
            json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        )
        return corpus_path, False


def validate_context_corpus(
    prompt_path: Path,
    tokenizer: Any,
    *,
    segments: int,
    tokens_per_segment: int,
) -> tuple[int, ...]:
    """Validate the exact prompts consumed by ``load_prompt_batch``."""
    from benchmarks.prompt_batch_io import load_prompt_batch

    expected_segments = int(segments)
    expected_tokens = int(tokens_per_segment)
    prompts = load_prompt_batch(
        Path(prompt_path),
        batch_size=expected_segments,
        split_context_prompts=True,
    )
    token_lengths = tuple(
        len(tokenizer.encode(prompt, add_special_tokens=False))
        for prompt in prompts
    )
    if (
        len(token_lengths) != expected_segments
        or any(length != expected_tokens for length in token_lengths)
    ):
        raise ValueError(
            "context corpus wire prompt token length mismatch: "
            f"actual={token_lengths}, expected_segments={expected_segments}, "
            f"expected_tokens={expected_tokens}"
        )
    return token_lengths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local tokenizer model path")
    parser.add_argument("--segments", type=int, required=True)
    parser.add_argument("--tokens-per-segment", type=int, required=True)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    output_group = parser.add_mutually_exclusive_group(required=True)
    output_group.add_argument("--output", type=Path)
    output_group.add_argument("--cache-dir", type=Path)
    output_group.add_argument("--validate", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if int(args.segments) <= 0:
        raise SystemExit("--segments must be positive")
    if int(args.tokens_per_segment) <= 0:
        raise SystemExit("--tokens-per-segment must be positive")

    model_path = Path(args.model).expanduser().resolve()
    if not model_path.is_dir():
        raise SystemExit(f"local tokenizer model not found: {model_path}")
    if args.validate is not None:
        validate_path = Path(args.validate).expanduser().resolve()
        if not validate_path.is_file():
            raise SystemExit(f"context corpus not found: {validate_path}")
        token_lengths = validate_context_corpus(
            validate_path,
            _load_local_tokenizer(model_path),
            segments=int(args.segments),
            tokens_per_segment=int(args.tokens_per_segment),
        )
        print(
            "context corpus validated: "
            f"segments={len(token_lengths)} tokens_per_segment={token_lengths[0]} "
            f"path={validate_path}"
        )
        return 0

    source_path = Path(args.source).expanduser().resolve()
    if not source_path.is_file():
        raise SystemExit(f"source prompt not found: {source_path}")
    if args.cache_dir is not None:
        output_path, cache_hit = ensure_context_corpus_cached(
            cache_dir=args.cache_dir,
            source_path=source_path,
            model_path=model_path,
            segments=int(args.segments),
            tokens_per_segment=int(args.tokens_per_segment),
        )
        print(
            f"context corpus cache {'hit' if cache_hit else 'miss'}: {output_path}",
            file=os.sys.stderr,
        )
        print(f"context_corpus_path={output_path}")
        return 0

    source_text = source_path.read_text(encoding="utf-8")
    tokenizer = _load_local_tokenizer(model_path)
    corpus = build_context_corpus(
        source_text,
        tokenizer,
        segments=int(args.segments),
        tokens_per_segment=int(args.tokens_per_segment),
    )
    # Do not call Path.resolve() here: it follows an existing final symlink and
    # turns an otherwise atomic replacement into an overwrite of its victim.
    # Parent components may be normalized by the caller, but the final path
    # must retain its own identity so write_text_atomic can reject a symlink.
    output_path = Path(
        os.path.abspath(os.path.expanduser(os.fspath(args.output)))
    )
    write_text_atomic(output_path, corpus)
    print(
        "context corpus ready: "
        f"segments={int(args.segments)} "
        f"tokens_per_segment={int(args.tokens_per_segment)} "
        f"source={source_path} output={output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
