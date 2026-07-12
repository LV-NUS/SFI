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
CACHE_SCHEMA = "sfi.context_corpus_cache.v2"
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


def build_context_corpus(
    source_text: str,
    tokenizer: Any,
    *,
    segments: int,
    tokens_per_segment: int,
) -> str:
    """Token-slice ``source_text`` into the prompt loader's exact wire format."""
    segment_count = int(segments)
    segment_tokens = int(tokens_per_segment)
    if segment_count <= 0:
        raise ValueError("segments must be positive")
    if segment_tokens <= 0:
        raise ValueError("tokens_per_segment must be positive")
    if not source_text:
        raise ValueError("source prompt is empty")

    token_ids = list(tokenizer.encode(source_text, add_special_tokens=False))

    def _render_exact_prompt(start: int, segment_index: int) -> tuple[str, int]:
        available = len(token_ids) - start
        if available <= 0:
            raise ValueError(
                "source prompt is too short for exact wire prompt: "
                f"segment={segment_index}, available_source_tokens={available}"
            )
        prefix_tokens = len(
            tokenizer.encode(CONTEXT_DELIMITER, add_special_tokens=False)
        )
        candidate = min(available, max(1, segment_tokens - prefix_tokens))
        seen: set[int] = set()
        measured: dict[int, tuple[str, int]] = {}

        def _measure(count: int) -> tuple[str, int]:
            cached = measured.get(count)
            if cached is not None:
                return cached
            decoded = str(
                tokenizer.decode(
                    token_ids[start : start + count],
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
                return prompt, candidate
            next_candidate = candidate + (segment_tokens - actual)
            if next_candidate < 1 or next_candidate > available:
                break
            candidate = next_candidate

        center = candidate
        lower = max(1, center - 64)
        upper = min(available, center + 64)
        for candidate in sorted(
            range(lower, upper + 1), key=lambda value: abs(value - center)
        ):
            prompt, actual = _measure(candidate)
            if actual == segment_tokens:
                return prompt, candidate

        closest_count, (_, closest_actual) = min(
            measured.items(),
            key=lambda item: abs(item[1][1] - segment_tokens),
        )
        raise ValueError(
            "unable to render exact wire prompt token length: "
            f"segment={segment_index}, expected={segment_tokens}, "
            f"closest={closest_actual}, source_tokens_used={closest_count}, "
            f"available_source_tokens={available}"
        )

    rendered_segments: list[str] = []
    source_cursor = 0
    for segment_index in range(segment_count):
        prompt, consumed = _render_exact_prompt(source_cursor, segment_index)
        rendered_segments.append(prompt)
        source_cursor += consumed
    return "\n".join(rendered_segments)


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
        return bool(
            isinstance(manifest, dict)
            and manifest.get("identity") == identity
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
        corpus = build_context_corpus(
            source_text,
            tokenizer,
            segments=int(segments),
            tokens_per_segment=int(tokens_per_segment),
        )
        write_text_atomic(corpus_path, corpus)
        manifest = {
            "identity": identity,
            "corpus_sha256": _file_sha256(corpus_path),
            "corpus_size_bytes": int(corpus_path.stat().st_size),
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
