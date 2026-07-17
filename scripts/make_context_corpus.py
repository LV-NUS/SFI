#!/usr/bin/env python3
"""Build deterministic ``Context:`` segments for throughput benchmarks."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable


CONTEXT_DELIMITER = "Context:"
ESCAPED_CONTEXT_DELIMITER = "Context："
CACHE_SCHEMA = "sfi.context_corpus_cache.v5"
SOURCE_STREAM_MODE = "cyclic_layout_body_token_stream_v1"
SOURCE_PHASE_SOLVER = "minimal_exact_cyclic_phase_v1"
WIRE_VALIDATION_CONTRACT = "load_prompt_batch_exact_v1"
SOURCE_LAYOUT_VALIDATION_CONTRACT = "load_prompt_batch_source_layout_v1"
RAW_SOURCE_LAYOUT = "raw_body_v1"
LONGBENCH_SOURCE_LAYOUT = "longbench_text_envelope_v1"
LONGBENCH_TEXT_OPEN = "<text>"
LONGBENCH_TEXT_CLOSE = "</text>"
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


def _split_source_layout(source_text: str) -> tuple[str, str, str, str]:
    """Return one explicit envelope and the only cyclic body token source."""

    if not source_text:
        raise ValueError("source prompt is empty")
    open_count = source_text.count(LONGBENCH_TEXT_OPEN)
    close_count = source_text.count(LONGBENCH_TEXT_CLOSE)
    if open_count == 0 and close_count == 0:
        return RAW_SOURCE_LAYOUT, "", source_text, ""
    if open_count != 1 or close_count != 1:
        raise ValueError(
            "malformed LongBench text envelope: expected exactly one "
            f"{LONGBENCH_TEXT_OPEN!r} and one {LONGBENCH_TEXT_CLOSE!r}"
        )

    open_start = source_text.index(LONGBENCH_TEXT_OPEN)
    body_start = open_start + len(LONGBENCH_TEXT_OPEN)
    close_start = source_text.index(LONGBENCH_TEXT_CLOSE)
    if close_start <= body_start:
        raise ValueError("malformed LongBench text envelope: markers are out of order")

    instruction = source_text[:open_start].strip()
    body = source_text[body_start:close_start].strip()
    question = source_text[close_start + len(LONGBENCH_TEXT_CLOSE) :].strip()
    if not instruction:
        raise ValueError("LongBench text envelope is missing its instruction")
    if not body:
        raise ValueError("LongBench text envelope has an empty body")
    if not question:
        raise ValueError("LongBench text envelope is missing its question")

    prefix = f"{instruction}\n\n{LONGBENCH_TEXT_OPEN}\n"
    suffix = f"\n{LONGBENCH_TEXT_CLOSE}\n\n{question}"
    return LONGBENCH_SOURCE_LAYOUT, prefix, body, suffix


def _escaped_source_layout(source_text: str) -> tuple[str, str, str, str]:
    source_layout, fixed_prefix, source_body, fixed_suffix = _split_source_layout(
        source_text
    )
    return (
        source_layout,
        fixed_prefix.replace(CONTEXT_DELIMITER, ESCAPED_CONTEXT_DELIMITER),
        source_body.replace(CONTEXT_DELIMITER, ESCAPED_CONTEXT_DELIMITER),
        fixed_suffix.replace(CONTEXT_DELIMITER, ESCAPED_CONTEXT_DELIMITER),
    )


def _source_layout_template_sha256(fixed_prefix: str, fixed_suffix: str) -> str:
    digest = hashlib.sha256()
    digest.update(fixed_prefix.encode("utf-8"))
    digest.update(b"\0")
    digest.update(fixed_suffix.encode("utf-8"))
    return digest.hexdigest()


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
    # ``Context:`` is the on-wire record delimiter consumed by
    # load_prompt_batch.  Long sources can legitimately contain the same text
    # far beyond the short-context smoke range.  Classify one explicit source
    # layout first.  A LongBench source keeps its instruction and question on
    # every request; only the <text> body is a cyclic capacity source.  This
    # prevents batch rows from starting in arbitrary prose/JSON without a
    # task.  Marker-free inputs retain the explicit raw-body contract used by
    # unit fixtures and custom corpora.  Partial/malformed envelopes fail.
    source_layout, fixed_prefix, source_body, fixed_suffix = _escaped_source_layout(
        source_text
    )
    source_body_token_ids = list(
        tokenizer.encode(source_body, add_special_tokens=False)
    )
    if not source_body_token_ids:
        raise ValueError("source prompt body tokenizes to zero tokens")

    fixed_wire_tokens = len(
        tokenizer.encode(
            CONTEXT_DELIMITER + fixed_prefix + fixed_suffix,
            add_special_tokens=False,
        )
    )
    if segment_tokens <= fixed_wire_tokens:
        raise ValueError(
            "tokens_per_segment cannot hold the source layout envelope and "
            "a non-empty body: "
            f"tokens_per_segment={segment_tokens}, "
            f"fixed_wire_tokens={fixed_wire_tokens}, layout={source_layout}"
        )

    # The source is benchmark content, not a capacity limit.  Treat its tokens
    # as one deterministic cyclic stream so every requested batch row can be
    # built exactly, even when aggregate batch tokens exceed the source file.
    # This is an offline corpus-build operation; inference never executes it.
    def _source_slice(start: int, count: int) -> list[int]:
        if count <= 0:
            return []
        source_size = len(source_body_token_ids)
        offset = int(start) % source_size
        first_count = min(int(count), source_size - offset)
        result = source_body_token_ids[offset : offset + first_count]
        remaining = int(count) - first_count
        if remaining > 0:
            full_cycles, tail = divmod(remaining, source_size)
            if full_cycles:
                result.extend(source_body_token_ids * full_cycles)
            if tail:
                result.extend(source_body_token_ids[:tail])
        return result

    def _try_render_exact_prompt(
        start: int,
        segment_index: int,
    ) -> tuple[tuple[str, int] | None, tuple[int, int]]:
        candidate = max(1, segment_tokens - fixed_wire_tokens)
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
            if source_layout == LONGBENCH_SOURCE_LAYOUT and (
                LONGBENCH_TEXT_OPEN in decoded
                or LONGBENCH_TEXT_CLOSE in decoded
            ):
                raise ValueError(
                    "decoded LongBench body chunk recreated a reserved envelope "
                    f"marker: segment={segment_index}"
                )
            rendered_body = (
                decoded.strip()
                if source_layout == RAW_SOURCE_LAYOUT
                else decoded
            )
            prompt = (
                CONTEXT_DELIMITER
                + fixed_prefix
                + rendered_body
                + fixed_suffix
            )
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

    # Decode/render/re-encode is not length-surjective at every token boundary:
    # a fixed source start can jump from N-1 to N+1 wire tokens.  Select one
    # corpus-wide cyclic phase, then keep every segment contiguous and
    # non-overlapping.  This retires per-segment source switching/skipping and
    # makes the chosen stream deterministic for every batch shape.
    # One complete token period is both necessary and sufficient: every later
    # cyclic phase repeats a start already considered here.  Do not impose an
    # arbitrary retry cap that could reject an existing canonical exact window.
    phase_count = len(source_body_token_ids)
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
                "source_layout_mode": source_layout,
                "source_phase": int(source_phase),
                "source_phase_period_tokens": int(len(source_body_token_ids)),
                "source_body_tokens_consumed_by_segment": consumed_by_segment,
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


def _read_regular_bytes(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    with os.fdopen(fd, "rb") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise OSError(f"not a regular file: {path}")
        return handle.read()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(_read_regular_bytes(Path(path)))
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
    source_snapshot = source.read_bytes()
    return _context_corpus_cache_identity_from_snapshot(
        source_snapshot=source_snapshot,
        model_path=model,
        segments=segments,
        tokens_per_segment=tokens_per_segment,
    )


def _context_corpus_cache_identity_from_snapshot(
    *,
    source_snapshot: bytes,
    model_path: Path,
    segments: int,
    tokens_per_segment: int,
) -> dict[str, object]:
    model = Path(model_path).expanduser().resolve()
    if int(segments) <= 0:
        raise ValueError("segments must be positive")
    if int(tokens_per_segment) <= 0:
        raise ValueError("tokens_per_segment must be positive")
    source_text = bytes(source_snapshot).decode("utf-8")
    source_layout, fixed_prefix, _, fixed_suffix = _escaped_source_layout(
        source_text
    )
    return {
        "schema": CACHE_SCHEMA,
        "source_sha256": hashlib.sha256(source_snapshot).hexdigest(),
        "tokenizer_fingerprint": tokenizer_model_fingerprint(model),
        "source_delimiter_escape": ESCAPED_CONTEXT_DELIMITER,
        "source_layout_mode": source_layout,
        "source_layout_template_sha256": _source_layout_template_sha256(
            fixed_prefix,
            fixed_suffix,
        ),
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
    source_snapshot: bytes,
    tokenizer: Any,
) -> bool:
    try:
        validated_manifest_path, _, _ = validate_context_corpus_manifest(
            corpus_path=corpus_path,
            expected_identity=identity,
            source_snapshot=source_snapshot,
            tokenizer=tokenizer,
        )
        return validated_manifest_path == manifest_path
    except (OSError, TypeError, ValueError):
        return False


def validate_context_corpus_manifest(
    *,
    corpus_path: Path,
    expected_identity: dict[str, object],
    source_snapshot: bytes,
    tokenizer: Any,
) -> tuple[Path, str, dict[str, object]]:
    """Validate one v5 corpus/manifest pair against its complete identity.

    This is the sole manifest contract shared by cache admission, run
    preflight, and postflight.  It deliberately validates immutable snapshots
    of regular files so symlink and partial-contract paths cannot become a
    second release route.
    """

    corpus = Path(corpus_path)
    manifest_path = corpus.with_suffix(".manifest.json")
    if corpus.is_symlink() or manifest_path.is_symlink():
        raise ValueError("corpus and manifest must be regular non-symlink files")
    corpus_bytes = _read_regular_bytes(corpus)
    manifest_bytes = _read_regular_bytes(manifest_path)
    manifest = json.loads(
        manifest_bytes.decode("utf-8"),
        parse_constant=_reject_json_constant,
    )
    if not isinstance(manifest, dict):
        raise ValueError("corpus manifest root is not an object")
    if manifest.get("identity") != expected_identity:
        raise ValueError("corpus manifest identity mismatch")

    expected_segments = int(expected_identity["segments"])
    expected_tokens = int(expected_identity["tokens_per_segment"])
    wire_token_lengths = manifest.get("wire_token_lengths")
    body_tokens_consumed = manifest.get("source_body_tokens_consumed_by_segment")
    layout_verified = manifest.get("source_layout_verified_by_segment")
    phase = manifest.get("source_phase")
    phase_period = manifest.get("source_phase_period_tokens")
    expected_template = expected_identity.get("source_layout_template_sha256")
    exact_fields = {
        "source_layout_mode": expected_identity.get("source_layout_mode"),
        "source_layout_template_sha256": expected_template,
        "source_layout_validation_contract": SOURCE_LAYOUT_VALIDATION_CONTRACT,
        "wire_validation_contract": WIRE_VALIDATION_CONTRACT,
        "corpus_size_bytes": len(corpus_bytes),
        "corpus_sha256": hashlib.sha256(corpus_bytes).hexdigest(),
    }
    for field, expected in exact_fields.items():
        if manifest.get(field) != expected:
            raise ValueError(f"corpus manifest field mismatch: {field}")

    source_text = bytes(source_snapshot).decode("utf-8")
    if hashlib.sha256(source_snapshot).hexdigest() != expected_identity.get(
        "source_sha256"
    ):
        raise ValueError("corpus source snapshot identity mismatch")
    expected_corpus, expected_build_report = _build_context_corpus_with_report(
        source_text,
        tokenizer,
        segments=expected_segments,
        tokens_per_segment=expected_tokens,
    )
    if corpus_bytes != expected_corpus.encode("utf-8"):
        raise ValueError("corpus content does not match canonical v5 rendering")
    for field, expected in expected_build_report.items():
        if manifest.get(field) != expected:
            raise ValueError(f"corpus manifest build proof mismatch: {field}")
    if (
        not isinstance(wire_token_lengths, list)
        or len(wire_token_lengths) != expected_segments
        or any(
            type(length) is not int or length != expected_tokens
            for length in wire_token_lengths
        )
    ):
        raise ValueError("corpus manifest wire-token proof mismatch")
    if (
        not isinstance(layout_verified, list)
        or len(layout_verified) != expected_segments
        or any(value is not True for value in layout_verified)
    ):
        raise ValueError("corpus manifest layout proof mismatch")
    if (
        not isinstance(body_tokens_consumed, list)
        or len(body_tokens_consumed) != expected_segments
        or any(type(count) is not int or count <= 0 for count in body_tokens_consumed)
    ):
        raise ValueError("corpus manifest body-consumption proof mismatch")
    if (
        type(phase) is not int
        or type(phase_period) is not int
        or phase_period <= 0
        or not 0 <= phase < phase_period
    ):
        raise ValueError("corpus manifest cyclic-phase proof mismatch")
    return (
        manifest_path,
        hashlib.sha256(manifest_bytes).hexdigest(),
        manifest,
    )


def _load_local_tokenizer(model_path: Path) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True,
    )


def _load_context_prompts(prompt_path: Path, *, segments: int) -> tuple[str, ...]:
    from benchmarks.prompt_batch_io import load_prompt_batch

    return tuple(
        load_prompt_batch(
            Path(prompt_path),
            batch_size=int(segments),
            split_context_prompts=True,
        )
    )


def validate_context_corpus_source_layout(
    prompt_path: Path,
    *,
    segments: int,
    source_layout: str,
    fixed_prefix: str,
    fixed_suffix: str,
) -> tuple[bool, ...]:
    """Validate the source envelope after the benchmark loader consumes it."""

    prompts = _load_context_prompts(prompt_path, segments=int(segments))
    expected_segments = int(segments)
    if len(prompts) != expected_segments:
        raise ValueError(
            "context corpus source layout row count mismatch: "
            f"actual={len(prompts)}, expected={expected_segments}"
        )
    if source_layout == RAW_SOURCE_LAYOUT:
        if fixed_prefix or fixed_suffix:
            raise ValueError("raw source layout cannot carry an envelope")
        return tuple(True for _ in prompts)
    if source_layout != LONGBENCH_SOURCE_LAYOUT:
        raise ValueError(f"unknown source layout: {source_layout!r}")

    expected_prefix = CONTEXT_DELIMITER + fixed_prefix
    verified: list[bool] = []
    for index, prompt in enumerate(prompts):
        row_verified = bool(
            prompt.startswith(expected_prefix)
            and prompt.endswith(fixed_suffix)
            and prompt.count(LONGBENCH_TEXT_OPEN) == 1
            and prompt.count(LONGBENCH_TEXT_CLOSE) == 1
        )
        if not row_verified:
            raise ValueError(
                "context corpus LongBench envelope verification failed: "
                f"segment={index}"
            )
        body_start = len(expected_prefix)
        body_end = len(prompt) - len(fixed_suffix)
        if body_end <= body_start or not prompt[body_start:body_end].strip():
            raise ValueError(
                "context corpus LongBench body is empty after loading: "
                f"segment={index}"
            )
        verified.append(True)
    return tuple(verified)


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
    if not source.is_file():
        raise ValueError(f"source prompt not found: {source}")
    source_snapshot = source.read_bytes()
    identity = _context_corpus_cache_identity_from_snapshot(
        source_snapshot=source_snapshot,
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
        tokenizer = tokenizer_loader(model)
        if _cache_entry_is_valid(
            corpus_path=corpus_path,
            manifest_path=manifest_path,
            identity=identity,
            source_snapshot=source_snapshot,
            tokenizer=tokenizer,
        ):
            return corpus_path, True

        write_context_corpus_artifacts(
            output_path=corpus_path,
            source_snapshot=source_snapshot,
            identity=identity,
            tokenizer=tokenizer,
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
    expected_segments = int(segments)
    expected_tokens = int(tokens_per_segment)
    prompts = _load_context_prompts(
        Path(prompt_path),
        segments=expected_segments,
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


def write_context_corpus_artifacts(
    *,
    output_path: Path,
    source_snapshot: bytes,
    identity: dict[str, object],
    tokenizer: Any,
) -> tuple[Path, Path]:
    """Atomically publish a corpus and its mandatory sibling v5 manifest."""

    output = Path(output_path)
    source_text = bytes(source_snapshot).decode("utf-8")
    source_layout, fixed_prefix, _, fixed_suffix = _escaped_source_layout(
        source_text
    )
    corpus, build_report = _build_context_corpus_with_report(
        source_text,
        tokenizer,
        segments=int(identity["segments"]),
        tokens_per_segment=int(identity["tokens_per_segment"]),
    )
    write_text_atomic(output, corpus)
    wire_token_lengths = validate_context_corpus(
        output,
        tokenizer,
        segments=int(identity["segments"]),
        tokens_per_segment=int(identity["tokens_per_segment"]),
    )
    layout_verified = validate_context_corpus_source_layout(
        output,
        segments=int(identity["segments"]),
        source_layout=source_layout,
        fixed_prefix=fixed_prefix,
        fixed_suffix=fixed_suffix,
    )
    manifest_path = output.with_suffix(".manifest.json")
    manifest = {
        "identity": identity,
        "corpus_sha256": _file_sha256(output),
        "corpus_size_bytes": len(_read_regular_bytes(output)),
        "wire_validation_contract": WIRE_VALIDATION_CONTRACT,
        "wire_token_lengths": list(wire_token_lengths),
        "source_layout_validation_contract": SOURCE_LAYOUT_VALIDATION_CONTRACT,
        "source_layout_template_sha256": _source_layout_template_sha256(
            fixed_prefix,
            fixed_suffix,
        ),
        "source_layout_verified_by_segment": list(layout_verified),
        **build_report,
    }
    write_text_atomic(
        manifest_path,
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
    )
    validate_context_corpus_manifest(
        corpus_path=output,
        expected_identity=identity,
        source_snapshot=source_snapshot,
        tokenizer=tokenizer,
    )
    return output, manifest_path


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
    parser.add_argument(
        "--format",
        choices=("human", "manifest-tsv"),
        default="human",
    )
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
    source_path = Path(args.source).expanduser().resolve()
    if not source_path.is_file():
        raise SystemExit(f"source prompt not found: {source_path}")
    source_snapshot = source_path.read_bytes()
    if args.validate is not None:
        validate_path = Path(
            os.path.abspath(os.path.expanduser(os.fspath(args.validate)))
        )
        if validate_path.is_symlink():
            raise SystemExit(f"context corpus must not be a symlink: {validate_path}")
        if not validate_path.is_file():
            raise SystemExit(f"context corpus not found: {validate_path}")
        identity = _context_corpus_cache_identity_from_snapshot(
            source_snapshot=source_snapshot,
            model_path=model_path,
            segments=int(args.segments),
            tokens_per_segment=int(args.tokens_per_segment),
        )
        tokenizer = _load_local_tokenizer(model_path)
        manifest_path, manifest_sha256, manifest = (
            validate_context_corpus_manifest(
                corpus_path=validate_path,
                expected_identity=identity,
                source_snapshot=source_snapshot,
                tokenizer=tokenizer,
            )
        )
        token_lengths = tuple(int(value) for value in manifest["wire_token_lengths"])
        layout_verified = tuple(
            bool(value)
            for value in manifest["source_layout_verified_by_segment"]
        )
        source_layout = str(manifest["source_layout_mode"])
        if args.format == "manifest-tsv":
            print(
                "\t".join(
                    (
                        str(manifest_path),
                        manifest_sha256,
                        str(identity["schema"]),
                        str(manifest["source_layout_mode"]),
                        str(manifest["source_layout_validation_contract"]),
                        str(len(layout_verified)),
                    )
                )
            )
        else:
            print(
                "context corpus validated: "
                f"segments={len(token_lengths)} "
                f"tokens_per_segment={token_lengths[0]} "
                f"source_layout={source_layout} "
                f"layout_verified={all(layout_verified)} "
                f"path={validate_path}"
            )
        return 0

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

    tokenizer = _load_local_tokenizer(model_path)
    identity = _context_corpus_cache_identity_from_snapshot(
        source_snapshot=source_snapshot,
        model_path=model_path,
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
    write_context_corpus_artifacts(
        output_path=output_path,
        source_snapshot=source_snapshot,
        identity=identity,
        tokenizer=tokenizer,
    )
    print(
        "context corpus ready: "
        f"segments={int(args.segments)} "
        f"tokens_per_segment={int(args.tokens_per_segment)} "
        f"source={source_path} output={output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
