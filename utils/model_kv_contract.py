"""Model-derived KV capacity identity for exact throughput runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


MODEL_KV_CONTRACT_SCHEMA = "sfi.model_kv_contract.v1"
_DTYPE_BYTES = {
    "bfloat16": 2,
    "bf16": 2,
    "float16": 2,
    "fp16": 2,
    "half": 2,
    "float32": 4,
    "fp32": 4,
}


def _positive_int(config: dict[str, Any], field: str) -> int:
    value = config.get(field)
    if type(value) is not int or value <= 0:
        raise ValueError(f"model config {field} must be a positive integer")
    return value


def derive_model_kv_contract(
    model: Path,
    *,
    tensor_parallel_size: int,
) -> dict[str, object]:
    """Derive exact per-rank KV bytes/token from one local model config."""
    tp = int(tensor_parallel_size)
    if tp <= 0:
        raise ValueError("tensor_parallel_size must be positive")
    config_path = (Path(model) / "config.json").resolve(strict=True)
    if not config_path.is_file():
        raise ValueError(f"model config is not a regular file: {config_path}")
    if any(character in str(config_path) for character in ("\n", "\r", "\t")):
        raise ValueError("model config path contains control characters")
    config_bytes = config_path.read_bytes()
    try:
        config = json.loads(config_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid model config JSON: {config_path}") from exc
    if not isinstance(config, dict):
        raise ValueError("model config root must be an object")

    num_layers = _positive_int(config, "num_hidden_layers")
    num_kv_heads = _positive_int(config, "num_key_value_heads")
    if num_kv_heads % tp != 0:
        raise ValueError(
            "num_key_value_heads must be divisible by tensor_parallel_size: "
            f"kv_heads={num_kv_heads}:tp={tp}"
        )
    head_dim_value = config.get("head_dim")
    if type(head_dim_value) is int and head_dim_value > 0:
        head_dim = head_dim_value
    else:
        hidden_size = _positive_int(config, "hidden_size")
        num_attention_heads = _positive_int(config, "num_attention_heads")
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_attention_heads"
            )
        head_dim = hidden_size // num_attention_heads

    dtype = str(config.get("torch_dtype", config.get("dtype", "")) or "").lower()
    dtype_bytes = _DTYPE_BYTES.get(dtype)
    if dtype_bytes is None:
        raise ValueError(f"unsupported model KV dtype: {dtype!r}")
    total_bytes = num_layers * num_kv_heads * head_dim * 2 * dtype_bytes
    if total_bytes % tp != 0:
        raise ValueError(
            "model KV bytes/token must be divisible by tensor_parallel_size"
        )
    return {
        "schema": MODEL_KV_CONTRACT_SCHEMA,
        "model_config_path": str(config_path),
        "model_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "num_hidden_layers": num_layers,
        "num_key_value_heads": num_kv_heads,
        "head_dim": head_dim,
        "dtype": dtype,
        "dtype_bytes": dtype_bytes,
        "tensor_parallel_size": tp,
        "total_bytes_per_token": total_bytes,
        "per_rank_bytes_per_token": total_bytes // tp,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tensor-parallel-size", type=int, required=True)
    parser.add_argument("--format", choices=("json", "tsv"), default="json")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    contract = derive_model_kv_contract(
        args.model,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    if args.format == "json":
        print(json.dumps(contract, sort_keys=True, separators=(",", ":")))
    else:
        fields = (
            "schema",
            "model_config_path",
            "model_config_sha256",
            "num_hidden_layers",
            "num_key_value_heads",
            "head_dim",
            "dtype",
            "dtype_bytes",
            "tensor_parallel_size",
            "total_bytes_per_token",
            "per_rank_bytes_per_token",
        )
        print("\t".join(str(contract[field]) for field in fields))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
