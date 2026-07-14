from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
FA_ROOT = Path(
    os.environ.get(
        "VLLM_SPARSE_FA3_UPSTREAM_ROOT",
        str(ROOT / "third_party_upstreams/vllm-project-flash-attention"),
    )
).expanduser().resolve()
sys.path.insert(0, str(FA_ROOT))

# fa4_c1_compact_recent_removed: CR fake-compile markers removed (C1); mixed_page markers kept
EXPECTED_MARKERS = (
    "sm100_cute_mixed_page_no_selected_fake_compile_probe_ok",
    "sm100_cute_mixed_page_selected_brow_fake_compile_probe_ok",
    "sm100_cute_mixed_page_selected_bhrow_capture_fake_compile_probe_ok",
    "sm100_cute_mixed_page_rrp_fake_compile_probe_ok",
)


def _emit_marker(marker: str) -> None:
    if marker not in EXPECTED_MARKERS:
        raise RuntimeError(f"unexpected marker: {marker}")
    print(marker)

flash_attn_pkg = types.ModuleType("flash_attn")
flash_attn_pkg.__path__ = [str(FA_ROOT / "flash_attn")]
sys.modules.setdefault("flash_attn", flash_attn_pkg)

os.environ.setdefault("FLASH_ATTENTION_ARCH", "sm_100a")
os.environ.setdefault("CUTE_DSL_ARCH", "sm_100a")

try:
    # fa4_c1_compact_recent_removed: CR-only imports removed (C1); mixed_page entry kept
    from flash_attn.cute.interface import (  # noqa: E402
        flash_attn_varlen_mixed_page_func,
    )
    from flash_attn.cute.testing import maybe_fake_tensor_mode  # noqa: E402
except ModuleNotFoundError as exc:
    missing = exc.name or str(exc)
    print(
        f"sm100_cute_compact_recent_fake_compile_probe_missing_dependency: {missing}",
        file=sys.stderr,
    )
    raise SystemExit(2) from exc


@maybe_fake_tensor_mode(fake=True)
def main() -> int:
    device = "cuda"
    dtype = torch.bfloat16
    batch = 1
    seqlen_q = 1
    num_q_heads = 8
    num_kv_heads = 8
    head_dim = 128
    page_size = 128
    compact_tokens = 128
    recent_tokens = 128
    q = torch.empty((batch * seqlen_q, num_q_heads, head_dim), device=device, dtype=dtype)
    k = torch.empty((2, page_size, num_kv_heads, head_dim), device=device, dtype=dtype)
    v = torch.empty((2, page_size, num_kv_heads, head_dim), device=device, dtype=dtype)
    k_compact = torch.empty((1, page_size, num_kv_heads, head_dim), device=device, dtype=dtype)
    v_compact = torch.empty((1, page_size, num_kv_heads, head_dim), device=device, dtype=dtype)
    cu_seqlens_q = torch.tensor([0, seqlen_q], device=device, dtype=torch.int32)
    seqused_k = torch.tensor([compact_tokens + recent_tokens], device=device, dtype=torch.int32)
    page_table = torch.tensor([[0, 1]], device=device, dtype=torch.int32)
    resolved_page_table_row_ptr_u64 = torch.empty((batch * num_kv_heads,), device=device, dtype=torch.int64)
    compact_base = torch.tensor([0], device=device, dtype=torch.int32)
    compact_valid = torch.tensor([compact_tokens], device=device, dtype=torch.int32)
    recent_first = torch.tensor([1], device=device, dtype=torch.int32)
    recent_page_count = torch.tensor([1], device=device, dtype=torch.int32)
    recent_len = torch.tensor([recent_tokens], device=device, dtype=torch.int32)
    # fa4_c1_compact_recent_removed: 4 CR fake-compile calls removed (C1); shared capture tensors kept for mixed-page
    capture_scores = torch.empty(
        (batch, num_q_heads, 1, compact_tokens + recent_tokens),
        device=device,
        dtype=torch.float32,
    )
    capture_row_index_i32 = torch.zeros((batch,), device=device, dtype=torch.int32)
    row_capture_last_n_i32 = torch.ones((batch,), device=device, dtype=torch.int32)
    flash_attn_varlen_mixed_page_func(
        q,
        k,
        v,
        cu_seqlens_q,
        seqused_k,
        page_table,
        max_seqlen_q=seqlen_q,
        max_seqlen_k=compact_tokens + recent_tokens,
        causal=True,
        num_splits=0,
        _arch=100,
    )
    _emit_marker("sm100_cute_mixed_page_no_selected_fake_compile_probe_ok")
    selected_page_table_brow = torch.tensor([[0, 1]], device=device, dtype=torch.int32)
    row_consume_mode = torch.ones((batch,), device=device, dtype=torch.int32)
    selected_seqused = torch.full(
        (batch * num_kv_heads,),
        compact_tokens + recent_tokens,
        device=device,
        dtype=torch.int32,
    )
    resolved_seqused = selected_seqused
    flash_attn_varlen_mixed_page_func(
        q,
        k,
        v,
        cu_seqlens_q,
        seqused_k,
        page_table,
        max_seqlen_q=seqlen_q,
        max_seqlen_k=compact_tokens + recent_tokens,
        causal=True,
        num_splits=0,
        _arch=100,
        selected_page_table_i32=selected_page_table_brow,
        row_consume_mode_i32=row_consume_mode,
        selected_seqused_k_by_head_i32=selected_seqused,
    )
    _emit_marker("sm100_cute_mixed_page_selected_brow_fake_compile_probe_ok")
    selected_page_table_bhrow = selected_page_table_brow.repeat_interleave(
        num_kv_heads,
        dim=0,
    )
    flash_attn_varlen_mixed_page_func(
        q,
        k,
        v,
        cu_seqlens_q,
        seqused_k,
        page_table,
        max_seqlen_q=seqlen_q,
        max_seqlen_k=compact_tokens + recent_tokens,
        causal=True,
        num_splits=0,
        _arch=100,
        selected_page_table_i32=selected_page_table_bhrow,
        row_consume_mode_i32=row_consume_mode,
        selected_seqused_k_by_head_i32=selected_seqused,
        capture_scores=capture_scores,
        capture_row_index_i32=capture_row_index_i32,
        row_capture_last_n_i32=row_capture_last_n_i32,
    )
    _emit_marker("sm100_cute_mixed_page_selected_bhrow_capture_fake_compile_probe_ok")
    flash_attn_varlen_mixed_page_func(
        q,
        k,
        v,
        cu_seqlens_q,
        seqused_k,
        page_table,
        max_seqlen_q=seqlen_q,
        max_seqlen_k=compact_tokens + recent_tokens,
        causal=True,
        num_splits=0,
        _arch=100,
        page_resolver_kind=4,
        resolved_page_table_row_ptr_u64=resolved_page_table_row_ptr_u64,
        resolved_seqused_k_by_head_i32=resolved_seqused,
        graph_replay_carriers=True,
    )
    _emit_marker("sm100_cute_mixed_page_rrp_fake_compile_probe_ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
