# Installation Guide

SFI is a set of Python modules that integrate into an existing vLLM installation via runtime patching &mdash; no vLLM source modification is required.

<br>

## Prerequisites

| Dependency | Version | Link |
|:-----------|:--------|:-----|
| Python | &ge; 3.10 | [python.org](https://www.python.org/downloads/) |
| PyTorch | &ge; 2.4 with CUDA | [pytorch.org](https://pytorch.org/get-started/locally/) |
| vLLM | tested with v0.10 | [docs.vllm.ai](https://docs.vllm.ai/en/latest/getting_started/installation.html) |
| Triton | tested with v3.4 | typically installed with PyTorch/vLLM |

SFI itself has no additional Python package requirements beyond what vLLM provides.

<br>

## Step 1 &ensp; Clone

```bash
git clone https://github.com/LV-NUS/SFI.git
cd SFI
```

<br>

## Step 2 &ensp; Set PYTHONPATH

SFI uses Python's `sitecustomize` mechanism to auto-patch vLLM worker processes. The SFI repo root must be on `PYTHONPATH`:

```bash
export PYTHONPATH="$(pwd):${PYTHONPATH}"
```

<details>
<summary>&ensp;<b>How the patching works</b></summary>
<br>

When `PYTHONPATH` includes the SFI repo root, Python automatically imports `sitecustomize.py` on every process start (including vLLM workers). If `VLLM_SPARSE_CONTROLLER_JSON` is set, `sitecustomize.py` calls `patches/patch_installer.py`, which monkey-patches vLLM's `unified_attention` function and creates a global `VLLMSparseController`. No vLLM source files are modified.

</details>

<br>

## Step 3 &ensp; Compile CUDA Extensions

SFI includes custom CUDA kernels that must be compiled once (~1 minute):

```bash
python -c "
from utils.bounds_kernel_ext import _require_ext as _require_bounds
from utils.selector_pipeline_ext import _require_ext as _require_pipeline
_require_bounds(); _require_pipeline()
print('CUDA extensions compiled successfully.')
"
```

> [!NOTE]
> If compilation fails, check that `CUDA_HOME` or `TORCH_CUDA_ARCH_LIST` is set correctly, and that `torch.cuda.is_available()` returns `True`. Extensions are cached after the first build.

<br>

## Step 4 &ensp; Quick Verify

Run a throughput sweep to verify the full pipeline. Download any Qwen3 model from HuggingFace:

```bash
CUDA_VISIBLE_DEVICES=0 VLLM_WORKER_MULTIPROC_METHOD=spawn \
  python benchmarks/run_sweep.py --mode sparse \
    --model <MODEL_PATH> \
    --batch-size 16 \
    --context-lengths 8192,16384 \
    --warmup-runs 1 --measure-runs 1
```

If you see a throughput summary table with `AllDecode tok/s` values, the full pipeline is working.

<br>

## Step 5 &ensp; Smoke Test (Optional)

For a more thorough check, the bs=2 parity harness runs both dense and sparse decoding and validates that both complete without error:

```bash
CUDA_VISIBLE_DEVICES=0 VLLM_WORKER_MULTIPROC_METHOD=spawn \
  python benchmarks/needle_bs2_sparse_check.py \
    --model <MODEL_PATH> \
    --prompt benchmarks/needle_prompt_two_parts.txt \
    --log-path logs/needle_smoke.jsonl \
    --smoke-test \
    --max-new-tokens 256 \
    --k-min 32 --sink 4 --recent 256 \
    --refresh-interval 64 --alpha-k-head 1536 --prefill-last-n 16
```

> [!NOTE]
> `--smoke-test` runs dense then sparse with no reference files needed. If the script prints `Smoke test passed`, both decoding paths are working correctly.

<br>

---

<br>

## Configuration

SFI is configured via the `VLLM_SPARSE_CONTROLLER_JSON` environment variable.

<details>
<summary>&ensp;<b>Sparse controller config</b></summary>

<br>

```json
{
  "enabled": true,
  "tau": 1.0,
  "k_min": 32,
  "sink": 4,
  "recent": 256,
  "refresh_interval": 64,
  "prefill_last_n_query": 16,
  "alpha_fair": {
    "k_head": 2048,
    "soft_alpha": 1.0,
    "cross_head_alpha": 0.85,
    "cross_head_temperature": 2.0,
    "prior_pos_power": 1.8,
    "prior_pos_eta": 0.4
  },
  "trigger": {
    "refresh_interval": 64,
    "enable_sentence_triggers": true,
    "min_refresh_gap": 16
  }
}
```

| Field | Description |
|:------|:------------|
| `sink` | Number of initial tokens always retained (attention sinks) |
| `recent` | Number of most-recent tokens always retained |
| `refresh_interval` | Dense refresh every N decode steps |
| `alpha_fair.k_head` | Tokens selected per attention head &mdash; controls sparsity vs. quality tradeoff. Use 4096 for LongBench quality evaluation, 1536&ndash;2048 for throughput benchmarks. |
| `alpha_fair.cross_head_alpha` | Cross-head coordination strength |
| `trigger.enable_sentence_triggers` | Enable sentence-boundary detection for refresh timing. When `true`, SFI detects sentence endings (via punctuation heuristics) and uses them as natural refresh points &mdash; improving quality since sentence boundaries often coincide with semantic shifts. |
| `trigger.min_refresh_gap` | Minimum steps between consecutive refreshes &mdash; prevents overly frequent refreshes when sentences are short. |

</details>

<details>
<summary>&ensp;<b>Required environment variables</b></summary>

<br>

```bash
export PYTHONPATH="/path/to/SFI:${PYTHONPATH}"
export VLLM_ATTENTION_BACKEND="TRITON_ATTN_VLLM_V1"
export VLLM_USE_TRITON_KERNEL="1"
```

These are set automatically by `scripts/run_longbench_sparse.sh` and `benchmarks/run_sweep.py`. You only need to set them manually when launching vLLM directly.

</details>

<br>

## LongBench Setup

1. Clone [LongBench](https://github.com/THUDM/LongBench) and configure `LongBench/config/model2path.json` with your model path.

2. Run:
   ```bash
   GPU_DEVICES=0 SPARSE_K_HEAD=4096 bash scripts/run_longbench_sparse.sh
   ```

> See `scripts/run_longbench_sparse.sh` for all configurable environment variables.

<br>

## Troubleshooting

| Problem | Solution |
|:--------|:---------|
| CUDA extensions fail to compile | Ensure `torch.cuda.is_available()` returns `True`. Set `TORCH_CUDA_ARCH_LIST` to your GPU arch (e.g., `"8.0"` for A100, `"10.0"` for B200). |
| `sitecustomize.py` not found | Verify: `python -c "import sitecustomize; print(sitecustomize.__file__)"` |
| Sparse controller not activating | Check `VLLM_SPARSE_CONTROLLER_JSON` is valid JSON. Debug: `VLLM_SPARSE_SITE_LOG=1`, then check `/tmp/vllm_sparse_site.log`. |
| OOM during benchmarks | Reduce `--batch-size` or `--max-new-tokens`. Lower `--gpu-mem-util` to 0.8. |
