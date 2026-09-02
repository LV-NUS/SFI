# Installation Guide

SFI has two distinct installation paths. Use the binary runtime for H20
deployment and the public source tree for paper reproduction or development.

<br>

| Path | Intended use | Dependency contract |
|:---|:---|:---|
| **H20 binary runtime r53** | Protected deployment package; recommended for H20 canary | H20/SM90, CPython 3.12, PyTorch 2.11.0+cu130, CUDA 13.0, vLLM 0.22.1 |
| **Research/source tree** | Reproduction, inspection, and development | Historical public stack documented below |

Do not mix commands, environment variables, wheels, or compiled caches between
these two paths.

<br>

## H20 Binary Runtime r53

The binary package contains prebuilt SM90 CUDA objects and compiled Python
extensions. It does not contain SFI Python/C/CUDA source and does not require
compiling or installing a custom SFI Triton package on the deployment host.
PyTorch or vLLM may still carry their own upstream Triton dependency.

### 1. Download and verify

```bash
curl -fLO \
  "https://github.com/LV-NUS/SFI/releases/download/sfi-runtime-h20-r53-20260902/sfi-runtime-h20-r53-20260902.tar.gz"
tar -xzf "sfi-runtime-h20-r53-20260902.tar.gz"
cd "sfi-runtime-h20-r53-20260902"

sha256sum -c "SHA256SUMS"
```

The checksum command must report `OK` for `README_CN.md`,
`RELEASE_MANIFEST.json`, and the wheel.

### 2. Install into the qualified environment

Use an isolated environment that already contains the exact dependency stack.
Do not install the wheel globally and do not use `--force-reinstall` on PyTorch
or vLLM.

```bash
SFI_VENV="/path/to/py312-vllm-0.22.1-env"

"${SFI_VENV}/bin/python" -m pip install --no-deps \
  "./sfi_runtime-1.0.53.dev20260902-cp312-cp312-linux_x86_64.whl"

env -u PYTHONPATH -u PYTHONHOME \
  PYTHONNOUSERSITE=1 \
  "${SFI_VENV}/bin/sfi-runtime" selfcheck --json
```

Continue only when `selfcheck` returns `status: PASS`. This verifies the
binary, dependency, and GPU contract; it is not a substitute for
checkpoint-specific accuracy qualification.

### 3. Select exactly one acceleration rail

Sparse prefill with dense decode:

```bash
export CUDA_VISIBLE_DEVICES=0
export SFI_ENABLE=1
export SFI_PREFILL_ACCEL=1
export SFI_DECODE_ACCEL=0
export SFI_BUDGET_PROFILE=prefill_h20_stable

env -u PYTHONPATH -u PYTHONHOME \
  PYTHONNOUSERSITE=1 \
  "${SFI_VENV}/bin/sfi-runtime" show-config --json
```

Dense prefill with SFI decode:

```bash
export CUDA_VISIBLE_DEVICES=0
export SFI_ENABLE=1
export SFI_PREFILL_ACCEL=0
export SFI_DECODE_ACCEL=1
export SFI_BUDGET_PROFILE=decode_h20_stable

env -u PYTHONPATH -u PYTHONHOME \
  PYTHONNOUSERSITE=1 \
  "${SFI_VENV}/bin/sfi-runtime" show-config --json
```

The two rails are independently qualified and must not be enabled in the same
service process.

### 4. Start the service

Set `MODEL_PATH` to any supported Qwen3 text checkpoint that fits the selected
GPU/TP deployment. The exact published r53 qualification used
`Qwen3.6-35B-A3B-FP8` on one H20.

```bash
MODEL_PATH="/models/Qwen3.6-35B-A3B-FP8"
MODEL_NAME="qwen36-35b"
PORT="8000"

exec env -u PYTHONPATH -u PYTHONHOME \
  PYTHONNOUSERSITE=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  VLLM_NO_USAGE_STATS=1 \
  "${SFI_VENV}/bin/sfi-runtime" serve -- \
    --model "${MODEL_PATH}" \
    --served-model-name "${MODEL_NAME}" \
    --port "${PORT}" \
    --max-model-len 165888 \
    --gpu-memory-utilization 0.92 \
    --max-num-seqs 8 \
    --max-num-batched-tokens 73728 \
    --long-prefill-token-threshold 8192 \
    --kv-cache-memory-bytes 32590397440 \
    --block-size 32 \
    --no-enable-prefix-caching \
    --attention-config '{"backend":"FLASH_ATTN","flash_attn_version":3}' \
    --no-async-scheduling
```

The memory and batching values above are the one-H20 qualification settings,
not universal values for every Qwen3 size. Adjusting them requires a new
capacity and performance check.

### 5. Health check and rollback

```bash
curl -fsS "http://127.0.0.1:${PORT}/health"
```

Return to dense by stopping the service and restarting with:

```bash
export SFI_ENABLE=0
export SFI_PREFILL_ACCEL=0
export SFI_DECODE_ACCEL=0
export SFI_BUDGET_PROFILE=frozen
```

Do not replace the wheel while a service process is running. The bundled
`README_CN.md` contains the frozen parameters, r53 accuracy boundary, B/A/B
speed results, log checks, and binary-protection boundary.

<br>

## Supported Qwen3 Models

The H20 runtime supports the following text-only causal-LM architectures.
“Supported” means runtime-compatible; it does not mean that every checkpoint
inherits the exact r53 H20 accuracy and speed numbers.

| Runtime architecture | Supported checkpoints |
|:---|:---|
| `Qwen3ForCausalLM` | Qwen3-0.6B, 1.7B, 4B, 8B, 14B, 32B; Qwen3-4B-Instruct-2507; Qwen3-4B-Thinking-2507 |
| `Qwen3MoeForCausalLM` | Qwen3-30B-A3B and Qwen3-235B-A22B, including Instruct-2507 and Thinking-2507; Qwen3-Coder-30B-A3B-Instruct; Qwen3-Coder-480B-A35B-Instruct |
| `Qwen3NextForCausalLM` | Qwen3-Next-80B-A3B-Instruct; Qwen3-Next-80B-A3B-Thinking |
| `Qwen3_5MoeForConditionalGeneration` (hybrid) | Qwen3.6-35B-A3B-FP8 |

- **Exact r53/H20 qualification:** Qwen3.6-35B-A3B-FP8.
- **Existing project evaluation:** Qwen3-4B, Qwen3-30B-A3B,
  Qwen3-235B-A22B and their evaluated Thinking configurations.
- **Other models in the table:** supported through the same runtime
  architecture, but require model-specific ACC and speed validation before
  production use.

Excluded from the current support claim: Qwen3-VL/Omni/Audio,
Embedding/Reranker, multimodal requests, and non-Qwen3 architectures. Large
MoE/Coder/Next checkpoints may require multiple GPUs; TP topology and memory
capacity are separate from model-architecture support.

<br>

---

<br>

## Research / Source Installation

The public source tree integrates into an existing vLLM installation via
runtime patching. No vLLM source modification is required. This is the
historical research path, not the r53 H20 binary contract.

### Source prerequisites

| Dependency | Version | Link |
|:-----------|:--------|:-----|
| Python | &ge; 3.10 | [python.org](https://www.python.org/downloads/) |
| PyTorch | &ge; 2.4 with CUDA | [pytorch.org](https://pytorch.org/get-started/locally/) |
| vLLM | tested with v0.10 | [docs.vllm.ai](https://docs.vllm.ai/en/latest/getting_started/installation.html) |
| Triton | tested with v3.4 | typically installed with PyTorch/vLLM |

SFI itself has no additional Python package requirements beyond what vLLM provides.

<br>

### Step 1 &ensp; Clone

```bash
git clone https://github.com/LV-NUS/SFI.git
cd SFI
```

<br>

### Step 2 &ensp; Set PYTHONPATH

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

### Step 3 &ensp; Compile CUDA Extensions

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

### Step 4 &ensp; Quick Verify

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

### Step 5 &ensp; Smoke Test (Optional)

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

## Research Runtime Configuration

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

## Research LongBench Setup

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
