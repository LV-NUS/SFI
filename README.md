<div align="center">

<br>

# Slow-Fast Inference (SFI)

**Training-Free Inference Acceleration via Within-Sentence Support Stability**

<br>

<a href="https://arxiv.org/abs/2603.12038"><img src="https://img.shields.io/badge/arXiv-2603.12038-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white" alt="Paper"></a>&nbsp;&nbsp;
<a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue?style=for-the-badge" alt="License"></a>&nbsp;&nbsp;
<a href="https://github.com/vllm-project/vllm"><img src="https://img.shields.io/badge/H20_Runtime-vLLM_0.22.1-blueviolet?style=for-the-badge" alt="H20 runtime vLLM 0.22.1"></a>&nbsp;&nbsp;
<a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/H20_Runtime-Python_3.12-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="H20 runtime Python 3.12"></a>

<br><br>

<a href="#overview">Overview</a>&ensp;&middot;&ensp;
<a href="#demo">Demo</a>&ensp;&middot;&ensp;
<a href="#method">Method</a>&ensp;&middot;&ensp;
<a href="#results">Results</a>&ensp;&middot;&ensp;
<a href="#supported-models">Models</a>&ensp;&middot;&ensp;
<a href="#installation">Installation</a>&ensp;&middot;&ensp;
<a href="#running-experiments">Experiments</a>&ensp;&middot;&ensp;
<a href="#citation">Citation</a>

<br><br>

<img src="assets/motivation_v5.png" width="72%" alt="SFI teaser" />

<br><br>

**Up to 14.4&times; end-to-end decode speedup &mdash; training-free, applied to existing checkpoints.**

<br>

</div>

## Overview

> *Do not pay the cost of full-history attention at every step when the model's useful support has not meaningfully changed.*

**Slow-Fast Inference (SFI)** accelerates long-context autoregressive decoding by exploiting the observation that **attention support often evolves more slowly than token generation**. Within a sentence or short coherent span, the set of critical tokens tends to remain stable rather than changing abruptly at every step.

Based on this, SFI decouples decoding into two paths:

| | Fast Step | Slow Step |
|:---:|:---|:---|
| **What** | Reuse compact sparse memory | Run dense full attention |
| **When** | Most steps (cheap) | Semantic boundaries / refresh budget (occasional) |
| **Why** | Support hasn't changed | Time to refresh the sparse support |

A training-free **Selector** converts dense-attention evidence from slow steps into reusable sparse memory for subsequent fast steps. SFI requires **no retraining**, works with existing checkpoints, and combines **asynchronous maintenance** with a **memory-coalesced sparse kernel** to translate algorithmic sparsity into real wall-clock gains.

<br>

## Demo

<div align="center">

https://github.com/user-attachments/assets/2b4277f9-72ee-4ff2-bf4d-442bb58a1ba9

</div>

<br>

## Method

<div align="center">
<img src="assets/method_new4.png" width="72%" alt="SFI method overview" />
</div>

<br>

1. **Fast path** &mdash; most steps attend only to a managed sparse state: **sink tokens** + **selected tokens** + **recent tokens**
2. **Slow path** &mdash; at semantic boundaries or when a refresh budget is exhausted, a dense refresh step runs full attention
3. **Selector update** &mdash; dense-attention evidence from the slow step is converted into sparse support for the next fast-step segment

The majority of decoding steps take the fast path. Dense attention is reserved for moments when support is likely to shift.

<details>
<summary>&ensp;<b>System design</b></summary>

<br>

<div align="center">
<img src="assets/infra_system.png" width="72%" alt="SFI system design" />
</div>

<br>

- **Asynchronous slow-step maintenance** &mdash; selector execution and cache reorganization overlap with the decoding stream
- **Memory-coalesced sparse kernel** &mdash; packs reusable long-range KV entries into a compact segment; recent tokens are read in place

</details>

<br>

## Results

SFI achieves **1.6&ndash;14.4&times; end-to-end decode speedup** while **preserving quality close to dense full attention** across long-context and long-CoT workloads.

<details open>
<summary>&ensp;<b>End-to-end throughput</b></summary>

<br>

<div align="center">
<img src="assets/speed.png" width="72%" alt="SFI throughput" />
</div>

<br>

Representative speedups at 128K context: &ensp; **Qwen3-4B** 14.36&times; &ensp;&middot;&ensp; **Qwen3-30B-A3B** 11.98&times; &ensp;&middot;&ensp; **Qwen3-235B-A22B** 13.49&times;

</details>

<details>
<summary>&ensp;<b>Kernel-level speedup</b>&ensp;<sub>KV length 16K &middot; batch=16 &middot; bf16</sub></summary>

<br>

| Retention (%) | 1.6 | 6.3 | 12.5 | 25.0 | 37.5 | 50.0 | 75.0 | 98.4 | 100 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Speedup** | **10.67&times;** | **9.56&times;** | **7.15&times;** | **3.96&times;** | **2.75&times;** | **2.10&times;** | **1.43&times;** | 1.10&times; | 1.00&times; |

</details>

<details>
<summary>&ensp;<b>LongBench-V1 quality</b>&ensp;<sub>17 tasks &middot; 3 model scales</sub></summary>

<br>

| Category | Task | 4B Slow | 4B **SFI** | 30B-A3B Slow | 30B-A3B **SFI** | 235B-A22B Slow | 235B-A22B **SFI** |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| *Single-Doc QA* | Qasper | 40.60 | **44.20** | 38.96 | **42.14** | **45.77** | 44.67 |
| | MultiFieldQA-en | 46.56 | **49.31** | 50.32 | **53.22** | **51.21** | 50.92 |
| | MultiFieldQA-zh | 61.21 | **63.81** | 63.42 | **66.00** | **67.08** | 66.37 |
| *Multi-Doc QA* | HotpotQA | 55.34 | **59.00** | 61.68 | **63.37** | 67.13 | **67.65** |
| | 2WikiMQA | 42.02 | **44.50** | 54.68 | **55.98** | 64.14 | **65.31** |
| | MuSiQue | 24.79 | **25.76** | **32.22** | 31.95 | 42.86 | **43.44** |
| | DuReader | 21.85 | **23.77** | 21.40 | **23.44** | **25.38** | 24.58 |
| *Summarization* | GovReport | 27.87 | **29.72** | 29.40 | **30.18** | **31.68** | 31.28 |
| | QMSum | 22.11 | **22.21** | 21.66 | **21.92** | **22.81** | 22.72 |
| | MultiNews | **24.06** | 24.04 | **23.52** | 23.46 | **23.46** | 23.33 |
| *Few-Shot* | TREC | 73.00 | **75.00** | 77.50 | **78.50** | **77.50** | **77.50** |
| | TriviaQA | 85.29 | **85.62** | **91.56** | 91.06 | 91.86 | **92.10** |
| | SAMSum | 39.12 | **39.99** | 39.12 | **39.72** | 41.09 | **41.30** |
| | LSHT | 30.25 | **37.75** | 42.50 | **47.25** | 51.00 | **52.00** |
| *Synthetic & Code* | PassageRet-en | **100.0** | **100.0** | **100.0** | **100.0** | **100.0** | **100.0** |
| | LCC | **4.48** | 4.32 | 24.82 | **25.13** | **61.32** | 61.10 |
| | RepoBench-P | 5.17 | **5.28** | 24.76 | **24.92** | 62.56 | **63.18** |
| **Average** | | 41.40 | **43.19** | 46.91 | **48.13** | 54.52 | **54.56** |

SFI matches or improves full-KV decoding on most subsets, with the clearest average gains at 4B (+1.8) and 30B-A3B (+1.2).

</details>

<details>
<summary>&ensp;<b>LongBench-V2 quality</b></summary>

<br>

| Model | Method | Overall | Easy | Hard | Short | Medium | Long |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Qwen3-4B | Slow | 34.2 | 37.5 | 32.2 | **35.0** | 33.5 | **34.3** |
| | **SFI** | **34.8** | **38.5** | **32.5** | **35.0** | **34.9** | **34.3** |
| Qwen3-30B-A3B | Slow | **35.1** | **36.0** | **34.6** | **38.5** | **32.8** | **28.6** |
| | **SFI** | **35.1** | **36.0** | **34.6** | **38.5** | **32.8** | **28.6** |
| Qwen3-235B-A22B | Slow | **46.0** | **50.0** | **43.7** | **48.0** | **44.1** | 47.6 |
| | **SFI** | **46.0** | **50.0** | **43.7** | 47.5 | **44.1** | **52.4** |

SFI remains stable across scales, with the strongest gain on the longest-context subset (+4.8 on Qwen3-235B-A22B).

</details>

<details>
<summary>&ensp;<b>Long-CoT reasoning</b>&ensp;<sub>GPQA &amp; MMLU</sub></summary>

<br>

| Model | GPQA Slow | GPQA **SFI** | MMLU Slow | MMLU **SFI** |
|:---|:---:|:---:|:---:|:---:|
| Qwen3-4B-Thinking | **64.14** | 63.70 | **63.00** | **63.00** |
| Qwen3-30B-A3B-Thinking | 69.70 | **71.21** | **70.90** | 70.60 |
| Qwen3-235B-A22B-Thinking | **80.80** | **80.80** | 90.09 | **90.30** |

SFI matches the full-KV baseline at medium and large scales, with only minor variation at 4B.

</details>

<br>

## Supported Models

SFI supports the **text-only causal-LM members of the Qwen3 family**. Runtime
support and benchmark qualification are reported separately: a compatible
checkpoint can use the runtime, but it does not inherit accuracy or speed
numbers measured on another checkpoint.

| Runtime architecture | Supported checkpoints |
|:---|:---|
| `Qwen3ForCausalLM` | Qwen3-0.6B, 1.7B, 4B, 8B, 14B, 32B; Qwen3-4B-Instruct-2507; Qwen3-4B-Thinking-2507 |
| `Qwen3MoeForCausalLM` | Qwen3-30B-A3B and Qwen3-235B-A22B, including their Instruct-2507 and Thinking-2507 variants; Qwen3-Coder-30B-A3B-Instruct; Qwen3-Coder-480B-A35B-Instruct |
| `Qwen3NextForCausalLM` | Qwen3-Next-80B-A3B-Instruct; Qwen3-Next-80B-A3B-Thinking |
| `Qwen3_5MoeForConditionalGeneration` (hybrid) | Qwen3.6-35B-A3B-FP8 |

Evidence levels:

- **r53 exact-wheel H20 qualification:** Qwen3.6-35B-A3B-FP8. The r53
  LongBench V2 and H20 B/A/B numbers apply only to this checkpoint.
- **Existing project evaluation:** Qwen3-4B, Qwen3-30B-A3B,
  Qwen3-235B-A22B and their evaluated Thinking configurations.
- **Architecture-compatible:** the remaining checkpoints in the table share a
  supported attention/runtime contract, but still require checkpoint-specific
  accuracy and speed qualification before production deployment.

Qwen3-VL, Omni, Audio, Embedding, Reranker, multimodal requests, and non-Qwen3
architectures are not included in this release's support claim. Model size,
GPU count, tensor parallelism, and available KV-cache memory remain deployment
capacity constraints; r53 performance qualification used one NVIDIA H20 96GB.

<br>

## Installation

### H20 binary runtime (recommended for deployment)

The source-free r53 wheel is available from the
**[SFI Runtime H20 r53 release](https://github.com/LV-NUS/SFI/releases/tag/sfi-runtime-h20-r53-20260902)**.
It contains prebuilt SM90 CUDA objects and compiled Python extensions; no SFI
source build or custom SFI Triton package is required on the deployment host.

| Component | Required version |
|:---|:---|
| GPU | NVIDIA H20 / SM90 |
| Python | CPython 3.12 |
| PyTorch | 2.11.0+cu130 |
| PyTorch CUDA | 13.0 |
| vLLM | 0.22.1 |

```bash
# Download and unpack the single r53 delivery asset.
curl -fLO \
  "https://github.com/LV-NUS/SFI/releases/download/sfi-runtime-h20-r53-20260902/sfi-runtime-h20-r53-20260902.tar.gz"
tar -xzf "sfi-runtime-h20-r53-20260902.tar.gz"
cd "sfi-runtime-h20-r53-20260902"

# Verify the README, manifest, and wheel.
sha256sum -c "SHA256SUMS"

# Install into an isolated environment that already contains the exact
# PyTorch/vLLM stack above. Do not install globally.
SFI_VENV="/path/to/py312-vllm-0.22.1-env"
"${SFI_VENV}/bin/python" -m pip install --no-deps \
  "./sfi_runtime-1.0.53.dev20260902-cp312-cp312-linux_x86_64.whl"

# The runtime fails closed if the binary, dependency, or GPU contract differs.
env -u PYTHONPATH -u PYTHONHOME \
  PYTHONNOUSERSITE=1 \
  "${SFI_VENV}/bin/sfi-runtime" selfcheck --json
```

> [!NOTE]
> Acceleration is disabled by default. Enable exactly one qualified rail:
> `prefill_h20_stable` (sparse prefill + dense decode) or
> `decode_h20_stable` (dense prefill + SFI decode). The two rails must not be
> enabled in the same service process.

Complete activation, serving, health-check, rollback, accuracy, and speed
instructions are in **[docs/INSTALL.md](docs/INSTALL.md)** and in the
`README_CN.md` bundled with the release.

<details>
<summary>&ensp;<b>Research/source installation</b></summary>

<br>

The public source tree remains available for paper reproduction and
development. This path is separate from the protected H20 binary runtime and
uses its own historical dependency matrix.

```bash
git clone https://github.com/LV-NUS/SFI.git
cd SFI
export PYTHONPATH="$(pwd):${PYTHONPATH}"

python -c "
from utils.bounds_kernel_ext import _require_ext as _require_bounds
from utils.selector_pipeline_ext import _require_ext as _require_pipeline
_require_bounds(); _require_pipeline()
print('CUDA extensions compiled successfully.')
"

CUDA_VISIBLE_DEVICES=0 VLLM_WORKER_MULTIPROC_METHOD=spawn \
  python benchmarks/run_sweep.py --mode sparse \
    --model <MODEL_PATH> \
    --batch-size 16 \
    --context-lengths 8192,16384 \
    --warmup-runs 1 --measure-runs 1
```

</details>

<details>
<summary>&ensp;<b>How SFI integrates into vLLM</b></summary>

<br>

SFI patches vLLM at runtime &mdash; no source modification required.

```
Python startup
     │
     ▼
┌────────────────────────┐
│   sitecustomize.py     │  Auto-imported on every Python process start
└──────────┬─────────────┘
           │  reads VLLM_SPARSE_CONTROLLER_JSON
           ▼
┌────────────────────────┐
│   patch_installer.py   │  Monkey-patches vLLM's unified_attention
└──────────┬─────────────┘
           │  creates & registers
           ▼
┌────────────────────────┐
│ VLLMSparseController   │  Orchestrates sparse decode / refresh / selection
└────────────────────────┘
```

1. `PYTHONPATH` includes SFI root &rarr; Python finds `sitecustomize.py` on startup
2. `VLLM_SPARSE_CONTROLLER_JSON` env var triggers patch installation
3. `patch_installer.py` patches `unified_attention` and creates a global controller
4. During decoding, the controller routes each layer through fast (sparse) or slow (dense refresh) path

</details>

<details>
<summary>&ensp;<b>Repository structure</b></summary>

```
SFI/
├── patches/                         # Core SFI runtime (~25k lines)
│   ├── vllm_sparse_patch.py         #   Main controller: VLLMSparseController
│   ├── patch_installer.py           #   vLLM monkey-patch installer
│   ├── controller_mixins/           #   6 behavior mixins (selector, refresh, profile, ...)
│   ├── decode_runtime/              #   Decode-step workers
│   ├── refresh_runtime/             #   Refresh-step workers
│   ├── selector_runtime/            #   Selector workers (alpha-fair selection)
│   └── ARCHITECTURE.md              #   Detailed architecture doc
├── triton_kernel/                   # Triton kernel implementations
│   ├── flash_attn_score_dump_fwd.py #   Sparse FlashAttention with score dump
│   └── alpha_selector_kernel.py     #   Alpha-fair selector + cross-head mutex
├── hybrid_selectors/                # Selector configuration
├── utils/                           # CUDA extensions and utilities
├── benchmarks/                      # Performance benchmarking
│   ├── run_sweep.py                 #   End-to-end throughput sweep
│   ├── bench_compact_dense_perf.py  #   Kernel-level benchmark
│   └── needle_bs2_sparse_check.py   #   Regression / smoke test (bs=2 parity)
├── scripts/                         # LongBench experiment scripts
├── LongBench/                       # Long-context evaluation
├── docs/INSTALL.md                  # Detailed installation guide
└── sitecustomize.py                 # Auto-patches vLLM on Python startup
```

</details>

<br>

## Running Experiments

### 1.&ensp;Throughput Sweep

```bash
# Dense baseline
CUDA_VISIBLE_DEVICES=0 VLLM_WORKER_MULTIPROC_METHOD=spawn \
  python benchmarks/run_sweep.py --mode dense \
    --model <MODEL_PATH> \
    --batch-size 16 \
    --context-lengths 8192,16384,32768,65536,131072

# Sparse (SFI)
CUDA_VISIBLE_DEVICES=0 VLLM_WORKER_MULTIPROC_METHOD=spawn \
  python benchmarks/run_sweep.py --mode sparse \
    --model <MODEL_PATH> \
    --batch-size 16 \
    --context-lengths 8192,16384,32768,65536,131072
```

> Results: `benchmarks/sweep_{mode}_results.json` &ensp;&middot;&ensp; Full reference: [`benchmarks/SWEEP_BENCHMARK.md`](benchmarks/SWEEP_BENCHMARK.md)

### 2.&ensp;Kernel-level Benchmark

```bash
CUDA_VISIBLE_DEVICES=0 python benchmarks/bench_compact_dense_perf.py \
  --num-layers 36 --num-kv-heads 8 --head-dim 128 \
  --context-lengths 8192,16384,32768,65536,131072
```

### 3.&ensp;LongBench Evaluation

```bash
# Sparse (k_head=4096 for long-context quality)
GPU_DEVICES=0 SPARSE_K_HEAD=4096 bash scripts/run_longbench_sparse.sh

# Dense baseline
bash scripts/run_longbench_baseline.sh
```

> See `scripts/run_longbench_sparse.sh` for all configurable parameters.

<br>

## TODO

- [x] SFI framework with vLLM + Triton kernel
- [ ] FlashAttention / CUDA kernel backend integration
- [ ] SGLang backend support
- [ ] Code release (pending external evaluation)

<br>

## Citation

```bibtex
@article{xie2026slow,
  title   = {Slow-Fast Inference: Training-Free Inference Acceleration
             via Within-Sentence Support Stability},
  author  = {Xie, Xingyu and Yu, Zhaochen and Liao, Yue and
             Wang, Tao and Toh, Kim-Chuan and Yan, Shuicheng},
  journal = {arXiv preprint arXiv:2603.12038},
  year    = {2026}
}
```

## Contact

For questions, please open an [issue](https://github.com/LV-NUS/SFI/issues) or email [xyxie@pku.edu.cn](mailto:xyxie@pku.edu.cn).
