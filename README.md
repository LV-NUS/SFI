<div align="center">

<br>

# Slow-Fast Inference (SFI)

**Training-Free Inference Acceleration via Within-Sentence Support Stability**

<br>

<a href="https://arxiv.org/abs/2603.12038"><img src="https://img.shields.io/badge/arXiv-2603.12038-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white" alt="Paper"></a>&nbsp;&nbsp;
<a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue?style=for-the-badge" alt="License"></a>&nbsp;&nbsp;
<a href="https://github.com/vllm-project/vllm"><img src="https://img.shields.io/badge/vLLM-v1_engine-blueviolet?style=for-the-badge" alt="vLLM"></a>&nbsp;&nbsp;
<a href="https://github.com/vllm-project/flash-attention"><img src="https://img.shields.io/badge/FlashAttention-3_%2F_4-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="FlashAttention"></a>&nbsp;&nbsp;
<a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-%E2%89%A53.10-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"></a>

<br><br>

<a href="#quick-start">Quick Start</a>&ensp;&middot;&ensp;
<a href="#overview">Overview</a>&ensp;&middot;&ensp;
<a href="#supported-hardware">Hardware</a>&ensp;&middot;&ensp;
<a href="#installation">Installation</a>&ensp;&middot;&ensp;
<a href="#testing--benchmarking">Testing</a>&ensp;&middot;&ensp;
<a href="#adapting-to-your-machine">Your Machine</a>&ensp;&middot;&ensp;
<a href="#results">Results</a>&ensp;&middot;&ensp;
<a href="#citation">Citation</a>

<br><br>

<img src="assets/motivation_v5.png" width="72%" alt="SFI teaser" />

<br><br>

**⚡ This is the `cuda-kernel` branch — SFI on natively patched FlashAttention-3/4 CUDA kernels,<br>with full-CUDA-graph decode and fully asynchronous sparse-memory maintenance.**

<br>

</div>

> [!IMPORTANT]
> **Which branch do I want?**
>
> | Branch | Kernel backend | Best for |
> |:--|:--|:--|
> | **`cuda-kernel`** (this) | patched FlashAttention‑3 / FlashAttention‑4 CUDA kernels, full CUDA graph | maximum performance on A100 / H100 / B200 |
> | [`triton-kernel`](https://github.com/LV-NUS/SFI/tree/triton-kernel) | Triton kernels | portability, quick experimentation |

<br>

## Quick Start

Four commands from zero to a verified, benchmarked install. Every script
prints an explicit **PASS / FAIL verdict** — nothing needs eyeballing.

```bash
# 1. Clone this branch
git clone -b cuda-kernel https://github.com/LV-NUS/SFI.git && cd SFI

# 2. Build the patched FlashAttention kernels
#    (clones the pinned upstream, applies kernel_patches/, builds ~25-40 min)
bash scripts/setup_flash_attention.sh              # A100/H100 (FA3)
#   bash scripts/setup_flash_attention.sh --with-fa4    # B200 (adds FA4 CuTe)

# 3. Verify end-to-end (production gate: kernel-route proof + producer + lifecycle)
bash scripts/run_one_shot.sh 0 /path/to/any-qwen3-model
#    expected tail:  gate_passed=True producer_gate_passed=True decode_tps=...
#                    ONE-SHOT PASS

# 4. Measure throughput on a pre-tuned tier (exclusive GPU!)
bash scripts/run_speed.sh 0 /path/to/model bs8x12k sparse   # SFI
bash scripts/run_speed.sh 0 /path/to/model bs8x12k dense    # your local baseline
#    expected tail:  decode_tps=...  SPEED RUN OK
```

> [!TIP]
> Online serving and LongBench evaluation are two more one-liners — see
> [Testing & Benchmarking](#testing--benchmarking). Detailed install matters
> (prerequisites, env vars, memory budgeting, troubleshooting) live in
> **[docs/INSTALL.md](docs/INSTALL.md)**.

<br>

## Overview

> *Do not pay the cost of full-history attention at every step when the model's useful support has not meaningfully changed.*

**Slow-Fast Inference (SFI)** accelerates long-context autoregressive decoding by exploiting the observation that **attention support often evolves more slowly than token generation**. Within a sentence or short coherent span, the set of critical tokens tends to remain stable rather than changing abruptly at every step.

Based on this, SFI decouples decoding into two paths:

| | Fast Step | Slow Step |
|:---:|:---|:---|
| **What** | Attend to compact sparse memory | Run dense full attention |
| **When** | Most steps (cheap) | Sentence boundaries / refresh budget (occasional) |
| **Why** | Support hasn't changed | Time to refresh the sparse support |

A training-free **Selector** converts dense-attention evidence from slow steps into reusable sparse memory for subsequent fast steps. SFI requires **no retraining** and works with existing checkpoints.

**What the CUDA kernel edition adds** over the algorithm itself:

- **Native kernels** — the fast path runs a patched FlashAttention forward
  that reads a *compact, memory-coalesced KV segment* plus in-place recent
  tokens (mixed-page dual-source attention), not a masked dense kernel.
- **Full-CUDA-graph decode** — every decode step replays one captured graph;
  sparse bookkeeping never breaks graph capture.
- **Fully asynchronous slow path** — refresh attention, selection, and
  compact-KV rebuild run on side streams/workers and overlap decoding;
  outputs are deterministic functions of each step's submitted state
  (event-ordered, race-audited).
- **Deterministic by construction** — repeated runs produce identical
  token streams; the one-shot gate checks this class of invariants.

<details>
<summary>&ensp;<b>Method &amp; system design</b></summary>

<br>

<div align="center">
<img src="assets/method_new4.png" width="72%" alt="SFI method overview" />
</div>

<br>

1. **Fast path** — most steps attend only to a managed sparse state: **sink tokens** + **selected tokens** (compact segment) + **recent tokens** (in place)
2. **Slow path** — at sentence boundaries or when the refresh budget is exhausted, a dense refresh step runs full attention
3. **Selector update** — dense-attention evidence is converted into the sparse support for the next fast-step segment, asynchronously

<div align="center">
<img src="assets/infra_system.png" width="72%" alt="SFI system design" />
</div>

</details>

<details>
<summary>&ensp;<b>Demo</b></summary>

<br>

<div align="center">

https://github.com/user-attachments/assets/2b4277f9-72ee-4ff2-bf4d-442bb58a1ba9

</div>

</details>

<br>

## Supported Hardware

| GPU | Arch | Kernel | Build artifact | Status |
|:--|:--:|:--|:--|:--|
| A100-class Ampere | SM80 | **FA3** (C++/CUTLASS) | prebuilt `.so` | ✅ fully validated end-to-end (correctness gate, determinism, throughput) |
| H100 / H800 Hopper | SM90 | **FA3** (C++/CUTLASS) | prebuilt `.so` | ✅ kernel port complete, compile- and resource-verified |
| B200 Blackwell | SM100 | **FA4** (CuTe DSL) | runtime JIT | ✅ supported — throughput-validated on B200; CuTe kernel patch + benchmarks included |

Architecture selection is two values: `setup_flash_attention.sh` auto-detects
the build-time `TORCH_CUDA_ARCH_LIST`, and the run scripts default the
run-time `VLLM_FLASH_ATTN_VERSION` to `3` (set `4` on SM100). See
[Adapting to your machine](#adapting-to-your-machine).

<br>

## Installation

The short version (details, prerequisites, and every env var:
**[docs/INSTALL.md](docs/INSTALL.md)**):

1. **Clone** this branch (`git clone -b cuda-kernel ...`).
2. **Build kernels**: `bash scripts/setup_flash_attention.sh`
   — clones [vllm-project/flash-attention](https://github.com/vllm-project/flash-attention)
   at the pinned base commit, applies
   [`kernel_patches/`](kernel_patches/README.md), builds
   `_vllm_fa3_C*.so` in-tree (~25–40 min). FA4/SM100 kernels are Python and
   JIT-compile at runtime instead.
3. **Verify**: `bash scripts/run_one_shot.sh 0 <model>` → `ONE-SHOT PASS`.

<details>
<summary>&ensp;<b>How SFI integrates into vLLM</b>&ensp;<sub>runtime patching — no vLLM source modification</sub></summary>

<br>

```
Python process start
      │
      ▼
┌──────────────────────────────┐   PYTHONPATH includes the SFI root, so Python
│       sitecustomize.py       │   auto-imports this in every process —
└──────────────┬───────────────┘   including vLLM's spawned workers
               │  VLLM_SPARSE_FA3_UPSTREAM_ROOT set?
               ▼
┌──────────────────────────────┐   loads the patched flash-attention clone and
│  patches/fa3_native/install  │   bridges its kernels into vLLM's flash-attn
└──────────────┬───────────────┘   interface: FA3 → prebuilt _vllm_fa3_C*.so,
               │                    FA4 → CuTe DSL kernels (JIT, cached)
               │  VLLM_SPARSE_CONTROLLER_JSON set?
               ▼
┌──────────────────────────────┐   monkey-patches vLLM's attention entry point
│   patches/patch_installer    │   and the v1 model-runner step hooks
└──────────────┬───────────────┘   (_prepare_inputs / _dummy_run /
               │                    _update_states); creates the controller
               ▼
┌──────────────────────────────┐   per step: fast path (compact KV, one CUDA
│    VLLMSparseController      │   graph replay) or slow path (dense refresh +
└──────────────────────────────┘   async selector / compact-KV rebuild)
```

Each stage is gated by its own env var: without
`VLLM_SPARSE_CONTROLLER_JSON` the sparse controller never installs and vLLM
runs its normal dense path (still on the patched FlashAttention build when
the kernel bridge is active); without both, the process is vanilla vLLM.

</details>

<details>
<summary>&ensp;<b>Repository structure</b></summary>

```
SFI/  (branch: cuda-kernel)
├── kernel_patches/                  # ★ FlashAttention kernel patches + apply/build guide
│   ├── sfi_fa3_sm80_sm90.patch      #   FA3: mixed-page dual-source compact-KV forward
│   └── sfi_fa4_sm100_cute.patch     #   FA4 CuTe (SM100), stacks on the FA3 patch
├── patches/                         # ★ SFI runtime (~80k lines): controller, dispatch,
│   ├── patch_installer.py           #   route authority, deterministic async workers
│   ├── vllm_sparse_patch.py
│   ├── fa3_native/                  #   vendored-kernel bridge + install
│   ├── decode_runtime/  refresh_runtime/  selector_runtime/  fa_sparse_runtime/
│   └── controller_mixins/
├── scripts/
│   ├── setup_flash_attention.sh     #   clone + patch + build kernels
│   ├── run_one_shot.sh              #   correctness gate (PASS/FAIL)
│   ├── run_speed.sh                 #   throughput tiers (pre-tuned recipes)
│   ├── serve_sparse.sh              #   OpenAI-compatible server with SFI
│   ├── run_longbench_v2.sh          #   LongBench v2 against the server
│   └── make_context_corpus.py       #   benchmark corpus generator
├── benchmarks/                      # offline runners, e2e/kernel benches, corpora
├── utils/                           # JIT CUDA helper extensions (selector/bounds)
├── hybrid_selectors/                # alpha-fair selector
├── triton_kernel/                   # selector-side Triton kernels
├── sitecustomize.py                 # auto-injection entry point
└── docs/INSTALL.md                  # detailed install / config / troubleshooting
```

</details>

<br>

## Testing & Benchmarking

Three test axes, one script each. All scripts are self-judging (exit code +
printed verdict) and write their artifacts under `out/`.

### 1&ensp;·&ensp;Correctness — one-shot gated e2e

```bash
bash scripts/run_one_shot.sh <GPU> <MODEL>                      # production gate
bash scripts/run_one_shot.sh <GPU> <MODEL> tag --with-reference # + dense compare
```

Runs the production pipeline (full-CUDA-graph decode, async bootstrap +
refresh, selector) on a built-in 2×long-context preset. The gate verifies:
child exit, **kernel route proof** (`route=mixed_page_attn_varlen_func` in the
trace — the sparse kernel really ran), producer contract, output health, and
lifecycle invariants. **Expect `ONE-SHOT PASS`.**

`--with-reference` additionally runs a dense pass and prints an answer-level
sparse-vs-dense comparison for inspection. The comparator is strict
(boxed-answer, else normalized full text), so chain-of-thought outputs rarely
match verbatim even when the answers agree — it informs, it doesn't gate.

### 2&ensp;·&ensp;Speed — pre-tuned throughput tiers

```bash
bash scripts/run_speed.sh <GPU> <MODEL> bs8x12k sparse
bash scripts/run_speed.sh <GPU> <MODEL> bs8x12k dense     # same-tier baseline
```

| Tier | batch × context | KV pool | `--max-model-len` |
|:--|:--:|:--:|:--:|
| `bs8x12k` | 8 × 12k | 18 GiB | 16384 |
| `bs8x16k` | 8 × 16k | 22 GiB | 20480 |
| `bs4x24k` | 4 × 24k | 16 GiB | 28672 |
| `bs2x30k` | 2 × 30k | 16 GiB | 36864 |

In sparse mode the KV pool must hold the **full KV plus the compact-page
lease**: `KVB ≥ batch × (ctx + max_new) × KV-bytes/token + slots ×
blocks/slot × 16 × KV-bytes/token`. An undersized pool doesn't crash — the
vLLM scheduler silently serializes the batch and throughput roughly halves;
`run_speed.sh` preflights this and warns.

Any tier scales via env overrides (`BS CTX KVB MML CORPUS`), and the corpus is
generated automatically (tokenizer-measured `Context:` segments; see
`scripts/make_context_corpus.py`). Judge by the printed `decode_tps`, full
decode length, refresh counts, zero fallbacks — the script checks all of this.
**Throughput requires an exclusive idle GPU.**

The script also prints `all_decode_tps` — throughput over the steady window
that starts once **every** request has finished its chunked prefill. On
long-context / large-batch tiers the head of the decode window interleaves
with prefill of the later requests (both modes pay it equally; at 8 × 32k it
is ~80% of wall time), which dilutes `decode_tps` toward 1×. `decode_tps`
stays honest about end-to-end latency; `all_decode_tps` is the fair
steady-state decode comparison.

Reference decode throughput measured with these exact scripts — one
A100-40GB, Qwen3-4B, `max_new=256`, `k_head=1536`, sparse and dense
paired on the same tree and day:

| batch × context | **SFI** (tok/s) | dense (tok/s) | speedup |
|:--:|:--:|:--:|:--:|
| 8 × 12k | **256.1** | 170.5 | **1.50×** |
| 8 × 16k | **187.7** | 127.7 | **1.47×** |
| 4 × 24k | **116.8** | 80.9 | **1.44×** |
| 2 × 30k | **95.1** | 64.1 | **1.48×** |
| 1 × 64k | **73.2** | 59.0 | 1.24× |
| 1 × 96k | **67.3** | 46.9 | **1.43×** |

On this small model / single 40 GB card, decode time is weight-bandwidth-heavy
and the attention share is modest — these ratios are the *floor* of what SFI
delivers. The speedup grows with the attention share of the step (larger
batches × longer contexts, larger-KV models): kernel-level gains at low
retention reach ~10× (table in [Results](#results)), and the paper's 128K
end-to-end runs reach up to 14×.

**Multi-GPU (tensor parallel).** `TP=N` plus a GPU list runs any tier across
multiple cards — including contexts that don't fit a single card's memory
(e.g. 2 × 128k on two 40 GB cards):

```bash
TP=2 BS=2 CTX=128000 KVB=20401094656 MML=132096 \
  bash scripts/run_speed.sh "0,1" <MODEL> bs2x30k sparse   # KVB is per GPU
```

The runner applies the required TP engine settings automatically; see
[docs/INSTALL.md](docs/INSTALL.md#adapting-to-your-machine) for details.

### 3&ensp;·&ensp;Quality — LongBench over an SFI server

```bash
# terminal 1: serve with quality settings (k_head=4096)
K_HEAD=4096 MML=65536 bash scripts/serve_sparse.sh <GPU> <MODEL> 8000

# terminal 2: official LongBench v2 harness against the server
LONGBENCH_ROOT=/path/to/LongBench bash scripts/run_longbench_v2.sh <MODEL_NAME> 4
```

`serve_sparse.sh` is also the production serving entry point: it assembles the
controller JSON and the full env chain, and starts an OpenAI-compatible
`vllm serve` with full CUDA graph. Proof that the sparse path is live:
`/tmp/vllm_sparse_site.log` (patch install) and
`VLLM_SPARSE_FA3_ROUTE_TRACE_LOG` (per-step kernel routing). Paper-setting
quality results are in [Results](#results).

<br>

## Adapting to Your Machine

Everything machine-specific reduces to four decisions
(full guide: [docs/INSTALL.md](docs/INSTALL.md#adapting-to-your-machine)):

| Decision | How |
|:--|:--|
| **Which kernel** | your GPU arch: SM80/SM90 → FA3 (`VLLM_FLASH_ATTN_VERSION=3`), SM100 → FA4 (`=4`). Build-time arch (`TORCH_CUDA_ARCH_LIST`) is auto-detected by the setup script |
| **CUDA toolkit** | point `CUDA_HOME` at any local ≥ 12.0 toolkit before building (≥ 12.8 for Blackwell) |
| **Memory budget** | size `--max-model-len` to your real need; on shared GPUs or big models **pin the KV pool** with `VLLM_KV_CACHE_MEMORY_BYTES` — SFI's runtime state (~2–3 GB at 8 slots) lives *outside* vLLM's `gpu-memory-utilization` accounting |
| **Quality vs speed** | one dial: `k_head` (4096 for evals ↔ 1536–2048 for throughput); keep `compact_blocks_per_slot ≥ k_head/16` and `slots = your batch` |

Failure signatures (GPU stolen by a neighbor, expected harness exit 2 in
free-corpus mode, corpus segment contract, ...) are tabulated in
[docs/INSTALL.md → Troubleshooting](docs/INSTALL.md#troubleshooting).

<br>

## Results

Paper results (algorithm-level, Triton backend; the CUDA edition above is the
same algorithm on faster kernels):

SFI achieves **1.6–14.4× end-to-end decode speedup** while **preserving quality close to dense full attention** across long-context and long-CoT workloads.

<details>
<summary>&ensp;<b>End-to-end throughput</b></summary>

<br>

<div align="center">
<img src="assets/speed.png" width="72%" alt="SFI throughput" />
</div>

<br>

Representative speedups at 128K context: &ensp; **Qwen3-4B** 14.36× &ensp;·&ensp; **Qwen3-30B-A3B** 11.98× &ensp;·&ensp; **Qwen3-235B-A22B** 13.49×

</details>

<details>
<summary>&ensp;<b>Kernel-level speedup</b>&ensp;<sub>KV length 16K · batch=16 · bf16</sub></summary>

<br>

| Retention (%) | 1.6 | 6.3 | 12.5 | 25.0 | 37.5 | 50.0 | 75.0 | 98.4 | 100 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Speedup** | **10.67×** | **9.56×** | **7.15×** | **3.96×** | **2.75×** | **2.10×** | **1.43×** | 1.10× | 1.00× |

</details>

<details>
<summary>&ensp;<b>LongBench-V1 quality</b>&ensp;<sub>17 tasks · 3 model scales</sub></summary>

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

</details>

<details>
<summary>&ensp;<b>Long-CoT reasoning</b>&ensp;<sub>GPQA &amp; MMLU</sub></summary>

<br>

| Model | GPQA Slow | GPQA **SFI** | MMLU Slow | MMLU **SFI** |
|:---|:---:|:---:|:---:|:---:|
| Qwen3-4B-Thinking | **64.14** | 63.70 | **63.00** | **63.00** |
| Qwen3-30B-A3B-Thinking | 69.70 | **71.21** | **70.90** | 70.60 |
| Qwen3-235B-A22B-Thinking | **80.80** | **80.80** | 90.09 | **90.30** |

</details>

<br>

## Roadmap

- [x] SFI framework with vLLM + Triton kernel (`triton-kernel` branch)
- [x] **FlashAttention-3 CUDA kernel backend — SM80 / SM90 (this branch)**
- [x] **FlashAttention-4 CuTe kernel backend — SM100 (this branch)**
- [x] SM100/B200 kernel throughput validation (measured on B200)
- [x] **Multi-GPU tensor-parallel sparse decode (validated end-to-end at 2 × 128k)**
- [ ] SGLang backend support

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
